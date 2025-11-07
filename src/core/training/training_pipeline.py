"""
SereneSense Training Pipeline
Enterprise-grade training system for military vehicle sound detection.

Features:
- Distributed training support
- Experiment tracking with MLflow and W&B
- Automatic model checkpointing
- Advanced learning rate scheduling
- Mixed precision training
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
import json
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ML tracking
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from core.models.audioMAE.model import AudioMAE, AudioMAEConfig
from core.data.loaders.mad_loader import MADDataModule
from core.core.audio_processor import AudioConfig
from core.utils.metrics import calculate_metrics
from core.utils.visualization import plot_training_curves, plot_confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model settings
    model_name: str = "audioMAE"
    model_config: Dict = field(default_factory=dict)

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10

    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4

    # Checkpointing
    save_every_n_epochs: int = 5
    save_best_only: bool = True

    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

    # Paths
    output_dir: str = "models/checkpoints"
    log_dir: str = "logs/training"

    # Experiment tracking
    experiment_name: str = "serenesense-audioMAE"
    use_wandb: bool = True
    use_mlflow: bool = True

    # Validation
    val_every_n_epochs: int = 1
    test_after_training: bool = True

    # Reproducibility
    seed: int = 42


class SereneSenseTrainer:
    """
    Enterprise-grade trainer for military vehicle sound detection models.

    Features:
    - Multi-GPU distributed training
    - Automatic mixed precision
    - Advanced learning rate scheduling
    - Comprehensive experiment tracking
    - Model checkpointing and recovery
    - Detailed metrics and visualization
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        data_module: MADDataModule,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.model = model
        self.data_module = data_module
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training state
        self.current_epoch = 0
        self.best_val_score = float("-inf")
        self.early_stopping_counter = 0
        self.training_history = defaultdict(list)

        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.log_dir = Path(config.log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_distributed()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_scaler()
        self._setup_experiment_tracking()

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _setup_distributed(self):
        """Setup distributed training"""
        if self.config.distributed:
            dist.init_process_group(backend="nccl")
            self.device = torch.device(f"cuda:{self.config.local_rank}")
            torch.cuda.set_device(self.device)

    def _setup_model(self):
        """Setup model for training"""
        self.model = self.model.to(self.device)

        if self.config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
            )

    def _setup_optimizer(self):
        """Setup optimizer"""
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs, eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=10, verbose=True
            )
        else:
            self.scheduler = None

    def _setup_loss(self):
        """Setup loss function"""
        # Get class weights for balanced training
        class_weights = self.data_module.get_class_weights()
        class_weights = class_weights.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=self.config.label_smoothing
        )

    def _setup_scaler(self):
        """Setup gradient scaler for mixed precision"""
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def _setup_experiment_tracking(self):
        """Setup experiment tracking with W&B and MLflow"""
        if self.config.use_wandb and WANDB_AVAILABLE and self.config.local_rank == 0:
            wandb.init(
                project="serenesense", name=self.config.experiment_name, config=self.config.__dict__
            )

        if self.config.use_mlflow and MLFLOW_AVAILABLE and self.config.local_rank == 0:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run()
            mlflow.log_params(self.config.__dict__)

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training results and metrics
        """
        logger.info("Starting training...")
        start_time = time.time()

        # Setup data loaders
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch

                # Training phase
                train_metrics = self._train_epoch(train_loader)

                # Validation phase
                if epoch % self.config.val_every_n_epochs == 0:
                    val_metrics = self._validate_epoch(val_loader)
                else:
                    val_metrics = {}

                # Update learning rate
                self._update_scheduler(val_metrics.get("accuracy", 0))

                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)

                # Save checkpoint
                if self._should_save_checkpoint(epoch, val_metrics):
                    self._save_checkpoint(epoch, val_metrics)

                # Early stopping check
                if self._should_early_stop(val_metrics):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self._cleanup_training()

        # Final evaluation
        results = self._finalize_training(start_time)

        return results

    def _train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(data_loader):
            # Move data to device
            spectrograms = batch["spectrograms"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(spectrograms, labels=labels, mode="classification")
                    loss = outputs["loss"]

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(spectrograms, labels=labels, mode="classification")
                loss = outputs["loss"]

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                # Optimizer step
                self.optimizer.step()

            # Update metrics
            batch_size = spectrograms.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs["predictions"]
            correct_predictions += (predictions == labels).sum().item()

            # Log batch progress
            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(data_loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )

        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        epoch_time = time.time() - epoch_start_time

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "epoch_time": epoch_time,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    def _validate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                spectrograms = batch["spectrograms"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(spectrograms, labels=labels, mode="classification")
                else:
                    outputs = self.model(spectrograms, labels=labels, mode="classification")

                # Update metrics
                batch_size = spectrograms.size(0)
                total_loss += outputs["loss"].item() * batch_size
                total_samples += batch_size

                # Collect predictions
                all_predictions.extend(outputs["predictions"].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / total_samples
        metrics = calculate_metrics(all_labels, all_predictions)
        metrics["loss"] = avg_loss

        return metrics

    def _update_scheduler(self, val_score: float):
        """Update learning rate scheduler"""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_score)
        else:
            self.scheduler.step()

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to experiment tracking systems"""
        # Store in history
        for key, value in train_metrics.items():
            self.training_history[f"train_{key}"].append(value)

        for key, value in val_metrics.items():
            self.training_history[f"val_{key}"].append(value)

        # Console logging
        train_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])

        logger.info(f"Epoch {epoch} - Train: {train_str}")
        if val_metrics:
            logger.info(f"Epoch {epoch} - Val: {val_str}")

        # W&B logging
        if self.config.use_wandb and WANDB_AVAILABLE and self.config.local_rank == 0:
            log_dict = {}
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = value
            for key, value in val_metrics.items():
                log_dict[f"val/{key}"] = value
            log_dict["epoch"] = epoch

            wandb.log(log_dict)

        # MLflow logging
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self.config.local_rank == 0:
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value, step=epoch)
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=epoch)

    def _should_save_checkpoint(self, epoch: int, val_metrics: Dict) -> bool:
        """Determine if checkpoint should be saved"""
        if epoch % self.config.save_every_n_epochs == 0:
            return True

        if self.config.save_best_only and val_metrics:
            val_score = val_metrics.get("accuracy", 0)
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                return True

        return False

    def _save_checkpoint(self, epoch: int, val_metrics: Dict):
        """Save model checkpoint"""
        if self.config.local_rank != 0:
            return  # Only save on main process

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.module.state_dict()
                if self.config.distributed
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_score": self.best_val_score,
            "config": self.config,
            "training_history": dict(self.training_history),
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"{self.config.experiment_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if val_metrics and val_metrics.get("accuracy", 0) >= self.best_val_score:
            best_path = self.output_dir / f"{self.config.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint at epoch {epoch}")

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _should_early_stop(self, val_metrics: Dict) -> bool:
        """Check if training should stop early"""
        if not self.config.early_stopping or not val_metrics:
            return False

        val_score = val_metrics.get("accuracy", 0)

        if val_score > self.best_val_score + self.config.min_delta:
            self.best_val_score = val_score
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.config.patience

    def _cleanup_training(self):
        """Clean up after training"""
        if self.config.use_wandb and WANDB_AVAILABLE and self.config.local_rank == 0:
            wandb.finish()

        if self.config.use_mlflow and MLFLOW_AVAILABLE and self.config.local_rank == 0:
            mlflow.end_run()

        if self.config.distributed:
            dist.destroy_process_group()

    def _finalize_training(self, start_time: float) -> Dict[str, Any]:
        """Finalize training and return results"""
        total_time = time.time() - start_time

        results = {
            "best_val_score": self.best_val_score,
            "total_epochs": self.current_epoch + 1,
            "total_time": total_time,
            "training_history": dict(self.training_history),
        }

        # Test evaluation
        if self.config.test_after_training:
            test_loader = self.data_module.test_dataloader()
            test_metrics = self._validate_epoch(test_loader)
            results["test_metrics"] = test_metrics

            logger.info("Test Results:")
            for key, value in test_metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        # Generate training report
        if self.config.local_rank == 0:
            self._generate_training_report(results)

        logger.info(f"Training completed in {total_time:.2f} seconds")

        return results

    def _generate_training_report(self, results: Dict[str, Any]):
        """Generate comprehensive training report"""
        report_dir = self.log_dir / f"{self.config.experiment_name}_report"
        report_dir.mkdir(exist_ok=True)

        # Save training curves
        plot_training_curves(self.training_history, save_path=report_dir / "training_curves.png")

        # Save configuration
        with open(report_dir / "config.yaml", "w") as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False)

        # Save results
        with open(report_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Training report saved to: {report_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if self.config.distributed:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_score = checkpoint["best_val_score"]
        self.training_history = defaultdict(list, checkpoint.get("training_history", {}))

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint


def create_trainer(
    config_path: str, model_config_path: Optional[str] = None, checkpoint_path: Optional[str] = None
) -> SereneSenseTrainer:
    """
    Factory function to create trainer from configuration files.

    Args:
        config_path: Path to training configuration
        model_config_path: Path to model configuration
        checkpoint_path: Path to checkpoint for resuming training

    Returns:
        Configured trainer instance
    """
    # Load configurations
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    training_config = TrainingConfig(**config_dict.get("training", {}))

    if model_config_path:
        with open(model_config_path, "r") as f:
            model_config_dict = yaml.safe_load(f)
        audioMAE_config = AudioMAEConfig(**model_config_dict.get("model", {}))
    else:
        audioMAE_config = AudioMAEConfig(**config_dict.get("model", {}))

    audio_config = AudioConfig(**config_dict.get("audio", {}))

    # Create model
    model = AudioMAE(audioMAE_config)

    # Create data module
    from core.data.loaders.mad_loader import MADConfig

    mad_config = MADConfig(**config_dict.get("data", {}))
    data_module = MADDataModule(mad_config, audio_config, training_config.batch_size)

    # Create trainer
    trainer = SereneSenseTrainer(training_config, model, data_module)

    # Load checkpoint if provided
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)

    return trainer
