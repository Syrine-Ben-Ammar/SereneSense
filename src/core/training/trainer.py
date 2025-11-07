#
# Plan:
# 1. Create SereneSenseTrainer class as the main training orchestrator
# 2. Implement TrainingConfig dataclass for comprehensive training configuration
# 3. Add TrainingState class for tracking training progress and metrics
# 4. Implement ModelWrapper for unified model interface across architectures
# 5. Support for distributed training, mixed precision, and gradient accumulation
# 6. Integration with experiment tracking (Weights & Biases, MLflow)
# 7. Advanced features like early stopping, checkpointing, and resuming
# 8. Comprehensive logging and monitoring capabilities
#

"""
SereneSense Trainer: Advanced Training Orchestrator

This module implements the main training infrastructure for SereneSense models,
providing a unified interface for training AudioMAE, AST, and BEATs models
with enterprise-grade features and optimizations.

Key Features:
- Unified training interface for all model architectures
- Automatic mixed precision training
- Distributed training support
- Advanced checkpointing and resuming
- Comprehensive experiment tracking
- Real-time monitoring and visualization
- Memory and performance optimizations
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json

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

from ..models.audioMAE import AudioMAEFineTuner
from ..models.AST import ASTFineTuner
from ..models.BEATs import BEATsFineTuner
from ..utils.config import ConfigManager
from ..utils.metrics import MetricCalculator
from ..utils.device_utils import DeviceManager
from .loss_functions import CombinedLoss
from .optimizers import OptimizerFactory
from .schedulers import SchedulerFactory

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Comprehensive training configuration for SereneSense models"""

    # Model configuration
    model_type: str = "audioMAE"  # "audioMAE", "ast", "beats"
    model_config_path: str = ""  # Path to model configuration
    pretrained_path: Optional[str] = None  # Path to pre-trained weights

    # Training hyperparameters
    epochs: int = 100  # Number of training epochs
    batch_size: int = 32  # Training batch size
    validation_batch_size: Optional[int] = None  # Validation batch size (defaults to batch_size)
    learning_rate: float = 1e-4  # Initial learning rate
    weight_decay: float = 1e-4  # Weight decay for regularization

    # Optimization settings
    optimizer: str = "adamw"  # Optimizer type
    scheduler: str = "cosine_warmup"  # Learning rate scheduler
    warmup_epochs: int = 10  # Warmup epochs
    warmup_start_lr: float = 1e-6  # Starting learning rate for warmup

    # Regularization
    gradient_clip_norm: float = 1.0  # Gradient clipping norm
    label_smoothing: float = 0.1  # Label smoothing factor
    dropout: float = 0.1  # Dropout rate

    # Mixed precision training
    mixed_precision: bool = True  # Enable automatic mixed precision
    grad_scale_init: float = 2**16  # Initial gradient scale

    # Distributed training
    distributed: bool = False  # Enable distributed training
    local_rank: int = 0  # Local rank for distributed training
    world_size: int = 1  # World size for distributed training

    # Data loading
    num_workers: int = 4  # Number of data loader workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    persistent_workers: bool = True  # Keep workers alive between epochs

    # Checkpointing and saving
    save_dir: str = "checkpoints"  # Directory to save checkpoints
    save_every_n_epochs: int = 5  # Save checkpoint every N epochs
    save_best_only: bool = True  # Only save best validation checkpoints
    max_checkpoints: int = 5  # Maximum number of checkpoints to keep

    # Early stopping
    early_stopping: bool = True  # Enable early stopping
    patience: int = 15  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement for early stopping
    monitor_metric: str = "val_accuracy"  # Metric to monitor for early stopping

    # Logging and monitoring
    log_every_n_steps: int = 10  # Log metrics every N steps
    validate_every_n_epochs: int = 1  # Validate every N epochs

    # Experiment tracking
    use_wandb: bool = False  # Enable Weights & Biases logging
    use_mlflow: bool = False  # Enable MLflow logging
    experiment_name: str = "serenesense_training"  # Experiment name
    project_name: str = "serenesense"  # Project name for experiment tracking

    # Advanced features
    gradient_accumulation_steps: int = 1  # Gradient accumulation steps
    find_unused_parameters: bool = False  # For distributed training
    compile_model: bool = False  # Enable model compilation (PyTorch 2.0+)

    # Resuming training
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint to resume from
    reset_optimizer: bool = False  # Reset optimizer when resuming
    reset_scheduler: bool = False  # Reset scheduler when resuming

    # Custom configurations
    custom_configs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    """Training state for tracking progress and metrics"""

    # Training progress
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    epochs_without_improvement: int = 0

    # Timing
    epoch_start_time: float = 0.0
    training_start_time: float = 0.0
    total_training_time: float = 0.0

    # Metrics history
    train_metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    val_metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Learning rate history
    lr_history: List[float] = field(default_factory=list)

    # Loss history
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)

    # Checkpointing
    last_checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None


class ModelWrapper:
    """
    Unified wrapper for different SereneSense model architectures.

    Provides a consistent interface for training different model types
    (AudioMAE, AST, BEATs) with their specific fine-tuning strategies.
    """

    def __init__(self, model_type: str, config_path: str, pretrained_path: Optional[str] = None):
        self.model_type = model_type.lower()
        self.config_path = config_path
        self.pretrained_path = pretrained_path

        # Load model configuration
        config_manager = ConfigManager()
        self.model_config = config_manager.load_config(config_path)

        # Initialize model based on type
        if self.model_type == "audiomae":
            from ..models.audioMAE import AudioMAEConfig

            model_config = AudioMAEConfig(**self.model_config)
            self.model = AudioMAEFineTuner(model_config, pretrained_path=pretrained_path)
        elif self.model_type == "ast":
            from ..models.AST import ASTConfig

            model_config = ASTConfig(**self.model_config)
            self.model = ASTFineTuner(model_config, pretrained_path=pretrained_path)
        elif self.model_type == "beats":
            from ..models.BEATs import BEATsConfig

            model_config = BEATsConfig(**self.model_config)
            self.model = BEATsFineTuner(model_config, pretrained_path=pretrained_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized {model_type} model with {self.count_parameters():,} parameters")

    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_optimizer_parameters(self) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer"""
        if hasattr(self.model, "get_optimizer_parameters"):
            return self.model.get_optimizer_parameters()
        else:
            return [{"params": self.model.parameters()}]

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Perform training step"""
        return self.model.training_step(batch, batch_idx)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Perform validation step"""
        return self.model.validation_step(batch, batch_idx)

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make predictions"""
        return self.model.predict(batch["spectrograms"])


class SereneSenseTrainer:
    """
    Main trainer class for SereneSense models.

    Provides comprehensive training infrastructure with enterprise-grade
    features including distributed training, mixed precision, experiment
    tracking, and advanced optimization strategies.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()

        # Setup device and distributed training
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device()
        self._setup_distributed()

        # Initialize model
        self.model_wrapper = ModelWrapper(
            model_type=config.model_type,
            config_path=config.model_config_path,
            pretrained_path=config.pretrained_path,
        )
        self.model = self.model_wrapper.model

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup distributed model
        if self.config.distributed and dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
            )

        # Model compilation (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.config.mixed_precision else None

        # Initialize experiment tracking
        self._setup_experiment_tracking()

        # Create save directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SereneSense trainer for {config.model_type}")
        logger.info(f"Model parameters: {self.model_wrapper.count_parameters():,}")
        logger.info(f"Trainable parameters: {self.model_wrapper.count_trainable_parameters():,}")

    def _setup_distributed(self):
        """Setup distributed training"""
        if self.config.distributed:
            if not dist.is_available():
                raise RuntimeError("Distributed training not available")

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            self.config.local_rank = dist.get_rank()
            self.config.world_size = dist.get_world_size()
            torch.cuda.set_device(self.config.local_rank)

            logger.info(
                f"Distributed training: rank {self.config.local_rank}/{self.config.world_size}"
            )

    def _setup_experiment_tracking(self):
        """Setup experiment tracking services"""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.__dict__,
            )
            logger.info("Initialized Weights & Biases tracking")

        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run()
            mlflow.log_params(self.config.__dict__)
            logger.info("Initialized MLflow tracking")

    def setup_training(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_function: Optional[nn.Module] = None,
    ):
        """
        Setup training components.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_function: Custom loss function (optional)
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup distributed samplers
        if self.config.distributed:
            if isinstance(train_dataloader.sampler, DistributedSampler):
                self.train_sampler = train_dataloader.sampler
            else:
                logger.warning("Distributed training enabled but no DistributedSampler found")
                self.train_sampler = None
        else:
            self.train_sampler = None

        # Setup loss function
        if loss_function is not None:
            self.loss_function = loss_function
        else:
            self.loss_function = CombinedLoss()

        # Setup optimizer
        optimizer_factory = OptimizerFactory()
        param_groups = self.model_wrapper.get_optimizer_parameters()
        self.optimizer = optimizer_factory.create_optimizer(
            optimizer_type=self.config.optimizer,
            parameters=param_groups,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        scheduler_factory = SchedulerFactory()
        total_steps = len(train_dataloader) * self.config.epochs
        warmup_steps = len(train_dataloader) * self.config.warmup_epochs

        self.scheduler = scheduler_factory.create_scheduler(
            scheduler_type=self.config.scheduler,
            optimizer=self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_start_lr=self.config.warmup_start_lr,
        )

        logger.info("Training setup completed")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        epoch_start_time = time.time()

        # Set epoch for distributed sampler
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.state.epoch)

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Training step
            step_metrics = self._training_step(batch, batch_idx)

            # Accumulate metrics
            for key, value in step_metrics.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key].append(value.item())
                else:
                    epoch_metrics[key].append(value)

            # Log step metrics
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                self._log_step_metrics(step_metrics, batch_idx)

            self.state.global_step += 1

        # Calculate epoch averages
        epoch_avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}

        # Update training time
        epoch_time = time.time() - epoch_start_time
        epoch_avg_metrics["epoch_time"] = epoch_time
        self.state.total_training_time += epoch_time

        return epoch_avg_metrics

    def _training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Perform single training step"""

        # Forward pass with mixed precision
        if self.config.mixed_precision:
            with autocast():
                step_output = self.model_wrapper.training_step(batch, batch_idx)
                loss = step_output["loss"]
        else:
            step_output = self.model_wrapper.training_step(batch, batch_idx)
            loss = step_output["loss"]

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (if accumulation complete)
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.mixed_precision:
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                self.optimizer.step()

            self.optimizer.zero_grad()

            # Scheduler step (if step-based)
            if hasattr(self.scheduler, "step") and not hasattr(self.scheduler, "step_epoch"):
                self.scheduler.step()

        # Scale loss back for logging
        step_output["loss"] = loss * self.config.gradient_accumulation_steps

        return step_output

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Move batch to device
                batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Validation step
                step_metrics = self.model_wrapper.validation_step(batch, batch_idx)

                # Accumulate metrics
                for key, value in step_metrics.items():
                    if isinstance(value, torch.Tensor):
                        epoch_metrics[key].append(value.item())
                    else:
                        epoch_metrics[key].append(value)

        # Calculate epoch averages
        epoch_avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}

        return epoch_avg_metrics

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        self.state.training_start_time = time.time()

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch
            self.state.epoch_start_time = time.time()

            # Training phase
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            train_metrics = self.train_epoch()

            # Validation phase
            if (epoch + 1) % self.config.validate_every_n_epochs == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}

            # Update metrics history
            for key, value in train_metrics.items():
                self.state.train_metrics[key].append(value)

            for key, value in val_metrics.items():
                self.state.val_metrics[key].append(value)

            # Learning rate tracking
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.state.lr_history.append(current_lr)

            # Scheduler step (if epoch-based)
            if hasattr(self.scheduler, "step_epoch"):
                if val_metrics and "val_loss" in val_metrics:
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()
            elif hasattr(self.scheduler, "step") and hasattr(self.scheduler, "last_epoch"):
                self.scheduler.step()

            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics, epoch)

            # Check for improvement
            monitor_value = val_metrics.get(
                self.config.monitor_metric, train_metrics.get("loss", float("inf"))
            )
            if monitor_value > self.state.best_metric + self.config.min_delta:
                self.state.best_metric = monitor_value
                self.state.epochs_without_improvement = 0

                # Save best checkpoint
                if self.config.save_best_only:
                    checkpoint_path = self.save_checkpoint(epoch, is_best=True)
                    self.state.best_checkpoint_path = checkpoint_path
            else:
                self.state.epochs_without_improvement += 1

            # Save regular checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                checkpoint_path = self.save_checkpoint(epoch, is_best=False)
                self.state.last_checkpoint_path = checkpoint_path

            # Early stopping check
            if (
                self.config.early_stopping
                and self.state.epochs_without_improvement >= self.config.patience
            ):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Training completed
        total_time = time.time() - self.state.training_start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Final experiment tracking
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()

    def _log_step_metrics(self, metrics: Dict[str, Any], step: int):
        """Log step-level metrics"""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {f"train_step/{k}": v for k, v in metrics.items()}, step=self.state.global_step
            )

        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                mlflow.log_metric(f"train_step_{key}", value, step=self.state.global_step)

    def _log_epoch_metrics(
        self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int
    ):
        """Log epoch-level metrics"""
        all_metrics = {}

        # Add training metrics
        for key, value in train_metrics.items():
            all_metrics[f"train/{key}"] = value

        # Add validation metrics
        for key, value in val_metrics.items():
            all_metrics[f"val/{key}"] = value

        # Add learning rate
        all_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Add epoch number
        all_metrics["epoch"] = epoch

        # Log to experiment tracking
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(all_metrics, step=epoch)

        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            for key, value in all_metrics.items():
                mlflow.log_metric(key, value, step=epoch)

        # Log to console
        logger.info(
            f"Epoch {epoch + 1} - " + " - ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()])
        )

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """Save model checkpoint"""
        checkpoint_data = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "training_state": self.state.__dict__,
            "config": self.config.__dict__,
        }

        # Determine filename
        if is_best:
            filename = f"best_model_{self.config.model_type}_{epoch:03d}.pth"
        else:
            filename = f"checkpoint_{self.config.model_type}_{epoch:03d}.pth"

        checkpoint_path = Path(self.config.save_dir) / filename
        torch.save(checkpoint_data, checkpoint_path)

        logger.info(f"Saved {'best ' if is_best else ''}checkpoint: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint["state_dict"])

        # Load optimizer state
        if not self.config.reset_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state
        if (
            not self.config.reset_scheduler
            and "scheduler" in checkpoint
            and checkpoint["scheduler"]
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        # Load scaler state
        if self.scaler and "scaler" in checkpoint and checkpoint["scaler"]:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # Load training state
        if "training_state" in checkpoint:
            for key, value in checkpoint["training_state"].items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)

        # Update epoch
        self.state.epoch = checkpoint["epoch"] + 1

        logger.info(f"Resumed training from epoch {self.state.epoch}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        checkpoint_dir = Path(self.config.save_dir)

        # Get all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob(f"checkpoint_{self.config.model_type}_*.pth"))

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old checkpoints (keep only max_checkpoints)
        for checkpoint_file in checkpoint_files[self.config.max_checkpoints :]:
            checkpoint_file.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint_file}")

    def get_model(self) -> nn.Module:
        """Get the underlying model (unwrapped from DDP if necessary)"""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
