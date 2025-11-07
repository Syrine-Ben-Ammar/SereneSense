#
# Plan:
# 1. Create base callback interface for training events
# 2. Implement EarlyStoppingCallback for preventing overfitting
# 3. Add ModelCheckpointCallback for saving best models
# 4. Create MetricsLoggerCallback for comprehensive logging
# 5. Implement LearningRateLoggerCallback for LR monitoring
# 6. Add GradientNormCallback for training diagnostics
# 7. Create AttentionVisualizationCallback for model analysis
# 8. Implement integration callbacks (WandB, MLflow, TensorBoard)
#

"""
Training Callbacks for SereneSense

This module implements comprehensive callback system for monitoring,
controlling, and enhancing the training process of SereneSense models.

Key Features:
- Early stopping and model checkpointing
- Comprehensive metrics logging and visualization
- Integration with experiment tracking platforms
- Training diagnostics and monitoring
- Attention and gradient analysis
- Learning rate and optimization monitoring
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import warnings

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

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Abstract base class for training callbacks.

    Defines the interface that all callbacks must implement to handle
    various training events and provide monitoring capabilities.
    """

    def __init__(self):
        self.trainer = None
        self.model = None

    def set_trainer(self, trainer):
        """Set reference to trainer"""
        self.trainer = trainer
        self.model = trainer.model

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training"""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training"""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch"""
        pass

    def on_batch_begin(self, batch_idx: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch"""
        pass

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict] = None):
        """Called at the end of each batch"""
        pass

    def on_validation_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of validation"""
        pass

    def on_validation_end(self, logs: Optional[Dict] = None):
        """Called at the end of validation"""
        pass


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback to prevent overfitting.

    Monitors a specified metric and stops training when the metric
    stops improving for a specified number of epochs.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs with no improvement to wait
            mode: 'min' or 'max' for optimization direction
            restore_best_weights: Whether to restore best weights on stop
            verbose: Whether to print early stopping messages
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_weights = None

        if mode == "min":
            self.monitor_op = lambda x, y: x < y - self.min_delta
            self.best = float("inf")
        elif mode == "max":
            self.monitor_op = lambda x, y: x > y + self.min_delta
            self.best = float("-inf")
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'.")

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Reset early stopping state"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_weights = None

        if self.mode == "min":
            self.best = float("inf")
        else:
            self.best = float("-inf")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check for early stopping condition"""
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Early stopping monitor '{self.monitor}' not found in logs")
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0

            # Save best weights
            if self.restore_best_weights:
                if hasattr(self.model, "state_dict"):
                    self.best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    logger.info(
                        f"Best {self.monitor}: {self.best:.6f} at epoch {self.best_epoch + 1}"
                    )

                # Restore best weights
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info("Restored best weights")

                # Signal trainer to stop
                if hasattr(self.trainer, "stop_training"):
                    self.trainer.stop_training = True

                return True  # Signal to stop training

        return False


class ModelCheckpointCallback(Callback):
    """
    Model checkpoint callback for saving best models.

    Saves model checkpoints based on monitored metrics and provides
    flexible checkpoint management strategies.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_freq: Union[int, str] = "epoch",
        save_weights_only: bool = False,
        verbose: bool = True,
        max_checkpoints: int = 5,
    ):
        """
        Initialize model checkpoint callback.

        Args:
            filepath: Path template for saving checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for optimization direction
            save_best_only: Whether to save only best models
            save_freq: Frequency of saving ('epoch' or integer for batch frequency)
            save_weights_only: Whether to save only model weights
            verbose: Whether to print checkpoint messages
            max_checkpoints: Maximum number of checkpoints to keep
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.max_checkpoints = max_checkpoints

        self.best = float("inf") if mode == "min" else float("-inf")
        self.checkpoints = []

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def _should_save(self, current: float) -> bool:
        """Check if current metric warrants saving"""
        if self.mode == "min":
            return current < self.best
        else:
            return current > self.best

    def _save_checkpoint(self, epoch: int, logs: Dict, is_best: bool = False):
        """Save model checkpoint"""
        # Format filepath
        if is_best:
            filepath = self.filepath.replace(".pth", "_best.pth")
        else:
            filepath = self.filepath.replace(".pth", f"_epoch_{epoch:03d}.pth")

        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": logs,
            "monitor": self.monitor,
            "monitor_value": logs.get(self.monitor, None),
        }

        # Add optimizer and scheduler state if available
        if hasattr(self.trainer, "optimizer"):
            checkpoint_data["optimizer_state_dict"] = self.trainer.optimizer.state_dict()

        if hasattr(self.trainer, "scheduler") and self.trainer.scheduler:
            checkpoint_data["scheduler_state_dict"] = self.trainer.scheduler.state_dict()

        # Save checkpoint
        torch.save(checkpoint_data, filepath)

        if self.verbose:
            logger.info(f"Saved {'best ' if is_best else ''}checkpoint: {filepath}")

        # Track checkpoints for cleanup
        if not is_best:
            self.checkpoints.append((filepath, epoch))
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by epoch and remove oldest
            self.checkpoints.sort(key=lambda x: x[1])

            while len(self.checkpoints) > self.max_checkpoints:
                filepath, epoch = self.checkpoints.pop(0)
                if Path(filepath).exists():
                    Path(filepath).unlink()
                    if self.verbose:
                        logger.debug(f"Removed old checkpoint: {filepath}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Handle epoch end checkpoint saving"""
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Checkpoint monitor '{self.monitor}' not found in logs")
            current = float("inf") if self.mode == "min" else float("-inf")

        if self.save_best_only:
            if self._should_save(current):
                self.best = current
                self._save_checkpoint(epoch, logs, is_best=True)
        else:
            # Save regular checkpoint
            self._save_checkpoint(epoch, logs, is_best=False)

            # Also save best if better
            if self._should_save(current):
                self.best = current
                self._save_checkpoint(epoch, logs, is_best=True)


class MetricsLoggerCallback(Callback):
    """
    Comprehensive metrics logging callback.

    Logs training and validation metrics to various outputs including
    console, files, and structured logs for analysis.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_freq: int = 10,
        save_metrics: bool = True,
        plot_metrics: bool = False,
    ):
        """
        Initialize metrics logger callback.

        Args:
            log_dir: Directory for saving log files
            log_freq: Frequency of logging (in batches)
            save_metrics: Whether to save metrics to files
            plot_metrics: Whether to generate metric plots
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.save_metrics = save_metrics
        self.plot_metrics = plot_metrics

        self.metrics_history = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        self.batch_count = 0

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize logging"""
        self.metrics_history.clear()
        self.epoch_metrics.clear()
        self.batch_count = 0

        logger.info(f"Metrics logging initialized. Log directory: {self.log_dir}")

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict] = None):
        """Log batch metrics"""
        if logs is None:
            return

        self.batch_count += 1

        # Log to console at specified frequency
        if self.batch_count % self.log_freq == 0:
            log_str = f"Batch {self.batch_count}"
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    log_str += f" - {key}: {value:.4f}"
            logger.info(log_str)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log epoch metrics"""
        if logs is None:
            return

        # Store metrics history
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
                self.epoch_metrics[key].append((epoch, value))

        # Log to console
        log_str = f"Epoch {epoch + 1}"
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                log_str += f" - {key}: {value:.4f}"
        logger.info(log_str)

        # Save metrics to file
        if self.save_metrics:
            self._save_metrics_to_file(epoch)

    def _save_metrics_to_file(self, epoch: int):
        """Save metrics to JSON file"""
        metrics_file = self.log_dir / "metrics.json"

        metrics_data = {
            "epoch": epoch,
            "metrics_history": dict(self.metrics_history),
            "latest_metrics": dict(self.epoch_metrics),
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

    def plot_training_curves(self):
        """Plot training curves"""
        if not self.plot_metrics:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return

        # Plot loss curves
        if "train_loss" in self.metrics_history and "val_loss" in self.metrics_history:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(self.metrics_history["train_loss"], label="Training Loss")
            plt.plot(self.metrics_history["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)

            # Plot accuracy curves
            if "train_accuracy" in self.metrics_history and "val_accuracy" in self.metrics_history:
                plt.subplot(1, 2, 2)
                plt.plot(self.metrics_history["train_accuracy"], label="Training Accuracy")
                plt.plot(self.metrics_history["val_accuracy"], label="Validation Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title("Training and Validation Accuracy")
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(self.log_dir / "training_curves.png", dpi=300, bbox_inches="tight")
            plt.close()


class LearningRateLoggerCallback(Callback):
    """
    Learning rate monitoring callback.

    Tracks and logs learning rate changes throughout training,
    providing insights into optimization dynamics.
    """

    def __init__(self, log_freq: int = 1):
        """
        Initialize learning rate logger.

        Args:
            log_freq: Frequency of logging learning rates (in epochs)
        """
        super().__init__()
        self.log_freq = log_freq
        self.lr_history = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log learning rates"""
        if not hasattr(self.trainer, "optimizer"):
            return

        # Get current learning rates
        current_lrs = [group["lr"] for group in self.trainer.optimizer.param_groups]
        self.lr_history.append(current_lrs)

        # Log at specified frequency
        if (epoch + 1) % self.log_freq == 0:
            if len(current_lrs) == 1:
                logger.info(f"Epoch {epoch + 1} - Learning Rate: {current_lrs[0]:.6f}")
            else:
                lr_str = " - ".join([f"LR_{i}: {lr:.6f}" for i, lr in enumerate(current_lrs)])
                logger.info(f"Epoch {epoch + 1} - {lr_str}")

    def plot_lr_schedule(self):
        """Plot learning rate schedule"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return

        if not self.lr_history:
            return

        plt.figure(figsize=(10, 6))

        # Plot each parameter group
        lr_array = np.array(self.lr_history)
        for i in range(lr_array.shape[1]):
            plt.plot(lr_array[:, i], label=f"Param Group {i}")

        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        plt.yscale("log")
        plt.show()


class GradientNormCallback(Callback):
    """
    Gradient norm monitoring callback.

    Tracks gradient norms to diagnose training issues like
    vanishing or exploding gradients.
    """

    def __init__(self, log_freq: int = 10, norm_type: float = 2.0):
        """
        Initialize gradient norm callback.

        Args:
            log_freq: Frequency of logging gradient norms
            norm_type: Type of norm to compute
        """
        super().__init__()
        self.log_freq = log_freq
        self.norm_type = norm_type
        self.grad_norms = []
        self.batch_count = 0

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict] = None):
        """Monitor gradient norms"""
        if not hasattr(self.model, "parameters"):
            return

        self.batch_count += 1

        # Compute gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type

        total_norm = total_norm ** (1.0 / self.norm_type)
        self.grad_norms.append(total_norm)

        # Log at specified frequency
        if self.batch_count % self.log_freq == 0:
            logger.debug(f"Batch {self.batch_count} - Gradient Norm: {total_norm:.6f}")

            # Check for potential issues
            if total_norm > 100.0:
                logger.warning(f"Large gradient norm detected: {total_norm:.6f}")
            elif total_norm < 1e-6:
                logger.warning(f"Very small gradient norm detected: {total_norm:.6f}")

    def plot_gradient_norms(self):
        """Plot gradient norm history"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return

        if not self.grad_norms:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.grad_norms)
        plt.xlabel("Batch")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm Evolution")
        plt.grid(True)
        plt.yscale("log")
        plt.show()


class ActivationStatisticsCallback(Callback):
    """
    Activation statistics monitoring callback.

    Tracks activation statistics to monitor model health
    and detect issues like dead neurons or saturation.
    """

    def __init__(self, modules_to_monitor: List[str] = None, log_freq: int = 100):
        """
        Initialize activation statistics callback.

        Args:
            modules_to_monitor: List of module names to monitor
            log_freq: Frequency of logging statistics
        """
        super().__init__()
        self.modules_to_monitor = modules_to_monitor or ["attention", "mlp", "head"]
        self.log_freq = log_freq
        self.activation_stats = defaultdict(list)
        self.hooks = []
        self.batch_count = 0

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Setup activation hooks"""

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Compute statistics
                    mean = output.mean().item()
                    std = output.std().item()
                    dead_ratio = (output == 0).float().mean().item()

                    self.activation_stats[name].append(
                        {"mean": mean, "std": std, "dead_ratio": dead_ratio}
                    )

            return hook

        # Register hooks
        for name, module in self.model.named_modules():
            if any(monitor in name.lower() for monitor in self.modules_to_monitor):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict] = None):
        """Log activation statistics"""
        self.batch_count += 1

        if self.batch_count % self.log_freq == 0:
            for module_name, stats_list in self.activation_stats.items():
                if stats_list:
                    latest_stats = stats_list[-1]
                    logger.debug(
                        f"Module {module_name} - "
                        f"Mean: {latest_stats['mean']:.4f}, "
                        f"Std: {latest_stats['std']:.4f}, "
                        f"Dead: {latest_stats['dead_ratio']:.2%}"
                    )


class AttentionVisualizationCallback(Callback):
    """
    Attention visualization callback for transformer models.

    Captures and visualizes attention patterns to understand
    what the model is focusing on during audio processing.
    """

    def __init__(self, save_dir: str = "attention_maps", save_freq: int = 10, num_samples: int = 4):
        """
        Initialize attention visualization callback.

        Args:
            save_dir: Directory to save attention visualizations
            save_freq: Frequency of saving visualizations (in epochs)
            num_samples: Number of samples to visualize per save
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.num_samples = num_samples

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save attention visualizations"""
        if (epoch + 1) % self.save_freq != 0:
            return

        if not hasattr(self.trainer, "val_dataloader"):
            return

        try:
            self._visualize_attention(epoch)
        except Exception as e:
            logger.warning(f"Failed to generate attention visualizations: {e}")

    def _visualize_attention(self, epoch: int):
        """Generate attention visualizations"""
        self.model.eval()

        # Get a batch from validation set
        val_batch = next(iter(self.trainer.val_dataloader))

        # Move to device
        if hasattr(self.trainer, "device"):
            val_batch = {
                k: v.to(self.trainer.device) if isinstance(v, torch.Tensor) else v
                for k, v in val_batch.items()
            }

        # Extract attention maps
        with torch.no_grad():
            if hasattr(self.model, "get_attention_maps"):
                attention_maps = self.model.get_attention_maps(
                    val_batch["spectrograms"][: self.num_samples]
                )
                self._plot_attention_maps(attention_maps, epoch)

    def _plot_attention_maps(self, attention_maps: List[torch.Tensor], epoch: int):
        """Plot and save attention maps"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for attention visualization")
            return

        for layer_idx, attention in enumerate(attention_maps):
            # attention shape: [batch, heads, seq_len, seq_len]
            if len(attention.shape) != 4:
                continue

            batch_size, num_heads, seq_len, _ = attention.shape

            # Plot attention for first sample
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f"Attention Maps - Layer {layer_idx} - Epoch {epoch + 1}")

            for head_idx in range(min(8, num_heads)):
                row = head_idx // 4
                col = head_idx % 4

                ax = axes[row, col]
                attn_map = attention[0, head_idx].cpu().numpy()

                im = ax.imshow(attn_map, cmap="Blues", aspect="auto")
                ax.set_title(f"Head {head_idx}")
                ax.set_xlabel("Key Position")
                ax.set_ylabel("Query Position")

                plt.colorbar(im, ax=ax)

            plt.tight_layout()
            plt.savefig(
                self.save_dir / f"attention_layer_{layer_idx}_epoch_{epoch:03d}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()


class WandBCallback(Callback):
    """
    Weights & Biases integration callback.

    Logs metrics, gradients, and model artifacts to Weights & Biases
    for comprehensive experiment tracking and visualization.
    """

    def __init__(
        self,
        project: str = "serenesense",
        log_freq: int = 10,
        log_gradients: bool = False,
        log_model: bool = False,
    ):
        """
        Initialize W&B callback.

        Args:
            project: W&B project name
            log_freq: Frequency of logging metrics
            log_gradients: Whether to log gradient histograms
            log_model: Whether to log model artifacts
        """
        super().__init__()

        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBCallback")

        self.project = project
        self.log_freq = log_freq
        self.log_gradients = log_gradients
        self.log_model = log_model
        self.batch_count = 0

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize W&B logging"""
        if not wandb.run:
            wandb.init(project=self.project)

        # Log model architecture
        if self.log_model:
            wandb.watch(self.model, log="all")

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict] = None):
        """Log batch metrics to W&B"""
        if logs is None:
            return

        self.batch_count += 1

        if self.batch_count % self.log_freq == 0:
            wandb.log({f"batch/{k}": v for k, v in logs.items()}, step=self.batch_count)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log epoch metrics to W&B"""
        if logs is None:
            return

        wandb.log(logs, step=epoch)

        # Log gradient histograms
        if self.log_gradients:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu())}, step=epoch)


class MLflowCallback(Callback):
    """
    MLflow integration callback.

    Logs metrics, parameters, and model artifacts to MLflow
    for experiment tracking and model management.
    """

    def __init__(
        self, experiment_name: str = "serenesense", log_freq: int = 10, log_model: bool = False
    ):
        """
        Initialize MLflow callback.

        Args:
            experiment_name: MLflow experiment name
            log_freq: Frequency of logging metrics
            log_model: Whether to log model artifacts
        """
        super().__init__()

        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow is required for MLflowCallback")

        self.experiment_name = experiment_name
        self.log_freq = log_freq
        self.log_model = log_model
        self.batch_count = 0

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize MLflow tracking"""
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()

        # Log model parameters
        if hasattr(self.trainer, "config"):
            mlflow.log_params(self.trainer.config.__dict__)

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict] = None):
        """Log batch metrics to MLflow"""
        if logs is None:
            return

        self.batch_count += 1

        if self.batch_count % self.log_freq == 0:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"batch_{key}", value, step=self.batch_count)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log epoch metrics to MLflow"""
        if logs is None:
            return

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=epoch)

    def on_train_end(self, logs: Optional[Dict] = None):
        """End MLflow run"""
        # Log final model
        if self.log_model:
            mlflow.pytorch.log_model(self.model, "model")

        mlflow.end_run()


class TensorBoardCallback(Callback):
    """
    TensorBoard integration callback.

    Logs metrics, model graph, and other visualizations to TensorBoard
    for monitoring and analysis during training.
    """

    def __init__(self, log_dir: str = "runs", log_freq: int = 10, log_graph: bool = True):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs
            log_freq: Frequency of logging metrics
            log_graph: Whether to log model graph
        """
        super().__init__()

        if not TENSORBOARD_AVAILABLE:
            raise ImportError("tensorboard is required for TensorBoardCallback")

        self.log_dir = log_dir
        self.log_freq = log_freq
        self.log_graph = log_graph
        self.batch_count = 0
        self.writer = None

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize TensorBoard writer"""
        self.writer = SummaryWriter(self.log_dir)

        # Log model graph
        if self.log_graph and hasattr(self.trainer, "val_dataloader"):
            try:
                sample_batch = next(iter(self.trainer.val_dataloader))
                if hasattr(self.trainer, "device"):
                    sample_input = sample_batch["spectrograms"][:1].to(self.trainer.device)
                else:
                    sample_input = sample_batch["spectrograms"][:1]
                self.writer.add_graph(self.model, sample_input)
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict] = None):
        """Log batch metrics to TensorBoard"""
        if logs is None or self.writer is None:
            return

        self.batch_count += 1

        if self.batch_count % self.log_freq == 0:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"batch/{key}", value, self.batch_count)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log epoch metrics to TensorBoard"""
        if logs is None or self.writer is None:
            return

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


class CallbackList:
    """
    Container for managing multiple callbacks.

    Provides a unified interface for calling multiple callbacks
    and handling their execution in the correct order.
    """

    def __init__(self, callbacks: List[Callback] = None):
        """
        Initialize callback list.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []

    def set_trainer(self, trainer):
        """Set trainer for all callbacks"""
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def add_callback(self, callback: Callback):
        """Add a callback to the list"""
        callback.set_trainer(getattr(self, "trainer", None))
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callback):
        """Remove a callback from the list"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Call on_train_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Call on_train_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch_idx: int, logs: Optional[Dict] = None):
        """Call on_batch_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, logs)

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict] = None):
        """Call on_batch_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, logs)

    def on_validation_begin(self, logs: Optional[Dict] = None):
        """Call on_validation_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_validation_begin(logs)

    def on_validation_end(self, logs: Optional[Dict] = None):
        """Call on_validation_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_validation_end(logs)
