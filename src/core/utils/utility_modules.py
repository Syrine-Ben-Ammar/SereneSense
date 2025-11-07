"""
SereneSense Utility Modules
Essential utilities for logging, metrics, visualization, and device management.
"""

# =============================================================================
# serenesense/utils/logging.py - Logging Configuration
# =============================================================================
import logging
import sys
from pathlib import Path
from typing import Optional, Union
from rich.logging import RichHandler
from rich.console import Console
import json
import time


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    use_rich: bool = True,
    json_logs: bool = False,
) -> logging.Logger:
    """
    Setup comprehensive logging for SereneSense.

    Args:
        level: Logging level
        log_file: Optional file to write logs to
        format_string: Custom format string
        use_rich: Use Rich for colorized console output
        json_logs: Output logs in JSON format

    Returns:
        Configured logger
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    root_logger.setLevel(level)

    # Format string
    if format_string is None:
        if json_logs:
            format_string = None  # Will use custom JSON formatter
        else:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    if use_rich and not json_logs:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(level)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)

        if json_logs:
            console_handler.setFormatter(JSONFormatter())
        else:
            formatter = logging.Formatter(format_string)
            console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)

        if json_logs:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return root_logger


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class PerformanceLogger:
    """Context manager for performance logging"""

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.name} in {elapsed:.3f}s")
        else:
            self.logger.error(f"Failed {self.name} after {elapsed:.3f}s: {exc_val}")


# =============================================================================
# serenesense/utils/metrics.py - Evaluation Metrics
# =============================================================================
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """Container for classification metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]

    # Per-class metrics
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray

    # Additional metrics
    roc_auc: Optional[float] = None
    average_precision: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "per_class_precision": self.per_class_precision.tolist(),
            "per_class_recall": self.per_class_recall.tolist(),
            "per_class_f1": self.per_class_f1.tolist(),
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "average_precision": (
                float(self.average_precision) if self.average_precision is not None else None
            ),
        }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = "weighted",
) -> ClassificationMetrics:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        class_names: Class names for reporting
        average: Averaging strategy for multi-class metrics

    Returns:
        ClassificationMetrics object
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # ROC AUC and Average Precision (if probabilities provided)
    roc_auc = None
    avg_precision = None

    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                avg_precision = average_precision_score(y_true, y_proba[:, 1])
            else:
                # Multi-class classification
                roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
                avg_precision = average_precision_score(y_true, y_proba, average=average)
        except Exception as e:
            logging.warning(f"Could not calculate ROC AUC / Average Precision: {e}")

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=cm,
        classification_report=report,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        roc_auc=roc_auc,
        average_precision=avg_precision,
    )


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 5) -> float:
    """Calculate top-k accuracy"""
    top_k_predictions = np.argsort(y_proba, axis=1)[:, -k:]
    correct = 0

    for i, true_label in enumerate(y_true):
        if true_label in top_k_predictions[i]:
            correct += 1

    return correct / len(y_true)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate balanced accuracy (macro-averaged recall)"""
    return recall_score(y_true, y_pred, average="macro")


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Cohen's Kappa coefficient"""
    from sklearn.metrics import cohen_kappa_score

    return cohen_kappa_score(y_true, y_pred)


class MetricsTracker:
    """Track metrics during training"""

    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "epoch_time": [],
        }

    def update(self, metrics: Dict[str, float], split: str = "train"):
        """Update metrics history"""
        for key, value in metrics.items():
            history_key = f"{split}_{key}" if split else key
            if history_key not in self.history:
                self.history[history_key] = []
            self.history[history_key].append(value)

    def get_best_metric(self, metric_name: str, mode: str = "max") -> Tuple[int, float]:
        """Get best metric value and epoch"""
        values = self.history.get(metric_name, [])
        if not values:
            return 0, 0.0

        if mode == "max":
            best_epoch = np.argmax(values)
            best_value = values[best_epoch]
        else:
            best_epoch = np.argmin(values)
            best_value = values[best_epoch]

        return best_epoch, best_value

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        summary = {}

        # Best validation metrics
        if "val_accuracy" in self.history:
            best_epoch, best_acc = self.get_best_metric("val_accuracy", "max")
            summary["best_val_accuracy"] = best_acc
            summary["best_val_epoch"] = best_epoch

        if "val_loss" in self.history:
            best_epoch, best_loss = self.get_best_metric("val_loss", "min")
            summary["best_val_loss"] = best_loss
            summary["best_val_loss_epoch"] = best_epoch

        # Final metrics
        for key, values in self.history.items():
            if values:
                summary[f"final_{key}"] = values[-1]

        return summary


# =============================================================================
# serenesense/utils/visualization.py - Visualization Tools
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix with annotations.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        history: Training history dictionary
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    plt.style.use("seaborn-v0_8")

    # Determine subplots needed
    metrics = []
    if "train_loss" in history or "val_loss" in history:
        metrics.append("loss")
    if "train_accuracy" in history or "val_accuracy" in history:
        metrics.append("accuracy")
    if "learning_rate" in history:
        metrics.append("learning_rate")

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0], figsize[1]))

    if n_plots == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric == "loss":
            if "train_loss" in history:
                ax.plot(history["train_loss"], label="Train Loss", color=colors[0], linewidth=2)
            if "val_loss" in history:
                ax.plot(history["val_loss"], label="Val Loss", color=colors[1], linewidth=2)
            ax.set_ylabel("Loss")
            ax.set_title("Loss Curves")

        elif metric == "accuracy":
            if "train_accuracy" in history:
                ax.plot(
                    history["train_accuracy"], label="Train Accuracy", color=colors[0], linewidth=2
                )
            if "val_accuracy" in history:
                ax.plot(history["val_accuracy"], label="Val Accuracy", color=colors[1], linewidth=2)
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy Curves")

        elif metric == "learning_rate":
            ax.plot(history["learning_rate"], label="Learning Rate", color=colors[2], linewidth=2)
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.set_yscale("log")

        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot class distribution bar chart"""
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    bars = ax.bar(classes, counts, color="steelblue", alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Classes", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_audio_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "Audio Waveform",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot audio waveform"""
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)

    time_axis = np.linspace(0, len(waveform) / sample_rate, len(waveform))

    ax.plot(time_axis, waveform, color="darkblue", linewidth=0.5)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_spectrogram(
    spectrogram: np.ndarray,
    sample_rate: int,
    hop_length: int,
    title: str = "Spectrogram",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot mel-spectrogram"""
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)

    # Time and frequency axes
    time_frames = spectrogram.shape[1]
    freq_bins = spectrogram.shape[0]

    times = np.linspace(0, time_frames * hop_length / sample_rate, time_frames)
    freqs = np.linspace(0, sample_rate / 2, freq_bins)

    # Plot spectrogram
    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="viridis",
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Frequency (Hz)", fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_interactive_dashboard(
    metrics: Dict[str, Any], save_path: Optional[str] = None
) -> go.Figure:
    """Create interactive Plotly dashboard"""

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Training Curves",
            "Confusion Matrix",
            "Class Distribution",
            "Performance Metrics",
        ],
        specs=[
            [{"secondary_y": False}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
    )

    # Training curves
    if "history" in metrics:
        history = metrics["history"]
        epochs = list(range(len(history.get("train_loss", []))))

        if "train_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history["train_loss"], name="Train Loss", line=dict(color="blue")
                ),
                row=1,
                col=1,
            )
        if "val_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history["val_loss"], name="Val Loss", line=dict(color="red")
                ),
                row=1,
                col=1,
            )

    # Confusion matrix
    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        fig.add_trace(go.Heatmap(z=cm, colorscale="Blues", showscale=False), row=1, col=2)

    # Class distribution
    if "class_distribution" in metrics:
        classes = list(metrics["class_distribution"].keys())
        counts = list(metrics["class_distribution"].values())

        fig.add_trace(go.Bar(x=classes, y=counts, marker_color="steelblue"), row=2, col=1)

    # Performance indicator
    if "accuracy" in metrics:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics["accuracy"] * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Accuracy (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                        {"range": [80, 100], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=2,
            col=2,
        )

    fig.update_layout(title="SereneSense Model Performance Dashboard", showlegend=False, height=800)

    if save_path:
        fig.write_html(save_path)

    return fig


# =============================================================================
# serenesense/utils/device_utils.py - Device Management Utilities
# =============================================================================
import torch
import platform
import subprocess
import psutil
from typing import Dict, Optional, Any


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information"""
    info = {
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "pytorch_version": torch.__version__,
    }

    # CUDA information
    if torch.cuda.is_available():
        info.update(
            {
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_devices": [],
            }
        )

        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append(
                {
                    "name": device_props.name,
                    "memory_total": device_props.total_memory,
                    "memory_available": torch.cuda.memory_reserved(i),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                }
            )
    else:
        info["cuda_available"] = False

    return info


def setup_device(local_rank: Optional[int] = None) -> torch.device:
    """Setup device for training/inference"""
    if torch.cuda.is_available():
        if local_rank is not None:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda")

        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    else:
        device = torch.device("cpu")

        # Optimize CPU settings
        torch.set_num_threads(psutil.cpu_count())

    return device


def detect_platform() -> str:
    """Auto-detect deployment platform"""
    try:
        # Check for Jetson
        if Path("/etc/nv_tegra_release").exists():
            return "jetson"

        # Check for Raspberry Pi
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
            if "raspberry pi" in cpuinfo or "bcm" in cpuinfo:
                return "raspberry_pi"

        # Check for WSL
        if "microsoft" in platform.uname().release.lower():
            return "wsl"

        # Check for cloud instances
        system = platform.system().lower()
        if system == "linux":
            return "linux"
        elif system == "darwin":
            return "macos"
        elif system == "windows":
            return "windows"

    except Exception:
        pass

    return "unknown"


def optimize_for_platform(platform_name: str) -> Dict[str, Any]:
    """Get platform-specific optimization settings"""
    optimizations = {
        "jetson": {
            "precision": "fp16",
            "batch_size": 1,
            "num_workers": 2,
            "pin_memory": True,
            "use_tensorrt": True,
        },
        "raspberry_pi": {
            "precision": "int8",
            "batch_size": 1,
            "num_workers": 1,
            "pin_memory": False,
            "use_onnx": True,
        },
        "cpu": {
            "precision": "fp32",
            "batch_size": 8,
            "num_workers": psutil.cpu_count() // 2,
            "pin_memory": False,
        },
        "gpu": {
            "precision": "fp16",
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
        },
    }

    return optimizations.get(platform_name, optimizations["cpu"])


def check_memory_usage() -> Dict[str, float]:
    """Check current memory usage"""
    memory_info = {
        "cpu_memory_percent": psutil.virtual_memory().percent,
        "cpu_memory_available": psutil.virtual_memory().available / (1024**3),  # GB
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info[f"gpu_{i}_memory_allocated"] = torch.cuda.memory_allocated(i) / (
                1024**3
            )  # GB
            memory_info[f"gpu_{i}_memory_reserved"] = torch.cuda.memory_reserved(i) / (
                1024**3
            )  # GB

    return memory_info
