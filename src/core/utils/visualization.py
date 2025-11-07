"""
Visualization Utilities for SereneSense

This module provides comprehensive visualization tools for the SereneSense
military vehicle sound detection system.

Features:
- Audio waveform and spectrogram plotting
- Training curve visualization
- Confusion matrix and classification reports
- Model architecture visualization
- Real-time inference monitoring
- Performance benchmarking plots
- Interactive dashboards

Example:
    >>> from core.utils.visualization import plot_spectrogram, plot_confusion_matrix
    >>> 
    >>> # Plot spectrogram
    >>> plot_spectrogram(audio_data, sample_rate=16000, save_path="spec.png")
    >>> 
    >>> # Plot confusion matrix
    >>> plot_confusion_matrix(y_true, y_pred, class_names=classes, save_path="cm.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import warnings

# Third-party imports
try:
    import librosa
    import librosa.display

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available for audio visualization", ImportWarning)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available for interactive visualization", ImportWarning)

try:
    from sklearn.metrics import confusion_matrix, classification_report

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8" if hasattr(plt.style, "seaborn-v0_8") else "default")
sns.set_palette("husl")

# Custom colormap for spectrograms
SPECTROGRAM_CMAP = LinearSegmentedColormap.from_list(
    "spectrogram",
    [
        "#000033",
        "#000055",
        "#0000ff",
        "#0055ff",
        "#00ffff",
        "#55ff00",
        "#ffff00",
        "#ff5500",
        "#ff0000",
    ],
)


class AudioVisualizer:
    """
    Comprehensive audio data visualization.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize audio visualizer.

        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_waveform(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        title: str = "Audio Waveform",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot audio waveform.

        Args:
            audio: Audio signal array
            sample_rate: Sample rate in Hz
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Time axis
        time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))

        # Plot waveform
        ax.plot(time_axis, audio, linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"Duration: {len(audio) / sample_rate:.2f}s\n"
        stats_text += f"RMS: {np.sqrt(np.mean(audio**2)):.4f}\n"
        stats_text += f"Peak: {np.max(np.abs(audio)):.4f}"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved waveform plot to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        n_mels: Optional[int] = None,
        title: str = "Spectrogram",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot audio spectrogram.

        Args:
            audio: Audio signal array
            sample_rate: Sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands (if None, plots STFT)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        if not LIBROSA_AVAILABLE:
            logger.error("librosa required for spectrogram plotting")
            return None

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if n_mels is not None:
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            img = librosa.display.specshow(
                mel_spec_db,
                sr=sample_rate,
                hop_length=hop_length,
                x_axis="time",
                y_axis="mel",
                ax=ax,
                cmap=SPECTROGRAM_CMAP,
            )
            ax.set_ylabel("Mel Frequency")

        else:
            # STFT spectrogram
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

            img = librosa.display.specshow(
                stft_db,
                sr=sample_rate,
                hop_length=hop_length,
                x_axis="time",
                y_axis="hz",
                ax=ax,
                cmap=SPECTROGRAM_CMAP,
            )
            ax.set_ylabel("Frequency (Hz)")

        ax.set_xlabel("Time (s)")
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.set_label("Power (dB)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved spectrogram plot to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_feature_comparison(
        self,
        features_dict: Dict[str, np.ndarray],
        feature_names: List[str],
        title: str = "Feature Comparison",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot comparison of different audio features.

        Args:
            features_dict: Dictionary mapping feature names to feature arrays
            feature_names: List of feature names to plot
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        n_features = len(feature_names)
        fig, axes = plt.subplots(
            n_features, 1, figsize=(self.figsize[0], self.figsize[1] * n_features // 2)
        )

        if n_features == 1:
            axes = [axes]

        for i, feature_name in enumerate(feature_names):
            if feature_name in features_dict:
                feature_data = features_dict[feature_name]

                if feature_data.ndim == 2:
                    # 2D feature (e.g., spectrogram)
                    im = axes[i].imshow(feature_data, aspect="auto", origin="lower", cmap="viridis")
                    plt.colorbar(im, ax=axes[i])
                else:
                    # 1D feature (e.g., RMS, spectral centroid)
                    axes[i].plot(feature_data)

                axes[i].set_title(f"{feature_name}")
                axes[i].set_xlabel("Time")

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved feature comparison plot to {save_path}")

        if show:
            plt.show()

        return fig


class TrainingVisualizer:
    """
    Training and model performance visualization.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize training visualizer.

        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        metrics: List[str] = None,
        title: str = "Training Curves",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot training curves.

        Args:
            history: Dictionary containing training history
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]

        available_metrics = [m for m in metrics if m in history]
        n_metrics = len(available_metrics)

        if n_metrics == 0:
            logger.warning("No metrics found in history")
            return None

        fig, axes = plt.subplots(1, n_metrics, figsize=(self.figsize[0], self.figsize[1] // 2))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            # Plot training curve
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                axes[i].plot(epochs, history[metric], "b-", label=f"Training {metric}", linewidth=2)

            # Plot validation curve if available
            val_metric = f"val_{metric}"
            if val_metric in history:
                epochs = range(1, len(history[val_metric]) + 1)
                axes[i].plot(
                    epochs, history[val_metric], "r-", label=f"Validation {metric}", linewidth=2
                )

            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f"{metric.capitalize()} Curve")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # Add best value annotation
            if val_metric in history:
                best_epoch = (
                    np.argmax(history[val_metric])
                    if "acc" in metric
                    else np.argmin(history[val_metric])
                )
                best_value = history[val_metric][best_epoch]
                axes[i].annotate(
                    f"Best: {best_value:.3f}",
                    xy=(best_epoch + 1, best_value),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved training curves to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            normalize: Whether to normalize the matrix
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for confusion matrix")
            return None

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Handle division by zero

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved confusion matrix to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_class_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        title: str = "Per-Class Performance",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot per-class performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for classification report")
            return None

        from sklearn.metrics import precision_recall_fscore_support

        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Create dataframe for plotting
        metrics_data = {
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support,
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] // 2))

        # Plot metrics
        x = np.arange(len(class_names))
        width = 0.25

        ax1.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax1.bar(x, recall, width, label="Recall", alpha=0.8)
        ax1.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

        ax1.set_xlabel("Class")
        ax1.set_ylabel("Score")
        ax1.set_title("Classification Metrics by Class")
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Plot support
        ax2.bar(x, support, alpha=0.8, color="green")
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Number of Samples")
        ax2.set_title("Class Support")
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved class performance plot to {save_path}")

        if show:
            plt.show()

        return fig


class PerformanceVisualizer:
    """
    Performance benchmarking and monitoring visualization.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize performance visualizer.

        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_latency_vs_throughput(
        self,
        benchmark_data: List[Dict[str, Any]],
        title: str = "Latency vs Throughput",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot latency vs throughput scatter plot.

        Args:
            benchmark_data: List of benchmark results
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Extract data
        latencies = [d["latency"] for d in benchmark_data]
        throughputs = [d["throughput"] for d in benchmark_data]
        batch_sizes = [d.get("batch_size", 1) for d in benchmark_data]

        # Create scatter plot with batch size as color
        scatter = ax.scatter(
            latencies, throughputs, c=batch_sizes, cmap="viridis", alpha=0.7, s=100
        )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Batch Size")

        # Add annotations for interesting points
        for i, (lat, thr, bs) in enumerate(zip(latencies, throughputs, batch_sizes)):
            if bs in [1, max(batch_sizes)] or thr == max(throughputs):
                ax.annotate(f"BS={bs}", (lat, thr), xytext=(5, 5), textcoords="offset points")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved latency vs throughput plot to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_memory_usage(
        self,
        memory_data: Dict[str, List[float]],
        title: str = "Memory Usage Over Time",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot memory usage over time.

        Args:
            memory_data: Dictionary containing memory usage data
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for memory_type, values in memory_data.items():
            time_points = np.arange(len(values))
            ax.plot(time_points, values, label=memory_type, linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Memory Usage (GB)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved memory usage plot to {save_path}")

        if show:
            plt.show()

        return fig


class InteractiveDashboard:
    """
    Interactive visualization dashboard using Plotly.
    """

    def __init__(self):
        """Initialize interactive dashboard."""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly required for interactive dashboard")
            return

        self.fig = None

    def create_training_dashboard(self, history: Dict[str, List[float]]) -> go.Figure:
        """
        Create interactive training dashboard.

        Args:
            history: Training history dictionary

        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available")
            return None

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Loss", "Accuracy", "Learning Rate", "Metrics Summary"),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"type": "table"}],
            ],
        )

        epochs = list(range(1, len(history.get("loss", [])) + 1))

        # Loss plot
        if "loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history["loss"], name="Training Loss", line=dict(color="blue")
                ),
                row=1,
                col=1,
            )
        if "val_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history["val_loss"], name="Validation Loss", line=dict(color="red")
                ),
                row=1,
                col=1,
            )

        # Accuracy plot
        if "accuracy" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["accuracy"],
                    name="Training Accuracy",
                    line=dict(color="green"),
                ),
                row=1,
                col=2,
            )
        if "val_accuracy" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val_accuracy"],
                    name="Validation Accuracy",
                    line=dict(color="orange"),
                ),
                row=1,
                col=2,
            )

        # Learning rate plot
        if "lr" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history["lr"], name="Learning Rate", line=dict(color="purple")
                ),
                row=2,
                col=1,
            )

        # Summary table
        if "val_accuracy" in history and "val_loss" in history:
            best_epoch = np.argmax(history["val_accuracy"])
            summary_data = {
                "Metric": [
                    "Best Validation Accuracy",
                    "Best Validation Loss",
                    "Final Learning Rate",
                ],
                "Value": [
                    f"{max(history['val_accuracy']):.4f}",
                    f"{min(history['val_loss']):.4f}",
                    f"{history.get('lr', [0])[-1]:.6f}" if "lr" in history else "N/A",
                ],
                "Epoch": [
                    str(best_epoch + 1),
                    str(np.argmin(history["val_loss"]) + 1),
                    str(len(epochs)),
                ],
            }

            fig.add_trace(
                go.Table(
                    header=dict(values=list(summary_data.keys()), fill_color="paleturquoise"),
                    cells=dict(values=list(summary_data.values()), fill_color="lavender"),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(title="Training Dashboard", showlegend=True, height=600)

        self.fig = fig
        return fig

    def create_realtime_monitor(self) -> go.Figure:
        """
        Create real-time monitoring dashboard.

        Returns:
            Plotly figure object for real-time updates
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available")
            return None

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Inference Latency", "Throughput", "Memory Usage", "System Resources"),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
            ],
        )

        # Initialize empty traces
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Latency (ms)", mode="lines+markers"), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=[], y=[], name="Throughput (FPS)", mode="lines+markers"), row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=[], y=[], name="Memory (GB)", mode="lines+markers"), row=2, col=1
        )

        fig.add_trace(go.Scatter(x=[], y=[], name="CPU (%)", mode="lines+markers"), row=2, col=2)

        fig.update_layout(title="Real-time Performance Monitor", showlegend=True, height=600)

        self.fig = fig
        return fig

    def save_dashboard(self, file_path: str, format: str = "html") -> None:
        """
        Save dashboard to file.

        Args:
            file_path: Output file path
            format: Output format ('html', 'png', 'pdf')
        """
        if not PLOTLY_AVAILABLE or self.fig is None:
            logger.error("No figure to save or Plotly not available")
            return

        if format == "html":
            pyo.plot(self.fig, filename=file_path, auto_open=False)
        else:
            self.fig.write_image(file_path, format=format)

        logger.info(f"Saved dashboard to {file_path}")


# Convenience functions
def plot_spectrogram(audio: np.ndarray, sample_rate: int = 16000, **kwargs) -> plt.Figure:
    """
    Convenience function to plot spectrogram.

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        **kwargs: Additional arguments for AudioVisualizer.plot_spectrogram

    Returns:
        matplotlib Figure object
    """
    visualizer = AudioVisualizer()
    return visualizer.plot_spectrogram(audio, sample_rate, **kwargs)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], **kwargs
) -> plt.Figure:
    """
    Convenience function to plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        **kwargs: Additional arguments for TrainingVisualizer.plot_confusion_matrix

    Returns:
        matplotlib Figure object
    """
    visualizer = TrainingVisualizer()
    return visualizer.plot_confusion_matrix(y_true, y_pred, class_names, **kwargs)


def plot_training_curves(history: Dict[str, List[float]], **kwargs) -> plt.Figure:
    """
    Convenience function to plot training curves.

    Args:
        history: Training history
        **kwargs: Additional arguments for TrainingVisualizer.plot_training_curves

    Returns:
        matplotlib Figure object
    """
    visualizer = TrainingVisualizer()
    return visualizer.plot_training_curves(history, **kwargs)


def save_plot(fig: plt.Figure, path: str, dpi: int = 300, **kwargs) -> None:
    """
    Save matplotlib figure with high quality settings.

    Args:
        fig: matplotlib Figure object
        path: Output file path
        dpi: Figure DPI
        **kwargs: Additional arguments for plt.savefig
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    logger.info(f"Saved plot to {path}")


def create_visualization_report(
    results: Dict[str, Any], output_dir: str = "outputs/visualizations"
) -> None:
    """
    Create a comprehensive visualization report.

    Args:
        results: Dictionary containing results to visualize
        output_dir: Output directory for visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training curves
    if "history" in results:
        fig = plot_training_curves(results["history"], show=False)
        if fig:
            save_plot(fig, output_path / "training_curves.png")
            plt.close(fig)

    # Confusion matrix
    if "y_true" in results and "y_pred" in results and "class_names" in results:
        fig = plot_confusion_matrix(
            results["y_true"], results["y_pred"], results["class_names"], show=False
        )
        if fig:
            save_plot(fig, output_path / "confusion_matrix.png")
            plt.close(fig)

    # Performance metrics
    if "benchmark_data" in results:
        visualizer = PerformanceVisualizer()
        fig = visualizer.plot_latency_vs_throughput(results["benchmark_data"], show=False)
        if fig:
            save_plot(fig, output_path / "performance_benchmark.png")
            plt.close(fig)

    logger.info(f"Visualization report saved to {output_dir}")
