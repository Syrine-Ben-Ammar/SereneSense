"""
SereneSense Utilities Package

This package contains utility functions and classes for the SereneSense
military vehicle sound detection system.

Modules:
    - config_parser: Configuration file parsing and validation
    - device_utils: Device detection and hardware utilities
    - logging: Logging configuration and utilities
    - metrics: Evaluation metrics and performance monitoring
    - visualization: Data visualization and plotting utilities

Usage:
    >>> from core.utils import setup_logging, get_device_info
    >>> from core.utils.config_parser import ConfigParser
    >>> from core.utils.metrics import ClassificationMetrics
"""

import sys
import warnings
from typing import Dict, Any, Optional
from pathlib import Path

# Version information
__version__ = "1.0.0"

# Import core utilities
try:
    from .logging import setup_logging, get_logger
    from .config_parser import ConfigParser, load_config, validate_config
    from .device_utils import (
        get_device_info,
        get_optimal_device,
        check_gpu_availability,
        get_memory_info,
    )
    from .metrics import ClassificationMetrics, compute_metrics, log_metrics
    from .visualization import (
        plot_confusion_matrix,
        plot_training_curves,
        plot_spectrogram,
        save_plot,
    )

except ImportError as e:
    warnings.warn(f"Some utilities could not be imported: {e}", ImportWarning)


# Utility functions for common operations
def setup_environment(config_path: Optional[str] = None, log_level: str = "INFO") -> Dict[str, Any]:
    """
    Setup the SereneSense environment with logging and configuration.

    Args:
        config_path: Path to configuration file
        log_level: Logging level

    Returns:
        Dictionary containing environment information
    """
    # Setup logging
    logger = setup_logging(level=log_level)

    # Load configuration if provided
    config = None
    if config_path:
        try:
            config = load_config(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # Get device information
    device_info = get_device_info()
    logger.info(f"Device: {device_info['device_type']}")

    if device_info["gpu_available"]:
        logger.info(f"GPU: {device_info['gpu_name']} " f"({device_info['gpu_memory_gb']:.1f}GB)")

    return {"config": config, "device_info": device_info, "logger": logger}


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.

    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {}

    # Core dependencies
    try:
        import torch

        dependencies["torch"] = True
        dependencies["torch_version"] = torch.__version__
        dependencies["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        dependencies["torch"] = False

    try:
        import torchaudio

        dependencies["torchaudio"] = True
        dependencies["torchaudio_version"] = torchaudio.__version__
    except ImportError:
        dependencies["torchaudio"] = False

    try:
        import librosa

        dependencies["librosa"] = True
        dependencies["librosa_version"] = librosa.__version__
    except ImportError:
        dependencies["librosa"] = False

    try:
        import numpy

        dependencies["numpy"] = True
        dependencies["numpy_version"] = numpy.__version__
    except ImportError:
        dependencies["numpy"] = False

    try:
        import sklearn

        dependencies["sklearn"] = True
        dependencies["sklearn_version"] = sklearn.__version__
    except ImportError:
        dependencies["sklearn"] = False

    # Optional dependencies
    try:
        import wandb

        dependencies["wandb"] = True
    except ImportError:
        dependencies["wandb"] = False

    try:
        import mlflow

        dependencies["mlflow"] = True
    except ImportError:
        dependencies["mlflow"] = False

    try:
        import tensorrt

        dependencies["tensorrt"] = True
    except ImportError:
        dependencies["tensorrt"] = False

    try:
        import onnx

        dependencies["onnx"] = True
    except ImportError:
        dependencies["onnx"] = False

    return dependencies


def validate_environment() -> bool:
    """
    Validate that the environment is properly set up for SereneSense.

    Returns:
        True if environment is valid, False otherwise
    """
    deps = check_dependencies()

    # Check required dependencies
    required = ["torch", "torchaudio", "librosa", "numpy", "sklearn"]
    missing = [dep for dep in required if not deps.get(dep, False)]

    if missing:
        print(f"Missing required dependencies: {missing}")
        return False

    # Check PyTorch version compatibility
    if deps.get("torch"):
        import torch

        version = torch.__version__
        major, minor = map(int, version.split(".")[:2])
        if major < 2 or (major == 2 and minor < 1):
            print(
                f"PyTorch version {version} is not supported. " "Please upgrade to 2.1.0 or later."
            )
            return False

    return True


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.

    Returns:
        Dictionary containing system information
    """
    import platform
    import psutil

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "dependencies": check_dependencies(),
    }

    # Add device information
    info.update(get_device_info())

    return info


def create_directories(base_dir: str) -> Dict[str, Path]:
    """
    Create standard directory structure for SereneSense.

    Args:
        base_dir: Base directory path

    Returns:
        Dictionary mapping directory names to Path objects
    """
    base_path = Path(base_dir)

    directories = {
        "data": base_path / "data",
        "models": base_path / "models",
        "logs": base_path / "logs",
        "outputs": base_path / "outputs",
        "cache": base_path / "cache",
        "experiments": base_path / "experiments",
    }

    # Create subdirectories
    subdirs = {
        "data_raw": directories["data"] / "raw",
        "data_processed": directories["data"] / "processed",
        "data_cache": directories["data"] / "cache",
        "models_checkpoints": directories["models"] / "checkpoints",
        "models_pretrained": directories["models"] / "pretrained",
        "models_optimized": directories["models"] / "optimized",
        "logs_training": directories["logs"] / "training",
        "logs_inference": directories["logs"] / "inference",
        "outputs_predictions": directories["outputs"] / "predictions",
        "outputs_visualizations": directories["outputs"] / "visualizations",
    }

    directories.update(subdirs)

    # Create all directories
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)

    return directories


# Export main utilities
__all__ = [
    # Setup functions
    "setup_environment",
    "setup_logging",
    "get_logger",
    # Configuration utilities
    "ConfigParser",
    "load_config",
    "validate_config",
    # Device utilities
    "get_device_info",
    "get_optimal_device",
    "check_gpu_availability",
    "get_memory_info",
    # Metrics utilities
    "ClassificationMetrics",
    "compute_metrics",
    "log_metrics",
    # Visualization utilities
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_spectrogram",
    "save_plot",
    # Environment utilities
    "check_dependencies",
    "validate_environment",
    "get_system_info",
    "create_directories",
]

# Package initialization message
if __name__ != "__main__":
    # Only show this when imported, not when run directly
    import logging

    logger = logging.getLogger(__name__)
    logger.debug(f"SereneSense utilities v{__version__} loaded")

# Convenience imports for backward compatibility
try:
    # Import commonly used functions to top level
    from .config_parser import load_config as load_config
    from .device_utils import get_device_info as get_device_info
    from .logging import setup_logging as setup_logging

except ImportError:
    # If imports fail, continue without them
    pass
