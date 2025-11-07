"""
SereneSense Core Package

This package contains the core functionality for the SereneSense
military vehicle sound detection system.

Modules:
    - audio_processor: Audio preprocessing and feature extraction pipeline
    - feature_extractor: Advanced feature extraction algorithms
    - model_manager: Model loading, management, and optimization

The core package provides the fundamental building blocks for audio processing,
feature extraction, and model management that are used throughout the SereneSense system.

Example:
    >>> from core.core import AudioProcessor, FeatureExtractor, ModelManager
    >>> 
    >>> # Initialize components
    >>> processor = AudioProcessor(config)
    >>> extractor = FeatureExtractor(config)
    >>> manager = ModelManager(config)
    >>> 
    >>> # Process audio
    >>> audio_data = processor.load_audio("path/to/audio.wav")
    >>> features = extractor.extract(audio_data)
    >>> 
    >>> # Load and use model
    >>> model = manager.load_model("audioMAE")
    >>> predictions = model(features)
"""

import logging
from typing import Dict, Any, Optional
import warnings

# Version information
__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# Import core components
try:
    from .audio_processor import AudioProcessor
    from .feature_extractor import FeatureExtractor
    from .model_manager import ModelManager

except ImportError as e:
    warnings.warn(f"Some core components could not be imported: {e}", ImportWarning)
    logger.warning(f"Import error in core package: {e}")

# Default configurations for core components
DEFAULT_AUDIO_CONFIG = {
    "sample_rate": 16000,
    "duration": 10.0,
    "normalize": True,
    "remove_silence": False,
    "preemphasis": 0.97,
}

DEFAULT_FEATURE_CONFIG = {
    "spectrogram": {
        "n_fft": 1024,
        "hop_length": 160,
        "win_length": 1024,
        "window": "hann",
        "n_mels": 128,
        "f_min": 50,
        "f_max": 8000,
        "power": 2.0,
    },
    "normalization": {"type": "instance", "mean": [0.485], "std": [0.229]},
}

DEFAULT_MODEL_CONFIG = {
    "device": "auto",
    "precision": "fp32",
    "optimization": {"jit_compile": False, "channels_last": False, "mixed_precision": False},
    "cache": {"enabled": True, "max_size": "2GB"},
}


class CorePipeline:
    """
    Integrated pipeline combining audio processing, feature extraction, and model management.

    This class provides a high-level interface that combines all core components
    into a single, easy-to-use pipeline for audio processing and inference.
    """

    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the core pipeline.

        Args:
            audio_config: Audio processing configuration
            feature_config: Feature extraction configuration
            model_config: Model management configuration
        """
        # Merge with defaults
        self.audio_config = {**DEFAULT_AUDIO_CONFIG, **(audio_config or {})}
        self.feature_config = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}
        self.model_config = {**DEFAULT_MODEL_CONFIG, **(model_config or {})}

        # Initialize components
        self._audio_processor = None
        self._feature_extractor = None
        self._model_manager = None

        logger.info("CorePipeline initialized")

    @property
    def audio_processor(self) -> "AudioProcessor":
        """Get audio processor instance (lazy initialization)."""
        if self._audio_processor is None:
            self._audio_processor = AudioProcessor(self.audio_config)
        return self._audio_processor

    @property
    def feature_extractor(self) -> "FeatureExtractor":
        """Get feature extractor instance (lazy initialization)."""
        if self._feature_extractor is None:
            self._feature_extractor = FeatureExtractor(self.feature_config)
        return self._feature_extractor

    @property
    def model_manager(self) -> "ModelManager":
        """Get model manager instance (lazy initialization)."""
        if self._model_manager is None:
            self._model_manager = ModelManager(self.model_config)
        return self._model_manager

    def process_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary containing processed audio data and metadata
        """
        # Load and preprocess audio
        audio_data = self.audio_processor.load_audio(file_path)

        # Extract features
        features = self.feature_extractor.extract(audio_data)

        return {
            "audio_data": audio_data,
            "features": features,
            "metadata": {
                "file_path": file_path,
                "sample_rate": self.audio_config["sample_rate"],
                "duration": len(audio_data) / self.audio_config["sample_rate"],
                "feature_shape": features.shape if hasattr(features, "shape") else None,
            },
        }

    def predict(self, audio_input, model_name: str = None) -> Dict[str, Any]:
        """
        Make predictions on audio input.

        Args:
            audio_input: Audio file path or preprocessed audio data
            model_name: Name of model to use for prediction

        Returns:
            Dictionary containing predictions and confidence scores
        """
        # Process input
        if isinstance(audio_input, str):
            # File path provided
            processed = self.process_audio_file(audio_input)
            features = processed["features"]
        else:
            # Assume preprocessed audio data
            features = self.feature_extractor.extract(audio_input)

        # Load model if needed
        if model_name:
            model = self.model_manager.load_model(model_name)
        else:
            model = self.model_manager.get_default_model()

        # Make prediction
        predictions = self.model_manager.predict(model, features)

        return predictions

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.

        Returns:
            Dictionary containing pipeline information
        """
        return {
            "audio_config": self.audio_config,
            "feature_config": self.feature_config,
            "model_config": self.model_config,
            "components": {
                "audio_processor": self._audio_processor is not None,
                "feature_extractor": self._feature_extractor is not None,
                "model_manager": self._model_manager is not None,
            },
        }


def create_pipeline(config: Optional[Dict[str, Any]] = None) -> CorePipeline:
    """
    Create a core pipeline with configuration.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Initialized CorePipeline instance
    """
    if config is None:
        config = {}

    return CorePipeline(
        audio_config=config.get("audio", None),
        feature_config=config.get("features", None),
        model_config=config.get("model", None),
    )


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate core configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if configuration is valid
    """
    try:
        # Check required fields
        audio_config = config.get("audio", {})
        if "sample_rate" in audio_config:
            assert isinstance(audio_config["sample_rate"], int)
            assert audio_config["sample_rate"] > 0

        feature_config = config.get("features", {})
        if "spectrogram" in feature_config:
            spec_config = feature_config["spectrogram"]
            if "n_fft" in spec_config:
                assert isinstance(spec_config["n_fft"], int)
                assert spec_config["n_fft"] > 0

        model_config = config.get("model", {})
        if "device" in model_config:
            assert isinstance(model_config["device"], str)

        return True

    except (AssertionError, KeyError, TypeError) as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def get_supported_formats() -> Dict[str, list]:
    """
    Get supported audio formats and model types.

    Returns:
        Dictionary containing supported formats
    """
    return {
        "audio_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        "model_types": ["audioMAE", "ast", "beats"],
        "feature_types": ["mel_spectrogram", "mfcc", "chroma", "spectral_contrast"],
        "devices": ["cpu", "cuda", "mps"],
    }


# Export main classes and functions
__all__ = [
    # Main classes
    "AudioProcessor",
    "FeatureExtractor",
    "ModelManager",
    "CorePipeline",
    # Factory functions
    "create_pipeline",
    # Utility functions
    "validate_config",
    "get_supported_formats",
    # Default configurations
    "DEFAULT_AUDIO_CONFIG",
    "DEFAULT_FEATURE_CONFIG",
    "DEFAULT_MODEL_CONFIG",
]

# Package initialization
logger.info(f"SereneSense Core v{__version__} loaded")

# Check for critical dependencies
try:
    import torch
    import torchaudio
    import librosa
    import numpy as np

    logger.debug("All critical dependencies available")

except ImportError as e:
    logger.warning(f"Missing critical dependency: {e}")
    warnings.warn(f"Missing critical dependency: {e}", ImportWarning)
