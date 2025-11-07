"""
SereneSense Data Package

This package contains data handling components for the SereneSense
military vehicle sound detection system.

Modules:
    - loaders: Dataset loaders for MAD, AudioSet, FSD50K
    - augmentation: Audio data augmentation techniques
    - preprocessing: Audio preprocessing and feature preparation

The data package provides comprehensive data pipeline functionality including:
- Multi-format dataset loading and management
- Real-time and batch audio augmentation
- Preprocessing pipelines for different model architectures
- Data validation and quality control
- Efficient data streaming and caching

Example:
    >>> from core.data import MADDataset, AudioAugmentation
    >>> from core.data.preprocessing import SpectrogramProcessor
    >>> 
    >>> # Load dataset
    >>> dataset = MADDataset("path/to/mad", split="train")
    >>> 
    >>> # Setup augmentation
    >>> augmenter = AudioAugmentation(config)
    >>> 
    >>> # Create data pipeline
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
"""

import logging
from typing import Dict, Any, List, Optional, Union
import warnings

# Version information
__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# Import core data components
try:
    from .loaders.mad_loader import MADDataset, MADDataLoader
    from .loaders.audioset_loader import AudioSetDataset, AudioSetDataLoader
    from .loaders.fsd50k_loader import FSD50KDataset, FSD50KDataLoader
    
except ImportError as e:
    warnings.warn(f"Some data loaders could not be imported: {e}", ImportWarning)
    logger.warning(f"Import error in data loaders: {e}")

try:
    from .augmentation.time_domain import TimeDomainAugmentation
    from .augmentation.frequency_domain import FrequencyDomainAugmentation
    from .augmentation.spec_augment import SpecAugment
    
except ImportError as e:
    warnings.warn(f"Some augmentation modules could not be imported: {e}", ImportWarning)
    logger.warning(f"Import error in augmentation: {e}")

try:
    from .preprocessing.spectrograms import SpectrogramProcessor
    from .preprocessing.normalization import AudioNormalizer
    from .preprocessing.segmentation import AudioSegmenter
    
except ImportError as e:
    warnings.warn(f"Some preprocessing modules could not be imported: {e}", ImportWarning)
    logger.warning(f"Import error in preprocessing: {e}")

# Default configurations for data components
DEFAULT_LOADER_CONFIG = {
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,
    "shuffle": True,
    "drop_last": True
}

DEFAULT_AUGMENTATION_CONFIG = {
    "probability": 0.8,
    "time_domain": {
        "time_shift": {"enabled": True, "max_shift": 0.1},
        "pitch_shift": {"enabled": True, "max_steps": 2},
        "noise_injection": {"enabled": True, "snr_range": [10, 50]}
    },
    "frequency_domain": {
        "freq_mask": {"enabled": True, "max_mask_pct": 0.15},
        "time_mask": {"enabled": True, "max_mask_pct": 0.10}
    }
}

DEFAULT_PREPROCESSING_CONFIG = {
    "sample_rate": 16000,
    "duration": 10.0,
    "normalize": True,
    "remove_silence": False,
    "spectrogram": {
        "n_fft": 1024,
        "hop_length": 160,
        "n_mels": 128,
        "f_min": 50,
        "f_max": 8000
    }
}

# Dataset registry for dynamic loading
DATASET_REGISTRY = {
    "mad": "MADDataset",
    "audioset": "AudioSetDataset", 
    "fsd50k": "FSD50KDataset"
}

# Class mapping for military vehicle detection
MILITARY_CLASSES = {
    "helicopter": 0,
    "fighter_aircraft": 1,
    "military_vehicle": 2,
    "truck": 3,
    "footsteps": 4,
    "speech": 5,
    "background": 6
}

# Inverse mapping
CLASS_NAMES = {v: k for k, v in MILITARY_CLASSES.items()}


class DataPipeline:
    """
    Integrated data pipeline combining loading, augmentation, and preprocessing.
    
    This class provides a high-level interface for creating complete data
    processing pipelines for training and inference.
    """
    
    def __init__(self, 
                 dataset_config: Dict[str, Any],
                 augmentation_config: Optional[Dict[str, Any]] = None,
                 preprocessing_config: Optional[Dict[str, Any]] = None):
        """
        Initialize data pipeline.
        
        Args:
            dataset_config: Dataset configuration
            augmentation_config: Augmentation configuration
            preprocessing_config: Preprocessing configuration
        """
        self.dataset_config = dataset_config
        self.augmentation_config = augmentation_config or DEFAULT_AUGMENTATION_CONFIG
        self.preprocessing_config = preprocessing_config or DEFAULT_PREPROCESSING_CONFIG
        
        # Initialize components
        self._dataset = None
        self._augmenter = None
        self._preprocessor = None
        
        logger.info("DataPipeline initialized")
    
    def create_dataset(self, dataset_name: str, data_path: str, 
                      split: str = "train") -> 'BaseDataset':
        """
        Create dataset instance.
        
        Args:
            dataset_name: Name of dataset (mad, audioset, fsd50k)
            data_path: Path to dataset
            split: Dataset split (train, val, test)
            
        Returns:
            Dataset instance
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_class_name = DATASET_REGISTRY[dataset_name]
        
        if dataset_name == "mad":
            self._dataset = MADDataset(
                data_path, 
                split=split, 
                config=self.dataset_config
            )
        elif dataset_name == "audioset":
            self._dataset = AudioSetDataset(
                data_path,
                split=split,
                config=self.dataset_config
            )
        elif dataset_name == "fsd50k":
            self._dataset = FSD50KDataset(
                data_path,
                split=split,
                config=self.dataset_config
            )
        
        return self._dataset
    
    def create_dataloader(self, dataset: 'BaseDataset', 
                         loader_config: Optional[Dict[str, Any]] = None) -> 'torch.utils.data.DataLoader':
        """
        Create data loader for dataset.
        
        Args:
            dataset: Dataset instance
            loader_config: Data loader configuration
            
        Returns:
            Configured data loader
        """
        import torch.utils.data
        
        config = {**DEFAULT_LOADER_CONFIG, **(loader_config or {})}
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=config["drop_last"]
        )
    
    def setup_augmentation(self) -> 'BaseAugmentation':
        """
        Setup audio augmentation pipeline.
        
        Returns:
            Configured augmentation instance
        """
        # This would typically combine multiple augmentation techniques
        # For now, return a placeholder
        from .augmentation.time_domain import TimeDomainAugmentation
        
        self._augmenter = TimeDomainAugmentation(self.augmentation_config)
        return self._augmenter
    
    def setup_preprocessing(self) -> 'BasePreprocessor':
        """
        Setup audio preprocessing pipeline.
        
        Returns:
            Configured preprocessing instance
        """
        from .preprocessing.spectrograms import SpectrogramProcessor
        
        self._preprocessor = SpectrogramProcessor(self.preprocessing_config)
        return self._preprocessor
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary containing pipeline information
        """
        return {
            "dataset_config": self.dataset_config,
            "augmentation_config": self.augmentation_config,
            "preprocessing_config": self.preprocessing_config,
            "components": {
                "dataset": self._dataset is not None,
                "augmenter": self._augmenter is not None,
                "preprocessor": self._preprocessor is not None
            }
        }


def create_mad_pipeline(data_path: str, 
                       split: str = "train",
                       config: Optional[Dict[str, Any]] = None) -> DataPipeline:
    """
    Create a MAD dataset pipeline with default configurations.
    
    Args:
        data_path: Path to MAD dataset
        split: Dataset split
        config: Optional configuration overrides
        
    Returns:
        Configured data pipeline
    """
    default_config = {
        "dataset": {"sample_rate": 16000, "duration": 10.0},
        "augmentation": DEFAULT_AUGMENTATION_CONFIG,
        "preprocessing": DEFAULT_PREPROCESSING_CONFIG
    }
    
    if config:
        # Merge configurations
        for key, value in config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    pipeline = DataPipeline(
        dataset_config=default_config["dataset"],
        augmentation_config=default_config["augmentation"],
        preprocessing_config=default_config["preprocessing"]
    )
    
    # Create MAD dataset
    pipeline.create_dataset("mad", data_path, split)
    
    return pipeline


def create_audioset_pipeline(data_path: str,
                           split: str = "train",
                           config: Optional[Dict[str, Any]] = None) -> DataPipeline:
    """
    Create an AudioSet dataset pipeline with default configurations.
    
    Args:
        data_path: Path to AudioSet dataset
        split: Dataset split
        config: Optional configuration overrides
        
    Returns:
        Configured data pipeline
    """
    default_config = {
        "dataset": {"sample_rate": 16000, "duration": 10.0},
        "augmentation": DEFAULT_AUGMENTATION_CONFIG,
        "preprocessing": DEFAULT_PREPROCESSING_CONFIG
    }
    
    if config:
        for key, value in config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    pipeline = DataPipeline(
        dataset_config=default_config["dataset"],
        augmentation_config=default_config["augmentation"],
        preprocessing_config=default_config["preprocessing"]
    )
    
    # Create AudioSet dataset
    pipeline.create_dataset("audioset", data_path, split)
    
    return pipeline


def get_class_weights(dataset_name: str, data_path: str) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset_name: Name of dataset
        data_path: Path to dataset
        
    Returns:
        Dictionary mapping class indices to weights
    """
    try:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Create dataset to get class distribution
        if dataset_name == "mad":
            dataset = MADDataset(data_path, split="train")
        elif dataset_name == "audioset":
            dataset = AudioSetDataset(data_path, split="train") 
        elif dataset_name == "fsd50k":
            dataset = FSD50KDataset(data_path, split="train")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Get all labels
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        
        labels = np.array(labels)
        unique_classes = np.unique(labels)
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=labels
        )
        
        return {cls: weight for cls, weight in zip(unique_classes, class_weights)}
        
    except ImportError:
        logger.warning("scikit-learn not available for class weight computation")
        return {}
    except Exception as e:
        logger.error(f"Error computing class weights: {e}")
        return {}


def validate_dataset(dataset_path: str, dataset_name: str) -> Dict[str, Any]:
    """
    Validate dataset structure and contents.
    
    Args:
        dataset_path: Path to dataset
        dataset_name: Name of dataset
        
    Returns:
        Validation results
    """
    from pathlib import Path
    
    validation_results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }
    
    try:
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            validation_results["errors"].append(f"Dataset path does not exist: {dataset_path}")
            return validation_results
        
        # Dataset-specific validation
        if dataset_name == "mad":
            validation_results = _validate_mad_dataset(dataset_path, validation_results)
        elif dataset_name == "audioset":
            validation_results = _validate_audioset_dataset(dataset_path, validation_results)
        elif dataset_name == "fsd50k":
            validation_results = _validate_fsd50k_dataset(dataset_path, validation_results)
        else:
            validation_results["errors"].append(f"Unknown dataset: {dataset_name}")
            return validation_results
        
        # Set validation status
        validation_results["valid"] = len(validation_results["errors"]) == 0
        
    except Exception as e:
        validation_results["errors"].append(f"Validation error: {str(e)}")
    
    return validation_results


def _validate_mad_dataset(dataset_path: Path, results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate MAD dataset structure."""
    # Check for required directories/files
    required_items = ["audio", "metadata.csv"]
    
    for item in required_items:
        item_path = dataset_path / item
        if not item_path.exists():
            results["errors"].append(f"Missing required item: {item}")
    
    # Check audio files
    audio_dir = dataset_path / "audio"
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav"))
        results["statistics"]["num_audio_files"] = len(audio_files)
        
        if len(audio_files) == 0:
            results["warnings"].append("No audio files found")
    
    return results


def _validate_audioset_dataset(dataset_path: Path, results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate AudioSet dataset structure."""
    # AudioSet validation logic
    required_items = ["audio", "metadata"]
    
    for item in required_items:
        item_path = dataset_path / item
        if not item_path.exists():
            results["errors"].append(f"Missing required item: {item}")
    
    return results


def _validate_fsd50k_dataset(dataset_path: Path, results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate FSD50K dataset structure."""
    # FSD50K validation logic
    required_items = ["FSD50K.dev_audio", "FSD50K.eval_audio", "FSD50K.metadata"]
    
    for item in required_items:
        item_path = dataset_path / item
        if not item_path.exists():
            results["errors"].append(f"Missing required item: {item}")
    
    return results


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about supported datasets.
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Dataset information
    """
    dataset_info = {
        "mad": {
            "name": "Military Audio Detection Dataset",
            "classes": list(MILITARY_CLASSES.keys()),
            "num_classes": len(MILITARY_CLASSES),
            "sample_rate": 16000,
            "total_samples": 8075,
            "license": "CC BY 4.0",
            "url": "https://zenodo.org/record/7088442"
        },
        "audioset": {
            "name": "AudioSet",
            "classes": "632 classes",
            "num_classes": 632,
            "sample_rate": 16000,
            "total_samples": 2084320,
            "license": "CC BY 4.0",
            "url": "https://research.google.com/audioset/"
        },
        "fsd50k": {
            "name": "FSD50K",
            "classes": "200 classes", 
            "num_classes": 200,
            "sample_rate": 44100,
            "total_samples": 51197,
            "license": "CC BY 4.0",
            "url": "https://zenodo.org/record/4060432"
        }
    }
    
    return dataset_info.get(dataset_name, {})


# Export main classes and functions
__all__ = [
    # Dataset loaders
    "MADDataset",
    "MADDataLoader", 
    "AudioSetDataset",
    "AudioSetDataLoader",
    "FSD50KDataset",
    "FSD50KDataLoader",
    
    # Augmentation
    "TimeDomainAugmentation",
    "FrequencyDomainAugmentation", 
    "SpecAugment",
    
    # Preprocessing
    "SpectrogramProcessor",
    "AudioNormalizer",
    "AudioSegmenter",
    
    # Pipeline
    "DataPipeline",
    
    # Factory functions
    "create_mad_pipeline",
    "create_audioset_pipeline",
    
    # Utility functions
    "get_class_weights",
    "validate_dataset",
    "get_dataset_info",
    
    # Constants
    "MILITARY_CLASSES",
    "CLASS_NAMES",
    "DATASET_REGISTRY",
    "DEFAULT_LOADER_CONFIG",
    "DEFAULT_AUGMENTATION_CONFIG", 
    "DEFAULT_PREPROCESSING_CONFIG",
]

# Package initialization
logger.info(f"SereneSense Data v{__version__} loaded")

# Check for critical dependencies
try:
    import torch
    import torchaudio
    import numpy as np
    
    logger.debug("All critical data dependencies available")
    
except ImportError as e:
    logger.warning(f"Missing critical dependency: {e}")
    warnings.warn(f"Missing critical dependency: {e}", ImportWarning)