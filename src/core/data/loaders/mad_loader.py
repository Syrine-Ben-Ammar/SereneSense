"""
MAD Dataset Loader for SereneSense

This module provides data loading functionality for the Military Audio Detection (MAD)
dataset, which is the primary dataset for SereneSense military vehicle sound detection.

The MAD dataset contains 8,075 audio samples across 7 classes:
- Helicopter (1,200 samples)
- Fighter Aircraft (1,100 samples)  
- Military Vehicle (1,500 samples)
- Truck (1,300 samples)
- Footsteps (1,200 samples)
- Speech (1,200 samples)
- Background (1,475 samples)

Features:
- Efficient audio loading with caching
- Stratified train/validation/test splits
- Data validation and quality control
- Metadata parsing and management
- Support for data augmentation
- Class balancing utilities

Example:
    >>> from core.data.loaders.mad_loader import MADDataset
    >>> 
    >>> # Load training set
    >>> dataset = MADDataset("path/to/mad", split="train")
    >>> 
    >>> # Get sample
    >>> audio, label = dataset[0]
    >>> print(f"Audio shape: {audio.shape}, Label: {label}")
"""

import os
import csv
import json
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
from collections import defaultdict, Counter
import warnings

# Third-party imports
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available for splitting", ImportWarning)

logger = logging.getLogger(__name__)

# MAD dataset class mapping
MAD_CLASSES = {
    "Helicopter": 0,
    "Fighter Aircraft": 1, 
    "Military Vehicle": 2,
    "Truck": 3,
    "Footsteps": 4,
    "Speech": 5,
    "Background": 6
}

# Inverse mapping
MAD_CLASS_NAMES = {v: k for k, v in MAD_CLASSES.items()}

# Expected dataset statistics
MAD_STATISTICS = {
    "total_samples": 8075,
    "sample_rate": 16000,
    "duration_hours": 8.96,
    "classes": 7,
    "class_counts": {
        "Helicopter": 1200,
        "Fighter Aircraft": 1100,
        "Military Vehicle": 1500,
        "Truck": 1300,
        "Footsteps": 1200,
        "Speech": 1200,
        "Background": 1475
    }
}


class MADDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the Military Audio Detection (MAD) dataset.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 split: str = "train",
                 config: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        """
        Initialize MAD dataset.
        
        Args:
            data_dir: Path to MAD dataset directory
            split: Dataset split ('train', 'val', 'test', 'all')
            config: Dataset configuration
            transform: Optional transform to apply to audio
            target_transform: Optional transform to apply to targets
            download: Whether to download the dataset if not found
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or {}
        self.transform = transform
        self.target_transform = target_transform
        
        # Configuration parameters
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.duration = self.config.get("duration", None)
        self.normalize = self.config.get("normalize", True)
        self.cache_audio = self.config.get("cache_audio", False)
        
        # Split configuration
        self.split_ratios = self.config.get("split_ratios", {
            "train": 0.7, "val": 0.15, "test": 0.15
        })
        self.random_state = self.config.get("random_state", 42)
        
        # Initialize dataset
        self._audio_cache = {}
        self._metadata = None
        self._file_list = []
        self._labels = []
        
        # Setup dataset
        self._setup_dataset(download)
        
        logger.info(f"MAD Dataset initialized - Split: {split}, "
                   f"Samples: {len(self._file_list)}")
    
    def _setup_dataset(self, download: bool = False) -> None:
        """Setup dataset by loading metadata and creating splits."""
        
        # Check if dataset exists
        if not self._check_dataset_exists():
            if download:
                self._download_dataset()
            else:
                raise FileNotFoundError(f"MAD dataset not found at {self.data_dir}")
        
        # Load metadata
        self._load_metadata()
        
        # Create splits
        self._create_splits()
        
        # Validate dataset
        self._validate_dataset()
    
    def _check_dataset_exists(self) -> bool:
        """Check if dataset exists and has required structure."""
        required_paths = [
            self.data_dir / "audio",
            self.data_dir / "metadata.csv"
        ]
        
        return all(path.exists() for path in required_paths)
    
    def _download_dataset(self) -> None:
        """Download MAD dataset."""
        logger.warning("Automatic download not implemented. "
                      "Please download MAD dataset manually from: "
                      "https://zenodo.org/record/7088442")
        raise NotImplementedError("Automatic download not implemented")
    
    def _load_metadata(self) -> None:
        """Load and parse dataset metadata."""
        metadata_path = self.data_dir / "metadata.csv"
        
        if not metadata_path.exists():
            # Try to create metadata from directory structure
            self._create_metadata_from_structure()
        else:
            # Load existing metadata
            try:
                self._metadata = pd.read_csv(metadata_path)
                logger.info(f"Loaded metadata for {len(self._metadata)} samples")
                
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self._create_metadata_from_structure()
        
        # Ensure required columns exist
        required_columns = ["filename", "class", "split"]
        missing_columns = [col for col in required_columns if col not in self._metadata.columns]
        
        if missing_columns:
            logger.warning(f"Missing metadata columns: {missing_columns}")
            self._create_metadata_from_structure()
    
    def _create_metadata_from_structure(self) -> None:
        """Create metadata from directory structure."""
        logger.info("Creating metadata from directory structure")
        
        audio_dir = self.data_dir / "audio"
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
        metadata_rows = []
        
        # Scan audio files
        for audio_file in audio_dir.rglob("*.wav"):
            # Extract class from filename or directory structure
            # This is dataset-specific and may need adjustment
            relative_path = audio_file.relative_to(audio_dir)
            
            # Try to infer class from directory structure or filename
            class_name = self._infer_class_from_path(relative_path)
            
            if class_name in MAD_CLASSES:
                metadata_rows.append({
                    "filename": str(relative_path),
                    "class": class_name,
                    "class_id": MAD_CLASSES[class_name],
                    "split": None  # Will be assigned later
                })
        
        if not metadata_rows:
            raise ValueError("No valid audio files found")
        
        self._metadata = pd.DataFrame(metadata_rows)
        
        # Save metadata
        metadata_path = self.data_dir / "metadata.csv"
        self._metadata.to_csv(metadata_path, index=False)
        logger.info(f"Created metadata for {len(metadata_rows)} samples")
    
    def _infer_class_from_path(self, file_path: Path) -> str:
        """
        Infer class label from file path.
        
        Args:
            file_path: Relative path to audio file
            
        Returns:
            Inferred class name
        """
        path_str = str(file_path).lower()
        
        # Check for class keywords in path
        for class_name in MAD_CLASSES.keys():
            class_keywords = class_name.lower().split()
            if any(keyword in path_str for keyword in class_keywords):
                return class_name
        
        # Additional keyword mapping
        keyword_mapping = {
            "heli": "Helicopter",
            "rotor": "Helicopter", 
            "aircraft": "Fighter Aircraft",
            "jet": "Fighter Aircraft",
            "plane": "Fighter Aircraft",
            "vehicle": "Military Vehicle",
            "tank": "Military Vehicle",
            "truck": "Truck",
            "foot": "Footsteps",
            "step": "Footsteps",
            "walk": "Footsteps",
            "speech": "Speech",
            "voice": "Speech",
            "talk": "Speech",
            "background": "Background",
            "noise": "Background",
            "ambient": "Background"
        }
        
        for keyword, class_name in keyword_mapping.items():
            if keyword in path_str:
                return class_name
        
        # Default to background if no class detected
        logger.warning(f"Could not infer class for {file_path}, defaulting to Background")
        return "Background"
    
    def _create_splits(self) -> None:
        """Create train/validation/test splits."""
        
        # Check if splits already exist in metadata
        if "split" in self._metadata.columns and not self._metadata["split"].isna().all():
            logger.info("Using existing splits from metadata")
        else:
            logger.info("Creating new stratified splits")
            self._assign_splits()
        
        # Filter metadata for current split
        if self.split == "all":
            split_metadata = self._metadata
        else:
            split_metadata = self._metadata[self._metadata["split"] == self.split]
        
        if len(split_metadata) == 0:
            raise ValueError(f"No samples found for split: {self.split}")
        
        # Create file list and labels
        self._file_list = []
        self._labels = []
        
        for _, row in split_metadata.iterrows():
            audio_path = self.data_dir / "audio" / row["filename"]
            
            # Check if file exists
            if audio_path.exists():
                self._file_list.append(audio_path)
                self._labels.append(row["class_id"] if "class_id" in row else MAD_CLASSES[row["class"]])
            else:
                logger.warning(f"Audio file not found: {audio_path}")
        
        logger.info(f"Split '{self.split}' contains {len(self._file_list)} samples")
    
    def _assign_splits(self) -> None:
        """Assign train/validation/test splits using stratified sampling."""
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using random splits")
            self._assign_random_splits()
            return
        
        # Get class labels for stratification
        if "class_id" not in self._metadata.columns:
            self._metadata["class_id"] = self._metadata["class"].map(MAD_CLASSES)
        
        labels = self._metadata["class_id"].values
        indices = np.arange(len(self._metadata))
        
        # Split ratios
        train_ratio = self.split_ratios["train"]
        val_ratio = self.split_ratios["val"]
        test_ratio = self.split_ratios["test"]
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Split ratios sum to {total_ratio}, normalizing")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        # First split: train vs (val + test)
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            indices, labels,
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=self.random_state
        )
        
        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_size = val_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=(1 - val_size),
                stratify=temp_labels,
                random_state=self.random_state
            )
        elif val_ratio > 0:
            val_indices = temp_indices
            test_indices = []
        else:
            val_indices = []
            test_indices = temp_indices
        
        # Assign splits
        self._metadata["split"] = None
        self._metadata.loc[train_indices, "split"] = "train"
        if len(val_indices) > 0:
            self._metadata.loc[val_indices, "split"] = "val"
        if len(test_indices) > 0:
            self._metadata.loc[test_indices, "split"] = "test"
        
        # Save updated metadata
        metadata_path = self.data_dir / "metadata.csv"
        self._metadata.to_csv(metadata_path, index=False)
        
        # Log split statistics
        split_counts = self._metadata["split"].value_counts()
        logger.info(f"Split assignment complete: {split_counts.to_dict()}")
    
    def _assign_random_splits(self) -> None:
        """Assign splits randomly when stratification is not available."""
        np.random.seed(self.random_state)
        
        n_samples = len(self._metadata)
        indices = np.random.permutation(n_samples)
        
        # Calculate split boundaries
        train_end = int(n_samples * self.split_ratios["train"])
        val_end = train_end + int(n_samples * self.split_ratios["val"])
        
        # Assign splits
        self._metadata["split"] = None
        self._metadata.iloc[indices[:train_end], self._metadata.columns.get_loc("split")] = "train"
        self._metadata.iloc[indices[train_end:val_end], self._metadata.columns.get_loc("split")] = "val"
        self._metadata.iloc[indices[val_end:], self._metadata.columns.get_loc("split")] = "test"
        
        # Save metadata
        metadata_path = self.data_dir / "metadata.csv"
        self._metadata.to_csv(metadata_path, index=False)
    
    def _validate_dataset(self) -> None:
        """Validate dataset integrity."""
        
        # Check file existence
        missing_files = []
        for file_path in self._file_list:
            if not file_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing audio files")
            # Remove missing files
            valid_indices = [i for i, path in enumerate(self._file_list) if path.exists()]
            self._file_list = [self._file_list[i] for i in valid_indices]
            self._labels = [self._labels[i] for i in valid_indices]
        
        # Check class distribution
        class_counts = Counter(self._labels)
        logger.info(f"Class distribution: {class_counts}")
        
        # Warn about class imbalance
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 5:
            logger.warning(f"High class imbalance detected (ratio: {imbalance_ratio:.2f})")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._file_list)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (audio_tensor, label)
        """
        # Get file path and label
        file_path = self._file_list[index]
        label = self._labels[index]
        
        # Load audio
        audio = self._load_audio(file_path, index)
        
        # Apply transforms
        if self.transform is not None:
            audio = self.transform(audio)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return audio, label
    
    def _load_audio(self, file_path: Path, index: int) -> torch.Tensor:
        """
        Load audio file with caching support.
        
        Args:
            file_path: Path to audio file
            index: Sample index for caching
            
        Returns:
            Audio tensor
        """
        # Check cache first
        if self.cache_audio and index in self._audio_cache:
            return self._audio_cache[index]
        
        try:
            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(str(file_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Adjust duration if specified
            if self.duration is not None:
                target_length = int(self.duration * self.sample_rate)
                current_length = waveform.shape[1]
                
                if current_length < target_length:
                    # Pad with zeros
                    padding = target_length - current_length
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                elif current_length > target_length:
                    # Crop (center crop for deterministic behavior)
                    start = (current_length - target_length) // 2
                    waveform = waveform[:, start:start + target_length]
            
            # Normalize if requested
            if self.normalize:
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # Cache if enabled
            if self.cache_audio:
                self._audio_cache[index] = waveform
            
            return waveform
            
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            # Return silent audio as fallback
            duration = self.duration or 1.0
            return torch.zeros(1, int(duration * self.sample_rate))
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for loss balancing.
        
        Returns:
            Tensor of class weights
        """
        class_counts = Counter(self._labels)
        total_samples = len(self._labels)
        num_classes = len(MAD_CLASSES)
        
        # Calculate inverse frequency weights
        weights = []
        for class_id in range(num_classes):
            count = class_counts.get(class_id, 1)  # Avoid division by zero
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution for current split.
        
        Returns:
            Dictionary mapping class names to counts
        """
        class_counts = Counter(self._labels)
        return {MAD_CLASS_NAMES[class_id]: count 
                for class_id, count in class_counts.items()}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary containing dataset statistics
        """
        class_distribution = self.get_class_distribution()
        
        return {
            "name": "Military Audio Detection (MAD)",
            "split": self.split,
            "num_samples": len(self),
            "num_classes": len(MAD_CLASSES),
            "class_names": list(MAD_CLASSES.keys()),
            "class_distribution": class_distribution,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "data_dir": str(self.data_dir),
            "config": self.config
        }
    
    def export_split_info(self, output_path: str) -> None:
        """
        Export split information to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        split_info = {
            "dataset": "MAD",
            "split": self.split,
            "samples": []
        }
        
        for i, (file_path, label) in enumerate(zip(self._file_list, self._labels)):
            split_info["samples"].append({
                "index": i,
                "filename": str(file_path.relative_to(self.data_dir)),
                "class_id": label,
                "class_name": MAD_CLASS_NAMES[label]
            })
        
        with open(output_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Split info exported to {output_path}")


class MADDataLoader:
    """
    Factory class for creating MAD data loaders with different configurations.
    """
    
    @staticmethod
    def create_dataloader(dataset: MADDataset,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         pin_memory: bool = True,
                         drop_last: bool = True,
                         collate_fn: Optional[Callable] = None) -> torch.utils.data.DataLoader:
        """
        Create a data loader for MAD dataset.
        
        Args:
            dataset: MAD dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            collate_fn: Optional custom collate function
            
        Returns:
            Configured data loader
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
    
    @staticmethod
    def create_train_loader(data_dir: str, 
                           config: Optional[Dict[str, Any]] = None,
                           **loader_kwargs) -> torch.utils.data.DataLoader:
        """Create training data loader."""
        dataset = MADDataset(data_dir, split="train", config=config)
        return MADDataLoader.create_dataloader(
            dataset, shuffle=True, **loader_kwargs
        )
    
    @staticmethod
    def create_val_loader(data_dir: str,
                         config: Optional[Dict[str, Any]] = None,
                         **loader_kwargs) -> torch.utils.data.DataLoader:
        """Create validation data loader."""
        dataset = MADDataset(data_dir, split="val", config=config)
        return MADDataLoader.create_dataloader(
            dataset, shuffle=False, **loader_kwargs
        )
    
    @staticmethod
    def create_test_loader(data_dir: str,
                          config: Optional[Dict[str, Any]] = None,
                          **loader_kwargs) -> torch.utils.data.DataLoader:
        """Create test data loader."""
        dataset = MADDataset(data_dir, split="test", config=config)
        return MADDataLoader.create_dataloader(
            dataset, shuffle=False, **loader_kwargs
        )


# Convenience functions
def load_mad_dataset(data_dir: str, split: str = "train", 
                    **kwargs) -> MADDataset:
    """
    Convenience function to load MAD dataset.
    
    Args:
        data_dir: Path to MAD dataset
        split: Dataset split
        **kwargs: Additional arguments for MADDataset
        
    Returns:
        MAD dataset instance
    """
    return MADDataset(data_dir, split=split, **kwargs)


def create_mad_loaders(data_dir: str,
                      config: Optional[Dict[str, Any]] = None,
                      batch_size: int = 32,
                      num_workers: int = 4) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test data loaders for MAD dataset.
    
    Args:
        data_dir: Path to MAD dataset
        config: Dataset configuration
        batch_size: Batch size for all loaders
        num_workers: Number of workers for all loaders
        
    Returns:
        Dictionary containing train/val/test data loaders
    """
    loaders = {}
    
    for split in ["train", "val", "test"]:
        try:
            dataset = MADDataset(data_dir, split=split, config=config)
            
            if len(dataset) > 0:
                shuffle = (split == "train")
                loaders[split] = MADDataLoader.create_dataloader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers
                )
            else:
                logger.warning(f"Empty dataset for split: {split}")
                
        except Exception as e:
            logger.error(f"Failed to create loader for split {split}: {e}")
    
    return loaders


def analyze_mad_dataset(data_dir: str) -> Dict[str, Any]:
    """
    Analyze MAD dataset and return comprehensive statistics.
    
    Args:
        data_dir: Path to MAD dataset
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Load full dataset
        dataset = MADDataset(data_dir, split="all")
        
        analysis = {
            "total_samples": len(dataset),
            "class_distribution": dataset.get_class_distribution(),
            "split_info": {}
        }
        
        # Analyze each split
        for split in ["train", "val", "test"]:
            try:
                split_dataset = MADDataset(data_dir, split=split)
                analysis["split_info"][split] = {
                    "num_samples": len(split_dataset),
                    "class_distribution": split_dataset.get_class_distribution()
                }
            except Exception as e:
                logger.warning(f"Could not analyze split {split}: {e}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        return {"error": str(e)}