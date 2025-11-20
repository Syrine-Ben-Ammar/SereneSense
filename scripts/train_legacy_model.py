#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Train Legacy CNN/CRNN Models
=============================
Script to train legacy MFCC-based CNN or CRNN models for comparison purposes.

Usage:
    python scripts/train_legacy_model.py --model cnn --epochs 150 --batch-size 32
    python scripts/train_legacy_model.py --model crnn --epochs 300 --batch-size 16
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.models.legacy import CNNMFCCModel, CRNNMFCCModel, LegacyModelType, LegacyModelConfig
from core.models.legacy.legacy_config import CNNConfig, CRNNConfig, MFCCConfig, SpecAugmentConfig
from core.data.preprocessing.legacy_mfcc import LegacyMFCCPreprocessor
from core.data.augmentation.legacy_specaugment import LegacySpecAugmentTransform

logger = logging.getLogger(__name__)
DEFAULT_NUM_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))
DEFAULT_MAX_PENDING_BATCHES = 8


def _normalize_label_value(raw_label: Any) -> str:
    """Convert raw label values (bytes/np scalars) to a normalized string."""
    if isinstance(raw_label, (bytes, np.bytes_)):
        return raw_label.decode("utf-8").strip()
    if isinstance(raw_label, np.generic):
        return str(raw_label.item()).strip()
    return str(raw_label).strip()


def build_label_mapping(
    h5_paths: Iterable[Path], label_key: str = "labels"
) -> Dict[str, int]:
    """
    Build a consistent label-to-index mapping across multiple HDF5 files.

    Args:
        h5_paths: Iterable of HDF5 file paths.
        label_key: Dataset name that stores labels.

    Returns:
        Mapping from label string to contiguous integer id.
    """
    label_values = []
    for path in h5_paths:
        with h5py.File(path, "r") as h5_file:
            if label_key not in h5_file:
                raise KeyError(f"Missing '{label_key}' dataset in {path}")
            raw_labels = h5_file[label_key][:]
            label_values.extend(_normalize_label_value(value) for value in raw_labels)

    if not label_values:
        raise ValueError("No labels found while building label mapping.")

    unique_labels = sorted(set(label_values))
    return {label: idx for idx, label in enumerate(unique_labels)}

def load_split_metadata(h5_path: Path) -> Optional[Dict[str, Any]]:
    """Load metadata.json next to an HDF5 split if it exists."""
    meta_path = h5_path.parent / "metadata.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_class_weights(label_counts: Dict[str, int], label_mapping: Dict[str, int]) -> np.ndarray:
    """Compute inverse-frequency class weights aligned with label_mapping indices."""
    total = sum(label_counts.values())
    num_classes = len(label_mapping)
    if total == 0 or num_classes == 0:
        raise ValueError("Cannot compute class weights with empty counts.")
    weights = np.ones(num_classes, dtype=np.float32)
    for label, count in label_counts.items():
        index = label_mapping.get(str(label))
        if index is None or count <= 0:
            continue
        weights[index] = total / (num_classes * count)
    return weights


class H5pyDataset(Dataset):
    """Dataset wrapper for SereneSense MAD HDF5 files with optional MFCC conversion."""

    def __init__(
        self,
        file_path: Union[Path, str],
        *,
        expected_shape: Optional[Tuple[int, int, int]] = None,
        preprocessor: Optional[LegacyMFCCPreprocessor] = None,
        feature_key: Optional[str] = None,
        label_key: Optional[str] = None,
        label_mapping: Optional[Dict[str, int]] = None,
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file does not exist: {self.file_path}")

        self.expected_shape = expected_shape
        self.preprocessor = preprocessor
        self.label_mapping = label_mapping
        self._preferred_feature_key = feature_key
        self._preferred_label_key = label_key

        metadata = self._inspect_file()
        self.feature_key = metadata["feature_key"]
        self.label_key = metadata["label_key"]
        self.length = metadata["length"]
        self.feature_sample_shape = metadata["feature_shape"]
        self.features_are_audio = (
            len(self.feature_sample_shape) == 1 or self.feature_key == "audio"
        )

        if self.features_are_audio and self.preprocessor is None:
            raise ValueError(
                "Audio waveforms detected but no LegacyMFCCPreprocessor was provided. "
                "Pass a preprocessor to convert waveforms to MFCC tensors."
            )

        self._file: Optional[h5py.File] = None
        self._feature_dataset = None
        self._label_dataset = None

    def _inspect_file(self) -> Dict[str, Any]:
        with h5py.File(self.file_path, "r") as h5_file:
            feature_key = self._resolve_dataset_key(
                h5_file,
                self._preferred_feature_key,
                fallback=("features", "mfcc", "audio"),
            )
            label_key = self._resolve_dataset_key(
                h5_file, self._preferred_label_key, fallback=("labels",)
            )

            feature_len = h5_file[feature_key].shape[0]
            label_len = h5_file[label_key].shape[0]
            if feature_len != label_len:
                raise ValueError(
                    f"Feature/label length mismatch in {self.file_path.name}: "
                    f"{feature_len} vs {label_len}"
                )

            feature_shape = h5_file[feature_key].shape[1:]

        return {
            "feature_key": feature_key,
            "label_key": label_key,
            "length": feature_len,
            "feature_shape": feature_shape,
        }

    @staticmethod
    def _resolve_dataset_key(
        h5_file: h5py.File,
        preferred_key: Optional[str],
        fallback: Tuple[str, ...],
    ) -> str:
        candidates = []
        if preferred_key:
            candidates.append(preferred_key)
        for key in fallback:
            if key not in candidates:
                candidates.append(key)

        for key in candidates:
            if key in h5_file:
                return key

        raise KeyError(
            f"None of the candidate datasets {candidates} were found in {h5_file.filename}."
        )

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
            self._feature_dataset = self._file[self.feature_key]
            self._label_dataset = self._file[self.label_key]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        feature = self._feature_dataset[index]
        label = self._label_dataset[index]

        feature_tensor = self._prepare_feature(feature)
        label_tensor = torch.tensor(self._encode_label(label), dtype=torch.long)
        return feature_tensor, label_tensor

    def _prepare_feature(self, feature: np.ndarray) -> torch.Tensor:
        if self.features_are_audio:
            feature_array = self.preprocessor.process_audio(feature)
            feature_tensor = torch.as_tensor(feature_array, dtype=torch.float32)
            feature_tensor = feature_tensor.permute(2, 0, 1)  # channels, freq, time
        else:
            feature_tensor = torch.as_tensor(feature, dtype=torch.float32)
            feature_tensor = self._ensure_channel_first(feature_tensor)

        if self.expected_shape is not None:
            expected_channels, expected_freq, expected_time = self.expected_shape
            if feature_tensor.shape != (expected_channels, expected_freq, expected_time):
                raise ValueError(
                    f"Feature at {self.file_path.name} index out of expected shape "
                    f"{self.expected_shape}, got {tuple(feature_tensor.shape)}"
                )
        return feature_tensor

    def _ensure_channel_first(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError(f"Unsupported feature rank: {tensor.ndim}")

        if self.expected_shape is None:
            return tensor

        channels, freq, time = self.expected_shape
        shape = tuple(tensor.shape)
        if shape == (channels, freq, time):
            return tensor
        if shape == (freq, time, channels):
            return tensor.permute(2, 0, 1)
        if shape == (freq, channels, time):
            return tensor.permute(1, 0, 2)
        if shape == (channels, time, freq):
            return tensor.permute(0, 2, 1)
        if shape == (time, freq, channels):
            return tensor.permute(2, 1, 0)
        if shape == (time, channels, freq):
            return tensor.permute(1, 2, 0)

        raise ValueError(
            f"Unable to reconcile feature shape {shape} with expected {self.expected_shape}"
        )

    def _encode_label(self, raw_label: Any) -> int:
        normalized = _normalize_label_value(raw_label)
        if self.label_mapping is not None:
            if normalized not in self.label_mapping:
                raise KeyError(
                    f"Label '{normalized}' missing from provided label mapping."
                )
            return self.label_mapping[normalized]

        if normalized.isdigit():
            return int(normalized)

        try:
            return int(float(normalized))
        except ValueError as exc:
            raise ValueError(
                "Encountered non-numeric label but no mapping was provided."
            ) from exc

    def __del__(self):
        if self._file is not None:
            self._file.close()


class LegacyModelTrainer:
    """Trainer for legacy models."""

    def __init__(
        self,
        model: nn.Module,
        config: LegacyModelConfig,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        augment_fn: Optional[callable] = None,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            augment_fn: Optional augmentation function

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Apply augmentation
            if augment_fn is not None:
                features = augment_fn(features)

            # Forward pass
            with autocast(enabled=self.use_amp):
                logits = self.model(features)
                if self.use_amp:
                    logits = logits.float()
                loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Evaluate model.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (accuracy, average_loss)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Evaluating", leave=False)
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    logits = self.model(features)
                    if self.use_amp:
                        logits = logits.float()
                    loss = criterion(logits, labels)

                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        return accuracy, avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 150,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        augment_fn: Optional[callable] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            augment_fn: Optional augmentation function
            class_weights: Optional tensor of per-class weights for CrossEntropyLoss

        Returns:
            Dictionary with training history
        """
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        best_accuracy = 0.0
        best_epoch = 0
        best_model_state = None

        print(f"Starting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion, augment_fn)

            # Evaluate
            val_accuracy, val_loss = self.evaluate(val_loader, criterion)

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])

            # Log
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

            # Scheduler
            scheduler.step(val_accuracy)

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                best_model_state = self.model.state_dict().copy()

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with accuracy: {best_accuracy:.4f}")

        history["best_accuracy"] = best_accuracy
        history["best_epoch"] = best_epoch
        return history


def create_dummy_dataset(
    num_samples: int = 100, num_classes: int = 7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dummy dataset for testing."""
    # CNN: (batch, channels, freq, time) = (batch, 3, 40, 92)
    features = torch.randn(num_samples, 3, 40, 92)
    labels = torch.randint(0, num_classes, (num_samples,))
    return features, labels


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train legacy CNN/CRNN models")
    parser.add_argument(
        "--model", type=str, choices=["cnn", "crnn"], default="cnn", help="Model type (cnn or crnn)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Save checkpoint path")
    parser.add_argument("--test-run", action="store_true", help="Run with dummy data")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader workers (0 disables multiprocessing)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches prefetched per worker (requires num-workers > 0)",
    )
    parser.add_argument(
        "--max-pending-batches",
        type=int,
        default=DEFAULT_MAX_PENDING_BATCHES,
        help=(
            "Upper bound on the total number of batches that can sit in the DataLoader "
            "queue simultaneously (num_workers * prefetch_factor)."
        ),
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help=(
            "Keep DataLoader workers alive between epochs. Enable only if you have "
            "enough RAM to hold multiple prefetched batches."
        ),
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable CUDA pinned-memory staging for faster host-to-device transfers",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=None,
        help=(
            "Optional path to save per-epoch training history (JSON). "
            "Defaults to outputs/history/{model}_{timestamp}.json."
        ),
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(__file__).parent.parent / "data" / "processed" / "mad"
    train_file: Optional[Path] = None
    val_file: Optional[Path] = None
    label_mapping: Optional[Dict[str, int]] = None
    train_metadata: Optional[Dict[str, Any]] = None
    val_metadata: Optional[Dict[str, Any]] = None

    preprocessor_kwargs: Dict[str, Any] = {}
    if args.test_run:
        logger.info("Running with dummy data (test mode)...")
    else:
        logger.info("Loading real dataset from MAD files...")
        train_file = data_dir / "train" / "train.h5"
        val_file = data_dir / "validation" / "validation.h5"
        train_metadata = load_split_metadata(train_file)
        val_metadata = load_split_metadata(val_file)

        missing_files = [path for path in (train_file, val_file) if not path.exists()]
        if missing_files:
            missing_str = ", ".join(str(path) for path in missing_files)
            raise FileNotFoundError(f"Missing MAD dataset file(s): {missing_str}")

        label_mapping = build_label_mapping([train_file, val_file])
        expected_classes = None
        for metadata in (train_metadata, val_metadata):
            if metadata and metadata.get("num_classes"):
                expected_classes = metadata["num_classes"]
                break
        if expected_classes is not None and len(label_mapping) != expected_classes:
            raise ValueError(
                f"Discovered {len(label_mapping)} labels in HDF5 but metadata reports "
                f"{expected_classes}. Please regenerate data via scripts/prepare_data.py."
            )
        logger.info(f"Detected {len(label_mapping)} unique labels in MAD dataset.")

    # Model configuration
    if args.model == "cnn":
        config = LegacyModelConfig(
            model_type=LegacyModelType.CNN,
            device=args.device,
        )
        epochs = args.epochs or 150
        batch_size = args.batch_size or 32
    else:
        config = LegacyModelConfig(
            model_type=LegacyModelType.CRNN,
            device=args.device,
        )
        epochs = args.epochs or 300
        batch_size = args.batch_size or 16

    if label_mapping is not None:
        num_classes = len(label_mapping)
        config.cnn.num_classes = num_classes
        config.crnn.num_classes = num_classes

    if args.model == "cnn":
        model = CNNMFCCModel(config)
    else:
        model = CRNNMFCCModel(config)

    mfcc_duration = (
        config.mfcc.target_duration if args.model == "cnn" else config.mfcc.crnn_duration
    )

    logger.info(f"Model: {model.model_name}")
    logger.info(f"Parameters: {model.count_parameters():,}")

    class_weights_tensor: Optional[torch.Tensor] = None
    if args.test_run:
        features_train, labels_train = create_dummy_dataset(100)
        features_val, labels_val = create_dummy_dataset(20)
        train_dataset = TensorDataset(features_train, labels_train)
        val_dataset = TensorDataset(features_val, labels_val)
    else:
        assert train_file is not None and val_file is not None
        assert label_mapping is not None

        preprocessor_kwargs = dict(
            sample_rate=config.mfcc.sample_rate,
            duration=mfcc_duration,
            n_mfcc=config.mfcc.n_mfcc,
            n_mels=config.mfcc.n_mels,
            n_fft=config.mfcc.n_fft,
            hop_length=config.mfcc.hop_length,
            use_deltas=config.mfcc.use_deltas,
            use_delta_deltas=config.mfcc.use_delta_deltas,
            normalize=config.mfcc.normalize,
        )

        train_preprocessor = LegacyMFCCPreprocessor(**preprocessor_kwargs)
        val_preprocessor = LegacyMFCCPreprocessor(**preprocessor_kwargs)
        preprocessor_output_shape = train_preprocessor.output_shape  # (freq, time, channels)
        dataset_expected_shape = (
            preprocessor_output_shape[2],
            preprocessor_output_shape[0],
            preprocessor_output_shape[1],
        )

        train_dataset = H5pyDataset(
            train_file,
            expected_shape=dataset_expected_shape,
            preprocessor=train_preprocessor,
            label_mapping=label_mapping,
        )
        val_dataset = H5pyDataset(
            val_file,
            expected_shape=dataset_expected_shape,
            preprocessor=val_preprocessor,
            label_mapping=label_mapping,
        )
        if train_metadata and train_metadata.get("label_counts"):
            weights_np = compute_class_weights(train_metadata["label_counts"], label_mapping)
            class_weights_tensor = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
            logger.info("Using class weights derived from training split metadata.")
        else:
            logger.warning(
                "Training metadata missing label_counts. Proceeding without class weighting."
            )

    # Create data loaders
    num_workers = max(0, args.num_workers)
    pin_memory = device.type == "cuda" and not args.no_pin_memory
    persistent_workers = bool(args.persistent_workers and num_workers > 0)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        requested_prefetch = max(1, args.prefetch_factor)
        if args.max_pending_batches and args.max_pending_batches > 0:
            pending_budget = max(args.max_pending_batches, 1)
            max_prefetch_from_budget = max(1, pending_budget // num_workers)
            prefetch_factor = min(requested_prefetch, max_prefetch_from_budget)
        else:
            prefetch_factor = requested_prefetch
        loader_kwargs["prefetch_factor"] = prefetch_factor

    logger.info(
        "DataLoader config | batch=%d | workers=%d | prefetch_factor=%s | "
        "persistent_workers=%s | pin_memory=%s",
        batch_size,
        num_workers,
        loader_kwargs.get("prefetch_factor", "n/a"),
        persistent_workers,
        pin_memory,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Create augmentation (for training data)
    if args.model == "cnn":
        augment_fn = LegacySpecAugmentTransform(
            freq_mask_param=15,
            num_freq_masks=2,
            time_mask_param=10,
            num_time_masks=2,
            apply_prob=0.8,
            training=True,
        )
    else:
        augment_fn = LegacySpecAugmentTransform(
            freq_mask_param=15,
            num_freq_masks=2,
            time_mask_param=10,
            num_time_masks=2,
            apply_prob=0.8,
            training=True,
        )

    # Train
    trainer = LegacyModelTrainer(model, config, device)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=args.learning_rate,
        augment_fn=augment_fn,
        class_weights=class_weights_tensor,
    )
    history_path = None
    default_history_dir = Path("outputs") / "history"
    if args.history_path:
        history_path = Path(args.history_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        history_path = default_history_dir / f"{args.model}_{timestamp}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_summary = {
        "model": model.model_name,
        "model_type": args.model,
        "epochs_requested": epochs,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_accuracy": history["val_accuracy"],
        "learning_rate": history["learning_rate"],
        "best_accuracy": history.get("best_accuracy"),
        "best_epoch": history.get("best_epoch"),
        "dataset": {
            "train_file": str(train_file) if not args.test_run else "dummy",
            "val_file": str(val_file) if not args.test_run else "dummy",
            "num_classes": len(label_mapping) if label_mapping else None,
        },
        "config": {
            "mfcc": preprocessor_kwargs if not args.test_run else {},
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": loader_kwargs.get("prefetch_factor"),
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
            "learning_rate": args.learning_rate,
        },
    }
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history_summary, handle, indent=2)
    logger.info(f"Training history saved to {history_path}")

    # Save checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(
            checkpoint_path,
            metrics={"best_accuracy": history["best_accuracy"], "best_epoch": history.get("best_epoch", 0)},
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {history['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
