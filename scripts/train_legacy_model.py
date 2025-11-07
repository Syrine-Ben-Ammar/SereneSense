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
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.models.legacy import CNNMFCCModel, CRNNMFCCModel, LegacyModelType, LegacyModelConfig
from core.models.legacy.legacy_config import CNNConfig, CRNNConfig, MFCCConfig, SpecAugmentConfig
from core.data.preprocessing.legacy_mfcc import LegacyMFCCPreprocessor
from core.data.augmentation.legacy_specaugment import LegacySpecAugmentTransform

logger = logging.getLogger(__name__)


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
            logits = self.model(features)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
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

                logits = self.model(features)
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

        Returns:
            Dictionary with training history
        """
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        best_accuracy = 0.0
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
                best_model_state = self.model.state_dict().copy()

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with accuracy: {best_accuracy:.4f}")

        history["best_accuracy"] = best_accuracy
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
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Save checkpoint path")
    parser.add_argument("--test-run", action="store_true", help="Run with dummy data")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model configuration
    if args.model == "cnn":
        config = LegacyModelConfig(
            model_type=LegacyModelType.CNN,
            device=args.device,
        )
        model = CNNMFCCModel(config)
        epochs = args.epochs or 150
        batch_size = args.batch_size or 32
    else:
        config = LegacyModelConfig(
            model_type=LegacyModelType.CRNN,
            device=args.device,
        )
        model = CRNNMFCCModel(config)
        epochs = args.epochs or 300
        batch_size = args.batch_size or 16

    logger.info(f"Model: {model.model_name}")
    logger.info(f"Parameters: {model.count_parameters():,}")

    # Create dummy dataset for testing
    if args.test_run:
        logger.info("Running with dummy data (test mode)...")
        features_train, labels_train = create_dummy_dataset(100)
        features_val, labels_val = create_dummy_dataset(20)
    else:
        logger.info("Loading real dataset... (not implemented in this script)")
        raise NotImplementedError("Load real dataset from MAD or other source")

    # Create data loaders
    train_dataset = TensorDataset(features_train, labels_train)
    val_dataset = TensorDataset(features_val, labels_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
    )

    # Save checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(checkpoint_path, metrics={"best_accuracy": history["best_accuracy"]})
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {history['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
