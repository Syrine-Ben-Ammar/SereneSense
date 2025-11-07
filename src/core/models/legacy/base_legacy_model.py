# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Base Legacy Model Class
=======================
Abstract base class for legacy CNN/CRNN models with common functionality.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class LegacyModelOutput:
    """Standardized output structure for legacy models."""

    logits: torch.Tensor  # Raw model outputs
    predictions: torch.Tensor  # Predicted class indices
    probabilities: torch.Tensor  # Softmax probabilities
    features: Optional[torch.Tensor] = None  # Extracted features (pre-classification)


class BaseLegacyModel(nn.Module, ABC):
    """
    Abstract base class for legacy models (CNN/CRNN).

    Provides common functionality for:
    - Model configuration and initialization
    - Checkpoint saving/loading
    - Feature extraction
    - Inference
    """

    def __init__(self, config: Any, model_name: str = "legacy_model"):
        """
        Initialize base legacy model.

        Args:
            config: LegacyModelConfig dataclass instance
            model_name: Name identifier for the model
        """
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.device = torch.device(config.device)

    @abstractmethod
    def _build_architecture(self) -> None:
        """Build the model architecture. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, LegacyModelOutput]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, freq, time)
            return_features: Whether to return intermediate features

        Returns:
            Logits or LegacyModelOutput depending on return_features
        """
        pass

    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extraction part of the model (before classification head).
        Useful for transfer learning or feature analysis.

        Returns:
            nn.Module representing the feature extractor
        """
        # Default: return everything except the last layer
        # Subclasses can override for more specific behavior
        if hasattr(self, 'features') and hasattr(self, 'classifier'):
            return self.features
        return self

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get softmax probabilities for input.

        Args:
            x: Input tensor

        Returns:
            Probabilities tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            if isinstance(logits, LegacyModelOutput):
                return logits.probabilities
            return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.

        Args:
            x: Input tensor

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=-1)

    def save_checkpoint(
        self, checkpoint_path: Union[str, Path], optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0, metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            epoch: Current epoch number
            metrics: Optional dictionary of metrics to save
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'config': self.config,
            'epoch': epoch,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(
        self, checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce state dict matching

        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

        # Return metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'model_name': checkpoint.get('model_name', 'unknown'),
        }
        return metadata

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary information.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            'non_trainable_parameters': sum(
                p.numel() for p in self.parameters() if not p.requires_grad
            ),
            'device': str(self.device),
            'config': str(self.config),
        }

    def to_device(self, device: Union[str, torch.device]) -> 'BaseLegacyModel':
        """
        Move model to device.

        Args:
            device: Device to move to ('cpu', 'cuda', etc.)

        Returns:
            Self for chaining
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        return self.to(self.device)

    def freeze_backbone(self) -> None:
        """Freeze all parameters except classification head."""
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze classification head
        if hasattr(self, 'classifier'):
            for param in self.classifier.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.eval()
        torch.set_grad_enabled(False)

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.train()
        torch.set_grad_enabled(True)
