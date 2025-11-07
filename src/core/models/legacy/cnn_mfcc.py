# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
CNN MFCC Model
==============
2D-CNN model trained on MFCC features for audio classification.
Architecture from the legacy notebooks: train_mad_mfcc_gpu_v2.ipynb

Model Size: ~242K parameters
Input: MFCC + delta + delta-delta features (40, 92, 3)
Output: 7-class probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import logging

from .base_legacy_model import BaseLegacyModel, LegacyModelOutput
from .legacy_config import LegacyModelConfig, CNNConfig

logger = logging.getLogger(__name__)


class CNNMFCCModel(BaseLegacyModel):
    """
    2D-CNN architecture for MFCC-based audio classification.

    Architecture:
        Input (40, 92, 3) [MFCC, time, channels]
        ↓
        Conv2D(48, 3×3) + BatchNorm + MaxPool(2,2) + Dropout(0.25)
        ↓
        Conv2D(96, 3×3) + BatchNorm + MaxPool(2,2) + Dropout(0.30)
        ↓
        Conv2D(192, 3×3) + BatchNorm + MaxPool(2,2) + Dropout(0.30)
        ↓
        GlobalAveragePooling2D
        ↓
        Dense(160) + Dropout(0.35)
        ↓
        Dense(7) + Softmax

    Hyperparameters from original notebook:
        - MFCC coefficients: 40
        - Time steps: 92 (3 seconds at 512 hop length)
        - Channels: 3 (MFCC, delta, delta-delta)
        - Optimizer: Adam (lr=1e-3)
        - Epochs: 150
        - Batch size: 32
        - Loss: Cross-entropy with class weights
    """

    def __init__(self, config: LegacyModelConfig):
        """
        Initialize CNN MFCC model.

        Args:
            config: LegacyModelConfig instance
        """
        super().__init__(config, model_name="cnn_mfcc")
        self.config = config
        self.arch_config = config.cnn

        # Build architecture
        self._build_architecture()

        # Log model summary
        logger.info(f"Initialized {self.model_name}")
        logger.info(f"Total parameters: {self.count_parameters():,}")

    def _build_architecture(self) -> None:
        """Build the CNN architecture."""
        arch = self.arch_config

        # Feature extraction blocks
        self.features = nn.Sequential()

        # Block 1: Conv → BatchNorm → MaxPool → Dropout
        self.features.append(
            nn.Conv2d(
                in_channels=arch.input_shape[2],  # 3 (MFCC channels)
                out_channels=arch.conv_filters[0],  # 48
                kernel_size=arch.conv_kernel_size,  # (3, 3)
                padding=1,
            )
        )
        if arch.batch_norm:
            self.features.append(nn.BatchNorm2d(arch.conv_filters[0]))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(
            nn.MaxPool2d(kernel_size=arch.pool_size, stride=arch.pool_strides)
        )
        self.features.append(nn.Dropout(arch.dropout_rates[0]))

        # Block 2: Conv → BatchNorm → MaxPool → Dropout
        self.features.append(
            nn.Conv2d(
                in_channels=arch.conv_filters[0],  # 48
                out_channels=arch.conv_filters[1],  # 96
                kernel_size=arch.conv_kernel_size,
                padding=1,
            )
        )
        if arch.batch_norm:
            self.features.append(nn.BatchNorm2d(arch.conv_filters[1]))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(
            nn.MaxPool2d(kernel_size=arch.pool_size, stride=arch.pool_strides)
        )
        self.features.append(nn.Dropout(arch.dropout_rates[1]))

        # Block 3: Conv → BatchNorm → MaxPool → Dropout
        self.features.append(
            nn.Conv2d(
                in_channels=arch.conv_filters[1],  # 96
                out_channels=arch.conv_filters[2],  # 192
                kernel_size=arch.conv_kernel_size,
                padding=1,
            )
        )
        if arch.batch_norm:
            self.features.append(nn.BatchNorm2d(arch.conv_filters[2]))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(
            nn.MaxPool2d(kernel_size=arch.pool_size, stride=arch.pool_strides)
        )
        self.features.append(nn.Dropout(arch.dropout_rates[2]))

        # Global average pooling
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(arch.conv_filters[2], arch.dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(arch.dropout_rates[3]),
            nn.Linear(arch.dense_units, arch.num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, LegacyModelOutput]:
        """
        Forward pass through the CNN.

        Args:
            x: Input tensor of shape (batch_size, channels, freq, time)
                Expected shape: (B, 3, 40, 92) [channels, MFCC, time]
            return_features: Whether to return intermediate features

        Returns:
            If return_features=False: logits tensor of shape (batch_size, num_classes)
            If return_features=True: LegacyModelOutput with features and probabilities
        """
        # Feature extraction
        feat = self.features(x)  # (B, 192, 1, 1)
        feat_flat = feat.view(feat.size(0), -1)  # (B, 192)

        # Classification
        logits = self.classifier(feat_flat)  # (B, num_classes)

        if return_features:
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            return LegacyModelOutput(
                logits=logits,
                predictions=predictions,
                probabilities=probabilities,
                features=feat_flat,
            )

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the model (before classification).

        Args:
            x: Input tensor

        Returns:
            Feature tensor from before the classification head
        """
        feat = self.features(x)
        return feat.view(feat.size(0), -1)

    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extraction part of the model."""
        return self.features

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str, config: Optional[LegacyModelConfig] = None
    ) -> 'CNNMFCCModel':
        """
        Load a pre-trained CNN model.

        Args:
            checkpoint_path: Path to checkpoint file
            config: LegacyModelConfig (will be loaded from checkpoint if None)

        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if config is None:
            config = checkpoint.get('config')
            if config is None:
                raise ValueError("Config not found in checkpoint")

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pre-trained CNN model from {checkpoint_path}")
        return model
