# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
CRNN MFCC Model
===============
Convolutional Recurrent Neural Network trained on MFCC features for audio classification.
Architecture from the legacy notebooks: train_mad_crnn_gpu.ipynb

Model Size: ~1.5M parameters
Input: MFCC + delta + delta-delta features (40, 124, 3) for 4-second audio
Output: 7-class probabilities

Key Difference from CNN: Uses BiLSTM for temporal modeling, capturing
longer-range dependencies in audio sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import logging

from .base_legacy_model import BaseLegacyModel, LegacyModelOutput
from .legacy_config import LegacyModelConfig, CRNNConfig

logger = logging.getLogger(__name__)


class CRNNMFCCModel(BaseLegacyModel):
    """
    Convolutional Recurrent Neural Network for MFCC-based audio classification.

    Architecture:
        Input (40, 124, 3) [MFCC, time, channels] for 4-second audio
        ↓
        Conv2D(48, 3×3) + BatchNorm + MaxPool(2,1) + Dropout(0.20)
        ↓
        Conv2D(96, 3×3) + BatchNorm + MaxPool(2,1) + Dropout(0.25)
        ↓
        Conv2D(192, 3×3) + BatchNorm + MaxPool(2,1) + Dropout(0.30)
        ↓
        Reshape to (time_steps, 960) for RNN input
        ↓
        Bidirectional LSTM(128, return_sequences=True)
        ↓
        Bidirectional LSTM(64, return_sequences=True)
        ↓
        GlobalAvgPool + GlobalMaxPool (concatenate) → 256 features
        ↓
        Dense(160) + Dropout(0.35)
        ↓
        Dense(7) + Softmax

    Key Hyperparameters:
        - Input duration: 4 seconds (longer than CNN)
        - Conv layers: Same as CNN (48, 96, 192 filters)
        - LSTM units: 128, 64
        - MaxPool: (2,1) to preserve temporal dimension
        - Optimizer: Adam (lr=1e-3)
        - Epochs: 300
        - Batch size: 16 (smaller due to larger model)
        - Loss: Cross-entropy with class weights

    Advantages over CNN:
        - BiLSTM captures temporal dependencies explicitly
        - Better for audio with temporal patterns
        - Larger capacity (1.5M vs 242K parameters)
    """

    def __init__(self, config: LegacyModelConfig):
        """
        Initialize CRNN MFCC model.

        Args:
            config: LegacyModelConfig instance
        """
        super().__init__(config, model_name="crnn_mfcc")
        self.config = config
        self.arch_config = config.crnn

        # Build architecture
        self._build_architecture()

        # Log model summary
        logger.info(f"Initialized {self.model_name}")
        logger.info(f"Total parameters: {self.count_parameters():,}")

    def _build_architecture(self) -> None:
        """Build the CRNN architecture."""
        arch = self.arch_config

        # Convolutional feature extraction
        self.conv_layers = nn.Sequential()

        # Block 1: Conv → BatchNorm → MaxPool → Dropout
        self.conv_layers.append(
            nn.Conv2d(
                in_channels=arch.input_shape[2],  # 3
                out_channels=arch.conv_filters[0],  # 48
                kernel_size=arch.conv_kernel_size,
                padding=1,
            )
        )
        if arch.batch_norm:
            self.conv_layers.append(nn.BatchNorm2d(arch.conv_filters[0]))
        self.conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers.append(
            nn.MaxPool2d(kernel_size=arch.pool_size, stride=arch.pool_strides)
        )
        self.conv_layers.append(nn.Dropout(arch.conv_dropout_rates[0]))

        # Block 2: Conv → BatchNorm → MaxPool → Dropout
        self.conv_layers.append(
            nn.Conv2d(
                in_channels=arch.conv_filters[0],  # 48
                out_channels=arch.conv_filters[1],  # 96
                kernel_size=arch.conv_kernel_size,
                padding=1,
            )
        )
        if arch.batch_norm:
            self.conv_layers.append(nn.BatchNorm2d(arch.conv_filters[1]))
        self.conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers.append(
            nn.MaxPool2d(kernel_size=arch.pool_size, stride=arch.pool_strides)
        )
        self.conv_layers.append(nn.Dropout(arch.conv_dropout_rates[1]))

        # Block 3: Conv → BatchNorm → MaxPool → Dropout
        self.conv_layers.append(
            nn.Conv2d(
                in_channels=arch.conv_filters[1],  # 96
                out_channels=arch.conv_filters[2],  # 192
                kernel_size=arch.conv_kernel_size,
                padding=1,
            )
        )
        if arch.batch_norm:
            self.conv_layers.append(nn.BatchNorm2d(arch.conv_filters[2]))
        self.conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers.append(
            nn.MaxPool2d(kernel_size=arch.pool_size, stride=arch.pool_strides)
        )
        self.conv_layers.append(nn.Dropout(arch.conv_dropout_rates[2]))

        # After conv layers: (B, 192, 5, 124) → reshape to (B, 124, 960)
        # 192 filters * (40/8) freq dimension = 960 features per time step

        # Bidirectional LSTM layers
        self.lstm_layers = nn.Sequential()

        # LSTM 1: Bidirectional
        self.lstm_layers.append(
            nn.LSTM(
                input_size=960,  # Features from conv
                hidden_size=arch.lstm_units[0],  # 128
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=arch.rnn_dropout if len(arch.lstm_units) > 1 else 0,
            )
        )

        # LSTM 2: Bidirectional
        self.lstm_layers.append(
            nn.LSTM(
                input_size=arch.lstm_units[0] * 2,  # 128*2 from previous bidirectional
                hidden_size=arch.lstm_units[1],  # 64
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=arch.rnn_dropout,
            )
        )

        # Pooling and dense layers
        # After LSTM: (B, time_steps, 128) from bidirectional
        # Pool across time → (B, 128) avg + (B, 128) max = (B, 256)

        self.classifier = nn.Sequential(
            nn.Linear(arch.lstm_units[-1] * 2 * 2, arch.dense_units),  # 256 → 160
            nn.ReLU(inplace=True),
            nn.Dropout(arch.final_dropout),
            nn.Linear(arch.dense_units, arch.num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, LegacyModelOutput]:
        """
        Forward pass through the CRNN.

        Args:
            x: Input tensor of shape (batch_size, channels, freq, time)
                Expected shape: (B, 3, 40, 124) [channels, MFCC, time]
            return_features: Whether to return intermediate features

        Returns:
            If return_features=False: logits tensor of shape (batch_size, num_classes)
            If return_features=True: LegacyModelOutput with features and probabilities
        """
        batch_size = x.size(0)

        # Convolutional feature extraction
        conv_out = self.conv_layers(x)  # (B, 192, 5, 124)

        # Reshape for LSTM: (B, 192, 5, 124) → (B, 124, 960)
        # Merge frequency and filter dimensions
        conv_out = conv_out.transpose(2, 3)  # (B, 192, 124, 5)
        conv_out = conv_out.contiguous().view(batch_size, conv_out.size(2), -1)  # (B, 124, 960)

        # LSTM processing
        lstm_out = conv_out
        for lstm_layer in self.lstm_layers:
            lstm_out, _ = lstm_layer(lstm_out)  # (B, time, hidden*2)

        # Pool across time dimension
        avg_pool = torch.mean(lstm_out, dim=1)  # (B, 128)
        max_pool = torch.max(lstm_out, dim=1)[0]  # (B, 128)

        # Concatenate pooling results
        feat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 256)

        # Classification
        logits = self.classifier(feat)  # (B, num_classes)

        if return_features:
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            return LegacyModelOutput(
                logits=logits,
                predictions=predictions,
                probabilities=probabilities,
                features=feat,
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
        batch_size = x.size(0)
        conv_out = self.conv_layers(x)
        conv_out = conv_out.transpose(2, 3)
        conv_out = conv_out.contiguous().view(batch_size, conv_out.size(2), -1)

        lstm_out = conv_out
        for lstm_layer in self.lstm_layers:
            lstm_out, _ = lstm_layer(lstm_out)

        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool = torch.max(lstm_out, dim=1)[0]
        feat = torch.cat([avg_pool, max_pool], dim=1)

        return feat

    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extraction part of the model (conv + LSTM)."""
        return nn.Sequential(self.conv_layers, self.lstm_layers)

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str, config: Optional[LegacyModelConfig] = None
    ) -> 'CRNNMFCCModel':
        """
        Load a pre-trained CRNN model.

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
        logger.info(f"Loaded pre-trained CRNN model from {checkpoint_path}")
        return model
