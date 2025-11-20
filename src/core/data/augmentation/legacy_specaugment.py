# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Legacy SpecAugment Implementation
==================================
SpecAugment data augmentation technique for audio spectrograms.
Applied to MFCC features during training.

Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method
for Automatic Speech Recognition" (2019)
"""

import numpy as np
import torch
from typing import Union, Optional, Tuple
import random
import logging

logger = logging.getLogger(__name__)


class LegacySpecAugment:
    """
    SpecAugment augmentation for spectral features (MFCC, mel-spectrograms).

    Applies:
        1. Frequency masking: Mask a contiguous region of frequencies
        2. Time masking: Mask a contiguous region of time steps
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        num_freq_masks: int = 2,
        time_mask_param: int = 10,
        num_time_masks: int = 2,
        apply_prob: float = 0.8,
    ):
        """
        Initialize SpecAugment.

        Args:
            freq_mask_param: Maximum width of frequency masks (percentage or absolute)
            num_freq_masks: Number of frequency masks to apply
            time_mask_param: Maximum width of time masks (percentage or absolute)
            num_time_masks: Number of time masks to apply
            apply_prob: Probability of applying augmentation
        """
        self.freq_mask_param = freq_mask_param
        self.num_freq_masks = num_freq_masks
        self.time_mask_param = time_mask_param
        self.num_time_masks = num_time_masks
        self.apply_prob = apply_prob

    def __call__(self, features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply SpecAugment to features.

        Args:
            features: Feature matrix of shape (batch, channels, n_features, time), (n_features, time, channels), or (n_features, time)

        Returns:
            Augmented features of same shape
        """
        # Skip augmentation with probability
        if random.random() > self.apply_prob:
            return features

        # Convert to numpy if needed
        is_torch = isinstance(features, torch.Tensor)
        if is_torch:
            features_np = features.cpu().numpy()
        else:
            features_np = features.copy()

        # Handle 4D input (batch, channels, freq, time) - apply to each item in batch
        if features_np.ndim == 4:
            batch_size = features_np.shape[0]
            augmented = features_np.copy()
            for i in range(batch_size):
                # Process each item: (channels, freq, time)
                for c in range(features_np.shape[1]):
                    # Apply to each channel: (freq, time)
                    item = augmented[i, c, :, :]
                    for _ in range(self.num_freq_masks):
                        item = self._apply_freq_mask_2d(item)
                    for _ in range(self.num_time_masks):
                        item = self._apply_time_mask_2d(item)
                    augmented[i, c, :, :] = item
        # Handle 3D input (n_features, time, channels)
        elif features_np.ndim == 3:
            augmented = features_np.copy()
            for _ in range(self.num_freq_masks):
                augmented = self._apply_freq_mask_3d(augmented)
            for _ in range(self.num_time_masks):
                augmented = self._apply_time_mask_3d(augmented)
        else:
            # 2D input (n_features, time)
            augmented = features_np.copy()
            for _ in range(self.num_freq_masks):
                augmented = self._apply_freq_mask_2d(augmented)
            for _ in range(self.num_time_masks):
                augmented = self._apply_time_mask_2d(augmented)

        # Convert back to torch if needed
        if is_torch:
            augmented = torch.from_numpy(augmented).to(features.dtype).to(features.device)

        return augmented

    def _apply_freq_mask_2d(self, features: np.ndarray) -> np.ndarray:
        """Apply frequency masking to 2D features (n_features, time)."""
        n_features, n_frames = features.shape
        mask_width = random.randint(0, self.freq_mask_param)

        if mask_width > 0:
            f_start = random.randint(0, n_features - mask_width)
            features[f_start : f_start + mask_width, :] = 0

        return features

    def _apply_time_mask_2d(self, features: np.ndarray) -> np.ndarray:
        """Apply time masking to 2D features."""
        n_features, n_frames = features.shape
        mask_width = random.randint(0, self.time_mask_param)

        if mask_width > 0:
            t_start = random.randint(0, n_frames - mask_width)
            features[:, t_start : t_start + mask_width] = 0

        return features

    def _apply_freq_mask_3d(self, features: np.ndarray) -> np.ndarray:
        """Apply frequency masking to 3D features (n_features, time, channels)."""
        n_features, n_frames, n_channels = features.shape
        mask_width = random.randint(0, self.freq_mask_param)

        if mask_width > 0:
            f_start = random.randint(0, n_features - mask_width)
            features[f_start : f_start + mask_width, :, :] = 0

        return features

    def _apply_time_mask_3d(self, features: np.ndarray) -> np.ndarray:
        """Apply time masking to 3D features."""
        n_features, n_frames, n_channels = features.shape
        mask_width = random.randint(0, self.time_mask_param)

        if mask_width > 0:
            t_start = random.randint(0, n_frames - mask_width)
            features[:, t_start : t_start + mask_width, :] = 0

        return features

    @staticmethod
    def apply_frequency_mask(
        features: np.ndarray, mask_width: int, position: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply a single frequency mask.

        Args:
            features: Feature matrix
            mask_width: Width of mask in frequency dimension
            position: Starting position (random if None)

        Returns:
            Masked features
        """
        features = features.copy()
        n_features = features.shape[0]

        if position is None:
            position = random.randint(0, max(0, n_features - mask_width))

        if mask_width > 0:
            end_pos = min(position + mask_width, n_features)
            features[position:end_pos] = 0

        return features

    @staticmethod
    def apply_time_mask(
        features: np.ndarray, mask_width: int, position: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply a single time mask.

        Args:
            features: Feature matrix
            mask_width: Width of mask in time dimension
            position: Starting position (random if None)

        Returns:
            Masked features
        """
        features = features.copy()
        n_frames = features.shape[1] if features.ndim >= 2 else features.shape[0]

        if position is None:
            position = random.randint(0, max(0, n_frames - mask_width))

        if mask_width > 0:
            end_pos = min(position + mask_width, n_frames)
            if features.ndim == 2:
                features[:, position:end_pos] = 0
            else:
                features[:, position:end_pos, :] = 0

        return features


class LegacySpecAugmentTransform:
    """PyTorch-compatible SpecAugment transform for DataLoader integration."""

    def __init__(
        self,
        freq_mask_param: int = 15,
        num_freq_masks: int = 2,
        time_mask_param: int = 10,
        num_time_masks: int = 2,
        apply_prob: float = 0.8,
        training: bool = True,
    ):
        """
        Initialize SpecAugment transform.

        Args:
            freq_mask_param: Frequency mask parameter
            num_freq_masks: Number of frequency masks
            time_mask_param: Time mask parameter
            num_time_masks: Number of time masks
            apply_prob: Probability of applying augmentation
            training: Only apply when training=True
        """
        self.augmenter = LegacySpecAugment(
            freq_mask_param=freq_mask_param,
            num_freq_masks=num_freq_masks,
            time_mask_param=time_mask_param,
            num_time_masks=num_time_masks,
            apply_prob=apply_prob,
        )
        self.training = training

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation if in training mode."""
        if self.training:
            return self.augmenter(x)
        return x

    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self.training = training
