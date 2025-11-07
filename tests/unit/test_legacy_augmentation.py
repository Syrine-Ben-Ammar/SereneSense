# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Unit Tests for Legacy SpecAugment
==================================
Test SpecAugment data augmentation.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.data.augmentation.legacy_specaugment import LegacySpecAugment, LegacySpecAugmentTransform


class TestLegacySpecAugment:
    """Test SpecAugment augmentation."""

    @pytest.fixture
    def augmenter(self):
        """Create augmenter."""
        return LegacySpecAugment(
            freq_mask_param=15,
            num_freq_masks=2,
            time_mask_param=10,
            num_time_masks=2,
            apply_prob=1.0,  # Always apply for testing
        )

    @pytest.fixture
    def features_2d(self):
        """Create 2D feature matrix."""
        return np.random.randn(40, 92)  # (mfcc, time)

    @pytest.fixture
    def features_3d(self):
        """Create 3D feature matrix."""
        return np.random.randn(40, 92, 3)  # (mfcc, time, channels)

    def test_augmenter_creation(self, augmenter):
        """Test augmenter creation."""
        assert augmenter is not None

    def test_augment_2d_numpy(self, augmenter, features_2d):
        """Test augmentation on 2D numpy array."""
        original = features_2d.copy()
        augmented = augmenter(features_2d)

        assert augmented.shape == original.shape
        assert isinstance(augmented, np.ndarray)
        # Should be different (masked)
        assert not np.allclose(augmented, original)

    def test_augment_3d_numpy(self, augmenter, features_3d):
        """Test augmentation on 3D numpy array."""
        original = features_3d.copy()
        augmented = augmenter(features_3d)

        assert augmented.shape == original.shape
        assert isinstance(augmented, np.ndarray)
        assert not np.allclose(augmented, original)

    def test_augment_2d_torch(self, augmenter, features_2d):
        """Test augmentation on 2D torch tensor."""
        features_torch = torch.from_numpy(features_2d)
        augmented = augmenter(features_torch)

        assert isinstance(augmented, torch.Tensor)
        assert augmented.shape == features_torch.shape

    def test_augment_3d_torch(self, augmenter, features_3d):
        """Test augmentation on 3D torch tensor."""
        features_torch = torch.from_numpy(features_3d)
        augmented = augmenter(features_torch)

        assert isinstance(augmented, torch.Tensor)
        assert augmented.shape == features_torch.shape

    def test_frequency_masking(self, features_2d):
        """Test frequency masking."""
        augmenter = LegacySpecAugment(
            freq_mask_param=10,
            num_freq_masks=1,
            time_mask_param=0,
            num_time_masks=0,
            apply_prob=1.0,
        )

        original = features_2d.copy()
        augmented = augmenter(features_2d)

        # Should have some zeros from masking
        assert np.any(augmented == 0)

    def test_time_masking(self, features_2d):
        """Test time masking."""
        augmenter = LegacySpecAugment(
            freq_mask_param=0,
            num_freq_masks=0,
            time_mask_param=10,
            num_time_masks=1,
            apply_prob=1.0,
        )

        original = features_2d.copy()
        augmented = augmenter(features_2d)

        # Should have some zeros from masking
        assert np.any(augmented == 0)

    def test_no_augmentation_prob_zero(self, augmenter, features_2d):
        """Test no augmentation when apply_prob=0."""
        augmenter.apply_prob = 0.0
        original = features_2d.copy()
        augmented = augmenter(features_2d)

        assert np.allclose(augmented, original)

    def test_mask_width_limits(self, features_2d):
        """Test mask width doesn't exceed feature dimensions."""
        augmenter = LegacySpecAugment(
            freq_mask_param=50,  # 50% of 40 = 20
            num_freq_masks=1,
            time_mask_param=50,  # 50% of 92 = 46
            num_time_masks=1,
            apply_prob=1.0,
        )

        augmented = augmenter(features_2d)
        assert augmented.shape == features_2d.shape

    def test_multiple_masks(self, features_2d):
        """Test multiple frequency and time masks."""
        augmenter = LegacySpecAugment(
            freq_mask_param=15,
            num_freq_masks=3,
            time_mask_param=10,
            num_time_masks=3,
            apply_prob=1.0,
        )

        augmented = augmenter(features_2d)
        assert augmented.shape == features_2d.shape
        # Should have significant masking with 6 total masks
        assert np.sum(augmented == 0) > np.sum(features_2d == 0)

    def test_static_freq_mask(self, features_2d):
        """Test static frequency mask method."""
        masked = LegacySpecAugment.apply_frequency_mask(features_2d, mask_width=10)

        assert masked.shape == features_2d.shape
        assert np.any(masked == 0)

    def test_static_time_mask(self, features_2d):
        """Test static time mask method."""
        masked = LegacySpecAugment.apply_time_mask(features_2d, mask_width=10)

        assert masked.shape == features_2d.shape
        assert np.any(masked == 0)

    def test_deterministic_masking(self, features_2d):
        """Test deterministic masking with fixed position."""
        features = features_2d.copy()
        masked1 = LegacySpecAugment.apply_frequency_mask(features, mask_width=5, position=5)
        masked2 = LegacySpecAugment.apply_frequency_mask(features, mask_width=5, position=5)

        assert np.allclose(masked1, masked2)


class TestLegacySpecAugmentTransform:
    """Test PyTorch-compatible SpecAugment transform."""

    @pytest.fixture
    def transform_train(self):
        """Create transform for training."""
        return LegacySpecAugmentTransform(
            freq_mask_param=15,
            num_freq_masks=2,
            time_mask_param=10,
            num_time_masks=2,
            apply_prob=1.0,
            training=True,
        )

    @pytest.fixture
    def transform_eval(self):
        """Create transform for evaluation."""
        return LegacySpecAugmentTransform(training=False)

    @pytest.fixture
    def dummy_tensor(self):
        """Create dummy tensor."""
        return torch.randn(4, 3, 40, 92)

    def test_transform_creation(self, transform_train):
        """Test transform creation."""
        assert transform_train is not None

    def test_augment_in_training_mode(self, transform_train, dummy_tensor):
        """Test augmentation in training mode."""
        original = dummy_tensor.clone()
        augmented = transform_train(dummy_tensor)

        assert augmented.shape == original.shape
        # Should be different due to augmentation
        assert not torch.allclose(augmented, original)

    def test_no_augment_in_eval_mode(self, transform_eval, dummy_tensor):
        """Test no augmentation in eval mode."""
        original = dummy_tensor.clone()
        augmented = transform_eval(dummy_tensor)

        assert torch.allclose(augmented, original)

    def test_set_training_true(self, transform_eval, dummy_tensor):
        """Test setting training mode to True."""
        transform_eval.set_training(True)
        assert transform_eval.training

        # Should augment now
        augmented = transform_eval(dummy_tensor)
        assert augmented.shape == dummy_tensor.shape

    def test_set_training_false(self, transform_train, dummy_tensor):
        """Test setting training mode to False."""
        transform_train.set_training(False)
        assert not transform_train.training

        # Should not augment
        original = dummy_tensor.clone()
        augmented = transform_train(dummy_tensor)
        assert torch.allclose(augmented, original)

    def test_batch_processing(self, transform_train):
        """Test augmentation on batch."""
        batch = torch.randn(8, 3, 40, 92)
        augmented = transform_train(batch)

        assert augmented.shape == batch.shape

    def test_preserves_dtype(self, transform_train):
        """Test that augmentation preserves dtype."""
        float_tensor = torch.randn(4, 3, 40, 92, dtype=torch.float32)
        augmented = transform_train(float_tensor)

        assert augmented.dtype == float_tensor.dtype

    def test_preserves_device(self, transform_train):
        """Test that augmentation preserves device."""
        if torch.cuda.is_available():
            cuda_tensor = torch.randn(4, 3, 40, 92, device="cuda")
            augmented = transform_train(cuda_tensor)
            assert augmented.device == cuda_tensor.device

        cpu_tensor = torch.randn(4, 3, 40, 92, device="cpu")
        augmented = transform_train(cpu_tensor)
        assert augmented.device == cpu_tensor.device
