# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Unit Tests for Legacy Feature Extraction
==========================================
Test MFCC feature extraction and preprocessing.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.data.preprocessing.legacy_mfcc import MFCCExtractor, LegacyMFCCPreprocessor


class TestMFCCExtractor:
    """Test MFCC extractor."""

    @pytest.fixture
    def extractor(self):
        """Create MFCC extractor."""
        return MFCCExtractor(
            sample_rate=16000,
            n_mfcc=40,
            n_mels=64,
            use_deltas=True,
            use_delta_deltas=True,
        )

    @pytest.fixture
    def dummy_audio(self):
        """Create dummy audio signal."""
        # 3 seconds at 16kHz
        return np.random.randn(16000 * 3).astype(np.float32)

    def test_extractor_creation(self, extractor):
        """Test extractor can be created."""
        assert extractor is not None
        assert extractor.n_mfcc == 40

    def test_extract_mfcc_shape(self, extractor, dummy_audio):
        """Test MFCC extraction produces correct shape."""
        features = extractor.extract(dummy_audio)

        # Should have 40 * 3 = 120 features (MFCC + delta + delta-delta)
        assert features.shape[0] == 120, f"Expected 120 features, got {features.shape[0]}"
        # Time dimension should be reasonable (3s at 16kHz with hop=512)
        assert features.shape[1] > 50, f"Expected >50 time steps, got {features.shape[1]}"

    def test_extract_without_deltas(self):
        """Test MFCC extraction without deltas."""
        extractor = MFCCExtractor(
            n_mfcc=40,
            use_deltas=False,
            use_delta_deltas=False,
        )
        audio = np.random.randn(16000 * 3)
        features = extractor.extract(audio)

        # Should have only 40 features (no delta, no delta-delta)
        assert features.shape[0] == 40

    def test_extract_with_shape(self, extractor, dummy_audio):
        """Test extraction with target shape."""
        target_shape = (120, 92)
        features = extractor.extract_with_shape(dummy_audio, target_shape)

        assert features.shape == target_shape

    def test_normalization(self, extractor, dummy_audio):
        """Test that features are normalized."""
        features = extractor.extract(dummy_audio)

        # Check per-coefficient normalization (mean should be close to 0)
        means = np.mean(features, axis=1)
        assert np.all(np.abs(means) < 0.5), "Features not properly normalized"

    def test_padding_in_extraction(self, extractor):
        """Test short audio is padded correctly."""
        short_audio = np.random.randn(16000 * 1)  # 1 second
        target_shape = (120, 92)

        features = extractor.extract_with_shape(short_audio, target_shape)
        assert features.shape == target_shape
        # Last few time steps should be padded with zeros
        assert np.allclose(features[:, -10:], 0), "Padding not applied correctly"

    def test_cropping_in_extraction(self, extractor):
        """Test long audio is cropped correctly."""
        long_audio = np.random.randn(16000 * 10)  # 10 seconds
        target_shape = (120, 92)

        features = extractor.extract_with_shape(long_audio, target_shape)
        assert features.shape == target_shape

    def test_feature_statistics(self, dummy_audio):
        """Test feature statistics computation."""
        extractor = MFCCExtractor()
        features = extractor.extract(dummy_audio)

        mean, std = MFCCExtractor.compute_statistics(features)

        assert mean.shape[0] == features.shape[0]
        assert std.shape[0] == features.shape[0]
        assert np.all(std > 0), "Std should be positive"

    def test_different_durations(self):
        """Test extraction with different audio durations."""
        extractor = MFCCExtractor(sample_rate=16000)

        # Test 1s, 3s, 5s
        for duration_s in [1, 3, 5]:
            audio = np.random.randn(16000 * duration_s)
            features = extractor.extract(audio)

            # More time = more frames (roughly proportional)
            assert features.shape[0] == 120  # Always 120 features (MFCC + deltas)
            assert features.shape[1] > 0  # Should have some time steps


class TestLegacyMFCCPreprocessor:
    """Test legacy MFCC preprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor."""
        return LegacyMFCCPreprocessor(
            sample_rate=16000,
            duration=3.0,
            n_mfcc=40,
            use_deltas=True,
            use_delta_deltas=True,
        )

    @pytest.fixture
    def dummy_audio(self):
        """Create dummy audio."""
        return np.random.randn(16000 * 3)

    def test_preprocessor_creation(self, preprocessor):
        """Test preprocessor creation."""
        assert preprocessor is not None
        assert preprocessor.sample_rate == 16000
        assert preprocessor.duration == 3.0

    def test_output_shape_property(self, preprocessor):
        """Test output shape property."""
        shape = preprocessor.output_shape
        assert isinstance(shape, tuple)
        assert len(shape) == 3
        assert shape[0] == 40  # n_mfcc

    def test_process_audio(self, preprocessor, dummy_audio):
        """Test audio processing."""
        features = preprocessor.process_audio(dummy_audio)

        expected_shape = preprocessor.output_shape
        assert features.shape == expected_shape

    def test_process_audio_short(self, preprocessor):
        """Test processing short audio (should be padded)."""
        short_audio = np.random.randn(16000 * 1)  # 1 second
        features = preprocessor.process_audio(short_audio)

        expected_shape = preprocessor.output_shape
        assert features.shape == expected_shape

    def test_process_audio_long(self, preprocessor):
        """Test processing long audio (should be cropped)."""
        long_audio = np.random.randn(16000 * 10)  # 10 seconds
        features = preprocessor.process_audio(long_audio)

        expected_shape = preprocessor.output_shape
        assert features.shape == expected_shape

    def test_feature_range(self, preprocessor, dummy_audio):
        """Test features are in reasonable range."""
        features = preprocessor.process_audio(dummy_audio)

        # Normalized features should be roughly in [-3, 3]
        assert np.all(np.abs(features) < 10), "Features out of expected range"

    def test_reproducibility(self, preprocessor, dummy_audio):
        """Test that processing is deterministic."""
        features1 = preprocessor.process_audio(dummy_audio.copy())
        features2 = preprocessor.process_audio(dummy_audio.copy())

        assert np.allclose(features1, features2), "Processing not deterministic"

    def test_crnn_preprocessor(self):
        """Test CRNN preprocessor (4 seconds)."""
        preprocessor = LegacyMFCCPreprocessor(
            sample_rate=16000,
            duration=4.0,
            n_mfcc=40,
        )

        audio = np.random.randn(16000 * 4)
        features = preprocessor.process_audio(audio)

        expected_shape = preprocessor.output_shape
        assert features.shape == expected_shape
        assert features.shape[0] == 40

    def test_without_deltas(self):
        """Test preprocessor without delta features."""
        preprocessor = LegacyMFCCPreprocessor(
            sample_rate=16000,
            duration=3.0,
            n_mfcc=40,
            use_deltas=False,
            use_delta_deltas=False,
        )

        audio = np.random.randn(16000 * 3)
        features = preprocessor.process_audio(audio)

        # Should have (40, T, 1) shape when no deltas
        assert features.shape[0] == 40
        assert features.shape[2] == 1  # Only MFCC channel


class TestMFCCNumericalStability:
    """Test numerical stability of MFCC extraction."""

    def test_silent_audio(self):
        """Test extraction on silent audio."""
        extractor = MFCCExtractor()
        silent_audio = np.zeros(16000 * 3)

        features = extractor.extract(silent_audio)

        # Should handle silent audio without NaN
        assert not np.any(np.isnan(features)), "NaN in features from silent audio"
        assert not np.any(np.isinf(features)), "Inf in features from silent audio"

    def test_very_loud_audio(self):
        """Test extraction on very loud audio."""
        extractor = MFCCExtractor()
        loud_audio = np.ones(16000 * 3) * 100.0

        features = extractor.extract(loud_audio)

        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_mixed_positive_negative(self):
        """Test extraction on mixed positive/negative audio."""
        extractor = MFCCExtractor()
        mixed_audio = np.sin(np.arange(16000 * 3) * 2 * np.pi / 16000)

        features = extractor.extract(mixed_audio)

        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
