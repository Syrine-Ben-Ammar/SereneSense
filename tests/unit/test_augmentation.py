"""
Unit tests for SereneSense data augmentation modules.

Tests all augmentation techniques for:
- Time domain augmentations (noise, time stretch, pitch shift)
- Frequency domain augmentations (frequency masking, filtering)
- SpecAugment implementations
- Parameter validation
- Output shape consistency
- Augmentation strength control
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data.augmentation.time_domain import (
    AddNoise,
    TimeStretch,
    PitchShift,
    TimeShift,
    Clip,
    Normalize,
    RandomGain,
)
from core.data.augmentation.frequency_domain import (
    FrequencyMasking,
    TimeMasking,
    BandpassFilter,
    HighpassFilter,
    LowpassFilter,
    SpectralRolloff,
)
from core.data.augmentation.spec_augment import SpecAugment


class TestTimeDomainAugmentations:
    """Test time domain augmentation techniques."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 16000
        duration = 2.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        # Generate a simple sine wave
        audio = torch.sin(2 * np.pi * 440 * t)  # A4 note
        return audio.unsqueeze(0)  # Add channel dimension

    def test_add_noise_initialization(self):
        """Test AddNoise augmentation initialization."""
        augment = AddNoise(noise_level=0.1)
        assert augment.noise_level == 0.1

        # Test with range
        augment_range = AddNoise(noise_level=(0.05, 0.15))
        assert augment_range.noise_level == (0.05, 0.15)

    def test_add_noise_forward(self, sample_audio):
        """Test AddNoise forward pass."""
        augment = AddNoise(noise_level=0.1)

        # Apply augmentation
        augmented = augment(sample_audio)

        # Check output properties
        assert augmented.shape == sample_audio.shape
        assert not torch.equal(augmented, sample_audio)  # Should be different
        assert torch.isfinite(augmented).all()  # No NaN or Inf values

        # Check that noise was actually added (variance should increase)
        original_var = torch.var(sample_audio)
        augmented_var = torch.var(augmented)
        assert augmented_var > original_var

    def test_add_noise_deterministic(self, sample_audio):
        """Test AddNoise deterministic behavior with seed."""
        augment = AddNoise(noise_level=0.1)

        # Set seed and apply twice
        torch.manual_seed(42)
        augmented1 = augment(sample_audio)

        torch.manual_seed(42)
        augmented2 = augment(sample_audio)

        # Should be identical with same seed
        assert torch.allclose(augmented1, augmented2, atol=1e-6)

    def test_time_stretch_initialization(self):
        """Test TimeStretch augmentation initialization."""
        augment = TimeStretch(stretch_factor=1.2)
        assert augment.stretch_factor == 1.2

        # Test with range
        augment_range = TimeStretch(stretch_factor=(0.8, 1.2))
        assert augment_range.stretch_factor == (0.8, 1.2)

    def test_time_stretch_forward(self, sample_audio):
        """Test TimeStretch forward pass."""
        augment = TimeStretch(stretch_factor=1.5)

        original_length = sample_audio.shape[-1]
        augmented = augment(sample_audio)

        # Length should change (approximately)
        expected_length = int(original_length * 1.5)
        assert abs(augmented.shape[-1] - expected_length) < 100  # Allow some tolerance
        assert augmented.shape[0] == sample_audio.shape[0]  # Channel dimension preserved

    def test_pitch_shift_initialization(self):
        """Test PitchShift augmentation initialization."""
        augment = PitchShift(semitones=2)
        assert augment.semitones == 2

        # Test with range
        augment_range = PitchShift(semitones=(-4, 4))
        assert augment_range.semitones == (-4, 4)

    def test_pitch_shift_forward(self, sample_audio):
        """Test PitchShift forward pass."""
        augment = PitchShift(semitones=4)

        augmented = augment(sample_audio)

        # Shape should be preserved
        assert augmented.shape == sample_audio.shape
        assert not torch.equal(augmented, sample_audio)  # Should be different
        assert torch.isfinite(augmented).all()

    def test_time_shift_initialization(self):
        """Test TimeShift augmentation initialization."""
        augment = TimeShift(max_shift=0.1)
        assert augment.max_shift == 0.1

    def test_time_shift_forward(self, sample_audio):
        """Test TimeShift forward pass."""
        augment = TimeShift(max_shift=0.2)

        augmented = augment(sample_audio)

        # Shape should be preserved
        assert augmented.shape == sample_audio.shape
        assert torch.isfinite(augmented).all()

        # Should be different (unless shift is 0)
        # We can't guarantee they're different due to random shift

    def test_clip_forward(self, sample_audio):
        """Test Clip augmentation forward pass."""
        augment = Clip(min_val=-0.5, max_val=0.5)

        # Create audio with values outside clip range
        extreme_audio = sample_audio * 2.0  # Scale up

        augmented = augment(extreme_audio)

        # Values should be clipped
        assert torch.max(augmented) <= 0.5
        assert torch.min(augmented) >= -0.5
        assert augmented.shape == extreme_audio.shape

    def test_normalize_forward(self, sample_audio):
        """Test Normalize augmentation forward pass."""
        augment = Normalize()

        # Scale audio to have different amplitude
        scaled_audio = sample_audio * 3.0

        augmented = augment(scaled_audio)

        # Should be normalized to [-1, 1] range
        assert torch.max(torch.abs(augmented)) <= 1.0
        assert augmented.shape == scaled_audio.shape

    def test_random_gain_initialization(self):
        """Test RandomGain augmentation initialization."""
        augment = RandomGain(gain_range=(0.5, 2.0))
        assert augment.gain_range == (0.5, 2.0)

    def test_random_gain_forward(self, sample_audio):
        """Test RandomGain forward pass."""
        augment = RandomGain(gain_range=(2.0, 2.0))  # Fixed gain for testing

        augmented = augment(sample_audio)

        # Should be scaled by gain factor
        expected = sample_audio * 2.0
        assert torch.allclose(augmented, expected, atol=1e-6)
        assert augmented.shape == sample_audio.shape


class TestFrequencyDomainAugmentations:
    """Test frequency domain augmentation techniques."""

    @pytest.fixture
    def sample_spectrogram(self):
        """Create sample spectrogram for testing."""
        # Create a sample mel spectrogram (freq_bins, time_frames)
        freq_bins = 128
        time_frames = 100
        spectrogram = torch.randn(freq_bins, time_frames)
        return spectrogram.unsqueeze(0)  # Add batch dimension

    def test_frequency_masking_initialization(self):
        """Test FrequencyMasking augmentation initialization."""
        augment = FrequencyMasking(freq_mask_param=27)
        assert augment.freq_mask_param == 27

    def test_frequency_masking_forward(self, sample_spectrogram):
        """Test FrequencyMasking forward pass."""
        augment = FrequencyMasking(freq_mask_param=20)

        augmented = augment(sample_spectrogram)

        # Shape should be preserved
        assert augmented.shape == sample_spectrogram.shape

        # Some frequencies should be masked (set to 0)
        masked_frequencies = (augmented == 0).any(dim=-1).sum()
        assert masked_frequencies > 0  # At least some frequencies should be masked

    def test_time_masking_initialization(self):
        """Test TimeMasking augmentation initialization."""
        augment = TimeMasking(time_mask_param=40)
        assert augment.time_mask_param == 40

    def test_time_masking_forward(self, sample_spectrogram):
        """Test TimeMasking forward pass."""
        augment = TimeMasking(time_mask_param=10)

        augmented = augment(sample_spectrogram)

        # Shape should be preserved
        assert augmented.shape == sample_spectrogram.shape

        # Some time frames should be masked
        masked_frames = (augmented == 0).any(dim=1).sum()
        assert masked_frames > 0  # At least some frames should be masked

    def test_bandpass_filter_initialization(self):
        """Test BandpassFilter augmentation initialization."""
        augment = BandpassFilter(low_freq=300, high_freq=3400, sample_rate=16000)
        assert augment.low_freq == 300
        assert augment.high_freq == 3400
        assert augment.sample_rate == 16000

    def test_bandpass_filter_forward(self):
        """Test BandpassFilter forward pass."""
        sample_rate = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))

        # Create signal with multiple frequencies
        low_freq_signal = torch.sin(2 * np.pi * 100 * t)  # Below filter range
        mid_freq_signal = torch.sin(2 * np.pi * 1000 * t)  # Within filter range
        high_freq_signal = torch.sin(2 * np.pi * 5000 * t)  # Above filter range

        mixed_signal = (low_freq_signal + mid_freq_signal + high_freq_signal).unsqueeze(0)

        augment = BandpassFilter(low_freq=500, high_freq=2000, sample_rate=sample_rate)
        filtered = augment(mixed_signal)

        # Shape should be preserved
        assert filtered.shape == mixed_signal.shape
        assert torch.isfinite(filtered).all()

    def test_highpass_filter_forward(self):
        """Test HighpassFilter forward pass."""
        sample_rate = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))

        # Create signal with low and high frequency components
        low_freq = torch.sin(2 * np.pi * 100 * t)
        high_freq = torch.sin(2 * np.pi * 2000 * t)
        mixed_signal = (low_freq + high_freq).unsqueeze(0)

        augment = HighpassFilter(cutoff_freq=1000, sample_rate=sample_rate)
        filtered = augment(mixed_signal)

        assert filtered.shape == mixed_signal.shape
        assert torch.isfinite(filtered).all()

    def test_lowpass_filter_forward(self):
        """Test LowpassFilter forward pass."""
        sample_rate = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))

        # Create signal with low and high frequency components
        low_freq = torch.sin(2 * np.pi * 500 * t)
        high_freq = torch.sin(2 * np.pi * 3000 * t)
        mixed_signal = (low_freq + high_freq).unsqueeze(0)

        augment = LowpassFilter(cutoff_freq=1000, sample_rate=sample_rate)
        filtered = augment(mixed_signal)

        assert filtered.shape == mixed_signal.shape
        assert torch.isfinite(filtered).all()

    def test_spectral_rolloff_forward(self, sample_spectrogram):
        """Test SpectralRolloff forward pass."""
        augment = SpectralRolloff(rolloff_factor=0.85)

        augmented = augment(sample_spectrogram)

        # Shape should be preserved
        assert augmented.shape == sample_spectrogram.shape
        assert torch.isfinite(augmented).all()


class TestSpecAugment:
    """Test SpecAugment implementation."""

    @pytest.fixture
    def sample_spectrogram(self):
        """Create sample spectrogram for testing."""
        batch_size = 2
        freq_bins = 128
        time_frames = 100
        return torch.randn(batch_size, freq_bins, time_frames)

    def test_spec_augment_initialization(self):
        """Test SpecAugment initialization."""
        augment = SpecAugment(
            freq_mask_param=27, time_mask_param=40, num_freq_mask=2, num_time_mask=2
        )

        assert augment.freq_mask_param == 27
        assert augment.time_mask_param == 40
        assert augment.num_freq_mask == 2
        assert augment.num_time_mask == 2

    def test_spec_augment_forward(self, sample_spectrogram):
        """Test SpecAugment forward pass."""
        augment = SpecAugment(
            freq_mask_param=20, time_mask_param=30, num_freq_mask=1, num_time_mask=1
        )

        augmented = augment(sample_spectrogram)

        # Shape should be preserved
        assert augmented.shape == sample_spectrogram.shape

        # Some values should be masked (set to 0)
        num_masked = (augmented == 0).sum()
        assert num_masked > 0  # At least some values should be masked

    def test_spec_augment_multiple_masks(self, sample_spectrogram):
        """Test SpecAugment with multiple masks."""
        augment = SpecAugment(
            freq_mask_param=15, time_mask_param=20, num_freq_mask=3, num_time_mask=3
        )

        augmented = augment(sample_spectrogram)

        # Shape should be preserved
        assert augmented.shape == sample_spectrogram.shape

        # More masks should result in more masked values
        num_masked = (augmented == 0).sum()
        assert num_masked > 0

    def test_spec_augment_no_masking(self, sample_spectrogram):
        """Test SpecAugment with no masking."""
        augment = SpecAugment(
            freq_mask_param=0, time_mask_param=0, num_freq_mask=0, num_time_mask=0
        )

        augmented = augment(sample_spectrogram)

        # Should be identical to input
        assert torch.equal(augmented, sample_spectrogram)

    def test_spec_augment_batch_consistency(self, sample_spectrogram):
        """Test SpecAugment batch processing consistency."""
        augment = SpecAugment(
            freq_mask_param=20, time_mask_param=30, num_freq_mask=1, num_time_mask=1
        )

        # Process batch
        batch_augmented = augment(sample_spectrogram)

        # Process individually
        individual_augmented = torch.stack(
            [augment(sample_spectrogram[i : i + 1]) for i in range(sample_spectrogram.shape[0])]
        ).squeeze(1)

        # Shapes should match
        assert batch_augmented.shape == individual_augmented.shape


class TestAugmentationCompositions:
    """Test combinations of augmentation techniques."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 16000
        duration = 2.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * np.pi * 440 * t)
        return audio.unsqueeze(0)

    def test_sequential_time_domain_augmentations(self, sample_audio):
        """Test sequential application of time domain augmentations."""
        # Create sequence of augmentations
        augmentations = [
            AddNoise(noise_level=0.05),
            RandomGain(gain_range=(0.8, 1.2)),
            Clip(min_val=-0.9, max_val=0.9),
        ]

        augmented = sample_audio
        for augment in augmentations:
            augmented = augment(augmented)

        # Final result should have correct shape and be finite
        assert augmented.shape == sample_audio.shape
        assert torch.isfinite(augmented).all()
        assert torch.max(torch.abs(augmented)) <= 0.9  # Clipped

    def test_combined_time_frequency_augmentations(self, sample_audio):
        """Test combination of time and frequency domain augmentations."""
        # Apply time domain augmentation
        time_augment = AddNoise(noise_level=0.1)
        augmented_audio = time_augment(sample_audio)

        # Convert to spectrogram (mock)
        spectrogram = torch.stft(
            augmented_audio.squeeze(0), n_fft=512, hop_length=256, return_complex=True
        ).abs()
        spectrogram = spectrogram.unsqueeze(0)

        # Apply frequency domain augmentation
        freq_augment = FrequencyMasking(freq_mask_param=10)
        augmented_spec = freq_augment(spectrogram)

        # Check final result
        assert augmented_spec.shape == spectrogram.shape
        assert torch.isfinite(augmented_spec).all()

    def test_augmentation_probability(self, sample_audio):
        """Test augmentation with probability control."""

        class ProbabilisticAugment:
            def __init__(self, augment, prob=0.5):
                self.augment = augment
                self.prob = prob

            def __call__(self, x):
                if torch.rand(1).item() < self.prob:
                    return self.augment(x)
                return x

        # Create probabilistic augmentation
        prob_augment = ProbabilisticAugment(AddNoise(noise_level=0.1), prob=0.0)  # Never apply

        augmented = prob_augment(sample_audio)

        # Should be identical (probability = 0)
        assert torch.equal(augmented, sample_audio)

    def test_augmentation_pipeline(self, sample_audio):
        """Test complete augmentation pipeline."""

        class AugmentationPipeline:
            def __init__(self, augmentations):
                self.augmentations = augmentations

            def __call__(self, x):
                for augment in self.augmentations:
                    x = augment(x)
                return x

        # Create pipeline
        pipeline = AugmentationPipeline(
            [
                AddNoise(noise_level=0.02),
                RandomGain(gain_range=(0.9, 1.1)),
                TimeShift(max_shift=0.1),
                Normalize(),
            ]
        )

        augmented = pipeline(sample_audio)

        # Check final properties
        assert augmented.shape == sample_audio.shape
        assert torch.isfinite(augmented).all()
        assert torch.max(torch.abs(augmented)) <= 1.0  # Normalized


class TestAugmentationEdgeCases:
    """Test augmentation behavior with edge cases."""

    def test_zero_audio_input(self):
        """Test augmentations with zero audio input."""
        zero_audio = torch.zeros(1, 16000)

        # Test various augmentations
        augmentations = [
            AddNoise(noise_level=0.1),
            RandomGain(gain_range=(0.5, 2.0)),
            Normalize(),
            Clip(min_val=-0.5, max_val=0.5),
        ]

        for augment in augmentations:
            result = augment(zero_audio)
            assert result.shape == zero_audio.shape
            assert torch.isfinite(result).all()

    def test_extreme_parameter_values(self):
        """Test augmentations with extreme parameter values."""
        sample_audio = torch.randn(1, 16000)

        # Test with extreme noise level
        extreme_noise = AddNoise(noise_level=10.0)
        result = extreme_noise(sample_audio)
        assert torch.isfinite(result).all()

        # Test with extreme gain
        extreme_gain = RandomGain(gain_range=(0.001, 0.001))
        result = extreme_gain(sample_audio)
        assert torch.max(torch.abs(result)) < torch.max(torch.abs(sample_audio))

    def test_single_sample_input(self):
        """Test augmentations with single sample input."""
        single_sample = torch.randn(1, 1)

        # Most augmentations should handle single sample gracefully
        simple_augments = [
            AddNoise(noise_level=0.1),
            RandomGain(gain_range=(0.5, 2.0)),
            Clip(min_val=-1.0, max_val=1.0),
            Normalize(),
        ]

        for augment in simple_augments:
            result = augment(single_sample)
            assert result.shape == single_sample.shape
            assert torch.isfinite(result).all()

    def test_very_long_audio(self):
        """Test augmentations with very long audio."""
        # Create 1 minute of audio
        long_audio = torch.randn(1, 16000 * 60)

        # Test that augmentations can handle long sequences
        augment = AddNoise(noise_level=0.05)
        result = augment(long_audio)

        assert result.shape == long_audio.shape
        assert torch.isfinite(result).all()


class TestAugmentationParameterValidation:
    """Test parameter validation in augmentations."""

    def test_invalid_noise_level(self):
        """Test AddNoise with invalid noise level."""
        with pytest.raises(ValueError):
            AddNoise(noise_level=-0.1)  # Negative noise level

    def test_invalid_gain_range(self):
        """Test RandomGain with invalid gain range."""
        with pytest.raises(ValueError):
            RandomGain(gain_range=(2.0, 1.0))  # min > max

    def test_invalid_frequency_mask_param(self):
        """Test FrequencyMasking with invalid parameters."""
        with pytest.raises(ValueError):
            FrequencyMasking(freq_mask_param=-1)  # Negative parameter

    def test_invalid_time_mask_param(self):
        """Test TimeMasking with invalid parameters."""
        with pytest.raises(ValueError):
            TimeMasking(time_mask_param=-1)  # Negative parameter

    def test_invalid_filter_frequencies(self):
        """Test filters with invalid frequency parameters."""
        with pytest.raises(ValueError):
            BandpassFilter(low_freq=2000, high_freq=1000, sample_rate=16000)  # low > high

        with pytest.raises(ValueError):
            HighpassFilter(cutoff_freq=-100, sample_rate=16000)  # Negative frequency

        with pytest.raises(ValueError):
            LowpassFilter(cutoff_freq=20000, sample_rate=16000)  # Above Nyquist


class TestAugmentationReproducibility:
    """Test augmentation reproducibility."""

    def test_deterministic_with_seed(self):
        """Test that augmentations are deterministic with seed."""
        sample_audio = torch.randn(1, 16000)

        # Test with random augmentation
        augment = AddNoise(noise_level=0.1)

        # Apply with same seed twice
        torch.manual_seed(42)
        result1 = augment(sample_audio)

        torch.manual_seed(42)
        result2 = augment(sample_audio)

        assert torch.allclose(result1, result2, atol=1e-6)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        sample_audio = torch.randn(1, 16000)
        augment = AddNoise(noise_level=0.1)

        # Apply with different seeds
        torch.manual_seed(42)
        result1 = augment(sample_audio)

        torch.manual_seed(123)
        result2 = augment(sample_audio)

        # Results should be different
        assert not torch.allclose(result1, result2, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
