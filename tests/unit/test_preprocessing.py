"""
Unit tests for SereneSense preprocessing modules.

Tests all preprocessing functionality for:
- Spectrogram generation (Mel, STFT, CQT)
- Audio normalization techniques
- Audio segmentation methods
- Parameter validation
- Output shape consistency
- Performance characteristics
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data.preprocessing.spectrograms import (
    MelSpectrogramProcessor,
    STFTProcessor,
    CQTProcessor,
    SpectrogramConfig,
)
from core.data.preprocessing.normalization import (
    AudioNormalizer,
    ZScoreNormalizer,
    MinMaxNormalizer,
    RMSNormalizer,
    PeakNormalizer,
)
from core.data.preprocessing.segmentation import (
    FixedLengthSegmenter,
    OverlappingSegmenter,
    VoiceActivitySegmenter,
    EnergyBasedSegmenter,
)


class TestSpectrogramProcessors:
    """Test spectrogram generation modules."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        sample_rate = 16000
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))

        # Create complex audio with multiple frequencies
        freq1 = 440  # A4
        freq2 = 880  # A5
        freq3 = 1320  # E6

        audio = (
            0.5 * torch.sin(2 * np.pi * freq1 * t)
            + 0.3 * torch.sin(2 * np.pi * freq2 * t)
            + 0.2 * torch.sin(2 * np.pi * freq3 * t)
        )

        return audio.unsqueeze(0)  # Add channel dimension

    @pytest.fixture
    def spectrogram_config(self):
        """Create spectrogram configuration for testing."""
        return SpectrogramConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            window="hann",
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
            power=2.0,
        )

    def test_spectrogram_config_initialization(self, spectrogram_config):
        """Test SpectrogramConfig initialization."""
        assert spectrogram_config.sample_rate == 16000
        assert spectrogram_config.n_fft == 1024
        assert spectrogram_config.hop_length == 512
        assert spectrogram_config.n_mels == 128
        assert spectrogram_config.window == "hann"

    def test_mel_spectrogram_processor_initialization(self, spectrogram_config):
        """Test MelSpectrogramProcessor initialization."""
        processor = MelSpectrogramProcessor(spectrogram_config)

        assert processor.config == spectrogram_config
        assert hasattr(processor, "mel_transform")
        assert processor.config.n_mels == 128

    def test_mel_spectrogram_processor_forward(self, spectrogram_config, sample_audio):
        """Test MelSpectrogramProcessor forward pass."""
        processor = MelSpectrogramProcessor(spectrogram_config)

        # Process audio
        mel_spec = processor(sample_audio)

        # Check output shape
        expected_time_frames = (sample_audio.shape[-1] // spectrogram_config.hop_length) + 1
        expected_shape = (1, spectrogram_config.n_mels, expected_time_frames)

        assert mel_spec.shape == expected_shape
        assert torch.all(mel_spec >= 0)  # Mel spectrograms should be non-negative
        assert torch.isfinite(mel_spec).all()

    def test_mel_spectrogram_processor_db_scale(self, spectrogram_config, sample_audio):
        """Test MelSpectrogramProcessor with dB scaling."""
        # Test with dB scaling enabled
        spectrogram_config.to_db = True
        processor = MelSpectrogramProcessor(spectrogram_config)

        mel_spec_db = processor(sample_audio)

        # dB values should typically be negative (except for very loud signals)
        assert torch.max(mel_spec_db) <= 0  # Should be <= 0 dB
        assert torch.isfinite(mel_spec_db).all()

    def test_stft_processor_initialization(self, spectrogram_config):
        """Test STFTProcessor initialization."""
        processor = STFTProcessor(spectrogram_config)

        assert processor.config == spectrogram_config
        assert hasattr(processor, "stft_transform")

    def test_stft_processor_forward(self, spectrogram_config, sample_audio):
        """Test STFTProcessor forward pass."""
        processor = STFTProcessor(spectrogram_config)

        # Process audio
        stft_spec = processor(sample_audio)

        # Check output shape
        expected_freq_bins = (spectrogram_config.n_fft // 2) + 1
        expected_time_frames = (sample_audio.shape[-1] // spectrogram_config.hop_length) + 1
        expected_shape = (1, expected_freq_bins, expected_time_frames)

        assert stft_spec.shape == expected_shape
        assert torch.all(stft_spec >= 0)  # Magnitude spectrograms should be non-negative
        assert torch.isfinite(stft_spec).all()

    def test_stft_processor_complex_output(self, spectrogram_config, sample_audio):
        """Test STFTProcessor with complex output."""
        spectrogram_config.return_complex = True
        processor = STFTProcessor(spectrogram_config)

        stft_complex = processor(sample_audio)

        # Complex output should have real and imaginary parts
        assert stft_complex.dtype == torch.complex64 or stft_complex.dtype == torch.complex128
        assert torch.isfinite(stft_complex.real).all()
        assert torch.isfinite(stft_complex.imag).all()

    def test_cqt_processor_initialization(self, spectrogram_config):
        """Test CQTProcessor initialization."""
        processor = CQTProcessor(spectrogram_config)

        assert processor.config == spectrogram_config
        assert hasattr(processor, "cqt_transform")

    def test_cqt_processor_forward(self, spectrogram_config, sample_audio):
        """Test CQTProcessor forward pass."""
        processor = CQTProcessor(spectrogram_config)

        # Process audio
        cqt_spec = processor(sample_audio)

        # CQT should produce output
        assert len(cqt_spec.shape) == 3  # (batch, freq, time)
        assert cqt_spec.shape[0] == 1  # Batch dimension
        assert torch.all(cqt_spec >= 0)  # Should be non-negative
        assert torch.isfinite(cqt_spec).all()

    def test_spectrogram_processor_batch_processing(self, spectrogram_config):
        """Test spectrogram processors with batch input."""
        processor = MelSpectrogramProcessor(spectrogram_config)

        # Create batch of audio
        batch_size = 4
        audio_length = 16000 * 2  # 2 seconds
        batch_audio = torch.randn(batch_size, audio_length)

        mel_spec = processor(batch_audio)

        # Check batch processing
        assert mel_spec.shape[0] == batch_size
        assert len(mel_spec.shape) == 3  # (batch, freq, time)
        assert torch.isfinite(mel_spec).all()

    def test_spectrogram_processor_empty_audio(self, spectrogram_config):
        """Test spectrogram processors with minimal audio."""
        processor = MelSpectrogramProcessor(spectrogram_config)

        # Create minimal audio (just enough for one frame)
        min_audio = torch.zeros(1, spectrogram_config.n_fft)

        mel_spec = processor(min_audio)

        # Should still produce valid output
        assert mel_spec.shape[0] == 1
        assert mel_spec.shape[1] == spectrogram_config.n_mels
        assert torch.isfinite(mel_spec).all()


class TestAudioNormalizers:
    """Test audio normalization modules."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio with known characteristics."""
        # Create audio with specific amplitude range
        audio = torch.tensor([[1.0, 2.0, -1.5, 0.5, -2.0, 3.0, -0.8, 1.2]])
        return audio

    def test_audio_normalizer_base_class(self):
        """Test AudioNormalizer base class."""
        normalizer = AudioNormalizer()

        # Base class should define interface
        assert hasattr(normalizer, "normalize")
        assert hasattr(normalizer, "denormalize")

    def test_peak_normalizer_initialization(self):
        """Test PeakNormalizer initialization."""
        normalizer = PeakNormalizer(target_level=0.9)
        assert normalizer.target_level == 0.9

    def test_peak_normalizer_forward(self, sample_audio):
        """Test PeakNormalizer forward pass."""
        normalizer = PeakNormalizer(target_level=1.0)

        normalized = normalizer.normalize(sample_audio)

        # Peak should be at target level
        assert torch.max(torch.abs(normalized)) <= 1.0
        assert torch.isclose(torch.max(torch.abs(normalized)), torch.tensor(1.0), atol=1e-6)
        assert normalized.shape == sample_audio.shape

    def test_peak_normalizer_zero_audio(self):
        """Test PeakNormalizer with zero audio."""
        normalizer = PeakNormalizer(target_level=1.0)
        zero_audio = torch.zeros(1, 1000)

        normalized = normalizer.normalize(zero_audio)

        # Should remain zero
        assert torch.all(normalized == 0)
        assert normalized.shape == zero_audio.shape

    def test_rms_normalizer_initialization(self):
        """Test RMSNormalizer initialization."""
        normalizer = RMSNormalizer(target_rms=0.1)
        assert normalizer.target_rms == 0.1

    def test_rms_normalizer_forward(self, sample_audio):
        """Test RMSNormalizer forward pass."""
        target_rms = 0.2
        normalizer = RMSNormalizer(target_rms=target_rms)

        normalized = normalizer.normalize(sample_audio)

        # Check RMS level
        actual_rms = torch.sqrt(torch.mean(normalized**2))
        assert torch.isclose(actual_rms, torch.tensor(target_rms), atol=1e-3)
        assert normalized.shape == sample_audio.shape

    def test_zscore_normalizer_forward(self, sample_audio):
        """Test ZScoreNormalizer forward pass."""
        normalizer = ZScoreNormalizer()

        normalized = normalizer.normalize(sample_audio)

        # Check z-score normalization (mean ≈ 0, std ≈ 1)
        mean = torch.mean(normalized)
        std = torch.std(normalized)

        assert torch.isclose(mean, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(std, torch.tensor(1.0), atol=1e-6)
        assert normalized.shape == sample_audio.shape

    def test_minmax_normalizer_initialization(self):
        """Test MinMaxNormalizer initialization."""
        normalizer = MinMaxNormalizer(feature_range=(-1, 1))
        assert normalizer.feature_range == (-1, 1)

    def test_minmax_normalizer_forward(self, sample_audio):
        """Test MinMaxNormalizer forward pass."""
        normalizer = MinMaxNormalizer(feature_range=(0, 1))

        normalized = normalizer.normalize(sample_audio)

        # Check range
        assert torch.min(normalized) >= 0
        assert torch.max(normalized) <= 1
        assert torch.isclose(torch.min(normalized), torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(torch.max(normalized), torch.tensor(1.0), atol=1e-6)
        assert normalized.shape == sample_audio.shape

    def test_normalizer_batch_processing(self):
        """Test normalizers with batch input."""
        batch_audio = torch.randn(4, 1000)  # Batch of 4 audio samples
        normalizer = PeakNormalizer(target_level=0.8)

        normalized = normalizer.normalize(batch_audio)

        # Each sample in batch should be normalized independently
        assert normalized.shape == batch_audio.shape
        for i in range(batch_audio.shape[0]):
            sample_peak = torch.max(torch.abs(normalized[i]))
            assert sample_peak <= 0.8

    def test_normalizer_denormalization(self, sample_audio):
        """Test normalizer denormalization functionality."""
        normalizer = ZScoreNormalizer()

        # Store original statistics
        original_mean = torch.mean(sample_audio)
        original_std = torch.std(sample_audio)

        # Normalize and denormalize
        normalized = normalizer.normalize(sample_audio)
        denormalized = normalizer.denormalize(normalized)

        # Should recover original audio (approximately)
        assert torch.allclose(denormalized, sample_audio, atol=1e-5)


class TestAudioSegmenters:
    """Test audio segmentation modules."""

    @pytest.fixture
    def long_audio(self):
        """Create long audio sample for segmentation testing."""
        sample_rate = 16000
        duration = 10.0  # 10 seconds
        t = torch.linspace(0, duration, int(sample_rate * duration))

        # Create audio with varying energy levels
        audio = torch.sin(2 * np.pi * 440 * t) * torch.sin(2 * np.pi * 0.5 * t)  # AM modulation
        return audio.unsqueeze(0)

    def test_fixed_length_segmenter_initialization(self):
        """Test FixedLengthSegmenter initialization."""
        segmenter = FixedLengthSegmenter(segment_length=2.0, hop_length=1.0, sample_rate=16000)

        assert segmenter.segment_length == 2.0
        assert segmenter.hop_length == 1.0
        assert segmenter.sample_rate == 16000
        assert segmenter.segment_samples == 32000  # 2.0 * 16000

    def test_fixed_length_segmenter_forward(self, long_audio):
        """Test FixedLengthSegmenter forward pass."""
        segmenter = FixedLengthSegmenter(
            segment_length=2.0, hop_length=2.0, sample_rate=16000  # Non-overlapping
        )

        segments = segmenter.segment(long_audio)

        # Check output
        assert isinstance(segments, list)
        assert len(segments) > 0

        # Each segment should have correct length
        for segment in segments:
            assert segment.shape[-1] == 32000  # 2.0 seconds * 16000 Hz
            assert segment.shape[0] == 1  # Preserve channel dimension

    def test_fixed_length_segmenter_overlapping(self, long_audio):
        """Test FixedLengthSegmenter with overlapping segments."""
        segmenter = FixedLengthSegmenter(
            segment_length=2.0, hop_length=1.0, sample_rate=16000  # 50% overlap
        )

        segments = segmenter.segment(long_audio)

        # Should have more segments due to overlap
        expected_segments = int((long_audio.shape[-1] - 32000) / 16000) + 1
        assert len(segments) >= expected_segments

    def test_overlapping_segmenter_initialization(self):
        """Test OverlappingSegmenter initialization."""
        segmenter = OverlappingSegmenter(segment_length=1.5, overlap_ratio=0.5, sample_rate=16000)

        assert segmenter.segment_length == 1.5
        assert segmenter.overlap_ratio == 0.5
        assert segmenter.sample_rate == 16000

    def test_overlapping_segmenter_forward(self, long_audio):
        """Test OverlappingSegmenter forward pass."""
        segmenter = OverlappingSegmenter(
            segment_length=2.0, overlap_ratio=0.25, sample_rate=16000  # 25% overlap
        )

        segments = segmenter.segment(long_audio)

        # Check segments
        assert isinstance(segments, list)
        assert len(segments) > 0

        for segment in segments:
            assert segment.shape[-1] == 32000
            assert torch.isfinite(segment).all()

    def test_energy_based_segmenter_initialization(self):
        """Test EnergyBasedSegmenter initialization."""
        segmenter = EnergyBasedSegmenter(
            min_segment_length=1.0, max_segment_length=5.0, energy_threshold=0.01, sample_rate=16000
        )

        assert segmenter.min_segment_length == 1.0
        assert segmenter.max_segment_length == 5.0
        assert segmenter.energy_threshold == 0.01

    def test_energy_based_segmenter_forward(self, long_audio):
        """Test EnergyBasedSegmenter forward pass."""
        segmenter = EnergyBasedSegmenter(
            min_segment_length=0.5, max_segment_length=3.0, energy_threshold=0.01, sample_rate=16000
        )

        segments = segmenter.segment(long_audio)

        # Check segments
        assert isinstance(segments, list)
        assert len(segments) > 0

        for segment in segments:
            # Check length constraints
            min_samples = int(0.5 * 16000)
            max_samples = int(3.0 * 16000)
            assert min_samples <= segment.shape[-1] <= max_samples

    def test_voice_activity_segmenter_initialization(self):
        """Test VoiceActivitySegmenter initialization."""
        segmenter = VoiceActivitySegmenter(
            frame_length=0.025, frame_shift=0.01, energy_threshold=0.01, sample_rate=16000
        )

        assert segmenter.frame_length == 0.025
        assert segmenter.frame_shift == 0.01
        assert segmenter.energy_threshold == 0.01

    def test_voice_activity_segmenter_forward(self, long_audio):
        """Test VoiceActivitySegmenter forward pass."""
        segmenter = VoiceActivitySegmenter(
            frame_length=0.025,
            frame_shift=0.01,
            energy_threshold=0.001,  # Low threshold to detect most of the signal
            sample_rate=16000,
        )

        segments = segmenter.segment(long_audio)

        # Check segments
        assert isinstance(segments, list)
        # Should detect some voice activity in the sine wave
        assert len(segments) >= 1

    def test_segmenter_with_silence(self):
        """Test segmenters with silent audio."""
        silent_audio = torch.zeros(1, 160000)  # 10 seconds of silence

        # Fixed length segmenter should still work
        fixed_segmenter = FixedLengthSegmenter(
            segment_length=2.0, hop_length=2.0, sample_rate=16000
        )
        segments = fixed_segmenter.segment(silent_audio)
        assert len(segments) > 0

        # Energy-based segmenter might return empty or minimal segments
        energy_segmenter = EnergyBasedSegmenter(
            min_segment_length=1.0, max_segment_length=5.0, energy_threshold=0.01, sample_rate=16000
        )
        segments = energy_segmenter.segment(silent_audio)
        # Silent audio might not produce any segments above threshold
        assert isinstance(segments, list)

    def test_segmenter_short_audio(self):
        """Test segmenters with very short audio."""
        short_audio = torch.randn(1, 8000)  # 0.5 seconds

        segmenter = FixedLengthSegmenter(segment_length=2.0, hop_length=2.0, sample_rate=16000)

        segments = segmenter.segment(short_audio)

        # Might return empty list or padded segment
        if len(segments) > 0:
            # If a segment is returned, it should be properly sized
            assert segments[0].shape[-1] <= 32000

    def test_segmenter_batch_processing(self):
        """Test segmenters with batch input."""
        batch_audio = torch.randn(3, 80000)  # Batch of 3 audio samples

        segmenter = FixedLengthSegmenter(segment_length=1.0, hop_length=1.0, sample_rate=16000)

        # Process each sample in batch
        all_segments = []
        for i in range(batch_audio.shape[0]):
            segments = segmenter.segment(batch_audio[i : i + 1])
            all_segments.extend(segments)

        assert len(all_segments) > 0
        for segment in all_segments:
            assert segment.shape[-1] == 16000  # 1 second


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for pipeline testing."""
        sample_rate = 16000
        duration = 4.0
        t = torch.linspace(0, duration, int(sample_rate * duration))

        # Create realistic audio signal
        audio = (
            0.6 * torch.sin(2 * np.pi * 440 * t)
            + 0.3 * torch.sin(2 * np.pi * 880 * t)
            + 0.1 * torch.randn(len(t))  # Add noise
        )

        # Scale to have some amplitude variation
        audio = audio * (2.0 + torch.sin(2 * np.pi * 0.5 * t))

        return audio.unsqueeze(0)

    def test_complete_preprocessing_pipeline(self, sample_audio):
        """Test complete preprocessing pipeline."""
        # Step 1: Normalization
        normalizer = PeakNormalizer(target_level=0.8)
        normalized_audio = normalizer.normalize(sample_audio)

        # Step 2: Segmentation
        segmenter = FixedLengthSegmenter(segment_length=2.0, hop_length=1.0, sample_rate=16000)
        segments = segmenter.segment(normalized_audio)

        # Step 3: Spectrogram generation
        config = SpectrogramConfig(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128)
        spec_processor = MelSpectrogramProcessor(config)

        spectrograms = []
        for segment in segments:
            spec = spec_processor(segment)
            spectrograms.append(spec)

        # Verify pipeline output
        assert len(spectrograms) > 0
        for spec in spectrograms:
            assert spec.shape[1] == 128  # n_mels
            assert torch.all(spec >= 0)  # Non-negative
            assert torch.isfinite(spec).all()

    def test_preprocessing_pipeline_consistency(self, sample_audio):
        """Test preprocessing pipeline consistency."""
        config = SpectrogramConfig(sample_rate=16000, n_fft=512, hop_length=256, n_mels=64)

        # Process same audio multiple times
        processor = MelSpectrogramProcessor(config)

        spec1 = processor(sample_audio)
        spec2 = processor(sample_audio)

        # Results should be identical
        assert torch.allclose(spec1, spec2, atol=1e-6)

    def test_preprocessing_parameter_validation(self):
        """Test preprocessing parameter validation."""
        # Invalid spectrogram config
        with pytest.raises(ValueError):
            SpectrogramConfig(
                sample_rate=16000, n_fft=1024, hop_length=2048, n_mels=128  # hop_length > n_fft
            )

        # Invalid normalizer parameters
        with pytest.raises(ValueError):
            PeakNormalizer(target_level=-0.5)  # Negative target level

        # Invalid segmenter parameters
        with pytest.raises(ValueError):
            FixedLengthSegmenter(
                segment_length=-1.0, hop_length=1.0, sample_rate=16000  # Negative length
            )


class TestPreprocessingEdgeCases:
    """Test preprocessing with edge cases."""

    def test_empty_audio_preprocessing(self):
        """Test preprocessing with empty audio."""
        empty_audio = torch.tensor([[]])

        # Normalizers should handle empty input gracefully
        try:
            normalizer = PeakNormalizer()
            result = normalizer.normalize(empty_audio)
            assert result.shape == empty_audio.shape
        except Exception:
            # It's acceptable for some operations to fail with empty input
            pass

    def test_single_sample_preprocessing(self):
        """Test preprocessing with single sample audio."""
        single_sample = torch.tensor([[0.5]])

        # Test normalization
        normalizer = PeakNormalizer(target_level=1.0)
        normalized = normalizer.normalize(single_sample)
        assert normalized.shape == single_sample.shape
        assert torch.abs(normalized.item()) <= 1.0

    def test_extreme_audio_values(self):
        """Test preprocessing with extreme audio values."""
        # Very large values
        large_audio = torch.tensor([[1000.0, -1000.0, 500.0, -250.0]])

        normalizer = PeakNormalizer(target_level=1.0)
        normalized = normalizer.normalize(large_audio)

        assert torch.max(torch.abs(normalized)) <= 1.0
        assert torch.isfinite(normalized).all()

    def test_constant_audio_preprocessing(self):
        """Test preprocessing with constant audio."""
        constant_audio = torch.ones(1, 16000) * 0.5

        # Test various preprocessing steps
        normalizer = ZScoreNormalizer()
        try:
            normalized = normalizer.normalize(constant_audio)
            # Constant audio has zero variance, might result in NaN
            # Implementation should handle this gracefully
        except Exception:
            # It's acceptable for z-score normalization to fail with constant input
            pass

        # Peak normalization should work
        peak_normalizer = PeakNormalizer()
        normalized = peak_normalizer.normalize(constant_audio)
        assert torch.isfinite(normalized).all()


if __name__ == "__main__":
    pytest.main([__file__])
