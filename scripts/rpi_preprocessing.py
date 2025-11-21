"""
Audio preprocessing module for Raspberry Pi deployment.

This module provides efficient audio preprocessing for real-time
military vehicle sound detection on Raspberry Pi 5.

Features:
- Audio loading and resampling
- Mel-spectrogram generation
- Normalization matching training pipeline
- Optimized for ARM CPU

Author: SereneSense Team
Date: 2025-11-21
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Audio preprocessor for Raspberry Pi deployment.

    Matches the preprocessing pipeline used during training.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 10.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        power: float = 2.0
    ):
        """
        Initialize audio preprocessor.

        Args:
            sample_rate: Target sample rate (Hz)
            duration: Target audio duration (seconds)
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length between frames
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz)
            power: Power for spectrogram (2.0 = power, 1.0 = magnitude)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.power = power

        # Calculate expected number of samples
        self.expected_samples = int(sample_rate * duration)

        # Normalization parameters (from ImageNet, used in training)
        self.mean = 0.485
        self.std = 0.229

        print(f"AudioPreprocessor initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {duration} sec ({self.expected_samples} samples)")
        print(f"  Mel bins: {n_mels}")
        print(f"  FFT size: {n_fft}")
        print(f"  Hop length: {hop_length}")
        print(f"  Frequency range: {fmin}-{fmax} Hz")

    def load_audio(
        self,
        audio_path: str,
        offset: float = 0.0
    ) -> np.ndarray:
        """
        Load audio file and prepare for processing.

        Args:
            audio_path: Path to audio file
            offset: Time offset to start reading (seconds)

        Returns:
            Audio waveform as numpy array (mono, normalized)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                offset=offset,
                duration=self.duration
            )

            # Adjust length to exactly expected samples
            audio = self._adjust_audio_length(audio)

            return audio

        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}")

    def load_audio_from_array(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Load audio from numpy array (for real-time capture).

        Args:
            audio: Audio waveform array
            sr: Sample rate of input audio

        Returns:
            Processed audio waveform (mono, resampled, normalized)
        """
        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sample_rate
            )

        # Adjust length
        audio = self._adjust_audio_length(audio)

        return audio

    def _adjust_audio_length(self, audio: np.ndarray) -> np.ndarray:
        """
        Adjust audio to expected length (pad or crop).

        Args:
            audio: Input audio waveform

        Returns:
            Audio waveform with correct length
        """
        current_samples = len(audio)

        if current_samples < self.expected_samples:
            # Pad with zeros
            padding = self.expected_samples - current_samples
            audio = np.pad(audio, (0, padding), mode='constant')

        elif current_samples > self.expected_samples:
            # Crop from center
            start = (current_samples - self.expected_samples) // 2
            audio = audio[start:start + self.expected_samples]

        return audio

    def generate_mel_spectrogram(
        self,
        audio: np.ndarray
    ) -> np.ndarray:
        """
        Generate mel-spectrogram from audio waveform.

        Args:
            audio: Audio waveform (mono, normalized)

        Returns:
            Mel-spectrogram as numpy array (n_mels, time_frames)
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    def normalize_spectrogram(
        self,
        mel_spec: np.ndarray
    ) -> np.ndarray:
        """
        Normalize mel-spectrogram to match training distribution.

        Args:
            mel_spec: Mel-spectrogram (n_mels, time_frames)

        Returns:
            Normalized mel-spectrogram
        """
        # Min-max normalization to [0, 1]
        mel_min = mel_spec.min()
        mel_max = mel_spec.max()

        if mel_max > mel_min:
            mel_spec_norm = (mel_spec - mel_min) / (mel_max - mel_min)
        else:
            mel_spec_norm = np.zeros_like(mel_spec)

        # Standardize using ImageNet statistics (used in training)
        mel_spec_norm = (mel_spec_norm - self.mean) / self.std

        return mel_spec_norm

    def resize_spectrogram(
        self,
        mel_spec: np.ndarray,
        target_size: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Resize spectrogram to target size using bilinear interpolation.

        Args:
            mel_spec: Mel-spectrogram (n_mels, time_frames)
            target_size: Target size (height, width)

        Returns:
            Resized spectrogram
        """
        from scipy.ndimage import zoom

        # Calculate zoom factors
        zoom_factors = (
            target_size[0] / mel_spec.shape[0],
            target_size[1] / mel_spec.shape[1]
        )

        # Resize using bilinear interpolation
        mel_spec_resized = zoom(mel_spec, zoom_factors, order=1)

        return mel_spec_resized

    def preprocess(
        self,
        audio_input,
        input_type: str = 'file'
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline: audio → spectrogram → normalized.

        Args:
            audio_input: Either file path (str) or audio array (np.ndarray)
            input_type: 'file' or 'array'

        Returns:
            Preprocessed spectrogram ready for model input
            Shape: (1, 1, 128, 128) - (batch, channels, height, width)
        """
        # Step 1: Load audio
        if input_type == 'file':
            audio = self.load_audio(audio_input)
        elif input_type == 'array':
            audio, sr = audio_input
            audio = self.load_audio_from_array(audio, sr)
        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        # Step 2: Generate mel-spectrogram
        mel_spec = self.generate_mel_spectrogram(audio)

        # Step 3: Resize to 128x128
        mel_spec = self.resize_spectrogram(mel_spec, target_size=(128, 128))

        # Step 4: Normalize
        mel_spec = self.normalize_spectrogram(mel_spec)

        # Step 5: Add batch and channel dimensions
        mel_spec = mel_spec[np.newaxis, np.newaxis, :, :]  # (1, 1, 128, 128)

        # Convert to float32 for ONNX Runtime
        mel_spec = mel_spec.astype(np.float32)

        return mel_spec


# Convenience function for quick preprocessing
def preprocess_audio_file(audio_path: str) -> np.ndarray:
    """
    Quick preprocessing of audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Preprocessed spectrogram (1, 1, 128, 128)
    """
    preprocessor = AudioPreprocessor()
    return preprocessor.preprocess(audio_path, input_type='file')


def preprocess_audio_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Quick preprocessing of audio array.

    Args:
        audio: Audio waveform
        sr: Sample rate

    Returns:
        Preprocessed spectrogram (1, 1, 128, 128)
    """
    preprocessor = AudioPreprocessor()
    return preprocessor.preprocess((audio, sr), input_type='array')


if __name__ == "__main__":
    """Test preprocessing pipeline."""

    print("=" * 70)
    print("Audio Preprocessing Test")
    print("=" * 70)

    # Create preprocessor
    preprocessor = AudioPreprocessor()

    # Test with dummy audio
    print("\nTesting with dummy audio...")
    dummy_audio = np.random.randn(16000 * 10)  # 10 seconds @ 16kHz

    mel_spec = preprocessor.preprocess((dummy_audio, 16000), input_type='array')

    print(f"\nOutput spectrogram shape: {mel_spec.shape}")
    print(f"Expected shape: (1, 1, 128, 128)")
    print(f"Data type: {mel_spec.dtype}")
    print(f"Value range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")

    assert mel_spec.shape == (1, 1, 128, 128), "Shape mismatch!"
    assert mel_spec.dtype == np.float32, "Data type mismatch!"

    print("\n✓ Preprocessing test passed!")
    print("=" * 70)
