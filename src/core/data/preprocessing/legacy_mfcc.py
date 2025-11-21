# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Legacy MFCC Feature Extraction
===============================
MFCC feature extraction for legacy CNN/CRNN models.
Provides handcrafted features: MFCC + delta + delta-delta
"""

import numpy as np
import librosa
from typing import Optional, Tuple


class MFCCExtractor:
    """Extract MFCC features with delta and delta-delta components."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        use_deltas: bool = True,
        use_delta_deltas: bool = True,
        delta_window: int = 3,
        normalize: bool = True,
        normalization_type: str = "zscore",
    ):
        """
        Initialize MFCC extractor.

        Args:
            sample_rate: Sampling rate in Hz
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Number of samples between frames
            f_min: Minimum frequency
            f_max: Maximum frequency (None = Nyquist)
            use_deltas: Include delta (velocity) features
            use_delta_deltas: Include delta-delta (acceleration) features
            delta_window: Window for computing delta features
            normalize: Normalize features
            normalization_type: "zscore" or "minmax"
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.use_deltas = use_deltas
        self.use_delta_deltas = use_delta_deltas
        sanitized_window = max(3, delta_window)
        if sanitized_window % 2 == 0:
            sanitized_window += 1
        self.delta_window = sanitized_window
        self.normalize = normalize
        self.normalization_type = normalization_type

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features with optional delta and delta-delta.

        Args:
            audio: Audio signal as numpy array

        Returns:
            Feature matrix of shape (n_features, time_steps)
            where n_features = n_mfcc * (1 + use_deltas + use_delta_deltas)
        """
        # Compute MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.f_min,
            fmax=self.f_max,
        )  # (n_mfcc, time)

        # Normalize MFCC per coefficient
        if self.normalize:
            mfcc = self._normalize_features(mfcc)

        # Compute delta and delta-delta if requested
        if self.use_deltas:
            delta = librosa.feature.delta(mfcc, width=self.delta_window)
            if self.normalize:
                delta = self._normalize_features(delta)
        else:
            delta = None

        if self.use_delta_deltas:
            delta_delta = librosa.feature.delta(
                mfcc, order=2, width=self.delta_window
            )
            if self.normalize:
                delta_delta = self._normalize_features(delta_delta)
        else:
            delta_delta = None

        # Stack features
        features = [mfcc]
        if delta is not None:
            features.append(delta)
        if delta_delta is not None:
            features.append(delta_delta)

        # Stack along feature axis: (n_mfcc*channels, time)
        stacked = np.vstack(features)

        return stacked

    def extract_with_shape(
        self, audio: np.ndarray, target_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Extract MFCC and pad/crop to target shape.

        Args:
            audio: Audio signal
            target_shape: Target shape (n_features, n_frames)
                          If None, returns extracted features as-is

        Returns:
            Feature matrix of shape target_shape or extracted shape
        """
        features = self.extract(audio)

        if target_shape is None:
            return features

        # Pad or crop to target shape
        n_features, n_frames = target_shape
        current_frames = features.shape[1]

        if current_frames < n_frames:
            # Pad with zeros
            padding = ((0, 0), (0, n_frames - current_frames))
            features = np.pad(features, padding, mode='constant', constant_values=0)
        elif current_frames > n_frames:
            # Crop to target length
            features = features[:, :n_frames]

        return features[:n_features, :]

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        """
        Normalize features per coefficient (z-score normalization).

        Args:
            features: Feature matrix (n_features, time)

        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-10  # Add epsilon
        return (features - mean) / std

    @staticmethod
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std for normalization (useful for datasets).

        Args:
            features: Feature matrix or batch

        Returns:
            Tuple of (mean, std)
        """
        mean = np.mean(features, axis=-1, keepdims=True)
        std = np.std(features, axis=-1, keepdims=True)
        return mean, std


class LegacyMFCCPreprocessor:
    """Preprocessor for legacy models combining audio loading and MFCC extraction."""

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mfcc: int = 40,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        use_deltas: bool = True,
        use_delta_deltas: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize legacy MFCC preprocessor.

        Args:
            sample_rate: Target sample rate
            duration: Target audio duration in seconds
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length
            use_deltas: Include delta features
            use_delta_deltas: Include delta-delta features
            normalize: Normalize features
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.use_deltas = use_deltas
        self.use_delta_deltas = use_delta_deltas

        # Initialize extractor
        self.extractor = MFCCExtractor(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            use_deltas=use_deltas,
            use_delta_deltas=use_delta_deltas,
            normalize=normalize,
        )

        # Compute target shape dynamically using a dummy signal
        dummy_audio = np.zeros(self.n_samples, dtype=np.float32)
        dummy_features = self.extractor.extract(dummy_audio)
        self.target_shape = dummy_features.shape

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio to MFCC features.

        Args:
            audio: Audio signal (mono)

        Returns:
            MFCC features of shape (n_features, n_frames)
        """
        # Resample if needed
        if hasattr(audio, 'sr') and audio.sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=audio.sr, target_sr=self.sample_rate)

        # Pad or crop to target duration
        if len(audio) < self.n_samples:
            audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
        else:
            audio = audio[: self.n_samples]

        # Extract MFCC with target shape
        features = self.extractor.extract_with_shape(audio, self.target_shape)

        # Reshape for model: (n_features, n_frames) → (n_mfcc, n_frames, channels)
        n_mfcc = 40
        n_frames = features.shape[1]
        channels = 1 + self.use_deltas + self.use_delta_deltas

        # Reshape: (n_features, n_frames) → (n_mfcc, n_frames, channels)
        reshaped = features.reshape(n_mfcc, n_frames, channels)

        return reshaped

    def process_file(self, file_path: str) -> np.ndarray:
        """
        Process audio file to MFCC features.

        Args:
            file_path: Path to audio file

        Returns:
            MFCC features of shape (n_mfcc, n_frames, channels)
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return self.process_audio(audio)

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Get output shape (n_mfcc, n_frames, channels)."""
        n_mfcc = 40
        n_frames = self.target_shape[1]
        channels = 1 + self.use_deltas + self.use_delta_deltas
        return (n_mfcc, n_frames, channels)
