"""
Feature Extractor for SereneSense

This module provides comprehensive audio feature extraction capabilities
for the SereneSense military vehicle sound detection system.

Features:
- Mel-scale spectrograms for transformer models
- MFCC features for classical ML approaches
- Spectral features (centroid, bandwidth, rolloff, etc.)
- Chroma and tonnetz features
- Raw STFT spectrograms
- Time-domain features (ZCR, RMS, etc.)
- Configurable feature extraction pipelines
- GPU acceleration support

Example:
    >>> from core.core.feature_extractor import FeatureExtractor
    >>> 
    >>> # Initialize extractor
    >>> config = {"spectrogram": {"n_mels": 128, "n_fft": 1024}}
    >>> extractor = FeatureExtractor(config)
    >>> 
    >>> # Extract features
    >>> features = extractor.extract(audio_data, sample_rate=16000)
"""

import numpy as np
import torch
import torchaudio
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party imports
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available for advanced features", ImportWarning)

try:
    import scipy.signal
    import scipy.stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type definitions
AudioData = Union[np.ndarray, torch.Tensor]
FeatureData = Union[np.ndarray, torch.Tensor]


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    feature_type: str
    parameters: Dict[str, Any]
    normalize: bool = True
    cache_key: Optional[str] = None


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, audio: AudioData, sample_rate: int) -> FeatureData:
        """Extract features from audio data."""
        pass

    @abstractmethod
    def get_feature_shape(self, audio_length: int, sample_rate: int) -> Tuple[int, ...]:
        """Get expected feature shape for given audio length."""
        pass


class SpectrogramExtractor(BaseFeatureExtractor):
    """
    Extract mel-scale spectrograms optimized for transformer models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize spectrogram extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # STFT parameters
        self.n_fft = config.get("n_fft", 1024)
        self.hop_length = config.get("hop_length", 160)  # 10ms at 16kHz
        self.win_length = config.get("win_length", 1024)
        self.window = config.get("window", "hann")

        # Mel filterbank parameters
        self.n_mels = config.get("n_mels", 128)
        self.f_min = config.get("f_min", 50)
        self.f_max = config.get("f_max", 8000)
        self.mel_scale = config.get("mel_scale", "htk")

        # Power spectrum parameters
        self.power = config.get("power", 2.0)
        self.normalized = config.get("normalized", False)

        # Log transformation
        self.log_mel = config.get("log_mel", True)
        self.log_offset = config.get("log_offset", 1e-6)

        # Normalization
        self.normalize = config.get("normalize", True)
        self.norm_type = config.get("norm_type", "instance")  # instance, batch, global

        logger.debug(
            f"SpectrogramExtractor initialized: {self.n_mels} mels, "
            f"{self.n_fft} FFT, {self.hop_length} hop"
        )

    def extract(self, audio: AudioData, sample_rate: int) -> torch.Tensor:
        """
        Extract mel-scale spectrogram features.

        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio

        Returns:
            Mel-spectrogram features as tensor
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        # Ensure 1D for spectrogram computation
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        # Create mel spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=self.power,
            normalized=self.normalized,
            mel_scale=self.mel_scale,
        )

        # Extract mel spectrogram
        mel_spec = mel_transform(audio_tensor)

        # Apply log transformation
        if self.log_mel:
            mel_spec = torch.log(mel_spec + self.log_offset)

        # Normalize if requested
        if self.normalize:
            mel_spec = self._normalize_spectrogram(mel_spec)

        return mel_spec

    def _normalize_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrogram using specified method.

        Args:
            spec: Input spectrogram

        Returns:
            Normalized spectrogram
        """
        if self.norm_type == "instance":
            # Instance normalization (per spectrogram)
            mean = torch.mean(spec)
            std = torch.std(spec)
            spec = (spec - mean) / (std + 1e-8)

        elif self.norm_type == "global":
            # Global normalization with fixed statistics
            # These are typical values for log mel spectrograms
            mean = -4.2677393
            std = 4.5689974
            spec = (spec - mean) / std

        elif self.norm_type == "min_max":
            # Min-max normalization
            min_val = torch.min(spec)
            max_val = torch.max(spec)
            spec = (spec - min_val) / (max_val - min_val + 1e-8)

        return spec

    def get_feature_shape(self, audio_length: int, sample_rate: int) -> Tuple[int, int]:
        """
        Get expected mel spectrogram shape.

        Args:
            audio_length: Length of audio in samples
            sample_rate: Sample rate

        Returns:
            Tuple of (n_mels, time_frames)
        """
        time_frames = 1 + (audio_length - self.n_fft) // self.hop_length
        return (self.n_mels, time_frames)


class MFCCExtractor(BaseFeatureExtractor):
    """
    Extract Mel-Frequency Cepstral Coefficients.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MFCC extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # MFCC parameters
        self.n_mfcc = config.get("n_mfcc", 13)
        self.n_mels = config.get("n_mels", 128)
        self.n_fft = config.get("n_fft", 1024)
        self.hop_length = config.get("hop_length", 160)

        # Delta features
        self.delta = config.get("delta", False)
        self.delta_delta = config.get("delta_delta", False)

        logger.debug(f"MFCCExtractor initialized: {self.n_mfcc} coefficients")

    def extract(self, audio: AudioData, sample_rate: int) -> torch.Tensor:
        """
        Extract MFCC features.

        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio

        Returns:
            MFCC features as tensor
        """
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        # Create MFCC transform
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": self.n_fft, "hop_length": self.hop_length, "n_mels": self.n_mels},
        )

        # Extract MFCC
        mfcc = mfcc_transform(audio_tensor)

        # Add delta features if requested
        features = [mfcc]

        if self.delta:
            delta_mfcc = torchaudio.functional.compute_deltas(mfcc)
            features.append(delta_mfcc)

        if self.delta_delta:
            if not self.delta:
                delta_mfcc = torchaudio.functional.compute_deltas(mfcc)
            delta_delta_mfcc = torchaudio.functional.compute_deltas(delta_mfcc)
            features.append(delta_delta_mfcc)

        # Concatenate all features
        if len(features) > 1:
            mfcc = torch.cat(features, dim=0)

        return mfcc

    def get_feature_shape(self, audio_length: int, sample_rate: int) -> Tuple[int, int]:
        """Get expected MFCC shape."""
        time_frames = 1 + (audio_length - self.n_fft) // self.hop_length

        # Calculate total coefficients
        total_coeffs = self.n_mfcc
        if self.delta:
            total_coeffs += self.n_mfcc
        if self.delta_delta:
            total_coeffs += self.n_mfcc

        return (total_coeffs, time_frames)


class SpectralFeaturesExtractor(BaseFeatureExtractor):
    """
    Extract various spectral features using librosa.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize spectral features extractor.

        Args:
            config: Configuration dictionary
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required for spectral features")

        self.config = config
        self.features = config.get(
            "features",
            [
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "zero_crossing_rate",
                "rms",
            ],
        )

        # STFT parameters
        self.n_fft = config.get("n_fft", 1024)
        self.hop_length = config.get("hop_length", 160)

        logger.debug(f"SpectralFeaturesExtractor initialized: {len(self.features)} features")

    def extract(self, audio: AudioData, sample_rate: int) -> torch.Tensor:
        """
        Extract spectral features.

        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio

        Returns:
            Spectral features as tensor
        """
        # Convert to numpy for librosa
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        features = []

        for feature_name in self.features:
            if feature_name == "spectral_centroid":
                feature = librosa.feature.spectral_centroid(
                    y=audio_np, sr=sample_rate, hop_length=self.hop_length
                )[0]

            elif feature_name == "spectral_bandwidth":
                feature = librosa.feature.spectral_bandwidth(
                    y=audio_np, sr=sample_rate, hop_length=self.hop_length
                )[0]

            elif feature_name == "spectral_rolloff":
                feature = librosa.feature.spectral_rolloff(
                    y=audio_np, sr=sample_rate, hop_length=self.hop_length
                )[0]

            elif feature_name == "zero_crossing_rate":
                feature = librosa.feature.zero_crossing_rate(
                    y=audio_np, hop_length=self.hop_length
                )[0]

            elif feature_name == "rms":
                feature = librosa.feature.rms(y=audio_np, hop_length=self.hop_length)[0]

            elif feature_name == "spectral_contrast":
                feature = librosa.feature.spectral_contrast(
                    y=audio_np, sr=sample_rate, hop_length=self.hop_length
                )
                feature = feature.mean(axis=0)  # Average across bands

            else:
                logger.warning(f"Unknown spectral feature: {feature_name}")
                continue

            features.append(feature)

        if not features:
            raise ValueError("No valid spectral features extracted")

        # Stack features
        feature_array = np.stack(features, axis=0)

        return torch.from_numpy(feature_array).float()

    def get_feature_shape(self, audio_length: int, sample_rate: int) -> Tuple[int, int]:
        """Get expected spectral features shape."""
        time_frames = 1 + (audio_length - self.n_fft) // self.hop_length
        return (len(self.features), time_frames)


class ChromaExtractor(BaseFeatureExtractor):
    """
    Extract chroma features for harmonic analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize chroma extractor.

        Args:
            config: Configuration dictionary
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required for chroma features")

        self.config = config
        self.n_chroma = config.get("n_chroma", 12)
        self.hop_length = config.get("hop_length", 160)

        logger.debug(f"ChromaExtractor initialized: {self.n_chroma} chroma bins")

    def extract(self, audio: AudioData, sample_rate: int) -> torch.Tensor:
        """
        Extract chroma features.

        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio

        Returns:
            Chroma features as tensor
        """
        # Convert to numpy for librosa
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio_np, sr=sample_rate, hop_length=self.hop_length, n_chroma=self.n_chroma
        )

        return torch.from_numpy(chroma).float()

    def get_feature_shape(self, audio_length: int, sample_rate: int) -> Tuple[int, int]:
        """Get expected chroma shape."""
        time_frames = 1 + (audio_length - 1024) // self.hop_length  # Default n_fft=1024
        return (self.n_chroma, time_frames)


class FeatureExtractor:
    """
    Main feature extraction pipeline supporting multiple feature types.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.extractors: Dict[str, BaseFeatureExtractor] = {}

        # Initialize extractors based on configuration
        self._setup_extractors()

        # Feature combination settings
        self.combine_features = config.get("combine_features", False)
        self.feature_weights = config.get("feature_weights", {})

        # Caching
        self.enable_cache = config.get("enable_cache", False)
        self._feature_cache: Dict[str, torch.Tensor] = {}

        logger.info(f"FeatureExtractor initialized with {len(self.extractors)} extractors")

    def _setup_extractors(self) -> None:
        """Setup feature extractors based on configuration."""

        # Mel spectrogram (primary for transformers)
        if "spectrogram" in self.config:
            self.extractors["spectrogram"] = SpectrogramExtractor(self.config["spectrogram"])

        # MFCC features
        if "mfcc" in self.config:
            self.extractors["mfcc"] = MFCCExtractor(self.config["mfcc"])

        # Spectral features
        if "spectral" in self.config:
            self.extractors["spectral"] = SpectralFeaturesExtractor(self.config["spectral"])

        # Chroma features
        if "chroma" in self.config:
            self.extractors["chroma"] = ChromaExtractor(self.config["chroma"])

        if not self.extractors:
            # Default to mel spectrogram
            logger.warning("No extractors configured, using default mel spectrogram")
            self.extractors["spectrogram"] = SpectrogramExtractor(
                {"n_mels": 128, "n_fft": 1024, "hop_length": 160}
            )

    def extract(
        self, audio: AudioData, sample_rate: int, feature_types: Optional[List[str]] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from audio data.

        Args:
            audio: Input audio data
            sample_rate: Sample rate of audio
            feature_types: List of feature types to extract (None for all)

        Returns:
            Features as tensor or dictionary of feature tensors
        """
        if feature_types is None:
            feature_types = list(self.extractors.keys())

        features = {}

        for feature_type in feature_types:
            if feature_type not in self.extractors:
                logger.warning(f"Unknown feature type: {feature_type}")
                continue

            # Check cache
            cache_key = None
            if self.enable_cache:
                cache_key = self._get_cache_key(audio, sample_rate, feature_type)
                if cache_key in self._feature_cache:
                    features[feature_type] = self._feature_cache[cache_key]
                    continue

            try:
                # Extract features
                feature_data = self.extractors[feature_type].extract(audio, sample_rate)
                features[feature_type] = feature_data

                # Cache if enabled
                if self.enable_cache and cache_key:
                    self._feature_cache[cache_key] = feature_data

            except Exception as e:
                logger.error(f"Failed to extract {feature_type} features: {e}")
                continue

        if not features:
            raise RuntimeError("No features were successfully extracted")

        # Return single tensor if only one feature type, otherwise dictionary
        if len(features) == 1:
            return next(iter(features.values()))
        elif self.combine_features:
            return self._combine_features(features)
        else:
            return features

    def _combine_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine multiple feature types into a single tensor.

        Args:
            features: Dictionary of feature tensors

        Returns:
            Combined feature tensor
        """
        # Find common time dimension
        time_dims = [feat.shape[-1] for feat in features.values()]
        min_time = min(time_dims)

        combined = []

        for feature_type, feature_data in features.items():
            # Truncate to minimum time dimension
            if feature_data.shape[-1] > min_time:
                feature_data = feature_data[..., :min_time]

            # Apply feature weights if specified
            weight = self.feature_weights.get(feature_type, 1.0)
            feature_data = feature_data * weight

            combined.append(feature_data)

        # Concatenate along feature dimension
        return torch.cat(combined, dim=0)

    def _get_cache_key(self, audio: AudioData, sample_rate: int, feature_type: str) -> str:
        """Generate cache key for features."""
        # Simple hash-based cache key
        if isinstance(audio, torch.Tensor):
            audio_hash = hash(audio.data.tobytes())
        else:
            audio_hash = hash(audio.tobytes())

        return f"{feature_type}_{audio_hash}_{sample_rate}"

    def extract_batch(
        self, audio_batch: torch.Tensor, sample_rate: int, feature_types: Optional[List[str]] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from a batch of audio data.

        Args:
            audio_batch: Batch of audio data (batch_size, channels, time)
            sample_rate: Sample rate of audio
            feature_types: List of feature types to extract

        Returns:
            Batch of features
        """
        batch_size = audio_batch.shape[0]

        # Extract features for each sample in batch
        batch_features = []

        for i in range(batch_size):
            audio_sample = audio_batch[i]
            features = self.extract(audio_sample, sample_rate, feature_types)
            batch_features.append(features)

        # Stack into batch
        if isinstance(batch_features[0], torch.Tensor):
            return torch.stack(batch_features)
        else:
            # Dictionary of features
            result = {}
            for key in batch_features[0].keys():
                result[key] = torch.stack([f[key] for f in batch_features])
            return result

    def get_feature_shapes(self, audio_length: int, sample_rate: int) -> Dict[str, Tuple[int, ...]]:
        """
        Get expected feature shapes for given audio length.

        Args:
            audio_length: Length of audio in samples
            sample_rate: Sample rate

        Returns:
            Dictionary mapping feature types to their shapes
        """
        shapes = {}

        for feature_type, extractor in self.extractors.items():
            try:
                shape = extractor.get_feature_shape(audio_length, sample_rate)
                shapes[feature_type] = shape
            except Exception as e:
                logger.warning(f"Failed to get shape for {feature_type}: {e}")

        return shapes

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._feature_cache.clear()
        logger.debug("Feature cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about feature cache."""
        return {
            "enabled": self.enable_cache,
            "size": len(self._feature_cache),
            "memory_usage": sum(
                feat.element_size() * feat.nelement() for feat in self._feature_cache.values()
            )
            / (1024**2),  # MB
        }

    def visualize_features(
        self,
        features: Union[torch.Tensor, Dict[str, torch.Tensor]],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize extracted features.

        Args:
            features: Feature tensor(s) to visualize
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt

            if isinstance(features, dict):
                # Multiple feature types
                n_features = len(features)
                fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))

                if n_features == 1:
                    axes = [axes]

                for i, (feature_type, feature_data) in enumerate(features.items()):
                    if isinstance(feature_data, torch.Tensor):
                        feature_data = feature_data.cpu().numpy()

                    im = axes[i].imshow(feature_data, aspect="auto", origin="lower")
                    axes[i].set_title(f"{feature_type} Features")
                    axes[i].set_xlabel("Time")
                    axes[i].set_ylabel("Feature Dimension")
                    plt.colorbar(im, ax=axes[i])

            else:
                # Single feature type
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()

                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(features, aspect="auto", origin="lower")
                ax.set_title("Extracted Features")
                ax.set_xlabel("Time")
                ax.set_ylabel("Feature Dimension")
                plt.colorbar(im, ax=ax)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Feature visualization saved to {save_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not available for visualization")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()


# Convenience functions
def extract_mel_spectrogram(
    audio: AudioData,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 160,
) -> torch.Tensor:
    """
    Convenience function to extract mel spectrogram.

    Args:
        audio: Input audio data
        sample_rate: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Mel spectrogram tensor
    """
    config = {
        "spectrogram": {
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "log_mel": True,
            "normalize": True,
        }
    }

    extractor = FeatureExtractor(config)
    return extractor.extract(audio, sample_rate)


def extract_mfcc(audio: AudioData, sample_rate: int = 16000, n_mfcc: int = 13) -> torch.Tensor:
    """
    Convenience function to extract MFCC features.

    Args:
        audio: Input audio data
        sample_rate: Sample rate
        n_mfcc: Number of MFCC coefficients

    Returns:
        MFCC features tensor
    """
    config = {"mfcc": {"n_mfcc": n_mfcc, "delta": True, "delta_delta": True}}

    extractor = FeatureExtractor(config)
    return extractor.extract(audio, sample_rate)
