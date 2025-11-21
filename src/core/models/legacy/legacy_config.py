# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Legacy Model Configuration
===========================
Configuration dataclasses and enums for legacy CNN/CRNN models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Optional, Any


class LegacyModelType(str, Enum):
    """Enumeration of available legacy model types."""
    CNN = "cnn_mfcc"
    CRNN = "crnn_mfcc"


@dataclass
class MFCCConfig:
    """MFCC feature extraction configuration."""

    # Audio processing
    sample_rate: int = 16000  # Sampling rate in Hz
    n_mfcc: int = 40  # Number of MFCC coefficients
    n_mels: int = 64  # Number of mel bands
    n_fft: int = 1024  # FFT window size
    hop_length: int = 512  # Number of samples between successive frames
    f_min: float = 0.0  # Minimum frequency
    f_max: Optional[float] = None  # Maximum frequency (None = Nyquist)

    # MFCC-specific
    use_deltas: bool = True  # Include delta (velocity) features
    use_delta_deltas: bool = True  # Include delta-delta (acceleration) features
    delta_window: int = 3  # Window for computing delta features (must be odd >=3)

    # Normalization
    normalize: bool = True  # Normalize MFCC features per coefficient
    normalization_type: str = "zscore"  # "zscore" or "minmax"

    # Duration
    target_duration: float = 3.0  # Target audio duration in seconds (CNN)
    crnn_duration: float = 4.0  # CRNN uses longer duration

    def __post_init__(self):
        """Validate configuration."""
        if self.n_mfcc <= 0:
            raise ValueError("n_mfcc must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.delta_window < 3:
            self.delta_window = 3
        if self.delta_window % 2 == 0:
            self.delta_window += 1


@dataclass
class SpecAugmentConfig:
    """SpecAugment data augmentation configuration."""

    # Frequency masking
    freq_mask_param: int = 15  # Maximum frequency mask width (percentage)
    num_freq_masks: int = 2  # Number of frequency masks to apply

    # Time masking
    time_mask_param: int = 10  # Maximum time mask width (percentage)
    num_time_masks: int = 2  # Number of time masks to apply

    # Probability of applying augmentation
    apply_prob: float = 0.8  # Probability of applying SpecAugment

    def __post_init__(self):
        """Validate configuration."""
        if not (0 <= self.freq_mask_param <= 100):
            raise ValueError("freq_mask_param must be between 0 and 100")
        if not (0 <= self.time_mask_param <= 100):
            raise ValueError("time_mask_param must be between 0 and 100")
        if not (0 <= self.apply_prob <= 1):
            raise ValueError("apply_prob must be between 0 and 1")


@dataclass
class CNNConfig:
    """CNN-specific architecture configuration."""

    # Architecture
    num_classes: int = 7
    input_shape: Tuple[int, int, int] = (40, 92, 3)  # (mfcc, time, channels)

    # Conv layers configuration
    conv_filters: Tuple[int, ...] = (48, 96, 192)  # Filters for each conv layer
    conv_kernel_size: Tuple[int, int] = (3, 3)  # Conv kernel size
    pool_size: Tuple[int, int] = (2, 2)  # MaxPool size
    pool_strides: Optional[Tuple[int, int]] = None  # Pool strides (defaults to pool_size)

    # Regularization
    dropout_rates: Tuple[float, ...] = (0.25, 0.30, 0.30, 0.35)  # Dropout after each layer
    batch_norm: bool = True  # Use batch normalization

    # Dense layers
    dense_units: int = 160  # Hidden dense layer units

    # Training
    use_class_weights: bool = True  # Balance imbalanced classes


@dataclass
class CRNNConfig:
    """CRNN-specific architecture configuration."""

    # Architecture
    num_classes: int = 7
    input_shape: Tuple[int, int, int] = (40, 124, 3)  # (mfcc, time, channels) for 4s audio

    # Conv layers configuration
    conv_filters: Tuple[int, ...] = (48, 96, 192)  # Same as CNN
    conv_kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 1)  # Note: different from CNN (2,1) for time preservation
    pool_strides: Optional[Tuple[int, int]] = None

    # Regularization
    conv_dropout_rates: Tuple[float, ...] = (0.20, 0.25, 0.30)  # Conv layer dropouts
    rnn_dropout: float = 0.0  # LSTM dropout
    recurrent_dropout: float = 0.0  # LSTM recurrent dropout
    batch_norm: bool = True

    # LSTM configuration
    lstm_units: Tuple[int, ...] = (128, 64)  # Units for each LSTM layer
    return_sequences: bool = True  # Return sequences for bidirectional processing

    # Dense layers
    dense_units: int = 160  # Hidden dense layer units
    final_dropout: float = 0.35

    # Pooling after LSTM
    use_max_pooling: bool = True  # Use both avg and max pooling

    # Training
    use_class_weights: bool = True


@dataclass
class LegacyModelConfig:
    """Complete configuration for legacy models."""

    # Model selection
    model_type: LegacyModelType = LegacyModelType.CNN

    # Feature extraction
    mfcc: MFCCConfig = field(default_factory=MFCCConfig)

    # Augmentation
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)

    # Architecture-specific
    cnn: CNNConfig = field(default_factory=CNNConfig)
    crnn: CRNNConfig = field(default_factory=CRNNConfig)

    # Device and precision
    device: str = "cuda"  # "cuda" or "cpu"
    dtype: str = "float32"  # "float32" or "float16" for mixed precision

    # Checkpoint and logging
    checkpoint_dir: str = "checkpoints/legacy"
    log_dir: str = "logs/legacy"

    @property
    def active_arch_config(self) -> Any:
        """Get active architecture config based on model type."""
        if self.model_type == LegacyModelType.CNN:
            return self.cnn
        elif self.model_type == LegacyModelType.CRNN:
            return self.crnn
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def __post_init__(self):
        """Validate configuration."""
        if self.model_type not in LegacyModelType:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")
        if self.dtype not in ["float32", "float16"]:
            raise ValueError("dtype must be 'float32' or 'float16'")
