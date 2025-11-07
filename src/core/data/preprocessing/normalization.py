#
# Plan:
# 1. Implement comprehensive audio normalization techniques
# 2. RMS, peak, LUFS (EBU R128) normalization for broadcast standards
# 3. Z-score and min-max normalization for ML preprocessing
# 4. Spectral normalization for frequency domain processing
# 5. Adaptive normalization based on content analysis
# 6. Real-time normalization for streaming audio
# 7. Robust normalization for military field conditions
#

import torch
import torch.nn as nn
import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class NormalizationConfig:
    """Configuration for audio normalization."""
    
    # Target levels
    target_rms: float = -20.0  # dB
    target_peak: float = -3.0  # dB
    target_lufs: float = -23.0  # LUFS (EBU R128 standard)
    
    # Protection limits
    max_gain: float = 30.0  # dB
    min_gain: float = -30.0  # dB
    
    # Statistical normalization
    target_mean: float = 0.0
    target_std: float = 1.0
    
    # Min-max normalization
    target_min: float = 0.0
    target_max: float = 1.0
    
    # Processing parameters
    eps: float = 1e-8  # Small value to avoid division by zero
    use_gpu: bool = True


class AudioNormalizer(nn.Module):
    """Base class for audio normalizers."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize audio normalizer.
        
        Args:
            use_gpu: Use GPU acceleration when available
        """
        super().__init__()
        self.use_gpu = use_gpu and torch.cuda.is_available()
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Normalized audio tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _ensure_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to GPU if available and requested."""
        if self.use_gpu and not tensor.is_cuda:
            return tensor.cuda()
        return tensor
    
    def _validate_input(self, audio: torch.Tensor) -> torch.Tensor:
        """Validate and prepare input audio."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        elif audio.dim() > 2:
            raise ValueError(f"Audio must be 1D or 2D, got {audio.dim()}D")
        
        return self._ensure_gpu(audio)
    
    def _db_to_linear(self, db: float) -> float:
        """Convert dB to linear scale."""
        return 10 ** (db / 20.0)
    
    def _linear_to_db(self, linear: float, eps: float = 1e-8) -> float:
        """Convert linear scale to dB."""
        return 20 * math.log10(max(linear, eps))


class RMSNormalization(AudioNormalizer):
    """RMS-based audio normalization."""
    
    def __init__(
        self,
        target_rms: float = -20.0,  # dB
        max_gain: float = 30.0,     # dB
        eps: float = 1e-8,
        use_gpu: bool = True
    ):
        """
        Initialize RMS normalization.
        
        Args:
            target_rms: Target RMS level in dB
            max_gain: Maximum allowed gain in dB
            eps: Small value for numerical stability
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(use_gpu)
        
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.eps = eps
        self.target_rms_linear = self._db_to_linear(target_rms)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to audio.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            RMS normalized audio tensor
        """
        audio = self._validate_input(audio)
        
        # Calculate RMS
        rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
        
        # Avoid division by zero
        rms = torch.clamp(rms, min=self.eps)
        
        # Calculate gain
        gain = self.target_rms_linear / rms
        
        # Limit gain to prevent excessive amplification
        max_gain_linear = self._db_to_linear(self.max_gain)
        gain = torch.clamp(gain, max=max_gain_linear)
        
        # Apply gain
        normalized_audio = audio * gain
        
        return normalized_audio


class PeakNormalization(AudioNormalizer):
    """Peak-based audio normalization."""
    
    def __init__(
        self,
        target_peak: float = -3.0,  # dB
        max_gain: float = 30.0,     # dB
        eps: float = 1e-8,
        use_gpu: bool = True
    ):
        """
        Initialize peak normalization.
        
        Args:
            target_peak: Target peak level in dB
            max_gain: Maximum allowed gain in dB
            eps: Small value for numerical stability
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(use_gpu)
        
        self.target_peak = target_peak
        self.max_gain = max_gain
        self.eps = eps
        self.target_peak_linear = self._db_to_linear(target_peak)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply peak normalization to audio.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            Peak normalized audio tensor
        """
        audio = self._validate_input(audio)
        
        # Calculate peak
        peak = torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]
        
        # Avoid division by zero
        peak = torch.clamp(peak, min=self.eps)
        
        # Calculate gain
        gain = self.target_peak_linear / peak
        
        # Limit gain to prevent excessive amplification
        max_gain_linear = self._db_to_linear(self.max_gain)
        gain = torch.clamp(gain, max=max_gain_linear)
        
        # Apply gain
        normalized_audio = audio * gain
        
        return normalized_audio


class LUFSNormalization(AudioNormalizer):
    """LUFS (Loudness Units relative to Full Scale) normalization following EBU R128 standard."""
    
    def __init__(
        self,
        target_lufs: float = -23.0,  # LUFS
        max_gain: float = 30.0,      # dB
        sample_rate: int = 16000,
        use_gpu: bool = True
    ):
        """
        Initialize LUFS normalization.
        
        Args:
            target_lufs: Target LUFS level
            max_gain: Maximum allowed gain in dB
            sample_rate: Audio sample rate
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(use_gpu)
        
        self.target_lufs = target_lufs
        self.max_gain = max_gain
        self.sample_rate = sample_rate
        
        # Pre-filter coefficients for LUFS measurement (simplified)
        # This is a simplified implementation of the K-weighting filter
        self._setup_k_weighting_filter()
    
    def _setup_k_weighting_filter(self):
        """Setup K-weighting filter for LUFS measurement (simplified)."""
        # Simplified K-weighting filter
        # In practice, this should be a proper implementation of the EBU R128 filter
        
        # High-pass filter at ~38 Hz
        self.hp_cutoff = 38.0 / (self.sample_rate / 2)  # Normalized frequency
        
        # High-frequency shelving filter at ~1.5 kHz
        self.hf_cutoff = 1500.0 / (self.sample_rate / 2)  # Normalized frequency
        
        logger.warning("LUFS normalization uses simplified K-weighting filter")
    
    def _apply_k_weighting(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply simplified K-weighting filter."""
        # This is a very simplified implementation
        # For production use, implement proper EBU R128 K-weighting filter
        
        # Simple high-pass filtering approximation
        if audio.shape[-1] < 3:
            return audio
        
        # Simple difference equation for high-pass effect
        filtered = audio.clone()
        alpha = 0.99  # High-pass filter coefficient
        
        for i in range(1, audio.shape[-1]):
            filtered[..., i] = alpha * (filtered[..., i-1] + audio[..., i] - audio[..., i-1])
        
        return filtered
    
    def _calculate_lufs(self, audio: torch.Tensor) -> torch.Tensor:
        """Calculate LUFS measurement."""
        # Apply K-weighting filter
        weighted_audio = self._apply_k_weighting(audio)
        
        # Calculate mean square
        mean_square = torch.mean(weighted_audio ** 2, dim=-1, keepdim=True)
        
        # Convert to LUFS
        lufs = -0.691 + 10 * torch.log10(mean_square + 1e-8)
        
        return lufs
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply LUFS normalization to audio.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            LUFS normalized audio tensor
        """
        audio = self._validate_input(audio)
        
        # Calculate current LUFS
        current_lufs = self._calculate_lufs(audio)
        
        # Calculate required gain
        gain_db = self.target_lufs - current_lufs
        
        # Limit gain
        gain_db = torch.clamp(gain_db, min=-self.max_gain, max=self.max_gain)
        
        # Convert to linear gain
        gain_linear = torch.pow(10, gain_db / 20.0)
        
        # Apply gain
        normalized_audio = audio * gain_linear
        
        return normalized_audio


class ZScoreNormalization(AudioNormalizer):
    """Z-score (standard score) normalization."""
    
    def __init__(
        self,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        eps: float = 1e-8,
        use_gpu: bool = True
    ):
        """
        Initialize Z-score normalization.
        
        Args:
            target_mean: Target mean value
            target_std: Target standard deviation
            eps: Small value for numerical stability
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(use_gpu)
        
        self.target_mean = target_mean
        self.target_std = target_std
        self.eps = eps
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply Z-score normalization to audio.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            Z-score normalized audio tensor
        """
        audio = self._validate_input(audio)
        
        # Calculate statistics
        mean = torch.mean(audio, dim=-1, keepdim=True)
        std = torch.std(audio, dim=-1, keepdim=True)
        
        # Avoid division by zero
        std = torch.clamp(std, min=self.eps)
        
        # Apply Z-score normalization
        normalized_audio = (audio - mean) / std
        
        # Scale to target mean and std
        normalized_audio = normalized_audio * self.target_std + self.target_mean
        
        return normalized_audio


class MinMaxNormalization(AudioNormalizer):
    """Min-max normalization to specified range."""
    
    def __init__(
        self,
        target_min: float = 0.0,
        target_max: float = 1.0,
        eps: float = 1e-8,
        use_gpu: bool = True
    ):
        """
        Initialize min-max normalization.
        
        Args:
            target_min: Target minimum value
            target_max: Target maximum value
            eps: Small value for numerical stability
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(use_gpu)
        
        self.target_min = target_min
        self.target_max = target_max
        self.eps = eps
        self.target_range = target_max - target_min
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply min-max normalization to audio.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            Min-max normalized audio tensor
        """
        audio = self._validate_input(audio)
        
        # Calculate min and max
        audio_min = torch.min(audio, dim=-1, keepdim=True)[0]
        audio_max = torch.max(audio, dim=-1, keepdim=True)[0]
        
        # Calculate range
        audio_range = audio_max - audio_min
        
        # Avoid division by zero
        audio_range = torch.clamp(audio_range, min=self.eps)
        
        # Apply min-max normalization
        normalized_audio = (audio - audio_min) / audio_range
        
        # Scale to target range
        normalized_audio = normalized_audio * self.target_range + self.target_min
        
        return normalized_audio


class SpectralNormalization(AudioNormalizer):
    """Spectral normalization for frequency domain processing."""
    
    def __init__(
        self,
        method: str = 'layer',  # 'layer', 'instance', 'batch'
        eps: float = 1e-8,
        use_gpu: bool = True
    ):
        """
        Initialize spectral normalization.
        
        Args:
            method: Normalization method ('layer', 'instance', 'batch')
            eps: Small value for numerical stability
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(use_gpu)
        
        self.method = method
        self.eps = eps
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral normalization to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [batch, channels, freq, time] or [freq, time]
            
        Returns:
            Normalized spectrogram
        """
        if spectrogram.dim() < 2:
            raise ValueError("Spectrogram must be at least 2D")
        
        spectrogram = self._ensure_gpu(spectrogram)
        
        if self.method == 'layer':
            return self._layer_normalize(spectrogram)
        elif self.method == 'instance':
            return self._instance_normalize(spectrogram)
        elif self.method == 'batch':
            return self._batch_normalize(spectrogram)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _layer_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Layer normalization across frequency dimension."""
        if x.dim() == 2:  # [freq, time]
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
        elif x.dim() == 3:  # [channels, freq, time]
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
        elif x.dim() == 4:  # [batch, channels, freq, time]
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        
        return (x - mean) / (std + self.eps)
    
    def _instance_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Instance normalization per sample."""
        if x.dim() == 2:  # [freq, time]
            mean = x.mean()
            std = x.std()
        elif x.dim() == 3:  # [channels, freq, time]
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True)
        elif x.dim() == 4:  # [batch, channels, freq, time]
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        
        return (x - mean) / (std + self.eps)
    
    def _batch_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Batch normalization across batch dimension."""
        if x.dim() < 3:
            raise ValueError("Batch normalization requires at least 3D tensor")
        
        if x.dim() == 3:  # [batch, freq, time]
            mean = x.mean(dim=(0, 2), keepdim=True)
            std = x.std(dim=(0, 2), keepdim=True)
        elif x.dim() == 4:  # [batch, channels, freq, time]
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        
        return (x - mean) / (std + self.eps)


class AdaptiveNormalization(AudioNormalizer):
    """Adaptive normalization that adjusts based on audio content."""
    
    def __init__(
        self,
        noise_threshold: float = -60.0,  # dB
        speech_threshold: float = -30.0,  # dB
        music_threshold: float = -20.0,  # dB
        use_gpu: bool = True
    ):
        """
        Initialize adaptive normalization.
        
        Args:
            noise_threshold: Threshold for noise detection (dB)
            speech_threshold: Threshold for speech detection (dB)
            music_threshold: Threshold for music detection (dB)
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(use_gpu)
        
        self.noise_threshold = noise_threshold
        self.speech_threshold = speech_threshold
        self.music_threshold = music_threshold
        
        # Initialize different normalizers
        self.rms_normalizer = RMSNormalization(-20.0, use_gpu=use_gpu)
        self.peak_normalizer = PeakNormalization(-3.0, use_gpu=use_gpu)
        self.lufs_normalizer = LUFSNormalization(-23.0, use_gpu=use_gpu)
    
    def _analyze_content(self, audio: torch.Tensor) -> str:
        """Analyze audio content to determine appropriate normalization."""
        # Calculate RMS level
        rms = torch.sqrt(torch.mean(audio ** 2))
        rms_db = 20 * torch.log10(rms + 1e-8)
        
        # Simple content classification based on level
        if rms_db < self.noise_threshold:
            return 'noise'
        elif rms_db < self.speech_threshold:
            return 'speech'
        elif rms_db < self.music_threshold:
            return 'music'
        else:
            return 'loud'
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive normalization based on content analysis.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Adaptively normalized audio tensor
        """
        audio = self._validate_input(audio)
        
        # Analyze content
        content_type = self._analyze_content(audio)
        
        # Apply appropriate normalization
        if content_type == 'noise':
            # For noise, use more aggressive normalization
            return self.peak_normalizer(audio)
        elif content_type == 'speech':
            # For speech, use RMS normalization
            return self.rms_normalizer(audio)
        elif content_type in ['music', 'loud']:
            # For music/loud content, use LUFS normalization
            return self.lufs_normalizer(audio)
        else:
            # Default to RMS normalization
            return self.rms_normalizer(audio)


class RobustNormalization(AudioNormalizer):
    """Robust normalization for military field conditions with outlier handling."""
    
    def __init__(
        self,
        target_level: float = -20.0,  # dB
        percentile: float = 95.0,     # Use 95th percentile instead of max
        max_gain: float = 30.0,       # dB
        smoothing_factor: float = 0.1,
        use_gpu: bool = True
    ):
        """
        Initialize robust normalization.
        
        Args:
            target_level: Target level in dB
            percentile: Percentile to use for level calculation
            max_gain: Maximum allowed gain in dB
            smoothing_factor: Smoothing factor for level tracking
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(use_gpu)
        
        self.target_level = target_level
        self.percentile = percentile
        self.max_gain = max_gain
        self.smoothing_factor = smoothing_factor
        self.target_level_linear = self._db_to_linear(target_level)
        
        # Running estimate of audio level
        self.running_level = None
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply robust normalization to audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Robustly normalized audio tensor
        """
        audio = self._validate_input(audio)
        
        # Calculate percentile-based level (more robust than peak)
        audio_abs = torch.abs(audio)
        level = torch.quantile(audio_abs, self.percentile / 100.0, dim=-1, keepdim=True)
        
        # Update running level estimate
        if self.running_level is None:
            self.running_level = level
        else:
            self.running_level = (
                (1 - self.smoothing_factor) * self.running_level +
                self.smoothing_factor * level
            )
        
        # Use running level for normalization
        level = torch.clamp(self.running_level, min=1e-8)
        
        # Calculate gain
        gain = self.target_level_linear / level
        
        # Limit gain
        max_gain_linear = self._db_to_linear(self.max_gain)
        gain = torch.clamp(gain, max=max_gain_linear)
        
        # Apply gain
        normalized_audio = audio * gain
        
        return normalized_audio


def create_normalizer(
    normalizer_type: str = 'rms',
    config: Optional[NormalizationConfig] = None,
    **kwargs
) -> AudioNormalizer:
    """
    Factory function to create audio normalizers.
    
    Args:
        normalizer_type: Type of normalizer ('rms', 'peak', 'lufs', 'zscore', 'minmax', 'spectral', 'adaptive', 'robust')
        config: Normalization configuration
        **kwargs: Override configuration parameters
        
    Returns:
        Audio normalizer instance
    """
    if config is None:
        config = NormalizationConfig()
    
    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create appropriate normalizer
    if normalizer_type == 'rms':
        return RMSNormalization(
            target_rms=config.target_rms,
            max_gain=config.max_gain,
            eps=config.eps,
            use_gpu=config.use_gpu
        )
    elif normalizer_type == 'peak':
        return PeakNormalization(
            target_peak=config.target_peak,
            max_gain=config.max_gain,
            eps=config.eps,
            use_gpu=config.use_gpu
        )
    elif normalizer_type == 'lufs':
        return LUFSNormalization(
            target_lufs=config.target_lufs,
            max_gain=config.max_gain,
            use_gpu=config.use_gpu
        )
    elif normalizer_type == 'zscore':
        return ZScoreNormalization(
            target_mean=config.target_mean,
            target_std=config.target_std,
            eps=config.eps,
            use_gpu=config.use_gpu
        )
    elif normalizer_type == 'minmax':
        return MinMaxNormalization(
            target_min=config.target_min,
            target_max=config.target_max,
            eps=config.eps,
            use_gpu=config.use_gpu
        )
    elif normalizer_type == 'spectral':
        return SpectralNormalization(
            eps=config.eps,
            use_gpu=config.use_gpu
        )
    elif normalizer_type == 'adaptive':
        return AdaptiveNormalization(use_gpu=config.use_gpu)
    elif normalizer_type == 'robust':
        return RobustNormalization(
            target_level=config.target_rms,
            max_gain=config.max_gain,
            use_gpu=config.use_gpu
        )
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer_type}")


def get_normalization_info():
    """
    Get information about available normalization types.
    
    Returns:
        Dictionary with normalization information
    """
    return {
        'available_types': [
            'rms', 'peak', 'lufs', 'zscore', 'minmax', 'spectral', 'adaptive', 'robust'
        ],
        'description': {
            'rms': 'RMS-based normalization for consistent loudness',
            'peak': 'Peak-based normalization to prevent clipping',
            'lufs': 'LUFS normalization following EBU R128 standard',
            'zscore': 'Z-score normalization for ML preprocessing',
            'minmax': 'Min-max normalization to specified range',
            'spectral': 'Spectral normalization for frequency domain',
            'adaptive': 'Adaptive normalization based on content analysis',
            'robust': 'Robust normalization for field conditions'
        },
        'recommended_for_military': 'robust'
    }