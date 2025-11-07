#
# Plan:
# 1. Implement frequency-domain augmentations for spectrograms
# 2. Frequency masking, time masking for SpecAugment-style augmentation
# 3. Spectral filtering, normalization techniques
# 4. Mel-spectrogram specific augmentations
# 5. Military-specific frequency augmentations (engine enhancement, filtering)
# 6. GPU acceleration support
# 7. Preserve critical frequency information for vehicle classification
#

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import random
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class FrequencyAugmentationConfig:
    """Configuration for frequency-domain augmentations."""
    
    # Basic spectral augmentations
    freq_mask: bool = True
    freq_mask_param: int = 20
    freq_mask_prob: float = 0.5
    num_freq_masks: int = 1
    
    time_mask: bool = True
    time_mask_param: int = 40
    time_mask_prob: float = 0.5
    num_time_masks: int = 1
    
    # Spectral filtering
    spectral_filtering: bool = True
    filter_prob: float = 0.3
    filter_types: List[str] = None
    
    # Spectral normalization
    spectral_normalize: bool = True
    normalize_prob: float = 0.4
    
    # Enhancement/suppression
    enhance_harmonics: bool = True
    harmonics_prob: float = 0.2
    
    suppress_noise: bool = True
    noise_suppress_prob: float = 0.3
    
    # Mel-specific augmentations
    mel_stretch: bool = True
    mel_stretch_prob: float = 0.2
    mel_stretch_range: Tuple[float, float] = (0.9, 1.1)
    
    # Global settings
    augmentation_prob: float = 0.8
    max_augmentations: int = 2
    
    def __post_init__(self):
        if self.filter_types is None:
            self.filter_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']


class FrequencyMask:
    """Apply frequency masking to spectrograms."""
    
    def __init__(
        self,
        mask_param: int = 20,
        num_masks: int = 1,
        mask_value: float = 0.0
    ):
        """
        Initialize frequency masking.
        
        Args:
            mask_param: Maximum frequency mask size
            num_masks: Number of masks to apply
            mask_value: Value to fill masked regions
        """
        self.mask_param = mask_param
        self.num_masks = num_masks
        self.mask_value = mask_value
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [channels, freq, time] or [freq, time]
            
        Returns:
            Masked spectrogram
        """
        masked = spectrogram.clone()
        freq_dim = -2  # Frequency dimension
        
        freq_bins = masked.shape[freq_dim]
        
        for _ in range(self.num_masks):
            # Random mask size
            mask_size = random.randint(1, min(self.mask_param, freq_bins // 4))
            
            # Random start position
            mask_start = random.randint(0, freq_bins - mask_size)
            
            # Apply mask
            if masked.dim() == 2:  # [freq, time]
                masked[mask_start:mask_start + mask_size, :] = self.mask_value
            elif masked.dim() == 3:  # [channels, freq, time]
                masked[:, mask_start:mask_start + mask_size, :] = self.mask_value
            else:
                raise ValueError(f"Unsupported spectrogram shape: {masked.shape}")
        
        return masked


class TimeMask:
    """Apply time masking to spectrograms."""
    
    def __init__(
        self,
        mask_param: int = 40,
        num_masks: int = 1,
        mask_value: float = 0.0
    ):
        """
        Initialize time masking.
        
        Args:
            mask_param: Maximum time mask size
            num_masks: Number of masks to apply
            mask_value: Value to fill masked regions
        """
        self.mask_param = mask_param
        self.num_masks = num_masks
        self.mask_value = mask_value
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [channels, freq, time] or [freq, time]
            
        Returns:
            Masked spectrogram
        """
        masked = spectrogram.clone()
        time_dim = -1  # Time dimension
        
        time_frames = masked.shape[time_dim]
        
        for _ in range(self.num_masks):
            # Random mask size
            mask_size = random.randint(1, min(self.mask_param, time_frames // 4))
            
            # Random start position
            mask_start = random.randint(0, time_frames - mask_size)
            
            # Apply mask
            if masked.dim() == 2:  # [freq, time]
                masked[:, mask_start:mask_start + mask_size] = self.mask_value
            elif masked.dim() == 3:  # [channels, freq, time]
                masked[:, :, mask_start:mask_start + mask_size] = self.mask_value
            else:
                raise ValueError(f"Unsupported spectrogram shape: {masked.shape}")
        
        return masked


class SpectralNormalize:
    """Normalize spectrogram using various methods."""
    
    def __init__(
        self,
        method: str = 'instance',
        eps: float = 1e-8
    ):
        """
        Initialize spectral normalization.
        
        Args:
            method: Normalization method ('instance', 'batch', 'layer', 'minmax')
            eps: Small value for numerical stability
        """
        self.method = method
        self.eps = eps
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral normalization.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Normalized spectrogram
        """
        if self.method == 'instance':
            return self._instance_normalize(spectrogram)
        elif self.method == 'batch':
            return self._batch_normalize(spectrogram)
        elif self.method == 'layer':
            return self._layer_normalize(spectrogram)
        elif self.method == 'minmax':
            return self._minmax_normalize(spectrogram)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _instance_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Instance normalization (per sample)."""
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + self.eps)
    
    def _batch_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Batch normalization (across batch dimension)."""
        if x.dim() < 3:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
            squeeze_result = True
        else:
            squeeze_result = False
        
        mean = x.mean(dim=(0, -2, -1), keepdim=True)
        std = x.std(dim=(0, -2, -1), keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        
        if squeeze_result:
            normalized = normalized.squeeze(0)
        
        return normalized
    
    def _layer_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Layer normalization (across frequency dimension)."""
        mean = x.mean(dim=-2, keepdim=True)
        std = x.std(dim=-2, keepdim=True)
        return (x - mean) / (std + self.eps)
    
    def _minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max normalization to [0, 1] range."""
        min_val = x.min()
        max_val = x.max()
        
        if max_val > min_val:
            return (x - min_val) / (max_val - min_val)
        else:
            return x


class FilterBank:
    """Apply various spectral filters."""
    
    def __init__(
        self,
        filter_type: str = 'lowpass',
        cutoff_range: Tuple[float, float] = (0.1, 0.9),
        order: int = 4
    ):
        """
        Initialize filter bank.
        
        Args:
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
            cutoff_range: Range of normalized cutoff frequencies
            order: Filter order
        """
        self.filter_type = filter_type
        self.cutoff_range = cutoff_range
        self.order = order
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral filtering.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Filtered spectrogram
        """
        if self.filter_type == 'lowpass':
            return self._lowpass_filter(spectrogram)
        elif self.filter_type == 'highpass':
            return self._highpass_filter(spectrogram)
        elif self.filter_type == 'bandpass':
            return self._bandpass_filter(spectrogram)
        elif self.filter_type == 'bandstop':
            return self._bandstop_filter(spectrogram)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def _lowpass_filter(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply lowpass filter by zeroing high frequencies."""
        cutoff = random.uniform(*self.cutoff_range)
        freq_bins = spectrogram.shape[-2]
        cutoff_bin = int(cutoff * freq_bins)
        
        filtered = spectrogram.clone()
        filtered[..., cutoff_bin:, :] *= 0.1  # Attenuate rather than zero
        
        return filtered
    
    def _highpass_filter(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply highpass filter by zeroing low frequencies."""
        cutoff = random.uniform(*self.cutoff_range)
        freq_bins = spectrogram.shape[-2]
        cutoff_bin = int(cutoff * freq_bins)
        
        filtered = spectrogram.clone()
        filtered[..., :cutoff_bin, :] *= 0.1  # Attenuate rather than zero
        
        return filtered
    
    def _bandpass_filter(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filter."""
        low_cutoff = random.uniform(self.cutoff_range[0], 0.5)
        high_cutoff = random.uniform(0.5, self.cutoff_range[1])
        
        if low_cutoff >= high_cutoff:
            low_cutoff, high_cutoff = high_cutoff, low_cutoff
        
        freq_bins = spectrogram.shape[-2]
        low_bin = int(low_cutoff * freq_bins)
        high_bin = int(high_cutoff * freq_bins)
        
        filtered = spectrogram.clone()
        filtered[..., :low_bin, :] *= 0.1
        filtered[..., high_bin:, :] *= 0.1
        
        return filtered
    
    def _bandstop_filter(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply bandstop (notch) filter."""
        low_cutoff = random.uniform(self.cutoff_range[0], 0.6)
        high_cutoff = random.uniform(low_cutoff + 0.1, self.cutoff_range[1])
        
        freq_bins = spectrogram.shape[-2]
        low_bin = int(low_cutoff * freq_bins)
        high_bin = int(high_cutoff * freq_bins)
        
        filtered = spectrogram.clone()
        filtered[..., low_bin:high_bin, :] *= 0.1  # Attenuate band
        
        return filtered


class MelSpecAugment:
    """Mel-spectrogram specific augmentations."""
    
    def __init__(
        self,
        stretch_range: Tuple[float, float] = (0.9, 1.1),
        shift_range: Tuple[int, int] = (-5, 5)
    ):
        """
        Initialize Mel-specific augmentations.
        
        Args:
            stretch_range: Range for mel-scale stretching
            shift_range: Range for mel-scale shifting (bins)
        """
        self.stretch_range = stretch_range
        self.shift_range = shift_range
    
    def __call__(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply Mel-specific augmentations.
        
        Args:
            mel_spectrogram: Input mel-spectrogram
            
        Returns:
            Augmented mel-spectrogram
        """
        augmented = mel_spectrogram
        
        # Random choice of augmentation
        aug_choice = random.choice(['stretch', 'shift', 'none'])
        
        if aug_choice == 'stretch':
            augmented = self._mel_stretch(augmented)
        elif aug_choice == 'shift':
            augmented = self._mel_shift(augmented)
        
        return augmented
    
    def _mel_stretch(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Stretch mel-spectrogram in frequency dimension."""
        stretch_factor = random.uniform(*self.stretch_range)
        
        if abs(stretch_factor - 1.0) < 0.01:
            return mel_spec
        
        # Interpolate in frequency dimension
        original_shape = mel_spec.shape
        
        if mel_spec.dim() == 2:  # [freq, time]
            mel_spec_expanded = mel_spec.unsqueeze(0).unsqueeze(0)  # [1, 1, freq, time]
        elif mel_spec.dim() == 3:  # [channels, freq, time]
            mel_spec_expanded = mel_spec.unsqueeze(1)  # [channels, 1, freq, time]
        else:
            return mel_spec
        
        # Calculate new frequency size
        new_freq_size = int(original_shape[-2] * stretch_factor)
        
        # Interpolate
        stretched = F.interpolate(
            mel_spec_expanded,
            size=(new_freq_size, original_shape[-1]),
            mode='bilinear',
            align_corners=False
        )
        
        # Crop or pad to original frequency size
        if new_freq_size > original_shape[-2]:
            # Crop
            start_idx = (new_freq_size - original_shape[-2]) // 2
            stretched = stretched[..., start_idx:start_idx + original_shape[-2], :]
        elif new_freq_size < original_shape[-2]:
            # Pad
            pad_amount = original_shape[-2] - new_freq_size
            pad_top = pad_amount // 2
            pad_bottom = pad_amount - pad_top
            stretched = F.pad(stretched, (0, 0, pad_top, pad_bottom))
        
        # Restore original shape
        if mel_spec.dim() == 2:
            stretched = stretched.squeeze(0).squeeze(0)
        elif mel_spec.dim() == 3:
            stretched = stretched.squeeze(1)
        
        return stretched
    
    def _mel_shift(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Shift mel-spectrogram in frequency dimension."""
        shift_amount = random.randint(*self.shift_range)
        
        if shift_amount == 0:
            return mel_spec
        
        # Circular shift in frequency dimension
        shifted = torch.roll(mel_spec, shifts=shift_amount, dims=-2)
        
        return shifted


class HarmonicEnhancement:
    """Enhance harmonic patterns in spectrograms."""
    
    def __init__(
        self,
        enhancement_factor: float = 1.2,
        harmonic_range: Tuple[int, int] = (2, 8)
    ):
        """
        Initialize harmonic enhancement.
        
        Args:
            enhancement_factor: Factor to enhance harmonics
            harmonic_range: Range of harmonics to consider
        """
        self.enhancement_factor = enhancement_factor
        self.harmonic_range = harmonic_range
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Enhance harmonic patterns in spectrogram.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Enhanced spectrogram
        """
        enhanced = spectrogram.clone()
        
        # Find potential fundamental frequencies
        freq_bins = spectrogram.shape[-2]
        
        # Focus on lower frequencies for fundamental detection
        low_freq_region = freq_bins // 4
        
        # Simple harmonic enhancement
        for fundamental_bin in range(5, low_freq_region):  # Skip very low bins
            for harmonic in range(*self.harmonic_range):
                harmonic_bin = fundamental_bin * harmonic
                
                if harmonic_bin < freq_bins:
                    # Enhance harmonic based on fundamental energy
                    fundamental_energy = enhanced[..., fundamental_bin, :].mean()
                    
                    if fundamental_energy > enhanced.mean() * 0.5:  # Significant fundamental
                        enhancement = (self.enhancement_factor - 1.0) * random.uniform(0.5, 1.0)
                        enhanced[..., harmonic_bin, :] *= (1.0 + enhancement)
        
        return enhanced


class FrequencyAugmentation:
    """
    Main frequency-domain augmentation class.
    """
    
    def __init__(
        self,
        config: Optional[FrequencyAugmentationConfig] = None,
        use_gpu: bool = True
    ):
        """
        Initialize frequency augmentation pipeline.
        
        Args:
            config: Augmentation configuration
            use_gpu: Use GPU acceleration when available
        """
        self.config = config or FrequencyAugmentationConfig()
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize individual augmentations
        self.freq_mask = FrequencyMask(
            self.config.freq_mask_param,
            self.config.num_freq_masks
        )
        
        self.time_mask = TimeMask(
            self.config.time_mask_param,
            self.config.num_time_masks
        )
        
        self.spectral_normalizer = SpectralNormalize()
        self.mel_augment = MelSpecAugment()
        self.harmonic_enhancer = HarmonicEnhancement()
        
        # Initialize filters
        self.filters = [
            FilterBank(filter_type) for filter_type in self.config.filter_types
        ]
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply random frequency-domain augmentations.
        
        Args:
            spectrogram: Input spectrogram [channels, freq, time] or [freq, time]
            
        Returns:
            Augmented spectrogram
        """
        if random.random() > self.config.augmentation_prob:
            return spectrogram
        
        # Move to GPU if available
        if self.use_gpu and not spectrogram.is_cuda:
            spectrogram = spectrogram.cuda()
        
        # Collect available augmentations
        augmentations = []
        
        if self.config.freq_mask and random.random() < self.config.freq_mask_prob:
            augmentations.append(('freq_mask', self.freq_mask))
        
        if self.config.time_mask and random.random() < self.config.time_mask_prob:
            augmentations.append(('time_mask', self.time_mask))
        
        if self.config.spectral_filtering and random.random() < self.config.filter_prob:
            filter_aug = random.choice(self.filters)
            augmentations.append(('filter', filter_aug))
        
        if self.config.spectral_normalize and random.random() < self.config.normalize_prob:
            augmentations.append(('normalize', self.spectral_normalizer))
        
        if self.config.enhance_harmonics and random.random() < self.config.harmonics_prob:
            augmentations.append(('harmonics', self.harmonic_enhancer))
        
        if self.config.mel_stretch and random.random() < self.config.mel_stretch_prob:
            augmentations.append(('mel_augment', self.mel_augment))
        
        # Randomly select and apply augmentations
        if augmentations:
            num_augmentations = min(len(augmentations), self.config.max_augmentations)
            selected_augs = random.sample(augmentations, num_augmentations)
            
            # Shuffle order
            random.shuffle(selected_augs)
            
            augmented = spectrogram
            for name, aug_func in selected_augs:
                try:
                    augmented = aug_func(augmented)
                except Exception as e:
                    logger.warning(f"Frequency augmentation '{name}' failed: {e}")
                    continue
            
            return augmented
        
        return spectrogram
    
    def get_augmentation_names(self) -> List[str]:
        """Get list of available augmentation names."""
        return [
            'freq_mask', 'time_mask', 'filter', 'normalize',
            'harmonics', 'mel_augment'
        ]