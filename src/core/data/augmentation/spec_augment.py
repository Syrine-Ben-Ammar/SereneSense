#
# Plan:
# 1. Implement SpecAugment (frequency masking, time masking, time warping)
# 2. Multiple masking strategies for different use cases
# 3. Adaptive masking based on spectrogram characteristics
# 4. Military-specific SpecAugment configurations
# 5. GPU-optimized implementations
# 6. Support for both mel-spectrograms and raw spectrograms
# 7. Configurable masking policies for different model architectures
#

import torch
import torch.nn.functional as F
import numpy as np
import random
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class SpecAugmentConfig:
    """Configuration for SpecAugment augmentation."""
    
    # Frequency masking
    freq_mask_param: int = 20
    num_freq_masks: int = 2
    freq_mask_prob: float = 0.8
    
    # Time masking
    time_mask_param: int = 40
    num_time_masks: int = 2
    time_mask_prob: float = 0.8
    
    # Time warping
    time_warp_param: int = 40
    time_warp_prob: float = 0.5
    
    # Masking values
    mask_value: float = 0.0  # Value to fill masked regions
    
    # Adaptive masking
    adaptive_masking: bool = True
    min_mask_size: int = 1
    
    # Policy presets
    policy: str = 'standard'  # 'mild', 'standard', 'aggressive'


class FreqMasking:
    """Frequency masking for SpecAugment."""
    
    def __init__(
        self,
        mask_param: int = 20,
        num_masks: int = 1,
        mask_value: float = 0.0,
        adaptive: bool = True
    ):
        """
        Initialize frequency masking.
        
        Args:
            mask_param: Maximum frequency mask width
            num_masks: Number of frequency masks to apply
            mask_value: Value to fill masked regions
            adaptive: Use adaptive masking based on spectrogram size
        """
        self.mask_param = mask_param
        self.num_masks = num_masks
        self.mask_value = mask_value
        self.adaptive = adaptive
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [batch, channels, freq, time] or [freq, time]
            
        Returns:
            Frequency masked spectrogram
        """
        if spectrogram.numel() == 0:
            return spectrogram
        
        masked = spectrogram.clone()
        
        # Get frequency dimension size
        if masked.dim() == 2:  # [freq, time]
            freq_size = masked.shape[0]
        elif masked.dim() == 3:  # [channels, freq, time]
            freq_size = masked.shape[1]
        elif masked.dim() == 4:  # [batch, channels, freq, time]
            freq_size = masked.shape[2]
        else:
            raise ValueError(f"Unsupported spectrogram shape: {masked.shape}")
        
        # Adaptive mask parameter
        if self.adaptive:
            effective_mask_param = min(self.mask_param, freq_size // 4)
        else:
            effective_mask_param = self.mask_param
        
        # Apply multiple frequency masks
        for _ in range(self.num_masks):
            if effective_mask_param <= 0:
                continue
                
            # Random mask width
            mask_width = random.randint(1, effective_mask_param)
            
            # Random start position
            mask_start = random.randint(0, max(0, freq_size - mask_width))
            mask_end = mask_start + mask_width
            
            # Apply mask based on tensor dimensions
            if masked.dim() == 2:  # [freq, time]
                masked[mask_start:mask_end, :] = self.mask_value
            elif masked.dim() == 3:  # [channels, freq, time]
                masked[:, mask_start:mask_end, :] = self.mask_value
            elif masked.dim() == 4:  # [batch, channels, freq, time]
                masked[:, :, mask_start:mask_end, :] = self.mask_value
        
        return masked


class TimeMasking:
    """Time masking for SpecAugment."""
    
    def __init__(
        self,
        mask_param: int = 40,
        num_masks: int = 1,
        mask_value: float = 0.0,
        adaptive: bool = True
    ):
        """
        Initialize time masking.
        
        Args:
            mask_param: Maximum time mask width
            num_masks: Number of time masks to apply
            mask_value: Value to fill masked regions
            adaptive: Use adaptive masking based on spectrogram size
        """
        self.mask_param = mask_param
        self.num_masks = num_masks
        self.mask_value = mask_value
        self.adaptive = adaptive
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [batch, channels, freq, time] or [freq, time]
            
        Returns:
            Time masked spectrogram
        """
        if spectrogram.numel() == 0:
            return spectrogram
        
        masked = spectrogram.clone()
        
        # Get time dimension size
        time_size = masked.shape[-1]
        
        # Adaptive mask parameter
        if self.adaptive:
            effective_mask_param = min(self.mask_param, time_size // 4)
        else:
            effective_mask_param = self.mask_param
        
        # Apply multiple time masks
        for _ in range(self.num_masks):
            if effective_mask_param <= 0:
                continue
                
            # Random mask width
            mask_width = random.randint(1, effective_mask_param)
            
            # Random start position
            mask_start = random.randint(0, max(0, time_size - mask_width))
            mask_end = mask_start + mask_width
            
            # Apply mask based on tensor dimensions
            if masked.dim() == 2:  # [freq, time]
                masked[:, mask_start:mask_end] = self.mask_value
            elif masked.dim() == 3:  # [channels, freq, time]
                masked[:, :, mask_start:mask_end] = self.mask_value
            elif masked.dim() == 4:  # [batch, channels, freq, time]
                masked[:, :, :, mask_start:mask_end] = self.mask_value
        
        return masked


class TimeWarping:
    """Time warping for SpecAugment."""
    
    def __init__(
        self,
        warp_param: int = 40,
        mode: str = 'bicubic'
    ):
        """
        Initialize time warping.
        
        Args:
            warp_param: Maximum time warp distance
            mode: Interpolation mode for warping
        """
        self.warp_param = warp_param
        self.mode = mode
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [batch, channels, freq, time] or [freq, time]
            
        Returns:
            Time warped spectrogram
        """
        if spectrogram.numel() == 0:
            return spectrogram
        
        # Time warping requires at least 2D input
        original_shape = spectrogram.shape
        
        # Ensure 4D tensor [batch, channels, freq, time]
        if spectrogram.dim() == 2:  # [freq, time]
            spec_4d = spectrogram.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
            squeeze_channel = True
        elif spectrogram.dim() == 3:  # [channels, freq, time]
            spec_4d = spectrogram.unsqueeze(0)
            squeeze_batch = True
            squeeze_channel = False
        elif spectrogram.dim() == 4:  # [batch, channels, freq, time]
            spec_4d = spectrogram
            squeeze_batch = False
            squeeze_channel = False
        else:
            raise ValueError(f"Unsupported spectrogram shape: {original_shape}")
        
        batch_size, channels, freq_size, time_size = spec_4d.shape
        
        # Skip if time dimension is too small
        if time_size < 10:
            return spectrogram
        
        # Generate warping path
        warped_spec = self._warp_time_dimension(spec_4d)
        
        # Restore original shape
        if squeeze_batch and squeeze_channel:
            warped_spec = warped_spec.squeeze(0).squeeze(0)
        elif squeeze_batch:
            warped_spec = warped_spec.squeeze(0)
        
        return warped_spec
    
    def _warp_time_dimension(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time warping using grid sampling."""
        batch_size, channels, freq_size, time_size = spectrogram.shape
        
        # Create warping control points
        # Use a single control point in the middle of the time axis
        center_point = time_size // 2
        
        # Random displacement within warp_param
        displacement = random.randint(-self.warp_param, self.warp_param)
        
        # Clamp displacement to valid range
        displacement = max(-center_point, min(displacement, time_size - center_point - 1))
        
        if displacement == 0:
            return spectrogram
        
        # Create coordinate grid
        time_grid = torch.linspace(-1, 1, time_size, device=spectrogram.device)
        freq_grid = torch.linspace(-1, 1, freq_size, device=spectrogram.device)
        
        # Create warping function
        warped_time_grid = time_grid.clone()
        
        # Apply smooth warping around the center point
        center_normalized = (center_point / time_size) * 2 - 1  # Convert to [-1, 1]
        displacement_normalized = (displacement / time_size) * 2  # Convert to [-1, 1] scale
        
        # Smooth warping function (using tanh for smoothness)
        sigma = 0.3  # Controls the width of the warping region
        warp_weights = torch.tanh((time_grid - center_normalized) / sigma)
        warped_time_grid = time_grid + displacement_normalized * warp_weights * 0.5
        
        # Clamp to valid range
        warped_time_grid = torch.clamp(warped_time_grid, -1, 1)
        
        # Create sampling grid
        grid_freq, grid_time = torch.meshgrid(freq_grid, warped_time_grid, indexing='ij')
        grid = torch.stack([grid_time, grid_freq], dim=-1)  # [freq, time, 2]
        
        # Expand grid for batch processing
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Apply grid sampling
        try:
            warped = F.grid_sample(
                spectrogram,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
        except Exception as e:
            logger.warning(f"Time warping failed: {e}")
            return spectrogram
        
        return warped


class CombinedSpecAugment:
    """Combined SpecAugment applying all augmentations."""
    
    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        time_warp_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        freq_mask_prob: float = 0.8,
        time_mask_prob: float = 0.8,
        time_warp_prob: float = 0.5,
        mask_value: float = 0.0
    ):
        """
        Initialize combined SpecAugment.
        
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            time_warp_param: Maximum time warp distance
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks
            freq_mask_prob: Probability of applying frequency masking
            time_mask_prob: Probability of applying time masking
            time_warp_prob: Probability of applying time warping
            mask_value: Value to fill masked regions
        """
        self.freq_masking = FreqMasking(
            freq_mask_param, num_freq_masks, mask_value
        )
        self.time_masking = TimeMasking(
            time_mask_param, num_time_masks, mask_value
        )
        self.time_warping = TimeWarping(time_warp_param)
        
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.time_warp_prob = time_warp_prob
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply combined SpecAugment.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Augmented spectrogram
        """
        augmented = spectrogram
        
        # Apply augmentations in order
        # 1. Time warping (if enabled)
        if random.random() < self.time_warp_prob:
            try:
                augmented = self.time_warping(augmented)
            except Exception as e:
                logger.warning(f"Time warping failed: {e}")
        
        # 2. Frequency masking
        if random.random() < self.freq_mask_prob:
            try:
                augmented = self.freq_masking(augmented)
            except Exception as e:
                logger.warning(f"Frequency masking failed: {e}")
        
        # 3. Time masking
        if random.random() < self.time_mask_prob:
            try:
                augmented = self.time_masking(augmented)
            except Exception as e:
                logger.warning(f"Time masking failed: {e}")
        
        return augmented


class SpecAugment:
    """
    Main SpecAugment class with configurable policies.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        time_warp_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        freq_mask_prob: float = 0.8,
        time_mask_prob: float = 0.8,
        time_warp_prob: float = 0.5,
        mask_value: float = 0.0,
        policy: str = 'standard'
    ):
        """
        Initialize SpecAugment with policy configuration.
        
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            time_warp_param: Maximum time warp distance
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks
            freq_mask_prob: Probability of applying frequency masking
            time_mask_prob: Probability of applying time masking
            time_warp_prob: Probability of applying time warping
            mask_value: Value to fill masked regions
            policy: Augmentation policy ('mild', 'standard', 'aggressive')
        """
        # Apply policy presets
        if policy == 'mild':
            freq_mask_param = min(15, freq_mask_param)
            time_mask_param = min(25, time_mask_param)
            time_warp_param = min(20, time_warp_param)
            num_freq_masks = 1
            num_time_masks = 1
            freq_mask_prob = 0.6
            time_mask_prob = 0.6
            time_warp_prob = 0.3
        elif policy == 'aggressive':
            freq_mask_param = max(30, freq_mask_param)
            time_mask_param = max(60, time_mask_param)
            time_warp_param = max(60, time_warp_param)
            num_freq_masks = max(3, num_freq_masks)
            num_time_masks = max(3, num_time_masks)
            freq_mask_prob = 0.9
            time_mask_prob = 0.9
            time_warp_prob = 0.7
        # 'standard' uses provided parameters as-is
        
        self.combined_augment = CombinedSpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            time_warp_param=time_warp_param,
            num_freq_masks=num_freq_masks,
            num_time_masks=num_time_masks,
            freq_mask_prob=freq_mask_prob,
            time_mask_prob=time_mask_prob,
            time_warp_prob=time_warp_prob,
            mask_value=mask_value
        )
        
        self.policy = policy
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Augmented spectrogram
        """
        return self.combined_augment(spectrogram)
    
    @classmethod
    def from_config(cls, config: SpecAugmentConfig) -> 'SpecAugment':
        """
        Create SpecAugment from configuration.
        
        Args:
            config: SpecAugment configuration
            
        Returns:
            SpecAugment instance
        """
        return cls(
            freq_mask_param=config.freq_mask_param,
            time_mask_param=config.time_mask_param,
            time_warp_param=config.time_warp_param,
            num_freq_masks=config.num_freq_masks,
            num_time_masks=config.num_time_masks,
            freq_mask_prob=config.freq_mask_prob,
            time_mask_prob=config.time_mask_prob,
            time_warp_prob=config.time_warp_prob,
            mask_value=config.mask_value,
            policy=config.policy
        )


# Predefined SpecAugment policies for different use cases
SPECAUGMENT_POLICIES = {
    'mild': {
        'freq_mask_param': 15,
        'time_mask_param': 25,
        'time_warp_param': 20,
        'num_freq_masks': 1,
        'num_time_masks': 1,
        'freq_mask_prob': 0.6,
        'time_mask_prob': 0.6,
        'time_warp_prob': 0.3
    },
    'standard': {
        'freq_mask_param': 20,
        'time_mask_param': 40,
        'time_warp_param': 40,
        'num_freq_masks': 2,
        'num_time_masks': 2,
        'freq_mask_prob': 0.8,
        'time_mask_prob': 0.8,
        'time_warp_prob': 0.5
    },
    'aggressive': {
        'freq_mask_param': 30,
        'time_mask_param': 60,
        'time_warp_param': 60,
        'num_freq_masks': 3,
        'num_time_masks': 3,
        'freq_mask_prob': 0.9,
        'time_mask_prob': 0.9,
        'time_warp_prob': 0.7
    },
    'military_optimized': {
        'freq_mask_param': 25,
        'time_mask_param': 35,
        'time_warp_param': 30,
        'num_freq_masks': 2,
        'num_time_masks': 2,
        'freq_mask_prob': 0.7,
        'time_mask_prob': 0.8,
        'time_warp_prob': 0.4  # Less time warping to preserve temporal patterns
    }
}


def create_specaugment(policy: str = 'standard', **kwargs) -> SpecAugment:
    """
    Create SpecAugment with predefined policy.
    
    Args:
        policy: Policy name ('mild', 'standard', 'aggressive', 'military_optimized')
        **kwargs: Override parameters
        
    Returns:
        SpecAugment instance
    """
    if policy not in SPECAUGMENT_POLICIES:
        raise ValueError(f"Unknown policy: {policy}. Available: {list(SPECAUGMENT_POLICIES.keys())}")
    
    params = SPECAUGMENT_POLICIES[policy].copy()
    params.update(kwargs)
    params['policy'] = policy
    
    return SpecAugment(**params)


def get_specaugment_info() -> Dict[str, Any]:
    """
    Get information about available SpecAugment policies.
    
    Returns:
        Dictionary with policy information
    """
    return {
        'available_policies': list(SPECAUGMENT_POLICIES.keys()),
        'policy_details': SPECAUGMENT_POLICIES,
        'description': {
            'mild': 'Light augmentation for stable training',
            'standard': 'Balanced augmentation for general use',
            'aggressive': 'Heavy augmentation for robust models',
            'military_optimized': 'Optimized for military vehicle detection'
        }
    }