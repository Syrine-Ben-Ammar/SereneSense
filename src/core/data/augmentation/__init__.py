"""
SereneSense Data Augmentation Module
Comprehensive audio augmentation techniques for military vehicle sound detection.

This module provides:
- Time domain augmentations (noise, time stretching, pitch shifting)
- Frequency domain augmentations (spectral masking, filtering)
- SpecAugment implementation for transformer models
- Military-specific augmentations (radio static, engine harmonics)
- GPU-accelerated augmentation pipelines

All augmentations are designed to increase robustness for military deployment
while preserving acoustic characteristics essential for vehicle classification.
"""

from .time_domain import (
    TimeAugmentation,
    AddNoise,
    TimeStretch,
    PitchShift,
    VolumeChange,
    Reverb,
    WindNoise,
    RadioStatic,
    EngineHarmonics
)

from .frequency_domain import (
    FrequencyAugmentation,
    FrequencyMask,
    TimeMask,
    SpectralNormalize,
    FilterBank,
    MelSpecAugment
)

from .spec_augment import (
    SpecAugment,
    FreqMasking,
    TimeMasking,
    TimeWarping,
    CombinedSpecAugment
)

# Augmentation configurations for different use cases
MILITARY_AUGMENTATIONS = {
    'basic': {
        'add_noise': {'prob': 0.8, 'noise_level': [0.001, 0.015]},
        'time_stretch': {'prob': 0.5, 'rate_range': [0.8, 1.25]},
        'pitch_shift': {'prob': 0.5, 'semitones_range': [-4, 4]},
        'volume_change': {'prob': 0.7, 'volume_range': [0.5, 2.0]}
    },
    'field_conditions': {
        'add_noise': {'prob': 0.9, 'noise_level': [0.005, 0.025]},
        'wind_noise': {'prob': 0.3, 'intensity_range': [0.1, 0.4]},
        'radio_static': {'prob': 0.2, 'intensity_range': [0.05, 0.15]},
        'reverb': {'prob': 0.4, 'room_size_range': [0.1, 0.8]},
        'time_stretch': {'prob': 0.6, 'rate_range': [0.85, 1.15]},
        'pitch_shift': {'prob': 0.4, 'semitones_range': [-2, 2]}
    },
    'heavy_augmentation': {
        'add_noise': {'prob': 0.95, 'noise_level': [0.01, 0.03]},
        'time_stretch': {'prob': 0.8, 'rate_range': [0.7, 1.3]},
        'pitch_shift': {'prob': 0.7, 'semitones_range': [-6, 6]},
        'volume_change': {'prob': 0.8, 'volume_range': [0.3, 2.5]},
        'reverb': {'prob': 0.5, 'room_size_range': [0.1, 0.9]},
        'wind_noise': {'prob': 0.4, 'intensity_range': [0.1, 0.5]},
        'radio_static': {'prob': 0.3, 'intensity_range': [0.05, 0.2]},
        'engine_harmonics': {'prob': 0.2, 'intensity_range': [0.1, 0.3]}
    }
}

SPECAUGMENT_CONFIGS = {
    'mild': {
        'freq_mask_param': 15,
        'time_mask_param': 25,
        'num_freq_masks': 1,
        'num_time_masks': 1,
        'time_warp_param': 20
    },
    'standard': {
        'freq_mask_param': 20,
        'time_mask_param': 40,
        'num_freq_masks': 2,
        'num_time_masks': 2,
        'time_warp_param': 40
    },
    'aggressive': {
        'freq_mask_param': 30,
        'time_mask_param': 60,
        'num_freq_masks': 3,
        'num_time_masks': 3,
        'time_warp_param': 60
    }
}


def create_augmentation_pipeline(
    config_name: str = 'basic',
    specaugment_config: str = 'standard',
    use_gpu: bool = True
):
    """
    Create a complete augmentation pipeline for military vehicle detection.
    
    Args:
        config_name: Augmentation configuration ('basic', 'field_conditions', 'heavy_augmentation')
        specaugment_config: SpecAugment configuration ('mild', 'standard', 'aggressive')
        use_gpu: Use GPU-accelerated augmentations when available
        
    Returns:
        Combined augmentation pipeline
    """
    if config_name not in MILITARY_AUGMENTATIONS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MILITARY_AUGMENTATIONS.keys())}")
    
    if specaugment_config not in SPECAUGMENT_CONFIGS:
        raise ValueError(f"Unknown SpecAugment config: {specaugment_config}. Available: {list(SPECAUGMENT_CONFIGS.keys())}")
    
    # Create time domain augmentations
    time_config = MILITARY_AUGMENTATIONS[config_name]
    time_aug = TimeAugmentation(config=time_config, use_gpu=use_gpu)
    
    # Create frequency domain augmentations
    freq_aug = FrequencyAugmentation(use_gpu=use_gpu)
    
    # Create SpecAugment
    spec_config = SPECAUGMENT_CONFIGS[specaugment_config]
    spec_aug = SpecAugment(**spec_config)
    
    return CombinedAugmentation(
        time_augmentation=time_aug,
        frequency_augmentation=freq_aug,
        spec_augmentation=spec_aug
    )


class CombinedAugmentation:
    """
    Combined augmentation pipeline applying time, frequency, and spectrogram augmentations.
    """
    
    def __init__(
        self,
        time_augmentation: TimeAugmentation,
        frequency_augmentation: FrequencyAugmentation,
        spec_augmentation: SpecAugment,
        apply_prob: float = 0.8
    ):
        """
        Initialize combined augmentation pipeline.
        
        Args:
            time_augmentation: Time domain augmentation module
            frequency_augmentation: Frequency domain augmentation module
            spec_augmentation: SpecAugment module
            apply_prob: Probability of applying augmentations
        """
        self.time_aug = time_augmentation
        self.freq_aug = frequency_augmentation
        self.spec_aug = spec_augmentation
        self.apply_prob = apply_prob
    
    def __call__(self, audio, spectrogram=None):
        """
        Apply combined augmentations.
        
        Args:
            audio: Input audio tensor [channels, time]
            spectrogram: Input spectrogram tensor [freq, time] (optional)
            
        Returns:
            Augmented audio and spectrogram
        """
        import torch
        import random
        
        # Decide whether to apply augmentations
        if random.random() > self.apply_prob:
            return audio, spectrogram
        
        # Apply time domain augmentations to audio
        if self.time_aug and random.random() < 0.7:
            audio = self.time_aug(audio)
        
        # Generate spectrogram if not provided
        if spectrogram is None and (self.freq_aug or self.spec_aug):
            import torchaudio
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )
            db_transform = torchaudio.transforms.AmplitudeToDB()
            spectrogram = db_transform(mel_transform(audio))
        
        # Apply frequency domain augmentations
        if self.freq_aug and spectrogram is not None and random.random() < 0.6:
            spectrogram = self.freq_aug(spectrogram)
        
        # Apply SpecAugment
        if self.spec_aug and spectrogram is not None and random.random() < 0.8:
            spectrogram = self.spec_aug(spectrogram)
        
        return audio, spectrogram


def get_augmentation_info():
    """
    Get information about available augmentation configurations.
    
    Returns:
        Dictionary with augmentation information
    """
    return {
        'military_configs': list(MILITARY_AUGMENTATIONS.keys()),
        'specaugment_configs': list(SPECAUGMENT_CONFIGS.keys()),
        'available_augmentations': {
            'time_domain': [
                'AddNoise', 'TimeStretch', 'PitchShift', 'VolumeChange',
                'Reverb', 'WindNoise', 'RadioStatic', 'EngineHarmonics'
            ],
            'frequency_domain': [
                'FrequencyMask', 'TimeMask', 'SpectralNormalize', 
                'FilterBank', 'MelSpecAugment'
            ],
            'specaugment': [
                'FreqMasking', 'TimeMasking', 'TimeWarping', 'CombinedSpecAugment'
            ]
        }
    }


__all__ = [
    # Time domain augmentations
    'TimeAugmentation',
    'AddNoise',
    'TimeStretch', 
    'PitchShift',
    'VolumeChange',
    'Reverb',
    'WindNoise',
    'RadioStatic',
    'EngineHarmonics',
    
    # Frequency domain augmentations
    'FrequencyAugmentation',
    'FrequencyMask',
    'TimeMask',
    'SpectralNormalize',
    'FilterBank',
    'MelSpecAugment',
    
    # SpecAugment
    'SpecAugment',
    'FreqMasking',
    'TimeMasking',
    'TimeWarping',
    'CombinedSpecAugment',
    
    # Pipeline utilities
    'CombinedAugmentation',
    'create_augmentation_pipeline',
    'get_augmentation_info',
    'MILITARY_AUGMENTATIONS',
    'SPECAUGMENT_CONFIGS'
]