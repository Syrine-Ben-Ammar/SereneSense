"""
SereneSense Data Preprocessing Module
Advanced audio preprocessing for military vehicle sound detection.

This module provides:
- Spectrogram generation (Mel, STFT, CQT)
- Audio normalization techniques
- Audio segmentation and windowing
- Feature extraction and enhancement
- Real-time preprocessing pipelines
- GPU-accelerated processing

All preprocessing is optimized for military deployment requirements:
- Low latency for real-time detection
- Robust performance in noisy environments
- Consistent feature extraction across different hardware
- Memory-efficient processing for edge devices
"""

from .spectrograms import (
    SpectrogramGenerator,
    MelSpectrogramGenerator,
    STFTGenerator,
    CQTGenerator,
    LogMelSpectrogram,
    PowerSpectrogram,
    MagnitudeSpectrogram
)

from .normalization import (
    AudioNormalizer,
    RMSNormalization,
    PeakNormalization,
    LUFSNormalization,
    ZScoreNormalization,
    MinMaxNormalization,
    SpectralNormalization
)

from .segmentation import (
    AudioSegmenter,
    FixedLengthSegmenter,
    OverlapSegmenter,
    SilenceBasedSegmenter,
    EnergyBasedSegmenter,
    OnsetBasedSegmenter,
    RealTimeSegmenter
)

# Preprocessing configurations for different use cases
PREPROCESSING_CONFIGS = {
    'real_time': {
        'sample_rate': 16000,
        'window_length': 0.5,  # 500ms windows for low latency
        'hop_length': 0.25,    # 250ms hop for real-time processing
        'n_mels': 128,
        'n_fft': 1024,
        'normalization': 'rms',
        'preemphasis': 0.97
    },
    'batch_processing': {
        'sample_rate': 16000,
        'window_length': 2.0,  # 2-second windows for accuracy
        'hop_length': 1.0,     # 1-second hop
        'n_mels': 128,
        'n_fft': 2048,
        'normalization': 'lufs',
        'preemphasis': 0.97
    },
    'edge_optimized': {
        'sample_rate': 16000,
        'window_length': 1.0,  # 1-second windows for edge devices
        'hop_length': 0.5,     # 500ms hop
        'n_mels': 64,          # Reduced features for efficiency
        'n_fft': 1024,
        'normalization': 'peak',
        'preemphasis': 0.97
    },
    'high_quality': {
        'sample_rate': 22050,  # Higher sample rate for better quality
        'window_length': 2.0,
        'hop_length': 0.5,
        'n_mels': 256,         # More features for accuracy
        'n_fft': 4096,
        'normalization': 'lufs',
        'preemphasis': 0.97
    }
}

# Spectrogram type configurations
SPECTROGRAM_CONFIGS = {
    'mel': {
        'n_mels': 128,
        'fmin': 0,
        'fmax': 8000,
        'mel_scale': 'htk'
    },
    'mel_military': {
        'n_mels': 128,
        'fmin': 20,    # Focus on frequencies above 20 Hz
        'fmax': 4000,  # Most vehicle sounds below 4 kHz
        'mel_scale': 'slaney'
    },
    'stft': {
        'n_fft': 2048,
        'hop_length': 512,
        'win_length': 2048,
        'window': 'hann'
    },
    'cqt': {
        'hop_length': 512,
        'fmin': 20,
        'n_bins': 84,
        'bins_per_octave': 12
    }
}

# Normalization configurations
NORMALIZATION_CONFIGS = {
    'rms': {
        'target_rms': -20,  # dB
        'max_gain': 30
    },
    'peak': {
        'target_peak': -3,  # dB
        'max_gain': 30
    },
    'lufs': {
        'target_lufs': -23,  # EBU R128 standard
        'max_gain': 30
    },
    'zscore': {
        'mean': 0.0,
        'std': 1.0
    },
    'minmax': {
        'min_val': 0.0,
        'max_val': 1.0
    }
}


def create_preprocessing_pipeline(
    config_name: str = 'batch_processing',
    spectrogram_type: str = 'mel',
    use_gpu: bool = True,
    **kwargs
):
    """
    Create a complete preprocessing pipeline.
    
    Args:
        config_name: Configuration preset ('real_time', 'batch_processing', 'edge_optimized', 'high_quality')
        spectrogram_type: Type of spectrogram ('mel', 'stft', 'cqt')
        use_gpu: Use GPU acceleration when available
        **kwargs: Override configuration parameters
        
    Returns:
        Complete preprocessing pipeline
    """
    if config_name not in PREPROCESSING_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(PREPROCESSING_CONFIGS.keys())}")
    
    if spectrogram_type not in SPECTROGRAM_CONFIGS:
        raise ValueError(f"Unknown spectrogram type: {spectrogram_type}. Available: {list(SPECTROGRAM_CONFIGS.keys())}")
    
    # Get configuration
    config = PREPROCESSING_CONFIGS[config_name].copy()
    config.update(kwargs)
    
    # Create components
    normalizer = create_normalizer(config['normalization'])
    segmenter = create_segmenter(config)
    spectrogram_generator = create_spectrogram_generator(spectrogram_type, config, use_gpu)
    
    return PreprocessingPipeline(
        normalizer=normalizer,
        segmenter=segmenter,
        spectrogram_generator=spectrogram_generator,
        config=config
    )


def create_normalizer(normalization_type: str):
    """
    Create audio normalizer.
    
    Args:
        normalization_type: Type of normalization
        
    Returns:
        Audio normalizer instance
    """
    if normalization_type == 'rms':
        return RMSNormalization(**NORMALIZATION_CONFIGS['rms'])
    elif normalization_type == 'peak':
        return PeakNormalization(**NORMALIZATION_CONFIGS['peak'])
    elif normalization_type == 'lufs':
        return LUFSNormalization(**NORMALIZATION_CONFIGS['lufs'])
    elif normalization_type == 'zscore':
        return ZScoreNormalization(**NORMALIZATION_CONFIGS['zscore'])
    elif normalization_type == 'minmax':
        return MinMaxNormalization(**NORMALIZATION_CONFIGS['minmax'])
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")


def create_segmenter(config: dict):
    """
    Create audio segmenter.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Audio segmenter instance
    """
    if config.get('window_length') and config.get('hop_length'):
        return FixedLengthSegmenter(
            segment_length=config['window_length'],
            hop_length=config['hop_length'],
            sample_rate=config['sample_rate']
        )
    else:
        return FixedLengthSegmenter(
            segment_length=2.0,
            hop_length=1.0,
            sample_rate=config['sample_rate']
        )


def create_spectrogram_generator(spec_type: str, config: dict, use_gpu: bool = True):
    """
    Create spectrogram generator.
    
    Args:
        spec_type: Type of spectrogram
        config: Configuration dictionary
        use_gpu: Use GPU acceleration
        
    Returns:
        Spectrogram generator instance
    """
    spec_config = SPECTROGRAM_CONFIGS.get(spec_type, SPECTROGRAM_CONFIGS['mel'])
    
    if spec_type == 'mel' or spec_type == 'mel_military':
        return MelSpectrogramGenerator(
            sample_rate=config['sample_rate'],
            n_fft=config.get('n_fft', 2048),
            hop_length=config.get('n_fft', 2048) // 4,
            n_mels=config.get('n_mels', 128),
            fmin=spec_config.get('fmin', 0),
            fmax=spec_config.get('fmax', None),
            use_gpu=use_gpu
        )
    elif spec_type == 'stft':
        return STFTGenerator(
            n_fft=config.get('n_fft', 2048),
            hop_length=config.get('n_fft', 2048) // 4,
            win_length=config.get('n_fft', 2048),
            use_gpu=use_gpu
        )
    elif spec_type == 'cqt':
        return CQTGenerator(
            sample_rate=config['sample_rate'],
            hop_length=config.get('n_fft', 2048) // 4,
            fmin=spec_config.get('fmin', 20),
            n_bins=spec_config.get('n_bins', 84),
            use_gpu=use_gpu
        )
    else:
        raise ValueError(f"Unknown spectrogram type: {spec_type}")


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline combining normalization, segmentation, and spectrogram generation.
    """
    
    def __init__(
        self,
        normalizer,
        segmenter,
        spectrogram_generator,
        config: dict
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            normalizer: Audio normalizer
            segmenter: Audio segmenter
            spectrogram_generator: Spectrogram generator
            config: Configuration dictionary
        """
        self.normalizer = normalizer
        self.segmenter = segmenter
        self.spectrogram_generator = spectrogram_generator
        self.config = config
    
    def __call__(self, audio, sample_rate=None):
        """
        Process audio through complete pipeline.
        
        Args:
            audio: Input audio tensor or numpy array
            sample_rate: Sample rate (if different from config)
            
        Returns:
            Processed spectrograms and metadata
        """
        import torch
        
        # Convert to tensor if needed
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure correct sample rate
        if sample_rate and sample_rate != self.config['sample_rate']:
            audio = self._resample(audio, sample_rate, self.config['sample_rate'])
        
        # Normalize audio
        normalized_audio = self.normalizer(audio)
        
        # Segment audio
        segments = self.segmenter(normalized_audio)
        
        # Generate spectrograms for each segment
        spectrograms = []
        for segment in segments:
            spectrogram = self.spectrogram_generator(segment)
            spectrograms.append(spectrogram)
        
        if spectrograms:
            spectrograms = torch.stack(spectrograms)
        else:
            # Return empty tensor with correct shape
            dummy_spec = self.spectrogram_generator(torch.zeros(1, int(self.config['sample_rate'] * 0.1)))
            spectrograms = torch.zeros(0, *dummy_spec.shape[1:])
        
        return spectrograms
    
    def _resample(self, audio, orig_sr, target_sr):
        """Resample audio to target sample rate."""
        import torchaudio
        
        if orig_sr == target_sr:
            return audio
        
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(audio)
    
    def process_file(self, file_path):
        """
        Process audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Processed spectrograms and metadata
        """
        import torchaudio
        
        # Load audio file
        audio, sample_rate = torchaudio.load(file_path)
        
        # Process through pipeline
        spectrograms = self(audio, sample_rate)
        
        return spectrograms
    
    def get_config(self):
        """Get pipeline configuration."""
        return self.config.copy()


def get_preprocessing_info():
    """
    Get information about available preprocessing configurations.
    
    Returns:
        Dictionary with preprocessing information
    """
    return {
        'preprocessing_configs': list(PREPROCESSING_CONFIGS.keys()),
        'spectrogram_types': list(SPECTROGRAM_CONFIGS.keys()),
        'normalization_types': list(NORMALIZATION_CONFIGS.keys()),
        'config_details': {
            'preprocessing': PREPROCESSING_CONFIGS,
            'spectrograms': SPECTROGRAM_CONFIGS,
            'normalization': NORMALIZATION_CONFIGS
        }
    }


__all__ = [
    # Spectrogram generators
    'SpectrogramGenerator',
    'MelSpectrogramGenerator',
    'STFTGenerator',
    'CQTGenerator',
    'LogMelSpectrogram',
    'PowerSpectrogram',
    'MagnitudeSpectrogram',
    
    # Normalizers
    'AudioNormalizer',
    'RMSNormalization',
    'PeakNormalization',
    'LUFSNormalization',
    'ZScoreNormalization',
    'MinMaxNormalization',
    'SpectralNormalization',
    
    # Segmenters
    'AudioSegmenter',
    'FixedLengthSegmenter',
    'OverlapSegmenter',
    'SilenceBasedSegmenter',
    'EnergyBasedSegmenter',
    'OnsetBasedSegmenter',
    'RealTimeSegmenter',
    
    # Pipeline utilities
    'PreprocessingPipeline',
    'create_preprocessing_pipeline',
    'create_normalizer',
    'create_segmenter',
    'create_spectrogram_generator',
    'get_preprocessing_info',
    
    # Configuration constants
    'PREPROCESSING_CONFIGS',
    'SPECTROGRAM_CONFIGS', 
    'NORMALIZATION_CONFIGS'
]