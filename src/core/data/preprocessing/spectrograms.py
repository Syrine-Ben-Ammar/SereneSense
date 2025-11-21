#
# Plan:
# 1. Implement various spectrogram generation techniques (Mel, STFT, CQT)
# 2. GPU-accelerated spectrogram computation for real-time processing
# 3. Military-optimized frequency ranges and parameters
# 4. Efficient caching and batch processing capabilities
# 5. Support for different windowing functions and overlap strategies
# 6. Log-mel, power, and magnitude spectrograms
# 7. Real-time streaming spectrogram generation
#

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class SpectrogramConfig:
    """Configuration for spectrogram generation."""
    
    # Basic parameters
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = 'hann'
    
    # Frequency parameters
    fmin: float = 0.0
    fmax: Optional[float] = None
    
    # Mel parameters
    n_mels: int = 128
    mel_scale: str = 'htk'  # 'htk' or 'slaney'
    
    # Output parameters
    power: float = 2.0
    normalized: bool = False
    
    # GPU acceleration
    use_gpu: bool = True
    center: bool = True
    pad_mode: str = 'reflect'


class SpectrogramGenerator(nn.Module):
    """Base class for spectrogram generators."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: str = 'hann',
        use_gpu: bool = True
    ):
        """
        Initialize base spectrogram generator.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length
            window: Window function name
            use_gpu: Use GPU acceleration when available
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Create window tensor
        if window == 'hann':
            window_tensor = torch.hann_window(self.win_length)
        elif window == 'hamming':
            window_tensor = torch.hamming_window(self.win_length)
        elif window == 'blackman':
            window_tensor = torch.blackman_window(self.win_length)
        elif window == 'bartlett':
            window_tensor = torch.bartlett_window(self.win_length)
        else:
            window_tensor = torch.hann_window(self.win_length)
        
        # Register as buffer so it moves with the model
        self.register_buffer('window_tensor', window_tensor)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Generate spectrogram from waveform."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _ensure_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to GPU if available and requested."""
        if self.use_gpu and not tensor.is_cuda:
            return tensor.cuda()
        return tensor
    
    def _validate_input(self, waveform: torch.Tensor) -> torch.Tensor:
        """Validate and prepare input waveform."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        elif waveform.dim() > 2:
            raise ValueError(f"Waveform must be 1D or 2D, got {waveform.dim()}D")
        
        return self._ensure_gpu(waveform)


class MelSpectrogramGenerator(SpectrogramGenerator):
    """Generate mel-spectrograms optimized for military vehicle detection."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: str = 'hann',
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        mel_scale: str = 'htk',
        power: float = 2.0,
        normalized: bool = False,
        use_gpu: bool = True
    ):
        """
        Initialize mel-spectrogram generator.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length
            window: Window function name
            n_mels: Number of mel filterbanks
            fmin: Minimum frequency
            fmax: Maximum frequency (None = sample_rate/2)
            mel_scale: Mel scale type ('htk' or 'slaney')
            power: Power for magnitude spectrogram
            normalized: Whether to normalize mel filterbank
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(sample_rate, n_fft, hop_length, win_length, window, use_gpu)
        
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2.0
        self.mel_scale = mel_scale
        self.power = power
        self.normalized = normalized
        
        # Create mel filterbank
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            window_fn=self._get_window_fn(),
            n_mels=n_mels,
            f_min=fmin,
            f_max=self.fmax,
            power=power,
            normalized=normalized,
            mel_scale=mel_scale,
            center=True,
            pad_mode='reflect'
        )
        
        # Move to GPU if requested
        if self.use_gpu:
            self.mel_transform = self.mel_transform.cuda()
    
    def _get_window_fn(self) -> Callable:
        """Get window function."""
        if self.window == 'hann':
            return torch.hann_window
        elif self.window == 'hamming':
            return torch.hamming_window
        elif self.window == 'blackman':
            return torch.blackman_window
        elif self.window == 'bartlett':
            return torch.bartlett_window
        else:
            return torch.hann_window
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Generate mel-spectrogram from waveform.
        
        Args:
            waveform: Input waveform [channels, time] or [time]
            
        Returns:
            Mel-spectrogram [channels, mels, time]
        """
        waveform = self._validate_input(waveform)
        
        # Generate mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        return mel_spec


class LogMelSpectrogram(MelSpectrogramGenerator):
    """Generate log mel-spectrograms with configurable log base."""
    
    def __init__(
        self,
        log_base: str = 'natural',  # 'natural', '10', '2'
        log_offset: float = 1e-8,
        **kwargs
    ):
        """
        Initialize log mel-spectrogram generator.
        
        Args:
            log_base: Base for logarithm ('natural', '10', '2')
            log_offset: Small value added before log to avoid -inf
            **kwargs: Arguments for MelSpectrogramGenerator
        """
        super().__init__(**kwargs)
        
        self.log_base = log_base
        self.log_offset = log_offset
        
        # Create appropriate transform
        if log_base == 'natural':
            self.log_transform = lambda x: torch.log(x + log_offset)
        elif log_base == '10':
            self.log_transform = lambda x: torch.log10(x + log_offset)
        elif log_base == '2':
            self.log_transform = lambda x: torch.log2(x + log_offset)
        else:
            raise ValueError(f"Unsupported log base: {log_base}")
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Generate log mel-spectrogram from waveform.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Log mel-spectrogram
        """
        # Get mel-spectrogram
        mel_spec = super().forward(waveform)
        
        # Apply logarithm
        log_mel_spec = self.log_transform(mel_spec)
        
        return log_mel_spec


class STFTGenerator(SpectrogramGenerator):
    """Generate STFT spectrograms."""
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
        normalized: bool = False,
        onesided: bool = True,
        return_complex: bool = False,
        use_gpu: bool = True
    ):
        """
        Initialize STFT generator.
        
        Args:
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window length
            window: Window function name
            center: Whether to center the signal
            pad_mode: Padding mode
            normalized: Whether to normalize by window sum
            onesided: Whether to return one-sided FFT
            return_complex: Whether to return complex values
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(16000, n_fft, hop_length, win_length, window, use_gpu)
        
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.return_complex = return_complex
        
        # Create STFT transform
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            window_fn=self._get_window_fn(),
            power=None if return_complex else 2.0,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided
        )
        
        if self.use_gpu:
            self.stft_transform = self.stft_transform.cuda()
    
    def _get_window_fn(self) -> Callable:
        """Get window function."""
        if self.window == 'hann':
            return torch.hann_window
        elif self.window == 'hamming':
            return torch.hamming_window
        elif self.window == 'blackman':
            return torch.blackman_window
        elif self.window == 'bartlett':
            return torch.bartlett_window
        else:
            return torch.hann_window
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Generate STFT spectrogram from waveform.
        
        Args:
            waveform: Input waveform
            
        Returns:
            STFT spectrogram
        """
        waveform = self._validate_input(waveform)
        
        if self.return_complex:
            # Compute complex STFT manually
            stft_complex = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window_tensor,
                center=self.center,
                pad_mode=self.pad_mode,
                normalized=self.normalized,
                onesided=self.onesided,
                return_complex=True
            )
            return stft_complex
        else:
            # Use torchaudio transform for magnitude/power
            spec = self.stft_transform(waveform)
            return spec


class CQTGenerator(SpectrogramGenerator):
    """Generate Constant-Q Transform spectrograms."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 512,
        fmin: float = 20.0,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        tuning: float = 0.0,
        window: str = 'hann',
        use_gpu: bool = True
    ):
        """
        Initialize CQT generator.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length
            fmin: Minimum frequency
            n_bins: Number of frequency bins
            bins_per_octave: Bins per octave
            tuning: Tuning offset in cents
            window: Window function name
            use_gpu: Use GPU acceleration when available
        """
        # CQT doesn't use n_fft in the same way, but we need it for the parent class
        super().__init__(sample_rate, 2048, hop_length, None, window, use_gpu)
        
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.tuning = tuning
        
        # Note: torchaudio doesn't have a built-in CQT transform
        # This is a simplified implementation focusing on the interface
        logger.warning("CQT implementation is simplified. Consider using librosa for full CQT functionality.")
        
        # Calculate Q factor
        self.Q = 1.0 / (2 ** (1.0 / bins_per_octave) - 1)
        
        # Pre-compute frequency bins
        self.frequencies = self._compute_frequencies()
    
    def _compute_frequencies(self) -> torch.Tensor:
        """Compute CQT frequency bins."""
        frequencies = []
        for k in range(self.n_bins):
            freq = self.fmin * (2 ** (k / self.bins_per_octave))
            frequencies.append(freq)
        
        return torch.tensor(frequencies)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Generate CQT spectrogram from waveform.
        
        Args:
            waveform: Input waveform
            
        Returns:
            CQT spectrogram (simplified implementation)
        """
        waveform = self._validate_input(waveform)
        
        # Simplified CQT using multiple STFT with different window sizes
        # This is not a true CQT but provides similar multi-resolution analysis
        
        batch_size, num_channels, length = waveform.shape[0], waveform.shape[0], waveform.shape[-1]
        num_frames = (length - self.hop_length) // self.hop_length + 1
        
        cqt_spec = torch.zeros(
            num_channels, self.n_bins, num_frames,
            device=waveform.device, dtype=waveform.dtype
        )
        
        # Process each frequency bin
        for k, freq in enumerate(self.frequencies):
            # Calculate window length based on frequency
            window_length = int(self.sample_rate * self.Q / freq)
            window_length = min(window_length, length // 4)  # Ensure reasonable size
            
            if window_length < 32:  # Skip very small windows
                continue
            
            # Create window
            if self.window == 'hann':
                window = torch.hann_window(window_length, device=waveform.device)
            else:
                window = torch.hann_window(window_length, device=waveform.device)
            
            # Compute STFT for this frequency band
            try:
                stft = torch.stft(
                    waveform.squeeze(0) if waveform.dim() > 1 else waveform,
                    n_fft=window_length,
                    hop_length=self.hop_length,
                    win_length=window_length,
                    window=window,
                    center=True,
                    return_complex=True
                )
                
                # Extract magnitude at the target frequency bin
                freq_bin = int(freq * window_length / self.sample_rate)
                freq_bin = min(freq_bin, stft.shape[0] - 1)
                
                magnitude = torch.abs(stft[freq_bin, :])
                
                # Store in CQT spectrogram
                min_frames = min(magnitude.shape[0], cqt_spec.shape[-1])
                if waveform.dim() > 1:
                    cqt_spec[:, k, :min_frames] = magnitude[:min_frames].unsqueeze(0)
                else:
                    cqt_spec[0, k, :min_frames] = magnitude[:min_frames]
                    
            except Exception as e:
                logger.warning(f"CQT computation failed for frequency {freq}: {e}")
                continue
        
        return cqt_spec


class PowerSpectrogram(STFTGenerator):
    """Generate power spectrograms."""
    
    def __init__(self, power: float = 2.0, **kwargs):
        """
        Initialize power spectrogram generator.
        
        Args:
            power: Power for magnitude spectrogram (2.0 for power, 1.0 for magnitude)
            **kwargs: Arguments for STFTGenerator
        """
        super().__init__(**kwargs)
        self.power = power
        
        # Update transform to use specific power
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=self._get_window_fn(),
            power=power,
            normalized=self.normalized,
            center=self.center,
            pad_mode=self.pad_mode,
            onesided=self.onesided
        )
        
        if self.use_gpu:
            self.stft_transform = self.stft_transform.cuda()


class MagnitudeSpectrogram(PowerSpectrogram):
    """Generate magnitude spectrograms (power=1.0)."""
    
    def __init__(self, **kwargs):
        """Initialize magnitude spectrogram generator."""
        super().__init__(power=1.0, **kwargs)


class MultiResolutionSpectrogram(nn.Module):
    """Generate multi-resolution spectrograms with multiple window sizes."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft_list: List[int] = [1024, 2048, 4096],
        hop_length_ratio: float = 0.25,
        n_mels: int = 128,
        use_gpu: bool = True
    ):
        """
        Initialize multi-resolution spectrogram generator.
        
        Args:
            sample_rate: Audio sample rate
            n_fft_list: List of FFT sizes for different resolutions
            hop_length_ratio: Hop length as ratio of n_fft
            n_mels: Number of mel filterbanks
            use_gpu: Use GPU acceleration when available
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft_list = n_fft_list
        self.hop_length_ratio = hop_length_ratio
        self.n_mels = n_mels
        self.use_gpu = use_gpu
        
        # Create mel-spectrogram generators for each resolution
        self.generators = nn.ModuleList()
        for n_fft in n_fft_list:
            hop_length = int(n_fft * hop_length_ratio)
            generator = MelSpectrogramGenerator(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                use_gpu=use_gpu
            )
            self.generators.append(generator)
    
    def forward(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multi-resolution spectrograms.
        
        Args:
            waveform: Input waveform
            
        Returns:
            List of spectrograms at different resolutions
        """
        spectrograms = []
        
        for generator in self.generators:
            spec = generator(waveform)
            spectrograms.append(spec)
        
        return spectrograms


class StreamingSpectrogramGenerator:
    """Generate spectrograms for streaming audio with overlap-add processing."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: float = 0.5,  # seconds
        hop_length: float = 0.25,   # seconds
        spectrogram_generator: Optional[SpectrogramGenerator] = None,
        use_gpu: bool = True
    ):
        """
        Initialize streaming spectrogram generator.
        
        Args:
            sample_rate: Audio sample rate
            frame_length: Length of each frame in seconds
            hop_length: Hop length in seconds
            spectrogram_generator: Spectrogram generator to use
            use_gpu: Use GPU acceleration when available
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.use_gpu = use_gpu
        
        # Frame and hop sizes in samples
        self.frame_size = int(frame_length * sample_rate)
        self.hop_size = int(hop_length * sample_rate)
        
        # Default spectrogram generator
        if spectrogram_generator is None:
            self.spec_generator = MelSpectrogramGenerator(
                sample_rate=sample_rate,
                use_gpu=use_gpu
            )
        else:
            self.spec_generator = spectrogram_generator
        
        # Buffer for overlap processing
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset internal buffer."""
        self.buffer = torch.zeros(0)
        if self.use_gpu:
            self.buffer = self.buffer.cuda()
    
    def process_chunk(self, audio_chunk: torch.Tensor) -> List[torch.Tensor]:
        """
        Process audio chunk and return spectrograms.
        
        Args:
            audio_chunk: Input audio chunk
            
        Returns:
            List of spectrograms generated from frames
        """
        if audio_chunk.dim() > 1:
            audio_chunk = audio_chunk.squeeze()
        
        if self.use_gpu and not audio_chunk.is_cuda:
            audio_chunk = audio_chunk.cuda()
        
        # Add to buffer
        self.buffer = torch.cat([self.buffer, audio_chunk])
        
        spectrograms = []
        
        # Process complete frames
        while len(self.buffer) >= self.frame_size:
            # Extract frame
            frame = self.buffer[:self.frame_size]
            
            # Generate spectrogram
            spec = self.spec_generator(frame.unsqueeze(0))
            spectrograms.append(spec)
            
            # Advance buffer by hop size
            self.buffer = self.buffer[self.hop_size:]
        
        return spectrograms
    
    def flush(self) -> List[torch.Tensor]:
        """
        Process remaining audio in buffer.
        
        Returns:
            List of spectrograms from remaining buffer
        """
        spectrograms = []
        
        # Process remaining buffer if it has enough samples
        if len(self.buffer) > self.frame_size // 2:  # At least half frame
            # Pad to frame size if needed
            if len(self.buffer) < self.frame_size:
                padding = self.frame_size - len(self.buffer)
                self.buffer = torch.cat([
                    self.buffer,
                    torch.zeros(padding, device=self.buffer.device)
                ])
            
            # Generate final spectrogram
            spec = self.spec_generator(self.buffer[:self.frame_size].unsqueeze(0))
            spectrograms.append(spec)
        
        # Reset buffer
        self.reset_buffer()
        
        return spectrograms


def create_spectrogram_generator(
    spec_type: str = 'mel',
    config: Optional[SpectrogramConfig] = None,
    **kwargs
) -> SpectrogramGenerator:
    """
    Factory function to create spectrogram generators.
    
    Args:
        spec_type: Type of spectrogram ('mel', 'log_mel', 'stft', 'cqt', 'power', 'magnitude')
        config: Spectrogram configuration
        **kwargs: Override configuration parameters
        
    Returns:
        Spectrogram generator instance
    """
    if config is None:
        config = SpectrogramConfig()
    
    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create appropriate generator
    if spec_type == 'mel':
        return MelSpectrogramGenerator(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=config.window,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
            mel_scale=config.mel_scale,
            power=config.power,
            normalized=config.normalized,
            use_gpu=config.use_gpu
        )
    elif spec_type == 'log_mel':
        return LogMelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=config.window,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
            mel_scale=config.mel_scale,
            power=config.power,
            normalized=config.normalized,
            use_gpu=config.use_gpu
        )
    elif spec_type == 'stft':
        return STFTGenerator(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=config.window,
            normalized=config.normalized,
            use_gpu=config.use_gpu
        )
    elif spec_type == 'cqt':
        return CQTGenerator(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            fmin=config.fmin,
            use_gpu=config.use_gpu
        )
    elif spec_type == 'power':
        return PowerSpectrogram(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=config.window,
            power=config.power,
            normalized=config.normalized,
            use_gpu=config.use_gpu
        )
    elif spec_type == 'magnitude':
        return MagnitudeSpectrogram(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=config.window,
            normalized=config.normalized,
            use_gpu=config.use_gpu
        )
    else:
        raise ValueError(f"Unknown spectrogram type: {spec_type}")


def get_spectrogram_info():
    """
    Get information about available spectrogram types.

    Returns:
        Dictionary with spectrogram information
    """
    return {
        'available_types': [
            'mel', 'log_mel', 'stft', 'cqt', 'power', 'magnitude'
        ],
        'description': {
            'mel': 'Mel-scale spectrogram optimized for audio classification',
            'log_mel': 'Log mel-spectrogram with configurable log base',
            'stft': 'Short-time Fourier transform spectrogram',
            'cqt': 'Constant-Q transform (simplified implementation)',
            'power': 'Power spectrogram (magnitude squared)',
            'magnitude': 'Magnitude spectrogram'
        },
        'recommended_for_military': 'mel'
    }


# Backward compatibility aliases
# These aliases provide compatibility with legacy code that expects *Processor naming
SpectrogramProcessor = SpectrogramGenerator
MelSpectrogramProcessor = MelSpectrogramGenerator
STFTProcessor = STFTGenerator
CQTProcessor = CQTGenerator