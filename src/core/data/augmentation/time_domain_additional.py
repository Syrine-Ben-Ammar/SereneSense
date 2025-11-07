#
# Plan:
# 1. Implement comprehensive time-domain augmentations for military audio
# 2. Add noise (Gaussian, environmental), time stretching, pitch shifting
# 3. Volume changes, reverb simulation, wind noise
# 4. Military-specific augmentations: radio static, engine harmonics, Doppler effects
# 5. GPU acceleration support where possible
# 6. Configurable probability and intensity parameters
# 7. Preserve audio characteristics essential for vehicle classification
#

import torch
import torchaudio
import numpy as np
import random
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TimeAugmentationConfig:
    """Configuration for time-domain augmentations."""
    
    # Basic augmentations
    add_noise: bool = True
    noise_level: Tuple[float, float] = (0.001, 0.015)
    noise_prob: float = 0.8
    
    time_stretch: bool = True
    time_stretch_range: Tuple[float, float] = (0.8, 1.25)
    time_stretch_prob: float = 0.5
    
    pitch_shift: bool = True
    pitch_shift_range: Tuple[int, int] = (-4, 4)  # semitones
    pitch_shift_prob: float = 0.4
    
    volume_change: bool = True
    volume_range: Tuple[float, float] = (0.5, 2.0)
    volume_prob: float = 0.7
    
    # Environmental augmentations
    add_reverb: bool = True
    reverb_prob: float = 0.3
    room_size_range: Tuple[float, float] = (0.1, 0.8)
    
    add_wind: bool = True
    wind_prob: float = 0.2
    wind_intensity_range: Tuple[float, float] = (0.1, 0.4)
    
    # Military-specific augmentations
    radio_static: bool = True
    radio_static_prob: float = 0.1
    static_intensity_range: Tuple[float, float] = (0.05, 0.15)
    
    engine_harmonics: bool = True
    harmonics_prob: float = 0.15
    harmonics_intensity_range: Tuple[float, float] = (0.1, 0.3)
    
    doppler_effect: bool = True
    doppler_prob: float = 0.1
    speed_range: Tuple[float, float] = (10.0, 80.0)  # km/h
    
    # Global settings
    augmentation_prob: float = 0.8
    max_augmentations: int = 3


class AddNoise:
    """Add various types of noise to audio signals."""
    
    def __init__(
        self,
        noise_level: Tuple[float, float] = (0.001, 0.015),
        noise_type: str = 'gaussian'
    ):
        """
        Initialize noise augmentation.
        
        Args:
            noise_level: Range of noise levels to apply
            noise_type: Type of noise ('gaussian', 'pink', 'brown')
        """
        self.noise_level = noise_level
        self.noise_type = noise_type
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add noise to waveform."""
        noise_intensity = random.uniform(*self.noise_level)
        
        if self.noise_type == 'gaussian':
            noise = self._gaussian_noise(waveform, noise_intensity)
        elif self.noise_type == 'pink':
            noise = self._pink_noise(waveform, noise_intensity)
        elif self.noise_type == 'brown':
            noise = self._brown_noise(waveform, noise_intensity)
        else:
            noise = self._gaussian_noise(waveform, noise_intensity)
        
        # Scale noise relative to signal RMS
        signal_rms = torch.sqrt(torch.mean(waveform ** 2))
        noise = noise * signal_rms * noise_intensity
        
        return waveform + noise
    
    def _gaussian_noise(self, waveform: torch.Tensor, intensity: float) -> torch.Tensor:
        """Generate Gaussian white noise."""
        return torch.randn_like(waveform) * intensity
    
    def _pink_noise(self, waveform: torch.Tensor, intensity: float) -> torch.Tensor:
        """Generate pink (1/f) noise."""
        # Create pink noise using filtering
        white_noise = torch.randn_like(waveform)
        
        # Simple pink noise filter (approximation)
        if waveform.dim() > 1:
            for channel in range(waveform.shape[0]):
                for i in range(1, white_noise.shape[-1]):
                    white_noise[channel, i] = 0.7 * white_noise[channel, i] + 0.3 * white_noise[channel, i-1]
        else:
            for i in range(1, white_noise.shape[-1]):
                white_noise[i] = 0.7 * white_noise[i] + 0.3 * white_noise[i-1]
        
        return white_noise * intensity
    
    def _brown_noise(self, waveform: torch.Tensor, intensity: float) -> torch.Tensor:
        """Generate brown (1/f²) noise."""
        white_noise = torch.randn_like(waveform)
        
        # Brown noise filter (stronger low-pass)
        if waveform.dim() > 1:
            for channel in range(waveform.shape[0]):
                for i in range(1, white_noise.shape[-1]):
                    white_noise[channel, i] = 0.9 * white_noise[channel, i-1] + 0.1 * white_noise[channel, i]
        else:
            for i in range(1, white_noise.shape[-1]):
                white_noise[i] = 0.9 * white_noise[i-1] + 0.1 * white_noise[i]
        
        return white_noise * intensity


class TimeStretch:
    """Time stretching without changing pitch."""
    
    def __init__(
        self,
        stretch_range: Tuple[float, float] = (0.8, 1.25),
        sample_rate: int = 16000
    ):
        """
        Initialize time stretch augmentation.
        
        Args:
            stretch_range: Range of stretch factors (1.0 = no change)
            sample_rate: Audio sample rate
        """
        self.stretch_range = stretch_range
        self.sample_rate = sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time stretching to waveform."""
        stretch_factor = random.uniform(*self.stretch_range)
        
        if abs(stretch_factor - 1.0) < 0.01:  # Skip if minimal change
            return waveform
        
        try:
            # Use phase vocoder for high-quality time stretching
            return self._phase_vocoder_stretch(waveform, stretch_factor)
        except Exception as e:
            logger.warning(f"Phase vocoder failed, using simple resampling: {e}")
            return self._simple_stretch(waveform, stretch_factor)
    
    def _phase_vocoder_stretch(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """High-quality time stretching using phase vocoder."""
        # Ensure 1D for STFT
        original_shape = waveform.shape
        if waveform.dim() > 1:
            waveform_1d = waveform.squeeze()
        else:
            waveform_1d = waveform
        
        # STFT parameters
        n_fft = 2048
        hop_length = 512
        
        # Compute STFT
        stft = torch.stft(
            waveform_1d,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=waveform.device),
            return_complex=True
        )
        
        # Time stretch in frequency domain
        stretched_length = int(stft.shape[-1] / factor)
        
        if stretched_length > 0:
            # Interpolate magnitude and phase separately
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Stretch magnitude
            stretched_mag = torch.nn.functional.interpolate(
                magnitude.unsqueeze(0),
                size=(magnitude.shape[0], stretched_length),
                mode='linear',
                align_corners=False
            ).squeeze(0)
            
            # Stretch phase (more complex, simplified here)
            stretched_phase = torch.nn.functional.interpolate(
                phase.unsqueeze(0),
                size=(phase.shape[0], stretched_length),
                mode='linear',
                align_corners=False
            ).squeeze(0)
            
            # Reconstruct complex spectrogram
            stretched_stft = stretched_mag * torch.exp(1j * stretched_phase)
            
            # Convert back to time domain
            stretched = torch.istft(
                stretched_stft,
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft, device=waveform.device)
            )
            
            # Restore original shape
            if len(original_shape) > 1:
                stretched = stretched.unsqueeze(0)
            
            return stretched
        
        return waveform
    
    def _simple_stretch(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Simple time stretching using interpolation."""
        original_length = waveform.shape[-1]
        new_length = int(original_length / factor)
        
        if new_length > 0:
            return torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        
        return waveform


class PitchShift:
    """Pitch shifting to simulate Doppler effects."""
    
    def __init__(
        self,
        semitone_range: Tuple[int, int] = (-4, 4),
        sample_rate: int = 16000
    ):
        """
        Initialize pitch shift augmentation.
        
        Args:
            semitone_range: Range of semitones to shift
            sample_rate: Audio sample rate
        """
        self.semitone_range = semitone_range
        self.sample_rate = sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting to waveform."""
        semitones = random.randint(*self.semitone_range)
        
        if semitones == 0:
            return waveform
        
        # Convert semitones to frequency ratio
        pitch_factor = 2 ** (semitones / 12.0)
        
        try:
            return self._pitch_shift_with_resampling(waveform, pitch_factor)
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return waveform
    
    def _pitch_shift_with_resampling(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Pitch shift using time stretching + resampling."""
        # Step 1: Time stretch by 1/factor (without pitch change)
        time_stretcher = TimeStretch(stretch_range=(1/factor, 1/factor))
        stretched = time_stretcher(waveform)
        
        # Step 2: Resample to original length (changes pitch)
        original_length = waveform.shape[-1]
        stretched_length = stretched.shape[-1]
        
        if stretched_length != original_length:
            # Use interpolation to restore original length
            stretched = torch.nn.functional.interpolate(
                stretched.unsqueeze(0),
                size=original_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        
        return stretched


class VolumeChange:
    """Volume (amplitude) scaling."""
    
    def __init__(self, volume_range: Tuple[float, float] = (0.5, 2.0)):
        """
        Initialize volume change augmentation.
        
        Args:
            volume_range: Range of volume scaling factors
        """
        self.volume_range = volume_range
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply volume change to waveform."""
        volume_factor = random.uniform(*self.volume_range)
        return waveform * volume_factor


class Reverb:
    """Add reverberation to simulate different acoustic environments."""
    
    def __init__(
        self,
        room_size_range: Tuple[float, float] = (0.1, 0.8),
        sample_rate: int = 16000
    ):
        """
        Initialize reverb augmentation.
        
        Args:
            room_size_range: Range of room sizes (0-1)
            sample_rate: Audio sample rate
        """
        self.room_size_range = room_size_range
        self.sample_rate = sample_rate
        self._impulse_cache = {}
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply reverb to waveform."""
        room_size = random.uniform(*self.room_size_range)
        
        # Generate or retrieve impulse response
        impulse_key = f"{room_size:.2f}"
        if impulse_key not in self._impulse_cache:
            self._impulse_cache[impulse_key] = self._generate_impulse_response(room_size)
        
        impulse_response = self._impulse_cache[impulse_key]
        
        # Apply convolution
        return self._convolve_with_impulse(waveform, impulse_response)
    
    def _generate_impulse_response(self, room_size: float) -> torch.Tensor:
        """Generate simple impulse response for reverb."""
        # Simple exponential decay impulse response
        decay_time = room_size * 0.5  # seconds
        impulse_length = int(decay_time * self.sample_rate)
        
        if impulse_length == 0:
            return torch.tensor([1.0])
        
        t = torch.arange(impulse_length, dtype=torch.float32) / self.sample_rate
        
        # Exponential decay with some randomness
        decay_rate = 3.0 / decay_time  # T60 approximation
        impulse = torch.exp(-decay_rate * t)
        
        # Add some late reflections
        late_reflections = torch.randn(impulse_length) * 0.1 * impulse
        impulse = impulse + late_reflections
        
        # Normalize
        impulse = impulse / torch.max(torch.abs(impulse))
        
        return impulse
    
    def _convolve_with_impulse(self, waveform: torch.Tensor, impulse: torch.Tensor) -> torch.Tensor:
        """Convolve waveform with impulse response."""
        # Ensure impulse is on same device
        impulse = impulse.to(waveform.device)
        
        if waveform.dim() == 1:
            # 1D convolution
            convolved = torch.conv1d(
                waveform.unsqueeze(0).unsqueeze(0),
                impulse.unsqueeze(0).unsqueeze(0),
                padding=len(impulse) // 2
            ).squeeze()
        else:
            # Multi-channel
            convolved = torch.conv1d(
                waveform.unsqueeze(0),
                impulse.unsqueeze(0).unsqueeze(0),
                padding=len(impulse) // 2,
                groups=waveform.shape[0]
            ).squeeze(0)
        
        # Trim to original length
        original_length = waveform.shape[-1]
        if convolved.shape[-1] > original_length:
            convolved = convolved[..., :original_length]
        
        return convolved


class WindNoise:
    """Add wind noise to simulate outdoor conditions."""
    
    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.1, 0.4),
        sample_rate: int = 16000
    ):
        """
        Initialize wind noise augmentation.
        
        Args:
            intensity_range: Range of wind noise intensity
            sample_rate: Audio sample rate
        """
        self.intensity_range = intensity_range
        self.sample_rate = sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add wind noise to waveform."""
        intensity = random.uniform(*self.intensity_range)
        
        # Generate wind-like noise (low-frequency emphasis)
        wind_noise = self._generate_wind_noise(waveform.shape, intensity)
        
        return waveform + wind_noise
    
    def _generate_wind_noise(self, shape: torch.Size, intensity: float) -> torch.Tensor:
        """Generate wind-like noise with low-frequency emphasis."""
        # Start with white noise
        noise = torch.randn(shape)
        
        # Apply low-pass filtering to emphasize low frequencies
        # Simple IIR filter approximation
        filtered_noise = torch.zeros_like(noise)
        alpha = 0.1  # Low-pass filter coefficient
        
        if noise.dim() == 1:
            filtered_noise[0] = noise[0]
            for i in range(1, len(noise)):
                filtered_noise[i] = alpha * noise[i] + (1 - alpha) * filtered_noise[i-1]
        else:
            filtered_noise[:, 0] = noise[:, 0]
            for i in range(1, noise.shape[-1]):
                filtered_noise[:, i] = alpha * noise[:, i] + (1 - alpha) * filtered_noise[:, i-1]
        
        # Add some gustiness (amplitude modulation)
        time_steps = shape[-1]
        t = torch.arange(time_steps, dtype=torch.float32) / self.sample_rate
        gust_freq = random.uniform(0.1, 2.0)  # Hz
        amplitude_mod = 1.0 + 0.5 * torch.sin(2 * math.pi * gust_freq * t)
        
        if noise.dim() > 1:
            amplitude_mod = amplitude_mod.unsqueeze(0).expand_as(filtered_noise)
        
        return filtered_noise * amplitude_mod * intensity


class RadioStatic:
    """Add radio static/interference."""
    
    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.05, 0.15),
        sample_rate: int = 16000
    ):
        """
        Initialize radio static augmentation.
        
        Args:
            intensity_range: Range of static intensity
            sample_rate: Audio sample rate
        """
        self.intensity_range = intensity_range
        self.sample_rate = sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add radio static to waveform."""
        intensity = random.uniform(*self.intensity_range)
        
        # Generate static noise
        static = self._generate_static(waveform.shape, intensity)
        
        return waveform + static
    
    def _generate_static(self, shape: torch.Size, intensity: float) -> torch.Tensor:
        """Generate radio static noise."""
        # High-frequency noise with crackles
        static = torch.randn(shape) * intensity
        
        # Add occasional crackles (impulses)
        crackle_prob = 0.001  # Probability per sample
        crackles = torch.rand(shape) < crackle_prob
        crackle_amplitude = intensity * 5.0
        
        static += crackles.float() * crackle_amplitude * torch.randn(shape)
        
        return static


class EngineHarmonics:
    """Enhance or add engine harmonic patterns."""
    
    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.1, 0.3),
        sample_rate: int = 16000
    ):
        """
        Initialize engine harmonics augmentation.
        
        Args:
            intensity_range: Range of harmonics intensity
            sample_rate: Audio sample rate
        """
        self.intensity_range = intensity_range
        self.sample_rate = sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add engine harmonics to waveform."""
        intensity = random.uniform(*self.intensity_range)
        
        # Generate harmonic content
        harmonics = self._generate_harmonics(waveform.shape, intensity)
        
        return waveform + harmonics
    
    def _generate_harmonics(self, shape: torch.Size, intensity: float) -> torch.Tensor:
        """Generate engine-like harmonic patterns."""
        time_steps = shape[-1]
        t = torch.arange(time_steps, dtype=torch.float32) / self.sample_rate
        
        # Random fundamental frequency (typical engine range)
        fundamental = random.uniform(30.0, 120.0)  # Hz
        
        # Generate harmonic series
        harmonics = torch.zeros(shape)
        
        for harmonic in range(1, 6):  # First 5 harmonics
            freq = fundamental * harmonic
            if freq < self.sample_rate / 2:  # Avoid aliasing
                amplitude = intensity / harmonic  # Decreasing amplitude
                
                # Add some randomness to make it more realistic
                amplitude *= random.uniform(0.5, 1.5)
                phase = random.uniform(0, 2 * math.pi)
                
                harmonic_signal = amplitude * torch.sin(2 * math.pi * freq * t + phase)
                
                if harmonics.dim() > 1:
                    harmonic_signal = harmonic_signal.unsqueeze(0).expand_as(harmonics)
                
                harmonics += harmonic_signal
        
        return harmonics


class TimeAugmentation:
    """
    Main time-domain augmentation class combining all techniques.
    """
    
    def __init__(
        self,
        config: Optional[TimeAugmentationConfig] = None,
        use_gpu: bool = True
    ):
        """
        Initialize time augmentation pipeline.
        
        Args:
            config: Augmentation configuration
            use_gpu: Use GPU acceleration when available
        """
        self.config = config or TimeAugmentationConfig()
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize individual augmentations
        self.noise_aug = AddNoise(self.config.noise_level)
        self.time_stretch_aug = TimeStretch(self.config.time_stretch_range)
        self.pitch_shift_aug = PitchShift(self.config.pitch_shift_range)
        self.volume_aug = VolumeChange(self.config.volume_range)
        self.reverb_aug = Reverb(self.config.room_size_range)
        self.wind_aug = WindNoise(self.config.wind_intensity_range)
        self.static_aug = RadioStatic(self.config.static_intensity_range)
        self.harmonics_aug = EngineHarmonics(self.config.harmonics_intensity_range)
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random time-domain augmentations.
        
        Args:
            waveform: Input audio tensor [channels, time]
            
        Returns:
            Augmented audio tensor
        """
        if random.random() > self.config.augmentation_prob:
            return waveform
        
        # Move to GPU if available
        if self.use_gpu and not waveform.is_cuda:
            waveform = waveform.cuda()
        
        # Collect available augmentations
        augmentations = []
        
        if self.config.add_noise and random.random() < self.config.noise_prob:
            augmentations.append(('noise', self.noise_aug))
        
        if self.config.time_stretch and random.random() < self.config.time_stretch_prob:
            augmentations.append(('time_stretch', self.time_stretch_aug))
        
        if self.config.pitch_shift and random.random() < self.config.pitch_shift_prob:
            augmentations.append(('pitch_shift', self.pitch_shift_aug))
        
        if self.config.volume_change and random.random() < self.config.volume_prob:
            augmentations.append(('volume', self.volume_aug))
        
        if self.config.add_reverb and random.random() < self.config.reverb_prob:
            augmentations.append(('reverb', self.reverb_aug))
        
        if self.config.add_wind and random.random() < self.config.wind_prob:
            augmentations.append(('wind', self.wind_aug))
        
        if self.config.radio_static and random.random() < self.config.radio_static_prob:
            augmentations.append(('static', self.static_aug))
        
        if self.config.engine_harmonics and random.random() < self.config.harmonics_prob:
            augmentations.append(('harmonics', self.harmonics_aug))
        
        # Randomly select and apply augmentations
        if augmentations:
            num_augmentations = min(len(augmentations), self.config.max_augmentations)
            selected_augs = random.sample(augmentations, num_augmentations)
            
            # Shuffle order
            random.shuffle(selected_augs)
            
            augmented = waveform
            for name, aug_func in selected_augs:
                try:
                    augmented = aug_func(augmented)
                except Exception as e:
                    logger.warning(f"Augmentation '{name}' failed: {e}")
                    continue
            
            return augmented
        
        return waveform
    
    def get_augmentation_names(self) -> List[str]:
        """Get list of available augmentation names."""
        return [
            'noise', 'time_stretch', 'pitch_shift', 'volume',
            'reverb', 'wind', 'static', 'harmonics'
        ]