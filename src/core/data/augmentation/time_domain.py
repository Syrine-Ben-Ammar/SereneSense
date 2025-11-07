"""
Comprehensive Data Augmentation for Military Audio Detection
Advanced augmentation techniques for improving model robustness.

Features:
- Time-domain augmentations (noise, speed, volume)
- Frequency-domain augmentations (masking, filtering)
- SpecAugment for spectrograms
- Environmental simulation
- Military-specific augmentations
"""

import torch
import torchaudio
import numpy as np
import random
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """Augmentation configuration"""
    # Time domain augmentations
    add_noise: bool = True
    noise_level: Tuple[float, float] = (0.001, 0.015)
    
    time_stretch: bool = True
    time_stretch_range: Tuple[float, float] = (0.8, 1.25)
    
    pitch_shift: bool = True
    pitch_shift_range: Tuple[int, int] = (-4, 4)  # semitones
    
    volume_change: bool = True
    volume_range: Tuple[float, float] = (0.5, 2.0)
    
    # Frequency domain augmentations
    freq_mask: bool = True
    freq_mask_param: int = 20
    
    time_mask: bool = True
    time_mask_param: int = 40
    
    # Environmental augmentations
    add_reverb: bool = True
    reverb_prob: float = 0.3
    
    add_wind: bool = True
    wind_prob: float = 0.2
    
    # Military-specific augmentations
    radio_static: bool = True
    radio_static_prob: float = 0.1
    
    engine_harmonics: bool = True
    harmonics_prob: float = 0.15
    
    # Probabilities
    augmentation_prob: float = 0.8
    mix_prob: float = 0.2  # Probability of mixing multiple samples


class TimeAugmentation:
    """
    Time-domain audio augmentations for military vehicle sounds.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        
        # Pre-compute noise patterns for efficiency
        self._noise_cache = {}
        self._reverb_impulses = self._generate_reverb_impulses()
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random time-domain augmentations"""
        if random.random() > self.config.augmentation_prob:
            return waveform
        
        # Apply augmentations in random order
        augmentations = []
        
        if self.config.add_noise and random.random() < 0.7:
            augmentations.append(self.add_gaussian_noise)
        
        if self.config.time_stretch and random.random() < 0.4:
            augmentations.append(self.time_stretch)
        
        if self.config.pitch_shift and random.random() < 0.3:
            augmentations.append(self.pitch_shift)
        
        if self.config.volume_change and random.random() < 0.5:
            augmentations.append(self.volume_change)
        
        if self.config.add_reverb and random.random() < self.config.reverb_prob:
            augmentations.append(self.add_reverb)
        
        if self.config.add_wind and random.random() < self.config.wind_prob:
            augmentations.append(self.add_wind_noise)
        
        if self.config.radio_static and random.random() < self.config.radio_static_prob:
            augmentations.append(self.add_radio_static)
        
        # Shuffle and apply augmentations
        random.shuffle(augmentations)
        augmented = waveform
        
        for aug_func in augmentations:
            try:
                augmented = aug_func(augmented)
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                continue
        
        return augmented
    
    def add_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to simulate environmental conditions"""
        noise_level = random.uniform(*self.config.noise_level)
        noise = torch.randn_like(waveform) * noise_level
        
        # Scale noise relative to signal amplitude
        signal_rms = torch.sqrt(torch.mean(waveform ** 2))
        noise = noise * signal_rms
        
        return waveform + noise
    
    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Time stretching without pitch change"""
        stretch_factor = random.uniform(*self.config.time_stretch_range)
        
        # Use phase vocoder for high-quality time stretching
        stft = torch.stft(
            waveform.squeeze(),
            n_fft=2048,
            hop_length=512,
            return_complex=True
        )
        
        # Stretch by modifying hop length
        stretched_length = int(stft.shape[-1] / stretch_factor)
        
        if stretched_length > 0:
            # Interpolate in frequency domain
            stretched_stft = torch.nn.functional.interpolate(
                stft.unsqueeze(0).real,
                size=(stft.shape[0], stretched_length),
                mode='linear',
                align_corners=False
            ).squeeze(0)
            
            # Convert back to time domain
            stretched = torch.istft(
                torch.complex(stretched_stft, torch.zeros_like(stretched_stft)),
                n_fft=2048,
                hop_length=512
            )
            
            # Ensure same number of channels
            if waveform.dim() == 2:
                stretched = stretched.unsqueeze(0)
            
            return stretched
        
        return waveform
    
    def pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pitch shifting to simulate Doppler effects"""
        semitones = random.randint(*self.config.pitch_shift_range)
        
        if semitones == 0:
            return waveform
        
        # Convert semitones to frequency ratio
        pitch_factor = 2 ** (semitones / 12.0)
        
        # Use time stretching followed by resampling
        stretched = self._stretch_by_factor(waveform, 1 / pitch_factor)
        
        return stretched
    
    def _stretch_by_factor(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Helper function for pitch shifting"""
        try:
            # Simple resampling approach
            original_length = waveform.shape[-1]
            new_length = int(original_length * factor)
            
            if new_length > 0:
                resampled = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(0)
                
                # Pad or trim to original length
                if new_length > original_length:
                    return resampled[..., :original_length]
                else:
                    padding = original_length - new_length
                    return torch.nn.functional.pad(resampled, (0, padding))
            
        except Exception:
            pass
        
        return waveform
    
    def volume_change(self, waveform: torch.Tensor) -> torch.Tensor:
        """Random volume changes to simulate distance variations"""
        volume_factor = random.uniform(*self.config.volume_range)
        return waveform * volume_factor
    
    def add_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add reverb to simulate different environments"""
        if not self._reverb_impulses:
            return waveform
        
        # Select random reverb impulse
        impulse = random.choice(self._reverb_impulses)
        
        # Ensure impulse is shorter than waveform
        if impulse.shape[-1] >= waveform.shape[-1]:
            impulse = impulse[..., :waveform.shape[-1] // 4]
        
        # Convolve with impulse response
        try:
            convolved = torch.nn.functional.conv1d(
                waveform.unsqueeze(0),
                impulse.unsqueeze(0).unsqueeze(0),
                padding=impulse.shape[-1] // 2
            ).squeeze(0)
            
            # Trim to original length
            if convolved.shape[-1] > waveform.shape[-1]:
                convolved = convolved[..., :waveform.shape[-1]]
            
            # Mix with original (dry/wet mix)
            wet_level = random.uniform(0.1, 0.4)
            return waveform * (1 - wet_level) + convolved * wet_level
            
        except Exception:
            return waveform
    
    def _generate_reverb_impulses(self) -> List[torch.Tensor]:
        """Generate synthetic reverb impulse responses"""
        impulses = []
        
        # Generate different room impulses
        for room_size in [0.1, 0.3, 0.6, 1.0]:  # Small to large rooms
            length = int(16000 * room_size)  # 1 second max
            
            # Exponential decay
            decay = torch.exp(-torch.linspace(0, 5, length))
            
            # Add randomness
            noise = torch.randn(length) * 0.1
            
            impulse = decay * (1 + noise)
            impulses.append(impulse)
        
        return impulses
    
    def add_wind_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add wind noise for outdoor scenarios"""
        # Generate low-frequency wind noise
        length = waveform.shape[-1]
        
        # Create wind noise (low-pass filtered noise)
        wind = torch.randn(length)
        
        # Apply low-pass filter (approximate)
        wind_filtered = torch.zeros_like(wind)
        alpha = 0.01  # Filter coefficient
        
        for i in range(1, length):
            wind_filtered[i] = alpha * wind[i] + (1 - alpha) * wind_filtered[i-1]
        
        # Scale and mix
        wind_level = random.uniform(0.005, 0.02)
        wind_filtered = wind_filtered * wind_level
        
        if waveform.dim() == 2:
            wind_filtered = wind_filtered.unsqueeze(0)
        
        return waveform + wind_filtered
    
    def add_radio_static(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add radio static for communication interference"""
        # High-frequency crackling noise
        static_length = random.randint(100, 1000)
        static_start = random.randint(0, max(1, waveform.shape[-1] - static_length))
        
        # Generate static
        static = torch.randn(static_length) * 0.05
        
        # Apply to random section
        augmented = waveform.clone()
        if waveform.dim() == 1:
            augmented[static_start:static_start + static_length] += static
        else:
            augmented[0, static_start:static_start + static_length] += static
        
        return augmented


class FrequencyAugmentation:
    """
    Frequency-domain augmentations for spectrograms.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply random frequency-domain augmentations"""
        if random.random() > self.config.augmentation_prob:
            return spectrogram
        
        augmented = spectrogram
        
        # Frequency masking
        if self.config.freq_mask and random.random() < 0.5:
            augmented = self.frequency_mask(augmented)
        
        # Time masking
        if self.config.time_mask and random.random() < 0.5:
            augmented = self.time_mask(augmented)
        
        # Engine harmonics enhancement
        if self.config.engine_harmonics and random.random() < self.config.harmonics_prob:
            augmented = self.enhance_engine_harmonics(augmented)
        
        return augmented
    
    def frequency_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking (SpecAugment)"""
        freq_bins = spectrogram.shape[-2]
        
        # Random frequency mask
        mask_size = random.randint(1, min(self.config.freq_mask_param, freq_bins // 4))
        mask_start = random.randint(0, freq_bins - mask_size)
        
        masked = spectrogram.clone()
        masked[..., mask_start:mask_start + mask_size, :] = 0
        
        return masked
    
    def time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking (SpecAugment)"""
        time_frames = spectrogram.shape[-1]
        
        # Random time mask
        mask_size = random.randint(1, min(self.config.time_mask_param, time_frames // 4))
        mask_start = random.randint(0, time_frames - mask_size)
        
        masked = spectrogram.clone()
        masked[..., mask_start:mask_start + mask_size] = 0
        
        return masked
    
    def enhance_engine_harmonics(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Enhance engine harmonics for military vehicles"""
        # Identify potential harmonic frequencies
        enhanced = spectrogram.clone()
        
        # Enhance low-frequency harmonics (engine fundamentals)
        low_freq_bins = spectrogram.shape[-2] // 8  # Bottom 1/8 of frequencies
        
        # Apply gentle enhancement
        enhancement_factor = random.uniform(1.05, 1.15)
        enhanced[..., :low_freq_bins, :] *= enhancement_factor
        
        return enhanced
    
    def add_harmonic_distortion(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Add harmonic distortion patterns"""
        # Create harmonic patterns
        freq_bins, time_frames = spectrogram.shape[-2:]
        
        # Generate harmonic series
        fundamental_freq = random.randint(5, 20)  # Low frequency fundamental
        harmonics = []
        
        for harmonic in range(2, 6):  # 2nd to 5th harmonics
            harmonic_freq = fundamental_freq * harmonic
            if harmonic_freq < freq_bins:
                harmonics.append(harmonic_freq)
        
        # Add harmonic energy
        for harmonic_freq in harmonics:
            if harmonic_freq < freq_bins:
                harmonic_strength = random.uniform(0.1, 0.3)
                enhanced_spectrogram = spectrogram.clone()
                enhanced_spectrogram[..., harmonic_freq, :] += harmonic_strength
        
        return spectrogram


class MilitarySpecificAugmentation:
    """
    Military-specific augmentations for vehicle sound detection.
    """
    
    def __init__(self):
        # Predefined engine signatures
        self.helicopter_patterns = self._create_helicopter_patterns()
        self.jet_patterns = self._create_jet_patterns()
        self.vehicle_patterns = self._create_vehicle_patterns()
    
    def simulate_doppler_effect(self, waveform: torch.Tensor, approach_speed: float = 50.0) -> torch.Tensor:
        """Simulate Doppler effect for moving vehicles"""
        # Sound speed (m/s)
        sound_speed = 343.0
        
        # Calculate frequency shift
        doppler_factor = sound_speed / (sound_speed - approach_speed)
        
        # Apply pitch shift based on Doppler factor
        if doppler_factor != 1.0:
            semitones = 12 * math.log2(doppler_factor)
            
            # Apply gradual pitch change
            length = waveform.shape[-1]
            pitch_curve = torch.linspace(semitones, -semitones, length)
            
            # Apply time-varying pitch shift (simplified)
            return self._apply_variable_pitch_shift(waveform, pitch_curve)
        
        return waveform
    
    def _apply_variable_pitch_shift(self, waveform: torch.Tensor, pitch_curve: torch.Tensor) -> torch.Tensor:
        """Apply time-varying pitch shift"""
        # Simplified implementation - in practice, use more sophisticated methods
        chunk_size = 1024
        chunks = []
        
        for i in range(0, waveform.shape[-1], chunk_size):
            chunk = waveform[..., i:i + chunk_size]
            if chunk.shape[-1] > 0:
                # Average pitch shift for this chunk
                avg_pitch = pitch_curve[i:i + chunk_size].mean().item()
                
                # Apply pitch shift to chunk
                pitch_factor = 2 ** (avg_pitch / 12.0)
                if pitch_factor != 1.0:
                    chunk = self._simple_pitch_shift(chunk, pitch_factor)
                
                chunks.append(chunk)
        
        if chunks:
            return torch.cat(chunks, dim=-1)[..., :waveform.shape[-1]]
        
        return waveform
    
    def _simple_pitch_shift(self, chunk: torch.Tensor, factor: float) -> torch.Tensor:
        """Simple pitch shift implementation"""
        try:
            new_length = int(chunk.shape[-1] / factor)
            if new_length > 0:
                resampled = torch.nn.functional.interpolate(
                    chunk.unsqueeze(0) if chunk.dim() == 1 else chunk,
                    size=new_length,
                    mode='linear',
                    align_corners=False
                )
                
                # Pad or trim to original length
                original_length = chunk.shape[-1]
                if new_length > original_length:
                    return resampled[..., :original_length]
                else:
                    padding = original_length - new_length
                    return torch.nn.functional.pad(resampled, (0, padding))
        except Exception:
            pass
        
        return chunk
    
    def _create_helicopter_patterns(self) -> List[torch.Tensor]:
        """Create helicopter rotor patterns"""
        patterns = []
        
        # Main rotor patterns (different RPMs)
        for rpm in [200, 250, 300, 350]:
            freq = rpm / 60.0  # Hz
            duration = 2.0
            sample_rate = 16000
            
            t = torch.linspace(0, duration, int(sample_rate * duration))
            
            # Main rotor fundamental + harmonics
            pattern = torch.sin(2 * math.pi * freq * t)
            pattern += 0.3 * torch.sin(2 * math.pi * freq * 2 * t)  # 2nd harmonic
            pattern += 0.1 * torch.sin(2 * math.pi * freq * 3 * t)  # 3rd harmonic
            
            # Tail rotor (higher frequency)
            tail_freq = freq * 5.2
            pattern += 0.2 * torch.sin(2 * math.pi * tail_freq * t)
            
            patterns.append(pattern.unsqueeze(0))
        
        return patterns
    
    def _create_jet_patterns(self) -> List[torch.Tensor]:
        """Create jet engine patterns"""
        patterns = []
        
        # Jet engine broadband noise with specific characteristics
        for engine_type in ['turbofan', 'turbojet']:
            duration = 2.0
            sample_rate = 16000
            length = int(sample_rate * duration)
            
            # Broadband noise
            pattern = torch.randn(length)
            
            # Shape spectrum for jet characteristics
            if engine_type == 'turbofan':
                # Lower frequency emphasis
                pattern = self._apply_lowpass_filter(pattern, cutoff=0.3)
            else:
                # Higher frequency content
                pattern = self._apply_highpass_filter(pattern, cutoff=0.1)
            
            patterns.append(pattern.unsqueeze(0))
        
        return patterns
    
    def _create_vehicle_patterns(self) -> List[torch.Tensor]:
        """Create ground vehicle patterns"""
        patterns = []
        
        # Different engine types
        engine_configs = [
            {'fundamental': 30, 'harmonics': [2, 3, 4]},  # Diesel
            {'fundamental': 40, 'harmonics': [2, 4, 6]},  # Gasoline
            {'fundamental': 25, 'harmonics': [2, 3]},     # Heavy diesel
        ]
        
        for config in engine_configs:
            duration = 2.0
            sample_rate = 16000
            t = torch.linspace(0, duration, int(sample_rate * duration))
            
            # Engine fundamental
            pattern = torch.sin(2 * math.pi * config['fundamental'] * t)
            
            # Add harmonics
            for harmonic in config['harmonics']:
                freq = config['fundamental'] * harmonic
                amplitude = 1.0 / harmonic  # Decreasing amplitude
                pattern += amplitude * torch.sin(2 * math.pi * freq * t)
            
            # Add engine roughness
            roughness = torch.randn_like(t) * 0.1
            pattern += roughness
            
            patterns.append(pattern.unsqueeze(0))
        
        return patterns
    
    def _apply_lowpass_filter(self, signal: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Simple lowpass filter"""
        # Simplified implementation
        filtered = torch.zeros_like(signal)
        alpha = cutoff
        
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
        
        return filtered
    
    def _apply_highpass_filter(self, signal: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Simple highpass filter"""
        # Simplified implementation
        lowpass = self._apply_lowpass_filter(signal, cutoff)
        return signal - lowpass


class AudioMixup:
    """
    Mixup augmentation for audio samples.
    Mixes two audio samples with random weights.
    """
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
    
    def __call__(
        self,
        sample1: Dict[str, torch.Tensor],
        sample2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply mixup to two samples"""
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Mix spectrograms
        mixed_spectrogram = lam * sample1['spectrogram'] + (1 - lam) * sample2['spectrogram']
        
        # Mix labels (soft labels)
        label1_onehot = torch.zeros(7)  # Assuming 7 classes
        label2_onehot = torch.zeros(7)
        
        label1_onehot[sample1['label']] = 1
        label2_onehot[sample2['label']] = 1
        
        mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot
        
        return {
            'spectrogram': mixed_spectrogram,
            'label': mixed_label,  # Soft label
            'filename': f"mixup_{sample1['filename']}_{sample2['filename']}",
            'duration': torch.max(sample1['duration'], sample2['duration'])
        }


# Factory function for creating augmentation pipeline
def create_augmentation_pipeline(
    time_aug: bool = True,
    freq_aug: bool = True,
    military_aug: bool = True,
    mixup: bool = False,
    config: Optional[AugmentationConfig] = None
) -> Callable:
    """
    Create comprehensive augmentation pipeline.
    
    Args:
        time_aug: Enable time-domain augmentations
        freq_aug: Enable frequency-domain augmentations
        military_aug: Enable military-specific augmentations
        mixup: Enable mixup augmentation
        config: Augmentation configuration
        
    Returns:
        Augmentation function
    """
    config = config or AugmentationConfig()
    
    augmentations = []
    
    if time_aug:
        augmentations.append(TimeAugmentation(config))
    
    if freq_aug:
        freq_augmenter = FrequencyAugmentation(config)
        augmentations.append(lambda x: freq_augmenter(x) if x.dim() >= 2 else x)
    
    if military_aug:
        military_augmenter = MilitarySpecificAugmentation()
        augmentations.append(lambda x: military_augmenter.simulate_doppler_effect(x))
    
    def augmentation_pipeline(data):
        """Apply augmentation pipeline"""
        if isinstance(data, dict):
            # Handle dataset samples
            waveform = data.get('waveform')
            spectrogram = data.get('spectrogram')
            
            if waveform is not None:
                for aug in augmentations:
                    if hasattr(aug, '__self__') and isinstance(aug.__self__, TimeAugmentation):
                        waveform = aug(waveform)
                data['waveform'] = waveform
            
            if spectrogram is not None:
                for aug in augmentations:
                    if hasattr(aug, '__self__') and isinstance(aug.__self__, FrequencyAugmentation):
                        spectrogram = aug(spectrogram)
                data['spectrogram'] = spectrogram
            
            return data
        else:
            # Handle raw tensors
            augmented = data
            for aug in augmentations:
                augmented = aug(augmented)
            return augmented
    
    return augmentation_pipeline