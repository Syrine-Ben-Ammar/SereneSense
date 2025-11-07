#
# Plan:
# 1. Implement various audio segmentation strategies
# 2. Fixed-length segmentation with overlap for training
# 3. Silence-based segmentation for removing quiet periods
# 4. Energy-based segmentation for event detection
# 5. Onset-based segmentation for detecting vehicle events
# 6. Real-time segmentation for streaming audio
# 7. Adaptive segmentation based on content analysis
# 8. Memory-efficient processing for large audio files
#

import torch
import torch.nn as nn
import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class SegmentationConfig:
    """Configuration for audio segmentation."""
    
    # Fixed-length segmentation
    segment_length: float = 2.0  # seconds
    hop_length: float = 1.0      # seconds
    
    # Silence detection
    silence_threshold: float = -40.0  # dB
    min_silence_duration: float = 0.1  # seconds
    
    # Energy-based segmentation
    energy_threshold: float = 0.01
    energy_window: float = 0.1   # seconds
    
    # Onset detection
    onset_threshold: float = 0.3
    onset_window: float = 0.05   # seconds
    
    # General parameters
    sample_rate: int = 16000
    min_segment_length: float = 0.5  # seconds
    max_segment_length: float = 10.0  # seconds
    
    # Processing parameters
    use_gpu: bool = True
    padding_mode: str = 'constant'  # 'constant', 'reflect', 'replicate'


class AudioSegmenter(nn.Module):
    """Base class for audio segmenters."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        use_gpu: bool = True
    ):
        """
        Initialize audio segmenter.
        
        Args:
            sample_rate: Audio sample rate
            use_gpu: Use GPU acceleration when available
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu and torch.cuda.is_available()
    
    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Segment audio into multiple chunks.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            List of audio segments
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
    
    def _samples_to_time(self, samples: int) -> float:
        """Convert samples to time in seconds."""
        return samples / self.sample_rate
    
    def _time_to_samples(self, time: float) -> int:
        """Convert time in seconds to samples."""
        return int(time * self.sample_rate)


class FixedLengthSegmenter(AudioSegmenter):
    """Segment audio into fixed-length chunks with optional overlap."""
    
    def __init__(
        self,
        segment_length: float = 2.0,  # seconds
        hop_length: float = 1.0,      # seconds
        sample_rate: int = 16000,
        padding_mode: str = 'constant',
        use_gpu: bool = True
    ):
        """
        Initialize fixed-length segmenter.
        
        Args:
            segment_length: Length of each segment in seconds
            hop_length: Hop length between segments in seconds
            sample_rate: Audio sample rate
            padding_mode: Padding mode for incomplete segments
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(sample_rate, use_gpu)
        
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.padding_mode = padding_mode
        
        # Convert to samples
        self.segment_samples = self._time_to_samples(segment_length)
        self.hop_samples = self._time_to_samples(hop_length)
    
    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Segment audio into fixed-length chunks.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            List of audio segments
        """
        audio = self._validate_input(audio)
        
        segments = []
        num_channels, total_samples = audio.shape
        
        # Calculate number of segments
        if total_samples < self.segment_samples:
            # Pad if audio is shorter than segment length
            pad_amount = self.segment_samples - total_samples
            
            if self.padding_mode == 'constant':
                padded_audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='constant', value=0)
            elif self.padding_mode == 'reflect':
                padded_audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='reflect')
            elif self.padding_mode == 'replicate':
                padded_audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='replicate')
            else:
                padded_audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='constant', value=0)
            
            segments.append(padded_audio)
        else:
            # Extract overlapping segments
            start_sample = 0
            
            while start_sample + self.segment_samples <= total_samples:
                end_sample = start_sample + self.segment_samples
                segment = audio[:, start_sample:end_sample]
                segments.append(segment)
                
                start_sample += self.hop_samples
            
            # Handle remaining audio if any
            if start_sample < total_samples:
                remaining_samples = total_samples - start_sample
                
                if remaining_samples >= self.segment_samples // 2:  # At least half segment
                    # Pad the remaining audio
                    remaining_audio = audio[:, start_sample:]
                    pad_amount = self.segment_samples - remaining_samples
                    
                    if self.padding_mode == 'constant':
                        padded_remaining = torch.nn.functional.pad(remaining_audio, (0, pad_amount), mode='constant', value=0)
                    elif self.padding_mode == 'reflect':
                        padded_remaining = torch.nn.functional.pad(remaining_audio, (0, pad_amount), mode='reflect')
                    elif self.padding_mode == 'replicate':
                        padded_remaining = torch.nn.functional.pad(remaining_audio, (0, pad_amount), mode='replicate')
                    else:
                        padded_remaining = torch.nn.functional.pad(remaining_audio, (0, pad_amount), mode='constant', value=0)
                    
                    segments.append(padded_remaining)
        
        return segments


class OverlapSegmenter(FixedLengthSegmenter):
    """Segmenter with configurable overlap percentage."""
    
    def __init__(
        self,
        segment_length: float = 2.0,  # seconds
        overlap_ratio: float = 0.5,   # 50% overlap
        sample_rate: int = 16000,
        padding_mode: str = 'constant',
        use_gpu: bool = True
    ):
        """
        Initialize overlap segmenter.
        
        Args:
            segment_length: Length of each segment in seconds
            overlap_ratio: Overlap ratio (0.0 = no overlap, 0.5 = 50% overlap)
            sample_rate: Audio sample rate
            padding_mode: Padding mode for incomplete segments
            use_gpu: Use GPU acceleration when available
        """
        # Calculate hop length from overlap ratio
        hop_length = segment_length * (1.0 - overlap_ratio)
        
        super().__init__(
            segment_length=segment_length,
            hop_length=hop_length,
            sample_rate=sample_rate,
            padding_mode=padding_mode,
            use_gpu=use_gpu
        )
        
        self.overlap_ratio = overlap_ratio


class SilenceBasedSegmenter(AudioSegmenter):
    """Segment audio based on silence detection."""
    
    def __init__(
        self,
        silence_threshold: float = -40.0,  # dB
        min_silence_duration: float = 0.1,  # seconds
        min_segment_length: float = 0.5,    # seconds
        max_segment_length: float = 10.0,   # seconds
        sample_rate: int = 16000,
        use_gpu: bool = True
    ):
        """
        Initialize silence-based segmenter.
        
        Args:
            silence_threshold: Silence threshold in dB
            min_silence_duration: Minimum silence duration to trigger split
            min_segment_length: Minimum segment length
            max_segment_length: Maximum segment length
            sample_rate: Audio sample rate
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(sample_rate, use_gpu)
        
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        
        # Convert to samples
        self.min_silence_samples = self._time_to_samples(min_silence_duration)
        self.min_segment_samples = self._time_to_samples(min_segment_length)
        self.max_segment_samples = self._time_to_samples(max_segment_length)
        
        # Convert threshold to linear
        self.silence_threshold_linear = 10 ** (silence_threshold / 20.0)
    
    def _detect_silence(self, audio: torch.Tensor) -> torch.Tensor:
        """Detect silence regions in audio."""
        # Calculate RMS in small windows
        window_size = self._time_to_samples(0.01)  # 10ms windows
        
        if audio.shape[-1] < window_size:
            # Audio too short, check if it's silent
            rms = torch.sqrt(torch.mean(audio ** 2, dim=-1))
            return (rms < self.silence_threshold_linear).unsqueeze(-1).expand_as(audio)
        
        # Unfold audio into windows
        unfolded = audio.unfold(-1, window_size, window_size // 2)
        
        # Calculate RMS for each window
        rms = torch.sqrt(torch.mean(unfolded ** 2, dim=-1))
        
        # Detect silence
        is_silent = rms < self.silence_threshold_linear
        
        # Interpolate to original length
        silence_mask = torch.nn.functional.interpolate(
            is_silent.unsqueeze(0).float(),
            size=audio.shape[-1],
            mode='nearest'
        ).squeeze(0).bool()
        
        return silence_mask
    
    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Segment audio based on silence detection.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            List of audio segments
        """
        audio = self._validate_input(audio)
        
        # Detect silence
        silence_mask = self._detect_silence(audio)
        
        # Find silence regions
        silence_changes = torch.diff(silence_mask.float(), dim=-1)
        silence_starts = torch.where(silence_changes == 1)[1] + 1
        silence_ends = torch.where(silence_changes == -1)[1] + 1
        
        # Handle edge cases
        if silence_mask[0, 0]:  # Starts with silence
            silence_starts = torch.cat([torch.tensor([0], device=silence_starts.device), silence_starts])
        
        if silence_mask[0, -1]:  # Ends with silence
            silence_ends = torch.cat([silence_ends, torch.tensor([audio.shape[-1]], device=silence_ends.device)])
        
        # Ensure we have pairs
        min_len = min(len(silence_starts), len(silence_ends))
        silence_starts = silence_starts[:min_len]
        silence_ends = silence_ends[:min_len]
        
        # Find segments between silences
        segments = []
        last_end = 0
        
        for silence_start, silence_end in zip(silence_starts, silence_ends):
            # Check if silence is long enough
            silence_duration = silence_end - silence_start
            
            if silence_duration >= self.min_silence_samples:
                # Add segment before silence
                if silence_start > last_end:
                    segment_length = silence_start - last_end
                    
                    if segment_length >= self.min_segment_samples:
                        # Check if segment is too long
                        if segment_length > self.max_segment_samples:
                            # Split long segment
                            segment_audio = audio[:, last_end:silence_start]
                            subsegments = self._split_long_segment(segment_audio)
                            segments.extend(subsegments)
                        else:
                            segment = audio[:, last_end:silence_start]
                            segments.append(segment)
                
                last_end = silence_end
        
        # Add final segment
        if last_end < audio.shape[-1]:
            final_length = audio.shape[-1] - last_end
            
            if final_length >= self.min_segment_samples:
                if final_length > self.max_segment_samples:
                    # Split long segment
                    segment_audio = audio[:, last_end:]
                    subsegments = self._split_long_segment(segment_audio)
                    segments.extend(subsegments)
                else:
                    segment = audio[:, last_end:]
                    segments.append(segment)
        
        # If no segments found, return the whole audio (if long enough)
        if not segments and audio.shape[-1] >= self.min_segment_samples:
            if audio.shape[-1] > self.max_segment_samples:
                segments = self._split_long_segment(audio)
            else:
                segments = [audio]
        
        return segments
    
    def _split_long_segment(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """Split a long segment into smaller chunks."""
        segments = []
        total_samples = audio.shape[-1]
        
        start = 0
        while start < total_samples:
            end = min(start + self.max_segment_samples, total_samples)
            segment = audio[:, start:end]
            
            if segment.shape[-1] >= self.min_segment_samples:
                segments.append(segment)
            
            start += self.max_segment_samples
        
        return segments


class EnergyBasedSegmenter(AudioSegmenter):
    """Segment audio based on energy changes."""
    
    def __init__(
        self,
        energy_threshold: float = 0.01,
        energy_window: float = 0.1,    # seconds
        min_segment_length: float = 0.5,  # seconds
        max_segment_length: float = 10.0,  # seconds
        sample_rate: int = 16000,
        use_gpu: bool = True
    ):
        """
        Initialize energy-based segmenter.
        
        Args:
            energy_threshold: Energy threshold for segmentation
            energy_window: Window size for energy calculation
            min_segment_length: Minimum segment length
            max_segment_length: Maximum segment length
            sample_rate: Audio sample rate
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(sample_rate, use_gpu)
        
        self.energy_threshold = energy_threshold
        self.energy_window = energy_window
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        
        # Convert to samples
        self.energy_window_samples = self._time_to_samples(energy_window)
        self.min_segment_samples = self._time_to_samples(min_segment_length)
        self.max_segment_samples = self._time_to_samples(max_segment_length)
    
    def _calculate_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """Calculate energy in sliding windows."""
        if audio.shape[-1] < self.energy_window_samples:
            # Audio too short, return single energy value
            energy = torch.mean(audio ** 2, dim=-1, keepdim=True)
            return energy.expand(-1, audio.shape[-1])
        
        # Unfold audio into windows
        unfolded = audio.unfold(-1, self.energy_window_samples, self.energy_window_samples // 2)
        
        # Calculate energy for each window
        energy = torch.mean(unfolded ** 2, dim=-1)
        
        # Interpolate to original length
        energy_full = torch.nn.functional.interpolate(
            energy.unsqueeze(0),
            size=audio.shape[-1],
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        return energy_full
    
    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Segment audio based on energy changes.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            List of audio segments
        """
        audio = self._validate_input(audio)
        
        # Calculate energy
        energy = self._calculate_energy(audio)
        
        # Find energy peaks (potential segment boundaries)
        energy_diff = torch.diff(energy, dim=-1)
        
        # Find significant energy changes
        threshold = self.energy_threshold * torch.max(energy)
        significant_changes = torch.abs(energy_diff) > threshold
        
        # Find change points
        change_points = torch.where(significant_changes[0])[0] + 1
        
        # Add start and end points
        all_points = torch.cat([
            torch.tensor([0], device=change_points.device),
            change_points,
            torch.tensor([audio.shape[-1]], device=change_points.device)
        ])
        
        # Remove duplicates and sort
        all_points = torch.unique(all_points, sorted=True)
        
        # Create segments
        segments = []
        
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            
            segment_length = end - start
            
            # Check segment length constraints
            if segment_length >= self.min_segment_samples:
                if segment_length > self.max_segment_samples:
                    # Split long segment using fixed-length approach
                    segment_audio = audio[:, start:end]
                    fixed_segmenter = FixedLengthSegmenter(
                        segment_length=self.max_segment_length,
                        hop_length=self.max_segment_length,
                        sample_rate=self.sample_rate,
                        use_gpu=self.use_gpu
                    )
                    subsegments = fixed_segmenter(segment_audio)
                    segments.extend(subsegments)
                else:
                    segment = audio[:, start:end]
                    segments.append(segment)
        
        # If no valid segments found, fall back to fixed-length segmentation
        if not segments:
            fixed_segmenter = FixedLengthSegmenter(
                segment_length=min(self.max_segment_length, self._samples_to_time(audio.shape[-1])),
                hop_length=min(self.max_segment_length, self._samples_to_time(audio.shape[-1])),
                sample_rate=self.sample_rate,
                use_gpu=self.use_gpu
            )
            segments = fixed_segmenter(audio)
        
        return segments


class OnsetBasedSegmenter(AudioSegmenter):
    """Segment audio based on onset detection."""
    
    def __init__(
        self,
        onset_threshold: float = 0.3,
        onset_window: float = 0.05,   # seconds
        min_segment_length: float = 0.5,  # seconds
        max_segment_length: float = 10.0,  # seconds
        sample_rate: int = 16000,
        use_gpu: bool = True
    ):
        """
        Initialize onset-based segmenter.
        
        Args:
            onset_threshold: Onset detection threshold
            onset_window: Window size for onset detection
            min_segment_length: Minimum segment length
            max_segment_length: Maximum segment length
            sample_rate: Audio sample rate
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(sample_rate, use_gpu)
        
        self.onset_threshold = onset_threshold
        self.onset_window = onset_window
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        
        # Convert to samples
        self.onset_window_samples = self._time_to_samples(onset_window)
        self.min_segment_samples = self._time_to_samples(min_segment_length)
        self.max_segment_samples = self._time_to_samples(max_segment_length)
    
    def _detect_onsets(self, audio: torch.Tensor) -> torch.Tensor:
        """Detect onsets using spectral difference."""
        # Compute short-time Fourier transform
        stft = torch.stft(
            audio.squeeze(0) if audio.dim() > 1 else audio,
            n_fft=1024,
            hop_length=self.onset_window_samples,
            return_complex=True
        )
        
        # Calculate magnitude spectrogram
        magnitude = torch.abs(stft)
        
        # Calculate spectral difference
        spectral_diff = torch.diff(magnitude, dim=-1)
        
        # Only consider positive differences (increases in energy)
        spectral_diff = torch.clamp(spectral_diff, min=0)
        
        # Sum across frequency bins
        onset_strength = torch.sum(spectral_diff, dim=0)
        
        # Normalize
        if torch.max(onset_strength) > 0:
            onset_strength = onset_strength / torch.max(onset_strength)
        
        return onset_strength
    
    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Segment audio based on onset detection.
        
        Args:
            audio: Input audio tensor [channels, time] or [time]
            
        Returns:
            List of audio segments
        """
        audio = self._validate_input(audio)
        
        # Detect onsets
        onset_strength = self._detect_onsets(audio)
        
        # Find peaks in onset strength
        onset_peaks = onset_strength > self.onset_threshold
        
        # Convert back to sample indices
        hop_length = self.onset_window_samples
        onset_samples = torch.where(onset_peaks)[0] * hop_length
        
        # Add start and end points
        all_points = torch.cat([
            torch.tensor([0], device=onset_samples.device),
            onset_samples,
            torch.tensor([audio.shape[-1]], device=onset_samples.device)
        ])
        
        # Remove duplicates and sort
        all_points = torch.unique(all_points, sorted=True)
        
        # Create segments
        segments = []
        
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            
            segment_length = end - start
            
            # Check segment length constraints
            if segment_length >= self.min_segment_samples:
                if segment_length > self.max_segment_samples:
                    # Split long segment
                    segment_audio = audio[:, start:end]
                    fixed_segmenter = FixedLengthSegmenter(
                        segment_length=self.max_segment_length,
                        hop_length=self.max_segment_length,
                        sample_rate=self.sample_rate,
                        use_gpu=self.use_gpu
                    )
                    subsegments = fixed_segmenter(segment_audio)
                    segments.extend(subsegments)
                else:
                    segment = audio[:, start:end]
                    segments.append(segment)
        
        # If no valid segments found, fall back to fixed-length segmentation
        if not segments:
            fixed_segmenter = FixedLengthSegmenter(
                segment_length=min(self.max_segment_length, self._samples_to_time(audio.shape[-1])),
                hop_length=min(self.max_segment_length, self._samples_to_time(audio.shape[-1])),
                sample_rate=self.sample_rate,
                use_gpu=self.use_gpu
            )
            segments = fixed_segmenter(audio)
        
        return segments


class RealTimeSegmenter(AudioSegmenter):
    """Real-time segmenter for streaming audio processing."""
    
    def __init__(
        self,
        segment_length: float = 1.0,  # seconds
        sample_rate: int = 16000,
        use_gpu: bool = True
    ):
        """
        Initialize real-time segmenter.
        
        Args:
            segment_length: Length of each segment in seconds
            sample_rate: Audio sample rate
            use_gpu: Use GPU acceleration when available
        """
        super().__init__(sample_rate, use_gpu)
        
        self.segment_length = segment_length
        self.segment_samples = self._time_to_samples(segment_length)
        
        # Buffer for accumulating audio
        self.buffer = torch.zeros(0)
        if self.use_gpu:
            self.buffer = self.buffer.cuda()
    
    def reset_buffer(self):
        """Reset the internal buffer."""
        self.buffer = torch.zeros(0)
        if self.use_gpu:
            self.buffer = self.buffer.cuda()
    
    def process_chunk(self, audio_chunk: torch.Tensor) -> List[torch.Tensor]:
        """
        Process audio chunk and return complete segments.
        
        Args:
            audio_chunk: Input audio chunk
            
        Returns:
            List of complete segments (if any)
        """
        if audio_chunk.dim() > 1:
            audio_chunk = audio_chunk.squeeze()
        
        if self.use_gpu and not audio_chunk.is_cuda:
            audio_chunk = audio_chunk.cuda()
        
        # Add to buffer
        self.buffer = torch.cat([self.buffer, audio_chunk])
        
        segments = []
        
        # Extract complete segments
        while len(self.buffer) >= self.segment_samples:
            segment = self.buffer[:self.segment_samples]
            segments.append(segment.unsqueeze(0))  # Add channel dimension
            
            # Remove processed samples from buffer
            self.buffer = self.buffer[self.segment_samples:]
        
        return segments
    
    def flush(self) -> List[torch.Tensor]:
        """
        Flush remaining audio in buffer as final segment.
        
        Returns:
            List containing final segment (if any)
        """
        segments = []
        
        if len(self.buffer) > 0:
            # Pad buffer to segment length if needed
            if len(self.buffer) < self.segment_samples:
                pad_amount = self.segment_samples - len(self.buffer)
                self.buffer = torch.nn.functional.pad(self.buffer, (0, pad_amount))
            
            segments.append(self.buffer.unsqueeze(0))
            self.reset_buffer()
        
        return segments
    
    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Process audio (for compatibility with other segmenters).
        
        Args:
            audio: Input audio tensor
            
        Returns:
            List of segments
        """
        # Reset buffer and process all at once
        self.reset_buffer()
        segments = self.process_chunk(audio.squeeze() if audio.dim() > 1 else audio)
        final_segments = self.flush()
        segments.extend(final_segments)
        return segments


def create_segmenter(
    segmenter_type: str = 'fixed',
    config: Optional[SegmentationConfig] = None,
    **kwargs
) -> AudioSegmenter:
    """
    Factory function to create audio segmenters.
    
    Args:
        segmenter_type: Type of segmenter ('fixed', 'overlap', 'silence', 'energy', 'onset', 'realtime')
        config: Segmentation configuration
        **kwargs: Override configuration parameters
        
    Returns:
        Audio segmenter instance
    """
    if config is None:
        config = SegmentationConfig()
    
    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create appropriate segmenter
    if segmenter_type == 'fixed':
        return FixedLengthSegmenter(
            segment_length=config.segment_length,
            hop_length=config.hop_length,
            sample_rate=config.sample_rate,
            padding_mode=config.padding_mode,
            use_gpu=config.use_gpu
        )
    elif segmenter_type == 'overlap':
        overlap_ratio = 1.0 - (config.hop_length / config.segment_length)
        return OverlapSegmenter(
            segment_length=config.segment_length,
            overlap_ratio=overlap_ratio,
            sample_rate=config.sample_rate,
            padding_mode=config.padding_mode,
            use_gpu=config.use_gpu
        )
    elif segmenter_type == 'silence':
        return SilenceBasedSegmenter(
            silence_threshold=config.silence_threshold,
            min_silence_duration=config.min_silence_duration,
            min_segment_length=config.min_segment_length,
            max_segment_length=config.max_segment_length,
            sample_rate=config.sample_rate,
            use_gpu=config.use_gpu
        )
    elif segmenter_type == 'energy':
        return EnergyBasedSegmenter(
            energy_threshold=config.energy_threshold,
            energy_window=config.energy_window,
            min_segment_length=config.min_segment_length,
            max_segment_length=config.max_segment_length,
            sample_rate=config.sample_rate,
            use_gpu=config.use_gpu
        )
    elif segmenter_type == 'onset':
        return OnsetBasedSegmenter(
            onset_threshold=config.onset_threshold,
            onset_window=config.onset_window,
            min_segment_length=config.min_segment_length,
            max_segment_length=config.max_segment_length,
            sample_rate=config.sample_rate,
            use_gpu=config.use_gpu
        )
    elif segmenter_type == 'realtime':
        return RealTimeSegmenter(
            segment_length=config.segment_length,
            sample_rate=config.sample_rate,
            use_gpu=config.use_gpu
        )
    else:
        raise ValueError(f"Unknown segmenter type: {segmenter_type}")


def get_segmentation_info():
    """
    Get information about available segmentation types.
    
    Returns:
        Dictionary with segmentation information
    """
    return {
        'available_types': [
            'fixed', 'overlap', 'silence', 'energy', 'onset', 'realtime'
        ],
        'description': {
            'fixed': 'Fixed-length segments with configurable hop',
            'overlap': 'Fixed-length segments with percentage overlap',
            'silence': 'Segments based on silence detection',
            'energy': 'Segments based on energy changes',
            'onset': 'Segments based on onset detection',
            'realtime': 'Real-time segmentation for streaming'
        },
        'recommended_for_military': 'energy'
    }