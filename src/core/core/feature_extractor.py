"""
SereneSense Core Audio Processor
Enterprise-grade audio preprocessing for military vehicle detection.
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
from pathlib import Path
import logging
from dataclasses import dataclass
import librosa

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio processing configuration"""

    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    window_length: float = 2.0  # seconds
    overlap: float = 0.5  # overlap ratio
    normalize: bool = True
    apply_log: bool = True
    mel_scale: str = "htk"  # or "slaney"


class AudioProcessor:
    """
    Enterprise-grade audio processor for military vehicle sound detection.

    Features:
    - Multi-format audio loading (WAV, MP3, FLAC, etc.)
    - High-quality mel-spectrogram generation
    - Robust normalization and preprocessing
    - Batch processing capabilities
    - Memory-efficient streaming
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.mel_transform = None
        self._setup_transforms()

    def _setup_transforms(self):
        """Initialize audio transforms"""
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            mel_scale=self.config.mel_scale,
            norm="slaney",
            normalized=True,
        )

        if self.config.apply_log:
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def load_audio(
        self, file_path: Union[str, Path], offset: float = 0.0, duration: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file with robust error handling.

        Args:
            file_path: Path to audio file
            offset: Start time in seconds
            duration: Duration to load in seconds

        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            waveform, sr = torchaudio.load(
                file_path,
                frame_offset=int(offset * self.config.sample_rate),
                num_frames=int(duration * self.config.sample_rate) if duration else -1,
            )

            # Resample if necessary
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            return waveform, self.config.sample_rate

        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise

    def compute_mel_spectrogram(
        self, waveform: torch.Tensor, normalize: bool = None
    ) -> torch.Tensor:
        """
        Compute mel-spectrogram from waveform.

        Args:
            waveform: Input waveform tensor [channels, time]
            normalize: Whether to normalize (overrides config)

        Returns:
            Mel-spectrogram tensor [channels, mel_bins, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)

        # Apply log scaling
        if self.config.apply_log:
            mel_spec = self.amplitude_to_db(mel_spec)

        # Normalize
        normalize = normalize if normalize is not None else self.config.normalize
        if normalize:
            mel_spec = self._normalize_spectrogram(mel_spec)

        return mel_spec

    def _normalize_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram to [-1, 1] range"""
        # Per-channel normalization
        for i in range(spec.shape[0]):
            channel = spec[i]
            mean = channel.mean()
            std = channel.std()
            if std > 1e-8:
                spec[i] = (channel - mean) / std
            else:
                spec[i] = channel - mean

        return spec

    def segment_audio(
        self,
        waveform: torch.Tensor,
        window_length: Optional[float] = None,
        overlap: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """
        Segment audio into overlapping windows.

        Args:
            waveform: Input waveform [channels, time]
            window_length: Window length in seconds (overrides config)
            overlap: Overlap ratio (overrides config)

        Returns:
            List of audio segments
        """
        window_length = window_length or self.config.window_length
        overlap = overlap or self.config.overlap

        window_samples = int(window_length * self.config.sample_rate)
        hop_samples = int(window_samples * (1 - overlap))

        segments = []
        start = 0

        while start + window_samples <= waveform.shape[-1]:
            segment = waveform[..., start : start + window_samples]
            segments.append(segment)
            start += hop_samples

        return segments

    def process_audio_file(
        self, file_path: Union[str, Path], return_segments: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Complete audio processing pipeline for a single file.

        Args:
            file_path: Path to audio file
            return_segments: Whether to return individual segments

        Returns:
            Mel-spectrogram(s) ready for model inference
        """
        # Load audio
        waveform, _ = self.load_audio(file_path)

        if return_segments:
            # Segment audio
            segments = self.segment_audio(waveform)

            # Process each segment
            mel_specs = []
            for segment in segments:
                mel_spec = self.compute_mel_spectrogram(segment)
                mel_specs.append(mel_spec)

            return torch.stack(mel_specs), segments
        else:
            # Process entire audio
            mel_spec = self.compute_mel_spectrogram(waveform)
            return mel_spec

    def batch_process(
        self, file_paths: List[Union[str, Path]], max_batch_size: int = 32
    ) -> torch.Tensor:
        """
        Process multiple audio files in batches.

        Args:
            file_paths: List of audio file paths
            max_batch_size: Maximum batch size for processing

        Returns:
            Batched mel-spectrograms [batch, channels, mel_bins, time_frames]
        """
        all_specs = []

        for i in range(0, len(file_paths), max_batch_size):
            batch_paths = file_paths[i : i + max_batch_size]
            batch_specs = []

            for path in batch_paths:
                try:
                    mel_spec = self.process_audio_file(path)
                    batch_specs.append(mel_spec)
                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    continue

            if batch_specs:
                # Pad to same length if necessary
                batch_specs = self._pad_to_same_length(batch_specs)
                all_specs.extend(batch_specs)

        return torch.stack(all_specs) if all_specs else torch.empty(0)

    def _pad_to_same_length(self, spectrograms: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pad spectrograms to the same temporal length"""
        if not spectrograms:
            return spectrograms

        max_length = max(spec.shape[-1] for spec in spectrograms)

        padded_specs = []
        for spec in spectrograms:
            if spec.shape[-1] < max_length:
                pad_length = max_length - spec.shape[-1]
                spec = torch.nn.functional.pad(spec, (0, pad_length))
            padded_specs.append(spec)

        return padded_specs

    def extract_features(self, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract additional audio features for military vehicle detection.

        Args:
            mel_spec: Mel-spectrogram tensor

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Spectral centroid
        features["spectral_centroid"] = torch.mean(mel_spec, dim=-1)

        # Spectral rolloff
        features["spectral_rolloff"] = torch.quantile(mel_spec, 0.85, dim=-2)

        # Zero crossing rate (approximate from mel-spec)
        features["zcr"] = torch.std(mel_spec, dim=-1)

        # Spectral contrast
        features["spectral_contrast"] = (
            torch.max(mel_spec, dim=-2)[0] - torch.min(mel_spec, dim=-2)[0]
        )

        # MFCC-like features (using mel-spec)
        features["mfcc"] = torch.fft.fft(mel_spec, dim=-2).real[:, :13]  # First 13 coefficients

        return features

    def validate_audio(self, waveform: torch.Tensor) -> bool:
        """
        Validate audio quality for military vehicle detection.

        Args:
            waveform: Input waveform

        Returns:
            True if audio meets quality requirements
        """
        # Check for silence
        if torch.max(torch.abs(waveform)) < 1e-6:
            logger.warning("Audio is too quiet (possible silence)")
            return False

        # Check for clipping
        if torch.max(torch.abs(waveform)) > 0.99:
            logger.warning("Audio may be clipped")

        # Check duration
        duration = waveform.shape[-1] / self.config.sample_rate
        if duration < 0.1:
            logger.warning("Audio too short for reliable detection")
            return False

        return True

    def preprocess_for_model(
        self, mel_spec: torch.Tensor, target_shape: Tuple[int, int] = (128, 128)
    ) -> torch.Tensor:
        """
        Preprocess mel-spectrogram for specific model requirements.

        Args:
            mel_spec: Input mel-spectrogram
            target_shape: Target (height, width) for model input

        Returns:
            Preprocessed spectrogram ready for model
        """
        # Ensure 4D tensor [batch, channels, height, width]
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)

        # Resize to target shape
        mel_spec = torch.nn.functional.interpolate(
            mel_spec, size=target_shape, mode="bilinear", align_corners=False
        )

        return mel_spec


class RealTimeAudioProcessor(AudioProcessor):
    """
    Real-time audio processor for streaming military vehicle detection.
    Optimized for low-latency edge deployment.
    """

    def __init__(self, config: AudioConfig, buffer_size: int = 4096):
        super().__init__(config)
        self.buffer_size = buffer_size
        self.audio_buffer = torch.zeros(1, buffer_size)
        self.buffer_ptr = 0

    def add_audio_chunk(self, chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Add audio chunk to circular buffer and return mel-spec if ready.

        Args:
            chunk: Audio chunk to add

        Returns:
            Mel-spectrogram if enough audio accumulated, None otherwise
        """
        chunk_size = chunk.shape[-1]

        # Add to buffer
        if self.buffer_ptr + chunk_size <= self.buffer_size:
            self.audio_buffer[..., self.buffer_ptr : self.buffer_ptr + chunk_size] = chunk
            self.buffer_ptr += chunk_size
        else:
            # Buffer overflow - shift and add
            overflow = self.buffer_ptr + chunk_size - self.buffer_size
            self.audio_buffer = torch.roll(self.audio_buffer, -overflow, dims=-1)
            self.audio_buffer[..., -chunk_size:] = chunk
            self.buffer_ptr = self.buffer_size

        # Check if we have enough audio for processing
        window_samples = int(self.config.window_length * self.config.sample_rate)
        if self.buffer_ptr >= window_samples:
            # Extract window and compute mel-spectrogram
            window = self.audio_buffer[..., :window_samples]
            mel_spec = self.compute_mel_spectrogram(window)
            return mel_spec

        return None

    def reset_buffer(self):
        """Reset the audio buffer"""
        self.audio_buffer.fill_(0)
        self.buffer_ptr = 0


def create_audio_processor(config_path: Optional[str] = None) -> AudioProcessor:
    """
    Factory function to create AudioProcessor with configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured AudioProcessor instance
    """
    if config_path:
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = AudioConfig(**config_dict.get("audio", {}))
    else:
        config = AudioConfig()

    return AudioProcessor(config)
