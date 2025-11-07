"""
Audio Processor for SereneSense

This module provides comprehensive audio processing capabilities for the SereneSense
military vehicle sound detection system.

Features:
- Audio loading from multiple formats (WAV, MP3, FLAC, etc.)
- Resampling and format conversion
- Audio normalization and preprocessing
- Silence detection and removal
- Audio segmentation and windowing
- Real-time audio streaming support
- Robust error handling and validation

Example:
    >>> from core.core.audio_processor import AudioProcessor
    >>> 
    >>> # Initialize processor
    >>> config = {"sample_rate": 16000, "duration": 10.0}
    >>> processor = AudioProcessor(config)
    >>> 
    >>> # Load and process audio
    >>> audio_data = processor.load_audio("audio.wav")
    >>> processed = processor.preprocess(audio_data)
"""

import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import warnings
from dataclasses import dataclass

# Third-party imports
try:
    import librosa
    import soundfile as sf

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa/soundfile not available", ImportWarning)

try:
    import scipy.signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type definitions
AudioData = Union[np.ndarray, torch.Tensor]
PathLike = Union[str, Path]


@dataclass
class AudioMetadata:
    """Container for audio metadata."""

    file_path: Optional[str] = None
    sample_rate: int = 16000
    duration: float = 0.0
    channels: int = 1
    bit_depth: int = 16
    format: str = "unknown"
    codec: str = "unknown"
    rms_level: float = 0.0
    peak_level: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_centroid: float = 0.0


class AudioProcessor:
    """
    Comprehensive audio processing pipeline for SereneSense.

    This class handles all audio preprocessing tasks including loading,
    resampling, normalization, and format conversion.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize audio processor.

        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config

        # Audio parameters
        self.target_sample_rate = config.get("sample_rate", 16000)
        self.target_duration = config.get("duration", 10.0)
        self.target_channels = config.get("channels", 1)

        # Processing parameters
        self.normalize = config.get("normalize", True)
        self.remove_silence = config.get("remove_silence", False)
        self.preemphasis_coeff = config.get("preemphasis", 0.97)

        # Quality control
        self.min_duration = config.get("min_duration", 0.1)
        self.max_duration = config.get("max_duration", 60.0)
        self.clip_threshold = config.get("clip_threshold", 0.99)

        # Supported formats
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

        logger.info(
            f"AudioProcessor initialized - SR: {self.target_sample_rate}Hz, "
            f"Duration: {self.target_duration}s"
        )

    def load_audio(
        self, file_path: PathLike, offset: float = 0.0, duration: Optional[float] = None
    ) -> Tuple[AudioData, AudioMetadata]:
        """
        Load audio file with robust format support.

        Args:
            file_path: Path to audio file
            offset: Start time offset in seconds
            duration: Duration to load in seconds (None for full file)

        Returns:
            Tuple of (audio_data, metadata)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported or corrupted
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}")

        try:
            # Try torchaudio first (fastest)
            audio_data, metadata = self._load_with_torchaudio(file_path, offset, duration)

        except Exception as e:
            logger.debug(f"Torchaudio failed, trying librosa: {e}")

            if not LIBROSA_AVAILABLE:
                raise RuntimeError("Both torchaudio and librosa failed, no fallback available")

            try:
                # Fallback to librosa
                audio_data, metadata = self._load_with_librosa(file_path, offset, duration)

            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load audio with both backends: " f"torchaudio: {e}, librosa: {e2}"
                )

        # Validate loaded audio
        self._validate_audio(audio_data, metadata)

        logger.debug(
            f"Loaded audio: {metadata.duration:.2f}s, "
            f"{metadata.sample_rate}Hz, {metadata.channels}ch"
        )

        return audio_data, metadata

    def _load_with_torchaudio(
        self, file_path: Path, offset: float = 0.0, duration: Optional[float] = None
    ) -> Tuple[torch.Tensor, AudioMetadata]:
        """
        Load audio using torchaudio backend.

        Args:
            file_path: Path to audio file
            offset: Start time offset in seconds
            duration: Duration to load in seconds

        Returns:
            Tuple of (audio_tensor, metadata)
        """
        # Get file info
        info = torchaudio.info(str(file_path))

        # Calculate frame parameters
        frame_offset = int(offset * info.sample_rate)
        num_frames = int(duration * info.sample_rate) if duration else -1

        # Load audio
        waveform, sample_rate = torchaudio.load(
            str(file_path), frame_offset=frame_offset, num_frames=num_frames
        )

        # Create metadata
        metadata = AudioMetadata(
            file_path=str(file_path),
            sample_rate=sample_rate,
            duration=waveform.shape[1] / sample_rate,
            channels=waveform.shape[0],
            format=file_path.suffix.lower(),
            codec=getattr(info, "codec", "unknown"),
        )

        # Convert to mono if needed
        if waveform.shape[0] > 1 and self.target_channels == 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            metadata.channels = 1

        return waveform, metadata

    def _load_with_librosa(
        self, file_path: Path, offset: float = 0.0, duration: Optional[float] = None
    ) -> Tuple[np.ndarray, AudioMetadata]:
        """
        Load audio using librosa backend.

        Args:
            file_path: Path to audio file
            offset: Start time offset in seconds
            duration: Duration to load in seconds

        Returns:
            Tuple of (audio_array, metadata)
        """
        # Load audio
        waveform, sample_rate = librosa.load(
            str(file_path),
            sr=None,  # Keep original sample rate
            mono=(self.target_channels == 1),
            offset=offset,
            duration=duration,
        )

        # Ensure waveform is 2D (channels, time)
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)

        # Create metadata
        metadata = AudioMetadata(
            file_path=str(file_path),
            sample_rate=sample_rate,
            duration=waveform.shape[1] / sample_rate,
            channels=waveform.shape[0],
            format=file_path.suffix.lower(),
        )

        return waveform, metadata

    def _validate_audio(self, audio_data: AudioData, metadata: AudioMetadata) -> None:
        """
        Validate loaded audio data.

        Args:
            audio_data: Audio data array/tensor
            metadata: Audio metadata

        Raises:
            ValueError: If audio is invalid
        """
        # Check duration
        if metadata.duration < self.min_duration:
            raise ValueError(f"Audio too short: {metadata.duration:.2f}s < {self.min_duration}s")

        if metadata.duration > self.max_duration:
            logger.warning(f"Audio very long: {metadata.duration:.2f}s > {self.max_duration}s")

        # Check for clipping
        if isinstance(audio_data, torch.Tensor):
            max_val = torch.max(torch.abs(audio_data)).item()
        else:
            max_val = np.max(np.abs(audio_data))

        if max_val > self.clip_threshold:
            logger.warning(f"Potential clipping detected: peak = {max_val:.3f}")

        # Check for silence
        if isinstance(audio_data, torch.Tensor):
            rms = torch.sqrt(torch.mean(audio_data**2)).item()
        else:
            rms = np.sqrt(np.mean(audio_data**2))

        if rms < 1e-6:
            logger.warning("Audio appears to be silent or very quiet")

    def preprocess(
        self, audio_data: AudioData, metadata: Optional[AudioMetadata] = None
    ) -> AudioData:
        """
        Apply preprocessing pipeline to audio data.

        Args:
            audio_data: Input audio data
            metadata: Optional metadata for context

        Returns:
            Preprocessed audio data
        """
        # Convert to tensor if needed
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data.float()

        # Ensure 2D tensor (channels, time)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Resample if needed
        if metadata and metadata.sample_rate != self.target_sample_rate:
            audio_tensor = self._resample(
                audio_tensor, metadata.sample_rate, self.target_sample_rate
            )

        # Convert to mono if needed
        if audio_tensor.shape[0] > 1 and self.target_channels == 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        # Remove silence if requested
        if self.remove_silence:
            audio_tensor = self._remove_silence(audio_tensor)

        # Apply pre-emphasis
        if self.preemphasis_coeff > 0:
            audio_tensor = self._apply_preemphasis(audio_tensor, self.preemphasis_coeff)

        # Normalize audio
        if self.normalize:
            audio_tensor = self._normalize_audio(audio_tensor)

        # Adjust duration
        audio_tensor = self._adjust_duration(
            audio_tensor, self.target_duration, self.target_sample_rate
        )

        return audio_tensor

    def _resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio tensor
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio tensor
        """
        if orig_sr == target_sr:
            return audio

        # Use torchaudio resampling
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr, dtype=audio.dtype
        )

        return resampler(audio)

    def _remove_silence(
        self,
        audio: torch.Tensor,
        threshold: float = 0.01,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> torch.Tensor:
        """
        Remove silent regions from audio.

        Args:
            audio: Input audio tensor
            threshold: RMS threshold for silence detection
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis

        Returns:
            Audio with silence removed
        """
        # Calculate RMS energy
        rms = torch.sqrt(
            torch.nn.functional.conv1d(
                audio.unsqueeze(0) ** 2,
                torch.ones(1, 1, frame_length) / frame_length,
                stride=hop_length,
                padding=frame_length // 2,
            ).squeeze(0)
        )

        # Find non-silent frames
        non_silent = rms > threshold

        if not torch.any(non_silent):
            logger.warning("All audio detected as silence, keeping original")
            return audio

        # Expand to sample level
        non_silent_samples = torch.repeat_interleave(non_silent, hop_length, dim=1)

        # Ensure same length
        if non_silent_samples.shape[1] > audio.shape[1]:
            non_silent_samples = non_silent_samples[:, : audio.shape[1]]
        elif non_silent_samples.shape[1] < audio.shape[1]:
            # Pad with last value
            pad_size = audio.shape[1] - non_silent_samples.shape[1]
            padding = non_silent_samples[:, -1:].repeat(1, pad_size)
            non_silent_samples = torch.cat([non_silent_samples, padding], dim=1)

        return audio * non_silent_samples.float()

    def _apply_preemphasis(self, audio: torch.Tensor, coeff: float) -> torch.Tensor:
        """
        Apply pre-emphasis filter to audio.

        Args:
            audio: Input audio tensor
            coeff: Pre-emphasis coefficient

        Returns:
            Pre-emphasized audio
        """
        # Apply pre-emphasis: y[n] = x[n] - coeff * x[n-1]
        return audio - torch.nn.functional.pad(audio * coeff, (1, 0))[:, :-1]

    def _normalize_audio(
        self, audio: torch.Tensor, method: str = "peak", target_level: float = 0.9
    ) -> torch.Tensor:
        """
        Normalize audio to target level.

        Args:
            audio: Input audio tensor
            method: Normalization method ('peak', 'rms', 'lufs')
            target_level: Target normalization level

        Returns:
            Normalized audio
        """
        if method == "peak":
            # Peak normalization
            peak = torch.max(torch.abs(audio))
            if peak > 0:
                audio = audio * (target_level / peak)

        elif method == "rms":
            # RMS normalization
            rms = torch.sqrt(torch.mean(audio**2))
            if rms > 0:
                audio = audio * (target_level / rms)

        else:
            logger.warning(f"Unknown normalization method: {method}, using peak")
            peak = torch.max(torch.abs(audio))
            if peak > 0:
                audio = audio * (target_level / peak)

        return audio

    def _adjust_duration(
        self, audio: torch.Tensor, target_duration: float, sample_rate: int
    ) -> torch.Tensor:
        """
        Adjust audio duration by padding or cropping.

        Args:
            audio: Input audio tensor
            target_duration: Target duration in seconds
            sample_rate: Sample rate

        Returns:
            Duration-adjusted audio
        """
        target_samples = int(target_duration * sample_rate)
        current_samples = audio.shape[1]

        if current_samples < target_samples:
            # Pad with zeros
            pad_samples = target_samples - current_samples
            audio = torch.nn.functional.pad(audio, (0, pad_samples))

        elif current_samples > target_samples:
            # Crop (random crop for training, center crop for inference)
            if hasattr(self, "training") and self.training:
                # Random crop
                start_idx = torch.randint(0, current_samples - target_samples + 1, (1,)).item()
            else:
                # Center crop
                start_idx = (current_samples - target_samples) // 2

            audio = audio[:, start_idx : start_idx + target_samples]

        return audio

    def extract_metadata(self, audio: AudioData, sample_rate: int) -> AudioMetadata:
        """
        Extract comprehensive metadata from audio.

        Args:
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            AudioMetadata object
        """
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
        else:
            audio_np = audio

        # Ensure 1D for analysis
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        metadata = AudioMetadata(
            sample_rate=sample_rate,
            duration=len(audio_np) / sample_rate,
            channels=1 if audio_np.ndim == 1 else audio.shape[0],
            rms_level=float(np.sqrt(np.mean(audio_np**2))),
            peak_level=float(np.max(np.abs(audio_np))),
            zero_crossing_rate=float(np.mean(np.diff(np.sign(audio_np)) != 0)),
        )

        # Spectral centroid (requires librosa)
        if LIBROSA_AVAILABLE:
            try:
                spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate)[0]
                metadata.spectral_centroid = float(np.mean(spectral_centroid))
            except Exception as e:
                logger.debug(f"Failed to compute spectral centroid: {e}")

        return metadata

    def process_batch(
        self, audio_files: List[PathLike], batch_size: int = 32
    ) -> Tuple[torch.Tensor, List[AudioMetadata]]:
        """
        Process multiple audio files in batches.

        Args:
            audio_files: List of audio file paths
            batch_size: Batch size for processing

        Returns:
            Tuple of (batched_audio_tensor, metadata_list)
        """
        processed_audio = []
        metadata_list = []

        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i : i + batch_size]
            batch_audio = []
            batch_metadata = []

            for file_path in batch_files:
                try:
                    audio_data, metadata = self.load_audio(file_path)
                    processed = self.preprocess(audio_data, metadata)

                    batch_audio.append(processed)
                    batch_metadata.append(metadata)

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue

            if batch_audio:
                # Stack batch
                batch_tensor = torch.stack(batch_audio)
                processed_audio.append(batch_tensor)
                metadata_list.extend(batch_metadata)

        if not processed_audio:
            raise RuntimeError("No audio files were successfully processed")

        # Concatenate all batches
        all_audio = torch.cat(processed_audio, dim=0)

        return all_audio, metadata_list

    def save_audio(
        self,
        audio: AudioData,
        file_path: PathLike,
        sample_rate: Optional[int] = None,
        format: str = "wav",
    ) -> None:
        """
        Save audio data to file.

        Args:
            audio: Audio data to save
            file_path: Output file path
            sample_rate: Sample rate (uses target_sample_rate if None)
            format: Output format
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        if isinstance(audio, torch.Tensor):
            audio_tensor = audio
        else:
            audio_tensor = torch.from_numpy(audio).float()

        # Ensure 2D
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Save using torchaudio
        torchaudio.save(str(file_path), audio_tensor, sample_rate)

        logger.info(f"Saved audio to {file_path}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration.

        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)

        # Update derived parameters
        self.target_sample_rate = self.config.get("sample_rate", self.target_sample_rate)
        self.target_duration = self.config.get("duration", self.target_duration)
        self.normalize = self.config.get("normalize", self.normalize)

        logger.info("Audio processor configuration updated")


# Convenience functions
def load_audio_simple(
    file_path: PathLike, sample_rate: int = 16000, duration: float = 10.0
) -> Tuple[torch.Tensor, AudioMetadata]:
    """
    Simple audio loading function.

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        duration: Target duration

    Returns:
        Tuple of (audio_tensor, metadata)
    """
    config = {"sample_rate": sample_rate, "duration": duration, "normalize": True}

    processor = AudioProcessor(config)
    audio_data, metadata = processor.load_audio(file_path)
    processed_audio = processor.preprocess(audio_data, metadata)

    return processed_audio, metadata


def validate_audio_file(file_path: PathLike) -> Dict[str, Any]:
    """
    Validate audio file and return information.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary containing validation results
    """
    try:
        config = {"sample_rate": 16000, "duration": None}
        processor = AudioProcessor(config)

        audio_data, metadata = processor.load_audio(file_path)

        return {"valid": True, "metadata": metadata, "error": None}

    except Exception as e:
        return {"valid": False, "metadata": None, "error": str(e)}
