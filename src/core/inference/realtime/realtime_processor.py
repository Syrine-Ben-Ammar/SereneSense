#
# Plan:
# 1. Create RealTimeInferenceEngine for low-latency military vehicle detection
# 2. Implement CircularAudioBuffer for continuous audio streaming
# 3. Add StreamingProcessor for real-time audio preprocessing
# 4. Create LatencyTracker for performance monitoring
# 5. Implement AudioStreamConfig for flexible configuration
# 6. Add memory optimization and hardware acceleration support
# 7. Support for multiple audio input sources (microphone, file, network)
# 8. Implement confidence thresholding and alert generation
#

"""
Real-Time Inference Engine for SereneSense

This module provides high-performance real-time inference capabilities
for military vehicle sound detection with sub-20ms latency requirements.

Key Features:
- Real-time audio streaming and processing
- Circular buffer management for continuous operation
- Hardware acceleration (CUDA, TensorRT, OpenVINO)
- Sub-20ms latency for edge deployment
- Confidence-based alerting system
- Memory optimization for resource-constrained devices
- Multi-threaded processing pipeline
"""

import torch
import torch.nn.functional as F
import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque
import warnings

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from ..core.audio_processor import AudioProcessor
from ..core.feature_extractor import FeatureExtractor
from ..core.model_manager import ModelManager
try:
    from ..utils.device_utils import DeviceManager
except ImportError:
    DeviceManager = None

logger = logging.getLogger(__name__)


@dataclass
class AudioStreamConfig:
    """Configuration for real-time audio streaming"""

    # Audio parameters
    sample_rate: int = 16000  # Sample rate in Hz
    channels: int = 1  # Number of audio channels (mono)
    chunk_size: int = 1024  # Audio chunk size in samples
    buffer_duration_ms: float = 1000.0  # Buffer duration in milliseconds

    # Processing parameters
    hop_length: int = 512  # Hop length for spectrogram
    n_fft: int = 1024  # FFT size
    n_mels: int = 128  # Number of mel bins
    target_length: int = 128  # Target sequence length

    # Real-time parameters
    max_latency_ms: float = 20.0  # Maximum allowed latency
    confidence_threshold: float = 0.8  # Confidence threshold for alerts
    overlap_ratio: float = 0.5  # Overlap ratio between chunks

    # Hardware parameters
    device: str = "auto"  # Device for inference ("cpu", "cuda", "auto")
    precision: str = "fp32"  # Inference precision ("fp32", "fp16", "int8")
    max_batch_size: int = 1  # Maximum batch size for inference

    # Input source
    input_source: str = "microphone"  # "microphone", "file", "network"
    input_device_id: Optional[int] = None  # Audio input device ID
    input_file_path: Optional[str] = None  # Path to audio file

    # Alert system
    enable_alerts: bool = True  # Enable alert generation
    alert_cooldown_ms: float = 5000.0  # Cooldown between alerts

    # Performance optimization
    use_threaded_processing: bool = True  # Use multi-threaded processing
    memory_optimization: bool = True  # Enable memory optimizations
    warmup_iterations: int = 10  # Number of warmup iterations


class CircularAudioBuffer:
    """
    Circular buffer for continuous audio streaming.

    Efficiently manages audio data with automatic overwriting
    of old data when buffer is full.
    """

    def __init__(self, sample_rate: int, duration_ms: float, channels: int = 1):
        """
        Initialize circular audio buffer.

        Args:
            sample_rate: Audio sample rate
            duration_ms: Buffer duration in milliseconds
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.channels = channels

        # Calculate buffer size
        self.buffer_size = int(sample_rate * duration_ms / 1000.0) * channels

        # Initialize buffer
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_index = 0
        self.samples_written = 0

        # Thread safety
        self.lock = threading.Lock()

        logger.info(f"Initialized circular buffer: {self.buffer_size} samples ({duration_ms}ms)")

    def write(self, audio_data: np.ndarray):
        """
        Write audio data to buffer.

        Args:
            audio_data: Audio data to write
        """
        with self.lock:
            data_length = len(audio_data)

            # Handle wrapping around buffer
            if self.write_index + data_length <= self.buffer_size:
                # No wrapping needed
                self.buffer[self.write_index : self.write_index + data_length] = audio_data
            else:
                # Wrapping needed
                first_chunk_size = self.buffer_size - self.write_index
                second_chunk_size = data_length - first_chunk_size

                self.buffer[self.write_index :] = audio_data[:first_chunk_size]
                self.buffer[:second_chunk_size] = audio_data[first_chunk_size:]

            # Update write index
            self.write_index = (self.write_index + data_length) % self.buffer_size
            self.samples_written += data_length

    def read(self, num_samples: int, offset_samples: int = 0) -> np.ndarray:
        """
        Read audio data from buffer.

        Args:
            num_samples: Number of samples to read
            offset_samples: Offset from current write position

        Returns:
            Audio data array
        """
        with self.lock:
            if self.samples_written < num_samples + offset_samples:
                # Not enough data available
                return np.zeros(num_samples, dtype=np.float32)

            # Calculate read start position
            read_start = (self.write_index - num_samples - offset_samples) % self.buffer_size

            # Read data (handle wrapping)
            if read_start + num_samples <= self.buffer_size:
                # No wrapping needed
                data = self.buffer[read_start : read_start + num_samples].copy()
            else:
                # Wrapping needed
                first_chunk_size = self.buffer_size - read_start
                second_chunk_size = num_samples - first_chunk_size

                data = np.concatenate([self.buffer[read_start:], self.buffer[:second_chunk_size]])

            return data

    def get_latest_chunk(self, chunk_size: int) -> np.ndarray:
        """Get the most recent chunk of audio data"""
        return self.read(chunk_size, offset_samples=0)

    def is_ready(self, required_samples: int) -> bool:
        """Check if buffer has enough samples for processing"""
        with self.lock:
            return self.samples_written >= required_samples

    def reset(self):
        """Reset buffer to empty state"""
        with self.lock:
            self.buffer.fill(0)
            self.write_index = 0
            self.samples_written = 0


class LatencyTracker:
    """
    Tracks and monitors inference latency for performance optimization.
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize latency tracker.

        Args:
            history_size: Number of latency measurements to keep
        """
        self.history = deque(maxlen=history_size)
        self.total_inferences = 0
        self.start_time = None

    def start_timing(self):
        """Start timing an inference"""
        self.start_time = time.perf_counter()

    def end_timing(self) -> float:
        """
        End timing and record latency.

        Returns:
            Latency in milliseconds
        """
        if self.start_time is None:
            return 0.0

        end_time = time.perf_counter()
        latency_ms = (end_time - self.start_time) * 1000.0

        self.history.append(latency_ms)
        self.total_inferences += 1
        self.start_time = None

        return latency_ms

    def get_statistics(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.history:
            return {}

        latencies = list(self.history)
        return {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "max_latency_ms": np.max(latencies),
            "min_latency_ms": np.min(latencies),
            "std_latency_ms": np.std(latencies),
            "total_inferences": self.total_inferences,
        }

    def is_within_target(self, target_latency_ms: float) -> bool:
        """Check if recent latencies are within target"""
        if not self.history:
            return True

        recent_latencies = list(self.history)[-10:]  # Last 10 measurements
        mean_recent = np.mean(recent_latencies)
        return mean_recent <= target_latency_ms


class StreamingProcessor:
    """
    Processes streaming audio data for real-time inference.

    Handles audio preprocessing, feature extraction, and model inference
    in an optimized pipeline for low-latency operation.
    """

    def __init__(self, config: AudioStreamConfig):
        """
        Initialize streaming processor.

        Args:
            config: Audio streaming configuration
        """
        self.config = config

        # Initialize components
        self.audio_processor = AudioProcessor(
            {
                "sample_rate": config.sample_rate,
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "n_mels": config.n_mels,
            }
        )

        self.feature_extractor = FeatureExtractor(
            {"target_length": config.target_length, "sample_rate": config.sample_rate}
        )

        # Device management
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device() if config.device == "auto" else config.device

        # Preprocessing cache for efficiency
        self.preprocessing_cache = {}

        logger.info(f"Initialized streaming processor on device: {self.device}")

    def preprocess_audio_chunk(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio chunk for inference.

        Args:
            audio_chunk: Raw audio chunk

        Returns:
            Preprocessed spectrogram tensor
        """
        # Ensure proper shape
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk.reshape(1, -1)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Extract features
        features = self.feature_extractor.extract_features(audio_tensor, self.config.__dict__)

        # Move to device
        features = features.to(self.device)

        # Add batch dimension if needed
        if features.dim() == 3:
            features = features.unsqueeze(0)

        return features

    def apply_overlap_processing(
        self, current_chunk: np.ndarray, previous_chunk: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply overlap processing to smooth transitions between chunks.

        Args:
            current_chunk: Current audio chunk
            previous_chunk: Previous audio chunk for overlap

        Returns:
            Processed audio chunk with overlap
        """
        if previous_chunk is None:
            return current_chunk

        overlap_samples = int(len(current_chunk) * self.config.overlap_ratio)

        if overlap_samples > 0 and len(previous_chunk) >= overlap_samples:
            # Apply window function for smooth transition
            window = np.hanning(overlap_samples * 2)
            fade_out = window[:overlap_samples]
            fade_in = window[overlap_samples:]

            # Combine overlapping regions
            overlap_region = (
                previous_chunk[-overlap_samples:] * fade_out
                + current_chunk[:overlap_samples] * fade_in
            )

            # Construct final chunk
            processed_chunk = np.concatenate(
                [previous_chunk[:-overlap_samples], overlap_region, current_chunk[overlap_samples:]]
            )

            return processed_chunk

        return current_chunk

    def optimize_for_speed(self):
        """Apply speed optimizations"""
        if self.config.memory_optimization:
            # Clear preprocessing cache periodically
            self.preprocessing_cache.clear()

            # Force garbage collection
            import gc

            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class RealTimeInferenceEngine:
    """
    High-performance real-time inference engine for military vehicle detection.

    Provides sub-20ms latency inference with continuous audio streaming,
    optimized for edge deployment on Jetson and Raspberry Pi devices.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[AudioStreamConfig] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize real-time inference engine.

        Args:
            model_path: Path to trained model
            config: Audio streaming configuration
            class_names: Names of detection classes
        """
        self.model_path = model_path
        self.config = config or AudioStreamConfig()
        self.class_names = class_names or [
            "helicopter",
            "fighter_aircraft",
            "military_vehicle",
            "truck",
            "footsteps",
            "speech",
            "background",
        ]

        # Initialize components
        self.model_manager = ModelManager()
        self.streaming_processor = StreamingProcessor(self.config)
        self.latency_tracker = LatencyTracker()

        # Audio buffer
        self.audio_buffer = CircularAudioBuffer(
            sample_rate=self.config.sample_rate,
            duration_ms=self.config.buffer_duration_ms,
            channels=self.config.channels,
        )

        # Load and optimize model
        self._load_model()
        self._optimize_model()

        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.audio_thread = None
        self.previous_chunk = None

        # Alert system
        self.last_alert_time = 0
        self.alert_callbacks = []

        # Performance monitoring
        self.inference_count = 0
        self.total_processing_time = 0

        logger.info("Real-time inference engine initialized")

    def _load_model(self):
        """Load and prepare model for inference"""
        self.model = self.model_manager.load_model(
            model_path=self.model_path,
            device=self.config.device,
            optimization_level=2 if self.config.memory_optimization else 1,
        )

        # Set model to evaluation mode
        self.model.eval()

        # Warmup model
        self._warmup_model()

        logger.info(f"Model loaded from {self.model_path}")

    def _optimize_model(self):
        """Apply model optimizations for real-time inference"""
        if self.config.precision == "fp16" and torch.cuda.is_available():
            self.model = self.model.half()
            logger.info("Applied FP16 optimization")

        # Compile model if supported (PyTorch 2.0+)
        if hasattr(torch, "compile") and self.config.memory_optimization:
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")

    def _warmup_model(self):
        """Warmup model with dummy inputs"""
        logger.info("Warming up model...")

        # Create dummy input
        dummy_input = torch.randn(
            self.config.max_batch_size,
            1,  # channels
            self.config.n_mels,
            self.config.target_length,
            device=self.streaming_processor.device,
        )

        if self.config.precision == "fp16":
            dummy_input = dummy_input.half()

        # Warmup iterations
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                self.latency_tracker.start_timing()
                _ = self.model(dummy_input)
                self.latency_tracker.end_timing()

        warmup_stats = self.latency_tracker.get_statistics()
        logger.info(
            f"Model warmup completed. Average latency: {warmup_stats.get('mean_latency_ms', 0):.2f}ms"
        )

    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """
        Add callback function for detection alerts.

        Args:
            callback: Function to call when detection occurs
        """
        self.alert_callbacks.append(callback)

    def _trigger_alert(self, detection_result: Dict):
        """Trigger alert callbacks"""
        current_time = time.time() * 1000  # Convert to milliseconds

        # Check cooldown
        if current_time - self.last_alert_time < self.config.alert_cooldown_ms:
            return

        # Check confidence threshold
        if detection_result["confidence"] < self.config.confidence_threshold:
            return

        self.last_alert_time = current_time

        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(detection_result)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _audio_capture_thread(self):
        """Thread function for audio capture"""
        if not SOUNDDEVICE_AVAILABLE and not PYAUDIO_AVAILABLE:
            logger.error("No audio library available. Install sounddevice or pyaudio.")
            return

        try:
            if SOUNDDEVICE_AVAILABLE:
                self._capture_with_sounddevice()
            elif PYAUDIO_AVAILABLE:
                self._capture_with_pyaudio()
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            self.is_running = False

    def _capture_with_sounddevice(self):
        """Audio capture using sounddevice"""

        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")

            # Convert to mono if needed
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]

            # Write to buffer
            self.audio_buffer.write(audio_data)

        # Start audio stream
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_size,
            device=self.config.input_device_id,
            callback=audio_callback,
            dtype=np.float32,
        ):
            while self.is_running:
                time.sleep(0.1)

    def _capture_with_pyaudio(self):
        """Audio capture using pyaudio"""
        import pyaudio

        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            input_device_index=self.config.input_device_id,
        )

        try:
            while self.is_running:
                audio_data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                self.audio_buffer.write(audio_array)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _processing_thread(self):
        """Thread function for audio processing and inference"""
        chunk_samples = int(self.config.sample_rate * 0.1)  # 100ms chunks

        while self.is_running:
            try:
                # Check if buffer has enough data
                if not self.audio_buffer.is_ready(chunk_samples):
                    time.sleep(0.01)  # Wait 10ms
                    continue

                # Get audio chunk
                audio_chunk = self.audio_buffer.get_latest_chunk(chunk_samples)

                # Apply overlap processing
                if self.previous_chunk is not None:
                    audio_chunk = self.streaming_processor.apply_overlap_processing(
                        audio_chunk, self.previous_chunk
                    )

                self.previous_chunk = audio_chunk

                # Process and infer
                result = self._process_audio_chunk(audio_chunk)

                # Handle detection
                if result and self.config.enable_alerts:
                    self._trigger_alert(result)

                # Performance optimization
                if self.inference_count % 100 == 0:
                    self.streaming_processor.optimize_for_speed()

            except Exception as e:
                logger.error(f"Processing thread error: {e}")
                time.sleep(0.1)

    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process audio chunk and perform inference.

        Args:
            audio_chunk: Audio data to process

        Returns:
            Detection result or None
        """
        self.latency_tracker.start_timing()

        try:
            # Preprocess audio
            features = self.streaming_processor.preprocess_audio_chunk(audio_chunk)

            # Inference
            with torch.no_grad():
                logits = self.model(features)
                probabilities = torch.softmax(logits, dim=-1)

                # Get prediction
                pred_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0, pred_idx].item()

                # Create result
                result = {
                    "prediction": self.class_names[pred_idx],
                    "prediction_id": pred_idx,
                    "confidence": confidence,
                    "probabilities": probabilities[0].cpu().numpy().tolist(),
                    "timestamp": time.time(),
                    "latency_ms": 0,  # Will be updated below
                }

                # Update performance metrics
                latency_ms = self.latency_tracker.end_timing()
                result["latency_ms"] = latency_ms

                self.inference_count += 1
                self.total_processing_time += latency_ms

                # Log performance periodically
                if self.inference_count % 100 == 0:
                    avg_latency = self.total_processing_time / self.inference_count
                    logger.debug(
                        f"Average latency: {avg_latency:.2f}ms over {self.inference_count} inferences"
                    )

                return result

        except Exception as e:
            logger.error(f"Inference error: {e}")
            self.latency_tracker.end_timing()
            return None

    def start_streaming(self):
        """Start real-time audio streaming and processing"""
        if self.is_running:
            logger.warning("Inference engine is already running")
            return

        logger.info("Starting real-time inference engine...")
        self.is_running = True

        # Start audio capture thread
        if self.config.input_source == "microphone":
            self.audio_thread = threading.Thread(target=self._audio_capture_thread, daemon=True)
            self.audio_thread.start()

        # Start processing thread
        if self.config.use_threaded_processing:
            self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
            self.processing_thread.start()

        logger.info("Real-time inference engine started")

    def stop_streaming(self):
        """Stop real-time audio streaming and processing"""
        if not self.is_running:
            return

        logger.info("Stopping real-time inference engine...")
        self.is_running = False

        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        logger.info("Real-time inference engine stopped")

    def process_audio_file(self, file_path: str) -> List[Dict]:
        """
        Process audio file in real-time simulation.

        Args:
            file_path: Path to audio file

        Returns:
            List of detection results
        """
        import librosa

        # Load audio file
        audio, sr = librosa.load(file_path, sr=self.config.sample_rate, mono=True)

        # Process in chunks
        chunk_size = int(self.config.sample_rate * 0.1)  # 100ms chunks
        results = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            result = self._process_audio_chunk(chunk)
            if result:
                result["file_position_s"] = i / self.config.sample_rate
                results.append(result)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        latency_stats = self.latency_tracker.get_statistics()

        stats = {
            "latency_statistics": latency_stats,
            "total_inferences": self.inference_count,
            "average_latency_ms": self.total_processing_time / max(self.inference_count, 1),
            "is_within_target": self.latency_tracker.is_within_target(self.config.max_latency_ms),
            "buffer_status": {
                "samples_written": self.audio_buffer.samples_written,
                "buffer_size": self.audio_buffer.buffer_size,
                "utilization": min(
                    1.0, self.audio_buffer.samples_written / self.audio_buffer.buffer_size
                ),
            },
        }

        return stats

    def __enter__(self):
        """Context manager entry"""
        self.start_streaming()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_streaming()
