#
# Plan:
# 1. Create real-time inference pipeline with <20ms latency requirement
# 2. Implement circular buffer for continuous audio streaming
# 3. Multi-threading for audio I/O and model inference
# 4. Support for Jetson Orin Nano (<10ms) and RPi 5 + AI HAT+ (<20ms)
# 5. Integration with TensorRT and ONNX optimization
# 6. Configurable detection thresholds and callbacks
# 7. Real-time spectrogram computation and processing
#

"""
Real-time Military Vehicle Detection Inference Pipeline
Optimized for edge deployment with <20ms latency requirements.

Features:
- Circular buffer audio streaming
- Real-time spectrogram computation
- Model inference optimization
- Multi-threading for audio I/O
- Configurable detection thresholds
- Low-latency processing pipeline
"""

import torch
import torchaudio
import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import sounddevice as sd
from collections import deque
import json

from core.core.audio_processor import AudioProcessor
from core.core.model_manager import ModelManager
from core.utils.device_utils import get_optimal_device
from core.utils.config_parser import ConfigParser

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result data structure"""

    label: str
    confidence: float
    class_id: int
    timestamp: float
    processing_time: float
    audio_segment_start: float
    audio_segment_end: float
    raw_logits: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "audio_segment_start": self.audio_segment_start,
            "audio_segment_end": self.audio_segment_end,
        }


@dataclass
class InferenceConfig:
    """Real-time inference configuration"""

    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1024  # Audio buffer chunk size
    channels: int = 1
    window_length: float = 2.0  # seconds
    overlap: float = 0.5

    # Detection settings
    confidence_threshold: float = 0.7
    class_names: List[str] = None

    # Performance settings
    device: str = "auto"  # auto, cuda, cpu, tensorrt
    batch_size: int = 1
    max_queue_size: int = 10
    processing_timeout: float = 0.1  # seconds

    # Model settings
    model_path: str = "models/serenesense_best.pth"
    optimization: str = "none"  # none, tensorrt, onnx
    precision: str = "fp32"  # fp32, fp16, int8

    # Callback settings
    detection_callback: Optional[Callable] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [
                "helicopter",
                "fighter_aircraft",
                "military_vehicle",
                "truck",
                "footsteps",
                "speech",
                "background",
            ]


class AudioStreamManager:
    """
    Manages real-time audio streaming with circular buffering.
    Optimized for low-latency military vehicle detection.
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize audio stream manager.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.is_streaming = False
        self.buffer_lock = threading.Lock()

        # Calculate buffer sizes
        self.window_samples = int(config.window_length * config.sample_rate)
        self.overlap_samples = int(self.window_samples * config.overlap)
        self.hop_samples = self.window_samples - self.overlap_samples

        # Circular buffer for audio data
        self.buffer_size = self.window_samples * 3  # 3x window size
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.last_read_pos = 0

        # Audio stream
        self.stream = None

        logger.info(f"AudioStreamManager initialized:")
        logger.info(f"  Window: {config.window_length}s ({self.window_samples} samples)")
        logger.info(f"  Overlap: {config.overlap} ({self.overlap_samples} samples)")
        logger.info(f"  Hop: {self.hop_samples} samples")
        logger.info(f"  Buffer size: {self.buffer_size} samples")

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Audio callback for real-time streaming.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Stream status
        """
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Convert to mono if needed
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]

        # Write to circular buffer
        with self.buffer_lock:
            end_pos = self.write_pos + len(audio_data)

            if end_pos <= self.buffer_size:
                # Simple case: no wraparound
                self.audio_buffer[self.write_pos : end_pos] = audio_data
            else:
                # Wraparound case
                first_part = self.buffer_size - self.write_pos
                self.audio_buffer[self.write_pos :] = audio_data[:first_part]
                self.audio_buffer[: end_pos - self.buffer_size] = audio_data[first_part:]

            self.write_pos = end_pos % self.buffer_size

    def get_audio_window(self) -> Optional[np.ndarray]:
        """
        Get next audio window for processing.

        Returns:
            Audio window or None if not enough data
        """
        with self.buffer_lock:
            # Check if we have enough new data
            available_samples = (self.write_pos - self.last_read_pos) % self.buffer_size

            if available_samples < self.hop_samples:
                return None

            # Extract window
            start_pos = (self.last_read_pos + self.hop_samples) % self.buffer_size
            end_pos = (start_pos + self.window_samples) % self.buffer_size

            if end_pos > start_pos:
                # Simple case: no wraparound
                window = self.audio_buffer[start_pos:end_pos].copy()
            else:
                # Wraparound case
                window = np.concatenate(
                    [self.audio_buffer[start_pos:], self.audio_buffer[:end_pos]]
                )

            self.last_read_pos = start_pos

            return window

    def start_stream(self):
        """Start audio streaming"""
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.chunk_size,
                dtype=np.float32,
            )
            self.stream.start()
            self.is_streaming = True
            logger.info("Audio streaming started")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise

    def stop_stream(self):
        """Stop audio streaming"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_streaming = False
        logger.info("Audio streaming stopped")


class RealTimeInference:
    """
    Real-time military vehicle detection system.
    Optimized for <20ms latency on edge devices.
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize real-time inference system.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.is_running = False

        # Initialize components
        self.device = get_optimal_device(config.device)
        self.audio_processor = AudioProcessor(
            {
                "sample_rate": config.sample_rate,
                "n_mels": 128,
                "n_fft": 1024,
                "hop_length": 512,
                "win_length": 1024,
                "normalize": True,
            }
        )

        # Load model
        self.model_manager = ModelManager()
        self.model = self.model_manager.load_model(
            config.model_path,
            device=self.device,
            optimization=config.optimization,
            precision=config.precision,
        )

        # Audio streaming
        self.stream_manager = AudioStreamManager(config)

        # Processing queues
        self.audio_queue = queue.Queue(maxsize=config.max_queue_size)
        self.result_queue = queue.Queue(maxsize=config.max_queue_size)

        # Threading
        self.processing_thread = None
        self.callback_thread = None

        # Statistics
        self.stats = {
            "processed_windows": 0,
            "detections": 0,
            "avg_processing_time": 0.0,
            "max_processing_time": 0.0,
        }

        logger.info(f"RealTimeInference initialized on {self.device}")
        logger.info(f"Model: {config.model_path}")
        logger.info(f"Optimization: {config.optimization}")
        logger.info(f"Classes: {config.class_names}")

    def _process_audio_window(
        self, audio_window: np.ndarray, timestamp: float
    ) -> Optional[DetectionResult]:
        """
        Process audio window through model.

        Args:
            audio_window: Audio data window
            timestamp: Window timestamp

        Returns:
            Detection result or None
        """
        start_time = time.time()

        try:
            # Convert to spectrogram
            spectrogram = self.audio_processor.to_spectrogram(
                torch.from_numpy(audio_window).unsqueeze(0)
            )

            # Prepare input
            if len(spectrogram.shape) == 3:
                spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension

            spectrogram = spectrogram.to(self.device)

            # Model inference
            with torch.no_grad():
                if hasattr(self.model, "predict"):
                    logits = self.model.predict(spectrogram)
                else:
                    logits = self.model(spectrogram)

                # Get predictions
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)

                confidence = confidence.item()
                predicted_class = predicted_class.item()

            processing_time = time.time() - start_time

            # Update statistics
            self.stats["processed_windows"] += 1
            self.stats["avg_processing_time"] = (
                self.stats["avg_processing_time"] * (self.stats["processed_windows"] - 1)
                + processing_time
            ) / self.stats["processed_windows"]
            self.stats["max_processing_time"] = max(
                self.stats["max_processing_time"], processing_time
            )

            # Check confidence threshold
            if confidence >= self.config.confidence_threshold:
                self.stats["detections"] += 1

                return DetectionResult(
                    label=self.config.class_names[predicted_class],
                    confidence=confidence,
                    class_id=predicted_class,
                    timestamp=timestamp,
                    processing_time=processing_time,
                    audio_segment_start=timestamp - self.config.window_length,
                    audio_segment_end=timestamp,
                    raw_logits=logits,
                )

            return None

        except Exception as e:
            logger.error(f"Error processing audio window: {e}")
            return None

    def _processing_worker(self):
        """Worker thread for audio processing"""
        logger.info("Processing worker started")

        while self.is_running:
            try:
                # Get audio window from stream
                audio_window = self.stream_manager.get_audio_window()

                if audio_window is not None:
                    # Process window
                    timestamp = time.time()
                    result = self._process_audio_window(audio_window, timestamp)

                    # Add result to queue if detection found
                    if result is not None:
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            logger.warning("Result queue full, dropping detection")

                else:
                    # No new audio data, sleep briefly
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
                if self.is_running:
                    time.sleep(0.1)

    def _callback_worker(self):
        """Worker thread for handling detection callbacks"""
        logger.info("Callback worker started")

        while self.is_running:
            try:
                # Get detection result
                result = self.result_queue.get(timeout=0.1)

                # Call detection callback if configured
                if self.config.detection_callback:
                    try:
                        self.config.detection_callback(result)
                    except Exception as e:
                        logger.error(f"Error in detection callback: {e}")

                logger.info(
                    f"Detection: {result.label} ({result.confidence:.3f}) "
                    f"at {result.timestamp:.3f}s (processed in {result.processing_time*1000:.1f}ms)"
                )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in callback worker: {e}")

    def start(self):
        """Start real-time inference"""
        if self.is_running:
            logger.warning("Real-time inference already running")
            return

        logger.info("Starting real-time inference...")
        self.is_running = True

        # Start audio streaming
        self.stream_manager.start_stream()

        # Start worker threads
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.callback_thread = threading.Thread(target=self._callback_worker, daemon=True)

        self.processing_thread.start()
        self.callback_thread.start()

        logger.info("Real-time inference started successfully")

    def stop(self):
        """Stop real-time inference"""
        if not self.is_running:
            return

        logger.info("Stopping real-time inference...")
        self.is_running = False

        # Stop audio streaming
        self.stream_manager.stop_stream()

        # Wait for threads to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.callback_thread:
            self.callback_thread.join(timeout=1.0)

        logger.info("Real-time inference stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            **self.stats,
            "is_running": self.is_running,
            "audio_streaming": self.stream_manager.is_streaming,
            "queue_sizes": {
                "audio": self.audio_queue.qsize(),
                "results": self.result_queue.qsize(),
            },
        }

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# Example usage and demo functions
def example_detection_callback(result: DetectionResult):
    """Example detection callback"""
    print(f"🚨 DETECTION: {result.label} ({result.confidence:.2f}) at {result.timestamp:.2f}s")

    # Here you could:
    # - Send alerts to monitoring system
    # - Log to database
    # - Trigger other actions


def create_real_time_detector(config_path: str = None) -> RealTimeInference:
    """
    Create real-time detector from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured real-time inference system
    """
    if config_path:
        config_dict = ConfigParser.load_config(config_path)
        config = InferenceConfig(**config_dict.get("inference", {}))
    else:
        config = InferenceConfig()

    # Set default callback if none provided
    if config.detection_callback is None:
        config.detection_callback = example_detection_callback

    return RealTimeInference(config)


if __name__ == "__main__":
    # Demo: Real-time military vehicle detection
    logging.basicConfig(level=logging.INFO)

    # Create detector
    detector = create_real_time_detector()

    try:
        print("🎧 Starting real-time military vehicle detection...")
        print("   Listening for: helicopter, fighter_aircraft, military_vehicle")
        print("   Press Ctrl+C to stop")

        with detector:
            while True:
                time.sleep(1)

                # Print statistics every 10 seconds
                stats = detector.get_statistics()
                if stats["processed_windows"] % 100 == 0 and stats["processed_windows"] > 0:
                    print(
                        f"📊 Stats: {stats['processed_windows']} windows, "
                        f"{stats['detections']} detections, "
                        f"avg: {stats['avg_processing_time']*1000:.1f}ms"
                    )

    except KeyboardInterrupt:
        print("\n🛑 Stopping detector...")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
