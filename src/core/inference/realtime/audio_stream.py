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

from core.core.audio_processor import RealTimeAudioProcessor, AudioConfig
from core.models.audioMAE.model import AudioMAE
from core.inference.optimization.tensorrt import TensorRTOptimizer
from core.inference.optimization.onnx_export import ONNXExporter

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
    device: str = "cuda"  # cuda, cpu, tensorrt
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
        self.config = config
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=config.max_queue_size)
        self.stream = None

        # Audio buffer for continuous processing
        buffer_size = int(config.sample_rate * config.window_length * 2)
        self.audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_ptr = 0
        self.buffer_lock = threading.Lock()

        # Statistics
        self.stats = {
            "chunks_processed": 0,
            "buffer_overflows": 0,
            "queue_overflows": 0,
            "avg_chunk_time": 0.0,
        }

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Audio input callback function"""
        if status:
            logger.warning(f"Audio callback status: {status}")

        try:
            # Convert to mono if stereo
            if indata.shape[1] > 1:
                audio_chunk = np.mean(indata, axis=1)
            else:
                audio_chunk = indata[:, 0]

            # Add to circular buffer
            with self.buffer_lock:
                chunk_size = len(audio_chunk)

                if self.buffer_ptr + chunk_size <= len(self.audio_buffer):
                    self.audio_buffer[self.buffer_ptr : self.buffer_ptr + chunk_size] = audio_chunk
                    self.buffer_ptr += chunk_size
                else:
                    # Buffer overflow - shift and add
                    overflow = self.buffer_ptr + chunk_size - len(self.audio_buffer)
                    self.audio_buffer = np.roll(self.audio_buffer, -overflow)
                    self.audio_buffer[-chunk_size:] = audio_chunk
                    self.buffer_ptr = len(self.audio_buffer)
                    self.stats["buffer_overflows"] += 1

            # Add to processing queue
            try:
                self.audio_queue.put_nowait({"audio": audio_chunk.copy(), "timestamp": time.time()})
                self.stats["chunks_processed"] += 1
            except queue.Full:
                self.stats["queue_overflows"] += 1

        except Exception as e:
            logger.error(f"Audio callback error: {e}")

    def start_streaming(self):
        """Start audio streaming"""
        if self.is_streaming:
            return

        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                callback=self._audio_callback,
                dtype=np.float32,
            )

            self.stream.start()
            self.is_streaming = True
            logger.info("Audio streaming started")

        except Exception as e:
            logger.error(f"Failed to start audio streaming: {e}")
            raise

    def stop_streaming(self):
        """Stop audio streaming"""
        if not self.is_streaming:
            return

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.is_streaming = False
        logger.info("Audio streaming stopped")

    def get_audio_window(self) -> Optional[np.ndarray]:
        """Get audio window for processing"""
        with self.buffer_lock:
            window_samples = int(self.config.sample_rate * self.config.window_length)

            if self.buffer_ptr >= window_samples:
                # Extract window
                window = self.audio_buffer[:window_samples].copy()
                return window

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return self.stats.copy()


class ModelInferenceEngine:
    """
    Optimized model inference engine for real-time detection.
    Supports multiple acceleration backends.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Load and optimize model
        self.model = self._load_model()
        self._optimize_model()

        # Audio processor
        audio_config = AudioConfig(
            sample_rate=config.sample_rate, window_length=config.window_length
        )
        self.audio_processor = RealTimeAudioProcessor(audio_config)

        # Performance tracking
        self.inference_times = deque(maxlen=100)

        logger.info(f"Model loaded on device: {self.device}")

    def _load_model(self) -> torch.Module:
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(self.config.model_path, map_location=self.device)

            # Extract model state dict
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Initialize model (you might need to adjust this based on your model config)
            from core.models.audioMAE.model import AudioMAE, AudioMAEConfig

            model_config = AudioMAEConfig(num_classes=len(self.config.class_names))
            model = AudioMAE(model_config)

            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device)

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _optimize_model(self):
        """Optimize model for inference"""
        if self.config.optimization == "tensorrt":
            self._optimize_with_tensorrt()
        elif self.config.optimization == "onnx":
            self._optimize_with_onnx()
        elif self.config.precision == "fp16" and self.device.type == "cuda":
            self.model = self.model.half()

    def _optimize_with_tensorrt(self):
        """Optimize model with TensorRT"""
        try:
            optimizer = TensorRTOptimizer()
            self.model = optimizer.optimize(
                self.model,
                input_shape=(1, 1, 128, 128),  # Adjust based on your input
                precision=self.config.precision,
            )
            logger.info("Model optimized with TensorRT")
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")

    def _optimize_with_onnx(self):
        """Optimize model with ONNX Runtime"""
        try:
            exporter = ONNXExporter()
            onnx_path = "temp_model.onnx"
            exporter.export(self.model, input_shape=(1, 1, 128, 128), output_path=onnx_path)

            import onnxruntime as ort

            # Setup ONNX Runtime session
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
            self.use_onnx = True

            logger.info("Model optimized with ONNX Runtime")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
            self.use_onnx = False

    def predict(self, audio_data: np.ndarray) -> DetectionResult:
        """
        Perform inference on audio data.

        Args:
            audio_data: Raw audio data

        Returns:
            Detection result
        """
        start_time = time.time()

        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)

            # Compute mel-spectrogram
            mel_spec = self.audio_processor.compute_mel_spectrogram(audio_tensor)

            # Preprocess for model
            input_tensor = self.audio_processor.preprocess_for_model(mel_spec)
            input_tensor = input_tensor.to(self.device)

            # Model inference
            with torch.no_grad():
                if hasattr(self, "use_onnx") and self.use_onnx:
                    # ONNX Runtime inference
                    ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
                    ort_outputs = self.ort_session.run(None, ort_inputs)
                    logits = torch.from_numpy(ort_outputs[0])
                else:
                    # PyTorch inference
                    if self.config.precision == "fp16" and self.device.type == "cuda":
                        input_tensor = input_tensor.half()

                    outputs = self.model(input_tensor, mode="classification")
                    logits = outputs["logits"]

            # Get predictions
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)

            confidence = confidence.item()
            predicted_class = predicted_class.item()

            # Create result
            result = DetectionResult(
                label=self.config.class_names[predicted_class],
                confidence=confidence,
                class_id=predicted_class,
                timestamp=time.time(),
                processing_time=time.time() - start_time,
                audio_segment_start=start_time - self.config.window_length,
                audio_segment_end=start_time,
                raw_logits=logits.cpu() if self.config.device != "cpu" else logits,
            )

            # Track performance
            self.inference_times.append(result.processing_time)

            return result

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Return default result
            return DetectionResult(
                label="unknown",
                confidence=0.0,
                class_id=-1,
                timestamp=time.time(),
                processing_time=time.time() - start_time,
                audio_segment_start=start_time - self.config.window_length,
                audio_segment_end=start_time,
            )

    def get_avg_inference_time(self) -> float:
        """Get average inference time"""
        if self.inference_times:
            return sum(self.inference_times) / len(self.inference_times)
        return 0.0


class RealTimeDetector:
    """
    Complete real-time military vehicle detection system.
    Integrates audio streaming, processing, and model inference.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.is_running = False

        # Components
        self.stream_manager = AudioStreamManager(config)
        self.inference_engine = ModelInferenceEngine(config)

        # Processing thread
        self.processing_thread = None
        self.stop_event = threading.Event()

        # Results storage
        self.recent_detections = deque(maxlen=100)
        self.detection_counts = {name: 0 for name in config.class_names}

        logger.info("Real-time detector initialized")

    def start(self):
        """Start real-time detection"""
        if self.is_running:
            return

        try:
            # Start audio streaming
            self.stream_manager.start_streaming()

            # Start processing thread
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            self.is_running = True
            logger.info("Real-time detection started")

        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop real-time detection"""
        if not self.is_running:
            return

        # Stop processing
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

        # Stop audio streaming
        self.stream_manager.stop_streaming()

        self.is_running = False
        logger.info("Real-time detection stopped")

    def _processing_loop(self):
        """Main processing loop"""
        while not self.stop_event.is_set():
            try:
                # Get audio window
                audio_window = self.stream_manager.get_audio_window()

                if audio_window is not None:
                    # Perform inference
                    result = self.inference_engine.predict(audio_window)

                    # Filter by confidence threshold
                    if result.confidence >= self.config.confidence_threshold:
                        self._handle_detection(result)

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Processing loop error: {e}")

    def _handle_detection(self, result: DetectionResult):
        """Handle a positive detection"""
        # Store result
        self.recent_detections.append(result)
        self.detection_counts[result.label] += 1

        # Log detection
        logger.info(
            f"DETECTION: {result.label} ({result.confidence:.3f}) "
            f"at {result.timestamp:.2f}s (processing: {result.processing_time*1000:.1f}ms)"
        )

        # Call user callback if provided
        if self.config.detection_callback:
            try:
                self.config.detection_callback(result)
            except Exception as e:
                logger.error(f"Detection callback failed: {e}")

    def get_recent_detections(self, n: int = 10) -> List[DetectionResult]:
        """Get N most recent detections"""
        return list(self.recent_detections)[-n:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "is_running": self.is_running,
            "total_detections": sum(self.detection_counts.values()),
            "detection_counts": self.detection_counts.copy(),
            "avg_inference_time": self.inference_engine.get_avg_inference_time(),
            "audio_stats": self.stream_manager.get_stats(),
            "recent_detections_count": len(self.recent_detections),
        }

        if self.recent_detections:
            recent_confidences = [det.confidence for det in self.recent_detections]
            stats["avg_confidence"] = sum(recent_confidences) / len(recent_confidences)
            stats["max_confidence"] = max(recent_confidences)
            stats["min_confidence"] = min(recent_confidences)

        return stats

    def export_detections(self, filepath: str):
        """Export detection history to JSON file"""
        detections_data = []

        for detection in self.recent_detections:
            detection_dict = {
                "label": detection.label,
                "confidence": detection.confidence,
                "class_id": detection.class_id,
                "timestamp": detection.timestamp,
                "processing_time": detection.processing_time,
                "audio_segment_start": detection.audio_segment_start,
                "audio_segment_end": detection.audio_segment_end,
            }
            detections_data.append(detection_dict)

        with open(filepath, "w") as f:
            json.dump(detections_data, f, indent=2)

        logger.info(f"Exported {len(detections_data)} detections to {filepath}")


def create_realtime_detector(config_path: str) -> RealTimeDetector:
    """
    Factory function to create real-time detector from configuration.

    Args:
        config_path: Path to inference configuration file

    Returns:
        Configured RealTimeDetector instance
    """
    import yaml

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    inference_config = InferenceConfig(**config_dict.get("inference", {}))

    return RealTimeDetector(inference_config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create detector with default configuration
    config = InferenceConfig(
        model_path="models/serenesense_best.pth",
        confidence_threshold=0.7,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    def detection_callback(result: DetectionResult):
        """Example detection callback"""
        if result.label in ["helicopter", "fighter_aircraft", "military_vehicle"]:
            print(f"🚨 MILITARY VEHICLE DETECTED: {result.label} ({result.confidence:.2f})")

    config.detection_callback = detection_callback

    # Create and start detector
    detector = RealTimeDetector(config)

    try:
        detector.start()
        print("Real-time detection running... Press Ctrl+C to stop")

        # Monitor performance
        while True:
            time.sleep(10)
            stats = detector.get_statistics()
            print(
                f"Stats: {stats['total_detections']} detections, "
                f"avg inference: {stats['avg_inference_time']*1000:.1f}ms"
            )

    except KeyboardInterrupt:
        print("\nStopping detection...")
        detector.stop()

        # Export results
        detector.export_detections("detection_results.json")
        print("Detection results exported")
