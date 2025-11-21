"""
Real-time military vehicle sound detection on Raspberry Pi 5.

This script provides real-time audio capture and classification using
the trained AudioMAE model optimized for edge deployment.

Features:
- Real-time audio capture from microphone
- Efficient preprocessing pipeline
- ONNX Runtime inference (INT8 optimized)
- Continuous detection with configurable intervals
- Display predictions with confidence scores

Author: SereneSense Team
Date: 2025-11-21
"""

import numpy as np
import onnxruntime as ort
import pyaudio
import argparse
import time
import sys
from pathlib import Path
from collections import deque
from datetime import datetime

# Import preprocessing module
from rpi_preprocessing import AudioPreprocessor


class MilitaryVehicleDetector:
    """
    Real-time military vehicle sound detector for Raspberry Pi 5.
    """

    # Class labels (MAD dataset)
    CLASS_LABELS = [
        "Helicopter",
        "Fighter Aircraft",
        "Military Vehicle",
        "Truck",
        "Footsteps",
        "Speech",
        "Background"
    ]

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        duration: float = 10.0,
        confidence_threshold: float = 0.5,
        use_gpu: bool = False
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to ONNX model file
            sample_rate: Audio sample rate (Hz)
            duration: Audio clip duration for inference (seconds)
            confidence_threshold: Minimum confidence for valid detection
            use_gpu: Use GPU acceleration if available (unlikely on RPi)
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.confidence_threshold = confidence_threshold

        print("=" * 70)
        print("Military Vehicle Sound Detector - Raspberry Pi 5")
        print("=" * 70)

        # Initialize preprocessor
        print("\nInitializing audio preprocessor...")
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            duration=duration
        )

        # Initialize ONNX Runtime session
        print(f"\nLoading ONNX model: {model_path}")
        self._initialize_model(use_gpu)

        # Audio capture settings
        self.chunk_size = 1024  # Samples per read
        self.audio_buffer = deque(maxlen=int(sample_rate * duration))

        print("\n✓ Detector initialized successfully!")
        print("=" * 70)

    def _initialize_model(self, use_gpu: bool = False):
        """Initialize ONNX Runtime inference session."""

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Select execution provider
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Create inference session
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"  Model loaded: {Path(self.model_path).name}")
        print(f"  Execution provider: {self.session.get_providers()[0]}")
        print(f"  Input: {self.input_name}")
        print(f"  Output: {self.output_name}")

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def predict(self, audio: np.ndarray) -> tuple:
        """
        Run inference on audio data.

        Args:
            audio: Audio waveform (mono, normalized)

        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Preprocess audio
        spectrogram = self.preprocessor.preprocess(
            (audio, self.sample_rate),
            input_type='array'
        )

        # Run inference
        start_time = time.time()
        logits = self.session.run(
            [self.output_name],
            {self.input_name: spectrogram}
        )[0]
        inference_time = (time.time() - start_time) * 1000  # ms

        # Convert logits to probabilities
        probabilities = self.softmax(logits[0])

        # Get predicted class and confidence
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return predicted_class, confidence, probabilities, inference_time

    def capture_audio_realtime(
        self,
        duration: float = None
    ) -> np.ndarray:
        """
        Capture audio from microphone in real-time.

        Args:
            duration: Recording duration (seconds), defaults to self.duration

        Returns:
            Audio waveform as numpy array
        """
        if duration is None:
            duration = self.duration

        # Initialize PyAudio
        audio_interface = pyaudio.PyAudio()

        # Open audio stream
        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,  # Mono
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        print(f"Recording {duration} seconds of audio...")

        # Calculate number of chunks to record
        num_chunks = int(self.sample_rate / self.chunk_size * duration)

        # Record audio
        frames = []
        for _ in range(num_chunks):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)

        # Stop and close stream
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

        # Normalize to [-1, 1]
        audio_data = audio_data.astype(np.float32) / 32768.0

        return audio_data

    def run_continuous_detection(
        self,
        interval: float = 10.0,
        max_detections: int = None,
        verbose: bool = True
    ):
        """
        Run continuous detection loop.

        Args:
            interval: Time between detections (seconds)
            max_detections: Maximum number of detections (None = infinite)
            verbose: Print detailed information
        """
        print("\n" + "=" * 70)
        print("Starting Continuous Detection")
        print("=" * 70)
        print(f"Detection interval: {interval} seconds")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Press Ctrl+C to stop\n")

        detection_count = 0

        try:
            while True:
                # Check if max detections reached
                if max_detections is not None and detection_count >= max_detections:
                    print(f"\nReached maximum detections ({max_detections})")
                    break

                # Capture audio
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Capturing audio...")

                audio = self.capture_audio_realtime(duration=interval)

                # Run prediction
                pred_class, confidence, probabilities, inf_time = self.predict(audio)
                class_name = self.CLASS_LABELS[pred_class]

                # Display results
                print(f"[{timestamp}] Detection #{detection_count + 1}")
                print(f"  Predicted: {class_name}")
                print(f"  Confidence: {confidence:.2%}")
                print(f"  Inference time: {inf_time:.1f} ms")

                if verbose:
                    print(f"  All probabilities:")
                    for i, (label, prob) in enumerate(zip(self.CLASS_LABELS, probabilities)):
                        marker = "→" if i == pred_class else " "
                        print(f"    {marker} {label:20s}: {prob:.2%}")

                # Check confidence threshold
                if confidence < self.confidence_threshold:
                    print(f"  ⚠ Low confidence (< {self.confidence_threshold:.2%})")

                print("-" * 70)

                detection_count += 1

        except KeyboardInterrupt:
            print(f"\n\nDetection stopped by user.")
            print(f"Total detections: {detection_count}")
            print("=" * 70)

    def run_file_detection(self, audio_path: str):
        """
        Run detection on audio file.

        Args:
            audio_path: Path to audio file
        """
        print(f"\nRunning detection on file: {audio_path}")

        # Load audio
        audio = self.preprocessor.load_audio(audio_path)

        # Run prediction
        pred_class, confidence, probabilities, inf_time = self.predict(audio)
        class_name = self.CLASS_LABELS[pred_class]

        # Display results
        print(f"\nDetection Results:")
        print(f"  File: {Path(audio_path).name}")
        print(f"  Predicted: {class_name}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Inference time: {inf_time:.1f} ms")
        print(f"\n  All probabilities:")
        for i, (label, prob) in enumerate(zip(self.CLASS_LABELS, probabilities)):
            marker = "→" if i == pred_class else " "
            print(f"    {marker} {label:20s}: {prob:.2%}")

        print("=" * 70)


def main():
    """Main entry point for Raspberry Pi deployment."""

    parser = argparse.ArgumentParser(
        description="Real-time military vehicle sound detection on Raspberry Pi 5"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='audiomae_int8.onnx',
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['realtime', 'file'],
        default='realtime',
        help='Detection mode: realtime or file'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Audio file path (required for file mode)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=10.0,
        help='Detection interval in seconds (realtime mode)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for valid detections'
    )
    parser.add_argument(
        '--max-detections',
        type=int,
        default=None,
        help='Maximum number of detections (realtime mode)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed probability distributions'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration if available'
    )

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Create detector
    detector = MilitaryVehicleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        use_gpu=args.gpu
    )

    # Run detection
    if args.mode == 'realtime':
        detector.run_continuous_detection(
            interval=args.interval,
            max_detections=args.max_detections,
            verbose=args.verbose
        )
    elif args.mode == 'file':
        if args.file is None:
            print("Error: --file argument required for file mode")
            sys.exit(1)
        if not Path(args.file).exists():
            print(f"Error: Audio file not found: {args.file}")
            sys.exit(1)
        detector.run_file_detection(args.file)


if __name__ == "__main__":
    main()
