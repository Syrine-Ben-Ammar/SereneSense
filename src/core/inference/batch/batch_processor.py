#
# Plan:
# 1. Create batch inference system for processing large audio datasets
# 2. Support multiple audio formats and directory processing
# 3. Implement efficient batching with configurable batch sizes
# 4. Progress tracking and result aggregation
# 5. Support for distributed processing across multiple GPUs
# 6. Export results in multiple formats (JSON, CSV, TXT)
# 7. Integration with optimized models (TensorRT, ONNX)
#

"""
Batch Military Vehicle Detection Inference
High-throughput processing for large audio datasets.

Features:
- Multi-format audio support (WAV, MP3, FLAC, M4A)
- Configurable batch processing
- Progress tracking and logging
- Result export (JSON, CSV, TXT)
- Multi-GPU support
- Memory-efficient processing
"""

import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import json
import csv
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from collections import defaultdict

from core.core.audio_processor import AudioProcessor
from core.core.model_manager import ModelManager
from core.utils.device_utils import get_optimal_device, get_available_gpus
from core.utils.config_parser import ConfigParser
from core.inference.real_time import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Batch inference configuration"""

    # Model settings
    model_path: str = "models/serenesense_best.pth"
    optimization: str = "none"  # none, tensorrt, onnx
    precision: str = "fp32"  # fp32, fp16, int8

    # Batch processing
    batch_size: int = 32
    num_workers: int = 4
    device: str = "auto"  # auto, cuda, cpu
    multi_gpu: bool = True

    # Audio settings
    sample_rate: int = 16000
    window_length: float = 2.0
    overlap: float = 0.5
    confidence_threshold: float = 0.7

    # Output settings
    output_dir: str = "results"
    save_format: List[str] = None  # json, csv, txt
    save_raw_logits: bool = False
    save_spectrograms: bool = False

    # Processing settings
    max_duration: Optional[float] = None  # Maximum audio duration to process
    chunk_processing: bool = True  # Process long files in chunks

    def __post_init__(self):
        if self.save_format is None:
            self.save_format = ["json", "csv"]


@dataclass
class AudioFileInfo:
    """Audio file information"""

    filepath: Path
    filename: str
    duration: float
    sample_rate: int
    channels: int
    file_size: int


@dataclass
class BatchResult:
    """Batch processing result"""

    file_info: AudioFileInfo
    detections: List[DetectionResult]
    processing_time: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_info": asdict(self.file_info),
            "detections": [det.to_dict() for det in self.detections],
            "processing_time": self.processing_time,
            "error": self.error,
            "num_detections": len(self.detections),
            "detection_summary": self._get_detection_summary(),
        }

    def _get_detection_summary(self) -> Dict[str, int]:
        """Get summary of detections by class"""
        summary = defaultdict(int)
        for detection in self.detections:
            summary[detection.label] += 1
        return dict(summary)


class AudioLoader:
    """Efficient audio loading with caching and preprocessing"""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

    def __init__(self, config: BatchConfig):
        """
        Initialize audio loader.

        Args:
            config: Batch configuration
        """
        self.config = config
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

    def get_audio_files(self, input_path: Union[str, Path]) -> List[Path]:
        """
        Get list of audio files to process.

        Args:
            input_path: Input file or directory path

        Returns:
            List of audio file paths
        """
        input_path = Path(input_path)

        if input_path.is_file():
            if input_path.suffix.lower() in self.SUPPORTED_FORMATS:
                return [input_path]
            else:
                raise ValueError(f"Unsupported audio format: {input_path.suffix}")

        elif input_path.is_dir():
            audio_files = []
            for ext in self.SUPPORTED_FORMATS:
                audio_files.extend(input_path.rglob(f"*{ext}"))

            if not audio_files:
                raise ValueError(f"No audio files found in {input_path}")

            return sorted(audio_files)

        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def get_file_info(self, filepath: Path) -> AudioFileInfo:
        """
        Get audio file information.

        Args:
            filepath: Audio file path

        Returns:
            Audio file information
        """
        try:
            info = torchaudio.info(str(filepath))
            return AudioFileInfo(
                filepath=filepath,
                filename=filepath.name,
                duration=info.num_frames / info.sample_rate,
                sample_rate=info.sample_rate,
                channels=info.num_channels,
                file_size=filepath.stat().st_size,
            )
        except Exception as e:
            logger.error(f"Error getting info for {filepath}: {e}")
            raise

    def load_audio_segments(self, filepath: Path) -> List[Tuple[torch.Tensor, float]]:
        """
        Load audio and split into segments for processing.

        Args:
            filepath: Audio file path

        Returns:
            List of (audio_tensor, timestamp) tuples
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(filepath))

            # Resample if needed
            if sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.config.sample_rate
                )
                waveform = resampler(waveform)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Check duration limit
            duration = waveform.shape[1] / self.config.sample_rate
            if self.config.max_duration and duration > self.config.max_duration:
                max_samples = int(self.config.max_duration * self.config.sample_rate)
                waveform = waveform[:, :max_samples]
                duration = self.config.max_duration

            # Split into segments
            segments = []
            window_samples = int(self.config.window_length * self.config.sample_rate)
            overlap_samples = int(window_samples * self.config.overlap)
            hop_samples = window_samples - overlap_samples

            for start_sample in range(0, waveform.shape[1] - window_samples + 1, hop_samples):
                end_sample = start_sample + window_samples
                segment = waveform[:, start_sample:end_sample]
                timestamp = start_sample / self.config.sample_rate

                segments.append((segment.squeeze(0), timestamp))

            return segments

        except Exception as e:
            logger.error(f"Error loading audio {filepath}: {e}")
            raise


class BatchInference:
    """
    High-throughput batch inference for military vehicle detection.
    Supports multi-GPU processing and various output formats.
    """

    def __init__(self, config: BatchConfig):
        """
        Initialize batch inference system.

        Args:
            config: Batch configuration
        """
        self.config = config

        # Setup devices
        if config.device == "auto":
            self.devices = (
                get_available_gpus() if config.multi_gpu else [get_optimal_device("auto")]
            )
        else:
            self.devices = [config.device]

        logger.info(f"Using devices: {self.devices}")

        # Load models for each device
        self.models = {}
        self.model_manager = ModelManager()

        for device in self.devices:
            model = self.model_manager.load_model(
                config.model_path,
                device=device,
                optimization=config.optimization,
                precision=config.precision,
            )
            model.eval()
            self.models[device] = model

        # Audio loader
        self.audio_loader = AudioLoader(config)

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Class names (should be loaded from model or config)
        self.class_names = [
            "helicopter",
            "fighter_aircraft",
            "military_vehicle",
            "truck",
            "footsteps",
            "speech",
            "background",
        ]

        logger.info(f"BatchInference initialized with {len(self.devices)} device(s)")

    def _process_audio_batch(self, audio_batch: List[torch.Tensor], device: str) -> torch.Tensor:
        """
        Process batch of audio segments.

        Args:
            audio_batch: List of audio tensors
            device: Device to use for processing

        Returns:
            Batch predictions
        """
        try:
            # Convert to spectrograms
            spectrograms = []
            for audio in audio_batch:
                spec = self.audio_loader.audio_processor.to_spectrogram(audio.unsqueeze(0))
                spectrograms.append(spec)

            # Stack into batch
            batch_tensor = torch.stack(spectrograms).to(device)

            # Model inference
            model = self.models[device]
            with torch.no_grad():
                if hasattr(model, "predict"):
                    logits = model.predict(batch_tensor)
                else:
                    logits = model(batch_tensor)

            return logits

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise

    def _process_file(self, filepath: Path, device: str) -> BatchResult:
        """
        Process single audio file.

        Args:
            filepath: Audio file path
            device: Device to use for processing

        Returns:
            Batch processing result
        """
        start_time = time.time()

        try:
            # Get file info
            file_info = self.audio_loader.get_file_info(filepath)

            # Load audio segments
            segments = self.audio_loader.load_audio_segments(filepath)

            if not segments:
                return BatchResult(
                    file_info=file_info,
                    detections=[],
                    processing_time=time.time() - start_time,
                    error="No audio segments extracted",
                )

            # Process in batches
            detections = []
            batch_size = self.config.batch_size

            for i in range(0, len(segments), batch_size):
                batch_segments = segments[i : i + batch_size]
                batch_audio = [seg[0] for seg in batch_segments]
                batch_timestamps = [seg[1] for seg in batch_segments]

                # Process batch
                logits = self._process_audio_batch(batch_audio, device)
                probabilities = torch.softmax(logits, dim=-1)

                # Extract detections
                for j, (probs, timestamp) in enumerate(zip(probabilities, batch_timestamps)):
                    confidence, predicted_class = torch.max(probs, dim=-1)
                    confidence = confidence.item()
                    predicted_class = predicted_class.item()

                    if confidence >= self.config.confidence_threshold:
                        detection = DetectionResult(
                            label=self.class_names[predicted_class],
                            confidence=confidence,
                            class_id=predicted_class,
                            timestamp=timestamp,
                            processing_time=0.0,  # Will be set at file level
                            audio_segment_start=timestamp,
                            audio_segment_end=timestamp + self.config.window_length,
                            raw_logits=logits[j] if self.config.save_raw_logits else None,
                        )
                        detections.append(detection)

            processing_time = time.time() - start_time

            return BatchResult(
                file_info=file_info, detections=detections, processing_time=processing_time
            )

        except Exception as e:
            error_msg = f"Error processing {filepath}: {str(e)}"
            logger.error(error_msg)

            return BatchResult(
                file_info=self.audio_loader.get_file_info(filepath),
                detections=[],
                processing_time=time.time() - start_time,
                error=error_msg,
            )

    def process_files(
        self, input_path: Union[str, Path], output_name: str = "batch_results"
    ) -> List[BatchResult]:
        """
        Process audio files in batch.

        Args:
            input_path: Input file or directory path
            output_name: Output filename prefix

        Returns:
            List of batch results
        """
        logger.info(f"Starting batch processing: {input_path}")

        # Get audio files
        audio_files = self.audio_loader.get_audio_files(input_path)
        logger.info(f"Found {len(audio_files)} audio files")

        # Process files
        results = []

        if len(self.devices) == 1:
            # Single device processing
            device = self.devices[0]
            for filepath in tqdm(audio_files, desc="Processing files"):
                result = self._process_file(filepath, device)
                results.append(result)

        else:
            # Multi-device processing
            with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
                device_cycle = iter(self.devices * (len(audio_files) // len(self.devices) + 1))

                future_to_file = {
                    executor.submit(self._process_file, filepath, next(device_cycle)): filepath
                    for filepath in audio_files
                }

                for future in tqdm(
                    future_to_file, desc="Processing files (multi-GPU)", total=len(audio_files)
                ):
                    result = future.result()
                    results.append(result)

        # Save results
        self._save_results(results, output_name)

        # Print summary
        self._print_summary(results)

        return results

    def _save_results(self, results: List[BatchResult], output_name: str):
        """
        Save results in specified formats.

        Args:
            results: List of batch results
            output_name: Output filename prefix
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        for format_type in self.config.save_format:
            if format_type == "json":
                self._save_json(results, f"{output_name}_{timestamp}.json")
            elif format_type == "csv":
                self._save_csv(results, f"{output_name}_{timestamp}.csv")
            elif format_type == "txt":
                self._save_txt(results, f"{output_name}_{timestamp}.txt")

    def _save_json(self, results: List[BatchResult], filename: str):
        """Save results as JSON"""
        output_path = self.output_dir / filename

        json_data = {
            "metadata": {
                "total_files": len(results),
                "total_detections": sum(len(r.detections) for r in results),
                "processing_config": asdict(self.config),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": [result.to_dict() for result in results],
        }

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    def _save_csv(self, results: List[BatchResult], filename: str):
        """Save results as CSV"""
        output_path = self.output_dir / filename

        rows = []
        for result in results:
            if result.detections:
                for detection in result.detections:
                    rows.append(
                        {
                            "filename": result.file_info.filename,
                            "filepath": str(result.file_info.filepath),
                            "file_duration": result.file_info.duration,
                            "detection_label": detection.label,
                            "confidence": detection.confidence,
                            "timestamp": detection.timestamp,
                            "segment_start": detection.audio_segment_start,
                            "segment_end": detection.audio_segment_end,
                            "processing_time": result.processing_time,
                            "error": result.error,
                        }
                    )
            else:
                # Add row even if no detections
                rows.append(
                    {
                        "filename": result.file_info.filename,
                        "filepath": str(result.file_info.filepath),
                        "file_duration": result.file_info.duration,
                        "detection_label": "None",
                        "confidence": 0.0,
                        "timestamp": 0.0,
                        "segment_start": 0.0,
                        "segment_end": 0.0,
                        "processing_time": result.processing_time,
                        "error": result.error,
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        logger.info(f"Results saved to {output_path}")

    def _save_txt(self, results: List[BatchResult], filename: str):
        """Save results as text report"""
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write("SereneSense Batch Processing Report\n")
            f.write("=" * 40 + "\n\n")

            # Summary
            total_files = len(results)
            total_detections = sum(len(r.detections) for r in results)
            avg_processing_time = np.mean([r.processing_time for r in results])

            f.write(f"Total files processed: {total_files}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Average processing time: {avg_processing_time:.3f}s\n\n")

            # Per-file results
            for result in results:
                f.write(f"File: {result.file_info.filename}\n")
                f.write(f"  Duration: {result.file_info.duration:.2f}s\n")
                f.write(f"  Processing time: {result.processing_time:.3f}s\n")
                f.write(f"  Detections: {len(result.detections)}\n")

                if result.error:
                    f.write(f"  Error: {result.error}\n")

                for detection in result.detections:
                    f.write(
                        f"    {detection.timestamp:.2f}s: {detection.label} "
                        f"({detection.confidence:.3f})\n"
                    )

                f.write("\n")

        logger.info(f"Report saved to {output_path}")

    def _print_summary(self, results: List[BatchResult]):
        """Print processing summary"""
        total_files = len(results)
        successful_files = len([r for r in results if r.error is None])
        total_detections = sum(len(r.detections) for r in results)
        total_duration = sum(r.file_info.duration for r in results)
        total_processing_time = sum(r.processing_time for r in results)

        # Detection breakdown
        detection_counts = defaultdict(int)
        for result in results:
            for detection in result.detections:
                detection_counts[detection.label] += 1

        print("\n" + "=" * 50)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Files processed: {successful_files}/{total_files}")
        print(f"Total audio duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
        print(f"Total processing time: {total_processing_time:.1f}s")
        print(f"Speed factor: {total_duration/total_processing_time:.1f}x real-time")
        print(f"Total detections: {total_detections}")
        print()

        if detection_counts:
            print("Detection breakdown:")
            for label, count in sorted(detection_counts.items()):
                print(f"  {label}: {count}")

        print("=" * 50)


def create_batch_processor(config_path: str = None) -> BatchInference:
    """
    Create batch processor from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured batch inference system
    """
    if config_path:
        config_dict = ConfigParser.load_config(config_path)
        config = BatchConfig(**config_dict.get("batch_inference", {}))
    else:
        config = BatchConfig()

    return BatchInference(config)


if __name__ == "__main__":
    # Demo: Batch processing
    import argparse

    parser = argparse.ArgumentParser(description="SereneSense Batch Inference")
    parser.add_argument("input_path", help="Input audio file or directory")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-name", default="batch_results", help="Output filename prefix")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create processor
    if args.config:
        processor = create_batch_processor(args.config)
    else:
        config = BatchConfig(
            batch_size=args.batch_size, device=args.device, confidence_threshold=args.confidence
        )
        processor = BatchInference(config)

    # Process files
    try:
        print(f"🎧 Starting batch processing: {args.input_path}")
        results = processor.process_files(args.input_path, args.output_name)
        print("✅ Batch processing completed successfully")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Batch processing failed: {e}")
