#!/usr/bin/env python3
"""
SereneSense Performance Benchmarking Script

Plan:
1. Import benchmarking modules from core.evaluation.benchmarks
2. Parse command line arguments for benchmark configuration
3. Support multiple benchmark types: accuracy, latency, throughput, memory usage, power consumption
4. Handle different model formats (PyTorch, ONNX, TensorRT)
5. Support multiple platforms (CPU, GPU, Jetson, Raspberry Pi)
6. Generate comprehensive benchmark reports
7. Export results in multiple formats (JSON, CSV, HTML)
8. Compare performance across different models and configurations
9. Provide statistical analysis and confidence intervals

This script provides comprehensive performance evaluation for SereneSense models.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.utils.logging import setup_logging
from core.utils.config_parser import ConfigParser
from core.utils.device_utils import get_device_info, get_memory_usage, get_power_usage
from core.evaluation.benchmarks.accuracy import AccuracyBenchmark
from core.evaluation.benchmarks.latency import LatencyBenchmark
from core.evaluation.benchmarks.power import PowerBenchmark
from core.evaluation.reports.generator import ReportGenerator
from core.core.model_manager import ModelManager
from core.core.audio_processor import AudioProcessor
from core.data.loaders.mad_loader import MADDataset

logger = logging.getLogger(__name__)


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking suite for SereneSense models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance benchmarker.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.results = {}

        # Initialize benchmarks
        self.accuracy_benchmark = AccuracyBenchmark(config.get("accuracy", {}))
        self.latency_benchmark = LatencyBenchmark(config.get("latency", {}))
        self.power_benchmark = PowerBenchmark(config.get("power", {}))

        # Initialize components
        self.model_manager = ModelManager(config.get("model_manager", {}))
        self.audio_processor = AudioProcessor(config.get("audio_processor", {}))

        logger.info(f"Benchmarker initialized on device: {self.device}")

    def benchmark_model(
        self, model_path: str, test_data_path: str, benchmark_types: List[str]
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmarks on a model.

        Args:
            model_path: Path to model file
            test_data_path: Path to test dataset
            benchmark_types: List of benchmark types to run

        Returns:
            Benchmark results
        """
        logger.info(f"Starting benchmark for model: {model_path}")

        model_results = {
            "model_path": model_path,
            "device": str(self.device),
            "timestamp": time.time(),
            "benchmarks": {},
        }

        try:
            # Load model
            model = self.model_manager.load_model(model_path)
            model = model.to(self.device)
            model.eval()

            # Load test data
            test_dataset = self._load_test_dataset(test_data_path)

            # Run requested benchmarks
            for benchmark_type in benchmark_types:
                logger.info(f"Running {benchmark_type} benchmark...")

                if benchmark_type == "accuracy":
                    results = self._benchmark_accuracy(model, test_dataset)
                elif benchmark_type == "latency":
                    results = self._benchmark_latency(model, test_dataset)
                elif benchmark_type == "throughput":
                    results = self._benchmark_throughput(model, test_dataset)
                elif benchmark_type == "memory":
                    results = self._benchmark_memory_usage(model, test_dataset)
                elif benchmark_type == "power":
                    results = self._benchmark_power_consumption(model, test_dataset)
                elif benchmark_type == "robustness":
                    results = self._benchmark_robustness(model, test_dataset)
                else:
                    logger.warning(f"Unknown benchmark type: {benchmark_type}")
                    continue

                model_results["benchmarks"][benchmark_type] = results
                logger.info(f"{benchmark_type.capitalize()} benchmark completed")

            return model_results

        except Exception as e:
            logger.error(f"Benchmark failed for {model_path}: {e}")
            model_results["error"] = str(e)
            return model_results

    def _load_test_dataset(self, test_data_path: str):
        """Load test dataset for benchmarking."""
        if "mad" in test_data_path.lower():
            dataset = MADDataset(data_dir=test_data_path, split="test", transform=None)
        else:
            # Generic audio dataset loader
            raise NotImplementedError(f"Dataset type not supported: {test_data_path}")

        return dataset

    def _benchmark_accuracy(self, model: torch.nn.Module, test_dataset) -> Dict[str, Any]:
        """Benchmark model accuracy."""
        return self.accuracy_benchmark.run_benchmark(model, test_dataset)

    def _benchmark_latency(self, model: torch.nn.Module, test_dataset) -> Dict[str, Any]:
        """Benchmark model inference latency."""
        results = {}

        # Warmup
        sample_batch = self._get_sample_batch(test_dataset, batch_size=1)
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_batch)

        # Latency measurements
        latencies = []
        num_samples = self.config.get("latency", {}).get("num_samples", 100)

        for i in range(num_samples):
            sample_batch = self._get_sample_batch(test_dataset, batch_size=1)

            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(sample_batch)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Statistical analysis
        latencies = np.array(latencies)
        results = {
            "avg_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "num_samples": num_samples,
        }

        return results

    def _benchmark_throughput(self, model: torch.nn.Module, test_dataset) -> Dict[str, Any]:
        """Benchmark model throughput."""
        results = {}

        batch_sizes = self.config.get("throughput", {}).get("batch_sizes", [1, 4, 8, 16])
        duration_seconds = self.config.get("throughput", {}).get("duration_seconds", 30)

        for batch_size in batch_sizes:
            # Skip if batch size too large for device
            try:
                sample_batch = self._get_sample_batch(test_dataset, batch_size)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Skipping batch size {batch_size}: out of memory")
                    continue
                raise

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(sample_batch)

            # Throughput measurement
            start_time = time.time()
            num_batches = 0

            while (time.time() - start_time) < duration_seconds:
                sample_batch = self._get_sample_batch(test_dataset, batch_size)
                with torch.no_grad():
                    _ = model(sample_batch)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                num_batches += 1

            elapsed_time = time.time() - start_time
            samples_per_second = (num_batches * batch_size) / elapsed_time

            results[f"batch_size_{batch_size}"] = {
                "samples_per_second": float(samples_per_second),
                "batches_processed": num_batches,
                "elapsed_time_seconds": float(elapsed_time),
            }

        return results

    def _benchmark_memory_usage(self, model: torch.nn.Module, test_dataset) -> Dict[str, Any]:
        """Benchmark memory usage."""
        results = {}

        # Model memory
        model_params = sum(p.numel() for p in model.parameters())
        model_size_mb = model_params * 4 / (1024 * 1024)  # Assuming fp32

        results["model_parameters"] = model_params
        results["model_size_mb"] = float(model_size_mb)

        # Peak memory usage during inference
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

            sample_batch = self._get_sample_batch(test_dataset, batch_size=1)
            with torch.no_grad():
                _ = model(sample_batch)

            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)

            results["peak_gpu_memory_mb"] = float(peak_memory_mb)

        # System memory usage
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        results["system_memory_mb"] = float(memory_info.rss / (1024 * 1024))
        results["virtual_memory_mb"] = float(memory_info.vms / (1024 * 1024))

        return results

    def _benchmark_power_consumption(self, model: torch.nn.Module, test_dataset) -> Dict[str, Any]:
        """Benchmark power consumption."""
        return self.power_benchmark.run_benchmark(model, test_dataset)

    def _benchmark_robustness(self, model: torch.nn.Module, test_dataset) -> Dict[str, Any]:
        """Benchmark model robustness to noise and distortions."""
        results = {}

        # Test with different noise levels
        noise_levels = [0.01, 0.05, 0.1, 0.2]

        for noise_level in noise_levels:
            # Add noise to test samples
            noisy_accuracy = self._test_with_noise(model, test_dataset, noise_level)
            results[f"accuracy_noise_{noise_level}"] = noisy_accuracy

        return results

    def _test_with_noise(self, model: torch.nn.Module, test_dataset, noise_level: float) -> float:
        """Test model accuracy with added noise."""
        correct = 0
        total = 0

        # Test on subset of data
        num_samples = min(100, len(test_dataset))

        for i in range(num_samples):
            sample, target = test_dataset[i]

            # Add noise
            noise = torch.randn_like(sample) * noise_level
            noisy_sample = sample + noise

            # Make prediction
            noisy_sample = noisy_sample.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = model(noisy_sample)
                pred = output.argmax(dim=1)

            if pred.item() == target:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def _get_sample_batch(self, test_dataset, batch_size: int) -> torch.Tensor:
        """Get a sample batch from test dataset."""
        indices = np.random.choice(len(test_dataset), size=batch_size, replace=True)

        samples = []
        for idx in indices:
            sample, _ = test_dataset[idx]
            samples.append(sample)

        batch = torch.stack(samples).to(self.device)
        return batch

    def compare_models(
        self, model_paths: List[str], test_data_path: str, benchmark_types: List[str]
    ) -> Dict[str, Any]:
        """
        Compare performance across multiple models.

        Args:
            model_paths: List of model file paths
            test_data_path: Path to test dataset
            benchmark_types: List of benchmark types to run

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(model_paths)} models")

        comparison_results = {"models": [], "comparison": {}, "summary": {}}

        # Benchmark each model
        for model_path in model_paths:
            results = self.benchmark_model(model_path, test_data_path, benchmark_types)
            comparison_results["models"].append(results)

        # Generate comparison analysis
        comparison_results["comparison"] = self._analyze_comparison(
            comparison_results["models"], benchmark_types
        )

        # Generate summary
        comparison_results["summary"] = self._generate_summary(
            comparison_results["models"], benchmark_types
        )

        return comparison_results

    def _analyze_comparison(
        self, model_results: List[Dict[str, Any]], benchmark_types: List[str]
    ) -> Dict[str, Any]:
        """Analyze performance comparison between models."""
        comparison = {}

        for benchmark_type in benchmark_types:
            comparison[benchmark_type] = {}

            # Extract metrics for comparison
            if benchmark_type == "accuracy":
                metrics = ["overall_accuracy", "per_class_accuracy"]
            elif benchmark_type == "latency":
                metrics = ["avg_latency_ms", "p95_latency_ms"]
            elif benchmark_type == "throughput":
                metrics = ["samples_per_second"]
            elif benchmark_type == "memory":
                metrics = ["model_size_mb", "peak_gpu_memory_mb"]
            else:
                continue

            for metric in metrics:
                values = []
                model_names = []

                for result in model_results:
                    if benchmark_type in result.get("benchmarks", {}):
                        benchmark_data = result["benchmarks"][benchmark_type]
                        if metric in benchmark_data:
                            values.append(benchmark_data[metric])
                            model_names.append(Path(result["model_path"]).stem)

                if values:
                    comparison[benchmark_type][metric] = {
                        "values": values,
                        "model_names": model_names,
                        "best_model": (
                            model_names[np.argmax(values)]
                            if metric != "avg_latency_ms"
                            else model_names[np.argmin(values)]
                        ),
                        "worst_model": (
                            model_names[np.argmin(values)]
                            if metric != "avg_latency_ms"
                            else model_names[np.argmax(values)]
                        ),
                    }

        return comparison

    def _generate_summary(
        self, model_results: List[Dict[str, Any]], benchmark_types: List[str]
    ) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            "total_models": len(model_results),
            "benchmark_types": benchmark_types,
            "recommendations": [],
        }

        # Find best performing models for each benchmark
        best_models = {}
        for benchmark_type in benchmark_types:
            best_accuracy = 0
            best_latency = float("inf")
            best_model = None

            for result in model_results:
                if benchmark_type in result.get("benchmarks", {}):
                    benchmark_data = result["benchmarks"][benchmark_type]

                    if benchmark_type == "accuracy":
                        accuracy = benchmark_data.get("overall_accuracy", 0)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = Path(result["model_path"]).stem

                    elif benchmark_type == "latency":
                        latency = benchmark_data.get("avg_latency_ms", float("inf"))
                        if latency < best_latency:
                            best_latency = latency
                            best_model = Path(result["model_path"]).stem

            if best_model:
                best_models[benchmark_type] = best_model

        summary["best_models"] = best_models

        # Generate recommendations
        if "accuracy" in best_models and "latency" in best_models:
            if best_models["accuracy"] == best_models["latency"]:
                summary["recommendations"].append(
                    f"Model {best_models['accuracy']} provides the best balance of accuracy and latency"
                )
            else:
                summary["recommendations"].append(
                    f"For highest accuracy: {best_models['accuracy']}"
                )
                summary["recommendations"].append(f"For lowest latency: {best_models['latency']}")

        return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark SereneSense model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Models and data
    parser.add_argument(
        "--models", type=str, nargs="+", required=True, help="Path(s) to model files to benchmark"
    )

    parser.add_argument("--test-data", type=str, required=True, help="Path to test dataset")

    parser.add_argument("--config", type=str, help="Benchmark configuration file")

    # Benchmark types
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=["accuracy", "latency", "throughput", "memory", "power", "robustness"],
        default=["accuracy", "latency"],
        help="Types of benchmarks to run",
    )

    # Output options
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmarks", help="Output directory for results"
    )

    parser.add_argument(
        "--output-format",
        type=str,
        nargs="+",
        choices=["json", "csv", "html"],
        default=["json"],
        help="Output format(s)",
    )

    parser.add_argument(
        "--report-name", type=str, default="benchmark_report", help="Name for output report"
    )

    # Benchmark parameters
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run benchmarks on",
    )

    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples for latency benchmark"
    )

    parser.add_argument(
        "--duration", type=int, default=30, help="Duration in seconds for throughput benchmark"
    )

    # Comparison options
    parser.add_argument(
        "--compare", action="store_true", help="Generate comparison analysis for multiple models"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def load_benchmark_config(config_path: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    """Load benchmark configuration."""
    config = {}

    # Load from file if provided
    if config_path and Path(config_path).exists():
        parser = ConfigParser()
        config = parser.load_config(config_path)

    # Override with command line arguments
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config.update(
        {
            "device": device,
            "latency": {"num_samples": args.num_samples},
            "throughput": {"duration_seconds": args.duration, "batch_sizes": [1, 4, 8, 16]},
        }
    )

    return config


def main():
    """Main benchmarking function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 80)
    logger.info("SereneSense Performance Benchmarking")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_benchmark_config(args.config, args)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize benchmarker
        benchmarker = PerformanceBenchmarker(config)

        # Log benchmark info
        logger.info(f"Models: {args.models}")
        logger.info(f"Test data: {args.test_data}")
        logger.info(f"Benchmarks: {args.benchmarks}")
        logger.info(f"Device: {config['device']}")

        # Run benchmarks
        if len(args.models) == 1 or not args.compare:
            # Single model benchmarking
            for model_path in args.models:
                logger.info(f"Benchmarking model: {model_path}")

                results = benchmarker.benchmark_model(model_path, args.test_data, args.benchmarks)

                # Save results
                model_name = Path(model_path).stem
                output_file = output_dir / f"{args.report_name}_{model_name}"

                # Save in requested formats
                for output_format in args.output_format:
                    if output_format == "json":
                        with open(f"{output_file}.json", "w") as f:
                            json.dump(results, f, indent=2)
                    elif output_format == "csv":
                        # Convert to DataFrame and save
                        df = pd.json_normalize(results)
                        df.to_csv(f"{output_file}.csv", index=False)

                logger.info(f"Results saved to {output_file}.*")

        else:
            # Multi-model comparison
            logger.info("Running multi-model comparison...")

            comparison_results = benchmarker.compare_models(
                args.models, args.test_data, args.benchmarks
            )

            # Save comparison results
            output_file = output_dir / args.report_name

            # Save in requested formats
            for output_format in args.output_format:
                if output_format == "json":
                    with open(f"{output_file}_comparison.json", "w") as f:
                        json.dump(comparison_results, f, indent=2)
                elif output_format == "csv":
                    # Save summary as CSV
                    summary_df = pd.DataFrame([comparison_results["summary"]])
                    summary_df.to_csv(f"{output_file}_summary.csv", index=False)
                elif output_format == "html":
                    # Generate HTML report
                    report_generator = ReportGenerator()
                    html_report = report_generator.generate_comparison_report(comparison_results)
                    with open(f"{output_file}_comparison.html", "w") as f:
                        f.write(html_report)

            # Print summary
            logger.info("Benchmark Summary:")
            logger.info("-" * 40)
            for recommendation in comparison_results["summary"]["recommendations"]:
                logger.info(f"• {recommendation}")

            logger.info(f"Comparison results saved to {output_file}_comparison.*")

        logger.info("Benchmarking completed successfully!")

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
