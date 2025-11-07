#!/usr/bin/env python3
#
# Plan:
# 1. Create comprehensive model evaluation script for SereneSense
# 2. Support for multiple evaluation metrics and benchmarks
# 3. Test on multiple datasets (MAD, AudioSet, custom test sets)
# 4. Performance profiling (latency, throughput, memory usage)
# 5. Generate detailed evaluation reports with visualizations
# 6. Compare multiple models and export results
# 7. Support for edge device evaluation and optimization validation
#

"""
SereneSense Model Evaluation Script
Comprehensive evaluation and benchmarking of military vehicle detection models.

Usage:
    python scripts/evaluate_model.py --model models/audioMAE_best.pth --dataset mad
    python scripts/evaluate_model.py --model models/ --compare-models --output-dir results/
    python scripts/evaluate_model.py --model models/optimized/ --edge-evaluation --device jetson
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
import json
import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models.audioMAE.model import AudioMAE
from core.models.AST.model import ASTModel
from core.models.BEATs.model import BEATsModel
from core.data.loaders.mad_loader import MADDataset
from core.data.loaders.audioset_loader import AudioSetLoader
from core.core.audio_processor import AudioProcessor
from core.utils.config_parser import ConfigParser
from core.utils.device_utils import get_optimal_device
from core.utils.logging import setup_logging
from core.inference.batch import BatchInference, BatchConfig
from core.inference.real_time import RealTimeInference, InferenceConfig

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate SereneSense military vehicle detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model file or directory containing models"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["audioMAE", "ast", "beats", "auto"],
        default="auto",
        help="Model architecture type (auto-detect if not specified)",
    )
    parser.add_argument("--config", type=str, help="Path to model configuration file")

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mad", "audioset", "custom"],
        default="mad",
        help="Dataset to evaluate on",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory for datasets")
    parser.add_argument(
        "--test-split", type=str, default="test", help="Dataset split to use for evaluation"
    )
    parser.add_argument("--custom-data", type=str, help="Path to custom test dataset directory")

    # Evaluation configuration
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )

    # Performance evaluation
    parser.add_argument(
        "--benchmark-performance",
        action="store_true",
        help="Run performance benchmarking (latency, throughput)",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
        help="Number of iterations for performance benchmarking",
    )
    parser.add_argument(
        "--memory-profiling", action="store_true", help="Enable memory usage profiling"
    )

    # Edge evaluation
    parser.add_argument(
        "--edge-evaluation",
        action="store_true",
        help="Evaluate optimized models for edge deployment",
    )
    parser.add_argument(
        "--target-platform",
        type=str,
        choices=["jetson", "raspberry_pi", "cpu", "gpu"],
        default="cpu",
        help="Target platform for edge evaluation",
    )

    # Comparison and analysis
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare multiple models (if model path is directory)",
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate comprehensive evaluation report"
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save model predictions for analysis"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results", help="Output directory for results"
    )
    parser.add_argument("--experiment-name", type=str, help="Name for this evaluation experiment")

    # Device configuration
    parser.add_argument("--device", type=str, default="auto", help="Device to use for evaluation")

    # Logging and debugging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


class ModelEvaluator:
    """
    Comprehensive model evaluator for SereneSense models.
    Handles evaluation metrics, performance benchmarking, and report generation.
    """

    def __init__(self, device: str = "auto", output_dir: str = "evaluation_results"):
        """
        Initialize model evaluator.

        Args:
            device: Device to use for evaluation
            output_dir: Output directory for results
        """
        self.device = get_optimal_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Class names for military vehicle detection
        self.class_names = [
            "helicopter",
            "fighter_aircraft",
            "military_vehicle",
            "truck",
            "footsteps",
            "speech",
            "background",
        ]

        logger.info(f"Model evaluator initialized on {self.device}")

    def load_model(self, model_path: str, model_type: str = "auto") -> nn.Module:
        """
        Load model from file.

        Args:
            model_path: Path to model file
            model_type: Model architecture type

        Returns:
            Loaded PyTorch model
        """
        logger.info(f"Loading model from {model_path}")

        # Auto-detect model type from filename if needed
        if model_type == "auto":
            model_path_lower = str(model_path).lower()
            if "audioMAE" in model_path_lower:
                model_type = "audioMAE"
            elif "ast" in model_path_lower:
                model_type = "ast"
            elif "beats" in model_path_lower:
                model_type = "beats"
            else:
                logger.warning("Could not auto-detect model type, defaulting to AudioMAE")
                model_type = "audioMAE"

        # Create model architecture
        num_classes = len(self.class_names)

        if model_type == "audioMAE":
            model = AudioMAE(num_classes=num_classes)
        elif model_type == "ast":
            model = ASTModel(num_classes=num_classes)
        elif model_type == "beats":
            model = BEATsModel(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights
        if str(model_path).endswith(".pth") or str(model_path).endswith(".pt"):
            # PyTorch model
            state_dict = torch.load(model_path, map_location=self.device)

            # Handle different state dict formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            model.load_state_dict(state_dict, strict=False)

        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        model = model.to(self.device)
        model.eval()

        logger.info(f"Model loaded: {model_type} with {num_classes} classes")
        return model

    def create_dataset(
        self, dataset_type: str, data_dir: str, split: str = "test", custom_data: str = None
    ):
        """
        Create evaluation dataset.

        Args:
            dataset_type: Type of dataset
            data_dir: Data directory
            split: Dataset split
            custom_data: Custom dataset path

        Returns:
            Dataset object
        """
        logger.info(f"Creating {dataset_type} dataset (split: {split})")

        if dataset_type == "mad":
            dataset = MADDataset(
                data_dir=data_dir, split=split, transform=None  # No augmentation for evaluation
            )

        elif dataset_type == "audioset":
            loader = AudioSetLoader(data_dir)
            dataset = loader.get_dataset(split=split)

        elif dataset_type == "custom":
            if not custom_data:
                raise ValueError("Custom data path required for custom dataset")

            # Create custom dataset from directory
            from core.data.loaders.mad_loader import MADDataset

            dataset = MADDataset(
                data_dir=custom_data, split="all", transform=None  # Use all files in directory
            )

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        logger.info(f"Dataset created with {len(dataset)} samples")
        return dataset

    def evaluate_accuracy(
        self, model: nn.Module, dataset, batch_size: int = 32, num_workers: int = 4
    ) -> dict:
        """
        Evaluate model accuracy on dataset.

        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model accuracy...")

        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        model.eval()
        all_predictions = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = model(data)
                probabilities = torch.softmax(outputs, dim=1)

                # Get predictions and confidences
                confidences, predictions = torch.max(probabilities, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        confidences = np.array(all_confidences)

        # Calculate metrics
        accuracy = np.mean(predictions == targets)

        # Per-class metrics
        report = classification_report(
            targets, predictions, target_names=self.class_names, output_dict=True, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(targets, predictions)

        metrics = {
            "accuracy": accuracy,
            "num_samples": len(targets),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": predictions.tolist(),
            "targets": targets.tolist(),
            "confidences": confidences.tolist(),
        }

        logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}")
        return metrics

    def benchmark_performance(
        self, model: nn.Module, input_shape: tuple = (1, 1, 128, 128), iterations: int = 100
    ) -> dict:
        """
        Benchmark model performance (latency, throughput, memory).

        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            iterations: Number of benchmark iterations

        Returns:
            Performance metrics dictionary
        """
        logger.info(f"Benchmarking model performance ({iterations} iterations)...")

        model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        logger.info("Warming up...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark inference time
        logger.info("Benchmarking inference time...")
        inference_times = []

        with torch.no_grad():
            for i in range(iterations):
                start_time = time.time()
                _ = model(dummy_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms

                if (i + 1) % 20 == 0:
                    logger.info(f"Completed {i + 1}/{iterations} iterations")

        # Calculate statistics
        inference_times = np.array(inference_times)

        # Memory usage
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "gpu_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            }

        performance_metrics = {
            "latency_ms": {
                "mean": float(np.mean(inference_times)),
                "std": float(np.std(inference_times)),
                "min": float(np.min(inference_times)),
                "max": float(np.max(inference_times)),
                "p50": float(np.percentile(inference_times, 50)),
                "p95": float(np.percentile(inference_times, 95)),
                "p99": float(np.percentile(inference_times, 99)),
            },
            "throughput_fps": 1000.0 / np.mean(inference_times),
            "batch_size": input_shape[0],
            "iterations": iterations,
            "memory_stats": memory_stats,
        }

        logger.info(f"Performance benchmark completed:")
        logger.info(f"  Mean latency: {performance_metrics['latency_ms']['mean']:.2f}ms")
        logger.info(f"  Throughput: {performance_metrics['throughput_fps']:.1f} FPS")

        return performance_metrics

    def generate_visualizations(self, metrics: dict, model_name: str):
        """
        Generate evaluation visualizations.

        Args:
            metrics: Evaluation metrics
            model_name: Name of the model
        """
        logger.info("Generating visualizations...")

        # Set up matplotlib
        plt.style.use(
            "seaborn-v0_8"
            if hasattr(plt.style, "available") and "seaborn-v0_8" in plt.style.available
            else "default"
        )

        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(metrics["confusion_matrix"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )

        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        confusion_matrix_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Per-class performance
        report = metrics["classification_report"]
        classes = [cls for cls in self.class_names if cls in report]

        precision_scores = [report[cls]["precision"] for cls in classes]
        recall_scores = [report[cls]["recall"] for cls in classes]
        f1_scores = [report[cls]["f1-score"] for cls in classes]

        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision_scores, width, label="Precision", alpha=0.8)
        plt.bar(x, recall_scores, width, label="Recall", alpha=0.8)
        plt.bar(x + width, f1_scores, width, label="F1-Score", alpha=0.8)

        plt.xlabel("Classes")
        plt.ylabel("Score")
        plt.title(f"Per-Class Performance - {model_name}")
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.tight_layout()

        performance_path = self.output_dir / f"{model_name}_per_class_performance.png"
        plt.savefig(performance_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualizations saved to {self.output_dir}")

    def save_evaluation_report(self, results: dict, experiment_name: str):
        """
        Save comprehensive evaluation report.

        Args:
            results: Evaluation results
            experiment_name: Name of the experiment
        """
        logger.info("Saving evaluation report...")

        # Create report
        report = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "evaluation_results": results,
            "summary": {
                "total_models_evaluated": len(results),
                "best_accuracy": max(
                    [
                        r["accuracy_metrics"]["accuracy"]
                        for r in results.values()
                        if "accuracy_metrics" in r
                    ]
                ),
                "device_used": str(self.device),
            },
        }

        # Save JSON report
        json_path = self.output_dir / f"{experiment_name}_evaluation_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save CSV summary for multiple models
        if len(results) > 1:
            summary_data = []

            for model_name, model_results in results.items():
                if "accuracy_metrics" in model_results:
                    acc_metrics = model_results["accuracy_metrics"]
                    perf_metrics = model_results.get("performance_metrics", {})

                    row = {
                        "model_name": model_name,
                        "accuracy": acc_metrics["accuracy"],
                        "num_samples": acc_metrics["num_samples"],
                    }

                    # Add per-class F1 scores
                    if "classification_report" in acc_metrics:
                        for class_name in self.class_names:
                            if class_name in acc_metrics["classification_report"]:
                                row[f"{class_name}_f1"] = acc_metrics["classification_report"][
                                    class_name
                                ]["f1-score"]

                    # Add performance metrics
                    if "latency_ms" in perf_metrics:
                        row["mean_latency_ms"] = perf_metrics["latency_ms"]["mean"]
                        row["throughput_fps"] = perf_metrics["throughput_fps"]

                    summary_data.append(row)

            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_path = self.output_dir / f"{experiment_name}_model_comparison.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Model comparison saved to {csv_path}")

        logger.info(f"Evaluation report saved to {json_path}")


def evaluate_single_model(evaluator: ModelEvaluator, model_path: str, args) -> dict:
    """
    Evaluate a single model.

    Args:
        evaluator: Model evaluator instance
        model_path: Path to model file
        args: Command line arguments

    Returns:
        Evaluation results dictionary
    """
    model_name = Path(model_path).stem
    logger.info(f"Evaluating model: {model_name}")

    results = {}

    try:
        # Load model
        model = evaluator.load_model(model_path, args.model_type)

        # Create dataset
        dataset = evaluator.create_dataset(
            args.dataset, args.data_dir, args.test_split, args.custom_data
        )

        # Evaluate accuracy
        accuracy_metrics = evaluator.evaluate_accuracy(
            model, dataset, args.batch_size, args.num_workers
        )
        results["accuracy_metrics"] = accuracy_metrics

        # Performance benchmarking
        if args.benchmark_performance:
            performance_metrics = evaluator.benchmark_performance(
                model, iterations=args.benchmark_iterations
            )
            results["performance_metrics"] = performance_metrics

        # Generate visualizations
        if args.generate_report:
            evaluator.generate_visualizations(accuracy_metrics, model_name)

        logger.info(f"Model {model_name} evaluation completed")

    except Exception as e:
        logger.error(f"Failed to evaluate model {model_name}: {e}")
        results["error"] = str(e)

    return results


def evaluate_multiple_models(evaluator: ModelEvaluator, model_dir: str, args) -> dict:
    """
    Evaluate multiple models in a directory.

    Args:
        evaluator: Model evaluator instance
        model_dir: Directory containing model files
        args: Command line arguments

    Returns:
        Combined evaluation results
    """
    model_dir = Path(model_dir)
    model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))

    if not model_files:
        raise ValueError(f"No model files found in {model_dir}")

    logger.info(f"Found {len(model_files)} models to evaluate")

    results = {}

    for model_file in model_files:
        model_results = evaluate_single_model(evaluator, str(model_file), args)
        results[model_file.stem] = model_results

    return results


def main():
    """Main evaluation function"""
    args = parse_arguments()

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level, debug=args.debug)

    logger.info("🔍 Starting SereneSense model evaluation")
    logger.info(f"Arguments: {vars(args)}")

    # Create experiment name
    experiment_name = (
        args.experiment_name or f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Create evaluator
    evaluator = ModelEvaluator(args.device, args.output_dir)

    try:
        # Determine if single model or multiple models
        model_path = Path(args.model)

        if model_path.is_file():
            # Single model evaluation
            logger.info(f"Evaluating single model: {model_path}")
            results = {model_path.stem: evaluate_single_model(evaluator, str(model_path), args)}

        elif model_path.is_dir() and args.compare_models:
            # Multiple model evaluation
            logger.info(f"Evaluating multiple models in: {model_path}")
            results = evaluate_multiple_models(evaluator, str(model_path), args)

        else:
            raise ValueError(f"Invalid model path or missing --compare-models flag: {model_path}")

        # Save evaluation report
        if args.generate_report:
            evaluator.save_evaluation_report(results, experiment_name)

        # Print summary
        logger.info("📊 Evaluation Summary:")
        for model_name, model_results in results.items():
            if "error" in model_results:
                logger.error(f"  {model_name}: FAILED - {model_results['error']}")
            else:
                if "accuracy_metrics" in model_results:
                    accuracy = model_results["accuracy_metrics"]["accuracy"]
                    logger.info(f"  {model_name}: Accuracy = {accuracy:.4f}")

                if "performance_metrics" in model_results:
                    latency = model_results["performance_metrics"]["latency_ms"]["mean"]
                    throughput = model_results["performance_metrics"]["throughput_fps"]
                    logger.info(f"    Performance: {latency:.2f}ms latency, {throughput:.1f} FPS")

        logger.info("✅ Evaluation completed successfully!")
        logger.info(f"Results saved to: {evaluator.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
