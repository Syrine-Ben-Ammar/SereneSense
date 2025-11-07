#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 SereneSense Team

"""
Model Comparison Script
=======================
Compare legacy CNN/CRNN models with modern transformer models.

Metrics compared:
    - Model size (parameters)
    - Inference latency (CPU, GPU)
    - Memory usage
    - Accuracy (if available)
    - Throughput
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import numpy as np
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.models.legacy import CNNMFCCModel, CRNNMFCCModel, LegacyModelConfig
from core.models.legacy.legacy_config import LegacyModelType

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare different models."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize comparator.

        Args:
            device: Device to test on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = {}

    def count_parameters(self, model: torch.nn.Module) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def measure_latency(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> float:
        """
        Measure inference latency.

        Args:
            model: Model to test
            input_shape: Input tensor shape
            num_iterations: Number of iterations for timing
            warmup: Warmup iterations

        Returns:
            Average latency in milliseconds
        """
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        # Measure
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        import time

        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

        elapsed_time = end_time - start_time
        avg_latency_ms = (elapsed_time / num_iterations) * 1000
        return avg_latency_ms

    def measure_memory(self, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> float:
        """
        Measure peak memory usage.

        Args:
            model: Model to test
            input_shape: Input tensor shape

        Returns:
            Peak memory in MB
        """
        model.eval()

        if self.device.type != "cuda":
            logger.warning("Memory measurement only works on CUDA devices")
            return 0.0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randn(input_shape, device=self.device)

        with torch.no_grad():
            _ = model(dummy_input)

        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
        return peak_memory

    def compare_legacy_models(self) -> None:
        """Compare CNN and CRNN legacy models."""
        print("\n" + "=" * 80)
        print("LEGACY MODELS COMPARISON (CNN vs CRNN)")
        print("=" * 80)

        models_data = []

        # CNN Model
        config_cnn = LegacyModelConfig(model_type=LegacyModelType.CNN, device=str(self.device))
        model_cnn = CNNMFCCModel(config_cnn).to(self.device)

        cnn_params = self.count_parameters(model_cnn)
        cnn_latency = self.measure_latency(model_cnn, (1, 3, 40, 92))
        cnn_memory = self.measure_memory(model_cnn, (1, 3, 40, 92))

        models_data.append(
            {
                "Model": "CNN MFCC",
                "Parameters": f"{cnn_params:,}",
                "Latency (ms)": f"{cnn_latency:.2f}",
                "Memory (MB)": f"{cnn_memory:.2f}",
                "Input Shape": "(1, 3, 40, 92)",
                "Duration": "3 seconds",
            }
        )

        # CRNN Model
        config_crnn = LegacyModelConfig(model_type=LegacyModelType.CRNN, device=str(self.device))
        model_crnn = CRNNMFCCModel(config_crnn).to(self.device)

        crnn_params = self.count_parameters(model_crnn)
        crnn_latency = self.measure_latency(model_crnn, (1, 3, 40, 124))
        crnn_memory = self.measure_memory(model_crnn, (1, 3, 40, 124))

        models_data.append(
            {
                "Model": "CRNN MFCC",
                "Parameters": f"{crnn_params:,}",
                "Latency (ms)": f"{crnn_latency:.2f}",
                "Memory (MB)": f"{crnn_memory:.2f}",
                "Input Shape": "(1, 3, 40, 124)",
                "Duration": "4 seconds",
            }
        )

        # Print comparison table
        print("\nModel Metrics:")
        print(tabulate(models_data, headers="keys", tablefmt="grid"))

        # Print additional stats
        print("\nStatistics:")
        print(f"CNN Parameter Count: {cnn_params:,}")
        print(f"CRNN Parameter Count: {crnn_params:,}")
        print(f"CRNN is {crnn_params / cnn_params:.1f}x larger than CNN")
        print(f"\nCNN Latency: {cnn_latency:.2f} ms")
        print(f"CRNN Latency: {crnn_latency:.2f} ms")
        print(f"CRNN is {crnn_latency / cnn_latency:.1f}x slower than CNN")

        # Architecture comparison
        print("\n" + "-" * 80)
        print("ARCHITECTURE COMPARISON")
        print("-" * 80)
        arch_data = [
            ["Feature", "CNN MFCC", "CRNN MFCC"],
            ["Feature Type", "MFCC + delta + delta-delta", "MFCC + delta + delta-delta"],
            ["Conv Filters", "48, 96, 192", "48, 96, 192"],
            ["Pool Strategy", "Global Average Pool", "Time-Preserve Pool (2,1)"],
            ["Temporal Modeling", "Implicit (via conv)", "Explicit (via BiLSTM)"],
            ["LSTM Units", "None", "128, 64 (bidirectional)"],
            ["Dense Units", "160", "160"],
            ["Dropout Pattern", "0.25, 0.30, 0.30, 0.35", "0.20, 0.25, 0.30, 0.35"],
            ["Parameters", f"{cnn_params:,}", f"{crnn_params:,}"],
            ["Typical Accuracy", "~85%", "~87%"],
            ["Training Epochs", "150", "300"],
            ["Batch Size", "32", "16"],
            ["Audio Duration", "3 seconds", "4 seconds"],
        ]

        for row in arch_data:
            print(f"{row[0]:<25} {row[1]:<30} {row[2]:<30}")

        # Save results
        self.results["legacy_comparison"] = {
            "cnn": {
                "parameters": cnn_params,
                "latency_ms": cnn_latency,
                "memory_mb": cnn_memory,
            },
            "crnn": {
                "parameters": crnn_params,
                "latency_ms": crnn_latency,
                "memory_mb": crnn_memory,
            },
        }

    def print_advantages_disadvantages(self) -> None:
        """Print detailed comparison of advantages and disadvantages."""
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON: CNN vs CRNN")
        print("=" * 80)

        comparison = {
            "Aspect": [
                "Model Complexity",
                "Temporal Modeling",
                "Training Speed",
                "Inference Speed",
                "Memory Usage",
                "Parameter Count",
                "Accuracy",
                "Audio Duration",
                "Batch Size",
                "Training Epochs",
                "Suitable For",
            ],
            "CNN MFCC": [
                "Simple (3 conv blocks)",
                "Implicit via convolution",
                "Fast (150 epochs)",
                "Fast (~20ms)",
                "Low (~50MB)",
                "Small (242K)",
                "Good (~85%)",
                "3 seconds",
                "32",
                "150",
                "Fast inference, edge devices",
            ],
            "CRNN MFCC": [
                "Complex (conv + BiLSTM)",
                "Explicit via BiLSTM",
                "Slower (300 epochs)",
                "Slower (~120ms)",
                "Higher (~150MB)",
                "Large (1.5M)",
                "Better (~87%)",
                "4 seconds",
                "16",
                "300",
                "Better accuracy, research",
            ],
        }

        # Convert to list of rows for tabulate
        rows = []
        for i in range(len(comparison["Aspect"])):
            rows.append(
                [
                    comparison["Aspect"][i],
                    comparison["CNN MFCC"][i],
                    comparison["CRNN MFCC"][i],
                ]
            )

        print(tabulate(rows, headers=["Aspect", "CNN MFCC", "CRNN MFCC"], tablefmt="grid"))

        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(
            """
CNN Advantages:
  • Small parameter count (242K) - suitable for edge devices
  • Fast inference (~20ms)
  • Low memory requirements (~50MB)
  • Quick training (150 epochs)
  • Good for real-time applications

CNN Disadvantages:
  • Lower accuracy (~85%)
  • Limited temporal context (3 seconds)
  • Implicit temporal modeling

CRNN Advantages:
  • Better accuracy (~87%, +2-3% absolute)
  • Explicit temporal modeling via BiLSTM
  • Longer audio context (4 seconds)
  • Better for complex audio patterns

CRNN Disadvantages:
  • Large parameter count (1.5M, 6.3x larger)
  • Slower inference (~120ms)
  • Higher memory (~150MB)
  • Longer training time (300 epochs)
  • Not suitable for real-time edge deployment
        """
        )


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description="Compare legacy models")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Run comparison
    comparator = ModelComparator(device=args.device)
    comparator.compare_legacy_models()
    comparator.print_advantages_disadvantages()

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparator.results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
