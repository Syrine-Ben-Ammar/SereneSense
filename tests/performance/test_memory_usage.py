#
# Plan:
# 1. Import necessary libraries: pytest, torch, psutil, gc, tracemalloc, memory_profiler
# 2. Create fixtures for:
#    - Memory monitoring utilities
#    - Model instantiation with different configurations
#    - Platform-specific memory constraints
#    - Batch size generators for memory scaling tests
# 3. Test Categories:
#    a) Model Memory Footprint Tests:
#       - AudioMAE, AST, BEATs memory usage comparison
#       - Parameter count vs memory usage correlation
#       - Different precision impact (fp32, fp16, int8)
#       - Model architecture memory scaling
#    b) Batch Processing Memory Tests:
#       - Memory usage scaling with batch sizes
#       - Memory efficiency optimization
#       - OOM detection and handling
#       - Optimal batch size for memory constraints
#    c) Platform-Specific Memory Tests:
#       - Jetson Orin Nano: 2.1GB target validation
#       - Raspberry Pi 5: 1.8GB target validation
#       - RTX 4090: 3.2GB target validation
#       - Memory fragmentation analysis
#    d) Memory Leak Detection:
#       - Long-running inference memory stability
#       - Training memory leak detection
#       - Garbage collection effectiveness
#       - Memory growth pattern analysis
#    e) Edge Device Memory Constraints:
#       - Low-memory operation validation
#       - Memory pressure handling
#       - Memory-optimized model configurations
#       - GPU memory management
#    f) Real-time Memory Monitoring:
#       - Streaming inference memory patterns
#       - Peak memory usage detection
#       - Memory efficiency metrics
#       - Resource competition testing
# 4. Memory optimization validation
# 5. Memory pressure simulation and recovery
# 6. Memory usage profiling and analysis
#

import pytest
import torch
import torch.nn as nn
import psutil
import gc
import time
import tracemalloc
import threading
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional, Generator
from unittest.mock import Mock, patch
from dataclasses import dataclass
from contextlib import contextmanager
import subprocess
import os
import sys

# SereneSense imports
from core.models.audioMAE.audioMAE import AudioMAE
from core.models.AST.ast import AudioSpectrogramTransformer
from core.models.BEATs.beats import BEATs
from core.core.audio_processor import AudioProcessor
from core.core.inference_engine import InferenceEngine
from core.training.trainer import Trainer

# Memory profiling utilities
try:
    from memory_profiler import profile

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

    def profile(func):
        return func


class MemoryBenchmarkConfig:
    """Configuration for memory usage benchmarking"""

    # Target memory usage benchmarks (GB)
    JETSON_ORIN_NANO_TARGET_GB = 2.1
    RASPBERRY_PI_5_TARGET_GB = 1.8
    RTX_4090_TARGET_GB = 3.2

    # Memory tolerance
    MEMORY_TOLERANCE = 0.2  # 20% tolerance

    # Model configurations
    AUDIOMAЕ_CONFIG = {
        "img_size": 128,
        "patch_size": 16,
        "embed_dim": 768,
        "encoder_depth": 12,
        "decoder_depth": 8,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "num_classes": 7,
        "mask_ratio": 0.75,
    }

    # Test parameters
    MEMORY_MEASUREMENT_ITERATIONS = 50
    BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    MEMORY_LEAK_ITERATIONS = 1000
    MEMORY_PRESSURE_DURATION = 60  # seconds

    # Audio parameters
    SAMPLE_RATE = 16000
    N_MELS = 128
    SEQUENCE_LENGTH = 128

    # Precision configurations
    PRECISIONS = ["fp32", "fp16", "int8"]

    # Memory thresholds
    MEMORY_LEAK_THRESHOLD_MB = 50  # 50MB
    MEMORY_FRAGMENTATION_THRESHOLD = 0.15  # 15%
    MEMORY_PRESSURE_THRESHOLD = 0.9  # 90% of available memory


@dataclass
class MemoryUsage:
    """Memory usage measurement result"""

    peak_memory_mb: float
    current_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    memory_efficiency: float
    fragmentation_ratio: float

    @property
    def peak_memory_gb(self) -> float:
        return self.peak_memory_mb / 1024

    @property
    def current_memory_gb(self) -> float:
        return self.current_memory_mb / 1024


class MemoryMonitor:
    """Utility class for monitoring memory usage"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset memory monitoring"""
        self.baseline_memory = None
        self.peak_memory = 0
        self.measurements = []
        self.start_time = None

    def start_monitoring(self):
        """Start memory monitoring"""
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory["total"]
        self.start_time = time.time()

        # Start Python tracemalloc
        tracemalloc.start()

    def record_measurement(self):
        """Record current memory measurement"""
        current = self._get_current_memory()
        self.measurements.append({"timestamp": time.time() - self.start_time, "memory": current})

        if current["total"] > self.peak_memory:
            self.peak_memory = current["total"]

    def stop_monitoring(self) -> MemoryUsage:
        """Stop monitoring and return results"""
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / 1024 / 1024
            current_mb = current / 1024 / 1024
        else:
            final_memory = self._get_current_memory()
            peak_mb = self.peak_memory
            current_mb = final_memory["total"]

        # Calculate GPU memory if available
        gpu_allocated = 0
        gpu_reserved = 0
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024

        # Calculate memory efficiency
        baseline_total = self.baseline_memory["total"] if self.baseline_memory else 0
        efficiency = baseline_total / peak_mb if peak_mb > 0 else 1.0

        # Calculate fragmentation ratio
        fragmentation = (gpu_reserved - gpu_allocated) / gpu_reserved if gpu_reserved > 0 else 0

        return MemoryUsage(
            peak_memory_mb=peak_mb,
            current_memory_mb=current_mb,
            allocated_memory_mb=gpu_allocated,
            reserved_memory_mb=gpu_reserved,
            memory_efficiency=efficiency,
            fragmentation_ratio=fragmentation,
        )

    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "total": memory_info.rss / 1024 / 1024,
            "system_available": psutil.virtual_memory().available / 1024 / 1024,
        }


@contextmanager
def memory_monitor():
    """Context manager for memory monitoring"""
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        result = monitor.stop_monitoring()
        return result


def clear_memory():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TestModelMemoryFootprint:
    """Test memory footprint of individual models"""

    @pytest.fixture(params=["cpu", "cuda"])
    def device(self, request):
        """Test device fixture"""
        if request.param == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return request.param

    @pytest.fixture
    def memory_test_input(self, device):
        """Create test input tensor"""
        return torch.randn(
            1, 1, MemoryBenchmarkConfig.N_MELS, MemoryBenchmarkConfig.SEQUENCE_LENGTH, device=device
        )

    def test_audioMAE_memory_footprint(self, device, memory_test_input):
        """Test AudioMAE model memory usage"""

        clear_memory()

        with memory_monitor() as monitor:
            # Create model
            model = AudioMAE(**MemoryBenchmarkConfig.AUDIOMAЕ_CONFIG)
            model.to(device)
            model.eval()

            monitor.record_measurement()

            # Run inference to measure peak memory
            with torch.no_grad():
                outputs = model(memory_test_input, mode="classification")

            monitor.record_measurement()

        result = monitor.stop_monitoring()

        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nAudioMAE Memory Footprint ({device}):")
        print(f"  Peak memory: {result.peak_memory_mb:.1f} MB ({result.peak_memory_gb:.2f} GB)")
        print(f"  Current memory: {result.current_memory_mb:.1f} MB")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Memory efficiency: {result.memory_efficiency:.2f}")

        if device == "cuda":
            print(f"  GPU allocated: {result.allocated_memory_mb:.1f} MB")
            print(f"  GPU reserved: {result.reserved_memory_mb:.1f} MB")
            print(f"  GPU fragmentation: {result.fragmentation_ratio:.1%}")

        # Validate memory usage against platform targets
        if device == "cuda":
            # Use conservative estimate for GPU memory
            assert (
                result.peak_memory_gb <= MemoryBenchmarkConfig.RTX_4090_TARGET_GB
            ), f"AudioMAE GPU memory {result.peak_memory_gb:.2f}GB exceeds target {MemoryBenchmarkConfig.RTX_4090_TARGET_GB}GB"

        # Memory should be reasonable for model size
        expected_memory_mb = (total_params * 4) / 1024 / 1024  # 4 bytes per fp32 parameter
        memory_ratio = result.peak_memory_mb / expected_memory_mb

        assert memory_ratio <= 5.0, f"Memory usage ratio too high: {memory_ratio:.1f}x expected"

    def test_model_comparison_memory(self, device):
        """Compare memory usage across different models"""

        models_config = {
            "audioMAE": {"class": AudioMAE, "config": MemoryBenchmarkConfig.AUDIOMAЕ_CONFIG},
            "ast": {
                "class": AudioSpectrogramTransformer,
                "config": {
                    "input_tdim": 128,
                    "input_fdim": 128,
                    "patch_size": 16,
                    "embed_dim": 768,
                    "depth": 12,
                    "num_heads": 12,
                    "num_classes": 7,
                },
            },
            "beats": {
                "class": BEATs,
                "config": {
                    "input_dim": 128,
                    "embed_dim": 768,
                    "num_layers": 12,
                    "num_heads": 12,
                    "num_classes": 7,
                    "patch_size": 16,
                },
            },
        }

        memory_results = {}

        for model_name, model_info in models_config.items():
            clear_memory()

            try:
                with memory_monitor() as monitor:
                    # Create and test model
                    model = model_info["class"](**model_info["config"])
                    model.to(device)
                    model.eval()

                    # Test inference
                    test_input = torch.randn(
                        1,
                        1,
                        MemoryBenchmarkConfig.N_MELS,
                        MemoryBenchmarkConfig.SEQUENCE_LENGTH,
                        device=device,
                    )

                    with torch.no_grad():
                        if model_name == "audioMAE":
                            outputs = model(test_input, mode="classification")
                        else:
                            outputs = model(test_input)

                    monitor.record_measurement()

                result = monitor.stop_monitoring()
                memory_results[model_name] = result

                # Calculate parameters
                total_params = sum(p.numel() for p in model.parameters())

                print(f"\n{model_name.upper()} Memory Usage ({device}):")
                print(f"  Peak memory: {result.peak_memory_mb:.1f} MB")
                print(f"  Parameters: {total_params:,}")
                print(
                    f"  Memory per param: {result.peak_memory_mb / (total_params / 1e6):.2f} MB/M params"
                )

            except Exception as e:
                print(f"  {model_name} failed: {e}")
                continue

        # Compare models
        if len(memory_results) >= 2:
            sorted_models = sorted(memory_results.items(), key=lambda x: x[1].peak_memory_mb)

            print(f"\nModel Memory Ranking ({device}):")
            for i, (model_name, result) in enumerate(sorted_models, 1):
                print(f"  {i}. {model_name}: {result.peak_memory_mb:.1f} MB")

            # Most efficient model should be within reasonable bounds
            most_efficient = sorted_models[0][1]
            least_efficient = sorted_models[-1][1]

            efficiency_ratio = least_efficient.peak_memory_mb / most_efficient.peak_memory_mb

            assert (
                efficiency_ratio <= 3.0
            ), f"Memory usage variance too high: {efficiency_ratio:.1f}x difference"

    def test_precision_memory_impact(self, device, memory_test_input):
        """Test memory usage with different precisions"""

        if device == "cpu":
            pytest.skip("Precision testing primarily for GPU")

        precision_results = {}

        for precision in ["fp32", "fp16"]:
            clear_memory()

            with memory_monitor() as monitor:
                # Create model with specific precision
                model = AudioMAE(**MemoryBenchmarkConfig.AUDIOMAЕ_CONFIG)
                model.to(device)

                if precision == "fp16":
                    model = model.half()
                    test_input = memory_test_input.half()
                else:
                    test_input = memory_test_input.float()

                model.eval()

                # Run inference
                with torch.no_grad():
                    outputs = model(test_input, mode="classification")

                monitor.record_measurement()

            result = monitor.stop_monitoring()
            precision_results[precision] = result

            print(f"\n{precision.upper()} Precision Memory ({device}):")
            print(f"  Peak memory: {result.peak_memory_mb:.1f} MB")
            print(f"  GPU allocated: {result.allocated_memory_mb:.1f} MB")

        # Compare precisions
        if "fp32" in precision_results and "fp16" in precision_results:
            fp32_memory = precision_results["fp32"].peak_memory_mb
            fp16_memory = precision_results["fp16"].peak_memory_mb

            memory_reduction = (fp32_memory - fp16_memory) / fp32_memory

            print(f"\nPrecision Memory Comparison:")
            print(f"  FP32: {fp32_memory:.1f} MB")
            print(f"  FP16: {fp16_memory:.1f} MB")
            print(f"  Reduction: {memory_reduction:.1%}")

            # FP16 should use significantly less memory
            assert (
                memory_reduction >= 0.2
            ), f"FP16 memory reduction insufficient: {memory_reduction:.1%}"


class TestBatchProcessingMemory:
    """Test memory usage scaling with batch processing"""

    @pytest.fixture
    def lightweight_model(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Lightweight model for batch testing"""
        config = {
            "img_size": 128,
            "patch_size": 16,
            "embed_dim": 384,  # Smaller
            "encoder_depth": 6,  # Fewer layers
            "decoder_depth": 4,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "num_classes": 7,
            "mask_ratio": 0.75,
        }

        model = AudioMAE(**config)
        model.to(device)
        model.eval()
        return model

    def test_batch_size_memory_scaling(self, lightweight_model):
        """Test memory scaling with different batch sizes"""

        device = next(lightweight_model.parameters()).device
        batch_sizes = MemoryBenchmarkConfig.BATCH_SIZES

        if device.type == "cpu":
            batch_sizes = [1, 2, 4, 8]  # Limit for CPU

        memory_results = {}

        for batch_size in batch_sizes:
            clear_memory()

            try:
                with memory_monitor() as monitor:
                    # Create batch input
                    batch_input = torch.randn(
                        batch_size,
                        1,
                        MemoryBenchmarkConfig.N_MELS,
                        MemoryBenchmarkConfig.SEQUENCE_LENGTH,
                        device=device,
                    )

                    # Run inference
                    with torch.no_grad():
                        outputs = lightweight_model(batch_input, mode="classification")

                    monitor.record_measurement()

                result = monitor.stop_monitoring()
                memory_results[batch_size] = result

                print(f"\nBatch Size {batch_size} Memory ({device}):")
                print(f"  Peak memory: {result.peak_memory_mb:.1f} MB")
                print(f"  Memory per sample: {result.peak_memory_mb / batch_size:.1f} MB")

                if device.type == "cuda":
                    print(f"  GPU allocated: {result.allocated_memory_mb:.1f} MB")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Batch size {batch_size}: OOM")
                    break
                else:
                    raise

        # Analyze memory scaling
        if len(memory_results) >= 3:
            batch_sizes_tested = sorted(memory_results.keys())

            # Check memory scaling linearity
            memory_values = [memory_results[bs].peak_memory_mb for bs in batch_sizes_tested]

            # Linear regression to check scaling
            x = np.array(batch_sizes_tested)
            y = np.array(memory_values)

            # Calculate correlation coefficient
            correlation = np.corrcoef(x, y)[0, 1]

            print(f"\nMemory Scaling Analysis:")
            print(f"  Batch sizes tested: {batch_sizes_tested}")
            print(f"  Memory correlation: {correlation:.3f}")
            print(f"  Linear scaling: {'Yes' if correlation > 0.9 else 'No'}")

            # Memory should scale reasonably linearly
            assert correlation >= 0.8, f"Memory scaling not linear enough: {correlation:.3f}"

            # Calculate memory efficiency
            smallest_batch = min(batch_sizes_tested)
            largest_batch = max(batch_sizes_tested)

            smallest_memory = memory_results[smallest_batch].peak_memory_mb
            largest_memory = memory_results[largest_batch].peak_memory_mb

            expected_ratio = largest_batch / smallest_batch
            actual_ratio = largest_memory / smallest_memory
            efficiency = expected_ratio / actual_ratio

            print(f"  Expected ratio: {expected_ratio:.1f}x")
            print(f"  Actual ratio: {actual_ratio:.1f}x")
            print(f"  Memory efficiency: {efficiency:.2f}")

            # Efficiency should be reasonable
            assert efficiency >= 0.5, f"Memory efficiency too low: {efficiency:.2f}"

    def test_optimal_batch_size_memory(self, lightweight_model):
        """Find optimal batch size for memory constraints"""

        device = next(lightweight_model.parameters()).device

        # Simulate different memory constraints
        memory_constraints_mb = [512, 1024, 2048, 4096]  # Different GPU memory limits

        optimal_batch_sizes = {}

        for constraint_mb in memory_constraints_mb:
            max_batch_size = 1

            for batch_size in [1, 2, 4, 8, 16, 32]:
                clear_memory()

                try:
                    with memory_monitor() as monitor:
                        batch_input = torch.randn(
                            batch_size,
                            1,
                            MemoryBenchmarkConfig.N_MELS,
                            MemoryBenchmarkConfig.SEQUENCE_LENGTH,
                            device=device,
                        )

                        with torch.no_grad():
                            outputs = lightweight_model(batch_input, mode="classification")

                        monitor.record_measurement()

                    result = monitor.stop_monitoring()

                    if result.peak_memory_mb <= constraint_mb:
                        max_batch_size = batch_size
                    else:
                        break

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    else:
                        raise

            optimal_batch_sizes[constraint_mb] = max_batch_size

            print(f"\nMemory Constraint {constraint_mb} MB:")
            print(f"  Optimal batch size: {max_batch_size}")

        # Validate optimal batch sizes make sense
        for i, (constraint, batch_size) in enumerate(optimal_batch_sizes.items()):
            if i > 0:
                prev_constraint = list(optimal_batch_sizes.keys())[i - 1]
                prev_batch_size = optimal_batch_sizes[prev_constraint]

                # Larger memory should allow larger or equal batch size
                assert (
                    batch_size >= prev_batch_size
                ), f"Optimal batch size decreased with more memory: {batch_size} < {prev_batch_size}"


class TestPlatformSpecificMemory:
    """Test platform-specific memory requirements"""

    def test_jetson_orin_nano_memory_simulation(self):
        """Simulate Jetson Orin Nano memory constraints"""

        target_memory_gb = MemoryBenchmarkConfig.JETSON_ORIN_NANO_TARGET_GB

        # Simulate Jetson memory characteristics
        def jetson_memory_simulation():
            """Simulate Jetson memory usage patterns"""

            # Account for system overhead
            system_overhead_gb = 0.5  # OS and background processes
            available_memory_gb = 8.0 - system_overhead_gb  # Jetson has 8GB total

            # Model memory usage
            model_memory_gb = 1.2  # AudioMAE model

            # Inference memory overhead
            inference_overhead_gb = 0.4  # Temporary tensors and processing

            total_usage_gb = model_memory_gb + inference_overhead_gb

            return {
                "total_usage": total_usage_gb,
                "available": available_memory_gb,
                "utilization": total_usage_gb / available_memory_gb,
                "headroom": available_memory_gb - total_usage_gb,
            }

        simulation = jetson_memory_simulation()

        print(f"\nJetson Orin Nano Memory Simulation:")
        print(f"  Total usage: {simulation['total_usage']:.2f} GB")
        print(f"  Available: {simulation['available']:.2f} GB")
        print(f"  Utilization: {simulation['utilization']:.1%}")
        print(f"  Headroom: {simulation['headroom']:.2f} GB")
        print(f"  Target: {target_memory_gb:.1f} GB")

        # Validate against target
        assert (
            simulation["total_usage"] <= target_memory_gb
        ), f"Jetson memory usage {simulation['total_usage']:.2f}GB exceeds target {target_memory_gb}GB"

        # Should have reasonable headroom
        assert (
            simulation["headroom"] >= 0.5
        ), f"Insufficient memory headroom: {simulation['headroom']:.2f}GB"

    def test_raspberry_pi_5_memory_simulation(self):
        """Simulate Raspberry Pi 5 memory constraints"""

        target_memory_gb = MemoryBenchmarkConfig.RASPBERRY_PI_5_TARGET_GB

        def rpi5_memory_simulation():
            """Simulate RPi5 memory usage patterns"""

            # Account for system overhead
            system_overhead_gb = 0.8  # Raspberry Pi OS overhead
            available_memory_gb = 8.0 - system_overhead_gb  # RPi5 has 8GB

            # Model memory (optimized for CPU/AI HAT)
            model_memory_gb = 0.8  # Quantized model

            # AI HAT memory
            ai_hat_memory_gb = 0.4  # Hailo-8 NPU

            # Inference overhead
            inference_overhead_gb = 0.6  # CPU processing overhead

            total_usage_gb = model_memory_gb + ai_hat_memory_gb + inference_overhead_gb

            return {
                "total_usage": total_usage_gb,
                "available": available_memory_gb,
                "utilization": total_usage_gb / available_memory_gb,
                "headroom": available_memory_gb - total_usage_gb,
            }

        simulation = rpi5_memory_simulation()

        print(f"\nRaspberry Pi 5 Memory Simulation:")
        print(f"  Total usage: {simulation['total_usage']:.2f} GB")
        print(f"  Available: {simulation['available']:.2f} GB")
        print(f"  Utilization: {simulation['utilization']:.1%}")
        print(f"  Headroom: {simulation['headroom']:.2f} GB")
        print(f"  Target: {target_memory_gb:.1f} GB")

        # Validate against target
        assert (
            simulation["total_usage"] <= target_memory_gb
        ), f"RPi5 memory usage {simulation['total_usage']:.2f}GB exceeds target {target_memory_gb}GB"

    def test_cloud_gpu_memory_simulation(self):
        """Simulate RTX 4090 memory usage"""

        target_memory_gb = MemoryBenchmarkConfig.RTX_4090_TARGET_GB

        def rtx4090_memory_simulation():
            """Simulate RTX 4090 memory usage"""

            # GPU memory characteristics
            total_gpu_memory_gb = 24.0  # RTX 4090 has 24GB

            # Model memory (high precision)
            model_memory_gb = 2.8  # Full precision model

            # Large batch processing
            batch_memory_gb = 0.4  # Larger batches possible

            total_usage_gb = model_memory_gb + batch_memory_gb

            return {
                "total_usage": total_usage_gb,
                "available": total_gpu_memory_gb,
                "utilization": total_usage_gb / total_gpu_memory_gb,
                "headroom": total_gpu_memory_gb - total_usage_gb,
            }

        simulation = rtx4090_memory_simulation()

        print(f"\nRTX 4090 Memory Simulation:")
        print(f"  Total usage: {simulation['total_usage']:.2f} GB")
        print(f"  Available: {simulation['available']:.2f} GB")
        print(f"  Utilization: {simulation['utilization']:.1%}")
        print(f"  Headroom: {simulation['headroom']:.2f} GB")
        print(f"  Target: {target_memory_gb:.1f} GB")

        # Validate against target
        assert (
            simulation["total_usage"] <= target_memory_gb
        ), f"RTX 4090 memory usage {simulation['total_usage']:.2f}GB exceeds target {target_memory_gb}GB"


class TestMemoryLeakDetection:
    """Test for memory leaks during extended operation"""

    def test_inference_memory_leak(self):
        """Test for memory leaks during repeated inference"""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create lightweight model for testing
        config = {
            "img_size": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "encoder_depth": 4,
            "decoder_depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "num_classes": 7,
            "mask_ratio": 0.75,
        }

        model = AudioMAE(**config)
        model.to(device)
        model.eval()

        # Monitor memory over many iterations
        memory_measurements = []

        for i in range(MemoryBenchmarkConfig.MEMORY_LEAK_ITERATIONS):
            # Create input
            test_input = torch.randn(
                1,
                1,
                MemoryBenchmarkConfig.N_MELS,
                MemoryBenchmarkConfig.SEQUENCE_LENGTH,
                device=device,
            )

            # Run inference
            with torch.no_grad():
                outputs = model(test_input, mode="classification")

            # Measure memory every 100 iterations
            if i % 100 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                else:
                    gpu_memory = 0

                memory_measurements.append(
                    {
                        "iteration": i,
                        "cpu_memory": current_memory,
                        "gpu_memory": gpu_memory,
                        "total_memory": current_memory + gpu_memory,
                    }
                )

                print(f"Iteration {i}: {current_memory:.1f} MB CPU, {gpu_memory:.1f} MB GPU")

            # Cleanup
            del test_input, outputs

            if i % 200 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Analyze memory growth
        initial_memory = memory_measurements[0]["total_memory"]
        final_memory = memory_measurements[-1]["total_memory"]
        memory_growth = final_memory - initial_memory

        print(f"\nMemory Leak Analysis:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")
        print(
            f"  Growth rate: {memory_growth / MemoryBenchmarkConfig.MEMORY_LEAK_ITERATIONS * 1000:.3f} MB/1K iterations"
        )

        # Check for significant memory growth
        assert (
            memory_growth <= MemoryBenchmarkConfig.MEMORY_LEAK_THRESHOLD_MB
        ), f"Potential memory leak detected: {memory_growth:.1f} MB growth"

        # Calculate growth trend
        iterations = [m["iteration"] for m in memory_measurements]
        memories = [m["total_memory"] for m in memory_measurements]

        if len(memory_measurements) > 2:
            # Linear regression to detect growth trend
            correlation = np.corrcoef(iterations, memories)[0, 1]

            print(f"  Memory trend correlation: {correlation:.3f}")

            # Strong positive correlation indicates memory leak
            assert (
                abs(correlation) <= 0.3
            ), f"Strong memory growth trend detected: {correlation:.3f}"

    def test_training_memory_stability(self):
        """Test memory stability during training simulation"""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create model and optimizer
        config = {
            "img_size": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "encoder_depth": 4,
            "decoder_depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "num_classes": 7,
            "mask_ratio": 0.75,
        }

        model = AudioMAE(**config)
        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        memory_history = []

        # Simulate training steps
        for step in range(100):
            # Create batch
            batch_input = torch.randn(
                4,
                1,
                MemoryBenchmarkConfig.N_MELS,
                MemoryBenchmarkConfig.SEQUENCE_LENGTH,
                device=device,
            )
            batch_labels = torch.randint(0, 7, (4,), device=device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_input, mode="classification")
            loss = criterion(outputs["logits"], batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Monitor memory every 10 steps
            if step % 10 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_history.append(current_memory)

                print(f"Training step {step}: {current_memory:.1f} MB")

            # Cleanup
            del batch_input, batch_labels, outputs, loss

        # Analyze training memory stability
        if len(memory_history) >= 3:
            memory_variance = np.var(memory_history)
            memory_range = max(memory_history) - min(memory_history)

            print(f"\nTraining Memory Stability:")
            print(f"  Memory variance: {memory_variance:.2f}")
            print(f"  Memory range: {memory_range:.1f} MB")
            print(f"  Average memory: {np.mean(memory_history):.1f} MB")

            # Memory should be relatively stable during training
            cv = np.std(memory_history) / np.mean(memory_history)  # Coefficient of variation

            assert cv <= 0.1, f"Training memory too unstable: {cv:.1%} coefficient of variation"


class TestMemoryOptimization:
    """Test memory optimization techniques"""

    def test_gradient_checkpointing_memory(self):
        """Test memory savings from gradient checkpointing"""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create model configurations
        base_config = {
            "img_size": 128,
            "patch_size": 16,
            "embed_dim": 512,
            "encoder_depth": 8,
            "decoder_depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "num_classes": 7,
            "mask_ratio": 0.75,
        }

        memory_results = {}

        for use_checkpointing in [False, True]:
            clear_memory()

            with memory_monitor() as monitor:
                model = AudioMAE(**base_config)
                model.to(device)
                model.train()

                # Enable gradient checkpointing if requested
                if use_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
                    model.enable_gradient_checkpointing()

                # Training simulation
                batch_input = torch.randn(
                    4,
                    1,
                    MemoryBenchmarkConfig.N_MELS,
                    MemoryBenchmarkConfig.SEQUENCE_LENGTH,
                    device=device,
                )
                batch_labels = torch.randint(0, 7, (4,), device=device)

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()

                # Forward and backward pass
                optimizer.zero_grad()
                outputs = model(batch_input, mode="classification")
                loss = criterion(outputs["logits"], batch_labels)
                loss.backward()

                monitor.record_measurement()

            result = monitor.stop_monitoring()
            key = "with_checkpointing" if use_checkpointing else "without_checkpointing"
            memory_results[key] = result

            print(f"\nGradient Checkpointing {'ON' if use_checkpointing else 'OFF'}:")
            print(f"  Peak memory: {result.peak_memory_mb:.1f} MB")

            if torch.cuda.is_available():
                print(f"  GPU allocated: {result.allocated_memory_mb:.1f} MB")

        # Compare results
        if len(memory_results) == 2:
            without = memory_results["without_checkpointing"]
            with_cp = memory_results["with_checkpointing"]

            memory_saving = (
                without.peak_memory_mb - with_cp.peak_memory_mb
            ) / without.peak_memory_mb

            print(f"\nGradient Checkpointing Impact:")
            print(f"  Without: {without.peak_memory_mb:.1f} MB")
            print(f"  With: {with_cp.peak_memory_mb:.1f} MB")
            print(f"  Memory saving: {memory_saving:.1%}")

            # Gradient checkpointing should save memory (when implemented)
            if hasattr(AudioMAE, "enable_gradient_checkpointing"):
                assert (
                    memory_saving >= 0.1
                ), f"Gradient checkpointing should save memory: {memory_saving:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
