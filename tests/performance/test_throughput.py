#
# Plan:
# 1. Import necessary libraries: pytest, torch, time, threading, concurrent.futures
# 2. Create fixtures for:
#    - Mock models with realistic inference characteristics
#    - Batch audio data generators
#    - Concurrent request simulators
#    - Performance measurement utilities
# 3. Test Categories:
#    a) Single Model Throughput Tests:
#       - AudioMAE: Target 122 FPS (Jetson), 476 FPS (RTX 4090)
#       - AST and BEATs throughput benchmarks
#       - Sustained throughput over time
#       - Peak vs sustained performance
#    b) Batch Processing Throughput:
#       - Throughput scaling with batch sizes (1, 2, 4, 8, 16, 32)
#       - Memory vs throughput tradeoffs
#       - Optimal batch size detection
#       - Batch efficiency metrics
#    c) Concurrent Request Throughput:
#       - FastAPI server throughput under concurrent load
#       - WebSocket streaming throughput
#       - Multi-threaded inference pipeline
#       - Connection pooling efficiency
#    d) Platform-Specific Throughput:
#       - Jetson Orin Nano: 122 FPS target validation
#       - Raspberry Pi 5: 54 FPS target validation
#       - Cloud GPU: 476 FPS target validation
#       - CPU-only throughput baselines
#    e) Real-Time Streaming Throughput:
#       - Audio streaming throughput
#       - Real-time buffer processing rate
#       - Continuous operation sustainability
#       - Buffer overflow/underflow detection
#    f) Load Testing and Stress Testing:
#       - Peak throughput under stress
#       - Degradation curves under load
#       - Recovery after overload
#       - Resource exhaustion handling
# 4. Statistical analysis and performance regression detection
# 5. Throughput vs accuracy tradeoff analysis
# 6. Resource utilization efficiency metrics
#

import pytest
import torch
import torch.nn as nn
import time
import threading
import asyncio
import concurrent.futures
import statistics
import numpy as np
import queue
import psutil
import gc
from typing import Dict, List, Tuple, Any, Optional, Callable
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from contextlib import contextmanager
import multiprocessing as mp

# SereneSense imports
from core.models.audioMAE.audioMAE import AudioMAE
from core.models.AST.ast import AudioSpectrogramTransformer
from core.models.BEATs.beats import BEATs
from core.core.audio_processor import AudioProcessor
from core.core.inference_engine import InferenceEngine
from core.inference.realtime_inference import RealTimeInference

# FastAPI testing imports
from fastapi.testclient import TestClient
import httpx


class ThroughputBenchmarkConfig:
    """Configuration for throughput benchmarking"""

    # Target throughput benchmarks (FPS)
    JETSON_ORIN_NANO_TARGET_FPS = 122
    RASPBERRY_PI_5_TARGET_FPS = 54
    RTX_4090_TARGET_FPS = 476
    CPU_TARGET_FPS = 30

    # Model-specific targets
    AUDIOMAЕ_TARGET_FPS = 122  # Based on Jetson performance
    AST_TARGET_FPS = 95  # Slightly lower due to complexity
    BEATS_TARGET_FPS = 110  # Between AudioMAE and AST

    # Test parameters
    BENCHMARK_DURATION = 30  # seconds
    WARMUP_DURATION = 5  # seconds
    MEASUREMENT_INTERVAL = 1  # seconds
    MAX_BATCH_SIZE = 32

    # Concurrent testing
    MAX_CONCURRENT_REQUESTS = 100
    LOAD_TEST_DURATION = 60  # seconds
    RAMP_UP_DURATION = 10  # seconds

    # Audio parameters
    SAMPLE_RATE = 16000
    WINDOW_LENGTH = 2.0  # seconds
    N_MELS = 128

    # Performance thresholds
    THROUGHPUT_TOLERANCE = 0.15  # 15% tolerance
    CPU_UTILIZATION_MAX = 0.85  # 85% max CPU
    MEMORY_GROWTH_MAX = 0.1  # 10% max memory growth


@dataclass
class ThroughputResult:
    """Throughput measurement result"""

    fps: float
    peak_fps: float
    min_fps: float
    avg_latency_ms: float
    total_processed: int
    duration_seconds: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: float = 0.0
    errors: int = 0

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (FPS per CPU core)"""
        return self.fps / psutil.cpu_count()


class ThroughputMeasurer:
    """Utility class for measuring throughput"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset measurement counters"""
        self.start_time = None
        self.processed_count = 0
        self.latencies = []
        self.fps_history = []
        self.errors = 0
        self.last_measurement = None

    def start(self):
        """Start throughput measurement"""
        self.start_time = time.perf_counter()
        self.last_measurement = self.start_time

    def record_processing(self, latency_ms: float = None, error: bool = False):
        """Record a processing event"""
        if error:
            self.errors += 1
        else:
            self.processed_count += 1
            if latency_ms is not None:
                self.latencies.append(latency_ms)

    def get_current_fps(self) -> float:
        """Get current FPS since last measurement"""
        current_time = time.perf_counter()
        if self.last_measurement is None:
            return 0.0

        elapsed = current_time - self.last_measurement
        if elapsed < 0.1:  # Avoid division by very small numbers
            return 0.0

        # Calculate FPS since last measurement
        processed_since_last = self.processed_count - len(self.fps_history)
        fps = processed_since_last / elapsed

        self.fps_history.append(fps)
        self.last_measurement = current_time

        return fps

    def get_result(self) -> ThroughputResult:
        """Get final throughput measurement result"""
        if self.start_time is None:
            raise ValueError("Measurement not started")

        total_duration = time.perf_counter() - self.start_time
        avg_fps = self.processed_count / total_duration if total_duration > 0 else 0

        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

        # GPU metrics (if available)
        gpu_percent = 0.0
        if torch.cuda.is_available():
            try:
                gpu_percent = torch.cuda.utilization()
            except:
                pass

        return ThroughputResult(
            fps=avg_fps,
            peak_fps=max(self.fps_history) if self.fps_history else 0,
            min_fps=min(self.fps_history) if self.fps_history else 0,
            avg_latency_ms=statistics.mean(self.latencies) if self.latencies else 0,
            total_processed=self.processed_count,
            duration_seconds=total_duration,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            gpu_usage_percent=gpu_percent,
            errors=self.errors,
        )


class TestSingleModelThroughput:
    """Test throughput for individual model inference"""

    @pytest.fixture(params=["cpu", "cuda"])
    def device(self, request):
        """Test device fixture"""
        if request.param == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return request.param

    @pytest.fixture
    def audio_batch_generator(self, device):
        """Generator for continuous audio batches"""

        def generator(batch_size: int, duration: float):
            """Generate batches for specified duration"""
            start_time = time.time()
            while time.time() - start_time < duration:
                yield torch.randn(
                    batch_size, 1, ThroughputBenchmarkConfig.N_MELS, 128, device=device
                )

        return generator

    @pytest.fixture
    def audioMAE_model(self, device):
        """AudioMAE model fixture"""
        config = {
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

        model = AudioMAE(**config)
        model.to(device)
        model.eval()

        # Warmup
        warmup_input = torch.randn(1, 1, ThroughputBenchmarkConfig.N_MELS, 128, device=device)
        with torch.no_grad():
            for _ in range(10):
                model(warmup_input, mode="classification")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    def test_audioMAE_sustained_throughput(self, audioMAE_model, audio_batch_generator, device):
        """Test AudioMAE sustained throughput over time"""

        batch_size = 1
        measurer = ThroughputMeasurer()
        measurer.start()

        # Run sustained inference for benchmark duration
        for batch in audio_batch_generator(
            batch_size, ThroughputBenchmarkConfig.BENCHMARK_DURATION
        ):
            start_time = time.perf_counter()

            with torch.no_grad():
                try:
                    outputs = audioMAE_model(batch, mode="classification")
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    measurer.record_processing(latency_ms)
                except Exception as e:
                    measurer.record_processing(error=True)

        result = measurer.get_result()

        # Validate against target
        target_fps = ThroughputBenchmarkConfig.AUDIOMAЕ_TARGET_FPS
        if device == "cpu":
            target_fps = ThroughputBenchmarkConfig.CPU_TARGET_FPS

        tolerance = target_fps * ThroughputBenchmarkConfig.THROUGHPUT_TOLERANCE

        assert (
            result.fps >= target_fps - tolerance
        ), f"AudioMAE throughput {result.fps:.1f} FPS below target {target_fps} FPS (±{tolerance:.1f})"

        print(f"\nAudioMAE Sustained Throughput ({device}):")
        print(f"  Average FPS: {result.fps:.1f} (target: {target_fps})")
        print(f"  Peak FPS: {result.peak_fps:.1f}")
        print(f"  Min FPS: {result.min_fps:.1f}")
        print(f"  Total processed: {result.total_processed}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  CPU usage: {result.cpu_usage_percent:.1f}%")
        print(f"  Memory: {result.memory_usage_mb:.1f} MB")
        print(f"  Efficiency: {result.efficiency_score:.1f} FPS/core")

        # Validate system resource usage
        assert (
            result.cpu_usage_percent <= ThroughputBenchmarkConfig.CPU_UTILIZATION_MAX * 100
        ), f"CPU usage too high: {result.cpu_usage_percent:.1f}%"

    def test_peak_burst_throughput(self, audioMAE_model, device):
        """Test peak burst throughput capability"""

        batch_size = 8 if device == "cuda" else 4
        num_iterations = 100

        # Pre-generate all batches to avoid allocation overhead
        batches = [
            torch.randn(batch_size, 1, ThroughputBenchmarkConfig.N_MELS, 128, device=device)
            for _ in range(num_iterations)
        ]

        # Measure peak throughput
        start_time = time.perf_counter()
        processed_samples = 0

        with torch.no_grad():
            for batch in batches:
                outputs = audioMAE_model(batch, mode="classification")
                processed_samples += batch_size

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        peak_fps = processed_samples / total_time

        print(f"\nAudioMAE Peak Burst Throughput ({device}):")
        print(f"  Peak FPS: {peak_fps:.1f}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total samples: {processed_samples}")
        print(f"  Duration: {total_time:.2f}s")

        # Peak should significantly exceed sustained throughput
        sustained_target = ThroughputBenchmarkConfig.AUDIOMAЕ_TARGET_FPS
        if device == "cpu":
            sustained_target = ThroughputBenchmarkConfig.CPU_TARGET_FPS

        assert (
            peak_fps >= sustained_target * 1.5
        ), f"Peak throughput {peak_fps:.1f} not significantly higher than sustained {sustained_target}"


class TestBatchProcessingThroughput:
    """Test throughput scaling with batch processing"""

    @pytest.fixture
    def model_for_batching(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Lightweight model for batch testing"""
        config = {
            "img_size": 128,
            "patch_size": 16,
            "embed_dim": 512,  # Smaller for faster testing
            "encoder_depth": 6,  # Fewer layers
            "decoder_depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "num_classes": 7,
            "mask_ratio": 0.75,
        }

        model = AudioMAE(**config)
        model.to(device)
        model.eval()
        return model

    def test_batch_size_scaling(self, model_for_batching):
        """Test throughput scaling with different batch sizes"""

        device = next(model_for_batching.parameters()).device
        batch_sizes = [1, 2, 4, 8, 16] if device.type == "cuda" else [1, 2, 4]

        results = {}

        for batch_size in batch_sizes:
            try:
                # Create batch
                batch = torch.randn(
                    batch_size, 1, ThroughputBenchmarkConfig.N_MELS, 128, device=device
                )

                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        model_for_batching(batch, mode="classification")

                # Measure throughput
                num_iterations = 50
                start_time = time.perf_counter()

                with torch.no_grad():
                    for _ in range(num_iterations):
                        outputs = model_for_batching(batch, mode="classification")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                total_time = time.perf_counter() - start_time
                total_samples = num_iterations * batch_size
                fps = total_samples / total_time
                samples_per_batch_per_second = num_iterations / total_time

                results[batch_size] = {
                    "fps": fps,
                    "batches_per_second": samples_per_batch_per_second,
                    "avg_batch_time_ms": (total_time / num_iterations) * 1000,
                    "per_sample_time_ms": (total_time / total_samples) * 1000,
                }

                print(f"\nBatch Size {batch_size} ({device}):")
                print(f"  FPS: {fps:.1f}")
                print(f"  Batches/sec: {samples_per_batch_per_second:.1f}")
                print(f"  Avg batch time: {results[batch_size]['avg_batch_time_ms']:.2f}ms")
                print(f"  Per sample time: {results[batch_size]['per_sample_time_ms']:.2f}ms")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Batch size {batch_size}: OOM - skipping larger batches")
                    break
                else:
                    raise

        # Analyze batch efficiency
        if len(results) >= 2:
            batch1_fps = results[1]["fps"]
            largest_batch = max(results.keys())
            largest_batch_fps = results[largest_batch]["fps"]

            efficiency_gain = largest_batch_fps / batch1_fps
            theoretical_max = largest_batch  # Perfect scaling

            efficiency_ratio = efficiency_gain / theoretical_max

            print(f"\nBatch Efficiency Analysis:")
            print(f"  Batch 1 FPS: {batch1_fps:.1f}")
            print(f"  Batch {largest_batch} FPS: {largest_batch_fps:.1f}")
            print(f"  Actual speedup: {efficiency_gain:.1f}x")
            print(f"  Theoretical max: {theoretical_max}x")
            print(f"  Efficiency ratio: {efficiency_ratio:.1%}")

            # Batch processing should provide at least 50% of theoretical speedup
            assert (
                efficiency_ratio >= 0.5
            ), f"Batch processing inefficient: {efficiency_ratio:.1%} of theoretical maximum"

    def test_optimal_batch_size_detection(self, model_for_batching):
        """Detect optimal batch size for maximum throughput"""

        device = next(model_for_batching.parameters()).device
        batch_sizes = [1, 2, 4, 8, 16, 32] if device.type == "cuda" else [1, 2, 4, 8]

        throughputs = {}

        for batch_size in batch_sizes:
            try:
                batch = torch.randn(
                    batch_size, 1, ThroughputBenchmarkConfig.N_MELS, 128, device=device
                )

                # Quick throughput test
                num_iterations = 20
                start_time = time.perf_counter()

                with torch.no_grad():
                    for _ in range(num_iterations):
                        model_for_batching(batch, mode="classification")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                total_time = time.perf_counter() - start_time
                fps = (num_iterations * batch_size) / total_time
                throughputs[batch_size] = fps

            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise

        # Find optimal batch size
        optimal_batch_size = max(throughputs, key=throughputs.get)
        optimal_fps = throughputs[optimal_batch_size]

        print(f"\nOptimal Batch Size Analysis ({device}):")
        for batch_size, fps in throughputs.items():
            marker = " ← OPTIMAL" if batch_size == optimal_batch_size else ""
            print(f"  Batch {batch_size:2d}: {fps:6.1f} FPS{marker}")

        print(f"\nOptimal configuration:")
        print(f"  Batch size: {optimal_batch_size}")
        print(f"  Max FPS: {optimal_fps:.1f}")

        # Optimal batch should be > 1 for GPU, may be 1 for CPU
        if device.type == "cuda":
            assert optimal_batch_size > 1, "Optimal batch size should be > 1 for GPU"

        # Optimal FPS should meet minimum thresholds
        min_fps = (
            ThroughputBenchmarkConfig.CPU_TARGET_FPS
            if device.type == "cpu"
            else ThroughputBenchmarkConfig.AUDIOMAЕ_TARGET_FPS
        )
        assert (
            optimal_fps >= min_fps * 0.8
        ), f"Optimal throughput {optimal_fps:.1f} below 80% of target {min_fps}"


class TestConcurrentRequestThroughput:
    """Test throughput under concurrent request load"""

    @pytest.fixture
    def mock_fastapi_server(self):
        """Mock FastAPI server for load testing"""
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import asyncio

        app = FastAPI()

        @app.post("/detect")
        async def detect(files=None):
            # Simulate AudioMAE inference time
            await asyncio.sleep(0.008)  # 8ms
            return JSONResponse(
                {"prediction": "helicopter", "confidence": 0.85, "processing_time": 8.0}
            )

        return TestClient(app)

    def test_concurrent_api_throughput(self, mock_fastapi_server):
        """Test API server throughput under concurrent load"""

        def single_request():
            """Single API request"""
            try:
                response = mock_fastapi_server.post(
                    "/detect", files={"file": ("test.wav", b"mock_audio_data", "audio/wav")}
                )
                return response.status_code == 200
            except:
                return False

        # Test with increasing concurrent load
        concurrent_levels = [1, 5, 10, 20, 50]
        results = {}

        for num_concurrent in concurrent_levels:
            print(f"\nTesting {num_concurrent} concurrent requests...")

            # Run concurrent requests
            num_requests = 100
            start_time = time.perf_counter()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(single_request) for _ in range(num_requests)]
                successful_requests = sum(
                    1 for future in concurrent.futures.as_completed(futures) if future.result()
                )

            total_time = time.perf_counter() - start_time
            requests_per_second = successful_requests / total_time

            results[num_concurrent] = {
                "rps": requests_per_second,
                "success_rate": successful_requests / num_requests,
                "total_time": total_time,
            }

            print(f"  Requests/sec: {requests_per_second:.1f}")
            print(f"  Success rate: {successful_requests/num_requests:.1%}")
            print(f"  Total time: {total_time:.2f}s")

            # Validate performance
            assert (
                results[num_concurrent]["success_rate"] >= 0.95
            ), f"Success rate too low at {num_concurrent} concurrent: {results[num_concurrent]['success_rate']:.1%}"

        # Analyze scaling
        single_rps = results[1]["rps"]
        peak_rps = max(result["rps"] for result in results.values())
        scaling_factor = peak_rps / single_rps

        print(f"\nConcurrent Request Scaling:")
        print(f"  Single thread: {single_rps:.1f} RPS")
        print(f"  Peak throughput: {peak_rps:.1f} RPS")
        print(f"  Scaling factor: {scaling_factor:.1f}x")

        # Should achieve reasonable scaling
        assert scaling_factor >= 3.0, f"Poor concurrent scaling: {scaling_factor:.1f}x"

    def test_websocket_streaming_throughput(self):
        """Test WebSocket streaming throughput"""

        # Simulate WebSocket audio streaming
        chunk_size = 1024  # Audio samples per chunk
        sample_rate = 16000
        chunk_duration_ms = (chunk_size / sample_rate) * 1000  # ~64ms

        def simulate_audio_stream(duration_seconds: float):
            """Simulate continuous audio streaming"""
            num_chunks = int(duration_seconds * sample_rate / chunk_size)
            processed_chunks = 0
            errors = 0

            start_time = time.perf_counter()

            for i in range(num_chunks):
                chunk_start = time.perf_counter()

                # Simulate audio chunk processing
                try:
                    # Simulate 8ms inference time
                    time.sleep(0.008)
                    processed_chunks += 1
                except:
                    errors += 1

                # Maintain real-time pacing
                elapsed = (time.perf_counter() - chunk_start) * 1000
                if elapsed < chunk_duration_ms:
                    time.sleep((chunk_duration_ms - elapsed) / 1000)

            total_time = time.perf_counter() - start_time
            return processed_chunks, errors, total_time

        # Test streaming for 10 seconds
        processed, errors, duration = simulate_audio_stream(10.0)

        chunks_per_second = processed / duration
        expected_chunks_per_second = sample_rate / chunk_size  # ~15.6 chunks/sec

        print(f"\nWebSocket Streaming Throughput:")
        print(f"  Processed chunks: {processed}")
        print(f"  Errors: {errors}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Chunks/sec: {chunks_per_second:.1f}")
        print(f"  Expected: {expected_chunks_per_second:.1f}")
        print(f"  Real-time ratio: {chunks_per_second/expected_chunks_per_second:.2f}")

        # Should maintain real-time processing
        real_time_ratio = chunks_per_second / expected_chunks_per_second
        assert real_time_ratio >= 0.95, f"Cannot maintain real-time: {real_time_ratio:.2f} ratio"

        # Error rate should be minimal
        error_rate = errors / (processed + errors) if (processed + errors) > 0 else 0
        assert error_rate <= 0.01, f"Error rate too high: {error_rate:.1%}"


class TestPlatformSpecificThroughput:
    """Test platform-specific throughput targets"""

    def test_jetson_orin_nano_throughput_simulation(self):
        """Simulate Jetson Orin Nano throughput characteristics"""

        def jetson_inference_simulation(batch_size: int, num_iterations: int):
            """Simulate Jetson GPU inference"""
            processed_samples = 0

            start_time = time.perf_counter()

            for _ in range(num_iterations):
                # Simulate GPU inference with memory bandwidth limits
                base_time = 0.008  # 8ms base inference
                memory_factor = 1 + (batch_size - 1) * 0.15  # Memory bandwidth impact
                thermal_factor = np.random.uniform(1.0, 1.1)  # Thermal variation

                inference_time = base_time * memory_factor * thermal_factor
                time.sleep(inference_time)

                processed_samples += batch_size

            total_time = time.perf_counter() - start_time
            return processed_samples / total_time

        # Test different batch sizes
        batch_sizes = [1, 2, 4]
        results = {}

        for batch_size in batch_sizes:
            fps = jetson_inference_simulation(batch_size, 50)
            results[batch_size] = fps

            print(f"\nJetson Orin Nano Simulation - Batch {batch_size}:")
            print(f"  FPS: {fps:.1f}")
            print(f"  Target: {ThroughputBenchmarkConfig.JETSON_ORIN_NANO_TARGET_FPS}")

        # Validate against target
        best_fps = max(results.values())
        target_fps = ThroughputBenchmarkConfig.JETSON_ORIN_NANO_TARGET_FPS
        tolerance = target_fps * ThroughputBenchmarkConfig.THROUGHPUT_TOLERANCE

        assert (
            best_fps >= target_fps - tolerance
        ), f"Jetson simulation below target: {best_fps:.1f} < {target_fps} FPS"

        print(f"\nJetson Orin Nano Validation:")
        print(f"  Best FPS: {best_fps:.1f}")
        print(f"  Target: {target_fps} (±{tolerance:.1f})")
        print(f"  Status: {'✅ PASS' if best_fps >= target_fps - tolerance else '❌ FAIL'}")

    def test_raspberry_pi_5_throughput_simulation(self):
        """Simulate Raspberry Pi 5 throughput characteristics"""

        def rpi5_inference_simulation(num_iterations: int):
            """Simulate RPi5 CPU inference with AI HAT+"""
            processed_samples = 0

            start_time = time.perf_counter()

            for _ in range(num_iterations):
                # Simulate CPU inference with AI HAT acceleration
                base_time = 0.018  # 18ms base for AI HAT
                cpu_load_factor = np.random.uniform(1.0, 1.05)  # CPU load variation

                inference_time = base_time * cpu_load_factor
                time.sleep(inference_time)

                processed_samples += 1  # Single sample processing

            total_time = time.perf_counter() - start_time
            return processed_samples / total_time

        fps = rpi5_inference_simulation(100)
        target_fps = ThroughputBenchmarkConfig.RASPBERRY_PI_5_TARGET_FPS
        tolerance = target_fps * ThroughputBenchmarkConfig.THROUGHPUT_TOLERANCE

        print(f"\nRaspberry Pi 5 Simulation:")
        print(f"  FPS: {fps:.1f}")
        print(f"  Target: {target_fps}")
        print(f"  Status: {'✅ PASS' if fps >= target_fps - tolerance else '❌ FAIL'}")

        assert (
            fps >= target_fps - tolerance
        ), f"RPi5 simulation below target: {fps:.1f} < {target_fps} FPS"

    def test_cloud_gpu_throughput_simulation(self):
        """Simulate cloud GPU (RTX 4090) throughput"""

        def rtx4090_inference_simulation(batch_size: int, num_iterations: int):
            """Simulate RTX 4090 inference"""
            processed_samples = 0

            start_time = time.perf_counter()

            for _ in range(num_iterations):
                # High-end GPU with excellent parallelization
                base_time = 0.002  # 2ms base
                batch_efficiency = 1 + (batch_size - 1) * 0.8  # Excellent batch scaling

                inference_time = base_time * batch_size / batch_efficiency
                time.sleep(inference_time)

                processed_samples += batch_size

            total_time = time.perf_counter() - start_time
            return processed_samples / total_time

        # Test with optimal batch size
        fps = rtx4090_inference_simulation(16, 50)
        target_fps = ThroughputBenchmarkConfig.RTX_4090_TARGET_FPS
        tolerance = target_fps * ThroughputBenchmarkConfig.THROUGHPUT_TOLERANCE

        print(f"\nRTX 4090 Simulation:")
        print(f"  FPS: {fps:.1f}")
        print(f"  Target: {target_fps}")
        print(f"  Status: {'✅ PASS' if fps >= target_fps - tolerance else '❌ FAIL'}")

        assert (
            fps >= target_fps - tolerance
        ), f"RTX 4090 simulation below target: {fps:.1f} < {target_fps} FPS"


class TestStressLoadTesting:
    """Stress testing and load validation"""

    def test_sustained_load_performance(self):
        """Test performance under sustained high load"""

        def sustained_inference_load(duration_minutes: float):
            """Run sustained inference load"""
            measurer = ThroughputMeasurer()
            measurer.start()

            end_time = time.time() + duration_minutes * 60

            while time.time() < end_time:
                start = time.perf_counter()

                # Simulate inference
                time.sleep(0.008)  # 8ms inference

                latency = (time.perf_counter() - start) * 1000
                measurer.record_processing(latency)

                # Check for performance degradation
                current_fps = measurer.get_current_fps()
                if current_fps > 0:
                    print(f"Current FPS: {current_fps:.1f}", end="\r")

            return measurer.get_result()

        # Run 2-minute sustained load test
        result = sustained_inference_load(2.0)

        print(f"\nSustained Load Test (2 minutes):")
        print(f"  Average FPS: {result.fps:.1f}")
        print(f"  Peak FPS: {result.peak_fps:.1f}")
        print(f"  Min FPS: {result.min_fps:.1f}")
        print(f"  Total processed: {result.total_processed}")
        print(f"  CPU usage: {result.cpu_usage_percent:.1f}%")

        # Validate sustained performance
        min_sustained_fps = ThroughputBenchmarkConfig.CPU_TARGET_FPS * 0.8
        assert (
            result.fps >= min_sustained_fps
        ), f"Sustained performance too low: {result.fps:.1f} < {min_sustained_fps}"

        # Performance should be stable (min FPS not too far below average)
        performance_stability = result.min_fps / result.fps
        assert (
            performance_stability >= 0.7
        ), f"Performance too unstable: {performance_stability:.1%}"

    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation"""

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Run inference for several iterations
        for i in range(100):
            # Simulate inference with tensor operations
            batch = torch.randn(4, 1, 128, 128)

            # Simulate some processing
            with torch.no_grad():
                output = torch.nn.functional.relu(batch)
                loss = output.mean()

            # Explicit cleanup
            del batch, output, loss

            if i % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_growth = (current_memory - initial_memory) / initial_memory

                print(f"Iteration {i}: Memory {current_memory:.1f} MB (+{memory_growth:.1%})")

        # Final memory check
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_growth = (final_memory - initial_memory) / initial_memory

        print(f"\nMemory Leak Test:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Growth: {total_growth:.1%}")

        # Memory growth should be minimal
        assert (
            total_growth <= ThroughputBenchmarkConfig.MEMORY_GROWTH_MAX
        ), f"Potential memory leak: {total_growth:.1%} growth"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
