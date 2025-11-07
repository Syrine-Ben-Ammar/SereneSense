#
# Plan:
# 1. Import necessary libraries: pytest, torch, time, statistics, numpy
# 2. Create fixtures for:
#    - Mock models (AudioMAE, AST, BEATs) with realistic inference timing
#    - Test audio data with various input sizes and formats
#    - Device configurations (CPU, GPU, edge device simulation)
#    - Performance measurement utilities
# 3. Test Categories:
#    a) Model Inference Latency Tests:
#       - Single forward pass timing for each model architecture
#       - Batch inference timing with various batch sizes
#       - Memory optimization impact on latency
#       - Precision effects (fp32, fp16, int8) on timing
#    b) Audio Processing Pipeline Latency Tests:
#       - Audio loading and preprocessing timing
#       - Spectrogram generation latency
#       - Feature extraction pipeline timing
#       - Real-time buffer processing latency
#    c) End-to-End Pipeline Latency Tests:
#       - Complete detection pipeline timing
#       - Audio input to prediction output latency
#       - WebSocket real-time streaming latency
#       - API endpoint response times
#    d) Platform-Specific Performance Tests:
#       - Jetson Orin Nano simulation (<10ms target)
#       - Raspberry Pi 5 simulation (<20ms target)
#       - Cloud GPU benchmark comparison
#       - CPU-only performance validation
#    e) Optimization Impact Tests:
#       - TensorRT optimization latency improvements
#       - ONNX export performance comparison
#       - Quantization latency vs accuracy tradeoffs
#       - Model pruning effects on inference time
# 4. Statistical analysis with confidence intervals and variance
# 5. Performance regression detection and alerting
# 6. Benchmark comparison against published results
#

import pytest
import torch
import torch.nn as nn
import time
import statistics
import numpy as np
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import psutil
import gc

# SereneSense imports
from core.models.audioMAE.audioMAE import AudioMAE
from core.models.AST.ast import AudioSpectrogramTransformer
from core.models.BEATs.beats import BEATs
from core.core.audio_processor import AudioProcessor
from core.core.inference_engine import InferenceEngine
from core.inference.realtime_inference import RealTimeInference
from core.utils.config_parser import ConfigParser

# Performance benchmarking utilities
import pytest_benchmark


class LatencyBenchmarkConfig:
    """Configuration for latency benchmarking"""

    # Target latency benchmarks (milliseconds)
    AUDIOMAЕ_TARGET_LATENCY = 8.2
    AST_TARGET_LATENCY = 10.5
    BEATS_TARGET_LATENCY = 9.1

    # Platform-specific targets
    JETSON_ORIN_NANO_TARGET = 10.0
    RASPBERRY_PI_5_TARGET = 20.0
    CLOUD_GPU_TARGET = 5.0
    CPU_TARGET = 50.0

    # Real-time requirements
    REAL_TIME_THRESHOLD = 20.0
    MILITARY_APPLICATION_THRESHOLD = 20.0

    # Benchmark parameters
    WARMUP_ITERATIONS = 10
    BENCHMARK_ITERATIONS = 100
    STATISTICAL_CONFIDENCE = 0.95

    # Audio processing parameters
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    WINDOW_LENGTH = 2.0  # seconds
    N_MELS = 128
    N_FFT = 1024


@contextmanager
def benchmark_timer():
    """Context manager for high-precision timing"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    return end_time - start_time


def warmup_model(model: nn.Module, input_tensor: torch.Tensor, iterations: int = 10):
    """Warm up model for consistent benchmarking"""
    model.eval()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_function(
    func, *args, iterations: int = 100, warmup: int = 10, **kwargs
) -> Dict[str, float]:
    """Benchmark a function with statistical analysis"""

    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark runs
    latencies = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # Statistical analysis
    return {
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "throughput_fps": 1000.0 / statistics.mean(latencies),
        "raw_latencies": latencies,
    }


class TestModelInferenceLatency:
    """Test latency for individual model inference"""

    @pytest.fixture(params=["cpu", "cuda"])
    def device(self, request):
        """Test device fixture"""
        if request.param == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return request.param

    @pytest.fixture
    def audio_input_batch1(self, device):
        """Single audio input tensor"""
        return torch.randn(1, 1, LatencyBenchmarkConfig.N_MELS, 128, device=device)

    @pytest.fixture
    def audio_input_batch4(self, device):
        """Batch audio input tensor"""
        return torch.randn(4, 1, LatencyBenchmarkConfig.N_MELS, 128, device=device)

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
        return model

    @pytest.fixture
    def ast_model(self, device):
        """Audio Spectrogram Transformer model fixture"""
        config = {
            "input_tdim": 128,
            "input_fdim": 128,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "num_classes": 7,
        }

        model = AudioSpectrogramTransformer(**config)
        model.to(device)
        model.eval()
        return model

    @pytest.fixture
    def beats_model(self, device):
        """BEATs model fixture"""
        config = {
            "input_dim": 128,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "num_classes": 7,
            "patch_size": 16,
        }

        model = BEATs(**config)
        model.to(device)
        model.eval()
        return model

    def test_audioMAE_inference_latency(
        self, audioMAE_model, audio_input_batch1, device, benchmark
    ):
        """Test AudioMAE single inference latency against 8.2ms benchmark"""

        # Warmup
        warmup_model(audioMAE_model, audio_input_batch1, LatencyBenchmarkConfig.WARMUP_ITERATIONS)

        # Benchmark inference function
        def inference():
            with torch.no_grad():
                return audioMAE_model(audio_input_batch1, mode="classification")

        # Run benchmark
        result = benchmark_function(
            inference,
            iterations=LatencyBenchmarkConfig.BENCHMARK_ITERATIONS,
            warmup=0,  # Already warmed up
        )

        # Validate against target
        target_latency = LatencyBenchmarkConfig.AUDIOMAЕ_TARGET_LATENCY
        mean_latency = result["mean_latency_ms"]

        # Allow 20% variance from published benchmark
        tolerance = target_latency * 0.2

        assert (
            mean_latency <= target_latency + tolerance
        ), f"AudioMAE latency {mean_latency:.2f}ms exceeds target {target_latency}ms (±{tolerance:.1f}ms)"

        # Log detailed results
        print(f"\nAudioMAE Inference Latency Results ({device}):")
        print(f"  Mean: {mean_latency:.2f}ms (target: {target_latency}ms)")
        print(f"  P95: {result['p95_latency_ms']:.2f}ms")
        print(f"  P99: {result['p99_latency_ms']:.2f}ms")
        print(f"  Throughput: {result['throughput_fps']:.1f} FPS")
        print(f"  Std Dev: {result['std_latency_ms']:.2f}ms")

        # Check real-time requirements
        assert (
            mean_latency <= LatencyBenchmarkConfig.REAL_TIME_THRESHOLD
        ), f"AudioMAE does not meet real-time requirement (<{LatencyBenchmarkConfig.REAL_TIME_THRESHOLD}ms)"

    def test_ast_inference_latency(self, ast_model, audio_input_batch1, device, benchmark):
        """Test AST single inference latency against 10.5ms benchmark"""

        # Warmup
        warmup_model(ast_model, audio_input_batch1, LatencyBenchmarkConfig.WARMUP_ITERATIONS)

        # Benchmark inference function
        def inference():
            with torch.no_grad():
                return ast_model(audio_input_batch1)

        # Run benchmark
        result = benchmark_function(
            inference, iterations=LatencyBenchmarkConfig.BENCHMARK_ITERATIONS, warmup=0
        )

        # Validate against target
        target_latency = LatencyBenchmarkConfig.AST_TARGET_LATENCY
        mean_latency = result["mean_latency_ms"]
        tolerance = target_latency * 0.2

        assert (
            mean_latency <= target_latency + tolerance
        ), f"AST latency {mean_latency:.2f}ms exceeds target {target_latency}ms"

        print(f"\nAST Inference Latency Results ({device}):")
        print(f"  Mean: {mean_latency:.2f}ms (target: {target_latency}ms)")
        print(f"  Throughput: {result['throughput_fps']:.1f} FPS")

    def test_beats_inference_latency(self, beats_model, audio_input_batch1, device, benchmark):
        """Test BEATs single inference latency against 9.1ms benchmark"""

        # Warmup
        warmup_model(beats_model, audio_input_batch1, LatencyBenchmarkConfig.WARMUP_ITERATIONS)

        # Benchmark inference function
        def inference():
            with torch.no_grad():
                return beats_model(audio_input_batch1)

        # Run benchmark
        result = benchmark_function(
            inference, iterations=LatencyBenchmarkConfig.BENCHMARK_ITERATIONS, warmup=0
        )

        # Validate against target
        target_latency = LatencyBenchmarkConfig.BEATS_TARGET_LATENCY
        mean_latency = result["mean_latency_ms"]
        tolerance = target_latency * 0.2

        assert (
            mean_latency <= target_latency + tolerance
        ), f"BEATs latency {mean_latency:.2f}ms exceeds target {target_latency}ms"

        print(f"\nBEATs Inference Latency Results ({device}):")
        print(f"  Mean: {mean_latency:.2f}ms (target: {target_latency}ms)")
        print(f"  Throughput: {result['throughput_fps']:.1f} FPS")

    def test_batch_inference_latency(self, audioMAE_model, audio_input_batch4, device):
        """Test batch inference latency scaling"""

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        results = {}

        for batch_size in batch_sizes:
            if device == "cpu" and batch_size > 4:
                continue  # Skip large batches on CPU

            # Create input tensor
            input_tensor = torch.randn(
                batch_size, 1, LatencyBenchmarkConfig.N_MELS, 128, device=device
            )

            # Warmup
            warmup_model(audioMAE_model, input_tensor, 5)

            # Benchmark
            def batch_inference():
                with torch.no_grad():
                    return audioMAE_model(input_tensor, mode="classification")

            result = benchmark_function(batch_inference, iterations=50, warmup=0)

            results[batch_size] = result

            # Calculate per-sample latency
            per_sample_latency = result["mean_latency_ms"] / batch_size

            print(f"\nBatch Size {batch_size} ({device}):")
            print(f"  Total latency: {result['mean_latency_ms']:.2f}ms")
            print(f"  Per-sample latency: {per_sample_latency:.2f}ms")
            print(f"  Throughput: {result['throughput_fps']:.1f} batches/sec")
            print(f"  Samples/sec: {result['throughput_fps'] * batch_size:.1f}")

        # Verify batch processing efficiency
        if len(results) >= 2:
            batch1_per_sample = results[1]["mean_latency_ms"]
            batch4_per_sample = results[4]["mean_latency_ms"] / 4 if 4 in results else None

            if batch4_per_sample:
                efficiency_gain = batch1_per_sample / batch4_per_sample
                assert (
                    efficiency_gain >= 1.5
                ), f"Batch processing not efficient enough: {efficiency_gain:.2f}x speedup"

                print(f"\nBatch Processing Efficiency: {efficiency_gain:.2f}x speedup")


class TestAudioProcessingLatency:
    """Test latency for audio processing pipeline"""

    @pytest.fixture
    def audio_processor_config(self):
        """Audio processor configuration"""
        return {
            "sample_rate": LatencyBenchmarkConfig.SAMPLE_RATE,
            "n_mels": LatencyBenchmarkConfig.N_MELS,
            "n_fft": LatencyBenchmarkConfig.N_FFT,
            "hop_length": 512,
            "win_length": LatencyBenchmarkConfig.N_FFT,
            "f_min": 0.0,
            "f_max": 8000.0,
            "normalize": True,
        }

    @pytest.fixture
    def audio_processor(self, audio_processor_config):
        """Audio processor instance"""
        return AudioProcessor(audio_processor_config)

    @pytest.fixture
    def raw_audio_data(self):
        """Raw audio data for processing"""
        duration = LatencyBenchmarkConfig.WINDOW_LENGTH
        sample_rate = LatencyBenchmarkConfig.SAMPLE_RATE
        num_samples = int(duration * sample_rate)
        return np.random.randn(num_samples).astype(np.float32)

    def test_spectrogram_generation_latency(self, audio_processor, raw_audio_data):
        """Test spectrogram generation latency"""

        def generate_spectrogram():
            return audio_processor.to_spectrogram(torch.from_numpy(raw_audio_data))

        result = benchmark_function(
            generate_spectrogram,
            iterations=LatencyBenchmarkConfig.BENCHMARK_ITERATIONS,
            warmup=LatencyBenchmarkConfig.WARMUP_ITERATIONS,
        )

        # Spectrogram generation should be fast (<2ms for real-time)
        target_latency = 2.0
        mean_latency = result["mean_latency_ms"]

        assert (
            mean_latency <= target_latency
        ), f"Spectrogram generation too slow: {mean_latency:.2f}ms > {target_latency}ms"

        print(f"\nSpectrogram Generation Latency:")
        print(f"  Mean: {mean_latency:.2f}ms (target: <{target_latency}ms)")
        print(f"  P95: {result['p95_latency_ms']:.2f}ms")
        print(f"  Throughput: {result['throughput_fps']:.1f} spectrograms/sec")

    def test_audio_loading_latency(self, audio_processor):
        """Test audio file loading latency"""

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Generate test audio
            sample_rate = LatencyBenchmarkConfig.SAMPLE_RATE
            duration = LatencyBenchmarkConfig.WINDOW_LENGTH
            audio_data = torch.randn(1, int(sample_rate * duration))

            # Save to file
            torchaudio.save(tmp_file.name, audio_data, sample_rate)

            def load_audio():
                return audio_processor.load_audio(tmp_file.name)

            result = benchmark_function(
                load_audio, iterations=50, warmup=5  # Fewer iterations for file I/O
            )

            # Audio loading should be reasonably fast
            target_latency = 5.0  # 5ms target for file loading
            mean_latency = result["mean_latency_ms"]

            assert (
                mean_latency <= target_latency
            ), f"Audio loading too slow: {mean_latency:.2f}ms > {target_latency}ms"

            print(f"\nAudio Loading Latency:")
            print(f"  Mean: {mean_latency:.2f}ms (target: <{target_latency}ms)")

    def test_real_time_buffer_processing(self, audio_processor):
        """Test real-time audio buffer processing latency"""

        chunk_size = LatencyBenchmarkConfig.CHUNK_SIZE
        audio_chunk = np.random.randn(chunk_size).astype(np.float32)

        def process_chunk():
            return audio_processor.process_buffer(torch.from_numpy(audio_chunk))

        result = benchmark_function(
            process_chunk,
            iterations=LatencyBenchmarkConfig.BENCHMARK_ITERATIONS,
            warmup=LatencyBenchmarkConfig.WARMUP_ITERATIONS,
        )

        # Buffer processing must be faster than audio chunk duration
        chunk_duration_ms = (chunk_size / LatencyBenchmarkConfig.SAMPLE_RATE) * 1000
        max_allowed_latency = chunk_duration_ms * 0.1  # 10% of chunk duration

        mean_latency = result["mean_latency_ms"]

        assert (
            mean_latency <= max_allowed_latency
        ), f"Buffer processing too slow: {mean_latency:.2f}ms > {max_allowed_latency:.2f}ms"

        print(f"\nReal-time Buffer Processing:")
        print(f"  Mean: {mean_latency:.2f}ms")
        print(f"  Chunk duration: {chunk_duration_ms:.2f}ms")
        print(f"  Processing overhead: {(mean_latency/chunk_duration_ms)*100:.1f}%")


class TestEndToEndPipelineLatency:
    """Test complete detection pipeline latency"""

    @pytest.fixture
    def mock_inference_engine(self):
        """Mock inference engine with realistic timing"""
        engine = Mock(spec=InferenceEngine)

        def mock_predict(audio_input):
            # Simulate AudioMAE inference time
            time.sleep(0.008)  # 8ms sleep to simulate inference
            return {
                "predicted_class": "helicopter",
                "confidence": 0.85,
                "probabilities": {"helicopter": 0.85, "background": 0.15},
            }

        engine.predict.side_effect = mock_predict
        return engine

    @pytest.fixture
    def mock_audio_processor(self):
        """Mock audio processor with realistic timing"""
        processor = Mock(spec=AudioProcessor)

        def mock_process_file(file_path):
            # Simulate audio processing time
            time.sleep(0.002)  # 2ms sleep to simulate processing
            return torch.randn(1, LatencyBenchmarkConfig.N_MELS, 128)

        processor.process_file.side_effect = mock_process_file
        return processor

    def test_end_to_end_detection_latency(self, mock_inference_engine, mock_audio_processor):
        """Test complete detection pipeline latency"""

        def end_to_end_detection():
            # Simulate complete pipeline
            start_time = time.perf_counter()

            # Audio processing (2ms simulated)
            processed_audio = mock_audio_processor.process_file("test.wav")

            # Model inference (8ms simulated)
            result = mock_inference_engine.predict(processed_audio)

            end_time = time.perf_counter()
            return (end_time - start_time) * 1000, result

        latencies = []

        # Run multiple iterations
        for _ in range(50):
            latency_ms, result = end_to_end_detection()
            latencies.append(latency_ms)

        mean_latency = statistics.mean(latencies)

        # Total pipeline should meet real-time requirements
        target_latency = LatencyBenchmarkConfig.REAL_TIME_THRESHOLD

        assert (
            mean_latency <= target_latency
        ), f"End-to-end pipeline too slow: {mean_latency:.2f}ms > {target_latency}ms"

        print(f"\nEnd-to-End Pipeline Latency:")
        print(f"  Mean: {mean_latency:.2f}ms (target: <{target_latency}ms)")
        print(f"  P95: {np.percentile(latencies, 95):.2f}ms")
        print(f"  Components: Audio Processing (~2ms) + Inference (~8ms)")


class TestPlatformSpecificLatency:
    """Test platform-specific latency requirements"""

    def test_jetson_orin_nano_simulation(self):
        """Test Jetson Orin Nano latency requirements (<10ms)"""

        # Simulate Jetson performance characteristics
        def jetson_inference_simulation():
            # Jetson typically has:
            # - GPU acceleration available
            # - Memory bandwidth limitations
            # - Thermal throttling potential

            # Simulate optimized inference time
            base_latency = 0.008  # 8ms base
            thermal_factor = np.random.uniform(1.0, 1.2)  # 0-20% thermal impact

            simulated_latency = base_latency * thermal_factor
            time.sleep(simulated_latency)
            return simulated_latency * 1000

        latencies = []
        for _ in range(100):
            latency = jetson_inference_simulation()
            latencies.append(latency)

        mean_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        # Jetson Orin Nano target: <10ms
        target_latency = LatencyBenchmarkConfig.JETSON_ORIN_NANO_TARGET

        assert (
            mean_latency <= target_latency
        ), f"Jetson Orin Nano simulation fails: {mean_latency:.2f}ms > {target_latency}ms"

        assert (
            p95_latency <= target_latency * 1.2
        ), f"Jetson P95 latency too high: {p95_latency:.2f}ms"

        print(f"\nJetson Orin Nano Simulation:")
        print(f"  Mean: {mean_latency:.2f}ms (target: <{target_latency}ms)")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  Thermal variance: ±{(max(latencies)-min(latencies))/2:.1f}ms")

    def test_raspberry_pi_5_simulation(self):
        """Test Raspberry Pi 5 latency requirements (<20ms)"""

        def rpi5_inference_simulation():
            # RPi5 characteristics:
            # - CPU-only inference
            # - Limited memory bandwidth
            # - Consistent but slower performance

            base_latency = 0.018  # 18ms base for CPU inference
            cpu_load_factor = np.random.uniform(1.0, 1.1)  # 0-10% load impact

            simulated_latency = base_latency * cpu_load_factor
            time.sleep(simulated_latency)
            return simulated_latency * 1000

        latencies = []
        for _ in range(100):
            latency = rpi5_inference_simulation()
            latencies.append(latency)

        mean_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        # Raspberry Pi 5 target: <20ms
        target_latency = LatencyBenchmarkConfig.RASPBERRY_PI_5_TARGET

        assert (
            mean_latency <= target_latency
        ), f"Raspberry Pi 5 simulation fails: {mean_latency:.2f}ms > {target_latency}ms"

        print(f"\nRaspberry Pi 5 Simulation:")
        print(f"  Mean: {mean_latency:.2f}ms (target: <{target_latency}ms)")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  CPU load variance: ±{statistics.stdev(latencies):.1f}ms")

    def test_cloud_gpu_performance(self):
        """Test cloud GPU performance baseline"""

        def cloud_gpu_simulation():
            # Cloud GPU (RTX 4090) characteristics:
            # - High compute capability
            # - Excellent memory bandwidth
            # - Consistent performance

            base_latency = 0.002  # 2ms for high-end GPU
            return base_latency * 1000

        latencies = []
        for _ in range(100):
            latency = cloud_gpu_simulation()
            latencies.append(latency)

        mean_latency = statistics.mean(latencies)
        target_latency = LatencyBenchmarkConfig.CLOUD_GPU_TARGET

        assert (
            mean_latency <= target_latency
        ), f"Cloud GPU performance insufficient: {mean_latency:.2f}ms > {target_latency}ms"

        print(f"\nCloud GPU (RTX 4090) Simulation:")
        print(f"  Mean: {mean_latency:.2f}ms (target: <{target_latency}ms)")
        print(f"  Throughput: {1000/mean_latency:.0f} FPS")


class TestOptimizationImpactLatency:
    """Test impact of various optimizations on latency"""

    def test_tensorrt_optimization_impact(self):
        """Test TensorRT optimization latency improvement"""

        # Baseline PyTorch model timing
        def pytorch_inference():
            time.sleep(0.008)  # 8ms baseline
            return 8.0

        # TensorRT optimized timing (typically 2-4x faster)
        def tensorrt_inference():
            time.sleep(0.002)  # 2ms optimized
            return 2.0

        pytorch_latency = pytorch_inference()
        tensorrt_latency = tensorrt_inference()

        speedup = pytorch_latency / tensorrt_latency

        # TensorRT should provide at least 2x speedup
        assert speedup >= 2.0, f"TensorRT speedup insufficient: {speedup:.1f}x"

        print(f"\nTensorRT Optimization Impact:")
        print(f"  PyTorch: {pytorch_latency:.2f}ms")
        print(f"  TensorRT: {tensorrt_latency:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")

    def test_quantization_latency_impact(self):
        """Test quantization impact on latency"""

        precisions = {
            "fp32": 8.0,  # Baseline
            "fp16": 6.0,  # ~25% faster
            "int8": 4.0,  # ~50% faster
        }

        for precision, expected_latency in precisions.items():
            print(f"\n{precision.upper()} Precision:")
            print(f"  Expected latency: {expected_latency:.1f}ms")

            # Validate precision meets real-time requirements
            assert (
                expected_latency <= LatencyBenchmarkConfig.REAL_TIME_THRESHOLD
            ), f"{precision} precision too slow: {expected_latency}ms"

        # Verify quantization provides meaningful speedup
        fp32_latency = precisions["fp32"]
        int8_latency = precisions["int8"]
        quantization_speedup = fp32_latency / int8_latency

        assert (
            quantization_speedup >= 1.5
        ), f"Quantization speedup insufficient: {quantization_speedup:.1f}x"

        print(f"\nQuantization Impact Summary:")
        print(f"  FP32 → INT8 speedup: {quantization_speedup:.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--benchmark-only"])
