"""
Integration tests for SereneSense inference pipeline.

Tests end-to-end inference functionality including:
- Real-time audio processing
- Batch inference processing
- Model optimization pipeline
- Audio streaming and buffering
- Latency and throughput benchmarks
- Edge device deployment simulation
- WebSocket inference integration
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import asyncio
import time
import sys
from unittest.mock import Mock, patch, AsyncMock
import json
import threading
from queue import Queue

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.inference.real_time import RealTimeDetector, AudioBuffer
from core.inference.batch import BatchProcessor
from core.inference.optimization.tensorrt import TensorRTOptimizer
from core.inference.optimization.onnx_export import ONNXExporter
from core.inference.optimization.quantization import ModelQuantizer
from core.models.audioMAE.model import AudioMAE
from core.models.AST.model import AST
from core.core.audio_processor import AudioProcessor
from core.core.model_manager import ModelManager


class TestRealTimeInference:
    """Test real-time inference functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        config = {
            "input_dim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 10,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)
        model.eval()
        return model

    @pytest.fixture
    def audio_processor_config(self):
        """Create audio processor configuration."""
        return {
            "sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 512,
            "window_length": 1024,
            "normalize": True,
        }

    @pytest.fixture
    def real_time_config(self):
        """Create real-time detector configuration."""
        return {
            "sample_rate": 16000,
            "chunk_size": 1024,
            "buffer_size": 32000,  # 2 seconds
            "hop_length": 512,
            "confidence_threshold": 0.7,
            "max_detections_per_second": 10,
            "enable_voice_activity_detection": False,
        }

    def test_audio_buffer_initialization(self, real_time_config):
        """Test audio buffer initialization."""
        buffer = AudioBuffer(
            buffer_size=real_time_config["buffer_size"], sample_rate=real_time_config["sample_rate"]
        )

        assert buffer.buffer_size == real_time_config["buffer_size"]
        assert buffer.sample_rate == real_time_config["sample_rate"]
        assert len(buffer.buffer) == real_time_config["buffer_size"]
        assert buffer.write_position == 0

    def test_audio_buffer_write_and_read(self, real_time_config):
        """Test audio buffer write and read operations."""
        buffer = AudioBuffer(
            buffer_size=real_time_config["buffer_size"], sample_rate=real_time_config["sample_rate"]
        )

        # Write audio chunk
        chunk_size = 1024
        audio_chunk = np.random.randn(chunk_size).astype(np.float32)
        buffer.write(audio_chunk)

        # Read audio segment
        segment_length = 16000  # 1 second
        audio_segment = buffer.read(segment_length)

        assert len(audio_segment) == segment_length
        assert isinstance(audio_segment, np.ndarray)
        assert audio_segment.dtype == np.float32

    def test_audio_buffer_circular_behavior(self, real_time_config):
        """Test audio buffer circular behavior."""
        buffer_size = 8000
        buffer = AudioBuffer(buffer_size=buffer_size, sample_rate=16000)

        # Write more data than buffer size
        chunk_size = 1024
        num_chunks = 10  # Total: 10240 samples > 8000

        for i in range(num_chunks):
            audio_chunk = np.ones(chunk_size) * i  # Each chunk has unique value
            buffer.write(audio_chunk)

        # Read full buffer
        full_buffer = buffer.read(buffer_size)

        # Should contain the most recent data
        assert len(full_buffer) == buffer_size
        # The buffer should wrap around, so we shouldn't see the earliest chunks
        assert not np.all(full_buffer == 0)  # First chunk (all zeros) should be overwritten

    def test_real_time_detector_initialization(
        self, simple_model, audio_processor_config, real_time_config
    ):
        """Test real-time detector initialization."""
        audio_processor = AudioProcessor(audio_processor_config)

        detector = RealTimeDetector(
            model=simple_model,
            audio_processor=audio_processor,
            config=real_time_config,
            device=torch.device("cpu"),
        )

        assert detector.model == simple_model
        assert detector.audio_processor == audio_processor
        assert detector.config == real_time_config
        assert detector.device == torch.device("cpu")
        assert isinstance(detector.audio_buffer, AudioBuffer)

    def test_real_time_detector_process_chunk(
        self, simple_model, audio_processor_config, real_time_config
    ):
        """Test real-time detector chunk processing."""
        audio_processor = AudioProcessor(audio_processor_config)
        detector = RealTimeDetector(
            model=simple_model,
            audio_processor=audio_processor,
            config=real_time_config,
            device=torch.device("cpu"),
        )

        # Process audio chunk
        chunk_size = real_time_config["chunk_size"]
        audio_chunk = np.random.randn(chunk_size).astype(np.float32)

        result = detector.process_chunk(audio_chunk)

        # Should return detection result
        assert isinstance(result, dict)
        if "predictions" in result:
            assert isinstance(result["predictions"], (list, np.ndarray, torch.Tensor))
        if "confidence" in result:
            assert isinstance(result["confidence"], (float, np.floating))
        if "timestamp" in result:
            assert isinstance(result["timestamp"], (int, float))

    def test_real_time_detector_streaming_simulation(
        self, simple_model, audio_processor_config, real_time_config
    ):
        """Test real-time detector with streaming simulation."""
        audio_processor = AudioProcessor(audio_processor_config)
        detector = RealTimeDetector(
            model=simple_model,
            audio_processor=audio_processor,
            config=real_time_config,
            device=torch.device("cpu"),
        )

        # Simulate streaming audio
        chunk_size = real_time_config["chunk_size"]
        num_chunks = 10
        results = []

        for i in range(num_chunks):
            # Generate audio chunk with some pattern
            t = np.linspace(i * chunk_size / 16000, (i + 1) * chunk_size / 16000, chunk_size)
            audio_chunk = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # A4 note

            result = detector.process_chunk(audio_chunk)
            results.append(result)

        assert len(results) == num_chunks

        # Each result should be valid
        for result in results:
            assert isinstance(result, dict)
            # Results might be None if not enough data accumulated
            if result is not None:
                assert "timestamp" in result

    def test_real_time_detector_latency(
        self, simple_model, audio_processor_config, real_time_config
    ):
        """Test real-time detector latency."""
        audio_processor = AudioProcessor(audio_processor_config)
        detector = RealTimeDetector(
            model=simple_model,
            audio_processor=audio_processor,
            config=real_time_config,
            device=torch.device("cpu"),
        )

        # Measure processing latency
        chunk_size = real_time_config["chunk_size"]
        audio_chunk = np.random.randn(chunk_size).astype(np.float32)

        # Warmup
        for _ in range(5):
            detector.process_chunk(audio_chunk)

        # Measure latency
        latencies = []
        num_measurements = 20

        for _ in range(num_measurements):
            start_time = time.perf_counter()
            result = detector.process_chunk(audio_chunk)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        # For real-time processing, latency should be reasonable
        # This is just a smoke test - actual requirements depend on use case
        assert avg_latency < 100  # Less than 100ms on average
        assert max_latency < 200  # Maximum latency less than 200ms

        print(f"Average latency: {avg_latency:.2f}ms, Max latency: {max_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_real_time_detector_async_processing(
        self, simple_model, audio_processor_config, real_time_config
    ):
        """Test real-time detector async processing."""
        audio_processor = AudioProcessor(audio_processor_config)
        detector = RealTimeDetector(
            model=simple_model,
            audio_processor=audio_processor,
            config=real_time_config,
            device=torch.device("cpu"),
        )

        # Simulate async audio streaming
        async def audio_stream():
            chunk_size = real_time_config["chunk_size"]
            for i in range(5):
                audio_chunk = np.random.randn(chunk_size).astype(np.float32)
                yield audio_chunk
                await asyncio.sleep(0.01)  # Small delay to simulate real streaming

        results = []
        async for chunk in audio_stream():
            # Process chunk (simulating async processing)
            result = await asyncio.get_event_loop().run_in_executor(
                None, detector.process_chunk, chunk
            )
            results.append(result)

        assert len(results) == 5


class TestBatchInference:
    """Test batch inference functionality."""

    @pytest.fixture
    def batch_model(self):
        """Create model for batch testing."""
        config = {
            "input_tdim": 1000,
            "input_fdim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "num_classes": 10,
        }
        model = AST(config)
        model.eval()
        return model

    @pytest.fixture
    def batch_config(self):
        """Create batch processor configuration."""
        return {
            "batch_size": 4,
            "num_workers": 2,
            "prefetch_factor": 2,
            "device": "cpu",
            "mixed_precision": False,
            "output_format": "json",
        }

    @pytest.fixture
    def sample_batch_data(self):
        """Create sample batch data."""
        batch_size = 8
        time_dim = 1000
        freq_dim = 128

        # Create batch of spectrograms
        spectrograms = torch.randn(batch_size, freq_dim, time_dim)
        filenames = [f"audio_{i:03d}.wav" for i in range(batch_size)]

        return list(zip(spectrograms, filenames))

    def test_batch_processor_initialization(self, batch_model, batch_config):
        """Test batch processor initialization."""
        processor = BatchProcessor(model=batch_model, config=batch_config)

        assert processor.model == batch_model
        assert processor.config == batch_config
        assert processor.batch_size == batch_config["batch_size"]
        assert processor.device == torch.device(batch_config["device"])

    def test_batch_processor_single_batch(self, batch_model, batch_config, sample_batch_data):
        """Test batch processor with single batch."""
        processor = BatchProcessor(model=batch_model, config=batch_config)

        # Process single batch
        batch_spectrograms = torch.stack([item[0] for item in sample_batch_data[:4]])
        batch_filenames = [item[1] for item in sample_batch_data[:4]]

        results = processor.process_batch(batch_spectrograms, batch_filenames)

        assert isinstance(results, list)
        assert len(results) == 4

        for result in results:
            assert "filename" in result
            assert "predictions" in result
            assert "confidence" in result
            assert isinstance(result["predictions"], (list, np.ndarray))

    def test_batch_processor_multiple_batches(self, batch_model, batch_config, sample_batch_data):
        """Test batch processor with multiple batches."""
        processor = BatchProcessor(model=batch_model, config=batch_config)

        # Process all data in batches
        all_results = processor.process_dataset(sample_batch_data)

        assert isinstance(all_results, list)
        assert len(all_results) == len(sample_batch_data)

        # Check that all samples were processed
        processed_filenames = {result["filename"] for result in all_results}
        expected_filenames = {item[1] for item in sample_batch_data}
        assert processed_filenames == expected_filenames

    def test_batch_processor_throughput(self, batch_model, batch_config):
        """Test batch processor throughput."""
        processor = BatchProcessor(model=batch_model, config=batch_config)

        # Create larger dataset for throughput testing
        num_samples = 32
        time_dim = 1000
        freq_dim = 128

        large_dataset = []
        for i in range(num_samples):
            spectrogram = torch.randn(freq_dim, time_dim)
            filename = f"throughput_test_{i:03d}.wav"
            large_dataset.append((spectrogram, filename))

        # Measure processing time
        start_time = time.perf_counter()
        results = processor.process_dataset(large_dataset)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        throughput = num_samples / processing_time

        assert len(results) == num_samples
        print(f"Throughput: {throughput:.2f} samples/second")

        # Throughput should be reasonable (this is platform dependent)
        assert throughput > 1.0  # At least 1 sample per second

    def test_batch_processor_memory_efficiency(self, batch_model, batch_config):
        """Test batch processor memory efficiency."""
        processor = BatchProcessor(model=batch_model, config=batch_config)

        # Monitor memory usage during processing
        import psutil

        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process moderately large batch
        batch_size = 16
        spectrograms = torch.randn(batch_size, 128, 1000)
        filenames = [f"memory_test_{i}.wav" for i in range(batch_size)]

        results = processor.process_batch(spectrograms, filenames)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        assert len(results) == batch_size
        print(f"Memory increase: {memory_increase:.2f} MB")

        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase


class TestModelOptimization:
    """Test model optimization for inference."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def optimization_model(self):
        """Create model for optimization testing."""
        config = {
            "input_dim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 3,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,  # Disable dropout for inference
            "num_classes": 10,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)
        model.eval()
        return model

    def test_onnx_export(self, optimization_model, temp_dir):
        """Test ONNX model export."""
        exporter = ONNXExporter()

        # Create dummy input
        dummy_input = torch.randn(1, 1024, 128)

        # Export to ONNX
        onnx_path = temp_dir / "model.onnx"
        export_info = exporter.export_model(
            model=optimization_model,
            dummy_input=dummy_input,
            output_path=str(onnx_path),
            input_names=["audio_input"],
            output_names=["predictions"],
            dynamic_axes={"audio_input": {1: "sequence_length"}},
        )

        assert onnx_path.exists()
        assert "export_time" in export_info
        assert "model_size_mb" in export_info

        # Verify ONNX model can be loaded
        try:
            import onnx

            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
        except ImportError:
            pytest.skip("ONNX not available")

    def test_model_quantization(self, optimization_model, temp_dir):
        """Test model quantization."""
        quantizer = ModelQuantizer()

        # Prepare calibration data
        calibration_data = [torch.randn(1, 1024, 128) for _ in range(10)]

        # Quantize model
        quantized_path = temp_dir / "quantized_model.pth"
        quantization_info = quantizer.quantize_model(
            model=optimization_model,
            calibration_data=calibration_data,
            output_path=str(quantized_path),
            quantization_type="dynamic",
        )

        assert quantized_path.exists()
        assert "quantization_time" in quantization_info
        assert "size_reduction" in quantization_info

        # Load and test quantized model
        quantized_model = torch.load(quantized_path, map_location="cpu")

        # Test inference with quantized model
        test_input = torch.randn(1, 1024, 128)
        with torch.no_grad():
            output = quantized_model(test_input)

        assert output.shape == (1, 10)  # num_classes
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tensorrt_optimization(self, optimization_model, temp_dir):
        """Test TensorRT optimization (requires CUDA)."""
        try:
            optimizer = TensorRTOptimizer()

            # First export to ONNX
            exporter = ONNXExporter()
            dummy_input = torch.randn(1, 1024, 128)
            onnx_path = temp_dir / "model.onnx"

            exporter.export_model(
                model=optimization_model, dummy_input=dummy_input, output_path=str(onnx_path)
            )

            # Optimize with TensorRT
            trt_path = temp_dir / "model.trt"
            optimization_info = optimizer.optimize_model(
                onnx_path=str(onnx_path),
                output_path=str(trt_path),
                precision="fp16",
                max_batch_size=4,
            )

            assert trt_path.exists()
            assert "optimization_time" in optimization_info
            assert "speedup_factor" in optimization_info

        except ImportError:
            pytest.skip("TensorRT not available")

    def test_optimization_pipeline(self, optimization_model, temp_dir):
        """Test complete optimization pipeline."""
        # Step 1: Export to ONNX
        exporter = ONNXExporter()
        dummy_input = torch.randn(1, 1024, 128)
        onnx_path = temp_dir / "pipeline_model.onnx"

        export_info = exporter.export_model(
            model=optimization_model, dummy_input=dummy_input, output_path=str(onnx_path)
        )

        # Step 2: Quantize original model
        quantizer = ModelQuantizer()
        calibration_data = [torch.randn(1, 1024, 128) for _ in range(5)]
        quantized_path = temp_dir / "pipeline_quantized.pth"

        quantization_info = quantizer.quantize_model(
            model=optimization_model,
            calibration_data=calibration_data,
            output_path=str(quantized_path),
            quantization_type="dynamic",
        )

        # Verify all optimization steps completed
        assert onnx_path.exists()
        assert quantized_path.exists()
        assert export_info["export_time"] > 0
        assert quantization_info["quantization_time"] > 0

        # Compare model sizes
        original_size = sum(p.numel() * p.element_size() for p in optimization_model.parameters())
        onnx_size = onnx_path.stat().st_size
        quantized_size = quantized_path.stat().st_size

        print(f"Original model: {original_size / 1024 / 1024:.2f} MB")
        print(f"ONNX model: {onnx_size / 1024 / 1024:.2f} MB")
        print(f"Quantized model: {quantized_size / 1024 / 1024:.2f} MB")


class TestEdgeDeploymentSimulation:
    """Test edge deployment simulation."""

    @pytest.fixture
    def edge_model(self):
        """Create lightweight model for edge testing."""
        config = {
            "input_dim": 64,
            "patch_size": 8,
            "embed_dim": 128,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "num_classes": 5,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(config)
        model.eval()
        return model

    @pytest.fixture
    def edge_config(self):
        """Create edge deployment configuration."""
        return {
            "device": "cpu",  # Simulate edge device
            "precision": "fp32",
            "max_batch_size": 1,
            "enable_optimization": True,
            "latency_target_ms": 50,
            "power_budget_watts": 10,
            "memory_limit_mb": 512,
        }

    def test_edge_inference_latency(self, edge_model, edge_config):
        """Test inference latency on simulated edge device."""
        model = edge_model
        device = torch.device(edge_config["device"])
        model = model.to(device)

        # Simulate edge input size
        input_size = (1, 512, 64)  # Smaller input for edge
        test_input = torch.randn(*input_size).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)

        # Measure latency
        latencies = []
        num_runs = 50

        for _ in range(num_runs):
            start_time = time.perf_counter()

            with torch.no_grad():
                output = model(test_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"Edge inference - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

        # Check against target latency
        target_latency = edge_config["latency_target_ms"]
        assert (
            avg_latency < target_latency
        ), f"Average latency {avg_latency:.2f}ms exceeds target {target_latency}ms"
        assert p95_latency < target_latency * 1.5, f"P95 latency {p95_latency:.2f}ms too high"

    def test_edge_memory_usage(self, edge_model, edge_config):
        """Test memory usage on simulated edge device."""
        import psutil

        process = psutil.Process()

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        model = edge_model
        device = torch.device(edge_config["device"])
        model = model.to(device)

        # Load model and perform inference
        input_size = (1, 512, 64)
        test_input = torch.randn(*input_size).to(device)

        with torch.no_grad():
            output = model(test_input)

        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - baseline_memory

        print(f"Edge memory usage: {memory_usage:.2f} MB")

        # Check against memory limit
        memory_limit = edge_config["memory_limit_mb"]
        assert (
            memory_usage < memory_limit
        ), f"Memory usage {memory_usage:.2f}MB exceeds limit {memory_limit}MB"

    def test_edge_throughput(self, edge_model, edge_config):
        """Test throughput on simulated edge device."""
        model = edge_model
        device = torch.device(edge_config["device"])
        model = model.to(device)

        # Test throughput with small batches
        batch_size = edge_config["max_batch_size"]
        input_size = (batch_size, 512, 64)

        # Measure throughput
        num_batches = 20
        total_samples = num_batches * batch_size

        start_time = time.perf_counter()

        for _ in range(num_batches):
            test_input = torch.randn(*input_size).to(device)
            with torch.no_grad():
                output = model(test_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_time = end_time - start_time
        throughput = total_samples / total_time

        print(f"Edge throughput: {throughput:.2f} samples/second")

        # Throughput should be reasonable for edge device
        assert throughput > 1.0, "Edge throughput too low"

    def test_edge_power_simulation(self, edge_model, edge_config):
        """Test simulated power consumption."""
        # This is a simulation since we can't measure actual power
        # In practice, this would integrate with hardware power monitoring

        model = edge_model
        device = torch.device(edge_config["device"])
        model = model.to(device)

        # Estimate computational load
        input_size = (1, 512, 64)
        test_input = torch.randn(*input_size).to(device)

        # Count FLOPs (simplified estimation)
        total_params = sum(p.numel() for p in model.parameters())

        # Simulate power consumption based on model size and activity
        base_power = 2.0  # Base power consumption in watts
        compute_power = total_params / 1e6 * 0.1  # Rough estimate
        estimated_power = base_power + compute_power

        print(f"Estimated power consumption: {estimated_power:.2f} watts")

        # Check against power budget
        power_budget = edge_config["power_budget_watts"]
        assert (
            estimated_power < power_budget
        ), f"Estimated power {estimated_power:.2f}W exceeds budget {power_budget}W"


class TestInferencePipelineIntegration:
    """Test complete inference pipeline integration."""

    @pytest.fixture
    def pipeline_components(self):
        """Create complete pipeline components."""
        # Model
        model_config = {
            "input_dim": 128,
            "patch_size": 16,
            "embed_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "num_classes": 8,
            "mask_ratio": 0.75,
        }
        model = AudioMAE(model_config)
        model.eval()

        # Audio processor
        audio_config = {
            "sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 512,
            "normalize": True,
        }
        audio_processor = AudioProcessor(audio_config)

        # Model manager
        manager_config = {"model_cache_size": 3, "device": "cpu", "enable_optimization": True}
        model_manager = ModelManager(manager_config)

        return {"model": model, "audio_processor": audio_processor, "model_manager": model_manager}

    def test_end_to_end_inference_pipeline(self, pipeline_components):
        """Test complete end-to-end inference pipeline."""
        model = pipeline_components["model"]
        audio_processor = pipeline_components["audio_processor"]

        # Simulate raw audio input
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        raw_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # A4 note

        # Step 1: Audio preprocessing
        processed_audio = audio_processor.process_audio(raw_audio)

        # Step 2: Model inference
        with torch.no_grad():
            predictions = model(processed_audio.unsqueeze(0))

        # Step 3: Post-processing
        probabilities = torch.softmax(predictions, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]

        # Verify pipeline output
        assert predictions.shape == (1, 8)  # batch_size=1, num_classes=8
        assert 0 <= predicted_class.item() < 8
        assert 0.0 <= confidence.item() <= 1.0

        print(f"Predicted class: {predicted_class.item()}, Confidence: {confidence.item():.3f}")

    def test_pipeline_error_handling(self, pipeline_components):
        """Test pipeline error handling."""
        model = pipeline_components["model"]
        audio_processor = pipeline_components["audio_processor"]

        # Test with invalid input shapes
        invalid_inputs = [
            np.array([]),  # Empty array
            np.random.randn(100),  # Too short
            np.random.randn(1000000),  # Too long
            np.full(16000, np.inf),  # Invalid values
        ]

        for invalid_input in invalid_inputs:
            try:
                processed = audio_processor.process_audio(invalid_input)
                # If processing succeeds, inference should handle it gracefully
                with torch.no_grad():
                    output = model(processed.unsqueeze(0))
                    assert torch.isfinite(output).all()
            except (ValueError, RuntimeError) as e:
                # Expected for some invalid inputs
                print(f"Handled invalid input: {type(e).__name__}")

    def test_pipeline_performance_monitoring(self, pipeline_components):
        """Test pipeline with performance monitoring."""
        model = pipeline_components["model"]
        audio_processor = pipeline_components["audio_processor"]

        # Performance monitoring
        processing_times = {"audio_processing": [], "model_inference": [], "total_pipeline": []}

        # Test multiple samples
        num_samples = 10

        for i in range(num_samples):
            # Generate test audio
            duration = 1.0 + np.random.uniform(-0.2, 0.2)  # Varying duration
            t = np.linspace(0, duration, int(16000 * duration))
            raw_audio = np.sin(2 * np.pi * (440 + i * 10) * t).astype(np.float32)

            # Time entire pipeline
            total_start = time.perf_counter()

            # Time audio processing
            audio_start = time.perf_counter()
            processed_audio = audio_processor.process_audio(raw_audio)
            audio_end = time.perf_counter()
            processing_times["audio_processing"].append(audio_end - audio_start)

            # Time model inference
            inference_start = time.perf_counter()
            with torch.no_grad():
                predictions = model(processed_audio.unsqueeze(0))
            inference_end = time.perf_counter()
            processing_times["model_inference"].append(inference_end - inference_start)

            total_end = time.perf_counter()
            processing_times["total_pipeline"].append(total_end - total_start)

        # Analyze performance
        for stage, times in processing_times.items():
            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            print(f"{stage}: {avg_time:.2f}±{std_time:.2f}ms")

        # Performance assertions
        avg_total_time = np.mean(processing_times["total_pipeline"]) * 1000
        assert avg_total_time < 200, f"Pipeline too slow: {avg_total_time:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__])
