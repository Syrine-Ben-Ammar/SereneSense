"""
SereneSense Inference Module

This module provides comprehensive inference capabilities for the SereneSense
military vehicle sound detection system, supporting both real-time and batch
inference with advanced optimization techniques.

Key Features:
- Real-time inference with circular buffering
- Batch inference for large-scale processing
- Model optimization (quantization, TensorRT, ONNX)
- Edge deployment optimizations
- Multi-model ensemble inference
- Streaming audio processing
- Low-latency optimizations

Components:
    - real_time: Real-time inference engine with sub-20ms latency
    - batch: Batch processing for large audio datasets
    - optimization: Model optimization and acceleration
    - ensemble: Multi-model ensemble inference
    - streaming: Streaming audio processing utilities

Performance Features:
- Hardware acceleration (CUDA, TensorRT, OpenVINO)
- Memory optimization for edge devices
- Automatic mixed precision inference
- Dynamic batching for throughput optimization
- Model caching and preloading
"""

from .real_time import (
    RealTimeInferenceEngine,
    StreamingProcessor,
    CircularAudioBuffer,
    LatencyTracker,
    AudioStreamConfig,
)

from .batch import (
    BatchInferenceEngine,
    DatasetProcessor,
    BatchConfig,
    ResultsAggregator,
    ParallelProcessor,
)

from .optimization import (
    ModelOptimizer,
    QuantizationConfig,
    TensorRTOptimizer,
    ONNXOptimizer,
    OpenVINOOptimizer,
    OptimizationMetrics,
)

from .ensemble import (
    EnsembleInferenceEngine,
    VotingStrategy,
    WeightedEnsemble,
    DynamicEnsemble,
    EnsembleConfig,
)

from .streaming import (
    AudioStreamManager,
    StreamingConfig,
    AudioChunker,
    OverlapProcessor,
    StreamingMetrics,
)

__all__ = [
    # Real-time inference
    "RealTimeInferenceEngine",
    "StreamingProcessor",
    "CircularAudioBuffer",
    "LatencyTracker",
    "AudioStreamConfig",
    # Batch inference
    "BatchInferenceEngine",
    "DatasetProcessor",
    "BatchConfig",
    "ResultsAggregator",
    "ParallelProcessor",
    # Model optimization
    "ModelOptimizer",
    "QuantizationConfig",
    "TensorRTOptimizer",
    "ONNXOptimizer",
    "OpenVINOOptimizer",
    "OptimizationMetrics",
    # Ensemble inference
    "EnsembleInferenceEngine",
    "VotingStrategy",
    "WeightedEnsemble",
    "DynamicEnsemble",
    "EnsembleConfig",
    # Streaming processing
    "AudioStreamManager",
    "StreamingConfig",
    "AudioChunker",
    "OverlapProcessor",
    "StreamingMetrics",
]

# Inference configuration presets
INFERENCE_PRESETS = {
    "real_time_jetson": {
        "backend": "tensorrt",
        "precision": "fp16",
        "max_batch_size": 1,
        "optimization_level": "O2",
        "memory_optimization": True,
        "target_latency_ms": 10.0,
        "buffer_size_ms": 500.0,
    },
    "real_time_raspberry_pi": {
        "backend": "pytorch",
        "precision": "int8",
        "max_batch_size": 1,
        "optimization_level": "O1",
        "memory_optimization": True,
        "target_latency_ms": 20.0,
        "buffer_size_ms": 1000.0,
    },
    "batch_server": {
        "backend": "pytorch",
        "precision": "fp32",
        "max_batch_size": 64,
        "optimization_level": "O1",
        "memory_optimization": False,
        "parallel_workers": 8,
        "gpu_memory_fraction": 0.8,
    },
    "ensemble_high_accuracy": {
        "models": ["audioMAE", "ast", "beats"],
        "voting_strategy": "weighted",
        "confidence_threshold": 0.8,
        "optimization_level": "O1",
        "ensemble_method": "late_fusion",
    },
    "streaming_production": {
        "backend": "onnx",
        "precision": "fp16",
        "chunk_size_ms": 100.0,
        "overlap_ratio": 0.25,
        "max_concurrent_streams": 10,
        "optimization_level": "O2",
    },
}


def get_inference_preset(preset_name: str, **overrides):
    """
    Get an inference configuration preset with optional overrides.

    Args:
        preset_name: Name of the preset configuration
        **overrides: Parameters to override in the preset

    Returns:
        Inference configuration dictionary
    """
    if preset_name not in INFERENCE_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(INFERENCE_PRESETS.keys())}"
        )

    config = INFERENCE_PRESETS[preset_name].copy()
    config.update(overrides)
    return config


def create_inference_engine(
    model_path: str, config_preset: str = None, inference_type: str = "real_time", **kwargs
):
    """
    Create an inference engine with specified configuration.

    Args:
        model_path: Path to the trained model
        config_preset: Name of configuration preset to use
        inference_type: Type of inference ('real_time', 'batch', 'ensemble')
        **kwargs: Additional configuration parameters

    Returns:
        Configured inference engine
    """
    # Get preset configuration
    if config_preset:
        config = get_inference_preset(config_preset, **kwargs)
    else:
        config = kwargs

    # Create appropriate inference engine
    if inference_type == "real_time":
        from .real_time import RealTimeInferenceEngine

        return RealTimeInferenceEngine(model_path, **config)

    elif inference_type == "batch":
        from .batch import BatchInferenceEngine

        return BatchInferenceEngine(model_path, **config)

    elif inference_type == "ensemble":
        from .ensemble import EnsembleInferenceEngine

        return EnsembleInferenceEngine(model_path, **config)

    else:
        raise ValueError(f"Unknown inference type: {inference_type}")


# Performance benchmarking utilities
BENCHMARK_CONFIGS = {
    "latency_test": {
        "num_samples": 1000,
        "warmup_iterations": 100,
        "measure_memory": True,
        "measure_throughput": True,
        "input_sizes": [(128, 128), (256, 256)],
        "batch_sizes": [1, 4, 8, 16],
    },
    "throughput_test": {
        "duration_seconds": 60,
        "target_qps": [10, 50, 100, 200],
        "concurrent_requests": [1, 4, 8, 16],
        "measure_accuracy": True,
    },
    "memory_test": {
        "profile_memory": True,
        "measure_peak_usage": True,
        "track_allocations": True,
        "test_batch_sizes": [1, 8, 32, 64],
    },
}


def run_inference_benchmark(engine, benchmark_type: str = "latency_test", **config_overrides):
    """
    Run inference benchmarks on an engine.

    Args:
        engine: Inference engine to benchmark
        benchmark_type: Type of benchmark to run
        **config_overrides: Override benchmark configuration

    Returns:
        Benchmark results dictionary
    """
    from .benchmark import InferenceBenchmark

    config = BENCHMARK_CONFIGS.get(benchmark_type, {})
    config.update(config_overrides)

    benchmark = InferenceBenchmark(engine, **config)
    return benchmark.run()


# Hardware compatibility matrix
HARDWARE_COMPATIBILITY = {
    "jetson_orin_nano": {
        "supported_backends": ["pytorch", "tensorrt", "onnx"],
        "recommended_precision": "fp16",
        "max_memory_gb": 8,
        "compute_capability": 8.7,
        "recommended_batch_size": 4,
    },
    "jetson_xavier_nx": {
        "supported_backends": ["pytorch", "tensorrt", "onnx"],
        "recommended_precision": "fp16",
        "max_memory_gb": 8,
        "compute_capability": 7.2,
        "recommended_batch_size": 2,
    },
    "raspberry_pi_5": {
        "supported_backends": ["pytorch", "onnx", "openvino"],
        "recommended_precision": "int8",
        "max_memory_gb": 8,
        "cpu_cores": 4,
        "recommended_batch_size": 1,
    },
    "intel_nuc": {
        "supported_backends": ["pytorch", "onnx", "openvino"],
        "recommended_precision": "fp32",
        "max_memory_gb": 32,
        "cpu_cores": 8,
        "recommended_batch_size": 8,
    },
    "aws_gpu": {
        "supported_backends": ["pytorch", "tensorrt", "onnx"],
        "recommended_precision": "fp32",
        "max_memory_gb": 16,
        "compute_capability": 8.6,
        "recommended_batch_size": 32,
    },
}


def get_hardware_recommendations(hardware_type: str):
    """
    Get recommended inference configuration for specific hardware.

    Args:
        hardware_type: Type of hardware platform

    Returns:
        Recommended configuration dictionary
    """
    if hardware_type not in HARDWARE_COMPATIBILITY:
        raise ValueError(
            f"Unknown hardware: {hardware_type}. Available: {list(HARDWARE_COMPATIBILITY.keys())}"
        )

    return HARDWARE_COMPATIBILITY[hardware_type].copy()


def optimize_for_hardware(
    model_path: str, hardware_type: str, optimization_level: str = "balanced"
):
    """
    Optimize model for specific hardware platform.

    Args:
        model_path: Path to the model to optimize
        hardware_type: Target hardware platform
        optimization_level: Level of optimization ('speed', 'balanced', 'accuracy')

    Returns:
        Path to optimized model
    """
    from .optimization import ModelOptimizer

    # Get hardware recommendations
    hw_config = get_hardware_recommendations(hardware_type)

    # Create optimizer with hardware-specific settings
    optimizer = ModelOptimizer(
        target_backend=hw_config["supported_backends"][0],
        precision=hw_config["recommended_precision"],
        optimization_level=optimization_level,
        max_batch_size=hw_config["recommended_batch_size"],
    )

    # Optimize model
    optimized_path = optimizer.optimize(model_path)
    return optimized_path


# Edge deployment utilities
EDGE_CONFIGS = {
    "ultra_low_latency": {
        "max_latency_ms": 5.0,
        "memory_limit_mb": 512,
        "cpu_usage_limit": 0.8,
        "precision": "int8",
        "optimization_aggressive": True,
    },
    "low_power": {
        "power_limit_watts": 5.0,
        "memory_limit_mb": 1024,
        "cpu_usage_limit": 0.5,
        "precision": "int8",
        "sleep_between_inferences": True,
    },
    "high_accuracy": {
        "precision": "fp32",
        "ensemble_models": True,
        "confidence_threshold": 0.95,
        "memory_limit_mb": 4096,
    },
}


def deploy_to_edge(
    model_path: str,
    deployment_type: str = "ultra_low_latency",
    target_device: str = "jetson_orin_nano",
):
    """
    Deploy model to edge device with optimized configuration.

    Args:
        model_path: Path to model to deploy
        deployment_type: Type of edge deployment
        target_device: Target edge device

    Returns:
        Edge deployment configuration
    """
    edge_config = EDGE_CONFIGS.get(deployment_type, {})
    hw_config = get_hardware_recommendations(target_device)

    # Merge configurations
    deployment_config = {**edge_config, **hw_config}

    # Optimize model for edge deployment
    optimized_model = optimize_for_hardware(model_path, target_device, "speed")

    return {
        "optimized_model_path": optimized_model,
        "deployment_config": deployment_config,
        "recommended_settings": edge_config,
    }


# Version and compatibility information
__version__ = "1.0.0"
__author__ = "SereneSense Team"
__description__ = "High-performance inference engine for military vehicle sound detection"

# Supported model formats
SUPPORTED_MODEL_FORMATS = [
    ".pth",  # PyTorch
    ".onnx",  # ONNX
    ".trt",  # TensorRT
    ".xml",  # OpenVINO
    ".pb",  # TensorFlow
]

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
