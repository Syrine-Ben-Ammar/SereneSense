"""
SereneSense Model Optimization Module

This module provides optimizations for edge deployment including:
- TensorRT optimization for NVIDIA hardware
- ONNX export and optimization  
- Model quantization (INT8, FP16)
- Model pruning and compression
- Edge-specific optimizations for Jetson and Raspberry Pi
"""

from .tensorrt import TensorRTOptimizer, TensorRTConfig
from .onnx_export import ONNXExporter, ONNXConfig
from .quantization import QuantizationOptimizer, QuantizationConfig
from .pruning import PruningOptimizer, PruningConfig

__all__ = [
    "TensorRTOptimizer",
    "TensorRTConfig",
    "ONNXExporter",
    "ONNXConfig",
    "QuantizationOptimizer",
    "QuantizationConfig",
    "PruningOptimizer",
    "PruningConfig",
]

__version__ = "1.0.0"
