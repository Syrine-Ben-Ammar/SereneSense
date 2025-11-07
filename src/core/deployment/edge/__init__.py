"""
SereneSense Edge Deployment Module

This module provides deployment solutions for edge devices including
NVIDIA Jetson Orin Nano and Raspberry Pi platforms.

Features:
- Jetson Orin Nano deployment with TensorRT optimization
- Raspberry Pi deployment with ONNX optimization  
- Hardware-specific optimizations and configurations
- Power management and thermal monitoring
- Edge-specific model optimization
- Device health monitoring and diagnostics
"""

from .jetson_deployment import JetsonDeployment, JetsonConfig
from .raspberry_pi_deployment import RaspberryPiDeployment, RaspberryPiConfig
from .edge_optimizer import EdgeOptimizer, EdgeConfig

__all__ = [
    "JetsonDeployment",
    "JetsonConfig",
    "RaspberryPiDeployment",
    "RaspberryPiConfig",
    "EdgeOptimizer",
    "EdgeConfig",
]

__version__ = "1.0.0"
