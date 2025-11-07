"""
SereneSense Deployment Module

This module provides comprehensive deployment solutions for military vehicle 
sound detection across various platforms and environments.

Components:
- FastAPI server for REST and WebSocket APIs
- Edge deployment for Jetson Orin Nano and Raspberry Pi
- Docker containerization for scalable deployment
- Monitoring and health checks
- CI/CD integration utilities
"""

from .api.fastapi_server import SereneSenseAPI, create_api_server
from .edge.jetson_deployment import JetsonDeployment
from .edge.raspberry_pi_deployment import RaspberryPiDeployment
from .monitoring.health_check import HealthChecker
from .monitoring.metrics import MetricsCollector

__all__ = [
    "SereneSenseAPI",
    "create_api_server",
    "JetsonDeployment",
    "RaspberryPiDeployment",
    "HealthChecker",
    "MetricsCollector",
]

__version__ = "1.0.0"
