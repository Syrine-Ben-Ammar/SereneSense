"""
SereneSense API Module

This module provides REST and WebSocket APIs for military vehicle sound detection.

Features:
- FastAPI-based REST API for batch processing
- WebSocket API for real-time streaming
- Authentication and authorization
- Rate limiting and request validation
- Comprehensive API documentation
- Health checks and metrics endpoints
"""

from .fastapi_server import SereneSenseAPI, create_api_server
from .websocket_handler import WebSocketHandler
from .auth import AuthenticationManager
from .rate_limiter import RateLimiter

__all__ = [
    "SereneSenseAPI",
    "create_api_server",
    "WebSocketHandler",
    "AuthenticationManager",
    "RateLimiter",
]

__version__ = "1.0.0"
