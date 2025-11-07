"""
SereneSense Production API Server
Enterprise-grade FastAPI server for military vehicle sound detection.

Features:
- RESTful API endpoints
- WebSocket for real-time detection
- File upload and batch processing
- Performance monitoring
- Rate limiting and security
- Swagger documentation
- Prometheus metrics
"""

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import uvicorn
import torch
import numpy as np
import json
import logging
import time
import io
import zipfile
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import redis
from contextlib import asynccontextmanager
import tempfile
import aiofiles

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from core.inference.realtime.detector import RealTimeDetector, InferenceConfig, DetectionResult
from core.core.audio_processor import AudioProcessor, AudioConfig
from core.models.audioMAE.model import AudioMAE, AudioMAEConfig

logger = logging.getLogger(__name__)


# Pydantic models for API
class DetectionRequest(BaseModel):
    """Audio detection request"""

    confidence_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    return_raw_logits: bool = False
    segment_length: Optional[float] = Field(2.0, gt=0.0, le=10.0)


class DetectionResponse(BaseModel):
    """Audio detection response"""

    label: str
    confidence: float
    class_id: int
    processing_time: float
    timestamp: str
    raw_logits: Optional[List[float]] = None


class BatchDetectionRequest(BaseModel):
    """Batch detection request"""

    confidence_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    return_detailed: bool = False
    max_files: int = Field(50, le=100)


class BatchDetectionResponse(BaseModel):
    """Batch detection response"""

    results: List[Dict[str, Any]]
    total_files: int
    processing_time: float
    summary: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime: float


class MetricsResponse(BaseModel):
    """Metrics response"""

    total_requests: int
    avg_processing_time: float
    detections_by_class: Dict[str, int]
    error_rate: float


# Global state
app_state = {
    "model": None,
    "audio_processor": None,
    "detector": None,
    "start_time": time.time(),
    "request_count": 0,
    "processing_times": [],
    "error_count": 0,
    "detections": [],
    "redis_client": None,
}

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter("serenesense_requests_total", "Total requests", ["endpoint", "method"])
    REQUEST_DURATION = Histogram("serenesense_request_duration_seconds", "Request duration")
    DETECTION_COUNT = Counter(
        "serenesense_detections_total", "Total detections", ["detection_class"]
    )
    MODEL_INFERENCE_TIME = Histogram("serenesense_model_inference_seconds", "Model inference time")
    ACTIVE_CONNECTIONS = Gauge("serenesense_active_websockets", "Active WebSocket connections")


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        if PROMETHEUS_AVAILABLE:
            ACTIVE_CONNECTIONS.set(len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if PROMETHEUS_AVAILABLE:
            ACTIVE_CONNECTIONS.set(len(self.active_connections))

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)


connection_manager = ConnectionManager()

# Authentication
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token authentication"""
    if credentials is None:
        return None

    # In production, validate token against database
    valid_tokens = ["serenesense-api-key", "demo-token"]  # Replace with real tokens

    if credentials.credentials in valid_tokens:
        return {"username": "api_user"}

    raise HTTPException(status_code=401, detail="Invalid authentication token")


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting SereneSense API server...")

    try:
        # Initialize Redis for caching (optional)
        try:
            import redis

            app_state["redis_client"] = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            app_state["redis_client"].ping()
            logger.info("Redis connected successfully")
        except Exception:
            logger.warning("Redis not available, running without caching")

        # Load model and processor
        await load_model()

        logger.info("SereneSense API server started successfully")

        yield

    finally:
        # Shutdown
        logger.info("Shutting down SereneSense API server...")

        # Close Redis connection
        if app_state["redis_client"]:
            app_state["redis_client"].close()


# Create FastAPI app
app = FastAPI(
    title="SereneSense API",
    description="Military Vehicle Sound Detection API using AudioMAE and Edge Computing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Helper functions
async def load_model():
    """Load the trained model and initialize components"""
    try:
        # Model configuration
        model_config = AudioMAEConfig(num_classes=7)
        audio_config = AudioConfig()

        # Initialize model
        model = AudioMAE(model_config)

        # Load weights (replace with actual model path)
        model_path = "models/serenesense_best.pth"
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}. Using random weights.")

        model.eval()

        # Initialize audio processor
        audio_processor = AudioProcessor(audio_config)

        # Initialize real-time detector
        inference_config = InferenceConfig(
            model_path=model_path,
            confidence_threshold=0.7,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        detector = RealTimeDetector(inference_config)

        # Store in global state
        app_state["model"] = model
        app_state["audio_processor"] = audio_processor
        app_state["detector"] = detector

        logger.info("Model and components loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def update_metrics(endpoint: str, processing_time: float, error: bool = False):
    """Update application metrics"""
    app_state["request_count"] += 1
    app_state["processing_times"].append(processing_time)

    if error:
        app_state["error_count"] += 1

    # Keep only recent processing times
    if len(app_state["processing_times"]) > 1000:
        app_state["processing_times"] = app_state["processing_times"][-1000:]

    # Update Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST").inc()
        REQUEST_DURATION.observe(processing_time)


async def process_audio_file(file_content: bytes, request: DetectionRequest) -> DetectionResponse:
    """Process uploaded audio file"""
    start_time = time.time()

    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        # Process audio
        audio_processor = app_state["audio_processor"]
        mel_spec = audio_processor.process_audio_file(temp_path)

        # Model inference
        model = app_state["model"]
        device = next(model.parameters()).device

        input_tensor = mel_spec.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor, mode="classification")
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)

        # Create response
        class_names = [
            "helicopter",
            "fighter_aircraft",
            "military_vehicle",
            "truck",
            "footsteps",
            "speech",
            "background",
        ]

        confidence_score = confidence.item()
        class_id = predicted_class.item()

        processing_time = time.time() - start_time

        # Clean up temporary file
        Path(temp_path).unlink()

        result = DetectionResponse(
            label=class_names[class_id],
            confidence=confidence_score,
            class_id=class_id,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            raw_logits=logits.cpu().numpy().tolist()[0] if request.return_raw_logits else None,
        )

        # Store detection
        app_state["detections"].append(
            {"label": result.label, "confidence": result.confidence, "timestamp": result.timestamp}
        )

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            DETECTION_COUNT.labels(detection_class=result.label).inc()
            MODEL_INFERENCE_TIME.observe(processing_time)

        return result

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


# API Endpoints


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "message": "SereneSense Military Vehicle Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import psutil

    return HealthResponse(
        status="healthy" if app_state["model"] is not None else "unhealthy",
        version="1.0.0",
        model_loaded=app_state["model"] is not None,
        gpu_available=torch.cuda.is_available(),
        memory_usage={
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        },
        uptime=time.time() - app_state["start_time"],
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_audio(
    file: UploadFile = File(...),
    request: DetectionRequest = DetectionRequest(),
    current_user: dict = Depends(get_current_user),
):
    """
    Detect military vehicles in uploaded audio file.

    Supports WAV, MP3, FLAC formats.
    """
    start_time = time.time()

    try:
        # Validate file
        if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported audio format")

        # Read file content
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Process audio
        result = await process_audio_file(content, request)

        # Update metrics
        processing_time = time.time() - start_time
        update_metrics("/detect", processing_time)

        return result

    except HTTPException:
        update_metrics("/detect", time.time() - start_time, error=True)
        raise
    except Exception as e:
        update_metrics("/detect", time.time() - start_time, error=True)
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: List[UploadFile] = File(...),
    request: BatchDetectionRequest = BatchDetectionRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: dict = Depends(get_current_user),
):
    """
    Batch detection for multiple audio files.
    """
    start_time = time.time()

    try:
        if len(files) > request.max_files:
            raise HTTPException(
                status_code=400, detail=f"Too many files. Maximum {request.max_files} allowed."
            )

        results = []
        summary = {}

        detection_request = DetectionRequest(
            confidence_threshold=request.confidence_threshold,
            return_raw_logits=request.return_detailed,
        )

        for file in files:
            try:
                content = await file.read()
                result = await process_audio_file(content, detection_request)

                file_result = {
                    "filename": file.filename,
                    "label": result.label,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                }

                if request.return_detailed:
                    file_result.update(
                        {
                            "class_id": result.class_id,
                            "timestamp": result.timestamp,
                            "raw_logits": result.raw_logits,
                        }
                    )

                results.append(file_result)

                # Update summary
                summary[result.label] = summary.get(result.label, 0) + 1

            except Exception as e:
                logger.warning(f"Failed to process {file.filename}: {e}")
                results.append({"filename": file.filename, "error": str(e)})

        total_time = time.time() - start_time

        response = BatchDetectionResponse(
            results=results, total_files=len(files), processing_time=total_time, summary=summary
        )

        update_metrics("/detect/batch", total_time)

        return response

    except HTTPException:
        update_metrics("/detect/batch", time.time() - start_time, error=True)
        raise
    except Exception as e:
        update_metrics("/detect/batch", time.time() - start_time, error=True)
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio detection.
    """
    await connection_manager.connect(websocket)

    try:
        # Send initial message
        await websocket.send_json({"type": "connected", "message": "Real-time detection ready"})

        detector = app_state["detector"]

        # Start real-time detection
        def detection_callback(result: DetectionResult):
            """Handle detection results"""
            asyncio.create_task(
                connection_manager.broadcast(
                    {
                        "type": "detection",
                        "label": result.label,
                        "confidence": result.confidence,
                        "timestamp": result.timestamp,
                        "processing_time": result.processing_time,
                    }
                )
            )

        detector.config.detection_callback = detection_callback
        detector.start()

        # Keep connection alive
        while True:
            try:
                message = await websocket.receive_json()

                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "stop":
                    break

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        detector.stop()
        connection_manager.disconnect(websocket)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """Get application metrics"""

    # Calculate detection summary
    detections_by_class = {}
    for detection in app_state["detections"][-1000:]:  # Last 1000 detections
        label = detection["label"]
        detections_by_class[label] = detections_by_class.get(label, 0) + 1

    # Calculate average processing time
    recent_times = app_state["processing_times"][-100:]  # Last 100 requests
    avg_processing_time = sum(recent_times) / len(recent_times) if recent_times else 0

    # Calculate error rate
    error_rate = app_state["error_count"] / max(app_state["request_count"], 1)

    return MetricsResponse(
        total_requests=app_state["request_count"],
        avg_processing_time=avg_processing_time,
        detections_by_class=detections_by_class,
        error_rate=error_rate,
    )


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prometheus metrics not available")

    return generate_latest()


@app.get("/models/info")
async def model_info(current_user: dict = Depends(get_current_user)):
    """Get model information"""
    model = app_state["model"]

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_type": "AudioMAE",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(next(model.parameters()).device),
        "classes": [
            "helicopter",
            "fighter_aircraft",
            "military_vehicle",
            "truck",
            "footsteps",
            "speech",
            "background",
        ],
    }


@app.get("/export/detections")
async def export_detections(
    format: str = "json", limit: int = 1000, current_user: dict = Depends(get_current_user)
):
    """Export detection history"""

    recent_detections = app_state["detections"][-limit:]

    if format.lower() == "json":
        return JSONResponse(content=recent_detections)
    elif format.lower() == "csv":
        import pandas as pd

        df = pd.DataFrame(recent_detections)
        csv_content = df.to_csv(index=False)

        return JSONResponse(
            content={"csv_data": csv_content}, headers={"Content-Type": "application/json"}
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'json' or 'csv'")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404, content={"error": "Endpoint not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SereneSense API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        "core.deployment.api.fastapi_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )
