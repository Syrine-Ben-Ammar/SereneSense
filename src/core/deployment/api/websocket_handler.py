#
# Plan:
# 1. Create dedicated WebSocket handler for real-time audio streaming
# 2. Support multiple connection types (live audio, file streaming, batch)
# 3. Audio format handling and real-time buffering
# 4. Connection management and cleanup
# 5. Message protocol for configuration and control
# 6. Error handling and reconnection support
# 7. Performance monitoring and metrics
#

"""
WebSocket Handler for Real-time Military Vehicle Detection
Handles real-time audio streaming and detection via WebSocket connections.

Features:
- Real-time audio streaming from clients
- Multiple audio formats support
- Connection management and cleanup
- Configurable detection parameters
- Error handling and recovery
- Performance monitoring
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import base64
import struct

from fastapi import WebSocket, WebSocketDisconnect
import torch
import numpy as np
import torchaudio

from core.inference.real_time import RealTimeInference, InferenceConfig, DetectionResult
from core.core.audio_processor import AudioProcessor
from core.utils.device_utils import get_optimal_device

logger = logging.getLogger(__name__)


class WebSocketMessage:
    """WebSocket message types and structure"""

    # Message types
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    AUDIO_DATA = "audio_data"
    AUDIO_FILE = "audio_file"
    CONFIG_UPDATE = "config_update"
    START_DETECTION = "start_detection"
    STOP_DETECTION = "stop_detection"
    PING = "ping"
    PONG = "pong"

    # Response types
    DETECTION = "detection"
    STATUS = "status"
    ERROR = "error"
    ACK = "ack"


class AudioBuffer:
    """
    Circular audio buffer for real-time streaming.
    Handles audio format conversion and buffering.
    """

    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 10.0):
        """
        Initialize audio buffer.

        Args:
            sample_rate: Target sample rate
            buffer_duration: Buffer duration in seconds
        """
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0

        # Audio processing
        self.audio_processor = AudioProcessor(
            {
                "sample_rate": sample_rate,
                "n_mels": 128,
                "n_fft": 1024,
                "hop_length": 512,
                "win_length": 1024,
                "normalize": True,
            }
        )

    def add_audio_data(self, audio_data: np.ndarray, source_sample_rate: int = None):
        """
        Add audio data to buffer.

        Args:
            audio_data: Audio samples
            source_sample_rate: Original sample rate (if different from target)
        """
        # Resample if needed
        if source_sample_rate and source_sample_rate != self.sample_rate:
            audio_data = self._resample_audio(audio_data, source_sample_rate, self.sample_rate)

        # Ensure float32 format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Add to circular buffer
        data_length = len(audio_data)
        end_pos = self.write_pos + data_length

        if end_pos <= self.buffer_size:
            # Simple case: no wraparound
            self.buffer[self.write_pos : end_pos] = audio_data
        else:
            # Wraparound case
            first_part = self.buffer_size - self.write_pos
            self.buffer[self.write_pos :] = audio_data[:first_part]
            self.buffer[: end_pos - self.buffer_size] = audio_data[first_part:]

        self.write_pos = end_pos % self.buffer_size
        self.samples_written += data_length

    def get_latest_window(self, window_duration: float = 2.0) -> Optional[np.ndarray]:
        """
        Get latest audio window for processing.

        Args:
            window_duration: Window duration in seconds

        Returns:
            Audio window or None if insufficient data
        """
        window_samples = int(window_duration * self.sample_rate)

        if self.samples_written < window_samples:
            return None

        # Extract latest window
        start_pos = (self.write_pos - window_samples) % self.buffer_size
        end_pos = self.write_pos

        if start_pos < end_pos:
            # Simple case: no wraparound
            window = self.buffer[start_pos:end_pos].copy()
        else:
            # Wraparound case
            window = np.concatenate([self.buffer[start_pos:], self.buffer[:end_pos]])

        return window

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio

        # Convert to tensor for resampling
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        # Resample
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = resampler(audio_tensor)

        return resampled.squeeze(0).numpy()


class ConnectionManager:
    """Manages WebSocket connections and their state"""

    def __init__(self):
        """Initialize connection manager"""
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.active_detectors: Dict[str, RealTimeInference] = {}

    async def connect(self, websocket: WebSocket, connection_id: str):
        """
        Accept new WebSocket connection.

        Args:
            websocket: WebSocket instance
            connection_id: Unique connection identifier
        """
        await websocket.accept()

        self.connections[connection_id] = {
            "websocket": websocket,
            "connected_at": time.time(),
            "audio_buffer": AudioBuffer(),
            "config": {
                "sample_rate": 16000,
                "confidence_threshold": 0.7,
                "detection_active": False,
            },
            "stats": {
                "messages_received": 0,
                "detections_sent": 0,
                "bytes_received": 0,
                "errors": 0,
            },
        }

        logger.info(f"WebSocket connection established: {connection_id}")

        # Send welcome message
        await self.send_message(
            connection_id,
            {
                "type": WebSocketMessage.STATUS,
                "data": {
                    "connection_id": connection_id,
                    "status": "connected",
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

    async def disconnect(self, connection_id: str):
        """
        Disconnect WebSocket connection.

        Args:
            connection_id: Connection identifier
        """
        if connection_id in self.connections:
            # Stop detection if active
            if connection_id in self.active_detectors:
                self.active_detectors[connection_id].stop()
                del self.active_detectors[connection_id]

            # Remove connection
            del self.connections[connection_id]

            logger.info(f"WebSocket connection closed: {connection_id}")

    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """
        Send message to specific connection.

        Args:
            connection_id: Connection identifier
            message: Message to send
        """
        if connection_id not in self.connections:
            return

        try:
            websocket = self.connections[connection_id]["websocket"]
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id)

    async def broadcast_message(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """
        Broadcast message to all connections.

        Args:
            message: Message to broadcast
            exclude: Connection ID to exclude from broadcast
        """
        tasks = []
        for connection_id in self.connections:
            if connection_id != exclude:
                task = self.send_message(connection_id, message)
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics for all connections"""
        stats = {
            "total_connections": len(self.connections),
            "active_detections": len(self.active_detectors),
            "connections": {},
        }

        for connection_id, conn_data in self.connections.items():
            stats["connections"][connection_id] = {
                "connected_at": conn_data["connected_at"],
                "uptime_seconds": time.time() - conn_data["connected_at"],
                "stats": conn_data["stats"],
                "config": conn_data["config"],
            }

        return stats


class WebSocketHandler:
    """
    WebSocket handler for real-time military vehicle detection.
    Manages connections, audio streaming, and real-time inference.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize WebSocket handler.

        Args:
            model_path: Path to detection model
            device: Device for inference
        """
        self.model_path = model_path
        self.device = get_optimal_device(device)
        self.connection_manager = ConnectionManager()

        logger.info(f"WebSocket handler initialized with model: {model_path}")

    async def handle_connection(self, websocket: WebSocket):
        """
        Handle new WebSocket connection.

        Args:
            websocket: WebSocket instance
        """
        connection_id = str(uuid.uuid4())

        try:
            # Accept connection
            await self.connection_manager.connect(websocket, connection_id)

            # Handle messages
            while True:
                try:
                    # Receive message
                    message_text = await websocket.receive_text()
                    await self._handle_message(connection_id, message_text)

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling message from {connection_id}: {e}")
                    await self._send_error(connection_id, str(e))

        finally:
            # Clean up connection
            await self.connection_manager.disconnect(connection_id)

    async def _handle_message(self, connection_id: str, message_text: str):
        """
        Handle incoming WebSocket message.

        Args:
            connection_id: Connection identifier
            message_text: Message content
        """
        try:
            message = json.loads(message_text)
            message_type = message.get("type")

            # Update stats
            conn_data = self.connection_manager.connections[connection_id]
            conn_data["stats"]["messages_received"] += 1
            conn_data["stats"]["bytes_received"] += len(message_text)

            # Handle different message types
            if message_type == WebSocketMessage.AUDIO_DATA:
                await self._handle_audio_data(connection_id, message)

            elif message_type == WebSocketMessage.AUDIO_FILE:
                await self._handle_audio_file(connection_id, message)

            elif message_type == WebSocketMessage.CONFIG_UPDATE:
                await self._handle_config_update(connection_id, message)

            elif message_type == WebSocketMessage.START_DETECTION:
                await self._start_detection(connection_id)

            elif message_type == WebSocketMessage.STOP_DETECTION:
                await self._stop_detection(connection_id)

            elif message_type == WebSocketMessage.PING:
                await self._handle_ping(connection_id)

            else:
                await self._send_error(connection_id, f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            await self._send_error(connection_id, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._send_error(connection_id, str(e))

    async def _handle_audio_data(self, connection_id: str, message: Dict[str, Any]):
        """Handle real-time audio data"""
        try:
            data = message.get("data", {})

            # Decode audio data
            if "audio_base64" in data:
                # Base64 encoded audio
                audio_bytes = base64.b64decode(data["audio_base64"])
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

            elif "audio_samples" in data:
                # Direct sample values
                audio_data = np.array(data["audio_samples"], dtype=np.float32)

            else:
                await self._send_error(connection_id, "No audio data found in message")
                return

            # Get connection data
            conn_data = self.connection_manager.connections[connection_id]
            sample_rate = data.get("sample_rate", conn_data["config"]["sample_rate"])

            # Add to audio buffer
            conn_data["audio_buffer"].add_audio_data(audio_data, sample_rate)

            # Process audio if detection is active
            if (
                conn_data["config"]["detection_active"]
                and connection_id in self.connection_manager.active_detectors
            ):
                await self._process_audio_window(connection_id)

            # Send acknowledgment
            await self.connection_manager.send_message(
                connection_id,
                {
                    "type": WebSocketMessage.ACK,
                    "data": {
                        "message_type": WebSocketMessage.AUDIO_DATA,
                        "samples_received": len(audio_data),
                        "timestamp": time.time(),
                    },
                },
            )

        except Exception as e:
            await self._send_error(connection_id, f"Error processing audio data: {e}")

    async def _handle_audio_file(self, connection_id: str, message: Dict[str, Any]):
        """Handle audio file upload"""
        try:
            data = message.get("data", {})

            # Decode file data
            file_data = base64.b64decode(data["file_base64"])
            file_format = data.get("format", "wav")

            # Process file (simplified - in production would save to temp file)
            # For now, treat as raw audio samples
            if file_format in ["wav", "raw"]:
                audio_data = np.frombuffer(file_data, dtype=np.float32)

                # Get connection data
                conn_data = self.connection_manager.connections[connection_id]
                sample_rate = data.get("sample_rate", conn_data["config"]["sample_rate"])

                # Add to buffer
                conn_data["audio_buffer"].add_audio_data(audio_data, sample_rate)

                # Send acknowledgment
                await self.connection_manager.send_message(
                    connection_id,
                    {
                        "type": WebSocketMessage.ACK,
                        "data": {
                            "message_type": WebSocketMessage.AUDIO_FILE,
                            "file_size": len(file_data),
                            "samples_loaded": len(audio_data),
                            "timestamp": time.time(),
                        },
                    },
                )

            else:
                await self._send_error(connection_id, f"Unsupported file format: {file_format}")

        except Exception as e:
            await self._send_error(connection_id, f"Error processing audio file: {e}")

    async def _handle_config_update(self, connection_id: str, message: Dict[str, Any]):
        """Handle configuration update"""
        try:
            config_updates = message.get("data", {})
            conn_data = self.connection_manager.connections[connection_id]

            # Update configuration
            for key, value in config_updates.items():
                if key in conn_data["config"]:
                    conn_data["config"][key] = value

            # Send acknowledgment
            await self.connection_manager.send_message(
                connection_id,
                {
                    "type": WebSocketMessage.ACK,
                    "data": {
                        "message_type": WebSocketMessage.CONFIG_UPDATE,
                        "updated_config": conn_data["config"],
                        "timestamp": time.time(),
                    },
                },
            )

        except Exception as e:
            await self._send_error(connection_id, f"Error updating config: {e}")

    async def _start_detection(self, connection_id: str):
        """Start real-time detection for connection"""
        try:
            conn_data = self.connection_manager.connections[connection_id]

            if connection_id in self.connection_manager.active_detectors:
                await self._send_error(connection_id, "Detection already active")
                return

            # Create detection callback
            def detection_callback(result: DetectionResult):
                asyncio.create_task(self._send_detection(connection_id, result))

            # Create real-time inference
            config = InferenceConfig(
                model_path=self.model_path,
                device=self.device,
                confidence_threshold=conn_data["config"]["confidence_threshold"],
                detection_callback=detection_callback,
            )

            detector = RealTimeInference(config)
            detector.start()

            self.connection_manager.active_detectors[connection_id] = detector
            conn_data["config"]["detection_active"] = True

            # Send status update
            await self.connection_manager.send_message(
                connection_id,
                {
                    "type": WebSocketMessage.STATUS,
                    "data": {"detection_status": "started", "timestamp": time.time()},
                },
            )

        except Exception as e:
            await self._send_error(connection_id, f"Error starting detection: {e}")

    async def _stop_detection(self, connection_id: str):
        """Stop real-time detection for connection"""
        try:
            if connection_id in self.connection_manager.active_detectors:
                self.connection_manager.active_detectors[connection_id].stop()
                del self.connection_manager.active_detectors[connection_id]

            conn_data = self.connection_manager.connections[connection_id]
            conn_data["config"]["detection_active"] = False

            # Send status update
            await self.connection_manager.send_message(
                connection_id,
                {
                    "type": WebSocketMessage.STATUS,
                    "data": {"detection_status": "stopped", "timestamp": time.time()},
                },
            )

        except Exception as e:
            await self._send_error(connection_id, f"Error stopping detection: {e}")

    async def _handle_ping(self, connection_id: str):
        """Handle ping message"""
        await self.connection_manager.send_message(
            connection_id, {"type": WebSocketMessage.PONG, "data": {"timestamp": time.time()}}
        )

    async def _process_audio_window(self, connection_id: str):
        """Process audio window for detection"""
        try:
            conn_data = self.connection_manager.connections[connection_id]
            audio_window = conn_data["audio_buffer"].get_latest_window()

            if (
                audio_window is not None
                and connection_id in self.connection_manager.active_detectors
            ):
                # The actual detection will be handled by the real-time detector
                # This is just a placeholder for additional processing if needed
                pass

        except Exception as e:
            logger.error(f"Error processing audio window for {connection_id}: {e}")

    async def _send_detection(self, connection_id: str, detection: DetectionResult):
        """Send detection result to client"""
        try:
            # Update stats
            conn_data = self.connection_manager.connections[connection_id]
            conn_data["stats"]["detections_sent"] += 1

            # Send detection
            await self.connection_manager.send_message(
                connection_id,
                {
                    "type": WebSocketMessage.DETECTION,
                    "data": {
                        "detection_id": str(uuid.uuid4()),
                        "timestamp": detection.timestamp,
                        "label": detection.label,
                        "confidence": detection.confidence,
                        "segment_start": detection.audio_segment_start,
                        "segment_end": detection.audio_segment_end,
                        "processing_time_ms": detection.processing_time * 1000,
                    },
                },
            )

        except Exception as e:
            logger.error(f"Error sending detection to {connection_id}: {e}")

    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to client"""
        try:
            # Update stats
            conn_data = self.connection_manager.connections[connection_id]
            conn_data["stats"]["errors"] += 1

            await self.connection_manager.send_message(
                connection_id,
                {
                    "type": WebSocketMessage.ERROR,
                    "data": {"error": error_message, "timestamp": time.time()},
                },
            )
        except Exception as e:
            logger.error(f"Error sending error message to {connection_id}: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return self.connection_manager.get_connection_stats()


# Example WebSocket client for testing
class WebSocketClient:
    """Example WebSocket client for testing"""

    def __init__(self, url: str):
        """Initialize WebSocket client"""
        self.url = url
        self.websocket = None

    async def connect(self):
        """Connect to WebSocket server"""
        import websockets

        self.websocket = await websockets.connect(self.url)
        logger.info(f"Connected to {self.url}")

    async def send_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Send audio data to server"""
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket server")

        # Encode audio data
        audio_base64 = base64.b64encode(audio_data.tobytes()).decode("utf-8")

        message = {
            "type": WebSocketMessage.AUDIO_DATA,
            "data": {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "timestamp": time.time(),
            },
        }

        await self.websocket.send(json.dumps(message))

    async def start_detection(self):
        """Start detection"""
        message = {"type": WebSocketMessage.START_DETECTION, "data": {}}
        await self.websocket.send(json.dumps(message))

    async def listen_for_detections(self):
        """Listen for detection results"""
        while True:
            try:
                message_text = await self.websocket.recv()
                message = json.loads(message_text)

                if message["type"] == WebSocketMessage.DETECTION:
                    detection = message["data"]
                    print(f"🚨 Detection: {detection['label']} ({detection['confidence']:.2f})")

                elif message["type"] == WebSocketMessage.ERROR:
                    print(f"❌ Error: {message['data']['error']}")

            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break

    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None


if __name__ == "__main__":
    # Demo: WebSocket handler
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket Handler Demo")
    parser.add_argument("--model-path", required=True, help="Path to detection model")
    parser.add_argument("--device", default="auto", help="Device for inference")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Create WebSocket handler
        handler = WebSocketHandler(args.model_path, args.device)

        print("🎧 WebSocket handler initialized")
        print("   Use this handler in your FastAPI application")
        print("   Example endpoint: @app.websocket('/ws/realtime')")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"WebSocket handler initialization failed: {e}")
