#
# Plan:
# 1. Create comprehensive health checking system for SereneSense deployment
# 2. Monitor system components (model, hardware, API, database)
# 3. Health status reporting with detailed diagnostics
# 4. Automated health checks with configurable intervals
# 5. Alert generation for critical health issues
# 6. Recovery suggestions and troubleshooting guidance
# 7. Integration with external monitoring systems
#

"""
Health Checker for SereneSense Deployment
Monitors system health and provides diagnostic information.

Features:
- Comprehensive component health monitoring
- Automated health checks with configurable intervals
- Detailed diagnostic information
- Alert generation for critical issues
- Recovery recommendations
- Integration with monitoring systems
"""

import os
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import requests

import torch

from core.utils.device_utils import get_optimal_device

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a system component"""

    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "details": self.details or {},
        }


@dataclass
class SystemHealth:
    """Overall system health information"""

    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime
    system_info: Dict[str, Any]
    alerts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "overall_status": self.overall_status.value,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "timestamp": self.timestamp.isoformat(),
            "system_info": self.system_info,
            "alerts": self.alerts,
        }


class ComponentChecker:
    """Base class for component health checkers"""

    def __init__(self, name: str, timeout: float = 5.0):
        """
        Initialize component checker.

        Args:
            name: Component name
            timeout: Check timeout in seconds
        """
        self.name = name
        self.timeout = timeout

    def check_health(self) -> ComponentHealth:
        """
        Check component health.

        Returns:
            Component health information
        """
        start_time = time.time()

        try:
            status, message, details = self._perform_check()
            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name=self.name,
                status=status,
                message=message,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                details=details,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check failed for {self.name}: {e}")

            return ComponentHealth(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                details={"error": str(e)},
            )

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """
        Perform the actual health check.

        Returns:
            Tuple of (status, message, details)
        """
        raise NotImplementedError("Subclasses must implement _perform_check")


class ModelHealthChecker(ComponentChecker):
    """Health checker for ML model components"""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize model health checker.

        Args:
            model_path: Path to model file
            device: Device for model loading
        """
        super().__init__("model")
        self.model_path = model_path
        self.device = get_optimal_device(device)
        self.model = None
        self.model_loaded = False

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check model health"""
        details = {
            "model_path": self.model_path,
            "device": str(self.device),
            "model_loaded": self.model_loaded,
        }

        # Check if model file exists
        if not os.path.exists(self.model_path):
            return (HealthStatus.CRITICAL, f"Model file not found: {self.model_path}", details)

        # Check model file size
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        details["model_size_mb"] = model_size_mb

        if model_size_mb < 1:
            return (
                HealthStatus.WARNING,
                f"Model file appears too small: {model_size_mb:.1f}MB",
                details,
            )

        # Try to load model if not already loaded
        if not self.model_loaded:
            try:
                self.model = torch.load(self.model_path, map_location=self.device)
                self.model.eval()
                self.model_loaded = True
                details["model_loaded"] = True
            except Exception as e:
                return (HealthStatus.CRITICAL, f"Failed to load model: {str(e)}", details)

        # Test model inference
        try:
            test_input = torch.randn(1, 1, 128, 128).to(self.device)

            with torch.no_grad():
                start_time = time.time()
                output = self.model(test_input)
                inference_time = (time.time() - start_time) * 1000

            details["inference_time_ms"] = inference_time
            details["output_shape"] = list(output.shape)

            # Check inference time
            if inference_time > 100:  # 100ms threshold
                return (
                    HealthStatus.WARNING,
                    f"Model inference slow: {inference_time:.1f}ms",
                    details,
                )

            return (
                HealthStatus.HEALTHY,
                f"Model healthy, inference: {inference_time:.1f}ms",
                details,
            )

        except Exception as e:
            return (HealthStatus.CRITICAL, f"Model inference failed: {str(e)}", details)


class SystemResourceChecker(ComponentChecker):
    """Health checker for system resources"""

    def __init__(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0,
    ):
        """
        Initialize system resource checker.

        Args:
            cpu_threshold: CPU usage threshold (%)
            memory_threshold: Memory usage threshold (%)
            disk_threshold: Disk usage threshold (%)
        """
        super().__init__("system_resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check system resource health"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100

        # Temperature (if available)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature if available
                for name, entries in temps.items():
                    if "cpu" in name.lower() or "core" in name.lower():
                        temperature = entries[0].current
                        break
        except:
            pass

        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk_percent,
            "disk_free_gb": disk.free / (1024**3),
            "temperature_c": temperature,
        }

        # Determine status based on thresholds
        status = HealthStatus.HEALTHY
        messages = []

        if cpu_percent > self.cpu_threshold:
            status = (
                HealthStatus.WARNING if status == HealthStatus.HEALTHY else HealthStatus.CRITICAL
            )
            messages.append(f"High CPU usage: {cpu_percent:.1f}%")

        if memory_percent > self.memory_threshold:
            status = (
                HealthStatus.WARNING if status == HealthStatus.HEALTHY else HealthStatus.CRITICAL
            )
            messages.append(f"High memory usage: {memory_percent:.1f}%")

        if disk_percent > self.disk_threshold:
            status = HealthStatus.CRITICAL
            messages.append(f"High disk usage: {disk_percent:.1f}%")

        if temperature and temperature > 80:
            status = (
                HealthStatus.WARNING if status == HealthStatus.HEALTHY else HealthStatus.CRITICAL
            )
            messages.append(f"High temperature: {temperature:.1f}°C")

        if status == HealthStatus.HEALTHY:
            message = f"Resources healthy (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"
        else:
            message = "; ".join(messages)

        return status, message, details


class GPUHealthChecker(ComponentChecker):
    """Health checker for GPU resources"""

    def __init__(self, memory_threshold: float = 90.0, temperature_threshold: float = 80.0):
        """
        Initialize GPU health checker.

        Args:
            memory_threshold: GPU memory usage threshold (%)
            temperature_threshold: GPU temperature threshold (°C)
        """
        super().__init__("gpu")
        self.memory_threshold = memory_threshold
        self.temperature_threshold = temperature_threshold

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check GPU health"""
        if not torch.cuda.is_available():
            return (HealthStatus.WARNING, "CUDA not available", {"cuda_available": False})

        try:
            # Get GPU information
            device_count = torch.cuda.device_count()
            gpu_info = []

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory

                memory_percent = (memory_reserved / memory_total) * 100

                gpu_data = {
                    "device_id": i,
                    "name": props.name,
                    "memory_allocated_gb": memory_allocated / (1024**3),
                    "memory_reserved_gb": memory_reserved / (1024**3),
                    "memory_total_gb": memory_total / (1024**3),
                    "memory_percent": memory_percent,
                    "compute_capability": f"{props.major}.{props.minor}",
                }

                gpu_info.append(gpu_data)

            details = {"device_count": device_count, "gpus": gpu_info}

            # Check for issues
            status = HealthStatus.HEALTHY
            messages = []

            for gpu in gpu_info:
                if gpu["memory_percent"] > self.memory_threshold:
                    status = HealthStatus.WARNING
                    messages.append(
                        f"GPU {gpu['device_id']} high memory: {gpu['memory_percent']:.1f}%"
                    )

            if status == HealthStatus.HEALTHY:
                message = f"{device_count} GPU(s) healthy"
            else:
                message = "; ".join(messages)

            return status, message, details

        except Exception as e:
            return (HealthStatus.CRITICAL, f"GPU check failed: {str(e)}", {"error": str(e)})


class APIHealthChecker(ComponentChecker):
    """Health checker for API endpoints"""

    def __init__(self, base_url: str, endpoints: List[str] = None):
        """
        Initialize API health checker.

        Args:
            base_url: Base URL for API
            endpoints: List of endpoints to check
        """
        super().__init__("api")
        self.base_url = base_url.rstrip("/")
        self.endpoints = endpoints or ["/health", "/metrics"]

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check API health"""
        details = {"base_url": self.base_url, "endpoints_checked": []}

        status = HealthStatus.HEALTHY
        messages = []

        for endpoint in self.endpoints:
            url = f"{self.base_url}{endpoint}"

            try:
                start_time = time.time()
                response = requests.get(url, timeout=self.timeout)
                response_time = (time.time() - start_time) * 1000

                endpoint_info = {
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "accessible": response.status_code < 400,
                }

                details["endpoints_checked"].append(endpoint_info)

                if response.status_code >= 400:
                    status = (
                        HealthStatus.WARNING
                        if status == HealthStatus.HEALTHY
                        else HealthStatus.CRITICAL
                    )
                    messages.append(f"{endpoint}: HTTP {response.status_code}")
                elif response_time > 5000:  # 5 second threshold
                    status = HealthStatus.WARNING
                    messages.append(f"{endpoint}: slow response {response_time:.0f}ms")

            except requests.exceptions.RequestException as e:
                status = HealthStatus.CRITICAL
                messages.append(f"{endpoint}: {str(e)}")

                endpoint_info = {"endpoint": endpoint, "error": str(e), "accessible": False}
                details["endpoints_checked"].append(endpoint_info)

        if status == HealthStatus.HEALTHY:
            message = f"All {len(self.endpoints)} endpoints healthy"
        else:
            message = "; ".join(messages)

        return status, message, details


class HealthChecker:
    """
    Main health checker that coordinates all component checks.
    Provides overall system health monitoring and alerting.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize health checker.

        Args:
            config: Health checker configuration
        """
        self.config = config or {}
        self.checkers: Dict[str, ComponentChecker] = {}
        self.monitoring = False
        self.monitor_thread = None
        self.check_interval = self.config.get("check_interval", 60)  # seconds
        self.alert_callbacks: List[Callable] = []

        # Setup default checkers
        self._setup_default_checkers()

        logger.info("Health checker initialized")

    def _setup_default_checkers(self):
        """Setup default health checkers"""
        # System resources checker
        self.add_checker(
            SystemResourceChecker(
                cpu_threshold=self.config.get("cpu_threshold", 80.0),
                memory_threshold=self.config.get("memory_threshold", 85.0),
                disk_threshold=self.config.get("disk_threshold", 90.0),
            )
        )

        # GPU checker (if available)
        if torch.cuda.is_available():
            self.add_checker(
                GPUHealthChecker(
                    memory_threshold=self.config.get("gpu_memory_threshold", 90.0),
                    temperature_threshold=self.config.get("gpu_temperature_threshold", 80.0),
                )
            )

        # Model checker (if model path provided)
        model_path = self.config.get("model_path")
        if model_path:
            self.add_checker(
                ModelHealthChecker(model_path=model_path, device=self.config.get("device", "auto"))
            )

        # API checker (if base URL provided)
        api_url = self.config.get("api_base_url")
        if api_url:
            self.add_checker(
                APIHealthChecker(
                    base_url=api_url, endpoints=self.config.get("api_endpoints", ["/health"])
                )
            )

    def add_checker(self, checker: ComponentChecker):
        """Add a component checker"""
        self.checkers[checker.name] = checker
        logger.info(f"Added health checker: {checker.name}")

    def remove_checker(self, name: str):
        """Remove a component checker"""
        if name in self.checkers:
            del self.checkers[name]
            logger.info(f"Removed health checker: {name}")

    def add_alert_callback(self, callback: Callable[[SystemHealth], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)

    def check_health(self) -> Dict[str, Any]:
        """
        Perform health check on all components.

        Returns:
            System health information
        """
        logger.debug("Performing health check...")

        components = {}
        alerts = []

        # Check each component
        for name, checker in self.checkers.items():
            try:
                component_health = checker.check_health()
                components[name] = component_health

                # Generate alerts for critical/warning status
                if component_health.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                    alert_msg = f"{name.upper()}: {component_health.message}"
                    alerts.append(alert_msg)

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check error: {str(e)}",
                    last_check=datetime.now(timezone.utc),
                )
                alerts.append(f"{name.upper()}: Health check error")

        # Determine overall status
        overall_status = self._calculate_overall_status(components)

        # Create system health object
        system_health = SystemHealth(
            overall_status=overall_status,
            components=components,
            timestamp=datetime.now(timezone.utc),
            system_info=self._get_system_info(),
            alerts=alerts,
        )

        # Trigger alert callbacks if needed
        if alerts and self.alert_callbacks:
            for callback in self.alert_callbacks:
                try:
                    callback(system_health)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        return system_health.to_dict()

    def _calculate_overall_status(self, components: Dict[str, ComponentHealth]) -> HealthStatus:
        """Calculate overall system status"""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [comp.status for comp in components.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "hostname": os.uname().nodename,
            "platform": os.uname().sysname,
            "architecture": os.uname().machine,
            "python_version": os.sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "uptime_seconds": time.time() - psutil.boot_time(),
        }

    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring:
            logger.warning("Health monitoring already running")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info(f"Started health monitoring (interval: {self.check_interval}s)")

    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("Stopped health monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                health_data = self.check_health()

                # Log health status
                overall_status = health_data["overall_status"]
                if overall_status in ["warning", "critical"]:
                    logger.warning(f"System health: {overall_status}")
                    for alert in health_data["alerts"]:
                        logger.warning(f"ALERT: {alert}")
                else:
                    logger.debug(f"System health: {overall_status}")

                # Wait for next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)


def create_health_checker(config_path: str = None) -> HealthChecker:
    """
    Create health checker from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured health checker
    """
    from core.utils.config_parser import ConfigParser

    if config_path:
        config_dict = ConfigParser.load_config(config_path)
        health_config = config_dict.get("health_check", {})
    else:
        health_config = {}

    return HealthChecker(health_config)


# Example alert callback functions
def log_alert_callback(system_health: SystemHealth):
    """Example alert callback that logs alerts"""
    for alert in system_health.alerts:
        logger.warning(f"HEALTH ALERT: {alert}")


def webhook_alert_callback(webhook_url: str):
    """Create webhook alert callback"""

    def callback(system_health: SystemHealth):
        try:
            payload = {
                "timestamp": system_health.timestamp.isoformat(),
                "status": system_health.overall_status.value,
                "alerts": system_health.alerts,
                "hostname": system_health.system_info.get("hostname", "unknown"),
            }

            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")

    return callback


if __name__ == "__main__":
    # Demo: Health checking system
    import argparse

    parser = argparse.ArgumentParser(description="Health Checker Demo")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--model-path", help="Model path to check")
    parser.add_argument("--api-url", help="API URL to check")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Create health checker
        if args.config:
            health_checker = create_health_checker(args.config)
        else:
            config = {}
            if args.model_path:
                config["model_path"] = args.model_path
            if args.api_url:
                config["api_base_url"] = args.api_url
            config["check_interval"] = args.interval

            health_checker = HealthChecker(config)

        # Add alert callback
        health_checker.add_alert_callback(log_alert_callback)

        print("🏥 SereneSense Health Checker")
        print(f"   Checking {len(health_checker.checkers)} components")

        # Perform initial health check
        health_data = health_checker.check_health()

        print(f"\n📊 Health Status: {health_data['overall_status'].upper()}")
        print("   Components:")
        for name, component in health_data["components"].items():
            status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "❌", "unknown": "❓"}
            icon = status_icon.get(component["status"], "❓")
            print(f"     {icon} {name}: {component['message']}")

        if health_data["alerts"]:
            print("\n🚨 Alerts:")
            for alert in health_data["alerts"]:
                print(f"     {alert}")

        # Start monitoring if requested
        if args.monitor:
            print(f"\n🔄 Starting continuous monitoring (interval: {args.interval}s)")
            print("   Press Ctrl+C to stop")

            health_checker.start_monitoring()

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping monitoring...")
                health_checker.stop_monitoring()

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Health checker demo failed: {e}")
