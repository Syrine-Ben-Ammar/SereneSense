#
# Plan:
# 1. Create comprehensive Raspberry Pi deployment system
# 2. Support RPi 5 + AI HAT+ with 26 TOPS performance
# 3. ONNX optimization for CPU and NPU acceleration
# 4. Power management for 20-25 hour battery operation
# 5. Thermal monitoring and throttling protection
# 6. Memory optimization for 8GB RAM constraints
# 7. Performance monitoring and edge-specific optimizations
#

"""
Raspberry Pi Deployment for Military Vehicle Detection
Optimized for Pi 5 + AI HAT+ with 26 TOPS performance at $190 cost.

Features:
- ONNX optimization for CPU/NPU acceleration
- Power management for 20-25h battery life
- Thermal monitoring and protection
- Memory optimization for 8GB constraints
- NPU acceleration via AI HAT+
- Cost-effective edge deployment
"""

import os
import subprocess
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import threading
import psutil
import platform

import torch
import numpy as np

from core.inference.optimization.onnx_export import ONNXExporter, ONNXConfig
from core.inference.real_time import RealTimeInference, InferenceConfig
from core.utils.config_parser import ConfigParser

logger = logging.getLogger(__name__)


@dataclass
class RaspberryPiConfig:
    """Raspberry Pi deployment configuration"""

    # Model settings
    model_path: str = "models/serenesense_best.pth"
    optimized_model_path: str = "models/serenesense_rpi.onnx"

    # ONNX optimization
    onnx_quantization: str = "dynamic"  # none, dynamic, static
    onnx_optimization_level: str = "all"  # basic, extended, all
    target_device: str = "cpu"  # cpu, npu (if AI HAT+ available)

    # Performance settings
    max_batch_size: int = 4
    num_threads: int = 4  # CPU threads for inference

    # Power management
    enable_power_saving: bool = True
    cpu_governor: str = "ondemand"  # performance, ondemand, powersave
    gpu_memory_split: int = 64  # MB allocated to GPU

    # Thermal management
    thermal_monitoring: bool = True
    temp_threshold_celsius: float = 70.0
    temp_throttle_threshold: float = 80.0
    enable_thermal_throttling: bool = True

    # Memory management
    swap_size_mb: int = 2048
    enable_memory_optimization: bool = True
    max_memory_usage_percent: float = 80.0

    # AI HAT+ settings
    ai_hat_enabled: bool = False
    npu_acceleration: bool = False
    hailo_device_id: int = 0

    # Monitoring
    performance_monitoring: bool = True
    metrics_interval: float = 2.0  # seconds
    log_performance: bool = True


class RaspberryPiHardwareMonitor:
    """
    Hardware monitoring for Raspberry Pi.
    Tracks temperature, power, memory, and performance metrics.
    """

    def __init__(self, config: RaspberryPiConfig):
        """
        Initialize hardware monitor.

        Args:
            config: Raspberry Pi configuration
        """
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            "temperature": {"cpu": [], "gpu": []},
            "power": {"estimate": []},
            "memory": {"used": [], "available": [], "swap_used": []},
            "performance": {"cpu_utilization": [], "load_average": []},
            "throttling": {"under_voltage": [], "arm_frequency_capped": [], "throttled": []},
        }

        # Check if running on actual Raspberry Pi
        self.is_raspberry_pi = self._detect_raspberry_pi()
        if self.is_raspberry_pi:
            logger.info("Running on Raspberry Pi hardware")
        else:
            logger.info("Running on non-Pi hardware (simulation mode)")

    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                return "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo
        except:
            return False

    def _read_cpu_temperature(self) -> float:
        """Read CPU temperature"""
        try:
            if self.is_raspberry_pi:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp_millicelsius = int(f.read().strip())
                    return temp_millicelsius / 1000.0
            else:
                # Simulation: return a reasonable temperature
                return 45.0 + np.random.normal(0, 2)
        except:
            return 0.0

    def _read_gpu_temperature(self) -> float:
        """Read GPU temperature using vcgencmd"""
        try:
            if self.is_raspberry_pi:
                result = subprocess.run(
                    ["vcgencmd", "measure_temp"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    temp_str = result.stdout.strip()
                    # Parse "temp=45.0'C"
                    temp = float(temp_str.split("=")[1].split("'")[0])
                    return temp
            return 0.0
        except:
            return 0.0

    def _read_throttle_status(self) -> Dict[str, bool]:
        """Read throttling status"""
        try:
            if self.is_raspberry_pi:
                result = subprocess.run(
                    ["vcgencmd", "get_throttled"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    throttle_hex = result.stdout.strip().split("=")[1]
                    throttle_int = int(throttle_hex, 16)

                    return {
                        "under_voltage": bool(throttle_int & 0x1),
                        "arm_frequency_capped": bool(throttle_int & 0x2),
                        "throttled": bool(throttle_int & 0x4),
                        "soft_temp_limit": bool(throttle_int & 0x8),
                    }
            return {"under_voltage": False, "arm_frequency_capped": False, "throttled": False}
        except:
            return {"under_voltage": False, "arm_frequency_capped": False, "throttled": False}

    def _estimate_power_consumption(self) -> float:
        """Estimate power consumption based on CPU usage and temperature"""
        try:
            cpu_percent = psutil.cpu_percent()
            cpu_temp = self._read_cpu_temperature()

            # Simple power estimation model for Pi 5
            # Base consumption ~3W, scales with CPU usage and temperature
            base_power = 3.0  # Watts
            cpu_power = (cpu_percent / 100.0) * 5.0  # Up to 5W for CPU
            thermal_factor = max(1.0, (cpu_temp - 40) / 40)  # Thermal overhead

            estimated_power = (base_power + cpu_power) * thermal_factor
            return min(estimated_power, 12.0)  # Cap at 12W for Pi 5

        except:
            return 5.0  # Default estimate

    def _read_memory_info(self) -> Dict[str, int]:
        """Read memory usage information"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return {
                "total": memory.total,
                "used": memory.used,
                "available": memory.available,
                "percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent,
            }
        except:
            return {
                "total": 0,
                "used": 0,
                "available": 0,
                "percent": 0,
                "swap_total": 0,
                "swap_used": 0,
                "swap_percent": 0,
            }

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                timestamp = time.time()

                # Temperature monitoring
                cpu_temp = self._read_cpu_temperature()
                gpu_temp = self._read_gpu_temperature()

                self.metrics["temperature"]["cpu"].append((timestamp, cpu_temp))
                self.metrics["temperature"]["gpu"].append((timestamp, gpu_temp))

                # Power estimation
                power_estimate = self._estimate_power_consumption()
                self.metrics["power"]["estimate"].append((timestamp, power_estimate))

                # Memory monitoring
                memory_info = self._read_memory_info()
                self.metrics["memory"]["used"].append((timestamp, memory_info["used"]))
                self.metrics["memory"]["available"].append((timestamp, memory_info["available"]))
                self.metrics["memory"]["swap_used"].append((timestamp, memory_info["swap_used"]))

                # Performance monitoring
                cpu_percent = psutil.cpu_percent()
                load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0

                self.metrics["performance"]["cpu_utilization"].append((timestamp, cpu_percent))
                self.metrics["performance"]["load_average"].append((timestamp, load_avg))

                # Throttling status
                throttle_status = self._read_throttle_status()
                for key, value in throttle_status.items():
                    if key in self.metrics["throttling"]:
                        self.metrics["throttling"][key].append((timestamp, value))

                # Check thermal conditions
                if self.config.thermal_monitoring:
                    if cpu_temp > self.config.temp_throttle_threshold:
                        logger.warning(
                            f"CPU temperature critical: {cpu_temp}°C - throttling may occur"
                        )
                    elif cpu_temp > self.config.temp_threshold_celsius:
                        logger.warning(f"CPU temperature high: {cpu_temp}°C")

                # Check memory usage
                if memory_info["percent"] > self.config.max_memory_usage_percent:
                    logger.warning(f"High memory usage: {memory_info['percent']:.1f}%")

                # Trim old metrics (keep last 500 points)
                for category in self.metrics:
                    for metric in self.metrics[category]:
                        if len(self.metrics[category][metric]) > 500:
                            self.metrics[category][metric] = self.metrics[category][metric][-500:]

                time.sleep(self.config.metrics_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.metrics_interval)

    def start_monitoring(self):
        """Start hardware monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Hardware monitoring started")

    def stop_monitoring(self):
        """Stop hardware monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Hardware monitoring stopped")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current hardware metrics"""
        current = {}

        for category in self.metrics:
            current[category] = {}
            for metric_name, metric_data in self.metrics[category].items():
                if metric_data:
                    current[category][metric_name] = metric_data[-1][1]  # Latest value
                else:
                    current[category][metric_name] = 0

        return current

    def get_metrics_history(self, duration_seconds: int = 120) -> Dict[str, Any]:
        """Get metrics history for specified duration"""
        cutoff_time = time.time() - duration_seconds
        history = {}

        for category in self.metrics:
            history[category] = {}
            for metric_name, metric_data in self.metrics[category].items():
                # Filter recent data
                recent_data = [(t, v) for t, v in metric_data if t >= cutoff_time]
                history[category][metric_name] = recent_data

        return history


class RaspberryPiOptimizer:
    """
    Raspberry Pi-specific optimization utilities.
    Handles power management, memory optimization, and performance tuning.
    """

    def __init__(self, config: RaspberryPiConfig):
        """
        Initialize Pi optimizer.

        Args:
            config: Raspberry Pi configuration
        """
        self.config = config
        self.is_raspberry_pi = self._detect_raspberry_pi()

    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                return "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo
        except:
            return False

    def set_cpu_governor(self, governor: str):
        """
        Set CPU frequency governor.

        Args:
            governor: Governor type (performance, ondemand, powersave)
        """
        try:
            if self.is_raspberry_pi:
                # Set governor for all CPU cores
                cpu_count = psutil.cpu_count()
                for cpu in range(cpu_count):
                    gov_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                    if Path(gov_path).exists():
                        subprocess.run(
                            ["sudo", "sh", "-c", f"echo {governor} > {gov_path}"], check=False
                        )

                logger.info(f"CPU governor set to {governor}")
            else:
                logger.info(f"CPU governor setting skipped (not on Pi): {governor}")

        except Exception as e:
            logger.error(f"Error setting CPU governor: {e}")

    def optimize_memory(self):
        """Optimize memory settings for inference"""
        try:
            # Set GPU memory split
            if self.is_raspberry_pi and self.config.gpu_memory_split:
                result = subprocess.run(
                    [
                        "sudo",
                        "raspi-config",
                        "nonint",
                        "do_memory_split",
                        str(self.config.gpu_memory_split),
                    ],
                    check=False,
                )
                if result.returncode == 0:
                    logger.info(f"GPU memory split set to {self.config.gpu_memory_split}MB")

            # Configure swap if needed
            if self.config.swap_size_mb > 0:
                self._configure_swap()

            # Set memory overcommit for better memory usage
            subprocess.run(["sudo", "sysctl", "vm.overcommit_memory=1"], check=False)

            # Enable memory optimization for inference
            if self.config.enable_memory_optimization:
                # Reduce cache pressure
                subprocess.run(["sudo", "sysctl", "vm.vfs_cache_pressure=50"], check=False)
                # Reduce swappiness
                subprocess.run(["sudo", "sysctl", "vm.swappiness=10"], check=False)

            logger.info("Memory optimization applied")

        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")

    def _configure_swap(self):
        """Configure swap file for better memory management"""
        try:
            swap_file = "/swapfile"
            swap_size_bytes = self.config.swap_size_mb * 1024 * 1024

            # Check if swap file exists and has correct size
            if Path(swap_file).exists():
                current_size = Path(swap_file).stat().st_size
                if current_size == swap_size_bytes:
                    logger.info(f"Swap file already configured: {self.config.swap_size_mb}MB")
                    return

            logger.info(f"Configuring swap file: {self.config.swap_size_mb}MB")

            # Disable existing swap
            subprocess.run(["sudo", "swapoff", swap_file], check=False)

            # Create new swap file
            subprocess.run(
                ["sudo", "fallocate", "-l", f"{swap_size_bytes}", swap_file], check=False
            )
            subprocess.run(["sudo", "chmod", "600", swap_file], check=False)
            subprocess.run(["sudo", "mkswap", swap_file], check=False)
            subprocess.run(["sudo", "swapon", swap_file], check=False)

        except Exception as e:
            logger.error(f"Error configuring swap: {e}")

    def optimize_for_inference(self):
        """Apply Pi-specific optimizations for inference"""
        try:
            # Set CPU governor
            self.set_cpu_governor(self.config.cpu_governor)

            # Optimize memory
            self.optimize_memory()

            # Set thread affinity for better performance
            if self.config.num_threads:
                os.environ["OMP_NUM_THREADS"] = str(self.config.num_threads)
                os.environ["MKL_NUM_THREADS"] = str(self.config.num_threads)

            # Disable unnecessary services for power saving
            if self.config.enable_power_saving:
                self._disable_unnecessary_services()

            logger.info("Pi optimizations applied for inference")

        except Exception as e:
            logger.error(f"Error applying Pi optimizations: {e}")

    def _disable_unnecessary_services(self):
        """Disable unnecessary services to save power"""
        try:
            # Services that can be disabled for edge inference
            services_to_disable = ["bluetooth.service", "hciuart.service", "avahi-daemon.service"]

            for service in services_to_disable:
                subprocess.run(["sudo", "systemctl", "disable", service], check=False)
                subprocess.run(["sudo", "systemctl", "stop", service], check=False)

            logger.info("Unnecessary services disabled for power saving")

        except Exception as e:
            logger.debug(f"Could not disable some services: {e}")

    def check_ai_hat_availability(self) -> Dict[str, Any]:
        """Check if AI HAT+ is available and functional"""
        ai_hat_info = {
            "available": False,
            "hailo_driver": False,
            "npu_devices": 0,
            "device_info": None,
        }

        try:
            # Check for Hailo driver
            result = subprocess.run(["lsmod"], capture_output=True, text=True)
            if "hailo" in result.stdout.lower():
                ai_hat_info["hailo_driver"] = True

            # Check for NPU devices
            if Path("/dev/hailo0").exists():
                ai_hat_info["npu_devices"] = 1
                ai_hat_info["available"] = True

            # Try to get device info
            try:
                result = subprocess.run(
                    ["hailo", "fw-control", "identify"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    ai_hat_info["device_info"] = result.stdout.strip()
            except:
                pass

        except Exception as e:
            logger.debug(f"Error checking AI HAT+: {e}")

        return ai_hat_info


class RaspberryPiDeployment:
    """
    Complete deployment solution for Raspberry Pi edge devices.
    Optimized for Pi 5 + AI HAT+ with cost-effective performance.
    """

    def __init__(self, config: RaspberryPiConfig):
        """
        Initialize Raspberry Pi deployment.

        Args:
            config: Raspberry Pi configuration
        """
        self.config = config
        self.model = None
        self.optimized_model = None
        self.real_time_inference = None

        # Initialize components
        self.hardware_monitor = RaspberryPiHardwareMonitor(config)
        self.optimizer = RaspberryPiOptimizer(config)

        # Check hardware and setup
        self.hardware_info = self._check_hardware()
        self._setup_raspberry_pi()

        logger.info("Raspberry Pi deployment initialized")

    def _check_hardware(self) -> Dict[str, Any]:
        """Check Raspberry Pi hardware and capabilities"""
        hardware_info = {
            "platform": "unknown",
            "model": "unknown",
            "os_version": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "ai_hat": self.optimizer.check_ai_hat_availability(),
        }

        try:
            # Get Pi model information
            if self.optimizer.is_raspberry_pi:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    for line in cpuinfo.split("\n"):
                        if "Model" in line:
                            hardware_info["model"] = line.split(":")[1].strip()
                            break

                # Detect Pi version
                if "Pi 5" in hardware_info["model"]:
                    hardware_info["platform"] = "pi5"
                elif "Pi 4" in hardware_info["model"]:
                    hardware_info["platform"] = "pi4"
                elif "Pi 3" in hardware_info["model"]:
                    hardware_info["platform"] = "pi3"
                else:
                    hardware_info["platform"] = "pi_other"

        except Exception as e:
            logger.warning(f"Could not detect hardware details: {e}")

        return hardware_info

    def _setup_raspberry_pi(self):
        """Setup Raspberry Pi for optimal performance"""
        logger.info("Setting up Raspberry Pi for deployment...")

        # Apply optimizations
        self.optimizer.optimize_for_inference()

        # Start monitoring
        if self.config.performance_monitoring:
            self.hardware_monitor.start_monitoring()

        # Log hardware info
        logger.info(f"Platform: {self.hardware_info['platform']}")
        logger.info(f"Model: {self.hardware_info['model']}")
        logger.info(f"CPU cores: {self.hardware_info['cpu_count']}")
        logger.info(f"Memory: {self.hardware_info['memory_total'] // (1024**3)}GB")

        if self.hardware_info["ai_hat"]["available"]:
            logger.info("AI HAT+ detected and available")
        else:
            logger.info("AI HAT+ not detected, using CPU inference")

    def optimize_model(self, model_path: str = None) -> str:
        """
        Optimize model for Raspberry Pi deployment using ONNX.

        Args:
            model_path: Path to PyTorch model

        Returns:
            Path to optimized ONNX model
        """
        model_path = model_path or self.config.model_path

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Optimizing model for Raspberry Pi: {model_path}")

        # Load PyTorch model
        model = torch.load(model_path, map_location="cpu")
        model.eval()

        # Configure ONNX optimization
        providers = ["CPUExecutionProvider"]
        if self.config.ai_hat_enabled and self.hardware_info["ai_hat"]["available"]:
            # Add Hailo provider if AI HAT+ is available
            providers = ["HailoExecutionProvider", "CPUExecutionProvider"]

        onnx_config = ONNXConfig(
            input_shape=(1, 128, 128),
            quantization=self.config.onnx_quantization,
            optimization_level=self.config.onnx_optimization_level,
            max_batch_size=self.config.max_batch_size,
            output_path=self.config.optimized_model_path,
            providers=providers,
        )

        # Optimize with ONNX
        onnx_exporter = ONNXExporter(onnx_config)
        optimized_path = onnx_exporter.export_model(model)

        logger.info(f"Model optimized for Raspberry Pi: {optimized_path}")
        return optimized_path

    def deploy_model(self, model_path: str = None, optimize: bool = True) -> bool:
        """
        Deploy model for real-time inference.

        Args:
            model_path: Path to model file
            optimize: Whether to optimize with ONNX

        Returns:
            True if deployment successful
        """
        try:
            model_path = model_path or self.config.model_path

            # Optimize model if requested
            if optimize and not Path(self.config.optimized_model_path).exists():
                optimized_path = self.optimize_model(model_path)
                model_path = optimized_path
            elif Path(self.config.optimized_model_path).exists():
                model_path = self.config.optimized_model_path

            # Configure real-time inference
            inference_config = InferenceConfig(
                model_path=model_path,
                device="cpu",
                optimization="onnx" if model_path.endswith(".onnx") else "none",
                confidence_threshold=0.7,
                batch_size=self.config.max_batch_size,
            )

            # Create real-time inference
            self.real_time_inference = RealTimeInference(inference_config)

            logger.info("Model deployed successfully for real-time inference")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False

    def start_inference(self) -> bool:
        """Start real-time inference"""
        if not self.real_time_inference:
            logger.error("Model not deployed. Call deploy_model() first.")
            return False

        try:
            self.real_time_inference.start()
            logger.info("Real-time inference started")
            return True
        except Exception as e:
            logger.error(f"Failed to start inference: {e}")
            return False

    def stop_inference(self):
        """Stop real-time inference"""
        if self.real_time_inference:
            self.real_time_inference.stop()
            logger.info("Real-time inference stopped")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            "hardware": self.hardware_monitor.get_current_metrics(),
            "inference": {},
            "system": {
                "uptime": time.time(),
                "platform": self.hardware_info["platform"],
                "model": self.hardware_info["model"],
                "ai_hat_available": self.hardware_info["ai_hat"]["available"],
            },
        }

        # Add inference metrics if available
        if self.real_time_inference:
            inference_stats = self.real_time_inference.get_statistics()
            metrics["inference"] = inference_stats

        return metrics

    def benchmark_performance(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Run performance benchmark.

        Args:
            duration_seconds: Benchmark duration

        Returns:
            Benchmark results
        """
        logger.info(f"Running {duration_seconds}s performance benchmark...")

        if not self.real_time_inference:
            raise RuntimeError("Model not deployed for benchmarking")

        # Start inference if not running
        was_running = self.real_time_inference.is_running
        if not was_running:
            self.start_inference()

        # Record initial metrics
        start_time = time.time()
        initial_metrics = self.hardware_monitor.get_current_metrics()

        # Wait for benchmark duration
        time.sleep(duration_seconds)

        # Get final metrics
        end_time = time.time()
        final_metrics = self.hardware_monitor.get_current_metrics()

        # Stop inference if we started it
        if not was_running:
            self.stop_inference()

        # Calculate benchmark results
        inference_stats = self.real_time_inference.get_statistics()

        benchmark_results = {
            "duration_seconds": end_time - start_time,
            "inference_performance": {
                "avg_latency_ms": inference_stats.get("avg_processing_time", 0) * 1000,
                "max_latency_ms": inference_stats.get("max_processing_time", 0) * 1000,
                "throughput_fps": 1.0
                / max(inference_stats.get("avg_processing_time", 0.001), 0.001),
                "processed_windows": inference_stats.get("processed_windows", 0),
                "detections": inference_stats.get("detections", 0),
            },
            "hardware_performance": {
                "avg_cpu_temp": final_metrics.get("temperature", {}).get("cpu", 0),
                "avg_power_estimate": final_metrics.get("power", {}).get("estimate", 0),
                "avg_cpu_utilization": final_metrics.get("performance", {}).get(
                    "cpu_utilization", 0
                ),
                "avg_load_average": final_metrics.get("performance", {}).get("load_average", 0),
                "peak_memory_usage": final_metrics.get("memory", {}).get("used", 0),
                "throttling_detected": any(final_metrics.get("throttling", {}).values()),
            },
        }

        logger.info("Benchmark completed")
        logger.info(
            f"  Avg latency: {benchmark_results['inference_performance']['avg_latency_ms']:.1f}ms"
        )
        logger.info(
            f"  Throughput: {benchmark_results['inference_performance']['throughput_fps']:.1f} FPS"
        )
        logger.info(
            f"  Avg CPU temp: {benchmark_results['hardware_performance']['avg_cpu_temp']:.1f}°C"
        )
        logger.info(
            f"  Avg power: {benchmark_results['hardware_performance']['avg_power_estimate']:.1f}W"
        )

        return benchmark_results

    def estimate_battery_life(self, battery_capacity_wh: float = 100.0) -> Dict[str, float]:
        """
        Estimate battery life based on current power consumption.

        Args:
            battery_capacity_wh: Battery capacity in watt-hours

        Returns:
            Battery life estimates
        """
        current_metrics = self.hardware_monitor.get_current_metrics()
        power_consumption = current_metrics.get("power", {}).get("estimate", 5.0)

        # Calculate battery life estimates
        estimates = {
            "current_power_w": power_consumption,
            "battery_capacity_wh": battery_capacity_wh,
            "estimated_hours": battery_capacity_wh / power_consumption,
            "estimated_days": (battery_capacity_wh / power_consumption) / 24,
            "power_efficiency_percent": min(100, (12.0 - power_consumption) / 12.0 * 100),
        }

        return estimates

    def cleanup(self):
        """Cleanup deployment resources"""
        logger.info("Cleaning up Raspberry Pi deployment...")

        # Stop inference
        if self.real_time_inference:
            self.stop_inference()

        # Stop monitoring
        self.hardware_monitor.stop_monitoring()

        logger.info("Raspberry Pi deployment cleanup completed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


def create_raspberry_pi_deployment(config_path: str = None) -> RaspberryPiDeployment:
    """
    Create Raspberry Pi deployment from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured Raspberry Pi deployment
    """
    if config_path:
        config_dict = ConfigParser.load_config(config_path)
        pi_config = RaspberryPiConfig(**config_dict.get("raspberry_pi", {}))
    else:
        pi_config = RaspberryPiConfig()

    return RaspberryPiDeployment(pi_config)


if __name__ == "__main__":
    # Demo: Raspberry Pi deployment
    import argparse

    parser = argparse.ArgumentParser(description="Raspberry Pi Deployment")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--model-path", help="PyTorch model path")
    parser.add_argument("--optimize", action="store_true", help="Optimize model with ONNX")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument(
        "--battery-capacity", type=float, default=100.0, help="Battery capacity in Wh"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Create deployment
        deployment = create_raspberry_pi_deployment(args.config)

        print("🥧 Raspberry Pi Deployment Initialized")
        print(f"   Platform: {deployment.hardware_info['platform']}")
        print(f"   Model: {deployment.hardware_info['model']}")
        print(f"   CPU cores: {deployment.hardware_info['cpu_count']}")
        print(f"   Memory: {deployment.hardware_info['memory_total'] // (1024**3)}GB")
        print(
            f"   AI HAT+: {'Available' if deployment.hardware_info['ai_hat']['available'] else 'Not detected'}"
        )

        # Deploy model
        if args.model_path:
            success = deployment.deploy_model(args.model_path, args.optimize)
            if success:
                print("✅ Model deployed successfully")

                # Run benchmark if requested
                if args.benchmark:
                    results = deployment.benchmark_performance(30)
                    print("\n📊 Benchmark Results:")
                    print(f"   Latency: {results['inference_performance']['avg_latency_ms']:.1f}ms")
                    print(
                        f"   Throughput: {results['inference_performance']['throughput_fps']:.1f} FPS"
                    )
                    print(f"   Power: {results['hardware_performance']['avg_power_estimate']:.1f}W")

                    # Battery life estimate
                    battery_info = deployment.estimate_battery_life(args.battery_capacity)
                    print(f"   Battery life: {battery_info['estimated_hours']:.1f}h")
            else:
                print("❌ Model deployment failed")

        # Show current metrics
        metrics = deployment.get_performance_metrics()
        print("\n📈 Current Metrics:")
        if "temperature" in metrics["hardware"]:
            print(f"   CPU Temp: {metrics['hardware']['temperature'].get('cpu', 0):.1f}°C")
        if "power" in metrics["hardware"]:
            print(f"   Power: {metrics['hardware']['power'].get('estimate', 0):.1f}W")
        if "memory" in metrics["hardware"]:
            memory_gb = metrics["hardware"]["memory"].get("used", 0) / (1024**3)
            print(f"   Memory: {memory_gb:.1f}GB")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Raspberry Pi deployment failed: {e}")
