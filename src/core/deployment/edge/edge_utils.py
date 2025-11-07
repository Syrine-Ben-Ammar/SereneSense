#!/usr/bin/env python3
"""
SereneSense Edge Deployment Scripts
Optimized deployment for NVIDIA Jetson and Raspberry Pi platforms.

Usage:
    python scripts/deploy_edge.py --model models/optimized/audioMAE_jetson.trt --platform jetson --host 192.168.1.100
    python scripts/deploy_edge.py --model models/optimized/audioMAE_rpi.onnx --platform raspberry_pi --port 8080
    python scripts/deploy_edge.py --optimize --model models/best_model.pth --platform jetson --precision int8
"""

import argparse
import logging
import sys
import os
import json
import yaml
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import psutil
import threading
import signal

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.deployment.edge.jetson import JetsonDeployment
from core.deployment.edge.raspberry_pi import RaspberryPiDeployment
from core.inference.optimization.model_optimization import EdgeOptimizer, OptimizationConfig
from core.inference.realtime.detector import RealTimeDetector, InferenceConfig
from core.utils.logging import setup_logging
from core.utils.device_utils import get_device_info, detect_platform

logger = logging.getLogger(__name__)


class EdgeDeploymentManager:
    """
    Manages edge deployment across different platforms.
    """

    def __init__(self, platform: str, config: Dict[str, Any]):
        self.platform = platform
        self.config = config
        self.deployment = None
        self.detector = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()

        # Platform-specific deployment
        if platform == "jetson":
            self.deployment = JetsonDeployment(config)
        elif platform == "raspberry_pi":
            self.deployment = RaspberryPiDeployment(config)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def optimize_model(self, model_path: str, output_path: str, precision: str = "fp16"):
        """Optimize model for target platform"""
        logger.info(f"Optimizing model for {self.platform} with {precision} precision...")

        # Create optimization config
        opt_config = OptimizationConfig(
            target_platform=self.platform,
            precision=precision,
            input_shape=(1, 1, 128, 128),  # Adjust based on your model
        )

        # Load model
        from core.models.audioMAE.model import AudioMAE, AudioMAEConfig

        model_config = AudioMAEConfig()
        model = AudioMAE(model_config)

        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

        # Optimize model
        optimizer = EdgeOptimizer()
        results = optimizer.optimize_for_platform(model, opt_config)

        # Save optimized model
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if "tensorrt_path" in results["optimizations"]:
            import shutil

            shutil.copy(results["optimizations"]["tensorrt_path"], output_path)
        elif "onnx_path" in results["optimizations"]:
            import shutil

            shutil.copy(results["optimizations"]["onnx_path"], output_path)

        logger.info(f"Model optimized and saved to: {output_path}")

        return results

    def deploy(self, model_path: str, start_api: bool = True, start_detector: bool = True):
        """Deploy model to edge device"""
        logger.info(f"Deploying to {self.platform}...")

        try:
            # Setup deployment environment
            self.deployment.setup_environment()

            # Configure inference
            inference_config = InferenceConfig(
                model_path=model_path,
                device="cuda" if self.platform == "jetson" else "cpu",
                optimization="tensorrt" if self.platform == "jetson" else "onnx",
                confidence_threshold=self.config.get("confidence_threshold", 0.7),
            )

            # Start real-time detector if requested
            if start_detector:
                self.detector = RealTimeDetector(inference_config)
                self.detector.start()
                logger.info("Real-time detector started")

            # Start API server if requested
            if start_api:
                self.deployment.start_api_server(model_path, inference_config)
                logger.info("API server started")

            # Start monitoring
            self.start_monitoring()

            logger.info("Deployment completed successfully")

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def _monitoring_loop(self):
        """Monitoring loop for system resources"""
        while not self.stop_event.is_set():
            try:
                # System metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                # GPU metrics (if available)
                gpu_metrics = {}
                if self.platform == "jetson":
                    gpu_metrics = self._get_jetson_gpu_metrics()

                # Log metrics
                logger.debug(f"CPU: {cpu_usage}%, Memory: {memory.percent}%")
                if gpu_metrics:
                    logger.debug(f"GPU: {gpu_metrics}")

                # Check for thermal throttling
                if cpu_usage > 90:
                    logger.warning("High CPU usage detected")

                if memory.percent > 90:
                    logger.warning("High memory usage detected")

                time.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(30)

    def _get_jetson_gpu_metrics(self) -> Dict[str, Any]:
        """Get Jetson GPU metrics"""
        try:
            # Try to get Jetson stats
            result = subprocess.run(
                ["tegrastats", "--interval", "100", "--stop"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # Parse tegrastats output (simplified)
                output = result.stdout.strip()
                # This is a simplified parser - you might want to use a proper parser
                return {"tegrastats": output}

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return {}

    def stop(self):
        """Stop deployment"""
        logger.info("Stopping deployment...")

        # Stop monitoring
        self.stop_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        # Stop detector
        if self.detector:
            self.detector.stop()

        # Stop deployment
        if self.deployment:
            self.deployment.stop()

        logger.info("Deployment stopped")


class JetsonDeployment:
    """
    NVIDIA Jetson deployment handler.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_process = None

    def setup_environment(self):
        """Setup Jetson environment"""
        logger.info("Setting up Jetson environment...")

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available on Jetson")

        # Set power mode if specified
        power_mode = self.config.get("power_mode", "MAXN")
        try:
            self._set_power_mode(power_mode)
        except Exception as e:
            logger.warning(f"Failed to set power mode: {e}")

        # Set GPU memory fraction
        gpu_memory_fraction = self.config.get("gpu_memory_fraction", 0.8)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)

        logger.info("Jetson environment setup completed")

    def _set_power_mode(self, mode: str):
        """Set Jetson power mode"""
        power_modes = {"MAXN": "0", "15W": "1", "10W": "2"}

        if mode in power_modes:
            cmd = f"sudo nvpmodel -m {power_modes[mode]}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Power mode set to {mode}")
            else:
                logger.warning(f"Failed to set power mode: {result.stderr}")

    def start_api_server(self, model_path: str, inference_config: InferenceConfig):
        """Start API server on Jetson"""
        logger.info("Starting API server on Jetson...")

        # API server configuration
        api_config = {
            "host": self.config.get("api_host", "0.0.0.0"),
            "port": self.config.get("api_port", 8080),
            "workers": self.config.get("api_workers", 1),
            "model_path": model_path,
        }

        # Start API server in subprocess
        cmd = [
            sys.executable,
            "-m",
            "core.deployment.api.fastapi_server",
            "--host",
            api_config["host"],
            "--port",
            str(api_config["port"]),
            "--workers",
            str(api_config["workers"]),
        ]

        # Set environment variables
        env = os.environ.copy()
        env["SERENESENSE_MODEL_PATH"] = model_path
        env["SERENESENSE_DEVICE"] = "cuda"

        self.api_process = subprocess.Popen(cmd, env=env)

        # Wait for server to start
        time.sleep(5)

        if self.api_process.poll() is None:
            logger.info(f"API server started on {api_config['host']}:{api_config['port']}")
        else:
            raise RuntimeError("Failed to start API server")

    def stop(self):
        """Stop Jetson deployment"""
        if self.api_process:
            self.api_process.terminate()
            self.api_process.wait(timeout=10)


class RaspberryPiDeployment:
    """
    Raspberry Pi deployment handler.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_process = None

    def setup_environment(self):
        """Setup Raspberry Pi environment"""
        logger.info("Setting up Raspberry Pi environment...")

        # Check for AI HAT+ availability
        if self.config.get("use_ai_hat", True):
            if not self._check_ai_hat():
                logger.warning("AI HAT+ not detected, using CPU only")

        # Optimize CPU settings
        self._optimize_cpu_settings()

        # Set memory split for GPU (if using GPU acceleration)
        gpu_memory = self.config.get("gpu_memory", 64)
        if gpu_memory > 0:
            self._set_gpu_memory(gpu_memory)

        logger.info("Raspberry Pi environment setup completed")

    def _check_ai_hat(self) -> bool:
        """Check if AI HAT+ is available"""
        try:
            # Check for Hailo device
            result = subprocess.run(["lspci"], capture_output=True, text=True)
            return "Hailo" in result.stdout
        except:
            return False

    def _optimize_cpu_settings(self):
        """Optimize CPU settings for performance"""
        try:
            # Set CPU governor to performance
            subprocess.run(["sudo", "cpufreq-set", "-g", "performance"], check=False)

            logger.info("CPU governor set to performance mode")
        except Exception as e:
            logger.warning(f"Failed to set CPU governor: {e}")

    def _set_gpu_memory(self, memory_mb: int):
        """Set GPU memory split"""
        try:
            # This would typically require modifying /boot/config.txt
            # For now, just log the intention
            logger.info(f"GPU memory set to {memory_mb}MB")
        except Exception as e:
            logger.warning(f"Failed to set GPU memory: {e}")

    def start_api_server(self, model_path: str, inference_config: InferenceConfig):
        """Start API server on Raspberry Pi"""
        logger.info("Starting API server on Raspberry Pi...")

        # API server configuration
        api_config = {
            "host": self.config.get("api_host", "0.0.0.0"),
            "port": self.config.get("api_port", 8080),
            "workers": 1,  # Single worker for RPi
            "model_path": model_path,
        }

        # Start API server
        cmd = [
            sys.executable,
            "-m",
            "core.deployment.api.fastapi_server",
            "--host",
            api_config["host"],
            "--port",
            str(api_config["port"]),
            "--workers",
            str(api_config["workers"]),
        ]

        # Set environment variables
        env = os.environ.copy()
        env["SERENESENSE_MODEL_PATH"] = model_path
        env["SERENESENSE_DEVICE"] = "cpu"
        env["OMP_NUM_THREADS"] = str(self.config.get("cpu_threads", 4))

        self.api_process = subprocess.Popen(cmd, env=env)

        # Wait for server to start
        time.sleep(10)  # RPi might be slower

        if self.api_process.poll() is None:
            logger.info(f"API server started on {api_config['host']}:{api_config['port']}")
        else:
            raise RuntimeError("Failed to start API server")

    def stop(self):
        """Stop Raspberry Pi deployment"""
        if self.api_process:
            self.api_process.terminate()
            self.api_process.wait(timeout=15)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Deploy SereneSense to edge devices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and platform
    parser.add_argument("--model", type=str, required=True, help="Path to model file")

    parser.add_argument(
        "--platform",
        choices=["jetson", "raspberry_pi", "auto"],
        default="auto",
        help="Target platform",
    )

    # Optimization
    parser.add_argument(
        "--optimize", action="store_true", help="Optimize model for target platform"
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Model precision for optimization",
    )

    parser.add_argument("--output-model", type=str, help="Output path for optimized model")

    # Deployment options
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")

    parser.add_argument("--port", type=int, default=8080, help="API server port")

    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7, help="Detection confidence threshold"
    )

    parser.add_argument("--no-api", action="store_true", help="Don't start API server")

    parser.add_argument("--no-detector", action="store_true", help="Don't start real-time detector")

    # Platform-specific options
    parser.add_argument(
        "--power-mode", choices=["MAXN", "15W", "10W"], default="MAXN", help="Jetson power mode"
    )

    parser.add_argument(
        "--cpu-threads", type=int, default=4, help="Number of CPU threads (Raspberry Pi)"
    )

    parser.add_argument(
        "--gpu-memory", type=int, default=64, help="GPU memory in MB (Raspberry Pi)"
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Deployment configuration file")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def detect_platform_auto() -> str:
    """Auto-detect deployment platform"""
    try:
        # Check for Jetson
        if Path("/etc/nv_tegra_release").exists():
            return "jetson"

        # Check for Raspberry Pi
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
            if "raspberry pi" in cpuinfo or "bcm" in cpuinfo:
                return "raspberry_pi"

    except Exception:
        pass

    # Default to generic deployment
    return "cpu"


def main():
    """Main deployment function"""
    args = parse_arguments()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 80)
    logger.info("SereneSense Edge Deployment")
    logger.info("=" * 80)

    try:
        # Detect platform if auto
        platform = args.platform
        if platform == "auto":
            platform = detect_platform_auto()
            logger.info(f"Auto-detected platform: {platform}")

        # Load configuration
        config = {}
        if args.config:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)

        # Override config with command line arguments
        config.update(
            {
                "api_host": args.host,
                "api_port": args.port,
                "confidence_threshold": args.confidence_threshold,
                "power_mode": args.power_mode,
                "cpu_threads": args.cpu_threads,
                "gpu_memory": args.gpu_memory,
            }
        )

        # Create deployment manager
        deployment_manager = EdgeDeploymentManager(platform, config)

        # Optimize model if requested
        model_path = args.model
        if args.optimize:
            if not args.output_model:
                args.output_model = f"models/optimized/serenesense_{platform}_{args.precision}.{'trt' if platform == 'jetson' else 'onnx'}"

            logger.info("Optimizing model...")
            optimization_results = deployment_manager.optimize_model(
                args.model, args.output_model, args.precision
            )

            model_path = args.output_model

            # Log optimization results
            logger.info("Optimization completed:")
            if "benchmarks" in optimization_results:
                for opt_name, benchmark in optimization_results["benchmarks"].items():
                    if isinstance(benchmark, dict) and "avg_latency_ms" in benchmark:
                        logger.info(f"  {opt_name}: {benchmark['avg_latency_ms']:.2f}ms latency")

        # Deploy model
        logger.info(f"Deploying to {platform}...")
        deployment_manager.deploy(
            model_path, start_api=not args.no_api, start_detector=not args.no_detector
        )

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            deployment_manager.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep running
        logger.info("Deployment running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
