"""
Device Utilities for SereneSense

This module provides device detection, hardware optimization, and
resource management utilities for the SereneSense system.

Features:
- GPU/CPU detection and configuration
- Memory management and optimization
- Hardware-specific optimizations
- Edge device detection (Jetson, Raspberry Pi)
- Performance profiling and monitoring
- Dynamic device selection

Example:
    >>> from core.utils.device_utils import get_device_info, get_optimal_device
    >>> 
    >>> # Get device information
    >>> device_info = get_device_info()
    >>> print(f"Device: {device_info['device_type']}")
    >>> 
    >>> # Get optimal device for training
    >>> device = get_optimal_device(task="training")
    >>> print(f"Using device: {device}")
"""

import os
import platform
import subprocess
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

# Third-party imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available", ImportWarning)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available", ImportWarning)

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import py3nvml.py3nvml as nvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Container for device information."""

    device_type: str  # 'cuda', 'cpu', 'mps', 'edge'
    device_name: str
    compute_capability: Optional[str] = None
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    cores: int = 0
    architecture: str = ""
    platform_info: str = ""
    optimization_features: List[str] = None

    def __post_init__(self):
        if self.optimization_features is None:
            self.optimization_features = []


@dataclass
class PerformanceProfile:
    """Container for performance profiling results."""

    device: str
    task: str
    batch_size: int
    throughput: float  # samples/second
    latency: float  # milliseconds
    memory_usage: float  # GB
    power_consumption: Optional[float] = None  # watts
    temperature: Optional[float] = None  # celsius


class DeviceDetector:
    """
    Comprehensive device detection and characterization.
    """

    def __init__(self):
        """Initialize device detector."""
        self._cache: Dict[str, Any] = {}
        self._initialize_libraries()

    def _initialize_libraries(self) -> None:
        """Initialize hardware detection libraries."""
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self._nvml_initialized = False
        else:
            self._nvml_initialized = False

    def detect_device_type(self) -> str:
        """
        Detect the primary device type.

        Returns:
            Device type string ('cuda', 'mps', 'cpu', 'edge')
        """
        # Check for CUDA
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"

        # Check for Apple Metal Performance Shaders
        if TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        # Check for edge devices
        if self._is_edge_device():
            return "edge"

        # Default to CPU
        return "cpu"

    def _is_edge_device(self) -> bool:
        """
        Check if running on an edge device.

        Returns:
            True if edge device detected
        """
        # Check for Jetson
        if self._is_jetson():
            return True

        # Check for Raspberry Pi
        if self._is_raspberry_pi():
            return True

        return False

    def _is_jetson(self) -> bool:
        """
        Check if running on NVIDIA Jetson.

        Returns:
            True if Jetson detected
        """
        try:
            # Check for Jetson-specific files
            jetson_files = [
                "/proc/device-tree/model",
                "/proc/device-tree/compatible",
                "/etc/nv_tegra_release",
            ]

            for file_path in jetson_files:
                if Path(file_path).exists():
                    try:
                        with open(file_path, "r") as f:
                            content = f.read().lower()
                            if any(keyword in content for keyword in ["jetson", "tegra", "nvidia"]):
                                return True
                    except:
                        continue

            # Check for jetson_clocks utility
            try:
                subprocess.run(["which", "jetson_clocks"], check=True, capture_output=True)
                return True
            except:
                pass

        except Exception as e:
            logger.debug(f"Error checking for Jetson: {e}")

        return False

    def _is_raspberry_pi(self) -> bool:
        """
        Check if running on Raspberry Pi.

        Returns:
            True if Raspberry Pi detected
        """
        try:
            # Check for Pi-specific files
            pi_files = ["/proc/device-tree/model", "/proc/cpuinfo"]

            for file_path in pi_files:
                if Path(file_path).exists():
                    try:
                        with open(file_path, "r") as f:
                            content = f.read().lower()
                            if any(
                                keyword in content for keyword in ["raspberry", "broadcom", "bcm"]
                            ):
                                return True
                    except:
                        continue

        except Exception as e:
            logger.debug(f"Error checking for Raspberry Pi: {e}")

        return False

    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Get detailed CPU information.

        Returns:
            Dictionary containing CPU information
        """
        info = {
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cores_physical": 0,
            "cores_logical": 0,
            "frequency_current": 0.0,
            "frequency_max": 0.0,
            "cache_l1": 0,
            "cache_l2": 0,
            "cache_l3": 0,
            "features": [],
        }

        if PSUTIL_AVAILABLE:
            try:
                info["cores_physical"] = psutil.cpu_count(logical=False) or 0
                info["cores_logical"] = psutil.cpu_count(logical=True) or 0

                # Get frequency information
                freq_info = psutil.cpu_freq()
                if freq_info:
                    info["frequency_current"] = freq_info.current
                    info["frequency_max"] = freq_info.max

            except Exception as e:
                logger.debug(f"Error getting CPU info with psutil: {e}")

        # Get CPU features on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()

                # Extract features
                for line in cpuinfo.split("\n"):
                    if line.startswith("flags") or line.startswith("Features"):
                        features = line.split(":")[1].strip().split()
                        info["features"] = features
                        break

            except Exception as e:
                logger.debug(f"Error reading /proc/cpuinfo: {e}")

        return info

    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed GPU information.

        Returns:
            List of dictionaries containing GPU information
        """
        gpus = []

        if not TORCH_AVAILABLE:
            return gpus

        # CUDA GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = self._get_cuda_gpu_info(i)
                gpus.append(gpu_info)

        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = self._get_mps_gpu_info()
            gpus.append(gpu_info)

        return gpus

    def _get_cuda_gpu_info(self, device_id: int) -> Dict[str, Any]:
        """
        Get CUDA GPU information.

        Args:
            device_id: CUDA device ID

        Returns:
            Dictionary containing GPU information
        """
        info = {
            "id": device_id,
            "type": "cuda",
            "name": "Unknown",
            "memory_total": 0,
            "memory_free": 0,
            "compute_capability": "0.0",
            "multiprocessors": 0,
            "cuda_cores": 0,
            "tensor_cores": False,
            "features": [],
        }

        try:
            # Basic PyTorch info
            props = torch.cuda.get_device_properties(device_id)
            info["name"] = props.name
            info["memory_total"] = props.total_memory
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["multiprocessors"] = props.multi_processor_count

            # Estimate CUDA cores (rough approximation)
            # This is a simplified estimation based on compute capability
            if props.major >= 7:  # Volta/Turing/Ampere
                info["cuda_cores"] = props.multi_processor_count * 64
            elif props.major == 6:  # Pascal
                info["cuda_cores"] = props.multi_processor_count * 64
            elif props.major == 5:  # Maxwell
                info["cuda_cores"] = props.multi_processor_count * 128

            # Check for Tensor Cores (Volta and later)
            info["tensor_cores"] = props.major >= 7

            # Memory information
            info["memory_free"] = torch.cuda.memory_reserved(device_id)

            # Features
            features = []
            if props.major >= 7:
                features.append("tensor_cores")
            if props.major >= 8:
                features.append("ampere_features")

            info["features"] = features

        except Exception as e:
            logger.debug(f"Error getting CUDA GPU {device_id} info: {e}")

        return info

    def _get_mps_gpu_info(self) -> Dict[str, Any]:
        """
        Get Apple Metal Performance Shaders GPU information.

        Returns:
            Dictionary containing MPS GPU information
        """
        info = {
            "id": 0,
            "type": "mps",
            "name": "Apple Silicon GPU",
            "memory_total": 0,
            "memory_free": 0,
            "features": ["metal", "unified_memory"],
        }

        try:
            # Try to get more specific information about Apple Silicon
            if platform.system() == "Darwin":
                # Get system memory as unified memory approximation
                if PSUTIL_AVAILABLE:
                    memory = psutil.virtual_memory()
                    # Apple Silicon uses unified memory
                    info["memory_total"] = memory.total
                    info["memory_free"] = memory.available

        except Exception as e:
            logger.debug(f"Error getting MPS GPU info: {e}")

        return info

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get system memory information.

        Returns:
            Dictionary containing memory information
        """
        info = {
            "total_gb": 0.0,
            "available_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
            "swap_total_gb": 0.0,
            "swap_used_gb": 0.0,
        }

        if PSUTIL_AVAILABLE:
            try:
                # Virtual memory
                vmem = psutil.virtual_memory()
                info["total_gb"] = vmem.total / (1024**3)
                info["available_gb"] = vmem.available / (1024**3)
                info["used_gb"] = vmem.used / (1024**3)
                info["free_gb"] = vmem.free / (1024**3)

                # Swap memory
                swap = psutil.swap_memory()
                info["swap_total_gb"] = swap.total / (1024**3)
                info["swap_used_gb"] = swap.used / (1024**3)

            except Exception as e:
                logger.debug(f"Error getting memory info: {e}")

        return info

    def get_platform_info(self) -> Dict[str, Any]:
        """
        Get platform and OS information.

        Returns:
            Dictionary containing platform information
        """
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }

        # Add distribution info for Linux
        if platform.system() == "Linux":
            try:
                # Try to get distribution info
                with open("/etc/os-release", "r") as f:
                    os_release = {}
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            os_release[key] = value.strip('"')

                    info["distribution"] = os_release.get("NAME", "Unknown")
                    info["distribution_version"] = os_release.get("VERSION", "Unknown")

            except Exception as e:
                logger.debug(f"Error reading OS release info: {e}")

        return info


class DeviceOptimizer:
    """
    Device-specific optimization utilities.
    """

    def __init__(self, device_info: DeviceInfo):
        """
        Initialize device optimizer.

        Args:
            device_info: Device information
        """
        self.device_info = device_info
        self.optimizations_applied: List[str] = []

    def optimize_for_device(self) -> Dict[str, Any]:
        """
        Apply device-specific optimizations.

        Returns:
            Dictionary containing optimization results
        """
        optimizations = {}

        if self.device_info.device_type == "cuda":
            optimizations.update(self._optimize_cuda())
        elif self.device_info.device_type == "cpu":
            optimizations.update(self._optimize_cpu())
        elif self.device_info.device_type == "mps":
            optimizations.update(self._optimize_mps())
        elif self.device_info.device_type == "edge":
            optimizations.update(self._optimize_edge())

        return optimizations

    def _optimize_cuda(self) -> Dict[str, Any]:
        """
        Apply CUDA-specific optimizations.

        Returns:
            Dictionary containing CUDA optimizations
        """
        optimizations = {}

        if not TORCH_AVAILABLE:
            return optimizations

        try:
            # Enable optimized attention if available
            if hasattr(torch.backends.cuda, "sdp_kernel"):
                torch.backends.cuda.enable_flash_sdp(True)
                optimizations["flash_attention"] = True

            # Enable TensorFloat-32 on Ampere GPUs
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                optimizations["tf32"] = True

            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            optimizations["cudnn_benchmark"] = True

            # Set memory format to channels last for better performance
            optimizations["channels_last"] = True

            self.optimizations_applied.extend(optimizations.keys())

        except Exception as e:
            logger.warning(f"Error applying CUDA optimizations: {e}")

        return optimizations

    def _optimize_cpu(self) -> Dict[str, Any]:
        """
        Apply CPU-specific optimizations.

        Returns:
            Dictionary containing CPU optimizations
        """
        optimizations = {}

        try:
            # Set optimal number of threads
            if TORCH_AVAILABLE:
                num_threads = min(self.device_info.cores, 8)  # Cap at 8 for diminishing returns
                torch.set_num_threads(num_threads)
                optimizations["num_threads"] = num_threads

                # Enable CPU optimizations
                torch.set_num_interop_threads(1)  # Reduce thread contention
                optimizations["interop_threads"] = 1

            # Set OpenMP threads
            os.environ["OMP_NUM_THREADS"] = str(min(self.device_info.cores, 8))
            optimizations["omp_threads"] = min(self.device_info.cores, 8)

            # Set Intel MKL threads if available
            os.environ["MKL_NUM_THREADS"] = str(min(self.device_info.cores, 8))
            optimizations["mkl_threads"] = min(self.device_info.cores, 8)

            self.optimizations_applied.extend(optimizations.keys())

        except Exception as e:
            logger.warning(f"Error applying CPU optimizations: {e}")

        return optimizations

    def _optimize_mps(self) -> Dict[str, Any]:
        """
        Apply Apple Silicon MPS optimizations.

        Returns:
            Dictionary containing MPS optimizations
        """
        optimizations = {}

        try:
            # MPS-specific optimizations
            optimizations["unified_memory"] = True
            optimizations["metal_performance_shaders"] = True

            self.optimizations_applied.extend(optimizations.keys())

        except Exception as e:
            logger.warning(f"Error applying MPS optimizations: {e}")

        return optimizations

    def _optimize_edge(self) -> Dict[str, Any]:
        """
        Apply edge device optimizations.

        Returns:
            Dictionary containing edge optimizations
        """
        optimizations = {}

        try:
            # Conservative thread settings for edge devices
            if TORCH_AVAILABLE:
                num_threads = min(self.device_info.cores, 4)
                torch.set_num_threads(num_threads)
                optimizations["num_threads"] = num_threads

            # Memory optimization
            optimizations["memory_efficient"] = True

            # Power optimization
            optimizations["power_efficient"] = True

            self.optimizations_applied.extend(optimizations.keys())

        except Exception as e:
            logger.warning(f"Error applying edge optimizations: {e}")

        return optimizations


class PerformanceProfiler:
    """
    Profile device performance for different tasks.
    """

    def __init__(self, device: str):
        """
        Initialize performance profiler.

        Args:
            device: Device to profile
        """
        self.device = device
        self.profiles: List[PerformanceProfile] = []

    def profile_inference(
        self,
        model,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 4, 8, 16],
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> List[PerformanceProfile]:
        """
        Profile inference performance.

        Args:
            model: Model to profile
            input_shape: Input tensor shape
            batch_sizes: Batch sizes to test
            num_warmup: Number of warmup runs
            num_runs: Number of measurement runs

        Returns:
            List of performance profiles
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for profiling")
            return []

        profiles = []

        for batch_size in batch_sizes:
            try:
                profile = self._profile_single_batch_size(
                    model, input_shape, batch_size, num_warmup, num_runs
                )
                profiles.append(profile)

            except Exception as e:
                logger.warning(f"Error profiling batch size {batch_size}: {e}")

        self.profiles.extend(profiles)
        return profiles

    def _profile_single_batch_size(
        self, model, input_shape: Tuple[int, ...], batch_size: int, num_warmup: int, num_runs: int
    ) -> PerformanceProfile:
        """
        Profile a single batch size.

        Args:
            model: Model to profile
            input_shape: Input tensor shape
            batch_size: Batch size to test
            num_warmup: Number of warmup runs
            num_runs: Number of measurement runs

        Returns:
            Performance profile
        """
        import time

        # Create input tensor
        full_shape = (batch_size,) + input_shape[1:]
        x = torch.randn(full_shape, device=self.device)

        model.eval()
        model.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)

        # Synchronize if CUDA
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        # Measure
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_runs
        avg_time_per_sample = avg_time_per_batch / batch_size

        throughput = 1.0 / avg_time_per_sample  # samples per second
        latency = avg_time_per_sample * 1000  # milliseconds

        # Memory usage
        memory_usage = 0.0
        if self.device.startswith("cuda"):
            memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB

        return PerformanceProfile(
            device=self.device,
            task="inference",
            batch_size=batch_size,
            throughput=throughput,
            latency=latency,
            memory_usage=memory_usage,
        )


# Convenience functions
def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information.

    Returns:
        Dictionary containing device information
    """
    detector = DeviceDetector()

    info = {
        "device_type": detector.detect_device_type(),
        "platform": detector.get_platform_info(),
        "cpu": detector.get_cpu_info(),
        "memory": detector.get_memory_info(),
        "gpus": detector.get_gpu_info(),
    }

    # Add convenience fields
    info["gpu_available"] = len(info["gpus"]) > 0
    info["gpu_count"] = len(info["gpus"])

    if info["gpus"]:
        # Add primary GPU info
        primary_gpu = info["gpus"][0]
        info["gpu_name"] = primary_gpu.get("name", "Unknown")
        info["gpu_memory_gb"] = primary_gpu.get("memory_total", 0) / (1024**3)

    return info


def get_optimal_device(task: str = "inference", memory_requirement_gb: float = 1.0) -> str:
    """
    Get the optimal device for a specific task.

    Args:
        task: Task type ('training', 'inference', 'edge')
        memory_requirement_gb: Memory requirement in GB

    Returns:
        Optimal device string
    """
    device_info = get_device_info()

    # For training, prefer GPU with sufficient memory
    if task == "training":
        if device_info["gpu_available"]:
            for gpu in device_info["gpus"]:
                gpu_memory_gb = gpu.get("memory_total", 0) / (1024**3)
                if gpu_memory_gb >= memory_requirement_gb:
                    if gpu["type"] == "cuda":
                        return f"cuda:{gpu['id']}"
                    elif gpu["type"] == "mps":
                        return "mps"

            # Fallback to first GPU if memory requirement not met
            if device_info["gpus"]:
                gpu = device_info["gpus"][0]
                if gpu["type"] == "cuda":
                    return f"cuda:{gpu['id']}"
                elif gpu["type"] == "mps":
                    return "mps"

        return "cpu"

    # For inference, prefer any available GPU
    elif task == "inference":
        if device_info["gpu_available"]:
            gpu = device_info["gpus"][0]
            if gpu["type"] == "cuda":
                return f"cuda:{gpu['id']}"
            elif gpu["type"] == "mps":
                return "mps"

        return "cpu"

    # For edge deployment, prefer CPU or edge-optimized devices
    elif task == "edge":
        if device_info["device_type"] == "edge":
            # Check if CUDA is available on edge (e.g., Jetson)
            if device_info["gpu_available"]:
                gpu = device_info["gpus"][0]
                if gpu["type"] == "cuda":
                    return f"cuda:{gpu['id']}"

            return "cpu"
        else:
            return "cpu"

    # Default fallback
    return "cpu"


def check_gpu_availability() -> Tuple[bool, List[str]]:
    """
    Check GPU availability and return device list.

    Returns:
        Tuple of (gpu_available, device_list)
    """
    device_list = []

    if not TORCH_AVAILABLE:
        return False, device_list

    # Check CUDA
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_list.append(f"cuda:{i}")

    # Check MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_list.append("mps")

    return len(device_list) > 0, device_list


def get_memory_info(device: Optional[str] = None) -> Dict[str, float]:
    """
    Get memory information for a specific device.

    Args:
        device: Device string (e.g., 'cuda:0', 'cpu')

    Returns:
        Dictionary containing memory information in GB
    """
    memory_info = {}

    if device is None or device == "cpu":
        # System memory
        if PSUTIL_AVAILABLE:
            vmem = psutil.virtual_memory()
            memory_info.update(
                {
                    "total": vmem.total / (1024**3),
                    "available": vmem.available / (1024**3),
                    "used": vmem.used / (1024**3),
                    "free": vmem.free / (1024**3),
                }
            )

    elif device.startswith("cuda") and TORCH_AVAILABLE:
        # GPU memory
        try:
            device_id = int(device.split(":")[1]) if ":" in device else 0

            memory_info.update(
                {
                    "total": torch.cuda.get_device_properties(device_id).total_memory / (1024**3),
                    "allocated": torch.cuda.memory_allocated(device_id) / (1024**3),
                    "reserved": torch.cuda.memory_reserved(device_id) / (1024**3),
                    "free": (
                        torch.cuda.get_device_properties(device_id).total_memory
                        - torch.cuda.memory_reserved(device_id)
                    )
                    / (1024**3),
                }
            )

        except Exception as e:
            logger.warning(f"Error getting GPU memory info: {e}")

    return memory_info


def optimize_device(device: str) -> Dict[str, Any]:
    """
    Apply device-specific optimizations.

    Args:
        device: Device string

    Returns:
        Dictionary containing applied optimizations
    """
    device_info_dict = get_device_info()

    # Create DeviceInfo object
    device_info = DeviceInfo(
        device_type=device_info_dict["device_type"],
        device_name=device_info_dict.get("gpu_name", "CPU"),
        memory_total_gb=device_info_dict.get("gpu_memory_gb", 0),
        cores=device_info_dict["cpu"]["cores_logical"],
        architecture=device_info_dict["cpu"]["architecture"],
        platform_info=device_info_dict["platform"]["system"],
    )

    optimizer = DeviceOptimizer(device_info)
    return optimizer.optimize_for_device()


def monitor_device_resources(device: str, duration: float = 1.0) -> Dict[str, Any]:
    """
    Monitor device resource usage.

    Args:
        device: Device to monitor
        duration: Monitoring duration in seconds

    Returns:
        Dictionary containing resource usage statistics
    """
    import time

    stats = {
        "cpu_percent": [],
        "memory_percent": [],
        "gpu_utilization": [],
        "gpu_memory_percent": [],
    }

    start_time = time.time()

    while time.time() - start_time < duration:
        # CPU and system memory
        if PSUTIL_AVAILABLE:
            stats["cpu_percent"].append(psutil.cpu_percent())
            stats["memory_percent"].append(psutil.virtual_memory().percent)

        # GPU stats
        if device.startswith("cuda") and TORCH_AVAILABLE:
            try:
                device_id = int(device.split(":")[1]) if ":" in device else 0

                # Memory usage
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                used_memory = torch.cuda.memory_allocated(device_id)
                memory_percent = (used_memory / total_memory) * 100
                stats["gpu_memory_percent"].append(memory_percent)

                # GPU utilization (if NVML available)
                if NVML_AVAILABLE:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        stats["gpu_utilization"].append(util.gpu)
                    except:
                        pass

            except Exception as e:
                logger.debug(f"Error monitoring GPU: {e}")

        time.sleep(0.1)  # 100ms sampling rate

    # Calculate statistics
    result = {}
    for key, values in stats.items():
        if values:
            result[key] = {
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
            }

    return result


class DeviceManager:
    """
    Simple device manager for training.

    Wraps device detection and selection functionality for use in training scripts.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize device manager.

        Args:
            device: Optional device string. If None, will auto-detect optimal device.
        """
        self._device = device
        self._device_info = None

    def get_device(self) -> str:
        """
        Get the device to use for training.

        Returns:
            Device string (e.g., 'cuda:0', 'cpu', 'mps')
        """
        if self._device is not None:
            return self._device

        # Auto-detect optimal device
        return get_optimal_device(task="training")

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information.

        Returns:
            Dictionary containing device information
        """
        if self._device_info is None:
            self._device_info = get_device_info()
        return self._device_info

    def optimize_device(self, device: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply device-specific optimizations.

        Args:
            device: Optional device string. If None, uses current device.

        Returns:
            Dictionary containing applied optimizations
        """
        target_device = device or self.get_device()
        return optimize_device(target_device)
