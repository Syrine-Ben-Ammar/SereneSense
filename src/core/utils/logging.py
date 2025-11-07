"""
Logging Utilities for SereneSense

This module provides comprehensive logging functionality for the SereneSense
military vehicle sound detection system. It includes structured logging,
performance monitoring, and integration with experiment tracking.

Features:
- Structured logging with JSON format support
- Performance timing and profiling
- Integration with MLflow and Weights & Biases
- Distributed training logging support
- Log rotation and archival
- Real-time log streaming for monitoring

Example:
    >>> from core.utils.logging import setup_logging, get_logger, log_metrics
    >>> 
    >>> # Setup logging
    >>> setup_logging(level="INFO", log_dir="logs")
    >>> 
    >>> # Get logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started")
    >>> 
    >>> # Log metrics
    >>> log_metrics({"accuracy": 0.91, "loss": 0.23}, step=100)
"""

import logging
import logging.handlers
import json
import time
import sys
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import functools
from contextlib import contextmanager

# Third-party imports
try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Global logger instances
_loggers: Dict[str, logging.Logger] = {}
_log_config: Dict[str, Any] = {}
_metrics_loggers: List[Any] = []


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Formats log records as JSON for easy parsing and analysis.
    """

    def __init__(self, include_extra: bool = True):
        """
        Initialize JSON formatter.

        Args:
            include_extra: Include extra fields in log record
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add thread information for distributed training
        if hasattr(record, "thread"):
            log_data["thread"] = record.thread
            log_data["thread_name"] = record.threadName

        # Add process information
        log_data["process"] = record.process
        log_data["process_name"] = getattr(record, "processName", "MainProcess")

        # Add exception information
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ]:
                    try:
                        json.dumps(value)  # Test if serializable
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)

        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """
    Logger for performance metrics and timing information.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.

        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.metrics: Dict[str, List[float]] = {}

    def start_timer(self, name: str) -> None:
        """
        Start a named timer.

        Args:
            name: Timer name
        """
        self.timers[name] = time.time()

    def end_timer(self, name: str, log_result: bool = True) -> float:
        """
        End a named timer and return elapsed time.

        Args:
            name: Timer name
            log_result: Whether to log the result

        Returns:
            Elapsed time in seconds
        """
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' not found")
            return 0.0

        elapsed = time.time() - self.timers[name]
        del self.timers[name]

        if log_result:
            self.logger.info(
                f"Timer '{name}': {elapsed:.4f}s",
                extra={"timer_name": name, "elapsed_time": elapsed},
            )

        # Store in metrics
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(elapsed)

        return elapsed

    def increment_counter(self, name: str, value: int = 1) -> None:
        """
        Increment a named counter.

        Args:
            name: Counter name
            value: Increment value
        """
        self.counters[name] = self.counters.get(name, 0) + value

    def log_counter(self, name: str) -> None:
        """
        Log current counter value.

        Args:
            name: Counter name
        """
        if name in self.counters:
            self.logger.info(
                f"Counter '{name}': {self.counters[name]}",
                extra={"counter_name": name, "counter_value": self.counters[name]},
            )

    def log_system_metrics(self) -> None:
        """
        Log system performance metrics.
        """
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            self.logger.info(
                "System metrics",
                extra={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_used_gb": memory.used / (1024**3),
                },
            )

            # Log GPU metrics if available
            try:
                import torch

                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)

                        self.logger.info(
                            f"GPU {i} metrics",
                            extra={
                                "gpu_id": i,
                                "gpu_memory_allocated_gb": memory_allocated,
                                "gpu_memory_reserved_gb": memory_reserved,
                            },
                        )
            except ImportError:
                pass

    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing code blocks.

        Args:
            name: Timer name

        Yields:
            None
        """
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)


class MetricsLogger:
    """
    Logger for machine learning metrics and experiment tracking.
    """

    def __init__(self, logger: logging.Logger, use_wandb: bool = False, use_mlflow: bool = False):
        """
        Initialize metrics logger.

        Args:
            logger: Base logger instance
            use_wandb: Enable Weights & Biases logging
            use_mlflow: Enable MLflow logging
        """
        self.logger = logger
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        if self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("Weights & Biases not available, disabling wandb logging")

        if self.use_mlflow and not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, disabling mlflow logging")

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""
    ) -> None:
        """
        Log metrics to all configured backends.

        Args:
            metrics: Dictionary of metric name -> value
            step: Step number for time series logging
            prefix: Prefix to add to metric names
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Log to standard logger
        metric_str = ", ".join(
            [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
        )

        extra_data = {"metrics": metrics}
        if step is not None:
            extra_data["step"] = step
            metric_str += f" (step: {step})"

        self.logger.info(f"Metrics - {metric_str}", extra=extra_data)

        # Log to Weights & Biases
        if self.use_wandb and wandb.run is not None:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log to wandb: {e}")

        # Log to MLflow
        if self.use_mlflow:
            try:
                if step is not None:
                    for name, value in metrics.items():
                        mlflow.log_metric(name, value, step=step)
                else:
                    mlflow.log_metrics(metrics)
            except Exception as e:
                self.logger.warning(f"Failed to log to mlflow: {e}")

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters/hyperparameters.

        Args:
            params: Dictionary of parameter name -> value
        """
        self.logger.info("Parameters", extra={"parameters": params})

        # Log to Weights & Biases
        if self.use_wandb and wandb.run is not None:
            try:
                wandb.config.update(params)
            except Exception as e:
                self.logger.warning(f"Failed to log parameters to wandb: {e}")

        # Log to MLflow
        if self.use_mlflow:
            try:
                mlflow.log_params(params)
            except Exception as e:
                self.logger.warning(f"Failed to log parameters to mlflow: {e}")

    def log_artifacts(self, artifacts: Dict[str, str]) -> None:
        """
        Log artifacts (files, models, etc.).

        Args:
            artifacts: Dictionary of artifact name -> file path
        """
        self.logger.info("Artifacts", extra={"artifacts": artifacts})

        # Log to Weights & Biases
        if self.use_wandb and wandb.run is not None:
            try:
                for name, path in artifacts.items():
                    wandb.save(path, base_path=os.path.dirname(path))
            except Exception as e:
                self.logger.warning(f"Failed to log artifacts to wandb: {e}")

        # Log to MLflow
        if self.use_mlflow:
            try:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path)
            except Exception as e:
                self.logger.warning(f"Failed to log artifacts to mlflow: {e}")


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    distributed_rank: Optional[int] = None,
) -> logging.Logger:
    """
    Setup comprehensive logging for SereneSense.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_file: Specific log file name
        json_format: Use JSON formatting
        console_output: Enable console output
        file_output: Enable file output
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        distributed_rank: Rank for distributed training (None for single process)

    Returns:
        Configured logger instance
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create log directory if needed
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Determine log file path
    if file_output and log_dir:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if distributed_rank is not None:
                log_file = f"serenesense_rank{distributed_rank}_{timestamp}.log"
            else:
                log_file = f"serenesense_{timestamp}.log"

        log_file_path = Path(log_dir) / log_file
    else:
        log_file_path = None

    # Get root logger for SereneSense
    logger = logging.getLogger("serenesense")
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        if COLORLOG_AVAILABLE and console_output:
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_output and log_file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)

        # Use JSON formatter for file output if requested
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    # Store configuration
    global _log_config
    _log_config = {
        "level": level,
        "log_dir": log_dir,
        "log_file": str(log_file_path) if log_file_path else None,
        "json_format": json_format,
        "console_output": console_output,
        "file_output": file_output,
        "distributed_rank": distributed_rank,
    }

    # Log setup information
    logger.info(
        f"Logging setup complete - Level: {level}, "
        f"File: {log_file_path if log_file_path else 'None'}"
    )

    return logger


def get_logger(
    name: str,
    performance_logging: bool = False,
    metrics_logging: bool = False,
    use_wandb: bool = False,
    use_mlflow: bool = False,
) -> logging.Logger:
    """
    Get a logger instance with optional performance and metrics logging.

    Args:
        name: Logger name (typically __name__)
        performance_logging: Enable performance logging features
        metrics_logging: Enable metrics logging features
        use_wandb: Enable Weights & Biases integration
        use_mlflow: Enable MLflow integration

    Returns:
        Configured logger instance with optional features
    """
    # Get or create logger
    if name not in _loggers:
        logger = logging.getLogger(name)

        # Add performance logging capability
        if performance_logging:
            logger.perf = PerformanceLogger(logger)

        # Add metrics logging capability
        if metrics_logging:
            logger.metrics = MetricsLogger(logger, use_wandb, use_mlflow)
            _metrics_loggers.append(logger.metrics)

        _loggers[name] = logger

    return _loggers[name]


def timing_decorator(logger_name: Optional[str] = None):
    """
    Decorator for timing function execution.

    Args:
        logger_name: Name of logger to use (uses function's module if None)

    Returns:
        Decorator function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger_name
            if logger_name is None:
                logger_name = func.__module__

            logger = get_logger(logger_name, performance_logging=True)

            if hasattr(logger, "perf"):
                timer_name = f"{func.__name__}"
                with logger.perf.timer(timer_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_metrics(
    metrics: Dict[str, Union[float, int]],
    step: Optional[int] = None,
    prefix: str = "",
    logger_name: str = "serenesense",
) -> None:
    """
    Convenience function to log metrics to all configured backends.

    Args:
        metrics: Dictionary of metric name -> value
        step: Step number for time series logging
        prefix: Prefix to add to metric names
        logger_name: Name of logger to use
    """
    logger = get_logger(logger_name, metrics_logging=True)

    if hasattr(logger, "metrics"):
        logger.metrics.log_metrics(metrics, step, prefix)
    else:
        # Fallback to standard logging
        metric_str = ", ".join(
            [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
        )
        if step is not None:
            metric_str += f" (step: {step})"
        logger.info(f"Metrics - {metric_str}")


def log_system_resources(logger_name: str = "core.system") -> None:
    """
    Log current system resource usage.

    Args:
        logger_name: Name of logger to use
    """
    logger = get_logger(logger_name, performance_logging=True)

    if hasattr(logger, "perf"):
        logger.perf.log_system_metrics()
    else:
        logger.warning("Performance logging not enabled for system resource logging")


def configure_distributed_logging(rank: int, world_size: int) -> None:
    """
    Configure logging for distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Only log to console from rank 0
    console_output = rank == 0

    # Setup logging with rank-specific configuration
    setup_logging(
        level="INFO",
        log_dir="logs/distributed",
        console_output=console_output,
        file_output=True,
        distributed_rank=rank,
    )

    logger = get_logger(f"core.rank{rank}")
    logger.info(f"Distributed logging configured - Rank: {rank}/{world_size}")


# Context manager for temporary log level changes
@contextmanager
def log_level(level: str, logger_name: str = "serenesense"):
    """
    Temporarily change log level for a code block.

    Args:
        level: Temporary log level
        logger_name: Logger to modify

    Yields:
        Logger instance
    """
    logger = get_logger(logger_name)
    original_level = logger.level

    try:
        logger.setLevel(getattr(logging, level.upper()))
        yield logger
    finally:
        logger.setLevel(original_level)


# Cleanup function
def cleanup_logging() -> None:
    """
    Cleanup logging resources and close handlers.
    """
    # Close all handlers
    for logger in _loggers.values():
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    # Clear logger cache
    _loggers.clear()
    _metrics_loggers.clear()

    print("Logging cleanup complete")


# Register cleanup function
import atexit

atexit.register(cleanup_logging)
