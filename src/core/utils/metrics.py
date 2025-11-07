#
# Plan:
# 1. Create comprehensive metrics collection system for SereneSense deployment
# 2. Performance metrics (latency, throughput, accuracy, resource usage)
# 3. Business metrics (detections, alerts, user activity)
# 4. System metrics (CPU, memory, GPU, disk, network)
# 5. Time-series data storage and aggregation
# 6. Metrics export for external monitoring systems (Prometheus, InfluxDB)
# 7. Real-time dashboards and alerting integration
#

"""
Metrics Collector for SereneSense Deployment
Collects, stores, and exports performance and system metrics.

Features:
- Comprehensive metrics collection
- Time-series data storage
- Real-time metric aggregation
- Export to monitoring systems
- Dashboard integration
- Performance analytics
"""

import time
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
import csv
from pathlib import Path

import psutil

try:
    import prometheus_client

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    prometheus_client = None

logger = logging.getLogger(__name__)


def compute_classification_metrics(predictions, targets):
    """
    Compute classification metrics.

    Args:
        predictions: Predicted labels or logits
        targets: Ground truth labels

    Returns:
        Dictionary with accuracy, precision, recall, and f1 metrics
    """
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        import numpy as np

        # Convert to numpy if needed
        if hasattr(predictions, 'cpu'):
            predictions = predictions.cpu().numpy()
        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()

        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        # If predictions are probabilities/logits, get argmax
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)

        # Compute metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    except ImportError:
        # Fallback if sklearn not available
        import numpy as np
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        if predictions.ndim > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)

        accuracy = float(np.mean(predictions == targets))
        return {
            'accuracy': accuracy,
            'precision': accuracy,
            'recall': accuracy,
            'f1': accuracy
        }


@dataclass
class MetricData:
    """Individual metric data point"""

    name: str
    value: Union[int, float]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


@dataclass
class MetricSummary:
    """Summary statistics for a metric"""

    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "p99": self.p99,
        }


class TimeSeriesBuffer:
    """
    Time-series buffer for storing metric data points.
    Automatically handles data retention and aggregation.
    """

    def __init__(self, name: str, max_points: int = 10000, retention_hours: int = 24):
        """
        Initialize time-series buffer.

        Args:
            name: Metric name
            max_points: Maximum number of data points to store
            retention_hours: Data retention in hours
        """
        self.name = name
        self.max_points = max_points
        self.retention_seconds = retention_hours * 3600
        self.data_points = deque(maxlen=max_points)
        self.lock = threading.Lock()

    def add_point(
        self, value: Union[int, float], timestamp: float = None, labels: Dict[str, str] = None
    ):
        """
        Add data point to buffer.

        Args:
            value: Metric value
            timestamp: Timestamp (current time if None)
            labels: Metric labels
        """
        if timestamp is None:
            timestamp = time.time()

        metric_data = MetricData(
            name=self.name, value=value, timestamp=timestamp, labels=labels or {}
        )

        with self.lock:
            self.data_points.append(metric_data)
            self._cleanup_old_data()

    def _cleanup_old_data(self):
        """Remove data points older than retention period"""
        cutoff_time = time.time() - self.retention_seconds

        while self.data_points and self.data_points[0].timestamp < cutoff_time:
            self.data_points.popleft()

    def get_points(self, since: float = None, labels: Dict[str, str] = None) -> List[MetricData]:
        """
        Get data points from buffer.

        Args:
            since: Return points since this timestamp
            labels: Filter by labels

        Returns:
            List of matching data points
        """
        with self.lock:
            points = list(self.data_points)

        # Filter by timestamp
        if since is not None:
            points = [p for p in points if p.timestamp >= since]

        # Filter by labels
        if labels:
            points = [p for p in points if all(p.labels.get(k) == v for k, v in labels.items())]

        return points

    def get_summary(
        self, since: float = None, labels: Dict[str, str] = None
    ) -> Optional[MetricSummary]:
        """
        Get summary statistics for the metric.

        Args:
            since: Calculate summary since this timestamp
            labels: Filter by labels

        Returns:
            Metric summary or None if no data
        """
        points = self.get_points(since, labels)

        if not points:
            return None

        values = [p.value for p in points]
        values.sort()

        count = len(values)
        total = sum(values)

        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = k - f
            if f == len(data) - 1:
                return data[f]
            return data[f] * (1 - c) + data[f + 1] * c

        return MetricSummary(
            name=self.name,
            count=count,
            sum=total,
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            p95=percentile(values, 95),
            p99=percentile(values, 99),
        )

    def get_rate(self, window_seconds: int = 60) -> float:
        """
        Get rate of metric per second over time window.

        Args:
            window_seconds: Time window for rate calculation

        Returns:
            Rate per second
        """
        since = time.time() - window_seconds
        points = self.get_points(since)

        if len(points) < 2:
            return 0.0

        time_span = points[-1].timestamp - points[0].timestamp
        if time_span <= 0:
            return 0.0

        return len(points) / time_span


class PerformanceMetrics:
    """Performance metrics for SereneSense system"""

    def __init__(self):
        """Initialize performance metrics"""
        self.metrics = {}
        self.start_time = time.time()

        # Request metrics
        self.request_count = 0
        self.request_latencies = TimeSeriesBuffer("request_latency_ms")
        self.request_rates = TimeSeriesBuffer("requests_per_second")

        # Detection metrics
        self.detection_count = 0
        self.detection_confidences = TimeSeriesBuffer("detection_confidence")
        self.detection_rates = TimeSeriesBuffer("detections_per_minute")

        # Error metrics
        self.error_count = 0
        self.error_rates = TimeSeriesBuffer("errors_per_minute")

        # Model metrics
        self.inference_latencies = TimeSeriesBuffer("inference_latency_ms")
        self.model_accuracy = TimeSeriesBuffer("model_accuracy")

        # System metrics
        self.cpu_usage = TimeSeriesBuffer("cpu_percent")
        self.memory_usage = TimeSeriesBuffer("memory_percent")
        self.gpu_usage = TimeSeriesBuffer("gpu_percent")
        self.disk_usage = TimeSeriesBuffer("disk_percent")

    def record_request(self, latency_ms: float, endpoint: str = None, status_code: int = 200):
        """Record API request metrics"""
        self.request_count += 1
        self.request_latencies.add_point(latency_ms, labels={"endpoint": endpoint or "unknown"})

        # Record error if status code indicates failure
        if status_code >= 400:
            self.error_count += 1

    def record_detection(self, confidence: float, label: str = None):
        """Record detection metrics"""
        self.detection_count += 1
        self.detection_confidences.add_point(confidence, labels={"label": label or "unknown"})

    def record_inference(self, latency_ms: float, model_type: str = None):
        """Record model inference metrics"""
        self.inference_latencies.add_point(latency_ms, labels={"model": model_type or "unknown"})

    def record_accuracy(self, accuracy: float, dataset: str = None):
        """Record model accuracy metrics"""
        self.model_accuracy.add_point(accuracy, labels={"dataset": dataset or "unknown"})

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        now = time.time()
        uptime = now - self.start_time

        # Request metrics
        request_summary = self.request_latencies.get_summary(since=now - 3600)  # Last hour
        request_rate = self.request_latencies.get_rate(60)  # Per minute

        # Detection metrics
        detection_summary = self.detection_confidences.get_summary(since=now - 3600)
        detection_rate = self.detection_confidences.get_rate(60)

        # Inference metrics
        inference_summary = self.inference_latencies.get_summary(since=now - 3600)

        return {
            "uptime_seconds": uptime,
            "timestamp": now,
            "requests": {
                "total": self.request_count,
                "rate_per_minute": request_rate * 60,
                "latency_summary": request_summary.to_dict() if request_summary else None,
            },
            "detections": {
                "total": self.detection_count,
                "rate_per_minute": detection_rate * 60,
                "confidence_summary": detection_summary.to_dict() if detection_summary else None,
            },
            "inference": {
                "latency_summary": inference_summary.to_dict() if inference_summary else None
            },
            "errors": {
                "total": self.error_count,
                "rate_per_minute": self.error_rates.get_rate(60) * 60,
            },
        }


class SystemMetricsCollector:
    """Collects system resource metrics"""

    def __init__(self, collection_interval: float = 5.0):
        """
        Initialize system metrics collector.

        Args:
            collection_interval: Collection interval in seconds
        """
        self.collection_interval = collection_interval
        self.collecting = False
        self.collector_thread = None

        # Metric buffers
        self.cpu_usage = TimeSeriesBuffer("cpu_percent")
        self.memory_usage = TimeSeriesBuffer("memory_usage")
        self.disk_usage = TimeSeriesBuffer("disk_usage")
        self.network_io = TimeSeriesBuffer("network_io")

        # GPU metrics (if available)
        self.gpu_available = self._check_gpu_availability()
        if self.gpu_available:
            self.gpu_usage = TimeSeriesBuffer("gpu_usage")
            self.gpu_memory = TimeSeriesBuffer("gpu_memory")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def start_collection(self):
        """Start system metrics collection"""
        if self.collecting:
            logger.warning("System metrics collection already running")
            return

        self.collecting = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()

        logger.info(f"Started system metrics collection (interval: {self.collection_interval}s)")

    def stop_collection(self):
        """Stop system metrics collection"""
        self.collecting = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)

        logger.info("Stopped system metrics collection")

    def _collection_loop(self):
        """Main collection loop"""
        while self.collecting:
            try:
                timestamp = time.time()

                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_usage.add_point(cpu_percent, timestamp)

                # Memory metrics
                memory = psutil.virtual_memory()
                self.memory_usage.add_point(memory.percent, timestamp, labels={"type": "virtual"})

                # Disk metrics
                disk = psutil.disk_usage("/")
                disk_percent = (disk.used / disk.total) * 100
                self.disk_usage.add_point(disk_percent, timestamp)

                # Network metrics
                net_io = psutil.net_io_counters()
                if hasattr(self, "_prev_net_io"):
                    bytes_sent_rate = (
                        net_io.bytes_sent - self._prev_net_io.bytes_sent
                    ) / self.collection_interval
                    bytes_recv_rate = (
                        net_io.bytes_recv - self._prev_net_io.bytes_recv
                    ) / self.collection_interval

                    self.network_io.add_point(
                        bytes_sent_rate, timestamp, labels={"direction": "sent"}
                    )
                    self.network_io.add_point(
                        bytes_recv_rate, timestamp, labels={"direction": "received"}
                    )

                self._prev_net_io = net_io

                # GPU metrics
                if self.gpu_available:
                    self._collect_gpu_metrics(timestamp)

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in system metrics collection: {e}")
                time.sleep(self.collection_interval)

    def _collect_gpu_metrics(self, timestamp: float):
        """Collect GPU metrics"""
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                # GPU utilization (simplified - would need nvidia-ml-py for accurate data)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory

                memory_percent = (memory_reserved / memory_total) * 100

                self.gpu_memory.add_point(memory_percent, timestamp, labels={"device": str(i)})

        except Exception as e:
            logger.debug(f"GPU metrics collection failed: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        now = time.time()

        # Get latest values
        cpu_summary = self.cpu_usage.get_summary(since=now - 60)
        memory_summary = self.memory_usage.get_summary(since=now - 60)
        disk_summary = self.disk_usage.get_summary(since=now - 60)

        metrics = {
            "timestamp": now,
            "cpu": cpu_summary.to_dict() if cpu_summary else None,
            "memory": memory_summary.to_dict() if memory_summary else None,
            "disk": disk_summary.to_dict() if disk_summary else None,
        }

        if self.gpu_available:
            gpu_summary = self.gpu_memory.get_summary(since=now - 60)
            metrics["gpu"] = gpu_summary.to_dict() if gpu_summary else None

        return metrics


class PrometheusExporter:
    """Exports metrics to Prometheus format"""

    def __init__(self, registry=None):
        """
        Initialize Prometheus exporter.

        Args:
            registry: Prometheus registry (creates default if None)
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus client not available")

        self.registry = registry or prometheus_client.CollectorRegistry()
        self.metrics = {}

        # Create Prometheus metrics
        self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metric objects"""
        # Counter metrics
        self.request_counter = prometheus_client.Counter(
            "serenesense_requests_total",
            "Total number of requests",
            ["endpoint", "status"],
            registry=self.registry,
        )

        self.detection_counter = prometheus_client.Counter(
            "serenesense_detections_total",
            "Total number of detections",
            ["label"],
            registry=self.registry,
        )

        # Histogram metrics
        self.request_duration = prometheus_client.Histogram(
            "serenesense_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint"],
            registry=self.registry,
        )

        self.inference_duration = prometheus_client.Histogram(
            "serenesense_inference_duration_seconds",
            "Inference duration in seconds",
            ["model"],
            registry=self.registry,
        )

        # Gauge metrics
        self.system_cpu = prometheus_client.Gauge(
            "serenesense_cpu_percent", "CPU usage percentage", registry=self.registry
        )

        self.system_memory = prometheus_client.Gauge(
            "serenesense_memory_percent", "Memory usage percentage", registry=self.registry
        )

    def update_from_performance_metrics(self, perf_metrics: PerformanceMetrics):
        """Update Prometheus metrics from performance metrics"""
        # Get recent metrics
        now = time.time()

        # Request metrics
        recent_requests = perf_metrics.request_latencies.get_points(since=now - 60)
        for point in recent_requests:
            endpoint = point.labels.get("endpoint", "unknown")
            self.request_duration.labels(endpoint=endpoint).observe(point.value / 1000)

        # Detection metrics
        recent_detections = perf_metrics.detection_confidences.get_points(since=now - 60)
        for point in recent_detections:
            label = point.labels.get("label", "unknown")
            self.detection_counter.labels(label=label).inc()

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        return prometheus_client.generate_latest(self.registry).decode("utf-8")


class MetricsCollector:
    """
    Main metrics collector that coordinates all metric collection.
    Provides unified interface for metrics management.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize metrics collector.

        Args:
            config: Metrics configuration
        """
        self.config = config or {}

        # Initialize metric components
        self.performance_metrics = PerformanceMetrics()
        self.system_metrics = SystemMetricsCollector(
            collection_interval=self.config.get("system_collection_interval", 5.0)
        )

        # Prometheus exporter (if enabled)
        self.prometheus_exporter = None
        if self.config.get("prometheus_enabled", False) and PROMETHEUS_AVAILABLE:
            self.prometheus_exporter = PrometheusExporter()

        # Metrics export
        self.export_enabled = self.config.get("export_enabled", False)
        self.export_interval = self.config.get("export_interval", 60)
        self.export_path = self.config.get("export_path", "metrics")

        # Export thread
        self.exporting = False
        self.export_thread = None

        logger.info("Metrics collector initialized")

    def start_collection(self):
        """Start all metrics collection"""
        # Start system metrics collection
        self.system_metrics.start_collection()

        # Start metrics export if enabled
        if self.export_enabled:
            self._start_export()

        logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop all metrics collection"""
        # Stop system metrics
        self.system_metrics.stop_collection()

        # Stop export
        if self.exporting:
            self._stop_export()

        logger.info("Metrics collection stopped")

    def record_request(self, method: str, endpoint: str, status_code: int, processing_time: float):
        """
        Record API request metrics.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            processing_time: Processing time in seconds
        """
        latency_ms = processing_time * 1000
        self.performance_metrics.record_request(latency_ms, endpoint, status_code)

        # Update Prometheus metrics if available
        if self.prometheus_exporter:
            self.prometheus_exporter.request_counter.labels(
                endpoint=endpoint, status=str(status_code)
            ).inc()

            self.prometheus_exporter.request_duration.labels(endpoint=endpoint).observe(
                processing_time
            )

    def record_detections(self, count: int):
        """Record number of detections"""
        for _ in range(count):
            self.performance_metrics.record_detection(1.0)  # Simplified

    def record_detection(self, label: str, confidence: float):
        """
        Record individual detection.

        Args:
            label: Detection label
            confidence: Detection confidence
        """
        self.performance_metrics.record_detection(confidence, label)

        if self.prometheus_exporter:
            self.prometheus_exporter.detection_counter.labels(label=label).inc()

    def record_error(self):
        """Record error occurrence"""
        self.performance_metrics.error_count += 1

    def record_inference_time(self, latency_ms: float, model_type: str = None):
        """
        Record model inference time.

        Args:
            latency_ms: Inference latency in milliseconds
            model_type: Type of model used
        """
        self.performance_metrics.record_inference(latency_ms, model_type)

        if self.prometheus_exporter:
            self.prometheus_exporter.inference_duration.labels(
                model=model_type or "unknown"
            ).observe(latency_ms / 1000)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        metrics = {
            "timestamp": time.time(),
            "performance": self.performance_metrics.get_summary(),
            "system": self.system_metrics.get_current_metrics(),
        }

        return metrics

    def get_prometheus_metrics(self) -> Optional[str]:
        """Get Prometheus format metrics"""
        if self.prometheus_exporter:
            # Update from current metrics
            self.prometheus_exporter.update_from_performance_metrics(self.performance_metrics)

            # Update system metrics
            system_metrics = self.system_metrics.get_current_metrics()
            if system_metrics.get("cpu"):
                self.prometheus_exporter.system_cpu.set(system_metrics["cpu"]["mean"])
            if system_metrics.get("memory"):
                self.prometheus_exporter.system_memory.set(system_metrics["memory"]["mean"])

            return self.prometheus_exporter.get_metrics_text()

        return None

    def _start_export(self):
        """Start metrics export thread"""
        self.exporting = True
        self.export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self.export_thread.start()

        logger.info(f"Started metrics export (interval: {self.export_interval}s)")

    def _stop_export(self):
        """Stop metrics export thread"""
        self.exporting = False
        if self.export_thread:
            self.export_thread.join(timeout=5)

        logger.info("Stopped metrics export")

    def _export_loop(self):
        """Main export loop"""
        export_dir = Path(self.export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        while self.exporting:
            try:
                # Get current metrics
                metrics = self.get_metrics()

                # Export to JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_file = export_dir / f"metrics_{timestamp}.json"

                with open(json_file, "w") as f:
                    json.dump(metrics, f, indent=2, default=str)

                # Export to CSV (performance metrics only)
                self._export_performance_csv(export_dir, timestamp, metrics["performance"])

                # Cleanup old export files (keep last 24 hours)
                self._cleanup_export_files(export_dir)

                time.sleep(self.export_interval)

            except Exception as e:
                logger.error(f"Error in metrics export: {e}")
                time.sleep(self.export_interval)

    def _export_performance_csv(
        self, export_dir: Path, timestamp: str, performance_metrics: Dict[str, Any]
    ):
        """Export performance metrics to CSV"""
        try:
            csv_file = export_dir / f"performance_{timestamp}.csv"

            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(["metric", "value", "timestamp"])

                # Write performance data
                ts = performance_metrics.get("timestamp", time.time())

                # Request metrics
                if performance_metrics.get("requests"):
                    req_data = performance_metrics["requests"]
                    writer.writerow(["requests_total", req_data.get("total", 0), ts])
                    writer.writerow(
                        ["requests_rate_per_minute", req_data.get("rate_per_minute", 0), ts]
                    )

                # Detection metrics
                if performance_metrics.get("detections"):
                    det_data = performance_metrics["detections"]
                    writer.writerow(["detections_total", det_data.get("total", 0), ts])
                    writer.writerow(
                        ["detections_rate_per_minute", det_data.get("rate_per_minute", 0), ts]
                    )

        except Exception as e:
            logger.error(f"Error exporting performance CSV: {e}")

    def _cleanup_export_files(self, export_dir: Path):
        """Cleanup old export files"""
        try:
            cutoff_time = time.time() - 24 * 3600  # 24 hours ago

            for file_path in export_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()

        except Exception as e:
            logger.debug(f"Error cleaning up export files: {e}")


def create_metrics_collector(config_path: str = None) -> MetricsCollector:
    """
    Create metrics collector from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured metrics collector
    """
    from core.utils.config_parser import ConfigParser

    if config_path:
        config_dict = ConfigParser.load_config(config_path)
        metrics_config = config_dict.get("metrics", {})
    else:
        metrics_config = {}

    return MetricsCollector(metrics_config)


if __name__ == "__main__":
    # Demo: Metrics collection system
    import argparse

    parser = argparse.ArgumentParser(description="Metrics Collector Demo")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--export-path", default="metrics", help="Export directory")
    parser.add_argument("--duration", type=int, default=60, help="Demo duration in seconds")
    parser.add_argument("--prometheus", action="store_true", help="Enable Prometheus export")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Create metrics collector
        if args.config:
            collector = create_metrics_collector(args.config)
        else:
            config = {
                "export_enabled": True,
                "export_path": args.export_path,
                "export_interval": 10,
                "prometheus_enabled": args.prometheus,
            }
            collector = MetricsCollector(config)

        print("📊 SereneSense Metrics Collector Demo")
        print(f"   Export path: {args.export_path}")
        print(f"   Duration: {args.duration} seconds")

        # Start collection
        collector.start_collection()

        # Simulate some activity
        print("\n🔄 Simulating system activity...")

        for i in range(args.duration):
            # Simulate API requests
            collector.record_request(
                method="POST",
                endpoint="/detect",
                status_code=200,
                processing_time=0.05 + (i % 10) * 0.01,
            )

            # Simulate detections
            if i % 5 == 0:
                collector.record_detection("helicopter", 0.85 + (i % 20) * 0.01)

            # Simulate inference
            collector.record_inference_time(15.0 + (i % 30) * 0.5, "audioMAE")

            time.sleep(1)

            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{args.duration} seconds")

        # Get final metrics
        metrics = collector.get_metrics()

        print("\n📈 Final Metrics Summary:")
        if "performance" in metrics:
            perf = metrics["performance"]
            if "requests" in perf:
                print(f"   Total requests: {perf['requests'].get('total', 0)}")
                print(f"   Request rate: {perf['requests'].get('rate_per_minute', 0):.1f}/min")

            if "detections" in perf:
                print(f"   Total detections: {perf['detections'].get('total', 0)}")
                print(f"   Detection rate: {perf['detections'].get('rate_per_minute', 0):.1f}/min")

        # Show Prometheus metrics if enabled
        if args.prometheus:
            prom_metrics = collector.get_prometheus_metrics()
            if prom_metrics:
                print(f"\n🔍 Prometheus metrics: {len(prom_metrics.split(chr(10)))} lines")

        # Stop collection
        collector.stop_collection()
        print("\n✅ Demo completed")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Metrics collector demo failed: {e}")
