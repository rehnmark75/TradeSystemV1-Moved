# core/concurrent_execution/performance_counters.py
"""
Performance Counters - High-resolution performance monitoring and analytics
Provides microsecond-level timing and system performance insights
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import psutil
import os

# Try to import high-resolution timing
try:
    # Linux/Unix high-resolution timer
    CLOCK_MONOTONIC_RAW = time.CLOCK_MONOTONIC_RAW
    high_res_timer = lambda: time.clock_gettime(CLOCK_MONOTONIC_RAW)
except AttributeError:
    # Fallback to standard timer
    high_res_timer = time.perf_counter


class CounterType(Enum):
    """Types of performance counters"""
    TIMING = "timing"              # Execution time measurements
    THROUGHPUT = "throughput"      # Operations per second
    LATENCY = "latency"           # Response time distributions
    RESOURCE = "resource"         # CPU, memory, I/O usage
    CUSTOM = "custom"             # Application-specific metrics


@dataclass
class TimingMeasurement:
    """High-resolution timing measurement"""
    start_time: float
    end_time: float
    duration_us: float  # Duration in microseconds
    context: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0

    @property
    def duration_seconds(self) -> float:
        return self.duration_us / 1_000_000.0


@dataclass
class PerformanceStats:
    """Statistical analysis of performance measurements"""
    count: int = 0
    total: float = 0.0
    min_value: float = float('inf')
    max_value: float = 0.0

    # Percentile tracking
    p50: float = 0.0  # Median
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p99_9: float = 0.0

    # Recent measurements for trend analysis
    recent_values: deque = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)

    def update(self, value: float):
        """Update statistics with new measurement"""
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.recent_values.append(value)

        # Update percentiles
        if len(self.recent_values) >= 10:
            self._calculate_percentiles()

    def _calculate_percentiles(self):
        """Calculate percentile values from recent measurements"""
        sorted_values = sorted(self.recent_values)
        n = len(sorted_values)

        if n > 0:
            self.p50 = sorted_values[int(n * 0.50)]
            self.p90 = sorted_values[int(n * 0.90)]
            self.p95 = sorted_values[int(n * 0.95)]
            self.p99 = sorted_values[int(n * 0.99)]
            self.p99_9 = sorted_values[int(n * 0.999)]


class PerformanceCounter:
    """
    High-performance counter with microsecond resolution

    Features:
    - Sub-microsecond timing precision
    - Statistical analysis
    - Trend detection
    - Minimal overhead measurement
    - Thread-safe operations
    """

    def __init__(self,
                 name: str,
                 counter_type: CounterType,
                 max_history: int = 10000):

        self.name = name
        self.counter_type = counter_type
        self.max_history = max_history

        # Measurement storage
        self.measurements: deque[TimingMeasurement] = deque(maxlen=max_history)
        self.stats = PerformanceStats()

        # Thread safety
        self.lock = threading.Lock()

        # State for timing operations
        self.active_timers: Dict[int, float] = {}  # thread_id -> start_time

        # Performance optimization
        self.measurement_overhead_us = self._calibrate_measurement_overhead()

    def start_timer(self, context: Optional[str] = None) -> int:
        """
        Start a timing measurement

        Returns:
            timer_id: Unique identifier for this timing session
        """
        thread_id = threading.get_ident()
        start_time = high_res_timer()

        with self.lock:
            self.active_timers[thread_id] = start_time

        return thread_id

    def stop_timer(self, timer_id: Optional[int] = None, context: Optional[str] = None) -> float:
        """
        Stop a timing measurement

        Args:
            timer_id: Timer ID from start_timer (uses current thread if None)
            context: Optional context description

        Returns:
            duration_us: Duration in microseconds
        """
        end_time = high_res_timer()
        thread_id = timer_id or threading.get_ident()

        with self.lock:
            start_time = self.active_timers.pop(thread_id, None)

            if start_time is None:
                return 0.0

            # Calculate duration in microseconds
            duration_seconds = end_time - start_time
            duration_us = duration_seconds * 1_000_000

            # Subtract measurement overhead for accuracy
            duration_us = max(0, duration_us - self.measurement_overhead_us)

            # Create measurement record
            measurement = TimingMeasurement(
                start_time=start_time,
                end_time=end_time,
                duration_us=duration_us,
                context=context
            )

            # Store measurement
            self.measurements.append(measurement)
            self.stats.update(duration_us)

            return duration_us

    def record_value(self, value: float, context: Optional[str] = None):
        """Record a direct performance value (not timing-based)"""
        with self.lock:
            self.stats.update(value)

            # Create measurement record for consistency
            current_time = high_res_timer()
            measurement = TimingMeasurement(
                start_time=current_time,
                end_time=current_time,
                duration_us=value,
                context=context
            )
            self.measurements.append(measurement)

    def get_stats(self) -> PerformanceStats:
        """Get current performance statistics"""
        with self.lock:
            return self.stats

    def get_recent_measurements(self, count: int = 100) -> List[TimingMeasurement]:
        """Get recent measurements"""
        with self.lock:
            return list(self.measurements)[-count:]

    def reset(self):
        """Reset all measurements and statistics"""
        with self.lock:
            self.measurements.clear()
            self.stats = PerformanceStats()
            self.active_timers.clear()

    def _calibrate_measurement_overhead(self) -> float:
        """Calibrate measurement overhead for accurate timing"""
        try:
            # Measure the overhead of the timing operations
            calibration_runs = 1000
            overhead_measurements = []

            for _ in range(calibration_runs):
                start = high_res_timer()
                end = high_res_timer()
                overhead_us = (end - start) * 1_000_000
                overhead_measurements.append(overhead_us)

            # Use median to avoid outliers
            overhead_measurements.sort()
            median_overhead = overhead_measurements[calibration_runs // 2]

            return median_overhead

        except Exception:
            # Fallback to conservative estimate
            return 0.1  # 0.1 microseconds


class SystemPerformanceCounters:
    """
    System-wide performance monitoring

    Features:
    - CPU usage tracking
    - Memory monitoring
    - I/O statistics
    - Network monitoring
    - Process-specific metrics
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # Performance counters
        self.cpu_counter = PerformanceCounter("system_cpu", CounterType.RESOURCE)
        self.memory_counter = PerformanceCounter("system_memory", CounterType.RESOURCE)
        self.disk_io_counter = PerformanceCounter("disk_io", CounterType.RESOURCE)

        # System monitoring
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Baseline measurements
        self.last_disk_io = None
        self.last_network_io = None
        self.last_measurement_time = None

    def start(self):
        """Start system monitoring"""
        self.is_monitoring = True

        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SystemPerformanceCounters"
        )
        self.monitoring_thread.start()

        self.logger.info("📊 System performance monitoring started")

    def stop(self):
        """Stop system monitoring"""
        self.is_monitoring = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("📊 System performance monitoring stopped")

    def get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()

            # Network I/O metrics
            network_io = psutil.net_io_counters()

            # Process count
            process_count = len(psutil.pids())

            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "total_percent": cpu_percent,
                    "per_core": cpu_per_core,
                    "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_percent": memory.percent,
                    "swap_percent": swap.percent
                },
                "disk_io": {
                    "read_count": disk_io.read_count if disk_io else 0,
                    "write_count": disk_io.write_count if disk_io else 0,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0
                },
                "network_io": {
                    "bytes_sent": network_io.bytes_sent if network_io else 0,
                    "bytes_recv": network_io.bytes_recv if network_io else 0,
                    "packets_sent": network_io.packets_sent if network_io else 0,
                    "packets_recv": network_io.packets_recv if network_io else 0
                },
                "processes": {
                    "total_count": process_count
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Error collecting system metrics: {e}")
            return {}

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self.get_current_system_metrics()

                if metrics:
                    # Update counters
                    if "cpu" in metrics:
                        self.cpu_counter.record_value(metrics["cpu"]["total_percent"])

                    if "memory" in metrics:
                        self.memory_counter.record_value(metrics["memory"]["used_percent"])

                    # Calculate I/O rates
                    self._update_io_metrics(metrics)

                time.sleep(1.0)  # Sample every second

            except Exception as e:
                self.logger.error(f"❌ Error in system monitoring loop: {e}")
                time.sleep(5.0)

    def _update_io_metrics(self, metrics: Dict[str, Any]):
        """Update I/O performance metrics"""
        current_time = time.time()

        if "disk_io" in metrics:
            disk_io = metrics["disk_io"]

            if self.last_disk_io and self.last_measurement_time:
                time_delta = current_time - self.last_measurement_time

                if time_delta > 0:
                    # Calculate I/O rates
                    read_rate = (disk_io["read_bytes"] - self.last_disk_io["read_bytes"]) / time_delta
                    write_rate = (disk_io["write_bytes"] - self.last_disk_io["write_bytes"]) / time_delta

                    total_io_rate = (read_rate + write_rate) / (1024 * 1024)  # MB/s
                    self.disk_io_counter.record_value(total_io_rate)

            self.last_disk_io = disk_io

        self.last_measurement_time = current_time


class WorkerPerformanceCounters:
    """
    Performance counters for backtest worker processes

    Features:
    - Worker-specific metrics
    - Inter-process communication
    - Resource usage tracking
    - Execution profiling
    """

    def __init__(self, worker_id: int, logger: logging.Logger = None):
        self.worker_id = worker_id
        self.logger = logger or logging.getLogger(f"WorkerPerformanceCounters-{worker_id}")

        # Worker-specific counters
        self.counters: Dict[str, PerformanceCounter] = {}

        # Standard counters for all workers
        self._initialize_standard_counters()

        # Worker state
        self.start_time = time.time()
        self.is_tracking = False

    def _initialize_standard_counters(self):
        """Initialize standard performance counters"""
        counter_definitions = [
            ("signal_processing", CounterType.TIMING),
            ("database_query", CounterType.TIMING),
            ("indicator_calculation", CounterType.TIMING),
            ("validation_check", CounterType.TIMING),
            ("memory_allocation", CounterType.TIMING),
            ("cache_access", CounterType.LATENCY),
            ("throughput_signals_per_second", CounterType.THROUGHPUT),
            ("memory_usage", CounterType.RESOURCE)
        ]

        for name, counter_type in counter_definitions:
            self.counters[name] = PerformanceCounter(
                f"worker_{self.worker_id}_{name}",
                counter_type
            )

    def start_execution_tracking(self):
        """Start tracking execution performance"""
        self.is_tracking = True
        self.start_time = time.time()
        self.logger.info(f"📊 Performance tracking started for worker {self.worker_id}")

    def stop_execution_tracking(self):
        """Stop tracking execution performance"""
        self.is_tracking = False
        self.logger.info(f"📊 Performance tracking stopped for worker {self.worker_id}")

    def get_counter(self, name: str) -> Optional[PerformanceCounter]:
        """Get a specific performance counter"""
        return self.counters.get(name)

    def start_timer(self, counter_name: str, context: Optional[str] = None) -> int:
        """Start timing for a specific operation"""
        counter = self.counters.get(counter_name)
        if counter:
            return counter.start_timer(context)
        return 0

    def stop_timer(self, counter_name: str, timer_id: int = None, context: Optional[str] = None) -> float:
        """Stop timing for a specific operation"""
        counter = self.counters.get(counter_name)
        if counter:
            return counter.stop_timer(timer_id, context)
        return 0.0

    def record_value(self, counter_name: str, value: float, context: Optional[str] = None):
        """Record a performance value"""
        counter = self.counters.get(counter_name)
        if counter:
            counter.record_value(value, context)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.is_tracking:
            return {"worker_id": self.worker_id, "status": "not_tracking"}

        stats = {
            "worker_id": self.worker_id,
            "uptime_seconds": time.time() - self.start_time,
            "counters": {}
        }

        for name, counter in self.counters.items():
            counter_stats = counter.get_stats()
            stats["counters"][name] = {
                "count": counter_stats.count,
                "average": counter_stats.average,
                "min": counter_stats.min_value,
                "max": counter_stats.max_value,
                "p50": counter_stats.p50,
                "p95": counter_stats.p95,
                "p99": counter_stats.p99
            }

        return stats

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get high-level performance summary"""
        stats = self.get_statistics()

        if stats.get("status") == "not_tracking":
            return stats

        # Calculate derived metrics
        signal_counter = self.counters.get("signal_processing")
        throughput_counter = self.counters.get("throughput_signals_per_second")

        summary = {
            "worker_id": self.worker_id,
            "uptime_seconds": stats["uptime_seconds"],
            "performance_overview": {
                "signals_processed": signal_counter.stats.count if signal_counter else 0,
                "avg_signal_processing_time_ms": signal_counter.stats.average / 1000 if signal_counter else 0,
                "current_throughput": throughput_counter.stats.recent_values[-1] if throughput_counter and throughput_counter.stats.recent_values else 0,
                "memory_efficiency": self._calculate_memory_efficiency()
            }
        }

        return summary

    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory usage efficiency"""
        memory_counter = self.counters.get("memory_usage")
        if not memory_counter or memory_counter.stats.count == 0:
            return 100.0

        # Simple efficiency metric: 100 - (current_usage / max_usage * 100)
        current_usage = memory_counter.stats.recent_values[-1] if memory_counter.stats.recent_values else 0
        max_usage = memory_counter.stats.max_value

        if max_usage == 0:
            return 100.0

        return max(0, 100 - (current_usage / max_usage * 100))


# Utility functions

def create_timing_context(counter: PerformanceCounter, context: str = None):
    """Create a context manager for automatic timing"""

    class TimingContext:
        def __init__(self, perf_counter: PerformanceCounter, ctx: str = None):
            self.counter = perf_counter
            self.context = ctx
            self.timer_id = None

        def __enter__(self):
            self.timer_id = self.counter.start_timer(self.context)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.counter.stop_timer(self.timer_id, self.context)

    return TimingContext(counter, context)


def benchmark_function(func: Callable, *args, iterations: int = 1000, **kwargs) -> Dict[str, Any]:
    """Benchmark a function's performance"""
    counter = PerformanceCounter("benchmark", CounterType.TIMING)

    for _ in range(iterations):
        timer_id = counter.start_timer()
        try:
            func(*args, **kwargs)
        finally:
            counter.stop_timer(timer_id)

    stats = counter.get_stats()

    return {
        "function": func.__name__,
        "iterations": iterations,
        "total_time_seconds": stats.total / 1_000_000,
        "average_time_us": stats.average,
        "min_time_us": stats.min_value,
        "max_time_us": stats.max_value,
        "p95_time_us": stats.p95,
        "p99_time_us": stats.p99
    }