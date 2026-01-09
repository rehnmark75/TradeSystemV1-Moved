# core/concurrent_execution/resource_monitor.py
"""
ResourceMonitor - High-performance system resource monitoring with circuit breakers
Ensures live scanner performance is never compromised by backtest execution
"""

import asyncio
import os
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import psutil
import gc

try:
    from .performance_counters import SystemPerformanceCounters
except ImportError:
    from .performance_counters import SystemPerformanceCounters


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    HALF_OPEN = "half_open"  # Testing if system recovered
    OPEN = "open"          # System overloaded, blocking new backtests


class AlertLevel(Enum):
    """Resource alert levels"""
    GREEN = "green"      # Normal operation
    YELLOW = "yellow"    # Warning - close monitoring
    ORANGE = "orange"    # High usage - reduce load
    RED = "red"         # Critical - emergency actions needed


@dataclass
class ResourceThresholds:
    """Resource monitoring thresholds"""
    # CPU thresholds (percentage)
    cpu_warning: float = 70.0
    cpu_critical: float = 85.0
    cpu_emergency: float = 95.0

    # Memory thresholds (percentage)
    memory_warning: float = 75.0
    memory_critical: float = 90.0
    memory_emergency: float = 95.0

    # Disk I/O thresholds (MB/s)
    disk_io_warning: float = 50.0
    disk_io_critical: float = 100.0
    disk_io_emergency: float = 200.0

    # Network I/O thresholds (MB/s)
    network_io_warning: float = 10.0
    network_io_critical: float = 50.0
    network_io_emergency: float = 100.0

    # Live scanner specific
    live_scanner_latency_warning_ms: float = 500.0
    live_scanner_latency_critical_ms: float = 1000.0
    live_scanner_latency_emergency_ms: float = 2000.0


@dataclass
class ResourceMetrics:
    """Current system resource metrics"""
    timestamp: datetime = field(default_factory=datetime.now)

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_per_core: List[float] = field(default_factory=list)
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0

    # Memory metrics
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    swap_percent: float = 0.0

    # Disk I/O metrics
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    disk_iops: float = 0.0

    # Network I/O metrics
    network_sent_mb_s: float = 0.0
    network_recv_mb_s: float = 0.0

    # Process-specific metrics
    backtest_processes_count: int = 0
    backtest_processes_memory_gb: float = 0.0
    live_scanner_memory_mb: float = 0.0
    live_scanner_cpu_percent: float = 0.0

    def get_alert_level(self, thresholds: ResourceThresholds) -> AlertLevel:
        """Determine overall alert level based on current metrics"""

        # Check for emergency conditions
        if (self.cpu_percent >= thresholds.cpu_emergency or
            self.memory_percent >= thresholds.memory_emergency or
            self.disk_read_mb_s + self.disk_write_mb_s >= thresholds.disk_io_emergency):
            return AlertLevel.RED

        # Check for critical conditions
        if (self.cpu_percent >= thresholds.cpu_critical or
            self.memory_percent >= thresholds.memory_critical or
            self.disk_read_mb_s + self.disk_write_mb_s >= thresholds.disk_io_critical):
            return AlertLevel.ORANGE

        # Check for warning conditions
        if (self.cpu_percent >= thresholds.cpu_warning or
            self.memory_percent >= thresholds.memory_warning or
            self.disk_read_mb_s + self.disk_write_mb_s >= thresholds.disk_io_warning):
            return AlertLevel.YELLOW

        return AlertLevel.GREEN


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 3          # Failures before opening
    success_threshold: int = 2          # Successes before closing
    timeout_seconds: float = 30.0       # Time before trying half-open
    half_open_timeout_seconds: float = 10.0  # Time to test half-open


class ResourceMonitor:
    """
    High-performance system resource monitor with circuit breaker protection

    Features:
    - Sub-second resource monitoring
    - Circuit breaker patterns for system protection
    - Live scanner prioritization
    - Predictive resource management
    - Automatic load shedding
    - Performance counter integration
    """

    def __init__(self,
                 config: 'ExecutionConfig',
                 logger: logging.Logger = None,
                 thresholds: ResourceThresholds = None):

        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.thresholds = thresholds or ResourceThresholds()

        # Circuit breaker configuration
        self.circuit_config = CircuitBreakerConfig()
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_failure_count = 0
        self.circuit_success_count = 0
        self.circuit_last_failure_time = None
        self.circuit_state_changed_time = datetime.now()

        # Monitoring state
        self.is_running = False
        self.monitoring_thread = None
        self.shutdown_requested = False

        # Resource history for trend analysis
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 300  # 5 minutes at 1-second intervals

        # Performance counters
        self.performance_counters = SystemPerformanceCounters(self.logger)

        # Callbacks for resource events
        self.alert_callbacks: List[Callable[[AlertLevel, ResourceMetrics], None]] = []
        self.circuit_breaker_callbacks: List[Callable[[CircuitBreakerState], None]] = []

        # Live scanner monitoring
        self.live_scanner_process = None
        self.live_scanner_latency_history: List[float] = []

        # Disk I/O baseline tracking
        self.last_disk_io_time = None
        self.last_disk_io_counters = None

        # Network I/O baseline tracking
        self.last_network_io_time = None
        self.last_network_io_counters = None

        self.logger.info("🔍 ResourceMonitor initialized")

    async def start(self):
        """Start resource monitoring"""
        self.is_running = True
        self.shutdown_requested = False

        # Start performance counters
        self.performance_counters.start()

        # Start monitoring loop
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self.monitoring_thread.start()

        self.logger.info("🚀 Resource monitoring started")

    async def stop(self):
        """Stop resource monitoring"""
        self.shutdown_requested = True

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        # Stop performance counters
        self.performance_counters.stop()

        self.is_running = False
        self.logger.info("🛑 Resource monitoring stopped")

    async def update(self):
        """Force update of resource metrics (async interface)"""
        metrics = self._collect_system_metrics()
        self._process_metrics(metrics)
        return metrics

    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics"""
        return self.metrics_history[-1] if self.metrics_history else ResourceMetrics()

    def get_alert_level(self) -> AlertLevel:
        """Get current system alert level"""
        current_metrics = self.get_current_metrics()
        return current_metrics.get_alert_level(self.thresholds)

    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self.circuit_breaker_state

    def get_trip_count(self) -> int:
        """Get total number of circuit breaker trips"""
        return self.circuit_failure_count

    def register_alert_callback(self, callback: Callable[[AlertLevel, ResourceMetrics], None]):
        """Register callback for resource alerts"""
        self.alert_callbacks.append(callback)

    def register_circuit_breaker_callback(self, callback: Callable[[CircuitBreakerState], None]):
        """Register callback for circuit breaker state changes"""
        self.circuit_breaker_callbacks.append(callback)

    def set_live_scanner_process(self, process_id: int):
        """Set the live scanner process for monitoring"""
        try:
            self.live_scanner_process = psutil.Process(process_id)
            self.logger.info(f"🎯 Live scanner process set: PID {process_id}")
        except psutil.NoSuchProcess:
            self.logger.error(f"❌ Live scanner process not found: PID {process_id}")

    def record_live_scanner_latency(self, latency_ms: float):
        """Record live scanner latency measurement"""
        self.live_scanner_latency_history.append(latency_ms)

        # Keep only recent history
        if len(self.live_scanner_latency_history) > 100:
            self.live_scanner_latency_history.pop(0)

        # Check for latency degradation
        if latency_ms > self.thresholds.live_scanner_latency_critical_ms:
            self._trigger_emergency_load_shedding("live_scanner_latency")

    def can_start_new_backtest(self) -> Tuple[bool, str]:
        """Check if system can handle a new backtest"""

        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            return False, "Circuit breaker is open"

        current_metrics = self.get_current_metrics()
        alert_level = current_metrics.get_alert_level(self.thresholds)

        if alert_level == AlertLevel.RED:
            return False, f"System in emergency state: CPU {current_metrics.cpu_percent:.1f}%, Memory {current_metrics.memory_percent:.1f}%"

        if alert_level == AlertLevel.ORANGE:
            # Allow only high-priority backtests
            return False, f"System in critical state: CPU {current_metrics.cpu_percent:.1f}%, Memory {current_metrics.memory_percent:.1f}%"

        # Check specific resource limits
        if current_metrics.memory_percent > self.thresholds.memory_critical:
            return False, f"Memory usage too high: {current_metrics.memory_percent:.1f}%"

        if current_metrics.cpu_percent > self.thresholds.cpu_critical:
            return False, f"CPU usage too high: {current_metrics.cpu_percent:.1f}%"

        return True, "System resources available"

    def predict_resource_availability(self, minutes_ahead: int = 5) -> Dict[str, Any]:
        """Predict resource availability based on current trends"""

        if len(self.metrics_history) < 10:
            return {"prediction": "insufficient_data"}

        try:
            # Simple linear trend analysis
            recent_metrics = self.metrics_history[-10:]

            # CPU trend
            cpu_values = [m.cpu_percent for m in recent_metrics]
            cpu_trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)

            # Memory trend
            memory_values = [m.memory_percent for m in recent_metrics]
            memory_trend = (memory_values[-1] - memory_values[0]) / len(memory_values)

            # Predict future values
            current_cpu = cpu_values[-1]
            current_memory = memory_values[-1]

            predicted_cpu = current_cpu + (cpu_trend * minutes_ahead * 60)  # per second trend
            predicted_memory = current_memory + (memory_trend * minutes_ahead * 60)

            prediction = {
                "minutes_ahead": minutes_ahead,
                "current_cpu_percent": current_cpu,
                "predicted_cpu_percent": predicted_cpu,
                "current_memory_percent": current_memory,
                "predicted_memory_percent": predicted_memory,
                "cpu_trend_per_minute": cpu_trend * 60,
                "memory_trend_per_minute": memory_trend * 60,
                "recommended_action": self._get_recommended_action(predicted_cpu, predicted_memory)
            }

            return prediction

        except Exception as e:
            self.logger.error(f"❌ Error predicting resource availability: {e}")
            return {"prediction": "error", "error": str(e)}

    # Internal methods

    def _monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        while self.is_running and not self.shutdown_requested:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()

                # Process metrics and trigger alerts
                self._process_metrics(metrics)

                # Sleep for monitoring interval
                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(5.0)  # Longer sleep on error

    def _collect_system_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system metrics"""

        try:
            current_time = time.time()
            metrics = ResourceMetrics()

            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

            # Load average (Unix systems only)
            try:
                load_avg = os.getloadavg()
                metrics.load_average_1m = load_avg[0]
                metrics.load_average_5m = load_avg[1]
            except (AttributeError, OSError):
                pass

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_available_gb = memory.available / (1024**3)

            # Swap metrics
            swap = psutil.swap_memory()
            metrics.swap_percent = swap.percent

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io and self.last_disk_io_counters and self.last_disk_io_time:
                time_delta = current_time - self.last_disk_io_time
                if time_delta > 0:
                    bytes_read_delta = disk_io.read_bytes - self.last_disk_io_counters.read_bytes
                    bytes_write_delta = disk_io.write_bytes - self.last_disk_io_counters.write_bytes

                    metrics.disk_read_mb_s = (bytes_read_delta / (1024**2)) / time_delta
                    metrics.disk_write_mb_s = (bytes_write_delta / (1024**2)) / time_delta

                    iops_delta = (disk_io.read_count + disk_io.write_count) - (
                        self.last_disk_io_counters.read_count + self.last_disk_io_counters.write_count
                    )
                    metrics.disk_iops = iops_delta / time_delta

            self.last_disk_io_time = current_time
            self.last_disk_io_counters = disk_io

            # Network I/O metrics
            network_io = psutil.net_io_counters()
            if network_io and self.last_network_io_counters and self.last_network_io_time:
                time_delta = current_time - self.last_network_io_time
                if time_delta > 0:
                    bytes_sent_delta = network_io.bytes_sent - self.last_network_io_counters.bytes_sent
                    bytes_recv_delta = network_io.bytes_recv - self.last_network_io_counters.bytes_recv

                    metrics.network_sent_mb_s = (bytes_sent_delta / (1024**2)) / time_delta
                    metrics.network_recv_mb_s = (bytes_recv_delta / (1024**2)) / time_delta

            self.last_network_io_time = current_time
            self.last_network_io_counters = network_io

            # Process-specific metrics
            metrics.backtest_processes_count, metrics.backtest_processes_memory_gb = self._get_backtest_process_metrics()

            # Live scanner metrics
            if self.live_scanner_process:
                try:
                    metrics.live_scanner_memory_mb = self.live_scanner_process.memory_info().rss / (1024**2)
                    metrics.live_scanner_cpu_percent = self.live_scanner_process.cpu_percent()
                except psutil.NoSuchProcess:
                    self.live_scanner_process = None

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Error collecting system metrics: {e}")
            return ResourceMetrics()  # Return empty metrics

    def _get_backtest_process_metrics(self) -> Tuple[int, float]:
        """Get metrics for all backtest worker processes"""
        try:
            backtest_processes = []
            total_memory_bytes = 0

            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if 'BacktestWorker' in proc.info['name']:
                        backtest_processes.append(proc)
                        total_memory_bytes += proc.info['memory_info'].rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                    continue

            total_memory_gb = total_memory_bytes / (1024**3)
            return len(backtest_processes), total_memory_gb

        except Exception as e:
            self.logger.error(f"❌ Error getting backtest process metrics: {e}")
            return 0, 0.0

    def _process_metrics(self, metrics: ResourceMetrics):
        """Process collected metrics and trigger appropriate actions"""

        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)

        # Determine alert level
        alert_level = metrics.get_alert_level(self.thresholds)

        # Process circuit breaker logic
        self._process_circuit_breaker(metrics, alert_level)

        # Trigger alert callbacks
        if alert_level != AlertLevel.GREEN:
            for callback in self.alert_callbacks:
                try:
                    callback(alert_level, metrics)
                except Exception as e:
                    self.logger.error(f"❌ Error in alert callback: {e}")

        # Log significant changes
        if alert_level in [AlertLevel.ORANGE, AlertLevel.RED]:
            self.logger.warning(
                f"⚠️ Resource alert {alert_level.value}: "
                f"CPU {metrics.cpu_percent:.1f}%, "
                f"Memory {metrics.memory_percent:.1f}%, "
                f"Disk I/O {metrics.disk_read_mb_s + metrics.disk_write_mb_s:.1f} MB/s"
            )

    def _process_circuit_breaker(self, metrics: ResourceMetrics, alert_level: AlertLevel):
        """Process circuit breaker logic based on current metrics"""

        current_time = datetime.now()

        if self.circuit_breaker_state == CircuitBreakerState.CLOSED:
            # Check for failure conditions
            if alert_level == AlertLevel.RED:
                self.circuit_failure_count += 1
                self.circuit_last_failure_time = current_time

                if self.circuit_failure_count >= self.circuit_config.failure_threshold:
                    self._open_circuit_breaker("Resource emergency threshold exceeded")

        elif self.circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if enough time has passed to try half-open
            time_since_state_change = (current_time - self.circuit_state_changed_time).total_seconds()

            if time_since_state_change >= self.circuit_config.timeout_seconds:
                self._set_circuit_breaker_state(CircuitBreakerState.HALF_OPEN)

        elif self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            # Check if system has recovered
            if alert_level in [AlertLevel.GREEN, AlertLevel.YELLOW]:
                self.circuit_success_count += 1

                if self.circuit_success_count >= self.circuit_config.success_threshold:
                    self._close_circuit_breaker("System resources recovered")
            else:
                # Still in bad state, go back to open
                self._open_circuit_breaker("System still in emergency state")

            # Timeout check for half-open state
            time_in_half_open = (current_time - self.circuit_state_changed_time).total_seconds()
            if time_in_half_open >= self.circuit_config.half_open_timeout_seconds:
                self._open_circuit_breaker("Half-open timeout exceeded")

    def _open_circuit_breaker(self, reason: str):
        """Open the circuit breaker"""
        self.logger.warning(f"⚡ Opening circuit breaker: {reason}")
        self._set_circuit_breaker_state(CircuitBreakerState.OPEN)
        self.circuit_success_count = 0  # Reset success count

    def _close_circuit_breaker(self, reason: str):
        """Close the circuit breaker"""
        self.logger.info(f"✅ Closing circuit breaker: {reason}")
        self._set_circuit_breaker_state(CircuitBreakerState.CLOSED)
        self.circuit_failure_count = 0  # Reset failure count
        self.circuit_success_count = 0  # Reset success count

    def _set_circuit_breaker_state(self, new_state: CircuitBreakerState):
        """Set circuit breaker state and notify callbacks"""
        old_state = self.circuit_breaker_state
        self.circuit_breaker_state = new_state
        self.circuit_state_changed_time = datetime.now()

        if old_state != new_state:
            self.logger.info(f"🔄 Circuit breaker state changed: {old_state.value} → {new_state.value}")

            # Notify callbacks
            for callback in self.circuit_breaker_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    self.logger.error(f"❌ Error in circuit breaker callback: {e}")

    def _trigger_emergency_load_shedding(self, reason: str):
        """Trigger emergency load shedding to protect live scanner"""
        self.logger.warning(f"🚨 Emergency load shedding triggered: {reason}")

        # This would trigger immediate reduction of backtest load
        # Implementation would depend on integration with execution manager
        pass

    def _get_recommended_action(self, predicted_cpu: float, predicted_memory: float) -> str:
        """Get recommended action based on predicted resource usage"""

        if predicted_cpu > 90 or predicted_memory > 90:
            return "emergency_stop_all_backtests"
        elif predicted_cpu > 80 or predicted_memory > 80:
            return "reduce_backtest_load"
        elif predicted_cpu > 70 or predicted_memory > 70:
            return "monitor_closely"
        else:
            return "normal_operation"

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary"""
        current_metrics = self.get_current_metrics()
        alert_level = self.get_alert_level()

        return {
            "timestamp": current_metrics.timestamp.isoformat(),
            "alert_level": alert_level.value,
            "circuit_breaker_state": self.circuit_breaker_state.value,
            "system_resources": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "disk_io_mb_s": current_metrics.disk_read_mb_s + current_metrics.disk_write_mb_s,
                "load_average_1m": current_metrics.load_average_1m
            },
            "backtest_processes": {
                "count": current_metrics.backtest_processes_count,
                "memory_usage_gb": current_metrics.backtest_processes_memory_gb
            },
            "live_scanner": {
                "memory_usage_mb": current_metrics.live_scanner_memory_mb,
                "cpu_percent": current_metrics.live_scanner_cpu_percent,
                "avg_latency_ms": sum(self.live_scanner_latency_history[-10:]) / max(len(self.live_scanner_latency_history[-10:]), 1)
            },
            "circuit_breaker": {
                "trip_count": self.circuit_failure_count,
                "last_trip": self.circuit_last_failure_time.isoformat() if self.circuit_last_failure_time else None,
                "time_in_current_state_seconds": (datetime.now() - self.circuit_state_changed_time).total_seconds()
            }
        }