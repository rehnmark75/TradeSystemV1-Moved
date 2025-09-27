# core/concurrent_execution/process_isolation.py
"""
ProcessIsolation - Clean separation between live and backtest processes
Ensures ultra-low latency for live scanning while managing backtest execution
"""

import os
import sys
import time
import signal
import logging
import multiprocessing as mp
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil

try:
    from .resource_monitor import ResourceMonitor, AlertLevel
except ImportError:
    from .resource_monitor import ResourceMonitor, AlertLevel


class ProcessType(Enum):
    """Types of processes in the system"""
    LIVE_SCANNER = "live_scanner"
    BACKTEST_WORKER = "backtest_worker"
    EXECUTION_MANAGER = "execution_manager"
    RESOURCE_MONITOR = "resource_monitor"


class ProcessState(Enum):
    """Process execution states"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    CRASHED = "crashed"


@dataclass
class ProcessConfig:
    """Configuration for a managed process"""
    process_type: ProcessType
    process_id: Optional[int] = None
    command: Optional[List[str]] = None
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)

    # Resource limits
    max_memory_mb: Optional[int] = None
    cpu_affinity: Optional[List[int]] = None
    nice_value: int = 0
    priority_class: Optional[str] = None

    # Monitoring
    enable_monitoring: bool = True
    restart_on_failure: bool = False
    max_restart_attempts: int = 3
    restart_delay_seconds: float = 5.0

    # Isolation settings
    use_separate_namespace: bool = False
    limit_system_calls: bool = False
    isolate_network: bool = False


@dataclass
class ProcessMetrics:
    """Real-time process metrics"""
    process_id: int
    process_type: ProcessType
    state: ProcessState

    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0

    # Performance metrics
    start_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    restart_count: int = 0

    # System interaction
    open_files: int = 0
    threads: int = 0
    io_counters: Optional[Dict[str, int]] = None

    # Process-specific metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class ProcessManager:
    """
    High-performance process isolation and management system

    Features:
    - CPU affinity management for optimal performance
    - Memory isolation and limits
    - Process priority management
    - Real-time process monitoring
    - Automatic failure recovery
    - Resource-based process scheduling
    - Live scanner priority protection
    """

    def __init__(self,
                 config: 'ExecutionConfig',
                 logger: logging.Logger = None,
                 resource_monitor: Optional[ResourceMonitor] = None):

        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.resource_monitor = resource_monitor

        # Process tracking
        self.managed_processes: Dict[int, ProcessConfig] = {}
        self.process_metrics: Dict[int, ProcessMetrics] = {}
        self.process_handlers: Dict[int, subprocess.Popen] = {}

        # Live scanner special handling
        self.live_scanner_pid: Optional[int] = None
        self.live_scanner_priority_protection = True

        # System information
        self.cpu_count = os.cpu_count()
        self.available_cpus = list(range(self.cpu_count))
        self.reserved_cpus = set()  # CPUs reserved for live scanner

        # Monitoring and control
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_requested = False

        # Process isolation features
        self.namespace_support = self._check_namespace_support()
        self.cgroup_support = self._check_cgroup_support()

        # Statistics
        self.total_processes_started = 0
        self.total_processes_failed = 0
        self.total_restarts_performed = 0

        self.logger.info(f"🏗️ ProcessManager initialized:")
        self.logger.info(f"   CPU cores available: {self.cpu_count}")
        self.logger.info(f"   Namespace support: {'✅' if self.namespace_support else '❌'}")
        self.logger.info(f"   CGroup support: {'✅' if self.cgroup_support else '❌'}")

    async def start(self):
        """Start the process manager"""
        self.is_running = True
        self.shutdown_requested = False

        # Reserve CPU cores for live scanner
        self._setup_cpu_affinity_management()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ProcessManager-Monitor"
        )
        self.monitoring_thread.start()

        self.logger.info("🚀 ProcessManager started")

    async def stop(self):
        """Stop the process manager and all managed processes"""
        self.shutdown_requested = True

        # Gracefully stop all managed processes
        await self._stop_all_processes()

        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10.0)

        self.is_running = False
        self.logger.info("🛑 ProcessManager stopped")

    def register_live_scanner(self, process_id: int) -> bool:
        """
        Register the live scanner process for priority protection

        Args:
            process_id: PID of the live scanner process

        Returns:
            bool: True if successfully registered
        """

        try:
            # Verify process exists
            process = psutil.Process(process_id)
            process_name = process.name()

            self.live_scanner_pid = process_id

            # Configure live scanner for optimal performance
            config = ProcessConfig(
                process_type=ProcessType.LIVE_SCANNER,
                process_id=process_id,
                nice_value=self.config.live_scanner_nice_value,
                enable_monitoring=True,
                restart_on_failure=False  # Live scanner handles its own lifecycle
            )

            # Set high priority and CPU affinity
            self._optimize_live_scanner_process(process)

            # Register for monitoring
            self.managed_processes[process_id] = config
            self.process_metrics[process_id] = ProcessMetrics(
                process_id=process_id,
                process_type=ProcessType.LIVE_SCANNER,
                state=ProcessState.RUNNING,
                start_time=datetime.fromtimestamp(process.create_time())
            )

            self.logger.info(f"🎯 Live scanner registered: PID {process_id} ({process_name})")
            return True

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"❌ Failed to register live scanner PID {process_id}: {e}")
            return False

    def start_backtest_worker(self,
                            worker_id: int,
                            command: List[str],
                            working_directory: str = None,
                            environment: Dict[str, str] = None) -> Optional[int]:
        """
        Start a new backtest worker process with isolation

        Args:
            worker_id: Unique worker identifier
            command: Command to execute
            working_directory: Working directory for the process
            environment: Environment variables

        Returns:
            int: Process ID if successful, None otherwise
        """

        try:
            # Create process configuration
            config = ProcessConfig(
                process_type=ProcessType.BACKTEST_WORKER,
                command=command,
                working_directory=working_directory,
                environment_variables=environment or {},
                max_memory_mb=self.config.backtest_worker_memory_limit_mb,
                nice_value=self.config.backtest_worker_nice_value,
                enable_monitoring=True,
                restart_on_failure=False,  # Backtest workers are ephemeral
                cpu_affinity=self._allocate_cpu_affinity_for_worker(worker_id)
            )

            # Start the process with isolation
            process_handle = self._start_isolated_process(config)

            if process_handle is None:
                return None

            process_id = process_handle.pid

            # Register process for management
            config.process_id = process_id
            self.managed_processes[process_id] = config
            self.process_handlers[process_id] = process_handle

            # Initialize metrics tracking
            self.process_metrics[process_id] = ProcessMetrics(
                process_id=process_id,
                process_type=ProcessType.BACKTEST_WORKER,
                state=ProcessState.RUNNING,
                start_time=datetime.now(),
                custom_metrics={'worker_id': worker_id}
            )

            self.total_processes_started += 1

            self.logger.info(f"🔧 Started backtest worker: PID {process_id} (worker_id: {worker_id})")
            return process_id

        except Exception as e:
            self.logger.error(f"❌ Failed to start backtest worker {worker_id}: {e}")
            self.total_processes_failed += 1
            return None

    def stop_process(self, process_id: int, timeout_seconds: float = 30.0) -> bool:
        """
        Stop a managed process gracefully

        Args:
            process_id: PID of the process to stop
            timeout_seconds: Maximum time to wait for graceful shutdown

        Returns:
            bool: True if process stopped successfully
        """

        if process_id not in self.managed_processes:
            self.logger.warning(f"⚠️ Process {process_id} not managed by ProcessManager")
            return False

        try:
            config = self.managed_processes[process_id]
            process_handle = self.process_handlers.get(process_id)

            self.logger.info(f"🛑 Stopping process {process_id} ({config.process_type.value})")

            if process_handle:
                # Use process handle if available
                process_handle.terminate()

                try:
                    process_handle.wait(timeout=timeout_seconds)
                    stopped = True
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination failed
                    self.logger.warning(f"⚠️ Force killing process {process_id}")
                    process_handle.kill()
                    process_handle.wait()
                    stopped = True

            else:
                # Use psutil for external processes
                try:
                    process = psutil.Process(process_id)
                    process.terminate()

                    # Wait for graceful termination
                    process.wait(timeout=timeout_seconds)
                    stopped = True

                except psutil.TimeoutExpired:
                    # Force kill
                    process.kill()
                    process.wait()
                    stopped = True

                except psutil.NoSuchProcess:
                    # Process already terminated
                    stopped = True

            # Clean up tracking
            self._cleanup_process(process_id)

            if stopped:
                self.logger.info(f"✅ Process {process_id} stopped successfully")

            return stopped

        except Exception as e:
            self.logger.error(f"❌ Failed to stop process {process_id}: {e}")
            return False

    def pause_process(self, process_id: int) -> bool:
        """Pause a process execution"""

        if process_id not in self.managed_processes:
            return False

        try:
            process = psutil.Process(process_id)
            process.suspend()

            if process_id in self.process_metrics:
                self.process_metrics[process_id].state = ProcessState.PAUSED

            self.logger.info(f"⏸️ Process {process_id} paused")
            return True

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"❌ Failed to pause process {process_id}: {e}")
            return False

    def resume_process(self, process_id: int) -> bool:
        """Resume a paused process"""

        if process_id not in self.managed_processes:
            return False

        try:
            process = psutil.Process(process_id)
            process.resume()

            if process_id in self.process_metrics:
                self.process_metrics[process_id].state = ProcessState.RUNNING

            self.logger.info(f"▶️ Process {process_id} resumed")
            return True

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"❌ Failed to resume process {process_id}: {e}")
            return False

    def get_process_metrics(self, process_id: int) -> Optional[ProcessMetrics]:
        """Get real-time metrics for a specific process"""
        return self.process_metrics.get(process_id)

    def get_all_process_metrics(self) -> Dict[int, ProcessMetrics]:
        """Get metrics for all managed processes"""
        return self.process_metrics.copy()

    def get_live_scanner_performance(self) -> Dict[str, Any]:
        """Get live scanner performance metrics"""

        if not self.live_scanner_pid:
            return {"status": "not_registered"}

        metrics = self.process_metrics.get(self.live_scanner_pid)
        if not metrics:
            return {"status": "no_metrics"}

        return {
            "status": "running",
            "pid": self.live_scanner_pid,
            "cpu_percent": metrics.cpu_percent,
            "memory_mb": metrics.memory_mb,
            "uptime_seconds": metrics.uptime_seconds,
            "state": metrics.state.value,
            "priority_protection": self.live_scanner_priority_protection,
            "reserved_cpus": list(self.reserved_cpus)
        }

    def adjust_process_priority(self, process_id: int, priority_adjustment: int) -> bool:
        """Adjust process priority (nice value)"""

        try:
            process = psutil.Process(process_id)
            current_nice = process.nice()
            new_nice = max(-20, min(19, current_nice + priority_adjustment))

            process.nice(new_nice)

            self.logger.info(f"🎯 Adjusted process {process_id} priority: {current_nice} → {new_nice}")
            return True

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"❌ Failed to adjust priority for process {process_id}: {e}")
            return False

    def enable_live_scanner_protection(self):
        """Enable live scanner priority protection"""
        self.live_scanner_priority_protection = True
        self.logger.info("🛡️ Live scanner priority protection enabled")

    def disable_live_scanner_protection(self):
        """Disable live scanner priority protection"""
        self.live_scanner_priority_protection = False
        self.logger.info("⚠️ Live scanner priority protection disabled")

    # Internal methods

    def _setup_cpu_affinity_management(self):
        """Setup CPU affinity management for process isolation"""

        try:
            # Reserve CPU cores for live scanner (typically core 0 and 1)
            reserved_count = max(1, int(self.cpu_count * 0.2))  # Reserve 20% of CPUs
            self.reserved_cpus = set(range(reserved_count))

            self.logger.info(f"🎯 Reserved {reserved_count} CPU cores for live scanner: {list(self.reserved_cpus)}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to setup CPU affinity management: {e}")

    def _allocate_cpu_affinity_for_worker(self, worker_id: int) -> Optional[List[int]]:
        """Allocate CPU cores for a backtest worker"""

        if not self.config.worker_affinity_enabled:
            return None

        try:
            # Allocate from non-reserved CPUs
            available_cpus = [cpu for cpu in self.available_cpus if cpu not in self.reserved_cpus]

            if not available_cpus:
                return None

            # Round-robin allocation
            allocated_cpu = available_cpus[worker_id % len(available_cpus)]
            return [allocated_cpu]

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to allocate CPU affinity for worker {worker_id}: {e}")
            return None

    def _optimize_live_scanner_process(self, process: psutil.Process):
        """Optimize live scanner process for maximum performance"""

        try:
            # Set high priority (low nice value)
            process.nice(self.config.live_scanner_nice_value)

            # Set CPU affinity to reserved cores
            if self.reserved_cpus and hasattr(process, 'cpu_affinity'):
                process.cpu_affinity(list(self.reserved_cpus))

            self.logger.info(f"⚡ Live scanner optimized: nice={self.config.live_scanner_nice_value}, "
                           f"CPU affinity={list(self.reserved_cpus)}")

        except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
            self.logger.warning(f"⚠️ Failed to optimize live scanner process: {e}")

    def _start_isolated_process(self, config: ProcessConfig) -> Optional[subprocess.Popen]:
        """Start a process with isolation features"""

        try:
            # Build environment
            env = os.environ.copy()
            env.update(config.environment_variables)

            # Build command with potential namespace isolation
            command = config.command.copy()

            # Add isolation wrapper if supported
            if config.use_separate_namespace and self.namespace_support:
                command = ['unshare', '--pid', '--fork'] + command

            # Start process
            process = subprocess.Popen(
                command,
                cwd=config.working_directory,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=lambda: self._setup_process_limits(config)
            )

            # Apply post-creation optimizations
            self._apply_post_creation_settings(process, config)

            return process

        except Exception as e:
            self.logger.error(f"❌ Failed to start isolated process: {e}")
            return None

    def _setup_process_limits(self, config: ProcessConfig):
        """Setup process limits (called in preexec_fn)"""

        try:
            # Set nice value
            os.nice(config.nice_value)

            # Set memory limit using resource module
            if config.max_memory_mb:
                try:
                    import resource
                    memory_limit = config.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
                except ImportError:
                    pass

        except Exception as e:
            # Can't log here as we're in a subprocess
            pass

    def _apply_post_creation_settings(self, process: subprocess.Popen, config: ProcessConfig):
        """Apply settings after process creation"""

        try:
            psutil_process = psutil.Process(process.pid)

            # Set CPU affinity
            if config.cpu_affinity and hasattr(psutil_process, 'cpu_affinity'):
                psutil_process.cpu_affinity(config.cpu_affinity)

            # Additional optimizations based on process type
            if config.process_type == ProcessType.BACKTEST_WORKER:
                # Lower I/O priority for backtest workers
                try:
                    psutil_process.ionice(psutil.IOPRIO_CLASS_IDLE)
                except (AttributeError, psutil.NoSuchProcess):
                    pass

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.warning(f"⚠️ Failed to apply post-creation settings: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop for all managed processes"""

        while self.is_running and not self.shutdown_requested:
            try:
                self._update_process_metrics()
                self._check_process_health()

                # Check for resource-based process management
                if self.resource_monitor:
                    self._handle_resource_based_management()

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                self.logger.error(f"❌ Error in process monitoring loop: {e}")
                time.sleep(5.0)

    def _update_process_metrics(self):
        """Update metrics for all managed processes"""

        for process_id in list(self.managed_processes.keys()):
            try:
                process = psutil.Process(process_id)
                metrics = self.process_metrics.get(process_id)

                if not metrics:
                    continue

                # Update resource usage
                metrics.cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                metrics.memory_mb = memory_info.rss / (1024 * 1024)
                metrics.memory_percent = process.memory_percent()

                # Update runtime information
                if metrics.start_time:
                    metrics.uptime_seconds = (datetime.now() - metrics.start_time).total_seconds()

                # Update system interaction metrics
                metrics.open_files = process.num_fds() if hasattr(process, 'num_fds') else 0
                metrics.threads = process.num_threads()

                try:
                    io_counters = process.io_counters()
                    metrics.io_counters = {
                        'read_count': io_counters.read_count,
                        'write_count': io_counters.write_count,
                        'read_bytes': io_counters.read_bytes,
                        'write_bytes': io_counters.write_bytes
                    }
                except (psutil.AccessDenied, AttributeError):
                    pass

            except psutil.NoSuchProcess:
                # Process terminated, clean up
                self._cleanup_process(process_id)
            except Exception as e:
                self.logger.error(f"❌ Error updating metrics for process {process_id}: {e}")

    def _check_process_health(self):
        """Check health of all managed processes"""

        for process_id, config in list(self.managed_processes.items()):
            try:
                process = psutil.Process(process_id)

                # Check if process is responsive
                if not process.is_running():
                    self._handle_process_failure(process_id, "process_not_running")
                    continue

                # Check memory limits
                if config.max_memory_mb:
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    if memory_mb > config.max_memory_mb * 1.1:  # 10% tolerance
                        self.logger.warning(f"⚠️ Process {process_id} exceeding memory limit: "
                                          f"{memory_mb:.1f}MB > {config.max_memory_mb}MB")

            except psutil.NoSuchProcess:
                self._handle_process_failure(process_id, "process_terminated")
            except Exception as e:
                self.logger.error(f"❌ Error checking health of process {process_id}: {e}")

    def _handle_resource_based_management(self):
        """Handle resource-based process management"""

        if not self.resource_monitor:
            return

        try:
            alert_level = self.resource_monitor.get_alert_level()

            if alert_level in [AlertLevel.ORANGE, AlertLevel.RED]:
                # High resource usage - take protective action
                if self.live_scanner_priority_protection:
                    self._protect_live_scanner_resources()

            # Implement additional resource-based policies here

        except Exception as e:
            self.logger.error(f"❌ Error in resource-based management: {e}")

    def _protect_live_scanner_resources(self):
        """Protect live scanner from resource contention"""

        try:
            # Reduce priority of backtest workers
            backtest_workers = [
                pid for pid, config in self.managed_processes.items()
                if config.process_type == ProcessType.BACKTEST_WORKER
            ]

            for worker_pid in backtest_workers:
                self.adjust_process_priority(worker_pid, 5)  # Lower priority (higher nice)

            if backtest_workers:
                self.logger.info(f"🛡️ Protected live scanner: reduced priority of {len(backtest_workers)} workers")

        except Exception as e:
            self.logger.error(f"❌ Error protecting live scanner resources: {e}")

    def _handle_process_failure(self, process_id: int, reason: str):
        """Handle process failure"""

        config = self.managed_processes.get(process_id)
        if not config:
            return

        self.logger.warning(f"⚠️ Process {process_id} failed: {reason}")

        # Update metrics
        if process_id in self.process_metrics:
            self.process_metrics[process_id].state = ProcessState.FAILED

        # Handle restart if configured
        if config.restart_on_failure and config.max_restart_attempts > 0:
            self._attempt_process_restart(process_id)

        # Clean up if not restarting
        else:
            self._cleanup_process(process_id)

    def _attempt_process_restart(self, process_id: int):
        """Attempt to restart a failed process"""

        config = self.managed_processes.get(process_id)
        metrics = self.process_metrics.get(process_id)

        if not config or not metrics:
            return

        if metrics.restart_count >= config.max_restart_attempts:
            self.logger.error(f"❌ Process {process_id} exceeded max restart attempts")
            self._cleanup_process(process_id)
            return

        try:
            self.logger.info(f"🔄 Restarting process {process_id} (attempt {metrics.restart_count + 1})")

            # Wait before restart
            time.sleep(config.restart_delay_seconds)

            # Clean up old process
            self._cleanup_process(process_id, remove_tracking=False)

            # Restart process with same configuration
            if config.process_type == ProcessType.BACKTEST_WORKER:
                new_pid = self.start_backtest_worker(
                    worker_id=metrics.custom_metrics.get('worker_id', 0),
                    command=config.command,
                    working_directory=config.working_directory,
                    environment=config.environment_variables
                )

                if new_pid:
                    # Update restart count
                    new_metrics = self.process_metrics.get(new_pid)
                    if new_metrics:
                        new_metrics.restart_count = metrics.restart_count + 1

                    self.total_restarts_performed += 1
                    self.logger.info(f"✅ Process restarted: {process_id} → {new_pid}")
                else:
                    self.logger.error(f"❌ Failed to restart process {process_id}")

        except Exception as e:
            self.logger.error(f"❌ Error restarting process {process_id}: {e}")

    def _cleanup_process(self, process_id: int, remove_tracking: bool = True):
        """Clean up process resources"""

        try:
            # Close process handle if exists
            process_handle = self.process_handlers.pop(process_id, None)
            if process_handle:
                try:
                    process_handle.terminate()
                    process_handle.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass

            # Remove from tracking if requested
            if remove_tracking:
                self.managed_processes.pop(process_id, None)
                self.process_metrics.pop(process_id, None)

                # Update live scanner tracking
                if process_id == self.live_scanner_pid:
                    self.live_scanner_pid = None

        except Exception as e:
            self.logger.error(f"❌ Error cleaning up process {process_id}: {e}")

    async def _stop_all_processes(self):
        """Stop all managed processes"""

        process_ids = list(self.managed_processes.keys())

        self.logger.info(f"🛑 Stopping {len(process_ids)} managed processes")

        # Stop backtest workers first
        backtest_workers = [
            pid for pid, config in self.managed_processes.items()
            if config.process_type == ProcessType.BACKTEST_WORKER
        ]

        for pid in backtest_workers:
            self.stop_process(pid, timeout_seconds=10.0)

        # Don't stop live scanner - it manages its own lifecycle

    def _check_namespace_support(self) -> bool:
        """Check if namespace isolation is supported"""
        try:
            result = subprocess.run(['unshare', '--help'],
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _check_cgroup_support(self) -> bool:
        """Check if cgroup support is available"""
        try:
            return os.path.exists('/sys/fs/cgroup')
        except OSError:
            return False

    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get process manager statistics"""

        active_processes_by_type = defaultdict(int)
        total_memory_by_type = defaultdict(float)

        for pid, config in self.managed_processes.items():
            process_type = config.process_type.value
            active_processes_by_type[process_type] += 1

            metrics = self.process_metrics.get(pid)
            if metrics:
                total_memory_by_type[process_type] += metrics.memory_mb

        return {
            "total_processes_started": self.total_processes_started,
            "total_processes_failed": self.total_processes_failed,
            "total_restarts_performed": self.total_restarts_performed,
            "active_processes": dict(active_processes_by_type),
            "memory_usage_by_type_mb": dict(total_memory_by_type),
            "live_scanner_registered": self.live_scanner_pid is not None,
            "live_scanner_protection_enabled": self.live_scanner_priority_protection,
            "cpu_management": {
                "total_cpus": self.cpu_count,
                "reserved_cpus": list(self.reserved_cpus),
                "worker_affinity_enabled": self.config.worker_affinity_enabled
            },
            "isolation_features": {
                "namespace_support": self.namespace_support,
                "cgroup_support": self.cgroup_support
            }
        }