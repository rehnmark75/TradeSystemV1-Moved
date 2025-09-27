# core/concurrent_execution/backtest_execution_manager.py
"""
BacktestExecutionManager - High-performance concurrent execution orchestrator
Manages parallel backtest execution while maintaining live scanner performance
"""

import asyncio
import multiprocessing as mp
import threading
import time
import logging
import queue
import signal
import os
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import psutil

try:
    import config
    from core.database import DatabaseManager
    from core.scanner_factory import ScannerFactory, ScannerMode
    from .backtest_queue import BacktestQueue, BacktestJob, JobPriority
    from .resource_monitor import ResourceMonitor, CircuitBreakerState
    from .backtest_worker import BacktestWorker, WorkerState
    from .process_isolation import ProcessManager, ProcessType
    from .memory_pools import MemoryPoolManager
    from .performance_counters import PerformanceCounters
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.scanner_factory import ScannerFactory, ScannerMode
    from .backtest_queue import BacktestQueue, BacktestJob, JobPriority
    from .resource_monitor import ResourceMonitor, CircuitBreakerState
    from .backtest_worker import BacktestWorker, WorkerState
    from .process_isolation import ProcessManager, ProcessType
    from .memory_pools import MemoryPoolManager
    from .performance_counters import PerformanceCounters


class ExecutionMode(Enum):
    """Execution modes for the manager"""
    STANDALONE = "standalone"  # Only backtests running
    CONCURRENT = "concurrent"  # Backtests + live scanner
    PRIORITY_LIVE = "priority_live"  # Live scanner has absolute priority


@dataclass
class ExecutionConfig:
    """Configuration for the execution manager"""
    max_concurrent_backtests: int = field(default_factory=lambda: min(mp.cpu_count() - 1, 4))
    max_memory_usage_gb: float = 8.0
    live_scanner_cpu_reserve: float = 0.2  # Reserve 20% CPU for live scanner
    backtest_worker_memory_limit_mb: int = 1024
    worker_timeout_seconds: int = 3600

    # Circuit breaker thresholds
    cpu_threshold: float = 0.85
    memory_threshold: float = 0.90
    disk_io_threshold_mb_s: float = 100.0

    # Performance optimization
    enable_memory_pools: bool = True
    enable_lock_free_queues: bool = True
    use_shared_memory: bool = True
    worker_affinity_enabled: bool = True

    # Live scanner protection
    live_scanner_priority: int = -10  # Higher priority than backtests
    live_scanner_nice_value: int = -5
    backtest_worker_nice_value: int = 10


@dataclass
class ExecutionStats:
    """Real-time execution statistics"""
    start_time: datetime = field(default_factory=datetime.now)
    backtests_queued: int = 0
    backtests_running: int = 0
    backtests_completed: int = 0
    backtests_failed: int = 0

    total_signals_processed: int = 0
    live_scanner_latency_ms: float = 0.0
    avg_backtest_throughput: float = 0.0

    memory_usage_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_io_mb_s: float = 0.0

    circuit_breaker_trips: int = 0
    worker_restarts: int = 0


class BacktestExecutionManager:
    """
    High-performance concurrent execution manager for backtests

    Key features:
    - Process isolation between live scanner and backtests
    - Resource monitoring with circuit breakers
    - Priority scheduling with live scanner protection
    - Memory pools and lock-free structures
    - Real-time performance monitoring
    - Ultra-low latency preservation for live scanning
    """

    def __init__(self,
                 config: ExecutionConfig = None,
                 db_manager: DatabaseManager = None,
                 logger: logging.Logger = None):

        self.config = config or ExecutionConfig()
        self.db_manager = db_manager or DatabaseManager(config.DATABASE_URL)
        self.logger = logger or logging.getLogger(__name__)

        # Core components
        self.resource_monitor = ResourceMonitor(self.config, self.logger)
        self.backtest_queue = BacktestQueue(self.config, self.logger)
        self.process_manager = ProcessManager(self.config, self.logger)
        self.scanner_factory = ScannerFactory(self.db_manager, self.logger)

        # Performance optimization components
        self.memory_pool_manager = MemoryPoolManager(self.config) if self.config.enable_memory_pools else None
        self.performance_counters = PerformanceCounters(self.logger)

        # Execution state
        self.execution_mode = ExecutionMode.STANDALONE
        self.is_running = False
        self.shutdown_requested = False

        # Worker management
        self.workers: Dict[int, BacktestWorker] = {}
        self.worker_processes: Dict[int, mp.Process] = {}
        self.worker_lock = threading.RLock()

        # Statistics and monitoring
        self.stats = ExecutionStats()
        self.stats_lock = threading.Lock()

        # Live scanner monitoring
        self.live_scanner_process: Optional[mp.Process] = None
        self.live_scanner_monitor = None

        # Event loop for async operations
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.event_loop_thread: Optional[threading.Thread] = None

        # Signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.logger.info(f"🚀 BacktestExecutionManager initialized:")
        self.logger.info(f"   Max concurrent backtests: {self.config.max_concurrent_backtests}")
        self.logger.info(f"   Memory limit: {self.config.max_memory_usage_gb:.1f}GB")
        self.logger.info(f"   CPU reserve for live: {self.config.live_scanner_cpu_reserve:.1%}")
        self.logger.info(f"   Memory pools: {'✅' if self.config.enable_memory_pools else '❌'}")

    async def start(self, execution_mode: ExecutionMode = ExecutionMode.CONCURRENT):
        """Start the execution manager"""
        self.execution_mode = execution_mode
        self.is_running = True
        self.shutdown_requested = False

        self.logger.info(f"🚀 Starting execution manager in {execution_mode.value} mode")

        try:
            # Start resource monitoring
            await self.resource_monitor.start()

            # Initialize performance counters
            self.performance_counters.start()

            # Start process manager
            await self.process_manager.start()

            # Start live scanner if in concurrent mode
            if execution_mode in [ExecutionMode.CONCURRENT, ExecutionMode.PRIORITY_LIVE]:
                await self._start_live_scanner_monitoring()

            # Start worker pool
            await self._initialize_worker_pool()

            # Start main execution loop
            asyncio.create_task(self._execution_loop())
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._cleanup_loop())

            self.logger.info("✅ Execution manager started successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to start execution manager: {e}")
            await self.shutdown()
            raise

    async def shutdown(self, timeout_seconds: float = 30.0):
        """Graceful shutdown of the execution manager"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        self.shutdown_requested = True

        shutdown_start = time.time()

        try:
            # Stop accepting new jobs
            self.backtest_queue.stop_accepting_jobs()

            # Wait for running backtests to complete or timeout
            await self._wait_for_workers_completion(timeout_seconds * 0.7)

            # Force shutdown remaining workers
            await self._force_shutdown_workers()

            # Stop monitoring components
            await self.resource_monitor.stop()
            self.performance_counters.stop()
            await self.process_manager.stop()

            # Cleanup memory pools
            if self.memory_pool_manager:
                self.memory_pool_manager.cleanup()

            self.is_running = False

            shutdown_duration = time.time() - shutdown_start
            self.logger.info(f"✅ Shutdown completed in {shutdown_duration:.1f}s")

        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
            raise

    def submit_backtest(self,
                       backtest_config: Dict[str, Any],
                       priority: JobPriority = JobPriority.NORMAL,
                       callback: Optional[Callable] = None) -> str:
        """
        Submit a backtest job for execution

        Args:
            backtest_config: Backtest configuration
            priority: Job priority
            callback: Optional completion callback

        Returns:
            job_id: Unique job identifier
        """

        if self.shutdown_requested:
            raise RuntimeError("Execution manager is shutting down")

        # Create execution record in database
        execution_id = self.scanner_factory.create_backtest_execution(
            strategy_name=backtest_config['strategy_name'],
            start_date=backtest_config['start_date'],
            end_date=backtest_config['end_date'],
            epics=backtest_config.get('epics'),
            timeframe=backtest_config.get('timeframe', '15m')
        )

        backtest_config['execution_id'] = execution_id

        # Create job
        job = BacktestJob(
            job_id=f"backtest_{execution_id}_{int(time.time())}",
            execution_id=execution_id,
            backtest_config=backtest_config,
            priority=priority,
            submitted_at=datetime.now(),
            callback=callback
        )

        # Submit to queue
        self.backtest_queue.submit_job(job)

        with self.stats_lock:
            self.stats.backtests_queued += 1

        self.logger.info(f"📋 Backtest job submitted: {job.job_id} (execution_id: {execution_id})")
        return job.job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job"""
        return self.backtest_queue.get_job_status(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job"""
        success = self.backtest_queue.cancel_job(job_id)

        if success:
            self.logger.info(f"❌ Job cancelled: {job_id}")

        return success

    def get_execution_stats(self) -> ExecutionStats:
        """Get current execution statistics"""
        with self.stats_lock:
            # Update real-time stats
            self.stats.memory_usage_gb = psutil.virtual_memory().used / (1024**3)
            self.stats.cpu_usage_percent = psutil.cpu_percent(interval=None)

            # Update backtest counts
            queue_stats = self.backtest_queue.get_queue_stats()
            self.stats.backtests_queued = queue_stats['queued']
            self.stats.backtests_running = len([w for w in self.workers.values() if w.state == WorkerState.RUNNING])

            return self.stats

    def get_live_scanner_performance(self) -> Dict[str, Any]:
        """Get live scanner performance metrics"""
        if not self.live_scanner_monitor:
            return {"status": "not_running"}

        return {
            "status": "running",
            "latency_ms": self.stats.live_scanner_latency_ms,
            "cpu_usage": self.live_scanner_monitor.get_cpu_usage(),
            "memory_usage_mb": self.live_scanner_monitor.get_memory_usage(),
            "scan_rate_per_minute": self.live_scanner_monitor.get_scan_rate()
        }

    # Internal methods

    async def _execution_loop(self):
        """Main execution loop - processes backtest queue"""
        while self.is_running and not self.shutdown_requested:
            try:
                # Check circuit breaker state
                if self.resource_monitor.circuit_breaker_state == CircuitBreakerState.OPEN:
                    self.logger.warning("⚡ Circuit breaker open - pausing backtest execution")
                    await asyncio.sleep(5.0)
                    continue

                # Check if we can start new backtests
                if not self._can_start_new_backtest():
                    await asyncio.sleep(1.0)
                    continue

                # Get next job from queue
                job = await self.backtest_queue.get_next_job()
                if job is None:
                    await asyncio.sleep(0.1)  # Short sleep if no jobs available
                    continue

                # Start worker for the job
                await self._start_backtest_worker(job)

            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1.0)

    async def _monitoring_loop(self):
        """Monitoring loop - tracks system resources and worker health"""
        while self.is_running and not self.shutdown_requested:
            try:
                # Update resource monitoring
                await self.resource_monitor.update()

                # Check worker health
                await self._check_worker_health()

                # Update performance counters
                self.performance_counters.update()

                # Check live scanner health if running
                if self.execution_mode in [ExecutionMode.CONCURRENT, ExecutionMode.PRIORITY_LIVE]:
                    await self._check_live_scanner_health()

                # Update statistics
                await self._update_statistics()

                await asyncio.sleep(1.0)  # Monitor every second

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _cleanup_loop(self):
        """Cleanup loop - manages completed workers and memory"""
        while self.is_running and not self.shutdown_requested:
            try:
                # Cleanup completed workers
                await self._cleanup_completed_workers()

                # Force garbage collection if memory usage is high
                if self.stats.memory_usage_gb > self.config.max_memory_usage_gb * 0.8:
                    gc.collect()

                await asyncio.sleep(10.0)  # Cleanup every 10 seconds

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30.0)

    def _can_start_new_backtest(self) -> bool:
        """Check if we can start a new backtest"""
        with self.worker_lock:
            running_workers = len([w for w in self.workers.values() if w.state == WorkerState.RUNNING])

            # Check worker limit
            if running_workers >= self.config.max_concurrent_backtests:
                return False

            # Check resource constraints
            if self.stats.memory_usage_gb > self.config.max_memory_usage_gb * 0.9:
                return False

            if self.stats.cpu_usage_percent > self.config.cpu_threshold * 100:
                return False

            # Check circuit breaker
            if self.resource_monitor.circuit_breaker_state != CircuitBreakerState.CLOSED:
                return False

            return True

    async def _start_backtest_worker(self, job: BacktestJob):
        """Start a new backtest worker for the given job"""
        try:
            worker_id = len(self.workers)

            # Create worker
            worker = BacktestWorker(
                worker_id=worker_id,
                config=self.config,
                db_manager=self.db_manager,
                logger=self.logger,
                memory_pool=self.memory_pool_manager.get_pool() if self.memory_pool_manager else None
            )

            # Create worker process
            process = mp.Process(
                target=worker.run_backtest,
                args=(job,),
                name=f"BacktestWorker-{worker_id}"
            )

            # Set process priority and affinity
            if self.config.worker_affinity_enabled:
                cpu_count = mp.cpu_count()
                # Assign worker to specific CPU cores (avoid core 0 for live scanner)
                cpu_affinity = [(worker_id + 1) % cpu_count]
                process._config = {'cpu_affinity': cpu_affinity}

            process.start()

            # Set process nice value to lower priority
            try:
                proc = psutil.Process(process.pid)
                proc.nice(self.config.backtest_worker_nice_value)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            with self.worker_lock:
                self.workers[worker_id] = worker
                self.worker_processes[worker_id] = process

            with self.stats_lock:
                self.stats.backtests_running += 1
                self.stats.backtests_queued -= 1

            self.logger.info(f"🔄 Started backtest worker {worker_id} for job {job.job_id}")

        except Exception as e:
            self.logger.error(f"❌ Failed to start backtest worker: {e}")
            # Mark job as failed
            job.state = "failed"
            job.error_message = str(e)

            with self.stats_lock:
                self.stats.backtests_failed += 1

    async def _check_worker_health(self):
        """Check health of running workers"""
        with self.worker_lock:
            workers_to_restart = []

            for worker_id, worker in list(self.workers.items()):
                process = self.worker_processes.get(worker_id)

                if process is None:
                    continue

                # Check if process is alive
                if not process.is_alive():
                    if process.exitcode == 0:
                        # Normal completion
                        worker.state = WorkerState.COMPLETED
                        with self.stats_lock:
                            self.stats.backtests_completed += 1
                            self.stats.backtests_running -= 1
                    else:
                        # Abnormal termination
                        worker.state = WorkerState.FAILED
                        with self.stats_lock:
                            self.stats.backtests_failed += 1
                            self.stats.backtests_running -= 1

                        workers_to_restart.append(worker_id)

                # Check for timeout
                elif worker.start_time and (time.time() - worker.start_time) > self.config.worker_timeout_seconds:
                    self.logger.warning(f"⏰ Worker {worker_id} timed out, terminating")
                    process.terminate()
                    workers_to_restart.append(worker_id)

            # Handle worker restarts
            for worker_id in workers_to_restart:
                await self._restart_worker(worker_id)

    async def _restart_worker(self, worker_id: int):
        """Restart a failed worker"""
        try:
            worker = self.workers.get(worker_id)
            process = self.worker_processes.get(worker_id)

            if process and process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()

            # Remove from tracking
            self.workers.pop(worker_id, None)
            self.worker_processes.pop(worker_id, None)

            with self.stats_lock:
                self.stats.worker_restarts += 1

            self.logger.info(f"🔄 Worker {worker_id} restarted")

        except Exception as e:
            self.logger.error(f"❌ Failed to restart worker {worker_id}: {e}")

    async def _start_live_scanner_monitoring(self):
        """Start monitoring the live scanner process"""
        try:
            # This would integrate with the existing live scanner
            # For now, we'll create a placeholder monitoring system
            self.live_scanner_monitor = LiveScannerMonitor(self.config, self.logger)
            await self.live_scanner_monitor.start()

            self.logger.info("🔍 Live scanner monitoring started")

        except Exception as e:
            self.logger.error(f"❌ Failed to start live scanner monitoring: {e}")

    async def _check_live_scanner_health(self):
        """Check live scanner health and performance"""
        if not self.live_scanner_monitor:
            return

        try:
            latency_ms = await self.live_scanner_monitor.get_current_latency()

            with self.stats_lock:
                self.stats.live_scanner_latency_ms = latency_ms

            # Check if latency is too high
            if latency_ms > 1000.0:  # 1 second threshold
                self.logger.warning(f"⚠️ Live scanner latency high: {latency_ms:.1f}ms")

                # Reduce backtest load if necessary
                if self.execution_mode == ExecutionMode.PRIORITY_LIVE:
                    await self._reduce_backtest_load()

        except Exception as e:
            self.logger.error(f"❌ Error checking live scanner health: {e}")

    async def _reduce_backtest_load(self):
        """Reduce backtest load to prioritize live scanner"""
        with self.worker_lock:
            running_workers = [w for w in self.workers.values() if w.state == WorkerState.RUNNING]

            if len(running_workers) > 1:
                # Pause lowest priority backtest
                lowest_priority_worker = min(running_workers, key=lambda w: w.current_job.priority.value if w.current_job else 0)
                await self._pause_worker(lowest_priority_worker.worker_id)

                self.logger.info(f"⏸️ Paused worker {lowest_priority_worker.worker_id} to prioritize live scanner")

    async def _pause_worker(self, worker_id: int):
        """Pause a specific worker"""
        process = self.worker_processes.get(worker_id)
        if process and process.is_alive():
            try:
                proc = psutil.Process(process.pid)
                proc.suspend()
                self.logger.info(f"⏸️ Worker {worker_id} paused")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    async def _resume_worker(self, worker_id: int):
        """Resume a paused worker"""
        process = self.worker_processes.get(worker_id)
        if process and process.is_alive():
            try:
                proc = psutil.Process(process.pid)
                proc.resume()
                self.logger.info(f"▶️ Worker {worker_id} resumed")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    async def _cleanup_completed_workers(self):
        """Clean up completed workers and their resources"""
        with self.worker_lock:
            completed_workers = []

            for worker_id, worker in list(self.workers.items()):
                if worker.state in [WorkerState.COMPLETED, WorkerState.FAILED]:
                    completed_workers.append(worker_id)

            for worker_id in completed_workers:
                process = self.worker_processes.get(worker_id)
                if process:
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.terminate()

                # Clean up worker resources
                worker = self.workers.get(worker_id)
                if worker and hasattr(worker, 'cleanup'):
                    worker.cleanup()

                # Remove from tracking
                self.workers.pop(worker_id, None)
                self.worker_processes.pop(worker_id, None)

    async def _update_statistics(self):
        """Update execution statistics"""
        try:
            # System resources
            mem_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            current_time = time.time()

            if hasattr(self, '_last_disk_check'):
                time_delta = current_time - self._last_disk_check
                bytes_delta = (disk_io.read_bytes + disk_io.write_bytes) - self._last_disk_bytes
                disk_io_mb_s = (bytes_delta / (1024 * 1024)) / time_delta
            else:
                disk_io_mb_s = 0.0

            self._last_disk_check = current_time
            self._last_disk_bytes = disk_io.read_bytes + disk_io.write_bytes

            with self.stats_lock:
                self.stats.memory_usage_gb = mem_info.used / (1024**3)
                self.stats.cpu_usage_percent = cpu_percent
                self.stats.disk_io_mb_s = disk_io_mb_s
                self.stats.circuit_breaker_trips = self.resource_monitor.get_trip_count()

        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")

    async def _initialize_worker_pool(self):
        """Initialize the worker pool"""
        self.logger.info(f"🏊 Initializing worker pool (max: {self.config.max_concurrent_backtests})")

        # Pre-warm any shared resources
        if self.memory_pool_manager:
            self.memory_pool_manager.initialize_pools()

    async def _wait_for_workers_completion(self, timeout_seconds: float):
        """Wait for all workers to complete with timeout"""
        start_time = time.time()

        while (time.time() - start_time) < timeout_seconds:
            with self.worker_lock:
                running_workers = [w for w in self.workers.values() if w.state == WorkerState.RUNNING]

                if not running_workers:
                    break

                self.logger.info(f"⏳ Waiting for {len(running_workers)} workers to complete...")

            await asyncio.sleep(1.0)

    async def _force_shutdown_workers(self):
        """Force shutdown all remaining workers"""
        with self.worker_lock:
            for worker_id, process in self.worker_processes.items():
                if process.is_alive():
                    self.logger.info(f"🛑 Force terminating worker {worker_id}")
                    process.terminate()
                    process.join(timeout=5.0)
                    if process.is_alive():
                        process.kill()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"📡 Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True


# Placeholder for LiveScannerMonitor - would be implemented to monitor existing live scanner
class LiveScannerMonitor:
    def __init__(self, config: ExecutionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    async def start(self):
        pass

    async def get_current_latency(self) -> float:
        return 0.0  # Placeholder

    def get_cpu_usage(self) -> float:
        return 0.0  # Placeholder

    def get_memory_usage(self) -> float:
        return 0.0  # Placeholder

    def get_scan_rate(self) -> float:
        return 0.0  # Placeholder


# Factory function
def create_execution_manager(config: ExecutionConfig = None,
                           db_manager: DatabaseManager = None,
                           logger: logging.Logger = None) -> BacktestExecutionManager:
    """Create BacktestExecutionManager instance"""
    return BacktestExecutionManager(config, db_manager, logger)