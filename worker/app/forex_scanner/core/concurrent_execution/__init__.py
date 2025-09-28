# core/concurrent_execution/__init__.py
"""
High-Performance Concurrent Execution System for TradeSystemV1

This package provides a complete concurrent execution system for running backtests
in parallel with live scanning while maintaining ultra-low latency performance.

Key Components:
- BacktestExecutionManager: Main orchestrator for concurrent execution
- BacktestWorker: Individual worker processes with performance optimizations
- ResourceMonitor: System resource monitoring with circuit breaker protection
- BacktestQueue: Priority-based job scheduling with lock-free operations
- ProcessManager: Process isolation and CPU affinity management
- MemoryPoolManager: Zero-allocation memory management
- PerformanceCounters: Microsecond-level performance monitoring

Features:
- Process isolation between live and backtest execution
- Sub-second live scanning latency preservation
- Memory pools for zero-allocation hot paths
- Lock-free data structures for minimal contention
- Circuit breaker patterns for system protection
- Real-time resource monitoring and load balancing
- Automatic failure recovery and process management
"""

from .backtest_execution_manager import (
    BacktestExecutionManager,
    ExecutionMode,
    ExecutionConfig,
    ExecutionStats,
    create_execution_manager
)

from .backtest_worker import (
    BacktestWorker,
    WorkerState,
    WorkerStats,
    BacktestJob,
    create_backtest_worker,
    run_worker_process
)

from .resource_monitor import (
    ResourceMonitor,
    CircuitBreakerState,
    AlertLevel,
    ResourceThresholds,
    ResourceMetrics
)

from .backtest_queue import (
    BacktestQueue,
    JobPriority,
    JobState,
    QueueStats
)

from .process_isolation import (
    ProcessManager,
    ProcessType,
    ProcessState,
    ProcessConfig,
    ProcessMetrics
)

from .memory_pools import (
    MemoryPool,
    MemoryPoolManager,
    CachedDataBuffer,
    DataFramePool,
    NumpyArrayPool,
    PoolType,
    create_market_data_pools,
    create_backtest_pools
)

from .performance_counters import (
    PerformanceCounter,
    SystemPerformanceCounters,
    WorkerPerformanceCounters,
    CounterType,
    TimingMeasurement,
    PerformanceStats,
    create_timing_context,
    benchmark_function
)

# Version information
__version__ = "1.0.0"
__author__ = "TradeSystemV1 Development Team"

# Package-level constants
DEFAULT_MAX_CONCURRENT_BACKTESTS = 4
DEFAULT_MEMORY_LIMIT_GB = 8.0
DEFAULT_CPU_THRESHOLD = 0.85
DEFAULT_WORKER_TIMEOUT_SECONDS = 3600

# Quick start factory functions

def create_concurrent_execution_system(
    max_concurrent_backtests: int = DEFAULT_MAX_CONCURRENT_BACKTESTS,
    max_memory_usage_gb: float = DEFAULT_MEMORY_LIMIT_GB,
    enable_memory_pools: bool = True,
    enable_process_isolation: bool = True,
    db_manager=None,
    logger=None
) -> BacktestExecutionManager:
    """
    Create a complete concurrent execution system with optimal defaults

    Args:
        max_concurrent_backtests: Maximum number of parallel backtests
        max_memory_usage_gb: Maximum system memory usage
        enable_memory_pools: Enable zero-allocation memory pools
        enable_process_isolation: Enable CPU affinity and process isolation
        db_manager: Database manager instance
        logger: Logger instance

    Returns:
        BacktestExecutionManager: Configured execution manager
    """

    config = ExecutionConfig(
        max_concurrent_backtests=max_concurrent_backtests,
        max_memory_usage_gb=max_memory_usage_gb,
        enable_memory_pools=enable_memory_pools,
        worker_affinity_enabled=enable_process_isolation
    )

    return create_execution_manager(config, db_manager, logger)


def create_optimized_backtest_config(
    strategy_name: str,
    start_date,
    end_date,
    epics=None,
    timeframe: str = '15m',
    priority: JobPriority = JobPriority.NORMAL
) -> dict:
    """
    Create an optimized backtest configuration

    Args:
        strategy_name: Name of the trading strategy
        start_date: Backtest start date
        end_date: Backtest end date
        epics: List of currency pairs to test
        timeframe: Trading timeframe
        priority: Job execution priority

    Returns:
        dict: Optimized backtest configuration
    """

    return {
        'strategy_name': strategy_name,
        'start_date': start_date,
        'end_date': end_date,
        'epics': epics or [],
        'timeframe': timeframe,
        'priority': priority,
        # Optimization flags
        'enable_caching': True,
        'enable_vectorization': True,
        'batch_processing': True,
        'memory_optimization': True,
        'parallel_indicator_calculation': True
    }


# Performance monitoring utilities

class PerformanceProfiler:
    """Simple performance profiling context manager"""

    def __init__(self, operation_name: str, logger=None):
        self.operation_name = operation_name
        self.logger = logger
        self.counter = PerformanceCounter(operation_name, CounterType.TIMING)
        self.timer_id = None

    def __enter__(self):
        self.timer_id = self.counter.start_timer(self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_us = self.counter.stop_timer(self.timer_id)

        if self.logger:
            self.logger.info(f"⏱️ {self.operation_name}: {duration_us/1000:.2f}ms")

    def get_stats(self):
        return self.counter.get_stats()


# System health check utilities

def check_system_readiness(
    min_available_memory_gb: float = 2.0,
    max_cpu_usage_percent: float = 80.0,
    logger=None
) -> tuple[bool, str]:
    """
    Check if system is ready for concurrent backtest execution

    Args:
        min_available_memory_gb: Minimum required available memory
        max_cpu_usage_percent: Maximum acceptable CPU usage
        logger: Optional logger for diagnostics

    Returns:
        tuple: (is_ready, status_message)
    """

    try:
        import psutil

        # Check memory availability
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        if available_gb < min_available_memory_gb:
            return False, f"Insufficient memory: {available_gb:.1f}GB < {min_available_memory_gb}GB required"

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1.0)

        if cpu_percent > max_cpu_usage_percent:
            return False, f"High CPU usage: {cpu_percent:.1f}% > {max_cpu_usage_percent}% threshold"

        # Check disk space (basic check)
        disk = psutil.disk_usage('/')
        available_gb_disk = disk.free / (1024**3)

        if available_gb_disk < 1.0:  # Less than 1GB free
            return False, f"Low disk space: {available_gb_disk:.1f}GB available"

        return True, f"System ready: Memory {available_gb:.1f}GB, CPU {cpu_percent:.1f}%, Disk {available_gb_disk:.1f}GB"

    except Exception as e:
        return False, f"System check failed: {str(e)}"


# Export key classes and functions
__all__ = [
    # Main execution components
    'BacktestExecutionManager',
    'BacktestWorker',
    'ResourceMonitor',
    'BacktestQueue',
    'ProcessManager',

    # Memory management
    'MemoryPoolManager',
    'MemoryPool',
    'CachedDataBuffer',

    # Performance monitoring
    'PerformanceCounter',
    'SystemPerformanceCounters',
    'WorkerPerformanceCounters',

    # Configuration and enums
    'ExecutionConfig',
    'ExecutionMode',
    'JobPriority',
    'ProcessType',
    'AlertLevel',
    'CounterType',

    # Factory functions
    'create_concurrent_execution_system',
    'create_optimized_backtest_config',
    'create_execution_manager',
    'create_backtest_worker',
    'create_market_data_pools',
    'create_backtest_pools',

    # Utilities
    'PerformanceProfiler',
    'check_system_readiness',
    'create_timing_context',
    'benchmark_function',

    # Constants
    'DEFAULT_MAX_CONCURRENT_BACKTESTS',
    'DEFAULT_MEMORY_LIMIT_GB',
    'DEFAULT_CPU_THRESHOLD'
]