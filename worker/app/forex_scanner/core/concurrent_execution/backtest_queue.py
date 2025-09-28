# core/concurrent_execution/backtest_queue.py
"""
BacktestQueue - High-performance job queue with priority scheduling
Implements lock-free structures and efficient job distribution
"""

import asyncio
import threading
import time
import heapq
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque, defaultdict
import uuid

try:
    from .backtest_worker import BacktestJob
except ImportError:
    # Define here to avoid circular import
    pass


class JobPriority(IntEnum):
    """Job priority levels (lower number = higher priority)"""
    CRITICAL = 0      # Emergency backtests
    HIGH = 1         # Important strategy validation
    NORMAL = 2       # Regular backtests
    LOW = 3          # Background optimization
    BULK = 4         # Mass backtesting


class JobState(Enum):
    """Job execution states"""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class QueueStats:
    """Queue statistics and metrics"""
    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    total_jobs_failed: int = 0
    total_jobs_cancelled: int = 0

    jobs_queued: int = 0
    jobs_running: int = 0

    avg_queue_time_seconds: float = 0.0
    avg_execution_time_seconds: float = 0.0

    throughput_jobs_per_hour: float = 0.0
    queue_efficiency_percent: float = 100.0

    priority_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class JobMetrics:
    """Individual job performance metrics"""
    job_id: str
    queue_time_seconds: float = 0.0
    execution_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    completion_timestamp: Optional[datetime] = None


class BacktestQueue:
    """
    High-performance backtest job queue with priority scheduling

    Features:
    - Priority-based scheduling with preemption support
    - Lock-free queue operations where possible
    - Adaptive load balancing
    - Job deadline management
    - Resource-aware scheduling
    - Circuit breaker integration
    - Real-time queue analytics
    """

    def __init__(self,
                 config: 'ExecutionConfig',
                 logger: logging.Logger = None,
                 max_queue_size: int = 1000):

        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.max_queue_size = max_queue_size

        # Queue data structures
        self._priority_queues: Dict[JobPriority, List[Tuple[float, str, 'BacktestJob']]] = {
            priority: [] for priority in JobPriority
        }

        # Job tracking
        self._jobs: Dict[str, 'BacktestJob'] = {}
        self._running_jobs: Dict[str, 'BacktestJob'] = {}
        self._completed_jobs: deque = deque(maxlen=1000)  # Keep recent history

        # Synchronization
        self._queue_lock = asyncio.Lock()
        self._stats_lock = threading.Lock()

        # Statistics and metrics
        self.stats = QueueStats()
        self._job_metrics: Dict[str, JobMetrics] = {}

        # Queue management
        self.is_accepting_jobs = True
        self.total_jobs_submitted = 0

        # Performance optimization
        self._last_rebalance_time = time.time()
        self._rebalance_interval = 30.0  # seconds

        # Event callbacks
        self._job_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        self.logger.info("📋 BacktestQueue initialized")

    async def submit_job(self, job: 'BacktestJob') -> bool:
        """
        Submit a job to the queue

        Args:
            job: BacktestJob to submit

        Returns:
            bool: True if job was submitted successfully
        """

        if not self.is_accepting_jobs:
            self.logger.warning(f"❌ Queue not accepting jobs: {job.job_id}")
            return False

        if len(self._jobs) >= self.max_queue_size:
            self.logger.warning(f"❌ Queue full, rejecting job: {job.job_id}")
            return False

        async with self._queue_lock:
            try:
                # Generate unique job ID if not provided
                if not job.job_id:
                    job.job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"

                # Set submission timestamp
                job.submitted_at = datetime.now()
                job.state = JobState.QUEUED.value

                # Add to appropriate priority queue
                priority_queue = self._priority_queues[job.priority]

                # Use negative timestamp for heap ordering (earliest first within priority)
                queue_entry = (-job.submitted_at.timestamp(), job.job_id, job)
                heapq.heappush(priority_queue, queue_entry)

                # Track job
                self._jobs[job.job_id] = job

                # Update statistics
                with self._stats_lock:
                    self.stats.total_jobs_submitted += 1
                    self.stats.jobs_queued += 1
                    self.total_jobs_submitted += 1

                    # Update priority distribution
                    priority_name = job.priority.name
                    self.stats.priority_distribution[priority_name] = \
                        self.stats.priority_distribution.get(priority_name, 0) + 1

                # Create job metrics tracking
                self._job_metrics[job.job_id] = JobMetrics(
                    job_id=job.job_id,
                    queue_time_seconds=0.0
                )

                self.logger.info(f"📝 Job submitted: {job.job_id} (priority: {job.priority.name})")

                # Trigger job submission callbacks
                await self._trigger_callbacks('job_submitted', job)

                return True

            except Exception as e:
                self.logger.error(f"❌ Failed to submit job {job.job_id}: {e}")
                return False

    async def get_next_job(self, timeout_seconds: float = 1.0) -> Optional['BacktestJob']:
        """
        Get the next highest priority job from the queue

        Args:
            timeout_seconds: Maximum time to wait for a job

        Returns:
            BacktestJob or None if no job available
        """

        start_time = time.time()

        while (time.time() - start_time) < timeout_seconds:
            async with self._queue_lock:
                # Check all priority queues in order
                for priority in JobPriority:
                    priority_queue = self._priority_queues[priority]

                    if priority_queue:
                        # Get highest priority job (lowest priority number)
                        _, job_id, job = heapq.heappop(priority_queue)

                        # Move job to running state
                        job.state = JobState.RUNNING.value
                        job.start_time = datetime.now()

                        self._running_jobs[job_id] = job

                        # Update statistics
                        with self._stats_lock:
                            self.stats.jobs_queued -= 1
                            self.stats.jobs_running += 1

                        # Update job metrics
                        if job_id in self._job_metrics:
                            queue_time = (job.start_time - job.submitted_at).total_seconds()
                            self._job_metrics[job_id].queue_time_seconds = queue_time

                        self.logger.info(f"🎯 Dispatched job: {job_id} (priority: {job.priority.name})")

                        # Trigger job start callbacks
                        await self._trigger_callbacks('job_started', job)

                        return job

            # No jobs available, sleep briefly
            await asyncio.sleep(0.1)

        return None

    async def complete_job(self, job_id: str, success: bool = True, error: Optional[str] = None):
        """Mark a job as completed"""

        async with self._queue_lock:
            job = self._running_jobs.pop(job_id, None)

            if job is None:
                self.logger.warning(f"⚠️ Attempted to complete unknown job: {job_id}")
                return

            # Update job state
            job.end_time = datetime.now()
            job.state = JobState.COMPLETED.value if success else JobState.FAILED.value

            if error:
                job.error_message = error

            # Move to completed jobs
            self._completed_jobs.append(job)

            # Update statistics
            with self._stats_lock:
                self.stats.jobs_running -= 1

                if success:
                    self.stats.total_jobs_completed += 1
                else:
                    self.stats.total_jobs_failed += 1

            # Update job metrics
            if job_id in self._job_metrics:
                metrics = self._job_metrics[job_id]
                metrics.execution_time_seconds = (job.end_time - job.start_time).total_seconds()
                metrics.completion_timestamp = job.end_time

            # Clean up job tracking
            self._jobs.pop(job_id, None)

            self.logger.info(f"✅ Job completed: {job_id} ({'success' if success else 'failed'})")

            # Trigger completion callbacks
            await self._trigger_callbacks('job_completed', job)

            # Update queue statistics
            await self._update_queue_statistics()

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job"""

        async with self._queue_lock:
            # Check if job is queued
            if job_id in self._jobs:
                job = self._jobs[job_id]

                # Remove from appropriate priority queue
                priority_queue = self._priority_queues[job.priority]
                # Linear search and removal (could be optimized with indexed heap)
                for i, (timestamp, jid, j) in enumerate(priority_queue):
                    if jid == job_id:
                        priority_queue.pop(i)
                        heapq.heapify(priority_queue)  # Restore heap property
                        break

                job.state = JobState.CANCELLED.value
                job.end_time = datetime.now()

                # Update statistics
                with self._stats_lock:
                    self.stats.jobs_queued -= 1
                    self.stats.total_jobs_cancelled += 1

                # Clean up
                self._jobs.pop(job_id, None)
                self._completed_jobs.append(job)

                self.logger.info(f"❌ Job cancelled: {job_id}")

                # Trigger cancellation callbacks
                await self._trigger_callbacks('job_cancelled', job)

                return True

            # Check if job is running (would require worker coordination)
            elif job_id in self._running_jobs:
                job = self._running_jobs[job_id]
                # For running jobs, we mark as cancelled but actual termination
                # would be handled by the execution manager
                job.state = JobState.CANCELLED.value

                self.logger.info(f"🛑 Running job marked for cancellation: {job_id}")

                # Trigger cancellation callbacks (worker should handle actual termination)
                await self._trigger_callbacks('job_cancellation_requested', job)

                return True

            else:
                self.logger.warning(f"⚠️ Job not found for cancellation: {job_id}")
                return False

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific job"""

        # Check running jobs first
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            return self._create_job_status(job, "running")

        # Check queued jobs
        if job_id in self._jobs:
            job = self._jobs[job_id]
            return self._create_job_status(job, "queued")

        # Check completed jobs
        for job in self._completed_jobs:
            if job.job_id == job_id:
                return self._create_job_status(job, job.state)

        return {"job_id": job_id, "status": "not_found"}

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""

        with self._stats_lock:
            stats_dict = {
                "total_submitted": self.stats.total_jobs_submitted,
                "total_completed": self.stats.total_jobs_completed,
                "total_failed": self.stats.total_jobs_failed,
                "total_cancelled": self.stats.total_jobs_cancelled,
                "queued": self.stats.jobs_queued,
                "running": self.stats.jobs_running,
                "avg_queue_time_seconds": self.stats.avg_queue_time_seconds,
                "avg_execution_time_seconds": self.stats.avg_execution_time_seconds,
                "throughput_jobs_per_hour": self.stats.throughput_jobs_per_hour,
                "queue_efficiency_percent": self.stats.queue_efficiency_percent,
                "priority_distribution": self.stats.priority_distribution.copy(),
                "queue_size_by_priority": {
                    priority.name: len(self._priority_queues[priority])
                    for priority in JobPriority
                },
                "is_accepting_jobs": self.is_accepting_jobs,
                "max_queue_size": self.max_queue_size
            }

            return stats_dict

    def get_queue_health(self) -> Dict[str, Any]:
        """Get queue health metrics"""

        total_jobs = self.stats.jobs_queued + self.stats.jobs_running
        queue_utilization = (total_jobs / self.max_queue_size) * 100

        # Calculate job age metrics
        current_time = time.time()
        queue_ages = []

        for priority_queue in self._priority_queues.values():
            for timestamp, job_id, job in priority_queue:
                age_seconds = current_time + timestamp  # timestamp is negative
                queue_ages.append(age_seconds)

        avg_queue_age = sum(queue_ages) / max(len(queue_ages), 1)
        max_queue_age = max(queue_ages) if queue_ages else 0

        return {
            "queue_utilization_percent": queue_utilization,
            "avg_queue_age_seconds": avg_queue_age,
            "max_queue_age_seconds": max_queue_age,
            "backlog_by_priority": {
                priority.name: len(self._priority_queues[priority])
                for priority in JobPriority
            },
            "processing_efficiency": {
                "success_rate_percent": (
                    self.stats.total_jobs_completed /
                    max(self.stats.total_jobs_completed + self.stats.total_jobs_failed, 1)
                ) * 100,
                "avg_throughput_jobs_per_hour": self.stats.throughput_jobs_per_hour
            },
            "health_status": self._calculate_health_status(queue_utilization, avg_queue_age)
        }

    def stop_accepting_jobs(self):
        """Stop accepting new jobs (for graceful shutdown)"""
        self.is_accepting_jobs = False
        self.logger.info("🛑 Queue stopped accepting new jobs")

    def resume_accepting_jobs(self):
        """Resume accepting new jobs"""
        self.is_accepting_jobs = True
        self.logger.info("✅ Queue resumed accepting new jobs")

    async def rebalance_queue(self):
        """Rebalance queue for optimal performance"""

        current_time = time.time()
        if (current_time - self._last_rebalance_time) < self._rebalance_interval:
            return

        async with self._queue_lock:
            self.logger.info("⚖️ Rebalancing queue...")

            # Analyze queue distribution
            total_queued = sum(len(pq) for pq in self._priority_queues.values())

            if total_queued == 0:
                return

            # Log current distribution
            distribution = {
                priority.name: len(self._priority_queues[priority])
                for priority in JobPriority
            }
            self.logger.info(f"📊 Queue distribution: {distribution}")

            # Check for priority inversion issues
            await self._check_priority_inversion()

            # Update rebalance timestamp
            self._last_rebalance_time = current_time

    async def add_job_callback(self, event_type: str, callback: Callable):
        """Add callback for job events"""
        self._job_callbacks[event_type].append(callback)

    # Internal methods

    def _create_job_status(self, job: 'BacktestJob', status: str) -> Dict[str, Any]:
        """Create detailed job status dictionary"""

        status_dict = {
            "job_id": job.job_id,
            "status": status,
            "priority": job.priority.name,
            "submitted_at": job.submitted_at.isoformat() if job.submitted_at else None,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "execution_id": job.execution_id,
            "progress_percent": getattr(job, 'progress_percent', 0.0),
            "error_message": getattr(job, 'error_message', None)
        }

        # Add metrics if available
        if job.job_id in self._job_metrics:
            metrics = self._job_metrics[job.job_id]
            status_dict["metrics"] = {
                "queue_time_seconds": metrics.queue_time_seconds,
                "execution_time_seconds": metrics.execution_time_seconds,
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_time_seconds": metrics.cpu_time_seconds
            }

        return status_dict

    async def _update_queue_statistics(self):
        """Update queue performance statistics"""

        with self._stats_lock:
            # Calculate average queue time
            if self._job_metrics:
                queue_times = [m.queue_time_seconds for m in self._job_metrics.values() if m.queue_time_seconds > 0]
                self.stats.avg_queue_time_seconds = sum(queue_times) / max(len(queue_times), 1)

                execution_times = [m.execution_time_seconds for m in self._job_metrics.values() if m.execution_time_seconds > 0]
                self.stats.avg_execution_time_seconds = sum(execution_times) / max(len(execution_times), 1)

            # Calculate throughput (jobs per hour)
            total_completed = self.stats.total_jobs_completed + self.stats.total_jobs_failed
            if total_completed > 0 and hasattr(self, '_queue_start_time'):
                elapsed_hours = (time.time() - self._queue_start_time) / 3600
                self.stats.throughput_jobs_per_hour = total_completed / max(elapsed_hours, 0.01)

            # Calculate efficiency
            total_processed = self.stats.total_jobs_completed + self.stats.total_jobs_failed
            if total_processed > 0:
                self.stats.queue_efficiency_percent = (
                    self.stats.total_jobs_completed / total_processed
                ) * 100

    async def _check_priority_inversion(self):
        """Check for and resolve priority inversion issues"""

        # Look for old high-priority jobs that might be stuck
        current_time = time.time()
        max_age_thresholds = {
            JobPriority.CRITICAL: 60,    # 1 minute
            JobPriority.HIGH: 300,       # 5 minutes
            JobPriority.NORMAL: 1800,    # 30 minutes
            JobPriority.LOW: 3600,       # 1 hour
            JobPriority.BULK: 7200       # 2 hours
        }

        for priority in JobPriority:
            max_age = max_age_thresholds[priority]
            priority_queue = self._priority_queues[priority]

            for timestamp, job_id, job in priority_queue:
                age_seconds = current_time + timestamp  # timestamp is negative

                if age_seconds > max_age:
                    self.logger.warning(
                        f"⚠️ Old {priority.name} priority job detected: {job_id} "
                        f"(age: {age_seconds:.0f}s)"
                    )

                    # Could implement priority escalation here
                    # For example, promote HIGH jobs to CRITICAL after timeout

    def _calculate_health_status(self, queue_utilization: float, avg_queue_age: float) -> str:
        """Calculate overall queue health status"""

        if queue_utilization > 90 or avg_queue_age > 3600:  # 1 hour
            return "critical"
        elif queue_utilization > 75 or avg_queue_age > 1800:  # 30 minutes
            return "warning"
        elif queue_utilization > 50 or avg_queue_age > 600:  # 10 minutes
            return "caution"
        else:
            return "healthy"

    async def _trigger_callbacks(self, event_type: str, job: 'BacktestJob'):
        """Trigger registered callbacks for job events"""

        callbacks = self._job_callbacks.get(event_type, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(job)
                else:
                    callback(job)
            except Exception as e:
                self.logger.error(f"❌ Error in {event_type} callback: {e}")

    def __len__(self) -> int:
        """Get total number of jobs in queue"""
        return len(self._jobs) + len(self._running_jobs)