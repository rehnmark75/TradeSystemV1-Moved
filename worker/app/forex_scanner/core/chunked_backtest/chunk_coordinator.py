"""
ChunkCoordinator - Splits backtest date ranges into parallel chunks

Manages the creation and execution of chunked parallel backtests,
coordinating with the existing BacktestExecutionManager.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

try:
    from core.database import DatabaseManager
    from core.concurrent_execution.backtest_execution_manager import (
        BacktestExecutionManager,
        ExecutionConfig,
        ExecutionMode
    )
    from core.concurrent_execution.backtest_queue import JobPriority
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.concurrent_execution.backtest_execution_manager import (
        BacktestExecutionManager,
        ExecutionConfig,
        ExecutionMode
    )
    from forex_scanner.core.concurrent_execution.backtest_queue import JobPriority


@dataclass
class ChunkConfig:
    """Configuration for a single chunk"""
    chunk_index: int
    data_start: datetime      # When to start loading data (includes warmup)
    signal_start: datetime    # When to start recording signals
    signal_end: datetime      # When to stop recording signals

    def __post_init__(self):
        """Validate chunk configuration"""
        if self.signal_start >= self.signal_end:
            raise ValueError(f"signal_start ({self.signal_start}) must be before signal_end ({self.signal_end})")
        if self.data_start > self.signal_start:
            raise ValueError(f"data_start ({self.data_start}) must be before or equal to signal_start ({self.signal_start})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_index': self.chunk_index,
            'data_start': self.data_start.isoformat(),
            'signal_start': self.signal_start.isoformat(),
            'signal_end': self.signal_end.isoformat(),
        }


@dataclass
class ParallelRunConfig:
    """Configuration for a parallel backtest run"""
    epic: str
    strategy: str
    start_date: datetime
    end_date: datetime
    chunk_days: int = 7
    warmup_days: int = 2
    worker_count: int = 4
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epic': self.epic,
            'strategy': self.strategy,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'chunk_days': self.chunk_days,
            'warmup_days': self.warmup_days,
            'worker_count': self.worker_count,
            'config_overrides': self.config_overrides,
        }


@dataclass
class ChunkResult:
    """Result from a single chunk execution"""
    chunk_index: int
    execution_id: int
    job_id: str
    status: str  # 'completed', 'failed'
    signals_count: int = 0
    total_pips: float = 0.0
    error_message: Optional[str] = None


class ChunkCoordinator:
    """
    Coordinates parallel chunk execution for backtests

    Splits a date range into chunks with warmup overlap,
    submits them to BacktestExecutionManager, and tracks progress.
    """

    def __init__(self,
                 db_manager: DatabaseManager = None,
                 execution_manager: BacktestExecutionManager = None,
                 logger: logging.Logger = None):

        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.execution_manager = execution_manager

        # Track active runs
        self._active_runs: Dict[int, Dict[str, Any]] = {}

        self.logger.info("ChunkCoordinator initialized")

    def create_chunks(self,
                      start_date: datetime,
                      end_date: datetime,
                      chunk_days: int = 7,
                      warmup_days: int = 2) -> List[ChunkConfig]:
        """
        Split date range into chunks with warmup overlap

        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
            chunk_days: Days per chunk (default 7)
            warmup_days: Extra days for indicator warmup (default 2)

        Returns:
            List of ChunkConfig objects
        """
        chunks = []
        current = start_date
        chunk_index = 0

        while current < end_date:
            # Calculate chunk end (don't exceed total end date)
            chunk_end = min(current + timedelta(days=chunk_days), end_date)

            # Calculate warmup start (go back warmup_days for indicator calculation)
            warmup_start = current - timedelta(days=warmup_days)

            chunk = ChunkConfig(
                chunk_index=chunk_index,
                data_start=warmup_start,
                signal_start=current,
                signal_end=chunk_end
            )
            chunks.append(chunk)

            self.logger.debug(
                f"Created chunk {chunk_index}: "
                f"data={warmup_start.date()} -> signals={current.date()} to {chunk_end.date()}"
            )

            # Move to next chunk
            current = chunk_end
            chunk_index += 1

        self.logger.info(
            f"Created {len(chunks)} chunks for {(end_date - start_date).days} days "
            f"(chunk_days={chunk_days}, warmup_days={warmup_days})"
        )

        return chunks

    def calculate_optimal_chunks(self,
                                  total_days: int,
                                  worker_count: int) -> Tuple[int, int]:
        """
        Calculate optimal chunk size based on workers and duration

        Args:
            total_days: Total backtest duration in days
            worker_count: Number of available workers

        Returns:
            Tuple of (chunk_days, total_chunks)
        """
        # Aim for roughly equal chunks per worker
        # Minimum chunk size of 3 days, maximum of 14 days
        min_chunk_days = 3
        max_chunk_days = 14

        # Calculate ideal chunk size
        ideal_chunk_days = max(min_chunk_days, total_days // worker_count)
        chunk_days = min(max_chunk_days, ideal_chunk_days)

        # Calculate resulting chunk count
        total_chunks = (total_days + chunk_days - 1) // chunk_days  # Ceiling division

        self.logger.info(
            f"Optimal chunking: {total_days} days / {worker_count} workers = "
            f"{chunk_days} days/chunk ({total_chunks} total chunks)"
        )

        return chunk_days, total_chunks

    async def create_parallel_run(self, config: ParallelRunConfig) -> int:
        """
        Create a new parallel run record in the database

        Args:
            config: ParallelRunConfig with run parameters

        Returns:
            parallel_run_id
        """
        chunks = self.create_chunks(
            config.start_date,
            config.end_date,
            config.chunk_days,
            config.warmup_days
        )

        query = """
        INSERT INTO backtest_parallel_runs (
            epic, strategy, full_start_date, full_end_date,
            chunk_days, warmup_days, worker_count,
            total_chunks, status, config_snapshot
        ) VALUES (
            :epic, :strategy, :start_date, :end_date,
            :chunk_days, :warmup_days, :worker_count,
            :total_chunks, 'pending', :config_snapshot
        ) RETURNING id
        """

        params = {
            'epic': config.epic,
            'strategy': config.strategy,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'chunk_days': config.chunk_days,
            'warmup_days': config.warmup_days,
            'worker_count': config.worker_count,
            'total_chunks': len(chunks),
            'config_snapshot': json.dumps(config.to_dict())
        }

        try:
            result = self.db_manager.execute_query(query, params)
            parallel_run_id = result.iloc[0]['id']

            self.logger.info(f"Created parallel run {parallel_run_id} with {len(chunks)} chunks")

            return parallel_run_id

        except Exception as e:
            self.logger.error(f"Failed to create parallel run: {e}")
            raise

    async def run_parallel_backtest(self,
                                     config: ParallelRunConfig,
                                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Execute a parallel backtest with chunking

        Args:
            config: ParallelRunConfig with run parameters
            progress_callback: Optional callback(completed, total, chunk_result) for progress updates

        Returns:
            Dict with parallel_run_id and chunk_results
        """
        # Create the run record
        parallel_run_id = await self.create_parallel_run(config)

        # Create chunks
        chunks = self.create_chunks(
            config.start_date,
            config.end_date,
            config.chunk_days,
            config.warmup_days
        )

        # Update status to running
        await self._update_run_status(parallel_run_id, 'running')

        # Initialize execution manager if not provided
        if self.execution_manager is None:
            exec_config = ExecutionConfig(
                max_concurrent_backtests=config.worker_count
            )
            self.execution_manager = BacktestExecutionManager(
                config=exec_config,
                db_manager=self.db_manager,
                logger=self.logger
            )
            await self.execution_manager.start(ExecutionMode.STANDALONE)

        try:
            # Submit all chunks as jobs
            job_ids = []
            for chunk in chunks:
                backtest_config = {
                    'strategy_name': config.strategy,
                    'epics': [config.epic],
                    'start_date': chunk.data_start,
                    'end_date': chunk.signal_end,
                    'signal_start_filter': chunk.signal_start,  # Only record signals after warmup
                    'timeframe': '15m',
                    'parallel_run_id': parallel_run_id,
                    'chunk_index': chunk.chunk_index,
                    **config.config_overrides
                }

                job_id = self.execution_manager.submit_backtest(
                    backtest_config,
                    priority=JobPriority.NORMAL
                )
                job_ids.append((chunk.chunk_index, job_id))

                self.logger.info(
                    f"Submitted chunk {chunk.chunk_index} as job {job_id} "
                    f"({chunk.signal_start.date()} to {chunk.signal_end.date()})"
                )

            # Wait for all jobs to complete with progress tracking
            chunk_results = await self._wait_for_chunks(
                parallel_run_id,
                job_ids,
                progress_callback
            )

            # Update status based on results
            failed_chunks = [r for r in chunk_results if r.status == 'failed']
            if failed_chunks:
                await self._update_run_status(
                    parallel_run_id,
                    'failed',
                    f"{len(failed_chunks)} chunks failed"
                )
            else:
                await self._update_run_status(parallel_run_id, 'aggregating')

            return {
                'parallel_run_id': parallel_run_id,
                'chunk_results': chunk_results,
                'total_chunks': len(chunks),
                'completed_chunks': len([r for r in chunk_results if r.status == 'completed']),
                'failed_chunks': len(failed_chunks)
            }

        except Exception as e:
            self.logger.error(f"Parallel backtest failed: {e}")
            await self._update_run_status(parallel_run_id, 'failed', str(e))
            raise

    async def run_sequential_backtest(self,
                                       config: ParallelRunConfig,
                                       progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Execute a backtest with chunking but sequentially (simpler, more reliable).

        This is a fallback for when the parallel execution manager isn't available.

        Args:
            config: ParallelRunConfig with run parameters
            progress_callback: Optional callback(completed, total, chunk_result) for progress updates

        Returns:
            Dict with parallel_run_id and chunk_results
        """
        # Import the backtest orchestrator for running individual chunks
        try:
            from forex_scanner.core.trading.backtest_trading_orchestrator import BacktestTradingOrchestrator
            from forex_scanner.core.scanner import IntelligentForexScanner
            from forex_scanner.core.data_fetcher import DataFetcher
        except ImportError:
            from core.trading.backtest_trading_orchestrator import BacktestTradingOrchestrator
            from core.scanner import IntelligentForexScanner
            from core.data_fetcher import DataFetcher

        # Create the run record
        parallel_run_id = await self.create_parallel_run(config)

        # Create chunks
        chunks = self.create_chunks(
            config.start_date,
            config.end_date,
            config.chunk_days,
            config.warmup_days
        )

        # Update status to running
        await self._update_run_status(parallel_run_id, 'running')

        chunk_results = []
        chunk_execution_ids = []

        try:
            for i, chunk in enumerate(chunks):
                self.logger.info(
                    f"Processing chunk {i + 1}/{len(chunks)}: "
                    f"{chunk.signal_start.date()} to {chunk.signal_end.date()}"
                )

                try:
                    # Create orchestrator for this chunk
                    orchestrator = BacktestTradingOrchestrator(
                        strategy_name=config.strategy,
                        backtest_config={
                            'start_date': chunk.data_start,
                            'end_date': chunk.signal_end,
                            'signal_start_filter': chunk.signal_start,
                            'epics': [config.epic],
                            'timeframe': '15m',
                            **config.config_overrides
                        }
                    )

                    # Run the backtest for this chunk
                    result = orchestrator.run_backtest(
                        epic=config.epic,
                        start_date=chunk.data_start,
                        end_date=chunk.signal_end,
                        timeframe='15m'
                    )

                    # Get execution ID and stats
                    execution_id = result.get('execution_id', 0)
                    signals_count = result.get('signals_count', 0) or result.get('total_signals', 0)
                    total_pips = result.get('total_pips', 0.0)

                    chunk_result = ChunkResult(
                        chunk_index=i,
                        execution_id=execution_id,
                        job_id=f"chunk_{i}",
                        status='completed',
                        signals_count=signals_count,
                        total_pips=total_pips,
                        error_message=None
                    )

                    if execution_id:
                        chunk_execution_ids.append(execution_id)
                        await self._record_chunk_completion(parallel_run_id, execution_id)

                except Exception as chunk_error:
                    self.logger.error(f"Chunk {i} failed: {chunk_error}")
                    chunk_result = ChunkResult(
                        chunk_index=i,
                        execution_id=0,
                        job_id=f"chunk_{i}",
                        status='failed',
                        signals_count=0,
                        total_pips=0.0,
                        error_message=str(chunk_error)
                    )

                chunk_results.append(chunk_result)

                # Call progress callback
                if progress_callback:
                    progress_callback(i + 1, len(chunks), chunk_result)

            # Update final status
            failed_chunks = [r for r in chunk_results if r.status == 'failed']
            if failed_chunks:
                await self._update_run_status(
                    parallel_run_id,
                    'failed',
                    f"{len(failed_chunks)} chunks failed"
                )
            else:
                await self._update_run_status(parallel_run_id, 'aggregating')

            return {
                'parallel_run_id': int(parallel_run_id),
                'chunk_results': chunk_results,
                'total_chunks': len(chunks),
                'completed_chunks': len([r for r in chunk_results if r.status == 'completed']),
                'failed_chunks': len(failed_chunks)
            }

        except Exception as e:
            self.logger.error(f"Sequential backtest failed: {e}")
            await self._update_run_status(parallel_run_id, 'failed', str(e))
            raise

    async def _wait_for_chunks(self,
                                parallel_run_id: int,
                                job_ids: List[Tuple[int, str]],
                                progress_callback: Optional[callable] = None) -> List[ChunkResult]:
        """
        Wait for all chunk jobs to complete

        Args:
            parallel_run_id: ID of the parallel run
            job_ids: List of (chunk_index, job_id) tuples
            progress_callback: Optional callback for progress updates

        Returns:
            List of ChunkResult objects
        """
        results = []
        completed_count = 0
        total_count = len(job_ids)

        # Poll for completion
        pending_jobs = dict(job_ids)  # chunk_index -> job_id

        while pending_jobs:
            for chunk_index, job_id in list(pending_jobs.items()):
                status = self.execution_manager.get_job_status(job_id)

                if status['status'] in ('completed', 'failed'):
                    # Job finished
                    chunk_result = ChunkResult(
                        chunk_index=chunk_index,
                        execution_id=status.get('execution_id', 0),
                        job_id=job_id,
                        status=status['status'],
                        signals_count=status.get('signals_count', 0),
                        total_pips=status.get('total_pips', 0.0),
                        error_message=status.get('error_message')
                    )
                    results.append(chunk_result)

                    # Update database
                    if chunk_result.execution_id:
                        await self._record_chunk_completion(
                            parallel_run_id,
                            chunk_result.execution_id
                        )

                    # Remove from pending
                    del pending_jobs[chunk_index]
                    completed_count += 1

                    # Call progress callback
                    if progress_callback:
                        progress_callback(completed_count, total_count, chunk_result)

                    self.logger.info(
                        f"Chunk {chunk_index} {status['status']}: "
                        f"{chunk_result.signals_count} signals, {chunk_result.total_pips:+.1f} pips"
                    )

            if pending_jobs:
                await asyncio.sleep(1.0)  # Poll every second

        # Sort results by chunk index
        results.sort(key=lambda r: r.chunk_index)

        return results

    async def _update_run_status(self,
                                  parallel_run_id: int,
                                  status: str,
                                  error_message: Optional[str] = None):
        """Update parallel run status in database"""
        try:
            if status == 'running':
                query = """
                UPDATE backtest_parallel_runs
                SET status = :status, started_at = NOW()
                WHERE id = :run_id
                """
            elif status in ('completed', 'failed'):
                query = """
                UPDATE backtest_parallel_runs
                SET status = :status, completed_at = NOW(), error_message = :error
                WHERE id = :run_id
                """
            else:
                query = """
                UPDATE backtest_parallel_runs
                SET status = :status
                WHERE id = :run_id
                """

            params = {
                'run_id': parallel_run_id,
                'status': status,
                'error': error_message
            }

            self.db_manager.execute_query(query, params)

        except Exception as e:
            # Log but don't fail - this is status tracking
            self.logger.warning(f"Failed to update run status: {e}")

    async def _record_chunk_completion(self,
                                        parallel_run_id: int,
                                        execution_id: int):
        """Record a chunk completion in the database"""
        try:
            query = """
            SELECT update_parallel_run_progress(:run_id, :exec_id)
            """
            params = {
                'run_id': parallel_run_id,
                'exec_id': execution_id
            }
            self.db_manager.execute_query(query, params)

        except Exception as e:
            self.logger.warning(f"Failed to record chunk completion: {e}")

    def get_run_status(self, parallel_run_id: int) -> Dict[str, Any]:
        """
        Get status of a parallel run

        Args:
            parallel_run_id: ID of the parallel run

        Returns:
            Dict with run status and progress
        """
        query = """
        SELECT
            id, epic, strategy,
            full_start_date, full_end_date,
            chunk_days, warmup_days, worker_count,
            status, total_chunks, completed_chunks,
            chunk_execution_ids, aggregated_results,
            error_message,
            created_at, started_at, completed_at
        FROM backtest_parallel_runs
        WHERE id = :run_id
        """

        result = self.db_manager.execute_query(query, {'run_id': parallel_run_id})

        if result.empty:
            return {'status': 'not_found'}

        row = result.iloc[0]
        return {
            'id': row['id'],
            'epic': row['epic'],
            'strategy': row['strategy'],
            'status': row['status'],
            'progress': f"{row['completed_chunks']}/{row['total_chunks']}",
            'completed_chunks': row['completed_chunks'],
            'total_chunks': row['total_chunks'],
            'aggregated_results': row['aggregated_results'],
            'error_message': row['error_message'],
            'created_at': row['created_at'],
            'started_at': row['started_at'],
            'completed_at': row['completed_at'],
            'duration_seconds': (
                (row['completed_at'] - row['started_at']).total_seconds()
                if row['completed_at'] and row['started_at']
                else None
            )
        }


# Factory function
def create_chunk_coordinator(db_manager: DatabaseManager = None,
                             execution_manager: BacktestExecutionManager = None,
                             logger: logging.Logger = None) -> ChunkCoordinator:
    """Create ChunkCoordinator instance"""
    return ChunkCoordinator(db_manager, execution_manager, logger)
