#!/usr/bin/env python3
"""
multi_epic_optimizer.py - Multi-Epic Parameter Optimization

Runs comprehensive parameter optimization across all enabled epics with:
- Tiered execution modes (fast/medium/extended)
- Smart parameter filtering (R:R >= 1.5)
- Resume capability on interruption
- Progress bar with ETA
- Database + CSV result storage
- Auto-snapshot for best configs per epic
- Cross-epic comparison reports
- Optional parallel execution (multiple epics simultaneously)

Usage:
    # Fast mode - core parameters only (~6-7 hours sequential)
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode fast --days 30

    # Medium mode - extended parameters (~22-25 hours sequential)
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode medium --days 30

    # Extended mode - full sweep (~45-50 hours sequential)
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode extended --days 30

    # Specific epics only
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --epics EURUSD GBPUSD

    # Resume interrupted run
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --resume 12345

    # Dry run (show combinations without executing)
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode extended --dry-run

    # Enable parallel execution (auto-detect worker count)
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode fast --days 30 --parallel

    # Parallel with specific worker count
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode fast --days 30 --parallel 4

    # Parallel with memory limit
    docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode fast --days 30 --parallel --max-memory-gb 4
"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

# Suppress logging before imports
import logging
logging.disable(logging.CRITICAL)

import os
os.environ['LOG_LEVEL'] = 'ERROR'

import json
import time
import signal
import argparse
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from itertools import product
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# For resource checking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from forex_scanner.commands.enhanced_backtest_commands import EnhancedBacktestCommands
from forex_scanner.core.database import DatabaseManager
from forex_scanner.services.backtest_config_service import get_backtest_config_service
from forex_scanner import config

from forex_scanner.optimization_config import (
    PARAMETER_TIERS,
    RR_FILTER,
    EPIC_SHORTCUTS,
    EPIC_TO_SHORTCUT,
    SCORE_WEIGHTS,
    SCORE_NORMALIZATION,
    DEFAULT_MODE,
    DEFAULT_DAYS,
    DEFAULT_OUTPUT_DIR,
    get_tier_info,
    resolve_epic,
)


# ============================================
# MODULE-LEVEL WORKER FUNCTION (for multiprocessing)
# ============================================

def _worker_optimize_epic(
    epic: str,
    combinations: List[Dict],
    start_date: datetime,
    end_date: datetime,
    run_id: int,
    database_url: str,
    progress_dict: dict = None
) -> Tuple[str, List[dict]]:
    """
    Module-level worker function to optimize a single epic.
    Must be at module level for ProcessPoolExecutor pickling.

    Args:
        epic: Epic to optimize
        combinations: Parameter combinations to test
        start_date: Backtest start date
        end_date: Backtest end date
        run_id: Optimization run ID
        database_url: Database connection URL
        progress_dict: Shared dict for progress updates (multiprocessing.Manager)

    Returns:
        Tuple of (epic, list of result dicts)
    """
    # Worker-local imports and setup (each process needs its own)
    from forex_scanner.commands.enhanced_backtest_commands import EnhancedBacktestCommands
    from forex_scanner.core.database import DatabaseManager

    # Suppress logging in worker
    import logging
    logging.disable(logging.CRITICAL)

    # Worker-local instances - create fresh connections in this process
    local_backtest_cmd = EnhancedBacktestCommands()
    local_db = DatabaseManager(database_url)

    pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
    results = []

    try:
        for idx, params in enumerate(combinations):
            # Update progress
            if progress_dict is not None:
                progress_dict[epic] = f"🔄 {idx + 1}/{len(combinations)}"

            # Run backtest
            try:
                start_time = time.time()

                override = {
                    'fixed_stop_loss_pips': params.get('fixed_stop_loss_pips', 9),
                    'fixed_take_profit_pips': params.get('fixed_take_profit_pips', 15),
                    'min_confidence': params.get('min_confidence', 0.5),
                }

                success = local_backtest_cmd.run_enhanced_backtest(
                    epic=epic,
                    start_date=start_date,
                    end_date=end_date,
                    strategy='SMC_SIMPLE',
                    config_override=override,
                    use_historical_intelligence=False,
                    pipeline=False,
                    show_signals=False
                )

                duration = time.time() - start_time

                # Extract result from DB
                result_data = _worker_extract_result(local_db, epic, params, duration)
                results.append(result_data)

                # Store result in DB
                _worker_store_result(local_db, run_id, result_data)

            except Exception as e:
                results.append({
                    'epic': epic,
                    'params': params,
                    'status': 'error',
                    'error_message': str(e),
                    'duration_seconds': 0
                })

        # Mark epic as complete
        if progress_dict is not None:
            progress_dict[epic] = "✅ DONE"

    except Exception as e:
        if progress_dict is not None:
            progress_dict[epic] = f"❌ ERROR: {e}"

    finally:
        # Clean up worker DB connection
        try:
            local_db.close() if hasattr(local_db, 'close') else None
        except:
            pass

    return epic, results


def _worker_extract_result(db, epic: str, params: dict, duration: float) -> dict:
    """Extract latest backtest result (worker-safe)"""
    query = """
        SELECT
            id,
            total_signals,
            winning_trades,
            losing_trades,
            win_rate,
            profit_factor,
            total_profit_pips,
            avg_profit_pips,
            avg_loss_pips
        FROM backtest_executions
        WHERE epic = :epic
        ORDER BY created_at DESC
        LIMIT 1
    """

    try:
        result = db.execute_query(query, {'epic': epic})
        if result is not None and len(result) > 0:
            row = result.iloc[0]

            # Calculate composite score
            win_rate = float(row.get('win_rate', 0) or 0)
            profit_factor = float(row.get('profit_factor', 0) or 0)
            total_pips = float(row.get('total_profit_pips', 0) or 0)
            total_signals = int(row.get('total_signals', 0) or 0)

            # Composite score formula
            score = (
                (win_rate * 100) * 0.3 +
                min(profit_factor, 3.0) * 20 +
                (total_pips / 10) * 0.2 +
                min(total_signals / 10, 5) * 0.2
            )

            return {
                'epic': epic,
                'params': params,
                'execution_id': int(row.get('id', 0)),
                'total_signals': total_signals,
                'winners': int(row.get('winning_trades', 0) or 0),
                'losers': int(row.get('losing_trades', 0) or 0),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_pips': total_pips,
                'avg_profit_pips': float(row.get('avg_profit_pips', 0) or 0),
                'avg_loss_pips': float(row.get('avg_loss_pips', 0) or 0),
                'composite_score': score,
                'status': 'completed',
                'error_message': '',
                'duration_seconds': duration
            }

    except Exception as e:
        pass

    return {
        'epic': epic,
        'params': params,
        'status': 'error',
        'error_message': 'Failed to extract result',
        'duration_seconds': duration
    }


def _worker_store_result(db, run_id: int, result: dict) -> None:
    """Store result in DB (worker-safe)"""
    try:
        query = """
            INSERT INTO optimization_results (
                run_id, epic, params, execution_id,
                total_signals, winners, losers, win_rate,
                profit_factor, total_pips, avg_profit_pips, avg_loss_pips,
                composite_score, status, error_message, duration_seconds,
                created_at
            ) VALUES (
                :run_id, :epic, :params, :execution_id,
                :total_signals, :winners, :losers, :win_rate,
                :profit_factor, :total_pips, :avg_profit_pips, :avg_loss_pips,
                :composite_score, :status, :error_message, :duration_seconds,
                NOW()
            )
        """
        db.execute_query(query, {
            'run_id': run_id,
            'epic': result.get('epic', ''),
            'params': json.dumps(result.get('params', {})),
            'execution_id': result.get('execution_id', 0),
            'total_signals': result.get('total_signals', 0),
            'winners': result.get('winners', 0),
            'losers': result.get('losers', 0),
            'win_rate': result.get('win_rate', 0),
            'profit_factor': result.get('profit_factor', 0),
            'total_pips': result.get('total_pips', 0),
            'avg_profit_pips': result.get('avg_profit_pips', 0),
            'avg_loss_pips': result.get('avg_loss_pips', 0),
            'composite_score': result.get('composite_score', 0),
            'status': result.get('status', 'error'),
            'error_message': result.get('error_message', ''),
            'duration_seconds': result.get('duration_seconds', 0),
        })
    except Exception as e:
        pass  # Don't fail the worker for DB storage issues


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class OptimizationResult:
    """Single backtest result"""
    epic: str
    params: Dict[str, Any]
    execution_id: int = 0
    total_signals: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pips: float = 0.0
    avg_profit_pips: float = 0.0
    avg_loss_pips: float = 0.0
    composite_score: float = 0.0
    status: str = 'pending'
    error_message: str = ''
    duration_seconds: float = 0.0


@dataclass
class OptimizationRun:
    """Optimization run state"""
    run_id: int
    run_name: str
    mode: str
    epics: List[str]
    days: int
    start_date: datetime
    end_date: datetime
    parameter_grid: Dict[str, List[Any]]
    total_combinations: int
    completed: int = 0
    current_epic_idx: int = 0
    current_param_idx: int = 0
    status: str = 'pending'
    results: List[OptimizationResult] = field(default_factory=list)
    best_per_epic: Dict[str, OptimizationResult] = field(default_factory=dict)


# ============================================
# MAIN OPTIMIZER CLASS
# ============================================

class MultiEpicOptimizer:
    """
    Multi-epic parameter optimizer with resume capability.

    Features:
    - Sequential epic processing (one at a time)
    - Progress persistence for resume
    - Visual progress bar with ETA
    - Database result storage
    - CSV export
    - Auto-snapshot creation
    """

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        days: int = DEFAULT_DAYS,
        epics: List[str] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        auto_snapshot: bool = True,
        parallel_workers: int = None,  # None=sequential, 0=auto, N=fixed workers
        max_memory_gb: float = 8.0
    ):
        self.mode = mode
        self.days = days
        self.output_dir = output_dir
        self.auto_snapshot = auto_snapshot
        self.parallel_workers = parallel_workers
        self.max_memory_gb = max_memory_gb

        # Resolve epic shortcuts
        if epics:
            self.epics = [resolve_epic(e) for e in epics]
        else:
            self.epics = list(config.EPIC_LIST)

        self.backtest_cmd = EnhancedBacktestCommands()
        self.db = DatabaseManager(config.DATABASE_URL)
        self.config_service = get_backtest_config_service()
        self.logger = logging.getLogger(__name__)

        # State
        self.run: Optional[OptimizationRun] = None
        self.combinations: List[Dict] = []
        self.interrupted = False

        # Timing for ETA calculation
        self.test_times: List[float] = []
        self.optimization_start_time: Optional[datetime] = None

        # Progress bar state
        self.last_result: Optional[OptimizationResult] = None

        # Parallel execution state
        self._parallel_progress = {}  # {epic: status_string}

        # Setup interrupt handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle graceful interruption"""
        print("\n\n⚠️  Interrupt received - saving progress...")
        self.interrupted = True
        self._save_progress()

    def _suppress_logging(self):
        """Aggressively suppress all logging output"""
        logging.disable(logging.CRITICAL)
        for name in logging.root.manager.loggerDict:
            log = logging.getLogger(name)
            log.setLevel(logging.CRITICAL)
            log.disabled = True

    # ============================================
    # PARALLEL EXECUTION SUPPORT
    # ============================================

    def _get_safe_worker_count(self, max_workers: int = 4) -> int:
        """
        Determine safe worker count based on system resources.

        Args:
            max_workers: Maximum workers to consider

        Returns:
            Safe number of workers (1 = sequential fallback)
        """
        if not PSUTIL_AVAILABLE:
            print("⚠️  psutil not available, using conservative 2 workers")
            return min(2, max_workers)

        try:
            cpu_count = os.cpu_count() or 4
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024 ** 3)

            # Each worker needs ~500MB-1GB for backtest data
            memory_per_worker_gb = 0.75
            memory_based_workers = int(min(available_memory_gb, self.max_memory_gb) / memory_per_worker_gb)

            # Don't exceed CPU count - 1 (leave 1 core for system)
            cpu_based_workers = max(1, cpu_count - 1)

            # Don't exceed number of epics
            epic_based_workers = len(self.epics)

            safe_workers = min(max_workers, memory_based_workers, cpu_based_workers, epic_based_workers)
            safe_workers = max(1, safe_workers)  # At least 1

            print(f"🔧 Resource check: CPU={cpu_count}, Available RAM={available_memory_gb:.1f}GB")
            print(f"   Safe workers: {safe_workers} (CPU limit: {cpu_based_workers}, Memory limit: {memory_based_workers})")

            return safe_workers

        except Exception as e:
            print(f"⚠️  Error checking resources: {e}, using 2 workers")
            return min(2, max_workers)

    def _run_parallel_optimization(self, num_workers: int) -> None:
        """
        Run optimization with parallel epic processing.

        Args:
            num_workers: Number of parallel workers to use
        """
        import threading

        print(f"\n🚀 Starting PARALLEL optimization with {num_workers} workers")
        print(f"   Processing {len(self.epics)} epics with {len(self.combinations)} combinations each\n")

        # Create shared progress dict
        manager = Manager()
        progress_dict = manager.dict()

        # Initialize progress for all epics
        for epic in self.epics:
            pair_name = self._get_epic_shortcut(epic)
            progress_dict[epic] = "⏳ Queued"

        completed_epics = {}
        all_results = []

        # Background thread for periodic progress updates
        stop_progress_thread = threading.Event()

        def progress_reporter():
            """Print progress every 30 seconds"""
            while not stop_progress_thread.wait(30):
                self._print_parallel_progress(progress_dict)

        progress_thread = threading.Thread(target=progress_reporter, daemon=True)
        progress_thread.start()

        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all epics using module-level worker function
                # (module-level function avoids pickle issues with self)
                future_to_epic = {
                    executor.submit(
                        _worker_optimize_epic,  # Module-level function
                        epic,
                        self.combinations,
                        self.run.start_date,
                        self.run.end_date,
                        self.run.run_id,
                        config.DATABASE_URL,  # Pass URL, not connection
                        progress_dict
                    ): epic for epic in self.epics
                }

                # Print initial progress
                self._print_parallel_progress(progress_dict)

                # Process as they complete
                for future in as_completed(future_to_epic):
                    epic = future_to_epic[future]
                    pair_name = self._get_epic_shortcut(epic)

                    try:
                        epic_name, results = future.result()
                        completed_epics[epic] = results
                        all_results.extend(results)

                        # Find best for this epic
                        successful = [r for r in results if r.get('status') == 'completed']
                        if successful:
                            best = max(successful, key=lambda x: x.get('composite_score', 0))
                            self.run.best_per_epic[epic] = self._dict_to_result(best)

                            print(f"\n✅ {pair_name} completed: {len(results)} tests")
                            print(f"   Best: SL={best['params'].get('fixed_stop_loss_pips')}, "
                                  f"TP={best['params'].get('fixed_take_profit_pips')}, "
                                  f"Score={best.get('composite_score', 0):.1f}")

                            # Create snapshot if enabled
                            if self.auto_snapshot:
                                best_result = self._dict_to_result(best)
                                snapshot_name = self.create_snapshot(epic, best['params'], best_result)
                                if snapshot_name:
                                    print(f"   📦 Snapshot: {snapshot_name}")
                        else:
                            print(f"\n⚠️  {pair_name} completed: No successful tests")

                        # Update completed count
                        self.run.completed += len(results)

                    except Exception as e:
                        print(f"\n❌ {pair_name} failed: {e}")

                    # Print current progress
                    self._print_parallel_progress(progress_dict)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupt received - workers will complete current tests...")
            self.interrupted = True
        finally:
            # Stop the progress reporter thread
            stop_progress_thread.set()

        # Convert results to OptimizationResult objects
        for result_dict in all_results:
            self.run.results.append(self._dict_to_result(result_dict))

    def _dict_to_result(self, d: dict) -> OptimizationResult:
        """Convert result dict to OptimizationResult object"""
        return OptimizationResult(
            epic=d.get('epic', ''),
            params=d.get('params', {}),
            execution_id=d.get('execution_id', 0),
            total_signals=d.get('total_signals', 0),
            winners=d.get('winners', 0),
            losers=d.get('losers', 0),
            win_rate=d.get('win_rate', 0),
            profit_factor=d.get('profit_factor', 0),
            total_pips=d.get('total_pips', 0),
            avg_profit_pips=d.get('avg_profit_pips', 0),
            avg_loss_pips=d.get('avg_loss_pips', 0),
            composite_score=d.get('composite_score', 0),
            status=d.get('status', 'error'),
            error_message=d.get('error_message', ''),
            duration_seconds=d.get('duration_seconds', 0)
        )

    def _print_parallel_progress(self, progress_dict: dict) -> None:
        """Print progress for all parallel workers"""
        print("\n" + "-" * 50)
        print("📊 PARALLEL PROGRESS:")
        for epic in self.epics:
            pair_name = self._get_epic_shortcut(epic)
            status = progress_dict.get(epic, "Unknown")
            print(f"   {pair_name:8s}: {status}")
        print("-" * 50)

    # ============================================
    # COMBINATION GENERATION
    # ============================================

    def generate_combinations(self) -> List[Dict]:
        """Generate parameter combinations with R:R filtering"""
        params = PARAMETER_TIERS[self.mode]
        combinations = []

        keys = list(params.keys())

        for values in product(*params.values()):
            param_dict = dict(zip(keys, values))

            # Get SL and TP for R:R calculation
            sl = param_dict.get('fixed_stop_loss_pips', 10)
            tp = param_dict.get('fixed_take_profit_pips', 15)

            # Skip if missing SL/TP
            if not sl or not tp:
                continue

            rr = tp / sl

            # Apply R:R filter
            if rr < RR_FILTER['min_rr_ratio'] or rr > RR_FILTER['max_rr_ratio']:
                continue

            param_dict['rr_ratio'] = round(rr, 2)
            combinations.append(param_dict)

        return combinations

    # ============================================
    # DATABASE OPERATIONS
    # ============================================

    def create_run(self) -> int:
        """Create new optimization run in database"""
        run_name = f"multiepic_{datetime.now().strftime('%Y%m%d_%H%M')}_{self.mode}"

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)
        total = len(self.combinations) * len(self.epics)

        query = """
            INSERT INTO optimization_runs (
                run_name, run_mode, epics_to_test, days_tested,
                start_date, end_date, parameter_grid, total_combinations,
                status, started_at
            ) VALUES (
                :run_name, :run_mode, :epics, :days,
                :start_date, :end_date, :param_grid, :total,
                'running', NOW()
            ) RETURNING id
        """

        # Convert lists to PostgreSQL array format
        epics_pg = '{' + ','.join(f'"{e}"' for e in self.epics) + '}'

        result = self.db.execute_query(query, {
            'run_name': run_name,
            'run_mode': self.mode,
            'epics': epics_pg,
            'days': self.days,
            'start_date': start_date,
            'end_date': end_date,
            'param_grid': json.dumps(PARAMETER_TIERS[self.mode]),
            'total': total,
        })

        if result.empty:
            raise Exception("Failed to create optimization run")

        run_id = int(result.iloc[0]['id'])

        # Initialize run object
        self.run = OptimizationRun(
            run_id=run_id,
            run_name=run_name,
            mode=self.mode,
            epics=self.epics,
            days=self.days,
            start_date=start_date,
            end_date=end_date,
            parameter_grid=PARAMETER_TIERS[self.mode],
            total_combinations=total,
            status='running'
        )

        return run_id

    def load_run(self, run_id: int) -> bool:
        """Load existing run for resume"""
        query = """
            SELECT * FROM optimization_runs WHERE id = :run_id
        """
        result = self.db.execute_query(query, {'run_id': run_id})

        if result.empty:
            print(f"❌ Run ID {run_id} not found")
            return False

        row = result.iloc[0]

        # Restore state
        self.mode = row['run_mode']
        self.epics = list(row['epics_to_test']) if row['epics_to_test'] else self.epics
        self.days = row['days_tested']

        # Regenerate combinations
        self.combinations = self.generate_combinations()

        # Create run object
        self.run = OptimizationRun(
            run_id=run_id,
            run_name=row['run_name'],
            mode=self.mode,
            epics=self.epics,
            days=self.days,
            start_date=row['start_date'],
            end_date=row['end_date'],
            parameter_grid=PARAMETER_TIERS[self.mode],
            total_combinations=row['total_combinations'],
            completed=row['completed_combinations'] or 0,
            current_epic_idx=row['current_epic_idx'] or 0,
            current_param_idx=row['current_param_idx'] or 0,
            status='running'
        )

        print(f"✅ Resuming run {run_id}: {self.run.run_name}")
        print(f"   Progress: {self.run.completed}/{self.run.total_combinations} "
              f"({self.run.completed/self.run.total_combinations*100:.1f}%)")

        return True

    def _save_progress(self):
        """Save current progress to database"""
        if not self.run:
            return

        current_epic = None
        if self.run.current_epic_idx < len(self.epics):
            current_epic = self.epics[self.run.current_epic_idx]

        query = """
            UPDATE optimization_runs SET
                completed_combinations = :completed,
                current_epic = :current_epic,
                current_epic_idx = :epic_idx,
                current_param_idx = :param_idx,
                status = :status,
                paused_at = CASE WHEN :is_paused THEN NOW() ELSE paused_at END,
                updated_at = NOW()
            WHERE id = :run_id
        """

        self.db.execute_query(query, {
            'completed': self.run.completed,
            'current_epic': current_epic,
            'epic_idx': self.run.current_epic_idx,
            'param_idx': self.run.current_param_idx,
            'status': 'paused' if self.interrupted else self.run.status,
            'is_paused': self.interrupted,
            'run_id': self.run.run_id,
        })

    def _store_result(self, result: OptimizationResult):
        """Store individual result in database"""
        query = """
            INSERT INTO optimization_results (
                run_id, epic, execution_id, params_tested, rr_ratio,
                total_signals, winners, losers, win_rate, profit_factor,
                total_pips, avg_profit_pips, avg_loss_pips,
                composite_score, status, error_message, duration_seconds
            ) VALUES (
                :run_id, :epic, :exec_id, :params, :rr,
                :signals, :winners, :losers, :win_rate, :pf,
                :pips, :avg_profit, :avg_loss,
                :score, :status, :error, :duration
            )
        """

        self.db.execute_query(query, {
            'run_id': self.run.run_id,
            'epic': result.epic,
            'exec_id': result.execution_id,
            'params': json.dumps(result.params),
            'rr': result.params.get('rr_ratio'),
            'signals': result.total_signals,
            'winners': result.winners,
            'losers': result.losers,
            'win_rate': result.win_rate,
            'pf': result.profit_factor,
            'pips': result.total_pips,
            'avg_profit': result.avg_profit_pips,
            'avg_loss': result.avg_loss_pips,
            'score': result.composite_score,
            'status': result.status,
            'error': result.error_message,
            'duration': result.duration_seconds,
        })

    def _store_best_params(self, epic: str, result: OptimizationResult, snapshot_name: str = None):
        """Store best params for an epic"""
        query = """
            INSERT INTO optimization_best_params (
                run_id, epic, best_params, total_signals,
                win_rate, profit_factor, total_pips, composite_score,
                snapshot_name
            ) VALUES (
                :run_id, :epic, :params, :signals,
                :win_rate, :pf, :pips, :score,
                :snapshot
            )
            ON CONFLICT (run_id, epic) DO UPDATE SET
                best_params = EXCLUDED.best_params,
                total_signals = EXCLUDED.total_signals,
                win_rate = EXCLUDED.win_rate,
                profit_factor = EXCLUDED.profit_factor,
                total_pips = EXCLUDED.total_pips,
                composite_score = EXCLUDED.composite_score,
                snapshot_name = EXCLUDED.snapshot_name
        """

        self.db.execute_query(query, {
            'run_id': self.run.run_id,
            'epic': epic,
            'params': json.dumps(result.params),
            'signals': result.total_signals,
            'win_rate': result.win_rate,
            'pf': result.profit_factor,
            'pips': result.total_pips,
            'score': result.composite_score,
            'snapshot': snapshot_name,
        })

    # ============================================
    # BACKTEST EXECUTION
    # ============================================

    def run_single_backtest(
        self,
        epic: str,
        params: Dict,
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Run single backtest and return result"""

        # Suppress logging
        self._suppress_logging()

        # Remove calculated fields from override
        override = {k: v for k, v in params.items() if k != 'rr_ratio'}

        start_time = time.time()

        try:
            success = self.backtest_cmd.run_enhanced_backtest(
                epic=epic,
                start_date=start_date,
                end_date=end_date,
                strategy='SMC_SIMPLE',
                config_override=override,
                use_historical_intelligence=False,  # DISABLED for optimization
                pipeline=False,  # Speed mode
                show_signals=False
            )

            duration = time.time() - start_time
            self.test_times.append(duration)

            if success:
                return self._extract_result(epic, params, duration)
            else:
                return OptimizationResult(
                    epic=epic,
                    params=params,
                    status='failed',
                    error_message='Backtest returned False',
                    duration_seconds=duration
                )

        except Exception as e:
            return OptimizationResult(
                epic=epic,
                params=params,
                status='failed',
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )

    def _extract_result(self, epic: str, params: Dict, duration: float) -> OptimizationResult:
        """Extract results from latest backtest execution"""

        query = """
            SELECT
                be.id as execution_id,
                COUNT(bs.id) as total_signals,
                SUM(CASE WHEN bs.trade_result = 'win' THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN bs.trade_result = 'loss' THEN 1 ELSE 0 END) as losers,
                SUM(bs.pips_gained) as total_pips,
                AVG(CASE WHEN bs.trade_result = 'win' THEN bs.pips_gained END) as avg_profit,
                AVG(CASE WHEN bs.trade_result = 'loss' THEN ABS(bs.pips_gained) END) as avg_loss,
                SUM(CASE WHEN bs.trade_result = 'win' THEN bs.pips_gained ELSE 0 END) as gross_profit,
                SUM(CASE WHEN bs.trade_result = 'loss' THEN ABS(bs.pips_gained) ELSE 0 END) as gross_loss
            FROM backtest_executions be
            LEFT JOIN backtest_signals bs ON be.id = bs.execution_id
            WHERE be.status = 'COMPLETED'
            GROUP BY be.id, be.created_at
            ORDER BY be.created_at DESC
            LIMIT 1
        """

        result = self.db.execute_query(query)

        if result.empty:
            return OptimizationResult(
                epic=epic, params=params,
                status='failed', error_message='No results found',
                duration_seconds=duration
            )

        row = result.iloc[0]
        winners = int(row['winners'] or 0)
        losers = int(row['losers'] or 0)
        total = winners + losers

        win_rate = winners / max(total, 1)
        gross_profit = float(row['gross_profit'] or 0)
        gross_loss = float(row['gross_loss'] or 0)
        profit_factor = gross_profit / max(gross_loss, 0.01)
        total_pips = float(row['total_pips'] or 0)

        # Calculate composite score
        composite = self._calculate_composite_score(
            win_rate, profit_factor, total_pips, int(row['total_signals'] or 0)
        )

        return OptimizationResult(
            epic=epic,
            params=params,
            execution_id=int(row['execution_id']),
            total_signals=int(row['total_signals'] or 0),
            winners=winners,
            losers=losers,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pips=total_pips,
            avg_profit_pips=float(row['avg_profit'] or 0),
            avg_loss_pips=float(row['avg_loss'] or 0),
            composite_score=composite,
            status='completed',
            duration_seconds=duration
        )

    def _calculate_composite_score(
        self,
        win_rate: float,
        profit_factor: float,
        total_pips: float,
        signal_count: int
    ) -> float:
        """Calculate composite score for ranking"""
        # Normalize profit factor (cap at 3.0)
        pf_norm = min(profit_factor / SCORE_NORMALIZATION['profit_factor_cap'], 1.0)

        # Normalize pips
        pips_norm = total_pips / SCORE_NORMALIZATION['pips_divisor']
        if pips_norm < -1.0:
            pips_norm = -1.0  # Floor at -1.0

        # Normalize signal count (cap at 50)
        signals_norm = min(signal_count / SCORE_NORMALIZATION['signal_count_cap'], 1.0)

        # Weighted composite
        score = (
            win_rate * SCORE_WEIGHTS['win_rate'] +
            pf_norm * SCORE_WEIGHTS['profit_factor'] +
            pips_norm * SCORE_WEIGHTS['total_pips'] +
            signals_norm * SCORE_WEIGHTS['signal_count']
        )

        return score

    # ============================================
    # PROGRESS DISPLAY
    # ============================================

    def _get_epic_shortcut(self, epic: str) -> str:
        """Get short name for epic"""
        return EPIC_TO_SHORTCUT.get(epic, epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', ''))

    def _calculate_eta(self) -> str:
        """Calculate estimated time remaining"""
        if not self.test_times or len(self.test_times) < 2:
            return "Calculating..."

        # Use recent average (last 10 tests)
        recent_times = self.test_times[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        remaining = self.run.total_combinations - self.run.completed
        eta_seconds = avg_time * remaining

        if eta_seconds > 3600:
            hours = eta_seconds / 3600
            return f"{hours:.1f}h"
        elif eta_seconds > 60:
            minutes = eta_seconds / 60
            return f"{minutes:.0f}m"
        else:
            return f"{eta_seconds:.0f}s"

    def _print_progress_bar(self, epic: str, param_idx: int, total_params: int, params: Dict):
        """Print visual progress bar"""
        # Overall progress
        overall_pct = (self.run.completed / self.run.total_combinations) * 100
        epic_pct = (param_idx / total_params) * 100

        # Progress bar (30 chars)
        bar_width = 30
        filled = int(bar_width * epic_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)

        # Current params
        sl = params.get('fixed_stop_loss_pips', '-')
        tp = params.get('fixed_take_profit_pips', '-')
        conf = params.get('min_confidence', 0)

        # ETA
        eta = self._calculate_eta()

        # Average time per test
        avg_time = sum(self.test_times) / len(self.test_times) if self.test_times else 0

        # Build progress line
        epic_name = self._get_epic_shortcut(epic)
        progress_line = (
            f"\r{epic_name} [{bar}] {epic_pct:5.1f}% ({param_idx}/{total_params}) | "
            f"Overall: {overall_pct:5.1f}% | ETA: {eta} | "
            f"SL={sl}, TP={tp}, Conf={conf:.0%}"
        )

        # Print progress (overwrite line)
        print(progress_line, end='', flush=True)

    def _print_result_line(self, result: OptimizationResult):
        """Print result after progress bar"""
        if result.status == 'completed':
            status = '✅'
            details = (
                f"Signals={result.total_signals}, "
                f"Win={result.win_rate:.0%}, "
                f"PF={result.profit_factor:.2f}, "
                f"Pips={result.total_pips:+.1f}"
            )
        else:
            status = '❌'
            details = result.error_message[:50] if result.error_message else 'Unknown error'

        # Average time
        avg_time = sum(self.test_times) / len(self.test_times) if self.test_times else 0

        print(f"\n  {status} {details} | Avg: {avg_time:.0f}s/test", flush=True)

    # ============================================
    # SNAPSHOT CREATION
    # ============================================

    def create_snapshot(self, epic: str, params: Dict, result: OptimizationResult) -> Optional[str]:
        """Create snapshot for best params"""
        pair_name = self._get_epic_shortcut(epic).lower()
        snapshot_name = f"{pair_name}_optimized_{datetime.now().strftime('%Y%m%d')}"

        # Remove non-config params
        override = {k: v for k, v in params.items() if k != 'rr_ratio'}

        description = (
            f"Optimized {pair_name.upper()} config from {self.days}-day backtest ({self.mode} mode). "
            f"Win rate: {result.win_rate:.1%}, PF: {result.profit_factor:.2f}, "
            f"Pips: {result.total_pips:+.1f}, Signals: {result.total_signals}"
        )

        try:
            snapshot_id = self.config_service.create_snapshot(
                name=snapshot_name,
                parameter_overrides=override,
                description=description,
                created_by='multi_epic_optimizer',
                tags=['auto-optimized', pair_name, self.mode]
            )

            if snapshot_id:
                return snapshot_name
        except Exception as e:
            # Snapshot may already exist - try with timestamp
            try:
                snapshot_name = f"{pair_name}_optimized_{datetime.now().strftime('%Y%m%d_%H%M')}"
                snapshot_id = self.config_service.create_snapshot(
                    name=snapshot_name,
                    parameter_overrides=override,
                    description=description,
                    created_by='multi_epic_optimizer',
                    tags=['auto-optimized', pair_name, self.mode]
                )
                if snapshot_id:
                    return snapshot_name
            except Exception:
                pass

        return None

    # ============================================
    # CSV EXPORT
    # ============================================

    def export_csv(self, filepath: str):
        """Export results to CSV"""
        if not self.run or not self.run.results:
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [
                'epic', 'fixed_stop_loss_pips', 'fixed_take_profit_pips',
                'min_confidence', 'rr_ratio', 'sl_buffer_pips',
                'total_signals', 'winners', 'losers',
                'win_rate', 'profit_factor', 'total_pips',
                'composite_score', 'status', 'duration_seconds'
            ]
            writer.writerow(header)

            # Data
            for r in self.run.results:
                writer.writerow([
                    self._get_epic_shortcut(r.epic),
                    r.params.get('fixed_stop_loss_pips', ''),
                    r.params.get('fixed_take_profit_pips', ''),
                    r.params.get('min_confidence', ''),
                    r.params.get('rr_ratio', ''),
                    r.params.get('sl_buffer_pips', ''),
                    r.total_signals,
                    r.winners,
                    r.losers,
                    f"{r.win_rate:.4f}",
                    f"{r.profit_factor:.3f}",
                    f"{r.total_pips:.2f}",
                    f"{r.composite_score:.6f}",
                    r.status,
                    f"{r.duration_seconds:.1f}"
                ])

        print(f"\n📊 Results exported to: {filepath}")

    # ============================================
    # REPORTS
    # ============================================

    def print_comparison_report(self):
        """Print cross-epic comparison of best parameters"""
        if not self.run or not self.run.best_per_epic:
            return

        print("\n" + "=" * 100)
        print("CROSS-EPIC PARAMETER COMPARISON")
        print("=" * 100)

        # Header
        print(f"\n{'Epic':<10} {'SL':<5} {'TP':<5} {'R:R':<6} {'Conf':<6} "
              f"{'Signals':<8} {'Win%':<8} {'PF':<8} {'Pips':<12} {'Score':<10}")
        print("-" * 100)

        # Sort by composite score
        sorted_epics = sorted(
            self.run.best_per_epic.items(),
            key=lambda x: x[1].composite_score,
            reverse=True
        )

        for epic, result in sorted_epics:
            pair_name = self._get_epic_shortcut(epic)
            p = result.params

            print(f"{pair_name:<10} "
                  f"{p.get('fixed_stop_loss_pips', '-'):<5} "
                  f"{p.get('fixed_take_profit_pips', '-'):<5} "
                  f"{p.get('rr_ratio', 0):<6.2f} "
                  f"{p.get('min_confidence', 0)*100:<5.0f}% "
                  f"{result.total_signals:<8} "
                  f"{result.win_rate*100:<7.1f}% "
                  f"{result.profit_factor:<8.2f} "
                  f"{result.total_pips:<+12.1f} "
                  f"{result.composite_score:<10.4f}")

        print("=" * 100)

        # Best overall
        if sorted_epics:
            best_epic, best_result = sorted_epics[0]
            best_pair = self._get_epic_shortcut(best_epic)

            print(f"\n🏆 BEST PERFORMING PAIR: {best_pair}")
            print(f"   Parameters: SL={best_result.params.get('fixed_stop_loss_pips')}, "
                  f"TP={best_result.params.get('fixed_take_profit_pips')}, "
                  f"Conf={best_result.params.get('min_confidence', 0):.0%}")
            print(f"   Performance: Win Rate={best_result.win_rate:.1%}, "
                  f"PF={best_result.profit_factor:.2f}, "
                  f"Pips={best_result.total_pips:+.1f}")

            if best_result.params.get('snapshot_name'):
                print(f"   Snapshot: {best_result.params.get('snapshot_name')}")

    # ============================================
    # MAIN OPTIMIZATION LOOP
    # ============================================

    def _finalize_run(self):
        """Finalize the optimization run with report and export."""
        if not self.interrupted:
            self.run.status = 'completed'

            # Update run status in database
            query = """
                UPDATE optimization_runs SET
                    status = 'completed',
                    completed_at = NOW(),
                    best_overall_epic = :best_epic,
                    best_overall_params = :best_params,
                    best_overall_score = :best_score
                WHERE id = :run_id
            """

            best_epic = None
            best_params = None
            best_score = None

            if self.run.best_per_epic:
                sorted_best = sorted(
                    self.run.best_per_epic.items(),
                    key=lambda x: x[1].composite_score,
                    reverse=True
                )
                best_epic, best_result = sorted_best[0]
                best_params = json.dumps(best_result.params)
                best_score = best_result.composite_score

            self.db.execute_query(query, {
                'run_id': self.run.run_id,
                'best_epic': best_epic,
                'best_params': best_params,
                'best_score': best_score,
            })

            # Print comparison report
            self.print_comparison_report()

            # Export CSV
            csv_path = f"{self.output_dir}/optimization_{self.run.run_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            self.export_csv(csv_path)

            # Final stats
            total_duration = (datetime.now() - self.optimization_start_time).total_seconds()
            print(f"\n{'='*80}")
            print(f"✅ OPTIMIZATION COMPLETE")
            print(f"{'='*80}")
            print(f"Run ID:      {self.run.run_id}")
            print(f"Duration:    {total_duration / 3600:.1f} hours")
            print(f"Tests run:   {self.run.completed}")
            print(f"CSV export:  {csv_path}")
            print(f"{'='*80}")

        else:
            print(f"\n{'='*80}")
            print(f"⏸️  OPTIMIZATION PAUSED")
            print(f"{'='*80}")
            print(f"Progress saved. Resume with:")
            print(f"  python multi_epic_optimizer.py --resume {self.run.run_id}")
            print(f"{'='*80}")

        self._save_progress()

    def run_optimization(self, resume_run_id: int = None):
        """Main optimization loop"""

        # Generate combinations
        self.combinations = self.generate_combinations()

        if resume_run_id:
            if not self.load_run(resume_run_id):
                return
        else:
            # Create new run
            self.create_run()

        self.optimization_start_time = datetime.now()

        # Print header
        print("\n" + "=" * 80)
        print("🚀 MULTI-EPIC PARAMETER OPTIMIZATION")
        print("=" * 80)
        # Determine execution mode
        if self.parallel_workers is not None:
            if self.parallel_workers == 0:
                actual_workers = self._get_safe_worker_count(max_workers=8)
            else:
                actual_workers = min(self.parallel_workers, len(self.epics))
            execution_mode = f"PARALLEL ({actual_workers} workers)"
            est_hours = len(self.combinations) * len(self.epics) * 3 / 60 / actual_workers
        else:
            actual_workers = 0
            execution_mode = "SEQUENTIAL"
            est_hours = len(self.combinations) * len(self.epics) * 3 / 60

        print(f"Mode:        {self.mode.upper()}")
        print(f"Execution:   {execution_mode}")
        print(f"Epics:       {len(self.epics)} pairs")
        print(f"Period:      {self.days} days ({self.run.start_date.strftime('%Y-%m-%d')} to {self.run.end_date.strftime('%Y-%m-%d')})")
        print(f"Combinations: {len(self.combinations)} per epic")
        print(f"Total tests: {self.run.total_combinations}")
        print(f"Est. time:   ~{est_hours:.1f} hours")
        print(f"Historical Intelligence: DISABLED")
        print(f"Auto-snapshot: {'ENABLED' if self.auto_snapshot else 'DISABLED'}")
        print("=" * 80)
        print("\nPress Ctrl+C to pause and save progress.\n")

        # Run parallel or sequential
        if self.parallel_workers is not None and actual_workers > 1:
            self._run_parallel_optimization(actual_workers)
            # After parallel completion, skip to final report
            self._finalize_run()
            return

        # Main loop
        for epic_idx in range(self.run.current_epic_idx, len(self.epics)):
            if self.interrupted:
                break

            epic = self.epics[epic_idx]
            pair_name = self._get_epic_shortcut(epic)

            print(f"\n{'='*60}")
            print(f"📈 OPTIMIZING {pair_name} ({epic_idx + 1}/{len(self.epics)})")
            print(f"{'='*60}")

            epic_results = []

            # Start from current_param_idx if resuming this epic
            start_param_idx = self.run.current_param_idx if epic_idx == self.run.current_epic_idx else 0

            for param_idx in range(start_param_idx, len(self.combinations)):
                if self.interrupted:
                    break

                params = self.combinations[param_idx]
                self.run.current_epic_idx = epic_idx
                self.run.current_param_idx = param_idx

                # Print progress bar
                self._print_progress_bar(epic, param_idx + 1, len(self.combinations), params)

                # Run backtest
                result = self.run_single_backtest(
                    epic, params,
                    self.run.start_date, self.run.end_date
                )

                # Store result
                self._store_result(result)
                self.run.results.append(result)
                epic_results.append(result)
                self.run.completed += 1
                self.last_result = result

                # Print result line
                self._print_result_line(result)

            # Reset param index for next epic
            self.run.current_param_idx = 0

            # Find best for this epic
            successful = [r for r in epic_results if r.status == 'completed']
            if successful:
                best = max(successful, key=lambda x: x.composite_score)
                self.run.best_per_epic[epic] = best

                print(f"\n  🏆 Best for {pair_name}:")
                print(f"     SL={best.params.get('fixed_stop_loss_pips')}, "
                      f"TP={best.params.get('fixed_take_profit_pips')}, "
                      f"Conf={best.params.get('min_confidence', 0):.0%}")
                print(f"     Win Rate: {best.win_rate:.1%}, "
                      f"PF: {best.profit_factor:.2f}, "
                      f"Pips: {best.total_pips:+.1f}")

                # Create snapshot
                if self.auto_snapshot:
                    snapshot_name = self.create_snapshot(epic, best.params, best)
                    if snapshot_name:
                        print(f"     📦 Snapshot: {snapshot_name}")
                        self._store_best_params(epic, best, snapshot_name)
                    else:
                        self._store_best_params(epic, best)
                else:
                    self._store_best_params(epic, best)

        # Final report
        self._finalize_run()


# ============================================
# DRY RUN
# ============================================

def dry_run(mode: str, days: int, epics: List[str]):
    """Show what would be tested without executing"""
    optimizer = MultiEpicOptimizer(mode=mode, days=days, epics=epics)
    combos = optimizer.generate_combinations()
    tier_info = get_tier_info(mode)

    print("\n" + "=" * 60)
    print("DRY RUN - No tests will be executed")
    print("=" * 60)
    print(f"Mode:         {mode.upper()}")
    print(f"Days:         {days}")
    print(f"Epics:        {len(optimizer.epics)}")
    print(f"Parameters:   {', '.join(tier_info['parameters'])}")
    print(f"Raw combos:   {tier_info['raw_combinations']}")
    print(f"After R:R filter: {len(combos)}")
    print(f"Total tests:  {len(combos) * len(optimizer.epics)}")
    print(f"Est. time:    ~{len(combos) * len(optimizer.epics) * 3 / 60:.1f} hours")

    print(f"\nEpics to test:")
    for epic in optimizer.epics:
        print(f"  - {optimizer._get_epic_shortcut(epic)}")

    print(f"\nSample combinations (first 5):")
    for i, c in enumerate(combos[:5], 1):
        print(f"  {i}. SL={c.get('fixed_stop_loss_pips')}, "
              f"TP={c.get('fixed_take_profit_pips')}, "
              f"Conf={c.get('min_confidence', 0):.0%}, "
              f"R:R={c.get('rr_ratio', 0):.2f}")

    print("=" * 60)


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Epic Parameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mode - core parameters only (~6-7 hours sequential)
  python multi_epic_optimizer.py --mode fast --days 30

  # Medium mode - extended parameters (~22-25 hours sequential)
  python multi_epic_optimizer.py --mode medium --days 30

  # Extended mode - full sweep (~45-50 hours sequential)
  python multi_epic_optimizer.py --mode extended --days 30

  # Specific epics only
  python multi_epic_optimizer.py --mode fast --epics EURUSD GBPUSD USDJPY

  # Resume interrupted run
  python multi_epic_optimizer.py --resume 12345

  # Dry run (show combinations without executing)
  python multi_epic_optimizer.py --mode extended --dry-run

  # Disable auto-snapshot
  python multi_epic_optimizer.py --mode fast --no-snapshots

  # Enable parallel execution (auto-detect workers)
  python multi_epic_optimizer.py --mode fast --days 30 --parallel

  # Parallel with specific worker count
  python multi_epic_optimizer.py --mode fast --days 30 --parallel 4

  # Parallel with memory limit
  python multi_epic_optimizer.py --mode fast --days 30 --parallel --max-memory-gb 4
"""
    )

    parser.add_argument(
        '--mode', type=str, default='fast',
        choices=['fast', 'medium', 'extended'],
        help='Optimization mode: fast (~15 combos), medium (~50), extended (~100)'
    )

    parser.add_argument(
        '--days', type=int, default=30,
        help='Backtest period in days (default: 30)'
    )

    parser.add_argument(
        '--epics', nargs='+', type=str,
        help='Specific epic shortcuts to test (e.g., EURUSD GBPUSD)'
    )

    parser.add_argument(
        '--resume', type=int,
        help='Resume interrupted run by run ID'
    )

    parser.add_argument(
        '--output-dir', type=str,
        default='/app/forex_scanner/optimization_results',
        help='Output directory for CSV exports'
    )

    parser.add_argument(
        '--no-snapshots', action='store_true',
        help='Do not create snapshots for best configs'
    )

    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be tested without running'
    )

    parser.add_argument(
        '--parallel', nargs='?', const=0, type=int, default=None,
        help='Enable parallel execution. No value=auto-detect workers, or specify count (2-8)'
    )

    parser.add_argument(
        '--max-memory-gb', type=float, default=8.0,
        help='Maximum memory to use in GB (default: 8.0). Workers adjusted to fit.'
    )

    args = parser.parse_args()

    # Dry run mode
    if args.dry_run:
        dry_run(args.mode, args.days, args.epics)
        return

    # Create optimizer
    optimizer = MultiEpicOptimizer(
        mode=args.mode,
        days=args.days,
        epics=args.epics,
        output_dir=args.output_dir,
        auto_snapshot=not args.no_snapshots,
        parallel_workers=args.parallel,
        max_memory_gb=args.max_memory_gb
    )

    # Run optimization
    optimizer.run_optimization(resume_run_id=args.resume)


if __name__ == '__main__':
    main()
