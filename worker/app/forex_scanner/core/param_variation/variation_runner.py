"""
Parallel Variation Runner

Executes multiple parameter variations in parallel for the same epic.
Uses ProcessPoolExecutor for TRUE parallel execution on multiple CPU cores.
"""

import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

from .variation_result import VariationResult, ResultRanker

logger = logging.getLogger(__name__)


@dataclass
class VariationRunConfig:
    """Configuration for a variation run"""
    epic: str
    days: int
    strategy: str = 'SMC_SIMPLE'
    timeframe: str = '15m'
    max_workers: int = 4
    rank_by: str = 'composite_score'
    top_n: int = 10
    use_historical_intelligence: bool = False
    pipeline: bool = False

    # Date range (alternative to days)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


def _run_single_variation_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function that runs in a separate process.
    Must be defined at module level to be picklable.

    Args:
        args: Tuple of (params, variation_index, config_dict)

    Returns:
        Dictionary with result data (VariationResult can't be pickled directly)
    """
    params, variation_index, config_dict = args
    start_time = time.time()

    result_data = {
        'variation_index': variation_index,
        'params': params.copy(),
        'status': 'running',
        'error': None,
        'signal_count': 0,
        'win_count': 0,
        'loss_count': 0,
        'total_pips': 0.0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'avg_win_pips': 0.0,
        'avg_loss_pips': 0.0,
        'max_drawdown_pips': 0.0,
        'duration_seconds': 0.0
    }

    try:
        # Import inside worker to avoid pickling issues
        from forex_scanner.commands.enhanced_backtest_commands import EnhancedBacktestCommands

        # Calculate date range
        if config_dict.get('start_date') and config_dict.get('end_date'):
            start_date = config_dict['start_date']
            end_date = config_dict['end_date']
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config_dict['days'])

        # Create fresh backtest handler for this process
        enhanced_backtest = EnhancedBacktestCommands()

        # Run backtest with parameter override
        backtest_result = enhanced_backtest.run_enhanced_backtest(
            epic=config_dict['epic'],
            days=config_dict['days'],
            start_date=start_date,
            end_date=end_date,
            strategy=config_dict['strategy'],
            timeframe=config_dict['timeframe'],
            config_override=params,
            use_historical_intelligence=config_dict.get('use_historical_intelligence', False),
            pipeline=config_dict.get('pipeline', False),
            show_signals=False,  # Suppress output
            return_results=True  # Get result dict
        )

        # Extract metrics from result
        if backtest_result and isinstance(backtest_result, dict):
            _extract_metrics_to_dict(result_data, backtest_result)
            result_data['status'] = 'completed'
        else:
            result_data['status'] = 'failed'
            result_data['error'] = 'Backtest returned no results'

    except Exception as e:
        result_data['status'] = 'failed'
        result_data['error'] = str(e)

    result_data['duration_seconds'] = time.time() - start_time
    return result_data


def _extract_metrics_to_dict(result_data: Dict, backtest_result: Dict[str, Any]) -> None:
    """Extract metrics from backtest result into result_data dictionary"""

    # Get order logger for detailed stats if available
    order_logger = backtest_result.get('order_logger')
    if order_logger and hasattr(order_logger, 'get_summary_statistics'):
        stats = order_logger.get_summary_statistics()

        result_data['signal_count'] = stats.get('total_signals', 0)
        result_data['win_count'] = stats.get('winning_trades', 0)
        result_data['loss_count'] = stats.get('losing_trades', 0)
        result_data['total_pips'] = stats.get('total_pips', 0.0)
        result_data['win_rate'] = stats.get('win_rate', 0.0)
        result_data['profit_factor'] = stats.get('profit_factor', 0.0)
        result_data['avg_win_pips'] = stats.get('avg_win_pips', 0.0)
        result_data['avg_loss_pips'] = stats.get('avg_loss_pips', 0.0)
        result_data['max_drawdown_pips'] = stats.get('max_drawdown_pips', 0.0)

    else:
        # Fallback: extract from signals directly
        signals = backtest_result.get('all_signals', [])
        if not signals:
            # Try alternate key
            for epic_key in backtest_result:
                if isinstance(backtest_result[epic_key], list):
                    signals = backtest_result[epic_key]
                    break

        result_data['signal_count'] = len(signals) if signals else 0

        # Calculate metrics from signals
        if signals:
            wins = []
            losses = []

            for sig in signals:
                pips = sig.get('pips_gained', 0)
                if pips is None:
                    continue
                if pips > 0:
                    wins.append(pips)
                else:
                    losses.append(pips)

            result_data['win_count'] = len(wins)
            result_data['loss_count'] = len(losses)
            result_data['total_pips'] = sum(wins) + sum(losses)

            if result_data['signal_count'] > 0:
                result_data['win_rate'] = (result_data['win_count'] / result_data['signal_count']) * 100

            if wins:
                result_data['avg_win_pips'] = sum(wins) / len(wins)
            if losses:
                result_data['avg_loss_pips'] = sum(losses) / len(losses)

            # Profit factor
            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            if total_losses > 0:
                result_data['profit_factor'] = total_wins / total_losses


class ParallelVariationRunner:
    """
    Execute multiple parameter variations in parallel for the same epic.

    Uses ProcessPoolExecutor for TRUE parallel execution across CPU cores.
    Each worker process has its own database connection and backtest handler.
    """

    def __init__(
        self,
        config: VariationRunConfig,
        progress_callback: Optional[Callable[[int, int, VariationResult], None]] = None
    ):
        """
        Initialize the variation runner.

        Args:
            config: Variation run configuration
            progress_callback: Optional callback for progress updates
                              Called with (completed_count, total_count, latest_result)
        """
        self.config = config
        self.progress_callback = progress_callback
        self.ranker = ResultRanker()

        # Calculate date range
        if config.start_date and config.end_date:
            self.start_date = config.start_date
            self.end_date = config.end_date
        else:
            self.end_date = datetime.now()
            self.start_date = self.end_date - timedelta(days=config.days)

    def run_variations(
        self,
        param_sets: List[Dict[str, Any]],
    ) -> List[VariationResult]:
        """
        Execute all parameter variations in parallel using separate processes.

        Args:
            param_sets: List of parameter dictionaries to test

        Returns:
            List of VariationResult objects, sorted by rank
        """
        if not param_sets:
            logger.warning("No parameter sets to test")
            return []

        total = len(param_sets)

        # Limit workers to available CPUs and requested max
        available_cpus = mp.cpu_count()
        actual_workers = min(self.config.max_workers, available_cpus, total)

        logger.info(f"Starting PARALLEL variation testing: {total} combinations")
        logger.info(f"Epic: {self.config.epic}, Days: {self.config.days}")
        logger.info(f"Workers: {actual_workers} (requested: {self.config.max_workers}, CPUs: {available_cpus})")

        print(f"\n🚀 Starting parallel variation testing with {actual_workers} CPU processes...")
        print(f"   Testing {total} parameter combinations")

        # Track timing
        start_time = time.time()

        # Prepare config dict for worker (must be picklable)
        config_dict = {
            'epic': self.config.epic,
            'days': self.config.days,
            'strategy': self.config.strategy,
            'timeframe': self.config.timeframe,
            'use_historical_intelligence': self.config.use_historical_intelligence,
            'pipeline': self.config.pipeline,
            'start_date': self.start_date,
            'end_date': self.end_date
        }

        # Prepare arguments for workers
        worker_args = [
            (params, idx, config_dict)
            for idx, params in enumerate(param_sets)
        ]

        # Execute variations in parallel processes
        results: List[VariationResult] = []
        completed_count = 0

        # Use 'spawn' context for cleaner process isolation
        ctx = mp.get_context('spawn')

        with ProcessPoolExecutor(max_workers=actual_workers, mp_context=ctx) as executor:
            # Submit all variations
            future_to_index = {
                executor.submit(_run_single_variation_worker, args): args[1]
                for args in worker_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result_data = future.result()

                    # Convert dict back to VariationResult
                    result = self._dict_to_variation_result(result_data)
                    results.append(result)
                    completed_count += 1

                    # Log progress
                    status_emoji = "✓" if result.status == 'completed' else "✗"
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total - completed_count) / rate if rate > 0 else 0

                    print(
                        f"  {status_emoji} [{completed_count}/{total}] "
                        f"Var {idx}: {result.status} "
                        f"(signals={result.signal_count}, pips={result.total_pips:+.1f}) "
                        f"[{elapsed:.0f}s elapsed, ETA: {eta:.0f}s]"
                    )

                    # Call progress callback
                    if self.progress_callback:
                        self.progress_callback(completed_count, total, result)

                except Exception as e:
                    logger.error(f"Variation {idx} failed with exception: {e}")
                    print(f"  ✗ [{completed_count+1}/{total}] Variation {idx}: FAILED - {e}")

                    # Create failed result
                    failed_result = VariationResult(
                        variation_index=idx,
                        params=param_sets[idx],
                        status='failed',
                        error=str(e)
                    )
                    results.append(failed_result)
                    completed_count += 1

        # Calculate total duration
        total_duration = time.time() - start_time

        # Rank results
        ranked_results = self.ranker.rank(results, by=self.config.rank_by)

        # Log summary
        successful = len([r for r in ranked_results if r.status == 'completed'])
        failed = len(ranked_results) - successful

        print(f"\n✅ Variation testing complete:")
        print(f"   Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"   Avg per combination: {total_duration/total:.1f}s")
        print(f"   Successful: {successful}/{total}")
        if failed > 0:
            print(f"   Failed: {failed}")

        return ranked_results

    def _dict_to_variation_result(self, data: Dict[str, Any]) -> VariationResult:
        """Convert dictionary from worker back to VariationResult object"""
        result = VariationResult(
            variation_index=data['variation_index'],
            params=data['params'],
            status=data['status'],
            error=data.get('error')
        )

        result.signal_count = data.get('signal_count', 0)
        result.win_count = data.get('win_count', 0)
        result.loss_count = data.get('loss_count', 0)
        result.total_pips = data.get('total_pips', 0.0)
        result.win_rate = data.get('win_rate', 0.0)
        result.profit_factor = data.get('profit_factor', 0.0)
        result.avg_win_pips = data.get('avg_win_pips', 0.0)
        result.avg_loss_pips = data.get('avg_loss_pips', 0.0)
        result.max_drawdown_pips = data.get('max_drawdown_pips', 0.0)
        result.duration_seconds = data.get('duration_seconds', 0.0)

        # Calculate derived metrics
        result.calculate_derived_metrics()

        return result

    def format_results(self, results: List[VariationResult]) -> str:
        """Format results for display"""
        return self.ranker.format_table(
            results,
            top_n=self.config.top_n,
            param_names=list(results[0].params.keys()) if results else None
        )

    def export_results(self, results: List[VariationResult], filepath: str):
        """Export results to CSV"""
        self.ranker.export_csv(results, filepath)


def run_parallel_variations(
    epic: str,
    days: int,
    param_sets: List[Dict[str, Any]],
    strategy: str = 'SMC_SIMPLE',
    max_workers: int = 4,
    rank_by: str = 'composite_score',
    top_n: int = 10,
    use_historical_intelligence: bool = False,
    csv_export: Optional[str] = None
) -> List[VariationResult]:
    """
    Convenience function to run parallel parameter variation testing.

    Args:
        epic: Currency pair to test
        days: Number of days to backtest
        param_sets: List of parameter dictionaries to test
        strategy: Strategy name
        max_workers: Number of parallel workers
        rank_by: Metric to rank results by
        top_n: Number of top results to display
        use_historical_intelligence: Use stored market intelligence
        csv_export: Optional path to export results CSV

    Returns:
        List of ranked VariationResult objects
    """
    config = VariationRunConfig(
        epic=epic,
        days=days,
        strategy=strategy,
        max_workers=max_workers,
        rank_by=rank_by,
        top_n=top_n,
        use_historical_intelligence=use_historical_intelligence
    )

    runner = ParallelVariationRunner(config)
    results = runner.run_variations(param_sets)

    # Display results
    print("\n" + "=" * 70)
    print(f"🔬 Parameter Variation Results - {epic} ({days} days)")
    print("=" * 70)
    print(runner.format_results(results))
    print("=" * 70)

    # Show best parameters
    if results and results[0].status == 'completed':
        best = results[0]
        print(f"\n🏆 Best parameters: {best.params}")
        print(f"    Win rate: {best.win_rate:.1f}%")
        print(f"    Total pips: {best.total_pips:+.1f}")
        print(f"    Score: {best.composite_score:.3f}")

    # Export if requested
    if csv_export:
        runner.export_results(results, csv_export)
        print(f"\n📤 Results exported to: {csv_export}")

    return results
