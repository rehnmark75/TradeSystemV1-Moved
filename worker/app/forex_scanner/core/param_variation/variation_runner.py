"""
Parallel Variation Runner

Executes multiple parameter variations in parallel for the same epic.
Uses ThreadPoolExecutor for parallel execution with shared data cache.
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class ParallelVariationRunner:
    """
    Execute multiple parameter variations in parallel for the same epic.

    Key optimization: Pre-loads candle data once and shares via cache
    across all parallel variations.
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

        # Lazy-loaded backtest infrastructure
        self._enhanced_backtest = None

    def _get_enhanced_backtest(self):
        """Lazy-load EnhancedBacktestCommands"""
        if self._enhanced_backtest is None:
            from forex_scanner.commands.enhanced_backtest_commands import EnhancedBacktestCommands
            self._enhanced_backtest = EnhancedBacktestCommands()
        return self._enhanced_backtest

    def run_variations(
        self,
        param_sets: List[Dict[str, Any]],
    ) -> List[VariationResult]:
        """
        Execute all parameter variations in parallel.

        Args:
            param_sets: List of parameter dictionaries to test

        Returns:
            List of VariationResult objects, sorted by rank
        """
        if not param_sets:
            logger.warning("No parameter sets to test")
            return []

        total = len(param_sets)
        logger.info(f"Starting parallel variation testing: {total} combinations")
        logger.info(f"Epic: {self.config.epic}, Days: {self.config.days}")
        logger.info(f"Workers: {self.config.max_workers}")

        # Track timing
        start_time = time.time()

        # Execute variations in parallel
        results: List[VariationResult] = []
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all variations
            future_to_index = {
                executor.submit(self._run_single_variation, params, idx): idx
                for idx, params in enumerate(param_sets)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    # Log progress
                    status_emoji = "" if result.status == 'completed' else ""
                    logger.info(
                        f"  {status_emoji} [{completed_count}/{total}] "
                        f"Variation {idx}: {result.status} "
                        f"(signals={result.signal_count}, pips={result.total_pips:+.1f})"
                    )

                    # Call progress callback
                    if self.progress_callback:
                        self.progress_callback(completed_count, total, result)

                except Exception as e:
                    logger.error(f"Variation {idx} failed with exception: {e}")
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

        logger.info(f"\nVariation testing complete:")
        logger.info(f"  Total time: {total_duration:.1f}s")
        logger.info(f"  Successful: {successful}/{total}")
        if failed > 0:
            logger.info(f"  Failed: {failed}")

        return ranked_results

    def _run_single_variation(
        self,
        params: Dict[str, Any],
        variation_index: int
    ) -> VariationResult:
        """
        Run a single backtest with the given parameter override.

        Args:
            params: Parameter override dictionary
            variation_index: Index of this variation

        Returns:
            VariationResult with backtest metrics
        """
        start_time = time.time()

        result = VariationResult(
            variation_index=variation_index,
            params=params.copy(),
            status='running'
        )

        try:
            # Get backtest command handler
            enhanced_backtest = self._get_enhanced_backtest()

            # Run backtest with parameter override
            backtest_result = enhanced_backtest.run_enhanced_backtest(
                epic=self.config.epic,
                days=self.config.days,
                start_date=self.start_date,
                end_date=self.end_date,
                strategy=self.config.strategy,
                timeframe=self.config.timeframe,
                config_override=params,
                use_historical_intelligence=self.config.use_historical_intelligence,
                pipeline=self.config.pipeline,
                show_signals=False,  # Suppress output
                return_results=True  # Get result dict
            )

            # Extract metrics from result
            if backtest_result and isinstance(backtest_result, dict):
                result = self._extract_metrics(result, backtest_result)
                result.status = 'completed'
            else:
                result.status = 'failed'
                result.error = 'Backtest returned no results'

        except Exception as e:
            result.status = 'failed'
            result.error = str(e)
            logger.debug(f"Variation {variation_index} exception: {e}")

        result.duration_seconds = time.time() - start_time
        return result

    def _extract_metrics(
        self,
        result: VariationResult,
        backtest_result: Dict[str, Any]
    ) -> VariationResult:
        """Extract metrics from backtest result dictionary"""

        # Get order logger for detailed stats if available
        order_logger = backtest_result.get('order_logger')
        if order_logger and hasattr(order_logger, 'get_summary_statistics'):
            stats = order_logger.get_summary_statistics()

            result.signal_count = stats.get('total_signals', 0)
            result.win_count = stats.get('winning_trades', 0)
            result.loss_count = stats.get('losing_trades', 0)
            result.total_pips = stats.get('total_pips', 0.0)
            result.win_rate = stats.get('win_rate', 0.0)
            result.profit_factor = stats.get('profit_factor', 0.0)
            result.avg_win_pips = stats.get('avg_win_pips', 0.0)
            result.avg_loss_pips = stats.get('avg_loss_pips', 0.0)
            result.max_drawdown_pips = stats.get('max_drawdown_pips', 0.0)

        else:
            # Fallback: extract from signals directly
            signals = backtest_result.get('all_signals', [])
            if not signals:
                # Try alternate key
                for epic_key in backtest_result:
                    if isinstance(backtest_result[epic_key], list):
                        signals = backtest_result[epic_key]
                        break

            result.signal_count = len(signals) if signals else 0

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

                result.win_count = len(wins)
                result.loss_count = len(losses)
                result.total_pips = sum(wins) + sum(losses)

                if result.signal_count > 0:
                    result.win_rate = (result.win_count / result.signal_count) * 100

                if wins:
                    result.avg_win_pips = sum(wins) / len(wins)
                if losses:
                    result.avg_loss_pips = sum(losses) / len(losses)

                # Profit factor
                total_wins = sum(wins)
                total_losses = abs(sum(losses))
                if total_losses > 0:
                    result.profit_factor = total_wins / total_losses

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
    print(f" Parameter Variation Results - {epic} ({days} days)")
    print("=" * 70)
    print(runner.format_results(results))
    print("=" * 70)

    # Show best parameters
    if results and results[0].status == 'completed':
        best = results[0]
        print(f"\n Best parameters: {best.params}")
        print(f"    Win rate: {best.win_rate:.1f}%")
        print(f"    Total pips: {best.total_pips:+.1f}")
        print(f"    Score: {best.composite_score:.3f}")

    # Export if requested
    if csv_export:
        runner.export_results(results, csv_export)
        print(f"\n Results exported to: {csv_export}")

    return results
