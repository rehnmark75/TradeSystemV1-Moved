"""
ResultAggregator - Combines chunk results into unified metrics

Aggregates signals from multiple backtest chunks, calculates
cross-chunk metrics like drawdown and consecutive losses,
and stores final results.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


@dataclass
class AggregatedMetrics:
    """Aggregated metrics from all chunks"""
    # Signal counts
    total_signals: int = 0
    bull_signals: int = 0
    bear_signals: int = 0

    # Win/Loss
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0

    # Pips
    total_pips: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    max_win_pips: float = 0.0
    max_loss_pips: float = 0.0

    # Risk metrics
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_drawdown_pips: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Time analysis
    avg_trade_duration_minutes: float = 0.0
    signals_per_day: float = 0.0

    # Chunks info
    chunks_processed: int = 0
    chunks_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_signals': self.total_signals,
            'bull_signals': self.bull_signals,
            'bear_signals': self.bear_signals,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'breakeven_trades': self.breakeven_trades,
            'win_rate': round(self.win_rate, 4),
            'total_pips': round(self.total_pips, 2),
            'avg_win_pips': round(self.avg_win_pips, 2),
            'avg_loss_pips': round(self.avg_loss_pips, 2),
            'max_win_pips': round(self.max_win_pips, 2),
            'max_loss_pips': round(self.max_loss_pips, 2),
            'profit_factor': round(self.profit_factor, 3),
            'expectancy': round(self.expectancy, 2),
            'max_drawdown_pips': round(self.max_drawdown_pips, 2),
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'avg_trade_duration_minutes': round(self.avg_trade_duration_minutes, 1),
            'signals_per_day': round(self.signals_per_day, 2),
            'chunks_processed': self.chunks_processed,
            'chunks_failed': self.chunks_failed,
        }


@dataclass
class SignalRecord:
    """Normalized signal record for aggregation"""
    timestamp: datetime
    signal_type: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    pips_gained: float
    result: str  # 'win', 'loss', 'breakeven'
    duration_minutes: Optional[float] = None
    chunk_index: int = 0
    execution_id: int = 0


class ResultAggregator:
    """
    Aggregates backtest results from multiple chunks

    Handles:
    - Fetching signals from chunk executions
    - Sorting signals chronologically across chunks
    - Calculating metrics that require sequential data (drawdown, consecutive)
    - Storing final aggregated results
    """

    def __init__(self,
                 db_manager: Optional[DatabaseManager] = None,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.logger = logger or logging.getLogger(__name__)

    def aggregate_parallel_run(self, parallel_run_id: int) -> AggregatedMetrics:
        """
        Aggregate results from all chunks of a parallel run

        Args:
            parallel_run_id: ID of the parallel run

        Returns:
            AggregatedMetrics with combined results
        """
        self.logger.info(f"Aggregating results for parallel run {parallel_run_id}")

        # 1. Get chunk execution IDs
        chunk_execution_ids = self._get_chunk_executions(parallel_run_id)

        if not chunk_execution_ids:
            self.logger.warning(f"No chunk executions found for run {parallel_run_id}")
            return AggregatedMetrics()

        # 2. Fetch all signals from all chunks
        signals = self._fetch_chunk_signals(chunk_execution_ids)

        if not signals:
            self.logger.warning(f"No signals found for run {parallel_run_id}")
            return AggregatedMetrics(chunks_processed=len(chunk_execution_ids))

        # 3. Sort signals by timestamp (crucial for cross-chunk metrics)
        signals.sort(key=lambda s: s.timestamp)

        # 4. Calculate aggregated metrics
        metrics = self._calculate_metrics(signals)

        # 5. Get run metadata for additional calculations
        run_info = self._get_run_info(parallel_run_id)
        if run_info:
            total_days = (run_info['full_end_date'] - run_info['full_start_date']).days
            if total_days > 0:
                metrics.signals_per_day = metrics.total_signals / total_days

        metrics.chunks_processed = len(chunk_execution_ids)

        # 6. Store aggregated results
        self._store_aggregated_results(parallel_run_id, metrics)

        self.logger.info(
            f"Aggregation complete: {metrics.total_signals} signals, "
            f"{metrics.win_rate:.1%} win rate, {metrics.total_pips:+.1f} pips"
        )

        return metrics

    def _get_chunk_executions(self, parallel_run_id: int) -> List[int]:
        """Get execution IDs for all chunks in a parallel run"""
        query = """
        SELECT chunk_execution_ids
        FROM backtest_parallel_runs
        WHERE id = :run_id
        """

        try:
            result = self.db_manager.execute_query(query, {'run_id': parallel_run_id})
            if result.empty:
                return []

            execution_ids = result.iloc[0]['chunk_execution_ids']
            return execution_ids if execution_ids else []

        except Exception as e:
            self.logger.error(f"Failed to get chunk executions: {e}")
            return []

    def _fetch_chunk_signals(self, execution_ids: List[int]) -> List[SignalRecord]:
        """
        Fetch all signals from chunk executions

        Uses the existing backtest_signals table structure
        """
        if not execution_ids:
            return []

        query = """
        SELECT
            signal_timestamp,
            signal_type,
            entry_price,
            exit_price,
            stop_loss_price,
            take_profit_price,
            pips_gained,
            trade_result,
            holding_time_minutes,
            execution_id
        FROM backtest_signals
        WHERE execution_id = ANY(:exec_ids)
        ORDER BY signal_timestamp
        """

        try:
            result = self.db_manager.execute_query(query, {'exec_ids': execution_ids})

            signals = []
            for _, row in result.iterrows():
                signal = SignalRecord(
                    timestamp=row['signal_timestamp'],
                    signal_type=row['signal_type'],
                    entry_price=float(row['entry_price']) if row['entry_price'] else 0.0,
                    exit_price=float(row['exit_price']) if row['exit_price'] else None,
                    stop_loss=float(row['stop_loss_price']) if row['stop_loss_price'] else 0.0,
                    take_profit=float(row['take_profit_price']) if row['take_profit_price'] else 0.0,
                    pips_gained=float(row['pips_gained']) if row['pips_gained'] else 0.0,
                    result=row['trade_result'] or 'unknown',
                    duration_minutes=float(row['holding_time_minutes']) if row['holding_time_minutes'] else None,
                    execution_id=row['execution_id']
                )
                signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Failed to fetch chunk signals: {e}")
            return []

    def _calculate_metrics(self, signals: List[SignalRecord]) -> AggregatedMetrics:
        """
        Calculate aggregated metrics from sorted signals

        Args:
            signals: List of SignalRecord sorted by timestamp

        Returns:
            AggregatedMetrics
        """
        metrics = AggregatedMetrics()

        if not signals:
            return metrics

        metrics.total_signals = len(signals)

        # Count by type
        metrics.bull_signals = sum(1 for s in signals if s.signal_type in ('BUY', 'BULL'))
        metrics.bear_signals = sum(1 for s in signals if s.signal_type in ('SELL', 'BEAR'))

        # Win/Loss counts and pips
        wins = []
        losses = []
        durations = []

        for signal in signals:
            pips = signal.pips_gained

            if signal.result == 'win' or pips > 0:
                metrics.winning_trades += 1
                wins.append(pips)
            elif signal.result == 'loss' or pips < 0:
                metrics.losing_trades += 1
                losses.append(abs(pips))
            else:
                metrics.breakeven_trades += 1

            if signal.duration_minutes:
                durations.append(signal.duration_minutes)

        # Win rate
        total_closed = metrics.winning_trades + metrics.losing_trades
        if total_closed > 0:
            metrics.win_rate = metrics.winning_trades / total_closed

        # Pips statistics
        metrics.total_pips = sum(s.pips_gained for s in signals)

        if wins:
            metrics.avg_win_pips = sum(wins) / len(wins)
            metrics.max_win_pips = max(wins)

        if losses:
            metrics.avg_loss_pips = sum(losses) / len(losses)
            metrics.max_loss_pips = max(losses)

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            metrics.profit_factor = float('inf')

        # Expectancy
        if metrics.total_signals > 0:
            metrics.expectancy = metrics.total_pips / metrics.total_signals

        # Duration
        if durations:
            metrics.avg_trade_duration_minutes = sum(durations) / len(durations)

        # Sequential metrics (require sorted signals)
        metrics.max_drawdown_pips = self._calculate_max_drawdown(signals)
        metrics.max_consecutive_wins, metrics.max_consecutive_losses = \
            self._calculate_consecutive_streaks(signals)

        return metrics

    def _calculate_max_drawdown(self, signals: List[SignalRecord]) -> float:
        """
        Calculate maximum drawdown from sequential signals

        Drawdown = peak equity - current equity (measured in pips)
        """
        if not signals:
            return 0.0

        equity = 0.0
        peak_equity = 0.0
        max_drawdown = 0.0

        for signal in signals:
            equity += signal.pips_gained
            peak_equity = max(peak_equity, equity)
            drawdown = peak_equity - equity
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_consecutive_streaks(self, signals: List[SignalRecord]) -> Tuple[int, int]:
        """
        Calculate maximum consecutive wins and losses

        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses)
        """
        if not signals:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for signal in signals:
            if signal.result == 'win' or signal.pips_gained > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif signal.result == 'loss' or signal.pips_gained < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                # Breakeven - reset both
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    def _get_run_info(self, parallel_run_id: int) -> Optional[Dict[str, Any]]:
        """Get parallel run metadata"""
        query = """
        SELECT full_start_date, full_end_date, epic, strategy
        FROM backtest_parallel_runs
        WHERE id = :run_id
        """

        try:
            result = self.db_manager.execute_query(query, {'run_id': parallel_run_id})
            if result.empty:
                return None

            row = result.iloc[0]
            return {
                'full_start_date': row['full_start_date'],
                'full_end_date': row['full_end_date'],
                'epic': row['epic'],
                'strategy': row['strategy']
            }

        except Exception as e:
            self.logger.error(f"Failed to get run info: {e}")
            return None

    def _store_aggregated_results(self, parallel_run_id: int, metrics: AggregatedMetrics):
        """Store aggregated results in the parallel run record"""
        query = """
        SELECT finalize_parallel_run(:run_id, :results)
        """

        try:
            params = {
                'run_id': parallel_run_id,
                'results': json.dumps(metrics.to_dict())
            }
            self.db_manager.execute_query(query, params)

            self.logger.info(f"Stored aggregated results for run {parallel_run_id}")

        except Exception as e:
            self.logger.error(f"Failed to store aggregated results: {e}")

    def get_signals_for_chart(self, parallel_run_id: int) -> List[Dict[str, Any]]:
        """
        Get signals in format suitable for chart generation

        Args:
            parallel_run_id: ID of the parallel run

        Returns:
            List of signal dicts with timestamp, type, prices, pips
        """
        execution_ids = self._get_chunk_executions(parallel_run_id)

        if not execution_ids:
            return []

        query = """
        SELECT
            signal_timestamp as timestamp,
            signal_type as type,
            entry_price,
            exit_price,
            pips_gained as pips,
            trade_result as result
        FROM backtest_signals
        WHERE execution_id = ANY(:exec_ids)
        ORDER BY signal_timestamp
        """

        try:
            result = self.db_manager.execute_query(query, {'exec_ids': execution_ids})

            signals = []
            for _, row in result.iterrows():
                signals.append({
                    'timestamp': row['timestamp'],
                    'type': row['type'],
                    'entry_price': float(row['entry_price']) if row['entry_price'] else 0.0,
                    'exit_price': float(row['exit_price']) if row['exit_price'] else None,
                    'pips': float(row['pips']) if row['pips'] else 0.0,
                    'result': row['result']
                })

            return signals

        except Exception as e:
            self.logger.error(f"Failed to get signals for chart: {e}")
            return []


# Factory function
def create_result_aggregator(db_manager: Optional[DatabaseManager] = None,
                             logger: Optional[logging.Logger] = None) -> ResultAggregator:
    """Create ResultAggregator instance"""
    return ResultAggregator(db_manager, logger)
