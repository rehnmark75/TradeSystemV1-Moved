"""
Backtest Order Logger

Logs signals and trade outcomes to the database.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import json
import pandas as pd

from ..database.async_database_manager import AsyncDatabaseManager
from .trade_simulator import TradeResult


def to_datetime(ts: Union[datetime, pd.Timestamp, None]) -> Optional[datetime]:
    """Convert pandas Timestamp or datetime to Python datetime with UTC timezone."""
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        # Convert pandas Timestamp to Python datetime
        if ts.tzinfo is None:
            # Make timezone-aware (assume UTC)
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return ts.to_pydatetime()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    return ts


class BacktestOrderLogger:
    """
    Logs backtest signals and trade results to database.

    Features:
    - Creates execution records
    - Logs individual signals with outcomes
    - Calculates and updates execution statistics
    - Supports CSV export
    """

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)

        # Statistics tracking
        self.signals_logged = 0
        self.winners = 0
        self.losers = 0
        self.breakevens = 0
        self.total_pnl = 0.0
        self.all_signals: List[Dict] = []

    async def create_execution(
        self,
        strategy_name: str,
        start_date,
        end_date,
        tickers: List[str],
        timeframe: str = '1d',
        config: Optional[Dict] = None
    ) -> int:
        """
        Create a new backtest execution record.

        Args:
            strategy_name: Name of the strategy being tested
            start_date: Start date of backtest
            end_date: End date of backtest
            tickers: List of tickers tested
            timeframe: Timeframe used
            config: Strategy configuration parameters

        Returns:
            execution_id for the new record
        """
        execution_name = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        query = """
            INSERT INTO stock_backtest_executions
            (execution_name, strategy_name, start_date, end_date, tickers, timeframe, config_snapshot, status, started_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'running', NOW())
            RETURNING id
        """

        execution_id = await self.db.fetchval(
            query,
            execution_name,
            strategy_name,
            start_date,
            end_date,
            tickers,
            timeframe,
            json.dumps(config) if config else None
        )

        self.logger.info(f"Created backtest execution {execution_id}: {execution_name}")
        return execution_id

    async def log_signal(
        self,
        execution_id: int,
        signal,  # PullbackSignal or similar
        trade_result: Optional[TradeResult] = None,
        sector: Optional[str] = None
    ) -> int:
        """
        Log a signal and its trade result to the database.

        Args:
            execution_id: The backtest execution ID
            signal: Signal object with entry details
            trade_result: Trade simulation result (optional)
            sector: Stock sector

        Returns:
            signal_id for the new record
        """
        query = """
            INSERT INTO stock_backtest_signals
            (execution_id, ticker, signal_timestamp, signal_type,
             entry_price, stop_loss_price, take_profit_price, risk_reward_ratio,
             confidence, quality_tier,
             exit_price, exit_timestamp, exit_reason, trade_result,
             pnl_percent, holding_days,
             ema_20, ema_50, ema_100, ema_200, rsi, atr, pullback_percent,
             sector, volume, relative_volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
            RETURNING id
        """

        # Extract trade result data if available
        exit_price = trade_result.exit_price if trade_result else None
        exit_timestamp = to_datetime(trade_result.exit_timestamp) if trade_result else None
        exit_reason = trade_result.exit_reason if trade_result else None
        result_str = trade_result.trade_result if trade_result else None
        pnl_percent = trade_result.pnl_percent if trade_result else None
        holding_days = trade_result.holding_days if trade_result else None

        # Convert signal timestamp to proper datetime
        signal_ts = to_datetime(signal.signal_timestamp)

        # Extract indicator values - handle different signal types gracefully
        ema_20 = getattr(signal, 'ema_20', None) or getattr(signal, 'zlma', None)
        ema_50 = getattr(signal, 'ema_50', None) or getattr(signal, 'ema_15', None)
        ema_100 = getattr(signal, 'ema_100', None)
        ema_200 = getattr(signal, 'ema_200', None)
        rsi = getattr(signal, 'rsi', None)
        atr = getattr(signal, 'atr', None)
        pullback_percent = getattr(signal, 'pullback_percent', None) or getattr(signal, 'crossover_strength', None)
        volume = getattr(signal, 'volume', None)
        relative_volume = getattr(signal, 'relative_volume', None)

        signal_id = await self.db.fetchval(
            query,
            execution_id,
            signal.ticker,
            signal_ts,
            signal.signal_type,
            signal.entry_price,
            signal.stop_loss_price,
            signal.take_profit_price,
            signal.risk_reward_ratio,
            signal.confidence,
            signal.quality_tier,
            exit_price,
            exit_timestamp,
            exit_reason,
            result_str,
            pnl_percent,
            holding_days,
            ema_20,
            ema_50,
            ema_100,
            ema_200,
            rsi,
            atr,
            pullback_percent,
            sector or getattr(signal, 'sector', None),
            volume,
            relative_volume
        )

        # Track statistics
        self.signals_logged += 1
        if trade_result:
            if trade_result.trade_result == 'WIN':
                self.winners += 1
            elif trade_result.trade_result == 'LOSS':
                self.losers += 1
            else:
                self.breakevens += 1

            self.total_pnl += trade_result.pnl_percent

            # Store for later analysis
            self.all_signals.append({
                'signal_id': signal_id,
                'ticker': signal.ticker,
                'timestamp': signal.signal_timestamp,
                'signal_type': signal.signal_type,
                'entry_price': signal.entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'trade_result': result_str,
                'pnl_percent': pnl_percent,
                'confidence': signal.confidence,
                'quality_tier': signal.quality_tier
            })

        return signal_id

    async def finalize_execution(
        self,
        execution_id: int,
        status: str = 'completed',
        error_message: Optional[str] = None
    ):
        """
        Finalize execution record with statistics.

        Args:
            execution_id: The execution to finalize
            status: Final status ('completed', 'failed')
            error_message: Optional error message if failed
        """
        # Calculate statistics from logged signals
        total_trades = self.winners + self.losers + self.breakevens
        win_rate = (self.winners / total_trades * 100) if total_trades > 0 else 0

        # Calculate profit factor
        win_pnls = [s['pnl_percent'] for s in self.all_signals if s.get('trade_result') == 'WIN']
        loss_pnls = [abs(s['pnl_percent']) for s in self.all_signals if s.get('trade_result') == 'LOSS']

        total_wins = sum(win_pnls) if win_pnls else 0
        total_losses = sum(loss_pnls) if loss_pnls else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else None

        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0

        # Calculate max drawdown (simplified)
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        for s in sorted(self.all_signals, key=lambda x: x.get('timestamp', datetime.min)):
            if s.get('pnl_percent'):
                cumulative_pnl += s['pnl_percent']
                peak = max(peak, cumulative_pnl)
                drawdown = peak - cumulative_pnl
                max_drawdown = max(max_drawdown, drawdown)

        query = """
            UPDATE stock_backtest_executions
            SET status = $1,
                completed_at = NOW(),
                duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INTEGER,
                total_signals = $2,
                total_trades = $3,
                winners = $4,
                losers = $5,
                win_rate = $6,
                total_pnl_percent = $7,
                avg_win_percent = $8,
                avg_loss_percent = $9,
                profit_factor = $10,
                max_drawdown_percent = $11,
                error_message = $12
            WHERE id = $13
        """

        await self.db.execute(
            query,
            status,
            self.signals_logged,
            total_trades,
            self.winners,
            self.losers,
            round(win_rate, 2),
            round(self.total_pnl, 4),
            round(avg_win, 4) if avg_win else None,
            round(avg_loss, 4) if avg_loss else None,
            round(profit_factor, 4) if profit_factor else None,
            round(max_drawdown, 4),
            error_message,
            execution_id
        )

        self.logger.info(
            f"Finalized execution {execution_id}: {total_trades} trades, "
            f"{win_rate:.1f}% win rate, {self.total_pnl:.2f}% P&L"
        )

    async def export_to_csv(self, execution_id: int, filepath: str) -> bool:
        """
        Export execution signals to CSV file.

        Args:
            execution_id: Execution to export
            filepath: Path to output CSV file

        Returns:
            True if successful
        """
        import csv

        query = """
            SELECT
                s.ticker, s.signal_timestamp, s.signal_type,
                s.entry_price, s.stop_loss_price, s.take_profit_price,
                s.exit_price, s.exit_timestamp, s.exit_reason,
                s.trade_result, s.pnl_percent, s.holding_days,
                s.confidence, s.quality_tier,
                s.ema_20, s.ema_50, s.ema_100, s.ema_200,
                s.rsi, s.atr, s.pullback_percent,
                s.sector, s.relative_volume
            FROM stock_backtest_signals s
            WHERE s.execution_id = $1
            ORDER BY s.signal_timestamp
        """

        rows = await self.db.fetch(query, execution_id)

        if not rows:
            self.logger.warning(f"No signals found for execution {execution_id}")
            return False

        # Write to CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Ticker', 'Timestamp', 'Type', 'Entry', 'StopLoss', 'TakeProfit',
                'Exit', 'ExitTime', 'ExitReason', 'Result', 'PnL%', 'HoldingDays',
                'Confidence', 'Quality', 'EMA20', 'EMA50', 'EMA100', 'EMA200',
                'RSI', 'ATR', 'Pullback%', 'Sector', 'RelVolume'
            ])

            for row in rows:
                writer.writerow([
                    row['ticker'],
                    row['signal_timestamp'].strftime('%Y-%m-%d') if row['signal_timestamp'] else '',
                    row['signal_type'],
                    f"{row['entry_price']:.4f}" if row['entry_price'] else '',
                    f"{row['stop_loss_price']:.4f}" if row['stop_loss_price'] else '',
                    f"{row['take_profit_price']:.4f}" if row['take_profit_price'] else '',
                    f"{row['exit_price']:.4f}" if row['exit_price'] else '',
                    row['exit_timestamp'].strftime('%Y-%m-%d') if row['exit_timestamp'] else '',
                    row['exit_reason'] or '',
                    row['trade_result'] or '',
                    f"{row['pnl_percent']:.2f}" if row['pnl_percent'] else '',
                    row['holding_days'] or '',
                    f"{row['confidence']:.3f}" if row['confidence'] else '',
                    row['quality_tier'] or '',
                    f"{row['ema_20']:.2f}" if row['ema_20'] else '',
                    f"{row['ema_50']:.2f}" if row['ema_50'] else '',
                    f"{row['ema_100']:.2f}" if row['ema_100'] else '',
                    f"{row['ema_200']:.2f}" if row['ema_200'] else '',
                    f"{row['rsi']:.1f}" if row['rsi'] else '',
                    f"{row['atr']:.4f}" if row['atr'] else '',
                    f"{row['pullback_percent']:.2f}" if row['pullback_percent'] else '',
                    row['sector'] or '',
                    f"{row['relative_volume']:.2f}" if row['relative_volume'] else ''
                ])

        self.logger.info(f"Exported {len(rows)} signals to {filepath}")
        return True

    async def get_execution_summary(self, execution_id: int) -> Optional[Dict]:
        """Get summary of an execution."""
        query = """
            SELECT * FROM stock_backtest_executions WHERE id = $1
        """
        row = await self.db.fetchrow(query, execution_id)
        return dict(row) if row else None

    def reset_stats(self):
        """Reset internal statistics for new execution."""
        self.signals_logged = 0
        self.winners = 0
        self.losers = 0
        self.breakevens = 0
        self.total_pnl = 0.0
        self.all_signals = []
