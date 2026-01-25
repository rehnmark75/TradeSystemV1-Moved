"""
Signal Performance Tracker

Tracks and analyzes the performance of generated signals:
1. Updates signal status based on price movement
2. Calculates win/loss rates by scanner and tier
3. Tracks average R-multiple achieved
4. Provides performance reports

Performance Metrics:
- Win Rate: % of signals that hit TP1
- Average Win: Average % gain on winners
- Average Loss: Average % loss on losers
- Profit Factor: Sum of wins / Sum of losses
- Expectancy: Expected value per trade
- R-Multiple: Actual risk/reward achieved
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalOutcome(Enum):
    """Signal outcome status"""
    PENDING = "pending"       # Signal active, not yet triggered
    TRIGGERED = "triggered"   # Entry hit
    PARTIAL_EXIT = "partial_exit"  # Hit TP1, holding for TP2
    WIN_TP1 = "win_tp1"       # Hit TP1
    WIN_TP2 = "win_tp2"       # Hit TP2
    LOSS = "loss"             # Hit stop loss
    EXPIRED = "expired"       # Signal validity expired
    CANCELLED = "cancelled"   # Manually cancelled


@dataclass
class PerformanceMetrics:
    """Performance metrics for a scanner or time period"""
    total_signals: int = 0
    triggered_signals: int = 0
    closed_signals: int = 0
    expired_signals: int = 0
    cancelled_signals: int = 0

    # Win/Loss stats
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    # Return stats
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_win_pct: float = 0.0
    max_loss_pct: float = 0.0

    # R-Multiple stats
    avg_r_multiple: float = 0.0
    max_r_multiple: float = 0.0
    min_r_multiple: float = 0.0

    # Derived metrics
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Time stats
    avg_hold_time_hours: float = 0.0
    avg_time_to_trigger: float = 0.0

    # Quality tier breakdown
    a_plus_signals: int = 0
    a_signals: int = 0
    b_signals: int = 0
    c_signals: int = 0
    d_signals: int = 0


class PerformanceTracker:
    """
    Track and analyze signal performance.

    Features:
    - Update signal status based on price movement
    - Calculate win/loss statistics
    - Track performance by scanner and quality tier
    - Generate performance reports
    """

    def __init__(self, db_manager):
        self.db = db_manager

    async def update_signal_statuses(self) -> Dict[str, int]:
        """
        Update status of all active signals based on current prices.

        Returns:
            Dict with counts of status changes
        """
        logger.info("Updating signal statuses...")

        results = {
            'triggered': 0,
            'win_tp1': 0,
            'win_tp2': 0,
            'loss': 0,
            'expired': 0,
            'unchanged': 0
        }

        # Get active signals
        active_signals = await self._get_active_signals()
        logger.info(f"Checking {len(active_signals)} active signals")

        for signal in active_signals:
            outcome = await self._check_signal_outcome(signal)

            if outcome == 'unchanged':
                results['unchanged'] += 1
            else:
                await self._update_signal_status(signal, outcome)
                results[outcome] = results.get(outcome, 0) + 1

        logger.info(f"Status updates: {results}")
        return results

    async def _get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active/triggered signals"""
        query = """
            SELECT s.*, m.price_change_1d, c.close as current_price
            FROM stock_scanner_signals s
            LEFT JOIN stock_screening_metrics m ON s.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            LEFT JOIN LATERAL (
                SELECT close FROM stock_candles_synthesized
                WHERE ticker = s.ticker AND timeframe = '1d'
                ORDER BY timestamp DESC LIMIT 1
            ) c ON true
            WHERE s.status IN ('active', 'triggered', 'partial_exit')
            ORDER BY s.signal_timestamp DESC
        """
        rows = await self.db.fetch(query)
        return [dict(r) for r in rows]

    async def _check_signal_outcome(self, signal: Dict[str, Any]) -> str:
        """
        Check if a signal hit any exit level.

        Returns:
            Outcome string: 'triggered', 'win_tp1', 'win_tp2', 'loss', 'expired', 'unchanged'
        """
        ticker = signal['ticker']
        signal_type = signal['signal_type']
        status = signal['status']
        entry = float(signal['entry_price'])
        stop = float(signal['stop_loss'])
        tp1 = float(signal['take_profit_1'])
        tp2 = float(signal['take_profit_2'] or tp1 * 1.5)
        signal_time = signal['signal_timestamp']

        # Get current price
        current_price = signal.get('current_price')
        if not current_price:
            return 'unchanged'
        current_price = float(current_price)

        # Check expiration (5 days for active signals)
        if status == 'active':
            # Handle timezone-aware datetimes
            now = datetime.now()
            if signal_time.tzinfo is not None:
                signal_time = signal_time.replace(tzinfo=None)
            days_active = (now - signal_time).days
            if days_active > 5:
                return 'expired'

        # Check outcomes based on signal type
        if signal_type == 'BUY':
            # Long position logic
            if current_price <= stop:
                return 'loss'
            if status == 'partial_exit' and current_price >= tp2:
                return 'win_tp2'
            if current_price >= tp1:
                return 'win_tp1' if status != 'partial_exit' else 'win_tp2'
            if status == 'active' and current_price >= entry:
                return 'triggered'
        else:  # SELL
            # Short position logic
            if current_price >= stop:
                return 'loss'
            if status == 'partial_exit' and current_price <= tp2:
                return 'win_tp2'
            if current_price <= tp1:
                return 'win_tp1' if status != 'partial_exit' else 'win_tp2'
            if status == 'active' and current_price <= entry:
                return 'triggered'

        return 'unchanged'

    async def _update_signal_status(self, signal: Dict[str, Any], outcome: str):
        """Update signal status in database"""
        signal_id = signal['id']
        current_price = signal.get('current_price', signal['entry_price'])

        # Map outcome to status
        status_map = {
            'triggered': 'triggered',
            'win_tp1': 'partial_exit',
            'win_tp2': 'closed',
            'loss': 'closed',
            'expired': 'expired'
        }
        new_status = status_map.get(outcome, 'active')

        # Calculate realized P&L
        entry = float(signal['entry_price'])
        stop = float(signal['stop_loss'])
        current = float(current_price)

        if signal['signal_type'] == 'BUY':
            pnl_pct = ((current - entry) / entry) * 100
            r_multiple = (current - entry) / (entry - stop) if entry != stop else 0
        else:
            pnl_pct = ((entry - current) / entry) * 100
            r_multiple = (entry - current) / (stop - entry) if stop != entry else 0

        # Update query - use explicit cast to VARCHAR to avoid type inference issues
        # The CASE expression uses $1::VARCHAR to ensure consistent type deduction
        query = """
            UPDATE stock_scanner_signals
            SET status = $1::VARCHAR,
                close_timestamp = CASE WHEN $1::VARCHAR IN ('closed', 'expired') THEN NOW() ELSE close_timestamp END,
                close_price = $2,
                realized_pnl_pct = $3,
                realized_r_multiple = $4,
                exit_reason = $5
            WHERE id = $6
        """

        exit_reason = f"Signal {outcome.replace('_', ' ')}"

        await self.db.execute(
            query,
            new_status,
            current,
            pnl_pct,
            r_multiple,
            exit_reason,
            signal_id
        )

        logger.info(f"Updated {signal['ticker']}: {outcome} (P&L: {pnl_pct:+.2f}%, R: {r_multiple:.2f})")

    async def get_scanner_performance(
        self,
        scanner_name: str = None,
        days: int = 30
    ) -> PerformanceMetrics:
        """
        Get performance metrics for a scanner or all scanners.

        Args:
            scanner_name: Specific scanner (None = all)
            days: Number of days to look back

        Returns:
            PerformanceMetrics dataclass
        """
        scanner_filter = ""
        if scanner_name:
            scanner_filter = f"AND scanner_name = '{scanner_name}'"

        query = f"""
            SELECT
                COUNT(*) as total_signals,
                COUNT(*) FILTER (WHERE status = 'triggered') as triggered,
                COUNT(*) FILTER (WHERE status = 'closed') as closed,
                COUNT(*) FILTER (WHERE status = 'expired') as expired,
                COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
                COUNT(*) FILTER (WHERE realized_pnl_pct > 0) as wins,
                COUNT(*) FILTER (WHERE realized_pnl_pct <= 0 AND status = 'closed') as losses,
                AVG(realized_pnl_pct) FILTER (WHERE realized_pnl_pct > 0) as avg_win,
                AVG(realized_pnl_pct) FILTER (WHERE realized_pnl_pct <= 0 AND status = 'closed') as avg_loss,
                MAX(realized_pnl_pct) as max_win,
                MIN(realized_pnl_pct) FILTER (WHERE realized_pnl_pct < 0) as max_loss,
                AVG(realized_r_multiple) as avg_r,
                MAX(realized_r_multiple) as max_r,
                MIN(realized_r_multiple) as min_r,
                AVG(EXTRACT(EPOCH FROM (close_timestamp - signal_timestamp))/3600)
                    FILTER (WHERE close_timestamp IS NOT NULL) as avg_hold_hours,
                COUNT(*) FILTER (WHERE quality_tier = 'A+') as a_plus,
                COUNT(*) FILTER (WHERE quality_tier = 'A') as a,
                COUNT(*) FILTER (WHERE quality_tier = 'B') as b,
                COUNT(*) FILTER (WHERE quality_tier = 'C') as c,
                COUNT(*) FILTER (WHERE quality_tier = 'D') as d
            FROM stock_scanner_signals
            WHERE signal_timestamp >= NOW() - INTERVAL '{days} days'
            {scanner_filter}
        """

        row = await self.db.fetchrow(query)

        if not row:
            return PerformanceMetrics()

        metrics = PerformanceMetrics(
            total_signals=row['total_signals'] or 0,
            triggered_signals=row['triggered'] or 0,
            closed_signals=row['closed'] or 0,
            expired_signals=row['expired'] or 0,
            cancelled_signals=row['cancelled'] or 0,
            wins=row['wins'] or 0,
            losses=row['losses'] or 0,
            avg_win_pct=float(row['avg_win'] or 0),
            avg_loss_pct=abs(float(row['avg_loss'] or 0)),
            max_win_pct=float(row['max_win'] or 0),
            max_loss_pct=abs(float(row['max_loss'] or 0)),
            avg_r_multiple=float(row['avg_r'] or 0),
            max_r_multiple=float(row['max_r'] or 0),
            min_r_multiple=float(row['min_r'] or 0),
            avg_hold_time_hours=float(row['avg_hold_hours'] or 0),
            a_plus_signals=row['a_plus'] or 0,
            a_signals=row['a'] or 0,
            b_signals=row['b'] or 0,
            c_signals=row['c'] or 0,
            d_signals=row['d'] or 0,
        )

        # Calculate derived metrics
        total_closed = metrics.wins + metrics.losses
        if total_closed > 0:
            metrics.win_rate = (metrics.wins / total_closed) * 100

        if metrics.losses > 0 and metrics.avg_loss_pct > 0:
            total_wins = metrics.wins * metrics.avg_win_pct
            total_losses = metrics.losses * metrics.avg_loss_pct
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        if total_closed > 0:
            win_pct = metrics.wins / total_closed
            loss_pct = metrics.losses / total_closed
            metrics.expectancy = (win_pct * metrics.avg_win_pct) - (loss_pct * metrics.avg_loss_pct)

        return metrics

    async def get_performance_by_tier(
        self,
        days: int = 30
    ) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics broken down by quality tier"""
        tiers = ['A+', 'A', 'B', 'C', 'D']
        results = {}

        for tier in tiers:
            query = f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE realized_pnl_pct > 0) as wins,
                    COUNT(*) FILTER (WHERE realized_pnl_pct <= 0 AND status = 'closed') as losses,
                    AVG(realized_pnl_pct) FILTER (WHERE realized_pnl_pct > 0) as avg_win,
                    AVG(realized_pnl_pct) FILTER (WHERE realized_pnl_pct <= 0 AND status = 'closed') as avg_loss,
                    AVG(realized_r_multiple) as avg_r
                FROM stock_scanner_signals
                WHERE signal_timestamp >= NOW() - INTERVAL '{days} days'
                AND quality_tier = $1
            """
            row = await self.db.fetchrow(query, tier)

            if row and row['total'] > 0:
                total_closed = (row['wins'] or 0) + (row['losses'] or 0)
                win_rate = ((row['wins'] or 0) / total_closed * 100) if total_closed > 0 else 0

                results[tier] = PerformanceMetrics(
                    total_signals=row['total'],
                    wins=row['wins'] or 0,
                    losses=row['losses'] or 0,
                    win_rate=win_rate,
                    avg_win_pct=float(row['avg_win'] or 0),
                    avg_loss_pct=abs(float(row['avg_loss'] or 0)),
                    avg_r_multiple=float(row['avg_r'] or 0)
                )

        return results

    async def record_daily_performance(self) -> None:
        """Record daily performance snapshot to history table"""
        today = datetime.now().date()

        # Get all scanners
        scanners_query = """
            SELECT DISTINCT scanner_name FROM stock_scanner_signals
            WHERE signal_timestamp >= NOW() - INTERVAL '30 days'
        """
        scanners = await self.db.fetch(scanners_query)

        for scanner_row in scanners:
            scanner_name = scanner_row['scanner_name']
            metrics = await self.get_scanner_performance(scanner_name, days=30)

            # Insert/update performance record
            insert_query = """
                INSERT INTO stock_scanner_performance (
                    scanner_name, evaluation_date, evaluation_period,
                    total_signals, signals_triggered, signals_closed, signals_expired,
                    win_rate, avg_win_pct, avg_loss_pct, profit_factor,
                    avg_r_multiple, max_r_multiple, min_r_multiple, expectancy,
                    a_plus_signals, a_signals, b_signals,
                    avg_hold_time_hours
                ) VALUES (
                    $1, $2, 'daily',
                    $3, $4, $5, $6,
                    $7, $8, $9, $10,
                    $11, $12, $13, $14,
                    $15, $16, $17,
                    $18
                )
                ON CONFLICT (scanner_name, evaluation_date, evaluation_period)
                DO UPDATE SET
                    total_signals = EXCLUDED.total_signals,
                    signals_triggered = EXCLUDED.signals_triggered,
                    signals_closed = EXCLUDED.signals_closed,
                    win_rate = EXCLUDED.win_rate,
                    profit_factor = EXCLUDED.profit_factor,
                    expectancy = EXCLUDED.expectancy
            """

            await self.db.execute(
                insert_query,
                scanner_name, today,
                metrics.total_signals, metrics.triggered_signals,
                metrics.closed_signals, metrics.expired_signals,
                metrics.win_rate, metrics.avg_win_pct, metrics.avg_loss_pct,
                metrics.profit_factor, metrics.avg_r_multiple,
                metrics.max_r_multiple, metrics.min_r_multiple,
                metrics.expectancy, metrics.a_plus_signals, metrics.a_signals,
                metrics.b_signals, metrics.avg_hold_time_hours
            )

        logger.info(f"Recorded daily performance for {len(scanners)} scanners")

    async def generate_performance_report(self, days: int = 30) -> str:
        """
        Generate a text performance report.

        Returns:
            Formatted performance report string
        """
        # Get overall metrics
        overall = await self.get_scanner_performance(days=days)

        # Get per-scanner metrics
        scanner_metrics = {}
        scanners_query = """
            SELECT DISTINCT scanner_name FROM stock_scanner_signals
            WHERE signal_timestamp >= NOW() - INTERVAL '%s days'
        """ % days
        scanners = await self.db.fetch(scanners_query)

        for scanner_row in scanners:
            scanner_name = scanner_row['scanner_name']
            scanner_metrics[scanner_name] = await self.get_scanner_performance(
                scanner_name, days
            )

        # Get tier breakdown
        tier_metrics = await self.get_performance_by_tier(days)

        # Build report
        report = []
        report.append("=" * 60)
        report.append(f"SIGNAL PERFORMANCE REPORT - Last {days} Days")
        report.append("=" * 60)

        # Overall stats
        report.append("\n📊 OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Signals: {overall.total_signals}")
        report.append(f"Closed Trades: {overall.closed_signals}")
        report.append(f"Win Rate: {overall.win_rate:.1f}%")
        report.append(f"Avg Win: +{overall.avg_win_pct:.2f}%")
        report.append(f"Avg Loss: -{overall.avg_loss_pct:.2f}%")
        report.append(f"Profit Factor: {overall.profit_factor:.2f}")
        report.append(f"Expectancy: {overall.expectancy:+.2f}%")
        report.append(f"Avg R-Multiple: {overall.avg_r_multiple:.2f}")

        # Scanner breakdown
        report.append("\n📈 BY SCANNER")
        report.append("-" * 40)
        for scanner, metrics in sorted(scanner_metrics.items()):
            report.append(
                f"{scanner}: {metrics.total_signals} signals, "
                f"{metrics.win_rate:.0f}% win, "
                f"PF: {metrics.profit_factor:.2f}"
            )

        # Tier breakdown
        report.append("\n🎯 BY QUALITY TIER")
        report.append("-" * 40)
        for tier, metrics in tier_metrics.items():
            if metrics.total_signals > 0:
                report.append(
                    f"{tier}: {metrics.total_signals} signals, "
                    f"{metrics.win_rate:.0f}% win, "
                    f"Avg R: {metrics.avg_r_multiple:.2f}"
                )

        report.append("\n" + "=" * 60)

        return "\n".join(report)
