"""
Broker Trade Analyzer

Fetches and analyzes actual trading performance from the RoboMarkets broker API.
Provides statistics on open positions and closed trades including:
- Win/loss rates
- Profit/loss metrics
- Trade duration analysis
- Performance by ticker/side
- Equity curve and drawdown analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TradeStatistics:
    """Comprehensive trade statistics from broker data."""
    # Time period
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # Open positions
    open_positions: int = 0
    open_unrealized_pnl: float = 0.0

    # Win/Loss metrics
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # Profit metrics
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    gross_profit: float = 0.0

    # Average metrics
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_profit_pct: float = 0.0
    avg_loss_pct: float = 0.0

    # Best/Worst
    largest_win: float = 0.0
    largest_loss: float = 0.0
    largest_win_pct: float = 0.0
    largest_loss_pct: float = 0.0

    # Risk metrics
    profit_factor: float = 0.0
    risk_reward_ratio: float = 0.0
    expectancy: float = 0.0

    # Streak metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    current_streak_type: str = ""

    # Duration metrics
    avg_trade_duration_hours: float = 0.0
    avg_win_duration_hours: float = 0.0
    avg_loss_duration_hours: float = 0.0
    longest_trade_hours: float = 0.0
    shortest_trade_hours: float = 0.0

    # By side
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_profit: float = 0.0
    short_profit: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # By ticker breakdown
    by_ticker: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Daily breakdown
    by_day: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Equity curve data points
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class BrokerTradeAnalyzer:
    """
    Analyze trading performance from broker API data.

    Fetches open positions and closed trades from RoboMarkets
    and computes comprehensive statistics for dashboard display.
    """

    def __init__(self, robomarkets_client):
        """
        Initialize analyzer with RoboMarkets client.

        Args:
            robomarkets_client: Configured RoboMarketsClient instance
        """
        self.client = robomarkets_client

    async def get_open_positions_summary(self) -> Dict[str, Any]:
        """
        Get summary of current open positions.

        Returns:
            Dict with open position statistics
        """
        try:
            positions = await self.client.get_positions()

            if not positions:
                return {
                    "count": 0,
                    "total_unrealized_pnl": 0.0,
                    "positions": [],
                    "by_side": {"long": 0, "short": 0},
                    "by_ticker": {}
                }

            total_unrealized = sum(p.unrealized_pnl for p in positions)
            by_side = {"long": 0, "short": 0}
            by_ticker = defaultdict(lambda: {"count": 0, "unrealized_pnl": 0.0})

            position_list = []
            for p in positions:
                by_side[p.side] += 1
                by_ticker[p.ticker]["count"] += 1
                by_ticker[p.ticker]["unrealized_pnl"] += p.unrealized_pnl

                position_list.append({
                    "deal_id": p.deal_id,
                    "ticker": p.ticker,
                    "side": p.side,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                    "duration_hours": (
                        (datetime.utcnow() - p.opened_at).total_seconds() / 3600
                        if p.opened_at else 0
                    )
                })

            return {
                "count": len(positions),
                "total_unrealized_pnl": round(total_unrealized, 2),
                "positions": position_list,
                "by_side": dict(by_side),
                "by_ticker": dict(by_ticker)
            }

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return {"error": str(e), "count": 0, "positions": []}

    async def get_closed_trades_summary(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get summary of closed trades for a time period.

        Args:
            days: Number of days to look back

        Returns:
            Dict with closed trade statistics
        """
        try:
            history_from = datetime.utcnow() - timedelta(days=days)
            deals = await self.client.get_all_deal_history(history_from=history_from)

            # Filter to only closed deals
            closed_deals = [d for d in deals if d.get("close_time")]

            if not closed_deals:
                return {
                    "count": 0,
                    "total_profit": 0.0,
                    "trades": [],
                    "period_days": days
                }

            trade_list = []
            for d in closed_deals:
                duration_hours = 0
                if d.get("open_time") and d.get("close_time"):
                    duration_hours = (d["close_time"] - d["open_time"]).total_seconds() / 3600

                trade_list.append({
                    "deal_id": d["deal_id"],
                    "ticker": d["ticker"],
                    "side": d["side"],
                    "quantity": d["quantity"],
                    "open_price": d["open_price"],
                    "close_price": d["close_price"],
                    "profit": d["profit"],
                    "profit_pct": d["profit_pct"],
                    "open_time": d["open_time"].isoformat() if d.get("open_time") else None,
                    "close_time": d["close_time"].isoformat() if d.get("close_time") else None,
                    "duration_hours": round(duration_hours, 2)
                })

            total_profit = sum(d["profit"] for d in closed_deals)

            return {
                "count": len(closed_deals),
                "total_profit": round(total_profit, 2),
                "trades": trade_list,
                "period_days": days
            }

        except Exception as e:
            logger.error(f"Failed to get closed trades: {e}")
            return {"error": str(e), "count": 0, "trades": []}

    async def calculate_statistics(
        self,
        days: int = 30,
        include_open: bool = True
    ) -> TradeStatistics:
        """
        Calculate comprehensive trading statistics.

        Args:
            days: Number of days to analyze
            include_open: Whether to include open position stats

        Returns:
            TradeStatistics dataclass with all metrics
        """
        stats = TradeStatistics()
        stats.period_end = datetime.utcnow()
        stats.period_start = stats.period_end - timedelta(days=days)

        try:
            # Get closed trades
            history_from = stats.period_start
            deals = await self.client.get_all_deal_history(history_from=history_from)
            closed_deals = [d for d in deals if d.get("close_time")]

            # Get open positions if requested
            if include_open:
                positions = await self.client.get_positions()
                stats.open_positions = len(positions) if positions else 0
                stats.open_unrealized_pnl = sum(p.unrealized_pnl for p in positions) if positions else 0.0

            if not closed_deals:
                return stats

            # Basic counts
            stats.total_trades = len(closed_deals)

            # Separate wins and losses
            winners = [d for d in closed_deals if d["profit"] > 0]
            losers = [d for d in closed_deals if d["profit"] < 0]
            breakeven = [d for d in closed_deals if d["profit"] == 0]

            stats.winning_trades = len(winners)
            stats.losing_trades = len(losers)
            stats.breakeven_trades = len(breakeven)

            # Win rate
            if stats.total_trades > 0:
                stats.win_rate = (stats.winning_trades / stats.total_trades) * 100
                stats.loss_rate = (stats.losing_trades / stats.total_trades) * 100

            # Profit metrics
            stats.total_profit = sum(d["profit"] for d in winners) if winners else 0.0
            stats.total_loss = abs(sum(d["profit"] for d in losers)) if losers else 0.0
            stats.net_profit = sum(d["profit"] for d in closed_deals)
            stats.gross_profit = stats.total_profit - stats.total_loss

            # Averages
            if winners:
                stats.avg_profit = stats.total_profit / len(winners)
                stats.avg_profit_pct = statistics.mean(d["profit_pct"] for d in winners)
            if losers:
                stats.avg_loss = stats.total_loss / len(losers)
                stats.avg_loss_pct = abs(statistics.mean(d["profit_pct"] for d in losers))
            if closed_deals:
                stats.avg_trade = stats.net_profit / len(closed_deals)

            # Best/Worst
            if winners:
                best = max(winners, key=lambda x: x["profit"])
                stats.largest_win = best["profit"]
                stats.largest_win_pct = max(d["profit_pct"] for d in winners)
            if losers:
                worst = min(losers, key=lambda x: x["profit"])
                stats.largest_loss = abs(worst["profit"])
                stats.largest_loss_pct = abs(min(d["profit_pct"] for d in losers))

            # Risk metrics
            if stats.total_loss > 0:
                stats.profit_factor = stats.total_profit / stats.total_loss
            if stats.avg_loss > 0:
                stats.risk_reward_ratio = stats.avg_profit / stats.avg_loss

            # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
            if stats.total_trades > 0:
                win_pct = stats.winning_trades / stats.total_trades
                loss_pct = stats.losing_trades / stats.total_trades
                stats.expectancy = (win_pct * stats.avg_profit) - (loss_pct * stats.avg_loss)

            # Calculate streaks
            stats.max_consecutive_wins, stats.max_consecutive_losses, \
                stats.current_streak, stats.current_streak_type = self._calculate_streaks(closed_deals)

            # Duration metrics
            durations = []
            win_durations = []
            loss_durations = []

            for d in closed_deals:
                if d.get("open_time") and d.get("close_time"):
                    hours = (d["close_time"] - d["open_time"]).total_seconds() / 3600
                    durations.append(hours)
                    if d["profit"] > 0:
                        win_durations.append(hours)
                    elif d["profit"] < 0:
                        loss_durations.append(hours)

            if durations:
                stats.avg_trade_duration_hours = statistics.mean(durations)
                stats.longest_trade_hours = max(durations)
                stats.shortest_trade_hours = min(durations)
            if win_durations:
                stats.avg_win_duration_hours = statistics.mean(win_durations)
            if loss_durations:
                stats.avg_loss_duration_hours = statistics.mean(loss_durations)

            # By side analysis
            long_trades = [d for d in closed_deals if d["side"] == "long"]
            short_trades = [d for d in closed_deals if d["side"] == "short"]

            stats.long_trades = len(long_trades)
            stats.short_trades = len(short_trades)

            if long_trades:
                long_wins = len([d for d in long_trades if d["profit"] > 0])
                stats.long_win_rate = (long_wins / len(long_trades)) * 100
                stats.long_profit = sum(d["profit"] for d in long_trades)

            if short_trades:
                short_wins = len([d for d in short_trades if d["profit"] > 0])
                stats.short_win_rate = (short_wins / len(short_trades)) * 100
                stats.short_profit = sum(d["profit"] for d in short_trades)

            # By ticker breakdown
            stats.by_ticker = self._calculate_ticker_breakdown(closed_deals)

            # By day breakdown
            stats.by_day = self._calculate_daily_breakdown(closed_deals)

            # Equity curve and drawdown
            stats.equity_curve, stats.max_drawdown, stats.max_drawdown_pct = \
                self._calculate_equity_curve(closed_deals)

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return stats

    def _calculate_streaks(self, deals: List[Dict]) -> Tuple[int, int, int, str]:
        """Calculate win/loss streaks from deals."""
        if not deals:
            return 0, 0, 0, ""

        # Sort by close time
        sorted_deals = sorted(deals, key=lambda x: x.get("close_time") or datetime.min)

        max_wins = 0
        max_losses = 0
        current_streak = 0
        current_type = ""

        win_streak = 0
        loss_streak = 0

        for d in sorted_deals:
            if d["profit"] > 0:
                win_streak += 1
                loss_streak = 0
                max_wins = max(max_wins, win_streak)
                current_streak = win_streak
                current_type = "win"
            elif d["profit"] < 0:
                loss_streak += 1
                win_streak = 0
                max_losses = max(max_losses, loss_streak)
                current_streak = loss_streak
                current_type = "loss"
            else:
                # Breakeven doesn't affect streak
                pass

        return max_wins, max_losses, current_streak, current_type

    def _calculate_ticker_breakdown(self, deals: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by ticker."""
        breakdown = defaultdict(lambda: {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "win_rate": 0.0
        })

        for d in deals:
            ticker = d["ticker"]
            breakdown[ticker]["trades"] += 1
            breakdown[ticker]["profit"] += d["profit"]
            if d["profit"] > 0:
                breakdown[ticker]["wins"] += 1
            elif d["profit"] < 0:
                breakdown[ticker]["losses"] += 1

        # Calculate win rates
        for ticker, data in breakdown.items():
            if data["trades"] > 0:
                data["win_rate"] = (data["wins"] / data["trades"]) * 100
            data["profit"] = round(data["profit"], 2)

        return dict(breakdown)

    def _calculate_daily_breakdown(self, deals: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by day."""
        breakdown = defaultdict(lambda: {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0.0
        })

        for d in deals:
            if d.get("close_time"):
                day = d["close_time"].strftime("%Y-%m-%d")
                breakdown[day]["trades"] += 1
                breakdown[day]["profit"] += d["profit"]
                if d["profit"] > 0:
                    breakdown[day]["wins"] += 1
                elif d["profit"] < 0:
                    breakdown[day]["losses"] += 1

        # Round profits
        for day, data in breakdown.items():
            data["profit"] = round(data["profit"], 2)

        return dict(breakdown)

    def _calculate_equity_curve(
        self,
        deals: List[Dict],
        starting_equity: float = 10000.0
    ) -> Tuple[List[Tuple[datetime, float]], float, float]:
        """
        Calculate equity curve and maximum drawdown.

        Args:
            deals: List of closed deals
            starting_equity: Starting equity value

        Returns:
            Tuple of (equity_curve_points, max_drawdown, max_drawdown_pct)
        """
        if not deals:
            return [], 0.0, 0.0

        # Sort by close time
        sorted_deals = sorted(
            [d for d in deals if d.get("close_time")],
            key=lambda x: x["close_time"]
        )

        if not sorted_deals:
            return [], 0.0, 0.0

        equity = starting_equity
        peak_equity = equity
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

        curve = [(sorted_deals[0]["close_time"], equity)]

        for d in sorted_deals:
            equity += d["profit"]
            curve.append((d["close_time"], equity))

            # Track peak and drawdown
            if equity > peak_equity:
                peak_equity = equity

            drawdown = peak_equity - equity
            drawdown_pct = (drawdown / peak_equity) * 100 if peak_equity > 0 else 0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        return curve, round(max_drawdown, 2), round(max_drawdown_pct, 2)

    async def generate_report(self, days: int = 30) -> str:
        """
        Generate a text-based performance report.

        Args:
            days: Number of days to analyze

        Returns:
            Formatted text report
        """
        stats = await self.calculate_statistics(days=days)

        report = []
        report.append("=" * 60)
        report.append(f"BROKER TRADE PERFORMANCE REPORT - Last {days} Days")
        report.append("=" * 60)

        # Open positions
        report.append(f"\n{'='*20} OPEN POSITIONS {'='*20}")
        report.append(f"Count: {stats.open_positions}")
        report.append(f"Unrealized P&L: ${stats.open_unrealized_pnl:,.2f}")

        # Closed trades
        report.append(f"\n{'='*20} CLOSED TRADES {'='*20}")
        report.append(f"Total Trades: {stats.total_trades}")
        report.append(f"Wins: {stats.winning_trades} | Losses: {stats.losing_trades} | Breakeven: {stats.breakeven_trades}")
        report.append(f"Win Rate: {stats.win_rate:.1f}%")

        # Profit metrics
        report.append(f"\n{'='*20} PROFIT METRICS {'='*20}")
        report.append(f"Net Profit: ${stats.net_profit:,.2f}")
        report.append(f"Total Profit: ${stats.total_profit:,.2f}")
        report.append(f"Total Loss: ${stats.total_loss:,.2f}")
        report.append(f"Avg Win: ${stats.avg_profit:,.2f} ({stats.avg_profit_pct:.2f}%)")
        report.append(f"Avg Loss: ${stats.avg_loss:,.2f} ({stats.avg_loss_pct:.2f}%)")
        report.append(f"Largest Win: ${stats.largest_win:,.2f} ({stats.largest_win_pct:.2f}%)")
        report.append(f"Largest Loss: ${stats.largest_loss:,.2f} ({stats.largest_loss_pct:.2f}%)")

        # Risk metrics
        report.append(f"\n{'='*20} RISK METRICS {'='*20}")
        report.append(f"Profit Factor: {stats.profit_factor:.2f}")
        report.append(f"Risk/Reward Ratio: {stats.risk_reward_ratio:.2f}")
        report.append(f"Expectancy: ${stats.expectancy:,.2f}")
        report.append(f"Max Drawdown: ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.1f}%)")

        # Streaks
        report.append(f"\n{'='*20} STREAKS {'='*20}")
        report.append(f"Max Consecutive Wins: {stats.max_consecutive_wins}")
        report.append(f"Max Consecutive Losses: {stats.max_consecutive_losses}")
        report.append(f"Current Streak: {stats.current_streak} {stats.current_streak_type}s")

        # Duration
        report.append(f"\n{'='*20} TRADE DURATION {'='*20}")
        report.append(f"Avg Trade Duration: {stats.avg_trade_duration_hours:.1f} hours")
        report.append(f"Avg Win Duration: {stats.avg_win_duration_hours:.1f} hours")
        report.append(f"Avg Loss Duration: {stats.avg_loss_duration_hours:.1f} hours")

        # By side
        report.append(f"\n{'='*20} BY SIDE {'='*20}")
        report.append(f"Long Trades: {stats.long_trades} ({stats.long_win_rate:.1f}% win rate) = ${stats.long_profit:,.2f}")
        report.append(f"Short Trades: {stats.short_trades} ({stats.short_win_rate:.1f}% win rate) = ${stats.short_profit:,.2f}")

        # Top tickers
        if stats.by_ticker:
            report.append(f"\n{'='*20} TOP TICKERS {'='*20}")
            sorted_tickers = sorted(
                stats.by_ticker.items(),
                key=lambda x: x[1]["profit"],
                reverse=True
            )[:5]
            for ticker, data in sorted_tickers:
                report.append(
                    f"{ticker}: {data['trades']} trades, "
                    f"{data['win_rate']:.0f}% win, ${data['profit']:,.2f}"
                )

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# Factory function for easy instantiation
async def create_analyzer() -> BrokerTradeAnalyzer:
    """
    Create a BrokerTradeAnalyzer with configured client.

    Returns:
        Configured BrokerTradeAnalyzer instance
    """
    from stock_scanner.core.trading.robomarkets_client import RoboMarketsClient
    from stock_scanner.config import (
        ROBOMARKETS_API_KEY,
        ROBOMARKETS_ACCOUNT_ID
    )

    client = RoboMarketsClient(
        api_key=ROBOMARKETS_API_KEY,
        account_id=ROBOMARKETS_ACCOUNT_ID
    )

    return BrokerTradeAnalyzer(client)


class BrokerTradeSync:
    """
    Sync broker trades to local database for persistent storage and fast querying.
    """

    def __init__(self, db_manager, robomarkets_client):
        """
        Initialize sync service.

        Args:
            db_manager: AsyncDatabaseManager instance
            robomarkets_client: Configured RoboMarketsClient instance
        """
        self.db = db_manager
        self.client = robomarkets_client

    async def sync_positions(self) -> Dict[str, int]:
        """
        Sync open positions from broker to database.

        Returns:
            Dict with sync statistics
        """
        logger.info("Syncing open positions from broker...")

        try:
            positions = await self.client.get_positions()

            inserted = 0
            updated = 0

            for p in positions:
                # Check if position exists
                existing = await self.db.fetchrow(
                    "SELECT id FROM broker_trades WHERE deal_id = $1",
                    p.deal_id
                )

                if existing:
                    # Update existing position
                    await self.db.execute("""
                        UPDATE broker_trades SET
                            current_price = $1,
                            profit = $2,
                            updated_at = NOW()
                        WHERE deal_id = $3
                    """, p.current_price, p.unrealized_pnl, p.deal_id)
                    updated += 1
                else:
                    # Insert new position
                    await self.db.execute("""
                        INSERT INTO broker_trades (
                            deal_id, ticker, side, quantity, open_price,
                            stop_loss, take_profit, profit, status, open_time
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (deal_id) DO UPDATE SET
                            profit = EXCLUDED.profit,
                            updated_at = NOW()
                    """,
                        p.deal_id, p.ticker, p.side, p.quantity, p.entry_price,
                        p.stop_loss, p.take_profit, p.unrealized_pnl, 'open', p.opened_at
                    )
                    inserted += 1

            logger.info(f"Synced positions: {inserted} inserted, {updated} updated")
            return {"inserted": inserted, "updated": updated, "total": len(positions)}

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            raise

    async def sync_closed_trades(self, days: int = 30) -> Dict[str, int]:
        """
        Sync closed trades from broker to database.

        Args:
            days: Number of days of history to sync

        Returns:
            Dict with sync statistics
        """
        logger.info(f"Syncing closed trades from last {days} days...")

        try:
            history_from = datetime.utcnow() - timedelta(days=days)
            deals = await self.client.get_all_deal_history(history_from=history_from)

            inserted = 0
            updated = 0

            for d in deals:
                # Only process closed deals (status == 'closed' or has close_time)
                if d.get("status") != "closed" and not d.get("close_time"):
                    continue

                # Calculate duration
                duration_hours = None
                if d.get("open_time") and d.get("close_time"):
                    duration_hours = (d["close_time"] - d["open_time"]).total_seconds() / 3600

                # Upsert trade
                result = await self.db.execute("""
                    INSERT INTO broker_trades (
                        deal_id, ticker, side, quantity, open_price, close_price,
                        stop_loss, take_profit, profit, profit_pct, status,
                        open_time, close_time, duration_hours
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (deal_id) DO UPDATE SET
                        close_price = EXCLUDED.close_price,
                        profit = EXCLUDED.profit,
                        profit_pct = EXCLUDED.profit_pct,
                        status = EXCLUDED.status,
                        close_time = EXCLUDED.close_time,
                        duration_hours = EXCLUDED.duration_hours,
                        updated_at = NOW()
                """,
                    d["deal_id"], d["ticker"], d["side"], d["quantity"],
                    d["open_price"], d["close_price"], d["stop_loss"], d["take_profit"],
                    d["profit"], d["profit_pct"], "closed",
                    d["open_time"], d["close_time"], duration_hours
                )

                # Check if it was insert or update
                if "INSERT" in str(result):
                    inserted += 1
                else:
                    updated += 1

            # Mark positions that are now closed
            await self.db.execute("""
                UPDATE broker_trades
                SET status = 'closed'
                WHERE status = 'open'
                AND deal_id IN (
                    SELECT deal_id FROM broker_trades
                    WHERE close_time IS NOT NULL
                )
            """)

            logger.info(f"Synced closed trades: {inserted} inserted, {updated} updated")
            return {"inserted": inserted, "updated": updated, "total": len(deals)}

        except Exception as e:
            logger.error(f"Failed to sync closed trades: {e}")
            raise

    async def sync_all(self, days: int = 30) -> Dict[str, Any]:
        """
        Full sync of positions, closed trades, and account balance.

        Args:
            days: Number of days of history to sync

        Returns:
            Combined sync statistics
        """
        logger.info(f"Starting full broker sync (last {days} days)...")

        # Log sync start
        log_id = await self.db.fetchval("""
            INSERT INTO broker_sync_log (sync_type, started_at, status)
            VALUES ('full', NOW(), 'running')
            RETURNING id
        """)

        try:
            positions_result = await self.sync_positions()
            trades_result = await self.sync_closed_trades(days=days)
            balance_result = await self.sync_account_balance()

            total_fetched = positions_result["total"] + trades_result["total"] + 1  # +1 for balance
            total_inserted = positions_result["inserted"] + trades_result["inserted"] + 1
            total_updated = positions_result["updated"] + trades_result["updated"]

            # Log sync completion
            await self.db.execute("""
                UPDATE broker_sync_log SET
                    records_fetched = $1,
                    records_inserted = $2,
                    records_updated = $3,
                    completed_at = NOW(),
                    status = 'completed'
                WHERE id = $4
            """, total_fetched, total_inserted, total_updated, log_id)

            logger.info(f"Full sync completed: {total_fetched} fetched, {total_inserted} inserted, {total_updated} updated")

            return {
                "positions": positions_result,
                "trades": trades_result,
                "balance": balance_result,
                "total_fetched": total_fetched,
                "total_inserted": total_inserted,
                "total_updated": total_updated
            }

        except Exception as e:
            # Log sync failure
            await self.db.execute("""
                UPDATE broker_sync_log SET
                    completed_at = NOW(),
                    status = 'failed',
                    error_message = $1
                WHERE id = $2
            """, str(e), log_id)
            raise

    async def get_trades_from_db(
        self,
        days: int = 30,
        status: str = None,
        ticker: str = None
    ) -> List[Dict]:
        """
        Get trades from local database.

        Args:
            days: Number of days to look back
            status: Filter by status ('open' or 'closed')
            ticker: Filter by ticker

        Returns:
            List of trade records
        """
        query = """
            SELECT * FROM broker_trades
            WHERE open_time >= NOW() - INTERVAL '%s days'
        """ % days

        params = []
        param_idx = 1

        if status:
            query += f" AND status = ${param_idx}"
            params.append(status)
            param_idx += 1

        if ticker:
            query += f" AND ticker = ${param_idx}"
            params.append(ticker)
            param_idx += 1

        query += " ORDER BY open_time DESC"

        rows = await self.db.fetch(query, *params) if params else await self.db.fetch(query)
        return [dict(r) for r in rows]

    async def get_statistics_from_db(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate statistics from local database.

        Args:
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        # Get closed trades stats
        closed_stats = await self.db.fetchrow("""
            SELECT
                COUNT(*) as total_trades,
                COUNT(*) FILTER (WHERE profit > 0) as winning_trades,
                COUNT(*) FILTER (WHERE profit < 0) as losing_trades,
                COUNT(*) FILTER (WHERE profit = 0) as breakeven_trades,
                COALESCE(SUM(profit) FILTER (WHERE profit > 0), 0) as total_profit,
                COALESCE(ABS(SUM(profit) FILTER (WHERE profit < 0)), 0) as total_loss,
                COALESCE(SUM(profit), 0) as net_profit,
                COALESCE(AVG(profit) FILTER (WHERE profit > 0), 0) as avg_win,
                COALESCE(ABS(AVG(profit) FILTER (WHERE profit < 0)), 0) as avg_loss,
                COALESCE(MAX(profit), 0) as largest_win,
                COALESCE(ABS(MIN(profit)), 0) as largest_loss,
                COALESCE(AVG(profit_pct) FILTER (WHERE profit > 0), 0) as avg_win_pct,
                COALESCE(ABS(AVG(profit_pct) FILTER (WHERE profit < 0)), 0) as avg_loss_pct,
                COALESCE(AVG(duration_hours), 0) as avg_duration,
                COUNT(*) FILTER (WHERE side = 'long') as long_trades,
                COUNT(*) FILTER (WHERE side = 'short') as short_trades,
                COALESCE(SUM(profit) FILTER (WHERE side = 'long'), 0) as long_profit,
                COALESCE(SUM(profit) FILTER (WHERE side = 'short'), 0) as short_profit
            FROM broker_trades
            WHERE status = 'closed'
            AND close_time >= NOW() - INTERVAL '%s days'
        """ % days)

        # Get open positions stats
        open_stats = await self.db.fetchrow("""
            SELECT
                COUNT(*) as open_count,
                COALESCE(SUM(profit), 0) as unrealized_pnl
            FROM broker_trades
            WHERE status = 'open'
        """)

        # Calculate derived metrics
        total = closed_stats["total_trades"] or 0
        wins = closed_stats["winning_trades"] or 0
        losses = closed_stats["losing_trades"] or 0

        win_rate = (wins / total * 100) if total > 0 else 0
        profit_factor = (
            float(closed_stats["total_profit"]) / float(closed_stats["total_loss"])
            if closed_stats["total_loss"] and float(closed_stats["total_loss"]) > 0 else 0
        )

        # Calculate long/short win rates
        long_wins = await self.db.fetchval("""
            SELECT COUNT(*) FROM broker_trades
            WHERE status = 'closed' AND side = 'long' AND profit > 0
            AND close_time >= NOW() - INTERVAL '%s days'
        """ % days) or 0

        short_wins = await self.db.fetchval("""
            SELECT COUNT(*) FROM broker_trades
            WHERE status = 'closed' AND side = 'short' AND profit > 0
            AND close_time >= NOW() - INTERVAL '%s days'
        """ % days) or 0

        long_total = closed_stats["long_trades"] or 0
        short_total = closed_stats["short_trades"] or 0

        long_win_rate = (long_wins / long_total * 100) if long_total > 0 else 0
        short_win_rate = (short_wins / short_total * 100) if short_total > 0 else 0

        # Expectancy
        if total > 0:
            win_pct = wins / total
            loss_pct = losses / total
            expectancy = (win_pct * float(closed_stats["avg_win"])) - (loss_pct * float(closed_stats["avg_loss"]))
        else:
            expectancy = 0

        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "breakeven_trades": closed_stats["breakeven_trades"] or 0,
            "win_rate": round(win_rate, 2),
            "total_profit": float(closed_stats["total_profit"] or 0),
            "total_loss": float(closed_stats["total_loss"] or 0),
            "net_profit": float(closed_stats["net_profit"] or 0),
            "avg_win": float(closed_stats["avg_win"] or 0),
            "avg_loss": float(closed_stats["avg_loss"] or 0),
            "avg_win_pct": float(closed_stats["avg_win_pct"] or 0),
            "avg_loss_pct": float(closed_stats["avg_loss_pct"] or 0),
            "largest_win": float(closed_stats["largest_win"] or 0),
            "largest_loss": float(closed_stats["largest_loss"] or 0),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy, 2),
            "avg_duration_hours": float(closed_stats["avg_duration"] or 0),
            "long_trades": long_total,
            "short_trades": short_total,
            "long_win_rate": round(long_win_rate, 2),
            "short_win_rate": round(short_win_rate, 2),
            "long_profit": float(closed_stats["long_profit"] or 0),
            "short_profit": float(closed_stats["short_profit"] or 0),
            "open_positions": open_stats["open_count"] or 0,
            "open_unrealized_pnl": float(open_stats["unrealized_pnl"] or 0),
        }

    async def get_ticker_breakdown(self, days: int = 30) -> List[Dict]:
        """Get performance breakdown by ticker from database."""
        rows = await self.db.fetch("""
            SELECT
                ticker,
                COUNT(*) as trades,
                COUNT(*) FILTER (WHERE profit > 0) as wins,
                COUNT(*) FILTER (WHERE profit < 0) as losses,
                COALESCE(SUM(profit), 0) as profit,
                CASE WHEN COUNT(*) > 0
                    THEN ROUND((COUNT(*) FILTER (WHERE profit > 0)::numeric / COUNT(*) * 100), 1)
                    ELSE 0
                END as win_rate
            FROM broker_trades
            WHERE status = 'closed'
            AND close_time >= NOW() - INTERVAL '%s days'
            GROUP BY ticker
            ORDER BY profit DESC
        """ % days)
        return [dict(r) for r in rows]

    async def get_daily_breakdown(self, days: int = 30) -> List[Dict]:
        """Get performance breakdown by day from database."""
        rows = await self.db.fetch("""
            SELECT
                DATE(close_time) as trade_date,
                COUNT(*) as trades,
                COUNT(*) FILTER (WHERE profit > 0) as wins,
                COUNT(*) FILTER (WHERE profit < 0) as losses,
                COALESCE(SUM(profit), 0) as profit
            FROM broker_trades
            WHERE status = 'closed'
            AND close_time >= NOW() - INTERVAL '%s days'
            GROUP BY DATE(close_time)
            ORDER BY trade_date
        """ % days)
        return [dict(r) for r in rows]

    async def get_equity_curve(self, days: int = 30, starting_equity: float = 10000.0) -> List[Dict]:
        """Get equity curve data from database."""
        rows = await self.db.fetch("""
            SELECT close_time, profit
            FROM broker_trades
            WHERE status = 'closed'
            AND close_time >= NOW() - INTERVAL '%s days'
            ORDER BY close_time
        """ % days)

        equity = starting_equity
        curve = []

        for r in rows:
            equity += float(r["profit"])
            curve.append({
                "date": r["close_time"].isoformat() if r["close_time"] else None,
                "equity": round(equity, 2)
            })

        return curve

    async def sync_account_balance(self) -> Dict[str, Any]:
        """
        Sync account balance from broker to database.

        Returns:
            Dict with current balance info
        """
        logger.info("Syncing account balance from broker...")

        try:
            balance = await self.client.get_account_balance()

            # Insert new balance snapshot
            await self.db.execute("""
                INSERT INTO broker_account_balance (total_value, invested, available, recorded_at)
                VALUES ($1, $2, $3, $4)
            """, balance.total_value, balance.invested, balance.available, balance.timestamp)

            logger.info(f"Account balance synced: ${balance.total_value:,.2f}")

            return {
                "total_value": balance.total_value,
                "invested": balance.invested,
                "available": balance.available,
                "timestamp": balance.timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to sync account balance: {e}")
            raise

    async def get_account_balance_history(self, days: int = 30) -> List[Dict]:
        """
        Get account balance history from database.

        Args:
            days: Number of days of history to retrieve

        Returns:
            List of balance snapshots ordered by date
        """
        rows = await self.db.fetch("""
            SELECT total_value, invested, available, recorded_at
            FROM broker_account_balance
            WHERE recorded_at >= NOW() - INTERVAL '%s days'
            ORDER BY recorded_at
        """ % days)
        return [dict(r) for r in rows]

    async def get_latest_account_balance(self) -> Optional[Dict]:
        """
        Get the most recent account balance from database.

        Returns:
            Latest balance record or None
        """
        row = await self.db.fetchrow("""
            SELECT total_value, invested, available, recorded_at
            FROM broker_account_balance
            ORDER BY id DESC
            LIMIT 1
        """)
        return dict(row) if row else None

    async def get_account_balance_trend(self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate account balance trend over a period.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with trend information
        """
        # Get oldest and newest balances in the period
        oldest = await self.db.fetchrow("""
            SELECT total_value, recorded_at
            FROM broker_account_balance
            WHERE recorded_at >= NOW() - INTERVAL '%s days'
            ORDER BY recorded_at ASC
            LIMIT 1
        """ % days)

        newest = await self.db.fetchrow("""
            SELECT total_value, recorded_at
            FROM broker_account_balance
            ORDER BY id DESC
            LIMIT 1
        """)

        if not oldest or not newest:
            return {
                "change": 0.0,
                "change_pct": 0.0,
                "trend": "neutral",
                "start_value": 0.0,
                "end_value": 0.0,
                "period_days": days
            }

        start_value = float(oldest["total_value"])
        end_value = float(newest["total_value"])
        change = end_value - start_value
        change_pct = (change / start_value * 100) if start_value > 0 else 0

        if change > 0:
            trend = "up"
        elif change < 0:
            trend = "down"
        else:
            trend = "neutral"

        return {
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "trend": trend,
            "start_value": round(start_value, 2),
            "end_value": round(end_value, 2),
            "period_days": days
        }
