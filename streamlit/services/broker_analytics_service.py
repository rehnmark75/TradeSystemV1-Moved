"""
Broker Analytics Service for Streamlit

Provides access to broker trading data stored in the PostgreSQL database.
Data is synced from RoboMarkets API via the stock_scanner broker-sync command.

Note: This reads from the local database, not directly from the API.
Run `docker exec task-worker python -m stock_scanner.main broker-sync` to sync data.
"""

import streamlit as st
import psycopg2
import psycopg2.extras
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import os

logger = logging.getLogger(__name__)


@dataclass
class BrokerTradeStats:
    """Trading statistics from broker data."""
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Open positions
    open_positions: int = 0
    open_unrealized_pnl: float = 0.0

    # Win/Loss metrics
    win_rate: float = 0.0

    # Profit metrics
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0

    # Averages
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    avg_profit_pct: float = 0.0
    avg_loss_pct: float = 0.0

    # Best/Worst
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk metrics
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Streak metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Duration metrics
    avg_trade_duration_hours: float = 0.0

    # By side
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_profit: float = 0.0
    short_profit: float = 0.0

    # Breakdowns
    by_ticker: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_day: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Equity curve
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    # Data freshness
    last_sync: Optional[datetime] = None

    # Error handling
    error: Optional[str] = None


class BrokerAnalyticsService:
    """
    Service for fetching broker trading data from the database for Streamlit dashboards.

    Data is stored in the 'stocks' database and synced from RoboMarkets API
    via the stock_scanner CLI's broker-sync command.
    """

    def __init__(self):
        self.db_host = os.getenv("POSTGRES_HOST", "postgres")
        self.db_port = os.getenv("POSTGRES_PORT", "5432")
        self.db_user = os.getenv("POSTGRES_USER", "postgres")
        self.db_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.db_name = "stocks"

    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pass,
            database=self.db_name
        )

    @property
    def is_configured(self) -> bool:
        """Check if database connection is available."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def _check_table_exists(self) -> bool:
        """Check if broker_trades table exists."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = 'broker_trades'
                        )
                    """)
                    return cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to check table: {e}")
            return False

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the last successful sync time."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT completed_at FROM broker_sync_log
                        WHERE status = 'completed'
                        ORDER BY completed_at DESC LIMIT 1
                    """)
                    row = cur.fetchone()
                    return row[0] if row else None
        except Exception:
            return None

    @st.cache_data(ttl=60)
    def get_open_positions(_self) -> Dict[str, Any]:
        """
        Get current open positions from database.

        Returns:
            Dict with position data and summary
        """
        if not _self._check_table_exists():
            return {
                "error": "Broker data not synced. Run: docker exec task-worker python -m stock_scanner.main broker-sync",
                "positions": [],
                "count": 0
            }

        try:
            with _self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            deal_id, ticker, side, quantity, open_price,
                            current_price, profit, stop_loss, take_profit,
                            open_time, updated_at
                        FROM broker_trades
                        WHERE status = 'open'
                        ORDER BY open_time DESC
                    """)
                    rows = cur.fetchall()

            positions = []
            total_unrealized = 0.0
            by_side = {"long": 0, "short": 0}
            by_ticker = defaultdict(lambda: {"count": 0, "unrealized_pnl": 0.0})

            for row in rows:
                unrealized = float(row["profit"] or 0)
                ticker = row["ticker"]
                side = row["side"]

                position = {
                    "deal_id": row["deal_id"],
                    "ticker": ticker,
                    "side": side,
                    "quantity": float(row["quantity"] or 0),
                    "entry_price": float(row["open_price"] or 0),
                    "current_price": float(row["current_price"] or 0),
                    "unrealized_pnl": unrealized,
                    "stop_loss": float(row["stop_loss"]) if row["stop_loss"] else None,
                    "take_profit": float(row["take_profit"]) if row["take_profit"] else None,
                    "opened_at": row["open_time"].isoformat() if row["open_time"] else None,
                }

                # Calculate profit percentage
                if position["entry_price"] and position["entry_price"] > 0 and position["current_price"]:
                    if side == "long":
                        position["profit_pct"] = (
                            (position["current_price"] - position["entry_price"]) /
                            position["entry_price"]
                        ) * 100
                    else:
                        position["profit_pct"] = (
                            (position["entry_price"] - position["current_price"]) /
                            position["entry_price"]
                        ) * 100
                else:
                    position["profit_pct"] = 0.0

                positions.append(position)
                total_unrealized += unrealized
                by_side[side] += 1
                by_ticker[ticker]["count"] += 1
                by_ticker[ticker]["unrealized_pnl"] += unrealized

            return {
                "positions": positions,
                "count": len(positions),
                "total_unrealized_pnl": round(total_unrealized, 2),
                "by_side": dict(by_side),
                "by_ticker": dict(by_ticker),
                "fetched_at": datetime.utcnow().isoformat(),
                "last_sync": _self.get_last_sync_time()
            }

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return {"error": str(e), "positions": [], "count": 0}

    @st.cache_data(ttl=60)
    def get_closed_trades(_self, days: int = 30) -> Dict[str, Any]:
        """
        Get closed trades from database.

        Args:
            days: Number of days of history to fetch

        Returns:
            Dict with trade data and summary
        """
        if not _self._check_table_exists():
            return {
                "error": "Broker data not synced. Run: docker exec task-worker python -m stock_scanner.main broker-sync",
                "trades": [],
                "count": 0
            }

        try:
            history_from = datetime.utcnow() - timedelta(days=days)

            with _self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            deal_id, ticker, side, quantity, open_price, close_price,
                            profit, profit_pct, open_time, close_time, duration_hours
                        FROM broker_trades
                        WHERE status = 'closed'
                        AND close_time >= %s
                        ORDER BY close_time DESC
                    """, (history_from,))
                    rows = cur.fetchall()

            closed_trades = []
            for row in rows:
                trade = {
                    "deal_id": row["deal_id"],
                    "ticker": row["ticker"],
                    "side": row["side"],
                    "quantity": float(row["quantity"] or 0),
                    "open_price": float(row["open_price"] or 0),
                    "close_price": float(row["close_price"] or 0),
                    "profit": float(row["profit"] or 0),
                    "profit_pct": float(row["profit_pct"] or 0),
                    "open_time": row["open_time"].isoformat() if row["open_time"] else None,
                    "close_time": row["close_time"].isoformat() if row["close_time"] else None,
                    "duration_hours": float(row["duration_hours"] or 0),
                }
                closed_trades.append(trade)

            total_profit = sum(t["profit"] for t in closed_trades)

            return {
                "trades": closed_trades,
                "count": len(closed_trades),
                "total_profit": round(total_profit, 2),
                "period_days": days,
                "fetched_at": datetime.utcnow().isoformat(),
                "last_sync": _self.get_last_sync_time()
            }

        except Exception as e:
            logger.error(f"Failed to get closed trades: {e}")
            return {"error": str(e), "trades": [], "count": 0}

    def calculate_statistics(_self, days: int = 30) -> BrokerTradeStats:
        """
        Calculate comprehensive trading statistics from database.

        Args:
            days: Number of days to analyze

        Returns:
            BrokerTradeStats dataclass with all metrics
        """
        stats = BrokerTradeStats()
        stats.period_end = datetime.utcnow()
        stats.period_start = stats.period_end - timedelta(days=days)
        stats.last_sync = _self.get_last_sync_time()

        if not _self._check_table_exists():
            stats.error = "Broker data not synced. Run: docker exec task-worker python -m stock_scanner.main broker-sync"
            return stats

        try:
            # Get positions and trades
            positions_data = _self.get_open_positions()
            trades_data = _self.get_closed_trades(days=days)

            if positions_data.get("error"):
                stats.error = positions_data["error"]
                return stats

            if trades_data.get("error"):
                stats.error = trades_data["error"]
                return stats

            # Open positions
            stats.open_positions = positions_data["count"]
            stats.open_unrealized_pnl = positions_data["total_unrealized_pnl"]

            closed_trades = trades_data["trades"]
            if not closed_trades:
                return stats

            # Basic counts
            stats.total_trades = len(closed_trades)
            winners = [t for t in closed_trades if t["profit"] > 0]
            losers = [t for t in closed_trades if t["profit"] < 0]

            stats.winning_trades = len(winners)
            stats.losing_trades = len(losers)

            # Win rate
            if stats.total_trades > 0:
                stats.win_rate = (stats.winning_trades / stats.total_trades) * 100

            # Profit metrics
            stats.total_profit = sum(t["profit"] for t in winners) if winners else 0.0
            stats.total_loss = abs(sum(t["profit"] for t in losers)) if losers else 0.0
            stats.net_profit = sum(t["profit"] for t in closed_trades)

            # Averages
            if winners:
                stats.avg_profit = stats.total_profit / len(winners)
                stats.avg_profit_pct = statistics.mean(t["profit_pct"] for t in winners)
            if losers:
                stats.avg_loss = stats.total_loss / len(losers)
                stats.avg_loss_pct = abs(statistics.mean(t["profit_pct"] for t in losers))

            # Best/Worst
            if winners:
                stats.largest_win = max(t["profit"] for t in winners)
            if losers:
                stats.largest_loss = abs(min(t["profit"] for t in losers))

            # Risk metrics
            if stats.total_loss > 0:
                stats.profit_factor = stats.total_profit / stats.total_loss
            if stats.avg_loss > 0 and stats.total_trades > 0:
                win_pct = stats.winning_trades / stats.total_trades
                loss_pct = stats.losing_trades / stats.total_trades
                stats.expectancy = (win_pct * stats.avg_profit) - (loss_pct * stats.avg_loss)

            # Streaks
            stats.max_consecutive_wins, stats.max_consecutive_losses = _self._calculate_streaks(closed_trades)

            # Duration
            durations = [t["duration_hours"] for t in closed_trades if t.get("duration_hours")]
            if durations:
                stats.avg_trade_duration_hours = statistics.mean(durations)

            # By side
            long_trades = [t for t in closed_trades if t["side"] == "long"]
            short_trades = [t for t in closed_trades if t["side"] == "short"]

            stats.long_trades = len(long_trades)
            stats.short_trades = len(short_trades)

            if long_trades:
                long_wins = len([t for t in long_trades if t["profit"] > 0])
                stats.long_win_rate = (long_wins / len(long_trades)) * 100
                stats.long_profit = sum(t["profit"] for t in long_trades)

            if short_trades:
                short_wins = len([t for t in short_trades if t["profit"] > 0])
                stats.short_win_rate = (short_wins / len(short_trades)) * 100
                stats.short_profit = sum(t["profit"] for t in short_trades)

            # By ticker
            stats.by_ticker = _self._calculate_ticker_breakdown(closed_trades)

            # By day
            stats.by_day = _self._calculate_daily_breakdown(closed_trades)

            # Equity curve and drawdown
            stats.equity_curve, stats.max_drawdown, stats.max_drawdown_pct = \
                _self._calculate_equity_curve(closed_trades)

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            stats.error = str(e)
            return stats

    def _calculate_streaks(self, trades: List[Dict]) -> Tuple[int, int]:
        """Calculate max win and loss streaks."""
        if not trades:
            return 0, 0

        sorted_trades = sorted(
            [t for t in trades if t.get("close_time")],
            key=lambda x: x["close_time"]
        )

        max_wins = 0
        max_losses = 0
        win_streak = 0
        loss_streak = 0

        for t in sorted_trades:
            if t["profit"] > 0:
                win_streak += 1
                loss_streak = 0
                max_wins = max(max_wins, win_streak)
            elif t["profit"] < 0:
                loss_streak += 1
                win_streak = 0
                max_losses = max(max_losses, loss_streak)

        return max_wins, max_losses

    def _calculate_ticker_breakdown(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Calculate stats by ticker."""
        breakdown = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0})

        for t in trades:
            ticker = t["ticker"]
            breakdown[ticker]["trades"] += 1
            breakdown[ticker]["profit"] += t["profit"]
            if t["profit"] > 0:
                breakdown[ticker]["wins"] += 1
            elif t["profit"] < 0:
                breakdown[ticker]["losses"] += 1

        for ticker, data in breakdown.items():
            if data["trades"] > 0:
                data["win_rate"] = (data["wins"] / data["trades"]) * 100
            data["profit"] = round(data["profit"], 2)

        return dict(breakdown)

    def _calculate_daily_breakdown(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Calculate stats by day."""
        breakdown = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0})

        for t in trades:
            if t.get("close_time"):
                day = t["close_time"][:10]  # YYYY-MM-DD
                breakdown[day]["trades"] += 1
                breakdown[day]["profit"] += t["profit"]
                if t["profit"] > 0:
                    breakdown[day]["wins"] += 1
                elif t["profit"] < 0:
                    breakdown[day]["losses"] += 1

        for day, data in breakdown.items():
            data["profit"] = round(data["profit"], 2)

        return dict(breakdown)

    def _calculate_equity_curve(
        self,
        trades: List[Dict],
        starting_equity: float = 10000.0
    ) -> Tuple[List[Tuple[str, float]], float, float]:
        """Calculate equity curve and max drawdown."""
        if not trades:
            return [], 0.0, 0.0

        sorted_trades = sorted(
            [t for t in trades if t.get("close_time")],
            key=lambda x: x["close_time"]
        )

        if not sorted_trades:
            return [], 0.0, 0.0

        equity = starting_equity
        peak_equity = equity
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

        curve = [(sorted_trades[0]["close_time"], equity)]

        for t in sorted_trades:
            equity += t["profit"]
            curve.append((t["close_time"], equity))

            if equity > peak_equity:
                peak_equity = equity

            drawdown = peak_equity - equity
            drawdown_pct = (drawdown / peak_equity) * 100 if peak_equity > 0 else 0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        return curve, round(max_drawdown, 2), round(max_drawdown_pct, 2)


    @st.cache_data(ttl=60)
    def get_account_balance(_self) -> Dict[str, Any]:
        """
        Get the latest account balance from database.

        Returns:
            Dict with current balance and trend information
        """
        try:
            with _self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get latest balance
                    cur.execute("""
                        SELECT total_value, invested, available, recorded_at
                        FROM broker_account_balance
                        ORDER BY id DESC
                        LIMIT 1
                    """)
                    latest = cur.fetchone()

                    if not latest:
                        return {
                            "error": "No balance data. Run: docker exec task-worker python -m stock_scanner.main broker-sync",
                            "total_value": 0.0,
                            "invested": 0.0,
                            "available": 0.0
                        }

                    return {
                        "total_value": float(latest["total_value"]),
                        "invested": float(latest["invested"]),
                        "available": float(latest["available"]),
                        "recorded_at": latest["recorded_at"],
                        "last_sync": _self.get_last_sync_time()
                    }

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {
                "error": str(e),
                "total_value": 0.0,
                "invested": 0.0,
                "available": 0.0
            }

    @st.cache_data(ttl=60)
    def get_account_balance_trend(_self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate account balance trend over a period.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with trend information including change amount and percentage
        """
        try:
            with _self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get oldest balance in the period
                    cur.execute("""
                        SELECT total_value, recorded_at
                        FROM broker_account_balance
                        WHERE recorded_at >= NOW() - INTERVAL '%s days'
                        ORDER BY recorded_at ASC
                        LIMIT 1
                    """ % days)
                    oldest = cur.fetchone()

                    # Get newest balance
                    cur.execute("""
                        SELECT total_value, recorded_at
                        FROM broker_account_balance
                        ORDER BY id DESC
                        LIMIT 1
                    """)
                    newest = cur.fetchone()

                    if not oldest or not newest:
                        return {
                            "change": 0.0,
                            "change_pct": 0.0,
                            "trend": "neutral",
                            "start_value": 0.0,
                            "end_value": 0.0,
                            "period_days": days,
                            "data_points": 0
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

                    # Count data points
                    cur.execute("""
                        SELECT COUNT(*) FROM broker_account_balance
                        WHERE recorded_at >= NOW() - INTERVAL '%s days'
                    """ % days)
                    count = cur.fetchone()[0]

                    return {
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                        "trend": trend,
                        "start_value": round(start_value, 2),
                        "end_value": round(end_value, 2),
                        "period_days": days,
                        "data_points": count
                    }

        except Exception as e:
            logger.error(f"Failed to get account balance trend: {e}")
            return {
                "error": str(e),
                "change": 0.0,
                "change_pct": 0.0,
                "trend": "neutral",
                "period_days": days
            }

    @st.cache_data(ttl=60)
    def get_account_balance_history(_self, days: int = 30) -> List[Dict]:
        """
        Get account balance history for charting.

        Args:
            days: Number of days of history

        Returns:
            List of balance records ordered by date
        """
        try:
            with _self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT total_value, invested, available, recorded_at
                        FROM broker_account_balance
                        WHERE recorded_at >= NOW() - INTERVAL '%s days'
                        ORDER BY recorded_at
                    """ % days)
                    rows = cur.fetchall()

                    return [
                        {
                            "total_value": float(r["total_value"]),
                            "invested": float(r["invested"]),
                            "available": float(r["available"]),
                            "recorded_at": r["recorded_at"].isoformat() if r["recorded_at"] else None
                        }
                        for r in rows
                    ]

        except Exception as e:
            logger.error(f"Failed to get account balance history: {e}")
            return []


@st.cache_resource
def get_broker_service() -> BrokerAnalyticsService:
    """Get cached broker analytics service instance."""
    return BrokerAnalyticsService()
