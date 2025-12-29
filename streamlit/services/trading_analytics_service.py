"""
Trading Analytics Service

Provides data access layer for core trading analytics including:
- Trading statistics (win rate, profit/loss, etc.)
- Trades dataframe with enhanced columns
- Strategy performance metrics
- Trade analysis data

All fetch methods are cached with @st.cache_data for performance.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass
import logging

from services.db_utils import get_psycopg2_connection

logger = logging.getLogger(__name__)


@dataclass
class TradingStatistics:
    """Data class for trading statistics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    pending_trades: int
    total_profit_loss: float
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    avg_win_duration: float
    avg_loss_duration: float
    total_volume: float
    active_pairs: List[str]
    best_pair: str
    worst_pair: str


class TradingAnalyticsService:
    """Service for fetching core trading analytics from the database."""

    def __init__(self):
        self.db_type = "trading"

    def _get_connection(self):
        """Get a database connection from the pool."""
        return get_psycopg2_connection(self.db_type)

    @st.cache_data(ttl=300)
    def fetch_trading_statistics(_self, days_back: int = 7, pairs_filter: tuple = None) -> Optional[TradingStatistics]:
        """
        Fetch comprehensive trading statistics from trade_log table (cached 5 min).

        Args:
            days_back: Number of days to look back
            pairs_filter: Tuple of symbols to filter (hashable for caching)

        Returns:
            TradingStatistics dataclass or None
        """
        conn = _self._get_connection()
        if not conn:
            return None

        try:
            with conn.cursor() as cursor:
                # Base query with enhanced filtering
                # Only count truly pending orders (pending, pending_limit)
                # Exclude terminal states: limit_not_filled, limit_rejected, limit_cancelled
                base_query = """
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losing_trades,
                    COUNT(CASE WHEN status IN ('pending', 'pending_limit') THEN 1 END) as pending_trades,
                    COALESCE(SUM(profit_loss), 0) as total_profit_loss,
                    COALESCE(AVG(CASE WHEN profit_loss > 0 THEN profit_loss END), 0) as avg_profit,
                    COALESCE(AVG(CASE WHEN profit_loss < 0 THEN profit_loss END), 0) as avg_loss,
                    COALESCE(MAX(profit_loss), 0) as largest_win,
                    COALESCE(MIN(profit_loss), 0) as largest_loss,
                    COUNT(DISTINCT symbol) as unique_pairs
                FROM trade_log
                WHERE timestamp >= %s
                """

                params = [datetime.now() - timedelta(days=days_back)]

                if pairs_filter:
                    base_query += " AND symbol = ANY(%s)"
                    params.append(list(pairs_filter))

                cursor.execute(base_query, params)
                result = cursor.fetchone()

                if not result or result[0] == 0:
                    return TradingStatistics(
                        total_trades=0, winning_trades=0, losing_trades=0, pending_trades=0,
                        total_profit_loss=0.0, win_rate=0.0, avg_profit=0.0, avg_loss=0.0,
                        profit_factor=0.0, largest_win=0.0, largest_loss=0.0,
                        avg_win_duration=0.0, avg_loss_duration=0.0, total_volume=0.0,
                        active_pairs=[], best_pair="None", worst_pair="None"
                    )

                # Calculate additional metrics
                total_trades, winning_trades, losing_trades, pending_trades = result[0:4]
                total_profit_loss, avg_profit, avg_loss, largest_win, largest_loss = result[4:9]

                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                profit_factor = abs(avg_profit * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')

                # Get pair-specific statistics
                pair_stats_query = """
                SELECT
                    symbol,
                    COUNT(*) as trades,
                    COALESCE(SUM(profit_loss), 0) as total_pnl,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins
                FROM trade_log
                WHERE timestamp >= %s
                """

                if pairs_filter:
                    pair_stats_query += " AND symbol = ANY(%s)"

                pair_stats_query += " GROUP BY symbol ORDER BY total_pnl DESC"

                cursor.execute(pair_stats_query, params)
                pair_results = cursor.fetchall()

                active_pairs = [row[0] for row in pair_results]
                best_pair = pair_results[0][0] if pair_results else "None"
                worst_pair = pair_results[-1][0] if pair_results else "None"

                return TradingStatistics(
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    pending_trades=pending_trades,
                    total_profit_loss=float(total_profit_loss),
                    win_rate=win_rate,
                    avg_profit=float(avg_profit),
                    avg_loss=float(avg_loss),
                    profit_factor=profit_factor,
                    largest_win=float(largest_win),
                    largest_loss=float(largest_loss),
                    avg_win_duration=0.0,
                    avg_loss_duration=0.0,
                    total_volume=0.0,
                    active_pairs=active_pairs,
                    best_pair=best_pair,
                    worst_pair=worst_pair
                )

        except Exception as e:
            logger.error(f"Error fetching trading statistics: {e}")
            return None
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_trades_dataframe(_self, days_back: int = 7, pairs_filter: tuple = None) -> pd.DataFrame:
        """
        Fetch detailed trades data as DataFrame (cached 5 min).

        Args:
            days_back: Number of days to look back
            pairs_filter: Tuple of symbols to filter (hashable for caching)

        Returns:
            DataFrame with trade data and enhanced columns
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                t.id, t.symbol, t.entry_price, t.direction, t.timestamp, t.status,
                t.profit_loss, t.pnl_currency, t.deal_id, t.sl_price, t.tp_price,
                t.closed_at, t.alert_id, a.strategy
            FROM trade_log t
            LEFT JOIN alert_history a ON t.alert_id = a.id
            WHERE t.timestamp >= %s
            """

            params = [datetime.now() - timedelta(days=days_back)]

            if pairs_filter:
                query += " AND t.symbol = ANY(%s)"
                params.append(list(pairs_filter))

            query += " ORDER BY t.timestamp DESC"

            df = pd.read_sql_query(query, conn, params=params)

            # Enhance DataFrame
            if not df.empty:
                # Determine trade result based on status and profit_loss
                # Only pending_limit and pending are truly "pending" orders
                # limit_not_filled, limit_rejected, limit_cancelled are terminal states (not pending)
                def get_trade_result(row):
                    status = row.get('status', '')
                    pnl = row.get('profit_loss')

                    # Check P&L first for completed trades
                    if pd.notna(pnl):
                        return 'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'BREAKEVEN'

                    # For NULL P&L, check status to determine actual state
                    if status in ('pending', 'pending_limit'):
                        return 'PENDING'
                    elif status == 'tracking':
                        return 'OPEN'
                    elif status == 'limit_not_filled':
                        return 'EXPIRED'
                    elif status == 'limit_rejected':
                        return 'REJECTED'
                    elif status == 'limit_cancelled':
                        return 'CANCELLED'
                    else:
                        # Fallback for any other status with NULL P&L
                        return 'PENDING'

                df['trade_result'] = df.apply(get_trade_result, axis=1)

                # Format P&L display based on trade result
                def format_pnl(row):
                    pnl = row.get('profit_loss')
                    result = row.get('trade_result', '')
                    currency = row.get('pnl_currency', '')

                    if pd.notna(pnl):
                        return f"{pnl:.2f} {currency}"
                    elif result == 'OPEN':
                        return "Open"
                    elif result == 'EXPIRED':
                        return "Not Filled"
                    elif result == 'REJECTED':
                        return "Rejected"
                    elif result == 'CANCELLED':
                        return "Cancelled"
                    else:
                        return "Pending"

                df['profit_loss_formatted'] = df.apply(format_pnl, axis=1)

            return df

        except Exception as e:
            logger.error(f"Error fetching trades data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_strategy_performance(_self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch strategy performance data (cached 5 min).

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with strategy performance metrics
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            strategy_query = """
            SELECT
                a.strategy,
                COUNT(t.*) as total_trades,
                COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
                COALESCE(SUM(t.profit_loss), 0) as total_pnl,
                COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
                COALESCE(AVG(a.confidence_score), 0) as avg_confidence,
                COALESCE(MAX(t.profit_loss), 0) as best_trade,
                COALESCE(MIN(t.profit_loss), 0) as worst_trade,
                COUNT(DISTINCT t.symbol) as pairs_traded
            FROM trade_log t
            INNER JOIN alert_history a ON t.alert_id = a.id
            WHERE t.timestamp >= %s
            GROUP BY a.strategy
            ORDER BY total_pnl DESC
            """

            df = pd.read_sql_query(
                strategy_query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                df['win_rate'] = (df['wins'] / df['total_trades'] * 100).round(1)
                df['profit_factor'] = df.apply(
                    lambda row: (row['wins'] * abs(row['avg_pnl'])) / (row['losses'] * abs(row['avg_pnl']))
                    if row['losses'] > 0 and row['avg_pnl'] < 0 else float('inf'), axis=1
                )

            return df

        except Exception as e:
            logger.error(f"Error fetching strategy performance: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def fetch_latest_closed_trade_id(_self) -> int:
        """
        Fetch the ID of the most recent trade entry that is closed (cached 1 min).

        Returns:
            Trade ID or 1 as fallback
        """
        conn = _self._get_connection()
        if not conn:
            return 1  # Default fallback

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM trade_log
                    WHERE status = 'closed'
                    ORDER BY id DESC
                    LIMIT 1
                """)
                result = cursor.fetchone()
                return result[0] if result else 1
        except Exception as e:
            logger.warning(f"Error fetching latest closed trade: {e}")
            return 1  # Default fallback
        finally:
            conn.close()

    @st.cache_data(ttl=180)
    def fetch_filled_trades_for_analysis(_self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch only filled trades (closed or tracking) for analysis (cached 3 min).

        Args:
            limit: Maximum number of trades to return

        Returns:
            DataFrame with filled trades and display-friendly columns
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            # Only fetch trades that were actually filled (closed or currently tracking)
            # Exclude: pending, pending_limit, limit_not_filled, limit_rejected, limit_cancelled
            query = """
                SELECT
                    id, symbol, direction, timestamp, status,
                    profit_loss, pnl_currency
                FROM trade_log
                WHERE status IN ('closed', 'tracking')
                ORDER BY timestamp DESC
                LIMIT %s
            """
            df = pd.read_sql_query(query, conn, params=[limit])

            if not df.empty:
                # Create display-friendly columns
                df['symbol_short'] = df['symbol'].str.replace('CS.D.', '').str.replace('.MINI.IP', '').str.replace('.CEEM.IP', '')
                df['pnl_display'] = df.apply(
                    lambda row: f"{row['profit_loss']:+.2f} {row['pnl_currency']}" if pd.notna(row['profit_loss']) else "Open",
                    axis=1
                )

            return df
        except Exception as e:
            logger.warning(f"Error fetching filled trades for analysis: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_pair_performance(_self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch performance breakdown by currency pair (cached 5 min).

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with pair performance metrics
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                symbol,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses,
                COALESCE(SUM(profit_loss), 0) as total_pnl,
                COALESCE(AVG(profit_loss), 0) as avg_pnl,
                COALESCE(MAX(profit_loss), 0) as best_trade,
                COALESCE(MIN(profit_loss), 0) as worst_trade
            FROM trade_log
            WHERE timestamp >= %s
            GROUP BY symbol
            ORDER BY total_pnl DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                df['win_rate'] = (df['wins'] / df['total_trades'] * 100).round(1)

            return df

        except Exception as e:
            logger.error(f"Error fetching pair performance: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_daily_pnl(_self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch daily profit/loss data for charting (cached 5 min).

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with date, daily_pnl, cumulative_pnl
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                DATE(timestamp) as date,
                SUM(profit_loss) as daily_pnl,
                COUNT(*) as trade_count,
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses
            FROM trade_log
            WHERE timestamp >= %s
            AND profit_loss IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                df['cumulative_pnl'] = df['daily_pnl'].cumsum()
                df['daily_win_rate'] = (df['wins'] / df['trade_count'] * 100).round(1)

            return df

        except Exception as e:
            logger.error(f"Error fetching daily PnL: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def get_available_pairs(_self) -> List[str]:
        """
        Get list of all pairs that have trades (cached 1 min).

        Returns:
            List of unique symbol names
        """
        conn = _self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT symbol
                    FROM trade_log
                    WHERE symbol IS NOT NULL
                    ORDER BY symbol
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching available pairs: {e}")
            return []
        finally:
            conn.close()
