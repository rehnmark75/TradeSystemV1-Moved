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

            # Enhance DataFrame using vectorized operations (much faster than .apply())
            if not df.empty:
                # Determine trade result using vectorized numpy operations
                # Initialize with 'PENDING' as default
                df['trade_result'] = 'PENDING'

                # Handle P&L-based results first (highest priority)
                has_pnl = df['profit_loss'].notna()
                df.loc[has_pnl & (df['profit_loss'] > 0), 'trade_result'] = 'WIN'
                df.loc[has_pnl & (df['profit_loss'] < 0), 'trade_result'] = 'LOSS'
                df.loc[has_pnl & (df['profit_loss'] == 0), 'trade_result'] = 'BREAKEVEN'

                # Handle status-based results for NULL P&L
                no_pnl = ~has_pnl
                df.loc[no_pnl & df['status'].isin(['pending', 'pending_limit']), 'trade_result'] = 'PENDING'
                df.loc[no_pnl & (df['status'] == 'tracking'), 'trade_result'] = 'OPEN'
                df.loc[no_pnl & (df['status'] == 'limit_not_filled'), 'trade_result'] = 'EXPIRED'
                df.loc[no_pnl & (df['status'] == 'limit_rejected'), 'trade_result'] = 'REJECTED'
                df.loc[no_pnl & (df['status'] == 'limit_cancelled'), 'trade_result'] = 'CANCELLED'

                # Format P&L display using vectorized operations
                # Start with status-based text for NULL P&L
                pnl_text_map = {
                    'OPEN': 'Open',
                    'EXPIRED': 'Not Filled',
                    'REJECTED': 'Rejected',
                    'CANCELLED': 'Cancelled',
                    'PENDING': 'Pending'
                }
                df['profit_loss_formatted'] = df['trade_result'].map(pnl_text_map).fillna('Pending')

                # Override with formatted P&L where available
                has_pnl_mask = df['profit_loss'].notna()
                if has_pnl_mask.any():
                    df.loc[has_pnl_mask, 'profit_loss_formatted'] = (
                        df.loc[has_pnl_mask, 'profit_loss'].apply(lambda x: f"{x:.2f}") +
                        ' ' +
                        df.loc[has_pnl_mask, 'pnl_currency'].fillna('')
                    )

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

    @st.cache_data(ttl=60)
    def fetch_scalp_mae_analysis(_self, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch MAE (Maximum Adverse Excursion) analysis for scalp trades (cached 1 min).

        This shows how much price retraced against entry before moving favorably,
        useful for optimizing VSL (Virtual Stop Loss) settings.

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with MAE analysis for scalp trades
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                id,
                symbol,
                direction,
                entry_price,
                timestamp,
                status,
                profit_loss,
                vsl_peak_profit_pips as mfe_pips,
                vsl_mae_pips as mae_pips,
                vsl_mae_price as mae_price,
                vsl_mae_timestamp as mae_time,
                virtual_sl_pips,
                vsl_stage,
                vsl_breakeven_triggered as hit_breakeven,
                vsl_stage1_triggered as hit_stage1,
                vsl_stage2_triggered as hit_stage2
            FROM trade_log
            WHERE is_scalp_trade = true
            AND timestamp >= %s
            ORDER BY timestamp DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                # Create display-friendly columns
                df['symbol_short'] = df['symbol'].str.replace('CS.D.', '').str.replace('.MINI.IP', '').str.replace('.CEEM.IP', '')

                # Calculate MAE as % of VSL
                df['mae_pct_of_vsl'] = (df['mae_pips'] / df['virtual_sl_pips'] * 100).round(1)

                # Determine trade result
                df['result'] = df.apply(
                    lambda row: 'WIN' if row['profit_loss'] and row['profit_loss'] > 0
                    else 'LOSS' if row['profit_loss'] and row['profit_loss'] < 0
                    else 'OPEN' if row['status'] == 'tracking'
                    else 'PENDING', axis=1
                )

            return df

        except Exception as e:
            logger.error(f"Error fetching scalp MAE analysis: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def fetch_mae_summary_by_pair(_self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch MAE summary statistics grouped by currency pair (cached 1 min).

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with MAE statistics per pair
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                symbol,
                COUNT(*) as total_trades,
                AVG(vsl_mae_pips) as avg_mae_pips,
                MAX(vsl_mae_pips) as max_mae_pips,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vsl_mae_pips) as median_mae_pips,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY vsl_mae_pips) as p75_mae_pips,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY vsl_mae_pips) as p90_mae_pips,
                AVG(vsl_peak_profit_pips) as avg_mfe_pips,
                MAX(vsl_peak_profit_pips) as max_mfe_pips,
                AVG(virtual_sl_pips) as avg_vsl_setting,
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses
            FROM trade_log
            WHERE is_scalp_trade = true
            AND timestamp >= %s
            AND vsl_mae_pips IS NOT NULL
            GROUP BY symbol
            ORDER BY total_trades DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                df['symbol_short'] = df['symbol'].str.replace('CS.D.', '').str.replace('.MINI.IP', '').str.replace('.CEEM.IP', '')
                df['win_rate'] = (df['wins'] / df['total_trades'] * 100).round(1)
                # Round numeric columns
                for col in ['avg_mae_pips', 'max_mae_pips', 'median_mae_pips', 'p75_mae_pips', 'p90_mae_pips', 'avg_mfe_pips', 'max_mfe_pips']:
                    if col in df.columns:
                        df[col] = df[col].round(1)

            return df

        except Exception as e:
            logger.error(f"Error fetching MAE summary by pair: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def fetch_entry_timing_analysis(_self, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch entry timing analysis data for scalp trades (cached 1 min).

        Analyzes entry quality by looking at:
        - Entry type (PULLBACK, MOMENTUM, MICRO_PULLBACK)
        - Signal trigger (SWING_PULLBACK, SWING_PULLBACK+PIN, etc.)
        - Time from entry to MAE (how quickly price moved against us)
        - MFE before MAE (did price move favorably at all)
        - Signal price vs entry price (slippage)

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with entry timing analysis
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                t.id,
                t.symbol,
                t.direction,
                t.entry_price,
                t.timestamp as trade_timestamp,
                t.status,
                t.profit_loss,
                t.vsl_peak_profit_pips as mfe_pips,
                t.vsl_mae_pips as mae_pips,
                t.vsl_mae_price,
                t.vsl_mae_timestamp as mae_timestamp,
                t.virtual_sl_pips,
                t.vsl_stage,
                t.closed_at,
                t.is_scalp_trade,
                -- Alert/Signal data
                a.id as alert_id,
                a.alert_timestamp as signal_timestamp,
                a.price as signal_price,
                a.confidence_score,
                a.signal_trigger,
                a.trigger_type,
                -- Entry type from strategy_indicators JSON
                a.strategy_indicators->'tier3_entry'->>'entry_type' as entry_type,
                a.strategy_indicators->'tier3_entry'->>'order_type' as order_type,
                a.strategy_indicators->'tier3_entry'->>'limit_offset_pips' as limit_offset_pips,
                a.strategy_indicators->'tier3_entry'->>'pullback_depth' as pullback_depth,
                a.strategy_indicators->'tier3_entry'->>'in_optimal_zone' as in_optimal_zone,
                -- Pattern and divergence data (v3.3.0)
                a.pattern_type,
                a.pattern_strength,
                a.rsi_divergence_detected,
                a.rsi_divergence,
                -- HTF alignment
                a.htf_candle_direction,
                a.market_session
            FROM trade_log t
            LEFT JOIN alert_history a ON t.alert_id = a.id
            WHERE t.timestamp >= %s
            AND t.status IN ('closed', 'tracking', 'expired')
            ORDER BY t.timestamp DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                # Create display-friendly columns
                df['symbol_short'] = df['symbol'].str.replace('CS.D.', '').str.replace('.MINI.IP', '').str.replace('.CEEM.IP', '')

                # Determine trade result
                df['result'] = df.apply(
                    lambda row: 'WIN' if row['profit_loss'] and row['profit_loss'] > 0
                    else 'LOSS' if row['profit_loss'] and row['profit_loss'] < 0
                    else 'OPEN' if row['status'] == 'tracking'
                    else 'PENDING', axis=1
                )

                # Calculate time to MAE (seconds from entry to worst point)
                df['time_to_mae_seconds'] = None
                mask = df['mae_timestamp'].notna() & df['trade_timestamp'].notna()
                if mask.any():
                    df.loc[mask, 'time_to_mae_seconds'] = (
                        pd.to_datetime(df.loc[mask, 'mae_timestamp']) -
                        pd.to_datetime(df.loc[mask, 'trade_timestamp'])
                    ).dt.total_seconds()

                # Calculate slippage (signal price vs entry price)
                df['slippage_pips'] = None
                # Get pip values per pair (simplified)
                jpy_pairs = df['symbol'].str.contains('JPY', na=False)
                pip_divisor = pd.Series(0.0001, index=df.index)
                pip_divisor[jpy_pairs] = 0.01

                mask = df['signal_price'].notna() & df['entry_price'].notna()
                if mask.any():
                    price_diff = df.loc[mask, 'entry_price'] - df.loc[mask, 'signal_price'].astype(float)
                    # For BUY: positive slippage = worse entry (paid more)
                    # For SELL: negative slippage = worse entry (sold for less)
                    df.loc[mask, 'slippage_pips'] = price_diff / pip_divisor[mask]
                    # Normalize: positive = worse entry for both directions
                    sell_mask = mask & (df['direction'] == 'SELL')
                    df.loc[sell_mask, 'slippage_pips'] = -df.loc[sell_mask, 'slippage_pips']

                # Convert numeric strings
                for col in ['pullback_depth', 'limit_offset_pips']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Flag trades with zero MFE (immediate adverse movement)
                df['zero_mfe'] = (df['mfe_pips'].fillna(0) == 0) | (df['mfe_pips'].fillna(0) < 0.5)

                # Calculate trade duration
                df['duration_minutes'] = None
                mask = df['closed_at'].notna() & df['trade_timestamp'].notna()
                if mask.any():
                    df.loc[mask, 'duration_minutes'] = (
                        pd.to_datetime(df.loc[mask, 'closed_at']) -
                        pd.to_datetime(df.loc[mask, 'trade_timestamp'])
                    ).dt.total_seconds() / 60

            return df

        except Exception as e:
            logger.error(f"Error fetching entry timing analysis: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def fetch_entry_timing_summary(_self, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch entry timing summary grouped by entry type (cached 1 min).

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with summary statistics per entry type
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                COALESCE(a.strategy_indicators->'tier3_entry'->>'entry_type', 'UNKNOWN') as entry_type,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
                COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
                COALESCE(SUM(t.profit_loss), 0) as total_pnl,
                AVG(t.vsl_mae_pips) as avg_mae_pips,
                AVG(t.vsl_peak_profit_pips) as avg_mfe_pips,
                -- Count trades with zero/minimal MFE (bad timing indicator)
                COUNT(CASE WHEN COALESCE(t.vsl_peak_profit_pips, 0) < 0.5 THEN 1 END) as zero_mfe_count,
                AVG(a.confidence_score) as avg_confidence,
                AVG(CAST(a.strategy_indicators->'tier3_entry'->>'pullback_depth' AS FLOAT)) as avg_pullback_depth
            FROM trade_log t
            LEFT JOIN alert_history a ON t.alert_id = a.id
            WHERE t.timestamp >= %s
            AND t.status IN ('closed', 'expired')
            AND t.profit_loss IS NOT NULL
            GROUP BY a.strategy_indicators->'tier3_entry'->>'entry_type'
            ORDER BY total_trades DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                df['win_rate'] = (df['wins'] / df['total_trades'] * 100).round(1)
                df['zero_mfe_pct'] = (df['zero_mfe_count'] / df['total_trades'] * 100).round(1)
                # Round numeric columns
                for col in ['avg_pnl', 'total_pnl', 'avg_mae_pips', 'avg_mfe_pips', 'avg_confidence', 'avg_pullback_depth']:
                    if col in df.columns:
                        df[col] = df[col].round(2)

            return df

        except Exception as e:
            logger.error(f"Error fetching entry timing summary: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def fetch_entry_timing_by_trigger(_self, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch entry timing summary grouped by signal trigger type (cached 1 min).

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with summary statistics per trigger type
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                COALESCE(NULLIF(a.signal_trigger, ''), 'STANDARD') as signal_trigger,
                COALESCE(a.strategy_indicators->'tier3_entry'->>'entry_type', 'UNKNOWN') as entry_type,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
                COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
                COALESCE(SUM(t.profit_loss), 0) as total_pnl,
                AVG(t.vsl_mae_pips) as avg_mae_pips,
                AVG(t.vsl_peak_profit_pips) as avg_mfe_pips,
                COUNT(CASE WHEN COALESCE(t.vsl_peak_profit_pips, 0) < 0.5 THEN 1 END) as zero_mfe_count,
                AVG(a.confidence_score) as avg_confidence
            FROM trade_log t
            LEFT JOIN alert_history a ON t.alert_id = a.id
            WHERE t.timestamp >= %s
            AND t.status IN ('closed', 'expired')
            AND t.profit_loss IS NOT NULL
            GROUP BY a.signal_trigger, a.strategy_indicators->'tier3_entry'->>'entry_type'
            ORDER BY total_trades DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[datetime.now() - timedelta(days=days_back)]
            )

            if not df.empty:
                df['win_rate'] = (df['wins'] / df['total_trades'] * 100).round(1)
                df['zero_mfe_pct'] = (df['zero_mfe_count'] / df['total_trades'] * 100).round(1)
                for col in ['avg_pnl', 'total_pnl', 'avg_mae_pips', 'avg_mfe_pips', 'avg_confidence']:
                    if col in df.columns:
                        df[col] = df[col].round(2)

            return df

        except Exception as e:
            logger.error(f"Error fetching entry timing by trigger: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
