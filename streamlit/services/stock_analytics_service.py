"""
Stock Analytics Service

Provides data access layer for the Stock Scanner dashboard.
Connects to the 'stocks' database and retrieves watchlist, metrics, and signals data.
"""

import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class StockAnalyticsService:
    """Service for fetching stock scanner data from the stocks database."""

    def __init__(self):
        self._conn = None

    def _get_connection_string(self) -> str:
        """Get database connection string from secrets or environment."""
        try:
            return st.secrets.database.stocks_connection_string
        except Exception:
            return "postgresql://postgres:postgres@postgres:5432/stocks"

    def _get_connection(self):
        """Get a database connection."""
        try:
            return psycopg2.connect(self._get_connection_string())
        except Exception as e:
            logger.error(f"Failed to connect to stocks database: {e}")
            return None

    def _execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query and return results as list of dicts."""
        conn = self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
        finally:
            conn.close()

    # =========================================================================
    # OVERVIEW QUERIES
    # =========================================================================

    @st.cache_data(ttl=300)
    def get_overview_stats(_self) -> Dict[str, Any]:
        """Get high-level statistics for the overview page."""
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get latest calculation date
                cursor.execute("""
                    SELECT MAX(calculation_date) as latest_date
                    FROM stock_screening_metrics
                """)
                latest_date = cursor.fetchone()['latest_date']

                # Count stocks with metrics
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM stock_screening_metrics
                    WHERE calculation_date = %s
                """, (latest_date,))
                total_with_metrics = cursor.fetchone()['count']

                # Count stocks in watchlist by tier
                cursor.execute("""
                    SELECT
                        tier,
                        COUNT(*) as count,
                        ROUND(AVG(score)::numeric, 1) as avg_score,
                        ROUND(AVG(atr_percent)::numeric, 2) as avg_atr
                    FROM stock_watchlist
                    WHERE calculation_date = %s
                    GROUP BY tier
                    ORDER BY tier
                """, (latest_date,))
                tier_stats = {row['tier']: dict(row) for row in cursor.fetchall()}

                total_watchlist = sum(t['count'] for t in tier_stats.values())

                # Count active signals (last 7 days)
                cursor.execute("""
                    SELECT
                        signal_type,
                        COUNT(*) as count
                    FROM stock_zlma_signals
                    WHERE signal_timestamp > NOW() - INTERVAL '7 days'
                    GROUP BY signal_type
                """)
                signal_counts = {row['signal_type']: row['count'] for row in cursor.fetchall()}

                return {
                    'latest_date': latest_date,
                    'total_with_metrics': total_with_metrics,
                    'total_watchlist': total_watchlist,
                    'tier_stats': tier_stats,
                    'buy_signals': signal_counts.get('BUY', 0),
                    'sell_signals': signal_counts.get('SELL', 0),
                    'total_signals': signal_counts.get('BUY', 0) + signal_counts.get('SELL', 0)
                }
        except Exception as e:
            logger.error(f"Failed to get overview stats: {e}")
            return {}
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_top_opportunities(_self, limit: int = 10) -> pd.DataFrame:
        """Get top-ranked stocks from watchlist with any active signals."""
        query = """
            SELECT
                w.rank_overall,
                w.tier,
                w.ticker,
                i.name,
                w.score,
                w.current_price,
                w.atr_percent,
                ROUND((w.avg_dollar_volume / 1000000)::numeric, 1) as dollar_vol_m,
                w.relative_volume,
                w.price_change_20d,
                w.trend_strength,
                COALESCE(s.signal_type, '') as signal_type,
                COALESCE(s.confidence, 0) as signal_confidence
            FROM stock_watchlist w
            JOIN stock_instruments i ON w.ticker = i.ticker
            LEFT JOIN LATERAL (
                SELECT signal_type, confidence
                FROM stock_zlma_signals
                WHERE ticker = w.ticker
                AND signal_timestamp > NOW() - INTERVAL '7 days'
                ORDER BY signal_timestamp DESC
                LIMIT 1
            ) s ON true
            WHERE w.calculation_date = (
                SELECT MAX(calculation_date) FROM stock_watchlist
            )
            ORDER BY w.rank_overall
            LIMIT %s
        """
        results = _self._execute_query(query, (limit,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_recent_signals(_self, hours: int = 24) -> pd.DataFrame:
        """Get signals from the last N hours."""
        query = """
            SELECT
                s.signal_timestamp,
                s.ticker,
                i.name,
                s.signal_type,
                s.entry_price,
                s.stop_loss,
                s.take_profit,
                s.confidence,
                CASE
                    WHEN s.signal_type = 'BUY' THEN
                        ROUND(((s.take_profit - s.entry_price) / (s.entry_price - s.stop_loss))::numeric, 1)
                    ELSE
                        ROUND(((s.entry_price - s.take_profit) / (s.stop_loss - s.entry_price))::numeric, 1)
                END as risk_reward,
                w.tier,
                w.score
            FROM stock_zlma_signals s
            JOIN stock_instruments i ON s.ticker = i.ticker
            LEFT JOIN stock_watchlist w ON s.ticker = w.ticker
                AND w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            WHERE s.signal_timestamp > NOW() - INTERVAL '%s hours'
            ORDER BY s.signal_timestamp DESC
        """
        results = _self._execute_query(query % hours, ())
        return pd.DataFrame(results) if results else pd.DataFrame()

    # =========================================================================
    # WATCHLIST QUERIES
    # =========================================================================

    @st.cache_data(ttl=300)
    def get_watchlist(_self,
                      tiers: List[int] = None,
                      min_score: float = 0,
                      max_score: float = 100,
                      min_atr: float = 0,
                      max_atr: float = 100,
                      min_dollar_vol: float = 0,
                      trends: List[str] = None,
                      ma_alignments: List[str] = None,
                      min_rsi: float = 0,
                      max_rsi: float = 100,
                      min_rvol: float = 0,
                      max_rvol: float = 100,
                      has_signal: bool = None,
                      is_new_to_tier: bool = None) -> pd.DataFrame:
        """Get filtered watchlist data."""

        query = """
            SELECT
                w.rank_overall,
                w.rank_in_tier,
                w.tier,
                w.ticker,
                i.name,
                w.score,
                w.volume_score,
                w.volatility_score,
                w.momentum_score,
                w.relative_strength_score,
                w.current_price,
                w.atr_percent,
                ROUND((w.avg_dollar_volume / 1000000)::numeric, 1) as dollar_vol_m,
                w.relative_volume,
                m.price_change_1d,
                m.price_change_5d,
                w.price_change_20d,
                w.trend_strength,
                m.ma_alignment,
                m.rsi_14,
                w.is_new_to_tier,
                w.tier_change,
                COALESCE(sig.signal_count, 0) as signal_count,
                COALESCE(sig.latest_signal, '') as latest_signal
            FROM stock_watchlist w
            JOIN stock_instruments i ON w.ticker = i.ticker
            LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            LEFT JOIN LATERAL (
                SELECT
                    COUNT(*) as signal_count,
                    (SELECT signal_type FROM stock_zlma_signals
                     WHERE ticker = w.ticker
                     ORDER BY signal_timestamp DESC LIMIT 1) as latest_signal
                FROM stock_zlma_signals
                WHERE ticker = w.ticker
                AND signal_timestamp > NOW() - INTERVAL '7 days'
            ) sig ON true
            WHERE w.calculation_date = (
                SELECT MAX(calculation_date) FROM stock_watchlist
            )
            AND w.score BETWEEN %s AND %s
            AND w.atr_percent BETWEEN %s AND %s
            AND (w.avg_dollar_volume / 1000000) >= %s
            AND COALESCE(m.rsi_14, 50) BETWEEN %s AND %s
            AND COALESCE(w.relative_volume, 1) BETWEEN %s AND %s
        """

        params = [min_score, max_score, min_atr, max_atr, min_dollar_vol,
                  min_rsi, max_rsi, min_rvol, max_rvol]

        # Add tier filter
        if tiers:
            query += " AND w.tier = ANY(%s)"
            params.append(tiers)

        # Add trend filter
        if trends:
            query += " AND w.trend_strength = ANY(%s)"
            params.append(trends)

        # Add MA alignment filter
        if ma_alignments:
            query += " AND m.ma_alignment = ANY(%s)"
            params.append(ma_alignments)

        # Add signal filter
        if has_signal is True:
            query += " AND COALESCE(sig.signal_count, 0) > 0"
        elif has_signal is False:
            query += " AND COALESCE(sig.signal_count, 0) = 0"

        # Add new to tier filter
        if is_new_to_tier is True:
            query += " AND w.is_new_to_tier = true"

        query += " ORDER BY w.rank_overall"

        results = _self._execute_query(query, tuple(params))
        return pd.DataFrame(results) if results else pd.DataFrame()

    # =========================================================================
    # SIGNAL QUERIES
    # =========================================================================

    @st.cache_data(ttl=60)
    def get_all_signals(_self,
                        signal_type: str = None,
                        days_back: int = 7,
                        min_confidence: float = 0) -> pd.DataFrame:
        """Get all signals with filtering options."""
        # Build query with days_back as string interpolation (safe since it's an int)
        query = f"""
            SELECT
                s.id,
                s.signal_timestamp,
                s.ticker,
                i.name,
                s.signal_type,
                s.entry_price,
                s.stop_loss,
                s.take_profit,
                s.zlma_value,
                s.ema_value,
                s.atr_value,
                s.level_top,
                s.level_bottom,
                s.confidence,
                CASE
                    WHEN s.signal_type = 'BUY' THEN
                        ROUND(((s.take_profit - s.entry_price) / NULLIF(s.entry_price - s.stop_loss, 0))::numeric, 2)
                    ELSE
                        ROUND(((s.entry_price - s.take_profit) / NULLIF(s.stop_loss - s.entry_price, 0))::numeric, 2)
                END as risk_reward,
                w.tier,
                w.score,
                w.trend_strength,
                m.current_price
            FROM stock_zlma_signals s
            JOIN stock_instruments i ON s.ticker = i.ticker
            LEFT JOIN stock_watchlist w ON s.ticker = w.ticker
                AND w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            LEFT JOIN stock_screening_metrics m ON s.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE s.signal_timestamp > NOW() - INTERVAL '{int(days_back)} days'
            AND s.confidence >= %s
        """

        params = [min_confidence]

        if signal_type:
            query += " AND s.signal_type = %s"
            params.append(signal_type)

        query += " ORDER BY s.signal_timestamp DESC"

        results = _self._execute_query(query, tuple(params))
        return pd.DataFrame(results) if results else pd.DataFrame()

    # =========================================================================
    # STOCK DETAIL QUERIES
    # =========================================================================

    @st.cache_data(ttl=300)
    def get_stock_details(_self, ticker: str) -> Dict[str, Any]:
        """Get detailed information for a single stock."""
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get instrument info
                cursor.execute("""
                    SELECT ticker, name, exchange, sector, market_cap
                    FROM stock_instruments
                    WHERE ticker = %s
                """, (ticker,))
                instrument = cursor.fetchone()

                if not instrument:
                    return {}

                # Get latest metrics
                cursor.execute("""
                    SELECT *
                    FROM stock_screening_metrics
                    WHERE ticker = %s
                    ORDER BY calculation_date DESC
                    LIMIT 1
                """, (ticker,))
                metrics = cursor.fetchone()

                # Get watchlist info
                cursor.execute("""
                    SELECT *
                    FROM stock_watchlist
                    WHERE ticker = %s
                    ORDER BY calculation_date DESC
                    LIMIT 1
                """, (ticker,))
                watchlist = cursor.fetchone()

                # Get signals for this stock
                cursor.execute("""
                    SELECT *
                    FROM stock_zlma_signals
                    WHERE ticker = %s
                    ORDER BY signal_timestamp DESC
                    LIMIT 10
                """, (ticker,))
                signals = [dict(row) for row in cursor.fetchall()]

                return {
                    'instrument': dict(instrument) if instrument else {},
                    'metrics': dict(metrics) if metrics else {},
                    'watchlist': dict(watchlist) if watchlist else {},
                    'signals': signals
                }
        except Exception as e:
            logger.error(f"Failed to get stock details for {ticker}: {e}")
            return {}
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_daily_candles(_self, ticker: str, days: int = 60) -> pd.DataFrame:
        """Get daily candle data for charting."""
        query = """
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM stock_candles_synthesized
            WHERE ticker = %s AND timeframe = '1d'
            ORDER BY timestamp DESC
            LIMIT %s
        """
        results = _self._execute_query(query, (ticker, days))
        df = pd.DataFrame(results) if results else pd.DataFrame()
        if not df.empty:
            df = df.sort_values('timestamp')
        return df

    @st.cache_data(ttl=300)
    def get_all_tickers(_self) -> List[str]:
        """Get list of all tickers in watchlist."""
        query = """
            SELECT DISTINCT ticker
            FROM stock_watchlist
            WHERE calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            ORDER BY ticker
        """
        results = _self._execute_query(query, ())
        return [r['ticker'] for r in results]

    @st.cache_data(ttl=300)
    def get_ticker_search(_self, search_term: str, limit: int = 20) -> List[Dict]:
        """Search for tickers by symbol or name."""
        query = """
            SELECT
                w.ticker,
                i.name,
                w.tier,
                w.score
            FROM stock_watchlist w
            JOIN stock_instruments i ON w.ticker = i.ticker
            WHERE w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            AND (
                w.ticker ILIKE %s
                OR i.name ILIKE %s
            )
            ORDER BY w.rank_overall
            LIMIT %s
        """
        search_pattern = f"%{search_term}%"
        results = _self._execute_query(query, (search_pattern, search_pattern, limit))
        return results


# Singleton instance
@st.cache_resource
def get_stock_service() -> StockAnalyticsService:
    """Get or create the stock analytics service singleton."""
    return StockAnalyticsService()
