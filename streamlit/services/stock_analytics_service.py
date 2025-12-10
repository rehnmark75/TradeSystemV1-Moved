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

                # Get SMC trend breakdown
                cursor.execute("""
                    SELECT
                        COALESCE(smc_trend, 'Unknown') as smc_trend,
                        COUNT(*) as count
                    FROM stock_screening_metrics
                    WHERE calculation_date = %s
                    GROUP BY smc_trend
                """, (latest_date,))
                smc_trends = {row['smc_trend']: row['count'] for row in cursor.fetchall()}

                # Get upcoming earnings count (next 14 days)
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM stock_instruments
                    WHERE earnings_date IS NOT NULL
                    AND earnings_date >= CURRENT_DATE
                    AND earnings_date <= CURRENT_DATE + INTERVAL '14 days'
                    AND is_active = TRUE
                """)
                upcoming_earnings = cursor.fetchone()['count']

                # Get high short interest count (>15% float)
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM stock_instruments
                    WHERE short_percent_float >= 15
                    AND is_active = TRUE
                """)
                high_short_interest = cursor.fetchone()['count']

                return {
                    'latest_date': latest_date,
                    'total_with_metrics': total_with_metrics,
                    'total_watchlist': total_watchlist,
                    'tier_stats': tier_stats,
                    'buy_signals': signal_counts.get('BUY', 0),
                    'sell_signals': signal_counts.get('SELL', 0),
                    'total_signals': signal_counts.get('BUY', 0) + signal_counts.get('SELL', 0),
                    'smc_bullish': smc_trends.get('Bullish', 0),
                    'smc_bearish': smc_trends.get('Bearish', 0),
                    'smc_neutral': smc_trends.get('Neutral', 0),
                    'upcoming_earnings': upcoming_earnings,
                    'high_short_interest': high_short_interest,
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
                COALESCE(m.smc_trend, '') as smc_trend,
                COALESCE(m.smc_bias, '') as smc_bias,
                COALESCE(m.premium_discount_zone, '') as zone,
                COALESCE(m.smc_confluence_score, 0) as smc_score,
                COALESCE(s.signal_type, '') as signal_type,
                COALESCE(s.confidence, 0) as signal_confidence,
                i.earnings_date,
                i.beta,
                i.short_percent_float
            FROM stock_watchlist w
            JOIN stock_instruments i ON w.ticker = i.ticker
            LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
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
                      smc_trends: List[str] = None,
                      smc_zones: List[str] = None,
                      min_rsi: float = 0,
                      max_rsi: float = 100,
                      min_rvol: float = 0,
                      max_rvol: float = 100,
                      has_signal: bool = None,
                      is_new_to_tier: bool = None,
                      earnings_within_days: int = None,
                      min_short_interest: float = None,
                      rsi_signals: List[str] = None,
                      sma_cross_signals: List[str] = None,
                      macd_cross_signals: List[str] = None,
                      high_low_signals: List[str] = None,
                      candlestick_patterns: List[str] = None) -> pd.DataFrame:
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
                COALESCE(m.smc_trend, '') as smc_trend,
                COALESCE(m.smc_bias, '') as smc_bias,
                COALESCE(m.premium_discount_zone, '') as smc_zone,
                COALESCE(m.zone_position, 50) as zone_position,
                COALESCE(m.smc_confluence_score, 0) as smc_score,
                COALESCE(m.last_bos_type, '') as last_bos,
                w.is_new_to_tier,
                w.tier_change,
                COALESCE(sig.signal_count, 0) as signal_count,
                COALESCE(sig.latest_signal, '') as latest_signal,
                i.earnings_date,
                i.beta,
                i.short_percent_float,
                i.short_ratio,
                i.analyst_rating,
                i.target_price,
                -- Enhanced signal columns
                COALESCE(w.rsi_signal, '') as rsi_signal,
                COALESCE(w.sma_cross_signal, '') as sma_cross_signal,
                COALESCE(w.macd_cross_signal, '') as macd_cross_signal,
                COALESCE(w.high_low_signal, '') as high_low_signal,
                COALESCE(w.gap_signal, '') as gap_signal,
                COALESCE(w.candlestick_pattern, '') as candlestick_pattern,
                COALESCE(w.pct_from_52w_high, -100) as pct_from_52w_high
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

        # Add SMC trend filter
        if smc_trends:
            query += " AND m.smc_trend = ANY(%s)"
            params.append(smc_trends)

        # Add SMC zone filter
        if smc_zones:
            query += " AND m.premium_discount_zone = ANY(%s)"
            params.append(smc_zones)

        # Add signal filter
        if has_signal is True:
            query += " AND COALESCE(sig.signal_count, 0) > 0"
        elif has_signal is False:
            query += " AND COALESCE(sig.signal_count, 0) = 0"

        # Add new to tier filter
        if is_new_to_tier is True:
            query += " AND w.is_new_to_tier = true"

        # Add earnings date filter
        if earnings_within_days is not None:
            query += f" AND i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE AND i.earnings_date <= CURRENT_DATE + INTERVAL '{int(earnings_within_days)} days'"

        # Add short interest filter
        if min_short_interest is not None:
            query += " AND COALESCE(i.short_percent_float, 0) >= %s"
            params.append(min_short_interest)

        # Enhanced signal filters
        if rsi_signals:
            query += " AND w.rsi_signal = ANY(%s)"
            params.append(rsi_signals)

        if sma_cross_signals:
            query += " AND w.sma_cross_signal = ANY(%s)"
            params.append(sma_cross_signals)

        if macd_cross_signals:
            query += " AND w.macd_cross_signal = ANY(%s)"
            params.append(macd_cross_signals)

        if high_low_signals:
            query += " AND w.high_low_signal = ANY(%s)"
            params.append(high_low_signals)

        if candlestick_patterns:
            query += " AND w.candlestick_pattern = ANY(%s)"
            params.append(candlestick_patterns)

        query += " ORDER BY w.rank_overall"

        results = _self._execute_query(query, tuple(params))
        return pd.DataFrame(results) if results else pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_upcoming_earnings(_self, days_ahead: int = 14) -> pd.DataFrame:
        """Get stocks with earnings in the next N days."""
        query = f"""
            SELECT
                i.ticker,
                i.name,
                i.earnings_date,
                i.earnings_date_estimated,
                i.beta,
                i.analyst_rating,
                i.target_price,
                w.tier,
                w.score,
                m.close as current_price,
                m.trend,
                COALESCE(m.smc_trend, '') as smc_trend
            FROM stock_instruments i
            LEFT JOIN stock_watchlist w ON i.ticker = w.ticker
                AND w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.earnings_date IS NOT NULL
            AND i.earnings_date >= CURRENT_DATE
            AND i.earnings_date <= CURRENT_DATE + INTERVAL '{int(days_ahead)} days'
            AND i.is_active = TRUE
            ORDER BY i.earnings_date, i.ticker
        """
        results = _self._execute_query(query, ())
        return pd.DataFrame(results) if results else pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_high_short_interest(_self, min_percent: float = 10.0) -> pd.DataFrame:
        """Get stocks with high short interest (squeeze potential)."""
        query = """
            SELECT
                i.ticker,
                i.name,
                i.short_ratio,
                i.short_percent_float,
                i.beta,
                w.tier,
                w.score,
                m.close as current_price,
                m.relative_volume,
                COALESCE(m.smc_trend, '') as smc_trend
            FROM stock_instruments i
            LEFT JOIN stock_watchlist w ON i.ticker = w.ticker
                AND w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.short_percent_float >= %s
            AND i.is_active = TRUE
            ORDER BY i.short_percent_float DESC
        """
        results = _self._execute_query(query, (min_percent,))
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
                m.current_price,
                COALESCE(m.smc_trend, '') as smc_trend,
                COALESCE(m.smc_bias, '') as smc_bias,
                COALESCE(m.premium_discount_zone, '') as smc_zone,
                COALESCE(m.smc_confluence_score, 0) as smc_score,
                CASE
                    WHEN s.signal_type = m.smc_bias THEN 'Aligned'
                    WHEN m.smc_bias IS NULL OR m.smc_bias = '' THEN 'Unknown'
                    ELSE 'Divergent'
                END as smc_alignment
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

    # =========================================================================
    # TOP PICKS QUERIES
    # =========================================================================

    @st.cache_data(ttl=60)  # Reduced TTL to allow Claude analysis to show sooner
    def get_daily_top_picks(_self) -> Dict[str, Any]:
        """
        Get daily top picks categorized by setup type.

        Returns picks in three categories:
        - momentum: Trending stocks with bullish confirmation
        - breakout: Stocks near 52W highs with volume surge
        - mean_reversion: Oversold stocks with reversal patterns
        """
        conn = _self._get_connection()
        if not conn:
            return {'momentum': [], 'breakout': [], 'mean_reversion': [], 'total_picks': 0}

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get latest calculation date
                cursor.execute("""
                    SELECT MAX(calculation_date) as latest_date
                    FROM stock_watchlist
                """)
                latest_date = cursor.fetchone()['latest_date']

                if not latest_date:
                    return {'momentum': [], 'breakout': [], 'mean_reversion': [], 'total_picks': 0, 'date': None}

                # Query for eligible candidates with hard exclusions built into SQL
                # Also pull Claude analysis from stock_scanner_signals as fallback
                cursor.execute("""
                    SELECT
                        w.ticker,
                        i.name,
                        w.tier,
                        w.score as watchlist_score,
                        w.current_price,
                        w.atr_percent,
                        w.relative_volume,
                        w.price_change_20d,
                        w.trend_strength,
                        w.rsi_signal,
                        w.sma20_signal,
                        w.sma50_signal,
                        w.sma_cross_signal,
                        w.macd_cross_signal,
                        w.high_low_signal,
                        w.gap_signal,
                        w.candlestick_pattern,
                        w.pct_from_52w_high,
                        w.calculation_date,
                        -- Claude analysis columns (prefer watchlist, fallback to scanner signals)
                        COALESCE(w.claude_grade, ss.claude_grade) as claude_grade,
                        COALESCE(w.claude_score, ss.claude_score) as claude_score,
                        COALESCE(w.claude_action, ss.claude_action) as claude_action,
                        COALESCE(w.claude_thesis, ss.claude_thesis) as claude_thesis,
                        COALESCE(w.claude_conviction, ss.claude_conviction) as claude_conviction,
                        COALESCE(w.claude_key_strengths, ss.claude_key_strengths) as claude_key_strengths,
                        COALESCE(w.claude_key_risks, ss.claude_key_risks) as claude_key_risks,
                        COALESCE(w.claude_position_rec, ss.claude_position_rec) as claude_position_rec,
                        COALESCE(w.claude_stop_adjustment, ss.claude_stop_adjustment) as claude_stop_adjustment,
                        COALESCE(w.claude_time_horizon, ss.claude_time_horizon) as claude_time_horizon,
                        COALESCE(w.claude_analyzed_at, ss.claude_analyzed_at) as claude_analyzed_at,
                        m.price_change_1d,
                        m.price_change_5d,
                        m.rsi_14,
                        m.macd_histogram,
                        m.perf_1w,
                        m.perf_1m
                    FROM stock_watchlist w
                    JOIN stock_instruments i ON w.ticker = i.ticker
                    LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
                        AND m.calculation_date = w.calculation_date
                    LEFT JOIN LATERAL (
                        SELECT claude_grade, claude_score, claude_action, claude_thesis,
                               claude_conviction, claude_key_strengths, claude_key_risks,
                               claude_position_rec, claude_stop_adjustment, claude_time_horizon,
                               claude_analyzed_at
                        FROM stock_scanner_signals
                        WHERE ticker = w.ticker
                        AND claude_grade IS NOT NULL
                        AND signal_timestamp > NOW() - INTERVAL '7 days'
                        ORDER BY signal_timestamp DESC
                        LIMIT 1
                    ) ss ON true
                    WHERE w.calculation_date = %s
                      -- Hard exclusions
                      AND w.tier <= 3  -- Exclude low liquidity
                      AND COALESCE(w.sma_cross_signal, '') != 'death_cross'  -- No death crosses
                      AND NOT (
                          COALESCE(w.rsi_signal, '') = 'overbought_extreme'
                          AND COALESCE(w.sma_cross_signal, '') = 'bearish'
                      )  -- No overbought extreme + bearish
                      AND NOT (
                          COALESCE(w.candlestick_pattern, '') IN ('bearish_engulfing', 'hanging_man', 'shooting_star', 'strong_bearish', 'bearish_marubozu')
                          AND COALESCE(w.rsi_signal, '') NOT IN ('oversold', 'oversold_extreme')
                      )  -- No bearish patterns unless oversold
                      AND NOT (
                          COALESCE(w.gap_signal, '') IN ('gap_down_large', 'gap_down')
                          AND COALESCE(w.relative_volume, 0) > 2.0
                      )  -- No panic selling
                      AND NOT (
                          COALESCE(w.high_low_signal, '') = 'new_low'
                          AND COALESCE(w.sma_cross_signal, '') = 'bearish'
                      )  -- No falling knives
                    ORDER BY w.score DESC
                """, (latest_date,))

                candidates = [dict(row) for row in cursor.fetchall()]

                if not candidates:
                    return {
                        'momentum': [], 'breakout': [], 'mean_reversion': [],
                        'total_picks': 0, 'date': str(latest_date)
                    }

                # Score and categorize candidates
                momentum_picks = []
                breakout_picks = []
                reversion_picks = []

                for c in candidates:
                    pick = _self._score_candidate(c)
                    if pick:
                        if pick['category'] == 'Momentum':
                            momentum_picks.append(pick)
                        elif pick['category'] == 'Breakout':
                            breakout_picks.append(pick)
                        elif pick['category'] == 'Mean Reversion':
                            reversion_picks.append(pick)

                # Sort by score and apply limits
                momentum_picks.sort(key=lambda x: x['total_score'], reverse=True)
                breakout_picks.sort(key=lambda x: x['total_score'], reverse=True)
                reversion_picks.sort(key=lambda x: x['total_score'], reverse=True)

                # Apply limits (7 momentum, 6 breakout, 5 reversion)
                momentum_final = [p for p in momentum_picks[:7] if p['total_score'] >= 50]
                breakout_final = [p for p in breakout_picks[:6] if p['total_score'] >= 55]
                reversion_final = [p for p in reversion_picks[:5] if p['total_score'] >= 45]

                # Assign ranks
                for i, p in enumerate(momentum_final, 1):
                    p['rank'] = i
                for i, p in enumerate(breakout_final, 1):
                    p['rank'] = i
                for i, p in enumerate(reversion_final, 1):
                    p['rank'] = i

                return {
                    'date': str(latest_date),
                    'momentum': momentum_final,
                    'breakout': breakout_final,
                    'mean_reversion': reversion_final,
                    'total_picks': len(momentum_final) + len(breakout_final) + len(reversion_final),
                    'stats': {
                        'candidates_analyzed': len(candidates),
                        'avg_score_momentum': round(sum(p['total_score'] for p in momentum_final) / len(momentum_final), 1) if momentum_final else 0,
                        'avg_score_breakout': round(sum(p['total_score'] for p in breakout_final) / len(breakout_final), 1) if breakout_final else 0,
                        'avg_score_reversion': round(sum(p['total_score'] for p in reversion_final) / len(reversion_final), 1) if reversion_final else 0,
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get daily top picks: {e}")
            return {'momentum': [], 'breakout': [], 'mean_reversion': [], 'total_picks': 0}
        finally:
            conn.close()

    def _score_candidate(_self, candidate: Dict) -> Optional[Dict]:
        """Score a candidate and determine its category."""
        # Extract values
        tier = candidate.get('tier', 4)
        relative_volume = float(candidate.get('relative_volume') or 0)
        atr_pct = float(candidate.get('atr_percent') or 0)
        rsi_signal = candidate.get('rsi_signal') or ''
        sma_cross_signal = candidate.get('sma_cross_signal') or ''
        macd_cross_signal = candidate.get('macd_cross_signal') or ''
        high_low_signal = candidate.get('high_low_signal') or ''
        gap_signal = candidate.get('gap_signal') or ''
        candlestick_pattern = candidate.get('candlestick_pattern') or ''
        price_change_1d = float(candidate.get('price_change_1d') or 0)
        price_change_5d = float(candidate.get('price_change_5d') or 0)

        # === BASE SCORE (40 points max) ===
        base_score = 0.0
        tier_scores = {1: 20, 2: 15, 3: 10}
        base_score += tier_scores.get(tier, 0)

        # Volume score (0-10 pts)
        if relative_volume >= 1.5:
            base_score += 10
        elif relative_volume >= 1.2:
            base_score += 5
        elif relative_volume >= 1.0:
            base_score += 2

        # ATR sweet spot (0-10 pts)
        if 2.0 <= atr_pct <= 8.0:
            base_score += 10
        elif atr_pct > 0:
            base_score += 5

        # === SIGNAL SCORE (40 points max) ===
        signal_score = 0.0
        signal_count = 0

        # Trend signals (0-15 pts)
        if sma_cross_signal == 'golden_cross':
            signal_score += 12
            signal_count += 1
        elif sma_cross_signal == 'bullish':
            signal_score += 8
            signal_count += 1

        # Momentum signals (0-15 pts)
        if macd_cross_signal == 'bullish_cross':
            signal_score += 10
            signal_count += 1
        elif macd_cross_signal == 'bullish':
            signal_score += 6
            signal_count += 1

        # Position signals (0-10 pts)
        if high_low_signal in ['new_high', 'near_high']:
            signal_score += 10
            signal_count += 1
        elif rsi_signal in ['oversold', 'oversold_extreme'] and high_low_signal != 'new_low':
            signal_score += 8
            signal_count += 1

        # Pattern confirmation (0-10 pts)
        bullish_patterns = ['bullish_engulfing', 'hammer', 'dragonfly_doji', 'bullish_marubozu', 'strong_bullish', 'inverted_hammer']
        if candlestick_pattern in ['bullish_engulfing', 'bullish_marubozu']:
            signal_score += 10
            signal_count += 1
        elif candlestick_pattern in bullish_patterns:
            signal_score += 7
            signal_count += 1

        # Gap bonus (0-5 pts)
        if gap_signal == 'gap_up_large':
            signal_score += 5
        elif gap_signal == 'gap_up':
            signal_score += 3

        # === CONFLUENCE BONUS (20 points max) ===
        if signal_count >= 4:
            confluence_bonus = 20.0
        elif signal_count >= 3:
            confluence_bonus = 12.0
        elif signal_count >= 2:
            confluence_bonus = 5.0
        else:
            confluence_bonus = 0.0

        total_score = base_score + signal_score + confluence_bonus

        # === DETERMINE CATEGORY ===
        category = None

        # Breakout: Near highs + volume surge + gap
        if (high_low_signal in ['near_high', 'new_high']
            and relative_volume >= 1.3
            and (gap_signal in ['gap_up', 'gap_up_large'] or sma_cross_signal == 'golden_cross')):
            category = 'Breakout'
        # Momentum: Bullish trend + not overbought extreme
        elif (sma_cross_signal in ['golden_cross', 'bullish']
            and rsi_signal not in ['overbought_extreme']
            and macd_cross_signal in ['bullish_cross', 'bullish']
            and high_low_signal not in ['new_low', 'near_low']):
            category = 'Momentum'
        # Mean Reversion: Oversold + reversal pattern
        elif (rsi_signal in ['oversold', 'oversold_extreme']
            and sma_cross_signal not in ['death_cross', 'bearish']
            and candlestick_pattern in bullish_patterns):
            category = 'Mean Reversion'
        # Secondary Momentum
        elif (sma_cross_signal in ['golden_cross', 'bullish']
            and price_change_5d > 2.0
            and rsi_signal not in ['overbought_extreme']):
            category = 'Momentum'
        # Secondary Mean Reversion
        elif (rsi_signal in ['oversold', 'oversold_extreme']
            and sma_cross_signal == 'bullish'
            and high_low_signal != 'new_low'):
            category = 'Mean Reversion'

        if category is None:
            return None

        # Build signals summary
        signals = []
        if sma_cross_signal in ['golden_cross', 'bullish']:
            signals.append('SMA Bullish' if sma_cross_signal == 'bullish' else 'Golden Cross')
        if macd_cross_signal in ['bullish_cross', 'bullish']:
            signals.append('MACD Bullish' if macd_cross_signal == 'bullish' else 'MACD Cross')
        if high_low_signal in ['new_high', 'near_high']:
            signals.append('Near 52W High' if high_low_signal == 'near_high' else 'New 52W High')
        if rsi_signal in ['oversold', 'oversold_extreme']:
            signals.append('Oversold')
        if candlestick_pattern in bullish_patterns:
            signals.append(candlestick_pattern.replace('_', ' ').title())
        if gap_signal in ['gap_up', 'gap_up_large']:
            signals.append('Gap Up')

        # Calculate suggested stop
        suggested_stop_pct = min(atr_pct * 1.5, 8.0)

        pick = {
            'ticker': candidate['ticker'],
            'name': candidate.get('name', ''),
            'category': category,
            'total_score': round(total_score, 2),
            'base_score': round(base_score, 2),
            'signal_score': round(signal_score, 2),
            'confluence_bonus': round(confluence_bonus, 2),
            'signal_count': signal_count,
            'signals_summary': ', '.join(signals) if signals else 'Mixed',
            'current_price': float(candidate.get('current_price') or 0),
            'price_change_1d': price_change_1d,
            'price_change_5d': price_change_5d,
            'atr_percent': atr_pct,
            'relative_volume': relative_volume,
            'tier': tier,
            'rsi_signal': rsi_signal,
            'sma_cross_signal': sma_cross_signal,
            'macd_cross_signal': macd_cross_signal,
            'high_low_signal': high_low_signal,
            'gap_signal': gap_signal,
            'candlestick_pattern': candlestick_pattern,
            'suggested_stop_pct': round(suggested_stop_pct, 2),
            'risk_reward_ratio': round(2.0 / suggested_stop_pct * atr_pct, 2) if suggested_stop_pct > 0 else 0,
            # Include calculation_date for Claude analysis lookup
            'calculation_date': candidate.get('calculation_date'),
            # Include existing Claude analysis from database if available
            'claude_grade': candidate.get('claude_grade'),
            'claude_score': candidate.get('claude_score'),
            'claude_action': candidate.get('claude_action'),
            'claude_thesis': candidate.get('claude_thesis'),
            'claude_conviction': candidate.get('claude_conviction'),
            'claude_key_strengths': candidate.get('claude_key_strengths'),
            'claude_key_risks': candidate.get('claude_key_risks'),
            'claude_position_rec': candidate.get('claude_position_rec'),
            'claude_stop_adjustment': candidate.get('claude_stop_adjustment'),
            'claude_time_horizon': candidate.get('claude_time_horizon'),
            'claude_analyzed_at': candidate.get('claude_analyzed_at'),
        }
        return pick

    # =========================================================================
    # CLAUDE ANALYSIS FOR TOP PICKS
    # =========================================================================

    def _get_claude_api_key(_self) -> Optional[str]:
        """Get Claude API key from secrets or environment."""
        import os
        # Try streamlit secrets first (under [api] section)
        try:
            return st.secrets.api.claude_api_key
        except Exception:
            pass
        # Try root level
        try:
            return st.secrets.get("CLAUDE_API_KEY")
        except Exception:
            pass
        # Fallback to environment variable
        return os.environ.get('CLAUDE_API_KEY')

    def _get_stored_claude_analysis(_self, ticker: str, calculation_date) -> Optional[Dict[str, Any]]:
        """Check if Claude analysis already exists in database for this ticker/date."""
        conn = _self._get_connection()
        if not conn:
            return None

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT claude_grade, claude_score, claude_action, claude_thesis,
                           claude_conviction, claude_key_strengths, claude_key_risks,
                           claude_position_rec, claude_stop_adjustment, claude_time_horizon,
                           claude_analyzed_at
                    FROM stock_watchlist
                    WHERE ticker = %s AND calculation_date = %s
                      AND claude_analyzed_at IS NOT NULL
                """, (ticker, calculation_date))
                row = cursor.fetchone()
                if row and row['claude_grade']:
                    return {
                        'success': True,
                        'claude_grade': row['claude_grade'],
                        'claude_score': row['claude_score'],
                        'claude_action': row['claude_action'],
                        'claude_thesis': row['claude_thesis'],
                        'claude_conviction': row['claude_conviction'],
                        'claude_key_strengths': row['claude_key_strengths'] or [],
                        'claude_key_risks': row['claude_key_risks'] or [],
                        'claude_position_rec': row['claude_position_rec'],
                        'claude_stop_adjustment': row['claude_stop_adjustment'],
                        'claude_time_horizon': row['claude_time_horizon'],
                        'from_cache': True
                    }
        except Exception as e:
            logger.error(f"Error checking stored Claude analysis: {e}")
        finally:
            conn.close()
        return None

    def get_latest_claude_analysis_from_watchlist(_self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent Claude analysis for a ticker from stock_watchlist.

        This retrieves analysis done via Top Picks so it can be displayed in Deep Dive.
        Returns the most recent analysis regardless of calculation_date.
        """
        conn = _self._get_connection()
        if not conn:
            return None

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT claude_grade, claude_score, claude_action, claude_thesis,
                           claude_conviction, claude_key_strengths, claude_key_risks,
                           claude_position_rec, claude_stop_adjustment, claude_time_horizon,
                           claude_analyzed_at, calculation_date
                    FROM stock_watchlist
                    WHERE ticker = %s AND claude_analyzed_at IS NOT NULL
                    ORDER BY claude_analyzed_at DESC
                    LIMIT 1
                """, (ticker,))
                row = cursor.fetchone()
                if row and row['claude_grade']:
                    return {
                        'success': True,
                        'rating': row['claude_grade'],
                        'confidence_score': row['claude_score'],
                        'recommendation': row['claude_action'],
                        'thesis': row['claude_thesis'],
                        'conviction': row['claude_conviction'],
                        'key_factors': row['claude_key_strengths'] or [],
                        'risk_assessment': row['claude_key_risks'] or [],
                        'position_sizing': row['claude_position_rec'],
                        'stop_adjustment': row['claude_stop_adjustment'],
                        'time_horizon': row['claude_time_horizon'],
                        'analyzed_at': row['claude_analyzed_at'],
                        'calculation_date': row['calculation_date'],
                        'source': 'top_picks'
                    }
        except Exception as e:
            logger.error(f"Error fetching Claude analysis from watchlist for {ticker}: {e}")
        finally:
            conn.close()
        return None

    def _store_claude_analysis(_self, ticker: str, calculation_date, analysis: Dict[str, Any]) -> bool:
        """Store Claude analysis in database."""
        conn = _self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE stock_watchlist
                    SET claude_grade = %s,
                        claude_score = %s,
                        claude_action = %s,
                        claude_thesis = %s,
                        claude_conviction = %s,
                        claude_key_strengths = %s,
                        claude_key_risks = %s,
                        claude_position_rec = %s,
                        claude_stop_adjustment = %s,
                        claude_time_horizon = %s,
                        claude_tokens_used = %s,
                        claude_model = %s,
                        claude_analyzed_at = NOW()
                    WHERE ticker = %s AND calculation_date = %s
                """, (
                    analysis.get('claude_grade'),
                    analysis.get('claude_score'),
                    analysis.get('claude_action'),
                    analysis.get('claude_thesis'),
                    analysis.get('claude_conviction'),
                    analysis.get('claude_key_strengths', []),
                    analysis.get('claude_key_risks', []),
                    analysis.get('claude_position_rec'),
                    analysis.get('claude_stop_adjustment'),
                    analysis.get('claude_time_horizon'),
                    analysis.get('claude_tokens_used', 0),
                    analysis.get('claude_model', ''),
                    ticker,
                    calculation_date
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing Claude analysis for {ticker}: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def _build_standard_analysis_prompt(_self, pick: Dict[str, Any], technical: Dict[str, Any]) -> str:
        """Build a standard institutional-grade analysis prompt matching the scanner approach."""
        ticker = pick.get('ticker', 'UNKNOWN')
        name = pick.get('name', ticker)
        category = pick.get('category', 'unknown')
        total_score = pick.get('total_score', 0)
        tier = pick.get('tier', 3)
        tier_label = {1: 'A (High Liquidity)', 2: 'B (Medium)', 3: 'C (Lower)'}.get(tier, 'C')

        # Price info
        current_price = pick.get('current_price', 0)
        price_change_1d = pick.get('price_change_1d', 0)
        price_change_5d = pick.get('price_change_5d', 0)

        # Risk parameters
        suggested_stop_pct = pick.get('suggested_stop_pct', 5.0)
        stop_price = current_price * (1 - suggested_stop_pct / 100)
        target_price = current_price * 1.10  # 10% target
        rr_ratio = pick.get('risk_reward_ratio', 2.0)

        # Technical section
        rsi_signal = pick.get('rsi_signal', 'neutral')
        sma_cross = pick.get('sma_cross_signal', 'neutral')
        macd_cross = pick.get('macd_cross_signal', 'neutral')
        high_low = pick.get('high_low_signal', 'neutral')
        gap_signal = pick.get('gap_signal', 'none')
        pattern = pick.get('candlestick_pattern', 'none')
        rel_vol = pick.get('relative_volume', 1.0)
        atr_pct = pick.get('atr_percent', 0)

        # Additional technical from metrics
        rsi_14 = technical.get('rsi_14', 50)
        macd_hist = technical.get('macd_histogram', 0)

        # Signals summary
        signals_summary = pick.get('signals_summary', '')

        prompt = f"""You are a Senior Equity Analyst. Analyze this stock signal with institutional rigor.

## SIGNAL OVERVIEW
**{ticker}** ({name}) | BUY Signal
Category: {category} | Quality: {tier_label} ({total_score}/100)

## RISK PARAMETERS
Entry: ${current_price:.2f} | Stop: ${stop_price:.2f} (-{suggested_stop_pct:.1f}%)
Target: ${target_price:.2f} (+10%) | Risk/Reward: {rr_ratio:.1f}:1

## TECHNICAL ANALYSIS
Price: ${current_price:.2f} | 1D: {price_change_1d:+.1f}% | 5D: {price_change_5d:+.1f}%
RSI(14): {rsi_14:.0f} ({rsi_signal}) | MACD Hist: {macd_hist:+.3f} ({macd_cross})
Volume: {rel_vol:.1f}x relative | ATR: {atr_pct:.1f}%
MA Cross: {sma_cross} | 52W Position: {high_low} | Gap: {gap_signal}
Pattern: {pattern}

## CONFLUENCE FACTORS
{signals_summary}

---
Analyze and respond in this exact JSON format:
{{
  "grade": "A+/A/B/C/D",
  "score": 1-10,
  "conviction": "HIGH/MEDIUM/LOW",
  "action": "STRONG BUY/BUY/HOLD/AVOID",
  "key_strengths": ["strength1", "strength2"],
  "key_risks": ["risk1", "risk2"],
  "thesis": "2-3 sentence investment thesis explaining your reasoning",
  "position_recommendation": "Full/Half/Quarter/Skip",
  "stop_adjustment": "Tighten/Keep/Widen",
  "time_horizon": "Intraday/Swing/Position"
}}"""
        return prompt

    def _parse_claude_response(_self, response_text: str) -> Dict[str, Any]:
        """Parse Claude API response into structured dict."""
        import json
        import re

        # Try to extract JSON from response
        json_data = None

        # Try direct JSON parse first
        try:
            json_data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        if not json_data:
            patterns = [
                r'```json\s*([\s\S]*?)\s*```',
                r'```\s*([\s\S]*?)\s*```',
                r'\{[\s\S]*\}',
            ]
            for pattern in patterns:
                match = re.search(pattern, response_text)
                if match:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    try:
                        json_data = json.loads(json_str.strip())
                        break
                    except json.JSONDecodeError:
                        continue

        if not json_data:
            return {'success': False, 'error': 'Failed to parse JSON response'}

        # Normalize and validate
        grade = str(json_data.get('grade', 'C')).upper().strip()
        if grade not in ['A+', 'A', 'B', 'C', 'D']:
            if grade.startswith('A') and '+' in grade:
                grade = 'A+'
            elif grade.startswith('A'):
                grade = 'A'
            elif grade.startswith('B'):
                grade = 'B'
            elif grade.startswith('D'):
                grade = 'D'
            else:
                grade = 'C'

        try:
            score = max(1, min(10, int(json_data.get('score', 5))))
        except (ValueError, TypeError):
            score = 5

        action = str(json_data.get('action', 'HOLD')).upper().strip()
        if action not in ['STRONG BUY', 'BUY', 'HOLD', 'AVOID']:
            if 'STRONG' in action and 'BUY' in action:
                action = 'STRONG BUY'
            elif 'BUY' in action:
                action = 'BUY'
            elif 'AVOID' in action or 'SELL' in action:
                action = 'AVOID'
            else:
                action = 'HOLD'

        conviction = str(json_data.get('conviction', 'MEDIUM')).upper().strip()
        if conviction not in ['HIGH', 'MEDIUM', 'LOW']:
            conviction = 'MEDIUM'

        # Extract lists
        strengths = json_data.get('key_strengths', [])
        if isinstance(strengths, str):
            strengths = [strengths]
        risks = json_data.get('key_risks', [])
        if isinstance(risks, str):
            risks = [risks]

        # Position recommendation
        pos_rec = str(json_data.get('position_recommendation', 'Quarter')).title()
        if pos_rec not in ['Full', 'Half', 'Quarter', 'Skip']:
            pos_rec = 'Quarter'

        # Stop adjustment
        stop_adj = str(json_data.get('stop_adjustment', 'Keep')).title()
        if stop_adj not in ['Tighten', 'Keep', 'Widen']:
            stop_adj = 'Keep'

        # Time horizon
        horizon = str(json_data.get('time_horizon', 'Swing')).title()
        if horizon not in ['Intraday', 'Swing', 'Position']:
            horizon = 'Swing'

        return {
            'success': True,
            'claude_grade': grade,
            'claude_score': score,
            'claude_action': action,
            'claude_thesis': str(json_data.get('thesis', ''))[:500],
            'claude_conviction': conviction,
            'claude_key_strengths': [str(s) for s in strengths[:5]],
            'claude_key_risks': [str(r) for r in risks[:5]],
            'claude_position_rec': pos_rec,
            'claude_stop_adjustment': stop_adj,
            'claude_time_horizon': horizon,
        }

    def _get_technical_data_for_pick(_self, ticker: str, calculation_date) -> Dict[str, Any]:
        """Fetch additional technical data from screening metrics."""
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT rsi_14, macd_histogram, price_change_1d, price_change_5d,
                           perf_1w, perf_1m, relative_volume
                    FROM stock_screening_metrics
                    WHERE ticker = %s AND calculation_date = %s
                """, (ticker, calculation_date))
                row = cursor.fetchone()
                return dict(row) if row else {}
        except Exception as e:
            logger.error(f"Error fetching technical data for {ticker}: {e}")
            return {}
        finally:
            conn.close()

    def analyze_top_pick_with_claude(_self, pick: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze a top pick using Claude AI (same approach as scanner).

        Uses Sonnet model with standard analysis level for institutional-grade analysis.
        First checks database for existing analysis. If not found (or force_refresh=True),
        calls Claude API and stores the result.

        Args:
            pick: Top pick data including ticker, signals, and technical data
            force_refresh: If True, skip cache and call Claude API

        Returns:
            Dict with Claude analysis (grade, score, action, thesis, strengths, risks, etc.)
        """
        from datetime import date

        ticker = pick.get('ticker', '')
        if not ticker:
            return {'success': False, 'error': 'No ticker provided'}

        # Get calculation date from pick or use yesterday
        calculation_date = pick.get('calculation_date')
        if not calculation_date:
            calculation_date = date.today() - timedelta(days=1)

        # Check database first (unless force refresh)
        if not force_refresh:
            stored = _self._get_stored_claude_analysis(ticker, calculation_date)
            if stored:
                logger.info(f"Using cached Claude analysis for {ticker}")
                return stored

        # Get API key
        api_key = _self._get_claude_api_key()
        if not api_key:
            return {'success': False, 'error': 'Claude API key not configured'}

        try:
            # Try to import anthropic
            try:
                import anthropic
            except ImportError:
                return {'success': False, 'error': 'anthropic package not installed. Run: pip install anthropic'}

            # Fetch additional technical data
            technical = _self._get_technical_data_for_pick(ticker, calculation_date)

            # Build the standard institutional-grade prompt (same as scanner)
            prompt = _self._build_standard_analysis_prompt(pick, technical)

            # Call Claude API with Sonnet model (same as scanner standard analysis)
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",  # Use Sonnet for standard analysis (same as scanner)
                max_tokens=300,  # Standard analysis token limit
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response using robust parser
            response_text = response.content[0].text.strip()
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            analysis = _self._parse_claude_response(response_text)

            if analysis.get('success'):
                # Add token tracking
                analysis['claude_tokens_used'] = tokens_used
                analysis['claude_model'] = 'claude-sonnet-4-20250514'

                # Store in database for future use
                _self._store_claude_analysis(ticker, calculation_date, analysis)
                logger.info(f"Stored Claude analysis for {ticker} (Sonnet, {tokens_used} tokens)")

                return analysis
            else:
                return analysis

        except Exception as e:
            logger.error(f"Failed to analyze top pick {ticker}: {e}")
            return {'success': False, 'error': str(e)[:100]}

    def batch_analyze_top_picks(_self, picks: List[Dict[str, Any]], max_picks: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Batch analyze multiple top picks with Claude.

        Args:
            picks: List of top pick dicts
            max_picks: Maximum picks to analyze (to control costs)

        Returns:
            Dict mapping ticker to Claude analysis result
        """
        results = {}
        for i, pick in enumerate(picks[:max_picks]):
            ticker = pick.get('ticker', '')
            if ticker:
                results[ticker] = _self.analyze_top_pick_with_claude(pick)
        return results

    # =========================================================================
    # SCANNER SIGNALS
    # =========================================================================

    @st.cache_data(ttl=60, show_spinner=False)
    def get_scanner_signals(
        _self,
        scanner_name: str = None,
        status: str = 'active',
        min_score: int = None,
        min_claude_grade: str = None,
        claude_analyzed_only: bool = False,
        signal_date_from: str = None,
        signal_date_to: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get scanner signals from database.

        Args:
            scanner_name: Filter by specific scanner
            status: Filter by status ('active', 'triggered', 'closed', etc.)
            min_score: Minimum composite score
            min_claude_grade: Minimum Claude grade (A+, A, B, C, D)
            claude_analyzed_only: Only show signals with Claude analysis
            signal_date_from: Filter signals from this date (str: YYYY-MM-DD)
            signal_date_to: Filter signals up to this date (str: YYYY-MM-DD)
            limit: Maximum results

        Returns:
            List of signal dictionaries
        """
        conditions = []
        params = []

        if scanner_name:
            params.append(scanner_name)
            conditions.append("scanner_name = %s")

        if status:
            params.append(status)
            conditions.append("status = %s")

        if min_score:
            params.append(min_score)
            conditions.append("composite_score >= %s")

        if claude_analyzed_only:
            conditions.append("claude_analyzed_at IS NOT NULL")

        if min_claude_grade:
            grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
            valid_grades = [g for g, v in grade_order.items() if v >= grade_order.get(min_claude_grade, 1)]
            grades_str = ', '.join(f"'{g}'" for g in valid_grades)
            conditions.append(f"claude_grade IN ({grades_str})")

        # Date range filters
        if signal_date_from:
            params.append(signal_date_from)
            conditions.append("DATE(signal_timestamp) >= %s")

        if signal_date_to:
            params.append(signal_date_to)
            conditions.append("DATE(signal_timestamp) <= %s")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Use a subquery to get only the latest signal per ticker/scanner combination
        # This prevents showing stale signals when a new scan has been run
        query = f"""
            WITH latest_signals AS (
                SELECT DISTINCT ON (ticker, scanner_name)
                    *
                FROM stock_scanner_signals
                WHERE {where_clause}
                ORDER BY ticker, scanner_name, signal_timestamp DESC
            )
            SELECT
                s.*,
                i.name as company_name
            FROM latest_signals s
            LEFT JOIN stock_instruments i ON s.ticker = i.ticker
            ORDER BY
                CASE WHEN s.claude_analyzed_at IS NOT NULL THEN 0 ELSE 1 END,
                COALESCE(s.claude_score, 0) DESC,
                s.composite_score DESC,
                s.signal_timestamp DESC
            LIMIT {limit}
        """

        return _self._execute_query(query, tuple(params) if params else None)

    @st.cache_data(ttl=60)
    def get_scanner_stats(_self) -> Dict[str, Any]:
        """Get statistics about scanner signals."""
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Total signals with Claude analysis stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_signals,
                        COUNT(*) FILTER (WHERE status = 'active') as active_signals,
                        COUNT(*) FILTER (WHERE quality_tier IN ('A+', 'A')) as high_quality,
                        COUNT(*) FILTER (WHERE DATE(signal_timestamp) = CURRENT_DATE) as today_signals,
                        COUNT(*) FILTER (WHERE claude_analyzed_at IS NOT NULL) as claude_analyzed,
                        COUNT(*) FILTER (WHERE claude_grade IN ('A+', 'A')) as claude_high_grade,
                        COUNT(*) FILTER (WHERE claude_action = 'STRONG BUY') as claude_strong_buys,
                        COUNT(*) FILTER (WHERE claude_action = 'BUY') as claude_buys,
                        COUNT(*) FILTER (WHERE claude_analyzed_at IS NULL AND status = 'active') as awaiting_analysis
                    FROM stock_scanner_signals
                """)
                totals = dict(cursor.fetchone())

                # By scanner
                cursor.execute("""
                    SELECT
                        scanner_name,
                        COUNT(*) as signal_count,
                        ROUND(AVG(composite_score)::numeric, 1) as avg_score,
                        COUNT(*) FILTER (WHERE status = 'active') as active_count
                    FROM stock_scanner_signals
                    GROUP BY scanner_name
                    ORDER BY signal_count DESC
                """)
                by_scanner = [dict(r) for r in cursor.fetchall()]

                # By tier
                cursor.execute("""
                    SELECT
                        quality_tier,
                        COUNT(*) as count
                    FROM stock_scanner_signals
                    WHERE status = 'active'
                    GROUP BY quality_tier
                    ORDER BY
                        CASE quality_tier
                            WHEN 'A+' THEN 1
                            WHEN 'A' THEN 2
                            WHEN 'B' THEN 3
                            WHEN 'C' THEN 4
                            WHEN 'D' THEN 5
                        END
                """)
                by_tier = {r['quality_tier']: r['count'] for r in cursor.fetchall()}

                return {
                    **totals,
                    'by_scanner': by_scanner,
                    'by_tier': by_tier
                }

        except Exception as e:
            logger.error(f"Failed to get scanner stats: {e}")
            return {}
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_scanner_performance(_self, days: int = 30) -> List[Dict[str, Any]]:
        """Get scanner performance history."""
        query = """
            SELECT *
            FROM stock_scanner_performance
            WHERE evaluation_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY evaluation_date DESC, scanner_name
        """
        return _self._execute_query(query, (days,))

    def get_signal_details(_self, signal_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific signal."""
        results = _self._execute_query("""
            SELECT
                s.*,
                i.name as company_name,
                i.sector
            FROM stock_scanner_signals s
            LEFT JOIN stock_instruments i ON s.ticker = i.ticker
            WHERE s.id = %s
        """, (signal_id,))
        return results[0] if results else {}

    def reanalyze_signal(_self, signal_id: int) -> Dict[str, Any]:
        """
        Trigger Claude AI re-analysis for a specific signal.

        This calls the worker container's batch analysis script for a single signal.

        Args:
            signal_id: The signal ID to re-analyze

        Returns:
            Dict with success status and message
        """
        import subprocess
        import json

        try:
            # First, clear the existing Claude analysis
            conn = _self._get_connection()
            if not conn:
                return {'success': False, 'error': 'Database connection failed'}

            with conn.cursor() as cursor:
                # Clear existing analysis to mark it for re-analysis
                cursor.execute("""
                    UPDATE stock_scanner_signals
                    SET claude_analyzed_at = NULL,
                        claude_grade = NULL,
                        claude_score = NULL,
                        claude_action = NULL,
                        claude_thesis = NULL,
                        claude_key_strengths = NULL,
                        claude_key_risks = NULL,
                        claude_conviction = NULL,
                        claude_position_rec = NULL,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING ticker
                """, (signal_id,))
                result = cursor.fetchone()
                conn.commit()

                if not result:
                    return {'success': False, 'error': f'Signal {signal_id} not found'}

                ticker = result[0]

            conn.close()

            # Run the batch analysis script for this specific signal
            # The script will pick it up since claude_analyzed_at is now NULL
            cmd = [
                'docker', 'exec', 'task-worker',
                'python', '-m', 'stock_scanner.scripts.run_batch_claude_analysis',
                '--limit', '1'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'message': f'Re-analysis triggered for {ticker}',
                    'ticker': ticker
                }
            else:
                return {
                    'success': False,
                    'error': f'Analysis failed: {result.stderr[:200]}'
                }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Analysis timed out after 2 minutes'}
        except Exception as e:
            logger.error(f"Failed to re-analyze signal {signal_id}: {e}")
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # DEEP DIVE TAB METHODS
    # =========================================================================

    @st.cache_data(ttl=300)
    def get_scanner_signals_for_ticker(_self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get scanner signals with Claude analysis for a specific ticker.

        Returns signals from stock_scanner_signals table including all
        Claude analysis fields (grade, thesis, strengths, risks, etc.)
        """
        query = """
            SELECT
                s.*,
                i.name as company_name,
                i.sector
            FROM stock_scanner_signals s
            LEFT JOIN stock_instruments i ON s.ticker = i.ticker
            WHERE s.ticker = %s
            ORDER BY s.signal_timestamp DESC
            LIMIT %s
        """
        return _self._execute_query(query, (ticker, limit))

    @st.cache_data(ttl=300)
    def get_full_fundamentals(_self, ticker: str) -> Dict[str, Any]:
        """
        Get all fundamental data from stock_instruments for Deep Dive display.

        Returns 50+ columns including valuation, growth, profitability,
        risk metrics, ownership, and analyst data.
        """
        query = """
            SELECT *
            FROM stock_instruments
            WHERE ticker = %s
        """
        results = _self._execute_query(query, (ticker,))
        return results[0] if results else {}

    def analyze_stock_with_claude(
        _self,
        ticker: str,
        signal: Dict[str, Any],
        metrics: Dict[str, Any],
        fundamentals: Dict[str, Any],
        candles: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Direct Claude API call with chart vision for stock analysis.

        This method:
        1. Generates a professional chart image with MAs and indicators
        2. Builds a comprehensive analysis prompt
        3. Calls Claude API with vision
        4. Parses response and saves to database
        5. Returns analysis dict

        Args:
            ticker: Stock ticker symbol
            signal: Signal dict with entry/stop/target
            metrics: Technical metrics from stock_screening_metrics
            fundamentals: Fundamental data from stock_instruments
            candles: OHLCV DataFrame for chart generation

        Returns:
            Dict with grade, score, action, thesis, strengths, risks, etc.
        """
        import anthropic
        import base64
        import io
        import json
        import re

        try:
            # Get API key from secrets (try multiple locations)
            api_key = None
            try:
                api_key = st.secrets.get("CLAUDE_API_KEY")  # Top level
            except Exception:
                pass
            if not api_key:
                try:
                    api_key = st.secrets.api.claude_api_key  # Under [api] section
                except Exception:
                    pass
            if not api_key:
                return {'error': 'CLAUDE_API_KEY not configured in secrets'}

            # 1. Generate chart image
            chart_base64 = _self._generate_analysis_chart(ticker, candles, signal, metrics)

            # 2. Build analysis prompt
            prompt = _self._build_deep_dive_prompt(ticker, signal, metrics, fundamentals)

            # 3. Call Claude API with vision
            client = anthropic.Anthropic(api_key=api_key)

            if chart_base64:
                message_content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": chart_base64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            else:
                message_content = [{"type": "text", "text": prompt}]

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                messages=[{"role": "user", "content": message_content}]
            )

            # 4. Parse response
            response_text = response.content[0].text if response.content else ''
            analysis = _self._parse_claude_analysis_response(response_text)

            # 5. Save to database if signal has an ID
            signal_id = signal.get('id')
            if signal_id and analysis.get('grade'):
                _self._save_claude_analysis(signal_id, analysis)

            # Add metadata
            analysis['tokens_used'] = response.usage.input_tokens + response.usage.output_tokens
            analysis['analyzed_at'] = datetime.now().isoformat()
            analysis['has_chart'] = chart_base64 is not None

            return analysis

        except anthropic.RateLimitError:
            return {'error': 'Claude API rate limit exceeded. Please try again later.'}
        except anthropic.APIError as e:
            return {'error': f'Claude API error: {str(e)}'}
        except Exception as e:
            logger.error(f"Failed to analyze {ticker} with Claude: {e}")
            return {'error': str(e)}

    def _generate_analysis_chart(
        _self,
        ticker: str,
        candles: pd.DataFrame,
        signal: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a professional chart image for Claude vision analysis.

        Creates a chart with:
        - Candlesticks
        - SMA 20/50/200 moving averages
        - Entry/Stop/Target horizontal lines
        - Volume bars
        - RSI indicator (if space allows)

        Returns base64-encoded PNG string.
        """
        try:
            import mplfinance as mpf
            import matplotlib.pyplot as plt
            from io import BytesIO
            import base64

            if candles.empty:
                return None

            # Prepare data for mplfinance
            df = candles.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Calculate moving averages
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()

            # Build addplots for moving averages
            addplots = []

            if df['SMA20'].notna().any():
                addplots.append(mpf.make_addplot(df['SMA20'], color='blue', width=1, label='SMA20'))
            if df['SMA50'].notna().any():
                addplots.append(mpf.make_addplot(df['SMA50'], color='orange', width=1, label='SMA50'))
            if df['SMA200'].notna().any():
                addplots.append(mpf.make_addplot(df['SMA200'], color='purple', width=1, label='SMA200'))

            # Add signal levels as horizontal lines
            hlines_dict = {'hlines': [], 'colors': [], 'linestyle': [], 'linewidths': []}

            if signal:
                entry = signal.get('entry_price')
                stop = signal.get('stop_loss')
                target = signal.get('take_profit_1') or signal.get('take_profit')

                if entry:
                    hlines_dict['hlines'].append(float(entry))
                    hlines_dict['colors'].append('green')
                    hlines_dict['linestyle'].append('--')
                    hlines_dict['linewidths'].append(1.5)

                if stop:
                    hlines_dict['hlines'].append(float(stop))
                    hlines_dict['colors'].append('red')
                    hlines_dict['linestyle'].append('--')
                    hlines_dict['linewidths'].append(1.5)

                if target:
                    hlines_dict['hlines'].append(float(target))
                    hlines_dict['colors'].append('blue')
                    hlines_dict['linestyle'].append('--')
                    hlines_dict['linewidths'].append(1.5)

            # Chart style
            mc = mpf.make_marketcolors(
                up='#26a69a', down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume='inherit'
            )
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#e0e0e0',
                figcolor='white',
                facecolor='white'
            )

            # Create chart
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                addplot=addplots if addplots else None,
                hlines=hlines_dict if hlines_dict['hlines'] else None,
                figsize=(12, 8),
                title=f'{ticker} - 60 Day Analysis',
                returnfig=True
            )

            # Save to base64
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)

            return chart_base64

        except Exception as e:
            logger.error(f"Chart generation failed for {ticker}: {e}")
            return None

    def _build_deep_dive_prompt(
        _self,
        ticker: str,
        signal: Dict[str, Any],
        metrics: Dict[str, Any],
        fundamentals: Dict[str, Any]
    ) -> str:
        """Build comprehensive analysis prompt for Deep Dive."""

        # Signal info
        signal_type = signal.get('signal_type', 'BUY')
        scanner = signal.get('scanner_name', 'unknown')
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        target = signal.get('take_profit_1') or signal.get('take_profit', 0)
        score = signal.get('composite_score', 0)
        tier = signal.get('quality_tier', 'B')

        # Calculate R:R
        if entry and stop and target:
            risk = abs(float(entry) - float(stop))
            reward = abs(float(target) - float(entry))
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            rr_ratio = 0

        # Technical data
        rsi = metrics.get('rsi_14', 50)
        macd = metrics.get('macd_histogram', 0)
        trend = metrics.get('trend_classification', 'neutral')
        ma_align = metrics.get('ma_alignment', 'mixed')
        smc_trend = metrics.get('smc_trend', 'neutral')
        smc_bias = metrics.get('smc_bias', 'neutral')

        # Fundamental data
        company = fundamentals.get('name', ticker)
        sector = fundamentals.get('sector', 'Unknown')
        pe = fundamentals.get('trailing_pe', 'N/A')
        growth = fundamentals.get('earnings_growth', 'N/A')
        beta = fundamentals.get('beta', 'N/A')

        # Determine if this is a signal analysis or general stock analysis
        is_signal = signal_type not in ['ANALYSIS', 'N/A', None, '']

        if is_signal:
            signal_section = f"""## SIGNAL OVERVIEW
**{ticker}** ({company}) | {signal_type} Signal
Sector: {sector} | Scanner: {scanner}
Quality: {tier} tier ({score}/100)

## RISK PARAMETERS
Entry: ${entry} | Stop: ${stop} | Target: ${target}
Reward/Risk: {rr_ratio:.1f}:1"""
        else:
            signal_section = f"""## STOCK OVERVIEW
**{ticker}** ({company})
Sector: {sector}
Current Price: ${entry}
Quality Score: {score}/100"""

        prompt = f"""You are a Senior Equity Analyst. Analyze this stock with institutional rigor.

## CHART ANALYSIS (CRITICAL)
The attached chart shows 60-day price history with:
- Candlesticks (green=up, red=down)
- Moving averages: SMA20 (blue), SMA50 (orange), SMA200 (purple)
- Volume bars at bottom

IMPORTANT: Carefully examine the chart for:
1. Trend quality and direction (clean vs choppy)
2. Price position relative to moving averages
3. Support and resistance levels
4. Volume confirmation of price moves
5. Any visual patterns (flags, wedges, double bottoms, etc.)

{signal_section}

## TECHNICAL SNAPSHOT
RSI(14): {rsi} | MACD: {macd}
Trend: {trend} | MA Alignment: {ma_align}
SMC Trend: {smc_trend} | SMC Bias: {smc_bias}

## FUNDAMENTALS
P/E: {pe} | Earnings Growth: {growth}% | Beta: {beta}

Respond in this exact JSON format:
{{"grade":"A+/A/B/C/D","score":1-10,"action":"STRONG BUY/BUY/HOLD/AVOID","conviction":"HIGH/MEDIUM/LOW","thesis":"2-3 sentence investment thesis","key_strengths":["strength1","strength2","strength3"],"key_risks":["risk1","risk2","risk3"],"position_recommendation":"Full/Half/Quarter/Skip","time_horizon":"Intraday/Swing/Position","stop_adjustment":"N/A"}}"""

        return prompt

    def _parse_claude_analysis_response(_self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's JSON response into analysis dict."""
        import json
        import re

        try:
            # Try to find JSON block - handle nested arrays/objects
            # Look for JSON starting with { and ending with }
            # Use a more robust approach that handles nested structures

            # First try: find JSON between first { and last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx + 1]
                analysis = json.loads(json_str)

                # Validate required fields exist
                if 'grade' in analysis:
                    return analysis

            # Second try: look for JSON in code blocks
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if code_block_match:
                analysis = json.loads(code_block_match.group(1))
                if 'grade' in analysis:
                    return analysis

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Response: {response_text[:500]}")
        except Exception as e:
            logger.warning(f"Parse error: {e}")

        # Fallback: return empty analysis with error
        return {
            'grade': 'C',
            'score': 5,
            'action': 'HOLD',
            'conviction': 'LOW',
            'thesis': 'Unable to parse analysis response',
            'key_strengths': [],
            'key_risks': ['Analysis parsing failed'],
            'position_recommendation': 'Skip',
            'time_horizon': 'N/A',
            'error': 'Failed to parse response'
        }

    def _save_claude_analysis(_self, signal_id: int, analysis: Dict[str, Any]):
        """Save Claude analysis to database."""
        conn = _self._get_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE stock_scanner_signals
                    SET
                        claude_grade = %s,
                        claude_score = %s,
                        claude_action = %s,
                        claude_conviction = %s,
                        claude_thesis = %s,
                        claude_key_strengths = %s,
                        claude_key_risks = %s,
                        claude_position_rec = %s,
                        claude_time_horizon = %s,
                        claude_stop_adjustment = %s,
                        claude_analyzed_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """, (
                    analysis.get('grade'),
                    analysis.get('score'),
                    analysis.get('action'),
                    analysis.get('conviction'),
                    analysis.get('thesis'),
                    analysis.get('key_strengths'),
                    analysis.get('key_risks'),
                    analysis.get('position_recommendation'),
                    analysis.get('time_horizon'),
                    analysis.get('stop_adjustment'),
                    signal_id
                ))
                conn.commit()
                logger.info(f"Saved Claude analysis for signal {signal_id}: {analysis.get('grade')}")
        except Exception as e:
            logger.error(f"Failed to save Claude analysis for signal {signal_id}: {e}")
            conn.rollback()
        finally:
            conn.close()


# Singleton instance
@st.cache_resource
def get_stock_service() -> StockAnalyticsService:
    """Get or create the stock analytics service singleton."""
    return StockAnalyticsService()
