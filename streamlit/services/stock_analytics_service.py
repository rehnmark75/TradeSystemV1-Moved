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

    @st.cache_data(ttl=300)
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

        return {
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
            'risk_reward_ratio': round(2.0 / suggested_stop_pct * atr_pct, 2) if suggested_stop_pct > 0 else 0
        }

    # =========================================================================
    # SCANNER SIGNALS
    # =========================================================================

    @st.cache_data(ttl=60)
    def get_scanner_signals(
        _self,
        scanner_name: str = None,
        status: str = 'active',
        min_score: int = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get scanner signals from database.

        Args:
            scanner_name: Filter by specific scanner
            status: Filter by status ('active', 'triggered', 'closed', etc.)
            min_score: Minimum composite score
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
            ORDER BY s.composite_score DESC, s.signal_timestamp DESC
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
                # Total signals
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_signals,
                        COUNT(*) FILTER (WHERE status = 'active') as active_signals,
                        COUNT(*) FILTER (WHERE quality_tier IN ('A+', 'A')) as high_quality,
                        COUNT(*) FILTER (WHERE DATE(signal_timestamp) = CURRENT_DATE) as today_signals
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


# Singleton instance
@st.cache_resource
def get_stock_service() -> StockAnalyticsService:
    """Get or create the stock analytics service singleton."""
    return StockAnalyticsService()
