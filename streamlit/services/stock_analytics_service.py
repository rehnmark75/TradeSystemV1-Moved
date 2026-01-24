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
        """Get a database connection from pool."""
        try:
            from services.db_utils import get_psycopg2_connection
            return get_psycopg2_connection("stocks")
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {e}")
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
                w.avg_daily_change_5d,
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
                i.short_percent_float,
                -- TradingView summary counts
                COALESCE(m.tv_osc_buy, 0) as tv_osc_buy,
                COALESCE(m.tv_osc_sell, 0) as tv_osc_sell,
                COALESCE(m.tv_osc_neutral, 0) as tv_osc_neutral,
                COALESCE(m.tv_ma_buy, 0) as tv_ma_buy,
                COALESCE(m.tv_ma_sell, 0) as tv_ma_sell,
                COALESCE(m.tv_ma_neutral, 0) as tv_ma_neutral,
                COALESCE(m.tv_overall_signal, 'NEUTRAL') as tv_overall_signal,
                COALESCE(m.tv_overall_score, 0) as tv_overall_score,
                -- Individual TradingView indicators (for detail view)
                m.rsi_14, m.macd, m.macd_signal,
                m.stoch_k, m.stoch_d, m.cci_20,
                m.adx_14, m.plus_di, m.minus_di,
                m.ao_value, m.momentum_10,
                m.stoch_rsi_k, m.stoch_rsi_d,
                m.williams_r, m.bull_power, m.bear_power,
                m.ultimate_osc,
                m.ema_10, m.ema_20, m.ema_30, m.ema_50, m.ema_100, m.ema_200,
                m.sma_10, m.sma_20, m.sma_30, m.sma_50, m.sma_100, m.sma_200,
                m.ichimoku_base, m.vwma_20
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
        """Get signals from the last N hours from stock_scanner_signals table.

        Includes Claude AI analysis fields (grade, action, thesis) and news sentiment when available.
        """
        query = """
            SELECT
                s.signal_timestamp,
                s.ticker,
                i.name,
                s.signal_type,
                s.entry_price,
                s.stop_loss,
                s.take_profit_1 as take_profit,
                s.composite_score as confidence,
                s.scanner_name,
                s.quality_tier,
                s.claude_grade,
                s.claude_action,
                s.claude_score,
                s.claude_thesis,
                s.news_sentiment_score,
                s.news_sentiment_level,
                s.news_headlines_count,
                CASE
                    WHEN s.signal_type = 'BUY' AND s.stop_loss > 0 AND s.entry_price > s.stop_loss THEN
                        ROUND(((s.take_profit_1 - s.entry_price) / NULLIF(s.entry_price - s.stop_loss, 0))::numeric, 1)
                    WHEN s.signal_type = 'SELL' AND s.stop_loss > 0 AND s.stop_loss > s.entry_price THEN
                        ROUND(((s.entry_price - s.take_profit_1) / NULLIF(s.stop_loss - s.entry_price, 0))::numeric, 1)
                    ELSE NULL
                END as risk_reward,
                w.tier,
                w.score,
                w.avg_daily_change_5d
            FROM stock_scanner_signals s
            LEFT JOIN stock_instruments i ON s.ticker = i.ticker
            LEFT JOIN stock_watchlist w ON s.ticker = w.ticker
                AND w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            WHERE s.signal_timestamp > NOW() - INTERVAL '%s hours'
            AND s.status = 'active'
            ORDER BY
                CASE WHEN s.claude_grade IS NOT NULL THEN 0 ELSE 1 END,
                s.signal_timestamp DESC
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
                w.avg_daily_change_5d,
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
                    (SELECT signal_type FROM stock_scanner_signals
                     WHERE ticker = w.ticker AND status = 'active'
                     ORDER BY signal_timestamp DESC LIMIT 1) as latest_signal
                FROM stock_scanner_signals
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
                        w.avg_daily_change_5d,
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
                        -- DAQ columns
                        w.daq_score,
                        w.daq_grade,
                        w.daq_mtf_score,
                        w.daq_volume_score,
                        w.daq_smc_score,
                        w.daq_quality_score,
                        w.daq_catalyst_score,
                        w.daq_news_score,
                        w.daq_regime_score,
                        w.daq_sector_score,
                        w.daq_earnings_risk,
                        w.daq_analyzed_at,
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

            # Generate chart for vision analysis (same as deep dive)
            chart_base64 = None
            try:
                candles = _self.get_daily_candles(ticker, days=60)
                if candles is not None and not candles.empty:
                    # Create a minimal signal dict for chart generation
                    signal_for_chart = {
                        'entry_price': pick.get('current_price'),
                        'stop_loss': None,
                        'take_profit_1': None
                    }
                    chart_base64 = _self._generate_analysis_chart(
                        ticker, candles, signal_for_chart, technical or {}
                    )
                    if chart_base64:
                        logger.debug(f"Generated chart for top pick {ticker}")
            except Exception as e:
                logger.warning(f"Chart generation failed for {ticker}: {e}")

            # Build the standard institutional-grade prompt (same as scanner)
            prompt = _self._build_standard_analysis_prompt(pick, technical)

            # Call Claude API with Sonnet model (same as scanner standard analysis)
            # Use vision API if chart is available for richer analysis
            client = anthropic.Anthropic(api_key=api_key)

            if chart_base64:
                # Vision request with chart image
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
                # Text-only request (fallback)
                message_content = [{"type": "text", "text": prompt}]

            response = client.messages.create(
                model="claude-sonnet-4-20250514",  # Use Sonnet for standard analysis (same as scanner)
                max_tokens=400,  # Increased for richer vision analysis
                messages=[{"role": "user", "content": message_content}]
            )

            # Parse response using robust parser
            response_text = response.content[0].text.strip()
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            analysis = _self._parse_claude_response(response_text)

            if analysis.get('success'):
                # Add token tracking
                analysis['claude_tokens_used'] = tokens_used
                analysis['claude_model'] = 'claude-sonnet-4-20250514'
                analysis['has_chart'] = chart_base64 is not None

                # Store in database for future use
                _self._store_claude_analysis(ticker, calculation_date, analysis)
                logger.info(f"Stored Claude analysis for {ticker} (Sonnet, {tokens_used} tokens, chart={'yes' if chart_base64 else 'no'})")

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
        min_rs_percentile: int = None,
        max_rs_percentile: int = None,
        rs_trend: str = None,
        limit: int = 100,
        order_by: str = 'score'
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
            min_rs_percentile: Minimum RS percentile (0-100)
            max_rs_percentile: Maximum RS percentile (0-100)
            rs_trend: Filter by RS trend ('improving', 'stable', 'deteriorating')
            limit: Maximum results
            order_by: 'score' (default) or 'timestamp' for most recent first

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

        # RS filters (applied after join with metrics table)
        rs_conditions = []
        if min_rs_percentile is not None:
            rs_conditions.append(f"m.rs_percentile >= {min_rs_percentile}")
        if max_rs_percentile is not None:
            rs_conditions.append(f"m.rs_percentile <= {max_rs_percentile}")
        if rs_trend:
            rs_conditions.append(f"m.rs_trend = '{rs_trend}'")

        # Use a subquery to get only the latest signal per ticker/scanner combination
        # This prevents showing stale signals when a new scan has been run
        # Also inherit news sentiment from most recent analyzed signal for the same ticker
        # Note: We explicitly list columns (excluding news_*) to avoid duplicate column names
        # when using COALESCE for inherited news data
        # days_active shows how many unique days the signal has been firing (persistence indicator)
        # Note: signal_persistence uses status filter only (not date filters) to show total persistence
        query = f"""
            WITH latest_signals AS (
                SELECT DISTINCT ON (ticker, scanner_name)
                    *
                FROM stock_scanner_signals
                WHERE {where_clause}
                ORDER BY ticker, scanner_name, signal_timestamp DESC
            ),
            signal_persistence AS (
                SELECT
                    ticker,
                    scanner_name,
                    COUNT(DISTINCT DATE(signal_timestamp)) as days_active,
                    MIN(DATE(signal_timestamp)) as first_signal_date
                FROM stock_scanner_signals
                WHERE status = 'active'
                GROUP BY ticker, scanner_name
            ),
            latest_news AS (
                SELECT DISTINCT ON (ticker)
                    ticker,
                    news_sentiment_score as inherited_news_score,
                    news_sentiment_level as inherited_news_level,
                    news_headlines_count as inherited_news_count,
                    news_factors as inherited_news_factors,
                    news_analyzed_at as inherited_news_analyzed_at
                FROM stock_scanner_signals
                WHERE news_sentiment_score IS NOT NULL
                ORDER BY ticker, news_analyzed_at DESC
            ),
            open_trades AS (
                SELECT DISTINCT
                    SPLIT_PART(ticker, '.', 1) as ticker,
                    side,
                    profit,
                    open_price,
                    current_price
                FROM broker_trades
                WHERE status = 'open'
            )
            SELECT
                s.id, s.signal_timestamp, s.scanner_name, s.ticker, s.signal_type,
                s.entry_price, s.stop_loss, s.take_profit_1, s.take_profit_2,
                s.risk_reward_ratio, s.risk_percent, s.composite_score, s.quality_tier,
                s.trend_score, s.momentum_score, s.volume_score, s.pattern_score,
                s.confluence_score, s.setup_description, s.confluence_factors,
                s.timeframe, s.market_regime, s.suggested_position_size_pct,
                s.max_risk_per_trade_pct, s.status, s.trigger_timestamp,
                s.close_timestamp, s.close_price, s.realized_pnl_pct,
                s.realized_r_multiple, s.exit_reason, s.created_at, s.updated_at,
                s.claude_grade, s.claude_score, s.claude_conviction, s.claude_action,
                s.claude_thesis, s.claude_key_strengths, s.claude_key_risks,
                s.claude_position_rec, s.claude_stop_adjustment, s.claude_time_horizon,
                s.claude_raw_response, s.claude_analyzed_at, s.claude_tokens_used,
                s.claude_latency_ms, s.claude_model,
                i.name as company_name,
                COALESCE(s.news_sentiment_score, n.inherited_news_score) as news_sentiment_score,
                COALESCE(s.news_sentiment_level, n.inherited_news_level) as news_sentiment_level,
                COALESCE(s.news_headlines_count, n.inherited_news_count) as news_headlines_count,
                COALESCE(s.news_factors, n.inherited_news_factors) as news_factors,
                COALESCE(s.news_analyzed_at, n.inherited_news_analyzed_at) as news_analyzed_at,
                m.avg_daily_change_5d,
                COALESCE(p.days_active, 1) as days_active,
                p.first_signal_date,
                CASE WHEN t.ticker IS NOT NULL THEN true ELSE false END as in_trade,
                t.side as trade_side,
                t.profit as trade_profit,
                t.open_price as trade_open_price,
                t.current_price as trade_current_price,
                m.rs_vs_spy,
                m.rs_percentile,
                m.rs_trend,
                sa.rs_vs_spy as sector_rs,
                sa.sector_stage,
                i.sector,
                -- Deep Analysis Quality (DAQ) fields
                d.daq_score,
                d.daq_grade,
                d.mtf_score,
                d.volume_score as daq_volume_score,
                d.smc_score as daq_smc_score,
                d.quality_score as daq_quality_score,
                d.catalyst_score as daq_catalyst_score,
                d.news_score as daq_news_score,
                d.regime_score as daq_regime_score,
                d.sector_score as daq_sector_score,
                d.earnings_within_7d,
                d.high_short_interest,
                d.sector_underperforming,
                d.analysis_timestamp as daq_analyzed_at,
                -- Trading metrics for Trade Plan (from screening metrics)
                m.atr_14,
                m.atr_percent,
                m.swing_high,
                m.swing_low,
                m.swing_high_date,
                m.swing_low_date,
                m.relative_volume,
                -- Earnings from instruments
                i.earnings_date,
                CASE
                    WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
                    THEN (i.earnings_date - CURRENT_DATE)
                    ELSE NULL
                END as days_to_earnings
            FROM latest_signals s
            LEFT JOIN stock_instruments i ON s.ticker = i.ticker
            LEFT JOIN latest_news n ON s.ticker = n.ticker
            LEFT JOIN signal_persistence p ON s.ticker = p.ticker AND s.scanner_name = p.scanner_name
            LEFT JOIN stock_screening_metrics m ON s.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            LEFT JOIN open_trades t ON s.ticker = t.ticker
            LEFT JOIN sector_analysis sa ON i.sector = sa.sector
                AND sa.calculation_date = (SELECT MAX(calculation_date) FROM sector_analysis)
            LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
        """

        # Add RS filter conditions if any
        if rs_conditions:
            query += "\n            WHERE " + " AND ".join(rs_conditions)

        # Add ORDER BY based on parameter
        if order_by == 'timestamp':
            query += """
            ORDER BY s.signal_timestamp DESC
            """
        else:
            # Default: order by score (Claude analyzed first, then by score)
            query += """
            ORDER BY
                CASE WHEN s.claude_analyzed_at IS NOT NULL THEN 0 ELSE 1 END,
                COALESCE(s.claude_score, 0) DESC,
                s.composite_score DESC,
                s.signal_timestamp DESC
            """

        query += f"LIMIT {limit}"

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

                # By scanner - include ALL registered scanners, even those without signals
                cursor.execute("""
                    SELECT
                        r.scanner_name,
                        COALESCE(s.signal_count, 0) as signal_count,
                        COALESCE(s.avg_score, 0) as avg_score,
                        COALESCE(s.active_count, 0) as active_count
                    FROM stock_signal_scanners r
                    LEFT JOIN (
                        SELECT
                            scanner_name,
                            COUNT(*) as signal_count,
                            ROUND(AVG(composite_score)::numeric, 1) as avg_score,
                            COUNT(*) FILTER (WHERE status = 'active') as active_count
                        FROM stock_scanner_signals
                        GROUP BY scanner_name
                    ) s ON r.scanner_name = s.scanner_name
                    WHERE r.is_active = true
                    ORDER BY COALESCE(s.signal_count, 0) DESC, r.scanner_name
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

    def enrich_signal_with_news(_self, signal_id: int) -> Dict[str, Any]:
        """
        Trigger news sentiment enrichment for a specific signal.

        Fetches news from Finnhub and analyzes sentiment using VADER.

        Args:
            signal_id: The signal ID to enrich with news

        Returns:
            Dict with success status, sentiment data, and message
        """
        import os
        import requests
        from datetime import datetime, timedelta

        try:
            # Get the signal's ticker first
            conn = _self._get_connection()
            if not conn:
                return {'success': False, 'error': 'Database connection failed'}

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT id, ticker, news_sentiment_score, news_analyzed_at
                    FROM stock_scanner_signals
                    WHERE id = %s
                """, (signal_id,))
                signal = cursor.fetchone()

            if not signal:
                conn.close()
                return {'success': False, 'error': f'Signal {signal_id} not found'}

            ticker = signal['ticker']

            # Get Finnhub API key
            finnhub_api_key = os.getenv('FINNHUB_API_KEY', '')
            if not finnhub_api_key:
                conn.close()
                return {'success': False, 'error': 'FINNHUB_API_KEY not configured'}

            # Fetch news from Finnhub
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)

            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker.upper(),
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'token': finnhub_api_key
            }

            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 200:
                conn.close()
                return {'success': False, 'error': f'Finnhub API error: {response.status_code}'}

            articles = response.json()

            if not articles:
                # Update signal with no news found
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE stock_scanner_signals
                        SET news_sentiment_score = 0,
                            news_sentiment_level = 'neutral',
                            news_headlines_count = 0,
                            news_factors = ARRAY['No news articles found'],
                            news_analyzed_at = NOW()
                        WHERE id = %s
                    """, (signal_id,))
                conn.commit()
                conn.close()
                return {
                    'success': True,
                    'message': f'No news found for {ticker}',
                    'ticker': ticker,
                    'articles_count': 0
                }

            # Analyze sentiment using VADER
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()

                scores = []
                for article in articles[:50]:  # Limit to 50 articles
                    headline = article.get('headline', '')
                    summary = article.get('summary', '')
                    text = f"{headline} {summary}"

                    if text.strip():
                        sentiment = analyzer.polarity_scores(text)
                        scores.append(sentiment['compound'])

                if scores:
                    avg_score = sum(scores) / len(scores)
                else:
                    avg_score = 0.0

                # Classify sentiment level
                if avg_score >= 0.5:
                    level = 'very_bullish'
                elif avg_score >= 0.15:
                    level = 'bullish'
                elif avg_score <= -0.5:
                    level = 'very_bearish'
                elif avg_score <= -0.15:
                    level = 'bearish'
                else:
                    level = 'neutral'

                # Build factors
                factors = []
                level_labels = {
                    'very_bullish': 'Strong positive',
                    'bullish': 'Positive',
                    'neutral': 'Neutral',
                    'bearish': 'Negative',
                    'very_bearish': 'Strong negative'
                }
                factors.append(f"{level_labels[level]} news sentiment ({avg_score:.2f})")
                factors.append(f"{len(scores)} news articles analyzed")

                # Add top headline
                if articles:
                    top_headline = articles[0].get('headline', '')[:80]
                    if top_headline:
                        factors.append(f'Key: "{top_headline}..."')

            except ImportError:
                # VADER not available, use simple keyword analysis
                avg_score = 0.0
                level = 'neutral'
                factors = ['Sentiment analysis unavailable (VADER not installed)']

            # Update signal in database
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE stock_scanner_signals
                    SET news_sentiment_score = %s,
                        news_sentiment_level = %s,
                        news_headlines_count = %s,
                        news_factors = %s,
                        news_analyzed_at = NOW()
                    WHERE id = %s
                """, (avg_score, level, len(scores), factors, signal_id))

                # Also cache news articles
                for article in articles[:20]:  # Cache top 20
                    cursor.execute("""
                        INSERT INTO stock_news_cache
                            (ticker, headline, summary, source, url, published_at, sentiment_score, finnhub_id)
                        VALUES (%s, %s, %s, %s, %s, to_timestamp(%s), %s, %s)
                        ON CONFLICT (ticker, finnhub_id) DO UPDATE
                        SET headline = EXCLUDED.headline,
                            summary = EXCLUDED.summary,
                            fetched_at = NOW()
                    """, (
                        ticker,
                        article.get('headline', '')[:500],
                        article.get('summary', '')[:1000] if article.get('summary') else None,
                        article.get('source', '')[:100],
                        article.get('url', ''),
                        article.get('datetime', 0),
                        None,  # Individual sentiment not calculated
                        article.get('id')
                    ))

            conn.commit()
            conn.close()

            return {
                'success': True,
                'message': f'News enrichment completed for {ticker}',
                'ticker': ticker,
                'sentiment_score': avg_score,
                'sentiment_level': level,
                'articles_count': len(scores)
            }

        except requests.Timeout:
            return {'success': False, 'error': 'Finnhub API request timed out'}
        except Exception as e:
            logger.error(f"Failed to enrich signal {signal_id} with news: {e}")
            return {'success': False, 'error': str(e)}

    def get_signal_news_articles(_self, signal_id: int) -> List[Dict[str, Any]]:
        """
        Get cached news articles for a signal's ticker.

        Args:
            signal_id: The signal ID

        Returns:
            List of news articles from cache
        """
        try:
            # First get the ticker
            signal_result = _self._execute_query("""
                SELECT ticker FROM stock_scanner_signals WHERE id = %s
            """, (signal_id,))

            if not signal_result:
                return []

            ticker = signal_result[0]['ticker']

            # Get cached news articles
            articles = _self._execute_query("""
                SELECT
                    headline,
                    summary,
                    source,
                    url,
                    published_at,
                    sentiment_score
                FROM stock_news_cache
                WHERE ticker = %s
                  AND fetched_at > NOW() - INTERVAL '24 hours'
                ORDER BY published_at DESC
                LIMIT 5
            """, (ticker,))

            return articles

        except Exception as e:
            logger.error(f"Failed to get news articles for signal {signal_id}: {e}")
            return []

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


    # =========================================================================
    # NEW DASHBOARD METHODS (Stock Scanner Redesign)
    # =========================================================================

    @st.cache_data(ttl=60)
    def get_dashboard_stats(_self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard statistics for the redesigned stock scanner page.

        Returns metrics for: active signals, high quality signals, Claude analyzed,
        today's signals, scanner leaderboard, and tier distribution.
        """
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Core signal stats
                cursor.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'active') as active_signals,
                        COUNT(*) FILTER (WHERE status = 'active' AND quality_tier IN ('A+', 'A')) as high_quality,
                        COUNT(*) FILTER (WHERE status = 'active' AND claude_analyzed_at IS NOT NULL) as claude_analyzed,
                        COUNT(*) FILTER (WHERE DATE(signal_timestamp) = CURRENT_DATE AND status = 'active') as today_signals,
                        COUNT(*) FILTER (WHERE status = 'active' AND signal_type = 'BUY') as buy_signals,
                        COUNT(*) FILTER (WHERE status = 'active' AND signal_type = 'SELL') as sell_signals
                    FROM stock_scanner_signals
                """)
                stats = dict(cursor.fetchone())

                # Quality tier distribution for active signals
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
                tier_dist = {r['quality_tier']: r['count'] for r in cursor.fetchall()}
                stats['tier_distribution'] = tier_dist

                # Signal distribution by scanner
                cursor.execute("""
                    SELECT
                        scanner_name,
                        COUNT(*) as count
                    FROM stock_scanner_signals
                    WHERE status = 'active'
                    GROUP BY scanner_name
                    ORDER BY count DESC
                """)
                scanner_dist = {r['scanner_name']: r['count'] for r in cursor.fetchall()}
                stats['scanner_distribution'] = scanner_dist

                # Last scan time
                cursor.execute("""
                    SELECT MAX(signal_timestamp) as last_scan
                    FROM stock_scanner_signals
                """)
                result = cursor.fetchone()
                stats['last_scan'] = result['last_scan'] if result else None

                # Total stocks scanned (from instruments)
                cursor.execute("""
                    SELECT COUNT(*) as count FROM stock_instruments WHERE is_active = true
                """)
                stats['total_stocks'] = cursor.fetchone()['count']

                return stats

        except Exception as e:
            logger.error(f"Failed to get dashboard stats: {e}")
            return {}
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_scanner_leaderboard(_self) -> List[Dict[str, Any]]:
        """
        Get scanners ranked by profit factor from backtest executions.

        Returns list of scanners with their backtest performance metrics.
        Maps scanner names to backtest strategy names (case-insensitive, handles variations).
        """
        query = """
            WITH scanner_mapping AS (
                -- Map scanner names to backtest strategy names
                -- 8 active scanners after consolidation
                SELECT scanner_name,
                    CASE
                        WHEN scanner_name = 'breakout_confirmation' THEN 'BREAKOUT'
                        WHEN scanner_name = 'trend_reversal' THEN 'TREND_REVERSAL'
                        WHEN scanner_name = 'rsi_divergence' THEN 'RSI_DIVERGENCE'
                        WHEN scanner_name = 'gap_and_go' THEN 'GAP_AND_GO'
                        WHEN scanner_name = 'macd_momentum' THEN 'MACD_MOMENTUM'
                        WHEN scanner_name = 'ema_pullback' THEN 'EMA_PULLBACK'
                        WHEN scanner_name = 'zlma_trend' THEN 'ZLMA_CROSSOVER'
                        WHEN scanner_name = 'reversal_scanner' THEN 'REVERSAL'
                        ELSE UPPER(REPLACE(scanner_name, '_', '_'))
                    END as strategy_name
                FROM stock_signal_scanners
                WHERE is_active = true
            )
            SELECT
                s.scanner_name,
                COALESCE(b.profit_factor, 0) as profit_factor,
                COALESCE(b.win_rate, 0) as win_rate,
                COALESCE(b.total_trades, 0) as total_trades,
                COALESCE(b.winners, 0) as winners,
                COALESCE(b.losers, 0) as losers,
                COALESCE(active.signal_count, 0) as active_signals
            FROM stock_signal_scanners s
            JOIN scanner_mapping m ON s.scanner_name = m.scanner_name
            LEFT JOIN LATERAL (
                SELECT
                    profit_factor,
                    win_rate,
                    total_trades,
                    winners,
                    losers
                FROM stock_backtest_executions
                WHERE strategy_name = m.strategy_name
                  AND status = 'completed'
                  AND total_trades > 0  -- Skip empty backtest runs
                ORDER BY completed_at DESC
                LIMIT 1
            ) b ON true
            LEFT JOIN (
                -- Map old signal scanner names to new scanner names
                SELECT
                    CASE
                        WHEN scanner_name = 'trend_reversal' THEN 'reversal_scanner'
                        WHEN scanner_name = 'trend_momentum' THEN 'ema_pullback'
                        ELSE scanner_name
                    END as mapped_scanner_name,
                    COUNT(*) as signal_count
                FROM stock_scanner_signals
                WHERE status = 'active'
                GROUP BY 1
            ) active ON s.scanner_name = active.mapped_scanner_name
            WHERE s.is_active = true
            ORDER BY COALESCE(b.profit_factor, 0) DESC, s.scanner_name
        """
        return _self._execute_query(query, ())

    @st.cache_data(ttl=60)
    def get_top_signals(_self, limit: int = 5, min_tier: str = 'A') -> List[Dict[str, Any]]:
        """
        Get top quality active signals for dashboard display.

        Args:
            limit: Maximum number of signals to return
            min_tier: Minimum quality tier ('A+', 'A', 'B', 'C', 'D')

        Returns:
            List of top signals with company info
        """
        tier_filter = "('A+', 'A')" if min_tier == 'A' else "('A+')" if min_tier == 'A+' else "('A+', 'A', 'B')"

        query = f"""
            WITH signal_persistence AS (
                SELECT
                    ticker,
                    scanner_name,
                    COUNT(DISTINCT DATE(signal_timestamp)) as days_active
                FROM stock_scanner_signals
                WHERE status = 'active'
                AND quality_tier IN {tier_filter}
                GROUP BY ticker, scanner_name
            )
            SELECT * FROM (
                SELECT DISTINCT ON (s.ticker, s.scanner_name)
                    s.id,
                    s.ticker,
                    i.name as company_name,
                    s.signal_type,
                    s.composite_score,
                    s.quality_tier,
                    s.scanner_name,
                    s.entry_price,
                    s.stop_loss,
                    s.take_profit_1,
                    s.risk_reward_ratio,
                    s.claude_grade,
                    s.claude_action,
                    s.signal_timestamp,
                    COALESCE(p.days_active, 1) as days_active
                FROM stock_scanner_signals s
                LEFT JOIN stock_instruments i ON s.ticker = i.ticker
                LEFT JOIN signal_persistence p ON s.ticker = p.ticker AND s.scanner_name = p.scanner_name
                WHERE s.status = 'active'
                AND s.quality_tier IN {tier_filter}
                ORDER BY s.ticker, s.scanner_name, s.signal_timestamp DESC
            ) deduped
            ORDER BY composite_score DESC, signal_timestamp DESC
            LIMIT %s
        """
        return _self._execute_query(query, (limit,))

    @st.cache_data(ttl=300)
    def get_scanner_backtest_metrics(_self, scanner_name: str) -> Dict[str, Any]:
        """
        Get detailed backtest metrics for a specific scanner.

        Args:
            scanner_name: Name of the scanner

        Returns:
            Dict with profit_factor, win_rate, avg_r_multiple, etc.
        """
        query = """
            SELECT
                strategy_name as scanner_name,
                profit_factor,
                win_rate,
                total_trades,
                winning_trades,
                losing_trades,
                avg_r_multiple,
                max_r_multiple,
                min_r_multiple,
                max_drawdown_pct,
                total_return_pct,
                sharpe_ratio,
                start_date,
                end_date,
                completed_at
            FROM stock_backtest_executions
            WHERE strategy_name = %s
            ORDER BY completed_at DESC
            LIMIT 1
        """
        results = _self._execute_query(query, (scanner_name,))
        return results[0] if results else {}

    # =========================================================================
    # WATCHLIST METHODS (5 Predefined Technical Screens)
    # =========================================================================

    @st.cache_data(ttl=60)
    def get_watchlist_results(_self, watchlist_name: str, scan_date: str = None) -> pd.DataFrame:
        """
        Get results for a specific predefined watchlist.

        For crossover watchlists (ema_50_crossover, ema_20_crossover, macd_bullish_cross):
        - Shows only active entries
        - Days is calculated from crossover_date to today

        For event watchlists (gap_up_continuation, rsi_oversold_bounce):
        - Shows entries for the selected date
        - Days is always 1 (single-day events)

        Args:
            watchlist_name: One of: ema_50_crossover, ema_20_crossover,
                           macd_bullish_cross, gap_up_continuation, rsi_oversold_bounce
            scan_date: Optional date string (YYYY-MM-DD). For event watchlists only.

        Returns:
            DataFrame with stocks matching the watchlist criteria
        """
        # Crossover watchlists show all active entries (ignores scan_date)
        crossover_watchlists = {'ema_50_crossover', 'ema_20_crossover', 'macd_bullish_cross'}

        if watchlist_name in crossover_watchlists:
            # For crossover watchlists: show all active entries, calculate days from crossover_date
            query = """
                WITH open_trades AS (
                    SELECT DISTINCT
                        SPLIT_PART(ticker, '.', 1) as ticker,
                        side,
                        profit,
                        open_price,
                        current_price
                    FROM broker_trades
                    WHERE status = 'open'
                )
                SELECT
                    w.ticker,
                    i.name,
                    w.price,
                    w.volume,
                    w.avg_volume,
                    w.ema_20,
                    w.ema_50,
                    w.ema_200,
                    w.rsi_14,
                    w.macd,
                    w.macd_histogram,
                    w.gap_pct,
                    w.price_change_1d,
                    w.scan_date,
                    w.crossover_date,
                    (CURRENT_DATE - w.crossover_date) + 1 as days_on_list,
                    CASE WHEN s.ticker IS NOT NULL THEN true ELSE false END as has_signal,
                    s.quality_tier as signal_tier,
                    s.signal_type,
                    COALESCE(w.avg_daily_change_5d, m.avg_daily_change_5d) as avg_daily_change_5d,
                    m.rs_vs_spy,
                    CASE WHEN t.ticker IS NOT NULL THEN true ELSE false END as in_trade,
                    t.side as trade_side,
                    t.profit as trade_profit,
                    w.daq_score,
                    w.daq_grade,
                    -- DAQ component scores
                    w.daq_mtf_score,
                    w.daq_volume_score,
                    w.daq_smc_score,
                    w.daq_quality_score,
                    w.daq_catalyst_score,
                    w.daq_news_score,
                    w.daq_regime_score,
                    w.daq_sector_score,
                    -- DAQ risk flags
                    w.daq_earnings_risk,
                    w.daq_high_short_interest,
                    w.daq_sector_underperforming,
                    -- Trading metrics (Trade Plan)
                    w.atr_14,
                    w.atr_percent,
                    w.swing_high,
                    w.swing_low,
                    w.swing_high_date,
                    w.swing_low_date,
                    w.nearest_ob_price,
                    w.nearest_ob_type,
                    w.nearest_ob_distance,
                    w.suggested_entry_low,
                    w.suggested_entry_high,
                    w.suggested_stop_loss,
                    w.suggested_target_1,
                    w.suggested_target_2,
                    w.risk_reward_ratio,
                    w.risk_percent,
                    w.volume_trend,
                    w.relative_volume,
                    -- RS trend from watchlist (if populated) or fallback to metrics
                    COALESCE(w.rs_percentile, m.rs_percentile) as rs_percentile,
                    COALESCE(w.rs_trend, m.rs_trend) as rs_trend,
                    -- Earnings from instruments
                    i.earnings_date,
                    CASE
                        WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
                        THEN (i.earnings_date - CURRENT_DATE)
                        ELSE NULL
                    END as days_to_earnings
                FROM stock_watchlist_results w
                LEFT JOIN stock_instruments i ON w.ticker = i.ticker
                LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
                    AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
                LEFT JOIN LATERAL (
                    SELECT ticker, quality_tier, signal_type
                    FROM stock_scanner_signals
                    WHERE ticker = w.ticker AND status = 'active'
                    ORDER BY composite_score DESC
                    LIMIT 1
                ) s ON true
                LEFT JOIN open_trades t ON w.ticker = t.ticker
                WHERE w.watchlist_name = %s
                AND w.status = 'active'
                ORDER BY w.volume DESC
            """
            results = _self._execute_query(query, (watchlist_name,))
        else:
            # For event watchlists: show entries for specific date, days is always 1
            query = """
                WITH open_trades AS (
                    SELECT DISTINCT
                        SPLIT_PART(ticker, '.', 1) as ticker,
                        side,
                        profit,
                        open_price,
                        current_price
                    FROM broker_trades
                    WHERE status = 'open'
                )
                SELECT
                    w.ticker,
                    i.name,
                    w.price,
                    w.volume,
                    w.avg_volume,
                    w.ema_20,
                    w.ema_50,
                    w.ema_200,
                    w.rsi_14,
                    w.macd,
                    w.macd_histogram,
                    w.gap_pct,
                    w.price_change_1d,
                    w.scan_date,
                    w.crossover_date,
                    1 as days_on_list,
                    CASE WHEN s.ticker IS NOT NULL THEN true ELSE false END as has_signal,
                    s.quality_tier as signal_tier,
                    s.signal_type,
                    COALESCE(w.avg_daily_change_5d, m.avg_daily_change_5d) as avg_daily_change_5d,
                    m.rs_vs_spy,
                    CASE WHEN t.ticker IS NOT NULL THEN true ELSE false END as in_trade,
                    t.side as trade_side,
                    t.profit as trade_profit,
                    w.daq_score,
                    w.daq_grade,
                    -- DAQ component scores
                    w.daq_mtf_score,
                    w.daq_volume_score,
                    w.daq_smc_score,
                    w.daq_quality_score,
                    w.daq_catalyst_score,
                    w.daq_news_score,
                    w.daq_regime_score,
                    w.daq_sector_score,
                    -- DAQ risk flags
                    w.daq_earnings_risk,
                    w.daq_high_short_interest,
                    w.daq_sector_underperforming,
                    -- Trading metrics (Trade Plan)
                    w.atr_14,
                    w.atr_percent,
                    w.swing_high,
                    w.swing_low,
                    w.swing_high_date,
                    w.swing_low_date,
                    w.nearest_ob_price,
                    w.nearest_ob_type,
                    w.nearest_ob_distance,
                    w.suggested_entry_low,
                    w.suggested_entry_high,
                    w.suggested_stop_loss,
                    w.suggested_target_1,
                    w.suggested_target_2,
                    w.risk_reward_ratio,
                    w.risk_percent,
                    w.volume_trend,
                    w.relative_volume,
                    -- RS trend from watchlist (if populated) or fallback to metrics
                    COALESCE(w.rs_percentile, m.rs_percentile) as rs_percentile,
                    COALESCE(w.rs_trend, m.rs_trend) as rs_trend,
                    -- Earnings from instruments
                    i.earnings_date,
                    CASE
                        WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
                        THEN (i.earnings_date - CURRENT_DATE)
                        ELSE NULL
                    END as days_to_earnings
                FROM stock_watchlist_results w
                LEFT JOIN stock_instruments i ON w.ticker = i.ticker
                LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
                    AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
                LEFT JOIN LATERAL (
                    SELECT ticker, quality_tier, signal_type
                    FROM stock_scanner_signals
                    WHERE ticker = w.ticker AND status = 'active'
                    ORDER BY composite_score DESC
                    LIMIT 1
                ) s ON true
                LEFT JOIN open_trades t ON w.ticker = t.ticker
                WHERE w.watchlist_name = %s
                AND w.scan_date = COALESCE(%s::date, (
                    SELECT MAX(scan_date)
                    FROM stock_watchlist_results
                    WHERE watchlist_name = %s
                ))
                ORDER BY w.volume DESC
            """
            results = _self._execute_query(query, (watchlist_name, scan_date, watchlist_name))

        return pd.DataFrame(results) if results else pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_watchlist_available_dates(_self, watchlist_name: str = None) -> List[str]:
        """
        Get list of available scan dates for watchlists (last 30 days).

        Args:
            watchlist_name: Optional filter by specific watchlist

        Returns:
            List of date strings in YYYY-MM-DD format, newest first
        """
        if watchlist_name:
            query = """
                SELECT DISTINCT scan_date
                FROM stock_watchlist_results
                WHERE watchlist_name = %s
                  AND scan_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY scan_date DESC
            """
            results = _self._execute_query(query, (watchlist_name,))
        else:
            query = """
                SELECT DISTINCT scan_date
                FROM stock_watchlist_results
                WHERE scan_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY scan_date DESC
            """
            results = _self._execute_query(query, ())

        return [str(r['scan_date']) for r in results] if results else []

    @st.cache_data(ttl=60)
    def get_watchlist_stats(_self, scan_date: str = None) -> Dict[str, Any]:
        """
        Get statistics for all predefined watchlists.

        For crossover watchlists: counts all active entries (ignores scan_date)
        For event watchlists: counts entries for the specified date

        Args:
            scan_date: Optional date string (YYYY-MM-DD). Only used for event watchlists.

        Returns dict with 'counts' (per watchlist), 'last_scan', and 'total_stocks_scanned'.
        """
        # Get counts for crossover watchlists (all active entries)
        crossover_query = """
            SELECT
                watchlist_name,
                COUNT(*) as stock_count,
                MAX(scan_date) as last_scan
            FROM stock_watchlist_results
            WHERE watchlist_name IN ('ema_50_crossover', 'ema_20_crossover', 'macd_bullish_cross')
            AND status = 'active'
            GROUP BY watchlist_name
        """
        crossover_results = _self._execute_query(crossover_query, ())

        # Get counts for event watchlists (specific date)
        if scan_date:
            event_query = """
                SELECT
                    watchlist_name,
                    COUNT(*) as stock_count,
                    MAX(scan_date) as last_scan
                FROM stock_watchlist_results
                WHERE watchlist_name IN ('gap_up_continuation', 'rsi_oversold_bounce')
                AND scan_date = %s::date
                GROUP BY watchlist_name
            """
            event_results = _self._execute_query(event_query, (scan_date,))
        else:
            event_query = """
                SELECT
                    watchlist_name,
                    COUNT(*) as stock_count,
                    MAX(scan_date) as last_scan
                FROM stock_watchlist_results
                WHERE watchlist_name IN ('gap_up_continuation', 'rsi_oversold_bounce')
                AND scan_date = (
                    SELECT MAX(scan_date) FROM stock_watchlist_results
                    WHERE watchlist_name IN ('gap_up_continuation', 'rsi_oversold_bounce')
                )
                GROUP BY watchlist_name
            """
            event_results = _self._execute_query(event_query, ())

        # Combine results
        all_results = crossover_results + event_results
        counts = {r['watchlist_name']: r['stock_count'] for r in all_results}

        # Get last scan date (most recent from any watchlist)
        last_scan = max((r['last_scan'] for r in all_results), default=None) if all_results else None

        # Get total stocks scanned (from stock_instruments)
        total_query = "SELECT COUNT(*) as total FROM stock_instruments WHERE is_active = true"
        total_result = _self._execute_query(total_query, ())
        total_stocks = total_result[0]['total'] if total_result else 0

        return {
            'counts': counts,
            'last_scan': last_scan,
            'total_stocks_scanned': total_stocks
        }

    @st.cache_data(ttl=60)
    def get_all_active_tickers_from_instruments(_self) -> List[str]:
        """
        Get all active tickers from stock_instruments table.
        Used for the new scanning approach that checks all stocks.
        """
        query = """
            SELECT ticker
            FROM stock_instruments
            WHERE is_active = true
            ORDER BY ticker
        """
        results = _self._execute_query(query, ())
        return [r['ticker'] for r in results]

    # =========================================================================
    # RELATIVE STRENGTH & MARKET CONTEXT
    # =========================================================================

    @st.cache_data(ttl=300)
    def get_market_regime(_self) -> Dict[str, Any]:
        """
        Get current market regime and breadth indicators.

        Returns:
            Dict with regime, SPY metrics, breadth indicators, and strategy recommendations
        """
        # First try to get from market_context table
        query = """
            SELECT
                calculation_date,
                market_regime,
                spy_price,
                spy_sma50,
                spy_sma200,
                spy_vs_sma50_pct,
                spy_vs_sma200_pct,
                spy_trend,
                pct_above_sma200,
                pct_above_sma50,
                pct_above_sma20,
                new_highs_count,
                new_lows_count,
                high_low_ratio,
                advancing_count,
                declining_count,
                ad_ratio,
                avg_atr_pct,
                volatility_regime,
                recommended_strategies
            FROM market_context
            ORDER BY calculation_date DESC
            LIMIT 1
        """
        results = _self._execute_query(query, ())
        if results:
            import json
            result = results[0]
            # Parse JSONB if needed
            if result.get('recommended_strategies') and isinstance(result['recommended_strategies'], str):
                result['recommended_strategies'] = json.loads(result['recommended_strategies'])
            return result

        # Fallback: Calculate from existing stock data
        return _self._calculate_market_regime_fallback()

    def _calculate_market_regime_fallback(_self) -> Dict[str, Any]:
        """
        Calculate market regime from existing stock screening metrics.

        This is used when the market_context table is empty.
        Derives market regime from breadth indicators when SPY data is not available.
        """
        # Get breadth from all stocks
        breadth_query = """
            SELECT
                COUNT(*) FILTER (WHERE current_price > sma_200) as above_200,
                COUNT(*) FILTER (WHERE current_price > sma_50) as above_50,
                COUNT(*) FILTER (WHERE current_price > sma_20) as above_20,
                COUNT(*) FILTER (WHERE trend_strength IN ('strong_up', 'up')) as bullish,
                COUNT(*) FILTER (WHERE trend_strength IN ('strong_down', 'down')) as bearish,
                COUNT(*) as total,
                AVG(atr_percent) as avg_atr,
                AVG(current_price) as avg_price
            FROM stock_screening_metrics
            WHERE calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        """
        breadth_result = _self._execute_query(breadth_query, ())

        if not breadth_result:
            return {}

        # Build regime data from breadth metrics
        b = breadth_result[0]
        total = int(b.get('total', 1) or 1)

        result = {}

        # Calculate breadth percentages
        pct_above_200 = (int(b.get('above_200', 0) or 0) / total) * 100 if total > 0 else 0
        pct_above_50 = (int(b.get('above_50', 0) or 0) / total) * 100 if total > 0 else 0
        pct_above_20 = (int(b.get('above_20', 0) or 0) / total) * 100 if total > 0 else 0

        result['pct_above_sma200'] = pct_above_200
        result['pct_above_sma50'] = pct_above_50
        result['pct_above_sma20'] = pct_above_20

        # Advance/Decline
        advancing = int(b.get('bullish', 0) or 0)
        declining = int(b.get('bearish', 0) or 0)
        result['advancing_count'] = advancing
        result['declining_count'] = declining
        result['ad_ratio'] = advancing / declining if declining > 0 else float(advancing)

        # Volatility
        avg_atr = float(b.get('avg_atr', 0) or 0)
        result['avg_atr_pct'] = avg_atr

        if avg_atr < 2:
            result['volatility_regime'] = 'low'
        elif avg_atr < 4:
            result['volatility_regime'] = 'normal'
        elif avg_atr < 6:
            result['volatility_regime'] = 'high'
        else:
            result['volatility_regime'] = 'extreme'

        # Placeholder for high/low counts
        result['new_highs_count'] = 0
        result['new_lows_count'] = 0
        result['high_low_ratio'] = 1.0

        # Derive market regime from breadth (since SPY data not available)
        # Bull: >60% above SMA200, >50% above SMA50
        # Bear: <40% above SMA200
        if pct_above_200 > 60 and pct_above_50 > 50:
            result['market_regime'] = 'bull_confirmed'
            result['spy_trend'] = 'rising'
        elif pct_above_200 > 50:
            result['market_regime'] = 'bull_weakening'
            result['spy_trend'] = 'flat'
        elif pct_above_200 > 40:
            result['market_regime'] = 'bear_weakening'
            result['spy_trend'] = 'flat'
        else:
            result['market_regime'] = 'bear_confirmed'
            result['spy_trend'] = 'falling'

        # Approximate SPY values from breadth (for display purposes)
        # These are estimates based on market breadth
        avg_price = float(b.get('avg_price', 0) or 0)
        result['spy_price'] = round(avg_price * 5, 2) if avg_price else 0  # Rough estimate
        result['spy_sma50'] = result['spy_price'] * 0.98 if pct_above_50 > 50 else result['spy_price'] * 1.02
        result['spy_sma200'] = result['spy_price'] * 0.95 if pct_above_200 > 50 else result['spy_price'] * 1.05
        result['spy_vs_sma50_pct'] = pct_above_50 - 50  # Approximation
        result['spy_vs_sma200_pct'] = pct_above_200 - 50  # Approximation

        # Strategy recommendations based on regime
        regime = result.get('market_regime', 'unknown')
        if regime == 'bull_confirmed':
            result['recommended_strategies'] = {
                'trend_following': 0.8,
                'breakout': 0.7,
                'pullback': 0.6,
                'mean_reversion': 0.2
            }
        elif regime == 'bull_weakening':
            result['recommended_strategies'] = {
                'trend_following': 0.5,
                'breakout': 0.4,
                'pullback': 0.7,
                'mean_reversion': 0.4
            }
        elif regime == 'bear_weakening':
            result['recommended_strategies'] = {
                'trend_following': 0.3,
                'breakout': 0.3,
                'pullback': 0.5,
                'mean_reversion': 0.6
            }
        else:
            result['recommended_strategies'] = {
                'trend_following': 0.2,
                'breakout': 0.2,
                'pullback': 0.3,
                'mean_reversion': 0.7
            }

        return result

    @st.cache_data(ttl=300)
    def get_sector_analysis(_self) -> pd.DataFrame:
        """
        Get sector rotation analysis with RS rankings.

        Returns:
            DataFrame with sector metrics sorted by RS vs SPY
        """
        query = """
            SELECT
                sector,
                sector_etf,
                sector_return_1d,
                sector_return_5d,
                sector_return_20d,
                rs_vs_spy,
                rs_percentile,
                rs_trend,
                stocks_in_sector,
                pct_above_sma50,
                pct_bullish_trend,
                top_stocks,
                sector_stage
            FROM sector_analysis
            WHERE calculation_date = (SELECT MAX(calculation_date) FROM sector_analysis)
            ORDER BY rs_vs_spy DESC
        """
        results = _self._execute_query(query, ())
        if results:
            import json
            df = pd.DataFrame(results)
            # Parse JSONB columns
            if 'top_stocks' in df.columns:
                df['top_stocks'] = df['top_stocks'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            return df

        # Fallback: Calculate from stock fundamentals and metrics
        return _self._calculate_sector_analysis_fallback()

    def _calculate_sector_analysis_fallback(_self) -> pd.DataFrame:
        """
        Calculate sector analysis from existing stock data.

        This is used when the sector_analysis table is empty.
        """
        # Sector ETF mapping
        sector_etfs = {
            'Technology': 'XLK',
            'Health Care': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Communication Services': 'XLC',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB',
        }

        query = """
            SELECT
                i.sector,
                COUNT(*) as stocks_in_sector,
                AVG(m.rs_vs_spy) as avg_rs,
                AVG(m.rs_percentile) as avg_rs_percentile,
                COUNT(*) FILTER (WHERE m.current_price > m.sma_50) * 100.0 / NULLIF(COUNT(*), 0) as pct_above_sma50,
                COUNT(*) FILTER (WHERE m.trend_strength IN ('strong_up', 'up')) * 100.0 / NULLIF(COUNT(*), 0) as pct_bullish_trend,
                AVG(m.price_change_1d) as sector_return_1d,
                AVG(m.price_change_5d) as sector_return_5d,
                AVG(m.price_change_20d) as sector_return_20d
            FROM stock_instruments i
            JOIN stock_screening_metrics m ON i.ticker = m.ticker
            WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
              AND i.sector IS NOT NULL
              AND i.sector <> ''
            GROUP BY i.sector
            ORDER BY avg_rs DESC NULLS LAST
        """
        results = _self._execute_query(query, ())

        if not results:
            return pd.DataFrame()

        sectors = []
        for r in results:
            sector = r.get('sector')
            avg_rs = r.get('avg_rs')
            avg_rs_pct = r.get('avg_rs_percentile')

            # Determine sector stage based on RS and trend
            pct_bullish = r.get('pct_bullish_trend', 0) or 0
            if avg_rs and avg_rs > 1.0:
                if pct_bullish > 50:
                    stage = 'leading'
                else:
                    stage = 'weakening'
            else:
                if pct_bullish > 40:
                    stage = 'improving'
                else:
                    stage = 'lagging'

            # Get top stocks in sector
            top_stocks = _self._get_top_stocks_in_sector(sector, limit=5)

            sectors.append({
                'sector': sector,
                'sector_etf': sector_etfs.get(sector, ''),
                'rs_vs_spy': avg_rs,
                'rs_percentile': int(avg_rs_pct) if avg_rs_pct else None,
                'rs_trend': 'stable',  # Would need historical data for trend
                'sector_return_1d': r.get('sector_return_1d'),
                'sector_return_5d': r.get('sector_return_5d'),
                'sector_return_20d': r.get('sector_return_20d'),
                'stocks_in_sector': r.get('stocks_in_sector', 0),
                'pct_above_sma50': r.get('pct_above_sma50'),
                'pct_bullish_trend': r.get('pct_bullish_trend'),
                'sector_stage': stage,
                'top_stocks': top_stocks,
            })

        return pd.DataFrame(sectors)

    def _get_top_stocks_in_sector(_self, sector: str, limit: int = 5) -> list:
        """Get top stocks in a sector by RS percentile."""
        query = """
            SELECT
                m.ticker,
                m.rs_percentile,
                m.rs_trend,
                m.current_price
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON m.ticker = i.ticker
            WHERE i.sector = %s
              AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
              AND m.rs_percentile IS NOT NULL
            ORDER BY m.rs_percentile DESC
            LIMIT %s
        """
        results = _self._execute_query(query, (sector, limit))
        return [
            {
                'ticker': r.get('ticker'),
                'rs_percentile': r.get('rs_percentile'),
                'rs_trend': r.get('rs_trend'),
                'price': float(r.get('current_price')) if r.get('current_price') else None
            }
            for r in results
        ] if results else []

    @st.cache_data(ttl=300)
    def get_rs_leaders(_self, min_rs_percentile: int = 70, limit: int = 50) -> pd.DataFrame:
        """
        Get stocks with strong relative strength.

        Args:
            min_rs_percentile: Minimum RS percentile (default 70 = top 30%)
            limit: Maximum results

        Returns:
            DataFrame with RS leaders
        """
        query = """
            SELECT
                m.ticker,
                i.name,
                i.sector,
                m.current_price,
                m.rs_vs_spy,
                m.rs_percentile,
                m.rs_trend,
                m.price_change_20d,
                m.trend_strength,
                m.ma_alignment,
                m.atr_percent,
                m.rsi_14,
                m.pct_from_52w_high
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON m.ticker = i.ticker
            WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
              AND m.rs_percentile >= %s
              AND m.rs_percentile IS NOT NULL
            ORDER BY m.rs_percentile DESC
            LIMIT %s
        """
        results = _self._execute_query(query, (min_rs_percentile, limit))
        return pd.DataFrame(results) if results else pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_smc_zones_for_ticker(_self, ticker: str) -> Dict[str, Any]:
        """
        Get Smart Money Concepts data for a specific ticker.

        Returns:
            Dict with SMC zones, BOS/CHOCH events, and current analysis
            in the format expected by the chart component:
            - trend: SMC trend direction
            - bias: Market bias
            - confluence_score: SMC confluence score
            - premium_discount_zone: Current zone
            - order_blocks: List of order block dicts
            - bos_choch_events: List of BOS/CHOCH events
        """
        query = """
            SELECT
                smc_trend,
                smc_bias,
                last_bos_type,
                last_bos_date,
                last_bos_price,
                last_choch_type,
                last_choch_date,
                swing_high,
                swing_low,
                swing_high_date,
                swing_low_date,
                premium_discount_zone,
                zone_position,
                weekly_range_high,
                weekly_range_low,
                nearest_ob_type,
                nearest_ob_price,
                nearest_ob_distance,
                smc_confluence_score,
                current_price
            FROM stock_screening_metrics
            WHERE ticker = %s
              AND calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        """
        results = _self._execute_query(query, (ticker,))
        if not results:
            return {}

        row = results[0]

        # Transform to format expected by chart component
        smc_data = {
            'trend': row.get('smc_trend', 'Unknown'),
            'bias': row.get('smc_bias', 'Unknown'),
            'confluence_score': row.get('smc_confluence_score', 0),
            'premium_discount_zone': row.get('premium_discount_zone', 'Unknown'),
            'zone_position': row.get('zone_position'),
            'swing_high': row.get('swing_high'),
            'swing_low': row.get('swing_low'),
            'swing_high_date': row.get('swing_high_date'),
            'swing_low_date': row.get('swing_low_date'),
            'weekly_range_high': row.get('weekly_range_high'),
            'weekly_range_low': row.get('weekly_range_low'),
            'order_blocks': [],
            'bos_choch_events': []
        }

        # Build order blocks from nearest OB data
        # Create a supply/demand zone from swing high/low
        swing_high = row.get('swing_high')
        swing_low = row.get('swing_low')
        current_price = row.get('current_price')
        nearest_ob_type = row.get('nearest_ob_type')
        nearest_ob_price = row.get('nearest_ob_price')

        if swing_high and swing_low:
            range_size = float(swing_high) - float(swing_low)
            zone_buffer = range_size * 0.03  # 3% buffer for zones

            # Add supply zone (around swing high)
            smc_data['order_blocks'].append({
                'type': 'bearish',
                'price_high': float(swing_high),
                'price_low': float(swing_high) - zone_buffer,
                'strength': 'strong'
            })

            # Add demand zone (around swing low)
            smc_data['order_blocks'].append({
                'type': 'bullish',
                'price_high': float(swing_low) + zone_buffer,
                'price_low': float(swing_low),
                'strength': 'strong'
            })

            # Add nearest order block if different from swing levels
            if nearest_ob_price and nearest_ob_type:
                ob_price = float(nearest_ob_price)
                if abs(ob_price - float(swing_high)) > zone_buffer and abs(ob_price - float(swing_low)) > zone_buffer:
                    smc_data['order_blocks'].append({
                        'type': nearest_ob_type.lower() if nearest_ob_type else 'bullish',
                        'price_high': ob_price + zone_buffer/2,
                        'price_low': ob_price - zone_buffer/2,
                        'strength': 'moderate'
                    })

        # Build BOS/CHOCH events list
        if row.get('last_bos_type') and row.get('last_bos_price'):
            smc_data['bos_choch_events'].append({
                'event_type': f"BOS ({row.get('last_bos_type')})",
                'price': float(row.get('last_bos_price')),
                'date': row.get('last_bos_date')
            })

        if row.get('last_choch_type') and row.get('last_choch_date'):
            # CHOCH doesn't have a specific price in the DB, use swing level
            choch_price = swing_high if row.get('last_choch_type') == 'Bearish' else swing_low
            if choch_price:
                smc_data['bos_choch_events'].append({
                    'event_type': f"CHOCH ({row.get('last_choch_type')})",
                    'price': float(choch_price),
                    'date': row.get('last_choch_date')
                })

        return smc_data

    @st.cache_data(ttl=300)
    def get_position_sizing_data(_self, ticker: str) -> Dict[str, Any]:
        """
        Get data needed for position sizing calculations.

        Args:
            ticker: Stock ticker

        Returns:
            Dict with current price, ATR, recent signals with stop loss levels
        """
        # Get current price and ATR
        metrics_query = """
            SELECT
                current_price,
                atr_14,
                atr_percent,
                avg_daily_change_5d
            FROM stock_screening_metrics
            WHERE ticker = %s
              AND calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        """
        metrics = _self._execute_query(metrics_query, (ticker,))

        # Get most recent signal with stop loss
        signal_query = """
            SELECT
                entry_price,
                stop_loss,
                take_profit_1,
                risk_reward_ratio,
                risk_percent
            FROM stock_scanner_signals
            WHERE ticker = %s
              AND status = 'active'
            ORDER BY signal_timestamp DESC
            LIMIT 1
        """
        signals = _self._execute_query(signal_query, (ticker,))

        result = {
            'ticker': ticker,
            'metrics': metrics[0] if metrics else {},
            'signal': signals[0] if signals else {}
        }

        # Calculate suggested stop based on ATR if no signal
        if metrics and not signals:
            atr = float(metrics[0].get('atr_14') or 0)
            price = float(metrics[0].get('current_price') or 0)
            if atr > 0 and price > 0:
                result['suggested_stop_long'] = round(price - (2 * atr), 2)
                result['suggested_stop_short'] = round(price + (2 * atr), 2)

        return result

    @st.cache_data(ttl=60)
    def get_custom_screen_results(_self, filters: List[Dict], sort_by: str = 'rs_percentile', limit: int = 100) -> pd.DataFrame:
        """
        Execute a custom screen with flexible filters.

        Args:
            filters: List of filter dicts with 'field', 'operator', 'value'
                     e.g., [{"field": "rs_percentile", "operator": ">=", "value": 80}]
            sort_by: Column to sort by (default: rs_percentile)
            limit: Maximum results

        Returns:
            DataFrame with matching stocks
        """
        # Base query
        base_query = """
            SELECT
                m.ticker,
                i.name,
                i.sector,
                m.current_price,
                m.rs_vs_spy,
                m.rs_percentile,
                m.rs_trend,
                m.price_change_1d,
                m.price_change_5d,
                m.price_change_20d,
                m.trend_strength,
                m.ma_alignment,
                m.rsi_14,
                m.atr_percent,
                m.relative_volume,
                m.smc_trend,
                m.smc_confluence_score,
                m.pct_from_52w_high
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON m.ticker = i.ticker
            WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        """

        conditions = []
        params = []

        # Valid operators
        valid_operators = {'=', '!=', '>', '>=', '<', '<=', 'IN', 'LIKE', 'ILIKE'}

        for f in filters:
            field = f.get('field', '')
            operator = f.get('operator', '=')
            value = f.get('value')

            # Sanitize field name (only allow known columns)
            allowed_fields = {
                'rs_percentile', 'rs_vs_spy', 'rs_trend', 'price_change_1d', 'price_change_5d',
                'price_change_20d', 'trend_strength', 'ma_alignment', 'rsi_14', 'atr_percent',
                'relative_volume', 'smc_trend', 'smc_confluence_score', 'pct_from_52w_high',
                'sector', 'current_price'
            }

            if field not in allowed_fields:
                continue

            if operator.upper() not in valid_operators:
                continue

            # Build condition
            if operator.upper() == 'IN':
                if isinstance(value, list):
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"m.{field} IN ({placeholders})")
                    params.extend(value)
            else:
                conditions.append(f"m.{field} {operator} %s")
                params.append(value)

        # Add conditions to query
        if conditions:
            base_query += " AND " + " AND ".join(conditions)

        # Add sorting
        if sort_by in ['rs_percentile', 'rs_vs_spy', 'price_change_20d', 'rsi_14', 'atr_percent', 'relative_volume']:
            base_query += f" ORDER BY m.{sort_by} DESC NULLS LAST"
        else:
            base_query += " ORDER BY m.rs_percentile DESC NULLS LAST"

        base_query += f" LIMIT {limit}"

        results = _self._execute_query(base_query, tuple(params) if params else None)
        return pd.DataFrame(results) if results else pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_hourly_candles(_self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get hourly candle data for multi-timeframe charting."""
        query = """
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM stock_candles
            WHERE ticker = %s
              AND timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp ASC
        """
        results = _self._execute_query(query % ('%s', days), (ticker,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_weekly_candles(_self, ticker: str, weeks: int = 52) -> pd.DataFrame:
        """Get weekly candle data synthesized from daily."""
        query = """
            SELECT
                date_trunc('week', timestamp) as timestamp,
                (array_agg(open ORDER BY timestamp))[1] as open,
                MAX(high) as high,
                MIN(low) as low,
                (array_agg(close ORDER BY timestamp DESC))[1] as close,
                SUM(volume) as volume
            FROM stock_candles_synthesized
            WHERE ticker = %s
              AND timeframe = '1d'
              AND timestamp >= NOW() - INTERVAL '%s weeks'
            GROUP BY date_trunc('week', timestamp)
            ORDER BY timestamp ASC
        """
        results = _self._execute_query(query % ('%s', weeks), (ticker,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    # =========================================================================
    # CHART MARKERS - Signals and Watchlist Events
    # =========================================================================

    @st.cache_data(ttl=300)
    def get_chart_markers_for_ticker(_self, ticker: str, days: int = 90) -> Dict[str, List[Dict]]:
        """
        Get signal and watchlist events for chart markers.

        Returns a dict with:
        - 'signals': List of signal events with date, type, and metadata
        - 'watchlist_events': List of watchlist events with date, watchlist name, and type

        Args:
            ticker: Stock ticker symbol
            days: Lookback period in days

        Returns:
            Dict with 'signals' and 'watchlist_events' lists
        """
        result = {
            'signals': [],
            'watchlist_events': []
        }

        # Get signal history for this ticker
        signal_query = """
            SELECT
                signal_timestamp::date as event_date,
                signal_timestamp,
                signal_type,
                scanner_name,
                quality_tier,
                composite_score,
                entry_price,
                stop_loss,
                take_profit_1,
                status
            FROM stock_scanner_signals
            WHERE ticker = %s
              AND signal_timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY signal_timestamp DESC
        """
        signals = _self._execute_query(signal_query % ('%s', days), (ticker,))
        result['signals'] = signals if signals else []

        # Get watchlist history for this ticker
        # For crossover watchlists: use crossover_date as the event date
        # For event watchlists: use scan_date
        # Query both active entries AND historical entries within the lookback period
        watchlist_query = """
            SELECT
                COALESCE(crossover_date, scan_date) as event_date,
                scan_date,
                crossover_date,
                watchlist_name,
                status,
                price,
                rsi_14,
                gap_pct,
                macd,
                macd_signal
            FROM stock_watchlist_results
            WHERE ticker = %s
              AND (
                  -- Include active entries regardless of date (shows current watchlist membership)
                  status = 'active'
                  OR
                  -- Include historical entries where event happened in lookback period
                  COALESCE(crossover_date, scan_date) >= CURRENT_DATE - INTERVAL '%s days'
              )
            ORDER BY COALESCE(crossover_date, scan_date) DESC
        """
        watchlist_events = _self._execute_query(watchlist_query % ('%s', days), (ticker,))
        result['watchlist_events'] = watchlist_events if watchlist_events else []

        return result

    @st.cache_data(ttl=300)
    def get_watchlist_history_for_ticker(_self, ticker: str, days: int = 90) -> pd.DataFrame:
        """
        Get complete watchlist history for a specific ticker.

        Shows all watchlist appearances with dates and relevant metrics.

        Args:
            ticker: Stock ticker symbol
            days: Lookback period in days

        Returns:
            DataFrame with watchlist history
        """
        query = """
            SELECT
                w.scan_date,
                w.crossover_date,
                w.watchlist_name,
                w.status,
                w.price,
                w.volume,
                w.rsi_14,
                w.ema_20,
                w.ema_50,
                w.ema_200,
                w.macd,
                w.macd_signal,
                w.gap_pct,
                w.price_change_1d,
                CASE
                    WHEN w.watchlist_name IN ('ema_50_crossover', 'ema_20_crossover', 'macd_bullish_cross')
                    THEN (CURRENT_DATE - w.crossover_date) + 1
                    ELSE 1
                END as days_on_list
            FROM stock_watchlist_results w
            WHERE w.ticker = %s
              AND w.scan_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY w.scan_date DESC, w.watchlist_name
        """
        results = _self._execute_query(query % ('%s', days), (ticker,))
        return pd.DataFrame(results) if results else pd.DataFrame()

    # =========================================================================
    # DEEP ANALYSIS QUERIES
    # =========================================================================

    @st.cache_data(ttl=300)
    def get_deep_analysis_for_signals(_self, signal_ids: List[int] = None) -> Dict[int, Dict]:
        """
        Get deep analysis data for signals.

        Args:
            signal_ids: List of signal IDs to fetch (None = all recent)

        Returns:
            Dict mapping signal_id to deep analysis data
        """
        if signal_ids:
            placeholders = ','.join(['%s'] * len(signal_ids))
            query = f"""
                SELECT
                    signal_id,
                    ticker,
                    daq_score,
                    daq_grade,
                    mtf_score,
                    volume_score,
                    smc_score,
                    quality_score,
                    catalyst_score,
                    news_score,
                    regime_score,
                    sector_score,
                    earnings_within_7d,
                    high_short_interest,
                    sector_underperforming,
                    analysis_timestamp
                FROM stock_deep_analysis
                WHERE signal_id IN ({placeholders})
            """
            results = _self._execute_query(query, tuple(signal_ids))
        else:
            query = """
                SELECT
                    signal_id,
                    ticker,
                    daq_score,
                    daq_grade,
                    mtf_score,
                    volume_score,
                    smc_score,
                    quality_score,
                    catalyst_score,
                    news_score,
                    regime_score,
                    sector_score,
                    earnings_within_7d,
                    high_short_interest,
                    sector_underperforming,
                    analysis_timestamp
                FROM stock_deep_analysis
                WHERE analysis_timestamp > NOW() - INTERVAL '7 days'
            """
            results = _self._execute_query(query, ())

        return {r['signal_id']: r for r in results} if results else {}

    @st.cache_data(ttl=300)
    def get_deep_analysis_summary(_self, days: int = 7) -> Dict[str, Any]:
        """
        Get summary statistics for deep analysis.

        Args:
            days: Number of days to look back

        Returns:
            Dict with summary statistics
        """
        query = """
            SELECT
                COUNT(*) as total,
                ROUND(AVG(daq_score)::numeric, 1) as avg_daq,
                ROUND(AVG(mtf_score)::numeric, 1) as avg_mtf,
                ROUND(AVG(volume_score)::numeric, 1) as avg_volume,
                ROUND(AVG(smc_score)::numeric, 1) as avg_smc,
                ROUND(AVG(quality_score)::numeric, 1) as avg_quality,
                ROUND(AVG(news_score)::numeric, 1) as avg_news,
                ROUND(AVG(regime_score)::numeric, 1) as avg_regime,
                ROUND(AVG(sector_score)::numeric, 1) as avg_sector,
                SUM(CASE WHEN daq_grade = 'A+' THEN 1 ELSE 0 END) as grade_a_plus,
                SUM(CASE WHEN daq_grade = 'A' THEN 1 ELSE 0 END) as grade_a,
                SUM(CASE WHEN daq_grade = 'B' THEN 1 ELSE 0 END) as grade_b,
                SUM(CASE WHEN daq_grade = 'C' THEN 1 ELSE 0 END) as grade_c,
                SUM(CASE WHEN daq_grade = 'D' THEN 1 ELSE 0 END) as grade_d
            FROM stock_deep_analysis
            WHERE analysis_timestamp > NOW() - INTERVAL '%s days'
        """
        results = _self._execute_query(query % days, ())
        return results[0] if results else {}

    @st.cache_data(ttl=300)
    def get_deep_analysis_detail(_self, signal_id: int) -> Dict[str, Any]:
        """
        Get detailed deep analysis for a specific signal.

        Args:
            signal_id: Signal ID

        Returns:
            Dict with full deep analysis data including JSONB details
        """
        query = """
            SELECT
                d.*,
                s.ticker,
                s.scanner_name,
                s.quality_tier as signal_tier,
                s.composite_score as signal_score,
                s.signal_type,
                s.entry_price,
                s.stop_loss,
                s.take_profit_1
            FROM stock_deep_analysis d
            JOIN stock_scanner_signals s ON d.signal_id = s.id
            WHERE d.signal_id = %s
        """
        results = _self._execute_query(query, (signal_id,))
        return results[0] if results else {}

    @st.cache_data(ttl=300)
    def get_signals_with_deep_analysis(_self, hours: int = 24, min_daq: int = None) -> pd.DataFrame:
        """
        Get signals joined with deep analysis data.

        Args:
            hours: Hours to look back
            min_daq: Minimum DAQ score filter

        Returns:
            DataFrame with signals and deep analysis
        """
        query = """
            SELECT
                s.id as signal_id,
                s.signal_timestamp,
                s.ticker,
                i.name,
                s.signal_type,
                s.entry_price,
                s.stop_loss,
                s.take_profit_1 as take_profit,
                s.composite_score as confidence,
                s.scanner_name,
                s.quality_tier,
                s.claude_grade,
                s.claude_action,
                s.news_sentiment_score,
                s.news_sentiment_level,
                -- Deep Analysis fields
                d.daq_score,
                d.daq_grade,
                d.mtf_score,
                d.volume_score,
                d.smc_score,
                d.quality_score as daq_quality_score,
                d.news_score as daq_news_score,
                d.regime_score,
                d.sector_score,
                d.earnings_within_7d,
                d.high_short_interest,
                d.sector_underperforming,
                -- Watchlist data
                w.tier as watchlist_tier,
                w.score as watchlist_score
            FROM stock_scanner_signals s
            LEFT JOIN stock_instruments i ON s.ticker = i.ticker
            LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
            LEFT JOIN stock_watchlist w ON s.ticker = w.ticker
                AND w.calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            WHERE s.signal_timestamp > NOW() - INTERVAL '%s hours'
            AND s.status = 'active'
        """

        if min_daq:
            query += f" AND d.daq_score >= {min_daq}"

        query += """
            ORDER BY
                CASE WHEN d.daq_score IS NOT NULL THEN 0 ELSE 1 END,
                d.daq_score DESC NULLS LAST,
                s.signal_timestamp DESC
        """

        results = _self._execute_query(query % hours, ())
        return pd.DataFrame(results) if results else pd.DataFrame()


# Service instance - no caching to allow code changes to take effect
# Individual methods use @st.cache_data for query caching
_service_instance = None


def get_stock_service() -> StockAnalyticsService:
    """Get or create the stock analytics service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = StockAnalyticsService()
    return _service_instance
