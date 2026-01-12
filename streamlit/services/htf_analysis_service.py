"""
HTF (Higher Timeframe) Analysis Service

Provides data and statistics for analyzing 4H candle direction correlation
with trade outcomes. Used by the HTF Analysis tab.
"""

import streamlit as st
import pandas as pd
import logging
from contextlib import contextmanager
from typing import Dict
from services.db_utils import get_psycopg2_pool

logger = logging.getLogger(__name__)


@contextmanager
def get_db_connection(db_type: str = "trading"):
    """Get a connection from the centralized pool."""
    pool = get_psycopg2_pool(db_type)
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


class HTFAnalysisService:
    """Service for HTF candle direction analysis."""

    @st.cache_data(ttl=180)
    def get_htf_direction_stats(_self, days: int = 30) -> Dict:
        """
        Get statistics on 4H candle direction at signal time.

        Returns breakdown of:
        - Signal direction vs 4H candle alignment
        - Win rate by alignment status
        - Distribution of 4H directions
        """
        try:
            with get_db_connection("trading") as conn:
                query = """
                WITH alert_outcomes AS (
                    SELECT
                        ah.id as alert_id,
                        ah.alert_timestamp,
                        ah.epic,
                        ah.pair,
                        ah.signal_type,
                        ah.htf_candle_direction,
                        ah.htf_candle_direction_prev,
                        ah.claude_approved,
                        tl.id as trade_id,
                        tl.profit_loss,
                        tl.status as trade_status,
                        CASE
                            WHEN tl.profit_loss > 0 THEN 'WIN'
                            WHEN tl.profit_loss < 0 THEN 'LOSS'
                            WHEN tl.profit_loss = 0 THEN 'BREAKEVEN'
                            ELSE 'PENDING'
                        END as outcome
                    FROM alert_history ah
                    LEFT JOIN trade_log tl ON tl.alert_id = ah.id
                    WHERE ah.alert_timestamp >= NOW() - INTERVAL '%s days'
                      AND ah.htf_candle_direction IS NOT NULL
                )
                SELECT
                    signal_type,
                    htf_candle_direction,
                    htf_candle_direction_prev,
                    outcome,
                    trade_status,
                    COUNT(*) as count,
                    SUM(CASE WHEN profit_loss IS NOT NULL THEN profit_loss ELSE 0 END) as total_pnl,
                    AVG(CASE WHEN profit_loss IS NOT NULL THEN profit_loss ELSE NULL END) as avg_pnl
                FROM alert_outcomes
                GROUP BY signal_type, htf_candle_direction, htf_candle_direction_prev, outcome, trade_status
                ORDER BY signal_type, htf_candle_direction, outcome
                """

                df = pd.read_sql_query(query, conn, params=[days])

                if df.empty:
                    return {'has_data': False}

                return {
                    'has_data': True,
                    'raw_data': df,
                    'days': days
                }

        except Exception as e:
            logger.error(f"Error fetching HTF direction stats: {e}")
            return {'has_data': False, 'error': str(e)}

    @st.cache_data(ttl=180)
    def get_alignment_analysis(_self, days: int = 30) -> pd.DataFrame:
        """
        Analyze signal alignment with 4H candle direction.

        Aligned: BULL signal + BULLISH 4H candle, or BEAR signal + BEARISH 4H candle
        Counter: BULL signal + BEARISH 4H candle, or BEAR signal + BULLISH 4H candle
        """
        try:
            with get_db_connection("trading") as conn:
                query = """
                WITH alert_outcomes AS (
                    SELECT
                        ah.id as alert_id,
                        ah.signal_type,
                        ah.htf_candle_direction,
                        ah.htf_candle_direction_prev,
                        CASE
                            WHEN (ah.signal_type = 'BULL' AND ah.htf_candle_direction = 'BULLISH')
                              OR (ah.signal_type = 'BEAR' AND ah.htf_candle_direction = 'BEARISH')
                            THEN 'ALIGNED'
                            WHEN (ah.signal_type = 'BULL' AND ah.htf_candle_direction = 'BEARISH')
                              OR (ah.signal_type = 'BEAR' AND ah.htf_candle_direction = 'BULLISH')
                            THEN 'COUNTER'
                            ELSE 'NEUTRAL'
                        END as alignment,
                        tl.profit_loss,
                        CASE
                            WHEN tl.profit_loss > 0 THEN 'WIN'
                            WHEN tl.profit_loss < 0 THEN 'LOSS'
                            WHEN tl.profit_loss = 0 THEN 'BREAKEVEN'
                            ELSE NULL
                        END as outcome
                    FROM alert_history ah
                    LEFT JOIN trade_log tl ON tl.alert_id = ah.id
                    WHERE ah.alert_timestamp >= NOW() - INTERVAL '%s days'
                      AND ah.htf_candle_direction IS NOT NULL
                )
                SELECT
                    alignment,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) as total_trades,
                    COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) as wins,
                    COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) as losses,
                    COUNT(CASE WHEN outcome = 'BREAKEVEN' THEN 1 END) as breakeven,
                    ROUND(
                        CASE WHEN COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) > 0
                        THEN COUNT(CASE WHEN outcome = 'WIN' THEN 1 END)::numeric /
                             COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) * 100
                        ELSE 0 END, 1
                    ) as win_rate,
                    COALESCE(SUM(profit_loss), 0) as total_pnl,
                    ROUND(COALESCE(AVG(profit_loss), 0)::numeric, 2) as avg_pnl
                FROM alert_outcomes
                GROUP BY alignment
                ORDER BY alignment
                """

                df = pd.read_sql_query(query, conn, params=[days])
                return df

        except Exception as e:
            logger.error(f"Error fetching alignment analysis: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=180)
    def get_two_candle_pattern_analysis(_self, days: int = 30) -> pd.DataFrame:
        """
        Analyze two-candle patterns (current + previous 4H candle).

        Patterns like:
        - BULLISH_BULLISH (momentum continuation)
        - BEARISH_BULLISH (potential reversal)
        - etc.
        """
        try:
            with get_db_connection("trading") as conn:
                query = """
                WITH alert_outcomes AS (
                    SELECT
                        ah.id as alert_id,
                        ah.signal_type,
                        ah.htf_candle_direction || '_' || COALESCE(ah.htf_candle_direction_prev, 'UNKNOWN') as pattern,
                        tl.profit_loss,
                        CASE
                            WHEN tl.profit_loss > 0 THEN 'WIN'
                            WHEN tl.profit_loss < 0 THEN 'LOSS'
                            WHEN tl.profit_loss = 0 THEN 'BREAKEVEN'
                            ELSE NULL
                        END as outcome
                    FROM alert_history ah
                    LEFT JOIN trade_log tl ON tl.alert_id = ah.id
                    WHERE ah.alert_timestamp >= NOW() - INTERVAL '%s days'
                      AND ah.htf_candle_direction IS NOT NULL
                )
                SELECT
                    pattern,
                    signal_type,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) as total_trades,
                    COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) as wins,
                    COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) as losses,
                    ROUND(
                        CASE WHEN COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) > 0
                        THEN COUNT(CASE WHEN outcome = 'WIN' THEN 1 END)::numeric /
                             COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) * 100
                        ELSE 0 END, 1
                    ) as win_rate,
                    COALESCE(SUM(profit_loss), 0) as total_pnl,
                    ROUND(COALESCE(AVG(profit_loss), 0)::numeric, 2) as avg_pnl
                FROM alert_outcomes
                GROUP BY pattern, signal_type
                HAVING COUNT(*) >= 2
                ORDER BY total_trades DESC, win_rate DESC
                """

                df = pd.read_sql_query(query, conn, params=[days])
                return df

        except Exception as e:
            logger.error(f"Error fetching two-candle pattern analysis: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=180)
    def get_pair_htf_performance(_self, days: int = 30) -> pd.DataFrame:
        """
        Analyze HTF alignment performance by currency pair.
        """
        try:
            with get_db_connection("trading") as conn:
                query = """
                WITH alert_outcomes AS (
                    SELECT
                        ah.pair,
                        ah.signal_type,
                        ah.htf_candle_direction,
                        CASE
                            WHEN (ah.signal_type = 'BULL' AND ah.htf_candle_direction = 'BULLISH')
                              OR (ah.signal_type = 'BEAR' AND ah.htf_candle_direction = 'BEARISH')
                            THEN 'ALIGNED'
                            ELSE 'COUNTER'
                        END as alignment,
                        tl.profit_loss,
                        CASE
                            WHEN tl.profit_loss > 0 THEN 'WIN'
                            WHEN tl.profit_loss < 0 THEN 'LOSS'
                            ELSE NULL
                        END as outcome
                    FROM alert_history ah
                    LEFT JOIN trade_log tl ON tl.alert_id = ah.id
                    WHERE ah.alert_timestamp >= NOW() - INTERVAL '%s days'
                      AND ah.htf_candle_direction IS NOT NULL
                      AND ah.htf_candle_direction != 'NEUTRAL'
                )
                SELECT
                    pair,
                    alignment,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) as total_trades,
                    COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) as wins,
                    COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) as losses,
                    ROUND(
                        CASE WHEN COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) > 0
                        THEN COUNT(CASE WHEN outcome = 'WIN' THEN 1 END)::numeric /
                             COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) * 100
                        ELSE 0 END, 1
                    ) as win_rate,
                    COALESCE(SUM(profit_loss), 0) as total_pnl
                FROM alert_outcomes
                GROUP BY pair, alignment
                HAVING COUNT(*) >= 2
                ORDER BY pair, alignment
                """

                df = pd.read_sql_query(query, conn, params=[days])
                return df

        except Exception as e:
            logger.error(f"Error fetching pair HTF performance: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=180)
    def get_htf_direction_distribution(_self, days: int = 30) -> pd.DataFrame:
        """
        Get distribution of 4H candle directions at signal time.
        """
        try:
            with get_db_connection("trading") as conn:
                query = """
                SELECT
                    htf_candle_direction as direction,
                    signal_type,
                    COUNT(*) as count
                FROM alert_history
                WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
                  AND htf_candle_direction IS NOT NULL
                GROUP BY htf_candle_direction, signal_type
                ORDER BY htf_candle_direction, signal_type
                """

                df = pd.read_sql_query(query, conn, params=[days])
                return df

        except Exception as e:
            logger.error(f"Error fetching HTF direction distribution: {e}")
            return pd.DataFrame()
