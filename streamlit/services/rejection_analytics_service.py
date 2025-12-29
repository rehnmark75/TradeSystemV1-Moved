"""
Rejection Analytics Service

Provides data access layer for SMC and EMA strategy rejection analytics.
All fetch methods are cached with @st.cache_data for performance.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import logging

from services.db_utils import get_psycopg2_connection

logger = logging.getLogger(__name__)


class RejectionAnalyticsService:
    """Service for fetching strategy rejection data from the trading database."""

    def __init__(self):
        self.db_type = "trading"

    def _get_connection(self):
        """Get a database connection from the pool."""
        return get_psycopg2_connection(self.db_type)

    # =========================================================================
    # SMC SIMPLE REJECTIONS
    # =========================================================================

    @st.cache_data(ttl=300)
    def fetch_smc_rejections(
        _self, days: int, stage_filter: str, pair_filter: str, session_filter: str
    ) -> pd.DataFrame:
        """
        Fetch SMC Simple strategy rejection data from database (cached 5 min).

        Args:
            days: Number of days to look back
            stage_filter: Rejection stage or 'All'
            pair_filter: Currency pair or 'All'
            session_filter: Market session or 'All'

        Returns:
            DataFrame with rejection data
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                id,
                scan_timestamp,
                epic,
                pair,
                rejection_stage,
                rejection_reason,
                attempted_direction,
                current_price,
                market_hour,
                market_session,
                ema_4h_value,
                ema_distance_pips,
                price_position_vs_ema,
                atr_15m,
                atr_percentile,
                volume_ratio,
                swing_high_level,
                swing_low_level,
                pullback_depth,
                fib_zone,
                swing_range_pips,
                potential_entry,
                potential_stop_loss,
                potential_take_profit,
                potential_risk_pips,
                potential_reward_pips,
                potential_rr_ratio,
                confidence_score,
                strategy_version,
                sr_blocking_level,
                sr_blocking_type,
                sr_blocking_distance_pips,
                sr_path_blocked_pct,
                target_distance_pips
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            """

            params = [days]

            if stage_filter != "All":
                query += " AND rejection_stage = %s"
                params.append(stage_filter)

            if pair_filter != "All":
                query += " AND (pair = %s OR epic LIKE %s)"
                params.append(pair_filter)
                params.append(f"%{pair_filter}%")

            if session_filter != "All":
                query += " AND market_session = %s"
                params.append(session_filter)

            query += " ORDER BY scan_timestamp DESC LIMIT 1000"

            df = pd.read_sql_query(query, conn, params=params)
            return df

        except Exception as e:
            if "does not exist" in str(e):
                logger.info("SMC Rejections table not yet created")
            else:
                logger.error(f"Error fetching SMC rejections: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_smc_rejection_stats(_self, days: int) -> Dict[str, Any]:
        """
        Fetch aggregated SMC rejection statistics (cached 5 min).

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with aggregated statistics
        """
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            # Total and by-stage counts
            query = """
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT epic) as unique_pairs,
                rejection_stage,
                COUNT(*) as stage_count
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY rejection_stage
            ORDER BY stage_count DESC
            """

            df = pd.read_sql_query(query, conn, params=[days])

            if df.empty:
                return {'total': 0, 'unique_pairs': 0, 'by_stage': {}}

            stats = {
                'total': int(df['total'].iloc[0]) if not df.empty else 0,
                'unique_pairs': int(df['unique_pairs'].iloc[0]) if not df.empty else 0,
                'by_stage': dict(zip(df['rejection_stage'], df['stage_count']))
            }

            # Near-miss count (high confidence rejects)
            near_miss_query = """
            SELECT COUNT(*) as near_misses
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            AND rejection_stage = 'CONFIDENCE'
            AND confidence_score >= 0.45
            """
            near_miss_df = pd.read_sql_query(near_miss_query, conn, params=[days])
            stats['near_misses'] = int(near_miss_df['near_misses'].iloc[0]) if not near_miss_df.empty else 0

            # Most rejected pair
            pair_query = """
            SELECT pair, COUNT(*) as count
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY pair
            ORDER BY count DESC
            LIMIT 1
            """
            pair_df = pd.read_sql_query(pair_query, conn, params=[days])
            stats['most_rejected_pair'] = pair_df['pair'].iloc[0] if not pair_df.empty else 'N/A'

            return stats

        except Exception as e:
            if "does not exist" not in str(e):
                logger.error(f"Error fetching SMC rejection stats: {e}")
            return {'total': 0, 'unique_pairs': 0, 'by_stage': {}, 'near_misses': 0, 'most_rejected_pair': 'N/A'}
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def get_smc_filter_options(_self) -> Dict[str, List[str]]:
        """
        Get unique filter options for SMC rejections (cached 1 min).

        Returns:
            Dictionary with stages, pairs, and sessions lists
        """
        conn = _self._get_connection()
        if not conn:
            return {'stages': ['All'], 'pairs': ['All'], 'sessions': ['All']}

        try:
            stages = ['All']
            pairs = ['All']
            sessions = ['All']

            # Check table exists
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'smc_simple_rejections'
                    )
                """)
                if not cursor.fetchone()[0]:
                    return {'stages': stages, 'pairs': pairs, 'sessions': sessions, 'table_exists': False}

            # Get unique stages
            stage_df = pd.read_sql_query(
                "SELECT DISTINCT rejection_stage FROM smc_simple_rejections ORDER BY rejection_stage",
                conn
            )
            stages.extend(stage_df['rejection_stage'].tolist())

            # Get unique pairs
            pair_df = pd.read_sql_query(
                "SELECT DISTINCT pair FROM smc_simple_rejections WHERE pair IS NOT NULL ORDER BY pair",
                conn
            )
            pairs.extend(pair_df['pair'].tolist())

            # Get unique sessions
            session_df = pd.read_sql_query(
                "SELECT DISTINCT market_session FROM smc_simple_rejections WHERE market_session IS NOT NULL ORDER BY market_session",
                conn
            )
            sessions.extend(session_df['market_session'].tolist())

            return {'stages': stages, 'pairs': pairs, 'sessions': sessions, 'table_exists': True}

        except Exception as e:
            logger.error(f"Error fetching SMC filter options: {e}")
            return {'stages': ['All'], 'pairs': ['All'], 'sessions': ['All'], 'table_exists': False}
        finally:
            conn.close()

    # =========================================================================
    # EMA DOUBLE REJECTIONS
    # =========================================================================

    @st.cache_data(ttl=300)
    def fetch_ema_double_rejections(
        _self, days: int, stage_filter: str, pair_filter: str, session_filter: str
    ) -> pd.DataFrame:
        """
        Fetch EMA Double Confirmation strategy rejection data from database (cached 5 min).

        Args:
            days: Number of days to look back
            stage_filter: Rejection stage or 'All'
            pair_filter: Currency pair or 'All'
            session_filter: Market session or 'All'

        Returns:
            DataFrame with rejection data
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                id,
                scan_timestamp,
                epic,
                pair,
                rejection_stage,
                rejection_reason,
                attempted_direction,
                current_price,
                market_hour,
                market_session,
                ema_fast_value,
                ema_slow_value,
                ema_trend_value,
                ema_fast_slow_separation_pips,
                htf_ema_value,
                htf_price_position,
                htf_distance_pips,
                successful_crossover_count,
                pending_crossover_count,
                adx_value,
                adx_trending,
                atr_15m,
                atr_pips,
                rsi_value,
                order_type,
                limit_offset_pips,
                potential_entry,
                potential_stop_loss,
                potential_take_profit,
                potential_risk_pips,
                potential_reward_pips,
                potential_rr_ratio,
                confidence_score,
                strategy_version
            FROM ema_double_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            """

            params = [days]

            if stage_filter != "All":
                query += " AND rejection_stage = %s"
                params.append(stage_filter)

            if pair_filter != "All":
                query += " AND (pair = %s OR epic LIKE %s)"
                params.append(pair_filter)
                params.append(f"%{pair_filter}%")

            if session_filter != "All":
                query += " AND market_session = %s"
                params.append(session_filter)

            query += " ORDER BY scan_timestamp DESC LIMIT 1000"

            df = pd.read_sql_query(query, conn, params=params)
            return df

        except Exception as e:
            if "does not exist" in str(e):
                logger.info("EMA Double Rejections table not yet created")
            else:
                logger.error(f"Error fetching EMA Double rejections: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_ema_double_rejection_stats(_self, days: int) -> Dict[str, Any]:
        """
        Fetch aggregated EMA Double rejection statistics (cached 5 min).

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with aggregated statistics
        """
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            # Total and by-stage counts
            query = """
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT epic) as unique_pairs,
                rejection_stage,
                COUNT(*) as stage_count
            FROM ema_double_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY rejection_stage
            ORDER BY stage_count DESC
            """

            df = pd.read_sql_query(query, conn, params=[days])

            if df.empty:
                return {'total': 0, 'unique_pairs': 0, 'by_stage': {}}

            stats = {
                'total': int(df['total'].iloc[0]) if not df.empty else 0,
                'unique_pairs': int(df['unique_pairs'].iloc[0]) if not df.empty else 0,
                'by_stage': dict(zip(df['rejection_stage'], df['stage_count']))
            }

            # Near-miss count (high confidence rejects)
            near_miss_query = """
            SELECT COUNT(*) as near_misses
            FROM ema_double_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            AND rejection_stage = 'CONFIDENCE'
            AND confidence_score >= 0.45
            """
            near_miss_df = pd.read_sql_query(near_miss_query, conn, params=[days])
            stats['near_misses'] = int(near_miss_df['near_misses'].iloc[0]) if not near_miss_df.empty else 0

            # Most rejected pair
            pair_query = """
            SELECT pair, COUNT(*) as count
            FROM ema_double_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY pair
            ORDER BY count DESC
            LIMIT 1
            """
            pair_df = pd.read_sql_query(pair_query, conn, params=[days])
            stats['most_rejected_pair'] = pair_df['pair'].iloc[0] if not pair_df.empty else 'N/A'

            # Average crossover count (EMA-specific)
            avg_query = """
            SELECT ROUND(AVG(successful_crossover_count)::numeric, 1) as avg_crossovers
            FROM ema_double_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            AND successful_crossover_count IS NOT NULL
            """
            avg_df = pd.read_sql_query(avg_query, conn, params=[days])
            stats['avg_crossover_count'] = float(avg_df['avg_crossovers'].iloc[0]) if not avg_df.empty and avg_df['avg_crossovers'].iloc[0] else 0

            return stats

        except Exception as e:
            if "does not exist" not in str(e):
                logger.error(f"Error fetching EMA Double rejection stats: {e}")
            return {'total': 0, 'unique_pairs': 0, 'by_stage': {}, 'near_misses': 0, 'most_rejected_pair': 'N/A', 'avg_crossover_count': 0}
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def get_ema_filter_options(_self) -> Dict[str, List[str]]:
        """
        Get unique filter options for EMA Double rejections (cached 1 min).

        Returns:
            Dictionary with stages, pairs, and sessions lists
        """
        conn = _self._get_connection()
        if not conn:
            return {'stages': ['All'], 'pairs': ['All'], 'sessions': ['All']}

        try:
            stages = ['All']
            pairs = ['All']
            sessions = ['All']

            # Check table exists
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'ema_double_rejections'
                    )
                """)
                if not cursor.fetchone()[0]:
                    return {'stages': stages, 'pairs': pairs, 'sessions': sessions, 'table_exists': False}

            # Get unique stages
            stage_df = pd.read_sql_query(
                "SELECT DISTINCT rejection_stage FROM ema_double_rejections ORDER BY rejection_stage",
                conn
            )
            stages.extend(stage_df['rejection_stage'].tolist())

            # Get unique pairs
            pair_df = pd.read_sql_query(
                "SELECT DISTINCT pair FROM ema_double_rejections WHERE pair IS NOT NULL ORDER BY pair",
                conn
            )
            pairs.extend(pair_df['pair'].tolist())

            # Get unique sessions
            session_df = pd.read_sql_query(
                "SELECT DISTINCT market_session FROM ema_double_rejections WHERE market_session IS NOT NULL ORDER BY market_session",
                conn
            )
            sessions.extend(session_df['market_session'].tolist())

            return {'stages': stages, 'pairs': pairs, 'sessions': sessions, 'table_exists': True}

        except Exception as e:
            logger.error(f"Error fetching EMA filter options: {e}")
            return {'stages': ['All'], 'pairs': ['All'], 'sessions': ['All'], 'table_exists': False}
        finally:
            conn.close()
