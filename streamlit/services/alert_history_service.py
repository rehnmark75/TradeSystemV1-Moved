"""
Alert History Service

Provides data access layer for alert history analytics including:
- Alert history with Claude analysis data
- Filter options for strategies and pairs
- Alert statistics

All fetch methods are cached with @st.cache_data for performance.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import logging

from services.db_utils import get_psycopg2_connection

logger = logging.getLogger(__name__)


class AlertHistoryService:
    """Service for fetching alert history data from the database."""

    def __init__(self):
        self.db_type = "trading"

    def _get_connection(self):
        """Get a database connection from the pool."""
        return get_psycopg2_connection(self.db_type)

    @st.cache_data(ttl=180)
    def fetch_alert_history(
        _self, days: int, status_filter: str, strategy_filter: str, pair_filter: str
    ) -> pd.DataFrame:
        """
        Fetch alert history with Claude analysis data from database (cached 3 min).

        Args:
            days: Number of days to look back
            status_filter: 'All', 'Approved', or 'Rejected'
            strategy_filter: Strategy name or 'All'
            pair_filter: Currency pair or 'All'

        Returns:
            DataFrame with alert history data
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            # Build base query
            query = """
            SELECT
                id,
                alert_timestamp,
                epic,
                pair,
                signal_type,
                strategy,
                price,
                market_session,
                claude_score,
                claude_decision,
                claude_approved,
                claude_reason,
                claude_mode,
                claude_raw_response,
                status,
                alert_level
            FROM alert_history
            WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
            """

            params = [days]

            # Add status filter
            if status_filter == "Approved":
                query += " AND (claude_approved = TRUE OR claude_decision = 'APPROVE')"
            elif status_filter == "Rejected":
                query += " AND (claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED')"

            # Add strategy filter
            if strategy_filter != "All":
                query += " AND strategy = %s"
                params.append(strategy_filter)

            # Add pair filter
            if pair_filter != "All":
                query += " AND (pair = %s OR epic LIKE %s)"
                params.append(pair_filter)
                params.append(f"%{pair_filter}%")

            query += " ORDER BY alert_timestamp DESC LIMIT 500"

            df = pd.read_sql_query(query, conn, params=params)
            return df

        except Exception as e:
            logger.error(f"Error fetching alert history: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=60)
    def get_filter_options(_self) -> Dict[str, List[str]]:
        """
        Get unique filter options for alert history (cached 1 min).

        Returns:
            Dictionary with 'strategies' and 'pairs' lists
        """
        conn = _self._get_connection()
        if not conn:
            return {'strategies': ['All'], 'pairs': ['All']}

        try:
            strategies = ['All']
            pairs = ['All']

            # Get unique strategies
            strat_df = pd.read_sql_query(
                "SELECT DISTINCT strategy FROM alert_history WHERE strategy IS NOT NULL ORDER BY strategy",
                conn
            )
            strategies.extend(strat_df['strategy'].tolist())

            # Get unique pairs
            pair_df = pd.read_sql_query(
                "SELECT DISTINCT pair FROM alert_history WHERE pair IS NOT NULL ORDER BY pair",
                conn
            )
            pairs.extend(pair_df['pair'].tolist())

            return {'strategies': strategies, 'pairs': pairs}

        except Exception as e:
            logger.error(f"Error fetching alert filter options: {e}")
            return {'strategies': ['All'], 'pairs': ['All']}
        finally:
            conn.close()

    @st.cache_data(ttl=180)
    def fetch_alert_stats(_self, days: int) -> Dict[str, Any]:
        """
        Fetch aggregated alert statistics (cached 3 min).

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with aggregated statistics
        """
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            query = """
            SELECT
                COUNT(*) as total_alerts,
                COUNT(CASE WHEN claude_approved = TRUE OR claude_decision = 'APPROVE' THEN 1 END) as approved,
                COUNT(CASE WHEN claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED' THEN 1 END) as rejected,
                ROUND(AVG(claude_score)::numeric, 2) as avg_score,
                COUNT(DISTINCT strategy) as unique_strategies,
                COUNT(DISTINCT pair) as unique_pairs
            FROM alert_history
            WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
            """

            df = pd.read_sql_query(query, conn, params=[days])

            if df.empty:
                return {
                    'total_alerts': 0,
                    'approved': 0,
                    'rejected': 0,
                    'avg_score': 0,
                    'approval_rate': 0,
                    'unique_strategies': 0,
                    'unique_pairs': 0
                }

            row = df.iloc[0]
            total = row['total_alerts'] or 0
            approved = row['approved'] or 0

            return {
                'total_alerts': int(total),
                'approved': int(approved),
                'rejected': int(row['rejected'] or 0),
                'avg_score': float(row['avg_score'] or 0),
                'approval_rate': (approved / total * 100) if total > 0 else 0,
                'unique_strategies': int(row['unique_strategies'] or 0),
                'unique_pairs': int(row['unique_pairs'] or 0)
            }

        except Exception as e:
            logger.error(f"Error fetching alert stats: {e}")
            return {}
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_strategy_approval_rates(_self, days: int) -> pd.DataFrame:
        """
        Fetch approval rates by strategy (cached 5 min).

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with strategy, total, approved, rejected, approval_rate
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                strategy,
                COUNT(*) as total,
                COUNT(CASE WHEN claude_approved = TRUE OR claude_decision = 'APPROVE' THEN 1 END) as approved,
                COUNT(CASE WHEN claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED' THEN 1 END) as rejected,
                ROUND(AVG(claude_score)::numeric, 2) as avg_score
            FROM alert_history
            WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
            AND strategy IS NOT NULL
            GROUP BY strategy
            ORDER BY total DESC
            """

            df = pd.read_sql_query(query, conn, params=[days])

            if not df.empty:
                df['approval_rate'] = (df['approved'] / df['total'] * 100).round(1)

            return df

        except Exception as e:
            logger.error(f"Error fetching strategy approval rates: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_hourly_alert_distribution(_self, days: int) -> pd.DataFrame:
        """
        Fetch alert distribution by hour (cached 5 min).

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with hour, total, approved, rejected
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                EXTRACT(HOUR FROM alert_timestamp)::int as hour,
                COUNT(*) as total,
                COUNT(CASE WHEN claude_approved = TRUE OR claude_decision = 'APPROVE' THEN 1 END) as approved,
                COUNT(CASE WHEN claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED' THEN 1 END) as rejected
            FROM alert_history
            WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY EXTRACT(HOUR FROM alert_timestamp)
            ORDER BY hour
            """

            return pd.read_sql_query(query, conn, params=[days])

        except Exception as e:
            logger.error(f"Error fetching hourly distribution: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
