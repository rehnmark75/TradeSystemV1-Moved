"""
Unfilled Orders Service

Provides data access layer for unfilled order analysis including:
- Summary statistics
- Detailed unfilled order analysis
- Per-epic breakdown

All fetch methods are cached with @st.cache_data for performance.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
import logging

from services.db_utils import get_psycopg2_connection

logger = logging.getLogger(__name__)


class UnfilledOrdersService:
    """Service for fetching unfilled order analysis data from the database."""

    def __init__(self):
        self.db_type = "trading"

    def _get_connection(self):
        """Get a database connection from the pool."""
        return get_psycopg2_connection(self.db_type)

    @st.cache_data(ttl=300)
    def fetch_summary(_self) -> pd.DataFrame:
        """
        Fetch unfilled order summary data (cached 5 min).

        Returns:
            DataFrame with summary statistics
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = "SELECT * FROM v_unfilled_order_summary"
            return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching unfilled order summary: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_detailed_analysis(_self) -> pd.DataFrame:
        """
        Fetch detailed unfilled order analysis (cached 5 min).

        Returns:
            DataFrame with detailed order analysis
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                id, symbol, direction, order_time, expiry_time,
                entry_level, stop_loss, take_profit, price_at_expiry,
                gap_to_entry_pips, would_fill_4h, outcome_4h,
                would_fill_24h, outcome_24h, signal_quality,
                max_favorable_pips, max_adverse_pips, alert_id
            FROM v_unfilled_order_analysis
            WHERE symbol NOT LIKE '%CEEM%'
            ORDER BY order_time DESC
            """
            return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching unfilled order details: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_epic_breakdown(_self) -> pd.DataFrame:
        """
        Fetch per-epic unfilled order breakdown (cached 5 min).

        Returns:
            DataFrame with per-epic statistics
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                symbol,
                COUNT(*) as total_unfilled,
                SUM(CASE WHEN signal_quality = 'GOOD_SIGNAL' THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN signal_quality = 'BAD_SIGNAL' THEN 1 ELSE 0 END) as bad,
                SUM(CASE WHEN signal_quality = 'INCONCLUSIVE' THEN 1 ELSE 0 END) as inconclusive,
                ROUND(AVG(gap_to_entry_pips)::numeric, 1) as avg_gap_pips,
                ROUND(AVG(max_favorable_pips)::numeric, 1) as avg_favorable,
                ROUND(AVG(max_adverse_pips)::numeric, 1) as avg_adverse
            FROM v_unfilled_order_analysis
            WHERE symbol NOT LIKE '%CEEM%'
            GROUP BY symbol
            ORDER BY total_unfilled DESC
            """
            return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching epic breakdown: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def check_view_exists(_self) -> bool:
        """Check if the required view exists in database."""
        conn = _self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.views
                        WHERE table_name = 'v_unfilled_order_analysis'
                    )
                """)
                return cursor.fetchone()[0]
        except Exception:
            return False
        finally:
            conn.close()
