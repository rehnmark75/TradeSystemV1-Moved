"""
Market Intelligence Analytics Service

Provides data access layer for market intelligence analytics including:
- Signal-based market intelligence from alert_history
- Comprehensive market intelligence from market_intelligence_history
- Cached queries for performance

All fetch methods are cached with @st.cache_data for performance.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

from services.db_utils import get_psycopg2_connection

logger = logging.getLogger(__name__)


class MarketIntelligenceAnalyticsService:
    """Service for fetching market intelligence data from the database."""

    def __init__(self):
        self.db_type = "trading"

    def _get_connection(self):
        """Get a database connection from the pool."""
        return get_psycopg2_connection(self.db_type)

    @st.cache_data(ttl=3600)  # Cache schema for 1 hour - columns rarely change
    def _get_market_intelligence_columns(_self) -> list:
        """
        Get column names for market_intelligence_history table (cached 1 hour).

        This avoids querying information_schema on every request.
        """
        conn = _self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'market_intelligence_history'
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching column names: {e}")
            return []
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_signal_based_intelligence(
        _self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch market intelligence data from alert_history (cached 5 min).

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with signal-based market intelligence
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                a.id,
                a.alert_timestamp,
                a.epic,
                a.strategy,
                a.signal_type,
                a.confidence_score,
                a.strategy_metadata,
                (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'dominant_regime') as regime,
                (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'confidence')::float as regime_confidence,
                (a.strategy_metadata::json->'market_intelligence'->'session_analysis'->>'current_session') as session,
                (a.strategy_metadata::json->'market_intelligence'->>'volatility_level') as volatility_level,
                (a.strategy_metadata::json->'market_intelligence'->>'intelligence_source') as intelligence_source
            FROM alert_history a
            WHERE a.alert_timestamp >= %s
              AND a.alert_timestamp <= %s
              AND a.strategy_metadata IS NOT NULL
              AND (a.strategy_metadata::json->'market_intelligence') IS NOT NULL
            ORDER BY a.alert_timestamp DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            # Clean and convert data types
            if not df.empty:
                df = df.where(pd.notnull(df), None)

            return df

        except Exception as e:
            logger.error(f"Error fetching signal-based intelligence: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_comprehensive_intelligence(
        _self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch comprehensive market intelligence from market_intelligence_history (cached 5 min).

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with comprehensive market intelligence
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            # Use cached column list instead of querying information_schema every time
            existing_columns = _self._get_market_intelligence_columns()

            if not existing_columns:
                return pd.DataFrame()

            # Base columns that should always exist
            base_columns = [
                "mih.id",
                "mih.scan_timestamp",
                "mih.scan_cycle_id",
                "mih.epic_list",
                "mih.epic_count",
                "mih.dominant_regime as regime",
                "mih.regime_confidence",
                "mih.current_session as session",
                "mih.session_volatility as volatility_level",
                "mih.market_bias",
                "mih.average_trend_strength",
                "mih.average_volatility",
                "mih.risk_sentiment",
                "mih.recommended_strategy",
                "mih.confidence_threshold",
                "mih.intelligence_source",
                "mih.regime_trending_score",
                "mih.regime_ranging_score",
                "mih.regime_breakout_score",
                "mih.regime_reversal_score",
                "mih.regime_high_vol_score",
                "mih.regime_low_vol_score"
            ]

            # Add new columns only if they exist
            optional_columns = []
            if 'individual_epic_regimes' in existing_columns:
                optional_columns.append("mih.individual_epic_regimes")
            if 'pair_analyses' in existing_columns:
                optional_columns.append("mih.pair_analyses")

            # Combine all columns
            all_columns = base_columns + optional_columns

            query = f"""
            SELECT
                {', '.join(all_columns)}
            FROM market_intelligence_history mih
            WHERE mih.scan_timestamp >= %s
              AND mih.scan_timestamp <= %s
            ORDER BY mih.scan_timestamp DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            # Clean and convert data types
            if not df.empty:
                df = df.where(pd.notnull(df), None)
                if 'scan_timestamp' in df.columns:
                    df['scan_timestamp'] = pd.to_datetime(df['scan_timestamp'])

            return df

        except Exception as e:
            logger.error(f"Error fetching comprehensive intelligence: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def check_table_exists(_self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        conn = _self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    )
                """, (table_name,))
                return cursor.fetchone()[0]
        except Exception:
            return False
        finally:
            conn.close()
