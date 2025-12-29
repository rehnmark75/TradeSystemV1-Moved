"""
Trade Analysis Service

Provides data access layer for individual trade analysis including:
- Filled trades for analysis
- Trailing stop analysis (via FastAPI)
- Signal analysis (via FastAPI)
- Outcome analysis (via FastAPI)

All fetch methods are cached with @st.cache_data for performance.
"""

import streamlit as st
import pandas as pd
import requests
import logging
from typing import Optional, Dict, Any

from services.db_utils import get_psycopg2_connection

logger = logging.getLogger(__name__)

# FastAPI connection settings
FASTAPI_BASE_URL = "http://fastapi-dev:8000"
FASTAPI_HEADERS = {
    "X-APIM-Gateway": "verified",
    "X-API-KEY": "436abe054a074894a0517e5172f0e5b6"
}


class TradeAnalysisService:
    """Service for fetching and analyzing individual trades."""

    def __init__(self):
        self.db_type = "trading"

    def _get_connection(self):
        """Get a database connection from the pool."""
        return get_psycopg2_connection(self.db_type)

    @st.cache_data(ttl=180)
    def fetch_filled_trades_for_analysis(_self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch only filled trades (closed or tracking) for analysis (cached 3 min).

        Args:
            limit: Maximum number of trades to fetch

        Returns:
            DataFrame with filled trades data
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

    def get_trailing_stop_analysis(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch trailing stop analysis for a trade from FastAPI.

        Args:
            trade_id: Trade ID to analyze

        Returns:
            Dict with analysis data or None if failed
        """
        try:
            response = requests.get(
                f"{FASTAPI_BASE_URL}/api/trade-analysis/trade/{trade_id}",
                headers=FASTAPI_HEADERS,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"error": "not_found", "message": f"Trade {trade_id} not found"}
            else:
                return {"error": "api_error", "message": f"API Error: {response.status_code} - {response.text}"}

        except requests.Timeout:
            return {"error": "timeout", "message": "Request timed out"}
        except Exception as e:
            return {"error": "exception", "message": str(e)}

    def get_signal_analysis(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch signal analysis for a trade from FastAPI.

        Args:
            trade_id: Trade ID to analyze

        Returns:
            Dict with signal analysis data or None if failed
        """
        try:
            response = requests.get(
                f"{FASTAPI_BASE_URL}/api/trade-analysis/signal/{trade_id}",
                headers=FASTAPI_HEADERS,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"error": "not_found", "message": f"Trade {trade_id} not found"}
            else:
                return {"error": "api_error", "message": f"API Error: {response.status_code} - {response.text}"}

        except requests.Timeout:
            return {"error": "timeout", "message": "Request timed out"}
        except Exception as e:
            return {"error": "exception", "message": str(e)}

    def get_outcome_analysis(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch outcome analysis for a trade from FastAPI.

        Args:
            trade_id: Trade ID to analyze

        Returns:
            Dict with outcome analysis data or None if failed
        """
        try:
            response = requests.get(
                f"{FASTAPI_BASE_URL}/api/trade-analysis/outcome/{trade_id}",
                headers=FASTAPI_HEADERS,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"error": "not_found", "message": f"Trade {trade_id} not found"}
            else:
                return {"error": "api_error", "message": f"API Error: {response.status_code} - {response.text}"}

        except requests.Timeout:
            return {"error": "timeout", "message": "Request timed out"}
        except Exception as e:
            return {"error": "exception", "message": str(e)}
