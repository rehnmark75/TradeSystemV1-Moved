"""
Scan Performance Snapshot Analytics Service

Provides data access layer for scan performance snapshot analytics including:
- Per-epic scan snapshots with indicator data
- Signal generation tracking
- Rejection analysis
- Market regime and session analysis

All fetch methods are cached with @st.cache_data for performance.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

from services.db_utils import get_psycopg2_connection

logger = logging.getLogger(__name__)


class PerformanceSnapshotService:
    """Service for fetching scan performance snapshot data from the database."""

    def __init__(self):
        self.db_type = "trading"

    def _get_connection(self):
        """Get a database connection from the pool."""
        return get_psycopg2_connection(self.db_type)

    @st.cache_data(ttl=300)
    def fetch_scan_snapshots(
        _self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch scan performance snapshots (cached 5 min).

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with scan snapshots
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                id,
                scan_cycle_id,
                scan_timestamp,
                epic,
                pair_name,
                signal_generated,
                signal_type,
                signal_id,
                rejection_reason,
                rejection_details,
                raw_confidence,
                final_confidence,
                confidence_threshold,
                current_price,
                spread_pips,
                ema_9,
                ema_21,
                ema_50,
                ema_200,
                ema_bias_4h,
                macd_line,
                macd_signal,
                macd_histogram,
                macd_trend,
                rsi_14,
                rsi_zone,
                efficiency_ratio,
                er_classification,
                atr_14,
                atr_pips,
                atr_percentile,
                volatility_state,
                bb_upper,
                bb_middle,
                bb_lower,
                bb_width,
                bb_width_percentile,
                bb_position,
                adx,
                plus_di,
                minus_di,
                adx_trend_strength,
                market_regime,
                regime_confidence,
                session,
                session_volatility,
                near_order_block,
                ob_type,
                ob_distance_pips,
                near_fvg,
                fvg_type,
                fvg_distance_pips,
                liquidity_sweep_detected,
                liquidity_sweep_type,
                smart_money_score,
                smart_money_validated,
                mtf_alignment,
                mtf_confluence_score,
                entry_quality_score,
                fib_zone_distance,
                created_at
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
            ORDER BY scan_timestamp DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            if not df.empty:
                df['scan_timestamp'] = pd.to_datetime(df['scan_timestamp'])
                # Convert decimal columns to float for plotting
                numeric_cols = [
                    'raw_confidence', 'final_confidence', 'confidence_threshold',
                    'current_price', 'spread_pips', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
                    'macd_line', 'macd_signal', 'macd_histogram', 'rsi_14',
                    'efficiency_ratio', 'atr_14', 'atr_pips', 'atr_percentile',
                    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_width_percentile',
                    'adx', 'plus_di', 'minus_di', 'regime_confidence',
                    'ob_distance_pips', 'fvg_distance_pips', 'smart_money_score',
                    'mtf_confluence_score', 'entry_quality_score', 'fib_zone_distance'
                ]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            logger.error(f"Error fetching scan snapshots: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_signals_only(
        _self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch only scan snapshots where signals were generated (cached 5 min).

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with signal snapshots
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT *
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
              AND signal_generated = true
            ORDER BY scan_timestamp DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            if not df.empty:
                df['scan_timestamp'] = pd.to_datetime(df['scan_timestamp'])

            return df

        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def fetch_rejections(
        _self, start_date: datetime, end_date: datetime, rejection_reason: str = None
    ) -> pd.DataFrame:
        """
        Fetch rejected scan snapshots (cached 5 min).

        Args:
            start_date: Start date for query
            end_date: End date for query
            rejection_reason: Optional filter by rejection reason

        Returns:
            DataFrame with rejection snapshots
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            reason_filter = ""
            params = [start_date, end_date + timedelta(days=1)]

            if rejection_reason:
                reason_filter = "AND rejection_reason = %s"
                params.append(rejection_reason)

            query = f"""
            SELECT *
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
              AND signal_generated = false
              AND rejection_reason IS NOT NULL
              {reason_filter}
            ORDER BY scan_timestamp DESC
            """

            df = pd.read_sql_query(query, conn, params=params)

            if not df.empty:
                df['scan_timestamp'] = pd.to_datetime(df['scan_timestamp'])

            return df

        except Exception as e:
            logger.error(f"Error fetching rejections: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_scan_summary(_self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Get summary statistics for scan performance.

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            Dict with summary statistics
        """
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            query = """
            SELECT
                COUNT(*) as total_scans,
                COUNT(DISTINCT scan_cycle_id) as scan_cycles,
                COUNT(DISTINCT epic) as unique_epics,
                SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals_generated,
                SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
                SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
                AVG(raw_confidence) as avg_raw_confidence,
                AVG(final_confidence) as avg_final_confidence,
                AVG(CASE WHEN signal_generated THEN final_confidence END) as avg_signal_confidence,
                COUNT(DISTINCT rejection_reason) as rejection_types,
                SUM(CASE WHEN rejection_reason = 'confidence' THEN 1 ELSE 0 END) as confidence_rejections,
                SUM(CASE WHEN rejection_reason = 'dedup' THEN 1 ELSE 0 END) as dedup_rejections,
                MIN(scan_timestamp) as first_scan,
                MAX(scan_timestamp) as last_scan
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            if not df.empty:
                row = df.iloc[0]
                return {
                    'total_scans': int(row['total_scans'] or 0),
                    'scan_cycles': int(row['scan_cycles'] or 0),
                    'unique_epics': int(row['unique_epics'] or 0),
                    'signals_generated': int(row['signals_generated'] or 0),
                    'buy_signals': int(row['buy_signals'] or 0),
                    'sell_signals': int(row['sell_signals'] or 0),
                    'avg_raw_confidence': float(row['avg_raw_confidence'] or 0),
                    'avg_final_confidence': float(row['avg_final_confidence'] or 0),
                    'avg_signal_confidence': float(row['avg_signal_confidence'] or 0),
                    'rejection_types': int(row['rejection_types'] or 0),
                    'confidence_rejections': int(row['confidence_rejections'] or 0),
                    'dedup_rejections': int(row['dedup_rejections'] or 0),
                    'first_scan': row['first_scan'],
                    'last_scan': row['last_scan'],
                    'signal_rate': float(row['signals_generated'] or 0) / float(row['total_scans']) if row['total_scans'] else 0
                }
            return {}

        except Exception as e:
            logger.error(f"Error fetching scan summary: {e}")
            return {}
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_epic_summary(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get per-epic summary statistics.

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with per-epic stats
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                epic,
                pair_name,
                COUNT(*) as total_scans,
                SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
                SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
                SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
                AVG(raw_confidence) as avg_raw_confidence,
                AVG(final_confidence) as avg_final_confidence,
                AVG(rsi_14) as avg_rsi,
                AVG(adx) as avg_adx,
                AVG(atr_pips) as avg_atr_pips,
                AVG(spread_pips) as avg_spread,
                MODE() WITHIN GROUP (ORDER BY market_regime) as dominant_regime,
                MODE() WITHIN GROUP (ORDER BY volatility_state) as dominant_volatility,
                SUM(CASE WHEN rejection_reason = 'confidence' THEN 1 ELSE 0 END) as confidence_rejections,
                SUM(CASE WHEN rejection_reason = 'dedup' THEN 1 ELSE 0 END) as dedup_rejections
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
            GROUP BY epic, pair_name
            ORDER BY signals DESC, total_scans DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            if not df.empty:
                # Calculate signal rate
                df['signal_rate'] = df['signals'] / df['total_scans']

                # Convert numeric columns
                numeric_cols = ['avg_raw_confidence', 'avg_final_confidence', 'avg_rsi',
                               'avg_adx', 'avg_atr_pips', 'avg_spread', 'signal_rate']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            logger.error(f"Error fetching epic summary: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_regime_distribution(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get market regime distribution.

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with regime distribution
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                market_regime,
                COUNT(*) as count,
                AVG(regime_confidence) as avg_confidence,
                SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
                AVG(CASE WHEN signal_generated THEN final_confidence END) as avg_signal_confidence
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
              AND market_regime IS NOT NULL
            GROUP BY market_regime
            ORDER BY count DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching regime distribution: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_session_distribution(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get trading session distribution.

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with session distribution
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                session,
                session_volatility,
                COUNT(*) as count,
                SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
                AVG(raw_confidence) as avg_confidence,
                AVG(atr_pips) as avg_atr_pips,
                AVG(spread_pips) as avg_spread
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
              AND session IS NOT NULL
            GROUP BY session, session_volatility
            ORDER BY count DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching session distribution: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_rejection_analysis(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get rejection reason analysis.

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with rejection analysis
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                rejection_reason,
                COUNT(*) as count,
                AVG(raw_confidence) as avg_raw_confidence,
                AVG(final_confidence) as avg_final_confidence,
                AVG(confidence_threshold) as avg_threshold,
                COUNT(DISTINCT epic) as affected_epics
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
              AND rejection_reason IS NOT NULL
            GROUP BY rejection_reason
            ORDER BY count DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching rejection analysis: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_indicator_stats(_self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Get indicator statistics for signals vs non-signals.

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            Dict with indicator comparisons
        """
        conn = _self._get_connection()
        if not conn:
            return {}

        try:
            query = """
            SELECT
                signal_generated,
                AVG(rsi_14) as avg_rsi,
                AVG(adx) as avg_adx,
                AVG(efficiency_ratio) as avg_er,
                AVG(atr_pips) as avg_atr,
                AVG(bb_width_percentile) as avg_bb_percentile,
                AVG(smart_money_score) as avg_smc_score,
                AVG(mtf_confluence_score) as avg_mtf_score,
                AVG(entry_quality_score) as avg_entry_quality,
                COUNT(*) as count
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
            GROUP BY signal_generated
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            result = {}
            for _, row in df.iterrows():
                key = 'signals' if row['signal_generated'] else 'non_signals'
                result[key] = {
                    'avg_rsi': float(row['avg_rsi'] or 0),
                    'avg_adx': float(row['avg_adx'] or 0),
                    'avg_er': float(row['avg_er'] or 0),
                    'avg_atr': float(row['avg_atr'] or 0),
                    'avg_bb_percentile': float(row['avg_bb_percentile'] or 0),
                    'avg_smc_score': float(row['avg_smc_score'] or 0),
                    'avg_mtf_score': float(row['avg_mtf_score'] or 0),
                    'avg_entry_quality': float(row['avg_entry_quality'] or 0),
                    'count': int(row['count'])
                }

            return result

        except Exception as e:
            logger.error(f"Error fetching indicator stats: {e}")
            return {}
        finally:
            conn.close()

    @st.cache_data(ttl=300)
    def get_scan_timeline(_self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get scan timeline aggregated by hour.

        Args:
            start_date: Start date for query
            end_date: End date for query

        Returns:
            DataFrame with hourly scan stats
        """
        conn = _self._get_connection()
        if not conn:
            return pd.DataFrame()

        try:
            query = """
            SELECT
                DATE_TRUNC('hour', scan_timestamp) as hour,
                COUNT(*) as total_scans,
                COUNT(DISTINCT scan_cycle_id) as scan_cycles,
                SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
                AVG(raw_confidence) as avg_confidence,
                AVG(atr_pips) as avg_atr
            FROM scan_performance_snapshot
            WHERE scan_timestamp >= %s
              AND scan_timestamp <= %s
            GROUP BY DATE_TRUNC('hour', scan_timestamp)
            ORDER BY hour
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date + timedelta(days=1)]
            )

            if not df.empty:
                df['hour'] = pd.to_datetime(df['hour'])

            return df

        except Exception as e:
            logger.error(f"Error fetching scan timeline: {e}")
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
