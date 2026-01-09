"""
Backtest History Service

Provides data access for backtest executions and results.
Used by the backtest results tab component.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os

try:
    from sqlalchemy import create_engine, text
except ImportError:
    create_engine = None
    text = None


class BacktestService:
    """
    Service for fetching and managing backtest data.

    Features:
    - List backtest executions with filtering
    - Fetch signals for specific execution
    - Get performance metrics
    - Chart URL retrieval
    """

    def __init__(self):
        """Initialize service with database connection."""
        self.database_url = os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/forex'
        )
        self._engine = None

    @property
    def engine(self):
        """Lazy-load database engine."""
        if self._engine is None and create_engine:
            self._engine = create_engine(self.database_url)
        return self._engine

    @st.cache_data(ttl=60)
    def fetch_backtest_executions(_self, days: int = 30, strategy: str = "All") -> pd.DataFrame:
        """
        Fetch backtest executions with optional filters.

        Args:
            days: Number of days to look back
            strategy: Strategy filter ("All" for all strategies)

        Returns:
            DataFrame with backtest executions
        """
        if not _self.engine:
            return pd.DataFrame()

        start_date = datetime.now() - timedelta(days=days)

        query = """
        SELECT
            be.id,
            be.execution_name,
            be.strategy_name,
            be.start_time,
            be.end_time,
            be.data_start_date,
            be.data_end_date,
            be.epics_tested,
            be.status,
            be.total_candles_processed,
            be.execution_duration_seconds,
            be.chart_url,
            be.chart_object_name,
            COALESCE(bs.signal_count, 0) as signal_count,
            COALESCE(bs.win_count, 0) as win_count,
            COALESCE(bs.loss_count, 0) as loss_count,
            COALESCE(bs.total_pips, 0) as total_pips
        FROM backtest_executions be
        LEFT JOIN (
            SELECT
                execution_id,
                COUNT(*) as signal_count,
                SUM(CASE WHEN pips_gained > 0 THEN 1 ELSE 0 END) as win_count,
                SUM(CASE WHEN pips_gained <= 0 THEN 1 ELSE 0 END) as loss_count,
                COALESCE(SUM(pips_gained), 0) as total_pips
            FROM backtest_signals
            GROUP BY execution_id
        ) bs ON be.id = bs.execution_id
        WHERE be.start_time >= :start_date
        """

        params = {'start_date': start_date}

        if strategy != "All":
            query += " AND be.strategy_name = :strategy"
            params['strategy'] = strategy

        query += " ORDER BY be.start_time DESC"

        try:
            with _self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            st.error(f"Failed to fetch backtests: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=60)
    def fetch_backtest_signals(_self, execution_id: int) -> pd.DataFrame:
        """
        Fetch signals for a specific backtest execution.

        Args:
            execution_id: Backtest execution ID

        Returns:
            DataFrame with signals
        """
        if not _self.engine:
            return pd.DataFrame()

        query = """
        SELECT
            id,
            signal_timestamp,
            epic,
            signal_type,
            strategy_name,
            entry_price,
            exit_price,
            stop_loss_price,
            take_profit_price,
            pips_gained,
            trade_result,
            confidence_score,
            market_intelligence
        FROM backtest_signals
        WHERE execution_id = :exec_id
        ORDER BY signal_timestamp
        """

        try:
            with _self.engine.connect() as conn:
                result = conn.execute(text(query), {'exec_id': execution_id})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            st.error(f"Failed to fetch signals: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_filter_options(_self) -> Dict[str, List[str]]:
        """
        Get available filter options from database.

        Returns:
            Dict with 'strategies' list
        """
        if not _self.engine:
            return {'strategies': ['All']}

        try:
            with _self.engine.connect() as conn:
                # Get unique strategies
                result = conn.execute(text("""
                    SELECT DISTINCT strategy_name
                    FROM backtest_executions
                    WHERE strategy_name IS NOT NULL
                    ORDER BY strategy_name
                """))
                strategies = ['All'] + [row[0] for row in result.fetchall()]

                return {'strategies': strategies}
        except Exception as e:
            return {'strategies': ['All']}

    def get_execution_details(_self, execution_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific execution.

        Args:
            execution_id: Backtest execution ID

        Returns:
            Dict with execution details or None
        """
        if not _self.engine:
            return None

        query = """
        SELECT
            id,
            execution_name,
            strategy_name,
            start_time,
            end_time,
            data_start_date,
            data_end_date,
            epics_tested,
            timeframes,
            status,
            total_candles_processed,
            execution_duration_seconds,
            chart_url,
            chart_object_name,
            config_snapshot
        FROM backtest_executions
        WHERE id = :exec_id
        """

        try:
            with _self.engine.connect() as conn:
                result = conn.execute(text(query), {'exec_id': execution_id})
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return None
        except Exception as e:
            return None

    def calculate_performance_metrics(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics from signals.

        Args:
            signals_df: DataFrame with signals

        Returns:
            Dict with performance metrics
        """
        if signals_df.empty:
            return {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pips': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0
            }

        total = len(signals_df)
        wins = len(signals_df[signals_df['pips_gained'] > 0])
        losses = len(signals_df[signals_df['pips_gained'] <= 0])
        total_pips = signals_df['pips_gained'].sum()

        win_pips = signals_df[signals_df['pips_gained'] > 0]['pips_gained'].sum()
        loss_pips = abs(signals_df[signals_df['pips_gained'] <= 0]['pips_gained'].sum())

        avg_win = signals_df[signals_df['pips_gained'] > 0]['pips_gained'].mean() if wins > 0 else 0
        avg_loss = abs(signals_df[signals_df['pips_gained'] <= 0]['pips_gained'].mean()) if losses > 0 else 0

        win_rate = (wins / total * 100) if total > 0 else 0
        profit_factor = (win_pips / loss_pips) if loss_pips > 0 else float('inf') if win_pips > 0 else 0
        expectancy = total_pips / total if total > 0 else 0

        return {
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
