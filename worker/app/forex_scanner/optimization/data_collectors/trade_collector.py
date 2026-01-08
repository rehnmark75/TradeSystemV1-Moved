# optimization/data_collectors/trade_collector.py
"""
Trade data collector for unified parameter optimizer.
Collects executed trade outcomes from trade_log joined with alert_history.
"""

import logging
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional

from .base_collector import BaseCollector

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


class TradeCollector(BaseCollector):
    """Collects trade outcome data from trade_log and alert_history"""

    def __init__(self, db_manager: DatabaseManager = None):
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)

    def collect(self, days: int = 30, epics: List[str] = None) -> pd.DataFrame:
        """
        Collect trade outcomes with signal context.

        Args:
            days: Number of days of historical data
            epics: Optional list of epics to filter

        Returns:
            DataFrame with columns:
                - epic, pair, direction, pips_gained, is_winner
                - confidence_score, volume_ratio, market_regime, market_session
                - smart_money_score, entry_timestamp, exit_timestamp
        """
        self.logger.info(f"Collecting trade data for last {days} days...")

        # Build epic filter
        epic_filter = ""
        if epics:
            epic_list = "', '".join(epics)
            epic_filter = f"AND t.symbol IN ('{epic_list}')"

        query = f"""
        SELECT
            t.symbol as epic,
            CASE
                WHEN t.symbol LIKE '%%.EURUSD.%%' THEN 'EURUSD'
                WHEN t.symbol LIKE '%%.GBPUSD.%%' THEN 'GBPUSD'
                WHEN t.symbol LIKE '%%.USDJPY.%%' THEN 'USDJPY'
                WHEN t.symbol LIKE '%%.AUDUSD.%%' THEN 'AUDUSD'
                WHEN t.symbol LIKE '%%.USDCHF.%%' THEN 'USDCHF'
                WHEN t.symbol LIKE '%%.USDCAD.%%' THEN 'USDCAD'
                WHEN t.symbol LIKE '%%.NZDUSD.%%' THEN 'NZDUSD'
                WHEN t.symbol LIKE '%%.EURJPY.%%' THEN 'EURJPY'
                WHEN t.symbol LIKE '%%.GBPJPY.%%' THEN 'GBPJPY'
                WHEN t.symbol LIKE '%%.AUDJPY.%%' THEN 'AUDJPY'
                ELSE SPLIT_PART(t.symbol, '.', 3)
            END as pair,
            t.direction,
            t.pips_gained,
            CASE WHEN t.pips_gained > 0 THEN true ELSE false END as is_winner,
            t.entry_price,
            t.sl_price as stop_loss,
            t.limit_price as take_profit,
            t.timestamp as entry_timestamp,
            t.closed_at as exit_timestamp,
            t.profit_loss,
            -- Alert context (if available)
            ah.confidence_score,
            ah.volume_ratio,
            ah.smart_money_score,
            ah.market_regime,
            ah.market_session,
            ah.rsi as rsi_14,
            ah.atr as atr_pips,
            ah.macd_histogram
        FROM trade_log t
        LEFT JOIN alert_history ah ON t.alert_id = ah.id
        WHERE t.closed_at >= NOW() - INTERVAL '{days} days'
          AND t.status = 'closed'
          AND t.pips_gained IS NOT NULL
          {epic_filter}
        ORDER BY t.closed_at DESC
        """

        df = self._execute_query(query)

        if df.empty:
            self.logger.warning("No trade data found")
            return df

        # Normalize direction
        df['direction'] = df['direction'].apply(self._normalize_direction)

        # Add computed columns
        df['profit_factor_contribution'] = df.apply(
            lambda row: row['pips_gained'] if row['is_winner'] else 0, axis=1
        )
        df['loss_contribution'] = df.apply(
            lambda row: abs(row['pips_gained']) if not row['is_winner'] else 0, axis=1
        )

        self.logger.info(f"Collected {len(df)} trades")
        return df

    def get_summary_by_epic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics grouped by epic.

        Returns DataFrame with:
            - epic, total_trades, winners, losers, win_rate
            - total_pips, avg_pips, profit_factor
        """
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby('epic').agg(
            total_trades=('epic', 'count'),
            winners=('is_winner', 'sum'),
            total_pips=('pips_gained', 'sum'),
            avg_pips=('pips_gained', 'mean'),
            gross_profit=('profit_factor_contribution', 'sum'),
            gross_loss=('loss_contribution', 'sum')
        ).reset_index()

        summary['losers'] = summary['total_trades'] - summary['winners']
        summary['win_rate'] = summary['winners'] / summary['total_trades']
        summary['profit_factor'] = summary.apply(
            lambda row: row['gross_profit'] / row['gross_loss']
            if row['gross_loss'] > 0 else float('inf'), axis=1
        )

        return summary[['epic', 'total_trades', 'winners', 'losers',
                       'win_rate', 'total_pips', 'avg_pips', 'profit_factor']]

    def get_summary_by_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics grouped by epic and direction.

        Returns DataFrame with direction-specific win rates and pips.
        """
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby(['epic', 'direction']).agg(
            trades=('epic', 'count'),
            winners=('is_winner', 'sum'),
            total_pips=('pips_gained', 'sum'),
            avg_pips=('pips_gained', 'mean')
        ).reset_index()

        summary['win_rate'] = summary['winners'] / summary['trades']

        return summary
