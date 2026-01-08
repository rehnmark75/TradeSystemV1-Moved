# optimization/data_collectors/market_intel_collector.py
"""
Market intelligence data collector for unified parameter optimizer.
Collects market regime and session data from market_intelligence_history.
"""

import logging
import pandas as pd
from typing import List

from .base_collector import BaseCollector

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


class MarketIntelCollector(BaseCollector):
    """Collects market intelligence data for regime correlation analysis"""

    def __init__(self, db_manager: DatabaseManager = None):
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)

    def collect(self, days: int = 30, epics: List[str] = None) -> pd.DataFrame:
        """
        Collect market intelligence snapshots.

        Args:
            days: Number of days of historical data
            epics: Not used (market intel is global)

        Returns:
            DataFrame with columns:
                - scan_timestamp, dominant_regime, regime_confidence
                - current_session, session_volatility
                - regime scores (trending, ranging, breakout, etc.)
        """
        self.logger.info(f"Collecting market intelligence for last {days} days...")

        query = f"""
        SELECT
            scan_timestamp,
            dominant_regime,
            regime_confidence,
            current_session,
            session_volatility,
            market_bias,
            average_trend_strength,
            average_volatility,
            volatility_percentile,
            risk_sentiment,
            -- Individual regime scores
            regime_trending_score,
            regime_ranging_score,
            regime_breakout_score,
            regime_reversal_score,
            regime_high_vol_score,
            regime_low_vol_score,
            -- For per-epic correlation
            individual_epic_regimes
        FROM market_intelligence_history
        WHERE scan_timestamp >= NOW() - INTERVAL '{days} days'
        ORDER BY scan_timestamp ASC
        """

        df = self._execute_query(query)

        if df.empty:
            self.logger.warning("No market intelligence data found")
            return df

        self.logger.info(f"Collected {len(df)} market intelligence snapshots")
        return df

    def get_regime_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get distribution of market regimes over the period.

        Returns DataFrame with regime counts and percentages.
        """
        if df.empty:
            return pd.DataFrame()

        dist = df['dominant_regime'].value_counts().reset_index()
        dist.columns = ['regime', 'count']
        dist['percentage'] = dist['count'] / len(df) * 100

        return dist

    def get_session_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get distribution of sessions over the period.

        Returns DataFrame with session counts and percentages.
        """
        if df.empty:
            return pd.DataFrame()

        dist = df['current_session'].value_counts().reset_index()
        dist.columns = ['session', 'count']
        dist['percentage'] = dist['count'] / len(df) * 100

        return dist

    def correlate_with_trades(
        self,
        market_intel_df: pd.DataFrame,
        trade_df: pd.DataFrame,
        time_tolerance_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Correlate trade outcomes with market regime at entry time.

        Joins trade data with closest market intelligence snapshot
        to determine which regimes produce best results.

        Args:
            market_intel_df: Market intelligence data
            trade_df: Trade outcome data (must have entry_timestamp)
            time_tolerance_minutes: Max time difference for join

        Returns:
            Trade DataFrame enriched with market regime at entry time
        """
        if market_intel_df.empty or trade_df.empty:
            return trade_df

        # Convert timestamps
        market_intel_df = market_intel_df.copy()
        trade_df = trade_df.copy()

        market_intel_df['scan_timestamp'] = pd.to_datetime(market_intel_df['scan_timestamp'])
        trade_df['entry_timestamp'] = pd.to_datetime(trade_df['entry_timestamp'])

        # For each trade, find the closest market intel snapshot
        def find_closest_regime(entry_ts):
            if pd.isna(entry_ts):
                return None, None

            time_diffs = abs(market_intel_df['scan_timestamp'] - entry_ts)
            min_idx = time_diffs.idxmin()

            if time_diffs[min_idx].total_seconds() > time_tolerance_minutes * 60:
                return None, None

            return (
                market_intel_df.loc[min_idx, 'dominant_regime'],
                market_intel_df.loc[min_idx, 'current_session']
            )

        # Apply correlation
        results = trade_df['entry_timestamp'].apply(find_closest_regime)
        trade_df['regime_at_entry'] = [r[0] for r in results]
        trade_df['session_at_entry'] = [r[1] for r in results]

        return trade_df
