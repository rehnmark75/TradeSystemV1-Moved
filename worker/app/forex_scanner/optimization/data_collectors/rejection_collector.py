# optimization/data_collectors/rejection_collector.py
"""
Rejection data collector for unified parameter optimizer.
Collects rejected signal outcomes from smc_simple_rejections and smc_rejection_outcomes.
"""

import logging
import pandas as pd
from typing import List

from .base_collector import BaseCollector

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


class RejectionCollector(BaseCollector):
    """Collects rejection outcome data from smc_simple_rejections"""

    def __init__(self, db_manager: DatabaseManager = None):
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)

    def collect(self, days: int = 30, epics: List[str] = None) -> pd.DataFrame:
        """
        Collect rejection outcomes (would-be winners/losers).

        Args:
            days: Number of days of historical data
            epics: Optional list of epics to filter

        Returns:
            DataFrame with columns:
                - epic, pair, direction, rejection_stage
                - confidence_score, volume_ratio, pullback_depth
                - outcome (HIT_TP/HIT_SL), would_be_winner, potential_profit_pips
                - market_session, market_hour
        """
        self.logger.info(f"Collecting rejection data for last {days} days...")

        # Build epic filter
        epic_filter = ""
        if epics:
            epic_list = "', '".join(epics)
            epic_filter = f"AND r.epic IN ('{epic_list}')"

        # Cast numeric columns to FLOAT to avoid Decimal type issues with pandas
        # COMPREHENSIVE DATA COLLECTION - All available indicators for proper analysis
        query = f"""
        SELECT
            r.epic,
            r.pair,
            r.attempted_direction as direction,
            r.rejection_stage,
            r.rejection_reason,
            r.scan_timestamp,
            r.market_session,
            r.market_hour,
            r.is_market_hours,

            -- Core signal parameters (cast to float)
            CAST(r.confidence_score AS FLOAT) as confidence_score,
            CAST(r.volume_ratio AS FLOAT) as volume_ratio,
            CAST(r.pullback_depth AS FLOAT) as pullback_depth,
            r.fib_zone as fib_level,

            -- ATR & Volatility
            CAST(r.atr_15m AS FLOAT) as atr_15m,
            CAST(r.atr_5m AS FLOAT) as atr_5m,
            CAST(r.atr_percentile AS FLOAT) as atr_percentile,
            r.volatility_state,
            CAST(r.bb_width_percentile AS FLOAT) as bb_width_percentile,

            -- Swing structure
            CAST(r.swing_high_level AS FLOAT) as swing_high_level,
            CAST(r.swing_low_level AS FLOAT) as swing_low_level,
            r.swing_lookback_bars,
            r.swings_found_count,
            r.last_swing_bars_ago,
            CAST(r.swing_range_pips AS FLOAT) as swing_range_pips,

            -- Entry/Exit levels
            CAST(r.potential_entry AS FLOAT) as potential_entry,
            CAST(r.potential_stop_loss AS FLOAT) as potential_stop_loss,
            CAST(r.potential_take_profit AS FLOAT) as potential_take_profit,
            CAST(r.potential_rr_ratio AS FLOAT) as potential_rr_ratio,
            CAST(r.potential_risk_pips AS FLOAT) as potential_risk_pips,
            CAST(r.potential_reward_pips AS FLOAT) as potential_reward_pips,

            -- MACD (histogram STRENGTH matters, not just direction!)
            r.macd_aligned,
            r.macd_momentum,  -- categorical: 'bullish'/'bearish'
            CAST(r.macd_histogram AS FLOAT) as macd_histogram,  -- numeric histogram value
            CAST(r.macd_line AS FLOAT) as macd_line,
            CAST(r.macd_signal AS FLOAT) as macd_signal,

            -- Efficiency Ratio (trend quality measure)
            CAST(r.efficiency_ratio AS FLOAT) as efficiency_ratio,

            -- Market Regime
            r.market_regime_detected,

            -- Bollinger Bands
            CAST(r.bb_upper AS FLOAT) as bb_upper,
            CAST(r.bb_middle AS FLOAT) as bb_middle,
            CAST(r.bb_lower AS FLOAT) as bb_lower,
            CAST(r.bb_width AS FLOAT) as bb_width,
            CAST(r.bb_percent_b AS FLOAT) as bb_percent_b,

            -- ADX (trend strength)
            CAST(r.adx_value AS FLOAT) as adx_value,
            r.adx_trend_strength,

            -- Stochastics
            CAST(r.stoch_k AS FLOAT) as stoch_k,
            CAST(r.stoch_d AS FLOAT) as stoch_d,
            r.stoch_zone,

            -- Supertrend
            CAST(r.supertrend_value AS FLOAT) as supertrend_value,
            r.supertrend_direction,

            -- RSI
            r.rsi_zone,

            -- EMAs
            CAST(r.ema_9 AS FLOAT) as ema_9,
            CAST(r.ema_21 AS FLOAT) as ema_21,
            CAST(r.ema_50 AS FLOAT) as ema_50,
            CAST(r.ema_200 AS FLOAT) as ema_200,
            r.price_vs_ema_200,
            CAST(r.ema_4h_value AS FLOAT) as ema_4h_value,
            CAST(r.ema_distance_pips AS FLOAT) as ema_distance_pips,
            r.price_position_vs_ema,

            -- KAMA
            CAST(r.kama_value AS FLOAT) as kama_value,
            CAST(r.kama_er AS FLOAT) as kama_er,
            r.kama_trend,

            -- Outcome analysis (cast to float)
            o.outcome,
            CAST(o.outcome_price AS FLOAT) as outcome_price,
            o.outcome_timestamp,
            CAST(o.potential_profit_pips AS FLOAT) as potential_profit_pips,
            CAST(o.max_favorable_excursion_pips AS FLOAT) as mfe_pips,
            CAST(o.max_adverse_excursion_pips AS FLOAT) as mae_pips,
            o.time_to_outcome_minutes,
            o.time_to_mfe_minutes,
            o.time_to_mae_minutes,
            o.candle_count_analyzed,
            CAST(o.data_quality_score AS FLOAT) as data_quality_score,
            CASE WHEN o.outcome = 'HIT_TP' THEN true ELSE false END as would_be_winner
        FROM smc_simple_rejections r
        JOIN smc_rejection_outcomes o ON r.id = o.rejection_id
        WHERE r.scan_timestamp >= NOW() - INTERVAL '{days} days'
          AND o.outcome IN ('HIT_TP', 'HIT_SL')
          AND r.attempted_direction IS NOT NULL
          {epic_filter}
        ORDER BY r.scan_timestamp DESC
        """

        df = self._execute_query(query)

        if df.empty:
            self.logger.warning("No rejection outcome data found")
            return df

        # Normalize direction
        df['direction'] = df['direction'].apply(self._normalize_direction)

        self.logger.info(f"Collected {len(df)} rejection outcomes")
        return df

    def get_summary_by_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get rejection outcome summary by stage.

        Shows win rate of rejected signals per rejection stage,
        helping identify which filters are over-rejecting good signals.
        """
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby(['epic', 'rejection_stage']).agg(
            total_rejections=('epic', 'count'),
            would_be_winners=('would_be_winner', 'sum'),
            avg_potential_pips=('potential_profit_pips', 'mean'),
            avg_mfe=('mfe_pips', 'mean'),
            avg_mae=('mae_pips', 'mean')
        ).reset_index()

        summary['would_be_losers'] = summary['total_rejections'] - summary['would_be_winners']
        summary['would_be_win_rate'] = summary['would_be_winners'] / summary['total_rejections']

        # Calculate missed profit (sum of positive outcomes)
        missed_profit = df[df['would_be_winner']].groupby(['epic', 'rejection_stage']).agg(
            missed_pips=('potential_profit_pips', 'sum')
        ).reset_index()

        summary = summary.merge(missed_profit, on=['epic', 'rejection_stage'], how='left')
        summary['missed_pips'] = summary['missed_pips'].fillna(0)

        return summary

    def get_summary_by_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get rejection outcome summary by direction.

        Shows if BULL or BEAR signals are being over-rejected.
        """
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby(['epic', 'direction']).agg(
            total_rejections=('epic', 'count'),
            would_be_winners=('would_be_winner', 'sum'),
            avg_potential_pips=('potential_profit_pips', 'mean'),
            avg_confidence=('confidence_score', 'mean')
        ).reset_index()

        summary['would_be_win_rate'] = summary['would_be_winners'] / summary['total_rejections']

        return summary

    def get_parameter_correlation(self, df: pd.DataFrame, parameter: str) -> pd.DataFrame:
        """
        Analyze correlation between a parameter and rejection outcomes.

        Args:
            df: Rejection data
            parameter: Column name to analyze (e.g., 'confidence_score', 'volume_ratio')

        Returns:
            DataFrame showing win rate by parameter bins
        """
        if df.empty or parameter not in df.columns:
            return pd.DataFrame()

        # Filter out nulls
        df_valid = df[df[parameter].notna()].copy()

        if len(df_valid) < 10:
            return pd.DataFrame()

        # Create bins (quartiles)
        try:
            df_valid['param_bin'] = pd.qcut(df_valid[parameter], q=4, duplicates='drop')
        except ValueError:
            # Not enough unique values for quartiles
            df_valid['param_bin'] = pd.cut(df_valid[parameter], bins=4)

        summary = df_valid.groupby(['epic', 'param_bin'], observed=True).agg(
            count=('epic', 'count'),
            would_be_winners=('would_be_winner', 'sum')
        ).reset_index()

        summary['would_be_win_rate'] = summary['would_be_winners'] / summary['count']
        summary['parameter'] = parameter

        return summary
