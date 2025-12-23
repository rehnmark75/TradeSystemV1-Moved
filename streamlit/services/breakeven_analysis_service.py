"""
Breakeven Analysis Service

Analyzes historical trade MFE/MAE data to calculate optimal breakeven trigger points
per epic and direction. Provides recommendations for trailing stop configuration.

Results are cached in the breakeven_analysis_cache table for fast Streamlit page loads.

Usage:
    service = BreakevenAnalysisService()

    # Get cached results (fast)
    results = service.get_cached_analysis()

    # Run fresh analysis and update cache
    results = service.run_full_analysis(trades_per_group=10)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from decimal import Decimal
import logging
import json
import sys
import os

# Add worker path for config import
worker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'worker', 'app'))
if worker_path not in sys.path:
    sys.path.insert(0, worker_path)

from services.db_utils import DatabaseContextManager, get_psycopg2_connection

logger = logging.getLogger(__name__)


@dataclass
class MFEMAEResult:
    """MFE/MAE calculation result for a single trade"""
    trade_id: int
    symbol: str
    direction: str
    mfe_pips: float
    mae_pips: float
    time_to_mfe_minutes: int
    time_to_mae_minutes: int
    moved_to_breakeven: bool
    pips_gained: float
    profit_loss: float
    is_winner: bool


@dataclass
class EpicAnalysis:
    """Analysis result for a single epic/direction combination"""
    epic: str
    direction: str
    trade_count: int
    win_rate: float
    avg_mfe: float
    median_mfe: float
    percentile_25_mfe: float
    percentile_75_mfe: float
    avg_mae: float
    median_mae: float
    percentile_75_mae: float
    optimal_be_trigger: float
    conservative_be_trigger: float
    current_be_trigger: float
    recommendation: str
    priority: str
    confidence: str
    be_reach_rate: float
    be_protection_rate: float
    be_profit_rate: float
    analysis_notes: str
    trades_analyzed: List[int]


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, handling Decimal and None."""
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        return default


def format_epic_display(epic: str) -> str:
    """Clean epic name for display (remove broker prefixes/suffixes)."""
    return epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')


class BreakevenAnalysisService:
    """Service for analyzing and recommending optimal breakeven triggers."""

    # Pip multiplier lookup
    PIP_MULTIPLIERS = {
        'JPY': 100,
        'CEEM': 1,
        'DEFAULT': 10000
    }

    # Default BE trigger if not in config
    DEFAULT_BE_TRIGGER = 25

    def __init__(self):
        self.trailing_config = self._load_trailing_config()

    def _load_trailing_config(self) -> Dict[str, Any]:
        """Load current trailing stop configuration."""
        try:
            from forex_scanner.config_trailing_stops import PAIR_TRAILING_CONFIGS
            return PAIR_TRAILING_CONFIGS
        except ImportError:
            logger.warning("Could not load trailing stop config, using defaults")
            return {}

    def _get_pip_multiplier(self, symbol: str) -> int:
        """Get pip multiplier based on currency pair type."""
        symbol_upper = symbol.upper()
        if 'CEEM' in symbol_upper:
            return self.PIP_MULTIPLIERS['CEEM']
        if 'JPY' in symbol_upper:
            return self.PIP_MULTIPLIERS['JPY']
        return self.PIP_MULTIPLIERS['DEFAULT']

    def _get_current_be_trigger(self, symbol: str) -> float:
        """Get current BE trigger from config."""
        # Try exact match first
        config = self.trailing_config.get(symbol, {})
        if config:
            return config.get('break_even_trigger_points', self.DEFAULT_BE_TRIGGER)

        # Try partial match (e.g., 'EURUSD' in 'CS.D.EURUSD.MINI.IP')
        for epic_key, epic_config in self.trailing_config.items():
            if symbol in epic_key or epic_key in symbol:
                return epic_config.get('break_even_trigger_points', self.DEFAULT_BE_TRIGGER)

        return self.DEFAULT_BE_TRIGGER

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    def get_cached_analysis(self, max_age_hours: int = 24) -> pd.DataFrame:
        """
        Get cached analysis results from database.

        Args:
            max_age_hours: Only return results newer than this many hours

        Returns:
            DataFrame with cached analysis results
        """
        query = """
        SELECT
            epic, direction, trade_count, win_rate,
            avg_mfe, median_mfe, percentile_25_mfe, percentile_75_mfe,
            avg_mae, median_mae, percentile_75_mae,
            optimal_be_trigger, conservative_be_trigger, current_be_trigger,
            recommendation, priority, confidence,
            be_reach_rate, be_protection_rate, be_profit_rate,
            analysis_notes, analyzed_at
        FROM breakeven_analysis_cache
        WHERE analyzed_at > NOW() - INTERVAL '%s hours'
        ORDER BY
            CASE priority
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                ELSE 3
            END,
            epic, direction
        """

        try:
            with DatabaseContextManager("trading") as conn:
                df = pd.read_sql(query, conn, params=(max_age_hours,))
                return df
        except Exception as e:
            logger.error(f"Error fetching cached analysis: {e}")
            return pd.DataFrame()

    def _save_analysis_to_cache(self, analysis: EpicAnalysis) -> bool:
        """Save a single analysis result to the cache table."""
        query = """
        INSERT INTO breakeven_analysis_cache (
            epic, direction, trade_count, win_rate,
            avg_mfe, median_mfe, percentile_25_mfe, percentile_75_mfe,
            avg_mae, median_mae, percentile_75_mae,
            optimal_be_trigger, conservative_be_trigger, current_be_trigger,
            recommendation, priority, confidence,
            be_reach_rate, be_protection_rate, be_profit_rate,
            analysis_notes, analyzed_at, trades_analyzed
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, NOW(), %s
        )
        ON CONFLICT (epic, direction) DO UPDATE SET
            trade_count = EXCLUDED.trade_count,
            win_rate = EXCLUDED.win_rate,
            avg_mfe = EXCLUDED.avg_mfe,
            median_mfe = EXCLUDED.median_mfe,
            percentile_25_mfe = EXCLUDED.percentile_25_mfe,
            percentile_75_mfe = EXCLUDED.percentile_75_mfe,
            avg_mae = EXCLUDED.avg_mae,
            median_mae = EXCLUDED.median_mae,
            percentile_75_mae = EXCLUDED.percentile_75_mae,
            optimal_be_trigger = EXCLUDED.optimal_be_trigger,
            conservative_be_trigger = EXCLUDED.conservative_be_trigger,
            current_be_trigger = EXCLUDED.current_be_trigger,
            recommendation = EXCLUDED.recommendation,
            priority = EXCLUDED.priority,
            confidence = EXCLUDED.confidence,
            be_reach_rate = EXCLUDED.be_reach_rate,
            be_protection_rate = EXCLUDED.be_protection_rate,
            be_profit_rate = EXCLUDED.be_profit_rate,
            analysis_notes = EXCLUDED.analysis_notes,
            analyzed_at = NOW(),
            trades_analyzed = EXCLUDED.trades_analyzed
        """

        try:
            with DatabaseContextManager("trading") as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (
                        analysis.epic, analysis.direction, analysis.trade_count, analysis.win_rate,
                        analysis.avg_mfe, analysis.median_mfe, analysis.percentile_25_mfe, analysis.percentile_75_mfe,
                        analysis.avg_mae, analysis.median_mae, analysis.percentile_75_mae,
                        analysis.optimal_be_trigger, analysis.conservative_be_trigger, analysis.current_be_trigger,
                        analysis.recommendation, analysis.priority, analysis.confidence,
                        analysis.be_reach_rate, analysis.be_protection_rate, analysis.be_profit_rate,
                        analysis.analysis_notes, json.dumps(analysis.trades_analyzed)
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error saving analysis to cache: {e}")
            return False

    def clear_cache(self) -> bool:
        """Clear all cached analysis results."""
        try:
            with DatabaseContextManager("trading") as conn:
                with conn.cursor() as cursor:
                    cursor.execute("TRUNCATE TABLE breakeven_analysis_cache")
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def fetch_trades_for_analysis(
        self,
        trades_per_group: int = 10,
        epic_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch closed trades grouped by epic and direction.

        Args:
            trades_per_group: Number of recent trades per epic/direction
            epic_filter: Optional list of epics to analyze

        Returns:
            DataFrame with trade data
        """
        query = """
        WITH ranked_trades AS (
            SELECT
                id,
                symbol,
                direction,
                entry_price,
                sl_price,
                tp_price,
                profit_loss,
                pips_gained,
                timestamp,
                closed_at,
                moved_to_breakeven,
                exit_price_calculated,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol, direction
                    ORDER BY closed_at DESC
                ) as row_num
            FROM trade_log
            WHERE status = 'closed'
              AND closed_at IS NOT NULL
              AND entry_price IS NOT NULL
              AND sl_price IS NOT NULL
        """

        params = [trades_per_group]

        if epic_filter:
            query += " AND symbol = ANY(%s)"
            params.append(epic_filter)

        query += """
        )
        SELECT * FROM ranked_trades
        WHERE row_num <= %s
        ORDER BY symbol, direction, closed_at DESC
        """

        # Move trades_per_group to end for the final WHERE
        params_final = params[1:] + [trades_per_group] if len(params) > 1 else [trades_per_group]

        try:
            with DatabaseContextManager("trading") as conn:
                if epic_filter:
                    return pd.read_sql(query, conn, params=(epic_filter, trades_per_group))
                else:
                    # Simplified query without filter
                    simple_query = """
                    WITH ranked_trades AS (
                        SELECT
                            id, symbol, direction, entry_price, sl_price, tp_price,
                            profit_loss, pips_gained, timestamp, closed_at,
                            moved_to_breakeven, exit_price_calculated,
                            ROW_NUMBER() OVER (PARTITION BY symbol, direction ORDER BY closed_at DESC) as row_num
                        FROM trade_log
                        WHERE status = 'closed' AND closed_at IS NOT NULL
                              AND entry_price IS NOT NULL AND sl_price IS NOT NULL
                    )
                    SELECT * FROM ranked_trades WHERE row_num <= %s
                    ORDER BY symbol, direction, closed_at DESC
                    """
                    return pd.read_sql(simple_query, conn, params=(trades_per_group,))
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return pd.DataFrame()

    def fetch_candles_for_trade(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        timeframe: int = 5
    ) -> pd.DataFrame:
        """Fetch candles for MFE/MAE calculation."""
        query = """
        SELECT
            start_time,
            epic,
            timeframe,
            open,
            high,
            low,
            close,
            volume
        FROM ig_candles
        WHERE epic = %s
          AND timeframe = %s
          AND start_time >= %s
          AND start_time <= %s
        ORDER BY start_time ASC
        """

        try:
            with DatabaseContextManager("trading") as conn:
                return pd.read_sql(query, conn, params=(symbol, timeframe, entry_time, exit_time))
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # MFE/MAE CALCULATION
    # =========================================================================

    def calculate_mfe_mae(
        self,
        trade: pd.Series,
        candles: pd.DataFrame
    ) -> Optional[MFEMAEResult]:
        """Calculate MFE/MAE for a single trade."""
        if candles.empty:
            return None

        symbol = trade['symbol']
        pip_multiplier = self._get_pip_multiplier(symbol)
        entry_price = safe_float(trade['entry_price'])
        direction = trade['direction']

        if entry_price <= 0:
            return None

        # CEEM epics have trade data scaled by 10000x vs candle data
        # Trade: 11633, Candle: 1.1633 - need to normalize entry price AND use correct pip multiplier
        if 'CEEM' in symbol.upper() and entry_price > 1000:
            entry_price = entry_price / 10000.0
            pip_multiplier = 10000  # After normalization, use standard EUR pip multiplier
            logger.debug(f"CEEM price normalized: {trade['entry_price']} -> {entry_price}, pip_multiplier: {pip_multiplier}")

        mfe_pips = 0.0
        mae_pips = 0.0
        mfe_time = 0
        mae_time = 0
        entry_time = trade['timestamp']

        for idx, candle in candles.iterrows():
            high = safe_float(candle['high'])
            low = safe_float(candle['low'])
            candle_time = candle['start_time']

            if direction == "BUY":
                favorable = (high - entry_price) * pip_multiplier
                adverse = (entry_price - low) * pip_multiplier
            else:  # SELL
                favorable = (entry_price - low) * pip_multiplier
                adverse = (high - entry_price) * pip_multiplier

            if favorable > mfe_pips:
                mfe_pips = favorable
                if isinstance(candle_time, datetime) and isinstance(entry_time, datetime):
                    mfe_time = int((candle_time - entry_time).total_seconds() / 60)

            if adverse > mae_pips:
                mae_pips = adverse
                if isinstance(candle_time, datetime) and isinstance(entry_time, datetime):
                    mae_time = int((candle_time - entry_time).total_seconds() / 60)

        pips_gained = safe_float(trade.get('pips_gained', 0))
        profit_loss = safe_float(trade.get('profit_loss', 0))

        # Determine if winner
        if pips_gained != 0:
            is_winner = pips_gained > 2
        else:
            is_winner = profit_loss > 5

        return MFEMAEResult(
            trade_id=int(trade['id']),
            symbol=trade['symbol'],
            direction=direction,
            mfe_pips=round(mfe_pips, 1),
            mae_pips=round(mae_pips, 1),
            time_to_mfe_minutes=mfe_time,
            time_to_mae_minutes=mae_time,
            moved_to_breakeven=bool(trade.get('moved_to_breakeven', False)),
            pips_gained=pips_gained,
            profit_loss=profit_loss,
            is_winner=is_winner
        )

    # =========================================================================
    # OPTIMAL TRIGGER CALCULATION
    # =========================================================================

    def calculate_optimal_trigger(
        self,
        mfe_values: List[float],
        mae_values: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate optimal BE trigger from MFE/MAE distributions.

        Strategy:
        1. Lock at 60% of median MFE (capture most of typical favorable move)
        2. Ensure trigger is above 75th percentile MAE (survive normal drawdowns)
        3. Compare to current config and flag significant differences
        """
        if not mfe_values or len(mfe_values) < 3:
            return {
                'optimal_trigger': None,
                'conservative_trigger': None,
                'confidence': 'low',
                'metrics': {},
                'notes': 'Insufficient data (need at least 3 trades)'
            }

        mfe_array = np.array(mfe_values)
        mae_array = np.array(mae_values) if mae_values else np.array([0])

        # Core MFE metrics
        median_mfe = float(np.median(mfe_array))
        avg_mfe = float(np.mean(mfe_array))
        percentile_25_mfe = float(np.percentile(mfe_array, 25))
        percentile_75_mfe = float(np.percentile(mfe_array, 75))

        # MAE metrics
        median_mae = float(np.median(mae_array))
        percentile_75_mae = float(np.percentile(mae_array, 75))

        # Optimal BE trigger calculation
        # Formula: Lock at 60% of median MFE, but ensure above worst drawdowns
        base_trigger = median_mfe * 0.6

        # Safety floor: trigger must be above 75th percentile MAE
        safety_floor = percentile_75_mae * 1.2

        optimal_trigger = max(base_trigger, safety_floor)

        # Conservative alternative: use 25th percentile MFE (works for weak trades too)
        conservative_trigger = max(percentile_25_mfe * 0.7, safety_floor)

        # Confidence based on sample size and MFE consistency
        mfe_std = float(np.std(mfe_array))
        coefficient_of_variation = mfe_std / avg_mfe if avg_mfe > 0 else 1.0

        if len(mfe_values) >= 8 and coefficient_of_variation < 0.5:
            confidence = 'high'
        elif len(mfe_values) >= 5 and coefficient_of_variation < 0.7:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'optimal_trigger': round(optimal_trigger, 1),
            'conservative_trigger': round(conservative_trigger, 1),
            'confidence': confidence,
            'metrics': {
                'median_mfe': round(median_mfe, 1),
                'avg_mfe': round(avg_mfe, 1),
                'percentile_25_mfe': round(percentile_25_mfe, 1),
                'percentile_75_mfe': round(percentile_75_mfe, 1),
                'median_mae': round(median_mae, 1),
                'percentile_75_mae': round(percentile_75_mae, 1),
                'mfe_std': round(mfe_std, 1),
                'sample_size': len(mfe_values)
            },
            'notes': f'Based on {len(mfe_values)} trades'
        }

    # =========================================================================
    # EFFICIENCY ANALYSIS
    # =========================================================================

    def analyze_efficiency(
        self,
        results: List[MFEMAEResult],
        current_be: float
    ) -> Dict[str, Any]:
        """Analyze BE efficiency for a group of trades."""
        if not results:
            return {
                'be_reach_rate': 0,
                'be_protection_rate': 0,
                'be_profit_rate': 0,
                'recommendation': 'NO_DATA',
                'reason': 'No trades to analyze'
            }

        be_moved_trades = [r for r in results if r.moved_to_breakeven]

        # How many trades reached the BE trigger level?
        trades_reached_be = sum(1 for r in results if r.mfe_pips >= current_be)
        be_reach_rate = trades_reached_be / len(results) * 100

        # For trades that moved to BE
        if be_moved_trades:
            # How many hit BE exactly (small profit/loss)?
            be_hit_exact = sum(1 for r in be_moved_trades if abs(r.pips_gained) < 3)
            # How many continued to profit?
            be_continued_profit = sum(1 for r in be_moved_trades if r.pips_gained > 3)

            be_protection_rate = be_hit_exact / len(be_moved_trades) * 100
            be_profit_rate = be_continued_profit / len(be_moved_trades) * 100
        else:
            be_protection_rate = 0
            be_profit_rate = 0

        # Recommendation logic
        if be_reach_rate < 30:
            recommendation = "LOWER_TRIGGER"
            reason = f"Only {be_reach_rate:.0f}% of trades reach BE level - trigger may be too high"
        elif be_protection_rate > 60:
            recommendation = "TRIGGER_TOO_TIGHT"
            reason = f"{be_protection_rate:.0f}% of BE trades exit at exactly BE - may be cutting winners short"
        elif be_profit_rate > 60:
            recommendation = "TRIGGER_OPTIMAL"
            reason = f"{be_profit_rate:.0f}% of BE trades continue to profit - good balance"
        else:
            recommendation = "REVIEW_NEEDED"
            reason = "Mixed results - manual review recommended"

        return {
            'be_reach_rate': round(be_reach_rate, 1),
            'be_protection_rate': round(be_protection_rate, 1),
            'be_profit_rate': round(be_profit_rate, 1),
            'recommendation': recommendation,
            'reason': reason
        }

    # =========================================================================
    # RECOMMENDATION GENERATION
    # =========================================================================

    def _generate_recommendation(
        self,
        current_be: float,
        optimal_data: Dict[str, Any],
        efficiency: Dict[str, Any],
        epic: str,
        direction: str
    ) -> Dict[str, Any]:
        """Generate actionable recommendation based on analysis."""
        optimal_be = optimal_data.get('optimal_trigger')

        if optimal_be is None:
            return {
                'action': 'NO_CHANGE',
                'reason': 'Insufficient data for recommendation',
                'priority': 'low'
            }

        difference = optimal_be - current_be
        pct_difference = abs(difference) / current_be * 100 if current_be > 0 else 100

        # Decision matrix
        if pct_difference < 10:
            action = "KEEP_CURRENT"
            reason = f"Current ({current_be:.0f}) is within 10% of optimal ({optimal_be:.0f})"
            priority = "low"
        elif difference < 0 and pct_difference > 20:
            action = "LOWER_TRIGGER"
            reason = f"Lower from {current_be:.0f} to {optimal_be:.0f} pips (-{abs(difference):.0f})"
            priority = "high" if efficiency.get('be_reach_rate', 100) < 40 else "medium"
        elif difference > 0 and pct_difference > 20:
            action = "RAISE_TRIGGER"
            reason = f"Raise from {current_be:.0f} to {optimal_be:.0f} pips (+{difference:.0f})"
            priority = "medium" if efficiency.get('be_protection_rate', 0) > 50 else "low"
        else:
            action = "MINOR_ADJUSTMENT"
            reason = f"Consider adjusting from {current_be:.0f} to {optimal_be:.0f} pips"
            priority = "low"

        return {
            'action': action,
            'reason': reason,
            'priority': priority
        }

    # =========================================================================
    # MAIN ANALYSIS METHODS
    # =========================================================================

    def analyze_epic_direction(
        self,
        trades: pd.DataFrame,
        epic: str,
        direction: str
    ) -> Optional[EpicAnalysis]:
        """Perform full analysis for an epic/direction combination."""
        subset = trades[
            (trades['symbol'] == epic) &
            (trades['direction'] == direction)
        ]

        if len(subset) < 3:
            logger.info(f"Skipping {epic} {direction}: only {len(subset)} trades")
            return None

        # Calculate MFE/MAE for each trade
        results: List[MFEMAEResult] = []
        trade_ids: List[int] = []

        for _, trade in subset.iterrows():
            candles = self.fetch_candles_for_trade(
                symbol=trade['symbol'],
                entry_time=trade['timestamp'],
                exit_time=trade['closed_at']
            )

            if len(candles) > 0:
                result = self.calculate_mfe_mae(trade, candles)
                if result:
                    results.append(result)
                    trade_ids.append(result.trade_id)

        if len(results) < 3:
            logger.info(f"Skipping {epic} {direction}: only {len(results)} trades with candle data")
            return None

        # Extract MFE/MAE values
        mfe_values = [r.mfe_pips for r in results]
        mae_values = [r.mae_pips for r in results]

        # Calculate optimal trigger
        optimal_data = self.calculate_optimal_trigger(mfe_values, mae_values)

        # Get current config
        current_be = self._get_current_be_trigger(epic)

        # Analyze efficiency
        efficiency = self.analyze_efficiency(results, current_be)

        # Generate recommendation
        rec = self._generate_recommendation(
            current_be, optimal_data, efficiency, epic, direction
        )

        # Calculate stats
        winners = [r for r in results if r.is_winner]
        win_rate = len(winners) / len(results) * 100

        metrics = optimal_data.get('metrics', {})

        return EpicAnalysis(
            epic=epic,
            direction=direction,
            trade_count=len(results),
            win_rate=round(win_rate, 1),
            avg_mfe=metrics.get('avg_mfe', 0),
            median_mfe=metrics.get('median_mfe', 0),
            percentile_25_mfe=metrics.get('percentile_25_mfe', 0),
            percentile_75_mfe=metrics.get('percentile_75_mfe', 0),
            avg_mae=metrics.get('median_mae', 0),  # Using median for avg display
            median_mae=metrics.get('median_mae', 0),
            percentile_75_mae=metrics.get('percentile_75_mae', 0),
            optimal_be_trigger=optimal_data.get('optimal_trigger', 0) or 0,
            conservative_be_trigger=optimal_data.get('conservative_trigger', 0) or 0,
            current_be_trigger=current_be,
            recommendation=rec['action'],
            priority=rec['priority'],
            confidence=optimal_data.get('confidence', 'low'),
            be_reach_rate=efficiency.get('be_reach_rate', 0),
            be_protection_rate=efficiency.get('be_protection_rate', 0),
            be_profit_rate=efficiency.get('be_profit_rate', 0),
            analysis_notes=f"{rec['reason']}. {efficiency.get('reason', '')}",
            trades_analyzed=trade_ids
        )

    def run_full_analysis(
        self,
        trades_per_group: int = 10,
        epic_filter: Optional[List[str]] = None,
        save_to_cache: bool = True
    ) -> List[EpicAnalysis]:
        """
        Run full analysis for all epics and save to cache.

        Args:
            trades_per_group: Number of recent trades per epic/direction to analyze
            epic_filter: Optional list of specific epics to analyze
            save_to_cache: Whether to save results to database cache

        Returns:
            List of EpicAnalysis objects
        """
        logger.info(f"Starting full breakeven analysis (trades_per_group={trades_per_group})")

        trades = self.fetch_trades_for_analysis(trades_per_group, epic_filter)

        if trades.empty:
            logger.warning("No trades found for analysis")
            return []

        logger.info(f"Fetched {len(trades)} trades for analysis")

        results: List[EpicAnalysis] = []

        # Get unique epic/direction combinations
        for epic in trades['symbol'].unique():
            for direction in ['BUY', 'SELL']:
                logger.info(f"Analyzing {epic} {direction}...")
                analysis = self.analyze_epic_direction(trades, epic, direction)
                if analysis:
                    results.append(analysis)
                    if save_to_cache:
                        self._save_analysis_to_cache(analysis)

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        results.sort(key=lambda x: (priority_order.get(x.priority, 3), x.epic))

        logger.info(f"Analysis complete: {len(results)} epic/direction pairs analyzed")

        return results

    def get_summary_dataframe(
        self,
        analyses: List[EpicAnalysis]
    ) -> pd.DataFrame:
        """Convert analysis results to display DataFrame."""
        if not analyses:
            return pd.DataFrame()

        rows = []
        for a in analyses:
            epic_display = format_epic_display(a.epic)

            # Calculate difference
            diff = a.optimal_be_trigger - a.current_be_trigger
            diff_str = f"+{diff:.0f}" if diff > 0 else f"{diff:.0f}"

            rows.append({
                'Epic': epic_display,
                'Dir': a.direction,
                'Trades': a.trade_count,
                'Win%': f"{a.win_rate:.0f}%",
                'Avg MFE': f"{a.avg_mfe:.0f}",
                'Med MFE': f"{a.median_mfe:.0f}",
                'Avg MAE': f"{a.avg_mae:.0f}",
                'Optimal BE': f"{a.optimal_be_trigger:.0f}",
                'Current BE': f"{a.current_be_trigger:.0f}",
                'Diff': diff_str,
                'Action': a.recommendation,
                'Priority': a.priority.upper(),
                'Confidence': a.confidence.upper()
            })

        return pd.DataFrame(rows)

    def get_available_epics(self) -> List[str]:
        """Get list of epics with recent trades for filter dropdown."""
        query = """
        SELECT DISTINCT symbol
        FROM trade_log
        WHERE status = 'closed'
          AND closed_at > NOW() - INTERVAL '60 days'
        ORDER BY symbol
        """

        try:
            with DatabaseContextManager("trading") as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching available epics: {e}")
            return []
