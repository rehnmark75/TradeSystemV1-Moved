"""
SMC Rejection Outcome Analyzer

Daily batch job that analyzes rejected SMC Simple signals to determine
whether they would have been profitable with fixed SL=9 pips and TP=15 pips.

Usage:
    # Run once (analyze yesterday's rejections)
    docker exec -it task-worker python /app/forex_scanner/monitoring/rejection_outcome_analyzer.py

    # Run with custom lookback days
    docker exec -it task-worker python /app/forex_scanner/monitoring/rejection_outcome_analyzer.py --days 3

    # Run for specific date
    docker exec -it task-worker python /app/forex_scanner/monitoring/rejection_outcome_analyzer.py --date 2025-12-27

    # Dry run (analyze but don't save)
    docker exec -it task-worker python /app/forex_scanner/monitoring/rejection_outcome_analyzer.py --dry-run

Created: 2025-12-28
"""

import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database and configuration - use direct SQLAlchemy to avoid complex imports
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")

# DEPRECATED: These are now loaded from database via get_smc_simple_config()
# Keeping as fallback defaults only
_DEFAULT_FIXED_STOP_LOSS_PIPS = 9
_DEFAULT_FIXED_TAKE_PROFIT_PIPS = 15

PAIR_INFO = {
    'CS.D.EURUSD.CEEM.IP': {'pair': 'EURUSD', 'pip_multiplier': 10000},
    'CS.D.GBPUSD.MINI.IP': {'pair': 'GBPUSD', 'pip_multiplier': 10000},
    'CS.D.USDJPY.MINI.IP': {'pair': 'USDJPY', 'pip_multiplier': 100},
    'CS.D.EURJPY.MINI.IP': {'pair': 'EURJPY', 'pip_multiplier': 100},
    'CS.D.GBPJPY.MINI.IP': {'pair': 'GBPJPY', 'pip_multiplier': 100},
    'CS.D.AUDUSD.MINI.IP': {'pair': 'AUDUSD', 'pip_multiplier': 10000},
    'CS.D.USDCHF.MINI.IP': {'pair': 'USDCHF', 'pip_multiplier': 10000},
    'CS.D.USDCAD.MINI.IP': {'pair': 'USDCAD', 'pip_multiplier': 10000},
    'CS.D.NZDUSD.MINI.IP': {'pair': 'NZDUSD', 'pip_multiplier': 10000},
    'CS.D.AUDJPY.MINI.IP': {'pair': 'AUDJPY', 'pip_multiplier': 100},
    'CS.D.CADJPY.MINI.IP': {'pair': 'CADJPY', 'pip_multiplier': 100},
    'CS.D.CHFJPY.MINI.IP': {'pair': 'CHFJPY', 'pip_multiplier': 100},
    'CS.D.NZDJPY.MINI.IP': {'pair': 'NZDJPY', 'pip_multiplier': 100},
    'CS.D.EURGBP.MINI.IP': {'pair': 'EURGBP', 'pip_multiplier': 10000},
}


class SimpleDatabaseManager:
    """Simple database manager using SQLAlchemy for standalone operation"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        logger.info(f"Database connected: {database_url.split('@')[-1]}")

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                query_upper = query.strip().upper()

                if query_upper.startswith('INSERT') or query_upper.startswith('UPDATE'):
                    conn.commit()
                    if 'RETURNING' in query_upper:
                        return pd.DataFrame(result.fetchall(), columns=result.keys())
                    return pd.DataFrame()
                else:
                    return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise


def _safe_int(value) -> Optional[int]:
    """Safely convert value to int, handling NaN and None."""
    if value is None:
        return None
    try:
        import numpy as np
        import math
        # Check for NaN (works for both numpy and Python floats)
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, np.floating) and np.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


class OutcomeType(Enum):
    """Possible outcomes for rejected signals"""
    HIT_TP = "HIT_TP"           # Would have hit take profit first
    HIT_SL = "HIT_SL"           # Would have hit stop loss first
    STILL_OPEN = "STILL_OPEN"   # Neither hit within analysis window
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Not enough candle data


@dataclass
class RejectionOutcome:
    """Data class for rejection outcome analysis result"""
    rejection_id: int
    epic: str
    pair: str
    rejection_timestamp: datetime
    rejection_stage: str
    attempted_direction: str
    market_session: Optional[str]
    market_hour: Optional[int]
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    spread_at_rejection: Optional[float]
    outcome: OutcomeType
    outcome_price: Optional[float]
    outcome_timestamp: Optional[datetime]
    time_to_outcome_minutes: Optional[int]
    max_favorable_excursion_pips: float
    max_adverse_excursion_pips: float
    mfe_timestamp: Optional[datetime]
    mae_timestamp: Optional[datetime]
    time_to_mfe_minutes: Optional[int]
    time_to_mae_minutes: Optional[int]
    potential_profit_pips: float
    candle_count_analyzed: int
    data_quality_score: float
    analysis_notes: Optional[str]


class RejectionOutcomeAnalyzer:
    """
    Analyzes SMC Simple rejected signals to determine hypothetical outcomes.

    Success Definition:
    - A rejected signal is considered a "would-be winner" if:
      - For BUY signals: price would have hit TP (entry + 15 pips) BEFORE
        hitting SL (entry - 9 pips)
      - For SELL signals: price would have hit TP (entry - 15 pips) BEFORE
        hitting SL (entry + 9 pips)

    Features:
    - Batch analysis of rejections from specified date range
    - MFE/MAE calculation for each rejected signal
    - Data quality scoring
    - Database persistence of results
    """

    def __init__(self, db_manager: SimpleDatabaseManager = None):
        """
        Initialize the analyzer.

        Args:
            db_manager: SimpleDatabaseManager instance (will create if None)
        """
        self.db_manager = db_manager or SimpleDatabaseManager(DATABASE_URL)
        self.logger = logging.getLogger(__name__)

        # Configuration - load from database with fallback to defaults
        self.fixed_sl_pips = _DEFAULT_FIXED_STOP_LOSS_PIPS
        self.fixed_tp_pips = _DEFAULT_FIXED_TAKE_PROFIT_PIPS
        self._smc_config = None
        try:
            from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
            self._smc_config = get_smc_simple_config()
            self.fixed_sl_pips = self._smc_config.fixed_stop_loss_pips or _DEFAULT_FIXED_STOP_LOSS_PIPS
            self.fixed_tp_pips = self._smc_config.fixed_take_profit_pips or _DEFAULT_FIXED_TAKE_PROFIT_PIPS
            self.logger.info("Loaded SL/TP config from database")
        except Exception as e:
            self.logger.warning(f"Could not load database config, using defaults: {e}")

        # Analysis window: How long to look for outcome (hours after rejection)
        self.max_analysis_hours = 48  # 48 hours max (swing mode)
        self.max_analysis_hours_scalp = 8  # 8 hours max for scalp mode
        self.min_candles_required_5m = 12  # At least 12 5-minute candles (1 hour)
        self.min_candles_required_1m = 60  # At least 60 1-minute candles (1 hour)

        self.logger.info(f"RejectionOutcomeAnalyzer initialized")
        self.logger.info(f"  Fixed SL: {self.fixed_sl_pips} pips, Fixed TP: {self.fixed_tp_pips} pips")
        self.logger.info(f"  Analysis window: {self.max_analysis_hours} hours max")

    def get_pip_multiplier(self, epic: str) -> int:
        """Get pip multiplier for an epic (10000 for most pairs, 100 for JPY)"""
        pair_info = PAIR_INFO.get(epic, {})
        return pair_info.get('pip_multiplier', 10000)

    def get_pip_value(self, epic: str) -> float:
        """Get pip value (price movement per pip) for an epic"""
        multiplier = self.get_pip_multiplier(epic)
        return 1.0 / multiplier

    def get_pair_sl_tp(self, epic: str) -> tuple:
        """Get SL/TP for a specific pair (per-pair or global fallback).

        Returns:
            tuple: (sl_pips, tp_pips)
        """
        if self._smc_config:
            sl = self._smc_config.get_pair_fixed_stop_loss(epic)
            tp = self._smc_config.get_pair_fixed_take_profit(epic)
            if sl and tp:
                return (sl, tp)
        return (self.fixed_sl_pips, self.fixed_tp_pips)

    def is_scalp_rejection(self, rejection_stage: str) -> bool:
        """
        Determine if a rejection is scalp-related (needs 1m candles).

        Args:
            rejection_stage: The rejection stage name

        Returns:
            True if scalp rejection, False otherwise
        """
        scalp_stages = [
            'SCALP_ENTRY_FILTER',
            'SCALP_SPREAD',
            'SCALP_QUALIFICATION',
            'VOLUME_LOW'  # Often occurs in scalp mode
        ]
        return rejection_stage in scalp_stages

    def get_unanalyzed_rejections(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch rejections that haven't been analyzed yet.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with rejection data
        """
        query = """
        SELECT
            r.id as rejection_id,
            r.scan_timestamp as rejection_timestamp,
            r.epic,
            r.pair,
            r.rejection_stage,
            r.attempted_direction,
            r.market_session,
            r.market_hour,
            r.current_price,
            r.bid_price,
            r.ask_price,
            r.spread_pips,
            r.potential_entry,
            r.potential_stop_loss,
            r.potential_take_profit
        FROM smc_simple_rejections r
        LEFT JOIN smc_rejection_outcomes o ON r.id = o.rejection_id
        WHERE r.scan_timestamp >= :start_date
          AND r.scan_timestamp < :end_date
          AND r.attempted_direction IS NOT NULL
          AND o.id IS NULL
        ORDER BY r.scan_timestamp ASC
        """

        try:
            df = self.db_manager.execute_query(query, {
                'start_date': start_date,
                'end_date': end_date
            })
            self.logger.info(f"Found {len(df)} unanalyzed rejections between {start_date} and {end_date}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch rejections: {e}")
            return pd.DataFrame()

    def fetch_price_data(self, epic: str, start_time: datetime, end_time: datetime, timeframe: int = 5) -> pd.DataFrame:
        """
        Fetch candles for outcome analysis from ig_candles table.

        Args:
            epic: Currency pair epic
            start_time: Analysis window start
            end_time: Analysis window end
            timeframe: Candle timeframe in minutes (1 for scalp, 5 for swing)

        Returns:
            DataFrame with OHLC data
        """
        query = """
        SELECT
            start_time as timestamp,
            open,
            high,
            low,
            close
        FROM ig_candles
        WHERE epic = :epic
          AND timeframe = :timeframe
          AND start_time >= :start_time
          AND start_time <= :end_time
        ORDER BY start_time ASC
        """

        try:
            df = self.db_manager.execute_query(query, {
                'epic': epic,
                'timeframe': timeframe,
                'start_time': start_time,
                'end_time': end_time
            })
            self.logger.debug(f"Fetched {len(df)} {timeframe}m candles for {epic}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch {timeframe}m candle data for {epic}: {e}")
            return pd.DataFrame()

    def calculate_entry_exit_prices(
        self,
        direction: str,
        current_price: float,
        bid_price: Optional[float],
        ask_price: Optional[float],
        pip_value: float
    ) -> Tuple[float, float, float]:
        """
        Calculate entry, SL, and TP prices based on direction.

        For BUY: Entry = ask (or current + spread/2), SL = entry - 9 pips, TP = entry + 15 pips
        For SELL: Entry = bid (or current - spread/2), SL = entry + 9 pips, TP = entry - 15 pips

        Args:
            direction: 'BULL' or 'BEAR'
            current_price: Current market price
            bid_price: Bid price (may be None)
            ask_price: Ask price (may be None)
            pip_value: Value of one pip for this pair

        Returns:
            Tuple of (entry_price, stop_loss_price, take_profit_price)
        """
        sl_distance = self.fixed_sl_pips * pip_value
        tp_distance = self.fixed_tp_pips * pip_value

        if direction == 'BULL':
            # BUY: Enter at ask, exit at bid
            if ask_price and ask_price > 0:
                entry_price = ask_price
            else:
                # Estimate ask from current price (add typical half-spread)
                entry_price = current_price + (0.5 * pip_value)

            stop_loss_price = entry_price - sl_distance
            take_profit_price = entry_price + tp_distance

        else:  # BEAR
            # SELL: Enter at bid, exit at ask
            if bid_price and bid_price > 0:
                entry_price = bid_price
            else:
                # Estimate bid from current price (subtract typical half-spread)
                entry_price = current_price - (0.5 * pip_value)

            stop_loss_price = entry_price + sl_distance
            take_profit_price = entry_price - tp_distance

        return entry_price, stop_loss_price, take_profit_price

    def determine_outcome(
        self,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        candles: pd.DataFrame,
        pip_value: float,
        rejection_timestamp: datetime,
        min_candles_required: int,
        max_analysis_hours: int
    ) -> Dict[str, Any]:
        """
        Determine if price would have hit TP or SL first.

        For BUY trades:
        - TP hit when HIGH >= take_profit_price
        - SL hit when LOW <= stop_loss_price

        For SELL trades:
        - TP hit when LOW <= take_profit_price
        - SL hit when HIGH >= stop_loss_price

        Also calculates MFE/MAE during the analysis window.

        Args:
            direction: 'BULL' or 'BEAR'
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            candles: DataFrame with OHLC data
            pip_value: Value of one pip
            rejection_timestamp: When the rejection occurred

        Returns:
            Dictionary with outcome details
        """
        result = {
            'outcome': OutcomeType.INSUFFICIENT_DATA,
            'outcome_price': None,
            'outcome_timestamp': None,
            'time_to_outcome_minutes': None,
            'max_favorable_excursion_pips': 0.0,
            'max_adverse_excursion_pips': 0.0,
            'mfe_timestamp': None,
            'mae_timestamp': None,
            'time_to_mfe_minutes': None,
            'time_to_mae_minutes': None,
            'candle_count_analyzed': len(candles),
            'data_quality_score': 0.0,
            'analysis_notes': None
        }

        if candles.empty or len(candles) < min_candles_required:
            result['analysis_notes'] = f"Insufficient data: {len(candles)} candles (min: {min_candles_required})"
            return result

        # Calculate data quality score based on candle count
        # Determine candle size from timeframe (5m for swing, 1m for scalp)
        candle_size_minutes = 5 if min_candles_required <= 12 else 1
        expected_candles = (max_analysis_hours * 60) / candle_size_minutes
        result['data_quality_score'] = min(1.0, len(candles) / expected_candles)

        # Track MFE and MAE
        mfe = 0.0  # Maximum Favorable Excursion (in pips)
        mae = 0.0  # Maximum Adverse Excursion (in pips)
        mfe_time = None
        mae_time = None

        # Iterate through candles to find outcome
        for idx, candle in candles.iterrows():
            candle_time = candle['timestamp']
            # Ensure candle_time is timezone-aware for comparison
            if hasattr(candle_time, 'tzinfo') and candle_time.tzinfo is None:
                candle_time = candle_time.replace(tzinfo=timezone.utc)
            high = float(candle['high'])
            low = float(candle['low'])

            if direction == 'BULL':
                # For BUY: favorable = price going up, adverse = price going down
                favorable_excursion = (high - entry_price) / pip_value
                adverse_excursion = (entry_price - low) / pip_value

                # Check if TP hit (high reaches TP)
                tp_hit = high >= take_profit_price
                # Check if SL hit (low reaches SL)
                sl_hit = low <= stop_loss_price

            else:  # BEAR
                # For SELL: favorable = price going down, adverse = price going up
                favorable_excursion = (entry_price - low) / pip_value
                adverse_excursion = (high - entry_price) / pip_value

                # Check if TP hit (low reaches TP)
                tp_hit = low <= take_profit_price
                # Check if SL hit (high reaches SL)
                sl_hit = high >= stop_loss_price

            # Update MFE/MAE
            if favorable_excursion > mfe:
                mfe = favorable_excursion
                mfe_time = candle_time

            if adverse_excursion > mae:
                mae = adverse_excursion
                mae_time = candle_time

            # Determine outcome (first one to hit wins)
            if tp_hit and sl_hit:
                # Both hit in same candle - check which would hit first
                # Use a simple heuristic: if open is closer to TP, TP hit first
                candle_open = float(candle['open'])

                if direction == 'BULL':
                    tp_distance_from_open = abs(take_profit_price - candle_open)
                    sl_distance_from_open = abs(stop_loss_price - candle_open)
                else:
                    tp_distance_from_open = abs(candle_open - take_profit_price)
                    sl_distance_from_open = abs(candle_open - stop_loss_price)

                if tp_distance_from_open <= sl_distance_from_open:
                    result['outcome'] = OutcomeType.HIT_TP
                    result['outcome_price'] = take_profit_price
                else:
                    result['outcome'] = OutcomeType.HIT_SL
                    result['outcome_price'] = stop_loss_price

                result['outcome_timestamp'] = candle_time
                break

            elif tp_hit:
                result['outcome'] = OutcomeType.HIT_TP
                result['outcome_price'] = take_profit_price
                result['outcome_timestamp'] = candle_time
                break

            elif sl_hit:
                result['outcome'] = OutcomeType.HIT_SL
                result['outcome_price'] = stop_loss_price
                result['outcome_timestamp'] = candle_time
                break

        # If neither hit, mark as STILL_OPEN
        if result['outcome'] == OutcomeType.INSUFFICIENT_DATA and len(candles) >= min_candles_required:
            result['outcome'] = OutcomeType.STILL_OPEN
            result['analysis_notes'] = f"Neither TP nor SL hit within {len(candles)} candles"

        # Calculate time to outcome
        if result['outcome_timestamp']:
            time_diff = result['outcome_timestamp'] - rejection_timestamp
            if hasattr(time_diff, 'total_seconds'):
                result['time_to_outcome_minutes'] = int(time_diff.total_seconds() / 60)

        # Store MFE/MAE
        result['max_favorable_excursion_pips'] = round(mfe, 2)
        result['max_adverse_excursion_pips'] = round(mae, 2)
        result['mfe_timestamp'] = mfe_time
        result['mae_timestamp'] = mae_time

        # Calculate time to MFE/MAE
        if mfe_time:
            time_diff = mfe_time - rejection_timestamp
            if hasattr(time_diff, 'total_seconds'):
                result['time_to_mfe_minutes'] = int(time_diff.total_seconds() / 60)

        if mae_time:
            time_diff = mae_time - rejection_timestamp
            if hasattr(time_diff, 'total_seconds'):
                result['time_to_mae_minutes'] = int(time_diff.total_seconds() / 60)

        return result

    def analyze_rejection(self, rejection: pd.Series) -> Optional[RejectionOutcome]:
        """
        Analyze a single rejection and determine its hypothetical outcome.

        Args:
            rejection: Row from smc_simple_rejections table

        Returns:
            RejectionOutcome with analysis results, or None if insufficient data
        """
        rejection_id = int(rejection['rejection_id'])
        epic = rejection['epic']
        pair = rejection['pair']
        rejection_timestamp = rejection['rejection_timestamp']
        direction = rejection['attempted_direction']

        # Handle pandas Timestamp or datetime
        if hasattr(rejection_timestamp, 'to_pydatetime'):
            rejection_timestamp = rejection_timestamp.to_pydatetime()

        # Ensure rejection_timestamp is timezone-aware
        if rejection_timestamp.tzinfo is None:
            rejection_timestamp = rejection_timestamp.replace(tzinfo=timezone.utc)

        # Get pip value for this pair
        pip_value = self.get_pip_value(epic)

        # Calculate entry/SL/TP prices
        current_price = float(rejection['current_price']) if rejection['current_price'] else 0
        bid_price = float(rejection['bid_price']) if rejection['bid_price'] else None
        ask_price = float(rejection['ask_price']) if rejection['ask_price'] else None

        if current_price == 0:
            self.logger.warning(f"Rejection {rejection_id}: No price data available")
            return None

        entry_price, stop_loss_price, take_profit_price = self.calculate_entry_exit_prices(
            direction, current_price, bid_price, ask_price, pip_value
        )

        # Determine if this is a scalp rejection (needs 1m candles)
        rejection_stage = rejection['rejection_stage']
        is_scalp = self.is_scalp_rejection(rejection_stage)

        # Set parameters based on rejection type
        if is_scalp:
            timeframe = 1  # 1-minute candles for scalp
            max_hours = self.max_analysis_hours_scalp
            min_candles = self.min_candles_required_1m
            self.logger.debug(f"Scalp rejection detected: using 1m candles, {max_hours}h window")
        else:
            timeframe = 5  # 5-minute candles for swing
            max_hours = self.max_analysis_hours
            min_candles = self.min_candles_required_5m
            self.logger.debug(f"Swing rejection detected: using 5m candles, {max_hours}h window")

        # Fetch candle data for analysis window
        analysis_start = rejection_timestamp
        analysis_end = rejection_timestamp + timedelta(hours=max_hours)

        candles = self.fetch_price_data(epic, analysis_start, analysis_end, timeframe)

        # Determine outcome
        outcome_data = self.determine_outcome(
            direction, entry_price, stop_loss_price, take_profit_price,
            candles, pip_value, rejection_timestamp, min_candles, max_hours
        )

        # Calculate potential profit/loss in pips
        if outcome_data['outcome'] == OutcomeType.HIT_TP:
            potential_profit_pips = self.fixed_tp_pips
        elif outcome_data['outcome'] == OutcomeType.HIT_SL:
            potential_profit_pips = -self.fixed_sl_pips
        else:
            potential_profit_pips = 0.0

        # Create RejectionOutcome object
        return RejectionOutcome(
            rejection_id=rejection_id,
            epic=epic,
            pair=pair,
            rejection_timestamp=rejection_timestamp,
            rejection_stage=rejection['rejection_stage'],
            attempted_direction=direction,
            market_session=rejection.get('market_session'),
            market_hour=_safe_int(rejection.get('market_hour')),
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            spread_at_rejection=float(rejection['spread_pips']) if rejection.get('spread_pips') else None,
            outcome=outcome_data['outcome'],
            outcome_price=outcome_data['outcome_price'],
            outcome_timestamp=outcome_data['outcome_timestamp'],
            time_to_outcome_minutes=outcome_data['time_to_outcome_minutes'],
            max_favorable_excursion_pips=outcome_data['max_favorable_excursion_pips'],
            max_adverse_excursion_pips=outcome_data['max_adverse_excursion_pips'],
            mfe_timestamp=outcome_data['mfe_timestamp'],
            mae_timestamp=outcome_data['mae_timestamp'],
            time_to_mfe_minutes=outcome_data['time_to_mfe_minutes'],
            time_to_mae_minutes=outcome_data['time_to_mae_minutes'],
            potential_profit_pips=potential_profit_pips,
            candle_count_analyzed=outcome_data['candle_count_analyzed'],
            data_quality_score=outcome_data['data_quality_score'],
            analysis_notes=outcome_data['analysis_notes']
        )

    def save_outcome(self, outcome: RejectionOutcome) -> bool:
        """
        Save analysis outcome to database.

        Args:
            outcome: RejectionOutcome to save

        Returns:
            True if saved successfully
        """
        query = """
        INSERT INTO smc_rejection_outcomes (
            rejection_id, epic, pair, rejection_timestamp,
            rejection_stage, attempted_direction, market_session, market_hour,
            entry_price, stop_loss_price, take_profit_price, spread_at_rejection,
            outcome, outcome_price, outcome_timestamp, time_to_outcome_minutes,
            max_favorable_excursion_pips, max_adverse_excursion_pips,
            mfe_timestamp, mae_timestamp, time_to_mfe_minutes, time_to_mae_minutes,
            potential_profit_pips, candle_count_analyzed, data_quality_score,
            analysis_notes, fixed_sl_pips, fixed_tp_pips
        ) VALUES (
            :rejection_id, :epic, :pair, :rejection_timestamp,
            :rejection_stage, :attempted_direction, :market_session, :market_hour,
            :entry_price, :stop_loss_price, :take_profit_price, :spread_at_rejection,
            :outcome, :outcome_price, :outcome_timestamp, :time_to_outcome_minutes,
            :max_favorable_excursion_pips, :max_adverse_excursion_pips,
            :mfe_timestamp, :mae_timestamp, :time_to_mfe_minutes, :time_to_mae_minutes,
            :potential_profit_pips, :candle_count_analyzed, :data_quality_score,
            :analysis_notes, :fixed_sl_pips, :fixed_tp_pips
        )
        """

        try:
            self.db_manager.execute_query(query, {
                'rejection_id': outcome.rejection_id,
                'epic': outcome.epic,
                'pair': outcome.pair,
                'rejection_timestamp': outcome.rejection_timestamp,
                'rejection_stage': outcome.rejection_stage,
                'attempted_direction': outcome.attempted_direction,
                'market_session': outcome.market_session,
                'market_hour': outcome.market_hour,
                'entry_price': outcome.entry_price,
                'stop_loss_price': outcome.stop_loss_price,
                'take_profit_price': outcome.take_profit_price,
                'spread_at_rejection': outcome.spread_at_rejection,
                'outcome': outcome.outcome.value,
                'outcome_price': outcome.outcome_price,
                'outcome_timestamp': outcome.outcome_timestamp,
                'time_to_outcome_minutes': outcome.time_to_outcome_minutes,
                'max_favorable_excursion_pips': outcome.max_favorable_excursion_pips,
                'max_adverse_excursion_pips': outcome.max_adverse_excursion_pips,
                'mfe_timestamp': outcome.mfe_timestamp,
                'mae_timestamp': outcome.mae_timestamp,
                'time_to_mfe_minutes': outcome.time_to_mfe_minutes,
                'time_to_mae_minutes': outcome.time_to_mae_minutes,
                'potential_profit_pips': outcome.potential_profit_pips,
                'candle_count_analyzed': outcome.candle_count_analyzed,
                'data_quality_score': outcome.data_quality_score,
                'analysis_notes': outcome.analysis_notes,
                'fixed_sl_pips': self.fixed_sl_pips,
                'fixed_tp_pips': self.fixed_tp_pips
            })
            return True
        except Exception as e:
            self.logger.error(f"Failed to save outcome for rejection {outcome.rejection_id}: {e}")
            return False

    def run_daily_analysis(self, days_back: int = 1, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run daily batch analysis of rejections.

        Args:
            days_back: Number of days to analyze (default 1 = yesterday)
            dry_run: If True, analyze but don't save results

        Returns:
            Summary statistics of the analysis run
        """
        # Calculate date range - analyze rejections from N days ago
        # We need rejections old enough that we have price data to analyze outcomes
        end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days_back)

        # But we need the rejection to be at least max_analysis_hours old
        # to have complete outcome data
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.max_analysis_hours)
        if end_date > cutoff_time:
            end_date = cutoff_time

        self.logger.info(f"=" * 60)
        self.logger.info(f"SMC Rejection Outcome Analysis")
        self.logger.info(f"=" * 60)
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"SL: {self.fixed_sl_pips} pips, TP: {self.fixed_tp_pips} pips")
        self.logger.info(f"Dry run: {dry_run}")
        self.logger.info(f"=" * 60)

        # Fetch unanalyzed rejections
        rejections = self.get_unanalyzed_rejections(start_date, end_date)

        if rejections.empty:
            self.logger.info("No unanalyzed rejections found in date range")
            return {
                'total_analyzed': 0,
                'winners': 0,
                'losers': 0,
                'still_open': 0,
                'insufficient_data': 0,
                'errors': 0
            }

        # Track statistics
        stats = {
            'total_analyzed': 0,
            'winners': 0,
            'losers': 0,
            'still_open': 0,
            'insufficient_data': 0,
            'errors': 0,
            'by_stage': {},
            'by_pair': {}
        }

        # Analyze each rejection
        for idx, rejection in rejections.iterrows():
            try:
                outcome = self.analyze_rejection(rejection)

                if outcome is None:
                    stats['errors'] += 1
                    continue

                stats['total_analyzed'] += 1

                # Track outcome type
                if outcome.outcome == OutcomeType.HIT_TP:
                    stats['winners'] += 1
                elif outcome.outcome == OutcomeType.HIT_SL:
                    stats['losers'] += 1
                elif outcome.outcome == OutcomeType.STILL_OPEN:
                    stats['still_open'] += 1
                else:
                    stats['insufficient_data'] += 1

                # Track by stage
                stage = outcome.rejection_stage
                if stage not in stats['by_stage']:
                    stats['by_stage'][stage] = {'winners': 0, 'losers': 0, 'other': 0}

                if outcome.outcome == OutcomeType.HIT_TP:
                    stats['by_stage'][stage]['winners'] += 1
                elif outcome.outcome == OutcomeType.HIT_SL:
                    stats['by_stage'][stage]['losers'] += 1
                else:
                    stats['by_stage'][stage]['other'] += 1

                # Track by pair
                pair = outcome.pair
                if pair not in stats['by_pair']:
                    stats['by_pair'][pair] = {'winners': 0, 'losers': 0, 'other': 0}

                if outcome.outcome == OutcomeType.HIT_TP:
                    stats['by_pair'][pair]['winners'] += 1
                elif outcome.outcome == OutcomeType.HIT_SL:
                    stats['by_pair'][pair]['losers'] += 1
                else:
                    stats['by_pair'][pair]['other'] += 1

                # Save outcome (unless dry run)
                if not dry_run:
                    self.save_outcome(outcome)

                # Progress logging
                if stats['total_analyzed'] % 50 == 0:
                    self.logger.info(f"Processed {stats['total_analyzed']} rejections...")

            except Exception as e:
                self.logger.error(f"Error analyzing rejection {rejection.get('rejection_id')}: {e}")
                stats['errors'] += 1

        # Log summary
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"ANALYSIS COMPLETE")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Total analyzed: {stats['total_analyzed']}")
        self.logger.info(f"Would-be winners: {stats['winners']}")
        self.logger.info(f"Would-be losers: {stats['losers']}")
        self.logger.info(f"Still open: {stats['still_open']}")
        self.logger.info(f"Insufficient data: {stats['insufficient_data']}")
        self.logger.info(f"Errors: {stats['errors']}")

        # Calculate win rate
        decided = stats['winners'] + stats['losers']
        if decided > 0:
            win_rate = (stats['winners'] / decided) * 100
            self.logger.info(f"Would-be win rate: {win_rate:.1f}%")
            stats['would_be_win_rate'] = win_rate

        # Log by stage
        if stats['by_stage']:
            self.logger.info(f"\nBy Rejection Stage:")
            for stage, data in sorted(stats['by_stage'].items()):
                stage_decided = data['winners'] + data['losers']
                if stage_decided > 0:
                    stage_wr = (data['winners'] / stage_decided) * 100
                    self.logger.info(f"  {stage}: {data['winners']}W / {data['losers']}L ({stage_wr:.1f}% WR)")

        return stats

    def generate_summary_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate summary report for AI parameter suggestions.

        Args:
            days: Number of days to include in report

        Returns:
            Dictionary with aggregated statistics suitable for AI analysis
        """
        query = """
        SELECT
            rejection_stage,
            COUNT(*) as total,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as losers,
            ROUND(AVG(max_favorable_excursion_pips)::numeric, 2) as avg_mfe,
            ROUND(AVG(max_adverse_excursion_pips)::numeric, 2) as avg_mae,
            ROUND(AVG(time_to_outcome_minutes)::numeric, 0) as avg_time_mins
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL ':days days'
          AND outcome IN ('HIT_TP', 'HIT_SL')
        GROUP BY rejection_stage
        ORDER BY total DESC
        """

        try:
            df = self.db_manager.execute_query(query.replace(':days', str(days)), {})
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            return []


def main():
    """CLI entry point for running the analyzer"""
    parser = argparse.ArgumentParser(
        description='Analyze SMC rejection outcomes to determine if rejected signals would have been profitable'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days to analyze (default: 1 = yesterday)'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Specific date to analyze (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze but do not save results to database'
    )
    args = parser.parse_args()

    try:
        analyzer = RejectionOutcomeAnalyzer()

        if args.date:
            # Analyze specific date - would need custom logic
            logger.info(f"Analyzing specific date: {args.date}")
            # For now, just use days_back approach
            target_date = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            days_ago = (datetime.now(timezone.utc) - target_date).days
            result = analyzer.run_daily_analysis(days_back=days_ago, dry_run=args.dry_run)
        else:
            result = analyzer.run_daily_analysis(days_back=args.days, dry_run=args.dry_run)

        # Print final summary
        print(f"\n{'=' * 60}")
        print(f"REJECTION OUTCOME ANALYSIS COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total analyzed: {result.get('total_analyzed', 0)}")
        print(f"Would-be winners: {result.get('winners', 0)}")
        print(f"Would-be losers: {result.get('losers', 0)}")

        if result.get('would_be_win_rate'):
            print(f"Would-be win rate: {result['would_be_win_rate']:.1f}%")

        print(f"Still open: {result.get('still_open', 0)}")
        print(f"Errors: {result.get('errors', 0)}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
