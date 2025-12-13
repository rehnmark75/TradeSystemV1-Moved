"""
Crossover History Tracker
=========================
In-memory state tracking for EMA crossover success patterns.

This module tracks EMA 21/50 crossovers and validates their "success" by
checking if price stays on the favorable side of EMA 21 for a configurable
number of candles.

Pattern follows AlertDeduplicationManager for state management architecture.

Usage:
    tracker = CrossoverHistoryTracker(success_candles=4, lookback_hours=48)

    # On each candle:
    tracker.validate_pending_crossovers(df, epic)
    crossover = tracker.detect_crossover(df, epic)
    if crossover:
        should_signal, confidence_boost, metadata = tracker.should_generate_signal(
            epic, crossover.direction, crossover.timestamp
        )
        if should_signal:
            # Generate entry signal
            tracker.record_trade_taken(epic, crossover.direction, crossover.timestamp)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging


@dataclass
class CrossoverEvent:
    """Represents a single EMA crossover event"""
    timestamp: datetime
    direction: str  # 'BULL' or 'BEAR'
    price_at_crossover: float
    ema_fast: float
    ema_slow: float
    candle_index: int = -1  # Index in DataFrame when detected
    success: Optional[bool] = None  # None = pending validation, True/False = validated
    validation_timestamp: Optional[datetime] = None
    candles_validated: int = 0  # How many candles have been checked
    candles_held: int = 0  # How many candles price stayed favorable

    def is_pending(self) -> bool:
        """Check if this crossover is still pending validation"""
        return self.success is None

    def __repr__(self) -> str:
        status = "PENDING" if self.success is None else ("SUCCESS" if self.success else "FAILED")
        return f"CrossoverEvent({self.direction}, {self.timestamp}, {status}, held={self.candles_held})"


@dataclass
class PairCrossoverState:
    """Tracks crossover state for a single currency pair"""
    epic: str
    successful_bull_crossovers: List[CrossoverEvent] = field(default_factory=list)
    successful_bear_crossovers: List[CrossoverEvent] = field(default_factory=list)
    pending_bull_crossover: Optional[CrossoverEvent] = None
    pending_bear_crossover: Optional[CrossoverEvent] = None
    last_trade_direction: Optional[str] = None
    last_trade_timestamp: Optional[datetime] = None

    def get_successful_count(self, direction: str) -> int:
        """Get count of successful crossovers for a direction"""
        if direction == 'BULL':
            return len(self.successful_bull_crossovers)
        elif direction == 'BEAR':
            return len(self.successful_bear_crossovers)
        return 0

    def get_successful_crossovers(self, direction: str) -> List[CrossoverEvent]:
        """Get list of successful crossovers for a direction"""
        if direction == 'BULL':
            return self.successful_bull_crossovers
        elif direction == 'BEAR':
            return self.successful_bear_crossovers
        return []

    def add_successful_crossover(self, event: CrossoverEvent):
        """Add a validated successful crossover"""
        if event.direction == 'BULL':
            self.successful_bull_crossovers.append(event)
        elif event.direction == 'BEAR':
            self.successful_bear_crossovers.append(event)

    def clear_successful_crossovers(self, direction: str):
        """Clear successful crossovers for a direction (after trade taken)"""
        if direction == 'BULL':
            self.successful_bull_crossovers = []
        elif direction == 'BEAR':
            self.successful_bear_crossovers = []


class CrossoverHistoryTracker:
    """
    Tracks EMA 21/50 crossover history and success patterns.

    Pattern Based On: AlertDeduplicationManager (in-memory state with dict)

    Success Definition:
    - Bullish: Close stays above EMA 21 for `success_candles` consecutive candles
    - Bearish: Close stays below EMA 21 for `success_candles` consecutive candles

    Entry Signal Logic:
    - After 2 successful crossovers in SAME direction within lookback window
    - Take the 3rd crossover as entry signal
    - Reset counter for that direction after trade is taken
    """

    def __init__(
        self,
        success_candles: int = 4,
        lookback_hours: int = 48,
        min_crossovers_for_signal: int = 2,
        max_validation_candles: int = 6,
        extended_success_candles: int = 6,
        extended_success_bonus: float = 0.05,
        ema_fast_period: int = 21,
        ema_slow_period: int = 50,
        logger: logging.Logger = None
    ):
        """
        Initialize the crossover history tracker.

        Args:
            success_candles: Number of candles price must stay favorable to validate success
            lookback_hours: Hours to look back for prior successful crossovers
            min_crossovers_for_signal: Successful crossovers needed before entry (default 2)
            max_validation_candles: Maximum candles to wait before marking crossover failed
            extended_success_candles: Candles for extended success bonus
            extended_success_bonus: Confidence bonus for extended success
            ema_fast_period: Fast EMA period (for column naming)
            ema_slow_period: Slow EMA period (for column naming)
            logger: Optional logger instance
        """
        self.success_candles = success_candles
        self.lookback_hours = lookback_hours
        self.min_crossovers_for_signal = min_crossovers_for_signal
        self.max_validation_candles = max_validation_candles
        self.extended_success_candles = extended_success_candles
        self.extended_success_bonus = extended_success_bonus
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period

        self.logger = logger or logging.getLogger(__name__)

        # In-memory state storage: {epic: PairCrossoverState}
        self._pair_states: Dict[str, PairCrossoverState] = {}

        # Column names for EMAs
        self._ema_fast_col = f'ema_{ema_fast_period}'
        self._ema_slow_col = f'ema_{ema_slow_period}'

        self.logger.info("=" * 60)
        self.logger.info("CrossoverHistoryTracker initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"  EMAs: {ema_fast_period}/{ema_slow_period}")
        self.logger.info(f"  Success validation: {success_candles} candles")
        self.logger.info(f"  Lookback window: {lookback_hours} hours")
        self.logger.info(f"  Required crossovers for signal: {min_crossovers_for_signal}")
        self.logger.info("=" * 60)

    def _get_or_create_state(self, epic: str) -> PairCrossoverState:
        """Get or create state for a currency pair"""
        if epic not in self._pair_states:
            self._pair_states[epic] = PairCrossoverState(epic=epic)
        return self._pair_states[epic]

    def _get_ema_values(self, row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """Extract EMA values from a DataFrame row"""
        # Try specific column names first
        ema_fast = row.get(self._ema_fast_col)
        ema_slow = row.get(self._ema_slow_col)

        # Fallback to generic names
        if ema_fast is None:
            ema_fast = row.get('ema_fast') or row.get('ema_21') or row.get('ema_short')
        if ema_slow is None:
            ema_slow = row.get('ema_slow') or row.get('ema_50') or row.get('ema_long')

        return ema_fast, ema_slow

    def detect_crossover(
        self,
        df: pd.DataFrame,
        epic: str
    ) -> Optional[CrossoverEvent]:
        """
        Detect if a new EMA crossover occurred on the latest candle.

        A bullish crossover: EMA fast crosses ABOVE EMA slow
        A bearish crossover: EMA fast crosses BELOW EMA slow

        Args:
            df: DataFrame with OHLC and EMA data
            epic: Currency pair identifier

        Returns:
            CrossoverEvent if new crossover detected, None otherwise
        """
        if df is None or len(df) < 2:
            return None

        try:
            # Get current and previous candle
            curr = df.iloc[-1]
            prev = df.iloc[-2]

            # Get EMA values
            curr_ema_fast, curr_ema_slow = self._get_ema_values(curr)
            prev_ema_fast, prev_ema_slow = self._get_ema_values(prev)

            # Validate we have all required values
            if any(v is None or pd.isna(v) for v in [curr_ema_fast, curr_ema_slow, prev_ema_fast, prev_ema_slow]):
                return None

            # Get timestamp
            timestamp = curr.get('start_time') or curr.name
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            elif not isinstance(timestamp, (datetime, pd.Timestamp)):
                timestamp = datetime.now()

            # Detect crossover direction
            direction = None

            # Bullish crossover: was below, now above
            if prev_ema_fast <= prev_ema_slow and curr_ema_fast > curr_ema_slow:
                direction = 'BULL'
            # Bearish crossover: was above, now below
            elif prev_ema_fast >= prev_ema_slow and curr_ema_fast < curr_ema_slow:
                direction = 'BEAR'

            if direction is None:
                return None

            # Create crossover event
            event = CrossoverEvent(
                timestamp=timestamp,
                direction=direction,
                price_at_crossover=float(curr.get('close', 0)),
                ema_fast=float(curr_ema_fast),
                ema_slow=float(curr_ema_slow),
                candle_index=len(df) - 1
            )

            # Store as pending for validation
            state = self._get_or_create_state(epic)
            if direction == 'BULL':
                state.pending_bull_crossover = event
            else:
                state.pending_bear_crossover = event

            self.logger.info(f"[{epic}] New {direction} crossover detected at {timestamp}")
            self.logger.debug(f"  EMA Fast: {curr_ema_fast:.5f}, EMA Slow: {curr_ema_slow:.5f}")
            self.logger.debug(f"  Price: {event.price_at_crossover:.5f}")

            return event

        except Exception as e:
            self.logger.error(f"[{epic}] Error detecting crossover: {e}")
            return None

    def validate_pending_crossovers(
        self,
        df: pd.DataFrame,
        epic: str
    ) -> List[CrossoverEvent]:
        """
        Validate any pending crossovers by checking if price stayed favorable.

        Success criteria:
        - Bullish: Close stayed above EMA fast for success_candles consecutive candles
        - Bearish: Close stayed below EMA fast for success_candles consecutive candles

        Args:
            df: DataFrame with recent OHLC and EMA data
            epic: Currency pair identifier

        Returns:
            List of newly validated successful crossovers
        """
        if df is None or len(df) < 2:
            return []

        state = self._get_or_create_state(epic)
        validated = []

        # Validate bullish pending crossover
        if state.pending_bull_crossover and state.pending_bull_crossover.is_pending():
            result = self._validate_single_crossover(df, state.pending_bull_crossover, epic)
            if result is not None:
                if result:
                    validated.append(state.pending_bull_crossover)
                    state.add_successful_crossover(state.pending_bull_crossover)
                    self.logger.info(f"[{epic}] BULL crossover VALIDATED as SUCCESS (held {state.pending_bull_crossover.candles_held} candles)")
                else:
                    self.logger.info(f"[{epic}] BULL crossover FAILED validation")
                state.pending_bull_crossover = None

        # Validate bearish pending crossover
        if state.pending_bear_crossover and state.pending_bear_crossover.is_pending():
            result = self._validate_single_crossover(df, state.pending_bear_crossover, epic)
            if result is not None:
                if result:
                    validated.append(state.pending_bear_crossover)
                    state.add_successful_crossover(state.pending_bear_crossover)
                    self.logger.info(f"[{epic}] BEAR crossover VALIDATED as SUCCESS (held {state.pending_bear_crossover.candles_held} candles)")
                else:
                    self.logger.info(f"[{epic}] BEAR crossover FAILED validation")
                state.pending_bear_crossover = None

        # Cleanup expired crossovers using the SIMULATION time (latest candle time)
        # This is critical for backtesting to work correctly
        current_simulation_time = self._get_latest_timestamp(df)
        self._cleanup_expired_crossovers(epic, current_simulation_time)

        return validated

    def _get_latest_timestamp(self, df: pd.DataFrame) -> datetime:
        """Extract the latest timestamp from the DataFrame for simulation time"""
        try:
            latest_row = df.iloc[-1]
            timestamp = latest_row.get('start_time') or latest_row.name

            if isinstance(timestamp, str):
                return pd.to_datetime(timestamp)
            elif isinstance(timestamp, (datetime, pd.Timestamp)):
                return timestamp
        except Exception:
            pass

        # Fallback to real time if can't extract
        try:
            from datetime import timezone
            return datetime.now(timezone.utc)
        except ImportError:
            return datetime.utcnow()

    def _validate_single_crossover(
        self,
        df: pd.DataFrame,
        event: CrossoverEvent,
        epic: str
    ) -> Optional[bool]:
        """
        Validate a single pending crossover.

        Returns:
            True if validated successful, False if failed, None if still pending
        """
        try:
            # Find candles since crossover
            candles_since = len(df) - 1 - event.candle_index

            if candles_since < 1:
                return None  # Not enough candles yet

            # Check each candle since crossover
            favorable_count = 0

            for i in range(event.candle_index + 1, len(df)):
                row = df.iloc[i]
                close = row.get('close', 0)
                ema_fast, _ = self._get_ema_values(row)

                if ema_fast is None or pd.isna(ema_fast):
                    continue

                # Check if price stayed favorable
                is_favorable = False
                if event.direction == 'BULL':
                    is_favorable = close > ema_fast
                else:  # BEAR
                    is_favorable = close < ema_fast

                if is_favorable:
                    favorable_count += 1
                else:
                    # Price crossed back - crossover failed
                    event.success = False
                    event.candles_validated = candles_since
                    event.validation_timestamp = datetime.now()
                    return False

            event.candles_validated = candles_since
            event.candles_held = favorable_count

            # Check if we have enough favorable candles
            if favorable_count >= self.success_candles:
                event.success = True
                event.validation_timestamp = datetime.now()
                return True

            # Check if we've waited too long
            if candles_since >= self.max_validation_candles:
                event.success = False
                event.validation_timestamp = datetime.now()
                return False

            # Still pending
            return None

        except Exception as e:
            self.logger.error(f"[{epic}] Error validating crossover: {e}")
            return None

    def should_generate_signal(
        self,
        epic: str,
        current_direction: str,
        current_timestamp: datetime
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Determine if we should generate an entry signal.

        Signal is generated when:
        1. We have at least `min_crossovers_for_signal` successful crossovers
        2. All successful crossovers are in the SAME direction as current
        3. All successful crossovers are within the lookback window

        Args:
            epic: Currency pair identifier
            current_direction: Direction of the current crossover ('BULL' or 'BEAR')
            current_timestamp: Timestamp of the current crossover

        Returns:
            Tuple of:
            - should_signal: bool - True if entry signal should be generated
            - confidence_boost: float - Additional confidence from prior success
            - metadata: dict - Information about the successful crossovers
        """
        state = self._get_or_create_state(epic)

        # Get successful crossovers for this direction
        successful = state.get_successful_crossovers(current_direction)

        # Filter to those within lookback window (use helper for timezone handling)
        cutoff_time = current_timestamp - timedelta(hours=self.lookback_hours)
        valid_crossovers = [
            c for c in successful
            if self._is_timestamp_valid(c.timestamp, cutoff_time)
        ]

        # Build metadata
        metadata = {
            'successful_count': len(valid_crossovers),
            'required_count': self.min_crossovers_for_signal,
            'direction': current_direction,
            'lookback_hours': self.lookback_hours,
            'crossover_history': [
                {
                    'timestamp': c.timestamp.isoformat() if hasattr(c.timestamp, 'isoformat') else str(c.timestamp),
                    'candles_held': c.candles_held,
                    'price': c.price_at_crossover
                }
                for c in valid_crossovers[-5:]  # Last 5 for logging
            ]
        }

        # Check if we have enough successful crossovers
        if len(valid_crossovers) < self.min_crossovers_for_signal:
            self.logger.debug(f"[{epic}] Not enough successful {current_direction} crossovers: "
                            f"{len(valid_crossovers)}/{self.min_crossovers_for_signal}")
            return False, 0.0, metadata

        # Calculate confidence boost based on how well prior crossovers performed
        confidence_boost = 0.0

        for crossover in valid_crossovers:
            # Bonus for extended success (held longer than required)
            if crossover.candles_held >= self.extended_success_candles:
                confidence_boost += self.extended_success_bonus

        # Cap confidence boost
        confidence_boost = min(confidence_boost, 0.15)

        self.logger.info(f"[{epic}] Signal conditions MET for {current_direction}!")
        self.logger.info(f"  Prior successful crossovers: {len(valid_crossovers)}")
        self.logger.info(f"  Confidence boost: +{confidence_boost:.1%}")

        return True, confidence_boost, metadata

    def record_trade_taken(
        self,
        epic: str,
        direction: str,
        timestamp: datetime
    ):
        """
        Record that a trade was taken and reset the counter for that direction.

        Args:
            epic: Currency pair identifier
            direction: Direction of the trade ('BULL' or 'BEAR')
            timestamp: Timestamp of the trade
        """
        state = self._get_or_create_state(epic)

        # Record trade
        state.last_trade_direction = direction
        state.last_trade_timestamp = timestamp

        # Reset counter for this direction
        state.clear_successful_crossovers(direction)

        self.logger.info(f"[{epic}] Trade taken: {direction} at {timestamp}")
        self.logger.info(f"  Counter reset for {direction} direction")

    def _cleanup_expired_crossovers(self, epic: str, current_time: datetime = None):
        """Remove crossovers outside the lookback window

        Args:
            epic: Currency pair identifier
            current_time: The current simulation time (for backtesting). If None, uses real time.
        """
        state = self._get_or_create_state(epic)

        # Use provided current_time for backtesting, otherwise use real time
        if current_time is None:
            try:
                from datetime import timezone
                current_time = datetime.now(timezone.utc)
            except ImportError:
                current_time = datetime.utcnow()

        cutoff_time = current_time - timedelta(hours=self.lookback_hours)

        # Clean bull crossovers - handle both timezone-aware and naive timestamps
        state.successful_bull_crossovers = [
            c for c in state.successful_bull_crossovers
            if self._is_timestamp_valid(c.timestamp, cutoff_time)
        ]

        # Clean bear crossovers
        state.successful_bear_crossovers = [
            c for c in state.successful_bear_crossovers
            if self._is_timestamp_valid(c.timestamp, cutoff_time)
        ]

    def _is_timestamp_valid(self, ts: datetime, cutoff: datetime) -> bool:
        """Check if timestamp is after cutoff, handling timezone mismatches"""
        try:
            # If both are naive or both are aware, compare directly
            if (ts.tzinfo is None) == (cutoff.tzinfo is None):
                return ts >= cutoff

            # If timestamp is aware and cutoff is naive, make cutoff aware
            if ts.tzinfo is not None and cutoff.tzinfo is None:
                from datetime import timezone
                cutoff = cutoff.replace(tzinfo=timezone.utc)
                return ts >= cutoff

            # If timestamp is naive and cutoff is aware, make timestamp aware
            if ts.tzinfo is None and cutoff.tzinfo is not None:
                from datetime import timezone
                ts = ts.replace(tzinfo=timezone.utc)
                return ts >= cutoff

            return True  # Default to keeping the crossover
        except Exception:
            return True  # On error, keep the crossover

    def get_state_summary(self, epic: str = None) -> Dict[str, Any]:
        """
        Get summary of tracker state for debugging/monitoring.

        Args:
            epic: Optional specific pair to get state for. If None, returns all.

        Returns:
            Dictionary with state information
        """
        if epic:
            if epic not in self._pair_states:
                return {'epic': epic, 'status': 'no_data'}

            state = self._pair_states[epic]
            return {
                'epic': epic,
                'successful_bull_count': len(state.successful_bull_crossovers),
                'successful_bear_count': len(state.successful_bear_crossovers),
                'pending_bull': state.pending_bull_crossover is not None,
                'pending_bear': state.pending_bear_crossover is not None,
                'last_trade': {
                    'direction': state.last_trade_direction,
                    'timestamp': str(state.last_trade_timestamp) if state.last_trade_timestamp else None
                }
            }

        # Return all pairs
        return {
            epic: self.get_state_summary(epic)
            for epic in self._pair_states.keys()
        }

    def reset_state(self, epic: str = None):
        """
        Reset tracker state.

        Args:
            epic: Specific pair to reset. If None, resets all pairs.
        """
        if epic:
            if epic in self._pair_states:
                del self._pair_states[epic]
                self.logger.info(f"[{epic}] State reset")
        else:
            self._pair_states = {}
            self.logger.info("All states reset")
