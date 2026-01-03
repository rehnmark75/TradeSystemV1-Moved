"""
Crossover History Tracker
=========================
State tracking for EMA crossover success patterns with optional database persistence.

This module tracks EMA 21/50 crossovers and validates their "success" by
checking if price stays on the favorable side of EMA 21 for a configurable
number of candles.

Pattern follows AlertDeduplicationManager for state management architecture.

v2.0.0 CHANGES (Database Persistence):
    - NEW: Database persistence for crossover state (survives restarts)
    - NEW: USE_DATABASE_STATE config option to enable/disable persistence
    - NEW: Automatic table creation on initialization
    - NEW: Load state from database on startup
    - NEW: Save state to database on changes (successful crossovers, trades)
    - FIXED: "Chicken and egg" problem where MIN_SUCCESSFUL_CROSSOVERS=1
             would never trigger because state was lost on restart

Usage:
    # In-memory mode (original behavior):
    tracker = CrossoverHistoryTracker(success_candles=4, lookback_hours=48)

    # Database persistence mode:
    tracker = CrossoverHistoryTracker(
        success_candles=4,
        lookback_hours=48,
        db_manager=db_manager,
        use_database_state=True
    )

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
import json
from dataclasses import dataclass, field, asdict
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
    # Track the timestamp of the last seen crossover to avoid re-detecting
    last_bull_crossover_timestamp: Optional[datetime] = None
    last_bear_crossover_timestamp: Optional[datetime] = None

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
        logger: logging.Logger = None,
        db_manager=None,
        use_database_state: bool = False
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
            db_manager: Optional DatabaseManager for persistence
            use_database_state: Whether to persist state to database
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

        # Database persistence settings
        self.db_manager = db_manager
        self.use_database_state = use_database_state and db_manager is not None

        # In-memory state storage: {epic: PairCrossoverState}
        self._pair_states: Dict[str, PairCrossoverState] = {}

        # Column names for EMAs
        self._ema_fast_col = f'ema_{ema_fast_period}'
        self._ema_slow_col = f'ema_{ema_slow_period}'

        # Initialize database table if using persistence
        if self.use_database_state:
            self._initialize_database_table()
            self._load_all_states_from_db()

        self.logger.info("=" * 60)
        self.logger.info("CrossoverHistoryTracker initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"  EMAs: {ema_fast_period}/{ema_slow_period}")
        self.logger.info(f"  Success validation: {success_candles} candles")
        self.logger.info(f"  Lookback window: {lookback_hours} hours")
        self.logger.info(f"  Required crossovers for signal: {min_crossovers_for_signal}")
        self.logger.info(f"  Database persistence: {'ENABLED' if self.use_database_state else 'DISABLED'}")
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

            # Check if we already have a pending crossover for this same timestamp
            # This prevents re-detecting the same crossover on every scan cycle
            state = self._get_or_create_state(epic)

            # Check if this crossover timestamp has already been processed
            last_crossover_ts = state.last_bull_crossover_timestamp if direction == 'BULL' else state.last_bear_crossover_timestamp
            if last_crossover_ts is not None:
                # Compare timestamps - if same, skip detection
                if str(last_crossover_ts)[:16] == str(timestamp)[:16]:
                    # Already processed this crossover - check if we have a pending one to return
                    existing_pending = state.pending_bull_crossover if direction == 'BULL' else state.pending_bear_crossover
                    if existing_pending is not None:
                        # Return existing pending for continued validation
                        return existing_pending
                    else:
                        # Crossover was already validated (success or fail), don't re-detect
                        return None

            # This is a genuinely new crossover
            # Create crossover event
            event = CrossoverEvent(
                timestamp=timestamp,
                direction=direction,
                price_at_crossover=float(curr.get('close', 0)),
                ema_fast=float(curr_ema_fast),
                ema_slow=float(curr_ema_slow),
                candle_index=len(df) - 1
            )

            # Store as pending for validation and track the timestamp
            if direction == 'BULL':
                state.pending_bull_crossover = event
                state.last_bull_crossover_timestamp = timestamp
            else:
                state.pending_bear_crossover = event
                state.last_bear_crossover_timestamp = timestamp

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
        state_changed = False
        if state.pending_bull_crossover and state.pending_bull_crossover.is_pending():
            result = self._validate_single_crossover(df, state.pending_bull_crossover, epic)
            if result is not None:
                if result:
                    validated.append(state.pending_bull_crossover)
                    state.add_successful_crossover(state.pending_bull_crossover)
                    self.logger.info(f"[{epic}] BULL crossover VALIDATED as SUCCESS (held {state.pending_bull_crossover.candles_held} candles)")
                    state_changed = True
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
                    state_changed = True
                else:
                    self.logger.info(f"[{epic}] BEAR crossover FAILED validation")
                state.pending_bear_crossover = None

        # Cleanup expired crossovers using the SIMULATION time (latest candle time)
        # This is critical for backtesting to work correctly
        current_simulation_time = self._get_latest_timestamp(df)
        self._cleanup_expired_crossovers(epic, current_simulation_time)

        # Save state to database if changed (successful crossover validated)
        if state_changed and self.use_database_state:
            self._save_state_to_db(epic)

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
            # Find the crossover candle by timestamp instead of index
            # This fixes the issue where candle_index becomes stale across scan cycles
            crossover_ts = event.timestamp
            crossover_idx = None

            # Search for the candle matching the crossover timestamp
            for i in range(len(df)):
                row = df.iloc[i]
                row_ts = row.get('start_time') or row.name
                if isinstance(row_ts, str):
                    row_ts = pd.to_datetime(row_ts)

                # Compare timestamps (up to minutes)
                if str(row_ts)[:16] == str(crossover_ts)[:16]:
                    crossover_idx = i
                    break

            if crossover_idx is None:
                # Crossover candle not in current DataFrame - likely scrolled out
                # Mark as failed since we can't validate
                self.logger.debug(f"[{epic}] Crossover candle not found in DataFrame, marking as failed")
                event.success = False
                event.validation_timestamp = datetime.now()
                return False

            # Find candles since crossover
            candles_since = len(df) - 1 - crossover_idx

            if candles_since < 1:
                return None  # Not enough candles yet

            # Check each candle since crossover
            favorable_count = 0

            for i in range(crossover_idx + 1, len(df)):
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
                    self.logger.debug(f"[{epic}] {event.direction} crossover FAILED - price crossed back after {favorable_count} candles")
                    return False

            event.candles_validated = candles_since
            event.candles_held = favorable_count

            # Check if we have enough favorable candles
            if favorable_count >= self.success_candles:
                event.success = True
                event.validation_timestamp = datetime.now()
                self.logger.info(f"[{epic}] {event.direction} crossover VALIDATED - held {favorable_count} candles")
                return True

            # Check if we've waited too long
            if candles_since >= self.max_validation_candles:
                event.success = False
                event.validation_timestamp = datetime.now()
                self.logger.debug(f"[{epic}] {event.direction} crossover FAILED - max validation time exceeded")
                return False

            # Still pending
            self.logger.debug(f"[{epic}] {event.direction} crossover still pending - {favorable_count}/{self.success_candles} candles held")
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

        # Save state to database after trade is recorded
        if self.use_database_state:
            self._save_state_to_db(epic)

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
                # Also reset in database if using persistence
                if self.use_database_state:
                    self._delete_state_from_db(epic)
        else:
            self._pair_states = {}
            self.logger.info("All states reset")
            # Also reset all in database if using persistence
            if self.use_database_state:
                self._delete_all_states_from_db()

    # =========================================================================
    # DATABASE PERSISTENCE METHODS (v2.0.0)
    # =========================================================================

    def _initialize_database_table(self):
        """Create the ema_double_crossover_state table if it doesn't exist"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            # Create main state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ema_double_crossover_state (
                    id SERIAL PRIMARY KEY,
                    epic VARCHAR(50) NOT NULL UNIQUE,

                    -- Successful crossover history (JSON arrays)
                    successful_bull_crossovers JSON DEFAULT '[]',
                    successful_bear_crossovers JSON DEFAULT '[]',

                    -- Pending crossovers (JSON objects or null)
                    pending_bull_crossover JSON DEFAULT NULL,
                    pending_bear_crossover JSON DEFAULT NULL,

                    -- Last trade info
                    last_trade_direction VARCHAR(10),
                    last_trade_timestamp TIMESTAMP,

                    -- Last crossover timestamps (to avoid re-detection)
                    last_bull_crossover_timestamp TIMESTAMP,
                    last_bear_crossover_timestamp TIMESTAMP,

                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for fast lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ema_crossover_epic
                ON ema_double_crossover_state(epic)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ema_crossover_updated
                ON ema_double_crossover_state(updated_at)
            ''')

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info("✅ Database table 'ema_double_crossover_state' initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database table: {e}")
            raise

    def _load_all_states_from_db(self):
        """Load all crossover states from database on startup"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT epic, successful_bull_crossovers, successful_bear_crossovers,
                       pending_bull_crossover, pending_bear_crossover,
                       last_trade_direction, last_trade_timestamp,
                       last_bull_crossover_timestamp, last_bear_crossover_timestamp
                FROM ema_double_crossover_state
            ''')

            rows = cursor.fetchall()
            loaded_count = 0

            for row in rows:
                epic = row[0]
                state = self._deserialize_state_from_row(row)
                if state:
                    self._pair_states[epic] = state
                    loaded_count += 1

            cursor.close()
            conn.close()

            self.logger.info(f"✅ Loaded {loaded_count} crossover states from database")

            # Log summary of loaded states
            for epic, state in self._pair_states.items():
                bull_count = len(state.successful_bull_crossovers)
                bear_count = len(state.successful_bear_crossovers)
                if bull_count > 0 or bear_count > 0:
                    self.logger.info(f"   [{epic}] Bull: {bull_count}, Bear: {bear_count} successful crossovers")

        except Exception as e:
            self.logger.error(f"❌ Failed to load states from database: {e}")
            # Don't raise - fall back to empty state

    def _save_state_to_db(self, epic: str):
        """Save a single pair's state to database"""
        if not self.use_database_state:
            return

        try:
            state = self._pair_states.get(epic)
            if not state:
                return

            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            # Serialize crossover lists and pending crossovers
            bull_crossovers_json = json.dumps([
                self._serialize_crossover_event(c) for c in state.successful_bull_crossovers
            ])
            bear_crossovers_json = json.dumps([
                self._serialize_crossover_event(c) for c in state.successful_bear_crossovers
            ])
            pending_bull_json = json.dumps(
                self._serialize_crossover_event(state.pending_bull_crossover)
            ) if state.pending_bull_crossover else None
            pending_bear_json = json.dumps(
                self._serialize_crossover_event(state.pending_bear_crossover)
            ) if state.pending_bear_crossover else None

            # Upsert state
            cursor.execute('''
                INSERT INTO ema_double_crossover_state
                (epic, successful_bull_crossovers, successful_bear_crossovers,
                 pending_bull_crossover, pending_bear_crossover,
                 last_trade_direction, last_trade_timestamp,
                 last_bull_crossover_timestamp, last_bear_crossover_timestamp,
                 updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (epic) DO UPDATE SET
                    successful_bull_crossovers = EXCLUDED.successful_bull_crossovers,
                    successful_bear_crossovers = EXCLUDED.successful_bear_crossovers,
                    pending_bull_crossover = EXCLUDED.pending_bull_crossover,
                    pending_bear_crossover = EXCLUDED.pending_bear_crossover,
                    last_trade_direction = EXCLUDED.last_trade_direction,
                    last_trade_timestamp = EXCLUDED.last_trade_timestamp,
                    last_bull_crossover_timestamp = EXCLUDED.last_bull_crossover_timestamp,
                    last_bear_crossover_timestamp = EXCLUDED.last_bear_crossover_timestamp,
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                epic,
                bull_crossovers_json,
                bear_crossovers_json,
                pending_bull_json,
                pending_bear_json,
                state.last_trade_direction,
                state.last_trade_timestamp,
                state.last_bull_crossover_timestamp,
                state.last_bear_crossover_timestamp
            ))

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.debug(f"💾 Saved state to DB for {epic}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save state to database for {epic}: {e}")

    def _delete_state_from_db(self, epic: str):
        """Delete a single pair's state from database"""
        if not self.use_database_state:
            return

        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                'DELETE FROM ema_double_crossover_state WHERE epic = %s',
                (epic,)
            )

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(f"🗑️ Deleted state from DB for {epic}")

        except Exception as e:
            self.logger.error(f"❌ Failed to delete state from database for {epic}: {e}")

    def _delete_all_states_from_db(self):
        """Delete all states from database"""
        if not self.use_database_state:
            return

        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute('DELETE FROM ema_double_crossover_state')

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info("🗑️ Deleted all states from DB")

        except Exception as e:
            self.logger.error(f"❌ Failed to delete all states from database: {e}")

    def _serialize_crossover_event(self, event: CrossoverEvent) -> Optional[Dict]:
        """Serialize a CrossoverEvent to a JSON-compatible dictionary"""
        if event is None:
            return None

        return {
            'timestamp': event.timestamp.isoformat() if event.timestamp else None,
            'direction': event.direction,
            'price_at_crossover': event.price_at_crossover,
            'ema_fast': event.ema_fast,
            'ema_slow': event.ema_slow,
            'candle_index': event.candle_index,
            'success': event.success,
            'validation_timestamp': event.validation_timestamp.isoformat() if event.validation_timestamp else None,
            'candles_validated': event.candles_validated,
            'candles_held': event.candles_held
        }

    def _deserialize_crossover_event(self, data: Dict) -> Optional[CrossoverEvent]:
        """Deserialize a dictionary to a CrossoverEvent"""
        if data is None:
            return None

        try:
            timestamp = None
            if data.get('timestamp'):
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))

            validation_timestamp = None
            if data.get('validation_timestamp'):
                validation_timestamp = datetime.fromisoformat(
                    data['validation_timestamp'].replace('Z', '+00:00')
                )

            return CrossoverEvent(
                timestamp=timestamp,
                direction=data.get('direction', 'BULL'),
                price_at_crossover=float(data.get('price_at_crossover', 0)),
                ema_fast=float(data.get('ema_fast', 0)),
                ema_slow=float(data.get('ema_slow', 0)),
                candle_index=int(data.get('candle_index', -1)),
                success=data.get('success'),
                validation_timestamp=validation_timestamp,
                candles_validated=int(data.get('candles_validated', 0)),
                candles_held=int(data.get('candles_held', 0))
            )
        except Exception as e:
            self.logger.warning(f"Failed to deserialize crossover event: {e}")
            return None

    def _deserialize_state_from_row(self, row: tuple) -> Optional[PairCrossoverState]:
        """Deserialize a database row to PairCrossoverState"""
        try:
            epic = row[0]
            bull_json = row[1] if row[1] else '[]'
            bear_json = row[2] if row[2] else '[]'
            pending_bull_json = row[3]
            pending_bear_json = row[4]
            last_trade_direction = row[5]
            last_trade_timestamp = row[6]
            last_bull_ts = row[7]
            last_bear_ts = row[8]

            # Parse JSON arrays
            bull_data = json.loads(bull_json) if isinstance(bull_json, str) else bull_json
            bear_data = json.loads(bear_json) if isinstance(bear_json, str) else bear_json

            # Parse pending crossovers
            pending_bull_data = None
            if pending_bull_json:
                pending_bull_data = json.loads(pending_bull_json) if isinstance(pending_bull_json, str) else pending_bull_json

            pending_bear_data = None
            if pending_bear_json:
                pending_bear_data = json.loads(pending_bear_json) if isinstance(pending_bear_json, str) else pending_bear_json

            # Deserialize crossover events
            successful_bull = [
                self._deserialize_crossover_event(c) for c in bull_data
                if self._deserialize_crossover_event(c) is not None
            ]
            successful_bear = [
                self._deserialize_crossover_event(c) for c in bear_data
                if self._deserialize_crossover_event(c) is not None
            ]

            pending_bull = self._deserialize_crossover_event(pending_bull_data)
            pending_bear = self._deserialize_crossover_event(pending_bear_data)

            return PairCrossoverState(
                epic=epic,
                successful_bull_crossovers=successful_bull,
                successful_bear_crossovers=successful_bear,
                pending_bull_crossover=pending_bull,
                pending_bear_crossover=pending_bear,
                last_trade_direction=last_trade_direction,
                last_trade_timestamp=last_trade_timestamp,
                last_bull_crossover_timestamp=last_bull_ts,
                last_bear_crossover_timestamp=last_bear_ts
            )

        except Exception as e:
            self.logger.error(f"Failed to deserialize state from row: {e}")
            return None
