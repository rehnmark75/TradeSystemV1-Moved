"""
Master Pattern Phase Tracker
============================
State machine for tracking AMD (Accumulation, Manipulation, Distribution) phases
per currency pair for the ICT Power of 3 strategy.

Phases:
1. WAITING - Looking for accumulation range
2. ACCUMULATION - Range detected, tracking bounds
3. MANIPULATION - Sweep detected, waiting for structure shift
4. DISTRIBUTION - Structure shifted, looking for entry
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class AMDPhase(Enum):
    """AMD phase states."""
    WAITING = "waiting"                   # Looking for accumulation
    ACCUMULATION = "accumulation"         # Range detected, tracking bounds
    MANIPULATION = "manipulation"         # Sweep detected, waiting for structure shift
    DISTRIBUTION = "distribution"         # Structure shifted, looking for entry
    COMPLETED = "completed"               # Signal generated, waiting for reset


class SweepDirection(Enum):
    """Direction of the manipulation sweep."""
    SWEEP_LOWS = "sweep_lows"             # Bullish setup (swept lows, expect up)
    SWEEP_HIGHS = "sweep_highs"           # Bearish setup (swept highs, expect down)


@dataclass
class AccumulationZone:
    """Data class for accumulation zone details."""
    range_high: float
    range_low: float
    daily_open: float
    candle_count: int
    start_time: datetime
    end_time: datetime
    atr_ratio: float                      # Current ATR / Baseline ATR
    volume_profile_valid: bool = True     # Whether volume declined during accumulation

    @property
    def range_pips(self) -> float:
        """Calculate range in price units (not pips - caller converts)."""
        return self.range_high - self.range_low

    @property
    def midpoint(self) -> float:
        """Calculate midpoint of accumulation range."""
        return (self.range_high + self.range_low) / 2

    @property
    def is_below_daily_open(self) -> bool:
        """Check if accumulation is below daily open (bullish bias)."""
        return self.midpoint < self.daily_open

    @property
    def is_above_daily_open(self) -> bool:
        """Check if accumulation is above daily open (bearish bias)."""
        return self.midpoint > self.daily_open


@dataclass
class ManipulationEvent:
    """Data class for manipulation (Judas swing) details."""
    sweep_direction: SweepDirection
    sweep_level: float                    # The price level swept (SL level)
    sweep_candle_high: float
    sweep_candle_low: float
    sweep_candle_close: float
    sweep_candle_volume: float
    timestamp: datetime
    volume_ratio: float                   # Sweep volume / Average volume
    rejection_wick_ratio: float           # Wick size / Total candle range
    extension_pips: float                 # How far beyond range the sweep went

    @property
    def is_valid_rejection(self) -> bool:
        """Check if rejection wick is significant."""
        return self.rejection_wick_ratio >= 0.65

    @property
    def has_volume_spike(self) -> bool:
        """Check if volume spiked on sweep."""
        return self.volume_ratio >= 1.5


@dataclass
class StructureShiftEvent:
    """Data class for structure shift (BOS/ChoCH) details."""
    direction: str                        # 'bullish' or 'bearish'
    break_type: str                       # 'BOS' or 'ChoCH'
    break_candle_index: int
    break_price: float
    timestamp: datetime
    volume_ratio: float = 1.0             # Volume on shift candle / Average
    body_ratio: float = 0.0               # Body size / Candle range


@dataclass
class EntryZone:
    """Data class for entry zone details."""
    zone_type: str                        # 'FVG' or 'MITIGATION'
    zone_high: float
    zone_low: float
    significance: float = 1.0             # Quality score 0-1
    timestamp: Optional[datetime] = None

    @property
    def midpoint(self) -> float:
        """Calculate zone midpoint."""
        return (self.zone_high + self.zone_low) / 2


@dataclass
class AMDState:
    """
    Complete state for AMD pattern tracking on a single pair.
    """
    pair: str
    phase: AMDPhase = AMDPhase.WAITING
    session_date: Optional[date] = None

    # Phase data
    accumulation: Optional[AccumulationZone] = None
    manipulation: Optional[ManipulationEvent] = None
    structure_shift: Optional[StructureShiftEvent] = None
    entry_zone: Optional[EntryZone] = None

    # Timing
    phase_start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None

    # Metadata
    confidence_score: float = 0.0
    rejection_reason: Optional[str] = None

    def reset(self, keep_date: bool = False):
        """Reset state for new AMD cycle."""
        self.phase = AMDPhase.WAITING
        self.accumulation = None
        self.manipulation = None
        self.structure_shift = None
        self.entry_zone = None
        self.phase_start_time = None
        self.confidence_score = 0.0
        self.rejection_reason = None

        if not keep_date:
            self.session_date = None

    def get_signal_direction(self) -> Optional[str]:
        """Get signal direction based on manipulation sweep."""
        if self.manipulation is None:
            return None

        if self.manipulation.sweep_direction == SweepDirection.SWEEP_LOWS:
            return 'BULL'
        else:
            return 'BEAR'


class MasterPatternPhaseTracker:
    """
    Tracks AMD phases for multiple currency pairs.

    Usage:
        tracker = MasterPatternPhaseTracker()

        # On each candle update
        tracker.update_accumulation(pair, accumulation_zone)
        tracker.update_manipulation(pair, manipulation_event)
        tracker.update_structure_shift(pair, structure_shift)
        tracker.update_entry_zone(pair, entry_zone)

        # Check if ready for signal
        if tracker.is_ready_for_signal(pair):
            state = tracker.get_state(pair)
            # Generate signal from state
            tracker.mark_completed(pair)
    """

    def __init__(
        self,
        accumulation_timeout_hours: int = 10,
        manipulation_timeout_hours: int = 2,
        structure_timeout_candles: int = 15,
        entry_timeout_candles: int = 20
    ):
        """
        Initialize phase tracker.

        Args:
            accumulation_timeout_hours: Max hours in accumulation phase
            manipulation_timeout_hours: Max hours waiting for manipulation
            structure_timeout_candles: Max candles for structure shift
            entry_timeout_candles: Max candles to wait for entry
        """
        self.states: Dict[str, AMDState] = {}

        # Timeouts
        self.accumulation_timeout = timedelta(hours=accumulation_timeout_hours)
        self.manipulation_timeout = timedelta(hours=manipulation_timeout_hours)
        self.structure_timeout_candles = structure_timeout_candles
        self.entry_timeout_candles = entry_timeout_candles

        self.logger = logging.getLogger(f"{__name__}.MasterPatternPhaseTracker")

    def get_state(self, pair: str) -> AMDState:
        """Get or create state for a pair."""
        if pair not in self.states:
            self.states[pair] = AMDState(pair=pair)
        return self.states[pair]

    def reset_pair(self, pair: str, keep_date: bool = False):
        """Reset state for a pair."""
        state = self.get_state(pair)
        state.reset(keep_date)
        self.logger.debug(f"[{pair}] State reset")

    def reset_all(self):
        """Reset all pair states."""
        for pair in self.states:
            self.reset_pair(pair)

    def check_daily_reset(self, pair: str, current_date: date):
        """Reset state if it's a new trading day."""
        state = self.get_state(pair)
        if state.session_date != current_date:
            self.logger.info(f"[{pair}] New trading day detected, resetting state")
            state.reset()
            state.session_date = current_date

    # =========================================================================
    # PHASE TRANSITIONS
    # =========================================================================

    def update_accumulation(
        self,
        pair: str,
        accumulation: AccumulationZone,
        current_time: datetime
    ) -> bool:
        """
        Update state with detected accumulation zone.

        Args:
            pair: Currency pair
            accumulation: Detected accumulation zone
            current_time: Current timestamp

        Returns:
            True if accumulation was accepted and phase transitioned
        """
        state = self.get_state(pair)

        # Check if we're in correct phase
        if state.phase not in [AMDPhase.WAITING, AMDPhase.ACCUMULATION]:
            self.logger.debug(f"[{pair}] Ignoring accumulation update - phase is {state.phase.value}")
            return False

        # Store accumulation
        state.accumulation = accumulation
        state.phase = AMDPhase.ACCUMULATION
        state.phase_start_time = current_time
        state.last_update_time = current_time
        state.session_date = current_time.date()

        self.logger.info(
            f"[{pair}] Accumulation detected: "
            f"high={accumulation.range_high:.5f}, low={accumulation.range_low:.5f}, "
            f"candles={accumulation.candle_count}, ATR ratio={accumulation.atr_ratio:.2f}"
        )

        return True

    def update_manipulation(
        self,
        pair: str,
        manipulation: ManipulationEvent,
        current_time: datetime
    ) -> bool:
        """
        Update state with detected manipulation (Judas swing).

        Args:
            pair: Currency pair
            manipulation: Detected manipulation event
            current_time: Current timestamp

        Returns:
            True if manipulation was accepted and phase transitioned
        """
        state = self.get_state(pair)

        # Must have accumulation first
        if state.phase != AMDPhase.ACCUMULATION:
            self.logger.debug(
                f"[{pair}] Ignoring manipulation update - "
                f"no accumulation detected (phase={state.phase.value})"
            )
            return False

        # Check accumulation timeout
        if state.phase_start_time:
            elapsed = current_time - state.phase_start_time
            if elapsed > self.accumulation_timeout:
                self.logger.warning(f"[{pair}] Accumulation timeout - resetting")
                state.reset(keep_date=True)
                return False

        # Store manipulation
        state.manipulation = manipulation
        state.phase = AMDPhase.MANIPULATION
        state.phase_start_time = current_time
        state.last_update_time = current_time

        direction = "BULLISH" if manipulation.sweep_direction == SweepDirection.SWEEP_LOWS else "BEARISH"
        self.logger.info(
            f"[{pair}] Manipulation detected: {direction}, "
            f"sweep_level={manipulation.sweep_level:.5f}, "
            f"volume_ratio={manipulation.volume_ratio:.2f}, "
            f"wick_ratio={manipulation.rejection_wick_ratio:.2f}"
        )

        return True

    def update_structure_shift(
        self,
        pair: str,
        structure_shift: StructureShiftEvent,
        current_time: datetime
    ) -> bool:
        """
        Update state with detected structure shift (BOS/ChoCH).

        Args:
            pair: Currency pair
            structure_shift: Detected structure shift
            current_time: Current timestamp

        Returns:
            True if structure shift was accepted and phase transitioned
        """
        state = self.get_state(pair)

        # Must have manipulation first
        if state.phase != AMDPhase.MANIPULATION:
            self.logger.debug(
                f"[{pair}] Ignoring structure shift - "
                f"no manipulation detected (phase={state.phase.value})"
            )
            return False

        # Validate direction matches sweep
        expected_direction = 'bullish' if state.manipulation.sweep_direction == SweepDirection.SWEEP_LOWS else 'bearish'
        if structure_shift.direction != expected_direction:
            self.logger.warning(
                f"[{pair}] Structure shift direction ({structure_shift.direction}) "
                f"doesn't match expected ({expected_direction})"
            )
            return False

        # Store structure shift
        state.structure_shift = structure_shift
        state.phase = AMDPhase.DISTRIBUTION
        state.phase_start_time = current_time
        state.last_update_time = current_time

        self.logger.info(
            f"[{pair}] Structure shift detected: {structure_shift.direction} {structure_shift.break_type}, "
            f"break_price={structure_shift.break_price:.5f}"
        )

        return True

    def update_entry_zone(
        self,
        pair: str,
        entry_zone: EntryZone,
        current_time: datetime
    ) -> bool:
        """
        Update state with detected entry zone.

        Args:
            pair: Currency pair
            entry_zone: Detected entry zone (FVG or mitigation)
            current_time: Current timestamp

        Returns:
            True if entry zone was accepted
        """
        state = self.get_state(pair)

        # Must have structure shift first
        if state.phase != AMDPhase.DISTRIBUTION:
            self.logger.debug(
                f"[{pair}] Ignoring entry zone - "
                f"no structure shift (phase={state.phase.value})"
            )
            return False

        # Store entry zone
        state.entry_zone = entry_zone
        state.last_update_time = current_time

        self.logger.info(
            f"[{pair}] Entry zone detected: {entry_zone.zone_type}, "
            f"high={entry_zone.zone_high:.5f}, low={entry_zone.zone_low:.5f}"
        )

        return True

    # =========================================================================
    # STATE QUERIES
    # =========================================================================

    def is_ready_for_signal(self, pair: str) -> bool:
        """
        Check if all AMD phases complete and ready for signal.

        Args:
            pair: Currency pair

        Returns:
            True if ready to generate signal
        """
        state = self.get_state(pair)

        return (
            state.phase == AMDPhase.DISTRIBUTION and
            state.accumulation is not None and
            state.manipulation is not None and
            state.structure_shift is not None and
            state.entry_zone is not None
        )

    def get_phase(self, pair: str) -> AMDPhase:
        """Get current phase for a pair."""
        return self.get_state(pair).phase

    def get_signal_direction(self, pair: str) -> Optional[str]:
        """Get expected signal direction based on manipulation sweep."""
        return self.get_state(pair).get_signal_direction()

    def mark_completed(self, pair: str):
        """Mark AMD cycle as completed (signal generated)."""
        state = self.get_state(pair)
        state.phase = AMDPhase.COMPLETED
        self.logger.info(f"[{pair}] AMD cycle completed, signal generated")

    def mark_rejected(self, pair: str, reason: str):
        """Mark AMD cycle as rejected with reason."""
        state = self.get_state(pair)
        state.rejection_reason = reason
        self.logger.info(f"[{pair}] AMD cycle rejected: {reason}")
        state.reset(keep_date=True)

    def check_timeouts(self, pair: str, current_time: datetime) -> Optional[str]:
        """
        Check if any phase has timed out.

        Args:
            pair: Currency pair
            current_time: Current timestamp

        Returns:
            Timeout reason if timed out, None otherwise
        """
        state = self.get_state(pair)

        if state.phase_start_time is None:
            return None

        elapsed = current_time - state.phase_start_time

        if state.phase == AMDPhase.ACCUMULATION:
            if elapsed > self.accumulation_timeout:
                return "Accumulation phase timeout"

        elif state.phase == AMDPhase.MANIPULATION:
            if elapsed > self.manipulation_timeout:
                return "Manipulation phase timeout"

        return None

    # =========================================================================
    # CONFIDENCE CALCULATION
    # =========================================================================

    def calculate_confidence(
        self,
        pair: str,
        rr_ratio: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate confidence score for AMD setup.

        Args:
            pair: Currency pair
            rr_ratio: Risk-reward ratio of the setup
            weights: Optional custom weights

        Returns:
            Confidence score 0.0-1.0
        """
        state = self.get_state(pair)

        if not self.is_ready_for_signal(pair):
            return 0.0

        # Default weights
        if weights is None:
            weights = {
                'accumulation_quality': 0.20,
                'manipulation_clarity': 0.25,
                'structure_shift_strength': 0.25,
                'entry_zone_quality': 0.15,
                'rr_quality': 0.15,
            }

        confidence = 0.0

        # 1. Accumulation Quality (20%)
        acc = state.accumulation
        # ATR score: 0.85 threshold = 50% score, lower = better, max at 0.5 ATR ratio
        atr_score = max(0, min(1.0, (0.90 - acc.atr_ratio) / 0.40))  # 0.50=100%, 0.90=0%
        duration_score = min(acc.candle_count / 12, 1.0)  # 12 candles = optimal (60 min on 5m)
        volume_score = 1.0 if acc.volume_profile_valid else 0.7

        accumulation_score = (atr_score * 0.4 + duration_score * 0.4 + volume_score * 0.2)
        confidence += accumulation_score * weights['accumulation_quality']

        # 2. Manipulation Clarity (25%)
        manip = state.manipulation
        # Volume spike: any spike above 0.8x gets partial credit, 1.5x = full
        volume_spike_score = min(max(0, (manip.volume_ratio - 0.8) / 0.7), 1.0)
        # Wick score: 50% wick gets partial credit, 75% = full
        wick_score = min(max(0, (manip.rejection_wick_ratio - 0.50) / 0.25), 1.0)

        manipulation_score = (volume_spike_score * 0.4 + wick_score * 0.6)
        confidence += manipulation_score * weights['manipulation_clarity']

        # 3. Structure Shift Strength (25%)
        shift = state.structure_shift
        shift_type_score = 1.0 if shift.break_type == 'BOS' else 0.8  # ChoCH still valid
        shift_volume_score = min(max(0, (shift.volume_ratio - 0.7) / 0.8), 1.0)  # 0.7x to 1.5x
        body_score = min(max(0, (shift.body_ratio - 0.4) / 0.3), 1.0)  # 40% to 70%

        structure_score = (shift_type_score * 0.4 + shift_volume_score * 0.3 + body_score * 0.3)
        confidence += structure_score * weights['structure_shift_strength']

        # 4. Entry Zone Quality (15%)
        entry = state.entry_zone
        zone_type_score = 1.0 if entry.zone_type == 'FVG' else 0.75  # Mitigation still valid

        entry_score = zone_type_score * max(entry.significance, 0.5)  # Min 50% significance
        confidence += entry_score * weights['entry_zone_quality']

        # 5. R:R Quality (15%)
        # R:R 2.0 = 50% score, 3.0 = 100% score
        rr_score = max(0, min((rr_ratio - 1.5) / 1.5, 1.0))  # 1.5-3.0 range
        confidence += rr_score * weights['rr_quality']

        # Store and return
        state.confidence_score = min(confidence, 1.0)
        return state.confidence_score

    # =========================================================================
    # DEBUG / LOGGING
    # =========================================================================

    def get_state_summary(self, pair: str) -> Dict:
        """Get summary of current state for a pair."""
        state = self.get_state(pair)

        return {
            'pair': pair,
            'phase': state.phase.value,
            'session_date': str(state.session_date) if state.session_date else None,
            'has_accumulation': state.accumulation is not None,
            'has_manipulation': state.manipulation is not None,
            'has_structure_shift': state.structure_shift is not None,
            'has_entry_zone': state.entry_zone is not None,
            'signal_direction': state.get_signal_direction(),
            'confidence': state.confidence_score,
            'rejection_reason': state.rejection_reason,
        }

    def log_all_states(self):
        """Log summary of all pair states."""
        self.logger.info("=" * 50)
        self.logger.info("Master Pattern Phase Tracker - All States")
        self.logger.info("=" * 50)

        for pair, state in self.states.items():
            summary = self.get_state_summary(pair)
            self.logger.info(f"  {pair}: {summary['phase']}")
            if summary['signal_direction']:
                self.logger.info(f"    Direction: {summary['signal_direction']}")
            if summary['confidence'] > 0:
                self.logger.info(f"    Confidence: {summary['confidence']:.2%}")

        self.logger.info("=" * 50)


# Module-level instance for convenience
phase_tracker = MasterPatternPhaseTracker()
