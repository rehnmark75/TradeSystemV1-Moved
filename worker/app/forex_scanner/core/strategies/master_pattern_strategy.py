"""
Master Pattern (ICT Power of 3 / AMD) Strategy
===============================================
Implements the ICT Power of 3 trading pattern:
- Accumulation: Asian session consolidation (00:00-08:00 UTC)
- Manipulation: Judas swing/liquidity sweep during London open (08:00-10:00 UTC)
- Distribution: Real move + entry on FVG pullback

Entry Logic:
- Bullish: Accumulation → sweep lows → BOS/ChoCH up → enter on FVG
- Bearish: Accumulation → sweep highs → BOS/ChoCH down → enter on FVG

Expected Performance:
- Win Rate: 42-48%
- Profit Factor: 1.6-1.9
- Average R:R: 2.3-2.8
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_strategy import BaseStrategy

# Import helpers
from .helpers.master_pattern_session_analyzer import (
    MasterPatternSessionAnalyzer,
    is_asian_session,
    is_london_open,
    is_distribution_window,
    get_daily_open,
    get_asian_range
)
from .helpers.master_pattern_phase_tracker import (
    MasterPatternPhaseTracker,
    AMDPhase,
    SweepDirection,
    AccumulationZone,
    ManipulationEvent,
    StructureShiftEvent,
    EntryZone,
    AMDState
)

# Import SMC helpers
try:
    from .helpers.smc_market_structure import SMCMarketStructure, StructureBreak
    from .helpers.smc_fair_value_gaps import SMCFairValueGaps
except ImportError:
    SMCMarketStructure = None
    SMCFairValueGaps = None

# Import configuration
try:
    from configdata.strategies import config_master_pattern as config
except ImportError:
    try:
        from forex_scanner.configdata.strategies import config_master_pattern as config
    except ImportError:
        config = None


class MasterPatternStrategy(BaseStrategy):
    """
    ICT Power of 3 (AMD) Trading Strategy

    Phases:
    1. ACCUMULATION: Detect Asian session consolidation
    2. MANIPULATION: Detect Judas swing that sweeps liquidity
    3. DISTRIBUTION: Enter after structure shift on FVG pullback
    """

    def __init__(self, data_fetcher=None, **kwargs):
        """Initialize Master Pattern strategy."""
        super().__init__('MASTER_PATTERN')

        self.data_fetcher = data_fetcher
        self.config = config

        # Initialize helpers
        self.session_analyzer = MasterPatternSessionAnalyzer()
        self.phase_tracker = MasterPatternPhaseTracker(
            accumulation_timeout_hours=config.ACCUMULATION_TIMEOUT_HOURS if config else 10,
            manipulation_timeout_hours=config.MANIPULATION_TIMEOUT_HOURS if config else 2,
            structure_timeout_candles=config.STRUCTURE_TIMEOUT_CANDLES if config else 15,
            entry_timeout_candles=config.ENTRY_TIMEOUT_CANDLES if config else 20
        )

        # Initialize SMC helpers
        self.smc_structure = SMCMarketStructure(logger=self.logger) if SMCMarketStructure else None
        self.fvg_detector = SMCFairValueGaps(logger=self.logger) if SMCFairValueGaps else None

        # Load configuration
        self._load_config()

        self.logger.info(f"[{self.name}] Master Pattern (ICT Power of 3) strategy initialized v{self.version}")

    def _load_config(self):
        """Load configuration parameters."""
        if config is None:
            self.logger.warning(f"[{self.name}] Config not found, using defaults")
            self._set_defaults()
            return

        # Metadata
        self.version = getattr(config, 'STRATEGY_VERSION', '1.0.0')

        # Accumulation parameters
        self.min_accumulation_candles = getattr(config, 'MIN_ACCUMULATION_CANDLES_5M', 10)
        self.max_accumulation_range_pips = getattr(config, 'MAX_ACCUMULATION_RANGE_PIPS', 30)
        self.atr_compression_threshold = getattr(config, 'ATR_COMPRESSION_THRESHOLD', 0.60)
        self.validate_accumulation_volume = getattr(config, 'VALIDATE_ACCUMULATION_VOLUME', True)

        # Manipulation parameters
        self.min_sweep_pips = getattr(config, 'MIN_SWEEP_EXTENSION_PIPS', 3)
        self.max_sweep_pips = getattr(config, 'MAX_SWEEP_EXTENSION_PIPS', 15)
        self.sweep_volume_multiplier = getattr(config, 'SWEEP_VOLUME_MULTIPLIER', 1.5)
        self.min_rejection_wick_ratio = getattr(config, 'MIN_REJECTION_WICK_RATIO', 0.65)

        # Structure shift parameters
        self.require_structure_shift = getattr(config, 'REQUIRE_STRUCTURE_SHIFT', True)
        self.min_bos_volume_multiplier = getattr(config, 'MIN_BOS_VOLUME_MULTIPLIER', 1.4)

        # Entry parameters
        self.max_fvg_distance_pips = getattr(config, 'MAX_FVG_DISTANCE_PIPS', 20)
        self.min_fvg_size_pips = getattr(config, 'MIN_FVG_SIZE_PIPS', 2)

        # Risk parameters
        self.min_rr_ratio = getattr(config, 'MIN_RR_RATIO', 2.0)
        self.sl_buffer_pips = getattr(config, 'SL_BUFFER_PIPS', 3)
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE_THRESHOLD', 0.65)

        # Pair config
        self.enabled_pairs = getattr(config, 'ENABLED_PAIRS', [])
        self.pair_pip_values = getattr(config, 'PAIR_PIP_VALUES', {})

    def _set_defaults(self):
        """Set default configuration values."""
        self.version = '1.0.0'
        self.min_accumulation_candles = 10
        self.max_accumulation_range_pips = 30
        self.atr_compression_threshold = 0.60
        self.validate_accumulation_volume = True
        self.min_sweep_pips = 3
        self.max_sweep_pips = 15
        self.sweep_volume_multiplier = 1.5
        self.min_rejection_wick_ratio = 0.65
        self.require_structure_shift = True
        self.min_bos_volume_multiplier = 1.4
        self.max_fvg_distance_pips = 20
        self.min_fvg_size_pips = 2
        self.min_rr_ratio = 2.0
        self.sl_buffer_pips = 3
        self.min_confidence = 0.65
        self.enabled_pairs = []
        self.pair_pip_values = {}

    def get_required_indicators(self) -> List[str]:
        """Return list of required indicators."""
        return ['open', 'high', 'low', 'close', 'volume', 'atr']

    def get_pip_value(self, epic: str) -> float:
        """Get pip value for an epic."""
        if config and hasattr(config, 'get_pip_value'):
            return config.get_pip_value(epic)
        return self.pair_pip_values.get(epic, 0.0001)

    def detect_signal(
        self,
        df: pd.DataFrame,
        epic: str,
        spread_pips: float = 1.5,
        timeframe: str = '15m'
    ) -> Optional[Dict]:
        """
        Main signal detection entry point.

        This method orchestrates the AMD pattern detection across multiple timeframes.
        For proper AMD detection, this should be called with 15m data.

        Args:
            df: DataFrame with OHLCV data (should be 15m for AMD)
            epic: Instrument epic code
            spread_pips: Spread in pips
            timeframe: Timeframe of the data

        Returns:
            Signal dictionary or None
        """
        try:
            # Validate data
            if df is None or len(df) < 50:
                return None

            # Extract pair name
            pair = self._extract_pair_from_epic(epic)

            # Get current timestamp - handle various index types
            current_time = self._get_current_timestamp(df)
            if current_time is None:
                self.logger.warning(f"[{pair}] Could not determine timestamp from DataFrame")
                return None

            # Check daily reset
            self.phase_tracker.check_daily_reset(pair, current_time.date())

            # Check for phase timeouts
            timeout_reason = self.phase_tracker.check_timeouts(pair, current_time)
            if timeout_reason:
                self.logger.debug(f"[{pair}] {timeout_reason}")
                self.phase_tracker.reset_pair(pair, keep_date=True)

            # Get current phase
            current_phase = self.phase_tracker.get_phase(pair)

            # Process based on current phase
            if current_phase == AMDPhase.WAITING:
                # Look for accumulation
                self._check_accumulation(df, epic, pair, current_time)

            elif current_phase == AMDPhase.ACCUMULATION:
                # Look for manipulation
                self._check_manipulation(df, epic, pair, current_time)

            elif current_phase == AMDPhase.MANIPULATION:
                # Look for structure shift
                self._check_structure_shift(df, epic, pair, current_time)

            elif current_phase == AMDPhase.DISTRIBUTION:
                # Look for entry
                signal = self._check_entry(df, epic, pair, spread_pips, current_time)
                if signal:
                    self.phase_tracker.mark_completed(pair)
                    return signal

            return None

        except Exception as e:
            self.logger.error(f"[{self.name}] Signal detection error for {epic}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def detect_signal_multi_timeframe(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        epic: str,
        pair: str,
        spread_pips: float = 1.5
    ) -> Optional[Dict]:
        """
        Multi-timeframe signal detection for AMD pattern.

        Uses:
        - 5m for precise accumulation range and entry timing
        - 15m for structure shift and manipulation detection

        Args:
            df_5m: 5-minute OHLCV data
            df_15m: 15-minute OHLCV data
            epic: Instrument epic code
            pair: Currency pair name
            spread_pips: Spread in pips

        Returns:
            Signal dictionary or None
        """
        try:
            self.logger.info(f"[{self.name}] Multi-TF detection called for {pair}")

            # Validate data
            if df_5m is None or len(df_5m) < 50:
                self.logger.debug(f"[{pair}] Insufficient 5m data: {len(df_5m) if df_5m is not None else 0}")
                return None
            if df_15m is None or len(df_15m) < 30:
                self.logger.debug(f"[{pair}] Insufficient 15m data: {len(df_15m) if df_15m is not None else 0}")
                return None

            self.logger.info(f"[{pair}] Data validation passed: 5m={len(df_5m)}, 15m={len(df_15m)}")

            # Get current timestamp - handle various index types
            current_time = self._get_current_timestamp(df_15m)
            if current_time is None:
                self.logger.warning(f"[{pair}] Could not determine timestamp from DataFrame")
                return None

            self.logger.info(f"[{pair}] Current timestamp: {current_time}")

            # Check daily reset
            self.phase_tracker.check_daily_reset(pair, current_time.date())

            # Check for phase timeouts
            timeout_reason = self.phase_tracker.check_timeouts(pair, current_time)
            if timeout_reason:
                self.logger.debug(f"[{pair}] {timeout_reason}")
                self.phase_tracker.reset_pair(pair, keep_date=True)

            # Get current phase
            current_phase = self.phase_tracker.get_phase(pair)
            self.logger.info(f"[{pair}] Current phase: {current_phase.value} at {current_time.strftime('%H:%M')} UTC")

            # Process based on current phase
            if current_phase == AMDPhase.WAITING:
                # Look for accumulation using 5m data for precision
                self._check_accumulation(df_5m, epic, pair, current_time)

            elif current_phase == AMDPhase.ACCUMULATION:
                # Look for manipulation using 15m data
                self._check_manipulation(df_15m, epic, pair, current_time)

            elif current_phase == AMDPhase.MANIPULATION:
                # Look for structure shift using 15m data
                self._check_structure_shift(df_15m, epic, pair, current_time)

            elif current_phase == AMDPhase.DISTRIBUTION:
                # Look for entry using 5m data for precision
                signal = self._check_entry(df_5m, epic, pair, spread_pips, current_time)
                if signal:
                    self.phase_tracker.mark_completed(pair)
                    return signal

            return None

        except Exception as e:
            self.logger.error(f"[{self.name}] Multi-TF signal detection error: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    # =========================================================================
    # PHASE 1: ACCUMULATION DETECTION
    # =========================================================================

    def _check_accumulation(
        self,
        df: pd.DataFrame,
        epic: str,
        pair: str,
        current_time: datetime
    ) -> bool:
        """
        Check for accumulation (Asian session consolidation).

        Criteria:
        1. Current time is in Asian session (00:00-08:00 UTC)
        2. Price is in tight range (< max_range_pips)
        3. ATR is compressed (< 60% of baseline)
        4. Volume is declining (optional)
        """
        try:
            # Only check during Asian session
            is_asian = self.session_analyzer.is_asian_session(current_time)
            self.logger.info(f"[{pair}] Checking accumulation at {current_time.strftime('%H:%M')} UTC - is_asian={is_asian}")

            if not is_asian:
                return False

            # Get Asian session range
            asian_range = self.session_analyzer.get_asian_session_range(df, current_time.date())
            if asian_range is None:
                self.logger.info(f"[{pair}] No Asian range data found")
                return False

            self.logger.info(f"[{pair}] Asian range: candles={asian_range['candle_count']}, high={asian_range['high']:.5f}, low={asian_range['low']:.5f}")

            # Check minimum candles
            if asian_range['candle_count'] < self.min_accumulation_candles:
                self.logger.info(f"[{pair}] Not enough accumulation candles: {asian_range['candle_count']} < {self.min_accumulation_candles}")
                return False

            # Get pip value for range calculation
            pip_value = self.get_pip_value(epic)
            range_pips = asian_range['range'] / pip_value

            # Check range is not too wide
            max_range = self._get_pair_max_range(pair)
            if range_pips > max_range:
                self.logger.info(f"[{pair}] Accumulation range too wide: {range_pips:.1f} > {max_range}")
                return False

            # Check ATR compression
            atr_ratio = self._calculate_atr_ratio(df)
            threshold = self._get_pair_atr_threshold(pair)

            self.logger.info(f"[{pair}] ATR ratio: {atr_ratio:.2f}, threshold: {threshold}")

            if atr_ratio > threshold:
                self.logger.info(f"[{pair}] ATR not compressed: {atr_ratio:.2f} > {threshold}")
                return False

            # Check volume profile (optional)
            volume_valid = True
            if self.validate_accumulation_volume:
                volume_valid = self._check_volume_declining(df, asian_range)

            # Get daily open for bias
            daily_open = asian_range.get('daily_open', asian_range['midpoint'])

            # Create accumulation zone
            accumulation = AccumulationZone(
                range_high=asian_range['high'],
                range_low=asian_range['low'],
                daily_open=daily_open,
                candle_count=asian_range['candle_count'],
                start_time=asian_range['start_time'],
                end_time=asian_range['end_time'],
                atr_ratio=atr_ratio,
                volume_profile_valid=volume_valid
            )

            # Update phase tracker
            return self.phase_tracker.update_accumulation(pair, accumulation, current_time)

        except Exception as e:
            self.logger.error(f"[{pair}] Accumulation check error: {e}")
            return False

    def _calculate_atr_ratio(self, df: pd.DataFrame, period: int = 14, baseline_period: int = 100) -> float:
        """Calculate current ATR ratio vs baseline."""
        try:
            if 'atr' in df.columns and not df['atr'].isna().all():
                current_atr = df['atr'].iloc[-1]
                baseline_atr = df['atr'].iloc[-baseline_period:].mean() if len(df) >= baseline_period else df['atr'].mean()
                return current_atr / baseline_atr if baseline_atr > 0 else 1.0

            # Calculate ATR manually
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)

            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            current_atr = tr.rolling(window=period).mean().iloc[-1]
            baseline_atr = tr.rolling(window=period).mean().iloc[-baseline_period:].mean() if len(df) >= baseline_period else tr.mean()

            return current_atr / baseline_atr if baseline_atr > 0 else 1.0

        except Exception as e:
            self.logger.debug(f"ATR ratio calculation error: {e}")
            return 1.0

    def _check_volume_declining(self, df: pd.DataFrame, asian_range: Dict) -> bool:
        """Check if volume declined during accumulation."""
        try:
            if 'volume' not in df.columns:
                return True  # Skip validation if no volume

            # Get Asian session data
            asian_data = self.session_analyzer.get_session_data(df, 'asian')
            if len(asian_data) < 4:
                return True

            # Compare first half vs second half volume
            midpoint = len(asian_data) // 2
            first_half_vol = asian_data['volume'].iloc[:midpoint].mean()
            second_half_vol = asian_data['volume'].iloc[midpoint:].mean()

            # Volume should not increase significantly
            ratio = getattr(config, 'ACCUMULATION_VOLUME_DECLINE_RATIO', 1.15) if config else 1.15
            return second_half_vol < first_half_vol * ratio

        except Exception as e:
            self.logger.debug(f"Volume profile check error: {e}")
            return True

    def _get_pair_max_range(self, pair: str) -> float:
        """Get maximum accumulation range for pair."""
        if config and hasattr(config, 'get_pair_calibration'):
            calibration = config.get_pair_calibration(pair)
            return calibration.get('max_range_pips', self.max_accumulation_range_pips)
        return self.max_accumulation_range_pips

    def _get_pair_atr_threshold(self, pair: str) -> float:
        """Get ATR compression threshold for pair."""
        if config and hasattr(config, 'get_pair_calibration'):
            calibration = config.get_pair_calibration(pair)
            return calibration.get('atr_compression', self.atr_compression_threshold)
        return self.atr_compression_threshold

    # =========================================================================
    # PHASE 2: MANIPULATION DETECTION (JUDAS SWING)
    # =========================================================================

    def _check_manipulation(
        self,
        df: pd.DataFrame,
        epic: str,
        pair: str,
        current_time: datetime
    ) -> bool:
        """
        Check for manipulation (Judas swing / liquidity sweep).

        Criteria:
        1. Time is in London open window (08:00-10:00 UTC)
        2. Price sweeps beyond accumulation range
        3. Strong rejection (wick > 65% of candle)
        4. Volume spike on sweep candle
        """
        try:
            # Only check during London open
            is_london = self.session_analyzer.is_london_open(current_time)
            self.logger.info(f"[{pair}] Checking manipulation at {current_time.strftime('%H:%M')} UTC - is_london_open={is_london}")

            if not is_london:
                return False

            # Get state with accumulation data
            state = self.phase_tracker.get_state(pair)
            if state.accumulation is None:
                self.logger.info(f"[{pair}] No accumulation data in state")
                return False

            self.logger.info(f"[{pair}] Accumulation range: {state.accumulation.range_low:.5f} - {state.accumulation.range_high:.5f}")

            acc = state.accumulation
            pip_value = self.get_pip_value(epic)

            # Get recent candles for sweep detection
            london_data = self.session_analyzer.get_session_data(df, 'london_open', current_time.date())
            self.logger.info(f"[{pair}] London candles found: {len(london_data)}")
            if len(london_data) == 0:
                return False

            # Look for sweep in recent candles
            self.logger.info(f"[{pair}] Checking for sweeps: acc_low={acc.range_low:.5f}, acc_high={acc.range_high:.5f}")
            for i in range(len(london_data)):
                candle = london_data.iloc[i]
                candle_time = london_data.index[i]
                if isinstance(candle_time, pd.Timestamp):
                    candle_time = candle_time.to_pydatetime()

                self.logger.info(f"[{pair}] London candle {i}: low={candle['low']:.5f}, high={candle['high']:.5f}")

                # Check for sweep below lows (bullish setup)
                if candle['low'] < acc.range_low:
                    sweep_pips = (acc.range_low - candle['low']) / pip_value
                    self.logger.info(f"[{pair}] Potential SWEEP LOWS detected: candle_low={candle['low']:.5f}, sweep={sweep_pips:.1f} pips")

                    if self._validate_sweep(candle, df, sweep_pips, 'sweep_lows'):
                        manipulation = ManipulationEvent(
                            sweep_direction=SweepDirection.SWEEP_LOWS,
                            sweep_level=candle['low'],
                            sweep_candle_high=candle['high'],
                            sweep_candle_low=candle['low'],
                            sweep_candle_close=candle['close'],
                            sweep_candle_volume=candle.get('volume', 0),
                            timestamp=candle_time,
                            volume_ratio=self._calculate_volume_ratio(df, i),
                            rejection_wick_ratio=self._calculate_wick_ratio(candle, 'lower'),
                            extension_pips=sweep_pips
                        )
                        return self.phase_tracker.update_manipulation(pair, manipulation, current_time)

                # Check for sweep above highs (bearish setup)
                if candle['high'] > acc.range_high:
                    sweep_pips = (candle['high'] - acc.range_high) / pip_value

                    if self._validate_sweep(candle, df, sweep_pips, 'sweep_highs'):
                        manipulation = ManipulationEvent(
                            sweep_direction=SweepDirection.SWEEP_HIGHS,
                            sweep_level=candle['high'],
                            sweep_candle_high=candle['high'],
                            sweep_candle_low=candle['low'],
                            sweep_candle_close=candle['close'],
                            sweep_candle_volume=candle.get('volume', 0),
                            timestamp=candle_time,
                            volume_ratio=self._calculate_volume_ratio(df, i),
                            rejection_wick_ratio=self._calculate_wick_ratio(candle, 'upper'),
                            extension_pips=sweep_pips
                        )
                        return self.phase_tracker.update_manipulation(pair, manipulation, current_time)

            return False

        except Exception as e:
            self.logger.error(f"[{pair}] Manipulation check error: {e}")
            return False

    def _validate_sweep(
        self,
        candle: pd.Series,
        df: pd.DataFrame,
        sweep_pips: float,
        sweep_type: str
    ) -> bool:
        """Validate sweep meets criteria."""
        # Check sweep extension is within bounds
        if sweep_pips < self.min_sweep_pips or sweep_pips > self.max_sweep_pips:
            return False

        # Check rejection wick
        wick_type = 'lower' if sweep_type == 'sweep_lows' else 'upper'
        wick_ratio = self._calculate_wick_ratio(candle, wick_type)
        if wick_ratio < self.min_rejection_wick_ratio:
            return False

        # Check price rejected back into range
        if sweep_type == 'sweep_lows':
            # For bullish sweep, close should be above the low
            if candle['close'] <= candle['low']:
                return False
        else:
            # For bearish sweep, close should be below the high
            if candle['close'] >= candle['high']:
                return False

        return True

    def _calculate_wick_ratio(self, candle: pd.Series, wick_type: str) -> float:
        """Calculate wick ratio (wick size / total candle range)."""
        try:
            candle_range = candle['high'] - candle['low']
            if candle_range <= 0:
                return 0.0

            body_high = max(candle['open'], candle['close'])
            body_low = min(candle['open'], candle['close'])

            if wick_type == 'lower':
                wick = body_low - candle['low']
            else:  # upper
                wick = candle['high'] - body_high

            return wick / candle_range

        except Exception:
            return 0.0

    def _calculate_volume_ratio(self, df: pd.DataFrame, candle_index: int, lookback: int = 20) -> float:
        """Calculate volume ratio vs average."""
        try:
            if 'volume' not in df.columns:
                return 1.0

            current_volume = df.iloc[candle_index]['volume']
            avg_volume = df['volume'].iloc[max(0, candle_index - lookback):candle_index].mean()

            return current_volume / avg_volume if avg_volume > 0 else 1.0

        except Exception:
            return 1.0

    # =========================================================================
    # PHASE 3: STRUCTURE SHIFT DETECTION
    # =========================================================================

    def _check_structure_shift(
        self,
        df: pd.DataFrame,
        epic: str,
        pair: str,
        current_time: datetime
    ) -> bool:
        """
        Check for structure shift (BOS or ChoCH) after manipulation.

        Criteria:
        1. BOS or ChoCH detected in direction matching sweep
        2. Break candle has good volume
        3. Body close beyond structure level
        """
        try:
            self.logger.info(f"[{pair}] 🔍 Checking structure shift at {current_time.strftime('%H:%M')} UTC")

            if not self.require_structure_shift:
                # If structure shift not required, skip to entry
                state = self.phase_tracker.get_state(pair)
                if state.manipulation:
                    # Create placeholder structure shift
                    direction = 'bullish' if state.manipulation.sweep_direction == SweepDirection.SWEEP_LOWS else 'bearish'
                    structure_shift = StructureShiftEvent(
                        direction=direction,
                        break_type='IMPLIED',
                        break_candle_index=len(df) - 1,
                        break_price=df.iloc[-1]['close'],
                        timestamp=current_time
                    )
                    return self.phase_tracker.update_structure_shift(pair, structure_shift, current_time)
                return False

            if self.smc_structure is None:
                self.logger.warning(f"[{pair}] SMC Market Structure helper not available")
                return False

            # Get state
            state = self.phase_tracker.get_state(pair)
            if state.manipulation is None:
                return False

            # Expected direction based on manipulation
            expected_direction = 'bullish' if state.manipulation.sweep_direction == SweepDirection.SWEEP_LOWS else 'bearish'

            # Analyze structure using SMC helper
            smc_config = {
                'swing_length': 5,
                'structure_lookback': 20,
            }

            df_analyzed = self.smc_structure.analyze_market_structure(df.copy(), smc_config, epic)

            # Log structure breaks found
            num_breaks = len(self.smc_structure.structure_breaks) if self.smc_structure.structure_breaks else 0
            self.logger.info(f"[{pair}] SMC found {num_breaks} structure breaks, expected direction: {expected_direction}")

            # Look for structure breaks after manipulation
            manipulation_time = state.manipulation.timestamp
            if isinstance(manipulation_time, datetime):
                manipulation_time = pd.Timestamp(manipulation_time)

            # Determine if DataFrame index is tz-aware
            df_tz = None
            if isinstance(df.index, pd.DatetimeIndex):
                df_tz = df.index.tz

            # Ensure manipulation_time is tz-aware if needed
            if manipulation_time.tz is None and df_tz is not None:
                manipulation_time = manipulation_time.tz_localize('UTC')
            elif manipulation_time.tz is not None and df_tz is None:
                manipulation_time = manipulation_time.tz_localize(None)

            # Get structure breaks from the analyzer
            matching_breaks = 0
            for structure_break in self.smc_structure.structure_breaks:
                # Normalize structure break timestamp for comparison
                break_ts = structure_break.timestamp
                if isinstance(break_ts, pd.Timestamp):
                    if break_ts.tz is None and manipulation_time.tz is not None:
                        break_ts = break_ts.tz_localize('UTC')
                    elif break_ts.tz is not None and manipulation_time.tz is None:
                        break_ts = break_ts.tz_localize(None)

                # Check if break is after manipulation
                if break_ts < manipulation_time:
                    self.logger.info(f"[{pair}] Structure break at {break_ts} is BEFORE manipulation at {manipulation_time}, skipping")
                    continue

                # Check direction matches
                if structure_break.direction != expected_direction:
                    self.logger.info(f"[{pair}] Structure break direction {structure_break.direction} != expected {expected_direction}, skipping")
                    continue

                matching_breaks += 1
                self.logger.info(f"[{pair}] Found potential structure break #{matching_breaks}: {structure_break.break_type} {structure_break.direction}")

                # Calculate actual volume and body ratios for structure shift quality
                # Note: structure_break.index might be a positional index into the analyzed data
                break_idx = structure_break.index
                if break_idx >= len(df):
                    break_idx = len(df) - 1

                break_candle = df.iloc[break_idx]

                # Calculate volume ratio vs average
                # Note: Forex data from IG Markets often has no/minimal volume data
                volume_ratio = 1.0
                has_valid_volume = False
                if 'volume' in df.columns:
                    # Check if volume data is meaningful (not all zeros or constant)
                    volume_data = df['volume'].iloc[max(0, break_idx-20):break_idx]
                    if volume_data.std() > 0:  # Volume has variation
                        avg_volume = volume_data.mean()
                        if avg_volume > 0:
                            volume_ratio = break_candle['volume'] / avg_volume
                            has_valid_volume = True

                # If no valid volume data, use candle range as proxy for conviction
                if not has_valid_volume:
                    # Use price range as volume proxy - larger moves = more conviction
                    recent_ranges = (df['high'].iloc[max(0, break_idx-20):break_idx] -
                                   df['low'].iloc[max(0, break_idx-20):break_idx])
                    avg_range = recent_ranges.mean()
                    current_range = break_candle['high'] - break_candle['low']
                    if avg_range > 0:
                        volume_ratio = current_range / avg_range
                    self.logger.info(f"[{pair}] Using range-based volume proxy: {volume_ratio:.2f}x")

                # Calculate body ratio (body size / total range)
                candle_range = break_candle['high'] - break_candle['low']
                body_size = abs(break_candle['close'] - break_candle['open'])
                body_ratio = body_size / candle_range if candle_range > 0 else 0.5

                # Validate structure shift quality
                min_volume_ratio = getattr(config, 'MIN_BOS_VOLUME_MULTIPLIER', 1.4) if config else 1.4
                min_body_ratio = getattr(config, 'MIN_BOS_CANDLE_BODY_RATIO', 0.60) if config else 0.60

                # Check if structure shift meets quality requirements
                if volume_ratio < min_volume_ratio:
                    self.logger.info(f"[{pair}] ❌ Structure shift volume too low: {volume_ratio:.2f}x < {min_volume_ratio}x required")
                    continue  # Check next structure break

                if body_ratio < min_body_ratio:
                    self.logger.info(f"[{pair}] ❌ Structure shift body ratio too low: {body_ratio:.1%} < {min_body_ratio:.0%} required")
                    continue  # Check next structure break

                self.logger.info(f"[{pair}] ✅ Valid structure shift: {structure_break.break_type} {structure_break.direction} "
                               f"(volume: {volume_ratio:.2f}x, body: {body_ratio:.1%})")

                # Found valid structure shift with quality confirmation
                structure_shift = StructureShiftEvent(
                    direction=structure_break.direction,
                    break_type=structure_break.break_type,
                    break_candle_index=structure_break.index,
                    break_price=structure_break.break_price,
                    timestamp=structure_break.timestamp,
                    volume_ratio=volume_ratio,
                    body_ratio=body_ratio
                )

                return self.phase_tracker.update_structure_shift(pair, structure_shift, current_time)

            self.logger.info(f"[{pair}] ⚠️ No valid structure shift found after {matching_breaks} potential breaks checked")
            return False

        except Exception as e:
            self.logger.error(f"[{pair}] Structure shift check error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    # =========================================================================
    # PHASE 4: ENTRY DETECTION
    # =========================================================================

    def _check_entry(
        self,
        df: pd.DataFrame,
        epic: str,
        pair: str,
        spread_pips: float,
        current_time: datetime
    ) -> Optional[Dict]:
        """
        Check for entry conditions and generate signal.

        Criteria:
        1. Price has pulled back to entry zone (FVG or mitigation)
        2. Entry confirmation candle pattern (optional)
        3. R:R meets minimum requirement
        4. Confidence meets threshold
        """
        try:
            self.logger.info(f"[{pair}] Checking entry conditions at {current_time.strftime('%H:%M')} UTC")

            # Check entry cutoff time (use config value, default 16:00 UTC)
            entry_cutoff = getattr(config, 'ENTRY_CUTOFF_TIME', time(16, 0)) if config else time(16, 0)
            if not self.session_analyzer.is_entry_allowed(current_time, cutoff=entry_cutoff):
                self.logger.info(f"[{pair}] Entry time past cutoff ({entry_cutoff.strftime('%H:%M')} UTC)")
                return None

            # Get state
            state = self.phase_tracker.get_state(pair)
            if not self.phase_tracker.is_ready_for_signal(pair) and state.entry_zone is None:
                # Try to find entry zone
                entry_zone = self._find_entry_zone(df, epic, pair, state)
                if entry_zone:
                    self.phase_tracker.update_entry_zone(pair, entry_zone, current_time)

            # Refresh state
            state = self.phase_tracker.get_state(pair)
            if state.entry_zone is None:
                return None

            # Get current price
            latest = df.iloc[-1]
            current_price = latest['close']
            pip_value = self.get_pip_value(epic)

            self.logger.info(f"[{pair}] Entry zone: {state.entry_zone.zone_low:.5f} - {state.entry_zone.zone_high:.5f}, current price: {current_price:.5f}")

            # Determine expected direction from structure shift
            direction = state.get_signal_direction()

            # Check if price is in or approaching entry zone
            # For BEAR: we want to sell when price pulls UP into/near the zone
            # For BULL: we want to buy when price pulls DOWN into/near the zone
            zone_buffer = 5 * pip_value  # 5 pip buffer for entry zone

            in_zone = state.entry_zone.zone_low <= current_price <= state.entry_zone.zone_high
            at_zone_for_bear = direction == 'BEAR' and current_price >= state.entry_zone.zone_low - zone_buffer
            at_zone_for_bull = direction == 'BULL' and current_price <= state.entry_zone.zone_high + zone_buffer

            if not (in_zone or at_zone_for_bear or at_zone_for_bull):
                self.logger.info(f"[{pair}] Price not in entry zone (direction={direction})")
                return None

            self.logger.info(f"[{pair}] Price acceptable for {direction} entry (in_zone={in_zone}, at_zone_bear={at_zone_for_bear}, at_zone_bull={at_zone_for_bull})")

            # Calculate SL and TP
            direction = state.get_signal_direction()
            sl, tp = self._calculate_sl_tp(state, current_price, pip_value, df)

            self.logger.info(f"[{pair}] PRICE IN ZONE! direction={direction}, SL={sl:.5f}, TP={tp:.5f}")

            # Check R:R
            risk_pips = abs(current_price - sl) / pip_value
            reward_pips = abs(tp - current_price) / pip_value
            rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0

            self.logger.info(f"[{pair}] R:R calculation: risk={risk_pips:.1f}, reward={reward_pips:.1f}, ratio={rr_ratio:.2f}")

            if rr_ratio < self.min_rr_ratio:
                self.logger.info(f"[{pair}] R:R too low: {rr_ratio:.2f} < {self.min_rr_ratio}")
                return None

            # Calculate confidence
            confidence = self.phase_tracker.calculate_confidence(pair, rr_ratio)

            # Check confidence threshold
            min_conf = self._get_pair_min_confidence(pair)
            if confidence < min_conf:
                self.logger.info(f"[{pair}] Confidence too low: {confidence:.2%} < {min_conf:.2%}")
                return None

            # Generate signal
            signal = self._create_signal(
                epic=epic,
                pair=pair,
                direction=direction,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                risk_pips=risk_pips,
                reward_pips=reward_pips,
                rr_ratio=rr_ratio,
                confidence=confidence,
                state=state,
                latest_row=latest,
                spread_pips=spread_pips
            )

            self.logger.info(
                f"[{pair}] Signal generated: {direction}, "
                f"entry={current_price:.5f}, SL={sl:.5f}, TP={tp:.5f}, "
                f"R:R={rr_ratio:.2f}, conf={confidence:.2%}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[{pair}] Entry check error: {e}")
            return None

    def _find_entry_zone(
        self,
        df: pd.DataFrame,
        epic: str,
        pair: str,
        state: AMDState
    ) -> Optional[EntryZone]:
        """Find entry zone (FVG or mitigation)."""
        try:
            if state.structure_shift is None:
                return None

            direction = state.get_signal_direction()
            current_price = df.iloc[-1]['close']
            pip_value = self.get_pip_value(epic)

            # Try to find FVG first
            if self.fvg_detector:
                fvg_config = {
                    'min_gap_size': self.min_fvg_size_pips * pip_value,
                    'lookback_bars': 20,
                }

                try:
                    fvgs = self.fvg_detector.detect_fair_value_gaps(df, fvg_config)
                    if fvgs:
                        # Find FVG in correct direction
                        for fvg in fvgs:
                            if direction == 'BULL' and fvg.get('type') == 'bullish':
                                return EntryZone(
                                    zone_type='FVG',
                                    zone_high=fvg['high'],
                                    zone_low=fvg['low'],
                                    significance=fvg.get('significance', 1.0),
                                    timestamp=fvg.get('timestamp')
                                )
                            elif direction == 'BEAR' and fvg.get('type') == 'bearish':
                                return EntryZone(
                                    zone_type='FVG',
                                    zone_high=fvg['high'],
                                    zone_low=fvg['low'],
                                    significance=fvg.get('significance', 1.0),
                                    timestamp=fvg.get('timestamp')
                                )
                except Exception as fvg_error:
                    self.logger.debug(f"FVG detection error: {fvg_error}")

            # Fallback: Use mitigation zone (50% of manipulation candle)
            if state.manipulation:
                manip = state.manipulation
                candle_mid = (manip.sweep_candle_high + manip.sweep_candle_low) / 2
                zone_width = self.sl_buffer_pips * pip_value * 2

                return EntryZone(
                    zone_type='MITIGATION',
                    zone_high=candle_mid + zone_width,
                    zone_low=candle_mid - zone_width,
                    significance=0.6,  # Lower significance for mitigation
                    timestamp=state.manipulation.timestamp
                )

            return None

        except Exception as e:
            self.logger.debug(f"Entry zone detection error: {e}")
            return None

    def _calculate_sl_tp(
        self,
        state: AMDState,
        entry_price: float,
        pip_value: float,
        df: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit with minimum stop distance validation."""
        direction = state.get_signal_direction()
        manip = state.manipulation

        # Minimum stop distance (15 pips for standard pairs, 30 for JPY crosses)
        # This prevents tight stops that get easily stopped out by noise
        min_stop_pips = getattr(config, 'MIN_STOP_DISTANCE_PIPS', 15) if config else 15

        # Calculate ATR for dynamic buffer
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
        buffer = self.sl_buffer_pips * pip_value
        if atr:
            buffer = max(buffer, atr * 0.5)

        if direction == 'BULL':
            # SL below manipulation sweep level
            sl = manip.sweep_level - buffer
            # Ensure SL is below entry (validate direction)
            if sl >= entry_price:
                # Use minimum stop distance instead
                sl = entry_price - (min_stop_pips * pip_value)
                self.logger.warning(f"SL above entry for BULL, using min stop: {sl:.5f}")

            # Ensure minimum stop distance
            risk_pips = (entry_price - sl) / pip_value
            if risk_pips < min_stop_pips:
                sl = entry_price - (min_stop_pips * pip_value)
                self.logger.info(f"Adjusted SL to meet min stop: {min_stop_pips} pips")

            # TP at R:R based
            risk = entry_price - sl
            tp = entry_price + (risk * self.min_rr_ratio)
        else:
            # SL above manipulation sweep level
            sl = manip.sweep_level + buffer
            # Ensure SL is above entry (validate direction)
            if sl <= entry_price:
                # Use minimum stop distance instead
                sl = entry_price + (min_stop_pips * pip_value)
                self.logger.warning(f"SL below entry for BEAR, using min stop: {sl:.5f}")

            # Ensure minimum stop distance
            risk_pips = (sl - entry_price) / pip_value
            if risk_pips < min_stop_pips:
                sl = entry_price + (min_stop_pips * pip_value)
                self.logger.info(f"Adjusted SL to meet min stop: {min_stop_pips} pips")

            # TP at R:R based
            risk = sl - entry_price
            tp = entry_price - (risk * self.min_rr_ratio)

        return sl, tp

    def _get_pair_min_confidence(self, pair: str) -> float:
        """Get minimum confidence for pair."""
        if config and hasattr(config, 'get_pair_min_confidence'):
            return config.get_pair_min_confidence(pair)
        return self.min_confidence

    def _create_signal(
        self,
        epic: str,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_pips: float,
        reward_pips: float,
        rr_ratio: float,
        confidence: float,
        state: AMDState,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Dict:
        """Create signal dictionary."""

        # Calculate stop and limit distances in pips
        pip_value = self.get_pip_value(epic)
        stop_distance = int(abs(entry_price - stop_loss) / pip_value)
        limit_distance = int(abs(take_profit - entry_price) / pip_value)

        signal = {
            'signal_type': direction,
            'epic': epic,
            'pair': pair,
            'timeframe': '15m',
            'timestamp': latest_row.name if hasattr(latest_row, 'name') else datetime.utcnow(),
            'price': entry_price,
            'strategy': self.name,
            'confidence': confidence,
            'confidence_score': confidence,

            # Risk management
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_distance': stop_distance,
            'limit_distance': limit_distance,
            'risk_pips': risk_pips,
            'reward_pips': reward_pips,
            'rr_ratio': rr_ratio,

            # AMD metadata
            'strategy_metadata': {
                'pattern': 'ICT_POWER_OF_3',
                'accumulation': {
                    'range_high': state.accumulation.range_high if state.accumulation else None,
                    'range_low': state.accumulation.range_low if state.accumulation else None,
                    'daily_open': state.accumulation.daily_open if state.accumulation else None,
                    'atr_ratio': state.accumulation.atr_ratio if state.accumulation else None,
                    'candle_count': state.accumulation.candle_count if state.accumulation else None,
                },
                'manipulation': {
                    'sweep_direction': state.manipulation.sweep_direction.value if state.manipulation else None,
                    'sweep_level': state.manipulation.sweep_level if state.manipulation else None,
                    'volume_ratio': state.manipulation.volume_ratio if state.manipulation else None,
                    'rejection_wick_ratio': state.manipulation.rejection_wick_ratio if state.manipulation else None,
                },
                'structure_shift': {
                    'type': state.structure_shift.break_type if state.structure_shift else None,
                    'direction': state.structure_shift.direction if state.structure_shift else None,
                },
                'entry_zone': {
                    'type': state.entry_zone.zone_type if state.entry_zone else None,
                    'zone_high': state.entry_zone.zone_high if state.entry_zone else None,
                    'zone_low': state.entry_zone.zone_low if state.entry_zone else None,
                },
                'version': self.version,
            },

            # Trigger reason
            'trigger_reason': (
                f"ICT Power of 3: {state.manipulation.sweep_direction.value if state.manipulation else 'unknown'} "
                f"→ {state.structure_shift.break_type if state.structure_shift else 'N/A'} "
                f"→ {state.entry_zone.zone_type if state.entry_zone else 'N/A'} entry"
            ),
        }

        # Add execution prices
        signal = self.add_execution_prices(signal, spread_pips)

        return signal

    def _get_current_timestamp(self, df: pd.DataFrame) -> Optional[datetime]:
        """
        Safely extract the current timestamp from a DataFrame.

        Handles various index types:
        - DatetimeIndex: directly extract timestamp
        - RangeIndex: look for timestamp/start_time column
        - Falls back to utcnow() if no timestamp found

        Args:
            df: DataFrame with OHLCV data

        Returns:
            datetime object or None if cannot determine
        """
        try:
            # Case 1: DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                current_time = df.index[-1]
                if isinstance(current_time, pd.Timestamp):
                    return current_time.to_pydatetime()
                return current_time

            # Case 2: Look for timestamp columns
            timestamp_cols = ['timestamp', 'start_time', 'datetime', 'time', 'date']
            for col in timestamp_cols:
                if col in df.columns:
                    ts_val = df[col].iloc[-1]
                    if isinstance(ts_val, pd.Timestamp):
                        return ts_val.to_pydatetime()
                    elif isinstance(ts_val, datetime):
                        return ts_val
                    elif isinstance(ts_val, str):
                        return pd.to_datetime(ts_val).to_pydatetime()

            # Case 3: Fallback to current UTC time (for live trading)
            self.logger.debug("No timestamp found in DataFrame, using utcnow()")
            return datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Error extracting timestamp: {e}")
            return None

    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract pair name from epic code."""
        parts = epic.upper().split('.')
        for part in parts:
            if len(part) in [6, 7] and any(curr in part for curr in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']):
                return part
        return epic
