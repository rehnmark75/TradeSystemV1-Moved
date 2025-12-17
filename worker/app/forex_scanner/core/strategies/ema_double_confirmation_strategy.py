"""
EMA Double Confirmation Strategy
================================
VERSION: 2.0.0
DATE: 2025-12-16
STATUS: Phase 2 - Limit Orders for Better Entry Timing

A trading strategy that waits for two successful EMA 21/50 crossovers before
taking the third crossover as an entry signal.

Strategy Logic:
1. Detect EMA 21/50 crossovers on 15-minute timeframe
2. Validate crossover "success" = price stays on favorable side of EMA 21
   for SUCCESS_CANDLES consecutive candles (default: 4 = 1 hour)
3. After 2 successful crossovers in SAME direction within lookback window,
   take the 3rd crossover as entry signal
4. Reset counter for that direction after trade is taken

Rationale:
- First crossover: Market testing direction
- Second crossover: Confirms market respects EMA structure
- Third crossover: High-probability entry with proven pattern

v2.0.0 CHANGES (Limit Orders):
    - NEW: Limit order support with ATR-based price offsets
    - v2.3.0: CHANGED to stop-entry style (momentum confirmation)
    - BUY orders placed ABOVE market (enter when price breaks up)
    - SELL orders placed BELOW market (enter when price breaks down)
    - Max offset: 3 pips, 6-minute auto-expiry
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.crossover_history_tracker import CrossoverHistoryTracker
from .helpers.smc_fair_value_gaps import SMCFairValueGaps, FVGType

# Import config with fallback
try:
    from configdata import config
    from configdata.strategies import config_ema_double_confirmation as strategy_config
except ImportError:
    try:
        from forex_scanner.configdata import config
        from forex_scanner.configdata.strategies import config_ema_double_confirmation as strategy_config
    except ImportError:
        config = None
        strategy_config = None


class EMADoubleConfirmationStrategy(BaseStrategy):
    """
    EMA Double Confirmation Strategy

    Waits for 2 successful EMA crossovers before taking the 3rd as entry.

    Key Features:
    - Tracks EMA 21/50 crossover patterns
    - Validates crossover success (price stays favorable for N candles)
    - Generates signal on 3rd crossover after 2 successful ones
    - Configurable parameters via config file
    """

    def __init__(
        self,
        data_fetcher=None,
        backtest_mode: bool = False,
        epic: str = None,
        pipeline_mode: bool = True,
        db_manager=None
    ):
        """
        Initialize the EMA Double Confirmation Strategy.

        Args:
            data_fetcher: DataFetcher instance for historical data
            backtest_mode: Whether running in backtest mode
            epic: Optional specific epic for initialization
            pipeline_mode: Whether running in live pipeline mode
            db_manager: Optional DatabaseManager for persistent state (v2.0.0)
        """
        # Initialize base strategy
        super().__init__('ema_double_confirmation')

        self.data_fetcher = data_fetcher
        self.backtest_mode = backtest_mode
        self.epic = epic
        self.db_manager = db_manager
        self.price_adjuster = PriceAdjuster()

        # Load configuration
        self._load_config()

        # Initialize crossover tracker with optional database persistence
        # v2.0.0: Database persistence fixes "chicken and egg" problem with MIN_SUCCESSFUL_CROSSOVERS
        self.crossover_tracker = CrossoverHistoryTracker(
            success_candles=self.success_candles,
            lookback_hours=self.lookback_hours,
            min_crossovers_for_signal=self.min_successful_crossovers,
            max_validation_candles=self.max_validation_candles,
            extended_success_candles=self.extended_success_candles,
            extended_success_bonus=self.extended_success_bonus,
            ema_fast_period=self.ema_fast,
            ema_slow_period=self.ema_slow,
            logger=self.logger,
            db_manager=db_manager,
            use_database_state=self.use_database_state
        )

        # Optimal params (for base class compatibility)
        self.optimal_params = None

        self._log_initialization()

    def _load_config(self):
        """Load strategy parameters from config module"""
        # Core EMA parameters
        self.ema_fast = getattr(strategy_config, 'EMA_FAST_PERIOD', 21) if strategy_config else 21
        self.ema_slow = getattr(strategy_config, 'EMA_SLOW_PERIOD', 50) if strategy_config else 50
        self.ema_trend = getattr(strategy_config, 'EMA_TREND_PERIOD', 200) if strategy_config else 200

        # Success validation parameters
        self.success_candles = getattr(strategy_config, 'SUCCESS_CANDLES', 4) if strategy_config else 4
        self.max_validation_candles = getattr(strategy_config, 'MAX_VALIDATION_CANDLES', 6) if strategy_config else 6

        # Lookback parameters
        self.lookback_hours = getattr(strategy_config, 'LOOKBACK_HOURS', 48) if strategy_config else 48
        self.min_successful_crossovers = getattr(strategy_config, 'MIN_SUCCESSFUL_CROSSOVERS', 2) if strategy_config else 2

        # Confidence parameters
        self.min_confidence = getattr(strategy_config, 'MIN_CONFIDENCE', 0.60) if strategy_config else 0.60
        self.base_confidence = getattr(strategy_config, 'BASE_CONFIDENCE', 0.50) if strategy_config else 0.50
        self.confidence_weights = getattr(strategy_config, 'CONFIDENCE_WEIGHTS', {
            'crossover_quality': 0.30,
            'prior_success_rate': 0.25,
            'trend_alignment': 0.25,
            'market_conditions': 0.20,
        }) if strategy_config else {
            'crossover_quality': 0.30,
            'prior_success_rate': 0.25,
            'trend_alignment': 0.25,
            'market_conditions': 0.20,
        }

        # Extended success bonus
        self.extended_success_candles = getattr(strategy_config, 'EXTENDED_SUCCESS_CANDLES', 6) if strategy_config else 6
        self.extended_success_bonus = getattr(strategy_config, 'EXTENDED_SUCCESS_BONUS', 0.05) if strategy_config else 0.05

        # Risk management
        self.stop_atr_multiplier = getattr(strategy_config, 'STOP_ATR_MULTIPLIER', 2.0) if strategy_config else 2.0
        self.target_atr_multiplier = getattr(strategy_config, 'TARGET_ATR_MULTIPLIER', 4.0) if strategy_config else 4.0
        self.min_stop_pips = getattr(strategy_config, 'MIN_STOP_PIPS', 15) if strategy_config else 15
        self.max_stop_pips = getattr(strategy_config, 'MAX_STOP_PIPS', 50) if strategy_config else 50
        self.min_target_pips = getattr(strategy_config, 'MIN_TARGET_PIPS', 30) if strategy_config else 30

        # Session filter
        self.session_filter_enabled = getattr(strategy_config, 'SESSION_FILTER_ENABLED', True) if strategy_config else True

        # Higher Timeframe Trend Filter (4H EMA 50)
        self.htf_trend_filter_enabled = getattr(strategy_config, 'HTF_TREND_FILTER_ENABLED', True) if strategy_config else True
        self.htf_timeframe = getattr(strategy_config, 'HTF_TIMEFRAME', '4h') if strategy_config else '4h'
        self.htf_ema_period = getattr(strategy_config, 'HTF_EMA_PERIOD', 50) if strategy_config else 50
        self.htf_min_bars = getattr(strategy_config, 'HTF_MIN_BARS', 60) if strategy_config else 60

        # Logging
        self.debug_logging = getattr(strategy_config, 'ENABLE_DEBUG_LOGGING', True) if strategy_config else True
        self.log_crossovers = getattr(strategy_config, 'LOG_CROSSOVER_DETECTION', True) if strategy_config else True

        # FVG Confirmation Filter
        self.fvg_confirmation_enabled = getattr(strategy_config, 'FVG_CONFIRMATION_ENABLED', False) if strategy_config else False
        self.fvg_lookback_candles = getattr(strategy_config, 'FVG_LOOKBACK_CANDLES', 10) if strategy_config else 10
        self.fvg_min_size_pips = getattr(strategy_config, 'FVG_MIN_SIZE_PIPS', 2) if strategy_config else 2

        # ADX Trend Strength Filter
        self.adx_filter_enabled = getattr(strategy_config, 'ADX_FILTER_ENABLED', False) if strategy_config else False
        self.adx_min_value = getattr(strategy_config, 'ADX_MIN_VALUE', 20) if strategy_config else 20
        self.adx_period = getattr(strategy_config, 'ADX_PERIOD', 14) if strategy_config else 14

        # Minimum bars required
        self.min_bars = max(self.ema_slow + 10, 60)  # Need enough for EMA 50 + buffer

        # HTF data cache (for backtesting efficiency)
        self._htf_cache = {}

        # v2.0.0: Limit Order Configuration
        self.limit_order_enabled = getattr(strategy_config, 'LIMIT_ORDER_ENABLED', True) if strategy_config else True
        self.limit_expiry_minutes = getattr(strategy_config, 'LIMIT_EXPIRY_MINUTES', 6) if strategy_config else 6
        self.limit_offset_atr_factor = getattr(strategy_config, 'LIMIT_OFFSET_ATR_FACTOR', 0.25) if strategy_config else 0.25
        self.limit_offset_min_pips = getattr(strategy_config, 'LIMIT_OFFSET_MIN_PIPS', 2.0) if strategy_config else 2.0
        self.limit_offset_max_pips = getattr(strategy_config, 'LIMIT_OFFSET_MAX_PIPS', 6.0) if strategy_config else 6.0
        self.min_risk_after_offset_pips = getattr(strategy_config, 'MIN_RISK_AFTER_OFFSET_PIPS', 5.0) if strategy_config else 5.0
        self.max_risk_after_offset_pips = getattr(strategy_config, 'MAX_RISK_AFTER_OFFSET_PIPS', 45.0) if strategy_config else 45.0

        # v2.0.0: Database Persistence for Crossover State
        # Fixes "chicken and egg" problem with MIN_SUCCESSFUL_CROSSOVERS requirement
        self.use_database_state = getattr(strategy_config, 'USE_DATABASE_STATE', False) if strategy_config else False

    def _log_initialization(self):
        """Log strategy initialization details"""
        self.logger.info("=" * 60)
        self.logger.info("EMA Double Confirmation Strategy initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"  EMA Periods: {self.ema_fast}/{self.ema_slow}/{self.ema_trend}")
        self.logger.info(f"  Success validation: {self.success_candles} candles")
        self.logger.info(f"  Lookback window: {self.lookback_hours} hours")
        self.logger.info(f"  Required crossovers: {self.min_successful_crossovers}")
        self.logger.info(f"  Min confidence: {self.min_confidence:.0%}")
        self.logger.info(f"  HTF Trend Filter: {self.htf_timeframe} EMA {self.htf_ema_period} ({'enabled' if self.htf_trend_filter_enabled else 'disabled'})")
        self.logger.info(f"  FVG Confirmation: {'enabled' if self.fvg_confirmation_enabled else 'disabled'} (lookback={self.fvg_lookback_candles}, min_size={self.fvg_min_size_pips}pips)")
        self.logger.info(f"  ADX Filter: {'enabled' if self.adx_filter_enabled else 'disabled'} (min={self.adx_min_value})")
        self.logger.info(f"  Database State: {'ENABLED' if self.use_database_state else 'DISABLED'} (persistence for crossover history)")
        self.logger.info(f"  Backtest mode: {self.backtest_mode}")
        self.logger.info("=" * 60)

    def get_required_indicators(self) -> List[str]:
        """Return list of required indicators for this strategy"""
        return [
            'close', 'open', 'high', 'low', 'start_time',
            f'ema_{self.ema_fast}', f'ema_{self.ema_slow}',
            'atr', 'rsi'
        ]

    def detect_signal(
        self,
        df: pd.DataFrame,
        epic: str,
        spread_pips: float = 1.5,
        timeframe: str = '15m',
        evaluation_time: datetime = None
    ) -> Optional[Dict]:
        """
        Main signal detection method.

        Flow:
        1. Ensure EMAs are calculated
        2. Validate any pending crossovers from previous candles
        3. Detect new crossover on current candle
        4. Check if entry conditions met (2 prior successful crossovers)
        5. Generate signal if 3rd crossover after 2 successful ones

        Args:
            df: DataFrame with OHLC and indicator data
            epic: Currency pair identifier
            spread_pips: Spread in pips for execution
            timeframe: Timeframe being analyzed
            evaluation_time: Optional specific evaluation time

        Returns:
            Signal dictionary if conditions met, None otherwise
        """
        try:
            log_prefix = f"[EMA_DOUBLE] [{epic}]"

            # Validate data requirements
            if not self._validate_data(df, epic):
                return None

            # Step 1: Ensure EMAs are calculated
            df_enhanced = self._ensure_emas(df)

            # Step 2: Validate any pending crossovers
            validated = self.crossover_tracker.validate_pending_crossovers(df_enhanced, epic)
            if validated and self.log_crossovers:
                for event in validated:
                    self.logger.info(f"{log_prefix} Prior crossover validated: {event}")

            # Step 3: Detect new crossover on current candle
            crossover_event = self.crossover_tracker.detect_crossover(df_enhanced, epic)

            if crossover_event is None:
                # No new crossover - nothing to do
                return None

            direction = crossover_event.direction
            timestamp = crossover_event.timestamp

            # Only log if this is a genuinely new crossover (not a re-detection of pending)
            # The tracker handles logging for new crossovers, so we just log the state
            if self.log_crossovers:
                # Log current successful crossover counts
                state_summary = self.crossover_tracker.get_state_summary(epic)
                self.logger.info(f"{log_prefix} State: successful_BULL={state_summary.get('successful_bull_count', 0)}, successful_BEAR={state_summary.get('successful_bear_count', 0)}")

            # Step 4: Check if entry conditions met (2 prior successful crossovers)
            should_signal, confidence_boost, metadata = self.crossover_tracker.should_generate_signal(
                epic, direction, timestamp
            )

            if not should_signal:
                self.logger.info(f"{log_prefix} Not enough prior confirmations: {metadata.get('successful_count', 0)}/{self.min_successful_crossovers} required")
                return None

            # Step 5: Check session filter (skip in backtest mode)
            if not self.backtest_mode and self.session_filter_enabled:
                if not self._is_session_allowed(timestamp):
                    self.logger.info(f"{log_prefix} Signal filtered: Outside trading session")
                    return None

            # Step 5b: Check 4H EMA 50 trend filter
            if self.htf_trend_filter_enabled:
                htf_aligned, htf_details = self._check_htf_trend_filter(df_enhanced, epic, direction)
                if not htf_aligned:
                    self.logger.info(f"{log_prefix} Signal filtered: {direction} not aligned with {self.htf_timeframe} EMA {self.htf_ema_period}")
                    self.logger.info(f"  {htf_details}")
                    return None

            # Step 5c: Check FVG confirmation filter
            if self.fvg_confirmation_enabled:
                fvg_confirmed, fvg_details = self._check_fvg_confirmation(df_enhanced, epic, direction)
                if not fvg_confirmed:
                    self.logger.info(f"{log_prefix} Signal filtered: No confirming FVG for {direction}")
                    self.logger.info(f"  {fvg_details}")
                    return None

            # Step 5d: Check ADX trend strength filter
            if self.adx_filter_enabled:
                adx_ok, adx_details = self._check_adx_trend_strength(df_enhanced, epic)
                if not adx_ok:
                    self.logger.info(f"{log_prefix} Signal filtered: Market not trending")
                    self.logger.info(f"  {adx_details}")
                    return None

            # Step 6: Generate signal
            latest_row = df_enhanced.iloc[-1]
            signal = self._create_signal(
                signal_type=direction,
                epic=epic,
                timeframe=timeframe,
                latest_row=latest_row,
                spread_pips=spread_pips,
                metadata=metadata,
                confidence_boost=confidence_boost
            )

            if signal is None:
                return None

            # Check confidence threshold
            if signal.get('confidence', 0) < self.min_confidence:
                self.logger.info(f"{log_prefix} Signal rejected: Confidence {signal['confidence']:.1%} < {self.min_confidence:.1%}")
                return None

            # Record that we're taking this trade
            self.crossover_tracker.record_trade_taken(epic, direction, timestamp)

            self.logger.info(f"{log_prefix} ✅ SIGNAL GENERATED: {direction}")
            self.logger.info(f"  Confidence: {signal['confidence']:.1%}")
            self.logger.info(f"  Prior confirmations: {metadata.get('successful_count', 0)}")
            self.logger.info(f"  SL: {signal['stop_distance']} pips, TP: {signal['limit_distance']} pips")

            return signal

        except Exception as e:
            self.logger.error(f"[EMA_DOUBLE] Error in signal detection: {e}", exc_info=True)
            return None

    def _validate_data(self, df: pd.DataFrame, epic: str) -> bool:
        """Validate DataFrame has required data"""
        if df is None:
            self.logger.warning(f"[{epic}] DataFrame is None")
            return False

        if len(df) < self.min_bars:
            self.logger.warning(f"[{epic}] Insufficient data: {len(df)} < {self.min_bars} bars")
            return False

        # Check for required columns
        required = ['close', 'high', 'low']
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.logger.warning(f"[{epic}] Missing columns: {missing}")
            return False

        return True

    def _ensure_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs if not present in DataFrame"""
        df = df.copy()

        # Define column names
        ema_fast_col = f'ema_{self.ema_fast}'
        ema_slow_col = f'ema_{self.ema_slow}'
        ema_trend_col = f'ema_{self.ema_trend}'

        # Calculate EMAs if missing
        if ema_fast_col not in df.columns:
            df[ema_fast_col] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()

        if ema_slow_col not in df.columns:
            df[ema_slow_col] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        if ema_trend_col not in df.columns and self.ema_trend:
            df[ema_trend_col] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()

        # Add generic names for compatibility with tracker
        df['ema_fast'] = df[ema_fast_col]
        df['ema_slow'] = df[ema_slow_col]
        if ema_trend_col in df.columns:
            df['ema_trend'] = df[ema_trend_col]

        return df

    def _is_session_allowed(self, timestamp: datetime) -> bool:
        """Check if timestamp is within allowed trading sessions"""
        if strategy_config and hasattr(strategy_config, 'is_session_allowed'):
            hour_utc = timestamp.hour if hasattr(timestamp, 'hour') else 12
            return strategy_config.is_session_allowed(hour_utc)

        # Default: Allow London + NY sessions (07:00 - 21:00 UTC)
        hour_utc = timestamp.hour if hasattr(timestamp, 'hour') else 12
        return 7 <= hour_utc <= 21

    def _check_htf_trend_filter(
        self,
        df: pd.DataFrame,
        epic: str,
        direction: str
    ) -> tuple:
        """
        Check if signal aligns with higher timeframe trend (4H EMA 50).

        For BULL signals: Price must be ABOVE 4H EMA 50
        For BEAR signals: Price must be BELOW 4H EMA 50

        Args:
            df: 15m DataFrame with current data
            epic: Currency pair identifier
            direction: Signal direction ('BULL' or 'BEAR')

        Returns:
            Tuple of (is_aligned: bool, details: str)
        """
        try:
            # Get current price from 15m data
            current_price = df.iloc[-1]['close']

            # Get 4H EMA 50 value
            htf_ema = self._get_htf_ema_value(df, epic)

            if htf_ema is None:
                # Can't determine HTF trend - allow signal but log warning
                # In live trading, we'd always have enough data, so this mainly affects backtests
                self.logger.debug(f"[{epic}] HTF filter: EMA unavailable (insufficient data), allowing signal")
                return True, "HTF EMA unavailable, allowing signal"

            # Check alignment
            if direction == 'BULL':
                is_aligned = current_price > htf_ema
                details = f"Price {current_price:.5f} {'>' if is_aligned else '<'} 4H EMA50 {htf_ema:.5f}"
            else:  # BEAR
                is_aligned = current_price < htf_ema
                details = f"Price {current_price:.5f} {'<' if is_aligned else '>'} 4H EMA50 {htf_ema:.5f}"

            self.logger.info(f"[{epic}] HTF filter: {direction} - {details} -> {'PASS' if is_aligned else 'REJECT'}")
            return is_aligned, details

        except Exception as e:
            self.logger.error(f"Error checking HTF trend filter: {e}")
            return True, f"HTF filter error: {e}"

    def _get_htf_ema_value(self, df_15m: pd.DataFrame, epic: str) -> Optional[float]:
        """
        Get the 4H EMA value by resampling 15m data to 4H.

        This method resamples the 15m data to 4H and calculates the configured EMA period.

        Args:
            df_15m: 15m DataFrame
            epic: Currency pair identifier

        Returns:
            4H EMA value or None if insufficient data
        """
        try:
            # Need at least EMA_period * 4 bars of 15m data for reasonable 4H EMA
            # For EMA 21: 21 * 4 = 84 bars (~21 hours of 15m data)
            # For EMA 50: 50 * 4 = 200 bars (~50 hours of 15m data)
            min_15m_bars = self.htf_ema_period * 4

            if len(df_15m) < min_15m_bars:
                self.logger.debug(f"[{epic}] Insufficient 15m data for HTF EMA: {len(df_15m)} < {min_15m_bars}")
                return None

            # Create a copy with proper datetime index
            df_work = df_15m.copy()

            # Ensure we have a datetime index
            if 'start_time' in df_work.columns:
                df_work['datetime'] = pd.to_datetime(df_work['start_time'])
                df_work = df_work.set_index('datetime')
            elif not isinstance(df_work.index, pd.DatetimeIndex):
                if hasattr(df_work.index, 'to_datetime'):
                    df_work.index = pd.to_datetime(df_work.index)

            # Resample to 4H
            df_4h = df_work.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()

            if len(df_4h) < self.htf_ema_period:
                self.logger.debug(f"[{epic}] Insufficient 4H bars after resample: {len(df_4h)} < {self.htf_ema_period}")
                return None

            # Calculate EMA 50 on 4H data
            df_4h['ema_50'] = df_4h['close'].ewm(span=self.htf_ema_period, adjust=False).mean()

            # Return the most recent 4H EMA 50 value
            htf_ema = df_4h['ema_50'].iloc[-1]

            if self.debug_logging:
                self.logger.debug(f"[{epic}] 4H EMA50 = {htf_ema:.5f} (from {len(df_4h)} 4H bars)")

            return htf_ema

        except Exception as e:
            self.logger.error(f"Error calculating HTF EMA: {e}")
            return None

    def _check_fvg_confirmation(
        self,
        df: pd.DataFrame,
        epic: str,
        direction: str
    ) -> tuple:
        """
        Check if there is a Fair Value Gap (FVG) in the signal direction.

        FVGs indicate institutional imbalance and momentum commitment.
        A crossover + FVG = strong directional conviction.

        Args:
            df: 15m DataFrame with current data
            epic: Currency pair identifier
            direction: Signal direction ('BULL' or 'BEAR')

        Returns:
            Tuple of (is_confirmed: bool, details: str)
        """
        try:
            # Need at least lookback_candles + 3 for FVG detection
            if len(df) < self.fvg_lookback_candles + 3:
                self.logger.debug(f"[{epic}] FVG filter: Insufficient data")
                return True, "Insufficient data for FVG detection, allowing signal"

            # Get the last N candles for FVG detection
            df_recent = df.tail(self.fvg_lookback_candles + 5).copy()

            # Initialize FVG detector
            fvg_detector = SMCFairValueGaps(logger=self.logger)

            # Create config for FVG detection
            fvg_config = {
                'fvg_min_size': self.fvg_min_size_pips,
                'fvg_max_age': 20,
                'fvg_fill_threshold': 0.5
            }

            # Detect FVGs in recent data
            df_with_fvg = fvg_detector.detect_fair_value_gaps(df_recent, fvg_config)

            # Check for FVG matching direction in recent candles
            bullish_fvg_found = False
            bearish_fvg_found = False
            fvg_details = []

            for fvg in fvg_detector.fair_value_gaps:
                # Only check recent FVGs (within lookback window)
                if fvg.age_bars <= self.fvg_lookback_candles:
                    if fvg.gap_type == FVGType.BULLISH:
                        bullish_fvg_found = True
                        fvg_details.append(f"Bullish FVG {fvg.gap_size_pips:.1f}pips at {fvg.low_price:.5f}-{fvg.high_price:.5f}")
                    elif fvg.gap_type == FVGType.BEARISH:
                        bearish_fvg_found = True
                        fvg_details.append(f"Bearish FVG {fvg.gap_size_pips:.1f}pips at {fvg.low_price:.5f}-{fvg.high_price:.5f}")

            # Check alignment
            if direction == 'BULL':
                is_confirmed = bullish_fvg_found
                if is_confirmed:
                    details = f"Bullish FVG confirmed: {', '.join(fvg_details)}"
                else:
                    details = f"No bullish FVG in last {self.fvg_lookback_candles} candles. Found: {', '.join(fvg_details) if fvg_details else 'none'}"
            else:  # BEAR
                is_confirmed = bearish_fvg_found
                if is_confirmed:
                    details = f"Bearish FVG confirmed: {', '.join(fvg_details)}"
                else:
                    details = f"No bearish FVG in last {self.fvg_lookback_candles} candles. Found: {', '.join(fvg_details) if fvg_details else 'none'}"

            self.logger.info(f"[{epic}] FVG filter: {direction} -> {'PASS' if is_confirmed else 'REJECT'}")
            self.logger.debug(f"[{epic}] FVG details: {details}")

            return is_confirmed, details

        except Exception as e:
            self.logger.error(f"Error checking FVG confirmation: {e}")
            return True, f"FVG filter error: {e}"

    def _check_adx_trend_strength(
        self,
        df: pd.DataFrame,
        epic: str
    ) -> tuple:
        """
        Check if market is trending using ADX indicator.

        ADX > 20 indicates a trending market (better for EMA crossovers)
        ADX < 20 indicates a ranging/choppy market (avoid signals)

        Args:
            df: DataFrame with indicator data
            epic: Currency pair identifier

        Returns:
            Tuple of (is_trending: bool, details: str)
        """
        try:
            # Get ADX value from DataFrame
            latest_row = df.iloc[-1]
            adx = latest_row.get('adx')

            # If ADX not in DataFrame, try to calculate it
            if adx is None or pd.isna(adx):
                adx = self._calculate_adx(df)

            if adx is None:
                self.logger.debug(f"[{epic}] ADX filter: ADX unavailable, allowing signal")
                return True, "ADX unavailable, allowing signal"

            # Check if ADX is above threshold
            is_trending = adx >= self.adx_min_value
            details = f"ADX={adx:.1f} {'≥' if is_trending else '<'} {self.adx_min_value} threshold"

            self.logger.info(f"[{epic}] ADX filter: {details} -> {'PASS' if is_trending else 'REJECT'}")

            return is_trending, details

        except Exception as e:
            self.logger.error(f"Error checking ADX trend strength: {e}")
            return True, f"ADX filter error: {e}"

    def _calculate_adx(self, df: pd.DataFrame, period: int = None) -> Optional[float]:
        """
        Calculate ADX if not present in DataFrame.

        Args:
            df: DataFrame with OHLC data
            period: ADX period (default from config)

        Returns:
            ADX value or None if calculation fails
        """
        try:
            period = period or self.adx_period

            if len(df) < period + 1:
                return None

            # Calculate True Range
            high = df['high']
            low = df['low']
            close = df['close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            # Calculate smoothed TR, +DM, -DM
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean()

            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else None

        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return None

    def _calculate_confidence(
        self,
        latest_row: pd.Series,
        direction: str,
        metadata: Dict,
        confidence_boost: float
    ) -> float:
        """
        Calculate signal confidence based on multiple factors.

        Factors:
        1. Crossover quality (EMA separation)
        2. Prior success rate boost
        3. Trend alignment (EMA 200)
        4. Market conditions (RSI, volatility)

        Args:
            latest_row: Latest DataFrame row
            direction: Signal direction ('BULL' or 'BEAR')
            metadata: Crossover history metadata
            confidence_boost: Bonus from prior crossover performance

        Returns:
            Confidence score between 0 and 1
        """
        confidence = self.base_confidence

        try:
            # Factor 1: Crossover quality (EMA separation)
            ema_fast = latest_row.get('ema_fast', 0)
            ema_slow = latest_row.get('ema_slow', 0)

            if ema_fast and ema_slow and ema_slow != 0:
                separation = abs(ema_fast - ema_slow) / abs(ema_slow)
                # More separation = higher confidence (cap at 15% boost)
                crossover_quality = min(0.15, separation * 10)
                confidence += crossover_quality * self.confidence_weights.get('crossover_quality', 0.30)

            # Factor 2: Prior success rate boost (from tracker)
            confidence += confidence_boost * self.confidence_weights.get('prior_success_rate', 0.25)

            # Factor 3: Trend alignment (with EMA 200)
            ema_trend = latest_row.get('ema_trend') or latest_row.get(f'ema_{self.ema_trend}')
            close = latest_row.get('close', 0)

            if ema_trend and close:
                # Check if price is on the correct side of trend EMA
                if direction == 'BULL' and close > ema_trend:
                    confidence += 0.10 * self.confidence_weights.get('trend_alignment', 0.25)
                elif direction == 'BEAR' and close < ema_trend:
                    confidence += 0.10 * self.confidence_weights.get('trend_alignment', 0.25)

            # Factor 4: Market conditions (RSI)
            rsi = latest_row.get('rsi', 50)

            if rsi:
                # For BULL: RSI 30-60 is favorable (not overbought)
                # For BEAR: RSI 40-70 is favorable (not oversold)
                if direction == 'BULL' and 30 < rsi < 60:
                    confidence += 0.05 * self.confidence_weights.get('market_conditions', 0.20)
                elif direction == 'BEAR' and 40 < rsi < 70:
                    confidence += 0.05 * self.confidence_weights.get('market_conditions', 0.20)

            # Cap confidence at 95%
            return min(0.95, max(0.30, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return self.base_confidence

    def _create_signal(
        self,
        signal_type: str,
        epic: str,
        timeframe: str,
        latest_row: pd.Series,
        spread_pips: float,
        metadata: Dict,
        confidence_boost: float
    ) -> Optional[Dict]:
        """
        Create complete signal dictionary.

        Args:
            signal_type: 'BULL' or 'BEAR'
            epic: Currency pair identifier
            timeframe: Timeframe string
            latest_row: Latest DataFrame row
            spread_pips: Spread in pips
            metadata: Crossover history metadata
            confidence_boost: Confidence bonus from prior crossovers

        Returns:
            Complete signal dictionary or None on error
        """
        try:
            # Create base signal using parent class method
            signal = self.create_base_signal(signal_type, epic, timeframe, latest_row)

            # Calculate confidence
            confidence = self._calculate_confidence(
                latest_row, signal_type, metadata, confidence_boost
            )
            signal['confidence'] = confidence
            signal['confidence_score'] = confidence

            # Add execution prices using parent class method
            signal = self.add_execution_prices(signal, spread_pips)

            # Get the current market price before limit order calculation
            market_price = signal.get('entry_price', latest_row.get('close', 0))

            # v2.0.0: Calculate limit entry with offset
            self.logger.info(f"\n📊 LIMIT ORDER: Calculating Entry Offset")
            entry_price, limit_offset_pips = self._calculate_limit_entry(
                market_price, signal_type, latest_row, epic
            )

            # Determine order type based on config
            order_type = 'limit' if self.limit_order_enabled and limit_offset_pips > 0 else 'market'
            self.logger.info(f"   📋 Order Type: {order_type.upper()}")

            # Update entry price with limit entry
            signal['entry_price'] = entry_price
            signal['market_price'] = market_price

            # Calculate SL/TP (using the new entry price)
            sl_tp = self._calculate_sl_tp(signal, epic, latest_row, spread_pips)
            signal['stop_distance'] = sl_tp['stop_distance']
            signal['limit_distance'] = sl_tp['limit_distance']

            # v2.0.0: Risk sanity check for limit orders with offset
            if order_type == 'limit':
                risk_pips = sl_tp['stop_distance']
                if risk_pips < self.min_risk_after_offset_pips:
                    self.logger.info(f"   ❌ Risk too small after offset ({risk_pips:.1f} < {self.min_risk_after_offset_pips} pips)")
                    return None
                if risk_pips > self.max_risk_after_offset_pips:
                    self.logger.info(f"   ❌ Risk too large after offset ({risk_pips:.1f} > {self.max_risk_after_offset_pips} pips)")
                    return None

            # v2.0.0: Add limit order fields
            signal['order_type'] = order_type
            signal['limit_offset_pips'] = round(limit_offset_pips, 1)
            signal['limit_expiry_minutes'] = self.limit_expiry_minutes if order_type == 'limit' else None

            # Add strategy-specific metadata
            signal['trigger_reason'] = f"EMA {self.ema_fast}/{self.ema_slow} crossover (3rd after 2 successful)"
            signal['strategy_metadata'] = {
                'prior_successful_crossovers': metadata.get('successful_count', 0),
                'lookback_hours': self.lookback_hours,
                'success_candles_required': self.success_candles,
                'crossover_history': metadata.get('crossover_history', []),
                'order_type': order_type,
                'limit_offset_pips': limit_offset_pips if order_type == 'limit' else 0.0,
                'limit_expiry_minutes': self.limit_expiry_minutes if order_type == 'limit' else None
            }

            return signal

        except Exception as e:
            self.logger.error(f"Error creating signal: {e}", exc_info=True)
            return None

    def _calculate_sl_tp(
        self,
        signal: Dict,
        epic: str,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Dict[str, int]:
        """
        Calculate stop loss and take profit.

        Uses ATR-based calculation with min/max bounds.

        Args:
            signal: Signal dictionary
            epic: Currency pair identifier
            latest_row: Latest DataFrame row
            spread_pips: Spread in pips

        Returns:
            Dictionary with stop_distance and limit_distance in pips
        """
        # Try to use base class method first
        try:
            return self.calculate_optimal_sl_tp(signal, epic, latest_row, spread_pips)
        except Exception:
            pass

        # Fallback: ATR-based calculation
        atr = latest_row.get('atr', 0)

        # Determine pip multiplier based on pair
        pair = self._extract_pair_from_epic(epic)
        if 'JPY' in pair:
            atr_pips = atr * 100  # JPY pairs: 0.01 = 1 pip
        else:
            atr_pips = atr * 10000  # Standard pairs: 0.0001 = 1 pip

        if atr_pips > 0:
            # ATR-based stops
            raw_stop = atr_pips * self.stop_atr_multiplier
            raw_target = atr_pips * self.target_atr_multiplier

            # Apply bounds
            stop_distance = max(self.min_stop_pips, min(self.max_stop_pips, int(raw_stop)))
            limit_distance = max(self.min_target_pips, int(raw_target))

            # Ensure minimum R:R
            min_rr = 1.5
            if limit_distance < stop_distance * min_rr:
                limit_distance = int(stop_distance * 2.0)
        else:
            # Fallback to defaults
            stop_distance = 20
            limit_distance = 40

        return {
            'stop_distance': stop_distance,
            'limit_distance': limit_distance
        }

    def _calculate_limit_entry(
        self,
        current_close: float,
        direction: str,
        latest_row: pd.Series,
        epic: str
    ) -> tuple:
        """
        Calculate limit entry price with offset for momentum confirmation.

        v2.3.0: Stop-entry style - confirm price is moving in intended direction:
        - BUY orders placed ABOVE current price (enter when price breaks up)
        - SELL orders placed BELOW current price (enter when price breaks down)
        - Max offset: 3 pips (user request)

        Args:
            current_close: Current market price
            direction: Trade direction ('BULL' or 'BEAR')
            latest_row: DataFrame row with indicator data
            epic: Currency pair identifier

        Returns:
            Tuple of (limit_entry_price, offset_pips)
        """
        if not self.limit_order_enabled:
            # Limit orders disabled - return current price (market order behavior)
            return current_close, 0.0

        # Determine pip value for the pair
        pair = self._extract_pair_from_epic(epic)
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        # Get ATR for offset calculation
        atr = latest_row.get('atr', 0)
        if atr and atr > 0:
            # Convert ATR to pips
            atr_pips = atr / pip_value
            # Calculate offset as percentage of ATR, clamped to min/max
            offset_pips = atr_pips * self.limit_offset_atr_factor
            offset_pips = min(max(offset_pips, self.limit_offset_min_pips), self.limit_offset_max_pips)
        else:
            # Fallback if ATR unavailable
            offset_pips = self.limit_offset_min_pips

        self.logger.info(f"   📉 Limit offset: {offset_pips:.1f} pips (ATR-based)")

        # Calculate offset in price terms
        offset = offset_pips * pip_value

        # Apply offset based on direction (stop-entry style: confirm direction continuation)
        if direction == 'BULL':
            # BUY: Place limit order ABOVE current price (enter when price breaks up)
            limit_entry = current_close + offset
        else:
            # SELL: Place limit order BELOW current price (enter when price breaks down)
            limit_entry = current_close - offset

        self.logger.info(f"   📍 Market price: {current_close:.5f}")
        self.logger.info(f"   📍 Limit entry: {limit_entry:.5f} ({offset_pips:.1f} pips momentum confirmation)")

        return limit_entry, offset_pips

    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract pair name from epic code (e.g., CS.D.GBPUSD.MINI.IP -> GBPUSD)"""
        parts = epic.upper().split('.')
        for part in parts:
            if len(part) in [6, 7] and any(curr in part for curr in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']):
                return part
        return 'DEFAULT'

    def get_tracker_state(self, epic: str = None) -> Dict:
        """Get current state of the crossover tracker for debugging"""
        return self.crossover_tracker.get_state_summary(epic)

    def reset_tracker(self, epic: str = None):
        """Reset tracker state (useful for testing)"""
        self.crossover_tracker.reset_state(epic)
