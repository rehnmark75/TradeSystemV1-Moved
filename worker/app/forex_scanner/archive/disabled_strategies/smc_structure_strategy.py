#!/usr/bin/env python3
"""
SMC Pure Structure Strategy
Structure-based trading using Smart Money Concepts (pure price action)

VERSION: 2.9.0 (OpenAI Recommendations)
DATE: 2025-11-30
STATUS: Testing - OpenAI recommendations: Body-Close BOS, ATR Displacement, Unmitigated OB

Performance Metrics (v2.8.5 Baseline - 90 days):
- Total Signals: 55
- Win Rate: 38.2%
- Profit Factor: 0.53
- Issue: Entries at local price extremes cause immediate adverse movement

Expected Performance (v2.9.0 with Pullback Filter):
- Total Signals: 40-50 (quality over quantity)
- Win Rate: 45-50% (+7-12% improvement)
- Profit Factor: 1.0-1.5 (+0.5-1.0 improvement)
- Key Fix: Validates retracement depth before entry

Strategy Logic:
1. Identify HTF trend (4H structure)
2. Detect BOS/CHoCH on 15m timeframe
3. Identify last opposing Order Block (institutional accumulation zone)
4. Wait for price to RETRACE to OB zone
5. **NEW: Validate pullback depth (30-60% retracement from BOS extreme)**
6. Detect REJECTION at OB level (wick rejection, engulfing, bounce)
7. Enter at OB with tight stop loss (5-8 pips beyond OB)
8. Target next structure level

Version History:
- v2.9.0 (2025-11-29): Pullback Depth Filter - validates retracement from BOS extreme
  Problem: Entries at local extremes (68% of losers had immediate adverse movement)
  Solution: Require 30-60% retracement from BOS swing before entry
- v2.8.5 (2025-11-16): Quality optimization (confidence cap, pair blacklist)
- v2.7.1 (2025-11-15): Swing Proximity Filter (replaces PD filter)
- v2.2.0 (2025-11-03): Order Block Re-entry implementation (major enhancement)
- v2.1.1 (2025-11-03): Added session filter (disabled), fixed timestamp bug
- v2.1.0 (2025-11-02): Phase 2.1 baseline - HTF alignment enabled
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

# Import helper modules
from .helpers.smc_trend_structure import SMCTrendStructure
from .helpers.smc_support_resistance import SMCSupportResistance
from .helpers.smc_candlestick_patterns import SMCCandlestickPatterns
from .helpers.smc_market_structure import SMCMarketStructure
from .helpers.zero_lag_liquidity import ZeroLagLiquidity
from .helpers.smc_order_blocks import SMCOrderBlocks


class SMCStructureStrategy:
    """
    Pure structure-based strategy using Smart Money Concepts

    Entry Requirements (ALL must be met):
    1. HTF trend identified (4H showing clear HH/HL or LH/LL)
    2. Price pulled back to key S/R level
    3. Rejection pattern confirmed (pin bar, engulfing, etc.)
    4. Signal direction aligns with HTF trend
    5. Structure-based stop loss calculated
    6. R:R meets minimum requirement
    """

    def __init__(self, config, logger=None):
        """Initialize SMC Structure Strategy"""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize helper modules
        self.trend_analyzer = SMCTrendStructure(logger=self.logger)
        self.sr_detector = SMCSupportResistance(logger=self.logger)
        self.pattern_detector = SMCCandlestickPatterns(logger=self.logger)
        self.market_structure = SMCMarketStructure(logger=self.logger)
        self.zero_lag = ZeroLagLiquidity(logger=self.logger)
        self.ob_detector = SMCOrderBlocks(logger=self.logger)

        # Initialize cooldown state tracking
        self.pair_cooldowns = {}  # {pair: last_signal_time}
        self.last_global_signal_time = None
        self.active_signals_count = 0

        # Signal deduplication tracking
        self.recent_signals = {}  # {pair: [(timestamp, price), ...]}

        # Load configuration
        self._load_config()

        self.logger.info("✅ SMC Structure Strategy initialized")
        self.logger.info(f"   HTF Timeframe: {self.htf_timeframe}")
        self.logger.info(f"   Min Pattern Strength: {self.min_pattern_strength}")
        self.logger.info(f"   Min R:R Ratio: {self.min_rr_ratio}")
        self.logger.info(f"   Min TP: {self.min_tp_pips} pips")
        self.logger.info(f"   SR Proximity: {self.sr_proximity_pips} pips")
        if self.cooldown_enabled:
            self.logger.info(f"   Cooldown: {self.signal_cooldown_hours}h per-pair, {self.global_cooldown_minutes}m global")
        self.logger.info(f"   OB Re-entry: {'✅ ENABLED' if self.ob_reentry_enabled else '❌ DISABLED'}")
        if self.ob_reentry_enabled:
            self.logger.info(f"   OB Lookback: {self.ob_lookback_bars} bars")
            self.logger.info(f"   OB Re-entry zone: {self.ob_reentry_zone}")

    def _load_config(self):
        """Load strategy configuration"""
        # Entry timeframe for signal detection
        self.entry_timeframe = getattr(self.config, 'SMC_ENTRY_TIMEFRAME', '15m')

        # Higher timeframe for trend analysis
        self.htf_timeframe = getattr(self.config, 'SMC_HTF_TIMEFRAME', '4h')
        self.htf_lookback = getattr(self.config, 'SMC_HTF_LOOKBACK', 100)

        # Support/Resistance detection
        self.sr_lookback = getattr(self.config, 'SMC_SR_LOOKBACK', 100)
        self.sr_proximity_pips = getattr(self.config, 'SMC_SR_PROXIMITY_PIPS', 20)

        # Pattern detection
        self.min_pattern_strength = getattr(self.config, 'SMC_MIN_PATTERN_STRENGTH', 0.70)
        self.pattern_lookback_bars = getattr(self.config, 'SMC_PATTERN_LOOKBACK_BARS', 5)

        # Risk management
        self.sl_buffer_pips = getattr(self.config, 'SMC_SL_BUFFER_PIPS', 5)
        self.min_rr_ratio = getattr(self.config, 'SMC_MIN_RR_RATIO', 2.0)
        self.min_tp_pips = getattr(self.config, 'SMC_MIN_TP_PIPS', 18)  # Phase 3.0: Minimum TP requirement

        # Partial profit settings
        self.partial_profit_enabled = getattr(self.config, 'SMC_PARTIAL_PROFIT_ENABLED', True)
        self.partial_profit_percent = getattr(self.config, 'SMC_PARTIAL_PROFIT_PERCENT', 50)
        self.partial_profit_rr = getattr(self.config, 'SMC_PARTIAL_PROFIT_RR', 1.5)

        # Cooldown system (anti-clustering)
        self.cooldown_enabled = getattr(self.config, 'SMC_COOLDOWN_ENABLED', True)
        self.signal_cooldown_hours = getattr(self.config, 'SMC_SIGNAL_COOLDOWN_HOURS', 4)
        self.global_cooldown_minutes = getattr(self.config, 'SMC_GLOBAL_COOLDOWN_MINUTES', 30)
        self.max_concurrent_signals = getattr(self.config, 'SMC_MAX_CONCURRENT_SIGNALS', 3)
        self.cooldown_enforcement = getattr(self.config, 'SMC_COOLDOWN_ENFORCEMENT', 'strict')

        # BOS/CHoCH re-entry parameters
        self.bos_choch_enabled = getattr(self.config, 'SMC_BOS_CHOCH_REENTRY_ENABLED', True)
        self.bos_choch_timeframe = getattr(self.config, 'SMC_BOS_CHOCH_TIMEFRAME', '15m')
        self.require_1h_alignment = getattr(self.config, 'SMC_REQUIRE_1H_ALIGNMENT', True)
        self.require_4h_alignment = getattr(self.config, 'SMC_REQUIRE_4H_ALIGNMENT', True)
        self.htf_alignment_lookback = getattr(self.config, 'SMC_HTF_ALIGNMENT_LOOKBACK', 50)
        self.reentry_zone_pips = getattr(self.config, 'SMC_REENTRY_ZONE_PIPS', 10)
        self.max_wait_bars = getattr(self.config, 'SMC_MAX_WAIT_BARS', 20)
        self.min_bos_significance = getattr(self.config, 'SMC_MIN_BOS_SIGNIFICANCE', 0.6)
        self.bos_stop_pips = getattr(self.config, 'SMC_BOS_STOP_PIPS', 10)
        self.patterns_optional = getattr(self.config, 'SMC_PATTERNS_OPTIONAL', True)

        # Zero Lag Liquidity parameters
        self.use_zero_lag_entry = getattr(self.config, 'SMC_USE_ZERO_LAG_ENTRY', True)
        self.zero_lag_wick_threshold = getattr(self.config, 'SMC_ZERO_LAG_WICK_THRESHOLD', 0.6)
        self.zero_lag_lookback = getattr(self.config, 'SMC_ZERO_LAG_LOOKBACK', 20)

        # Session filter configuration
        self.session_filter_enabled = getattr(self.config, 'SMC_SESSION_FILTER_ENABLED', False)
        self.block_asian_session = getattr(self.config, 'SMC_BLOCK_ASIAN_SESSION', True)

        # TIER 1 Momentum filter configuration
        self.momentum_filter_enabled = getattr(self.config, 'SMC_MOMENTUM_FILTER_ENABLED', False)
        self.momentum_lookback_candles = getattr(self.config, 'SMC_MOMENTUM_LOOKBACK_CANDLES', 3)
        self.momentum_min_aligned_candles = getattr(self.config, 'SMC_MOMENTUM_MIN_ALIGNED_CANDLES', 2)

        # Order Block Re-entry configuration (v2.2.0)
        self.ob_reentry_enabled = getattr(self.config, 'SMC_OB_REENTRY_ENABLED', True)
        self.ob_lookback_bars = getattr(self.config, 'SMC_OB_LOOKBACK_BARS', 20)
        self.ob_reentry_zone = getattr(self.config, 'SMC_OB_REENTRY_ZONE', 'lower_50')
        self.ob_require_rejection = getattr(self.config, 'SMC_OB_REQUIRE_REJECTION', True)
        self.ob_rejection_min_wick = getattr(self.config, 'SMC_OB_REJECTION_MIN_WICK_RATIO', 0.60)
        self.ob_sl_buffer_pips = getattr(self.config, 'SMC_OB_SL_BUFFER_PIPS', 5)

        # Swing Proximity Filter configuration (v2.7.1)
        self.swing_proximity_filter_enabled = getattr(self.config, 'SMC_SWING_PROXIMITY_FILTER_ENABLED', True)
        self.swing_exhaustion_threshold = getattr(self.config, 'SMC_SWING_EXHAUSTION_THRESHOLD', 0.20)

        # Pullback Depth Filter configuration (Phase 3 - Entry Timing)
        self.pullback_filter_enabled = getattr(self.config, 'SMC_PULLBACK_FILTER_ENABLED', True)
        self.pullback_min_retracement = getattr(self.config, 'SMC_PULLBACK_MIN_RETRACEMENT', 0.30)
        self.pullback_max_retracement = getattr(self.config, 'SMC_PULLBACK_MAX_RETRACEMENT', 0.60)

    def _check_cooldown(self, pair: str, current_time: datetime) -> tuple[bool, str]:
        """
        Check if pair is in cooldown period

        Returns:
            (can_trade, reason) - True if can trade, False if in cooldown
        """
        if not self.cooldown_enabled:
            return True, ""

        # Check max concurrent signals
        if self.active_signals_count >= self.max_concurrent_signals:
            return False, f"Max concurrent signals reached ({self.active_signals_count}/{self.max_concurrent_signals})"

        # Check per-pair cooldown
        if pair in self.pair_cooldowns:
            last_signal_time = self.pair_cooldowns[pair]
            cooldown_expires = last_signal_time + timedelta(hours=self.signal_cooldown_hours)
            if current_time < cooldown_expires:
                hours_remaining = (cooldown_expires - current_time).total_seconds() / 3600
                return False, f"Pair cooldown active ({hours_remaining:.1f}h remaining)"

        # Check global cooldown
        if self.last_global_signal_time:
            cooldown_expires = self.last_global_signal_time + timedelta(minutes=self.global_cooldown_minutes)
            if current_time < cooldown_expires:
                minutes_remaining = (cooldown_expires - current_time).total_seconds() / 60
                return False, f"Global cooldown active ({minutes_remaining:.0f}m remaining)"

        return True, ""

    def _update_cooldown(self, pair: str, current_time: datetime):
        """Update cooldown state after signal generated"""
        if not self.cooldown_enabled:
            return

        self.pair_cooldowns[pair] = current_time
        self.last_global_signal_time = current_time
        self.active_signals_count += 1

    def reset_cooldowns(self):
        """Reset all cooldowns - call this at the start of each backtest to ensure fresh state

        This prevents stale cooldowns from previous backtests blocking signals in new backtests.
        The strategy instance persists in Docker container memory, so cooldowns can carry over.
        """
        self.pair_cooldowns = {}
        self.last_global_signal_time = None
        self.active_signals_count = 0
        self.recent_signals = {}
        if self.logger:
            self.logger.info("🔄 SMC Structure strategy cooldowns reset for new backtest")

    def _is_duplicate_signal(self, pair: str, entry_price: float, current_time: datetime) -> tuple[bool, str]:
        """
        Check if signal is duplicate of recent signal

        Args:
            pair: Currency pair
            entry_price: Proposed entry price
            current_time: Current timestamp

        Returns:
            (is_duplicate, reason)
        """
        # Deduplication window: 4 hours and 5 pips
        time_window = timedelta(hours=4)
        price_threshold_pips = 5
        pip_value = 0.01 if 'JPY' in pair else 0.0001
        price_threshold = price_threshold_pips * pip_value

        # Initialize pair tracking if needed
        if pair not in self.recent_signals:
            self.recent_signals[pair] = []

        # Clean old signals (older than 4 hours)
        self.recent_signals[pair] = [
            (ts, price) for ts, price in self.recent_signals[pair]
            if current_time - ts < time_window
        ]

        # Check for duplicates
        for signal_time, signal_price in self.recent_signals[pair]:
            price_diff = abs(entry_price - signal_price)
            time_diff = current_time - signal_time

            if price_diff < price_threshold:
                hours_ago = time_diff.total_seconds() / 3600
                return True, f"Duplicate signal (same price {price_diff/pip_value:.1f} pips, {hours_ago:.1f}h ago)"

        return False, ""

    def _record_signal(self, pair: str, entry_price: float, current_time: datetime):
        """Record signal for deduplication"""
        if pair not in self.recent_signals:
            self.recent_signals[pair] = []
        self.recent_signals[pair].append((current_time, entry_price))

    def _detect_liquidity_sweep(self, df: pd.DataFrame, direction: str, current_idx: int, pair: str) -> dict:
        """
        Detect if price has taken out recent swing highs/lows (liquidity sweep)

        SMC Concept: Smart money takes liquidity (stops above highs/below lows) before reversing
        - SELL setup: Price should take out recent swing high (liquidity grab above resistance)
        - BUY setup: Price should take out recent swing low (liquidity grab below support)

        Args:
            df: Price dataframe
            direction: 'bearish' for SELL, 'bullish' for BUY
            current_idx: Current bar index
            pair: Currency pair

        Returns:
            dict with:
                - has_sweep: bool - True if liquidity sweep detected
                - sweep_level: float - Price level that was swept
                - bars_since_sweep: int - Bars since sweep occurred
                - sweep_type: str - 'high' or 'low'
        """
        # Load configuration
        liquidity_sweep_enabled = getattr(self.config, 'SMC_LIQUIDITY_SWEEP_ENABLED', True)
        lookback = getattr(self.config, 'SMC_LIQUIDITY_SWEEP_LOOKBACK', 20)
        min_bars = getattr(self.config, 'SMC_LIQUIDITY_SWEEP_MIN_BARS', 2)
        max_bars = getattr(self.config, 'SMC_LIQUIDITY_SWEEP_MAX_BARS', 10)

        if not liquidity_sweep_enabled:
            return {'has_sweep': True, 'sweep_level': 0.0, 'bars_since_sweep': 0, 'sweep_type': 'none'}

        # Get lookback window (excluding current bar)
        start_idx = max(0, current_idx - lookback)
        lookback_df = df.iloc[start_idx:current_idx]

        if len(lookback_df) < 3:
            return {'has_sweep': False, 'sweep_level': 0.0, 'bars_since_sweep': 0, 'sweep_type': 'none'}

        pip_value = 0.01 if 'JPY' in pair else 0.0001

        if direction == 'bearish':
            # SELL setup: Look for recent swing high that was taken out
            # Find swing highs in lookback period (high higher than 2 bars before and after)
            swing_highs = []
            for i in range(2, len(lookback_df) - 2):
                if (lookback_df['high'].iloc[i] > lookback_df['high'].iloc[i-1] and
                    lookback_df['high'].iloc[i] > lookback_df['high'].iloc[i-2] and
                    lookback_df['high'].iloc[i] > lookback_df['high'].iloc[i+1] and
                    lookback_df['high'].iloc[i] > lookback_df['high'].iloc[i+2]):
                    swing_highs.append({
                        'level': lookback_df['high'].iloc[i],
                        'bars_ago': len(lookback_df) - i
                    })

            if not swing_highs:
                return {'has_sweep': False, 'sweep_level': 0.0, 'bars_since_sweep': 0, 'sweep_type': 'high'}

            # Find the most recent swing high
            recent_swing = max(swing_highs, key=lambda x: -x['bars_ago'])  # Most recent (lowest bars_ago)
            swing_level = recent_swing['level']

            # Check if price took out this high in recent bars (within max_bars)
            recent_window = df.iloc[current_idx - max_bars:current_idx]
            if len(recent_window) == 0:
                return {'has_sweep': False, 'sweep_level': 0.0, 'bars_since_sweep': 0, 'sweep_type': 'high'}

            # Price must have exceeded swing high
            highest_recent = recent_window['high'].max()
            if highest_recent <= swing_level:
                return {'has_sweep': False, 'sweep_level': swing_level, 'bars_since_sweep': 0, 'sweep_type': 'high'}

            # Find when the sweep occurred
            sweep_idx = None
            for i in range(len(recent_window)):
                if recent_window['high'].iloc[i] > swing_level:
                    sweep_idx = i
                    break

            if sweep_idx is None:
                return {'has_sweep': False, 'sweep_level': swing_level, 'bars_since_sweep': 0, 'sweep_type': 'high'}

            bars_since_sweep = len(recent_window) - sweep_idx - 1

            # Must be at least min_bars after the sweep (allow reversal to begin)
            if bars_since_sweep < min_bars:
                return {'has_sweep': False, 'sweep_level': swing_level, 'bars_since_sweep': bars_since_sweep, 'sweep_type': 'high'}

            return {
                'has_sweep': True,
                'sweep_level': swing_level,
                'bars_since_sweep': bars_since_sweep,
                'sweep_type': 'high'
            }

        else:  # direction == 'bullish'
            # BUY setup: Look for recent swing low that was taken out
            # Find swing lows in lookback period (low lower than 2 bars before and after)
            swing_lows = []
            for i in range(2, len(lookback_df) - 2):
                if (lookback_df['low'].iloc[i] < lookback_df['low'].iloc[i-1] and
                    lookback_df['low'].iloc[i] < lookback_df['low'].iloc[i-2] and
                    lookback_df['low'].iloc[i] < lookback_df['low'].iloc[i+1] and
                    lookback_df['low'].iloc[i] < lookback_df['low'].iloc[i+2]):
                    swing_lows.append({
                        'level': lookback_df['low'].iloc[i],
                        'bars_ago': len(lookback_df) - i
                    })

            if not swing_lows:
                return {'has_sweep': False, 'sweep_level': 0.0, 'bars_since_sweep': 0, 'sweep_type': 'low'}

            # Find the most recent swing low
            recent_swing = max(swing_lows, key=lambda x: -x['bars_ago'])  # Most recent (lowest bars_ago)
            swing_level = recent_swing['level']

            # Check if price took out this low in recent bars (within max_bars)
            recent_window = df.iloc[current_idx - max_bars:current_idx]
            if len(recent_window) == 0:
                return {'has_sweep': False, 'sweep_level': 0.0, 'bars_since_sweep': 0, 'sweep_type': 'low'}

            # Price must have gone below swing low
            lowest_recent = recent_window['low'].min()
            if lowest_recent >= swing_level:
                return {'has_sweep': False, 'sweep_level': swing_level, 'bars_since_sweep': 0, 'sweep_type': 'low'}

            # Find when the sweep occurred
            sweep_idx = None
            for i in range(len(recent_window)):
                if recent_window['low'].iloc[i] < swing_level:
                    sweep_idx = i
                    break

            if sweep_idx is None:
                return {'has_sweep': False, 'sweep_level': swing_level, 'bars_since_sweep': 0, 'sweep_type': 'low'}

            bars_since_sweep = len(recent_window) - sweep_idx - 1

            # Must be at least min_bars after the sweep (allow reversal to begin)
            if bars_since_sweep < min_bars:
                return {'has_sweep': False, 'sweep_level': swing_level, 'bars_since_sweep': bars_since_sweep, 'sweep_type': 'low'}

            return {
                'has_sweep': True,
                'sweep_level': swing_level,
                'bars_since_sweep': bars_since_sweep,
                'sweep_type': 'low'
            }

    def _get_trading_session(self, timestamp):
        """
        Determine current forex trading session based on UTC time

        Args:
            timestamp: datetime object

        Returns:
            str: Session name ('ASIAN', 'LONDON', 'NEW_YORK', 'ASIAN_LATE')
        """
        hour_utc = timestamp.hour

        # Session definitions (UTC)
        if 0 <= hour_utc < 7:
            return 'ASIAN'
        elif 7 <= hour_utc < 15:
            return 'LONDON'
        elif 15 <= hour_utc < 22:
            return 'NEW_YORK'
        else:
            return 'ASIAN_LATE'

    def _validate_session_quality(self, timestamp):
        """
        TIER 1 FILTER: Session-Based Quality Filter

        Hypothesis: Asian session (0-7 UTC) generates false signals due to low liquidity
        and range-bound behavior. London/NY sessions provide cleaner structure-based moves.

        Args:
            timestamp: datetime object

        Returns:
            tuple: (is_valid, reason_string)
        """
        if not self.session_filter_enabled:
            return True, "Session filter disabled"

        session = self._get_trading_session(timestamp)
        hour_utc = timestamp.hour

        # Block Asian session entirely (low liquidity, ranging)
        if self.block_asian_session and (session == 'ASIAN' or session == 'ASIAN_LATE'):
            return False, f"Asian session ({hour_utc}:00 UTC) - low liquidity ranging market"

        # Bonus log for high-quality sessions
        if 12 <= hour_utc < 15:  # London/NY overlap
            return True, f"Session overlap ({hour_utc}:00 UTC) - highest liquidity"
        elif 7 <= hour_utc < 9:  # London open
            return True, f"London open ({hour_utc}:00 UTC) - high volatility"

        return True, f"{session} session ({hour_utc}:00 UTC) - acceptable"

    def _validate_pullback_momentum(self, df_15m: pd.DataFrame, trade_direction: str) -> tuple:
        """
        TIER 1 FILTER: Pullback Momentum Validator

        Prevents counter-momentum entries by checking recent 15m candle bias.
        Requires minimum number of recent candles aligned with trade direction.

        For BULL signals: Requires 8/12 recent 15m candles to be bullish (close > open)
        For BEAR signals: Requires 8/12 recent 15m candles to be bearish (close < open)
        12 candles × 15m = 3 hours of recent price action

        Args:
            df_15m: 15m timeframe OHLCV data (entry timeframe)
            trade_direction: 'BULL' or 'BEAR'

        Returns:
            tuple: (is_valid, reason_string)
        """
        if not self.momentum_filter_enabled:
            return True, "Momentum filter disabled"

        # Get recent completed candles (exclude current candle)
        lookback = self.momentum_lookback_candles
        recent_candles = df_15m.iloc[-(lookback + 1):-1]

        if len(recent_candles) < lookback:
            return True, f"Insufficient data ({len(recent_candles)} < {lookback} candles)"

        # Calculate aligned candles (bullish or bearish)
        if trade_direction == 'BULL':
            # Count bullish candles (close > open)
            aligned = (recent_candles['close'] > recent_candles['open']).sum()
            candle_type = "bullish"
        else:
            # Count bearish candles (close < open)
            aligned = (recent_candles['close'] < recent_candles['open']).sum()
            candle_type = "bearish"

        # Check if sufficient aligned candles
        min_required = self.momentum_min_aligned_candles
        is_valid = aligned >= min_required

        if is_valid:
            return True, f"Momentum confirmed: {aligned}/{lookback} {candle_type} 15m candles (3h lookback)"
        else:
            return False, f"Momentum lacking: {aligned}/{lookback} {candle_type} 15m candles (need {min_required})"

    def _validate_swing_proximity(
        self,
        current_price: float,
        trade_direction: str,
        swing_highs: List[Dict],
        swing_lows: List[Dict],
        pip_value: float
    ) -> tuple:
        """
        TIER 1 FILTER: Swing Proximity Validator (v2.7.1)

        Validates entry price distance from significant swing levels to prevent exhaustion zone entries.
        REPLACES the failed Premium/Discount zone filter with structure-based logic.

        CRITICAL DIFFERENCE FROM PD FILTER:
          - PD Filter (FAILED): Used arbitrary 33% zones based on fixed lookback range
          - Swing Filter (NEW): Uses actual HTF swing highs/lows from trend structure analysis

        WHY PD FILTER FAILED (v2.6.7 analysis):
          - Rejected ALL 8 winners in Phase 2.6.3 (SELL in discount = valid trend continuation)
          - "SELL in discount = selling at bottom" was WRONG assumption in strong trends
          - Reality: SELL in discount during downtrend = VALID continuation setup

        HOW SWING FILTER IS BETTER:
          - BUY signals: Reject if too close to swing HIGH (chasing/exhaustion)
          - SELL signals: Reject if too close to swing LOW (chasing/exhaustion)
          - Allows trend continuations at proper pullback distances
          - Adaptive: Uses real swing points, not arbitrary % zones

        Example (SELL in downtrend):
          Swing High: 154.50
          Swing Low:  153.90 (range = 60 pips)
          Entry: 154.30
          Position: (154.30 - 153.90) / 60 = 67% from low ✅ ALLOW (good pullback from low)
          Entry: 154.00
          Position: (154.00 - 153.90) / 60 = 17% from low ❌ REJECT (too close to low = exhaustion)

        Args:
            current_price: Entry price
            trade_direction: 'BULL' or 'BEAR'
            swing_highs: List of swing high dicts from HTF analysis
            swing_lows: List of swing low dicts from HTF analysis
            pip_value: Pip value for distance calculation

        Returns:
            tuple: (is_valid, reason_string)
        """
        if not self.swing_proximity_filter_enabled:
            return True, "Swing proximity filter disabled"

        if not swing_highs or not swing_lows:
            return True, "Insufficient swing data - allowing entry"

        last_swing_high = swing_highs[-1]['price']
        last_swing_low = swing_lows[-1]['price']
        swing_range = last_swing_high - last_swing_low

        if swing_range <= 0:
            return True, "Invalid swing range - allowing entry"

        # Exhaustion threshold from config (default 20%)
        exhaustion_threshold = self.swing_exhaustion_threshold

        if trade_direction == 'BULL':
            # For BUY: Check distance from swing HIGH (avoid buying near tops)
            distance_from_high = last_swing_high - current_price
            position_in_range = distance_from_high / swing_range
            distance_pips = distance_from_high / pip_value

            if position_in_range < exhaustion_threshold:
                return False, (
                    f"BUY rejected - Too close to swing high (exhaustion/chasing zone)\n"
                    f"   Current price: {current_price:.5f}\n"
                    f"   Swing High: {last_swing_high:.5f} (only {distance_pips:.1f} pips away)\n"
                    f"   Swing Low: {last_swing_low:.5f}\n"
                    f"   Swing Range: {swing_range/pip_value:.1f} pips\n"
                    f"   Position: {position_in_range*100:.0f}% from high (need >{exhaustion_threshold*100:.0f}%)\n"
                    f"   💡 Buying near swing highs = chasing price, not pullback entry\n"
                    f"   💡 PD filter would use arbitrary zones - this uses REAL swing structure"
                )
            else:
                return True, (
                    f"BUY allowed - Good distance from swing high ({distance_pips:.1f} pips)\n"
                    f"   Position: {position_in_range*100:.0f}% from high (range: {swing_range/pip_value:.1f} pips)\n"
                    f"   ✅ Proper pullback entry, not exhaustion zone"
                )

        else:  # BEAR
            # For SELL: Check distance from swing LOW (avoid selling near bottoms)
            distance_from_low = current_price - last_swing_low
            position_in_range = distance_from_low / swing_range
            distance_pips = distance_from_low / pip_value

            if position_in_range < exhaustion_threshold:
                return False, (
                    f"SELL rejected - Too close to swing low (exhaustion/chasing zone)\n"
                    f"   Current price: {current_price:.5f}\n"
                    f"   Swing High: {last_swing_high:.5f}\n"
                    f"   Swing Low: {last_swing_low:.5f} (only {distance_pips:.1f} pips away)\n"
                    f"   Swing Range: {swing_range/pip_value:.1f} pips\n"
                    f"   Position: {position_in_range*100:.0f}% from low (need >{exhaustion_threshold*100:.0f}%)\n"
                    f"   💡 Selling near swing lows = chasing price, not pullback entry\n"
                    f"   💡 PD filter rejected this type (WRONG!) - swing filter is smarter"
                )
            else:
                return True, (
                    f"SELL allowed - Good distance from swing low ({distance_pips:.1f} pips)\n"
                    f"   Position: {position_in_range*100:.0f}% from low (range: {swing_range/pip_value:.1f} pips)\n"
                    f"   ✅ Proper pullback entry, not exhaustion zone\n"
                    f"   ✅ PD filter would reject trend continuations - this allows them"
                )

    def _validate_pullback_depth(
        self,
        current_price: float,
        bos_direction: str,
        df_15m: pd.DataFrame,
        order_block: Optional[Dict],
        pip_value: float
    ) -> tuple:
        """
        PULLBACK DEPTH FILTER (Phase 3 - Entry Timing Optimization)

        Validates that price has ACTUALLY retraced from the BOS swing extreme
        before entering. Prevents entries at local price extremes where
        immediate adverse movement is likely.

        The Problem This Solves:
        - Analysis showed 68% of losers had immediate adverse movement (>5 pips in 2 bars)
        - Current OB check only validates price is IN the zone
        - Missing: validation that we've had meaningful retracement from BOS level

        For BULLISH BOS (looking to buy on pullback):
            - BOS created new swing high (price broke above previous structure)
            - We want to BUY on pullback to Order Block
            - Entry should be BELOW the BOS high by at least MIN_RETRACEMENT%
            - Example: BOS high=1.1050, OB=1.1020, range=30 pips
              - Min 30% retracement: Entry must be < 1.1041 (9+ pips from high)
              - If entry at 1.1048, only 2 pips pullback = TOO CLOSE TO EXTREME

        For BEARISH BOS (looking to sell on pullback):
            - BOS created new swing low (price broke below previous structure)
            - We want to SELL on pullback to Order Block
            - Entry should be ABOVE the BOS low by at least MIN_RETRACEMENT%
            - Example: BOS low=1.0950, OB=1.0980, range=30 pips
              - Min 30% retracement: Entry must be > 1.0959 (9+ pips from low)

        Args:
            current_price: Current entry price
            bos_direction: 'bullish' or 'bearish'
            df_15m: 15m DataFrame for finding BOS extremes
            order_block: Order block dict (contains high/low levels)
            pip_value: Pip value for the pair

        Returns:
            tuple: (is_valid, reason_string)
        """
        if not self.pullback_filter_enabled:
            return True, "Pullback filter disabled"

        if order_block is None:
            return True, "No order block - pullback filter skipped"

        # Get recent price action to find BOS extreme
        lookback_bars = 20  # Same as OB lookback
        recent_df = df_15m.tail(lookback_bars)

        if len(recent_df) < 5:
            return True, "Insufficient data for pullback calculation"

        if bos_direction == 'bullish':
            # For BULLISH BOS:
            # - Recent high = BOS swing extreme (where smart money pushed price)
            # - OB low = pullback target zone
            # - Entry should be meaningfully below the high
            bos_extreme = float(recent_df['high'].max())
            ob_level = order_block['low']  # Bottom of OB for bullish entry

            # Calculate range from BOS extreme to OB
            pullback_range = bos_extreme - ob_level

            if pullback_range <= 0:
                return True, "Invalid pullback range - allowing entry"

            # Calculate current retracement from BOS extreme
            # For bullish: how far below the high is current price?
            retracement_distance = bos_extreme - current_price
            retracement_pct = retracement_distance / pullback_range

            # Calculate pip values for logging
            range_pips = pullback_range / pip_value
            retracement_pips = retracement_distance / pip_value
            min_required_pips = (pullback_range * self.pullback_min_retracement) / pip_value
            max_allowed_pips = (pullback_range * self.pullback_max_retracement) / pip_value

            # Check minimum retracement (must have pulled back enough)
            if retracement_pct < self.pullback_min_retracement:
                return False, (
                    f"❌ PULLBACK FILTER: BUY entry too close to BOS high\n"
                    f"   BOS High: {bos_extreme:.5f} (recent swing extreme)\n"
                    f"   OB Low: {ob_level:.5f} (pullback target)\n"
                    f"   Current Price: {current_price:.5f}\n"
                    f"   Pullback Range: {range_pips:.1f} pips\n"
                    f"   Current Retracement: {retracement_pips:.1f} pips ({retracement_pct*100:.0f}%)\n"
                    f"   Minimum Required: {min_required_pips:.1f} pips ({self.pullback_min_retracement*100:.0f}%)\n"
                    f"   💡 Entry at local HIGH = immediate adverse movement likely\n"
                    f"   💡 Wait for deeper pullback toward OB before entry"
                )

            # Check maximum retracement (shouldn't have pulled back too much)
            if retracement_pct > self.pullback_max_retracement:
                return False, (
                    f"❌ PULLBACK FILTER: BUY entry too deep - structure may be failing\n"
                    f"   BOS High: {bos_extreme:.5f}\n"
                    f"   OB Low: {ob_level:.5f}\n"
                    f"   Current Price: {current_price:.5f}\n"
                    f"   Pullback Range: {range_pips:.1f} pips\n"
                    f"   Current Retracement: {retracement_pips:.1f} pips ({retracement_pct*100:.0f}%)\n"
                    f"   Maximum Allowed: {max_allowed_pips:.1f} pips ({self.pullback_max_retracement*100:.0f}%)\n"
                    f"   💡 Price pulled back too far - BOS structure may be failing\n"
                    f"   💡 Deep retracements often signal trend reversal"
                )

            # Valid pullback
            return True, (
                f"✅ PULLBACK FILTER PASSED (BUY)\n"
                f"   BOS High: {bos_extreme:.5f}\n"
                f"   OB Low: {ob_level:.5f}\n"
                f"   Current Price: {current_price:.5f}\n"
                f"   Retracement: {retracement_pips:.1f} pips ({retracement_pct*100:.0f}%)\n"
                f"   Valid Range: {self.pullback_min_retracement*100:.0f}%-{self.pullback_max_retracement*100:.0f}%\n"
                f"   ✅ Proper pullback entry, not at local extreme"
            )

        else:  # bearish
            # For BEARISH BOS:
            # - Recent low = BOS swing extreme (where smart money pushed price)
            # - OB high = pullback target zone
            # - Entry should be meaningfully above the low
            bos_extreme = float(recent_df['low'].min())
            ob_level = order_block['high']  # Top of OB for bearish entry

            # Calculate range from BOS extreme to OB
            pullback_range = ob_level - bos_extreme

            if pullback_range <= 0:
                return True, "Invalid pullback range - allowing entry"

            # Calculate current retracement from BOS extreme
            # For bearish: how far above the low is current price?
            retracement_distance = current_price - bos_extreme
            retracement_pct = retracement_distance / pullback_range

            # Calculate pip values for logging
            range_pips = pullback_range / pip_value
            retracement_pips = retracement_distance / pip_value
            min_required_pips = (pullback_range * self.pullback_min_retracement) / pip_value
            max_allowed_pips = (pullback_range * self.pullback_max_retracement) / pip_value

            # Check minimum retracement (must have pulled back enough)
            if retracement_pct < self.pullback_min_retracement:
                return False, (
                    f"❌ PULLBACK FILTER: SELL entry too close to BOS low\n"
                    f"   BOS Low: {bos_extreme:.5f} (recent swing extreme)\n"
                    f"   OB High: {ob_level:.5f} (pullback target)\n"
                    f"   Current Price: {current_price:.5f}\n"
                    f"   Pullback Range: {range_pips:.1f} pips\n"
                    f"   Current Retracement: {retracement_pips:.1f} pips ({retracement_pct*100:.0f}%)\n"
                    f"   Minimum Required: {min_required_pips:.1f} pips ({self.pullback_min_retracement*100:.0f}%)\n"
                    f"   💡 Entry at local LOW = immediate adverse movement likely\n"
                    f"   💡 Wait for deeper pullback toward OB before entry"
                )

            # Check maximum retracement (shouldn't have pulled back too much)
            if retracement_pct > self.pullback_max_retracement:
                return False, (
                    f"❌ PULLBACK FILTER: SELL entry too deep - structure may be failing\n"
                    f"   BOS Low: {bos_extreme:.5f}\n"
                    f"   OB High: {ob_level:.5f}\n"
                    f"   Current Price: {current_price:.5f}\n"
                    f"   Pullback Range: {range_pips:.1f} pips\n"
                    f"   Current Retracement: {retracement_pips:.1f} pips ({retracement_pct*100:.0f}%)\n"
                    f"   Maximum Allowed: {max_allowed_pips:.1f} pips ({self.pullback_max_retracement*100:.0f}%)\n"
                    f"   💡 Price pulled back too far - BOS structure may be failing\n"
                    f"   💡 Deep retracements often signal trend reversal"
                )

            # Valid pullback
            return True, (
                f"✅ PULLBACK FILTER PASSED (SELL)\n"
                f"   BOS Low: {bos_extreme:.5f}\n"
                f"   OB High: {ob_level:.5f}\n"
                f"   Current Price: {current_price:.5f}\n"
                f"   Retracement: {retracement_pips:.1f} pips ({retracement_pct*100:.0f}%)\n"
                f"   Valid Range: {self.pullback_min_retracement*100:.0f}%-{self.pullback_max_retracement*100:.0f}%\n"
                f"   ✅ Proper pullback entry, not at local extreme"
            )

    def _validate_pullback_depth_universal(
        self,
        current_price: float,
        direction: str,
        df: pd.DataFrame,
        pip_value: float
    ) -> tuple:
        """
        UNIVERSAL PULLBACK DEPTH FILTER (Phase 3 - Entry Timing)

        Validates entry isn't at local price extremes using recent swing structure.
        This is a SIMPLIFIED version for use in STEP 3D path where no Order Block exists.

        Uses recent high/low from swing range to determine if entry is at an extreme.

        For BULLISH entries (BUY):
            - Entry should NOT be too close to recent HIGH (buying at top)
            - Entry should have meaningful pullback from recent high

        For BEARISH entries (SELL):
            - Entry should NOT be too close to recent LOW (selling at bottom)
            - Entry should have meaningful pullback from recent low

        Args:
            current_price: Current entry price
            direction: 'bullish' or 'bearish'
            df: Price DataFrame (15m or 1H)
            pip_value: Pip value for the pair

        Returns:
            tuple: (is_valid, reason_string)
        """
        if not self.pullback_filter_enabled:
            return True, "Pullback filter disabled"

        if df is None or len(df) < 20:
            return True, "Insufficient data for pullback calculation"

        # Use recent 20 bars for swing analysis
        recent_df = df.tail(20)

        swing_high = float(recent_df['high'].max())
        swing_low = float(recent_df['low'].min())
        swing_range = swing_high - swing_low

        if swing_range <= 0:
            return True, "Invalid swing range - allowing entry"

        # Calculate pip values
        range_pips = swing_range / pip_value

        # For very tight ranges, skip filter (low volatility)
        if range_pips < 15:
            return True, f"Swing range too small ({range_pips:.1f} pips) - allowing entry"

        if direction == 'bullish':
            # For BULLISH (BUY): Check we're not buying at the top
            # Distance from swing high (how far below the high)
            distance_from_high = swing_high - current_price
            position_pct = distance_from_high / swing_range  # 0% = at high, 100% = at low

            distance_pips = distance_from_high / pip_value
            min_required_pips = (swing_range * self.pullback_min_retracement) / pip_value

            if position_pct < self.pullback_min_retracement:
                return False, (
                    f"❌ UNIVERSAL PULLBACK FILTER: BUY too close to swing high\n"
                    f"   Swing High: {swing_high:.5f}\n"
                    f"   Swing Low: {swing_low:.5f}\n"
                    f"   Current Price: {current_price:.5f}\n"
                    f"   Distance from High: {distance_pips:.1f} pips ({position_pct*100:.0f}%)\n"
                    f"   Minimum Required: {min_required_pips:.1f} pips ({self.pullback_min_retracement*100:.0f}%)\n"
                    f"   💡 Buying at local HIGH = poor entry timing"
                )

            return True, (
                f"✅ UNIVERSAL PULLBACK: BUY entry OK\n"
                f"   Distance from High: {distance_pips:.1f} pips ({position_pct*100:.0f}%)\n"
                f"   ✅ Good pullback from swing high"
            )

        else:  # bearish
            # For BEARISH (SELL): Check we're not selling at the bottom
            # Distance from swing low (how far above the low)
            distance_from_low = current_price - swing_low
            position_pct = distance_from_low / swing_range  # 0% = at low, 100% = at high

            distance_pips = distance_from_low / pip_value
            min_required_pips = (swing_range * self.pullback_min_retracement) / pip_value

            if position_pct < self.pullback_min_retracement:
                return False, (
                    f"❌ UNIVERSAL PULLBACK FILTER: SELL too close to swing low\n"
                    f"   Swing High: {swing_high:.5f}\n"
                    f"   Swing Low: {swing_low:.5f}\n"
                    f"   Current Price: {current_price:.5f}\n"
                    f"   Distance from Low: {distance_pips:.1f} pips ({position_pct*100:.0f}%)\n"
                    f"   Minimum Required: {min_required_pips:.1f} pips ({self.pullback_min_retracement*100:.0f}%)\n"
                    f"   💡 Selling at local LOW = poor entry timing"
                )

            return True, (
                f"✅ UNIVERSAL PULLBACK: SELL entry OK\n"
                f"   Distance from Low: {distance_pips:.1f} pips ({position_pct*100:.0f}%)\n"
                f"   ✅ Good pullback from swing low"
            )

    def detect_signal(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        epic: str,
        pair: str,
        df_15m: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Detect SMC structure-based trading signal

        Args:
            df_1h: 1H timeframe OHLCV data (entry timeframe)
            df_4h: 4H timeframe OHLCV data (higher timeframe for trend)
            epic: IG Markets epic code
            pair: Currency pair name
            df_15m: Optional 15m timeframe for BOS/CHoCH detection

        Returns:
            Signal dict or None if no valid signal
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🔍 SMC Structure Strategy - Signal Detection")
        self.logger.info(f"   Pair: {pair} ({epic})")
        self.logger.info(f"   Entry TF: 15m | HTF: {self.htf_timeframe} | SR/Confirmation TF: 1H")
        self.logger.info(f"{'='*70}")

        # Get candle timestamp (for backtesting compatibility)
        # CRITICAL FIX: Use column data if index is reset to integers (same fix as SMC_SIMPLE)
        if 'start_time' in df_1h.columns:
            candle_timestamp = df_1h['start_time'].iloc[-1]
        elif 'timestamp' in df_1h.columns:
            candle_timestamp = df_1h['timestamp'].iloc[-1]
        else:
            candle_timestamp = df_1h.index[-1]

        # Convert to datetime for cooldown check
        # Handle various timestamp types: pd.Timestamp, datetime, numpy.int64 (nanoseconds)
        if hasattr(candle_timestamp, 'to_pydatetime'):
            current_time = candle_timestamp.to_pydatetime()
        elif isinstance(candle_timestamp, (int, np.integer)):
            # numpy.int64 nanosecond timestamp - convert to datetime
            current_time = pd.Timestamp(candle_timestamp).to_pydatetime()
        else:
            current_time = candle_timestamp

        # Check cooldown before processing - USE CANDLE TIME, NOT datetime.now()!
        can_trade, cooldown_reason = self._check_cooldown(pair, current_time)
        if not can_trade:
            self.logger.info(f"   ⏱️  {cooldown_reason} - SKIPPING")
            return None

        # TIER 0 FILTER: Pair Blacklist (v2.8.5 - Quality Optimization)
        blacklist_pairs = getattr(self.config, 'SMC_BLACKLIST_PAIRS', [])
        if pair in blacklist_pairs:
            self.logger.info(f"\n🚫 PAIR BLACKLIST FILTER:")
            self.logger.info(f"   ❌ {pair} is blacklisted (poor historical performance)")
            self.logger.info(f"   💡 Based on v2.8.4 analysis: Low win rate on this pair")
            return None

        # TIER 1 FILTER: Session Quality Check (use candle timestamp, not current time)
        session_valid, session_reason = self._validate_session_quality(candle_timestamp)
        if not session_valid:
            self.logger.info(f"\n🕐 [SESSION FILTER] {session_reason}")
            self.logger.info(f"   ❌ SIGNAL REJECTED - Avoid low-quality trading sessions")
            return None
        else:
            self.logger.info(f"\n🕐 [SESSION FILTER] {session_reason}")

        # Get pip value
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        try:
            # STEP 1: Identify HTF trend (4H structure)
            self.logger.info(f"\n📊 STEP 1: Analyzing HTF Trend Structure ({self.htf_timeframe})")

            trend_analysis = self.trend_analyzer.analyze_trend(
                df=df_4h,
                epic=epic,
                lookback=self.htf_lookback
            )

            self.logger.info(f"   Trend: {trend_analysis['trend']}")
            self.logger.info(f"   Strength: {trend_analysis['strength']*100:.0f}%")
            self.logger.info(f"   Structure: {trend_analysis['structure_type']}")
            self.logger.info(f"   Swing Highs: {len(trend_analysis['swing_highs'])}")
            self.logger.info(f"   Swing Lows: {len(trend_analysis['swing_lows'])}")

            if trend_analysis['in_pullback']:
                self.logger.info(f"   ✅ In Pullback: {trend_analysis['pullback_depth']*100:.0f}% retracement")

            # PRIMARY: Use BOS/CHoCH on HTF to determine trend direction
            self.logger.info(f"\n🔍 Detecting BOS/CHoCH on HTF ({self.htf_timeframe}) for trend direction...")

            # Analyze market structure - returns DataFrame with BOS/CHoCH annotations
            df_4h_with_structure = self.market_structure.analyze_market_structure(
                df=df_4h,
                epic=epic,
                config=vars(self.config) if hasattr(self.config, '__dict__') else {}
            )

            # Get last BOS/CHoCH direction from the annotated DataFrame
            # v2.9.0: Pass config for body-close BOS validation
            bos_config = vars(self.config) if hasattr(self.config, '__dict__') else {}
            bos_choch_direction = self.market_structure.get_last_bos_choch_direction(df_4h_with_structure, bos_config)

            # ALWAYS use BOS/CHoCH as primary trend indicator (smart money direction)
            if bos_choch_direction in ['bullish', 'bearish']:
                # Use BOS/CHoCH direction as trend
                final_trend = 'BULL' if bos_choch_direction == 'bullish' else 'BEAR'

                # PHASE 2.8.0: DYNAMIC HTF STRENGTH - Use new multi-factor quality calculation
                # ALWAYS use the dynamic strength from SMCTrendStructure.analyze_trend()
                # This provides true quality-based scoring (30-100% distribution)
                final_strength = trend_analysis['strength']

                if trend_analysis['trend'] == final_trend:
                    # BOS/CHoCH aligns with swing structure - optimal setup
                    self.logger.info(f"   ✅ BOS/CHoCH: {bos_choch_direction.upper()} → {final_trend}")
                    self.logger.info(f"   ✅ Swing structure ALIGNS: {trend_analysis['structure_type']}")
                    self.logger.info(f"   🎯 DYNAMIC HTF Strength: {final_strength*100:.0f}%")
                else:
                    # BOS/CHoCH differs from swing structure - still use dynamic strength but log conflict
                    # The dynamic calculation factors in the swing quality regardless
                    self.logger.info(f"   ✅ BOS/CHoCH: {bos_choch_direction.upper()} → {final_trend}")
                    self.logger.info(f"   ⚠️  Swing structure differs: {trend_analysis['trend']} ({trend_analysis['structure_type']})")
                    self.logger.info(f"   🎯 DYNAMIC HTF Strength: {final_strength*100:.0f}% (from multi-factor analysis)")
                    self.logger.info(f"      ℹ️  Using quality-based strength regardless of BOS/swing alignment")
            else:
                # No BOS/CHoCH found - reject signal
                self.logger.info(f"   ❌ No BOS/CHoCH detected on HTF - SIGNAL REJECTED")
                self.logger.info(f"   ℹ️  Swing structure: {trend_analysis['trend']} (not sufficient without BOS/CHoCH)")
                return None

            self.logger.info(f"   ✅ HTF Trend confirmed: {final_trend} (strength: {final_strength*100:.0f}%)")

            # PHASE 2.6.1: Multi-filter approach for signal frequency optimization
            # Filter 1: Exclude UNKNOWN HTF (0% strength, 28.0% WR)
            exclude_unknown_htf = getattr(self.config, 'SMC_EXCLUDE_UNKNOWN_HTF', True)
            if exclude_unknown_htf and final_strength == 0:
                self.logger.info(f"\n❌ HTF DATA QUALITY FILTER: Signal rejected")
                self.logger.info(f"   HTF strength: 0% (UNKNOWN)")
                self.logger.info(f"   💡 UNKNOWN HTF signals: 28.0% WR vs 60% HTF: 32.6% WR")
                self.logger.info(f"   💡 Insufficient HTF data for reliable trend context")
                return None

            # Filter 2: HTF strength minimum (v2.6.0 → v2.6.1: 75% → 60%)
            # Phase 2.6.1: Lowered to 60% to increase signal frequency
            # All signals are either 0% or 60% (no distribution) - need to fix HTF calc in Phase 2.6.2
            min_htf_strength = getattr(self.config, 'SMC_MIN_HTF_STRENGTH', 0.60)
            if final_strength < min_htf_strength:
                self.logger.info(f"\n❌ HTF STRENGTH FILTER: Signal rejected")
                self.logger.info(f"   Current HTF strength: {final_strength*100:.0f}%")
                self.logger.info(f"   Minimum required: {min_htf_strength*100:.0f}%")
                self.logger.info(f"   💡 Phase 2.6.1: Threshold lowered to {min_htf_strength*100:.0f}% for more signals")
                return None

            self.logger.info(f"   ✅ HTF strength filter passed: {final_strength*100:.0f}% >= {min_htf_strength*100:.0f}%")

            # Initialize direction_str from final_trend (may be overridden by BOS/CHoCH later)
            direction_str = 'bullish' if final_trend == 'BULL' else 'bearish'

            # TIER 1 FILTER: Pullback Momentum Validator
            self.logger.info(f"\n🎯 TIER 1 FILTER: Validating Pullback Momentum (15m timeframe)")
            momentum_valid, momentum_reason = self._validate_pullback_momentum(
                df_15m=df_15m,
                trade_direction=final_trend
            )

            if not momentum_valid:
                self.logger.info(f"   ❌ {momentum_reason} - SIGNAL REJECTED")
                self.logger.info(f"   💡 Counter-momentum entry detected - waiting for aligned candles")
                return None
            else:
                self.logger.info(f"   ✅ {momentum_reason}")

            # TIER 1 FILTER: Swing Proximity Validator (v2.7.1)
            # Replaces failed Premium/Discount filter with structure-based logic
            self.logger.info(f"\n📏 TIER 1 FILTER: Validating Swing Proximity (Structure-Based)")

            # Get current price for proximity check
            current_price = df_1h['close'].iloc[-1]

            swing_proximity_valid, proximity_reason = self._validate_swing_proximity(
                current_price=current_price,
                trade_direction=final_trend,
                swing_highs=trend_analysis['swing_highs'],
                swing_lows=trend_analysis['swing_lows'],
                pip_value=pip_value
            )

            if not swing_proximity_valid:
                self.logger.info(f"\n❌ SWING PROXIMITY FILTER: Signal rejected")
                self.logger.info(f"   {proximity_reason}")
                self.logger.info(f"   💡 NOTE: This REPLACES the failed Premium/Discount filter")
                self.logger.info(f"   💡 PD filter used arbitrary zones - this uses REAL swing structure")
                return None
            else:
                self.logger.info(f"   {proximity_reason}")

            # STEP 2: Detect S/R levels (optional - for confidence boost)
            self.logger.info(f"\n🎯 STEP 2: Detecting Support/Resistance Levels")

            levels = self.sr_detector.detect_levels(
                df=df_1h,
                epic=epic,
                lookback=self.sr_lookback
            )

            self.logger.info(f"   Support levels: {len(levels['support'])}")
            self.logger.info(f"   Resistance levels: {len(levels['resistance'])}")
            self.logger.info(f"   Demand zones: {len(levels['demand_zones'])}")
            self.logger.info(f"   Supply zones: {len(levels['supply_zones'])}")

            # Get current price
            current_price = df_1h['close'].iloc[-1]

            # NOTE: Price extreme filter removed (Phase 1 fix)
            # Filter had conceptual flaw: confused "high in trend range" with "at swing extreme"
            # Protection now provided by R:R ratio (2:1 minimum) and stop loss placement
            # Agent analysis: R:R filter is economically-driven (better than geometric range checks)

            # Check which levels we're near (OPTIONAL - for confidence boost)
            if final_trend == 'BULL':
                # For bullish trend, look for pullback to support/demand zones
                relevant_levels = levels['support'] + levels['demand_zones']
                level_type = 'support/demand'
            else:
                # For bearish trend, look for pullback to resistance/supply zones
                relevant_levels = levels['resistance'] + levels['supply_zones']
                level_type = 'resistance/supply'

            nearest_level = self.sr_detector.find_nearest_level(
                current_price=current_price,
                levels=relevant_levels,
                proximity_pips=self.sr_proximity_pips,
                pip_value=pip_value
            )

            sr_confluence_boost = 0.0  # Default: no S/R boost
            if nearest_level:
                sr_confluence_boost = 0.15 * nearest_level['strength']  # Up to +15% confidence
                self.logger.info(f"   ✅ Near {level_type} level (BONUS):")
                self.logger.info(f"      Price: {nearest_level['price']:.5f}")
                self.logger.info(f"      Distance: {nearest_level['distance_pips']:.1f} pips")
                self.logger.info(f"      Strength: {nearest_level['strength']*100:.0f}%")
                self.logger.info(f"      Confidence Boost: +{sr_confluence_boost*100:.1f}%")
            else:
                self.logger.info(f"   ℹ️  No nearby {level_type} level (no S/R boost)")
                # Create a minimal level dict for later use
                nearest_level = {
                    'price': current_price,
                    'type': level_type,
                    'strength': 0.0,
                    'distance_pips': 999.0,
                    'touch_count': 0
                }

            # STEP 3: Confirm rejection pattern (optional if BOS/CHoCH mode enabled)
            self.logger.info(f"\n📍 STEP 3: Detecting Rejection Pattern")

            # Get recent bars for pattern detection
            recent_bars = df_1h.tail(self.pattern_lookback_bars)

            rejection_pattern = self.pattern_detector.detect_rejection_pattern(
                df=recent_bars,
                direction=final_trend,
                min_strength=self.min_pattern_strength
            )

            if not rejection_pattern:
                # If patterns are optional (BOS/CHoCH mode), create minimal pattern
                if self.patterns_optional:
                    self.logger.info(f"   ℹ️  No rejection pattern found, but patterns are optional (structure-only mode)")

                    # STEP 3A: Detect BOS/CHoCH on 15m (if available and enabled)
                    bos_choch_info = None
                    if self.bos_choch_enabled and df_15m is not None and len(df_15m) > 0:
                        self.logger.info(f"\n🔄 STEP 3A: Detecting BOS/CHoCH on 15m Timeframe")

                        bos_choch_info = self._detect_bos_choch_15m(df_15m, epic)

                        if bos_choch_info:
                            # DIAGNOSTIC: Log BOS/CHoCH direction for tracking bullish/bearish ratio
                            self.logger.info(f"   🔍 [DIAGNOSTIC] BOS/CHoCH Direction: {bos_choch_info['direction'].upper()}")
                            # Validate HTF alignment
                            htf_aligned = self._validate_htf_alignment(
                                bos_direction=bos_choch_info['direction'],
                                df_1h=df_1h,
                                df_4h=df_4h,
                                epic=epic
                            )

                            if not htf_aligned:
                                self.logger.info(f"   ❌ BOS/CHoCH detected but HTF not aligned - SIGNAL REJECTED")
                                # DIAGNOSTIC: Track bearish rejection reasons
                                if bos_choch_info['direction'] == 'bearish':
                                    self.logger.info(f"   🔍 [BEARISH DIAGNOSTIC] Rejected at HTF alignment check")
                                return None

                            # NEW: Order Block Re-entry Logic (v2.2.0)
                            if self.ob_reentry_enabled:
                                self.logger.info(f"\n📦 STEP 3B: Order Block Re-entry Detection")

                                # Identify last opposing Order Block
                                last_ob = self._identify_last_opposing_ob(
                                    df_15m=df_15m,
                                    bos_index=len(df_15m) - 1,
                                    bos_direction=bos_choch_info['direction'],
                                    pip_value=pip_value
                                )

                                if not last_ob:
                                    self.logger.info(f"   ❌ No opposing Order Block found before BOS - SIGNAL REJECTED")
                                    self.logger.info(f"   💡 Institutional accumulation zone not identified")
                                    # DIAGNOSTIC: Track bearish rejection reasons
                                    if bos_choch_info['direction'] == 'bearish':
                                        self.logger.info(f"   🔍 [BEARISH DIAGNOSTIC] Rejected - no opposing OB")
                                    return None

                                self.logger.info(f"   ✅ Order Block identified:")
                                self.logger.info(f"      Type: {last_ob['type']}")
                                self.logger.info(f"      Level: {last_ob['low']:.5f} - {last_ob['high']:.5f}")
                                self.logger.info(f"      Size: {last_ob['size_pips']:.1f} pips")
                                self.logger.info(f"      Re-entry zone: {last_ob['reentry_low']:.5f} - {last_ob['reentry_high']:.5f}")

                                # v2.9.0: ATR Displacement Filter (OpenAI Priority 2)
                                self.logger.info(f"\n📈 STEP 3A2: ATR Displacement Filter (v2.9.0)")
                                atr_valid, atr_reason, atr_info = self._validate_atr_displacement(
                                    df=df_15m,
                                    direction=bos_choch_info['direction'],
                                    pip_value=pip_value
                                )

                                if not atr_valid:
                                    self.logger.info(f"   {atr_reason}")
                                    return None

                                self.logger.info(f"   {atr_reason}")

                                # v2.9.0: Unmitigated OB Check (OpenAI Priority 3)
                                self.logger.info(f"\n📦 STEP 3A3: Unmitigated OB Filter (v2.9.0)")
                                is_mitigated, mitigation_reason = self._is_ob_mitigated(
                                    df=df_15m,
                                    order_block=last_ob,
                                    pip_value=pip_value
                                )

                                if is_mitigated:
                                    self.logger.info(f"   {mitigation_reason}")
                                    return None

                                self.logger.info(f"   {mitigation_reason}")

                                # Check if price has retraced to OB zone
                                current_low = float(df_15m['low'].iloc[-1])
                                current_high = float(df_15m['high'].iloc[-1])

                                in_ob_zone = self._is_price_in_ob_zone(
                                    current_price=current_price,
                                    current_low=current_low,
                                    current_high=current_high,
                                    order_block=last_ob
                                )

                                if not in_ob_zone:
                                    # Price hasn't retraced to OB yet - wait
                                    distance_pips = abs(current_price - last_ob['mid']) / pip_value
                                    self.logger.info(f"   ⏳ Waiting for retracement to OB ({distance_pips:.1f} pips away)")
                                    return None

                                self.logger.info(f"   ✅ Price in OB re-entry zone")

                                # Check for rejection at OB
                                rejection_signal = self._detect_ob_rejection(
                                    df_15m=df_15m,
                                    direction=bos_choch_info['direction'],
                                    ob_level=last_ob['mid']
                                )

                                if not rejection_signal:
                                    self.logger.info(f"   ⏳ Waiting for rejection signal at OB")
                                    return None

                                self.logger.info(f"   ✅ OB Rejection detected:")
                                self.logger.info(f"      Type: {rejection_signal['type']}")
                                self.logger.info(f"      Strength: {rejection_signal['strength']*100:.0f}%")

                                # STEP 3B2: Pullback Depth Filter (Phase 3 - Entry Timing)
                                # Validates price has ACTUALLY retraced from BOS swing, not just IN OB zone
                                self.logger.info(f"\n📐 STEP 3B2: Pullback Depth Filter (Entry Timing Optimization)")

                                pullback_valid, pullback_reason = self._validate_pullback_depth(
                                    current_price=current_price,
                                    bos_direction=bos_choch_info['direction'],
                                    df_15m=df_15m,
                                    order_block=last_ob,
                                    pip_value=pip_value
                                )

                                if not pullback_valid:
                                    self.logger.info(f"   {pullback_reason}")
                                    return None

                                self.logger.info(f"   {pullback_reason}")

                                # STEP 3C: Premium/Discount Zone Validation for Entry Timing
                                self.logger.info(f"\n💎 STEP 3C: Premium/Discount Zone Entry Timing")

                                zone_info = self.market_structure.get_premium_discount_zone(
                                    df=df_15m,
                                    current_price=current_price,
                                    lookback_bars=50
                                )

                                if zone_info:
                                    zone = zone_info['zone']
                                    direction = bos_choch_info['direction']

                                    # Convert range size to pips
                                    range_pips = zone_info['range_size_pips'] / pip_value

                                    self.logger.info(f"   📊 15m Range: {range_pips:.1f} pips")
                                    self.logger.info(f"      High: {zone_info['range_high']:.5f}")
                                    self.logger.info(f"      Mid: {zone_info['range_mid']:.5f}")
                                    self.logger.info(f"      Low: {zone_info['range_low']:.5f}")
                                    self.logger.info(f"   📍 Current Zone: {zone.upper()}")
                                    self.logger.info(f"   📈 Price Position: {zone_info['price_position']*100:.1f}%")

                                    # PHASE 2.6.4: DIRECTIONAL ZONE FILTER
                                    # Prevent selling at bottoms (SELL in discount) and buying at tops (BUY in premium)
                                    # Analysis: Premium SELL = 45.8% WR, Discount SELL = 16.7% WR
                                    directional_zone_filter = getattr(self.config, 'SMC_DIRECTIONAL_ZONE_FILTER', False)

                                    if directional_zone_filter:
                                        entry_quality_buy = zone_info.get('entry_quality_buy', 0.0)
                                        entry_quality_sell = zone_info.get('entry_quality_sell', 0.0)

                                        # Check SELL signals (bearish direction)
                                        if direction == 'bearish':
                                            if zone == 'discount':
                                                # REJECT: SELL in discount = selling at the bottom
                                                self.logger.info(f"\n❌ DIRECTIONAL ZONE FILTER: SELL signal rejected")
                                                self.logger.info(f"   Reason: SELL in DISCOUNT zone (selling at market low)")
                                                self.logger.info(f"   Zone: {zone.upper()} (bottom 33% of range)")
                                                self.logger.info(f"   Price Position: {zone_info['price_position']*100:.1f}%")
                                                self.logger.info(f"   Entry quality for SELL: {entry_quality_sell*100:.0f}%")
                                                self.logger.info(f"   💡 Smart money SELLs in PREMIUM zones (top 33%), not at lows")
                                                self.logger.info(f"   💡 Analysis: Premium SELL = 45.8% WR vs Discount SELL = 16.7% WR")
                                                return None
                                            elif zone == 'premium':
                                                self.logger.info(f"   ✅ DIRECTIONAL ZONE: SELL in PREMIUM (selling high)")
                                                self.logger.info(f"   Entry quality: {entry_quality_sell*100:.0f}%")
                                            else:  # equilibrium
                                                self.logger.info(f"   ⚠️  DIRECTIONAL ZONE: SELL in EQUILIBRIUM (neutral)")
                                                self.logger.info(f"   Entry quality: {entry_quality_sell*100:.0f}%")

                                        # Check BUY signals (bullish direction)
                                        else:  # direction == 'bullish'
                                            if zone == 'premium':
                                                # REJECT: BUY in premium = buying at the top
                                                self.logger.info(f"\n❌ DIRECTIONAL ZONE FILTER: BUY signal rejected")
                                                self.logger.info(f"   Reason: BUY in PREMIUM zone (buying at market high)")
                                                self.logger.info(f"   Zone: {zone.upper()} (top 33% of range)")
                                                self.logger.info(f"   Price Position: {zone_info['price_position']*100:.1f}%")
                                                self.logger.info(f"   Entry quality for BUY: {entry_quality_buy*100:.0f}%")
                                                self.logger.info(f"   💡 Smart money BUYs in DISCOUNT zones (bottom 33%), not at highs")
                                                self.logger.info(f"   💡 SELL in premium = 45.8% WR, BUY in premium = likely poor")
                                                return None
                                            elif zone == 'discount':
                                                self.logger.info(f"   ✅ DIRECTIONAL ZONE: BUY in DISCOUNT (buying low)")
                                                self.logger.info(f"   Entry quality: {entry_quality_buy*100:.0f}%")
                                            else:  # equilibrium
                                                self.logger.info(f"   ⚠️  DIRECTIONAL ZONE: BUY in EQUILIBRIUM (neutral)")
                                                self.logger.info(f"   Entry quality: {entry_quality_buy*100:.0f}%")

                                    # Legacy validation (disabled when premium_zone_only=True)
                                    if direction == 'bullish':
                                        entry_quality = zone_info['entry_quality_buy']
                                        if zone == 'premium':
                                            self.logger.info(f"   📍 BULLISH entry in PREMIUM zone")
                                            self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}%")
                                        elif zone == 'equilibrium':
                                            self.logger.info(f"   📍 BULLISH entry in EQUILIBRIUM zone")
                                            self.logger.info(f"   💡 Entry quality: {entry_quality*100:.0f}%")
                                        else:
                                            self.logger.info(f"   📍 BULLISH entry in DISCOUNT zone")
                                            self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}%")
                                    else:  # bearish
                                        entry_quality = zone_info['entry_quality_sell']
                                        if zone == 'premium':
                                            self.logger.info(f"   📍 BEARISH entry in PREMIUM zone")
                                            self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}%")
                                        elif zone == 'equilibrium':
                                            self.logger.info(f"   📍 BEARISH entry in EQUILIBRIUM zone")
                                            self.logger.info(f"   💡 Entry quality: {entry_quality*100:.0f}%")
                                        else:
                                            self.logger.info(f"   📍 BEARISH entry in DISCOUNT zone")
                                            self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}%")
                                else:
                                    self.logger.info(f"   ⚠️  Could not calculate premium/discount zones")

                                # Use OB level for entry, not BOS level
                                rejection_level = last_ob['mid']
                                direction_str = bos_choch_info['direction']

                                self.logger.info(f"\n✅ ORDER BLOCK RE-ENTRY CONFIRMED:")
                                self.logger.info(f"   Entry: {current_price:.5f}")
                                self.logger.info(f"   OB Level: {rejection_level:.5f}")
                                self.logger.info(f"   BOS Type: {bos_choch_info['type']}")
                                self.logger.info(f"   Significance: {bos_choch_info['significance']*100:.0f}%")

                            else:
                                # Fallback to old logic if OB re-entry disabled
                                # Check if price is in re-entry zone
                                in_reentry_zone = self._check_reentry_zone(
                                    current_price=current_price,
                                    structure_level=bos_choch_info['level'],
                                    pip_value=pip_value
                                )

                                if not in_reentry_zone:
                                    distance_pips = abs(current_price - bos_choch_info['level']) / pip_value
                                    self.logger.info(f"   ⏳ Price not in re-entry zone ({distance_pips:.1f} pips from BOS level) - waiting for pullback")
                                    return None

                                # Use BOS/CHoCH level as structure
                                rejection_level = bos_choch_info['level']
                                direction_str = bos_choch_info['direction']

                                self.logger.info(f"   ✅ BOS/CHoCH confirmed with HTF alignment and re-entry:")
                                self.logger.info(f"      Type: {bos_choch_info['type']}")
                                self.logger.info(f"      Direction: {direction_str}")
                                self.logger.info(f"      Level: {rejection_level:.5f}")
                                self.logger.info(f"      Significance: {bos_choch_info['significance']*100:.0f}%")
                        else:
                            self.logger.info(f"   ℹ️  No BOS/CHoCH detected on 15m, using fallback structure")
                            # Fallback to recent swing
                            if final_trend == 'BULL':
                                rejection_level = df_1h['low'].tail(10).min()
                                direction_str = 'bullish'
                            else:
                                rejection_level = df_1h['high'].tail(10).max()
                                direction_str = 'bearish'
                    else:
                        # No 15m data or BOS/CHoCH disabled, use recent swing
                        self.logger.info(f"   ℹ️  BOS/CHoCH detection disabled or no 15m data, using recent swing")
                        if final_trend == 'BULL':
                            rejection_level = df_1h['low'].tail(10).min()
                            direction_str = 'bullish'
                        else:
                            rejection_level = df_1h['high'].tail(10).max()
                            direction_str = 'bearish'

                    # STEP 3B: Use Zero Lag Liquidity for precise entry timing (if enabled)
                    if self.use_zero_lag_entry:
                        self.logger.info(f"\n💧 STEP 3B: Checking Zero Lag Liquidity Entry Trigger")

                        zero_lag_signal = self.zero_lag.get_entry_signal(
                            df=df_1h,
                            structure_level=rejection_level,
                            direction=direction_str,
                            pip_value=pip_value
                        )

                        if zero_lag_signal:
                            self.logger.info(f"   ✅ Zero Lag entry trigger detected:")
                            self.logger.info(f"      Type: {zero_lag_signal['type']}")
                            self.logger.info(f"      Signal: {zero_lag_signal['signal']}")
                            self.logger.info(f"      Entry: {zero_lag_signal['entry_price']:.5f}")
                            self.logger.info(f"      Confidence: {zero_lag_signal['confidence']*100:.0f}%")
                            self.logger.info(f"      Description: {zero_lag_signal['description']}")

                            # Use Zero Lag entry
                            rejection_pattern = {
                                'pattern_type': f"zero_lag_{zero_lag_signal['type']}",
                                'strength': zero_lag_signal['confidence'],
                                'entry_price': zero_lag_signal['entry_price'],
                                'rejection_level': rejection_level,
                                'description': zero_lag_signal['description']
                            }
                        else:
                            # No Zero Lag trigger yet, wait for better entry
                            self.logger.info(f"   ⏳ No Zero Lag entry trigger at structure level - waiting for better entry")
                            return None
                    else:
                        # Zero Lag disabled, use immediate entry
                        rejection_pattern = {
                            'pattern_type': 'structure_only',
                            'strength': 0.5,  # Moderate strength for structure-only
                            'entry_price': current_price,
                            'rejection_level': rejection_level,
                            'description': 'Structure-based entry (no specific pattern)'
                        }

                        self.logger.info(f"   ✅ Structure-based entry (no pattern required):")
                        self.logger.info(f"      Entry: {rejection_pattern['entry_price']:.5f}")
                        self.logger.info(f"      Rejection Level: {rejection_pattern['rejection_level']:.5f}")
                else:
                    self.logger.info(f"   ❌ No strong rejection pattern (min strength {self.min_pattern_strength*100:.0f}%) - SIGNAL REJECTED")
                    return None
            else:
                self.logger.info(f"   ✅ Rejection pattern detected:")
                self.logger.info(f"      Type: {rejection_pattern['pattern_type']}")
                self.logger.info(f"      Strength: {rejection_pattern['strength']*100:.0f}%")
                self.logger.info(f"      Entry: {rejection_pattern['entry_price']:.5f}")
                self.logger.info(f"      Rejection Level: {rejection_pattern['rejection_level']:.5f}")
                self.logger.info(f"      Description: {rejection_pattern['description']}")

            # STEP 3C_UNIVERSAL: Universal Pullback Filter for ALL entries (Phase 3 Entry Timing)
            # This applies to entries NOT going through OB re-entry path (majority of signals)
            self.logger.info(f"\n📐 STEP 3C_UNIVERSAL: Pullback Depth Filter (Entry Timing)")

            pullback_valid, pullback_reason = self._validate_pullback_depth_universal(
                current_price=current_price,
                direction=direction_str,
                df=df_15m if df_15m is not None and len(df_15m) > 0 else df_1h,
                pip_value=pip_value
            )

            if not pullback_valid:
                self.logger.info(f"   {pullback_reason}")
                return None

            self.logger.info(f"   {pullback_reason}")

            # STEP 3D: Premium/Discount Zone Entry Timing (UNIVERSAL CHECK FOR ALL ENTRIES)
            self.logger.info(f"\n💎 STEP 3D: Premium/Discount Zone Entry Timing Validation")

            zone_info = self.market_structure.get_premium_discount_zone(
                df=df_15m if df_15m is not None and len(df_15m) > 0 else df_1h,
                current_price=current_price,
                lookback_bars=50
            )

            if zone_info:
                zone = zone_info['zone']

                # Convert range size to pips
                range_pips = zone_info['range_size_pips'] / pip_value

                self.logger.info(f"   📊 Range Analysis (last 50 bars): {range_pips:.1f} pips")
                self.logger.info(f"      High: {zone_info['range_high']:.5f}")
                self.logger.info(f"      Mid: {zone_info['range_mid']:.5f}")
                self.logger.info(f"      Low: {zone_info['range_low']:.5f}")
                self.logger.info(f"   📍 Current Zone: {zone.upper()}")
                self.logger.info(f"   📈 Price Position: {zone_info['price_position']*100:.1f}% of range")

                # PHASE 2.6.1: PREMIUM ZONE ONLY filter (Universal check)
                # Analysis: Premium = 45.8% WR, Discount = 16.7% WR, Equilibrium = 15.4% WR
                premium_zone_only = getattr(self.config, 'SMC_PREMIUM_ZONE_ONLY', False)

                self.logger.info(f"   🎯 HTF Trend Context: {final_trend} (strength: {final_strength*100:.0f}%)")

                # PHASE 2.6.4: DIRECTIONAL ZONE FILTER (Universal check for all entries)
                # Prevent selling at bottoms (SELL in discount) and buying at tops (BUY in premium)
                directional_zone_filter = getattr(self.config, 'SMC_DIRECTIONAL_ZONE_FILTER', False)

                if directional_zone_filter:
                    entry_quality_buy = zone_info.get('entry_quality_buy', 0.0)
                    entry_quality_sell = zone_info.get('entry_quality_sell', 0.0)

                    # Check SELL signals (bearish direction)
                    if direction_str == 'bearish':
                        if zone == 'discount':
                            # REJECT: SELL in discount = selling at the bottom
                            self.logger.info(f"\n❌ DIRECTIONAL ZONE FILTER: SELL signal rejected")
                            self.logger.info(f"   Reason: SELL in DISCOUNT zone (selling at market low)")
                            self.logger.info(f"   Zone: {zone.upper()} (bottom 33% of range)")
                            self.logger.info(f"   Price Position: {zone_info['price_position']*100:.1f}%")
                            self.logger.info(f"   Entry quality for SELL: {entry_quality_sell*100:.0f}%")
                            self.logger.info(f"   💡 Smart money SELLs in PREMIUM zones (top 33%), not at lows")
                            self.logger.info(f"   💡 Analysis: Premium SELL = 45.8% WR vs Discount SELL = 16.7% WR")
                            return None
                        elif zone == 'premium':
                            self.logger.info(f"   ✅ DIRECTIONAL ZONE: SELL in PREMIUM (selling high)")
                            self.logger.info(f"   Entry quality: {entry_quality_sell*100:.0f}%")
                        else:  # equilibrium
                            self.logger.info(f"   ⚠️  DIRECTIONAL ZONE: SELL in EQUILIBRIUM (neutral)")
                            self.logger.info(f"   Entry quality: {entry_quality_sell*100:.0f}%")

                    # Check BUY signals (bullish direction)
                    else:  # direction_str == 'bullish'
                        if zone == 'premium':
                            # REJECT: BUY in premium = buying at the top
                            self.logger.info(f"\n❌ DIRECTIONAL ZONE FILTER: BUY signal rejected")
                            self.logger.info(f"   Reason: BUY in PREMIUM zone (buying at market high)")
                            self.logger.info(f"   Zone: {zone.upper()} (top 33% of range)")
                            self.logger.info(f"   Price Position: {zone_info['price_position']*100:.1f}%")
                            self.logger.info(f"   Entry quality for BUY: {entry_quality_buy*100:.0f}%")
                            self.logger.info(f"   💡 Smart money BUYs in DISCOUNT zones (bottom 33%), not at highs")
                            self.logger.info(f"   💡 SELL in premium = 45.8% WR, BUY in premium = likely poor")
                            return None
                        elif zone == 'discount':
                            self.logger.info(f"   ✅ DIRECTIONAL ZONE: BUY in DISCOUNT (buying low)")
                            self.logger.info(f"   Entry quality: {entry_quality_buy*100:.0f}%")
                        else:  # equilibrium
                            self.logger.info(f"   ⚠️  DIRECTIONAL ZONE: BUY in EQUILIBRIUM (neutral)")
                            self.logger.info(f"   Entry quality: {entry_quality_buy*100:.0f}%")

                if premium_zone_only:
                    # Filter 3: Only accept PREMIUM zone signals (both BULL and BEAR)
                    if zone != 'premium':
                        self.logger.info(f"\n❌ PREMIUM ZONE FILTER: Signal rejected")
                        self.logger.info(f"   Current zone: {zone.upper()}")
                        self.logger.info(f"   💡 Performance by zone:")
                        self.logger.info(f"      Premium: 45.8% WR (16/35 winners)")
                        self.logger.info(f"      Discount: 16.7% WR (4/24 winners)")
                        self.logger.info(f"      Equilibrium: 15.4% WR (2/13 winners)")
                        self.logger.info(f"   💡 Phase 2.6.1: Premium zone only for signal quality")
                        return None

                    self.logger.info(f"   ✅ PREMIUM ZONE: Signal in highest probability zone (45.8% WR)")

                # Display entry quality information (no rejection)
                if direction_str == 'bullish':
                    entry_quality = zone_info['entry_quality_buy']
                    self.logger.info(f"   📍 BULLISH entry in {zone.upper()} zone")
                    self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}%")
                else:  # bearish
                    entry_quality = zone_info['entry_quality_sell']
                    self.logger.info(f"   📍 BEARISH entry in {zone.upper()} zone")
                    self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}%")
            else:
                self.logger.info(f"   ⚠️  Could not calculate premium/discount zones - proceeding without zone filter")

            # STEP 3D2: Liquidity Sweep Filter (Phase 2.6.5)
            # SMC Concept: Smart money takes liquidity (stops) before reversing
            # SELL: Must sweep recent highs (liquidity above resistance)
            # BUY: Must sweep recent lows (liquidity below support)
            self.logger.info(f"\n💧 STEP 3D2: Liquidity Sweep Filter (SMC Smart Money Concept)")

            # Use the correct dataframe (15m preferred, fallback to 1h)
            entry_df = df_15m if df_15m is not None and len(df_15m) > 0 else df_1h

            # Detect liquidity sweep
            liquidity_sweep = self._detect_liquidity_sweep(
                df=entry_df,
                direction=direction_str,
                current_idx=len(entry_df) - 1,  # Current bar index
                pair=pair
            )

            if not liquidity_sweep['has_sweep']:
                sweep_type = liquidity_sweep['sweep_type']
                sweep_level = liquidity_sweep['sweep_level']
                bars_since = liquidity_sweep['bars_since_sweep']

                self.logger.info(f"   ❌ LIQUIDITY SWEEP FILTER: Signal rejected")
                self.logger.info(f"   Direction: {direction_str.upper()}")
                self.logger.info(f"   Sweep type: {sweep_type.upper()}")

                if sweep_level > 0:
                    if direction_str == 'bearish':
                        self.logger.info(f"   Recent swing high: {sweep_level:.5f}")
                        self.logger.info(f"   Reason: SELL signal requires recent high to be swept (liquidity grab)")
                    else:
                        self.logger.info(f"   Recent swing low: {sweep_level:.5f}")
                        self.logger.info(f"   Reason: BUY signal requires recent low to be swept (liquidity grab)")

                    if bars_since > 0:
                        self.logger.info(f"   Bars since sweep: {bars_since} (minimum: {getattr(self.config, 'SMC_LIQUIDITY_SWEEP_MIN_BARS', 2)})")
                        self.logger.info(f"   💡 Sweep too recent - need {getattr(self.config, 'SMC_LIQUIDITY_SWEEP_MIN_BARS', 2) - bars_since} more bars for reversal")
                    else:
                        self.logger.info(f"   💡 No liquidity sweep detected in last {getattr(self.config, 'SMC_LIQUIDITY_SWEEP_MAX_BARS', 10)} bars")
                else:
                    self.logger.info(f"   💡 No swing {sweep_type} found in lookback period ({getattr(self.config, 'SMC_LIQUIDITY_SWEEP_LOOKBACK', 20)} bars)")

                self.logger.info(f"   💡 SMC Concept: Smart money takes liquidity before reversing")
                self.logger.info(f"   💡 Phase 2.6.4 Directional filter: 60% HTF + directional = 28 signals, 21.4% WR (poor)")
                self.logger.info(f"   💡 Phase 2.6.5 adds liquidity sweep for quality confirmation")
                return None

            # Log successful sweep or disabled state
            sweep_type = liquidity_sweep['sweep_type']
            sweep_level = liquidity_sweep['sweep_level']
            bars_since = liquidity_sweep['bars_since_sweep']

            # Check if filter is disabled (sweep_type='none' indicates bypass)
            if sweep_type == 'none':
                self.logger.info(f"   ⏭️  SKIPPED: Filter disabled (SMC_LIQUIDITY_SWEEP_ENABLED = False)")
            else:
                self.logger.info(f"   ✅ LIQUIDITY SWEEP DETECTED: {sweep_type.upper()} swept")
                if direction_str == 'bearish':
                    self.logger.info(f"   SELL Setup: Recent high {sweep_level:.5f} taken out {bars_since} bars ago")
                    self.logger.info(f"   💡 Smart money grabbed liquidity above resistance → now reversing")
                else:
                    self.logger.info(f"   BUY Setup: Recent low {sweep_level:.5f} taken out {bars_since} bars ago")
                    self.logger.info(f"   💡 Smart money grabbed liquidity below support → now reversing")

            # STEP 3E: Equilibrium Zone Confidence Filter (Phase 2.3)
            # Neutral zones require higher confidence due to lack of zone edge
            if zone_info and zone_info['zone'] == 'equilibrium':
                # Calculate preliminary confidence to check threshold
                # Note: rr_ratio not yet calculated, use minimum value (2.0) for preliminary check
                htf_score = trend_analysis['strength'] * 0.4
                pattern_score = rejection_pattern['strength'] * 0.3
                sr_score = nearest_level['strength'] * 0.2
                rr_score = min(self.min_rr_ratio / 4.0, 1.0) * 0.1  # Use min_rr_ratio as placeholder
                preliminary_confidence = htf_score + pattern_score + sr_score + rr_score

                MIN_EQUILIBRIUM_CONFIDENCE = 0.65  # v2.9.1: Lowered from 75% to allow more signals through OpenAI quality filters

                if preliminary_confidence < MIN_EQUILIBRIUM_CONFIDENCE:
                    self.logger.info(f"\n🎯 STEP 3E: Equilibrium Zone Confidence Filter")
                    self.logger.info(f"   ❌ EQUILIBRIUM entry with insufficient confidence")
                    self.logger.info(f"   📊 Confidence: {preliminary_confidence*100:.0f}% < {MIN_EQUILIBRIUM_CONFIDENCE*100:.0f}% (minimum for neutral zones)")
                    self.logger.info(f"   💡 Neutral zone = no edge → requires stronger confluences")
                    return None
                else:
                    self.logger.info(f"\n🎯 STEP 3E: Equilibrium Zone Confidence Filter")
                    self.logger.info(f"   ✅ Sufficient confidence for equilibrium entry: {preliminary_confidence*100:.0f}%")

            # STEP 4: Calculate structure-based stop loss
            self.logger.info(f"\n🛑 STEP 4: Calculating Structure-Based Stop Loss")

            # Stop loss goes beyond the structure that would invalidate the trade
            if final_trend == 'BULL':
                # For longs, stop below rejection level (swing low)
                structure_invalidation = rejection_pattern['rejection_level']
                stop_loss = structure_invalidation - (self.sl_buffer_pips * pip_value)
            else:
                # For shorts, stop above rejection level (swing high)
                structure_invalidation = rejection_pattern['rejection_level']
                stop_loss = structure_invalidation + (self.sl_buffer_pips * pip_value)

            entry_price = rejection_pattern['entry_price']
            risk_pips = abs(entry_price - stop_loss) / pip_value

            # Validate entry vs stop loss relationship (catch logic errors)
            if final_trend == 'BULL':
                if entry_price <= stop_loss:
                    self.logger.error(f"❌ Invalid BULL entry: entry {entry_price:.5f} <= stop {stop_loss:.5f}")
                    self.logger.error(f"   This indicates a bug in pattern entry logic!")
                    return None
            else:  # BEAR
                if entry_price >= stop_loss:
                    self.logger.error(f"❌ Invalid BEAR entry: entry {entry_price:.5f} >= stop {stop_loss:.5f}")
                    self.logger.error(f"   This indicates a bug in pattern entry logic!")
                    return None

            self.logger.info(f"   ✅ Entry price validation passed")

            # Check for duplicate signals (same price within 4 hours)
            is_duplicate, dup_reason = self._is_duplicate_signal(pair, entry_price, current_time)
            if is_duplicate:
                self.logger.info(f"   ⚠️  {dup_reason} - SKIPPING")
                return None
            self.logger.info(f"   Structure Invalidation: {structure_invalidation:.5f}")
            self.logger.info(f"   Stop Loss: {stop_loss:.5f}")
            self.logger.info(f"   Risk: {risk_pips:.1f} pips")

            # STEP 5: Calculate take profit (next structure level)
            self.logger.info(f"\n🎯 STEP 5: Calculating Structure-Based Take Profit")

            # Find next structure level in direction of trade
            if final_trend == 'BULL':
                # For longs, target next resistance/supply zone
                target_levels = levels['resistance'] + levels['supply_zones']
                # Filter levels above entry
                target_levels = [l for l in target_levels if l['price'] > entry_price]
            else:
                # For shorts, target next support/demand zone
                target_levels = levels['support'] + levels['demand_zones']
                # Filter levels below entry
                target_levels = [l for l in target_levels if l['price'] < entry_price]

            if not target_levels:
                self.logger.info(f"   ⚠️  No structure level for TP, using minimum R:R of {self.min_rr_ratio}")
                reward_pips = risk_pips * self.min_rr_ratio
                if final_trend == 'BULL':
                    take_profit = entry_price + (reward_pips * pip_value)
                else:
                    take_profit = entry_price - (reward_pips * pip_value)
            else:
                # Sort by distance and take nearest
                if final_trend == 'BULL':
                    target_levels.sort(key=lambda x: x['price'])
                else:
                    target_levels.sort(key=lambda x: x['price'], reverse=True)

                target_level = target_levels[0]
                take_profit = target_level['price']
                reward_pips = abs(take_profit - entry_price) / pip_value

                self.logger.info(f"   Target Level: {target_level['price']:.5f}")
                self.logger.info(f"   Strength: {target_level['strength']*100:.0f}%")

            # Calculate R:R ratio
            rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0

            self.logger.info(f"   Take Profit: {take_profit:.5f}")
            self.logger.info(f"   Reward: {reward_pips:.1f} pips")
            self.logger.info(f"   R:R Ratio: {rr_ratio:.2f}")

            # STEP 6: Validate R:R ratio
            if rr_ratio < self.min_rr_ratio:
                self.logger.info(f"   ❌ R:R too low ({rr_ratio:.2f} < {self.min_rr_ratio}) - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ R:R meets minimum requirement ({rr_ratio:.2f} >= {self.min_rr_ratio})")

            # STEP 6B: Validate minimum TP (Phase 3.0)
            # Filter tight-range setups where TP is too small for meaningful profit
            if reward_pips < self.min_tp_pips:
                self.logger.info(f"   ❌ TP too small ({reward_pips:.1f} pips < {self.min_tp_pips} pips minimum) - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ TP meets minimum requirement ({reward_pips:.1f} pips >= {self.min_tp_pips} pips)")

            # Calculate partial profit if enabled
            partial_tp = None
            if self.partial_profit_enabled:
                partial_reward_pips = risk_pips * self.partial_profit_rr
                if final_trend == 'BULL':
                    partial_tp = entry_price + (partial_reward_pips * pip_value)
                else:
                    partial_tp = entry_price - (partial_reward_pips * pip_value)

                self.logger.info(f"\n💰 Partial Profit Settings:")
                self.logger.info(f"   Enabled: Yes")
                self.logger.info(f"   Partial TP: {partial_tp:.5f}")
                self.logger.info(f"   Partial R:R: {self.partial_profit_rr}")
                self.logger.info(f"   Close Percent: {self.partial_profit_percent}%")

            # Calculate confidence score (0.0 to 1.0)
            # Based on: HTF strength (40%), pattern strength (30%), SR strength (20%), R:R ratio (10%)
            htf_score = trend_analysis['strength'] * 0.4
            pattern_score = rejection_pattern['strength'] * 0.3
            sr_score = nearest_level['strength'] * 0.2
            rr_score = min(rr_ratio / 4.0, 1.0) * 0.1  # Normalize R:R (4:1 = perfect)

            confidence = htf_score + pattern_score + sr_score + rr_score

            # STEP 6: Confidence Range Filter (v2.8.5 - Quality Optimization)
            # Reject both low-confidence AND overconfident signals
            # Analysis showed 70-76% confidence = 0% WR (overconfident paradox)
            # Optimal range: 50-70% (55-60% had 73% WR in analysis)
            MIN_CONFIDENCE = getattr(self.config, 'SMC_MIN_CONFIDENCE', 0.50)
            MAX_CONFIDENCE = getattr(self.config, 'SMC_MAX_CONFIDENCE', 0.70)

            if confidence < MIN_CONFIDENCE:
                self.logger.info(f"\n🎯 STEP 6: Confidence Range Filter")
                self.logger.info(f"   ❌ Signal confidence too low: {confidence*100:.0f}% < {MIN_CONFIDENCE*100:.0f}%")
                self.logger.info(f"   💡 Minimum confidence required for entry quality")
                self.logger.info(f"   📊 Breakdown: HTF={htf_score*100:.0f}% Pattern={pattern_score*100:.0f}% SR={sr_score*100:.0f}% RR={rr_score*100:.0f}%")
                return None

            if confidence > MAX_CONFIDENCE:
                self.logger.info(f"\n🎯 STEP 6: Confidence Range Filter")
                self.logger.info(f"   ❌ Signal overconfident: {confidence*100:.0f}% > {MAX_CONFIDENCE*100:.0f}%")
                self.logger.info(f"   💡 CONFIDENCE PARADOX: 70-76% signals had 0% WR in v2.8.4 analysis")
                self.logger.info(f"   💡 Overconfidence indicates false positives (misaligned factors)")
                self.logger.info(f"   📊 Breakdown: HTF={htf_score*100:.0f}% Pattern={pattern_score*100:.0f}% SR={sr_score*100:.0f}% RR={rr_score*100:.0f}%")
                return None

            # BUILD SIGNAL
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✅ VALID SMC STRUCTURE SIGNAL DETECTED")
            self.logger.info(f"{'='*70}")

            signal = {
                'strategy': 'SMC_STRUCTURE',
                'signal_type': final_trend,  # For validator/reporting (BULL or BEAR)
                'signal': final_trend,  # For backward compatibility
                'confidence_score': round(confidence, 2),  # 0.0 to 1.0
                'epic': epic,
                'pair': pair,
                'timeframe': '1h',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'partial_tp': partial_tp,
                'partial_percent': self.partial_profit_percent if self.partial_profit_enabled else None,
                'risk_pips': risk_pips,
                'reward_pips': reward_pips,
                'rr_ratio': rr_ratio,
                'timestamp': datetime.now(),

                # Signal details
                'htf_trend': final_trend,
                'htf_strength': trend_analysis['strength'],
                'htf_structure': trend_analysis['structure_type'],
                'in_pullback': trend_analysis['in_pullback'],
                'pullback_depth': trend_analysis['pullback_depth'],

                'sr_level': nearest_level['price'],
                'sr_type': nearest_level['type'],
                'sr_strength': nearest_level['strength'],
                'sr_distance_pips': nearest_level['distance_pips'],

                'pattern_type': rejection_pattern['pattern_type'],
                'pattern_strength': rejection_pattern['strength'],
                'rejection_level': rejection_pattern['rejection_level'],

                # SMC-specific strategy indicators (preserved for alert_history)
                'strategy_indicators': {
                    'bos_choch': {
                        'htf_direction': direction_str,
                        'htf_trend': final_trend,
                        'htf_strength': trend_analysis['strength'],
                        'structure_type': trend_analysis['structure_type']
                    },
                    'htf_data': {
                        'timeframe': self.htf_timeframe,
                        'trend': final_trend,
                        'strength': trend_analysis['strength'],
                        'in_pullback': trend_analysis['in_pullback'],
                        'pullback_depth': trend_analysis['pullback_depth'],
                        'swing_highs': trend_analysis.get('swing_highs', 0),
                        'swing_lows': trend_analysis.get('swing_lows', 0)
                    },
                    'sr_data': {
                        'level_price': nearest_level['price'],
                        'level_type': nearest_level['type'],
                        'level_strength': nearest_level['strength'],
                        'distance_pips': nearest_level['distance_pips'],
                        'touch_count': nearest_level.get('touch_count', 1)
                    },
                    'pattern_data': {
                        'pattern_type': rejection_pattern['pattern_type'],
                        'pattern_strength': rejection_pattern['strength'],
                        'rejection_level': rejection_pattern['rejection_level'],
                        'entry_price': entry_price
                    },
                    'rr_data': {
                        'risk_pips': risk_pips,
                        'reward_pips': reward_pips,
                        'rr_ratio': rr_ratio,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'partial_tp': partial_tp if partial_tp else None,
                        'partial_percent': self.partial_profit_percent if partial_tp else None
                    },
                    'confidence_breakdown': {
                        'total': round(confidence, 4),
                        'htf_score': round(htf_score, 4),
                        'pattern_score': round(pattern_score, 4),
                        'sr_score': round(sr_score, 4),
                        'rr_score': round(rr_score, 4)
                    },
                    'indicator_count': 6,  # bos_choch, HTF, SR, Pattern, R:R, confidence_breakdown
                    'data_source': 'smc_structure_analysis'
                },

                # Readable description
                'description': self._build_signal_description(
                    trend_analysis, nearest_level, rejection_pattern, rr_ratio
                )
            }

            self.logger.info(f"\n📋 Signal Summary:")
            self.logger.info(f"   Direction: {signal['signal']}")
            self.logger.info(f"   Entry: {signal['entry_price']:.5f}")
            self.logger.info(f"   Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
            self.logger.info(f"   Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
            if partial_tp:
                self.logger.info(f"   Partial TP: {partial_tp:.5f} ({self.partial_profit_percent}% at {self.partial_profit_rr}R)")
            self.logger.info(f"   R:R Ratio: {signal['rr_ratio']:.2f}")
            self.logger.info(f"\n   {signal['description']}")
            self.logger.info(f"{'='*70}\n")

            # Record signal for deduplication
            self._record_signal(pair, entry_price, current_time)

            # Update cooldown state after successful signal generation
            self._update_cooldown(pair, current_time)

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting SMC signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_signal_description(
        self,
        trend_analysis: Dict,
        sr_level: Dict,
        pattern: Dict,
        rr_ratio: float
    ) -> str:
        """Build human-readable signal description"""

        desc_parts = []

        # Trend context
        desc_parts.append(f"{trend_analysis['trend']} trend ({trend_analysis['structure_type']})")

        # Pullback context
        if trend_analysis['in_pullback']:
            desc_parts.append(f"pullback to {sr_level['type']}")
        else:
            desc_parts.append(f"at {sr_level['type']}")

        # Pattern
        pattern_name = pattern['pattern_type'].replace('_', ' ')
        desc_parts.append(f"with {pattern_name}")

        # R:R
        desc_parts.append(f"({rr_ratio:.1f}R)")

        return ", ".join(desc_parts).capitalize()

    def _validate_htf_alignment(self, bos_direction: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame, epic: str) -> bool:
        """
        Validate that 1H and 4H timeframes align with BOS/CHoCH direction

        Args:
            bos_direction: 'bullish' or 'bearish' from BOS/CHoCH
            df_1h: 1H DataFrame
            df_4h: 4H DataFrame
            epic: Currency pair

        Returns:
            True if HTF alignment confirmed, False otherwise
        """
        self.logger.info(f"\n🔍 Validating HTF Alignment ({bos_direction} BOS/CHoCH)")

        # Check 1H alignment (if required)
        if self.require_1h_alignment:
            trend_1h = self.trend_analyzer.analyze_trend(
                df=df_1h,
                epic=epic,
                lookback=self.htf_alignment_lookback
            )

            self.logger.info(f"   1H Trend: {trend_1h['trend']} ({trend_1h['strength']*100:.0f}%)")

            # Convert BOS direction to trend type
            expected_trend_1h = 'BULL' if bos_direction == 'bullish' else 'BEAR'

            if trend_1h['trend'] != expected_trend_1h:
                self.logger.info(f"   ❌ 1H trend mismatch (expected {expected_trend_1h}) - SIGNAL REJECTED")
                return False

            self.logger.info(f"   ✅ 1H aligned with {bos_direction} direction")

        # Check 4H alignment (if required)
        if self.require_4h_alignment:
            trend_4h = self.trend_analyzer.analyze_trend(
                df=df_4h,
                epic=epic,
                lookback=self.htf_alignment_lookback
            )

            self.logger.info(f"   4H Trend: {trend_4h['trend']} ({trend_4h['strength']*100:.0f}%)")

            expected_trend_4h = 'BULL' if bos_direction == 'bullish' else 'BEAR'

            if trend_4h['trend'] != expected_trend_4h:
                self.logger.info(f"   ❌ 4H trend mismatch (expected {expected_trend_4h}) - SIGNAL REJECTED")
                return False

            self.logger.info(f"   ✅ 4H aligned with {bos_direction} direction")

        self.logger.info(f"   ✅ HTF Alignment Confirmed")
        return True

    def _check_reentry_zone(self, current_price: float, structure_level: float, pip_value: float) -> bool:
        """
        Check if current price is in re-entry zone around structure level

        Args:
            current_price: Current market price
            structure_level: BOS/CHoCH level price
            pip_value: Pip value for pair

        Returns:
            True if in re-entry zone, False otherwise
        """
        zone_tolerance = self.reentry_zone_pips * pip_value
        distance_from_level = abs(current_price - structure_level)

        in_zone = distance_from_level <= zone_tolerance

        if in_zone:
            distance_pips = distance_from_level / pip_value
            self.logger.info(f"   ✅ Price in re-entry zone ({distance_pips:.1f} pips from structure level)")

        return in_zone

    def _identify_last_opposing_ob(self, df_15m: pd.DataFrame, bos_index: int, bos_direction: str, pip_value: float) -> Optional[Dict]:
        """
        Identify the LAST ORDER BLOCK before BOS that opposes the new direction.

        For BULLISH BOS:
        - Find last BEARISH order block before bullish displacement
        - This is where institutions accumulated longs (created bearish OB as liquidity)

        For BEARISH BOS:
        - Find last BULLISH order block before bearish displacement
        - This is where institutions accumulated shorts

        Args:
            df_15m: 15m timeframe data
            bos_index: Index where BOS occurred
            bos_direction: 'bullish' or 'bearish'
            pip_value: Pip value for the pair

        Returns:
            Dict with OB info or None if no valid OB found
        """
        if not self.ob_reentry_enabled:
            return None

        lookback = min(self.ob_lookback_bars, bos_index)

        # Search backwards from BOS for opposing order block
        for i in range(bos_index - 1, max(0, bos_index - lookback), -1):
            candle = df_15m.iloc[i]

            if bos_direction == 'bullish':
                # Look for bearish OB (consolidation before bullish move)
                # Bearish OB characteristics:
                # - Red candle (close < open)
                # - Followed by bullish displacement
                # - At least 3 pips in size

                if candle['close'] < candle['open']:
                    ob_size_pips = (candle['open'] - candle['close']) / pip_value

                    if ob_size_pips >= 3:
                        # Check if followed by bullish move
                        if i < bos_index - 1:
                            next_candles = df_15m.iloc[i+1:i+4]
                            bullish_move = (next_candles['close'] > next_candles['open']).sum() >= 2

                            if bullish_move:
                                return {
                                    'type': 'bearish',
                                    'index': i,
                                    'high': float(candle['high']),
                                    'low': float(candle['low']),
                                    'open': float(candle['open']),
                                    'close': float(candle['close']),
                                    'mid': float((candle['high'] + candle['low']) / 2),
                                    'reentry_high': float((candle['high'] + candle['low']) / 2),  # Mid-point
                                    'reentry_low': float(candle['low']),  # Bottom of OB
                                    'size_pips': ob_size_pips,
                                    'timestamp': df_15m.index[i]
                                }

            else:  # bearish BOS
                # Look for bullish OB (consolidation before bearish move)
                # Bullish OB characteristics:
                # - Green candle (close > open)
                # - Followed by bearish displacement
                # - At least 3 pips in size

                if candle['close'] > candle['open']:
                    ob_size_pips = (candle['close'] - candle['open']) / pip_value

                    if ob_size_pips >= 3:
                        # Check if followed by bearish move
                        if i < bos_index - 1:
                            next_candles = df_15m.iloc[i+1:i+4]
                            bearish_move = (next_candles['close'] < next_candles['open']).sum() >= 2

                            if bearish_move:
                                return {
                                    'type': 'bullish',
                                    'index': i,
                                    'high': float(candle['high']),
                                    'low': float(candle['low']),
                                    'open': float(candle['open']),
                                    'close': float(candle['close']),
                                    'mid': float((candle['high'] + candle['low']) / 2),
                                    'reentry_high': float(candle['high']),  # Top of OB
                                    'reentry_low': float((candle['high'] + candle['low']) / 2),  # Mid-point
                                    'size_pips': ob_size_pips,
                                    'timestamp': df_15m.index[i]
                                }

        return None  # No valid OB found

    def _is_price_in_ob_zone(self, current_price: float, current_low: float, current_high: float, order_block: Dict) -> bool:
        """
        Check if current price is in Order Block re-entry zone.

        Args:
            current_price: Current close price
            current_low: Current candle low
            current_high: Current candle high
            order_block: OB dict with reentry_high and reentry_low

        Returns:
            True if price has entered OB zone
        """
        reentry_high = order_block['reentry_high']
        reentry_low = order_block['reentry_low']

        # Check if current candle touched or entered the OB zone
        if order_block['type'] == 'bearish':
            # For bullish BOS, wait for retrace to bearish OB (support)
            # Price should come down to OB zone
            return current_low <= reentry_high and current_low >= reentry_low
        else:
            # For bearish BOS, wait for retrace to bullish OB (resistance)
            # Price should come up to OB zone
            return current_high >= reentry_low and current_high <= reentry_high

    def _detect_ob_rejection(self, df_15m: pd.DataFrame, direction: str, ob_level: float) -> Optional[Dict]:
        """
        Detect rejection signals at Order Block level.

        Rejection types:
        1. Wick rejection (60%+ wick, small body)
        2. Engulfing candle
        3. Simple bounce (close back inside OB)

        Args:
            df_15m: 15m timeframe data
            direction: 'bullish' or 'bearish'
            ob_level: Order block level to check rejection from

        Returns:
            Dict with rejection info or None
        """
        if not self.ob_require_rejection:
            return {'type': 'no_confirmation_required', 'strength': 0.50}

        current = df_15m.iloc[-1]
        previous = df_15m.iloc[-2] if len(df_15m) > 1 else None

        if direction == 'bullish':
            # Look for bullish rejection at OB support
            wick_length = current['low'] - min(current['open'], current['close'])
            body_length = abs(current['close'] - current['open'])
            total_length = current['high'] - current['low']

            # 1. Wick rejection (price tested OB and rejected up)
            if total_length > 0 and wick_length > total_length * self.ob_rejection_min_wick and current['close'] > current['open']:
                return {
                    'type': 'wick_rejection',
                    'strength': 0.75,
                    'wick_ratio': wick_length / total_length if total_length > 0 else 0
                }

            # 2. Engulfing pattern
            if previous is not None and current['close'] > previous['open'] and current['open'] < previous['close']:
                return {
                    'type': 'bullish_engulfing',
                    'strength': 0.80
                }

            # 3. Simple bullish close above OB level
            if current['low'] <= ob_level and current['close'] > ob_level:
                return {
                    'type': 'ob_bounce',
                    'strength': 0.65
                }

        else:  # bearish
            # Look for bearish rejection at OB resistance
            wick_length = max(current['open'], current['close']) - current['high']
            body_length = abs(current['close'] - current['open'])
            total_length = current['high'] - current['low']

            # 1. Wick rejection
            if total_length > 0 and wick_length > total_length * self.ob_rejection_min_wick and current['close'] < current['open']:
                return {
                    'type': 'wick_rejection',
                    'strength': 0.75,
                    'wick_ratio': wick_length / total_length if total_length > 0 else 0
                }

            # 2. Bearish engulfing
            if previous is not None and current['close'] < previous['open'] and current['open'] > previous['close']:
                return {
                    'type': 'bearish_engulfing',
                    'strength': 0.80
                }

            # 3. Simple bearish close below OB level
            if current['high'] >= ob_level and current['close'] < ob_level:
                return {
                    'type': 'ob_rejection',
                    'strength': 0.65
                }

        return None  # No rejection detected

    def _calculate_atr(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate Average True Range (ATR) for displacement validation.

        v2.9.0: ATR-Based Displacement Filter (OpenAI Priority 2)
        Used to validate that BOS moves have institutional momentum.

        Args:
            df: DataFrame with high, low, close
            period: ATR period (default 20)

        Returns:
            ATR value
        """
        if len(df) < period + 1:
            return 0.0

        # Calculate True Range
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr_values = []
        for i in range(1, len(df)):
            hl = high[i] - low[i]  # High - Low
            hc = abs(high[i] - close[i-1])  # High - Previous Close
            lc = abs(low[i] - close[i-1])  # Low - Previous Close
            tr = max(hl, hc, lc)
            tr_values.append(tr)

        # Calculate ATR (simple moving average of TR)
        if len(tr_values) >= period:
            atr = sum(tr_values[-period:]) / period
            return float(atr)

        return float(sum(tr_values) / len(tr_values)) if tr_values else 0.0

    def _validate_atr_displacement(self, df: pd.DataFrame, direction: str, pip_value: float) -> tuple:
        """
        v2.9.0: ATR-Based Displacement Filter (OpenAI Priority 2)

        Validates that BOS has meaningful momentum using ATR multiples.
        SMC Concept: Institutional moves create displacement, not small bounces.

        Args:
            df: DataFrame (15m timeframe)
            direction: 'bullish' or 'bearish'
            pip_value: Pip value for the pair

        Returns:
            tuple: (is_valid, reason_string, displacement_info)
        """
        atr_enabled = getattr(self.config, 'SMC_ATR_DISPLACEMENT_ENABLED', True)
        if not atr_enabled:
            return True, "ATR displacement filter disabled", {}

        atr_period = getattr(self.config, 'SMC_ATR_DISPLACEMENT_PERIOD', 20)
        bos_min_atr = getattr(self.config, 'SMC_BOS_MIN_ATR_MULTIPLE', 0.4)

        # Calculate ATR
        atr = self._calculate_atr(df, atr_period)
        if atr == 0:
            return True, "Insufficient data for ATR calculation", {}

        # Get current candle metrics
        current = df.iloc[-1]
        candle_range = current['high'] - current['low']
        body_size = abs(current['close'] - current['open'])

        # Calculate displacement as ATR multiple
        displacement_atr = candle_range / atr if atr > 0 else 0
        body_atr = body_size / atr if atr > 0 else 0

        # Validate minimum displacement
        if displacement_atr < bos_min_atr:
            atr_pips = atr / pip_value
            range_pips = candle_range / pip_value
            min_required = bos_min_atr * atr
            min_pips = min_required / pip_value

            return False, (
                f"❌ ATR DISPLACEMENT FILTER: Insufficient momentum\n"
                f"   ATR({atr_period}): {atr_pips:.1f} pips\n"
                f"   BOS Candle Range: {range_pips:.1f} pips ({displacement_atr:.2f}x ATR)\n"
                f"   Minimum Required: {min_pips:.1f} pips ({bos_min_atr}x ATR)\n"
                f"   💡 Institutional moves create displacement > {bos_min_atr}x ATR\n"
                f"   💡 Small moves often signal false breakouts"
            ), {'atr': atr, 'displacement': displacement_atr}

        atr_pips = atr / pip_value
        range_pips = candle_range / pip_value

        return True, (
            f"✅ ATR DISPLACEMENT: Valid institutional momentum\n"
            f"   ATR({atr_period}): {atr_pips:.1f} pips\n"
            f"   BOS Displacement: {range_pips:.1f} pips ({displacement_atr:.2f}x ATR)\n"
            f"   ✅ Exceeds minimum {bos_min_atr}x ATR threshold"
        ), {'atr': atr, 'displacement': displacement_atr}

    def _is_ob_mitigated(self, df: pd.DataFrame, order_block: Dict, pip_value: float) -> tuple:
        """
        v2.9.0: Unmitigated Order Block Tracking (OpenAI Priority 3)

        Checks if an Order Block has already been "mitigated" (tested).
        SMC Concept: Only fresh, untested OBs provide valid re-entry opportunities.

        An OB is considered mitigated if price has already revisited and
        penetrated into the zone significantly after initial formation.

        Args:
            df: DataFrame with price data after OB formation
            order_block: Order Block dict with high, low, index
            pip_value: Pip value for the pair

        Returns:
            tuple: (is_mitigated, reason_string)
        """
        tracking_enabled = getattr(self.config, 'SMC_UNMITIGATED_OB_TRACKING', True)
        if not tracking_enabled:
            return False, "OB mitigation tracking disabled"

        mitigation_threshold = getattr(self.config, 'SMC_OB_MITIGATION_THRESHOLD', 0.50)

        ob_high = order_block['high']
        ob_low = order_block['low']
        ob_index = order_block['index']
        ob_size = ob_high - ob_low

        if ob_size == 0:
            return False, "OB has zero size - cannot calculate mitigation"

        # Check price action AFTER OB formation
        # Skip the first few candles immediately after OB (impulse move)
        check_start = min(ob_index + 4, len(df) - 1)

        if check_start >= len(df) - 1:
            return False, "Not enough data after OB to check mitigation"

        # Scan for mitigation (price entering the OB zone significantly)
        for i in range(check_start, len(df) - 1):  # Exclude current candle
            candle = df.iloc[i]

            if order_block['type'] == 'bearish':
                # For bearish OB (bullish setup): check if price came UP into OB
                # Mitigation = price high entered OB zone significantly
                if candle['high'] >= ob_low:
                    # Calculate penetration depth
                    penetration = min(candle['high'], ob_high) - ob_low
                    penetration_pct = penetration / ob_size

                    if penetration_pct >= mitigation_threshold:
                        penetration_pips = penetration / pip_value
                        return True, (
                            f"❌ OB MITIGATED: Already tested (not fresh)\n"
                            f"   OB Zone: {ob_low:.5f} - {ob_high:.5f}\n"
                            f"   Mitigated at bar {i} (index)\n"
                            f"   Penetration: {penetration_pips:.1f} pips ({penetration_pct*100:.0f}% of OB)\n"
                            f"   Threshold: {mitigation_threshold*100:.0f}%\n"
                            f"   💡 Mitigated OBs have reduced probability\n"
                            f"   💡 Fresh OBs provide higher-probability entries"
                        )

            else:  # bullish OB
                # For bullish OB (bearish setup): check if price came DOWN into OB
                # Mitigation = price low entered OB zone significantly
                if candle['low'] <= ob_high:
                    # Calculate penetration depth
                    penetration = ob_high - max(candle['low'], ob_low)
                    penetration_pct = penetration / ob_size

                    if penetration_pct >= mitigation_threshold:
                        penetration_pips = penetration / pip_value
                        return True, (
                            f"❌ OB MITIGATED: Already tested (not fresh)\n"
                            f"   OB Zone: {ob_low:.5f} - {ob_high:.5f}\n"
                            f"   Mitigated at bar {i} (index)\n"
                            f"   Penetration: {penetration_pips:.1f} pips ({penetration_pct*100:.0f}% of OB)\n"
                            f"   Threshold: {mitigation_threshold*100:.0f}%\n"
                            f"   💡 Mitigated OBs have reduced probability\n"
                            f"   💡 Fresh OBs provide higher-probability entries"
                        )

        return False, (
            f"✅ OB UNMITIGATED: Fresh opportunity\n"
            f"   OB Zone: {ob_low:.5f} - {ob_high:.5f}\n"
            f"   ✅ No significant penetration since formation\n"
            f"   ✅ First test = highest probability entry"
        )

    def _calculate_bos_quality(self, df_15m: pd.DataFrame, direction: str) -> float:
        """
        Calculate BOS/CHoCH quality score based on candle characteristics.

        Quality factors:
        - Body size (decisive move)
        - Clean break (minimal wick on break side)
        - Volume confirmation (if available)

        Args:
            df_15m: 15m DataFrame
            direction: 'bullish' or 'bearish'

        Returns:
            Quality score 0.0 to 1.0
        """
        current = df_15m.iloc[-1]

        # Calculate candle metrics
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']

        if total_range == 0:
            return 0.0

        # Calculate average candle size for comparison
        recent_candles = df_15m.iloc[-20:]
        avg_body_size = abs(recent_candles['close'] - recent_candles['open']).mean()
        avg_range = (recent_candles['high'] - recent_candles['low']).mean()

        quality_score = 0.0

        # Factor 1: Large decisive body (0.0 to 0.4)
        if avg_body_size > 0:
            body_ratio = body_size / avg_body_size
            if body_ratio >= 1.5:  # 50% larger than average
                quality_score += 0.4
            elif body_ratio >= 1.2:  # 20% larger
                quality_score += 0.25
            elif body_ratio >= 1.0:  # Average or above
                quality_score += 0.15

        # Factor 2: Clean break - minimal wick on break side (0.0 to 0.3)
        if direction == 'bullish':
            # For bullish break, want small upper wick (clean breakout up)
            upper_wick = current['high'] - max(current['open'], current['close'])
            wick_ratio = upper_wick / total_range if total_range > 0 else 1.0

            if wick_ratio <= 0.15:  # Very clean (<15% wick)
                quality_score += 0.3
            elif wick_ratio <= 0.25:  # Decent (<25% wick)
                quality_score += 0.2
            elif wick_ratio <= 0.35:  # Acceptable (<35% wick)
                quality_score += 0.1
        else:  # bearish
            # For bearish break, want small lower wick (clean breakdown)
            lower_wick = min(current['open'], current['close']) - current['low']
            wick_ratio = lower_wick / total_range if total_range > 0 else 1.0

            if wick_ratio <= 0.15:
                quality_score += 0.3
            elif wick_ratio <= 0.25:
                quality_score += 0.2
            elif wick_ratio <= 0.35:
                quality_score += 0.1

        # Factor 3: Volume confirmation (0.0 to 0.3) - if volume data available
        if 'volume' in df_15m.columns:
            current_volume = current['volume']
            avg_volume = recent_candles['volume'].mean()

            if avg_volume > 0 and current_volume > 0:
                volume_ratio = current_volume / avg_volume

                if volume_ratio >= 1.5:  # 50% above average
                    quality_score += 0.3
                elif volume_ratio >= 1.2:  # 20% above average
                    quality_score += 0.2
                elif volume_ratio >= 1.0:  # Average or above
                    quality_score += 0.1

        return min(quality_score, 1.0)  # Cap at 1.0

    def _detect_bos_choch_15m(self, df_15m: pd.DataFrame, epic: str) -> Optional[Dict]:
        """
        Detect BOS/CHoCH on 15m timeframe using fractal-based detection

        Args:
            df_15m: 15m DataFrame
            epic: Currency pair

        Returns:
            Dict with BOS/CHoCH info or None if no break detected
        """
        self.logger.info(f"\n📊 Detecting BOS/CHoCH on 15m timeframe")

        # Use the same approach as HTF: analyze market structure then use fractal fallback
        df_15m_with_structure = self.market_structure.analyze_market_structure(
            df=df_15m,
            config=vars(self.config) if hasattr(self.config, '__dict__') else {},
            epic=epic,
            timeframe='15m'
        )

        # Use fractal-based detection (same as HTF)
        # v2.9.0: Pass config for body-close BOS validation
        bos_config = vars(self.config) if hasattr(self.config, '__dict__') else {}
        bos_choch_direction = self.market_structure.get_last_bos_choch_direction(df_15m_with_structure, bos_config)

        if not bos_choch_direction:
            self.logger.info(f"   ℹ️  No recent BOS/CHoCH detected")
            return None

        # Calculate BOS/CHoCH quality score
        quality_score = self._calculate_bos_quality(df_15m, bos_choch_direction)

        # Require minimum quality threshold
        # OPTIMIZED: Increased from 0.60 to 0.65 based on Test 26 analysis
        # Test 26 had 63 signals (above target) - need more selectivity
        MIN_BOS_QUALITY = 0.65  # 65% minimum quality

        if quality_score < MIN_BOS_QUALITY:
            self.logger.info(f"   ❌ Weak BOS/CHoCH detected - quality too low")
            self.logger.info(f"      Quality: {quality_score*100:.0f}% < {MIN_BOS_QUALITY*100:.0f}% (minimum)")
            self.logger.info(f"      Direction: {bos_choch_direction}")
            self.logger.info(f"      💡 Weak/indecisive structure break - avoiding entry")
            return None

        # Get current price for level
        current_price = float(df_15m['close'].iloc[-1])

        # Determine break type based on direction (simplified - we don't distinguish BOS vs ChoCH here)
        break_type = "BOS"  # Can be refined later to distinguish BOS from ChoCH

        self.logger.info(f"   ✅ {break_type} detected:")
        self.logger.info(f"      Direction: {bos_choch_direction}")
        self.logger.info(f"      Level: {current_price:.5f}")
        self.logger.info(f"      Quality: {quality_score*100:.0f}% (strong)")
        self.logger.info(f"      Detection: Fractal-based with quality filter")

        return {
            'type': break_type,
            'direction': bos_choch_direction,
            'level': current_price,
            'significance': 0.70 + (quality_score * 0.30),  # 0.70 to 1.00 based on quality
            'quality': quality_score,
            'timestamp': df_15m.index[-1] if hasattr(df_15m.index[-1], 'to_pydatetime') else None
        }

    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return "SMC_STRUCTURE"

    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return "Pure structure-based strategy using Smart Money Concepts (price action only)"


def create_smc_structure_strategy(config=None, **kwargs) -> SMCStructureStrategy:
    """
    Factory function to create SMC Structure strategy instance

    Args:
        config: Configuration module (if None, imports config_smc_structure)
        **kwargs: Additional arguments passed to strategy

    Returns:
        SMCStructureStrategy instance
    """
    if config is None:
        # Import config using absolute import from configdata
        # (backtest_cli adds /app/forex_scanner to path, so 'configdata' is accessible)
        from configdata.strategies import config_smc_structure as config

    return SMCStructureStrategy(config=config, **kwargs)
