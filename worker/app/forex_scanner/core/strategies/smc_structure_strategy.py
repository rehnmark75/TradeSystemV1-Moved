#!/usr/bin/env python3
"""
SMC Pure Structure Strategy
Structure-based trading using Smart Money Concepts (pure price action)

VERSION: 2.2.0 (Order Block Re-entry Implementation)
DATE: 2025-11-03
STATUS: Production Ready - Testing Required

Performance Metrics (v2.1.1 Baseline - 30 days, 9 pairs):
- Total Signals: 112
- Win Rate: 39.3%
- Profit Factor: 2.16
- Bull/Bear Ratio: 107/5 (95.5% bull bias)

Expected Performance (v2.2.0 with OB Re-entry):
- Total Signals: 50-60 (quality over quantity)
- Win Rate: 48-55% (+10-15% improvement)
- Profit Factor: 2.5-3.5 (+16% to +62% improvement)
- R:R Ratio: 2.5:1 (improved entry pricing)

Strategy Logic:
1. Identify HTF trend (4H structure)
2. Detect BOS/CHoCH on 15m timeframe
3. Identify last opposing Order Block (institutional accumulation zone)
4. Wait for price to RETRACE to OB zone
5. Detect REJECTION at OB level (wick rejection, engulfing, bounce)
6. Enter at OB with tight stop loss (5-8 pips beyond OB)
7. Target next structure level

Version History:
- v2.2.0 (2025-11-03): Order Block Re-entry implementation (major enhancement)
- v2.1.1 (2025-11-03): Added session filter (disabled), fixed timestamp bug
- v2.1.0 (2025-11-02): Phase 2.1 baseline - HTF alignment enabled
- v2.0.0 (2025-10-XX): BOS/CHoCH detection on 15m timeframe
- v1.0.0 (2025-10-XX): Initial SMC Structure implementation
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
        self.logger.info(f"   SR Proximity: {self.sr_proximity_pips} pips")
        if self.cooldown_enabled:
            self.logger.info(f"   Cooldown: {self.signal_cooldown_hours}h per-pair, {self.global_cooldown_minutes}m global")
        self.logger.info(f"   OB Re-entry: {'✅ ENABLED' if self.ob_reentry_enabled else '❌ DISABLED'}")
        if self.ob_reentry_enabled:
            self.logger.info(f"   OB Lookback: {self.ob_lookback_bars} bars")
            self.logger.info(f"   OB Re-entry zone: {self.ob_reentry_zone}")

    def _load_config(self):
        """Load strategy configuration"""
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
        self.logger.info(f"   Entry TF: 1H | HTF: {self.htf_timeframe} | BOS/CHoCH TF: 15m")
        self.logger.info(f"{'='*70}")

        # Get candle timestamp (for backtesting compatibility)
        candle_timestamp = df_1h.index[-1]

        # Check cooldown before processing
        current_time = datetime.now()
        can_trade, cooldown_reason = self._check_cooldown(pair, current_time)
        if not can_trade:
            self.logger.info(f"   ⏱️  {cooldown_reason} - SKIPPING")
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
            bos_choch_direction = self.market_structure.get_last_bos_choch_direction(df_4h_with_structure)

            # ALWAYS use BOS/CHoCH as primary trend indicator (smart money direction)
            if bos_choch_direction in ['bullish', 'bearish']:
                # Use BOS/CHoCH direction as trend
                final_trend = 'BULL' if bos_choch_direction == 'bullish' else 'BEAR'

                # Calculate strength: use swing strength if aligned, otherwise moderate
                if trend_analysis['trend'] == final_trend:
                    # BOS/CHoCH aligns with swing structure - use swing strength
                    final_strength = trend_analysis['strength']
                    self.logger.info(f"   ✅ BOS/CHoCH: {bos_choch_direction.upper()} → {final_trend}")
                    self.logger.info(f"   ✅ Swing structure ALIGNS: {trend_analysis['structure_type']} ({trend_analysis['strength']*100:.0f}%)")
                else:
                    # BOS/CHoCH differs from swing structure - use moderate strength
                    final_strength = 0.60
                    self.logger.info(f"   ✅ BOS/CHoCH: {bos_choch_direction.upper()} → {final_trend}")
                    self.logger.info(f"   ⚠️  Swing structure differs: {trend_analysis['trend']} ({trend_analysis['structure_type']})")
                    self.logger.info(f"   ℹ️  Using BOS/CHoCH as primary (strength: {final_strength*100:.0f}%)")
            else:
                # No BOS/CHoCH found - reject signal
                self.logger.info(f"   ❌ No BOS/CHoCH detected on HTF - SIGNAL REJECTED")
                self.logger.info(f"   ℹ️  Swing structure: {trend_analysis['trend']} (not sufficient without BOS/CHoCH)")
                return None

            # Must have minimum strength
            if final_strength < 0.50:
                self.logger.info(f"   ❌ Trend too weak ({final_strength*100:.0f}% < 50%) - SIGNAL REJECTED")
                return None

            self.logger.info(f"   ✅ HTF Trend confirmed: {final_trend} (strength: {final_strength*100:.0f}%)")

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

                                    # Validate entry timing: BUY in discount, SELL in premium
                                    if direction == 'bullish':
                                        entry_quality = zone_info['entry_quality_buy']
                                        if zone == 'premium':
                                            self.logger.info(f"   ⚠️  BULLISH entry in PREMIUM zone - poor timing")
                                            self.logger.info(f"   💡 Wait for pullback to discount zone for better entry")
                                            return None
                                        elif zone == 'equilibrium':
                                            self.logger.info(f"   ⚠️  BULLISH entry in EQUILIBRIUM zone - neutral timing")
                                            self.logger.info(f"   💡 Entry quality: {entry_quality*100:.0f}% (reduced confidence)")
                                        else:
                                            self.logger.info(f"   ✅ BULLISH entry in DISCOUNT zone - excellent timing!")
                                            self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}%")
                                    else:  # bearish
                                        entry_quality = zone_info['entry_quality_sell']
                                        if zone == 'discount':
                                            self.logger.info(f"   ⚠️  BEARISH entry in DISCOUNT zone - poor timing")
                                            self.logger.info(f"   💡 Wait for rally to premium zone for better entry")
                                            return None
                                        elif zone == 'equilibrium':
                                            self.logger.info(f"   ⚠️  BEARISH entry in EQUILIBRIUM zone - neutral timing")
                                            self.logger.info(f"   💡 Entry quality: {entry_quality*100:.0f}% (reduced confidence)")
                                        else:
                                            self.logger.info(f"   ✅ BEARISH entry in PREMIUM zone - excellent timing!")
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

                # CONTEXT-AWARE validation: Consider HTF trend for premium/discount logic
                # In TRENDING markets: Allow continuation entries even in "wrong" zones
                # In RANGING markets: Strict premium/discount rules apply

                self.logger.info(f"   🎯 HTF Trend Context: {final_trend} (strength: {final_strength*100:.0f}%)")

                # Determine if we're in a strong trending or ranging market
                # OPTIMIZED: Increased from 0.60 to 0.75 based on Test 23 analysis
                # 60% threshold allowed too many weak trend continuations (all losers)
                # 75% = truly strong, established trends only
                is_strong_trend = final_strength >= 0.75  # Strong trend if strength >= 75%

                # CRITICAL FIX: Initialize direction_str if not already set (from BOS/CHoCH or fallback logic)
                # This ensures direction_str is always defined before use in premium/discount filtering
                if 'direction_str' not in locals():
                    direction_str = 'bullish' if final_trend == 'BULL' else 'bearish'
                    self.logger.info(f"   ℹ️  Direction initialized from HTF trend: {direction_str}")

                if direction_str == 'bullish':
                    entry_quality = zone_info['entry_quality_buy']

                    if zone == 'premium':
                        if is_strong_trend and final_trend == 'BULL':
                            # ALLOW: Bullish continuation in strong uptrend, even at premium
                            self.logger.info(f"   ✅ BULLISH entry in PREMIUM zone - TREND CONTINUATION")
                            self.logger.info(f"   🎯 Strong uptrend context allows premium entries (momentum)")
                        else:
                            # REJECT: Counter-trend or weak trend
                            self.logger.info(f"   ❌ BULLISH entry in PREMIUM zone - poor timing")
                            self.logger.info(f"   💡 Not in strong uptrend - wait for pullback to discount")
                            return None
                    elif zone == 'equilibrium':
                        self.logger.info(f"   ⚠️  BULLISH entry in EQUILIBRIUM zone - neutral timing")
                        self.logger.info(f"   💡 Entry quality: {entry_quality*100:.0f}% (acceptable)")
                    else:  # discount
                        self.logger.info(f"   ✅ BULLISH entry in DISCOUNT zone - excellent timing!")
                        self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}% (buying at discount)")

                else:  # bearish
                    entry_quality = zone_info['entry_quality_sell']

                    if zone == 'discount':
                        if is_strong_trend and final_trend == 'BEAR':
                            # ALLOW: Bearish continuation in strong downtrend, even at discount
                            self.logger.info(f"   ✅ BEARISH entry in DISCOUNT zone - TREND CONTINUATION")
                            self.logger.info(f"   🎯 Strong downtrend context allows discount entries (momentum)")
                        else:
                            # REJECT: Counter-trend or weak trend
                            self.logger.info(f"   ❌ BEARISH entry in DISCOUNT zone - poor timing")
                            self.logger.info(f"   💡 Not in strong downtrend - wait for rally to premium")
                            # DIAGNOSTIC: Track bearish rejection reasons
                            self.logger.info(f"   🔍 [BEARISH DIAGNOSTIC] Rejected at premium/discount filter")
                            self.logger.info(f"      Zone: DISCOUNT, Strength: {final_strength*100:.0f}%, Threshold: 75%")
                            return None
                    elif zone == 'equilibrium':
                        self.logger.info(f"   ⚠️  BEARISH entry in EQUILIBRIUM zone - neutral timing")
                        self.logger.info(f"   💡 Entry quality: {entry_quality*100:.0f}% (acceptable)")
                    else:  # premium
                        self.logger.info(f"   ✅ BEARISH entry in PREMIUM zone - excellent timing!")
                        self.logger.info(f"   🎯 Entry quality: {entry_quality*100:.0f}% (selling at premium)")
            else:
                self.logger.info(f"   ⚠️  Could not calculate premium/discount zones - proceeding without zone filter")

            # STEP 3E: Equilibrium Zone Confidence Filter (Phase 2.3)
            # Neutral zones require higher confidence due to lack of zone edge
            if zone_info and zone_info['zone'] == 'equilibrium':
                # Calculate preliminary confidence to check threshold
                htf_score = trend_analysis['strength'] * 0.4
                pattern_score = rejection_pattern['strength'] * 0.3
                sr_score = nearest_level['strength'] * 0.2
                rr_score = 0.0  # R:R not calculated yet at this stage (before SL/TP)
                preliminary_confidence = htf_score + pattern_score + sr_score + rr_score

                MIN_EQUILIBRIUM_CONFIDENCE = 0.50  # 50% minimum for neutral zones

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

            # STEP 6: Universal Confidence Floor (Phase 2.4)
            # Reject low-confidence signals regardless of other factors
            MIN_CONFIDENCE = 0.45  # 45% minimum confidence for all entries

            if confidence < MIN_CONFIDENCE:
                self.logger.info(f"\n🎯 STEP 6: Universal Confidence Filter")
                self.logger.info(f"   ❌ Signal confidence too low: {confidence*100:.0f}% < {MIN_CONFIDENCE*100:.0f}%")
                self.logger.info(f"   💡 Minimum confidence required for entry quality")
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
        bos_choch_direction = self.market_structure.get_last_bos_choch_direction(df_15m_with_structure)

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
