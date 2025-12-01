#!/usr/bin/env python3
"""
SMC Simple Strategy - 3-Tier EMA-Based Trend Following

VERSION: 1.0.0
DATE: 2025-11-30
STATUS: Production - Initial Release

Strategy Architecture:
    TIER 1: 4H 50 EMA for directional bias (institutional standard)
    TIER 2: 1H swing break with body-close confirmation
    TIER 3: 15m pullback to Fibonacci zone for entry

Key Differences from SMC_STRUCTURE:
    - NO complex HTF strength calculations (uses simple EMA)
    - NO cascading filters (only 3 simple checks)
    - NO order block detection (uses swing structure)
    - SIMPLE volume confirmation (optional)
    - CLEAR entry rules (Fib pullback zones)

Expected Performance:
    - Signals: 15-25 per month (9 pairs)
    - Win Rate: 35-42%
    - Profit Factor: 1.4-1.8
    - Avg Win: 18-25 pips
    - Avg Loss: 10-12 pips
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta


class SMCSimpleStrategy:
    """
    Simplified SMC strategy using 3-tier EMA-based approach

    Entry Requirements:
    1. TIER 1: Price on correct side of 4H 50 EMA (trend bias)
    2. TIER 2: 1H candle BODY closes beyond swing high/low (momentum)
    3. TIER 3: 15m price pulls back to Fibonacci zone (entry timing)
    """

    def __init__(self, config, logger=None):
        """Initialize SMC Simple Strategy"""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Load configuration
        self._load_config()

        # State tracking
        self.pair_cooldowns = {}  # {pair: last_signal_time}
        self.recent_signals = {}  # {pair: [(timestamp, price), ...]}
        self.pending_entries = {}  # {pair: pending_entry_data}

        self.logger.info("=" * 60)
        self.logger.info("✅ SMC Simple Strategy v1.0.0 initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"   TIER 1: {self.ema_period} EMA on {self.htf_timeframe}")
        self.logger.info(f"   TIER 2: Swing break on {self.trigger_tf}")
        self.logger.info(f"   TIER 3: Pullback entry on {self.entry_tf}")
        self.logger.info(f"   Min R:R: {self.min_rr_ratio}")
        self.logger.info(f"   Fib Zone: {self.fib_min*100:.1f}% - {self.fib_max*100:.1f}%")
        self.logger.info("=" * 60)

    def _load_config(self):
        """Load configuration from config object"""
        # Import config module
        try:
            from configdata.strategies import config_smc_simple as smc_config
        except ImportError:
            try:
                from forex_scanner.configdata.strategies import config_smc_simple as smc_config
            except ImportError:
                from ..configdata.strategies import config_smc_simple as smc_config

        # Strategy metadata
        self.strategy_version = getattr(smc_config, 'STRATEGY_VERSION', '1.0.0')
        self.strategy_name = getattr(smc_config, 'STRATEGY_NAME', 'SMC_SIMPLE')

        # TIER 1: HTF Settings
        self.htf_timeframe = getattr(smc_config, 'HTF_TIMEFRAME', '4h')
        self.ema_period = getattr(smc_config, 'EMA_PERIOD', 50)
        self.ema_buffer_pips = getattr(smc_config, 'EMA_BUFFER_PIPS', 5)
        self.require_close_beyond_ema = getattr(smc_config, 'REQUIRE_CLOSE_BEYOND_EMA', True)
        self.min_distance_from_ema = getattr(smc_config, 'MIN_DISTANCE_FROM_EMA_PIPS', 10)

        # TIER 2: Trigger Settings
        self.trigger_tf = getattr(smc_config, 'TRIGGER_TIMEFRAME', '1h')
        self.swing_lookback = getattr(smc_config, 'SWING_LOOKBACK_BARS', 20)
        self.swing_strength = getattr(smc_config, 'SWING_STRENGTH_BARS', 3)
        self.require_body_close = getattr(smc_config, 'REQUIRE_BODY_CLOSE_BREAK', True)
        self.wick_tolerance_pips = getattr(smc_config, 'WICK_TOLERANCE_PIPS', 2)

        # Volume confirmation
        self.volume_enabled = getattr(smc_config, 'VOLUME_CONFIRMATION_ENABLED', True)
        self.volume_sma_period = getattr(smc_config, 'VOLUME_SMA_PERIOD', 20)
        self.volume_multiplier = getattr(smc_config, 'VOLUME_SPIKE_MULTIPLIER', 1.3)

        # TIER 3: Entry Settings
        self.entry_tf = getattr(smc_config, 'ENTRY_TIMEFRAME', '15m')
        self.pullback_enabled = getattr(smc_config, 'PULLBACK_ENABLED', True)
        self.fib_min = getattr(smc_config, 'FIB_PULLBACK_MIN', 0.236)
        self.fib_max = getattr(smc_config, 'FIB_PULLBACK_MAX', 0.500)
        self.fib_optimal = getattr(smc_config, 'FIB_OPTIMAL_ZONE', (0.382, 0.500))
        self.max_pullback_wait = getattr(smc_config, 'MAX_PULLBACK_WAIT_BARS', 12)
        self.pullback_confirm_bars = getattr(smc_config, 'PULLBACK_CONFIRMATION_BARS', 2)

        # Risk Management
        self.min_rr_ratio = getattr(smc_config, 'MIN_RR_RATIO', 1.5)
        self.optimal_rr = getattr(smc_config, 'OPTIMAL_RR_RATIO', 2.0)
        self.max_rr_ratio = getattr(smc_config, 'MAX_RR_RATIO', 4.0)
        self.sl_buffer_pips = getattr(smc_config, 'SL_BUFFER_PIPS', 8)
        self.min_tp_pips = getattr(smc_config, 'MIN_TP_PIPS', 15)
        self.use_swing_target = getattr(smc_config, 'USE_SWING_TARGET', True)

        # Session Filter
        self.session_filter_enabled = getattr(smc_config, 'SESSION_FILTER_ENABLED', True)
        self.block_asian = getattr(smc_config, 'BLOCK_ASIAN_SESSION', True)

        # Signal Limits
        self.max_signals_per_pair = getattr(smc_config, 'MAX_SIGNALS_PER_PAIR_PER_DAY', 1)
        self.cooldown_hours = getattr(smc_config, 'SIGNAL_COOLDOWN_HOURS', 4)

        # Confidence thresholds
        self.min_confidence = getattr(smc_config, 'MIN_CONFIDENCE_THRESHOLD', 0.50)

        # Debug
        self.debug_logging = getattr(smc_config, 'ENABLE_DEBUG_LOGGING', True)

    def detect_signal(
        self,
        df_trigger: pd.DataFrame,
        df_4h: pd.DataFrame,
        epic: str,
        pair: str,
        df_entry: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Detect trading signal using 3-tier approach

        Args:
            df_trigger: Trigger timeframe data (15m or 1h based on config)
            df_4h: 4H timeframe OHLCV data (bias timeframe)
            epic: IG Markets epic code
            pair: Currency pair name
            df_entry: Optional entry timeframe data (5m or 15m based on config)

        Returns:
            Signal dict or None if no valid signal
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🔍 SMC SIMPLE Strategy v{self.strategy_version} - Signal Detection")
        self.logger.info(f"   Pair: {pair} ({epic})")
        self.logger.info(f"   Timeframes: 4H (bias) → {self.trigger_tf} (trigger) → {self.entry_tf} (entry)")
        self.logger.info(f"   df_entry received: {'Yes, ' + str(len(df_entry)) + ' bars' if df_entry is not None else 'NO - None'}")
        self.logger.info(f"{'='*70}")

        # Get candle timestamp for session check
        # Try 'start_time' column first (common in backtest data), then fall back to index
        if 'start_time' in df_trigger.columns:
            candle_timestamp = df_trigger['start_time'].iloc[-1]
        elif 'timestamp' in df_trigger.columns:
            candle_timestamp = df_trigger['timestamp'].iloc[-1]
        else:
            candle_timestamp = df_trigger.index[-1]

        # Get pip value
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        try:
            # ================================================================
            # PRE-FILTER: Session Check
            # ================================================================
            if self.session_filter_enabled:
                session_valid, session_reason = self._check_session(candle_timestamp)
                if not session_valid:
                    self.logger.info(f"\n🕐 SESSION FILTER: {session_reason}")
                    return None
                self.logger.info(f"\n🕐 SESSION: {session_reason}")

            # ================================================================
            # PRE-FILTER: Cooldown Check
            # ================================================================
            # Convert Timestamp to datetime for cooldown comparison (handles backtest properly)
            # Handle various timestamp types: pd.Timestamp, datetime, numpy.int64 (nanoseconds)
            if hasattr(candle_timestamp, 'to_pydatetime'):
                candle_dt = candle_timestamp.to_pydatetime()
            elif isinstance(candle_timestamp, (int, np.integer)):
                # numpy.int64 nanosecond timestamp - convert to datetime
                candle_dt = pd.Timestamp(candle_timestamp).to_pydatetime()
            else:
                candle_dt = candle_timestamp
            cooldown_valid, cooldown_reason = self._check_cooldown(pair, candle_dt)
            if not cooldown_valid:
                self.logger.info(f"\n⏱️  COOLDOWN: {cooldown_reason}")
                return None

            # ================================================================
            # TIER 1: 4H EMA Directional Bias
            # ================================================================
            self.logger.info(f"\n📊 TIER 1: Checking 4H {self.ema_period} EMA Bias")

            ema_result = self._check_ema_bias(df_4h, pip_value)

            if not ema_result['valid']:
                self.logger.info(f"   ❌ {ema_result['reason']}")
                return None

            direction = ema_result['direction']  # 'BULL' or 'BEAR'
            ema_value = ema_result['ema_value']
            ema_distance = ema_result['distance_pips']

            self.logger.info(f"   ✅ Direction: {direction}")
            self.logger.info(f"   ✅ 50 EMA: {ema_value:.5f}")
            self.logger.info(f"   ✅ Distance: {ema_distance:.1f} pips from EMA")

            # ================================================================
            # TIER 2: Swing Break Confirmation (15m or 1H based on config)
            # ================================================================
            self.logger.info(f"\n📈 TIER 2: Checking {self.trigger_tf} Swing Break")

            swing_result = self._check_swing_break(df_trigger, direction, pip_value)

            if not swing_result['valid']:
                self.logger.info(f"   ❌ {swing_result['reason']}")
                return None

            swing_level = swing_result['swing_level']
            break_candle = swing_result['break_candle']
            volume_confirmed = swing_result['volume_confirmed']

            self.logger.info(f"   ✅ Swing {'High' if direction == 'BEAR' else 'Low'} Break: {swing_level:.5f}")
            self.logger.info(f"   ✅ Body Close Confirmed: Yes")
            if self.volume_enabled:
                self.logger.info(f"   {'✅' if volume_confirmed else '⚠️ '} Volume Spike: {'Yes' if volume_confirmed else 'No (optional)'}")

            # ================================================================
            # TIER 3: Pullback Entry Zone (5m or 15m based on config)
            # ================================================================
            self.logger.info(f"\n🎯 TIER 3: Checking {self.entry_tf} Pullback Zone")

            if df_entry is None or len(df_entry) < 10:
                self.logger.info(f"   ⚠️  No {self.entry_tf} data available, using {self.trigger_tf} for entry")
                entry_df = df_trigger
            else:
                entry_df = df_entry

            pullback_result = self._check_pullback_zone(
                entry_df,
                direction,
                swing_level,
                break_candle,
                pip_value
            )

            if not pullback_result['valid']:
                self.logger.info(f"   ❌ {pullback_result['reason']}")
                return None

            entry_price = pullback_result['entry_price']
            pullback_depth = pullback_result['pullback_depth']
            in_optimal_zone = pullback_result['in_optimal_zone']

            self.logger.info(f"   ✅ Pullback Depth: {pullback_depth*100:.1f}%")
            self.logger.info(f"   ✅ Entry Price: {entry_price:.5f}")
            self.logger.info(f"   {'✅' if in_optimal_zone else '⚠️ '} Optimal Zone: {'Yes' if in_optimal_zone else 'No'}")

            # ================================================================
            # STEP 4: Calculate Stop Loss and Take Profit
            # ================================================================
            self.logger.info(f"\n🛑 STEP 4: Calculating SL/TP")

            # Stop loss beyond swing level
            if direction == 'BULL':
                stop_loss = swing_level - (self.sl_buffer_pips * pip_value)
            else:
                stop_loss = swing_level + (self.sl_buffer_pips * pip_value)

            risk_pips = abs(entry_price - stop_loss) / pip_value

            # Find take profit (next swing structure)
            tp_result = self._calculate_take_profit(
                df_trigger, direction, entry_price, risk_pips, pip_value
            )

            take_profit = tp_result['take_profit']
            reward_pips = tp_result['reward_pips']
            rr_ratio = tp_result['rr_ratio']

            self.logger.info(f"   Stop Loss: {stop_loss:.5f} ({risk_pips:.1f} pips)")
            self.logger.info(f"   Take Profit: {take_profit:.5f} ({reward_pips:.1f} pips)")
            self.logger.info(f"   R:R Ratio: {rr_ratio:.2f}")

            # Validate R:R
            if rr_ratio < self.min_rr_ratio:
                self.logger.info(f"   ❌ R:R too low ({rr_ratio:.2f} < {self.min_rr_ratio})")
                return None

            self.logger.info(f"   ✅ R:R meets minimum ({rr_ratio:.2f} >= {self.min_rr_ratio})")

            # Validate minimum TP
            if reward_pips < self.min_tp_pips:
                self.logger.info(f"   ❌ TP too small ({reward_pips:.1f} < {self.min_tp_pips} pips)")
                return None

            self.logger.info(f"   ✅ TP meets minimum ({reward_pips:.1f} >= {self.min_tp_pips} pips)")

            # ================================================================
            # STEP 5: Calculate Confidence Score
            # ================================================================
            confidence = self._calculate_confidence(
                ema_distance=ema_distance,
                volume_confirmed=volume_confirmed,
                in_optimal_zone=in_optimal_zone,
                rr_ratio=rr_ratio,
                pullback_depth=pullback_depth
            )

            if confidence < self.min_confidence:
                self.logger.info(f"\n❌ Confidence too low: {confidence*100:.0f}% < {self.min_confidence*100:.0f}%")
                return None

            self.logger.info(f"\n📊 Confidence: {confidence*100:.0f}%")

            # ================================================================
            # BUILD SIGNAL
            # ================================================================
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✅ VALID SMC SIMPLE SIGNAL DETECTED")
            self.logger.info(f"{'='*70}")

            signal = {
                'strategy': 'SMC_SIMPLE',
                'signal_type': direction,
                'signal': direction,
                'confidence_score': round(confidence, 2),
                'epic': epic,
                'pair': pair,
                'timeframe': '15m',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'partial_tp': None,  # Not using partial TP in simple version
                'partial_percent': None,
                'risk_pips': round(risk_pips, 1),
                'reward_pips': round(reward_pips, 1),
                'rr_ratio': round(rr_ratio, 2),
                'timestamp': candle_dt,  # Use candle timestamp for backtest compatibility

                # Tier-specific data
                'ema_value': ema_value,
                'ema_distance_pips': ema_distance,
                'swing_level': swing_level,
                'pullback_depth': pullback_depth,
                'volume_confirmed': volume_confirmed,
                'in_optimal_zone': in_optimal_zone,

                # Strategy indicators (for alert_history compatibility)
                'strategy_indicators': {
                    'tier1_ema': {
                        'timeframe': self.htf_timeframe,
                        'ema_period': self.ema_period,
                        'ema_value': ema_value,
                        'distance_pips': ema_distance,
                        'direction': direction
                    },
                    'tier2_swing': {
                        'timeframe': self.trigger_tf,
                        'swing_level': swing_level,
                        'body_close_confirmed': True,
                        'volume_confirmed': volume_confirmed
                    },
                    'tier3_entry': {
                        'timeframe': self.entry_tf,
                        'entry_price': entry_price,
                        'pullback_depth': pullback_depth,
                        'fib_zone': f"{self.fib_min*100:.1f}%-{self.fib_max*100:.1f}%",
                        'in_optimal_zone': in_optimal_zone
                    },
                    'risk_management': {
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_pips': risk_pips,
                        'reward_pips': reward_pips,
                        'rr_ratio': rr_ratio
                    },
                    'confidence_breakdown': {
                        'total': confidence,
                        'ema_alignment': min(ema_distance / 30, 1.0) * 0.25,
                        'volume_bonus': 0.15 if volume_confirmed else 0.05,
                        'pullback_quality': (1.0 if in_optimal_zone else 0.6) * 0.25,
                        'rr_quality': min(rr_ratio / 3.0, 1.0) * 0.20,
                        'fib_accuracy': (1.0 - abs(pullback_depth - 0.382) / 0.382) * 0.15
                    },
                    'indicator_count': 4,
                    'data_source': 'smc_simple_3tier'
                },

                # Description
                'description': self._build_description(
                    direction, ema_distance, pullback_depth, rr_ratio
                )
            }

            self.logger.info(f"\n📋 Signal Summary:")
            self.logger.info(f"   Direction: {signal['signal']}")
            self.logger.info(f"   Entry: {signal['entry_price']:.5f}")
            self.logger.info(f"   Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
            self.logger.info(f"   Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
            self.logger.info(f"   R:R Ratio: {signal['rr_ratio']:.2f}")
            self.logger.info(f"\n   {signal['description']}")
            self.logger.info(f"{'='*70}\n")

            # Update cooldown (use candle timestamp for backtest compatibility)
            self._update_cooldown(pair, candle_dt)

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error detecting SMC Simple signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _check_ema_bias(self, df_4h: pd.DataFrame, pip_value: float) -> Dict:
        """
        TIER 1: Check 4H 50 EMA directional bias

        Returns:
            Dict with: valid, direction, ema_value, distance_pips, reason
        """
        if len(df_4h) < self.ema_period + 1:
            return {
                'valid': False,
                'reason': f"Insufficient 4H data ({len(df_4h)} < {self.ema_period + 1} bars)"
            }

        # Calculate 50 EMA
        ema = df_4h['close'].ewm(span=self.ema_period, adjust=False).mean()
        ema_value = ema.iloc[-1]
        current_close = df_4h['close'].iloc[-1]

        # Calculate distance in pips
        distance_pips = abs(current_close - ema_value) / pip_value

        # Check if price is beyond EMA buffer zone
        buffer = self.ema_buffer_pips * pip_value

        if current_close > ema_value + buffer:
            direction = 'BULL'
        elif current_close < ema_value - buffer:
            direction = 'BEAR'
        else:
            return {
                'valid': False,
                'reason': f"Price in EMA buffer zone ({distance_pips:.1f} pips, need >{self.ema_buffer_pips})"
            }

        # Check minimum distance from EMA
        if distance_pips < self.min_distance_from_ema:
            return {
                'valid': False,
                'reason': f"Too close to EMA ({distance_pips:.1f} pips < {self.min_distance_from_ema})"
            }

        # Verify candle CLOSED beyond EMA (not just wicked)
        if self.require_close_beyond_ema:
            if direction == 'BULL' and current_close <= ema_value:
                return {
                    'valid': False,
                    'reason': "Candle did not CLOSE above EMA"
                }
            elif direction == 'BEAR' and current_close >= ema_value:
                return {
                    'valid': False,
                    'reason': "Candle did not CLOSE below EMA"
                }

        return {
            'valid': True,
            'direction': direction,
            'ema_value': ema_value,
            'distance_pips': distance_pips,
            'reason': f"{direction} bias confirmed"
        }

    def _check_swing_break(self, df: pd.DataFrame, direction: str, pip_value: float) -> Dict:
        """
        TIER 2: Check swing break with body-close confirmation on trigger timeframe

        IMPROVED: Now checks for RECENT breaks within lookback, not just current candle
        This allows detecting entries after a break has occurred and price is continuing

        Returns:
            Dict with: valid, swing_level, break_candle, volume_confirmed, reason
        """
        if len(df) < self.swing_lookback + 1:
            return {
                'valid': False,
                'reason': f"Insufficient {self.trigger_tf} data ({len(df)} < {self.swing_lookback + 1} bars)"
            }

        # Find swing points
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        volumes = df['volume'].values if 'volume' in df.columns else None

        swing_highs = []
        swing_lows = []

        # Find swings up to (but not including) the last swing_strength bars
        # so we have time to confirm them
        for i in range(self.swing_strength, len(df) - self.swing_strength - 1):
            # Check for swing high
            is_swing_high = True
            for j in range(1, self.swing_strength + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append((i, highs[i]))

            # Check for swing low
            is_swing_low = True
            for j in range(1, self.swing_strength + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append((i, lows[i]))

        # Check for break based on direction
        current_idx = len(df) - 1
        current_close = closes[-1]
        current_open = opens[-1]

        # v1.3.0: Expanded lookback window - check ANY swing break in full lookback
        # This allows for pullback entries even when price has retraced from break
        break_check_bars = self.swing_lookback  # Use full lookback (20 bars on 15m)

        if direction == 'BULL':
            # For bullish, need ANY swing high to have been broken within lookback
            if not swing_highs:
                return {'valid': False, 'reason': "No swing highs found"}

            # Get swing highs within lookback, sorted by time (oldest first)
            recent_highs = sorted(
                [sh for sh in swing_highs if current_idx - sh[0] <= self.swing_lookback],
                key=lambda x: x[0]
            )
            if not recent_highs:
                return {'valid': False, 'reason': "No recent swing highs in lookback"}

            # v1.3.0: FLEXIBLE SWING BREAK DETECTION
            # Check EACH swing high in lookback - if ANY was broken, we have a valid setup
            # This allows entries on pullbacks after a break, not just immediate continuations
            break_found = False
            best_swing_level = None
            best_swing_idx = None
            best_break_idx = None

            for swing_idx, swing_level in recent_highs:
                # Check all candles AFTER this swing for a break
                for check_idx in range(swing_idx + 1, current_idx + 1):
                    if highs[check_idx] > swing_level:
                        # Found a break of this swing!
                        if not break_found or swing_level > best_swing_level:
                            # Prefer higher swing levels (stronger breaks)
                            break_found = True
                            best_swing_level = swing_level
                            best_swing_idx = swing_idx
                            best_break_idx = check_idx
                        break  # Found break for this swing, check next swing

            if not break_found:
                # Fallback: check if current price is above the highest recent swing
                highest_swing = max(recent_highs, key=lambda x: x[1])
                return {
                    'valid': False,
                    'reason': f"Price did not break swing high {highest_swing[1]:.5f}"
                }

            swing_level = best_swing_level
            swing_idx = best_swing_idx
            break_candle_idx = best_break_idx

            # Use the break candle data for reference
            break_open = opens[break_candle_idx]
            break_close = closes[break_candle_idx]
            break_high = highs[break_candle_idx]
            break_low = lows[break_candle_idx]

        else:  # BEAR
            # For bearish, need ANY swing low to have been broken within lookback
            if not swing_lows:
                return {'valid': False, 'reason': "No swing lows found"}

            # Get swing lows within lookback, sorted by time (oldest first)
            recent_lows = sorted(
                [sl for sl in swing_lows if current_idx - sl[0] <= self.swing_lookback],
                key=lambda x: x[0]
            )
            if not recent_lows:
                return {'valid': False, 'reason': "No recent swing lows in lookback"}

            # v1.3.0: FLEXIBLE SWING BREAK DETECTION
            # Check EACH swing low in lookback - if ANY was broken, we have a valid setup
            break_found = False
            best_swing_level = None
            best_swing_idx = None
            best_break_idx = None

            for swing_idx, swing_level in recent_lows:
                # Check all candles AFTER this swing for a break
                for check_idx in range(swing_idx + 1, current_idx + 1):
                    if lows[check_idx] < swing_level:
                        # Found a break of this swing!
                        if not break_found or swing_level < best_swing_level:
                            # Prefer lower swing levels (stronger breaks)
                            break_found = True
                            best_swing_level = swing_level
                            best_swing_idx = swing_idx
                            best_break_idx = check_idx
                        break  # Found break for this swing, check next swing

            if not break_found:
                # Fallback: check if current price is below the lowest recent swing
                lowest_swing = min(recent_lows, key=lambda x: x[1])
                return {
                    'valid': False,
                    'reason': f"Price did not break swing low {lowest_swing[1]:.5f}"
                }

            swing_level = best_swing_level
            swing_idx = best_swing_idx
            break_candle_idx = best_break_idx

            # Use the break candle data for reference
            break_open = opens[break_candle_idx]
            break_close = closes[break_candle_idx]
            break_high = highs[break_candle_idx]
            break_low = lows[break_candle_idx]

        # Volume confirmation (optional) - check volume on break candle
        volume_confirmed = False
        if self.volume_enabled and volumes is not None:
            vol_sma = np.mean(volumes[max(0, break_candle_idx-self.volume_sma_period):break_candle_idx])
            break_vol = volumes[break_candle_idx]
            if vol_sma > 0:
                volume_confirmed = break_vol > vol_sma * self.volume_multiplier

        return {
            'valid': True,
            'swing_level': swing_level,
            'break_candle': {
                'open': break_open,
                'close': break_close,
                'high': break_high,
                'low': break_low,
                'bars_ago': current_idx - break_candle_idx  # NEW: How many bars ago break occurred
            },
            'volume_confirmed': volume_confirmed,
            'reason': f"Swing break confirmed ({current_idx - break_candle_idx} bars ago)"
        }

    def _check_pullback_zone(
        self,
        df: pd.DataFrame,
        direction: str,
        swing_level: float,
        break_candle: Dict,
        pip_value: float
    ) -> Dict:
        """
        TIER 3: Check if price has pulled back to Fibonacci zone

        Returns:
            Dict with: valid, entry_price, pullback_depth, in_optimal_zone, reason
        """
        current_close = df['close'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_high = df['high'].iloc[-1]

        # Calculate pullback range
        # For BULL: swing_level is the broken high, measure from low before break
        # For BEAR: swing_level is the broken low, measure from high before break

        if direction == 'BULL':
            # Pullback is measured from the break point to the swing low
            break_extreme = break_candle['high']  # Highest point of break candle

            # Find recent low before the break
            lookback_lows = df['low'].iloc[-self.max_pullback_wait:]
            recent_low = lookback_lows.min()

            # Calculate pullback depth (0 = at extreme, 1 = at swing)
            if break_extreme == recent_low:
                pullback_depth = 0.0
            else:
                pullback_depth = (break_extreme - current_close) / (break_extreme - recent_low)

            entry_price = current_close

            # Check if in Fib zone
            if pullback_depth < self.fib_min:
                return {
                    'valid': False,
                    'reason': f"Insufficient pullback ({pullback_depth*100:.1f}% < {self.fib_min*100:.1f}%)"
                }
            if pullback_depth > self.fib_max:
                return {
                    'valid': False,
                    'reason': f"Pullback too deep ({pullback_depth*100:.1f}% > {self.fib_max*100:.1f}%)"
                }

        else:  # BEAR
            break_extreme = break_candle['low']  # Lowest point of break candle

            # Find recent high before the break
            lookback_highs = df['high'].iloc[-self.max_pullback_wait:]
            recent_high = lookback_highs.max()

            # Calculate pullback depth
            if recent_high == break_extreme:
                pullback_depth = 0.0
            else:
                pullback_depth = (current_close - break_extreme) / (recent_high - break_extreme)

            entry_price = current_close

            # Check if in Fib zone
            if pullback_depth < self.fib_min:
                return {
                    'valid': False,
                    'reason': f"Insufficient pullback ({pullback_depth*100:.1f}% < {self.fib_min*100:.1f}%)"
                }
            if pullback_depth > self.fib_max:
                return {
                    'valid': False,
                    'reason': f"Pullback too deep ({pullback_depth*100:.1f}% > {self.fib_max*100:.1f}%)"
                }

        # Check if in optimal zone
        in_optimal = self.fib_optimal[0] <= pullback_depth <= self.fib_optimal[1]

        return {
            'valid': True,
            'entry_price': entry_price,
            'pullback_depth': pullback_depth,
            'in_optimal_zone': in_optimal,
            'reason': f"In Fib zone ({pullback_depth*100:.1f}%)"
        }

    def _calculate_take_profit(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        risk_pips: float,
        pip_value: float
    ) -> Dict:
        """Calculate take profit based on next swing structure

        IMPROVED v1.0.8:
        - Ensures minimum TP distance based on risk
        - Uses R:R-based fallback when swing targets are too close
        - Properly filters out swing levels that are too close to entry
        """

        # Find next swing level in trade direction
        highs = df['high'].values
        lows = df['low'].values

        # Calculate minimum TP distance (at least min_rr_ratio * risk)
        min_tp_distance = risk_pips * self.min_rr_ratio * pip_value

        take_profit = None

        if self.use_swing_target:
            if direction == 'BULL':
                # Find next resistance (recent swing high above entry)
                # Must be at least min_tp_distance away from entry
                swing_targets = []
                for i in range(self.swing_strength, len(df) - self.swing_strength):
                    is_swing_high = True
                    for j in range(1, self.swing_strength + 1):
                        if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                            is_swing_high = False
                            break
                    if is_swing_high:
                        distance = highs[i] - entry_price
                        # Only include if above entry AND far enough away
                        if distance >= min_tp_distance:
                            swing_targets.append(highs[i])

                if swing_targets:
                    take_profit = min(swing_targets)  # Nearest valid target

            else:  # BEAR
                # Find next support (recent swing low below entry)
                # Must be at least min_tp_distance away from entry
                swing_targets = []
                for i in range(self.swing_strength, len(df) - self.swing_strength):
                    is_swing_low = True
                    for j in range(1, self.swing_strength + 1):
                        if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                            is_swing_low = False
                            break
                    if is_swing_low:
                        distance = entry_price - lows[i]
                        # Only include if below entry AND far enough away
                        if distance >= min_tp_distance:
                            swing_targets.append(lows[i])

                if swing_targets:
                    take_profit = max(swing_targets)  # Nearest valid target

        # Fallback to R:R-based TP if no valid swing target found
        if take_profit is None:
            if direction == 'BULL':
                take_profit = entry_price + (risk_pips * self.optimal_rr * pip_value)
            else:
                take_profit = entry_price - (risk_pips * self.optimal_rr * pip_value)

        reward_pips = abs(take_profit - entry_price) / pip_value
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0

        # Cap R:R if too high (likely unrealistic)
        if rr_ratio > self.max_rr_ratio:
            reward_pips = risk_pips * self.max_rr_ratio
            if direction == 'BULL':
                take_profit = entry_price + (reward_pips * pip_value)
            else:
                take_profit = entry_price - (reward_pips * pip_value)
            rr_ratio = self.max_rr_ratio

        return {
            'take_profit': take_profit,
            'reward_pips': reward_pips,
            'rr_ratio': rr_ratio
        }

    def _calculate_confidence(
        self,
        ema_distance: float,
        volume_confirmed: bool,
        in_optimal_zone: bool,
        rr_ratio: float,
        pullback_depth: float
    ) -> float:
        """
        Calculate confidence score (0.0 to 1.0)

        Scoring:
        - EMA alignment strength: 25%
        - Volume confirmation: 15%
        - Pullback quality: 25%
        - R:R ratio quality: 20%
        - Fib accuracy: 15%
        """
        # EMA alignment (further from EMA = stronger trend)
        # 30+ pips = perfect, scale down from there
        ema_score = min(ema_distance / 30, 1.0) * 0.25

        # Volume confirmation bonus
        volume_score = 0.15 if volume_confirmed else 0.05

        # Pullback quality (in optimal zone = best)
        if in_optimal_zone:
            pullback_score = 1.0 * 0.25
        elif self.fib_min <= pullback_depth <= self.fib_max:
            pullback_score = 0.6 * 0.25
        else:
            pullback_score = 0.3 * 0.25

        # R:R quality (higher is better, cap at 3:1)
        rr_score = min(rr_ratio / 3.0, 1.0) * 0.20

        # Fib accuracy (how close to 0.382 golden zone)
        fib_accuracy = 1.0 - min(abs(pullback_depth - 0.382) / 0.382, 1.0)
        fib_score = fib_accuracy * 0.15

        confidence = ema_score + volume_score + pullback_score + rr_score + fib_score

        return min(confidence, 1.0)

    def _check_session(self, timestamp) -> Tuple[bool, str]:
        """Check if current time is in allowed trading session"""
        if isinstance(timestamp, pd.Timestamp):
            hour = timestamp.hour
        else:
            hour = datetime.now().hour

        # Asian session: 21:00-07:00 UTC
        if self.block_asian and (hour >= 21 or hour < 7):
            return False, f"Asian session blocked (hour={hour})"

        # London: 07:00-16:00, NY: 12:00-21:00
        if 7 <= hour <= 21:
            if 12 <= hour <= 16:
                return True, f"London/NY overlap (hour={hour})"
            elif 7 <= hour < 12:
                return True, f"London session (hour={hour})"
            else:
                return True, f"New York session (hour={hour})"
        else:
            return False, f"Outside trading hours (hour={hour})"

    def _check_cooldown(self, pair: str, current_time: datetime = None) -> Tuple[bool, str]:
        """Check if pair is in cooldown period

        Args:
            pair: Currency pair name
            current_time: Current candle timestamp (for backtest mode). If None, uses datetime.now()
        """
        if pair not in self.pair_cooldowns:
            return True, "No cooldown"

        last_signal = self.pair_cooldowns[pair]
        # Use provided time for backtest, or real time for live trading
        check_time = current_time if current_time is not None else datetime.now()

        # Handle timezone-aware vs naive datetime comparison
        if hasattr(last_signal, 'tzinfo') and last_signal.tzinfo is not None:
            if hasattr(check_time, 'tzinfo') and check_time.tzinfo is None:
                # last_signal is tz-aware, check_time is naive - make check_time naive too
                last_signal = last_signal.replace(tzinfo=None)
        elif hasattr(check_time, 'tzinfo') and check_time.tzinfo is not None:
            # check_time is tz-aware, last_signal is naive - make check_time naive
            check_time = check_time.replace(tzinfo=None)

        hours_since = (check_time - last_signal).total_seconds() / 3600

        if hours_since < self.cooldown_hours:
            return False, f"In cooldown ({hours_since:.1f}h < {self.cooldown_hours}h)"

        return True, f"Cooldown expired ({hours_since:.1f}h)"

    def _update_cooldown(self, pair: str, signal_time: datetime = None):
        """Update cooldown after signal generation

        Args:
            pair: Currency pair name
            signal_time: Signal timestamp (for backtest mode). If None, uses datetime.now()
        """
        self.pair_cooldowns[pair] = signal_time if signal_time is not None else datetime.now()

    def reset_cooldowns(self):
        """Reset all cooldowns - call this at the start of each backtest to ensure fresh state"""
        self.pair_cooldowns = {}
        if self.logger:
            self.logger.info("🔄 SMC Simple strategy cooldowns reset for new backtest")

    def _build_description(
        self,
        direction: str,
        ema_distance: float,
        pullback_depth: float,
        rr_ratio: float
    ) -> str:
        """Build human-readable signal description"""
        parts = []

        # Direction and EMA
        parts.append(f"{direction} trend")
        parts.append(f"{ema_distance:.0f} pips from 50 EMA")

        # Pullback
        parts.append(f"{pullback_depth*100:.0f}% pullback")

        # R:R
        parts.append(f"{rr_ratio:.1f}R")

        return ", ".join(parts).capitalize()


def create_smc_simple_strategy(config, logger=None):
    """Factory function to create SMC Simple Strategy instance"""
    return SMCSimpleStrategy(config, logger)
