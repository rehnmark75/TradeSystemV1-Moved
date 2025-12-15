#!/usr/bin/env python3
"""
ICT Silver Bullet Strategy

VERSION: 1.0.0
DATE: 2025-12-15
STATUS: Development

The Silver Bullet is a time-based Smart Money Concepts (SMC) strategy developed
by Michael Huddleston (Inner Circle Trader). It focuses on exploiting liquidity
sweeps and Fair Value Gaps (FVGs) within specific one-hour trading windows.

Entry Requirements:
1. Current time within Silver Bullet window (3-4AM, 10-11AM, or 2-3PM NY)
2. Liquidity sweep detected (BSL for shorts, SSL for longs)
3. Market Structure Shift (MSS) in direction of trade
4. Fair Value Gap (FVG) formed after MSS
5. Price retraces to FVG entry zone

Time Windows (New York Time):
- London Open: 03:00 - 04:00 AM (best for EUR/GBP)
- NY AM Session: 10:00 - 11:00 AM (BEST - London/NY overlap)
- NY PM Session: 02:00 - 03:00 PM (best for USD pairs)

Target Performance:
- Win Rate: 45%+
- Profit Factor: 1.5+
- Average Winner: 20-25 pips
- Target R:R: 2:1 to 3:1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Import helpers
from .helpers.silver_bullet_time_windows import (
    SilverBulletTimeWindows,
    SilverBulletSession
)
from .helpers.silver_bullet_liquidity import (
    SilverBulletLiquidity,
    LiquidityType,
    SweepStatus,
    LiquiditySweep
)
from .helpers.smc_fair_value_gaps import SMCFairValueGaps, FVGType, FVGStatus


class SilverBulletStrategy:
    """
    ICT Silver Bullet Strategy Implementation

    This strategy trades during specific time windows when institutional
    activity creates predictable liquidity patterns.

    Flow:
    1. Check if in valid Silver Bullet time window
    2. Detect liquidity levels (swing highs/lows)
    3. Identify liquidity sweep
    4. Confirm Market Structure Shift
    5. Find FVG for entry
    6. Generate signal with SL/TP
    """

    def __init__(self, config=None, logger: logging.Logger = None):
        """Initialize Silver Bullet Strategy"""
        self.logger = logger or logging.getLogger(__name__)

        # Load configuration
        self._load_config()

        # Initialize helpers - pass config to time_windows for dynamic window sizes
        self.time_windows = SilverBulletTimeWindows(self.logger, config=self._config_module)
        self.liquidity_detector = SilverBulletLiquidity(self.logger)
        self.fvg_detector = SMCFairValueGaps(self.logger)

        # State tracking
        self.pair_cooldowns = {}  # {pair: last_signal_time}
        self.session_signals = {}  # {pair: {session: signal_count}}

        self.logger.info("=" * 60)
        self.logger.info(f"✅ ICT Silver Bullet Strategy v{self.strategy_version} initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"   Time Windows: London Open, NY AM, NY PM")
        self.logger.info(f"   Entry TF: {self.entry_timeframe}")
        self.logger.info(f"   Min R:R: {self.min_rr_ratio}")
        self.logger.info(f"   Min TP: {self.min_tp_pips} pips")
        self.logger.info("=" * 60)

    def _load_config(self):
        """Load configuration from config file"""
        try:
            from ...configdata.strategies import config_silver_bullet as sb_config
        except ImportError:
            try:
                from configdata.strategies import config_silver_bullet as sb_config
            except ImportError:
                from forex_scanner.configdata.strategies import config_silver_bullet as sb_config

        # Store config module reference for time_windows helper
        self._config_module = sb_config

        # Strategy metadata
        self.strategy_version = getattr(sb_config, 'STRATEGY_VERSION', '1.0.0')
        self.strategy_name = getattr(sb_config, 'STRATEGY_NAME', 'SILVER_BULLET')

        # Time windows
        self.enabled_sessions = getattr(sb_config, 'ENABLED_SESSIONS', ['LONDON_OPEN', 'NY_AM', 'NY_PM'])
        self.session_quality = getattr(sb_config, 'SESSION_QUALITY', {
            'NY_AM': 1.00, 'NY_PM': 0.90, 'LONDON_OPEN': 0.85
        })

        # Liquidity detection
        self.liquidity_lookback = getattr(sb_config, 'LIQUIDITY_LOOKBACK_BARS', 20)
        self.swing_strength = getattr(sb_config, 'SWING_STRENGTH_BARS', 3)
        self.sweep_min_pips = getattr(sb_config, 'LIQUIDITY_SWEEP_MIN_PIPS', 3)
        self.sweep_max_pips = getattr(sb_config, 'LIQUIDITY_SWEEP_MAX_PIPS', 15)
        self.require_sweep_rejection = getattr(sb_config, 'REQUIRE_SWEEP_REJECTION', True)
        self.sweep_max_age = getattr(sb_config, 'SWEEP_MAX_AGE_BARS', 10)

        # Market Structure Shift
        self.mss_lookback = getattr(sb_config, 'MSS_LOOKBACK_BARS', 15)
        self.mss_min_break_pips = getattr(sb_config, 'MSS_MIN_BREAK_PIPS', 2)
        self.mss_require_body_close = getattr(sb_config, 'MSS_REQUIRE_BODY_CLOSE', True)

        # FVG parameters
        self.fvg_min_size_pips = getattr(sb_config, 'FVG_MIN_SIZE_PIPS', 2)
        self.fvg_max_age = getattr(sb_config, 'FVG_MAX_AGE_BARS', 10)
        self.fvg_entry_zone_min = getattr(sb_config, 'FVG_ENTRY_ZONE_MIN', 0.0)
        self.fvg_entry_zone_max = getattr(sb_config, 'FVG_ENTRY_ZONE_MAX', 0.5)
        self.fvg_max_fill = getattr(sb_config, 'FVG_MAX_FILL_PERCENTAGE', 0.5)

        # Risk management
        self.min_rr_ratio = getattr(sb_config, 'MIN_RR_RATIO', 2.0)
        self.optimal_rr_ratio = getattr(sb_config, 'OPTIMAL_RR_RATIO', 2.5)
        self.max_rr_ratio = getattr(sb_config, 'MAX_RR_RATIO', 4.0)
        self.min_tp_pips = getattr(sb_config, 'MIN_TP_PIPS', 15)
        self.max_sl_pips = getattr(sb_config, 'MAX_SL_PIPS', 20)
        self.sl_buffer_pips = getattr(sb_config, 'SL_BUFFER_PIPS', 2)
        self.use_atr_stop = getattr(sb_config, 'USE_ATR_STOP', True)
        self.sl_atr_multiplier = getattr(sb_config, 'SL_ATR_MULTIPLIER', 1.5)
        self.atr_period = getattr(sb_config, 'ATR_PERIOD', 14)

        # Confidence
        self.min_confidence = getattr(sb_config, 'MIN_CONFIDENCE_THRESHOLD', 0.55)
        self.confidence_weights = getattr(sb_config, 'CONFIDENCE_WEIGHTS', {
            'session_quality': 0.20,
            'sweep_quality': 0.25,
            'fvg_quality': 0.20,
            'mss_strength': 0.20,
            'htf_alignment': 0.15
        })

        # Timeframes
        self.htf_timeframe = getattr(sb_config, 'HTF_BIAS_TIMEFRAME', '1h')
        self.trigger_timeframe = getattr(sb_config, 'TRIGGER_TIMEFRAME', '15m')
        self.entry_timeframe = getattr(sb_config, 'ENTRY_TIMEFRAME', '5m')

        # Pair settings
        self.pair_min_tp = getattr(sb_config, 'PAIR_MIN_TP_PIPS', {})
        self.pair_max_sl = getattr(sb_config, 'PAIR_MAX_SL_PIPS', {})

        # Signal limits
        self.max_signals_per_day = getattr(sb_config, 'MAX_SIGNALS_PER_PAIR_PER_DAY', 2)
        self.cooldown_hours = getattr(sb_config, 'SIGNAL_COOLDOWN_HOURS', 2)
        self.max_signals_per_session = getattr(sb_config, 'MAX_SIGNALS_PER_SESSION', 1)

        # Debug
        self.debug_logging = getattr(sb_config, 'ENABLE_DEBUG_LOGGING', True)

    def detect_signal(
        self,
        df_entry: pd.DataFrame,
        df_htf: pd.DataFrame,
        epic: str,
        pair: str,
        df_trigger: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Detect Silver Bullet trading signal.

        Args:
            df_entry: Entry timeframe data (5m)
            df_htf: Higher timeframe data for bias (1h)
            epic: IG Markets epic code
            pair: Currency pair name
            df_trigger: Optional trigger timeframe data (15m)

        Returns:
            Signal dict or None if no valid signal
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🔫 ICT SILVER BULLET Strategy v{self.strategy_version}")
        self.logger.info(f"   Pair: {pair} ({epic})")
        self.logger.info(f"{'='*70}")

        # Get pip value for this pair
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        # Get candle timestamp
        candle_timestamp = self._get_timestamp(df_entry)

        try:
            # ================================================================
            # STEP 1: TIME WINDOW CHECK
            # ================================================================
            self.logger.info(f"\n🕐 STEP 1: Checking Silver Bullet Time Window")

            is_valid_time, session, time_reason = self.time_windows.is_in_silver_bullet_window(
                candle_timestamp,
                self.enabled_sessions
            )

            if not is_valid_time:
                self.logger.info(f"   ❌ {time_reason}")
                return None

            self.logger.info(f"   ✅ {time_reason}")

            # Check session signal limit
            if not self._check_session_limit(pair, session):
                self.logger.info(f"   ❌ Session signal limit reached for {pair}")
                return None

            # Check cooldown
            if not self._check_cooldown(pair, candle_timestamp):
                self.logger.info(f"   ❌ Pair in cooldown")
                return None

            # Get session quality
            session_quality = self.time_windows.get_session_quality(session)
            self.logger.info(f"   ✅ Session: {session.value} (quality: {session_quality:.0%})")

            # ================================================================
            # STEP 2: HTF BIAS
            # ================================================================
            self.logger.info(f"\n📊 STEP 2: Getting HTF Bias ({self.htf_timeframe})")

            htf_bias = self._get_htf_bias(df_htf, pip_value)

            if htf_bias is None:
                self.logger.info(f"   ❌ Could not determine HTF bias")
                return None

            self.logger.info(f"   ✅ HTF Bias: {htf_bias['direction']}")
            self.logger.info(f"   ✅ Trend Strength: {htf_bias['strength']:.0%}")

            # ================================================================
            # STEP 3: LIQUIDITY DETECTION
            # ================================================================
            self.logger.info(f"\n💧 STEP 3: Detecting Liquidity Levels")

            # Use trigger timeframe if available, otherwise entry
            liquidity_df = df_trigger if df_trigger is not None and len(df_trigger) > 10 else df_entry

            liquidity_levels = self.liquidity_detector.detect_liquidity_levels(
                df=liquidity_df,
                lookback_bars=self.liquidity_lookback,
                swing_strength=self.swing_strength,
                pip_value=pip_value
            )

            if not liquidity_levels:
                self.logger.info(f"   ❌ No liquidity levels found")
                return None

            self.logger.info(f"   ✅ Found {len(liquidity_levels)} liquidity levels")

            # ================================================================
            # STEP 4: LIQUIDITY SWEEP DETECTION
            # ================================================================
            self.logger.info(f"\n🌊 STEP 4: Checking for Liquidity Sweep")

            sweep = self.liquidity_detector.detect_liquidity_sweep(
                df=liquidity_df,
                liquidity_levels=liquidity_levels,
                min_sweep_pips=self.sweep_min_pips,
                max_sweep_pips=self.sweep_max_pips,
                pip_value=pip_value,
                require_rejection=self.require_sweep_rejection,
                max_sweep_age=self.sweep_max_age
            )

            if not sweep:
                self.logger.info(f"   ❌ No valid liquidity sweep detected")
                return None

            if sweep.status == SweepStatus.BREAKOUT:
                self.logger.info(f"   ❌ Sweep too deep ({sweep.sweep_pips:.1f} pips) - likely breakout")
                return None

            # QUALITY FILTER: Prefer CLEAN sweeps, but allow PARTIAL with conditions
            # PENDING sweeps (< 3 bars old) have very low win rate - reject them
            if sweep.status == SweepStatus.PENDING:
                self.logger.info(f"   ❌ PENDING sweep ({sweep.rejection_candles} bars) - too fresh, waiting for confirmation...")
                return None

            # PARTIAL sweeps: allow if they've had time to mature (>= 8 bars old)
            # OR if they have rejection confirmed
            if sweep.status == SweepStatus.PARTIAL:
                if not sweep.rejection_confirmed and sweep.rejection_candles < 8:
                    self.logger.info(f"   ❌ PARTIAL sweep ({sweep.rejection_candles} bars) - waiting for maturity or rejection")
                    return None
                # Allow mature PARTIAL sweeps (>= 8 bars) even without explicit rejection
                # The sweep itself may have caused the price to pause/reverse slightly
                self.logger.info(f"   ✅ PARTIAL sweep accepted ({sweep.rejection_candles} bars old, rejection: {sweep.rejection_confirmed})")

            self.logger.info(f"   ✅ {sweep.liquidity_level.liquidity_type.value} Sweep: {sweep.sweep_pips:.1f} pips")
            self.logger.info(f"   ✅ Status: {sweep.status.value}")
            self.logger.info(f"   ✅ Rejection: {'Yes' if sweep.rejection_confirmed else 'Pending'}")

            # Determine trade direction based on sweep
            # SSL sweep = bullish (buy), BSL sweep = bearish (sell)
            if sweep.liquidity_level.liquidity_type == LiquidityType.SSL:
                trade_direction = 'BULL'
            else:
                trade_direction = 'BEAR'

            # QUALITY FILTER: Require HTF alignment for high-quality entries
            # Counter-trend entries have significantly lower win rate
            if trade_direction != htf_bias['direction']:
                # Only allow counter-trend if HTF trend is weak (< 60% strength)
                if htf_bias['strength'] >= 0.60:
                    self.logger.info(f"   ❌ Counter-trend entry rejected: {trade_direction} vs HTF {htf_bias['direction']} (strength: {htf_bias['strength']:.0%})")
                    return None
                else:
                    self.logger.info(f"   ⚠️  Weak HTF trend ({htf_bias['strength']:.0%}) - allowing counter-trend entry")

            # ================================================================
            # STEP 5: MARKET STRUCTURE SHIFT (MSS)
            # ================================================================
            self.logger.info(f"\n📈 STEP 5: Checking Market Structure Shift")

            mss_result = self._check_market_structure_shift(
                df=df_entry,
                direction=trade_direction,
                sweep=sweep,
                pip_value=pip_value
            )

            if not mss_result['valid']:
                self.logger.info(f"   ❌ {mss_result['reason']}")
                return None

            self.logger.info(f"   ✅ MSS Confirmed: {trade_direction}")
            self.logger.info(f"   ✅ Break Level: {mss_result['break_level']:.5f}")

            # ================================================================
            # STEP 6: FVG DETECTION
            # ================================================================
            self.logger.info(f"\n🎯 STEP 6: Finding Fair Value Gap Entry")

            fvg_config = {
                'fvg_min_size': self.fvg_min_size_pips,
                'fvg_max_age': self.fvg_max_age,
                'fvg_fill_threshold': self.fvg_max_fill
            }

            df_with_fvg = self.fvg_detector.detect_fair_value_gaps(df_entry, fvg_config)

            # Debug: Log FVGs found
            fvg_count = len(self.fvg_detector.fair_value_gaps) if self.fvg_detector.fair_value_gaps else 0
            self.logger.info(f"   📊 FVGs detected: {fvg_count}")

            # Find valid FVG for entry
            fvg_entry = self._find_fvg_entry(
                df=df_with_fvg,
                direction=trade_direction,
                sweep=sweep,
                mss_result=mss_result,
                pip_value=pip_value
            )

            if not fvg_entry:
                self.logger.info(f"   ❌ No valid FVG entry found")
                return None

            self.logger.info(f"   ✅ FVG Entry: {fvg_entry['entry_price']:.5f}")
            self.logger.info(f"   ✅ FVG Size: {fvg_entry['fvg_size_pips']:.1f} pips")

            # ================================================================
            # STEP 7: CALCULATE SL/TP
            # ================================================================
            self.logger.info(f"\n🛑 STEP 7: Calculating Stop Loss & Take Profit")

            sl_tp = self._calculate_sl_tp(
                df=df_entry,
                direction=trade_direction,
                entry_price=fvg_entry['entry_price'],
                fvg=fvg_entry,
                sweep=sweep,
                pip_value=pip_value,
                pair=pair
            )

            if not sl_tp['valid']:
                self.logger.info(f"   ❌ {sl_tp['reason']}")
                return None

            self.logger.info(f"   ✅ Stop Loss: {sl_tp['stop_loss']:.5f} ({sl_tp['risk_pips']:.1f} pips)")
            self.logger.info(f"   ✅ Take Profit: {sl_tp['take_profit']:.5f} ({sl_tp['reward_pips']:.1f} pips)")
            self.logger.info(f"   ✅ R:R Ratio: {sl_tp['rr_ratio']:.2f}")

            # ================================================================
            # STEP 8: CALCULATE CONFIDENCE
            # ================================================================
            self.logger.info(f"\n📊 STEP 8: Calculating Confidence Score")

            confidence = self._calculate_confidence(
                session=session,
                sweep=sweep,
                fvg_entry=fvg_entry,
                mss_result=mss_result,
                htf_bias=htf_bias,
                trade_direction=trade_direction
            )

            if confidence < self.min_confidence:
                self.logger.info(f"   ❌ Confidence too low: {confidence:.0%} < {self.min_confidence:.0%}")
                return None

            self.logger.info(f"   ✅ Confidence: {confidence:.0%}")

            # ================================================================
            # BUILD SIGNAL
            # ================================================================
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✅ VALID SILVER BULLET SIGNAL DETECTED")
            self.logger.info(f"{'='*70}")

            # Calculate ATR for additional context
            atr = self._calculate_atr(df_entry, pip_value)

            signal = {
                'strategy': 'SILVER_BULLET',
                'signal_type': trade_direction,
                'signal': trade_direction,
                'confidence_score': round(confidence, 2),
                'epic': epic,
                'pair': pair,
                'timeframe': self.entry_timeframe,

                # Prices
                'entry_price': fvg_entry['entry_price'],
                'stop_loss': sl_tp['stop_loss'],
                'take_profit': sl_tp['take_profit'],

                # Risk metrics
                'risk_pips': round(sl_tp['risk_pips'], 1),
                'reward_pips': round(sl_tp['reward_pips'], 1),
                'rr_ratio': round(sl_tp['rr_ratio'], 2),

                # Silver Bullet specific data
                'session': session.value,
                'session_quality': session_quality,
                'sweep_type': sweep.liquidity_level.liquidity_type.value,
                'sweep_pips': round(sweep.sweep_pips, 1),
                'sweep_rejection': sweep.rejection_confirmed,
                'fvg_size_pips': round(fvg_entry['fvg_size_pips'], 1),
                'mss_break_level': mss_result['break_level'],

                # Technical indicators
                'atr': round(atr, 6) if atr else 0.0,
                'htf_bias': htf_bias['direction'],
                'htf_strength': htf_bias['strength'],

                # Timestamp
                'timestamp': candle_timestamp,

                # Metadata for Claude prompt builder
                'metadata': {
                    # Session information
                    'session': session.value,
                    'session_quality': session_quality,

                    # Liquidity sweep data
                    'sweep_type': sweep.liquidity_level.liquidity_type.value,
                    'sweep_status': sweep.status.value,
                    'sweep_pips': round(sweep.sweep_pips, 1),
                    'sweep_age_bars': sweep.rejection_candles,
                    'liquidity_level': sweep.liquidity_level.price,
                    'rejection_confirmed': sweep.rejection_confirmed,

                    # Market Structure Shift data
                    'mss_confirmed': True,
                    'mss_break_pips': mss_result.get('break_pips', 0),
                    'mss_direction': trade_direction,

                    # FVG data
                    'fvg_type': 'BULLISH_FVG' if trade_direction == 'BULL' else 'BEARISH_FVG',
                    'fvg_size_pips': round(fvg_entry['fvg_size_pips'], 1),
                    'fvg_fill_percentage': fvg_entry.get('fvg_fill_pct', 0),
                    'entry_type': fvg_entry.get('entry_type', 'IMMEDIATE'),

                    # HTF alignment
                    'htf_direction': htf_bias['direction'],
                    'htf_strength': htf_bias['strength'],
                    'htf_aligned': htf_bias['direction'] == trade_direction
                },

                # Strategy indicators for compatibility
                'strategy_indicators': {
                    'silver_bullet': {
                        'session': session.value,
                        'session_quality': session_quality,
                        'sweep': {
                            'type': sweep.liquidity_level.liquidity_type.value,
                            'pips': sweep.sweep_pips,
                            'status': sweep.status.value,
                            'rejection': sweep.rejection_confirmed
                        },
                        'mss': {
                            'confirmed': True,
                            'break_level': mss_result['break_level'],
                            'break_pips': mss_result.get('break_pips', 0)
                        },
                        'fvg': {
                            'high': fvg_entry['fvg_high'],
                            'low': fvg_entry['fvg_low'],
                            'size_pips': fvg_entry['fvg_size_pips']
                        }
                    },
                    'htf_bias': htf_bias,
                    'confidence_breakdown': self._get_confidence_breakdown(
                        session, sweep, fvg_entry, mss_result, htf_bias, trade_direction
                    )
                },

                # Description
                'description': self._build_description(
                    trade_direction, session, sweep, fvg_entry, sl_tp
                )
            }

            self.logger.info(f"\n📋 Signal Summary:")
            self.logger.info(f"   Direction: {signal['signal']}")
            self.logger.info(f"   Session: {signal['session']}")
            self.logger.info(f"   Entry: {signal['entry_price']:.5f}")
            self.logger.info(f"   SL: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
            self.logger.info(f"   TP: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
            self.logger.info(f"   R:R: {signal['rr_ratio']:.2f}")
            self.logger.info(f"\n   {signal['description']}")
            self.logger.info(f"{'='*70}\n")

            # Update tracking
            self._update_tracking(pair, session, candle_timestamp)

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error in Silver Bullet signal detection: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_htf_bias(self, df_htf: pd.DataFrame, pip_value: float) -> Optional[Dict]:
        """Get higher timeframe directional bias"""
        try:
            # Minimum 20 bars for EMA calculation (was 50, reduced for backtest compatibility)
            if len(df_htf) < 20:
                self.logger.debug(f"   ⚠️ Insufficient HTF data: {len(df_htf)} bars (need 20)")
                return None

            # Calculate EMAs for bias
            close = df_htf['close']
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()

            current_close = close.iloc[-1]
            current_ema_20 = ema_20.iloc[-1]
            current_ema_50 = ema_50.iloc[-1]

            # Determine direction
            if current_close > current_ema_20 > current_ema_50:
                direction = 'BULL'
                strength = min((current_close - current_ema_50) / (current_ema_20 - current_ema_50 + 0.00001), 1.0)
            elif current_close < current_ema_20 < current_ema_50:
                direction = 'BEAR'
                strength = min((current_ema_50 - current_close) / (current_ema_50 - current_ema_20 + 0.00001), 1.0)
            else:
                # Mixed signals - use recent momentum
                recent_change = (close.iloc[-1] - close.iloc[-5]) / pip_value
                direction = 'BULL' if recent_change > 0 else 'BEAR'
                strength = 0.5

            return {
                'direction': direction,
                'strength': abs(strength),
                'ema_20': current_ema_20,
                'ema_50': current_ema_50,
                'close': current_close
            }

        except Exception as e:
            self.logger.error(f"Error getting HTF bias: {e}")
            return None

    def _check_market_structure_shift(
        self,
        df: pd.DataFrame,
        direction: str,
        sweep: LiquiditySweep,
        pip_value: float
    ) -> Dict:
        """Check for Market Structure Shift after liquidity sweep"""
        try:
            if len(df) < self.mss_lookback:
                return {'valid': False, 'reason': 'Insufficient data for MSS check'}

            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            opens = df['open'].values

            current_idx = len(df) - 1

            # Find swings for MSS detection
            swing_highs = []
            swing_lows = []

            for i in range(2, min(self.mss_lookback, len(df) - 2)):
                idx = current_idx - i

                # Swing high
                if highs[idx] > highs[idx-1] and highs[idx] > highs[idx+1]:
                    swing_highs.append((idx, highs[idx]))

                # Swing low
                if lows[idx] < lows[idx-1] and lows[idx] < lows[idx+1]:
                    swing_lows.append((idx, lows[idx]))

            min_break = self.mss_min_break_pips * pip_value

            if direction == 'BULL':
                # For bullish MSS: need to break above a recent swing high
                if not swing_highs:
                    return {'valid': False, 'reason': 'No swing highs for MSS'}

                # Find most recent swing high that was broken
                for swing_idx, swing_level in sorted(swing_highs, key=lambda x: x[0], reverse=True):
                    # Check if broken after sweep
                    for i in range(swing_idx + 1, current_idx + 1):
                        if self.mss_require_body_close:
                            if closes[i] > swing_level + min_break:
                                return {
                                    'valid': True,
                                    'break_level': swing_level,
                                    'break_pips': (closes[i] - swing_level) / pip_value,
                                    'reason': 'Bullish MSS confirmed'
                                }
                        else:
                            if highs[i] > swing_level + min_break:
                                return {
                                    'valid': True,
                                    'break_level': swing_level,
                                    'break_pips': (highs[i] - swing_level) / pip_value,
                                    'reason': 'Bullish MSS confirmed'
                                }

                return {'valid': False, 'reason': 'No bullish MSS - swing high not broken'}

            else:  # BEAR
                # For bearish MSS: need to break below a recent swing low
                if not swing_lows:
                    return {'valid': False, 'reason': 'No swing lows for MSS'}

                # Find most recent swing low that was broken
                for swing_idx, swing_level in sorted(swing_lows, key=lambda x: x[0], reverse=True):
                    # Check if broken after sweep
                    for i in range(swing_idx + 1, current_idx + 1):
                        if self.mss_require_body_close:
                            if closes[i] < swing_level - min_break:
                                return {
                                    'valid': True,
                                    'break_level': swing_level,
                                    'break_pips': (swing_level - closes[i]) / pip_value,
                                    'reason': 'Bearish MSS confirmed'
                                }
                        else:
                            if lows[i] < swing_level - min_break:
                                return {
                                    'valid': True,
                                    'break_level': swing_level,
                                    'break_pips': (swing_level - lows[i]) / pip_value,
                                    'reason': 'Bearish MSS confirmed'
                                }

                return {'valid': False, 'reason': 'No bearish MSS - swing low not broken'}

        except Exception as e:
            self.logger.error(f"Error checking MSS: {e}")
            return {'valid': False, 'reason': f'MSS check error: {e}'}

    def _find_fvg_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        sweep: LiquiditySweep,
        mss_result: Dict,
        pip_value: float
    ) -> Optional[Dict]:
        """Find a valid FVG for entry"""
        try:
            current_price = df['close'].iloc[-1]
            current_idx = len(df) - 1

            # Get FVGs from detector
            fvgs = self.fvg_detector.fair_value_gaps

            if not fvgs:
                self.logger.debug(f"   No FVGs detected by detector")
                return None

            self.logger.debug(f"   Found {len(fvgs)} FVGs, looking for {direction} entry")

            # Filter FVGs by direction and status
            valid_fvgs = []
            for i, fvg in enumerate(fvgs):
                fvg_info = f"FVG#{i+1}: {fvg.gap_type.value if hasattr(fvg.gap_type, 'value') else fvg.gap_type}, size={fvg.gap_size_pips:.1f}pips"

                # Check direction alignment
                if direction == 'BULL' and fvg.gap_type != FVGType.BULLISH:
                    self.logger.debug(f"   ❌ {fvg_info} - REJECTED: direction mismatch (need BULLISH)")
                    continue
                if direction == 'BEAR' and fvg.gap_type != FVGType.BEARISH:
                    self.logger.debug(f"   ❌ {fvg_info} - REJECTED: direction mismatch (need BEARISH)")
                    continue

                # Check if active or partially filled
                if fvg.status not in [FVGStatus.ACTIVE, FVGStatus.PARTIALLY_FILLED]:
                    age_info = f"age={current_idx - fvg.start_index}bars" if hasattr(fvg, 'start_index') else ""
                    self.logger.debug(f"   ❌ {fvg_info} - REJECTED: status={fvg.status} {age_info} (need ACTIVE/PARTIALLY_FILLED)")
                    continue

                # Check age
                fvg.age_bars = current_idx - fvg.start_index
                if fvg.age_bars > self.fvg_max_age:
                    self.logger.debug(f"   ❌ {fvg_info} - REJECTED: age={fvg.age_bars} bars (max={self.fvg_max_age})")
                    continue

                # Check fill percentage
                if fvg.fill_percentage > self.fvg_max_fill:
                    self.logger.debug(f"   ❌ {fvg_info} - REJECTED: fill={fvg.fill_percentage:.0%} (max={self.fvg_max_fill:.0%})")
                    continue

                # Check minimum size
                if fvg.gap_size_pips < self.fvg_min_size_pips:
                    self.logger.debug(f"   ❌ {fvg_info} - REJECTED: size={fvg.gap_size_pips:.1f} pips (min={self.fvg_min_size_pips})")
                    continue

                self.logger.debug(f"   ✅ {fvg_info} - PASSED all filters")
                valid_fvgs.append(fvg)

            if not valid_fvgs:
                self.logger.debug(f"   No FVGs passed filter criteria (direction, status, age, fill, size)")
                return None

            self.logger.debug(f"   {len(valid_fvgs)} FVGs passed filters")

            # Sort by recency and significance
            valid_fvgs.sort(key=lambda f: (f.significance, -f.age_bars), reverse=True)

            # Check if current price is in any FVG
            self.logger.debug(f"   Current price: {current_price:.5f}, checking FVG zones...")
            for fvg in valid_fvgs:
                self.logger.debug(f"   FVG zone: {fvg.low_price:.5f} - {fvg.high_price:.5f}")
                if fvg.low_price <= current_price <= fvg.high_price:
                    # Price is in FVG - use optimal FVG edge for best R:R
                    # For BULL: enter at FVG low (bottom) for better entry
                    # For BEAR: enter at FVG high (top) for better entry
                    if direction == 'BULL':
                        optimal_entry = fvg.low_price
                    else:  # BEAR
                        optimal_entry = fvg.high_price

                    return {
                        'entry_price': optimal_entry,
                        'fvg_high': fvg.high_price,
                        'fvg_low': fvg.low_price,
                        'fvg_size_pips': fvg.gap_size_pips,
                        'fvg_age': fvg.age_bars,
                        'fvg_significance': fvg.significance,
                        'in_fvg': True
                    }

            # Price not in FVG - check if approaching one
            # NOTE: For approaching FVGs, we set entry at optimal FVG edge for best R:R
            # BULL: Enter at FVG low (bottom) - wait for pullback
            # BEAR: Enter at FVG high (top) - wait for rally
            self.logger.debug(f"   Price not in any FVG, checking if approaching...")
            for fvg in valid_fvgs:
                distance_to_fvg = 0
                if direction == 'BULL':
                    # For bullish, we want FVG below current price (to enter on pullback)
                    if current_price > fvg.high_price:
                        distance_to_fvg = (current_price - fvg.high_price) / pip_value
                        self.logger.debug(f"   BULL: Price above FVG, distance={distance_to_fvg:.1f} pips")
                        if distance_to_fvg <= 35:
                            # Use FVG LOW as entry (optimal for bullish - better R:R)
                            return {
                                'entry_price': fvg.low_price,  # FIXED: was fvg.high_price (worst entry)
                                'fvg_high': fvg.high_price,
                                'fvg_low': fvg.low_price,
                                'fvg_size_pips': fvg.gap_size_pips,
                                'fvg_age': fvg.age_bars,
                                'fvg_significance': fvg.significance,
                                'in_fvg': False,
                                'distance_pips': distance_to_fvg
                            }
                    else:
                        self.logger.debug(f"   BULL: Price below FVG high - wrong for bullish")
                else:  # BEAR
                    # For bearish, we want FVG above current price
                    if current_price < fvg.low_price:
                        distance_to_fvg = (fvg.low_price - current_price) / pip_value
                        self.logger.debug(f"   BEAR: Price below FVG, distance={distance_to_fvg:.1f} pips")
                        if distance_to_fvg <= 35:
                            # Use FVG HIGH as entry (optimal for bearish - better R:R)
                            return {
                                'entry_price': fvg.high_price,  # FIXED: was fvg.low_price (worst entry)
                                'fvg_high': fvg.high_price,
                                'fvg_low': fvg.low_price,
                                'fvg_size_pips': fvg.gap_size_pips,
                                'fvg_age': fvg.age_bars,
                                'fvg_significance': fvg.significance,
                                'in_fvg': False,
                                'distance_pips': distance_to_fvg
                            }
                    else:
                        self.logger.debug(f"   BEAR: Price above FVG low - wrong for bearish")

            self.logger.debug(f"   No valid FVG entry - price not in zone and not approaching within 35 pips")
            return None

        except Exception as e:
            self.logger.error(f"Error finding FVG entry: {e}")
            return None

    def _calculate_sl_tp(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        fvg: Dict,
        sweep: LiquiditySweep,
        pip_value: float,
        pair: str
    ) -> Dict:
        """Calculate stop loss and take profit"""
        try:
            # Get pair-specific settings
            min_tp = self.pair_min_tp.get(pair, self.pair_min_tp.get('DEFAULT', self.min_tp_pips))
            max_sl = self.pair_max_sl.get(pair, self.pair_max_sl.get('DEFAULT', self.max_sl_pips))

            # Calculate ATR-based stop if enabled
            atr_sl = 0
            if self.use_atr_stop:
                atr = self._calculate_atr(df, pip_value)
                if atr > 0:
                    atr_sl = atr * self.sl_atr_multiplier

            # Calculate stop loss
            sl_buffer = self.sl_buffer_pips * pip_value

            if direction == 'BULL':
                # SL below FVG low or sweep low - use the LOWER for safer stop
                fvg_sl = fvg['fvg_low'] - sl_buffer
                sweep_sl = sweep.liquidity_level.price - sl_buffer

                # Use the lower (further) of the two for safer risk management
                stop_loss = min(fvg_sl, sweep_sl)

                # Ensure MINIMUM ATR distance - use max() to widen SL if needed
                if atr_sl > 0:
                    min_sl_price = entry_price - atr_sl
                    stop_loss = min(stop_loss, min_sl_price)  # Take the lower (wider) SL

            else:  # BEAR
                # SL above FVG high or sweep high - use the HIGHER for safer stop
                fvg_sl = fvg['fvg_high'] + sl_buffer
                sweep_sl = sweep.liquidity_level.price + sl_buffer

                # Use the higher (further) of the two for safer risk management
                stop_loss = max(fvg_sl, sweep_sl)

                # Ensure MINIMUM ATR distance - use max() to widen SL if needed
                if atr_sl > 0:
                    max_sl_price = entry_price + atr_sl
                    stop_loss = max(stop_loss, max_sl_price)  # Take the higher (wider) SL

            # Calculate risk
            risk_pips = abs(entry_price - stop_loss) / pip_value

            # Check max SL
            if risk_pips > max_sl:
                return {
                    'valid': False,
                    'reason': f'SL too large ({risk_pips:.1f} pips > {max_sl} max)'
                }

            # Calculate volatility-adjusted minimum TP
            # ATR-based dynamic TP targets better profit in different conditions
            atr_pips = (atr_sl / self.sl_atr_multiplier) / pip_value if atr_sl > 0 else 0
            if atr_pips < 8:  # Low volatility
                dynamic_min_tp = 12  # Lower but achievable target
            elif atr_pips < 15:  # Normal volatility
                dynamic_min_tp = min_tp  # Use pair-specific default
            else:  # High volatility
                dynamic_min_tp = min_tp + 5  # Higher target, more room

            # Calculate take profit
            # Primary: Target opposite liquidity
            target_price, target_pips = self.liquidity_detector.get_sweep_target(sweep, pip_value)

            if target_pips >= dynamic_min_tp:
                take_profit = target_price
                reward_pips = target_pips
            else:
                # Fallback: Use R:R ratio with dynamic min TP
                reward_pips = risk_pips * self.optimal_rr_ratio
                if reward_pips < dynamic_min_tp:
                    reward_pips = dynamic_min_tp

                if direction == 'BULL':
                    take_profit = entry_price + (reward_pips * pip_value)
                else:
                    take_profit = entry_price - (reward_pips * pip_value)

            # Calculate R:R
            rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0

            # Validate R:R
            if rr_ratio < self.min_rr_ratio:
                return {
                    'valid': False,
                    'reason': f'R:R too low ({rr_ratio:.2f} < {self.min_rr_ratio})'
                }

            # Cap R:R if too high
            if rr_ratio > self.max_rr_ratio:
                reward_pips = risk_pips * self.max_rr_ratio
                if direction == 'BULL':
                    take_profit = entry_price + (reward_pips * pip_value)
                else:
                    take_profit = entry_price - (reward_pips * pip_value)
                rr_ratio = self.max_rr_ratio

            return {
                'valid': True,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_pips': risk_pips,
                'reward_pips': reward_pips,
                'rr_ratio': rr_ratio
            }

        except Exception as e:
            self.logger.error(f"Error calculating SL/TP: {e}")
            return {'valid': False, 'reason': f'SL/TP calculation error: {e}'}

    def _calculate_confidence(
        self,
        session: SilverBulletSession,
        sweep: LiquiditySweep,
        fvg_entry: Dict,
        mss_result: Dict,
        htf_bias: Dict,
        trade_direction: str
    ) -> float:
        """Calculate overall confidence score"""
        try:
            weights = self.confidence_weights

            # Session quality (20%)
            session_score = self.time_windows.get_session_quality(session) * weights['session_quality']

            # Sweep quality (25%)
            sweep_score = self.liquidity_detector.calculate_sweep_quality(sweep) * weights['sweep_quality']

            # FVG quality (20%)
            fvg_score = fvg_entry.get('fvg_significance', 0.5)
            if fvg_entry.get('in_fvg', False):
                fvg_score *= 1.1  # Bonus for being in FVG
            fvg_score = min(fvg_score, 1.0) * weights['fvg_quality']

            # MSS strength (20%)
            mss_score = min(mss_result.get('break_pips', 0) / 10, 1.0) * weights['mss_strength']

            # HTF alignment (15%)
            htf_score = htf_bias['strength']
            if htf_bias['direction'] != trade_direction:
                htf_score *= 0.5  # Penalty for going against HTF
            htf_score *= weights['htf_alignment']

            # Total confidence
            confidence = session_score + sweep_score + fvg_score + mss_score + htf_score

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _get_confidence_breakdown(
        self,
        session: SilverBulletSession,
        sweep: LiquiditySweep,
        fvg_entry: Dict,
        mss_result: Dict,
        htf_bias: Dict,
        trade_direction: str
    ) -> Dict:
        """Get detailed confidence breakdown for logging"""
        return {
            'session_quality': self.time_windows.get_session_quality(session),
            'sweep_quality': self.liquidity_detector.calculate_sweep_quality(sweep),
            'fvg_quality': fvg_entry.get('fvg_significance', 0.5),
            'mss_strength': min(mss_result.get('break_pips', 0) / 10, 1.0),
            'htf_alignment': htf_bias['strength'] if htf_bias['direction'] == trade_direction else htf_bias['strength'] * 0.5
        }

    def _calculate_atr(self, df: pd.DataFrame, pip_value: float) -> float:
        """Calculate ATR"""
        try:
            if len(df) < self.atr_period + 1:
                return 0.0

            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])

            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(true_range[-self.atr_period:])

            return atr

        except Exception:
            return 0.0

    def _get_timestamp(self, df: pd.DataFrame) -> datetime:
        """Get timestamp from dataframe"""
        try:
            if 'start_time' in df.columns:
                ts = df['start_time'].iloc[-1]
            elif 'timestamp' in df.columns:
                ts = df['timestamp'].iloc[-1]
            else:
                ts = df.index[-1]

            if isinstance(ts, pd.Timestamp):
                return ts.to_pydatetime()
            return ts

        except Exception:
            return datetime.utcnow()

    def _check_cooldown(self, pair: str, current_time: datetime) -> bool:
        """Check if pair is in cooldown"""
        if pair not in self.pair_cooldowns:
            return True

        last_signal = self.pair_cooldowns[pair]

        # Handle timezone differences
        if hasattr(last_signal, 'tzinfo') and last_signal.tzinfo is not None:
            if hasattr(current_time, 'tzinfo') and current_time.tzinfo is None:
                last_signal = last_signal.replace(tzinfo=None)

        hours_since = (current_time - last_signal).total_seconds() / 3600

        return hours_since >= self.cooldown_hours

    def _check_session_limit(self, pair: str, session: SilverBulletSession) -> bool:
        """Check if session signal limit reached"""
        if pair not in self.session_signals:
            return True

        session_counts = self.session_signals.get(pair, {})
        current_count = session_counts.get(session.value, 0)

        return current_count < self.max_signals_per_session

    def _update_tracking(self, pair: str, session: SilverBulletSession, timestamp: datetime):
        """Update cooldown and session tracking"""
        self.pair_cooldowns[pair] = timestamp

        if pair not in self.session_signals:
            self.session_signals[pair] = {}

        if session.value not in self.session_signals[pair]:
            self.session_signals[pair][session.value] = 0

        self.session_signals[pair][session.value] += 1

    def reset_daily_tracking(self):
        """Reset daily tracking - call at start of new day"""
        self.session_signals = {}
        self.logger.info("🔄 Silver Bullet daily tracking reset")

    def reset_cooldowns(self):
        """Reset cooldowns - call at start of backtest"""
        self.pair_cooldowns = {}
        self.session_signals = {}
        self.logger.info("🔄 Silver Bullet cooldowns reset")

    def _build_description(
        self,
        direction: str,
        session: SilverBulletSession,
        sweep: LiquiditySweep,
        fvg: Dict,
        sl_tp: Dict
    ) -> str:
        """Build human-readable signal description"""
        sweep_type = "SSL" if sweep.liquidity_level.liquidity_type == LiquidityType.SSL else "BSL"

        return (
            f"Silver Bullet {direction} during {session.value} session. "
            f"{sweep_type} sweep ({sweep.sweep_pips:.1f} pips) with FVG entry. "
            f"R:R {sl_tp['rr_ratio']:.1f}:1"
        )
