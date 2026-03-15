#!/usr/bin/env python3
"""
FVG Retest Strategy - Dual-Mode BOS + FVG Entry System

VERSION: 1.0.0
DATE: 2026-03-15

Dual-mode strategy running alongside SMC_SIMPLE:
  - Type A (Deep Value / The Tap): BOS → FVG created → wait for price to retrace into FVG
  - Type B (Institutional Initiation / The Runaway): High-velocity BOS → enter immediately

Flow: 1H macro (200 EMA) → 5m BOS classification → 5m entry (immediate or deferred)
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    from .strategy_registry import register_strategy, StrategyInterface
    from .helpers.smc_fair_value_gaps import SMCFairValueGaps, FairValueGap, FVGType, FVGStatus
except ImportError:
    from forex_scanner.core.strategies.strategy_registry import register_strategy, StrategyInterface
    from forex_scanner.core.strategies.helpers.smc_fair_value_gaps import SMCFairValueGaps, FairValueGap, FVGType, FVGStatus

try:
    from services.fvg_retest_config_service import get_fvg_retest_config
except ImportError:
    from forex_scanner.services.fvg_retest_config_service import get_fvg_retest_config

try:
    from config import PAIR_INFO
except ImportError:
    try:
        from forex_scanner.config import PAIR_INFO
    except ImportError:
        PAIR_INFO = {}


@dataclass
class PendingFVGSetup:
    """Tracks a Type A pending setup waiting for FVG tap"""
    epic: str
    direction: str  # 'BULL' or 'BEAR'
    fvg: FairValueGap
    swing_level: float  # Broken swing level (invalidation reference)
    bos_time: datetime
    expiry_time: datetime
    htf_ema_value: float
    confidence_base: float  # Pre-calculated base confidence from BOS quality


@register_strategy('FVG_RETEST')
class FVGRetestStrategy(StrategyInterface):
    """
    Dual-mode FVG Retest strategy.

    Runs on 1H (macro) + 5m (trigger/entry) timeframes.
    """

    def __init__(self, config=None, db_manager=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._config = None
        self._data_fetcher = None

        # FVG detector instance
        self._fvg_detector = SMCFairValueGaps(logger=self.logger)

        # State: pending Type A setups per pair
        self._pending_setups: Dict[str, List[PendingFVGSetup]] = {}

        # Persistent major swing levels per pair (Pine Script: lastMajorPH/lastMajorPL)
        self._major_highs: Dict[str, float] = {}
        self._major_lows: Dict[str, float] = {}

        # BOS re-detection guard: track last break_idx per pair to avoid re-firing
        self._last_bos_break_idx: Dict[str, int] = {}

        # Cooldown tracking per pair
        self._cooldowns: Dict[str, datetime] = {}

        self.logger.info("🔷 FVG Retest strategy initialized (dual-mode: Deep Value + Initiation)")

    @property
    def strategy_name(self) -> str:
        return 'FVG_RETEST'

    def get_required_timeframes(self) -> List[str]:
        return ['1h', '5m']

    def reset_cooldowns(self) -> None:
        """Reset cooldowns and pending setups (for backtesting)"""
        self._cooldowns.clear()
        self._pending_setups.clear()
        self._major_highs.clear()
        self._major_lows.clear()
        self._last_bos_break_idx.clear()

    def flush_rejections(self) -> None:
        pass

    def _get_config(self):
        """Lazy-load config from database"""
        if self._config is None:
            try:
                self._config = get_fvg_retest_config()
            except Exception as e:
                self.logger.warning(f"FVG Retest config load failed, using defaults: {e}")
                from forex_scanner.services.fvg_retest_config_service import FVGRetestConfig
                self._config = FVGRetestConfig()
        return self._config

    def _get_current_time(self) -> datetime:
        """Get current time, respecting backtest mode"""
        if self._data_fetcher and hasattr(self._data_fetcher, 'current_backtest_time'):
            bt_time = self._data_fetcher.current_backtest_time
            if bt_time is not None:
                if isinstance(bt_time, pd.Timestamp):
                    return bt_time.to_pydatetime()
                return bt_time
        return datetime.utcnow()

    def _get_pip_value(self, epic: str) -> float:
        """Get pip value for pair"""
        info = PAIR_INFO.get(epic, {})
        multiplier = info.get('pip_multiplier', 10000)
        return 1.0 / multiplier

    def detect_signal(
        self,
        df_trigger: pd.DataFrame,
        df_4h: pd.DataFrame,
        epic: str,
        pair: str,
        df_entry: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Main signal detection entry point.

        Args:
            df_trigger: 5m timeframe data (BOS detection + entry)
            df_4h: 1H timeframe data (macro bias with 200 EMA)
            epic: IG Markets epic code
            pair: Currency pair name
            df_entry: Unused (kept for interface compatibility)
        """
        config = self._get_config()
        if not config.enabled:
            return None

        if not config.is_pair_enabled(epic):
            return None

        pip_value = self._get_pip_value(epic)
        now = self._get_current_time()

        # Check cooldown
        if epic in self._cooldowns:
            cooldown_end = self._cooldowns[epic]
            if now < cooldown_end:
                return None

        # Validate data
        if df_4h is None or len(df_4h) < config.htf_ema_period + 10:
            self.logger.debug(f"[FVG_RETEST] {pair}: Insufficient 1H data ({len(df_4h) if df_4h is not None else 0} bars, need {config.htf_ema_period + 10})")
            return None

        if df_trigger is None or len(df_trigger) < 30:
            self.logger.debug(f"[FVG_RETEST] {pair}: Insufficient 5m data ({len(df_trigger) if df_trigger is not None else 0} bars)")
            return None

        # Expire stale pending setups
        self._expire_setups(epic, now, df_trigger, pip_value, config)

        # =====================================================================
        # TIER 1: HTF Macro Filter (1H 200 EMA + candle direction)
        # =====================================================================
        htf_result = self._check_htf_bias(df_4h, config)
        if htf_result is None:
            self.logger.debug(f"[FVG_RETEST] {pair}: HTF filter failed")
            return None

        htf_direction = htf_result['direction']
        htf_ema_value = htf_result['ema_value']
        htf_aligned = htf_result.get('htf_aligned', True)

        # =====================================================================
        # TIER 3 (checked before TIER 2): Check pending Type A setups for tap
        # =====================================================================
        tap_signal = self._check_pending_taps(
            epic, pair, df_trigger, htf_direction, pip_value, config, now, htf_aligned
        )
        if tap_signal:
            self._set_cooldown(epic, config, now)
            return tap_signal

        # =====================================================================
        # TIER 2: Scan for new BOS on 5m
        # =====================================================================
        bos_result = self._detect_swing_break(df_trigger, htf_direction, pip_value, config, epic)
        if not bos_result or not bos_result.get('valid'):
            return None

        # BOS re-detection guard: skip if we already processed this exact break
        break_idx = bos_result['break_candle']['idx']
        last_idx = self._last_bos_break_idx.get(epic, -1)
        if break_idx == last_idx:
            return None
        self._last_bos_break_idx[epic] = break_idx

        swing_level = bos_result['swing_level']
        break_candle = bos_result['break_candle']

        # Classify the BOS
        classification = self._classify_bos(
            df_trigger, bos_result, pip_value, config
        )

        if classification['type'] == 'B' and config.initiation_enabled:
            # Type B: Institutional Initiation - enter immediately
            signal = self._build_type_b_signal(
                epic, pair, df_trigger, htf_direction, swing_level,
                break_candle, classification, htf_ema_value, pip_value, config, now, htf_aligned
            )
            if signal:
                self._set_cooldown(epic, config, now)
                return signal

        # Type A: Deep Value - find FVGs and store as pending
        self._create_pending_setup(
            epic, pair, df_trigger, htf_direction, swing_level,
            bos_result, htf_ema_value, pip_value, config, now
        )

        return None

    # =========================================================================
    # TIER 1: HTF BIAS
    # =========================================================================

    def _check_htf_bias(self, df_1h: pd.DataFrame, config) -> Optional[Dict]:
        """Check 1H 200 EMA position and candle direction"""
        ema_col = f'ema_{config.htf_ema_period}'

        # Calculate EMA if not present
        if ema_col not in df_1h.columns:
            df_1h = df_1h.copy()
            df_1h[ema_col] = df_1h['close'].ewm(span=config.htf_ema_period, adjust=False).mean()

        if df_1h[ema_col].isna().all():
            return None

        last_closed = df_1h.iloc[-2] if len(df_1h) > 1 else df_1h.iloc[-1]
        ema_value = last_closed[ema_col]
        close = last_closed['close']
        candle_open = last_closed['open']

        if pd.isna(ema_value):
            return None

        # Determine direction: EMA position is the gate, candle direction is a modifier
        # Pine Script: macroBullish = close > ema4H and c4H > o4H
        # But requiring both eliminates 60-70% of windows (pullback candles in trends)
        # Fix: Use EMA position only as gate, candle alignment stored for confidence scoring
        price_above_ema = close > ema_value
        candle_bullish = close > candle_open

        if price_above_ema:
            return {
                'direction': 'BULL',
                'ema_value': ema_value,
                'htf_aligned': candle_bullish,  # True = strong, False = counter-candle
            }
        else:
            return {
                'direction': 'BEAR',
                'ema_value': ema_value,
                'htf_aligned': not candle_bullish,
            }

    # =========================================================================
    # TIER 2: SWING BREAK DETECTION (adapted from SMC_SIMPLE)
    # =========================================================================

    def _detect_swing_break(self, df: pd.DataFrame, direction: str, pip_value: float, config, epic: str = '') -> Optional[Dict]:
        """
        Detect BOS (Break of Structure) on 5m timeframe.

        Cherry-picked from Pine Script "Institutional SMC v9":
        1. Major pivot detection (strength=5 both sides) - only breaks significant structure
        2. Close-based breaks (ta.crossover/crossunder on close) - no false wicks
        3. Volume gating on ALL entries (volume > SMA*1.1) - institutional participation
        4. Persistent major levels per pair - only breaks the latest major swing
        """
        lookback = config.swing_lookback_bars
        strength = config.swing_strength_bars

        if len(df) < lookback + 1:
            return None

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values

        # Volume data
        if 'ltv' in df.columns:
            volumes = df['ltv'].values
        elif 'volume' in df.columns:
            volumes = df['volume'].values
        else:
            volumes = None

        # Find MAJOR swing points (Pine: pivotLen=5, requires N bars on each side)
        swing_highs = []
        swing_lows = []

        for i in range(strength, len(df) - strength):
            is_high = all(highs[i] >= highs[i - j] for j in range(1, strength + 1))
            if i + strength < len(df):
                is_high = is_high and all(highs[i] >= highs[i + j] for j in range(1, strength + 1))
            if is_high:
                swing_highs.append((i, highs[i]))

            is_low = all(lows[i] <= lows[i - j] for j in range(1, strength + 1))
            if i + strength < len(df):
                is_low = is_low and all(lows[i] <= lows[i + j] for j in range(1, strength + 1))
            if is_low:
                swing_lows.append((i, lows[i]))

        # Update persistent major levels (Pine: lastMajorPH / lastMajorPL)
        if swing_highs:
            latest_sh = max(swing_highs, key=lambda x: x[0])
            self._major_highs[epic] = latest_sh[1]
        if swing_lows:
            latest_sl = max(swing_lows, key=lambda x: x[0])
            self._major_lows[epic] = latest_sl[1]

        current_idx = len(df) - 1

        # Volume gating moved to _classify_bos() for Type B only.
        # Type A (FVG tap) doesn't require institutional volume at BOS detection —
        # it requires quality at the FVG entry point instead.

        if direction == 'BULL':
            # Pine: bullBreak = ta.crossover(close, lastMajorPH)
            major_level = self._major_highs.get(epic)
            if major_level is None:
                return None

            # CLOSE-based break detection (Pine: crossover on close, not wick)
            # Look for a recent candle whose CLOSE crossed above the major high
            break_idx = None
            for i in range(max(current_idx - 10, 0), current_idx + 1):
                if closes[i] > major_level and (i == 0 or closes[i - 1] <= major_level):
                    break_idx = i

            if break_idx is None:
                return None

            swing_level = major_level
            swing_idx = max((sh[0] for sh in swing_highs if sh[1] == major_level), default=break_idx - 5)

            # Find opposite swing (swing low before the broken high)
            prior_lows = [sl for sl in swing_lows if sl[0] < swing_idx]
            if prior_lows:
                opposite_swing = max(prior_lows, key=lambda x: x[0])[1]
            else:
                opposite_swing = min(lows[max(0, swing_idx - lookback):max(1, swing_idx)])

        else:  # BEAR
            # Pine: bearBreak = ta.crossunder(close, lastMajorPL)
            major_level = self._major_lows.get(epic)
            if major_level is None:
                return None

            # CLOSE-based break detection
            break_idx = None
            for i in range(max(current_idx - 10, 0), current_idx + 1):
                if closes[i] < major_level and (i == 0 or closes[i - 1] >= major_level):
                    break_idx = i

            if break_idx is None:
                return None

            swing_level = major_level
            swing_idx = max((sl[0] for sl in swing_lows if sl[1] == major_level), default=break_idx - 5)

            prior_highs = [sh for sh in swing_highs if sh[0] < swing_idx]
            if prior_highs:
                opposite_swing = max(prior_highs, key=lambda x: x[0])[1]
            else:
                opposite_swing = max(highs[max(0, swing_idx - lookback):max(1, swing_idx)])

        return {
            'valid': True,
            'swing_level': swing_level,
            'swing_idx': swing_idx,
            'opposite_swing': opposite_swing,
            'break_idx': break_idx,
            'break_candle': {
                'open': opens[break_idx],
                'close': closes[break_idx],
                'high': highs[break_idx],
                'low': lows[break_idx],
                'idx': break_idx,
                'bars_ago': current_idx - break_idx,
            },
            'volumes': volumes,
        }

    # =========================================================================
    # BOS CLASSIFICATION
    # =========================================================================

    def _classify_bos(self, df: pd.DataFrame, bos_result: Dict, pip_value: float, config) -> Dict:
        """
        Classify BOS as Type A or Type B.

        Type B requires ALL three:
        1. Displacement: break candle body > displacement_atr_multiplier * ATR
        2. Follow-through: next N candles close in BOS direction
        3. Volume: break candle volume > volume SMA
        """
        break_candle = bos_result['break_candle']
        break_idx = break_candle['idx']
        volumes = bos_result.get('volumes')
        current_idx = len(df) - 1

        # Calculate ATR
        atr = self._calculate_atr(df, config.atr_period)
        if atr <= 0:
            return {'type': 'A', 'displacement_ratio': 0, 'follow_through': 0, 'volume_spike': False}

        # 1. Displacement check
        body_size = abs(break_candle['close'] - break_candle['open'])
        displacement_ratio = body_size / atr
        displacement_met = displacement_ratio >= config.displacement_atr_multiplier

        # 2. Follow-through check (use available bars only, don't require future bars)
        required_ft = config.follow_through_candles
        ft_count = 0
        direction = 'BULL' if break_candle['close'] > break_candle['open'] else 'BEAR'

        closes = df['close'].values
        opens = df['open'].values

        available_ft_bars = current_idx - break_idx
        effective_required = min(required_ft, available_ft_bars)

        if effective_required > 0:
            for i in range(1, effective_required + 1):
                ft_idx = break_idx + i
                if ft_idx > current_idx:
                    break
                if direction == 'BULL' and closes[ft_idx] > opens[ft_idx]:
                    ft_count += 1
                elif direction == 'BEAR' and closes[ft_idx] < opens[ft_idx]:
                    ft_count += 1

        follow_through_met = effective_required > 0 and ft_count >= effective_required

        # 3. Volume check
        volume_met = False
        if volumes is not None:
            vol_sma_end = max(0, break_idx)
            vol_sma_start = max(0, vol_sma_end - config.volume_sma_period)
            if vol_sma_end > vol_sma_start:
                vol_sma = np.mean(volumes[vol_sma_start:vol_sma_end])
                if vol_sma > 0:
                    volume_met = volumes[break_idx] > vol_sma * config.volume_threshold_multiplier

        is_type_b = displacement_met and follow_through_met and volume_met

        return {
            'type': 'B' if is_type_b else 'A',
            'displacement_ratio': displacement_ratio,
            'displacement_met': displacement_met,
            'follow_through': ft_count,
            'follow_through_met': follow_through_met,
            'volume_spike': volume_met,
            'atr': atr,
        }

    # =========================================================================
    # TYPE A: PENDING SETUP MANAGEMENT
    # =========================================================================

    def _create_pending_setup(
        self, epic, pair, df_trigger, direction, swing_level,
        bos_result, htf_ema_value, pip_value, config, now
    ):
        """
        Detect FVGs near BOS and store as pending setups.

        Cherry-pick from Pine Script: FVG must form CONCURRENTLY with the BOS
        (within a few bars of the break candle), not anywhere in the lookback.
        Pine checks: bullBreak AND isBullFVG on the SAME bar.
        We allow a small window (±3 bars from break) to account for our scan interval.
        """
        break_idx = bos_result.get('break_idx', len(df_trigger) - 1)

        # Detect FVGs on trigger data
        fvg_config = {
            'fvg_min_size': config.fvg_min_size_pips,
            'fvg_max_age': config.fvg_max_age_bars,
            'pip_value': pip_value,
            'pair': pair,
        }
        self._fvg_detector.detect_fair_value_gaps(df_trigger, fvg_config)

        # Filter FVGs: direction match, active, not already inside, AND near BOS
        current_price = df_trigger.iloc[-1]['close']
        target_type = FVGType.BULLISH if direction == 'BULL' else FVGType.BEARISH

        # FVG bar proximity window (Pine: same bar; we allow ±6 bars = 30min on 5m)
        fvg_proximity_bars = 6

        valid_fvgs = []
        for fvg in self._fvg_detector.fair_value_gaps:
            if fvg.gap_type != target_type:
                continue
            if fvg.status not in (FVGStatus.ACTIVE, FVGStatus.PARTIALLY_FILLED):
                continue
            if fvg.fill_percentage > config.fvg_max_fill_pct:
                continue
            # Price must NOT already be inside the FVG (otherwise it's an immediate entry, not a tap setup)
            if fvg.low_price <= current_price <= fvg.high_price:
                continue
            # CONCURRENT FVG CHECK: FVG must have formed near the break candle
            # Pine Script requires FVG on the exact same bar as break
            if abs(fvg.start_index - break_idx) > fvg_proximity_bars:
                continue
            valid_fvgs.append(fvg)

        if not valid_fvgs:
            self.logger.debug(f"[FVG_RETEST] {pair}: BOS found but no valid FVGs for Type A setup")
            return

        # Sort by significance and take the best
        valid_fvgs.sort(key=lambda f: f.significance, reverse=True)

        # Calculate base confidence from BOS quality
        classification = self._classify_bos(df_trigger, bos_result, pip_value, config)
        base_confidence = min(classification['displacement_ratio'] / 3.0, 0.20)  # 0-20% from displacement

        # Store pending setups (up to max_pending_per_pair)
        if epic not in self._pending_setups:
            self._pending_setups[epic] = []

        expiry = now + timedelta(hours=config.setup_expiry_hours)
        added = 0

        for fvg in valid_fvgs:
            if len(self._pending_setups[epic]) >= config.max_pending_per_pair:
                break

            setup = PendingFVGSetup(
                epic=epic,
                direction=direction,
                fvg=fvg,
                swing_level=swing_level,
                bos_time=now,
                expiry_time=expiry,
                htf_ema_value=htf_ema_value,
                confidence_base=base_confidence,
            )
            self._pending_setups[epic].append(setup)
            added += 1

        if added > 0:
            self.logger.info(
                f"🔷 [FVG_RETEST] {pair}: Stored {added} Type A pending setup(s) "
                f"({direction}, swing={swing_level:.5f}, expires {expiry.strftime('%H:%M')})"
            )

    def _check_pending_taps(
        self, epic, pair, df_trigger, htf_direction, pip_value, config, now, htf_aligned: bool = True
    ) -> Optional[Dict]:
        """Check if price has tapped into any pending FVG zone"""
        setups = self._pending_setups.get(epic, [])
        if not setups:
            return None

        current_price = df_trigger.iloc[-1]['close']
        current_low = df_trigger.iloc[-1]['low']
        current_high = df_trigger.iloc[-1]['high']

        for setup in setups:
            # Direction must still match HTF
            if setup.direction != htf_direction:
                continue

            fvg = setup.fvg

            # Check if price has entered the FVG zone
            price_in_fvg = False
            if setup.direction == 'BULL':
                # For bullish: price dips INTO the FVG (wick or close touches FVG zone)
                price_in_fvg = current_low <= fvg.high_price and current_high >= fvg.low_price
            else:
                # For bearish: price rises INTO the FVG
                price_in_fvg = current_high >= fvg.low_price and current_low <= fvg.high_price

            if not price_in_fvg:
                continue

            # TAP DETECTED - build signal
            self.logger.info(
                f"🎯 [FVG_RETEST] {pair}: Type A FVG TAP detected! "
                f"({setup.direction}, FVG zone {fvg.low_price:.5f}-{fvg.high_price:.5f})"
            )

            signal = self._build_type_a_signal(
                epic, pair, df_trigger, setup, pip_value, config, now, htf_aligned
            )

            # Remove this setup regardless of signal validity
            self._pending_setups[epic].remove(setup)

            if signal:
                return signal

        return None

    def _expire_setups(self, epic, now, df_trigger, pip_value, config):
        """Remove expired or filled pending setups"""
        if epic not in self._pending_setups:
            return

        current_price = df_trigger.iloc[-1]['close']
        remaining = []

        for setup in self._pending_setups[epic]:
            # Time expiry
            if now > setup.expiry_time:
                continue

            # Check if FVG has been mostly filled
            fvg = setup.fvg
            gap_size = fvg.high_price - fvg.low_price
            if gap_size > 0:
                if setup.direction == 'BULL':
                    fill = max(0, fvg.high_price - max(current_price, fvg.low_price)) / gap_size
                else:
                    fill = max(0, min(current_price, fvg.high_price) - fvg.low_price) / gap_size
                if fill > config.fvg_max_fill_pct:
                    continue

            remaining.append(setup)

        self._pending_setups[epic] = remaining

    # =========================================================================
    # SIGNAL BUILDING
    # =========================================================================

    def _build_type_a_signal(
        self, epic, pair, df_trigger, setup: PendingFVGSetup, pip_value, config, now, htf_aligned: bool = True
    ) -> Optional[Dict]:
        """Build signal for Type A (Deep Value / FVG Tap) entry"""
        fvg = setup.fvg
        current_price = df_trigger.iloc[-1]['close']

        # SL: beyond FVG boundary + buffer
        sl_buffer = config.get_pair_sl_buffer(epic) * pip_value

        if setup.direction == 'BULL':
            sl_price = fvg.low_price - sl_buffer
            sl_pips = (current_price - sl_price) / pip_value
        else:
            sl_price = fvg.high_price + sl_buffer
            sl_pips = (sl_price - current_price) / pip_value

        # Minimum SL floor - reject signals with unrealistic tiny stops
        min_sl_pips = 8.0
        if sl_pips < min_sl_pips:
            self.logger.debug(f"[FVG_RETEST] {pair}: Type A SL too tight ({sl_pips:.1f} < {min_sl_pips} pips)")
            return None

        # Use fixed SL if configured and tighter
        fixed_sl = config.get_pair_fixed_stop_loss(epic)
        if fixed_sl and sl_pips > fixed_sl * 1.5:
            # FVG-based SL is too wide, use fixed
            if setup.direction == 'BULL':
                sl_price = current_price - fixed_sl * pip_value
            else:
                sl_price = current_price + fixed_sl * pip_value
            sl_pips = fixed_sl

        # TP: fixed per-pair config with R:R enforcement
        fixed_tp = config.get_pair_fixed_take_profit(epic)
        tp_pips = fixed_tp

        if setup.direction == 'BULL':
            tp_price = current_price + tp_pips * pip_value
        else:
            tp_price = current_price - tp_pips * pip_value

        # R:R check
        rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
        if rr_ratio < config.min_rr_ratio:
            self.logger.debug(f"[FVG_RETEST] {pair}: Type A R:R too low ({rr_ratio:.2f} < {config.min_rr_ratio})")
            return None

        # Confidence scoring (5 components × 20%)
        confidence = self._calculate_type_a_confidence(
            setup, df_trigger, rr_ratio, pip_value, config, htf_aligned
        )

        min_conf = config.get_pair_min_confidence(epic)
        if confidence < min_conf:
            self.logger.debug(f"[FVG_RETEST] {pair}: Type A confidence too low ({confidence:.2f} < {min_conf})")
            return None

        signal_type = 'BUY' if setup.direction == 'BULL' else 'SELL'

        return {
            'signal': signal_type,
            'signal_type': signal_type,
            'strategy': 'FVG_RETEST',
            'entry_type': 'DEEP_VALUE',
            'epic': epic,
            'pair': pair,
            'entry_price': current_price,
            'stop_loss': sl_price,
            'take_profit': tp_price,
            'sl_pips': round(sl_pips, 1),
            'tp_pips': round(tp_pips, 1),
            'risk_pips': round(sl_pips, 1),
            'reward_pips': round(tp_pips, 1),
            'stop_distance': round(sl_pips, 1),
            'limit_distance': round(tp_pips, 1),
            'rr_ratio': round(rr_ratio, 2),
            'confidence_score': round(confidence, 3),
            'fvg_zone': f"{fvg.low_price:.5f}-{fvg.high_price:.5f}",
            'fvg_size_pips': round(fvg.gap_size_pips, 1),
            'fvg_significance': round(fvg.significance, 3),
            'swing_level': setup.swing_level,
            'setup_age_minutes': round((now - setup.bos_time).total_seconds() / 60, 1),
            'timestamp': now.isoformat(),
            'is_scalp_trade': False,
        }

    def _build_type_b_signal(
        self, epic, pair, df_trigger, direction, swing_level,
        break_candle, classification, htf_ema_value, pip_value, config, now, htf_aligned: bool = True
    ) -> Optional[Dict]:
        """Build signal for Type B (Institutional Initiation) entry"""
        current_price = df_trigger.iloc[-1]['close']

        # SL: beyond the broken swing level + buffer
        sl_buffer = config.get_pair_sl_buffer(epic) * pip_value

        if direction == 'BULL':
            sl_price = swing_level - sl_buffer
            sl_pips = (current_price - sl_price) / pip_value
        else:
            sl_price = swing_level + sl_buffer
            sl_pips = (sl_price - current_price) / pip_value

        # Minimum SL floor - reject signals with unrealistic tiny stops
        min_sl_pips = 8.0
        if sl_pips < min_sl_pips:
            self.logger.debug(f"[FVG_RETEST] {pair}: Type B SL too tight ({sl_pips:.1f} < {min_sl_pips} pips)")
            return None

        # Clamp SL to fixed max
        fixed_sl = config.get_pair_fixed_stop_loss(epic)
        if fixed_sl and sl_pips > fixed_sl * 1.5:
            if direction == 'BULL':
                sl_price = current_price - fixed_sl * pip_value
            else:
                sl_price = current_price + fixed_sl * pip_value
            sl_pips = fixed_sl

        # TP
        fixed_tp = config.get_pair_fixed_take_profit(epic)
        tp_pips = fixed_tp

        if direction == 'BULL':
            tp_price = current_price + tp_pips * pip_value
        else:
            tp_price = current_price - tp_pips * pip_value

        # R:R check
        rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
        if rr_ratio < config.min_rr_ratio:
            self.logger.debug(f"[FVG_RETEST] {pair}: Type B R:R too low ({rr_ratio:.2f} < {config.min_rr_ratio})")
            return None

        # Confidence scoring
        confidence = self._calculate_type_b_confidence(
            classification, df_trigger, direction, htf_ema_value,
            rr_ratio, pip_value, config, htf_aligned
        )

        min_conf = config.get_pair_min_confidence(epic)
        if confidence < min_conf:
            self.logger.debug(f"[FVG_RETEST] {pair}: Type B confidence too low ({confidence:.2f} < {min_conf})")
            return None

        signal_type = 'BUY' if direction == 'BULL' else 'SELL'

        self.logger.info(
            f"⚡ [FVG_RETEST] {pair}: Type B INITIATION signal! "
            f"{signal_type} @ {current_price:.5f}, displacement={classification['displacement_ratio']:.2f}x ATR"
        )

        return {
            'signal': signal_type,
            'signal_type': signal_type,
            'strategy': 'FVG_RETEST',
            'entry_type': 'INITIATION',
            'epic': epic,
            'pair': pair,
            'entry_price': current_price,
            'stop_loss': sl_price,
            'take_profit': tp_price,
            'sl_pips': round(sl_pips, 1),
            'tp_pips': round(tp_pips, 1),
            'risk_pips': round(sl_pips, 1),
            'reward_pips': round(tp_pips, 1),
            'stop_distance': round(sl_pips, 1),
            'limit_distance': round(tp_pips, 1),
            'rr_ratio': round(rr_ratio, 2),
            'confidence_score': round(confidence, 3),
            'displacement_ratio': round(classification['displacement_ratio'], 2),
            'follow_through': classification['follow_through'],
            'volume_spike': classification['volume_spike'],
            'swing_level': swing_level,
            'timestamp': now.isoformat(),
            'is_scalp_trade': False,
        }

    # =========================================================================
    # CONFIDENCE SCORING
    # =========================================================================

    def _calculate_type_a_confidence(self, setup, df_trigger, rr_ratio, pip_value, config, htf_aligned: bool = True) -> float:
        """5-component confidence for Type A (Deep Value)"""
        scores = []

        # 1. HTF alignment (20%) - EMA position always passes, candle direction is modifier
        scores.append(0.20 if htf_aligned else 0.10)

        # 2. FVG significance (20%) - from FVG detector
        fvg_score = min(setup.fvg.significance, 1.0) * 0.20
        scores.append(fvg_score)

        # 3. Volume at BOS (20%) - base confidence from classification
        vol_score = min(setup.confidence_base * 5, 0.20)  # Scale up from 0-0.04 to 0-0.20
        scores.append(vol_score)

        # 4. Tap depth (20%) - how deep into FVG price has penetrated
        current_price = df_trigger.iloc[-1]['close']
        fvg = setup.fvg
        gap_size = fvg.high_price - fvg.low_price
        if gap_size > 0:
            if setup.direction == 'BULL':
                depth = (fvg.high_price - current_price) / gap_size
            else:
                depth = (current_price - fvg.low_price) / gap_size
            depth = max(0, min(depth, 1.0))
            # Optimal depth is 50-80% (deep tap, not fully filled)
            if 0.3 <= depth <= 0.8:
                tap_score = 0.20
            elif depth > 0:
                tap_score = 0.12
            else:
                tap_score = 0.05
        else:
            tap_score = 0.05
        scores.append(tap_score)

        # 5. R:R quality (20%) - scaled to 2.5 max
        rr_score = min(rr_ratio / 2.5, 1.0) * 0.20
        scores.append(rr_score)

        confidence = sum(scores)
        return max(0.0, min(confidence, config.max_confidence))

    def _calculate_type_b_confidence(
        self, classification, df_trigger, direction, htf_ema_value,
        rr_ratio, pip_value, config, htf_aligned: bool = True
    ) -> float:
        """5-component confidence for Type B (Initiation)"""
        scores = []

        # 1. HTF alignment (20%) - EMA position always passes, candle direction is modifier
        scores.append(0.20 if htf_aligned else 0.10)

        # 2. Displacement ratio (20%) - how strong the break candle was
        disp_ratio = classification['displacement_ratio']
        disp_score = min(disp_ratio / 3.0, 1.0) * 0.20  # 3x ATR = full score
        scores.append(disp_score)

        # 3. Volume (20%)
        vol_score = 0.20 if classification['volume_spike'] else 0.08
        scores.append(vol_score)

        # 4. Follow-through strength (20%)
        ft = classification['follow_through']
        required = config.follow_through_candles
        ft_score = min(ft / max(required, 1), 1.0) * 0.20
        scores.append(ft_score)

        # 5. R:R quality (20%)
        rr_score = min(rr_ratio / 2.5, 1.0) * 0.20
        scores.append(rr_score)

        confidence = sum(scores)
        return max(0.0, min(confidence, config.max_confidence))

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from DataFrame"""
        if len(df) < period + 1:
            return 0.0

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        tr_values = []
        for i in range(1, len(df)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return np.mean(tr_values) if tr_values else 0.0

        return np.mean(tr_values[-period:])

    def _set_cooldown(self, epic: str, config, now: datetime):
        """Set cooldown for pair"""
        cooldown_minutes = config.get_pair_cooldown_minutes(epic)
        self._cooldowns[epic] = now + timedelta(minutes=cooldown_minutes)
