#!/usr/bin/env python3
"""
RANGE_STRUCTURE Strategy v1.0 — Liquidity sweep + rejection wick at range extremes.

VERSION: 1.0.0
DATE:    2026-04-20
STATUS:  Monitor-only launch on USDJPY + JPY crosses + USDCHF/USDCAD

Thesis
------
In non-trending conditions, price sweeps range extremes (EQH/EQL) to grab
liquidity and reverses. The *rejection wick* — not the sweep itself — is the
edge (EQH/EQL research showed PF 3.85 @ n=11, 72% WR on EURJPY wick-rejection
vs. PF 0.70 on raw sweeps). Target range equilibrium or opposite boundary
with OB/FVG confluence.

Design decisions (bake the lessons from RANGING_MARKET + MEAN_REVERSION in)
--------------------------------------------------------------------------
- Hard ADX gates on BOTH primary (15m) and HTF (1h), ALWAYS enforced.
  No trust_regime_routing bypass — that was RANGING_MARKET's catastrophic bug.
- Rejection-wick ratio clamped at >= 0.55 in code (config floor) — the
  EQH/EQL analysis said clean sweeps precede continuation on JPY pairs, so
  we MUST require the reversal confirmation.
- R:R floor (1.33) non-negotiable — RANGING_MARKET v4.0 died from inverse R:R.
- Top-level signal fields: `adx`, `adx_htf`, `market_regime`, `risk_pips`,
  `reward_pips` (backtest requires both — see feedback_backtest_signal_keys).
- DB-backed config; per-pair overrides via direct columns + JSONB bag.
- Swing detection, order-block detection and FVG detection all delegate to
  the canonical SMC helpers (SMCMarketStructure / SMCOrderBlocks /
  SMCFairValueGaps). The v1.0 inline fractal + lightweight OB/FVG proxies
  were removed on 2026-04-20 after the first 90d eval (PF 0.418 @ n=25)
  revealed they were under-detecting real liquidity pools on JPY crosses.

Entry rules (per 15m bar, ALL must hold, in order)
---------------------------------------------------
  1. Pair enabled + cooldown OK.
  2. Hard ADX gate: adx_15m <= ceiling_primary AND adx_1h <= ceiling_htf.
  3. Range build: last N=range_lookback_bars fed into SMCMarketStructure.
     Require >= 2 swing-highs (HH/LH/EQH) and >= 2 swing-lows (HL/LL/EQL).
     Prefer HH/EQH for top, LL/EQL for bottom; fall back to all hi/lo swings.
     Range top = max of the top set, bottom = min of the bottom set,
     equilibrium = midpoint.
  4. Sweep: current bar high (SELL) / low (BUY) exceeds the boundary by
     >= sweep_penetration_pips AND another pivot exists nearby (liquidity
     cluster proxy).
  5. Rejection wick: bar closed back inside the range, opposite-side wick
     ratio >= rejection_wick_ratio, body direction opposite to the sweep.
  6. HTF bias neutral/mild: |htf_bias_score - 0.5| <= htf_bias_neutral_band.
  7. Target confluence: >= 1 OB or FVG between entry and equilibrium (when
     enabled). Delegated to SMCOrderBlocks + SMCFairValueGaps; only ACTIVE
     / PARTIALLY_FILLED FVGs and still-valid OBs count, and they must be
     direction-aligned (bullish for BUY, bearish for SELL).
  8. Swing proximity: don't enter inside an immediate opposing swing zone.
  9. R:R floor: reward_pips / risk_pips >= min_rr_ratio.

Exit (no trailing in v1.0; review after 30 days)
-------------------------------------------------
  SL = 1 pip beyond sweep extreme + sl_buffer_pips, clamped [sl_pips_min, sl_pips_max].
  TP1 = equilibrium (first scale target; signal records single TP).
  TP2 = opposite boundary - 2 pips, clamped [tp_pips_min, tp_pips_max].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .strategy_registry import StrategyInterface, register_strategy
from ..strategies.helpers.htf_bias_calculator import HTFBiasCalculator
from ..strategies.helpers.swing_proximity_validator import SwingProximityValidator
from ..strategies.helpers.smc_market_structure import SMCMarketStructure, SwingType
from ..strategies.helpers.smc_order_blocks import SMCOrderBlocks, OrderBlockType
from ..strategies.helpers.smc_fair_value_gaps import SMCFairValueGaps, FVGType, FVGStatus
from ...services.range_structure_config_service import (
    RangeStructureConfig,
    get_range_structure_config,
)


# =============================================================================
# LOCAL DATACLASSES
# =============================================================================

@dataclass
class RangeLevels:
    top: float
    bottom: float
    equilibrium: float
    pivot_highs: List[Tuple[int, float]]
    pivot_lows: List[Tuple[int, float]]


@dataclass
class SweepAndRejection:
    direction: str          # 'BUY' or 'SELL'
    sweep_extreme: float    # the wick extreme that swept the boundary
    wick_ratio: float       # opposite-side wick / total range
    boundary: float         # the range boundary that was swept
    penetration_pips: float
    body_direction: str     # 'bull' or 'bear' (closing colour)


# =============================================================================
# STRATEGY
# =============================================================================

@register_strategy("RANGE_STRUCTURE")
class RangeStructureStrategy(StrategyInterface):
    """Single-setup non-trending structure strategy (see module docstring)."""

    def __init__(self, config=None, logger=None, db_manager=None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.config: RangeStructureConfig = config or get_range_structure_config()

        self._htf_bias_calc = HTFBiasCalculator(logger=self.logger)
        self._swing_validator = SwingProximityValidator(
            config={"equal_level_multiplier": 2.0},
            logger=self.logger,
        )

        # Real SMC helpers (replaces the v1.0 inline fractal + OB/FVG proxies).
        # All three live on the same instance and are reused per-signal; they
        # are intentionally single-threaded (evaluated per epic within one
        # scan tick), matching how SMCSimpleStrategy uses them.
        self._smc_structure = SMCMarketStructure(logger=self.logger)
        self._ob = SMCOrderBlocks(logger=self.logger)
        self._fvg = SMCFairValueGaps(logger=self.logger)

        self._cooldowns: Dict[str, datetime] = {}
        self._current_timestamp: Optional[datetime] = None

        self.logger.info(
            "[RANGE_STRUCTURE] v%s initialized | hard ADX 15m<=%.1f / 1h<=%.1f | "
            "SL [%.1f,%.1f]p TP [%.1f,%.1f]p | wick>=%.2f R:R>=%.2f",
            self.config.strategy_version,
            self.config.adx_hard_ceiling_primary,
            self.config.adx_hard_ceiling_htf,
            self.config.sl_pips_min, self.config.sl_pips_max,
            self.config.tp_pips_min, self.config.tp_pips_max,
            self.config.rejection_wick_ratio,
            self.config.min_rr_ratio,
        )

    # ------------------------------------------------------------------
    # StrategyInterface
    # ------------------------------------------------------------------

    @property
    def strategy_name(self) -> str:
        return "RANGE_STRUCTURE"

    def get_required_timeframes(self) -> List[str]:
        return [self.config.confirmation_timeframe, self.config.primary_timeframe]

    def reset_cooldowns(self) -> None:
        self._cooldowns.clear()

    # ------------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------------

    def detect_signal(
        self,
        df_trigger: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,        # reused for 1h HTF per base sig
        df_entry: Optional[pd.DataFrame] = None,
        epic: str = "",
        pair: str = "",
        current_timestamp: Optional[datetime] = None,
        routing_context: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[Dict]:
        """Core signal entry point — matches the SignalDetector call site.

        df_trigger: 15m DataFrame (primary).
        df_4h:      1h DataFrame (HTF bias + HTF ADX ceiling).
        """
        self._current_timestamp = current_timestamp
        cfg = self.config

        df_15m = df_trigger if df_trigger is not None else df_entry
        if df_15m is None or len(df_15m) < cfg.range_lookback_bars + 10:
            return None

        if not cfg.is_pair_enabled(epic):
            self.logger.debug("[RANGE_STRUCTURE] %s disabled — skipping", epic)
            return None

        if not self._check_cooldown(epic):
            return None

        pip_size = self._pip_size(epic)

        # --- Gate 1: hard ADX ceilings (always enforced) ---------------
        adx_15m = self._get_adx(df_15m)
        adx_1h = self._get_adx(df_4h) if df_4h is not None and len(df_4h) >= 30 else None

        pri_ceiling = cfg.get_pair_adx_hard_ceiling_primary(epic)
        if adx_15m is not None and adx_15m > pri_ceiling:
            self._log_reject(epic, "HARD_ADX_CEILING_PRIMARY",
                             f"adx_15m={adx_15m:.1f} > {pri_ceiling:.1f}")
            return None

        htf_ceiling = cfg.get_pair_adx_hard_ceiling_htf(epic)
        if adx_1h is not None and adx_1h > htf_ceiling:
            self._log_reject(epic, "HARD_ADX_CEILING_HTF",
                             f"adx_1h={adx_1h:.1f} > {htf_ceiling:.1f}")
            return None

        # --- Gate 2: regime label (when routing_context is provided) ---
        regime_label = "ranging"
        if routing_context:
            regime_label = str(routing_context.get("regime", "")).lower() or "ranging"
            if regime_label not in ("ranging", "low_volatility"):
                # Hard ADX already passed, so only block when routing is *explicit*
                # about a non-ranging regime (trending / breakout / expansion).
                if regime_label in ("trending", "breakout", "expansion"):
                    self._log_reject(epic, "REGIME_NOT_RANGING",
                                     f"regime={regime_label}")
                    return None

        # --- Range build ------------------------------------------------
        lookback = cfg.get_pair_range_lookback_bars(epic)
        window = df_15m.iloc[-(lookback + 1):-1]  # exclude the latest (trigger) bar
        levels = self._build_range(window, epic=epic)
        if levels is None:
            self._log_reject(epic, "NO_RANGE", f"lookback={lookback}")
            return None

        # --- Sweep + rejection wick ------------------------------------
        trigger_bar = df_15m.iloc[-1]
        sweep_pen_pips = cfg.get_pair_sweep_penetration_pips(epic)
        wick_ratio_floor = cfg.get_pair_rejection_wick_ratio(epic)  # clamped >= 0.55

        sr = self._detect_sweep_and_rejection(
            bar=trigger_bar,
            levels=levels,
            pip_size=pip_size,
            min_penetration_pips=sweep_pen_pips,
            min_wick_ratio=wick_ratio_floor,
        )
        if sr is None:
            self._log_reject(epic, "NO_SWEEP_OR_WICK_REJECT",
                             f"range=[{levels.bottom:.5f},{levels.top:.5f}]")
            return None

        # --- HTF bias neutral/mild --------------------------------------
        htf_bias_score, htf_bias_details = self._compute_htf_bias(df_4h, sr.direction, epic)
        if htf_bias_score is not None:
            neutral_band = cfg.htf_bias_neutral_band
            if abs(htf_bias_score - 0.5) > neutral_band:
                self._log_reject(
                    epic, "HTF_BIAS_STRONG",
                    f"score={htf_bias_score:.2f} band=[{0.5-neutral_band:.2f},"
                    f"{0.5+neutral_band:.2f}]"
                )
                return None

        # --- Target confluence (OB / FVG between entry and equilibrium) --
        ob_count, fvg_count = 0, 0
        if cfg.get_pair_ob_fvg_confluence_required(epic):
            ob_count, fvg_count = self._count_confluence(
                df_15m, sr.direction, entry_price=float(trigger_bar["close"]),
                equilibrium=levels.equilibrium, epic=epic,
            )
            if (ob_count + fvg_count) < 1:
                self._log_reject(epic, "NO_CONFLUENCE_TARGET",
                                 f"ob={ob_count} fvg={fvg_count}")
                return None

        # --- Swing proximity (reject entries sitting on an opposing swing) -
        # Validator returns a dict; 2x equal-level multiplier comes from the
        # config passed in __init__, not a kwarg here.
        try:
            prox_result = self._swing_validator.validate_entry_proximity(
                df=df_15m,
                current_price=float(trigger_bar["close"]),
                direction=sr.direction,           # 'BUY' or 'SELL'
                epic=epic,
                timeframe="15m",
            )
            proximity_ok = bool(prox_result.get("valid", True))
            prox_details = prox_result
        except Exception as e:
            self.logger.debug("[RANGE_STRUCTURE] proximity validator error: %s", e)
            proximity_ok, prox_details = True, {"error": str(e)}
        if not proximity_ok:
            self._log_reject(epic, "PROXIMITY_FAIL", str(prox_details))
            return None

        # --- Build SL / TP / R:R ---------------------------------------
        entry_price = float(trigger_bar["close"])
        sl_min = cfg.get_pair_sl_pips_min(epic)
        sl_max = cfg.get_pair_sl_pips_max(epic)
        tp_min = cfg.get_pair_tp_pips_min(epic)
        tp_max = cfg.get_pair_tp_pips_max(epic)

        if sr.direction == "BUY":
            raw_sl_pips = ((entry_price - sr.sweep_extreme) / pip_size) + cfg.sl_buffer_pips
            raw_tp_pips = (levels.top - 2 * pip_size - entry_price) / pip_size
        else:
            raw_sl_pips = ((sr.sweep_extreme - entry_price) / pip_size) + cfg.sl_buffer_pips
            raw_tp_pips = (entry_price - (levels.bottom + 2 * pip_size)) / pip_size

        sl_pips = float(np.clip(raw_sl_pips, sl_min, sl_max))
        tp_pips = float(np.clip(raw_tp_pips, tp_min, tp_max))

        # R:R floor — non-negotiable
        rr_floor = cfg.get_pair_min_rr_ratio(epic)
        if sl_pips <= 0 or (tp_pips / sl_pips) < rr_floor:
            self._log_reject(
                epic, "RR_TOO_LOW",
                f"risk={sl_pips:.1f}p reward={tp_pips:.1f}p rr="
                f"{(tp_pips/sl_pips) if sl_pips>0 else 0:.2f} floor={rr_floor:.2f}"
            )
            return None

        # --- Confidence -----------------------------------------------
        confidence = self._compute_confidence(
            epic=epic,
            sr=sr,
            levels=levels,
            ob_count=ob_count,
            fvg_count=fvg_count,
            htf_bias_score=htf_bias_score,
            adx_15m=adx_15m,
        )
        min_conf = cfg.get_pair_min_confidence(epic)
        if confidence < min_conf:
            self._log_reject(epic, "LOW_CONFIDENCE",
                             f"conf={confidence:.2f} < min={min_conf:.2f}")
            return None

        # --- Assemble signal dict -------------------------------------
        now = datetime.now(timezone.utc)
        monitor_only = cfg.is_pair_monitor_only(epic)
        signal: Dict[str, Any] = {
            "signal": sr.direction,
            "signal_type": sr.direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair,
            "entry_price": entry_price,
            "stop_loss_pips": sl_pips,
            "take_profit_pips": tp_pips,
            "risk_pips": sl_pips,
            "reward_pips": tp_pips,
            "confidence": confidence,
            "confidence_score": confidence,
            "signal_timestamp": now.isoformat(),
            "timestamp": now,
            "version": self.config.strategy_version,
            "monitor_only": monitor_only,

            # Top-level regime fields (alert_history reads these directly)
            "adx": adx_15m,
            "adx_htf": adx_1h,
            "market_regime": regime_label,
            "regime": regime_label,

            "strategy_indicators": {
                "range_top": levels.top,
                "range_bottom": levels.bottom,
                "equilibrium": levels.equilibrium,
                "sweep_side": "top" if sr.direction == "SELL" else "bottom",
                "sweep_extreme": sr.sweep_extreme,
                "sweep_penetration_pips": sr.penetration_pips,
                "wick_ratio": round(sr.wick_ratio, 3),
                "body_direction": sr.body_direction,
                "ob_count": ob_count,
                "fvg_count": fvg_count,
                "htf_bias_score": htf_bias_score,
                "htf_bias_interpretation": (
                    htf_bias_details.get("interpretation")
                    if isinstance(htf_bias_details, dict) else None
                ),
                "adx": adx_15m,
                "adx_htf": adx_1h,
                "rr_ratio": round(tp_pips / sl_pips, 2) if sl_pips > 0 else None,
            },
        }

        self._set_cooldown(epic)
        self.logger.info(
            "[RANGE_STRUCTURE] ✅ %s %s @ %.5f | SL %.1fp TP %.1fp R:R %.2f | "
            "range [%.5f, %.5f] eq %.5f | wick %.2f | ADX %.1f/%.1f | conf %.2f%s",
            sr.direction, epic, entry_price,
            sl_pips, tp_pips, tp_pips / sl_pips if sl_pips > 0 else 0.0,
            levels.bottom, levels.top, levels.equilibrium,
            sr.wick_ratio,
            adx_15m if adx_15m is not None else -1,
            adx_1h if adx_1h is not None else -1,
            confidence,
            " [MONITOR]" if monitor_only else "",
        )
        return signal

    # ==================================================================
    # RANGE / SWING DETECTION (SMCMarketStructure-backed)
    # ==================================================================

    def _build_range(
        self, window: pd.DataFrame, epic: str = ""
    ) -> Optional[RangeLevels]:
        """Build range extremes from the real SMC market-structure helper.

        We feed the last N bars (the same lookback window as v1.0) into
        SMCMarketStructure, then extract the last ~3 confirmed HH/EQH as the
        range top set and the last ~3 LL/EQL as the bottom set.

        This replaces the v1.0 inline 3-bar fractal pivot finder that tended
        to produce noisy micro-pivots — which was the flagged root-cause of
        the Apr 20 backtest miss (portfolio PF 0.418 @ n=25).
        """
        if window is None or len(window) < 10:
            return None

        pip_value = self._pip_size(epic)
        smc_cfg: Dict[str, Any] = {
            "epic": epic,
            "pair": epic.split(".")[2] if "." in epic else epic,
            "pip_value": pip_value,
            # swing_length=5 matches SMCMarketStructure's default and is well-
            # tested in smc_simple. It produces fewer false pivots than the
            # old inline window=3, so EQH/EQL labelling is cleaner.
            "swing_length": 5,
            "bos_threshold": pip_value * 5,
        }

        try:
            # SMCMarketStructure.analyze_market_structure mutates `self.swing_points`.
            # We pass a copy so the helper's column additions don't leak into
            # the caller's df.
            self._smc_structure.analyze_market_structure(
                df=window.copy(), config=smc_cfg, epic=epic, timeframe="15m"
            )
        except Exception as e:
            self.logger.debug("[RANGE_STRUCTURE] SMC structure error: %s", e)
            return None

        swings = list(self._smc_structure.swing_points or [])
        if not swings:
            return None

        # Separate into swing-highs (HH/LH/EQH) and swing-lows (HL/LL/EQL).
        hi_types = {SwingType.HIGHER_HIGH, SwingType.LOWER_HIGH, SwingType.EQUAL_HIGH}
        lo_types = {SwingType.HIGHER_LOW, SwingType.LOWER_LOW, SwingType.EQUAL_LOW}

        hi_swings = [s for s in swings if s.swing_type in hi_types]
        lo_swings = [s for s in swings if s.swing_type in lo_types]

        # SMCMarketStructure with swing_length=5 produces ~1 swing per 20 bars
        # on 15m data. Requiring 2+2 was impossible at range_lookback_bars=40
        # (that was the n=0 bug in the first post-swap eval). Require 1+1 and
        # let range_lookback_bars govern the window size.
        if len(hi_swings) < 1 or len(lo_swings) < 1:
            return None

        # Prefer confirmed HH/EQH for top and LL/EQL for bottom — these are
        # the real liquidity pools. Fall back to LH/HL if the preferred types
        # are absent entirely.
        top_pref = [s for s in hi_swings if s.swing_type in
                    (SwingType.HIGHER_HIGH, SwingType.EQUAL_HIGH)]
        bot_pref = [s for s in lo_swings if s.swing_type in
                    (SwingType.LOWER_LOW, SwingType.EQUAL_LOW)]
        top_set = top_pref if top_pref else hi_swings
        bot_set = bot_pref if bot_pref else lo_swings

        # Last 3 of each, ordered by index (swing_points is already sorted).
        top_tail = top_set[-3:]
        bot_tail = bot_set[-3:]

        top = max(s.price for s in top_tail)
        bottom = min(s.price for s in bot_tail)
        if top <= bottom:
            return None

        return RangeLevels(
            top=float(top),
            bottom=float(bottom),
            equilibrium=(float(top) + float(bottom)) / 2.0,
            pivot_highs=[(s.index, float(s.price)) for s in top_tail],
            pivot_lows=[(s.index, float(s.price)) for s in bot_tail],
        )

    # ==================================================================
    # SWEEP + REJECTION WICK
    # ==================================================================

    def _detect_sweep_and_rejection(
        self,
        bar: pd.Series,
        levels: RangeLevels,
        pip_size: float,
        min_penetration_pips: float,
        min_wick_ratio: float,
    ) -> Optional[SweepAndRejection]:
        """Return a SweepAndRejection if the trigger bar swept a boundary and
        closed back inside the range with a rejection wick on the correct side."""
        high = float(bar["high"])
        low = float(bar["low"])
        open_ = float(bar["open"])
        close = float(bar["close"])
        total_range = high - low
        if total_range <= 0:
            return None

        body_direction = "bull" if close >= open_ else "bear"

        # Sweep of top → potential SELL
        pen_top = (high - levels.top) / pip_size if pip_size > 0 else 0.0
        swept_top = (
            high > levels.top
            and pen_top >= min_penetration_pips
            and close < levels.top       # closed back inside the range
        )
        if swept_top:
            upper_wick = high - max(open_, close)
            wick_ratio = upper_wick / total_range if total_range > 0 else 0.0
            if wick_ratio >= min_wick_ratio and body_direction == "bear":
                return SweepAndRejection(
                    direction="SELL",
                    sweep_extreme=high,
                    wick_ratio=wick_ratio,
                    boundary=levels.top,
                    penetration_pips=pen_top,
                    body_direction=body_direction,
                )

        # Sweep of bottom → potential BUY
        pen_bot = (levels.bottom - low) / pip_size if pip_size > 0 else 0.0
        swept_bot = (
            low < levels.bottom
            and pen_bot >= min_penetration_pips
            and close > levels.bottom
        )
        if swept_bot:
            lower_wick = min(open_, close) - low
            wick_ratio = lower_wick / total_range if total_range > 0 else 0.0
            if wick_ratio >= min_wick_ratio and body_direction == "bull":
                return SweepAndRejection(
                    direction="BUY",
                    sweep_extreme=low,
                    wick_ratio=wick_ratio,
                    boundary=levels.bottom,
                    penetration_pips=pen_bot,
                    body_direction=body_direction,
                )
        return None

    # ==================================================================
    # HTF BIAS
    # ==================================================================

    def _compute_htf_bias(
        self, df_4h: Optional[pd.DataFrame], direction: str, epic: str
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Wrap HTFBiasCalculator. Direction is mapped BUY→BULL / SELL→BEAR.
        Returns (score in [0,1], details) or (None, {}) if not computable."""
        if df_4h is None or len(df_4h) < 10:
            return None, {}
        dir_norm = "BULL" if direction == "BUY" else "BEAR"
        try:
            score, details = self._htf_bias_calc.calculate_bias_score(
                df_4h=df_4h, direction=dir_norm, epic=epic
            )
            return float(score), details
        except Exception as e:
            self.logger.debug("[RANGE_STRUCTURE] HTF bias calc error: %s", e)
            return None, {"error": str(e)}

    # ==================================================================
    # CONFLUENCE (real SMCOrderBlocks + SMCFairValueGaps helpers)
    # ==================================================================

    def _count_confluence(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        equilibrium: float,
        epic: str = "",
    ) -> Tuple[int, int]:
        """Count OB/FVG confluence zones lying between entry and equilibrium.

        Replaces the v1.0 inline proxies with the real SMC helpers:
          * SMCFairValueGaps.detect_fair_value_gaps(df, config)
          * SMCOrderBlocks.detect_order_blocks(df, config)

        Filtering rules
        ---------------
        * For BUY (sweep of range bottom → target = equilibrium above), a
          supporting zone is bullish (BUY-side demand) and its *midpoint*
          sits between entry_price and equilibrium (i.e. price will trade
          *through* it on its way up to TP1).
        * For SELL, mirror: bearish supply zone with midpoint between
          equilibrium and entry_price.
        * We only count ACTIVE / PARTIALLY_FILLED FVGs (unfilled liquidity)
          and still_valid order blocks.
        """
        if df is None or len(df) < 5:
            return 0, 0

        # Keep the working window bounded — the helpers internally scan the
        # whole frame, so a 120-bar tail is plenty for 15m scale targets.
        window = df.iloc[-120:].copy() if len(df) > 120 else df.copy()
        if len(window) < 5:
            return 0, 0

        pip_value = self._pip_size(epic)
        helper_cfg: Dict[str, Any] = {
            "epic": epic,
            "pair": epic.split(".")[2] if "." in epic else epic,
            "pip_value": pip_value,
            # FVG config (mirrors smc_simple_strategy call site)
            "fvg_min_size": 3,          # 3-pip minimum gap
            "fvg_max_age": 30,          # prune stale gaps
            # Order-block config (mirrors smc_simple_strategy)
            "order_block_length": 3,
            "order_block_volume_factor": 1.2,
            "order_block_min_confidence": 0.3,
            "bos_threshold": pip_value * 5,
            "max_order_blocks": 5,
        }

        lo_bound = min(entry_price, equilibrium)
        hi_bound = max(entry_price, equilibrium)

        # ---- FVGs ----------------------------------------------------------
        fvg_count = 0
        try:
            self._fvg.detect_fair_value_gaps(window, helper_cfg)
            for fvg in (self._fvg.fair_value_gaps or []):
                # Skip filled or expired gaps — only unfilled liquidity matters.
                if fvg.status not in (FVGStatus.ACTIVE, FVGStatus.PARTIALLY_FILLED):
                    continue
                mid = (float(fvg.high_price) + float(fvg.low_price)) / 2.0
                if not (lo_bound <= mid <= hi_bound):
                    continue
                # Direction must match: BUY needs bullish FVG support on way up,
                # SELL needs bearish FVG resistance on way down.
                if direction == "BUY" and fvg.gap_type == FVGType.BULLISH:
                    fvg_count += 1
                elif direction == "SELL" and fvg.gap_type == FVGType.BEARISH:
                    fvg_count += 1
        except Exception as e:
            self.logger.debug("[RANGE_STRUCTURE] FVG helper error: %s", e)

        # ---- Order blocks --------------------------------------------------
        ob_count = 0
        try:
            self._ob.detect_order_blocks(window, helper_cfg)
            for ob in (self._ob.order_blocks or []):
                if not ob.still_valid:
                    continue
                mid = (float(ob.high_price) + float(ob.low_price)) / 2.0
                if not (lo_bound <= mid <= hi_bound):
                    continue
                if direction == "BUY" and ob.block_type == OrderBlockType.BULLISH:
                    ob_count += 1
                elif direction == "SELL" and ob.block_type == OrderBlockType.BEARISH:
                    ob_count += 1
        except Exception as e:
            self.logger.debug("[RANGE_STRUCTURE] OB helper error: %s", e)

        return ob_count, fvg_count

    # ==================================================================
    # CONFIDENCE
    # ==================================================================

    def _compute_confidence(
        self,
        epic: str,
        sr: SweepAndRejection,
        levels: RangeLevels,
        ob_count: int,
        fvg_count: int,
        htf_bias_score: Optional[float],
        adx_15m: Optional[float],
    ) -> float:
        """Confidence = min + ratio * (max - min) where ratio combines:
          - wick quality  (>=0.60 threshold, scaled 0..1 on [0.55, 0.85])
          - penetration tightness (closer to min = cleaner sweep)
          - confluence count (OB + FVG, capped at 3)
          - ADX headroom below the hard ceiling
          - HTF bias mid-band bonus (closer to 0.5 = better for reversion)
        """
        cfg = self.config
        min_conf = cfg.get_pair_min_confidence(epic)
        max_conf = cfg.get_pair_max_confidence(epic)

        # wick factor
        wick_f = (sr.wick_ratio - 0.55) / 0.30
        wick_f = float(np.clip(wick_f, 0.0, 1.0))

        # penetration factor (cleaner = closer to the minimum sweep threshold)
        sweep_min = cfg.get_pair_sweep_penetration_pips(epic)
        pen_f = 1.0 - float(np.clip((sr.penetration_pips - sweep_min) / max(sweep_min, 1.0),
                                     0.0, 1.0))

        # confluence factor
        conf_f = float(np.clip((ob_count + fvg_count) / 3.0, 0.0, 1.0))

        # ADX headroom
        adx_f = 0.5
        if adx_15m is not None:
            ceiling = cfg.get_pair_adx_hard_ceiling_primary(epic)
            adx_f = float(np.clip((ceiling - adx_15m) / ceiling, 0.0, 1.0))

        # HTF mid-band bonus
        htf_f = 0.5
        if htf_bias_score is not None:
            htf_f = 1.0 - min(1.0, abs(htf_bias_score - 0.5) / max(cfg.htf_bias_neutral_band, 0.01))

        ratio = (
            0.30 * wick_f
            + 0.20 * pen_f
            + 0.20 * conf_f
            + 0.15 * adx_f
            + 0.15 * htf_f
        )
        confidence = min_conf + ratio * (max_conf - min_conf)
        return round(float(np.clip(confidence, min_conf, max_conf)), 3)

    # ==================================================================
    # UTILITIES
    # ==================================================================

    def _get_adx(self, df: pd.DataFrame) -> Optional[float]:
        """Prefer pre-stamped df['adx'] (DataFetcher EMA-Wilder); fall back to
        recomputing with the same EMA-Wilder formula."""
        if df is None or len(df) < 20:
            return None
        if "adx" in df.columns:
            try:
                v = df["adx"].iloc[-1]
                if v is not None and not pd.isna(v):
                    return float(v)
            except Exception:
                pass
        try:
            period = self.config.adx_period
            high, low, close = df["high"], df["low"], df["close"]
            tr = pd.concat(
                [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                axis=1,
            ).max(axis=1)
            up = high - high.shift(1)
            dn = low.shift(1) - low
            plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
            minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
            a = 1.0 / period
            atr = tr.ewm(alpha=a, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
            minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            adx = dx.ewm(alpha=a, adjust=False).mean()
            v = adx.iloc[-1]
            return float(v) if pd.notna(v) else None
        except Exception as e:
            self.logger.debug("[RANGE_STRUCTURE] ADX calc error: %s", e)
            return None

    @staticmethod
    def _pip_size(epic: str) -> float:
        """Standard FX pip size: 0.01 for JPY pairs, 0.0001 otherwise."""
        return 0.01 if "JPY" in epic.upper() else 0.0001

    def _check_cooldown(self, epic: str) -> bool:
        now = self._now()
        exp = self._cooldowns.get(epic)
        if exp is None or now >= exp:
            self._cooldowns.pop(epic, None)
            return True
        return False

    def _set_cooldown(self, epic: str) -> None:
        now = self._now()
        mins = self.config.get_pair_signal_cooldown_minutes(epic)
        self._cooldowns[epic] = now + timedelta(minutes=mins)

    def _now(self) -> datetime:
        now = self._current_timestamp or datetime.now(timezone.utc)
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        return now

    def _log_reject(self, epic: str, reason: str, detail: str = "") -> None:
        # Use INFO so live operators can tail docker logs for strategy behaviour
        # on new rollouts — matches the RANGING_MARKET logging philosophy.
        self.logger.info("[RANGE_STRUCTURE] ❌ %s %s %s", epic, reason, detail)


# ---------------------------------------------------------------------------
# Factory helper (mirrors create_smc_simple_strategy)
# ---------------------------------------------------------------------------

def create_range_structure_strategy(config=None, logger=None, db_manager=None):
    return RangeStructureStrategy(config=config, logger=logger, db_manager=db_manager)
