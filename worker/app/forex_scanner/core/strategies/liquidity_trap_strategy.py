#!/usr/bin/env python3
"""
LIQUIDITY_TRAP strategy — Prior-Day-Level Fade (1H trigger).

Detects failed breakouts of prior-day high/low:
  - Bar sweeps above PDH (prior-day high) but 1H CLOSE returns below PDH → SELL
  - Bar sweeps below PDL (prior-day low)  but 1H CLOSE returns above PDL → BUY

This is a structural liquidity-trap fade. Mechanically orthogonal to all
active trend strategies (SMC_SIMPLE/MOMENTUM/XAU_GOLD) which classify the
wick beyond PDH as a "Weak breakout body" rejection. Also orthogonal to
MEAN_REVERSION (needs extreme RSI + low-vol) and IMPULSE_FADE (5m late-US).

Edge validated via 60d gap analysis:
  - EURJPY: n=28/60d, MFE 33.3p, MAE 15.4p (MFE/MAE = 2.16×)
  - USDJPY: n=42/60d, MFE 23.5p, MAE 18.6p
  - ~70% of triggers had no concurrent active-strategy signal within ±2h
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from services.liquidity_trap_config_service import (
        LiquidityTrapConfig,
        get_liquidity_trap_config,
    )
except ImportError:
    from forex_scanner.services.liquidity_trap_config_service import (
        LiquidityTrapConfig,
        get_liquidity_trap_config,
    )

try:
    from ...alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
    except ImportError:
        StrategyRejectionManager = None  # type: ignore[assignment,misc]

from .strategy_registry import StrategyInterface, register_strategy

logger = logging.getLogger(__name__)


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


def _apply_config_override(config: LiquidityTrapConfig, overrides: Optional[Dict[str, Any]]) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        try:
            if isinstance(current, bool):
                new_val = str(value).strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                new_val = int(float(value))
            elif isinstance(current, float):
                new_val = float(value)
            else:
                new_val = value
            setattr(config, key, new_val)
            # Per-pair DB rows take precedence over the global attribute in get_for_pair(),
            # so also overwrite the key in every pair's row to ensure --override wins.
            for pair_row in config._pair_overrides.values():
                if key in pair_row:
                    pair_row[key] = new_val
        except Exception as exc:
            logger.warning("LIQUIDITY_TRAP override failed for %s=%r: %s", key, value, exc)


@register_strategy("LIQUIDITY_TRAP")
class LiquidityTrapStrategy(StrategyInterface):
    """Fade failed 1H breakouts of the prior calendar day's high/low."""

    def __init__(
        self,
        config: Optional[LiquidityTrapConfig] = None,
        logger=None,
        db_manager=None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self._config_override = config_override

        self.config = copy.deepcopy(config or get_liquidity_trap_config())
        _apply_config_override(self.config, self._config_override)

        self._cooldowns: Dict[str, datetime] = {}
        self._current_timestamp: Optional[datetime] = None

        self._rej_mgr = None
        if db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager("LIQUIDITY_TRAP", db_manager)
            except Exception:
                pass

        self.logger.info(
            "[LIQUIDITY_TRAP] v%s initialized | trigger_tf=%s "
            "buf=%.1fp wick_min=%.2f SL/TP=%.1f/%.1f cooldown=%dm",
            self.config.version,
            self.config.trigger_timeframe,
            self.config.breakout_buffer_pips,
            self.config.wick_fraction_min,
            self.config.fixed_stop_loss_pips,
            self.config.fixed_take_profit_pips,
            self.config.signal_cooldown_minutes,
        )

    # ------------------------------------------------------------------
    # StrategyInterface required methods
    # ------------------------------------------------------------------

    @property
    def strategy_name(self) -> str:
        return "LIQUIDITY_TRAP"

    def get_required_timeframes(self) -> List[str]:
        return ["1h"]

    def reset_cooldowns(self) -> None:
        """Called by BacktestScanner between pairs."""
        self._cooldowns.clear()

    def flush_rejections(self) -> None:
        if self._rej_mgr is not None:
            self._rej_mgr.flush()

    def get_config(self) -> LiquidityTrapConfig:
        return self.config

    # ------------------------------------------------------------------
    # Rejection helpers
    # ------------------------------------------------------------------

    def _log_rejection(
        self,
        epic: str,
        stage: str,
        value: Any,
        pair: str = "",
        direction: Optional[str] = None,
        hour_utc: Optional[int] = None,
    ) -> None:
        detail_str = f" ({value})" if value is not None else ""
        self.logger.info("[LIQUIDITY_TRAP] %s ❌ %s%s", epic, stage, detail_str)
        if self._rej_mgr is not None:
            try:
                self._rej_mgr.reject(
                    stage=stage,
                    reason=f"{stage}{detail_str}",
                    epic=epic,
                    pair=pair or epic,
                    direction=direction,
                    hour_utc=hour_utc,
                    scan_timestamp=self._current_timestamp,
                    details={"value": value},
                )
            except Exception:
                pass

    def _check_cooldown(self, epic: str, eval_ts: datetime, cooldown_mins: int) -> bool:
        """Return True if still in cooldown (should skip), False if OK to proceed."""
        if epic in self._cooldowns:
            elapsed = (eval_ts - self._cooldowns[epic]).total_seconds() / 60
            if elapsed < cooldown_mins:
                self.logger.debug(
                    "[LIQUIDITY_TRAP] %s: in cooldown (%.1f / %d min)", epic, elapsed, cooldown_mins
                )
                return True
        return False

    def _set_cooldown(self, epic: str, eval_ts: Optional[datetime] = None) -> None:
        self._cooldowns[epic] = eval_ts or datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    def detect_signal(
        self,
        df_trigger: Optional[pd.DataFrame] = None,
        epic: str = "",
        pair: str = "",
        spread_pips: float = 1.5,
        current_timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate the most-recent completed 1H candle for a liquidity-trap setup."""
        try:
            config = self.config
            pair_key = epic or pair or "UNKNOWN"

            # ── Step 1: Data validation ───────────────────────────────────
            if df_trigger is None or len(df_trigger) < 50:
                rows = len(df_trigger) if df_trigger is not None else 0
                self._log_rejection(pair_key, "INSUFFICIENT_DATA", rows)
                return None

            if not config.is_pair_enabled(pair_key):
                self.logger.debug("[LIQUIDITY_TRAP] %s: pair not enabled", pair_key)
                return None

            # ── Step 2: Timestamp ─────────────────────────────────────────
            if current_timestamp is None:
                ts: datetime = datetime.now(timezone.utc)
            else:
                ts = current_timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

            self._current_timestamp = ts
            hour_utc: int = int(ts.hour)

            # ── Step 3: Session check ─────────────────────────────────────
            enabled_hours = config.get_pair_enabled_hours(pair_key)
            if hour_utc not in enabled_hours:
                self._log_rejection(pair_key, "OUTSIDE_SESSION", hour_utc, hour_utc=hour_utc)
                return None

            # ── Step 4: Cooldown check ────────────────────────────────────
            cooldown_mins = config.get_pair_cooldown_minutes(pair_key)
            if self._check_cooldown(pair_key, ts, cooldown_mins):
                self._log_rejection(pair_key, "COOLDOWN", None, hour_utc=hour_utc)
                return None

            # ── Step 5: ATR check (on 1H bars) ────────────────────────────
            pip = _pip_size(pair or epic or "")

            tr = pd.concat([
                df_trigger["high"] - df_trigger["low"],
                (df_trigger["high"] - df_trigger["close"].shift(1)).abs(),
                (df_trigger["low"]  - df_trigger["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_val = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

            if pd.isna(atr_val) or atr_val <= 0:
                self._log_rejection(pair_key, "ATR_UNAVAILABLE", None, hour_utc=hour_utc)
                return None

            atr_pips = atr_val / pip

            min_atr = config.get_pair_min_atr(pair_key)
            max_atr = config.get_pair_max_atr(pair_key)
            if not (min_atr <= atr_pips <= max_atr):
                self._log_rejection(pair_key, "ATR_OUT_OF_RANGE", round(atr_pips, 1), hour_utc=hour_utc)
                return None

            # ── Step 6: Compute PDH/PDL ───────────────────────────────────
            # start_time is a COLUMN, not the index (DataFetcher convention)
            st_col = df_trigger["start_time"].copy()
            if hasattr(st_col.iloc[0], "tzinfo") and st_col.iloc[0].tzinfo is None:
                st_col = st_col.dt.tz_localize("UTC")

            today_date = ts.date()
            yesterday_date = today_date - timedelta(days=1)
            prev_day_mask = st_col.dt.date == yesterday_date
            prev_day_bars = df_trigger[prev_day_mask]

            if len(prev_day_bars) < 4:
                self._log_rejection(pair_key, "INSUFFICIENT_PREV_DAY_BARS", len(prev_day_bars), hour_utc=hour_utc)
                return None

            pdh = prev_day_bars["high"].max()
            pdl = prev_day_bars["low"].min()

            # ── Step 7: Inspect the last completed bar ────────────────────
            bar = df_trigger.iloc[-1]
            bar_open  = float(bar["open"])
            bar_high  = float(bar["high"])
            bar_low   = float(bar["low"])
            bar_close = float(bar["close"])
            bar_range = bar_high - bar_low

            if bar_range < pip * 2:  # degenerate / zero bar
                return None

            buffer = float(config.get_for_pair(pair_key, "breakout_buffer_pips", config.breakout_buffer_pips)) * pip

            # ── Step 8a: Failed-high detection (SELL) ─────────────────────
            sell_signal = None
            if bar_high > pdh + buffer and bar_close < pdh:
                wick_tail = bar_high - max(bar_open, bar_close)
                wick_fraction = wick_tail / bar_range
                wick_min = float(config.get_for_pair(pair_key, "wick_fraction_min", config.wick_fraction_min))

                if wick_fraction < wick_min:
                    self._log_rejection(pair_key, "WICK_TOO_SMALL_SELL", round(wick_fraction, 3),
                                        direction="SELL", hour_utc=hour_utc)
                    return None

                require_body = config.get_for_pair(pair_key, "require_opposing_body", config.require_opposing_body)
                if require_body and bar_close >= bar_open:  # needs bearish body
                    self._log_rejection(pair_key, "NOT_BEARISH_BODY_SELL", None,
                                        direction="SELL", hour_utc=hour_utc)
                    return None

                sell_signal = {
                    "direction": "SELL",
                    "level": pdh,
                    "wick_fraction": round(wick_fraction, 3),
                    "atr_pips": round(atr_pips, 1),
                }

            # ── Step 8b: Failed-low detection (BUY) ───────────────────────
            buy_signal = None
            if bar_low < pdl - buffer and bar_close > pdl:
                wick_tail = min(bar_open, bar_close) - bar_low
                wick_fraction = wick_tail / bar_range
                wick_min = float(config.get_for_pair(pair_key, "wick_fraction_min", config.wick_fraction_min))

                if wick_fraction < wick_min:
                    self._log_rejection(pair_key, "WICK_TOO_SMALL_BUY", round(wick_fraction, 3),
                                        direction="BUY", hour_utc=hour_utc)
                    return None

                require_body = config.get_for_pair(pair_key, "require_opposing_body", config.require_opposing_body)
                if require_body and bar_close <= bar_open:  # needs bullish body
                    self._log_rejection(pair_key, "NOT_BULLISH_BODY_BUY", None,
                                        direction="BUY", hour_utc=hour_utc)
                    return None

                buy_signal = {
                    "direction": "BUY",
                    "level": pdl,
                    "wick_fraction": round(wick_fraction, 3),
                    "atr_pips": round(atr_pips, 1),
                }

            # If both fire (extremely rare — same bar swept PDH and PDL), prefer SELL
            trigger = sell_signal or buy_signal
            if trigger is None:
                return None

            # ── Step 9: Compute SL/TP ─────────────────────────────────────
            direction   = trigger["direction"]
            entry_price = bar_close

            sl_buffer_price = float(config.get_for_pair(pair_key, "sl_buffer_pips", config.sl_buffer_pips)) * pip
            min_sl = float(config.get_for_pair(pair_key, "min_sl_pips", config.min_sl_pips)) * pip
            max_sl = float(config.get_for_pair(pair_key, "max_sl_pips", config.max_sl_pips)) * pip

            if direction == "SELL":
                raw_sl_dist = (bar_high + sl_buffer_price) - entry_price
            else:
                raw_sl_dist = entry_price - (bar_low - sl_buffer_price)

            raw_sl_dist = max(min_sl, min(max_sl, raw_sl_dist))
            sl_pips = raw_sl_dist / pip

            min_rr   = float(config.get_for_pair(pair_key, "min_rr_ratio", config.min_rr_ratio))
            max_tp_p = float(config.get_for_pair(pair_key, "max_tp_pips", config.max_tp_pips))
            tp_from_rr  = sl_pips * min_rr
            tp_from_atr = atr_pips * 1.5
            tp_pips = min(max_tp_p, max(tp_from_rr, tp_from_atr))

            # Check R:R (10% tolerance)
            if tp_pips / sl_pips < min_rr * 0.9:
                self._log_rejection(pair_key, "RR_TOO_LOW", round(tp_pips / sl_pips, 2),
                                    direction=direction, hour_utc=hour_utc)
                return None

            if direction == "SELL":
                stop_loss   = entry_price + raw_sl_dist
                take_profit = entry_price - (tp_pips * pip)
            else:
                stop_loss   = entry_price - raw_sl_dist
                take_profit = entry_price + (tp_pips * pip)

            # ── Step 10: Confidence ───────────────────────────────────────
            confidence = 0.65

            # Wick quality bonus
            if trigger["wick_fraction"] >= 0.50:
                confidence += 0.05

            # ATR in preferred band (10-22 pips)
            if 10.0 <= atr_pips <= 22.0:
                confidence += 0.05

            # Prime London / NY session hours
            if hour_utc in {7, 8, 9, 10, 13, 14, 15, 16}:
                confidence += 0.05

            conf_min = float(config.get_pair_min_confidence(pair_key))
            conf_max = float(config.get_pair_max_confidence(pair_key))
            confidence = max(conf_min, min(conf_max, confidence))

            # ── Step 11: Build and return signal ──────────────────────────
            signal_dict = {
                "signal":      direction,
                "signal_type": direction,
                "direction":   direction,
                "strategy":    self.strategy_name,
                "epic":        epic,
                "pair":        pair,
                "entry_price": round(entry_price, 6),
                "stop_loss":   round(stop_loss, 6),
                "take_profit": round(take_profit, 6),
                "risk_pips":   round(sl_pips, 1),
                "reward_pips": round(tp_pips, 1),
                "confidence":  round(confidence, 3),
                "confidence_score": round(confidence, 3),
                "entry_type":  "REVERSAL",
                "timeframe":   "1h",
                "monitor_only": config.is_pair_monitor_only(pair_key),
                "signal_timestamp": ts.isoformat(),
                "timestamp":   ts,
                "version":     "1.0.0",
                "strategy_indicators": {
                    "pdh": round(pdh, 6),
                    "pdl": round(pdl, 6),
                    "level": round(trigger["level"], 6),
                    "wick_fraction": trigger["wick_fraction"],
                    "atr_pips": trigger["atr_pips"],
                    "prev_day_bars": len(prev_day_bars),
                },
            }

            # Enrich with regime/ER/vol metrics
            try:
                from forex_scanner.core.strategies.helpers.smc_performance_metrics import enrich_signal_with_performance_metrics
                signal_dict = enrich_signal_with_performance_metrics(
                    signal_dict,
                    df_entry=None,
                    df_trigger=df_trigger,
                    df_htf=None,
                    epic=epic,
                    logger=self.logger,
                )
            except Exception:
                pass

            self._set_cooldown(pair_key, ts)
            self.logger.info(
                "[LIQUIDITY_TRAP] %s signal for %s: "
                "level=%.5f wick=%.2%% SL=%.1fp TP=%.1fp conf=%.2f",
                direction, epic,
                trigger["level"], trigger["wick_fraction"],
                sl_pips, tp_pips, confidence,
            )
            return signal_dict

        except Exception as exc:
            self.logger.error("[LIQUIDITY_TRAP] %s: detect_signal error: %s", pair or epic or "?", exc)
            return None


def create_liquidity_trap_strategy(
    config: Optional[LiquidityTrapConfig] = None,
    db_manager=None,
    logger=None,
    config_override: Optional[Dict[str, Any]] = None,
) -> LiquidityTrapStrategy:
    """Factory function for LiquidityTrapStrategy."""
    return LiquidityTrapStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager,
        config_override=config_override,
    )
