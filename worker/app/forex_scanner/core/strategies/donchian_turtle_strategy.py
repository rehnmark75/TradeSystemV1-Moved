#!/usr/bin/env python3
"""
Donchian Turtle strategy (S1 system, 1H bars).

Classic Turtle Trading rules applied to FX 1H bars:
  Entry:  close breaks above 20-bar Donchian high (LONG)
          or below 20-bar Donchian low (SHORT, disabled until 6-month review)
  Stop:   2×ATR14 from entry price
  TP:     200-pip safety cap — trailing stops exit the trade in practice
  Filter: long-only flag (long PF 1.21 vs short PF 1.04 over 4-year backtest)

Pairs: USDJPY (PF 1.37), USDCHF (PF 1.22), EURJPY (PF 1.18) — all monitor-only at launch.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

try:
    from ...services.donchian_turtle_config_service import (
        DonchianTurtleConfig,
        get_donchian_turtle_config,
    )
except ImportError:
    from forex_scanner.services.donchian_turtle_config_service import (
        DonchianTurtleConfig,
        get_donchian_turtle_config,
    )

from .strategy_registry import StrategyInterface, register_strategy

logger = logging.getLogger(__name__)


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


def _apply_config_override(config: DonchianTurtleConfig, overrides: Optional[Dict[str, Any]]) -> None:
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
            for pair_row in config._pair_overrides.values():
                if key in pair_row:
                    pair_row[key] = new_val
        except Exception as exc:
            logger.warning("DONCHIAN_TURTLE override failed for %s=%r: %s", key, value, exc)


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


@register_strategy("DONCHIAN_TURTLE")
class DonchianTurtleStrategy(StrategyInterface):
    """S1 Donchian channel breakout on 1H bars — trend-following, long-only initially."""

    @property
    def strategy_name(self) -> str:
        return "DONCHIAN_TURTLE"

    def get_required_timeframes(self) -> list:
        return ["1h"]

    def __init__(self, config_override: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._config_override = config_override
        self._last_signal_time: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # StrategyInterface
    # ------------------------------------------------------------------

    def detect_signal(
        self,
        df_trigger: pd.DataFrame,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        current_timestamp: Optional[datetime] = None,
        routing_context: Optional[Dict] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect a Donchian channel breakout on 1H bars.

        df_trigger is expected to be 1H OHLCV data (resampled from 5m).
        Returns a signal dict or None.
        """
        try:
            config = get_donchian_turtle_config()
            effective_override = config_override or self._config_override
            _apply_config_override(config, effective_override)

            if not config.is_pair_enabled(epic):
                logger.debug("[DONCHIAN_TURTLE] %s not enabled", epic)
                return None

            if current_timestamp is None:
                current_timestamp = datetime.now(timezone.utc)

            # Cooldown check
            cooldown_minutes = config.get_pair_cooldown_minutes(epic)
            last = self._last_signal_time.get(epic)
            if last is not None:
                elapsed = (current_timestamp - last).total_seconds() / 60
                if elapsed < cooldown_minutes:
                    logger.debug(
                        "[DONCHIAN_TURTLE] %s cooldown: %.0f / %d min",
                        pair, elapsed, cooldown_minutes,
                    )
                    return None

            entry_bars = config.get_pair_entry_bars(epic)
            exit_bars = config.get_pair_exit_bars(epic)
            atr_period = config.atr_period
            atr_mult = config.get_pair_atr_stop_multiplier(epic)
            pip = _pip_size(pair)

            min_bars = max(entry_bars, exit_bars, atr_period) + 5
            if df_trigger is None or len(df_trigger) < min_bars:
                logger.debug(
                    "[DONCHIAN_TURTLE] %s insufficient data (%d bars, need %d)",
                    pair, len(df_trigger) if df_trigger is not None else 0, min_bars,
                )
                return None

            # Use all rows up to and including the last completed candle.
            # iloc[-1] is the most recent bar (may be incomplete intra-hour);
            # entry logic reads the last closed bar via channel computed on prior bars.
            close = df_trigger["close"].iloc[-1]
            high_series = df_trigger["high"]
            low_series = df_trigger["low"]

            # 20-bar entry channel: high/low of bars [-entry_bars-1 : -1] (excludes current)
            entry_high = high_series.iloc[-(entry_bars + 1):-1].max()
            entry_low = low_series.iloc[-(entry_bars + 1):-1].min()

            # ATR for hard stop
            atr_series = _compute_atr(df_trigger, atr_period)
            atr_val = atr_series.iloc[-1]
            if pd.isna(atr_val) or atr_val <= 0:
                logger.debug("[DONCHIAN_TURTLE] %s ATR unavailable", pair)
                return None

            atr_pips = atr_val / pip
            sl_pips = round(atr_mult * atr_pips, 1)
            tp_pips = config.get_pair_fixed_take_profit(epic)  # safety cap; trailing exits first

            # Clamp SL to at least the config fallback (avoid absurdly tight stops on low-vol bars)
            fallback_sl = config.get_pair_fixed_stop_loss(epic)
            sl_pips = max(sl_pips, fallback_sl * 0.5)

            direction = None

            if close > entry_high:
                direction = "BUY"
                entry_price = close
                stop_price = entry_price - sl_pips * pip
                target_price = entry_price + tp_pips * pip

            elif not config.long_only and close < entry_low:
                direction = "SELL"
                entry_price = close
                stop_price = entry_price + sl_pips * pip
                target_price = entry_price - tp_pips * pip

            if direction is None:
                logger.debug(
                    "[DONCHIAN_TURTLE] %s no breakout (close=%.5f entry_high=%.5f entry_low=%.5f)",
                    pair, close, entry_high, entry_low,
                )
                return None

            monitor_only = config.is_pair_monitor_only(epic)
            confidence = self._compute_confidence(df_trigger, direction, entry_bars, atr_pips, pip)

            min_conf = config.get_pair_min_confidence(epic)
            max_conf = config.get_pair_max_confidence(epic)
            if not (min_conf <= confidence <= max_conf):
                logger.debug(
                    "[DONCHIAN_TURTLE] %s confidence %.2f outside [%.2f, %.2f]",
                    pair, confidence, min_conf, max_conf,
                )
                return None

            self._last_signal_time[epic] = current_timestamp

            signal = {
                "signal": direction,
                "strategy": "DONCHIAN_TURTLE",
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_pips": sl_pips,
                "reward_pips": tp_pips,
                "confidence": confidence,
                "monitor_only": monitor_only,
                "atr_pips": round(atr_pips, 1),
                "donchian_entry_high": round(entry_high, 5),
                "donchian_entry_low": round(entry_low, 5),
                "timeframe": "1h",
                "timestamp": current_timestamp.isoformat() if current_timestamp else None,
                "pair": pair,
                "epic": epic,
            }

            logger.info(
                "[DONCHIAN_TURTLE] %s %s signal: entry=%.5f SL=%.1f TP=%.1f ATR=%.1f pips "
                "conf=%.2f %s",
                pair, direction, entry_price, sl_pips, tp_pips, atr_pips, confidence,
                "(MONITOR)" if monitor_only else "(ACTIVE)",
            )
            return signal

        except Exception as exc:
            logger.error("[DONCHIAN_TURTLE] Error detecting signal for %s: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_bars: int,
        atr_pips: float,
        pip: float,
    ) -> float:
        """
        Simple confidence score (0–1):
        - Breakout strength relative to channel width
        - Recent candle body momentum alignment

        Confidence is used only as a soft gate against very weak breakouts.
        """
        try:
            close = df["close"].iloc[-1]
            high_series = df["high"]
            low_series = df["low"]

            entry_high = high_series.iloc[-(entry_bars + 1):-1].max()
            entry_low = low_series.iloc[-(entry_bars + 1):-1].min()
            channel_width_pips = (entry_high - entry_low) / pip

            if channel_width_pips <= 0:
                return 0.60

            if direction == "BUY":
                breakout_pips = (close - entry_high) / pip
            else:
                breakout_pips = (entry_low - close) / pip

            # Normalise: 0.5 base + up to 0.4 bonus for clean breakout (≥ 0.5× ATR)
            strength = min(breakout_pips / max(atr_pips * 0.5, 1.0), 1.0)
            confidence = 0.50 + 0.40 * strength

            # Momentum: last 3 candles aligned with direction adds +0.10
            bodies = (df["close"] - df["open"]).iloc[-4:-1]
            if direction == "BUY" and (bodies > 0).sum() >= 2:
                confidence += 0.05
            elif direction == "SELL" and (bodies < 0).sum() >= 2:
                confidence += 0.05

            return round(min(confidence, 0.95), 3)
        except Exception:
            return 0.60
