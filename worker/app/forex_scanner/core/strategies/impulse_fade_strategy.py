#!/usr/bin/env python3
"""
Impulse-fade strategy (5m primary, single-timeframe).

Fades large 5m candle bodies that appear during the late-US session window
(18:00–22:59 UTC). A candle whose body (in pips) exceeds N × ATR14 is
considered an impulse; the trade fades the move:
  - Bullish impulse → SHORT entry at next open
  - Bearish impulse → LONG entry at next open

Design notes
------------
- No HTF alignment: the edge is behavioural (late-session exhaustion after
  large moves) not trend-following, so HTF bias is not load-bearing.
- Inverted R:R (TP < SL) is compensated by the high WR requirement; the
  TradeValidator strategy_rr_overrides entry for IMPULSE_FADE permits this.
- ATR spike guard (max_atr_pips): blocks entries when volatility is so
  extreme (post-news) that the 2.2× threshold fires trivially.
- time_stop_candles metadata is included in the signal dict for manual
  handling; BacktestScanner does not yet consume it automatically.
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

try:
    from services.impulse_fade_config_service import (
        ImpulseFadeConfig,
        get_impulse_fade_config,
    )
except ImportError:
    from forex_scanner.services.impulse_fade_config_service import (
        ImpulseFadeConfig,
        get_impulse_fade_config,
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


def _apply_config_override(config: ImpulseFadeConfig, overrides: Optional[Dict[str, Any]]) -> None:
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
            logger.warning("IMPULSE_FADE override failed for %s=%r: %s", key, value, exc)


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """True-range ATR using OHLCV columns (no TA library dependency)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


@register_strategy("IMPULSE_FADE")
class ImpulseFadeStrategy(StrategyInterface):
    """Fade large-body 5m candles in the late-US session."""

    def __init__(
        self,
        config: Optional[ImpulseFadeConfig] = None,
        logger=None,
        db_manager=None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self._config_override = config_override

        self.config = copy.deepcopy(config or get_impulse_fade_config())
        _apply_config_override(self.config, self._config_override)

        self._cooldowns: Dict[str, datetime] = {}
        self._current_timestamp: Optional[datetime] = None

        self._rej_mgr = None
        if db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager("IMPULSE_FADE", db_manager)
            except Exception:
                pass

        self.logger.info(
            "[IMPULSE_FADE] v%s initialized | session=%d-%dUTC "
            "atr_mult=%.1f max_atr=%.1f SL/TP=%.1f/%.1f cooldown=%dm",
            self.config.version,
            self.config.session_start_hour,
            self.config.session_end_hour,
            self.config.atr_body_multiplier,
            self.config.max_atr_pips,
            self.config.fixed_stop_loss_pips,
            self.config.fixed_take_profit_pips,
            self.config.signal_cooldown_minutes,
        )

    # ------------------------------------------------------------------
    # StrategyInterface required methods
    # ------------------------------------------------------------------

    @property
    def strategy_name(self) -> str:
        return "IMPULSE_FADE"

    def get_required_timeframes(self):
        return ["5m"]

    def reset_cooldowns(self) -> None:
        """Called by BacktestScanner between pairs."""
        self._cooldowns.clear()

    def flush_rejections(self) -> None:
        if self._rej_mgr is not None:
            self._rej_mgr.flush()

    def get_config(self) -> ImpulseFadeConfig:
        return self.config

    def _reject(
        self,
        stage: str,
        reason: str,
        epic: str,
        pair: str,
        hour_utc: Optional[int] = None,
        direction: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        scan_timestamp: Optional[datetime] = None,
    ) -> None:
        self.logger.info("[IMPULSE_FADE] %s ❌ %s: %s", pair or epic, stage, reason)
        if self._rej_mgr is not None:
            self._rej_mgr.reject(
                stage=stage,
                reason=reason,
                epic=epic,
                pair=pair,
                direction=direction,
                hour_utc=hour_utc,
                scan_timestamp=scan_timestamp or self._current_timestamp,
                details=details,
            )

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    def detect_signal(
        self,
        df_trigger: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        epic: Optional[str] = None,
        pair: Optional[str] = None,
        spread_pips: float = 1.5,
        current_timestamp: Optional[datetime] = None,
        routing_context: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate the most-recent completed 5m candle for an impulse-fade setup."""
        try:
            cfg = self.config
            pip = _pip_size(pair or epic or "")
            pair_key = epic or pair or "UNKNOWN"
            self._current_timestamp = current_timestamp

            # Need enough rows for ATR + one completed candle
            min_rows = cfg.atr_period + 5
            if df_trigger is None or len(df_trigger) < min_rows:
                self.logger.debug("[IMPULSE_FADE] %s: insufficient data (%d rows)", pair_key, len(df_trigger) if df_trigger is not None else 0)
                self._reject("INSUFFICIENT_DATA", "not enough rows", pair_key, pair or pair_key)
                return None

            # Pair-level enable check
            if not cfg.is_pair_enabled(pair_key):
                self.logger.debug("[IMPULSE_FADE] %s: pair not enabled", pair_key)
                return None

            # Use the last completed candle (index -2 when live; index -1 acceptable in backtest)
            # Use -1 here; BacktestScanner advances timestamp so the "last" row is always completed.
            candle = df_trigger.iloc[-1]

            # Determine evaluation timestamp
            if current_timestamp is not None:
                eval_ts = current_timestamp
                if not isinstance(eval_ts, datetime):
                    eval_ts = pd.Timestamp(eval_ts).to_pydatetime()
                if eval_ts.tzinfo is None:
                    eval_ts = eval_ts.replace(tzinfo=timezone.utc)
            else:
                # Derive from candle start_time column (DataFetcher convention)
                if "start_time" in df_trigger.columns:
                    raw = df_trigger["start_time"].iloc[-1]
                    eval_ts = pd.Timestamp(raw).to_pydatetime()
                    if eval_ts.tzinfo is None:
                        eval_ts = eval_ts.replace(tzinfo=timezone.utc)
                else:
                    eval_ts = datetime.now(timezone.utc)

            self._current_timestamp = eval_ts
            utc_hour = eval_ts.hour

            # ── Session gate ─────────────────────────────────────────────
            start_h = cfg.get_pair_session_start_hour(pair_key)
            end_h = cfg.get_pair_session_end_hour(pair_key)
            in_window_1 = start_h <= utc_hour <= end_h
            start_h_2 = cfg.get_pair_session_start_hour_2(pair_key)
            end_h_2 = cfg.get_pair_session_end_hour_2(pair_key)
            in_window_2 = (start_h_2 is not None and end_h_2 is not None
                           and start_h_2 <= utc_hour <= end_h_2)
            in_session = in_window_1 or in_window_2
            if not in_session:
                self.logger.debug("[IMPULSE_FADE] %s: outside session (hour=%d)", pair_key, utc_hour)
                win2_str = f" or {start_h_2}-{end_h_2}" if start_h_2 is not None else ""
                self._reject("SESSION", f"hour={utc_hour} outside {start_h}-{end_h}{win2_str}",
                             pair_key, pair or pair_key, hour_utc=utc_hour,
                             details={"hour": utc_hour, "session_start": start_h, "session_end": end_h,
                                      "session_start_2": start_h_2, "session_end_2": end_h_2})
                return None

            # ── Cooldown check ────────────────────────────────────────────
            cooldown_mins = cfg.get_pair_cooldown_minutes(pair_key)
            if pair_key in self._cooldowns:
                elapsed = (eval_ts - self._cooldowns[pair_key]).total_seconds() / 60
                if elapsed < cooldown_mins:
                    self.logger.debug("[IMPULSE_FADE] %s: in cooldown (%.1f / %d min)", pair_key, elapsed, cooldown_mins)
                    self._reject("COOLDOWN", f"elapsed={elapsed:.1f}m < {cooldown_mins}m",
                                 pair_key, pair or pair_key, hour_utc=utc_hour,
                                 details={"elapsed_min": round(elapsed, 1), "cooldown_min": cooldown_mins})
                    return None

            # ── ATR calculation ───────────────────────────────────────────
            atr_series = _compute_atr(df_trigger, cfg.atr_period)
            atr_price = atr_series.iloc[-1]
            if pd.isna(atr_price) or atr_price <= 0:
                self.logger.debug("[IMPULSE_FADE] %s: ATR unavailable", pair_key)
                self._reject("ATR_UNAVAILABLE", "ATR is NaN or zero", pair_key, pair or pair_key,
                             hour_utc=utc_hour)
                return None

            atr_pips = atr_price / pip

            # ATR spike guard: block if market is in extreme post-news state
            max_atr = cfg.get_pair_max_atr_pips(pair_key)
            if atr_pips > max_atr:
                self.logger.debug("[IMPULSE_FADE] %s: ATR spike blocked (%.1f > %.1f pips)", pair_key, atr_pips, max_atr)
                self._reject("ATR_SPIKE", f"atr={atr_pips:.1f} > max={max_atr:.1f} pips",
                             pair_key, pair or pair_key, hour_utc=utc_hour,
                             details={"atr_pips": round(atr_pips, 2), "max_atr_pips": max_atr})
                return None

            # ── Impulse body check ────────────────────────────────────────
            open_p = float(candle["open"])
            close_p = float(candle["close"])
            body_price = abs(close_p - open_p)
            body_pips = body_price / pip

            threshold_mult = cfg.get_pair_atr_body_multiplier(pair_key)
            threshold_pips = threshold_mult * atr_pips

            if body_pips < threshold_pips:
                self.logger.debug(
                    "[IMPULSE_FADE] %s: body too small (%.1f < %.1f pips)", pair_key, body_pips, threshold_pips
                )
                self._reject("BODY_TOO_SMALL", f"body={body_pips:.1f} < threshold={threshold_pips:.1f} pips",
                             pair_key, pair or pair_key, hour_utc=utc_hour,
                             details={"body_pips": round(body_pips, 2), "threshold_pips": round(threshold_pips, 2),
                                      "atr_pips": round(atr_pips, 2), "atr_mult": threshold_mult})
                return None

            # ── Direction: fade the impulse ───────────────────────────────
            is_bullish = close_p > open_p
            direction = "SELL" if is_bullish else "BUY"

            entry_price = close_p

            sl_pips = cfg.get_pair_fixed_stop_loss(pair_key)
            tp_pips = cfg.get_pair_fixed_take_profit(pair_key)

            if direction == "SELL":
                stop_loss = entry_price + sl_pips * pip
                take_profit = entry_price - tp_pips * pip
            else:
                stop_loss = entry_price - sl_pips * pip
                take_profit = entry_price + tp_pips * pip

            # ── Confidence scoring ────────────────────────────────────────
            confidence = self._score_confidence(
                body_pips=body_pips,
                threshold_pips=threshold_pips,
                atr_pips=atr_pips,
                pair_key=pair_key,
            )

            min_conf = cfg.get_pair_min_confidence(pair_key)
            max_conf = cfg.get_pair_max_confidence(pair_key)
            if confidence < min_conf:
                self.logger.debug("[IMPULSE_FADE] %s: confidence %.2f < min %.2f", pair_key, confidence, min_conf)
                self._reject("LOW_CONFIDENCE", f"confidence={confidence:.3f} < min={min_conf:.3f}",
                             pair_key, pair or pair_key, hour_utc=utc_hour,
                             details={"confidence": round(confidence, 3), "min_confidence": min_conf,
                                      "body_pips": round(body_pips, 2), "atr_pips": round(atr_pips, 2)})
                return None
            confidence = min(confidence, max_conf)

            # ── Update cooldown & build signal ────────────────────────────
            self._cooldowns[pair_key] = eval_ts

            candle_time_str = str(eval_ts)

            signal = {
                "strategy": "IMPULSE_FADE",
                "signal": direction,
                "signal_type": direction,       # BacktestTrailingEngine reads signal_type for is_long
                "signal_timestamp": candle_time_str,
                "entry_price": entry_price,
                "stop_loss": round(stop_loss, 5),
                "take_profit": round(take_profit, 5),
                "risk_pips": sl_pips,
                "reward_pips": tp_pips,
                "confidence": round(confidence, 3),
                "confidence_score": round(confidence, 3),
                "timeframe": "5m",
                "pair": pair or pair_key,
                "epic": epic or pair_key,
                # Metadata for analysis / future time-stop support
                "body_pips": round(body_pips, 2),
                "atr_pips": round(atr_pips, 2),
                "body_multiplier": round(body_pips / atr_pips, 2),
                "time_stop_candles": cfg.time_stop_candles,
                "session_hour": utc_hour,
                "candle_time": candle_time_str,
                "monitor_only": cfg.is_pair_monitor_only(pair_key),
            }

            self.logger.info(
                "[IMPULSE_FADE] %s: %s signal | body=%.1f pips (%.1f× ATR) | "
                "entry=%.5f SL=%.5f TP=%.5f conf=%.2f",
                pair_key, direction, body_pips, body_pips / atr_pips,
                entry_price, stop_loss, take_profit, confidence,
            )

            try:
                from forex_scanner.core.strategies.helpers.smc_performance_metrics import enrich_signal_with_performance_metrics
                signal = enrich_signal_with_performance_metrics(
                    signal, df_entry=None, df_trigger=df_trigger, df_htf=df_4h, epic=epic or pair_key, logger=self.logger
                )
            except Exception as _pm_exc:
                self.logger.warning("[IMPULSE_FADE] Performance metrics failed: %s", _pm_exc)

            return signal

        except Exception as exc:
            self.logger.error("[IMPULSE_FADE] %s: detect_signal error: %s", pair or epic or "?", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_confidence(
        self,
        body_pips: float,
        threshold_pips: float,
        atr_pips: float,
        pair_key: str,
    ) -> float:
        """Score in [0, 1]. Base 0.55; bonus for body size relative to threshold."""
        base = 0.55

        # Excess ratio: how much larger the body is above the threshold
        # 0× excess → 0 bonus; 1× excess → +0.10; 2× → +0.15 (capped)
        excess_ratio = (body_pips - threshold_pips) / threshold_pips
        body_bonus = min(0.15, excess_ratio * 0.10)

        # Small session-timing bonus: signals closer to peak window (20-21 UTC) score slightly higher
        # (not applied here — kept for future per-config tuning via session_peak_bonus param)

        score = base + body_bonus
        return round(max(0.0, min(1.0, score)), 4)
