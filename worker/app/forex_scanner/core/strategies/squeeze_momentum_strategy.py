#!/usr/bin/env python3
"""
Squeeze Momentum strategy.

LazyBear-style BB/KC squeeze release on 15m bars with 1H EMA50 alignment,
ADX expansion, ATR-normalized momentum slope, and ATR SL/TP.
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from services.squeeze_momentum_config_service import (
        SqueezeMomentumConfig,
        get_squeeze_momentum_config,
    )
except ImportError:
    from forex_scanner.services.squeeze_momentum_config_service import (
        SqueezeMomentumConfig,
        get_squeeze_momentum_config,
    )

try:
    from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from ...alerts.strategy_rejection_manager import StrategyRejectionManager
    except ImportError:
        StrategyRejectionManager = None  # type: ignore[assignment,misc]

from .strategy_registry import StrategyInterface, register_strategy

logger = logging.getLogger(__name__)


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


def _coerce_override(current: Any, value: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    return value


def _apply_config_override(config: SqueezeMomentumConfig, overrides: Optional[Dict[str, Any]]) -> SqueezeMomentumConfig:
    if not overrides:
        return config
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        if hasattr(config, key):
            try:
                setattr(config, key, _coerce_override(getattr(config, key), value))
            except Exception as exc:
                logger.warning("[SQUEEZE_MOMENTUM] override failed for %s=%r: %s", key, value, exc)
        for row in config._pair_overrides.values():
            if key in row and row[key] is not None:
                try:
                    row[key] = _coerce_override(row[key], value)
                except Exception:
                    pass
    return config


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(df).rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)

    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _linreg_endpoint(values: np.ndarray) -> float:
    if len(values) == 0 or np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values.astype(float), 1)
    return float(intercept + slope * (len(values) - 1))


def _add_squeeze_columns(df: pd.DataFrame, cfg: SqueezeMomentumConfig) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    basis = close.rolling(cfg.bb_length).mean()
    dev = cfg.bb_mult * close.rolling(cfg.bb_length).std()
    out["sqz_bb_upper"] = basis + dev
    out["sqz_bb_lower"] = basis - dev

    kc_ma = close.rolling(cfg.kc_length).mean()
    range_series = _true_range(out) if cfg.use_true_range else (high - low)
    range_ma = range_series.rolling(cfg.kc_length).mean()
    out["sqz_kc_upper"] = kc_ma + range_ma * cfg.kc_mult
    out["sqz_kc_lower"] = kc_ma - range_ma * cfg.kc_mult

    out["sqz_on"] = (out["sqz_bb_lower"] > out["sqz_kc_lower"]) & (out["sqz_bb_upper"] < out["sqz_kc_upper"])
    out["sqz_off"] = (out["sqz_bb_lower"] < out["sqz_kc_lower"]) & (out["sqz_bb_upper"] > out["sqz_kc_upper"])
    out["sqz_no"] = ~(out["sqz_on"] | out["sqz_off"])

    highest_high = high.rolling(cfg.kc_length).max()
    lowest_low = low.rolling(cfg.kc_length).min()
    mean_extreme = (highest_high + lowest_low) / 2.0
    mean_source = (mean_extreme + close.rolling(cfg.kc_length).mean()) / 2.0
    linreg_source = close - mean_source
    out["sqz_momentum"] = linreg_source.rolling(cfg.kc_length).apply(_linreg_endpoint, raw=True)
    out["sqz_momentum_slope"] = out["sqz_momentum"].diff()
    out["atr"] = _atr(out, cfg.atr_period)
    out["adx"] = _adx(out, cfg.adx_period)
    return out


@register_strategy("SQUEEZE_MOMENTUM")
class SqueezeMomentumStrategy(StrategyInterface):
    """15m squeeze release with 1H trend and expansion filters."""

    @property
    def strategy_name(self) -> str:
        return "SQUEEZE_MOMENTUM"

    def get_required_timeframes(self) -> list:
        cfg = get_squeeze_momentum_config()
        return [cfg.entry_timeframe, cfg.htf_timeframe]

    def __init__(self, config_override: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._config_override = config_override
        self._last_signal_time: Dict[str, datetime] = {}
        self.db_manager = kwargs.get("db_manager")
        self._rej_mgr = None
        if self.db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager("SQUEEZE_MOMENTUM", self.db_manager)
            except Exception as exc:
                logger.warning("[SQUEEZE_MOMENTUM] rejection manager init failed: %s", exc)

    def reset_cooldowns(self) -> None:
        self._last_signal_time.clear()

    def flush_rejections(self) -> None:
        if self._rej_mgr is not None:
            self._rej_mgr.flush()

    def _reject(
        self,
        stage: str,
        reason: str,
        epic: str,
        pair: str,
        direction: Optional[str] = None,
        hour_utc: Optional[int] = None,
        scan_timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger.info("[SQUEEZE_MOMENTUM] %s ❌ %s: %s", pair or epic, stage, reason)
        if self._rej_mgr is not None:
            self._rej_mgr.reject(
                stage=stage,
                reason=reason,
                epic=epic,
                pair=pair,
                direction=direction,
                hour_utc=hour_utc,
                scan_timestamp=scan_timestamp,
                details=details,
            )

    def detect_signal(
        self,
        df_trigger: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        epic: str = "",
        pair: str = "",
        spread_pips: float = 1.5,
        current_timestamp: Optional[datetime] = None,
        config_override: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        cfg = _apply_config_override(get_squeeze_momentum_config(), config_override or self._config_override)

        if not cfg.is_pair_enabled(epic):
            return None

        if current_timestamp is None:
            current_timestamp = datetime.now(timezone.utc)
        if current_timestamp.tzinfo is None:
            current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)
        hour_utc = current_timestamp.hour

        if cfg.block_asian_session and (hour_utc >= 21 or hour_utc < 7):
            self._reject("SESSION", f"asian session blocked hour={hour_utc}", epic, pair, hour_utc=hour_utc, scan_timestamp=current_timestamp)
            return None

        cooldown_minutes = cfg.get_pair_cooldown_minutes(epic)
        last_signal = self._last_signal_time.get(epic)
        if last_signal is not None:
            elapsed = (current_timestamp - last_signal).total_seconds() / 60.0
            if elapsed < cooldown_minutes:
                self._reject(
                    "COOLDOWN",
                    f"elapsed={elapsed:.0f}m < {cooldown_minutes}m",
                    epic,
                    pair,
                    hour_utc=hour_utc,
                    scan_timestamp=current_timestamp,
                    details={"elapsed_minutes": round(elapsed, 1), "cooldown_minutes": cooldown_minutes},
                )
                return None

        min_bars = max(cfg.bb_length, cfg.kc_length, cfg.atr_period, cfg.adx_period) + cfg.squeeze_lookback_bars + 8
        if df_trigger is None or len(df_trigger) < min_bars:
            n = len(df_trigger) if df_trigger is not None else 0
            self._reject(
                "INSUFFICIENT_DATA",
                f"bars={n} < required={min_bars}",
                epic,
                pair,
                hour_utc=hour_utc,
                scan_timestamp=current_timestamp,
                details={"bars_available": n, "bars_required": min_bars},
            )
            return None

        if df_htf is None or len(df_htf) < cfg.htf_ema_period + 5:
            self._reject("INSUFFICIENT_HTF_DATA", "not enough HTF bars", epic, pair, hour_utc=hour_utc, scan_timestamp=current_timestamp)
            return None

        df = _add_squeeze_columns(df_trigger, cfg)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        if pd.isna(latest["sqz_momentum"]) or pd.isna(latest["atr"]) or latest["atr"] <= 0:
            self._reject("INDICATORS", "squeeze/ATR unavailable", epic, pair, hour_utc=hour_utc, scan_timestamp=current_timestamp)
            return None

        prior_window = df["sqz_on"].iloc[-(cfg.squeeze_lookback_bars + 1):-1]
        squeeze_count = int(prior_window.sum())
        release_ok = bool(latest["sqz_off"]) and (not cfg.require_release_bar or bool(prev["sqz_on"]))
        if squeeze_count < cfg.squeeze_min_bars or not release_ok:
            self._reject(
                "NO_SQUEEZE_RELEASE",
                f"squeeze_count={squeeze_count}, sqz_off={bool(latest['sqz_off'])}, prev_on={bool(prev['sqz_on'])}",
                epic,
                pair,
                hour_utc=hour_utc,
                scan_timestamp=current_timestamp,
                details={"squeeze_count": squeeze_count, "squeeze_min_bars": cfg.squeeze_min_bars},
            )
            return None

        momentum = float(latest["sqz_momentum"])
        slope = float(latest["sqz_momentum_slope"])
        atr_value = float(latest["atr"])
        slope_atr = abs(slope) / atr_value

        direction: Optional[str] = None
        if momentum > 0 and slope > 0:
            direction = "BUY"
        elif momentum < 0 and slope < 0:
            direction = "SELL"

        if direction is None:
            self._reject(
                "MOMENTUM_DIRECTION",
                f"momentum={momentum:.6f}, slope={slope:.6f}",
                epic,
                pair,
                hour_utc=hour_utc,
                scan_timestamp=current_timestamp,
            )
            return None

        if slope_atr < cfg.min_momentum_slope_atr:
            self._reject(
                "MOMENTUM_STRENGTH",
                f"slope_atr={slope_atr:.3f} < {cfg.min_momentum_slope_atr}",
                epic,
                pair,
                direction=direction,
                hour_utc=hour_utc,
                scan_timestamp=current_timestamp,
            )
            return None

        adx_value = float(latest["adx"]) if pd.notna(latest["adx"]) else None
        prev_adx = float(prev["adx"]) if pd.notna(prev["adx"]) else None
        adx_rising = adx_value is not None and prev_adx is not None and adx_value > prev_adx
        if adx_value is None or (adx_value < cfg.adx_min and (cfg.require_adx_rising and not adx_rising)):
            self._reject(
                "ADX_EXPANSION",
                f"adx={adx_value}, prev_adx={prev_adx}",
                epic,
                pair,
                direction=direction,
                hour_utc=hour_utc,
                scan_timestamp=current_timestamp,
            )
            return None

        htf_close = df_htf["close"].astype(float)
        htf_ema = htf_close.ewm(span=cfg.htf_ema_period, adjust=False).mean().iloc[-1]
        htf_last_close = htf_close.iloc[-1]
        if direction == "BUY" and htf_last_close <= htf_ema:
            self._reject("HTF_TREND", "BUY blocked: HTF close <= EMA", epic, pair, direction=direction, hour_utc=hour_utc, scan_timestamp=current_timestamp)
            return None
        if direction == "SELL" and htf_last_close >= htf_ema:
            self._reject("HTF_TREND", "SELL blocked: HTF close >= EMA", epic, pair, direction=direction, hour_utc=hour_utc, scan_timestamp=current_timestamp)
            return None

        pip = _pip_size(pair)
        stop_loss_pips = max(1.0, round((atr_value * cfg.stop_atr_multiplier) / pip, 1))
        take_profit_pips = max(1.0, round((atr_value * cfg.take_profit_atr_multiplier) / pip, 1))
        latest_close = float(latest["close"])
        confidence_ratio = min(1.0, max(0.0, (squeeze_count / max(cfg.squeeze_lookback_bars, 1)) * 0.45 + min(slope_atr / 0.12, 1.0) * 0.35 + min((adx_value or 0) / 35.0, 1.0) * 0.20))
        confidence = round(cfg.min_confidence + confidence_ratio * (cfg.max_confidence - cfg.min_confidence), 3)

        signal = {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair,
            "entry_price": latest_close,
            "stop_loss_pips": stop_loss_pips,
            "take_profit_pips": take_profit_pips,
            "risk_pips": stop_loss_pips,
            "reward_pips": take_profit_pips,
            "confidence_score": confidence,
            "confidence": confidence,
            "signal_timestamp": current_timestamp.isoformat(),
            "timestamp": current_timestamp,
            "version": cfg.version,
            "monitor_only": cfg.is_pair_monitor_only(epic),
            "adx": adx_value,
            "market_regime": "expansion" if latest["sqz_off"] else "compression",
            "regime": "expansion" if latest["sqz_off"] else "compression",
            "strategy_indicators": {
                "squeeze_count": squeeze_count,
                "sqz_on": bool(latest["sqz_on"]),
                "sqz_off": bool(latest["sqz_off"]),
                "momentum": momentum,
                "momentum_slope": slope,
                "momentum_slope_atr": slope_atr,
                "atr": atr_value,
                "adx": adx_value,
                "adx_rising": adx_rising,
                "htf_ema": float(htf_ema),
                "htf_close": float(htf_last_close),
                "bb_length": cfg.bb_length,
                "bb_mult": cfg.bb_mult,
                "kc_length": cfg.kc_length,
                "kc_mult": cfg.kc_mult,
            },
        }

        self._last_signal_time[epic] = current_timestamp
        logger.info(
            "[SQUEEZE_MOMENTUM] ✅ %s %s @ %.5f sqz=%s/%s slope_atr=%.3f adx=%.1f conf=%.2f",
            direction,
            epic,
            latest_close,
            squeeze_count,
            cfg.squeeze_lookback_bars,
            slope_atr,
            adx_value or 0.0,
            confidence,
        )
        return signal
