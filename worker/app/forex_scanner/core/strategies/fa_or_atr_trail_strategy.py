#!/usr/bin/env python3
"""FA_OR_ATR_TRAIL strategy.

Failed-auction + opening-range entries with ATR stop/target and tight ATR
trailing. This is the production-facing implementation of the research variant
derived from the Patrick Nill/PickMyTrade public-rule approximation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .strategy_registry import StrategyInterface, register_strategy

try:
    from services.fa_or_atr_trail_config_service import FAORATRTrailConfigService
except ImportError:
    from forex_scanner.services.fa_or_atr_trail_config_service import FAORATRTrailConfigService

try:
    from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from ...alerts.strategy_rejection_manager import StrategyRejectionManager
    except Exception:
        StrategyRejectionManager = None  # type: ignore[assignment,misc]


STRATEGY_NAME = "FA_OR_ATR_TRAIL"
VERSION = "0.2.0"

USDJPY_ATR_FLOOR_PIPS = 8.7


def _pip_size(epic: str) -> float:
    return 0.01 if "JPY" in (epic or "").upper() else 0.0001


def _wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def _session_label(hour: int) -> str:
    if 23 <= hour or hour < 7:
        return "asian"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 16:
        return "overlap"
    if 16 <= hour < 21:
        return "newyork"
    return "off_hours"


def _cfg_value(cfg: Any, epic: str, key: str, fallback: Any, overrides: Optional[Dict[str, Any]] = None) -> Any:
    if overrides and key in overrides:
        return overrides[key]
    if cfg is not None:
        return cfg.get_for_pair(epic, key, getattr(cfg, key, fallback))
    return fallback


@register_strategy(STRATEGY_NAME)
class FAORATRTrailStrategy(StrategyInterface):
    """Failed auction + opening range strategy with ATR trail exits."""

    def __init__(
        self,
        config=None,
        logger: Optional[logging.Logger] = None,
        db_manager=None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.config_override = config_override or {}

        self.session_start_hour = int(self.config_override.get("fa_or_session_start_hour", 8))
        self.session_end_hour = int(self.config_override.get("fa_or_session_end_hour", 20))
        self.adx_min = float(self.config_override.get("fa_or_adx_min", 18.0))
        self.min_slope_pips = float(self.config_override.get("fa_or_min_slope_pips", 0.3))
        self.max_vwap_atr = float(self.config_override.get("fa_or_max_vwap_atr", 3.0))
        self.opening_range_bars = int(self.config_override.get("fa_or_opening_range_bars", 6))
        self.value_area_std = float(self.config_override.get("fa_or_value_area_std", 0.70))
        self.vp_lookback = int(self.config_override.get("fa_or_vp_lookback", 15))
        self.cooldown_bars = int(self.config_override.get("fa_or_cooldown_bars", 5))

        self.sl_atr = float(self.config_override.get("fa_or_sl_atr", 1.2))
        self.tp_atr = float(self.config_override.get("fa_or_tp_atr", 2.0))
        self.trail_trigger_atr = float(self.config_override.get("fa_or_trail_trigger_atr", 0.25))
        self.trail_distance_atr = float(self.config_override.get("fa_or_trail_distance_atr", 0.10))

        self.atr_period = int(self.config_override.get("fa_or_atr_period", 14))
        self.adx_period = int(self.config_override.get("fa_or_adx_period", 14))
        self.rsi_period = int(self.config_override.get("fa_or_rsi_period", 14))
        self.htf_ema_period = int(self.config_override.get("fa_or_htf_ema_period", 50))
        self.min_htf_margin_atr = float(self.config_override.get("fa_or_min_htf_margin_atr", 1.0))

        self._last_signal_idx: Dict[str, int] = {}
        self._live_config_service = FAORATRTrailConfigService.get_instance()
        self._current_timestamp: Optional[datetime] = None
        self._pending_rejections: List[Dict[str, Any]] = []
        self._rej_mgr = None
        if db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager(STRATEGY_NAME, db_manager)
            except Exception:
                self._rej_mgr = None

        self.logger.info(
            "[FA_OR_ATR_TRAIL] initialized | session=%02d-%02d UTC ADX>=%.1f "
            "SL=%.2fATR TP=%.2fATR trail=%.2f/%.2fATR htf_margin>=%.1fATR",
            self.session_start_hour,
            self.session_end_hour,
            self.adx_min,
            self.sl_atr,
            self.tp_atr,
            self.trail_trigger_atr,
            self.trail_distance_atr,
            self.min_htf_margin_atr,
        )

    @property
    def strategy_name(self) -> str:
        return STRATEGY_NAME

    def get_required_timeframes(self) -> List[str]:
        return ["15m"]

    def reset_cooldowns(self) -> None:
        self._last_signal_idx.clear()

    def flush_rejections(self) -> None:
        if self._rej_mgr is not None:
            self._rej_mgr.flush()
        self._pending_rejections.clear()

    def detect_signal(
        self,
        df_trigger: pd.DataFrame = None,
        epic: str = "",
        pair: str = "",
        current_timestamp: datetime = None,
        spread_pips: float = 1.5,
        **kwargs,
    ) -> Optional[Dict]:
        cfg = None
        if not self.config_override.get("fa_or_ignore_db_pair_config", False):
            cfg = self._live_config_service.get_config()
            if not cfg.is_pair_enabled(epic):
                self._log_rejection(epic, pair, "PAIR_DISABLED", "pair is disabled in fa_or_atr_trail_pair_overrides")
                return None

        session_start_hour = int(_cfg_value(cfg, epic, "fa_or_session_start_hour", self.session_start_hour, self.config_override))
        session_end_hour = int(_cfg_value(cfg, epic, "fa_or_session_end_hour", self.session_end_hour, self.config_override))
        adx_min = float(_cfg_value(cfg, epic, "fa_or_adx_min", self.adx_min, self.config_override))
        max_vwap_atr = float(_cfg_value(cfg, epic, "fa_or_max_vwap_atr", self.max_vwap_atr, self.config_override))
        cooldown_bars = int(_cfg_value(cfg, epic, "fa_or_cooldown_bars", self.cooldown_bars, self.config_override))
        sl_atr = float(_cfg_value(cfg, epic, "fa_or_sl_atr", self.sl_atr, self.config_override))
        tp_atr = float(_cfg_value(cfg, epic, "fa_or_tp_atr", self.tp_atr, self.config_override))
        trail_trigger_atr = float(_cfg_value(cfg, epic, "fa_or_trail_trigger_atr", self.trail_trigger_atr, self.config_override))
        trail_distance_atr = float(_cfg_value(cfg, epic, "fa_or_trail_distance_atr", self.trail_distance_atr, self.config_override))
        usd_jpy_atr_floor_pips = float(_cfg_value(cfg, epic, "fa_or_usdjpy_atr_floor_pips", USDJPY_ATR_FLOOR_PIPS, self.config_override))
        if df_trigger is None or len(df_trigger) < 120:
            self._log_rejection(epic, pair, "INSUFFICIENT_DATA", f"need 120 bars, got {0 if df_trigger is None else len(df_trigger)}")
            return None

        df = self._normalize_df(df_trigger)
        if len(df) < 120:
            self._log_rejection(epic, pair, "INSUFFICIENT_DATA", f"need 120 normalized bars, got {len(df)}")
            return None

        df = self._add_indicators(df, epic)
        idx = len(df) - 1
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        ts = self._row_timestamp(row, current_timestamp)
        self._current_timestamp = ts
        hour = ts.hour

        if not (session_start_hour <= hour < session_end_hour):
            self._log_rejection(epic, pair, "SESSION", f"hour={hour} outside {session_start_hour}-{session_end_hour} UTC", hour_utc=hour)
            return None
        if pd.isna(row["atr"]) or pd.isna(row["adx"]) or pd.isna(row["ema50_4h"]):
            self._log_rejection(epic, pair, "INDICATOR_UNAVAILABLE", "ATR, ADX, or 4H EMA unavailable", hour_utc=hour)
            return None
        if row["adx"] < adx_min:
            self._log_rejection(
                epic,
                pair,
                "ADX",
                f"{float(row['adx']):.2f} < {adx_min:.2f}",
                hour_utc=hour,
                details={"adx": float(row["adx"]), "adx_min": adx_min},
            )
            return None
        if abs(row["close"] - row["vwap"]) > max_vwap_atr * row["atr"]:
            self._log_rejection(
                epic,
                pair,
                "VWAP_DISTANCE",
                "close too far from VWAP",
                hour_utc=hour,
                details={
                    "distance_atr": float(abs(row["close"] - row["vwap"]) / row["atr"]) if row["atr"] else None,
                    "max_vwap_atr": max_vwap_atr,
                },
            )
            return None

        last_idx = self._last_signal_idx.get(epic)
        if last_idx is not None and idx - last_idx <= cooldown_bars:
            self._log_rejection(epic, pair, "COOLDOWN", f"{idx - last_idx} bars since last signal", hour_utc=hour)
            return None

        direction, model = self._signal_model(df, row, prev)
        if direction is None:
            self._log_rejection(epic, pair, "NO_SETUP", "no failed-auction or opening-range setup", hour_utc=hour)
            return None

        pip = _pip_size(epic)
        atr_pips = float(row["atr"] / pip)
        if epic == "CS.D.USDJPY.MINI.IP" and atr_pips < usd_jpy_atr_floor_pips:
            self._log_rejection(
                epic,
                pair,
                "ATR_FLOOR",
                f"{atr_pips:.2f} pips < {usd_jpy_atr_floor_pips:.2f} pips",
                direction=direction,
                hour_utc=hour,
                details={"atr_pips": atr_pips, "floor_pips": usd_jpy_atr_floor_pips},
            )
            return None

        entry = float(row["close"])
        risk_pips = max(0.1, atr_pips * sl_atr)
        reward_pips = max(0.1, atr_pips * tp_atr)
        signal_type = "buy" if direction == "BUY" else "sell"

        if direction == "BUY":
            stop_loss = entry - risk_pips * pip
            take_profit = entry + reward_pips * pip
        else:
            stop_loss = entry + risk_pips * pip
            take_profit = entry - reward_pips * pip

        confidence = self._confidence(row, model, atr_pips)
        regime = self._regime_label(float(row["adx"]))
        htf_margin_atr = abs(float(row["close"]) - float(row["ema50_4h"])) / float(row["atr"])
        self._last_signal_idx[epic] = idx

        monitor_only = cfg.is_monitor_only(epic) if cfg else bool(self.config_override.get("monitor_only", False))

        signal = {
            "signal": direction,
            "signal_type": signal_type,
            "strategy": STRATEGY_NAME,
            "epic": epic,
            "pair": pair or epic.replace("CS.D.", "").replace(".MINI.IP", "").replace(".CEEM.IP", ""),
            "entry_price": entry,
            "current_price": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "stop_loss_price": stop_loss,
            "take_profit_price": take_profit,
            "risk_pips": risk_pips,
            "reward_pips": reward_pips,
            "stop_loss_pips": risk_pips,
            "take_profit_pips": reward_pips,
            "confidence_score": confidence,
            "confidence": confidence,
            "signal_timestamp": ts.isoformat(),
            "timestamp": ts,
            "version": VERSION,
            "entry_type": model,
            "market_regime": regime,
            "regime": regime,
            "strategy_regime": regime,
            "market_session": _session_label(hour),
            "adx": float(row["adx"]),
            "rsi": float(row["rsi"]) if pd.notna(row["rsi"]) else None,
            "atr": float(row["atr"]),
            "atr_pips": atr_pips,
            "timeframe": "15m",
            "fa_or_atr_trail_enabled": True,
            "monitor_only": monitor_only,
            "fa_or_trail_trigger_pips": atr_pips * trail_trigger_atr,
            "fa_or_trail_distance_pips": atr_pips * trail_distance_atr,
            "fa_or_model": model,
            "strategy_indicators": {
                "model": model,
                "ema9": float(row["ema9"]),
                "ema21": float(row["ema21"]),
                "ema50": float(row["ema50"]),
                "ema50_slope_pips": float(row["ema50_slope_pips"]),
                "ema50_4h": float(row["ema50_4h"]),
                "adx": float(row["adx"]),
                "rsi": float(row["rsi"]) if pd.notna(row["rsi"]) else None,
                "atr_pips": atr_pips,
                "vwap": float(row["vwap"]),
                "value_high": float(row["value_high"]) if pd.notna(row["value_high"]) else None,
                "value_low": float(row["value_low"]) if pd.notna(row["value_low"]) else None,
                "session_hour": hour,
                "usd_jpy_atr_floor_pips": usd_jpy_atr_floor_pips if epic == "CS.D.USDJPY.MINI.IP" else None,
                "htf_margin_atr": round(htf_margin_atr, 3),
                "htf_margin_pips": round(abs(float(row["close"]) - float(row["ema50_4h"])) / pip, 1),
            },
        }
        try:
            from forex_scanner.core.strategies.helpers.smc_performance_metrics import enrich_signal_with_performance_metrics
            signal = enrich_signal_with_performance_metrics(
                signal, df_entry=None, df_trigger=df_trigger, df_htf=None, epic=epic, logger=self.logger
            )
        except Exception as _pm_exc:
            self.logger.warning("[FA_OR_ATR_TRAIL] Performance metrics failed: %s", _pm_exc)
        return signal

    def _log_rejection(
        self,
        epic: str,
        pair: str,
        stage: str,
        reason: str,
        direction: Optional[str] = None,
        hour_utc: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = self._current_timestamp or datetime.now(timezone.utc)
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)

        self._pending_rejections.append({
            "epic": epic,
            "stage": stage,
            "reason": reason,
            "timestamp": ts.isoformat(),
        })

        self.logger.info("[FA_OR_ATR_TRAIL] %s ❌ %s: %s", pair or epic, stage, reason)

        if self._rej_mgr is not None:
            self._rej_mgr.reject(
                stage=stage,
                reason=reason,
                epic=epic,
                pair=pair or epic.replace("CS.D.", "").replace(".MINI.IP", "").replace(".CEEM.IP", ""),
                direction=direction,
                hour_utc=hour_utc if hour_utc is not None else ts.hour,
                scan_timestamp=ts,
                details=details,
            )

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "start_time" in out.columns:
            out["start_time"] = pd.to_datetime(out["start_time"], utc=True)
            out = out.sort_values("start_time").set_index("start_time", drop=False)
        elif isinstance(out.index, pd.DatetimeIndex):
            out = out.sort_index()
            out["start_time"] = pd.to_datetime(out.index, utc=True)
        else:
            out = out.reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in out.columns:
                out[col] = out[col].astype(float)
        if "volume" not in out.columns:
            out["volume"] = 1.0
        return out

    def _add_indicators(self, df: pd.DataFrame, epic: str) -> pd.DataFrame:
        out = df.copy()
        pip = _pip_size(epic)
        close = out["close"]
        high = out["high"]
        low = out["low"]

        out["ema9"] = close.ewm(span=9, adjust=False).mean()
        out["ema21"] = close.ewm(span=21, adjust=False).mean()
        out["ema50"] = close.ewm(span=50, adjust=False).mean()
        out["ema50_slope_pips"] = (out["ema50"] - out["ema50"].shift(5)) / pip

        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        out["atr"] = _wilder(tr, self.atr_period)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
        atr_w = _wilder(tr, self.adx_period).replace(0, np.nan)
        plus_di = 100 * _wilder(plus_dm, self.adx_period) / atr_w
        minus_di = 100 * _wilder(minus_dm, self.adx_period) / atr_w
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        out["adx"] = _wilder(dx, self.adx_period)

        delta = close.diff()
        gain = _wilder(delta.clip(lower=0), self.rsi_period)
        loss = _wilder((-delta).clip(lower=0), self.rsi_period)
        rs = gain / loss.replace(0, np.nan)
        out["rsi"] = 100 - (100 / (1 + rs))

        typical = (high + low + close) / 3
        vol = out["volume"].where(out["volume"] > 0, 1.0)
        if "start_time" in out.columns:
            session_date = pd.to_datetime(out["start_time"], utc=True).dt.date
        else:
            session_date = out.index.date
        out["vwap"] = (typical * vol).groupby(session_date).cumsum() / vol.groupby(session_date).cumsum()

        mean = close.rolling(self.vp_lookback).mean()
        std = close.rolling(self.vp_lookback).std()
        out["value_high"] = mean + self.value_area_std * std
        out["value_low"] = mean - self.value_area_std * std

        if isinstance(out.index, pd.DatetimeIndex):
            htf = out.resample("4h", label="right", closed="right").agg(close=("close", "last")).dropna()
            htf["ema50_4h"] = htf["close"].ewm(span=self.htf_ema_period, adjust=False).mean()
            out["ema50_4h"] = htf["ema50_4h"].reindex(out.index, method="ffill")
        else:
            out["ema50_4h"] = out["ema50"]
        return out

    def _signal_model(self, df: pd.DataFrame, row: pd.Series, prev: pd.Series) -> Tuple[Optional[str], Optional[str]]:
        body = abs(float(row["close"]) - float(row["open"]))
        strong_body = body >= 0.25 * float(row["atr"])

        if self._common_direction_filter(row, "BUY"):
            if (
                prev["low"] < prev["value_low"]
                and row["close"] > row["value_low"]
                and row["close"] > row["open"]
                and strong_body
            ):
                return "BUY", "FA"
            or_high, _, lock_ts = self._opening_range(df, row)
            if lock_ts is not None and row.name > lock_ts and row["close"] > or_high and prev["close"] <= or_high:
                return "BUY", "OR"

        if self._common_direction_filter(row, "SELL"):
            if (
                prev["high"] > prev["value_high"]
                and row["close"] < row["value_high"]
                and row["close"] < row["open"]
                and strong_body
            ):
                return "SELL", "FA"
            _, or_low, lock_ts = self._opening_range(df, row)
            if lock_ts is not None and row.name > lock_ts and row["close"] < or_low and prev["close"] >= or_low:
                return "SELL", "OR"

        return None, None

    def _common_direction_filter(self, row: pd.Series, direction: str) -> bool:
        min_margin = float(row["atr"]) * self.min_htf_margin_atr
        if direction == "BUY":
            return (
                row["ema9"] > row["ema21"] > row["ema50"]
                and row["ema50_slope_pips"] >= self.min_slope_pips
                and row["close"] > row["ema50_4h"] + min_margin
            )
        return (
            row["ema9"] < row["ema21"] < row["ema50"]
            and row["ema50_slope_pips"] <= -self.min_slope_pips
            and row["close"] < row["ema50_4h"] - min_margin
        )

    def _opening_range(self, df: pd.DataFrame, row: pd.Series) -> Tuple[float, float, Optional[pd.Timestamp]]:
        if not isinstance(df.index, pd.DatetimeIndex):
            return np.nan, np.nan, None
        day = row.name.date()
        day_df = df[df.index.date == day]
        session = day_df[
            (day_df.index.hour >= self.session_start_hour)
            & (day_df.index.hour < self.session_end_hour)
        ]
        if len(session) < self.opening_range_bars:
            return np.nan, np.nan, None
        first = session.iloc[: self.opening_range_bars]
        return float(first["high"].max()), float(first["low"].min()), first.index[-1]

    def _row_timestamp(self, row: pd.Series, current_timestamp: Optional[datetime]) -> datetime:
        if current_timestamp is not None:
            ts = pd.Timestamp(current_timestamp)
        elif "start_time" in row and pd.notna(row["start_time"]):
            ts = pd.Timestamp(row["start_time"])
        elif hasattr(row, "name") and isinstance(row.name, pd.Timestamp):
            ts = row.name
        else:
            ts = pd.Timestamp.now(tz=timezone.utc)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        return ts.to_pydatetime()

    @staticmethod
    def _regime_label(adx: float) -> str:
        if adx < 18:
            return "ranging"
        if adx < 25:
            return "weak_trend"
        return "trending"

    @staticmethod
    def _confidence(row: pd.Series, model: str, atr_pips: float) -> float:
        base = 0.66 if model == "FA" else 0.68
        adx_bonus = min(0.12, max(0.0, (float(row["adx"]) - 18.0) / 100.0))
        # Normalize slope by ATR so a 2-pip slope on a 10-pip ATR pair scores the same
        # as a 2-pip slope on a 2-pip ATR pair — avoids over-rewarding weak-trend slopes.
        slope_bonus = min(0.08, abs(float(row["ema50_slope_pips"])) / max(atr_pips * 4.0, 1.0))
        atr_bonus = min(0.06, atr_pips / 200.0)
        return round(min(0.9, base + adx_bonus + slope_bonus + atr_bonus), 3)


def create_fa_or_atr_trail_strategy(
    config=None,
    logger=None,
    db_manager=None,
    config_override: Optional[Dict[str, Any]] = None,
) -> FAORATRTrailStrategy:
    return FAORATRTrailStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager,
        config_override=config_override,
    )
