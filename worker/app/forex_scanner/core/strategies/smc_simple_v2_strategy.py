#!/usr/bin/env python3
"""
SMC Simple V2 - raw entry-model research strategy.

This intentionally does not inherit SMC_SIMPLE's filter stack. The first goal is
to test whether a small 5m structure context plus 1m entry patterns has a raw
edge before adding direction, session, regime, indicator, or pair filters.
"""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .strategy_registry import StrategyInterface, register_strategy

logger = logging.getLogger(__name__)

try:
    from forex_scanner.services.smc_simple_v2_config_service import get_smc_simple_v2_config_service
except ImportError:
    try:
        from services.smc_simple_v2_config_service import get_smc_simple_v2_config_service
    except ImportError:
        get_smc_simple_v2_config_service = None  # type: ignore[assignment]

try:
    from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from alerts.strategy_rejection_manager import StrategyRejectionManager
    except ImportError:
        StrategyRejectionManager = None  # type: ignore[assignment]


@dataclass
class SMCSimpleV2Config:
    enabled_epics: str = "CS.D.EURUSD.CEEM.IP"
    structure_lookback_bars: int = 12
    entry_lookback_bars: int = 25
    retest_tolerance_pips: float = 1.5
    sweep_tolerance_pips: float = 0.3
    min_break_pips: float = 0.2
    min_rejection_wick_ratio: float = 0.45
    max_rejection_body_ratio: float = 0.65
    min_confirm_body_ratio: float = 0.35
    sl_pips: float = 5.0
    tp_pips: float = 6.0
    directions: str = "BULL"
    entry_models: str = "REJECTION_BREAK"
    min_signal_gap_minutes: int = 60
    adx_min: float = 0.0
    adx_max: float = 0.0
    atr_percentile_min: float = 0.0
    atr_percentile_max: float = 0.0
    bb_width_percentile_min: float = 0.0
    bb_width_percentile_max: float = 0.0
    efficiency_ratio_min: float = 0.0
    ema200_mode: str = "OFF"  # OFF, ALIGN, COUNTER
    macd_mode: str = "OFF"    # OFF, ALIGN, COUNTER
    allowed_hours_utc: str = "7,8,9,10,11,12"
    monitor_only: bool = False
    base_confidence: float = 0.60


@register_strategy("SMC_SIMPLE_V2")
class SMCSimpleV2Strategy(StrategyInterface):
    strategy_version = "smc_simple_v2.0"
    uses_smart_money_analysis = False

    def __init__(
        self,
        config: Any = None,
        db_manager: Any = None,
        logger: Optional[logging.Logger] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.config_override = dict(config_override or {})
        self.config = SMCSimpleV2Config()
        self._last_signal_time: Dict[str, datetime] = {}
        self._apply_overrides()
        self._base_config = deepcopy(self.config)
        self._config_set = os.getenv("TRADING_CONFIG_SET", "demo")
        self._config_service = get_smc_simple_v2_config_service() if get_smc_simple_v2_config_service else None
        self._current_timestamp: Optional[datetime] = None

        self._rej_mgr = None
        if db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager("SMC_SIMPLE_V2", db_manager)
            except Exception:
                self._rej_mgr = None

    @property
    def strategy_name(self) -> str:
        return "SMC_SIMPLE_V2"

    def get_required_timeframes(self) -> List[str]:
        return ["5m", "1m"]

    def reset_cooldowns(self) -> None:
        return None

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
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger.debug("[SMC_SIMPLE_V2] %s ❌ %s: %s", pair or epic, stage, reason)
        if self._rej_mgr is not None:
            hour_utc = None
            ts = self._current_timestamp
            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                hour_utc = ts.hour
            self._rej_mgr.reject(
                stage=stage,
                reason=reason,
                epic=epic,
                pair=pair,
                direction=direction,
                hour_utc=hour_utc,
                scan_timestamp=self._current_timestamp,
                details=details,
            )

    def _apply_pair_row(self, epic: str, pair: str) -> bool:
        """Reset to base config, then apply DB pair row for this scan."""
        self.config = deepcopy(self._base_config)
        if self._config_service is None:
            return self._is_epic_enabled(epic)

        row = self._config_service.get_pair_config(epic, self._config_set)
        if not row:
            self._reject("PAIR_CONFIG", "no SMC_SIMPLE_V2 pair row", epic, pair)
            return False
        if not bool(row.get("is_enabled", False)):
            self._reject("PAIR_DISABLED", "pair disabled", epic, pair)
            return False

        mapping = {
            "structure_lookback_bars": ("structure_lookback_bars", int),
            "entry_lookback_bars": ("entry_lookback_bars", int),
            "retest_tolerance_pips": ("retest_tolerance_pips", float),
            "sweep_tolerance_pips": ("sweep_tolerance_pips", float),
            "min_break_pips": ("min_break_pips", float),
            "min_rejection_wick_ratio": ("min_rejection_wick_ratio", float),
            "max_rejection_body_ratio": ("max_rejection_body_ratio", float),
            "min_confirm_body_ratio": ("min_confirm_body_ratio", float),
            "fixed_stop_loss_pips": ("sl_pips", float),
            "fixed_take_profit_pips": ("tp_pips", float),
            "directions": ("directions", str),
            "entry_models": ("entry_models", str),
            "min_signal_gap_minutes": ("min_signal_gap_minutes", int),
            "adx_min": ("adx_min", float),
            "adx_max": ("adx_max", float),
            "atr_percentile_min": ("atr_percentile_min", float),
            "atr_percentile_max": ("atr_percentile_max", float),
            "bb_width_percentile_min": ("bb_width_percentile_min", float),
            "bb_width_percentile_max": ("bb_width_percentile_max", float),
            "efficiency_ratio_min": ("efficiency_ratio_min", float),
            "ema200_mode": ("ema200_mode", str),
            "macd_mode": ("macd_mode", str),
            "allowed_hours_utc": ("allowed_hours_utc", str),
            "base_confidence": ("base_confidence", float),
            "monitor_only": ("monitor_only", bool),
        }
        values: Dict[str, Any] = dict(row)
        bag = values.get("parameter_overrides") or {}
        if isinstance(bag, dict):
            values.update({k: v for k, v in bag.items() if v is not None})

        for key, (attr, caster) in mapping.items():
            value = values.get(key)
            if value is None:
                continue
            try:
                if caster is bool:
                    if isinstance(value, str):
                        value = value.strip().lower() in {"1", "true", "yes", "on", "t"}
                    else:
                        value = bool(value)
                else:
                    value = caster(value)
                setattr(self.config, attr, value)
            except Exception:
                self.logger.warning("[SMC_SIMPLE_V2] ignored invalid %s=%r for %s", key, value, epic)

        self.config.enabled_epics = epic
        return True

    def _apply_overrides(self) -> None:
        mapping = {
            "v2_enabled_epics": ("enabled_epics", str),
            "v2_structure_lookback_bars": ("structure_lookback_bars", int),
            "v2_entry_lookback_bars": ("entry_lookback_bars", int),
            "v2_retest_tolerance_pips": ("retest_tolerance_pips", float),
            "v2_sweep_tolerance_pips": ("sweep_tolerance_pips", float),
            "v2_min_break_pips": ("min_break_pips", float),
            "v2_min_rejection_wick_ratio": ("min_rejection_wick_ratio", float),
            "v2_max_rejection_body_ratio": ("max_rejection_body_ratio", float),
            "v2_min_confirm_body_ratio": ("min_confirm_body_ratio", float),
            "v2_sl_pips": ("sl_pips", float),
            "v2_tp_pips": ("tp_pips", float),
            "v2_directions": ("directions", str),
            "v2_entry_models": ("entry_models", str),
            "v2_min_signal_gap_minutes": ("min_signal_gap_minutes", int),
            "v2_adx_min": ("adx_min", float),
            "v2_adx_max": ("adx_max", float),
            "v2_atr_percentile_min": ("atr_percentile_min", float),
            "v2_atr_percentile_max": ("atr_percentile_max", float),
            "v2_bb_width_percentile_min": ("bb_width_percentile_min", float),
            "v2_bb_width_percentile_max": ("bb_width_percentile_max", float),
            "v2_efficiency_ratio_min": ("efficiency_ratio_min", float),
            "v2_ema200_mode": ("ema200_mode", str),
            "v2_macd_mode": ("macd_mode", str),
            "v2_allowed_hours_utc": ("allowed_hours_utc", str),
            "monitor_only": ("monitor_only", bool),
        }
        for key, (attr, caster) in mapping.items():
            if key not in self.config_override:
                continue
            value = self.config_override[key]
            if caster is bool:
                if isinstance(value, str):
                    value = value.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    value = bool(value)
            else:
                value = caster(value)
            setattr(self.config, attr, value)

        if "scalp_sl_pips" in self.config_override and "v2_sl_pips" not in self.config_override:
            self.config.sl_pips = float(self.config_override["scalp_sl_pips"])
        if "scalp_tp_pips" in self.config_override and "v2_tp_pips" not in self.config_override:
            self.config.tp_pips = float(self.config_override["scalp_tp_pips"])

    def detect_signal(
        self,
        df_trigger: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        df_entry: Optional[pd.DataFrame] = None,
        epic: str = "",
        pair: str = "",
        spread_pips: float = 1.5,
        current_timestamp: Optional[datetime] = None,
        **_: Any,
    ) -> Optional[Dict[str, Any]]:
        self._current_timestamp = current_timestamp
        if not self._apply_pair_row(epic, pair):
            return None

        if df_trigger is None or df_entry is None or len(df_trigger) < 20 or len(df_entry) < 8:
            self._reject(
                "INSUFFICIENT_DATA",
                "need >=20 5m bars and >=8 1m bars",
                epic,
                pair,
                details={"trigger_rows": 0 if df_trigger is None else len(df_trigger),
                         "entry_rows": 0 if df_entry is None else len(df_entry)},
            )
            return None
        if not self._is_epic_enabled(epic):
            self._reject("PAIR_DISABLED", "epic not in enabled_epics", epic, pair)
            return None

        trigger = self._normalise(df_trigger)
        entry = self._normalise(df_entry)
        if trigger is None or entry is None:
            self._reject("BAD_DATA", "missing OHLC data after normalization", epic, pair)
            return None

        context = self._detect_structure_context(trigger)
        if context is None:
            self._reject("NO_STRUCTURE_BREAK", "no 5m structure break", epic, pair)
            return None

        direction, structure_level, break_pips = context
        allowed = {d.strip().upper() for d in self.config.directions.split(",") if d.strip()}
        if direction not in allowed:
            self._reject(
                "DIRECTION_BLOCKED",
                f"{direction} not allowed",
                epic,
                pair,
                direction=direction,
                details={"allowed": sorted(allowed)},
            )
            return None

        filter_context = self._extract_filter_context(trigger, direction)
        if not self._passes_filters(filter_context):
            self._reject("FILTERS", "indicator filter failed", epic, pair, direction=direction, details=filter_context)
            return None

        last = entry.iloc[-1]
        timestamp = self._timestamp_from_row(last, current_timestamp)
        if not self._passes_hour_filter(timestamp):
            self._reject(
                "SESSION",
                f"hour={timestamp.hour} outside {self.config.allowed_hours_utc}",
                epic,
                pair,
                direction=direction,
                details={"hour": timestamp.hour, "allowed_hours_utc": self.config.allowed_hours_utc},
            )
            return None
        if self._is_in_signal_gap(epic, direction, timestamp):
            self._reject("COOLDOWN", "signal gap active", epic, pair, direction=direction)
            return None

        entry_model = self._detect_entry_model(entry, direction, structure_level)
        if entry_model is None:
            self._reject(
                "NO_ENTRY_MODEL",
                "no 1m entry model matched",
                epic,
                pair,
                direction=direction,
                details={"entry_models": self.config.entry_models},
            )
            return None

        entry_type, confidence_boost = entry_model
        entry_price = float(last["close"])
        pip = self._pip_size(epic, pair)
        sl_pips = float(self.config.sl_pips)
        tp_pips = float(self.config.tp_pips)

        if direction == "BULL":
            signal = "BUY"
            stop_loss = entry_price - sl_pips * pip
            take_profit = entry_price + tp_pips * pip
        else:
            signal = "SELL"
            stop_loss = entry_price + sl_pips * pip
            take_profit = entry_price - tp_pips * pip

        confidence = min(0.85, self.config.base_confidence + confidence_boost + min(break_pips / 20.0, 0.05))
        self._last_signal_time[f"{epic}:{direction}"] = timestamp

        return {
            "strategy": self.strategy_name,
            "version": self.strategy_version,
            "epic": epic,
            "pair": pair or epic,
            "timeframe": "1m",
            "signal": signal,
            "direction": direction,
            "signal_type": "bull" if signal == "BUY" else "bear",
            "entry_type": entry_type,
            "entry_price": entry_price,
            "market_price": entry_price,
            "signal_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_pips": sl_pips,
            "reward_pips": tp_pips,
            "rr_ratio": round(tp_pips / sl_pips, 2) if sl_pips else 0.0,
            "confidence": confidence,
            "confidence_score": confidence,
            "timestamp": timestamp,
            "signal_timestamp": timestamp,
            "market_timestamp": timestamp,
            "order_type": "market",
            "api_order_type": "MARKET",
            "monitor_only": bool(self.config.monitor_only),
            "skip_sr_validation": True,
            "scalp_mode": True,
            "v2_structure_level": structure_level,
            "v2_structure_break_pips": break_pips,
            "signal_conditions": {
                "entry_model": entry_type,
                "structure_level": structure_level,
                "structure_break_pips": break_pips,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
            },
            "strategy_indicators": {
                "v2_entry_model": entry_type,
                "v2_structure_direction": direction,
                "v2_structure_level": structure_level,
                "v2_adx_value": filter_context.get("adx_value"),
                "v2_atr_percentile": filter_context.get("atr_percentile"),
                "v2_bb_width_percentile": filter_context.get("bb_width_percentile"),
                "v2_efficiency_ratio": filter_context.get("efficiency_ratio"),
                "v2_ema200_mode": self.config.ema200_mode,
                "v2_macd_mode": self.config.macd_mode,
            },
        }

    def _normalise(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            return None
        clean = df.copy()
        for col in required:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean = clean.dropna(subset=list(required))
        clean = self._ensure_filter_indicators(clean)
        return clean if len(clean) else None

    def _ensure_filter_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()
        if "ema_200" not in clean.columns:
            clean["ema_200"] = clean["close"].ewm(span=200, adjust=False).mean()

        if "atr_percentile" not in clean.columns:
            prev_close = clean["close"].shift(1)
            tr = pd.concat(
                [
                    clean["high"] - clean["low"],
                    (clean["high"] - prev_close).abs(),
                    (clean["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.ewm(span=14, adjust=False).mean()
            clean["atr_percentile"] = atr.rolling(window=50, min_periods=10).rank(pct=True) * 100
            clean["atr_percentile"] = clean["atr_percentile"].fillna(50.0)

        if "bb_width_percentile" not in clean.columns:
            middle = clean["close"].rolling(window=20, min_periods=10).mean()
            stdev = clean["close"].rolling(window=20, min_periods=10).std()
            width = ((middle + 2 * stdev) - (middle - 2 * stdev)) / middle.replace(0, pd.NA)
            clean["bb_width_percentile"] = width.rolling(window=50, min_periods=10).rank(pct=True) * 100
            clean["bb_width_percentile"] = clean["bb_width_percentile"].fillna(50.0)

        if "efficiency_ratio" not in clean.columns:
            period = 10
            change = (clean["close"] - clean["close"].shift(period)).abs()
            volatility = clean["close"].diff().abs().rolling(window=period, min_periods=period).sum()
            clean["efficiency_ratio"] = (change / volatility.replace(0, pd.NA)).fillna(0.0)

        if "macd_histogram" not in clean.columns:
            ema12 = clean["close"].ewm(span=12, adjust=False).mean()
            ema26 = clean["close"].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            clean["macd_histogram"] = macd - signal

        return clean

    def _detect_structure_context(self, df_5m: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
        lookback = max(5, int(self.config.structure_lookback_bars))
        if len(df_5m) < lookback + 2:
            return None

        pip = self._pip_size("", "")
        recent = df_5m.iloc[-(lookback + 1):-1]
        last = df_5m.iloc[-1]
        swing_high = float(recent["high"].max())
        swing_low = float(recent["low"].min())
        close = float(last["close"])
        min_break = self.config.min_break_pips * pip

        if close > swing_high + min_break:
            return "BULL", swing_high, (close - swing_high) / pip
        if close < swing_low - min_break:
            return "BEAR", swing_low, (swing_low - close) / pip
        return None

    def _detect_entry_model(
        self,
        df_1m: pd.DataFrame,
        direction: str,
        structure_level: float,
    ) -> Optional[Tuple[str, float]]:
        checks = (
            self._rejection_break_entry,
            self._sweep_reclaim_entry,
            self._engulfing_entry,
            self._level_retest_entry,
        )
        for check in checks:
            result = check(df_1m, direction, structure_level)
            if result is not None:
                return result
        return None

    def _entry_model_allowed(self, name: str) -> bool:
        allowed = {m.strip().upper() for m in self.config.entry_models.split(",") if m.strip()}
        return name.upper() in allowed

    def _is_epic_enabled(self, epic: str) -> bool:
        enabled = {e.strip().upper() for e in self.config.enabled_epics.split(",") if e.strip()}
        return epic.upper() in enabled

    def _rejection_break_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        structure_level: float,
    ) -> Optional[Tuple[str, float]]:
        if len(df) < 3:
            return None
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        prev_range = max(float(prev["high"] - prev["low"]), 1e-9)
        curr_range = max(float(curr["high"] - curr["low"]), 1e-9)
        prev_body = abs(float(prev["close"] - prev["open"]))
        curr_body = abs(float(curr["close"] - curr["open"]))
        body_ok = prev_body / prev_range <= self.config.max_rejection_body_ratio
        confirm_ok = curr_body / curr_range >= self.config.min_confirm_body_ratio

        if direction == "BULL":
            wick = min(float(prev["open"]), float(prev["close"])) - float(prev["low"])
            if (
                body_ok
                and confirm_ok
                and wick / prev_range >= self.config.min_rejection_wick_ratio
                and float(curr["close"]) > float(prev["high"])
                and float(curr["close"]) > float(curr["open"])
            ):
                return ("REJECTION_BREAK", 0.05) if self._entry_model_allowed("REJECTION_BREAK") else None
        else:
            wick = float(prev["high"]) - max(float(prev["open"]), float(prev["close"]))
            if (
                body_ok
                and confirm_ok
                and wick / prev_range >= self.config.min_rejection_wick_ratio
                and float(curr["close"]) < float(prev["low"])
                and float(curr["close"]) < float(curr["open"])
            ):
                return ("REJECTION_BREAK", 0.05) if self._entry_model_allowed("REJECTION_BREAK") else None
        return None

    def _engulfing_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        structure_level: float,
    ) -> Optional[Tuple[str, float]]:
        if len(df) < 2:
            return None
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        if direction == "BULL":
            if (
                float(prev["close"]) < float(prev["open"])
                and float(curr["close"]) > float(curr["open"])
                and float(curr["close"]) >= float(prev["open"])
                and float(curr["open"]) <= float(prev["close"])
            ):
                return ("ENGULFING_CONTINUATION", 0.03) if self._entry_model_allowed("ENGULFING_CONTINUATION") else None
        else:
            if (
                float(prev["close"]) > float(prev["open"])
                and float(curr["close"]) < float(curr["open"])
                and float(curr["close"]) <= float(prev["open"])
                and float(curr["open"]) >= float(prev["close"])
            ):
                return ("ENGULFING_CONTINUATION", 0.03) if self._entry_model_allowed("ENGULFING_CONTINUATION") else None
        return None

    def _sweep_reclaim_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        structure_level: float,
    ) -> Optional[Tuple[str, float]]:
        lookback = max(5, min(int(self.config.entry_lookback_bars), len(df) - 2))
        if lookback < 5:
            return None
        pip = self._pip_size("", "")
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        window = df.iloc[-(lookback + 2):-2]
        tol = self.config.sweep_tolerance_pips * pip

        if direction == "BULL":
            prior_low = float(window["low"].min())
            if float(prev["low"]) < prior_low - tol and float(prev["close"]) > prior_low and float(curr["close"]) > float(prev["high"]):
                return ("LIQUIDITY_SWEEP_RECLAIM", 0.06) if self._entry_model_allowed("LIQUIDITY_SWEEP_RECLAIM") else None
        else:
            prior_high = float(window["high"].max())
            if float(prev["high"]) > prior_high + tol and float(prev["close"]) < prior_high and float(curr["close"]) < float(prev["low"]):
                return ("LIQUIDITY_SWEEP_RECLAIM", 0.06) if self._entry_model_allowed("LIQUIDITY_SWEEP_RECLAIM") else None
        return None

    def _level_retest_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        structure_level: float,
    ) -> Optional[Tuple[str, float]]:
        curr = df.iloc[-1]
        pip = self._pip_size("", "")
        tol = self.config.retest_tolerance_pips * pip

        if direction == "BULL":
            if float(curr["low"]) <= structure_level + tol and float(curr["close"]) > structure_level and float(curr["close"]) > float(curr["open"]):
                return ("BROKEN_LEVEL_RETEST", 0.04) if self._entry_model_allowed("BROKEN_LEVEL_RETEST") else None
        else:
            if float(curr["high"]) >= structure_level - tol and float(curr["close"]) < structure_level and float(curr["close"]) < float(curr["open"]):
                return ("BROKEN_LEVEL_RETEST", 0.04) if self._entry_model_allowed("BROKEN_LEVEL_RETEST") else None
        return None

    def _is_in_signal_gap(self, epic: str, direction: str, timestamp: datetime) -> bool:
        gap = int(self.config.min_signal_gap_minutes)
        if gap <= 0:
            return False
        key = f"{epic}:{direction}"
        last = self._last_signal_time.get(key)
        if last is None:
            return False
        return timestamp < last + timedelta(minutes=gap)

    def _extract_filter_context(self, df: pd.DataFrame, direction: str) -> Dict[str, Optional[float]]:
        last = df.iloc[-1]
        return {
            "direction": direction,
            "close": self._float_or_none(last.get("close")),
            "adx_value": self._float_or_none(last.get("adx_value")),
            "atr_percentile": self._float_or_none(last.get("atr_percentile")),
            "bb_width_percentile": self._float_or_none(last.get("bb_width_percentile")),
            "efficiency_ratio": self._float_or_none(last.get("efficiency_ratio")),
            "ema_200": self._float_or_none(last.get("ema_200")),
            "macd_histogram": self._float_or_none(last.get("macd_histogram")),
        }

    def _passes_filters(self, context: Dict[str, Optional[float]]) -> bool:
        if not self._passes_range(context.get("adx_value"), self.config.adx_min, self.config.adx_max):
            return False
        if not self._passes_range(context.get("atr_percentile"), self.config.atr_percentile_min, self.config.atr_percentile_max):
            return False
        if not self._passes_range(context.get("bb_width_percentile"), self.config.bb_width_percentile_min, self.config.bb_width_percentile_max):
            return False
        if self.config.efficiency_ratio_min > 0:
            er = context.get("efficiency_ratio")
            if er is None or er < self.config.efficiency_ratio_min:
                return False

        direction = str(context.get("direction") or "").upper()
        close = context.get("close")
        ema_200 = context.get("ema_200")
        ema_mode = self.config.ema200_mode.upper()
        if ema_mode != "OFF":
            if close is None or ema_200 is None:
                return False
            aligned = (direction == "BULL" and close > ema_200) or (direction == "BEAR" and close < ema_200)
            if ema_mode == "ALIGN" and not aligned:
                return False
            if ema_mode == "COUNTER" and aligned:
                return False

        macd = context.get("macd_histogram")
        macd_mode = self.config.macd_mode.upper()
        if macd_mode != "OFF":
            if macd is None:
                return False
            aligned = (direction == "BULL" and macd > 0) or (direction == "BEAR" and macd < 0)
            if macd_mode == "ALIGN" and not aligned:
                return False
            if macd_mode == "COUNTER" and aligned:
                return False
        return True

    def _passes_range(self, value: Optional[float], min_value: float, max_value: float) -> bool:
        if min_value <= 0 and max_value <= 0:
            return True
        if value is None:
            return False
        if min_value > 0 and value < min_value:
            return False
        if max_value > 0 and value > max_value:
            return False
        return True

    def _passes_hour_filter(self, timestamp: datetime) -> bool:
        raw = self.config.allowed_hours_utc.strip()
        if not raw:
            return True
        allowed = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                allowed.add(int(part))
            except ValueError:
                return False
        return timestamp.hour in allowed

    def _float_or_none(self, value: Any) -> Optional[float]:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    def _timestamp_from_row(self, row: pd.Series, fallback: Optional[datetime]) -> datetime:
        for key in ("start_time", "timestamp", "time"):
            if key in row and pd.notna(row[key]):
                try:
                    value = pd.to_datetime(row[key])
                    if value.tzinfo is None:
                        value = value.tz_localize(timezone.utc)
                    return value.to_pydatetime()
                except Exception:
                    pass
        return fallback or datetime.now(timezone.utc)

    def _pip_size(self, epic: str, pair: str) -> float:
        text = f"{epic} {pair}".upper()
        if "JPY" in text:
            return 0.01
        return 0.0001


def create_smc_simple_v2_strategy(
    config: Any = None,
    db_manager: Any = None,
    logger: Optional[logging.Logger] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> SMCSimpleV2Strategy:
    return SMCSimpleV2Strategy(
        config=config,
        db_manager=db_manager,
        logger=logger,
        config_override=config_override,
    )
