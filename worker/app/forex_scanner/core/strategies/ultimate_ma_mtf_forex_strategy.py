#!/usr/bin/env python3
"""
Ultimate MA MTF strategy adapted for forex.

This is a separate forex implementation inspired by the stock scanner's
UltimateMAMTFStrategy. It keeps the configurable MA color/cross logic, but
replaces stock-only RS/volume gates with forex-appropriate 15m trigger,
1h/4h trend confirmation, ATR risk, and ADX/RSI filters.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .strategy_registry import StrategyInterface, register_strategy

logger = logging.getLogger(__name__)


def _pip_size(pair: str, epic: str = "") -> float:
    text = f"{pair} {epic}".upper()
    return 0.01 if "JPY" in text else 0.0001


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


@register_strategy("ULTIMATE_MA_MTF_FOREX")
class UltimateMAMTFForexStrategy(StrategyInterface):
    """15m Ultimate MA trigger with 1h/4h forex trend confirmation."""

    MA_TYPES = {
        1: "SMA",
        2: "EMA",
        3: "WMA",
        4: "HULL",
        5: "VWMA",
        6: "RMA",
        7: "TEMA",
        8: "T3",
    }

    def __init__(self, config_override: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self.config_override = config_override or {}
        self.db_manager = kwargs.get("db_manager")
        self._last_signal_time: Dict[str, datetime] = {}

        self.ma_type = int(self.config_override.get("ma_type", 2))
        self.ma_length = int(self.config_override.get("ma_length", 20))
        self.t3_factor = float(self.config_override.get("t3_factor", 0.7))
        self.use_second_ma = bool(self.config_override.get("use_second_ma", True))
        self.second_ma_type = int(self.config_override.get("second_ma_type", 2))
        self.second_ma_length = int(self.config_override.get("second_ma_length", 50))
        self.second_t3_factor = float(self.config_override.get("second_t3_factor", 0.7))
        self.smoothing_bars = max(1, int(self.config_override.get("smoothing_bars", 2)))
        self.trigger_mode = str(self.config_override.get("trigger_mode", "line_color_close"))
        self.require_ma_slope = bool(self.config_override.get("require_ma_slope", True))
        self.require_second_ma_trend = bool(self.config_override.get("require_second_ma_trend", False))
        self.require_mtf_alignment = bool(self.config_override.get("require_mtf_alignment", True))
        self.min_mtf_aligned = int(self.config_override.get("min_mtf_aligned", 1))
        self.allow_shorts = bool(self.config_override.get("allow_shorts", True))

        self.atr_period = int(self.config_override.get("atr_period", 14))
        self.adx_period = int(self.config_override.get("adx_period", 14))
        self.htf_ema_period = int(self.config_override.get("htf_ema_period", 50))
        self.mtf_slope_bars = int(self.config_override.get("mtf_slope_bars", 3))
        self.min_ma_slope_atr = float(self.config_override.get("min_ma_slope_atr", 0.01))
        self.max_price_distance_atr = float(self.config_override.get("max_price_distance_atr", 1.7))
        self.min_adx = float(self.config_override.get("min_adx", 12.0))
        self.min_rsi_buy = float(self.config_override.get("min_rsi_buy", 45.0))
        self.max_rsi_buy = float(self.config_override.get("max_rsi_buy", 78.0))
        self.min_rsi_sell = float(self.config_override.get("min_rsi_sell", 22.0))
        self.max_rsi_sell = float(self.config_override.get("max_rsi_sell", 55.0))
        self.stop_loss_atr_mult = float(self.config_override.get("stop_loss_atr_mult", 1.4))
        self.take_profit_atr_mult = float(self.config_override.get("take_profit_atr_mult", 2.1))
        self.min_confidence = float(self.config_override.get("min_confidence", 0.55))
        self.cooldown_minutes = int(self.config_override.get("cooldown_minutes", 180))
        self.version = "1.1.0"

    @property
    def strategy_name(self) -> str:
        return "ULTIMATE_MA_MTF_FOREX"

    def get_required_timeframes(self) -> list:
        return ["15m", "1h", "4h"]

    def reset_cooldowns(self) -> None:
        self._last_signal_time.clear()

    def detect_signal(
        self,
        df_trigger: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        df_macro: Optional[pd.DataFrame] = None,
        epic: str = "",
        pair: str = "",
        spread_pips: float = 1.5,
        current_timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if current_timestamp is None:
            current_timestamp = datetime.now(timezone.utc)
        if current_timestamp.tzinfo is None:
            current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)

        last_signal = self._last_signal_time.get(epic)
        if last_signal is not None:
            elapsed = (current_timestamp - last_signal).total_seconds() / 60.0
            if elapsed < self.cooldown_minutes:
                return None

        min_bars = max(self.ma_length, self.second_ma_length if self.use_second_ma else 0, self.atr_period, self.adx_period) + self.smoothing_bars + 8
        if df_trigger is None or len(df_trigger) < min_bars:
            return None
        if df_htf is None or len(df_htf) < self.htf_ema_period + self.mtf_slope_bars + 2:
            return None
        if self.require_mtf_alignment and (df_macro is None or len(df_macro) < self.htf_ema_period + self.mtf_slope_bars + 2):
            return None

        df = df_trigger.copy()
        df["ult_ma1"] = self._moving_average(df, self.ma_type, self.ma_length, self.t3_factor)
        if self.use_second_ma:
            df["ult_ma2"] = self._moving_average(df, self.second_ma_type, self.second_ma_length, self.second_t3_factor)
        if "atr" not in df.columns:
            df["atr"] = _atr(df, self.atr_period)
        if "adx" not in df.columns:
            df["adx"] = _adx(df, self.adx_period)
        if "rsi" not in df.columns:
            df["rsi"] = self._rsi(df["close"].astype(float), 14)

        current = df.iloc[-1]
        previous = df.iloc[-2]
        ma1 = current["ult_ma1"]
        atr_value = current["atr"]
        if pd.isna(ma1) or pd.isna(atr_value) or float(atr_value) <= 0:
            return None

        close = float(current["close"])
        open_price = float(current["open"])
        atr_value = float(atr_value)
        ma1 = float(ma1)
        ma1_prev = float(previous["ult_ma1"]) if pd.notna(previous["ult_ma1"]) else np.nan
        ma1_smooth = df["ult_ma1"].iloc[-(self.smoothing_bars + 1)]
        previous_ma1_smooth = df["ult_ma1"].iloc[-(self.smoothing_bars + 2)]
        if pd.isna(ma1_prev) or pd.isna(ma1_smooth) or pd.isna(previous_ma1_smooth):
            return None

        ma_slope = ma1 - float(ma1_smooth)
        ma_slope_atr = abs(ma_slope) / atr_value
        price_distance_atr = abs(close - ma1) / atr_value
        previous_close = float(previous["close"])
        line_green = ma1 >= float(ma1_smooth)
        line_red = ma1 < float(ma1_smooth)
        previous_line_green = ma1_prev >= float(previous_ma1_smooth)
        previous_line_red = ma1_prev < float(previous_ma1_smooth)
        line_color_buy = close > ma1 and line_green and not (previous_close > ma1_prev and previous_line_green)
        line_color_sell = close < ma1 and line_red and not (previous_close < ma1_prev and previous_line_red)
        continuation_buy = close > ma1 and line_green and ma_slope > 0
        continuation_sell = close < ma1 and line_red and ma_slope < 0
        price_cross_up = open_price < ma1 and close > ma1
        price_cross_down = open_price > ma1 and close < ma1

        ma2 = None
        ma_cross_up = False
        ma_cross_down = False
        if self.use_second_ma:
            ma2_val = current["ult_ma2"]
            prev_ma2_val = previous["ult_ma2"]
            if pd.isna(ma2_val) or pd.isna(prev_ma2_val):
                return None
            ma2 = float(ma2_val)
            prev_ma2 = float(prev_ma2_val)
            ma_cross_up = ma1_prev <= prev_ma2 and ma1 > ma2
            ma_cross_down = ma1_prev >= prev_ma2 and ma1 < ma2

        direction = None
        trigger = None
        if self.trigger_mode == "line_color_close" and line_color_buy:
            direction, trigger = "BUY", "close_above_green_ma"
        elif self.trigger_mode == "line_color_close" and self.allow_shorts and line_color_sell:
            direction, trigger = "SELL", "close_below_red_ma"
        elif self.trigger_mode == "line_color_close" and continuation_buy:
            direction, trigger = "BUY", "green_ma_continuation"
        elif self.trigger_mode == "line_color_close" and self.allow_shorts and continuation_sell:
            direction, trigger = "SELL", "red_ma_continuation"
        elif self.trigger_mode in ("price_cross", "price_or_ma_cross") and price_cross_up:
            direction, trigger = "BUY", "price_cross_ma1"
        elif self.trigger_mode in ("price_cross", "price_or_ma_cross") and self.allow_shorts and price_cross_down:
            direction, trigger = "SELL", "price_cross_ma1"

        if direction is None and self.trigger_mode in ("ma_cross", "price_or_ma_cross"):
            if ma_cross_up:
                direction, trigger = "BUY", "ma1_cross_ma2"
            elif self.allow_shorts and ma_cross_down:
                direction, trigger = "SELL", "ma1_cross_ma2"
        if direction is None:
            return None

        if self.require_ma_slope:
            if direction == "BUY" and ma_slope <= 0:
                return None
            if direction == "SELL" and ma_slope >= 0:
                return None
            if ma_slope_atr < self.min_ma_slope_atr:
                return None

        if self.require_second_ma_trend and ma2 is not None:
            if direction == "BUY" and (close < ma2 or ma1 < ma2):
                return None
            if direction == "SELL" and (close > ma2 or ma1 > ma2):
                return None

        if price_distance_atr > self.max_price_distance_atr:
            return None

        adx_value = float(current["adx"]) if pd.notna(current["adx"]) else None
        if adx_value is None or adx_value < self.min_adx:
            return None

        rsi_value = float(current["rsi"]) if pd.notna(current["rsi"]) else None
        if rsi_value is None:
            return None
        if direction == "BUY" and not (self.min_rsi_buy <= rsi_value <= self.max_rsi_buy):
            return None
        if direction == "SELL" and not (self.min_rsi_sell <= rsi_value <= self.max_rsi_sell):
            return None

        htf_ok, htf_details = self._mtf_trend_ok(df_htf, direction, "1h")
        macro_ok, macro_details = self._mtf_trend_ok(df_macro, direction, "4h") if df_macro is not None else (True, {})
        aligned_count = int(htf_ok) + int(macro_ok)
        total_count = 2 if df_macro is not None else 1
        if self.require_mtf_alignment and aligned_count < min(self.min_mtf_aligned, total_count):
            return None

        pip = _pip_size(pair, epic)
        risk_pips = max(1.0, round((atr_value * self.stop_loss_atr_mult) / pip, 1))
        reward_pips = max(1.0, round((atr_value * self.take_profit_atr_mult) / pip, 1))
        confidence = self._calculate_confidence(
            trigger=trigger or "",
            ma_slope_atr=ma_slope_atr,
            price_distance_atr=price_distance_atr,
            adx=adx_value,
            mtf_alignment=aligned_count / max(total_count, 1),
            aligned_second_ma=(
                ma2 is None
                or (direction == "BUY" and ma1 > ma2 and close > ma2)
                or (direction == "SELL" and ma1 < ma2 and close < ma2)
            ),
        )
        if confidence < self.min_confidence:
            return None

        self._last_signal_time[epic] = current_timestamp
        return {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair,
            "entry_price": close,
            "risk_pips": risk_pips,
            "reward_pips": reward_pips,
            "stop_loss_pips": risk_pips,
            "take_profit_pips": reward_pips,
            "confidence_score": round(confidence, 3),
            "confidence": round(confidence, 3),
            "signal_timestamp": current_timestamp.isoformat(),
            "timestamp": current_timestamp,
            "version": self.version,
            "monitor_only": True,
            "entry_type": "MOMENTUM",
            "market_regime": "ma_mtf_trend",
            "regime": "ma_mtf_trend",
            "adx": adx_value,
            "rsi": rsi_value,
            "all_timeframes_aligned": aligned_count == total_count,
            "htf_bias": direction,
            "strategy_indicators": {
                "trigger_timeframe": "15m",
                "htf_timeframe": "1h",
                "macro_timeframe": "4h",
                "trigger": trigger,
                "ma_type": self.MA_TYPES.get(self.ma_type, "EMA"),
                "ma_length": self.ma_length,
                "ma_value": ma1,
                "ma_slope_atr": ma_slope_atr,
                "second_ma_type": self.MA_TYPES.get(self.second_ma_type) if self.use_second_ma else None,
                "second_ma_length": self.second_ma_length if self.use_second_ma else None,
                "second_ma_value": ma2,
                "price_distance_atr": price_distance_atr,
                "atr": atr_value,
                "adx": adx_value,
                "rsi": rsi_value,
                "mtf_aligned_count": aligned_count,
                "mtf_total_count": total_count,
                "mtf_alignment_ratio": aligned_count / max(total_count, 1),
                "monitor_only": True,
                "monitor_reason": "ULTIMATE_MA_MTF_FOREX is forward-monitoring only for all epics",
                "htf_details": htf_details,
                "macro_details": macro_details,
            },
        }

    def _mtf_trend_ok(self, df: Optional[pd.DataFrame], direction: str, label: str) -> tuple[bool, Dict[str, Any]]:
        if df is None or df.empty or len(df) < self.htf_ema_period + self.mtf_slope_bars + 2:
            return False, {"timeframe": label, "reason": "insufficient_data"}
        close = df["close"].astype(float)
        ema = close.ewm(span=self.htf_ema_period, adjust=False).mean()
        last_close = float(close.iloc[-1])
        last_ema = float(ema.iloc[-1])
        prior_ema = float(ema.iloc[-(self.mtf_slope_bars + 1)])
        ema_slope = last_ema - prior_ema
        ok = (last_close > last_ema and ema_slope >= 0) if direction == "BUY" else (last_close < last_ema and ema_slope <= 0)
        return ok, {
            "timeframe": label,
            "close": last_close,
            "ema_period": self.htf_ema_period,
            "ema": last_ema,
            "ema_slope": ema_slope,
            "aligned": ok,
        }

    def _moving_average(self, df: pd.DataFrame, ma_type: int, length: int, t3_factor: float) -> pd.Series:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)
        length = max(1, int(length))
        if ma_type == 1:
            return close.rolling(length).mean()
        if ma_type == 2:
            return close.ewm(span=length, adjust=False).mean()
        if ma_type == 3:
            return self._wma(close, length)
        if ma_type == 4:
            half = max(1, int(length / 2))
            root = max(1, int(round(np.sqrt(length))))
            return self._wma(2 * self._wma(close, half) - self._wma(close, length), root)
        if ma_type == 5:
            denom = volume.rolling(length).sum().replace(0, np.nan)
            return (close * volume).rolling(length).sum() / denom
        if ma_type == 6:
            return close.ewm(alpha=1 / length, adjust=False).mean()
        if ma_type == 7:
            ema1 = close.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            ema3 = ema2.ewm(span=length, adjust=False).mean()
            return 3 * (ema1 - ema2) + ema3
        return self._t3(close, length, t3_factor)

    @staticmethod
    def _wma(series: pd.Series, length: int) -> pd.Series:
        weights = np.arange(1, length + 1, dtype=float)
        weight_sum = weights.sum()
        return series.rolling(length).apply(lambda values: float(np.dot(values, weights) / weight_sum), raw=True)

    @staticmethod
    def _t3(series: pd.Series, length: int, factor: float) -> pd.Series:
        def gd(src: pd.Series) -> pd.Series:
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return ema1 * (1 + factor) - ema2 * factor

        return gd(gd(gd(series)))

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _calculate_confidence(
        self,
        trigger: str,
        ma_slope_atr: float,
        price_distance_atr: float,
        adx: float,
        mtf_alignment: float,
        aligned_second_ma: bool,
    ) -> float:
        trigger_score = 0.18 if trigger in ("ma1_cross_ma2", "close_above_green_ma", "close_below_red_ma") else 0.14
        slope_score = min(1.0, ma_slope_atr / 0.12) * 0.24
        location_score = max(0.0, 1.0 - min(price_distance_atr / max(self.max_price_distance_atr, 0.01), 1.0)) * 0.18
        adx_score = min(1.0, adx / 35.0) * 0.16
        mtf_score = min(1.0, mtf_alignment) * 0.18
        second_ma_score = 0.04 if aligned_second_ma else 0.0
        return min(0.95, 0.32 + trigger_score + slope_score + location_score + adx_score + mtf_score + second_ma_score)
