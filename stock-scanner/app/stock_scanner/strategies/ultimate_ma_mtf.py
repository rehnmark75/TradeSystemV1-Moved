"""
Ultimate MA MTF strategy for stocks.

Based on ChrisMoody's CM_Ultimate_MA_MTF_V2 idea: configurable moving-average
type and MA direction coloring. The default scanner rule is deliberately
long-only: BUY when price first closes above a green/rising MA line while price
is above EMA50/EMA200 and relative strength is improving.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class UltimateMASignal:
    ticker: str
    signal_type: str
    signal_timestamp: datetime
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    confidence: float
    quality_tier: str

    trigger: str
    ma_type: str
    ma_length: int
    ma_value: float
    ma_slope_atr: float
    second_ma_type: Optional[str] = None
    second_ma_length: Optional[int] = None
    second_ma_value: Optional[float] = None
    price_distance_atr: Optional[float] = None
    adx: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    relative_volume: Optional[float] = None
    rs_percentile: Optional[float] = None
    rs_trend: Optional[str] = None
    sector: Optional[str] = None


class UltimateMAMTFStrategy:
    """Configurable MA color/close strategy for daily stock scans."""

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

    def __init__(
        self,
        ma_type: int = 2,
        ma_length: int = 20,
        t3_factor: float = 0.7,
        use_second_ma: bool = False,
        second_ma_type: int = 2,
        second_ma_length: int = 50,
        second_t3_factor: float = 0.7,
        smoothing_bars: int = 2,
        trigger_mode: str = "line_color_close",
        require_ma_slope: bool = True,
        require_second_ma_trend: bool = True,
        require_ema50_ema200: bool = True,
        require_rs_improving: bool = True,
        max_price_distance_atr: float = 1.5,
        min_ma_slope_atr: float = 0.01,
        min_adx: float = 15.0,
        min_rsi: float = 52.5,
        max_rsi: float = 62.5,
        min_relative_volume: float = 0.8,
        stop_loss_atr_mult: float = 1.5,
        take_profit_atr_mult: float = 2.5,
        min_confidence: float = 0.55,
        min_quality_tier: str = "C",
        allow_shorts: bool = False,
    ):
        self.ma_type = ma_type
        self.ma_length = ma_length
        self.t3_factor = t3_factor
        self.use_second_ma = use_second_ma
        self.second_ma_type = second_ma_type
        self.second_ma_length = second_ma_length
        self.second_t3_factor = second_t3_factor
        self.smoothing_bars = max(1, int(smoothing_bars))
        self.trigger_mode = trigger_mode
        self.require_ma_slope = require_ma_slope
        self.require_second_ma_trend = require_second_ma_trend
        self.require_ema50_ema200 = require_ema50_ema200
        self.require_rs_improving = require_rs_improving
        self.max_price_distance_atr = max_price_distance_atr
        self.min_ma_slope_atr = min_ma_slope_atr
        self.min_adx = min_adx
        self.min_rsi = min_rsi
        self.max_rsi = max_rsi
        self.min_relative_volume = min_relative_volume
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.min_confidence = min_confidence
        self.min_quality_tier = min_quality_tier
        self.allow_shorts = allow_shorts
        self.logger = logging.getLogger(__name__)

    def scan(
        self,
        df: pd.DataFrame,
        ticker: str,
        sector: Optional[str] = None,
    ) -> Optional[UltimateMASignal]:
        min_bars = max(self.ma_length, self.second_ma_length if self.use_second_ma else 0, 50) + 10
        if df.empty or len(df) < min_bars:
            return None

        required = ["open", "high", "low", "close", "volume", "atr", "adx", "relative_volume", "ema_50", "ema_200"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.logger.warning("%s: Missing columns for ultimate MA strategy: %s", ticker, missing)
            return None

        data = df.copy()
        data["ult_ma1"] = self._moving_average(
            data,
            ma_type=self.ma_type,
            length=self.ma_length,
            t3_factor=self.t3_factor,
        )
        if self.use_second_ma:
            data["ult_ma2"] = self._moving_average(
                data,
                ma_type=self.second_ma_type,
                length=self.second_ma_length,
                t3_factor=self.second_t3_factor,
            )

        current = data.iloc[-1]
        previous = data.iloc[-2]
        ma1 = current["ult_ma1"]
        atr = current["atr"]
        if pd.isna(ma1) or pd.isna(atr) or atr <= 0:
            return None

        close = float(current["close"])
        open_price = float(current["open"])
        atr = float(atr)
        ma1 = float(ma1)
        ma1_prev = float(previous["ult_ma1"]) if pd.notna(previous["ult_ma1"]) else np.nan
        ma1_smooth = data["ult_ma1"].iloc[-(self.smoothing_bars + 1)]
        if pd.isna(ma1_prev) or pd.isna(ma1_smooth):
            return None

        ma_slope = ma1 - float(ma1_smooth)
        ma_slope_atr = abs(ma_slope) / atr if atr > 0 else 0.0
        price_distance_atr = abs(close - ma1) / atr if atr > 0 else None
        previous_close = float(previous["close"])
        line_green = ma1 >= float(ma1_smooth)
        line_red = ma1 < float(ma1_smooth)
        previous_ma1_smooth = data["ult_ma1"].iloc[-(self.smoothing_bars + 2)]
        previous_line_green = pd.notna(previous_ma1_smooth) and ma1_prev >= float(previous_ma1_smooth)
        previous_line_red = pd.notna(previous_ma1_smooth) and ma1_prev < float(previous_ma1_smooth)
        line_color_buy = close > ma1 and line_green and not (previous_close > ma1_prev and previous_line_green)
        line_color_sell = close < ma1 and line_red and not (previous_close < ma1_prev and previous_line_red)
        price_cross_up = open_price < ma1 and close > ma1
        price_cross_down = open_price > ma1 and close < ma1

        ma_cross_up = False
        ma_cross_down = False
        ma2 = None
        if self.use_second_ma:
            ma2_val = current["ult_ma2"]
            prev_ma2_val = previous["ult_ma2"]
            if pd.isna(ma2_val) or pd.isna(prev_ma2_val):
                return None
            ma2 = float(ma2_val)
            prev_ma2 = float(prev_ma2_val)
            ma_cross_up = ma1_prev <= prev_ma2 and ma1 > ma2
            ma_cross_down = ma1_prev >= prev_ma2 and ma1 < ma2

        signal_type = None
        trigger = None
        if self.trigger_mode == "line_color_close" and line_color_buy:
            signal_type, trigger = "BUY", "close_above_green_ma"
        elif self.trigger_mode == "line_color_close" and self.allow_shorts and line_color_sell:
            signal_type, trigger = "SELL", "close_below_red_ma"
        elif self.trigger_mode in ("price_cross", "price_or_ma_cross") and price_cross_up:
            signal_type, trigger = "BUY", "price_cross_ma1"
        elif self.trigger_mode in ("price_cross", "price_or_ma_cross") and self.allow_shorts and price_cross_down:
            signal_type, trigger = "SELL", "price_cross_ma1"

        if signal_type is None and self.trigger_mode in ("ma_cross", "price_or_ma_cross"):
            if ma_cross_up:
                signal_type, trigger = "BUY", "ma1_cross_ma2"
            elif self.allow_shorts and ma_cross_down:
                signal_type, trigger = "SELL", "ma1_cross_ma2"

        if signal_type is None:
            return None

        if self.require_ema50_ema200:
            ema50 = float(current["ema_50"]) if pd.notna(current.get("ema_50")) else None
            ema200 = float(current["ema_200"]) if pd.notna(current.get("ema_200")) else None
            if ema50 is None or ema200 is None:
                return None
            if signal_type == "BUY" and (close <= ema50 or close <= ema200):
                return None
            if signal_type == "SELL" and (close >= ema50 or close >= ema200):
                return None

        rs_trend = str(current.get("rs_trend") or "").lower()
        if self.require_rs_improving:
            if signal_type == "BUY" and rs_trend != "improving":
                return None
            if signal_type == "SELL" and rs_trend != "deteriorating":
                return None

        if self.require_ma_slope:
            if signal_type == "BUY" and ma_slope <= 0:
                return None
            if signal_type == "SELL" and ma_slope >= 0:
                return None
            if ma_slope_atr < self.min_ma_slope_atr:
                return None

        if self.require_second_ma_trend and self.use_second_ma and ma2 is not None:
            if signal_type == "BUY" and close < ma2:
                return None
            if signal_type == "SELL" and close > ma2:
                return None

        if price_distance_atr is not None and price_distance_atr > self.max_price_distance_atr:
            return None

        adx = float(current["adx"]) if pd.notna(current["adx"]) else None
        if adx is None or adx < self.min_adx:
            return None

        rsi = float(current["rsi"]) if pd.notna(current.get("rsi")) else None
        if rsi is None or rsi < self.min_rsi or rsi > self.max_rsi:
            return None

        relative_volume = float(current["relative_volume"]) if pd.notna(current["relative_volume"]) else None
        if relative_volume is not None and relative_volume < self.min_relative_volume:
            return None
        rs_percentile = float(current["rs_percentile"]) if pd.notna(current.get("rs_percentile")) else None

        entry_price = close
        if signal_type == "BUY":
            stop_loss_price = entry_price - atr * self.stop_loss_atr_mult
            take_profit_price = entry_price + atr * self.take_profit_atr_mult
        else:
            stop_loss_price = entry_price + atr * self.stop_loss_atr_mult
            take_profit_price = entry_price - atr * self.take_profit_atr_mult

        risk = abs(entry_price - stop_loss_price)
        reward = abs(take_profit_price - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0.0

        confidence = self._calculate_confidence(
            trigger=trigger,
            ma_slope_atr=ma_slope_atr,
            price_distance_atr=price_distance_atr or 0.0,
            adx=adx,
            relative_volume=relative_volume or 1.0,
            aligned_second_ma=(
                not self.use_second_ma
                or ma2 is None
                or (signal_type == "BUY" and ma1 > ma2 and close > ma2)
                or (signal_type == "SELL" and ma1 < ma2 and close < ma2)
            ),
        )
        if confidence < self.min_confidence:
            return None

        quality_tier = self._quality_tier(confidence)
        if self._tier_rank(quality_tier) < self._tier_rank(self.min_quality_tier):
            return None

        return UltimateMASignal(
            ticker=ticker,
            signal_type=signal_type,
            signal_timestamp=current["timestamp"] if pd.notna(current.get("timestamp")) else datetime.now(),
            entry_price=round(entry_price, 4),
            stop_loss_price=round(stop_loss_price, 4),
            take_profit_price=round(take_profit_price, 4),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            confidence=round(confidence, 4),
            quality_tier=quality_tier,
            trigger=trigger or "unknown",
            ma_type=self.MA_TYPES.get(self.ma_type, "EMA"),
            ma_length=self.ma_length,
            ma_value=round(ma1, 4),
            ma_slope_atr=round(ma_slope_atr, 4),
            second_ma_type=self.MA_TYPES.get(self.second_ma_type) if self.use_second_ma else None,
            second_ma_length=self.second_ma_length if self.use_second_ma else None,
            second_ma_value=round(ma2, 4) if ma2 is not None else None,
            price_distance_atr=round(price_distance_atr, 4) if price_distance_atr is not None else None,
            adx=round(adx, 2) if adx is not None else None,
            rsi=round(rsi, 2) if rsi is not None else None,
            atr=round(atr, 4),
            relative_volume=round(relative_volume, 2) if relative_volume is not None else None,
            rs_percentile=round(rs_percentile, 2) if rs_percentile is not None else None,
            rs_trend=rs_trend or None,
            sector=sector,
        )

    def _moving_average(
        self,
        df: pd.DataFrame,
        ma_type: int,
        length: int,
        t3_factor: float,
    ) -> pd.Series:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
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

    def _calculate_confidence(
        self,
        trigger: str,
        ma_slope_atr: float,
        price_distance_atr: float,
        adx: float,
        relative_volume: float,
        aligned_second_ma: bool,
    ) -> float:
        trigger_score = 0.20 if trigger in ("ma1_cross_ma2", "close_above_green_ma", "close_below_red_ma") else 0.15
        slope_score = min(1.0, ma_slope_atr / 0.10) * 0.25
        location_score = max(0.0, 1.0 - min(price_distance_atr / max(self.max_price_distance_atr, 0.01), 1.0)) * 0.20
        adx_score = min(1.0, adx / 35.0) * 0.15
        volume_score = min(1.0, relative_volume / 2.0) * 0.15
        alignment_score = 0.05 if aligned_second_ma else 0.0
        return min(0.95, 0.35 + trigger_score + slope_score + location_score + adx_score + volume_score + alignment_score)

    @staticmethod
    def _quality_tier(confidence: float) -> str:
        if confidence >= 0.85:
            return "A+"
        if confidence >= 0.70:
            return "A"
        if confidence >= 0.60:
            return "B"
        if confidence >= 0.50:
            return "C"
        return "D"

    @staticmethod
    def _tier_rank(tier: str) -> int:
        return {"D": 0, "C": 1, "B": 2, "A": 3, "A+": 4}.get(tier, 0)

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy_name": "ULTIMATE_MA_MTF",
            "ma_type": self.ma_type,
            "ma_length": self.ma_length,
            "t3_factor": self.t3_factor,
            "use_second_ma": self.use_second_ma,
            "second_ma_type": self.second_ma_type,
            "second_ma_length": self.second_ma_length,
            "second_t3_factor": self.second_t3_factor,
            "smoothing_bars": self.smoothing_bars,
            "trigger_mode": self.trigger_mode,
            "require_ma_slope": self.require_ma_slope,
            "require_second_ma_trend": self.require_second_ma_trend,
            "require_ema50_ema200": self.require_ema50_ema200,
            "require_rs_improving": self.require_rs_improving,
            "max_price_distance_atr": self.max_price_distance_atr,
            "min_ma_slope_atr": self.min_ma_slope_atr,
            "min_adx": self.min_adx,
            "min_rsi": self.min_rsi,
            "max_rsi": self.max_rsi,
            "min_relative_volume": self.min_relative_volume,
            "stop_loss_atr_mult": self.stop_loss_atr_mult,
            "take_profit_atr_mult": self.take_profit_atr_mult,
            "min_confidence": self.min_confidence,
            "min_quality_tier": self.min_quality_tier,
            "allow_shorts": self.allow_shorts,
        }
