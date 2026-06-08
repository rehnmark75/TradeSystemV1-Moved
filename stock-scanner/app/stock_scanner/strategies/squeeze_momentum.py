"""
Squeeze Momentum Strategy for Stocks.

LazyBear-style Bollinger/Keltner squeeze release with trend, ADX, momentum,
and volume filters. Designed for daily stock backtests first, then scanner use.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class SqueezeMomentumSignal:
    ticker: str
    signal_type: str
    signal_timestamp: datetime
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    confidence: float
    quality_tier: str

    squeeze_count: int
    momentum: float
    momentum_slope: float
    momentum_slope_atr: float
    adx: Optional[float]
    rsi: Optional[float]
    atr: float
    sector: Optional[str] = None
    volume: Optional[int] = None
    relative_volume: Optional[float] = None

    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_100: Optional[float] = None
    ema_200: Optional[float] = None
    pullback_percent: Optional[float] = None


class SqueezeMomentumStrategy:
    """Daily squeeze-release momentum strategy for stocks."""

    def __init__(
        self,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
        squeeze_min_bars: int = 3,
        squeeze_lookback_bars: int = 8,
        require_release_bar: bool = True,
        min_momentum_slope_atr: float = 0.03,
        min_adx: float = 18.0,
        require_adx_rising: bool = True,
        min_relative_volume: float = 1.0,
        trend_ema: str = "ema_50",
        stop_loss_atr_mult: float = 1.5,
        take_profit_atr_mult: float = 2.5,
        min_confidence: float = 0.55,
        min_quality_tier: str = "C",
        allow_shorts: bool = True,
    ):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.squeeze_min_bars = squeeze_min_bars
        self.squeeze_lookback_bars = squeeze_lookback_bars
        self.require_release_bar = require_release_bar
        self.min_momentum_slope_atr = min_momentum_slope_atr
        self.min_adx = min_adx
        self.require_adx_rising = require_adx_rising
        self.min_relative_volume = min_relative_volume
        self.trend_ema = trend_ema
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
    ) -> Optional[SqueezeMomentumSignal]:
        min_bars = max(self.bb_length, self.kc_length, 20) + self.squeeze_lookback_bars + 8
        if df.empty or len(df) < min_bars:
            return None

        required = ["high", "low", "close", "atr", "adx", self.trend_ema]
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.logger.warning("%s: Missing columns for squeeze strategy: %s", ticker, missing)
            return None

        data = self._add_squeeze_columns(df)
        current = data.iloc[-1]
        previous = data.iloc[-2]

        if pd.isna(current["sqz_momentum"]) or pd.isna(current["atr"]) or current["atr"] <= 0:
            return None

        prior_window = data["sqz_on"].iloc[-(self.squeeze_lookback_bars + 1):-1]
        squeeze_count = int(prior_window.sum())
        release_ok = bool(current["sqz_off"]) and (
            not self.require_release_bar or bool(previous["sqz_on"])
        )
        if squeeze_count < self.squeeze_min_bars or not release_ok:
            return None

        momentum = float(current["sqz_momentum"])
        slope = float(current["sqz_momentum_slope"])
        atr = float(current["atr"])
        slope_atr = abs(slope) / atr if atr > 0 else 0.0

        signal_type = None
        if momentum > 0 and slope > 0:
            signal_type = "BUY"
        elif self.allow_shorts and momentum < 0 and slope < 0:
            signal_type = "SELL"
        if signal_type is None:
            return None

        if slope_atr < self.min_momentum_slope_atr:
            return None

        adx = float(current["adx"]) if pd.notna(current["adx"]) else None
        prev_adx = float(previous["adx"]) if pd.notna(previous["adx"]) else None
        adx_rising = adx is not None and prev_adx is not None and adx > prev_adx
        if adx is None or (adx < self.min_adx and (self.require_adx_rising and not adx_rising)):
            return None

        close = float(current["close"])
        trend_value = float(current[self.trend_ema])
        if signal_type == "BUY" and close <= trend_value:
            return None
        if signal_type == "SELL" and close >= trend_value:
            return None

        if pd.notna(current.get("relative_volume")) and float(current["relative_volume"]) < self.min_relative_volume:
            return None

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
            squeeze_count=squeeze_count,
            slope_atr=slope_atr,
            adx=adx or 0.0,
            relative_volume=float(current["relative_volume"]) if pd.notna(current.get("relative_volume")) else 1.0,
        )
        if confidence < self.min_confidence:
            return None

        quality_tier = self._quality_tier(confidence)
        if self._tier_rank(quality_tier) < self._tier_rank(self.min_quality_tier):
            return None

        return SqueezeMomentumSignal(
            ticker=ticker,
            signal_type=signal_type,
            signal_timestamp=current["timestamp"] if pd.notna(current.get("timestamp")) else datetime.now(),
            entry_price=round(entry_price, 4),
            stop_loss_price=round(stop_loss_price, 4),
            take_profit_price=round(take_profit_price, 4),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            confidence=round(confidence, 4),
            quality_tier=quality_tier,
            squeeze_count=squeeze_count,
            momentum=round(momentum, 4),
            momentum_slope=round(slope, 4),
            momentum_slope_atr=round(slope_atr, 4),
            adx=round(adx, 2) if adx is not None else None,
            rsi=round(float(current["rsi"]), 2) if pd.notna(current.get("rsi")) else None,
            atr=round(atr, 4),
            sector=sector,
            volume=int(current["volume"]) if pd.notna(current.get("volume")) else None,
            relative_volume=round(float(current["relative_volume"]), 2)
                if pd.notna(current.get("relative_volume")) else None,
            ema_20=round(float(current["ema_20"]), 4) if pd.notna(current.get("ema_20")) else None,
            ema_50=round(float(current["ema_50"]), 4) if pd.notna(current.get("ema_50")) else None,
            ema_100=round(float(current["ema_100"]), 4) if pd.notna(current.get("ema_100")) else None,
            ema_200=round(float(current["ema_200"]), 4) if pd.notna(current.get("ema_200")) else None,
            pullback_percent=round(float(current.get("pct_from_ema20", 0.0)), 4)
                if pd.notna(current.get("pct_from_ema20")) else None,
        )

    def _add_squeeze_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)

        basis = close.rolling(self.bb_length).mean()
        dev = self.bb_mult * close.rolling(self.bb_length).std()
        out["sqz_bb_upper"] = basis + dev
        out["sqz_bb_lower"] = basis - dev

        tr = self._true_range(out)
        kc_ma = close.rolling(self.kc_length).mean()
        range_ma = tr.rolling(self.kc_length).mean()
        out["sqz_kc_upper"] = kc_ma + range_ma * self.kc_mult
        out["sqz_kc_lower"] = kc_ma - range_ma * self.kc_mult

        out["sqz_on"] = (out["sqz_bb_lower"] > out["sqz_kc_lower"]) & (out["sqz_bb_upper"] < out["sqz_kc_upper"])
        out["sqz_off"] = (out["sqz_bb_lower"] < out["sqz_kc_lower"]) & (out["sqz_bb_upper"] > out["sqz_kc_upper"])

        highest_high = high.rolling(self.kc_length).max()
        lowest_low = low.rolling(self.kc_length).min()
        mean_extreme = (highest_high + lowest_low) / 2.0
        mean_source = (mean_extreme + close.rolling(self.kc_length).mean()) / 2.0
        linreg_source = close - mean_source
        out["sqz_momentum"] = linreg_source.rolling(self.kc_length).apply(self._linreg_endpoint, raw=True)
        out["sqz_momentum_slope"] = out["sqz_momentum"].diff()
        return out

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_close = df["close"].astype(float).shift(1)
        return pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

    @staticmethod
    def _linreg_endpoint(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values).any():
            return np.nan
        x = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x, values.astype(float), 1)
        return float(intercept + slope * (len(values) - 1))

    def _calculate_confidence(
        self,
        squeeze_count: int,
        slope_atr: float,
        adx: float,
        relative_volume: float,
    ) -> float:
        compression_score = min(1.0, squeeze_count / max(self.squeeze_lookback_bars, 1)) * 0.30
        momentum_score = min(1.0, slope_atr / 0.15) * 0.35
        adx_score = min(1.0, adx / 35.0) * 0.20
        volume_score = min(1.0, relative_volume / 2.0) * 0.15
        return min(0.95, 0.45 + compression_score + momentum_score + adx_score + volume_score)

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
            "strategy_name": "SQUEEZE_MOMENTUM",
            "bb_length": self.bb_length,
            "bb_mult": self.bb_mult,
            "kc_length": self.kc_length,
            "kc_mult": self.kc_mult,
            "squeeze_min_bars": self.squeeze_min_bars,
            "squeeze_lookback_bars": self.squeeze_lookback_bars,
            "require_release_bar": self.require_release_bar,
            "min_momentum_slope_atr": self.min_momentum_slope_atr,
            "min_adx": self.min_adx,
            "require_adx_rising": self.require_adx_rising,
            "min_relative_volume": self.min_relative_volume,
            "trend_ema": self.trend_ema,
            "stop_loss_atr_mult": self.stop_loss_atr_mult,
            "take_profit_atr_mult": self.take_profit_atr_mult,
            "min_confidence": self.min_confidence,
            "min_quality_tier": self.min_quality_tier,
            "allow_shorts": self.allow_shorts,
        }
