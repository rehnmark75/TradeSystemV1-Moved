"""
Long-only regime-adaptive composite stock strategy.

Combines the better parts of the existing scanner family:
- trend continuation: high retest and MA reclaim
- compression expansion: volatility contraction breakout
- range reversal: Wyckoff spring and bullish RSI divergence

The strategy never emits short signals.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeAdaptiveSignal:
    ticker: str
    signal_timestamp: datetime
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    confidence: float
    quality_tier: str
    regime: str
    entry_component: str
    trend_score: int
    momentum_score: int
    volume_score: int
    pattern_score: int
    confluence_score: int
    factors: List[str]
    raw_data: Dict[str, Any]


class RegimeAdaptiveCompositeStrategy:
    """Long-only strategy that switches entry logic by ticker regime."""

    def __init__(
        self,
        min_confidence: float = 0.62,
        min_rs_percentile: int = 55,
        min_avg_dollar_volume: float = 10_000_000,
        min_price: float = 5.0,
        max_price: float = 1000.0,
        max_stop_pct: float = 8.0,
        atr_stop_mult: float = 1.6,
        take_profit_rr: float = 2.4,
        min_trend_relative_volume: float = 1.0,
        min_range_relative_volume: float = 0.8,
        max_range_adx: float = 30.0,
        max_signal_risk_pct: float = 7.95,
    ):
        self.min_confidence = min_confidence
        self.min_rs_percentile = min_rs_percentile
        self.min_avg_dollar_volume = min_avg_dollar_volume
        self.min_price = min_price
        self.max_price = max_price
        self.max_stop_pct = max_stop_pct
        self.atr_stop_mult = atr_stop_mult
        self.take_profit_rr = take_profit_rr
        self.min_trend_relative_volume = min_trend_relative_volume
        self.min_range_relative_volume = min_range_relative_volume
        self.max_range_adx = max_range_adx
        self.max_signal_risk_pct = max_signal_risk_pct

    def scan(
        self,
        df: pd.DataFrame,
        ticker: str,
        candidate: Optional[Dict[str, Any]] = None,
    ) -> Optional[RegimeAdaptiveSignal]:
        candidate = candidate or {}
        if df.empty or len(df) < 80:
            return None

        data = self._prepare(df)
        current = data.iloc[-1]
        previous = data.iloc[-2]
        close = self._num(current.get("close"))
        atr = self._num(current.get("atr"))
        if close <= 0 or atr <= 0:
            return None

        if close < self.min_price or close > self.max_price:
            return None
        if self._num(candidate.get("avg_dollar_volume")) < self.min_avg_dollar_volume:
            return None

        regime = self._classify_regime(data, candidate)
        setups = []
        if regime == "trend":
            setups.extend([
                self._trend_high_retest(data, candidate),
                self._trend_ma_reclaim(data, candidate),
                self._compression_breakout(data, candidate),
            ])
        elif regime == "compression":
            setups.extend([
                self._compression_breakout(data, candidate),
                self._trend_ma_reclaim(data, candidate),
            ])
        else:
            setups.extend([
                self._range_wyckoff_spring(data, candidate),
                self._range_rsi_divergence(data, candidate),
            ])

        setups = [setup for setup in setups if setup]
        if not setups:
            return None
        setup = max(setups, key=lambda item: item["score"])
        confidence = min(0.95, max(0.0, setup["score"] / 100.0))
        if confidence < self.min_confidence:
            return None

        stop = self._calculate_stop(close, atr, setup)
        risk = close - stop
        if risk <= 0:
            return None
        risk_pct = risk / close * 100
        if risk_pct >= self.max_signal_risk_pct:
            return None
        take_profit = close + risk * self.take_profit_rr
        rr = (take_profit - close) / risk

        return RegimeAdaptiveSignal(
            ticker=ticker,
            signal_timestamp=current["timestamp"] if pd.notna(current.get("timestamp")) else datetime.now(),
            entry_price=round(close, 4),
            stop_loss_price=round(stop, 4),
            take_profit_price=round(take_profit, 4),
            risk_reward_ratio=round(rr, 2),
            confidence=round(confidence, 4),
            quality_tier=self._quality_tier(confidence),
            regime=regime,
            entry_component=setup["component"],
            trend_score=setup["trend_score"],
            momentum_score=setup["momentum_score"],
            volume_score=setup["volume_score"],
            pattern_score=setup["pattern_score"],
            confluence_score=setup["score"],
            factors=setup["factors"],
            raw_data={
                **candidate,
                "regime": regime,
                "entry_component": setup["component"],
                "close": close,
                "atr": atr,
                "adx": self._metric(current, candidate, "adx"),
                "rsi": self._metric(current, candidate, "rsi", "rsi_14"),
                "relative_volume": self._metric(current, candidate, "relative_volume"),
                "rs_percentile": self._num(candidate.get("rs_percentile")),
                "previous_close": self._num(previous.get("close")),
            },
        )

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        if "ema_20" not in data.columns:
            data["ema_20"] = close.ewm(span=20, adjust=False).mean()
        if "ema_50" not in data.columns:
            data["ema_50"] = close.ewm(span=50, adjust=False).mean()
        if "ema_200" not in data.columns:
            data["ema_200"] = close.ewm(span=200, adjust=False).mean()
        if "atr" not in data.columns or data["atr"].isna().all():
            prev_close = close.shift(1)
            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            data["atr"] = tr.rolling(14).mean()
        if "rsi" not in data.columns:
            data["rsi"] = self._rsi(close, 14)
        return data

    def _classify_regime(self, data: pd.DataFrame, candidate: Dict[str, Any]) -> str:
        current = data.iloc[-1]
        close = self._num(current.get("close"))
        atr = self._num(current.get("atr"))
        adx = self._metric(current, candidate, "adx")
        ema50 = self._num(current.get("ema_50"))
        ema200 = self._num(current.get("ema_200"))
        rs = self._num(candidate.get("rs_percentile"))
        rs_improving = str(candidate.get("rs_trend") or "").lower() == "improving"
        ranges = (data["high"].astype(float) - data["low"].astype(float)) / data["close"].astype(float) * 100
        recent_avg_range = float(ranges.iloc[-6:-1].mean())
        older_avg_range = float(ranges.iloc[-26:-6].mean())
        recent_high = float(data["high"].iloc[-16:-1].max())
        recent_low = float(data["low"].iloc[-16:-1].min())
        range_pct = ((recent_high - recent_low) / close) * 100 if close > 0 else 999

        trend_ok = adx >= 20 and close > ema50 and close > ema200 and ema50 >= ema200 * 0.98
        if trend_ok and (rs >= self.min_rs_percentile or rs_improving):
            return "trend"
        compression_ok = (
            older_avg_range > 0
            and recent_avg_range <= older_avg_range * 0.85
            and range_pct <= 8.0
            and close > recent_high
            and atr > 0
        )
        if compression_ok:
            return "compression"
        return "range"

    def _trend_high_retest(self, data: pd.DataFrame, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = data.iloc[-1]
        close = self._num(current.get("close"))
        prev_close = self._num(data.iloc[-2].get("close"))
        sma20 = self._num(candidate.get("sma_20")) or self._num(current.get("ema_20"))
        high_52w = self._num(candidate.get("high_52w")) or float(data["high"].iloc[-252:].max())
        if high_52w <= 0:
            return None
        pct_from_high = ((close - high_52w) / high_52w) * 100
        recent_low = float(data["low"].iloc[-15:].min())
        retest_depth = ((high_52w - recent_low) / high_52w) * 100
        if pct_from_high < -8 or retest_depth > 12 or close <= sma20 or close <= prev_close:
            return None

        score = 58
        score += min(12, int(max(0, pct_from_high + 8) * 1.5))
        score += min(12, int(self._num(candidate.get("rs_percentile")) / 8))
        score += 8 if self._num(candidate.get("macd_histogram")) > 0 else 0
        score += min(8, int(self._metric(current, candidate, "relative_volume") * 3))
        return self._setup(
            "high_retest_reclaim",
            min(score, 100),
            ["52-week high retest", "Reclaimed short-term mean", "Long-only trend continuation"],
            trend=88,
            momentum=70,
            volume=min(100, int(self._metric(current, candidate, "relative_volume") * 45)),
            pattern=86,
        )

    def _trend_ma_reclaim(self, data: pd.DataFrame, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = data.iloc[-1]
        previous = data.iloc[-2]
        close = self._num(current.get("close"))
        ema20 = self._num(current.get("ema_20"))
        ema50 = self._num(current.get("ema_50"))
        ema200 = self._num(current.get("ema_200"))
        atr = self._num(current.get("atr"))
        if atr <= 0:
            return None
        ma_slope_atr = (ema20 - self._num(data["ema_20"].iloc[-4])) / atr
        price_distance_atr = abs(close - ema20) / atr
        reclaim = self._num(previous.get("close")) <= self._num(previous.get("ema_20")) and close > ema20
        line_green_close = close > ema20 and ma_slope_atr > 0.01
        rsi = self._metric(current, candidate, "rsi", "rsi_14")
        relative_volume = self._metric(current, candidate, "relative_volume")
        if not (reclaim or line_green_close):
            return None
        if not (close > ema50 and close > ema200 and price_distance_atr <= 1.5 and 45 <= rsi <= 68):
            return None
        if relative_volume < self.min_trend_relative_volume:
            return None

        score = 56
        score += min(14, int(ma_slope_atr * 250))
        score += 8 if str(candidate.get("rs_trend") or "").lower() == "improving" else 0
        score += min(10, int(self._metric(current, candidate, "adx") / 3))
        score += min(8, int(relative_volume * 3))
        return self._setup(
            "ma_reclaim",
            min(score, 100),
            ["Close above rising EMA20", "Price above EMA50 and EMA200", f"RSI {rsi:.1f}"],
            trend=82,
            momentum=min(100, int(ma_slope_atr * 500)),
            volume=min(100, int(relative_volume * 45)),
            pattern=72,
        )

    def _compression_breakout(self, data: pd.DataFrame, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        lookback = 15
        if len(data) < lookback + 30:
            return None
        current = data.iloc[-1]
        close = self._num(current.get("close"))
        prior_high = float(data["high"].iloc[-lookback - 1:-1].max())
        recent_low = float(data["low"].iloc[-lookback - 1:-1].min())
        recent_range_pct = ((prior_high - recent_low) / close) * 100 if close > 0 else 999
        ranges = (data["high"].astype(float) - data["low"].astype(float)) / data["close"].astype(float) * 100
        recent_avg_range = float(ranges.iloc[-6:-1].mean())
        older_avg_range = float(ranges.iloc[-26:-6].mean())
        volume = self._num(current.get("volume"))
        avg_volume = self._num(candidate.get("avg_volume_20")) or float(data["volume"].iloc[-21:-1].mean())
        rs = self._num(candidate.get("rs_percentile"))
        if close <= prior_high or recent_range_pct > 8 or rs < self.min_rs_percentile:
            return None
        if older_avg_range <= 0 or recent_avg_range > older_avg_range * 0.85:
            return None
        if avg_volume <= 0 or volume < avg_volume * 1.3:
            return None

        score = 60
        score += min(15, int((older_avg_range / max(recent_avg_range, 0.1)) * 4))
        score += min(10, int((volume / avg_volume) * 4))
        score += min(10, int(rs / 10))
        score += 6 if self._num(candidate.get("macd_histogram")) > 0 else 0
        return self._setup(
            "volatility_contraction_breakout",
            min(score, 100),
            [f"Break above {lookback}D range", "Range contracted before breakout", f"Volume {volume / avg_volume:.1f}x average"],
            trend=72,
            momentum=78,
            volume=min(100, int((volume / avg_volume) * 42)),
            pattern=92,
        )

    def _range_wyckoff_spring(self, data: pd.DataFrame, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        lookback = 20
        if len(data) < lookback + 5:
            return None
        current = data.iloc[-1]
        close = self._num(current.get("close"))
        low = self._num(current.get("low"))
        high = self._num(current.get("high"))
        support = float(data["low"].iloc[-lookback - 1:-1].min())
        resistance = float(data["high"].iloc[-lookback - 1:-1].max())
        box_pct = ((resistance - support) / support) * 100 if support > 0 else 999
        day_range = high - low
        close_pos = (close - low) / day_range if day_range > 0 else 0
        volume = self._num(current.get("volume"))
        avg_volume = float(data["volume"].iloc[-21:-1].mean())
        relative_volume = self._metric(current, candidate, "relative_volume")
        adx = self._metric(current, candidate, "adx")
        if not (2 <= box_pct <= 12 and low < support and close > support and close_pos >= 0.55):
            return None
        if adx > self.max_range_adx or relative_volume < self.min_range_relative_volume:
            return None
        if avg_volume > 0 and volume > avg_volume * 1.4:
            return None

        score = 58
        score += min(14, int(close_pos * 16))
        score += 8 if self._metric(current, candidate, "rsi", "rsi_14") < 45 else 0
        score += 8 if close > self._num(current.get("ema_20")) else 0
        score += min(8, int(self._num(candidate.get("rs_percentile")) / 12))
        return self._setup(
            "wyckoff_spring_reclaim",
            min(score, 100),
            ["Range support spring", "Close reclaimed support", f"Box height {box_pct:.1f}%"],
            trend=45,
            momentum=66,
            volume=55,
            pattern=92,
            stop_anchor=low,
        )

    def _range_rsi_divergence(self, data: pd.DataFrame, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        lows = data["low"].astype(float).values
        rsi = data["rsi"].astype(float).values
        current = data.iloc[-1]
        adx = self._metric(current, candidate, "adx")
        relative_volume = self._metric(current, candidate, "relative_volume")
        if adx > self.max_range_adx or relative_volume < self.min_range_relative_volume:
            return None
        swings = self._swing_lows(lows)
        if len(swings) < 2:
            return None
        for current_idx, prior_idx in zip(reversed(swings[1:]), reversed(swings[:-1])):
            if len(lows) - current_idx > 6:
                continue
            if current_idx - prior_idx < 5 or current_idx - prior_idx > 25:
                continue
            if lows[current_idx] < lows[prior_idx] and rsi[current_idx] > rsi[prior_idx] and rsi[current_idx] <= 48:
                close = self._num(current.get("close"))
                if close <= self._num(data.iloc[-2].get("close")):
                    return None
                improvement = rsi[current_idx] - rsi[prior_idx]
                score = 57 + min(14, int(improvement * 2)) + min(8, int(self._num(candidate.get("rs_percentile")) / 12))
                return self._setup(
                    "bullish_rsi_divergence",
                    min(score, 100),
                    [f"RSI higher low {rsi[prior_idx]:.1f}->{rsi[current_idx]:.1f}", "Price made lower low", "Long-only range reversal"],
                    trend=42,
                    momentum=82,
                    volume=min(100, int(self._metric(current, candidate, "relative_volume") * 40)),
                    pattern=88,
                    stop_anchor=float(lows[current_idx]),
                )
        return None

    def _calculate_stop(self, close: float, atr: float, setup: Dict[str, Any]) -> float:
        stop_distance = atr * self.atr_stop_mult
        stop_distance = min(stop_distance, close * self.max_stop_pct / 100)
        stop_distance = max(stop_distance, close * 0.01)
        stop = close - stop_distance
        anchor = setup.get("stop_anchor")
        if anchor:
            stop = min(stop, float(anchor) - atr * 0.25)
            stop = max(stop, close * (1 - self.max_stop_pct / 100))
        return stop

    def _setup(
        self,
        component: str,
        score: int,
        factors: List[str],
        trend: int,
        momentum: int,
        volume: int,
        pattern: int,
        stop_anchor: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "component": component,
            "score": int(score),
            "factors": factors,
            "trend_score": int(max(0, min(100, trend))),
            "momentum_score": int(max(0, min(100, momentum))),
            "volume_score": int(max(0, min(100, volume))),
            "pattern_score": int(max(0, min(100, pattern))),
            "stop_anchor": stop_anchor,
        }

    @staticmethod
    def _swing_lows(lows: np.ndarray, window: int = 3) -> List[int]:
        swings = []
        for idx in range(window, len(lows) - window):
            area = lows[idx - window: idx + window + 1]
            if lows[idx] == np.min(area) and lows[idx] < lows[idx - 1] and lows[idx] < lows[idx + 1]:
                swings.append(idx)
        return swings

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _metric(self, row: pd.Series, candidate: Dict[str, Any], *names: str) -> float:
        for name in names:
            value = row.get(name)
            if pd.notna(value):
                return self._num(value)
            value = candidate.get(name)
            if value is not None:
                return self._num(value)
        return 0.0

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return result if np.isfinite(result) else default
        except (TypeError, ValueError):
            return default

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
