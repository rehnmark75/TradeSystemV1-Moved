#!/usr/bin/env python3
"""
Confluence Stack strategy.

Single strategy stream that combines active entry-model archetypes:
- trend confluence from Donchian, KAMA, SMC pullback, and liquidity sweep entries
- low-regime fade confluence from BB/RSI reversion, range fade, and impulse fade
- selected standalone high-quality fallback entries

The strategy name intentionally contains no pair/epic identifier.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .strategy_registry import StrategyInterface, register_strategy


logger = logging.getLogger(__name__)


def _pip(pair_or_epic: str) -> float:
    return 0.01 if "JPY" in (pair_or_epic or "").upper() else 0.0001


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.astype(float).ewm(span=n, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].astype(float).shift(1)
    tr = pd.concat(
        [
            df["high"].astype(float) - df["low"].astype(float),
            (df["high"].astype(float) - pc).abs(),
            (df["low"].astype(float) - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["high"].astype(float).diff()
    down_move = -df["low"].astype(float).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _kama(close: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30) -> Tuple[pd.Series, pd.Series]:
    close = close.astype(float)
    change = (close - close.shift(er_period)).abs()
    volatility = close.diff().abs().rolling(er_period).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = pd.Series(index=close.index, dtype=float)
    if close.empty:
        return out, er
    out.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        out.iloc[i] = out.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - out.iloc[i - 1])
    return out, er


def _as_utc(ts) -> datetime:
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@register_strategy("CONFLUENCE_STACK")
class ConfluenceStackStrategy(StrategyInterface):
    """One strategy made from multiple entry-model modules."""

    def __init__(self, config=None, logger=None, db_manager=None, config_override: Optional[Dict] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.config_override = config_override or {}
        self.cooldown_minutes = int(float(self.config_override.get("cooldown_minutes", 360)))
        self._last_signal_at: Dict[str, datetime] = {}

    @property
    def strategy_name(self) -> str:
        return "CONFLUENCE_STACK"

    def get_required_timeframes(self) -> List[str]:
        return ["5m", "15m", "1h", "4h"]

    def reset_cooldowns(self) -> None:
        self._last_signal_at.clear()

    def detect_signal(
        self,
        df_trigger: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        epic: str = "",
        pair: str = "",
        spread_pips: float = 1.5,
        current_timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[Dict]:
        df_5m = df_trigger
        if df_5m is None or df_15m is None or df_1h is None or df_4h is None:
            return None
        if len(df_5m) < 220 or len(df_15m) < 80 or len(df_1h) < 80 or len(df_4h) < 60:
            return None

        now = self._resolve_now(df_5m, current_timestamp)
        cooldown_key = epic or pair
        last = self._last_signal_at.get(cooldown_key)
        if last is not None and now - last < timedelta(minutes=self.cooldown_minutes):
            return None

        pip = _pip(pair or epic)
        modules = self._module_sides(df_5m, df_15m, df_1h, df_4h, pip)
        side, source, active_modules = self._select_side(modules, df_1h)
        if side == 0:
            return None

        close = float(df_5m["close"].astype(float).iloc[-1])
        direction = "BUY" if side == 1 else "SELL"
        sl_pips, tp_pips, hold_hint = self._risk_for_source(source)
        stop_loss = close - side * sl_pips * pip
        take_profit = close + side * tp_pips * pip
        confidence = self._confidence(source, active_modules)

        self._last_signal_at[cooldown_key] = now

        return {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair,
            "entry_price": close,
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "stop_loss_pips": sl_pips,
            "take_profit_pips": tp_pips,
            "risk_pips": sl_pips,
            "reward_pips": tp_pips,
            "risk_reward_ratio": round(tp_pips / sl_pips, 4) if sl_pips else 0,
            "confidence": confidence,
            "confidence_score": confidence,
            "signal_timestamp": now.isoformat(),
            "timestamp": now,
            "version": "1.0.0",
            "market_regime": source,
            "regime": source,
            "monitor_only": True,
            "strategy_indicators": {
                "entry_stack_source": source,
                "entry_modules": active_modules,
                "hold_hint_5m_bars": hold_hint,
                "cooldown_minutes": self.cooldown_minutes,
                "tier1_ema": {"timeframe": "4h"},
                "tier2_swing": {"timeframe": "15m"},
                "tier3_entry": {"timeframe": "5m"},
            },
        }

    def _module_sides(
        self,
        df5: pd.DataFrame,
        df15: pd.DataFrame,
        df1h: pd.DataFrame,
        df4h: pd.DataFrame,
        pip: float,
    ) -> Dict[str, int]:
        sides: Dict[str, int] = {}

        c5 = df5["close"].astype(float)
        c15 = df15["close"].astype(float)
        c1 = df1h["close"].astype(float)
        c4 = df4h["close"].astype(float)

        ma15 = c15.rolling(20).mean()
        sd15 = c15.rolling(20).std()
        upper15 = ma15 + 2.0 * sd15
        lower15 = ma15 - 2.0 * sd15
        r15 = _rsi(c15, 14)
        ax15 = _adx(df15, 14)
        ax1_at_15 = _adx(df1h, 14).reindex(df15.index, method="ffill")
        low_adx = bool((ax15.iloc[-1] <= 24) and (ax1_at_15.iloc[-1] <= 24))

        sides["MEAN_REVERSION_reject"] = 0
        if low_adx:
            if c15.iloc[-2] <= lower15.iloc[-2] and r15.iloc[-2] <= 30 and c15.iloc[-1] > lower15.iloc[-1]:
                sides["MEAN_REVERSION_reject"] = 1
            elif c15.iloc[-2] >= upper15.iloc[-2] and r15.iloc[-2] >= 70 and c15.iloc[-1] < upper15.iloc[-1]:
                sides["MEAN_REVERSION_reject"] = -1

        ma5 = c5.rolling(20).mean()
        sd5 = c5.rolling(20).std()
        upper5 = ma5 + 2.0 * sd5
        lower5 = ma5 - 2.0 * sd5
        r5 = _rsi(c5, 14)
        prior_hi = df5["high"].astype(float).rolling(48).max().shift(1)
        prior_lo = df5["low"].astype(float).rolling(48).min().shift(1)
        bw = (upper5.iloc[-1] - lower5.iloc[-1]) / pip
        bias1 = self._bias_1h(df1h)
        sides["RANGE_FADE_local_extreme"] = 0
        if 8 <= bw <= 80:
            dist_low = (c5.iloc[-1] - prior_lo.iloc[-1]) / pip
            dist_high = (prior_hi.iloc[-1] - c5.iloc[-1]) / pip
            if c5.iloc[-1] <= lower5.iloc[-1] and r5.iloc[-1] <= 32 and dist_low <= 8 and bias1 >= 0:
                sides["RANGE_FADE_local_extreme"] = 1
            elif c5.iloc[-1] >= upper5.iloc[-1] and r5.iloc[-1] >= 68 and dist_high <= 8 and bias1 <= 0:
                sides["RANGE_FADE_local_extreme"] = -1

        hi20 = df1h["high"].astype(float).rolling(20).max().shift(1)
        sides["DONCHIAN_TURTLE_20_long_only"] = 1 if c1.iloc[-1] > hi20.iloc[-1] else 0

        k, er = _kama(c5)
        ema200 = _ema(c5, 200)
        macd = _ema(c5, 12) - _ema(c5, 26)
        hist = macd - _ema(macd, 9)
        sides["KAMA_V2_cross_er_confirmed"] = 0
        if c5.iloc[-1] > k.iloc[-1] and c5.iloc[-2] <= k.iloc[-2] and er.iloc[-1] >= 0.35 and c5.iloc[-1] > ema200.iloc[-1] and hist.iloc[-1] > 0 and r5.iloc[-1] < 70:
            sides["KAMA_V2_cross_er_confirmed"] = 1
        elif c5.iloc[-1] < k.iloc[-1] and c5.iloc[-2] >= k.iloc[-2] and er.iloc[-1] >= 0.35 and c5.iloc[-1] < ema200.iloc[-1] and hist.iloc[-1] < 0 and r5.iloc[-1] > 30:
            sides["KAMA_V2_cross_er_confirmed"] = -1

        a5 = _atr(df5, 14) / pip
        body = (c5.iloc[-1] - float(df5["open"].astype(float).iloc[-1])) / pip
        hour = df5.index[-1].hour if isinstance(df5.index, pd.DatetimeIndex) else _as_utc(df5.iloc[-1].get("start_time")).hour
        sides["IMPULSE_FADE_late_us"] = 0
        if 18 <= hour <= 22 and abs(body) >= 2.2 * a5.iloc[-1] and a5.iloc[-1] <= 18:
            sides["IMPULSE_FADE_late_us"] = 1 if body < 0 else -1

        e4 = _ema(c4, 50)
        bias4 = 1 if c4.iloc[-1] > e4.iloc[-1] else -1
        ph = df15["high"].astype(float).rolling(24).max().shift(1)
        pl = df15["low"].astype(float).rolling(24).min().shift(1)
        rng15 = (df15["high"].astype(float).iloc[-1] - df15["low"].astype(float).iloc[-1])
        body_pct = abs(c15.iloc[-1] - float(df15["open"].astype(float).iloc[-1])) / max(rng15, 1e-9)
        sides["SMC_MOMENTUM_sweep_reject"] = 0
        sweep_low = 3 <= (pl.iloc[-1] - df15["low"].astype(float).iloc[-1]) / pip <= 18 and c15.iloc[-1] > pl.iloc[-1] and c15.iloc[-1] > df15["open"].astype(float).iloc[-1]
        sweep_high = 3 <= (df15["high"].astype(float).iloc[-1] - ph.iloc[-1]) / pip <= 18 and c15.iloc[-1] < ph.iloc[-1] and c15.iloc[-1] < df15["open"].astype(float).iloc[-1]
        if body_pct >= 0.25:
            if sweep_low and bias4 >= 0:
                sides["SMC_MOMENTUM_sweep_reject"] = 1
            elif sweep_high and bias4 <= 0:
                sides["SMC_MOMENTUM_sweep_reject"] = -1

        sides["SMC_SIMPLE_trend_pullback_resume"] = 0
        e1 = _ema(c1, 50)
        e1_slope = e1.iloc[-1] - e1.iloc[-7]
        pull = (c15.iloc[-1] - c15.iloc[-4]) / pip
        candle_body = (c15.iloc[-1] - float(df15["open"].astype(float).iloc[-1])) / pip
        if c1.iloc[-1] > e1.iloc[-1] and e1_slope > 0 and pull < -5 and candle_body > 1.5 and 40 < r15.iloc[-1] < 65:
            sides["SMC_SIMPLE_trend_pullback_resume"] = 1
        elif c1.iloc[-1] < e1.iloc[-1] and e1_slope < 0 and pull > 5 and candle_body < -1.5 and 35 < r15.iloc[-1] < 60:
            sides["SMC_SIMPLE_trend_pullback_resume"] = -1

        return sides

    def _select_side(self, modules: Dict[str, int], df1h: pd.DataFrame) -> Tuple[int, str, List[str]]:
        trend_names = ["DONCHIAN_TURTLE_20_long_only", "KAMA_V2_cross_er_confirmed", "SMC_SIMPLE_trend_pullback_resume", "SMC_MOMENTUM_sweep_reject"]
        fade_names = ["MEAN_REVERSION_reject", "RANGE_FADE_local_extreme", "IMPULSE_FADE_late_us"]

        trend_long = [n for n in trend_names if modules.get(n) == 1]
        trend_short = [n for n in trend_names if modules.get(n) == -1]
        if len(trend_long) >= 2 and len(trend_long) > len(trend_short):
            return 1, "trend_confluence", trend_long
        if len(trend_short) >= 2 and len(trend_short) > len(trend_long):
            return -1, "trend_confluence", trend_short

        ax = _adx(df1h, 14).iloc[-1]
        if pd.notna(ax) and ax < 22:
            fade_long = [n for n in fade_names if modules.get(n) == 1]
            fade_short = [n for n in fade_names if modules.get(n) == -1]
            if len(fade_long) >= 2 and len(fade_long) > len(fade_short):
                return 1, "low_regime_fade_confluence", fade_long
            if len(fade_short) >= 2 and len(fade_short) > len(fade_long):
                return -1, "low_regime_fade_confluence", fade_short

        for name in ["DONCHIAN_TURTLE_20_long_only", "SMC_MOMENTUM_sweep_reject", "IMPULSE_FADE_late_us", "MEAN_REVERSION_reject"]:
            side = modules.get(name, 0)
            if side:
                return side, f"fallback_{name}", [name]
        return 0, "none", []

    def _risk_for_source(self, source: str) -> Tuple[float, float, int]:
        if source == "low_regime_fade_confluence" or "IMPULSE_FADE" in source:
            return 16.0, 18.0, 72
        if "MEAN_REVERSION" in source:
            return 18.0, 24.0, 96
        if "SMC_MOMENTUM" in source:
            return 22.0, 30.0, 144
        return 22.0, 34.0, 168

    @staticmethod
    def _confidence(source: str, active_modules: List[str]) -> float:
        if source == "trend_confluence":
            return min(0.86, 0.72 + 0.04 * len(active_modules))
        if source == "low_regime_fade_confluence":
            return min(0.78, 0.66 + 0.04 * len(active_modules))
        return 0.64

    @staticmethod
    def _bias_1h(df1h: pd.DataFrame) -> int:
        c = df1h["close"].astype(float)
        e = _ema(c, 50)
        slope = e.iloc[-1] - e.iloc[-7]
        if c.iloc[-1] > e.iloc[-1] and slope > 0:
            return 1
        if c.iloc[-1] < e.iloc[-1] and slope < 0:
            return -1
        return 0

    @staticmethod
    def _resolve_now(df: pd.DataFrame, current_timestamp: Optional[datetime]) -> datetime:
        if current_timestamp is not None:
            return _as_utc(current_timestamp)
        if "start_time" in df.columns:
            return _as_utc(df["start_time"].iloc[-1])
        return _as_utc(df.index[-1])
