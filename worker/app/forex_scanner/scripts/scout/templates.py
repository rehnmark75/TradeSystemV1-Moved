"""Canonical strategy hypothesis library for the scout.

Each template is a pure function: (df, params) -> int signals array of the
same length as df (+1 BUY / -1 SELL / 0 flat). No cooldown, no SL/TP — the
simulator layers those on so PF differences trace to signal edge alone.

Naming: keep signal functions side-effect free and dependency-light.
Indicators computed inline so each template is trivially parallelizable
and there's no shared-state surprise across ProcessPoolExecutor workers.

Design constraint: all templates operate on a single timeframe passed in
(default 5m). Where a template references HTF context (e.g. EMA(200) trend
filter), the caller passes a 15m-derived column pre-joined to the 5m frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Indicator helpers (lightweight, private)
# ============================================================================

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _bollinger(close: pd.Series, period: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid + mult * sd, mid, mid - mult * sd


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd_line, signal)
    hist = macd_line - sig
    return macd_line, sig, hist


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = 1.0 / period
    atr = tr.ewm(alpha=a, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


# ============================================================================
# Template dataclass
# ============================================================================

@dataclass
class Template:
    name: str
    generate: Callable[[pd.DataFrame, Dict], np.ndarray]
    grid: List[Dict]
    description: str


# ============================================================================
# 1. TREND_EMA — buy pullback to fast EMA in aligned uptrend
# ============================================================================

def _gen_trend_ema(df: pd.DataFrame, p: Dict) -> np.ndarray:
    fast = _ema(df["close"], p["fast"])
    slow = _ema(df["close"], p["slow"])
    close = df["close"]

    # Uptrend: fast > slow AND slope(fast) > 0 for 3 bars
    up = (fast > slow) & (fast > fast.shift(3))
    down = (fast < slow) & (fast < fast.shift(3))

    # Pullback: price touches fast EMA (low crosses through or close within 0.1*ATR of fast)
    atr = _atr(df, 14)
    touched_fast = (df["low"] <= fast) & (df["high"] >= fast)

    sig = np.zeros(len(df), dtype=int)
    sig[(up & touched_fast).fillna(False).values] = 1
    sig[(down & touched_fast).fillna(False).values] = -1
    return sig


# ============================================================================
# 2. MEAN_REV_BB_RSI — classic oversold/overbought reversion
# ============================================================================

def _gen_mean_rev_bb_rsi(df: pd.DataFrame, p: Dict) -> np.ndarray:
    upper, mid, lower = _bollinger(df["close"], p["bb_period"], p["bb_mult"])
    rsi = _rsi(df["close"], p["rsi_period"])

    buy = (df["close"] <= lower) & (rsi <= p["rsi_os"])
    sell = (df["close"] >= upper) & (rsi >= p["rsi_ob"])

    sig = np.zeros(len(df), dtype=int)
    sig[buy.fillna(False).values] = 1
    sig[sell.fillna(False).values] = -1
    return sig


# ============================================================================
# 3. BREAKOUT_DONCHIAN — close outside N-bar high/low range
# ============================================================================

def _gen_breakout_donchian(df: pd.DataFrame, p: Dict) -> np.ndarray:
    n = p["n"]
    upper = df["high"].rolling(n).max().shift(1)  # prior N bars only
    lower = df["low"].rolling(n).min().shift(1)

    buy = df["close"] > upper
    sell = df["close"] < lower

    sig = np.zeros(len(df), dtype=int)
    sig[buy.fillna(False).values] = 1
    sig[sell.fillna(False).values] = -1
    return sig


# ============================================================================
# 4. RANGE_FADE_RSI — short at N-bar high + RSI overbought, mirror low
# ============================================================================

def _gen_range_fade_rsi(df: pd.DataFrame, p: Dict) -> np.ndarray:
    lookback = p["lookback"]
    upper = df["high"].rolling(lookback).max()
    lower = df["low"].rolling(lookback).min()
    rsi = _rsi(df["close"], 14)

    at_high = df["close"] >= upper
    at_low = df["close"] <= lower

    buy = at_low & (rsi <= p["rsi_os"])
    sell = at_high & (rsi >= p["rsi_ob"])

    sig = np.zeros(len(df), dtype=int)
    sig[buy.fillna(False).values] = 1
    sig[sell.fillna(False).values] = -1
    return sig


# ============================================================================
# 5. WICK_REJECTION — upper/lower wick >= ratio × body near N-bar boundary
# ============================================================================

def _gen_wick_rejection(df: pd.DataFrame, p: Dict) -> np.ndarray:
    ratio = p["ratio"]
    lookback = p.get("lookback", 20)
    upper = df["high"].rolling(lookback).max()
    lower = df["low"].rolling(lookback).min()

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    upper_wick = h - c.where(c > o, o)
    lower_wick = o.where(c > o, c) - l

    # Guard against div-by-zero on doji bars
    safe_body = body.where(body > 0, np.nan)

    at_high = h >= upper
    at_low = l <= lower

    sell = at_high & (upper_wick >= ratio * safe_body) & (c < o)
    buy = at_low & (lower_wick >= ratio * safe_body) & (c > o)

    sig = np.zeros(len(df), dtype=int)
    sig[buy.fillna(False).values] = 1
    sig[sell.fillna(False).values] = -1
    return sig


# ============================================================================
# 6. MACD_CONTINUATION — histogram zero-cross in trend direction
# ============================================================================

def _gen_macd_continuation(df: pd.DataFrame, p: Dict) -> np.ndarray:
    _, _, hist = _macd(df["close"], p["fast"], p["slow"], p["signal"])
    ema200 = _ema(df["close"], 200)

    # Zero-cross: hist crosses from negative to positive while close > EMA200
    hist_cross_up = (hist > 0) & (hist.shift(1) <= 0)
    hist_cross_down = (hist < 0) & (hist.shift(1) >= 0)

    buy = hist_cross_up & (df["close"] > ema200)
    sell = hist_cross_down & (df["close"] < ema200)

    sig = np.zeros(len(df), dtype=int)
    sig[buy.fillna(False).values] = 1
    sig[sell.fillna(False).values] = -1
    return sig  # cross events are already unique, no dedup needed


# ============================================================================
# 7. SQUEEZE_BREAKOUT — BB width percentile low, then break outside
# ============================================================================

def _gen_squeeze_breakout(df: pd.DataFrame, p: Dict) -> np.ndarray:
    upper, mid, lower = _bollinger(df["close"], 20, 2.0)
    width = upper - lower
    # Trailing 100-bar percentile of width
    width_pct = width.rolling(100).apply(lambda w: (w.iloc[-1] <= w).mean() * 100, raw=False)

    # Squeeze: width percentile <= threshold for N consecutive bars
    squeezed = (width_pct <= p["width_pct"]).rolling(p["squeeze_bars"]).apply(lambda s: s.all(), raw=True).fillna(0).astype(bool)

    # Breakout: squeezed AND close outside the band
    buy = squeezed.shift(1).fillna(False) & (df["close"] > upper)
    sell = squeezed.shift(1).fillna(False) & (df["close"] < lower)

    sig = np.zeros(len(df), dtype=int)
    sig[buy.values] = 1
    sig[sell.values] = -1
    return sig


# ============================================================================
# 8. MOMENTUM_ROC — N-bar ROC in top percentile + rising ADX
# ============================================================================

def _gen_momentum_roc(df: pd.DataFrame, p: Dict) -> np.ndarray:
    roc = df["close"].pct_change(p["roc_bars"])
    # Percentile over trailing 100 bars
    roc_pct = roc.rolling(100).apply(lambda w: (w.iloc[-1] <= w).mean() * 100, raw=False)
    adx = _adx(df, 14)
    adx_rising = adx > adx.shift(3)

    # Strong positive ROC + rising ADX
    buy = (roc_pct >= p["pct"]) & adx_rising & (roc > 0)
    # Strong negative ROC (bottom percentile) + rising ADX
    roc_pct_inv = roc.rolling(100).apply(lambda w: (w.iloc[-1] >= w).mean() * 100, raw=False)
    sell = (roc_pct_inv >= p["pct"]) & adx_rising & (roc < 0)

    sig = np.zeros(len(df), dtype=int)
    sig[buy.fillna(False).values] = 1
    sig[sell.fillna(False).values] = -1
    return sig


# Note: entry dedup (preventing a single sustained condition from firing
# on every bar) is handled by the simulator's walk-forward loop: once a
# trade opens, no new signals are taken until SL/TP/TIMEOUT + cooldown.
# This is simpler and more faithful to live scanner behavior than
# suppressing signals at template generation time.


# ============================================================================
# Template registry
# ============================================================================

TEMPLATES: List[Template] = [
    Template(
        name="TREND_EMA",
        generate=_gen_trend_ema,
        grid=[
            {"fast": 9,  "slow": 50},
            {"fast": 9,  "slow": 200},
            {"fast": 21, "slow": 50},
            {"fast": 21, "slow": 200},
        ],
        description="Buy pullback to fast EMA in aligned uptrend (EMAs ordered, fast rising)",
    ),
    Template(
        name="MEAN_REV_BB_RSI",
        generate=_gen_mean_rev_bb_rsi,
        grid=[
            {"bb_period": 20, "bb_mult": 2.0, "rsi_period": 14, "rsi_os": 30, "rsi_ob": 70},
            {"bb_period": 20, "bb_mult": 2.0, "rsi_period": 14, "rsi_os": 25, "rsi_ob": 75},
            {"bb_period": 20, "bb_mult": 2.5, "rsi_period": 14, "rsi_os": 30, "rsi_ob": 70},
            {"bb_period": 30, "bb_mult": 2.0, "rsi_period": 14, "rsi_os": 35, "rsi_ob": 65},
        ],
        description="Close outside BB + RSI extreme (classic mean reversion)",
    ),
    Template(
        name="BREAKOUT_DONCHIAN",
        generate=_gen_breakout_donchian,
        grid=[
            {"n": 10}, {"n": 20}, {"n": 55},
        ],
        description="Close above/below N-bar high/low channel (Donchian breakout)",
    ),
    Template(
        name="RANGE_FADE_RSI",
        generate=_gen_range_fade_rsi,
        grid=[
            {"lookback": 20, "rsi_os": 30, "rsi_ob": 70},
            {"lookback": 20, "rsi_os": 25, "rsi_ob": 75},
            {"lookback": 50, "rsi_os": 30, "rsi_ob": 70},
        ],
        description="Fade N-bar extreme when RSI confirms exhaustion",
    ),
    Template(
        name="WICK_REJECTION",
        generate=_gen_wick_rejection,
        grid=[
            {"ratio": 1.3, "lookback": 20},
            {"ratio": 1.5, "lookback": 20},
            {"ratio": 2.0, "lookback": 20},
            {"ratio": 1.5, "lookback": 50},
        ],
        description="Upper/lower wick >= ratio*body at N-bar boundary (prior EURJPY edge: PF 3.85 @ n=11)",
    ),
    Template(
        name="MACD_CONTINUATION",
        generate=_gen_macd_continuation,
        grid=[
            {"fast": 12, "slow": 26, "signal": 9},
            {"fast": 8,  "slow": 21, "signal": 5},
            {"fast": 5,  "slow": 13, "signal": 5},
        ],
        description="MACD histogram zero-cross in direction of EMA(200) trend filter",
    ),
    Template(
        name="SQUEEZE_BREAKOUT",
        generate=_gen_squeeze_breakout,
        grid=[
            {"squeeze_bars": 10, "width_pct": 20},
            {"squeeze_bars": 20, "width_pct": 20},
            {"squeeze_bars": 10, "width_pct": 10},
        ],
        description="BB width in bottom percentile for N bars, then break outside",
    ),
    Template(
        name="MOMENTUM_ROC",
        generate=_gen_momentum_roc,
        grid=[
            {"roc_bars": 5,  "pct": 90},
            {"roc_bars": 10, "pct": 90},
            {"roc_bars": 10, "pct": 80},
        ],
        description="N-bar rate-of-change in top percentile AND rising ADX",
    ),
]


TEMPLATE_BY_NAME: Dict[str, Template] = {t.name: t for t in TEMPLATES}
