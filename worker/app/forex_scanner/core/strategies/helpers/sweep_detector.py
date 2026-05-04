"""
Sweep + Rejection Wick Detector for SMC_MOMENTUM strategy.

Detects when a 15m bar sweeps a liquidity pool level and closes back inside
(rejection wick), indicating a potential reversal entry.

A valid sweep requires:
  - Bar high/low exceeds pool by sweep_min..sweep_max pips
  - Bar closes back inside the swept level (rejection)
  - Wick beyond level >= wick_min_pct_of_range of total candle range

Returns SweepResult if a valid sweep+rejection is detected on the latest bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default pip sizes
_PIP_FX = 0.0001   # 4-decimal pairs
_PIP_JPY = 0.01    # JPY pairs


def _pip_size(is_jpy: bool) -> float:
    return _PIP_JPY if is_jpy else _PIP_FX


@dataclass
class SweepResult:
    pool_level: float
    pool_type: str          # 'PRIOR_DAY_HIGH', 'PRIOR_DAY_LOW', 'SWING_HIGH', 'SWING_LOW'
    direction: str          # 'BUY' (low swept) or 'SELL' (high swept)
    excess_pips: float      # how far price moved beyond the pool level
    wick_pct: float         # wick-beyond-level as fraction of total candle range
    is_double_sweep: bool   # same pool swept consecutively within lookback_bars
    sweep_bar_index: int    # index in df of the sweep candle (usually -1)


def _wick_pct_of_range(
    bar_high: float,
    bar_low: float,
    pool_level: float,
    direction: str,
) -> float:
    """
    Return the wick-beyond-level as a fraction of the total candle range.
    direction='SELL' means high swept; direction='BUY' means low swept.
    Returns 0.0 if range is zero.
    """
    total_range = bar_high - bar_low
    if total_range <= 0:
        return 0.0
    if direction == "SELL":
        wick = bar_high - pool_level
    else:
        wick = pool_level - bar_low
    return max(0.0, wick / total_range)


def _check_double_sweep(
    df: pd.DataFrame,
    pool_level: float,
    direction: str,
    current_bar_idx: int,
    lookback_bars: int,
    sweep_min_pips: float,
    pip: float,
) -> bool:
    """
    Return True if the same pool was swept within lookback_bars bars before current_bar_idx.
    """
    start = max(0, current_bar_idx - lookback_bars)
    end = current_bar_idx  # exclusive (don't include current bar)
    if end <= start:
        return False

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for i in range(start, end):
        if direction == "SELL":
            excess = highs[i] - pool_level
            if excess >= sweep_min_pips * pip and closes[i] < pool_level:
                return True
        else:
            excess = pool_level - lows[i]
            if excess >= sweep_min_pips * pip and closes[i] > pool_level:
                return True
    return False


def detect_sweep(
    df: pd.DataFrame,
    pool_highs: List[float],
    pool_lows: List[float],
    sweep_min_pips: float = 3.0,
    sweep_max_pips: float = 15.0,
    wick_min_pct: float = 0.50,
    is_jpy: bool = False,
    double_sweep_lookback: int = 6,
) -> Optional[SweepResult]:
    """
    Inspect the latest bar (df.iloc[-1]) for a liquidity sweep + rejection.

    Checks pool_highs (SELL setups — high swept) and pool_lows (BUY setups — low swept).
    Returns the first matching SweepResult, or None.

    Priority: checks the strongest sweep (largest excess_pips) when multiple pools qualify.

    df must have columns: open, high, low, close (and a DatetimeIndex or RangeIndex).
    pool_highs / pool_lows are lists of price levels from liquidity_pool_detector.
    """
    if df is None or len(df) < 2:
        return None

    pip = _pip_size(is_jpy)
    min_excess = sweep_min_pips * pip
    max_excess = sweep_max_pips * pip

    bar = df.iloc[-1]
    bar_idx = len(df) - 1
    h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])

    candidates: List[SweepResult] = []

    # SELL setup: bar high sweeps a pool_high and closes back below it
    for level in pool_highs:
        excess = h - level
        if excess < min_excess or excess > max_excess:
            continue
        if c >= level:
            continue  # no rejection — close is still above

        wick_pct = _wick_pct_of_range(h, l, level, "SELL")
        if wick_pct < wick_min_pct:
            continue

        pool_type = "PRIOR_DAY_HIGH" if level == pool_highs[0] else "SWING_HIGH"
        is_double = _check_double_sweep(
            df, level, "SELL", bar_idx, double_sweep_lookback, sweep_min_pips, pip
        )
        candidates.append(
            SweepResult(
                pool_level=level,
                pool_type=pool_type,
                direction="SELL",
                excess_pips=round(excess / pip, 1),
                wick_pct=round(wick_pct, 3),
                is_double_sweep=is_double,
                sweep_bar_index=bar_idx,
            )
        )

    # BUY setup: bar low sweeps a pool_low and closes back above it
    for level in pool_lows:
        excess = level - l
        if excess < min_excess or excess > max_excess:
            continue
        if c <= level:
            continue  # no rejection — close is still below

        wick_pct = _wick_pct_of_range(h, l, level, "BUY")
        if wick_pct < wick_min_pct:
            continue

        pool_type = "PRIOR_DAY_LOW" if level == pool_lows[0] else "SWING_LOW"
        is_double = _check_double_sweep(
            df, level, "BUY", bar_idx, double_sweep_lookback, sweep_min_pips, pip
        )
        candidates.append(
            SweepResult(
                pool_level=level,
                pool_type=pool_type,
                direction="BUY",
                excess_pips=round(excess / pip, 1),
                wick_pct=round(wick_pct, 3),
                is_double_sweep=is_double,
                sweep_bar_index=bar_idx,
            )
        )

    if not candidates:
        return None

    # Return the candidate with the largest excess (strongest sweep)
    return max(candidates, key=lambda r: r.excess_pips)
