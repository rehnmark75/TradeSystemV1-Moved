"""
Liquidity Pool Detector for SMC_MOMENTUM strategy.

Identifies nearby liquidity pools on a 15m bar series:
  - Prior trading-day high / low
  - Recent 5-bar swing pivots (age ≤ swing_max_age_bars)

Returns lists of candidate pool prices that price may sweep.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_prior_day_hl(
    df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (prior_day_high, prior_day_low) for the trading day before
    the last bar in df.  Looks back up to 5 calendar days to skip weekends.

    df must have a 'start_time' column (or DatetimeIndex) and 'high', 'low' columns.
    Returns (None, None) if no prior day data found.
    """
    if df is None or df.empty:
        return None, None

    # DataFetcher returns start_time as a column, not the index
    if "start_time" in df.columns:
        ts_series = pd.to_datetime(df["start_time"], utc=True)
    else:
        ts_series = pd.to_datetime(df.index, utc=True)

    current_ts = ts_series.iloc[-1]
    current_date = current_ts.date()

    dates = ts_series.dt.date.values
    for days_back in range(1, 6):
        prior_date = (current_ts - timedelta(days=days_back)).date()
        mask = dates == prior_date
        prior_bars = df[mask]
        if len(prior_bars) >= 4:  # at least one hour's worth of 15m bars
            return float(prior_bars["high"].max()), float(prior_bars["low"].min())

    logger.debug(f"liquidity_pool_detector: no prior day bars found before {current_date}")
    return None, None


def get_swing_levels(
    df: pd.DataFrame,
    pivot_bars: int = 2,
    max_age_bars: int = 20,
) -> Tuple[List[float], List[float]]:
    """
    Return (swing_highs, swing_lows) from df using a (2*pivot_bars+1)-bar pivot rule.

    A swing high at bar i: high[i] > high[i-k] for all k in 1..pivot_bars
                      AND  high[i] > high[i+k] for all k in 1..pivot_bars
    Pivots are collected from the last max_age_bars bars (excluding the current bar).

    df must have 'high', 'low' columns and a numeric-or-datetime index.
    Returns lists of price levels (may be empty).
    """
    n = len(df)
    if n < 2 * pivot_bars + 2:
        return [], []

    # Work on the bars excluding the current (last) bar
    # Look back max_age_bars + pivot_bars to find pivots whose age <= max_age_bars
    lookback_start = max(0, n - max_age_bars - pivot_bars - 1)
    # Candidate pivot indices: from lookback_start to n-2 (exclude current bar)
    # A pivot at index i needs i-pivot_bars..i+pivot_bars all to exist
    highs = df["high"].values
    lows = df["low"].values

    swing_highs: List[float] = []
    swing_lows: List[float] = []

    for i in range(max(pivot_bars, lookback_start), n - pivot_bars - 1):
        age = (n - 1) - i  # bars ago
        if age > max_age_bars:
            continue

        window_h = highs[i - pivot_bars : i + pivot_bars + 1]
        if len(window_h) == 2 * pivot_bars + 1 and highs[i] == window_h.max():
            # Confirm it's strictly greater on both sides (not a flat top)
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append(float(highs[i]))

        window_l = lows[i - pivot_bars : i + pivot_bars + 1]
        if len(window_l) == 2 * pivot_bars + 1 and lows[i] == window_l.min():
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append(float(lows[i]))

    return swing_highs, swing_lows


def get_liquidity_pools(
    df: pd.DataFrame,
    pivot_bars: int = 2,
    max_age_bars: int = 20,
) -> Tuple[List[float], List[float]]:
    """
    Combined pool list: prior day H/L + recent swing pivots.

    Returns (pool_highs, pool_lows) — lists of candidate price levels.
    Deduplicates levels within 1 pip (non-JPY) / 10 pips (JPY) to avoid
    stacking multiple very-close levels.
    """
    pd_high, pd_low = get_prior_day_hl(df)
    swing_highs, swing_lows = get_swing_levels(df, pivot_bars, max_age_bars)

    pool_highs: List[float] = []
    pool_lows: List[float] = []

    if pd_high is not None:
        pool_highs.append(pd_high)
    pool_highs.extend(swing_highs)

    if pd_low is not None:
        pool_lows.append(pd_low)
    pool_lows.extend(swing_lows)

    return pool_highs, pool_lows
