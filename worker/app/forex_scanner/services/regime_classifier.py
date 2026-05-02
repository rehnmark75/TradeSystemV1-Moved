"""
Shared ADX / regime computation utilities.

All strategies that need ADX should call these helpers rather than
duplicating the EMA-Wilder formula inline.  DataFetcher already stamps
``df['adx']`` on candle frames it returns; ``get_adx_from_df`` prefers
that pre-computed column so live and backtest paths produce identical
values without redundant work.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """EMA-Wilder ADX series from OHLC data.

    Requires columns: high, low, close.
    Returns a Series aligned to df.index, NaN-free (filled with 0.0).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)


def get_adx_from_df(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Return the most recent ADX value from a candle DataFrame.

    Priority:
    1. ``df['adx']``          — stamped by DataFetcher (period=14)
    2. ``df[f'adx_{period}']`` — enrichment column added by strategies
    3. Recompute on-the-fly via ``compute_adx``

    Returns None if the frame is too short or computation fails.
    """
    for col in ("adx", f"adx_{period}"):
        if col in df.columns:
            try:
                v = df[col].iloc[-1]
                if v is not None and not pd.isna(v):
                    return float(v)
            except Exception:
                pass

    try:
        series = compute_adx(df, period)
        v = series.iloc[-1]
        if not pd.isna(v):
            return float(v)
    except Exception:
        pass

    return None
