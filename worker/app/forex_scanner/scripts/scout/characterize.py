"""Market characterization for the strategy scout.

Given a multi-timeframe candle set, produce the statistics that answer
"what kind of market is this?" — regime mix, volatility distribution,
mean-reversion vs trending signature, structural features, session
profile.

Every stat here is computed from raw candle data (no reliance on
alert_history or sidecar tables), so it doesn't inherit any historical
ADX/regime bugs. Uses canonical EMA-Wilder ADX from recompute_adx_regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Canonical ADX/regime (bypass caveat: this module is at
# /app/forex_scanner/scripts/, imported as forex_scanner.scripts.recompute_adx_regime)
from forex_scanner.scripts.recompute_adx_regime import (
    ema_wilder_adx,
    ema_wilder_atr,
    regime_from_adx,
)


# ============================================================================
# Regime distribution
# ============================================================================

def regime_distribution(df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    """Percent of bars in each canonical regime.

    Returns fractions summing to ~1.0 (N/A bars excluded)."""
    if df is None or len(df) < period * 3:
        return {"ranging": 0.0, "low_volatility": 0.0, "trending": 0.0, "breakout": 0.0, "n_bars": 0}
    adx = ema_wilder_adx(df, period=period)
    labels = adx.apply(regime_from_adx).dropna()
    total = len(labels)
    if total == 0:
        return {"ranging": 0.0, "low_volatility": 0.0, "trending": 0.0, "breakout": 0.0, "n_bars": 0}
    counts = labels.value_counts()
    return {
        "ranging": float(counts.get("ranging", 0)) / total,
        "low_volatility": float(counts.get("low_volatility", 0)) / total,
        "trending": float(counts.get("trending", 0)) / total,
        "breakout": float(counts.get("breakout", 0)) / total,
        "n_bars": int(total),
    }


# ============================================================================
# Volatility stats
# ============================================================================

def volatility_stats(df: pd.DataFrame, pip_value: float) -> Dict[str, float]:
    """ATR quartiles and BB-width distribution (in pips)."""
    atr = ema_wilder_atr(df, period=14)
    atr_pips = (atr / pip_value).dropna()
    bb_width_pips = _bb_width_pips(df, pip_value).dropna()

    return {
        "atr_p05": float(np.percentile(atr_pips, 5)) if len(atr_pips) else 0.0,
        "atr_p25": float(np.percentile(atr_pips, 25)) if len(atr_pips) else 0.0,
        "atr_p50": float(np.percentile(atr_pips, 50)) if len(atr_pips) else 0.0,
        "atr_p75": float(np.percentile(atr_pips, 75)) if len(atr_pips) else 0.0,
        "atr_p95": float(np.percentile(atr_pips, 95)) if len(atr_pips) else 0.0,
        "bb_width_p25": float(np.percentile(bb_width_pips, 25)) if len(bb_width_pips) else 0.0,
        "bb_width_p50": float(np.percentile(bb_width_pips, 50)) if len(bb_width_pips) else 0.0,
        "bb_width_p75": float(np.percentile(bb_width_pips, 75)) if len(bb_width_pips) else 0.0,
    }


def _bb_width_pips(df: pd.DataFrame, pip_value: float, period: int = 20, mult: float = 2.0) -> pd.Series:
    close = df["close"]
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ((ma + mult * sd) - (ma - mult * sd)) / pip_value


# ============================================================================
# Trend / mean-reversion signature
# ============================================================================

def mr_signature(df: pd.DataFrame) -> Dict[str, float]:
    """Autocorrelation at multiple lags, variance ratio, OU half-life.

    Autocorrelation < 0 at lag k = mean-reverting at that horizon.
    Variance ratio < 1 = mean-reverting, > 1 = trending, ~1 = random walk.
    Half-life = bars for price to revert halfway to mean (OU estimate).
    """
    close = df["close"].dropna()
    if len(close) < 200:
        return {
            "ac_lag1": 0.0, "ac_lag5": 0.0, "ac_lag20": 0.0, "ac_lag60": 0.0,
            "vr_k2": 1.0, "vr_k4": 1.0, "vr_k8": 1.0, "vr_k16": 1.0, "vr_k32": 1.0,
            "ou_half_life_bars": 0.0,
        }
    log_ret = np.log(close).diff().dropna()

    acs = {}
    for lag in (1, 5, 20, 60):
        if len(log_ret) > lag:
            acs[f"ac_lag{lag}"] = float(log_ret.autocorr(lag=lag))
        else:
            acs[f"ac_lag{lag}"] = 0.0

    # Variance ratio: Var(k-bar return) / (k * Var(1-bar return))
    var1 = log_ret.var()
    vrs = {}
    for k in (2, 4, 8, 16, 32):
        if len(log_ret) > k and var1 > 0:
            k_bar_ret = close.pct_change(k).dropna()
            vrs[f"vr_k{k}"] = float(k_bar_ret.var() / (k * var1))
        else:
            vrs[f"vr_k{k}"] = 1.0

    # OU half-life via regression of delta on lagged price
    # dP_t = theta * (mu - P_{t-1}) dt + noise  ⇒  ΔP = a + b*P_{t-1}
    # If b < 0, half-life = -ln(2)/b
    delta = close.diff().dropna()
    lag_price = close.shift(1).loc[delta.index]
    if len(delta) > 20 and lag_price.std() > 0:
        # Simple OLS b-hat
        x = lag_price.values
        y = delta.values
        x_mean = x.mean()
        y_mean = y.mean()
        b = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        # Require statistically meaningful negative b (not just numerical noise).
        # b in price/bar; for a pair around 1.0 with ATR ~0.001, noise-level b is ~1e-6.
        # Threshold b < -1e-4 filters out drift-scale false positives.
        if b < -1e-4:
            hl = float(-np.log(2) / b)
            # Cap at 1000 bars (≈3.5 days on 5m) — anything longer is effectively
            # "not mean-reverting on tradeable timescales"
            half_life = min(hl, 1000.0)
        else:
            half_life = 0.0  # not mean-reverting on tradeable horizon
    else:
        half_life = 0.0

    return {**acs, **vrs, "ou_half_life_bars": half_life}


# ============================================================================
# Structural stats (15m timeframe)
# ============================================================================

def structural_stats(df: pd.DataFrame, pip_value: float) -> Dict[str, float]:
    """Swing frequency, EQH/EQL density, range resolution."""
    if df is None or len(df) < 50:
        return {
            "swings_per_day": 0.0, "avg_swing_size_atr": 0.0,
            "eq_touch_rate_per_day": 0.0,
            "range_count": 0, "range_breakout_rate": 0.0,
        }

    # Pivot detection (strength=2)
    pivots = _detect_pivots(df, strength=2)
    n_days = max(1, (df.index[-1] - df.index[0]).total_seconds() / 86400)
    swings_per_day = len(pivots) / n_days

    # Average swing size in ATRs
    atr = ema_wilder_atr(df, period=14)
    swing_sizes_atr = []
    for i in range(1, len(pivots)):
        p_prev, p_curr = pivots[i - 1], pivots[i]
        if p_prev.is_high != p_curr.is_high:
            swing_pips = abs(p_curr.price - p_prev.price) / pip_value
            atr_pips_here = float(atr.iloc[p_curr.index]) / pip_value if atr.iloc[p_curr.index] > 0 else 1.0
            if atr_pips_here > 0:
                swing_sizes_atr.append(swing_pips / atr_pips_here)
    avg_swing_atr = float(np.mean(swing_sizes_atr)) if swing_sizes_atr else 0.0

    # EQH/EQL rate: count pivots within 1.5-pip tolerance of their immediate
    # predecessor of same polarity
    tol = 1.5 * pip_value
    eq_count = 0
    for i in range(1, len(pivots)):
        prev_same = [p for p in pivots[:i] if p.is_high == pivots[i].is_high]
        if prev_same and abs(pivots[i].price - prev_same[-1].price) <= tol:
            eq_count += 1
    eq_rate_per_day = eq_count / n_days

    # Range detection: consecutive bars where ATR is < 85% of trailing-50 avg
    atr_avg = atr.rolling(50).mean()
    compressed = (atr / atr_avg) < 0.85
    # Count runs of >=20 consecutive compressed bars
    ranges = _find_runs(compressed.fillna(False).values, min_length=20)
    range_count = len(ranges)

    # Range resolution: after each range ends, did price move >= 1 ATR within 10 bars?
    breakout_count = 0
    for start, end in ranges:
        if end + 10 >= len(df):
            continue
        range_high = df["high"].iloc[start:end + 1].max()
        range_low = df["low"].iloc[start:end + 1].min()
        atr_end = atr.iloc[end]
        if pd.isna(atr_end) or atr_end <= 0:
            continue
        post = df.iloc[end + 1:end + 11]
        went_up = (post["high"].max() - range_high) >= atr_end
        went_down = (range_low - post["low"].min()) >= atr_end
        if went_up or went_down:
            breakout_count += 1
    breakout_rate = breakout_count / max(1, range_count)

    return {
        "swings_per_day": float(swings_per_day),
        "avg_swing_size_atr": avg_swing_atr,
        "eq_touch_rate_per_day": float(eq_rate_per_day),
        "range_count": int(range_count),
        "range_breakout_rate": float(breakout_rate),
    }


@dataclass
class _Pivot:
    index: int
    price: float
    is_high: bool


def _detect_pivots(df: pd.DataFrame, strength: int = 2) -> List[_Pivot]:
    if df is None or len(df) < strength * 2 + 1:
        return []
    highs = df["high"].values
    lows = df["low"].values
    out: List[_Pivot] = []
    for i in range(strength, len(df) - strength):
        lh = highs[i - strength:i]; rh = highs[i + 1:i + 1 + strength]
        if highs[i] >= lh.max() and highs[i] >= rh.max():
            out.append(_Pivot(i, float(highs[i]), True))
        ll = lows[i - strength:i]; rl = lows[i + 1:i + 1 + strength]
        if lows[i] <= ll.min() and lows[i] <= rl.min():
            out.append(_Pivot(i, float(lows[i]), False))
    out.sort(key=lambda p: p.index)
    return out


def _find_runs(mask: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_length:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_length:
        runs.append((start, len(mask) - 1))
    return runs


# ============================================================================
# Session × regime heatmap
# ============================================================================

def session_regime_heatmap(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """DataFrame indexed by UTC hour, columns are regime fractions.

    Useful for spotting "EURJPY is ranging in London but trending in NY" patterns."""
    if df is None or len(df) < period * 3:
        return pd.DataFrame()
    adx = ema_wilder_adx(df, period=period)
    labels = adx.apply(regime_from_adx)

    # Index must be datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    hours = df.index.hour
    out = []
    for h in range(24):
        mask = (hours == h) & labels.notna().values
        n = int(mask.sum())
        if n == 0:
            out.append({"hour_utc": h, "n_bars": 0,
                       "ranging": 0.0, "low_volatility": 0.0, "trending": 0.0, "breakout": 0.0})
            continue
        labs = labels[mask]
        out.append({
            "hour_utc": h, "n_bars": n,
            "ranging": float((labs == "ranging").mean()),
            "low_volatility": float((labs == "low_volatility").mean()),
            "trending": float((labs == "trending").mean()),
            "breakout": float((labs == "breakout").mean()),
        })
    return pd.DataFrame(out)


# ============================================================================
# Aggregate into a single characterization dict for report.py
# ============================================================================

@dataclass
class Characterization:
    epic: str
    days: int
    pip_value: float
    candle_count_5m: int
    candle_count_15m: int
    candle_count_1h: int
    regime_15m: Dict[str, float] = field(default_factory=dict)
    regime_1h: Dict[str, float] = field(default_factory=dict)
    volatility_15m: Dict[str, float] = field(default_factory=dict)
    mr_signature_5m: Dict[str, float] = field(default_factory=dict)
    structural_15m: Dict[str, float] = field(default_factory=dict)
    session_heatmap: pd.DataFrame = field(default_factory=pd.DataFrame)


def characterize(
    epic: str,
    days: int,
    pip_value: float,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
) -> Characterization:
    return Characterization(
        epic=epic,
        days=days,
        pip_value=pip_value,
        candle_count_5m=len(df_5m),
        candle_count_15m=len(df_15m),
        candle_count_1h=len(df_1h),
        regime_15m=regime_distribution(df_15m),
        regime_1h=regime_distribution(df_1h),
        volatility_15m=volatility_stats(df_15m, pip_value),
        mr_signature_5m=mr_signature(df_5m),
        structural_15m=structural_stats(df_15m, pip_value),
        session_heatmap=session_regime_heatmap(df_15m),
    )
