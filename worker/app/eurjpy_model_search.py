#!/usr/bin/env python3
"""
EURJPY Model Search — pre-registered experiment.
See /home/hr/Projects/TradeSystemV1/EURJPY_MODEL_SEARCH.md for the full spec.

Searches a battery of entry models (trend, mean-reversion, pullback-in-trend,
session, volatility-breakout) at 1H and 4H, using a strict in-sample /
out-of-sample split.  Selection and confirmation gates are LOCKED per the spec.

Run inside task-worker:
    python /app/eurjpy_model_search.py [--smoke]

Redirect output to a durable file:
    python /app/eurjpy_model_search.py > /tmp/eurjpy_search.txt 2>&1

--smoke runs only EMA(20/50) cross at 1H to verify the pipeline produces
sane numbers (a few hundred trades, PF ~0.7–1.5) before the full battery.
"""

import sys
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2

# ---------------------------------------------------------------------------
# Constants / config (LOCKED per spec)
# ---------------------------------------------------------------------------

EPIC = "CS.D.EURJPY.MINI.IP"
PIP_SIZE = 0.01          # JPY pair: 1 pip = 0.01
COST_PIPS = 2.0          # round-trip cost

WINDOW_START = "2020-01-01"
WINDOW_END   = "2026-06-12"
IS_END       = "2023-12-31"   # in-sample: 2020-01-01 → 2023-12-31
OOS_START    = "2024-01-01"   # out-of-sample: 2024-01-01 → 2026-06-12

OOS_MID      = "2025-01-01"   # split OOS into two halves for consistency check

# ATR exit brackets (SL multiplier fixed; TP multiplier swept)
SL_ATR_MULT = 1.5
TP_ATR_MULTS = [1.0, 1.5, 2.0]
BAR_CAP = 200  # horizon cap per trade (in bars of same TF); timeouts excluded from headline PF

# Locked selection gates
IS_MIN_N_RESOLVED  = 150
IS_MIN_PF          = 1.3
TOP_N_CANDIDATES   = 5

OOS_MIN_PF         = 1.2
OOS_MIN_N_RESOLVED = 60
OOS_HALF_MIN_PF    = 1.0    # both OOS halves
OOS_SHRINKAGE_MAX  = 0.75   # OOS PF ≥ 0.75 × IS PF

DB_HOST     = "postgres"
DB_PORT     = 5432
DB_NAME     = "forex"
DB_USER     = "postgres"
DB_PASSWORD = "postgres"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
    )


def load_candles(conn, timeframe: int) -> pd.DataFrame:
    """
    Load EURJPY candles for the full window from ig_candles_backtest.
    Returns DataFrame indexed by start_time (UTC-aware timestamps are kept as-is).
    Columns: open, high, low, close.
    """
    sql = """
        SELECT start_time,
               open::float8 AS open,
               high::float8 AS high,
               low::float8  AS low,
               close::float8 AS close
        FROM ig_candles_backtest
        WHERE epic = %s
          AND timeframe = %s
          AND start_time >= %s
          AND start_time <= %s
        ORDER BY start_time ASC
    """
    df = pd.read_sql(
        sql, conn,
        params=(EPIC, timeframe, WINDOW_START, WINDOW_END),
        parse_dates=["start_time"],
    )
    df = df.set_index("start_time")
    return df


# ---------------------------------------------------------------------------
# Indicator helpers — ALL computed causally (no look-ahead)
#
# Convention: use only data through bar t to generate the signal that
# fires at bar t.  Entry occurs at bar t+1's open (modelled as close[t]),
# and first-passage scanning starts from bar t+1.
# ---------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (recursive, causal)."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def atr14(df: pd.DataFrame) -> pd.Series:
    """Wilder ATR(14) — causal."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False).mean()


def rsi14(close: pd.Series) -> pd.Series:
    """RSI(14) — Wilder smoothing, causal."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """
    Bollinger Bands.
    Returns (mid, upper, lower) — all causal pandas Series.
    """
    mid   = sma(close, n)
    std   = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def zscore_vs_sma20(close: pd.Series) -> pd.Series:
    """Z-score of close relative to SMA(20) with rolling std — causal."""
    m   = sma(close, 20)
    std = close.rolling(20).std(ddof=0)
    return (close - m) / std.replace(0, np.nan)


def donchian_hi(close: pd.Series, n: int) -> pd.Series:
    """Rolling n-bar max of *prior* bars (excludes current bar via shift(1))."""
    return close.shift(1).rolling(n).max()


def donchian_lo(close: pd.Series, n: int) -> pd.Series:
    return close.shift(1).rolling(n).min()


# ---------------------------------------------------------------------------
# Signal generators — EDGE-TRIGGERED
#
# Each returns a DataFrame with columns [bar_idx, direction, atr_at_entry]
# where bar_idx is the position in the full-TF DataFrame at which the signal
# fires (i.e. the TRIGGER bar t).  Entry is evaluated at bar t+1.
#
# Critical: signals must fire on the TRANSITION (cross / first breach),
# not the LEVEL.  This prevents the "thousands of trades" double-count bug.
# ---------------------------------------------------------------------------

def signals_ema_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """
    EMA(fast/slow) cross:
    - BUY  on bar where fast EMA crosses ABOVE slow EMA (fast[t] > slow[t] AND fast[t-1] <= slow[t-1])
    - SELL on bar where fast EMA crosses BELOW slow EMA
    """
    close = df["close"]
    atr   = atr14(df)
    f_ema = ema(close, fast)
    s_ema = ema(close, slow)
    cross_up   = (f_ema > s_ema) & (f_ema.shift(1) <= s_ema.shift(1))
    cross_down = (f_ema < s_ema) & (f_ema.shift(1) >= s_ema.shift(1))
    idxs, dirs, atrs = [], [], []
    for i in range(1, len(df)):
        if cross_up.iloc[i]:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
        elif cross_down.iloc[i]:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


def signals_donchian_breakout(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Donchian(n) channel breakout:
    - BUY  when close breaks ABOVE n-bar prior high (close > prior-high AND prev close <= prior-high)
    - SELL when close breaks BELOW n-bar prior low
    Entry fires on the first breakout bar; inside channel = flat.
    """
    close = df["close"]
    atr   = atr14(df)
    hi    = donchian_hi(close, n)
    lo    = donchian_lo(close, n)
    break_up   = (close > hi)  & (close.shift(1) <= hi.shift(1))
    break_down = (close < lo)  & (close.shift(1) >= lo.shift(1))
    idxs, dirs, atrs = [], [], []
    for i in range(n + 1, len(df)):
        if break_up.iloc[i]:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
        elif break_down.iloc[i]:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


def signals_tsmom(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Time-series momentum:
    Fire a BUY when the k-bar return TURNS positive (sign flips from ≤0 to >0).
    Fire a SELL when it turns negative.
    Edge-triggered on sign change.
    """
    close  = df["close"]
    atr    = atr14(df)
    ret_k  = close - close.shift(k)
    sign_  = np.sign(ret_k)
    flip_up   = (sign_ > 0) & (sign_.shift(1) <= 0)
    flip_down = (sign_ < 0) & (sign_.shift(1) >= 0)
    idxs, dirs, atrs = [], [], []
    for i in range(k + 1, len(df)):
        if flip_up.iloc[i]:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
        elif flip_down.iloc[i]:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


def signals_bb_fade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger(20,2) mean-reversion fade:
    - BUY:  close was BELOW lower band on bar t-1 (breach), then on bar t it closes
            back INSIDE the band (close[t] > lower[t]).
    - SELL: close was ABOVE upper band on bar t-1, then closes back inside.
    Edge-triggered on the re-entry candle.
    """
    close = df["close"]
    atr   = atr14(df)
    _, upper, lower = bollinger(close)
    # Breach then re-entry
    re_buy  = (close.shift(1) < lower.shift(1)) & (close >= lower)
    re_sell = (close.shift(1) > upper.shift(1)) & (close <= upper)
    idxs, dirs, atrs = [], [], []
    for i in range(21, len(df)):
        if re_buy.iloc[i]:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
        elif re_sell.iloc[i]:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


def signals_rsi_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI(14) oversold/overbought reversion:
    - BUY:  RSI crosses back ABOVE 30 from below (RSI[t] >= 30 AND RSI[t-1] < 30)
    - SELL: RSI crosses back BELOW 70 from above
    """
    close = df["close"]
    atr   = atr14(df)
    r     = rsi14(close)
    buy_  = (r >= 30) & (r.shift(1) < 30)
    sell_ = (r <= 70) & (r.shift(1) > 70)
    idxs, dirs, atrs = [], [], []
    for i in range(15, len(df)):
        if buy_.iloc[i]:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
        elif sell_.iloc[i]:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


def signals_zscore_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score vs SMA20 reversion:
    - BUY:  z crosses back ABOVE -2.0 from below
    - SELL: z crosses back BELOW +2.0 from above
    """
    close = df["close"]
    atr   = atr14(df)
    z     = zscore_vs_sma20(close)
    buy_  = (z >= -2.0) & (z.shift(1) < -2.0)
    sell_ = (z <= 2.0)  & (z.shift(1) > 2.0)
    idxs, dirs, atrs = [], [], []
    for i in range(21, len(df)):
        if buy_.iloc[i]:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
        elif sell_.iloc[i]:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


def signals_pullback_ema200(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA200 trend filter + RSI(14) pullback entry:
    Trend = long if close > EMA200, short if close < EMA200.
    Entry: RSI crosses back above 40 (from below) in an uptrend — dip-buy.
           RSI crosses back below 60 (from above) in a downtrend — rally-sell.
    Edge-triggered on the RSI bounce bar.
    """
    close = df["close"]
    atr   = atr14(df)
    e200  = ema(close, 200)
    r     = rsi14(close)
    # uptrend + RSI recovers from oversold dip (below 40 → above 40)
    buy_  = (close > e200) & (r >= 40) & (r.shift(1) < 40)
    # downtrend + RSI recovers from overbought (above 60 → below 60)
    sell_ = (close < e200) & (r <= 60) & (r.shift(1) > 60)
    idxs, dirs, atrs = [], [], []
    for i in range(201, len(df)):
        if buy_.iloc[i]:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
        elif sell_.iloc[i]:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


# ---------------------------------------------------------------------------
# Session models (1H only — degenerate at 4H)
# ---------------------------------------------------------------------------

def _london_range(df: pd.DataFrame):
    """
    Build the Asian session range (23:00–05:59 UTC) for each calendar day,
    return a dict: date -> (range_high, range_low) covering bars in that day.
    Only uses data from bars completely within 23:00–05:59 UTC.
    Importantly, a bar at e.g. 23:00 covers 23:00–23:59, which is pre-London.
    London open is defined as 06:00–08:59 UTC in this model.
    """
    # Use bar hour (UTC)
    hours = df.index.hour
    # Asian range hours: 23, 0, 1, 2, 3, 4, 5
    asian_mask = (hours >= 23) | (hours <= 5)
    asian = df[asian_mask].copy()
    # Group by calendar date of the LONDON session that follows
    # Asian bars at 23:00-23:59 belong to the *next* day's London session
    def session_date(ts):
        h = ts.hour
        d = ts.date()
        if h >= 23:
            from datetime import timedelta
            return d + timedelta(days=1)
        return d
    asian["session_date"] = [session_date(ts) for ts in asian.index]
    grouped = asian.groupby("session_date").agg(range_high=("high", "max"), range_low=("low", "min"))
    return grouped


def signals_london_open_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """
    London-open breakout of the Asian range.
    Strategy (1H only):
    - Identify the Asian session range for each trading day (bars 23–05 UTC).
    - At each London bar (06:00–08:59 UTC), check if close breaks the range:
      BUY  if close > range_high (first such bar each day)
      SELL if close < range_low  (first such bar each day)
    Edge-triggered: only the first breakout bar per day fires.
    """
    close = df["close"]
    atr   = atr14(df)
    range_df = _london_range(df)
    hours = df.index.hour
    # London bars: 06, 07, 08
    london_mask = (hours >= 6) & (hours <= 8)
    # For each day, track whether a signal has fired
    fired_dates = set()
    idxs, dirs, atrs = [], [], []
    for i in range(201, len(df)):
        if not london_mask.iloc[i]:
            continue
        bar_date = df.index[i].date()
        if bar_date in fired_dates:
            continue
        r = range_df.get(bar_date, None) if hasattr(range_df, "get") else None
        if r is None:
            try:
                if bar_date not in range_df.index:
                    continue
                rh = range_df.loc[bar_date, "range_high"]
                rl = range_df.loc[bar_date, "range_low"]
            except Exception:
                continue
        else:
            rh = r["range_high"] if hasattr(r, "__getitem__") else float("nan")
            rl = r["range_low"]  if hasattr(r, "__getitem__") else float("nan")
            # The above branch won't execute since range_df is a DataFrame
            continue
        c = close.iloc[i]
        if c > rh:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
            fired_dates.add(bar_date)
        elif c < rl:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
            fired_dates.add(bar_date)
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


# Fix: use proper pandas lookup
def signals_london_open_breakout_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    London-open breakout of the Asian range (correct implementation).
    """
    close = df["close"]
    atr   = atr14(df)
    range_df = _london_range(df)   # indexed by Python date
    hours = np.array(df.index.hour)          # numpy array — supports [i] directly
    london_mask = (hours >= 6) & (hours <= 8)
    fired_dates = set()
    idxs, dirs, atrs = [], [], []
    for i in range(201, len(df)):
        if not london_mask[i]:
            continue
        bar_date = df.index[i].date()
        if bar_date in fired_dates:
            continue
        if bar_date not in range_df.index:
            continue
        rh = range_df.at[bar_date, "range_high"]
        rl = range_df.at[bar_date, "range_low"]
        if pd.isna(rh) or pd.isna(rl):
            continue
        c = close.iloc[i]
        if c > rh:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
            fired_dates.add(bar_date)
        elif c < rl:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
            fired_dates.add(bar_date)
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


def signals_session_momentum(
    df: pd.DataFrame,
    base_signal_fn,
    best_hours: list,
) -> pd.DataFrame:
    """
    Hour-of-day-conditioned momentum:
    Filter a base signal (e.g. EMA cross) to fire only during the specified UTC hours.
    `best_hours` is selected ON IN-SAMPLE DATA ONLY and passed in at call time.
    """
    base = base_signal_fn(df)
    if base.empty:
        return base
    bar_hours = pd.Series(df.index.hour, index=range(len(df)))
    valid = base[base["bar_idx"].apply(lambda i: bar_hours.iloc[i] in best_hours)]
    return valid.reset_index(drop=True)


def signals_atr_channel_breakout(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    ATR-channel breakout:
    Channel = SMA(n) ± 1.5×ATR(14).
    BUY  when close first breaks ABOVE upper channel edge.
    SELL when close first breaks BELOW lower channel edge.
    Edge-triggered on the transition bar.
    """
    close  = df["close"]
    atr    = atr14(df)
    mid    = sma(close, n)
    upper  = mid + 1.5 * atr
    lower  = mid - 1.5 * atr
    break_up   = (close > upper) & (close.shift(1) <= upper.shift(1))
    break_down = (close < lower) & (close.shift(1) >= lower.shift(1))
    idxs, dirs, atrs = [], [], []
    for i in range(n + 1, len(df)):
        if break_up.iloc[i]:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
        elif break_down.iloc[i]:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


def signals_orb_london(df: pd.DataFrame, n_bars: int = 3) -> pd.DataFrame:
    """
    Opening-range breakout: first n_bars of London session (06:00–08:59 UTC).
    Build the range from bars 06, 07, 08 within each day.
    Signal fires on the first bar (09:00+) that breaks that range.
    Only implemented for 1H (degenerate at 4H).
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    atr    = atr14(df)
    hours  = np.array(df.index.hour)         # numpy array — supports [i] directly
    # Gather ORB ranges per trading day
    from datetime import date as date_type
    orb_ranges = {}   # date -> (orb_high, orb_low)
    for i in range(len(df)):
        h = hours[i]
        d = df.index[i].date()
        if h in (6, 7, 8):
            prev = orb_ranges.get(d, (float("-inf"), float("inf")))
            orb_ranges[d] = (
                max(prev[0], high.iloc[i]),
                min(prev[1], low.iloc[i]),
            )
    fired_dates = set()
    idxs, dirs, atrs = [], [], []
    for i in range(201, len(df)):
        h = hours[i]
        if h < 9:   # we only trade the breakout, not the range-building bars
            continue
        d = df.index[i].date()
        if d in fired_dates:
            continue
        if d not in orb_ranges:
            continue
        orb_hi, orb_lo = orb_ranges[d]
        if orb_hi == float("-inf") or orb_lo == float("inf"):
            continue
        c = close.iloc[i]
        if c > orb_hi:
            idxs.append(i); dirs.append("BUY");  atrs.append(atr.iloc[i])
            fired_dates.add(d)
        elif c < orb_lo:
            idxs.append(i); dirs.append("SELL"); atrs.append(atr.iloc[i])
            fired_dates.add(d)
    return pd.DataFrame({"bar_idx": idxs, "direction": dirs, "atr": atrs})


# ---------------------------------------------------------------------------
# Session hour selection helper (IS-only)
# ---------------------------------------------------------------------------

def best_session_hours(
    df: pd.DataFrame,
    base_signal_fn,
    tp_mult: float,
    is_end_idx: int,
) -> list:
    """
    Find the 3-hour UTC window (contiguous or custom set) that maximises PF
    on IN-SAMPLE data only.  Returns a list of UTC hours.

    Implementation: evaluate 6 session blocks and pick the best.
    This selection must use is_end_idx as the cutoff.
    """
    candidate_blocks = {
        "asian":    list(range(0, 7)),
        "london":   list(range(7, 13)),
        "ny":       list(range(13, 20)),
        "late_ny":  list(range(20, 24)),
        "london_open": [7, 8, 9],
        "london_close_ny": [11, 12, 13, 14],
    }
    base_sigs = base_signal_fn(df)
    if base_sigs.empty:
        return list(range(7, 13))  # default to London
    # Restrict to IS
    is_sigs = base_sigs[base_sigs["bar_idx"] < is_end_idx]
    if is_sigs.empty:
        return list(range(7, 13))

    best_pf  = -1.0
    best_hrs = list(range(7, 13))
    hours_series = pd.Series(df.index.hour, index=range(len(df)))

    for block_name, hrs in candidate_blocks.items():
        filtered = is_sigs[is_sigs["bar_idx"].apply(lambda i: hours_series.iloc[i] in hrs)]
        if len(filtered) < 20:
            continue
        trades = evaluate_signals(df, filtered, tp_mult, slice(None, is_end_idx))
        if trades.empty:
            continue
        res = trades[trades["outcome"] != "TIMEOUT"]
        if res.empty:
            continue
        pf = compute_pf(res)
        if pf is not None and pf > best_pf:
            best_pf  = pf
            best_hrs = hrs

    return best_hrs


# ---------------------------------------------------------------------------
# Trade evaluator — first-passage ATR-bracket
#
# Mirrors range_fade_diag3.py's approach:
#   - Entry at bar t+1 open (modelled as close[t] for simplicity;
#     true open is unavailable in this DB schema — close is standard).
#   - SL = entry_price - SL_ATR * atr  (BUY)
#   - TP = entry_price + TP_ATR * atr  (BUY)
#   - Scan bars t+1 … t+BAR_CAP:
#       - if hit SL and TP same bar → pessimistic straddle = LOSS
#       - else first passage wins
#   - Timeout if horizon reached without passage.
#   - MFE = max favorable excursion during the scan window (pips).
# ---------------------------------------------------------------------------

def evaluate_signals(
    df: pd.DataFrame,
    sigs: pd.DataFrame,
    tp_mult: float,
    idx_slice,           # slice or "all" for the full series
) -> pd.DataFrame:
    """
    Evaluate a signal DataFrame over the ATR-bracket exit model.

    Parameters
    ----------
    df : full-TF DataFrame (open, high, low, close)
    sigs : signal DataFrame with columns [bar_idx, direction, atr]
    tp_mult : TP multiplier on ATR
    idx_slice : slice restricting which signal bar_idxs to evaluate
                (e.g. slice(None, is_end_idx) for IS subset)

    Returns
    -------
    DataFrame with columns:
        bar_idx, direction, entry_price, atr, sl_pips, tp_pips,
        outcome ('WIN'/'LOSS'/'TIMEOUT'), pips_net, mfe_pips, entry_time
    """
    if sigs.empty:
        return pd.DataFrame()

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n_bars = len(df)

    records = []
    for _, row in sigs.iterrows():
        t = int(row["bar_idx"])
        # Apply the idx_slice filter
        if isinstance(idx_slice, slice):
            lo = idx_slice.start or 0
            hi = idx_slice.stop  or n_bars
            if t < lo or t >= hi:
                continue
        direction = row["direction"]
        atr_val   = float(row["atr"])
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # Entry at bar t+1 (modelled as close[t])
        entry = closes[t]
        sl_dist = SL_ATR_MULT * atr_val
        tp_dist = tp_mult     * atr_val
        sl_pips = sl_dist / PIP_SIZE
        tp_pips = tp_dist / PIP_SIZE

        if direction == "BUY":
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        # First-passage scan: bars t+1 … t+BAR_CAP
        scan_start = t + 1
        scan_end   = min(t + 1 + BAR_CAP, n_bars)

        outcome   = "TIMEOUT"
        pips_raw  = 0.0
        mfe_pips  = 0.0

        for j in range(scan_start, scan_end):
            h = highs[j]
            lo_bar = lows[j]

            if direction == "BUY":
                hit_tp = h >= tp_price
                hit_sl = lo_bar <= sl_price
                fav_j  = (h - entry) / PIP_SIZE
            else:
                hit_tp = lo_bar <= tp_price
                hit_sl = h >= sl_price
                fav_j  = (entry - lo_bar) / PIP_SIZE

            mfe_pips = max(mfe_pips, max(fav_j, 0.0))

            if hit_tp and hit_sl:
                # Same bar straddle → pessimistic = LOSS
                outcome  = "LOSS"
                pips_raw = -sl_pips
                break
            elif hit_tp:
                outcome  = "WIN"
                pips_raw = tp_pips
                break
            elif hit_sl:
                outcome  = "LOSS"
                pips_raw = -sl_pips
                break

        # Net pips after cost (deducted from every resolved trade and timeout)
        pips_net = pips_raw - COST_PIPS

        records.append({
            "bar_idx":     t,
            "direction":   direction,
            "entry_price": entry,
            "atr":         atr_val,
            "sl_pips":     sl_pips,
            "tp_pips":     tp_pips,
            "outcome":     outcome,
            "pips_raw":    pips_raw,
            "pips_net":    pips_net,
            "mfe_pips":    mfe_pips,
            "entry_time":  df.index[t],
        })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pf(resolved: pd.DataFrame) -> Optional[float]:
    """
    Profit factor on resolved (WIN/LOSS) trades using net pips.
    Excludes TIMEOUTs from denominator and numerator.
    """
    wins   = resolved.loc[resolved["outcome"] == "WIN",  "pips_net"].sum()
    losses = resolved.loc[resolved["outcome"] == "LOSS", "pips_net"].abs().sum()
    if losses == 0:
        return float("inf") if wins > 0 else None
    return round(wins / losses, 3)


def compute_metrics(trades: pd.DataFrame, tf_hours: float) -> dict:
    """
    Full metric set for a set of trades (all outcomes).
    `tf_hours` = bar width in hours (1 for 1H, 4 for 4H) — used for Sharpe.
    """
    n_total   = len(trades)
    if n_total == 0:
        return {
            "n_total": 0, "n_resolved": 0, "n_timeout": 0,
            "wr": None, "pf": None, "avg_pips": None,
            "sharpe": None, "trades_per_day": None,
            "median_mfe": None, "doa_frac": None,
            "timeout_pct": None,
        }

    resolved  = trades[trades["outcome"] != "TIMEOUT"]
    n_resolved = len(resolved)
    n_timeout  = n_total - n_resolved
    timeout_pct = round(100 * n_timeout / n_total, 1)

    wr  = round(resolved[resolved["outcome"] == "WIN"].shape[0] / max(n_resolved, 1), 3)
    pf  = compute_pf(resolved)
    avg_pips = round(resolved["pips_net"].mean(), 2) if n_resolved > 0 else None

    # Annualized Sharpe: per-trade Sharpe × sqrt(trades per year)
    # Trades per year = resolved trades in the window × (365.25 / n_days)
    if n_resolved >= 2:
        std_  = resolved["pips_net"].std(ddof=1)
        mean_ = resolved["pips_net"].mean()
        n_days = max(1, (trades["entry_time"].max() - trades["entry_time"].min()).days)
        trades_per_year = n_resolved * 365.25 / n_days
        if std_ > 0:
            sharpe = round((mean_ / std_) * np.sqrt(trades_per_year), 3)
        else:
            sharpe = None
    else:
        sharpe = None

    # Trades per calendar day (total, including timeouts)
    n_days = max(1, (trades["entry_time"].max() - trades["entry_time"].min()).days)
    trades_per_day = round(n_total / n_days, 4)

    # Entry quality
    median_mfe = round(float(np.median(trades["mfe_pips"])), 2)
    # Die-on-arrival: MFE < 0.25 × (entry ATR expressed in pips)
    doa_thresh = trades["atr"] / PIP_SIZE * 0.25
    doa_mask   = trades["mfe_pips"] < doa_thresh
    doa_frac   = round(float(doa_mask.mean()), 3)

    return {
        "n_total":       n_total,
        "n_resolved":    n_resolved,
        "n_timeout":     n_timeout,
        "timeout_pct":   timeout_pct,
        "wr":            wr,
        "pf":            pf,
        "avg_pips":      avg_pips,
        "sharpe":        sharpe,
        "trades_per_day": trades_per_day,
        "median_mfe":    median_mfe,
        "doa_frac":      doa_frac,
    }


# ---------------------------------------------------------------------------
# Battery definition
# ---------------------------------------------------------------------------

def build_battery(df_1h: pd.DataFrame, df_4h: pd.DataFrame, is_end_idx_1h: int, smoke: bool):
    """
    Returns a list of (label, tf_label, sigs_df, tf_hours) for every model.
    `is_end_idx_1h` is the IS cutoff index in df_1h (used for session-block selection).
    In smoke mode, only the two EMA-cross models are returned.
    """
    battery = []

    # ------------------------------------------------------------------
    # 1. TREND family
    # ------------------------------------------------------------------
    for tf_label, df, tf_hours in [("1H", df_1h, 1.0), ("4H", df_4h, 4.0)]:
        # EMA cross variants
        for fast, slow in [(20, 50), (50, 200)]:
            sigs = signals_ema_cross(df, fast, slow)
            battery.append((f"EMA_{fast}/{slow}_cross", tf_label, sigs, tf_hours))
            if smoke:
                break   # smoke: only EMA 20/50 at 1H
        if smoke:
            break

    if smoke:
        return battery

    for tf_label, df, tf_hours in [("1H", df_1h, 1.0), ("4H", df_4h, 4.0)]:
        # Donchian breakout
        for n in [20, 55]:
            sigs = signals_donchian_breakout(df, n)
            battery.append((f"Donchian_{n}_breakout", tf_label, sigs, tf_hours))

        # TSMOM
        k_vals = [30, 90] if tf_label == "1H" else [30, 90]
        for k in k_vals:
            sigs = signals_tsmom(df, k)
            battery.append((f"TSMOM_k{k}", tf_label, sigs, tf_hours))

    # ------------------------------------------------------------------
    # 2. MEAN-REVERSION family
    # ------------------------------------------------------------------
    for tf_label, df, tf_hours in [("1H", df_1h, 1.0), ("4H", df_4h, 4.0)]:
        battery.append(("BB_fade",          tf_label, signals_bb_fade(df),           tf_hours))
        battery.append(("RSI_reversion",    tf_label, signals_rsi_reversion(df),     tf_hours))
        battery.append(("Zscore_reversion", tf_label, signals_zscore_reversion(df),  tf_hours))

    # ------------------------------------------------------------------
    # 3. PULLBACK-IN-TREND family
    # ------------------------------------------------------------------
    for tf_label, df, tf_hours in [("1H", df_1h, 1.0), ("4H", df_4h, 4.0)]:
        battery.append(("EMA200_RSI_pullback", tf_label, signals_pullback_ema200(df), tf_hours))

    # ------------------------------------------------------------------
    # 4. SESSION family (1H only — degenerate at 4H)
    # ------------------------------------------------------------------
    # London-open breakout of Asian range
    sigs_lob = signals_london_open_breakout_v2(df_1h)
    battery.append(("LondonOpen_breakout", "1H", sigs_lob, 1.0))

    # Hour-of-day-conditioned: use EMA(20/50) cross as base, freeze best session IS-only
    # Default TP mult = 1.5 for session selection
    best_hrs = best_session_hours(df_1h, lambda d: signals_ema_cross(d, 20, 50), 1.5, is_end_idx_1h)
    sigs_base = signals_ema_cross(df_1h, 20, 50)
    hours_1h  = pd.Series(df_1h.index.hour, index=range(len(df_1h)))
    sigs_session = sigs_base[
        sigs_base["bar_idx"].apply(lambda i: hours_1h.iloc[i] in best_hrs)
    ].reset_index(drop=True)
    battery.append((f"Session_EMA20_50_hrs{best_hrs[0]}-{best_hrs[-1]}", "1H", sigs_session, 1.0))

    # ------------------------------------------------------------------
    # 5. VOLATILITY BREAKOUT family
    # ------------------------------------------------------------------
    # ORB London (1H only)
    sigs_orb = signals_orb_london(df_1h, n_bars=3)
    battery.append(("ORB_London_3bar", "1H", sigs_orb, 1.0))

    # ATR channel breakout
    for tf_label, df, tf_hours in [("1H", df_1h, 1.0), ("4H", df_4h, 4.0)]:
        sigs = signals_atr_channel_breakout(df, n=20)
        battery.append(("ATR_channel_breakout", tf_label, sigs, tf_hours))

    return battery


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

def fmt(v, spec=".3f", na="  N/A "):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    if isinstance(v, float) and np.isinf(v):
        return "  inf "
    return format(v, spec)


def fmt_pct(v, na="  N/A "):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return f"{v * 100:5.1f}%"


def bar_separator(n=110):
    print("=" * n)


def thin_sep(n=110):
    print("-" * n)


# ---------------------------------------------------------------------------
# Selection logic (LOCKED per spec)
# ---------------------------------------------------------------------------

def oos_passes(oos_pf, oos_n_resolved, oos_h1_pf, oos_h2_pf, is_pf) -> tuple:
    """
    Returns (passes: bool, reasons: list[str]).
    """
    reasons = []
    if oos_pf is None or oos_pf < OOS_MIN_PF:
        reasons.append(f"OOS PF {fmt(oos_pf)} < {OOS_MIN_PF}")
    if oos_n_resolved < OOS_MIN_N_RESOLVED:
        reasons.append(f"OOS n_resolved {oos_n_resolved} < {OOS_MIN_N_RESOLVED}")
    if oos_h1_pf is None or oos_h1_pf <= OOS_HALF_MIN_PF:
        reasons.append(f"OOS H1 PF {fmt(oos_h1_pf)} ≤ {OOS_HALF_MIN_PF}")
    if oos_h2_pf is None or oos_h2_pf <= OOS_HALF_MIN_PF:
        reasons.append(f"OOS H2 PF {fmt(oos_h2_pf)} ≤ {OOS_HALF_MIN_PF}")
    if is_pf is not None and oos_pf is not None:
        if oos_pf < OOS_SHRINKAGE_MAX * is_pf:
            reasons.append(
                f"OOS PF {fmt(oos_pf)} < 0.75 × IS PF {fmt(is_pf)} = {fmt(OOS_SHRINKAGE_MAX * is_pf)}"
            )
    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(smoke: bool = False):
    bar_separator()
    print("EURJPY MODEL SEARCH — Pre-registered experiment")
    print(f"Spec:     /home/hr/Projects/TradeSystemV1/EURJPY_MODEL_SEARCH.md")
    print(f"EPIC:     {EPIC}  |  PIP SIZE: {PIP_SIZE}  |  Cost: {COST_PIPS} pips RT")
    print(f"Window:   {WINDOW_START} → {WINDOW_END}")
    print(f"IN-SAMPLE:  {WINDOW_START} → {IS_END}  (4y)")
    print(f"OOS:        {OOS_START} → {WINDOW_END}  (~2.5y)")
    print(f"OOS halves: H1={OOS_START}→{OOS_MID}  H2={OOS_MID}→{WINDOW_END}")
    print(f"ATR exit:   SL={SL_ATR_MULT}×ATR  |  TP∈{TP_ATR_MULTS}×ATR  |  cap={BAR_CAP} bars")
    print(f"IS gates:   n_resolved≥{IS_MIN_N_RESOLVED} AND PF≥{IS_MIN_PF}  →  top {TOP_N_CANDIDATES} candidates")
    print(f"OOS gates:  PF≥{OOS_MIN_PF} AND n≥{OOS_MIN_N_RESOLVED} AND both halves PF>{OOS_HALF_MIN_PF} AND OOS≥{OOS_SHRINKAGE_MAX}×IS")
    if smoke:
        print("MODE:     SMOKE TEST — EMA(20/50) cross at 1H only")
    bar_separator()
    print()

    # -------------------------------------------------------------------------
    # Load candles
    # -------------------------------------------------------------------------
    print("Loading candles from ig_candles_backtest...")
    conn = get_conn()
    df_1h = load_candles(conn, 60)
    df_4h = load_candles(conn, 240)
    conn.close()

    print(f"  1H bars loaded: {len(df_1h):,}  ({df_1h.index[0]} → {df_1h.index[-1]})")
    print(f"  4H bars loaded: {len(df_4h):,}  ({df_4h.index[0]} → {df_4h.index[-1]})")
    print()

    # IS/OOS split indices
    is_end_dt  = pd.Timestamp(IS_END  + " 23:59:00")
    oos_start_dt = pd.Timestamp(OOS_START)
    oos_mid_dt   = pd.Timestamp(OOS_MID)

    # For 1H
    is_mask_1h  = df_1h.index <= is_end_dt
    oos_mask_1h = df_1h.index >= oos_start_dt
    oos_h1_mask_1h = (df_1h.index >= oos_start_dt) & (df_1h.index < oos_mid_dt)
    oos_h2_mask_1h = df_1h.index >= oos_mid_dt

    # Integer cutoffs (index positions)
    is_end_idx_1h = int(is_mask_1h.sum())

    # For 4H
    is_mask_4h  = df_4h.index <= is_end_dt
    oos_mask_4h = df_4h.index >= oos_start_dt
    oos_h1_mask_4h = (df_4h.index >= oos_start_dt) & (df_4h.index < oos_mid_dt)
    oos_h2_mask_4h = df_4h.index >= oos_mid_dt

    is_end_idx_4h = int(is_mask_4h.sum())

    print(f"IS bars:  1H={is_mask_1h.sum():,}  4H={is_mask_4h.sum():,}")
    print(f"OOS bars: 1H={oos_mask_1h.sum():,}  4H={oos_mask_4h.sum():,}")
    print()

    # -------------------------------------------------------------------------
    # Build battery
    # -------------------------------------------------------------------------
    print("Building signal battery...")
    battery = build_battery(df_1h, df_4h, is_end_idx_1h, smoke)
    print(f"  {len(battery)} model(s) × {len(TP_ATR_MULTS)} TP multipliers = "
          f"{len(battery) * len(TP_ATR_MULTS)} cells")
    print()

    # -------------------------------------------------------------------------
    # Evaluate every cell
    # -------------------------------------------------------------------------
    # results[model_label][tf_label][tp_mult] = {is: metrics, oos: metrics, ...}
    all_cells = []   # list of dicts with all info

    for (model_label, tf_label, sigs_df, tf_hours) in battery:
        df = df_1h if tf_label == "1H" else df_4h
        is_end_idx = is_end_idx_1h if tf_label == "1H" else is_end_idx_4h
        is_mask  = is_mask_1h  if tf_label == "1H" else is_mask_4h

        for tp_mult in TP_ATR_MULTS:
            # Evaluate on full series (IS+OOS together)
            all_trades = evaluate_signals(df, sigs_df, tp_mult, slice(None, None))
            if all_trades.empty:
                all_cells.append({
                    "model": model_label, "tf": tf_label, "tp": tp_mult,
                    "is": {"n_total": 0, "n_resolved": 0, "pf": None},
                    "oos": {"n_total": 0, "n_resolved": 0, "pf": None},
                    "is_metrics": compute_metrics(pd.DataFrame(), tf_hours),
                    "oos_metrics": compute_metrics(pd.DataFrame(), tf_hours),
                    "oos_h1_pf": None, "oos_h2_pf": None,
                })
                continue

            # Split by entry_time
            is_trades   = all_trades[all_trades["entry_time"] <= is_end_dt]
            oos_trades  = all_trades[all_trades["entry_time"] >= oos_start_dt]
            oos_h1 = oos_trades[oos_trades["entry_time"] < oos_mid_dt]
            oos_h2 = oos_trades[oos_trades["entry_time"] >= oos_mid_dt]

            is_m  = compute_metrics(is_trades,  tf_hours)
            oos_m = compute_metrics(oos_trades, tf_hours)

            oos_h1_res = oos_h1[oos_h1["outcome"] != "TIMEOUT"]
            oos_h2_res = oos_h2[oos_h2["outcome"] != "TIMEOUT"]
            oos_h1_pf  = compute_pf(oos_h1_res)
            oos_h2_pf  = compute_pf(oos_h2_res)

            all_cells.append({
                "model": model_label, "tf": tf_label, "tp": tp_mult,
                "is_metrics":  is_m,
                "oos_metrics": oos_m,
                "oos_h1_pf":   oos_h1_pf,
                "oos_h2_pf":   oos_h2_pf,
            })

    # -------------------------------------------------------------------------
    # Print full in-sample battery table
    # -------------------------------------------------------------------------
    bar_separator()
    print("IN-SAMPLE BATTERY TABLE (2020-01-01 → 2023-12-31)")
    print("PF and metrics on RESOLVED trades only.  n_resolved = total − timeouts.")
    print(f"{'Model':<38} {'TF':<4} {'TP×':<5}"
          f" {'n_res':>6} {'TO%':>5} {'WR':>6} {'PF':>7}"
          f" {'avgPips':>8} {'Sharpe':>7} {'tr/d':>6}"
          f" {'medMFE':>7} {'DOA%':>6}")
    thin_sep()

    for cell in all_cells:
        m    = cell["is_metrics"]
        n_r  = m["n_resolved"]
        flag = " *" if (n_r >= IS_MIN_N_RESOLVED and m["pf"] is not None and m["pf"] >= IS_MIN_PF) else "  "
        print(
            f"{cell['model']:<38} {cell['tf']:<4} {cell['tp']:<5.1f}"
            f" {n_r:>6} {fmt(m['timeout_pct'], '.1f'):>5} "
            f"{fmt_pct(m['wr']):>6} {fmt(m['pf'], '.3f'):>7}"
            f" {fmt(m['avg_pips'], '.2f'):>8} {fmt(m['sharpe'], '.3f'):>7}"
            f" {fmt(m['trades_per_day'], '.4f'):>6}"
            f" {fmt(m['median_mfe'], '.2f'):>7} {fmt_pct(m['doa_frac']):>6}"
            f"{flag}"
        )
    print()
    print("* = passes IS gate (n_resolved≥150 AND PF≥1.3)")
    bar_separator()
    print()

    # -------------------------------------------------------------------------
    # Selection: IS filter → rank by PF → top 5
    # -------------------------------------------------------------------------
    candidates = [
        cell for cell in all_cells
        if cell["is_metrics"]["n_resolved"] >= IS_MIN_N_RESOLVED
        and cell["is_metrics"]["pf"] is not None
        and cell["is_metrics"]["pf"] >= IS_MIN_PF
    ]
    candidates.sort(key=lambda c: c["is_metrics"]["pf"] or 0, reverse=True)
    top5 = candidates[:TOP_N_CANDIDATES]

    bar_separator()
    print(f"CANDIDATES AFTER IS FILTER (n_resolved≥{IS_MIN_N_RESOLVED}, PF≥{IS_MIN_PF}):"
          f"  {len(candidates)} qualify  →  taking top {len(top5)}")
    bar_separator()
    if not top5:
        print("  No candidates passed the IS filter.  NO-GO.")
    else:
        print(f"{'Rank':<5} {'Model':<38} {'TF':<4} {'TP×':<5}"
              f" {'IS_n':>6} {'IS_PF':>7}"
              f" {'OOS_n':>6} {'OOS_PF':>7} {'OOS_H1':>8} {'OOS_H2':>8}"
              f"  VERDICT")
        thin_sep()
        any_pass = False
        for rank, cell in enumerate(top5, 1):
            is_m   = cell["is_metrics"]
            oos_m  = cell["oos_metrics"]
            oos_h1_pf = cell["oos_h1_pf"]
            oos_h2_pf = cell["oos_h2_pf"]
            oos_n_r   = oos_m["n_resolved"]
            oos_pf    = oos_m["pf"]
            is_pf     = is_m["pf"]

            passes, reasons = oos_passes(oos_pf, oos_n_r, oos_h1_pf, oos_h2_pf, is_pf)
            verdict = "PASS" if passes else "FAIL"
            if passes:
                any_pass = True

            print(
                f"{rank:<5} {cell['model']:<38} {cell['tf']:<4} {cell['tp']:<5.1f}"
                f" {is_m['n_resolved']:>6} {fmt(is_pf, '.3f'):>7}"
                f" {oos_n_r:>6} {fmt(oos_pf, '.3f'):>7}"
                f" {fmt(oos_h1_pf, '.3f'):>8} {fmt(oos_h2_pf, '.3f'):>8}"
                f"  {verdict}"
            )
            if not passes:
                for r in reasons:
                    print(f"       └─ {r}")

        print()
        # Shrinkage visibility
        print("IS → OOS SHRINKAGE (full PF comparison):")
        for cell in top5:
            is_pf  = cell["is_metrics"]["pf"]
            oos_pf = cell["oos_metrics"]["pf"]
            shrink = f"{fmt(oos_pf, '.3f')} / {fmt(is_pf, '.3f')} = {fmt(oos_pf / is_pf if is_pf else None, '.3f')}" \
                if is_pf else "N/A"
            print(f"  {cell['model']} [{cell['tf']} TP×{cell['tp']}]: {shrink}")

    # -------------------------------------------------------------------------
    # GO / NO-GO
    # -------------------------------------------------------------------------
    print()
    bar_separator()
    print("FINAL DECISION")
    bar_separator()
    any_pass_final = any(
        oos_passes(
            c["oos_metrics"]["pf"],
            c["oos_metrics"]["n_resolved"],
            c["oos_h1_pf"],
            c["oos_h2_pf"],
            c["is_metrics"]["pf"],
        )[0]
        for c in top5
    ) if top5 else False

    if any_pass_final:
        print()
        print("  GO: ≥1 candidate passed OOS confirmation.")
        passing = [
            c for c in top5
            if oos_passes(
                c["oos_metrics"]["pf"],
                c["oos_metrics"]["n_resolved"],
                c["oos_h1_pf"],
                c["oos_h2_pf"],
                c["is_metrics"]["pf"],
            )[0]
        ]
        for c in passing:
            print(f"  → EURJPY model: {c['model']}  [{c['tf']}]  TP×{c['tp']}")
            print(f"     IS PF={fmt(c['is_metrics']['pf'])}"
                  f"  OOS PF={fmt(c['oos_metrics']['pf'])}"
                  f"  OOS H1={fmt(c['oos_h1_pf'])}"
                  f"  OOS H2={fmt(c['oos_h2_pf'])}")
        print()
        print("  Next step: wire as monitor-only (separate task).  DO NOT hand-tune further.")
    else:
        print()
        print("  NO-GO: No candidate survived OOS confirmation.")
        print("  → The broad IS search found no OOS-robust EURJPY entry model.")
        print("  → DO NOT hand-pick an IS winner and ship it — that is the overfit trap.")
        print("  → Document the result and revisit the hypothesis (data / edge-source).")

    bar_separator()
    print()
    print("Script complete.  Read-only — NO database writes performed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EURJPY Model Search (pre-registered)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-test mode: run only EMA(20/50) cross at 1H to verify "
            "the pipeline produces sane numbers before the full battery."
        ),
    )
    args = parser.parse_args()
    run(smoke=args.smoke)
