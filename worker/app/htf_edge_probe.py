#!/usr/bin/env python3
"""
HTF Edge Probe — pre-registered experiment
See: /home/hr/Projects/TradeSystemV1/HTF_EDGE_PROBE.md

Tests canonical trend signals (TSMOM, Donchian, MA-trend) at Daily and H4
on 9 FX pairs, 2020-01-01..2026-06-12, to determine whether a capturable
trend edge exists at HTF before building any bespoke strategy.

Run inside task-worker:
    python /app/htf_edge_probe.py [--smoke]

--smoke flag runs EURUSD-only over full 2020..2026 window as a sanity check.

Output is structured text to stdout — redirect to a durable file:
    python /app/htf_edge_probe.py > /tmp/htf_edge_results.txt 2>&1
"""

import sys
import argparse
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_HOST = "postgres"
DB_PORT = 5432
DB_NAME = "forex"
DB_USER = "postgres"

# NOTE: spec says "8 pairs" but enumerates 9 (GBPJPY deliberately excluded).
# We run all 9 listed pairs; the QUALIFIES threshold keeps ≥6 as written.
PAIRS = [
    ("EURUSD", "CS.D.EURUSD.CEEM.IP"),
    ("GBPUSD", "CS.D.GBPUSD.MINI.IP"),
    ("USDJPY", "CS.D.USDJPY.MINI.IP"),
    ("USDCAD", "CS.D.USDCAD.MINI.IP"),
    ("USDCHF", "CS.D.USDCHF.MINI.IP"),
    ("EURJPY", "CS.D.EURJPY.MINI.IP"),
    ("AUDJPY", "CS.D.AUDJPY.MINI.IP"),
    ("AUDUSD", "CS.D.AUDUSD.MINI.IP"),
    ("NZDUSD", "CS.D.NZDUSD.MINI.IP"),
]

# Pip size by pair
JPY_PAIRS = {"USDJPY", "EURJPY", "AUDJPY", "GBPJPY"}

def pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_PAIRS else 0.0001

# Cost in price units (2.0 pips round-trip)
def cost_in_price(pair: str, pips: float = 2.0) -> float:
    return pips * pip_size(pair)

# Signal params from spec
TSMOM_DAILY_K = [20, 60]
TSMOM_H4_K    = [30, 90]
DONCHIAN_DAILY_N = [20, 55]
DONCHIAN_H4_N    = [20, 55]

WINDOW_START = "2020-01-01"
WINDOW_END   = "2026-06-12"

# Sub-period boundaries for robustness check
SUBPERIODS = [
    ("2020-21", "2020-01-01", "2021-12-31"),
    ("2022-23", "2022-01-01", "2023-12-31"),
    ("2024-26", "2024-01-01", "2026-06-12"),
]

# QUALIFIES thresholds (locked pre-registration)
QUAL_FULL_PF  = 1.2
QUAL_SUBPD_PF = 1.0   # must exceed in EACH sub-period
QUAL_MIN_PAIRS = 6    # PF > 1.0 in at least 6 of 9 pairs

# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------

DB_PASSWORD = "postgres"

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )


def load_4h_candles(conn, epic: str) -> pd.DataFrame:
    """Load 4H candles from ig_candles_backtest for one epic."""
    sql = """
        SELECT start_time, open, high, low, close
        FROM ig_candles_backtest
        WHERE epic = %s
          AND timeframe = 240
          AND start_time >= %s
          AND start_time <= %s
        ORDER BY start_time ASC
    """
    df = pd.read_sql(sql, conn, params=(epic, WINDOW_START, WINDOW_END),
                     parse_dates=["start_time"])
    df = df.set_index("start_time")
    return df


def resample_4h_to_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 4H candles to Daily using calendar date grouping.

    Uses date(start_time) — NOT a fixed bar count — so weekend gaps and
    holiday-shortened sessions are handled correctly.

    OHLC aggregation:
      open  = first bar's open
      high  = max of bar highs
      low   = min of bar lows
      close = last bar's close
    """
    # Group by calendar date (UTC date of the bar open)
    df_4h = df_4h.copy()
    df_4h["date"] = df_4h.index.normalize()  # midnight of each bar's date

    daily = df_4h.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    daily.index.name = "start_time"
    # Keep only days with at least 2 bars (avoid fabricating days with single partial bars)
    bar_counts = df_4h.groupby("date").size()
    daily = daily[bar_counts >= 2]
    return daily


# ---------------------------------------------------------------------------
# Signal generators — NO look-ahead
#
# Rule: a signal computed from bar t's close takes effect at bar t+1 open,
# modeled as "position shifted 1 bar". Trades are entered/exited on the
# close of bar t+1 (close-to-close pricing).
#
# For rolling windows: use .shift(1) before the rolling call so bar t is
# NOT included in its own lookback.
# ---------------------------------------------------------------------------

def tsmom_signal(close: pd.Series, k: int) -> pd.Series:
    """
    Time-series momentum:
      +1 (long)  if close[t] > close[t-k]
      -1 (short) if close[t] < close[t-k]
    Signal is lagged by 1 bar to avoid look-ahead.
    """
    # trailing K-bar return: compare current close to close K bars ago
    trailing_ret = close - close.shift(k)
    # raw position from signal (before entry lag)
    raw_pos = np.sign(trailing_ret)
    raw_pos = raw_pos.replace(0, np.nan).ffill()  # carry position through zero-ret bars
    raw_pos = raw_pos.fillna(0).astype(int)
    # shift by 1: signal computed at bar t → position held from bar t+1
    position = raw_pos.shift(1).fillna(0).astype(int)
    return position


def donchian_signal(close: pd.Series, n: int) -> pd.Series:
    """
    Donchian / Turtle breakout:
      Enter long  on close > prior N-bar high
      Enter short on close < prior N-bar low
      Exit long   on close < prior N/2-bar low
      Exit short  on close > prior N/2-bar high
      Between entry and exit: hold previous position (flat otherwise)

    Computed from prior bars (shifted 1 so bar t not in its own window).
    Final position also shifted 1 bar for execution lag.
    """
    n_exit = max(1, n // 2)

    # Prior-bar extremes (shift(1) so window ends at t-1)
    prev_close = close.shift(1)
    roll_high_entry = prev_close.rolling(n).max()
    roll_low_entry  = prev_close.rolling(n).min()
    roll_high_exit  = prev_close.rolling(n_exit).max()
    roll_low_exit   = prev_close.rolling(n_exit).min()

    # Walk forward to build position (no vectorized shortcut — state machine)
    pos = 0
    positions = []
    for i in range(len(close)):
        c = close.iloc[i]
        rhe = roll_high_entry.iloc[i]
        rle = roll_low_entry.iloc[i]
        rxh = roll_high_exit.iloc[i]
        rxl = roll_low_exit.iloc[i]

        if pd.isna(rhe):
            positions.append(0)
            continue

        if pos == 0:
            # Look for entry
            if c > rhe:
                pos = 1
            elif c < rle:
                pos = -1
        elif pos == 1:
            # In long — check exit
            if c < rxl:
                pos = 0
        elif pos == -1:
            # In short — check exit
            if c > rxh:
                pos = 0

        positions.append(pos)

    raw_pos = pd.Series(positions, index=close.index)
    # Shift 1 bar: signal at t → position from bar t+1
    position = raw_pos.shift(1).fillna(0).astype(int)
    return position


def ma_trend_signal(close: pd.Series) -> pd.Series:
    """
    MA trend:
      Long  when EMA50 > EMA200 AND close > EMA50
      Short when EMA50 < EMA200 AND close < EMA50
      Flat  otherwise

    EMA computed from full history to bar t (no look-ahead concern for EMA
    itself, but we still shift the final position 1 bar for execution lag).
    """
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    raw_pos = pd.Series(0, index=close.index)
    long_cond  = (ema50 > ema200) & (close > ema50)
    short_cond = (ema50 < ema200) & (close < ema50)
    raw_pos[long_cond]  =  1
    raw_pos[short_cond] = -1

    # Shift 1 bar for execution lag
    position = raw_pos.shift(1).fillna(0).astype(int)
    return position


# ---------------------------------------------------------------------------
# Trade extraction and metrics
# ---------------------------------------------------------------------------

def extract_trades(
    close: pd.Series,
    position: pd.Series,
    pair: str,
    cost_pips: float = 2.0,
) -> pd.DataFrame:
    """
    Convert a position series to discrete trades.

    A trade begins when position changes FROM a non-zero state OR when going
    from 0 → non-zero. It ends at the next position change or signal end.

    Trade return (in pips) = signed price change from entry close to exit close
                             − round-trip cost (always subtracted).

    Returns a DataFrame with columns:
      entry_time, exit_time, direction, entry_price, exit_price,
      pips_raw, pips_net, win
    """
    cost = cost_in_price(pair, pips=cost_pips)
    ps = pip_size(pair)

    trades = []
    prev_pos = 0
    entry_time = None
    entry_price = None
    direction = 0

    for t in range(len(position)):
        cur_pos = position.iloc[t]
        c = close.iloc[t]

        if cur_pos != prev_pos:
            # Close existing trade if we were in one
            if prev_pos != 0 and entry_time is not None:
                exit_time  = close.index[t]
                exit_price = c
                # price move in direction of trade
                raw_change = (exit_price - entry_price) * prev_pos
                pips_raw = raw_change / ps
                pips_net = pips_raw - cost_pips  # always subtract cost
                trades.append({
                    "entry_time":  entry_time,
                    "exit_time":   exit_time,
                    "direction":   "LONG" if prev_pos == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "pips_raw":    round(pips_raw, 3),
                    "pips_net":    round(pips_net, 3),
                    "win":         pips_net > 0,
                })

            # Open new trade if entering non-zero position
            if cur_pos != 0:
                entry_time  = close.index[t]
                entry_price = c
                direction   = cur_pos
            else:
                entry_time  = None
                entry_price = None
                direction   = 0

        prev_pos = cur_pos

    # Close open trade at last bar if still active
    if prev_pos != 0 and entry_time is not None:
        exit_time  = close.index[-1]
        exit_price = close.iloc[-1]
        raw_change = (exit_price - entry_price) * prev_pos
        pips_raw   = raw_change / ps
        pips_net   = pips_raw - cost_pips
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   exit_time,
            "direction":   "LONG" if prev_pos == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "pips_raw":    round(pips_raw, 3),
            "pips_net":    round(pips_net, 3),
            "win":         pips_net > 0,
        })

    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "pips_raw", "pips_net", "win"
        ])
    return pd.DataFrame(trades)


def compute_pf(trades_df: pd.DataFrame) -> Optional[float]:
    """Profit factor from trade pips_net. Returns None if no losing trades."""
    if trades_df.empty:
        return None
    wins  = trades_df.loc[trades_df["win"], "pips_net"].sum()
    losses = trades_df.loc[~trades_df["win"], "pips_net"].abs().sum()
    if losses == 0:
        return float("inf") if wins > 0 else None
    return round(wins / losses, 3)


def compute_metrics(
    trades_df: pd.DataFrame,
    close: pd.Series,
    position: pd.Series,
    pair: str,
    cost_pips: float = 2.0,
) -> dict:
    """Compute full metric set for a signal on one pair."""
    n = len(trades_df)
    if n == 0:
        return {
            "n_trades": 0, "hit_rate": None, "pf": None,
            "avg_pips": None, "sharpe": None, "trades_per_day": None,
        }

    hit_rate = trades_df["win"].mean()
    pf = compute_pf(trades_df)
    avg_pips = trades_df["pips_net"].mean()

    # Equity curve: daily pip P&L from position × daily price change
    ps = pip_size(pair)
    cost_per_bar = cost_in_price(pair, pips=cost_pips) / ps  # cost in pips per bar

    # Mark-to-market returns in pips (no costs — we'll deduct at trade level
    # for PF/avg but use mark-to-market for Sharpe as it's more stable).
    # position[t] is already the position held DURING bar t (signals already
    # shifted 1 bar in the signal generators), so bar-t return = diff[t] * position[t].
    # Do NOT shift again here.
    price_ret_pips = close.diff() / ps * position.fillna(0)
    # Deduct cost on entry/exit bars
    pos_changes = position.diff().abs() > 0
    price_ret_pips -= pos_changes * cost_per_bar

    # Drop NaN at start
    eq_pips = price_ret_pips.dropna()

    if eq_pips.std() == 0 or len(eq_pips) < 2:
        sharpe = None
    else:
        # Determine bars per year for annualization
        n_obs = len(eq_pips)
        n_days = max(1, (close.index[-1] - close.index[0]).days)
        bars_per_year = n_obs / (n_days / 365.25)
        sharpe = round((eq_pips.mean() / eq_pips.std()) * np.sqrt(bars_per_year), 3)

    # Trades per day
    total_days = max(1, (close.index[-1] - close.index[0]).days)
    trades_per_day = round(n / total_days, 4)

    return {
        "n_trades":      n,
        "hit_rate":      round(hit_rate, 3),
        "pf":            pf,
        "avg_pips":      round(avg_pips, 2),
        "sharpe":        sharpe,
        "trades_per_day": trades_per_day,
    }


def subperiod_pf(
    trades_df: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    """Compute PF for each of the 3 locked sub-periods."""
    result = {}
    for label, sp_start, sp_end in SUBPERIODS:
        if trades_df.empty:
            result[label] = None
            continue
        mask = (
            (trades_df["entry_time"] >= pd.Timestamp(sp_start)) &
            (trades_df["entry_time"] <= pd.Timestamp(sp_end))
        )
        sub = trades_df[mask]
        result[label] = compute_pf(sub)
    return result


# ---------------------------------------------------------------------------
# QUALIFIES test (locked pre-registration)
# ---------------------------------------------------------------------------

def qualifies(
    full_pf: Optional[float],
    subpd_pfs: Dict[str, Optional[float]],
    per_pair_pfs: Dict[str, Optional[float]],
) -> Tuple[bool, List[str]]:
    """
    Returns (passes: bool, reasons: list of failure strings).
    Empty reasons list means QUALIFIES.
    """
    reasons = []

    # 1. Full-sample pooled PF ≥ 1.2
    if full_pf is None or full_pf < QUAL_FULL_PF:
        reasons.append(
            f"Full-sample pooled PF {full_pf} < {QUAL_FULL_PF}"
        )

    # 2. PF > 1.0 in EACH of the 3 sub-periods
    for label, pf in subpd_pfs.items():
        if pf is None or pf <= QUAL_SUBPD_PF:
            reasons.append(
                f"Sub-period {label} PF {pf} ≤ {QUAL_SUBPD_PF}"
            )

    # 3. PF > 1.0 in ≥ 6 pairs
    pairs_above = sum(
        1 for p, pf in per_pair_pfs.items()
        if pf is not None and pf > 1.0
    )
    n_pairs_total = len(per_pair_pfs)
    if pairs_above < QUAL_MIN_PAIRS:
        reasons.append(
            f"Only {pairs_above}/{n_pairs_total} pairs with PF > 1.0 "
            f"(need {QUAL_MIN_PAIRS})"
        )

    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------------------
# Per-pair signal computation (single pair, single signal)
# ---------------------------------------------------------------------------

def run_signal_on_pair(
    close_daily: pd.Series,
    close_4h: pd.Series,
    pair: str,
    signal_family: str,
    param_label: str,
    param_value,
    cost_pips: float = 2.0,
) -> Tuple[pd.DataFrame, dict, Dict[str, Optional[float]]]:
    """
    Run one signal variant on one pair (daily or H4).
    Returns (trades_df, metrics_dict, subperiod_pfs).
    """
    if signal_family == "TSMOM_DAILY":
        close = close_daily
        position = tsmom_signal(close, k=param_value)
    elif signal_family == "TSMOM_H4":
        close = close_4h
        position = tsmom_signal(close, k=param_value)
    elif signal_family == "DONCHIAN_DAILY":
        close = close_daily
        position = donchian_signal(close, n=param_value)
    elif signal_family == "DONCHIAN_H4":
        close = close_4h
        position = donchian_signal(close, n=param_value)
    elif signal_family == "MA_DAILY":
        close = close_daily
        position = ma_trend_signal(close)
    elif signal_family == "MA_H4":
        close = close_4h
        position = ma_trend_signal(close)
    else:
        raise ValueError(f"Unknown signal family: {signal_family}")

    trades_df = extract_trades(close, position, pair, cost_pips=cost_pips)
    metrics   = compute_metrics(trades_df, close, position, pair, cost_pips=cost_pips)
    sp_pfs    = subperiod_pf(trades_df)
    return trades_df, metrics, sp_pfs


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pf(pf):
    if pf is None:
        return "  N/A "
    if pf == float("inf"):
        return "  inf "
    return f"{pf:6.3f}"

def fmt_pct(v):
    if v is None:
        return "  N/A "
    return f"{v*100:5.1f}%"

def fmt_f(v, fmt=".2f"):
    if v is None:
        return "  N/A "
    return format(v, fmt)


# ---------------------------------------------------------------------------
# Main probe runner
# ---------------------------------------------------------------------------

def run_probe(smoke_only: bool = False, pairs_override: Optional[List] = None):
    pairs_to_use = pairs_override if pairs_override is not None else PAIRS
    print("=" * 80)
    print("HTF EDGE PROBE — Pre-registered experiment (see HTF_EDGE_PROBE.md)")
    print(f"Window: {WINDOW_START} → {WINDOW_END}")
    print(f"Pairs:  {len(pairs_to_use)} (spec says 8 but lists 9; GBPJPY excluded per spec)")
    print(f"Cost:   2.0 pips round-trip  |  Stress test: 4.0 pips")
    print(f"QUALIFIES requires: pooled PF ≥ {QUAL_FULL_PF}, PF > 1.0 in all 3 "
          f"sub-periods, PF > 1.0 in ≥ {QUAL_MIN_PAIRS} pairs")
    if smoke_only:
        print("MODE:   SMOKE TEST — EURUSD only, full 2020-2026 window")
    print("=" * 80)
    print()

    conn = get_connection()

    if smoke_only:
        pairs_to_run = pairs_to_use[:1]
    else:
        pairs_to_run = pairs_to_use

    # -----------------------------------------------------------------------
    # Define all signal variants
    # -----------------------------------------------------------------------
    signal_variants = []
    for k in TSMOM_DAILY_K:
        signal_variants.append(("TSMOM_DAILY", f"K{k}", k))
    for k in TSMOM_H4_K:
        signal_variants.append(("TSMOM_H4", f"K{k}", k))
    for n in DONCHIAN_DAILY_N:
        signal_variants.append(("DONCHIAN_DAILY", f"N{n}", n))
    for n in DONCHIAN_H4_N:
        signal_variants.append(("DONCHIAN_H4", f"N{n}", n))
    signal_variants.append(("MA_DAILY", "EMA50/200", None))
    signal_variants.append(("MA_H4",    "EMA50/200", None))

    # Family grouping for QUALIFIES reporting
    families = {
        "TSMOM": ["TSMOM_DAILY", "TSMOM_H4"],
        "DONCHIAN": ["DONCHIAN_DAILY", "DONCHIAN_H4"],
        "MA_TREND": ["MA_DAILY", "MA_H4"],
    }

    # -----------------------------------------------------------------------
    # Load candles once per pair
    # -----------------------------------------------------------------------
    print("Loading candles...")
    pair_data = {}
    for pair_name, epic in pairs_to_run:
        print(f"  {pair_name} ({epic})...", end=" ", flush=True)
        df_4h = load_4h_candles(conn, epic)
        df_daily = resample_4h_to_daily(df_4h)
        pair_data[pair_name] = {
            "close_4h":    df_4h["close"],
            "close_daily": df_daily["close"],
        }
        print(f"4H bars: {len(df_4h):,}  |  Daily bars: {len(df_daily):,}")

    conn.close()
    print()

    # -----------------------------------------------------------------------
    # Run all signal variants on all pairs, collect results
    # -----------------------------------------------------------------------
    # results[sig_family][param_label][pair_name] = (trades_df, metrics, sp_pfs)
    results: Dict[str, Dict[str, Dict[str, tuple]]] = {}

    for sig_family, param_label, param_value in signal_variants:
        results.setdefault(sig_family, {}).setdefault(param_label, {})

    for pair_name, _ in pairs_to_run:
        close_daily = pair_data[pair_name]["close_daily"]
        close_4h    = pair_data[pair_name]["close_4h"]

        for sig_family, param_label, param_value in signal_variants:
            trades_df, metrics, sp_pfs = run_signal_on_pair(
                close_daily, close_4h, pair_name,
                sig_family, param_label, param_value,
                cost_pips=2.0,
            )
            results[sig_family][param_label][pair_name] = (trades_df, metrics, sp_pfs)

    # -----------------------------------------------------------------------
    # Print per-signal-variant report
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("PER-SIGNAL × PAIR RESULTS (2.0-pip cost)")
    print("=" * 80)

    for sig_family, param_label, param_value in signal_variants:
        label = f"{sig_family} {param_label}"
        print(f"\n--- {label} ---")
        print(f"{'Pair':<10} {'N':>6} {'WR':>7} {'PF':>8} {'AvgPips':>9} "
              f"{'Sharpe':>8} {'Tr/day':>8}  |  "
              f"{'2020-21':>8} {'2022-23':>8} {'2024-26':>8}")
        print("-" * 86)

        for pair_name, _ in pairs_to_run:
            trades_df, m, sp_pfs = results[sig_family][param_label][pair_name]
            sp_str = "  ".join(
                fmt_pf(sp_pfs.get(sp_label))
                for sp_label, _, _ in SUBPERIODS
            )
            print(
                f"{pair_name:<10} "
                f"{m['n_trades']:>6} "
                f"{fmt_pct(m['hit_rate']):>7} "
                f"{fmt_pf(m['pf']):>8} "
                f"{fmt_f(m['avg_pips']):>9} "
                f"{fmt_f(m['sharpe']):>8} "
                f"{fmt_f(m['trades_per_day'], '.4f'):>8}  |  "
                f"{sp_str}"
            )

    # -----------------------------------------------------------------------
    # Pooled (across-pair) metrics per signal variant
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("POOLED ACROSS PAIRS (2.0-pip cost) + 4.0-pip stress PF")
    print("=" * 80)
    print(f"{'Signal':<28} {'N_total':>8} {'PoolPF':>8} {'PoolPF4p':>10}  |  "
          f"{'2020-21':>8} {'2022-23':>8} {'2024-26':>8}  |  "
          f"{'Pairs>1.0':>10}")
    print("-" * 100)

    # Store pooled PF per variant for QUALIFIES evaluation
    pooled_summary = {}  # key = (sig_family, param_label)

    for sig_family, param_label, param_value in signal_variants:
        label = f"{sig_family} {param_label}"

        all_trades = []
        per_pair_pf = {}

        for pair_name, _ in pairs_to_run:
            trades_df, m, sp_pfs = results[sig_family][param_label][pair_name]
            if not trades_df.empty:
                all_trades.append(trades_df)
            per_pair_pf[pair_name] = m["pf"]

        pooled_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()

        # 2.0-pip pooled PF
        pooled_pf_2 = compute_pf(pooled_trades) if not pooled_trades.empty else None

        # Sub-period pooled PF
        sp_pf_pooled = subperiod_pf(pooled_trades) if not pooled_trades.empty else {
            lbl: None for lbl, _, _ in SUBPERIODS
        }

        # 4.0-pip stress: need to re-run (trade returns differ by additional 2 pips)
        # We recompute from stored trades: pips_net_4p = pips_net_2p - 2.0
        if not pooled_trades.empty:
            stress = pooled_trades.copy()
            stress["pips_net_4p"] = stress["pips_net"] - 2.0
            stress["win_4p"]      = stress["pips_net_4p"] > 0
            wins_4p   = stress.loc[stress["win_4p"], "pips_net_4p"].sum()
            losses_4p = stress.loc[~stress["win_4p"], "pips_net_4p"].abs().sum()
            pf_4p = round(wins_4p / losses_4p, 3) if losses_4p > 0 else None
        else:
            pf_4p = None

        # Pairs above PF 1.0
        pairs_above = sum(
            1 for pf in per_pair_pf.values()
            if pf is not None and pf > 1.0
        )

        sp_str = "  ".join(
            fmt_pf(sp_pf_pooled.get(sp_label))
            for sp_label, _, _ in SUBPERIODS
        )

        n_total = len(pooled_trades)

        print(
            f"{label:<28} "
            f"{n_total:>8} "
            f"{fmt_pf(pooled_pf_2):>8} "
            f"{fmt_pf(pf_4p):>10}  |  "
            f"{sp_str}  |  "
            f"{pairs_above:>3}/{len(pairs_to_run)}"
        )

        pooled_summary[(sig_family, param_label)] = {
            "full_pf":     pooled_pf_2,
            "sp_pfs":      sp_pf_pooled,
            "per_pair_pf": per_pair_pf,
            "n_total":     n_total,
            "pf_4p":       pf_4p,
            "pairs_above": pairs_above,
        }

    # -----------------------------------------------------------------------
    # Per-pair PF breakdown matrix
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("PER-PAIR PF MATRIX — one column per variant (not pooled), 2.0-pip cost")
    print("=" * 80)

    for family_name, fam_signal_keys in families.items():
        print(f"\n  {family_name}")
        # collect all variants in this family
        fam_variants = [
            (sig_family, param_label)
            for sig_family, param_label, _ in signal_variants
            if sig_family in fam_signal_keys
        ]

        hdr = f"{'Pair':<10}" + "".join(
            f"  {f'{sf} {pl}':<18}" for sf, pl in fam_variants
        )
        print("  " + hdr)
        print("  " + "-" * len(hdr))

        for pair_name, _ in pairs_to_run:
            row = f"  {pair_name:<10}"
            for sf, pl in fam_variants:
                pf = results[sf][pl][pair_name][1]["pf"]
                indicator = " *" if (pf is not None and pf > 1.0) else "  "
                row += f"  {fmt_pf(pf)}{indicator:<16}"
            print(row)

    # -----------------------------------------------------------------------
    # QUALIFIES evaluation per signal family
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("QUALIFIES TEST — per signal variant (locked pre-registration)")
    print("  Criterion 1: Full-sample pooled PF ≥ 1.2")
    print("  Criterion 2: Pooled PF > 1.0 in EACH of 3 sub-periods (2020-21, 2022-23, 2024-26)")
    print(f"  Criterion 3: Per-pair PF > 1.0 in ≥ {QUAL_MIN_PAIRS} of {len(pairs_to_run)} pairs")
    print("=" * 80)

    any_qualifies = False
    family_qualifies: Dict[str, bool] = {f: False for f in families}

    for sig_family, param_label, param_value in signal_variants:
        key = (sig_family, param_label)
        if key not in pooled_summary:
            continue
        s = pooled_summary[key]
        passes, reasons = qualifies(s["full_pf"], s["sp_pfs"], s["per_pair_pf"])
        label = f"{sig_family} {param_label}"
        verdict = "QUALIFIES" if passes else "FAILS"
        print(f"\n  {label:<32}  =>  {verdict}")
        if not passes:
            for r in reasons:
                print(f"    ✗ {r}")
        if passes:
            any_qualifies = True
            for fam_name, fam_keys in families.items():
                if sig_family in fam_keys:
                    family_qualifies[fam_name] = True

    # -----------------------------------------------------------------------
    # Per-family summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SIGNAL FAMILY SUMMARY")
    print("=" * 80)
    for fam_name in ["TSMOM", "DONCHIAN", "MA_TREND"]:
        q = family_qualifies.get(fam_name, False)
        print(f"  {fam_name:<20}  =>  {'QUALIFIES (≥1 variant passes)' if q else 'FAILS (no variant qualifies)'}")

    # -----------------------------------------------------------------------
    # FINAL DECISION (locked)
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("FINAL DECISION")
    print("=" * 80)
    if any_qualifies:
        print()
        print("  GO-TO-BUILD: ≥1 canonical signal QUALIFIES.")
        print("  → Build one clean strategy around the qualifying survivor.")
        print("  → Re-confirm OOS on a held-out slice before monitor-only demo.")
        print("  → NO tuning to rescue near-misses.")
    else:
        print()
        print("  NO-GO: No canonical trend signal qualifies on these pairs after costs.")
        print("  → Even documented HTF trend edges do not survive after 2.0-pip cost.")
        print("  → Do NOT add filters to failing canonical signals — that is the treadmill.")
        print("  → Reconsider the lane (instrument / edge-source).")
    print()
    print("NOTE: Spec says '8 pairs' but enumerates 9 pairs. Script ran all 9 listed.")
    print("      QUALIFIES threshold kept at ≥6 pairs (majority, as per spec intent).")
    if smoke_only:
        print()
        print("*** SMOKE TEST ONLY — re-run without --smoke for full 9-pair results ***")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTF Edge Probe")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run EURUSD only (smoke test) — do NOT run full 9-pair job",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Run a single named pair only, e.g. --pair USDJPY (for targeted smoke tests)",
    )
    args = parser.parse_args()

    pairs_override = None
    if args.pair:
        pair_upper = args.pair.upper()
        matched = [(n, e) for n, e in PAIRS if n == pair_upper]
        if not matched:
            print(f"ERROR: pair '{pair_upper}' not in PAIRS list.")
            print(f"Available: {[n for n, _ in PAIRS]}")
            sys.exit(1)
        pairs_override = matched

    run_probe(smoke_only=args.smoke, pairs_override=pairs_override)
