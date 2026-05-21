#!/usr/bin/env python3
"""
Donchian Channel Trend-Follow Validator - Turtle Trading adapted for FX

Classic Turtle Trading rules applied to FX 1H and 4H bars:

  System 1 (S1): 20-bar entry channel, 10-bar exit channel
  System 2 (S2): 55-bar entry channel, 20-bar exit channel
  S1-filtered:   S1 with original Turtle filter (skip next entry if last trade won)

Entry:  first close that exceeds the N-bar Donchian high (long) or low (short)
Exit:   first close that breaches the M-bar Donchian exit channel (opposite side)
Stop:   hard ATR stop (2x ATR from entry) to cap runaway losers
No pyramiding. One position at a time per pair.

Configs tested:
  s1_1h      - S1 (20/10), 1H bars
  s1_1h_filt - S1 (20/10), 1H bars, skip-if-last-won filter
  s2_1h      - S2 (55/20), 1H bars
  s2_4h      - S2 (55/20), 4H bars (broader trend)

Usage (Docker):
  docker exec -it task-worker python /app/tmp_donchian_turtle.py
  docker exec -it task-worker python /app/tmp_donchian_turtle.py --years 5
  docker exec -it task-worker python /app/tmp_donchian_turtle.py --years 5 --tf 1h
"""

import argparse
import os
from dataclasses import dataclass
import math
from typing import Optional

import pandas as pd
import psycopg2

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")

PAIRS = [
    ("CS.D.EURUSD.CEEM.IP", "EURUSD", 0.0001),
    ("CS.D.GBPUSD.MINI.IP", "GBPUSD", 0.0001),
    ("CS.D.EURJPY.MINI.IP", "EURJPY", 0.01),
    ("CS.D.GBPJPY.MINI.IP", "GBPJPY", 0.01),
    ("CS.D.USDJPY.MINI.IP", "USDJPY", 0.01),
    ("CS.D.AUDUSD.MINI.IP", "AUDUSD", 0.0001),
    ("CS.D.USDCAD.MINI.IP", "USDCAD", 0.0001),
    ("CS.D.USDCHF.MINI.IP", "USDCHF", 0.0001),
]


# Config

@dataclass
class Config:
    name:           str
    label:          str
    tf_minutes:     int    # bar timeframe in minutes (60=1H, 240=4H)
    entry_bars:     int    # Donchian entry channel lookback
    exit_bars:      int    # Donchian exit channel lookback
    atr_stop_mult:  float  # hard stop: N x ATR from entry (0 = no hard stop)
    s1_filter:      bool   # skip entry if last trade was a winner


CONFIGS = [
    Config("s1_1h",      "S1: 20/10 channel, 1H bars",
           tf_minutes=60,  entry_bars=20, exit_bars=10, atr_stop_mult=2.0, s1_filter=False),
    Config("s1_1h_filt", "S1 + skip-if-won filter, 1H bars",
           tf_minutes=60,  entry_bars=20, exit_bars=10, atr_stop_mult=2.0, s1_filter=True),
    Config("s2_1h",      "S2: 55/20 channel, 1H bars",
           tf_minutes=60,  entry_bars=55, exit_bars=20, atr_stop_mult=2.0, s1_filter=False),
    Config("s2_4h",      "S2: 55/20 channel, 4H bars",
           tf_minutes=240, entry_bars=55, exit_bars=20, atr_stop_mult=2.0, s1_filter=False),
]


# Data

def load_1m(epic: str, years: int) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    sql = """
        SELECT start_time, open, high, low, close
        FROM ig_candles_backtest
        WHERE epic = %s AND timeframe = 1
          AND start_time >= NOW() - INTERVAL %s
        ORDER BY start_time
    """
    df = pd.read_sql(sql, conn, params=(epic, f"{years} years"), parse_dates=["start_time"])
    conn.close()
    df = df.set_index("start_time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def resample_ohlc(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    return df.resample(f"{minutes}min", closed="left", label="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),    close=("close", "last"),
    ).dropna()


# Indicators

def wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"]  - pc).abs()], axis=1).max(axis=1)
    return wilder_smooth(tr, period)


# Simulation

def simulate(df: pd.DataFrame, cfg: Config, pip: float) -> list[dict]:
    """Bar-by-bar Donchian breakout simulation. One position at a time."""

    # Pre-compute channels (shift(1) = use only completed bars, no lookahead)
    entry_hi = df["high"].rolling(cfg.entry_bars).max().shift(1)
    entry_lo = df["low"].rolling(cfg.entry_bars).min().shift(1)
    exit_hi  = df["high"].rolling(cfg.exit_bars).max().shift(1)
    exit_lo  = df["low"].rolling(cfg.exit_bars).min().shift(1)
    atr      = compute_atr(df)

    trades: list[dict] = []
    direction  = "none"
    entry_px   = 0.0
    entry_idx  = pd.Timestamp("1970-01-01", tz="UTC")
    hard_stop  = 0.0
    in_trade   = False
    last_won: Optional[bool] = None  # for S1 filter

    bars = list(df.itertuples())

    for row in bars:
        ts = row.Index
        e_hi = entry_hi[ts]
        e_lo = entry_lo[ts]
        x_hi = exit_hi[ts]
        x_lo = exit_lo[ts]
        atr_val = atr[ts]

        if any(math.isnan(v) for v in [e_hi, e_lo, x_hi, x_lo, atr_val]):
            continue

        # Check exit of open position
        if in_trade:
            exit_px  = None
            exit_why = None

            if direction == "long":
                if cfg.atr_stop_mult > 0 and row.close <= hard_stop:
                    exit_px, exit_why = hard_stop, "atr_stop"
                elif row.close < x_lo:
                    exit_px, exit_why = row.close, "channel"
            else:  # short
                if cfg.atr_stop_mult > 0 and row.close >= hard_stop:
                    exit_px, exit_why = hard_stop, "atr_stop"
                elif row.close > x_hi:
                    exit_px, exit_why = row.close, "channel"

            if exit_px is not None:
                pnl = ((exit_px - entry_px) if direction == "long"
                       else (entry_px - exit_px)) / pip
                duration_bars = sum(1 for b in bars
                                    if entry_idx <= b.Index < ts)
                trades.append({
                    "entry_time":  entry_idx,
                    "exit_time":   ts,
                    "direction":   direction,
                    "pnl_pips":    round(pnl, 2),
                    "exit_why":    exit_why,
                    "year":        entry_idx.year,
                    "duration":    duration_bars,
                })
                last_won = pnl > 0
                in_trade = False
                # Don't enter on same bar as exit

        # Check entry
        if not in_trade:
            skip = cfg.s1_filter and (last_won is True)

            if not skip:
                if row.close > e_hi:
                    direction  = "long"
                    entry_px   = row.close
                    entry_idx  = ts
                    hard_stop  = entry_px - atr_val * cfg.atr_stop_mult if cfg.atr_stop_mult > 0 else -1e9
                    in_trade   = True
                elif row.close < e_lo:
                    direction  = "short"
                    entry_px   = row.close
                    entry_idx  = ts
                    hard_stop  = entry_px + atr_val * cfg.atr_stop_mult if cfg.atr_stop_mult > 0 else 1e9
                    in_trade   = True

    # Close any open position at end of data
    if in_trade and len(bars) > 0:
        last = bars[-1]
        pnl  = ((last.close - entry_px) if direction == "long"
                else (entry_px - last.close)) / pip
        trades.append({
            "entry_time": entry_idx, "exit_time": last.Index,
            "direction": direction, "pnl_pips": round(pnl, 2),
            "exit_why": "end_of_data", "year": entry_idx.year,
            "duration": len(bars),
        })

    return trades


# Output

def pf_str(w: pd.Series, l: pd.Series) -> str:
    lsum = l.abs().sum()
    return "inf" if lsum == 0 else f"{w.sum() / lsum:.2f}"


def row_stats(sub: pd.DataFrame) -> str:
    if not len(sub):
        return "n=   0"
    w = sub[sub["pnl_pips"] > 0]
    l = sub[sub["pnl_pips"] <= 0]
    wr  = len(w) / len(sub) * 100
    pf  = pf_str(w["pnl_pips"], l["pnl_pips"])
    net = sub["pnl_pips"].sum()
    aw  = w["pnl_pips"].mean() if len(w) else 0.0
    al  = l["pnl_pips"].abs().mean() if len(l) else 0.0
    return (f"n={len(sub):4d}  WR={wr:5.1f}%  PF={pf:>5s}  "
            f"AvgW={aw:+6.1f}  AvgL={al:+6.1f}  Net={net:+8.1f}")


def print_config_results(cfg: Config, results: dict, years: int) -> None:
    W = 88
    print(f"\n{'=' * W}")
    print(f"  {cfg.name.upper()}  -  {cfg.label}")
    print(f"{'-' * W}")

    combined = pd.concat(
        [pd.DataFrame(t).assign(pair=p) for p, t in results.items() if t],
        ignore_index=True,
    ) if any(results.values()) else pd.DataFrame()

    # Per-pair
    print(f"\n  Per-pair:")
    for pair_name, trades in results.items():
        if not trades:
            print(f"    {pair_name:8s}: NO TRADES")
            continue
        tdf = pd.DataFrame(trades)
        print(f"    {pair_name:8s}: {row_stats(tdf)}")

    if combined.empty:
        return

    print(f"\n  Combined ({len(combined)} trades / {years}y):")
    print(f"    {row_stats(combined)}")

    # Direction
    print(f"\n  Direction:")
    for d in ["long", "short"]:
        sub = combined[combined["direction"] == d]
        if len(sub):
            print(f"    {d:5s}: {row_stats(sub)}")

    # Year-by-year
    print(f"\n  Year-by-year (combined):")
    for yr, sub in combined.groupby("year"):
        if len(sub):
            print(f"    {yr}: {row_stats(sub)}")

    # Trade duration buckets
    print(f"\n  Trade duration (bars):")
    dur_buckets = [
        ("Short  (<  5 bars)", combined[combined["duration"] <  5]),
        ("Medium ( 5-20 bars)", combined[(combined["duration"] >= 5)  & (combined["duration"] < 20)]),
        ("Long   (20-50 bars)", combined[(combined["duration"] >= 20) & (combined["duration"] < 50)]),
        ("Trend  (50+ bars)",   combined[combined["duration"] >= 50]),
    ]
    for label, sub in dur_buckets:
        if len(sub):
            w = sub[sub["pnl_pips"] > 0]
            l = sub[sub["pnl_pips"] <= 0]
            print(f"    {label}: n={len(sub):4d}  WR={len(w)/len(sub)*100:5.1f}%  "
                  f"PF={pf_str(w['pnl_pips'], l['pnl_pips']):>5s}  "
                  f"avg={sub['pnl_pips'].mean():+.1f} pips")

    # Exit type
    print(f"\n  Exit type:")
    for etype, label in [("channel", "Channel exit"), ("atr_stop", "ATR hard stop"),
                          ("end_of_data", "End of data")]:
        sub = combined[combined["exit_why"] == etype]
        if len(sub):
            print(f"    {label:16s}: n={len(sub):4d}  avg={sub['pnl_pips'].mean():+.1f} pips")


# Main

def main():
    parser = argparse.ArgumentParser(description="Donchian Turtle FX Validator")
    parser.add_argument("--years", type=int, default=4,
                        help="Years of history (default: 4)")
    parser.add_argument("--tf",    type=str, default=None,
                        help="Filter to single timeframe: '1h' or '4h'")
    args = parser.parse_args()

    tf_filter = None
    if args.tf:
        tf_filter = 60 if args.tf.lower() == "1h" else 240

    W = 88
    print(f"\n{'=' * W}")
    print(f"  Donchian Channel Trend-Follow Validator - Turtle Trading for FX")
    print(f"  {args.years}y Dukascopy 1m -> resampled  /  {len(PAIRS)} pairs  /  {len(CONFIGS)} configs")
    print(f"{'=' * W}\n")

    # Pre-load and resample all pairs once, for both 1H and 4H
    dfs: dict[tuple, dict] = {}   # (epic, pair, pip) -> {60: df_1h, 240: df_4h}
    print("  Loading data...")
    needed_tfs = set(
        cfg.tf_minutes for cfg in CONFIGS
        if tf_filter is None or cfg.tf_minutes == tf_filter
    )
    for epic, pair_name, pip in PAIRS:
        df_1m = load_1m(epic, args.years)
        resampled = {tf: resample_ohlc(df_1m, tf) for tf in needed_tfs}
        dfs[(epic, pair_name, pip)] = resampled
        sizes = "  ".join(f"{tf}m:{len(df):,}bars" for tf, df in resampled.items())
        print(f"    {pair_name}: {sizes}")
    print()

    # Run configs
    for cfg in CONFIGS:
        if tf_filter is not None and cfg.tf_minutes != tf_filter:
            continue

        results = {}
        for (epic, pair_name, pip), resampled in dfs.items():
            df = resampled[cfg.tf_minutes]
            results[pair_name] = simulate(df, cfg, pip)

        print_config_results(cfg, results, args.years)

    print(f"\n{'=' * W}\n")


if __name__ == "__main__":
    main()
