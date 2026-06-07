#!/usr/bin/env python3
"""Evaluate the LuxAlgo-style MTF 20 pip scalper on historical candles.

This is a direct Python conversion of the Pine strategy logic:
  - Entry timeframe EMA 9/21 crossover
  - 1h EMA 200 trend filter, with completed HTF values only
  - UTC session gate 08:00-22:00
  - Fixed TP/SL of 20/10 pips
  - Daily pip goal gate of 20 pips

Usage:
  python /app/forex_scanner/scripts/eval_luxquant_mtf_scalper.py --pair USDCHF --timeframe 15 --days 60
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import psycopg2


DEFAULT_DB_URL = "postgresql://postgres:postgres@postgres:5432/forex"

PAIR_EPICS = {
    "AUDJPY": "CS.D.AUDJPY.MINI.IP",
    "AUDUSD": "CS.D.AUDUSD.MINI.IP",
    "EURJPY": "CS.D.EURJPY.MINI.IP",
    "EURUSD": "CS.D.EURUSD.CEEM.IP",
    "GBPUSD": "CS.D.GBPUSD.MINI.IP",
    "NZDUSD": "CS.D.NZDUSD.MINI.IP",
    "USDCAD": "CS.D.USDCAD.MINI.IP",
    "USDCHF": "CS.D.USDCHF.MINI.IP",
    "USDJPY": "CS.D.USDJPY.MINI.IP",
}


@dataclass(frozen=True)
class Params:
    pair: str
    epic: str
    timeframe: int
    days: int
    htf_timeframe: int = 60
    htf_ema_len: int = 200
    fast_len: int = 9
    slow_len: int = 21
    tp_pips: float = 20.0
    sl_pips: float = 10.0
    daily_goal_pips: float = 20.0
    session_start_hour: int = 8
    session_end_hour: int = 22
    max_hold_bars: int = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="USDCHF", help="Pair symbol, for example USDCHF.")
    parser.add_argument("--epic", default="", help="Override full IG epic.")
    parser.add_argument("--timeframe", type=int, default=15, help="Entry timeframe in minutes.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--tp-pips", type=float, default=20.0)
    parser.add_argument("--sl-pips", type=float, default=10.0)
    parser.add_argument("--daily-goal-pips", type=float, default=20.0)
    parser.add_argument("--max-hold-bars", type=int, default=192)
    parser.add_argument("--show-trades", action="store_true")
    return parser.parse_args()


def pip_size(epic: str) -> float:
    return 0.01 if "JPY" in epic.upper() else 0.0001


def load_candles(conn, epic: str, timeframe: int, days: int, warmup_days: int) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=days + warmup_days)
    query = """
        SELECT start_time, open, high, low, close, volume
          FROM ig_candles_backtest
         WHERE epic = %s
           AND timeframe = %s
           AND start_time >= %s
           AND start_time <= %s
         ORDER BY start_time
    """
    df = pd.read_sql(query, conn, params=(epic, timeframe, start, end))
    if df.empty:
        return df
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(None)
    return df.set_index("start_time").sort_index()


def add_indicators(entry_df: pd.DataFrame, htf_df: pd.DataFrame, params: Params) -> pd.DataFrame:
    df = entry_df.copy()
    close = df["close"].astype(float)
    df["fast_ema"] = close.ewm(span=params.fast_len, adjust=False).mean()
    df["slow_ema"] = close.ewm(span=params.slow_len, adjust=False).mean()
    df["buy_cross"] = (df["fast_ema"] > df["slow_ema"]) & (
        df["fast_ema"].shift(1) <= df["slow_ema"].shift(1)
    )
    df["sell_cross"] = (df["fast_ema"] < df["slow_ema"]) & (
        df["fast_ema"].shift(1) >= df["slow_ema"].shift(1)
    )

    htf = htf_df.copy()
    htf["htf_ema"] = htf["close"].astype(float).ewm(span=params.htf_ema_len, adjust=False).mean()

    # Pine request.security(..., lookahead_off): entry bars only see the last completed HTF bar.
    htf_completed = htf["htf_ema"].shift(1)
    df["htf_ema"] = htf_completed.reindex(df.index, method="ffill")
    df["in_session"] = (df.index.hour >= params.session_start_hour) & (
        df.index.hour < params.session_end_hour
    )
    return df


def simulate(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    pip = pip_size(params.epic)
    trades: List[Dict] = []
    daily_pips = 0.0
    current_day: Optional[datetime.date] = None
    i = 0

    while i < len(df):
        row = df.iloc[i]
        ts = df.index[i]
        day = ts.date()
        if current_day != day:
            current_day = day
            daily_pips = 0.0

        if daily_pips >= params.daily_goal_pips:
            i += 1
            continue

        if pd.isna(row["htf_ema"]) or not bool(row["in_session"]):
            i += 1
            continue

        direction = ""
        if row["close"] > row["htf_ema"] and bool(row["buy_cross"]):
            direction = "BUY"
        elif row["close"] < row["htf_ema"] and bool(row["sell_cross"]):
            direction = "SELL"

        if not direction:
            i += 1
            continue

        entry_price = float(row["close"])
        if direction == "BUY":
            tp_price = entry_price + params.tp_pips * pip
            sl_price = entry_price - params.sl_pips * pip
        else:
            tp_price = entry_price - params.tp_pips * pip
            sl_price = entry_price + params.sl_pips * pip

        outcome = "TIMEOUT"
        exit_idx = min(i + params.max_hold_bars, len(df) - 1)
        exit_price = float(df["close"].iloc[exit_idx])

        for j in range(i + 1, min(i + params.max_hold_bars + 1, len(df))):
            high = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])
            if direction == "BUY":
                # Conservative when TP and SL occur in the same candle.
                if low <= sl_price:
                    outcome = "SL"
                    exit_idx = j
                    exit_price = sl_price
                    break
                if high >= tp_price:
                    outcome = "TP"
                    exit_idx = j
                    exit_price = tp_price
                    break
            else:
                if high >= sl_price:
                    outcome = "SL"
                    exit_idx = j
                    exit_price = sl_price
                    break
                if low <= tp_price:
                    outcome = "TP"
                    exit_idx = j
                    exit_price = tp_price
                    break

        sign = 1 if direction == "BUY" else -1
        pips = (exit_price - entry_price) * sign / pip
        if outcome == "TP":
            daily_pips += params.tp_pips
        elif outcome == "SL":
            daily_pips -= params.sl_pips
        else:
            daily_pips += pips

        trades.append(
            {
                "entry_time": ts,
                "exit_time": df.index[exit_idx],
                "direction": direction,
                "entry": entry_price,
                "exit": exit_price,
                "outcome": outcome,
                "pips": pips,
                "daily_pips_after": daily_pips,
            }
        )
        i = max(exit_idx + 1, i + 1)

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, params: Params, first_ts, last_ts) -> None:
    print("LUXQUANT MTF - 20 Pips Scalper Python Test")
    print(f"Pair: {params.pair} ({params.epic})")
    print(f"Entry timeframe: {params.timeframe}m | HTF: {params.htf_timeframe}m EMA{params.htf_ema_len}")
    print(f"Window: {first_ts} to {last_ts}")
    print(f"TP/SL: {params.tp_pips:.1f}/{params.sl_pips:.1f} pips | Session: 08:00-22:00 UTC")

    if trades.empty:
        print("Trades: 0")
        return

    wins = int((trades["pips"] > 0).sum())
    losses = int((trades["pips"] < 0).sum())
    total = len(trades)
    gross_win = float(trades.loc[trades["pips"] > 0, "pips"].sum())
    gross_loss = abs(float(trades.loc[trades["pips"] < 0, "pips"].sum()))
    profit_factor = gross_win / gross_loss if gross_loss else float("inf")
    total_pips = float(trades["pips"].sum())
    avg_pips = float(trades["pips"].mean())
    max_dd = max_drawdown(trades["pips"])

    print(f"Trades: {total} | Wins: {wins} | Losses: {losses} | Win rate: {wins / total:.1%}")
    print(f"Net pips: {total_pips:.1f} | Avg/trade: {avg_pips:.2f} | PF: {profit_factor:.2f}")
    print(f"Max pip drawdown: {max_dd:.1f}")
    print("Outcomes:")
    for outcome, count in trades["outcome"].value_counts().items():
        print(f"  {outcome}: {count}")


def max_drawdown(pips: pd.Series) -> float:
    curve = pips.cumsum()
    peak = curve.cummax()
    dd = curve - peak
    return abs(float(dd.min())) if len(dd) else 0.0


def main() -> int:
    args = parse_args()
    pair = args.pair.upper()
    epic = args.epic or PAIR_EPICS.get(pair)
    if not epic:
        raise SystemExit(f"Unknown pair {pair}; pass --epic.")

    params = Params(
        pair=pair,
        epic=epic,
        timeframe=args.timeframe,
        days=args.days,
        tp_pips=args.tp_pips,
        sl_pips=args.sl_pips,
        daily_goal_pips=args.daily_goal_pips,
        max_hold_bars=args.max_hold_bars,
    )

    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    warmup_days = max(20, int(params.htf_ema_len * params.htf_timeframe / 1440) + 5)
    with psycopg2.connect(db_url) as conn:
        entry_df = load_candles(conn, params.epic, params.timeframe, params.days, warmup_days)
        htf_df = load_candles(conn, params.epic, params.htf_timeframe, params.days, warmup_days)

    if entry_df.empty:
        raise SystemExit(f"No {params.timeframe}m candles found for {params.epic}.")
    if htf_df.empty:
        raise SystemExit(f"No {params.htf_timeframe}m candles found for {params.epic}.")

    cutoff = datetime.utcnow() - timedelta(days=params.days)
    df = add_indicators(entry_df, htf_df, params)
    df = df[df.index >= cutoff]
    trades = simulate(df, params)
    summarize(trades, params, df.index.min(), df.index.max())

    if args.show_trades and not trades.empty:
        print("\nRecent trades:")
        print(trades.tail(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
