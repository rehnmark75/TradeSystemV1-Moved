#!/usr/bin/env python3
"""Mine GBPUSD RANGE_FADE backtest signals for dynamic filter candidates."""
from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2


PIP = 0.0001


@dataclass
class Candidate:
    label: str
    n: int
    wins: int
    losses: int
    pf: float
    expectancy: float
    buy_hours: tuple[int, ...]
    sell_hours: tuple[int, ...]


def db_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")


def load_signals(execution_id: int) -> pd.DataFrame:
    sql = """
        SELECT
            signal_timestamp,
            signal_type,
            trade_result,
            pips_gained,
            confidence_score,
            entry_price,
            (indicator_values::jsonb->>'rsi')::numeric AS rsi,
            (indicator_values::jsonb->>'atr')::numeric AS atr,
            (indicator_values::jsonb->>'htf_bias') AS htf_bias,
            (indicator_values::jsonb->>'macd_histogram')::numeric AS macd_histogram,
            (indicator_values::jsonb->>'ema_21')::numeric AS ema_21,
            (indicator_values::jsonb->>'ema_50')::numeric AS ema_50,
            (indicator_values::jsonb->>'ema_200')::numeric AS ema_200
        FROM backtest_signals
        WHERE execution_id = %s
        ORDER BY signal_timestamp
    """
    conn = psycopg2.connect(db_url())
    try:
        df = pd.read_sql(sql, conn, params=(execution_id,))
    finally:
        conn.close()
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
    df["hour"] = df["signal_timestamp"].dt.hour
    df["is_win"] = df["trade_result"].eq("win")
    df["ema200_dist_pips"] = (df["entry_price"].astype(float) - df["ema_200"].astype(float)) / PIP
    df["macd_pips"] = df["macd_histogram"].astype(float).abs() / PIP
    return df


def powerset(values: Iterable[int], max_size: int) -> Iterable[tuple[int, ...]]:
    items = tuple(values)
    if not items:
        yield ()
        return
    for size in range(1, min(max_size, len(items)) + 1):
        yield from itertools.combinations(items, size)


def evaluate(
    label: str,
    pips: np.ndarray,
    wins_arr: np.ndarray,
    mask: np.ndarray,
    buy_hours: tuple[int, ...],
    sell_hours: tuple[int, ...],
) -> Candidate | None:
    n = int(mask.sum())
    if n == 0:
        return None
    wins = int(wins_arr[mask].sum())
    losses = n - wins
    kept_pips = pips[mask]
    profit = float(kept_pips[kept_pips > 0].sum())
    loss = float((-kept_pips[kept_pips < 0]).sum())
    pf = profit / loss if loss else 999.0
    expectancy = float(kept_pips.sum() / n)
    return Candidate(label, n, wins, losses, pf, expectancy, buy_hours, sell_hours)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-id", type=int, default=6742)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    df = load_signals(args.execution_id)
    buy_hours_all = sorted(df.loc[df["signal_type"].eq("BULL"), "hour"].unique())
    sell_hours_all = sorted(df.loc[df["signal_type"].eq("BEAR"), "hour"].unique())
    candidates: list[Candidate] = []
    pips = df["pips_gained"].astype(float).to_numpy()
    wins_arr = df["is_win"].to_numpy(dtype=bool)
    signal_type = df["signal_type"].to_numpy()
    hours = df["hour"].to_numpy()

    htf_modes = {
        "any_htf": np.ones(len(df), dtype=bool),
        "aligned_htf": (
            (df["signal_type"].eq("BULL") & df["htf_bias"].eq("bullish"))
            | (df["signal_type"].eq("BEAR") & df["htf_bias"].eq("bearish"))
        ).to_numpy(),
        "no_neutral": df["htf_bias"].ne("neutral").to_numpy(),
        "no_neutral_buys": (df["signal_type"].eq("BEAR") | df["htf_bias"].eq("bullish")).to_numpy(),
    }
    rsi_modes = {
        "any_rsi": np.ones(len(df), dtype=bool),
        "sell64_buy37": (
            (df["signal_type"].eq("BULL") & (df["rsi"].astype(float) <= 37))
            | (df["signal_type"].eq("BEAR") & (df["rsi"].astype(float) >= 64))
        ).to_numpy(),
        "sell65_buy38": (
            (df["signal_type"].eq("BULL") & (df["rsi"].astype(float) <= 38))
            | (df["signal_type"].eq("BEAR") & (df["rsi"].astype(float) >= 65))
        ).to_numpy(),
    }
    macd_modes = {
        "any_macd": np.ones(len(df), dtype=bool),
        "macd_abs_1p": (df["macd_pips"] >= 1.0).to_numpy(),
        "macd_abs_2p": (df["macd_pips"] >= 2.0).to_numpy(),
    }

    for buy_hours in powerset(buy_hours_all, 6):
        buy_mask = (signal_type == "BULL") & np.isin(hours, buy_hours)
        for sell_hours in powerset(sell_hours_all, 6):
            hour_mask = buy_mask | ((signal_type == "BEAR") & np.isin(hours, sell_hours))
            if int(hour_mask.sum()) < args.min_trades:
                continue
            for htf_label, htf_mask in htf_modes.items():
                for rsi_label, rsi_mask in rsi_modes.items():
                    for macd_label, macd_mask in macd_modes.items():
                        label = f"{htf_label}/{rsi_label}/{macd_label}"
                        mask = hour_mask & htf_mask & rsi_mask & macd_mask
                        candidate = evaluate(label, pips, wins_arr, mask, buy_hours, sell_hours)
                        if candidate and candidate.n >= args.min_trades:
                            candidates.append(candidate)

    candidates.sort(key=lambda c: (-abs(c.pf - 2.0), c.expectancy, c.n), reverse=True)
    by_pf = sorted(candidates, key=lambda c: (c.pf, c.expectancy, c.n), reverse=True)

    print(f"signals={len(df)} candidates={len(candidates)} min_trades={args.min_trades}")
    print("\n=== closest to PF 2 ===")
    for c in candidates[: args.top]:
        print(
            f"n={c.n:3d} pf={c.pf:5.2f} exp={c.expectancy:6.2f} W/L={c.wins}/{c.losses} "
            f"buy_hours={','.join(map(str, c.buy_hours))} sell_hours={','.join(map(str, c.sell_hours))} {c.label}"
        )
    print("\n=== highest PF ===")
    for c in by_pf[: args.top]:
        print(
            f"n={c.n:3d} pf={c.pf:5.2f} exp={c.expectancy:6.2f} W/L={c.wins}/{c.losses} "
            f"buy_hours={','.join(map(str, c.buy_hours))} sell_hours={','.join(map(str, c.sell_hours))} {c.label}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
