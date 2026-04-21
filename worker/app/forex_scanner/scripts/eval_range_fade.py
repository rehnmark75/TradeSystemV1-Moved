#!/usr/bin/env python3
"""Curated parameter sweep for the EURUSD range-fade prototype.

Runs a bounded set of strategy configurations through the existing backtest CLI,
parses the headline metrics, and prints the strongest candidates.

Usage:
    docker exec task-worker python /app/forex_scanner/scripts/eval_range_fade.py
"""
from __future__ import annotations

import itertools
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


BACKTEST_CMD = ["python", "/app/forex_scanner/backtest_cli.py"]
EPIC = "CS.D.EURUSD.CEEM.IP"
STRATEGY = "RANGE_FADE"


@dataclass
class RunResult:
    days: int
    overrides: Dict[str, object]
    total_signals: int
    winners: int
    losers: int
    win_rate: float
    profit_factor: float
    expectancy: float


def _override_args(overrides: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in overrides.items():
        args.extend(["--override", f"{key}={value}"])
    return args


def _parse_metric(text: str, label: str) -> Optional[float]:
    match = re.search(rf"{re.escape(label)}:\s+(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _parse_int_metric(text: str, label: str) -> Optional[int]:
    match = re.search(rf"{re.escape(label)}:\s+(\d+)", text)
    return int(match.group(1)) if match else None


def run_backtest(days: int, overrides: Dict[str, object]) -> Optional[RunResult]:
    cmd = BACKTEST_CMD + [
        "--epic", EPIC,
        "--days", str(days),
        "--strategy", STRATEGY,
    ] + _override_args(overrides)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(f"[FAIL] days={days} overrides={overrides}\n{output[-2000:]}\n", file=sys.stderr)
        return None

    total_signals = _parse_int_metric(output, "Total Signals")
    winners = _parse_int_metric(output, "Winners")
    losers = _parse_int_metric(output, "Losers")
    win_rate = _parse_metric(output, "Win Rate")
    profit_factor = _parse_metric(output, "Profit Factor")
    expectancy = _parse_metric(output, "Expectancy")

    if total_signals == 0 or "No signals generated during backtest" in output:
        return RunResult(
            days=days,
            overrides=overrides,
            total_signals=0,
            winners=0,
            losers=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
        )

    if None in (total_signals, winners, losers, win_rate, profit_factor, expectancy):
        print(f"[PARSE_FAIL] days={days} overrides={overrides}\n{output[-2000:]}\n", file=sys.stderr)
        return None

    return RunResult(
        days=days,
        overrides=overrides,
        total_signals=total_signals,
        winners=winners,
        losers=losers,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
    )


def curated_configs() -> Iterable[Dict[str, object]]:
    shared = {
        "monitor_only": "false",
    }

    rsi_pairs = [(30, 70), (32, 68)]
    session_windows = [(7, 20), (7, 18), (8, 20)]
    proximity = [4.0]
    max_ranges = [14.0, 16.0]
    allow_neutral = ["true"]
    sl_tp = [(8, 12), (9, 13), (8, 16)]

    for (rsi_os, rsi_ob), (session_start, session_end), prox, max_range, neutral, (sl, tp) in itertools.product(
        rsi_pairs, session_windows, proximity, max_ranges, allow_neutral, sl_tp
    ):
        yield {
            **shared,
            "rsi_oversold": rsi_os,
            "rsi_overbought": rsi_ob,
            "london_start_hour_utc": session_start,
            "new_york_end_hour_utc": session_end,
            "range_proximity_pips": prox,
            "max_current_range_pips": max_range,
            "allow_neutral_htf": neutral,
            "fixed_stop_loss_pips": sl,
            "fixed_take_profit_pips": tp,
        }


def score(result: RunResult) -> tuple:
    return (
        result.profit_factor,
        result.expectancy,
        result.win_rate,
        result.total_signals,
    )


def main() -> int:
    configs = list(curated_configs())
    print(f"[sweep] running {len(configs)} curated configs on 30d window", flush=True)

    first_pass: List[RunResult] = []
    for idx, cfg in enumerate(configs, start=1):
        res = run_backtest(30, cfg)
        if res is None:
            continue
        first_pass.append(res)
        print(
            f"[30d {idx:03d}/{len(configs)}] pf={res.profit_factor:.2f} "
            f"exp={res.expectancy:.1f} wr={res.win_rate:.1f}% n={res.total_signals} cfg={cfg}"
        , flush=True)

    if not first_pass:
        print("[sweep] no successful runs", file=sys.stderr)
        return 1

    first_pass.sort(key=score, reverse=True)
    shortlist = first_pass[:12]

    print("\n[sweep] top 12 from 30d, rerunning on 90d", flush=True)
    second_pass: List[RunResult] = []
    for idx, candidate in enumerate(shortlist, start=1):
        res = run_backtest(90, candidate.overrides)
        if res is None:
            continue
        second_pass.append(res)
        print(
            f"[90d {idx:02d}/{len(shortlist)}] pf={res.profit_factor:.2f} "
            f"exp={res.expectancy:.1f} wr={res.win_rate:.1f}% n={res.total_signals} cfg={res.overrides}"
        , flush=True)

    second_pass.sort(key=score, reverse=True)

    print("\n=== TOP 10 : 30D ===", flush=True)
    for res in first_pass[:10]:
        print(
            f"pf={res.profit_factor:.2f} exp={res.expectancy:.1f} wr={res.win_rate:.1f}% "
            f"n={res.total_signals} cfg={res.overrides}"
        , flush=True)

    print("\n=== TOP 10 : 90D ===", flush=True)
    for res in second_pass[:10]:
        print(
            f"pf={res.profit_factor:.2f} exp={res.expectancy:.1f} wr={res.win_rate:.1f}% "
            f"n={res.total_signals} cfg={res.overrides}"
        , flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
