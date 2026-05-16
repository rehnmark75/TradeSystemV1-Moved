#!/usr/bin/env python3
"""GBPUSD parameter sweep for RANGE_FADE.

The sweep is deliberately bounded:
1. Test a broad but small grid on 30d.
2. Re-run the best candidates on 90d.
3. Print candidates closest to PF 2 with enough trades to be meaningful.

Usage:
    docker exec -i task-worker python /app/forex_scanner/scripts/sweep_gbpusd_range_fade.py
"""
from __future__ import annotations

import argparse
import itertools
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


BACKTEST_CMD = ["python", "/app/forex_scanner/backtest_cli.py"]
EPIC = "CS.D.GBPUSD.MINI.IP"
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


def _parse_float_last(text: str, label: str) -> Optional[float]:
    matches = re.findall(rf"{re.escape(label)}:\s+(-?\d+(?:\.\d+)?)", text)
    return float(matches[-1]) if matches else None


def _parse_int_last(text: str, label: str) -> Optional[int]:
    matches = re.findall(rf"{re.escape(label)}:\s+(\d+)", text)
    return int(matches[-1]) if matches else None


def run_backtest(days: int, overrides: Dict[str, object]) -> Optional[RunResult]:
    cmd = BACKTEST_CMD + [
        "--epic",
        EPIC,
        "--days",
        str(days),
        "--strategy",
        STRATEGY,
    ] + _override_args(overrides)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(f"[FAIL] days={days} overrides={overrides}\n{output[-2000:]}\n", file=sys.stderr)
        return None

    total = _parse_int_last(output, "Total Signals")
    if total is None or total == 0 or "No signals generated during backtest" in output:
        return RunResult(days, overrides, 0, 0, 0, 0.0, 0.0, 0.0)

    winners = _parse_int_last(output, "Winners")
    losers = _parse_int_last(output, "Losers")
    win_rate = _parse_float_last(output, "Win Rate")
    profit_factor = _parse_float_last(output, "Profit Factor")
    expectancy = _parse_float_last(output, "Expectancy")
    if None in (winners, losers, win_rate, profit_factor, expectancy):
        print(f"[PARSE_FAIL] days={days} overrides={overrides}\n{output[-2000:]}\n", file=sys.stderr)
        return None

    return RunResult(
        days=days,
        overrides=overrides,
        total_signals=total,
        winners=winners,
        losers=losers,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
    )


def curated_configs() -> Iterable[Dict[str, object]]:
    base = {
        "erf_profile": "5m",
        "monitor_only": "false",
        "allow_neutral_htf": "true",
        "signal_cooldown_minutes": 0,
    }

    rsi_pairs = [(32, 68), (34, 66), (36, 64), (38, 62), (40, 60)]
    sessions = [(6, 18), (6, 20), (7, 18), (7, 20), (8, 18)]
    directions = ["", "BUY", "SELL"]
    range_proximities = [3.0, 4.0, 5.0, 6.0]
    max_ranges = [12.0, 14.0, 16.0, 20.0]
    band_caps = [(4.0, 24.0), (5.0, 28.0), (6.0, 32.0), (8.0, 36.0)]
    rr_pairs = [(7, 12), (8, 12), (8, 14), (9, 14), (9, 16), (10, 16)]
    macd_mins = [0.0, 0.15, 0.3]
    bb_mults = [1.8, 2.0, 2.2]

    for (
        (rsi_os, rsi_ob),
        (start_hour, end_hour),
        allowed_directions,
        proximity,
        max_range,
        (min_band, max_band),
        (sl, tp),
        macd_min,
        bb_mult,
    ) in itertools.product(
        rsi_pairs,
        sessions,
        directions,
        range_proximities,
        max_ranges,
        band_caps,
        rr_pairs,
        macd_mins,
        bb_mults,
    ):
        yield {
            **base,
            "rsi_oversold": rsi_os,
            "rsi_overbought": rsi_ob,
            "london_start_hour_utc": start_hour,
            "new_york_end_hour_utc": end_hour,
            "allowed_directions": allowed_directions,
            "range_proximity_pips": proximity,
            "max_current_range_pips": max_range,
            "min_band_width_pips": min_band,
            "max_band_width_pips": max_band,
            "fixed_stop_loss_pips": sl,
            "fixed_take_profit_pips": tp,
            "min_macd_histogram_pips": macd_min,
            "bb_mult": bb_mult,
        }


def score_for_shortlist(result: RunResult, min_trades: int) -> tuple:
    if result.total_signals < min_trades:
        return (-999.0, result.profit_factor, result.expectancy)
    return (
        -abs(result.profit_factor - 2.0),
        result.expectancy,
        result.total_signals,
    )


def print_result(prefix: str, idx: int, total: int, res: RunResult) -> None:
    print(
        f"[{prefix} {idx:04d}/{total}] pf={res.profit_factor:5.2f} "
        f"exp={res.expectancy:7.2f} wr={res.win_rate:5.1f}% "
        f"n={res.total_signals:3d} cfg={res.overrides}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-days", type=int, default=30)
    parser.add_argument("--confirm-days", type=int, default=90)
    parser.add_argument("--sample", type=int, default=360, help="Deterministic sample size from the full grid")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap after sampling for quick experiments")
    parser.add_argument("--shortlist", type=int, default=24)
    parser.add_argument("--min-30d-trades", type=int, default=4)
    parser.add_argument("--min-90d-trades", type=int, default=10)
    args = parser.parse_args()

    configs = list(curated_configs())
    if args.sample > 0 and args.sample < len(configs):
        rng = random.Random(args.seed)
        configs = rng.sample(configs, args.sample)
    if args.limit > 0:
        configs = configs[: args.limit]

    print(f"[sweep] GBPUSD RANGE_FADE first pass: {len(configs)} configs on {args.first_days}d", flush=True)
    first_pass: List[RunResult] = []
    for idx, cfg in enumerate(configs, start=1):
        res = run_backtest(args.first_days, cfg)
        if res is None:
            continue
        first_pass.append(res)
        print_result(f"{args.first_days}d", idx, len(configs), res)

    if not first_pass:
        print("[sweep] no successful runs", file=sys.stderr)
        return 1

    eligible = [r for r in first_pass if r.total_signals >= args.min_30d_trades and r.profit_factor > 1.0]
    eligible.sort(key=lambda r: score_for_shortlist(r, args.min_30d_trades), reverse=True)
    shortlist = eligible[: args.shortlist]

    print(f"\n[sweep] confirming top {len(shortlist)} on {args.confirm_days}d", flush=True)
    confirmed: List[RunResult] = []
    for idx, candidate in enumerate(shortlist, start=1):
        res = run_backtest(args.confirm_days, candidate.overrides)
        if res is None:
            continue
        confirmed.append(res)
        print_result(f"{args.confirm_days}d", idx, len(shortlist), res)

    confirmed.sort(key=lambda r: score_for_shortlist(r, args.min_90d_trades), reverse=True)
    first_pass.sort(key=lambda r: score_for_shortlist(r, args.min_30d_trades), reverse=True)

    print("\n=== TOP 30D NEAR PF 2 ===", flush=True)
    for res in first_pass[:10]:
        print_result("top30", 0, 0, res)

    print("\n=== TOP CONFIRMED NEAR PF 2 ===", flush=True)
    for res in confirmed[:10]:
        print_result("top90", 0, 0, res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
