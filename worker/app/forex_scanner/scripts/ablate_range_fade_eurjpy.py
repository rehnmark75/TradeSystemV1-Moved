#!/usr/bin/env python3
"""Gate ablation for EURJPY on the RANGE_FADE strategy.

Mirror of ablate_range_fade.py but targeting EURJPY (JPY pair, 0.01 pip size).
Inverse-ablation design: start from permissive (all filter gates relaxed),
then add each gate back one at a time and measure signal count / PF delta.

Usage:
    docker exec task-worker python /app/forex_scanner/scripts/ablate_range_fade_eurjpy.py --days 90
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


BACKTEST_CMD = ["python", "/app/forex_scanner/backtest_cli.py"]
EPIC = "CS.D.EURJPY.MINI.IP"
STRATEGY = "RANGE_FADE"


BASELINE: Dict[str, object] = {
    "erf_profile": "5m",
    "monitor_only": "false",
    "rsi_oversold": 50,
    "rsi_overbought": 50,
    "london_start_hour_utc": 0,
    "new_york_end_hour_utc": 23,
    "range_proximity_pips": 999.0,
    "max_current_range_pips": 999.0,
    "min_band_width_pips": 0.0,
    "max_band_width_pips": 9999.0,
    "allow_neutral_htf": "true",
    "signal_cooldown_minutes": 0,
    "fixed_stop_loss_pips": 10,
    "fixed_take_profit_pips": 15,
}


ABLATIONS: List[Dict[str, object]] = [
    {"__label__": "PERMISSIVE baseline (no filter gates)"},
    {"__label__": "+ range_proximity=3pips", "range_proximity_pips": 3.0},
    {"__label__": "+ max_current_range=12pips", "max_current_range_pips": 12.0},
    {"__label__": "+ band_width 6..28", "min_band_width_pips": 6.0, "max_band_width_pips": 28.0},
    {"__label__": "+ session 6..18 UTC", "london_start_hour_utc": 6, "new_york_end_hour_utc": 18},
    {"__label__": "+ RSI 32/68 extremes", "rsi_oversold": 32, "rsi_overbought": 68},
    {"__label__": "+ strict HTF (no neutral)", "allow_neutral_htf": "false"},
    {"__label__": "+ cooldown 30min", "signal_cooldown_minutes": 30},
    {
        "__label__": "ALL gates on (shipped 5m config)",
        "range_proximity_pips": 3.0,
        "max_current_range_pips": 12.0,
        "min_band_width_pips": 6.0,
        "max_band_width_pips": 28.0,
        "london_start_hour_utc": 6,
        "new_york_end_hour_utc": 18,
        "rsi_oversold": 32,
        "rsi_overbought": 68,
        "allow_neutral_htf": "false",
        "signal_cooldown_minutes": 30,
    },
]


@dataclass
class RunResult:
    label: str
    days: int
    overrides: Dict[str, object]
    total_signals: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    ok: bool = True
    error: str = ""


def _override_args(overrides: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in overrides.items():
        if key.startswith("__"):
            continue
        args.extend(["--override", f"{key}={value}"])
    return args


def _parse_float_last(text: str, label: str) -> Optional[float]:
    matches = re.findall(rf"{re.escape(label)}:\s+(-?\d+(?:\.\d+)?)", text)
    return float(matches[-1]) if matches else None


def _parse_int_last(text: str, label: str) -> Optional[int]:
    matches = re.findall(rf"{re.escape(label)}:\s+(\d+)", text)
    return int(matches[-1]) if matches else None


def run_backtest(label: str, days: int, overrides: Dict[str, object]) -> RunResult:
    cmd = BACKTEST_CMD + ["--epic", EPIC, "--days", str(days), "--strategy", STRATEGY] + _override_args(overrides)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        return RunResult(label=label, days=days, overrides=overrides, ok=False, error=out[-1500:])

    total = _parse_int_last(out, "Total Signals")
    if total is None or total == 0:
        return RunResult(label=label, days=days, overrides=overrides, total_signals=0)

    winners = _parse_int_last(out, "Winners") or 0
    losers = _parse_int_last(out, "Losers") or 0
    wr = _parse_float_last(out, "Win Rate") or 0.0
    pf = _parse_float_last(out, "Profit Factor") or 0.0
    exp = _parse_float_last(out, "Expectancy") or 0.0

    return RunResult(
        label=label, days=days, overrides=overrides,
        total_signals=total, winners=winners, losers=losers,
        win_rate=wr, profit_factor=pf, expectancy=exp,
    )


def signals_per_month(res: RunResult) -> float:
    if res.days <= 0:
        return 0.0
    return res.total_signals * 30.0 / res.days


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    results: List[RunResult] = []
    baseline_res: Optional[RunResult] = None

    for i, ablation in enumerate(ABLATIONS, start=1):
        label = str(ablation.get("__label__", f"ablation_{i}"))
        overrides = {**BASELINE, **{k: v for k, v in ablation.items() if not k.startswith("__")}}
        print(f"[{i:02d}/{len(ABLATIONS)}] running: {label}", flush=True)
        res = run_backtest(label, args.days, overrides)

        if not res.ok:
            print(f"    FAILED: {res.error[-400:]}", flush=True)
        else:
            print(
                f"    n={res.total_signals:4d}  "
                f"sig/mo={signals_per_month(res):6.1f}  "
                f"pf={res.profit_factor:5.2f}  "
                f"wr={res.win_rate:5.1f}%  "
                f"exp={res.expectancy:6.2f}",
                flush=True,
            )

        results.append(res)
        if label.startswith("PERMISSIVE"):
            baseline_res = res

    print("\n" + "=" * 100, flush=True)
    print(f"ABLATION REPORT — EURJPY RANGE_FADE, {args.days}d window", flush=True)
    print("=" * 100, flush=True)
    print(f"{'gate added':<42} {'n':>5} {'sig/mo':>8} {'pf':>6} {'wr%':>6} {'exp':>7} {'d(sig/mo)':>10} {'d(pf)':>7}", flush=True)
    print("-" * 100, flush=True)

    base_spm = signals_per_month(baseline_res) if baseline_res and baseline_res.ok else 0.0
    base_pf = baseline_res.profit_factor if baseline_res and baseline_res.ok else 0.0

    for res in results:
        if not res.ok:
            print(f"{res.label:<42} FAILED", flush=True)
            continue
        spm = signals_per_month(res)
        d_spm = spm - base_spm
        d_pf = res.profit_factor - base_pf
        print(
            f"{res.label:<42} {res.total_signals:>5d} {spm:>8.1f} "
            f"{res.profit_factor:>6.2f} {res.win_rate:>6.1f} {res.expectancy:>7.2f} "
            f"{d_spm:>+10.1f} {d_pf:>+7.2f}",
            flush=True,
        )

    print("\nInterpretation (cumulative add-gate design):", flush=True)
    print("  * Baseline = permissive, NO filter gates applied. Each row adds ONE gate on top.", flush=True)
    print("  * d(sig/mo) large negative + d(pf) positive -> gate kills trades but lifts edge (KEEP)", flush=True)
    print("  * d(sig/mo) large negative + d(pf) ~0 or negative -> gate is pure sample-size killer (DROP)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
