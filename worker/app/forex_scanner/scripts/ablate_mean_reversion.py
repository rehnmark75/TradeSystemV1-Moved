#!/usr/bin/env python3
"""Gate ablation for the MEAN_REVERSION strategy (epic-parameterized).

Inverse-ablation design: start from a PERMISSIVE baseline (all filter gates
relaxed), then add each gate back one at a time. Reports signals/month and PF
delta vs. baseline so you can see which gates carry edge vs. which just shrink
sample size.

MEAN_REVERSION has five real filter gates in detect_signal():
    1. hard ADX primary (15m)        — adx_hard_ceiling_primary
    2. hard ADX HTF    (1h)          — adx_hard_ceiling_htf
    3. RSI extremity                 — rsi_oversold / rsi_overbought
    4. BB band extremity             — bb_mult (close must sit past ±mult·σ)
    5. signal cooldown               — signal_cooldown_minutes

Divergence, BB-touch, SR proximity, and min-oscillator-agreement exist as DB
columns but are NOT enforced at signal time — they only influence confidence.

JPY-pair awareness: SL/TP default to 15/22 pips for JPY crosses and 12/18 for
majors, matching the shipped v1.0 MEAN_REVERSION defaults. Override with
--sl/--tp if needed.

Usage:
    docker exec task-worker python /app/forex_scanner/scripts/ablate_mean_reversion.py --epic CS.D.USDCAD.MINI.IP
    docker exec task-worker python /app/forex_scanner/scripts/ablate_mean_reversion.py --epic CS.D.EURJPY.MINI.IP --days 90
    docker exec task-worker python /app/forex_scanner/scripts/ablate_mean_reversion.py --epic CS.D.USDCHF.MINI.IP --sl 10 --tp 15
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


BACKTEST_CMD = ["python", "/app/forex_scanner/backtest_cli.py"]
STRATEGY = "MEAN_REVERSION"


def default_sl_tp(epic: str) -> tuple:
    """JPY crosses have 10x larger pip values — use wider defaults."""
    return (15, 22) if "JPY" in epic.upper() else (12, 18)


def build_baseline(sl: int, tp: int) -> Dict[str, object]:
    """Permissive baseline: every filter gate disabled or maxed out."""
    return {
        # Pair must be on during backtest (some pairs ship with is_enabled=false)
        "is_enabled": "true",
        "monitor_only": "false",
        # Hard ADX gate — turn both ceilings sky-high AND flip master off for safety
        "hard_adx_gate_enabled": "false",
        "adx_hard_ceiling_primary": 999.0,
        "adx_hard_ceiling_htf": 999.0,
        # RSI thresholds at midpoint — direction trigger satisfied on any dip/rip
        "rsi_oversold": 50,
        "rsi_overbought": 50,
        # BB band tight — 1σ instead of 2σ means close crosses it far more often
        "bb_mult": 1.0,
        # No cooldown
        "signal_cooldown_minutes": 0,
        # Fixed SL/TP so R:R is constant across all cells
        "fixed_stop_loss_pips": sl,
        "fixed_take_profit_pips": tp,
    }


ABLATIONS: List[Dict[str, object]] = [
    {"__label__": "PERMISSIVE baseline (no filter gates)"},
    {"__label__": "+ ADX primary ceiling 22",
     "hard_adx_gate_enabled": "true", "adx_hard_ceiling_primary": 22.0},
    {"__label__": "+ ADX HTF ceiling 25",
     "hard_adx_gate_enabled": "true", "adx_hard_ceiling_htf": 25.0},
    {"__label__": "+ ADX both (primary 22, HTF 25)",
     "hard_adx_gate_enabled": "true",
     "adx_hard_ceiling_primary": 22.0, "adx_hard_ceiling_htf": 25.0},
    {"__label__": "+ RSI 30/70 extremes",
     "rsi_oversold": 30, "rsi_overbought": 70},
    {"__label__": "+ RSI 35/65 (softer)",
     "rsi_oversold": 35, "rsi_overbought": 65},
    {"__label__": "+ BB mult 2.0 (std 2σ band)",
     "bb_mult": 2.0},
    {"__label__": "+ BB mult 2.5 (wider band)",
     "bb_mult": 2.5},
    {"__label__": "+ cooldown 45min",
     "signal_cooldown_minutes": 45},
    {
        "__label__": "ALL gates on (shipped v1.0 config)",
        "hard_adx_gate_enabled": "true",
        "adx_hard_ceiling_primary": 22.0,
        "adx_hard_ceiling_htf": 25.0,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_mult": 2.0,
        "signal_cooldown_minutes": 45,
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


def run_backtest(label: str, days: int, overrides: Dict[str, object], epic: str) -> RunResult:
    cmd = BACKTEST_CMD + ["--epic", epic, "--days", str(days), "--strategy", STRATEGY] + _override_args(overrides)
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
    parser.add_argument("--epic", required=True, help="IG epic, e.g. CS.D.USDCAD.MINI.IP")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--sl", type=int, default=None, help="Fixed SL pips (auto-selected by JPY if omitted)")
    parser.add_argument("--tp", type=int, default=None, help="Fixed TP pips (auto-selected by JPY if omitted)")
    args = parser.parse_args()

    default_sl, default_tp = default_sl_tp(args.epic)
    sl = args.sl if args.sl is not None else default_sl
    tp = args.tp if args.tp is not None else default_tp
    baseline = build_baseline(sl, tp)
    pair_label = args.epic.split(".")[2] if args.epic.count(".") >= 2 else args.epic

    print(f"[ablate_mean_reversion] epic={args.epic} days={args.days} SL/TP={sl}/{tp}", flush=True)

    results: List[RunResult] = []
    baseline_res: Optional[RunResult] = None

    for i, ablation in enumerate(ABLATIONS, start=1):
        label = str(ablation.get("__label__", f"ablation_{i}"))
        overrides = {**baseline, **{k: v for k, v in ablation.items() if not k.startswith("__")}}
        print(f"[{i:02d}/{len(ABLATIONS)}] running: {label}", flush=True)
        res = run_backtest(label, args.days, overrides, args.epic)

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
    print(f"ABLATION REPORT — {pair_label} MEAN_REVERSION, {args.days}d window (SL/TP={sl}/{tp})", flush=True)
    print("=" * 100, flush=True)
    print(f"{'gate relaxed':<42} {'n':>5} {'sig/mo':>8} {'pf':>6} {'wr%':>6} {'exp':>7} {'d(sig/mo)':>10} {'d(pf)':>7}", flush=True)
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
    print("  * Baseline = permissive, NO filter gates applied. Then each row adds ONE gate on top.", flush=True)
    print("  * d(sig/mo) large negative + d(pf) positive -> gate kills trades but lifts edge (KEEP)", flush=True)
    print("  * d(sig/mo) large negative + d(pf) ~0 or negative -> gate is pure sample-size killer (DROP)", flush=True)
    print("  * d(sig/mo) small -> gate rarely binds, low cost either way", flush=True)
    print("  * Watch ADX pair vs. HTF separately — one may carry edge while the other just culls.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
