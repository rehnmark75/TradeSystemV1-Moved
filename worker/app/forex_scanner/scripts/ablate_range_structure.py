#!/usr/bin/env python3
"""Gate ablation for the RANGE_STRUCTURE strategy.

Inverse-ablation design: start from a permissive baseline, then add the main
RANGE_STRUCTURE gates back one at a time. This reuses the standalone
`eval_range_structure.py` execution path instead of touching the shared
backtest CLI or any currently running ablation scripts.

Usage (inside task-worker container):
    python /app/forex_scanner/scripts/ablate_range_structure.py --days 90
    python /app/forex_scanner/scripts/ablate_range_structure.py --pairs USDJPY,EURJPY
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2

# Allow `python script.py` to import project modules when run standalone.
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]  # .../worker/app
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forex_scanner.core.strategies.range_structure_strategy import RangeStructureStrategy
from forex_scanner.scripts.eval_range_structure import (
    DEFAULT_BASKET,
    FOREX_DB,
    aggregate,
    load_candles,
    resolve_epics,
    walk_and_simulate,
)
from forex_scanner.services.range_structure_config_service import get_range_structure_config


@dataclass
class RunResult:
    label: str
    days: int
    overrides: Dict[str, object]
    total_signals: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pips: float = 0.0
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    rr_realized: Optional[float] = None
    per_pair: Dict[str, Dict[str, float]] | None = None
    ok: bool = True
    error: str = ""


def build_baseline() -> Dict[str, object]:
    return {
        "adx_hard_ceiling_primary": 999.0,
        "adx_hard_ceiling_htf": 999.0,
        "sweep_penetration_pips": 0.0,
        "rejection_wick_ratio": 0.55,  # code clamp floor
        "ob_fvg_confluence_required": False,
        "min_rr_ratio": 1.00,
        "htf_bias_neutral_band": 0.50,
        "signal_cooldown_minutes": 0,
        "sl_pips_min": 4.0,
        "sl_pips_max": 20.0,
        "tp_pips_min": 6.0,
        "tp_pips_max": 30.0,
    }


ABLATIONS: List[Dict[str, object]] = [
    {"__label__": "PERMISSIVE baseline (no filter gates)"},
    {"__label__": "+ ADX ceilings 20/22", "adx_hard_ceiling_primary": 20.0, "adx_hard_ceiling_htf": 22.0},
    {"__label__": "+ sweep penetration 1.5p", "sweep_penetration_pips": 1.5},
    {"__label__": "+ wick ratio 0.60", "rejection_wick_ratio": 0.60},
    {"__label__": "+ HTF bias neutral band 0.40", "htf_bias_neutral_band": 0.40},
    {"__label__": "+ OB/FVG confluence required", "ob_fvg_confluence_required": True},
    {"__label__": "+ R:R floor 1.33", "min_rr_ratio": 1.33},
    {"__label__": "+ cooldown 60m", "signal_cooldown_minutes": 60},
    {
        "__label__": "STACK wick 0.60 + ADX 20/22",
        "rejection_wick_ratio": 0.60,
        "adx_hard_ceiling_primary": 20.0,
        "adx_hard_ceiling_htf": 22.0,
    },
    {
        "__label__": "STACK wick 0.60 + cooldown 60m",
        "rejection_wick_ratio": 0.60,
        "signal_cooldown_minutes": 60,
    },
    {
        "__label__": "STACK wick 0.60 + ADX 20/22 + cooldown 60m",
        "rejection_wick_ratio": 0.60,
        "adx_hard_ceiling_primary": 20.0,
        "adx_hard_ceiling_htf": 22.0,
        "signal_cooldown_minutes": 60,
    },
    {
        "__label__": "STACK wick 0.60 + sweep 1.5p",
        "rejection_wick_ratio": 0.60,
        "sweep_penetration_pips": 1.5,
    },
    {
        "__label__": "ALL gates on (shipped config)",
        "adx_hard_ceiling_primary": 20.0,
        "adx_hard_ceiling_htf": 22.0,
        "sweep_penetration_pips": 1.5,
        "rejection_wick_ratio": 0.60,
        "htf_bias_neutral_band": 0.40,
        "ob_fvg_confluence_required": True,
        "min_rr_ratio": 1.33,
        "signal_cooldown_minutes": 60,
        "sl_pips_min": 6.0,
        "sl_pips_max": 12.0,
        "tp_pips_min": 10.0,
        "tp_pips_max": 18.0,
    },
]


def signals_per_month(total_signals: int, days: int) -> float:
    if days <= 0:
        return 0.0
    return total_signals * 30.0 / days


def format_pf(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def _config_with_overrides(overrides: Dict[str, object]):
    cfg = get_range_structure_config()
    fields = {k: v for k, v in overrides.items() if not k.startswith("__")}
    return replace(cfg, **fields)


def run_eval(
    label: str,
    days: int,
    pairs: List[Tuple[str, str]],
    max_bars_held: int,
    overrides: Dict[str, object],
) -> RunResult:
    try:
        cfg = _config_with_overrides(overrides)
        strategy = RangeStructureStrategy(config=cfg)
        all_trades: List[pd.DataFrame] = []
        per_pair: Dict[str, Dict[str, float]] = {}

        with psycopg2.connect(os.getenv("FOREX_DATABASE_URL", FOREX_DB)) as conn:
            for short, epic in pairs:
                _, df15, df1h = load_candles(conn, epic, days)
                if df15.empty or df1h.empty:
                    per_pair[short] = {"n": 0, "pf": 0.0, "wr": 0.0, "pips": 0.0}
                    continue
                trades = walk_and_simulate(
                    strategy=strategy,
                    df15=df15,
                    df1h=df1h,
                    epic=epic,
                    short=short,
                    max_bars_held=max_bars_held,
                )
                agg = aggregate(trades)
                per_pair[short] = {
                    "n": int(agg["n"]),
                    "pf": 0.0 if agg["pf"] is None else float(agg["pf"]),
                    "wr": float((agg["wr"] or 0.0) * 100.0),
                    "pips": float(agg["total_pips"]),
                }
                if not trades.empty:
                    trades["epic"] = epic
                    all_trades.append(trades)

        if not all_trades:
            return RunResult(
                label=label, days=days, overrides=overrides, total_signals=0, per_pair=per_pair
            )

        agg = aggregate(pd.concat(all_trades, ignore_index=True))
        return RunResult(
            label=label,
            days=days,
            overrides=overrides,
            total_signals=int(agg["n"]),
            win_rate=float((agg["wr"] or 0.0) * 100.0),
            profit_factor=float("inf") if agg["pf"] is None else float(agg["pf"]),
            total_pips=float(agg["total_pips"]),
            avg_win=agg["avg_win"],
            avg_loss=agg["avg_loss"],
            rr_realized=agg["rr_realized"],
            per_pair=per_pair,
        )
    except Exception as exc:
        return RunResult(label=label, days=days, overrides=overrides, ok=False, error=str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_BASKET))
    parser.add_argument("--max-bars-held", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = resolve_epics(args.pairs)
    if not pairs:
        print("[ablate_range_structure] no valid pairs to test")
        return 1

    print(
        f"[ablate_range_structure] pairs={','.join(short for short, _ in pairs)} "
        f"days={args.days} max_bars_held={args.max_bars_held}",
        flush=True,
    )

    results: List[RunResult] = []
    baseline_res: Optional[RunResult] = None
    baseline = build_baseline()

    for i, ablation in enumerate(ABLATIONS, start=1):
        label = str(ablation.get("__label__", f"ablation_{i}"))
        overrides = {**baseline, **{k: v for k, v in ablation.items() if not k.startswith("__")}}
        print(f"[{i:02d}/{len(ABLATIONS)}] running: {label}", flush=True)
        res = run_eval(label, args.days, pairs, args.max_bars_held, overrides)

        if not res.ok:
            print(f"    FAILED: {res.error}", flush=True)
        else:
            print(
                f"    n={res.total_signals:4d}  "
                f"sig/mo={signals_per_month(res.total_signals, res.days):6.1f}  "
                f"pf={format_pf(res.profit_factor):>5}  "
                f"wr={res.win_rate:5.1f}%  "
                f"pips={res.total_pips:7.1f}",
                flush=True,
            )

        results.append(res)
        if label.startswith("PERMISSIVE"):
            baseline_res = res

    print("\n" + "=" * 112, flush=True)
    print(f"ABLATION REPORT — RANGE_STRUCTURE, {args.days}d window, pairs={','.join(short for short, _ in pairs)}", flush=True)
    print("=" * 112, flush=True)
    print(
        f"{'gate added':<40} {'n':>5} {'sig/mo':>8} {'pf':>6} {'wr%':>6} "
        f"{'pips':>8} {'d(sig/mo)':>10} {'d(pf)':>8}",
        flush=True,
    )
    print("-" * 112, flush=True)

    base_spm = signals_per_month(baseline_res.total_signals, baseline_res.days) if baseline_res and baseline_res.ok else 0.0
    base_pf = baseline_res.profit_factor if baseline_res and baseline_res.ok else 0.0

    for res in results:
        if not res.ok:
            print(f"{res.label:<40} FAILED", flush=True)
            continue
        spm = signals_per_month(res.total_signals, res.days)
        d_spm = spm - base_spm
        d_pf = res.profit_factor - base_pf if not (math.isinf(res.profit_factor) or math.isinf(base_pf)) else float("nan")
        d_pf_str = "n/a" if math.isnan(d_pf) else f"{d_pf:+.2f}"
        print(
            f"{res.label:<40} {res.total_signals:>5d} {spm:>8.1f} "
            f"{format_pf(res.profit_factor):>6} {res.win_rate:>6.1f} "
            f"{res.total_pips:>8.1f} {d_spm:>+10.1f} {d_pf_str:>8}",
            flush=True,
        )

    print("\nInterpretation (cumulative add-gate design):", flush=True)
    print("  * Baseline = permissive, then each row adds one shipped RANGE_STRUCTURE gate.", flush=True)
    print("  * Large negative d(sig/mo) with flat/negative d(pf) => gate is mostly choking sample size.", flush=True)
    print("  * Large negative d(sig/mo) with positive d(pf) => gate likely carries edge.", flush=True)
    print("  * Use the final ALL-gates row as the nearest match to current shipped config.", flush=True)

    print("\n" + "=" * 112, flush=True)
    print("PER-EPIC PF MATRIX", flush=True)
    print("=" * 112, flush=True)
    header = f"{'setting':<32} {'pair':<8} {'n':>5} {'pf':>6} {'wr%':>6} {'pips':>8}"
    print(header, flush=True)
    print("-" * 112, flush=True)
    for res in results:
        if not res.ok:
            print(f"{res.label:<32} FAILED", flush=True)
            continue
        pair_metrics = res.per_pair or {}
        for short, _ in pairs:
            metrics = pair_metrics.get(short, {"n": 0, "pf": 0.0, "wr": 0.0, "pips": 0.0})
            print(
                f"{res.label:<32} {short:<8} {int(metrics['n']):>5d} {format_pf(float(metrics['pf'])):>6} "
                f"{float(metrics['wr']):>6.1f} {float(metrics['pips']):>8.1f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
