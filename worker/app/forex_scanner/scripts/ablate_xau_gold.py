#!/usr/bin/env python3
"""Gate ablation for the XAU_GOLD strategy.

Inverse-ablation design: start from a PERMISSIVE baseline (all filter gates
relaxed / off), then add each gate back one at a time. Reports signals/month
and PF delta vs. baseline so you can see which gates carry edge vs. which
just shrink sample size.

The XAU_GOLD strategy has a richer gate surface than RANGE_FADE:
  * Regime gates:    block_ranging, block_expansion, adx_trending_threshold,
                     atr_expansion_pct
  * Session gate:    session_filter_enabled (London 07-10 + NY 13-20 UTC)
  * Tier-2 filters:  macd_filter_enabled, dxy_confluence_enabled,
                     bos_displacement_atr_mult
  * Tier-3 filters:  require_ob_or_fvg, fib_pullback_min/max
  * Post-filters:    rsi_neutral_min/max, min_confidence, signal_cooldown

SL/TP are pinned to the dataclass fixed-fallback (40/80 XAU pips, 1 pip = $0.1)
so the comparison across gates is clean — the ablation measures signal
selection, not R:R sizing.

Epic is parameterizable so that a Dukascopy backfill epic
(CS.D.CFEGOLD.DUKAS.IP) can be substituted for multi-year windows while
defaulting to the live IG epic.

Usage:
    docker exec task-worker python /app/forex_scanner/scripts/ablate_xau_gold.py
    docker exec task-worker python /app/forex_scanner/scripts/ablate_xau_gold.py --days 180
    docker exec task-worker python /app/forex_scanner/scripts/ablate_xau_gold.py \\
        --epic CS.D.CFEGOLD.DUKAS.IP --days 365
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


BACKTEST_CMD = ["python", "/app/forex_scanner/backtest_cli.py"]
STRATEGY = "XAU_GOLD"
DEFAULT_EPIC = "CS.D.CFEGOLD.CEE.IP"


def build_baseline(sl: int, tp: int) -> Dict[str, object]:
    """PERMISSIVE baseline: every filter gate switched off / relaxed.

    Keys must match XAUGoldConfig dataclass field names (snake_case) —
    apply_config_overrides() silently ignores unknown keys.
    """
    return {
        # Pin SL/TP so R:R variance doesn't confound gate deltas
        "fixed_sl_tp_override_enabled": "true",
        "fixed_stop_loss_pips": sl,
        "fixed_take_profit_pips": tp,

        # Regime — don't block anything
        "block_ranging": "false",
        "block_expansion": "false",
        "adx_trending_threshold": 0.0,
        "atr_expansion_pct": 100.0,

        # Session — accept all hours including Asian continuations
        "session_filter_enabled": "false",
        "asian_allowed": "true",

        # Tier-2 filters
        "macd_filter_enabled": "false",
        "dxy_confluence_enabled": "false",
        "bos_displacement_atr_mult": 0.0,

        # Tier-3 filters
        "require_ob_or_fvg": "false",
        "fib_pullback_min": 0.0,
        "fib_pullback_max": 1.5,

        # Post-filters
        "rsi_neutral_min": 0.0,
        "rsi_neutral_max": 100.0,
        "min_confidence": 0.0,
        "signal_cooldown_minutes": 0,
    }


# Each entry ADDS ONE gate on top of the permissive baseline.
# The final "ALL gates on" row restores the shipped v1.0 defaults so we can
# see the combined cost.
ABLATIONS: List[Dict[str, object]] = [
    {"__label__": "PERMISSIVE baseline (no filter gates)"},
    {"__label__": "+ block_ranging (ADX<20 blocked)", "block_ranging": "true"},
    {"__label__": "+ block_expansion (ATR pct>=85)",  "block_expansion": "true", "atr_expansion_pct": 85.0},
    {"__label__": "+ session 7-10 + 13-20 UTC",       "session_filter_enabled": "true", "asian_allowed": "false"},
    {"__label__": "+ MACD alignment on 1H",           "macd_filter_enabled": "true"},
    {"__label__": "+ DXY confluence",                 "dxy_confluence_enabled": "true"},
    {"__label__": "+ require OB/FVG confluence",      "require_ob_or_fvg": "true"},
    {"__label__": "+ fib zone 0.382..0.618",          "fib_pullback_min": 0.382, "fib_pullback_max": 0.618},
    {"__label__": "+ RSI neutral 40..60",             "rsi_neutral_min": 40.0, "rsi_neutral_max": 60.0},
    {"__label__": "+ ADX trending >= 25",             "adx_trending_threshold": 25.0},
    {"__label__": "+ BOS displacement >= 1.2 ATR",    "bos_displacement_atr_mult": 1.2},
    {"__label__": "+ min_confidence >= 0.58",         "min_confidence": 0.58},
    {"__label__": "+ cooldown 180 min",               "signal_cooldown_minutes": 180},
    {
        "__label__": "ALL gates on (shipped v1.0)",
        "block_ranging": "true",
        "block_expansion": "true",
        "atr_expansion_pct": 85.0,
        "session_filter_enabled": "true",
        "asian_allowed": "false",
        "macd_filter_enabled": "true",
        "dxy_confluence_enabled": "true",
        "require_ob_or_fvg": "true",
        "fib_pullback_min": 0.382,
        "fib_pullback_max": 0.618,
        "rsi_neutral_min": 40.0,
        "rsi_neutral_max": 60.0,
        "adx_trending_threshold": 25.0,
        "bos_displacement_atr_mult": 1.2,
        "min_confidence": 0.58,
        "signal_cooldown_minutes": 180,
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
    # findall()[-1] — bt emits pre-validation stats first; the authoritative
    # post-validation block comes last. This matches the parser fix documented
    # in feedback_ablation_before_launch.md memory.
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
    parser.add_argument("--epic", default=DEFAULT_EPIC,
                        help=f"IG epic (default {DEFAULT_EPIC}). Use CS.D.CFEGOLD.DUKAS.IP for multi-year windows.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--sl", type=int, default=40,
                        help="Fixed SL pips (XAU pip = 0.1 / $0.1). Default 40 pips = $4.")
    parser.add_argument("--tp", type=int, default=80,
                        help="Fixed TP pips (2.0 R:R vs default SL).")
    args = parser.parse_args()

    baseline = build_baseline(args.sl, args.tp)
    pair_label = args.epic.split(".")[2] if args.epic.count(".") >= 2 else args.epic

    print(f"[ablate_xau_gold] epic={args.epic} days={args.days} SL/TP={args.sl}/{args.tp}", flush=True)

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
    print(f"ABLATION REPORT — {pair_label} XAU_GOLD, {args.days}d window (SL/TP={args.sl}/{args.tp} XAU pips)", flush=True)
    print("=" * 100, flush=True)
    print(f"{'gate added'.ljust(42)} {'n':>5} {'sig/mo':>8} {'pf':>6} {'wr%':>6} {'exp':>7} {'d(sig/mo)':>10} {'d(pf)':>7}", flush=True)
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
    print("  * d(sig/mo) small -> gate rarely binds, low cost either way", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
