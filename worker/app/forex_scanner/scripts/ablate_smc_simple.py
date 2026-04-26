#!/usr/bin/env python3
"""Gate ablation for the SMC_SIMPLE strategy (multi-epic, scalp-mode).

Inverse-ablation design: start from a PERMISSIVE baseline (scalp qualifier off,
no session/cooldown/HTF/volume/MACD gates, min_confidence floored), then add
each gate back one at a time. Reports signals/month and PF delta vs. baseline
so you can see which gates carry edge vs. which just shrink sample size.

Matches live runtime: scalp mode on, trigger TF 5m. This is the configuration
the live scanner actually runs, so the ablation is directly actionable.

JPY-pair awareness: SL/TP default to 10/15 pips for JPY crosses and 8/12 for
majors. Pass --sl/--tp to override. Pass --no-fixed-sltp to let the per-pair
scalp SL/TP from the DB apply (recommended if you want to reproduce live).

Usage:
    # Single pair
    docker exec task-worker python /app/forex_scanner/scripts/ablate_smc_simple.py --epics EURUSD
    docker exec task-worker python /app/forex_scanner/scripts/ablate_smc_simple.py --epics CS.D.GBPUSD.MINI.IP --days 90

    # Multi-pair sweep (the deploy-selection workflow)
    docker exec task-worker python /app/forex_scanner/scripts/ablate_smc_simple.py \\
        --epics EURUSD,GBPUSD,USDJPY,USDCAD,USDCHF,NZDUSD,AUDUSD,EURJPY,AUDJPY \\
        --days 90

    # Let per-pair DB scalp SL/TP apply (mirrors live config exactly)
    docker exec task-worker python /app/forex_scanner/scripts/ablate_smc_simple.py \\
        --epics EURUSD,GBPUSD --days 90 --no-fixed-sltp
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


BACKTEST_CMD = ["python", "/app/forex_scanner/backtest_cli.py"]
STRATEGY = "SMC_SIMPLE"

# Epic shortcut map (matches bt.py convention: EURUSD uses CEEM, others use MINI)
EPIC_SHORTCUTS = {
    "EURUSD": "CS.D.EURUSD.CEEM.IP",
    "GBPUSD": "CS.D.GBPUSD.MINI.IP",
    "USDJPY": "CS.D.USDJPY.MINI.IP",
    "AUDUSD": "CS.D.AUDUSD.MINI.IP",
    "USDCHF": "CS.D.USDCHF.MINI.IP",
    "USDCAD": "CS.D.USDCAD.MINI.IP",
    "NZDUSD": "CS.D.NZDUSD.MINI.IP",
    "EURJPY": "CS.D.EURJPY.MINI.IP",
    "AUDJPY": "CS.D.AUDJPY.MINI.IP",
    "GBPJPY": "CS.D.GBPJPY.MINI.IP",
    "EURGBP": "CS.D.EURGBP.MINI.IP",
}


def resolve_epic(token: str) -> str:
    token = token.strip()
    upper = token.upper()
    if upper in EPIC_SHORTCUTS:
        return EPIC_SHORTCUTS[upper]
    return token  # Assume already a full epic


def pair_label(epic: str) -> str:
    parts = epic.split(".")
    return parts[2] if len(parts) >= 3 else epic


def default_sl_tp(epic: str) -> Tuple[int, int]:
    """JPY crosses have 10x larger pip values — use wider defaults."""
    return (10, 15) if "JPY" in epic.upper() else (8, 12)


def build_baseline(sl: Optional[int], tp: Optional[int]) -> Dict[str, object]:
    """Permissive baseline for scalp-mode ablation.

    CRITICAL: scalp_qualification_mode=ACTIVE is set here (not MONITORING, which
    is the DB default). In MONITORING mode the qualifier logs but never blocks,
    so every gate toggle produces identical signal counts — the ablation is
    meaningless. ACTIVE mode is required for the gates to actually bind.

    SL/TP are NOT fixed here: scalp mode overwrites fixed_stop_loss_pips/
    fixed_take_profit_pips from the per-pair scalp_sl_pips/scalp_tp_pips in
    _configure_scalp_mode(), so any fixed SL/TP override silently gets clobbered.
    Letting per-pair DB scalp SL/TP apply mirrors live exactly.
    """
    base: Dict[str, object] = {
        # ---- Master scalp qualifier ----
        "scalp_qualification_enabled": "false",   # Short-circuit: all sigs pass
        "scalp_qualification_mode": "ACTIVE",     # When toggled on, BLOCK not LOG
        # ---- Structure / bias gates ----
        "macd_alignment_filter_enabled": "false",
        "htf_bias_enabled": "false",
        "pattern_confirmation_enabled": "false",
        "rsi_divergence_enabled": "false",
        # ---- Session / cadence ----
        "allow_asian_session": "true",
        "cooldown_minutes": 0,
        "adaptive_cooldown_enabled": "false",
    }
    # min_confidence and fixed_sl/tp intentionally omitted: scalp mode clobbers
    # them from scalp_min_confidence / scalp_sl_pips / scalp_tp_pips. Override
    # those scalp-specific keys instead if you need different values.
    if sl is not None and tp is not None:
        base["scalp_sl_pips"] = sl
        base["scalp_tp_pips"] = tp
    return base


# Gates are added one at a time on top of the permissive baseline.
ABLATIONS: List[Dict[str, object]] = [
    {"__label__": "PERMISSIVE baseline (all gates off)"},
    {"__label__": "+ scalp qualifier master on",
        "scalp_qualification_enabled": "true"},
    {"__label__": "+ scalp RSI filter",
        "scalp_qualification_enabled": "true", "scalp_rsi_filter_enabled": "true"},
    {"__label__": "+ scalp two-pole filter",
        "scalp_qualification_enabled": "true", "scalp_two_pole_filter_enabled": "true"},
    {"__label__": "+ scalp MACD filter",
        "scalp_qualification_enabled": "true", "scalp_macd_filter_enabled": "true"},
    {"__label__": "+ scalp anti-chop filter",
        "scalp_qualification_enabled": "true", "scalp_anti_chop_enabled": "true"},
    {"__label__": "+ scalp body-dominance filter",
        "scalp_qualification_enabled": "true", "scalp_body_dominance_enabled": "true"},
    {"__label__": "+ scalp consecutive-candles filter",
        "scalp_qualification_enabled": "true", "scalp_consecutive_candles_enabled": "true"},
    {"__label__": "+ MACD alignment (HTF)",
        "macd_alignment_filter_enabled": "true"},
    {"__label__": "+ HTF bias filter",
        "htf_bias_enabled": "true"},
    {"__label__": "+ scalp_min_confidence 0.55",
        "scalp_min_confidence": 0.55},
    {"__label__": "+ cooldown 15min",
        "cooldown_minutes": 15},
    {"__label__": "+ block Asian session",
        "allow_asian_session": "false"},
    {
        "__label__": "ALL qualifier filters on (shipped scalp preset)",
        "scalp_qualification_enabled": "true",
        "scalp_rsi_filter_enabled": "true",
        "scalp_two_pole_filter_enabled": "true",
        "scalp_macd_filter_enabled": "true",
        "scalp_anti_chop_enabled": "true",
        "scalp_body_dominance_enabled": "true",
        "scalp_consecutive_candles_enabled": "true",
        "macd_alignment_filter_enabled": "true",
        "htf_bias_enabled": "true",
        "scalp_min_confidence": 0.55,
        "cooldown_minutes": 15,
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


@dataclass
class EpicReport:
    epic: str
    pair: str
    days: int
    sl: Optional[int]
    tp: Optional[int]
    results: List[RunResult] = field(default_factory=list)
    baseline: Optional[RunResult] = None


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


def run_backtest(label: str, days: int, overrides: Dict[str, object], epic: str,
                 timeframe: str, use_scalp: bool) -> RunResult:
    cmd = BACKTEST_CMD + [
        "--epic", epic,
        "--days", str(days),
        "--strategy", STRATEGY,
        "--timeframe", timeframe,
    ]
    if use_scalp:
        cmd.append("--scalp")
    cmd += _override_args(overrides)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        return RunResult(label=label, days=days, overrides=overrides, ok=False,
                         error=out[-1500:])

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


def run_epic(epic: str, days: int, sl: Optional[int], tp: Optional[int],
             timeframe: str, use_scalp: bool) -> EpicReport:
    baseline_overrides = build_baseline(sl, tp)
    report = EpicReport(epic=epic, pair=pair_label(epic), days=days, sl=sl, tp=tp)

    sl_tp_str = f"{sl}/{tp}" if sl is not None else "DB-per-pair"
    scalp_str = "scalp" if use_scalp else "non-scalp"
    print(f"\n{'#' * 100}", flush=True)
    print(f"# [{report.pair}] epic={epic} days={days} tf={timeframe} mode={scalp_str} SL/TP={sl_tp_str}", flush=True)
    print(f"{'#' * 100}", flush=True)

    for i, ablation in enumerate(ABLATIONS, start=1):
        label = str(ablation.get("__label__", f"ablation_{i}"))
        overrides = {**baseline_overrides,
                     **{k: v for k, v in ablation.items() if not k.startswith("__")}}
        print(f"[{i:02d}/{len(ABLATIONS)}] {report.pair} :: {label}", flush=True)
        res = run_backtest(label, days, overrides, epic, timeframe, use_scalp)

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

        report.results.append(res)
        if label.startswith("PERMISSIVE"):
            report.baseline = res

    return report


def print_epic_report(report: EpicReport) -> None:
    print("\n" + "=" * 108, flush=True)
    sl_tp_str = f"{report.sl}/{report.tp}" if report.sl is not None else "DB-per-pair"
    print(f"ABLATION REPORT — {report.pair} SMC_SIMPLE, {report.days}d window (SL/TP={sl_tp_str})", flush=True)
    print("=" * 108, flush=True)
    print(f"{'gate relaxed':<48} {'n':>5} {'sig/mo':>8} {'pf':>6} {'wr%':>6} {'exp':>7} {'d(sig/mo)':>10} {'d(pf)':>7}", flush=True)
    print("-" * 108, flush=True)

    base_spm = signals_per_month(report.baseline) if report.baseline and report.baseline.ok else 0.0
    base_pf = report.baseline.profit_factor if report.baseline and report.baseline.ok else 0.0

    for res in report.results:
        if not res.ok:
            print(f"{res.label:<48} FAILED", flush=True)
            continue
        spm = signals_per_month(res)
        d_spm = spm - base_spm
        d_pf = res.profit_factor - base_pf
        print(
            f"{res.label:<48} {res.total_signals:>5d} {spm:>8.1f} "
            f"{res.profit_factor:>6.2f} {res.win_rate:>6.1f} {res.expectancy:>7.2f} "
            f"{d_spm:>+10.1f} {d_pf:>+7.2f}",
            flush=True,
        )


def print_cross_pair_summary(reports: List[EpicReport]) -> None:
    if len(reports) < 2:
        return
    print("\n" + "=" * 108, flush=True)
    print("CROSS-PAIR SUMMARY — baseline vs. shipped preset (n ≥ 50 is the statistical-validity floor)", flush=True)
    print("=" * 108, flush=True)
    print(f"{'pair':<10} {'baseline n':>10} {'base sig/mo':>12} {'base pf':>8}   "
          f"{'shipped n':>10} {'ship sig/mo':>12} {'ship pf':>8} {'ship wr%':>9}   {'verdict':<22}", flush=True)
    print("-" * 108, flush=True)

    for rep in reports:
        baseline = rep.baseline
        shipped = next((r for r in rep.results if r.label.startswith("ALL")), None)
        if not baseline or not shipped or not baseline.ok or not shipped.ok:
            print(f"{rep.pair:<10} (incomplete run)", flush=True)
            continue

        base_spm = signals_per_month(baseline)
        ship_spm = signals_per_month(shipped)

        if shipped.total_signals < 20:
            verdict = "DROP (undersample)"
        elif shipped.total_signals < 50:
            verdict = "MARGINAL (n<50)"
        elif shipped.profit_factor >= 1.5:
            verdict = "DEPLOY candidate"
        elif shipped.profit_factor >= 1.2:
            verdict = "MONITOR-ONLY"
        else:
            verdict = "DROP (weak PF)"

        print(
            f"{rep.pair:<10} "
            f"{baseline.total_signals:>10d} {base_spm:>12.1f} {baseline.profit_factor:>8.2f}   "
            f"{shipped.total_signals:>10d} {ship_spm:>12.1f} {shipped.profit_factor:>8.2f} {shipped.win_rate:>8.1f}%   "
            f"{verdict:<22}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epics", required=True,
                        help="Comma-separated list of pair shortcuts or full epics, "
                             "e.g. 'EURUSD,GBPUSD,USDJPY' or 'CS.D.EURUSD.CEEM.IP'")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--timeframe", default="5m",
                        help="Backtest scan interval (default 5m matches live scalp cadence)")
    parser.add_argument("--no-scalp", action="store_true",
                        help="Disable --scalp mode (NOT recommended; live runs scalp)")
    parser.add_argument("--sl", type=int, default=None,
                        help="Fixed SL pips (auto 8/10 majors/JPY if omitted)")
    parser.add_argument("--tp", type=int, default=None,
                        help="Fixed TP pips (auto 12/15 majors/JPY if omitted)")
    parser.add_argument("--no-fixed-sltp", action="store_true",
                        help="Skip fixed SL/TP — let per-pair scalp SL/TP from DB apply")
    args = parser.parse_args()

    epics = [resolve_epic(e) for e in args.epics.split(",") if e.strip()]
    if not epics:
        parser.error("--epics produced an empty list")

    use_scalp = not args.no_scalp
    print(f"[ablate_smc_simple] epics={len(epics)} days={args.days} tf={args.timeframe} "
          f"scalp={use_scalp} fixed_sltp={not args.no_fixed_sltp}", flush=True)

    reports: List[EpicReport] = []
    for epic in epics:
        if args.no_fixed_sltp:
            sl, tp = None, None
        else:
            default_sl, default_tp = default_sl_tp(epic)
            sl = args.sl if args.sl is not None else default_sl
            tp = args.tp if args.tp is not None else default_tp

        report = run_epic(epic, args.days, sl, tp, args.timeframe, use_scalp)
        reports.append(report)
        print_epic_report(report)

    print_cross_pair_summary(reports)

    print("\nInterpretation (cumulative add-gate design):", flush=True)
    print("  * Baseline = permissive, qualifier master off. Each row adds ONE gate on top.", flush=True)
    print("  * d(sig/mo) large negative + d(pf) positive  -> gate kills trades but lifts edge (KEEP)", flush=True)
    print("  * d(sig/mo) large negative + d(pf) ~0 or neg -> pure sample-size killer (DROP)", flush=True)
    print("  * d(sig/mo) small                            -> gate rarely binds, low cost either way", flush=True)
    print("  * Shipped preset n<50 on 90d  -> pair is over-filtered; relax gates before deploying", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
