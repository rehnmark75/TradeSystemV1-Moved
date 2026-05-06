#!/usr/bin/env python3
"""Curated sweep for the trend-expansion POC.

This imports `eval_trend_expansion.py`, loads 5m candles once per pair, then
tests a bounded parameter grid in-process. It is intentionally research-only:
no strategy registration, no config writes, no alert/order side effects.

Usage:
  docker exec -it task-worker python /app/forex_scanner/scripts/sweep_trend_expansion.py --days 90
  docker exec -it task-worker python /app/forex_scanner/scripts/sweep_trend_expansion.py --days 180 --preset fast
  docker exec -it task-worker python /app/forex_scanner/scripts/sweep_trend_expansion.py --pair CS.D.EURJPY.MINI.IP --days 180 --preset broad
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
import psycopg2

import eval_trend_expansion as te


@dataclass(frozen=True)
class SweepConfig:
    breakout_bars: int
    adx_min: float
    adx_rising_bars: int
    htf_ema: int
    htf_slope_bars: int
    htf_min_slope_pips: float
    body_atr_min: float
    max_extension_atr: float
    session_start: int
    session_end: int
    direction: str
    rr_profile: str

    def to_args(self, base: argparse.Namespace) -> argparse.Namespace:
        args = argparse.Namespace(**vars(base))
        args.breakout_bars = self.breakout_bars
        args.adx_min = self.adx_min
        args.adx_rising_bars = self.adx_rising_bars
        args.htf_ema = self.htf_ema
        args.htf_slope_bars = self.htf_slope_bars
        args.htf_min_slope_pips = self.htf_min_slope_pips
        args.body_atr_min = self.body_atr_min
        args.max_extension_atr = self.max_extension_atr
        args.session_start = self.session_start
        args.session_end = self.session_end
        args.direction = self.direction
        return args

    def label(self) -> str:
        return (
            f"bo={self.breakout_bars} adx={self.adx_min:g}/{self.adx_rising_bars} "
            f"ema={self.htf_ema} slope={self.htf_slope_bars}/{self.htf_min_slope_pips:g} "
            f"body={self.body_atr_min:g} "
            f"ext={self.max_extension_atr:g} sess={self.session_start}-{self.session_end} "
            f"dir={self.direction} rr={self.rr_profile}"
        )


RR_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {"major_sl": 10.0, "major_tp": 18.0, "jpy_sl": 14.0, "jpy_tp": 24.0},
    "wide": {"major_sl": 12.0, "major_tp": 24.0, "jpy_sl": 16.0, "jpy_tp": 32.0},
    "tight": {"major_sl": 8.0, "major_tp": 14.0, "jpy_sl": 12.0, "jpy_tp": 20.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--pair", default="", help="Full IG epic. Omit to run the standard 9 FX pairs.")
    parser.add_argument("--preset", choices=["fast", "broad", "usdjpy"], default="fast")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=20, help="Portfolio min trades for ranking bonus.")
    parser.add_argument("--show-pairs", action="store_true", help="Print pair table for top configs.")

    # These mirror the evaluator defaults and can still be overridden globally.
    parser.add_argument("--htf-ema", type=int, default=50)
    parser.add_argument("--htf-slope-bars", type=int, default=6)
    parser.add_argument("--cooldown-bars", type=int, default=8)
    parser.add_argument("--max-hold-bars", type=int, default=64)
    parser.add_argument("--sl-pips", type=float, default=None)
    parser.add_argument("--tp-pips", type=float, default=None)
    parser.add_argument("--cost-pips", type=float, default=None, help="Round-trip cost/slippage. Defaults to evaluator auto-cost.")
    parser.add_argument("--direction", choices=["BOTH", "BUY", "SELL"], default="BOTH")
    return parser.parse_args()


def config_grid(preset: str) -> Iterable[SweepConfig]:
    if preset == "fast":
        breakout_bars = [12, 20]
        adx_min = [18.0, 22.0]
        adx_rising_bars = [0, 3]
        htf_ema = [50]
        htf_slope_bars = [6]
        htf_min_slope_pips = [2.0, 4.0]
        body_atr_min = [0.45, 0.65]
        max_extension_atr = [2.2, 3.0]
        sessions = [(6, 20), (7, 18)]
        directions = ["BOTH"]
        rr_profiles = ["balanced", "wide"]
    elif preset == "broad":
        breakout_bars = [8, 12, 20, 32]
        adx_min = [16.0, 20.0, 24.0, 28.0]
        adx_rising_bars = [0, 3]
        htf_ema = [50]
        htf_slope_bars = [6]
        htf_min_slope_pips = [1.5, 3.0, 5.0]
        body_atr_min = [0.35, 0.55, 0.75]
        max_extension_atr = [1.6, 2.2, 3.0]
        sessions = [(6, 20), (7, 18), (12, 20)]
        directions = ["BOTH"]
        rr_profiles = ["balanced", "wide", "tight"]
    else:
        breakout_bars = [12, 20, 32]
        adx_min = [18.0, 22.0, 26.0]
        adx_rising_bars = [0, 3]
        htf_ema = [34, 50]
        htf_slope_bars = [4, 8]
        htf_min_slope_pips = [4.0, 10.0, 20.0]
        body_atr_min = [0.45, 0.65]
        max_extension_atr = [2.2, 3.0]
        sessions = [(0, 8), (7, 18), (12, 20), (0, 23)]
        directions = ["BOTH", "BUY", "SELL"]
        rr_profiles = ["balanced", "wide"]

    for values in itertools.product(
        breakout_bars,
        adx_min,
        adx_rising_bars,
        htf_ema,
        htf_slope_bars,
        htf_min_slope_pips,
        body_atr_min,
        max_extension_atr,
        sessions,
        directions,
        rr_profiles,
    ):
        bo, adx, rising, ema, slope_bars, slope, body, ext, session, direction, rr = values
        yield SweepConfig(
            breakout_bars=bo,
            adx_min=adx,
            adx_rising_bars=rising,
            htf_ema=ema,
            htf_slope_bars=slope_bars,
            htf_min_slope_pips=slope,
            body_atr_min=body,
            max_extension_atr=ext,
            session_start=session[0],
            session_end=session[1],
            direction=direction,
            rr_profile=rr,
        )


def trade_config_for(epic: str, profile: str, sl_override: Optional[float], tp_override: Optional[float]) -> te.TradeConfig:
    if sl_override is not None or tp_override is not None:
        return te.default_trade_config(epic, sl_override, tp_override)
    profile_cfg = RR_PROFILES[profile]
    if "JPY" in epic.upper():
        return te.TradeConfig(profile_cfg["jpy_sl"], profile_cfg["jpy_tp"])
    return te.TradeConfig(profile_cfg["major_sl"], profile_cfg["major_tp"])


def score_result(stats: Dict[str, float], pair_stats: pd.DataFrame, min_trades: int) -> float:
    n = stats["n"]
    pf = stats["pf"]
    if n <= 0 or math.isnan(pf):
        return -999.0
    if n < min_trades:
        return -999.0
    finite_pf = min(pf, 5.0) if math.isfinite(pf) else 5.0
    breadth = int((pair_stats["n"] >= 3).sum()) if not pair_stats.empty else 0
    profitable_pairs = int((pair_stats["total_pips"] > 0).sum()) if not pair_stats.empty else 0
    return (
        finite_pf * 2.0
        + stats["expectancy"] * 0.20
        + breadth * 0.25
        + profitable_pairs * 0.10
    )


def run_config(
    cfg: SweepConfig,
    base_args: argparse.Namespace,
    candles: Dict[str, pd.DataFrame],
    days: int,
) -> Dict:
    args = cfg.to_args(base_args)
    all_trades: List[pd.DataFrame] = []
    pair_rows: List[Dict] = []

    for epic, df5 in candles.items():
        signals = te.build_signal_frame(df5, epic, args)
        cutoff = signals.index.max() - pd.Timedelta(days=days)
        signals = signals[signals.index >= cutoff]
        trade_cfg = trade_config_for(epic, cfg.rr_profile, base_args.sl_pips, base_args.tp_pips)
        trades = te.simulate_trades(signals, df5, epic, trade_cfg, args)
        stats = te.aggregate(trades)
        stats["pair"] = te.pair_label(epic)
        stats["raw_signals"] = int((signals["signal"] != "").sum())
        pair_rows.append(stats)
        if not trades.empty:
            trades["epic"] = epic
            all_trades.append(trades)

    pair_stats = pd.DataFrame(pair_rows)
    if all_trades:
        portfolio = pd.concat(all_trades, ignore_index=True)
        stats = te.aggregate(portfolio)
    else:
        portfolio = pd.DataFrame()
        stats = te.aggregate(portfolio)

    return {
        "config": cfg,
        "stats": stats,
        "pair_stats": pair_stats,
        "portfolio": portfolio,
        "score": score_result(stats, pair_stats, base_args.min_trades),
    }


def fmt_pf(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.2f}"


def main() -> int:
    args = parse_args()
    pairs = [args.pair] if args.pair else te.PAIRS
    configs = list(config_grid(args.preset))
    print(
        f"[sweep_trend_expansion] days={args.days} preset={args.preset} "
        f"configs={len(configs)} pairs={len(pairs)}",
        flush=True,
    )

    candles: Dict[str, pd.DataFrame] = {}
    with psycopg2.connect(te.FOREX_DB) as conn:
        end = te.latest_backtest_time(conn)
        for epic in pairs:
            df5 = te.load_5m(conn, epic, args.days, end)
            if df5.empty:
                print(f"[sweep_trend_expansion] no data: {epic}", flush=True)
                continue
            candles[epic] = df5

    if not candles:
        print("[sweep_trend_expansion] no candle data loaded", flush=True)
        return 1

    base_eval_args = argparse.Namespace(
        days=args.days,
        pair=args.pair,
        htf_ema=args.htf_ema,
        htf_slope_bars=args.htf_slope_bars,
        cooldown_bars=args.cooldown_bars,
        max_hold_bars=args.max_hold_bars,
        sl_pips=args.sl_pips,
        tp_pips=args.tp_pips,
        cost_pips=args.cost_pips,
        direction=args.direction,
        show_trades=False,
        session_start=6,
        session_end=20,
        breakout_bars=20,
        adx_min=22.0,
        adx_rising_bars=3,
        htf_min_slope_pips=4.0,
        body_atr_min=0.65,
        max_extension_atr=2.2,
    )
    base_eval_args.min_trades = args.min_trades

    results: List[Dict] = []
    for idx, cfg in enumerate(configs, start=1):
        result = run_config(cfg, base_eval_args, candles, args.days)
        results.append(result)
        stats = result["stats"]
        if idx % 25 == 0 or idx == len(configs):
            print(
                f"[{idx:04d}/{len(configs)}] "
                f"best_score={max(r['score'] for r in results):.2f} "
                f"last n={stats['n']} pf={fmt_pf(stats['pf'])} exp={stats['expectancy']:.1f}",
                flush=True,
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    print("\n=== TOP CONFIGS ===", flush=True)
    for rank, result in enumerate(results[: args.top], start=1):
        stats = result["stats"]
        pair_stats = result["pair_stats"]
        breadth = int((pair_stats["n"] >= 3).sum()) if not pair_stats.empty else 0
        profitable = int((pair_stats["total_pips"] > 0).sum()) if not pair_stats.empty else 0
        print(
            f"{rank:02d}. score={result['score']:.2f} "
            f"n={stats['n']:3d} wr={stats['wr']:.1%} pf={fmt_pf(stats['pf']):>5} "
            f"exp={stats['expectancy']:6.2f} total={stats['total_pips']:7.1f} "
            f"breadth={breadth} profitable_pairs={profitable} | {result['config'].label()}",
            flush=True,
        )
        if args.show_pairs and not pair_stats.empty:
            cols = ["pair", "raw_signals", "n", "wr", "pf", "expectancy", "total_pips", "tp", "sl", "timeout"]
            print(pair_stats[cols].to_string(index=False, float_format="{:.2f}".format), flush=True)
            print("", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
