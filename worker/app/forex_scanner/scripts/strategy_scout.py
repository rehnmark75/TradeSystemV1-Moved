#!/usr/bin/env python3
"""Strategy Scout — data-driven strategy-class selector.

Given an epic and a lookback window, scans historical candle data and:
  1. Characterizes the market (regime mix, volatility, mean-rev signature)
  2. Runs a fixed library of 8 canonical strategy templates × small param
     grids in parallel using ProcessPoolExecutor
  3. Computes PF with block-bootstrap CIs and regime-conditional breakdowns
  4. Emits markdown + CSV reports for human review / specialist-agent handoff

See plan file: /home/hr/.claude/plans/think-if-its-possible-sparkling-trinket.md
for full design rationale.

Usage:
  docker exec -it task-worker python /app/forex_scanner/scripts/strategy_scout.py \\
      --epic CS.D.EURJPY.MINI.IP --days 90 --output-dir /tmp/scout/eurjpy_90d/

  # Faster run (fewer combos):
  docker exec -it task-worker python /app/forex_scanner/scripts/strategy_scout.py \\
      --epic CS.D.EURJPY.MINI.IP --days 30 --workers 4 --output-dir /tmp/scout/smoke/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List

# Make /app importable
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

import numpy as np
import pandas as pd
import psycopg2

from forex_scanner.scripts.scout.simulate import simulate, aggregate, bootstrap_pf_ci
from forex_scanner.scripts.scout.templates import TEMPLATES
from forex_scanner.scripts.scout.characterize import characterize
from forex_scanner.scripts.scout.report import (
    write_characterization,
    write_strategy_scores,
    write_recommendations,
)
from forex_scanner.scripts.recompute_adx_regime import ema_wilder_adx, regime_from_adx

FOREX_DB = "postgresql://postgres:postgres@postgres:5432/forex"

# Uniform SL/TP across templates (plan decision: compare signal edge, not risk sizing)
DEFAULT_SL_PIPS = 8.0
DEFAULT_TP_PIPS = 16.0
DEFAULT_COOLDOWN_BARS = 6   # 30 min on 5m
DEFAULT_MAX_HOLD_BARS = 288  # 1 day on 5m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epic", required=True, help="Epic to analyze (e.g., CS.D.EURJPY.MINI.IP)")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--output-dir", default="/tmp/scout/",
                   help="Directory for artifacts (created if missing)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sl-pips", type=float, default=DEFAULT_SL_PIPS)
    p.add_argument("--tp-pips", type=float, default=DEFAULT_TP_PIPS)
    p.add_argument("--cooldown-bars", type=int, default=DEFAULT_COOLDOWN_BARS)
    p.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    p.add_argument("--only-templates", nargs="*", default=None,
                   help="Restrict to these template names (default: all)")
    p.add_argument("--bootstrap-n", type=int, default=1000,
                   help="Bootstrap resamples for PF CI")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def pip_size(epic: str) -> float:
    if "GOLD" in epic.upper() or "XAU" in epic.upper():
        return 0.1
    return 0.01 if "JPY" in epic.upper() else 0.0001


def load_candles(conn, epic: str, days: int, warmup_days: int = 7):
    """Return (df_5m, df_15m, df_1h) indexed by start_time (UTC)."""
    end = datetime.now()
    start = end - timedelta(days=days + warmup_days)
    q = """
        SELECT start_time, open, high, low, close
          FROM ig_candles_backtest
         WHERE epic = %s AND timeframe = 5
           AND start_time >= %s AND start_time <= %s
         ORDER BY start_time
    """
    df5 = pd.read_sql(q, conn, params=(epic, start, end))
    if df5.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df5["start_time"] = pd.to_datetime(df5["start_time"])
    df5 = df5.set_index("start_time")
    df15 = df5.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    df1h = df5.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    return df5, df15, df1h


# ============================================================================
# Single run: template × params  (executed in a worker process)
# ============================================================================

def _run_one(args: Dict) -> Dict:
    """Run one (template, params) combo on a df. Called from worker process.

    Worker gets a fresh Python process — template functions are re-imported
    and df is passed as pickled pandas DataFrame."""
    template_name = args["template_name"]
    params = args["params"]
    df = args["df"]
    regime_series = args["regime_series"]
    sl_pips = args["sl_pips"]
    tp_pips = args["tp_pips"]
    pip_value = args["pip_value"]
    cooldown_bars = args["cooldown_bars"]
    max_hold_bars = args["max_hold_bars"]
    bootstrap_n = args["bootstrap_n"]

    # Re-import inside worker
    from forex_scanner.scripts.scout.templates import TEMPLATE_BY_NAME
    from forex_scanner.scripts.scout.simulate import simulate, aggregate, bootstrap_pf_ci

    template = TEMPLATE_BY_NAME[template_name]
    signals = template.generate(df, params)
    trades = simulate(df, signals, sl_pips, tp_pips, pip_value, cooldown_bars, max_hold_bars)

    # Stamp regime at entry bar
    for t in trades:
        entry_idx = t["entry_idx"]
        if 0 <= entry_idx < len(regime_series):
            t["regime"] = regime_series.iloc[entry_idx]
        else:
            t["regime"] = None

    agg = aggregate(trades)

    # Regime-conditional aggregation
    by_regime: Dict[str, Dict] = {}
    df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
    for regime in ("trending", "ranging", "low_volatility", "breakout"):
        if df_trades.empty:
            by_regime[regime] = {"n_trades": 0, "pf": 0.0, "win_rate": 0.0}
            continue
        sub = df_trades[df_trades["regime"] == regime]
        sub_dicts = sub.to_dict("records")
        sub_agg = aggregate(sub_dicts)
        by_regime[regime] = {
            "n_trades": sub_agg["n_trades"],
            "pf": sub_agg["pf"],
            "win_rate": sub_agg["win_rate"],
        }

    # Bootstrap CI on overall PF
    ci = bootstrap_pf_ci(trades, n_bootstrap=bootstrap_n, block_size=5)

    return {
        "template": template_name,
        "params": params,
        **agg,
        "by_regime": by_regime,
        "pf_ci_lower": ci["lower"],
        "pf_ci_upper": ci["upper"],
        "pf_ci_median": ci["median"],
    }


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[scout] epic={args.epic} days={args.days} output={args.output_dir}")
    print(f"[scout] SL/TP = {args.sl_pips}/{args.tp_pips} pips (uniform)")

    # -------- Phase A: load candles --------
    conn = psycopg2.connect(FOREX_DB)
    df5, df15, df1h = load_candles(conn, args.epic, args.days)
    conn.close()
    if df5.empty:
        print(f"[scout] No candle data for {args.epic}. Is ig_candles_backtest populated?")
        return 1

    cutoff = df5.index.max() - pd.Timedelta(days=args.days)
    df5 = df5[df5.index >= cutoff]
    df15 = df15[df15.index >= cutoff]
    df1h = df1h[df1h.index >= cutoff]

    pip_value = pip_size(args.epic)
    print(f"[scout] loaded 5m={len(df5):,} 15m={len(df15):,} 1h={len(df1h):,} "
          f"({df5.index[0]} → {df5.index[-1]})  pip={pip_value}")

    # -------- Phase B: characterize --------
    print(f"[scout] characterizing market...")
    t_char = time.time()
    characterization = characterize(args.epic, args.days, pip_value, df5, df15, df1h)
    print(f"[scout] characterization done in {time.time()-t_char:.1f}s")

    # Regime series on 5m (needed for per-trade regime stamping).
    # Use 15m ADX forward-filled onto 5m index so each 5m bar inherits its
    # 15m parent's regime label.
    adx_15m = ema_wilder_adx(df15, period=14)
    regime_15m = adx_15m.apply(regime_from_adx)
    regime_5m = regime_15m.reindex(df5.index, method="ffill")

    # -------- Phase C: parallel template grid --------
    templates = [t for t in TEMPLATES
                 if args.only_templates is None or t.name in args.only_templates]
    combos = [(t.name, p) for t in templates for p in t.grid]
    print(f"[scout] running {len(combos)} (template, params) combos on {args.workers} workers...")

    worker_args = [
        {
            "template_name": name,
            "params": params,
            "df": df5,
            "regime_series": regime_5m,
            "sl_pips": args.sl_pips,
            "tp_pips": args.tp_pips,
            "pip_value": pip_value,
            "cooldown_bars": args.cooldown_bars,
            "max_hold_bars": args.max_hold_bars,
            "bootstrap_n": args.bootstrap_n,
        }
        for name, params in combos
    ]

    rows: List[Dict] = []
    t_sim = time.time()
    if args.workers <= 1:
        for wa in worker_args:
            rows.append(_run_one(wa))
            if args.verbose:
                print(f"  done: {rows[-1]['template']:22s} {str(rows[-1]['params'])[:50]:50s} "
                      f"n={rows[-1]['n_trades']:3d} PF={rows[-1]['pf']:.2f}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one, wa): wa for wa in worker_args}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    rows.append(r)
                    if args.verbose:
                        print(f"  done: {r['template']:22s} {str(r['params'])[:50]:50s} "
                              f"n={r['n_trades']:3d} PF={r['pf']:.2f}")
                except Exception as e:
                    wa = futures[fut]
                    print(f"  FAIL: {wa['template_name']} {wa['params']}: {e}")

    print(f"[scout] template grid done in {time.time()-t_sim:.1f}s")

    # -------- Phase E: reports --------
    print(f"[scout] writing reports...")
    write_characterization(characterization, os.path.join(args.output_dir, "characterization.md"))
    write_strategy_scores(rows, os.path.join(args.output_dir, "strategy_scores.csv"))
    write_recommendations(characterization, rows, os.path.join(args.output_dir, "recommendations.md"))

    # Console summary: top 5 by PF
    if rows:
        df_rows = pd.DataFrame(rows).sort_values("pf", ascending=False).head(5)
        print()
        print("TOP 5 by PF:")
        for _, r in df_rows.iterrows():
            pf_lower = r.get("pf_ci_lower", 0.0)
            print(f"  {r['template']:22s} {str(r['params'])[:40]:40s} "
                  f"n={int(r['n_trades']):4d} WR={r['win_rate']*100:5.1f}% "
                  f"PF={r['pf']:6.2f}  CI[{pf_lower:.2f}, {r.get('pf_ci_upper',0.0):.2f}]")

    print()
    print(f"[scout] total elapsed: {time.time()-t0:.1f}s")
    print(f"[scout] artifacts in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
