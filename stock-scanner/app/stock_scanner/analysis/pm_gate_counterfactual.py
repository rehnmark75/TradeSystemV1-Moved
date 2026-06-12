"""
PM Gate Counterfactual Analysis
================================
Evaluates whether the gate-softening change (allowing 'Stale PM' / 'No PM data'
candidates through if the live rescue confirms them) would have made money under
the current SL 2% / TP 7% day-trade bracket (no break-even move).

The script is designed to be re-run as forward data accrues — it reads live DB
rows and simulates outcomes from 1h candle data.

Usage (inside stock-scanner container):
    python -m stock_scanner.analysis.pm_gate_counterfactual

Simulation model:
  - SL 2%, TP 7%, NO break-even (matching current live config as of Jun 12 2026)
  - Entry: OPEN of first 1h bar STRICTLY AFTER the trade_date session open
            (the candidate snapshot is captured intraday; entry proxied as the
             next 13:30 UTC 1h bar on the same trade_date)
  - Path-walking engine and candle accessors reused from daytrade_edge_sim.py
  - Timeout: 10 trading days (same convention as daytrade_edge_sim.py)

Gate logic replicated from the new softened gate:
  - Rescue: intraday_relative_volume >= 2.0
  - Price at/above VWAP: broker_ask >= session_vwap  (or broker_last >= session_vwap)
  - Spread: broker_spread_pct <= run max_spread_pct
  - Score: candidate_score >= 65
  - Quote age: broker_quote_age_minutes <= 3

Null-snapshot policy:
  - If intraday_relative_volume IS NULL → rescue fails → candidate does NOT unlock
  - If session_vwap IS NULL → VWAP gate treated as UNKNOWN → candidate does NOT unlock
  (This is the realistic behaviour: the rescue needs live intraday data;
   if it was not captured, the unlock cannot be confirmed historically.)

Arms:
  - UNLOCKED: candidates that pass all five rescue gates above
  - ALL PM REJECTS: all Stale PM / No PM data rows (larger sample, looser bound)
  - BENCHMARK: candidates that passed the old gates and were ordered (ordered/order_submitted)
"""

from __future__ import annotations

import sys
import warnings
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# Reuse validated simulation machinery from daytrade_edge_sim
from stock_scanner.analysis.daytrade_edge_sim import (
    build_candle_index,
    build_daily_index,
    bars_after,
    atr_pct_before,
    walk_bars,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DB_URL = "postgresql://postgres:postgres@postgres:5432/stocks"

# Simulation bracket — current live config as of Jun 12 2026
SL_PCT = 2.0
TP_PCT = 7.0
BE_ENABLED = False          # No break-even arm per current config
TIMEOUT_TRADING_DAYS = 10  # ~10d, same as daytrade_edge_sim

# Rescue gate thresholds (matching new softened gate logic)
RESCUE_RVOL_MIN = 2.0
SCORE_MIN = 65.0
QUOTE_AGE_MAX_MINUTES = 3

# Run max_spread_pct comes from the run config JSONB (per-run);
# used per-row when available, else fallback to 0.50 (most recent runs)
DEFAULT_MAX_SPREAD_PCT = 0.50


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def fetch_df(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Pull PM-rejected candidates
# ---------------------------------------------------------------------------

PM_REJECT_SQL = """
SELECT
    c.id,
    c.run_id,
    c.trade_date,
    c.rank,
    c.ticker,
    c.status,
    c.reason,
    c.candidate_score,
    c.scanner_name,
    c.pm_status,
    c.broker_spread_pct,
    c.broker_quote_age_minutes,
    c.intraday_relative_volume,
    c.session_vwap,
    c.broker_ask,
    c.broker_last,
    c.broker_bid,
    c.relative_volume,
    (r.config->>'max_spread_pct')::numeric  AS run_max_spread_pct,
    (r.config->>'stop_loss_pct')::numeric   AS run_sl_pct,
    (r.config->>'take_profit_pct')::numeric AS run_tp_pct
FROM stock_auto_trade_candidates c
JOIN stock_auto_trade_runs r ON r.id = c.run_id
WHERE
    c.reason ILIKE '%Stale PM%'
    OR c.reason ILIKE '%No PM data%'
ORDER BY c.trade_date, c.id
"""

BENCHMARK_SQL = """
SELECT
    c.id,
    c.run_id,
    c.trade_date,
    c.rank,
    c.ticker,
    c.status,
    c.reason,
    c.candidate_score,
    c.scanner_name,
    c.broker_spread_pct,
    c.broker_quote_age_minutes,
    c.intraday_relative_volume,
    c.session_vwap,
    c.broker_ask,
    c.broker_last,
    c.broker_bid,
    (r.config->>'max_spread_pct')::numeric  AS run_max_spread_pct,
    (r.config->>'stop_loss_pct')::numeric   AS run_sl_pct,
    (r.config->>'take_profit_pct')::numeric AS run_tp_pct
FROM stock_auto_trade_candidates c
JOIN stock_auto_trade_runs r ON r.id = c.run_id
WHERE
    (c.status = 'order_submitted' OR c.status = 'ordered'
     OR c.reason ILIKE '%Order submitted%')
ORDER BY c.trade_date, c.id
"""


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_rescue_gates(row: dict) -> dict:
    """
    Returns a dict with per-gate pass/fail and a summary 'unlocked' boolean.

    Rules (all must pass for unlock):
      1. RVOL >= 2.0  (rescue confirmation)   — FAIL if NULL
      2. price >= session_vwap                 — FAIL if session_vwap NULL
      3. spread <= run_max_spread_pct
      4. score >= 65
      5. quote_age <= 3 min
    """
    rvol = row.get("intraday_relative_volume")
    vwap = row.get("session_vwap")
    ask = row.get("broker_ask") or row.get("broker_last")
    spread = row.get("broker_spread_pct")
    score = row.get("candidate_score")
    age = row.get("broker_quote_age_minutes")
    max_spread = float(row.get("run_max_spread_pct") or DEFAULT_MAX_SPREAD_PCT)

    # Gate 1: rescue RVOL
    if rvol is None:
        g1 = False
        g1_note = "NULL rvol"
    else:
        g1 = float(rvol) >= RESCUE_RVOL_MIN
        g1_note = f"rvol={float(rvol):.2f}"

    # Gate 2: price >= vwap
    if vwap is None or ask is None:
        g2 = False
        g2_note = "NULL vwap or price"
    else:
        g2 = float(ask) >= float(vwap)
        g2_note = f"ask={float(ask):.2f} vwap={float(vwap):.2f}"

    # Gate 3: spread
    if spread is None:
        g3 = False
        g3_note = "NULL spread"
    else:
        g3 = float(spread) <= max_spread
        g3_note = f"spread={float(spread):.3f} max={max_spread:.3f}"

    # Gate 4: score
    if score is None:
        g4 = False
        g4_note = "NULL score"
    else:
        g4 = float(score) >= SCORE_MIN
        g4_note = f"score={float(score):.1f}"

    # Gate 5: quote age
    if age is None:
        # quote_age = 0 was stored as NULL in early runs; treat 0/NULL as fresh
        g5 = True
        g5_note = "NULL age (treated as fresh)"
    else:
        g5 = int(age) <= QUOTE_AGE_MAX_MINUTES
        g5_note = f"age={int(age)}min"

    unlocked = g1 and g2 and g3 and g4 and g5

    return {
        "gate_rvol": g1,
        "gate_vwap": g2,
        "gate_spread": g3,
        "gate_score": g4,
        "gate_age": g5,
        "gate_rvol_note": g1_note,
        "gate_vwap_note": g2_note,
        "gate_spread_note": g3_note,
        "gate_score_note": g4_note,
        "gate_age_note": g5_note,
        "unlocked": unlocked,
    }


# ---------------------------------------------------------------------------
# Simulation engine for a single candidate
# ---------------------------------------------------------------------------

def simulate_candidate(
    row: dict,
    candles_by_ticker: dict,
    sl_pct: float = SL_PCT,
    tp_pct: float = TP_PCT,
) -> dict:
    """
    Simulate a fixed-bracket day-trade for a single candidate row.

    Entry convention: OPEN of first 1h bar STRICTLY AFTER snapshot time
    (proxied as the trade_date + 13:30 UTC — the standard US session open used
    by the validated simulator).  We pass trade_date + 12:30 UTC as the ref_time
    so the first bar strictly after it is the 13:30 UTC session open bar.

    Returns a result dict with outcome, pnl_pct, hold_bars, entry.
    """
    ticker = str(row.get("ticker", "")).split(".")[0]
    trade_date = row.get("trade_date")

    if trade_date is None:
        return {"outcome": "no_data", "pnl_pct": 0.0, "hold_bars": 0, "entry": 0.0}

    # Proxy entry time: 12:30 UTC so the strict > gives us 13:30 UTC open bar
    if isinstance(trade_date, (date, datetime)):
        ref_dt = pd.Timestamp(str(trade_date) + " 12:30:00")
    else:
        ref_dt = pd.Timestamp(str(trade_date) + " 12:30:00")

    bars, entry = bars_after(candles_by_ticker, ticker, ref_dt)

    if bars.empty or entry <= 0:
        return {"outcome": "no_data", "pnl_pct": 0.0, "hold_bars": 0, "entry": entry}

    sl = entry * (1 - sl_pct / 100)
    tp = entry * (1 + tp_pct / 100)
    bracket = {
        "sl": sl,
        "tp": tp,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "mode": f"fixed-{sl_pct}/{tp_pct}",
        "notional": 500.0,
        "be_trigger_pct": 9999.0,  # effectively disabled
    }

    result = walk_bars(entry, bracket, bars, be_enabled=BE_ENABLED)
    result["entry"] = entry
    return result


# ---------------------------------------------------------------------------
# Metrics aggregator
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    """
    Compute PF, WR, expectancy from a simulation result DataFrame.
    Columns expected: outcome (win/loss/timeout/no_data), pnl_pct.
    """
    valid = df[df["outcome"].isin(["win", "loss", "timeout"])].copy()
    n_total = len(df)
    n_valid = len(valid)
    n_no_data = (df["outcome"] == "no_data").sum()

    wins = valid[valid["outcome"] == "win"]
    losses = valid[valid["outcome"] == "loss"]
    timeouts = valid[valid["outcome"] == "timeout"]

    n_wins = len(wins)
    n_losses = len(losses)
    n_timeouts = len(timeouts)

    # WR: win / (win + loss), excluding timeouts (like the existing sim)
    n_resolved = n_wins + n_losses
    wr = n_wins / n_resolved if n_resolved > 0 else None

    gross_win = wins["pnl_pct"].sum() if n_wins > 0 else 0.0
    gross_loss = abs(losses["pnl_pct"].sum()) if n_losses > 0 else 0.0

    if gross_loss > 0:
        pf = round(gross_win / gross_loss, 3)
    elif gross_win > 0:
        pf = 9.99
    else:
        pf = None

    avg_win = wins["pnl_pct"].mean() if n_wins > 0 else None
    avg_loss = losses["pnl_pct"].mean() if n_losses > 0 else None

    # Expectancy per resolved trade (pips-pct equivalent)
    if n_resolved > 0:
        expectancy = valid[valid["outcome"].isin(["win", "loss"])]["pnl_pct"].mean()
    else:
        expectancy = None

    # Expectancy including timeouts (pessimistic — mark-to-market at timeout close)
    if n_valid > 0:
        expectancy_all = valid["pnl_pct"].mean()
    else:
        expectancy_all = None

    return {
        "label": label,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_no_data": int(n_no_data),
        "n_wins": int(n_wins),
        "n_losses": int(n_losses),
        "n_timeouts": int(n_timeouts),
        "n_resolved": int(n_resolved),
        "wr_pct": round(wr * 100, 1) if wr is not None else None,
        "pf": pf,
        "avg_win_pct": round(float(avg_win), 2) if avg_win is not None else None,
        "avg_loss_pct": round(float(avg_loss), 2) if avg_loss is not None else None,
        "expectancy_resolved_pct": round(float(expectancy), 3) if expectancy is not None else None,
        "expectancy_all_pct": round(float(expectancy_all), 3) if expectancy_all is not None else None,
        "timeout_frac_pct": round(n_timeouts / n_valid * 100, 1) if n_valid > 0 else None,
    }


def print_metrics(m: dict):
    wr = f"{m['wr_pct']:.1f}%" if m["wr_pct"] is not None else "N/A"
    pf = f"{m['pf']:.3f}" if m["pf"] is not None else "N/A"
    aw = f"{m['avg_win_pct']:.2f}%" if m["avg_win_pct"] is not None else "N/A"
    al = f"{m['avg_loss_pct']:.2f}%" if m["avg_loss_pct"] is not None else "N/A"
    er = f"{m['expectancy_resolved_pct']:.3f}%" if m["expectancy_resolved_pct"] is not None else "N/A"
    ea = f"{m['expectancy_all_pct']:.3f}%" if m["expectancy_all_pct"] is not None else "N/A"
    to = f"{m['timeout_frac_pct']:.1f}%" if m["timeout_frac_pct"] is not None else "N/A"

    print(f"  n_total={m['n_total']}  n_valid={m['n_valid']}  "
          f"wins={m['n_wins']}  losses={m['n_losses']}  timeouts={m['n_timeouts']}  "
          f"no_data={m['n_no_data']}")
    print(f"  WR={wr}  PF={pf}  avg_win={aw}  avg_loss={al}")
    print(f"  Expectancy (wins+losses only): {er}  Expectancy (incl timeouts): {ea}")
    print(f"  Timeout fraction: {to}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(conn):
    print("=" * 72)
    print("PM GATE COUNTERFACTUAL ANALYSIS")
    print(f"Run timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Bracket: SL={SL_PCT}%  TP={TP_PCT}%  BE=OFF  Horizon={TIMEOUT_TRADING_DAYS}d")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 0. Load candle data
    # ------------------------------------------------------------------
    print("\nPre-loading 1h candles and 1d daily data...")
    candles_by_ticker = build_candle_index(conn)
    daily_by_ticker = build_daily_index(conn)
    print(f"  1h bars: {sum(len(v) for v in candles_by_ticker.values()):,} "
          f"across {len(candles_by_ticker)} tickers")
    print(f"  1d bars: {sum(len(v) for v in daily_by_ticker.values()):,} "
          f"across {len(daily_by_ticker)} tickers")

    # ------------------------------------------------------------------
    # 1. Pull PM-reject population
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 1: PM-REJECT POPULATION")
    print("=" * 72)

    pm_df = fetch_df(conn, PM_REJECT_SQL)
    print(f"\nTotal PM-reject rows: {len(pm_df)}")
    if pm_df.empty:
        print("  No PM-reject rows found. Exiting.")
        return

    # Breakdown by reason type
    reason_counts = pm_df["reason"].value_counts()
    print("\nBreakdown by reject reason:")
    for reason, cnt in reason_counts.items():
        print(f"  {reason!r}: {cnt}")

    # Breakdown by date
    date_counts = pm_df["trade_date"].value_counts().sort_index()
    print("\nBreakdown by trade_date:")
    for dt, cnt in date_counts.items():
        print(f"  {dt}: {cnt}")

    # Null-snapshot diagnostic (critical for rescue gate applicability)
    n_null_rvol = pm_df["intraday_relative_volume"].isna().sum()
    n_null_vwap = pm_df["session_vwap"].isna().sum()
    print(f"\nSnapshot null counts (determines rescue gate applicability):")
    print(f"  intraday_relative_volume NULL: {n_null_rvol} / {len(pm_df)} "
          f"({100*n_null_rvol/len(pm_df):.0f}%)")
    print(f"  session_vwap NULL: {n_null_vwap} / {len(pm_df)} "
          f"({100*n_null_vwap/len(pm_df):.0f}%)")
    if n_null_rvol > 0 or n_null_vwap > 0:
        print("  NOTE: Rows with NULL intraday_rvol or session_vwap cannot satisfy")
        print("  the rescue gate — they do NOT unlock and are excluded from the")
        print("  UNLOCKED arm (they remain in the ALL-PM-REJECTS arm for sample size).")
        print("  The unlock rate for those rows is ~0 going forward unless the snapshot")
        print("  capture is fixed to always record these fields.")

    # ------------------------------------------------------------------
    # 2. Rescue gate evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 2: RESCUE GATE EVALUATION")
    print("=" * 72)

    gate_results = []
    for _, row in pm_df.iterrows():
        g = evaluate_rescue_gates(dict(row))
        g["id"] = row["id"]
        g["trade_date"] = row["trade_date"]
        g["ticker"] = row["ticker"]
        g["reason"] = row["reason"]
        g["pm_status"] = row["pm_status"]
        g["candidate_score"] = row["candidate_score"]
        g["scanner_name"] = row["scanner_name"]
        gate_results.append(g)

    gate_df = pd.DataFrame(gate_results)
    unlocked_df = gate_df[gate_df["unlocked"]].copy()
    locked_df = gate_df[~gate_df["unlocked"]].copy()

    print(f"\nTotal PM rejects evaluated: {len(gate_df)}")
    print(f"  UNLOCKED (pass all rescue gates): {len(unlocked_df)}")
    print(f"  Remain blocked:                   {len(locked_df)}")

    if len(gate_df) > 0:
        unlock_rate = len(unlocked_df) / len(gate_df) * 100
        print(f"  Unlock rate: {unlock_rate:.1f}%")

    # Per-gate failure breakdown
    print("\nPer-gate failure counts (of the locked rows):")
    for g_col, label in [
        ("gate_rvol", f"RVOL < {RESCUE_RVOL_MIN} (or NULL)"),
        ("gate_vwap", "Price < VWAP (or NULL vwap)"),
        ("gate_spread", "Spread > max_spread_pct"),
        ("gate_score", f"Score < {SCORE_MIN}"),
        ("gate_age", f"Quote age > {QUOTE_AGE_MAX_MINUTES} min"),
    ]:
        n_fail = (~locked_df[g_col]).sum() if len(locked_df) > 0 else 0
        pct = 100 * n_fail / len(locked_df) if len(locked_df) > 0 else 0
        print(f"  {label:45s}: {n_fail:3d} / {len(locked_df)} ({pct:.0f}%)")

    # Unlocked roster
    if not unlocked_df.empty:
        print(f"\nUnlocked candidates ({len(unlocked_df)} rows):")
        print(f"  {'date':>10s}  {'ticker':>6s}  {'scanner':>25s}  {'score':>6s}  "
              f"{'rvol':>6s}  {'vwap_gate':>9s}  {'spread_gate':>11s}")
        for _, row in unlocked_df.iterrows():
            rvol_note = row["gate_rvol_note"]
            vwap_note = "PASS" if row["gate_vwap"] else "FAIL"
            spread_note = "PASS" if row["gate_spread"] else "FAIL"
            score = row.get("candidate_score")
            score_s = f"{float(score):.1f}" if score is not None else "N/A"
            print(f"  {str(row['trade_date']):>10s}  {str(row['ticker']):>6s}  "
                  f"{str(row.get('scanner_name','?')):>25s}  {score_s:>6s}  "
                  f"{rvol_note:>6s}  {vwap_note:>9s}  {spread_note:>11s}")

    # Unlocks by date
    if not unlocked_df.empty:
        print(f"\nUnlocks per trade_date:")
        for dt, cnt in unlocked_df["trade_date"].value_counts().sort_index().items():
            print(f"  {dt}: {cnt}")

    # ------------------------------------------------------------------
    # 3. Simulation — UNLOCKED arm (primary)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 3: SIMULATION — UNLOCKED ARM (primary)")
    print("=" * 72)
    print(f"  Bracket: SL={SL_PCT}%  TP={TP_PCT}%  BE=OFF")
    print(f"  Entry: OPEN of first 1h bar after 13:30 UTC on trade_date")

    unlocked_merged = pm_df[pm_df["id"].isin(unlocked_df["id"])].copy()
    unlocked_results = []
    for _, row in unlocked_merged.iterrows():
        res = simulate_candidate(dict(row), candles_by_ticker, SL_PCT, TP_PCT)
        res["id"] = row["id"]
        res["ticker"] = row["ticker"]
        res["trade_date"] = row["trade_date"]
        res["scanner_name"] = row["scanner_name"]
        res["candidate_score"] = row["candidate_score"]
        res["reason"] = row["reason"]
        unlocked_results.append(res)

    unlocked_sim = pd.DataFrame(unlocked_results)

    if not unlocked_sim.empty:
        metrics_unlocked = compute_metrics(unlocked_sim, "UNLOCKED")
        print("\nOutcome detail (unlocked candidates):")
        for _, row in unlocked_sim.iterrows():
            pnl_s = f"{float(row['pnl_pct']):+.2f}%" if row["pnl_pct"] is not None else "N/A"
            print(f"  {str(row['trade_date']):>10s}  {str(row['ticker']):>6s}  "
                  f"{str(row.get('scanner_name','?')):>25s}  "
                  f"outcome={str(row['outcome']):>7s}  pnl={pnl_s:>8s}  "
                  f"entry={float(row['entry']):.2f}  hold_bars={row.get('hold_bars','?')}")

        print(f"\nUnlocked arm metrics:")
        print_metrics(metrics_unlocked)
    else:
        print("  No unlocked candidates to simulate.")
        metrics_unlocked = None

    # ------------------------------------------------------------------
    # 4. Simulation — ALL PM REJECTS arm (secondary, larger sample)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 4: SIMULATION — ALL PM REJECTS ARM (secondary, n-floor check)")
    print("=" * 72)
    print("  Includes all rows regardless of rescue gates — provides larger sample")
    print("  for directional signal assessment even where rescue data is absent.")

    all_pm_results = []
    for _, row in pm_df.iterrows():
        res = simulate_candidate(dict(row), candles_by_ticker, SL_PCT, TP_PCT)
        res["id"] = row["id"]
        res["ticker"] = row["ticker"]
        res["trade_date"] = row["trade_date"]
        res["scanner_name"] = row["scanner_name"]
        res["candidate_score"] = row["candidate_score"]
        res["reason"] = row["reason"]
        all_pm_results.append(res)

    all_pm_sim = pd.DataFrame(all_pm_results)
    metrics_all_pm = compute_metrics(all_pm_sim, "ALL PM REJECTS")
    print(f"\nAll PM rejects arm metrics:")
    print_metrics(metrics_all_pm)

    # ------------------------------------------------------------------
    # 5. Benchmark — ordered/gate-passing cohort
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 5: BENCHMARK — ORDERED/GATE-PASSING COHORT")
    print("=" * 72)

    bench_df = fetch_df(conn, BENCHMARK_SQL)
    print(f"\nBenchmark ordered candidates: {len(bench_df)}")
    if bench_df.empty:
        print("  No benchmark candidates found.")
        metrics_bench = None
    else:
        bench_results = []
        for _, row in bench_df.iterrows():
            res = simulate_candidate(dict(row), candles_by_ticker, SL_PCT, TP_PCT)
            res["id"] = row["id"]
            res["ticker"] = row["ticker"]
            res["trade_date"] = row["trade_date"]
            res["scanner_name"] = row["scanner_name"]
            res["candidate_score"] = row["candidate_score"]
            bench_results.append(res)

        bench_sim = pd.DataFrame(bench_results)
        metrics_bench = compute_metrics(bench_sim, "BENCHMARK (ordered)")

        print("\nBenchmark detail:")
        for _, row in bench_sim.iterrows():
            pnl_s = f"{float(row['pnl_pct']):+.2f}%" if row["pnl_pct"] is not None else "N/A"
            print(f"  {str(row['trade_date']):>10s}  {str(row['ticker']):>6s}  "
                  f"{str(row.get('scanner_name','?')):>25s}  "
                  f"outcome={str(row['outcome']):>7s}  pnl={pnl_s:>8s}  "
                  f"entry={float(row['entry']):.2f}")

        print(f"\nBenchmark metrics:")
        print_metrics(metrics_bench)

    # ------------------------------------------------------------------
    # 6. Comparison summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 6: COMPARISON SUMMARY")
    print("=" * 72)

    def _pf_str(m):
        if m is None or m["pf"] is None:
            return "N/A"
        return f"{m['pf']:.3f}"

    def _wr_str(m):
        if m is None or m["wr_pct"] is None:
            return "N/A"
        return f"{m['wr_pct']:.1f}%"

    def _n_res(m):
        if m is None:
            return "N/A"
        return str(m["n_resolved"])

    print(f"\n  {'Arm':30s}  {'n_resolved':>10s}  {'WR':>7s}  {'PF':>7s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*7}  {'-'*7}")
    for label, m in [
        ("UNLOCKED (rescue-confirmed)", metrics_unlocked),
        ("ALL PM REJECTS", metrics_all_pm),
        ("BENCHMARK (ordered)", metrics_bench),
    ]:
        print(f"  {label:30s}  {_n_res(m):>10s}  {_wr_str(m):>7s}  {_pf_str(m):>7s}")

    # ------------------------------------------------------------------
    # 7. Statistical caveats and verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SECTION 7: STATISTICAL CAVEATS AND VERDICT")
    print("=" * 72)

    print("""
  SAMPLE-SIZE THRESHOLDS (system convention):
    n_resolved < 15: directional-only, no confidence in rates
    n_resolved 15-49: exploratory, report with major caveats
    n_resolved >= 50: indicative (weight half)
    n_resolved >= 100: meaningful (single-regime caveat still applies)

  SIMULATION FIDELITY NOTES:
    - Entry is proxied at 13:30 UTC on trade_date (first 1h bar after 12:30 UTC).
      Actual intraday snapshot time is not stored in candidates; the candidate is
      captured mid-session so the true entry bar is uncertain within the session.
    - SL 2%, TP 7%, no BE arm — matches current live config (Jun 12 2026 audit).
    - Timeout at {td} trading days = mark-to-market at final bar close.
    - No spread cost modeled in simulation pnl_pct.

  NULL SNAPSHOT POLICY:
    - NULL intraday_rvol or session_vwap → rescue gate fails → NOT in UNLOCKED arm.
    - Jun 4 rows had NULL intraday_rvol/session_vwap (old schema, pre-column migration).
      These dates contribute to ALL-PM-REJECTS arm only, not UNLOCKED arm.
    - Going forward, both columns are populated; unlock rate is deterministic.

  SINGLE-REGIME CAVEAT:
    - All candidates are from 2026-06-04 to 2026-06-11 (8 calendar days, ~5 trading days).
    - This is a SINGLE market regime. Historical edge under this bracket cannot be
      extrapolated without multi-week or multi-regime data.
""".format(td=TIMEOUT_TRADING_DAYS))

    # Verdict
    print("  ONE-LINE VERDICT:")

    u_n = metrics_unlocked["n_resolved"] if metrics_unlocked else 0
    u_pf = metrics_unlocked["pf"] if metrics_unlocked else None
    a_n = metrics_all_pm["n_resolved"]
    a_pf = metrics_all_pm["pf"]
    b_n = metrics_bench["n_resolved"] if metrics_bench else 0
    b_pf = metrics_bench["pf"] if metrics_bench else None

    if u_n < 5:
        verdict_detail = (
            f"UNDETERMINED — UNLOCKED arm has n_resolved={u_n} (minimum 15 required "
            f"for even directional signal). Gate softening cannot be validated historically; "
            f"all judgment must come from forward data."
        )
    elif u_n < 15:
        if u_pf is not None and u_pf >= 1.0:
            verdict_detail = (
                f"WEAKLY SUPPORTED but UNDETERMINED — UNLOCKED n_resolved={u_n} (<15), "
                f"PF={u_pf:.3f} directionally positive. Insufficient for statistical confidence."
            )
        else:
            verdict_detail = (
                f"WEAKLY REFUTED but UNDETERMINED — UNLOCKED n_resolved={u_n} (<15), "
                f"PF={f'{u_pf:.3f}' if u_pf is not None else 'N/A'} directionally below 1.0. "
                f"Insufficient for statistical confidence."
            )
    else:
        if u_pf is not None and u_pf > 1.5:
            verdict_detail = (
                f"SUPPORTED (weak, single-regime) — UNLOCKED n_resolved={u_n}, "
                f"PF={u_pf:.3f} > 1.5, WR={metrics_unlocked['wr_pct']:.1f}%. "
                f"Positive directional evidence, but single-regime (5 days) limits conviction."
            )
        elif u_pf is not None and u_pf >= 1.0:
            verdict_detail = (
                f"UNDETERMINED — UNLOCKED n_resolved={u_n}, PF={u_pf:.3f} "
                f"(near breakeven). Cannot distinguish edge from noise at this n."
            )
        else:
            verdict_detail = (
                f"REFUTED (single-regime) — UNLOCKED n_resolved={u_n}, "
                f"PF={f'{u_pf:.3f}' if u_pf is not None else 'N/A'} < 1.0. "
                f"Gate softening appears harmful on available history."
            )

    print(f"  {verdict_detail}")

    print("\n" + "=" * 72)
    print("Script: stock-scanner/app/stock_scanner/analysis/pm_gate_counterfactual.py")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        conn = get_conn()
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        run_analysis(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
