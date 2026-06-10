"""
Ranker decomposition: WHY is the stock candidate_score anti-predictive of forward
outcome, and does it hold on the gated/traded subset?
=================================================================================
Three replays (grading/SMC daytrade + SMC swing) all found the candidate_score
rankers NEGATIVELY rank-correlated with forward outcome on the all-signals pool
(daytrade core ~-0.11; swing core ~-0.13 sim / -0.17 vs realized_pnl_pct). This
script chases that lead:

  A. COMPONENT decomposition — Spearman of each ranker input (rs_percentile,
     tv_overall_score, rs_trend) and the composite scores vs forward outcome,
     for BOTH day-trade (3/5 sim) and swing (native-bracket sim) horizons.
  B. MONOTONICITY — RS / TV decile -> mean forward outcome (real inversion vs tail).
  C. GATED/TRADED subset (the key caveat) — does the negative relationship persist
     on (i) the high-core-score subset (proxy for live score>=65), and (ii) the
     ACTUAL traded names in broker_trades (profit_pct ground truth)?
  D. WINDOW STABILITY — monthly Spearman (single-regime artifact check).
  E. EXTENSION CONFOUND — is high RS just "extended -> mean-reverts"?

Read-only. Reuses validated sims/labels (daytrade_edge_sim, grading_replay,
swing_replay). Usage:
    docker exec stock-scanner python -m stock_scanner.analysis.ranker_decomp

CAVEAT: all-signals analysis is in-sample on ~Dec25->Jun26 (single-ish regime),
and is the ALL-SIGNALS ranking (live VWAP/spread/intraday gates not reconstructable).
The broker_trades block (C-ii) is the only live-gated, real-execution ground truth.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import (
    get_conn, fetch_df, build_candle_index, build_daily_index, bars_after,
)
from stock_scanner.analysis.grading_replay import core_score, build_outcome_cache, spearman
from stock_scanner.analysis.swing_replay import swing_core, swing_outcome

SWING_H = 20  # trading days


def load(conn) -> pd.DataFrame:
    sql = """
        SELECT s.id AS signal_id, s.scanner_name, s.ticker, s.signal_timestamp,
               s.signal_date, s.entry_price::float8 AS entry_price,
               s.stop_loss::float8 AS stop_loss, s.take_profit_1::float8 AS take_profit_1,
               s.realized_pnl_pct::float8 AS realized_pnl_pct,
               m.relative_volume::float8 AS relative_volume,
               m.daily_range_percent::float8 AS daily_range_percent,
               m.rs_percentile::float8 AS rs_percentile, m.rs_trend AS rs_trend,
               m.tv_overall_score::float8 AS tv_overall_score,
               m.rsi_14::float8 AS rsi_14, m.atr_14::float8 AS atr_14,
               m.ema_20::float8 AS ema_20, m.price_change_5d::float8 AS price_change_5d,
               m.pct_from_52w_high::float8 AS pct_from_52w_high
        FROM stock_scanner_signals s
        LEFT JOIN stock_screening_metrics m
               ON m.ticker = s.ticker AND m.calculation_date = s.signal_date
        WHERE s.signal_type = 'BUY'
        ORDER BY s.signal_timestamp
    """
    df = fetch_df(conn, sql)
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"]).dt.tz_localize(None)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["rs_trend_sign"] = df["rs_trend"].map(
        {"improving": 1.0, "deteriorating": -1.0}).fillna(0.0)
    return df


def load_broker(conn) -> pd.DataFrame:
    """Closed broker trades joined to their signal's ranker inputs — the only
    live-gated, real-execution ground truth."""
    sql = """
        SELECT bt.profit_pct::float8 AS profit_pct, bt.ticker,
               bt.open_price::float8 AS entry_price,
               ss.scanner_name, ss.signal_date,
               m.rs_percentile::float8 AS rs_percentile, m.rs_trend AS rs_trend,
               m.tv_overall_score::float8 AS tv_overall_score,
               m.relative_volume::float8 AS relative_volume,
               m.daily_range_percent::float8 AS daily_range_percent,
               m.rsi_14::float8 AS rsi_14, m.atr_14::float8 AS atr_14,
               m.ema_20::float8 AS ema_20, m.price_change_5d::float8 AS price_change_5d,
               m.pct_from_52w_high::float8 AS pct_from_52w_high
        FROM broker_trades bt
        JOIN stock_scanner_signals ss ON bt.signal_id = ss.id
        LEFT JOIN stock_screening_metrics m
               ON m.ticker = ss.ticker AND m.calculation_date = ss.signal_date
        WHERE bt.status = 'closed' AND bt.profit_pct IS NOT NULL
    """
    df = fetch_df(conn, sql)
    if not df.empty:
        df["rs_trend_sign"] = df["rs_trend"].map(
            {"improving": 1.0, "deteriorating": -1.0}).fillna(0.0)
    return df


def sp(df, xcol, ycol):
    return spearman(df[xcol].astype(float).values, df[ycol].astype(float).values)


def run(conn):
    print("=" * 74)
    print("RANKER DECOMPOSITION — why is candidate_score anti-predictive?")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 74)

    df = load(conn)
    df["dt_core"] = df.apply(core_score, axis=1)
    df["sw_core"] = df.apply(swing_core, axis=1)
    have = df["rs_percentile"].notna() & df["tv_overall_score"].notna()
    df = df[have].copy()
    print(f"\nLoaded {len(df):,} BUY signals with ranker inputs. "
          f"Range {df['signal_date'].min().date()} -> {df['signal_date'].max().date()}.")

    print("\nSimulating outcomes (day-trade 3/5 + swing native-bracket)...")
    cb, dbi = build_candle_index(conn), build_daily_index(conn)
    dt_cache = build_outcome_cache(df, cb, dbi)   # pnl_pess = day-trade
    df = df.merge(dt_cache[["signal_id", "pnl_pess"]], on="signal_id", how="left")
    sw = []
    for _, s in df.iterrows():
        bars, entry = bars_after(cb, str(s["ticker"]).split(".")[0], s["signal_timestamp"])
        sw.append(swing_outcome(entry, float(s["stop_loss"] or 0),
                                float(s["take_profit_1"] or 0), bars, SWING_H))
    df["pnl_swing"] = sw
    df = df.rename(columns={"pnl_pess": "pnl_dt"})
    dd = df[df["pnl_dt"].notna()]
    sd = df[df["pnl_swing"].notna()]
    print(f"  day-trade outcomes: {len(dd):,} | swing outcomes: {len(sd):,}")

    # --- A. COMPONENT DECOMPOSITION ------------------------------------------
    print("\n" + "=" * 74)
    print("A. COMPONENT DECOMPOSITION — Spearman(input, forward outcome)")
    print("=" * 74)
    print(f"  {'input':22s}  {'vs day-trade':>13s}  {'vs swing':>10s}  {'vs realized_pnl':>16s}")
    rp = df[df["realized_pnl_pct"].notna()]
    for label, col in [("rs_percentile", "rs_percentile"), ("tv_overall_score", "tv_overall_score"),
                       ("rs_trend_sign", "rs_trend_sign"), ("daytrade core", "dt_core"),
                       ("swing core", "sw_core"), ("relative_volume", "relative_volume"),
                       ("daily_range_percent", "daily_range_percent")]:
        c_dt = sp(dd, col, "pnl_dt")
        c_sw = sp(sd, col, "pnl_swing")
        c_rp = sp(rp, col, "realized_pnl_pct")
        print(f"  {label:22s}  {c_dt:>+13.4f}  {c_sw:>+10.4f}  {c_rp:>+16.4f}")

    # --- B. MONOTONICITY ------------------------------------------------------
    print("\n" + "=" * 74)
    print("B. MONOTONICITY — mean forward outcome by RS / TV decile")
    print("=" * 74)
    for feat in ["rs_percentile", "tv_overall_score"]:
        sub = sd[sd[feat].notna()].copy()
        sub["bin"] = pd.qcut(sub[feat].rank(method="first"), 10, labels=False)
        print(f"\n  {feat} decile (1=low .. 10=high)  ->  mean swing pnl% | mean day-trade pnl% | n")
        g = sub.groupby("bin")
        for b, grp in g:
            dtm = grp["pnl_dt"].dropna().mean()
            print(f"    D{int(b)+1:>2}  {feat[:12]:12s} mean={grp[feat].mean():7.1f}  "
                  f"swing={grp['pnl_swing'].mean():+7.3f}  dt={dtm:+7.3f}  n={len(grp)}")

    # --- C. GATED / TRADED SUBSET --------------------------------------------
    print("\n" + "=" * 74)
    print("C. DOES IT HOLD ON THE GATED / TRADED SUBSET? (the key caveat)")
    print("=" * 74)
    print("\n  (i) High-core-score subset (proxy for live score>=65 gate):")
    for q, name in [(0.0, "all"), (0.5, "top-50%"), (0.75, "top-25%")]:
        sub = sd[sd["sw_core"] >= sd["sw_core"].quantile(q)]
        print(f"    sw_core {name:8s} (n={len(sub):4d}): "
              f"Spearman(sw_core, swing)={sp(sub,'sw_core','pnl_swing'):+.4f}  "
              f"Spearman(rs, swing)={sp(sub,'rs_percentile','pnl_swing'):+.4f}")
    print("\n  (ii) ACTUAL traded names (broker_trades, real profit_pct = live-gated ground truth):")
    bk = load_broker(conn)
    bk = bk[bk["rs_percentile"].notna()]
    if len(bk) >= 20:
        bk["dt_core"] = bk.apply(core_score, axis=1)
        bk["sw_core"] = bk.apply(swing_core, axis=1)
        print(f"    n={len(bk)} closed broker trades with ranker inputs. "
              f"Mean profit_pct={bk['profit_pct'].mean():+.3f}% (median {bk['profit_pct'].median():+.3f}%)")
        for label, col in [("rs_percentile", "rs_percentile"), ("tv_overall_score", "tv_overall_score"),
                           ("daytrade core", "dt_core"), ("swing core", "sw_core")]:
            print(f"      Spearman({label:18s}, profit_pct) = {sp(bk, col, 'profit_pct'):+.4f}")
        print("    NOTE: small n + single regime — directional, not conclusive.")
    else:
        print(f"    Too few broker trades with ranker inputs (n={len(bk)}).")

    # --- D. WINDOW STABILITY --------------------------------------------------
    print("\n" + "=" * 74)
    print("D. WINDOW STABILITY — monthly Spearman(sw_core, swing) & (rs, swing)")
    print("=" * 74)
    sd2 = sd.copy()
    sd2["month"] = sd2["signal_date"].dt.to_period("M").astype(str)
    print(f"  {'month':8s}  {'n':>5s}  {'rho(sw_core,swing)':>20s}  {'rho(rs,swing)':>15s}")
    for m, grp in sd2.groupby("month"):
        if len(grp) < 30:
            continue
        print(f"  {m:8s}  {len(grp):>5d}  {sp(grp,'sw_core','pnl_swing'):>+20.4f}  "
              f"{sp(grp,'rs_percentile','pnl_swing'):>+15.4f}")

    # --- E. EXTENSION CONFOUND ------------------------------------------------
    print("\n" + "=" * 74)
    print("E. EXTENSION CONFOUND — is high RS just 'extended -> mean-reverts'?")
    print("=" * 74)
    e = sd[sd["price_change_5d"].notna() & sd["pct_from_52w_high"].notna()].copy()
    print(f"  Spearman(rs_percentile, price_change_5d)   = {sp(e,'rs_percentile','price_change_5d'):+.4f}")
    print(f"  Spearman(rs_percentile, pct_from_52w_high) = {sp(e,'rs_percentile','pct_from_52w_high'):+.4f}")
    med = e["price_change_5d"].median()
    lo = e[e["price_change_5d"] <= med]
    hi = e[e["price_change_5d"] > med]
    print(f"  Spearman(rs, swing) | LOW 5d-extension  (n={len(lo)}): {sp(lo,'rs_percentile','pnl_swing'):+.4f}")
    print(f"  Spearman(rs, swing) | HIGH 5d-extension (n={len(hi)}): {sp(hi,'rs_percentile','pnl_swing'):+.4f}")
    print("  (If the negative rs->outcome corr concentrates in HIGH-extension, the")
    print("   mechanism is late-momentum mean-reversion, not RS being bad per se.)")

    print("\n" + "=" * 74)
    print("READ: A says which input drives it; B whether it's monotone; C whether it")
    print("survives gating/real execution (the decision-relevant test); D regime")
    print("stability; E the mechanism. Script: analysis/ranker_decomp.py")
    print("=" * 74)


def main():
    try:
        conn = get_conn()
    except Exception as e:
        print(f"ERROR: DB connect failed: {e}", file=sys.stderr); sys.exit(1)
    try:
        run(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
