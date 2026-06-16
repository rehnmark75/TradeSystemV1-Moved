"""
Scanner Composition Ablation Analysis
======================================
Reuses daytrade_edge_sim's validated engine to compute per-scanner metrics
and portfolio-level PF under different scanner inclusion sets.

Usage (inside stock-scanner container):
    python -m stock_scanner.analysis.scanner_ablation

Read-only: does NOT modify any table.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# Reuse the validated engine from daytrade_edge_sim
from stock_scanner.analysis.daytrade_edge_sim import (
    build_candle_index,
    build_daily_index,
    atr_pct_before,
    bars_after,
    build_bracket,
    walk_bars,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DB_URL = "postgresql://postgres:postgres@postgres:5432/stocks"

# Ablation sets (scanner_name values that match the DB)
LOSER_3 = {"gap_and_go", "breakout_confirmation", "squeeze_momentum"}  # all PF<1.0, high volume
ALL_BELOW_1 = {
    "gap_and_go", "breakout_confirmation", "squeeze_momentum",
    "macd_momentum", "ema_pullback",
    "premarket_catalyst", "ultimate_ma_mtf",  # added from task context
}
# Winners by PF>=1.2 and n>=150 (from task context values)
ADEQUATE_N_WINNERS = {"high_retest", "relative_strength_leader", "pocket_pivot", "sector_rotation_leader"}

SPLIT_DATE = "2026-03-01"  # separates pre-March (multi-regime) from later


def get_conn():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def fetch_df(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def compute_pf(df: pd.DataFrame, outcome_col: str, pnl_col: str):
    """Pooled PF = sum(gross wins) / abs(sum(gross losses))."""
    wins = df[outcome_col] == "win"
    losses = df[outcome_col] == "loss"
    gross_win = df.loc[wins, pnl_col].sum()
    gross_loss = abs(df.loc[losses, pnl_col].sum())
    n = len(df)
    n_wins = wins.sum()
    n_losses = losses.sum()
    n_resolved = n_wins + n_losses
    wr = n_wins / n_resolved * 100 if n_resolved > 0 else 0.0
    exp = df[pnl_col].mean() * 100 if n > 0 else 0.0  # avg pnl% per trade
    pf = gross_win / gross_loss if gross_loss > 0 else (9.99 if gross_win > 0 else None)
    return {
        "n": n,
        "n_wins": int(n_wins),
        "n_losses": int(n_losses),
        "n_resolved": int(n_resolved),
        "wr_pct": round(wr, 1),
        "pooled_pf": round(pf, 2) if pf is not None else None,
        "expectancy_pct_per_trade": round(exp, 3),
    }


def summarize_set(label: str, df: pd.DataFrame, baseline_n: int):
    """Print summary for a scanner inclusion set."""
    if df.empty:
        print(f"\n{label}: NO DATA")
        return
    metrics = compute_pf(df, "outcome", "pnl_pct")
    vol_pct = metrics["n"] / baseline_n * 100 if baseline_n > 0 else 0
    print(f"\n  {label}")
    print(f"    n={metrics['n']} ({vol_pct:.0f}% of baseline signals)")
    print(f"    Pooled PF:    {metrics['pooled_pf']}")
    print(f"    Win Rate:     {metrics['wr_pct']:.1f}%")
    print(f"    Expectancy:   {metrics['expectancy_pct_per_trade']:.3f}% avg pnl/trade")
    print(f"    n_wins={metrics['n_wins']} n_losses={metrics['n_losses']} "
          f"resolved={metrics['n_resolved']}")


def main():
    print("=" * 70)
    print("SCANNER COMPOSITION ABLATION")
    print(f"Run timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    conn = get_conn()

    # ------------------------------------------------------------------
    # 1. Load signals
    # ------------------------------------------------------------------
    signals_sql = """
        SELECT id, signal_timestamp, scanner_name, ticker, signal_type, signal_date
        FROM stock_scanner_signals
        WHERE signal_type = 'BUY'
        ORDER BY signal_timestamp
    """
    signals = fetch_df(conn, signals_sql)
    print(f"\nTotal BUY signals loaded: {len(signals):,}")

    # History span per scanner
    print("\nHistory span per scanner:")
    span = signals.groupby("scanner_name").agg(
        first=("signal_timestamp", "min"),
        last=("signal_timestamp", "max"),
        n=("id", "count"),
    ).reset_index()
    for _, r in span.sort_values("first").iterrows():
        days = (pd.Timestamp(r["last"]) - pd.Timestamp(r["first"])).days
        print(f"  {r['scanner_name']:35s}: n={r['n']:4d}, "
              f"{str(r['first'])[:10]} → {str(r['last'])[:10]} ({days}d)")

    # ------------------------------------------------------------------
    # 2. Load candles and run walk-forward simulation
    # ------------------------------------------------------------------
    print("\nLoading 1h candles...")
    candles_by_ticker = build_candle_index(conn)
    print(f"  {sum(len(v) for v in candles_by_ticker.values()):,} bars, "
          f"{len(candles_by_ticker)} tickers")

    print("Loading daily candles for ATR...")
    daily_by_ticker = build_daily_index(conn)
    print(f"  {sum(len(v) for v in daily_by_ticker.values()):,} bars, "
          f"{len(daily_by_ticker)} tickers")

    print(f"\nRunning sim on {len(signals):,} signals...")

    sim_rows = []
    no_data_count = 0

    for _, sig in signals.iterrows():
        base_ticker = str(sig["ticker"]).split(".")[0]
        sig_ts = sig["signal_timestamp"]

        atr_pct = atr_pct_before(daily_by_ticker, base_ticker, sig_ts)
        bar_df, entry = bars_after(candles_by_ticker, base_ticker, sig_ts)

        if bar_df.empty or entry <= 0:
            no_data_count += 1
            continue

        bracket = build_bracket(entry, None)  # fixed 3%SL/5%TP
        result = walk_bars(entry, bracket, bar_df)

        sim_rows.append({
            "signal_id": sig["id"],
            "scanner_name": sig["scanner_name"],
            "ticker": sig["ticker"],
            "signal_timestamp": sig["signal_timestamp"],
            "signal_date": sig["signal_date"],
            "outcome": result["outcome"],
            "pnl_pct": result["pnl_pct"] / 100.0,  # convert to fraction for pooled PF
            "pnl_pct_raw": result["pnl_pct"],
            "hold_bars": result["hold_bars"],
            "be_armed": result.get("be_armed", False),
        })

    sim_df = pd.DataFrame(sim_rows)
    print(f"\nSimulation complete: {len(sim_df):,} resolved, {no_data_count:,} dropped (no candle data)")

    # Collapse win/loss (exclude timeout for PF -- treat as partial)
    # For pooled PF: use pnl_pct_raw, wins = positive, losses = negative
    sim_wl = sim_df[sim_df["outcome"].isin(["win", "loss"])].copy()
    print(f"  Clean win/loss outcomes (excl. timeout/be): {len(sim_wl):,}")
    print(f"  Timeout outcomes: {(sim_df['outcome']=='timeout').sum():,}")
    print(f"  BE exits: {(sim_df['outcome']=='be').sum():,}")

    # ------------------------------------------------------------------
    # 3. Per-scanner metrics on full history (with pnl_pct_raw)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 1: PER-SCANNER FULL-HISTORY METRICS (fixed 3%SL/5%TP)")
    print("=" * 70)
    print(f"\n  {'scanner':35s}  {'n':>5s}  {'n_wl':>5s}  {'WR%':>6s}  {'pooled_PF':>10s}  "
          f"{'exp%/trade':>11s}  {'timeout%':>9s}  {'regime_flag':>20s}")

    all_scanners = sorted(sim_df["scanner_name"].unique())
    scanner_metrics = {}
    for scn in all_scanners:
        grp = sim_df[sim_df["scanner_name"] == scn]
        grp_wl = grp[grp["outcome"].isin(["win", "loss"])]
        n = len(grp)
        n_wl = len(grp_wl)
        n_wins = (grp_wl["outcome"] == "win").sum()
        n_losses = (grp_wl["outcome"] == "loss").sum()
        wr = n_wins / n_wl * 100 if n_wl > 0 else 0.0
        gross_win = grp_wl.loc[grp_wl["outcome"]=="win", "pnl_pct_raw"].sum()
        gross_loss = abs(grp_wl.loc[grp_wl["outcome"]=="loss", "pnl_pct_raw"].sum())
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (9.99 if gross_win > 0 else None)
        to_pct = (grp["outcome"] == "timeout").sum() / n * 100
        exp = grp["pnl_pct_raw"].mean()

        # History flag
        first_sig = pd.Timestamp(grp["signal_timestamp"].min()).tz_localize(None)
        last_sig = pd.Timestamp(grp["signal_timestamp"].max()).tz_localize(None)
        days_history = (last_sig - first_sig).days
        if first_sig >= pd.Timestamp("2026-05-26"):
            regime_flag = "SINGLE-REGIME (3wk)"
        elif first_sig >= pd.Timestamp("2026-06-08"):
            regime_flag = "SINGLE-REGIME (1wk)"
        elif days_history >= 90:
            regime_flag = "MULTI-REGIME (ok)"
        else:
            regime_flag = "PARTIAL-REGIME"

        pf_str = f"{pf:.2f}" if pf is not None else "  N/A"
        print(f"  {scn:35s}  {n:>5d}  {n_wl:>5d}  {wr:>6.1f}  {pf_str:>10s}  "
              f"{exp:>+11.2f}  {to_pct:>9.1f}  {regime_flag:>20s}")

        scanner_metrics[scn] = {
            "n": n, "n_wl": n_wl, "n_wins": int(n_wins), "n_losses": int(n_losses),
            "wr": wr, "pf": pf, "exp_pct": exp, "to_pct": to_pct, "regime_flag": regime_flag,
        }

    # ------------------------------------------------------------------
    # 4. ABLATION: Portfolio-level PF under inclusion sets
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 2: PORTFOLIO ABLATION")
    print("=" * 70)

    # Only use win/loss outcomes for pooled PF (timeout treated as 0 pnl)
    # Use pnl_pct_raw: +5 for win, -3 for loss, 0 for timeout/be
    sim_df["pnl_for_pf"] = 0.0
    sim_df.loc[sim_df["outcome"] == "win", "pnl_for_pf"] = sim_df.loc[sim_df["outcome"] == "win", "pnl_pct_raw"]
    sim_df.loc[sim_df["outcome"] == "loss", "pnl_for_pf"] = sim_df.loc[sim_df["outcome"] == "loss", "pnl_pct_raw"]

    def portfolio_pf(df: pd.DataFrame, label: str, baseline_n: int):
        if df.empty:
            print(f"\n  {label}: NO DATA")
            return
        wins = df["outcome"] == "win"
        losses = df["outcome"] == "loss"
        gross_win = df.loc[wins, "pnl_pct_raw"].sum()
        gross_loss = abs(df.loc[losses, "pnl_pct_raw"].sum())
        n = len(df)
        n_wins = wins.sum()
        n_losses = losses.sum()
        n_resolved = n_wins + n_losses
        wr = n_wins / n_resolved * 100 if n_resolved > 0 else 0.0
        pf = gross_win / gross_loss if gross_loss > 0 else (9.99 if gross_win > 0 else None)
        exp = df["pnl_pct_raw"].mean()
        vol_pct = n / baseline_n * 100

        pf_str = f"{pf:.3f}" if pf is not None else "N/A"
        print(f"\n  {label}")
        print(f"    n={n:,} ({vol_pct:.0f}% of baseline)   "
              f"pooled PF={pf_str}   WR={wr:.1f}%   exp={exp:+.2f}%/trade")
        print(f"    n_wins={n_wins} n_losses={n_losses} "
              f"n_resolved={n_resolved}")

    # Baseline
    baseline_n = len(sim_df)
    portfolio_pf(sim_df, "BASELINE (all scanners)", baseline_n)

    # Drop 3 losers
    keep_drop3 = sim_df[~sim_df["scanner_name"].isin(LOSER_3)]
    portfolio_pf(keep_drop3, "DROP losers (gap_and_go + breakout_confirmation + squeeze_momentum)", baseline_n)

    # Drop all PF<1.0 scanners
    # From full-history metrics, identify which ones are < 1.0
    losers_auto = {scn for scn, m in scanner_metrics.items()
                   if m["pf"] is not None and m["pf"] < 1.0}
    print(f"\n  (Auto-identified PF<1.0 scanners: {sorted(losers_auto)})")
    keep_no_losers = sim_df[~sim_df["scanner_name"].isin(losers_auto)]
    portfolio_pf(keep_no_losers, "DROP all PF<1.0 scanners (auto-identified)", baseline_n)

    # Keep only adequate-n winners (PF>=1.2, n>=150 from task context)
    keep_winners = sim_df[sim_df["scanner_name"].isin(ADEQUATE_N_WINNERS)]
    portfolio_pf(keep_winners, "KEEP only adequate-n winners (PF>=1.2, n>=150): high_retest + RSL + pocket_pivot + SRL", baseline_n)

    # ------------------------------------------------------------------
    # 5. ROBUSTNESS: TEMPORAL SPLIT (pre vs post split date)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 3: ROBUSTNESS — TEMPORAL SPLIT")
    print("=" * 70)
    print(f"\n  Split date: {SPLIT_DATE}")
    print(f"  EARLY half = before {SPLIT_DATE} (multi-regime, incl. Dec 2025 – Feb 2026)")
    print(f"  LATE half = {SPLIT_DATE} onwards")
    print(f"\n  Scanners that launched AFTER {SPLIT_DATE} cannot be split — flagged.")

    early_df = sim_df[pd.to_datetime(sim_df["signal_date"]) < pd.Timestamp(SPLIT_DATE)]
    late_df = sim_df[pd.to_datetime(sim_df["signal_date"]) >= pd.Timestamp(SPLIT_DATE)]
    print(f"\n  Early period signals (with outcome): {len(early_df):,}")
    print(f"  Late period signals (with outcome): {len(late_df):,}")

    # Per-scanner split
    print(f"\n  {'scanner':35s}  {'early_n':>7s}  {'early_PF':>9s}  {'early_WR':>9s}  "
          f"{'late_n':>6s}  {'late_PF':>8s}  {'late_WR':>8s}  {'stable?':>8s}")

    for scn in all_scanners:
        e_grp = early_df[early_df["scanner_name"] == scn]
        l_grp = late_df[late_df["scanner_name"] == scn]
        e_wl = e_grp[e_grp["outcome"].isin(["win", "loss"])]
        l_wl = l_grp[l_grp["outcome"].isin(["win", "loss"])]

        def pf_from(grp_wl):
            if len(grp_wl) < 5:
                return None, 0.0
            wins = grp_wl[grp_wl["outcome"] == "win"]
            losses = grp_wl[grp_wl["outcome"] == "loss"]
            gw = wins["pnl_pct_raw"].sum()
            gl = abs(losses["pnl_pct_raw"].sum())
            pf = gw / gl if gl > 0 else (9.99 if gw > 0 else None)
            wr = len(wins) / len(grp_wl) * 100
            return pf, wr

        e_pf, e_wr = pf_from(e_wl)
        l_pf, l_wr = pf_from(l_wl)

        e_n = len(e_grp)
        l_n = len(l_grp)

        if e_n == 0:
            stable = "SINGLE-REG"
        elif e_n < 30 or l_n < 30:
            stable = "THIN"
        elif e_pf is None or l_pf is None:
            stable = "?"
        else:
            # Both PF on same side of 1.0 = stable direction
            if (e_pf >= 1.0) == (l_pf >= 1.0):
                stable = "YES"
            else:
                stable = "UNSTABLE"

        e_pf_s = f"{e_pf:.2f}" if e_pf is not None else "  N/A"
        l_pf_s = f"{l_pf:.2f}" if l_pf is not None else "  N/A"
        e_wr_s = f"{e_wr:.1f}%" if e_pf is not None else "  N/A"
        l_wr_s = f"{l_wr:.1f}%" if l_pf is not None else "  N/A"

        print(f"  {scn:35s}  {e_n:>7d}  {e_pf_s:>9s}  {e_wr_s:>9s}  "
              f"{l_n:>6d}  {l_pf_s:>8s}  {l_wr_s:>8s}  {stable:>8s}")

    # ------------------------------------------------------------------
    # 6. POOL DEPTH ESTIMATE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 4: POOL DEPTH — can freed slots be filled by better scanners?")
    print("=" * 70)

    # Per trading day, count how many non-loser signals were available
    print("\n  Signals per trading day from NON-loser scanners (PF>=1.0, post May-26):")
    recent = sim_df[
        (~sim_df["scanner_name"].isin(LOSER_3)) &
        (pd.to_datetime(sim_df["signal_date"]) >= pd.Timestamp("2026-05-26"))
    ].copy()
    recent["signal_date_str"] = recent["signal_date"].astype(str)
    daily_depth = recent.groupby("signal_date_str")["signal_id"].count().reset_index()
    daily_depth.columns = ["signal_date", "n_signals"]
    print(f"\n  {'date':12s}  {'signals_from_good_scanners':>26s}")
    for _, r in daily_depth.sort_values("signal_date", ascending=False).head(20).iterrows():
        print(f"  {r['signal_date']:12s}  {r['n_signals']:>26d}")
    print(f"\n  Median per trading day (good scanners): {daily_depth['n_signals'].median():.0f}")
    print(f"  8 trade slots / run → pool is {'DEEP ENOUGH' if daily_depth['n_signals'].median() > 30 else 'THIN'} "
          f"(need >8 to fill freed slots)")

    # ------------------------------------------------------------------
    # 7. VERDICT TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 5: VERDICT TABLE")
    print("=" * 70)
    print(f"\n  Confidence tiers (n = resolved wins+losses in full-history):")
    print(f"    HIGH:    n>=200, multi-regime — stand behind the verdict")
    print(f"    MEDIUM:  n 100-199 — indicative, verify after more trades")
    print(f"    LOW:     n<100 OR single-regime — monitor only, do NOT act")

    print(f"\n  {'scanner':35s}  {'n_wl':>6s}  {'PF':>6s}  {'regime':>22s}  {'verdict':>12s}  {'action':>10s}")
    for scn in all_scanners:
        m = scanner_metrics[scn]
        n_wl = m["n_wl"]
        pf = m["pf"]
        regime = m["regime_flag"]
        pf_str = f"{pf:.2f}" if pf is not None else "N/A"

        if "SINGLE-REGIME" in regime:
            conf = "LOW (1-reg)"
            verdict = f"MONITOR"
            action = "HOLD"
        elif n_wl >= 200:
            if pf is not None and pf < 0.9:
                conf = "HIGH"
                verdict = "LOSER"
                action = "CAP/DROP"
            elif pf is not None and pf < 1.0:
                conf = "HIGH"
                verdict = "MARGINAL"
                action = "MONITOR"
            elif pf is not None and pf >= 1.2:
                conf = "HIGH"
                verdict = "WINNER"
                action = "KEEP"
            else:
                conf = "HIGH"
                verdict = "MARGINAL"
                action = "MONITOR"
        elif n_wl >= 100:
            if pf is not None and pf < 0.9:
                conf = "MEDIUM"
                verdict = "LOSER"
                action = "CAP/DROP"
            elif pf is not None and pf >= 1.2:
                conf = "MEDIUM"
                verdict = "WINNER"
                action = "KEEP"
            else:
                conf = "MEDIUM"
                verdict = "MARGINAL"
                action = "MONITOR"
        else:
            conf = "LOW (thin)"
            verdict = "INSUFFICIENT"
            action = "HOLD"

        print(f"  {scn:35s}  {n_wl:>6d}  {pf_str:>6s}  {regime:>22s}  "
              f"{verdict:>12s}  {action:>10s}")

    print("\n" + "=" * 70)
    print("Script: stock-scanner/app/stock_scanner/analysis/scanner_ablation.py")
    print("=" * 70)


if __name__ == "__main__":
    try:
        conn_test = get_conn()
        conn_test.close()
    except Exception as e:
        print(f"DB connect error: {e}", file=sys.stderr)
        sys.exit(1)
    main()
