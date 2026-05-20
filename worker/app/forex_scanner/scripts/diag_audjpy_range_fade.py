#!/usr/bin/env python3
"""
Comprehensive RANGE_FADE AUDJPY diagnostic.

Purpose
-------
Find WHERE the edge lives for AUDJPY so we can write data-backed gate rules.
Runs three permissive 90d backtests (bb_mult 1.5/1.8/2.0), enriches each
signal with 1h ADX from ig_candles_backtest, then cross-tabs outcomes on:

  A. Direction × HTF bias          (highest priority — asymmetry likely)
  B. Direction × Session bucket    (Asian/London/Overlap/NY/Late-NY)
  C. Direction × 1h ADX bucket     (<20 / 20-30 / >30)
  D. Direction × HTF bias × Session (3-way, n≥5 shown)
  E. Direction × RSI-at-entry bucket
  F. Direction × band-width relative bucket (tight/mid/wide vs recent)
  G. Day-of-week × direction
  H. Post-loss effect in same session window
  I. bb_mult sweep summary (compare the three runs)

Usage (inside task-worker):
  python /app/forex_scanner/scripts/diag_audjpy_range_fade.py
  python /app/forex_scanner/scripts/diag_audjpy_range_fade.py --no-rerun
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from datetime import timezone

import pandas as pd  # resolved inside Docker; Pyright can't see the container venv

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/forex_scanner")

EPIC = "CS.D.AUDJPY.MINI.IP"
DAYS = 90
# Permissive baseline uses 30d + moderate cooldown so the run finishes quickly
# (cooldown=0 + rsi=50/50 + 90d = ~1800 signals, multi-hour runtime).
# We widen RSI 35→40 / 60→60 and add 30 min cooldown — enough signals for
# pattern analysis while keeping runtime under 10 minutes.
PERMISSIVE_DAYS = 30
PIP = 0.01   # JPY cross


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_adx_series(rows):
    """Compute ADX(14) from list of (start_time, open, high, low, close)."""

    if len(rows) < 20:
        return None
    df = pd.DataFrame(rows, columns=["start_time", "open", "high", "low", "close"])
    df = df.sort_values("start_time").reset_index(drop=True)
    n = 14
    df["h-l"]  = df["high"] - df["low"]
    df["h-pc"] = (df["high"] - df["close"].shift(1)).abs()
    df["l-pc"] = (df["low"]  - df["close"].shift(1)).abs()
    df["tr"]   = df[["h-l", "h-pc", "l-pc"]].max(axis=1)
    df["+dm"]  = (df["high"] - df["high"].shift(1)).clip(lower=0).where(
        (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]), 0.0
    )
    df["-dm"]  = (df["low"].shift(1) - df["low"]).clip(lower=0).where(
        (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)), 0.0
    )
    a = 1.0 / n
    atr = df["tr"].ewm(alpha=a, adjust=False).mean()
    pdi = 100 * df["+dm"].ewm(alpha=a, adjust=False).mean() / atr.replace(0, float("nan"))
    mdi = 100 * df["-dm"].ewm(alpha=a, adjust=False).mean() / atr.replace(0, float("nan"))
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, float("nan"))
    adx = dx.ewm(alpha=a, adjust=False).mean()
    v = adx.iloc[-1]
    return float(v) if v == v else None  # NaN check


def session_label(hour: int) -> str:
    if 22 <= hour or hour < 6:
        return "Asian"
    if 6 <= hour < 12:
        return "London"
    if 12 <= hour < 16:
        return "Overlap"
    if 16 <= hour < 21:
        return "NY"
    return "Late"   # 21


def pf_str(wins_pips, loss_pips):
    if loss_pips <= 0:
        return "∞" if wins_pips > 0 else "0.00"
    return f"{wins_pips / loss_pips:.2f}"


def _parse_int(text: str, label: str):
    m = re.findall(rf"{re.escape(label)}:\s+(\d+)", text)
    return int(m[-1]) if m else None


def _parse_float(text: str, label: str):
    m = re.findall(rf"{re.escape(label)}:\s+(-?\d+(?:\.\d+)?)", text)
    return float(m[-1]) if m else None


# ── backtest runner ───────────────────────────────────────────────────────────

def run_permissive_bt(bb_mult: float, days: int = PERMISSIVE_DAYS) -> dict:
    """Run a near-permissive baseline with one bb_mult value.

    Uses 30d + 30 min cooldown + RSI 40/60 to avoid the 1800+ signal / multi-hour
    blowup that rsi=50/50 + cooldown=0 + 90d produces.
    """
    overrides = {
        "erf_profile": "5m",
        "monitor_only": "false",
        "rsi_oversold": 40,
        "rsi_overbought": 60,
        "london_start_hour_utc": 0,
        "new_york_end_hour_utc": 23,
        "range_proximity_pips": 999.0,
        "max_current_range_pips": 999.0,
        "min_band_width_pips": 0.0,
        "max_band_width_pips": 9999.0,
        "allow_neutral_htf": "true",
        "signal_cooldown_minutes": 30,
        "fixed_stop_loss_pips": 10,
        "fixed_take_profit_pips": 15,
        "bb_mult": bb_mult,
    }
    cmd = ["python", "/app/forex_scanner/backtest_cli.py",
           "--epic", EPIC, "--days", str(days), "--strategy", "RANGE_FADE"]
    for k, v in overrides.items():
        cmd += ["--override", f"{k}={v}"]

    print(f"  [bt] bb_mult={bb_mult} days={days} …", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    return {
        "bb_mult": bb_mult,
        "n": _parse_int(out, "Total Signals") or 0,
        "winners": _parse_int(out, "Winners") or 0,
        "losers": _parse_int(out, "Losers") or 0,
        "wr": _parse_float(out, "Win Rate") or 0.0,
        "pf": _parse_float(out, "Profit Factor") or 0.0,
        "exp": _parse_float(out, "Expectancy") or 0.0,
        "ok": proc.returncode == 0,
        "raw": out,
    }


# ── DB fetch ─────────────────────────────────────────────────────────────────

def fetch_signals_with_adx(conn, epic: str) -> pd.DataFrame:

    cur = conn.cursor()

    # Most recent execution_id for this epic
    cur.execute("""
        SELECT execution_id FROM backtest_signals
        WHERE epic = %s AND strategy_name = 'RANGE_FADE'
        ORDER BY created_at DESC LIMIT 1
    """, (epic,))
    row = cur.fetchone()
    if not row:
        print("  ❌ No RANGE_FADE signals in DB for", epic)
        return pd.DataFrame()
    exec_id = row[0]
    print(f"  Using execution_id={exec_id}", flush=True)

    cur.execute("""
        SELECT
            signal_timestamp, signal_type, trade_result, pips_gained,
            confidence_score,
            (indicator_values::jsonb->>'htf_bias')           AS htf_bias,
            (indicator_values::jsonb->>'rsi')::numeric        AS rsi,
            (indicator_values::jsonb->>'band_width_pips')::numeric AS bw_pips
        FROM backtest_signals
        WHERE execution_id = %s AND epic = %s
        ORDER BY signal_timestamp
    """, (exec_id, epic))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    records = []
    for sig_ts, sig_type, result, pips, conf, htf, rsi, bw in rows:
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        # Fetch 1h candles for ADX
        cur.execute("""
            SELECT start_time, open, high, low, close
            FROM ig_candles_backtest
            WHERE epic = %s AND timeframe = 60 AND start_time <= %s
            ORDER BY start_time DESC LIMIT 60
        """, (epic, sig_ts))
        candle_rows = cur.fetchall()
        adx_1h = compute_adx_series(candle_rows)

        is_win = str(result).lower() in ("win", "take_profit", "tp_hit")
        records.append({
            "ts": sig_ts,
            "direction": "BUY" if str(sig_type).upper() in ("BULL", "BUY") else "SELL",
            "is_win": is_win,
            "pips": float(pips) if pips is not None else 0.0,
            "conf": float(conf) if conf is not None else None,
            "htf": str(htf) if htf else "unknown",
            "rsi": float(rsi) if rsi is not None else None,
            "bw_pips": float(bw) if bw is not None else None,
            "adx_1h": adx_1h,
            "hour": sig_ts.astimezone(timezone.utc).hour,
            "dow": sig_ts.astimezone(timezone.utc).weekday(),  # 0=Mon
        })

    df = pd.DataFrame(records)
    df["session"] = df["hour"].apply(session_label)
    df["adx_bucket"] = pd.cut(
        df["adx_1h"].fillna(-1),
        bins=[-999, 0, 20, 30, 999],
        labels=["no_data", "<20", "20-30", ">30"],
    )
    # RSI buckets (directional)
    df["rsi_bucket"] = "n/a"
    buy_mask = df["direction"] == "BUY"
    sell_mask = df["direction"] == "SELL"
    df.loc[buy_mask, "rsi_bucket"] = pd.cut(
        df.loc[buy_mask, "rsi"].fillna(50),
        bins=[0, 25, 30, 35, 40, 45, 100],
        labels=["<25", "25-30", "30-35", "35-40", "40-45", ">45"],
    ).astype(str)
    df.loc[sell_mask, "rsi_bucket"] = pd.cut(
        df.loc[sell_mask, "rsi"].fillna(50),
        bins=[0, 55, 60, 65, 70, 75, 100],
        labels=["<55", "55-60", "60-65", "65-70", "70-75", ">75"],
    ).astype(str)
    return df


# ── reporting helpers ─────────────────────────────────────────────────────────

def crosstab(df: "pd.DataFrame", col1: str, col2: str, min_n: int = 5,
             title: str = "") -> None:
    """Print WR/PF for each (col1, col2) cell."""
    if title:
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}")
    header = f"  {'':16}  {'n':>4}  {'wins':>4}  {'WR%':>5}  {'avg_pips':>8}  PF"
    print(header)
    for v1 in sorted(df[col1].unique()):
        for v2 in sorted(df[col2].unique()):
            sub = df[(df[col1] == v1) & (df[col2] == v2)]
            if len(sub) < min_n:
                continue
            n = len(sub)
            wins = int(sub["is_win"].sum())
            wr = wins / n * 100
            avg_p = sub["pips"].mean()
            pos = sub.loc[sub["pips"] > 0, "pips"].sum()
            neg = (-sub.loc[sub["pips"] < 0, "pips"]).sum()
            pf = pf_str(pos, neg)
            tag = f"{v1}/{v2}"
            flag = " ◀" if wr >= 60 and n >= 10 else ""
            print(f"  {tag:<16}  {n:>4}  {wins:>4}  {wr:>5.1f}  {avg_p:>8.1f}  {pf}{flag}")


def triple_tab(df: "pd.DataFrame", c1: str, c2: str, c3: str,
               min_n: int = 5, title: str = "") -> None:
    if title:
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}")
    header = f"  {'':22}  {'n':>4}  {'WR%':>5}  {'avg_pips':>8}  PF"
    print(header)
    for v1 in sorted(df[c1].unique()):
        for v2 in sorted(df[c2].unique()):
            for v3 in sorted(df[c3].unique()):
                sub = df[(df[c1] == v1) & (df[c2] == v2) & (df[c3] == v3)]
                if len(sub) < min_n:
                    continue
                n = len(sub)
                wins = int(sub["is_win"].sum())
                wr = wins / n * 100
                avg_p = sub["pips"].mean()
                pos = sub.loc[sub["pips"] > 0, "pips"].sum()
                neg = (-sub.loc[sub["pips"] < 0, "pips"]).sum()
                pf = pf_str(pos, neg)
                tag = f"{v1}/{v2}/{v3}"
                flag = " ◀" if wr >= 62 and n >= 8 else ""
                print(f"  {tag:<22}  {n:>4}  {wr:>5.1f}  {avg_p:>8.1f}  {pf}{flag}")


# ── band width relative ───────────────────────────────────────────────────────

def bw_relative_analysis(df: pd.DataFrame) -> None:
    print(f"\n{'─'*70}")
    print("  F. Band-width percentile vs. outcome (relative to own distribution)")
    print(f"{'─'*70}")
    bw = df["bw_pips"].dropna()
    if len(bw) < 10:
        print("  insufficient band_width data")
        return
    q25, q50, q75 = bw.quantile([0.25, 0.50, 0.75])
    print(f"  Band-width quartiles:  Q1={q25:.1f}  Q2={q50:.1f}  Q3={q75:.1f}")

    def bw_bucket(v):
        if v is None or v != v:
            return "unknown"
        if v <= q25:
            return "tight(<Q1)"
        if v <= q50:
            return "mid(Q1-Q2)"
        if v <= q75:
            return "wide(Q2-Q3)"
        return "very_wide(>Q3)"

    df = df.copy()
    df["bw_bucket"] = df["bw_pips"].apply(bw_bucket)
    crosstab(df, "direction", "bw_bucket", min_n=5)


# ── post-loss effect ──────────────────────────────────────────────────────────

def post_loss_analysis(df: pd.DataFrame) -> None:
    print(f"\n{'─'*70}")
    print("  H. Post-loss effect in same session window")
    print(f"{'─'*70}")
    df = df.copy().sort_values("ts").reset_index(drop=True)
    df["prev_loss"] = (df["is_win"].shift(1) == False) & (df["session"].shift(1) == df["session"])
    for state in [False, True]:
        sub = df[df["prev_loss"] == state]
        n = len(sub)
        if n < 5:
            continue
        wins = int(sub["is_win"].sum())
        avg_p = sub["pips"].mean()
        pos = sub.loc[sub["pips"] > 0, "pips"].sum()
        neg = (-sub.loc[sub["pips"] < 0, "pips"]).sum()
        pf = pf_str(pos, neg)
        label = "after loss  " if state else "fresh entry "
        print(f"  {label}  n={n:>4}  WR={wins/n*100:>5.1f}%  avg={avg_p:>6.1f}  PF={pf}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    import psycopg2

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerun", action="store_true",
                        help="Skip backtests and use most recent DB data for bb_mult=1.5")
    parser.add_argument("--days", type=int, default=DAYS)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  RANGE_FADE AUDJPY {args.days}d — comprehensive entry diagnostic")
    print(f"{'='*70}\n")

    sweep_results = []

    if not args.no_rerun:
        print(f"Running three near-permissive backtests (bb_mult = 1.5, 1.8, 2.0) on {PERMISSIVE_DAYS}d …\n")
        for mult in [1.5, 1.8, 2.0]:
            res = run_permissive_bt(mult)
            sweep_results.append(res)
            if not res["ok"]:
                print(f"  ⚠  bb_mult={mult} backtest FAILED")
    else:
        print("[--no-rerun] Skipping backtests, using most recent DB signals.")

    # ── Section I: bb_mult sweep summary ─────────────────────────────────────
    if sweep_results:
        print(f"\n{'─'*70}")
        print("  I. bb_mult sweep summary (permissive baseline)")
        print(f"{'─'*70}")
        print(f"  {'bb_mult':>8}  {'n':>5}  {'WR%':>6}  {'PF':>6}  {'exp':>7}")
        best = sorted(sweep_results, key=lambda r: r["pf"], reverse=True)
        for r in best:
            mark = " ◀ best PF" if r is best[0] else ""
            print(f"  {r['bb_mult']:>8.1f}  {r['n']:>5}  {r['wr']:>6.1f}  "
                  f"{r['pf']:>6.2f}  {r['exp']:>7.2f}{mark}")

    # ── Fetch signals (most recent execution after rerun) ─────────────────────
    print("\nFetching signals from DB for multi-dim analysis …", flush=True)
    conn = psycopg2.connect(host="postgres", dbname="forex",
                            user="postgres", password="postgres")
    try:
        df = fetch_signals_with_adx(conn, EPIC)
    finally:
        conn.close()

    if df is None or len(df) == 0:
        print("❌ No signals to analyse.")
        return 1

    n_total = len(df)
    n_wins = int(df["is_win"].sum())
    wr_all = n_wins / n_total * 100
    pos_all = df.loc[df["pips"] > 0, "pips"].sum()
    neg_all = (-df.loc[df["pips"] < 0, "pips"]).sum()
    print(f"\n  Total signals: {n_total}  Winners: {n_wins}  "
          f"WR: {wr_all:.1f}%  PF: {pf_str(pos_all, neg_all)}")

    # ── A. Direction × HTF bias ───────────────────────────────────────────────
    crosstab(df, "direction", "htf", min_n=5,
             title="A. Direction × HTF bias  (★ PRIMARY discriminator)")

    # ── B. Direction × Session ────────────────────────────────────────────────
    crosstab(df, "direction", "session", min_n=5,
             title="B. Direction × Session bucket")

    # ── C. Direction × 1h ADX bucket ─────────────────────────────────────────
    crosstab(df, "direction", "adx_bucket", min_n=5,
             title="C. Direction × 1h ADX bucket")

    # ── D. Direction × HTF bias × Session ────────────────────────────────────
    triple_tab(df, "direction", "htf", "session", min_n=5,
               title="D. Direction × HTF bias × Session  (3-way, n≥5)")

    # ── E. RSI at entry by direction ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  E. RSI-at-entry bucket by direction")
    print(f"{'─'*70}")
    for d in ["BUY", "SELL"]:
        sub = df[df["direction"] == d].copy()
        print(f"\n  --- {d} ---")
        print(f"  {'rsi_bucket':>12}  {'n':>4}  {'WR%':>5}  {'avg_pips':>8}  PF")
        for bucket in sorted(sub["rsi_bucket"].unique()):
            g = sub[sub["rsi_bucket"] == bucket]
            if len(g) < 3:
                continue
            n = len(g)
            wins = int(g["is_win"].sum())
            avg_p = g["pips"].mean()
            pos = g.loc[g["pips"] > 0, "pips"].sum()
            neg = (-g.loc[g["pips"] < 0, "pips"]).sum()
            pf = pf_str(pos, neg)
            print(f"  {bucket:>12}  {n:>4}  {wins/n*100:>5.1f}  {avg_p:>8.1f}  {pf}")

    # ── F. Band-width relative ────────────────────────────────────────────────
    bw_relative_analysis(df)

    # ── G. Day-of-week × direction ────────────────────────────────────────────
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["dow_name"] = df["dow"].apply(lambda d: dow_names[d])
    crosstab(df, "direction", "dow_name", min_n=5,
             title="G. Day-of-week × direction")

    # ── H. Post-loss effect ───────────────────────────────────────────────────
    post_loss_analysis(df)

    # ── Summary: best single-gate candidates ─────────────────────────────────
    print(f"\n{'='*70}")
    print("  CANDIDATE GATES — cells with WR≥60% and n≥10 (marked ◀ above)")
    print("  Verify each via bt.py with JSONB override on AUDJPY demo row")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
