"""
Chandelier-trail PF recompute — HONEST (right-censored rows excluded).

The parent swing_horizon_test.py reports a headline chandelier PF that INCLUDES
right-censored ("truncated") signals — positions that never hit their trailing
stop and ran out of forward data, scored mark-to-market at the data edge by P&L
sign. That biases PF (an unresolved open position is counted as a realised
win/loss). zlma_trend's headline 1.35 carries ~12.5% such rows.

This script recomputes the chandelier PF two ways per scanner:
  (a) ALL rows         — reproduces the parent headline
  (b) EX-TRUNCATED     — only signals that actually resolved (trail fired or full
                         30-trading-day horizon elapsed). This is the honest PF.

Plus a temporal-stability split (older half vs newer half of EX-TRUNCATED
signals) — the chandelier has no fitted params, so this is a regime-stability
check, not a fit/eval split.

Read-only. Reuses the validated walk from swing_horizon_test.

Usage (inside stock-scanner container):
    python -m stock_scanner.analysis.trail_clean_recompute
"""
from __future__ import annotations

import warnings

import pandas as pd

from stock_scanner.analysis.swing_horizon_test import (
    get_conn,
    fetch_df,
    walk_daily_bars_chandelier,
    TRAIL_SL_PCT,
    TRAIL_ENGAGE_PCT,
    SWING_HORIZON_DAYS,
)

warnings.filterwarnings("ignore", category=FutureWarning)

MIN_N = 100  # only report scanners with at least this many signals


def pf_block(pnl: pd.Series, outcome: pd.Series) -> dict:
    wins = outcome == "win"
    losses = outcome == "loss"
    n_w = int(wins.sum())
    n_l = int(losses.sum())
    gw = float(pnl[wins].sum()) if n_w else 0.0
    gl = abs(float(pnl[losses].sum())) if n_l else 0.0
    pf = (gw / gl) if gl > 0 else (9.99 if gw > 0 else None)
    resolved = n_w + n_l
    wr = (n_w / resolved * 100) if resolved else None
    return {
        "n": len(pnl),
        "wr": wr,
        "pf": pf,
        "avg_w": float(pnl[wins].mean()) if n_w else None,
        "avg_l": float(pnl[losses].mean()) if n_l else None,
        "n_w": n_w,
        "n_l": n_l,
    }


def fmt(d: dict) -> str:
    pf = "  N/A" if d["pf"] is None else (">9.9 " if d["pf"] >= 9.99 else f"{d['pf']:5.2f}")
    wr = " N/A " if d["wr"] is None else f"{d['wr']:5.1f}"
    aw = "  N/A" if d["avg_w"] is None else f"{d['avg_w']:5.1f}"
    al = "  N/A" if d["avg_l"] is None else f"{d['avg_l']:5.1f}"
    return f"n={d['n']:5d}  WR={wr}  PF={pf}  avgW={aw}%  avgL={al}%"


def main():
    conn = get_conn()

    daily = fetch_df(conn, """
        SELECT ticker, timestamp, open, high, low, close
        FROM stock_candles_synthesized WHERE timeframe='1d'
        ORDER BY ticker, timestamp
    """)
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])
    if daily["timestamp"].dt.tz is not None:
        daily["timestamp"] = daily["timestamp"].dt.tz_localize(None)
    by_ticker = {str(t): g.reset_index(drop=True) for t, g in daily.groupby("ticker")}

    sigs = fetch_df(conn, """
        SELECT signal_timestamp, scanner_name, ticker, entry_price
        FROM stock_scanner_signals
        WHERE signal_type='BUY'
        ORDER BY signal_timestamp
    """)
    sigs["signal_timestamp"] = pd.to_datetime(sigs["signal_timestamp"])
    if sigs["signal_timestamp"].dt.tz is not None:
        sigs["signal_timestamp"] = sigs["signal_timestamp"].dt.tz_localize(None)

    rows = []
    for _, s in sigs.iterrows():
        base = str(s["ticker"]).split(".")[0]
        bars = by_ticker.get(base)
        if bars is None or bars.empty:
            continue
        mask = bars["timestamp"] > pd.Timestamp(s["signal_timestamp"])
        if not mask.any():
            continue
        sub = bars.iloc[bars[mask].index[0]:].reset_index(drop=True)
        entry = float(sub.iloc[0]["open"])
        if entry <= 0:
            continue
        r = walk_daily_bars_chandelier(
            entry, TRAIL_SL_PCT, TRAIL_SL_PCT, TRAIL_ENGAGE_PCT, sub, SWING_HORIZON_DAYS
        )
        if r["outcome"] == "no_data":
            continue
        rows.append({
            "scanner": s["scanner_name"],
            "ts": s["signal_timestamp"],
            "pnl": r["pnl_pct"],
            "outcome": r["outcome"],
            "truncated": r["truncated"],
        })

    df = pd.DataFrame(rows)
    print("=" * 92)
    print("CHANDELIER-TRAIL PF — HONEST RECOMPUTE (right-censored rows excluded)")
    print(f"  Trail: {TRAIL_SL_PCT:.0f}% initial SL, {TRAIL_SL_PCT:.0f}% trail below peak once +{TRAIL_ENGAGE_PCT:.0f}% reached")
    print(f"  Horizon: {SWING_HORIZON_DAYS} trading days. Total walked signals: {len(df):,}")
    print("=" * 92)
    print()

    counts = df["scanner"].value_counts()
    scanners = [s for s in counts.index if counts[s] >= MIN_N]
    # zlma first
    scanners = sorted(scanners, key=lambda s: (s != "zlma_trend", -counts[s]))

    print(f"  {'scanner':28s}  {'view':16s}  {'metrics'}")
    print(f"  {'-'*28}  {'-'*16}  {'-'*52}")
    for scn in scanners:
        sub = df[df["scanner"] == scn]
        clean = sub[~sub["truncated"]]
        trunc_pct = (sub["truncated"].mean() * 100) if len(sub) else 0
        all_m = pf_block(sub["pnl"], sub["outcome"])
        cln_m = pf_block(clean["pnl"], clean["outcome"])

        print(f"  {scn:28s}  {'ALL rows':16s}  {fmt(all_m)}   [trunc {trunc_pct:.0f}%]")
        print(f"  {'':28s}  {'EX-TRUNCATED':16s}  {fmt(cln_m)}   <-- honest")

        # temporal stability on EX-TRUNCATED
        if len(clean) >= 40:
            clean_sorted = clean.sort_values("ts")
            mid = len(clean_sorted) // 2
            early = clean_sorted.iloc[:mid]
            late = clean_sorted.iloc[mid:]
            em = pf_block(early["pnl"], early["outcome"])
            lm = pf_block(late["pnl"], late["outcome"])
            ep = "N/A" if em["pf"] is None else f"{em['pf']:.2f}"
            lp = "N/A" if lm["pf"] is None else f"{lm['pf']:.2f}"
            split_ts = clean_sorted.iloc[mid]["ts"]
            print(f"  {'':28s}  {'  temporal':16s}  early PF={ep} (n={em['n']})  |  late PF={lp} (n={lm['n']})  [split {split_ts.date()}]")
        print()

    conn.close()


if __name__ == "__main__":
    main()
