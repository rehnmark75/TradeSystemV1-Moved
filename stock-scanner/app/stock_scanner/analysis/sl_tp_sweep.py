"""
SL/TP Bracket Sweep — fixed % grid vs ATR-dynamic grid
======================================================
Sweeps stop-loss / take-profit brackets over all historical BUY signals using
the same entry convention as the validated sims (entry = open of first 1h bar
strictly after signal_timestamp; same-bar SL+TP -> SL-first conservative).
BE is OFF throughout (per Jun 11 BE-trigger sweep: BE is never additive),
which makes the walk closed-form per config.

Grids:
  FIXED:  SL% in {2,3,4,5,7} x TP% in {3,5,7,10,15}
  ATR:    SL = mult*ATR14%% (daily, no look-ahead), mult in {1.0,1.5,2.0,2.5},
          TP = rr*SL, rr in {1.0,1.5,2.0,3.0}
Horizons: 2/3/10 trading days (time-stop at close). H1/H2 time-split shown for
top portfolio cells (overfit guard).

Usage (inside stock-scanner container):
    python -m stock_scanner.analysis.sl_tp_sweep
Read-only.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import (
    atr_pct_before,
    bars_after,
    build_candle_index,
    build_daily_index,
    fetch_df,
    get_conn,
)

HORIZONS_DAYS = [2, 3, 10]
BARS_PER_DAY = 6
MAX_BARS = max(HORIZONS_DAYS) * BARS_PER_DAY

FIXED_SL = [2.0, 3.0, 4.0, 5.0, 7.0]
FIXED_TP = [3.0, 5.0, 7.0, 10.0, 15.0]
ATR_MULT = [1.0, 1.5, 2.0, 2.5]
ATR_RR = [1.0, 1.5, 2.0, 3.0]

PER_SCANNER_MIN_N = 100


def first_true(mask: np.ndarray):
    idx = np.argmax(mask)
    return int(idx) if mask[idx] else None


def walk(entry, highs, lows, closes, sl_pct, tp_pct):
    """Return (exit_idx or None, pnl_pct at bracket exit). SL-first on ties."""
    sl = entry * (1 - sl_pct / 100)
    tp = entry * (1 + tp_pct / 100)
    i_sl = first_true(lows <= sl)
    i_tp = first_true(highs >= tp)
    if i_sl is not None and (i_tp is None or i_sl <= i_tp):
        return i_sl, -sl_pct
    if i_tp is not None:
        return i_tp, tp_pct
    return None, None


def horizon_pnl(exit_idx, exit_pnl, entry, closes, h_bars):
    nb = min(h_bars, len(closes))
    if nb <= 0:
        return None
    if exit_idx is not None and exit_idx < nb:
        return exit_pnl
    return (float(closes[nb - 1]) - entry) / entry * 100


def metrics(pnls: np.ndarray) -> dict:
    pnls = pnls[~np.isnan(pnls)]
    gw = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls < 0].sum())
    pf = gw / gl if gl > 0 else (9.99 if gw > 0 else np.nan)
    return {"n": len(pnls), "pf": pf,
            "wr": (pnls > 0).mean() * 100 if len(pnls) else 0,
            "avg": pnls.mean() if len(pnls) else 0}


def main():
    print("=" * 100)
    print("SL/TP BRACKET SWEEP — fixed grid vs ATR-dynamic (BE OFF)")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 100)

    conn = get_conn()
    try:
        print("\nPre-loading candles...")
        candles_by_ticker = build_candle_index(conn)
        daily_by_ticker = build_daily_index(conn)
        signals = fetch_df(conn, """
            SELECT id, signal_timestamp, scanner_name, ticker
            FROM stock_scanner_signals
            WHERE signal_type = 'BUY'
            ORDER BY signal_timestamp
        """)
        print(f"  BUY signals: {len(signals):,}")

        # Precompute per-signal arrays
        sigs = []
        for _, sig in signals.iterrows():
            base_ticker = str(sig["ticker"]).split(".")[0]
            bars, entry = bars_after(candles_by_ticker, base_ticker, sig["signal_timestamp"])
            if bars.empty or entry <= 0:
                continue
            b = bars.iloc[:MAX_BARS]
            sigs.append({
                "scanner": sig["scanner_name"],
                "ts": sig["signal_timestamp"],
                "entry": entry,
                "highs": b["high"].astype(float).to_numpy(),
                "lows": b["low"].astype(float).to_numpy(),
                "closes": b["close"].astype(float).to_numpy(),
                "atr_pct": atr_pct_before(daily_by_ticker, base_ticker, sig["signal_timestamp"]),
            })
        print(f"  Valid signals: {len(sigs):,} "
              f"(with ATR: {sum(1 for s in sigs if s['atr_pct'] is not None):,})")
        ts_median = pd.Series([pd.Timestamp(str(s["ts"])).tz_localize(None) for s in sigs]).median()
        for s in sigs:
            s["half"] = "H1" if pd.Timestamp(str(s["ts"])).tz_localize(None) <= ts_median else "H2"
        print(f"  Time-split median: {ts_median} (H1 <= median < H2)")

        configs = ([("fixed", sl, tp) for sl in FIXED_SL for tp in FIXED_TP]
                   + [("atr", m, rr) for m in ATR_MULT for rr in ATR_RR])

        # rows: config_kind, p1, p2, horizon, scanner, half, pnl
        rows = []
        for kind, p1, p2 in configs:
            for s in sigs:
                if kind == "fixed":
                    sl_pct, tp_pct = p1, p2
                else:
                    if s["atr_pct"] is None:
                        continue
                    sl_pct = max(0.5, p1 * s["atr_pct"])
                    tp_pct = p2 * sl_pct
                exit_idx, exit_pnl = walk(s["entry"], s["highs"], s["lows"], s["closes"],
                                          sl_pct, tp_pct)
                for h in HORIZONS_DAYS:
                    pnl = horizon_pnl(exit_idx, exit_pnl, s["entry"], s["closes"],
                                      h * BARS_PER_DAY)
                    if pnl is None:
                        continue
                    rows.append((kind, p1, p2, h, s["scanner"], s["half"], pnl))
        res = pd.DataFrame(rows, columns=["kind", "p1", "p2", "horizon",
                                          "scanner", "half", "pnl"])
        print(f"  Simulated rows: {len(res):,}")

        def grid_table(sub: pd.DataFrame, kind: str, title: str):
            p1s = FIXED_SL if kind == "fixed" else ATR_MULT
            p2s = FIXED_TP if kind == "fixed" else ATR_RR
            p1_lab = "SL%" if kind == "fixed" else "ATRx"
            p2_lab = "TP%" if kind == "fixed" else "RR"
            print(f"\n  {title}  [rows={p1_lab}, cols={p2_lab}]  cell = PF (avg%)")
            print(f"    {'':>6s}" + "".join(f" {p2:>12.1f}" for p2 in p2s))
            for p1 in p1s:
                line = f"    {p1:>6.1f}"
                for p2 in p2s:
                    g = sub[(sub["p1"] == p1) & (sub["p2"] == p2)]
                    if g.empty:
                        line += f" {'N/A':>12s}"
                    else:
                        m = metrics(g["pnl"].to_numpy())
                        line += f" {m['pf']:5.2f}({m['avg']:+5.2f})"
                print(line)

        # Portfolio grids
        for h in HORIZONS_DAYS:
            print("\n" + "=" * 100)
            print(f"PORTFOLIO — {h} trading days")
            print("=" * 100)
            grid_table(res[(res["kind"] == "fixed") & (res["horizon"] == h)],
                       "fixed", "FIXED bracket")
            grid_table(res[(res["kind"] == "atr") & (res["horizon"] == h)],
                       "atr", "ATR-dynamic bracket")

        # Top portfolio cells with H1/H2 split (3d horizon)
        print("\n" + "=" * 100)
        print("ROBUSTNESS — top 8 portfolio cells at 3d, H1 vs H2 PF")
        print("=" * 100)
        sub3 = res[res["horizon"] == 3]
        cells = []
        for (kind, p1, p2), g in sub3.groupby(["kind", "p1", "p2"]):
            m = metrics(g["pnl"].to_numpy())
            m1 = metrics(g[g["half"] == "H1"]["pnl"].to_numpy())
            m2 = metrics(g[g["half"] == "H2"]["pnl"].to_numpy())
            cells.append({"kind": kind, "p1": p1, "p2": p2, **m,
                          "pf_h1": m1["pf"], "pf_h2": m2["pf"]})
        cells_df = pd.DataFrame(cells).sort_values("pf", ascending=False)
        print(f"  {'config':>22s} {'n':>6s} {'PF':>6s} {'avg%':>7s} {'WR%':>6s} "
              f"{'PF_H1':>7s} {'PF_H2':>7s}")
        for _, r in cells_df.head(8).iterrows():
            cfg = (f"fixed SL{r['p1']:.0f}/TP{r['p2']:.0f}" if r["kind"] == "fixed"
                   else f"atr {r['p1']:.1f}x RR{r['p2']:.1f}")
            print(f"  {cfg:>22s} {r['n']:>6d} {r['pf']:>6.2f} {r['avg']:>7.2f} "
                  f"{r['wr']:>6.1f} {r['pf_h1']:>7.2f} {r['pf_h2']:>7.2f}")
        # Current live cell for reference
        cur = cells_df[(cells_df["kind"] == "fixed") & (cells_df["p1"] == 3.0)
                       & (cells_df["p2"] == 5.0)]
        if not cur.empty:
            r = cur.iloc[0]
            print(f"  {'>> current SL3/TP5':>22s} {r['n']:>6d} {r['pf']:>6.2f} "
                  f"{r['avg']:>7.2f} {r['wr']:>6.1f} {r['pf_h1']:>7.2f} {r['pf_h2']:>7.2f}")

        # Per-scanner grids at 3d (n >= threshold)
        print("\n" + "=" * 100)
        print(f"PER-SCANNER — 3 trading days (scanners with n>={PER_SCANNER_MIN_N})")
        print("=" * 100)
        sub_f = sub3[sub3["kind"] == "fixed"]
        counts = sub_f[(sub_f["p1"] == FIXED_SL[0]) & (sub_f["p2"] == FIXED_TP[0])] \
            .groupby("scanner").size()
        for scanner in sorted(counts[counts >= PER_SCANNER_MIN_N].index):
            print(f"\n--- {scanner} (n={int(counts[scanner])}) ---")
            grid_table(sub_f[sub_f["scanner"] == scanner], "fixed", "FIXED")
            grid_table(sub3[(sub3["kind"] == "atr") & (sub3["scanner"] == scanner)],
                       "atr", "ATR-dynamic")

        print("\nCaveats: in-sample grid search — prefer PLATEAUS over single best "
              "cells; H1/H2 split is the overfit guard. All-signals population "
              "(not live-gated). No spread/slippage modeled. SL-first same-bar "
              "tiebreak (conservative). May-26 scanners single-regime.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
