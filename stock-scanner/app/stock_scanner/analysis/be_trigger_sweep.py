"""
Breakeven-Trigger Sweep
=======================
Sweeps the breakeven trigger through the validated hold-horizon sim engine to
answer: what should breakeven_trigger_usd be (or should BE be %-of-TP, or off)?

Uses the LIVE bracket (ATR-conditional, build_bracket) and pessimistic BE
walking. For each config and horizon, reports per-scanner and portfolio
PF/WR/avg, plus an attribution vs the BE-OFF baseline:
  saved  = BE-OFF outcome was LOSS, this config exited BE (BE rescued a loser)
  cost   = BE-OFF outcome was WIN,  this config exited BE (BE killed a winner)

Usage (inside stock-scanner container):
    python -m stock_scanner.analysis.be_trigger_sweep
Read-only.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from stock_scanner.analysis import daytrade_edge_sim as base
from stock_scanner.analysis.daytrade_edge_sim import (
    atr_pct_before,
    bars_after,
    build_bracket,
    build_candle_index,
    build_daily_index,
    fetch_df,
    get_conn,
    walk_bars,
)
from stock_scanner.analysis.hold_horizon_sim import derive_horizon

HORIZONS_DAYS = [2, 3, 10]
MAX_HORIZON = max(HORIZONS_DAYS)

# (label, kind, value): usd -> trigger = value/notional; tp_frac -> value*tp_pct; off
CONFIGS = [
    ("off", "off", None),
    ("usd10 (current)", "usd", 10.0),
    ("usd15", "usd", 15.0),
    ("usd20", "usd", 20.0),
    ("usd25", "usd", 25.0),
    ("usd30", "usd", 30.0),
    ("50%ofTP", "tp_frac", 0.50),
    ("70%ofTP", "tp_frac", 0.70),
    ("90%ofTP", "tp_frac", 0.90),
]


def main():
    print("=" * 100)
    print("BREAKEVEN-TRIGGER SWEEP (live ATR-conditional bracket, pessimistic BE)")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Horizons: {HORIZONS_DAYS} trading days; configs: {[c[0] for c in CONFIGS]}")
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

        base.TIMEOUT_TRADING_DAYS = MAX_HORIZON
        rows = []
        for i, sig in signals.iterrows():
            base_ticker = str(sig["ticker"]).split(".")[0]
            sig_ts = sig["signal_timestamp"]
            bars, entry = bars_after(candles_by_ticker, base_ticker, sig_ts)
            if bars.empty or entry <= 0:
                continue
            atr_pct = atr_pct_before(daily_by_ticker, base_ticker, sig_ts)
            bracket = build_bracket(entry, atr_pct)

            for label, kind, value in CONFIGS:
                b = dict(bracket)
                if kind == "off":
                    full = walk_bars(entry, b, bars, be_enabled=False)
                else:
                    if kind == "usd":
                        notional = b["notional"] or 500.0
                        b["be_trigger_pct"] = value / notional * 100
                    else:  # tp_frac
                        b["be_trigger_pct"] = value * b["tp_pct"]
                    full = walk_bars(entry, b, bars, be_enabled=True)
                for h in HORIZONS_DAYS:
                    d = derive_horizon(full, bars, entry, h)
                    rows.append({
                        "signal_id": sig["id"],
                        "scanner_name": sig["scanner_name"],
                        "config": label,
                        "horizon": h,
                        "outcome": d["outcome"],
                        "pnl_pct": d["pnl_pct"],
                    })
            if (i + 1) % 1000 == 0:
                print(f"  ...{i + 1:,} signals walked")

        res = pd.DataFrame(rows)
        res = res[res["outcome"] != "no_data"]

        def metrics(grp: pd.DataFrame) -> dict:
            pnl = grp["pnl_pct"].dropna().astype(float)
            gw = pnl[pnl > 0].sum()
            gl = abs(pnl[pnl < 0].sum())
            pf = (gw / gl) if gl > 0 else (9.99 if gw > 0 else None)
            return {
                "n": len(grp),
                "pf": pf,
                "wr": (pnl > 0).mean() * 100 if len(pnl) else 0,
                "avg": pnl.mean() if len(pnl) else 0,
                "n_be": int((grp["outcome"] == "be").sum()),
            }

        # Portfolio-level table per horizon, with saved/cost attribution vs off
        for h in HORIZONS_DAYS:
            sub_h = res[res["horizon"] == h]
            off = sub_h[sub_h["config"] == "off"].set_index("signal_id")
            print("\n" + "=" * 100)
            print(f"PORTFOLIO — {h} trading days (all scanners pooled)")
            print("=" * 100)
            print(f"  {'config':18s} {'n':>6s} {'PF':>6s} {'WR%':>6s} {'avg%':>7s} "
                  f"{'n_BE':>6s} {'saved':>6s} {'cost':>6s} {'net_pnl_delta%':>15s}")
            for label, _, _ in CONFIGS:
                grp = sub_h[sub_h["config"] == label]
                m = metrics(grp)
                if label == "off":
                    saved = cost = ""
                    delta = ""
                else:
                    g = grp.set_index("signal_id")
                    common = g.index.intersection(off.index)
                    go, oo = g.loc[common], off.loc[common]
                    be_mask = go["outcome"] == "be"
                    saved = int((be_mask & (oo["outcome"] == "loss")).sum())
                    cost = int((be_mask & (oo["outcome"] == "win")).sum())
                    delta = f"{(go['pnl_pct'].astype(float) - oo['pnl_pct'].astype(float)).sum():.0f}"
                pf_s = f"{m['pf']:.2f}" if m["pf"] is not None else "N/A"
                print(f"  {label:18s} {m['n']:>6d} {pf_s:>6s} {m['wr']:>6.1f} {m['avg']:>7.2f} "
                      f"{m['n_be']:>6d} {str(saved):>6s} {str(cost):>6s} {str(delta):>15s}")

        # Per-scanner best config at 2d and 3d (scanners with n>=100)
        for h in (2, 3):
            sub_h = res[res["horizon"] == h]
            print("\n" + "=" * 100)
            print(f"PER-SCANNER PF BY CONFIG — {h} trading days (n>=100 scanners)")
            print("=" * 100)
            labels = [c[0] for c in CONFIGS]
            hdr = f"  {'scanner':28s} {'n':>5s}" + "".join(f" {l[:9]:>9s}" for l in labels)
            print(hdr)
            for scanner, grp_s in sub_h.groupby("scanner_name"):
                n = len(grp_s[grp_s["config"] == "off"])
                if n < 100:
                    continue
                line = f"  {scanner:28s} {n:>5d}"
                for label in labels:
                    m = metrics(grp_s[grp_s["config"] == label])
                    line += f" {m['pf']:>9.2f}" if m["pf"] is not None else f" {'N/A':>9s}"
                print(line)

        print("\nCaveats: in-sample, all-signals population, pessimistic BE "
              "(same-bar arm+stop allowed -> BE configs are lower-bounded). "
              "May-26 scanners single-regime.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
