"""
Hold-Horizon Edge Simulator (1/2/3/5/10 trading days)
=====================================================
Re-simulates every BUY signal in stock_scanner_signals under the auto-trader
bracket model (same validated engine as daytrade_edge_sim) but reports
outcomes at multiple max-hold horizons. Motivation: real executions hold
~2-3 days, so neither the day-trade sim (flat same-day) nor the unlimited
swing sim (hold to TP2) matches actual trading.

Usage (inside stock-scanner container):
    python -m stock_scanner.analysis.hold_horizon_sim

Read-only. Walks each signal ONCE at the max horizon, then derives shorter
horizons: if the bracket exit happened at bar k <= h*6, the outcome is
identical for horizon h; otherwise the trade exits at the close of bar h*6
(time-stop). BE state does not affect a time-stop exit price, so this
derivation is exact.
"""

from __future__ import annotations

import sys
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

HORIZONS_DAYS = [1, 2, 3, 5, 10]
MAX_HORIZON = max(HORIZONS_DAYS)
BARS_PER_DAY = 6  # 1h bars per trading session


def derive_horizon(full: dict, bars: pd.DataFrame, entry: float, horizon_days: int) -> dict:
    """Derive the outcome at a shorter horizon from the max-horizon walk."""
    n_bars = horizon_days * BARS_PER_DAY
    exited_in_time = (
        full["outcome"] in ("win", "loss", "be")
        and full.get("bars_walked", 0) <= n_bars
    )
    if exited_in_time:
        return {"outcome": full["outcome"], "pnl_pct": full["pnl_pct"]}
    # Time-stop: exit at close of bar n_bars (or last available bar)
    idx = min(n_bars, len(bars)) - 1
    if idx < 0:
        return {"outcome": "no_data", "pnl_pct": None}
    close_price = float(bars.iloc[idx]["close"])
    return {"outcome": "timeout", "pnl_pct": (close_price - entry) / entry * 100}


def pf_of(grp: pd.DataFrame) -> float | None:
    """PF over ALL exits (wins, losses, BE, time-stops) — time-stop P&L is real P&L."""
    pnl = grp["pnl_pct"].dropna().astype(float)
    gross_win = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    if gross_loss > 0:
        return gross_win / gross_loss
    return 9.99 if gross_win > 0 else None


def main():
    print("=" * 78)
    print("HOLD-HORIZON EDGE SIMULATOR (auto-trader bracket, time-stop at N days)")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Horizons: {HORIZONS_DAYS} trading days; bracket 3%SL/5%TP (+ATR branch), BE@$10")
    print("=" * 78)

    conn = get_conn()
    try:
        print("\nPre-loading candles...")
        candles_by_ticker = build_candle_index(conn)
        daily_by_ticker = build_daily_index(conn)
        print(f"  1h tickers: {len(candles_by_ticker)}, daily tickers: {len(daily_by_ticker)}")

        signals = fetch_df(conn, """
            SELECT id, signal_timestamp, scanner_name, ticker, signal_type, signal_date
            FROM stock_scanner_signals
            WHERE signal_type = 'BUY'
            ORDER BY signal_timestamp
        """)
        print(f"  BUY signals: {len(signals):,}")

        # Walk every signal once at the max horizon
        base.TIMEOUT_TRADING_DAYS = MAX_HORIZON
        rows = []
        no_data = 0
        for i, sig in signals.iterrows():
            base_ticker = str(sig["ticker"]).split(".")[0]
            sig_ts = sig["signal_timestamp"]
            bars, entry = bars_after(candles_by_ticker, base_ticker, sig_ts)
            if bars.empty or entry <= 0:
                no_data += 1
                continue
            atr_pct = atr_pct_before(daily_by_ticker, base_ticker, sig_ts)

            fixed_bracket = build_bracket(entry, None)
            atr_bracket = build_bracket(entry, atr_pct)

            full_fixed = walk_bars(entry, fixed_bracket, bars)
            full_fixed_opt = walk_bars(entry, fixed_bracket, bars, optimistic_be=True)
            full_atr = walk_bars(entry, atr_bracket, bars)

            for h in HORIZONS_DAYS:
                for model, full in (
                    ("fixed_pess", full_fixed),
                    ("fixed_opt", full_fixed_opt),
                    ("atr_cond", full_atr),
                ):
                    d = derive_horizon(full, bars, entry, h)
                    rows.append({
                        "scanner_name": sig["scanner_name"],
                        "horizon": h,
                        "model": model,
                        "outcome": d["outcome"],
                        "pnl_pct": d["pnl_pct"],
                    })
            if (i + 1) % 1000 == 0:
                print(f"  ...{i + 1:,} signals walked")

        print(f"  Signals with no candle data skipped: {no_data}")
        res = pd.DataFrame(rows)
        res = res[res["outcome"] != "no_data"]

        # ------------------------------------------------------------------
        # Per-scanner x horizon table (fixed_pess primary; opt shown as band)
        # ------------------------------------------------------------------
        for model_label, models in (
            ("FIXED 3/5 BRACKET — PF band [pessimistic-BE .. optimistic-BE]",
             ("fixed_pess", "fixed_opt")),
            ("ATR-CONDITIONAL BRACKET (pessimistic BE)", ("atr_cond",)),
        ):
            print("\n" + "=" * 78)
            print(model_label)
            print("=" * 78)
            header = f"  {'scanner':30s} {'n':>5s}"
            for h in HORIZONS_DAYS:
                header += f" {f'{h}d':>13s}"
            print(header)
            print(f"  {'':30s} {'':>5s}" + "".join(
                f" {'PF (WR%)':>13s}" for _ in HORIZONS_DAYS))

            prim = models[0]
            sub = res[res["model"] == prim]
            for scanner, grp_all in sub.groupby("scanner_name"):
                n = len(grp_all[grp_all["horizon"] == HORIZONS_DAYS[0]])
                line = f"  {scanner:30s} {n:>5d}"
                for h in HORIZONS_DAYS:
                    grp = grp_all[grp_all["horizon"] == h]
                    pf = pf_of(grp)
                    if len(models) == 2:
                        grp_o = res[(res["model"] == models[1]) &
                                    (res["scanner_name"] == scanner) &
                                    (res["horizon"] == h)]
                        pf_o = pf_of(grp_o)
                    else:
                        pf_o = None
                    pnl = grp["pnl_pct"].dropna().astype(float)
                    wr = (pnl > 0).mean() * 100 if len(pnl) else 0
                    if pf is None:
                        cell = "N/A"
                    elif pf_o is not None and abs(pf_o - pf) >= 0.05:
                        cell = f"{pf:.2f}-{pf_o:.2f}"
                    else:
                        cell = f"{pf:.2f}({wr:.0f}%)"
                    line += f" {cell:>13s}"
                print(line)

        # ------------------------------------------------------------------
        # Detail at the 2d and 3d horizons (the real trading style)
        # ------------------------------------------------------------------
        print("\n" + "=" * 78)
        print("DETAIL: 2-3 DAY HORIZON (fixed bracket, pessimistic BE)")
        print("=" * 78)
        for h in (2, 3):
            print(f"\n--- {h} trading days ---")
            print(f"  {'scanner':30s} {'n':>5s} {'PF':>6s} {'WR%':>6s} "
                  f"{'avg%':>7s} {'win':>5s} {'loss':>5s} {'be':>5s} {'tstop':>6s} {'ts_avg%':>8s}")
            sub = res[(res["model"] == "fixed_pess") & (res["horizon"] == h)]
            stats = []
            for scanner, grp in sub.groupby("scanner_name"):
                pnl = grp["pnl_pct"].dropna().astype(float)
                ts = grp[grp["outcome"] == "timeout"]["pnl_pct"].dropna().astype(float)
                pf = pf_of(grp)
                stats.append({
                    "scanner": scanner, "n": len(grp), "pf": pf,
                    "wr": (pnl > 0).mean() * 100 if len(pnl) else 0,
                    "avg": pnl.mean() if len(pnl) else 0,
                    "win": (grp["outcome"] == "win").sum(),
                    "loss": (grp["outcome"] == "loss").sum(),
                    "be": (grp["outcome"] == "be").sum(),
                    "tstop": len(ts), "ts_avg": ts.mean() if len(ts) else 0,
                })
            for s in sorted(stats, key=lambda x: -(x["pf"] or 0)):
                pf_s = f"{s['pf']:.2f}" if s["pf"] is not None else "N/A"
                print(f"  {s['scanner']:30s} {s['n']:>5d} {pf_s:>6s} {s['wr']:>6.1f} "
                      f"{s['avg']:>7.2f} {s['win']:>5d} {s['loss']:>5d} {s['be']:>5d} "
                      f"{s['tstop']:>6d} {s['ts_avg']:>8.2f}")

        print("\nCaveats: all-signals population (not live-gated); in-sample; "
              "May-26 launches have <=2wk single-regime data; n<100 exploratory only.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
