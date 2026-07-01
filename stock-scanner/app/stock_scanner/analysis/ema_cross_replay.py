"""
EMA 9/21/50 Cross — Walk-Forward Backtest
=========================================
Validates the new `ema_cross_9_21_50` scanner under the REAL stock auto-trader
execution model and reports per-period PF / win-rate / expectancy.

WHY a custom replay (not bt_stocks.py): the live auto-trader bracket is NOT the
backtest_orchestrator's default. This harness reuses the *validated* path-walking
engine (`walk_bars` / `bars_after`) from daytrade_edge_sim.py but feeds it the
CURRENT live parameters read from the auto-trader settings (DB is master):

    SL 2.5%   TP 7.0%   ATR-stop OFF   breakeven OFF   notional $500
    entry = OPEN of first 1h bar STRICTLY AFTER the signal bar's close
            (daily close stamps at 20:00 UTC -> next-session 13:30 UTC open;
             no look-ahead)
    timeout = 10 trading days -> exit at close

Entry signals come from the SAME strategy class the live scanner runs
(EmaCross92150Strategy.detect_signals), so this measures the scanner's own edge.

Three views are reported:
  (1) ALL signals (raw scanner edge)
  (2) + EMA50 fresh-cross gate (route.ts live gate: above EMA50 AND crossed up
      within AUTO_TRADE_EMA50_MAX_CROSS_AGE_SESSIONS=10 sessions) — the only
      live candidate-pool gate that is reconstructable from daily candles.
  Each split into TRAIN (< 2026-04-01) / OOS (>= 2026-04-01).

CAVEATS (shared with the other replay harnesses): the live VWAP / positive-day /
spread / score>=65 / intraday-RVOL gates are LIVE-ONLY and not reconstructable,
so this is the ALL-SIGNALS ranking edge, not the realised live-traded P&L.
In-sample window is ~Dec-2025 -> Jun-2026 (single-ish regime) — read positives
cautiously. Read-only: modifies no table.

Usage (inside the stock-scanner container):
    docker exec stock-scanner python -m stock_scanner.analysis.ema_cross_replay
    docker exec stock-scanner python -m stock_scanner.analysis.ema_cross_replay --max-tickers 500
"""

from __future__ import annotations

import sys
import argparse
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import (
    get_conn, fetch_df, walk_bars, bars_after,
)
from stock_scanner.strategies.ema_cross_9_21_50 import EmaCross92150Strategy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- LIVE auto-trader params (stock_auto_trade_settings, read 2026-06-30) ---
SL_PCT = 2.5
TP_PCT = 7.0
NOTIONAL = 500.0
TIMEOUT_TRADING_DAYS = 10
EMA50_MAX_CROSS_AGE = 10   # AUTO_TRADE_EMA50_MAX_CROSS_AGE_SESSIONS

TRAIN_END = pd.Timestamp("2026-04-01")  # OOS = on/after this date


def load_settings_overrides(conn) -> None:
    """Pull the live bracket from the DB so the sim tracks production if it drifts."""
    global SL_PCT, TP_PCT, NOTIONAL
    df = fetch_df(conn, """
        SELECT key, value FROM stock_auto_trade_settings
        WHERE key IN ('AUTO_TRADE_STOP_LOSS_PCT','AUTO_TRADE_TAKE_PROFIT_PCT',
                      'MAX_ORDER_NOTIONAL_USD','AUTO_TRADE_ATR_STOP_ENABLED')
    """)
    if df.empty:
        return
    kv = {r["key"]: r["value"] for _, r in df.iterrows()}
    SL_PCT = float(kv.get("AUTO_TRADE_STOP_LOSS_PCT", SL_PCT))
    TP_PCT = float(kv.get("AUTO_TRADE_TAKE_PROFIT_PCT", TP_PCT))
    NOTIONAL = float(kv.get("MAX_ORDER_NOTIONAL_USD", NOTIONAL))
    atr_on = str(kv.get("AUTO_TRADE_ATR_STOP_ENABLED", "false")).lower() == "true"
    if atr_on:
        print("WARNING: live ATR stop is ENABLED but this replay models the "
              "fixed bracket only. Results understate ATR-branch trades.")


def ema50_cross_age(df: pd.DataFrame) -> pd.Series:
    """Bars since close last crossed UP through ema_50 (0 = crossed on this bar)."""
    close = pd.to_numeric(df["close"], errors="coerce")
    cross_up = (close > df["ema_50"]) & (close.shift(1) <= df["ema_50"].shift(1))
    idx = pd.Series(np.where(cross_up.values, np.arange(len(df)), np.nan), index=df.index)
    last_cross_pos = idx.ffill()
    age = pd.Series(np.arange(len(df)), index=df.index) - last_cross_pos
    return age  # NaN until the first cross


def top_liquid_tickers(daily: pd.DataFrame, n: int) -> set:
    """Return the n tickers with the highest median daily dollar-volume."""
    d = daily.copy()
    d["dollar_vol"] = pd.to_numeric(d["close"], errors="coerce") * pd.to_numeric(
        d["volume"], errors="coerce")
    med = d.groupby("ticker")["dollar_vol"].median().sort_values(ascending=False)
    return set(med.head(n).index.tolist())


def collect_signals(conn, max_tickers: Optional[int],
                    top_liquid: Optional[int]) -> pd.DataFrame:
    """Run the strategy across all daily history; return one row per signal."""
    daily = fetch_df(conn, """
        SELECT ticker, timestamp, open, high, low, close, volume
        FROM stock_candles_synthesized
        WHERE timeframe = '1d'
        ORDER BY ticker, timestamp
    """)
    if daily.empty:
        return pd.DataFrame()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])

    liquid_set = top_liquid_tickers(daily, top_liquid) if top_liquid else None
    if liquid_set is not None:
        print(f"  liquidity filter: keeping top {top_liquid} tickers by median $-volume")

    strat = EmaCross92150Strategy(stop_loss_pct=SL_PCT, take_profit_pct=TP_PCT)
    rows = []
    groups = list(daily.groupby("ticker", sort=True))
    if liquid_set is not None:
        groups = [(t, g) for t, g in groups if t in liquid_set]
    if max_tickers:
        groups = groups[:max_tickers]

    for tkr, g in groups:
        g = g.reset_index(drop=True)
        if len(g) < 60:
            continue
        g = strat.prepare(g)
        sig = strat.detect_signals(g)
        if not sig.any():
            continue
        g["_age50"] = ema50_cross_age(g)
        hits = g[sig]
        for _, r in hits.iterrows():
            rows.append({
                "ticker": tkr,
                "signal_ts": r["timestamp"],
                "close": float(r["close"]),
                "age50": float(r["_age50"]) if pd.notna(r["_age50"]) else np.nan,
            })
    return pd.DataFrame(rows)


def simulate(conn, signals: pd.DataFrame) -> pd.DataFrame:
    """Attach a simulated outcome (pnl_pct, pnl_usd, outcome) to every signal."""
    if signals.empty:
        return signals
    tickers = sorted(signals["ticker"].unique().tolist())
    # one query for all 1h candles of the signalled tickers
    candles = fetch_df(conn, """
        SELECT ticker, timestamp, open, high, low, close
        FROM stock_candles
        WHERE timeframe = '1h' AND ticker = ANY(%s)
        ORDER BY ticker, timestamp
    """, (tickers,))
    candles["timestamp"] = pd.to_datetime(candles["timestamp"])
    by_ticker = {str(t): g.reset_index(drop=True) for t, g in candles.groupby("ticker")}

    out = []
    for _, s in signals.iterrows():
        bars, entry = bars_after(by_ticker, s["ticker"], s["signal_ts"])
        if bars.empty or entry <= 0:
            continue
        sl = entry * (1 - SL_PCT / 100)
        tp = entry * (1 + TP_PCT / 100)
        bracket = {"sl": sl, "tp": tp, "be_trigger_pct": 0.0}
        res = walk_bars(entry, bracket, bars, be_enabled=False)
        if res["outcome"] == "no_data":
            continue
        pnl_pct = res["pnl_pct"]
        out.append({
            **s.to_dict(),
            "entry": entry,
            "outcome": res["outcome"],
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_pct / 100.0 * NOTIONAL,
            "hold_bars": res["hold_bars"],
        })
    return pd.DataFrame(out)


def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    wins = df[df["pnl_usd"] > 0]["pnl_usd"].sum()
    losses = -df[df["pnl_usd"] < 0]["pnl_usd"].sum()
    pf = (wins / losses) if losses > 0 else float("inf")
    wr = (df["pnl_usd"] > 0).mean() * 100
    return {
        "n": len(df),
        "pf": pf,
        "wr": wr,
        "avg_pnl_usd": df["pnl_usd"].mean(),
        "avg_pnl_pct": df["pnl_pct"].mean(),
        "total_usd": df["pnl_usd"].sum(),
        "avg_win_pct": df[df["pnl_pct"] > 0]["pnl_pct"].mean(),
        "avg_loss_pct": df[df["pnl_pct"] < 0]["pnl_pct"].mean(),
        "tp_rate": (df["outcome"] == "win").mean() * 100,
        "sl_rate": (df["outcome"] == "loss").mean() * 100,
        "to_rate": (df["outcome"] == "timeout").mean() * 100,
    }


def fmt(label: str, m: dict) -> str:
    if m["n"] == 0:
        return f"  {label:<28} n=0"
    pf = m["pf"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (
        f"  {label:<28} n={m['n']:<5} PF={pf_s:<5} WR={m['wr']:.1f}%  "
        f"exp=${m['avg_pnl_usd']:+.2f}/trade  total=${m['total_usd']:+.0f}  "
        f"[TP {m['tp_rate']:.0f}% / SL {m['sl_rate']:.0f}% / TO {m['to_rate']:.0f}%]"
    )


def report(sims: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("EMA 9/21/50 CROSS — walk-forward backtest under live auto-trader bracket")
    print(f"  bracket: SL {SL_PCT}%  TP {TP_PCT}%  ATR-off  BE-off  "
          f"notional ${NOTIONAL:.0f}  timeout {TIMEOUT_TRADING_DAYS}d")
    print("=" * 78)

    if sims.empty:
        print("No simulated trades (no signals had forward 1h candles).")
        return

    sims = sims.copy()
    sims["oos"] = sims["signal_ts"] >= TRAIN_END

    def block(title, sub):
        print(f"\n{title}")
        print(fmt("ALL", metrics(sub)))
        print(fmt("TRAIN (<2026-04-01)", metrics(sub[~sub["oos"]])))
        print(fmt("OOS  (>=2026-04-01)", metrics(sub[sub["oos"]])))

    block("[1] ALL SIGNALS (raw scanner edge)", sims)

    gated = sims[(sims["age50"].notna()) & (sims["age50"] <= EMA50_MAX_CROSS_AGE)]
    block(f"[2] + EMA50 fresh-cross gate (<= {EMA50_MAX_CROSS_AGE} sessions)", gated)

    # monthly stability (ALL signals)
    print("\n[3] Monthly (ALL signals):")
    sims["month"] = sims["signal_ts"].dt.to_period("M").astype(str)
    for mth, grp in sims.groupby("month"):
        print(fmt(mth, metrics(grp)))

    print("\nPF>1.0 = positive expectancy. Verdict should lean on the OOS rows and "
          "the fresh-cross-gated view (closest to the live traded set).")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=None,
                    help="cap universe (alphabetical) for a quick run")
    ap.add_argument("--top-liquid", type=int, default=None,
                    help="restrict to the N most liquid tickers by median $-volume")
    args = ap.parse_args()

    conn = get_conn()
    try:
        load_settings_overrides(conn)
        print(f"Collecting signals (top_liquid={args.top_liquid or '-'}, "
              f"max_tickers={args.max_tickers or 'all'}) ...")
        signals = collect_signals(conn, args.max_tickers, args.top_liquid)
        print(f"  {len(signals)} raw signals across "
              f"{signals['ticker'].nunique() if not signals.empty else 0} tickers")
        if signals.empty:
            return
        print("Simulating under live bracket ...")
        sims = simulate(conn, signals)
        print(f"  {len(sims)} trades simulated")
        report(sims)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
