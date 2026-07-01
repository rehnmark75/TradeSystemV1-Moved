"""
Smart Money Reclaim — Walk-Forward Backtest
===========================================
Validates the new `smart_money_reclaim` scanner under the REAL stock auto-trader
execution model and reports per-period PF / win-rate / expectancy.

Mirrors `ema_cross_replay.py`: it reuses the validated path-walking engine
(`walk_bars` / `bars_after`) from daytrade_edge_sim.py and feeds it the CURRENT
live bracket (DB is master):

    SL 2.5%   TP 7.0%   ATR-stop OFF   BE OFF   notional $500
    entry = OPEN of first 1h bar STRICTLY AFTER the signal bar's close
            (daily close stamps ~20:00 UTC -> next-session 13:30 UTC open; no
             look-ahead)
    timeout = 10 trading days

Entry signals come from the SAME strategy class the live scanner runs
(SmartMoneyReclaimStrategy.scan), evaluated WALK-FORWARD: for every ticker, on
each daily bar we feed the strategy only bars up to and including that bar, plus
the point-in-time screening metrics (relative_volume / rs_trend / rs_vs_spy AS-OF
that date) and the market_context breadth row AS-OF that date. So this measures
the scanner's own as-of-decision-time edge with no leakage.

Views reported:
  (1) ALL signals (raw scanner edge), split TRAIN (<2026-04-01) / OOS (>=).
  (2) Breadth split: PF on signals fired when pct_above_sma50 >= 50 vs < 50.
  (3) Random-entry baseline: same tickers/dates universe, random long entries.
  (4) Sweep-WITHOUT-reclaim control: bars that swept a swing low but whose close
      did NOT reclaim it (the null setup) — should show NO edge if the reclaim +
      MF-divergence is real.
  (5) Monthly stability (ALL signals).

CAVEATS (shared with the other replay harnesses): the live VWAP / positive-day /
spread / score>=65 / intraday-RVOL gates are LIVE-ONLY and not reconstructable,
so this is the ALL-SIGNALS ranking edge, not realised live-traded P&L. The
in-sample window is single-ish regime — read positives cautiously. Read-only:
modifies no table.

Usage (inside the stock-scanner container):
    docker exec stock-scanner python -m stock_scanner.analysis.smart_money_reclaim_replay
    docker exec stock-scanner python -m stock_scanner.analysis.smart_money_reclaim_replay --max-tickers 500
    docker exec stock-scanner python -m stock_scanner.analysis.smart_money_reclaim_replay --top-liquid 500
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
from stock_scanner.strategies.smart_money_reclaim import SmartMoneyReclaimStrategy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- LIVE auto-trader params (stock_auto_trade_settings, read 2026-06-30) ---
SL_PCT = 2.5
TP_PCT = 7.0
NOTIONAL = 500.0
TIMEOUT_TRADING_DAYS = 10

TRAIN_END = pd.Timestamp("2026-04-01")  # OOS = on/after this date
MIN_BARS = 110                          # strategy needs ~105+ bars of history
RANDOM_SEED = 7


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


def top_liquid_tickers(daily: pd.DataFrame, n: int) -> set:
    d = daily.copy()
    d["dollar_vol"] = pd.to_numeric(d["close"], errors="coerce") * pd.to_numeric(
        d["volume"], errors="coerce")
    med = d.groupby("ticker")["dollar_vol"].median().sort_values(ascending=False)
    return set(med.head(n).index.tolist())


def load_metrics_history(conn) -> dict:
    """Per-ticker screening-metrics history sorted by date, for as-of lookups."""
    m = fetch_df(conn, """
        SELECT ticker, calculation_date, relative_volume, rs_trend, rs_vs_spy
        FROM stock_screening_metrics
        ORDER BY ticker, calculation_date
    """)
    if m.empty:
        return {}
    m["calculation_date"] = pd.to_datetime(m["calculation_date"])
    return {str(t): g.reset_index(drop=True) for t, g in m.groupby("ticker")}


def load_market_history(conn) -> pd.DataFrame:
    """market_context breadth history sorted by date, for as-of lookups."""
    mk = fetch_df(conn, """
        SELECT calculation_date, market_regime, pct_above_sma50
        FROM market_context
        ORDER BY calculation_date
    """)
    if mk.empty:
        return mk
    mk["calculation_date"] = pd.to_datetime(mk["calculation_date"])
    return mk


def asof_metrics(metrics_hist: dict, ticker: str, bar_date: pd.Timestamp) -> Optional[dict]:
    g = metrics_hist.get(ticker)
    if g is None or g.empty:
        return None
    sub = g[g["calculation_date"] <= bar_date]
    if sub.empty:
        return None
    return sub.iloc[-1].to_dict()


def asof_market(market_hist: pd.DataFrame, bar_date: pd.Timestamp) -> Optional[dict]:
    if market_hist is None or market_hist.empty:
        return None
    sub = market_hist[market_hist["calculation_date"] <= bar_date]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return {
        "market_regime": row["market_regime"],
        "pct_above_sma50": row["pct_above_sma50"],
    }


def _swept_no_reclaim(low: np.ndarray, close: np.ndarray, T: int,
                      strat: SmartMoneyReclaimStrategy) -> bool:
    """Control: a bar that swept the swing low but did NOT reclaim it."""
    sweep_start = T - (strat.SWEEP_WINDOW - 1)
    swing_low, _ = strat._find_swing_low(low, sweep_start, strat.SWING_LOOKBACK)
    if swing_low is None:
        return False
    window_lows = low[sweep_start:T + 1]
    if np.isnan(window_lows).all():
        return False
    swept = float(np.nanmin(window_lows)) < swing_low
    reclaimed = close[T] >= swing_low * strat.reclaim_tol
    return swept and not reclaimed


def collect_signals(conn, max_tickers: Optional[int], top_liquid: Optional[int],
                    metrics_hist: dict, market_hist: pd.DataFrame) -> tuple:
    """Walk-forward per ticker per bar. Returns (signals_df, control_df, universe_df).

    universe_df = every (ticker, bar_date, close) evaluated (for the random baseline).
    """
    daily = fetch_df(conn, """
        SELECT ticker, timestamp, open, high, low, close, volume
        FROM stock_candles_synthesized
        WHERE timeframe = '1d'
        ORDER BY ticker, timestamp
    """)
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])

    liquid_set = top_liquid_tickers(daily, top_liquid) if top_liquid else None
    if liquid_set is not None:
        print(f"  liquidity filter: keeping top {top_liquid} tickers by median $-volume")

    strat = SmartMoneyReclaimStrategy(stop_loss_pct=SL_PCT, take_profit_pct=TP_PCT)
    groups = list(daily.groupby("ticker", sort=True))
    if liquid_set is not None:
        groups = [(t, g) for t, g in groups if t in liquid_set]
    if max_tickers:
        groups = groups[:max_tickers]

    sig_rows, ctrl_rows, univ_rows = [], [], []
    for tkr, g in groups:
        g = g.reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        gp = strat.prepare(g)
        low = pd.to_numeric(gp["low"], errors="coerce").to_numpy()
        close = pd.to_numeric(gp["close"], errors="coerce").to_numpy()
        n = len(gp)
        # step from first evaluable bar; walk forward, only past+current bars given
        for T in range(MIN_BARS - 1, n):
            bar_date = gp["timestamp"].iloc[T]
            sub = gp.iloc[: T + 1]
            univ_rows.append({"ticker": tkr, "signal_ts": bar_date,
                              "close": float(close[T])})
            mkt = asof_market(market_hist, bar_date)
            mets = asof_metrics(metrics_hist, tkr, bar_date)
            sig = strat.scan(sub, tkr, metrics=mets, market=mkt)
            if sig is not None:
                sig_rows.append({
                    "ticker": tkr,
                    "signal_ts": bar_date,
                    "close": float(sig.entry_price),
                    "score": sig.composite_score,
                    "zone_pos": sig.zone_pos,
                    "divergence_delta": sig.divergence_delta,
                    "rvol": sig.relative_volume,
                    "pct_above_sma50": (float(mkt["pct_above_sma50"])
                                        if mkt and mkt.get("pct_above_sma50") is not None
                                        else np.nan),
                })
            elif _swept_no_reclaim(low, close, T, strat):
                ctrl_rows.append({"ticker": tkr, "signal_ts": bar_date,
                                  "close": float(close[T])})
    return pd.DataFrame(sig_rows), pd.DataFrame(ctrl_rows), pd.DataFrame(univ_rows)


def simulate(conn, signals: pd.DataFrame) -> pd.DataFrame:
    """Attach a simulated outcome (pnl_pct, pnl_usd, outcome) to every signal."""
    if signals.empty:
        return signals
    tickers = sorted(signals["ticker"].unique().tolist())
    candles = fetch_df(conn, """
        SELECT ticker, timestamp, open, high, low, close
        FROM stock_candles
        WHERE timeframe = '1h' AND ticker = ANY(%s)
        ORDER BY ticker, timestamp
    """, (tickers,))
    if candles.empty:
        return pd.DataFrame()
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


def report(sims: pd.DataFrame, control_sims: pd.DataFrame,
           random_sims: pd.DataFrame, n_days: int) -> None:
    print("\n" + "=" * 82)
    print("SMART MONEY RECLAIM — walk-forward backtest under live auto-trader bracket")
    print(f"  bracket: SL {SL_PCT}%  TP {TP_PCT}%  ATR-off  BE-off  "
          f"notional ${NOTIONAL:.0f}  timeout {TIMEOUT_TRADING_DAYS}d")
    print("=" * 82)

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

    # signals/day
    if n_days > 0:
        print(f"\n    signals/trading-day (approx): {len(sims) / max(1, n_days):.2f}")

    # [2] breadth split
    print("\n[2] Breadth split (market_context pct_above_sma50 at signal time):")
    strong = sims[sims["pct_above_sma50"] >= 50]
    weak = sims[sims["pct_above_sma50"] < 50]
    print(fmt("breadth >= 50%", metrics(strong)))
    print(fmt("breadth <  50%", metrics(weak)))

    # [3] random-entry baseline
    print("\n[3] Random-entry baseline (same universe/dates, random longs):")
    print(fmt("RANDOM", metrics(random_sims)))

    # [4] sweep-without-reclaim control
    print("\n[4] Control: swept swing low but NO reclaim (null setup):")
    print(fmt("SWEEP-NO-RECLAIM", metrics(control_sims)))

    # [5] monthly stability
    print("\n[5] Monthly (ALL signals):")
    sims["month"] = sims["signal_ts"].dt.to_period("M").astype(str)
    for mth, grp in sims.groupby("month"):
        print(fmt(mth, metrics(grp)))

    print("\nPF>1.0 = positive expectancy. Verdict should lean on the OOS row, the "
          "breadth>=50 split, and whether the signal beats BOTH the random baseline "
          "and the sweep-no-reclaim control (else the reclaim+divergence add nothing).")
    print("=" * 82)


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
        print("Loading point-in-time metrics + market_context history ...")
        metrics_hist = load_metrics_history(conn)
        market_hist = load_market_history(conn)
        print(f"  metrics for {len(metrics_hist)} tickers; "
              f"{0 if market_hist is None else len(market_hist)} market_context rows")

        print(f"Collecting signals walk-forward (top_liquid={args.top_liquid or '-'}, "
              f"max_tickers={args.max_tickers or 'all'}) ...")
        signals, control, universe = collect_signals(
            conn, args.max_tickers, args.top_liquid, metrics_hist, market_hist)
        n_sig = 0 if signals.empty else len(signals)
        print(f"  {n_sig} raw signals across "
              f"{signals['ticker'].nunique() if not signals.empty else 0} tickers")
        if signals.empty:
            return

        n_days = universe["signal_ts"].dt.normalize().nunique() if not universe.empty else 0

        print("Simulating signals under live bracket ...")
        sims = simulate(conn, signals)
        print(f"  {len(sims)} signal trades simulated")

        print("Simulating sweep-no-reclaim control ...")
        control_sims = simulate(conn, control) if not control.empty else pd.DataFrame()
        print(f"  {0 if control_sims is None or control_sims.empty else len(control_sims)} control trades")

        print("Simulating random-entry baseline (matched sample) ...")
        if not universe.empty:
            rng = np.random.default_rng(RANDOM_SEED)
            k = min(len(universe), max(len(signals) * 5, 500))
            random_pick = universe.sample(n=k, random_state=int(rng.integers(1e9)))
            random_sims = simulate(conn, random_pick)
        else:
            random_sims = pd.DataFrame()
        print(f"  {0 if random_sims is None or random_sims.empty else len(random_sims)} random trades")

        report(sims,
               control_sims if control_sims is not None else pd.DataFrame(),
               random_sims if random_sims is not None else pd.DataFrame(),
               n_days)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
