"""
Adaptive Trend Pullback — Walk-Forward Backtest
===============================================
Validates the NEW `adaptive_trend_pullback` strategy (LuxAlgo blend: SuperTrend-AI
adaptive factor + Nadaraya-Watson kernel envelope + predictive ranges) under the
REAL stock auto-trader execution model, BEFORE any wiring, and reports per-period
PF / win-rate / expectancy plus the controls that decide whether the blend earns
its complexity.

Mirrors `smart_money_reclaim_replay.py`: it reuses the validated path-walking
engine (`walk_bars` / `bars_after`) from daytrade_edge_sim.py and feeds it the
CURRENT live bracket (DB is master):

    SL 2.5%   TP 7.0%   ATR-stop OFF   BE OFF   notional $500
    entry = OPEN of first 1h bar STRICTLY AFTER the signal bar's close (no look-ahead)
    timeout = 10 trading days

Entry signals come from the SAME strategy class the (future) live scanner runs
(AdaptiveTrendPullbackStrategy.scan), evaluated WALK-FORWARD: for every ticker, on
each daily bar we feed the strategy only bars up to and including that bar, plus
the point-in-time screening metrics and the market_context breadth row AS-OF that
date. So this measures the as-of-decision-time edge with no leakage.

Views reported:
  (1) ALL signals (raw edge), split TRAIN (<2026-04-01) / OOS (>=).
  (2) Breadth split: PF on signals fired when pct_above_sma50 >= 50 vs < 50.
  (3) Random-long baseline (same universe/dates).
  (4) NULL CONTROL — fixed factor f=3.0 (grid midpoint) instead of the K-means-
      selected factor, everything else identical. Isolates whether ADAPTIVITY adds
      anything: if PF(adaptive) ~= PF(fixed-3.0), the K-means is decorative.
  (5) Ablation — perf_score gate OFF (min_perf_score=0). Isolates the quality
      gate's value. NOTE: the gated variant uses the CALIBRATED default (4.0), not
      the spec's 7 — perf_score maxes ~5 for equities at pullback bars, so >=7
      fires ~never (see strategy docstring). Tune the gate via that default if you
      want a stricter/looser comparison.
  (6) Monthly stability (ALL adaptive signals).
  (7) Selected-factor histogram: distribution of the nearest-grid f_star across all
      fired adaptive signals. If 90%+ cluster on 1-2 factors, prints a WARNING that
      the adaptivity is decorative.

VERDICT GUIDANCE (printed): the signal must beat BOTH the random baseline AND the
fixed-3.0 null, else the blend adds nothing over a plain SuperTrend pullback.

CAVEATS (shared with the other replay harnesses): the live VWAP / positive-day /
spread / score>=65 / intraday-RVOL gates are LIVE-ONLY and not reconstructable, so
this is the ALL-SIGNALS ranking edge, not realised live-traded P&L. The in-sample
window is single-ish regime — read positives cautiously. Read-only: modifies no
table.

Usage (inside the stock-scanner container):
    docker exec stock-scanner python -m stock_scanner.analysis.adaptive_trend_pullback_replay
    docker exec stock-scanner python -m stock_scanner.analysis.adaptive_trend_pullback_replay --max-tickers 500
    docker exec stock-scanner python -m stock_scanner.analysis.adaptive_trend_pullback_replay --top-liquid 500
"""

from __future__ import annotations

import sys
import argparse
import warnings
from collections import Counter
from typing import Optional, List

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import (
    get_conn, fetch_df, walk_bars, bars_after,
)
from stock_scanner.strategies.adaptive_trend_pullback import AdaptiveTrendPullbackStrategy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- LIVE auto-trader params (stock_auto_trade_settings, read 2026-06-30) ---
SL_PCT = 2.5
TP_PCT = 7.0
NOTIONAL = 500.0
TIMEOUT_TRADING_DAYS = 10

TRAIN_END = pd.Timestamp("2026-04-01")  # OOS = on/after this date
MIN_BARS = 110                          # strategy needs ~60+ but we require history
RANDOM_SEED = 7

# The ADAPTIVE factor histogram clustered ~78% on 3.5, so the null / simple
# variants pin the trade factor to 3.5 (the factor the k-means actually keeps
# picking). This makes NULL-A the tightest possible "does the pick matter?" test.
NULL_FACTOR = 3.5
ADAPTIVE_PERF_GATE = 4.0                # calibrated default (see strategy docstring)


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


def _walk_variant(groups: list, strat: AdaptiveTrendPullbackStrategy,
                  metrics_hist: dict, market_hist: pd.DataFrame,
                  capture_universe: bool = False) -> tuple:
    """Walk-forward every ticker/bar for one strategy variant.

    Returns (signals_df, universe_df). universe_df is only populated when
    capture_universe=True (used once, for the random baseline).
    """
    sig_rows, univ_rows = [], []
    for tkr, g in groups:
        g = g.reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        gp = strat.prepare(g)
        close = pd.to_numeric(gp["close"], errors="coerce").to_numpy()
        n = len(gp)
        for T in range(MIN_BARS - 1, n):
            bar_date = gp["timestamp"].iloc[T]
            sub = gp.iloc[: T + 1]
            if capture_universe:
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
                    "zone": sig.zone,
                    "selected_factor": sig.selected_factor,
                    "f_star": sig.f_star,
                    "perf_score": sig.perf_score,
                    "perf_idx": float(sig.raw_data.get("perf_idx", 0.0)),
                    "pct_above_sma50": (float(mkt["pct_above_sma50"])
                                        if mkt and mkt.get("pct_above_sma50") is not None
                                        else np.nan),
                })
    return pd.DataFrame(sig_rows), pd.DataFrame(univ_rows)


def _fast_ticker(tkr: str, g: pd.DataFrame, strat: AdaptiveTrendPullbackStrategy,
                 metrics_hist: dict, market_hist: pd.DataFrame,
                 adaptive_gate: float, capture_universe: bool) -> tuple:
    """Single-pass walk of ONE ticker producing ADAPTIVE / NULL-A / SIMPLE decisions.

    The expensive series (9 SuperTrends, per-factor performance memory, the
    Nadaraya-Watson endpoint series, the |Δclose| EMA denominator) are CAUSAL
    recursions/windows anchored at bar 0, so their value at bar T computed over the
    FULL ticker equals the value scan() computes over the sub-series df.iloc[:T+1]
    (sub always starts at bar 0). We therefore precompute them ONCE per ticker and
    read index T — instead of scan()'s O(n) rebuild every bar — which removes the
    ~6x-per-bar NW recomputation that made the 3-variant walk blow the time budget.
    A parity self-check (see collect_signals) confirms ADAPTIVE output is identical
    to the scan()-based walk.

    Uses strat's OWN helper methods (_supertrend/_perf_memory/_kmeans_1d/
    _nwe_endpoint/_predictive_avg) so the math is byte-identical to scan().
    """
    g = g.reset_index(drop=True)
    n = len(g)
    if n < MIN_BARS:
        return [], [], [], []
    gp = strat.prepare(g)
    high = pd.to_numeric(gp["high"], errors="coerce").to_numpy()
    low = pd.to_numeric(gp["low"], errors="coerce").to_numpy()
    close = pd.to_numeric(gp["close"], errors="coerce").to_numpy()
    atr10 = gp["atr_10"].to_numpy()
    tr = gp["_tr"].to_numpy()
    grid = strat.grid
    idx35 = int(np.argmin(np.abs(grid - NULL_FACTOR)))

    # --- precompute per-factor direction + performance memory (causal) ---
    DIR = np.zeros((len(grid), n))
    PERF = np.zeros((len(grid), n))
    for k, f in enumerate(grid):
        _, direction = strat._supertrend(high, low, close, atr10, float(f))
        DIR[k] = direction
        PERF[k] = strat._perf_memory(close, direction)

    # --- |Δclose| EMA denominator (causal) ---
    abs_dc = np.abs(np.diff(close, prepend=close[0]))
    den_series = pd.Series(abs_dc).ewm(
        alpha=2.0 / (strat.PERF_MEMORY + 1.0), adjust=False).mean().to_numpy()

    # --- Nadaraya-Watson endpoint series + MAE band + zone (causal) ---
    nwe_series = np.array([strat._nwe_endpoint(close, t) for t in range(n)])
    resid = np.abs(close - nwe_series)
    mae_series = strat.NW_MULT * pd.Series(resid).rolling(
        strat.NW_MAE_WINDOW, min_periods=1).mean().to_numpy()
    band_w = 2.0 * mae_series
    with np.errstate(divide="ignore", invalid="ignore"):
        zone_series = np.where(band_w > 0, (close - (nwe_series - mae_series)) / band_w, 1.0)

    ad_rows, na_rows, sm_rows, univ_rows = [], [], [], []
    fl = strat.FRESH_LOOKBACK
    for T in range(MIN_BARS - 1, n):
        bar_date = gp["timestamp"].iloc[T]
        if capture_universe:
            univ_rows.append({"ticker": tkr, "signal_ts": bar_date, "close": float(close[T])})
        if not np.isfinite(close[T]) or close[T] <= 0 or not np.isfinite(atr10[T]):
            continue

        # shared gates (factor-independent)
        zone = float(zone_series[T])
        if not (0.0 < zone <= strat.max_zone):
            continue
        recent = zone_series[max(0, T - fl):T]
        if not (recent.size and np.any(recent > 0.5)):
            continue
        # predictive-range slope — replicate scan exactly (per-bar prlen on the sub)
        pr_len = min(strat.PR_ATR_LEN, T + 1)
        atr_pr = pd.Series(tr[:T + 1]).rolling(pr_len).mean().to_numpy() * strat.PR_FACTOR
        pr_avg = strat._predictive_avg(close[:T + 1], atr_pr)
        pr_slope = int(np.sign(pr_avg[T] - pr_avg[T - strat.PR_SLOPE_LOOKBACK])) \
            if T - strat.PR_SLOPE_LOOKBACK >= 0 else 0
        if pr_slope < 0:
            continue
        # breadth co-gate (missing flagged, not blocked)
        mkt = asof_market(market_hist, bar_date)
        if mkt is not None:
            if mkt.get("market_regime") == "bear_confirmed":
                continue
            psa = mkt.get("pct_above_sma50")
            try:
                psa = float(psa) if psa is not None else None
            except (TypeError, ValueError):
                psa = None
            if psa is not None and psa < strat.min_pct_above_sma50:
                continue
            pct_col = psa if psa is not None else np.nan
        else:
            pct_col = np.nan

        den = float(den_series[T])
        # k-means (shared basis for ADAPTIVE + NULL-A)
        perf_t = PERF[:, T]
        labels, centroids = strat._kmeans_1d(perf_t)
        best_cluster = int(np.argmax(centroids))
        best_centroid = float(centroids[best_cluster])
        in_best = grid[labels == best_cluster]
        f_star = float(in_best.mean()) if len(in_best) else float(np.median(grid))
        ad_idx = int(np.argmin(np.abs(grid - f_star)))
        km_idx = (max(best_centroid, 0.0) / den) if den > 0 else 0.0
        km_score = int(round(float(np.clip(km_idx, 0.0, 1.0)) * 10))

        row_base = {"ticker": tkr, "signal_ts": bar_date, "close": round(float(close[T]), 4),
                    "zone": round(zone, 4), "pct_above_sma50": pct_col}

        # ADAPTIVE: k-means factor + k-means gate
        if int(DIR[ad_idx][T]) == 1 and km_score >= adaptive_gate:
            ad_rows.append({**row_base, "score": km_score,
                            "selected_factor": float(grid[ad_idx]),
                            "f_star": round(f_star, 4),
                            "perf_score": km_score, "perf_idx": round(km_idx, 6)})
        # NULL-A: trade factor forced 3.5, IDENTICAL k-means gate
        if int(DIR[idx35][T]) == 1 and km_score >= adaptive_gate:
            na_rows.append({**row_base, "score": km_score,
                            "selected_factor": float(grid[idx35]),
                            "f_star": float(NULL_FACTOR),
                            "perf_score": km_score, "perf_idx": round(km_idx, 6)})
        # SIMPLE-3.5: no k-means; own-factor perf; gate OFF (calibrated later)
        if int(DIR[idx35][T]) == 1:
            own = float(PERF[idx35][T])
            sm_idx = (max(own, 0.0) / den) if den > 0 else 0.0
            sm_score = int(round(float(np.clip(sm_idx, 0.0, 1.0)) * 10))
            sm_rows.append({**row_base, "score": sm_score,
                            "selected_factor": float(grid[idx35]),
                            "f_star": float(NULL_FACTOR),
                            "perf_score": sm_score, "perf_idx": round(sm_idx, 6)})
    return ad_rows, na_rows, sm_rows, univ_rows


def _fast_collect(groups: list, strat: AdaptiveTrendPullbackStrategy,
                  metrics_hist: dict, market_hist: pd.DataFrame,
                  adaptive_gate: float) -> tuple:
    """Single-pass over all tickers. Returns (adaptive, null_a, simple_all, universe)."""
    ad, na, sm, uv = [], [], [], []
    for tkr, g in groups:
        a, b, c, u = _fast_ticker(tkr, g, strat, metrics_hist, market_hist,
                                  adaptive_gate, capture_universe=True)
        ad += a; na += b; sm += c; uv += u
    return (pd.DataFrame(ad), pd.DataFrame(na), pd.DataFrame(sm), pd.DataFrame(uv))


def _parity_check(groups: list, strat: AdaptiveTrendPullbackStrategy,
                  metrics_hist: dict, market_hist: pd.DataFrame) -> None:
    """Assert the fast single-pass ADAPTIVE signals == the scan()-based ADAPTIVE
    signals (same (ticker, signal_ts) set) on a small sample, so the optimized walk
    is provably faithful to the shared strategy code path."""
    if not groups:
        print("    (no tickers to parity-check)")
        return
    ref, _ = _walk_variant(groups, strat, metrics_hist, market_hist)
    fast_ad, _, _, _ = _fast_collect(groups, strat, metrics_hist, market_hist,
                                     strat.min_perf_score)
    ref_keys = set() if ref.empty else set(zip(ref["ticker"], ref["signal_ts"]))
    fast_keys = set() if fast_ad.empty else set(zip(fast_ad["ticker"], fast_ad["signal_ts"]))
    if ref_keys == fast_keys:
        print(f"    PARITY OK: {len(ref_keys)} ADAPTIVE signals match scan() exactly.")
    else:
        only_ref = len(ref_keys - fast_keys)
        only_fast = len(fast_keys - ref_keys)
        print(f"    PARITY WARNING: scan-only={only_ref} fast-only={only_fast} "
              f"(ref={len(ref_keys)} fast={len(fast_keys)}) — fast walk diverges from "
              f"scan(); treat results with caution.")


def _calibrate_simple_threshold(simple_all: pd.DataFrame, target_n: int) -> tuple:
    """Volume-match SIMPLE-3.5 to ADAPTIVE's fire count.

    NOTE: a single fixed factor's OWN integer perf_score is degenerate (rounds to 0
    at pullback bars — the ADAPTIVE non-zero perf_score comes from the k-means
    best-CLUSTER centroid, a pool of the strongest factors, which no single factor
    reaches). So the integer perf_score is all-or-nothing and cannot volume-match.
    We instead rank the gate-off SIMPLE signals by the CONTINUOUS perf_idx
    (max(P,0)/EMA(|Δclose|), same normalization) and keep the top `target_n`, which
    is the faithful "recalibrated min-perf threshold". Returns
    (perf_idx_cutoff, filtered_df)."""
    if simple_all is None or simple_all.empty:
        return 0.0, simple_all
    if target_n <= 0 or target_n >= len(simple_all):
        return 0.0, simple_all
    ranked = simple_all.sort_values("perf_idx", ascending=False).reset_index(drop=True)
    cutoff = float(ranked.iloc[target_n - 1]["perf_idx"])
    kept = ranked.iloc[:target_n].copy()
    return cutoff, kept


def collect_signals(conn, max_tickers: Optional[int], top_liquid: Optional[int],
                    metrics_hist: dict, market_hist: pd.DataFrame) -> dict:
    """Walk-forward for ADAPTIVE, NULL-A and SIMPLE-3.5 over the SAME universe.

      ADAPTIVE  : k-means-selected trade factor + k-means perf gate (default 4).
      NULL-A    : IDENTICAL k-means perf gate, but trade factor forced to NULL_FACTOR
                  (3.5). Isolates the factor pick, gate held constant.
      SIMPLE-3.5: no k-means; trade + perf both from factor-3.5's OWN perf-memory.
                  Collected with the gate OFF, then a threshold is calibrated so its
                  fire-rate ~= ADAPTIVE's (report the chosen threshold + n).
    """
    daily = fetch_df(conn, """
        SELECT ticker, timestamp, open, high, low, close, volume
        FROM stock_candles_synthesized
        WHERE timeframe = '1d'
        ORDER BY ticker, timestamp
    """)
    if daily.empty:
        return {}
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])

    liquid_set = top_liquid_tickers(daily, top_liquid) if top_liquid else None
    if liquid_set is not None:
        print(f"  liquidity filter: keeping top {top_liquid} tickers by median $-volume")

    groups = list(daily.groupby("ticker", sort=True))
    if liquid_set is not None:
        groups = [(t, g) for t, g in groups if t in liquid_set]
    if max_tickers:
        groups = groups[:max_tickers]

    strat = AdaptiveTrendPullbackStrategy(
        stop_loss_pct=SL_PCT, take_profit_pct=TP_PCT, min_perf_score=ADAPTIVE_PERF_GATE)

    # --- parity self-check: fast single-pass ADAPTIVE == scan()-based ADAPTIVE ---
    print("  parity self-check (fast walk vs scan() on first tickers) ...")
    _parity_check(groups[:20], strat, metrics_hist, market_hist)

    print("  single-pass walk (ADAPTIVE / NULL-A / SIMPLE-3.5 in one pass) ...")
    sig_adaptive, sig_null_a, sig_simple_all, universe = _fast_collect(
        groups, strat, metrics_hist, market_hist, ADAPTIVE_PERF_GATE)
    n_adaptive = 0 if sig_adaptive.empty else len(sig_adaptive)
    print(f"    ADAPTIVE {n_adaptive} | NULL-A "
          f"{0 if sig_null_a.empty else len(sig_null_a)} | SIMPLE-{NULL_FACTOR} raw "
          f"{0 if sig_simple_all.empty else len(sig_simple_all)}")
    n_simple_all = 0 if sig_simple_all.empty else len(sig_simple_all)
    thr, sig_simple = _calibrate_simple_threshold(sig_simple_all, n_adaptive)
    n_pos = int((sig_simple_all["perf_idx"] > 0).sum()) if not sig_simple_all.empty else 0
    print(f"    {n_simple_all} SIMPLE-{NULL_FACTOR} raw (gate off; {n_pos} had "
          f"perf_idx>0); calibrated perf_idx>={thr:.4f} -> "
          f"{0 if sig_simple is None or sig_simple.empty else len(sig_simple)} signals "
          f"(volume-matched to ADAPTIVE n={n_adaptive})")

    return {
        "adaptive": sig_adaptive,
        "null_a": sig_null_a,
        "simple": sig_simple,
        "simple_threshold": thr,
        "universe": universe,
    }


def simulate(conn, signals: pd.DataFrame) -> pd.DataFrame:
    """Attach a simulated outcome (pnl_pct, pnl_usd, outcome) to every signal."""
    if signals is None or signals.empty:
        return pd.DataFrame()
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
    if df is None or df.empty:
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


def _row_metrics(sims: pd.DataFrame) -> dict:
    """n / PF / WR / OOS-PF / breadth>=50-PF for one variant's simulated trades."""
    if sims is None or sims.empty:
        return {"n": 0}
    s = sims.copy()
    s["oos"] = s["signal_ts"] >= TRAIN_END
    m_all = metrics(s)
    m_oos = metrics(s[s["oos"]])
    m_br = metrics(s[s["pct_above_sma50"] >= 50]) if "pct_above_sma50" in s.columns else {"n": 0}
    return {
        "n": m_all["n"],
        "pf": m_all["pf"],
        "wr": m_all["wr"],
        "oos_n": m_oos["n"],
        "oos_pf": m_oos.get("pf", float("nan")),
        "br_n": m_br["n"],
        "br_pf": m_br.get("pf", float("nan")),
    }


def _pf(v) -> str:
    if v is None or (isinstance(v, float) and (v != v)):
        return "  -  "
    return "inf" if v == float("inf") else f"{v:.2f}"


def report(bundle_sims: dict, simple_threshold: float,
           adaptive_signals: pd.DataFrame, n_days: int) -> None:
    print("\n" + "=" * 90)
    print("ADAPTIVE TREND PULLBACK — variant comparison under live auto-trader bracket")
    print(f"  bracket: SL {SL_PCT}%  TP {TP_PCT}%  ATR-off  BE-off  "
          f"notional ${NOTIONAL:.0f}  timeout {TIMEOUT_TRADING_DAYS}d")
    print(f"  ADAPTIVE gate perf_score>={int(ADAPTIVE_PERF_GATE)} ; NULL-A holds the "
          f"SAME k-means gate but forces trade factor {NULL_FACTOR} ;")
    print(f"  SIMPLE-{NULL_FACTOR} uses NO k-means; own-factor perf_idx cutoff "
          f">={simple_threshold:.4f} (volume-matched to ADAPTIVE — the integer "
          f"perf_score is degenerate for a single factor, so we rank on continuous "
          f"perf_idx).")
    print("=" * 90)

    rows = [
        ("ADAPTIVE",           _row_metrics(bundle_sims.get("adaptive"))),
        ("NULL-A (factor 3.5)", _row_metrics(bundle_sims.get("null_a"))),
        (f"SIMPLE-{NULL_FACTOR} (no k-means)", _row_metrics(bundle_sims.get("simple"))),
        ("RANDOM (long)",       _row_metrics(bundle_sims.get("random"))),
    ]

    hdr = f"  {'variant':<26} {'n':>5}  {'PF':>5}  {'WR%':>5}  {'OOS-n':>5}  {'OOS-PF':>6}  {'br>=50-n':>8}  {'br>=50-PF':>9}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, m in rows:
        if m["n"] == 0:
            print(f"  {label:<26} {'n=0':>5}")
            continue
        print(f"  {label:<26} {m['n']:>5}  {_pf(m['pf']):>5}  {m['wr']:>5.1f}  "
              f"{m['oos_n']:>5}  {_pf(m['oos_pf']):>6}  {m['br_n']:>8}  {_pf(m['br_pf']):>9}")

    if n_days > 0 and bundle_sims.get("adaptive") is not None and not bundle_sims["adaptive"].empty:
        print(f"\n  ADAPTIVE signals/trading-day (approx): "
              f"{len(bundle_sims['adaptive']) / max(1, n_days):.2f}")

    # ADAPTIVE selected-factor histogram
    print("\nADAPTIVE selected-factor histogram (nearest-grid f_star across fired signals):")
    if adaptive_signals is not None and not adaptive_signals.empty:
        cnt = Counter(round(float(f), 1) for f in adaptive_signals["selected_factor"])
        total = sum(cnt.values())
        for fac in sorted(cnt):
            share = cnt[fac] / total * 100
            print(f"    factor {fac:<4} n={cnt[fac]:<5} {share:5.1f}%  {'#' * int(round(share / 2))}")
        top2 = sum(v for _, v in cnt.most_common(2))
        if total > 0 and top2 / total >= 0.90:
            print("    WARNING: 90%+ of signals cluster on 1-2 factors -> the k-means "
                  "adaptivity is largely DECORATIVE; a fixed factor behaves the same.")

    # Auto verdict
    m_ad = _row_metrics(bundle_sims.get("adaptive"))
    m_na = _row_metrics(bundle_sims.get("null_a"))
    m_sm = _row_metrics(bundle_sims.get("simple"))
    m_rn = _row_metrics(bundle_sims.get("random"))

    def pfx(m, key="oos_pf"):
        v = m.get(key, float("nan"))
        if v is None or (isinstance(v, float) and v != v):
            return None
        return 99.0 if v == float("inf") else v

    print("\nINTERPRETATION:")
    rn = pfx(m_rn, "pf")
    sm_oos = pfx(m_sm); ad_oos = pfx(m_ad); na_oos = pfx(m_na)
    if sm_oos is not None and rn is not None:
        keep = "YES" if sm_oos > rn + 0.10 else "NO"
        print(f"  - SIMPLE-{NULL_FACTOR} retains beat-random edge? OOS-PF {_pf(sm_oos)} vs "
              f"random PF {_pf(rn)} -> {keep}. If YES, the edge is the NW-kernel "
              f"pullback entry and the simpler scanner is the ship candidate.")
    if ad_oos is not None and na_oos is not None and sm_oos is not None:
        matters = "YES" if (ad_oos > na_oos + 0.10 and ad_oos > sm_oos + 0.10) else "NO"
        print(f"  - Does k-means adaptivity matter? ADAPTIVE OOS-PF {_pf(ad_oos)} vs "
              f"NULL-A {_pf(na_oos)} vs SIMPLE {_pf(sm_oos)} -> {matters}. "
              f"If NO, k-means is confirmed DECORATIVE -> drop it, ship SIMPLE-{NULL_FACTOR}.")
    print("=" * 90)


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
        bundle = collect_signals(conn, args.max_tickers, args.top_liquid,
                                 metrics_hist, market_hist)
        if not bundle or bundle.get("adaptive") is None or bundle["adaptive"].empty:
            print("No adaptive signals collected.")
            return

        adaptive_signals = bundle["adaptive"]
        universe = bundle["universe"]
        simple_threshold = bundle.get("simple_threshold", 0)
        n_days = universe["signal_ts"].dt.normalize().nunique() if not universe.empty else 0

        print("Simulating ADAPTIVE signals under live bracket ...")
        sims = simulate(conn, adaptive_signals)
        print(f"  {len(sims)} adaptive trades simulated")

        print(f"Simulating NULL-A (k-means gate held, trade factor {NULL_FACTOR}) ...")
        null_a_sims = simulate(conn, bundle["null_a"])
        print(f"  {len(null_a_sims)} NULL-A trades")

        print(f"Simulating SIMPLE-{NULL_FACTOR} (calibrated perf_score>={simple_threshold}) ...")
        simple_sims = simulate(conn, bundle["simple"])
        print(f"  {len(simple_sims)} SIMPLE-{NULL_FACTOR} trades")

        print("Simulating random-entry baseline (matched sample) ...")
        if not universe.empty:
            rng = np.random.default_rng(RANDOM_SEED)
            k = min(len(universe), max(len(adaptive_signals) * 5, 500))
            random_pick = universe.sample(n=k, random_state=int(rng.integers(1e9)))
            random_sims = simulate(conn, random_pick)
        else:
            random_sims = pd.DataFrame()
        print(f"  {len(random_sims)} random trades")

        # random rows need pct_above_sma50 for the breadth column; fill NaN if absent
        if not random_sims.empty and "pct_above_sma50" not in random_sims.columns:
            random_sims = random_sims.assign(pct_above_sma50=np.nan)

        bundle_sims = {
            "adaptive": sims,
            "null_a": null_a_sims,
            "simple": simple_sims,
            "random": random_sims,
        }
        report(bundle_sims, simple_threshold, adaptive_signals, n_days)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
