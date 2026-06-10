"""
Held-out walk-forward replay: does cross-scanner GRADING improve daytrade selection?
=====================================================================================
Tests two pre-registered grading terms layered on a reconstructed *core*
candidate_score, against day-trade-simulated outcomes:
  (A) cross-FAMILY confluence bonus  — reward agreement across independent
      scanner families (raw agreement-count would double-count one trend family).
  (B) shrunk, walk-forward day-trade EDGE prior per scanner — replaces the
      binary PF<1.0 floor with a graded, empirical-Bayes-shrunk term.

Usage (inside stock-scanner container, read-only — modifies no table):
    docker exec stock-scanner python -m stock_scanner.analysis.grading_replay

-------------------------------------------------------------------------------
PRE-REGISTERED (frozen before touching the eval window — see gate #3):
  - SCANNER_FAMILY map (literal dict below).
  - TRAIN_END / EVAL_START split.
  - EDGE_SHRINK_K, EDGE_SCALE_PTS, term clips. The confluence point magnitude is
    *measured on the train split only* and frozen for eval.
  - VERDICT SOURCE (gate #4): the within-decile incremental-information lift is
    the TIE-BREAKER. The top-N list-reconstruction is presentation only.
  - Outcomes are computed under BOTH pessimistic and optimistic break-even (gate
    #4): if the baseline->augmented eval expectancy delta is within the pess/opt
    band width, the verdict is UNCERTAIN. A null is a real result here.

CAVEATS:
  - POPULATION GAP (gate #1, inherited from daytrade_edge_sim.py): the live
    auto-trader gates on VWAP / spread / score>=65 / live-intraday-RVOL — all
    LIVE-ONLY and not reconstructable. Both arms omit them identically, so the
    grading DELTA is internally valid, but the verdict is about the ALL-SIGNALS
    RANKING, not the live traded set's realised P&L. A score-percentile gate
    robustness check (rough proxy for score>=65) is reported.
  - Edge prior uses only RESOLVED signals — a signal counts toward scanner s's
    prior at date D only if its simulated exit_time < D (gate #2). No leakage.
  - All metrics in-sample on a Dec-2025 -> Jun-2026 window (screening_metrics
    start). Single-ish regime; treat positive findings cautiously.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import (
    get_conn, fetch_df, build_bracket, walk_bars,
    build_candle_index, build_daily_index, atr_pct_before, bars_after,
)

# ---------------------------------------------------------------------------
# PRE-REGISTERED PARAMETERS  (do NOT tune on the eval window)
# ---------------------------------------------------------------------------
SCANNER_FAMILY = {
    "zlma_trend": "trend", "ema_pullback": "trend", "ultimate_ma_mtf": "trend",
    "squeeze_momentum": "trend", "macd_momentum": "trend", "pocket_pivot": "trend",
    "trend_momentum": "trend",
    "gap_and_go": "breakout", "breakout_confirmation": "breakout",
    "volatility_contraction_breakout": "breakout", "high_retest": "breakout",
    "short_squeeze_breakout": "breakout",
    "reversal_scanner": "reversal", "rsi_divergence": "reversal", "trend_reversal": "reversal",
    "premarket_catalyst": "catalyst", "earnings_drift": "catalyst",
    "sector_rotation_leader": "rotation", "relative_strength_leader": "rotation",
}
DEFAULT_FAMILY = "other"

TRAIN_END = pd.Timestamp("2026-03-31")   # tune confluence magnitude on/before this
EVAL_START = pd.Timestamp("2026-04-01")  # held-out window starts here

EDGE_SHRINK_K = 20.0     # empirical-Bayes pseudo-count (signals) -> shrink small-n scanners to pool
EDGE_SCALE_PTS = 10.0    # candidate_score points per 1.0% of (shrunk edge - pool) expectancy
EDGE_CLIP_PTS = 6.0      # bound the edge term
CONF_CLIP_PTS = 8.0      # bound the confluence term
TOP_N_GRID = [3, 5, 10]
HEADLINE_N = 5           # auto-trader's active-order cap
SCORE_GATE_PCTL = 0.50   # robustness: re-run delta on the top-half of core score (proxy for score>=65)


# ---------------------------------------------------------------------------
# 1. Load signals joined to as-of screening_metrics; build core baseline score
# ---------------------------------------------------------------------------

def load_signals(conn) -> pd.DataFrame:
    """BUY signals joined to screening_metrics on (ticker, calculation_date=signal_date).

    Only the screening_metrics-derived (reconstructable) terms of the live
    daytrade candidate_score are pulled; premarket / intraday / news / daq terms
    are LIVE-ONLY and intentionally omitted (cancel across both arms)."""
    sql = """
        SELECT s.id AS signal_id, s.scanner_name, s.ticker, s.signal_timestamp,
               s.signal_date, s.signal_type, s.entry_price::float8 AS entry_price,
               s.composite_score::float8 AS composite_score,
               m.relative_volume::float8     AS relative_volume,
               m.daily_range_percent::float8 AS daily_range_percent,
               m.rs_percentile::float8       AS rs_percentile,
               m.rs_trend                    AS rs_trend,
               m.tv_overall_score::float8    AS tv_overall_score,
               m.rsi_14::float8              AS rsi_14,
               m.atr_14::float8              AS atr_14,
               m.ema_20::float8              AS ema_20,
               m.price_change_5d::float8     AS price_change_5d,
               m.pct_from_52w_high::float8   AS pct_from_52w_high
        FROM stock_scanner_signals s
        LEFT JOIN stock_screening_metrics m
               ON m.ticker = s.ticker AND m.calculation_date = s.signal_date
        WHERE s.signal_type = 'BUY'
        ORDER BY s.signal_timestamp
    """
    df = fetch_df(conn, sql)
    if df.empty:
        return df
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"]).dt.tz_localize(None)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["family"] = df["scanner_name"].map(SCANNER_FAMILY).fillna(DEFAULT_FAMILY)
    return df


def core_score(r: pd.Series) -> float:
    """Reconstructed *core* (screening_metrics-only) subset of the live daytrade
    candidate_score (route.ts ~226-310). Premarket/intraday/news/daq terms omitted
    — they are identical across both arms so they cancel in the grading delta."""
    def g(v, default=0.0):
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    rvol = g(r["relative_volume"])
    rng = g(r["daily_range_percent"])
    rs = g(r["rs_percentile"])
    tv = g(r["tv_overall_score"])
    rsi = g(r["rsi_14"])
    atr = g(r["atr_14"])
    ema20 = g(r["ema_20"])
    entry = g(r["entry_price"])
    pc5 = g(r["price_change_5d"])
    pfh = g(r["pct_from_52w_high"], -100.0)

    score = 0.0
    score += 0.30 * min(rvol / 3.0 * 100, 100)
    score += 0.14 * min(rng / 5.0 * 100, 100) * min(rvol, 1.0)
    score += 0.16 * rs
    score += 0.12 * ((tv + 100) / 2.0)
    # entry-not-extended: 100 at/below EMA20, decaying ~20pts per ATR above it
    if atr > 0 and ema20 > 0 and entry > 0:
        score += 0.08 * max(0.0, 100 - 20 * max((entry - ema20) / atr, 0.0))
    else:
        score += 0.08 * 60  # neutral default (mirrors route.ts warmup fallback)
    # rs_trend tilt
    trend = r["rs_trend"]
    score += 5 if trend == "improving" else (-12 if trend == "deteriorating" else 0)
    # RSI overbought penalty
    score += -12 if rsi > 80 else (-6 if rsi > 75 else 0)
    # 5d extension penalty
    score += -4 if pc5 > 25 else 0
    # near-52w-high breakout bonus
    score += 3 if pfh >= -3 else 0
    return score


# ---------------------------------------------------------------------------
# 2. Outcome cache — day-trade sim per signal (pess + opt BE), with exit_time
# ---------------------------------------------------------------------------

def build_outcome_cache(df: pd.DataFrame, candles_by_ticker, daily_by_ticker) -> pd.DataFrame:
    """Run the validated sim engine on every signal. Returns per-signal
    pnl/outcome under pessimistic and optimistic BE plus the exit_time (used for
    walk-forward resolution gating)."""
    rows = []
    n = len(df)
    for i, (_, sig) in enumerate(df.iterrows()):
        base = str(sig["ticker"]).split(".")[0]
        ts = sig["signal_timestamp"]
        atr_pct = atr_pct_before(daily_by_ticker, base, ts)
        bars, entry = bars_after(candles_by_ticker, base, ts)
        rec = {"signal_id": sig["signal_id"], "exit_time": pd.NaT,
               "pnl_pess": np.nan, "out_pess": "no_data",
               "pnl_opt": np.nan, "out_opt": "no_data", "atr_pct": atr_pct}
        if not bars.empty and entry > 0:
            bracket = build_bracket(entry, None)  # fixed 3/5 (auto-trader default)
            pess = walk_bars(entry, bracket, bars)
            opt = walk_bars(entry, bracket, bars, optimistic_be=True)
            rec["pnl_pess"], rec["out_pess"] = pess["pnl_pct"], pess["outcome"]
            rec["pnl_opt"], rec["out_opt"] = opt["pnl_pct"], opt["outcome"]
            hb = int(pess.get("hold_bars") or 0)
            if 1 <= hb <= len(bars):
                rec["exit_time"] = pd.Timestamp(bars.iloc[hb - 1]["timestamp"])
        rows.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"  ...simulated {i + 1:,}/{n:,} signals")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Walk-forward edge prior  (RESOLVED signals only — gate #2)
# ---------------------------------------------------------------------------

def edge_prior_asof(cache_resolved: pd.DataFrame, date_d: pd.Timestamp, pnl_col: str):
    """Empirical-Bayes shrunk per-scanner expectancy from signals RESOLVED
    (exit_time < date_d). Returns (edge_by_scanner_centered_on_pool, pool_mean)."""
    sub = cache_resolved[cache_resolved["exit_time"] < date_d]
    sub = sub[sub[pnl_col].notna()]
    if sub.empty:
        return {}, 0.0
    pool_mean = float(sub[pnl_col].mean())
    edges = {}
    for scn, grp in sub.groupby("scanner_name"):
        nse = len(grp)
        sample = float(grp[pnl_col].mean())
        shrunk = (nse * sample + EDGE_SHRINK_K * pool_mean) / (nse + EDGE_SHRINK_K)
        edges[scn] = shrunk - pool_mean   # centered: neutral scanners ~0
    return edges, pool_mean


def confluence_counts(pool: pd.DataFrame) -> dict:
    """Per-ticker count of DISTINCT scanner families flagging it in this date's pool."""
    return pool.groupby("ticker")["family"].nunique().to_dict()


# ---------------------------------------------------------------------------
# 4. Selection-set metrics
# ---------------------------------------------------------------------------

def set_metrics(pnls: np.ndarray) -> dict:
    pnls = pnls[~np.isnan(pnls)]
    if len(pnls) == 0:
        return {"n": 0, "expectancy": np.nan, "hit": np.nan, "pf": np.nan}
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    pf = (wins / losses) if losses > 0 else (9.99 if wins > 0 else np.nan)
    return {"n": len(pnls), "expectancy": float(pnls.mean()),
            "hit": float((pnls > 0).mean() * 100), "pf": float(pf)}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5:
        return np.nan
    ra = pd.Series(a[m]).rank().values
    rb = pd.Series(b[m]).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------

def run(conn):
    print("=" * 74)
    print("CROSS-SCANNER GRADING — HELD-OUT WALK-FORWARD REPLAY")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Train<= {TRAIN_END.date()} | Eval>= {EVAL_START.date()}  (verdict = eval only)")
    print("=" * 74)

    df = load_signals(conn)
    if df.empty:
        print("No BUY signals."); return
    df["core"] = df.apply(core_score, axis=1)
    have_core = df["relative_volume"].notna() | df["rs_percentile"].notna()
    print(f"\nLoaded {len(df):,} BUY signals; {int(have_core.sum()):,} have screening_metrics "
          f"(replayable). Date range {df['signal_date'].min().date()} -> {df['signal_date'].max().date()}.")
    df = df[have_core].copy()

    # Cross-family co-occurrence: is the confluence lever even exercisable here?
    fam_dist = (df.groupby(["signal_date", "ticker"])["family"].nunique()
                .value_counts().sort_index())
    total_td = int(fam_dist.sum())
    multi_td = int(fam_dist[fam_dist.index >= 2].sum()) if (fam_dist.index >= 2).any() else 0
    print(f"\nCross-family co-occurrence (confluence lever exercisability):")
    for k, v in fam_dist.items():
        print(f"    {int(k)} distinct family(ies): {int(v):,} ticker-dates")
    print(f"    => only {multi_td}/{total_td} ({multi_td/total_td*100:.1f}%) ticker-dates have ANY "
          f"cross-family agreement — confluence is structurally sparse in this scanner set.")

    print("\nPre-loading candles + simulating day-trade outcomes (pess + opt BE)...")
    candles_by_ticker = build_candle_index(conn)
    daily_by_ticker = build_daily_index(conn)
    cache = build_outcome_cache(df, candles_by_ticker, daily_by_ticker)
    df = df.merge(cache, on="signal_id", how="left")
    resolved = df[df["exit_time"].notna()].copy()
    print(f"  Simulated. {len(resolved):,}/{len(df):,} signals resolved (have outcome).")

    # --- TRAIN: set the confluence point magnitude, then FREEZE for eval -------
    train = df[df["signal_date"] <= TRAIN_END].copy()
    conf_pts_schedule = _fit_confluence_on_train(train)
    print(f"\n[TRAIN] confluence point schedule (frozen for eval): {conf_pts_schedule}")

    # --- build per-date augmented scores over the FULL period ------------------
    cache_for_prior = df[["scanner_name", "exit_time", "pnl_pess", "pnl_opt"]].copy()

    def score_pool(pool: pd.DataFrame, date_d: pd.Timestamp, be: str) -> pd.DataFrame:
        pnl_col = "pnl_pess" if be == "pess" else "pnl_opt"
        edges, _ = edge_prior_asof(cache_for_prior, date_d, pnl_col)
        fam_counts = confluence_counts(pool)
        out = pool.copy()
        out["edge_term"] = out["scanner_name"].map(
            lambda s: float(np.clip(EDGE_SCALE_PTS * edges.get(s, 0.0), -EDGE_CLIP_PTS, EDGE_CLIP_PTS)))
        out["conf_term"] = out["ticker"].map(
            lambda t: conf_pts_schedule.get(min(fam_counts.get(t, 1), 3), 0.0))
        out["score_base"] = out["core"]
        out["score_aug"] = out["core"] + out["edge_term"] + out["conf_term"]
        return out

    # --- EVAL: held-out window -------------------------------------------------
    print("\n" + "=" * 74)
    print("EVAL (held-out): selected-set day-trade outcomes, baseline vs augmented")
    print("=" * 74)
    eval_dates = sorted(df[df["signal_date"] >= EVAL_START]["signal_date"].unique())
    delta_exp = {N: {} for N in TOP_N_GRID}  # delta_exp[N][be] = augmented-baseline expectancy
    for be in ("pess", "opt"):
        pnl_col = "pnl_pess" if be == "pess" else "pnl_opt"
        print(f"\n--- BE model: {be.upper()} ---")
        print(f"  {'N':>4}  {'arm':>9}  {'picks':>6}  {'expectancy%':>12}  {'hit%':>7}  {'PF':>6}")
        for N in TOP_N_GRID:
            base_pnls, aug_pnls = [], []
            for d in eval_dates:
                pool = df[df["signal_date"] == d]
                if pool.empty:
                    continue
                scored = score_pool(pool, pd.Timestamp(d), be)
                # one row per ticker (trade a ticker once): keep max score per arm
                base_pick = (scored.sort_values("score_base", ascending=False)
                             .drop_duplicates("ticker").head(N))
                aug_pick = (scored.sort_values("score_aug", ascending=False)
                            .drop_duplicates("ticker").head(N))
                base_pnls.extend(base_pick[pnl_col].tolist())
                aug_pnls.extend(aug_pick[pnl_col].tolist())
            mb = set_metrics(np.array(base_pnls, dtype=float))
            ma = set_metrics(np.array(aug_pnls, dtype=float))
            for arm, m in (("baseline", mb), ("augmented", ma)):
                print(f"  {N:>4}  {arm:>9}  {m['n']:>6}  {m['expectancy']:>12.3f}  "
                      f"{m['hit']:>7.1f}  {m['pf']:>6.2f}")
            if not np.isnan(mb["expectancy"]) and not np.isnan(ma["expectancy"]):
                d_exp = ma["expectancy"] - mb["expectancy"]
                delta_exp[N][be] = d_exp
                print(f"  {'':>4}  {'DELTA':>9}  {'':>6}  {d_exp:>12.3f}  "
                      f"{ma['hit']-mb['hit']:>7.1f}  {ma['pf']-mb['pf']:>6.2f}")

    # Automatic pess/opt band verdict per N (gate #4)
    print(f"\n  BAND VERDICT per N (sign must agree across BE; |delta| must exceed the")
    print(f"  pess->opt band width, else UNCERTAIN):")
    print(f"  {'N':>4}  {'delta_pess':>11}  {'delta_opt':>10}  {'band_w':>7}  {'verdict':>12}")
    for N in TOP_N_GRID:
        dp, do = delta_exp[N].get("pess"), delta_exp[N].get("opt")
        if dp is None or do is None:
            print(f"  {N:>4}  {'N/A':>11}  {'N/A':>10}  {'N/A':>7}  {'N/A':>12}"); continue
        band = abs(do - dp)
        sign_ok = (dp > 0) == (do > 0)
        conservative = min(abs(dp), abs(do))
        verdict = "POSITIVE" if (sign_ok and dp > 0 and conservative > band) else (
            "neg/flip" if not sign_ok or dp <= 0 else "UNCERTAIN")
        print(f"  {N:>4}  {dp:>+11.3f}  {do:>+10.3f}  {band:>7.3f}  {verdict:>12}")

    # --- TIE-BREAKER: rank-correlation + within-decile incremental lift --------
    print("\n" + "=" * 74)
    print("TIE-BREAKER (verdict source): does grading add information beyond core?")
    print("=" * 74)
    eval_pool = _score_full_eval(df, EVAL_START, score_pool)
    verdict = _print_tiebreaker_and_verdict(df, eval_pool)

    # --- robustness: score-percentile gate (proxy for live score>=65) ----------
    print("\n" + "=" * 74)
    print(f"ROBUSTNESS: delta on top-{int((1-SCORE_GATE_PCTL)*100)}% of core score (N={HEADLINE_N}, pess)")
    print("=" * 74)
    _print_gated_robustness(df, score_pool)

    _print_caveats(verdict)


def _fit_confluence_on_train(train: pd.DataFrame) -> dict:
    """Confluence magnitude = train-measured outcome lift of multi-family names,
    converted to score points and FROZEN. Tuned on train only (gate #3)."""
    if train.empty:
        return {1: 0.0, 2: 0.0, 3: 0.0}
    # per (date, ticker) family count within each train date's pool
    fam_by_dt = (train.groupby(["signal_date", "ticker"])["family"].nunique()
                 .rename("famcount").reset_index())
    t = train.merge(fam_by_dt, on=["signal_date", "ticker"], how="left")
    single = t[t["famcount"] <= 1]["pnl_pess"].dropna()
    multi2 = t[t["famcount"] == 2]["pnl_pess"].dropna()
    multi3 = t[t["famcount"] >= 3]["pnl_pess"].dropna()
    base = single.mean() if len(single) else 0.0
    def pts(grp):
        if len(grp) < 20:
            return 0.0   # too few to trust -> no bonus
        return float(np.clip((grp.mean() - base) * EDGE_SCALE_PTS, 0.0, CONF_CLIP_PTS))
    return {1: 0.0, 2: pts(multi2), 3: pts(multi3)}


def _score_full_eval(df, eval_start, score_pool) -> pd.DataFrame:
    """Score every eval-window signal (pess) for the tie-breaker analysis."""
    parts = []
    for d in sorted(df[df["signal_date"] >= eval_start]["signal_date"].unique()):
        pool = df[df["signal_date"] == d]
        if not pool.empty:
            parts.append(score_pool(pool, pd.Timestamp(d), "pess"))
    return pd.concat(parts) if parts else pd.DataFrame()


def _print_tiebreaker_and_verdict(df, eval_pool) -> str:
    if eval_pool.empty:
        print("  No eval signals."); return "NO_DATA"
    ep = eval_pool[eval_pool["pnl_pess"].notna()].copy()
    sb = spearman(ep["score_base"].values, ep["pnl_pess"].values)
    sa = spearman(ep["score_aug"].values, ep["pnl_pess"].values)
    grading = (ep["edge_term"] + ep["conf_term"]).values
    print(f"\n  Eval pool n={len(ep)}  (outcome=pessimistic day-trade pnl%)")
    print(f"  Spearman(core score, outcome):       {sb:+.4f}")
    print(f"  Spearman(augmented score, outcome):  {sa:+.4f}")
    print(f"  -> augmented {'IMPROVES' if (not np.isnan(sa) and not np.isnan(sb) and sa > sb) else 'does NOT improve'} rank-correlation")

    # within-decile incremental lift: does the grading component predict outcome
    # AFTER controlling for the core score?
    ep = ep.assign(dec=pd.qcut(ep["score_base"].rank(method="first"), 10, labels=False))
    corrs = []
    for _, g in ep.groupby("dec"):
        c = spearman(g["edge_term"].values + g["conf_term"].values, g["pnl_pess"].values)
        if not np.isnan(c):
            corrs.append(c)
    mean_incr = float(np.mean(corrs)) if corrs else np.nan
    print(f"  Mean within-core-decile Spearman(grading term, outcome): {mean_incr:+.4f}  "
          f"({len(corrs)} deciles)")
    incr_positive = (not np.isnan(mean_incr)) and mean_incr > 0
    print(f"  -> grading carries {'POSITIVE' if incr_positive else 'NO / NEGATIVE'} "
          f"incremental information beyond core score")
    return "INCR_POSITIVE" if incr_positive else "INCR_NULL"


def _print_gated_robustness(df, score_pool):
    base_pnls, aug_pnls = [], []
    for d in sorted(df[df["signal_date"] >= EVAL_START]["signal_date"].unique()):
        pool = df[df["signal_date"] == d]
        if pool.empty:
            continue
        scored = score_pool(pool, pd.Timestamp(d), "pess")
        thresh = scored["core"].quantile(SCORE_GATE_PCTL)
        scored = scored[scored["core"] >= thresh]
        if scored.empty:
            continue
        base_pnls.extend(scored.sort_values("score_base", ascending=False)
                         .drop_duplicates("ticker").head(HEADLINE_N)["pnl_pess"].tolist())
        aug_pnls.extend(scored.sort_values("score_aug", ascending=False)
                        .drop_duplicates("ticker").head(HEADLINE_N)["pnl_pess"].tolist())
    mb = set_metrics(np.array(base_pnls, dtype=float))
    ma = set_metrics(np.array(aug_pnls, dtype=float))
    print(f"  baseline : n={mb['n']}  expectancy={mb['expectancy']:.3f}%  hit={mb['hit']:.1f}%  PF={mb['pf']:.2f}")
    print(f"  augmented: n={ma['n']}  expectancy={ma['expectancy']:.3f}%  hit={ma['hit']:.1f}%  PF={ma['pf']:.2f}")
    if not np.isnan(mb["expectancy"]) and not np.isnan(ma["expectancy"]):
        print(f"  DELTA expectancy: {ma['expectancy']-mb['expectancy']:+.3f}%")


def _print_caveats(verdict: str):
    print("\n" + "=" * 74)
    print("VERDICT & CAVEATS")
    print("=" * 74)
    print(f"""
  TIE-BREAKER (pre-registered as the verdict source): {verdict}
    INCR_POSITIVE = grading adds rank-information beyond the core score on the
    held-out window; INCR_NULL = it does not (do not ship the term).

  Read the EVAL table the right way (gate #4): trust a positive expectancy delta
  ONLY if its sign is consistent across the PESS and OPT BE models AND its
  magnitude exceeds the pess->opt band width. If the delta flips sign or is
  smaller than the band, the honest verdict is UNCERTAIN.

  POPULATION GAP (gate #1): live VWAP/spread/score>=65/intraday-RVOL gates are
  not reconstructable; both arms omit them, so this measures the ALL-SIGNALS
  RANKING, not the live traded-set P&L. The score-percentile robustness block is
  a rough proxy for the score>=65 gate.

  WINDOW: in-sample, ~Dec-2025 -> Jun-2026 (single-ish regime). Several scanners
  launched 2026-05-26; the walk-forward edge prior correctly shrinks them to
  neutral until they accumulate resolved outcomes. Do not over-read a single
  positive window — this project's failure mode is in-sample winners regressing OOS.
""")
    print("Script: stock-scanner/app/stock_scanner/analysis/grading_replay.py")


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
