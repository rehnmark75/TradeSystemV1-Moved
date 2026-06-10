"""
Held-out replay: does SMC price-action STRUCTURE improve daytrade selection?
=============================================================================
The SMC fields (smc_confluence_score, premium_discount_zone/zone_position,
smc_bias/smc_trend, last_bos_type) are computed nightly into
stock_screening_metrics but are UNUSED by the live daytrade candidate ranking.
This tests whether layering a pre-registered SMC term on the reconstructed core
candidate_score improves day-trade-simulated selection outcomes.

Reuses the validated infra from daytrade_edge_sim.py (sim engine) and
grading_replay.py (core_score, outcome cache, metrics, band verdict).

Usage (read-only):
    docker exec stock-scanner python -m stock_scanner.analysis.smc_replay

-------------------------------------------------------------------------------
PRE-REGISTERED (frozen before eval):
  - SMC feature SIGNS are theory-driven for a BUY and fixed (no sign-fishing on
    eval): discount zone favorable, premium unfavorable; bullish smc_bias
    favorable; bullish last_bos favorable; higher confluence favorable.
  - Feature standardization stats (mean/std) and the directional composite are
    fit on the TRAIN split only, then frozen for eval.
  - SMC_SCALE_PTS is a pre-registered constant.
  - VERDICT SOURCE = within-core-decile incremental Spearman (tie-breaker);
    top-N list-reconstruction is presentation. Outcomes under pess + opt BE; a
    delta inside the band -> UNCERTAIN.

CAVEATS (same as grading_replay): live VWAP/spread/score>=65/intraday gates are
not reconstructable (both arms omit them) -> this measures the ALL-SIGNALS
RANKING, not live traded-set P&L. In-sample, single-ish regime, ~Dec25->Jun26.
SMC features are contemporaneous (as-of signal close, before next-session entry)
so NO walk-forward gating is needed and there is no look-ahead.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import (
    get_conn, fetch_df, build_candle_index, build_daily_index,
)
from stock_scanner.analysis.grading_replay import (
    core_score, build_outcome_cache, set_metrics, spearman,
)

# ---------------------------------------------------------------------------
# PRE-REGISTERED PARAMETERS
# ---------------------------------------------------------------------------
TRAIN_END = pd.Timestamp("2026-03-31")
EVAL_START = pd.Timestamp("2026-04-01")
SMC_SCALE_PTS = 4.0      # candidate_score points per unit standardized SMC composite
SMC_CLIP_PTS = 8.0       # bound the SMC term
TOP_N_GRID = [3, 5, 10]
SCORE_GATE_PCTL = 0.50   # robustness: re-test on top-half of core score

# Theory-driven feature signs for a BUY (pre-registered, NOT fit):
BIAS_SIGN = {"Bullish": 1.0, "Bearish": -1.0, "Neutral": 0.0}
BOS_SIGN = {"Bullish": 1.0, "Bearish": -1.0}


def load_signals_smc(conn) -> pd.DataFrame:
    sql = """
        SELECT s.id AS signal_id, s.scanner_name, s.ticker, s.signal_timestamp,
               s.signal_date, s.entry_price::float8 AS entry_price,
               m.relative_volume::float8 AS relative_volume,
               m.daily_range_percent::float8 AS daily_range_percent,
               m.rs_percentile::float8 AS rs_percentile, m.rs_trend AS rs_trend,
               m.tv_overall_score::float8 AS tv_overall_score,
               m.rsi_14::float8 AS rsi_14, m.atr_14::float8 AS atr_14,
               m.ema_20::float8 AS ema_20, m.price_change_5d::float8 AS price_change_5d,
               m.pct_from_52w_high::float8 AS pct_from_52w_high,
               -- SMC structure features
               m.smc_confluence_score::float8 AS smc_conf,
               m.zone_position::float8 AS zone_position,
               m.premium_discount_zone AS pd_zone,
               m.smc_bias AS smc_bias, m.last_bos_type AS bos_type
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
    # zone_position can carry 'NaN' floats -> coerce, fill neutral(50)
    df["zone_position"] = pd.to_numeric(df["zone_position"], errors="coerce")
    return df


def raw_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Theory-signed raw features for a BUY (higher = more favorable)."""
    out = pd.DataFrame(index=df.index)
    # discount favorable: 0 at equilibrium(50), + in discount(<50), - in premium(>50)
    out["f_zone"] = (50.0 - df["zone_position"].fillna(50.0))
    # cleaner structure favorable
    out["f_conf"] = df["smc_conf"]
    # directional alignment with a long
    out["f_bias"] = df["smc_bias"].map(BIAS_SIGN).fillna(0.0)
    out["f_bos"] = df["bos_type"].map(BOS_SIGN).fillna(0.0)
    return out


def main():
    try:
        conn = get_conn()
    except Exception as e:
        print(f"ERROR: DB connect failed: {e}", file=sys.stderr); sys.exit(1)
    try:
        run(conn)
    finally:
        conn.close()


def run(conn):
    print("=" * 74)
    print("SMC STRUCTURE — HELD-OUT REPLAY (does SMC improve daytrade selection?)")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Train<= {TRAIN_END.date()} | Eval>= {EVAL_START.date()}  (verdict = eval only)")
    print("=" * 74)

    df = load_signals_smc(conn)
    if df.empty:
        print("No BUY signals."); return
    df["core"] = df.apply(core_score, axis=1)
    have = df["smc_conf"].notna() & df["pd_zone"].notna() & (
        df["relative_volume"].notna() | df["rs_percentile"].notna())
    print(f"\nLoaded {len(df):,} BUY signals; {int(have.sum()):,} have core + SMC fields "
          f"(replayable). Range {df['signal_date'].min().date()} -> {df['signal_date'].max().date()}.")
    df = df[have].copy()

    print("\nPre-loading candles + simulating day-trade outcomes (pess + opt BE)...")
    cb, db_ = build_candle_index(conn), build_daily_index(conn)
    cache = build_outcome_cache(df, cb, db_)
    df = df.merge(cache, on="signal_id", how="left")
    df = df[df["pnl_pess"].notna()].copy()
    print(f"  {len(df):,} signals with day-trade outcome.")

    # --- raw features + TRAIN-fit standardization (frozen for eval) -----------
    raw = raw_smc_features(df)
    train_mask = df["signal_date"] <= TRAIN_END
    stats = {}
    for c in ["f_zone", "f_conf", "f_bias", "f_bos"]:
        tr = raw.loc[train_mask, c]
        mu, sd = float(tr.mean()), float(tr.std() or 1.0)
        stats[c] = (mu, sd)
        df[c + "_z"] = (raw[c] - mu) / (sd if sd != 0 else 1.0)
    # Directional composite (equal-weight, theory-signed). Confluence is a
    # non-directional quality signal -> include it too (cleaner structure better).
    df["smc_feat"] = df[["f_zone_z", "f_conf_z", "f_bias_z", "f_bos_z"]].mean(axis=1)
    df["smc_term"] = (SMC_SCALE_PTS * df["smc_feat"]).clip(-SMC_CLIP_PTS, SMC_CLIP_PTS)
    df["score_base"] = df["core"]
    df["score_aug"] = df["core"] + df["smc_term"]

    # --- TRAIN diagnostic: does ANY SMC feature predict day-trade outcome? ----
    print("\n" + "=" * 74)
    print("TRAIN diagnostic: per-feature Spearman vs day-trade pnl (pess) [in-sample]")
    print("=" * 74)
    tr = df[train_mask]
    print("  (theory expects POSITIVE corr for each, since features are BUY-signed)")
    for label, col in [("zone(discount+)", "f_zone_z"), ("confluence", "f_conf_z"),
                       ("smc_bias", "f_bias_z"), ("last_bos", "f_bos_z"),
                       ("COMPOSITE", "smc_feat")]:
        sp = spearman(tr[col].values, tr["pnl_pess"].values)
        flag = "" if np.isnan(sp) else (" <- theory holds" if sp > 0.01 else
                                        (" <- CONTRADICTS theory" if sp < -0.01 else " <- ~flat"))
        print(f"    {label:16s}: Spearman={sp:+.4f}{flag}")

    # --- EVAL: selection-set outcomes, baseline vs SMC-augmented --------------
    print("\n" + "=" * 74)
    print("EVAL (held-out): selected-set day-trade outcomes, baseline vs SMC-augmented")
    print("=" * 74)
    eval_dates = sorted(df[df["signal_date"] >= EVAL_START]["signal_date"].unique())
    delta_exp = {N: {} for N in TOP_N_GRID}
    for be in ("pess", "opt"):
        pnl_col = "pnl_pess" if be == "pess" else "pnl_opt"
        print(f"\n--- BE model: {be.upper()} ---")
        print(f"  {'N':>4}  {'arm':>9}  {'picks':>6}  {'expectancy%':>12}  {'hit%':>7}  {'PF':>6}")
        for N in TOP_N_GRID:
            bp, ap = [], []
            for d in eval_dates:
                pool = df[df["signal_date"] == d]
                if pool.empty:
                    continue
                bp.extend(pool.sort_values("score_base", ascending=False)
                          .drop_duplicates("ticker").head(N)[pnl_col].tolist())
                ap.extend(pool.sort_values("score_aug", ascending=False)
                          .drop_duplicates("ticker").head(N)[pnl_col].tolist())
            mb, ma = set_metrics(np.array(bp, float)), set_metrics(np.array(ap, float))
            for arm, m in (("baseline", mb), ("SMC-aug", ma)):
                print(f"  {N:>4}  {arm:>9}  {m['n']:>6}  {m['expectancy']:>12.3f}  "
                      f"{m['hit']:>7.1f}  {m['pf']:>6.2f}")
            if not np.isnan(mb["expectancy"]) and not np.isnan(ma["expectancy"]):
                de = ma["expectancy"] - mb["expectancy"]
                delta_exp[N][be] = de
                print(f"  {'':>4}  {'DELTA':>9}  {'':>6}  {de:>12.3f}  "
                      f"{ma['hit']-mb['hit']:>7.1f}  {ma['pf']-mb['pf']:>6.2f}")

    print(f"\n  BAND VERDICT per N (sign agree across BE AND |delta|>band, else UNCERTAIN):")
    print(f"  {'N':>4}  {'delta_pess':>11}  {'delta_opt':>10}  {'band_w':>7}  {'verdict':>12}")
    for N in TOP_N_GRID:
        dp, do = delta_exp[N].get("pess"), delta_exp[N].get("opt")
        if dp is None or do is None:
            print(f"  {N:>4}  {'N/A':>11}  {'N/A':>10}  {'N/A':>7}  {'N/A':>12}"); continue
        band = abs(do - dp)
        sign_ok = (dp > 0) == (do > 0)
        verdict = "POSITIVE" if (sign_ok and dp > 0 and min(abs(dp), abs(do)) > band) else (
            "neg/flip" if not sign_ok or dp <= 0 else "UNCERTAIN")
        print(f"  {N:>4}  {dp:>+11.3f}  {do:>+10.3f}  {band:>7.3f}  {verdict:>12}")

    # --- TIE-BREAKER: incremental info beyond core score ----------------------
    print("\n" + "=" * 74)
    print("TIE-BREAKER (verdict source): SMC info beyond core score, held-out")
    print("=" * 74)
    ev = df[df["signal_date"] >= EVAL_START].copy()
    sb = spearman(ev["score_base"].values, ev["pnl_pess"].values)
    sa = spearman(ev["score_aug"].values, ev["pnl_pess"].values)
    print(f"\n  Eval pool n={len(ev)}  (outcome = pessimistic day-trade pnl%)")
    print(f"  Spearman(core, outcome):          {sb:+.4f}")
    print(f"  Spearman(core+SMC, outcome):      {sa:+.4f}")
    print(f"  Spearman(SMC term, outcome):      {spearman(ev['smc_term'].values, ev['pnl_pess'].values):+.4f}")
    ev = ev.assign(dec=pd.qcut(ev["score_base"].rank(method="first"), 10, labels=False))
    corrs = [spearman(g["smc_term"].values, g["pnl_pess"].values)
             for _, g in ev.groupby("dec")]
    corrs = [c for c in corrs if not np.isnan(c)]
    mean_incr = float(np.mean(corrs)) if corrs else np.nan
    incr_pos = (not np.isnan(mean_incr)) and mean_incr > 0
    print(f"  Mean within-core-decile Spearman(SMC term, outcome): {mean_incr:+.4f} ({len(corrs)} deciles)")
    print(f"  -> SMC carries {'POSITIVE' if incr_pos else 'NO / NEGATIVE'} incremental information")

    # --- robustness on top-half core score ------------------------------------
    print("\n" + "=" * 74)
    print(f"ROBUSTNESS: delta on top-{int((1-SCORE_GATE_PCTL)*100)}% of core score (N=5, pess)")
    print("=" * 74)
    bp, ap = [], []
    for d in eval_dates:
        pool = df[df["signal_date"] == d]
        if pool.empty:
            continue
        pool = pool[pool["core"] >= pool["core"].quantile(SCORE_GATE_PCTL)]
        if pool.empty:
            continue
        bp.extend(pool.sort_values("score_base", ascending=False).drop_duplicates("ticker").head(5)["pnl_pess"].tolist())
        ap.extend(pool.sort_values("score_aug", ascending=False).drop_duplicates("ticker").head(5)["pnl_pess"].tolist())
    mb, ma = set_metrics(np.array(bp, float)), set_metrics(np.array(ap, float))
    print(f"  baseline: n={mb['n']} exp={mb['expectancy']:.3f}% hit={mb['hit']:.1f}% PF={mb['pf']:.2f}")
    print(f"  SMC-aug : n={ma['n']} exp={ma['expectancy']:.3f}% hit={ma['hit']:.1f}% PF={ma['pf']:.2f}")
    if not np.isnan(mb["expectancy"]) and not np.isnan(ma["expectancy"]):
        print(f"  DELTA expectancy: {ma['expectancy']-mb['expectancy']:+.3f}%")

    print("\n" + "=" * 74)
    print("VERDICT NOTES")
    print("=" * 74)
    print(f"""
  - The TRAIN per-feature diagnostic shows whether ANY SMC structure aspect
    carries day-trade signal at all (theory expects positive). If all are ~flat,
    SMC structure has no day-trade edge here regardless of the eval table.
  - Verdict source = within-core-decile incremental Spearman (above). Trust the
    EVAL delta only if its sign is consistent across PESS/OPT and exceeds the band.
  - POPULATION GAP: all-signals ranking, NOT live traded-set P&L (VWAP/score>=65
    gates not reconstructable). Single-ish in-sample regime; don't over-read.
  Script: stock-scanner/app/stock_scanner/analysis/smc_replay.py
""")


if __name__ == "__main__":
    main()
