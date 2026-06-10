"""
Held-out replay: does SMC structure improve SWING selection? (its native horizon)
=================================================================================
SMC (premium/discount, ~24d BOS, confluence) is a swing-horizon construct. The
day-trade test (smc_replay.py) found no day-trade edge — expected, given the
horizon mismatch. This re-points the SAME pre-registered SMC term at the SWING
horizon it is designed for:
  - Baseline ranker = the SWING candidate_score (route.ts swing mode:
    0.55*RS + 0.45*TV + rs_trend), NOT the daytrade core.
  - Outcomes = a NATIVE-BRACKET swing sim: the signal's own stop_loss /
    take_profit_1, NO day-trade break-even, horizon-truncated bars. Run at two
    horizons (10 / 20 trading days) as a robustness band.
  - Cross-checked against the stored realized_pnl_pct (the system's own swing
    resolver) as an independent swing label.

Reuses validated infra: walk_bars / bars_after / candle builders
(daytrade_edge_sim.py), set_metrics / spearman (grading_replay.py),
raw_smc_features + pre-registered signs (smc_replay.py).

Usage (read-only): docker exec stock-scanner python -m stock_scanner.analysis.swing_replay

PRE-REGISTERED: SMC feature signs (BUY-favorable: discount/bullish-bias/bullish-
BOS/high-confluence) and SMC_SCALE_PTS frozen; standardization fit on train only;
verdict source = within-swing-core-decile incremental Spearman (tie-breaker);
robustness = sign-consistency across the two horizons. ALL-SIGNALS ranking (live
gates not reconstructable); in-sample, single-ish regime — don't over-read.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import (
    get_conn, fetch_df, build_candle_index, walk_bars, bars_after,
)
from stock_scanner.analysis.grading_replay import set_metrics, spearman
from stock_scanner.analysis.smc_replay import raw_smc_features

TRAIN_END = pd.Timestamp("2026-03-31")
EVAL_START = pd.Timestamp("2026-04-01")
SMC_SCALE_PTS = 4.0
SMC_CLIP_PTS = 8.0
TOP_N_GRID = [3, 5, 10]
SCORE_GATE_PCTL = 0.50
SWING_HORIZONS = [20, 10]   # trading days; primary first
BARS_PER_DAY = 6            # 1h bars per trading day


def load_signals_swing(conn) -> pd.DataFrame:
    sql = """
        SELECT s.id AS signal_id, s.scanner_name, s.ticker, s.signal_timestamp,
               s.signal_date, s.entry_price::float8 AS entry_price,
               s.stop_loss::float8 AS stop_loss, s.take_profit_1::float8 AS take_profit_1,
               s.realized_pnl_pct::float8 AS realized_pnl_pct, s.status,
               m.rs_percentile::float8 AS rs_percentile, m.rs_trend AS rs_trend,
               m.tv_overall_score::float8 AS tv_overall_score,
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
    df["zone_position"] = pd.to_numeric(df["zone_position"], errors="coerce")
    return df


def swing_core(r: pd.Series) -> float:
    """Reconstructed SWING candidate_score (route.ts swing mode)."""
    def g(v, d=0.0):
        return d if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    score = 0.55 * g(r["rs_percentile"]) + 0.45 * ((g(r["tv_overall_score"]) + 100) / 2.0)
    t = r["rs_trend"]
    score += 5 if t == "improving" else (-15 if t == "deteriorating" else 0)
    return score


def swing_outcome(entry: float, sl: float, tp: float, bars: pd.DataFrame, horizon_days: int):
    """Native-bracket swing sim: signal's own SL/TP, NO break-even, bars truncated
    to the horizon. Returns pnl_pct (or NaN)."""
    if entry <= 0 or sl <= 0 or tp <= 0 or sl >= entry or tp <= entry or bars.empty:
        return np.nan
    nb = horizon_days * BARS_PER_DAY
    res = walk_bars(entry, {"sl": sl, "tp": tp, "be_trigger_pct": 999.0},
                    bars.head(nb), be_enabled=False)
    return res["pnl_pct"]


def build_swing_cache(df, candles_by_ticker) -> pd.DataFrame:
    rows = []
    n = len(df)
    for i, (_, sig) in enumerate(df.iterrows()):
        base = str(sig["ticker"]).split(".")[0]
        bars, entry = bars_after(candles_by_ticker, base, sig["signal_timestamp"])
        rec = {"signal_id": sig["signal_id"]}
        for h in SWING_HORIZONS:
            rec[f"pnl_h{h}"] = swing_outcome(entry, float(sig["stop_loss"] or 0),
                                             float(sig["take_profit_1"] or 0), bars, h)
        rows.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"  ...simulated {i + 1:,}/{n:,} signals")
    return pd.DataFrame(rows)


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
    H = SWING_HORIZONS[0]  # primary horizon
    pcol = f"pnl_h{H}"
    print("=" * 74)
    print(f"SMC STRUCTURE — SWING-HORIZON REPLAY (native SL/TP, {H}d primary horizon)")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Train<= {TRAIN_END.date()} | Eval>= {EVAL_START.date()}  (verdict = eval only)")
    print("=" * 74)

    df = load_signals_swing(conn)
    if df.empty:
        print("No BUY signals."); return
    df["core"] = df.apply(swing_core, axis=1)
    have = (df["smc_conf"].notna() & df["pd_zone"].notna()
            & df["rs_percentile"].notna() & df["stop_loss"].notna() & df["take_profit_1"].notna())
    print(f"\nLoaded {len(df):,} BUY signals; {int(have.sum()):,} have swing-core + SMC + native "
          f"SL/TP. Range {df['signal_date'].min().date()} -> {df['signal_date'].max().date()}.")
    df = df[have].copy()

    print(f"\nPre-loading candles + simulating SWING outcomes at horizons {SWING_HORIZONS} (trading days)...")
    cb = build_candle_index(conn)
    cache = build_swing_cache(df, cb)
    df = df.merge(cache, on="signal_id", how="left")
    df = df[df[pcol].notna()].copy()
    print(f"  {len(df):,} signals with swing outcome (h{H}).")

    # --- SMC term: train-fit standardization, theory signs (frozen) -----------
    raw = raw_smc_features(df)
    tmask = df["signal_date"] <= TRAIN_END
    for c in ["f_zone", "f_conf", "f_bias", "f_bos"]:
        tr = raw.loc[tmask, c]
        mu, sd = float(tr.mean()), float(tr.std() or 1.0)
        df[c + "_z"] = (raw[c] - mu) / (sd if sd != 0 else 1.0)
    df["smc_feat"] = df[["f_zone_z", "f_conf_z", "f_bias_z", "f_bos_z"]].mean(axis=1)
    df["smc_term"] = (SMC_SCALE_PTS * df["smc_feat"]).clip(-SMC_CLIP_PTS, SMC_CLIP_PTS)
    df["score_base"] = df["core"]
    df["score_aug"] = df["core"] + df["smc_term"]

    # --- TRAIN diagnostic: does SMC predict SWING outcome? --------------------
    print("\n" + "=" * 74)
    print(f"TRAIN diagnostic: per-feature Spearman vs SWING pnl (h{H}) [in-sample]")
    print("=" * 74)
    tr = df[tmask]
    print("  (theory expects POSITIVE corr; features are BUY-signed)")
    for label, col in [("zone(discount+)", "f_zone_z"), ("confluence", "f_conf_z"),
                       ("smc_bias", "f_bias_z"), ("last_bos", "f_bos_z"),
                       ("COMPOSITE", "smc_feat")]:
        sp = spearman(tr[col].values, tr[pcol].values)
        flag = "" if np.isnan(sp) else (" <- theory holds" if sp > 0.01 else
                                        (" <- CONTRADICTS" if sp < -0.01 else " <- ~flat"))
        print(f"    {label:16s}: Spearman={sp:+.4f}{flag}")

    # --- EVAL: baseline vs SMC-augmented, primary horizon ---------------------
    print("\n" + "=" * 74)
    print(f"EVAL (held-out): selected-set SWING outcomes (h{H}), baseline vs SMC-aug")
    print("=" * 74)
    eval_dates = sorted(df[df["signal_date"] >= EVAL_START]["signal_date"].unique())
    print(f"  {'N':>4}  {'arm':>9}  {'picks':>6}  {'expectancy%':>12}  {'hit%':>7}  {'PF':>6}")
    for N in TOP_N_GRID:
        bp, ap = [], []
        for d in eval_dates:
            pool = df[df["signal_date"] == d]
            if pool.empty:
                continue
            bp.extend(pool.sort_values("score_base", ascending=False).drop_duplicates("ticker").head(N)[pcol].tolist())
            ap.extend(pool.sort_values("score_aug", ascending=False).drop_duplicates("ticker").head(N)[pcol].tolist())
        mb, ma = set_metrics(np.array(bp, float)), set_metrics(np.array(ap, float))
        for arm, m in (("baseline", mb), ("SMC-aug", ma)):
            print(f"  {N:>4}  {arm:>9}  {m['n']:>6}  {m['expectancy']:>12.3f}  {m['hit']:>7.1f}  {m['pf']:>6.2f}")
        if not np.isnan(mb["expectancy"]) and not np.isnan(ma["expectancy"]):
            print(f"  {'':>4}  {'DELTA':>9}  {'':>6}  {ma['expectancy']-mb['expectancy']:>12.3f}  "
                  f"{ma['hit']-mb['hit']:>7.1f}  {ma['pf']-mb['pf']:>6.2f}")

    # --- HORIZON robustness band: delta sign across horizons (N=HEADLINE 5) ----
    print(f"\n  HORIZON robustness — N=5 expectancy delta at each horizon (sign must agree):")
    for h in SWING_HORIZONS:
        hcol = f"pnl_h{h}"
        bp, ap = [], []
        for d in eval_dates:
            pool = df[df["signal_date"] == d]
            if pool.empty:
                continue
            bp.extend(pool.sort_values("score_base", ascending=False).drop_duplicates("ticker").head(5)[hcol].tolist())
            ap.extend(pool.sort_values("score_aug", ascending=False).drop_duplicates("ticker").head(5)[hcol].tolist())
        mb, ma = set_metrics(np.array(bp, float)), set_metrics(np.array(ap, float))
        de = (ma["expectancy"] - mb["expectancy"]) if not np.isnan(ma["expectancy"]) and not np.isnan(mb["expectancy"]) else float("nan")
        print(f"    h{h:>2}d: baseline exp={mb['expectancy']:+.3f}%  SMC-aug exp={ma['expectancy']:+.3f}%  DELTA={de:+.3f}%")

    # --- TIE-BREAKER + cross-check --------------------------------------------
    print("\n" + "=" * 74)
    print("TIE-BREAKER (verdict source): SMC info beyond swing-core, held-out")
    print("=" * 74)
    ev = df[df["signal_date"] >= EVAL_START].copy()
    print(f"\n  Eval pool n={len(ev)}  (primary outcome = swing pnl h{H})")
    print(f"  Spearman(swing-core, outcome):        {spearman(ev['score_base'].values, ev[pcol].values):+.4f}")
    print(f"  Spearman(swing-core+SMC, outcome):    {spearman(ev['score_aug'].values, ev[pcol].values):+.4f}")
    print(f"  Spearman(SMC term, outcome):          {spearman(ev['smc_term'].values, ev[pcol].values):+.4f}")
    ev2 = ev.assign(dec=pd.qcut(ev["score_base"].rank(method="first"), 10, labels=False))
    corrs = [spearman(g["smc_term"].values, g[pcol].values) for _, g in ev2.groupby("dec")]
    corrs = [c for c in corrs if not np.isnan(c)]
    mean_incr = float(np.mean(corrs)) if corrs else np.nan
    incr_pos = (not np.isnan(mean_incr)) and mean_incr > 0
    print(f"  Mean within-swing-core-decile Spearman(SMC term, outcome): {mean_incr:+.4f} ({len(corrs)} deciles)")
    print(f"  -> SMC carries {'POSITIVE' if incr_pos else 'NO / NEGATIVE'} incremental info (swing)")

    # Independent cross-check vs the system's own swing resolver
    cc = ev[ev["realized_pnl_pct"].notna()]
    if len(cc) >= 30:
        print(f"\n  CROSS-CHECK vs stored realized_pnl_pct (system swing label, n={len(cc)} closed):")
        print(f"    Spearman(SMC term, realized_pnl_pct): {spearman(cc['smc_term'].values, cc['realized_pnl_pct'].values):+.4f}")
        print(f"    Spearman(swing-core, realized_pnl_pct): {spearman(cc['score_base'].values, cc['realized_pnl_pct'].values):+.4f}")

    # --- robustness on top-half swing-core ------------------------------------
    print("\n" + "=" * 74)
    print(f"ROBUSTNESS: delta on top-{int((1-SCORE_GATE_PCTL)*100)}% of swing-core (N=5, h{H})")
    print("=" * 74)
    bp, ap = [], []
    for d in eval_dates:
        pool = df[df["signal_date"] == d]
        if pool.empty:
            continue
        pool = pool[pool["core"] >= pool["core"].quantile(SCORE_GATE_PCTL)]
        if pool.empty:
            continue
        bp.extend(pool.sort_values("score_base", ascending=False).drop_duplicates("ticker").head(5)[pcol].tolist())
        ap.extend(pool.sort_values("score_aug", ascending=False).drop_duplicates("ticker").head(5)[pcol].tolist())
    mb, ma = set_metrics(np.array(bp, float)), set_metrics(np.array(ap, float))
    print(f"  baseline: n={mb['n']} exp={mb['expectancy']:+.3f}% hit={mb['hit']:.1f}% PF={mb['pf']:.2f}")
    print(f"  SMC-aug : n={ma['n']} exp={ma['expectancy']:+.3f}% hit={ma['hit']:.1f}% PF={ma['pf']:.2f}")
    if not np.isnan(mb["expectancy"]) and not np.isnan(ma["expectancy"]):
        print(f"  DELTA expectancy: {ma['expectancy']-mb['expectancy']:+.3f}%")

    print("\n" + "=" * 74)
    print("VERDICT NOTES")
    print("=" * 74)
    print(f"""
  - This is SMC on its NATIVE (swing) horizon: native SL/TP brackets, no day-trade
    BE, {SWING_HORIZONS}-trading-day windows. Verdict source = within-swing-core-decile
    incremental Spearman; require the N=5 delta to keep its sign across both horizons.
  - Cross-check vs realized_pnl_pct is an INDEPENDENT swing label (system resolver,
    closed signals only) — agreement raises confidence; disagreement = caution.
  - ALL-SIGNALS ranking, in-sample, single-ish regime. A positive here is a
    candidate to validate OOS, not a ship signal.
  Script: stock-scanner/app/stock_scanner/analysis/swing_replay.py
""")


if __name__ == "__main__":
    main()
