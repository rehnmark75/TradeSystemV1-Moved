"""
SL/TP Calibration & Trailing Stop Study — Stock Side
=====================================================
Produces a data-driven calibration of stop-loss / take-profit (and trailing-stop
variants) for the stock auto-trader. Analysis only — read-only, no DB writes.

Tasks
-----
1. Realized holding-period distribution (signal-level vs broker-execution).
2. MFE / MAE distributions per ATR% bucket over the realized multi-day horizon.
3. Derive & validate SL/TP per bucket vs the current live rule.
4. Trailing-stop variants vs baseline.

Key design decisions
--------------------
* Horizon: 15 trading days primary (~105 1h bars). Also tested 10d/20d for
  stability on a common non-truncated set.
* "Eventual winner" label = terminal_pct > 0 (candle path at horizon end),
  NOT realized_pnl_pct (that is the swing-model close, unrelated to bracket).
* Uniform PF/WR classifier: every exit pnl > 0 → win, < 0 → loss, ~0 → flat.
  PF = sum_wins / |sum_losses|, expectancy = mean(all pnl), WR = n_pos/total.
  Applied identically to baseline and all trailing variants (apples-to-apples).
* OOS split: first 65% of signal dates = IS; last 35% = OOS. Bucket MFE/MAE
  derivation on IS only (no leakage). May-2026 scanners land 100% in OOS.
* Truncation filter: signals with < n_bars forward candles excluded from
  calibration (their partial walk would compress all metrics toward 0).
  Horizon-stability uses common set (non-truncated at 20d horizon).
* Performance: `bars_after(ticker, ts)` is called ONCE per signal in the primary
  loop; result cached in `bar_cache` dict keyed by signal_id. All subsequent
  sweeps read the cache.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Import shared helpers
# ---------------------------------------------------------------------------
from stock_scanner.analysis.daytrade_edge_sim import (
    build_candle_index,
    build_daily_index,
    atr_pct_before,
    bars_after,
    fetch_df,
    get_conn,
    DEFAULT_SL_PCT,
    DEFAULT_TP_PCT,
    ATR_STOP_ENABLED,
    ATR_THRESHOLD_PCT,
    ATR_STOP_MULT,
    ATR_RR,
    BE_TRIGGER_USD,
    MAX_ORDER_NOTIONAL_USD,
    MAX_RISK_USD,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Horizon in 1h bars (US market ~6.5h/day → ~7 bars/trading day as in existing sim)
HORIZON_CONFIGS = {
    "10d":  70,   # ~70 bars
    "15d": 105,   # primary
    "20d": 140,
}
PRIMARY_HORIZON = "15d"

ATR_BUCKETS = [0, 2, 4, 7, float("inf")]
ATR_BUCKET_LABELS = ["<2%", "2-4%", "4-7%", ">=7%"]

OOS_SPLIT_QUANTILE = 0.65
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
MIN_N_BUCKET = 50

TRAIL_K_VALUES = [1.5, 2.0, 2.5]

FLAT_THRESHOLD = 0.05  # |pnl| < this → flat (neither win nor loss)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def atr_bucket(atr_pct: Optional[float]) -> str:
    if atr_pct is None or (isinstance(atr_pct, float) and np.isnan(atr_pct)):
        return "unknown"
    for i, (lo, hi) in enumerate(zip(ATR_BUCKETS[:-1], ATR_BUCKETS[1:])):
        if lo <= atr_pct < hi:
            return ATR_BUCKET_LABELS[i]
    return ATR_BUCKET_LABELS[-1]


def be_trigger_for_entry(entry: float) -> float:
    """Compute BE trigger % (mirrors live auto-trader: $10 / notional)."""
    qty = int(MAX_ORDER_NOTIONAL_USD // entry) if entry > 0 else 1
    notional = entry * max(qty, 1)
    return BE_TRIGGER_USD / notional * 100


# ---------------------------------------------------------------------------
# Walk: MFE/MAE (non-exiting path recorder)
# ---------------------------------------------------------------------------

def walk_mfe_mae(bars: pd.DataFrame, entry: float, n_bars: int) -> dict:
    """Record MFE%, MAE%, terminal_pct over a window without exiting."""
    if bars.empty or entry <= 0:
        return {"mfe_pct": np.nan, "mae_pct": np.nan,
                "terminal_pct": np.nan, "bars_avail": 0, "truncated": True}
    window = bars.iloc[:n_bars]
    actual = len(window)
    truncated = actual < n_bars
    highs = window["high"].astype(float).values
    lows = window["low"].astype(float).values
    terminal_close = float(window.iloc[-1]["close"])
    mfe_pct = (highs.max() - entry) / entry * 100 if actual > 0 else 0.0
    mae_pct = (entry - lows.min()) / entry * 100 if actual > 0 else 0.0
    terminal_pct = (terminal_close - entry) / entry * 100
    return {"mfe_pct": mfe_pct, "mae_pct": mae_pct,
            "terminal_pct": terminal_pct, "bars_avail": actual, "truncated": truncated}


# ---------------------------------------------------------------------------
# Walk: Fixed SL/TP bracket with BE
# ---------------------------------------------------------------------------

def walk_bracket(
    entry: float,
    bars: pd.DataFrame,
    sl_pct: float,
    tp_pct: float,
    n_bars: int,
    be_trigger_pct: float = 2.0,
    be_enabled: bool = True,
) -> dict:
    """
    Fixed bracket walk with optional BE. Returns pnl_pct and exit_reason.
    outcome classification is done by caller using FLAT_THRESHOLD.
    """
    if bars.empty or entry <= 0:
        return {"pnl_pct": 0.0, "hold_bars": 0,
                "be_armed": False, "exit_reason": "no_data", "truncated": False}

    window = bars.iloc[:n_bars]
    truncated = len(window) < n_bars
    sl = entry * (1 - sl_pct / 100)
    tp = entry * (1 + tp_pct / 100)
    be_trigger = entry * (1 + be_trigger_pct / 100)
    active_sl = sl
    be_armed = False

    for i, (_, bar) in enumerate(window.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])

        if be_enabled and not be_armed and high >= be_trigger:
            be_armed = True
            active_sl = entry

        if low <= active_sl:
            pnl = (active_sl - entry) / entry * 100
            return {"pnl_pct": pnl, "hold_bars": i + 1,
                    "be_armed": be_armed, "exit_reason": "sl", "truncated": truncated}

        if high >= tp:
            pnl = (tp - entry) / entry * 100
            return {"pnl_pct": pnl, "hold_bars": i + 1,
                    "be_armed": be_armed, "exit_reason": "tp", "truncated": truncated}

    close = float(window.iloc[-1]["close"])
    pnl = (close - entry) / entry * 100
    return {"pnl_pct": pnl, "hold_bars": len(window),
            "be_armed": be_armed, "exit_reason": "timeout", "truncated": truncated}


# ---------------------------------------------------------------------------
# Walk: Trailing stop variants
# ---------------------------------------------------------------------------

def walk_trailing(
    entry: float,
    bars: pd.DataFrame,
    atr_pct: Optional[float],
    n_bars: int,
    be_trigger_pct: float,
    variant: str = "be_then_trail",
    trail_k: float = 2.0,
) -> dict:
    """
    Trailing stop walk. All profitable exits (trail_sl, tp_cap, timeout+p>0)
    produce a positive pnl_pct. outcome classification by caller.

    Variants:
      'baseline'      — fixed 3%/5% + BE@$10 (control, delegates to walk_bracket)
      'be_then_trail' — hard 3%SL until BE arms; then trail hwm by k×ATR%
      'chandelier'    — always trail from hwm by k×ATR%; hard 3%SL as floor
      'staged'        — BE@+8% → lock+6%@+12% → lock+10%@+18% → 2×ATR trail
    """
    if bars.empty or entry <= 0:
        return {"pnl_pct": 0.0, "hold_bars": 0,
                "be_armed": False, "exit_reason": "no_data", "variant": variant}

    if variant == "baseline":
        atr_p = atr_pct if (atr_pct is not None and not np.isnan(atr_pct)) else None
        if (ATR_STOP_ENABLED and atr_p is not None and atr_p >= ATR_THRESHOLD_PCT):
            sl_p = ATR_STOP_MULT * atr_p
            tp_p = ATR_RR * sl_p
        else:
            sl_p, tp_p = DEFAULT_SL_PCT, DEFAULT_TP_PCT
        res = walk_bracket(entry, bars, sl_p, tp_p, n_bars, be_trigger_pct)
        res["variant"] = "baseline"
        return res

    # Effective trail distance
    if atr_pct is not None and not np.isnan(atr_pct) and atr_pct > 0:
        trail_dist_pct = trail_k * atr_pct
    else:
        trail_dist_pct = trail_k * 3.0  # fallback: k × 3% if no ATR

    window = bars.iloc[:n_bars]
    truncated = len(window) < n_bars
    hard_sl = entry * (1 - DEFAULT_SL_PCT / 100)  # always protect entry
    be_trigger = entry * (1 + be_trigger_pct / 100)
    be_armed = False
    active_sl = hard_sl
    hwm = entry

    # Staged variant progressive levels
    staged_levels = [(8.0, 0.0), (12.0, 6.0), (18.0, 10.0)]  # (trigger%, lock%)
    stage_idx = 0

    for i, (_, bar) in enumerate(window.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        if high > hwm:
            hwm = high

        current_gain_pct = (hwm - entry) / entry * 100

        if variant == "be_then_trail":
            if not be_armed and high >= be_trigger:
                be_armed = True
                active_sl = entry  # move to BE
            if be_armed:
                trail_sl = hwm * (1 - trail_dist_pct / 100)
                active_sl = max(active_sl, trail_sl)

        elif variant == "chandelier":
            # Trails from hwm at all times; hard floor = hard_sl
            trail_sl = hwm * (1 - trail_dist_pct / 100)
            active_sl = max(hard_sl, trail_sl)
            if active_sl >= entry:
                be_armed = True

        elif variant == "staged":
            for si in range(stage_idx, len(staged_levels) + 1):
                if si == len(staged_levels):
                    # Final stage: ATR trail
                    trail_sl_s = hwm * (1 - trail_dist_pct / 100)
                    active_sl = max(active_sl, trail_sl_s)
                    if active_sl >= entry:
                        be_armed = True
                    break
                trig_pct, lock_pct = staged_levels[si]
                if current_gain_pct >= trig_pct:
                    lock_sl = entry * (1 + lock_pct / 100)
                    if lock_sl > active_sl:
                        active_sl = lock_sl
                        stage_idx = si + 1
                        if lock_sl >= entry:
                            be_armed = True
                else:
                    break  # not yet at this stage

        # --- SL hit ---
        if low <= active_sl:
            pnl = (active_sl - entry) / entry * 100
            return {"pnl_pct": pnl, "hold_bars": i + 1,
                    "be_armed": be_armed, "exit_reason": "trail_sl",
                    "truncated": truncated, "variant": variant}

    close_final = float(window.iloc[-1]["close"])
    pnl = (close_final - entry) / entry * 100
    return {"pnl_pct": pnl, "hold_bars": len(window),
            "be_armed": be_armed, "exit_reason": "timeout",
            "truncated": truncated, "variant": variant}


# ---------------------------------------------------------------------------
# Uniform metrics (BUG-2 fix: same classifier for all variants)
# ---------------------------------------------------------------------------

def classify_pnl(pnl: float) -> str:
    """Uniform pnl classifier: win/loss/flat."""
    if pnl > FLAT_THRESHOLD:
        return "win"
    elif pnl < -FLAT_THRESHOLD:
        return "loss"
    return "flat"


def compute_metrics(pnl_list: list) -> dict:
    """
    Compute PF, WR, expectancy, avg win/loss from raw pnl list.
    Includes ALL outcomes (win, loss, flat/timeout) uniformly.
    """
    if not pnl_list:
        return {"pf": np.nan, "expectancy_pct": np.nan, "wr": np.nan, "n": 0,
                "n_wins": 0, "n_losses": 0, "avg_win_pct": np.nan, "avg_loss_pct": np.nan}
    pnl = np.array(pnl_list, dtype=float)
    wins = pnl[pnl > FLAT_THRESHOLD]
    losses = pnl[pnl < -FLAT_THRESHOLD]
    n = len(pnl)
    n_w = len(wins)
    n_l = len(losses)
    gp = wins.sum() if n_w > 0 else 0.0
    gl = abs(losses.sum()) if n_l > 0 else 0.0
    pf = gp / gl if gl > 0 else (9.99 if gp > 0 else np.nan)
    wr = n_w / n
    expectancy = pnl.mean()
    return {"pf": pf, "expectancy_pct": expectancy, "wr": wr, "n": n,
            "n_wins": n_w, "n_losses": n_l,
            "avg_win_pct": wins.mean() if n_w > 0 else np.nan,
            "avg_loss_pct": losses.mean() if n_l > 0 else np.nan}


def bootstrap_ci(pnl_list: list, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """Bootstrap 90% CI on mean expectancy."""
    vals = np.array([x for x in pnl_list if not np.isnan(x)], dtype=float)
    n = len(vals)
    if n < 10:
        return {"exp_p5": np.nan, "exp_p50": np.nan, "exp_p95": np.nan}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(vals, size=n, replace=True).mean() for _ in range(n_boot)])
    return {"exp_p5": float(np.percentile(boot, 5)),
            "exp_p50": float(np.percentile(boot, 50)),
            "exp_p95": float(np.percentile(boot, 95))}


def fmt_ci(ci: dict) -> str:
    p5, p95 = ci.get("exp_p5"), ci.get("exp_p95")
    if p5 is None or (isinstance(p5, float) and np.isnan(p5)):
        return "[CI: n/a]"
    return f"[{p5:+.2f}%,{p95:+.2f}%]"


def fmt_m(m: dict, extra: str = "") -> str:
    pf = m.get("pf")
    exp = m.get("expectancy_pct")
    wr = m.get("wr")
    aw = m.get("avg_win_pct")
    al = m.get("avg_loss_pct")
    n = m.get("n", 0)
    pf_s = f"{pf:.3f}" if pf is not None and not np.isnan(pf) else "  N/A"
    exp_s = f"{exp:+.3f}%" if exp is not None and not np.isnan(exp) else "   N/A"
    wr_s = f"{wr*100:.1f}%" if wr is not None and not np.isnan(wr) else " N/A"
    aw_s = f"{aw:+.2f}%" if aw is not None and not np.isnan(aw) else "  N/A"
    al_s = f"{al:+.2f}%" if al is not None and not np.isnan(al) else "  N/A"
    return (f"n={n:5d}  PF={pf_s}  WR={wr_s}  exp={exp_s}  "
            f"avgW={aw_s}  avgL={al_s}{extra}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_calibration(conn):
    print("=" * 74)
    print("STOCK SL/TP CALIBRATION & TRAILING STOP STUDY")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 74)

    # ======================================================================
    # TASK 1 — Realized holding period
    # ======================================================================
    print("\n" + "=" * 74)
    print("TASK 1: REALIZED HOLDING-PERIOD DISTRIBUTION")
    print("=" * 74)

    hold_sql = """
        SELECT scanner_name,
               EXTRACT(EPOCH FROM (close_timestamp - signal_timestamp))/3600 AS hold_h,
               realized_pnl_pct, signal_timestamp
        FROM stock_scanner_signals
        WHERE signal_type = 'BUY' AND close_timestamp IS NOT NULL
          AND status = 'closed'
          AND EXTRACT(EPOCH FROM (close_timestamp - signal_timestamp)) > 0
        ORDER BY signal_timestamp
    """
    hold_df = fetch_df(conn, hold_sql)
    print(f"\nSignal-close hold distribution (n={len(hold_df)} closed BUY signals)")
    print("CAVEAT: close_timestamp = SWING-MODEL close, not day-trade bracket exit.")
    print("        This overstates the actual execution hold.\n")

    def print_hold_stats(df: pd.DataFrame, label: str):
        h = df["hold_h"].astype(float)
        ps = {p: h.quantile(p / 100) for p in [25, 50, 75, 90, 95]}
        print(f"  {label} (n={len(df)}):")
        print(f"    p25={ps[25]/24:.1f}d  median={ps[50]/24:.1f}d  "
              f"p75={ps[75]/24:.1f}d  p90={ps[90]/24:.1f}d  p95={ps[95]/24:.1f}d")
        print(f"    hours: med={ps[50]:.0f}h  p75={ps[75]:.0f}h  p90={ps[90]:.0f}h")

    print_hold_stats(hold_df, "ALL scanners (swing-close)")
    print()
    for scn, grp in hold_df.groupby("scanner_name"):
        if len(grp) >= 20:
            print_hold_stats(grp, str(scn))

    bt_sql = """
        SELECT bt.duration_hours, bt.profit_pct, bt.open_time, ss.scanner_name
        FROM broker_trades bt
        LEFT JOIN stock_scanner_signals ss ON bt.signal_id = ss.id
        WHERE bt.status = 'closed' AND bt.duration_hours IS NOT NULL AND bt.duration_hours > 0
        ORDER BY bt.open_time
    """
    bt_df = fetch_df(conn, bt_sql)
    print(f"\nBroker-execution hold (n={len(bt_df)}) — USE THIS for horizon calibration.\n")
    print_hold_stats(bt_df.rename(columns={"duration_hours": "hold_h"}), "broker_trades (real exec)")
    bth = bt_df["duration_hours"].astype(float)
    print(f"\n  Breakdown by duration:")
    for lo, hi, lbl in [(0, 8, "<8h"), (8, 24, "8-24h"), (24, 72, "1-3d"),
                        (72, 168, "3-7d"), (168, 9999, ">7d")]:
        cnt = ((bth >= lo) & (bth < hi)).sum()
        print(f"    {lbl:10s}: {cnt:4d} ({cnt/len(bt_df)*100:.0f}%)")

    bt_p90 = float(bth.quantile(0.90))
    sw_p90 = float(hold_df["hold_h"].astype(float).quantile(0.90))
    print(f"\n  THREE-WAY HORIZON COMPARISON:")
    print(f"    Nominal (scanner intent)    :  1 trading day (~7 bars)")
    print(f"    Broker execution p90        : {bt_p90:.0f}h (~{bt_p90/24:.1f}d)")
    print(f"    Swing-close p90 (not exec)  : {sw_p90:.0f}h (~{sw_p90/24:.1f}d)")
    print(f"\n  HORIZON DECISION:")
    print(f"    Primary = 15d ({HORIZON_CONFIGS['15d']} bars). Covers broker-exec p90 with buffer.")
    print(f"    The swing-close p90 ({sw_p90/24:.0f}d) reflects multi-week positions where the day-")
    print(f"    trade bracket was never active. Using it would model a different system.")
    print(f"    Stability tested across 10d/15d/20d on a common non-truncated set.")

    # ======================================================================
    # Load candles + signals
    # ======================================================================
    print("\n" + "-" * 40)
    print("Pre-loading candle data...")
    candles_by_ticker = build_candle_index(conn)
    daily_by_ticker = build_daily_index(conn)
    print(f"  1h: {sum(len(v) for v in candles_by_ticker.values()):,} bars, "
          f"{len(candles_by_ticker)} tickers")
    print(f"  1d: {sum(len(v) for v in daily_by_ticker.values()):,} bars, "
          f"{len(daily_by_ticker)} tickers")

    signals_sql = """
        SELECT id, signal_timestamp, scanner_name, ticker, entry_price
        FROM stock_scanner_signals
        WHERE signal_type = 'BUY'
        ORDER BY signal_timestamp
    """
    signals = fetch_df(conn, signals_sql)
    print(f"\nLoaded {len(signals):,} BUY signals.")

    dates_sorted = signals["signal_timestamp"].sort_values()
    split_cutoff = dates_sorted.quantile(OOS_SPLIT_QUANTILE)
    signals["oos"] = signals["signal_timestamp"] > split_cutoff
    n_is = (~signals["oos"]).sum()
    n_oos = signals["oos"].sum()
    print(f"  OOS cutoff: {split_cutoff.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  IS={n_is}  OOS={n_oos}")
    print(f"  NOTE: May-2026 scanners (pocket_pivot etc.) fall 100% in OOS — ")
    print(f"        OOS validation for them is single-regime only, not true OOS.")

    # ======================================================================
    # Primary simulation loop — cache (bars, entry) per signal_id
    # ======================================================================
    n_bars_primary = HORIZON_CONFIGS[PRIMARY_HORIZON]
    n_bars_long = HORIZON_CONFIGS["20d"]  # for truncation-common-set

    print(f"\nRunning primary loop ({len(signals):,} signals, horizon={PRIMARY_HORIZON} / "
          f"{n_bars_primary} bars)...")
    print("  Caching forward bars per signal to avoid repeated bar-walks in sweeps.")

    bar_cache: dict = {}   # signal_id → {"bars": df, "entry": float}
    recs = []
    n_skipped = 0
    n_no_atr = 0

    for idx, sig in signals.iterrows():
        sig_id = int(sig["id"])
        base_t = str(sig["ticker"]).split(".")[0]
        ts = sig["signal_timestamp"]

        atr_pct_val = atr_pct_before(daily_by_ticker, base_t, ts)
        if atr_pct_val is None:
            n_no_atr += 1

        fwd, entry = bars_after(candles_by_ticker, base_t, ts)
        if fwd.empty or entry <= 0:
            n_skipped += 1
            continue

        # Cache the full 20d bar slice (for horizon-stability test)
        bar_cache[sig_id] = {
            "bars": fwd,
            "entry": entry,
            "atr_pct": atr_pct_val,
            "be_trigger_pct": be_trigger_for_entry(entry),
            "oos": bool(sig["oos"]),
            "scanner": str(sig["scanner_name"]),
            "ticker": str(sig["ticker"]),
            "signal_timestamp": ts,
        }

        # MFE/MAE at primary horizon
        mm = walk_mfe_mae(fwd, entry, n_bars_primary)
        # Truncated at 20d (common set for horizon stability)
        mm20 = walk_mfe_mae(fwd, entry, n_bars_long)

        # Live bracket at primary horizon
        if (ATR_STOP_ENABLED and atr_pct_val is not None and
                atr_pct_val >= ATR_THRESHOLD_PCT):
            live_sl = ATR_STOP_MULT * atr_pct_val
            live_tp = ATR_RR * live_sl
        else:
            live_sl, live_tp = DEFAULT_SL_PCT, DEFAULT_TP_PCT

        live_res = walk_bracket(entry, fwd, live_sl, live_tp,
                                n_bars_primary, be_trigger_for_entry(entry))

        recs.append({
            "signal_id": sig_id,
            "scanner": str(sig["scanner_name"]),
            "ticker": str(sig["ticker"]),
            "signal_timestamp": ts,
            "oos": bool(sig["oos"]),
            "entry": entry,
            "atr_pct": atr_pct_val,
            "atr_bucket": atr_bucket(atr_pct_val),
            # MFE/MAE
            "mfe_pct": mm["mfe_pct"],
            "mae_pct": mm["mae_pct"],
            "terminal_pct": mm["terminal_pct"],
            "truncated_15d": mm["truncated"],
            "truncated_20d": mm20["truncated"],
            # Live bracket outcome (used for diagnostics)
            "live_pnl": live_res["pnl_pct"],
            "live_exit": live_res["exit_reason"],
            "live_sl": live_sl,
            "live_tp": live_tp,
        })

        if (idx + 1) % 2000 == 0:
            print(f"  ...{idx + 1:,}/{len(signals):,}")

    sim_df = pd.DataFrame(recs)
    print(f"\n  Processed: {len(sim_df):,}  Skipped (no candles): {n_skipped}  "
          f"No ATR: {n_no_atr}")
    n_trunc_15 = sim_df["truncated_15d"].sum()
    n_trunc_20 = sim_df["truncated_20d"].sum()
    print(f"  Truncated at 15d: {n_trunc_15:,} ({n_trunc_15/len(sim_df)*100:.1f}%)")
    print(f"  Truncated at 20d: {n_trunc_20:,} ({n_trunc_20/len(sim_df)*100:.1f}%)")

    # Working set: non-truncated at primary horizon
    full15 = sim_df[~sim_df["truncated_15d"]].copy()
    # Common set for horizon stability: non-truncated at 20d
    full20_common = sim_df[~sim_df["truncated_20d"]].copy()
    print(f"\n  Working set (non-trunc 15d): {len(full15):,}")
    print(f"  Common set (non-trunc 20d): {len(full20_common):,}")

    # ======================================================================
    # TASK 2 — MFE/MAE per ATR bucket
    # ======================================================================
    print("\n" + "=" * 74)
    print("TASK 2: MFE/MAE DISTRIBUTIONS PER ATR% BUCKET")
    print(f"  Horizon: {PRIMARY_HORIZON} ({n_bars_primary} bars), non-truncated")
    print("  'Eventual winner' = terminal close > entry (candle path, NOT realized_pnl_pct)")
    print("  IMPORTANT: MFE/MAE derived from IS only to avoid OOS leakage in Task 3.")
    print("=" * 74)

    is_full = full15[~full15["oos"]].copy()
    oos_full = full15[full15["oos"]].copy()
    is_full["eventual_win"] = is_full["terminal_pct"] > 0
    is_winners = is_full[is_full["eventual_win"]]
    print(f"\n  IS set: {len(is_full):,}  IS eventual winners: {len(is_winners):,} "
          f"({len(is_winners)/len(is_full)*100:.1f}%)")

    print(f"\n  MFE/MAE table — IS WINNERS only (bucket derivation; IS only = no leakage)\n")
    hdr = (f"  {'Bucket':8s}  {'n_W':>5s}  "
           f"{'MFE_p50':>8s}  {'MFE_p75':>8s}  {'MFE×AT_p50':>11s}  {'MFE×AT_p75':>11s}  "
           f"{'MAE_p50':>8s}  {'MAE_p75':>8s}  {'MAE_p90':>8s}  "
           f"{'MAE×AT_p50':>11s}  {'MAE×AT_p75':>11s}  {'MAE×AT_p90':>11s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    bucket_stats: dict = {}  # IS-derived per-bucket stats
    for bkt in ATR_BUCKET_LABELS + ["unknown"]:
        w = is_winners[is_winners["atr_bucket"] == bkt].copy()
        n_w = len(w)
        if n_w < 5:
            continue
        valid_atr = w["atr_pct"].notna() & (w["atr_pct"] > 0)
        w_atr = w[valid_atr]

        def q(s, p): return s.quantile(p)

        mfe = w["mfe_pct"]
        mae = w["mae_pct"]
        thin = f" [THIN n={n_w}]" if n_w < MIN_N_BUCKET else ""

        if not w_atr.empty:
            mfe_x = w_atr["mfe_pct"] / w_atr["atr_pct"]
            mae_x = w_atr["mae_pct"] / w_atr["atr_pct"]
            mx_p50, mx_p75 = q(mfe_x, 0.50), q(mfe_x, 0.75)
            ax_p50, ax_p75, ax_p90 = q(mae_x, 0.50), q(mae_x, 0.75), q(mae_x, 0.90)
        else:
            mx_p50 = mx_p75 = ax_p50 = ax_p75 = ax_p90 = np.nan

        row = {
            "n_win": n_w,
            "mfe_p50": q(mfe, 0.50), "mfe_p75": q(mfe, 0.75),
            "mae_p50": q(mae, 0.50), "mae_p75": q(mae, 0.75), "mae_p90": q(mae, 0.90),
            "mfe_atr_p50": mx_p50, "mfe_atr_p75": mx_p75,
            "mae_atr_p50": ax_p50, "mae_atr_p75": ax_p75, "mae_atr_p90": ax_p90,
        }
        bucket_stats[bkt] = row

        def fs(v): return f"{v:.2f}" if not np.isnan(v) else "  N/A"
        print(f"  {bkt:8s}  {n_w:>5d}  "
              f"{fs(row['mfe_p50']):>8s}  {fs(row['mfe_p75']):>8s}  "
              f"{fs(mx_p50):>11s}  {fs(mx_p75):>11s}  "
              f"{fs(row['mae_p50']):>8s}  {fs(row['mae_p75']):>8s}  {fs(row['mae_p90']):>8s}  "
              f"{fs(ax_p50):>11s}  {fs(ax_p75):>11s}  {fs(ax_p90):>11s}{thin}")

    # All-winner summary
    all_mfe_p50 = is_winners["mfe_pct"].quantile(0.50)
    all_mae_p75 = is_winners["mae_pct"].quantile(0.75)
    all_mae_p90 = is_winners["mae_pct"].quantile(0.90)
    print(f"\n  All-winner IS summary:")
    print(f"    MFE p50={all_mfe_p50:.2f}%  — ceiling a trailing stop could capture")
    print(f"    Winner MAE p75={all_mae_p75:.2f}%  p90={all_mae_p90:.2f}%  — SL noise floor")

    print(f"\n  3% SL NOISE FLOOR AUDIT:")
    for bkt in ATR_BUCKET_LABELS:
        bkt_df = is_full[is_full["atr_bucket"] == bkt]
        if len(bkt_df) < 10:
            continue
        med_atr = bkt_df["atr_pct"].dropna().median()
        if pd.notna(med_atr) and med_atr > 0:
            ratio = 3.0 / med_atr
            noise = " <<< NOISE-STOPPED (3%SL < 0.7×ATR)" if ratio < 0.7 else ""
            print(f"    {bkt:8s}: median ATR={med_atr:.2f}%  3%/ATR={ratio:.2f}×{noise}")

    # ======================================================================
    # TASK 3 — SL/TP calibration
    # ======================================================================
    print("\n" + "=" * 74)
    print("TASK 3: SL/TP CALIBRATION vs CURRENT LIVE RULE")
    print(f"  IS split ≤{split_cutoff.strftime('%Y-%m-%d')}  |  OOS split >{split_cutoff.strftime('%Y-%m-%d')}")
    print("  Bucket stats from IS only. OOS evaluation is true hold-out.")
    print("=" * 74)

    # Derived SL/TP from IS bucket stats
    print(f"\n  Derived SL/TP candidates (IS winner MAE p75 → SL, MFE p50 → TP):\n")
    derived: dict = {}
    for bkt, stats in bucket_stats.items():
        sl_c = round(stats["mae_p75"], 2)
        tp_c = round(stats["mfe_p50"], 2)
        rr = tp_c / sl_c if sl_c > 0 else np.nan
        thin = f"  [THIN n={stats['n_win']}]" if stats["n_win"] < MIN_N_BUCKET else ""
        print(f"    {bkt:8s}: SL={sl_c:.2f}%  TP={tp_c:.2f}%  R:R={rr:.2f}{thin}")
        derived[bkt] = {"sl": sl_c, "tp": tp_c}

    # Helper: run bracket from cache subset
    def bracket_pnl_list(subset_ids: list, sl_pct: float, tp_pct: float,
                         n_bars: int) -> list:
        """Use bar_cache to run bracket walk for a list of signal_ids."""
        pnl_list = []
        for sid in subset_ids:
            c = bar_cache.get(sid)
            if c is None:
                continue
            res = walk_bracket(c["entry"], c["bars"], sl_pct, tp_pct,
                               n_bars, c["be_trigger_pct"])
            pnl_list.append(res["pnl_pct"])
        return pnl_list

    # Collect IS / OOS signal IDs per bucket
    def ids_for(subset_df: pd.DataFrame, bucket: Optional[str] = None) -> list:
        if bucket:
            return subset_df[subset_df["atr_bucket"] == bucket]["signal_id"].tolist()
        return subset_df["signal_id"].tolist()

    is_ids_all = ids_for(full15[~full15["oos"]])
    oos_ids_all = ids_for(full15[full15["oos"]])

    print(f"\n  CURRENT LIVE RULE vs DERIVED CANDIDATES:")
    print(f"  (PF and expectancy using uniform classifier: win=pnl>{FLAT_THRESHOLD}%, "
          f"loss=pnl<-{FLAT_THRESHOLD}%)\n")

    print(f"  {'Spec':35s}  {'Split':5s}  {'n':>5s}  {'PF':>7s}  "
          f"{'exp%':>8s}  {'WR':>6s}  {'avgW%':>7s}  {'avgL%':>7s}  {'CI90%':>20s}")
    print("  " + "-" * 115)

    for split_label, split_ids in [("IS", is_ids_all), ("OOS", oos_ids_all)]:
        # Current live (ATR-conditional for each signal using cached live_pnl)
        # We already have live_pnl pre-computed; re-derive from cache for consistency
        live_pnls = [bar_cache[sid]["entry"] and bar_cache[sid] and
                     # Run live rule from cache
                     None for sid in split_ids]
        # Actually re-run from cache using live rule
        live_pnls_real = []
        for sid in split_ids:
            c = bar_cache.get(sid)
            if c is None:
                continue
            atr_p = c["atr_pct"]
            if (ATR_STOP_ENABLED and atr_p is not None and
                    not np.isnan(atr_p) and atr_p >= ATR_THRESHOLD_PCT):
                sl_l = ATR_STOP_MULT * atr_p
                tp_l = ATR_RR * sl_l
            else:
                sl_l, tp_l = DEFAULT_SL_PCT, DEFAULT_TP_PCT
            res = walk_bracket(c["entry"], c["bars"], sl_l, tp_l,
                               n_bars_primary, c["be_trigger_pct"])
            live_pnls_real.append(res["pnl_pct"])
        m = compute_metrics(live_pnls_real)
        ci = bootstrap_ci(live_pnls_real)
        ci_s = fmt_ci(ci)
        live_lbl = f"CURRENT(3%SL/5%TP+ATR≥7% branch)"
        print(f"  {live_lbl:35s}  {split_label:5s}  {fmt_m(m, f'  {ci_s}')}")

    print()

    for bkt in ATR_BUCKET_LABELS:
        if bkt not in derived:
            continue
        sl_c, tp_c = derived[bkt]["sl"], derived[bkt]["tp"]
        is_bkt_ids = ids_for(full15[~full15["oos"]], bkt)
        oos_bkt_ids = ids_for(full15[full15["oos"]], bkt)
        n_is_bkt = len(is_bkt_ids)
        n_oos_bkt = len(oos_bkt_ids)
        thin = " [THIN]" if n_is_bkt < MIN_N_BUCKET else ""
        lbl = f"DERIVED_{bkt}(SL={sl_c:.1f}%/TP={tp_c:.1f}%){thin}"

        for split_label, bkt_ids in [("IS", is_bkt_ids), ("OOS", oos_bkt_ids)]:
            pnls = bracket_pnl_list(bkt_ids, sl_c, tp_c, n_bars_primary)
            if not pnls:
                continue
            m = compute_metrics(pnls)
            ci = bootstrap_ci(pnls)
            ci_s = fmt_ci(ci)
            print(f"  {lbl:35s}  {split_label:5s}  {fmt_m(m, f'  {ci_s}')}")
        print()

    # Fixed SL sweep (plateau test)
    print(f"\n  FIXED SL PLATEAU SWEEP (TP fixed at {DEFAULT_TP_PCT}%, all signals):")
    print(f"  Plateau = similar expectancy across ±50% SL band → robust choice.")
    print(f"  Spike = single peak with sharp decline on either side → overfit.\n")
    print(f"  {'SL%':>5s}  {'IS_PF':>7s}  {'IS_exp':>8s}  "
          f"{'OOS_PF':>7s}  {'OOS_exp':>8s}  {'IS_CI90%':>22s}  {'OOS_CI90%':>22s}  {'NOTE':>10s}")

    for sl_s in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]:
        pnls_is = bracket_pnl_list(is_ids_all, sl_s, DEFAULT_TP_PCT, n_bars_primary)
        pnls_oos = bracket_pnl_list(oos_ids_all, sl_s, DEFAULT_TP_PCT, n_bars_primary)
        m_is = compute_metrics(pnls_is)
        m_oos = compute_metrics(pnls_oos)
        ci_is = bootstrap_ci(pnls_is)
        ci_oos = bootstrap_ci(pnls_oos)

        pf_is = f"{m_is['pf']:.3f}" if not np.isnan(m_is.get("pf", np.nan)) else "  N/A"
        exp_is = (f"{m_is['expectancy_pct']:+.3f}%"
                  if not np.isnan(m_is.get("expectancy_pct", np.nan)) else "   N/A")
        pf_oos = f"{m_oos['pf']:.3f}" if not np.isnan(m_oos.get("pf", np.nan)) else "  N/A"
        exp_oos = (f"{m_oos['expectancy_pct']:+.3f}%"
                   if not np.isnan(m_oos.get("expectancy_pct", np.nan)) else "   N/A")
        note = " <-- live" if sl_s == DEFAULT_SL_PCT else ""
        print(f"  {sl_s:>5.1f}  {pf_is:>7s}  {exp_is:>8s}  "
              f"{pf_oos:>7s}  {exp_oos:>8s}  "
              f"{fmt_ci(ci_is):>22s}  {fmt_ci(ci_oos):>22s}{note}")

    # ======================================================================
    # TASK 4 — Trailing stop variants
    # ======================================================================
    print("\n" + "=" * 74)
    print("TASK 4: TRAILING STOP VARIANT COMPARISON")
    print(f"  Horizon={PRIMARY_HORIZON} ({n_bars_primary} bars). Same OOS split and signal set.")
    print(f"  UNIFORM CLASSIFIER: pnl>{FLAT_THRESHOLD}%→win, pnl<-{FLAT_THRESHOLD}%→loss, else flat.")
    print(f"  Profitable trail-stops are wins. Baseline uses same classifier.")
    print("=" * 74)

    print(f"\n  Legend:")
    print(f"    baseline:          fixed 3%SL/5%TP + BE@$10 (ATR-conditional for ≥7%)")
    print(f"    be_trail(k):       arms BE at $10 unreal; then trails hwm by k×ATR%")
    print(f"    chandelier(k):     always trails max-close by k×ATR%; hard 3%SL floor")
    print(f"    staged:            +8%→BE, +12%→lock+6%, +18%→lock+10%, then ATR×2 trail")
    print()

    variant_specs = [("baseline", None)]
    for k in TRAIL_K_VALUES:
        variant_specs.append((f"be_trail(k={k})", k))
    for k in TRAIL_K_VALUES:
        variant_specs.append((f"chandelier(k={k})", k))
    variant_specs.append(("staged", 2.0))

    hdr2 = (f"  {'Variant':28s}  {'Split':4s}  {'n':>5s}  {'WR':>6s}  "
            f"{'PF':>7s}  {'exp%':>8s}  {'avgW%':>7s}  {'avgL%':>7s}  "
            f"{'SL%':>5s}  {'TP%':>5s}  {'TO%':>5s}  {'CI90%':>20s}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    for split_label, split_ids, split_bool in [("IS", is_ids_all, False),
                                                ("OOS", oos_ids_all, True)]:
        for (vlabel, k_val) in variant_specs:
            # Determine variant name for walk_trailing
            if vlabel == "baseline":
                vname = "baseline"
            elif vlabel.startswith("be_trail"):
                vname = "be_then_trail"
            elif vlabel.startswith("chandelier"):
                vname = "chandelier"
            elif vlabel == "staged":
                vname = "staged"
            else:
                vname = "baseline"

            trail_k = k_val if k_val is not None else 2.0

            pnl_list = []
            ex_sl, ex_tp, ex_to = 0, 0, 0

            for sid in split_ids:
                c = bar_cache.get(sid)
                if c is None:
                    continue
                res = walk_trailing(c["entry"], c["bars"],
                                    c["atr_pct"] if not pd.isna(c.get("atr_pct", np.nan)) else None,
                                    n_bars_primary, c["be_trigger_pct"],
                                    variant=vname, trail_k=trail_k)
                pnl_list.append(res["pnl_pct"])
                er = res.get("exit_reason", "")
                if "sl" in er or "trail" in er:
                    ex_sl += 1
                elif "tp" in er:
                    ex_tp += 1
                elif "timeout" in er:
                    ex_to += 1

            m = compute_metrics(pnl_list)
            ci = bootstrap_ci(pnl_list)
            n_proc = len(pnl_list)

            pf_s = f"{m['pf']:.3f}" if not np.isnan(m.get("pf", np.nan)) else "  N/A"
            exp_s = (f"{m['expectancy_pct']:+.3f}%"
                     if not np.isnan(m.get("expectancy_pct", np.nan)) else "   N/A")
            wr_s = f"{m['wr']*100:.1f}%" if not np.isnan(m.get("wr", np.nan)) else " N/A"
            aw_s = f"{m.get('avg_win_pct', np.nan):+.2f}%" if not np.isnan(m.get("avg_win_pct", np.nan)) else "   N/A"
            al_s = f"{m.get('avg_loss_pct', np.nan):+.2f}%" if not np.isnan(m.get("avg_loss_pct", np.nan)) else "   N/A"
            sl_pct_s = f"{ex_sl/n_proc*100:.0f}%" if n_proc else "N/A"
            tp_pct_s = f"{ex_tp/n_proc*100:.0f}%" if n_proc else "N/A"
            to_pct_s = f"{ex_to/n_proc*100:.0f}%" if n_proc else "N/A"
            ci_s = fmt_ci(ci)
            print(f"  {vlabel:28s}  {split_label:4s}  {n_proc:>5d}  {wr_s:>6s}  "
                  f"{pf_s:>7s}  {exp_s:>8s}  {aw_s:>7s}  {al_s:>7s}  "
                  f"{sl_pct_s:>5s}  {tp_pct_s:>5s}  {to_pct_s:>5s}  "
                  f"{ci_s:>20s}")
        print()  # blank line between IS/OOS blocks

    # ======================================================================
    # HORIZON STABILITY CHECK
    # ======================================================================
    print("\n" + "=" * 74)
    print("HORIZON STABILITY: baseline at 10d/15d/20d (common non-trunc-20d set)")
    print("=" * 74)

    common_ids = full20_common["signal_id"].tolist()
    is_common = full20_common[~full20_common["oos"]]["signal_id"].tolist()
    oos_common = full20_common[full20_common["oos"]]["signal_id"].tolist()

    print(f"\n  Common set (non-trunc 20d): {len(common_ids)} total, "
          f"IS={len(is_common)}, OOS={len(oos_common)}\n")
    print(f"  {'Horizon':8s}  {'Split':5s}  {'n':>5s}  {'PF':>7s}  {'exp%':>8s}")

    for hz_name, hz_bars in HORIZON_CONFIGS.items():
        for split_label, split_ids_hz in [("IS", is_common), ("OOS", oos_common)]:
            pnls_hz = []
            for sid in split_ids_hz:
                c = bar_cache.get(sid)
                if c is None:
                    continue
                atr_p = c["atr_pct"]
                if (ATR_STOP_ENABLED and atr_p is not None and
                        not np.isnan(atr_p) and atr_p >= ATR_THRESHOLD_PCT):
                    sl_hz = ATR_STOP_MULT * atr_p
                    tp_hz = ATR_RR * sl_hz
                else:
                    sl_hz, tp_hz = DEFAULT_SL_PCT, DEFAULT_TP_PCT
                res = walk_bracket(c["entry"], c["bars"], sl_hz, tp_hz,
                                   hz_bars, c["be_trigger_pct"])
                pnls_hz.append(res["pnl_pct"])
            m = compute_metrics(pnls_hz)
            pf_s = f"{m['pf']:.3f}" if not np.isnan(m.get("pf", np.nan)) else "  N/A"
            exp_s = (f"{m['expectancy_pct']:+.3f}%"
                     if not np.isnan(m.get("expectancy_pct", np.nan)) else "   N/A")
            marker = " <-- primary" if hz_name == PRIMARY_HORIZON else ""
            print(f"  {hz_name:8s}  {split_label:5s}  {m.get('n', 0):>5d}  "
                  f"{pf_s:>7s}  {exp_s:>8s}{marker}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "=" * 74)
    print("SUMMARY VERDICT FRAMEWORK")
    print("=" * 74)
    print(f"""
  HONEST FRAMEWORK FOR READING THESE RESULTS:

  1. HOLDING PERIOD MISMATCH (CONFIRMED)
     The nominal '1-day daytrade' label does not match execution reality.
     Broker-execution p90 is ~{bt_p90:.0f}h (~{bt_p90/24:.1f}d). The day-trade bracket
     (SL/TP) simply persists until it hits, often over many days.

  2. MFE/MAE AND SL NOISE
     See Task 2 table. Key question: does winner MAE p75 exceed 3% in the
     2-4% and 4-7% ATR buckets? If so, the current 3% SL is prematurely
     stopping winning trades before they develop. The ×ATR columns
     generalize across the rotating universe.

  3. SL/TP CALIBRATION (OOS GUARDRAIL)
     The SL plateau sweep is the main deliverable. A plateau (similar
     expectancy across ±50% of SL) → the specific level is noise-robust.
     A single peak in IS that collapses OOS → overfit. The system has a
     documented history of in-sample wins collapsing OOS; treat any IS-only
     improvement with high skepticism. The bootstrap CI should also span
     positive expectancy before claiming a real edge.

  4. TRAILING STOPS
     BE-then-trail captures more of winner MFE p50 IF winners actually move
     enough to arm BE before reversing. If most winners are stopped at the
     3% SL before reaching the $10 BE trigger, trailing is irrelevant for
     them. The via_SL%/via_TP%/via_TO% columns tell you what fraction of
     trades each exit path covers. Compare the baseline avgW% vs trailing
     avgW% to see if trailing is actually extending winners or just adding
     noise-stops to profitable excursions.

  5. THIN BUCKETS
     Any bucket with n_win < {MIN_N_BUCKET} has insufficient data to anchor SL/TP.
     Do not change production config from thin-bucket results.

  APPLY NOTHING TO PRODUCTION. This is analysis only.
""")

    print(f"Script: stock-scanner/app/stock_scanner/analysis/sltp_calibration.py")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        conn = get_conn()
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        run_calibration(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
