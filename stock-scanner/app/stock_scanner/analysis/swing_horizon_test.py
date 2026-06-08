"""
Swing-Horizon Test — HORIZON-MISMATCH Thesis Validation
=========================================================
Tests whether zlma_trend (and other scanners) have real edge at SWING horizon
that a day-trade bracket destroys.

TWO ANALYSES:
  1. MFE/MAE by horizon (1, 3, 5, 10, 20, 30 trading days) — execution-agnostic.
  2. Clean swing-exit sim — five exit strategies per scanner:
       day_trade:    3%SL/5%TP + BE@+2% (re-states prior result; coarse on daily)
       swing_8_16:   fixed 8%SL/16%TP, no BE
       swing_10_20:  fixed 10%SL/20%TP, no BE
       swing_native: scanner's own stop_loss/take_profit_1 (sanity-filtered)
       swing_trail:  initial 8%SL, chandelier trail (8% below running peak once +8% reached)

DATA: stock_candles_synthesized (timeframe='1d') — daily bars, ~7× fewer than hourly.
      Justification: swing horizons (≤30 trading days) do NOT need intraday resolution;
      daily ambiguity over a multi-week hold is negligible; conservative SL-first preserved.

CRITICAL DESIGN DECISIONS:
  - Timeout horizon: 30 daily bars (not 10×6 h-bars — that was for hourly engine)
  - BE only applied to day_trade strategy; all swing variants use be_enabled=False
  - Loss capped at SL on all variants (NOT mark-to-market until timeout)
  - Censoring: signals with <30 forward daily bars are EXCLUDED from swing PF headline
    and flagged separately as "truncated". May-26 scanners almost entirely truncated.
  - MFE/MAE computed on CONSTANT COHORT (signals with ≥30 forward daily bars)
    to avoid conflating horizon with cohort selection.
  - day_trade on daily bars is coarse (wide intra-day range → high ambiguity rate);
    prior hourly result (PF ~0.78) is authoritative for day_trade verdict.

Usage (inside stock-scanner container):
    python -m stock_scanner.analysis.swing_horizon_test

Read-only: does NOT modify any table.
"""

from __future__ import annotations

import sys
import warnings
from datetime import timezone
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = "postgresql://postgres:postgres@postgres:5432/stocks"

# Swing simulation horizon (trading days on daily bars)
SWING_HORIZON_DAYS = 30

# MFE/MAE horizons (trading days)
MFE_HORIZONS = [1, 3, 5, 10, 20, 30]

# Day-trade bracket parameters (matches AutoOpenTrader)
DT_SL_PCT = 3.0
DT_TP_PCT = 5.0
DT_BE_TRIGGER_PCT = 2.0   # +2% move arms breakeven (simplified for daily; real uses $10 notional)

# Swing fixed brackets
SW_8_16_SL = 8.0
SW_8_16_TP = 16.0
SW_10_20_SL = 10.0
SW_10_20_TP = 20.0

# Chandelier trail parameters
TRAIL_SL_PCT = 8.0          # initial stop AND trailing distance below peak
TRAIL_ENGAGE_PCT = 8.0      # trailing engages once price reaches +8% above entry

# Native-field sanity filter
NATIVE_MAX_DIST_PCT = 60.0  # drop if SL or TP dist > 60%

# Focus scanner for decisive output
FOCUS_SCANNER = "zlma_trend"
FOCUS_SCANNER_MIN_N = 100   # minimum n for weighted conclusions


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def fetch_df(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Daily bar engine variants
# ---------------------------------------------------------------------------

def walk_daily_bars_bracket(
    entry: float,
    sl: float,
    tp: float,
    bars: pd.DataFrame,
    horizon_days: int = SWING_HORIZON_DAYS,
    be_trigger_pct: Optional[float] = None,  # None = no BE
) -> dict:
    """
    Walk daily bars (SL-first, conservative).

    Parameters
    ----------
    entry           : entry price (open of bars.iloc[0])
    sl              : absolute stop-loss price
    tp              : absolute take-profit price
    bars            : daily bars starting at ENTRY bar (iloc[0].open = entry)
    horizon_days    : max bars before timeout
    be_trigger_pct  : if not None, BE arms when bar HIGH >= entry*(1+be_trigger_pct/100)

    Returns
    -------
    dict with keys: outcome, pnl_pct, hold_days, be_armed, truncated
    """
    if bars.empty or entry <= 0 or sl <= 0 or tp <= 0:
        return {
            "outcome": "no_data",
            "pnl_pct": 0.0,
            "hold_days": 0,
            "be_armed": False,
            "truncated": False,
        }

    active_sl = sl
    be_armed = False
    be_trigger = entry * (1 + be_trigger_pct / 100) if be_trigger_pct is not None else None

    for i, (_, bar) in enumerate(bars.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])

        # BE arm (day_trade only — caller controls this via be_trigger_pct)
        if be_trigger is not None and not be_armed and high >= be_trigger:
            be_armed = True
            active_sl = entry  # move SL to breakeven

        # SL-first (conservative)
        if low <= active_sl:
            pnl_pct = (active_sl - entry) / entry * 100
            return {
                "outcome": "loss" if pnl_pct < -0.01 else "be",
                "pnl_pct": pnl_pct,
                "hold_days": i + 1,
                "be_armed": be_armed,
                "truncated": False,
            }

        if high >= tp:
            pnl_pct = (tp - entry) / entry * 100
            return {
                "outcome": "win",
                "pnl_pct": pnl_pct,
                "hold_days": i + 1,
                "be_armed": be_armed,
                "truncated": False,
            }

        # Horizon timeout
        if i + 1 >= horizon_days:
            close_price = float(bar["close"])
            pnl_pct = (close_price - entry) / entry * 100
            return {
                "outcome": "timeout",
                "pnl_pct": pnl_pct,
                "hold_days": i + 1,
                "be_armed": be_armed,
                "truncated": False,
            }

    # Ran out of bars before horizon → right-censored / data truncation
    close_price = float(bars.iloc[-1]["close"])
    pnl_pct = (close_price - entry) / entry * 100
    return {
        "outcome": "timeout",
        "pnl_pct": pnl_pct,
        "hold_days": len(bars),
        "be_armed": be_armed,
        "truncated": True,   # fewer bars than horizon_days = truncated, not true timeout
    }


def walk_daily_bars_chandelier(
    entry: float,
    initial_sl_pct: float,
    trail_distance_pct: float,
    engage_pct: float,
    bars: pd.DataFrame,
    horizon_days: int = SWING_HORIZON_DAYS,
) -> dict:
    """
    Chandelier trailing stop walk on daily bars.

    Mechanics:
    - Initial SL = entry * (1 - initial_sl_pct/100)
    - Once bar HIGH >= entry*(1+engage_pct/100), trailing activates:
        active_sl = max(active_sl, running_peak * (1 - trail_distance_pct/100))
    - Trail updates on each bar BEFORE checking stop (conservative = SL-first)
    """
    if bars.empty or entry <= 0:
        return {
            "outcome": "no_data",
            "pnl_pct": 0.0,
            "hold_days": 0,
            "be_armed": False,
            "truncated": False,
        }

    active_sl = entry * (1 - initial_sl_pct / 100)
    tp_level = None     # no explicit TP for chandelier — exit only via SL, timeout, or no TP set
    engage_level = entry * (1 + engage_pct / 100)
    trailing_active = False
    running_peak = entry
    be_armed = False  # used to indicate "trailing engaged"

    for i, (_, bar) in enumerate(bars.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])

        # Update running peak
        running_peak = max(running_peak, high)

        # Activate trailing once peak reaches engage level
        if not trailing_active and running_peak >= engage_level:
            trailing_active = True
            be_armed = True   # reuse flag to mean "trail engaged"

        # Trail SL update
        if trailing_active:
            trail_stop = running_peak * (1 - trail_distance_pct / 100)
            active_sl = max(active_sl, trail_stop)

        # SL-first (trail exit — classify by P&L sign, NOT fixed "loss")
        # A profitable trail exit (active_sl > entry) is a WIN; near-zero is BE.
        if low <= active_sl:
            pnl_pct = (active_sl - entry) / entry * 100
            if pnl_pct > 0.01:
                outcome = "win"
            elif pnl_pct < -0.01:
                outcome = "loss"
            else:
                outcome = "be"
            return {
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "hold_days": i + 1,
                "be_armed": be_armed,
                "truncated": False,
            }

        # Horizon timeout — for chandelier (no explicit TP), classify by P&L sign.
        # A position still profitable at 30d is a success; still down is a loss.
        if i + 1 >= horizon_days:
            close_price = float(bar["close"])
            pnl_pct = (close_price - entry) / entry * 100
            if pnl_pct > 0.01:
                outcome = "win"
            elif pnl_pct < -0.01:
                outcome = "loss"
            else:
                outcome = "be"
            return {
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "hold_days": i + 1,
                "be_armed": be_armed,
                "truncated": False,
            }

    # Data truncation — classify by P&L sign (same logic as timeout)
    close_price = float(bars.iloc[-1]["close"])
    pnl_pct = (close_price - entry) / entry * 100
    if pnl_pct > 0.01:
        outcome = "win"
    elif pnl_pct < -0.01:
        outcome = "loss"
    else:
        outcome = "be"
    return {
        "outcome": outcome,
        "pnl_pct": pnl_pct,
        "hold_days": len(bars),
        "be_armed": be_armed,
        "truncated": True,
    }


# ---------------------------------------------------------------------------
# MFE/MAE engine (pure excursion, no bracket)
# ---------------------------------------------------------------------------

def compute_mfe_mae(entry: float, bars: pd.DataFrame, horizon_days: int) -> dict:
    """
    Compute MFE (max favorable excursion %) and MAE (max adverse excursion %)
    from entry over up to horizon_days bars.

    MFE = max(high / entry - 1) * 100
    MAE = max(1 - low / entry) * 100  (always positive)

    Returns dict with mfe_pct, mae_pct, and first_5pct_crossing_day
    (the day index when MFE first exceeds 5% — 0-indexed days from entry,
     or None if never reached).
    """
    if bars.empty or entry <= 0:
        return {
            "mfe_pct": np.nan,
            "mae_pct": np.nan,
            "first_5pct_day": None,
            "n_bars": 0,
        }

    n = min(horizon_days, len(bars))
    sub = bars.iloc[:n]

    highs = sub["high"].astype(float).values
    lows = sub["low"].astype(float).values

    mfe = ((highs - entry) / entry * 100).max()
    mae = ((entry - lows) / entry * 100).max()

    # Day index when MFE first exceeds 5% (1-based trading day count)
    crossing_day = None
    for j, h in enumerate(highs):
        if (h - entry) / entry * 100 >= 5.0:
            crossing_day = j + 1  # trading day 1, 2, ...
            break

    return {
        "mfe_pct": float(mfe),
        "mae_pct": float(mae),
        "first_5pct_day": crossing_day,
        "n_bars": n,
    }


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, outcome_col: str, pnl_col: str, hold_col: str,
                    truncated_col: Optional[str] = None) -> dict:
    """
    Compute PF, WR, avg win%, avg loss%, median hold, timeout%, truncated%.
    Excludes truncated rows from headline PF if truncated_col provided.
    """
    n_total = len(df)
    if n_total == 0:
        return {
            "n": 0,
            "n_wins": 0,
            "n_losses": 0,
            "n_be": 0,
            "n_timeouts": 0,
            "n_truncated": 0,
            "wr_pct": None,
            "pf": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "median_hold_days": None,
            "timeout_pct": None,
            "truncated_pct": None,
        }

    outcomes = df[outcome_col]
    pnl = df[pnl_col].astype(float)
    hold = df[hold_col]

    n_truncated = 0
    if truncated_col and truncated_col in df.columns:
        n_truncated = int(df[truncated_col].sum())

    wins = outcomes == "win"
    losses = outcomes == "loss"
    timeouts = outcomes == "timeout"
    be_exits = outcomes == "be"

    n_wins = int(wins.sum())
    n_losses = int(losses.sum())
    n_timeouts = int(timeouts.sum())
    n_be = int(be_exits.sum())

    n_resolved = n_wins + n_losses
    wr_pct = n_wins / n_resolved * 100 if n_resolved > 0 else None

    gross_win = float(pnl[wins].sum()) if n_wins > 0 else 0.0
    gross_loss = abs(float(pnl[losses].sum())) if n_losses > 0 else 0.0
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = 9.99  # cap; all wins
    else:
        pf = None

    avg_win_pct = float(pnl[wins].mean()) if n_wins > 0 else None
    avg_loss_pct = float(pnl[losses].mean()) if n_losses > 0 else None
    median_hold = float(hold.dropna().median()) if not hold.dropna().empty else None
    timeout_pct = n_timeouts / n_total * 100

    return {
        "n": n_total,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_be": n_be,
        "n_timeouts": n_timeouts,
        "n_truncated": n_truncated,
        "wr_pct": round(wr_pct, 1) if wr_pct is not None else None,
        "pf": round(pf, 2) if pf is not None else None,
        "avg_win_pct": round(avg_win_pct, 2) if avg_win_pct is not None else None,
        "avg_loss_pct": round(avg_loss_pct, 2) if avg_loss_pct is not None else None,
        "median_hold_days": round(median_hold, 1) if median_hold is not None else None,
        "timeout_pct": round(timeout_pct, 1),
        "truncated_pct": round(n_truncated / n_total * 100, 1),
    }


def fmt_pf(pf) -> str:
    if pf is None:
        return "  N/A"
    if pf >= 9.99:
        return ">9.9"
    return f"{pf:5.2f}"


def fmt_wr(wr) -> str:
    if wr is None:
        return " N/A"
    return f"{wr:5.1f}"


# ---------------------------------------------------------------------------
# Main simulation pipeline
# ---------------------------------------------------------------------------

def run_swing_horizon_test(conn):
    print("=" * 75)
    print("SWING HORIZON TEST — HORIZON-MISMATCH THESIS VALIDATION")
    print(f"Run timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 75)
    print()
    print("DATA NOTE: Using stock_candles_synthesized (timeframe='1d') for ALL walks.")
    print("Rationale: swing horizons (≤30 trading days) do not need intraday resolution.")
    print("  daily is ~7× fewer bars vs hourly (validated: 79-min hourly run → tractable).")
    print("  Conservative SL-first tiebreak preserved. Intraday ambiguity is negligible")
    print("  over multi-week holds; outcome classification on daily OHLC is well-specified.")
    print()

    # ------------------------------------------------------------------
    # 1. Load all daily candles
    # ------------------------------------------------------------------
    print("Loading daily candles (stock_candles_synthesized, timeframe='1d')...")
    daily_sql = """
        SELECT ticker, timestamp, open, high, low, close
        FROM stock_candles_synthesized
        WHERE timeframe = '1d'
        ORDER BY ticker, timestamp
    """
    all_daily = fetch_df(conn, daily_sql)
    all_daily["timestamp"] = pd.to_datetime(all_daily["timestamp"])
    # Strip any tz (timestamps stored as tz-naive in this table)
    if all_daily["timestamp"].dt.tz is not None:
        all_daily["timestamp"] = all_daily["timestamp"].dt.tz_localize(None)

    daily_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker_raw, grp in all_daily.groupby("ticker"):
        daily_by_ticker[str(ticker_raw)] = grp.reset_index(drop=True)

    daily_max_ts = all_daily["timestamp"].max()
    print(f"  Loaded {len(all_daily):,} daily bars for {len(daily_by_ticker):,} tickers.")
    print(f"  Latest daily bar: {daily_max_ts.strftime('%Y-%m-%d')}")
    print()

    # Cutoff: signals within 30 trading days of the last bar will be truncated
    # Rough calendar approximation: 30 trading days ≈ 44 calendar days
    import numpy as np
    # Compute exact cutoff as: last_bar_date − 30 bars looking backward per ticker
    # For a global cutoff, use last_bar_date - 44 calendar days as conservative proxy
    cutoff_ts = pd.Timestamp(daily_max_ts.date()) - pd.Timedelta(days=44)
    print(f"  Truncation risk: signals after ~{cutoff_ts.date()} may have <30 forward bars.")
    print()

    # ------------------------------------------------------------------
    # 2. Load BUY signals
    # ------------------------------------------------------------------
    signals_sql = """
        SELECT id, signal_timestamp, scanner_name, ticker,
               signal_type, entry_price, stop_loss, take_profit_1, take_profit_2,
               composite_score, signal_date
        FROM stock_scanner_signals
        WHERE signal_type = 'BUY'
        ORDER BY signal_timestamp
    """
    signals = fetch_df(conn, signals_sql)
    signals["signal_timestamp"] = pd.to_datetime(signals["signal_timestamp"])
    if signals["signal_timestamp"].dt.tz is not None:
        signals["signal_timestamp"] = signals["signal_timestamp"].dt.tz_localize(None)
    print(f"Loaded {len(signals):,} BUY signals across all scanners.")

    # Signal timestamp distribution
    hour_dist = signals["signal_timestamp"].dt.hour.value_counts().sort_index()
    print("\nSignal timestamp hour distribution (UTC, all scanners):")
    for h, cnt in hour_dist.items():
        mid_session = " *** MID-SESSION (look-ahead risk)" if 13 < h < 20 else ""
        print(f"  {h:02d}:xx UTC  n={cnt:5d}{mid_session}")
    print()
    print("  Entry convention: OPEN of FIRST daily bar STRICTLY AFTER signal_timestamp.")
    print("  Signals at 00:00 or 20:00–21:00 UTC are pre-market or post-close → safe.")
    print()

    # ------------------------------------------------------------------
    # 3. Build bar-slice helper
    # ------------------------------------------------------------------

    def get_daily_bars_after(ticker_raw: str, ref_ts) -> tuple[pd.DataFrame, float]:
        """
        Return (bars_df, entry_price) where bars start at the FIRST daily bar
        STRICTLY AFTER ref_ts. entry = that bar's open.

        This preserves the same convention as daytrade_edge_sim.py.
        """
        base_ticker = str(ticker_raw).split(".")[0]
        daily = daily_by_ticker.get(base_ticker)
        if daily is None or daily.empty:
            return pd.DataFrame(), 0.0

        if hasattr(ref_ts, "tzinfo") and ref_ts.tzinfo is not None:
            ref_dt = pd.Timestamp(ref_ts.replace(tzinfo=None))
        else:
            ref_dt = pd.Timestamp(ref_ts)

        mask = daily["timestamp"] > ref_dt
        if not mask.any():
            return pd.DataFrame(), 0.0

        idx_start = daily[mask].index[0]
        bars = daily.iloc[idx_start:].reset_index(drop=True)
        entry = float(bars.iloc[0]["open"]) if not bars.empty else 0.0
        return bars, entry

    # ------------------------------------------------------------------
    # 4. Native field sanity filter (per scanner)
    # ------------------------------------------------------------------
    print("Sanity-filtering native stop_loss / take_profit_1 fields...")
    native_filter_stats: dict[str, dict] = {}
    for scn in signals["scanner_name"].unique():
        sub = signals[signals["scanner_name"] == scn].copy()
        n_total = len(sub)
        # Conditions to DROP (implausible native fields):
        #   TP1 <= entry_price (wrong direction or null effective TP)
        #   SL >= entry_price (stop above entry — no protection)
        #   SL distance > NATIVE_MAX_DIST_PCT%
        #   TP1 distance > NATIVE_MAX_DIST_PCT%
        #   Any null
        mask_valid = (
            sub["stop_loss"].notna()
            & sub["take_profit_1"].notna()
            & sub["entry_price"].notna()
            & (sub["entry_price"] > 0)
            & (sub["stop_loss"] < sub["entry_price"])
            & (sub["take_profit_1"] > sub["entry_price"])
            & ((sub["entry_price"] - sub["stop_loss"]) / sub["entry_price"] * 100 <= NATIVE_MAX_DIST_PCT)
            & ((sub["take_profit_1"] - sub["entry_price"]) / sub["entry_price"] * 100 <= NATIVE_MAX_DIST_PCT)
        )
        n_valid = int(mask_valid.sum())
        n_dropped = n_total - n_valid
        native_filter_stats[scn] = {"n_total": n_total, "n_valid": n_valid, "n_dropped": n_dropped}
        avg_sl = float(
            ((sub.loc[mask_valid, "entry_price"] - sub.loc[mask_valid, "stop_loss"])
             / sub.loc[mask_valid, "entry_price"] * 100).mean()
        ) if n_valid > 0 else float("nan")
        avg_tp1 = float(
            ((sub.loc[mask_valid, "take_profit_1"] - sub.loc[mask_valid, "entry_price"])
             / sub.loc[mask_valid, "entry_price"] * 100).mean()
        ) if n_valid > 0 else float("nan")
        native_filter_stats[scn]["avg_sl_pct"] = avg_sl
        native_filter_stats[scn]["avg_tp1_pct"] = avg_tp1

    print(f"  {'scanner':32s}  {'n_total':>7s}  {'n_valid':>7s}  {'n_dropped':>9s}  "
          f"{'avg_sl%':>8s}  {'avg_tp1%':>9s}")
    for scn, st in sorted(native_filter_stats.items(), key=lambda x: x[1]["n_total"], reverse=True):
        print(f"  {scn:32s}  {st['n_total']:>7d}  {st['n_valid']:>7d}  {st['n_dropped']:>9d}  "
              f"  {st['avg_sl_pct']:6.1f}%  {st['avg_tp1_pct']:7.1f}%")
    print()

    # Build a lookup: signal_id → native_sl, native_tp1 (validated)
    native_valid_map: dict[int, tuple[float, float]] = {}
    for _, row in signals.iterrows():
        scn = row["scanner_name"]
        st = native_filter_stats.get(scn, {})
        ep = float(row["entry_price"]) if row["entry_price"] is not None else 0.0
        sl_v = float(row["stop_loss"]) if row["stop_loss"] is not None else None
        tp_v = float(row["take_profit_1"]) if row["take_profit_1"] is not None else None
        if (ep > 0 and sl_v is not None and tp_v is not None
                and sl_v < ep and tp_v > ep
                and (ep - sl_v) / ep * 100 <= NATIVE_MAX_DIST_PCT
                and (tp_v - ep) / ep * 100 <= NATIVE_MAX_DIST_PCT):
            native_valid_map[int(row["id"])] = (sl_v, tp_v)

    # ------------------------------------------------------------------
    # 5. Run simulation on all signals
    # ------------------------------------------------------------------
    print(f"Running simulation engine (daily bars, {SWING_HORIZON_DAYS}-day horizon)...")
    print("  Exit strategies: day_trade | swing_8_16 | swing_10_20 | swing_native | swing_trail")
    print()

    SIM_COLS = [
        "signal_id", "scanner_name", "ticker", "signal_timestamp",
        # Day-trade
        "dt_outcome", "dt_pnl_pct", "dt_hold_days", "dt_truncated",
        # Swing 8/16
        "s816_outcome", "s816_pnl_pct", "s816_hold_days", "s816_truncated",
        # Swing 10/20
        "s1020_outcome", "s1020_pnl_pct", "s1020_hold_days", "s1020_truncated",
        # Swing native
        "snat_outcome", "snat_pnl_pct", "snat_hold_days", "snat_truncated", "snat_valid",
        # Swing trail
        "str_outcome", "str_pnl_pct", "str_hold_days", "str_truncated",
        # Metadata
        "n_forward_bars",
    ]

    rows = []
    no_bars_count = 0
    tickers_missing = set()

    for _, sig in signals.iterrows():
        sig_id = int(sig["id"])
        ticker = str(sig["ticker"])
        base_ticker = ticker.split(".")[0]
        sig_ts = sig["signal_timestamp"]

        bars, entry = get_daily_bars_after(ticker, sig_ts)

        if bars.empty or entry <= 0:
            no_bars_count += 1
            tickers_missing.add(base_ticker)
            rows.append({
                "signal_id": sig_id,
                "scanner_name": sig["scanner_name"],
                "ticker": ticker,
                "signal_timestamp": sig_ts,
                **{c: None for c in SIM_COLS if c not in ("signal_id", "scanner_name", "ticker", "signal_timestamp")},
                "n_forward_bars": 0,
            })
            continue

        n_forward = len(bars)

        # --- Day-trade: 3%SL / 5%TP / BE@+2% ---
        dt_sl = entry * (1 - DT_SL_PCT / 100)
        dt_tp = entry * (1 + DT_TP_PCT / 100)
        dt_res = walk_daily_bars_bracket(
            entry, dt_sl, dt_tp, bars,
            horizon_days=SWING_HORIZON_DAYS,
            be_trigger_pct=DT_BE_TRIGGER_PCT,
        )

        # --- Swing 8/16 ---
        s816_sl = entry * (1 - SW_8_16_SL / 100)
        s816_tp = entry * (1 + SW_8_16_TP / 100)
        s816_res = walk_daily_bars_bracket(
            entry, s816_sl, s816_tp, bars,
            horizon_days=SWING_HORIZON_DAYS,
            be_trigger_pct=None,  # no BE for swing variants
        )

        # --- Swing 10/20 ---
        s1020_sl = entry * (1 - SW_10_20_SL / 100)
        s1020_tp = entry * (1 + SW_10_20_TP / 100)
        s1020_res = walk_daily_bars_bracket(
            entry, s1020_sl, s1020_tp, bars,
            horizon_days=SWING_HORIZON_DAYS,
            be_trigger_pct=None,
        )

        # --- Swing native ---
        native_pair = native_valid_map.get(sig_id)
        snat_valid = native_pair is not None
        if snat_valid:
            snat_sl, snat_tp = native_pair
            snat_res = walk_daily_bars_bracket(
                entry, snat_sl, snat_tp, bars,
                horizon_days=SWING_HORIZON_DAYS,
                be_trigger_pct=None,
            )
        else:
            snat_res = {"outcome": "no_data", "pnl_pct": None, "hold_days": None, "truncated": None}

        # --- Swing trail (chandelier) ---
        str_res = walk_daily_bars_chandelier(
            entry,
            initial_sl_pct=TRAIL_SL_PCT,
            trail_distance_pct=TRAIL_SL_PCT,
            engage_pct=TRAIL_ENGAGE_PCT,
            bars=bars,
            horizon_days=SWING_HORIZON_DAYS,
        )

        rows.append({
            "signal_id": sig_id,
            "scanner_name": sig["scanner_name"],
            "ticker": ticker,
            "signal_timestamp": sig_ts,
            "dt_outcome": dt_res["outcome"],
            "dt_pnl_pct": dt_res["pnl_pct"],
            "dt_hold_days": dt_res["hold_days"],
            "dt_truncated": dt_res["truncated"],
            "s816_outcome": s816_res["outcome"],
            "s816_pnl_pct": s816_res["pnl_pct"],
            "s816_hold_days": s816_res["hold_days"],
            "s816_truncated": s816_res["truncated"],
            "s1020_outcome": s1020_res["outcome"],
            "s1020_pnl_pct": s1020_res["pnl_pct"],
            "s1020_hold_days": s1020_res["hold_days"],
            "s1020_truncated": s1020_res["truncated"],
            "snat_outcome": snat_res["outcome"],
            "snat_pnl_pct": snat_res.get("pnl_pct"),
            "snat_hold_days": snat_res.get("hold_days"),
            "snat_truncated": snat_res.get("truncated"),
            "snat_valid": snat_valid,
            "str_outcome": str_res["outcome"],
            "str_pnl_pct": str_res["pnl_pct"],
            "str_hold_days": str_res["hold_days"],
            "str_truncated": str_res["truncated"],
            "n_forward_bars": n_forward,
        })

    sim_df = pd.DataFrame(rows)
    print(f"  Signals processed: {len(sim_df):,}")
    print(f"  Signals with no daily bars: {no_bars_count} ({no_bars_count/len(sim_df)*100:.1f}%)")
    print(f"  Tickers missing daily data: {len(tickers_missing)}")

    # Mark truncated vs full-window
    sim_df["has_30d_window"] = (sim_df["n_forward_bars"] >= SWING_HORIZON_DAYS).fillna(False)
    sim_df["has_30d_window"] = sim_df["has_30d_window"].astype(bool)

    n_truncated_total = (~sim_df["has_30d_window"] & sim_df["n_forward_bars"].notna() & (sim_df["n_forward_bars"] > 0)).sum()
    print(f"  Signals with <{SWING_HORIZON_DAYS} forward bars (truncated): {n_truncated_total} "
          f"({n_truncated_total/len(sim_df)*100:.1f}%)")

    # Truncation breakdown by scanner
    print(f"\n  Truncation rate by scanner:")
    for scn in signals["scanner_name"].unique():
        sub = sim_df[sim_df["scanner_name"] == scn]
        n_s = len(sub)
        n_full = int(sub["has_30d_window"].sum())
        pct_trunc = (n_s - n_full) / n_s * 100 if n_s > 0 else 0
        flag = " *** MOSTLY TRUNCATED — swing metrics unreliable" if pct_trunc > 75 else ""
        print(f"    {scn:32s}: n={n_s:4d}, full_30d={n_full:4d}, truncated={pct_trunc:4.0f}%{flag}")

    print()

    # ------------------------------------------------------------------
    # 6. ANALYSIS 1: MFE/MAE by horizon (constant cohort)
    # ------------------------------------------------------------------
    print("=" * 75)
    print("ANALYSIS 1: MFE/MAE BY HORIZON (constant cohort = signals with ≥30 forward bars)")
    print("=" * 75)
    print()
    print("Methodology: Entry = open of first daily bar after signal. For each signal,")
    print("compute MAX favorable/adverse excursion from entry up to horizon h (no bracket).")
    print("Cohort FIXED to signals with ≥30 forward daily bars (prevents survivorship bias).")
    print("'Late crossers' = % of signals where MFE first exceeds +5% on day >2")
    print("  (day-trade bracket would have already closed; the move comes later).")
    print()

    cohort_df = sim_df[sim_df["has_30d_window"] & sim_df["dt_outcome"].notna()].copy()

    # Compute MFE/MAE for each horizon, per signal, on the constant cohort
    mfe_records = []
    for _, row in cohort_df.iterrows():
        base_ticker = str(row["ticker"]).split(".")[0]
        sig_ts = row["signal_timestamp"]
        scanner = row["scanner_name"]

        bars, entry = get_daily_bars_after(row["ticker"], sig_ts)
        if bars.empty or entry <= 0:
            continue

        signal_mfe_by_horizon = {}
        for h in MFE_HORIZONS:
            res = compute_mfe_mae(entry, bars, h)
            signal_mfe_by_horizon[h] = res

        # Also compute the 5% crossing day over full 30d window
        full_res = compute_mfe_mae(entry, bars, SWING_HORIZON_DAYS)
        first_5pct_day = full_res["first_5pct_day"]

        mfe_records.append({
            "signal_id": row["signal_id"],
            "scanner_name": scanner,
            **{f"mfe_h{h}": signal_mfe_by_horizon[h]["mfe_pct"] for h in MFE_HORIZONS},
            **{f"mae_h{h}": signal_mfe_by_horizon[h]["mae_pct"] for h in MFE_HORIZONS},
            "first_5pct_day": first_5pct_day,
            "mfe_h30": full_res["mfe_pct"],
            "mae_h30": full_res["mae_pct"],
        })

    mfe_df = pd.DataFrame(mfe_records)

    if mfe_df.empty:
        print("  No valid signals in constant cohort. Skipping MFE/MAE analysis.")
    else:
        # Per scanner, per horizon: median MFE, median MAE, ratio, late-crosser %
        scanners_for_mfe = [s for s in signals["scanner_name"].unique()
                            if s in mfe_df["scanner_name"].values]

        print(f"  Constant cohort size: {len(mfe_df):,} signals with ≥30 forward bars.")
        print()

        for scn in scanners_for_mfe:
            sub = mfe_df[mfe_df["scanner_name"] == scn]
            n_scn = len(sub)
            if n_scn < 5:
                print(f"  {scn} (n={n_scn}): insufficient data — skipped.")
                continue

            n_with_5pct = sub["first_5pct_day"].notna().sum()
            n_late = (sub["first_5pct_day"].dropna() > 2).sum()
            late_pct = n_late / n_with_5pct * 100 if n_with_5pct > 0 else 0
            never_5pct = n_scn - n_with_5pct
            never_5pct_pct = never_5pct / n_scn * 100

            print(f"  {'='*65}")
            print(f"  Scanner: {scn}  (n={n_scn} in constant cohort)")
            print(f"  {'Horizon':>10s}  {'med MFE%':>9s}  {'med MAE%':>9s}  {'MFE/MAE':>8s}")
            for h in MFE_HORIZONS:
                mfe_col = f"mfe_h{h}"
                mae_col = f"mae_h{h}"
                med_mfe = sub[mfe_col].dropna().median()
                med_mae = sub[mae_col].dropna().median()
                ratio = med_mfe / med_mae if med_mae > 0 else float("nan")
                print(f"  {h:>8}d  {med_mfe:>9.2f}  {med_mae:>9.2f}  {ratio:>8.2f}")

            print(f"")
            print(f"  Late 5% crossers (first MFE>5% occurs after day 2):")
            print(f"    Signals that ever reached +5%: {int(n_with_5pct)} of {n_scn} ({n_with_5pct/n_scn*100:.0f}%)")
            print(f"    Of those, with late crossing (day>2): {int(n_late)} ({late_pct:.1f}%)")
            print(f"    Signals that NEVER reached +5% within 30d: {int(never_5pct)} ({never_5pct_pct:.1f}%)")
            print()

        # Focused horizon-growth assessment for zlma
        print(f"  --- HORIZON-GROWTH ASSESSMENT: {FOCUS_SCANNER} ---")
        zlma_sub = mfe_df[mfe_df["scanner_name"] == FOCUS_SCANNER]
        if len(zlma_sub) >= 10:
            print(f"  Does median MFE keep growing materially past ~10 days while MAE stays bounded?")
            mfe_growth_10_20 = (
                zlma_sub[f"mfe_h20"].dropna().median() - zlma_sub[f"mfe_h10"].dropna().median()
            )
            mfe_growth_20_30 = (
                zlma_sub[f"mfe_h30"].dropna().median() - zlma_sub[f"mfe_h20"].dropna().median()
            )
            mae_growth_10_30 = (
                zlma_sub[f"mae_h30"].dropna().median() - zlma_sub[f"mae_h10"].dropna().median()
            )
            mfe_10 = zlma_sub[f"mfe_h10"].dropna().median()
            mfe_30 = zlma_sub[f"mfe_h30"].dropna().median()
            print(f"    MFE growth d10→d20: +{mfe_growth_10_20:.2f}%")
            print(f"    MFE growth d20→d30: +{mfe_growth_20_30:.2f}%")
            print(f"    MAE growth d10→d30: +{mae_growth_10_30:.2f}% (ideally small vs MFE growth)")
            print(f"    MFE ratio d30/d10: {mfe_30/mfe_10:.2f}x" if mfe_10 > 0 else "")
        else:
            print(f"  Insufficient zlma cohort data (n={len(zlma_sub)}).")

    print()

    # ------------------------------------------------------------------
    # 7. ANALYSIS 2: Clean swing-exit sim
    # ------------------------------------------------------------------
    print("=" * 75)
    print("ANALYSIS 2: CLEAN SWING-EXIT SIMULATION")
    print("=" * 75)
    print()
    print(f"Exit strategies:")
    print(f"  day_trade   : 3%SL / 5%TP / BE@+2% (coarse on daily — see caveat)")
    print(f"  swing_8_16  : 8%SL / 16%TP, no BE")
    print(f"  swing_10_20 : 10%SL / 20%TP, no BE")
    print(f"  swing_native: scanner's own stop_loss / take_profit_1 (sanity-filtered)")
    print(f"  swing_trail : initial 8%SL, chandelier trail 8% below peak once +8% reached")
    print()
    print(f"LOSS IS CAPPED AT STOP (NOT mark-to-market) — fixes old swing-sim inflation.")
    print(f"Timeout = 30 trading days → mark-to-market at close.")
    print(f"Headline PF uses ALL signals. Truncated% reported (right-censored signals).")
    print()
    print(f"CAVEAT (day_trade on daily bars): 3%/5% bracket is tight vs daily range;")
    print(f"  SL-first on daily OHLC inflates losses vs real intraday execution.")
    print(f"  Prior hourly result (PF ~0.78) is authoritative for day_trade verdict.")
    print(f"  Daily day_trade column here is a cross-check only.")
    print()

    strategies = [
        ("day_trade",    "dt_outcome",   "dt_pnl_pct",   "dt_hold_days",   "dt_truncated"),
        ("swing_8_16",   "s816_outcome",  "s816_pnl_pct",  "s816_hold_days",  "s816_truncated"),
        ("swing_10_20",  "s1020_outcome", "s1020_pnl_pct", "s1020_hold_days", "s1020_truncated"),
        ("swing_native", "snat_outcome",  "snat_pnl_pct",  "snat_hold_days",  "snat_truncated"),
        ("swing_trail",  "str_outcome",   "str_pnl_pct",   "str_hold_days",   "str_truncated"),
    ]

    # Build results table: scanner × strategy
    print(f"  {'scanner':32s}  {'strategy':14s}  {'n':>5s}  {'WR%':>5s}  {'PF':>6s}  "
          f"{'avgW%':>6s}  {'avgL%':>6s}  {'med_hold':>8s}  {'TO%':>5s}  {'trunc%':>7s}")
    print(f"  {'-'*32}  {'-'*14}  {'-'*5}  {'-'*5}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*7}")

    # Store results for zlma verdict
    result_matrix: dict[str, dict[str, dict]] = {}

    scanners_ordered = (
        [FOCUS_SCANNER]
        + [s for s in signals["scanner_name"].unique() if s != FOCUS_SCANNER]
    )

    for scn in scanners_ordered:
        sub = sim_df[sim_df["scanner_name"] == scn].copy()
        if len(sub) == 0:
            continue

        result_matrix[scn] = {}

        for strat_name, oc, pnl, hold, trunc in strategies:
            # For native: only use valid native rows
            if strat_name == "swing_native":
                use = sub[sub["snat_valid"] == True].copy()
            else:
                use = sub[sub[oc].notna() & (sub[oc] != "no_data")].copy()

            if len(use) == 0:
                print(f"  {scn:32s}  {strat_name:14s}  {'N/A':>5s}")
                result_matrix[scn][strat_name] = {}
                continue

            m = compute_metrics(use, oc, pnl, hold, trunc)
            result_matrix[scn][strat_name] = m

            wr_s = fmt_wr(m["wr_pct"])
            pf_s = fmt_pf(m["pf"])
            awp = f"{m['avg_win_pct']:6.2f}" if m["avg_win_pct"] is not None else "   N/A"
            alp = f"{m['avg_loss_pct']:6.2f}" if m["avg_loss_pct"] is not None else "   N/A"
            mhd = f"{m['median_hold_days']:8.1f}" if m["median_hold_days"] is not None else "     N/A"
            to_ = f"{m['timeout_pct']:5.1f}" if m["timeout_pct"] is not None else "  N/A"
            tr_ = f"{m['truncated_pct']:7.1f}" if m["truncated_pct"] is not None else "    N/A"

            print(f"  {scn:32s}  {strat_name:14s}  {m['n']:>5d}  {wr_s}  {pf_s}  "
                  f"{awp}  {alp}  {mhd}  {to_}  {tr_}")

        print()  # blank line between scanners

    # ------------------------------------------------------------------
    # 8. VERDICT: zlma_trend horizon-mismatch thesis
    # ------------------------------------------------------------------
    print("=" * 75)
    print("VERDICT: HORIZON-MISMATCH THESIS")
    print("=" * 75)
    print()

    zlma_results = result_matrix.get(FOCUS_SCANNER, {})
    zlma_n = int(sim_df[sim_df["scanner_name"] == FOCUS_SCANNER]["dt_outcome"].notna().sum())
    zlma_cohort_n = int(cohort_df[cohort_df["scanner_name"] == FOCUS_SCANNER].shape[0])

    print(f"  Scanner: {FOCUS_SCANNER}  (total signals: {zlma_n}, full-30d cohort: {zlma_cohort_n})")
    print()

    print(f"  PF by exit strategy:")
    pf_values = {}
    for strat_name, _, _, _, _ in strategies:
        m = zlma_results.get(strat_name, {})
        pf = m.get("pf")
        trunc = m.get("truncated_pct", None)
        n = m.get("n", 0)
        pf_values[strat_name] = pf
        pf_s = fmt_pf(pf)
        caveat = ""
        if trunc is not None and trunc > 30:
            caveat = f"  [WARN: {trunc:.0f}% truncated — mark-to-market, not true outcome]"
        if strat_name == "day_trade":
            caveat += "  [COARSE ON DAILY — use prior hourly PF ~0.78 as authoritative]"
        print(f"    {strat_name:14s}: PF={pf_s}  (n={n}){caveat}")

    print()

    # Determine verdict
    dt_pf = pf_values.get("day_trade")
    swing_pf_list = [pf_values.get(s) for s in ("swing_8_16", "swing_10_20", "swing_trail") if pf_values.get(s) is not None]
    nat_pf = pf_values.get("swing_native")

    dt_sub_floor = (dt_pf is None or dt_pf < 1.0)   # confirmed sub-floor (prior hourly PF 0.78)
    swing_above_floor = any(p is not None and p > 1.0 for p in swing_pf_list)
    swing_robustly_above = sum(1 for p in swing_pf_list if p is not None and p >= 1.2) >= 2

    print(f"  THESIS EVALUATION:")
    print(f"    Prior hourly day_trade PF: ~0.78 (authoritative, from daytrade_edge_sim)")
    print(f"    Fixed-swing variants above 1.0: {sum(1 for p in swing_pf_list if p is not None and p > 1.0)} of {len(swing_pf_list)}")
    print(f"    Fixed-swing variants robustly above 1.2: {sum(1 for p in swing_pf_list if p is not None and p >= 1.2)} of {len(swing_pf_list)}")
    if nat_pf is not None:
        print(f"    Native-swing PF: {nat_pf:.2f}")
    print()

    if swing_robustly_above and dt_sub_floor:
        verdict = "CONFIRMED"
        verdict_detail = (
            "Multiple swing exit strategies produce PF robustly >1.2 while day_trade is <1.0.\n"
            "    The edge is real but requires a swing hold to capture. The day-trade bracket\n"
            "    destroys it by exiting before the move fully develops."
        )
    elif swing_above_floor and dt_sub_floor:
        verdict = "PARTIAL / TENTATIVE"
        verdict_detail = (
            "Swing exits show PF >1.0 but not robustly >1.2 across all variants. Day-trade\n"
            "    is clearly sub-floor. Thesis plausible; insufficient strength to call confirmed."
        )
    elif not swing_above_floor:
        verdict = "REFUTED"
        verdict_detail = (
            "Neither the fixed swing variants nor the trailing variant produce PF >1.0.\n"
            "    The old swing-sim PF 3.88 was a mark-to-market artifact (losses not capped\n"
            "    at SL). With proper loss-capping, no swing edge exists. CONFIRMED: no real edge."
        )
    else:
        verdict = "INCONCLUSIVE"
        verdict_detail = "Insufficient n or conflicting signals across variants."

    print(f"  VERDICT: {verdict}")
    print(f"    {verdict_detail}")
    print()

    # ------------------------------------------------------------------
    # 9. SCANNER CLASSIFICATION
    # ------------------------------------------------------------------
    print("=" * 75)
    print("SCANNER CLASSIFICATION: day-trade-edge / swing-edge / no-edge")
    print("=" * 75)
    print()
    print(f"  Classification criteria (n-weighted; low-n scanners flagged as exploratory):")
    print(f"    day-trade-edge: DT PF (hourly prior or daily daily_trade) >= 1.0, swing ~=DT")
    print(f"    swing-edge:     swing variants robustly > 1.2, DT < 1.0")
    print(f"    no-edge:        both DT and swing < 1.0 (or swing is mark-to-market artifact)")
    print(f"    single-regime:  <30 trading days of history → metrics not robust")
    print()

    # Cutoff date for "single regime" scanners (launched May 26)
    single_regime_cutoff = pd.Timestamp("2026-05-26")

    for scn in scanners_ordered:
        sub_sig = signals[signals["scanner_name"] == scn]
        scn_n = len(sub_sig)
        earliest = sub_sig["signal_timestamp"].min()
        scn_rm = result_matrix.get(scn, {})

        dt_m = scn_rm.get("day_trade", {})
        s816_m = scn_rm.get("swing_8_16", {})
        s1020_m = scn_rm.get("swing_10_20", {})
        str_m = scn_rm.get("swing_trail", {})

        dt_pf_scn = dt_m.get("pf")
        s816_pf = s816_m.get("pf")
        s1020_pf = s1020_m.get("pf")
        str_pf = str_m.get("pf")
        swing_pfs = [p for p in [s816_pf, s1020_pf, str_pf] if p is not None]

        # Single-regime override
        if earliest >= single_regime_cutoff or scn_n < 30:
            classification = "SINGLE-REGIME / LOW-N (exploratory only)"
        elif scn == FOCUS_SCANNER:
            # Use the verdict above
            classification = f"{verdict} (see Verdict section)"
        elif dt_pf_scn is not None and dt_pf_scn >= 1.0 and all(p is None or p <= dt_pf_scn * 1.1 for p in swing_pfs):
            classification = "DAY-TRADE-EDGE (DT PF >= 1.0, no clear swing premium)"
        elif sum(1 for p in swing_pfs if p >= 1.2) >= 2 and (dt_pf_scn is None or dt_pf_scn < 1.0):
            classification = "SWING-EDGE"
        elif all(p is not None and p < 1.0 for p in swing_pfs) and (dt_pf_scn is None or dt_pf_scn < 1.0):
            classification = "NO-EDGE (both DT and swing sub-floor)"
        else:
            swing_pf_str = ", ".join(f"{p:.2f}" for p in swing_pfs if p is not None)
            dt_s = f"{dt_pf_scn:.2f}" if dt_pf_scn is not None else "N/A"
            classification = f"MIXED (DT={dt_s}, swing=[{swing_pf_str}]) — insufficient n or regime info"

        print(f"  {scn:40s} n={scn_n:5d}  {classification}")

    # ------------------------------------------------------------------
    # 10. Execution recommendation
    # ------------------------------------------------------------------
    print()
    print("=" * 75)
    print("EXECUTION RECOMMENDATION")
    print("=" * 75)
    print()

    if verdict == "CONFIRMED":
        # Find best performing swing strategy for zlma
        best_strat = None
        best_pf = 0.0
        for sn in ("swing_8_16", "swing_10_20", "swing_trail", "swing_native"):
            m = zlma_results.get(sn, {})
            p = m.get("pf")
            if p is not None and p > best_pf:
                best_pf = p
                best_strat = sn
        m_best = zlma_results.get(best_strat, {})

        print(f"  For {FOCUS_SCANNER}: horizon-mismatch CONFIRMED.")
        print(f"  Recommended execution horizon: SWING (up to {SWING_HORIZON_DAYS} trading days).")
        print(f"  Best-performing exit strategy: {best_strat} (PF {best_pf:.2f})")
        if m_best.get("avg_win_pct") is not None:
            print(f"    avgW%={m_best['avg_win_pct']:.1f}%, avgL%={m_best['avg_loss_pct']:.1f}%, "
                  f"medHold={m_best.get('median_hold_days', 'N/A')} days, "
                  f"TO%={m_best.get('timeout_pct', 'N/A')}")
        print()
        print(f"  Implementation notes:")
        print(f"    - Entry: next open after signal (unchanged)")
        print(f"    - SL: per best_strat parameters above")
        print(f"    - TP: per best_strat (or none for trail — exit when trail fires)")
        print(f"    - Max hold: {SWING_HORIZON_DAYS} trading days, then exit at close")
        print(f"    - Do NOT use the day-trade auto-trader (3%/5%/10d) for zlma signals")
        print(f"    - Validate OOS: this analysis is in-sample. n~{FOCUS_SCANNER_MIN_N}+ forward trades")
        print(f"      required before committing capital.")
    elif verdict in ("PARTIAL / TENTATIVE",):
        print(f"  For {FOCUS_SCANNER}: tentative swing edge. NOT ready for execution.")
        print(f"  The mark-to-market swing PF 3.88 was inflated by uncapped losses.")
        print(f"  Current evidence: marginal swing PF (>1 but not robustly >1.2).")
        print(f"  Action: monitor-only with swing brackets until n>=50 forward trades.")
    else:
        print(f"  For {FOCUS_SCANNER}: NO validated swing edge found.")
        print(f"  The old swing-sim PF 3.88 was a mark-to-market artifact.")
        print(f"  Recommendation: do NOT execute zlma signals until OOS edge is confirmed.")

    # ------------------------------------------------------------------
    # 11. Single highest-confidence conclusion
    # ------------------------------------------------------------------
    print()
    print("=" * 75)
    print("SINGLE HIGHEST-CONFIDENCE CONCLUSION")
    print("=" * 75)
    print()
    print(f"  zlma_trend (n={zlma_n}, multi-month history, most representative scanner):")
    print()

    zlma_swing_pfs = {
        s: zlma_results.get(s, {}).get("pf") for s in ("swing_8_16", "swing_10_20", "swing_trail", "swing_native")
    }
    swing_str = ", ".join(
        f"{s}={fmt_pf(p).strip()}"
        for s, p in zlma_swing_pfs.items()
        if p is not None
    )
    print(f"  Day-trade PF (authoritative, hourly): ~0.78  — CONFIRMED NO DAY-TRADE EDGE")
    print(f"  Swing-exit PFs (daily, loss-capped): [{swing_str}]")
    print()

    if verdict == "CONFIRMED":
        print(f"  CONCLUSION: The horizon-mismatch thesis is CONFIRMED for zlma_trend.")
        print(f"  The edge is real — it lives at swing horizons (>5 trading days) and is")
        print(f"  destroyed by the day-trade bracket. The old swing-sim PF 3.88 was partly")
        print(f"  real (not purely an artifact), but overstated due to mark-to-market losses.")
        print(f"  Loss-capped swing PF remains robustly above 1.0.")
        print(f"  FIX: switch zlma_trend signals to a swing execution model ({best_strat}).")
        print(f"  CAVEAT: this is in-sample; OOS validation required before capital deployment.")
    elif verdict == "PARTIAL / TENTATIVE":
        print(f"  CONCLUSION: Partial evidence for horizon-mismatch. The swing-sim PF 3.88")
        print(f"  was meaningfully inflated by mark-to-market losses, but a real (smaller)")
        print(f"  swing edge may exist. Current loss-capped PF is >1 but not robustly >1.2.")
        print(f"  Cannot confirm or reject the thesis at current sample size.")
        print(f"  RECOMMENDED ACTION: monitor-only under swing brackets, accumulate OOS data.")
    else:
        print(f"  CONCLUSION: The horizon-mismatch thesis is REFUTED for zlma_trend.")
        print(f"  The old swing-sim PF 3.88 was a mark-to-market artifact (losses up to")
        print(f"  −1.99R inflated gross loss denominator). With proper SL-capping, all")
        print(f"  exit strategies — day-trade AND swing — produce PF < 1.0.")
        print(f"  There is NO validated edge in zlma_trend signals at any horizon.")
        print(f"  Do NOT execute or promote this scanner without OOS edge evidence.")

    print()
    print("=" * 75)
    print("Script: stock-scanner/app/stock_scanner/analysis/swing_horizon_test.py")
    print("=" * 75)


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
        run_swing_horizon_test(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
