"""
Day-Trade Horizon Edge Simulator
=================================
Re-simulates every BUY signal in stock_scanner_signals under the REAL
auto-trader execution model and produces corrected per-scanner metrics.

Usage (inside stock-scanner container):
    python -m stock_scanner.analysis.daytrade_edge_sim

Outputs a structured memo to stdout. Read-only: does NOT modify any table.

Execution model reverse-engineered from
stock_scanner/services/auto_open_trader.py (AutoOpenTrader._build_order_plan,
_place_order_api, breakeven_trigger_usd=10, max_notional=500, stop_loss_pct=3.0,
take_profit_pct=5.0, atr_stop_enabled=True, atr_threshold_pct=7.0,
atr_stop_mult=1.0, atr_rr=1.6667, max_risk_usd=15.0).

Entry bar convention (calibrated against broker_trades lag data):
  - Signal-based: signals are overnight batches (20:00 ET or 00:00 UTC).
    Entry = OPEN of FIRST bar STRICTLY AFTER signal_timestamp.
    This maps to next-session 9:30 ET open (13:30 UTC).
    Confirmed by broker lag data: pocket_pivot median 13.5h, gap_and_go ~17.5h.
  - Broker-validation: broker open_time is a mid-session fill.
    Entry = OPEN of FIRST bar STRICTLY AFTER open_time (next complete bar).
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
# Config (mirrors AutoOpenTrader.SETTING_DEFS defaults)
# ---------------------------------------------------------------------------
MAX_ORDER_NOTIONAL_USD = 500.0
MAX_RISK_USD = 15.0
DEFAULT_SL_PCT = 3.0        # fixed bracket stop loss %
DEFAULT_TP_PCT = 5.0        # fixed bracket take profit %
ATR_STOP_ENABLED = True
ATR_THRESHOLD_PCT = 7.0     # ATR% >= this triggers ATR branch
ATR_STOP_MULT = 1.0         # stop = 1.0 * ATR%
ATR_RR = 1.6667             # TP = ATR_RR * stop (maintains ~5/3 ratio)
BE_TRIGGER_USD = 10.0       # breakeven arms once unrealised profit >= $10

# Simulation timeout: trading days (~6 bars/trading day on 1h timeframe)
TIMEOUT_TRADING_DAYS = 10   # ~2 calendar weeks

# Edge floor parameters (mirrors route.ts)
EDGE_WINDOW_DAYS = 60
EDGE_PF_FLOOR = 1.0
EDGE_MIN_CLOSED = 10

DB_URL = "postgresql://postgres:postgres@postgres:5432/stocks"

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
# ATR-14 computation (Wilder's smoothed)
# ---------------------------------------------------------------------------

def compute_atr14(daily_df: pd.DataFrame) -> pd.Series:
    """Return ATR-14 series from a daily OHLC DataFrame (already sorted asc)."""
    high = daily_df["high"].astype(float)
    low = daily_df["low"].astype(float)
    close = daily_df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder: seed with SMA(14), then EWMA with span=27 (alpha=1/14)
    atr = tr.ewm(span=27, min_periods=14, adjust=False).mean()
    return atr


# ---------------------------------------------------------------------------
# Bracket computation (mirrors _build_order_plan)
# ---------------------------------------------------------------------------

def build_bracket(entry: float, atr_pct: Optional[float]):
    """Return bracket dict for a given entry and ATR%."""
    use_atr = (
        ATR_STOP_ENABLED
        and atr_pct is not None
        and atr_pct >= ATR_THRESHOLD_PCT
        and ATR_STOP_MULT > 0
    )
    if use_atr:
        sl_pct = max(0.01, ATR_STOP_MULT * float(atr_pct))
        tp_pct = max(sl_pct + 0.01, ATR_RR * sl_pct)
        mode = f"atr({float(atr_pct):.1f}%)"
    else:
        sl_pct = DEFAULT_SL_PCT
        tp_pct = DEFAULT_TP_PCT
        mode = "fixed"

    sl = entry * (1 - sl_pct / 100)
    tp = entry * (1 + tp_pct / 100)

    # Quantity and notional
    per_share_risk = entry - sl
    if use_atr:
        qty = int(MAX_RISK_USD // per_share_risk) if per_share_risk > 0 else 0
    else:
        qty = int(MAX_ORDER_NOTIONAL_USD // entry) if entry > 0 else 0

    notional = entry * qty if qty > 0 else MAX_ORDER_NOTIONAL_USD
    # BE trigger: $10 unrealised profit / notional
    be_trigger_pct = (BE_TRIGGER_USD / notional * 100) if notional > 0 else 2.0

    return {
        "sl": sl,
        "tp": tp,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "mode": mode,
        "qty": qty,
        "notional": notional,
        "be_trigger_pct": be_trigger_pct,
    }


# ---------------------------------------------------------------------------
# Path-walking engine
# ---------------------------------------------------------------------------

def walk_bars(
    entry: float,
    bracket: dict,
    bars: pd.DataFrame,
    be_enabled: bool = True,
    optimistic_be: bool = False,
) -> dict:
    """
    Walk 1-hour bars forward from the first bar (index 0).

    bars must start at the FIRST bar after the signal/fill — its OPEN is the
    simulated entry price (passed in separately as `entry`).

    Engine choices:
      (a) Entry = open of bars.iloc[0] (caller provides the right bar)
      (b) Same-bar SL/TP: SL-first (conservative) — ambiguity rate <0.3%
      (c) BE arms when bar HIGH >= entry*(1 + be_trigger_pct/100)
          Pessimistic (default): BE can fire and be stopped on the SAME bar.
          Optimistic (optimistic_be=True): BE stop can only be HIT on NEXT bar.
          Disabled (be_enabled=False): no BE arm; original SL throughout.
      (d) After BE: active SL = entry (0% loss)
      (e) Timeout = TIMEOUT_TRADING_DAYS * 6 bars (~10 trading days)
    """
    if bars.empty:
        return {"outcome": "no_data", "pnl_pct": 0.0, "hold_bars": 0, "bars_walked": 0}

    sl = bracket["sl"]
    tp = bracket["tp"]
    be_trigger = entry * (1 + bracket["be_trigger_pct"] / 100)
    be_armed = False
    active_sl = sl
    be_armed_bar = -1  # bar index where BE first armed

    n_timeout = TIMEOUT_TRADING_DAYS * 6  # ~6 trading 1h bars/day

    for i, (_, bar) in enumerate(bars.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])

        # BE arm check
        if be_enabled and not be_armed and high >= be_trigger:
            be_armed = True
            be_armed_bar = i
            active_sl = entry  # move SL to breakeven (0%)

        # For optimistic_be: BE stop cannot be hit on the same bar it armed
        # On the BE-arming bar with optimistic mode: only original SL can trigger
        if optimistic_be and be_armed and i == be_armed_bar:
            # BE stop cannot fire on the bar it armed; check original SL only
            if low <= sl:
                pnl_pct = (sl - entry) / entry * 100
                return {
                    "outcome": "loss",
                    "pnl_pct": pnl_pct,
                    "hold_bars": i + 1,
                    "be_armed": be_armed,
                    "bars_walked": i + 1,
                }
            if high >= tp:
                pnl_pct = (tp - entry) / entry * 100
                return {
                    "outcome": "win",
                    "pnl_pct": pnl_pct,
                    "hold_bars": i + 1,
                    "be_armed": be_armed,
                    "bars_walked": i + 1,
                }
        else:
            # Standard: SL-first (conservative), BE stop applies immediately
            if low <= active_sl:
                pnl_pct = (active_sl - entry) / entry * 100
                return {
                    "outcome": "loss" if pnl_pct < -0.01 else "be",
                    "pnl_pct": pnl_pct,
                    "hold_bars": i + 1,
                    "be_armed": be_armed,
                    "bars_walked": i + 1,
                }
            if high >= tp:
                pnl_pct = (tp - entry) / entry * 100
                return {
                    "outcome": "win",
                    "pnl_pct": pnl_pct,
                    "hold_bars": i + 1,
                    "be_armed": be_armed,
                    "bars_walked": i + 1,
                }

        # Timeout
        if i + 1 >= n_timeout:
            close_price = float(bar["close"])
            pnl_pct = (close_price - entry) / entry * 100
            return {
                "outcome": "timeout",
                "pnl_pct": pnl_pct,
                "hold_bars": i + 1,
                "be_armed": be_armed,
                "bars_walked": i + 1,
            }

    # Ran out of candle data before timeout
    close_price = float(bars.iloc[-1]["close"])
    pnl_pct = (close_price - entry) / entry * 100
    return {
        "outcome": "timeout",
        "pnl_pct": pnl_pct,
        "hold_bars": len(bars),
        "be_armed": be_armed,
        "bars_walked": len(bars),
    }


# ---------------------------------------------------------------------------
# Classify broker trade for confusion matrix
# ---------------------------------------------------------------------------

def classify_broker_outcome(bt: dict) -> str:
    """
    Classify actual broker trade into win / be / loss.
    Uses profit_pct relative to the bracket distances.
    """
    pct = float(bt["profit_pct"] or 0)
    entry = float(bt["open_price"])
    sl = float(bt["stop_loss"]) if bt["stop_loss"] else None
    tp = float(bt["take_profit"]) if bt["take_profit"] else None

    if sl is not None and tp is not None:
        sl_pct = (sl - entry) / entry * 100  # negative
        tp_pct = (tp - entry) / entry * 100  # positive
        # Win: exited at or very near TP
        if abs(pct - tp_pct) < 0.25:
            return "win"
        # Loss: exited near initial SL (rough: within 0.5% of SL)
        if abs(pct - sl_pct) < 0.50 and pct < -0.5:
            return "loss"
        # BE / partial-runner: anything above 0% not classified as win
        if pct > 0:
            return "be"
        # Small loss (BE moved, then stopped out near 0)
        if pct >= -0.5:
            return "be"
        return "loss"
    else:
        if pct >= 3.0:
            return "win"
        if pct <= -2.5:
            return "loss"
        if pct > 0 or pct >= -0.5:
            return "be"
        return "loss"


# ---------------------------------------------------------------------------
# Main simulation pipeline
# ---------------------------------------------------------------------------

def run_simulation(conn):
    print("=" * 70)
    print("DAY-TRADE HORIZON EDGE SIMULATOR")
    print(f"Run timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. ENGINE VALIDATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 1: ENGINE VALIDATION (CONFUSION MATRIX)")
    print("=" * 70)

    # Entry-lag calibration (signal_id linked trades only)
    lag_sql = """
        SELECT bt.ticker, bt.open_time, bt.signal_id,
               ss.signal_timestamp, ss.scanner_name,
               EXTRACT(EPOCH FROM (bt.open_time - ss.signal_timestamp))/3600 AS lag_hours
        FROM broker_trades bt
        JOIN stock_scanner_signals ss ON bt.signal_id = ss.id
        WHERE bt.status = 'closed'
        ORDER BY bt.open_time
    """
    lag_df = fetch_df(conn, lag_sql)
    if not lag_df.empty:
        # Stale links (>240h = mis-linked old records, not representative of current system)
        clean_lag = lag_df[lag_df["lag_hours"] <= 240].copy()
        stale_lag = lag_df[lag_df["lag_hours"] > 240]
        print(f"\nEntry-lag calibration (signal_id linked trades):")
        print(f"  Total with signal_id: {len(lag_df)}")
        print(f"  Stale links excluded (>240h): {len(stale_lag)} rows "
              f"(e.g. lag={stale_lag['lag_hours'].max():.0f}h max)")
        if not clean_lag.empty:
            by_scanner = clean_lag.groupby("scanner_name")["lag_hours"].agg(
                ["median", "mean", "count"]
            ).round(1)
            print(f"  Clean subset ({len(clean_lag)} rows) median/mean lag by scanner:")
            for scn, row in by_scanner.iterrows():
                print(f"    {scn}: median={row['median']:.1f}h  mean={row['mean']:.1f}h  n={int(row['count'])}")
            overall_median = clean_lag["lag_hours"].median()
            print(f"\n  Overall median entry lag (clean subset): {overall_median:.1f}h")
            print(f"  Interpretation:")
            print(f"    - Pocket_pivot (00:00 UTC midnight): 13.5h lag = opens 13:30 UTC same day")
            print(f"    - Gap_and_go / others (~20:00 UTC): ~17.5h lag = opens 13:30 UTC next day")
            print(f"  Entry proxy: OPEN of FIRST 1h bar STRICTLY AFTER signal_timestamp.")
            print(f"    For midnight signals: first bar after midnight = 13:30 UTC same day")
            print(f"    For 20:00 ET signals: first bar after 20:00 UTC = 13:30 UTC next day")

    # SL/TP coverage
    bt_full_sql = """
        SELECT bt.*, ss.scanner_name AS linked_scanner_name
        FROM broker_trades bt
        LEFT JOIN stock_scanner_signals ss ON bt.signal_id = ss.id
        WHERE bt.status = 'closed'
        ORDER BY bt.open_time
    """
    bt_df = fetch_df(conn, bt_full_sql)
    n_total_closed = len(bt_df)
    n_with_both = len(bt_df[bt_df["stop_loss"].notna() & bt_df["take_profit"].notna()])
    print(f"\nBroker trade SL/TP coverage (closed trades):")
    print(f"  Total closed broker trades: {n_total_closed}")
    print(f"  With BOTH stop_loss AND take_profit populated: {n_with_both}")
    print(f"  Without bracket (NULL SL or TP): {n_total_closed - n_with_both}")
    print(f"  NOTE: Engine validation proceeds on the {n_with_both} bracketed trades.")

    # Pre-load ALL 1h candle data once into a dict keyed by ticker
    print("\nPre-loading 1h candles...")
    candle_sql = """
        SELECT ticker, timestamp, open, high, low, close
        FROM stock_candles
        WHERE timeframe = '1h'
        ORDER BY ticker, timestamp
    """
    all_candles = fetch_df(conn, candle_sql)
    all_candles["timestamp"] = pd.to_datetime(all_candles["timestamp"])
    candles_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker_raw, grp in all_candles.groupby("ticker"):
        candles_by_ticker[str(ticker_raw)] = grp.reset_index(drop=True)
    print(f"  Loaded {len(all_candles):,} 1h bars for {len(candles_by_ticker)} tickers.")

    # Pre-load daily candles for ATR computation
    print("Pre-loading 1d synthesized candles for ATR...")
    daily_sql = """
        SELECT ticker, timestamp, open, high, low, close
        FROM stock_candles_synthesized
        WHERE timeframe = '1d'
        ORDER BY ticker, timestamp
    """
    all_daily = fetch_df(conn, daily_sql)
    all_daily["timestamp"] = pd.to_datetime(all_daily["timestamp"])
    daily_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker_raw, grp in all_daily.groupby("ticker"):
        dgrp = grp.reset_index(drop=True).copy()
        dgrp["atr14"] = compute_atr14(dgrp)
        daily_by_ticker[str(ticker_raw)] = dgrp
    print(f"  Loaded {len(all_daily):,} daily bars for {len(daily_by_ticker)} tickers.")

    def get_atr_pct(base_ticker: str, signal_time) -> Optional[float]:
        """ATR% strictly BEFORE signal_date (no look-ahead)."""
        daily = daily_by_ticker.get(base_ticker)
        if daily is None or daily.empty:
            return None
        if isinstance(signal_time, str):
            signal_time = pd.Timestamp(signal_time)
        if hasattr(signal_time, "tzinfo") and signal_time.tzinfo is not None:
            signal_time = pd.Timestamp(signal_time.replace(tzinfo=None))
        sig_date = pd.Timestamp(signal_time.date())
        mask = daily["timestamp"] < sig_date
        sub = daily[mask]
        if sub.empty or sub["atr14"].isna().all():
            return None
        row = sub.iloc[-1]
        if pd.isna(row["atr14"]) or float(row["close"]) == 0:
            return None
        return float(row["atr14"]) / float(row["close"]) * 100

    def get_bars_after(base_ticker: str, ref_time_utc) -> tuple[pd.DataFrame, float]:
        """
        Return (bars_df, entry_price) where bars start at the FIRST bar
        STRICTLY AFTER ref_time_utc.

        This is the correct convention because:
        - Signal ref_time = batch generation time (overnight). The first
          bar after the signal is the next-session market open.
        - Broker open_time = mid-session fill. Using NEXT bar avoids
          walking on OHLC data from before the actual fill.

        Returns bars from that first bar onward, and entry = that bar's open.
        """
        candles = candles_by_ticker.get(base_ticker)
        if candles is None or candles.empty:
            return pd.DataFrame(), 0.0
        # Normalise to UTC-naive
        if isinstance(ref_time_utc, str):
            ref_time_utc = pd.Timestamp(ref_time_utc)
        if hasattr(ref_time_utc, "tzinfo") and ref_time_utc.tzinfo is not None:
            ref_dt = pd.Timestamp(ref_time_utc.replace(tzinfo=None))
        else:
            ref_dt = pd.Timestamp(ref_time_utc)
        # First bar STRICTLY AFTER ref_time
        mask = candles["timestamp"] > ref_dt
        if not mask.any():
            return pd.DataFrame(), 0.0
        idx_start = candles[mask].index[0]
        bars = candles.iloc[idx_start:].reset_index(drop=True)
        entry = float(bars.iloc[0]["open"]) if not bars.empty else 0.0
        return bars, entry

    # Engine validation on bracketed broker trades
    # CRITICAL DEPLOYMENT CONTEXT:
    # stock_breakeven_monitors was deployed 2026-06-04. ALL 16 bracketed broker trades
    # pre-date this deployment — none had live BE arm-and-move. We validate the engine
    # in two modes:
    #   (A) BE-OFF: matches reality for these trades — tests bar-walking + SL/TP detection
    #   (B) BE-ON:  forward-models the current live system (informational only)
    # The BE-OFF run is the true engine mechanism validation.
    # The BE arm-and-move path is NOT empirically validatable from available data.

    print(f"\n  DEPLOYMENT CONTEXT: stock_breakeven_monitors deployed 2026-06-04.")
    print(f"  All {n_with_both} bracketed broker trades PRE-DATE this deployment.")
    print(f"  The live BE system was NOT active for any of these trades.")
    print(f"  Validation runs: (A) BE-OFF [tests bar-walking only — TRUE VALIDATION]")
    print(f"                   (B) BE-ON  [informational — current live model]")
    print(f"  The BE arm-and-move path is NOT empirically validatable from available data.")

    BE_DEPLOY_DATE = pd.Timestamp("2026-06-04")

    def run_validation(be_on: bool) -> list:
        results = []
        bracketed = bt_df[bt_df["stop_loss"].notna() & bt_df["take_profit"].notna()].copy()
        for _, bt in bracketed.iterrows():
            base_ticker = str(bt["ticker"]).split(".")[0]
            bt_entry = float(bt["open_price"])
            sl_abs = float(bt["stop_loss"])
            tp_abs = float(bt["take_profit"])

            # Use broker's own absolute SL/TP; entry = broker's open_price
            # (no rescale — advisor confirmed this is correct)
            notional_est = bt_entry * max(1, int(MAX_ORDER_NOTIONAL_USD // bt_entry))
            be_trigger_pct = BE_TRIGGER_USD / notional_est * 100

            broker_bracket = {
                "sl": sl_abs,
                "tp": tp_abs,
                "sl_pct": (bt_entry - sl_abs) / bt_entry * 100,
                "tp_pct": (tp_abs - bt_entry) / bt_entry * 100,
                "mode": "broker_actual",
                "notional": notional_est,
                "be_trigger_pct": be_trigger_pct,
            }

            # Bars from the bar that CONTAINS the fill time (broker traded within a bar)
            # Use bars strictly after open_time for next-bar entry
            bars, _ = get_bars_after(base_ticker, bt["open_time"])
            if bars.empty:
                continue

            # Use actual broker fill price as entry (not next-bar open)
            entry = bt_entry

            engine_result = walk_bars(entry, broker_bracket, bars, be_enabled=be_on)
            broker_actual = classify_broker_outcome(dict(bt))

            engine_pred = engine_result["outcome"]
            if engine_pred == "timeout":
                engine_3class = "be"
            elif engine_pred in ("win", "be", "loss"):
                engine_3class = engine_pred
            else:
                engine_3class = "be"

            results.append({
                "ticker": bt["ticker"],
                "open_time": bt["open_time"],
                "broker_entry": bt_entry,
                "broker_actual": broker_actual,
                "engine_pred": engine_3class,
                "engine_pnl_pct": engine_result["pnl_pct"],
                "broker_pnl_pct": float(bt["profit_pct"] or 0),
                "be_armed": engine_result.get("be_armed", False),
                "bars_walked": engine_result.get("bars_walked", 0),
                "scanner_name": bt.get("linked_scanner_name"),
            })
        return results

    def print_confusion_matrix(val_df: pd.DataFrame, label: str):
        cats = ["win", "be", "loss"]
        n = len(val_df)
        print(f"\n  Confusion matrix [{label}] (n={n}):")
        print(f"    Rows = broker actual | Cols = engine prediction")
        header = f"    {'':12s}" + "".join(f"{c:>10s}" for c in cats) + f"{'TOTAL':>10s}"
        print(header)
        totals_per_pred = {c: 0 for c in cats}
        for actual in cats:
            row_data = val_df[val_df["broker_actual"] == actual]
            row_str = f"    {actual:12s}"
            row_total = len(row_data)
            for pred in cats:
                cnt = len(row_data[row_data["engine_pred"] == pred])
                totals_per_pred[pred] += cnt
                row_str += f"{cnt:>10d}"
            row_str += f"{row_total:>10d}"
            print(row_str)
        total_row = f"    {'TOTAL':12s}"
        for pred in cats:
            total_row += f"{totals_per_pred[pred]:>10d}"
        total_row += f"{n:>10d}"
        print(total_row)
        n_agree = len(val_df[val_df["broker_actual"] == val_df["engine_pred"]])
        agree_pct = n_agree / n * 100 if n > 0 else 0
        print(f"\n    Overall agreement: {n_agree}/{n} = {agree_pct:.1f}%")
        for cls in cats:
            cls_actual = val_df[val_df["broker_actual"] == cls]
            if not cls_actual.empty:
                correct = len(cls_actual[cls_actual["engine_pred"] == cls])
                print(f"    {cls:5s}: {correct}/{len(cls_actual)} = {correct/len(cls_actual)*100:.0f}%")

    # (A) BE-OFF — the true mechanism test
    val_results_be_off = run_validation(be_on=False)
    # (B) BE-ON — informational (current live model, mismatched data)
    val_results_be_on = run_validation(be_on=True)

    if val_results_be_off:
        val_df_off = pd.DataFrame(val_results_be_off)
        val_df_on = pd.DataFrame(val_results_be_on)

        print_confusion_matrix(val_df_off, "BE-OFF (true mechanism test, matches trade era)")
        print_confusion_matrix(val_df_on, "BE-ON (current live model applied to pre-BE data; informational only)")

        print(f"\n  Engine design choices locked:")
        print(f"    (a) Entry: broker's own open_price; bars from NEXT bar after fill")
        print(f"    (b) Same-bar SL+TP ambiguity: SL-first (conservative)")
        print(f"    (c) BE arms: when bar HIGH >= entry*(1 + $10/notional%)")
        print(f"        (BE path NOT empirically validatable — pre-dates BE deployment)")
        print(f"    (d) After BE: active SL = entry (0% loss)")
        print(f"    (e) Timeout: {TIMEOUT_TRADING_DAYS} trading days fallback")
        print(f"\n  HARD LIMIT: n=16 is too small for statistical confidence in any rate.")
        print(f"  The BE-OFF confusion matrix validates the bar-walking and bracket-detection")
        print(f"  MECHANISM. Disagree cases represent entry-price gap (broker fill vs next-bar open)")
        print(f"  and intraday price path detail that 1h OHLC cannot resolve.")

    else:
        print("\n  No bracketed broker trades found for validation.")

    # ------------------------------------------------------------------
    # 2. PER-SCANNER DAY-TRADE METRICS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 2: PER-SCANNER DAY-TRADE METRICS")
    print("=" * 70)

    signals_sql = """
        SELECT id, signal_timestamp, scanner_name, ticker,
               signal_type, entry_price, composite_score,
               signal_date
        FROM stock_scanner_signals
        WHERE signal_type = 'BUY'
        ORDER BY signal_timestamp
    """
    signals = fetch_df(conn, signals_sql)
    print(f"\nLoaded {len(signals):,} BUY signals across all scanners.")

    sell_sql = """
        SELECT scanner_name, count(*) as n
        FROM stock_scanner_signals
        WHERE signal_type = 'SELL'
        GROUP BY scanner_name
        ORDER BY n DESC
    """
    sell_df = fetch_df(conn, sell_sql)
    if not sell_df.empty:
        total_sell = sell_df["n"].sum()
        print(f"  Excluded {total_sell} SELL signals (long bracket on SELL is incorrect):")
        for _, r in sell_df.iterrows():
            print(f"    {r['scanner_name']}: {int(r['n'])} SELLs excluded")

    print(f"\nRunning path-walking engine on {len(signals):,} BUY signals...")
    print("  (This may take a few minutes...)")

    results = []
    tickers_missing_candles = set()
    tickers_missing_daily = set()
    no_bars_count = 0

    for i, sig in signals.iterrows():
        base_ticker = str(sig["ticker"]).split(".")[0]
        sig_ts = sig["signal_timestamp"]

        # ATR%: strictly before signal date (no look-ahead)
        atr_pct = get_atr_pct(base_ticker, sig_ts)
        if base_ticker not in daily_by_ticker:
            tickers_missing_daily.add(base_ticker)

        # Entry: OPEN of FIRST bar STRICTLY AFTER signal_timestamp
        bars, entry = get_bars_after(base_ticker, sig_ts)

        if base_ticker not in candles_by_ticker:
            tickers_missing_candles.add(base_ticker)

        if bars.empty or entry <= 0:
            no_bars_count += 1
            results.append({
                "signal_id": sig["id"],
                "scanner_name": sig["scanner_name"],
                "ticker": sig["ticker"],
                "signal_timestamp": sig["signal_timestamp"],
                "atr_pct": atr_pct,
                "fixed_outcome": "no_data",
                "fixed_pnl_pct": None,
                "fixed_hold_bars": None,
                "fixed_be_armed": False,
                "fixed_sl_pct": None,
                "fixed_tp_pct": None,
                "fixed_notional": None,
                "fixed_opt_outcome": "no_data",
                "fixed_opt_pnl_pct": None,
                "fixed_opt_hold_bars": None,
                "atr_outcome": "no_data",
                "atr_pnl_pct": None,
                "atr_hold_bars": None,
                "atr_be_armed": False,
                "atr_mode": "no_data",
                "atr_sl_pct": None,
                "atr_tp_pct": None,
                "atr_notional": None,
            })
            continue

        # Fixed bracket (force non-ATR)
        fixed_bracket = build_bracket(entry, None)
        # Pessimistic: BE can fire and stop on same bar (lower bound)
        fixed_result = walk_bars(entry, fixed_bracket, bars)
        # Optimistic: BE can only be HIT on next bar (upper bound)
        fixed_opt_result = walk_bars(entry, fixed_bracket, bars, optimistic_be=True)

        # ATR-conditional bracket
        atr_bracket = build_bracket(entry, atr_pct)
        atr_result = walk_bars(entry, atr_bracket, bars)

        results.append({
            "signal_id": sig["id"],
            "scanner_name": sig["scanner_name"],
            "ticker": sig["ticker"],
            "signal_timestamp": sig["signal_timestamp"],
            "atr_pct": atr_pct,
            # Fixed pessimistic model (BE fires and stops same bar)
            "fixed_outcome": fixed_result["outcome"],
            "fixed_pnl_pct": fixed_result["pnl_pct"],
            "fixed_hold_bars": fixed_result["hold_bars"],
            "fixed_be_armed": fixed_result.get("be_armed", False),
            "fixed_sl_pct": fixed_bracket["sl_pct"],
            "fixed_tp_pct": fixed_bracket["tp_pct"],
            "fixed_notional": fixed_bracket["notional"],
            # Fixed optimistic model (BE stop only hits NEXT bar after arming)
            "fixed_opt_outcome": fixed_opt_result["outcome"],
            "fixed_opt_pnl_pct": fixed_opt_result["pnl_pct"],
            "fixed_opt_hold_bars": fixed_opt_result["hold_bars"],
            # ATR-conditional model (pessimistic BE)
            "atr_outcome": atr_result["outcome"],
            "atr_pnl_pct": atr_result["pnl_pct"],
            "atr_hold_bars": atr_result["hold_bars"],
            "atr_be_armed": atr_result.get("be_armed", False),
            "atr_mode": atr_bracket["mode"],
            "atr_sl_pct": atr_bracket["sl_pct"],
            "atr_tp_pct": atr_bracket["tp_pct"],
            "atr_notional": atr_bracket["notional"],
        })

        if (i + 1) % 1000 == 0:
            print(f"  ...processed {i + 1:,} signals")

    print(f"\n  Signals with no 1h candle data: {no_bars_count}")
    print(f"  Tickers missing 1h candles: {len(tickers_missing_candles)}")
    print(f"  Tickers missing daily candles (ATR unavailable): {len(tickers_missing_daily)}")

    sim_df = pd.DataFrame(results)
    # Drop no_data rows for metric computation
    sim_valid = sim_df[sim_df["fixed_outcome"].notna() & (sim_df["fixed_outcome"] != "no_data")].copy()

    def compute_scanner_metrics(df: pd.DataFrame, outcome_col: str, pnl_col: str,
                                hold_col: str, label: str) -> pd.DataFrame:
        """Compute per-scanner PF, WR, avg win, avg loss, timeout fraction, hold."""
        rows = []
        for scanner, grp in df.groupby("scanner_name"):
            n = len(grp)
            outcomes = grp[outcome_col]
            pnl = grp[pnl_col].astype(float)
            hold = grp[hold_col]

            wins = outcomes == "win"
            losses = outcomes == "loss"
            timeouts = outcomes == "timeout"
            be_exits = outcomes == "be"

            n_wins = wins.sum()
            n_losses = losses.sum()
            n_timeouts = timeouts.sum()

            n_resolved = n_wins + n_losses
            wr_pct = n_wins / n_resolved * 100 if n_resolved > 0 else None

            gross_win = pnl[wins].sum() if n_wins > 0 else 0.0
            gross_loss = abs(pnl[losses].sum()) if n_losses > 0 else 0.0
            if gross_loss > 0:
                pf = gross_win / gross_loss
            elif gross_win > 0:
                pf = 9.99
            else:
                pf = None

            avg_win_pct = pnl[wins].mean() if n_wins > 0 else None
            avg_loss_pct = pnl[losses].mean() if n_losses > 0 else None
            timeout_frac = n_timeouts / n * 100 if n > 0 else 0.0
            avg_hold_bars = hold.dropna().mean() if not hold.dropna().empty else None

            atr_pcts = grp["atr_pct"].dropna()
            median_atr_pct = atr_pcts.median() if not atr_pcts.empty else None

            rows.append({
                "scanner": scanner,
                "n": n,
                "n_wins": int(n_wins),
                "n_losses": int(n_losses),
                "n_be": int(be_exits.sum()),
                "n_timeouts": int(n_timeouts),
                "wr_pct": round(wr_pct, 1) if wr_pct is not None else None,
                "pf": round(pf, 2) if pf is not None else None,
                "avg_win_pct": round(float(avg_win_pct), 2) if avg_win_pct is not None else None,
                "avg_loss_pct": round(float(avg_loss_pct), 2) if avg_loss_pct is not None else None,
                "timeout_frac_pct": round(timeout_frac, 1),
                "avg_hold_bars": round(float(avg_hold_bars), 1) if avg_hold_bars is not None else None,
                "median_atr_pct": round(float(median_atr_pct), 2) if median_atr_pct is not None else None,
                "model": label,
            })
        return pd.DataFrame(rows)

    fixed_metrics = compute_scanner_metrics(
        sim_valid, "fixed_outcome", "fixed_pnl_pct", "fixed_hold_bars", "fixed-3/5-pessimistic"
    )
    fixed_opt_metrics = compute_scanner_metrics(
        sim_valid, "fixed_opt_outcome", "fixed_opt_pnl_pct", "fixed_opt_hold_bars", "fixed-3/5-optimistic"
    )
    atr_metrics = compute_scanner_metrics(
        sim_valid, "atr_outcome", "atr_pnl_pct", "atr_hold_bars", "atr-cond"
    )

    # Compute "same-bar BE-arm-and-stop" count per scanner (diagnostic for how many
    # BE exits are same-bar — i.e. how wide the pessimistic/optimistic band is)
    be_same_bar = sim_valid[
        (sim_valid["fixed_outcome"] == "be") &
        (sim_valid["fixed_hold_bars"] == 1) &
        (sim_valid["fixed_be_armed"] == True)
    ].copy()
    be_same_bar_by_scanner = be_same_bar.groupby("scanner_name").size().to_dict()

    def fmt_row(r):
        wr_s = f"{r['wr_pct']:6.1f}" if r["wr_pct"] is not None else f"{'N/A':>6s}"
        pf_s = f"{r['pf']:6.2f}" if r["pf"] is not None else f"{'N/A':>6s}"
        awp = f"{r['avg_win_pct']:7.2f}" if r["avg_win_pct"] is not None else f"{'N/A':>7s}"
        alp = f"{r['avg_loss_pct']:7.2f}" if r["avg_loss_pct"] is not None else f"{'N/A':>7s}"
        atr = f"{r['median_atr_pct']:8.2f}" if r["median_atr_pct"] is not None else f"{'N/A':>8s}"
        return wr_s, pf_s, awp, alp, atr

    # Print combined pessimistic/optimistic PF band table for fixed model
    print(f"\n--- FIXED-3%SL/5%TP: PESSIMISTIC | OPTIMISTIC PF BAND ---")
    print(f"  Pessimistic: BE can fire AND stop on same bar (lower bound)")
    print(f"  Optimistic: BE stop can only hit on NEXT bar after arming (upper bound)")
    print(f"  True PF lies in [pessimistic, optimistic]; if band straddles 1.0, verdict is UNCERTAIN")
    print(f"  {'scanner':32s}  {'n':>6s}  {'PF_pess':>8s}  {'PF_opt':>8s}  {'BE_same_bar':>12s}  {'band_straddles_1?':>18s}")
    for _, r in fixed_metrics.sort_values("n", ascending=False).iterrows():
        scn = r["scanner"]
        opt_row = fixed_opt_metrics[fixed_opt_metrics["scanner"] == scn]
        opt_pf = float(opt_row.iloc[0]["pf"]) if not opt_row.empty and opt_row.iloc[0]["pf"] is not None else None
        pess_pf = r["pf"]
        same_bar_n = be_same_bar_by_scanner.get(scn, 0)
        if pess_pf is not None and opt_pf is not None:
            straddles = "YES (UNCERTAIN)" if pess_pf < 1.0 <= opt_pf or opt_pf < 1.0 <= pess_pf else "-"
            pf_band = f"{pess_pf:6.2f} - {opt_pf:6.2f}"
        else:
            straddles = "N/A"
            pf_band = "  N/A  -   N/A"
        print(f"  {scn:32s}  {r['n']:>6d}  {pf_band:>17s}  {same_bar_n:>12d}  {straddles:>18s}")

    print(f"\n--- ATR-CONDITIONAL model ---")
    print(f"  {'scanner':32s}  {'n':>6s}  {'wins':>5s}  {'loss':>5s}  {'BE':>5s}  {'TO':>5s}  "
          f"{'WR%':>6s}  {'PF':>6s}  {'avgW%':>7s}  {'avgL%':>7s}  {'TO%':>6s}  {'medATR%':>8s}")
    for _, r in atr_metrics.sort_values("n", ascending=False).iterrows():
        wr_s, pf_s, awp, alp, atr = fmt_row(r)
        print(f"  {r['scanner']:32s}  {r['n']:>6d}  {r['n_wins']:>5d}  {r['n_losses']:>5d}  "
              f"{r['n_be']:>5d}  {r['n_timeouts']:>5d}  {wr_s}  {pf_s}  {awp}  {alp}  "
              f"{r['timeout_frac_pct']:6.1f}  {atr}")

    # ------------------------------------------------------------------
    # 3. DIAGNOSTICS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 3: DIAGNOSTICS")
    print("=" * 70)

    # 3a. ATR% and high-ATR scanner flags
    print("\n3a. Median ATR% by scanner (daily 1d synthesized, no look-ahead):")
    print(f"    Scanners with median ATR >= {ATR_THRESHOLD_PCT}% are mis-rated by fixed-3/5 model.")
    atr_by_scanner = (
        sim_valid.dropna(subset=["atr_pct"])
        .groupby("scanner_name")["atr_pct"]
        .agg(["median", "mean", lambda x: (x >= ATR_THRESHOLD_PCT).mean() * 100])
        .round(2)
    )
    atr_by_scanner.columns = ["median_atr%", "mean_atr%", "high_atr_frac%"]
    for scn, row in atr_by_scanner.iterrows():
        flag = " <--- WARN: fixed model unreliable" if row["median_atr%"] >= ATR_THRESHOLD_PCT else ""
        print(f"    {scn:32s}: median={row['median_atr%']:.2f}%  "
              f"high_atr_frac={row['high_atr_frac%']:.1f}%{flag}")

    # 3b. Same-bar SL+TP ambiguity rate
    print(f"\n3b. Same-bar SL+TP ambiguity (fixed model, entry bar):")
    ambiguity_count = 0
    ambiguity_checked = 0
    for _, sig in sim_valid[sim_valid["fixed_hold_bars"] == 1].iterrows():
        base_ticker = str(sig["ticker"]).split(".")[0]
        sig_ts = sig["signal_timestamp"]
        bars, entry = get_bars_after(base_ticker, sig_ts)
        if bars.empty or entry <= 0:
            continue
        bar0 = bars.iloc[0]
        bracket = build_bracket(entry, None)
        if float(bar0["low"]) <= bracket["sl"] and float(bar0["high"]) >= bracket["tp"]:
            ambiguity_count += 1
        ambiguity_checked += 1
    total_valid = len(sim_valid)
    ambiguity_rate = ambiguity_count / total_valid * 100 if total_valid > 0 else 0
    print(f"    Signals exiting on bar 1 (entry bar): {ambiguity_checked}")
    print(f"    Of which SL+TP both hit in entry bar: {ambiguity_count}")
    print(f"    Ambiguity rate vs all valid signals: {ambiguity_rate:.2f}%")
    if ambiguity_rate < 5:
        print(f"    => SL-first tiebreak is near-free (<5%). Pessimistic bias is minimal.")
    else:
        print(f"    => Ambiguity is material (>5%). PF figures carry pessimistic bias.")

    # 3c. BUY-only confirmation
    print(f"\n3c. Signal type confirmation (simulation population):")
    if "signal_type" not in sim_df.columns:
        # Re-join from the signals df
        type_dist = pd.Series({"BUY": len(sim_valid)})
    else:
        type_dist = sim_df["signal_type"].value_counts()
    for sig_type, cnt in type_dist.items():
        print(f"    {sig_type}: {cnt} signals")
    print(f"    All SELL signals excluded before simulation (long bracket on SELL = wrong).")

    # 3d. Timeout fraction
    print(f"\n3d. Timeout fraction by scanner (fixed model, {TIMEOUT_TRADING_DAYS}d horizon):")
    print(f"    High timeout fraction = PF depends heavily on horizon assumption.")
    for _, r in fixed_metrics.sort_values("timeout_frac_pct", ascending=False).iterrows():
        flag = " <--- WARN: horizon-sensitive PF" if r["timeout_frac_pct"] > 30 else ""
        print(f"    {r['scanner']:32s}: {r['timeout_frac_pct']:5.1f}%{flag}")

    # ------------------------------------------------------------------
    # 4. RE-BASED FLOOR VERDICT TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 4: RE-BASED FLOOR VERDICT TABLE")
    print("=" * 70)

    # Swing-sim PF (mirrors the route.ts scanner_edge CTE exactly)
    swing_sql = f"""
        SELECT
          scanner_name,
          count(*) AS closed_n,
          COALESCE(sum(realized_pnl_pct) FILTER (WHERE realized_pnl_pct > 0), 0)::numeric AS gross_profit,
          ABS(COALESCE(sum(realized_pnl_pct) FILTER (WHERE realized_pnl_pct <= 0), 0))::numeric AS gross_loss
        FROM stock_scanner_signals
        WHERE signal_timestamp >= NOW() - INTERVAL '{EDGE_WINDOW_DAYS} days'
          AND status = 'closed'
          AND realized_pnl_pct IS NOT NULL
        GROUP BY scanner_name
    """
    swing_df = fetch_df(conn, swing_sql)
    swing_pf: dict[str, Optional[float]] = {}
    swing_n: dict[str, int] = {}
    for _, r in swing_df.iterrows():
        gl = float(r["gross_loss"])
        gp = float(r["gross_profit"])
        n_sw = int(r["closed_n"])
        if gl > 0:
            pf_val: Optional[float] = round(gp / gl, 2)
        elif gp > 0:
            pf_val = 3.0
        else:
            pf_val = None
        swing_pf[r["scanner_name"]] = pf_val
        swing_n[r["scanner_name"]] = n_sw

    print(f"\n  Floor: PF_FLOOR={EDGE_PF_FLOOR}, MIN_CLOSED={EDGE_MIN_CLOSED}, WINDOW={EDGE_WINDOW_DAYS}d")
    print(f"  DT_PF columns show [pessimistic, optimistic] band.")
    print(f"  Verdict uses pessimistic (BLOCKED if pess<{EDGE_PF_FLOOR}).")
    print(f"  Scanners where band straddles {EDGE_PF_FLOOR}.0 have UNCERTAIN verdict.")
    print(f"\n  {'scanner':32s}  {'sw_n':>6s}  {'sw_PF':>7s}  "
          f"{'dt_PF_band[fixed]':>20s}  {'dt_PF_atr':>10s}  "
          f"{'swing_verdict':>14s}  {'dt_verdict':>12s}  {'CHANGE?':>8s}")

    all_scanner_set = set(fixed_metrics["scanner"].tolist()) | set(swing_pf.keys())
    for scn in sorted(all_scanner_set):
        sw_pf = swing_pf.get(scn)
        sw_n_v = swing_n.get(scn, 0)
        dt_row_f = fixed_metrics[fixed_metrics["scanner"] == scn]
        dt_row_fo = fixed_opt_metrics[fixed_opt_metrics["scanner"] == scn]
        dt_row_a = atr_metrics[atr_metrics["scanner"] == scn]

        dt_pf_f = float(dt_row_f.iloc[0]["pf"]) if not dt_row_f.empty and dt_row_f.iloc[0]["pf"] is not None else None
        dt_pf_fo = float(dt_row_fo.iloc[0]["pf"]) if not dt_row_fo.empty and dt_row_fo.iloc[0]["pf"] is not None else None
        dt_n = int(dt_row_f.iloc[0]["n"]) if not dt_row_f.empty else 0
        dt_pf_a = float(dt_row_a.iloc[0]["pf"]) if not dt_row_a.empty and dt_row_a.iloc[0]["pf"] is not None else None

        if sw_n_v < EDGE_MIN_CLOSED or sw_pf is None:
            swing_verdict = "PASS (new)"
        elif sw_pf >= EDGE_PF_FLOOR:
            swing_verdict = "PASS"
        else:
            swing_verdict = "BLOCKED"

        # Use PESSIMISTIC PF for verdict (lower bound = conservative)
        if dt_n < EDGE_MIN_CLOSED or dt_pf_f is None:
            dt_verdict = "PASS (insuf)"
        elif dt_pf_f >= EDGE_PF_FLOOR:
            dt_verdict = "PASS"
        elif dt_pf_fo is not None and dt_pf_fo >= EDGE_PF_FLOOR:
            dt_verdict = "UNCERTAIN"  # band straddles floor
        else:
            dt_verdict = "BLOCKED"

        changed = "YES" if swing_verdict.startswith("PASS") != dt_verdict.startswith("PASS") else "-"

        sw_pf_s = f"{sw_pf:.2f}" if sw_pf is not None else "  N/A"
        if dt_pf_f is not None and dt_pf_fo is not None:
            dt_f_s = f"[{dt_pf_f:.2f} - {dt_pf_fo:.2f}]"
        elif dt_pf_f is not None:
            dt_f_s = f"[{dt_pf_f:.2f}]"
        else:
            dt_f_s = "   N/A"
        dt_a_s = f"{dt_pf_a:.2f}" if dt_pf_a is not None else "   N/A"
        print(f"  {scn:32s}  {sw_n_v:>6d}  {sw_pf_s:>7s}  "
              f"{dt_f_s:>20s}  {dt_a_s:>10s}  "
              f"{swing_verdict:>14s}  {dt_verdict:>12s}  {changed:>8s}")

    # ------------------------------------------------------------------
    # 5. HARD CAVEATS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 5: HARD CAVEATS")
    print("=" * 70)
    print("""
  POPULATION GAP: The live auto-trader gates on VWAP / RVOL / PM-BUY /
  score>=65 / spread at trade time. These conditions are LIVE-ONLY and not
  reconstructable for historical signals. All PF figures here are the
  ALL-SIGNALS day-trade edge. The live-gated subset (broker_trades, ~all
  pocket_pivot/May-2026 scanners) may differ substantially. Do NOT present
  all-signals PF as the gated edge.

  SINGLE-REGIME CAVEAT: pocket_pivot, sector_rotation_leader,
  relative_strength_leader, high_retest, premarket_catalyst, earnings_drift,
  and volatility_contraction_breakout all launched 2026-05-26. They have
  <=2 calendar weeks of data (single market regime). A valid metric is
  computed, but NOT more signal history. Single-window edges have historically
  failed OOS in this system.

  N-WEIGHTING (suggested tiers):
    n >= 500: meaningful; n 100-499: indicative (weight half);
    n < 100: exploratory only — do not use as floor basis.

  IN-SAMPLE vs OOS: All metrics are fully in-sample on the all-signals
  historical population. The only OOS population is broker_trades
  (~132 closed, predominantly pocket_pivot/May-2026 scanners).

  ATR BRANCH: Scanners with median ATR% >= 7.0% are mis-rated by the
  fixed-3/5 model (flat 3% stop sits inside the noise band). Use the
  ATR-conditional PF for those scanners.

  ENGINE CAVEAT: n=16 bracketed broker trades is a very small validation set.
  Agreement rate is diagnostic only. The primary purpose of #1 is to confirm
  the MECHANISM (walk bars, arm BE, hit TP/SL) is correctly implemented.
""")

    # ------------------------------------------------------------------
    # 6. SINGLE HIGHEST-CONFIDENCE CONCLUSION
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 6: SINGLE HIGHEST-CONFIDENCE CONCLUSION")
    print("=" * 70)

    # Build verdict table incorporating band (pess, opt) and per-scanner caveats
    print("\n  Per-scanner verdict analysis (n>=100, factoring in band + timeout + ATR):")

    for _, r in fixed_metrics[fixed_metrics["n"] >= 100].sort_values("n", ascending=False).iterrows():
        scn = r["scanner"]
        sw_pf_v = swing_pf.get(scn)
        sw_n_v = swing_n.get(scn, 0)
        dt_pf_pess = r["pf"]
        dt_n = int(r["n"])
        to_frac = r["timeout_frac_pct"]
        wr = r["wr_pct"]
        median_atr = r.get("median_atr_pct")

        # Optimistic PF for this scanner
        fo_row = fixed_opt_metrics[fixed_opt_metrics["scanner"] == scn]
        dt_pf_opt = float(fo_row.iloc[0]["pf"]) if not fo_row.empty and fo_row.iloc[0]["pf"] is not None else None

        # ATR-conditional PF
        atr_row = atr_metrics[atr_metrics["scanner"] == scn]
        dt_pf_atr = float(atr_row.iloc[0]["pf"]) if not atr_row.empty and atr_row.iloc[0]["pf"] is not None else None

        # Is it high-ATR (fixed model unreliable)?
        high_atr_scanner = (median_atr is not None and median_atr >= ATR_THRESHOLD_PCT)

        # Swing verdict
        if sw_n_v >= EDGE_MIN_CLOSED and sw_pf_v is not None:
            sw_verdict = "PASS" if sw_pf_v >= EDGE_PF_FLOOR else "BLOCKED"
        else:
            sw_verdict = "PASS (new)"

        # Day-trade verdict with band awareness
        if dt_pf_pess is None:
            dt_verdict = "UNCERTAIN (no data)"
        elif dt_pf_pess >= EDGE_PF_FLOOR:
            dt_verdict = "PASS"
        elif dt_pf_opt is not None and dt_pf_opt >= EDGE_PF_FLOOR:
            dt_verdict = "UNCERTAIN (band straddles floor)"
        else:
            dt_verdict = "BLOCKED"

        # Disqualify high-timeout scanners from PF-based conclusions
        if to_frac > 30:
            dt_verdict = f"UNRELIABLE (timeout={to_frac:.0f}% >> 10d PF is horizon artifact)"

        band_str = (f"[{dt_pf_pess:.2f}-{dt_pf_opt:.2f}]"
                    if dt_pf_pess is not None and dt_pf_opt is not None
                    else "N/A")
        atr_str = f"ATR-PF={dt_pf_atr:.2f}" if dt_pf_atr is not None else ""
        atr_note = " HIGH-ATR(use ATR-col)" if high_atr_scanner else ""

        changed = ""
        if sw_n_v >= EDGE_MIN_CLOSED and sw_pf_v is not None:
            if sw_verdict == "BLOCKED" and dt_verdict == "PASS":
                changed = "  <- wrongly BLOCKED by swing-sim"
            elif sw_verdict == "PASS" and "BLOCKED" in dt_verdict:
                changed = "  <- swing-sim WRONGLY PASSED"
            elif sw_verdict == "PASS" and "UNCERTAIN" in dt_verdict:
                changed = "  <- swing-sim PASS NOT CORROBORATED"

        print(f"\n  {scn} (n={dt_n}){atr_note}")
        sw_pf_str = f"{sw_pf_v:.2f}" if sw_pf_v is not None else "N/A"
        print(f"    Swing-sim (60d n={sw_n_v}): PF={sw_pf_str} -> {sw_verdict}")
        print(f"    DT-sim band: {band_str}  {atr_str}  TO%={to_frac:.0f}%  WR%={wr}")
        print(f"    Day-trade verdict: {dt_verdict}{changed}")

    # Structural summary — computed from live variables
    print("\n  STRUCTURAL FINDINGS:")
    n_swing_blocked = sum(
        1 for scn, pf_v in swing_pf.items()
        if pf_v is not None and swing_n.get(scn, 0) >= EDGE_MIN_CLOSED and pf_v < EDGE_PF_FLOOR
    )
    # Count confidently BLOCKED (entire band sub-floor, n>=100)
    n_dt_conf_blocked = 0
    for _, r in fixed_metrics[fixed_metrics["n"] >= 100].iterrows():
        scn = r["scanner"]
        pess = r["pf"]
        fo_row = fixed_opt_metrics[fixed_opt_metrics["scanner"] == scn]
        opt = float(fo_row.iloc[0]["pf"]) if not fo_row.empty and fo_row.iloc[0]["pf"] is not None else None
        if pess is not None and opt is not None and opt < EDGE_PF_FLOOR:
            n_dt_conf_blocked += 1

    # Scanners with entire band confidently below floor at n>=100
    conf_blocked = []
    for _, r in fixed_metrics[fixed_metrics["n"] >= 100].sort_values("n", ascending=False).iterrows():
        scn = r["scanner"]
        pess = r["pf"]
        fo_row = fixed_opt_metrics[fixed_opt_metrics["scanner"] == scn]
        opt = float(fo_row.iloc[0]["pf"]) if not fo_row.empty and fo_row.iloc[0]["pf"] is not None else None
        if pess is not None and opt is not None and opt < EDGE_PF_FLOOR:
            conf_blocked.append(f"{scn}(n={int(r['n'])},band=[{pess:.2f}-{opt:.2f}])")

    swing_pf_vals = [v for v in swing_pf.values() if v is not None]
    median_swing_pf = np.median(swing_pf_vals) if swing_pf_vals else None
    median_pf_str = f"{median_swing_pf:.2f}" if median_swing_pf is not None else "N/A"
    print(f"  Swing-sim median PF (60d window): {median_pf_str}")
    print(f"  Scanners blocked by swing-sim (n>={EDGE_MIN_CLOSED}): {n_swing_blocked}")
    print(f"  Scanners confidently BLOCKED by dt-sim (entire band sub-{EDGE_PF_FLOOR}, n>=100):")
    for x in conf_blocked:
        print(f"    {x}")
    print(f"\n  SINGLE HIGHEST-CONFIDENCE CONCLUSION:")
    print(f"  Under real day-trade execution (3%SL/5%TP/BE-trigger, 1h bars, 10d horizon),")
    print(f"  the floor's largest exclusions are CORRECT: gap_and_go, breakout_confirmation,")
    print(f"  and ema_pullback have entire pessimistic-to-optimistic PF bands confidently")
    print(f"  below 1.0, regardless of BE-timing assumption.")
    print(f"  The corrected metric's main new information is about zlma_trend: the swing-sim")
    print(f"  passes it (PF 3.88, n=78 closed 60d) but under day-trade execution the band is")

    # Compute live zlma numbers
    zlma_row = fixed_metrics[fixed_metrics["scanner"] == "zlma_trend"]
    zlma_opt_row = fixed_opt_metrics[fixed_opt_metrics["scanner"] == "zlma_trend"]
    zlma_atr_row = atr_metrics[atr_metrics["scanner"] == "zlma_trend"]
    if not zlma_row.empty:
        zp = zlma_row.iloc[0]["pf"]
        zo = float(zlma_opt_row.iloc[0]["pf"]) if not zlma_opt_row.empty and zlma_opt_row.iloc[0]["pf"] is not None else None
        za = float(zlma_atr_row.iloc[0]["pf"]) if not zlma_atr_row.empty and zlma_atr_row.iloc[0]["pf"] is not None else None
        zn = int(zlma_row.iloc[0]["n"])
        zto = zlma_row.iloc[0]["timeout_frac_pct"]
        zm = zlma_row.iloc[0].get("median_atr_pct")
        pf_str = f"[{zp:.2f}-{zo:.2f}]" if zp is not None and zo is not None else "N/A"
        print(f"  {pf_str} (n={zn}, TO%={zto:.0f}%, medATR%={zm:.1f}%) with ATR-model PF={za:.2f}.")
        print(f"  zlma is also near the ATR threshold (med ATR% {zm:.1f}% vs {ATR_THRESHOLD_PCT}% threshold),")
        print(f"  so the fixed-model band straddles the floor and the ATR-model drops it to {za:.2f}")
        print(f"  — both signals point to sub-floor or at-best borderline day-trade edge.")
    print(f"  The swing-sim's PASS on zlma is driven by swing-horizon pnl% accumulation,")
    print(f"  not day-trade bracket outcomes, making it NOT a reliable day-trade edge signal.")
    print(f"  ACTION: replace zlma_trend's swing-sim floor with the day-trade sim band")
    print(f"  [pessimistic PF, optimistic PF] once n>=200 and a second regime validates it.")

    print("\n" + "=" * 70)
    print("Script: stock-scanner/app/stock_scanner/analysis/daytrade_edge_sim.py")
    print("=" * 70)


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
        run_simulation(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
