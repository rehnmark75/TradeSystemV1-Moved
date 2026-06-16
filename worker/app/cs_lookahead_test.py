#!/usr/bin/env python3
"""Isolate the cross-timeframe lookahead in eurjpy_entry_model_combo_test.

DONCHIAN signal fires on the 1h CLOSE but is timestamped at the 1h bar
start_time, so simulate() enters at the 5m bar at the START of the hour.
That is ~55 min of future info. Re-time the signal to the bar CLOSE
(start + timeframe) and re-run the SAME simulate() to see the edge collapse.
"""
import pandas as pd
import numpy as np
import eurjpy_entry_model_combo_test as M

PIP = M.PIP
OOS_START = M.OOS_START
WINDOW_END = pd.Timestamp(M.WINDOW_END, tz="UTC")
IS_END = M.IS_END


def pf(trades, start, end):
    m = M.metrics(trades, start, end)
    return m["n"], m["pf"], m["wr"], m["pips"]


def run_model(name, df5, signals, sl, tp, hold, tf):
    ss = M.SignalSet(name, "breakout", tf, signals, sl, tp, hold)
    return M.simulate(df5, ss)


def main():
    df5 = M.load_candles(5)
    df1h = M.load_candles(60)

    c1 = df1h["close"]
    hi20 = df1h["high"].rolling(20).max().shift(1)
    lo20 = df1h["low"].rolling(20).min().shift(1)
    don_long = (c1 > hi20).to_numpy()
    don_short = (c1 < lo20).to_numpy()

    # (A) ORIGINAL: timestamp at 1h start_time (what the research script does)
    sig_orig = M.pack(df1h.index, don_long, don_short)

    # (B) CORRECTED: timestamp at 1h CLOSE = start + 1h, so entry is the first
    # 5m bar AFTER the breakout candle has actually closed.
    sig_fixed = sig_orig.copy()
    sig_fixed["time"] = pd.to_datetime(sig_fixed["time"], utc=True) + pd.Timedelta(hours=1)

    for label, sig in (("ORIGINAL (entry at hour-START, lookahead)", sig_orig),
                       ("CORRECTED (entry at hour-CLOSE, no lookahead)", sig_fixed)):
        tr = run_model("DON", df5, sig, 36, 72, 576, 60)
        n_is, pf_is, *_ = pf(tr, pd.Timestamp(M.WINDOW_START, tz="UTC"), IS_END)
        n_oos, pf_oos, wr_oos, pips_oos = pf(tr, OOS_START, WINDOW_END)
        avg = pips_oos / n_oos if n_oos else 0
        print(f"\nDONCHIAN_symmetric — {label}")
        print(f"   IS:  n={n_is:4d}  PF={pf_is:.2f}")
        print(f"   OOS: n={n_oos:4d}  PF={pf_oos:.2f}  WR={wr_oos:.1f}%  pips={pips_oos:+.0f}  avg={avg:+.2f}/trade")


if __name__ == "__main__":
    main()
