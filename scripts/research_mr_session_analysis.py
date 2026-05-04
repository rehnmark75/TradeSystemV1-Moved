#!/usr/bin/env python3
"""
Mean Reversion session breakdown research — all FX pairs.

Resamples 1m ig_candles → 5m, then simulates BB(20,2)+RSI(14) signals
across four session windows for both touch and rejection entry modes.
Also tests the low-vol regime filter (ATR + EMA slope) from the USDCHF research.

Output: per-pair, per-session, per-entry-mode: n, WR, PF, avg_pips
Cost assumption: 0.9 pip spread per trade (same as USDCHF research).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

DB_URL = os.getenv(
    "FOREX_DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/forex",
)

PAIRS = [
    ("CS.D.EURUSD.CEEM.IP",  "EURUSD",  0.0001),
    ("CS.D.GBPUSD.MINI.IP",  "GBPUSD",  0.0001),
    ("CS.D.NZDUSD.MINI.IP",  "NZDUSD",  0.0001),
    ("CS.D.AUDUSD.MINI.IP",  "AUDUSD",  0.0001),
    ("CS.D.USDCAD.MINI.IP",  "USDCAD",  0.0001),
    ("CS.D.USDJPY.MINI.IP",  "USDJPY",  0.01),
    ("CS.D.EURJPY.MINI.IP",  "EURJPY",  0.01),
    ("CS.D.AUDJPY.MINI.IP",  "AUDJPY",  0.01),
]

SESSIONS = {
    "Asian    (00-06)": (0, 6),
    "London   (07-11)": (7, 11),
    "NY       (12-17)": (12, 17),
    "Late-US  (18-22)": (18, 22),
}

# Trade parameters
SL_PIPS = 10.0
TP_PIPS = 7.0
COST_PIPS = 0.9
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OS = 31      # same as USDCHF research
RSI_OB = 69
COOLDOWN_BARS = 12  # 1-hour cooldown on 5m bars

# Low-vol regime filter thresholds (5m scale)
ATR_MAX_PIPS = 4.0   # research used 3.0; slightly relaxed
EMA_PERIOD = 50
EMA_LOOKBACK = 24    # candles
EMA_MAX_CHANGE_PIPS = 4.5


def load_1m(epic: str) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(
        """
        SELECT start_time, open, high, low, close
          FROM ig_candles
         WHERE epic = %s AND timeframe = 1
         ORDER BY start_time
        """,
        (epic,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    df = pd.DataFrame(rows, columns=["start_time", "open", "high", "low", "close"])
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df = df.set_index("start_time")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df


def resample_5m(df1m: pd.DataFrame) -> pd.DataFrame:
    df5 = df1m.resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()
    return df5


def compute_indicators(df: pd.DataFrame, pip: float) -> pd.DataFrame:
    close = df["close"]
    # BB
    ma = close.rolling(BB_PERIOD).mean()
    sd = close.rolling(BB_PERIOD).std()
    df["bb_upper"] = ma + BB_STD * sd
    df["bb_lower"] = ma - BB_STD * sd
    # RSI (Wilder EWM)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    # ATR14
    prev_close = close.shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pips"] = df["atr"] / pip
    # EMA50
    df["ema50"] = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    df["ema50_pips"] = df["ema50"] / pip
    return df


def simulate(df: pd.DataFrame, pip: float, entry_mode: str, session: tuple,
             use_low_vol_filter: bool = False) -> list:
    """Return list of trade pips (after cost). Positive = win."""
    s_start, s_end = session
    results = []
    last_signal_bar = -COOLDOWN_BARS - 1

    for i in range(max(BB_PERIOD, RSI_PERIOD, EMA_PERIOD + EMA_LOOKBACK) + 2, len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        utc_hour = ts.hour

        if not (s_start <= utc_hour <= s_end):
            continue
        if i - last_signal_bar < COOLDOWN_BARS:
            continue

        if pd.isna(row["bb_upper"]) or pd.isna(row["rsi"]):
            continue

        # Low-vol regime filter
        if use_low_vol_filter:
            atr_pips = row["atr_pips"]
            if pd.isna(atr_pips) or atr_pips > ATR_MAX_PIPS:
                continue
            ema_now = row["ema50_pips"]
            ema_prev = df["ema50_pips"].iloc[i - EMA_LOOKBACK]
            if pd.isna(ema_prev) or abs(ema_now - ema_prev) >= EMA_MAX_CHANGE_PIPS:
                continue

        direction = None

        if entry_mode == "touch":
            if row["close"] <= row["bb_lower"] and row["rsi"] <= RSI_OS:
                direction = "BUY"
            elif row["close"] >= row["bb_upper"] and row["rsi"] >= RSI_OB:
                direction = "SELL"
        else:  # rejection
            if i < 1:
                continue
            prev = df.iloc[i - 1]
            if pd.isna(prev["bb_lower"]) or pd.isna(prev["rsi"]):
                continue
            if (prev["close"] <= prev["bb_lower"] and prev["rsi"] <= RSI_OS
                    and row["close"] > row["bb_lower"]):
                direction = "BUY"
            elif (prev["close"] >= prev["bb_upper"] and prev["rsi"] >= RSI_OB
                    and row["close"] < row["bb_upper"]):
                direction = "SELL"

        if direction is None:
            continue

        last_signal_bar = i
        entry = row["close"]

        # Walk forward to determine outcome
        hit = None
        for j in range(i + 1, min(i + 60, len(df))):  # max 5h hold
            future = df.iloc[j]
            if direction == "BUY":
                if future["high"] >= entry + TP_PIPS * pip:
                    hit = "win"
                    break
                if future["low"] <= entry - SL_PIPS * pip:
                    hit = "loss"
                    break
            else:
                if future["low"] <= entry - TP_PIPS * pip:
                    hit = "win"
                    break
                if future["high"] >= entry + SL_PIPS * pip:
                    hit = "loss"
                    break

        if hit is None:
            hit = "loss"  # time-stop counts as loss at mid

        gross = TP_PIPS if hit == "win" else -SL_PIPS
        net = gross - COST_PIPS
        results.append(net)

    return results


def pf_wr(results: list):
    if not results:
        return 0, 0, 0
    wins = [r for r in results if r > 0]
    losses = [r for r in results if r <= 0]
    wr = len(wins) / len(results) * 100
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss
    return len(results), round(wr, 1), round(pf, 2)


def main():
    print(f"\n{'='*110}")
    print(f"{'MEAN REVERSION SESSION ANALYSIS — ALL FX PAIRS':^110}")
    print(f"BB({BB_PERIOD},{BB_STD}) + RSI({RSI_PERIOD}) | RSI OS/OB: {RSI_OS}/{RSI_OB} | SL/TP: {SL_PIPS}/{TP_PIPS} | Cost: {COST_PIPS} pip | ~7.5mo 5m data")
    print(f"{'='*110}")

    all_results = {}

    for epic, name, pip in PAIRS:
        print(f"\n{'─'*110}")
        print(f"  {name}  (pip={pip})")
        print(f"{'─'*110}")
        print(f"  {'Session':<20} {'Touch':>28} | {'Rejection':>28} | {'Touch+LowVol':>28}")
        print(f"  {'':20} {'n':>5} {'WR%':>7} {'PF':>7} {'avg':>7} | {'n':>5} {'WR%':>7} {'PF':>7} {'avg':>7} | {'n':>5} {'WR%':>7} {'PF':>7} {'avg':>7}")
        print(f"  {'-'*108}")

        df1m = load_1m(epic)
        if df1m.empty:
            print(f"  [no data]")
            continue

        df5 = resample_5m(df1m)
        df5 = compute_indicators(df5, pip)

        pair_data = {}
        for sess_name, sess_window in SESSIONS.items():
            t = simulate(df5, pip, "touch",     sess_window)
            r = simulate(df5, pip, "rejection", sess_window)
            tv = simulate(df5, pip, "touch",    sess_window, use_low_vol_filter=True)

            tn, twr, tpf = pf_wr(t)
            rn, rwr, rpf = pf_wr(r)
            tvn, tvwr, tvpf = pf_wr(tv)

            tavg = round(sum(t) / len(t), 2) if t else 0
            ravg = round(sum(r) / len(r), 2) if r else 0
            tvavg = round(sum(tv) / len(tv), 2) if tv else 0

            pair_data[sess_name] = {
                "touch": (tn, twr, tpf, tavg),
                "rejection": (rn, rwr, rpf, ravg),
                "touch_lv": (tvn, tvwr, tvpf, tvavg),
            }

            def flag(pf, n):
                if n < 10: return " "
                if pf >= 2.0: return "★★"
                if pf >= 1.5: return "★ "
                if pf >= 1.2: return "· "
                return "  "

            print(
                f"  {sess_name:<20} "
                f"{tn:>5} {twr:>6.1f}% {tpf:>6.2f} {tavg:>+7.2f}{flag(tpf,tn)} | "
                f"{rn:>5} {rwr:>6.1f}% {rpf:>6.2f} {ravg:>+7.2f}{flag(rpf,rn)} | "
                f"{tvn:>5} {tvwr:>6.1f}% {tvpf:>6.2f} {tvavg:>+7.2f}{flag(tvpf,tvn)}"
            )

        all_results[name] = pair_data

    # Summary: best config per pair
    print(f"\n\n{'='*110}")
    print(f"{'BEST CONFIGURATION PER PAIR (n≥20, PF≥1.5)':^110}")
    print(f"{'='*110}")
    print(f"  {'Pair':<8} {'Session':<20} {'Mode':<14} {'n':>5} {'WR%':>7} {'PF':>6} {'avg':>7}")
    print(f"  {'-'*75}")

    for name, sessions in all_results.items():
        best = None
        for sess_name, modes in sessions.items():
            for mode_key, (n, wr, pf, avg) in modes.items():
                if n >= 20 and pf >= 1.5:
                    if best is None or pf > best[5]:
                        mode_label = {"touch": "touch", "rejection": "rejection", "touch_lv": "touch+low-vol"}[mode_key]
                        best = (name, sess_name.strip(), mode_label, n, wr, pf, avg)
        if best:
            print(f"  {best[0]:<8} {best[1]:<20} {best[2]:<14} {best[3]:>5} {best[4]:>6.1f}% {best[5]:>6.2f} {best[6]:>+7.2f}")
        else:
            print(f"  {name:<8} — no config with n≥20 and PF≥1.5")

    print(f"\n★★ = PF≥2.0   ★ = PF≥1.5   · = PF≥1.2")
    print(f"Cost: {COST_PIPS} pip/trade. SL={SL_PIPS} TP={TP_PIPS}. Cooldown={COOLDOWN_BARS} bars (1h).")


if __name__ == "__main__":
    main()
