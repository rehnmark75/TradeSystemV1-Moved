#!/usr/bin/env python3
"""EURJPY outside-box scout.

Non-Donchian research pass focused on conditional behavior rather than chart
patterns: time-of-week, session, recent return state, volatility percentile,
EMA slope, RSI zones, and asymmetric fixed exits.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import psycopg2


DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
EPIC = "CS.D.EURJPY.MINI.IP"
PIP = 0.01


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params: dict
    sig_idx: np.ndarray
    sig_side: np.ndarray


def load(tf: int, years: int) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    sql = """
        SELECT start_time, open, high, low, close, volume
        FROM ig_candles_backtest
        WHERE epic = %s AND timeframe = %s
          AND start_time >= NOW() - INTERVAL %s
        ORDER BY start_time
    """
    df = pd.read_sql(sql, conn, params=(EPIC, tf, f"{years} years"), parse_dates=["start_time"])
    conn.close()
    df = df.drop_duplicates("start_time").set_index("start_time").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def pack(name: str, family: str, params: dict, idx: np.ndarray, side: int) -> Candidate:
    return Candidate(name, family, params, idx.astype(np.int32), np.full(len(idx), side, dtype=np.int8))


def simulate(
    df: pd.DataFrame,
    cand: Candidate,
    sl: float,
    tp: float,
    cost: float,
    cooldown: int,
    hold: int,
    be_trigger: float = 0.0,
    be_lock: float = 0.0,
) -> list[dict]:
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    index = df.index
    trades = []
    blocked_until = -1
    n = len(df)
    for side, i in zip(cand.sig_side, cand.sig_idx):
        i = int(i)
        side = int(side)
        if i <= blocked_until or i >= n - 2:
            continue
        entry = closes[i]
        sl_px = entry - side * sl * PIP
        tp_px = entry + side * tp * PIP
        end = min(i + hold, n - 1)
        exit_i = end
        exit_px = closes[end]
        outcome = "TIMEOUT"
        active_sl = sl_px
        armed_be = False
        for j in range(i + 1, end + 1):
            if side == 1:
                if be_trigger > 0 and not armed_be and highs[j] >= entry + be_trigger * PIP:
                    active_sl = max(active_sl, entry + be_lock * PIP)
                    armed_be = True
                hit_sl = lows[j] <= active_sl
                hit_tp = highs[j] >= tp_px
            else:
                if be_trigger > 0 and not armed_be and lows[j] <= entry - be_trigger * PIP:
                    active_sl = min(active_sl, entry - be_lock * PIP)
                    armed_be = True
                hit_sl = highs[j] >= active_sl
                hit_tp = lows[j] <= tp_px
            if hit_sl:
                exit_i = j
                exit_px = active_sl
                outcome = "SL" if not armed_be or abs(active_sl - sl_px) < 1e-12 else "BE"
                break
            if hit_tp:
                exit_i = j
                exit_px = tp_px
                outcome = "TP"
                break
        pips = ((exit_px - entry) * side / PIP) - cost
        trades.append({"entry_ts": index[i], "exit_ts": index[exit_i], "pips": pips, "side": side, "outcome": outcome})
        blocked_until = exit_i + cooldown
    return trades


def metrics(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    months = max((end - start).days / 30.4375, 0.01)
    if not trades:
        return {"n": 0, "tpm": 0, "pf": 0, "wr": 0, "pips": 0, "dd": 0}
    p = pd.Series([t["pips"] for t in trades], dtype=float)
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    cum = p.cumsum()
    return {
        "n": int(len(p)),
        "tpm": float(len(p) / months),
        "pf": float(wins / losses) if losses > 0 else 999.0,
        "wr": float((p > 0).mean()),
        "pips": float(p.sum()),
        "dd": float((cum.cummax() - cum).max()),
    }


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    f["hour"] = df.index.hour
    f["dow"] = df.index.dayofweek
    f["ret1"] = c.diff() / PIP
    f["ret3"] = c.diff(3) / PIP
    f["ret12"] = c.diff(12) / PIP
    f["rsi7"] = rsi(c, 7)
    f["rsi14"] = rsi(c, 14)
    a = atr(df, 14) / PIP
    f["atrp"] = a.rolling(400, min_periods=200).rank(pct=True)
    ema30 = c.ewm(span=30, adjust=False).mean()
    ema120 = c.ewm(span=120, adjust=False).mean()
    f["ema_slope"] = (ema30 - ema120) / PIP
    f["bar_body"] = (df["close"] - df["open"]) / PIP
    f["range"] = (df["high"] - df["low"]) / PIP
    return f


def make_candidates(df: pd.DataFrame, max_signal_tpm: float, min_signal_tpm: float, edge_only: bool):
    f = feature_frame(df)
    months = (df.index.max() - df.index.min()).days / 30.4375
    min_n = int(min_signal_tpm * months * 0.35)
    max_n = int(max_signal_tpm * months)

    masks: list[tuple[str, np.ndarray]] = []
    hour_groups = {
        "asia": np.isin(f["hour"], [0, 1, 2, 3, 4, 5]),
        "london_open": np.isin(f["hour"], [6, 7, 8, 9]),
        "london_mid": np.isin(f["hour"], [10, 11, 12, 13]),
        "ny_overlap": np.isin(f["hour"], [13, 14, 15, 16, 17]),
        "ny_late": np.isin(f["hour"], [18, 19, 20, 21]),
    }
    dow_groups = {
        "mon": f["dow"] == 0,
        "tue": f["dow"] == 1,
        "wed": f["dow"] == 2,
        "thu": f["dow"] == 3,
        "fri": f["dow"] == 4,
        "tue_thu": np.isin(f["dow"], [1, 2, 3]),
        "mon_fri": np.isin(f["dow"], [0, 4]),
    }
    regimes = {
        "atr_low": f["atrp"] < 0.33,
        "atr_mid": (f["atrp"] >= 0.33) & (f["atrp"] <= 0.66),
        "atr_high": f["atrp"] > 0.66,
        "ema_up": f["ema_slope"] > 0,
        "ema_down": f["ema_slope"] < 0,
        "rsi7_low": f["rsi7"] < 35,
        "rsi7_high": f["rsi7"] > 65,
        "rsi14_low": f["rsi14"] < 38,
        "rsi14_high": f["rsi14"] > 62,
        "mom3_up": f["ret3"] > 5,
        "mom3_down": f["ret3"] < -5,
        "mom12_up": f["ret12"] > 10,
        "mom12_down": f["ret12"] < -10,
        "large_green": f["bar_body"] > 4,
        "large_red": f["bar_body"] < -4,
        "small_bar": f["range"] < f["range"].rolling(400, min_periods=200).quantile(0.35),
    }
    for hname, hm in hour_groups.items():
        masks.append((hname, np.asarray(hm, dtype=bool)))
        for dname, dm in dow_groups.items():
            masks.append((f"{hname}+{dname}", np.asarray(hm & dm, dtype=bool)))
        for rname, rm in regimes.items():
            masks.append((f"{hname}+{rname}", np.asarray(hm & rm, dtype=bool)))
    for dname, dm in dow_groups.items():
        for rname, rm in regimes.items():
            masks.append((f"{dname}+{rname}", np.asarray(dm & rm, dtype=bool)))

    # A small second-order pass: combine the most practical regime masks with a
    # session. This finds things like "Tuesday London, high vol, short".
    regime_items = list(regimes.items())
    for hname, hm in hour_groups.items():
        for (r1, m1), (r2, m2) in combinations(regime_items, 2):
            if r1.split("_")[0] == r2.split("_")[0]:
                continue
            masks.append((f"{hname}+{r1}+{r2}", np.asarray(hm & m1 & m2, dtype=bool)))

    seen = set()
    for name, mask in masks:
        clean = mask & np.isfinite(f["atrp"].to_numpy(float))
        if edge_only:
            clean = clean & ~np.r_[False, clean[:-1]]
        idx = np.flatnonzero(clean)
        if len(idx) < min_n or len(idx) > max_n:
            continue
        for side, sname in ((1, "buy"), (-1, "sell")):
            key = (name, side, len(idx), int(idx[0]) if len(idx) else -1)
            if key in seen:
                continue
            seen.add(key)
            yield pack(f"{sname}_{name}", "conditional_behavior", {"condition": name, "side": sname}, idx, side)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=15)
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--cost-pips", type=float, default=1.2)
    ap.add_argument("--min-tpm", type=float, default=10)
    ap.add_argument("--max-signal-tpm", type=float, default=45)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--edge-only", action="store_true")
    ap.add_argument("--be-grid", action="store_true")
    ap.add_argument("--must-contain", default="")
    args = ap.parse_args()

    df = load(args.tf, args.years)
    print(f"Loaded EURJPY {args.tf}m: {len(df):,} rows {df.index.min()} -> {df.index.max()}", flush=True)
    exits = [
        (8, 12, 2, 48), (10, 15, 3, 72), (12, 18, 4, 96),
        (15, 30, 6, 160), (20, 30, 6, 192), (20, 40, 8, 240),
    ]
    management = [(0.0, 0.0)]
    if args.be_grid:
        management = [(0.0, 0.0), (5.0, 0.0), (6.0, 1.0), (8.0, 1.0), (10.0, 2.0), (12.0, 2.0)]
    results = []
    count = 0
    for cand in make_candidates(df, args.max_signal_tpm, args.min_tpm, args.edge_only):
        if args.must_contain and args.must_contain not in cand.name:
            continue
        count += 1
        for ex in exits:
            sl, tp, cd, hold = ex
            for be_trigger, be_lock in management:
                trades = simulate(df, cand, sl, tp, args.cost_pips, cd, hold, be_trigger, be_lock)
                m = metrics(trades, df.index.min(), df.index.max())
                if m["tpm"] >= args.min_tpm:
                    results.append((m["pf"], m["pips"], m["n"], cand, (sl, tp, cd, hold, be_trigger, be_lock), trades, m))
    results.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    q = [r for r in results if r[6]["pf"] >= 1.7]
    print(f"Evaluated candidates={count:,}, exit results={len(results):,}, qualified={len(q):,}", flush=True)
    split = pd.Timestamp("2025-01-01", tz="UTC")
    rows = q[: args.top] if q else results[: args.top]
    print("rank pf tpm n wr pips dd exit train_pf/test_pf name params")
    for rank, (_, _, _, cand, ex, trades, m) in enumerate(rows, 1):
        train = [t for t in trades if t["entry_ts"] < split]
        test = [t for t in trades if t["entry_ts"] >= split]
        tm = metrics(train, min((t["entry_ts"] for t in train), default=split), split)
        vm = metrics(test, split, df.index.max())
        sl, tp, cd, hold, be_trigger, be_lock = ex
        mgmt = "" if be_trigger <= 0 else f"/BE{be_trigger}+{be_lock}"
        print(f"{rank:02d} {m['pf']:.2f} {m['tpm']:.1f} {m['n']:4d} {m['wr']:.1%} {m['pips']:.0f} {m['dd']:.0f} SL{sl}/TP{tp}/cd{cd}/hold{hold}{mgmt} {tm['pf']:.2f}/{vm['pf']:.2f} {cand.name} {cand.params}")
    if rows:
        best = rows[0]
        monthly = pd.DataFrame(best[5]).assign(month=lambda x: x["entry_ts"].dt.to_period("M")).groupby("month")["pips"].agg(["count", "sum"])
        print("\nBest monthly tail:")
        print(monthly.tail(18).to_string())


if __name__ == "__main__":
    main()
