#!/usr/bin/env python3
"""EURJPY non-Donchian trend/momentum scout."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

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


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def pack(name: str, family: str, params: dict, sig: np.ndarray) -> Candidate:
    idx = np.flatnonzero(sig)
    return Candidate(name, family, params, idx.astype(np.int32), sig[idx].astype(np.int8))


def hour_mask(index: pd.DatetimeIndex, label: str) -> np.ndarray:
    h = index.hour
    if label == "all":
        return np.ones(len(index), dtype=bool)
    windows = {
        "london": (6, 18),
        "eu_us": (7, 20),
        "ny": (12, 21),
        "ny_late": (18, 21),
        "asia_london": (0, 12),
    }
    start, end = windows[label]
    return np.asarray((h >= start) & (h <= end), dtype=bool)


def make_candidates(df: pd.DataFrame, min_signal_tpm: float, max_signal_tpm: float):
    c = df["close"]
    o = df["open"]
    body = (c - o) / PIP
    rng = (df["high"] - df["low"]) / PIP
    a = atr(df, 14) / PIP
    ax = adx(df, 14)
    rr = rsi(c, 14)
    months = (df.index.max() - df.index.min()).days / 30.4375
    min_n = int(min_signal_tpm * months * 0.30)
    max_n = int(max_signal_tpm * months)

    ema_sets = [(20, 80), (30, 120)]
    hours = ("london", "eu_us", "ny")
    for fast, slow in ema_sets:
        ef = c.ewm(span=fast, adjust=False).mean()
        es = c.ewm(span=slow, adjust=False).mean()
        sep = (ef - es) / PIP
        slope = ef.diff(6) / PIP
        for sep_min in (3, 6, 10, 15):
            for slope_min in (2, 4, 6):
                trend_up = (sep > sep_min) & (slope > slope_min)
                trend_dn = (sep < -sep_min) & (slope < -slope_min)
                for mom_n, mom_min in ((6, 5), (12, 8), (24, 12)):
                    mom = c.diff(mom_n) / PIP
                    for adx_min in (20, 24):
                        long = trend_up & (mom > mom_min)
                        short = trend_dn & (mom < -mom_min)
                        if adx_min:
                            long &= ax > adx_min
                            short &= ax > adx_min
                        for hr in hours:
                            mask = hour_mask(df.index, hr)
                            sig = np.zeros(len(df), dtype=np.int8)
                            sig[(long.fillna(False).to_numpy()) & mask] = 1
                            sig[(short.fillna(False).to_numpy()) & mask] = -1
                            if min_n <= np.count_nonzero(sig) <= max_n:
                                yield pack(
                                    f"ema_momentum_{fast}_{slow}_sep{sep_min}_slope{slope_min}_m{mom_n}_{mom_min}_adx{adx_min}_{hr}",
                                    "ema_momentum",
                                    {"fast": fast, "slow": slow, "sep": sep_min, "slope": slope_min, "mom_n": mom_n, "mom_min": mom_min, "adx": adx_min, "hours": hr},
                                    sig,
                                )

                # Pullback continuation: trend exists, short pullback against it,
                # then candle closes back in trend direction. This is trendfollowing,
                # not high/low breakout.
                for pull_n, pull_min in ((3, 5), (4, 7)):
                    pull = c.diff(pull_n) / PIP
                    resume_long = body > 1.5
                    resume_short = body < -1.5
                    for rlo, rhi in ((40, 60), (45, 55)):
                        long = trend_up & (pull < -pull_min) & resume_long & (rr > rlo)
                        short = trend_dn & (pull > pull_min) & resume_short & (rr < rhi)
                        for adx_min in (20, 24):
                            l2 = long.copy()
                            s2 = short.copy()
                            if adx_min:
                                l2 &= ax > adx_min
                                s2 &= ax > adx_min
                            for hr in hours:
                                mask = hour_mask(df.index, hr)
                                sig = np.zeros(len(df), dtype=np.int8)
                                sig[(l2.fillna(False).to_numpy()) & mask] = 1
                                sig[(s2.fillna(False).to_numpy()) & mask] = -1
                                if min_n <= np.count_nonzero(sig) <= max_n:
                                    yield pack(
                                        f"trend_pullback_{fast}_{slow}_sep{sep_min}_slope{slope_min}_p{pull_n}_{pull_min}_r{rlo}_{rhi}_adx{adx_min}_{hr}",
                                        "trend_pullback_resume",
                                        {"fast": fast, "slow": slow, "sep": sep_min, "slope": slope_min, "pull_n": pull_n, "pull_min": pull_min, "rsi_lo": rlo, "rsi_hi": rhi, "adx": adx_min, "hours": hr},
                                        sig,
                                    )

        # Volatility expansion in trend: large directional body in established
        # EMA trend, filtered by ATR percentile.
        atrp = a.rolling(500, min_periods=250).rank(pct=True)
        for body_min in (5, 8, 12):
            for atr_bucket, atr_cond in (("mid_hi", atrp > 0.35), ("high", atrp > 0.66)):
                long = (ef > es) & (slope > 0) & (body > body_min) & atr_cond
                short = (ef < es) & (slope < 0) & (body < -body_min) & atr_cond
                for hr in hours:
                    mask = hour_mask(df.index, hr)
                    sig = np.zeros(len(df), dtype=np.int8)
                    sig[(long.fillna(False).to_numpy()) & mask] = 1
                    sig[(short.fillna(False).to_numpy()) & mask] = -1
                    if min_n <= np.count_nonzero(sig) <= max_n:
                        yield pack(
                            f"trend_body_expansion_{fast}_{slow}_b{body_min}_{atr_bucket}_{hr}",
                            "trend_body_expansion",
                            {"fast": fast, "slow": slow, "body_min": body_min, "atr_bucket": atr_bucket, "hours": hr},
                            sig,
                        )


def simulate(df: pd.DataFrame, cand: Candidate, sl: float, tp: float, cost: float, cooldown: int, hold: int, trail_start: float = 0.0, trail_gap: float = 0.0) -> list[dict]:
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
        active_sl = sl_px
        best = entry
        end = min(i + hold, n - 1)
        exit_i = end
        exit_px = closes[end]
        outcome = "TIMEOUT"
        for j in range(i + 1, end + 1):
            if side == 1:
                best = max(best, highs[j])
                if trail_start > 0 and best >= entry + trail_start * PIP:
                    active_sl = max(active_sl, best - trail_gap * PIP)
                hit_sl = lows[j] <= active_sl
                hit_tp = highs[j] >= tp_px
            else:
                best = min(best, lows[j])
                if trail_start > 0 and best <= entry - trail_start * PIP:
                    active_sl = min(active_sl, best + trail_gap * PIP)
                hit_sl = highs[j] >= active_sl
                hit_tp = lows[j] <= tp_px
            if hit_sl:
                exit_i = j
                exit_px = active_sl
                outcome = "SL"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=15)
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--cost-pips", type=float, default=1.2)
    ap.add_argument("--min-tpm", type=float, default=10)
    ap.add_argument("--max-signal-tpm", type=float, default=60)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--include-inverse", action="store_true")
    args = ap.parse_args()

    df = load(args.tf, args.years)
    print(f"Loaded EURJPY {args.tf}m: {len(df):,} {df.index.min()} -> {df.index.max()}", flush=True)
    exits = [
        (8, 12, 2, 48, 0, 0),
        (10, 15, 3, 72, 0, 0),
        (12, 18, 4, 96, 0, 0),
        (15, 30, 6, 160, 0, 0),
        (20, 40, 8, 240, 0, 0),
        (20, 60, 8, 320, 25, 18),
        (25, 75, 8, 420, 30, 22),
        (30, 90, 10, 500, 35, 25),
    ]
    if args.quick:
        exits = [(10, 15, 3, 72, 0, 0), (15, 30, 6, 160, 0, 0), (20, 60, 8, 320, 25, 18)]
    results = []
    count = 0
    for cand in make_candidates(df, args.min_tpm, args.max_signal_tpm):
        count += 1
        variants = [cand]
        if args.include_inverse:
            variants.append(Candidate(f"inverse_{cand.name}", f"inverse_{cand.family}", {**cand.params, "inverse": True}, cand.sig_idx, -cand.sig_side))
        for vcand in variants:
            for ex in exits:
                sl, tp, cd, hold, ts, tg = ex
                trades = simulate(df, vcand, sl, tp, args.cost_pips, cd, hold, ts, tg)
                m = metrics(trades, df.index.min(), df.index.max())
                if m["tpm"] >= args.min_tpm:
                    results.append((m["pf"], m["pips"], m["n"], vcand, ex, trades, m))
    results.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    q = [r for r in results if r[6]["pf"] >= 1.7 and r[6]["tpm"] >= args.min_tpm]
    print(f"Evaluated candidates={count:,}, exit results={len(results):,}, qualified={len(q):,}", flush=True)
    split = pd.Timestamp("2025-01-01", tz="UTC")
    rows = q[: args.top] if q else results[: args.top]
    print("rank pf tpm n wr pips dd exit train_pf/test_pf name params")
    for rank, (_, _, _, cand, ex, trades, m) in enumerate(rows, 1):
        train = [t for t in trades if t["entry_ts"] < split]
        test = [t for t in trades if t["entry_ts"] >= split]
        tm = metrics(train, min((t["entry_ts"] for t in train), default=split), split)
        vm = metrics(test, split, df.index.max())
        sl, tp, cd, hold, ts, tg = ex
        trail = "" if ts <= 0 else f"/TR{ts}:{tg}"
        print(f"{rank:02d} {m['pf']:.2f} {m['tpm']:.1f} {m['n']:4d} {m['wr']:.1%} {m['pips']:.0f} {m['dd']:.0f} SL{sl}/TP{tp}/cd{cd}/hold{hold}{trail} {tm['pf']:.2f}/{vm['pf']:.2f} {cand.name} {cand.params}")
    if rows:
        monthly = pd.DataFrame(rows[0][5]).assign(month=lambda x: x["entry_ts"].dt.to_period("M")).groupby("month")["pips"].agg(["count", "sum"])
        print("\nBest monthly tail:")
        print(monthly.tail(18).to_string())


if __name__ == "__main__":
    main()
