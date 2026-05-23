#!/usr/bin/env python3
"""Fast non-Donchian EURJPY candle-pattern scout."""

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
    tf: int
    params: dict
    sig_idx: np.ndarray
    sig_side: np.ndarray


def load_base(years: int) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    sql = """
        SELECT start_time, open, high, low, close, volume
        FROM ig_candles_backtest
        WHERE epic = %s AND timeframe = 5
          AND start_time >= NOW() - INTERVAL %s
        ORDER BY start_time
    """
    df = pd.read_sql(sql, conn, params=(EPIC, f"{years} years"), parse_dates=["start_time"])
    conn.close()
    df = df.drop_duplicates("start_time").set_index("start_time").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def resample(df: pd.DataFrame, tf: int) -> pd.DataFrame:
    if tf == 5:
        return df.copy()
    return df.resample(f"{tf}min", closed="left", label="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
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


def hour_mask(index: pd.DatetimeIndex, label: str) -> np.ndarray:
    if label == "all":
        return np.ones(len(index), dtype=bool)
    start, end = {
        "london": (6, 18),
        "eu_us": (7, 20),
        "asia_london": (0, 12),
        "ny": (12, 21),
    }[label]
    h = index.hour
    return np.asarray((h >= start) & (h <= end), dtype=bool)


def pack(name: str, family: str, tf: int, params: dict, sig: np.ndarray) -> Candidate:
    idx = np.flatnonzero(sig)
    return Candidate(name, family, tf, params, idx.astype(np.int32), sig[idx].astype(np.int8))


def candidates(df: pd.DataFrame, tf: int):
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - pd.concat([c, o], axis=1).max(axis=1)
    lower = pd.concat([c, o], axis=1).min(axis=1) - l
    loc = (c - l) / rng
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    adx14 = adx(df, 14)
    signed = np.sign(c.diff()).replace(0, np.nan).ffill().fillna(0)
    hours = ("all", "london", "eu_us", "asia_london")

    for rn in (7, 10, 14):
        rr = rsi(c, rn)
        for bn in (20, 32, 48):
            ma = c.rolling(bn).mean()
            sd = c.rolling(bn).std()
            z = (c - ma) / sd.replace(0, np.nan)
            for zthr in (1.4, 1.7, 2.0):
                for lo_thr, hi_thr in ((30, 70), (35, 65), (40, 60)):
                    base_long = (z < -zthr) & (rr < lo_thr)
                    base_short = (z > zthr) & (rr > hi_thr)
                    for filt in ("any", "adx_lt_24", "against_ema200"):
                        long = base_long.copy()
                        short = base_short.copy()
                        if filt == "adx_lt_24":
                            long &= adx14 < 24
                            short &= adx14 < 24
                        elif filt == "against_ema200":
                            long &= c < ema200
                            short &= c > ema200
                        for hr in hours:
                            mask = hour_mask(df.index, hr)
                            sig = np.zeros(len(df), dtype=np.int8)
                            sig[(long.fillna(False).to_numpy()) & mask] = 1
                            sig[(short.fillna(False).to_numpy()) & mask] = -1
                            yield pack(f"bb_rsi_{tf}_{rn}_{bn}_{zthr}_{lo_thr}_{hi_thr}_{filt}_{hr}", "bb_rsi_reversion", tf, {"rsi": rn, "bb": bn, "z": zthr, "rsi_lo": lo_thr, "rsi_hi": hi_thr, "filter": filt, "hours": hr}, sig)

    for rn in (7, 10, 14):
        rr = rsi(c, rn)
        for wick_min in (0.45, 0.55, 0.65):
            for close_pos in (0.55, 0.65):
                for lo_thr, hi_thr in ((30, 70), (35, 65), (40, 60)):
                    base_long = (lower / rng > wick_min) & (loc > close_pos) & (rr < lo_thr)
                    base_short = (upper / rng > wick_min) & (loc < 1 - close_pos) & (rr > hi_thr)
                    for filt in ("any", "adx_lt_26", "against_ema200"):
                        long = base_long.copy()
                        short = base_short.copy()
                        if filt == "adx_lt_26":
                            long &= adx14 < 26
                            short &= adx14 < 26
                        elif filt == "against_ema200":
                            long &= c < ema200
                            short &= c > ema200
                        for hr in hours:
                            mask = hour_mask(df.index, hr)
                            sig = np.zeros(len(df), dtype=np.int8)
                            sig[(long.fillna(False).to_numpy()) & mask] = 1
                            sig[(short.fillna(False).to_numpy()) & mask] = -1
                            yield pack(f"wick_rsi_{tf}_{rn}_{wick_min}_{close_pos}_{lo_thr}_{hi_thr}_{filt}_{hr}", "wick_rsi_reversal", tf, {"rsi": rn, "wick_min": wick_min, "close_pos": close_pos, "rsi_lo": lo_thr, "rsi_hi": hi_thr, "filter": filt, "hours": hr}, sig)

    impulse = body / rng
    for run_len in (2, 3, 4):
        up_run = (signed > 0).rolling(run_len).sum() == run_len
        down_run = (signed < 0).rolling(run_len).sum() == run_len
        for min_body in (0.55, 0.65):
            for filt in ("ema_trend", "adx_gt_18", "adx_lt_24"):
                long = up_run & (impulse > min_body)
                short = down_run & (impulse > min_body)
                if filt == "ema_trend":
                    long &= ema50 > ema200
                    short &= ema50 < ema200
                elif filt == "adx_gt_18":
                    long &= adx14 > 18
                    short &= adx14 > 18
                elif filt == "adx_lt_24":
                    long &= adx14 < 24
                    short &= adx14 < 24
                for hr in hours:
                    mask = hour_mask(df.index, hr)
                    sig = np.zeros(len(df), dtype=np.int8)
                    sig[(long.fillna(False).to_numpy()) & mask] = 1
                    sig[(short.fillna(False).to_numpy()) & mask] = -1
                    yield pack(f"candle_run_{tf}_{run_len}_{min_body}_{filt}_{hr}", "candle_run_continuation", tf, {"run_len": run_len, "min_body": min_body, "filter": filt, "hours": hr}, sig)

    for rev_len in (2, 3):
        up_prev = (signed.shift(1) > 0).rolling(rev_len).sum() == rev_len
        down_prev = (signed.shift(1) < 0).rolling(rev_len).sum() == rev_len
        for wick_min in (0.45, 0.55, 0.65):
            base_long = down_prev & (lower / rng > wick_min) & (loc > 0.55)
            base_short = up_prev & (upper / rng > wick_min) & (loc < 0.45)
            for filt in ("any", "adx_lt_24", "against_ema200"):
                long = base_long.copy()
                short = base_short.copy()
                if filt == "adx_lt_24":
                    long &= adx14 < 24
                    short &= adx14 < 24
                elif filt == "against_ema200":
                    long &= c < ema200
                    short &= c > ema200
                for hr in hours:
                    mask = hour_mask(df.index, hr)
                    sig = np.zeros(len(df), dtype=np.int8)
                    sig[(long.fillna(False).to_numpy()) & mask] = 1
                    sig[(short.fillna(False).to_numpy()) & mask] = -1
                    yield pack(f"exhaustion_{tf}_{rev_len}_{wick_min}_{filt}_{hr}", "exhaustion_reversal", tf, {"prev_run": rev_len, "wick_min": wick_min, "filter": filt, "hours": hr}, sig)


def simulate(df: pd.DataFrame, cand: Candidate, sl: float, tp: float, cost: float, cooldown: int, hold: int) -> list[dict]:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    idx = df.index
    trades = []
    blocked_until = -1
    n = len(df)
    for k, i in zip(cand.sig_side, cand.sig_idx):
        i = int(i)
        side = int(k)
        if i <= blocked_until or i >= n - 2:
            continue
        entry = closes[i]
        sl_px = entry - side * sl * PIP
        tp_px = entry + side * tp * PIP
        end = min(i + hold, n - 1)
        exit_i = end
        exit_px = closes[end]
        outcome = "TIMEOUT"
        if side == 1:
            hit_sl = np.flatnonzero(lows[i + 1 : end + 1] <= sl_px)
            hit_tp = np.flatnonzero(highs[i + 1 : end + 1] >= tp_px)
        else:
            hit_sl = np.flatnonzero(highs[i + 1 : end + 1] >= sl_px)
            hit_tp = np.flatnonzero(lows[i + 1 : end + 1] <= tp_px)
        first_sl = int(hit_sl[0]) if len(hit_sl) else None
        first_tp = int(hit_tp[0]) if len(hit_tp) else None
        if first_sl is not None and (first_tp is None or first_sl <= first_tp):
            exit_i = i + 1 + first_sl
            exit_px = sl_px
            outcome = "SL"
        elif first_tp is not None:
            exit_i = i + 1 + first_tp
            exit_px = tp_px
            outcome = "TP"
        pips = ((exit_px - entry) * side / PIP) - cost
        trades.append({"entry_ts": idx[i], "exit_ts": idx[exit_i], "direction": "BUY" if side == 1 else "SELL", "pips": pips, "outcome": outcome, "bars": exit_i - i})
        blocked_until = exit_i + cooldown
    return trades


def metrics(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    months = max((end - start).days / 30.4375, 0.01)
    if not trades:
        return {"n": 0, "tpm": 0, "pf": 0, "wr": 0, "pips": 0, "dd": 0, "avg": 0}
    p = pd.Series([t["pips"] for t in trades])
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
        "avg": float(p.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--cost-pips", type=float, default=1.2)
    ap.add_argument("--min-tpm", type=float, default=10)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tfs", default="5,15")
    ap.add_argument("--exit-set", choices=("small", "full"), default="small")
    ap.add_argument("--max-signal-tpm", type=float, default=80)
    ap.add_argument("--include-inverse", action="store_true")
    args = ap.parse_args()

    base = load_base(args.years)
    print(f"Loaded EURJPY: {len(base):,} 5m candles {base.index.min()} -> {base.index.max()}", flush=True)
    exit_sets = {
        "small": [(10, 15, 3, 120), (12, 18, 4, 144), (15, 30, 6, 288)],
        "full": [(8, 12, 3, 96), (10, 15, 3, 120), (12, 18, 4, 144), (12, 24, 4, 192), (15, 30, 6, 288), (20, 30, 6, 288)],
    }
    exits = exit_sets[args.exit_set]
    results = []
    tfs = tuple(int(x.strip()) for x in args.tfs.split(",") if x.strip())
    for tf in tfs:
        df = resample(base, tf)
        months = (df.index.max() - df.index.min()).days / 30.4375
        min_signals = int(args.min_tpm * months * 0.35)
        max_signals = int(args.max_signal_tpm * months)
        count = 0
        skipped_sparse = 0
        skipped_dense = 0
        for cand in candidates(df, tf):
            count += 1
            variants = [cand]
            if args.include_inverse:
                variants.append(Candidate(f"inverse_{cand.name}", f"inverse_{cand.family}", cand.tf, {**cand.params, "inverse": True}, cand.sig_idx, -cand.sig_side))
            for vcand in variants:
                if len(vcand.sig_idx) < min_signals:
                    skipped_sparse += 1
                    continue
                if len(vcand.sig_idx) > max_signals:
                    skipped_dense += 1
                    continue
                for ex in exits:
                    sl, tp, cd, hold = ex
                    trades = simulate(df, vcand, sl, tp, args.cost_pips, cd, hold)
                    m = metrics(trades, df.index.min(), df.index.max())
                    if m["tpm"] >= args.min_tpm:
                        results.append((m["pf"], m["pips"], m["n"], vcand, ex, trades, m, df.index.max()))
        print(f"Evaluated {count:,} non-Donchian variants on {tf}m; skipped sparse={skipped_sparse:,} dense={skipped_dense:,}; retained {len(results):,} exit results so far", flush=True)

    results.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    qualified = [r for r in results if r[6]["pf"] >= 1.7 and r[6]["tpm"] >= args.min_tpm]
    rows = qualified[: args.top] if qualified else results[: args.top]
    print(f"\nQualified PF>=1.70 and trades/month>={args.min_tpm}: {len(qualified)}")
    print("rank pf tpm n wr pips dd tf family exit train_pf/test_pf name params")
    split = pd.Timestamp("2025-01-01", tz="UTC")
    for rank, (_, _, _, cand, ex, trades, m, end) in enumerate(rows, 1):
        train = [t for t in trades if t["entry_ts"] < split]
        test = [t for t in trades if t["entry_ts"] >= split]
        tm = metrics(train, min((t["entry_ts"] for t in train), default=split), split)
        vm = metrics(test, split, end)
        sl, tp, cd, hold = ex
        print(f"{rank:02d} {m['pf']:.2f} {m['tpm']:.1f} {m['n']:4d} {m['wr']:.1%} {m['pips']:.0f} {m['dd']:.0f} {cand.tf}m {cand.family} SL{sl}/TP{tp}/cd{cd}/hold{hold} {tm['pf']:.2f}/{vm['pf']:.2f} {cand.name} {cand.params}")

    if rows:
        best = rows[0]
        trades = best[5]
        monthly = pd.DataFrame(trades).assign(month=lambda x: x["entry_ts"].dt.to_period("M")).groupby("month")["pips"].agg(["count", "sum"])
        print("\nBest monthly tail:")
        print(monthly.tail(18).to_string())


if __name__ == "__main__":
    main()
