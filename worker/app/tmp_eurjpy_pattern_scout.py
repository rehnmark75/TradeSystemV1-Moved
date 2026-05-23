#!/usr/bin/env python3
"""EURJPY candle-pattern scout.

Searches simple, deployable OHLC patterns on local ig_candles_backtest data.
The goal is not to curve-fit an exotic model; it is to find a repeatable rule
that clears minimum trade frequency and profit factor.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable

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
    timeframe: int
    params: dict
    signals: np.ndarray


def load_candles(years: int) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    sql = """
        SELECT start_time, open, high, low, close, volume
        FROM ig_candles_backtest
        WHERE epic = %s
          AND timeframe = 5
          AND start_time >= NOW() - INTERVAL %s
        ORDER BY start_time
    """
    df = pd.read_sql(sql, conn, params=(EPIC, f"{years} years"), parse_dates=["start_time"])
    conn.close()
    df = df.drop_duplicates("start_time").set_index("start_time").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 5:
        return df.copy()
    return df.resample(f"{minutes}min", closed="left", label="left").agg(
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


def apply_hours(sig: np.ndarray, idx: pd.DatetimeIndex, hours: tuple[int, int] | None) -> np.ndarray:
    if hours is None:
        return sig
    start, end = hours
    h = idx.hour
    mask = (h >= start) & (h <= end) if start <= end else ((h >= start) | (h <= end))
    return np.where(mask, sig, 0)


def simulate(
    df: pd.DataFrame,
    signals: np.ndarray,
    sl_pips: float,
    tp_pips: float,
    cost_pips: float,
    cooldown_bars: int,
    max_hold_bars: int,
) -> list[dict]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    idx = df.index
    trades: list[dict] = []
    i = 0
    cooldown_until = -1
    while i < len(df):
        if i <= cooldown_until or signals[i] == 0:
            i += 1
            continue
        sign = int(signals[i])
        entry = float(closes[i])
        sl = entry - sign * sl_pips * PIP
        tp = entry + sign * tp_pips * PIP
        outcome = "TIMEOUT"
        exit_i = min(i + max_hold_bars, len(df) - 1)
        exit_px = float(closes[exit_i])
        for j in range(i + 1, min(i + 1 + max_hold_bars, len(df))):
            if sign == 1:
                hit_sl = lows[j] <= sl
                hit_tp = highs[j] >= tp
            else:
                hit_sl = highs[j] >= sl
                hit_tp = lows[j] <= tp
            if hit_sl:
                outcome = "SL"
                exit_i = j
                exit_px = sl
                break
            if hit_tp:
                outcome = "TP"
                exit_i = j
                exit_px = tp
                break
        pips = ((exit_px - entry) * sign / PIP) - cost_pips
        trades.append(
            {
                "entry_ts": idx[i],
                "exit_ts": idx[exit_i],
                "direction": "BUY" if sign == 1 else "SELL",
                "pips": pips,
                "outcome": outcome,
                "bars": exit_i - i,
            }
        )
        cooldown_until = exit_i + cooldown_bars
        i = exit_i + 1
    return trades


def metrics(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    months = max((end - start).days / 30.4375, 0.01)
    if not trades:
        return {"n": 0, "tpm": 0, "pf": 0, "wr": 0, "pips": 0, "max_dd": 0}
    p = pd.Series([t["pips"] for t in trades], dtype=float)
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    cum = p.cumsum()
    dd = float((cum.cummax() - cum).max())
    return {
        "n": int(len(p)),
        "tpm": float(len(p) / months),
        "pf": float(wins / losses) if losses > 0 else 999.0,
        "wr": float((p > 0).mean()),
        "pips": float(p.sum()),
        "max_dd": dd,
        "avg": float(p.mean()),
    }


def make_candidates(df: pd.DataFrame, tf: int) -> Iterable[Candidate]:
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    atr14 = atr(df, 14)
    adx14 = adx(df, 14)
    hours_sets = {"all": None, "london": (6, 18), "eu_us": (7, 20), "asia_london": (0, 12)}

    for rsi_n in (7, 10, 14):
        rr = rsi(c, rsi_n)
        for bb_n in (20, 32, 48):
            ma = c.rolling(bb_n).mean()
            sd = c.rolling(bb_n).std()
            z = (c - ma) / sd.replace(0, np.nan)
            for zthr in (1.4, 1.7, 2.0):
                for rlo, rhi in ((30, 70), (35, 65), (25, 75), (40, 60)):
                    for trend_mode in ("any", "with_ema", "against_ema", "adx_lt_24"):
                        sig = np.zeros(len(df), dtype=np.int8)
                        long = (z < -zthr) & (rr < rlo)
                        short = (z > zthr) & (rr > rhi)
                        if trend_mode == "with_ema":
                            long &= c > ema200
                            short &= c < ema200
                        elif trend_mode == "against_ema":
                            long &= c < ema200
                            short &= c > ema200
                        elif trend_mode == "adx_lt_24":
                            long &= adx14 < 24
                            short &= adx14 < 24
                        sig[long.fillna(False).to_numpy()] = 1
                        sig[short.fillna(False).to_numpy()] = -1
                        for hname, hours in hours_sets.items():
                            yield Candidate(
                                f"bb_rsi_{tf}_{rsi_n}_{bb_n}_{zthr}_{rlo}_{rhi}_{trend_mode}_{hname}",
                                "bb_rsi_reversion",
                                tf,
                                {"rsi": rsi_n, "bb": bb_n, "z": zthr, "rsi_lo": rlo, "rsi_hi": rhi, "filter": trend_mode, "hours": hname},
                                apply_hours(sig, df.index, hours),
                            )

    for rsi_n in (7, 10, 14):
        rr = rsi(c, rsi_n)
        for wick_min in (0.35, 0.45, 0.55, 0.65):
            for close_pos in (0.45, 0.55, 0.65):
                loc = (c - l) / rng
                for rlo, rhi in ((30, 70), (35, 65), (40, 60)):
                    bull = (lower_wick / rng > wick_min) & (loc > close_pos) & (rr < rlo)
                    bear = (upper_wick / rng > wick_min) & (loc < 1 - close_pos) & (rr > rhi)
                    for filt in ("any", "adx_lt_26", "ema_side", "against_ema200"):
                        long = bull.copy()
                        short = bear.copy()
                        if filt == "adx_lt_26":
                            long &= adx14 < 26
                            short &= adx14 < 26
                        elif filt == "ema_side":
                            long &= c < ema50
                            short &= c > ema50
                        elif filt == "against_ema200":
                            long &= c < ema200
                            short &= c > ema200
                        sig = np.zeros(len(df), dtype=np.int8)
                        sig[long.fillna(False).to_numpy()] = 1
                        sig[short.fillna(False).to_numpy()] = -1
                        for hname, hours in hours_sets.items():
                            yield Candidate(
                                f"wick_rsi_{tf}_{rsi_n}_{wick_min}_{close_pos}_{rlo}_{rhi}_{filt}_{hname}",
                                "wick_rsi_reversal",
                                tf,
                                {"rsi": rsi_n, "wick_min": wick_min, "close_pos": close_pos, "rsi_lo": rlo, "rsi_hi": rhi, "filter": filt, "hours": hname},
                                apply_hours(sig, df.index, hours),
                            )

    ret = c.diff()
    signed = np.sign(ret).replace(0, np.nan).ffill().fillna(0)
    for run_len in (2, 3, 4):
        up_run = (signed > 0).rolling(run_len).sum() == run_len
        down_run = (signed < 0).rolling(run_len).sum() == run_len
        impulse = body / rng
        for min_body in (0.45, 0.55, 0.65):
            for filt in ("any", "ema_trend", "adx_gt_18", "adx_lt_24"):
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
                sig = np.zeros(len(df), dtype=np.int8)
                sig[long.fillna(False).to_numpy()] = 1
                sig[short.fillna(False).to_numpy()] = -1
                for hname, hours in hours_sets.items():
                    yield Candidate(
                        f"candle_run_{tf}_{run_len}_{min_body}_{filt}_{hname}",
                        "candle_run_continuation",
                        tf,
                        {"run_len": run_len, "min_body": min_body, "filter": filt, "hours": hname},
                        apply_hours(sig, df.index, hours),
                    )

    for rev_len in (2, 3):
        up_run_prev = (signed.shift(1) > 0).rolling(rev_len).sum() == rev_len
        down_run_prev = (signed.shift(1) < 0).rolling(rev_len).sum() == rev_len
        loc = (c - l) / rng
        for wick_min in (0.35, 0.45, 0.55):
            bull = down_run_prev & (lower_wick / rng > wick_min) & (loc > 0.55)
            bear = up_run_prev & (upper_wick / rng > wick_min) & (loc < 0.45)
            for filt in ("any", "adx_lt_24", "ema_side"):
                long = bull.copy()
                short = bear.copy()
                if filt == "adx_lt_24":
                    long &= adx14 < 24
                    short &= adx14 < 24
                elif filt == "ema_side":
                    long &= c < ema50
                    short &= c > ema50
                sig = np.zeros(len(df), dtype=np.int8)
                sig[long.fillna(False).to_numpy()] = 1
                sig[short.fillna(False).to_numpy()] = -1
                for hname, hours in hours_sets.items():
                    yield Candidate(
                        f"exhaustion_reversal_{tf}_{rev_len}_{wick_min}_{filt}_{hname}",
                        "exhaustion_reversal",
                        tf,
                        {"prev_run": rev_len, "wick_min": wick_min, "filter": filt, "hours": hname},
                        apply_hours(sig, df.index, hours),
                    )


def eval_candidate(df: pd.DataFrame, cand: Candidate, exits: list[tuple[float, float, int, int]], cost: float, min_tpm: float) -> list[dict]:
    out = []
    for sl, tp, cooldown, hold in exits:
        trades = simulate(df, cand.signals, sl, tp, cost, cooldown, hold)
        m = metrics(trades, df.index.min(), df.index.max())
        if m["tpm"] >= min_tpm:
            out.append({"cand": cand, "exit": (sl, tp, cooldown, hold), "trades": trades, "m": m})
    return out


def split_metrics(trades: list[dict], split: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, dict]:
    train = [t for t in trades if t["entry_ts"] < split]
    test = [t for t in trades if t["entry_ts"] >= split]
    start_train = min((t["entry_ts"] for t in train), default=split)
    return metrics(train, start_train, split), metrics(test, split, end)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--min-tpm", type=float, default=10.0)
    ap.add_argument("--cost-pips", type=float, default=1.2)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    base = load_candles(args.years)
    print(f"Loaded EURJPY 5m: {len(base):,} rows {base.index.min()} -> {base.index.max()}")
    exits = [
        (8, 10, 3, 72), (8, 12, 3, 96), (10, 12, 3, 96), (10, 15, 3, 120),
        (12, 18, 4, 144), (12, 24, 4, 192), (15, 22, 4, 192), (15, 30, 6, 288),
        (18, 27, 6, 288), (20, 30, 6, 288), (25, 40, 8, 384), (30, 45, 8, 384),
    ]
    results = []
    for tf in (5, 15):
        df = resample(base, tf)
        cands = 0
        for cand in make_candidates(df, tf):
            cands += 1
            results.extend(eval_candidate(df, cand, exits, args.cost_pips, args.min_tpm))
        print(f"Evaluated {cands:,} signal variants on {tf}m")

    results.sort(key=lambda r: (r["m"]["pf"], r["m"]["pips"], r["m"]["n"]), reverse=True)
    qualifying = [r for r in results if r["m"]["pf"] >= 1.7 and r["m"]["tpm"] >= args.min_tpm]
    print(f"\nQualified PF>=1.70 and trades/month>={args.min_tpm}: {len(qualifying)}")
    rows = qualifying[: args.top] if qualifying else results[: args.top]
    split = pd.Timestamp("2025-01-01", tz="UTC")
    print("\nrank pf tpm n wr pips dd tf family exit train_pf/test_pf name params")
    for rank, r in enumerate(rows, 1):
        cand = r["cand"]
        train_m, test_m = split_metrics(r["trades"], split, base.index.max())
        sl, tp, cooldown, hold = r["exit"]
        m = r["m"]
        print(
            f"{rank:02d} {m['pf']:.2f} {m['tpm']:.1f} {m['n']:4d} {m['wr']:.1%} "
            f"{m['pips']:.0f} {m['max_dd']:.0f} {cand.timeframe}m {cand.family} "
            f"SL{sl}/TP{tp}/cd{cooldown}/hold{hold} "
            f"{train_m['pf']:.2f}/{test_m['pf']:.2f} {cand.name} {cand.params}"
        )

    if qualifying:
        best = qualifying[0]
        df_trades = pd.DataFrame(best["trades"])
        by_month = df_trades.assign(month=df_trades["entry_ts"].dt.to_period("M")).groupby("month")["pips"].agg(["count", "sum"])
        print("\nBest candidate monthly count/pips tail:")
        print(by_month.tail(18).to_string())


if __name__ == "__main__":
    main()
