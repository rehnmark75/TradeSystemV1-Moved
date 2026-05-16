#!/usr/bin/env python3
"""Fast 90-day RANGE_FADE optimizer.

This is a candidate finder, not the final source of truth. It mirrors the
strategy's 5m indicators and 1h HTF bias on 15m scan points, then uses a
conservative fixed SL/TP simulation. Confirm winners with backtest_cli.py.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd
import psycopg2
import numpy as np


DEFAULT_EPIC = "CS.D.GBPUSD.MINI.IP"
PIP = 0.0001
DYNAMIC_OVERRIDE_KEYS = {
    "allowed_directions",
    "london_start_hour_utc",
    "new_york_end_hour_utc",
    "blocked_hours_utc",
    "buy_blocked_hours_utc",
    "sell_blocked_hours_utc",
    "buy_start_hour_utc",
    "buy_end_hour_utc",
    "sell_start_hour_utc",
    "sell_end_hour_utc",
    "buy_allowed_hours_utc",
    "sell_allowed_hours_utc",
    "buy_allowed_htf_biases",
    "sell_allowed_htf_biases",
    "buy_adx_ceiling",
    "sell_adx_ceiling",
}


@dataclass
class Candidate:
    params: Dict[str, object]
    signals: int
    winners: int
    losers: int
    profit: float
    loss: float
    pf: float
    expectancy: float
    win_rate: float


def db_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")


def load_candles(epic: str, days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end_sql = """
        SELECT max(start_time) FROM ig_candles_backtest
        WHERE epic = %s AND timeframe = 5
    """
    conn = psycopg2.connect(db_url())
    try:
        end_ts = pd.read_sql(end_sql, conn, params=(epic,)).iloc[0, 0]
        start_ts = pd.Timestamp(end_ts) - timedelta(days=days + 10)

        def read_tf(tf: int) -> pd.DataFrame:
            sql = """
                SELECT start_time, open, high, low, close, volume
                FROM ig_candles_backtest
                WHERE epic = %s AND timeframe = %s AND start_time >= %s AND start_time <= %s
                ORDER BY start_time
            """
            df = pd.read_sql(sql, conn, params=(epic, tf, start_ts, end_ts))
            df["start_time"] = pd.to_datetime(df["start_time"])
            return df.set_index("start_time").sort_index()

        return read_tf(5), read_tf(15), read_tf(60)
    finally:
        conn.close()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    up = high - high.shift(1)
    dn = low.shift(1) - low
    plus_dm = pd.Series(((up > dn) & (up > 0)) * up.fillna(0), index=df.index)
    minus_dm = pd.Series(((dn > up) & (dn > 0)) * dn.fillna(0), index=df.index)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def prepare_features(df5: pd.DataFrame, df60: pd.DataFrame) -> pd.DataFrame:
    f = df5.copy()
    close = f["close"].astype(float)
    ma = close.rolling(20).mean()
    sd = close.rolling(20).std()
    f["rsi"] = rsi(close, 14)
    f["adx"] = adx(f)
    f["bar_range_pips"] = (f["high"] - f["low"]) / PIP
    f["prior_high_144"] = f["high"].rolling(144).max().shift(1)
    f["prior_low_144"] = f["low"].rolling(144).min().shift(1)
    f["bb_mid"] = ma
    f["bb_sd"] = sd

    htf = df60.copy()
    ema = htf["close"].ewm(span=50, adjust=False).mean()
    slope_ref = ema.shift(3)
    htf["htf_bias"] = "none"
    htf.loc[(htf["close"] > ema) & (ema > slope_ref), "htf_bias"] = "bullish"
    htf.loc[(htf["close"] < ema) & (ema < slope_ref), "htf_bias"] = "bearish"
    htf.loc[htf["htf_bias"].eq("none"), "htf_bias"] = "neutral"
    f["htf_bias"] = htf["htf_bias"].reindex(f.index, method="ffill")
    return f


def parse_blocked(raw: str) -> set[int]:
    return {int(part.strip()) for part in str(raw or "").split(",") if part.strip().isdigit()}


def signal_rows(features: pd.DataFrame, scan_index: pd.Index, params: Dict[str, object]) -> pd.DataFrame:
    f = features.reindex(scan_index, method="ffill").dropna().copy()
    bb_mult = float(params["bb_mult"])
    f["upper"] = f["bb_mid"] + bb_mult * f["bb_sd"]
    f["lower"] = f["bb_mid"] - bb_mult * f["bb_sd"]
    f["band_width_pips"] = (f["upper"] - f["lower"]) / PIP
    f["distance_to_low_pips"] = (f["close"] - f["prior_low_144"]) / PIP
    f["distance_to_high_pips"] = (f["prior_high_144"] - f["close"]) / PIP

    blocked = parse_blocked(str(params.get("blocked_hours_utc", "")))
    start_hour = int(params["london_start_hour_utc"])
    end_hour = int(params["new_york_end_hour_utc"])
    hours = pd.Series(f.index.hour, index=f.index)
    allowed_hours = (hours >= start_hour) & (hours <= end_hour) & ~hours.isin(blocked)

    allow_neutral = bool(params["allow_neutral_htf"])
    buy_htf = f["htf_bias"].eq("bullish") | (allow_neutral & f["htf_bias"].eq("neutral"))
    sell_htf = f["htf_bias"].eq("bearish") | (allow_neutral & f["htf_bias"].eq("neutral"))
    buy_allowed_htf = str(params.get("buy_allowed_htf_biases", "") or "").strip()
    if buy_allowed_htf:
        buy_htf = f["htf_bias"].isin({part.strip() for part in buy_allowed_htf.split(",") if part.strip()})
    sell_allowed_htf = str(params.get("sell_allowed_htf_biases", "") or "").strip()
    if sell_allowed_htf:
        sell_htf = f["htf_bias"].isin({part.strip() for part in sell_allowed_htf.split(",") if part.strip()})

    common = (
        allowed_hours
        & (f["bar_range_pips"] <= float(params["max_current_range_pips"]))
        & (f["band_width_pips"] >= float(params["min_band_width_pips"]))
        & (f["band_width_pips"] <= float(params["max_band_width_pips"]))
    )
    buy_blocked = parse_blocked(str(params.get("buy_blocked_hours_utc", "")))
    sell_blocked = parse_blocked(str(params.get("sell_blocked_hours_utc", "")))
    buy_start = int(params.get("buy_start_hour_utc") or start_hour)
    buy_end = int(params.get("buy_end_hour_utc") or end_hour)
    sell_start = int(params.get("sell_start_hour_utc") or start_hour)
    sell_end = int(params.get("sell_end_hour_utc") or end_hour)
    buy_adx_ceiling = float(params.get("buy_adx_ceiling") or params["adx_ceiling"])
    sell_adx_ceiling = float(params.get("sell_adx_ceiling") or params["adx_ceiling"])
    buy_hours = (hours >= buy_start) & (hours <= buy_end) & ~hours.isin(buy_blocked)
    sell_hours = (hours >= sell_start) & (hours <= sell_end) & ~hours.isin(sell_blocked)
    buy_allowed_raw = str(params.get("buy_allowed_hours_utc", "") or "").strip()
    if buy_allowed_raw:
        buy_hours = hours.isin(parse_blocked(buy_allowed_raw))
    sell_allowed_raw = str(params.get("sell_allowed_hours_utc", "") or "").strip()
    if sell_allowed_raw:
        sell_hours = hours.isin(parse_blocked(sell_allowed_raw))
    buy = (
        common
        & buy_hours
        & (f["adx"] <= buy_adx_ceiling)
        & (f["close"] <= f["lower"])
        & (f["rsi"] <= int(params["rsi_oversold"]))
        & (f["distance_to_low_pips"] <= float(params["range_proximity_pips"]))
        & buy_htf
    )
    sell = (
        common
        & sell_hours
        & (f["adx"] <= sell_adx_ceiling)
        & (f["close"] >= f["upper"])
        & (f["rsi"] >= int(params["rsi_overbought"]))
        & (f["distance_to_high_pips"] <= float(params["range_proximity_pips"]))
        & sell_htf
    )
    dirs = pd.Series("", index=f.index)
    dirs[buy] = "BUY"
    dirs[sell] = "SELL"

    allowed_dirs = {part.strip().upper() for part in str(params.get("allowed_directions", "")).split(",") if part.strip()}
    out = f[dirs.ne("")].copy()
    out["direction"] = dirs[dirs.ne("")]
    if allowed_dirs:
        out = out[out["direction"].isin(allowed_dirs)]

    cooldown = int(params["signal_cooldown_minutes"])
    if cooldown <= 0 or out.empty:
        return out
    kept = []
    next_allowed = pd.Timestamp.min
    for ts, row in out.iterrows():
        if ts >= next_allowed:
            kept.append(ts)
            next_allowed = ts + timedelta(minutes=cooldown)
    return out.loc[kept]


def simulate(signals: pd.DataFrame, df15: pd.DataFrame, params: Dict[str, object]) -> Candidate:
    sl = float(params["fixed_stop_loss_pips"])
    tp = float(params["fixed_take_profit_pips"])
    spread_slip = 2.0
    wins = losses = 0
    profit = loss = 0.0

    for ts, sig in signals.iterrows():
        if ts not in df15.index:
            future = df15[df15.index > ts].head(200)
        else:
            loc = df15.index.get_loc(ts)
            future = df15.iloc[loc + 1 : loc + 201]
        if future.empty:
            continue

        is_buy = sig["direction"] == "BUY"
        fill = float(sig["close"]) + spread_slip * PIP if is_buy else float(sig["close"]) - spread_slip * PIP
        stop = fill - sl * PIP if is_buy else fill + sl * PIP
        target = fill + tp * PIP if is_buy else fill - tp * PIP
        pnl: Optional[float] = None
        for _, bar in future.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            if is_buy:
                if low <= stop:
                    pnl = -sl
                    break
                if high >= target:
                    pnl = tp
                    break
            else:
                if high >= stop:
                    pnl = -sl
                    break
                if low <= target:
                    pnl = tp
                    break
        if pnl is None:
            close = float(future.iloc[-1]["close"])
            pnl = (close - fill) / PIP if is_buy else (fill - close) / PIP

        if pnl > 0:
            wins += 1
            profit += pnl
        elif pnl < 0:
            losses += 1
            loss += -pnl

    total = wins + losses
    pf = profit / loss if loss > 0 else (999.0 if profit > 0 else 0.0)
    expectancy = (profit - loss) / total if total else 0.0
    win_rate = wins / total * 100 if total else 0.0
    return Candidate(params, total, wins, losses, profit, loss, pf, expectancy, win_rate)


def grid(focused: bool = False) -> Iterable[Dict[str, object]]:
    base = {
        "monitor_only": "false",
        "range_proximity_pips": 999.0,
        "max_current_range_pips": 999.0,
        "min_band_width_pips": 0.0,
        "max_band_width_pips": 9999.0,
        "range_lookback_bars": 144,
        "adx_ceiling": 999.0,
    }
    rsi_pairs = [(35, 65), (38, 62), (40, 60), (42, 58), (45, 55), (48, 52), (50, 50)]
    sessions = [(0, 23), (6, 18), (7, 20), (6, 20)]
    blocked_hours = ["", "0,2,6,18,20", "0,2,20", "6,18,20"]
    sell_blocked_hours = ["", "6"]
    neutral_options = [False, True]
    cooldowns = [0, 15, 30]
    bb_mults = [1.6, 1.8, 2.0, 2.2]
    rr_pairs = [(7, 10), (8, 10), (8, 12), (9, 12), (10, 14)]

    if focused:
        rsi_pairs = [(40, 60), (42, 58), (45, 55)]
        sessions = [(6, 18), (7, 20), (8, 16)]
        blocked_hours = ["0,2,20", "0,2,6,18,20"]
        sell_blocked_hours = ["", "6"]
        neutral_options = [True]
        cooldowns = [0, 15, 30]
        bb_mults = [1.8, 2.0]
        rr_pairs = [(7, 10), (8, 10), (8, 12)]
        directions = ["", "BUY", "SELL"]
    else:
        directions = [""]

    for rsi_os, rsi_ob in rsi_pairs:
        for start, end in sessions:
            for blocked in blocked_hours:
                for sell_blocked in sell_blocked_hours:
                    for neutral in neutral_options:
                        for cooldown in cooldowns:
                            for bb_mult in bb_mults:
                                for sl, tp in rr_pairs:
                                    for direction in directions:
                                        yield {
                                            **base,
                                            "rsi_oversold": rsi_os,
                                            "rsi_overbought": rsi_ob,
                                            "london_start_hour_utc": start,
                                            "new_york_end_hour_utc": end,
                                            "blocked_hours_utc": blocked,
                                            "sell_blocked_hours_utc": sell_blocked,
                                            "buy_adx_ceiling": 60.0,
                                            "sell_adx_ceiling": 25.0,
                                            "allow_neutral_htf": neutral,
                                            "signal_cooldown_minutes": cooldown,
                                            "bb_mult": bb_mult,
                                            "fixed_stop_loss_pips": sl,
                                            "fixed_take_profit_pips": tp,
                                            "allowed_directions": direction,
                                        }


def score(c: Candidate, target_trades: int) -> tuple:
    if c.signals < target_trades:
        return (-999, c.signals, c.pf)
    return (-abs(c.pf - 2.0), c.expectancy, c.signals)


def dynamic_overrides(params: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in params.items()
        if key in DYNAMIC_OVERRIDE_KEYS and value not in ("", None)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epic", default=DEFAULT_EPIC)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--target-trades", type=int, default=30)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()

    df5, df15, df60 = load_candles(args.epic, args.days)
    target_start = df5.index.max() - timedelta(days=args.days)
    scan_index = df15[df15.index >= target_start].index
    features = prepare_features(df5, df60)

    results: List[Candidate] = []
    configs = list(grid(args.focused))
    print(f"[fast-opt] epic={args.epic} configs={len(configs)} days={args.days} target_trades={args.target_trades}")
    for i, params in enumerate(configs, start=1):
        sigs = signal_rows(features, scan_index, params)
        if len(sigs) < max(5, args.target_trades // 2):
            continue
        res = simulate(sigs, df15, params)
        results.append(res)
        if i % 500 == 0:
            print(f"[fast-opt] scanned {i}/{len(configs)} kept={len(results)}", flush=True)

    results.sort(key=lambda c: score(c, args.target_trades), reverse=True)
    print("\n=== TOP CANDIDATES ===")
    for c in results[: args.top]:
        print(
            f"n={c.signals:3d} pf={c.pf:5.2f} wr={c.win_rate:5.1f}% "
            f"exp={c.expectancy:6.2f} W/L={c.winners}/{c.losers} "
            f"dynamic={dynamic_overrides(c.params)} params={c.params}",
            flush=True,
        )
    print("\n=== BEST PF BY FREQUENCY FLOOR ===")
    for floor in [30, 45, 60, 90, 120]:
        eligible = [c for c in results if c.signals >= floor]
        if not eligible:
            print(f"n>={floor}: none")
            continue
        best = max(eligible, key=lambda c: (c.pf, c.expectancy))
        print(
            f"n>={floor}: n={best.signals} pf={best.pf:.2f} wr={best.win_rate:.1f}% "
            f"exp={best.expectancy:.2f} dynamic={dynamic_overrides(best.params)} params={best.params}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
