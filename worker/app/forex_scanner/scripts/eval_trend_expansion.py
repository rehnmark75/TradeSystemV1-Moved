#!/usr/bin/env python3
"""Standalone trend-expansion strategy evaluator.

This is a POC script, not a production strategy. It pulls candles directly from
`ig_candles_backtest`, computes a simple trend/momentum breakout thesis, and
simulates first-hit SL/TP outcomes.

Thesis:
  - 4H trend bias is clear: close is above/below EMA50 and EMA50 slope agrees.
  - 1H regime is directional: ADX is above a configurable floor.
  - 15m entry candle closes through a recent N-bar high/low.
  - Entry candle has enough body-vs-ATR expansion, but is not too extended.

Usage:
  docker exec -it task-worker python /app/forex_scanner/scripts/eval_trend_expansion.py
  docker exec -it task-worker python /app/forex_scanner/scripts/eval_trend_expansion.py --days 180
  docker exec -it task-worker python /app/forex_scanner/scripts/eval_trend_expansion.py --pair CS.D.EURJPY.MINI.IP --show-trades
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2


FOREX_DB = os.getenv("FOREX_DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)

PAIRS = [
    "CS.D.AUDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP",
    "CS.D.EURJPY.MINI.IP",
    "CS.D.EURUSD.CEEM.IP",
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.NZDUSD.MINI.IP",
    "CS.D.USDCAD.MINI.IP",
    "CS.D.USDCHF.MINI.IP",
    "CS.D.USDJPY.MINI.IP",
]


@dataclass(frozen=True)
class TradeConfig:
    sl_pips: float
    tp_pips: float


def pair_label(epic: str) -> str:
    parts = epic.split(".")
    return parts[2] if len(parts) >= 3 else epic


def pip_size(epic: str) -> float:
    return 0.01 if "JPY" in epic.upper() else 0.0001


def default_trade_config(epic: str, sl: Optional[float], tp: Optional[float]) -> TradeConfig:
    if sl is not None and tp is not None:
        return TradeConfig(sl_pips=sl, tp_pips=tp)
    if "JPY" in epic.upper():
        return TradeConfig(sl_pips=14.0 if sl is None else sl, tp_pips=24.0 if tp is None else tp)
    return TradeConfig(sl_pips=10.0 if sl is None else sl, tp_pips=18.0 if tp is None else tp)


def default_cost_pips(epic: str, explicit_cost: Optional[float]) -> float:
    if explicit_cost is not None:
        return explicit_cost
    return 2.0 if "JPY" in epic.upper() else 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--pair", default="", help="Full IG epic. Omit to run the standard 9 FX pairs.")
    parser.add_argument("--breakout-bars", type=int, default=20, help="15m lookback high/low window.")
    parser.add_argument("--adx-min", type=float, default=22.0, help="Minimum 1H ADX.")
    parser.add_argument("--adx-rising-bars", type=int, default=3, help="Require 1H ADX now >= value N bars ago. 0 disables.")
    parser.add_argument("--htf-ema", type=int, default=50, help="4H EMA period for trend bias.")
    parser.add_argument("--htf-slope-bars", type=int, default=6, help="4H EMA slope lookback.")
    parser.add_argument("--htf-min-slope-pips", type=float, default=4.0, help="Minimum EMA slope over slope lookback.")
    parser.add_argument("--body-atr-min", type=float, default=0.65, help="Min 15m candle body / ATR14.")
    parser.add_argument("--max-extension-atr", type=float, default=2.2, help="Max 15m close distance from 1H EMA50 in ATR units.")
    parser.add_argument("--session-start", type=int, default=6, help="UTC hour inclusive.")
    parser.add_argument("--session-end", type=int, default=20, help="UTC hour inclusive.")
    parser.add_argument("--direction", choices=["BOTH", "BUY", "SELL"], default="BOTH")
    parser.add_argument("--sl-pips", type=float, default=None)
    parser.add_argument("--tp-pips", type=float, default=None)
    parser.add_argument("--cost-pips", type=float, default=None, help="Round-trip cost/slippage. Defaults: 1.0 majors, 2.0 JPY.")
    parser.add_argument("--cooldown-bars", type=int, default=8, help="15m bars to skip after a closed trade.")
    parser.add_argument("--max-hold-bars", type=int, default=64, help="15m bars before timeout exit.")
    parser.add_argument("--show-trades", action="store_true")
    return parser.parse_args()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def ema_wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    alpha = 1.0 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean()


def resample(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df5.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()


def latest_backtest_time(conn) -> pd.Timestamp:
    q = "SELECT MAX(start_time) FROM ig_candles_backtest WHERE timeframe = 5"
    return pd.Timestamp(pd.read_sql(q, conn).iloc[0, 0])


def load_5m(conn, epic: str, days: int, end: pd.Timestamp, warmup_days: int = 35) -> pd.DataFrame:
    start = end - timedelta(days=days + warmup_days)
    q = """
        SELECT start_time, open, high, low, close
          FROM ig_candles_backtest
         WHERE epic = %s
           AND timeframe = 5
           AND start_time >= %s
           AND start_time <= %s
         ORDER BY start_time
    """
    df = pd.read_sql(q, conn, params=(epic, start, end))
    if df.empty:
        return df
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df.set_index("start_time").sort_index()


def build_signal_frame(df5: pd.DataFrame, epic: str, args: argparse.Namespace) -> pd.DataFrame:
    pip = pip_size(epic)
    df15 = resample(df5, "15min")
    df1h = resample(df5, "1h")
    df4h = resample(df5, "4h")

    out = df15.copy()
    out["atr15"] = atr(out, 14)
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_atr"] = out["body"] / out["atr15"].replace(0, np.nan)
    out["prior_high"] = out["high"].shift(1).rolling(args.breakout_bars).max()
    out["prior_low"] = out["low"].shift(1).rolling(args.breakout_bars).min()

    # Shift HTF features by one completed bar before joining to 15m rows.
    # Resampled bars are labeled at their open, so unshifted 1H/4H values
    # would leak the final state of the still-forming candle.
    adx_1h = ema_wilder_adx(df1h).shift(1)
    ema_1h = ema(df1h["close"], 50).shift(1)
    atr_1h = atr(df1h, 14).shift(1)
    out["adx_1h"] = adx_1h.reindex(out.index, method="ffill")
    out["adx_1h_prev"] = adx_1h.shift(args.adx_rising_bars).reindex(out.index, method="ffill")
    out["ema_1h"] = ema_1h.reindex(out.index, method="ffill")
    out["atr_1h"] = atr_1h.reindex(out.index, method="ffill")

    ema_4h_raw = ema(df4h["close"], args.htf_ema)
    htf = pd.DataFrame(index=df4h.index)
    htf["ema_4h"] = ema_4h_raw.shift(1)
    htf["close_4h"] = df4h["close"].shift(1)
    htf["ema_slope_pips"] = (
        ema_4h_raw.shift(1) - ema_4h_raw.shift(args.htf_slope_bars + 1)
    ) / pip
    htf["bias"] = ""
    htf.loc[
        (htf["close_4h"] > htf["ema_4h"]) & (htf["ema_slope_pips"] >= args.htf_min_slope_pips),
        "bias",
    ] = "BULL"
    htf.loc[
        (htf["close_4h"] < htf["ema_4h"]) & (htf["ema_slope_pips"] <= -args.htf_min_slope_pips),
        "bias",
    ] = "BEAR"
    out["htf_bias"] = htf["bias"].reindex(out.index, method="ffill")
    out["htf_slope_pips"] = htf["ema_slope_pips"].reindex(out.index, method="ffill")

    hour_ok = (out.index.hour >= args.session_start) & (out.index.hour <= args.session_end)
    adx_ok = out["adx_1h"] >= args.adx_min
    if args.adx_rising_bars > 0:
        adx_ok = adx_ok & (out["adx_1h"] >= out["adx_1h_prev"])

    extension_atr = (out["close"] - out["ema_1h"]).abs() / out["atr_1h"].replace(0, np.nan)
    out["extension_atr"] = extension_atr
    quality_ok = (out["body_atr"] >= args.body_atr_min) & (extension_atr <= args.max_extension_atr)

    bull_break = (
        hour_ok
        & adx_ok
        & quality_ok
        & (out["htf_bias"] == "BULL")
        & (out["close"] > out["prior_high"])
        & (out["close"] > out["open"])
    )
    bear_break = (
        hour_ok
        & adx_ok
        & quality_ok
        & (out["htf_bias"] == "BEAR")
        & (out["close"] < out["prior_low"])
        & (out["close"] < out["open"])
    )

    out["signal"] = ""
    out.loc[bull_break, "signal"] = "BUY"
    out.loc[bear_break, "signal"] = "SELL"
    if args.direction == "BUY":
        out.loc[out["signal"] == "SELL", "signal"] = ""
    elif args.direction == "SELL":
        out.loc[out["signal"] == "BUY", "signal"] = ""
    return out


def simulate_trades(
    signals: pd.DataFrame,
    df5: pd.DataFrame,
    epic: str,
    cfg: TradeConfig,
    args: argparse.Namespace,
) -> pd.DataFrame:
    pip = pip_size(epic)
    cost_pips = default_cost_pips(epic, args.cost_pips)
    trades: List[Dict] = []
    cooldown_until = pd.Timestamp.min
    max_hold = timedelta(minutes=15 * args.max_hold_bars)

    signal_rows = signals[signals["signal"] != ""]
    for signal_ts, row in signal_rows.iterrows():
        entry_time = signal_ts + timedelta(minutes=15)
        if entry_time <= cooldown_until:
            continue

        side = row["signal"]
        path = df5[(df5.index >= entry_time) & (df5.index <= entry_time + max_hold)]
        if path.empty:
            continue

        first_candle = path.iloc[0]
        raw_entry = float(first_candle["open"])
        entry = raw_entry + cost_pips * pip if side == "BUY" else raw_entry - cost_pips * pip
        if side == "BUY":
            sl = entry - cfg.sl_pips * pip
            tp = entry + cfg.tp_pips * pip
        else:
            sl = entry + cfg.sl_pips * pip
            tp = entry - cfg.tp_pips * pip

        outcome = "TIMEOUT"
        exit_time = path.index[-1]
        exit_price = float(path["close"].iloc[-1])
        for ts, candle in path.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            if side == "BUY":
                if low <= sl:
                    outcome = "SL"
                    exit_time = ts
                    exit_price = sl
                    break
                if high >= tp:
                    outcome = "TP"
                    exit_time = ts
                    exit_price = tp
                    break
            else:
                if high >= sl:
                    outcome = "SL"
                    exit_time = ts
                    exit_price = sl
                    break
                if low <= tp:
                    outcome = "TP"
                    exit_time = ts
                    exit_price = tp
                    break

        direction = 1 if side == "BUY" else -1
        pips = (exit_price - entry) * direction / pip
        trades.append(
            {
                "pair": pair_label(epic),
                "entry_ts": entry_time,
                "exit_ts": exit_time,
                "side": side,
                "entry": entry,
                "raw_entry": raw_entry,
                "exit": exit_price,
                "pips": pips,
                "cost_pips": cost_pips,
                "outcome": outcome,
                "bars_held_5m": int((exit_time - entry_time).total_seconds() // 300),
                "adx_1h": float(row["adx_1h"]),
                "body_atr": float(row["body_atr"]),
                "extension_atr": float(row["extension_atr"]),
                "htf_slope_pips": float(row["htf_slope_pips"]),
            }
        )
        cooldown_until = exit_time + timedelta(minutes=15 * args.cooldown_bars)

    return pd.DataFrame(trades)


def aggregate(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "n": 0,
            "wr": np.nan,
            "pf": np.nan,
            "expectancy": 0.0,
            "total_pips": 0.0,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
        }
    wins = trades[trades["pips"] > 0]
    losses = trades[trades["pips"] < 0]
    pf = wins["pips"].sum() / -losses["pips"].sum() if not losses.empty else float("inf")
    return {
        "n": int(len(trades)),
        "wr": float((trades["pips"] > 0).mean()),
        "pf": float(pf),
        "expectancy": float(trades["pips"].mean()),
        "total_pips": float(trades["pips"].sum()),
        "avg_win": float(wins["pips"].mean()) if not wins.empty else np.nan,
        "avg_loss": float(losses["pips"].mean()) if not losses.empty else np.nan,
        "tp": int((trades["outcome"] == "TP").sum()),
        "sl": int((trades["outcome"] == "SL").sum()),
        "timeout": int((trades["outcome"] == "TIMEOUT").sum()),
    }


def main() -> int:
    args = parse_args()
    pairs = [args.pair] if args.pair else PAIRS

    print(
        "[trend_expansion] "
        f"days={args.days} breakout={args.breakout_bars}x15m "
        f"adx_min={args.adx_min} body_atr_min={args.body_atr_min} "
        f"max_ext_atr={args.max_extension_atr} session={args.session_start}-{args.session_end}UTC "
        f"direction={args.direction} cost={'auto' if args.cost_pips is None else args.cost_pips}",
        flush=True,
    )

    rows = []
    all_trades: List[pd.DataFrame] = []

    with psycopg2.connect(FOREX_DB) as conn:
        end = latest_backtest_time(conn)
        for epic in pairs:
            cfg = default_trade_config(epic, args.sl_pips, args.tp_pips)
            df5 = load_5m(conn, epic, args.days, end)
            if df5.empty:
                print(f"[trend_expansion] {epic}: no 5m backtest candles", flush=True)
                continue

            signals = build_signal_frame(df5, epic, args)
            cutoff = signals.index.max() - timedelta(days=args.days)
            signals = signals[signals.index >= cutoff]
            trades = simulate_trades(signals, df5, epic, cfg, args)
            stats = aggregate(trades)
            stats["pair"] = pair_label(epic)
            stats["sl_tp"] = f"{cfg.sl_pips:g}/{cfg.tp_pips:g}"
            stats["raw_signals"] = int((signals["signal"] != "").sum())
            rows.append(stats)
            if not trades.empty:
                trades["epic"] = epic
                all_trades.append(trades)

    if not rows:
        print("[trend_expansion] no results")
        return 1

    result = pd.DataFrame(rows)[
        ["pair", "raw_signals", "n", "wr", "pf", "expectancy", "total_pips", "avg_win", "avg_loss", "tp", "sl", "timeout", "sl_tp"]
    ]
    print()
    print(result.to_string(index=False, float_format="{:.3f}".format), flush=True)

    if all_trades:
        portfolio = pd.concat(all_trades, ignore_index=True)
        stats = aggregate(portfolio)
        print(
            "\n[PORTFOLIO] "
            f"n={stats['n']} WR={stats['wr']:.2%} PF={stats['pf']:.3f} "
            f"exp={stats['expectancy']:.2f}p total={stats['total_pips']:.1f}p "
            f"avg_win={stats['avg_win']:.1f}p avg_loss={stats['avg_loss']:.1f}p "
            f"TP/SL/TO={stats['tp']}/{stats['sl']}/{stats['timeout']}",
            flush=True,
        )
        if args.show_trades:
            print()
            cols = ["pair", "entry_ts", "side", "entry", "exit_ts", "outcome", "pips", "adx_1h", "body_atr", "extension_atr"]
            print(portfolio[cols].tail(80).to_string(index=False, float_format="{:.3f}".format), flush=True)
    else:
        print("\n[PORTFOLIO] no trades after cooldown/path simulation", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
