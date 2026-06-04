#!/usr/bin/env python3
"""
Intraday EMA Pullback + ADX Momentum backtest for FX.

Converted from the supplied Pine Script strategy:
  - EMA20/EMA50 trend filter
  - DMI/ADX(14), ADX > 20 and rising
  - 20-bar volume SMA filter
  - Pullback touch of EMA20 with close back through EMA20
  - ATR(14) stop = 2.0x ATR, target = 1.5x ATR

Default run tests all configured FX pairs on 15m bars using Dukascopy/IG
backtest candles.

Usage:
  docker exec -it task-worker python /app/tmp_ema_adx_pullback_nzdusd.py
  docker exec -it task-worker python /app/tmp_ema_adx_pullback_nzdusd.py --years 5
  docker exec -it task-worker python /app/tmp_ema_adx_pullback_nzdusd.py --pair NZDUSD
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import psycopg2

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")


@dataclass(frozen=True)
class Config:
    pip: float = 0.0001
    tf_minutes: int = 15
    short_ema_len: int = 20
    long_ema_len: int = 50
    adx_len: int = 14
    adx_min: float = 20.0
    vol_len: int = 20
    atr_len: int = 14
    sl_mult: float = 2.0
    tp_mult: float = 1.5


PAIRS = [
    ("CS.D.EURUSD.CEEM.IP", "EURUSD", 0.0001),
    ("CS.D.GBPUSD.MINI.IP", "GBPUSD", 0.0001),
    ("CS.D.EURJPY.MINI.IP", "EURJPY", 0.01),
    ("CS.D.USDJPY.MINI.IP", "USDJPY", 0.01),
    ("CS.D.AUDUSD.MINI.IP", "AUDUSD", 0.0001),
    ("CS.D.USDCAD.MINI.IP", "USDCAD", 0.0001),
    ("CS.D.USDCHF.MINI.IP", "USDCHF", 0.0001),
    ("CS.D.NZDUSD.MINI.IP", "NZDUSD", 0.0001),
]


def load_1m(epic: str, years: int) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    sql = """
        SELECT start_time, open, high, low, close, volume
        FROM ig_candles_backtest
        WHERE epic = %s AND timeframe = 1
          AND start_time >= NOW() - INTERVAL %s
        ORDER BY start_time
    """
    df = pd.read_sql(sql, conn, params=(epic, f"{years} years"), parse_dates=["start_time"])
    conn.close()
    if df.empty:
        return df
    df = df.set_index("start_time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    return df.resample(f"{minutes}min", closed="left", label="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()


def wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["ema_short"] = out["close"].ewm(span=cfg.short_ema_len, adjust=False).mean()
    out["ema_long"] = out["close"].ewm(span=cfg.long_ema_len, adjust=False).mean()

    prev_high = out["high"].shift(1)
    prev_low = out["low"].shift(1)
    prev_close = out["close"].shift(1)

    up_move = out["high"] - prev_high
    down_move = prev_low - out["low"]
    plus_dm = pd.Series(
        ((up_move > down_move) & (up_move > 0)).to_numpy() * up_move.to_numpy(),
        index=out.index,
    ).fillna(0.0)
    minus_dm = pd.Series(
        ((down_move > up_move) & (down_move > 0)).to_numpy() * down_move.to_numpy(),
        index=out.index,
    ).fillna(0.0)

    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["atr"] = wilder_smooth(tr, cfg.atr_len)
    adx_atr = wilder_smooth(tr, cfg.adx_len)
    out["di_plus"] = 100.0 * wilder_smooth(plus_dm, cfg.adx_len) / adx_atr
    out["di_minus"] = 100.0 * wilder_smooth(minus_dm, cfg.adx_len) / adx_atr
    dx = 100.0 * (out["di_plus"] - out["di_minus"]).abs() / (out["di_plus"] + out["di_minus"])
    out["adx"] = wilder_smooth(dx.replace([math.inf, -math.inf], math.nan).fillna(0.0), cfg.adx_len)
    out["avg_vol"] = out["volume"].rolling(cfg.vol_len).mean()
    return out


def simulate(df: pd.DataFrame, cfg: Config, pip: float) -> list[dict]:
    trades: list[dict] = []
    in_trade = False
    direction: Optional[str] = None
    entry_time: Optional[pd.Timestamp] = None
    entry_index = 0
    entry_price = stop_price = target_price = 0.0

    bars = list(df.itertuples())
    for i, row in enumerate(bars):
        ts = row.Index

        if in_trade:
            exit_price = None
            exit_why = None

            # Conservative intrabar assumption: if stop and target both trade, count stop first.
            if direction == "long":
                if row.low <= stop_price:
                    exit_price, exit_why = stop_price, "SL"
                elif row.high >= target_price:
                    exit_price, exit_why = target_price, "TP"
            else:
                if row.high >= stop_price:
                    exit_price, exit_why = stop_price, "SL"
                elif row.low <= target_price:
                    exit_price, exit_why = target_price, "TP"

            if exit_price is not None:
                pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "direction": direction,
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl_pips": round(pnl / pip, 2),
                        "exit_why": exit_why,
                        "duration_bars": i - entry_index,
                        "year": entry_time.year if entry_time is not None else ts.year,
                    }
                )
                in_trade = False

        if in_trade or i == 0:
            continue

        prev = bars[i - 1]
        required = [row.ema_short, row.ema_long, row.adx, prev.adx, row.avg_vol, row.atr]
        if any(pd.isna(v) for v in required):
            continue

        bullish = row.ema_short > row.ema_long
        bearish = row.ema_short < row.ema_long
        adx_rising = row.adx > prev.adx
        vol_filter = row.volume > row.avg_vol

        long_setup = (
            bullish
            and row.low < row.ema_short
            and row.close > row.ema_short
            and row.adx > cfg.adx_min
            and adx_rising
            and vol_filter
        )
        short_setup = (
            bearish
            and row.high > row.ema_short
            and row.close < row.ema_short
            and row.adx > cfg.adx_min
            and adx_rising
            and vol_filter
        )

        if long_setup:
            in_trade = True
            direction = "long"
            entry_time = ts
            entry_index = i
            entry_price = row.close
            stop_price = row.close - row.atr * cfg.sl_mult
            target_price = row.close + row.atr * cfg.tp_mult
        elif short_setup:
            in_trade = True
            direction = "short"
            entry_time = ts
            entry_index = i
            entry_price = row.close
            stop_price = row.close + row.atr * cfg.sl_mult
            target_price = row.close - row.atr * cfg.tp_mult

    if in_trade and bars:
        last = bars[-1]
        pnl = (last.close - entry_price) if direction == "long" else (entry_price - last.close)
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": last.Index,
                "direction": direction,
                "entry": entry_price,
                "exit": last.close,
                "pnl_pips": round(pnl / pip, 2),
                "exit_why": "END",
                "duration_bars": len(bars) - entry_index,
                "year": entry_time.year if entry_time is not None else last.Index.year,
            }
        )

    return trades


def stats_line(tdf: pd.DataFrame) -> str:
    if tdf.empty:
        return "n=0"
    wins = tdf[tdf["pnl_pips"] > 0]
    losses = tdf[tdf["pnl_pips"] <= 0]
    gross_loss = losses["pnl_pips"].abs().sum()
    pf = "inf" if gross_loss == 0 else f"{wins['pnl_pips'].sum() / gross_loss:.2f}"
    return (
        f"n={len(tdf):4d}  WR={len(wins) / len(tdf) * 100:5.1f}%  PF={pf:>5s}  "
        f"Avg={tdf['pnl_pips'].mean():+6.2f}  Net={tdf['pnl_pips'].sum():+8.1f} pips"
    )


def print_pair_results(pair: str, df: pd.DataFrame, trades: list[dict], cfg: Config, years: int) -> None:
    print("\n" + "=" * 92)
    print("  Intraday EMA Pullback + ADX Momentum")
    print(f"  {pair} {cfg.tf_minutes}m / {years}y / bars={len(df):,}")
    print(
        f"  EMA {cfg.short_ema_len}/{cfg.long_ema_len}, ADX>{cfg.adx_min:g} rising, "
        f"Vol>SMA{cfg.vol_len}, ATR SL/TP={cfg.sl_mult:g}x/{cfg.tp_mult:g}x"
    )
    print("=" * 92)

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("\n  NO TRADES\n")
        return

    print(f"\n  Overall: {stats_line(tdf)}")

    print("\n  Direction:")
    for direction in ["long", "short"]:
        print(f"    {direction:5s}: {stats_line(tdf[tdf['direction'] == direction])}")

    print("\n  Exit:")
    for exit_why, sub in tdf.groupby("exit_why"):
        print(f"    {exit_why:5s}: {stats_line(sub)}")

    print("\n  Year:")
    for year, sub in tdf.groupby("year"):
        print(f"    {year}: {stats_line(sub)}")

    print("\n  First 10 trades:")
    cols = ["entry_time", "exit_time", "direction", "entry", "exit", "pnl_pips", "exit_why"]
    print(tdf[cols].head(10).to_string(index=False))
    print()


def print_summary(results: dict[str, list[dict]], years: int) -> None:
    print("\n" + "=" * 92)
    print(f"  Summary - all epics / {years}y")
    print("=" * 92)
    print(f"  {'Pair':8s}  {'Trades':>6s}  {'WR':>7s}  {'PF':>6s}  {'Avg':>9s}  {'Net':>11s}")
    combined_frames = []
    for pair, trades in results.items():
        tdf = pd.DataFrame(trades)
        if tdf.empty:
            print(f"  {pair:8s}  {0:6d}  {'-':>7s}  {'-':>6s}  {'-':>9s}  {'-':>11s}")
            continue
        wins = tdf[tdf["pnl_pips"] > 0]
        losses = tdf[tdf["pnl_pips"] <= 0]
        gross_loss = losses["pnl_pips"].abs().sum()
        pf = "inf" if gross_loss == 0 else f"{wins['pnl_pips'].sum() / gross_loss:.2f}"
        print(
            f"  {pair:8s}  {len(tdf):6d}  {len(wins) / len(tdf) * 100:6.1f}%  "
            f"{pf:>6s}  {tdf['pnl_pips'].mean():+8.2f}  {tdf['pnl_pips'].sum():+10.1f}"
        )
        combined_frames.append(tdf.assign(pair=pair))

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        print(f"\n  Combined: {stats_line(combined)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="EMA pullback + ADX momentum FX backtest")
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--pair", type=str, default=None, help="Optional pair filter, e.g. NZDUSD")
    parser.add_argument("--details", action="store_true", help="Print detailed per-pair sections")
    args = parser.parse_args()

    cfg = Config()
    pairs = PAIRS
    if args.pair:
        wanted = args.pair.upper()
        pairs = [p for p in PAIRS if p[1] == wanted]
        if not pairs:
            raise SystemExit(f"Unknown pair {args.pair}. Known: {', '.join(p[1] for p in PAIRS)}")

    results: dict[str, list[dict]] = {}
    for epic, pair, pip in pairs:
        print(f"Loading {pair}...")
        df_1m = load_1m(epic, args.years)
        if df_1m.empty:
            print(f"  No 1m data found for {epic}")
            results[pair] = []
            continue
        df_15m = add_indicators(resample_ohlcv(df_1m, cfg.tf_minutes), cfg)
        trades = simulate(df_15m, cfg, pip)
        results[pair] = trades
        if args.details or len(pairs) == 1:
            print_pair_results(pair, df_15m, trades, cfg, args.years)

    print_summary(results, args.years)


if __name__ == "__main__":
    main()
