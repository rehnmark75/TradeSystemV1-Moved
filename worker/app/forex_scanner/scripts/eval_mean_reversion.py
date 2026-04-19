#!/usr/bin/env python3
"""Standalone mean-reversion strategy evaluator.

Bypasses the production backtest_cli entirely — pulls raw 5m candles from
ig_candles_backtest, computes BB/RSI/ADX directly, simulates trades with
fixed SL/TP. The point is to get PF/WR for the mean-reversion *thesis*
quickly so we know whether to invest in production-wiring it. If the PF
is dead even in this clean simulation, there's no point reviving the
archived MeanReversionStrategy + 3 helper modules + config + tables.

Signal logic (deliberately simple, mean-reversion classic):
  BUY  when:   close <= lower_BB AND RSI <= rsi_oversold AND
               1h_ADX <= htf_max  AND 15m_ADX <= primary_max
  SELL when:   close >= upper_BB AND RSI >= rsi_overbought AND
               1h_ADX <= htf_max  AND 15m_ADX <= primary_max

Trade simulation (cooldown-aware, no overlap):
  - Entry at signal candle's close
  - Fixed SL/TP in pips, first hit wins (using subsequent candle highs/lows)
  - Cooldown of N bars after entry before next signal can fire
  - PnL in currency (per unit lot) using pip values per pair

Usage:
  docker exec -it task-worker python /app/forex_scanner/scripts/eval_mean_reversion.py
  docker exec -it task-worker python /app/forex_scanner/scripts/eval_mean_reversion.py \
      --days 60 --primary-adx 22 --htf-adx 25 --rsi-os 30 --rsi-ob 70 \
      --sl-pips 15 --tp-pips 25 --cooldown-bars 10
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2


FOREX_DB = "postgresql://postgres:postgres@postgres:5432/forex"

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


def pip_size(epic: str) -> float:
    return 0.01 if "JPY" in epic else 0.0001


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--primary-adx", type=float, default=22.0,
                   help="Max 15m ADX (hard gate)")
    p.add_argument("--htf-adx", type=float, default=25.0,
                   help="Max 1h ADX (hard gate / HTF alignment)")
    p.add_argument("--rsi-period", type=int, default=14)
    p.add_argument("--rsi-os", type=float, default=30.0)
    p.add_argument("--rsi-ob", type=float, default=70.0)
    p.add_argument("--bb-period", type=int, default=20)
    p.add_argument("--bb-mult", type=float, default=2.0)
    p.add_argument("--sl-pips", type=float, default=15.0)
    p.add_argument("--tp-pips", type=float, default=25.0)
    p.add_argument("--cooldown-bars", type=int, default=10,
                   help="15m bars to skip after each entry")
    p.add_argument("--lot-size", type=float, default=1.0,
                   help="Per-trade size for PnL aggregation (£/pip-equivalent)")
    p.add_argument("--pair", default="", help="Run only one pair (epic full)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Indicators (EMA-Wilder ADX matches DataFetcher; standard BB & RSI)
# ---------------------------------------------------------------------------

def ema_wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = 1.0 / period
    atr = tr.ewm(alpha=a, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window=period).mean()
    sd = close.rolling(window=period).std()
    return ma + mult * sd, ma, ma - mult * sd


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_candles(conn, epic: str, days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_5m, df_15m, df_1h) covering the last `days` days plus warmup."""
    end = datetime.now()
    start = end - timedelta(days=days + 5)  # warmup buffer
    q = """
        SELECT start_time, open, high, low, close
          FROM ig_candles_backtest
         WHERE epic = %s AND timeframe = 5
           AND start_time >= %s AND start_time <= %s
         ORDER BY start_time
    """
    df5 = pd.read_sql(q, conn, params=(epic, start, end))
    if df5.empty:
        return df5, pd.DataFrame(), pd.DataFrame()
    df5["start_time"] = pd.to_datetime(df5["start_time"])
    df5 = df5.set_index("start_time")
    df15 = df5.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    df1h = df5.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    return df5, df15, df1h


# ---------------------------------------------------------------------------
# Signal generation + trade simulation
# ---------------------------------------------------------------------------

def generate_signals(df15: pd.DataFrame, df1h: pd.DataFrame, args) -> pd.DataFrame:
    """Tag every 15m bar with BUY/SELL/None based on the strategy rules."""
    out = df15.copy()
    out["adx_15m"] = ema_wilder_adx(out)
    out["rsi"] = rsi(out["close"], args.rsi_period)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(
        out["close"], args.bb_period, args.bb_mult
    )

    adx_1h = ema_wilder_adx(df1h)
    out["adx_1h"] = adx_1h.reindex(out.index, method="ffill")

    htf_ok = (out["adx_1h"].notna()) & (out["adx_1h"] <= args.htf_adx)
    pri_ok = (out["adx_15m"].notna()) & (out["adx_15m"] <= args.primary_adx)
    gate = htf_ok & pri_ok

    buy = gate & (out["close"] <= out["bb_lower"]) & (out["rsi"] <= args.rsi_os)
    sell = gate & (out["close"] >= out["bb_upper"]) & (out["rsi"] >= args.rsi_ob)

    out["signal"] = ""
    out.loc[buy, "signal"] = "BUY"
    out.loc[sell, "signal"] = "SELL"
    return out


def simulate(out: pd.DataFrame, sl_pips: float, tp_pips: float,
             cooldown_bars: int, pip: float, lot_size: float) -> pd.DataFrame:
    """Walk forward through the signal-tagged frame. Each entry is closed by
    the first SL or TP hit on subsequent candle highs/lows. Cooldown skips
    `cooldown_bars` bars after an entry before re-arming."""
    trades: List[Dict] = []
    n = len(out)
    i = 0
    sig_arr = out["signal"].values
    close_arr = out["close"].values
    high_arr = out["high"].values
    low_arr = out["low"].values
    ts = out.index

    while i < n:
        s = sig_arr[i]
        if not s:
            i += 1
            continue

        entry_idx = i
        entry_price = float(close_arr[i])

        if s == "BUY":
            sl_price = entry_price - sl_pips * pip
            tp_price = entry_price + tp_pips * pip
        else:
            sl_price = entry_price + sl_pips * pip
            tp_price = entry_price - tp_pips * pip

        outcome = "OPEN"
        exit_idx = None
        exit_price = entry_price
        for j in range(i + 1, min(i + 1 + 96, n)):  # max 96 bars (24h) before timeout
            hi = float(high_arr[j])
            lo = float(low_arr[j])
            if s == "BUY":
                if lo <= sl_price:
                    outcome = "SL"
                    exit_idx = j
                    exit_price = sl_price
                    break
                if hi >= tp_price:
                    outcome = "TP"
                    exit_idx = j
                    exit_price = tp_price
                    break
            else:
                if hi >= sl_price:
                    outcome = "SL"
                    exit_idx = j
                    exit_price = sl_price
                    break
                if lo <= tp_price:
                    outcome = "TP"
                    exit_idx = j
                    exit_price = tp_price
                    break

        if exit_idx is None:
            outcome = "TIMEOUT"
            exit_idx = min(i + 96, n - 1)
            exit_price = float(close_arr[exit_idx])

        sign = 1 if s == "BUY" else -1
        pips_gained = (exit_price - entry_price) * sign / pip
        pnl = pips_gained * lot_size  # currency-units-per-pip × lot_size

        trades.append({
            "entry_ts": ts[entry_idx],
            "exit_ts": ts[exit_idx],
            "signal": s,
            "entry": entry_price,
            "exit": exit_price,
            "pips": pips_gained,
            "pnl": pnl,
            "outcome": outcome,
            "bars_held": exit_idx - entry_idx,
        })
        # Move past the exit + cooldown
        i = max(exit_idx, i) + cooldown_bars + 1

    return pd.DataFrame(trades)


def aggregate(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {"n": 0, "wr": np.nan, "pf": np.nan,
                "total_pips": 0.0, "total_pnl": 0.0,
                "avg_win": np.nan, "avg_loss": np.nan,
                "sl_count": 0, "tp_count": 0, "timeout_count": 0}
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    pf = wins["pnl"].sum() / -losses["pnl"].sum() if not losses.empty else float("inf")
    return {
        "n": len(trades),
        "wr": (trades["pnl"] > 0).mean(),
        "pf": pf,
        "total_pips": trades["pips"].sum(),
        "total_pnl": trades["pnl"].sum(),
        "avg_win": wins["pips"].mean() if not wins.empty else np.nan,
        "avg_loss": losses["pips"].mean() if not losses.empty else np.nan,
        "sl_count": int((trades["outcome"] == "SL").sum()),
        "tp_count": int((trades["outcome"] == "TP").sum()),
        "timeout_count": int((trades["outcome"] == "TIMEOUT").sum()),
    }


def main() -> int:
    args = parse_args()
    print(f"[eval_mr] days={args.days} primary_adx={args.primary_adx} htf_adx={args.htf_adx}")
    print(f"[eval_mr] rsi_os={args.rsi_os} rsi_ob={args.rsi_ob} "
          f"bb={args.bb_period}/{args.bb_mult}σ sl/tp={args.sl_pips}/{args.tp_pips}p "
          f"cooldown={args.cooldown_bars} bars")

    conn = psycopg2.connect(FOREX_DB)
    pairs = [args.pair] if args.pair else PAIRS

    overall_trades: List[pd.DataFrame] = []
    rows = []
    for epic in pairs:
        df5, df15, df1h = load_candles(conn, epic, args.days)
        if df15.empty:
            print(f"[eval_mr] {epic}: no data")
            continue
        signals = generate_signals(df15, df1h, args)
        # Limit simulation window to last `days` (we loaded extra warmup)
        cutoff = signals.index.max() - pd.Timedelta(days=args.days)
        sim_df = signals[signals.index >= cutoff]
        trades = simulate(
            sim_df,
            sl_pips=args.sl_pips,
            tp_pips=args.tp_pips,
            cooldown_bars=args.cooldown_bars,
            pip=pip_size(epic),
            lot_size=args.lot_size,
        )
        agg = aggregate(trades)
        agg["pair"] = epic.split(".")[2]
        rows.append(agg)
        if not trades.empty:
            trades["epic"] = epic
            overall_trades.append(trades)

    if not rows:
        print("[eval_mr] no results")
        return 1

    out = pd.DataFrame(rows)[["pair", "n", "wr", "pf", "total_pips", "total_pnl",
                               "avg_win", "avg_loss", "sl_count", "tp_count", "timeout_count"]]
    print()
    print(out.to_string(index=False, float_format="{:.3f}".format))

    if overall_trades:
        all_trades = pd.concat(overall_trades, ignore_index=True)
        agg = aggregate(all_trades)
        print()
        print(f"[PORTFOLIO] n={agg['n']} WR={agg['wr']:.2%} "
              f"PF={agg['pf']:.3f} total_pips={agg['total_pips']:.1f} "
              f"total_pnl={agg['total_pnl']:.2f} "
              f"avg_win={agg['avg_win']:.1f}p avg_loss={agg['avg_loss']:.1f}p")

    return 0


if __name__ == "__main__":
    sys.exit(main())
