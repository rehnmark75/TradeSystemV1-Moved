#!/usr/bin/env python3
"""
EURJPY active-entry-model combination test.

This is a standalone research script. It does not write DB rows.

Run inside task-worker:
    python /app/eurjpy_entry_model_combo_test.py

Purpose:
    Compare simplified entry archetypes from the active strategy set on
    EURJPY, then test confluence combinations built from those entries.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2


DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")
EPIC = "CS.D.EURJPY.MINI.IP"
PIP = 0.01
COST_PIPS = 1.2

WINDOW_START = "2020-01-01"
WINDOW_END = "2026-06-12"
IS_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
OOS_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
OOS_MID = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")


@dataclass(frozen=True)
class SignalSet:
    name: str
    family: str
    timeframe: int
    signals: pd.DataFrame
    sl_pips: float
    tp_pips: float
    hold_bars_5m: int


def load_candles(timeframe: int) -> pd.DataFrame:
    sql = """
        SELECT start_time,
               open::float8 AS open,
               high::float8 AS high,
               low::float8 AS low,
               close::float8 AS close,
               COALESCE(volume, 0)::float8 AS volume
          FROM ig_candles_backtest
         WHERE epic = %s
           AND timeframe = %s
           AND start_time >= %s
           AND start_time <= %s
         ORDER BY start_time
    """
    with psycopg2.connect(DB_URL) as conn:
        df = pd.read_sql(sql, conn, params=(EPIC, timeframe, WINDOW_START, WINDOW_END), parse_dates=["start_time"])
    df = df.drop_duplicates("start_time").set_index("start_time").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


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


def kama(close: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30) -> tuple[pd.Series, pd.Series]:
    change = (close - close.shift(er_period)).abs()
    volatility = close.diff().abs().rolling(er_period).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = pd.Series(index=close.index, dtype=float)
    if close.empty:
        return out, er
    out.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        out.iloc[i] = out.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - out.iloc[i - 1])
    return out, er


def pack(
    index: pd.DatetimeIndex,
    mask_long: pd.Series | np.ndarray,
    mask_short: pd.Series | np.ndarray,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    long_arr = np.asarray(mask_long, dtype=bool)
    short_arr = np.asarray(mask_short, dtype=bool)
    side = np.zeros(len(index), dtype=np.int8)
    side[long_arr] = 1
    side[short_arr] = -1
    rows = np.flatnonzero(side)
    signal_times = index[rows] + pd.to_timedelta(timeframe_minutes, unit="m")
    return pd.DataFrame({"time": signal_times, "side": side[rows]}).drop_duplicates("time").reset_index(drop=True)


def htf_bias_at(signal_times: pd.Series, df_1h: pd.DataFrame) -> pd.Series:
    c = df_1h["close"]
    e = ema(c, 50)
    slope = e.diff(6)
    bias = pd.Series(0, index=df_1h.index, dtype=np.int8)
    bias[(c > e) & (slope > 0)] = 1
    bias[(c < e) & (slope < 0)] = -1
    return bias.reindex(pd.to_datetime(signal_times, utc=True), method="ffill").fillna(0).astype(np.int8)


def make_active_entry_models(df5: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame) -> list[SignalSet]:
    models: list[SignalSet] = []

    # MEAN_REVERSION: 15m BB + RSI touch and rejection entries with hard-ish ADX ceilings.
    c15 = df15["close"]
    ma15 = c15.rolling(20).mean()
    sd15 = c15.rolling(20).std()
    upper15 = ma15 + 2.0 * sd15
    lower15 = ma15 - 2.0 * sd15
    r15 = rsi(c15, 14)
    ax15 = adx(df15, 14)
    ax1h = adx(df1h, 14).reindex(df15.index, method="ffill")
    low_adx = (ax15 <= 24) & (ax1h <= 24)
    mr_touch = pack(df15.index, (c15 <= lower15) & (r15 <= 30) & low_adx, (c15 >= upper15) & (r15 >= 70) & low_adx, 15)
    mr_reject = pack(
        df15.index,
        (c15.shift(1) <= lower15.shift(1)) & (r15.shift(1) <= 30) & (c15 > lower15) & low_adx,
        (c15.shift(1) >= upper15.shift(1)) & (r15.shift(1) >= 70) & (c15 < upper15) & low_adx,
        15,
    )
    models.append(SignalSet("MEAN_REVERSION_touch", "mean_reversion", 15, mr_touch, 18, 24, 96))
    models.append(SignalSet("MEAN_REVERSION_reject", "mean_reversion", 15, mr_reject, 18, 24, 96))

    # RANGE_FADE: 5m BB + RSI extreme near prior local range, aligned to 1h EMA bias.
    c5 = df5["close"]
    ma5 = c5.rolling(20).mean()
    sd5 = c5.rolling(20).std()
    upper5 = ma5 + 2.0 * sd5
    lower5 = ma5 - 2.0 * sd5
    r5 = rsi(c5, 14)
    prior_hi = df5["high"].rolling(48).max().shift(1)
    prior_lo = df5["low"].rolling(48).min().shift(1)
    dist_low = (c5 - prior_lo) / PIP
    dist_high = (prior_hi - c5) / PIP
    bw = (upper5 - lower5) / PIP
    htf = htf_bias_at(pd.Series(df5.index), df1h).to_numpy()
    rf_long = (c5 <= lower5) & (r5 <= 32) & (dist_low <= 8) & (bw >= 8) & (bw <= 80) & (htf >= 0)
    rf_short = (c5 >= upper5) & (r5 >= 68) & (dist_high <= 8) & (bw >= 8) & (bw <= 80) & (htf <= 0)
    models.append(SignalSet("RANGE_FADE_local_extreme", "range_fade", 5, pack(df5.index, rf_long, rf_short, 5), 14, 18, 72))

    # DONCHIAN_TURTLE: 1h prior-channel breakout, long-only per live comments plus a symmetric scout.
    c1 = df1h["close"]
    hi20 = df1h["high"].rolling(20).max().shift(1)
    lo20 = df1h["low"].rolling(20).min().shift(1)
    don_long = c1 > hi20
    don_short = c1 < lo20
    models.append(SignalSet("DONCHIAN_TURTLE_20_long_only", "breakout", 60, pack(df1h.index, don_long, np.zeros(len(df1h), dtype=bool), 60), 36, 72, 576))
    models.append(SignalSet("DONCHIAN_20_symmetric_scout", "breakout", 60, pack(df1h.index, don_long, don_short, 60), 36, 72, 576))

    # KAMA_V2: adaptive MA cross with ER, EMA200, MACD histogram and RSI extreme filter.
    k, er = kama(c5)
    ema200 = ema(c5, 200)
    macd = ema(c5, 12) - ema(c5, 26)
    macd_hist = macd - ema(macd, 9)
    r5k = rsi(c5, 14)
    cross_up = (c5 > k) & (c5.shift(1) <= k.shift(1))
    cross_dn = (c5 < k) & (c5.shift(1) >= k.shift(1))
    k_long = cross_up & (er >= 0.35) & (c5 > ema200) & (macd_hist > 0) & (r5k < 70)
    k_short = cross_dn & (er >= 0.35) & (c5 < ema200) & (macd_hist < 0) & (r5k > 30)
    models.append(SignalSet("KAMA_V2_cross_er_confirmed", "kama", 5, pack(df5.index, k_long, k_short, 5), 12, 18, 72))

    # IMPULSE_FADE: late-session large body vs ATR, fade the candle.
    a5 = atr(df5, 14) / PIP
    body = (c5 - df5["open"]) / PIP
    hour5 = df5.index.hour
    late = (hour5 >= 18) & (hour5 <= 22)
    impulse = body.abs() >= (2.2 * a5)
    vol_ok = a5 <= 18
    models.append(SignalSet("IMPULSE_FADE_late_us", "impulse_fade", 5, pack(df5.index, (body < 0) & impulse & late & vol_ok, (body > 0) & impulse & late & vol_ok, 5), 18, 10, 36))

    # SMC_MOMENTUM: simplified liquidity sweep and rejection in 15m, with 4h EMA50 alignment.
    c4 = df4h["close"]
    bias4 = pd.Series(0, index=df4h.index, dtype=np.int8)
    e4 = ema(c4, 50)
    bias4[c4 > e4] = 1
    bias4[c4 < e4] = -1
    b4_at_15 = bias4.reindex(df15.index, method="ffill").fillna(0).astype(np.int8)
    ph = df15["high"].rolling(24).max().shift(1)
    pl = df15["low"].rolling(24).min().shift(1)
    rng15 = (df15["high"] - df15["low"]).replace(0, np.nan)
    body_pct = (df15["close"] - df15["open"]).abs() / rng15
    sweep_low = ((pl - df15["low"]) / PIP).between(3, 18) & (df15["close"] > pl) & (df15["close"] > df15["open"])
    sweep_high = ((df15["high"] - ph) / PIP).between(3, 18) & (df15["close"] < ph) & (df15["close"] < df15["open"])
    smc_long = sweep_low & (body_pct >= 0.25) & (b4_at_15 >= 0)
    smc_short = sweep_high & (body_pct >= 0.25) & (b4_at_15 <= 0)
    models.append(SignalSet("SMC_MOMENTUM_sweep_reject", "smc_momentum", 15, pack(df15.index, smc_long, smc_short, 15), 22, 30, 144))

    # SMC_SIMPLE-style trend pullback/resume: 1h/4h bias, 15m pullback, close resumes.
    e1 = ema(c1, 50)
    e1_slope = e1.diff(6)
    bias1 = pd.Series(0, index=df1h.index, dtype=np.int8)
    bias1[(c1 > e1) & (e1_slope > 0)] = 1
    bias1[(c1 < e1) & (e1_slope < 0)] = -1
    b1_at_15 = bias1.reindex(df15.index, method="ffill").fillna(0).astype(np.int8)
    pull = c15.diff(3) / PIP
    candle_body = (c15 - df15["open"]) / PIP
    smc_simple_long = (b1_at_15 == 1) & (pull < -5) & (candle_body > 1.5) & (r15 > 40) & (r15 < 65)
    smc_simple_short = (b1_at_15 == -1) & (pull > 5) & (candle_body < -1.5) & (r15 < 60) & (r15 > 35)
    models.append(SignalSet("SMC_SIMPLE_trend_pullback_resume", "smc_simple", 15, pack(df15.index, smc_simple_long, smc_simple_short, 15), 20, 30, 144))

    return models


def align_to_5m(models: Iterable[SignalSet], df5: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    idx = df5.index
    for model in models:
        s = pd.Series(0, index=idx, dtype=np.int8)
        if model.signals.empty:
            out[model.name] = s
            continue
        times = pd.to_datetime(model.signals["time"], utc=True)
        loc = idx.get_indexer(times, method="backfill")
        valid = loc >= 0
        loc = loc[valid]
        aligned_times = idx[loc]
        valid_times = pd.DatetimeIndex(times[valid])
        fresh = aligned_times - valid_times <= pd.Timedelta(minutes=5)
        loc = loc[fresh]
        sides = model.signals.loc[valid, "side"].astype(np.int8).to_numpy()
        sides = sides[fresh]
        s.iloc[loc] = sides
        out[model.name] = s
    return out


def confluence_models(models: list[SignalSet], df5: pd.DataFrame, df1h: pd.DataFrame) -> list[SignalSet]:
    aligned = align_to_5m(models, df5)
    names = list(aligned)
    out: list[SignalSet] = []

    def build(label: str, selected: list[str], min_votes: int, lookback_bars: int) -> pd.DataFrame:
        long_votes = pd.DataFrame({n: (aligned[n] == 1).rolling(lookback_bars, min_periods=1).max() for n in selected}).sum(axis=1)
        short_votes = pd.DataFrame({n: (aligned[n] == -1).rolling(lookback_bars, min_periods=1).max() for n in selected}).sum(axis=1)
        raw_long = (long_votes >= min_votes) & (long_votes > short_votes)
        raw_short = (short_votes >= min_votes) & (short_votes > long_votes)
        # Edge-trigger the confluence so a cluster produces one candidate.
        long_edge = raw_long & ~raw_long.shift(1, fill_value=False)
        short_edge = raw_short & ~raw_short.shift(1, fill_value=False)
        return pack(df5.index, long_edge, short_edge, 5)

    trend = [n for n in names if any(k in n for k in ("DONCHIAN", "KAMA", "SMC_SIMPLE", "SMC_MOMENTUM"))]
    fade = [n for n in names if any(k in n for k in ("MEAN_REVERSION", "RANGE_FADE", "IMPULSE_FADE"))]
    all_names = names

    out.append(SignalSet("COMBO_any2_all_60m", "combo", 5, build("all2", all_names, 2, 12), 20, 30, 144))
    out.append(SignalSet("COMBO_any3_all_90m", "combo", 5, build("all3", all_names, 3, 18), 20, 32, 144))
    out.append(SignalSet("COMBO_trend2_60m", "combo_trend", 5, build("trend2", trend, 2, 12), 22, 34, 168))
    out.append(SignalSet("COMBO_fade2_45m", "combo_fade", 5, build("fade2", fade, 2, 9), 16, 18, 72))

    # Regime switch: use trend confluence when 1h ADX is elevated, fade confluence otherwise.
    ax1 = adx(df1h, 14).reindex(df5.index, method="ffill")
    trend_sig = build("trend2", trend, 2, 12).set_index("time")["side"] if trend else pd.Series(dtype=np.int8)
    fade_sig = build("fade2", fade, 2, 9).set_index("time")["side"] if fade else pd.Series(dtype=np.int8)
    regime_side = pd.Series(0, index=df5.index, dtype=np.int8)
    if not trend_sig.empty:
        regime_side.loc[trend_sig.index.intersection(df5.index)] = trend_sig.loc[trend_sig.index.intersection(df5.index)]
        regime_side[(regime_side != 0) & (ax1 < 22)] = 0
    if not fade_sig.empty:
        fade_aligned = pd.Series(0, index=df5.index, dtype=np.int8)
        fade_aligned.loc[fade_sig.index.intersection(df5.index)] = fade_sig.loc[fade_sig.index.intersection(df5.index)]
        regime_side[(regime_side == 0) & (ax1 < 22)] = fade_aligned[(regime_side == 0) & (ax1 < 22)]
    out.append(SignalSet("COMBO_regime_trend_or_fade", "combo_regime", 5, pack(df5.index, regime_side == 1, regime_side == -1, 5), 20, 28, 144))
    return out


def unified_entry_stack_model(models: list[SignalSet], df5: pd.DataFrame, df1h: pd.DataFrame) -> SignalSet:
    """One EURJPY strategy stream using all active entry models as internal modules.

    Priority order:
      1. Trend confluence: at least two trend modules agree within 60 minutes.
      2. Low-regime fade confluence: at least two fade modules agree within 45 minutes
         while 1h ADX is below 22.
      3. Standalone high-quality modules: Donchian long, SMC momentum, impulse fade,
         and rejection-style mean reversion.

    The output is one de-duplicated signal stream. Diagnostic module names stay in
    the script so we can see what contributed, but this is the candidate strategy.
    """
    aligned = align_to_5m(models, df5)
    names = list(aligned)
    trend = [n for n in names if any(k in n for k in ("DONCHIAN", "KAMA", "SMC_SIMPLE", "SMC_MOMENTUM"))]
    fade = [n for n in names if any(k in n for k in ("MEAN_REVERSION", "RANGE_FADE", "IMPULSE_FADE"))]
    ax1 = adx(df1h, 14).reindex(df5.index, method="ffill")

    def votes(selected: list[str], side: int, lookback: int) -> pd.Series:
        if not selected:
            return pd.Series(0, index=df5.index, dtype=float)
        frame = pd.DataFrame({
            n: (aligned[n] == side).rolling(lookback, min_periods=1).max()
            for n in selected
        })
        return frame.sum(axis=1)

    trend_long = votes(trend, 1, 12)
    trend_short = votes(trend, -1, 12)
    fade_long = votes(fade, 1, 9)
    fade_short = votes(fade, -1, 9)

    side = pd.Series(0, index=df5.index, dtype=np.int8)

    trend_buy = (trend_long >= 2) & (trend_long > trend_short)
    trend_sell = (trend_short >= 2) & (trend_short > trend_long)
    side[trend_buy] = 1
    side[trend_sell] = -1

    low_regime = ax1 < 22
    fade_buy = low_regime & (side == 0) & (fade_long >= 2) & (fade_long > fade_short)
    fade_sell = low_regime & (side == 0) & (fade_short >= 2) & (fade_short > fade_long)
    side[fade_buy] = 1
    side[fade_sell] = -1

    fallback_order = [
        "DONCHIAN_TURTLE_20_long_only",
        "SMC_MOMENTUM_sweep_reject",
        "IMPULSE_FADE_late_us",
        "MEAN_REVERSION_reject",
    ]
    for name in fallback_order:
        if name not in aligned:
            continue
        s = aligned[name]
        side[(side == 0) & (s != 0)] = s[(side == 0) & (s != 0)]

    raw_long = side == 1
    raw_short = side == -1
    long_edge = raw_long & ~raw_long.shift(1, fill_value=False)
    short_edge = raw_short & ~raw_short.shift(1, fill_value=False)
    return SignalSet(
        "EURJPY_UNIFIED_ENTRY_STACK",
        "unified",
        5,
        pack(df5.index, long_edge, short_edge, 5),
        22,
        34,
        168,
    )


def simulate(df5: pd.DataFrame, model: SignalSet, cooldown_bars: int = 6) -> pd.DataFrame:
    if model.signals.empty:
        return pd.DataFrame()
    highs = df5["high"].to_numpy(float)
    lows = df5["low"].to_numpy(float)
    closes = df5["close"].to_numpy(float)
    idx = df5.index
    signal_times = pd.to_datetime(model.signals["time"], utc=True)
    loc = idx.get_indexer(signal_times, method="backfill")
    trades = []
    blocked_until = -1
    for raw_loc, signal_time, side in zip(loc, signal_times, model.signals["side"].to_numpy(np.int8)):
        if raw_loc < 0 or raw_loc <= blocked_until or raw_loc >= len(df5) - 2:
            continue
        if idx[raw_loc] - signal_time > pd.Timedelta(minutes=5):
            continue
        entry_i = raw_loc
        entry = closes[entry_i]
        sl_px = entry - side * model.sl_pips * PIP
        tp_px = entry + side * model.tp_pips * PIP
        end = min(entry_i + model.hold_bars_5m, len(df5) - 1)
        exit_i = end
        exit_px = closes[end]
        outcome = "TIMEOUT"
        mfe = 0.0
        mae = 0.0
        for j in range(entry_i + 1, end + 1):
            if side == 1:
                mfe = max(mfe, (highs[j] - entry) / PIP)
                mae = min(mae, (lows[j] - entry) / PIP)
                hit_sl = lows[j] <= sl_px
                hit_tp = highs[j] >= tp_px
            else:
                mfe = max(mfe, (entry - lows[j]) / PIP)
                mae = min(mae, (entry - highs[j]) / PIP)
                hit_sl = highs[j] >= sl_px
                hit_tp = lows[j] <= tp_px
            if hit_sl and hit_tp:
                exit_i = j
                exit_px = sl_px
                outcome = "SL_STRADDLE"
                break
            if hit_sl:
                exit_i = j
                exit_px = sl_px
                outcome = "SL"
                break
            if hit_tp:
                exit_i = j
                exit_px = tp_px
                outcome = "TP"
                break
        pips = ((exit_px - entry) * side / PIP) - COST_PIPS
        trades.append(
            {
                "model": model.name,
                "family": model.family,
                "entry_time": idx[entry_i],
                "exit_time": idx[exit_i],
                "side": int(side),
                "entry": entry,
                "exit": exit_px,
                "outcome": outcome,
                "pips": pips,
                "mfe": mfe,
                "mae": mae,
            }
        )
        blocked_until = exit_i + cooldown_bars
    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    t = trades[(trades["entry_time"] >= start) & (trades["entry_time"] <= end)] if not trades.empty else trades
    months = max((end - start).days / 30.4375, 0.01)
    if t.empty:
        return {"n": 0, "tpm": 0.0, "pf": 0.0, "wr": 0.0, "pips": 0.0, "avg": 0.0, "dd": 0.0}
    p = t["pips"].astype(float)
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    cum = p.cumsum()
    return {
        "n": int(len(t)),
        "tpm": float(len(t) / months),
        "pf": float(wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0),
        "wr": float((p > 0).mean()),
        "pips": float(p.sum()),
        "avg": float(p.mean()),
        "dd": float((cum.cummax() - cum).max()),
    }


def print_table(rows: list[dict], title: str, limit: int = 30) -> None:
    print("\n" + title)
    print("-" * 132)
    print(f"{'model':42s} {'IS n':>5s} {'IS PF':>6s} {'OOS n':>6s} {'OOS PF':>7s} {'OOS WR':>7s} {'OOS pips':>10s} {'H1 PF':>7s} {'H2 PF':>7s} {'OOS DD':>8s}")
    print("-" * 132)
    for r in rows[:limit]:
        print(
            f"{r['model'][:42]:42s} "
            f"{r['is']['n']:5d} {r['is']['pf']:6.2f} "
            f"{r['oos']['n']:6d} {r['oos']['pf']:7.2f} {100*r['oos']['wr']:6.1f}% "
            f"{r['oos']['pips']:10.1f} {r['h1']['pf']:7.2f} {r['h2']['pf']:7.2f} {r['oos']['dd']:8.1f}"
        )


def main() -> None:
    global COST_PIPS

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-oos-trades", type=int, default=40)
    ap.add_argument("--cost-pips", type=float, default=1.2)
    args = ap.parse_args()

    COST_PIPS = float(args.cost_pips)

    print(f"Loading EURJPY candles for {WINDOW_START}..{WINDOW_END}")
    df5 = load_candles(5)
    df15 = load_candles(15)
    df1h = load_candles(60)
    df4h = load_candles(240)
    print(f"Rows: 5m={len(df5)} 15m={len(df15)} 1h={len(df1h)} 4h={len(df4h)}")

    base = make_active_entry_models(df5, df15, df1h, df4h)
    unified = unified_entry_stack_model(base, df5, df1h)
    combos = confluence_models(base, df5, df1h)
    all_models = [unified] + base + combos

    rows = []
    for model in all_models:
        trades = simulate(df5, model)
        rows.append(
            {
                "model": model.name,
                "family": model.family,
                "is": metrics(trades, pd.Timestamp(WINDOW_START, tz="UTC"), IS_END),
                "oos": metrics(trades, OOS_START, pd.Timestamp(WINDOW_END, tz="UTC")),
                "h1": metrics(trades, OOS_START, OOS_MID - pd.Timedelta(seconds=1)),
                "h2": metrics(trades, OOS_MID, pd.Timestamp(WINDOW_END, tz="UTC")),
            }
        )

    rows_pf = sorted(rows, key=lambda r: (r["oos"]["pf"], r["oos"]["pips"], r["oos"]["n"]), reverse=True)
    rows_viable = [r for r in rows_pf if r["oos"]["n"] >= args.min_oos_trades]
    rows_consistent = [
        r for r in rows_viable
        if r["oos"]["pf"] >= 1.15 and r["h1"]["pf"] >= 1.0 and r["h2"]["pf"] >= 1.0 and r["oos"]["pips"] > 0
    ]

    unified_rows = [r for r in rows if r["model"] == "EURJPY_UNIFIED_ENTRY_STACK"]
    print_table(unified_rows, "Single Strategy Candidate")
    print_table(rows_pf, "Diagnostics: active entry modules and comparison combinations ranked by OOS PF")
    print_table(rows_viable, f"Viable sample size (OOS n >= {args.min_oos_trades})")
    print_table(rows_consistent, "Consistent candidates (OOS PF >= 1.15, both OOS halves PF >= 1.0)")

    print("\nInterpretation:")
    print("  - EURJPY_UNIFIED_ENTRY_STACK is one strategy stream; the other rows are diagnostics.")
    print("  - Internal entry modules are simplified but mapped to the live active strategy entry families.")
    print("  - Exits are normalized first-passage fixed SL/TP brackets, not production trailing.")
    print("  - Any candidate here still needs a production backtest using the real strategy pipeline before enabling.")


if __name__ == "__main__":
    main()
