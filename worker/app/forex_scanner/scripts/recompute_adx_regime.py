#!/usr/bin/env python3
"""Canonical ADX & regime recomputation for historical alerts.

Why this exists
---------------
`alert_history.adx` and `alert_history.market_regime` were stamped at signal
time by code that has changed several times (Mar 6, Apr 9, Apr 17 fixes).
Two specific historical bugs make those columns unreliable for retro-analysis:

1.  ADX formula divergence: DataFetcher uses EMA-Wilder smoothing
    (df.ewm(alpha=1/period, adjust=False).mean()) but earlier strategy
    fallbacks used SMA. Different strategies and different timeframes have
    sometimes been stored with different formulas.
2.  Regime label drift: signal_detector.py uses thresholds
       <20 → ranging, 20-25 → low_volatility, 25-50 → trending, >50 → breakout
    plus volatility-based downgrades (efficiency_ratio < 0.25, weekly
    oscillation). Earlier code paths sometimes overwrote the strategy's
    own label with the router's label or vice versa.

This utility recomputes ADX(14) on 5m / 15m / 1h timeframes for every
historical alert from raw 1m candles in `ig_candles` (or 5m candles in
`ig_candles_backtest` for older periods where 1m is sparse), using the
canonical EMA-Wilder formula. It also re-derives the structural regime
label using the canonical thresholds. Results are written to
`alert_history_recomputed` (a sidecar — alert_history is never modified).

Usage
-----
  # Recompute everything (idempotent — uses INSERT ... ON CONFLICT)
  docker exec -it task-worker python /app/forex_scanner/scripts/recompute_adx_regime.py

  # Only RANGING_MARKET alerts in last 90 days
  docker exec -it task-worker python /app/forex_scanner/scripts/recompute_adx_regime.py \
      --strategy RANGING_MARKET --lookback-days 90

  # Force-overwrite previously recomputed rows
  docker exec -it task-worker python /app/forex_scanner/scripts/recompute_adx_regime.py --overwrite
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2


FOREX_DB = "postgresql://postgres:postgres@postgres:5432/forex"

# Lookback windows tuned so we always have ≥ 30 bars to seed EMA-Wilder
WINDOW_HOURS_5M = 24
WINDOW_HOURS_15M = 48
WINDOW_HOURS_1H = 96


# ---------------------------------------------------------------------------
# Canonical formulas (EMA-Wilder, period 14)
# ---------------------------------------------------------------------------

def ema_wilder_adx(df_ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX using pandas EMA-Wilder smoothing — matches DataFetcher."""
    high, low, close = df_ohlc["high"], df_ohlc["low"], df_ohlc["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                         index=df_ohlc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                          index=df_ohlc.index)

    a = 1.0 / period
    atr = tr.ewm(alpha=a, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def ema_wilder_atr(df_ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df_ohlc["high"], df_ohlc["low"], df_ohlc["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def regime_from_adx(adx_value: Optional[float]) -> Optional[str]:
    """Canonical structural regime label from 1h ADX.

    Mirrors signal_detector.py thresholds:
      <20  → ranging
      <25  → low_volatility
      <=50 → trending
      >50  → breakout
    Returns None when ADX is missing.
    """
    if adx_value is None or pd.isna(adx_value):
        return None
    if adx_value < 20:
        return "ranging"
    if adx_value < 25:
        return "low_volatility"
    if adx_value <= 50:
        return "trending"
    return "breakout"


# ---------------------------------------------------------------------------
# Candle fetch
# ---------------------------------------------------------------------------

def fetch_ohlc(conn, epic: str, end: datetime, hours: int,
                timeframe: int) -> Tuple[pd.DataFrame, str]:
    """Fetch raw candles ending at `end`, preferring live ig_candles when fresh.

    Returns (df, source_label). source_label is recorded so a row's lineage
    (was it computed from live 1m or from backtest 5m?) is auditable.
    """
    start = end - timedelta(hours=hours)

    # Try ig_candles first (the live source)
    sql = """
        SELECT start_time, open, high, low, close
          FROM ig_candles
         WHERE epic = %s AND timeframe = %s
           AND start_time >= %s AND start_time <= %s
         ORDER BY start_time
    """
    df = pd.read_sql(sql, conn, params=(epic, timeframe, start, end))
    if not df.empty and len(df) >= 30:
        return df, f"ig_candles_tf{timeframe}"

    # Fallback to ig_candles_backtest (pre-synthesized)
    df = pd.read_sql(sql.replace("ig_candles", "ig_candles_backtest"),
                      conn, params=(epic, timeframe, start, end))
    if not df.empty and len(df) >= 30:
        return df, f"ig_candles_backtest_tf{timeframe}"

    return pd.DataFrame(), "none"


def resample(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df_5m.empty:
        return df_5m
    df = df_5m.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.set_index("start_time")
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()


def compute_adx_at(conn, epic: str, ts: datetime, target_rule: str,
                    base_timeframe: int = 5,
                    window_hours: int = 48) -> Tuple[Optional[float], str]:
    """Compute EMA-Wilder ADX(14) on `target_rule` ending at `ts`."""
    df_raw, source = fetch_ohlc(conn, epic, ts, window_hours, base_timeframe)
    if df_raw.empty:
        return None, source
    if target_rule == f"{base_timeframe}min":
        df = df_raw.copy()
        df["start_time"] = pd.to_datetime(df["start_time"])
        df = df.set_index("start_time")
    else:
        df = resample(df_raw, target_rule)
    if len(df) < 20:
        return None, source
    s = ema_wilder_adx(df).dropna()
    return (float(s.iloc[-1]) if not s.empty else None), source


def compute_atr_at(conn, epic: str, ts: datetime, target_rule: str = "15min",
                    base_timeframe: int = 5,
                    window_hours: int = 48) -> Optional[float]:
    df_raw, _ = fetch_ohlc(conn, epic, ts, window_hours, base_timeframe)
    if df_raw.empty:
        return None
    df = resample(df_raw, target_rule)
    if len(df) < 20:
        return None
    s = ema_wilder_atr(df).dropna()
    return float(s.iloc[-1]) if not s.empty else None


# ---------------------------------------------------------------------------
# Public per-alert helper (importable from elsewhere)
# ---------------------------------------------------------------------------

def recompute_for_alert(conn, alert_id: int, epic: str, ts: datetime) -> Dict:
    """Recompute canonical ADX/regime metrics for a single alert.

    Returns a dict ready to be inserted into alert_history_recomputed.
    """
    adx_5m, src_5m = compute_adx_at(conn, epic, ts, "5min", 5, WINDOW_HOURS_5M)
    adx_15m, src_15m = compute_adx_at(conn, epic, ts, "15min", 5, WINDOW_HOURS_15M)
    adx_1h, src_1h = compute_adx_at(conn, epic, ts, "1h", 5, WINDOW_HOURS_1H)
    atr_15m = compute_atr_at(conn, epic, ts, "15min", 5, WINDOW_HOURS_15M)

    regime = regime_from_adx(adx_1h)

    # Pick the most informative source label (HTF window covers the most history)
    candle_source = src_1h or src_15m or src_5m or "none"

    return {
        "alert_id": alert_id,
        "alert_timestamp": ts,
        "epic": epic,
        "adx_5m_canonical": adx_5m,
        "adx_15m_canonical": adx_15m,
        "adx_1h_canonical": adx_1h,
        "atr_15m_canonical": atr_15m,
        "regime_canonical": regime,
        "candle_source": candle_source,
    }


# ---------------------------------------------------------------------------
# Backfill driver
# ---------------------------------------------------------------------------

def fetch_alerts_to_process(conn, strategy: Optional[str], lookback_days: int,
                             overwrite: bool) -> pd.DataFrame:
    where = ["a.alert_timestamp >= NOW() - INTERVAL %s"]
    params = [f"{lookback_days} days"]
    if strategy:
        where.append("a.strategy = %s")
        params.append(strategy)
    if not overwrite:
        where.append("r.alert_id IS NULL")

    sql = f"""
        SELECT  a.id   AS alert_id,
                a.alert_timestamp,
                a.epic,
                a.adx          AS adx_stored,
                a.market_regime AS market_regime_stored
        FROM    alert_history a
        LEFT JOIN alert_history_recomputed r ON r.alert_id = a.id
        WHERE   {' AND '.join(where)}
        ORDER BY a.alert_timestamp
    """
    return pd.read_sql(sql, conn, params=tuple(params))


def upsert_recomputed(conn, row: Dict) -> None:
    sql = """
        INSERT INTO alert_history_recomputed (
            alert_id, alert_timestamp, epic,
            adx_5m_canonical, adx_15m_canonical, adx_1h_canonical, atr_15m_canonical,
            regime_canonical,
            adx_stored, market_regime_stored,
            candle_source
        )
        VALUES (
            %(alert_id)s, %(alert_timestamp)s, %(epic)s,
            %(adx_5m_canonical)s, %(adx_15m_canonical)s,
            %(adx_1h_canonical)s, %(atr_15m_canonical)s,
            %(regime_canonical)s,
            %(adx_stored)s, %(market_regime_stored)s,
            %(candle_source)s
        )
        ON CONFLICT (alert_id) DO UPDATE SET
            adx_5m_canonical    = EXCLUDED.adx_5m_canonical,
            adx_15m_canonical   = EXCLUDED.adx_15m_canonical,
            adx_1h_canonical    = EXCLUDED.adx_1h_canonical,
            atr_15m_canonical   = EXCLUDED.atr_15m_canonical,
            regime_canonical    = EXCLUDED.regime_canonical,
            adx_stored          = EXCLUDED.adx_stored,
            market_regime_stored = EXCLUDED.market_regime_stored,
            candle_source       = EXCLUDED.candle_source,
            recomputed_at       = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default=None, help="Limit to one strategy (e.g. RANGING_MARKET).")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute rows that already exist in alert_history_recomputed.")
    p.add_argument("--db", default=FOREX_DB)
    p.add_argument("--report", action="store_true",
                   help="After backfill, print divergence summary.")
    return p.parse_args()


def print_divergence_summary(conn, strategy: Optional[str]) -> None:
    where = "WHERE 1=1"
    params = []
    if strategy:
        where += " AND a.strategy = %s"
        params.append(strategy)
    sql = f"""
        SELECT  COUNT(*)                                                        AS total,
                AVG(ABS(r.adx_15m_diff))                                        AS mean_abs_15m_diff,
                AVG(CASE WHEN r.regime_match THEN 0 ELSE 1 END)::numeric(5,3)    AS regime_mismatch_rate,
                COUNT(*) FILTER (WHERE r.adx_stored IS NULL)                    AS stored_adx_null,
                COUNT(*) FILTER (WHERE r.market_regime_stored IS NULL)          AS stored_regime_null
          FROM  alert_history_recomputed r
          JOIN  alert_history a ON a.id = r.alert_id
          {where}
    """
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        cols = [d.name for d in cur.description]
        row = cur.fetchone()
    print("\n[divergence summary]")
    for c, v in zip(cols, row):
        print(f"  {c:>26}: {v}")


def main() -> int:
    args = parse_args()
    conn = psycopg2.connect(args.db)
    conn.autocommit = True

    print(f"[recompute] strategy={args.strategy or 'ALL'} "
          f"lookback={args.lookback_days}d overwrite={args.overwrite}")

    alerts = fetch_alerts_to_process(conn, args.strategy, args.lookback_days, args.overwrite)
    print(f"[recompute] {len(alerts)} alerts to process")
    if alerts.empty:
        return 0

    successes = failures = 0
    for i, a in enumerate(alerts.itertuples(index=False), start=1):
        try:
            row = recompute_for_alert(conn, a.alert_id, a.epic,
                                       pd.to_datetime(a.alert_timestamp).to_pydatetime())
            row["adx_stored"] = float(a.adx_stored) if a.adx_stored is not None else None
            row["market_regime_stored"] = a.market_regime_stored
            upsert_recomputed(conn, row)
            successes += 1
        except Exception as e:
            failures += 1
            print(f"[recompute] alert_id={a.alert_id} failed: {e}")

        if i % 50 == 0:
            print(f"[recompute] progress {i}/{len(alerts)} ok={successes} fail={failures}")

    print(f"[recompute] done: ok={successes} fail={failures}")
    if args.report:
        print_divergence_summary(conn, args.strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
