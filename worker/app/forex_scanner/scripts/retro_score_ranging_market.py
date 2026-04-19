#!/usr/bin/env python3
"""Retro-score RANGING_MARKET signals against proposed hard-ADX gates.

For each historical RANGING_MARKET alert:
  - 15m ADX is taken from alert_history.adx (the primary-TF value stamped at signal time)
  - 1h  ADX is computed from ig_candles_backtest 5m candles resampled to 1h, ending at the alert timestamp

Then sweep across candidate (primary_ceiling, htf_ceiling) combinations and report
the kept-set profit factor / win rate using trade_log.profit_loss as the outcome source.

Usage (inside the task-worker container, where pandas/psycopg2 are available):

  docker exec -it task-worker python /app/forex_scanner/scripts/retro_score_ranging_market.py
  docker exec -it task-worker python /app/forex_scanner/scripts/retro_score_ranging_market.py \
      --primary-ceilings 18,20,22,24,99 \
      --htf-ceilings 20,22,25,28,99 \
      --lookback-days 90
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import psycopg2


FOREX_DB = "postgresql://postgres:postgres@postgres:5432/forex"
HTF_LOOKBACK_HOURS = 48  # how far back to fetch 5m candles for 1h ADX computation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-ceilings", default="18,20,22,24,99",
                   help="Comma list of 15m ADX ceilings to sweep. Use 99 for 'no gate'.")
    p.add_argument("--htf-ceilings", default="20,22,25,28,99",
                   help="Comma list of 1h ADX ceilings to sweep. Use 99 for 'no gate'.")
    p.add_argument("--lookback-days", type=int, default=90,
                   help="How far back to pull RANGING_MARKET alerts.")
    p.add_argument("--db", default=FOREX_DB, help="Forex DB connection string.")
    p.add_argument("--per-epic", action="store_true",
                   help="Also report kept-set PF broken down by epic for the best ceiling combo.")
    p.add_argument("--canonical", action="store_true",
                   help="Use alert_history_recomputed canonical ADX values "
                        "(EMA-Wilder, computed from raw candles) instead of "
                        "alert_history.adx. Requires recompute_adx_regime.py to "
                        "have been run first.")
    return p.parse_args()


def parse_ceiling_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def fetch_alerts(conn, lookback_days: int, use_canonical: bool = False) -> pd.DataFrame:
    """Pull RANGING_MARKET alerts joined with trade_log outcomes.

    When use_canonical=True, ADX values are read from alert_history_recomputed
    (EMA-Wilder, computed from raw candles). Otherwise the stored alert_history
    columns are used. Includes signals with no trade match (outcome=NaN) so we
    can still report filter-rate metrics; PF/WR is computed only over signals
    with realized PnL.
    """
    if use_canonical:
        sql = """
            SELECT  a.id                  AS alert_id,
                    a.alert_timestamp,
                    a.epic,
                    a.signal_type,
                    COALESCE(r.regime_canonical, a.market_regime) AS market_regime,
                    r.adx_15m_canonical   AS adx_15m,
                    r.adx_1h_canonical    AS adx_1h_pre,
                    a.confidence_score,
                    t.profit_loss,
                    t.pips_gained,
                    t.status              AS trade_status
            FROM    alert_history a
            LEFT JOIN alert_history_recomputed r ON r.alert_id = a.id
            LEFT JOIN trade_log t ON t.alert_id = a.id
            WHERE   a.strategy = 'RANGING_MARKET'
              AND   a.alert_timestamp >= NOW() - INTERVAL %s
            ORDER BY a.alert_timestamp
        """
    else:
        sql = """
            SELECT  a.id            AS alert_id,
                    a.alert_timestamp,
                    a.epic,
                    a.signal_type,
                    a.market_regime,
                    a.adx           AS adx_15m,
                    NULL::numeric   AS adx_1h_pre,
                    a.confidence_score,
                    t.profit_loss,
                    t.pips_gained,
                    t.status        AS trade_status
            FROM    alert_history a
            LEFT JOIN trade_log t ON t.alert_id = a.id
            WHERE   a.strategy = 'RANGING_MARKET'
              AND   a.alert_timestamp >= NOW() - INTERVAL %s
            ORDER BY a.alert_timestamp
        """
    df = pd.read_sql(sql, conn, params=(f"{lookback_days} days",))
    df["alert_timestamp"] = pd.to_datetime(df["alert_timestamp"])
    df["adx_15m"] = pd.to_numeric(df["adx_15m"], errors="coerce")
    df["adx_1h_pre"] = pd.to_numeric(df["adx_1h_pre"], errors="coerce")
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce")
    return df


def adx_series(df_ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """EMA-Wilder ADX, matching DataFetcher._add_adx_indicator.

    The live `df['adx']` column (and therefore the value stamped into
    `alert_history.adx`) uses pandas `.ewm(alpha=1/period, adjust=False)`
    Wilder smoothing — NOT the SMA in RangingMarketStrategy._get_adx's
    fallback path. We replicate the EMA-Wilder formula here so the HTF
    ADX values we compute for the gate sweep are directly comparable to
    what the live strategy would see going forward.
    """
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

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean()


def compute_htf_adx(conn, epic: str, ts: datetime) -> Optional[float]:
    """ADX(14) on 1h candles ending at `ts`, sourced from ig_candles_backtest."""
    start = ts - timedelta(hours=HTF_LOOKBACK_HOURS)
    sql = """
        SELECT start_time, open, high, low, close
          FROM ig_candles_backtest
         WHERE epic = %s AND timeframe = 5
           AND start_time >= %s AND start_time <= %s
         ORDER BY start_time
    """
    df = pd.read_sql(sql, conn, params=(epic, start, ts))
    if df.empty or len(df) < 30:
        return None
    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.set_index("start_time")
    df_1h = df.resample("1h").agg({"open": "first", "high": "max",
                                     "low": "min", "close": "last"}).dropna()
    if len(df_1h) < 20:
        return None
    s = adx_series(df_1h, period=14).dropna()
    return float(s.iloc[-1]) if not s.empty else None


def enrich_with_htf_adx(conn, df: pd.DataFrame) -> pd.DataFrame:
    """Add an `adx_1h` column by computing per-row HTF ADX."""
    htf_values = []
    misses = 0
    for ts, epic in zip(df["alert_timestamp"], df["epic"]):
        v = compute_htf_adx(conn, epic, ts.to_pydatetime())
        if v is None:
            misses += 1
        htf_values.append(v)
    df = df.copy()
    df["adx_1h"] = htf_values
    print(f"[retro] HTF ADX computed for {len(df) - misses}/{len(df)} alerts "
          f"({misses} misses — usually pre-history candles)")
    return df


def score_combo(df: pd.DataFrame, primary_ceiling: float, htf_ceiling: float) -> dict:
    """Apply the gate to the alert set and return aggregate stats.

    Filter rule: keep signal iff (adx_15m <= primary_ceiling OR adx_15m is null)
                                AND (adx_1h  <= htf_ceiling     OR adx_1h is null).
    Null ADX is treated as 'unknown -> keep' so we don't penalize signals where
    the candle history was insufficient to compute the value.
    """
    primary_pass = (df["adx_15m"].isna()) | (df["adx_15m"] <= primary_ceiling)
    htf_pass = (df["adx_1h"].isna()) | (df["adx_1h"] <= htf_ceiling)
    kept = df[primary_pass & htf_pass]
    rejected = df[~(primary_pass & htf_pass)]

    kept_with_pnl = kept[kept["profit_loss"].notna()]
    rejected_with_pnl = rejected[rejected["profit_loss"].notna()]

    def stats(s: pd.DataFrame) -> dict:
        if len(s) == 0:
            return {"n": 0, "wr": np.nan, "pf": np.nan, "avg_pnl": np.nan, "total_pnl": np.nan}
        wins = s[s["profit_loss"] > 0]["profit_loss"].sum()
        losses = -s[s["profit_loss"] < 0]["profit_loss"].sum()
        wr = (s["profit_loss"] > 0).mean()
        pf = wins / losses if losses > 0 else np.inf
        return {"n": len(s), "wr": wr, "pf": pf,
                "avg_pnl": s["profit_loss"].mean(),
                "total_pnl": s["profit_loss"].sum()}

    return {
        "primary_ceiling": primary_ceiling,
        "htf_ceiling": htf_ceiling,
        "alerts_kept": len(kept),
        "alerts_rejected": len(rejected),
        "kept_traded": stats(kept_with_pnl),
        "rejected_traded": stats(rejected_with_pnl),
    }


def print_sweep(rows: List[dict]) -> None:
    print()
    print(f"{'pri':>4} {'htf':>4}  {'kept':>5} {'rej':>5}  "
          f"{'kept_n':>7} {'kept_wr':>8} {'kept_pf':>8} {'kept_pnl':>10}  "
          f"{'rej_n':>6} {'rej_pf':>7} {'rej_pnl':>10}")
    print("-" * 100)
    for r in rows:
        k, j = r["kept_traded"], r["rejected_traded"]
        def fmt(v, w, prec):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{'n/a':>{w}}"
            if isinstance(v, float) and np.isinf(v):
                return f"{'inf':>{w}}"
            return f"{v:>{w}.{prec}f}"
        print(
            f"{r['primary_ceiling']:>4.0f} {r['htf_ceiling']:>4.0f}  "
            f"{r['alerts_kept']:>5d} {r['alerts_rejected']:>5d}  "
            f"{k['n']:>7d} {fmt(k['wr'], 8, 3)} {fmt(k['pf'], 8, 3)} "
            f"{fmt(k['total_pnl'], 10, 2)}  "
            f"{j['n']:>6d} {fmt(j['pf'], 7, 3)} {fmt(j['total_pnl'], 10, 2)}"
        )


def per_epic_breakdown(df: pd.DataFrame, primary_ceiling: float, htf_ceiling: float) -> None:
    """Show kept-set PF per epic for the chosen ceiling combo."""
    primary_pass = (df["adx_15m"].isna()) | (df["adx_15m"] <= primary_ceiling)
    htf_pass = (df["adx_1h"].isna()) | (df["adx_1h"] <= htf_ceiling)
    kept = df[primary_pass & htf_pass]
    kept_with_pnl = kept[kept["profit_loss"].notna()]
    print(f"\nPer-epic kept-set PF at primary={primary_ceiling}, htf={htf_ceiling}:")
    print(f"{'epic':<24} {'n':>4} {'wr':>6} {'pf':>7} {'avg_pnl':>9} {'total':>10}")
    for epic, sub in kept_with_pnl.groupby("epic"):
        wins = sub[sub["profit_loss"] > 0]["profit_loss"].sum()
        losses = -sub[sub["profit_loss"] < 0]["profit_loss"].sum()
        wr = (sub["profit_loss"] > 0).mean()
        pf = wins / losses if losses > 0 else float("inf")
        print(f"{epic:<24} {len(sub):>4d} {wr:>6.2%} {pf:>7.3f} "
              f"{sub['profit_loss'].mean():>9.2f} {sub['profit_loss'].sum():>10.2f}")


def main() -> int:
    args = parse_args()
    primary_ceilings = parse_ceiling_list(args.primary_ceilings)
    htf_ceilings = parse_ceiling_list(args.htf_ceilings)

    print(f"[retro] Connecting to {args.db}")
    conn = psycopg2.connect(args.db)

    print(f"[retro] Pulling RANGING_MARKET alerts (last {args.lookback_days}d), "
          f"adx_source={'canonical (sidecar)' if args.canonical else 'alert_history.adx'}")
    df = fetch_alerts(conn, args.lookback_days, use_canonical=args.canonical)
    print(f"[retro] Fetched {len(df)} alerts; "
          f"{df['profit_loss'].notna().sum()} have trade PnL")

    if df.empty:
        print("[retro] No alerts found — nothing to score.")
        return 0

    if args.canonical and df["adx_1h_pre"].notna().sum() == len(df):
        print(f"[retro] Using pre-computed canonical 1h ADX from sidecar table")
        df["adx_1h"] = df["adx_1h_pre"]
    else:
        print("[retro] Computing 1h ADX per alert...")
        df = enrich_with_htf_adx(conn, df)

    # Baseline (no gate) for reference
    baseline = score_combo(df, primary_ceiling=999, htf_ceiling=999)
    bk = baseline["kept_traded"]
    print(f"\n[BASELINE — no gate]: traded n={bk['n']}, "
          f"WR={bk['wr']:.2%}, PF={bk['pf']:.3f}, total_pnl={bk['total_pnl']:.2f}")

    print(f"\n[retro] Sweeping {len(primary_ceilings)} × {len(htf_ceilings)} combos...")
    rows: List[dict] = []
    for pc in primary_ceilings:
        for hc in htf_ceilings:
            rows.append(score_combo(df, pc, hc))

    rows.sort(key=lambda r: (
        -(r["kept_traded"]["pf"] if not np.isnan(r["kept_traded"]["pf"]) else -1),
        -r["kept_traded"]["n"],
    ))
    print_sweep(rows)

    best = rows[0]
    print(f"\n[BEST]  primary={best['primary_ceiling']}, htf={best['htf_ceiling']}: "
          f"kept_n={best['kept_traded']['n']}, "
          f"PF={best['kept_traded']['pf']:.3f}, "
          f"WR={best['kept_traded']['wr']:.2%}, "
          f"total_pnl={best['kept_traded']['total_pnl']:.2f}")

    if args.per_epic:
        per_epic_breakdown(df, best["primary_ceiling"], best["htf_ceiling"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
