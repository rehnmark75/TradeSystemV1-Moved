#!/usr/bin/env python3
"""Standalone RANGE_STRUCTURE strategy evaluator.

Pattern mirrors `scripts/eval_mean_reversion.py`: we bypass the production
backtest_cli, pull raw 5m candles from `ig_candles_backtest`, resample to
15m/1h, and run the strategy's detection rules directly on each 15m bar.
Purpose is to get PF/WR/n quickly so we know whether the wick-rejection
thesis holds on USDJPY + JPY crosses at statistically useful n.

Usage (inside task-worker container):
    python /app/forex_scanner/scripts/eval_range_structure.py \
        --scalp --days 90 --canonical \
        --pairs USDJPY,EURJPY,AUDJPY,USDCHF,USDCAD

    # Pre-flight only — print 15m ATR distribution on ranging bars
    python /app/forex_scanner/scripts/eval_range_structure.py \
        --preflight --pairs USDJPY --days 90

The script writes per-pair + portfolio metrics to stdout and a JSON blob to
`scripts/results/range_structure_eval_<timestamp>.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

# Allow `python script.py` to import the strategy module when run standalone.
# Inside the container, PYTHONPATH already includes /app/forex_scanner, but
# when invoked with the full path we want to be robust.
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]  # .../worker/app
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forex_scanner.core.strategies.range_structure_strategy import (  # noqa: E402
    RangeStructureStrategy,
)
from forex_scanner.services.range_structure_config_service import (  # noqa: E402
    get_range_structure_config,
)


FOREX_DB = os.getenv(
    "FOREX_DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/forex",
)

# Short → full epic map (accepts both forms on --pairs)
PAIR_SHORT_TO_EPIC = {
    "EURUSD":  "CS.D.EURUSD.CEEM.IP",
    "GBPUSD":  "CS.D.GBPUSD.MINI.IP",
    "USDJPY":  "CS.D.USDJPY.MINI.IP",
    "AUDUSD":  "CS.D.AUDUSD.MINI.IP",
    "USDCHF":  "CS.D.USDCHF.MINI.IP",
    "USDCAD":  "CS.D.USDCAD.MINI.IP",
    "NZDUSD":  "CS.D.NZDUSD.MINI.IP",
    "EURJPY":  "CS.D.EURJPY.MINI.IP",
    "AUDJPY":  "CS.D.AUDJPY.MINI.IP",
    "GBPJPY":  "CS.D.GBPJPY.MINI.IP",
}

DEFAULT_BASKET = ["USDJPY", "EURJPY", "AUDJPY", "USDCHF", "USDCAD"]


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=90,
                   help="Backtest window (default 90).")
    p.add_argument("--pairs", type=str,
                   default=",".join(DEFAULT_BASKET),
                   help=f"Comma-separated pair shorthands or full epics. "
                        f"Default: {','.join(DEFAULT_BASKET)}")
    p.add_argument("--preflight", action="store_true",
                   help="Print 15m ATR distribution on ranging bars, then exit.")
    p.add_argument("--canonical", action="store_true",
                   help="Prefer alert_history_recomputed for regime labels "
                        "(falls back to alert_history with a warning).")
    p.add_argument("--scalp", action="store_true",
                   help="Scalp-mode flag kept for cli-parity with "
                        "eval_mean_reversion.py; does not change logic.")
    p.add_argument("--max-bars-held", type=int, default=96,
                   help="Timeout (15m bars) for open trades (default 96 = 24h).")
    p.add_argument("--output-json", type=str, default="",
                   help="Output JSON path (default: auto-generated).")
    return p.parse_args()


def resolve_epics(pairs_csv: str) -> List[Tuple[str, str]]:
    """Return [(short, epic), ...]. `pairs_csv` may contain shorthands or epics."""
    out = []
    for raw in pairs_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw in PAIR_SHORT_TO_EPIC:
            out.append((raw, PAIR_SHORT_TO_EPIC[raw]))
        elif raw.startswith("CS.D."):
            # reverse-lookup short name
            short = next(
                (s for s, e in PAIR_SHORT_TO_EPIC.items() if e == raw),
                raw.split(".")[2],
            )
            out.append((short, raw))
        else:
            print(f"[eval_rs] WARN: unknown pair '{raw}' — skipping", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_candles(
    conn, epic: str, days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_5m_raw, df_15m, df_1h). Warmup buffer of 5 days for indicators."""
    end = datetime.now()
    start = end - timedelta(days=days + 5)
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
# Pre-flight: ATR distribution on ranging 15m bars
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


def atr14(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / 14, adjust=False).mean()


def pip_size(epic: str) -> float:
    return 0.01 if "JPY" in epic.upper() else 0.0001


def preflight(conn, pairs: List[Tuple[str, str]], days: int) -> None:
    """Print 15m ATR(14) distribution on ranging bars (ADX < 20) per pair.
    Pulls raw 5m → resamples to 15m → tags ADX → filters ADX < 20."""
    print(f"\n[preflight] 15m ATR(14) distribution on ranging bars "
          f"(ADX<20), last {days} days\n")
    header = ("pair   n      atr_p25  atr_p50  atr_p75   atr_p90  "
              "sl_floor_rec")
    print(header)
    print("-" * len(header))
    for short, epic in pairs:
        _, df15, _ = load_candles(conn, epic, days)
        if df15.empty:
            print(f"{short:<6} (no data)")
            continue
        df15["adx"] = ema_wilder_adx(df15)
        df15["atr"] = atr14(df15)
        rng = df15[df15["adx"] < 20].dropna(subset=["atr"])
        if rng.empty:
            print(f"{short:<6} (no ranging bars)")
            continue
        pip = pip_size(epic)
        atr_pips = rng["atr"] / pip
        p25, p50, p75, p90 = atr_pips.quantile([0.25, 0.5, 0.75, 0.9])
        # Recommended SL floor: max(6, ceil(p50 * 1.3))
        rec_sl = max(6.0, float(np.ceil(p50 * 1.3)))
        print(f"{short:<6} {len(rng):<6d} {p25:7.1f}  {p50:7.1f}  {p75:7.1f}  "
              f"{p90:7.1f}  {rec_sl:8.1f}p")


# ---------------------------------------------------------------------------
# Canonical regime (best-effort)
# ---------------------------------------------------------------------------

def try_canonical_regime_table(conn, epic: str, days: int) -> Optional[pd.DataFrame]:
    """Attempt to read canonical regime from alert_history_recomputed.
    Returns None (with warning) if the table or expected columns are absent."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                 WHERE table_name = 'alert_history_recomputed'
            """)
            cols = {r[0] for r in cur.fetchall()}
        needed = {"epic", "alert_timestamp", "market_regime", "adx"}
        if not needed.issubset(cols):
            print("[eval_rs] WARN: alert_history_recomputed lacks expected "
                  "columns — falling back to alert_history (or unused).",
                  file=sys.stderr)
            return None
        end = datetime.now()
        start = end - timedelta(days=days + 5)
        q = """
            SELECT alert_timestamp, market_regime, adx
              FROM alert_history_recomputed
             WHERE epic = %s
               AND alert_timestamp >= %s AND alert_timestamp <= %s
             ORDER BY alert_timestamp
        """
        df = pd.read_sql(q, conn, params=(epic, start, end))
        if df.empty:
            return None
        df["alert_timestamp"] = pd.to_datetime(df["alert_timestamp"])
        return df.set_index("alert_timestamp")
    except Exception as e:
        print(f"[eval_rs] WARN: canonical regime lookup failed for {epic}: {e}",
              file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Strategy walk + trade sim
# ---------------------------------------------------------------------------

def walk_and_simulate(
    strategy: RangeStructureStrategy,
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    epic: str,
    short: str,
    max_bars_held: int,
) -> pd.DataFrame:
    """Walk bar-by-bar on 15m, call strategy.detect_signal at each bar, then
    simulate the resulting trade by scanning subsequent highs/lows for SL/TP."""
    cfg = strategy.config
    min_bars = cfg.range_lookback_bars + 10
    if len(df15) < min_bars + 5:
        return pd.DataFrame()

    # Reset cooldowns at start of walk — each pair's eval is independent.
    strategy.reset_cooldowns()

    trades: List[Dict] = []
    n = len(df15)
    pip = pip_size(epic)

    # Pre-align 1h frame by timestamp for fast slicing.
    df1h_idx = df1h.index

    i = min_bars
    while i < n:
        ts = df15.index[i]
        slice_15 = df15.iloc[: i + 1]
        # Grab 1h candles up through ts
        mask = df1h_idx <= ts
        slice_1h = df1h.loc[mask]

        try:
            signal = strategy.detect_signal(
                df_trigger=slice_15,
                df_4h=slice_1h,
                epic=epic,
                pair=short,
                current_timestamp=ts,
                routing_context={"regime": "ranging"},
            )
        except Exception as e:
            strategy.logger.debug(
                "[eval_rs] detect_signal raised at %s %s: %s", epic, ts, e)
            signal = None

        if not signal:
            i += 1
            continue

        entry_price = float(signal["entry_price"])
        sl_pips = float(signal["risk_pips"])
        tp_pips = float(signal["reward_pips"])
        direction = signal["signal"]
        if direction == "BUY":
            sl_price = entry_price - sl_pips * pip
            tp_price = entry_price + tp_pips * pip
        else:
            sl_price = entry_price + sl_pips * pip
            tp_price = entry_price - tp_pips * pip

        outcome = "OPEN"
        exit_idx = None
        exit_price = entry_price
        end_idx = min(i + 1 + max_bars_held, n)
        for j in range(i + 1, end_idx):
            hi = float(df15["high"].iloc[j])
            lo = float(df15["low"].iloc[j])
            if direction == "BUY":
                if lo <= sl_price:
                    outcome, exit_idx, exit_price = "SL", j, sl_price
                    break
                if hi >= tp_price:
                    outcome, exit_idx, exit_price = "TP", j, tp_price
                    break
            else:
                if hi >= sl_price:
                    outcome, exit_idx, exit_price = "SL", j, sl_price
                    break
                if lo <= tp_price:
                    outcome, exit_idx, exit_price = "TP", j, tp_price
                    break

        if exit_idx is None:
            outcome = "TIMEOUT"
            exit_idx = min(end_idx - 1, n - 1)
            exit_price = float(df15["close"].iloc[exit_idx])

        sign = 1 if direction == "BUY" else -1
        pips_gained = (exit_price - entry_price) * sign / pip
        trades.append({
            "entry_ts": df15.index[i],
            "exit_ts": df15.index[exit_idx],
            "signal": direction,
            "entry": entry_price,
            "exit": exit_price,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "pips": pips_gained,
            "outcome": outcome,
            "bars_held": exit_idx - i,
            "confidence": signal.get("confidence"),
            "wick_ratio": signal.get("strategy_indicators", {}).get("wick_ratio"),
            "rr_ratio": signal.get("strategy_indicators", {}).get("rr_ratio"),
        })
        # Respect the strategy's cooldown; skip ahead by cooldown-in-bars.
        cooldown_min = cfg.get_pair_signal_cooldown_minutes(epic)
        cooldown_bars = max(1, cooldown_min // 15)
        i = max(exit_idx, i) + cooldown_bars + 1

    return pd.DataFrame(trades)


def aggregate(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {"n": 0, "wr": None, "pf": None,
                "total_pips": 0.0, "avg_win": None, "avg_loss": None,
                "rr_realized": None,
                "sl_count": 0, "tp_count": 0, "timeout_count": 0}
    wins = trades[trades["pips"] > 0]
    losses = trades[trades["pips"] < 0]
    avg_win = float(wins["pips"].mean()) if not wins.empty else None
    avg_loss = float(losses["pips"].mean()) if not losses.empty else None
    if avg_win is not None and avg_loss is not None and avg_loss < 0:
        rr_realized = abs(avg_win / avg_loss)
    else:
        rr_realized = None
    pf = (wins["pips"].sum() / -losses["pips"].sum()) if not losses.empty else float("inf")
    return {
        "n": int(len(trades)),
        "wr": float((trades["pips"] > 0).mean()),
        "pf": float(pf) if np.isfinite(pf) else None,
        "total_pips": float(trades["pips"].sum()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_realized": rr_realized,
        "sl_count": int((trades["outcome"] == "SL").sum()),
        "tp_count": int((trades["outcome"] == "TP").sum()),
        "timeout_count": int((trades["outcome"] == "TIMEOUT").sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    pairs = resolve_epics(args.pairs)
    if not pairs:
        print("[eval_rs] no pairs to evaluate", file=sys.stderr)
        return 1

    conn = psycopg2.connect(FOREX_DB)

    if args.preflight:
        preflight(conn, pairs, args.days)
        return 0

    # Load + log config once (so operators see what's under test)
    cfg = get_range_structure_config()
    print(f"[eval_rs] v{cfg.strategy_version} | days={args.days} | "
          f"ADX<={cfg.adx_hard_ceiling_primary}/{cfg.adx_hard_ceiling_htf} | "
          f"wick>={cfg.rejection_wick_ratio} sweep>={cfg.sweep_penetration_pips}p | "
          f"SL[{cfg.sl_pips_min},{cfg.sl_pips_max}]p TP[{cfg.tp_pips_min},"
          f"{cfg.tp_pips_max}]p R:R>={cfg.min_rr_ratio} | "
          f"pairs={','.join(s for s, _ in pairs)}")
    if args.canonical:
        print("[eval_rs] --canonical flag set (regime recompute table is informational "
              "only in v1.0; strategy still runs its own hard ADX gates).")

    # Shared strategy instance (cooldowns are reset per-pair inside walk_and_simulate)
    strategy = RangeStructureStrategy(config=cfg)

    all_trades: List[pd.DataFrame] = []
    per_pair_rows = []
    for short, epic in pairs:
        if args.canonical:
            # best-effort load, purely informational
            _ = try_canonical_regime_table(conn, epic, args.days)

        _, df15, df1h = load_candles(conn, epic, args.days)
        if df15.empty or df1h.empty:
            print(f"[eval_rs] {short}: no data — skipped")
            continue

        trades = walk_and_simulate(
            strategy=strategy,
            df15=df15,
            df1h=df1h,
            epic=epic,
            short=short,
            max_bars_held=args.max_bars_held,
        )
        agg = aggregate(trades)
        agg["pair"] = short
        per_pair_rows.append(agg)
        if not trades.empty:
            trades["epic"] = epic
            all_trades.append(trades)

    if not per_pair_rows:
        print("[eval_rs] no results")
        return 1

    per_pair_df = pd.DataFrame(per_pair_rows)[
        ["pair", "n", "wr", "pf", "total_pips", "avg_win", "avg_loss",
         "rr_realized", "sl_count", "tp_count", "timeout_count"]
    ]
    print("\n" + "=" * 80)
    print("PER-PAIR RESULTS")
    print("=" * 80)
    # Safe formatting (avoid .format crash on None)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    print(per_pair_df.to_string(index=False))

    portfolio_agg: Dict = {}
    if all_trades:
        merged = pd.concat(all_trades, ignore_index=True)
        portfolio_agg = aggregate(merged)
        print("\n" + "=" * 80)
        print("PORTFOLIO")
        print("=" * 80)
        pf_v = portfolio_agg['pf']
        rr_v = portfolio_agg['rr_realized']
        aw_v = portfolio_agg['avg_win']
        al_v = portfolio_agg['avg_loss']
        pf_s = 'inf' if pf_v is None else f"{pf_v:.3f}"
        rr_s = '-' if rr_v is None else f"{rr_v:.2f}"
        aw_s = '-' if aw_v is None else f"{aw_v:.1f}p"
        al_s = '-' if al_v is None else f"{al_v:.1f}p"
        print(
            f"n={portfolio_agg['n']} "
            f"WR={portfolio_agg['wr']:.2%} "
            f"PF={pf_s} "
            f"RR_realized={rr_s} "
            f"total_pips={portfolio_agg['total_pips']:.1f} "
            f"avg_win={aw_s} "
            f"avg_loss={al_s}"
        )

    # Write JSON artefact
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_json) if args.output_json else \
        results_dir / f"range_structure_eval_{timestamp}.json"
    payload = {
        "strategy": "RANGE_STRUCTURE",
        "version": cfg.strategy_version,
        "run_at": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "config_snapshot": {
            "adx_hard_ceiling_primary": cfg.adx_hard_ceiling_primary,
            "adx_hard_ceiling_htf": cfg.adx_hard_ceiling_htf,
            "sweep_penetration_pips": cfg.sweep_penetration_pips,
            "rejection_wick_ratio": cfg.rejection_wick_ratio,
            "sl_pips_min": cfg.sl_pips_min,
            "sl_pips_max": cfg.sl_pips_max,
            "tp_pips_min": cfg.tp_pips_min,
            "tp_pips_max": cfg.tp_pips_max,
            "min_rr_ratio": cfg.min_rr_ratio,
        },
        "per_pair": per_pair_rows,
        "portfolio": portfolio_agg,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n[eval_rs] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
