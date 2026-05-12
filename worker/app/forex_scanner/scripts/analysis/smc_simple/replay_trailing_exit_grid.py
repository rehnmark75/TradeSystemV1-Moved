#!/usr/bin/env python3
"""Replay stored backtest signals through BacktestTrailingEngine exit variants."""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import psycopg2
import psycopg2.extras

try:
    from forex_scanner.core.backtesting.backtest_trailing_engine import BacktestTrailingEngine
except ImportError:
    from core.backtesting.backtest_trailing_engine import BacktestTrailingEngine


DEFAULT_DSN = os.getenv(
    "TRADING_DSN",
    "host=postgres dbname=forex user=postgres password=postgres",
)


@dataclass(frozen=True)
class Signal:
    id: int
    timestamp: Any
    direction: str
    entry_price: float
    risk_pips: float
    reward_pips: float
    confidence: float
    stored_pips: float
    stored_exit: str


@dataclass
class Stats:
    n: int = 0
    wins: int = 0
    pips: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    profit_targets: int = 0
    trailing_stops: int = 0
    stop_losses: int = 0

    def add(self, pips: float, exit_reason: str) -> None:
        self.n += 1
        self.pips += pips
        if pips > 0:
            self.wins += 1
            self.gross_win += pips
        else:
            self.gross_loss += pips

        reason = (exit_reason or "").upper()
        if reason == "PROFIT_TARGET":
            self.profit_targets += 1
        elif reason == "TRAILING_STOP":
            self.trailing_stops += 1
        elif reason == "STOP_LOSS":
            self.stop_losses += 1

    @property
    def wr(self) -> float:
        return 100.0 * self.wins / self.n if self.n else 0.0

    @property
    def exp(self) -> float:
        return self.pips / self.n if self.n else 0.0

    @property
    def pf(self) -> float:
        if self.gross_loss == 0:
            return 999.0 if self.gross_win > 0 else 0.0
        return self.gross_win / abs(self.gross_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay stored signals with trailing exit grids")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--execution-id", type=int, required=True)
    parser.add_argument("--epic", default="CS.D.EURJPY.MINI.IP")
    parser.add_argument("--timeframe", type=int, default=5)
    parser.add_argument("--max-bars", type=int, default=200)
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args()


def load_signals(conn, execution_id: int, epic: str) -> List[Signal]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, signal_timestamp, signal_type, entry_price, confidence_score,
                   pips_gained, exit_reason,
                   ABS(entry_price - stop_loss_price) * CASE WHEN %s ILIKE '%%JPY%%' THEN 100 ELSE 10000 END AS risk_pips,
                   ABS(take_profit_price - entry_price) * CASE WHEN %s ILIKE '%%JPY%%' THEN 100 ELSE 10000 END AS reward_pips
            FROM backtest_signals
            WHERE execution_id = %s
              AND epic = %s
              AND pips_gained IS NOT NULL
            ORDER BY signal_timestamp, id
            """,
            (epic, epic, execution_id, epic),
        )
        rows = cur.fetchall()

    return [
        Signal(
            id=int(row["id"]),
            timestamp=row["signal_timestamp"],
            direction=str(row["signal_type"]).upper(),
            entry_price=float(row["entry_price"]),
            risk_pips=float(row["risk_pips"] or 0.0),
            reward_pips=float(row["reward_pips"] or 0.0),
            confidence=float(row["confidence_score"]),
            stored_pips=float(row["pips_gained"]),
            stored_exit=str(row["exit_reason"] or ""),
        )
        for row in rows
    ]


def load_future_frames(conn, signals: Iterable[Signal], epic: str, timeframe: int, max_bars: int) -> Dict[int, pd.DataFrame]:
    frames: Dict[int, pd.DataFrame] = {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for signal in signals:
            end_time = signal.timestamp + timedelta(minutes=timeframe * (max_bars + 2))
            cur.execute(
                """
                SELECT start_time AS timestamp, open, high, low, close, volume
                FROM ig_candles_backtest
                WHERE epic = %s
                  AND timeframe = %s
                  AND start_time > %s
                  AND start_time <= %s
                ORDER BY start_time
                LIMIT %s
                """,
                (epic, timeframe, signal.timestamp, end_time, max_bars + 2),
            )
            rows = cur.fetchall()
            frames[signal.id] = pd.DataFrame(rows)
    return frames


def variant_grid() -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []

    def add(name: str, **overrides: Any) -> None:
        variants.append({"name": name, "overrides": overrides})

    add("current")
    for s1_trig, s1_lock, s2_trig, s2_lock in itertools.product(
        [15, 18, 20, 22],
        [8, 10, 12, 14],
        [20, 24, 28],
        [12, 15, 18, 20],
    ):
        if s2_trig <= s1_trig or s2_lock <= s1_lock:
            continue
        add(
            f"s1_{s1_trig}_{s1_lock}_s2_{s2_trig}_{s2_lock}",
            stage1_trigger_points=s1_trig,
            stage1_lock_points=s1_lock,
            stage2_trigger_points=s2_trig,
            stage2_lock_points=s2_lock,
        )

    for be in [10, 12, 14, 16]:
        add(f"be_{be}", early_breakeven_trigger_points=be, break_even_trigger_points=be)

    for check_bars, min_mfe, stop in itertools.product([2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]):
        add(
            f"efs_b{check_bars}_mfe{min_mfe}_stop{stop}",
            early_failure_stop_enabled=True,
            early_failure_check_bars=check_bars,
            early_failure_min_mfe_pips=min_mfe,
            early_failure_stop_pips=stop,
        )

    return variants


def replay_variant(
    variant: Dict[str, Any],
    signals: List[Signal],
    frames: Dict[int, pd.DataFrame],
    epic: str,
    max_bars: int,
) -> Stats:
    engine = BacktestTrailingEngine(
        epic=epic,
        is_scalp_trade=True,
        config_override=variant["overrides"],
        max_bars=max_bars,
        strategy="SMC_SIMPLE",
    )
    stats = Stats()
    for signal in signals:
        df = frames.get(signal.id)
        if df is None or df.empty:
            continue
        payload = {
            "epic": epic,
            "signal_timestamp": signal.timestamp,
            "signal_type": signal.direction,
            "entry_price": signal.entry_price,
            "risk_pips": signal.risk_pips,
            "reward_pips": signal.reward_pips,
            "confidence_score": signal.confidence,
            "strategy": "SMC_SIMPLE",
        }
        result = engine.simulate_trade(payload, df, signal_idx=0)
        stats.add(float(result.get("pips_gained") or 0.0), str(result.get("exit_reason") or ""))
    return stats


def main() -> int:
    args = parse_args()
    conn = psycopg2.connect(args.dsn)
    try:
        signals = load_signals(conn, args.execution_id, args.epic)
        frames = load_future_frames(conn, signals, args.epic, args.timeframe, args.max_bars)
    finally:
        conn.close()

    baseline = Stats()
    for signal in signals:
        baseline.add(signal.stored_pips, signal.stored_exit)

    rows = []
    for variant in variant_grid():
        stats = replay_variant(variant, signals, frames, args.epic, args.max_bars)
        rows.append((stats.pips, stats.pf, variant["name"], variant["overrides"], stats))

    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    print(f"Epic: {args.epic}")
    print(f"Execution: {args.execution_id}")
    print(
        "Stored: "
        f"n={baseline.n} wr={baseline.wr:.1f}% pf={baseline.pf:.2f} "
        f"exp={baseline.exp:.2f} pips={baseline.pips:.1f} "
        f"tp={baseline.profit_targets} trail={baseline.trailing_stops} sl={baseline.stop_losses}"
    )
    for _, _, name, overrides, stats in rows[: args.top]:
        print(
            f"{name:28s} n={stats.n:3d} wr={stats.wr:5.1f}% pf={stats.pf:5.2f} "
            f"exp={stats.exp:6.2f} pips={stats.pips:7.1f} "
            f"tp={stats.profit_targets:2d} trail={stats.trailing_stops:2d} sl={stats.stop_losses:2d} "
            f"overrides={overrides}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
