#!/usr/bin/env python3
"""Evaluate adaptive bucket gating plus conservative SL/TP replay.

Uses stored backtest_signals and replays signals chronologically. If both a
candidate SL and TP were touched, the simulation assumes SL first. This is
intentionally conservative because the stored MFE/MAE does not preserve exact
intra-trade path ordering.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import psycopg2
import psycopg2.extras


DEFAULT_DSN = os.getenv(
    "TRADING_DSN",
    "host=postgres dbname=forex user=postgres password=postgres",
)
DEFAULT_EPIC = "CS.D.EURUSD.CEEM.IP"


@dataclass(frozen=True)
class Signal:
    id: int
    execution_id: int
    timestamp: datetime
    direction: str
    confidence: float
    pips: float
    result: str
    mfe: Optional[float]
    mae: Optional[float]


@dataclass(frozen=True)
class Gate:
    name: str
    window: int
    min_trades: int
    min_expectancy: float
    min_wr: float
    pause_hours: int
    probe_after_hours: int
    max_confidence: Optional[float] = None


@dataclass
class State:
    recent: Deque[float]
    paused_until: Optional[datetime] = None


@dataclass
class Stats:
    n: int = 0
    wins: int = 0
    pips: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        self.pips += value
        if value > 0:
            self.wins += 1
            self.gross_win += value
        else:
            self.gross_loss += value

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
    parser = argparse.ArgumentParser(description="Evaluate adaptive gate + SL/TP grid")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--epic", default=DEFAULT_EPIC)
    parser.add_argument("--execution-id", type=int, action="append", required=True, dest="execution_ids")
    parser.add_argument("--sl-min", type=int, default=5)
    parser.add_argument("--sl-max", type=int, default=16)
    parser.add_argument("--tp-min", type=int, default=5)
    parser.add_argument("--tp-max", type=int, default=22)
    parser.add_argument("--min-trades-per-exec", type=int, default=25)
    return parser.parse_args()


def load_signals(conn, execution_ids: List[int], epic: str) -> List[Signal]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, execution_id, signal_timestamp, signal_type, confidence_score,
                   pips_gained, trade_result, max_favorable_excursion_pips,
                   max_adverse_excursion_pips
            FROM backtest_signals
            WHERE execution_id = ANY(%s)
              AND epic = %s
              AND trade_result IN ('win', 'loss')
              AND pips_gained IS NOT NULL
            ORDER BY execution_id, signal_timestamp, id
            """,
            (execution_ids, epic),
        )
        rows = cur.fetchall()

    return [
        Signal(
            id=int(row["id"]),
            execution_id=int(row["execution_id"]),
            timestamp=row["signal_timestamp"],
            direction=str(row["signal_type"]).upper(),
            confidence=float(row["confidence_score"]),
            pips=float(row["pips_gained"]),
            result=str(row["trade_result"]),
            mfe=float(row["max_favorable_excursion_pips"]) if row["max_favorable_excursion_pips"] is not None else None,
            mae=float(row["max_adverse_excursion_pips"]) if row["max_adverse_excursion_pips"] is not None else None,
        )
        for row in rows
    ]


def gate_candidates() -> List[Gate]:
    return [
        Gate("none", window=1, min_trades=999, min_expectancy=-99, min_wr=-1, pause_hours=0, probe_after_hours=0),
        Gate("adaptive_direction_w8_m6", 8, 6, -0.25, 0.45, 48, 12),
        Gate("adaptive_direction_w12_m8", 12, 8, -0.25, 0.45, 48, 12),
        Gate("adaptive_cap64_direction_w8_m6", 8, 6, -0.25, 0.45, 48, 12, max_confidence=0.64),
        Gate("adaptive_cap64_direction_w12_m8", 12, 8, -0.25, 0.45, 48, 12, max_confidence=0.64),
    ]


def should_probe(signal: Signal) -> bool:
    return (signal.id % 1000) / 1000.0 < 0.02


def gate_taken(signals: Iterable[Signal], gate: Gate) -> List[Signal]:
    if gate.name == "none":
        return list(signals)

    states: Dict[str, State] = {}
    taken: List[Signal] = []
    for signal in signals:
        state = states.setdefault(signal.direction, State(recent=deque(maxlen=gate.window)))

        if gate.max_confidence is not None and signal.confidence > gate.max_confidence:
            continue

        paused = state.paused_until is not None and signal.timestamp < state.paused_until
        if paused:
            probe_at = state.paused_until - timedelta(hours=max(gate.pause_hours - gate.probe_after_hours, 0))
            if signal.timestamp < probe_at or not should_probe(signal):
                continue

        taken.append(signal)
        state.recent.append(signal.pips)
        if len(state.recent) >= gate.min_trades:
            vals = list(state.recent)
            exp = sum(vals) / len(vals)
            wr = sum(1 for value in vals if value > 0) / len(vals)
            if exp < gate.min_expectancy and wr < gate.min_wr:
                state.paused_until = signal.timestamp + timedelta(hours=gate.pause_hours)
            elif state.paused_until is not None:
                state.paused_until = None

    return taken


def replay_exit(signal: Signal, sl: int, tp: int) -> float:
    if signal.mae is not None and signal.mae >= sl:
        return -float(sl)
    if signal.mfe is not None and signal.mfe >= tp:
        return float(tp)
    return signal.pips


def summarize(signals: Iterable[Signal], sl: int, tp: int) -> Stats:
    stats = Stats()
    for signal in signals:
        stats.add(replay_exit(signal, sl, tp))
    return stats


def main() -> int:
    args = parse_args()
    conn = psycopg2.connect(args.dsn)
    try:
        all_signals = load_signals(conn, args.execution_ids, args.epic)
    finally:
        conn.close()

    by_exec: Dict[int, List[Signal]] = defaultdict(list)
    for signal in all_signals:
        by_exec[signal.execution_id].append(signal)

    rows = []
    for gate in gate_candidates():
        gated_by_exec = {execution_id: gate_taken(signals, gate) for execution_id, signals in by_exec.items()}
        for sl in range(args.sl_min, args.sl_max + 1):
            for tp in range(args.tp_min, args.tp_max + 1):
                per_exec = [(execution_id, summarize(signals, sl, tp)) for execution_id, signals in sorted(gated_by_exec.items())]
                if any(stats.n < args.min_trades_per_exec for _, stats in per_exec):
                    continue
                total = Stats()
                for _, stats in per_exec:
                    total.n += stats.n
                    total.wins += stats.wins
                    total.pips += stats.pips
                    total.gross_win += stats.gross_win
                    total.gross_loss += stats.gross_loss
                rows.append((total.pips, total.pf, gate, sl, tp, total, per_exec))

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    print(f"Epic: {args.epic}")
    print(f"Executions: {', '.join(str(x) for x in sorted(by_exec))}")
    for _, _, gate, sl, tp, total, per_exec in rows[:20]:
        print(
            f"{gate.name:34s} SL={sl:2d} TP={tp:2d} "
            f"n={total.n:4d} wr={total.wr:5.1f}% pf={total.pf:5.2f} "
            f"exp={total.exp:6.2f} pips={total.pips:8.1f}"
        )
        for execution_id, stats in per_exec:
            print(
                f"      exec={execution_id:<5d} n={stats.n:4d} wr={stats.wr:5.1f}% "
                f"pf={stats.pf:5.2f} exp={stats.exp:6.2f} pips={stats.pips:8.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
