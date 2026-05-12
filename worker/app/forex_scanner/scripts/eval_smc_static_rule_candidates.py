#!/usr/bin/env python3
"""Evaluate simple SMC_SIMPLE EURUSD rule candidates across stored executions.

The purpose is anti-overfit screening. A rule must be explainable and improve
multiple backtest windows before it is worth promoting into strategy config.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import psycopg2
import psycopg2.extras


DEFAULT_DSN = os.getenv(
    "TRADING_DSN",
    "host=postgres dbname=forex user=postgres password=postgres",
)
DEFAULT_EPIC = "CS.D.EURUSD.CEEM.IP"


@dataclass(frozen=True)
class Signal:
    execution_id: int
    direction: str
    hour: int
    confidence: float
    pips: float
    result: str
    rsi: Optional[float]
    atr_pips: Optional[float]
    macd_hist: Optional[float]
    ema21: Optional[float]
    ema50: Optional[float]
    ema200: Optional[float]


@dataclass
class Stats:
    n: int = 0
    wins: int = 0
    pips: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0

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


@dataclass(frozen=True)
class Rule:
    name: str
    keep: Callable[[Signal], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMC_SIMPLE rule candidates")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--epic", default=DEFAULT_EPIC)
    parser.add_argument("--execution-id", type=int, action="append", dest="execution_ids")
    parser.add_argument("--latest", type=int, default=0, help="Use latest N completed SMC_SIMPLE executions for epic")
    parser.add_argument("--min-trades", type=int, default=40)
    return parser.parse_args()


def _num(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_execution_ids(conn, epic: str, count: int) -> List[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id
            FROM backtest_executions e
            JOIN backtest_signals s ON s.execution_id = e.id
            WHERE e.strategy_name = 'SMC_SIMPLE'
              AND e.status = 'completed'
              AND s.epic = %s
              AND s.trade_result IN ('win', 'loss')
            GROUP BY e.id, e.start_time
            ORDER BY e.start_time DESC
            LIMIT %s
            """,
            (epic, count),
        )
        return [int(row[0]) for row in cur.fetchall()]


def load_signals(conn, execution_ids: Iterable[int], epic: str) -> List[Signal]:
    ids = list(execution_ids)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT execution_id, signal_type, extract(hour from signal_timestamp)::int AS hour_utc,
                   confidence_score, pips_gained, trade_result, indicator_values
            FROM backtest_signals
            WHERE execution_id = ANY(%s)
              AND epic = %s
              AND trade_result IN ('win', 'loss')
              AND pips_gained IS NOT NULL
            ORDER BY execution_id, signal_timestamp, id
            """,
            (ids, epic),
        )
        rows = cur.fetchall()

    signals: List[Signal] = []
    for row in rows:
        indicators = row.get("indicator_values") or {}
        atr = _num(indicators.get("atr"))
        signals.append(
            Signal(
                execution_id=int(row["execution_id"]),
                direction=str(row["signal_type"]).upper(),
                hour=int(row["hour_utc"]),
                confidence=float(row["confidence_score"]),
                pips=float(row["pips_gained"]),
                result=str(row["trade_result"]),
                rsi=_num(indicators.get("rsi")),
                atr_pips=atr * 10000 if atr is not None else None,
                macd_hist=_num(indicators.get("macd_histogram")),
                ema21=_num(indicators.get("ema_21")),
                ema50=_num(indicators.get("ema_50")),
                ema200=_num(indicators.get("ema_200")),
            )
        )
    return signals


def aligned_macd(s: Signal) -> bool:
    if s.macd_hist is None:
        return False
    return (s.direction == "BULL" and s.macd_hist > 0) or (s.direction == "BEAR" and s.macd_hist < 0)


def aligned_ema21_50(s: Signal) -> bool:
    if s.ema21 is None or s.ema50 is None:
        return False
    return (s.direction == "BULL" and s.ema21 > s.ema50) or (s.direction == "BEAR" and s.ema21 < s.ema50)


def not_late_ny_bull(s: Signal) -> bool:
    return not (s.direction == "BULL" and 14 <= s.hour <= 18)


def build_rules() -> List[Rule]:
    rules: List[Rule] = [Rule("baseline", lambda s: True)]

    for min_conf in (0.50, 0.55, 0.60):
        rules.append(Rule(f"bull_min_conf_{min_conf:.2f}", lambda s, c=min_conf: s.direction != "BULL" or s.confidence >= c))

    rules.extend(
        [
            Rule("bull_block_14_18", not_late_ny_bull),
            Rule("bull_min_conf_055_block_14_18", lambda s: (s.direction != "BULL" or s.confidence >= 0.55) and not_late_ny_bull(s)),
            Rule("bull_min_conf_060_block_14_18", lambda s: (s.direction != "BULL" or s.confidence >= 0.60) and not_late_ny_bull(s)),
            Rule("require_macd_alignment", aligned_macd),
            Rule("require_ema21_50_alignment", aligned_ema21_50),
            Rule("bull_ema_align_bear_any", lambda s: s.direction == "BEAR" or aligned_ema21_50(s)),
            Rule("bull_conf060_bear_atr4", lambda s: (s.direction == "BULL" and s.confidence >= 0.60) or (s.direction == "BEAR" and (s.atr_pips or 0) >= 4.0)),
            Rule("bull_conf060_no_14_18_bear_atr4", lambda s: ((s.direction == "BULL" and s.confidence >= 0.60 and not_late_ny_bull(s)) or (s.direction == "BEAR" and (s.atr_pips or 0) >= 4.0))),
            Rule("block_low_atr", lambda s: (s.atr_pips or 0) >= 3.0),
            Rule("block_bull_atr_ge5", lambda s: s.direction != "BULL" or (s.atr_pips or 0) < 5.0),
            Rule("block_bull_rsi_45_65", lambda s: s.direction != "BULL" or not (45 <= (s.rsi or 50) < 65)),
            Rule("bear_only", lambda s: s.direction == "BEAR"),
        ]
    )
    return rules


def summarize(signals: Iterable[Signal]) -> Stats:
    stats = Stats()
    for signal in signals:
        stats.n += 1
        stats.pips += signal.pips
        if signal.pips > 0:
            stats.wins += 1
            stats.gross_win += signal.pips
        else:
            stats.gross_loss += signal.pips
    return stats


def main() -> int:
    args = parse_args()
    conn = psycopg2.connect(args.dsn)
    try:
        execution_ids = args.execution_ids
        if not execution_ids:
            execution_ids = latest_execution_ids(conn, args.epic, args.latest or 5)
        signals = load_signals(conn, execution_ids, args.epic)
    finally:
        conn.close()

    if not signals:
        raise SystemExit("No signals loaded")

    by_execution: Dict[int, List[Signal]] = {}
    for signal in signals:
        by_execution.setdefault(signal.execution_id, []).append(signal)

    print(f"Epic: {args.epic}")
    print(f"Executions: {', '.join(str(x) for x in sorted(by_execution))}")
    print()

    scored = []
    for rule in build_rules():
        per_exec = []
        valid = True
        for execution_id, exec_signals in sorted(by_execution.items()):
            kept = [signal for signal in exec_signals if rule.keep(signal)]
            stats = summarize(kept)
            if stats.n < args.min_trades:
                valid = False
            per_exec.append((execution_id, stats))
        total = summarize(signal for signal in signals if rule.keep(signal))
        scored.append((valid, total.pips, total.pf, rule, per_exec, total))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    for valid, _, _, rule, per_exec, total in scored:
        marker = "OK" if valid else "LOW_N"
        print(
            f"{marker:5s} {rule.name:34s} total n={total.n:4d} "
            f"wr={total.wr:5.1f}% pf={total.pf:5.2f} exp={total.exp:6.2f} pips={total.pips:8.1f}"
        )
        for execution_id, stats in per_exec:
            print(
                f"      exec={execution_id:<5d} n={stats.n:4d} wr={stats.wr:5.1f}% "
                f"pf={stats.pf:5.2f} exp={stats.exp:6.2f} pips={stats.pips:8.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
