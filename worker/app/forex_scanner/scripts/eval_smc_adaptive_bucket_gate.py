#!/usr/bin/env python3
"""Evaluate adaptive SMC_SIMPLE bucket gates on stored backtest signals.

The goal is to test dynamic gating before changing live strategy logic:

- Buckets are coarse, interpretable groups such as direction + UTC session.
- A bucket is paused only after enough recent trades show negative expectancy.
- Paused buckets are not permanently disabled; they reopen through timed probes.
- Results are replayed chronologically to avoid lookahead.

Run inside task-worker:

    docker exec -i task-worker python /app/forex_scanner/scripts/eval_smc_adaptive_bucket_gate.py \
        --execution-id 6516 --epic CS.D.EURUSD.CEEM.IP
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
    timestamp: datetime
    direction: str
    confidence: float
    pips: float
    result: str
    rsi: Optional[float]
    atr_pips: Optional[float]
    macd_histogram: Optional[float]


@dataclass
class GateConfig:
    name: str
    bucket_mode: str
    window: int
    min_trades: int
    min_expectancy_pips: float
    min_win_rate: float
    pause_hours: int
    probe_after_hours: int
    max_confidence: Optional[float] = None
    blocked_hours: Tuple[int, ...] = ()
    exploration_rate: float = 0.0


@dataclass
class BucketState:
    recent: Deque[float]
    paused_until: Optional[datetime] = None
    probes: int = 0
    pauses: int = 0


@dataclass
class EvalResult:
    config: GateConfig
    total: int = 0
    taken: int = 0
    skipped: int = 0
    wins: int = 0
    losses: int = 0
    pips: float = 0.0
    gross_win_pips: float = 0.0
    gross_loss_pips: float = 0.0
    pauses: int = 0
    probes: int = 0
    skipped_by_reason: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bucket_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return 100.0 * self.wins / self.taken if self.taken else 0.0

    @property
    def expectancy(self) -> float:
        return self.pips / self.taken if self.taken else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss_pips == 0:
            return 999.0 if self.gross_win_pips > 0 else 0.0
        return self.gross_win_pips / abs(self.gross_loss_pips)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adaptive SMC bucket gates")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--execution-id", type=int, default=None)
    parser.add_argument("--epic", default=DEFAULT_EPIC)
    parser.add_argument("--latest", action="store_true", help="Use latest completed SMC_SIMPLE execution for epic")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_latest_execution_id(conn, epic: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id
            FROM backtest_executions e
            JOIN backtest_signals s ON s.execution_id = e.id
            WHERE e.strategy_name = 'SMC_SIMPLE'
              AND e.status = 'completed'
              AND s.epic = %s
              AND s.trade_result IS NOT NULL
            GROUP BY e.id, e.start_time
            ORDER BY e.start_time DESC
            LIMIT 1
            """,
            (epic,),
        )
        row = cur.fetchone()
    if not row:
        raise SystemExit(f"No completed SMC_SIMPLE execution found for {epic}")
    return int(row[0])


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_signals(conn, execution_id: int, epic: str, limit: Optional[int]) -> List[Signal]:
    query = """
        SELECT id, signal_timestamp, signal_type, confidence_score, pips_gained,
               trade_result, indicator_values
        FROM backtest_signals
        WHERE execution_id = %s
          AND epic = %s
          AND trade_result IN ('win', 'loss')
          AND pips_gained IS NOT NULL
        ORDER BY signal_timestamp ASC, id ASC
    """
    params: List[Any] = [execution_id, epic]
    if limit:
        query += " LIMIT %s"
        params.append(limit)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    signals: List[Signal] = []
    for row in rows:
        indicators = row.get("indicator_values") or {}
        if isinstance(indicators, str):
            indicators = json.loads(indicators)
        atr = _num(indicators.get("atr"))
        signals.append(
            Signal(
                id=int(row["id"]),
                timestamp=row["signal_timestamp"],
                direction=str(row["signal_type"]).upper(),
                confidence=float(row["confidence_score"]),
                pips=float(row["pips_gained"]),
                result=str(row["trade_result"]),
                rsi=_num(indicators.get("rsi")),
                atr_pips=atr * 10000 if atr is not None else None,
                macd_histogram=_num(indicators.get("macd_histogram")),
            )
        )
    return signals


def utc_session(hour: int) -> str:
    if 7 <= hour <= 11:
        return "london_am"
    if 12 <= hour <= 14:
        return "overlap"
    if 15 <= hour <= 17:
        return "ny_late"
    if 18 <= hour <= 21:
        return "us_close"
    return "asian"


def rsi_zone(signal: Signal) -> str:
    if signal.rsi is None:
        return "rsi_unknown"
    if signal.rsi < 30:
        return "rsi_lt30"
    if signal.rsi < 45:
        return "rsi_30_45"
    if signal.rsi < 55:
        return "rsi_45_55"
    if signal.rsi < 65:
        return "rsi_55_65"
    if signal.rsi < 70:
        return "rsi_65_70"
    return "rsi_70_plus"


def atr_zone(signal: Signal) -> str:
    if signal.atr_pips is None:
        return "atr_unknown"
    if signal.atr_pips < 3:
        return "atr_lt3"
    if signal.atr_pips < 5:
        return "atr_3_5"
    if signal.atr_pips < 8:
        return "atr_5_8"
    return "atr_8_plus"


def macd_alignment(signal: Signal) -> str:
    if signal.macd_histogram is None:
        return "macd_unknown"
    aligned = (
        signal.direction == "BULL" and signal.macd_histogram > 0
    ) or (
        signal.direction == "BEAR" and signal.macd_histogram < 0
    )
    return "macd_aligned" if aligned else "macd_against"


def bucket_key(signal: Signal, mode: str) -> str:
    hour = signal.timestamp.hour
    if mode == "direction":
        return signal.direction
    if mode == "direction_session":
        return f"{signal.direction}:{utc_session(hour)}"
    if mode == "direction_hour4":
        return f"{signal.direction}:h{hour // 4 * 4:02d}_{hour // 4 * 4 + 3:02d}"
    if mode == "direction_rsi":
        return f"{signal.direction}:{rsi_zone(signal)}"
    if mode == "direction_session_rsi":
        return f"{signal.direction}:{utc_session(hour)}:{rsi_zone(signal)}"
    if mode == "direction_session_macd":
        return f"{signal.direction}:{utc_session(hour)}:{macd_alignment(signal)}"
    if mode == "direction_atr":
        return f"{signal.direction}:{atr_zone(signal)}"
    raise ValueError(f"Unknown bucket_mode: {mode}")


def should_explore(signal: Signal, cfg: GateConfig) -> bool:
    if cfg.exploration_rate <= 0:
        return False
    # Deterministic pseudo-random probe: stable across runs, no imported RNG state.
    return (signal.id % 1000) / 1000.0 < cfg.exploration_rate


def eval_config(signals: Iterable[Signal], cfg: GateConfig) -> EvalResult:
    result = EvalResult(config=cfg)
    states: Dict[str, BucketState] = {}
    bucket_pips: Dict[str, List[float]] = defaultdict(list)

    for signal in signals:
        result.total += 1
        hour = signal.timestamp.hour
        key = bucket_key(signal, cfg.bucket_mode)
        state = states.setdefault(key, BucketState(recent=deque(maxlen=cfg.window)))

        if cfg.max_confidence is not None and signal.confidence > cfg.max_confidence:
            result.skipped += 1
            result.skipped_by_reason["confidence_cap"] += 1
            continue

        if hour in cfg.blocked_hours:
            result.skipped += 1
            result.skipped_by_reason["blocked_hour"] += 1
            continue

        paused = state.paused_until is not None and signal.timestamp < state.paused_until
        if paused and not should_explore(signal, cfg):
            # A timed probe can reopen after probe_after_hours even before the full pause ends.
            probe_at = state.paused_until - timedelta(hours=max(cfg.pause_hours - cfg.probe_after_hours, 0))
            if signal.timestamp < probe_at:
                result.skipped += 1
                result.skipped_by_reason["adaptive_pause"] += 1
                continue

        if paused:
            state.probes += 1
            result.probes += 1

        result.taken += 1
        if signal.result == "win":
            result.wins += 1
            result.gross_win_pips += max(signal.pips, 0.0)
        else:
            result.losses += 1
            result.gross_loss_pips += min(signal.pips, 0.0)
        result.pips += signal.pips
        state.recent.append(signal.pips)
        bucket_pips[key].append(signal.pips)

        if len(state.recent) >= cfg.min_trades:
            recent = list(state.recent)
            expectancy = sum(recent) / len(recent)
            win_rate = sum(1 for p in recent if p > 0) / len(recent)
            if expectancy < cfg.min_expectancy_pips and win_rate < cfg.min_win_rate:
                state.paused_until = signal.timestamp + timedelta(hours=cfg.pause_hours)
                state.pauses += 1
                result.pauses += 1
            elif state.paused_until is not None:
                state.paused_until = None

    for key, vals in bucket_pips.items():
        result.bucket_stats[key] = {
            "n": len(vals),
            "pips": round(sum(vals), 2),
            "expectancy": round(sum(vals) / len(vals), 2) if vals else 0.0,
            "wr": round(100.0 * sum(1 for p in vals if p > 0) / len(vals), 1) if vals else 0.0,
        }
    return result


def candidate_configs() -> List[GateConfig]:
    configs: List[GateConfig] = []
    bucket_modes = [
        "direction",
        "direction_session",
        "direction_hour4",
        "direction_rsi",
        "direction_session_macd",
    ]
    for mode in bucket_modes:
        for window, min_trades in ((8, 6), (12, 8), (20, 12)):
            configs.append(
                GateConfig(
                    name=f"adaptive:{mode}:w{window}:m{min_trades}",
                    bucket_mode=mode,
                    window=window,
                    min_trades=min_trades,
                    min_expectancy_pips=-0.25,
                    min_win_rate=0.45,
                    pause_hours=48,
                    probe_after_hours=12,
                    max_confidence=None,
                    exploration_rate=0.02,
                )
            )
            configs.append(
                GateConfig(
                    name=f"adaptive_cap64:{mode}:w{window}:m{min_trades}",
                    bucket_mode=mode,
                    window=window,
                    min_trades=min_trades,
                    min_expectancy_pips=-0.25,
                    min_win_rate=0.45,
                    pause_hours=48,
                    probe_after_hours=12,
                    max_confidence=0.64,
                    exploration_rate=0.02,
                )
            )

    configs.extend(
        [
            GateConfig(
                name="static:cap64",
                bucket_mode="direction",
                window=12,
                min_trades=8,
                min_expectancy_pips=-99,
                min_win_rate=-1,
                pause_hours=0,
                probe_after_hours=0,
                max_confidence=0.64,
            ),
            GateConfig(
                name="static:block_15_17",
                bucket_mode="direction",
                window=12,
                min_trades=8,
                min_expectancy_pips=-99,
                min_win_rate=-1,
                pause_hours=0,
                probe_after_hours=0,
                blocked_hours=(15, 16, 17),
            ),
            GateConfig(
                name="static:cap64_block_15_17",
                bucket_mode="direction",
                window=12,
                min_trades=8,
                min_expectancy_pips=-99,
                min_win_rate=-1,
                pause_hours=0,
                probe_after_hours=0,
                max_confidence=0.64,
                blocked_hours=(15, 16, 17),
            ),
        ]
    )
    return configs


def baseline(signals: List[Signal]) -> EvalResult:
    cfg = GateConfig(
        name="baseline",
        bucket_mode="direction",
        window=1,
        min_trades=99,
        min_expectancy_pips=-99,
        min_win_rate=-1,
        pause_hours=0,
        probe_after_hours=0,
    )
    return eval_config(signals, cfg)


def print_result(res: EvalResult) -> None:
    cfg = res.config
    print(
        f"{cfg.name:48s} n={res.taken:4d}/{res.total:<4d} "
        f"wr={res.win_rate:5.1f}% pf={res.profit_factor:5.2f} "
        f"exp={res.expectancy:6.2f} pips={res.pips:8.1f} "
        f"skip={res.skipped:4d} pauses={res.pauses:3d} probes={res.probes:3d} "
        f"reasons={dict(res.skipped_by_reason)}"
    )


def main() -> int:
    args = parse_args()
    conn = psycopg2.connect(args.dsn)
    try:
        execution_id = args.execution_id
        if args.latest or execution_id is None:
            execution_id = load_latest_execution_id(conn, args.epic)
        signals = load_signals(conn, execution_id, args.epic, args.limit)
    finally:
        conn.close()

    if not signals:
        raise SystemExit("No signals loaded")

    print(f"Execution: {execution_id} | Epic: {args.epic} | Signals: {len(signals)}")
    print(f"Period: {signals[0].timestamp} -> {signals[-1].timestamp}\n")

    base = baseline(signals)
    print_result(base)
    print()

    results = [eval_config(signals, cfg) for cfg in candidate_configs()]
    results.sort(key=lambda r: (r.pips, r.profit_factor, r.taken), reverse=True)

    print("Top adaptive/static gates:")
    for res in results[:15]:
        print_result(res)

    best = results[0]
    print("\nBest bucket contribution:")
    for key, stats in sorted(best.bucket_stats.items(), key=lambda kv: kv[1]["pips"]):
        print(
            f"  {key:32s} n={stats['n']:4.0f} wr={stats['wr']:5.1f}% "
            f"exp={stats['expectancy']:6.2f} pips={stats['pips']:8.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
