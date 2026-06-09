"""Shadow-P&L reconstruction for paused cells (Phase 3, propose-only).

A paused cell keeps emitting monitor-only signals into ``alert_history``, but
those rows can't be trusted for outcomes (price = 0 ~60% of the time, no SL/TP
stamped). So we RECONSTRUCT each signal's outcome:

    alert_timestamp + epic  ->  entry price from ig_candles (1m base)
                            ->  apply the cell's fixed SL/TP pips
                            ->  walk forward over 1m candles to HIT_TP / HIT_SL

and compute a rolling shadow profit factor from the resolved outcomes.

IMPORTANT CAVEAT (deliberate for Phase A): this counts reconstructed *signals*,
not validated *trade-equivalents* — a paused cell logs every signal, including
ones that live cooldown / LPF / validation would have dropped (the divergence
can be 2-13x). So shadow counts run optimistic. That is acceptable while resume
is propose-only — surfacing exactly this kind of issue is the point of the
staged rollout — but it must be tightened before fully-auto resume (Phase C).

Strategies without a readable fixed SL/TP (e.g. ATR-based SMC_MOMENTUM) return
(None, None) from ``get_cell_sl_tp`` and are skipped, logged, by the caller.
"""
from __future__ import annotations

import bisect
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

from .adapters import StrategyAdapter, build_scope_clause
from .config import forex_dsn, strategy_config_dsn
from .evaluator import PerfStats, evaluate_performance

logger = logging.getLogger(__name__)


def pip_value(epic: str) -> float:
    """Price movement per pip. JPY -> 0.01, gold -> 0.1, else -> 0.0001."""
    e = (epic or "").upper()
    if "JPY" in e:
        return 0.01
    if "GOLD" in e or "CFEGOLD" in e:
        return 0.1
    return 0.0001


def _naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Drop tzinfo so DB-aware and code-naive datetimes compare cleanly."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def is_long(signal_type: Any) -> Optional[bool]:
    s = str(signal_type or "").upper()
    if s in ("BULL", "BUY", "LONG"):
        return True
    if s in ("BEAR", "SELL", "SHORT"):
        return False
    return None


def simulate_outcome(
    entry_price: float,
    long: bool,
    sl_pips: float,
    tp_pips: float,
    pip: float,
    forward_candles: List[Dict[str, Any]],
) -> Optional[float]:
    """Walk candles after entry; return +tp_pips / -sl_pips, or None if neither
    hit within the window. If a single candle spans both levels, resolve
    conservatively as the stop (worst case). Pure function (no DB)."""
    if long:
        tp = entry_price + tp_pips * pip
        sl = entry_price - sl_pips * pip
        for c in forward_candles:
            hit_sl = c["low"] <= sl
            hit_tp = c["high"] >= tp
            if hit_sl and hit_tp:
                return -sl_pips
            if hit_tp:
                return tp_pips
            if hit_sl:
                return -sl_pips
    else:
        tp = entry_price - tp_pips * pip
        sl = entry_price + sl_pips * pip
        for c in forward_candles:
            hit_sl = c["high"] >= sl
            hit_tp = c["low"] <= tp
            if hit_sl and hit_tp:
                return -sl_pips
            if hit_tp:
                return tp_pips
            if hit_sl:
                return -sl_pips
    return None


def get_cell_sl_tp(
    adapter: StrategyAdapter, epic: str, config_set: str,
    *, dsn: Optional[str] = None, conn: Any = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve fixed SL/TP pips for a cell: per-pair override first, then the
    strategy's global config. (None, None) if not fixed-SL/TP (e.g. ATR-based)."""
    own = conn is None
    if own:
        conn = psycopg2.connect(dsn or strategy_config_dsn())
    try:
        scope_sql, scope_params = build_scope_clause(adapter, epic, config_set)
        candidates = [
            (f"SELECT fixed_stop_loss_pips, fixed_take_profit_pips "
             f"FROM {adapter.table} WHERE {scope_sql} LIMIT 1", scope_params),
        ]
        gtable = adapter.table.replace("_pair_overrides", "_global_config")
        candidates.append(
            (f"SELECT fixed_stop_loss_pips, fixed_take_profit_pips FROM {gtable} "
             f"WHERE config_set = %s AND is_active = TRUE LIMIT 1", [config_set])
        )
        candidates.append(
            (f"SELECT fixed_stop_loss_pips, fixed_take_profit_pips FROM {gtable} "
             f"WHERE is_active = TRUE LIMIT 1", [])
        )
        for query, params in candidates:
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    row = cur.fetchone()
                if row and row[0] is not None and row[1] is not None:
                    return float(row[0]), float(row[1])
            except Exception:
                conn.rollback()  # missing column/table — try next candidate
                continue
        return None, None
    finally:
        if own:
            conn.close()


def _load_candles(epic: str, since_ts: datetime, conn: Any) -> List[Dict[str, Any]]:
    query = """
        SELECT start_time, open, high, low, close
        FROM ig_candles
        WHERE epic = %s AND timeframe = 1 AND start_time >= %s
        ORDER BY start_time ASC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, [epic, _naive(since_ts)])
        return [dict(r) for r in cur.fetchall()]


def _load_monitor_signals(
    strategy: str, epic: str, environment: str, since_ts: datetime, conn: Any
) -> List[Dict[str, Any]]:
    query = """
        SELECT alert_timestamp, signal_type
        FROM alert_history
        WHERE strategy = %s AND epic = %s AND alert_timestamp > %s
          AND coalesce(environment, %s) = %s
        ORDER BY alert_timestamp ASC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, [strategy, epic, _naive(since_ts), environment, environment])
        return [dict(r) for r in cur.fetchall()]


def compute_shadow_stats(
    adapter: StrategyAdapter,
    strategy: str,
    epic: str,
    config_set: str,
    since_ts: datetime,
    *,
    max_hold_hours: int = 48,
    forex_conn: Any = None,
    sc_conn: Any = None,
) -> Tuple[Optional[PerfStats], int, int]:
    """Reconstruct shadow outcomes for a paused cell's monitor-only signals
    since ``since_ts``. Returns (stats, n_signals, n_resolved). stats is None if
    SL/TP can't be resolved (cell skipped)."""
    sl_pips, tp_pips = get_cell_sl_tp(adapter, epic, config_set, conn=sc_conn)
    if sl_pips is None or tp_pips is None:
        logger.debug("[AutoPause] no fixed SL/TP for %s %s — shadow eval skipped",
                     strategy, epic)
        return None, 0, 0

    pip = pip_value(epic)
    own = forex_conn is None
    conn = forex_conn or psycopg2.connect(forex_dsn())
    try:
        candles = _load_candles(epic, since_ts, conn)
        signals = _load_monitor_signals(strategy, epic, config_set, since_ts, conn)
    finally:
        if own:
            conn.close()

    if not candles or not signals:
        return evaluate_performance([]), len(signals), 0

    times = [_naive(c["start_time"]) for c in candles]
    max_bars = max_hold_hours * 60
    outcomes: List[Dict[str, Any]] = []
    for sig in signals:
        long = is_long(sig.get("signal_type"))
        if long is None:
            continue
        sig_ts = _naive(sig.get("alert_timestamp"))
        if sig_ts is None:
            continue
        idx = bisect.bisect_left(times, sig_ts)
        if idx >= len(candles):
            continue  # no candle data at/after the signal yet
        entry_price = float(candles[idx]["open"])
        forward = candles[idx: idx + max_bars]
        pnl = simulate_outcome(entry_price, long, sl_pips, tp_pips, pip, forward)
        if pnl is not None:
            outcomes.append({"profit_loss": pnl, "pips_gained": None})

    # evaluate_performance expects most-recent-first for the streak; reverse.
    outcomes.reverse()
    stats = evaluate_performance(outcomes)
    return stats, len(signals), len(outcomes)
