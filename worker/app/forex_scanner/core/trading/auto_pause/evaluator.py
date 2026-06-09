"""Rolling-performance evaluation and the trip (pause) decision.

The trip rule is the FIXED, cross-strategy-validated rule:

    pause when   rolling-last-N PF < trip_pf_threshold AND n >= trip_min_trades
            OR   consecutive losses >= trip_max_consecutive_losses

The consecutive-loss trigger is the universal safeguard: it is the only part
that fires in a useful timeframe for borderline-frequency cells, where the
rolling-PF window never fills.

Profit factor is computed from a SINGLE P&L field per evaluation, never a
per-row ``coalesce`` — summing pips and SEK into one ratio would corrupt PF.
``profit_loss`` is used in practice (``trade_log.pips_gained`` is NULL); pips are
used only if no row carries ``profit_loss``. Rows lacking the chosen field are
skipped and logged. The pure functions ``evaluate_performance`` / ``decide_trip``
take plain rows and are DB-free so they can be unit-tested directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

from .config import AutoPauseParams, forex_dsn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerfStats:
    n: int
    pf: Optional[float]          # None when there are no losing trades (PF undefined)
    gross_win: float
    gross_loss: float
    win_rate: float
    consecutive_losses: int      # leading loss streak from the most recent trade


@dataclass(frozen=True)
class TripDecision:
    should_pause: bool
    reason: str


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_value_field(rows: List[Dict[str, Any]]) -> str:
    """Choose ONE P&L field for the whole set so PF never mixes units.

    ``profit_loss`` (SEK) is the populated field in practice; ``pips_gained`` is
    used only when no row carries ``profit_loss``. Committing to a single field —
    rather than per-row coalesce — keeps the PF ratio unit-consistent.
    """
    if any(r.get("profit_loss") is not None for r in rows):
        return "profit_loss"
    return "pips_gained"


def evaluate_performance(rows: List[Dict[str, Any]]) -> PerfStats:
    """Compute PF, win rate and the recent consecutive-loss streak.

    ``rows`` MUST be ordered most-recent-first (closed_at DESC) for the
    consecutive-loss streak to be meaningful. A single P&L field is used for the
    whole set (see ``_pick_value_field``); rows missing it are skipped + logged.
    """
    field = _pick_value_field(rows)
    n = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    consecutive_losses = 0
    streak_open = True
    skipped = 0

    for row in rows:
        v = _as_float(row.get(field))
        if v is None:
            skipped += 1
            continue
        n += 1
        if v > 0:
            wins += 1
            gross_win += v
        elif v < 0:
            gross_loss += -v
        # streak: count leading losses; any non-loss (win or breakeven) ends it
        if streak_open:
            if v < 0:
                consecutive_losses += 1
            else:
                streak_open = False

    if skipped:
        logger.warning(
            "[AutoPause] evaluate_performance skipped %d row(s) missing '%s' "
            "(enforcing single-unit PF)", skipped, field,
        )

    win_rate = (wins / n) if n else 0.0
    pf: Optional[float] = (gross_win / gross_loss) if gross_loss > 0 else None
    return PerfStats(
        n=n,
        pf=pf,
        gross_win=gross_win,
        gross_loss=gross_loss,
        win_rate=win_rate,
        consecutive_losses=consecutive_losses,
    )


def decide_trip(stats: PerfStats, params: AutoPauseParams) -> TripDecision:
    """Apply the fixed trip rule to evaluated stats."""
    # Universal safeguard first (works even when the PF window is too thin).
    if stats.consecutive_losses >= params.trip_max_consecutive_losses:
        return TripDecision(
            True,
            f"{stats.consecutive_losses} consecutive losses "
            f">= {params.trip_max_consecutive_losses}",
        )
    # Rolling profit-factor trigger (needs an adequate sample to be meaningful).
    if (
        stats.n >= params.trip_min_trades
        and stats.pf is not None
        and stats.pf < params.trip_pf_threshold
    ):
        return TripDecision(
            True,
            f"PF {stats.pf:.2f} < {params.trip_pf_threshold} over last {stats.n} trades",
        )
    return TripDecision(False, "")


def load_closed_trades(
    strategy: str,
    epic: str,
    environment: str,
    window: int,
    *,
    dsn: Optional[str] = None,
    conn: Any = None,
) -> List[Dict[str, Any]]:
    """Load the last ``window`` closed trades for a (strategy, epic, env) cell.

    Returns rows ordered most-recent-first with ``pips_gained`` and
    ``profit_loss``. Mirrors the ``adaptive_bucket_gate`` query shape (forex DB,
    ``trade_log JOIN alert_history``, env via coalesce).
    """
    query = """
        SELECT t.closed_at, t.pips_gained, t.profit_loss
        FROM trade_log t
        JOIN alert_history a ON a.id = t.alert_id
        WHERE t.symbol = %s
          AND a.strategy = %s
          AND t.status = 'closed'
          AND t.closed_at IS NOT NULL
          AND coalesce(t.pips_gained, t.profit_loss) IS NOT NULL
          AND coalesce(t.environment, a.environment, %s) = %s
        ORDER BY t.closed_at DESC
        LIMIT %s
    """
    params = [epic, strategy, environment, environment, window]

    own = conn is None
    if own:
        conn = psycopg2.connect(dsn or forex_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[AutoPause] load_closed_trades failed for %s %s: %s",
                       strategy, epic, exc)
        return []
    finally:
        if own:
            conn.close()
