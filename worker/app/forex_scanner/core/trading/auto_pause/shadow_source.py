"""Shadow-outcome data source and Trip Rule B (decay on the ref-grid series).

Why a second source: ``trade_log`` carries only a handful of closed trades per
(strategy, epic) cell per month, so the trade-based rolling window (Rule A)
never fills for most cells. ``monitor_only_outcomes`` holds the ref-grid
outcome (``ref_pnl_pips``) for EVERY logged signal (~40-170 per cell / 30d), so
a rolling decay series is available for any cell that emits signals — traded
or monitor-only.

The ref grid (FX 10/15 pips, gold 80/160) is a decay PROXY, not live P&L: PF
here is near-binary (+tp/-sl), so PF and win-rate are largely redundant except
for TIMEOUT rows (variable 24h net). Breakeven WR at RR 1.5 is 40%. Trip Rule B
therefore requires BOTH an absolute PF floor and a win-rate drop vs the FROZEN
enrollment baseline — a single-metric trip is too noisy at window 50
(sigma(WR) ~ 7pp).

    trip when   shadow_pf < shadow_trip_pf
          AND   shadow_wr < baseline_shadow_wr - shadow_trip_wr_drop
          AND   n >= shadow_min_outcomes
       OR       consecutive shadow losses >= shadow_max_consecutive_losses

Rows are shaped ``{"profit_loss": ref_pnl_pips}`` so ``evaluate_performance``
is reused unchanged (single-unit rule: the series is pips-only, never mixed
with trade_log SEK).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

from .config import AutoPauseParams, forex_dsn
from .evaluator import PerfStats, TripDecision

logger = logging.getLogger(__name__)


def load_shadow_outcomes(
    strategy: str,
    epic: str,
    environment: str,
    window: int,
    *,
    since: Any = None,
    dsn: Optional[str] = None,
    conn: Any = None,
) -> List[Dict[str, Any]]:
    """Last ``window`` RESOLVED ref-grid outcomes for a cell, most-recent-first.

    ``since`` (optional datetime) restricts to signals after that time — used by
    the resume path to measure only post-pause outcomes. Rows are shaped like
    trade rows (``profit_loss`` = ref_pnl_pips) so ``evaluate_performance``
    consumes them directly.
    """
    query = """
        SELECT signal_timestamp AS closed_at,
               ref_pnl_pips     AS profit_loss,
               NULL             AS pips_gained
        FROM monitor_only_outcomes
        WHERE strategy = %s
          AND epic = %s
          AND coalesce(environment, %s) = %s
          AND status = 'RESOLVED'
          AND ref_pnl_pips IS NOT NULL
          AND (%s::timestamptz IS NULL OR signal_timestamp > %s)
        ORDER BY signal_timestamp DESC
        LIMIT %s
    """
    params = [strategy, epic, environment, environment, since, since, window]

    own = conn is None
    if own:
        conn = psycopg2.connect(dsn or forex_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[AutoPause] load_shadow_outcomes failed for %s %s: %s",
                       strategy, epic, exc)
        return []
    finally:
        if own:
            conn.close()


def decide_trip_shadow(
    stats: PerfStats,
    baseline_shadow_wr: Optional[float],
    params: AutoPauseParams,
) -> TripDecision:
    """Apply Trip Rule B to shadow (ref-grid) stats.

    Requires a frozen ``baseline_shadow_wr`` for the WR-drop leg; without one
    the PF+WR trip is skipped (the consecutive-loss safeguard still applies) —
    a cell enrolled without a shadow baseline cannot express "decay".
    """
    if stats.consecutive_losses >= params.shadow_max_consecutive_losses:
        return TripDecision(
            True,
            f"{stats.consecutive_losses} consecutive shadow losses "
            f">= {params.shadow_max_consecutive_losses}",
        )
    if stats.n < params.shadow_min_outcomes:
        return TripDecision(False, "")
    if baseline_shadow_wr is None:
        return TripDecision(False, "")
    pf_bad = stats.pf is not None and stats.pf < params.shadow_trip_pf
    wr_bad = stats.win_rate < (baseline_shadow_wr - params.shadow_trip_wr_drop)
    if pf_bad and wr_bad:
        return TripDecision(
            True,
            f"shadow PF {stats.pf:.2f} < {params.shadow_trip_pf} AND "
            f"WR {stats.win_rate:.0%} < baseline {baseline_shadow_wr:.0%} "
            f"- {params.shadow_trip_wr_drop:.0%} over last {stats.n} outcomes",
        )
    return TripDecision(False, "")
