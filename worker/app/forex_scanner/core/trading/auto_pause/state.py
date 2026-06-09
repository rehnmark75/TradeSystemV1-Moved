"""Auto-pause runtime state (Phase 3).

Records when a cell was auto-paused (so resume can measure fresh shadow signals
+ cooldown) and tracks resume PROPOSALS. Phase 3 is propose-only — nothing here
auto-resumes a cell.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

import psycopg2

from .config import strategy_config_dsn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PauseState:
    strategy: str
    epic: str
    config_set: str
    state: str
    paused_at: datetime
    pause_reason: Optional[str]
    resume_proposed_at: Optional[datetime]
    resume_proposal_count: int


def _run(query: str, params: List[Any], fetch: bool, dsn: Optional[str], conn: Any):
    own = conn is None
    if own:
        conn = psycopg2.connect(dsn or strategy_config_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone() if fetch else None
        if own:
            conn.commit()
        return row
    finally:
        if own:
            conn.close()


def record_pause(
    strategy: str, epic: str, config_set: str, reason: str,
    *, dsn: Optional[str] = None, conn: Any = None,
) -> None:
    """Mark a cell paused. Preserves the original paused_at if already paused."""
    query = """
        INSERT INTO auto_pause_state (strategy, epic, config_set, state, paused_at, pause_reason)
        VALUES (%s, %s, %s, 'paused', now(), %s)
        ON CONFLICT (strategy, epic, config_set) DO UPDATE SET
            state = 'paused',
            paused_at = CASE WHEN auto_pause_state.state = 'paused'
                             THEN auto_pause_state.paused_at ELSE now() END,
            pause_reason = EXCLUDED.pause_reason,
            resume_proposed_at = CASE WHEN auto_pause_state.state = 'paused'
                                      THEN auto_pause_state.resume_proposed_at ELSE NULL END,
            resume_proposal_count = CASE WHEN auto_pause_state.state = 'paused'
                                         THEN auto_pause_state.resume_proposal_count ELSE 0 END,
            updated_at = now()
    """
    _run(query, [strategy, epic, config_set, reason], False, dsn, conn)


def get_pause_state(
    strategy: str, epic: str, config_set: str,
    *, dsn: Optional[str] = None, conn: Any = None,
) -> Optional[PauseState]:
    query = """
        SELECT strategy, epic, config_set, state, paused_at, pause_reason,
               resume_proposed_at, resume_proposal_count
        FROM auto_pause_state
        WHERE strategy = %s AND epic = %s AND config_set = %s
        LIMIT 1
    """
    row = _run(query, [strategy, epic, config_set], True, dsn, conn)
    if not row:
        return None
    return PauseState(
        strategy=row[0], epic=row[1], config_set=row[2], state=row[3],
        paused_at=row[4], pause_reason=row[5],
        resume_proposed_at=row[6], resume_proposal_count=int(row[7] or 0),
    )


def record_eval(
    strategy: str, epic: str, config_set: str,
    shadow_n: Optional[int], shadow_pf: Optional[float], proposed: bool,
    *, dsn: Optional[str] = None, conn: Any = None,
) -> None:
    """Persist the latest shadow evaluation; bump proposal counters if proposed."""
    if proposed:
        query = """
            UPDATE auto_pause_state
            SET last_eval_at = now(), shadow_n = %s, shadow_pf = %s,
                resume_proposed_at = now(),
                resume_proposal_count = resume_proposal_count + 1,
                updated_at = now()
            WHERE strategy = %s AND epic = %s AND config_set = %s
        """
    else:
        query = """
            UPDATE auto_pause_state
            SET last_eval_at = now(), shadow_n = %s, shadow_pf = %s, updated_at = now()
            WHERE strategy = %s AND epic = %s AND config_set = %s
        """
    _run(query, [shadow_n, shadow_pf, strategy, epic, config_set], False, dsn, conn)


def record_resume(
    strategy: str, epic: str, config_set: str,
    shadow_n: Optional[int], shadow_pf: Optional[float],
    *, dsn: Optional[str] = None, conn: Any = None,
) -> None:
    """Mark a cell auto-resumed (state -> 'resumed', stamp resumed_at)."""
    query = """
        UPDATE auto_pause_state
        SET state = 'resumed', resumed_at = now(), last_eval_at = now(),
            shadow_n = %s, shadow_pf = %s,
            resume_proposed_at = now(),
            resume_proposal_count = resume_proposal_count + 1,
            updated_at = now()
        WHERE strategy = %s AND epic = %s AND config_set = %s
    """
    _run(query, [shadow_n, shadow_pf, strategy, epic, config_set], False, dsn, conn)
