"""Auto-pause decision event log.

Every decision the layer takes — trips (enforced or dry-run), pauses, resume
proposals, resumes, and flip no-op errors — is recorded to
``auto_pause_events`` (strategy_config DB). system-monitor polls unnotified
rows and pushes Telegram; the trading-ui decay-monitor page reads the feed.

Recording must never break the scan loop: failures are logged and swallowed.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import psycopg2

from .config import strategy_config_dsn

logger = logging.getLogger(__name__)

EVENT_TYPES = {
    "trip",
    "pause",
    "dry_run_trip",
    "resume_proposed",
    "resumed",
    "flip_noop_error",
}


def record_event(
    event_type: str,
    strategy: str,
    epic: str,
    config_set: str,
    reason: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    *,
    dedupe_hours: Optional[float] = None,
    dsn: Optional[str] = None,
    conn: Any = None,
) -> bool:
    """Insert one event row. Returns False (and logs) on any failure.

    ``dedupe_hours``: skip the insert when the same (event_type, cell) already
    has an event within that window. The hourly orchestrator check re-evaluates
    a still-tripping cell every cycle — without dedupe a persistent dry-run trip
    would emit 24 Telegram messages/day.
    """
    if event_type not in EVENT_TYPES:
        logger.error("[AutoPause] record_event: unknown event_type %r", event_type)
        return False
    query = """
        INSERT INTO auto_pause_events
            (event_type, strategy, epic, config_set, reason, metrics)
        SELECT %s, %s, %s, %s, %s, %s
        WHERE %s::float IS NULL OR NOT EXISTS (
            SELECT 1 FROM auto_pause_events
            WHERE event_type = %s AND strategy = %s AND epic = %s
              AND config_set = %s
              AND created_at > now() - (%s::float * interval '1 hour')
        )
    """
    own = conn is None
    try:
        if own:
            conn = psycopg2.connect(dsn or strategy_config_dsn())
        with conn.cursor() as cur:
            cur.execute(
                query,
                [
                    event_type, strategy, epic, config_set, reason,
                    json.dumps(metrics) if metrics is not None else None,
                    dedupe_hours,
                    event_type, strategy, epic, config_set,
                    dedupe_hours if dedupe_hours is not None else 0.0,
                ],
            )
        if own:
            conn.commit()
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[AutoPause] record_event(%s %s %s) failed: %s",
                       event_type, strategy, epic, exc)
        return False
    finally:
        if own and conn is not None:
            conn.close()
