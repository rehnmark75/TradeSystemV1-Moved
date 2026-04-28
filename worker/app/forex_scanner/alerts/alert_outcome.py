"""Records the post-Claude execution verdict for an alert.

OrderManager.execute_signal() returns down 7 different paths (monitor_only,
validation_failed, paper_mode, execution_failed, method_not_available, error,
executed). Each path now writes back to alert_history so future investigations
into "why didn't this approved alert trade?" become a single column lookup.

Stateless on purpose: opens its own short-lived connection per call. Volume is
low (matches alert rate), so connection-pool overhead is irrelevant; the
trade-off is decoupling from OrderManager's existing dependency graph.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import psycopg2

logger = logging.getLogger(__name__)

_MAX_REASON_LEN = 500


def _truncate(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= _MAX_REASON_LEN else s[:_MAX_REASON_LEN - 3] + "..."


def _conn_str() -> str:
    return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")


def record_execution(alert_id: int) -> None:
    """Mark an alert as successfully executed (block_reason=NULL, executed_at=NOW())."""
    if not alert_id:
        return
    try:
        with psycopg2.connect(_conn_str()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE alert_history SET executed_at = NOW(), block_reason = NULL WHERE id = %s",
                    (alert_id,),
                )
    except Exception as e:
        logger.warning(f"⚠️ Could not record execution for alert #{alert_id}: {e}")


def record_block(alert_id: Optional[int], reason: str) -> None:
    """Mark an alert as blocked post-Claude with the given reason."""
    if not alert_id:
        return
    try:
        with psycopg2.connect(_conn_str()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE alert_history SET block_reason = %s WHERE id = %s",
                    (_truncate(reason), alert_id),
                )
    except Exception as e:
        logger.warning(f"⚠️ Could not record block_reason for alert #{alert_id}: {e}")


def reason_from_executor_error(error_msg: str) -> str:
    """Map a fastapi-dev / OrderExecutor error message to a stable block_reason category."""
    msg = (error_msg or "").lower()
    if "cooldown" in msg:
        return f"cooldown:{_truncate(error_msg)}"
    if "market" in msg and "closed" in msg:
        return f"market_closed:{_truncate(error_msg)}"
    if "spread" in msg:
        return f"spread:{_truncate(error_msg)}"
    if "margin" in msg or "insufficient" in msg:
        return f"margin:{_truncate(error_msg)}"
    return f"executor_error:{_truncate(error_msg)}"
