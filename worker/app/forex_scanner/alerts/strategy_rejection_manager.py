"""
Generic strategy rejection manager.

Persists rejection events from non-SMC strategies (MEAN_REVERSION, IMPULSE_FADE,
XAU_GOLD) to the shared `strategy_rejections` table for tuning analysis.

Design:
- Buffered writes (BATCH_SIZE=100) to minimise DB round-trips per scan cycle.
- Minimal schema: common indexed columns + JSONB `details` for per-strategy context.
- Caller calls flush() after each detect_signal() call (mirroring SMC pattern).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_json(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        import numpy as np

        def _coerce(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_coerce(i) for i in obj]
            return obj

        return json.dumps(_coerce(value))
    except Exception:
        try:
            return json.dumps(str(value))
        except Exception:
            return None


def _session_name(hour: Optional[int]) -> str:
    if hour is None:
        return "unknown"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 16:
        return "overlap"
    if 16 <= hour < 21:
        return "new_york"
    return "asian"


class StrategyRejectionManager:
    """Batched DB writer for strategy rejection events.

    Usage (inside a strategy):

        self._rej_mgr = StrategyRejectionManager("IMPULSE_FADE", db_manager)

        # inside detect_signal:
        self._rej_mgr.reject("SESSION", "outside session", epic, pair,
                              hour_utc=utc_hour, details={"hour": utc_hour})
        return None

        # after detect_signal completes (called by signal_detector):
        strategy.flush_rejections()  →  self._rej_mgr.flush()
    """

    BATCH_SIZE = 100
    INSERT_SQL = """
        INSERT INTO strategy_rejections
            (strategy, epic, pair, scan_timestamp, stage, reason,
             direction, hour_utc, session, details)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """

    def __init__(self, strategy_name: str, db_manager) -> None:
        if db_manager is None:
            raise ValueError("db_manager required")
        self.strategy_name = strategy_name
        self.db_manager = db_manager
        self._buffer: List[Dict] = []

    def reject(
        self,
        stage: str,
        reason: str,
        epic: str,
        pair: str,
        direction: Optional[str] = None,
        hour_utc: Optional[int] = None,
        scan_timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Buffer one rejection. Flushes automatically when batch is full."""
        self._buffer.append(
            {
                "strategy": self.strategy_name,
                "epic": epic,
                "pair": pair,
                "scan_timestamp": scan_timestamp or datetime.now(timezone.utc),
                "stage": stage,
                "reason": reason,
                "direction": direction,
                "hour_utc": hour_utc,
                "session": _session_name(hour_utc),
                "details": details,
            }
        )
        if len(self._buffer) >= self.BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        """Write all buffered rejections to the database and clear the buffer."""
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        conn = cursor = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            rows = [
                (
                    r["strategy"],
                    r["epic"],
                    r["pair"],
                    r["scan_timestamp"],
                    r["stage"],
                    r["reason"],
                    r["direction"],
                    r["hour_utc"],
                    r["session"],
                    _safe_json(r["details"]),
                )
                for r in batch
            ]
            cursor.executemany(self.INSERT_SQL, rows)
            conn.commit()
        except Exception as exc:
            logger.warning(
                "[StrategyRejectionManager] flush failed (%s): %s", self.strategy_name, exc
            )
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
