"""Auto-pause event watcher.

Polls ``auto_pause_events`` (strategy_config DB) for rows with
``notified_at IS NULL``, pushes each to Telegram, and stamps ``notified_at``.
The task-worker writes events but has no Telegram dependency; this poller is
the delivery leg (at-least-once: a send that fails leaves the row unstamped
and is retried next cycle).
"""
import logging
from typing import List, Optional

import psycopg2
import psycopg2.extras

from ..config import settings
from ..notifications.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

EVENT_STYLE = {
    "trip":            ("⏸️", "AUTO-PAUSE TRIP"),
    "pause":           ("⏸️", "STRATEGY PAUSED"),
    "dry_run_trip":    ("🟡", "DECAY TRIP (dry-run — not paused)"),
    "resume_proposed": ("🔔", "RESUME PROPOSED"),
    "resumed":         ("✅", "STRATEGY RESUMED"),
    "flip_noop_error": ("🚨", "AUTO-PAUSE FLIP FAILED"),
}

MAX_EVENTS_PER_CYCLE = 20


def _format_event(row: dict) -> str:
    emoji, title = EVENT_STYLE.get(row["event_type"], ("ℹ️", row["event_type"]))
    metrics = row.get("metrics") or {}
    lines = [
        f"{emoji} *{title}*",
        "",
        f"Strategy: `{row['strategy']}`",
        f"Epic: `{row['epic']}` ({row['config_set']})",
    ]
    if row.get("reason"):
        lines.append(f"Reason: {row['reason']}")
    parts = []
    if metrics.get("pf") is not None:
        parts.append(f"PF {metrics['pf']:.2f}")
    if metrics.get("win_rate") is not None:
        parts.append(f"WR {metrics['win_rate']:.0%}")
    if metrics.get("n") is not None:
        parts.append(f"n={metrics['n']}")
    if parts:
        lines.append("Metrics: " + ", ".join(parts))
    return "\n".join(lines)


class AutoPauseWatcher:
    def __init__(self, telegram_notifier: Optional[TelegramNotifier]):
        self.telegram = telegram_notifier

    def _fetch_unnotified(self, conn) -> List[dict]:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, event_type, strategy, epic, config_set, reason,
                       metrics, created_at
                FROM auto_pause_events
                WHERE notified_at IS NULL
                ORDER BY created_at ASC
                LIMIT %s
                """,
                [MAX_EVENTS_PER_CYCLE],
            )
            return [dict(r) for r in cur.fetchall()]

    def _mark_notified(self, conn, event_id: int) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auto_pause_events SET notified_at = now() WHERE id = %s",
                [event_id],
            )
        conn.commit()

    async def poll(self) -> int:
        """One poll cycle. Returns the number of events delivered."""
        if not settings.auto_pause_watch_enabled:
            return 0
        try:
            conn = psycopg2.connect(settings.strategy_config_database_url)
        except Exception as exc:
            logger.warning(f"AutoPauseWatcher: DB connect failed: {exc}")
            return 0

        delivered = 0
        try:
            rows = self._fetch_unnotified(conn)
            for row in rows:
                text = _format_event(row)
                if self.telegram and self.telegram.is_enabled():
                    sent = await self.telegram.send_message(text)
                    if not sent:
                        # Leave unstamped; retried next cycle.
                        logger.warning(
                            f"AutoPauseWatcher: Telegram send failed for event "
                            f"{row['id']} — will retry"
                        )
                        continue
                else:
                    # No Telegram configured: log loudly so events aren't silent,
                    # and stamp so the queue doesn't grow forever.
                    logger.warning(f"AutoPauseWatcher (no Telegram): {text}")
                self._mark_notified(conn, row["id"])
                delivered += 1
            if delivered:
                logger.info(f"AutoPauseWatcher: delivered {delivered} event(s)")
        except Exception as exc:
            logger.error(f"AutoPauseWatcher poll failed: {exc}")
        finally:
            conn.close()
        return delivered
