"""Auto-backtest verdict watcher.

Polls ``auto_backtest_runs`` (strategy_config DB) for terminal rows with
``notified_at IS NULL``, pushes each verdict to Telegram, and stamps
``notified_at``. The auto-backtest container writes runs but has no Telegram
dependency; this poller is the delivery leg (at-least-once: a send that fails
leaves the row unstamped and is retried next cycle).
"""
import logging
from typing import List, Optional

import psycopg2
import psycopg2.extras

from ..config import settings
from ..notifications.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

VERDICT_STYLE = {
    "PROMOTION_CANDIDATE": ("🟢", "AUTO-BACKTEST: PROMOTION CANDIDATE"),
    "MARGINAL":            ("🟠", "AUTO-BACKTEST: MARGINAL"),
    "NO_GO":               ("🔴", "AUTO-BACKTEST: NO-GO"),
    "NO_SIGNALS":          ("⚪", "AUTO-BACKTEST: NO SIGNALS (check cache)"),
    "UNKNOWN":             ("❓", "AUTO-BACKTEST: UNPARSEABLE RESULT"),
}

MAX_EVENTS_PER_CYCLE = 20


def _format_run(row: dict) -> str:
    if row["status"] == "FAILED":
        emoji, title = "🚨", "AUTO-BACKTEST FAILED"
    else:
        emoji, title = VERDICT_STYLE.get(row.get("verdict") or "UNKNOWN",
                                         ("ℹ️", "AUTO-BACKTEST"))
    trig = row.get("trigger_metrics") or {}
    res = row.get("results") or {}
    lines = [
        f"{emoji} *{title}*",
        "",
        f"Strategy: `{row['strategy']}`",
        f"Epic: `{row['epic']}` ({row['environment']})",
        (f"Trigger ({trig.get('window_days', '?')}d monitor): "
         f"edge {trig.get('edge_ratio', '?')}, "
         f"{trig.get('pct_mfe_favorable', '?')}% fav, "
         f"n={int(trig['n']) if trig.get('n') is not None else '?'}"),
    ]
    if row["status"] == "FAILED":
        err = (row.get("error") or "")[:300]
        lines.append(f"Error: {err}")
    else:
        parts = []
        if res.get("pf") is not None:
            parts.append(f"PF {res['pf']:.2f}")
        if res.get("win_rate_pct") is not None:
            parts.append(f"WR {res['win_rate_pct']:.0f}%")
        if res.get("expectancy_pips") is not None:
            parts.append(f"exp {res['expectancy_pips']:+.1f} pips")
        if res.get("total_closed") is not None:
            parts.append(f"{res['total_closed']} trades")
        if parts:
            lines.append(
                f"Backtest ({row['backtest_days']}d live-parity): "
                + ", ".join(parts))
        if row.get("verdict") == "PROMOTION_CANDIDATE":
            lines.append("→ Passed the live-parity gate. Worth a promotion "
                         "review — nothing has been auto-enabled.")
    return "\n".join(lines)


class AutoBacktestWatcher:
    def __init__(self, telegram_notifier: Optional[TelegramNotifier]):
        self.telegram = telegram_notifier

    def _fetch_unnotified(self, conn) -> List[dict]:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, strategy, epic, environment, trigger_metrics,
                       backtest_days, status, results, verdict, error, created_at
                FROM auto_backtest_runs
                WHERE notified_at IS NULL AND status IN ('COMPLETED', 'FAILED')
                ORDER BY created_at ASC
                LIMIT %s
                """,
                [MAX_EVENTS_PER_CYCLE],
            )
            return [dict(r) for r in cur.fetchall()]

    def _mark_notified(self, conn, run_id: int) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auto_backtest_runs SET notified_at = now() WHERE id = %s",
                [run_id],
            )
        conn.commit()

    async def poll(self) -> int:
        """One poll cycle. Returns the number of verdicts delivered."""
        if not settings.auto_backtest_watch_enabled:
            return 0
        try:
            conn = psycopg2.connect(settings.strategy_config_database_url)
        except Exception as exc:
            logger.warning(f"AutoBacktestWatcher: DB connect failed: {exc}")
            return 0

        delivered = 0
        try:
            rows = self._fetch_unnotified(conn)
            for row in rows:
                text = _format_run(row)
                if self.telegram and self.telegram.is_enabled():
                    sent = await self.telegram.send_message(text)
                    if not sent:
                        # Leave unstamped; retried next cycle.
                        logger.warning(
                            f"AutoBacktestWatcher: Telegram send failed for run "
                            f"{row['id']} — will retry"
                        )
                        continue
                else:
                    # No Telegram configured: log loudly so verdicts aren't
                    # silent, and stamp so the queue doesn't grow forever.
                    logger.warning(f"AutoBacktestWatcher (no Telegram): {text}")
                self._mark_notified(conn, row["id"])
                delivered += 1
            if delivered:
                logger.info(f"AutoBacktestWatcher: delivered {delivered} verdict(s)")
        except Exception as exc:
            logger.error(f"AutoBacktestWatcher poll failed: {exc}")
        finally:
            conn.close()
        return delivered
