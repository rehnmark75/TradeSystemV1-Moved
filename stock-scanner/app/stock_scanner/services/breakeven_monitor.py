"""
Breakeven Monitor - moves SL to entry after a configured unrealized profit.

The service is intentionally idempotent:
- pending limit orders stay pending until an open broker deal appears
- stops are only moved in the favorable direction
- moved monitors are not processed again
- dry-run mode is enabled by default through the caller
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("breakeven_monitor")


class BreakevenMonitor:
    DEFAULT_TRIGGER_USD = 10.0

    def __init__(self, db_manager, dry_run: bool = True):
        self.db = db_manager
        self.dry_run = dry_run

    async def ensure_schema(self):
        """Create monitor table for deployments where the stock migration has not rerun."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS stock_breakeven_monitors (
                id BIGSERIAL PRIMARY KEY,
                stock_order_id BIGINT REFERENCES stock_orders(id),
                robomarkets_order_id VARCHAR(100),
                robomarkets_deal_id VARCHAR(100),
                ticker VARCHAR(20) NOT NULL,
                broker_ticker VARCHAR(50),
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(12,4),
                entry_price DECIMAL(12,4),
                initial_stop_loss DECIMAL(12,4),
                breakeven_stop_price DECIMAL(12,4),
                take_profit DECIMAL(12,4),
                trigger_profit_usd DECIMAL(12,4) DEFAULT 10.00,
                poll_interval_seconds INTEGER DEFAULT 300,
                status VARCHAR(30) DEFAULT 'pending_fill',
                last_profit_usd DECIMAL(12,4),
                last_checked_at TIMESTAMP,
                moved_at TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_breakeven_monitors_active
            ON stock_breakeven_monitors(status, last_checked_at)
            WHERE status IN ('pending_fill', 'monitoring')
        """)

    async def check_once(self, client) -> Dict[str, Any]:
        """Run one monitor pass against active broker positions."""
        await self.ensure_schema()
        result = {
            "checked": 0,
            "matched": 0,
            "moved": 0,
            "dry_run": self.dry_run,
            "closed": 0,
            "errors": 0,
        }

        monitors = await self._get_due_monitors()
        if not monitors:
            logger.info("[Breakeven] No due monitors")
            return result

        result["checked"] = len(monitors)
        logger.info("[Breakeven] Checking %s due monitors (dry_run=%s)", len(monitors), self.dry_run)

        try:
            positions = await client.get_positions()
        except Exception as exc:
            logger.error("[Breakeven] Failed to fetch broker positions: %s", exc)
            result["errors"] += len(monitors)
            return result

        by_deal = {p.deal_id: p for p in positions if p.deal_id}
        by_ticker: Dict[str, List[Any]] = {}
        for p in positions:
            plain = self._plain_ticker(p.ticker)
            by_ticker.setdefault(plain, []).append(p)

        for monitor in monitors:
            try:
                position = self._match_position(monitor, by_deal, by_ticker)
                if not position:
                    await self._touch_pending(monitor["id"])
                    continue

                result["matched"] += 1
                moved = await self._process_position(client, monitor, position)
                if moved:
                    result["moved"] += 1
            except Exception as exc:
                result["errors"] += 1
                logger.error("[Breakeven] Monitor %s failed: %s", monitor["id"], exc)
                await self._mark_failed(monitor["id"], str(exc))

        return result

    async def _get_due_monitors(self) -> List[dict]:
        rows = await self.db.fetch("""
            SELECT *
            FROM stock_breakeven_monitors
            WHERE status IN ('pending_fill', 'monitoring')
              AND (
                last_checked_at IS NULL
                OR last_checked_at <= NOW() - (COALESCE(poll_interval_seconds, 300) || ' seconds')::interval
              )
            ORDER BY created_at
        """)
        return [dict(r) for r in rows]

    def _match_position(self, monitor: dict, by_deal: dict, by_ticker: dict):
        deal_id = monitor.get("robomarkets_deal_id")
        if deal_id and str(deal_id) in by_deal:
            return by_deal[str(deal_id)]

        ticker = str(monitor.get("ticker") or "").upper()
        side = "long" if monitor.get("side") == "buy" else "short"
        candidates = [p for p in by_ticker.get(ticker, []) if p.side == side]
        if not candidates:
            return None

        quantity = float(monitor["quantity"]) if monitor.get("quantity") else None
        if quantity:
            close_matches = [p for p in candidates if abs(float(p.quantity) - quantity) < 0.0001]
            if close_matches:
                candidates = close_matches

        candidates.sort(key=lambda p: p.opened_at or datetime.min, reverse=True)
        return candidates[0]

    async def _process_position(self, client, monitor: dict, position) -> bool:
        profit = float(position.unrealized_pnl or 0)
        trigger = float(monitor.get("trigger_profit_usd") or self.DEFAULT_TRIGGER_USD)
        entry = float(monitor.get("breakeven_stop_price") or monitor.get("entry_price") or position.entry_price)
        current_sl = float(position.stop_loss) if position.stop_loss is not None else None
        take_profit = float(position.take_profit) if position.take_profit is not None else (
            float(monitor["take_profit"]) if monitor.get("take_profit") else None
        )

        await self._mark_monitoring(monitor, position, profit, entry, take_profit)

        if profit < trigger:
            logger.info(
                "[Breakeven] %s deal %s profit %.2f below %.2f",
                monitor["ticker"], position.deal_id, profit, trigger
            )
            return False

        if not self._should_move_stop(position.side, current_sl, entry):
            await self._mark_moved(monitor["id"], position.deal_id, entry, profit, "already protected")
            return False

        if self.dry_run:
            logger.info(
                "[Breakeven] DRY RUN would move %s deal %s SL from %s to %.4f at profit %.2f",
                monitor["ticker"], position.deal_id, current_sl, entry, profit
            )
            await self._update_check(monitor["id"], position.deal_id, profit, None)
            return False

        await client.modify_position(
            deal_id=position.deal_id,
            stop_loss=entry,
            take_profit=take_profit,
        )
        await self._mark_moved(monitor["id"], position.deal_id, entry, profit, None)
        logger.info(
            "[Breakeven] Moved %s deal %s SL to %.4f at profit %.2f",
            monitor["ticker"], position.deal_id, entry, profit
        )
        return True

    def _should_move_stop(self, side: str, current_sl: Optional[float], entry: float) -> bool:
        if current_sl is None or current_sl <= 0:
            return True
        if side == "long":
            return current_sl < entry
        return current_sl > entry

    async def _mark_monitoring(self, monitor: dict, position, profit: float, entry: float, take_profit: Optional[float]):
        await self.db.execute("""
            UPDATE stock_breakeven_monitors
            SET
                status = 'monitoring',
                robomarkets_deal_id = COALESCE(NULLIF(robomarkets_deal_id, ''), $2),
                broker_ticker = COALESCE(broker_ticker, $3),
                entry_price = COALESCE(entry_price, $4),
                breakeven_stop_price = COALESCE(breakeven_stop_price, $4),
                take_profit = COALESCE(take_profit, $5),
                last_profit_usd = $6,
                last_checked_at = NOW(),
                error_message = NULL
            WHERE id = $1
        """, monitor["id"], position.deal_id, position.ticker, entry, take_profit, profit)

    async def _touch_pending(self, monitor_id: int):
        await self.db.execute("""
            UPDATE stock_breakeven_monitors
            SET last_checked_at = NOW()
            WHERE id = $1
        """, monitor_id)

    async def _update_check(self, monitor_id: int, deal_id: str, profit: float, error: Optional[str]):
        await self.db.execute("""
            UPDATE stock_breakeven_monitors
            SET robomarkets_deal_id = COALESCE(NULLIF(robomarkets_deal_id, ''), $2),
                last_profit_usd = $3,
                last_checked_at = NOW(),
                error_message = $4
            WHERE id = $1
        """, monitor_id, deal_id, profit, error)

    async def _mark_moved(self, monitor_id: int, deal_id: str, stop_price: float, profit: float, note: Optional[str]):
        await self.db.execute("""
            UPDATE stock_breakeven_monitors
            SET status = 'moved',
                robomarkets_deal_id = COALESCE(NULLIF(robomarkets_deal_id, ''), $2),
                breakeven_stop_price = $3,
                last_profit_usd = $4,
                last_checked_at = NOW(),
                moved_at = NOW(),
                error_message = $5
            WHERE id = $1
        """, monitor_id, deal_id, stop_price, profit, note)

    async def _mark_failed(self, monitor_id: int, error: str):
        await self.db.execute("""
            UPDATE stock_breakeven_monitors
            SET status = 'failed',
                last_checked_at = NOW(),
                error_message = $2
            WHERE id = $1
        """, monitor_id, error[:1000])

    def _plain_ticker(self, broker_ticker: str) -> str:
        return str(broker_ticker or "").split(".")[0].upper()
