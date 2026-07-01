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

    def __init__(self, db_manager, dry_run: bool = True,
                 trail_enabled: bool = False, trail_arm_pct: float = 2.0,
                 trail_atr_mult: float = 3.0, trail_tp_backstop_pct: float = 30.0,
                 trail_default_atr_pct: float = 3.0):
        self.db = db_manager
        self.dry_run = dry_run
        # ATR-trailing (validated Jul-2026, ema_cross_lab.py). OFF by default.
        self.trail_enabled = trail_enabled
        self.trail_arm_pct = trail_arm_pct
        self.trail_atr_mult = trail_atr_mult
        self.trail_tp_backstop_pct = trail_tp_backstop_pct
        self.trail_default_atr_pct = trail_default_atr_pct

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
        # ATR-trailing state columns (idempotent; mirrors migration 039)
        await self.db.execute("""
            ALTER TABLE stock_breakeven_monitors
                ADD COLUMN IF NOT EXISTS peak_price DECIMAL(12,4),
                ADD COLUMN IF NOT EXISTS tp_widened BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS trail_moves INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS trail_last_at TIMESTAMP
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
        # ATR-trailing takes over the exit when enabled (long-only day-trades).
        if self.trail_enabled and (monitor.get("side") == "buy"):
            return await self._process_trailing(client, monitor, position)

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

    # ------------------------------------------------------------------
    # ATR trailing stop (long-only). Keeps status='monitoring' so it trails
    # every poll. Widens the hard TP once so winners run past the fixed 7%.
    # ------------------------------------------------------------------
    async def _process_trailing(self, client, monitor: dict, position) -> bool:
        entry = float(monitor.get("entry_price") or position.entry_price or 0)
        price = float(position.current_price or position.entry_price or 0)
        current_sl = float(position.stop_loss) if position.stop_loss is not None else None
        profit = float(position.unrealized_pnl or 0)
        if entry <= 0 or price <= 0:
            await self._update_check(monitor["id"], position.deal_id, profit, "trail: bad price")
            return False

        # high-water mark
        stored_peak = float(monitor.get("peak_price") or 0) or entry
        peak = max(stored_peak, price)

        # widen the hard TP once so the trail (not a 7% cap) governs the exit
        backstop_tp = round(entry * (1 + self.trail_tp_backstop_pct / 100), 4)
        need_widen = (not monitor.get("tp_widened")) and (
            position.take_profit is None or float(position.take_profit) < backstop_tp
        )
        moved = False
        if need_widen:
            if self.dry_run:
                logger.info("[Trail] DRY RUN would widen %s TP -> %.4f", monitor["ticker"], backstop_tp)
            else:
                await client.modify_position(deal_id=position.deal_id,
                                             stop_loss=current_sl, take_profit=backstop_tp)
                logger.info("[Trail] %s TP widened -> %.4f", monitor["ticker"], backstop_tp)

        armed = price >= entry * (1 + self.trail_arm_pct / 100)
        new_sl = None
        if armed:
            atr_pct = await self._get_atr_pct(monitor["ticker"])
            trail_stop = round(peak * (1 - self.trail_atr_mult * atr_pct / 100), 4)
            # favorable-only, and never place the stop at/above current price
            if trail_stop < price and (current_sl is None or trail_stop > current_sl):
                new_sl = trail_stop

        if new_sl is not None:
            if self.dry_run:
                logger.info("[Trail] DRY RUN would move %s SL %s -> %.4f (peak %.4f, price %.4f)",
                            monitor["ticker"], current_sl, new_sl, peak, price)
            else:
                await client.modify_position(deal_id=position.deal_id,
                                             stop_loss=new_sl,
                                             take_profit=backstop_tp if (need_widen or monitor.get("tp_widened")) else position.take_profit)
                logger.info("[Trail] Moved %s SL -> %.4f (peak %.4f)", monitor["ticker"], new_sl, peak)
                moved = True

        await self._update_trail(monitor["id"], position.deal_id, profit, peak,
                                 widened=(need_widen and not self.dry_run) or bool(monitor.get("tp_widened")),
                                 trailed=(new_sl is not None and not self.dry_run))
        return moved

    async def _get_atr_pct(self, ticker: str) -> float:
        """Latest nightly daily ATR%% for the ticker; fallback to default."""
        try:
            row = await self.db.fetchrow("""
                SELECT atr_percent FROM stock_screening_metrics
                WHERE ticker = $1 AND atr_percent IS NOT NULL
                ORDER BY calculation_date DESC LIMIT 1
            """, str(ticker).upper())
            if row and row["atr_percent"] is not None:
                v = float(row["atr_percent"])
                if 0.1 <= v <= 25.0:
                    return v
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Trail] ATR lookup failed for %s: %s", ticker, exc)
        return self.trail_default_atr_pct

    async def _update_trail(self, monitor_id: int, deal_id: str, profit: float,
                            peak: float, widened: bool, trailed: bool):
        await self.db.execute("""
            UPDATE stock_breakeven_monitors
            SET status = 'monitoring',
                robomarkets_deal_id = COALESCE(NULLIF(robomarkets_deal_id, ''), $2),
                last_profit_usd = $3,
                peak_price = $4,
                tp_widened = tp_widened OR $5,
                trail_moves = trail_moves + CASE WHEN $6 THEN 1 ELSE 0 END,
                trail_last_at = CASE WHEN $6 THEN NOW() ELSE trail_last_at END,
                last_checked_at = NOW(),
                error_message = NULL
            WHERE id = $1
        """, monitor_id, deal_id, profit, peak, widened, trailed)

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
