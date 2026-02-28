"""
Stale Order Guardian - Cancels unfilled limit orders whose thesis is no longer valid.

Checks pending broker orders against three cancellation rules:
1. trade_ready flipped FALSE in stock_watchlist_results
2. Current price diverged > 1 ATR(14) from the limit price
3. Order unfilled for > 48 hours

Runs twice daily after broker syncs (09:35 ET and 16:05 ET).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any

logger = logging.getLogger("stale_order_guardian")


class StaleOrderGuardian:
    MAX_AGE_HOURS = 48

    def __init__(self, db_manager):
        self.db = db_manager

    async def check_and_cancel(self, client, db_pool=None) -> Dict[str, Any]:
        """
        Check all submitted orders and cancel stale ones.

        Args:
            client: RoboMarketsClient (already inside async with context)
            db_pool: unused, kept for interface compat

        Returns:
            {checked, cancelled, synced, reasons: [...]}
        """
        result = {
            'checked': 0,
            'cancelled': 0,
            'synced': 0,
            'reasons': [],
        }

        # 1. Get DB orders with status='submitted' that have a broker order ID
        db_orders = await self._get_submitted_orders()
        if not db_orders:
            logger.info("[Guardian] No submitted orders to check")
            return result

        result['checked'] = len(db_orders)
        logger.info(f"[Guardian] Checking {len(db_orders)} submitted orders")

        # 2. Get active orders from broker
        try:
            broker_orders = await client.get_orders(status='active')
        except Exception as e:
            logger.error(f"[Guardian] Failed to fetch broker orders: {e}")
            return result

        broker_order_ids = {
            str(o.get('order_id') or o.get('orderId') or o.get('id'))
            for o in broker_orders
        }
        logger.info(f"[Guardian] Broker has {len(broker_order_ids)} active orders")

        # 3. Build lookup data for checks
        tickers = list({o['ticker'] for o in db_orders})
        trade_ready_map = await self._get_trade_ready_map(tickers)
        atr_map = await self._get_atr_map(tickers)
        price_map = await self._get_latest_prices(tickers)

        now = datetime.now(timezone.utc)

        for order in db_orders:
            broker_id = order['robomarkets_order_id']
            ticker = order['ticker']
            db_id = order['id']

            # Order no longer active at broker → sync status
            if broker_id not in broker_order_ids:
                await self._sync_missing_order(order)
                result['synced'] += 1
                continue

            # Check cancellation rules
            cancel_reason = self._evaluate_rules(
                order, trade_ready_map, atr_map, price_map, now
            )

            if cancel_reason:
                try:
                    await client.cancel_order(broker_id)
                    await self._mark_cancelled(db_id, cancel_reason)
                    result['cancelled'] += 1
                    result['reasons'].append({
                        'ticker': ticker,
                        'db_id': db_id,
                        'broker_id': broker_id,
                        'reason': cancel_reason,
                    })
                    logger.info(
                        f"[Guardian] Cancelled order {db_id} ({ticker}): {cancel_reason}"
                    )
                except Exception as e:
                    logger.error(
                        f"[Guardian] Failed to cancel order {db_id} ({ticker}): {e}"
                    )

        logger.info(
            f"[Guardian] Done: checked={result['checked']}, "
            f"cancelled={result['cancelled']}, synced={result['synced']}"
        )
        return result

    def _evaluate_rules(
        self,
        order: dict,
        trade_ready_map: dict,
        atr_map: dict,
        price_map: dict,
        now: datetime,
    ) -> str | None:
        """Return cancel reason string, or None if order is still valid."""
        ticker = order['ticker']

        # Rule 1: trade_ready flipped FALSE
        trade_ready = trade_ready_map.get(ticker)
        if trade_ready is not None and not trade_ready:
            return "trade_ready is FALSE"

        # Rule 2: Price diverged > 1 ATR from limit price
        limit_price = order.get('price')
        current_price = price_map.get(ticker)
        atr = atr_map.get(ticker)
        if limit_price and current_price and atr and atr > 0:
            divergence = abs(float(current_price) - float(limit_price))
            if divergence > float(atr):
                return (
                    f"Price diverged {divergence:.2f} > 1 ATR ({float(atr):.2f})"
                )

        # Rule 3: Order too old (> 48 hours)
        created_at = order.get('created_at')
        if created_at:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_hours = (now - created_at).total_seconds() / 3600
            if age_hours > self.MAX_AGE_HOURS:
                return f"Order age {age_hours:.1f}h > {self.MAX_AGE_HOURS}h limit"

        return None

    # =========================================================================
    # Database helpers
    # =========================================================================

    async def _get_submitted_orders(self) -> List[dict]:
        """Get all orders with status='submitted' and a broker order ID."""
        rows = await self.db.fetch("""
            SELECT id, robomarkets_order_id, ticker, order_type, side,
                   quantity, price, stop_loss, take_profit, created_at
            FROM stock_orders
            WHERE status = 'submitted'
              AND robomarkets_order_id IS NOT NULL
              AND robomarkets_order_id != ''
            ORDER BY created_at
        """)
        return [dict(r) for r in rows]

    async def _get_trade_ready_map(self, tickers: List[str]) -> dict:
        """Get trade_ready status for each ticker from latest active watchlist."""
        if not tickers:
            return {}
        rows = await self.db.fetch("""
            SELECT DISTINCT ON (ticker)
                ticker, trade_ready
            FROM stock_watchlist_results
            WHERE ticker = ANY($1)
              AND status = 'active'
            ORDER BY ticker, scan_date DESC
        """, tickers)
        return {r['ticker']: r['trade_ready'] for r in rows}

    async def _get_atr_map(self, tickers: List[str]) -> dict:
        """Get ATR(14) for each ticker from screening metrics."""
        if not tickers:
            return {}
        rows = await self.db.fetch("""
            SELECT DISTINCT ON (ticker)
                ticker, atr_14
            FROM stock_screening_metrics
            WHERE ticker = ANY($1)
            ORDER BY ticker, calculation_date DESC
        """, tickers)
        return {r['ticker']: r['atr_14'] for r in rows}

    async def _get_latest_prices(self, tickers: List[str]) -> dict:
        """Get latest close price for each ticker from daily candles."""
        if not tickers:
            return {}
        rows = await self.db.fetch("""
            SELECT DISTINCT ON (ticker)
                ticker, close
            FROM stock_daily_candles
            WHERE ticker = ANY($1)
            ORDER BY ticker, timestamp DESC
        """, tickers)
        return {r['ticker']: r['close'] for r in rows}

    async def _mark_cancelled(self, order_id: int, reason: str):
        """Update order status to cancelled with reason."""
        await self.db.execute("""
            UPDATE stock_orders
            SET status = 'cancelled',
                error_message = $1,
                updated_at = NOW()
            WHERE id = $2
        """, reason, order_id)

    async def _sync_missing_order(self, order: dict):
        """
        DB order is 'submitted' but not in broker active set.
        Mark as filled (optimistic - broker filled or cancelled it).
        """
        db_id = order['id']
        await self.db.execute("""
            UPDATE stock_orders
            SET status = 'filled',
                filled_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
              AND status = 'submitted'
        """, db_id)
        logger.info(
            f"[Guardian] Order {db_id} ({order['ticker']}) no longer active at broker → marked filled"
        )
