# services/limit_order_sync.py
"""
Limit Order Sync Service - Monitors pending_limit orders and updates their status

This service addresses a critical gap where limit orders were never monitored after creation.
It polls IG's /workingorders endpoint to detect:
1. Filled orders -> Updates to 'tracking' status with deal_id
2. Expired orders -> Updates to 'expired' status
3. Cancelled orders -> Updates to 'cancelled' status

Author: Claude Code
Date: 2025-12-19
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text

from services.db import SessionLocal
from services.models import TradeLog
from services.ig_orders import get_working_orders
from config import API_BASE_URL

import httpx

# Configure logging
logger = logging.getLogger("limit_order_sync")
logger.setLevel(logging.INFO)

# Add file handler if not already present
if not logger.handlers:
    handler = logging.FileHandler("/app/logs/limit_order_sync.log")
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class LimitOrderSyncService:
    """
    Service to synchronize pending_limit orders with IG broker state.

    Detects when limit orders:
    - Fill and become active positions
    - Expire due to time limit
    - Get cancelled
    """

    def __init__(self, trading_headers: dict = None):
        self.trading_headers = trading_headers
        self.logger = logger

    async def sync_limit_orders(self, trading_headers: dict = None) -> Dict:
        """
        Main sync function - checks all pending_limit orders against IG state.

        Returns:
            Dict with sync statistics
        """
        if trading_headers:
            self.trading_headers = trading_headers

        stats = {
            "checked": 0,
            "filled": 0,
            "not_filled": 0,  # Price never reached before expiry
            "cancelled": 0,   # User cancelled
            "rejected": 0,    # Broker rejected
            "still_pending": 0,
            "errors": 0
        }

        try:
            self.logger.info("🔄 [LIMIT SYNC] Starting limit order synchronization...")

            with SessionLocal() as db:
                # Get all pending_limit orders
                pending_orders = db.query(TradeLog).filter(
                    TradeLog.status == "pending_limit"
                ).all()

                stats["checked"] = len(pending_orders)

                if not pending_orders:
                    self.logger.info("📭 [LIMIT SYNC] No pending limit orders to check")
                    return stats

                self.logger.info(f"📋 [LIMIT SYNC] Found {len(pending_orders)} pending limit orders")

                # Fetch current working orders from IG
                ig_working_orders = await self._fetch_working_orders()

                # Fetch current positions from IG (for filled orders)
                ig_positions = await self._fetch_positions()

                # Create lookup maps
                working_order_refs = self._create_working_order_map(ig_working_orders)
                position_refs = self._create_position_map(ig_positions)

                self.logger.info(f"📊 [LIMIT SYNC] IG State: {len(working_order_refs)} working orders, {len(position_refs)} positions")

                # Process each pending order
                for order in pending_orders:
                    try:
                        result = await self._process_pending_order(
                            order, working_order_refs, position_refs, db
                        )

                        if result == "filled":
                            stats["filled"] += 1
                        elif result == "limit_not_filled":
                            stats["not_filled"] += 1
                        elif result == "limit_cancelled":
                            stats["cancelled"] += 1
                        elif result == "limit_rejected":
                            stats["rejected"] += 1
                        elif result == "pending":
                            stats["still_pending"] += 1
                        else:
                            stats["errors"] += 1

                    except Exception as e:
                        self.logger.error(f"❌ [LIMIT SYNC] Error processing order {order.id}: {e}")
                        stats["errors"] += 1

                # Commit all changes
                db.commit()

            self.logger.info(f"✅ [LIMIT SYNC] Complete: {stats['filled']} filled, {stats['not_filled']} not filled, "
                           f"{stats['cancelled']} cancelled, {stats['rejected']} rejected, {stats['still_pending']} still pending")

            return stats

        except Exception as e:
            self.logger.error(f"❌ [LIMIT SYNC] Sync failed: {e}")
            import traceback
            self.logger.error(f"❌ [LIMIT SYNC] Traceback: {traceback.format_exc()}")
            stats["errors"] += 1
            return stats

    async def _fetch_working_orders(self) -> List[Dict]:
        """Fetch all working orders from IG API"""
        try:
            result = await get_working_orders(self.trading_headers)
            return result.get("workingOrders", [])
        except Exception as e:
            self.logger.error(f"❌ [LIMIT SYNC] Failed to fetch working orders: {e}")
            return []

    async def _fetch_positions(self) -> List[Dict]:
        """Fetch all open positions from IG API"""
        try:
            headers = {
                "X-IG-API-KEY": self.trading_headers["X-IG-API-KEY"],
                "CST": self.trading_headers["CST"],
                "X-SECURITY-TOKEN": self.trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{API_BASE_URL}/positions", headers=headers)
                response.raise_for_status()
                return response.json().get("positions", [])

        except Exception as e:
            self.logger.error(f"❌ [LIMIT SYNC] Failed to fetch positions: {e}")
            return []

    def _create_working_order_map(self, working_orders: List[Dict]) -> Dict[str, Dict]:
        """
        Create a lookup map of working orders by deal reference.

        IG working orders have:
        - workingOrderData.dealId: The working order ID
        - workingOrderData.dealReference: Our reference when placing the order
        """
        order_map = {}

        for order in working_orders:
            working_data = order.get("workingOrderData", {})
            deal_ref = working_data.get("dealReference", "")
            deal_id = working_data.get("dealId", "")

            if deal_ref:
                order_map[deal_ref] = {
                    "deal_id": deal_id,
                    "status": working_data.get("orderStatus", ""),
                    "epic": order.get("marketData", {}).get("epic", ""),
                    "direction": working_data.get("direction", ""),
                    "order_level": working_data.get("orderLevel"),
                    "good_till_date": working_data.get("goodTillDate"),
                    "raw": order
                }

        return order_map

    def _create_position_map(self, positions: List[Dict]) -> Dict[str, Dict]:
        """
        Create a lookup map of positions by deal reference.

        When a limit order fills, the position will have the same dealReference
        that was used when placing the order.
        """
        position_map = {}

        for pos in positions:
            position_data = pos.get("position", {})
            deal_ref = position_data.get("dealReference", "")
            deal_id = position_data.get("dealId", "")

            if deal_ref:
                position_map[deal_ref] = {
                    "deal_id": deal_id,
                    "epic": pos.get("market", {}).get("epic", ""),
                    "direction": position_data.get("direction", ""),
                    "open_level": position_data.get("openLevel"),
                    "stop_level": position_data.get("stopLevel"),
                    "limit_level": position_data.get("limitLevel"),
                    "size": position_data.get("size"),
                    "raw": pos
                }

        return position_map

    async def _process_pending_order(
        self,
        order: TradeLog,
        working_order_refs: Dict[str, Dict],
        position_refs: Dict[str, Dict],
        db: Session
    ) -> str:
        """
        Process a single pending_limit order and determine its current state.

        Returns:
            str: 'filled', 'expired', 'cancelled', 'pending', or 'error'
        """
        deal_ref = order.deal_reference

        if not deal_ref:
            self.logger.warning(f"⚠️ [LIMIT SYNC] Order {order.id} has no deal_reference")
            return "error"

        self.logger.debug(f"🔍 [LIMIT SYNC] Processing order {order.id} ({order.symbol}) ref={deal_ref}")

        # Case 1: Order filled - now exists as a position
        if deal_ref in position_refs:
            position_data = position_refs[deal_ref]
            return await self._handle_filled_order(order, position_data, db)

        # Case 2: Order still pending on IG
        if deal_ref in working_order_refs:
            working_data = working_order_refs[deal_ref]
            return self._handle_still_pending(order, working_data, db)

        # Case 3: Order not found in working orders OR positions
        # This means it was either cancelled, expired, or rejected
        return await self._handle_missing_order(order, db)

    async def _handle_filled_order(
        self,
        order: TradeLog,
        position_data: Dict,
        db: Session
    ) -> str:
        """Handle a limit order that has been filled and is now a position"""

        old_status = order.status
        deal_id = position_data["deal_id"]

        # Update trade_log with filled order data
        # Check current size to determine if it's a full or partial fill
        current_size = float(position_data.get("size", 1.0))
        original_size = order.current_size or 1.0

        if current_size < original_size:
            order.status = "partial_closed"  # Partial fill/close
        else:
            order.status = "tracking"  # Full fill - now should be monitored like regular trades
        order.deal_id = deal_id
        order.entry_price = float(position_data.get("open_level", order.entry_price))
        order.sl_price = float(position_data.get("stop_level")) if position_data.get("stop_level") else order.sl_price
        order.tp_price = float(position_data.get("limit_level")) if position_data.get("limit_level") else order.tp_price
        order.trigger_time = datetime.utcnow()
        order.current_size = float(position_data.get("size", 1.0))

        self.logger.info(
            f"🎯 [LIMIT FILLED] Order {order.id} ({order.symbol}) FILLED! "
            f"deal_id={deal_id}, entry={order.entry_price}, "
            f"status: {old_status} → tracking"
        )

        return "filled"

    def _handle_still_pending(
        self,
        order: TradeLog,
        working_data: Dict,
        db: Session
    ) -> str:
        """Handle an order that is still pending on IG"""

        # Check if it's past the expiry time
        if order.monitor_until and datetime.utcnow() > order.monitor_until:
            # Should have expired but IG still has it - might be timezone issue
            self.logger.warning(
                f"⚠️ [LIMIT SYNC] Order {order.id} past monitor_until but still on IG. "
                f"monitor_until={order.monitor_until}, now={datetime.utcnow()}"
            )

        self.logger.debug(f"⏳ [LIMIT SYNC] Order {order.id} still pending on IG")
        return "pending"

    async def _handle_missing_order(self, order: TradeLog, db: Session) -> str:
        """
        Handle an order not found on IG (not in working orders or positions).

        This could mean:
        1. Order expired (past good-till-date)
        2. Order was cancelled
        3. Order was rejected
        4. API timing issue (rare)
        """

        # Check if past expiry time - limit order wasn't filled before expiry
        if order.monitor_until and datetime.utcnow() > order.monitor_until:
            order.status = "limit_not_filled"
            order.trigger_time = datetime.utcnow()
            self.logger.info(
                f"⏰ [LIMIT NOT FILLED] Order {order.id} ({order.symbol}) - price never reached. "
                f"monitor_until={order.monitor_until}"
            )
            return "limit_not_filled"

        # Check activity/confirms endpoint to see what happened
        outcome = await self._check_deal_outcome(order.deal_reference)

        if outcome:
            status = outcome.get("status", "").upper()
            reason = outcome.get("reason", "")

            if status == "REJECTED":
                order.status = "limit_rejected"
                order.trigger_time = datetime.utcnow()
                self.logger.warning(
                    f"❌ [LIMIT REJECTED] Order {order.id} ({order.symbol}) rejected: {reason}"
                )
                return "limit_rejected"

            elif status == "DELETED" or "CANCELLED" in status.upper():
                order.status = "limit_cancelled"
                order.trigger_time = datetime.utcnow()
                self.logger.info(
                    f"🚫 [LIMIT CANCELLED] Order {order.id} ({order.symbol}) cancelled by user"
                )
                return "limit_cancelled"

        # If we can't determine what happened, mark as not filled if old enough
        age = datetime.utcnow() - order.timestamp if order.timestamp else timedelta(days=0)

        if age > timedelta(hours=24):
            order.status = "limit_not_filled"
            order.trigger_time = datetime.utcnow()
            self.logger.warning(
                f"⏰ [LIMIT NOT FILLED] Order {order.id} ({order.symbol}) not found and >24h old. "
                f"Price was never reached."
            )
            return "limit_not_filled"

        # Order is missing but relatively new - could be API timing issue
        self.logger.warning(
            f"⚠️ [LIMIT SYNC] Order {order.id} ({order.symbol}) not found on IG but only {age} old. "
            f"Will check again next cycle."
        )
        return "pending"

    async def _check_deal_outcome(self, deal_reference: str) -> Optional[Dict]:
        """Check IG confirms endpoint for deal outcome"""
        try:
            if not deal_reference:
                return None

            headers = {
                "X-IG-API-KEY": self.trading_headers["X-IG-API-KEY"],
                "CST": self.trading_headers["CST"],
                "X-SECURITY-TOKEN": self.trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "1"
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{API_BASE_URL}/confirms/{deal_reference}",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "status": data.get("dealStatus", ""),
                        "reason": data.get("reason", ""),
                        "deal_id": data.get("dealId", "")
                    }
                elif response.status_code == 404:
                    return {"status": "NOT_FOUND"}
                else:
                    return None

        except Exception as e:
            self.logger.debug(f"Could not check deal outcome for {deal_reference}: {e}")
            return None


async def sync_limit_orders(trading_headers: dict) -> Dict:
    """
    Standalone function to sync limit orders.
    Can be called from main.py scheduler.
    """
    service = LimitOrderSyncService(trading_headers)
    return await service.sync_limit_orders()


async def periodic_limit_order_sync(trading_headers_func, interval_seconds: int = 60):
    """
    Periodic sync function for limit orders.

    Args:
        trading_headers_func: Async function to get fresh trading headers
        interval_seconds: How often to check (default 60s)
    """
    logger.info(f"🚀 [LIMIT SYNC] Starting periodic sync every {interval_seconds}s")

    while True:
        try:
            trading_headers = await trading_headers_func()
            if trading_headers:
                await sync_limit_orders(trading_headers)
            else:
                logger.warning("⚠️ [LIMIT SYNC] No trading headers available")
        except Exception as e:
            logger.error(f"❌ [LIMIT SYNC] Periodic sync error: {e}")

        await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    """Test the limit order sync service"""
    import asyncio

    async def test():
        print("🧪 Testing Limit Order Sync Service...")

        # This would need real trading headers to work
        # For now, just test the database query part
        with SessionLocal() as db:
            pending = db.query(TradeLog).filter(
                TradeLog.status == "pending_limit"
            ).all()

            print(f"📋 Found {len(pending)} pending_limit orders:")
            for order in pending:
                print(f"   - ID {order.id}: {order.symbol} {order.direction} "
                      f"ref={order.deal_reference} limit={order.limit_price}")

        print("✅ Test complete")

    asyncio.run(test())
