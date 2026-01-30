"""
Standalone Adjust Stop Service
Provides direct access to adjust-stop logic without HTTP authentication requirements.
This service is used by internal trade monitoring systems that need to adjust stops/limits.
"""

import json
import logging
import httpx
from typing import Dict, Optional, Any
from sqlalchemy.orm import Session
from services.db import SessionLocal
from services.models import TradeLog
from dependencies import get_ig_auth_headers
from config import API_BASE_URL

logger = logging.getLogger("adjust_stop_service")


def ig_points_to_price(points: float, epic: str) -> float:
    """Convert points to price based on epic type"""
    if "JPY" in epic:
        return round(points * 0.01, 5)
    elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return round(points * 0.0001, 5)
    else:
        return round(points * 1.0, 5)


async def adjust_stop_logic(
    epic: str,
    stop_offset_points: Optional[int] = None,
    limit_offset_points: Optional[int] = None,
    adjust_direction_stop: str = "increase",
    adjust_direction_limit: str = "increase",
    new_stop: Optional[float] = None,
    new_limit: Optional[float] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Standalone adjust stop logic that can be called directly by internal services.

    Args:
        epic: Trading symbol (e.g., 'CS.D.GBPUSD.MINI.IP')
        stop_offset_points: Points to adjust stop by
        limit_offset_points: Points to adjust limit by
        adjust_direction_stop: 'increase' or 'decrease' stop
        adjust_direction_limit: 'increase' or 'decrease' limit
        new_stop: Absolute new stop level
        new_limit: Absolute new limit level
        dry_run: If True, don't actually send to IG

    Returns:
        Dict with status, dealId, and response details
    """

    try:
        # Get IG authentication headers
        trading_headers = await get_ig_auth_headers()

        headers = {
            "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
            "CST": trading_headers["CST"],
            "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
            "Accept": "application/json",
            "Version": "2"
        }

        # Get current positions
        async with httpx.AsyncClient() as client:
            pos_resp = await client.get(f"{API_BASE_URL}/positions", headers=headers)
            pos_resp.raise_for_status()
            positions = pos_resp.json()["positions"]

        # Find matching position
        match = next((p for p in positions if p["market"]["epic"] == epic), None)
        if not match:
            # Update database to mark trades as closed
            with SessionLocal() as db:
                updated = db.query(TradeLog).filter(
                    TradeLog.symbol == epic,
                    TradeLog.status.in_(["pending", "tracking", "break_even", "trailing", "stage2_profit_lock", "stage3_trailing"])
                ).update({TradeLog.status: "closed"})
                db.commit()

            return {
                "status": "closed",
                "message": f"No open position found for {epic}. Marked {updated} trades as closed."
            }

        position = match["position"]
        deal_id = position["dealId"]
        direction = position["direction"]

        # Build adjustment payload
        payload = {
            "guaranteedStop": False,
            "trailingStop": False
        }

        # Helper: normalize CEEM prices (IG returns scaled like 11988.2 instead of 1.19882)
        def normalize_ceem_price(price: float, epic: str) -> float:
            if "CEEM" in epic and price > 1000:
                return price / 10000.0
            return price

        # Helper: denormalize price back to CEEM points format for sending to IG
        # CRITICAL FIX (Jan 2026): IG expects CEEM stops in points format (11928.4 not 1.19284)
        def denormalize_ceem_price(price: float, epic: str) -> float:
            if "CEEM" in epic and price < 100:  # Price format, needs conversion to points
                return price * 10000.0
            return price

        # --- STOP logic ---
        if new_stop is not None:
            # Convert to CEEM points format if needed
            payload["stopLevel"] = denormalize_ceem_price(float(new_stop), epic)
        elif position.get("stopLevel"):
            old_stop = normalize_ceem_price(float(position["stopLevel"]), epic)
            offset = ig_points_to_price(float(stop_offset_points), epic) if stop_offset_points else 0.0002

            # Apply direction correctly based on adjustDirectionStop
            if adjust_direction_stop == "increase":
                # "increase" means move stop price UP (away from current price for both BUY/SELL)
                new_stop_level = old_stop + offset
            else:  # "decrease"
                # "decrease" means move stop price DOWN (closer to current price for both BUY/SELL)
                new_stop_level = old_stop - offset

            # Convert to CEEM points format if needed
            payload["stopLevel"] = denormalize_ceem_price(round(new_stop_level, 5), epic)

        elif position.get("stopDistance"):
            old_distance = float(position["stopDistance"])
            offset_points = float(stop_offset_points) if stop_offset_points else 2

            # Distance logic - decrease distance = tighter stop, increase distance = looser stop
            if adjust_direction_stop == "increase":
                # "increase" means make stop looser (increase distance from current price)
                new_distance = old_distance + offset_points
            else:  # "decrease"
                # "decrease" means make stop tighter (decrease distance from current price)
                new_distance = max(1, old_distance - offset_points)  # Don't go below 1

            payload["stopDistance"] = max(1, round(new_distance))

        # --- LIMIT logic ---
        if new_limit is not None:
            # Convert to CEEM points format if needed
            payload["limitLevel"] = denormalize_ceem_price(float(new_limit), epic)
        elif position.get("limitLevel"):
            old_limit = normalize_ceem_price(float(position["limitLevel"]), epic)
            offset = ig_points_to_price(float(limit_offset_points), epic) if limit_offset_points else 0.0002

            # Apply direction correctly based on adjustDirectionLimit
            if adjust_direction_limit == "increase":
                # "increase" means move limit price UP (better for BUY, worse for SELL)
                new_limit_level = old_limit + offset
            else:  # "decrease"
                # "decrease" means move limit price DOWN (worse for BUY, better for SELL)
                new_limit_level = old_limit - offset

            # Convert to CEEM points format if needed
            payload["limitLevel"] = denormalize_ceem_price(round(new_limit_level, 5), epic)

        elif position.get("limitDistance"):
            old_distance = float(position["limitDistance"])
            offset_points = float(limit_offset_points) if limit_offset_points else 2

            # Distance logic for limits
            if adjust_direction_limit == "increase":
                # "increase" means increase distance to limit (better profit target)
                new_distance = old_distance + offset_points
            else:  # "decrease"
                # "decrease" means decrease distance to limit (closer profit target)
                new_distance = max(1, old_distance - offset_points)

            payload["limitDistance"] = round(new_distance)

        # Validate payload
        if "stopLevel" not in payload and "stopDistance" not in payload:
            raise Exception("No stop value provided or available")
        if "limitLevel" not in payload and "limitDistance" not in payload:
            raise Exception("No limit value provided or available")

        # Enhanced logging for debugging
        logger.info(f"[ADJUST-STOP-SERVICE] {epic} Direction: {direction} (BUY/SELL position)")
        logger.info(f"[ADJUST-STOP-SERVICE] {epic} Stop direction: {adjust_direction_stop}, Limit direction: {adjust_direction_limit}")
        logger.info(f"[ADJUST-STOP-SERVICE] {epic} Stop offset: {stop_offset_points}, Limit offset: {limit_offset_points}")

        if position.get("stopLevel"):
            logger.info(f"[ADJUST-STOP-SERVICE] {epic} Old stop level: {position['stopLevel']} → New: {payload.get('stopLevel', 'N/A')}")
        if position.get("limitLevel"):
            logger.info(f"[ADJUST-STOP-SERVICE] {epic} Old limit level: {position['limitLevel']} → New: {payload.get('limitLevel', 'N/A')}")

        # Dry run response
        if dry_run:
            return {
                "status": "dry_run",
                "dealId": deal_id,
                "adjustDirectionStop": adjust_direction_stop,
                "adjustDirectionLimit": adjust_direction_limit,
                "sentPayload": payload,
                "note": "This is a dry run. No request was sent to IG."
            }

        # Send request to IG
        update_url = f"{API_BASE_URL}/positions/otc/{deal_id}"
        logger.info(f"➡ [ADJUST-STOP-SERVICE] PUT {update_url}")
        logger.info(f"➡ [ADJUST-STOP-SERVICE] Payload: {json.dumps(payload, indent=2)}")

        async with httpx.AsyncClient() as client:
            response = await client.put(update_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

        logger.info(f"✅ [ADJUST-STOP-SERVICE] {epic} adjustment successful")

        return {
            "status": "updated",
            "dealId": deal_id,
            "adjustDirectionStop": adjust_direction_stop,
            "adjustDirectionLimit": adjust_direction_limit,
            "sentPayload": payload,
            "apiResponse": result
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"❌ [ADJUST-STOP-SERVICE] HTTP ERROR {e.response.status_code} — {e.response.text}")
        return {
            "status": "error",
            "message": f"HTTP {e.response.status_code}: {e.response.text}"
        }
    except Exception as e:
        logger.exception(f"❌ [ADJUST-STOP-SERVICE] Unhandled exception")
        return {
            "status": "error",
            "message": f"Update failed: {str(e)}"
        }


def adjust_stop_sync(
    epic: str,
    stop_offset_points: Optional[int] = None,
    limit_offset_points: Optional[int] = None,
    adjust_direction_stop: str = "increase",
    adjust_direction_limit: str = "increase",
    new_stop: Optional[float] = None,
    new_limit: Optional[float] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Synchronous wrapper for adjust_stop_logic.
    Used by synchronous code that can't call async functions directly.
    """
    import asyncio

    # Check if there's already an event loop running
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, we need to use a different approach
        # Create a new event loop in a separate thread
        import concurrent.futures
        import threading

        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    adjust_stop_logic(
                        epic=epic,
                        stop_offset_points=stop_offset_points,
                        limit_offset_points=limit_offset_points,
                        adjust_direction_stop=adjust_direction_stop,
                        adjust_direction_limit=adjust_direction_limit,
                        new_stop=new_stop,
                        new_limit=new_limit,
                        dry_run=dry_run
                    )
                )
            finally:
                new_loop.close()

        # Run in a separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async)
            return future.result(timeout=30)  # 30 second timeout

    except RuntimeError:
        # No event loop running, we can use asyncio.run directly
        return asyncio.run(
            adjust_stop_logic(
                epic=epic,
                stop_offset_points=stop_offset_points,
                limit_offset_points=limit_offset_points,
                adjust_direction_stop=adjust_direction_stop,
                adjust_direction_limit=adjust_direction_limit,
                new_stop=new_stop,
                new_limit=new_limit,
                dry_run=dry_run
            )
        )