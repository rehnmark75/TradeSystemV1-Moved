import httpx
import json
import asyncio
import logging
from config import API_BASE_URL

# Add logger for better debugging
logger = logging.getLogger(__name__)

async def has_open_position(epic: str, auth_headers: dict) -> bool:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{API_BASE_URL}/positions", headers=auth_headers)
        response.raise_for_status()
        positions = response.json().get("positions", [])
        return any(pos["market"]["epic"] == epic for pos in positions)


async def place_market_order(auth_headers, market_epic, direction, currency_code, sl_limit, limit_distance=None):
    payload = {
        "epic": market_epic,
        "expiry": "-",
        "direction": direction,
        "size": 1.0,
        "orderType": "MARKET",
        "guaranteedStop": False,
        "forceOpen": True,
        "currencyCode": currency_code,
        "stopDistance": sl_limit,
        "limitDistance": limit_distance or 25  # Use provided limit_distance or fallback to 25
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/positions/otc",
            headers=auth_headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        return response.json()


# ENHANCED: Original function now uses retry logic
async def get_deal_confirmation(trading_headers: dict, deal_reference: str):
    """Original function now with retry logic for better reliability"""
    return await get_deal_confirmation_with_retry(trading_headers, deal_reference)


# NEW: Enhanced version with retry logic and exponential backoff
async def get_deal_confirmation_with_retry(trading_headers: dict, deal_reference: str, max_retries: int = 5):
    """
    IMPROVED: Get deal confirmation with retry logic and exponential backoff
    
    Args:
        trading_headers: IG API headers
        deal_reference: Deal reference from order placement
        max_retries: Maximum number of retry attempts
        
    Returns:
        Deal confirmation data
        
    Raises:
        HTTPException: If confirmation fails after all retries
    """
    url = f"{API_BASE_URL}/confirms/{deal_reference}"
    headers = {
        "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
        "CST": trading_headers["CST"],
        "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
        "Accept": "application/json",
        "Version": "1"
    }

    last_error = None
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    logger.info(f"✅ Deal confirmation successful on attempt {attempt + 1}")
                    return response.json()
                elif response.status_code == 404:
                    # Deal confirmation not ready yet, wait and retry
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                    logger.warning(f"⏳ Deal confirmation not ready (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    last_error = f"Deal confirmation not found after {wait_time}s wait"
                    continue
                else:
                    # Other HTTP errors
                    response.raise_for_status()
                    
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}: {e.response.text}"
            if e.response.status_code != 404:
                # Non-404 errors shouldn't be retried
                logger.error(f"❌ Deal confirmation failed with non-retryable error: {last_error}")
                raise
            # 404 errors will be retried
            
        except Exception as e:
            last_error = f"Network error: {str(e)}"
            logger.warning(f"⚠️ Deal confirmation attempt {attempt + 1} failed: {last_error}")
            
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)
                await asyncio.sleep(wait_time)
    
    # All retries exhausted
    error_msg = f"Deal confirmation failed after {max_retries} attempts. Last error: {last_error}"
    logger.error(f"❌ {error_msg}")
    raise httpx.HTTPStatusError(error_msg, request=None, response=None)


# NEW: Simple confirmation without retries (for cases where you want immediate failure)
async def get_deal_confirmation_simple(trading_headers: dict, deal_reference: str):
    """
    FALLBACK: Simple confirmation without retries (for cases where you want immediate failure)
    """
    url = f"{API_BASE_URL}/confirms/{deal_reference}"
    headers = {
        "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
        "CST": trading_headers["CST"],
        "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
        "Accept": "application/json",
        "Version": "1"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def is_deal_closed(deal_id: str, headers: dict) -> bool:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{API_BASE_URL}/positions",  # FIXED: Removed double slash
            headers=headers
        )
        response.raise_for_status()
        positions = response.json().get("positions", [])
        return not any(pos["position"]["dealId"] == deal_id for pos in positions)


# ENHANCED: Original function now uses retry logic
async def get_deal_confirmation_and_details(trading_headers: dict, deal_reference: str):
    """Get deal confirmation with additional details - now uses retry logic"""
    data = await get_deal_confirmation_with_retry(trading_headers, deal_reference)

    return {
        "dealId": data["dealId"],
        "level": float(data["level"]),
        "direction": data["direction"]
    }


def calculate_stop_distance(entry_level: float, new_stop_level: float, direction: str, scaling_factor: int = 10000) -> int:
    if direction == "BUY":
        return int((entry_level - new_stop_level) * scaling_factor)
    elif direction == "SELL":
        return int((new_stop_level - entry_level) * scaling_factor)
    else:
        raise ValueError("direction must be 'BUY' or 'SELL'")


async def update_stop_loss(deal_id: str, stop_distance: int, auth_headers: dict):
    """
    Updates the stop loss by distance (as required by IG).
    """
    url = f"{API_BASE_URL}/positions/otc/{deal_id}"
    payload = {
        "stopDistance": stop_distance,
        "guaranteedStop": False,
        "trailingStop": False
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.put(url, headers=auth_headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()


async def partial_close_position(
    deal_id: str,
    epic: str,
    direction: str,  # Original position direction (BUY/SELL)
    size_to_close: float,  # e.g., 0.5 for 50%
    auth_headers: dict
) -> dict:
    """
    Close part of a position using IG API DELETE method with POST workaround.

    Args:
        deal_id: Position deal ID
        epic: Market epic (for logging)
        direction: ORIGINAL position direction (BUY/SELL)
        size_to_close: Amount to close (0.5 for 50%)
        auth_headers: IG API auth headers

    Returns:
        dict: {"success": bool, "response": dict, "error": str, "size_closed": float}

    Note: Uses POST with _method:DELETE header due to IG API DELETE body limitation
    """
    try:
        # ✅ SAFEGUARD: Check if position exists and has enough size before attempting close
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                pos_resp = await client.get(
                    f"{API_BASE_URL}/positions",
                    headers=auth_headers
                )
                if pos_resp.status_code == 200:
                    positions = pos_resp.json().get("positions", [])
                    match = next((p for p in positions if p["position"]["dealId"] == deal_id), None)
                    if not match:
                        logger.warning(f"⚠️ [PARTIAL CLOSE] Position {deal_id} not found on IG - may already be closed")
                        return {
                            "success": False,
                            "error": "Position not found on IG",
                            "already_closed": True
                        }
                    current_size = float(match["position"].get("size", 0))
                    if current_size < size_to_close:
                        logger.warning(f"⚠️ [PARTIAL CLOSE] Position {deal_id} size {current_size} < requested {size_to_close}")
                        return {
                            "success": False,
                            "error": f"Position size {current_size} less than requested close size {size_to_close}",
                            "current_size": current_size
                        }
                    logger.info(f"✅ [PARTIAL CLOSE CHECK] Position {deal_id} has size {current_size}, closing {size_to_close}")
        except Exception as check_error:
            logger.warning(f"⚠️ [PARTIAL CLOSE] Could not verify position size: {check_error}, proceeding anyway")

        # Opposite direction for closing (SELL closes BUY, BUY closes SELL)
        close_direction = "SELL" if direction.upper() == "BUY" else "BUY"

        # IG API DELETE /positions/otc payload for partial close
        # WORKING METHOD: POST with _method:DELETE header + orderType
        payload = {
            "dealId": deal_id,
            "direction": close_direction,
            "size": size_to_close,
            "orderType": "MARKET"  # Required field for DELETE
        }

        # Add _method:DELETE header (IG API workaround for DELETE with body)
        headers = {
            **auth_headers,
            "_method": "DELETE",
            "Version": "1"
        }

        logger.info(f"📤 [PARTIAL CLOSE] Closing {size_to_close} of {epic} {direction} position")
        logger.info(f"   Deal ID: {deal_id}")
        logger.info(f"   Payload: {payload}")
        logger.info(f"   Using POST with _method:DELETE header")

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Use POST with _method:DELETE header (IG API requirement)
            response = await client.post(
                f"{API_BASE_URL}/positions/otc",
                headers=headers,
                json=payload
            )

            if response.status_code in [200, 201]:
                logger.info(f"✅ [PARTIAL CLOSE SUCCESS] {epic}: Closed {size_to_close} position")
                return {
                    "success": True,
                    "response": response.json(),
                    "size_closed": size_to_close
                }
            else:
                error_msg = response.text
                logger.error(f"❌ [PARTIAL CLOSE FAILED] {epic}: HTTP {response.status_code} - {error_msg}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {error_msg}"
                }

    except Exception as e:
        logger.error(f"❌ [PARTIAL CLOSE ERROR] {epic}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def get_point_value(epic: str) -> float:
    """Get point value for the instrument"""
    if "JPY" in epic:
        return 0.01
    elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return 0.0001
    return 1.0


def convert_stop_distance_to_price(entry_price: float, stop_distance_points: int, 
                                 direction: str, epic: str) -> float:
    """Convert stop distance in points to actual stop price level"""
    point_value = get_point_value(epic)
    stop_distance_price = stop_distance_points * point_value
    
    if direction.upper() == "BUY":
        # For BUY: stop is below entry price
        stop_price = entry_price - stop_distance_price
    else:  # SELL
        # For SELL: stop is above entry price  
        stop_price = entry_price + stop_distance_price
    
    return round(stop_price, 5)  # Round to 5 decimal places for forex


def convert_price_to_points(price_distance: float, epic: str) -> int:
    """Convert price distance to points"""
    point_value = get_point_value(epic)
    return int(round(price_distance / point_value))


# NEW: Additional helper functions for better deal confirmation handling

async def check_position_exists(epic: str, direction: str, trading_headers: dict, max_wait: int = 10) -> dict:
    """
    FALLBACK: Check if a position exists for the given epic and direction
    Useful when deal confirmation fails but we want to verify if the order went through
    
    Args:
        epic: Market epic
        direction: BUY or SELL
        trading_headers: IG API headers
        max_wait: Maximum seconds to wait for position to appear
        
    Returns:
        Position data if found, None otherwise
    """
    for wait_time in range(max_wait):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{API_BASE_URL}/positions", headers=trading_headers)
                response.raise_for_status()
                positions = response.json().get("positions", [])
                
                # Look for matching position
                for pos in positions:
                    if (pos["market"]["epic"] == epic and 
                        pos["position"]["direction"] == direction):
                        logger.info(f"✅ Found position: {epic} {direction} after {wait_time}s")
                        return pos
                
                # Wait 1 second before next check
                if wait_time < max_wait - 1:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.warning(f"⚠️ Error checking positions (attempt {wait_time + 1}): {e}")
            
    logger.warning(f"⚠️ No position found for {epic} {direction} after {max_wait}s")
    return None


async def get_deal_confirmation_with_fallback(trading_headers: dict, deal_reference: str, 
                                            epic: str, direction: str) -> dict:
    """
    COMPREHENSIVE: Try deal confirmation with retry, then fallback to position check
    
    Args:
        trading_headers: IG API headers
        deal_reference: Deal reference from order placement
        epic: Market epic
        direction: BUY or SELL
        
    Returns:
        Deal confirmation data (either from API or constructed from position)
    """
    try:
        # First, try normal deal confirmation with retries
        return await get_deal_confirmation_with_retry(trading_headers, deal_reference)
        
    except Exception as e:
        logger.warning(f"⚠️ Deal confirmation failed: {e}")
        logger.info("🔄 Attempting fallback - checking open positions...")
        
        # Fallback: Check if position exists
        position = await check_position_exists(epic, direction, trading_headers, max_wait=5)
        
        if position:
            # Create mock confirmation from position data
            deal_id = position["position"]["dealId"]
            entry_price = float(position["position"]["level"])
            
            logger.info(f"✅ Fallback successful: Deal ID {deal_id}, Entry: {entry_price}")
            return {
                "dealId": deal_id,
                "level": entry_price,
                "status": "ACCEPTED",  # Assume accepted since position exists
                "reason": "Fallback confirmation from position data",
                "direction": direction,
                "epic": epic
            }
        else:
            # No position found in fallback
            logger.error("❌ Fallback failed - position not found")
            raise Exception(f"Deal confirmation failed and position not found. Deal reference: {deal_reference}")


# NEW: Utility function for deal reference validation
def is_valid_deal_reference(deal_reference: str) -> bool:
    """
    Validate deal reference format
    IG deal references are typically alphanumeric strings
    """
    if not deal_reference:
        return False

    # Basic validation - IG deal references are usually 12-20 characters
    if len(deal_reference) < 10 or len(deal_reference) > 25:
        return False

    # Should be alphanumeric
    if not deal_reference.isalnum():
        return False

    return True


# =============================================================================
# WORKING ORDERS (LIMIT ORDERS) - NEW FUNCTIONALITY
# =============================================================================

async def place_working_order(
    auth_headers: dict,
    epic: str,
    direction: str,
    level: float,
    stop_distance: int,
    limit_distance: int,
    expiry_minutes: int,
    currency_code: str,
    size: float = 1.0,
    order_type: str = "STOP"
) -> dict:
    """
    Place a working order (STOP or LIMIT order) at a specified price level.

    Unlike market orders that execute immediately at current price,
    working orders wait until price reaches the specified level.

    Order Types (IG API):
        - STOP: Entry triggered when price BREAKS THROUGH the level
          * STOP BUY: Triggered when price rises TO or ABOVE level (momentum confirmation)
          * STOP SELL: Triggered when price falls TO or BELOW level (momentum confirmation)
        - LIMIT: Entry triggered when price REACHES the level from the opposite side
          * LIMIT BUY: Triggered when price falls TO or BELOW level (better price entry)
          * LIMIT SELL: Triggered when price rises TO or ABOVE level (better price entry)

    For stop-entry style (momentum confirmation):
        - BUY above current price → use STOP
        - SELL below current price → use STOP

    Args:
        auth_headers: IG API authentication headers
        epic: Market epic (e.g., 'CS.D.EURUSD.MINI.IP')
        direction: 'BUY' or 'SELL'
        level: Entry price level for the order
        stop_distance: Stop loss distance in points
        limit_distance: Take profit distance in points
        expiry_minutes: Minutes until order expires (auto-cancelled)
        currency_code: Currency code (e.g., 'GBP')
        size: Position size (default 1.0)
        order_type: 'STOP' (momentum confirmation) or 'LIMIT' (better price). Default: STOP

    Returns:
        dict: Response containing dealReference for the working order

    Raises:
        httpx.HTTPStatusError: If the API request fails
    """
    from datetime import datetime, timedelta

    # Calculate expiry time in UTC
    expiry_time = datetime.utcnow() + timedelta(minutes=expiry_minutes)
    good_till_date = expiry_time.strftime("%Y/%m/%d %H:%M:%S")

    # Validate order type
    order_type = order_type.upper()
    if order_type not in ["STOP", "LIMIT"]:
        logger.warning(f"⚠️ Invalid order type '{order_type}', defaulting to STOP")
        order_type = "STOP"

    # CEEM epics use scaled pricing (e.g., 11621.6 instead of 1.16216)
    # IG API expects level in scaled format for CEEM contracts
    level_for_api = level
    if "CEEM" in epic:
        level_for_api = level * 10000
        logger.info(f"📐 [CEEM] Scaling entry level: {level} → {level_for_api}")

    payload = {
        "epic": epic,
        "expiry": "-",
        "direction": direction.upper(),
        "size": size,
        "type": order_type,
        "level": level_for_api,
        "currencyCode": currency_code,
        "timeInForce": "GOOD_TILL_DATE",
        "goodTillDate": good_till_date,
        "guaranteedStop": False,
        "stopDistance": stop_distance,
        "limitDistance": limit_distance
    }

    logger.info(f"📤 [WORKING ORDER] Placing {order_type} order: {epic} {direction} at {level_for_api}")
    logger.info(f"   Stop: {stop_distance}pts, Limit: {limit_distance}pts, Expiry: {good_till_date} UTC")
    logger.info(f"   Payload: {json.dumps(payload, indent=2)}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/workingorders/otc",
            headers=auth_headers,
            data=json.dumps(payload)
        )

        if response.status_code in [200, 201]:
            result = response.json()
            logger.info(f"✅ [WORKING ORDER] Successfully placed: {result.get('dealReference', 'unknown')}")
            return result
        else:
            error_text = response.text
            logger.error(f"❌ [WORKING ORDER] Failed: HTTP {response.status_code} - {error_text}")
            response.raise_for_status()


async def delete_working_order(auth_headers: dict, deal_id: str) -> dict:
    """
    Cancel/delete a working order (limit order).

    Args:
        auth_headers: IG API authentication headers
        deal_id: The deal ID of the working order to cancel

    Returns:
        dict: Response containing dealReference for the cancellation

    Raises:
        httpx.HTTPStatusError: If the API request fails
    """
    logger.info(f"🗑️ [WORKING ORDER] Cancelling order: {deal_id}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.delete(
            f"{API_BASE_URL}/workingorders/otc/{deal_id}",
            headers=auth_headers
        )

        if response.status_code in [200, 201]:
            result = response.json()
            logger.info(f"✅ [WORKING ORDER] Successfully cancelled: {deal_id}")
            return result
        else:
            error_text = response.text
            logger.error(f"❌ [WORKING ORDER] Cancel failed: HTTP {response.status_code} - {error_text}")
            response.raise_for_status()


async def get_working_orders(auth_headers: dict) -> dict:
    """
    Get all pending working orders (limit orders).

    Args:
        auth_headers: IG API authentication headers

    Returns:
        dict: Response containing list of working orders

    Raises:
        httpx.HTTPStatusError: If the API request fails
    """
    logger.info("📋 [WORKING ORDER] Fetching all pending working orders")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{API_BASE_URL}/workingorders",
            headers=auth_headers
        )

        if response.status_code == 200:
            result = response.json()
            orders = result.get("workingOrders", [])
            logger.info(f"✅ [WORKING ORDER] Found {len(orders)} pending orders")
            return result
        else:
            error_text = response.text
            logger.error(f"❌ [WORKING ORDER] Fetch failed: HTTP {response.status_code} - {error_text}")
            response.raise_for_status()


async def has_working_order_for_epic(auth_headers: dict, epic: str) -> bool:
    """
    Check if there's already a pending working order for a specific epic.

    Args:
        auth_headers: IG API authentication headers
        epic: Market epic to check

    Returns:
        bool: True if a working order exists for this epic
    """
    try:
        result = await get_working_orders(auth_headers)
        orders = result.get("workingOrders", [])

        for order in orders:
            if order.get("marketData", {}).get("epic") == epic:
                logger.info(f"⚠️ [WORKING ORDER] Existing order found for {epic}")
                return True

        return False

    except Exception as e:
        logger.warning(f"⚠️ [WORKING ORDER] Error checking existing orders: {e}")
        return False