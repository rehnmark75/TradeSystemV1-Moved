import httpx
import json
import asyncio
import logging
from config import API_BASE_URL

# Add logger for better debugging
logger = logging.getLogger(__name__)

async def has_open_position(epic: str, auth_headers: dict) -> bool:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/positions", headers=auth_headers)
        response.raise_for_status()
        positions = response.json().get("positions", [])
        return any(pos["market"]["epic"] == epic for pos in positions)


async def place_market_order(auth_headers, market_epic, direction, currency_code,sl_limit):
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
        "limitDistance": 25
    }

    async with httpx.AsyncClient() as client:
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
            async with httpx.AsyncClient() as client:
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

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def is_deal_closed(deal_id: str, headers: dict) -> bool:
    async with httpx.AsyncClient() as client:
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

    async with httpx.AsyncClient() as client:
        response = await client.put(url, headers=auth_headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()


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
            async with httpx.AsyncClient() as client:
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