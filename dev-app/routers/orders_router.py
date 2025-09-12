from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json
import traceback
from services.ig_auth import ig_login
from services.ig_orders import has_open_position, place_market_order, get_deal_confirmation, get_deal_confirmation_and_details, calculate_stop_distance, get_deal_confirmation_simple, get_deal_confirmation_with_fallback, get_deal_confirmation_with_retry
from services.ig_market import get_current_bid_price, get_last_15m_candle_range,get_last_15m_candle_range_local
from services.ig_risk_utils import get_stop_distance_from_ema, price_to_ig_points
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from services.db import get_db, engine
from services.stream_api import trigger_trade_tracking
from services.models import TradeLog
import httpx
import time
from utils import get_point_value, convert_stop_distance_to_price, convert_limit_distance_to_price

from services.keyvault import get_secret
from config import EPIC_MAP, API_BASE_URL, TRADE_COOLDOWN_ENABLED, TRADE_COOLDOWN_MINUTES, EPIC_SPECIFIC_COOLDOWNS, TRADING_BLACKLIST
from dependencies import get_ig_auth_headers, ig_token_cache
import logging
import asyncio

router = APIRouter()

# Configure logger
logger = logging.getLogger("tradelogger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/dev-trade.log")
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    "%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

class TradeRequest(BaseModel):
    # Required fields
    epic: str
    direction: str

    # Optional fields
    size: Optional[float] = None
    stop_distance: Optional[int] = None
    limit_distance: Optional[int] = None
    custom_label: Optional[str] = None
    risk_reward: Optional[float] = 2.0  # Default RR
    alert_id: Optional[int] = None  # NEW: Include alert_id in request body

# Simple in-memory cache for positions
_cached_positions = {"data": None, "timestamp": 0}

async def get_cached_positions(headers: dict, cache_ttl: int = 5):
    now = time.time()
    if _cached_positions["data"] and now - _cached_positions["timestamp"] < cache_ttl:
        return _cached_positions["data"]

    async with httpx.AsyncClient() as client:
        r = await client.get(f"{API_BASE_URL}/positions", headers=headers)
        r.raise_for_status()
        positions = r.json()["positions"]
        _cached_positions["data"] = positions
        _cached_positions["timestamp"] = now
        return positions

def check_trade_cooldown(epic: str, db: Session) -> dict:
    """
    Check if an epic is still in cooldown period after a recent trade closure.
    
    Returns:
        dict: {
            "allowed": bool,
            "cooldown_remaining_minutes": int,
            "last_trade_closed_at": datetime,
            "message": str
        }
    """
    if not TRADE_COOLDOWN_ENABLED:
        return {"allowed": True, "message": "Cooldown disabled"}
    
    try:
        # Get cooldown period for this epic (epic-specific or default)
        cooldown_minutes = EPIC_SPECIFIC_COOLDOWNS.get(epic, TRADE_COOLDOWN_MINUTES)
        
        # Query for the most recent closed trade for this epic
        recent_trade = (
            db.query(TradeLog)
            .filter(
                TradeLog.symbol == epic,
                TradeLog.status == "closed",
                TradeLog.closed_at.isnot(None)
            )
            .order_by(TradeLog.closed_at.desc())
            .first()
        )
        
        if not recent_trade:
            return {"allowed": True, "message": "No recent closed trades found"}
            
        # Calculate time elapsed since last closure
        time_elapsed = datetime.utcnow() - recent_trade.closed_at
        elapsed_minutes = time_elapsed.total_seconds() / 60
        
        if elapsed_minutes < cooldown_minutes:
            # Still in cooldown period
            remaining_minutes = int(cooldown_minutes - elapsed_minutes)
            return {
                "allowed": False,
                "cooldown_remaining_minutes": remaining_minutes,
                "last_trade_closed_at": recent_trade.closed_at,
                "message": f"Epic {epic} is in cooldown for {remaining_minutes} more minutes. Last trade closed at {recent_trade.closed_at.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            }
        else:
            # Cooldown period has expired
            return {
                "allowed": True,
                "message": f"Cooldown expired ({int(elapsed_minutes)} minutes ago)"
            }
            
    except Exception as e:
        # Log error but allow trade to proceed (fail-open for safety)
        logger.error(f"Trade cooldown check failed for {epic}: {str(e)}")
        return {
            "allowed": True,
            "message": f"Cooldown check failed, allowing trade: {str(e)}"
        }

@router.post("/place-order")
async def ig_place_order(
    body: TradeRequest,
    alert_id: Optional[int] = None,  # Optional alert_id parameter
    trading_headers: dict = Depends(get_ig_auth_headers),
    db: Session = Depends(get_db)
):
    # BUGFIX: Initialize result variable at the beginning to prevent UnboundLocalError
    result = None
    
    epic = body.epic.strip().upper()
    direction = body.direction.strip().upper()
    logger.info(f"Place-Order: Parsed EPIC: {epic}, Direction: {direction}")
    
    # Log alert_id if provided
    alert_id = body.alert_id

    if direction not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="Direction must be BUY or SELL")

    size = body.size
    stop = body.stop_distance
    limit = body.limit_distance
    label = body.custom_label
    rr = body.risk_reward or 2.0

    symbol = EPIC_MAP.get(epic.upper())
    if not symbol:
        raise HTTPException(status_code=404, detail=f"No mapping found for epic: {epic}")

    # Check if epic is blacklisted from trading
    if epic.upper() in TRADING_BLACKLIST:
        logger.warning(f"🚫 Trading blocked for {epic}: Epic is blacklisted")
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Trading blocked",
                "message": f"Trading is temporarily disabled for {epic}",
                "epic": epic,
                "reason": "Market access restrictions"
            }
        )

    # Check trade cooldown before proceeding
    cooldown_result = check_trade_cooldown(symbol, db)
    if not cooldown_result["allowed"]:
        logger.info(f"🛑 Trade blocked by cooldown: {cooldown_result['message']}")
        raise HTTPException(
            status_code=429,  # Too Many Requests
            detail={
                "error": "Trade cooldown active",
                "message": cooldown_result["message"],
                "cooldown_remaining_minutes": cooldown_result.get("cooldown_remaining_minutes", 0),
                "last_trade_closed_at": cooldown_result.get("last_trade_closed_at"),
                "epic": symbol
            }
        )
    else:
        logger.info(f"✅ Cooldown check passed: {cooldown_result['message']}")

    try:
        # Check for existing position
        if await has_open_position(symbol, trading_headers):
            msg = f"Position already open for {symbol}, skipping order."
            logger.info(f"Position already open for {symbol}")
            logger.info(json.dumps(msg))
            return {"status": "skipped", "message": msg, "alert_id": alert_id}

        logger.info(json.dumps(f"No open position for {symbol}, placing order."))

        # Get current market data
        price_info = await get_current_bid_price(trading_headers, symbol)
        bid_price = price_info["bid_price"]
        currency_code = price_info["currency_code"]
        min_distance = price_info["min_distance"]
        logger.info(f"this is min_distance for epic {epic}: {min_distance}")
        
        # Configuration
        BUFFER_POINTS = 12
        FALLBACK_SL_LIMIT = 10

        # Calculate stop loss with buffer and fallback
        stop = None  # This should come from your signal or calculation

        if stop is not None:
            try:
                min_distance_with_buffer = min_distance + BUFFER_POINTS if min_distance is not None else FALLBACK_SL_LIMIT
                sl_limit = min_distance_with_buffer if stop < min_distance_with_buffer else stop
                logger.info(f"Using stop loss: {sl_limit} (min_distance: {min_distance}, buffer: {BUFFER_POINTS})")
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to calculate stop loss with buffer: {e}")
                sl_limit = FALLBACK_SL_LIMIT
                logger.info(f"Using fallback sl_limit: {sl_limit}")
        else:
            try:
                sl_limit = min_distance + BUFFER_POINTS if min_distance is not None else FALLBACK_SL_LIMIT
                logger.info(f"No stop provided, using min_distance + buffer: {sl_limit}")
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to process min_distance: {e}")
                sl_limit = FALLBACK_SL_LIMIT
                logger.info(f"Using fallback sl_limit: {sl_limit}")
        
        # BUGFIX: Ensure result is assigned in all code paths
        # Place the market order
        logger.info(f"📤 Placing market order: {symbol} {direction} with SL: {sl_limit}")
        result = await place_market_order(trading_headers, symbol, direction, currency_code, sl_limit)
        
        # BUGFIX: Add validation that result was actually returned
        if result is None:
            logger.error("❌ place_market_order returned None")
            raise HTTPException(status_code=500, detail="Order placement failed - no result returned")
        
        deal_reference = result.get("dealReference")
        if not deal_reference:
            logger.error(f"❌ Order placement failed - no deal reference returned: {result}")
            raise HTTPException(status_code=500, detail="Order placement failed - no deal reference")
        
        logger.info(f"✅ Order placed successfully, deal reference: {deal_reference}")
        
        # Get deal confirmation with retry logic
        logger.info(f"⏳ Waiting for deal confirmation: {deal_reference}")
        try:
            confirm = await get_deal_confirmation_with_retry(trading_headers, deal_reference, max_retries=5)
            logger.info(f"✅ Deal confirmation received: {confirm}")
        except Exception as confirm_error:
            logger.error(f"❌ Deal confirmation failed: {str(confirm_error)}")
            
            # FALLBACK: Try to find the deal in open positions
            logger.info("🔄 Attempting fallback - checking open positions for deal...")
            try:
                # Wait a bit for the position to appear
                await asyncio.sleep(2)
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{API_BASE_URL}/positions", headers=trading_headers)
                    response.raise_for_status()
                    positions = response.json().get("positions", [])
                    
                # Look for a recent position on this epic
                recent_position = None
                for pos in positions:
                    if (pos["market"]["epic"] == symbol and 
                        pos["position"]["direction"] == direction):
                        recent_position = pos
                        break
                
                if recent_position:
                    # Found the position, create mock confirmation
                    logger.info("✅ Found position in fallback check")
                    deal_id = recent_position["position"]["dealId"]
                    entry_price = float(recent_position["position"]["level"])
                    
                    confirm = {
                        "dealId": deal_id,
                        "level": entry_price,
                        "status": "ACCEPTED",  # Assume accepted since position exists
                        "reason": "Fallback confirmation from position data"
                    }
                    logger.info(f"✅ Using fallback confirmation: Deal ID {deal_id}, Entry: {entry_price}")
                else:
                    # Position not found in fallback
                    logger.error("❌ Fallback failed - position not found in open positions")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Deal confirmation failed and position not found in fallback check. Deal reference: {deal_reference}"
                    )
                    
            except Exception as fallback_error:
                logger.error(f"❌ Fallback position check failed: {str(fallback_error)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Deal confirmation failed and fallback check failed. Deal reference: {deal_reference}"
                )
        
        # Extract deal information
        deal_id = confirm.get("dealId")
        entry_price_raw = confirm.get("level")
        
        if entry_price_raw is None:
            logger.error(f"❌ Order confirmation missing 'level': {confirm}")
            raise HTTPException(status_code=500, detail="Order confirmation missing entry level.")

        if deal_id is None:
            logger.error(f"❌ Order confirmation missing 'dealId': {confirm}")
            raise HTTPException(status_code=500, detail="Order confirmation missing deal ID.")

        entry_price = float(entry_price_raw)
        logger.info(f"✅ Order confirmed: Deal ID {deal_id}, Entry Price: {entry_price}")
        
        # Convert to actual prices for database
        actual_stop_price = convert_stop_distance_to_price(entry_price, sl_limit, direction, epic)
        actual_limit_price = convert_limit_distance_to_price(entry_price, 15, direction, epic)
        
        logger.info(f"🔍 DEBUG: alert_id parameter value: {alert_id} (type: {type(alert_id)})")
        logger.info(f"🔍 DEBUG: alert_id is None: {alert_id is None}")
        logger.info(f"🔍 DEBUG: alert_id bool value: {bool(alert_id)}")

        # Save to database
        try:
            trade_log = TradeLog(
                symbol=symbol,
                entry_price=entry_price,
                direction=direction.upper(),
                limit_price=actual_limit_price,
                sl_price=actual_stop_price,
                deal_id=deal_id,
                min_stop_distance_points=min_distance,
                deal_reference=deal_reference,
                endpoint="dev",
                status="pending",
                alert_id=alert_id
            )
            
            db.add(trade_log)
            logger.info(f"🔍 DEBUG: TradeLog added to session, alert_id: {trade_log.alert_id}")
            db.commit()
            logger.info(f"✅ Trade logged: {symbol} {entry_price} {direction}")
            if alert_id is not None:
                logger.info(f"✅ Trade linked to alert_id: {alert_id}")
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"❌ DB write failed: {str(e)}")
            # Don't fail the whole request for DB issues
            
        finally:
            db.close()

        # Return success response
        return {
            "status": "success",
            "dealReference": deal_reference,
            "dealId": deal_id,
            "entry_price": entry_price,
            "stop_price": actual_stop_price,
            "limit_price": actual_limit_price,
            "alert_id": alert_id
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (these are expected errors)
        raise
        
    except Exception as e:
        # Log unexpected errors
        logger.error(f"❌ Unexpected error in place_order: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/update-stop")
async def update_stop_price(
    request: Request,
    trading_headers: dict = Depends(get_ig_auth_headers),
    db: Session = Depends(get_db)
):
    def ig_points_to_price(points: int, epic: str) -> float:
        if "JPY" in epic:
            point_value = 0.01
        elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            point_value = 0.0001
        else:
            point_value = 1.0
        return round(points * point_value, 5)

    try:
        body = await request.json()
        epic = body.get("epic")
        new_stop = body.get("new_stop")
        new_limit = body.get("new_limit")

        if not epic:
            logger.info("missing epic body")
            raise HTTPException(status_code=400, detail="Missing 'epic' in body")

        headers = {
            "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
            "CST": trading_headers["CST"],
            "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
            "Accept": "application/json",
            "Version": "2"
        }

        positions = await get_cached_positions(headers)
        match = next((p for p in positions if p["market"]["epic"] == epic), None)

        if not match:
            updated = db.query(TradeLog).filter(
                TradeLog.symbol == epic,
                TradeLog.status.in_(["pending", "tracking"])
            ).update({TradeLog.status: "closed"})
            db.commit()
            message = f"No open position found for {epic}. Marked {updated} trades as closed."
            logger.info(message)
            return {
                "status": "closed",
                "message": message
            }

        position = match["position"]
        deal_id = position["dealId"]

        current_stop = float(position["stopLevel"]) if position.get("stopLevel") else None
        current_limit = float(position["limitLevel"]) if position.get("limitLevel") else None

        payload = {
            "guaranteedStop": False,
            "trailingStop": False,
            "stopLevel": float(new_stop) if new_stop is not None else current_stop,
            "limitLevel": float(new_limit) if new_limit is not None else current_limit
        }
        logger.info("Generated payload:\n%s", json.dumps(payload, indent=4))

        if payload["stopLevel"] is None or payload["limitLevel"] is None:
            logger.info("Both stop and limit levels must be set to avoid accidental clearing.")
            raise HTTPException(status_code=400, detail="Both stop and limit levels must be set to avoid accidental clearing.")

        update_url = f"{API_BASE_URL}/positions/otc/{deal_id}"
        async with httpx.AsyncClient() as client:
            response = await client.put(update_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

        return {
            "status": "updated",
            "dealId": deal_id,
            "sentPayload": payload,
            "apiResponse": result
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTPStatusError while calling upstream: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error occurred")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "stream", "time": datetime.now().isoformat()}


@router.post("/modify-stop")
async def update_stop_price(
    request: Request,
    trading_headers: dict = Depends(get_ig_auth_headers),
    db: Session = Depends(get_db)
):
    def ig_points_to_price(points: float, epic: str) -> float:
        if "JPY" in epic:
            return round(points * 0.01, 5)
        elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return round(points * 0.0001, 5)
        else:
            return round(points * 1.0, 5)

    try:
        body = await request.json()
        epic = body.get("epic")
        direction_override = body.get("adjustDirection", "auto")  # "increase", "decrease", or "auto"
        stop_offset_points = body.get("stop_offset_points")
        limit_offset_points = body.get("limit_offset_points")

        if not epic:
            raise HTTPException(status_code=400, detail="Missing epic")

        # --- Get open positions ---
        url = f"{API_BASE_URL}/positions"
        headers = {
            "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
            "CST": trading_headers["CST"],
            "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
            "Accept": "application/json",
            "Version": "2"
        }

        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            positions = r.json()["positions"]

        # --- Find matching position ---
        match = next((p for p in positions if p["market"]["epic"] == epic), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"No open position for {epic}")

        position = match["position"]
        deal_id = position["dealId"]
        direction = position["direction"]

        # --- Direction-based signs ---
        if direction_override == "increase":
            step_sign = 1 if direction == "BUY" else -1
        elif direction_override == "decrease":
            step_sign = -1 if direction == "BUY" else 1
        else:  # auto follows direction
            step_sign = 1 if direction == "BUY" else -1

        payload = {
            "guaranteedStop": False,
            "trailingStop": False
        }

        # --- Stop logic ---
        if position.get("stopLevel"):
            old = float(position["stopLevel"])
            step = ig_points_to_price(float(stop_offset_points), epic) if stop_offset_points is not None else step_sign * 0.0002
            payload["stopLevel"] = round(old + step, 5)
        elif position.get("stopDistance"):
            old = float(position["stopDistance"])
            offset = float(stop_offset_points) if stop_offset_points is not None else 2
            payload["stopDistance"] = max(1, round(old - step_sign * offset))

        # --- Limit logic ---
        if position.get("limitLevel"):
            old = float(position["limitLevel"])
            step = ig_points_to_price(float(limit_offset_points), epic) if limit_offset_points is not None else step_sign * 0.0002
            payload["limitLevel"] = round(old + step, 5)
        elif position.get("limitDistance"):
            old = float(position["limitDistance"])
            offset = float(limit_offset_points) if limit_offset_points is not None else 2
            payload["limitDistance"] = round(old + step_sign * offset)

        logger.info(f"Preparing to update deal ID {deal_id} for {epic}")
        # --- Update position ---
        update_url = f"{API_BASE_URL}/positions/otc/{deal_id}"
        async with httpx.AsyncClient() as client:
            response = await client.put(update_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

        return {
            "status": "updated",
            "dealId": deal_id,
            "adjustDirection": direction_override,
            "sentPayload": payload,
            "apiResponse": result
        }

    except HTTPException as e:
        raise e
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(ex)}")

@router.post("/adjust-stop")
async def adjust_stop_price(
    request: Request,
    trading_headers: dict = Depends(get_ig_auth_headers),
    db: Session = Depends(get_db)
):
    def ig_points_to_price(points: float, epic: str) -> float:
        if "JPY" in epic:
            return round(points * 0.01, 5)
        elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
            return round(points * 0.0001, 5)
        else:
            return round(points * 1.0, 5)

    try:
        body = await request.json()
        epic = body.get("epic")
        if not epic:
            raise HTTPException(status_code=400, detail="Missing 'epic' in body")

        new_stop = body.get("new_stop")
        new_limit = body.get("new_limit")
        stop_offset_points = body.get("stop_offset_points")
        limit_offset_points = body.get("limit_offset_points")
        dry_run = body.get("dry_run", False)

        # Direction logic
        default_dir = body.get("adjustDirection", "auto")
        direction_stop = body.get("adjustDirectionStop", default_dir)
        direction_limit = body.get("adjustDirectionLimit", default_dir)

        headers = {
            "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
            "CST": trading_headers["CST"],
            "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
            "Accept": "application/json",
            "Version": "2"
        }

        async with httpx.AsyncClient() as client:
            pos_resp = await client.get(f"{API_BASE_URL}/positions", headers=headers)
            pos_resp.raise_for_status()
            positions = pos_resp.json()["positions"]

        match = next((p for p in positions if p["market"]["epic"] == epic), None)
        if not match:
            updated = db.query(TradeLog).filter(
                TradeLog.symbol == epic,
                TradeLog.status.in_(["pending", "tracking", "break_even", "trailing"])  # ✅ FIXED: Include all active statuses
            ).update({TradeLog.status: "closed"})
            db.commit()
            return {
                "status": "closed",
                "message": f"No open position found for {epic}. Marked {updated} trades as closed."
            }

        position = match["position"]
        deal_id = position["dealId"]
        direction = position["direction"]

        # ✅ FIXED: Better direction handling
        payload = {
            "guaranteedStop": False,
            "trailingStop": False
        }

        # --- STOP logic (COMPLETELY REWRITTEN)
        if new_stop is not None:
            payload["stopLevel"] = float(new_stop)
        elif position.get("stopLevel"):
            old_stop = float(position["stopLevel"])
            offset = ig_points_to_price(float(stop_offset_points), epic) if stop_offset_points else 0.0002
            
            # ✅ FIXED: Apply direction correctly based on adjustDirectionStop
            if direction_stop == "increase":
                # "increase" means move stop price UP (away from current price for both BUY/SELL)
                new_stop_level = old_stop + offset
            else:  # "decrease"
                # "decrease" means move stop price DOWN (closer to current price for both BUY/SELL)
                new_stop_level = old_stop - offset
            
            payload["stopLevel"] = round(new_stop_level, 5)
            
        elif position.get("stopDistance"):
            old_distance = float(position["stopDistance"])
            offset_points = float(stop_offset_points) if stop_offset_points else 2
            
            # ✅ FIXED: Distance logic - decrease distance = tighter stop, increase distance = looser stop
            if direction_stop == "increase":
                # "increase" means make stop looser (increase distance from current price)
                new_distance = old_distance + offset_points
            else:  # "decrease"
                # "decrease" means make stop tighter (decrease distance from current price)
                new_distance = max(1, old_distance - offset_points)  # Don't go below 1
            
            payload["stopDistance"] = max(1, round(new_distance))

        # --- LIMIT logic (FIXED)
        if new_limit is not None:
            payload["limitLevel"] = float(new_limit)
        elif position.get("limitLevel"):
            old_limit = float(position["limitLevel"])
            offset = ig_points_to_price(float(limit_offset_points), epic) if limit_offset_points else 0.0002
            
            # ✅ FIXED: Apply direction correctly based on adjustDirectionLimit
            if direction_limit == "increase":
                # "increase" means move limit price UP (better for BUY, worse for SELL)
                new_limit_level = old_limit + offset
            else:  # "decrease"
                # "decrease" means move limit price DOWN (worse for BUY, better for SELL)
                new_limit_level = old_limit - offset
            
            payload["limitLevel"] = round(new_limit_level, 5)
            
        elif position.get("limitDistance"):
            old_distance = float(position["limitDistance"])
            offset_points = float(limit_offset_points) if limit_offset_points else 2
            
            # ✅ FIXED: Distance logic for limits
            if direction_limit == "increase":
                # "increase" means increase distance to limit (better profit target)
                new_distance = old_distance + offset_points
            else:  # "decrease"
                # "decrease" means decrease distance to limit (closer profit target)
                new_distance = max(1, old_distance - offset_points)
            
            payload["limitDistance"] = round(new_distance)

        # --- Validate payload
        if "stopLevel" not in payload and "stopDistance" not in payload:
            raise HTTPException(status_code=400, detail="No stop value provided or available")
        if "limitLevel" not in payload and "limitDistance" not in payload:
            raise HTTPException(status_code=400, detail="No limit value provided or available")

        # ✅ ADDED: Enhanced logging for debugging
        logger.info(f"[ADJUST-STOP] {epic} Direction: {direction} (BUY/SELL position)")
        logger.info(f"[ADJUST-STOP] {epic} Stop direction: {direction_stop}, Limit direction: {direction_limit}")
        logger.info(f"[ADJUST-STOP] {epic} Stop offset: {stop_offset_points}, Limit offset: {limit_offset_points}")
        
        if position.get("stopLevel"):
            logger.info(f"[ADJUST-STOP] {epic} Old stop level: {position['stopLevel']} → New: {payload.get('stopLevel', 'N/A')}")
        if position.get("limitLevel"):
            logger.info(f"[ADJUST-STOP] {epic} Old limit level: {position['limitLevel']} → New: {payload.get('limitLevel', 'N/A')}")

        # --- Dry run response
        if dry_run:
            return {
                "status": "dry_run",
                "dealId": deal_id,
                "adjustDirectionStop": direction_stop,
                "adjustDirectionLimit": direction_limit,
                "sentPayload": payload,
                "note": "This is a dry run. No request was sent to IG."
            }

        # --- Send request to IG
        update_url = f"{API_BASE_URL}/positions/otc/{deal_id}"
        logger.info(f"➡ PUT {update_url}")
        logger.info(f"➡ Payload: {json.dumps(payload, indent=2)}")

        async with httpx.AsyncClient() as client:
            response = await client.put(update_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

        return {
            "status": "updated",
            "dealId": deal_id,
            "adjustDirectionStop": direction_stop,
            "adjustDirectionLimit": direction_limit,
            "sentPayload": payload,
            "apiResponse": result
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"[HTTP ERROR] {e.response.status_code} — {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled exception in /adjust-stop")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")
