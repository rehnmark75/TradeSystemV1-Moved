from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import json
import traceback
from services.ig_auth import ig_login
from services.ig_orders import (
    has_open_position, place_market_order, get_deal_confirmation,
    get_deal_confirmation_and_details, calculate_stop_distance,
    get_deal_confirmation_simple, get_deal_confirmation_with_fallback,
    get_deal_confirmation_with_retry,
    # NEW: Working order (limit order) functions
    place_working_order, delete_working_order, get_working_orders,
    has_working_order_for_epic
)
from services.sl_tp_calculator import calculate_trade_levels, validate_sl_tp_levels
from services.ig_market import get_current_bid_price
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from services.db import get_db, engine
from services.stream_api import trigger_trade_tracking
from services.models import TradeLog
import httpx
import time
from utils import get_point_value, convert_stop_distance_to_price, convert_limit_distance_to_price
from services.price_utils import ig_points_to_price

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
    use_provided_sl_tp: Optional[bool] = False  # Use scanner-provided SL/TP values
    custom_label: Optional[str] = None
    risk_reward: Optional[float] = 2.0  # Default RR
    alert_id: Optional[int] = None  # Include alert_id in request body

    # NEW: Limit order (working order) fields
    order_type: Optional[str] = "market"  # "market" or "limit"
    entry_level: Optional[float] = None   # Required for limit orders - the entry price
    limit_expiry_minutes: Optional[int] = 35  # Minutes until limit order expires (default 35)
    api_order_type: Optional[str] = "STOP"  # IG API order type: "STOP" (momentum) or "LIMIT" (better price)
    signal_price: Optional[float] = None  # Original signal price for slippage tracking

    # NEW: Scalping mode fields for Virtual Stop Loss
    is_scalp_trade: Optional[bool] = False  # True if this is a scalp trade requiring VSL
    virtual_sl_pips: Optional[float] = None  # Custom VSL distance (defaults to config if not set)

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
    Check if an epic is still in cooldown period after a recent trade closure OR opening.

    UPDATED: Now checks both trade closures AND trade openings to prevent back-to-back entries.

    Returns:
        dict: {
            "allowed": bool,
            "cooldown_remaining_minutes": int,
            "last_trade_closed_at": datetime,
            "last_trade_opened_at": datetime,
            "cooldown_type": str,  # "closure", "opening", "none", or "expired"
            "message": str
        }
    """
    if not TRADE_COOLDOWN_ENABLED:
        return {"allowed": True, "message": "Cooldown disabled"}
    
    try:
        # Get cooldown period for this epic (epic-specific or default)
        cooldown_minutes = EPIC_SPECIFIC_COOLDOWNS.get(epic, TRADE_COOLDOWN_MINUTES)
        
        # Find the most recent trade regardless of status, prioritizing closed_at when available
        recent_closed_trade = (
            db.query(TradeLog)
            .filter(
                TradeLog.symbol == epic,
                TradeLog.status.in_(["closed", "expired"]),
                TradeLog.closed_at.isnot(None)
            )
            .order_by(TradeLog.closed_at.desc())
            .first()
        )

        recent_expired_trade = (
            db.query(TradeLog)
            .filter(
                TradeLog.symbol == epic,
                TradeLog.status == "expired",
                TradeLog.closed_at.is_(None)
            )
            .order_by(TradeLog.timestamp.desc())
            .first()
        )

        # Determine which trade is more recent
        most_recent_trade = None
        effective_close_time = None

        if recent_closed_trade and recent_expired_trade:
            # Compare the closed trade's closed_at with expired trade's estimated close time
            estimated_close_1h = recent_expired_trade.timestamp + timedelta(hours=1)
            recent_expiry = datetime.utcnow() - timedelta(minutes=10)
            expired_estimated_close = min(estimated_close_1h, recent_expiry)

            if recent_closed_trade.closed_at > expired_estimated_close:
                most_recent_trade = recent_closed_trade
                effective_close_time = recent_closed_trade.closed_at
            else:
                most_recent_trade = recent_expired_trade
                effective_close_time = expired_estimated_close

        elif recent_closed_trade:
            most_recent_trade = recent_closed_trade
            effective_close_time = recent_closed_trade.closed_at

        elif recent_expired_trade:
            most_recent_trade = recent_expired_trade
            # For expired trades, estimate close time as either:
            # 1. Entry + 1 hour (reasonable trade duration)
            # 2. Current time - 10 minutes (if trade just expired)
            # Use whichever is earlier to avoid future timestamps
            estimated_close_1h = recent_expired_trade.timestamp + timedelta(hours=1)
            recent_expiry = datetime.utcnow() - timedelta(minutes=10)
            effective_close_time = min(estimated_close_1h, recent_expiry)

        # === NEW: Check for recent trade OPENINGS ===
        # IMPORTANT: Exclude limit orders that never filled (limit_not_filled, limit_rejected, limit_cancelled)
        # These should NOT trigger opening-based cooldown since no position was ever opened
        recent_opened_trade = (
            db.query(TradeLog)
            .filter(
                TradeLog.symbol == epic,
                TradeLog.timestamp.isnot(None),
                ~TradeLog.status.in_(["limit_not_filled", "limit_rejected", "limit_cancelled"])
            )
            .order_by(TradeLog.timestamp.desc())
            .first()
        )

        current_time = datetime.utcnow()

        # === Check closure-based cooldown ===
        closure_cooldown_active = False
        closure_remaining_minutes = 0
        if most_recent_trade and effective_close_time:
            # Handle timezone-naive timestamps by assuming UTC
            if effective_close_time.tzinfo is None:
                effective_close_time = effective_close_time.replace(tzinfo=None)
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=None)

            # ✅ TIMESTAMP VALIDATION FIX: Check for invalid future timestamps
            if effective_close_time > current_time:
                logger.warning(f"🚨 Invalid future closure timestamp for {epic}: {effective_close_time} (trade ID: {most_recent_trade.id})")
                logger.warning(f"   Current time: {datetime.utcnow()}")
                logger.warning(f"   Ignoring corrupt timestamp")
            else:
                time_elapsed = current_time - effective_close_time
                elapsed_minutes = time_elapsed.total_seconds() / 60

                if elapsed_minutes < cooldown_minutes:
                    closure_cooldown_active = True
                    closure_remaining_minutes = int(cooldown_minutes - elapsed_minutes)

        # === Check opening-based cooldown (NEW LOGIC) ===
        opening_cooldown_active = False
        opening_remaining_minutes = 0
        if recent_opened_trade and recent_opened_trade.timestamp:
            opening_time = recent_opened_trade.timestamp

            # Handle timezone-naive timestamps by assuming UTC
            if opening_time.tzinfo is None:
                opening_time = opening_time.replace(tzinfo=None)
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=None)

            # ✅ TIMESTAMP VALIDATION FIX: Check for invalid future timestamps
            if opening_time > current_time:
                logger.warning(f"🚨 Invalid future opening timestamp for {epic}: {opening_time} (trade ID: {recent_opened_trade.id})")
                logger.warning(f"   Current time: {datetime.utcnow()}")
                logger.warning(f"   Ignoring corrupt timestamp")
            else:
                time_elapsed = current_time - opening_time
                elapsed_minutes = time_elapsed.total_seconds() / 60

                if elapsed_minutes < cooldown_minutes:
                    opening_cooldown_active = True
                    opening_remaining_minutes = int(cooldown_minutes - elapsed_minutes)

        # === Return result based on which cooldown is active ===
        if closure_cooldown_active and opening_cooldown_active:
            # Both are active, return the one with more time remaining
            if closure_remaining_minutes >= opening_remaining_minutes:
                return {
                    "allowed": False,
                    "cooldown_remaining_minutes": closure_remaining_minutes,
                    "last_trade_closed_at": effective_close_time,
                    "last_trade_opened_at": recent_opened_trade.timestamp if recent_opened_trade else None,
                    "cooldown_type": "closure",
                    "message": f"Epic {epic} is in cooldown for {closure_remaining_minutes} more minutes. Last trade closed at {effective_close_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                }
            else:
                return {
                    "allowed": False,
                    "cooldown_remaining_minutes": opening_remaining_minutes,
                    "last_trade_closed_at": effective_close_time,
                    "last_trade_opened_at": recent_opened_trade.timestamp,
                    "cooldown_type": "opening",
                    "message": f"Epic {epic} is in cooldown for {opening_remaining_minutes} more minutes. Last trade opened at {recent_opened_trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC (prevents back-to-back entries)"
                }
        elif closure_cooldown_active:
            return {
                "allowed": False,
                "cooldown_remaining_minutes": closure_remaining_minutes,
                "last_trade_closed_at": effective_close_time,
                "last_trade_opened_at": recent_opened_trade.timestamp if recent_opened_trade else None,
                "cooldown_type": "closure",
                "message": f"Epic {epic} is in cooldown for {closure_remaining_minutes} more minutes. Last trade closed at {effective_close_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            }
        elif opening_cooldown_active:
            return {
                "allowed": False,
                "cooldown_remaining_minutes": opening_remaining_minutes,
                "last_trade_closed_at": effective_close_time,
                "last_trade_opened_at": recent_opened_trade.timestamp,
                "cooldown_type": "opening",
                "message": f"Epic {epic} is in cooldown for {opening_remaining_minutes} more minutes. Last trade opened at {recent_opened_trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC (prevents back-to-back entries)"
            }
        else:
            # No cooldown active
            if not most_recent_trade and not recent_opened_trade:
                return {
                    "allowed": True,
                    "message": "No recent trades found",
                    "cooldown_type": "none"
                }
            else:
                # Calculate how long ago the most recent action was
                last_action_time = None
                if effective_close_time and recent_opened_trade:
                    last_action_time = max(effective_close_time, recent_opened_trade.timestamp)
                elif effective_close_time:
                    last_action_time = effective_close_time
                elif recent_opened_trade:
                    last_action_time = recent_opened_trade.timestamp

                if last_action_time:
                    time_elapsed = current_time - last_action_time
                    elapsed_minutes = int(time_elapsed.total_seconds() / 60)
                    return {
                        "allowed": True,
                        "message": f"Cooldown expired ({elapsed_minutes} minutes ago)",
                        "cooldown_type": "expired"
                    }
                else:
                    return {
                        "allowed": True,
                        "message": "Cooldown expired",
                        "cooldown_type": "expired"
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

    # v3.2.0: Log scalp trade flag for VSL debugging
    logger.info(f"Place-Order: is_scalp_trade={body.is_scalp_trade}, virtual_sl_pips={body.virtual_sl_pips}")

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
        # Convert datetime to string for JSON serialization
        last_closed = cooldown_result.get("last_trade_closed_at")
        last_closed_str = last_closed.isoformat() if last_closed else None
        raise HTTPException(
            status_code=429,  # Too Many Requests
            detail={
                "error": "Trade cooldown active",
                "message": cooldown_result["message"],
                "cooldown_remaining_minutes": cooldown_result.get("cooldown_remaining_minutes", 0),
                "last_trade_closed_at": last_closed_str,
                "epic": symbol
            }
        )
    else:
        logger.info(f"✅ Cooldown check passed: {cooldown_result['message']}")

    try:
        # Check for existing position with error handling
        try:
            has_position = await has_open_position(symbol, trading_headers)
            if has_position:
                msg = f"Position already open for {symbol}, skipping order."
                logger.info(f"ℹ️ Position already open for {symbol}")
                # Return HTTP 409 (Conflict) - semantically correct for duplicate position
                raise HTTPException(
                    status_code=409,
                    detail={
                        "status": "skipped",
                        "message": msg,
                        "epic": symbol,
                        "alert_id": alert_id,
                        "reason": "duplicate_position"
                    }
                )
        except HTTPException:
            # Re-raise HTTP exceptions (these are expected)
            raise
        except Exception as pos_check_error:
            # FIXED: More selective error handling - only fail-open for network issues
            error_msg = str(pos_check_error).lower()

            # Only fail-open for network/timeout errors
            if 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
                logger.warning(f"⚠️ Position check failed due to network issue for {symbol}: {str(pos_check_error)}")
                logger.warning("   Proceeding with order placement (fail-open for network safety)")
                # Continue to order placement
            else:
                # For other errors, fail-safe (don't place order)
                logger.error(f"❌ Position check failed for {symbol}: {str(pos_check_error)}")
                logger.error(f"   Error type: {type(pos_check_error).__name__}")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Position check failed",
                        "message": f"Unable to verify existing positions for {symbol}",
                        "epic": symbol,
                        "alert_id": alert_id,
                        "reason": "position_check_failed",
                        "error_type": type(pos_check_error).__name__
                    }
                )

        logger.info(json.dumps(f"No open position for {symbol}, placing order."))

        # Get current market data including spread
        price_info = await get_current_bid_price(trading_headers, symbol)
        bid_price = price_info["bid_price"]
        offer_price = price_info["offer_price"]
        spread_pips = price_info["spread_pips"]
        currency_code = price_info["currency_code"]
        min_distance = price_info["min_distance"]
        logger.info(f"📊 Market data for {epic}: bid={bid_price:.5f}, offer={offer_price:.5f}, spread={spread_pips:.2f} pips, min_distance={min_distance}")

        # Check if spread is too wide (> 2 pips)
        MAX_SPREAD_PIPS = 2.0
        if spread_pips > MAX_SPREAD_PIPS:
            rejection_msg = (
                f"Spread too wide: {spread_pips:.2f} pips exceeds maximum {MAX_SPREAD_PIPS} pips. "
                f"Market conditions unfavorable - waiting for tighter spreads."
            )
            logger.warning(f"🚫 SPREAD REJECTED: {symbol} - {rejection_msg}")

            # Log rejection to alert_history database
            try:
                from services.models import AlertHistory
                spread_rejection = AlertHistory(
                    alert_timestamp=datetime.utcnow(),
                    epic=symbol,
                    pair=epic,
                    signal_type=direction,
                    strategy="order_validation",
                    confidence_score=0.0,
                    price=bid_price,
                    bid_price=bid_price,
                    ask_price=offer_price,
                    spread_pips=spread_pips,
                    timeframe="live",
                    status="rejected_spread",
                    order_status="rejected",
                    claude_reason=rejection_msg,
                    alert_message=f"Trade blocked: Spread {spread_pips:.2f} pips > {MAX_SPREAD_PIPS} pips maximum",
                    alert_level="WARNING"
                )
                db.add(spread_rejection)
                db.commit()
                logger.info(f"✅ Spread rejection logged to alert_history (id={spread_rejection.id})")
            except Exception as log_error:
                logger.error(f"❌ Failed to log spread rejection to database: {str(log_error)}")

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Spread too wide",
                    "message": rejection_msg,
                    "spread_pips": round(spread_pips, 2),
                    "max_spread_pips": MAX_SPREAD_PIPS,
                    "bid": bid_price,
                    "offer": offer_price,
                    "epic": symbol,
                    "alert_id": alert_id,
                    "reason": "spread_too_wide"
                }
            )

        # Check if min_distance is too large (> 4 points)
        if min_distance and min_distance > 4:
            logger.warning(
                f"⚠️ ORDER REJECTED: {symbol} broker min_distance={min_distance}pt exceeds maximum allowed (4pt). "
                f"Market conditions unfavorable - wide spreads or low liquidity detected."
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Broker minimum distance too large",
                    "message": f"Broker requires {min_distance}pt minimum distance, maximum allowed is 4pt. "
                             f"Market conditions not suitable for tight stop placement.",
                    "broker_min_distance": min_distance,
                    "max_allowed_min_distance": 4,
                    "epic": symbol,
                    "alert_id": alert_id,
                    "reason": "min_distance_too_large"
                }
            )

        # Determine SL/TP source: strategy-provided or ATR-calculated
        if body.use_provided_sl_tp and body.stop_distance and body.limit_distance:
            # Use strategy-calculated values
            sl_limit = body.stop_distance
            limit_distance = body.limit_distance
            strategy_requested_sl = sl_limit  # Save original for validation
            logger.info(f"✅ Using strategy-provided SL/TP: {sl_limit}/{limit_distance} for {symbol}")

            # 🛡️ SCALPING PROTECTION: Check if broker's min_distance is too large for tight scalping
            # Reject orders where broker requires 3x or more than strategy wants (indicates poor conditions)
            if min_distance and sl_limit < 10:  # Only for scalping strategies (< 10pt stops)
                if min_distance > (sl_limit * 2.5):  # Broker wants 2.5x+ what strategy wants
                    logger.warning(
                        f"🚫 SCALPING REJECTED: {symbol} broker min_distance={min_distance}pt is too large "
                        f"for scalping strategy ({sl_limit}pt requested). "
                        f"Market conditions unfavorable (wide spreads/low liquidity)."
                    )
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Broker minimum too large for scalping",
                            "message": f"Broker requires {min_distance}pt minimum, strategy wants {sl_limit}pt. "
                                     f"Market conditions not suitable for tight scalping stops.",
                            "broker_min_distance": min_distance,
                            "strategy_requested": sl_limit,
                            "epic": symbol,
                            "alert_id": alert_id,
                            "reason": "scalping_conditions_unfavorable"
                        }
                    )

            # Validate and adjust levels based on broker requirements
            validated_levels = validate_sl_tp_levels(sl_limit, limit_distance, min_distance)
            sl_limit = validated_levels["stopDistance"]
            limit_distance = validated_levels["limitDistance"]

            # Log any adjustments made
            if validated_levels["adjustments"]:
                for adjustment in validated_levels["adjustments"]:
                    logger.info(f"⚙️ {adjustment}")
        else:
            # Calculate ATR-based stop loss and take profit levels (fallback)
            try:
                trade_levels = await calculate_trade_levels(symbol, trading_headers, rr)
                sl_limit = trade_levels["stopDistance"]
                limit_distance = trade_levels["limitDistance"]

                # Validate and adjust levels based on broker requirements
                validated_levels = validate_sl_tp_levels(sl_limit, limit_distance, min_distance)
                sl_limit = validated_levels["stopDistance"]
                limit_distance = validated_levels["limitDistance"]

                # Log any adjustments made
                if validated_levels["adjustments"]:
                    for adjustment in validated_levels["adjustments"]:
                        logger.info(f"⚙️ {adjustment}")

                logger.info(f"📊 {symbol} ATR-based levels: SL={sl_limit}, TP={limit_distance}, RR={rr:.1f} ({trade_levels['calculationMethod']})")

            except Exception as e:
                logger.error(f"❌ Failed to calculate ATR-based levels for {symbol}: {e}")
                # Emergency fallback to conservative levels
                sl_limit = max(25, min_distance + 10) if min_distance else 25
                limit_distance = int(sl_limit * rr)
                logger.info(f"🆘 Using emergency fallback: SL={sl_limit}, TP={limit_distance}")

        # ✅ CRITICAL SAFETY CHECK: Validate stop loss is not suspiciously large
        # This prevents account-blowing bugs like the 100x multiplication error for JPY pairs
        max_allowed_sl = 100  # Maximum 100 points/pips for any pair
        max_allowed_tp = 200  # Maximum 200 points/pips for take profit

        if sl_limit > max_allowed_sl:
            logger.error(
                f"🚨 CRITICAL: Stop loss {sl_limit} exceeds maximum allowed {max_allowed_sl} points! "
                f"Epic: {symbol}, Direction: {direction}, Alert ID: {alert_id}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid stop loss",
                    "message": f"Stop loss {sl_limit} points exceeds maximum allowed {max_allowed_sl} points",
                    "stop_distance": sl_limit,
                    "max_allowed": max_allowed_sl,
                    "epic": symbol,
                    "alert_id": alert_id,
                    "safety_message": "This trade was blocked to protect your account from potential bugs"
                }
            )

        if limit_distance > max_allowed_tp:
            logger.warning(
                f"⚠️ Take profit {limit_distance} exceeds maximum allowed {max_allowed_tp} points! "
                f"Epic: {symbol}, Capping to {max_allowed_tp}"
            )
            limit_distance = max_allowed_tp

        logger.info(f"✅ Safety check passed: SL={sl_limit}/{max_allowed_sl}, TP={limit_distance}/{max_allowed_tp}")

        # =================================================================
        # ORDER TYPE ROUTING: Market vs Limit (Working) Orders
        # =================================================================
        is_limit_order = body.order_type == "limit" and body.entry_level is not None

        if is_limit_order:
            # LIMIT ORDER: Place working order at specified entry level
            logger.info(f"📤 Placing LIMIT order: {symbol} {direction} at entry level {body.entry_level}")
            logger.info(f"   SL: {sl_limit}pts, TP: {limit_distance}pts, Expiry: {body.limit_expiry_minutes} min")

            # Check if there's already a working order for this epic
            try:
                has_existing_order = await has_working_order_for_epic(trading_headers, symbol)
                if has_existing_order:
                    logger.info(f"ℹ️ Working order already exists for {symbol}")
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "status": "skipped",
                            "message": f"Working order already exists for {symbol}",
                            "epic": symbol,
                            "alert_id": alert_id,
                            "reason": "duplicate_working_order"
                        }
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"⚠️ Could not check existing working orders: {e}")

            try:
                # v3.3.0: Pass api_order_type for LIMIT vs STOP order distinction
                # STOP orders: momentum confirmation (entry when price breaks level)
                # LIMIT orders: better price entry (entry when price reaches level from opposite side)
                ig_order_type = body.api_order_type or "STOP"
                logger.info(f"   IG Order Type: {ig_order_type} (STOP=momentum, LIMIT=better price)")

                result = await place_working_order(
                    auth_headers=trading_headers,
                    epic=symbol,
                    direction=direction,
                    level=body.entry_level,
                    stop_distance=sl_limit,
                    limit_distance=limit_distance,
                    expiry_minutes=body.limit_expiry_minutes,
                    currency_code=currency_code,
                    size=body.size or 1.0,
                    order_type=ig_order_type
                )
                # For limit orders, we return early with a different response
                # since the order is pending, not immediately filled
                deal_reference = result.get("dealReference")
                logger.info(f"✅ Limit order placed successfully: {deal_reference}")

                # =================================================================
                # LOG LIMIT ORDER TO DATABASE (for tracking and analytics)
                # =================================================================
                try:
                    # Calculate actual stop/limit prices from entry level
                    actual_stop_price = convert_stop_distance_to_price(
                        body.entry_level, sl_limit, direction, symbol
                    )
                    actual_limit_price = convert_limit_distance_to_price(
                        body.entry_level, limit_distance, direction, symbol
                    )

                    # Calculate expiry time
                    expiry_time = datetime.utcnow() + timedelta(minutes=body.limit_expiry_minutes or 35)

                    # NOTE on column naming (see TradeLog model docstring for details):
                    # - entry_price = stop-entry level (momentum confirmation price)
                    # - limit_price = TAKE PROFIT level (not the limit order entry!)
                    # - sl_price = stop loss level

                    # Scalp trade detection (limit orders)
                    # VSL system disabled - scalp trades now use progressive trailing with scalp configs
                    is_scalp = body.is_scalp_trade or (sl_limit and sl_limit <= 8)

                    # VSL DISABLED (Jan 2026): No longer calculating VSL fields
                    # Scalp trades now use regular trailing system with SCALP_TRAILING_CONFIGS
                    virtual_sl_pips_value = None
                    virtual_sl_price_value = None

                    if is_scalp:
                        logger.info(f"⚡ [LIMIT] Scalp trade: Using scalp trailing configs (VSL disabled)")

                    # Calculate trigger_distance for limit orders (troubleshooting IG rejections)
                    # trigger_distance = gap between entry level and current market price
                    trigger_distance_value = None
                    try:
                        from services.ig_orders import get_point_value
                        point_value = get_point_value(symbol)
                        if point_value > 0:
                            # For BUY: entry is above ask, for SELL: entry is below bid
                            # bid_price is already fetched above from get_current_bid_price()
                            trigger_distance_value = abs(body.entry_level - bid_price) / point_value
                            logger.info(f"📊 [LIMIT] Trigger distance: {trigger_distance_value:.2f} pips "
                                       f"(entry={body.entry_level:.5f}, bid={bid_price:.5f}, min_distance={min_distance})")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not calculate trigger_distance: {e}")

                    trade_log = TradeLog(
                        symbol=symbol,
                        entry_price=body.entry_level,  # Stop-entry price (momentum confirmation)
                        direction=direction.upper(),
                        limit_price=actual_limit_price,  # Take profit price (NOT entry!)
                        sl_price=actual_stop_price,  # Stop loss price
                        deal_reference=deal_reference,
                        endpoint="dev-limit",  # Identify as limit order
                        status="pending_limit",  # New status for limit orders
                        alert_id=alert_id,
                        monitor_until=expiry_time,  # Use monitor_until for expiry tracking
                        # Scalp flag preserved for trailing system
                        is_scalp_trade=is_scalp,
                        # VSL fields deprecated (system disabled)
                        virtual_sl_pips=None,
                        virtual_sl_price=None,
                        # Troubleshooting data for IG rejections
                        min_stop_distance_points=min_distance,
                        trigger_distance=trigger_distance_value,
                    )

                    db.add(trade_log)
                    db.commit()
                    logger.info(f"✅ Limit order logged to database: {symbol} @ {body.entry_level}")
                    if is_scalp:
                        logger.info(f"   ⚡ Scalp trade: is_scalp_trade=True (using scalp trailing configs)")
                    logger.info(f"   Deal Ref: {deal_reference}, Expiry: {expiry_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    if alert_id:
                        logger.info(f"   Linked to alert_id: {alert_id}")

                except SQLAlchemyError as db_error:
                    db.rollback()
                    logger.error(f"❌ Failed to log limit order to DB: {str(db_error)}")
                    # Don't fail the request - order was placed successfully
                finally:
                    db.close()

                return {
                    "status": "pending",
                    "order_type": "limit",
                    "dealReference": deal_reference,
                    "entry_level": body.entry_level,
                    "stop_distance": sl_limit,
                    "limit_distance": limit_distance,
                    "expiry_minutes": body.limit_expiry_minutes,
                    "epic": symbol,
                    "direction": direction,
                    "alert_id": alert_id,
                    "message": f"Limit order placed at {body.entry_level}, expires in {body.limit_expiry_minutes} minutes"
                }

            except httpx.HTTPStatusError as broker_error:
                error_text = broker_error.response.text if hasattr(broker_error.response, 'text') else str(broker_error)

                # Calculate trigger_distance for rejection logging
                trigger_dist_log = None
                try:
                    from services.ig_orders import get_point_value
                    point_value = get_point_value(symbol)
                    if point_value > 0:
                        trigger_dist_log = abs(body.entry_level - bid_price) / point_value
                except:
                    pass

                # Enhanced logging for IG rejection troubleshooting
                logger.error(f"❌ Limit order REJECTED: {symbol} {direction}")
                logger.error(f"   Entry Level: {body.entry_level:.5f}, Current Bid: {bid_price:.5f}")
                logger.error(f"   Trigger Distance: {trigger_dist_log:.2f} pips" if trigger_dist_log else "   Trigger Distance: N/A")
                logger.error(f"   Min Stop Distance: {min_distance} points")
                logger.error(f"   SL: {sl_limit}pts, TP: {limit_distance}pts")
                logger.error(f"   Error: {error_text}")

                raise HTTPException(
                    status_code=broker_error.response.status_code if hasattr(broker_error, 'response') else 500,
                    detail={
                        "error": "Limit order placement failed",
                        "message": error_text[:500],
                        "epic": symbol,
                        "alert_id": alert_id,
                        # Include diagnostic data in rejection response
                        "entry_level": body.entry_level,
                        "current_bid": bid_price,
                        "trigger_distance_pips": trigger_dist_log,
                        "min_stop_distance_points": min_distance,
                        "sl_distance": sl_limit,
                        "tp_distance": limit_distance,
                    }
                )

        # =================================================================
        # MARKET ORDER: Original logic (unchanged)
        # =================================================================
        logger.info(f"📤 Placing MARKET order: {symbol} {direction} with SL: {sl_limit}, TP: {limit_distance}")

        # FIXED: Catch broker duplicate position rejection and convert to HTTP 409
        # 🛡️ SCALPING FAILSAFE: If broker rejects tight SL/TP, retry with 10pt minimum
        try:
            result = await place_market_order(trading_headers, symbol, direction, currency_code, sl_limit, limit_distance)
        except httpx.HTTPStatusError as broker_error:
            # Check if IG rejected due to existing position or SL/TP issue
            error_text = ""
            error_status = broker_error.response.status_code if hasattr(broker_error, 'response') else 500

            try:
                if hasattr(broker_error.response, 'text'):
                    error_text = broker_error.response.text.lower()
                elif hasattr(broker_error.response, 'json'):
                    error_json = broker_error.response.json()
                    error_text = str(error_json).lower()
                else:
                    error_text = str(broker_error).lower()
            except:
                error_text = str(broker_error).lower()

            # 🛡️ SCALPING FAILSAFE: Check if rejection is due to tight SL/TP
            sl_tp_rejection_indicators = [
                'stop', 'limit', 'distance', 'minimum', 'too close', 'invalid level',
                'working order', 'stop level', 'limit level'
            ]
            is_sl_tp_issue = any(indicator in error_text for indicator in sl_tp_rejection_indicators)
            failsafe_succeeded = False

            # If it's an SL/TP rejection and we're using scalping values (< 10), retry with 10pt
            if is_sl_tp_issue and (sl_limit < 10 or limit_distance < 10):
                original_sl = sl_limit
                original_tp = limit_distance

                # Apply failsafe: use 10pt minimum
                sl_limit = max(10, sl_limit)
                limit_distance = max(10, limit_distance)

                # Maintain risk/reward ratio if both were adjusted
                if original_sl < 10 and original_tp < 10:
                    limit_distance = max(10, int(sl_limit * 1.5))

                logger.warning(f"⚠️ Broker rejected tight SL/TP: {error_text[:200]}")
                logger.warning(f"🛡️ SCALPING FAILSAFE ACTIVATED: Retrying with SL {original_sl}pt→{sl_limit}pt, TP {original_tp}pt→{limit_distance}pt")

                # Retry the order with adjusted levels
                try:
                    result = await place_market_order(trading_headers, symbol, direction, currency_code, sl_limit, limit_distance)
                    logger.info(f"✅ Order placed successfully with failsafe levels: SL={sl_limit}, TP={limit_distance}")
                    failsafe_succeeded = True
                except Exception as retry_error:
                    logger.error(f"❌ Failsafe retry also failed: {str(retry_error)}")
                    raise

            # Only process duplicate/error handling if failsafe didn't succeed
            if not failsafe_succeeded:
                # Check for duplicate position indicators in error message
                duplicate_indicators = ['already open', 'position exists', 'duplicate', 'existing position']
                is_duplicate = any(indicator in error_text for indicator in duplicate_indicators)

                # Convert HTTP 500 with duplicate indicators OR HTTP 409 to consistent HTTP 409
                if is_duplicate or error_status == 409:
                    logger.info(f"ℹ️ Position already open for {symbol} (HTTP {error_status})")
                    logger.debug(f"   Broker response: {error_text[:200]}")
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "status": "skipped",
                            "message": f"Position already open for {symbol}",
                            "epic": symbol,
                            "alert_id": alert_id,
                            "reason": "duplicate_position"
                        }
                    )
                else:
                    # Re-raise other broker errors (will be caught by outer exception handler)
                    logger.error(f"❌ Broker rejected order for {symbol}: HTTP {error_status}")
                    logger.error(f"   Error details: {error_text[:500]}")
                    raise
        
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
        logger.info(f"✅ Order confirmed: Deal ID {deal_id}, Entry Price (raw): {entry_price}")

        # CEEM epics return scaled prices from IG API (11633.5 instead of 1.16335)
        # Normalize to standard forex format for consistent database storage
        if "CEEM" in symbol and entry_price > 1000:
            entry_price_normalized = entry_price / 10000.0
            logger.info(f"📐 CEEM price normalized: {entry_price} -> {entry_price_normalized}")
            entry_price = entry_price_normalized

        logger.info(f"✅ Order confirmed: Deal ID {deal_id}, Entry Price: {entry_price}")

        # Convert to actual prices for database
        # FIXED: Use normalized entry_price for all pairs (CEEM prices are already normalized)
        actual_stop_price = convert_stop_distance_to_price(entry_price, sl_limit, direction, symbol)
        actual_limit_price = convert_limit_distance_to_price(entry_price, limit_distance, direction, symbol)
        
        logger.info(f"🔍 DEBUG: alert_id parameter value: {alert_id} (type: {type(alert_id)})")
        logger.info(f"🔍 DEBUG: alert_id is None: {alert_id is None}")
        logger.info(f"🔍 DEBUG: alert_id bool value: {bool(alert_id)}")

        # Save to database
        try:
            # Check if this is a scalp trade (explicitly set or detected from tight SL)
            # VSL system disabled - scalp trades now use progressive trailing with scalp-specific configs
            is_scalp = body.is_scalp_trade or (sl_limit and sl_limit <= 8)  # <=8 pips = scalp trade

            # VSL DISABLED (Jan 2026): No longer calculating VSL fields
            # Scalp trades now use regular trailing system with SCALP_TRAILING_CONFIGS
            # These configs provide optimal 12-20 pip stops based on analysis
            virtual_sl_pips_value = None
            virtual_sl_price_value = None

            if is_scalp:
                logger.info(f"⚡ Scalp trade detected: Using scalp trailing configs (VSL disabled)")

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
                alert_id=alert_id,
                # Scalp flag preserved for trailing system to use scalp configs
                is_scalp_trade=is_scalp,
                # VSL fields deprecated (system disabled)
                virtual_sl_pips=None,
                virtual_sl_price=None,
            )

            db.add(trade_log)
            logger.info(f"🔍 DEBUG: TradeLog added to session, alert_id: {trade_log.alert_id}")
            db.commit()
            db.refresh(trade_log)  # Get the ID after commit
            logger.info(f"✅ Trade logged: {symbol} {entry_price} {direction}")
            if alert_id is not None:
                logger.info(f"✅ Trade linked to alert_id: {alert_id}")

            # VSL service registration disabled - using regular trailing system instead
            # if is_scalp:
            #     # VSL service no longer used - regular trailing handles scalp trades
            #     pass
                
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
async def modify_stop_price(
    request: Request,
    trading_headers: dict = Depends(get_ig_auth_headers),
    db: Session = Depends(get_db)
):
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


# =============================================================================
# WORKING ORDERS (LIMIT ORDERS) ENDPOINTS
# =============================================================================

@router.get("/working-orders")
async def list_working_orders(
    trading_headers: dict = Depends(get_ig_auth_headers)
):
    """
    List all pending working orders (limit orders).

    Returns a list of all limit orders that have been placed but not yet filled.
    """
    try:
        result = await get_working_orders(trading_headers)
        orders = result.get("workingOrders", [])

        logger.info(f"📋 [WORKING ORDERS] Retrieved {len(orders)} pending orders")

        return {
            "status": "success",
            "count": len(orders),
            "workingOrders": orders
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"❌ [WORKING ORDERS] Failed to fetch: {e}")
        raise HTTPException(
            status_code=e.response.status_code if hasattr(e, 'response') else 500,
            detail=f"Failed to fetch working orders: {str(e)}"
        )
    except Exception as e:
        logger.exception("Unhandled exception in /working-orders")
        raise HTTPException(status_code=500, detail=f"Failed to fetch working orders: {str(e)}")


@router.delete("/cancel-working-order/{deal_id}")
async def cancel_working_order_endpoint(
    deal_id: str,
    trading_headers: dict = Depends(get_ig_auth_headers)
):
    """
    Cancel a pending working order (limit order).

    Args:
        deal_id: The deal ID of the working order to cancel

    Returns:
        Confirmation of the cancellation
    """
    try:
        logger.info(f"🗑️ [WORKING ORDER] Cancellation requested for: {deal_id}")

        result = await delete_working_order(trading_headers, deal_id)

        logger.info(f"✅ [WORKING ORDER] Successfully cancelled: {deal_id}")

        return {
            "status": "cancelled",
            "dealId": deal_id,
            "dealReference": result.get("dealReference"),
            "message": f"Working order {deal_id} has been cancelled"
        }

    except httpx.HTTPStatusError as e:
        error_text = e.response.text if hasattr(e.response, 'text') else str(e)
        logger.error(f"❌ [WORKING ORDER] Cancel failed for {deal_id}: {error_text}")

        # Check if order doesn't exist
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Working order not found",
                    "message": f"No working order found with deal ID: {deal_id}",
                    "dealId": deal_id
                }
            )

        raise HTTPException(
            status_code=e.response.status_code if hasattr(e, 'response') else 500,
            detail=f"Failed to cancel working order: {error_text}"
        )
    except Exception as e:
        logger.exception(f"Unhandled exception cancelling working order {deal_id}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel working order: {str(e)}")
