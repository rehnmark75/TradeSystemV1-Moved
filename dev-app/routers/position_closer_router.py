"""
Position Closer Router - Weekend Protection API Endpoints
Provides endpoints for managing automatic position closure on Fridays.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import logging
from datetime import datetime, timezone

from services.position_closer import (
    check_and_close_positions,
    manual_close_positions,
    get_position_closer_status,
    get_position_closure_history
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status", response_model=Dict[str, Any])
async def get_position_closer_status_endpoint():
    """
    Get current status of the position closer service.

    Returns:
        Position closer status including next closure time and statistics
    """
    try:
        status = get_position_closer_status()
        return JSONResponse(content={
            "success": True,
            "data": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Error getting position closer status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-and-close", response_model=Dict[str, Any])
async def check_and_close_positions_endpoint(background_tasks: BackgroundTasks):
    """
    Check if it's Friday 20:30 UTC and close positions if needed.
    This is the main endpoint called by the scheduler.

    Returns:
        Result of position closure attempt
    """
    try:
        logger.info("🔍 Position closure check requested via API")
        result = await check_and_close_positions()

        # Log the result
        if result["action"] == "completed":
            logger.info(f"✅ Position closure completed: {result['positions_closed']} closed, {result['positions_failed']} failed")
        elif result["action"] == "skipped":
            logger.debug(f"ℹ️ Position closure skipped: {result['reason']}")
        else:
            logger.warning(f"⚠️ Position closure result: {result}")

        return JSONResponse(content={
            "success": True,
            "data": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error in position closure check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/manual-close-all", response_model=Dict[str, Any])
async def manual_close_all_positions_endpoint():
    """
    Manually close all open positions immediately.
    This bypasses the Friday 20:30 UTC check and is for emergency/testing use.

    Returns:
        Result of manual position closure
    """
    try:
        logger.warning("⚠️ Manual position closure requested via API - bypassing time checks")
        result = await manual_close_positions()

        if result["action"] == "completed":
            logger.warning(f"⚠️ Manual closure completed: {result['positions_closed']} closed, {result['positions_failed']} failed")

        return JSONResponse(content={
            "success": True,
            "data": result,
            "manual_override": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error in manual position closure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=Dict[str, Any])
async def get_closure_history_endpoint(limit: int = 5):
    """
    Get recent position closure history.

    Args:
        limit: Maximum number of recent closures to return (default 5, max 20)

    Returns:
        List of recent closure events
    """
    try:
        # Limit the maximum to prevent large responses
        limit = min(max(1, limit), 20)

        history = get_position_closure_history(limit)

        return JSONResponse(content={
            "success": True,
            "data": {
                "history": history,
                "count": len(history),
                "limit": limit
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error getting closure history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/next-closure-time", response_model=Dict[str, Any])
async def get_next_closure_time_endpoint():
    """
    Get information about the next scheduled position closure.

    Returns:
        Next closure time and countdown information
    """
    try:
        status = get_position_closer_status()

        return JSONResponse(content={
            "success": True,
            "data": {
                "next_closure_time": status["next_friday"],
                "time_until_closure": status["time_until_next_closure"],
                "closure_schedule": status["closure_schedule"],
                "current_time_utc": status["current_time_utc"]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error getting next closure time: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def position_closer_health_check():
    """
    Health check endpoint for position closer service.

    Returns:
        Health status of the position closer
    """
    try:
        status = get_position_closer_status()

        # Simple health check
        is_healthy = status["enabled"] and isinstance(status["current_time_utc"], str)

        return JSONResponse(content={
            "success": True,
            "healthy": is_healthy,
            "service": "position_closer",
            "enabled": status["enabled"],
            "current_time_utc": status["current_time_utc"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Position closer health check failed: {e}")
        raise HTTPException(status_code=500, detail={
            "success": False,
            "healthy": False,
            "service": "position_closer",
            "error": str(e)
        })