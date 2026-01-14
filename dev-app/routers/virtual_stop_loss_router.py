"""
Virtual Stop Loss REST API Endpoints

Provides endpoints for monitoring and managing the VSL service:
- GET /api/vsl/status - Service status and tracked positions
- POST /api/vsl/refresh - Force position sync from database
- DELETE /api/vsl/position/{trade_id} - Manual position removal
- GET /api/vsl/position/{trade_id} - Get specific position details
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from services.virtual_stop_loss_service import get_vsl_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vsl", tags=["virtual-stop-loss"])


@router.get("/status")
async def get_vsl_status():
    """
    Get Virtual Stop Loss service status.

    Returns:
        Service status including:
        - enabled: Whether VSL is enabled in config
        - running: Whether service is currently running
        - stream_connected: Whether Lightstreamer is connected
        - positions_tracked: Number of positions being monitored
        - positions: Details of each tracked position
        - stats: Service statistics
    """
    service = get_vsl_service()

    if not service:
        return {
            "status": "not_initialized",
            "enabled": False,
            "running": False,
            "message": "VSL service has not been initialized"
        }

    return service.get_status()


@router.post("/refresh")
async def refresh_positions():
    """
    Force refresh of tracked positions from database.

    Useful when positions are opened/closed outside normal flow.

    Returns:
        Refresh status and current position count
    """
    service = get_vsl_service()

    if not service:
        raise HTTPException(
            status_code=503,
            detail="VSL service not available"
        )

    if not service._running:
        raise HTTPException(
            status_code=503,
            detail="VSL service is not running"
        )

    try:
        await service._sync_scalp_positions()
        return {
            "status": "refreshed",
            "positions_tracked": len(service.positions),
            "epics_subscribed": list(service.epic_to_trades.keys())
        }
    except Exception as e:
        logger.error(f"[VSL API] Refresh failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Refresh failed: {str(e)}"
        )


@router.get("/position/{trade_id}")
async def get_position(trade_id: int):
    """
    Get details for a specific tracked position.

    Args:
        trade_id: Trade ID to lookup

    Returns:
        Position details including entry, VSL level, last price
    """
    service = get_vsl_service()

    if not service:
        raise HTTPException(
            status_code=503,
            detail="VSL service not available"
        )

    position = service.get_position(trade_id)

    if not position:
        raise HTTPException(
            status_code=404,
            detail=f"Position {trade_id} not found in VSL tracking"
        )

    return position


@router.delete("/position/{trade_id}")
async def remove_position(trade_id: int):
    """
    Manually remove a position from VSL tracking.

    Use this if you want to stop monitoring a specific position
    without closing it.

    Args:
        trade_id: Trade ID to remove from tracking

    Returns:
        Removal confirmation
    """
    service = get_vsl_service()

    if not service:
        raise HTTPException(
            status_code=503,
            detail="VSL service not available"
        )

    # Check if position exists
    if trade_id not in service.positions:
        raise HTTPException(
            status_code=404,
            detail=f"Position {trade_id} not found in VSL tracking"
        )

    service.remove_position(trade_id)

    return {
        "status": "removed",
        "trade_id": trade_id,
        "message": f"Position {trade_id} removed from VSL tracking"
    }


@router.post("/position/{trade_id}")
async def add_position_manual(trade_id: int):
    """
    Manually add a position to VSL tracking.

    Use this to start monitoring a specific scalp trade.

    Args:
        trade_id: Trade ID to add to tracking

    Returns:
        Addition confirmation
    """
    from services.db import SessionLocal
    from services.models import TradeLog

    service = get_vsl_service()

    if not service:
        raise HTTPException(
            status_code=503,
            detail="VSL service not available"
        )

    # Check if already tracked
    if trade_id in service.positions:
        return {
            "status": "already_tracked",
            "trade_id": trade_id,
            "message": f"Position {trade_id} is already being tracked"
        }

    # Get trade from database
    try:
        with SessionLocal() as db:
            trade = db.query(TradeLog).filter(TradeLog.id == trade_id).first()

            if not trade:
                raise HTTPException(
                    status_code=404,
                    detail=f"Trade {trade_id} not found in database"
                )

            if trade.status == "closed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Trade {trade_id} is already closed"
                )

            # Mark as scalp trade if not already
            if not trade.is_scalp_trade:
                trade.is_scalp_trade = True
                db.commit()

            # Add to VSL tracking
            success = service.add_scalp_position(trade)

            if success:
                return {
                    "status": "added",
                    "trade_id": trade_id,
                    "message": f"Position {trade_id} added to VSL tracking"
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to add position {trade_id} to VSL tracking"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VSL API] Error adding position {trade_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error adding position: {str(e)}"
        )


@router.get("/stats")
async def get_stats():
    """
    Get VSL service statistics.

    Returns:
        Statistics including trigger counts, processed updates, etc.
    """
    service = get_vsl_service()

    if not service:
        return {
            "status": "not_initialized",
            "stats": {}
        }

    return {
        "status": "ok" if service._running else "stopped",
        "stats": service._stats,
        "positions_count": len(service.positions),
        "epics_count": len(service.epic_to_trades),
    }


@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
        Health status with key indicators
    """
    service = get_vsl_service()

    if not service:
        return {
            "healthy": False,
            "reason": "Service not initialized"
        }

    stream_status = service.stream_manager.get_status() if service.stream_manager else {}

    return {
        "healthy": service._running and stream_status.get("connected", False),
        "running": service._running,
        "stream_connected": stream_status.get("connected", False),
        "positions_tracked": len(service.positions),
        "vsl_triggers": service._stats.get("vsl_triggered_count", 0),
        "price_updates": service._stats.get("price_updates_processed", 0),
    }
