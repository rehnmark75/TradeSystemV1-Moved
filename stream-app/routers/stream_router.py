### stream-app/routers/stream_router.py
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Query, Request
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session
from services.models import Candle
from services.db import get_db
from services.old_services.stream_manager import track_trade_subscription, tracked_trades_by_epic

from services.old_services.stream_manager import stream_prices, stream_status, active_stream, task_registry

# Import new services for real-time data
from services.alert_manager import get_alert_manager
from services.log_parser import get_log_parser
from igstream.sync_manager import get_stream_health_report

router = APIRouter()

class StreamRequest(BaseModel):
    epic: str
    headers: dict
    entry_price: float
    direction: str
    sl_price: float
    tp_price: float
    deal_id: str

@router.post("/start-stream")
async def start_stream(req: StreamRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            stream_prices,
            epic=req.epic,
            headers=req.headers,
            entry_price=req.entry_price,
            direction=req.direction,
            sl_price=req.sl_price,
            tp_price=req.tp_price,
            deal_id=req.deal_id
        )
        return {"status": "stream started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/track-trade")
def list_active_trades():
    return {
        "active_trades": [
            {
                "deal_id": deal_id,
                "epic": info["epic"],
                "entry_price": info["entry_price"],
                "direction": info["direction"]
            }
            for deal_id, info in tracked_trades_by_epic.items()
        ]
    }

@router.get("/status")
async def get_stream_status():
    return {
        "running": stream_status["running"],
        "epic": stream_status["epic"],
        "tick_count": stream_status["tick_count"]
    }

@router.post("/stop-stream")
async def stop_stream():
    active_stream["client"].unsubscribe(active_stream["subscription"])
    active_stream["client"].disconnect()
    return {"status": "triggered shutdown"}


@router.get("/candles/{epic}")
def get_recent_candles(epic: str, timeframe: int = 5, db: Session = Depends(get_db)):
    candles = (
        db.query(Candle)
        .filter(Candle.epic == epic, Candle.timeframe == timeframe)
        .order_by(Candle.start_time.desc())
        .limit(15)
        .all()
    )
    return list(reversed(candles))  # so it goes from oldest to newest

@router.get("/candle/latest/{epic}")
def get_latest_candle(epic: str, timeframe: int = Query(5), db: Session = Depends(get_db)):
    candle = (
        db.query(Candle)
        .filter(Candle.epic == epic, Candle.timeframe == timeframe)
        .order_by(Candle.start_time.desc())
        .first()
    )

    if not candle:
        raise HTTPException(status_code=404, detail=f"No {timeframe}m candle found for {epic}")

    return {
        "epic": candle.epic,
        "timeframe": candle.timeframe,
        "start_time": candle.start_time,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume
    }

@router.post("/track-trade/{deal_id}")
async def track_trade(deal_id: str, request: Request):
    """
    Trigger tracking of a live trade. Expects JSON with:
    - epic
    - entry_price
    - direction
    - headers (CST, XST, API key)
    - trailing_distance (optional, default 15)
    """
    body = await request.json()

    epic = body.get("epic")
    entry_price = body.get("entry_price")
    direction = body.get("direction")
    headers = body.get("headers")
    trailing_distance = body.get("trailing_distance", 15)

    if not all([epic, entry_price, direction, headers]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        track_trade_subscription(deal_id, epic, float(entry_price), direction, headers, trailing_distance=int(trailing_distance))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tracking failed: {str(e)}")

    return {"status": "tracking", "deal_id": deal_id, "trailing_distance": trailing_distance}

@router.get("/track-trade")
def list_active_trades():
    return {
        "tracked_epics": list(tracked_trades_by_epic.keys()),
        "active_trades": {
            epic: list(trades.keys())
            for epic, trades in tracked_trades_by_epic.items()
        }
    }

@router.delete("/track-trade/{deal_id}")
def stop_tracking_trade(deal_id: str):
    client = active_stream.get("client")
    if not client:
        return {"error": "No active Lightstreamer client"}

    if deal_id not in active_trades:
        return {"message": f"Deal {deal_id} is not being tracked."}

    client.unsubscribe(active_trades[deal_id])
    del active_trades[deal_id]
    return {"message": f"Unsubscribed from trade {deal_id}"}

# ========================================
# NEW ENDPOINTS FOR SYSTEM STATUS PAGE
# ========================================

@router.get("/health")
async def get_comprehensive_health():
    """Get comprehensive system health report"""
    try:
        # Get stream health from sync manager
        health_report = await get_stream_health_report()
        
        # Get alert summary
        alert_manager = get_alert_manager()
        alert_summary = alert_manager.get_alert_summary()
        
        return {
            "status": "healthy" if alert_summary["error_count"] == 0 else "issues",
            "timestamp": datetime.now().isoformat(),
            "stream_health": health_report,
            "alert_summary": alert_summary,
            "market_open": health_report.get("market_open", True) if isinstance(health_report, dict) else True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/alerts/recent")
async def get_recent_alerts(hours_back: int = Query(6, ge=1, le=24)):
    """Get recent system alerts"""
    try:
        alert_manager = get_alert_manager()
        alerts = alert_manager.get_recent_alerts(hours_back=hours_back)
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "timestamp": datetime.now().isoformat(),
            "hours_back": hours_back
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "alerts": [],
            "total_count": 0
        }

@router.get("/operations/recent")
async def get_recent_operations(hours_back: int = Query(6, ge=1, le=24)):
    """Get recent system operations from both logs and operation tracker"""
    try:
        # Get operations from the operation tracker (real-time tracking)
        from services.operation_tracker import get_operation_tracker
        operation_tracker = get_operation_tracker()
        tracked_operations = operation_tracker.get_recent_operations(hours_back=hours_back, max_count=30)
        
        # Also get operations from logs as fallback/supplement
        alert_manager = get_alert_manager()
        log_operations = alert_manager.get_operations_from_logs(hours_back=hours_back)
        
        # Combine operations (prefer tracked operations but include log operations for completeness)
        all_operations = tracked_operations + log_operations[-20:]  # Add last 20 log operations
        
        # Sort by timestamp and remove duplicates
        unique_operations = {}
        for op in all_operations:
            key = f"{op.get('time', '')}-{op.get('epic', '')}-{op.get('action', '')}"
            if key not in unique_operations:
                unique_operations[key] = op
        
        final_operations = list(unique_operations.values())
        
        # Sort by time (newest first)
        try:
            final_operations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        except:
            final_operations.sort(key=lambda x: x.get('time', ''), reverse=True)
        
        return {
            "operations": final_operations[:50],  # Limit to 50 most recent
            "total_count": len(final_operations),
            "timestamp": datetime.now().isoformat(),
            "hours_back": hours_back,
            "sources": ["operation_tracker", "logs"]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "operations": [],
            "total_count": 0
        }

@router.get("/logs/recent")
async def get_recent_logs(
    hours_back: int = Query(2, ge=1, le=12),
    max_entries: int = Query(100, ge=10, le=500)
):
    """Get recent log entries"""
    try:
        log_parser = get_log_parser()
        logs = log_parser.get_recent_logs(hours_back=hours_back, max_entries=max_entries)
        
        # Convert datetime objects to strings for JSON serialization
        serialized_logs = []
        for log in logs:
            log_dict = log.copy()
            log_dict["timestamp"] = log_dict["timestamp"].isoformat()
            serialized_logs.append(log_dict)
        
        return {
            "logs": serialized_logs,
            "total_count": len(logs),
            "timestamp": datetime.now().isoformat(),
            "hours_back": hours_back
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "logs": [],
            "total_count": 0
        }

@router.get("/system/summary")
async def get_system_summary():
    """Get comprehensive system summary"""
    try:
        log_parser = get_log_parser()
        health_summary = log_parser.get_system_health_summary(hours_back=2)
        
        alert_manager = get_alert_manager()
        alert_summary = alert_manager.get_alert_summary()
        
        # Try to get stream health
        try:
            stream_health = await get_stream_health_report()
            market_open = stream_health.get("market_open", True) if isinstance(stream_health, dict) else True
            total_streams = len(stream_health.get("streams", {})) if isinstance(stream_health, dict) else 0
        except:
            market_open = True
            total_streams = 0
            stream_health = {"status": "unavailable"}
        
        return {
            "status": "healthy" if health_summary["error_count"] == 0 and alert_summary["error_count"] == 0 else "issues",
            "timestamp": datetime.now().isoformat(),
            "market_open": market_open,
            "total_streams": total_streams,
            "recent_activity": {
                "total_log_entries": health_summary["total_entries"],
                "errors_last_hour": health_summary["error_count"],
                "warnings_last_hour": health_summary["warning_count"],
                "alerts_last_4hours": alert_summary["total_alerts"]
            },
            "health_indicators": {
                "stream_health": health_summary.get("stream_health", "unknown"),
                "gap_status": health_summary.get("gap_status", "unknown"),
                "last_error": health_summary.get("last_error"),
                "last_warning": health_summary.get("last_warning")
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
