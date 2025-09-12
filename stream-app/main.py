### stream-app/main.py
from fastapi import FastAPI
from routers import stream_router
#from services.stream_manager import stream_prices
#from services.stream_tasks import start_stream_task, stop_all_streams
#from services.stream_controller import start_all_streams, stop_all_streams
from igstream.sync_manager import start_all_streams, stop_all_streams, EPICS
from igstream.auto_backfill import AutoBackfillService
from services.models import IGCandle, Candle
from services.db import Base, engine

import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
import time


# ──────────────────────
# Logging setup
# ──────────────────────
# Set desired timezone via TZ
os.environ['TZ'] = 'Europe/Stockholm'  # Change to your desired timezone
time.tzset()  # Applies it

log_dir = "/app/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "fastapi-stream.log")

rotating_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
formatter.converter = time.localtime  # Use local time (now affected by TZ)
rotating_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(rotating_handler)
logger.addHandler(logging.StreamHandler())

app = FastAPI()

# Global reference to backfill service
backfill_service = None

@app.on_event("startup")
async def startup_event():
    global backfill_service
    
    # Start all streams
    await start_all_streams()
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    print("✅ Tables initialized.")
    
    # Start auto-backfill service - TEMPORARILY DISABLED DUE TO API RATE LIMITS
    # Will re-enable after Monday 2025-09-15 when weekly limit resets
    logger.info("⚠️ Auto-backfill service DISABLED to preserve API rate limits")
    backfill_service = AutoBackfillService(epics=EPICS)
    asyncio.create_task(backfill_service.run_continuous(check_interval_minutes=5))
    logger.info("✅ Auto-backfill service started (checking every 5 minutes)")

@app.on_event("shutdown")
async def shutdown_event():
    global backfill_service
    
    # Stop backfill service
    if backfill_service:
        backfill_service.stop()
        logger.info("✅ Auto-backfill service stopped")
    
    # Stop all streams
    await stop_all_streams()

app.include_router(stream_router.router, prefix="/stream")

# Add endpoint to check backfill status
@app.get("/backfill/status")
async def get_backfill_status():
    """Get the current status of the auto-backfill service"""
    if backfill_service:
        return backfill_service.get_statistics()
    else:
        return {"error": "Backfill service not initialized"}

@app.get("/backfill/gaps")
async def get_current_gaps():
    """Detect and report current gaps without backfilling"""
    from igstream.auto_backfill import detect_and_report_gaps
    
    try:
        report, stats = await detect_and_report_gaps(EPICS)
        return {
            "report": report,
            "statistics": stats
        }
    except Exception as e:
        return {"error": str(e)}