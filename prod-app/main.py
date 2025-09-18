from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from routers.orders_router import router as orders_router
from routers.position_closer_router import router as position_closer_router
from contextlib import asynccontextmanager
import logging
import os
import asyncio
from datetime import datetime, timezone
import httpx

# Try to import APScheduler, create simple scheduler if not available
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logging.warning("⚠️ APScheduler not available - position closer scheduling disabled")

os.makedirs("/app/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/uvicorn-prod.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = None


async def call_position_closer_endpoint():
    """
    Call the position closer endpoint via internal HTTP request.
    This allows us to use the same validation logic as manual calls.
    """
    try:
        logger.info("⏰ Scheduled position closure check triggered")

        # Make internal HTTP request to our own endpoint
        async with httpx.AsyncClient() as client:
            # Use localhost since we're calling our own API
            response = await client.post(
                "http://localhost:8000/position-closer/check-and-close",
                headers={"x-apim-gateway": "verified"},  # Add required gateway header
                timeout=60.0
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Scheduled position closure completed: {result}")
            else:
                logger.error(f"❌ Scheduled position closure failed: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"❌ Error in scheduled position closure: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifespan of the FastAPI app, including scheduler startup/shutdown.
    """
    global scheduler

    # Startup
    logger.info("🚀 Starting FastAPI application with position closer")

    if SCHEDULER_AVAILABLE:
        try:
            scheduler = AsyncIOScheduler()

            # Schedule position closer for Friday 20:30 UTC
            scheduler.add_job(
                call_position_closer_endpoint,
                CronTrigger(
                    day_of_week=4,  # Friday (0 = Monday)
                    hour=20,        # 20:00 UTC
                    minute=30,      # 30 minutes
                    timezone='UTC'
                ),
                id='friday_position_closer',
                name='Friday 20:30 UTC Position Closer',
                replace_existing=True
            )

            # Add a test job that runs every minute (can be disabled in production)
            # scheduler.add_job(
            #     call_position_closer_endpoint,
            #     CronTrigger(minute='*'),
            #     id='test_position_closer',
            #     name='Test Position Closer (Every Minute)',
            #     replace_existing=True
            # )

            scheduler.start()
            logger.info("✅ Position closer scheduler started - Friday 20:30 UTC")
            logger.info(f"   Next scheduled run: {scheduler.get_job('friday_position_closer').next_run_time}")

        except Exception as e:
            logger.error(f"❌ Failed to start position closer scheduler: {e}")
            scheduler = None
    else:
        logger.warning("⚠️ Scheduler not available - position closer must be triggered manually")

    yield

    # Shutdown
    logger.info("🛑 Shutting down FastAPI application")
    if scheduler:
        try:
            scheduler.shutdown()
            logger.info("✅ Position closer scheduler stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping scheduler: {e}")


app = FastAPI(lifespan=lifespan)

# Global middleware (if needed)
@app.middleware("http")
async def require_verified_gateway(request, call_next):
    if request.headers.get("x-apim-gateway") != "verified":
        return JSONResponse(status_code=403, content={"detail": "Access denied"})
    return await call_next(request)

# Optional root block route
@app.get("/")
@app.post("/")
def block_root():
    raise HTTPException(status_code=403, detail="****")

@app.get("/favicon.ico")
async def ignore_favicon():
    return Response(status_code=204)

@app.get("/validate")
def validate():
    return PlainTextResponse("API is working")

# Register routers
app.include_router(orders_router, prefix="/orders", tags=["orders"])
app.include_router(position_closer_router, prefix="/position-closer", tags=["position-closer"])