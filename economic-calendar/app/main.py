"""
Economic Calendar Service - Main Application
FastAPI service for scraping and serving economic calendar data
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import config
from database.connection import db_manager
from scraper.scheduler import economic_scheduler
from api.routes import api_router


# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logs directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                config.LOG_FILE,
                maxBytes=config.LOG_MAX_BYTES,
                backupCount=config.LOG_BACKUP_COUNT
            )
        ]
    )

    # Set specific logger levels
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    logger = logging.getLogger(__name__)

    # Startup
    logger.info("🚀 Starting Economic Calendar Service")

    try:
        # Test database connection
        if not db_manager.test_connection():
            logger.error("❌ Database connection failed")
            raise Exception("Database connection failed")

        # Create tables if they don't exist
        db_manager.create_tables()
        logger.info("✅ Database tables verified")

        # Start scheduler
        economic_scheduler.start()
        logger.info("✅ Scheduler started")

        # Log configuration
        logger.info(f"📊 Service configuration:")
        logger.info(f"   - Port: {config.PORT}")
        logger.info(f"   - Debug mode: {config.DEBUG}")
        logger.info(f"   - Focus currencies: {', '.join(config.FOCUS_CURRENCIES)}")
        logger.info(f"   - Weekly scrape: Sundays at {config.WEEKLY_SCRAPE_HOUR:02d}:{config.WEEKLY_SCRAPE_MINUTE:02d} UTC")

        logger.info("🎯 Economic Calendar Service is ready!")

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

    # Shutdown
    logger.info("🛑 Shutting down Economic Calendar Service")

    try:
        # Stop scheduler
        economic_scheduler.stop()
        logger.info("✅ Scheduler stopped")

        # Close database connections
        db_manager.close()
        logger.info("✅ Database connections closed")

    except Exception as e:
        logger.error(f"⚠️ Shutdown error: {e}")

    logger.info("👋 Economic Calendar Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Economic Calendar Service",
    description="Forex Factory economic calendar scraper and API",
    version="1.0.0",
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "economic-calendar",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/health",
            "status": "/api/v1/status",
            "events": "/api/v1/events",
            "upcoming": "/api/v1/events/upcoming",
            "scrape": "/api/v1/scrape/manual",
            "docs": "/docs" if config.DEBUG else "disabled"
        }
    }


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    try:
        # Quick database check
        is_db_healthy = db_manager.test_connection()

        # Check scheduler status
        scheduler_status = economic_scheduler.get_job_status()
        is_scheduler_healthy = scheduler_status.get('status') == 'running'

        is_healthy = is_db_healthy and is_scheduler_healthy

        return JSONResponse(
            content={
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "economic-calendar",
                "checks": {
                    "database": is_db_healthy,
                    "scheduler": is_scheduler_healthy
                }
            },
            status_code=200 if is_healthy else 503
        )

    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            },
            status_code=503
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if config.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

    # Stop scheduler
    try:
        economic_scheduler.stop()
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")

    # Close database connections
    try:
        db_manager.close()
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    sys.exit(0)


def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("🌟 Economic Calendar Service starting...")

    # Run the application
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
        access_log=config.DEBUG,
        reload=config.DEBUG,
        workers=1,  # Keep single worker for scheduler
        loop="asyncio"
    )


if __name__ == "__main__":
    main()