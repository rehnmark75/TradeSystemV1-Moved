"""
System Monitor - Main application entry point.
Monitors all Docker containers and sends notifications for issues.
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import settings
from .services.docker_monitor import DockerMonitor
from .services.health_checker import HealthChecker
from .services.metrics_collector import MetricsCollector
from .services.alert_manager import AlertManager
from .notifications.telegram_notifier import TelegramNotifier
from .notifications.email_notifier import EmailNotifier
from .api import routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/system_monitor.log"),
    ],
)
logger = logging.getLogger(__name__)

# Global instances
docker_monitor: DockerMonitor = None
health_checker: HealthChecker = None
metrics_collector: MetricsCollector = None
alert_manager: AlertManager = None
telegram_notifier: TelegramNotifier = None
email_notifier: EmailNotifier = None
scheduler: AsyncIOScheduler = None


async def monitoring_loop():
    """Main monitoring loop - runs periodically to check all containers."""
    global docker_monitor, health_checker, metrics_collector, alert_manager

    try:
        logger.debug("Running monitoring check...")

        # Get system health from Docker
        system_health = docker_monitor.get_system_health()

        # Run health checks on services
        health_results = await health_checker.check_all_services()

        # Update container health with service health check results
        for container in system_health.containers:
            if container.name in health_results:
                result = health_results[container.name]
                container.health_check_response_time_ms = result["response_time_ms"]
                container.consecutive_failures = result["state"].get("consecutive_failures", 0)
                container.last_health_check = result["state"].get("last_check")

                # Override health status if health check is failing
                if health_checker.is_failing(container.name):
                    from .models import HealthStatus
                    container.health_status = HealthStatus.UNHEALTHY

        # Store metrics
        if metrics_collector:
            metrics_collector.store_system_metrics(system_health)

        # Process alerts
        if alert_manager:
            alerts = await alert_manager.process_system_health(system_health)
            if alerts:
                logger.info(f"Generated {len(alerts)} alert(s)")

        logger.debug(f"Monitoring check complete. Health score: {system_health.health_score:.1f}%")

    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def cleanup_task():
    """Periodic cleanup of old metrics data."""
    global metrics_collector

    try:
        if metrics_collector:
            metrics_collector.cleanup_old_metrics()
            logger.info("Metrics cleanup completed")
    except Exception as e:
        logger.error(f"Error in cleanup task: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global docker_monitor, health_checker, metrics_collector, alert_manager
    global telegram_notifier, email_notifier, scheduler

    logger.info("=" * 50)
    logger.info("Starting System Monitor...")
    logger.info("=" * 50)

    # Initialize services
    try:
        # Docker monitor
        docker_monitor = DockerMonitor()
        if docker_monitor.is_connected():
            logger.info("✅ Docker monitor initialized")
        else:
            logger.error("❌ Failed to connect to Docker daemon")

        # Health checker
        health_checker = HealthChecker()
        logger.info("✅ Health checker initialized")

        # Metrics collector
        metrics_collector = MetricsCollector()
        metrics_collector.initialize_tables()
        logger.info("✅ Metrics collector initialized")

        # Notification services
        telegram_notifier = TelegramNotifier()
        if telegram_notifier.is_enabled():
            logger.info("✅ Telegram notifier initialized")
        else:
            logger.warning("⚠️ Telegram notifications disabled (not configured)")

        email_notifier = EmailNotifier()
        if email_notifier.is_enabled():
            logger.info("✅ Email notifier initialized")
        else:
            logger.warning("⚠️ Email notifications disabled (not configured)")

        # Alert manager
        alert_manager = AlertManager(
            metrics_collector=metrics_collector,
            telegram_notifier=telegram_notifier,
            email_notifier=email_notifier,
        )
        logger.info("✅ Alert manager initialized")

        # Inject dependencies into routes
        routes.docker_monitor = docker_monitor
        routes.health_checker = health_checker
        routes.metrics_collector = metrics_collector
        routes.alert_manager = alert_manager
        routes.telegram_notifier = telegram_notifier
        routes.email_notifier = email_notifier

        # Start scheduler
        scheduler = AsyncIOScheduler()

        # Main monitoring loop
        scheduler.add_job(
            monitoring_loop,
            trigger=IntervalTrigger(seconds=settings.monitor_interval),
            id="monitoring_loop",
            name="Container Monitoring",
            replace_existing=True,
        )

        # Daily cleanup task (run at 3 AM)
        scheduler.add_job(
            cleanup_task,
            trigger="cron",
            hour=3,
            minute=0,
            id="cleanup_task",
            name="Metrics Cleanup",
            replace_existing=True,
        )

        scheduler.start()
        logger.info(f"✅ Scheduler started (interval: {settings.monitor_interval}s)")

        # Run initial check
        await monitoring_loop()

        # Send startup notification
        if telegram_notifier and telegram_notifier.is_enabled():
            system_health = docker_monitor.get_system_health()
            await telegram_notifier.send_message(
                f"🚀 *System Monitor Started*\n\n"
                f"Monitoring {system_health.total_containers} containers\n"
                f"Health Score: {system_health.health_score:.0f}%\n"
                f"Status: {system_health.get_status_emoji()} {system_health.get_overall_status().upper()}"
            )

        logger.info("=" * 50)
        logger.info("System Monitor ready!")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Failed to initialize System Monitor: {e}")
        import traceback
        logger.error(traceback.format_exc())

    yield

    # Shutdown
    logger.info("Shutting down System Monitor...")

    if scheduler:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    if health_checker:
        await health_checker.close()
        logger.info("Health checker closed")

    # Send shutdown notification
    if telegram_notifier and telegram_notifier.is_enabled():
        await telegram_notifier.send_message(
            "🛑 *System Monitor Stopped*\n\n"
            "The monitoring service has been shut down."
        )

    logger.info("System Monitor shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="TradeSystemV1 System Monitor",
    description="Monitors Docker containers and sends notifications for system issues",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router)


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "TradeSystemV1 System Monitor",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "health": "/health",
            "status": "/api/v1/status",
            "containers": "/api/v1/containers",
            "metrics": "/api/v1/metrics",
            "alerts": "/api/v1/alerts",
            "docs": "/docs",
        },
    }
