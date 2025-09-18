from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.pool import ThreadPoolExecutor
import logging
import asyncio
from datetime import datetime
from typing import Optional

from scraper.forex_factory import ForexFactoryScraper
from config import config

logger = logging.getLogger(__name__)


class EconomicCalendarScheduler:
    """Scheduler for economic calendar scraping tasks"""

    def __init__(self):
        self.scheduler = None
        self.scraper = ForexFactoryScraper()
        self.setup_scheduler()

    def setup_scheduler(self):
        """Initialize the scheduler with optimized settings"""
        executors = {
            'default': ThreadPoolExecutor(max_workers=3),
        }

        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': 300  # 5 minutes
        }

        self.scheduler = AsyncIOScheduler(
            executors=executors,
            job_defaults=job_defaults,
            timezone=config.SCHEDULER_TIMEZONE
        )

        logger.info("Scheduler initialized")

    def start(self):
        """Start the scheduler and add jobs"""
        if not self.scheduler:
            self.setup_scheduler()

        # Add weekly scraping job
        self.add_weekly_scrape_job()

        # Add daily cleanup job
        self.add_daily_cleanup_job()

        # Add health check job
        self.add_health_check_job()

        self.scheduler.start()
        logger.info("Scheduler started with jobs")

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler stopped")

    def add_weekly_scrape_job(self):
        """Add weekly scraping job (Sundays at 23:00 UTC)"""
        trigger = CronTrigger(
            day_of_week=config.WEEKLY_SCRAPE_DAY,  # Sunday
            hour=config.WEEKLY_SCRAPE_HOUR,       # 23:00
            minute=config.WEEKLY_SCRAPE_MINUTE,   # 0
            timezone=config.SCHEDULER_TIMEZONE
        )

        self.scheduler.add_job(
            func=self.weekly_scrape_task,
            trigger=trigger,
            id='weekly_scrape',
            name='Weekly Economic Calendar Scrape',
            replace_existing=True
        )

        logger.info("Weekly scrape job scheduled for Sundays at 23:00 UTC")

    def add_daily_cleanup_job(self):
        """Add daily cleanup job (every day at 02:00 UTC)"""
        trigger = CronTrigger(
            hour=2,
            minute=0,
            timezone=config.SCHEDULER_TIMEZONE
        )

        self.scheduler.add_job(
            func=self.daily_cleanup_task,
            trigger=trigger,
            id='daily_cleanup',
            name='Daily Data Cleanup',
            replace_existing=True
        )

        logger.info("Daily cleanup job scheduled for 02:00 UTC")

    def add_health_check_job(self):
        """Add periodic health check job (every 30 minutes)"""
        trigger = CronTrigger(
            minute='*/30',
            timezone=config.SCHEDULER_TIMEZONE
        )

        self.scheduler.add_job(
            func=self.health_check_task,
            trigger=trigger,
            id='health_check',
            name='Periodic Health Check',
            replace_existing=True
        )

        logger.info("Health check job scheduled every 30 minutes")

    async def weekly_scrape_task(self):
        """Main weekly scraping task"""
        logger.info("Starting weekly economic calendar scrape")

        try:
            # Scrape current week
            current_week_events, current_week_log = self.scraper.scrape_week(week_offset=0)
            self.scraper.save_events_to_database(current_week_events, current_week_log)

            # Scrape next week
            next_week_events, next_week_log = self.scraper.scrape_week(week_offset=1)
            self.scraper.save_events_to_database(next_week_events, next_week_log)

            total_events = len(current_week_events) + len(next_week_events)
            logger.info(f"Weekly scrape completed successfully: {total_events} events")

            # Send notification if configured
            await self.send_scrape_notification(
                success=True,
                total_events=total_events,
                message=f"Weekly scrape completed: {total_events} events"
            )

        except Exception as e:
            logger.error(f"Weekly scrape failed: {e}")

            # Send failure notification
            await self.send_scrape_notification(
                success=False,
                total_events=0,
                message=f"Weekly scrape failed: {str(e)}"
            )

    async def daily_cleanup_task(self):
        """Daily cleanup task"""
        logger.info("Starting daily cleanup task")

        try:
            self.scraper.cleanup_old_data()
            logger.info("Daily cleanup completed successfully")

        except Exception as e:
            logger.error(f"Daily cleanup failed: {e}")

    async def health_check_task(self):
        """Periodic health check task"""
        try:
            # Check database connection
            from database.connection import db_manager
            is_healthy = db_manager.test_connection()

            if not is_healthy:
                logger.warning("Health check failed: Database connection issues")
            else:
                logger.debug("Health check passed")

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    async def manual_scrape(self, week_offset: int = 0) -> dict:
        """
        Trigger manual scrape

        Args:
            week_offset: Week offset (0=current, 1=next, -1=previous)

        Returns:
            Dict with scrape results
        """
        logger.info(f"Starting manual scrape for week offset {week_offset}")

        try:
            events, scrape_log = self.scraper.scrape_week(week_offset=week_offset)
            self.scraper.save_events_to_database(events, scrape_log)

            result = {
                'success': True,
                'events_count': len(events),
                'scrape_log_id': scrape_log.id,
                'duration': scrape_log.duration_seconds,
                'message': f"Manual scrape completed: {len(events)} events"
            }

            logger.info(f"Manual scrape completed: {len(events)} events")
            return result

        except Exception as e:
            error_msg = f"Manual scrape failed: {str(e)}"
            logger.error(error_msg)

            return {
                'success': False,
                'events_count': 0,
                'error': error_msg,
                'message': error_msg
            }

    async def send_scrape_notification(self, success: bool, total_events: int, message: str):
        """Send notification about scrape results (placeholder for future webhook/Slack integration)"""
        if config.WEBHOOK_URL:
            # TODO: Implement webhook notification
            logger.info(f"Would send webhook notification: {message}")

        if config.SLACK_TOKEN:
            # TODO: Implement Slack notification
            logger.info(f"Would send Slack notification: {message}")

    def get_job_status(self) -> dict:
        """Get status of scheduled jobs"""
        if not self.scheduler:
            return {'status': 'not_initialized', 'jobs': []}

        jobs = []
        for job in self.scheduler.get_jobs():
            job_info = {
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger),
                'active': True
            }
            jobs.append(job_info)

        return {
            'status': 'running' if self.scheduler.running else 'stopped',
            'timezone': str(self.scheduler.timezone),
            'jobs': jobs
        }

    def pause_job(self, job_id: str) -> bool:
        """Pause a specific job"""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Job '{job_id}' paused")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job '{job_id}': {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a specific job"""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Job '{job_id}' resumed")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job '{job_id}': {e}")
            return False

    def get_next_scheduled_scrape(self) -> Optional[datetime]:
        """Get the next scheduled scrape time"""
        try:
            job = self.scheduler.get_job('weekly_scrape')
            if job and job.next_run_time:
                return job.next_run_time
        except Exception as e:
            logger.error(f"Failed to get next scrape time: {e}")

        return None


# Global scheduler instance
economic_scheduler = EconomicCalendarScheduler()