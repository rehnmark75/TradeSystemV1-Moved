#!/usr/bin/env python3
"""
Stock Scanner Scheduler

Runs scheduled tasks for the stock scanner:
- Daily data updates after market close (10 PM ET / 3 AM UTC)
- Weekly instrument sync (Sunday midnight)

Usage:
    # Run as standalone scheduler
    docker exec task-worker python -m stock_scanner.scheduler

    # Or add to docker-compose as separate service
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, time, timedelta
import pytz

sys.path.insert(0, '/app')

from stock_scanner import config
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.core.data_fetcher import StockDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("stock_scheduler")


class StockScheduler:
    """
    Scheduler for stock data updates.

    Schedule:
    - Daily 10:30 PM ET (after market close + buffer): Update all stock data
    - Weekly Sunday 1:00 AM ET: Sync instrument list from RoboMarkets
    """

    # US Eastern timezone
    ET = pytz.timezone('America/New_York')

    # Schedule times (ET)
    DAILY_UPDATE_TIME = time(22, 30)  # 10:30 PM ET
    WEEKLY_SYNC_TIME = time(1, 0)     # 1:00 AM ET on Sunday
    WEEKLY_SYNC_DAY = 6               # Sunday = 6

    def __init__(self):
        self.db: AsyncDatabaseManager = None
        self.fetcher: StockDataFetcher = None
        self.running = False

    async def setup(self):
        """Initialize database and fetcher"""
        logger.info("Initializing Stock Scheduler...")

        self.db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
        await self.db.connect()

        self.fetcher = StockDataFetcher(db_manager=self.db)

        logger.info("Stock Scheduler initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.fetcher:
            await self.fetcher.close()
        if self.db:
            await self.db.close()
        logger.info("Stock Scheduler cleaned up")

    def get_next_daily_update(self) -> datetime:
        """Get next daily update time"""
        now = datetime.now(self.ET)
        target = datetime.combine(now.date(), self.DAILY_UPDATE_TIME)
        target = self.ET.localize(target)

        # If we've passed today's time, schedule for tomorrow
        if now >= target:
            target += timedelta(days=1)

        # Skip weekends (Saturday=5, Sunday=6)
        while target.weekday() in (5, 6):
            target += timedelta(days=1)

        return target

    def get_next_weekly_sync(self) -> datetime:
        """Get next weekly instrument sync time"""
        now = datetime.now(self.ET)
        target = datetime.combine(now.date(), self.WEEKLY_SYNC_TIME)
        target = self.ET.localize(target)

        # Find next Sunday
        days_until_sunday = (self.WEEKLY_SYNC_DAY - now.weekday()) % 7
        if days_until_sunday == 0 and now >= target:
            days_until_sunday = 7

        target += timedelta(days=days_until_sunday)
        return target

    async def daily_update(self):
        """Run daily data update for all stocks"""
        logger.info("=" * 60)
        logger.info("DAILY UPDATE - Starting stock data update")
        logger.info("=" * 60)

        start_time = datetime.now()

        try:
            # Get tickers needing update (older than 20 hours)
            query = """
                SELECT i.ticker
                FROM stock_instruments i
                LEFT JOIN (
                    SELECT ticker, MAX(timestamp) as last_ts
                    FROM stock_candles WHERE timeframe = '1h'
                    GROUP BY ticker
                ) c ON i.ticker = c.ticker
                WHERE i.is_active = TRUE AND i.is_tradeable = TRUE
                AND (c.last_ts IS NULL OR c.last_ts < NOW() - INTERVAL '20 hours')
            """
            rows = await self.db.fetch(query)
            tickers = [row['ticker'] for row in rows]

            if not tickers:
                logger.info("All tickers up to date, nothing to update")
                return

            logger.info(f"Updating {len(tickers)} tickers...")

            # Update with concurrency
            successful = 0
            failed = 0
            total_candles = 0

            semaphore = asyncio.Semaphore(5)

            async def update_ticker(ticker):
                async with semaphore:
                    try:
                        count = await self.fetcher.fetch_historical_data(
                            ticker, days=5, interval='1h'
                        )
                        return (ticker, count, None)
                    except Exception as e:
                        return (ticker, 0, str(e))

            tasks = [update_ticker(t) for t in tickers]
            results = await asyncio.gather(*tasks)

            for ticker, count, error in results:
                if error:
                    failed += 1
                elif count > 0:
                    successful += 1
                    total_candles += count
                else:
                    failed += 1

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info(f"Daily update complete:")
            logger.info(f"  - Duration: {int(elapsed//60)}m {int(elapsed%60)}s")
            logger.info(f"  - Successful: {successful}/{len(tickers)}")
            logger.info(f"  - Failed: {failed}")
            logger.info(f"  - New candles: {total_candles:,}")

        except Exception as e:
            logger.error(f"Daily update failed: {e}")

    async def weekly_sync(self):
        """Run weekly instrument sync"""
        logger.info("=" * 60)
        logger.info("WEEKLY SYNC - Syncing instruments from RoboMarkets")
        logger.info("=" * 60)

        try:
            stats = await self.fetcher.sync_us_stocks()

            logger.info(f"Weekly sync complete:")
            logger.info(f"  - Total from API: {stats['total_from_api']:,}")
            logger.info(f"  - US stocks: {stats['unique_tickers']:,}")
            logger.info(f"  - NYSE: {stats['nyse_count']:,}")
            logger.info(f"  - NASDAQ: {stats['nasdaq_count']:,}")
            logger.info(f"  - New: {stats['inserted']}")
            logger.info(f"  - Updated: {stats['updated']}")

        except Exception as e:
            logger.error(f"Weekly sync failed: {e}")

    async def run(self):
        """Main scheduler loop"""
        await self.setup()
        self.running = True

        logger.info("Stock Scheduler started")
        logger.info(f"  Daily update time: {self.DAILY_UPDATE_TIME} ET (Mon-Fri)")
        logger.info(f"  Weekly sync time: Sunday {self.WEEKLY_SYNC_TIME} ET")

        try:
            while self.running:
                now = datetime.now(self.ET)

                # Calculate next scheduled times
                next_daily = self.get_next_daily_update()
                next_weekly = self.get_next_weekly_sync()

                # Find which comes first
                if next_weekly < next_daily:
                    next_task = "weekly_sync"
                    next_time = next_weekly
                else:
                    next_task = "daily_update"
                    next_time = next_daily

                # Calculate sleep duration
                sleep_seconds = (next_time - now).total_seconds()

                logger.info(f"Next task: {next_task} at {next_time.strftime('%Y-%m-%d %H:%M %Z')}")
                logger.info(f"Sleeping for {int(sleep_seconds//3600)}h {int((sleep_seconds%3600)//60)}m")

                # Sleep until next task (check every minute for shutdown)
                while sleep_seconds > 0 and self.running:
                    await asyncio.sleep(min(60, sleep_seconds))
                    sleep_seconds -= 60

                if not self.running:
                    break

                # Execute the task
                if next_task == "daily_update":
                    await self.daily_update()
                else:
                    await self.weekly_sync()

        except asyncio.CancelledError:
            logger.info("Scheduler cancelled")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        finally:
            await self.cleanup()

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.running = False


async def run_once(task: str):
    """Run a specific task once"""
    scheduler = StockScheduler()
    await scheduler.setup()

    try:
        if task == "daily":
            await scheduler.daily_update()
        elif task == "weekly":
            await scheduler.weekly_sync()
        else:
            print(f"Unknown task: {task}")
    finally:
        await scheduler.cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stock Scanner Scheduler')
    parser.add_argument('command', nargs='?', default='run',
                       choices=['run', 'daily', 'weekly', 'status'],
                       help='Command to execute')
    args = parser.parse_args()

    if args.command == 'run':
        # Run continuous scheduler
        print("Starting Stock Scanner Scheduler...")
        print("Press Ctrl+C to stop")

        scheduler = StockScheduler()

        try:
            asyncio.run(scheduler.run())
        except KeyboardInterrupt:
            print("\nShutdown requested...")
            scheduler.stop()

    elif args.command == 'daily':
        # Run daily update once
        print("Running daily update...")
        asyncio.run(run_once('daily'))

    elif args.command == 'weekly':
        # Run weekly sync once
        print("Running weekly sync...")
        asyncio.run(run_once('weekly'))

    elif args.command == 'status':
        # Show scheduler status
        scheduler = StockScheduler()

        print("\nStock Scanner Scheduler Status")
        print("=" * 40)
        print(f"Daily update time: {scheduler.DAILY_UPDATE_TIME} ET (Mon-Fri)")
        print(f"Weekly sync time: Sunday {scheduler.WEEKLY_SYNC_TIME} ET")
        print(f"\nNext daily update: {scheduler.get_next_daily_update()}")
        print(f"Next weekly sync: {scheduler.get_next_weekly_sync()}")


if __name__ == '__main__':
    main()
