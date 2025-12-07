#!/usr/bin/env python3
"""
Daily Stock Data Update Script

Fetches new candles for all active stocks since their last update.
Designed to run daily after market close (after 4 PM ET).

Usage:
    docker exec task-worker python -m stock_scanner.scripts.daily_update
    docker exec task-worker python -m stock_scanner.scripts.daily_update --concurrency 10

Recommended cron (run at 10 PM ET / 3 AM UTC):
    0 3 * * 1-5 docker exec task-worker python -m stock_scanner.scripts.daily_update
"""

import asyncio
import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, '/app')

from stock_scanner import config
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.core.data_fetcher import StockDataFetcher

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


async def get_tickers_needing_update(db: AsyncDatabaseManager, hours_threshold: int = 24) -> list:
    """
    Get tickers that haven't been updated recently.

    Returns tickers where:
    - is_active = TRUE
    - No candles, OR latest candle is older than threshold
    """
    query = """
        SELECT i.ticker,
               MAX(c.timestamp) as last_candle,
               COUNT(c.id) as candle_count
        FROM stock_instruments i
        LEFT JOIN stock_candles c ON i.ticker = c.ticker AND c.timeframe = '1h'
        WHERE i.is_active = TRUE AND i.is_tradeable = TRUE
        GROUP BY i.ticker
        HAVING MAX(c.timestamp) IS NULL
           OR MAX(c.timestamp) < NOW() - INTERVAL '%s hours'
        ORDER BY MAX(c.timestamp) NULLS FIRST
    """ % hours_threshold

    rows = await db.fetch(query)
    return [(row['ticker'], row['last_candle'], row['candle_count']) for row in rows]


async def get_update_stats(db: AsyncDatabaseManager) -> dict:
    """Get current data statistics"""
    stats = {}

    # Total active tickers
    stats['active_tickers'] = await db.fetchval(
        "SELECT COUNT(*) FROM stock_instruments WHERE is_active = TRUE"
    )

    # Tickers with data
    stats['tickers_with_data'] = await db.fetchval(
        "SELECT COUNT(DISTINCT ticker) FROM stock_candles WHERE timeframe = '1h'"
    )

    # Total candles
    stats['total_candles'] = await db.fetchval(
        "SELECT COUNT(*) FROM stock_candles WHERE timeframe = '1h'"
    )

    # Latest candle timestamp
    stats['latest_candle'] = await db.fetchval(
        "SELECT MAX(timestamp) FROM stock_candles WHERE timeframe = '1h'"
    )

    # Oldest candle timestamp
    stats['oldest_candle'] = await db.fetchval(
        "SELECT MIN(timestamp) FROM stock_candles WHERE timeframe = '1h'"
    )

    return stats


async def update_single_ticker(fetcher: StockDataFetcher, ticker: str, days: int = 5) -> tuple:
    """
    Incremental update for a single ticker.
    Fetches last N days to catch any missed candles.
    """
    try:
        count = await fetcher.fetch_historical_data(ticker, days=days, interval='1h')
        return (ticker, count, None)
    except Exception as e:
        return (ticker, 0, str(e))


async def main():
    parser = argparse.ArgumentParser(description='Daily stock data update')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrent fetches (default: 5)')
    parser.add_argument('--days', type=int, default=5, help='Days to fetch per ticker (default: 5)')
    parser.add_argument('--hours', type=int, default=20, help='Update tickers older than N hours (default: 20)')
    parser.add_argument('--force', action='store_true', help='Force update all tickers')
    args = parser.parse_args()

    print("=" * 60)
    print("STOCK SCANNER - DAILY UPDATE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Days per ticker: {args.days}")
    print(f"Update threshold: {args.hours} hours")
    print("=" * 60)

    # Connect to database
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    try:
        # Show current stats
        stats = await get_update_stats(db)
        print(f"\n[STATS] Current database status:")
        print(f"   Active tickers: {stats['active_tickers']:,}")
        print(f"   Tickers with data: {stats['tickers_with_data']:,}")
        print(f"   Total 1h candles: {stats['total_candles']:,}")
        if stats['latest_candle']:
            print(f"   Latest candle: {stats['latest_candle']}")
        if stats['oldest_candle']:
            print(f"   Oldest candle: {stats['oldest_candle']}")

        # Get tickers needing update
        if args.force:
            # Force update all active tickers
            rows = await db.fetch(
                "SELECT ticker FROM stock_instruments WHERE is_active = TRUE ORDER BY ticker"
            )
            tickers_info = [(row['ticker'], None, 0) for row in rows]
            print(f"\n[INFO] Force updating ALL {len(tickers_info)} active tickers")
        else:
            tickers_info = await get_tickers_needing_update(db, args.hours)
            print(f"\n[INFO] Found {len(tickers_info)} tickers needing update")

        if not tickers_info:
            print("\n[OK] All tickers are up to date!")
            return

        # Initialize data fetcher
        fetcher = StockDataFetcher(db_manager=db)

        # Track progress
        total = len(tickers_info)
        completed = 0
        successful = 0
        failed = 0
        total_candles = 0
        start_time = time.time()

        # Process with concurrency
        semaphore = asyncio.Semaphore(args.concurrency)

        async def update_with_semaphore(ticker: str):
            async with semaphore:
                return await update_single_ticker(fetcher, ticker, args.days)

        print(f"\n[START] Updating {total} tickers...\n")

        # Create tasks
        tickers = [t[0] for t in tickers_info]
        tasks = [update_with_semaphore(ticker) for ticker in tickers]

        # Process with progress
        for coro in asyncio.as_completed(tasks):
            ticker, count, error = await coro
            completed += 1

            if error:
                failed += 1
            elif count > 0:
                successful += 1
                total_candles += count
            else:
                failed += 1

            # Progress
            pct = (completed / total) * 100
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0

            print(f"[{completed:4d}/{total}] {pct:5.1f}% | {ticker:8s} | "
                  f"{'OK' if not error else 'FAIL':4s} | "
                  f"ETA: {int(eta//60):02d}:{int(eta%60):02d}", end='\r')

            # Detailed progress every 500 tickers
            if completed % 500 == 0:
                print(f"\n[PROGRESS] {completed}/{total} ({pct:.1f}%) | "
                      f"Success: {successful} | Failed: {failed} | "
                      f"Candles: {total_candles:,}")

        # Final summary
        elapsed = time.time() - start_time
        print("\n")
        print("=" * 60)
        print("UPDATE COMPLETE")
        print("=" * 60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {int(elapsed//60)} minutes {int(elapsed%60)} seconds")
        print(f"Tickers updated: {successful}/{total}")
        print(f"Failed: {failed}")
        print(f"New candles: {total_candles:,}")

        # Show new stats
        new_stats = await get_update_stats(db)
        print(f"\n[STATS] Updated database status:")
        print(f"   Total 1h candles: {new_stats['total_candles']:,} (+{new_stats['total_candles'] - stats['total_candles']:,})")
        if new_stats['latest_candle']:
            print(f"   Latest candle: {new_stats['latest_candle']}")

        print("=" * 60)

        await fetcher.close()

    finally:
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
