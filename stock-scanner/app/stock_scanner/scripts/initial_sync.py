#!/usr/bin/env python3
"""
Initial Stock Data Sync Script

Fetches historical data for all US stocks in the database.
Shows progress and can be resumed if interrupted.

Usage:
    docker exec task-worker python -m stock_scanner.scripts.initial_sync
    docker exec task-worker python -m stock_scanner.scripts.initial_sync --days 30
    docker exec task-worker python -m stock_scanner.scripts.initial_sync --concurrency 10
"""

import asyncio
import argparse
import logging
import sys
import time
from datetime import datetime

# Add parent to path
sys.path.insert(0, '/app')

from stock_scanner import config
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.core.data_fetcher import StockDataFetcher

logging.basicConfig(
    level=logging.WARNING,  # Quiet mode - only warnings and errors
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


async def get_tickers_without_data(db: AsyncDatabaseManager) -> list:
    """Get tickers that don't have candle data yet"""
    query = """
        SELECT i.ticker
        FROM stock_instruments i
        LEFT JOIN (
            SELECT DISTINCT ticker FROM stock_candles WHERE timeframe = '1h'
        ) c ON i.ticker = c.ticker
        WHERE i.is_active = TRUE
          AND i.is_tradeable = TRUE
          AND c.ticker IS NULL
        ORDER BY i.ticker
    """
    rows = await db.fetch(query)
    return [row['ticker'] for row in rows]


async def get_all_tickers(db: AsyncDatabaseManager) -> list:
    """Get all active tradeable tickers"""
    query = """
        SELECT ticker FROM stock_instruments
        WHERE is_active = TRUE AND is_tradeable = TRUE
        ORDER BY ticker
    """
    rows = await db.fetch(query)
    return [row['ticker'] for row in rows]


async def fetch_single_ticker(fetcher: StockDataFetcher, ticker: str, days: int) -> tuple:
    """Fetch data for a single ticker, return (ticker, count, error)"""
    try:
        count = await fetcher.fetch_historical_data(ticker, days=days, interval='1h')
        return (ticker, count, None)
    except Exception as e:
        return (ticker, 0, str(e))


async def main():
    parser = argparse.ArgumentParser(description='Initial stock data sync')
    parser.add_argument('--days', type=int, default=60, help='Days of history to fetch (default: 60)')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrent fetches (default: 5)')
    parser.add_argument('--resume', action='store_true', help='Only fetch tickers without data')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of tickers (0=all)')
    args = parser.parse_args()

    print("=" * 60)
    print("STOCK SCANNER - INITIAL DATA SYNC")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Days of history: {args.days}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Mode: {'Resume (skip existing)' if args.resume else 'Full sync'}")
    print("=" * 60)

    # Connect to database
    db_url = config.STOCKS_DATABASE_URL
    db = AsyncDatabaseManager(db_url)
    await db.connect()

    try:
        # Get tickers to process
        if args.resume:
            tickers = await get_tickers_without_data(db)
            print(f"\n[INFO] Found {len(tickers)} tickers without data")
        else:
            tickers = await get_all_tickers(db)
            print(f"\n[INFO] Found {len(tickers)} total tickers")

        if args.limit > 0:
            tickers = tickers[:args.limit]
            print(f"[INFO] Limited to {len(tickers)} tickers")

        if not tickers:
            print("\n[OK] All tickers already have data!")
            return

        # Initialize data fetcher
        fetcher = StockDataFetcher(db_manager=db)

        # Track progress
        total = len(tickers)
        completed = 0
        successful = 0
        failed = 0
        total_candles = 0
        failed_tickers = []
        start_time = time.time()

        # Process in batches with concurrency
        semaphore = asyncio.Semaphore(args.concurrency)

        async def fetch_with_semaphore(ticker: str):
            async with semaphore:
                return await fetch_single_ticker(fetcher, ticker, args.days)

        print(f"\n[START] Fetching data for {total} tickers...\n")

        # Create all tasks
        tasks = [fetch_with_semaphore(ticker) for ticker in tickers]

        # Process with progress updates
        for coro in asyncio.as_completed(tasks):
            ticker, count, error = await coro
            completed += 1

            if error:
                failed += 1
                failed_tickers.append(ticker)
                status = "FAIL"
            elif count > 0:
                successful += 1
                total_candles += count
                status = f"OK ({count} candles)"
            else:
                failed += 1
                failed_tickers.append(ticker)
                status = "NO DATA"

            # Progress bar
            pct = (completed / total) * 100
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0

            # Print progress every ticker
            print(f"[{completed:4d}/{total}] {pct:5.1f}% | {ticker:8s} | {status:20s} | "
                  f"ETA: {int(eta//60):02d}:{int(eta%60):02d}", end='\r')

            # Detailed progress every 100 tickers
            if completed % 100 == 0:
                print(f"\n[PROGRESS] {completed}/{total} ({pct:.1f}%) | "
                      f"Success: {successful} | Failed: {failed} | "
                      f"Candles: {total_candles:,} | "
                      f"Rate: {rate:.1f}/sec")

        # Final summary
        elapsed = time.time() - start_time
        print("\n")
        print("=" * 60)
        print("SYNC COMPLETE")
        print("=" * 60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {int(elapsed//60)} minutes {int(elapsed%60)} seconds")
        print(f"Total tickers: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total candles: {total_candles:,}")
        print(f"Average rate: {total/elapsed:.2f} tickers/sec")

        if failed_tickers:
            print(f"\n[WARN] Failed tickers ({len(failed_tickers)}):")
            for t in failed_tickers[:20]:
                print(f"   - {t}")
            if len(failed_tickers) > 20:
                print(f"   ... and {len(failed_tickers) - 20} more")

            # Save failed tickers to file
            with open('/app/stock_scanner/logs/failed_tickers.txt', 'w') as f:
                f.write('\n'.join(failed_tickers))
            print(f"\n[INFO] Failed tickers saved to: /app/stock_scanner/logs/failed_tickers.txt")

        print("=" * 60)

        await fetcher.close()

    finally:
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
