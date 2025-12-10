#!/usr/bin/env python3
"""
Update Stock Fundamentals Script

Fetches fundamental data from yfinance for all stocks or specific tickers.
Includes retry logic for failed tickers and rate limiting to avoid API blocks.

Usage:
    docker exec task-worker python3 /app/stock_scanner/scripts/update_fundamentals.py
    docker exec task-worker python3 /app/stock_scanner/scripts/update_fundamentals.py --force
    docker exec task-worker python3 /app/stock_scanner/scripts/update_fundamentals.py --tickers AAPL,MSFT,GOOGL
    docker exec task-worker python3 /app/stock_scanner/scripts/update_fundamentals.py --watchlist-only
    docker exec task-worker python3 /app/stock_scanner/scripts/update_fundamentals.py --retry-failed
"""
import sys
sys.path.insert(0, '/app')

import asyncio
import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.core.fundamentals.fundamentals_fetcher import FundamentalsFetcher
from stock_scanner import config

# File to track failed tickers for retry
FAILED_TICKERS_FILE = Path('/app/stock_scanner/logs/failed_fundamentals.json')


def load_failed_tickers() -> Dict:
    """Load previously failed tickers from file."""
    if FAILED_TICKERS_FILE.exists():
        try:
            with open(FAILED_TICKERS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load failed tickers file: {e}")
    return {'failed': [], 'last_updated': None, 'retry_count': 0}


def save_failed_tickers(failed_data: Dict):
    """Save failed tickers to file for later retry."""
    FAILED_TICKERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    failed_data['last_updated'] = datetime.now().isoformat()
    with open(FAILED_TICKERS_FILE, 'w') as f:
        json.dump(failed_data, f, indent=2)
    logger.info(f"Saved {len(failed_data['failed'])} failed tickers to {FAILED_TICKERS_FILE}")


async def fetch_single_ticker(
    fetcher: FundamentalsFetcher,
    ticker: str,
    delay: float = 0.5,
    attempt: int = 1
) -> Tuple[str, bool, Optional[Dict], bool]:
    """
    Fetch fundamentals for a single ticker with delay.

    Returns:
        Tuple of (ticker, success, fundamentals_dict or None, is_rate_limited)
    """
    await asyncio.sleep(delay)  # Rate limiting delay

    try:
        fundamentals = await fetcher.fetch_fundamentals(ticker)
        if fundamentals:
            success = await fetcher.save_fundamentals(fundamentals)
            return (ticker, success, fundamentals if success else None, False)
        return (ticker, False, None, False)
    except Exception as e:
        error_str = str(e).lower()
        # Detect rate limiting errors
        is_rate_limited = any(x in error_str for x in [
            'rate limit', 'too many requests', '429', '401', 'unauthorized', 'invalid crumb'
        ])
        if is_rate_limited:
            logger.warning(f"Rate limited on {ticker}: {e}")
        else:
            logger.debug(f"Error fetching {ticker}: {e}")
        return (ticker, False, None, is_rate_limited)


async def run_fundamentals_update(
    force: bool = False,
    tickers: list = None,
    watchlist_only: bool = False,
    concurrency: int = 5,
    delay: float = 0.3,
    max_retries: int = 3,
    retry_failed: bool = False
):
    """
    Run fundamentals update with retry logic.

    Args:
        force: If True, update all stocks regardless of last update time
        tickers: Optional list of specific tickers to update
        watchlist_only: If True, only update stocks in the watchlist
        concurrency: Number of concurrent fetches (lower = less rate limiting)
        delay: Delay between requests in seconds
        max_retries: Maximum number of retry rounds for failed tickers
        retry_failed: If True, only retry previously failed tickers
    """
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    print("=" * 70)
    print("STOCK FUNDAMENTALS UPDATE (with retry logic)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Concurrency: {concurrency} | Delay: {delay}s | Max retries: {max_retries}")
    print("=" * 70)

    try:
        fetcher = FundamentalsFetcher(db_manager=db, thread_pool_size=4)

        # Determine which tickers to update
        if retry_failed:
            # Load previously failed tickers
            failed_data = load_failed_tickers()
            tickers_to_update = failed_data.get('failed', [])
            if not tickers_to_update:
                print("\nNo failed tickers to retry!")
                return
            print(f"\nRetrying {len(tickers_to_update)} previously failed tickers...")
        elif tickers:
            tickers_to_update = tickers
            print(f"\nUpdating {len(tickers)} specified tickers...")
        elif watchlist_only:
            rows = await db.fetch("""
                SELECT DISTINCT ticker FROM stock_watchlist
                ORDER BY ticker
            """)
            tickers_to_update = [row['ticker'] for row in rows]
            print(f"\nUpdating {len(tickers_to_update)} tickers from watchlist...")
        elif force:
            rows = await db.fetch("""
                SELECT ticker FROM stock_instruments
                WHERE is_active = TRUE AND is_tradeable = TRUE
                ORDER BY ticker
            """)
            tickers_to_update = [row['ticker'] for row in rows]
            print(f"\nForce updating ALL {len(tickers_to_update)} active tickers...")
        else:
            # Get stale tickers (not updated in 7 days)
            rows = await db.fetch("""
                SELECT ticker FROM stock_instruments
                WHERE is_active = TRUE AND is_tradeable = TRUE
                AND (fundamentals_updated_at IS NULL
                     OR fundamentals_updated_at < NOW() - INTERVAL '7 days')
                ORDER BY ticker
            """)
            tickers_to_update = [row['ticker'] for row in rows]
            print(f"\nUpdating {len(tickers_to_update)} stale tickers...")

        if not tickers_to_update:
            print("\nNo tickers to update!")
            await fetcher.close()
            return

        # Stats tracking
        total_stats = {
            'total': len(tickers_to_update),
            'successful': 0,
            'failed': 0,
            'with_earnings': 0,
            'with_beta': 0,
            'with_short_interest': 0,
            'with_growth': 0,
            'with_profitability': 0,
            'with_analyst_rating': 0,
            'with_dividend': 0,
            'with_52w_data': 0,
            'with_market_cap': 0,
            'with_sector': 0,
        }

        failed_tickers = []
        current_tickers = tickers_to_update.copy()
        retry_round = 0

        while current_tickers and retry_round <= max_retries:
            if retry_round > 0:
                print(f"\n{'='*70}")
                print(f"RETRY ROUND {retry_round}/{max_retries}")
                print(f"Retrying {len(current_tickers)} failed tickers...")
                print(f"Waiting 30 seconds before retry (rate limit cooldown)...")
                await asyncio.sleep(30)  # Cooldown before retry

            round_start = time.time()
            round_successful = 0
            round_failed = []

            # Process in batches with concurrency control
            semaphore = asyncio.Semaphore(concurrency)
            rate_limit_count = 0  # Track consecutive rate limits

            async def fetch_with_semaphore(ticker: str, idx: int) -> Tuple[str, bool, Optional[Dict], bool]:
                async with semaphore:
                    # Add progressive delay based on position
                    extra_delay = (idx % 10) * 0.1  # Stagger requests
                    return await fetch_single_ticker(fetcher, ticker, delay + extra_delay)

            # Create tasks
            tasks = [fetch_with_semaphore(t, i) for i, t in enumerate(current_tickers)]

            # Process with progress reporting
            completed = 0
            batch_size = 50

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                results = await asyncio.gather(*batch)

                batch_rate_limited = 0
                for ticker, success, fundamentals, is_rate_limited in results:
                    completed += 1

                    if is_rate_limited:
                        batch_rate_limited += 1
                        round_failed.append(ticker)
                    elif success and fundamentals:
                        round_successful += 1
                        total_stats['successful'] += 1
                        rate_limit_count = 0  # Reset on success

                        # Update detailed stats
                        if fundamentals.get('earnings_date'):
                            total_stats['with_earnings'] += 1
                        if fundamentals.get('beta'):
                            total_stats['with_beta'] += 1
                        if fundamentals.get('short_ratio') or fundamentals.get('short_percent_float'):
                            total_stats['with_short_interest'] += 1
                        if fundamentals.get('revenue_growth') or fundamentals.get('earnings_growth'):
                            total_stats['with_growth'] += 1
                        if fundamentals.get('profit_margin') or fundamentals.get('return_on_equity'):
                            total_stats['with_profitability'] += 1
                        if fundamentals.get('analyst_rating'):
                            total_stats['with_analyst_rating'] += 1
                        if fundamentals.get('dividend_yield'):
                            total_stats['with_dividend'] += 1
                        if fundamentals.get('fifty_two_week_high'):
                            total_stats['with_52w_data'] += 1
                        if fundamentals.get('market_cap'):
                            total_stats['with_market_cap'] += 1
                        if fundamentals.get('sector'):
                            total_stats['with_sector'] += 1
                    else:
                        round_failed.append(ticker)

                # If most of batch was rate limited, pause and backoff
                if batch_rate_limited > batch_size * 0.5:
                    rate_limit_count += 1
                    backoff_time = min(60 * rate_limit_count, 300)  # Max 5 min backoff
                    print(f"\n  ⚠️  Rate limiting detected! Pausing {backoff_time}s...")
                    await asyncio.sleep(backoff_time)

                # Progress report
                pct = (completed / len(current_tickers)) * 100
                elapsed = time.time() - round_start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(current_tickers) - completed) / rate if rate > 0 else 0

                print(f"  [{completed:4d}/{len(current_tickers)}] {pct:5.1f}% | "
                      f"OK: {round_successful} | Failed: {len(round_failed)} | "
                      f"ETA: {int(eta//60):02d}:{int(eta%60):02d}")

            # Update for next round
            current_tickers = round_failed
            total_stats['failed'] = len(round_failed)

            round_elapsed = time.time() - round_start
            print(f"\nRound {retry_round} complete in {int(round_elapsed//60)}m {int(round_elapsed%60)}s")
            print(f"  Successful: {round_successful} | Still failing: {len(round_failed)}")

            retry_round += 1

        # Save any remaining failed tickers for later retry
        if current_tickers:
            failed_data = {
                'failed': current_tickers,
                'retry_count': retry_round,
            }
            save_failed_tickers(failed_data)
            print(f"\n⚠️  {len(current_tickers)} tickers still failing after {max_retries} retries")
            print(f"   Saved to {FAILED_TICKERS_FILE} for later retry")
        else:
            # Clear the failed file if all succeeded
            if FAILED_TICKERS_FILE.exists():
                FAILED_TICKERS_FILE.unlink()
                print("\n✅ All tickers updated successfully! Cleared failed tickers file.")

        # Final Summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total attempted: {total_stats['total']}")
        print(f"Successful: {total_stats['successful']} ({100*total_stats['successful']/total_stats['total']:.1f}%)")
        print(f"Failed: {total_stats['failed']} ({100*total_stats['failed']/total_stats['total']:.1f}%)")
        print(f"\nData coverage (of successful):")
        if total_stats['successful'] > 0:
            print(f"  - With sector: {total_stats['with_sector']} ({100*total_stats['with_sector']/total_stats['successful']:.1f}%)")
            print(f"  - With market cap: {total_stats['with_market_cap']} ({100*total_stats['with_market_cap']/total_stats['successful']:.1f}%)")
            print(f"  - With earnings date: {total_stats['with_earnings']} ({100*total_stats['with_earnings']/total_stats['successful']:.1f}%)")
            print(f"  - With beta: {total_stats['with_beta']} ({100*total_stats['with_beta']/total_stats['successful']:.1f}%)")
            print(f"  - With short interest: {total_stats['with_short_interest']} ({100*total_stats['with_short_interest']/total_stats['successful']:.1f}%)")
            print(f"  - With growth data: {total_stats['with_growth']} ({100*total_stats['with_growth']/total_stats['successful']:.1f}%)")
            print(f"  - With profitability: {total_stats['with_profitability']} ({100*total_stats['with_profitability']/total_stats['successful']:.1f}%)")
            print(f"  - With analyst ratings: {total_stats['with_analyst_rating']} ({100*total_stats['with_analyst_rating']/total_stats['successful']:.1f}%)")
            print(f"  - With dividends: {total_stats['with_dividend']} ({100*total_stats['with_dividend']/total_stats['successful']:.1f}%)")
            print(f"  - With 52-week data: {total_stats['with_52w_data']} ({100*total_stats['with_52w_data']/total_stats['successful']:.1f}%)")

        await fetcher.close()

    finally:
        await db.close()

    print(f"\n✅ Done at {datetime.now().strftime('%H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description='Update stock fundamentals from yfinance')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force update all stocks regardless of last update time')
    parser.add_argument('--tickers', '-t', type=str,
                       help='Comma-separated list of specific tickers to update')
    parser.add_argument('--watchlist-only', '-w', action='store_true',
                       help='Only update stocks that are in the watchlist')
    parser.add_argument('--retry-failed', '-r', action='store_true',
                       help='Retry previously failed tickers from the log file')
    parser.add_argument('--concurrency', '-c', type=int, default=5,
                       help='Number of concurrent fetches (default: 5, lower = less rate limiting)')
    parser.add_argument('--delay', '-d', type=float, default=0.3,
                       help='Delay between requests in seconds (default: 0.3)')
    parser.add_argument('--max-retries', '-m', type=int, default=3,
                       help='Maximum retry rounds for failed tickers (default: 3)')

    args = parser.parse_args()

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]

    asyncio.run(run_fundamentals_update(
        force=args.force,
        tickers=tickers,
        watchlist_only=args.watchlist_only,
        concurrency=args.concurrency,
        delay=args.delay,
        max_retries=args.max_retries,
        retry_failed=args.retry_failed
    ))


if __name__ == '__main__':
    main()
