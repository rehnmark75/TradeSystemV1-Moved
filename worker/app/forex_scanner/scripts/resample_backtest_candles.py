#!/usr/bin/env python3
"""
Resample 1m candles into 5m, 15m, and 4h for the backtest table.

This script populates ig_candles_backtest with pre-computed candles
for fast backtesting without runtime resampling.

NOTE: The BacktestScanner now auto-validates and updates the backtest
candles table before each run. This script is useful for:
- Initial bulk population of the table
- Manual refresh of all data
- Specific epic updates

Usage:
    docker exec -it task-worker python /app/forex_scanner/scripts/resample_backtest_candles.py
    docker exec -it task-worker python /app/forex_scanner/scripts/resample_backtest_candles.py --epic CS.D.EURUSD.CEEM.IP
    docker exec -it task-worker python /app/forex_scanner/scripts/resample_backtest_candles.py --days 30
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Optional

# Setup path for imports
sys.path.insert(0, '/app')

from forex_scanner.core.database import DatabaseManager
from forex_scanner.core.backtest_candles_manager import BacktestCandlesManager
from forex_scanner import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_epics(db: DatabaseManager) -> List[str]:
    """Get all unique epics from the 1m candles table."""
    query = "SELECT DISTINCT epic FROM ig_candles WHERE timeframe = 1 ORDER BY epic"
    result = db.execute_query(query)
    return result['epic'].tolist() if not result.empty else []


def main():
    parser = argparse.ArgumentParser(description='Resample 1m candles for backtest table')
    parser.add_argument('--epic', type=str, help='Specific epic to resample (default: all)')
    parser.add_argument('--days', type=int, help='Only process last N days (default: all)')
    parser.add_argument('--timeframes', type=str, default='5,15,240',
                        help='Comma-separated target timeframes in minutes (default: 5,15,240)')
    args = parser.parse_args()

    # Parse timeframes
    target_timeframes = [int(tf.strip()) for tf in args.timeframes.split(',')]

    # Calculate since date
    since = None
    until = datetime.now()
    if args.days:
        since = datetime.now() - timedelta(days=args.days)
        logger.info(f"Processing data since {since.date()}")

    # Connect to database
    logger.info("Connecting to database...")
    db = DatabaseManager(config.DATABASE_URL)

    # Get epics to process
    if args.epic:
        epics = [args.epic]
    else:
        epics = get_epics(db)

    logger.info(f"Processing {len(epics)} epics: {epics}")
    logger.info(f"Target timeframes: {target_timeframes}")

    # Use the BacktestCandlesManager for resampling
    manager = BacktestCandlesManager(db, timeframes=target_timeframes)

    # Ensure table exists
    manager.create_table_if_needed()

    # Process each epic
    total_results = {tf: 0 for tf in target_timeframes}

    for epic in epics:
        logger.info(f"\nProcessing {epic}...")
        results = manager.resample_epic(epic, since=since, until=until)

        for tf, count in results.items():
            total_results[tf] += count

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for tf in target_timeframes:
        logger.info(f"  {tf}m: {total_results[tf]:,} candles")

    total = sum(total_results.values())
    logger.info(f"  Total: {total:,} candles")
    logger.info("=" * 50)

    # Verify
    verify_query = """
        SELECT timeframe, COUNT(*) as count
        FROM ig_candles_backtest
        GROUP BY timeframe
        ORDER BY timeframe
    """
    verify_result = db.execute_query(verify_query)
    logger.info("\nTable contents:")
    for _, row in verify_result.iterrows():
        logger.info(f"  {row['timeframe']}m: {row['count']:,} rows")


if __name__ == '__main__':
    main()
