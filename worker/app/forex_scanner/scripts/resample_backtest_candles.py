#!/usr/bin/env python3
"""
Resample 1m candles into 5m, 15m, and 4h for the backtest table.

This script populates ig_candles_backtest with pre-computed candles
for fast backtesting without runtime resampling.

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

import pandas as pd

# Setup path for imports
sys.path.insert(0, '/app')

from forex_scanner.core.database import DatabaseManager
from forex_scanner import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resample_ohlcv(df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    """
    Resample OHLCV data to a larger timeframe.

    Args:
        df: DataFrame with OHLCV columns and datetime index
        target_minutes: Target timeframe in minutes (5, 15, 240)

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'start_time' in df.columns:
            df = df.set_index('start_time')
        else:
            raise ValueError("DataFrame must have datetime index or start_time column")

    # Sort by time
    df = df.sort_index()

    # Resample rule
    rule = f'{target_minutes}min'

    # OHLCV aggregation
    resampled = df.resample(rule, closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'ltv': 'sum'
    }).dropna()

    return resampled


def get_epics(db: DatabaseManager) -> List[str]:
    """Get all unique epics from the 1m candles table."""
    query = "SELECT DISTINCT epic FROM ig_candles WHERE timeframe = 1 ORDER BY epic"
    result = db.execute_query(query)
    return result['epic'].tolist() if not result.empty else []


def resample_epic(
    db: DatabaseManager,
    epic: str,
    target_timeframes: List[int],
    since: Optional[datetime] = None,
    batch_size: int = 100000
) -> dict:
    """
    Resample 1m candles for a single epic into target timeframes.

    Args:
        db: Database manager
        epic: Epic to resample
        target_timeframes: List of target timeframes in minutes (e.g., [5, 15, 240])
        since: Only process data since this time (for incremental updates)
        batch_size: Number of 1m candles to process at once

    Returns:
        Dict with counts of inserted rows per timeframe
    """
    results = {tf: 0 for tf in target_timeframes}

    # Build query for 1m data
    if since:
        query = """
            SELECT start_time, open, high, low, close, volume, COALESCE(ltv, 0) as ltv
            FROM ig_candles
            WHERE epic = :epic AND timeframe = 1 AND start_time >= :since
            ORDER BY start_time
        """
        params = {'epic': epic, 'since': since}
    else:
        query = """
            SELECT start_time, open, high, low, close, volume, COALESCE(ltv, 0) as ltv
            FROM ig_candles
            WHERE epic = :epic AND timeframe = 1
            ORDER BY start_time
        """
        params = {'epic': epic}

    logger.info(f"  Fetching 1m data for {epic}...")
    df_1m = db.execute_query(query, params)

    if df_1m.empty:
        logger.warning(f"  No 1m data found for {epic}")
        return results

    logger.info(f"  Found {len(df_1m):,} 1m candles")

    # Set index for resampling
    df_1m['start_time'] = pd.to_datetime(df_1m['start_time'])
    df_1m = df_1m.set_index('start_time')

    # Resample to each target timeframe
    for tf_minutes in target_timeframes:
        logger.info(f"  Resampling to {tf_minutes}m...")

        df_resampled = resample_ohlcv(df_1m, tf_minutes)

        if df_resampled.empty:
            logger.warning(f"  No data after resampling to {tf_minutes}m")
            continue

        # Prepare for insert
        df_resampled = df_resampled.reset_index()
        df_resampled['epic'] = epic
        df_resampled['timeframe'] = tf_minutes
        df_resampled['resampled_from'] = 1

        # Delete existing data for this epic/timeframe (upsert)
        if since:
            delete_query = """
                DELETE FROM ig_candles_backtest
                WHERE epic = :epic AND timeframe = :tf AND start_time >= :since
            """
            db.execute_query(delete_query, {'epic': epic, 'tf': tf_minutes, 'since': since})
        else:
            delete_query = """
                DELETE FROM ig_candles_backtest
                WHERE epic = :epic AND timeframe = :tf
            """
            db.execute_query(delete_query, {'epic': epic, 'tf': tf_minutes})

        # Insert in batches
        inserted = 0
        for i in range(0, len(df_resampled), batch_size):
            batch = df_resampled.iloc[i:i + batch_size]

            # Build insert query
            values = []
            for _, row in batch.iterrows():
                values.append(f"""(
                    '{row['start_time']}',
                    '{epic}',
                    {tf_minutes},
                    {row['open']},
                    {row['high']},
                    {row['low']},
                    {row['close']},
                    {int(row['volume'])},
                    {int(row['ltv'])},
                    1
                )""")

            if values:
                insert_query = f"""
                    INSERT INTO ig_candles_backtest
                    (start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from)
                    VALUES {', '.join(values)}
                    ON CONFLICT (start_time, epic, timeframe) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        ltv = EXCLUDED.ltv
                """
                db.execute_query(insert_query)
                inserted += len(batch)

        results[tf_minutes] = inserted
        logger.info(f"  Inserted {inserted:,} {tf_minutes}m candles")

    return results


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

    # Process each epic
    total_results = {tf: 0 for tf in target_timeframes}

    for epic in epics:
        logger.info(f"\nProcessing {epic}...")
        results = resample_epic(db, epic, target_timeframes, since)

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
