#!/usr/bin/env python3
"""
Backfill HTF (4H) Candle Direction for Historical Alerts

This script populates htf_candle_direction and htf_candle_direction_prev
for existing alerts that don't have this data.

Usage:
    # Preview without making changes
    docker exec -it task-worker python /app/forex_scanner/scripts/backfill_htf_candle_direction.py --dry-run

    # Execute backfill for last 30 days
    docker exec -it task-worker python /app/forex_scanner/scripts/backfill_htf_candle_direction.py --days 30

    # Execute backfill for all historical alerts
    docker exec -it task-worker python /app/forex_scanner/scripts/backfill_htf_candle_direction.py --days 365
"""
import sys
sys.path.insert(0, '/app')

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import pandas as pd
import argparse


# Database connection
DB_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'forex',
    'user': 'postgres',
    'password': 'postgres'
}


def get_candle_direction(open_price: float, close_price: float) -> str:
    """Determine candle direction based on open/close prices."""
    if close_price > open_price:
        return 'BULLISH'
    elif close_price < open_price:
        return 'BEARISH'
    return 'NEUTRAL'


def get_4h_candles_for_timestamp(conn, epic: str, alert_time: datetime) -> pd.DataFrame:
    """
    Get 4H candles around the alert timestamp.

    ig_candles stores 5m data (timeframe=5), so we need to resample to 4H.
    We fetch 5m candles from ~10 hours before alert to cover at least 2 complete 4H bars.
    """
    # Fetch 5m candles (10 hours gives us buffer for 2+ complete 4H bars)
    start_time = alert_time - timedelta(hours=10)
    end_time = alert_time

    query = """
        SELECT start_time, open, high, low, close, COALESCE(ltv, volume) as volume
        FROM ig_candles
        WHERE epic = %s
          AND timeframe = 5
          AND start_time >= %s
          AND start_time <= %s
        ORDER BY start_time ASC
    """

    df = pd.read_sql_query(query, conn, params=[epic, start_time, end_time])

    if df.empty or len(df) < 48:  # Need at least 48 5m candles for one 4H bar
        return pd.DataFrame()

    # Set index for resampling
    df['start_time'] = pd.to_datetime(df['start_time'])
    df.set_index('start_time', inplace=True)

    # Resample to 4H
    df_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return df_4h


def backfill_alerts(days: int = 30, dry_run: bool = False):
    """Backfill htf_candle_direction for alerts from the last N days."""

    print(f"\n{'='*60}")
    print(f"HTF Candle Direction Backfill")
    print(f"{'='*60}")
    print(f"Days: {days}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will update database)'}")
    print(f"{'='*60}\n")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Get alerts without htf_candle_direction
    cursor.execute("""
        SELECT id, alert_timestamp, epic
        FROM alert_history
        WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
          AND htf_candle_direction IS NULL
        ORDER BY alert_timestamp DESC
    """, [days])

    alerts = cursor.fetchall()
    print(f"Found {len(alerts)} alerts to backfill\n")

    if not alerts:
        print("No alerts need backfilling.")
        cursor.close()
        conn.close()
        return

    updated = 0
    skipped = 0
    errors = 0

    for i, alert in enumerate(alerts):
        alert_id = alert['id']
        epic = alert['epic']
        alert_time = alert['alert_timestamp']

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(alerts)}...")

        # Get 4H candles
        df_4h = get_4h_candles_for_timestamp(conn, epic, alert_time)

        if len(df_4h) < 2:
            print(f"  [SKIP] Alert {alert_id}: insufficient 4H data for {epic} at {alert_time}")
            skipped += 1
            continue

        try:
            # Get last closed candle (iloc[-2] because iloc[-1] may be incomplete)
            last_candle = df_4h.iloc[-2]
            prev_candle = df_4h.iloc[-3] if len(df_4h) >= 3 else df_4h.iloc[-2]

            htf_dir = get_candle_direction(last_candle['open'], last_candle['close'])
            htf_dir_prev = get_candle_direction(prev_candle['open'], prev_candle['close'])

            if dry_run:
                print(f"  [DRY RUN] Alert {alert_id} ({epic[:20]}): {htf_dir} / {htf_dir_prev}")
            else:
                cursor.execute("""
                    UPDATE alert_history
                    SET htf_candle_direction = %s,
                        htf_candle_direction_prev = %s
                    WHERE id = %s
                """, [htf_dir, htf_dir_prev, alert_id])

            updated += 1

        except Exception as e:
            print(f"  [ERROR] Alert {alert_id}: {e}")
            errors += 1

    if not dry_run:
        conn.commit()
        print("\nChanges committed to database.")

    cursor.close()
    conn.close()

    print(f"\n{'='*60}")
    print(f"Backfill Complete")
    print(f"{'='*60}")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print(f"  Total:   {len(alerts)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Backfill HTF candle direction for historical alerts'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to backfill (default: 30)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without updating database'
    )
    args = parser.parse_args()

    backfill_alerts(days=args.days, dry_run=args.dry_run)
