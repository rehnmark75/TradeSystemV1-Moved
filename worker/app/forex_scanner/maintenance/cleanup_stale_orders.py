#!/usr/bin/env python3
"""
Cleanup script for stale pending orders in alert_history.

Orders that remain in 'pending' status for more than 30 minutes are likely expired
and should be marked as such to prevent blocking new signals due to cooldown logic.

This script can be run:
- Manually: python cleanup_stale_orders.py
- Via cron: */15 * * * * docker exec task-worker python /app/forex_scanner/maintenance/cleanup_stale_orders.py
- Via systemd timer

Author: Trading System Maintenance
Date: 2026-01-21
"""

import sys
import os
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config import DATABASE_URL
except ImportError:
    # Fallback for when running outside normal context
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/forex')


def cleanup_stale_pending_orders(max_age_minutes: int = 30, dry_run: bool = False):
    """
    Clean up pending orders that are older than max_age_minutes.

    Args:
        max_age_minutes: How old a pending order must be before expiring (default: 30)
        dry_run: If True, only show what would be updated without making changes

    Returns:
        Number of orders updated
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)

        # Find stale pending orders
        cursor.execute("""
            SELECT
                id,
                alert_timestamp,
                epic,
                signal_type,
                order_status,
                EXTRACT(EPOCH FROM (NOW() - alert_timestamp))/60 as minutes_ago
            FROM alert_history
            WHERE order_status = 'pending'
              AND alert_timestamp < %s
            ORDER BY alert_timestamp DESC
        """, (cutoff_time,))

        stale_orders = cursor.fetchall()

        if not stale_orders:
            print(f"✅ No stale pending orders found (threshold: {max_age_minutes} minutes)")
            cursor.close()
            conn.close()
            return 0

        print(f"\n🔍 Found {len(stale_orders)} stale pending orders:")
        print(f"{'='*80}")
        for order in stale_orders:
            print(f"  ID: {order['id']:6d} | {order['alert_timestamp']} | {order['epic']:20s} | "
                  f"{order['signal_type']:4s} | Age: {order['minutes_ago']:.1f} min")

        if dry_run:
            print(f"\n🔍 DRY RUN: Would update {len(stale_orders)} orders to 'expired' status")
            cursor.close()
            conn.close()
            return len(stale_orders)

        # Update stale orders to expired
        cursor.execute("""
            UPDATE alert_history
            SET order_status = 'expired'
            WHERE order_status = 'pending'
              AND alert_timestamp < %s
        """, (cutoff_time,))

        updated_count = cursor.rowcount
        conn.commit()

        print(f"\n✅ Updated {updated_count} pending orders to 'expired' status")

        cursor.close()
        conn.close()

        return updated_count

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return 0


def cleanup_all_pairs_stale_orders(max_age_minutes: int = 30):
    """
    Show summary of stale orders by pair and clean them up.

    Args:
        max_age_minutes: How old a pending order must be before expiring (default: 30)

    Returns:
        Number of orders updated
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)

        # Get summary by pair
        cursor.execute("""
            SELECT
                epic,
                COUNT(*) as stale_count,
                MAX(alert_timestamp) as most_recent,
                MIN(alert_timestamp) as oldest
            FROM alert_history
            WHERE order_status = 'pending'
              AND alert_timestamp < %s
            GROUP BY epic
            ORDER BY stale_count DESC
        """, (cutoff_time,))

        summary = cursor.fetchall()

        if summary:
            print(f"\n📊 Stale pending orders by pair (>{max_age_minutes} min old):")
            print(f"{'='*80}")
            print(f"{'Epic':<25s} | {'Count':>5s} | {'Most Recent':<20s} | {'Oldest':<20s}")
            print(f"{'-'*80}")
            for row in summary:
                print(f"{row['epic']:<25s} | {row['stale_count']:>5d} | "
                      f"{str(row['most_recent']):<20s} | {str(row['oldest']):<20s}")

        cursor.close()
        conn.close()

        # Now perform the cleanup
        return cleanup_stale_pending_orders(max_age_minutes=max_age_minutes, dry_run=False)

    except Exception as e:
        print(f"❌ Error during summary: {e}")
        return 0


if __name__ == "__main__":
    print(f"🧹 Starting cleanup of stale pending orders...")
    print(f"⏰ Timestamp: {datetime.now()}")

    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    max_age = 30

    for arg in sys.argv[1:]:
        if arg.startswith('--max-age='):
            max_age = int(arg.split('=')[1])

    if dry_run:
        print(f"🔍 DRY RUN MODE - No changes will be made")

    print(f"⏱️  Max age: {max_age} minutes\n")

    updated = cleanup_all_pairs_stale_orders(max_age_minutes=max_age)

    print(f"\n{'='*80}")
    print(f"✅ Cleanup complete: {updated} orders updated to 'expired' status")
    print(f"⏰ Completed at: {datetime.now()}")

    sys.exit(0)
