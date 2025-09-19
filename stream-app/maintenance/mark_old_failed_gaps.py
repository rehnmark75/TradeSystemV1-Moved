#!/usr/bin/env python3
"""
Maintenance script to mark old failed gaps as permanently failed
to prevent them from being repeatedly reported.

Usage:
    python3 mark_old_failed_gaps.py [--dry-run] [--days-old=7]
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, timezone

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.db import SessionLocal
from services.models import FailedGap
from igstream.gap_detector import GapDetector
from igstream.auto_backfill import AutoBackfillService

def find_and_mark_failed_gaps(dry_run=True, days_old=7, max_attempts=3):
    """
    Find gaps that have been attempted multiple times and mark them as failed

    Args:
        dry_run: If True, only print what would be done
        days_old: Only consider gaps older than this many days
        max_attempts: Consider gaps failed if attempted this many times
    """

    print(f"🔍 Scanning for gaps that should be marked as failed...")
    print(f"   Criteria: Older than {days_old} days OR attempted {max_attempts}+ times")

    # Major epics to check
    epics = [
        "CS.D.EURUSD.CEEM.IP",
        "CS.D.GBPUSD.MINI.IP",
        "CS.D.USDJPY.MINI.IP",
        "CS.D.AUDUSD.MINI.IP",
        "CS.D.USDCAD.MINI.IP",
        "CS.D.EURJPY.MINI.IP",
        "CS.D.AUDJPY.MINI.IP",
        "CS.D.NZDUSD.MINI.IP",
        "CS.D.USDCHF.MINI.IP"
    ]

    detector = GapDetector()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)

    total_gaps_found = 0
    gaps_to_mark = []

    # Check all epics for gaps
    for epic in epics:
        print(f"\n📊 Checking {epic}...")

        # Get gaps for the last 30 days to catch old ones
        gaps = detector.detect_gaps(epic, 5, start_time=cutoff_date - timedelta(days=30))

        for gap in gaps:
            gap_age_days = (datetime.now(timezone.utc) - gap['gap_start']).days

            # Mark old gaps or gaps that would likely fail
            should_mark = False
            reason = ""

            if gap_age_days > days_old:
                should_mark = True
                reason = f"old_gap_age_{gap_age_days}_days"
            elif gap['missing_candles'] == 1 and gap_age_days > 1:
                should_mark = True
                reason = "single_candle_gap_likely_no_data"

            if should_mark:
                gaps_to_mark.append({
                    'gap': gap,
                    'reason': reason,
                    'age_days': gap_age_days
                })
                total_gaps_found += 1

    print(f"\n📋 Summary: Found {total_gaps_found} gaps to mark as failed")

    if total_gaps_found == 0:
        print("✅ No gaps need to be marked as failed")
        return

    # Show what would be marked
    for item in gaps_to_mark[:10]:  # Show first 10
        gap = item['gap']
        print(f"   • {gap['epic']} {gap['timeframe']}m at {gap['gap_start']} "
              f"({item['age_days']} days old, {gap['missing_candles']} candles) - {item['reason']}")

    if len(gaps_to_mark) > 10:
        print(f"   ... and {len(gaps_to_mark) - 10} more")

    if dry_run:
        print(f"\n🔶 DRY RUN MODE - No changes made")
        print(f"   Run with --apply to actually mark these gaps as failed")
        return

    # Actually mark the gaps
    print(f"\n💾 Marking {len(gaps_to_mark)} gaps as failed...")

    with SessionLocal() as session:
        marked_count = 0

        for item in gaps_to_mark:
            gap = item['gap']
            reason = item['reason']

            try:
                # Check if already exists
                existing = session.query(FailedGap).filter(
                    FailedGap.epic == gap['epic'],
                    FailedGap.timeframe == gap['timeframe'],
                    FailedGap.gap_start == gap['gap_start']
                ).first()

                if existing:
                    print(f"   ⚠️  Gap already marked: {gap['epic']} at {gap['gap_start']}")
                    continue

                # Create new failed gap record
                failed_gap = FailedGap(
                    epic=gap['epic'],
                    timeframe=gap['timeframe'],
                    gap_start=gap['gap_start'],
                    gap_end=gap['gap_end'],
                    failure_reason=reason,
                    first_failed_at=datetime.now(timezone.utc),
                    last_attempted_at=datetime.now(timezone.utc),
                    attempt_count=1,
                    missing_candles=gap.get('missing_candles'),
                    gap_duration_minutes=gap.get('gap_duration_minutes')
                )

                session.add(failed_gap)
                marked_count += 1

            except Exception as e:
                print(f"   ❌ Error marking gap {gap['epic']} at {gap['gap_start']}: {e}")

        try:
            session.commit()
            print(f"✅ Successfully marked {marked_count} gaps as failed")
        except Exception as e:
            print(f"❌ Error committing changes: {e}")
            session.rollback()

def main():
    parser = argparse.ArgumentParser(description='Mark old failed gaps to prevent repeated reporting')
    parser.add_argument('--apply', action='store_true', help='Actually mark gaps (default is dry-run)')
    parser.add_argument('--days-old', type=int, default=7, help='Mark gaps older than this many days')
    parser.add_argument('--max-attempts', type=int, default=3, help='Mark gaps attempted this many times')

    args = parser.parse_args()

    print("🛠️  Failed Gap Maintenance Tool")
    print("=" * 50)

    find_and_mark_failed_gaps(
        dry_run=not args.apply,
        days_old=args.days_old,
        max_attempts=args.max_attempts
    )

if __name__ == "__main__":
    main()