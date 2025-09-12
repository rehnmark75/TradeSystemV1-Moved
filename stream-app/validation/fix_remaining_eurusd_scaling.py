#!/usr/bin/env python3
"""
Fix remaining EURUSD entries that still have unscaled prices (> 100)
These are entries that were added after our initial fix or during the backfill
"""

import logging
from datetime import datetime
from services.db import SessionLocal
from sqlalchemy import text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_remaining_scaling(dry_run=True):
    """
    Fix any remaining EURUSD entries with unscaled prices
    """
    epic = "CS.D.EURUSD.CEEM.IP"
    scaling_factor = 10000
    
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Fixing remaining unscaled EURUSD entries")
    logger.info(f"Epic: {epic}")
    logger.info(f"Looking for prices > 100 (should be ~1.17)")
    
    with SessionLocal() as session:
        # First, let's see what we're dealing with
        check_query = text("""
            SELECT COUNT(*) as count,
                   MIN(start_time) as first_time,
                   MAX(start_time) as last_time,
                   MIN(close) as min_close,
                   MAX(close) as max_close
            FROM ig_candles 
            WHERE epic = :epic 
            AND close > 100
            AND start_time >= '2025-09-08'
        """)
        
        result = session.execute(check_query, {'epic': epic}).fetchone()
        
        if result and result.count > 0:
            logger.info(f"Found {result.count} entries to fix")
            logger.info(f"Date range: {result.first_time} to {result.last_time}")
            logger.info(f"Price range: {result.min_close:.2f} to {result.max_close:.2f}")
            logger.info(f"Expected after fix: {result.min_close/scaling_factor:.5f} to {result.max_close/scaling_factor:.5f}")
            
            if not dry_run:
                # Apply the fix
                update_query = text("""
                    UPDATE ig_candles 
                    SET 
                        open = open / :scale,
                        high = high / :scale,
                        low = low / :scale,
                        close = close / :scale,
                        updated_at = NOW()
                    WHERE epic = :epic
                    AND close > 100
                    AND start_time >= '2025-09-08'
                """)
                
                result = session.execute(update_query, {
                    'scale': scaling_factor,
                    'epic': epic
                })
                
                rows_updated = result.rowcount
                session.commit()
                
                logger.info(f"✅ Successfully updated {rows_updated} records")
                
                # Verify the fix
                verify_query = text("""
                    SELECT COUNT(*) as remaining
                    FROM ig_candles 
                    WHERE epic = :epic 
                    AND close > 100
                """)
                
                remaining = session.execute(verify_query, {'epic': epic}).scalar()
                
                if remaining == 0:
                    logger.info("✅ All EURUSD prices are now correctly scaled")
                    
                    # Show a sample of the fixed data
                    sample_query = text("""
                        SELECT start_time, open, high, low, close
                        FROM ig_candles
                        WHERE epic = :epic
                        AND start_time >= '2025-09-08'
                        ORDER BY start_time DESC
                        LIMIT 5
                    """)
                    
                    samples = session.execute(sample_query, {'epic': epic}).fetchall()
                    logger.info("\nSample of fixed data:")
                    for sample in samples:
                        logger.info(f"  {sample.start_time}: O={sample.open:.5f} H={sample.high:.5f} L={sample.low:.5f} C={sample.close:.5f}")
                else:
                    logger.warning(f"⚠️ {remaining} entries still have prices > 100")
            else:
                logger.info("\nDRY RUN mode - no changes made")
                logger.info("Run with --live to apply the fix")
                
                # Show what would be affected
                sample_query = text("""
                    SELECT start_time, open, high, low, close, data_source
                    FROM ig_candles
                    WHERE epic = :epic
                    AND close > 100
                    AND start_time >= '2025-09-08'
                    ORDER BY start_time
                    LIMIT 5
                """)
                
                samples = session.execute(sample_query, {'epic': epic}).fetchall()
                logger.info("\nSample entries that would be fixed:")
                for sample in samples:
                    logger.info(f"  {sample.start_time}: C={sample.close:.1f} -> {sample.close/scaling_factor:.5f} (source: {sample.data_source})")
        else:
            logger.info("✅ No unscaled entries found - all EURUSD prices are correct!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fix remaining unscaled EURUSD prices')
    parser.add_argument('--live', action='store_true', 
                       help='Apply the fix (default is dry run)')
    args = parser.parse_args()
    
    fix_remaining_scaling(dry_run=not args.live)

if __name__ == "__main__":
    main()