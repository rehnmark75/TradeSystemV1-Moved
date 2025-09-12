#!/usr/bin/env python3
"""
Fix EURUSD CEEM scaling issue in historical data
Divides all prices by 10,000 to match standard forex format
"""

import logging
from datetime import datetime
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_, text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_eurusd_scaling(dry_run=True):
    """
    Fix the scaling issue for CS.D.EURUSD.CEEM.IP data
    All prices need to be divided by 10,000
    """
    epic = "CS.D.EURUSD.CEEM.IP"
    scaling_factor = 10000
    
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Starting EURUSD CEEM scaling fix")
    logger.info(f"Epic: {epic}")
    logger.info(f"Scaling factor: {scaling_factor}")
    
    with SessionLocal() as session:
        # Count affected records
        affected_count = session.query(IGCandle).filter(
            IGCandle.epic == epic
        ).count()
        
        logger.info(f"Found {affected_count} records to fix")
        
        if affected_count == 0:
            logger.info("No records found to fix")
            return
        
        # Get sample of current data
        sample = session.query(IGCandle).filter(
            IGCandle.epic == epic
        ).order_by(IGCandle.start_time.desc()).first()
        
        if sample:
            logger.info(f"Sample before fix - Time: {sample.start_time}, Close: {sample.close}")
            logger.info(f"Expected after fix - Close: {sample.close / scaling_factor:.5f}")
        
        if not dry_run:
            # Use SQL UPDATE for efficiency
            update_query = text("""
                UPDATE ig_candles 
                SET 
                    open = open / :scale,
                    high = high / :scale,
                    low = low / :scale,
                    close = close / :scale,
                    updated_at = NOW()
                WHERE epic = :epic
                AND open > 100  -- Safety check: only fix scaled values
            """)
            
            result = session.execute(update_query, {
                'scale': scaling_factor,
                'epic': epic
            })
            
            rows_updated = result.rowcount
            session.commit()
            
            logger.info(f"✅ Successfully updated {rows_updated} records")
            
            # Verify the fix
            sample_after = session.query(IGCandle).filter(
                IGCandle.epic == epic
            ).order_by(IGCandle.start_time.desc()).first()
            
            if sample_after:
                logger.info(f"Sample after fix - Time: {sample_after.start_time}, Close: {sample_after.close:.5f}")
                
                # Validate the range is reasonable for EURUSD
                if 0.8 < sample_after.close < 1.5:
                    logger.info("✅ Price range looks correct for EURUSD")
                else:
                    logger.warning(f"⚠️ Price {sample_after.close} may still be incorrect for EURUSD")
        else:
            logger.info("DRY RUN: No changes made. Run with --live to apply fixes")
            
            # Show what would be done
            logger.info("\nSQL that would be executed:")
            logger.info("""
                UPDATE ig_candles 
                SET 
                    open = open / 10000,
                    high = high / 10000,
                    low = low / 10000,
                    close = close / 10000,
                    updated_at = NOW()
                WHERE epic = 'CS.D.EURUSD.CEEM.IP'
                AND open > 100
            """)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fix EURUSD CEEM scaling issue')
    parser.add_argument('--live', action='store_true', 
                       help='Apply the fix (default is dry run)')
    args = parser.parse_args()
    
    fix_eurusd_scaling(dry_run=not args.live)

if __name__ == "__main__":
    main()