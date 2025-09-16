#!/usr/bin/env python3
"""
Fix Future Timestamp Corruption
Corrects trades with closed_at timestamps in the future that are breaking the cooldown system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from services.db import get_db_session
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_future_timestamps():
    """Fix trades with future closed_at timestamps"""

    with get_db_session() as db:
        # Find trades with future timestamps
        future_trades = db.execute(text("""
            SELECT
                id,
                symbol,
                timestamp as entry_time,
                closed_at,
                EXTRACT(EPOCH FROM (closed_at - NOW()))/60 as minutes_in_future
            FROM trade_log
            WHERE closed_at > NOW()
            ORDER BY closed_at DESC
        """)).fetchall()

        if not future_trades:
            logger.info("✅ No trades with future timestamps found")
            return

        logger.warning(f"🚨 Found {len(future_trades)} trades with future timestamps:")

        for trade in future_trades:
            logger.warning(f"   ID {trade.id} ({trade.symbol}): {trade.closed_at} ({trade.minutes_in_future:.1f} min in future)")

        # Strategy: Use entry_time as closed_at for these corrupted trades
        # This is a reasonable approximation since these are likely old trades

        logger.info("🔧 Fixing timestamps by using reasonable approximations...")

        fixed_count = 0
        for trade in future_trades:
            # For trades that have been "open" for more than 24 hours,
            # set closed_at to entry_time + 2 hours (reasonable trade duration)
            entry_time = trade.entry_time

            # If entry was days ago, assume trade closed after 2 hours
            if (datetime.utcnow() - entry_time).days > 0:
                estimated_close = entry_time.replace(hour=min(entry_time.hour + 2, 23))
            else:
                # If entry was today, assume closed 30 minutes ago
                estimated_close = datetime.utcnow().replace(minute=max(datetime.utcnow().minute - 30, 0))

            db.execute(text("""
                UPDATE trade_log
                SET closed_at = :estimated_close,
                    updated_at = NOW()
                WHERE id = :trade_id
            """), {
                "estimated_close": estimated_close,
                "trade_id": trade.id
            })

            logger.info(f"✅ Fixed trade {trade.id} ({trade.symbol}): {trade.closed_at} → {estimated_close}")
            fixed_count += 1

        db.commit()
        logger.info(f"🎉 Successfully fixed {fixed_count} trades with future timestamps")

        # Verify the fix worked
        remaining_future = db.execute(text("""
            SELECT COUNT(*) as count
            FROM trade_log
            WHERE closed_at > NOW()
        """)).fetchone()

        if remaining_future.count == 0:
            logger.info("✅ All future timestamps have been corrected!")
        else:
            logger.warning(f"⚠️ {remaining_future.count} trades still have future timestamps")

def verify_cooldown_system():
    """Verify that the cooldown system is now working correctly"""
    logger.info("🧪 Testing cooldown system after timestamp fix...")

    try:
        from routers.orders_router import check_trade_cooldown

        with get_db_session() as db:
            # Test GBP/USD cooldown
            result = check_trade_cooldown('CS.D.GBPUSD.MINI.IP', db)
            logger.info(f"📊 GBP/USD cooldown test:")
            logger.info(f"   Allowed: {result['allowed']}")
            logger.info(f"   Message: {result['message']}")

            # Test a few other pairs
            for epic in ['CS.D.EURUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP']:
                result = check_trade_cooldown(epic, db)
                logger.info(f"📊 {epic}: {'✅ Allowed' if result['allowed'] else '🛑 Blocked'}")

    except Exception as e:
        logger.error(f"❌ Error testing cooldown system: {e}")

if __name__ == "__main__":
    logger.info("🔧 Starting future timestamp corruption fix...")
    fix_future_timestamps()
    verify_cooldown_system()
    logger.info("✅ Timestamp fix completed!")