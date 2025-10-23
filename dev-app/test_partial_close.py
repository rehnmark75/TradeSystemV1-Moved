#!/usr/bin/env python3
"""
Test script to execute partial close on trade 1340
"""
import asyncio
import sys
from services.ig_orders import partial_close_position
from dependencies import get_ig_auth_headers
from services.models import TradeLog
from services.db import SessionLocal
from datetime import datetime

async def test_partial_close():
    """Test partial close on trade 1340"""

    # Get trade from database
    db = SessionLocal()
    try:
        trade = db.query(TradeLog).filter(TradeLog.id == 1340).first()

        if not trade:
            print("❌ Trade 1340 not found in database")
            return False

        print(f"📊 Trade Details:")
        print(f"   ID: {trade.id}")
        print(f"   Symbol: {trade.symbol}")
        print(f"   Deal ID: {trade.deal_id}")
        print(f"   Direction: {trade.direction}")
        print(f"   Current Size: {trade.current_size or 1.0}")
        print(f"   Partial Close Executed: {trade.partial_close_executed}")
        print()

        if trade.partial_close_executed:
            print("⚠️ Partial close already executed on this trade")
            return False

        # Get auth headers
        print("🔐 Getting authentication headers...")
        auth_headers = await get_ig_auth_headers()
        print("✅ Auth headers obtained")
        print()

        # Execute partial close
        print(f"📤 Attempting to close 0.5 of position (50%)...")
        print(f"   Deal ID: {trade.deal_id}")
        print(f"   Epic: {trade.symbol}")
        print(f"   Direction: {trade.direction}")
        print()

        result = await partial_close_position(
            deal_id=trade.deal_id,
            epic=trade.symbol,
            direction=trade.direction,
            size_to_close=0.5,
            auth_headers=auth_headers
        )

        print("📥 Result from IG API:")
        print(f"   Success: {result.get('success')}")

        if result.get('success'):
            print(f"   ✅ Size Closed: {result.get('size_closed')}")
            print(f"   Response: {result.get('response')}")
            print()

            # Update database
            print("💾 Updating database...")
            trade.current_size = 0.5
            trade.partial_close_executed = True
            trade.partial_close_time = datetime.utcnow()
            trade.status = "partial_closed"

            db.commit()
            print("✅ Database updated successfully")
            print()

            # Verify
            db.refresh(trade)
            print("📊 Updated Trade Details:")
            print(f"   Current Size: {trade.current_size}")
            print(f"   Partial Close Executed: {trade.partial_close_executed}")
            print(f"   Partial Close Time: {trade.partial_close_time}")
            print(f"   Status: {trade.status}")

            return True
        else:
            print(f"   ❌ Error: {result.get('error')}")
            return False

    except Exception as e:
        print(f"❌ Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 PARTIAL CLOSE TEST - Trade 1340")
    print("=" * 60)
    print()

    success = asyncio.run(test_partial_close())

    print()
    print("=" * 60)
    if success:
        print("✅ TEST PASSED - Partial close executed successfully!")
    else:
        print("❌ TEST FAILED - See errors above")
    print("=" * 60)

    sys.exit(0 if success else 1)
