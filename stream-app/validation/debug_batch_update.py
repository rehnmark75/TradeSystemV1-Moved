#!/usr/bin/env python3
"""
Debug script to understand why batch updates aren't persisting
Focuses on the specific timestamp the user reported: 2025-09-02 19:50:00
"""

from datetime import datetime, timezone
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_single_update():
    """Debug updating a single entry with detailed logging"""
    print("🔍 Debug: Single update with detailed logging")
    
    # Target the specific timestamp the user reported
    test_timestamp = datetime(2025, 9, 2, 19, 50, 0)
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    with SessionLocal() as session:
        # Find the entry
        entry = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time == test_timestamp
            )
        ).first()
        
        if not entry:
            print(f"❌ Entry not found for {test_timestamp}")
            return False
            
        print(f"✅ Found entry:")
        print(f"   Timestamp: {entry.start_time}")
        print(f"   Close: {entry.close}")
        print(f"   Data Source: {entry.data_source}")
        
        # Store original values
        original_close = entry.close
        original_source = entry.data_source
        
        # Test values
        test_close = 1.11111  # Obvious test value
        test_source = "debug_batch_update"
        
        print(f"\n🔧 Testing update method 1: Object modification")
        try:
            entry.close = test_close
            entry.data_source = test_source
            entry.updated_at = datetime.now()
            
            print(f"   Before commit - Close: {entry.close}, Source: {entry.data_source}")
            session.commit()
            print(f"   ✅ Method 1 commit successful")
            
            # Verify immediately
            session.refresh(entry)
            print(f"   After refresh - Close: {entry.close}, Source: {entry.data_source}")
            
        except Exception as e:
            print(f"   ❌ Method 1 failed: {e}")
            session.rollback()
            return False
        
        print(f"\n🔧 Testing update method 2: Query update")
        try:
            test_close_2 = 1.22222
            test_source_2 = "debug_query_update"
            
            rows_updated = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.start_time == test_timestamp
                )
            ).update({
                'close': test_close_2,
                'data_source': test_source_2,
                'updated_at': datetime.now()
            })
            
            print(f"   Rows updated: {rows_updated}")
            session.commit()
            print(f"   ✅ Method 2 commit successful")
            
            # Verify immediately
            updated_entry = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.start_time == test_timestamp
                )
            ).first()
            
            if updated_entry:
                print(f"   After query - Close: {updated_entry.close}, Source: {updated_entry.data_source}")
            else:
                print(f"   ❌ Entry disappeared after update!")
                
        except Exception as e:
            print(f"   ❌ Method 2 failed: {e}")
            session.rollback()
            return False
        
        # Restore original values
        print(f"\n🔄 Restoring original values...")
        try:
            session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.start_time == test_timestamp
                )
            ).update({
                'close': original_close,
                'data_source': original_source,
                'updated_at': datetime.now()
            })
            session.commit()
            print(f"   ✅ Restored to: Close={original_close}, Source={original_source}")
            
        except Exception as e:
            print(f"   ❌ Restore failed: {e}")
            
    return True

def debug_batch_update():
    """Debug batch update process similar to the weekly corrector"""
    print("\n🔍 Debug: Batch update process simulation")
    
    # Find some entries to test with
    with SessionLocal() as session:
        # Get a few EURUSD entries from today
        entries = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == "CS.D.EURUSD.CEEM.IP",
                IGCandle.timeframe == 5,
                IGCandle.start_time >= datetime(2025, 9, 2, 19, 0, 0),
                IGCandle.start_time <= datetime(2025, 9, 2, 20, 0, 0)
            )
        ).limit(3).all()
        
        if not entries:
            print("❌ No entries found for batch test")
            return False
            
        print(f"✅ Found {len(entries)} entries for batch test:")
        for entry in entries:
            print(f"   {entry.start_time}: {entry.close} ({entry.data_source})")
        
        # Store original values
        original_values = {}
        for entry in entries:
            original_values[entry.start_time] = {
                'close': entry.close,
                'data_source': entry.data_source
            }
        
        print(f"\n🔧 Testing batch update...")
        try:
            updates_made = 0
            for i, entry in enumerate(entries):
                test_close = 1.33330 + (i * 0.00001)  # Unique test values
                test_source = f"debug_batch_{i}"
                
                print(f"   Updating {entry.start_time}: {entry.close} -> {test_close}")
                
                # Use the same update method as the weekly corrector
                rows_updated = session.query(IGCandle).filter(
                    and_(
                        IGCandle.epic == entry.epic,
                        IGCandle.timeframe == entry.timeframe,
                        IGCandle.start_time == entry.start_time
                    )
                ).update({
                    'close': test_close,
                    'data_source': test_source,
                    'updated_at': datetime.now()
                })
                
                print(f"     Rows updated: {rows_updated}")
                updates_made += rows_updated
            
            print(f"   Total updates before commit: {updates_made}")
            session.commit()
            print(f"   ✅ Batch commit successful")
            
            # Verify all updates
            print(f"\n🔍 Verifying batch updates...")
            for i, original_entry in enumerate(entries):
                updated_entry = session.query(IGCandle).filter(
                    and_(
                        IGCandle.epic == original_entry.epic,
                        IGCandle.timeframe == original_entry.timeframe,
                        IGCandle.start_time == original_entry.start_time
                    )
                ).first()
                
                if updated_entry:
                    expected_close = 1.33330 + (i * 0.00001)
                    expected_source = f"debug_batch_{i}"
                    
                    print(f"   {updated_entry.start_time}: {updated_entry.close} ({updated_entry.data_source})")
                    
                    if (abs(updated_entry.close - expected_close) < 0.000001 and 
                        updated_entry.data_source == expected_source):
                        print(f"     ✅ Update successful")
                    else:
                        print(f"     ❌ Update failed - Expected: {expected_close}/{expected_source}")
                else:
                    print(f"   ❌ Entry disappeared: {original_entry.start_time}")
            
            # Restore original values
            print(f"\n🔄 Restoring original values...")
            for entry_time, original in original_values.items():
                session.query(IGCandle).filter(
                    and_(
                        IGCandle.epic == "CS.D.EURUSD.CEEM.IP",
                        IGCandle.timeframe == 5,
                        IGCandle.start_time == entry_time
                    )
                ).update({
                    'close': original['close'],
                    'data_source': original['data_source'],
                    'updated_at': datetime.now()
                })
            
            session.commit()
            print(f"   ✅ All values restored")
            
        except Exception as e:
            print(f"   ❌ Batch update failed: {e}")
            session.rollback()
            return False
    
    return True

def debug_session_isolation():
    """Test if session isolation is causing issues"""
    print("\n🔍 Debug: Session isolation test")
    
    test_timestamp = datetime(2025, 9, 2, 19, 50, 0)
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    # Session 1: Update
    print("   Opening session 1 for update...")
    with SessionLocal() as session1:
        entry = session1.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time == test_timestamp
            )
        ).first()
        
        if not entry:
            print("   ❌ Entry not found in session 1")
            return False
        
        original_source = entry.data_source
        test_source = "session_isolation_test"
        
        print(f"   Original source: {original_source}")
        print(f"   Updating to: {test_source}")
        
        entry.data_source = test_source
        entry.updated_at = datetime.now()
        session1.commit()
        print("   ✅ Session 1 committed")
    
    # Session 2: Verify
    print("   Opening session 2 for verification...")
    with SessionLocal() as session2:
        entry = session2.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time == test_timestamp
            )
        ).first()
        
        if entry:
            print(f"   Session 2 sees: {entry.data_source}")
            if entry.data_source == test_source:
                print("   ✅ Session isolation working correctly")
                
                # Restore original value
                entry.data_source = original_source
                entry.updated_at = datetime.now()
                session2.commit()
                print("   ✅ Restored original value")
                return True
            else:
                print("   ❌ Session isolation issue detected")
                return False
        else:
            print("   ❌ Entry not found in session 2")
            return False

if __name__ == "__main__":
    print("🧪 Starting database update debugging...")
    
    # Run all debug tests
    results = []
    
    print("\n" + "="*60)
    results.append(("Single Update", debug_single_update()))
    
    print("\n" + "="*60)
    results.append(("Batch Update", debug_batch_update()))
    
    print("\n" + "="*60)
    results.append(("Session Isolation", debug_session_isolation()))
    
    # Summary
    print("\n" + "="*60)
    print("🏁 DEBUG SUMMARY:")
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if all(passed for _, passed in results):
        print("\n✅ All tests passed - Database update mechanism is working correctly")
        print("💡 The issue might be in the weekly corrector's transaction handling or data filtering")
    else:
        print("\n❌ Some tests failed - Database update mechanism has issues")