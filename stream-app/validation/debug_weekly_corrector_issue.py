#!/usr/bin/env python3
"""
Debug script to replicate the exact issue in weekly_close_price_corrector.py
Focus on the specific timestamp and conditions that should trigger an update
"""

from datetime import datetime, timezone, timedelta
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_corrector_conditions():
    """Debug the exact conditions that should trigger a correction"""
    print("🔍 Debug: Replicating weekly corrector conditions")
    
    # Target the specific timestamp the user reported
    test_timestamp = datetime(2025, 9, 2, 19, 50, 0)
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    # Simulate what the weekly corrector would see
    with SessionLocal() as session:
        # Find entries in the time range (like weekly corrector does)
        entries = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time >= test_timestamp - timedelta(minutes=10),
                IGCandle.start_time <= test_timestamp + timedelta(minutes=10)
            )
        ).order_by(IGCandle.start_time).all()
        
        print(f"✅ Found {len(entries)} entries around target time:")
        for entry in entries:
            print(f"   {entry.start_time}: close={entry.close}, source={entry.data_source}")
        
        # Find the specific entry
        target_entry = None
        for entry in entries:
            if entry.start_time == test_timestamp:
                target_entry = entry
                break
        
        if not target_entry:
            print(f"❌ Target entry {test_timestamp} not found")
            return False
        
        print(f"\n🎯 Target entry found:")
        print(f"   Timestamp: {target_entry.start_time}")
        print(f"   Close: {target_entry.close}")
        print(f"   Data Source: {target_entry.data_source}")
        
        # Simulate API price (let's say the API would return a corrected price)
        api_close = 1.10500  # Different from current value
        current_close = target_entry.close
        
        # Calculate pip difference (like the weekly corrector does)
        pip_multiplier = 10000  # For EUR/USD (not JPY pair)
        difference_pips = abs(api_close - current_close) * pip_multiplier
        
        print(f"\n📊 Correction calculation:")
        print(f"   Current close: {current_close}")
        print(f"   API close: {api_close}")
        print(f"   Pip difference: {difference_pips:.1f}")
        print(f"   Threshold: 0.1 pips")
        print(f"   Should correct: {difference_pips > 0.1}")
        
        if difference_pips > 0.1:
            print(f"\n🔧 Applying correction (like weekly corrector)...")
            
            # Store original values
            original_close = target_entry.close
            original_source = target_entry.data_source
            
            try:
                # Use the exact same update method as weekly corrector
                update_data = {
                    'close': api_close,
                    'data_source': 'api_backfill_fixed',
                    'updated_at': datetime.now()
                }
                
                rows_updated = session.query(IGCandle).filter(
                    and_(
                        IGCandle.epic == epic,
                        IGCandle.timeframe == timeframe,
                        IGCandle.start_time == test_timestamp
                    )
                ).update(update_data)
                
                print(f"   Rows updated: {rows_updated}")
                
                if rows_updated > 0:
                    session.commit()
                    print(f"   ✅ Commit successful")
                    
                    # Verify immediately after commit (like weekly corrector does)
                    sample_check = session.query(IGCandle).filter(
                        IGCandle.data_source == 'api_backfill_fixed'
                    ).first()
                    
                    if sample_check:
                        print(f"   ✅ Verification: Found entry with api_backfill_fixed")
                        print(f"      Sample: {sample_check.start_time}, {sample_check.close}")
                    else:
                        print(f"   ❌ Verification failed: No entries with api_backfill_fixed found")
                    
                    # Check the specific entry
                    updated_entry = session.query(IGCandle).filter(
                        and_(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time == test_timestamp
                        )
                    ).first()
                    
                    if updated_entry:
                        print(f"   Specific entry check:")
                        print(f"      Close: {updated_entry.close} (expected: {api_close})")
                        print(f"      Source: {updated_entry.data_source} (expected: api_backfill_fixed)")
                        
                        if (abs(updated_entry.close - api_close) < 0.00001 and
                            updated_entry.data_source == 'api_backfill_fixed'):
                            print(f"      ✅ Update verified successfully")
                            success = True
                        else:
                            print(f"      ❌ Update verification failed")
                            success = False
                    else:
                        print(f"   ❌ Specific entry disappeared after update")
                        success = False
                    
                    # Restore original values for cleanup
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
                    print(f"   🔄 Restored original values")
                    
                    return success
                else:
                    print(f"   ❌ No rows were updated")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Update failed: {e}")
                session.rollback()
                return False
        else:
            print(f"   ⏭️ No correction needed (difference too small)")
            return True

def debug_weekly_corrector_session_handling():
    """Debug session handling like the weekly corrector does it"""
    print("\n🔍 Debug: Weekly corrector session handling simulation")
    
    # This simulates the chunked processing approach of the weekly corrector
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    # Create a session like the weekly corrector does
    with SessionLocal() as session:
        print("   Session created")
        
        # Get some entries to work with
        entries = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time >= datetime(2025, 9, 2, 19, 0, 0),
                IGCandle.start_time <= datetime(2025, 9, 2, 20, 0, 0)
            )
        ).limit(2).all()
        
        if len(entries) < 2:
            print("   ❌ Not enough entries for test")
            return False
        
        print(f"   Found {len(entries)} entries for session test:")
        for entry in entries:
            print(f"      {entry.start_time}: {entry.close} ({entry.data_source})")
        
        # Store original values
        original_values = {}
        for entry in entries:
            original_values[entry.start_time] = {
                'close': entry.close,
                'data_source': entry.data_source
            }
        
        corrections_made = 0
        
        # Simulate the weekly corrector's update loop
        for i, entry in enumerate(entries):
            test_close = 1.44440 + (i * 0.00001)  # Unique test values
            test_source = 'api_backfill_fixed'
            
            print(f"   Processing {entry.start_time}...")
            
            # Calculate pip difference (simulate API returning different value)
            difference_pips = abs(test_close - entry.close) * 10000
            print(f"      Pip difference: {difference_pips:.1f}")
            
            if difference_pips > 0.1:
                print(f"      Applying correction...")
                
                # Use exact same update method as weekly corrector
                update_data = {
                    'close': test_close,
                    'data_source': test_source,
                    'updated_at': datetime.now()
                }
                
                rows_updated = session.query(IGCandle).filter(
                    and_(
                        IGCandle.epic == epic,
                        IGCandle.timeframe == timeframe,
                        IGCandle.start_time == entry.start_time
                    )
                ).update(update_data)
                
                print(f"         Rows updated: {rows_updated}")
                if rows_updated > 0:
                    corrections_made += 1
        
        # Now commit like the weekly corrector does
        if corrections_made > 0:
            print(f"   Committing {corrections_made} corrections...")
            try:
                session.commit()
                print(f"   ✅ Commit successful")
                
                # Verify like weekly corrector does
                sample_check = session.query(IGCandle).filter(
                    IGCandle.data_source == 'api_backfill_fixed'
                ).first()
                
                if sample_check:
                    print(f"   ✅ Verification: Found updated entry")
                    print(f"      Sample: {sample_check.start_time}, source={sample_check.data_source}")
                else:
                    print(f"   ❌ Verification failed: No api_backfill_fixed entries found")
                
                # Check each entry individually
                print("   Checking individual entries:")
                all_updated = True
                for i, original_entry in enumerate(entries):
                    expected_close = 1.44440 + (i * 0.00001)
                    
                    updated_entry = session.query(IGCandle).filter(
                        and_(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time == original_entry.start_time
                        )
                    ).first()
                    
                    if updated_entry:
                        print(f"      {updated_entry.start_time}: {updated_entry.close} ({updated_entry.data_source})")
                        if (abs(updated_entry.close - expected_close) > 0.00001 or 
                            updated_entry.data_source != 'api_backfill_fixed'):
                            all_updated = False
                    else:
                        print(f"      ❌ Entry missing: {original_entry.start_time}")
                        all_updated = False
                
                success = all_updated
                
                # Restore original values
                print("   Restoring original values...")
                for entry_time, original in original_values.items():
                    session.query(IGCandle).filter(
                        and_(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time == entry_time
                        )
                    ).update({
                        'close': original['close'],
                        'data_source': original['data_source'],
                        'updated_at': datetime.now()
                    })
                
                session.commit()
                print("   ✅ Original values restored")
                
                return success
                
            except Exception as e:
                print(f"   ❌ Commit failed: {e}")
                session.rollback()
                return False
        else:
            print("   No corrections to commit")
            return True

if __name__ == "__main__":
    print("🧪 Starting weekly corrector debugging...")
    
    # Run debug tests
    results = []
    
    print("\n" + "="*60)
    results.append(("Corrector Conditions", debug_corrector_conditions()))
    
    print("\n" + "="*60)
    results.append(("Session Handling", debug_weekly_corrector_session_handling()))
    
    # Summary
    print("\n" + "="*60)
    print("🏁 DEBUG SUMMARY:")
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if all(passed for _, passed in results):
        print("\n✅ Weekly corrector pattern works correctly")
        print("💡 The issue must be in the API data fetching or filtering logic")
    else:
        print("\n❌ Found issue in weekly corrector pattern")