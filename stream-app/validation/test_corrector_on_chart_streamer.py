#!/usr/bin/env python3
"""
Test the weekly corrector on actual chart_streamer entries
This will help us understand if the corrector is working but just not finding the right entries
"""

from datetime import datetime, timezone, timedelta
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_
import logging

def test_corrector_on_chart_streamer():
    """Test correction on actual chart_streamer entries"""
    print("🧪 Testing corrector on actual chart_streamer entries")
    
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    with SessionLocal() as session:
        # Find recent chart_streamer entries
        chart_streamer_entries = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.data_source == 'chart_streamer'
            )
        ).order_by(IGCandle.start_time.desc()).limit(3).all()
        
        if not chart_streamer_entries:
            print("❌ No chart_streamer entries found")
            return False
        
        print(f"✅ Found {len(chart_streamer_entries)} chart_streamer entries:")
        for entry in chart_streamer_entries:
            print(f"   {entry.start_time}: close={entry.close:.5f}, source={entry.data_source}")
        
        # Store original values for restoration
        original_values = {}
        for entry in chart_streamer_entries:
            original_values[entry.start_time] = {
                'close': entry.close,
                'data_source': entry.data_source
            }
        
        # Test correcting these entries
        corrections_made = 0
        for i, entry in enumerate(chart_streamer_entries):
            # Simulate API returning a corrected price (add +8 pips to simulate the fix)
            corrected_close = entry.close + (8 / 10000)  # Add 8 pips
            difference_pips = abs(corrected_close - entry.close) * 10000
            
            print(f"\n🔧 Processing {entry.start_time}:")
            print(f"   Original: {entry.close:.5f}")
            print(f"   Corrected: {corrected_close:.5f}")
            print(f"   Difference: {difference_pips:.1f} pips")
            
            if difference_pips > 0.1:
                print(f"   ✅ Correction needed - applying update...")
                
                # Use the exact same method as weekly corrector
                update_data = {
                    'close': corrected_close,
                    'data_source': 'api_backfill_fixed',
                    'updated_at': datetime.now()
                }
                
                rows_updated = session.query(IGCandle).filter(
                    and_(
                        IGCandle.epic == epic,
                        IGCandle.timeframe == timeframe,
                        IGCandle.start_time == entry.start_time
                    )
                ).update(update_data)
                
                print(f"   Rows updated: {rows_updated}")
                if rows_updated > 0:
                    corrections_made += 1
                else:
                    print(f"   ❌ Update failed - no rows affected")
            else:
                print(f"   ⏭️ No correction needed")
        
        # Commit like weekly corrector does
        if corrections_made > 0:
            print(f"\n💾 Committing {corrections_made} corrections...")
            try:
                session.commit()
                print(f"✅ Commit successful")
                
                # Verify the commit worked
                sample_check = session.query(IGCandle).filter(
                    IGCandle.data_source == 'api_backfill_fixed'
                ).first()
                
                if sample_check:
                    print(f"✅ Verification: Found updated entry with api_backfill_fixed")
                else:
                    print(f"❌ Verification failed: No api_backfill_fixed entries found")
                
                # Check each corrected entry
                print(f"\n🔍 Verifying individual corrections:")
                success_count = 0
                for entry_time, original in original_values.items():
                    updated_entry = session.query(IGCandle).filter(
                        and_(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time == entry_time
                        )
                    ).first()
                    
                    if updated_entry:
                        expected_close = original['close'] + (8 / 10000)
                        close_match = abs(updated_entry.close - expected_close) < 0.000001
                        source_match = updated_entry.data_source == 'api_backfill_fixed'
                        
                        status = "✅" if (close_match and source_match) else "❌"
                        print(f"   {status} {entry_time}: {updated_entry.close:.5f} ({updated_entry.data_source})")
                        
                        if close_match and source_match:
                            success_count += 1
                    else:
                        print(f"   ❌ Entry missing: {entry_time}")
                
                # Restore original values
                print(f"\n🔄 Restoring original values...")
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
                print(f"✅ Restored {len(original_values)} entries")
                
                return success_count == corrections_made
                
            except Exception as e:
                print(f"❌ Commit failed: {e}")
                session.rollback()
                return False
        else:
            print("No corrections to commit")
            return True

def check_why_corrector_misses_entries():
    """Check if the weekly corrector's time filtering is the issue"""
    print("\n🔍 Checking weekly corrector's time filtering logic")
    
    epic = "CS.D.EURUSD.CEEM.IP" 
    timeframe = 5
    
    # Test the time range the weekly corrector would use
    # Default: 1 week back from today 20:00 UTC
    today = datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)
    one_week_ago = today - timedelta(weeks=1)
    
    print(f"Weekly corrector would check: {one_week_ago} to {today}")
    
    with SessionLocal() as session:
        # Count chart_streamer entries in that range
        entries_in_range = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.data_source == 'chart_streamer',
                IGCandle.start_time >= one_week_ago,
                IGCandle.start_time <= today
            )
        ).count()
        
        print(f"Chart_streamer entries in weekly range: {entries_in_range}")
        
        # Show some recent chart_streamer entries to see their timestamps
        recent_entries = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.data_source == 'chart_streamer'
            )
        ).order_by(IGCandle.start_time.desc()).limit(10).all()
        
        print(f"\nRecent chart_streamer entries (should be in range):")
        for entry in recent_entries:
            in_range = one_week_ago <= entry.start_time <= today
            status = "✅" if in_range else "❌"
            print(f"   {status} {entry.start_time}: {entry.close:.5f}")
        
        return entries_in_range

if __name__ == "__main__":
    print("🧪 Testing weekly corrector on chart_streamer entries...")
    
    results = []
    
    print("\n" + "="*60)
    results.append(("Corrector Test", test_corrector_on_chart_streamer()))
    
    print("\n" + "="*60)
    entries_in_range = check_why_corrector_misses_entries()
    results.append(("Range Check", entries_in_range > 0))
    
    print(f"\n" + "="*60)
    print("🏁 TEST SUMMARY:")
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if all(passed for _, passed in results):
        print("\n✅ Weekly corrector should work on chart_streamer entries")
        print("💡 The issue might be in the API data fetching or chunked processing")
    else:
        print(f"\n❌ Found issues with chart_streamer correction")
        if entries_in_range == 0:
            print("💡 No chart_streamer entries in the default weekly range!")
            print("   This explains why the corrector isn't finding anything to update.")