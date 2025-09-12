#!/usr/bin/env python3
"""
Analyze why the weekly close price corrector missed the specific corrupted entries
"""

from datetime import datetime, timezone, timedelta
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_
import logging

def analyze_why_missed():
    """Analyze why the weekly corrector didn't catch these specific entries"""
    print("🔍 Analyzing why weekly corrector missed the corrupted entries")
    
    # The specific timestamps that were corrupted
    corrupted_timestamps = [
        datetime(2025, 9, 1, 2, 40, 0),
        datetime(2025, 9, 1, 2, 45, 0),
        datetime(2025, 9, 1, 2, 50, 0),
        datetime(2025, 9, 1, 2, 55, 0)
    ]
    
    print(f"📅 Corrupted timestamps we just fixed:")
    for ts in corrupted_timestamps:
        print(f"   {ts}")
    
    # Check weekly corrector's default time range
    print(f"\n🔍 Checking weekly corrector's time range logic:")
    
    # Default weekly corrector logic: 1 week back from today 20:00 UTC
    today = datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)
    one_week_ago = today - timedelta(weeks=1)
    
    print(f"   Weekly corrector range: {one_week_ago} to {today}")
    
    # Check if the corrupted timestamps fall within the weekly range
    print(f"\n📊 Timestamp coverage analysis:")
    for ts in corrupted_timestamps:
        in_range = one_week_ago <= ts <= today
        status = "✅ COVERED" if in_range else "❌ MISSED"
        hours_diff = (today - ts).total_seconds() / 3600
        print(f"   {ts}: {status} ({hours_diff:.1f} hours ago)")
    
    # Check the chunked processing logic
    print(f"\n🔍 Checking chunked processing coverage:")
    
    # Weekly corrector uses 2-hour chunks
    chunk_size_hours = 2
    
    # Find which chunk these timestamps would fall into
    for ts in corrupted_timestamps:
        # Calculate which chunk this timestamp would be in
        hours_from_start = (ts - one_week_ago).total_seconds() / 3600
        chunk_number = int(hours_from_start // chunk_size_hours)
        chunk_start = one_week_ago + timedelta(hours=chunk_number * chunk_size_hours)
        chunk_end = chunk_start + timedelta(hours=chunk_size_hours)
        
        print(f"   {ts}:")
        print(f"      Would be in chunk {chunk_number}: {chunk_start} to {chunk_end}")
    
    # Check what the database looked like when we ran the corrector
    print(f"\n🔍 Checking database state when corrector was run:")
    
    with SessionLocal() as session:
        # Check current state of these timestamps
        for ts in corrupted_timestamps:
            entry = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == "CS.D.EURUSD.CEEM.IP",
                    IGCandle.timeframe == 5,
                    IGCandle.start_time == ts
                )
            ).first()
            
            if entry:
                print(f"   {ts}: {entry.close:.5f} ({entry.data_source}) - NOW CORRECTED")
            else:
                print(f"   {ts}: NOT FOUND")
    
    # Possible reasons analysis
    print(f"\n💡 POSSIBLE REASONS WHY WEEKLY CORRECTOR MISSED THESE:")
    
    # Reason 1: Time range
    any_outside_range = any(not (one_week_ago <= ts <= today) for ts in corrupted_timestamps)
    if any_outside_range:
        print("   1. ❌ TIMESTAMP OUTSIDE RANGE: Some entries were outside the 1-week window")
    else:
        print("   1. ✅ All timestamps were within the weekly range")
    
    # Reason 2: Data source filtering
    print("   2. 🔍 DATA SOURCE FILTERING:")
    print("      - Weekly corrector processes ALL entries in time range")
    print("      - It doesn't filter by data_source when finding entries to process")
    print("      - So data_source shouldn't be the issue")
    
    # Reason 3: API data availability
    print("   3. 🔍 API DATA AVAILABILITY:")
    print("      - Weekly corrector fetches API data for 2-hour chunks")
    print("      - If API didn't return data for these timestamps, no comparison would happen")
    print("      - Our manual fix showed API data WAS available")
    
    # Reason 4: Chunk processing logic
    print("   4. 🔍 CHUNK PROCESSING LOGIC:")
    print("      - Weekly corrector processes 2-hour chunks sequentially")
    print("      - If a chunk had no entries, it would skip API call")
    print("      - Let's check if these entries existed when corrector ran...")
    
    # Reason 5: Timing of when entries were created
    print("   5. 🔍 ENTRY CREATION TIMING:")
    print("      - These entries might have been created AFTER the weekly corrector ran")
    print("      - Stream data comes in real-time, corrector runs on historical data")
    print("      - If entries were created after the corrector's time window, they'd be missed")
    
    return True

def check_entry_creation_timing():
    """Check when these entries were actually created"""
    print(f"\n🔍 Checking entry creation timing...")
    
    corrupted_timestamps = [
        datetime(2025, 9, 1, 2, 40, 0),
        datetime(2025, 9, 1, 2, 45, 0),
        datetime(2025, 9, 1, 2, 50, 0),
        datetime(2025, 9, 1, 2, 55, 0)
    ]
    
    with SessionLocal() as session:
        print("📊 Entry creation analysis:")
        for ts in corrupted_timestamps:
            entry = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == "CS.D.EURUSD.CEEM.IP",
                    IGCandle.timeframe == 5,
                    IGCandle.start_time == ts
                )
            ).first()
            
            if entry:
                created_at = entry.created_at if entry.created_at else "Unknown"
                updated_at = entry.updated_at if entry.updated_at else "Unknown"
                
                print(f"   {ts}:")
                print(f"      Created: {created_at}")
                print(f"      Updated: {updated_at}")
                print(f"      Current source: {entry.data_source}")
                
                # Check if created_at is after the entry's start_time (indicating late creation)
                if entry.created_at and entry.created_at > ts + timedelta(hours=1):
                    delay_hours = (entry.created_at - ts).total_seconds() / 3600
                    print(f"      ⚠️ LATE CREATION: Entry created {delay_hours:.1f} hours after timestamp")
                else:
                    print(f"      ✅ Timely creation")

def check_weekly_corrector_logs():
    """Analyze what the weekly corrector would have seen"""
    print(f"\n🔍 Simulating what weekly corrector would have seen...")
    
    # Simulate the weekly corrector's chunk processing for the specific time range
    target_time = datetime(2025, 9, 1, 2, 40, 0)  # First corrupted entry
    chunk_start = datetime(2025, 9, 1, 2, 0, 0)   # 2-hour chunk: 02:00-04:00
    chunk_end = datetime(2025, 9, 1, 4, 0, 0)
    
    print(f"📊 Simulating chunk: {chunk_start} to {chunk_end}")
    
    with SessionLocal() as session:
        # Count entries in this chunk (like weekly corrector does)
        entries_in_chunk = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == "CS.D.EURUSD.CEEM.IP",
                IGCandle.timeframe == 5,
                IGCandle.start_time >= chunk_start,
                IGCandle.start_time < chunk_end
            )
        ).all()
        
        print(f"✅ Found {len(entries_in_chunk)} entries in chunk:")
        
        chart_streamer_count = 0
        for entry in entries_in_chunk:
            print(f"   {entry.start_time}: {entry.close:.5f} ({entry.data_source})")
            if entry.data_source == 'chart_streamer':
                chart_streamer_count += 1
        
        print(f"\n📊 Chart_streamer entries in chunk: {chart_streamer_count}")
        
        if len(entries_in_chunk) == 0:
            print("💡 FOUND THE ISSUE: Weekly corrector would see NO entries in this chunk!")
            print("   - If chunk is empty, corrector skips API call")
            print("   - No API call = no comparison = no corrections")
            return "empty_chunk"
        elif chart_streamer_count == 0:
            print("💡 POSSIBLE ISSUE: No chart_streamer entries to compare")
            print("   - Weekly corrector might have already processed this data")
            return "already_processed"
        else:
            print("💡 ENTRIES FOUND: Weekly corrector should have processed this chunk")
            print("   - There might be another reason why corrections weren't applied")
            return "should_have_processed"

if __name__ == "__main__":
    print("🧪 Analyzing why weekly corrector missed corrupted entries...")
    
    analyze_why_missed()
    check_entry_creation_timing()
    result = check_weekly_corrector_logs()
    
    print(f"\n🏁 ANALYSIS SUMMARY:")
    if result == "empty_chunk":
        print("   🎯 LIKELY REASON: Weekly corrector saw empty chunk, skipped API call")
        print("   💡 SOLUTION: Entries were created after corrector ran, or chunk logic issue")
    elif result == "already_processed":
        print("   🎯 LIKELY REASON: Data was already corrected when corrector ran")
        print("   💡 SOLUTION: These entries became corrupted after corrector ran")
    else:
        print("   🎯 UNCLEAR: Weekly corrector should have found and processed these entries")
        print("   💡 SOLUTION: Might be API timing, comparison logic, or processing order issue")