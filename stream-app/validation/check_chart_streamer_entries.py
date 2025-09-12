#!/usr/bin/env python3
"""
Check what entries actually have 'chart_streamer' as data_source
Focus on the specific epic and time range the user mentioned
"""

from datetime import datetime, timezone, timedelta
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_
import logging

def check_chart_streamer_entries():
    """Check what entries have chart_streamer data source"""
    print("🔍 Checking entries with 'chart_streamer' data source")
    
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    # Check around the time the user mentioned
    target_time = datetime(2025, 9, 2, 19, 50, 0)
    start_time = target_time - timedelta(hours=2)
    end_time = target_time + timedelta(hours=2)
    
    with SessionLocal() as session:
        # Find all entries with chart_streamer in the time range
        chart_streamer_entries = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.data_source == 'chart_streamer',
                IGCandle.start_time >= start_time,
                IGCandle.start_time <= end_time
            )
        ).order_by(IGCandle.start_time).all()
        
        print(f"✅ Found {len(chart_streamer_entries)} entries with 'chart_streamer' data source:")
        for entry in chart_streamer_entries:
            print(f"   {entry.start_time}: close={entry.close:.5f}, source={entry.data_source}")
        
        # Check all data sources in the range
        all_entries = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time >= start_time,
                IGCandle.start_time <= end_time
            )
        ).order_by(IGCandle.start_time).all()
        
        print(f"\n📊 Data source breakdown for {len(all_entries)} entries in range:")
        data_sources = {}
        for entry in all_entries:
            source = entry.data_source
            if source not in data_sources:
                data_sources[source] = []
            data_sources[source].append(entry)
        
        for source, entries in data_sources.items():
            print(f"   {source}: {len(entries)} entries")
            if len(entries) <= 5:  # Show all if few entries
                for entry in entries:
                    print(f"      {entry.start_time}: {entry.close:.5f}")
            else:  # Show first and last few
                print(f"      {entries[0].start_time}: {entries[0].close:.5f} (first)")
                print(f"      {entries[-1].start_time}: {entries[-1].close:.5f} (last)")
        
        # Check if the specific timestamp the user mentioned exists
        specific_entry = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time == target_time
            )
        ).first()
        
        if specific_entry:
            print(f"\n🎯 Specific timestamp {target_time}:")
            print(f"   Close: {specific_entry.close:.5f}")
            print(f"   Data Source: {specific_entry.data_source}")
        else:
            print(f"\n❌ Specific timestamp {target_time} not found")
        
        return len(chart_streamer_entries)

def check_broader_chart_streamer_entries():
    """Check chart_streamer entries across all time ranges"""
    print("\n🔍 Checking 'chart_streamer' entries across broader time range")
    
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    with SessionLocal() as session:
        # Count total chart_streamer entries
        total_chart_streamer = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.data_source == 'chart_streamer'
            )
        ).count()
        
        print(f"📊 Total 'chart_streamer' entries for {epic} {timeframe}m: {total_chart_streamer}")
        
        if total_chart_streamer > 0:
            # Get the newest and oldest chart_streamer entries
            newest = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.data_source == 'chart_streamer'
                )
            ).order_by(IGCandle.start_time.desc()).first()
            
            oldest = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.data_source == 'chart_streamer'
                )
            ).order_by(IGCandle.start_time.asc()).first()
            
            print(f"   Newest: {newest.start_time} (close: {newest.close:.5f})")
            print(f"   Oldest: {oldest.start_time} (close: {oldest.close:.5f})")
            
            # Show recent entries
            recent_entries = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.data_source == 'chart_streamer'
                )
            ).order_by(IGCandle.start_time.desc()).limit(10).all()
            
            print(f"\n   Last 10 'chart_streamer' entries:")
            for entry in recent_entries:
                print(f"      {entry.start_time}: {entry.close:.5f}")
        
        # Compare with other data sources
        print(f"\n📊 All data sources for {epic} {timeframe}m:")
        
        # Get count by data source
        from sqlalchemy import func
        source_counts = session.query(
            IGCandle.data_source,
            func.count(IGCandle.id).label('count')
        ).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe
            )
        ).group_by(IGCandle.data_source).all()
        
        for source, count in source_counts:
            print(f"   {source}: {count} entries")
        
        return total_chart_streamer

if __name__ == "__main__":
    print("🧪 Checking chart_streamer entries...")
    
    local_count = check_chart_streamer_entries()
    total_count = check_broader_chart_streamer_entries()
    
    print(f"\n🏁 SUMMARY:")
    print(f"   Local range (±2 hours): {local_count} chart_streamer entries")
    print(f"   Total database: {total_count} chart_streamer entries")
    
    if total_count == 0:
        print("\n💡 INSIGHT: No 'chart_streamer' entries found!")
        print("   This explains why the weekly corrector isn't finding anything to update.")
        print("   All data may have already been corrected or has different data_source values.")
    elif local_count == 0:
        print("\n💡 INSIGHT: No 'chart_streamer' entries in the target time range!")
        print("   The weekly corrector may be working on different time periods.")