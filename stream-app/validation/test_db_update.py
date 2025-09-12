#!/usr/bin/env python3
"""
Quick test to debug database update issue
"""

from datetime import datetime
from services.db import SessionLocal
from services.models import IGCandle
from sqlalchemy import and_

def test_database_update():
    print("🧪 Testing database update functionality...")
    
    # Target the specific timestamp you mentioned
    test_timestamp = datetime(2025, 9, 2, 19, 50, 0)
    epic = "CS.D.EURUSD.CEEM.IP"
    timeframe = 5
    
    with SessionLocal() as session:
        # First, find the entry
        entry = session.query(IGCandle).filter(
            and_(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time == test_timestamp
            )
        ).first()
        
        if entry:
            print(f"✅ Found entry:")
            print(f"   Timestamp: {entry.start_time}")
            print(f"   Close: {entry.close}")
            print(f"   Data Source: {entry.data_source}")
            
            # Test update
            old_close = entry.close
            test_close = 1.99999  # Obvious test value
            
            print(f"\n🔧 Testing update...")
            rows_updated = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.start_time == test_timestamp
                )
            ).update({
                'close': test_close,
                'data_source': 'test_update',
                'updated_at': datetime.now()
            })
            
            print(f"   Rows updated: {rows_updated}")
            
            # Commit
            session.commit()
            print("   ✅ Commit successful")
            
            # Verify
            updated_entry = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.start_time == test_timestamp
                )
            ).first()
            
            if updated_entry:
                print(f"\n🔍 After update:")
                print(f"   Close: {updated_entry.close} (was {old_close})")
                print(f"   Data Source: {updated_entry.data_source}")
                
                if updated_entry.close == test_close:
                    print("✅ Update SUCCESS!")
                    
                    # Restore original value
                    session.query(IGCandle).filter(
                        and_(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time == test_timestamp
                        )
                    ).update({
                        'close': old_close,
                        'data_source': 'chart_streamer'  # Restore original
                    })
                    session.commit()
                    print("✅ Restored original value")
                else:
                    print("❌ Update FAILED!")
            else:
                print("❌ Entry disappeared after update!")
                
        else:
            print(f"❌ Entry not found for {test_timestamp}")
            
            # Show what entries do exist around that time
            print(f"\n🔍 Entries near that time:")
            nearby = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.start_time >= test_timestamp - timedelta(minutes=30),
                    IGCandle.start_time <= test_timestamp + timedelta(minutes=30)
                )
            ).order_by(IGCandle.start_time).limit(10).all()
            
            for entry in nearby:
                print(f"   {entry.start_time}: {entry.close} ({entry.data_source})")

if __name__ == "__main__":
    test_database_update()