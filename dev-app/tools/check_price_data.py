#!/usr/bin/env python3
"""
Check if price data is fresh and accurate
"""

import sys
sys.path.append('/app')

from services.db import SessionLocal
from services.models import IGCandle
from config import DEFAULT_TEST_EPIC
from datetime import datetime, timedelta

def check_price_data():
    with SessionLocal() as db:
        # Get latest candles for USDJPY
        recent_candles = (db.query(IGCandle)
                         .filter(IGCandle.epic == DEFAULT_TEST_EPIC)
                         .order_by(IGCandle.start_time.desc())
                         .limit(10)
                         .all())
        
        print("=== USDJPY PRICE DATA ===")
        print(f"{'Time':<20} {'Close':<10} {'Age (min)':<10}")
        print("-" * 40)
        
        now = datetime.utcnow()
        
        for candle in recent_candles:
            age_minutes = (now - candle.start_time).total_seconds() / 60
            print(f"{candle.start_time} {candle.close:<10.5f} {age_minutes:<10.1f}")
        
        if recent_candles:
            latest = recent_candles[0]
            age_minutes = (now - latest.start_time).total_seconds() / 60
            
            print(f"\n=== ANALYSIS ===")
            print(f"Latest price: {latest.close}")
            print(f"Data age: {age_minutes:.1f} minutes")
            print(f"Real IG price: 144.972 (your manual check)")
            print(f"Database price: {latest.close}")
            print(f"Difference: {abs(144.972 - latest.close):.5f}")
            
            if age_minutes > 5:
                print(f"❌ STALE DATA: Price data is {age_minutes:.1f} minutes old!")
            elif abs(144.972 - latest.close) > 0.01:
                print(f"❌ PRICE MISMATCH: DB price doesn't match IG reality")
            else:
                print(f"✅ Data looks fresh and accurate")
        else:
            print("❌ NO PRICE DATA FOUND")

def manual_calculation():
    print(f"\n=== MANUAL CALCULATION (Real Price: 144.972) ===")
    
    entry = 145.04
    current = 144.972
    current_stop_db = 144.926
    
    moved_in_favor = entry - current  # For SELL
    print(f"Moved in favor: {moved_in_favor:.5f} ({moved_in_favor/0.01:.1f} points)")
    
    break_even_trigger = 7 * 0.01
    print(f"Break-even trigger: {break_even_trigger:.5f} ({break_even_trigger/0.01:.1f} points)")
    print(f"Should trigger break-even: {moved_in_favor >= break_even_trigger}")
    
    # Correct stop position
    safe_distance = 2 * 0.01
    correct_stop = current + safe_distance
    print(f"Stop should be at: {correct_stop:.5f} (current + 2 points)")
    print(f"Stop currently at: {current_stop_db:.5f}")
    print(f"Stop needs to move: {correct_stop - current_stop_db:.5f} ({(correct_stop - current_stop_db)/0.01:.1f} points)")

if __name__ == "__main__":
    check_price_data()
    manual_calculation()