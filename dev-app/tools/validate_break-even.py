#!/usr/bin/env python3
"""
Debug the break-even sequence that causes stop removal and recreation
"""

import sys
sys.path.append('/app')

from services.db import SessionLocal
from services.models import TradeLog, IGCandle
from datetime import datetime

def analyze_breakeven_issue():
    print("=== BREAK-EVEN SEQUENCE ANALYSIS ===")
    
    with SessionLocal() as db:
        # Find recent trades that might have this issue
        recent_trades = (db.query(TradeLog)
                        .filter(TradeLog.status.in_(["break_even", "trailing", "pending"]))
                        .order_by(TradeLog.id.desc())
                        .limit(3)
                        .all())
        
        for trade in recent_trades:
            print(f"\n--- Trade {trade.id} ({trade.symbol}) ---")
            print(f"Direction: {trade.direction}")
            print(f"Entry: {trade.entry_price}")
            print(f"Current SL in DB: {trade.sl_price}")
            print(f"Status: {trade.status}")
            print(f"Moved to BE: {getattr(trade, 'moved_to_breakeven', 'Unknown')}")
            
            # Get current price
            latest_candle = (db.query(IGCandle)
                           .filter(IGCandle.epic == trade.symbol, IGCandle.timeframe == 5)
                           .order_by(IGCandle.start_time.desc())
                           .first())
            
            if latest_candle:
                current_price = latest_candle.close
                print(f"Current Price: {current_price}")
                
                # Calculate what break-even should do
                if trade.direction.upper() == "SELL":
                    moved_in_favor = trade.entry_price - current_price
                    
                    print(f"Moved in favor: {moved_in_favor:.5f} ({moved_in_favor/0.01:.1f} points)")
                    
                    # What the break-even calculation should be
                    print(f"\n=== BREAK-EVEN CALCULATION ===")
                    print(f"Should move stop from: {trade.sl_price}")
                    print(f"Should move stop to: {trade.entry_price}")
                    stop_move_distance = trade.sl_price - trade.entry_price
                    adjustment_points = int(stop_move_distance / 0.01)
                    print(f"Stop move distance: {stop_move_distance:.5f}")
                    print(f"Adjustment points: {adjustment_points}")
                    
                    # Check if this violates minimum distance
                    min_distance_from_current = abs(trade.entry_price - current_price)
                    min_distance_points = min_distance_from_current / 0.01
                    print(f"Distance from current to entry: {min_distance_points:.1f} points")
                    
                    if min_distance_points < 4:
                        print("❌ PROBLEM: Entry price too close to current price!")
                        print("   IG probably rejects this and removes stop")
                    else:
                        print("✅ Distance seems safe")
                
                elif trade.direction.upper() == "BUY":
                    moved_in_favor = current_price - trade.entry_price
                    print(f"Moved in favor: {moved_in_favor:.5f} ({moved_in_favor/0.01:.1f} points)")
                    
                    print(f"\n=== BREAK-EVEN CALCULATION ===")
                    print(f"Should move stop from: {trade.sl_price}")
                    print(f"Should move stop to: {trade.entry_price}")
                    stop_move_distance = trade.entry_price - trade.sl_price
                    adjustment_points = int(stop_move_distance / 0.01)
                    print(f"Stop move distance: {stop_move_distance:.5f}")
                    print(f"Adjustment points: {adjustment_points}")
                    
                    min_distance_from_current = abs(current_price - trade.entry_price)
                    min_distance_points = min_distance_from_current / 0.01
                    print(f"Distance from current to entry: {min_distance_points:.1f} points")
                    
                    if min_distance_points < 4:
                        print("❌ PROBLEM: Entry price too close to current price!")
                        print("   IG probably rejects this and removes stop")
                    else:
                        print("✅ Distance seems safe")

def recommend_fixes():
    print(f"\n=== RECOMMENDED FIXES ===")
    print("1. **Validate before sending:**")
    print("   - Check distance from current price to target stop")
    print("   - Don't send if less than minimum distance")
    print()
    print("2. **Better error handling:**")
    print("   - Detect when stop gets removed")
    print("   - Set stop at safe distance instead of exact break-even")
    print()
    print("3. **Two-step approach:**")
    print("   - First: Move stop to safe distance from current price")
    print("   - Later: Move to exact break-even when price moves further")
    print()
    print("4. **Fix limit adjustment bug:**")
    print("   - Ensure limit_offset_points=0 doesn't change limit")
    print("   - Update FastAPI endpoint logic")

if __name__ == "__main__":
    analyze_breakeven_issue()
    recommend_fixes()