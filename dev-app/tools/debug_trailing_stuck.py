#!/usr/bin/env python3
"""
Debug why trailing is stuck at safe distance calculation
"""

import sys
sys.path.append('/app')

from services.db import SessionLocal
from services.models import TradeLog
from utils import get_point_value

def debug_trade_201():
    with SessionLocal() as db:
        trade = db.query(TradeLog).filter(TradeLog.id == 201).first()
        
        if not trade:
            print("Trade 201 not found")
            return
        
        print("=== TRADE 201 DEBUG ===")
        print(f"ID: {trade.id}")
        print(f"Symbol: {trade.symbol}")
        print(f"Direction: {trade.direction}")
        print(f"Status: {trade.status}")
        print(f"Entry Price: {trade.entry_price}")
        print(f"SL Price (DB): {trade.sl_price}")
        print(f"Last Trigger Price: {trade.last_trigger_price}")
        print(f"Moved to Breakeven: {getattr(trade, 'moved_to_breakeven', 'Unknown')}")
        print(f"Min Stop Distance: {getattr(trade, 'min_stop_distance_points', 'Unknown')}")
        
        # Get current price from latest candle
        from services.models import IGCandle
        latest_candle = (db.query(IGCandle)
                        .filter(IGCandle.epic == trade.symbol)
                        .order_by(IGCandle.start_time.desc())
                        .first())
        
        if latest_candle:
            current_price = latest_candle.close
            print(f"Current Price: {current_price}")
            
            # Calculate key metrics
            point_value = get_point_value(trade.symbol)
            print(f"Point Value: {point_value}")
            
            # Distance between current price and stop
            if trade.direction.upper() == "SELL":
                distance_points = (trade.sl_price - current_price) / point_value
                print(f"Distance (stop above current): {distance_points:.1f} points")
                
                # Break-even calculations
                break_even_trigger = 7 * point_value  # SCALPING_CONFIG break_even_trigger_points
                moved_in_favor = trade.entry_price - current_price
                print(f"Moved in favor: {moved_in_favor:.5f} ({moved_in_favor/point_value:.1f} points)")
                print(f"Break-even trigger: {break_even_trigger:.5f} ({break_even_trigger/point_value:.1f} points)")
                
                # Trailing calculations
                if getattr(trade, 'moved_to_breakeven', False):
                    additional_move = trade.entry_price - current_price - break_even_trigger
                    trail_trigger = 2 * point_value  # min_trail_distance
                    should_trail = additional_move >= trail_trigger
                    
                    print(f"Additional move: {additional_move:.5f} ({additional_move/point_value:.1f} points)")
                    print(f"Trail trigger needed: {trail_trigger:.5f} ({trail_trigger/point_value:.1f} points)")
                    print(f"Should trail: {should_trail}")
                    
                    if should_trail:
                        # Calculate what the safe trail level should be
                        safe_trail_level = current_price + (2 * point_value)  # 2 points above current for SELL
                        print(f"Safe trail level should be: {safe_trail_level:.5f}")
                        print(f"Current stop in DB: {trade.sl_price:.5f}")
                        
                        should_actually_trail = safe_trail_level < trade.sl_price
                        print(f"Should actually trail (new < current): {should_actually_trail}")
                        
                        if should_actually_trail:
                            adjustment_distance = trade.sl_price - safe_trail_level
                            adjustment_points = adjustment_distance / point_value
                            print(f"Adjustment distance: {adjustment_distance:.5f}")
                            print(f"Adjustment points: {adjustment_points:.2f}")
                        else:
                            print("❌ Trail level not better than current stop")
                    else:
                        print("❌ Not enough additional movement to trigger trailing")
                else:
                    print("❌ Not moved to break-even yet")
            
        else:
            print("❌ No current price data available")

if __name__ == "__main__":
    debug_trade_201()