#!/usr/bin/env python3
"""
Debug script for NZDUSD trailing issue
"""

def get_point_value(epic):
    if "JPY" in epic:
        return 0.01
    elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return 0.0001
    return 1.0

def debug_nzdusd_stage3():
    """Debug the exact NZDUSD scenario"""
    print("🔍 Debugging NZDUSD Stage 3 Trailing")
    print("=" * 50)

    # Your current situation
    epic = "CS.D.NZDUSD.MINI.IP"
    direction = "SELL"  # Assuming SELL based on logs

    # Example values based on your description
    entry_price = 0.59022  # From logs
    current_price = 0.58788  # 23.4 points profit
    current_stop = 0.58882   # 14 points profit

    point_value = get_point_value(epic)

    # Calculate current profit
    if direction == "BUY":
        current_profit_points = (current_price - entry_price) / point_value
        stop_profit_points = (current_stop - entry_price) / point_value
    else:  # SELL
        current_profit_points = (entry_price - current_price) / point_value
        stop_profit_points = (entry_price - current_stop) / point_value

    print(f"📊 Current Situation:")
    print(f"   Entry: {entry_price:.5f}")
    print(f"   Current: {current_price:.5f}")
    print(f"   Current Stop: {current_stop:.5f}")
    print(f"   Current Profit: {current_profit_points:.1f} points")
    print(f"   Stop Profit: {stop_profit_points:.1f} points")

    # Stage 3 calculation
    if current_profit_points >= 50:
        retracement_percentage = 0.15  # 15%
    elif current_profit_points >= 25:
        retracement_percentage = 0.20  # 20%
    else:
        retracement_percentage = 0.25  # 25%

    trail_distance_points = max(2, current_profit_points * retracement_percentage)
    trail_distance_price = trail_distance_points * point_value

    if direction == "BUY":
        calculated_trail_level = current_price - trail_distance_price
    else:  # SELL
        calculated_trail_level = current_price + trail_distance_price

    # Calculate what the new stop profit would be
    if direction == "BUY":
        new_stop_profit = (calculated_trail_level - entry_price) / point_value
    else:  # SELL
        new_stop_profit = (entry_price - calculated_trail_level) / point_value

    print(f"\n🎯 Stage 3 Calculation:")
    print(f"   Retracement: {retracement_percentage*100:.0f}%")
    print(f"   Trail Distance: {trail_distance_points:.1f} points")
    print(f"   Calculated Trail Level: {calculated_trail_level:.5f}")
    print(f"   New Stop Profit: {new_stop_profit:.1f} points")

    # Check "don't move backwards" logic
    should_update = False
    if direction == "BUY":
        should_update = calculated_trail_level > current_stop
        comparison = ">"
    else:  # SELL
        should_update = calculated_trail_level < current_stop
        comparison = "<"

    print(f"\n🔧 Logic Check:")
    print(f"   Direction: {direction}")
    print(f"   Calculated: {calculated_trail_level:.5f}")
    print(f"   Current Stop: {current_stop:.5f}")
    print(f"   Should Update: {calculated_trail_level:.5f} {comparison} {current_stop:.5f} = {should_update}")

    if should_update:
        improvement = new_stop_profit - stop_profit_points
        print(f"   ✅ SHOULD MOVE: Stop improves from {stop_profit_points:.1f}pts → {new_stop_profit:.1f}pts (+{improvement:.1f}pts)")
    else:
        print(f"   ❌ WON'T MOVE: Calculated trail is not better than current stop")
        print(f"      This suggests current stop ({stop_profit_points:.1f}pts) is already better than calculated ({new_stop_profit:.1f}pts)")

    return {
        'should_update': should_update,
        'current_profit': current_profit_points,
        'current_stop_profit': stop_profit_points,
        'calculated_stop_profit': new_stop_profit,
        'improvement': new_stop_profit - stop_profit_points if should_update else 0
    }

if __name__ == "__main__":
    result = debug_nzdusd_stage3()

    if not result['should_update']:
        print(f"\n🚨 ISSUE IDENTIFIED:")
        print(f"   The current stop ({result['current_stop_profit']:.1f}pts) is already BETTER")
        print(f"   than what Stage 3 would calculate ({result['calculated_stop_profit']:.1f}pts)")
        print(f"   This means previous updates worked, but Stage 3 isn't")
        print(f"   trailing progressively as price moves further in your favor.")
        print(f"\n💡 SOLUTION:")
        print(f"   Stage 3 should trail from the BEST achieved price, not current price")
    else:
        print(f"\n✅ LOGIC LOOKS CORRECT:")
        print(f"   Stop should improve by {result['improvement']:.1f}pts")
        print(f"   Check if the API call is actually being sent")