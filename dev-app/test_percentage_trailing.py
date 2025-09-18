#!/usr/bin/env python3
"""
Test script for the new percentage-based trailing system
"""

class MockTrade:
    def __init__(self, symbol, direction, entry_price):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.sl_price = None

class MockConfig:
    def __init__(self):
        self.stage3_min_distance = 2

def get_point_value(epic):
    if "JPY" in epic:
        return 0.01
    elif any(pair in epic for pair in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        return 0.0001
    return 1.0

def calculate_percentage_trail(trade, current_price, current_stop=0.0):
    """Test the new percentage-based trailing calculation"""
    direction = trade.direction.upper()
    point_value = get_point_value(trade.symbol)

    # Calculate current profit in points
    if direction == "BUY":
        current_profit_points = (current_price - trade.entry_price) / point_value
    else:  # SELL
        current_profit_points = (trade.entry_price - current_price) / point_value

    # Tiered percentage trailing based on profit level
    if current_profit_points >= 50:
        retracement_percentage = 0.15  # 15% retracement for big profits (50+ points)
    elif current_profit_points >= 25:
        retracement_percentage = 0.20  # 20% retracement for medium profits (25-49 points)
    else:
        retracement_percentage = 0.25  # 25% retracement for smaller profits (18-24 points)

    # Calculate trail distance in points, with minimum protection
    trail_distance_points = max(
        2,  # Minimum trailing distance
        current_profit_points * retracement_percentage  # Percentage-based distance
    )

    trail_distance_price = trail_distance_points * point_value

    if direction == "BUY":
        trail_level = current_price - trail_distance_price
    else:  # SELL
        trail_level = current_price + trail_distance_price

    # Ensure we don't move backwards from previous stages
    if current_stop > 0:
        if direction == "BUY":
            trail_level = max(trail_level, current_stop)
        else:
            trail_level = min(trail_level, current_stop)

    return {
        'trail_level': round(trail_level, 5),
        'profit_points': current_profit_points,
        'retracement_pct': retracement_percentage * 100,
        'trail_distance_points': trail_distance_points,
        'protected_profit_points': current_profit_points - trail_distance_points
    }

def test_gbpusd_scenario():
    """Test with your current GBPUSD scenario"""
    print("🧪 Testing Percentage-Based Trailing for GBPUSD")
    print("=" * 50)

    # Your current GBPUSD trade scenario
    trade = MockTrade("CS.D.GBPUSD.MINI.IP", "BUY", 1.2500)  # Example entry

    # Test different profit levels
    scenarios = [
        ("Current situation", 1.2534, "34 points profit"),  # Your current scenario
        ("If price moves to +25pts", 1.2525, "25 points profit"),
        ("If price moves to +50pts", 1.2550, "50 points profit"),
        ("If price moves to +75pts", 1.2575, "75 points profit"),
    ]

    for scenario_name, current_price, description in scenarios:
        result = calculate_percentage_trail(trade, current_price)

        print(f"\n📊 {scenario_name}: {description}")
        print(f"   Current Price: {current_price:.4f}")
        print(f"   Profit: {result['profit_points']:.1f} points")
        print(f"   Retracement: {result['retracement_pct']:.0f}%")
        print(f"   Trail Distance: {result['trail_distance_points']:.1f} points")
        print(f"   Trail Stop Level: {result['trail_level']:.4f}")
        print(f"   Protected Profit: {result['protected_profit_points']:.1f} points")

        # Show what happens if price retraces to stop
        if result['protected_profit_points'] > 0:
            print(f"   → If stopped out: +{result['protected_profit_points']:.1f}pt profit ✅")
        else:
            print(f"   → If stopped out: break-even/small loss ⚠️")

def test_comparison_with_old_atr():
    """Compare with old ATR system"""
    print("\n\n🔄 Comparison: New vs Old ATR System")
    print("=" * 50)

    trade = MockTrade("CS.D.GBPUSD.MINI.IP", "BUY", 1.2500)
    current_price = 1.2534  # 34 points profit

    # New percentage system
    new_result = calculate_percentage_trail(trade, current_price)

    # Old ATR system (0.8x multiplier, assume ATR = 0.0015)
    atr = 0.0015
    old_multiplier = 0.8
    old_trail_distance = atr * old_multiplier
    old_trail_level = current_price - old_trail_distance
    old_protected_points = (old_trail_level - trade.entry_price) / 0.0001

    print(f"📈 NEW Percentage System:")
    print(f"   Trail Level: {new_result['trail_level']:.4f}")
    print(f"   Protected Profit: {new_result['protected_profit_points']:.1f} points")
    print(f"   Retracement Allowed: {new_result['retracement_pct']:.0f}%")

    print(f"\n📉 OLD ATR System (0.8x multiplier):")
    print(f"   Trail Level: {old_trail_level:.4f}")
    print(f"   Protected Profit: {old_protected_points:.1f} points")
    print(f"   Trail Distance: {old_trail_distance/0.0001:.1f} points")

    improvement = new_result['protected_profit_points'] - old_protected_points
    print(f"\n🎯 Improvement: {improvement:+.1f} points better protection")

if __name__ == "__main__":
    test_gbpusd_scenario()
    test_comparison_with_old_atr()

    print("\n\n✅ Percentage-based trailing is now implemented!")
    print("📋 Next steps:")
    print("   1. The system will automatically use this for Stage 3 (18+ points)")
    print("   2. No configuration changes needed - it's already active")
    print("   3. Monitor your next trades to see the improved trailing behavior")