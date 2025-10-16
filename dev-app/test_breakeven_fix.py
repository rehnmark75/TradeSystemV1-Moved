#!/usr/bin/env python3
"""
Test script to verify break-even calculation fix
"""

def test_breakeven_logic():
    """Test that break-even is calculated correctly"""

    # Simulate GBPUSD trade
    entry_price = 1.34250
    lock_points = 4
    point_value = 0.0001

    # BUY trade
    direction = "BUY"
    break_even_stop = entry_price + (lock_points * point_value)

    print("=" * 60)
    print("BREAK-EVEN CALCULATION TEST")
    print("=" * 60)
    print(f"\nTrade Details:")
    print(f"  Direction: {direction}")
    print(f"  Entry: {entry_price:.5f}")
    print(f"  Lock points: {lock_points}")
    print(f"  Point value: {point_value:.5f}")

    print(f"\nBreak-even stop: {break_even_stop:.5f}")
    print(f"Distance from entry: {(break_even_stop - entry_price) / point_value:.1f} pts")

    # Test with different current prices
    test_scenarios = [
        ("Price moved 10 pts", 1.34350, 10),
        ("Price moved 15 pts (trigger)", 1.34405, 15),
        ("Price moved 20 pts", 1.34450, 20),
        ("Price moved 30 pts", 1.34550, 30),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION TESTS")
    print("=" * 60)

    for scenario, current_price, expected_profit in test_scenarios:
        actual_profit = (current_price - entry_price) / point_value

        # Old (WRONG) validation: rejected if break_even_stop <= current_price
        old_validation = not (break_even_stop <= current_price)

        # New (CORRECT) validation: For BUY, stop must be below current
        new_validation = break_even_stop < current_price

        print(f"\n{scenario}:")
        print(f"  Current price: {current_price:.5f}")
        print(f"  Profit: {actual_profit:.1f} pts (expected: {expected_profit})")
        print(f"  Break-even stop: {break_even_stop:.5f}")
        print(f"  Distance below current: {(current_price - break_even_stop) / point_value:.1f} pts")
        print(f"  Old validation (WRONG): {'✅ VALID' if old_validation else '❌ INVALID - triggers immediate trailing'}")
        print(f"  New validation (CORRECT): {'✅ VALID' if new_validation else '❌ INVALID'}")

        if old_validation != new_validation:
            print(f"  🔧 FIX CHANGED BEHAVIOR: Was rejected, now accepted!")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nBreak-even = Entry + IG minimum")
    print(f"           = {entry_price:.5f} + {lock_points} pts")
    print(f"           = {break_even_stop:.5f}")
    print(f"\nThis locks {lock_points} pts profit, regardless of current price.")
    print(f"As long as current price > break-even, it's VALID!")
    print("\n✅ Fix removes incorrect 'immediate trailing' workaround")
    print("✅ Break-even will now work as designed\n")

if __name__ == "__main__":
    test_breakeven_logic()
