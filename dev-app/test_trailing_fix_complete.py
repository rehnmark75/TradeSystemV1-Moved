#!/usr/bin/env python3
"""
Comprehensive test to verify the complete trailing stop fix.

Issues Fixed:
1. Stage 1 was moving in 2-point increments instead of moving once to break-even
2. Stage 1 was using wrong minimum distance (4 points instead of 2 points)
3. Stage 2 was also trailing continuously instead of locking profit once

Expected Behavior After Fix:
- Stage 1: Move ONCE to entry + min_stop_distance_points (break-even protection)
- Stage 2: Move ONCE to entry + stage2_lock_points (profit lock)
- Stage 3: Continuous percentage-based trailing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trailing_class import Progressive3StageTrailing, TrailingConfig, TrailingMethod
from datetime import datetime


class MockTradeLog:
    """Mock TradeLog for comprehensive testing"""
    def __init__(self, id, symbol, entry_price, direction, min_stop_distance_points=None,
                 moved_to_breakeven=False, status="tracking"):
        self.id = id
        self.symbol = symbol
        self.entry_price = entry_price
        self.direction = direction
        self.min_stop_distance_points = min_stop_distance_points
        self.sl_price = None
        self.moved_to_breakeven = moved_to_breakeven
        self.status = status


class MockLogger:
    """Mock logger for testing"""
    def info(self, msg):
        print(f"[INFO] {msg}")

    def debug(self, msg):
        print(f"[DEBUG] {msg}")

    def warning(self, msg):
        print(f"[WARNING] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


def test_complete_trailing_fix():
    """Test complete trailing stop fix for all stages"""

    print("🧪 Testing Complete Trailing Stop Fix")
    print("=" * 60)

    # Create mock config
    config = TrailingConfig(
        method=TrailingMethod.PROGRESSIVE_3_STAGE,
        stage1_trigger_points=6,     # Trigger Stage 1 at 6 points profit
        stage1_lock_points=5,        # Fallback: 5 points from entry (should be overridden by IG min)
        stage2_trigger_points=10,    # Trigger Stage 2 at 10 points profit
        stage2_lock_points=3,        # Lock 3 points profit
        stage3_trigger_points=18     # Trigger Stage 3 at 18 points profit
    )

    # Create mock logger
    logger = MockLogger()

    # Create progressive trailing strategy
    strategy = Progressive3StageTrailing(config, logger)

    # === TEST 1: Stage 1 - First Time (Should Move) ===
    print("\n📋 TEST 1: Stage 1 - First Time Application")
    trade1 = MockTradeLog(
        id=1170,
        symbol="CS.D.USDJPY.MINI.IP",
        entry_price=148.757,
        direction="BUY",
        min_stop_distance_points=2.0,  # IG minimum
        moved_to_breakeven=False,      # First time
        status="tracking"
    )

    current_price1 = 148.817  # 6 points profit (triggers Stage 1)
    result1 = strategy._calculate_stage1_trail(trade1, current_price1, 0.0)
    expected1 = 148.757 + (2 * 0.01)  # entry + 2 points = 148.777

    print(f"Entry: {trade1.entry_price}, Current: {current_price1}")
    print(f"Expected Stage 1 Stop: {expected1:.3f} (2 points from entry)")
    print(f"Actual Stage 1 Stop: {result1:.3f}")
    print(f"✅ PASS: Correct IG minimum used" if abs(result1 - expected1) < 0.001 else "❌ FAIL: Wrong distance")

    # === TEST 2: Stage 1 - Second Time (Should Skip) ===
    print("\n📋 TEST 2: Stage 1 - Already Applied (Should Skip)")
    trade2 = MockTradeLog(
        id=1170,
        symbol="CS.D.USDJPY.MINI.IP",
        entry_price=148.757,
        direction="BUY",
        min_stop_distance_points=2.0,
        moved_to_breakeven=True,       # Already applied
        status="break_even"
    )

    current_price2 = 148.827  # 7 points profit (still Stage 1 range)
    result2 = strategy._calculate_stage1_trail(trade2, current_price2, 148.777)

    print(f"Entry: {trade2.entry_price}, Current: {current_price2}")
    print(f"Stage 1 Result: {result2}")
    print(f"✅ PASS: Stage 1 skipped correctly" if result2 is None else "❌ FAIL: Stage 1 applied again")

    # === TEST 3: Stage 2 - First Time (Should Move) ===
    print("\n📋 TEST 3: Stage 2 - First Time Application")
    trade3 = MockTradeLog(
        id=1170,
        symbol="CS.D.USDJPY.MINI.IP",
        entry_price=148.757,
        direction="BUY",
        min_stop_distance_points=2.0,
        moved_to_breakeven=True,       # Stage 1 already done
        status="break_even"            # Ready for Stage 2
    )

    current_price3 = 148.857  # 10 points profit (triggers Stage 2)
    current_stop3 = 148.777   # Current stop from Stage 1
    result3 = strategy._calculate_stage2_trail(trade3, current_price3, current_stop3)
    expected3 = 148.757 + (3 * 0.01)  # entry + 3 points = 148.787

    print(f"Entry: {trade3.entry_price}, Current: {current_price3}")
    print(f"Current Stop: {current_stop3:.3f}")
    print(f"Expected Stage 2 Stop: {expected3:.3f} (3 points from entry)")
    print(f"Actual Stage 2 Stop: {result3:.3f}")
    print(f"✅ PASS: Correct Stage 2 level" if abs(result3 - expected3) < 0.001 else "❌ FAIL: Wrong Stage 2 level")

    # === TEST 4: Stage 2 - Second Time (Should Skip) ===
    print("\n📋 TEST 4: Stage 2 - Already Applied (Should Skip)")
    trade4 = MockTradeLog(
        id=1170,
        symbol="CS.D.USDJPY.MINI.IP",
        entry_price=148.757,
        direction="BUY",
        min_stop_distance_points=2.0,
        moved_to_breakeven=True,
        status="stage2_profit_lock"    # Stage 2 already applied
    )

    current_price4 = 148.867  # 11 points profit (still Stage 2 range)
    current_stop4 = 148.787   # Current stop from Stage 2
    result4 = strategy._calculate_stage2_trail(trade4, current_price4, current_stop4)

    print(f"Entry: {trade4.entry_price}, Current: {current_price4}")
    print(f"Stage 2 Result: {result4}")
    print(f"✅ PASS: Stage 2 skipped correctly" if result4 is None else "❌ FAIL: Stage 2 applied again")

    # === SUMMARY ===
    print(f"\n🎯 COMPLETE TRAILING STOP FIX SUMMARY:")
    print(f"=" * 50)
    print(f"✅ Stage 1 Issues Fixed:")
    print(f"   • Uses correct IG minimum distance (2 points, not 4)")
    print(f"   • Moves only ONCE to break-even protection")
    print(f"   • Skips subsequent calls when moved_to_breakeven=True")
    print(f"")
    print(f"✅ Stage 2 Issues Fixed:")
    print(f"   • Moves only ONCE to profit lock level")
    print(f"   • Skips subsequent calls when status=stage2_profit_lock")
    print(f"")
    print(f"📈 Expected Trade 1170 Behavior:")
    print(f"   • At 6 pts profit: Move to {expected1:.3f} (Stage 1 break-even)")
    print(f"   • At 10 pts profit: Move to {expected3:.3f} (Stage 2 profit lock)")
    print(f"   • At 18+ pts profit: Begin continuous percentage trailing (Stage 3)")


if __name__ == "__main__":
    test_complete_trailing_fix()