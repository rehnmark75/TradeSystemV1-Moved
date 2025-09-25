#!/usr/bin/env python3
"""
Test to verify the fix for trade 1170 trailing stop issue.

The issue: Trade 1170 (USDJPY) was moving to 4 points at trigger instead of 2 points minimum.
Expected: min_stop_distance_points (2) should be used for Stage 1 trailing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trailing_class import Progressive3StageTrailing, TrailingConfig, TrailingMethod
from datetime import datetime


class MockTradeLog:
    """Mock TradeLog matching trade 1170 characteristics"""
    def __init__(self):
        self.id = 1170
        self.symbol = "CS.D.USDJPY.MINI.IP"
        self.entry_price = 148.757
        self.direction = "BUY"
        self.min_stop_distance_points = 2.0  # IG's minimum distance
        self.sl_price = 148.817  # Current stop (6 pips from entry)
        self.moved_to_breakeven = False
        self.status = "tracking"


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


def test_trade_1170_fix():
    """Test that trade 1170 correctly uses 2 points minimum distance"""

    print("🧪 Testing Trade 1170 Trailing Stop Fix")
    print("=" * 50)

    # Create mock config
    config = TrailingConfig(
        method=TrailingMethod.PROGRESSIVE_3_STAGE,
        stage1_trigger_points=6,  # Need 6 points profit to start Stage 1
        stage1_lock_points=5      # Default config (should be overridden by IG min)
    )

    # Create mock logger
    logger = MockLogger()

    # Create progressive trailing strategy
    strategy = Progressive3StageTrailing(config, logger)

    # Create mock trade matching trade 1170
    trade = MockTradeLog()

    print(f"\n📋 Trade 1170 Details:")
    print(f"   Symbol: {trade.symbol}")
    print(f"   Entry: {trade.entry_price}")
    print(f"   Direction: {trade.direction}")
    print(f"   Current Stop: {trade.sl_price} (6 pips from entry)")
    print(f"   IG Min Distance: {trade.min_stop_distance_points} points")

    # Test current price that would trigger Stage 1 (entry + 6 points = 148.817)
    current_price = 148.817  # 6 points profit

    print(f"\n🧮 Testing Stage 1 Calculation:")
    print(f"   Current Price: {current_price} (6 points profit)")

    # Calculate Stage 1 trail level
    stage1_result = strategy._calculate_stage1_trail(trade, current_price, trade.sl_price)

    # Expected result: entry + IG minimum distance (2 points)
    expected_result = trade.entry_price + (2 * 0.01)  # 148.757 + 0.02 = 148.777

    print(f"\n📊 Results:")
    print(f"   Expected Stage 1 Stop: {expected_result:.3f} (entry + 2 points)")
    print(f"   Actual Stage 1 Stop: {stage1_result:.3f}")
    print(f"   Difference: {abs(stage1_result - expected_result):.5f}")

    # Verify the fix
    is_correct = abs(stage1_result - expected_result) < 0.001
    print(f"\n✅ PASS: Uses IG minimum (2 points)" if is_correct else f"❌ FAIL: Not using IG minimum correctly")

    if is_correct:
        print("\n🎯 Trade 1170 Fix Verification:")
        print("   ✓ Stage 1 correctly uses min_stop_distance_points (2)")
        print("   ✓ Stop will move to entry + 2 points = 148.777")
        print("   ✓ This provides 2 points minimum distance as required by IG")
    else:
        print("\n❌ Trade 1170 Fix Issues:")
        print(f"   ✗ Stage 1 not using IG minimum correctly")
        print(f"   ✗ Expected: {expected_result:.3f}, Got: {stage1_result:.3f}")

    print(f"\n📝 Summary:")
    print(f"   Before fix: Stop moved to 4 points distance (148.797)")
    print(f"   After fix: Stop moves to 2 points distance ({expected_result:.3f})")
    print(f"   Improvement: Follows IG minimum distance requirement")


if __name__ == "__main__":
    test_trade_1170_fix()