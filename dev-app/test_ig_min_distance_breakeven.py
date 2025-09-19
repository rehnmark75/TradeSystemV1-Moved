#!/usr/bin/env python3
"""
Test script to verify IG minimum distance breakeven implementation.

This script tests that AUDJPY trades use IG's minimum stop distance (2 points = 200 JPY)
instead of the configured stage1_lock_points (5 points = 500 JPY) for Stage 1 breakeven.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trailing_class import Progressive3StageTrailing, TrailingConfig, TrailingMethod
from services.models import TradeLog
from datetime import datetime


class MockTradeLog:
    """Mock TradeLog for testing"""
    def __init__(self, id, symbol, entry_price, direction, min_stop_distance_points=None):
        self.id = id
        self.symbol = symbol
        self.entry_price = entry_price
        self.direction = direction
        self.min_stop_distance_points = min_stop_distance_points
        self.sl_price = None
        self.moved_to_breakeven = False
        self.status = "pending"


class MockLogger:
    """Mock logger for testing"""
    def info(self, msg):
        print(f"[INFO] {msg}")

    def debug(self, msg):
        print(f"[DEBUG] {msg}")

    def warning(self, msg):
        print(f"[WARNING] {msg}")


def test_audjpy_ig_minimum_distance():
    """Test AUDJPY uses IG minimum distance instead of config lock points"""

    print("🧪 Testing AUDJPY IG Minimum Distance Breakeven")
    print("=" * 60)

    # Create mock config
    config = TrailingConfig(
        method=TrailingMethod.PROGRESSIVE_3_STAGE,
        stage1_lock_points=5  # Config says 5 points (500 JPY)
    )

    # Create mock logger
    logger = MockLogger()

    # Create progressive trailing strategy
    strategy = Progressive3StageTrailing(config, logger)

    # Test Case 1: AUDJPY with IG minimum distance
    print("\n📋 Test Case 1: AUDJPY with IG minimum distance (2 points)")
    trade_with_ig_min = MockTradeLog(
        id=123,
        symbol="CS.D.AUDJPY.MINI.IP",
        entry_price=96.750,
        direction="BUY",
        min_stop_distance_points=2.0  # IG says minimum 2 points
    )

    result1 = strategy._calculate_stage1_trail(trade_with_ig_min, 96.800, 0.0)
    expected1 = 96.750 + (2 * 0.01)  # entry + 2 points = 96.770

    print(f"Entry Price: {trade_with_ig_min.entry_price}")
    print(f"IG Min Distance: {trade_with_ig_min.min_stop_distance_points} points")
    print(f"Expected Stage 1 Stop: {expected1:.3f} (2 points = 200 JPY)")
    print(f"Actual Stage 1 Stop: {result1:.3f}")
    print(f"✅ PASS: Uses IG minimum" if abs(result1 - expected1) < 0.001 else "❌ FAIL")

    # Test Case 2: AUDJPY without IG minimum distance (fallback)
    print("\n📋 Test Case 2: AUDJPY without IG minimum distance (fallback to config)")
    trade_without_ig_min = MockTradeLog(
        id=124,
        symbol="CS.D.AUDJPY.MINI.IP",
        entry_price=96.750,
        direction="BUY",
        min_stop_distance_points=None  # No IG minimum available
    )

    result2 = strategy._calculate_stage1_trail(trade_without_ig_min, 96.800, 0.0)
    expected2 = 96.750 + (5 * 0.01)  # entry + 5 points = 96.800

    print(f"Entry Price: {trade_without_ig_min.entry_price}")
    print(f"IG Min Distance: None (fallback to config)")
    print(f"Expected Stage 1 Stop: {expected2:.3f} (5 points = 500 JPY)")
    print(f"Actual Stage 1 Stop: {result2:.3f}")
    print(f"✅ PASS: Uses config fallback" if abs(result2 - expected2) < 0.001 else "❌ FAIL")

    # Test Case 3: Non-JPY pair (should work normally)
    print("\n📋 Test Case 3: EURUSD (non-JPY pair)")
    trade_eurusd = MockTradeLog(
        id=125,
        symbol="CS.D.EURUSD.MINI.IP",
        entry_price=1.0850,
        direction="BUY",
        min_stop_distance_points=1.5  # IG minimum for EURUSD
    )

    result3 = strategy._calculate_stage1_trail(trade_eurusd, 1.0860, 0.0)
    expected3 = 1.0850 + (1.5 * 0.0001)  # entry + 1.5 points

    print(f"Entry Price: {trade_eurusd.entry_price}")
    print(f"IG Min Distance: {trade_eurusd.min_stop_distance_points} points")
    print(f"Expected Stage 1 Stop: {expected3:.5f}")
    print(f"Actual Stage 1 Stop: {result3:.5f}")
    print(f"✅ PASS: Uses IG minimum for non-JPY" if abs(result3 - expected3) < 0.00001 else "❌ FAIL")

    print("\n🎯 Summary:")
    print("- AUDJPY now uses IG minimum distance (2 points = 200 JPY) instead of config (5 points = 500 JPY)")
    print("- This gives trades more room to evolve while still protecting capital")
    print("- Fallback to config works when IG minimum is not available")
    print("- Works for both JPY and non-JPY pairs")


if __name__ == "__main__":
    test_audjpy_ig_minimum_distance()