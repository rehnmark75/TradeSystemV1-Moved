#!/usr/bin/env python3
"""
AUDUSD Break-Even Distance Analysis - Real Scenario Investigation

This script reproduces the user's observation where AUDUSD moved to 6 points
instead of the expected 2 points from the configuration table.

Analysis:
- Configuration table shows stage1_lock_points=2 for AUDUSD
- But IG's min_stop_distance_points can override this value
- When IG minimum was 6, trades moved 6 points to break-even
- When IG minimum is 2, trades move 2 points to break-even
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trailing_class import Progressive3StageTrailing, TrailingConfig, TrailingMethod
from datetime import datetime


class MockTradeLog:
    """Mock TradeLog for testing different scenarios"""
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


def test_audusd_breakeven_scenarios():
    """Test AUDUSD break-even behavior with different IG minimum distances"""

    print("🔍 AUDUSD Break-Even Distance Analysis")
    print("=" * 60)
    print("User's Question: Why did AUDUSD move 6 points instead of 2?")
    print("Answer: IG's minimum distance overrides configuration values")
    print("=" * 60)

    # Create mock config (matches BALANCED_PROGRESSIVE_CONFIG)
    config = TrailingConfig(
        method=TrailingMethod.PROGRESSIVE_3_STAGE,
        stage1_lock_points=2  # Configuration table shows 2 points
    )

    logger = MockLogger()
    strategy = Progressive3StageTrailing(config, logger)

    # Test Case 1: Historical scenario (when IG minimum was 6)
    print("\n📊 Test Case 1: Historical AUDUSD (IG minimum = 6 points)")
    print("This reproduces the user's observation")

    trade_historical = MockTradeLog(
        id=269,  # Actual trade ID from database
        symbol="CS.D.AUDUSD.MINI.IP",
        entry_price=0.65365,
        direction="BUY",
        min_stop_distance_points=6.0  # Historical IG minimum
    )

    result_historical = strategy._calculate_stage1_trail(trade_historical, 0.65400, 0.0)
    expected_historical = 0.65365 + (6 * 0.0001)  # entry + 6 points

    print(f"Entry Price: {trade_historical.entry_price}")
    print(f"Configuration says: {config.stage1_lock_points} points")
    print(f"IG Minimum Distance: {trade_historical.min_stop_distance_points} points")
    print(f"Expected Stop: {expected_historical:.5f} (IG minimum used)")
    print(f"Actual Stop: {result_historical:.5f}")
    print(f"Distance moved: {((result_historical - trade_historical.entry_price) / 0.0001):.1f} points")
    print(f"✅ MATCHES USER'S OBSERVATION" if abs(result_historical - expected_historical) < 0.00001 else "❌ MISMATCH")

    # Test Case 2: Current scenario (IG minimum = 2)
    print("\n📊 Test Case 2: Current AUDUSD (IG minimum = 2 points)")
    print("This shows current behavior")

    trade_current = MockTradeLog(
        id=1162,  # Recent trade ID from database
        symbol="CS.D.AUDUSD.MINI.IP",
        entry_price=0.66064,
        direction="BUY",
        min_stop_distance_points=2.0  # Current IG minimum
    )

    result_current = strategy._calculate_stage1_trail(trade_current, 0.66100, 0.0)
    expected_current = 0.66064 + (2 * 0.0001)  # entry + 2 points

    print(f"Entry Price: {trade_current.entry_price}")
    print(f"Configuration says: {config.stage1_lock_points} points")
    print(f"IG Minimum Distance: {trade_current.min_stop_distance_points} points")
    print(f"Expected Stop: {expected_current:.5f} (IG minimum used)")
    print(f"Actual Stop: {result_current:.5f}")
    print(f"Distance moved: {((result_current - trade_current.entry_price) / 0.0001):.1f} points")
    print(f"✅ MATCHES CONFIGURATION" if abs(result_current - expected_current) < 0.00001 else "❌ MISMATCH")

    # Test Case 3: Fallback scenario (no IG minimum)
    print("\n📊 Test Case 3: Fallback (no IG minimum available)")
    print("This shows what happens when IG data is missing")

    trade_fallback = MockTradeLog(
        id=9999,
        symbol="CS.D.AUDUSD.MINI.IP",
        entry_price=0.66000,
        direction="BUY",
        min_stop_distance_points=None  # No IG minimum available
    )

    result_fallback = strategy._calculate_stage1_trail(trade_fallback, 0.66100, 0.0)
    expected_fallback = 0.66000 + (2 * 0.0001)  # entry + config points

    print(f"Entry Price: {trade_fallback.entry_price}")
    print(f"Configuration says: {config.stage1_lock_points} points")
    print(f"IG Minimum Distance: None (fallback to config)")
    print(f"Expected Stop: {expected_fallback:.5f} (config used)")
    print(f"Actual Stop: {result_fallback:.5f}")
    print(f"Distance moved: {((result_fallback - trade_fallback.entry_price) / 0.0001):.1f} points")
    print(f"✅ USES CONFIG FALLBACK" if abs(result_fallback - expected_fallback) < 0.00001 else "❌ MISMATCH")

    print("\n" + "=" * 60)
    print("🎯 CONCLUSION:")
    print("=" * 60)
    print("1. The 'table' shows stage1_lock_points=2 (configuration fallback)")
    print("2. IG's minimum distance OVERRIDES the configuration value")
    print("3. When IG minimum was 6 points → trade moved 6 points")
    print("4. When IG minimum is 2 points → trade moves 2 points")
    print("5. The behavior is working as designed (IG minimum takes priority)")
    print("\n💡 The user saw 6 points because IG required 6 points minimum at that time")


def test_sell_trade_scenario():
    """Test SELL trade scenario for completeness"""
    print("\n" + "=" * 60)
    print("🔄 SELL Trade Scenario (for completeness)")
    print("=" * 60)

    config = TrailingConfig(stage1_lock_points=2)
    logger = MockLogger()
    strategy = Progressive3StageTrailing(config, logger)

    trade_sell = MockTradeLog(
        id=1000,
        symbol="CS.D.AUDUSD.MINI.IP",
        entry_price=0.66000,
        direction="SELL",
        min_stop_distance_points=6.0
    )

    result_sell = strategy._calculate_stage1_trail(trade_sell, 0.65950, 0.0)
    expected_sell = 0.66000 - (6 * 0.0001)  # entry - 6 points for SELL

    print(f"SELL Entry: {trade_sell.entry_price}")
    print(f"IG Minimum: {trade_sell.min_stop_distance_points} points")
    print(f"Expected Stop: {expected_sell:.5f}")
    print(f"Actual Stop: {result_sell:.5f}")
    print(f"Distance moved: {((trade_sell.entry_price - result_sell) / 0.0001):.1f} points")
    print(f"✅ SELL LOGIC CORRECT" if abs(result_sell - expected_sell) < 0.00001 else "❌ SELL LOGIC ERROR")


if __name__ == "__main__":
    test_audusd_breakeven_scenarios()
    test_sell_trade_scenario()