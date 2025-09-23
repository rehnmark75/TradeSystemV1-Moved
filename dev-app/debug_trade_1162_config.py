#!/usr/bin/env python3
"""
Debug Trade 1162 Configuration Issue

This script reproduces the exact configuration that was used for trade 1162
to understand why it moved 6 points instead of 2 points.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_trade_1162_config():
    """Debug the exact configuration used for trade 1162"""

    print("🔍 DEBUG: Trade 1162 Configuration Analysis")
    print("=" * 60)

    # Mock trade 1162 with exact database values
    class MockTrade:
        def __init__(self):
            self.id = 1162
            self.symbol = "CS.D.AUDUSD.MINI.IP"
            self.entry_price = 0.66064
            self.direction = "BUY"
            self.min_stop_distance_points = 2.0  # From database

    # Test the progressive config system
    from services.progressive_config import get_progressive_config_for_epic

    trade = MockTrade()

    print(f"Trade Details:")
    print(f"  ID: {trade.id}")
    print(f"  Symbol: {trade.symbol}")
    print(f"  Entry: {trade.entry_price}")
    print(f"  Direction: {trade.direction}")
    print(f"  IG Minimum: {trade.min_stop_distance_points}")

    # Get configuration WITHOUT trade object (baseline)
    config_baseline = get_progressive_config_for_epic(trade.symbol, enable_adaptive=False)
    print(f"\n📊 Baseline Configuration (without trade):")
    print(f"  stage1_trigger_points: {config_baseline.stage1_trigger_points}")
    print(f"  stage1_lock_points: {config_baseline.stage1_lock_points}")

    # Get configuration WITH trade object (dynamic)
    config_dynamic = get_progressive_config_for_epic(trade.symbol, enable_adaptive=False, trade=trade)
    print(f"\n📊 Dynamic Configuration (with trade):")
    print(f"  stage1_trigger_points: {config_dynamic.stage1_trigger_points}")
    print(f"  stage1_lock_points: {config_dynamic.stage1_lock_points}")

    # Test the Progressive3StageTrailing logic
    from trailing_class import Progressive3StageTrailing, TrailingConfig, TrailingMethod

    class MockLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def debug(self, msg): print(f"[DEBUG] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")

    logger = MockLogger()

    # Use the dynamic configuration
    strategy = Progressive3StageTrailing(config_dynamic, logger)

    print(f"\n🧪 Testing Stage 1 Break-Even Logic:")
    print(f"Current price: 0.66144 (8 points profit)")

    # Test stage 1 calculation
    stage1_result = strategy._calculate_stage1_trail(trade, 0.66144, 0.0)

    print(f"Expected (using IG minimum 2): {0.66064 + (2 * 0.0001):.5f}")
    print(f"Expected (using config lock 2): {0.66064 + (2 * 0.0001):.5f}")
    print(f"Expected (if using trigger 6): {0.66064 + (6 * 0.0001):.5f}")
    print(f"Actual result: {stage1_result:.5f}")

    distance_moved = (stage1_result - trade.entry_price) / 0.0001
    print(f"Distance moved: {distance_moved:.1f} points")

    # Determine which value was used
    if abs(distance_moved - 2) < 0.1:
        print("✅ CORRECT: Used IG minimum (2 points)")
    elif abs(distance_moved - 6) < 0.1:
        print("❌ BUG: Used trigger value (6 points) instead of lock value (2 points)")
    else:
        print(f"⚠️ UNKNOWN: Used unexpected value ({distance_moved:.1f} points)")


if __name__ == "__main__":
    debug_trade_1162_config()