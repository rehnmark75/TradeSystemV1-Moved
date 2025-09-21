#!/usr/bin/env python3
"""
Test script for container backtest service
Run this from within the Streamlit container
"""

import sys
import os

def test_container_backtest():
    """Test the container backtest service"""
    print("🧪 Testing Container Backtest Service")
    print("=" * 40)

    try:
        # Test imports
        print("1. Testing imports...")
        from services.container_backtest_service import get_container_backtest_service, BacktestConfig
        print("   ✅ Successfully imported container backtest service")

        # Test service initialization
        print("2. Testing service initialization...")
        service = get_container_backtest_service()
        print(f"   ✅ Service initialized: {type(service)}")

        # Test strategy discovery
        print("3. Testing strategy discovery...")
        strategies = service.get_available_strategies()
        print(f"   ✅ Found {len(strategies)} strategies:")
        for name, info in strategies.items():
            print(f"      - {name}: {info.display_name}")
            print(f"        {info.description}")

        # Test configuration
        print("4. Testing backtest configuration...")
        config = BacktestConfig(
            strategy_name='historical_signals',
            epic='CS.D.EURUSD.MINI.IP',
            days=3,
            timeframe='15m',
            parameters={'min_confidence': 0.7}
        )
        print(f"   ✅ Config created: {config.strategy_name}")

        # Test minimal backtest run (historical analysis)
        print("5. Testing backtest execution...")
        result = service.run_backtest(config)
        print(f"   ✅ Backtest completed:")
        print(f"      Success: {result.success}")
        print(f"      Strategy: {result.strategy_name}")
        print(f"      Epic: {result.epic}")
        print(f"      Signals: {result.total_signals}")
        print(f"      Execution time: {result.execution_time:.2f}s")

        if result.error_message:
            print(f"      Error: {result.error_message}")

        if result.signals:
            print(f"      Sample signal: {result.signals[0]}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_container_backtest()
    sys.exit(0 if success else 1)