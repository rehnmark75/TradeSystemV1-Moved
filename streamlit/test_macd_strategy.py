#!/usr/bin/env python3
"""
Test MACD Strategy Specifically
"""

import sys
import os
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/streamlit')

def test_macd_strategy():
    """Test MACD strategy specifically since it showed successful data fetching"""
    print("🧪 Testing MACD Strategy Execution")
    print("=" * 50)

    try:
        from services.worker_backtest_service import get_worker_backtest_service, BacktestConfig

        service = get_worker_backtest_service()

        # Test MACD strategy which showed successful data fetching in logs
        config = BacktestConfig(
            strategy_name='macd',
            epic='CS.D.AUDJPY.MINI.IP',  # Use the same epic that worked in logs
            days=7,  # Same as in logs
            timeframe='15m',
            parameters={}
        )

        print(f"🚀 Testing MACD strategy with {config.epic}")
        result = service.run_backtest(config)

        if result.success:
            print(f"✅ MACD backtest successful!")
            print(f"  Strategy: {result.strategy_name}")
            print(f"  Epic: {result.epic}")
            print(f"  Signals: {result.total_signals}")
            print(f"  Execution time: {result.execution_time:.2f}s")
            if result.performance_metrics:
                print(f"  Performance: {result.performance_metrics}")
            if result.signals:
                print(f"  Sample signals: {len(result.signals)} found")
                for i, signal in enumerate(result.signals[:3]):  # Show first 3
                    print(f"    Signal {i+1}: {signal}")
            return True
        else:
            print(f"❌ MACD backtest failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_macd_strategy()
    print("\n" + "=" * 50)
    if success:
        print("🎉 MACD strategy test PASSED! Real strategies are working!")
    else:
        print("⚠️ MACD strategy test FAILED.")