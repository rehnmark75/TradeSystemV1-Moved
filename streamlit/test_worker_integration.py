#!/usr/bin/env python3
"""
Test Worker Integration
Quick test to verify the worker backtest service integration
"""

import sys
import os
import logging

# Add current directory to path for imports
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/streamlit')

def test_imports():
    """Test if we can import the required services"""
    print("Testing imports...")
    try:
        from services.worker_backtest_service import get_worker_backtest_service, BacktestConfig
        from services.container_backtest_service import get_container_backtest_service
        print("✅ Successfully imported both services")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_worker_service():
    """Test worker service connectivity"""
    print("\nTesting worker service...")
    try:
        from services.worker_backtest_service import get_worker_backtest_service

        service = get_worker_backtest_service()
        print(f"✅ Worker service initialized: {type(service)}")

        # Test health check
        health = service.check_worker_health()
        print(f"Worker health: {health}")

        if health.get('status') == 'healthy':
            print("🔗 Worker container is healthy!")

            # Get worker info
            info = service.get_worker_info()
            print("Worker info:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            # Test strategy discovery
            strategies = service.get_available_strategies()
            print(f"✅ Found {len(strategies)} strategies from worker:")
            for name, strategy_info in strategies.items():
                print(f"  - {name}: {strategy_info.display_name}")

            return True, strategies
        else:
            print(f"⚠️ Worker unhealthy: {health}")
            return False, {}

    except Exception as e:
        print(f"❌ Worker service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_container_service():
    """Test container service"""
    print("\nTesting container service...")
    try:
        from services.container_backtest_service import get_container_backtest_service

        service = get_container_backtest_service()
        print(f"✅ Container service initialized: {type(service)}")

        strategies = service.get_available_strategies()
        print(f"✅ Found {len(strategies)} strategies from container:")
        for name, strategy_info in strategies.items():
            print(f"  - {name}: {strategy_info.display_name}")

        return True, strategies

    except Exception as e:
        print(f"❌ Container service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_backtest_execution():
    """Test actual backtest execution"""
    print("\nTesting backtest execution...")

    # Try worker first
    worker_available, worker_strategies = test_worker_service()

    if worker_available and worker_strategies:
        print("\n🚀 Testing worker backtest execution...")
        try:
            from services.worker_backtest_service import get_worker_backtest_service, BacktestConfig

            service = get_worker_backtest_service()

            # Use the first available strategy
            strategy_name = list(worker_strategies.keys())[0]
            print(f"Testing strategy: {strategy_name}")

            config = BacktestConfig(
                strategy_name=strategy_name,
                epic='CS.D.EURUSD.MINI.IP',
                days=3,
                timeframe='15m',
                parameters={}
            )

            result = service.run_backtest(config)

            if result.success:
                print(f"✅ Worker backtest successful!")
                print(f"  Strategy: {result.strategy_name}")
                print(f"  Signals: {result.total_signals}")
                print(f"  Execution time: {result.execution_time:.2f}s")
                if result.performance_metrics:
                    print(f"  Performance: {result.performance_metrics}")
                return True
            else:
                print(f"❌ Worker backtest failed: {result.error_message}")
                return False

        except Exception as e:
            print(f"❌ Worker backtest execution failed: {e}")
            import traceback
            traceback.print_exc()

    # Fallback to container service
    print("\n🔄 Testing container backtest execution...")
    try:
        from services.container_backtest_service import get_container_backtest_service, BacktestConfig

        service = get_container_backtest_service()
        strategies = service.get_available_strategies()

        if strategies:
            strategy_name = list(strategies.keys())[0]
            print(f"Testing strategy: {strategy_name}")

            config = BacktestConfig(
                strategy_name=strategy_name,
                epic='CS.D.EURUSD.MINI.IP',
                days=3,
                timeframe='15m',
                parameters={}
            )

            result = service.run_backtest(config)

            if result.success:
                print(f"✅ Container backtest successful!")
                print(f"  Strategy: {result.strategy_name}")
                print(f"  Signals: {result.total_signals}")
                print(f"  Execution time: {result.execution_time:.2f}s")
                if result.performance_metrics:
                    print(f"  Performance: {result.performance_metrics}")
                return True
            else:
                print(f"❌ Container backtest failed: {result.error_message}")
                return False
        else:
            print("❌ No strategies available in container service")
            return False

    except Exception as e:
        print(f"❌ Container backtest execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Testing Worker Integration")
    print("=" * 50)

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    success = True

    success &= test_imports()
    container_success, container_strategies = test_container_service()
    success &= container_success

    success &= test_backtest_execution()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The unified backtest system is working.")
    else:
        print("⚠️ Some tests failed. Check the output above.")

    return success

if __name__ == "__main__":
    main()