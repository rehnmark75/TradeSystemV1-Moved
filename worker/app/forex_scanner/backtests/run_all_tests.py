#!/usr/bin/env python3
"""
Master Test Runner for Enhanced Backtest System
Runs all test suites and provides comprehensive system validation
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
sys.path.insert(0, project_root)


def run_comprehensive_tests():
    """Run all test suites for the enhanced backtest system"""
    print("🚀 Enhanced Backtest System - Comprehensive Test Suite")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    test_results = {}
    total_start_time = time.time()

    # Test 1: Core System Tests
    print("📋 Test Suite 1: Core System Functionality")
    print("-" * 40)
    try:
        from test_enhanced_backtest_system import run_tests
        test_results['core_system'] = run_tests()
    except Exception as e:
        print(f"❌ Core system tests failed to run: {e}")
        test_results['core_system'] = False

    print("\n")

    # Test 2: Integration Tests
    print("🔗 Test Suite 2: System Integration")
    print("-" * 40)
    try:
        from test_integration import run_integration_tests
        test_results['integration'] = run_integration_tests()
    except Exception as e:
        print(f"❌ Integration tests failed to run: {e}")
        test_results['integration'] = False

    print("\n")

    # Test 3: Manual Validation Tests
    print("🔍 Test Suite 3: Manual Validation")
    print("-" * 40)
    test_results['manual_validation'] = run_manual_validation_tests()

    print("\n")

    # Test 4: Performance Tests
    print("⚡ Test Suite 4: Performance Validation")
    print("-" * 40)
    test_results['performance'] = run_performance_tests()

    # Calculate total execution time
    total_time = time.time() - total_start_time

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 60)

    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)

    for suite_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {suite_name.replace('_', ' ').title()}: {status}")

    print(f"\n🎯 Overall Success Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})")
    print(f"⏱️ Total Execution Time: {total_time:.2f} seconds")

    # Final recommendations
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - Enhanced Backtest System is ready for production!")
    elif passed_tests >= total_tests * 0.75:
        print("\n⚠️ Most tests passed - System is functional with minor issues")
    else:
        print("\n🚨 Multiple test failures - System requires attention before deployment")

    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return passed_tests == total_tests


def run_manual_validation_tests():
    """Run manual validation tests"""
    print("Running manual validation checks...")

    checks_passed = 0
    total_checks = 0

    # Check 1: Required files exist
    total_checks += 1
    print("🔍 Checking required files exist...")
    required_files = [
        'backtest_base.py',
        'parameter_manager.py',
        '../core/market_intelligence.py',
        'migration_utility.py'
    ]

    files_exist = True
    for file_path in required_files:
        full_path = os.path.join(script_dir, file_path)
        if os.path.exists(full_path):
            print(f"   ✅ {file_path} - Found")
        else:
            print(f"   ❌ {file_path} - Missing")
            files_exist = False

    if files_exist:
        checks_passed += 1

    # Check 2: Enhanced strategies exist
    total_checks += 1
    print("\n🔍 Checking enhanced strategy files...")
    enhanced_strategies = [
        'backtest_ema_enhanced.py',
        'backtest_macd_enhanced.py',
        'backtest_ichimoku_enhanced.py'
    ]

    enhanced_exist = 0
    for strategy_file in enhanced_strategies:
        full_path = os.path.join(script_dir, strategy_file)
        if os.path.exists(full_path):
            print(f"   ✅ {strategy_file} - Found")
            enhanced_exist += 1
        else:
            print(f"   ⚠️ {strategy_file} - Not found (this is expected if not migrated)")

    # At least some enhanced strategies should exist
    if enhanced_exist > 0:
        checks_passed += 1
        print(f"   ✅ {enhanced_exist} enhanced strategies found")

    # Check 3: Import validation
    total_checks += 1
    print("\n🔍 Checking core imports...")
    try:
        from backtests.backtest_base import BacktestBase, StandardSignal, StandardBacktestResult
        from backtests.parameter_manager import ParameterManager
        print("   ✅ Core imports successful")
        checks_passed += 1
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")

    # Check 4: Configuration validation
    total_checks += 1
    print("\n🔍 Checking configuration...")
    try:
        import config
        if hasattr(config, 'EPIC_LIST'):
            print("   ✅ EPIC_LIST configuration found")
            checks_passed += 1
        else:
            print("   ⚠️ EPIC_LIST not found in config")
    except ImportError:
        print("   ⚠️ Config module not accessible")

    success_rate = checks_passed / total_checks
    print(f"\n📊 Manual validation: {checks_passed}/{total_checks} checks passed ({success_rate:.1%})")

    return success_rate >= 0.75


def run_performance_tests():
    """Run basic performance validation tests"""
    print("Running performance validation...")

    # Test data creation performance
    start_time = time.time()
    try:
        import pandas as pd
        import numpy as np

        # Generate test data
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='15min')
        test_data = pd.DataFrame({
            'open': np.random.uniform(1.09, 1.11, 1000),
            'high': np.random.uniform(1.10, 1.12, 1000),
            'low': np.random.uniform(1.08, 1.10, 1000),
            'close': np.random.uniform(1.09, 1.11, 1000),
        }, index=dates)

        data_creation_time = time.time() - start_time
        print(f"   ✅ Test data creation: {data_creation_time:.3f}s")

        # Test backtest base initialization
        start_time = time.time()
        from backtests.backtest_base import BacktestBase

        class TestBacktest(BacktestBase):
            def initialize_strategy(self, epic=None):
                pass
            def run_strategy_backtest(self, df, epic, spread_pips, timeframe):
                return []

        backtest = TestBacktest("performance_test")
        init_time = time.time() - start_time
        print(f"   ✅ BacktestBase initialization: {init_time:.3f}s")

        # Performance thresholds
        if data_creation_time < 1.0 and init_time < 0.5:
            print("   ✅ Performance metrics within acceptable thresholds")
            return True
        else:
            print("   ⚠️ Performance metrics exceed thresholds")
            return False

    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False


def validate_system_readiness():
    """Quick system readiness check"""
    print("🎯 Quick System Readiness Check")
    print("-" * 30)

    readiness_score = 0
    max_score = 4

    # Check 1: Docker environment
    if 'DOCKER' in os.environ or os.path.exists('/.dockerenv'):
        print("   ✅ Running in Docker environment")
        readiness_score += 1
    else:
        print("   ⚠️ Not running in Docker (some tests may fail)")

    # Check 2: Core modules accessible
    try:
        import config
        print("   ✅ Configuration accessible")
        readiness_score += 1
    except ImportError:
        print("   ❌ Configuration not accessible")

    # Check 3: Database dependencies
    try:
        from backtests.parameter_manager import ParameterManager
        print("   ✅ Parameter management available")
        readiness_score += 1
    except Exception:
        print("   ⚠️ Parameter management may have issues")

    # Check 4: Data processing libraries
    try:
        import pandas as pd
        import numpy as np
        print("   ✅ Data processing libraries available")
        readiness_score += 1
    except ImportError:
        print("   ❌ Data processing libraries missing")

    readiness_percentage = (readiness_score / max_score) * 100
    print(f"\n📊 System Readiness: {readiness_score}/{max_score} ({readiness_percentage:.0f}%)")

    return readiness_percentage >= 75


if __name__ == "__main__":
    print("Starting comprehensive enhanced backtest system validation...\n")

    # Quick readiness check
    if not validate_system_readiness():
        print("\n⚠️ System readiness issues detected. Tests may fail.")
        print("Consider running within Docker environment for best results.")

    print("\n")

    # Run all tests
    success = run_comprehensive_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)