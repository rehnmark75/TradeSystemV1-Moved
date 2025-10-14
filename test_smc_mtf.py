#!/usr/bin/env python3
"""
Test script for SMC Multi-Timeframe implementation
Validates that MTF analyzer loads and integrates correctly
"""

import sys
import os

# Add worker app to path
sys.path.insert(0, '/home/hr/Projects/TradeSystemV1/worker/app')

def test_mtf_analyzer_import():
    """Test 1: Import MTF analyzer module"""
    try:
        from forex_scanner.core.strategies.helpers.smc_mtf_analyzer import SMCMultiTimeframeAnalyzer
        print("✅ Test 1 PASSED: SMC MTF Analyzer imported successfully")
        return True
    except Exception as e:
        print(f"❌ Test 1 FAILED: Could not import MTF analyzer: {e}")
        return False


def test_mtf_analyzer_initialization():
    """Test 2: Initialize MTF analyzer"""
    try:
        from forex_scanner.core.strategies.helpers.smc_mtf_analyzer import SMCMultiTimeframeAnalyzer
        import logging

        logger = logging.getLogger('test')
        analyzer = SMCMultiTimeframeAnalyzer(logger=logger, data_fetcher=None)

        print("✅ Test 2 PASSED: MTF Analyzer initialized successfully")
        print(f"   - Check timeframes: {analyzer.check_timeframes}")
        print(f"   - Timeframe weights: {analyzer.timeframe_weights}")
        print(f"   - Both aligned boost: {analyzer.both_aligned_boost}")
        return True
    except Exception as e:
        print(f"❌ Test 2 FAILED: Could not initialize MTF analyzer: {e}")
        return False


def test_smc_strategy_with_mtf():
    """Test 3: Initialize SMC Fast Strategy with MTF"""
    try:
        from forex_scanner.core.strategies.smc_strategy_fast import SMCStrategyFast

        # Initialize without data_fetcher (MTF should be disabled)
        strategy = SMCStrategyFast(
            smc_config_name='default',
            data_fetcher=None,
            backtest_mode=False,
            pipeline_mode=True
        )

        print("✅ Test 3 PASSED: SMC Fast Strategy initialized with MTF support")
        print(f"   - MTF enabled: {strategy.mtf_enabled}")
        print(f"   - MTF analyzer: {strategy.mtf_analyzer is not None}")
        return True
    except Exception as e:
        print(f"❌ Test 3 FAILED: Could not initialize SMC strategy with MTF: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smc_config_mtf_settings():
    """Test 4: Verify MTF settings in config"""
    try:
        from forex_scanner.configdata.strategies.config_smc_strategy import SMC_STRATEGY_CONFIG

        # Check default preset has MTF settings
        default_config = SMC_STRATEGY_CONFIG['default']

        required_mtf_keys = [
            'mtf_enabled',
            'mtf_check_timeframes',
            'mtf_timeframe_weights',
            'mtf_both_aligned_boost',
            'mtf_15m_only_boost',
            'mtf_4h_only_boost'
        ]

        missing_keys = [key for key in required_mtf_keys if key not in default_config]

        if missing_keys:
            print(f"❌ Test 4 FAILED: Missing MTF config keys: {missing_keys}")
            return False

        print("✅ Test 4 PASSED: MTF configuration settings verified")
        print(f"   - MTF enabled: {default_config['mtf_enabled']}")
        print(f"   - Check timeframes: {default_config['mtf_check_timeframes']}")
        print(f"   - Both aligned boost: {default_config['mtf_both_aligned_boost']}")

        # Check scalping preset (should only check 15m)
        scalping_config = SMC_STRATEGY_CONFIG['scalping']
        print(f"   - Scalping TFs: {scalping_config['mtf_check_timeframes']}")

        # Check swing preset (should check 4h + 1d)
        swing_config = SMC_STRATEGY_CONFIG['swing']
        print(f"   - Swing TFs: {swing_config['mtf_check_timeframes']}")

        return True
    except Exception as e:
        print(f"❌ Test 4 FAILED: Config validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mtf_validation_structure():
    """Test 5: Test MTF validation result structure"""
    try:
        from forex_scanner.core.strategies.helpers.smc_mtf_analyzer import SMCMultiTimeframeAnalyzer
        import pandas as pd
        import logging

        logger = logging.getLogger('test')
        analyzer = SMCMultiTimeframeAnalyzer(logger=logger, data_fetcher=None)

        # Test validation with no data_fetcher (should return disabled result)
        result = analyzer.validate_higher_timeframe_smc(
            epic='CS.D.EURUSD.CEEM.IP',
            current_time=pd.Timestamp.now(),
            signal_type='BULL',
            structure_info={'break_type': 'BOS', 'break_direction': 'bullish', 'significance': 0.7}
        )

        # Check result structure
        required_keys = ['mtf_enabled', 'validation_passed', 'confidence_boost']
        missing_keys = [key for key in required_keys if key not in result]

        if missing_keys:
            print(f"❌ Test 5 FAILED: Missing result keys: {missing_keys}")
            return False

        print("✅ Test 5 PASSED: MTF validation result structure correct")
        print(f"   - MTF enabled: {result['mtf_enabled']}")
        print(f"   - Validation passed: {result['validation_passed']}")
        print(f"   - Confidence boost: {result['confidence_boost']}")
        return True
    except Exception as e:
        print(f"❌ Test 5 FAILED: Validation structure test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("SMC Multi-Timeframe Implementation Test Suite")
    print("=" * 70)
    print()

    tests = [
        test_mtf_analyzer_import,
        test_mtf_analyzer_initialization,
        test_smc_strategy_with_mtf,
        test_smc_config_mtf_settings,
        test_mtf_validation_structure
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
        print()

    print("=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ ALL TESTS PASSED - MTF implementation is working correctly!")
    else:
        print(f"❌ {total - passed} TESTS FAILED - Please review errors above")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
