#!/usr/bin/env python3
"""
Test script for MACD Zero Line Filter
"""

import sys
sys.path.append('/app/forex_scanner')

import pandas as pd
from datetime import datetime
from core.strategies.helpers.macd_zero_line_filter import MACDZeroLineFilter

def create_test_data():
    """Create test data with different MACD line scenarios"""
    
    test_scenarios = [
        {
            'name': 'Bull Signal - Both Lines Below Zero (SHOULD PASS)',
            'macd_line': -0.001,
            'macd_signal': -0.0005,
            'signal_type': 'BULL',
            'expected': True
        },
        {
            'name': 'Bull Signal - Both Lines Above Zero (SHOULD FAIL)',
            'macd_line': 0.001,
            'macd_signal': 0.0005,
            'signal_type': 'BULL',
            'expected': False
        },
        {
            'name': 'Bull Signal - Mixed Lines (MACD below, Signal above)',
            'macd_line': -0.001,
            'macd_signal': 0.0005,
            'signal_type': 'BULL',
            'expected': True  # Should pass in non-strict mode (either line counts)
        },
        {
            'name': 'Bear Signal - Both Lines Above Zero (SHOULD PASS)',
            'macd_line': 0.001,
            'macd_signal': 0.0005,
            'signal_type': 'BEAR',
            'expected': True
        },
        {
            'name': 'Bear Signal - Both Lines Below Zero (SHOULD FAIL)',
            'macd_line': -0.001,
            'macd_signal': -0.0005,
            'signal_type': 'BEAR',
            'expected': False
        },
        {
            'name': 'Bear Signal - Mixed Lines (MACD above, Signal below)',
            'macd_line': 0.001,
            'macd_signal': -0.0005,
            'signal_type': 'BEAR',
            'expected': True  # Should pass in non-strict mode (either line counts)
        }
    ]
    
    return test_scenarios

def test_zero_line_filter():
    """Test the Zero Line filter with various scenarios"""
    
    print("🧪 Testing MACD Zero Line Filter")
    print("=" * 60)
    
    # Initialize filter
    zero_line_filter = MACDZeroLineFilter()
    
    print(f"\n📊 Filter Configuration:")
    config = zero_line_filter.get_filter_summary()
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Get test scenarios
    scenarios = create_test_data()
    
    print(f"\n🔬 Testing {len(scenarios)} Scenarios:")
    print("-" * 50)
    
    passed_tests = 0
    total_tests = len(scenarios)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Test {i}: {scenario['name']}")
        
        # Create DataFrame with test data
        df_data = {
            'timestamp': [datetime.now()],
            'macd_line': [scenario['macd_line']],
            'macd_signal': [scenario['macd_signal']],
            'close': [1.2345]  # Dummy close price
        }
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        # Test the filter
        is_valid, metadata = zero_line_filter.validate_signal(
            df, 
            scenario['signal_type'], 
            0,  # Current index
            'TEST_PAIR'
        )
        
        # Check result
        expected = scenario['expected']
        test_passed = is_valid == expected
        
        if test_passed:
            passed_tests += 1
            
        status = '✅ PASS' if test_passed else '❌ FAIL'
        print(f"   Result: {status}")
        print(f"   Expected: {'APPROVED' if expected else 'REJECTED'}")
        print(f"   Got: {'APPROVED' if is_valid else 'REJECTED'}")
        print(f"   MACD Line: {scenario['macd_line']:+.6f}")
        print(f"   Signal Line: {scenario['macd_signal']:+.6f}")
        print(f"   Reason: {metadata.get('reason', 'N/A')}")
        print(f"   Quality Score: {metadata.get('quality_score', 'N/A')}")
        
        # Show detailed checks
        if 'checks' in metadata:
            print(f"   Detailed Checks:")
            for check_name, check_data in metadata['checks'].items():
                check_status = '✅' if check_data['passed'] else '❌'
                print(f"     {check_status} {check_name}: {check_data['value']}")
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print(f"   ✅ All tests PASSED! Zero Line Filter is working correctly.")
    else:
        print(f"   ⚠️ Some tests FAILED. Review filter logic.")
    
    return passed_tests == total_tests

def test_configuration_scenarios():
    """Test different configuration scenarios"""
    
    print(f"\n\n🔧 Testing Configuration Scenarios:")
    print("=" * 60)
    
    # Test with filter disabled
    import config
    original_enabled = getattr(config, 'MACD_ZERO_LINE_FILTER_ENABLED', True)
    config.MACD_ZERO_LINE_FILTER_ENABLED = False
    
    disabled_filter = MACDZeroLineFilter()
    df = pd.DataFrame({
        'macd_line': [0.001],
        'macd_signal': [0.0005]
    })
    
    is_valid, metadata = disabled_filter.validate_signal(df, 'BULL', 0, 'TEST_PAIR')
    print(f"\n   Disabled Filter Test:")
    print(f"   Result: {'✅' if is_valid else '❌'} {metadata.get('filter', 'unknown')}")
    print(f"   Reason: {metadata.get('reason', 'N/A')}")
    
    # Restore original setting
    config.MACD_ZERO_LINE_FILTER_ENABLED = original_enabled
    
    print(f"\n✅ Zero Line Filter Testing Complete!")

if __name__ == "__main__":
    success = test_zero_line_filter()
    test_configuration_scenarios()
    
    if success:
        print(f"\n🎉 ZERO LINE FILTER IMPLEMENTATION SUCCESSFUL!")
        print(f"   The filter is ready for integration into MACD signal detection.")
    else:
        print(f"\n⚠️ Some tests failed. Please review the implementation.")