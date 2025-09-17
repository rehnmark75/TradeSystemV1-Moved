#!/usr/bin/env python3
"""
Test Overextension Filters
Quick test to verify the overextension filter implementation is working
"""

import sys
sys.path.append('/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner')

# Test the configuration imports
try:
    from configdata.strategies.config_ema_strategy import (
        check_stochastic_overextension,
        check_williams_r_overextension,
        check_rsi_extreme_overextension,
        calculate_composite_overextension_score,
        STOCHASTIC_OVEREXTENSION_ENABLED,
        WILLIAMS_R_OVEREXTENSION_ENABLED,
        RSI_EXTREME_OVEREXTENSION_ENABLED,
        COMPOSITE_OVEREXTENSION_ENABLED,
        STOCHASTIC_EXTREME_OVERBOUGHT,
        STOCHASTIC_EXTREME_OVERSOLD,
        WILLIAMS_R_EXTREME_OVERBOUGHT,
        WILLIAMS_R_EXTREME_OVERSOLD,
        RSI_EXTREME_OVERBOUGHT,
        RSI_EXTREME_OVERSOLD
    )
    print("✅ Configuration imports successful")
except Exception as e:
    print(f"❌ Configuration import failed: {e}")
    sys.exit(1)

# Print current configuration
print(f"\n📊 Current Overextension Filter Configuration:")
print(f"   Stochastic enabled: {STOCHASTIC_OVEREXTENSION_ENABLED}")
print(f"   Williams %R enabled: {WILLIAMS_R_OVEREXTENSION_ENABLED}")
print(f"   RSI extreme enabled: {RSI_EXTREME_OVEREXTENSION_ENABLED}")
print(f"   Composite enabled: {COMPOSITE_OVEREXTENSION_ENABLED}")
print(f"\n🎯 Thresholds:")
print(f"   Stochastic: {STOCHASTIC_EXTREME_OVERBOUGHT}/{STOCHASTIC_EXTREME_OVERSOLD}")
print(f"   Williams %R: {WILLIAMS_R_EXTREME_OVERBOUGHT}/{WILLIAMS_R_EXTREME_OVERSOLD}")
print(f"   RSI: {RSI_EXTREME_OVERBOUGHT}/{RSI_EXTREME_OVERSOLD}")

# Test cases with values that should trigger filters
test_cases = [
    {
        'name': 'Extremely Overbought',
        'stoch_k': 85.0,
        'stoch_d': 83.0,
        'williams_r': -10.0,
        'rsi': 85.0,
        'signal_direction': 'long'
    },
    {
        'name': 'Extremely Oversold',
        'stoch_k': 15.0,
        'stoch_d': 17.0,
        'williams_r': -90.0,
        'rsi': 15.0,
        'signal_direction': 'short'
    },
    {
        'name': 'Normal Conditions',
        'stoch_k': 50.0,
        'stoch_d': 52.0,
        'williams_r': -50.0,
        'rsi': 50.0,
        'signal_direction': 'long'
    }
]

print(f"\n🧪 Testing Overextension Filters:")
print("=" * 60)

for test_case in test_cases:
    print(f"\n📋 Test Case: {test_case['name']}")
    print(f"   Values: Stoch({test_case['stoch_k']:.1f}), Williams({test_case['williams_r']:.1f}), RSI({test_case['rsi']:.1f})")
    print(f"   Signal Direction: {test_case['signal_direction']}")

    # Test individual filters
    stoch_result = check_stochastic_overextension(
        test_case['stoch_k'], test_case['stoch_d'], test_case['signal_direction']
    )
    williams_result = check_williams_r_overextension(
        test_case['williams_r'], test_case['signal_direction']
    )
    rsi_result = check_rsi_extreme_overextension(
        test_case['rsi'], test_case['signal_direction']
    )

    print(f"   🔍 Individual Results:")
    print(f"      Stochastic: {'🚫 OVEREXTENDED' if stoch_result['overextended'] else '✅ OK'} (penalty: {stoch_result['penalty']:.3f})")
    print(f"      Williams %R: {'🚫 OVEREXTENDED' if williams_result['overextended'] else '✅ OK'} (penalty: {williams_result['penalty']:.3f})")
    print(f"      RSI: {'🚫 OVEREXTENDED' if rsi_result['overextended'] else '✅ OK'} (penalty: {rsi_result['penalty']:.3f})")

    # Test composite scoring
    composite_result = calculate_composite_overextension_score(
        test_case['stoch_k'], test_case['stoch_d'],
        test_case['williams_r'], test_case['rsi'],
        test_case['signal_direction']
    )

    print(f"   🎯 Composite Result:")
    print(f"      Triggered Indicators: {composite_result['indicators_triggered']}/3")
    print(f"      Composite Overextended: {'🚫 YES' if composite_result['composite_overextended'] else '✅ NO'}")
    print(f"      Total Penalty: {composite_result['total_penalty']:.3f}")
    print(f"      Recommendation: {composite_result['recommendation']}")

print(f"\n✅ Overextension filter test completed!")
print(f"💡 If you see penalties/blocks above, the filters are working correctly.")