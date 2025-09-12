#!/usr/bin/env python3
"""
Test script to verify Two-Pole Oscillator validation fixes
Tests the specific case that was failing: BULL signal with purple Two-Pole
"""

import sys
import os
sys.path.append('/datadrive/Trader/TradeSystemV1/worker/app')

import pandas as pd
import numpy as np
from datetime import datetime

def test_two_pole_validation():
    """Test the Two-Pole Oscillator validation fix"""
    
    print("🔧 Testing Two-Pole Oscillator validation fixes...")
    
    # Import the fixed modules
    from forex_scanner.core.strategies.helpers.ema_trend_validator import EMATrendValidator
    from forex_scanner.configdata import config
    
    print(f"✅ TWO_POLE_OSCILLATOR_ENABLED: {getattr(config, 'TWO_POLE_OSCILLATOR_ENABLED', 'NOT FOUND')}")
    
    # Create validator instance
    validator = EMATrendValidator()
    
    # Test Case 1: Missing Two-Pole data (should block signal)
    print("\n🧪 Test Case 1: Missing Two-Pole Data")
    missing_data_row = pd.Series({
        'close': 1.3527,
        'ema_short': 1.3520,
        'ema_long': 1.3499,
        'ema_trend': 1.3460
        # No two_pole_is_green or two_pole_is_purple
    })
    
    result = validator.validate_two_pole_color(missing_data_row, 'BULL')
    print(f"   Missing data validation result: {result} (should be False)")
    assert result == False, "Missing Two-Pole data should block signal"
    print("   ✅ PASSED: Missing data correctly blocked")
    
    # Test Case 2: Purple Two-Pole for BULL (should block)
    print("\n🧪 Test Case 2: Purple Two-Pole for BULL Signal")
    purple_bull_row = pd.Series({
        'close': 1.3527,
        'ema_short': 1.3520,
        'ema_long': 1.3499, 
        'ema_trend': 1.3460,
        'two_pole_osc': -0.1,
        'two_pole_is_green': False,
        'two_pole_is_purple': True
    })
    
    result = validator.validate_two_pole_color(purple_bull_row, 'BULL')
    print(f"   Purple Two-Pole BULL validation result: {result} (should be False)")
    assert result == False, "Purple Two-Pole should block BULL signal"
    print("   ✅ PASSED: Purple Two-Pole correctly blocked BULL signal")
    
    # Test Case 3: Green Two-Pole for BULL (should allow)  
    print("\n🧪 Test Case 3: Green Two-Pole for BULL Signal")
    green_bull_row = pd.Series({
        'close': 1.3527,
        'ema_short': 1.3520,
        'ema_long': 1.3499,
        'ema_trend': 1.3460,
        'two_pole_osc': -0.1,
        'two_pole_is_green': True,
        'two_pole_is_purple': False
    })
    
    result = validator.validate_two_pole_color(green_bull_row, 'BULL')
    print(f"   Green Two-Pole BULL validation result: {result} (should be True)")
    assert result == True, "Green Two-Pole should allow BULL signal"
    print("   ✅ PASSED: Green Two-Pole correctly allowed BULL signal")
    
    # Test Case 4: Green Two-Pole for BEAR (should block)
    print("\n🧪 Test Case 4: Green Two-Pole for BEAR Signal")
    green_bear_row = pd.Series({
        'close': 1.3527,
        'ema_short': 1.3520,
        'ema_long': 1.3499,
        'ema_trend': 1.3460,
        'two_pole_osc': 0.1,
        'two_pole_is_green': True,
        'two_pole_is_purple': False
    })
    
    result = validator.validate_two_pole_color(green_bear_row, 'BEAR')
    print(f"   Green Two-Pole BEAR validation result: {result} (should be False)")
    assert result == False, "Green Two-Pole should block BEAR signal"
    print("   ✅ PASSED: Green Two-Pole correctly blocked BEAR signal")
    
    # Test Case 5: Purple Two-Pole for BEAR (should allow)
    print("\n🧪 Test Case 5: Purple Two-Pole for BEAR Signal")
    purple_bear_row = pd.Series({
        'close': 1.3527,
        'ema_short': 1.3520,
        'ema_long': 1.3499,
        'ema_trend': 1.3460,
        'two_pole_osc': 0.1,
        'two_pole_is_green': False,
        'two_pole_is_purple': True
    })
    
    result = validator.validate_two_pole_color(purple_bear_row, 'BEAR')
    print(f"   Purple Two-Pole BEAR validation result: {result} (should be True)")
    assert result == True, "Purple Two-Pole should allow BEAR signal"
    print("   ✅ PASSED: Purple Two-Pole correctly allowed BEAR signal")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Two-Pole Oscillator validation is working correctly")
    print("❌ BULL signals will be BLOCKED when Two-Pole is purple")
    print("❌ Missing Two-Pole data will BLOCK all signals for safety")
    
    return True

if __name__ == "__main__":
    try:
        test_two_pole_validation()
        print("\n🎯 Fix Status: SUCCESSFUL")
        print("The bug that allowed BULL trades with purple Two-Pole has been FIXED.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)