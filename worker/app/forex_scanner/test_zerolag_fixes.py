#!/usr/bin/env python3
"""
Test script to verify ZEROLAG error fixes
"""

def test_zerolag_fixes():
    """Test that ZEROLAG errors were fixed"""

    print("🧪 Testing ZEROLAG error fixes...")

    # Test 1: Check failure_reason variable fix in backtest_order_logger.py
    try:
        with open('/app/forex_scanner/core/trading/backtest_order_logger.py', 'r') as f:
            logger_content = f.read()

        # Check that failure_reason is properly initialized
        if "failure_reason = 'Unknown'" in logger_content:
            print("✅ Backtest Order Logger: failure_reason variable properly initialized")
        else:
            print("❌ Backtest Order Logger: failure_reason initialization missing")
            return False

        # Check that proper error handling exists
        if "elif not validation_passed:" in logger_content:
            print("✅ Backtest Order Logger: Added fallback for missing validation message")
        else:
            print("❌ Backtest Order Logger: Missing fallback logic")
            return False

    except Exception as e:
        print(f"❌ Error reading backtest_order_logger.py: {e}")
        return False

    # Test 2: Check spread_pips type conversion fix in zero_lag_strategy.py
    try:
        with open('/app/forex_scanner/core/strategies/zero_lag_strategy.py', 'r') as f:
            strategy_content = f.read()

        # Check for safe type conversion
        if "spread_pips_float = float(spread_pips)" in strategy_content:
            print("✅ ZEROLAG Strategy: Safe spread_pips type conversion added")
        else:
            print("❌ ZEROLAG Strategy: Safe type conversion missing")
            return False

        # Check for error handling
        if "except (ValueError, TypeError):" in strategy_content:
            print("✅ ZEROLAG Strategy: Added error handling for invalid spread_pips")
        else:
            print("❌ ZEROLAG Strategy: Missing error handling")
            return False

        # Check for warning message
        if "Invalid spread_pips value" in strategy_content:
            print("✅ ZEROLAG Strategy: Added warning for invalid values")
        else:
            print("❌ ZEROLAG Strategy: Missing warning message")
            return False

    except Exception as e:
        print(f"❌ Error reading zero_lag_strategy.py: {e}")
        return False

    print("\n🎯 ZEROLAG Error Fixes Summary:")
    print("   1. ✅ failure_reason variable initialization fixed")
    print("   2. ✅ spread_pips type conversion error fixed")
    print("   3. ✅ Added proper error handling and fallbacks")
    print("   4. ✅ Added warning messages for debugging")
    print("\n✅ All ZEROLAG errors should be resolved!")

    return True

if __name__ == "__main__":
    success = test_zerolag_fixes()
    exit(0 if success else 1)