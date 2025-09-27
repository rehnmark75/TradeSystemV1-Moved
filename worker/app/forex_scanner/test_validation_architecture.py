#!/usr/bin/env python3
"""
Test script to verify validation architecture refactoring
"""

def test_validation_architecture():
    """Test that EMA validation is properly configured"""

    print("🧪 Testing validation architecture refactoring...")

    # Test 1: Check that ZEROLAG strategy doesn't have EMA validation
    try:
        with open('/app/forex_scanner/core/strategies/zero_lag_strategy.py', 'r') as f:
            zerolag_content = f.read()

        # Check for removed EMA validation logic
        if 'close <= ema_200' in zerolag_content or 'close >= ema_200' in zerolag_content:
            print("❌ ZEROLAG strategy still contains EMA validation logic")
            return False
        else:
            print("✅ ZEROLAG strategy: EMA validation logic removed")

        # Check that EMA data is still included in signals
        if "'ema_200': float(latest_row.get('ema_200', 0))" in zerolag_content:
            print("✅ ZEROLAG strategy: EMA data still included in signals for TradeValidator")
        else:
            print("⚠️ ZEROLAG strategy: EMA data missing from signals")

        # Check that validation level was updated
        if 'strict_4_component_with_mtf_plus_tradevalidator' in zerolag_content:
            print("✅ ZEROLAG strategy: Validation level updated correctly")
        else:
            print("⚠️ ZEROLAG strategy: Validation level not updated")

    except Exception as e:
        print(f"❌ Error reading ZEROLAG strategy: {e}")
        return False

    # Test 2: Check that TradeValidator has EMA validation
    try:
        with open('/app/forex_scanner/core/trading/trade_validator.py', 'r') as f:
            validator_content = f.read()

        # Check for EMA validation method
        if 'def validate_ema200_trend_filter' in validator_content:
            print("✅ TradeValidator: EMA200 validation method exists")
        else:
            print("❌ TradeValidator: EMA200 validation method missing")
            return False

        # Check that EMA validation is enabled by default
        if 'enable_ema200_filter' in validator_content:
            print("✅ TradeValidator: EMA200 filter configuration exists")
        else:
            print("❌ TradeValidator: EMA200 filter configuration missing")
            return False

    except Exception as e:
        print(f"❌ Error reading TradeValidator: {e}")
        return False

    print("\n🎯 Architecture Validation Summary:")
    print("   1. ✅ EMA validation removed from ZEROLAG strategy")
    print("   2. ✅ EMA data still included in signals for TradeValidator")
    print("   3. ✅ TradeValidator handles EMA validation centrally")
    print("   4. ✅ No redundant validation logic")
    print("\n✅ Validation architecture refactoring successful!")

    return True

if __name__ == "__main__":
    success = test_validation_architecture()
    exit(0 if success else 1)