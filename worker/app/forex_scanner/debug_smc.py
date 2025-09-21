#!/usr/bin/env python3
"""
Debug script to test why your problematic BUY signal passed trade validation
Run this to see exactly what's happening in Step 7 (EMA 200 filter)
"""

import sys
import os

# Add the project root to the path so we can import the modules
# Adjust this path to match your project structure
sys.path.append('/path/to/your/forex_scanner')

from core.trading.trade_validator import TradeValidator
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_your_problematic_signal():
    """Test your exact problematic signal data"""
    
    print("🔍 DEBUG: Testing your problematic BUY signal with current TradeValidator...")
    print("=" * 70)
    
    # Create the validator (same as used in production)
    validator = TradeValidator()
    
    # Your exact problematic signal data in the format it likely appears
    problematic_signal = {
        'epic': 'CS.D.EURUSD.CEEM.IP',
        'signal_type': 'BUY',  # This should be REJECTED!
        'confidence_score': 0.85,
        'strategy': 'combined_strategy',
        
        # Your actual indicator data - this is the key issue!
        'ema_data': {
            'ema_200': 171.16595997450574,  # EMA 200 = 171.166
            'ema_5': 171.0724597137261,     # Current price ≈ 171.072
            'ema_13': 171.01617800755076,
            'ema_50': 170.94254234611805,
            'ema_9': 171.03472264390382,
            'ema_21': 170.98716958816388
        },
        'macd_data': {
            'macd_line': 0.04952812159868358,
            'macd_signal': 0.04088821820603902,
            'macd_histogram': 0.00863990339264456  # Positive = MACD bullish
        },
        'other_indicators': {
            'atr': 0.15187499999999687,
            'bb_upper': 171.1754038162324,
            'bb_middle': 170.99975,
            'bb_lower': 170.8240961837676
        }
    }
    
    print("📊 SIGNAL DATA:")
    print(f"   Signal Type: {problematic_signal['signal_type']}")
    print(f"   EMA 200: {problematic_signal['ema_data']['ema_200']:.5f}")
    print(f"   EMA 5 (proxy for current price): {problematic_signal['ema_data']['ema_5']:.5f}")
    print(f"   Price > EMA 200? {problematic_signal['ema_data']['ema_5'] > problematic_signal['ema_data']['ema_200']}")
    print(f"   MACD Histogram: {problematic_signal['macd_data']['macd_histogram']:.8f} (positive = bullish)")
    print()
    
    # Test the CURRENT validator implementation step by step
    print("🔧 STEP-BY-STEP VALIDATION:")
    
    # Step 1-6: Other validations (should pass)
    print("   Steps 1-6: Basic validation...")
    
    # Step 7: EMA 200 filter - this is where the bug is!
    print("   Step 7: EMA 200 trend filter...")
    
    # Check if EMA 200 filter is enabled
    print(f"   EMA 200 filter enabled: {validator.enable_ema200_filter}")
    
    if validator.enable_ema200_filter:
        # Test the EMA 200 filter directly
        is_valid, reason = validator.validate_ema200_trend_filter(problematic_signal)
        print(f"   EMA 200 filter result: {'VALID' if is_valid else 'INVALID'}")
        print(f"   EMA 200 filter reason: {reason}")
        print()
        
        # Debug what fields the validator is actually finding
        print("🔍 DEBUGGING EMA 200 FILTER:")
        
        # Check what current price it finds
        current_price = (problematic_signal.get('current_price') or 
                        problematic_signal.get('entry_price') or 
                        problematic_signal.get('price') or 
                        problematic_signal.get('signal_price') or
                        problematic_signal.get('close_price'))
        
        print(f"   Current price found: {current_price} (from standard fields)")
        
        # Check what EMA 200 it finds
        ema_200 = problematic_signal.get('ema_200') or problematic_signal.get('ema_200_current')
        print(f"   EMA 200 found: {ema_200} (from standard fields)")
        
        # The bug: validator can't find the data because it's nested!
        if current_price is None and ema_200 is None:
            print("   🐛 BUG IDENTIFIED: Validator can't find price or EMA 200 data!")
            print("   🐛 Data is nested in 'ema_data' but validator only checks flat fields")
            print("   🐛 This causes the validator to ALLOW the signal (fail-safe mode)")
            
            # Show what happens in the current code
            print("\n   Current validator logic when data is missing:")
            print("   if current_price is None:")
            print("       return True, 'No current price data available'  # 🐛 BUG!")
            print("   if ema_200 is None:")
            print("       return True, 'No EMA 200 data available'      # 🐛 BUG!")
    
    # Full validation test
    print("\n🧪 FULL VALIDATION TEST:")
    is_valid, reason = validator.validate_signal_for_trading(problematic_signal)
    
    print(f"Final result: {'VALID' if is_valid else 'INVALID'}")
    print(f"Final reason: {reason}")
    
    # What SHOULD happen vs what DOES happen
    print("\n📊 ANALYSIS:")
    actual_price = problematic_signal['ema_data']['ema_5']  # 171.072
    actual_ema200 = problematic_signal['ema_data']['ema_200']  # 171.166
    
    print(f"   Current Price: {actual_price:.5f}")
    print(f"   EMA 200: {actual_ema200:.5f}")
    print(f"   Price > EMA 200: {actual_price > actual_ema200} (should be True for BUY)")
    print(f"   EXPECTED RESULT: INVALID (price below EMA 200 for BUY signal)")
    print(f"   ACTUAL RESULT: {'VALID' if is_valid else 'INVALID'}")
    
    if is_valid and actual_price <= actual_ema200:
        print("   🚨 VALIDATION BUG CONFIRMED: Invalid signal was allowed through!")
        return True  # Bug confirmed
    else:
        print("   ✅ Validation working correctly")
        return False  # No bug


def test_fixed_validator():
    """Test with the enhanced validator that should work correctly"""
    print("\n" + "=" * 70)
    print("🔧 TESTING WITH ENHANCED VALIDATOR (FIXED VERSION):")
    print("=" * 70)
    
    # This is the enhanced version that handles nested data
    def enhanced_validate_ema200_trend_filter(signal):
        """Enhanced version that handles your data format"""
        signal_type = signal.get('signal_type', '').upper() 
        
        # Enhanced current price detection
        current_price = None
        price_fields = ['current_price', 'entry_price', 'price', 'signal_price', 'close_price', 'ema_5']
        
        for field in price_fields:
            if field in signal and signal[field] is not None:
                current_price = float(signal[field])
                print(f"   Found current price in '{field}': {current_price:.5f}")
                break
        
        # Enhanced EMA 200 detection - handle nested data!
        ema_200 = None
        if 'ema_data' in signal and isinstance(signal['ema_data'], dict):
            ema_data = signal['ema_data']
            if 'ema_200' in ema_data:
                ema_200 = float(ema_data['ema_200'])
                print(f"   Found EMA 200 in nested ema_data: {ema_200:.5f}")
        
        # Standard field check as fallback
        if ema_200 is None:
            ema_200 = signal.get('ema_200') or signal.get('ema_200_current')
            if ema_200:
                print(f"   Found EMA 200 in standard field: {ema_200:.5f}")
        
        # CRITICAL: Reject if data is missing (don't allow!)
        if current_price is None:
            return False, "REJECTED: No current price data available"
        if ema_200 is None:
            return False, "REJECTED: No EMA 200 data available"
        
        # Apply the filter correctly
        if signal_type in ['BUY', 'BULL']:
            if current_price <= ema_200:
                return False, f"BUY REJECTED: Price {current_price:.5f} <= EMA200 {ema_200:.5f}"
            else:
                return True, f"BUY VALID: Price {current_price:.5f} > EMA200 {ema_200:.5f}"
        
        return True, "Unknown signal type"
    
    # Test the same problematic signal with enhanced validator
    problematic_signal = {
        'signal_type': 'BUY',
        'ema_data': {
            'ema_200': 171.16595997450574,  # 171.166
            'ema_5': 171.0724597137261      # 171.072 (current price proxy)
        }
    }
    
    is_valid, reason = enhanced_validate_ema200_trend_filter(problematic_signal)
    
    print(f"Enhanced validator result: {'VALID' if is_valid else 'INVALID'}")
    print(f"Enhanced validator reason: {reason}")
    
    expected_result = False  # Should be invalid (price < EMA 200 for BUY)
    if is_valid == expected_result:
        print("   ✅ Enhanced validator working correctly!")
    else:
        print("   ❌ Enhanced validator still has issues")
    
    return is_valid == expected_result


def main():
    """Main debug function"""
    print("🔍 TRADE VALIDATOR DEBUG - Why Your BUY Signal Passed")
    print("=" * 70)
    
    # Test current validator
    bug_confirmed = test_your_problematic_signal()
    
    # Test enhanced validator
    fix_works = test_fixed_validator()
    
    print("\n" + "=" * 70)
    print("📋 SUMMARY:")
    print("=" * 70)
    
    if bug_confirmed:
        print("🐛 BUG CONFIRMED: Current validator allows invalid BUY signals")
        print("   Root cause: Can't find price/EMA data in nested structure")
        print("   Current behavior: Allows signal when data is missing")
        print("   Should be: Reject signal when data is missing")
    else:
        print("✅ Current validator appears to be working correctly")
    
    if fix_works:
        print("✅ Enhanced validator correctly handles nested data format")
        print("   Fix: Looks for data in 'ema_data' structure")
        print("   Fix: Rejects signals when data is missing")
        print("   Fix: Properly validates price vs EMA 200 relationship")
    else:
        print("❌ Enhanced validator needs more work")
    
    print("\n🛠️ RECOMMENDED ACTIONS:")
    if bug_confirmed:
        print("1. Update validate_ema200_trend_filter() method in trade_validator.py")
        print("2. Add support for nested 'ema_data' structure")
        print("3. Change fail-safe behavior: reject when data missing")
        print("4. Test with your actual signal format")
        print("5. Add logging to see what data is actually found")
    
    print("\n🧪 To test this fix:")
    print("1. Apply the enhanced validate_ema200_trend_filter() method")
    print("2. Run your signal through the validator again")
    print("3. It should now REJECT the BUY signal (price < EMA 200)")


if __name__ == "__main__":
    main()