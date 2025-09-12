#!/usr/bin/env python3
"""
Test script for RSI momentum confirmation filter
"""

import sys
sys.path.append('/app/forex_scanner')

import pandas as pd
import pandas_ta as ta
from datetime import datetime

# Import the RSI filter
from core.strategies.helpers.macd_rsi_filter import MACDRSIFilter

def create_test_data():
    """Create test data with known RSI conditions"""
    dates = pd.date_range('2025-01-01', periods=50, freq='15min')
    
    # Create price data that will generate specific RSI patterns
    price_data = []
    base_price = 1.16000
    
    for i in range(50):
        # Create an uptrend followed by downtrend
        if i < 25:
            # Uptrend - RSI should rise above 50
            price = base_price + (i * 0.0001)
        else:
            # Downtrend - RSI should fall below 50  
            price = base_price + (24 * 0.0001) - ((i - 25) * 0.00015)
            
        price_data.append({
            'timestamp': dates[i],
            'open': price,
            'high': price + 0.00005,
            'low': price - 0.00005,
            'close': price,
            'volume': 1000
        })
    
    df = pd.DataFrame(price_data)
    df.set_index('timestamp', inplace=True)
    
    # Calculate RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    return df

def test_rsi_filter():
    """Test the RSI filter with known conditions"""
    
    print("🧪 Testing RSI Momentum Confirmation Filter")
    print("=" * 60)
    
    # Create test data
    df = create_test_data()
    
    # Initialize filter
    rsi_filter = MACDRSIFilter()
    
    print(f"\n📊 Filter Configuration:")
    config = rsi_filter.get_filter_summary()
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n📈 Test Data Overview:")
    print(f"   Total bars: {len(df)}")
    print(f"   RSI range: {df['rsi'].min():.1f} - {df['rsi'].max():.1f}")
    print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    print(f"\n🔬 Testing Signal Validation:")
    
    # Test scenarios
    test_cases = [
        {'idx': 30, 'signal': 'BULL', 'expected': 'Should PASS - uptrend, RSI > 50'},
        {'idx': 35, 'signal': 'BULL', 'expected': 'Should FAIL - downtrend starting'},
        {'idx': 40, 'signal': 'BEAR', 'expected': 'Should PASS - downtrend, RSI < 50'},
        {'idx': 20, 'signal': 'BEAR', 'expected': 'Should FAIL - uptrend, RSI > 50'}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        idx = test_case['idx']
        signal_type = test_case['signal']
        expected = test_case['expected']
        
        if idx < len(df):
            current_rsi = df.iloc[idx]['rsi']
            previous_rsi = df.iloc[idx-1]['rsi'] if idx > 0 else current_rsi
            
            is_valid, metadata = rsi_filter.validate_signal(df, signal_type, idx, 'TEST_PAIR')
            
            status = '✅' if is_valid else '❌'
            print(f"\n   Test {i}: {signal_type} signal at bar {idx}")
            print(f"   RSI: {current_rsi:.1f} (prev: {previous_rsi:.1f}, change: {current_rsi-previous_rsi:+.1f})")
            print(f"   Result: {status} {metadata['decision']}")
            print(f"   Reason: {metadata.get('reason', 'N/A')}")
            print(f"   Expected: {expected}")
            
            # Show detailed checks
            if 'checks' in metadata:
                for check_name, check_data in metadata['checks'].items():
                    check_status = '✅' if check_data['passed'] else '❌'
                    print(f"     {check_status} {check_name}: {check_data['value']}")

def test_config_scenarios():
    """Test different configuration scenarios"""
    
    print(f"\n\n🔧 Testing Configuration Scenarios:")
    print("=" * 60)
    
    # Test with RSI filter disabled
    import config
    original_enabled = getattr(config, 'MACD_RSI_FILTER_ENABLED', True)
    config.MACD_RSI_FILTER_ENABLED = False
    
    disabled_filter = MACDRSIFilter()
    df = create_test_data()
    
    is_valid, metadata = disabled_filter.validate_signal(df, 'BULL', 30, 'TEST_PAIR')
    print(f"\n   Disabled Filter Test:")
    print(f"   Result: {'✅' if is_valid else '❌'} {metadata.get('filter', 'unknown')}")
    print(f"   Reason: {metadata.get('reason', 'N/A')}")
    
    # Restore original setting
    config.MACD_RSI_FILTER_ENABLED = original_enabled
    
    print(f"\n✅ RSI Filter Testing Complete!")

if __name__ == "__main__":
    test_rsi_filter()
    test_config_scenarios()