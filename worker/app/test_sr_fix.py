#!/usr/bin/env python3
"""
Test the S/R Validator fix for detecting price AT support/resistance levels
"""
import sys
import numpy as np
import pandas as pd
import logging

# Direct import to avoid config loading issues
sys.path.insert(0, '/app')

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Import just the validator class directly
from forex_scanner.core.detection.support_resistance_validator import SupportResistanceValidator

def test_sell_at_support():
    """Test SELL signal AT a support level"""
    print('🧪 Testing S/R Validator Fix - Scenario: SELL at Support Level')
    print('=' * 70)

    # Create realistic price data with a support level at 0.6340
    dates = pd.date_range('2024-09-28 08:00', periods=200, freq='15min')

    price_data = []
    for i in range(200):
        if i < 100:
            # Earlier period - consolidation around 0.6340 (support formation)
            price = 0.6340 + np.random.normal(0, 0.0010)
        elif i < 150:
            # Price drops and bounces
            price = 0.6290 + np.random.normal(0, 0.0008)
        else:
            # Price rallies back toward the 0.6340 support
            progress = (i - 150) / 50
            price = 0.6290 + (0.6345 - 0.6290) * progress + np.random.normal(0, 0.0005)

        price_data.append(price)

    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'open': price_data,
        'high': [p + abs(np.random.normal(0, 0.0003)) for p in price_data],
        'low': [p - abs(np.random.normal(0, 0.0003)) for p in price_data],
        'close': price_data,
        'volume': np.random.randint(1000, 5000, 200)
    })

    # Create validator
    validator = SupportResistanceValidator(
        left_bars=15,
        right_bars=15,
        level_tolerance_pips=5.0,
        min_level_distance_pips=10.0
    )

    print(f'✅ Validator initialized: {validator.get_validation_summary()}')
    print()

    # Test SELL signal at 0.6345 (AT the support level from yesterday)
    test_signal = {
        'signal_type': 'SELL',
        'current_price': 0.6345,  # Right at the historical support
        'epic': 'CS.D.NZDUSD.MINI.IP'
    }

    print(f'📊 Test Signal: {test_signal["signal_type"]} @ {test_signal["current_price"]:.4f}')
    print()

    # Validate
    is_valid, reason, details = validator.validate_trade_direction(
        test_signal, df, test_signal['epic']
    )

    print(f'Result: {"✅ VALID" if is_valid else "❌ INVALID"}')
    print(f'Reason: {reason}')
    print()

    if details.get('support_levels'):
        print(f'📍 Support Levels Found: {[f"{s:.4f}" for s in details["support_levels"][:3]]}')
    if details.get('resistance_levels'):
        print(f'📍 Resistance Levels Found: {[f"{r:.4f}" for r in details["resistance_levels"][:3]]}')
    if details.get('nearest_support'):
        print(f'📍 Nearest Support: {details["nearest_support"]:.4f}')
    if details.get('nearest_resistance'):
        print(f'📍 Nearest Resistance: {details["nearest_resistance"]:.4f}')

    print()
    print('=' * 70)
    if not is_valid:
        print('✅ FIX WORKING: SELL signal correctly REJECTED at support level!')
        return True
    else:
        print('⚠️ WARNING: SELL signal was allowed at support level')
        return False


if __name__ == '__main__':
    success = test_sell_at_support()
    sys.exit(0 if success else 1)