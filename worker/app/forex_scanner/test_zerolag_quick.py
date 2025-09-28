#!/usr/bin/env python3
"""
Quick ZEROLAG strategy test to check signal generation
"""

import pandas as pd
import numpy as np
from core.strategies.zero_lag_strategy import ZeroLagStrategy

def test_zerolag_signal_generation():
    """Test if ZEROLAG strategy can generate basic signals"""
    print("🧪 Testing ZEROLAG strategy signal generation...")

    # Create test data
    test_data = {
        'open': [1.1000] * 100,
        'high': [1.1010] * 100,
        'low': [1.0990] * 100,
        'close': [1.1005] * 100,
        'volume': [1000] * 100,
    }

    # Add some trend to make signals possible
    for i in range(50, 100):
        test_data['close'][i] = 1.1005 + (i - 50) * 0.0001
        test_data['high'][i] = test_data['close'][i] + 0.0005
        test_data['low'][i] = test_data['close'][i] - 0.0005

    df = pd.DataFrame(test_data)

    # Test strategy
    strategy = ZeroLagStrategy(epic="CS.D.EURUSD.CEEM.IP")

    try:
        signal = strategy.detect_signal(
            df=df,
            epic="CS.D.EURUSD.CEEM.IP",
            spread_pips=1.5,
            timeframe="15m"
        )

        if signal:
            print(f"✅ Signal generated!")
            print(f"   Epic: {signal.get('epic', 'MISSING')}")
            print(f"   Signal Type: {signal.get('signal_type', 'MISSING')}")
            print(f"   Confidence: {signal.get('confidence_score', 0):.1%}")
            print(f"   Strategy: {signal.get('strategy', 'MISSING')}")

            # Check required fields
            required_fields = ['epic', 'signal_type', 'confidence_score']
            missing_fields = [field for field in required_fields if field not in signal or signal[field] is None]

            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                return False
            else:
                print(f"✅ All required fields present")
                return True
        else:
            print("❌ No signal generated")
            return False

    except Exception as e:
        print(f"❌ Error during signal generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_zerolag_signal_generation()
    exit(0 if success else 1)