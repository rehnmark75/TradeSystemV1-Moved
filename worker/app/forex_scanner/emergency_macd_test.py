#!/usr/bin/env python3
"""
EMERGENCY MACD STRATEGY TEST
Test the optimized MACD strategy to diagnose why no signals are being generated
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_data():
    """Create synthetic MACD data with known crossovers for testing"""
    # Create 200 bars of test data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='15min')

    # Create realistic price data
    np.random.seed(42)  # For reproducible results
    base_price = 1.1000
    price_changes = np.random.normal(0, 0.0001, 200)
    prices = [base_price]

    for change in price_changes[1:]:
        prices.append(prices[-1] + change)

    # Create OHLC data
    df = pd.DataFrame({
        'start_time': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.00005)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.00005)) for p in prices],
        'close': prices,
        'ltv': np.random.randint(1000, 10000, 200)
    })

    # Calculate EMAs for MACD
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # Calculate MACD
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']

    # Force some crossovers by manipulating the histogram
    # Add a bull crossover at index 100
    df.loc[99, 'macd_histogram'] = -0.00005
    df.loc[100, 'macd_histogram'] = 0.00005

    # Add a bear crossover at index 150
    df.loc[149, 'macd_histogram'] = 0.00005
    df.loc[150, 'macd_histogram'] = -0.00005

    print(f"✅ Created test data with {len(df)} bars")
    print(f"📊 MACD histogram range: {df['macd_histogram'].min():.8f} to {df['macd_histogram'].max():.8f}")
    print(f"📊 Forced crossovers at bars 100 (bull) and 150 (bear)")

    return df

def test_macd_strategy():
    """Test the MACD strategy with synthetic data"""
    try:
        # Import the strategy
        from core.strategies.macd_strategy import MACDStrategy

        print("🚨 EMERGENCY MACD STRATEGY TEST STARTING...")

        # Create test data
        df = create_test_data()

        # Initialize strategy
        strategy = MACDStrategy(
            epic='CS.D.EURUSD.CEEM.IP',
            timeframe='15m',
            backtest_mode=True
        )

        print(f"✅ Strategy initialized: {strategy.name}")

        # Test signal detection
        print("\n🔍 Testing signal detection...")

        result = strategy.detect_signal(
            df=df,
            epic='CS.D.EURUSD.CEEM.IP',
            spread_pips=1.5,
            timeframe='15m'
        )

        if result:
            print(f"✅ SIGNAL DETECTED!")
            print(f"   Type: {result.get('signal_type', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 'unknown')}")
            print(f"   Price: {result.get('price', 'unknown')}")
            print(f"   Full result: {result}")
        else:
            print("❌ NO SIGNAL DETECTED")

        # Test just the histogram crossover detection
        print("\n🔍 Testing histogram crossover detection directly...")

        df_with_signals = strategy.indicator_calculator.detect_macd_crossovers(
            df, 'CS.D.EURUSD.CEEM.IP', is_backtest=True
        )

        bull_signals = df_with_signals.get('bull_alert', pd.Series(False, index=df_with_signals.index)).sum()
        bear_signals = df_with_signals.get('bear_alert', pd.Series(False, index=df_with_signals.index)).sum()

        print(f"📊 Direct crossover test: {bull_signals} bull signals, {bear_signals} bear signals")

        if bull_signals > 0 or bear_signals > 0:
            print("✅ Crossovers detected in raw analysis!")
            # Show where
            bull_indices = df_with_signals[df_with_signals.get('bull_alert', False)].index.tolist()
            bear_indices = df_with_signals[df_with_signals.get('bear_alert', False)].index.tolist()
            print(f"   Bull crossovers at indices: {bull_indices}")
            print(f"   Bear crossovers at indices: {bear_indices}")
        else:
            print("❌ No crossovers detected even in raw analysis")

        return result

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🚨 EMERGENCY MACD STRATEGY DIAGNOSTIC TEST")
    print("=" * 50)

    result = test_macd_strategy()

    print("\n" + "=" * 50)
    if result:
        print("✅ TEST PASSED: Signal generation working!")
    else:
        print("❌ TEST FAILED: No signals generated - investigate logs above")
    print("=" * 50)

if __name__ == "__main__":
    main()