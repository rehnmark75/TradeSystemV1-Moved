#!/usr/bin/env python3
"""
REAL DATA MACD DIAGNOSTIC
Compare synthetic test (which works) vs real market data (which doesn't work)
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

def get_real_market_data():
    """Get real market data using the same method as backtest"""
    try:
        from core.database import DatabaseManager
        from core.data_fetcher import DataFetcher
        import config

        # Initialize components same as backtest
        db_manager = DatabaseManager(config.DATABASE_URL)
        data_fetcher = DataFetcher(db_manager, 'UTC')

        epic = 'CS.D.EURUSD.CEEM.IP'
        pair = 'EURUSD'
        timeframe = '15m'

        print(f"🔍 Fetching REAL market data for {epic} ({timeframe})...")

        # Get recent data (same as backtest would)
        df = data_fetcher.get_enhanced_data(
            epic=epic,
            pair=pair,
            timeframe=timeframe,
            lookback_hours=168  # 1 week like backtest
        )

        if df is None or df.empty:
            print("❌ No real data available")
            return None

        print(f"✅ Retrieved {len(df)} bars of real market data")
        print(f"📅 Date range: {df.iloc[0]['start_time']} to {df.iloc[-1]['start_time']}")

        return df, epic, timeframe

    except Exception as e:
        print(f"❌ Failed to get real market data: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_real_data_histogram(df):
    """Analyze MACD histogram values in real market data"""
    print("\n🔍 REAL DATA HISTOGRAM ANALYSIS")
    print("=" * 50)

    # Check if MACD data exists
    required_cols = ['macd_line', 'macd_signal', 'macd_histogram']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing MACD columns: {missing_cols}")

        # Calculate MACD if missing
        print("📊 Calculating MACD indicators...")
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd_line'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        print("✅ MACD calculated")

    # Analyze histogram
    histogram_values = df['macd_histogram'].dropna()
    if len(histogram_values) == 0:
        print("❌ No histogram values available")
        return

    hist_min = histogram_values.min()
    hist_max = histogram_values.max()
    hist_mean = histogram_values.mean()
    hist_std = histogram_values.std()
    hist_abs_max = histogram_values.abs().max()

    print(f"📊 REAL DATA Histogram Statistics:")
    print(f"   📈 Range: {hist_min:.8f} to {hist_max:.8f}")
    print(f"   📊 Mean: {hist_mean:.8f}")
    print(f"   📊 Std Dev: {hist_std:.8f}")
    print(f"   📊 Max Absolute: {hist_abs_max:.8f}")

    # Compare with our current thresholds
    current_thresholds = {
        'JPY': 0.000005,
        'Major': 0.000002
    }

    print(f"\n🎯 THRESHOLD ANALYSIS:")
    print(f"   Current Major pairs threshold: {current_thresholds['Major']:.8f}")
    print(f"   Current JPY pairs threshold: {current_thresholds['JPY']:.8f}")

    # Count values above thresholds
    above_major_threshold = (histogram_values.abs() >= current_thresholds['Major']).sum()
    above_jpy_threshold = (histogram_values.abs() >= current_thresholds['JPY']).sum()

    print(f"   Values above Major threshold: {above_major_threshold}/{len(histogram_values)} ({above_major_threshold/len(histogram_values)*100:.1f}%)")
    print(f"   Values above JPY threshold: {above_jpy_threshold}/{len(histogram_values)} ({above_jpy_threshold/len(histogram_values)*100:.1f}%)")

    # Count actual crossovers (zero-line crosses)
    histogram_prev = df['macd_histogram'].shift(1).dropna()
    current_hist = df['macd_histogram'].dropna()

    if len(current_hist) > 1:
        bull_crosses = ((current_hist > 0) & (histogram_prev <= 0)).sum()
        bear_crosses = ((current_hist < 0) & (histogram_prev >= 0)).sum()

        print(f"   Raw zero-line crossovers: {bull_crosses} bull, {bear_crosses} bear")

        # Check crossovers that meet thresholds
        bull_crosses_threshold = ((current_hist > 0) & (histogram_prev <= 0) & (current_hist.abs() >= current_thresholds['Major'])).sum()
        bear_crosses_threshold = ((current_hist < 0) & (histogram_prev >= 0) & (current_hist.abs() >= current_thresholds['Major'])).sum()

        print(f"   Crossovers above threshold: {bull_crosses_threshold} bull, {bear_crosses_threshold} bear")

    # Show recent values
    recent_hist = histogram_values.tail(20)
    print(f"\n📈 Recent 20 histogram values:")
    for i, val in enumerate(recent_hist):
        above_threshold = "✅" if abs(val) >= current_thresholds['Major'] else "❌"
        print(f"   {-19+i:2d}: {val:.8f} {above_threshold}")

    return histogram_values

def test_strategy_with_real_data(df, epic, timeframe):
    """Test MACD strategy with real market data"""
    print("\n🧪 TESTING STRATEGY WITH REAL DATA")
    print("=" * 50)

    try:
        from core.strategies.macd_strategy import MACDStrategy

        # Initialize strategy same way as backtest
        strategy = MACDStrategy(
            epic=epic,
            timeframe=timeframe,
            backtest_mode=True
        )

        print(f"✅ Strategy initialized for {epic} ({timeframe})")

        # Test with last 50 bars (reasonable test size)
        test_data = df.tail(50).copy()

        print(f"📊 Testing with {len(test_data)} bars of real data")
        print(f"📅 Test range: {test_data.iloc[0]['start_time']} to {test_data.iloc[-1]['start_time']}")

        # Test signal detection
        result = strategy.detect_signal(
            df=test_data,
            epic=epic,
            spread_pips=1.5,
            timeframe=timeframe
        )

        if result:
            print(f"✅ SIGNAL DETECTED with real data!")
            print(f"   Type: {result.get('signal_type', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 'unknown')}")
            print(f"   Price: {result.get('price', 'unknown')}")
        else:
            print("❌ NO SIGNAL DETECTED with real data")

        # Test direct crossover detection
        print("\n🔍 Testing direct crossover detection...")
        df_with_signals = strategy.indicator_calculator.detect_macd_crossovers(
            test_data, epic, is_backtest=True
        )

        bull_signals = df_with_signals.get('bull_alert', pd.Series(False, index=df_with_signals.index)).sum()
        bear_signals = df_with_signals.get('bear_alert', pd.Series(False, index=df_with_signals.index)).sum()

        print(f"📊 Direct crossover test: {bull_signals} bull signals, {bear_signals} bear signals")

        return result

    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_synthetic_vs_real():
    """Compare our working synthetic data vs failing real data"""
    print("\n🔄 COMPARING SYNTHETIC VS REAL DATA")
    print("=" * 50)

    # Synthetic data stats (from our test)
    print("📊 SYNTHETIC DATA (WORKING):")
    print("   Range: -0.00005266 to 0.00005934")
    print("   Threshold used: 0.000002 (Major pairs)")
    print("   Result: 10 bull, 11 bear crossovers detected")
    print("   Signals: ✅ 1 signal generated")

    # Real data
    real_data_result = get_real_market_data()
    if real_data_result:
        df, epic, timeframe = real_data_result

        print("\n📊 REAL DATA:")
        histogram_values = analyze_real_data_histogram(df)

        print("\n🧪 REAL DATA STRATEGY TEST:")
        strategy_result = test_strategy_with_real_data(df, epic, timeframe)

        if strategy_result:
            print("   Result: ✅ Signal generated with real data")
        else:
            print("   Result: ❌ No signals with real data")

        # Diagnosis
        print("\n🔍 DIAGNOSIS:")
        if histogram_values is not None:
            max_abs_real = histogram_values.abs().max()
            threshold = 0.000002

            if max_abs_real < threshold:
                print(f"❌ PROBLEM FOUND: Real data max histogram value ({max_abs_real:.8f}) is below threshold ({threshold:.8f})")
                print(f"   Suggested threshold: {max_abs_real * 0.1:.8f} (10% of max)")
            else:
                print(f"✅ Real data has values above threshold")

                # Check crossover frequency
                above_threshold_count = (histogram_values.abs() >= threshold).sum()
                percentage = above_threshold_count / len(histogram_values) * 100

                if percentage < 5:
                    print(f"❌ PROBLEM: Only {percentage:.1f}% of values are above threshold")
                    print(f"   Too restrictive - consider lowering threshold")
                else:
                    print(f"✅ {percentage:.1f}% of values are above threshold")

def main():
    """Main diagnostic function"""
    print("🚨 REAL DATA MACD DIAGNOSTIC")
    print("=" * 60)
    print("Comparing synthetic test data (works) vs real market data (doesn't work)")

    compare_synthetic_vs_real()

    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("If real data shows threshold issues, the solution is to lower thresholds further")
    print("=" * 60)

if __name__ == "__main__":
    main()