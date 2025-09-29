#!/usr/bin/env python3
"""
Quick Test: TradingView Signal Quality Implementation
Tests the improved mean reversion signal detection with LuxAlgo method
"""
import pandas as pd
import numpy as np
import sys
import os

# Add path for imports
sys.path.insert(0, '/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner')

try:
    from core.strategies.helpers.mean_reversion_signal_detector import MeanReversionSignalDetector
    print("✅ Successfully imported MeanReversionSignalDetector")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def create_test_data(n_rows=1000):
    """Create synthetic test data with oscillator patterns"""
    np.random.seed(42)

    # Create base price series with trend
    prices = 100 + np.cumsum(np.random.randn(n_rows) * 0.01)

    # Create RSI-like oscillator (14-period)
    returns = np.diff(prices, prepend=prices[0])
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)

    avg_gains = pd.Series(gains).rolling(14).mean()
    avg_losses = pd.Series(losses).rolling(14).mean()
    rs = avg_gains / avg_losses.replace(0, 0.01)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    # Create smoothed RSI (LuxAlgo method)
    smoothed_rsi = rsi.rolling(3).mean()

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='15T'),
        'close': prices,
        'high': prices + np.random.rand(n_rows) * 0.5,
        'low': prices - np.random.rand(n_rows) * 0.5,
        'open': prices + np.random.randn(n_rows) * 0.1,
        'rsi_14': rsi,
        'smoothed_rsi': smoothed_rsi,
        'luxalgo_oscillator': smoothed_rsi,
        'luxalgo_signal': smoothed_rsi.rolling(5).mean(),
        'luxalgo_histogram': smoothed_rsi - smoothed_rsi.rolling(5).mean(),
        'oscillator_bull_score': np.random.rand(n_rows) * 0.3 + 0.2,  # 0.2-0.5
        'oscillator_bear_score': np.random.rand(n_rows) * 0.3 + 0.2,  # 0.2-0.5
        'adx': np.random.rand(n_rows) * 40 + 15,  # 15-55 ADX range
        'atr': np.random.rand(n_rows) * 0.002 + 0.001  # ATR values
    })

    # Create some extreme oscillator conditions for testing
    # Add some oversold conditions (RSI < 20)
    oversold_indices = np.random.choice(range(100, n_rows-100), 20)
    df.loc[oversold_indices, 'smoothed_rsi'] = np.random.rand(20) * 15 + 5  # 5-20 range
    df.loc[oversold_indices, 'luxalgo_oscillator'] = df.loc[oversold_indices, 'smoothed_rsi']

    # Add some overbought conditions (RSI > 80)
    overbought_indices = np.random.choice(range(100, n_rows-100), 20)
    df.loc[overbought_indices, 'smoothed_rsi'] = np.random.rand(20) * 15 + 80  # 80-95 range
    df.loc[overbought_indices, 'luxalgo_oscillator'] = df.loc[overbought_indices, 'smoothed_rsi']

    return df

def test_signal_quality():
    """Test the improved signal quality"""
    print("\n🧪 Testing TradingView Signal Quality Implementation")
    print("=" * 60)

    # Create test data
    print("📊 Creating synthetic market data...")
    df = create_test_data(1000)

    # Initialize signal detector
    print("🔍 Initializing MeanReversionSignalDetector...")
    detector = MeanReversionSignalDetector()

    # Detect signals
    print("⚡ Running signal detection...")
    try:
        df_with_signals = detector.detect_mean_reversion_signals(df, epic='EUR_USD', is_backtest=True)

        # Analyze results
        bull_signals = df_with_signals['mean_reversion_bull'].sum()
        bear_signals = df_with_signals['mean_reversion_bear'].sum()
        total_signals = bull_signals + bear_signals

        # Calculate signal density (signals per day)
        total_days = len(df_with_signals) / (24 * 4)  # 15-min bars
        signals_per_day = total_signals / total_days if total_days > 0 else 0

        print("\n📈 SIGNAL QUALITY RESULTS:")
        print(f"   Total data points: {len(df_with_signals)}")
        print(f"   Simulation days: {total_days:.1f}")
        print(f"   Bull signals: {bull_signals}")
        print(f"   Bear signals: {bear_signals}")
        print(f"   Total signals: {total_signals}")
        print(f"   Signals per day: {signals_per_day:.1f}")

        # Analyze signal confidence
        signal_rows = df_with_signals[(df_with_signals['mean_reversion_bull']) |
                                     (df_with_signals['mean_reversion_bear'])]
        if len(signal_rows) > 0:
            avg_confidence = signal_rows['mr_confidence'].mean()
            max_confidence = signal_rows['mr_confidence'].max()
            min_confidence = signal_rows['mr_confidence'].min()

            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Max confidence: {max_confidence:.3f}")
            print(f"   Min confidence: {min_confidence:.3f}")

            # Expected vs actual results
            print(f"\n✅ QUALITY ASSESSMENT:")
            if signals_per_day <= 5:
                print(f"   ✅ Signal frequency: EXCELLENT (≤5/day)")
            elif signals_per_day <= 15:
                print(f"   ⚠️ Signal frequency: ACCEPTABLE (≤15/day)")
            else:
                print(f"   ❌ Signal frequency: EXCESSIVE (>{signals_per_day:.1f}/day)")

            if avg_confidence >= 0.65:
                print(f"   ✅ Signal confidence: EXCELLENT (≥65%)")
            elif avg_confidence >= 0.55:
                print(f"   ⚠️ Signal confidence: ACCEPTABLE (≥55%)")
            else:
                print(f"   ❌ Signal confidence: LOW (<55%)")
        else:
            print("   ❌ No signals generated")

        return total_signals, signals_per_day

    except Exception as e:
        print(f"❌ Signal detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

if __name__ == "__main__":
    print("🔥 TradingView Signal Quality Test")
    print("Testing LuxAlgo smoothed RSI crossover method...")

    total_signals, signals_per_day = test_signal_quality()

    print(f"\n🏆 FINAL RESULT:")
    if signals_per_day <= 5 and total_signals > 0:
        print(f"   ✅ SUCCESS: Generated {total_signals} quality signals ({signals_per_day:.1f}/day)")
        print(f"   📊 This is a massive improvement from 3591 signals!")
    elif total_signals == 0:
        print(f"   ❌ FAILURE: No signals generated (too restrictive)")
    else:
        print(f"   ⚠️ PARTIAL: {total_signals} signals ({signals_per_day:.1f}/day) - may need fine-tuning")