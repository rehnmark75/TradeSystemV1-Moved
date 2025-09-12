#!/usr/bin/env python3
"""
Quick MACD Debug Test - Check if MACD crossovers are being detected
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/app/forex_scanner')

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.strategies.macd_strategy import MACDStrategy
try:
    import config
except ImportError:
    from forex_scanner import config
import pandas as pd

def test_macd_detection():
    print("🧪 MACD Detection Debug Test")
    print("=" * 50)
    
    # Initialize components
    db_manager = DatabaseManager(config.DATABASE_URL)
    data_fetcher = DataFetcher(db_manager, 'UTC')
    strategy = MACDStrategy(data_fetcher=data_fetcher, backtest_mode=True)
    
    # Test both JPY and non-JPY pairs
    test_pairs = [
        ("CS.D.EURUSD.MINI.IP", "EURUSD"),  # Non-JPY (threshold: 0.00010) 
        ("CS.D.USDJPY.MINI.IP", "USDJPY")   # JPY (threshold: 0.010)
    ]
    
    for epic, pair in test_pairs:
        print(f"\n📊 Testing {epic} ({pair})")
        print("-" * 40)
        
        # Get recent data (7 days = 168 hours)
        df = data_fetcher.get_enhanced_data(
            epic=epic,
            pair=pair,
            timeframe="15m",
            lookback_hours=168
        )
        
        if df is None or len(df) < 50:
            print(f"❌ No data available for {epic}")
            continue
        
        print(f"✅ Got {len(df)} bars")
        print(f"📅 Date range: {df['start_time'].min()} to {df['start_time'].max()}")
        
        # Calculate MACD manually to check values  
        df_copy = df.copy()
        
        # Calculate EMAs for MACD
        df_copy['ema_12'] = df_copy['close'].ewm(span=12).mean()
        df_copy['ema_26'] = df_copy['close'].ewm(span=26).mean()
        df_copy['macd_line'] = df_copy['ema_12'] - df_copy['ema_26']
        df_copy['macd_signal'] = df_copy['macd_line'].ewm(span=9).mean()
        df_copy['macd_histogram'] = df_copy['macd_line'] - df_copy['macd_signal']
        df_copy['ema_200'] = df_copy['close'].ewm(span=200).mean()
        
        # Check for raw crossovers (before strength filter)
        df_copy['histogram_prev'] = df_copy['macd_histogram'].shift(1)
        raw_bull_crosses = (
            (df_copy['macd_histogram'] > 0) & 
            (df_copy['histogram_prev'] <= 0)
        )
        raw_bear_crosses = (
            (df_copy['macd_histogram'] < 0) & 
            (df_copy['histogram_prev'] >= 0)
        )
        
        raw_bull_count = raw_bull_crosses.sum()
        raw_bear_count = raw_bear_crosses.sum()
        
        print(f"\n📊 RAW MACD CROSSOVERS (before filtering):")
        print(f"   Bull crossovers: {raw_bull_count}")
        print(f"   Bear crossovers: {raw_bear_count}")
        
        # Show typical histogram values
        hist_values = df_copy['macd_histogram'].dropna()
        if len(hist_values) > 0:
            print(f"\n📈 HISTOGRAM VALUE RANGE:")
            print(f"   Min: {hist_values.min():.6f}")
            print(f"   Max: {hist_values.max():.6f}")
            print(f"   Mean absolute: {abs(hist_values).mean():.6f}")
        
        if raw_bull_count > 0 or raw_bear_count > 0:
            print(f"\n🔍 CROSSOVER DETAILS:")
            crossover_rows = df_copy[raw_bull_crosses | raw_bear_crosses]
            for _, row in crossover_rows.iterrows():
                signal_type = "BULL" if row['macd_histogram'] > 0 else "BEAR"
                strength_threshold = strategy.indicator_calculator.get_histogram_strength_threshold(epic)
                meets_strength = abs(row['macd_histogram']) >= strength_threshold
                
                print(f"   🎯 {row['start_time']}: {signal_type}")
                print(f"      Histogram: {row['macd_histogram']:.6f}")
                print(f"      Threshold: {strength_threshold:.6f}")
                print(f"      Meets strength: {'✅' if meets_strength else '❌'}")
                print(f"      Price: {row['close']:.5f}, EMA200: {row['ema_200']:.5f}")
                
                # Check trend alignment
                if signal_type == "BULL":
                    trend_ok = row['close'] > row['ema_200']
                else:
                    trend_ok = row['close'] < row['ema_200']
                print(f"      Trend aligned: {'✅' if trend_ok else '❌'}")
                print()
        
        # Now test with the strategy
        print(f"\n🎯 TESTING WITH MACD STRATEGY:")
        try:
            signal = strategy.detect_signal(df_copy, epic, spread_pips=1.5, timeframe="15m")
            if signal:
                print(f"✅ Strategy detected signal: {signal['signal_type']} with {signal['confidence']:.1%} confidence")
            else:
                print("❌ Strategy detected no signals")
                
        except Exception as e:
            print(f"❌ Strategy error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_macd_detection()