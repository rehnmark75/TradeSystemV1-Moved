#!/usr/bin/env python3
"""
Quick test script for optimized Ichimoku strategy settings
Tests the new balanced confidence and filter settings
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(periods=1000):
    """Create sample forex data for testing"""
    logger.info(f"🧪 Creating sample data with {periods} periods...")

    # Create date range
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods//20), periods=periods, freq='15T')

    # Create realistic forex price movements
    np.random.seed(42)  # For reproducible results
    base_price = 1.0950  # EUR/USD base

    # Generate price series with trend and noise
    trend = np.cumsum(np.random.normal(0, 0.0002, periods))
    noise = np.random.normal(0, 0.0005, periods)
    prices = base_price + trend + noise

    # Create OHLC data
    data = {
        'start_time': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 0.0003, periods)),
        'low': prices - np.abs(np.random.normal(0, 0.0003, periods)),
        'close': prices + np.random.normal(0, 0.0001, periods),
        'ltv': np.random.randint(1000, 5000, periods)
    }

    df = pd.DataFrame(data)
    df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])

    logger.info(f"✅ Sample data created: {len(df)} rows, price range {df['close'].min():.5f} - {df['close'].max():.5f}")
    return df

def test_optimized_ichimoku():
    """Test the optimized Ichimoku strategy"""
    logger.info("🌥️ Testing Optimized Ichimoku Strategy")
    logger.info("=" * 50)

    try:
        # Import the Ichimoku strategy
        from core.strategies.ichimoku_strategy import IchimokuStrategy
        logger.info("✅ Successfully imported IchimokuStrategy")

        # Create strategy instance
        strategy = IchimokuStrategy(
            backtest_mode=True,
            epic='CS.D.EURUSD.MINI.IP',
            timeframe='15m',
            use_optimized_parameters=False,
            pipeline_mode=False
        )
        logger.info("✅ Successfully created IchimokuStrategy instance")
        logger.info(f"📊 Strategy settings:")
        logger.info(f"   Min Confidence: {strategy.min_confidence:.1%}")
        logger.info(f"   Min Bars: {strategy.min_bars}")
        logger.info(f"   Tenkan Period: {strategy.tenkan_period}")
        logger.info(f"   Kijun Period: {strategy.kijun_period}")
        logger.info(f"   Cloud Shift: {strategy.cloud_shift}")

        # Create sample data
        df = create_sample_data(periods=200)  # ~14 days of 15m data

        logger.info(f"🔍 Testing signal detection with {len(df)} data points...")

        # Count signals generated
        signal_count = 0
        signals = []

        # Test on multiple windows to simulate scanner behavior
        for i in range(strategy.min_bars, len(df), 10):  # Test every 10 periods
            test_df = df.iloc[:i+1].copy()

            try:
                signal = strategy.detect_signal(
                    df=test_df,
                    epic='CS.D.EURUSD.MINI.IP',
                    spread_pips=1.5,
                    timeframe='15m'
                )

                if signal:
                    signal_count += 1
                    signals.append({
                        'timestamp': test_df.iloc[-1]['start_time'],
                        'signal_type': signal.get('signal_type'),
                        'confidence': signal.get('confidence', 0),
                        'price': signal.get('price', 0)
                    })
                    logger.info(f"📈 Signal {signal_count}: {signal['signal_type']} at {signal.get('price', 0):.5f} (confidence: {signal.get('confidence', 0):.1%})")

            except Exception as e:
                logger.warning(f"⚠️ Signal detection error at period {i}: {e}")
                continue

        # Report results
        logger.info("🏁 Test Results:")
        logger.info("=" * 30)
        logger.info(f"📊 Total signals generated: {signal_count}")
        logger.info(f"📈 Data periods tested: {len(df)}")
        logger.info(f"🎯 Signal frequency: {signal_count / (len(df) / strategy.min_bars):.2f} signals per valid window")

        if signals:
            confidences = [s['confidence'] for s in signals]
            logger.info(f"📊 Average confidence: {np.mean(confidences):.1%}")
            logger.info(f"📊 Min confidence: {np.min(confidences):.1%}")
            logger.info(f"📊 Max confidence: {np.max(confidences):.1%}")

            bull_signals = len([s for s in signals if s['signal_type'] == 'BULL'])
            bear_signals = len([s for s in signals if s['signal_type'] == 'BEAR'])
            logger.info(f"📈 Bull signals: {bull_signals}")
            logger.info(f"📉 Bear signals: {bear_signals}")

        # Assessment
        if signal_count == 0:
            logger.warning("⚠️ WARNING: No signals generated - settings may be too restrictive")
        elif signal_count > 50:
            logger.warning(f"⚠️ WARNING: {signal_count} signals may be too many - consider more restrictive settings")
        elif signal_count < 5:
            logger.warning(f"⚠️ WARNING: Only {signal_count} signals - settings may be too restrictive")
        else:
            logger.info(f"✅ GOOD: {signal_count} signals appears to be a reasonable balance")

        return signal_count, signals

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, []

if __name__ == "__main__":
    logger.info("🚀 Starting Optimized Ichimoku Strategy Test")
    signal_count, signals = test_optimized_ichimoku()

    if signal_count > 0:
        logger.info(f"🎉 Test completed successfully with {signal_count} signals")
    else:
        logger.error("💥 Test completed but no signals were generated")