#!/usr/bin/env python3
"""
Test script to verify MTF histogram alignment uses correct timing
"""
import sys
sys.path.insert(0, '/home/hr/Projects/TradeSystemV1/worker/app')

import pandas as pd
from datetime import datetime, timedelta
from forex_scanner.core.strategies.macd_strategy import MACDStrategy
from forex_scanner.core.backtest_data_fetcher import BacktestDataFetcher
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mtf_timing():
    """Test that MTF check uses signal bar timestamp correctly"""

    logger.info("=" * 60)
    logger.info("Testing MTF Histogram Alignment Timing")
    logger.info("=" * 60)

    # Initialize data fetcher
    data_fetcher = BacktestDataFetcher()

    # Initialize MACD strategy with data fetcher
    strategy = MACDStrategy(backtest_mode=True, data_fetcher=data_fetcher)

    # Create a test signal timestamp (e.g., 2025-10-13 14:15:00 UTC)
    signal_time = pd.Timestamp('2025-10-13 14:15:00', tz='UTC')

    logger.info(f"\n📅 Signal bar timestamp: {signal_time}")

    # Create a mock DataFrame with signal bar
    test_df = pd.DataFrame({
        'start_time': [signal_time],
        'close': [150.50],
        'macd_histogram': [0.05],
        'macd_line': [0.10],
        'macd_signal': [0.05]
    })

    # Set backtest time to signal time
    data_fetcher.current_backtest_time = signal_time

    logger.info(f"🔍 Data fetcher current_backtest_time: {data_fetcher.current_backtest_time}")
    logger.info(f"📊 Calling MTF check for USDJPY BULL signal...")

    # Call MTF check with signal timestamp
    result = strategy._check_mtf_histogram_alignment(
        df=test_df,
        signal_direction='BULL',
        epic='CS.D.USDJPY.MINI.IP',
        signal_bar_time=signal_time
    )

    logger.info(f"\n✅ MTF Check Result:")
    logger.info(f"   Aligned: {result.get('aligned')}")
    logger.info(f"   Misaligned timeframes: {result.get('misaligned_timeframes', [])}")
    logger.info(f"   Confidence penalty: {result.get('confidence_penalty', 0)}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Test completed successfully")
    logger.info("=" * 60)

if __name__ == '__main__':
    test_mtf_timing()
