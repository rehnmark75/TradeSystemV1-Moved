#!/usr/bin/env python3
"""
Test specific USDJPY signal at 2025-10-13 14:15:00 to see MTF histogram values
"""
import sys
sys.path.insert(0, '/home/hr/Projects/TradeSystemV1/worker/app')

import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import after path setup
from forex_scanner.core.backtest_data_fetcher import BacktestDataFetcher

def test_signal():
    """Check histogram values for USDJPY at the problematic timestamp"""

    logger.info("=" * 70)
    logger.info("Testing USDJPY BUY signal at 2025-10-13 14:15:00 UTC")
    logger.info("=" * 70)

    # Initialize data fetcher
    data_fetcher = BacktestDataFetcher()

    # Signal timestamp
    signal_time = pd.Timestamp('2025-10-13 14:15:00', tz='UTC')
    logger.info(f"\n📅 Signal Time: {signal_time}")

    # Set backtest time to signal time
    data_fetcher.current_backtest_time = signal_time

    # Fetch 15m data (signal timeframe)
    logger.info(f"\n🔍 Fetching 15m data for USDJPY...")
    df_15m = data_fetcher.get_enhanced_data(
        epic='CS.D.USDJPY.MINI.IP',
        pair='CS.D.USDJPY.MINI.IP',
        timeframe='15m',
        lookback_hours=24
    )

    if df_15m is not None and len(df_15m) > 0:
        latest_15m = df_15m.iloc[-1]
        logger.info(f"✅ 15m data: {len(df_15m)} bars")
        logger.info(f"   Latest bar: {latest_15m.get('start_time', 'N/A')}")
        logger.info(f"   15m Histogram: {latest_15m.get('macd_histogram', 'N/A'):.6f}")

        # Check if it's positive (bullish)
        hist_15m = latest_15m.get('macd_histogram', 0)
        if hist_15m > 0:
            logger.info(f"   ✅ 15m is BULLISH (histogram > 0)")
        else:
            logger.info(f"   ❌ 15m is BEARISH (histogram <= 0)")
    else:
        logger.error(f"❌ No 15m data available")
        return

    # Fetch 1H data
    logger.info(f"\n🔍 Fetching 1H data for USDJPY...")
    df_1h = data_fetcher.get_enhanced_data(
        epic='CS.D.USDJPY.MINI.IP',
        pair='CS.D.USDJPY.MINI.IP',
        timeframe='1h',
        lookback_hours=168
    )

    if df_1h is not None and len(df_1h) >= 2:
        # Use second-to-last (completed candle)
        completed_1h = df_1h.iloc[-2]
        latest_1h = df_1h.iloc[-1]
        logger.info(f"✅ 1H data: {len(df_1h)} bars")
        logger.info(f"   Latest bar: {latest_1h.get('start_time', 'N/A')}")
        logger.info(f"   Completed bar: {completed_1h.get('start_time', 'N/A')}")
        logger.info(f"   1H Histogram (completed): {completed_1h.get('macd_histogram', 'N/A'):.6f}")

        # Check if it's positive (bullish)
        hist_1h = completed_1h.get('macd_histogram', 0)
        if hist_1h > 0:
            logger.info(f"   ✅ 1H is BULLISH (histogram > 0)")
        else:
            logger.info(f"   ❌ 1H is BEARISH (histogram <= 0)")
    else:
        logger.error(f"❌ Insufficient 1H data")
        return

    # Fetch 4H data
    logger.info(f"\n🔍 Fetching 4H data for USDJPY...")
    df_4h = data_fetcher.get_enhanced_data(
        epic='CS.D.USDJPY.MINI.IP',
        pair='CS.D.USDJPY.MINI.IP',
        timeframe='4h',
        lookback_hours=168
    )

    if df_4h is not None and len(df_4h) >= 2:
        # Use second-to-last (completed candle)
        completed_4h = df_4h.iloc[-2]
        latest_4h = df_4h.iloc[-1]
        logger.info(f"✅ 4H data: {len(df_4h)} bars")
        logger.info(f"   Latest bar: {latest_4h.get('start_time', 'N/A')}")
        logger.info(f"   Completed bar: {completed_4h.get('start_time', 'N/A')}")
        logger.info(f"   4H Histogram (completed): {completed_4h.get('macd_histogram', 'N/A'):.6f}")

        # Check if it's positive (bullish)
        hist_4h = completed_4h.get('macd_histogram', 0)
        if hist_4h > 0:
            logger.info(f"   ✅ 4H is BULLISH (histogram > 0)")
        else:
            logger.info(f"   ❌ 4H is BEARISH (histogram <= 0)")
    else:
        logger.error(f"❌ Insufficient 4H data")
        return

    # Summary
    logger.info(f"\n" + "=" * 70)
    logger.info(f"📊 MTF ALIGNMENT SUMMARY for BULL signal:")
    logger.info(f"   15m: {'✅ BULLISH' if hist_15m > 0 else '❌ BEARISH'} ({hist_15m:.6f})")
    logger.info(f"   1H:  {'✅ BULLISH' if hist_1h > 0 else '❌ BEARISH'} ({hist_1h:.6f})")
    logger.info(f"   4H:  {'✅ BULLISH' if hist_4h > 0 else '❌ BEARISH'} ({hist_4h:.6f})")

    all_bullish = hist_15m > 0 and hist_1h > 0 and hist_4h > 0
    if all_bullish:
        logger.info(f"\n✅ VERDICT: Signal should be ALLOWED (all timeframes bullish)")
    else:
        logger.info(f"\n❌ VERDICT: Signal should be REJECTED (not all timeframes bullish)")
    logger.info("=" * 70)

if __name__ == '__main__':
    test_signal()
