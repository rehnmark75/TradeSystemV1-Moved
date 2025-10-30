#!/usr/bin/env python3
"""
Test SMC Structure Strategy
Validates the pure structure-based strategy implementation
"""

import sys
import os

# Add paths
sys.path.insert(0, '/app/app')
sys.path.insert(0, '/app')

import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Import after path setup
from forex_scanner.core.data_fetcher import DataFetcher
from forex_scanner.core.strategies.smc_structure_strategy import SMCStructureStrategy
from forex_scanner.configdata.strategies import config_smc_structure

def test_smc_structure():
    """Test SMC structure strategy on USDJPY"""

    logger.info("="*70)
    logger.info("SMC Structure Strategy Test")
    logger.info("="*70)

    # Initialize
    data_fetcher = DataFetcher()
    strategy = SMCStructureStrategy(config=config_smc_structure, logger=logger)

    # Test configuration
    epic = 'CS.D.USDJPY.MINI.IP'
    pair = 'USDJPY'

    logger.info(f"\n📊 Fetching data for {pair}...")
    logger.info(f"   Epic: {epic}")

    # Fetch 1H data (entry timeframe)
    df_1h = data_fetcher.get_enhanced_data(
        epic=epic,
        pair=pair,
        timeframe='1h',
        lookback_hours=200
    )

    # Fetch 4H data (HTF for trend)
    df_4h = data_fetcher.get_enhanced_data(
        epic=epic,
        pair=pair,
        timeframe='4h',
        lookback_hours=400  # 100 bars * 4 hours
    )

    logger.info(f"\n✅ Data Retrieved:")
    logger.info(f"   1H bars: {len(df_1h)}")
    logger.info(f"   4H bars: {len(df_4h)}")
    logger.info(f"   1H date range: {df_1h.index[0]} to {df_1h.index[-1]}")
    logger.info(f"   4H date range: {df_4h.index[0]} to {df_4h.index[-1]}")
    logger.info(f"   Current price (1H): {df_1h['close'].iloc[-1]:.5f}")

    # Run signal detection
    logger.info(f"\n🔍 Running SMC Structure Signal Detection...")

    signal = strategy.detect_signal(
        df_1h=df_1h,
        df_4h=df_4h,
        epic=epic,
        pair=pair
    )

    if signal:
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ SIGNAL DETECTED!")
        logger.info(f"{'='*70}")
        logger.info(f"Direction: {signal['signal']}")
        logger.info(f"Entry: {signal['entry_price']:.5f}")
        logger.info(f"Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
        logger.info(f"Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
        logger.info(f"R:R Ratio: {signal['rr_ratio']:.2f}")
        logger.info(f"\nHTF Trend: {signal['htf_trend']} ({signal['htf_structure']})")
        logger.info(f"Trend Strength: {signal['htf_strength']*100:.0f}%")
        logger.info(f"Pattern: {signal['pattern_type']} ({signal['pattern_strength']*100:.0f}%)")
        logger.info(f"S/R Level: {signal['sr_level']:.5f} ({signal['sr_type']})")
        logger.info(f"\nDescription: {signal['description']}")
        logger.info(f"{'='*70}")
    else:
        logger.info(f"\n{'='*70}")
        logger.info(f"❌ No signal detected")
        logger.info(f"{'='*70}")

    logger.info("\nTest Complete\n")

if __name__ == "__main__":
    try:
        test_smc_structure()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
