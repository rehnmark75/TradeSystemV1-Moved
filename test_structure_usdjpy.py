#!/usr/bin/env python3
"""
Test H4 Market Structure Detection on USDJPY
Uses real data from IG Markets via the data fetcher
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
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import after path setup
from forex_scanner.core.data_fetcher import DataFetcher
from forex_scanner.core.strategies.helpers.smc_market_structure import SMCMarketStructure

def test_usdjpy_structure():
    """Test structure detection on USDJPY"""

    logger.info("="*70)
    logger.info("USDJPY H4 Market Structure Detection Test")
    logger.info("="*70)

    # Initialize
    data_fetcher = DataFetcher()
    structure_analyzer = SMCMarketStructure(logger=logger)

    # Configuration (matching MACD strategy config)
    config = {
        'swing_length': 20,
        'structure_confirmation': 2,
        'min_structure_significance': 0.4,
    }

    # Fetch USDJPY H4 data
    epic = 'CS.D.USDJPY.MINI.IP'
    pair = 'USDJPY'
    lookback_hours = 80 * 4  # 80 bars * 4 hours

    logger.info(f"\n📊 Fetching {pair} data...")
    logger.info(f"   Epic: {epic}")
    logger.info(f"   Timeframe: 4H")
    logger.info(f"   Lookback: {lookback_hours} hours ({lookback_hours//4} bars)")

    h4_df = data_fetcher.get_enhanced_data(
        epic=epic,
        pair=pair,
        timeframe='4h',
        lookback_hours=lookback_hours
    )

    logger.info(f"\n✅ Data Retrieved:")
    logger.info(f"   Total bars: {len(h4_df)}")
    logger.info(f"   Date range: {h4_df.index[0]} to {h4_df.index[-1]}")
    logger.info(f"   Current price: {h4_df['close'].iloc[-1]:.5f}")

    # Run structure analysis
    logger.info(f"\n🔍 Running Structure Analysis...")
    logger.info(f"   Config: {config}")

    h4_with_structure = structure_analyzer.analyze_market_structure(
        df=h4_df,
        config=config,
        epic=epic,
        timeframe='4h'
    )

    # Count swing points
    swing_highs = h4_with_structure[h4_with_structure['swing_high'] == True]
    swing_lows = h4_with_structure[h4_with_structure['swing_low'] == True]

    logger.info(f"\n📈 Swing Points Detected:")
    logger.info(f"   Swing Highs: {len(swing_highs)}")
    logger.info(f"   Swing Lows: {len(swing_lows)}")
    logger.info(f"   Total Swings: {len(swing_highs) + len(swing_lows)}")

    if len(swing_highs) > 0:
        logger.info(f"\n   Last 3 Swing Highs:")
        for idx, row in swing_highs.tail(3).iterrows():
            logger.info(f"      {idx}: {row['high']:.5f} (type: {row['swing_type']}, strength: {row['swing_strength']:.2f})")

    if len(swing_lows) > 0:
        logger.info(f"\n   Last 3 Swing Lows:")
        for idx, row in swing_lows.tail(3).iterrows():
            logger.info(f"      {idx}: {row['low']:.5f} (type: {row['swing_type']}, strength: {row['swing_strength']:.2f})")

    # Analyze structure breaks
    structure_breaks = h4_with_structure[h4_with_structure['structure_break'] == True]

    logger.info(f"\n🏗️  Structure Breaks Detected:")
    logger.info(f"   Total: {len(structure_breaks)}")

    if len(structure_breaks) > 0:
        logger.info(f"\n   All Structure Breaks:")
        for idx, row in structure_breaks.iterrows():
            logger.info(f"      {idx}: {row['break_type']:5s} {row['break_direction']:8s} (sig: {row['structure_significance']:.3f})")

        # Get the most recent structure break
        last_break = structure_breaks.iloc[-1]

        logger.info(f"\n✅ MOST RECENT STRUCTURE BREAK:")
        logger.info(f"   Date: {last_break.name}")
        logger.info(f"   Type: {last_break['break_type']}")
        logger.info(f"   Direction: {last_break['break_direction']}")
        logger.info(f"   Significance: {last_break['structure_significance']:.3f}")
        logger.info(f"   Price: {last_break['close']:.5f}")

        # Test alignment with BEAR signal
        logger.info(f"\n📊 SIGNAL ALIGNMENT TEST:")
        logger.info(f"   Testing: BEAR signal (wants bearish structure)")
        logger.info(f"   Structure direction: {last_break['break_direction']}")

        signal_wants_bullish = False  # BEAR signal
        structure_is_bullish = (last_break['break_direction'] == 'bullish')
        aligned = (signal_wants_bullish == structure_is_bullish)

        if aligned:
            logger.info(f"   Result: ✅ ALIGNED - Signal would pass structure check")
        else:
            logger.info(f"   Result: ❌ NOT ALIGNED - Signal would be REJECTED")
            logger.info(f"   Reason: BEAR signal requires bearish structure, but structure is {last_break['break_direction']}")

        # Additional context
        bars_since_break = len(h4_df) - structure_breaks.index.get_loc(last_break.name) - 1
        logger.info(f"\n   Bars since last break: {bars_since_break}")
        logger.info(f"   Break still recent: {'✅ YES' if bars_since_break < 20 else '⚠️  Older than 20 bars'}")

    else:
        logger.info(f"   ❌ No structure breaks detected")
        logger.info(f"   This would cause signals to be REJECTED (no structure to validate)")

    logger.info("\n" + "="*70)
    logger.info("Test Complete")
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_usdjpy_structure()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
