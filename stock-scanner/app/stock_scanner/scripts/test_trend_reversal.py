"""
Quick test script for the Trend Reversal Scanner
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent path for imports
sys.path.insert(0, '/app')

from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.scanners.strategies.trend_reversal import (
    TrendReversalScanner,
    TrendReversalConfig
)
from stock_scanner.scanners.scoring import SignalScorer
from stock_scanner.config import get_database_url


async def main():
    """Test the trend reversal scanner"""
    logger.info("=" * 60)
    logger.info("TREND REVERSAL SCANNER TEST")
    logger.info("=" * 60)

    # Initialize database
    db_url = get_database_url()
    db = AsyncDatabaseManager(db_url)
    await db.connect()

    try:
        # Create scanner with default config
        scanner_config = TrendReversalConfig(
            min_score_threshold=50,  # Lower threshold for testing
            min_relative_volume=0.3,  # Lower for more results
        )
        scorer = SignalScorer()
        scanner = TrendReversalScanner(db, scanner_config, scorer)

        # Use yesterday's date
        calc_date = (datetime.now() - timedelta(days=1)).date()
        logger.info(f"Scanning for date: {calc_date}")

        # Run the scan
        signals = await scanner.scan(calc_date)

        # Report results
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"RESULTS: Found {len(signals)} trend reversal signals")
        logger.info("=" * 60)

        if signals:
            for i, signal in enumerate(signals[:10], 1):
                metrics = signal.raw_data.get('reversal_metrics', {})
                downtrend_low = signal.raw_data.get('downtrend_rsi_low', 0)

                logger.info(f"\n{i}. {signal.ticker} - Score: {signal.composite_score}")
                logger.info(f"   Entry: ${signal.entry_price:.2f} | Stop: ${signal.stop_loss:.2f} | TP1: ${signal.take_profit_1:.2f}")
                logger.info(f"   Quality: {signal.quality_tier.value} | R:R: {signal.risk_reward_ratio}")
                logger.info(f"   Reversal: {metrics.get('positive_days', 0)}/3 days positive, RSI +{metrics.get('rsi_improvement', 0):.0f} from {downtrend_low:.0f}")
                logger.info(f"   MACD improving: {metrics.get('macd_improving', False)} | SMA20 crossed: {metrics.get('sma20_crossed', False)}")
                logger.info(f"   Factors: {', '.join(signal.confluence_factors[:4])}")
        else:
            logger.info("No signals found. This could mean:")
            logger.info("  - No stocks match the reversal criteria today")
            logger.info("  - Filters might be too strict")

        # Show some stats
        logger.info("\n" + "=" * 60)
        logger.info("SIGNAL QUALITY DISTRIBUTION")
        logger.info("=" * 60)
        quality_counts = {}
        for signal in signals:
            tier = signal.quality_tier.value
            quality_counts[tier] = quality_counts.get(tier, 0) + 1
        for tier, count in sorted(quality_counts.items()):
            logger.info(f"  {tier}: {count} signals")

    finally:
        await db.close()

    logger.info("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(main())
