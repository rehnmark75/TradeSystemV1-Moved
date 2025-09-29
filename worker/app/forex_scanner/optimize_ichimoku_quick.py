#!/usr/bin/env python3
"""
Quick Ichimoku Parameter Optimization for EURUSD
Based on existing MACD optimizer pattern
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import itertools

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from core.data_fetcher import EnhancedDataFetcher
from core.strategies.ichimoku_strategy import IchimokuStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_ichimoku_for_epic(epic: str = 'CS.D.EURUSD.CEEM.IP', days: int = 30):
    """Quick optimization for Ichimoku parameters"""

    logger.info(f"🌥️ Starting Ichimoku optimization for {epic}")

    # Parameter ranges to test (smaller ranges for quick optimization)
    tenkan_periods = [7, 9, 12]        # Conversion line
    kijun_periods = [22, 26, 30]       # Base line
    confidence_thresholds = [0.45, 0.50, 0.55, 0.60]
    min_bars_options = [60, 70, 80]

    # Cloud filter settings to test
    cloud_filter_settings = [
        {'enabled': False},  # No cloud filter
        {'enabled': True, 'buffer_pips': 20.0},   # Moderate filter
        {'enabled': True, 'buffer_pips': 35.0},   # Generous filter
    ]

    best_config = None
    best_score = 0
    results = []

    try:
        # Initialize data fetcher
        data_fetcher = EnhancedDataFetcher()

        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        logger.info(f"📊 Fetching {days} days of data for {epic}")
        df = data_fetcher.get_data_range(epic, start_time, end_time, '15m')

        if df is None or len(df) < 100:
            logger.error(f"❌ Insufficient data for {epic}")
            return None

        logger.info(f"✅ Got {len(df)} data points for optimization")

        total_combinations = len(tenkan_periods) * len(kijun_periods) * len(confidence_thresholds) * len(min_bars_options) * len(cloud_filter_settings)
        logger.info(f"🔍 Testing {total_combinations} parameter combinations")

        combination_count = 0

        for tenkan, kijun, confidence, min_bars, cloud_settings in itertools.product(
            tenkan_periods, kijun_periods, confidence_thresholds, min_bars_options, cloud_filter_settings
        ):
            combination_count += 1

            try:
                # Create strategy with these parameters
                strategy = IchimokuStrategy(
                    backtest_mode=True,
                    epic=epic,
                    timeframe='15m',
                    use_optimized_parameters=False,
                    pipeline_mode=False
                )

                # Override parameters
                strategy.tenkan_period = tenkan
                strategy.kijun_period = kijun
                strategy.min_confidence = confidence
                strategy.min_bars = min_bars

                # Test signal generation
                signal_count = 0
                confidence_sum = 0

                # Test on multiple windows
                for i in range(min_bars + 50, len(df), 20):  # Test every 20 periods
                    test_df = df.iloc[:i+1].copy()

                    try:
                        signal = strategy.detect_signal(
                            df=test_df,
                            epic=epic,
                            spread_pips=1.5,
                            timeframe='15m'
                        )

                        if signal and signal.get('confidence', 0) > 0:
                            signal_count += 1
                            confidence_sum += signal.get('confidence', 0)

                    except Exception:
                        continue

                # Calculate score
                if signal_count > 0:
                    avg_confidence = confidence_sum / signal_count
                    # Ideal signal count is 10-30 for this period
                    signal_frequency_score = 1.0 - abs(signal_count - 20) / 30.0
                    signal_frequency_score = max(0.1, signal_frequency_score)

                    score = avg_confidence * signal_frequency_score * (signal_count / 100.0)
                else:
                    score = 0
                    avg_confidence = 0

                results.append({
                    'tenkan': tenkan,
                    'kijun': kijun,
                    'confidence': confidence,
                    'min_bars': min_bars,
                    'cloud_filter': cloud_settings,
                    'signal_count': signal_count,
                    'avg_confidence': avg_confidence,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_config = {
                        'tenkan_period': tenkan,
                        'kijun_period': kijun,
                        'confidence_threshold': confidence,
                        'min_bars': min_bars,
                        'cloud_filter_enabled': cloud_settings['enabled'],
                        'cloud_buffer_pips': cloud_settings.get('buffer_pips', 0),
                        'signal_count': signal_count,
                        'avg_confidence': avg_confidence,
                        'score': score
                    }

                if combination_count % 10 == 0:
                    logger.info(f"📊 Progress: {combination_count}/{total_combinations} - Best score: {best_score:.4f}")

            except Exception as e:
                logger.warning(f"⚠️ Error testing combination {combination_count}: {e}")
                continue

        # Report results
        logger.info("🏁 Optimization Complete!")
        logger.info("=" * 50)

        if best_config:
            logger.info(f"🎯 Best Configuration Found:")
            logger.info(f"   Tenkan Period: {best_config['tenkan_period']}")
            logger.info(f"   Kijun Period: {best_config['kijun_period']}")
            logger.info(f"   Confidence Threshold: {best_config['confidence_threshold']:.1%}")
            logger.info(f"   Min Bars: {best_config['min_bars']}")
            logger.info(f"   Cloud Filter: {'Enabled' if best_config['cloud_filter_enabled'] else 'Disabled'}")
            if best_config['cloud_filter_enabled']:
                logger.info(f"   Cloud Buffer: {best_config['cloud_buffer_pips']} pips")
            logger.info(f"   Signal Count: {best_config['signal_count']}")
            logger.info(f"   Avg Confidence: {best_config['avg_confidence']:.1%}")
            logger.info(f"   Optimization Score: {best_config['score']:.4f}")

            # Show top 5 configurations
            top_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
            logger.info(f"\n📈 Top 5 Configurations:")
            for i, result in enumerate(top_results, 1):
                logger.info(f"   {i}. T={result['tenkan']}, K={result['kijun']}, "
                          f"C={result['confidence']:.1%}, Signals={result['signal_count']}, "
                          f"Score={result['score']:.4f}")
        else:
            logger.error("❌ No valid configuration found")

        return best_config

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        return None

if __name__ == "__main__":
    result = optimize_ichimoku_for_epic()
    if result:
        print("✅ Optimization completed successfully!")
    else:
        print("❌ Optimization failed!")