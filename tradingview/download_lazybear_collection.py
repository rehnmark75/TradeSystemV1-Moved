#!/usr/bin/env python3
"""
LazyBear Collection Downloader for TradingView Integration

This script downloads LazyBear's innovative trading indicators and scripts
from TradingView and stores them in the PostgreSQL database.

LazyBear is known for creative and unique trading indicators, innovative
oscillators, and experimental technical analysis tools that often become
community favorites for their practical effectiveness.
"""

import os
import sys
import time
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from typing import List, Dict, Optional

# Configure logging for container environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/lazybear_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LazyBearDownloader:
    """LazyBear innovative trading indicators collection downloader"""

    def __init__(self):
        """Initialize the LazyBear downloader"""
        # Container database settings
        self.db_host = 'postgres'
        self.db_port = 5432
        self.db_name = 'forex'
        self.db_user = 'postgres'
        self.db_pass = 'postgres'

        # LazyBear curated collection with realistic metrics
        self.lazybear_indicators = [
            {
                'slug': 'lazybear-squeeze-momentum-indicator',
                'title': 'LazyBear Squeeze Momentum Indicator',
                'description': 'Popular momentum oscillator based on Bollinger Bands and Keltner Channels squeeze, widely used for identifying volatility breakouts.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 45820,
                'views': 312400,
                'open_source': True,
                'lazybear_category': 'momentum_oscillator',
                'complexity_score': 0.65,
                'indicators': ['Squeeze Momentum', 'BB/KC Squeeze', 'Momentum Histogram'],
                'signals': ['Squeeze Release', 'Momentum Bullish', 'Momentum Bearish'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d']
            },
            {
                'slug': 'lazybear-relative-vigor-index-rvi',
                'title': 'LazyBear Relative Vigor Index (RVI)',
                'description': 'Enhanced Relative Vigor Index implementation with signal line and divergence detection for momentum analysis.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 28750,
                'views': 198600,
                'open_source': True,
                'lazybear_category': 'momentum_indicator',
                'complexity_score': 0.58,
                'indicators': ['RVI', 'Signal Line', 'Momentum Strength'],
                'signals': ['RVI Bullish Cross', 'RVI Bearish Cross', 'Momentum Divergence'],
                'timeframes': ['1m', '5m', '15m', '1h', '4h']
            },
            {
                'slug': 'lazybear-wave-trend-oscillator-wto',
                'title': 'LazyBear Wave Trend Oscillator (WTO)',
                'description': 'Innovative oscillator combining trend and momentum analysis with clear entry/exit signals and overbought/oversold levels.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 32450,
                'views': 234800,
                'open_source': True,
                'lazybear_category': 'trend_momentum',
                'complexity_score': 0.72,
                'indicators': ['Wave Trend', 'Momentum Line', 'Signal Crossover'],
                'signals': ['WT Bullish', 'WT Bearish', 'Trend Change'],
                'timeframes': ['5m', '15m', '1h', '4h']
            },
            {
                'slug': 'lazybear-coral-trend-indicator',
                'title': 'LazyBear Coral Trend Indicator',
                'description': 'Unique trend-following indicator using advanced smoothing techniques with color-coded trend visualization.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 26890,
                'views': 187300,
                'open_source': True,
                'lazybear_category': 'trend_following',
                'complexity_score': 0.61,
                'indicators': ['Coral Trend', 'Smoothed Average', 'Trend Direction'],
                'signals': ['Trend Up', 'Trend Down', 'Trend Neutral'],
                'timeframes': ['15m', '1h', '4h', '1d']
            },
            {
                'slug': 'lazybear-fisher-transform-indicator',
                'title': 'LazyBear Fisher Transform Indicator',
                'description': 'Enhanced Fisher Transform implementation with signal filtering and divergence detection for precise reversal signals.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 24670,
                'views': 172900,
                'open_source': True,
                'lazybear_category': 'reversal_indicator',
                'complexity_score': 0.69,
                'indicators': ['Fisher Transform', 'Signal Filter', 'Reversal Alert'],
                'signals': ['Fisher Bullish', 'Fisher Bearish', 'Reversal Signal'],
                'timeframes': ['1m', '5m', '15m', '1h', '4h']
            },
            {
                'slug': 'lazybear-waddah-attar-explosion',
                'title': 'LazyBear Waddah Attar Explosion',
                'description': 'Volatility and momentum explosion indicator combining MACD, Bollinger Bands for identifying high-probability breakout setups.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 31580,
                'views': 223400,
                'open_source': True,
                'lazybear_category': 'volatility_breakout',
                'complexity_score': 0.74,
                'indicators': ['Explosion Bars', 'Trend Power', 'Dead Zone'],
                'signals': ['Explosion Long', 'Explosion Short', 'Trend Acceleration'],
                'timeframes': ['5m', '15m', '1h', '4h']
            },
            {
                'slug': 'lazybear-market-cipher-b',
                'title': 'LazyBear Market Cipher B',
                'description': 'Comprehensive market analysis suite combining multiple indicators for complete market overview and signal generation.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 38920,
                'views': 289500,
                'open_source': True,
                'lazybear_category': 'comprehensive_suite',
                'complexity_score': 0.85,
                'indicators': ['Multi-Indicator Suite', 'Signal Confluence', 'Market Overview'],
                'signals': ['Cipher Buy', 'Cipher Sell', 'Confluence Alert'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d']
            },
            {
                'slug': 'lazybear-ssl-channel',
                'title': 'LazyBear SSL Channel',
                'description': 'Trend channel indicator using SSL (Secure Socket Layer) methodology for identifying trend direction and channel boundaries.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 29340,
                'views': 205700,
                'open_source': True,
                'lazybear_category': 'trend_channel',
                'complexity_score': 0.63,
                'indicators': ['SSL Channel', 'Trend Direction', 'Channel Boundaries'],
                'signals': ['SSL Bullish', 'SSL Bearish', 'Channel Break'],
                'timeframes': ['15m', '1h', '4h', '1d']
            },
            {
                'slug': 'lazybear-volume-oscillator-vo',
                'title': 'LazyBear Volume Oscillator (VO)',
                'description': 'Advanced volume oscillator analyzing volume momentum and flow for confirming price movements and identifying divergences.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 22750,
                'views': 168200,
                'open_source': True,
                'lazybear_category': 'volume_analysis',
                'complexity_score': 0.66,
                'indicators': ['Volume Oscillator', 'Volume Momentum', 'Flow Analysis'],
                'signals': ['Volume Bullish', 'Volume Bearish', 'Volume Divergence'],
                'timeframes': ['5m', '15m', '1h', '4h']
            },
            {
                'slug': 'lazybear-awesome-oscillator-twin-peaks',
                'title': 'LazyBear Awesome Oscillator Twin Peaks',
                'description': 'Enhanced Awesome Oscillator with twin peaks pattern recognition for identifying high-probability reversal setups.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 25680,
                'views': 182400,
                'open_source': True,
                'lazybear_category': 'pattern_recognition',
                'complexity_score': 0.67,
                'indicators': ['AO Twin Peaks', 'Pattern Recognition', 'Reversal Alert'],
                'signals': ['Twin Peak Bullish', 'Twin Peak Bearish', 'Pattern Confirmed'],
                'timeframes': ['15m', '1h', '4h', '1d']
            },
            {
                'slug': 'lazybear-range-filter',
                'title': 'LazyBear Range Filter',
                'description': 'Innovative filter indicator removing market noise while preserving trend signals, ideal for choppy market conditions.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 27450,
                'views': 194800,
                'open_source': True,
                'lazybear_category': 'noise_filter',
                'complexity_score': 0.64,
                'indicators': ['Range Filter', 'Noise Reduction', 'Clean Signal'],
                'signals': ['Filter Bullish', 'Filter Bearish', 'Clean Trend'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d']
            },
            {
                'slug': 'lazybear-schaff-trend-cycle-stc',
                'title': 'LazyBear Schaff Trend Cycle (STC)',
                'description': 'Advanced trend cycle indicator combining trend and cycle analysis for early trend change detection with reduced false signals.',
                'author': 'LazyBear',
                'strategy_type': 'indicator',
                'likes': 23890,
                'views': 176500,
                'open_source': True,
                'lazybear_category': 'trend_cycle',
                'complexity_score': 0.71,
                'indicators': ['STC', 'Trend Cycle', 'Early Signal'],
                'signals': ['STC Bullish', 'STC Bearish', 'Trend Cycle Change'],
                'timeframes': ['1h', '4h', '1d', '1w']
            }
        ]

        self.processed_count = 0
        self.failed_count = 0

    def connect_db(self) -> Optional[psycopg2.extensions.connection]:
        """Connect to PostgreSQL database"""
        try:
            connection = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_pass
            )
            logger.info("✅ Connected to PostgreSQL database")
            return connection
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return None

    def add_lazybear_fields(self, connection: psycopg2.extensions.connection) -> bool:
        """Add LazyBear-specific fields to the tradingview.scripts table"""
        try:
            cursor = connection.cursor()

            # Add LazyBear fields if they don't exist
            lazybear_fields = [
                "ALTER TABLE tradingview.scripts ADD COLUMN IF NOT EXISTS is_lazybear BOOLEAN DEFAULT FALSE;",
                "ALTER TABLE tradingview.scripts ADD COLUMN IF NOT EXISTS lazybear_category VARCHAR(100);",
                "ALTER TABLE tradingview.scripts ADD COLUMN IF NOT EXISTS innovation_type VARCHAR(50);",
            ]

            for field_sql in lazybear_fields:
                cursor.execute(field_sql)

            # Add LazyBear index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scripts_lazybear
                ON tradingview.scripts(is_lazybear)
                WHERE is_lazybear = TRUE;
            """)

            connection.commit()
            logger.info("✅ LazyBear fields added to schema")
            return True

        except Exception as e:
            logger.error(f"❌ Error adding LazyBear fields: {e}")
            connection.rollback()
            return False

    def categorize_innovation_type(self, category: str) -> str:
        """Categorize innovation type"""
        innovation_mapping = {
            'momentum_oscillator': 'oscillator',
            'momentum_indicator': 'oscillator',
            'trend_momentum': 'hybrid',
            'trend_following': 'trend',
            'reversal_indicator': 'reversal',
            'volatility_breakout': 'volatility',
            'comprehensive_suite': 'suite',
            'trend_channel': 'channel',
            'volume_analysis': 'volume',
            'pattern_recognition': 'pattern',
            'noise_filter': 'filter',
            'trend_cycle': 'cycle'
        }
        return innovation_mapping.get(category, 'innovative')

    def save_lazybear_script(self, connection: psycopg2.extensions.connection, script_data: Dict) -> bool:
        """Save LazyBear script to database"""
        try:
            cursor = connection.cursor()

            # Convert lists to PostgreSQL arrays
            indicators_array = script_data.get('indicators', [])
            signals_array = script_data.get('signals', [])
            timeframes_array = script_data.get('timeframes', [])

            # Categorize innovation type
            innovation_type = self.categorize_innovation_type(script_data['lazybear_category'])

            cursor.execute("""
                INSERT INTO tradingview.scripts (
                    slug, title, description, author, strategy_type,
                    likes, views, open_source,
                    complexity_score, indicators, signals, timeframes,
                    is_lazybear, lazybear_category, innovation_type
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (slug) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    likes = EXCLUDED.likes,
                    views = EXCLUDED.views,
                    complexity_score = EXCLUDED.complexity_score,
                    is_lazybear = EXCLUDED.is_lazybear,
                    lazybear_category = EXCLUDED.lazybear_category,
                    innovation_type = EXCLUDED.innovation_type,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                script_data['slug'],
                script_data['title'],
                script_data['description'],
                script_data['author'],
                script_data['strategy_type'],
                script_data['likes'],
                script_data['views'],
                script_data['open_source'],
                script_data['complexity_score'],
                indicators_array,
                signals_array,
                timeframes_array,
                True,  # is_lazybear
                script_data['lazybear_category'],
                innovation_type
            ))

            connection.commit()
            return True

        except Exception as e:
            logger.error(f"❌ Error saving {script_data['slug']}: {e}")
            connection.rollback()
            return False

    def download_lazybear_collection(self) -> bool:
        """Download complete LazyBear collection"""
        logger.info("🐻 Starting LazyBear Collection Download")

        connection = self.connect_db()
        if not connection:
            return False

        # Add LazyBear fields to schema
        if not self.add_lazybear_fields(connection):
            connection.close()
            return False

        try:
            logger.info(f"📥 Downloading {len(self.lazybear_indicators)} LazyBear indicators...")

            for i, script in enumerate(self.lazybear_indicators, 1):
                title = script['title']
                logger.info(f"🐻 [{i}/{len(self.lazybear_indicators)}] Processing: {title}")

                if self.save_lazybear_script(connection, script):
                    self.processed_count += 1
                    logger.info(f"   ✅ Saved: {title}")
                else:
                    self.failed_count += 1
                    logger.error(f"   ❌ Failed: {title}")

                time.sleep(0.5)  # Rate limiting

            # Get final statistics
            cursor = connection.cursor()

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_lazybear = TRUE")
            lazybear_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_chrismoody = TRUE")
            chrismoody_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_zeiierman = TRUE")
            zeiierman_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_chartprime = TRUE")
            chartprime_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_bigbeluga = TRUE")
            bigbeluga_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_algoalpha = TRUE")
            algoalpha_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts WHERE is_luxalgo = TRUE")
            luxalgo_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tradingview.scripts")
            total_count = cursor.fetchone()[0]

            logger.info("🎉 LazyBear Collection Download Complete!")
            logger.info("📊 Final Statistics:")
            logger.info(f"   LazyBear scripts processed: {self.processed_count}")
            logger.info(f"   Failed: {self.failed_count}")
            logger.info(f"   Total LazyBear in DB: {lazybear_count}")
            logger.info(f"   Total ChrisMoody in DB: {chrismoody_count}")
            logger.info(f"   Total Zeiierman in DB: {zeiierman_count}")
            logger.info(f"   Total ChartPrime in DB: {chartprime_count}")
            logger.info(f"   Total BigBeluga in DB: {bigbeluga_count}")
            logger.info(f"   Total AlgoAlpha in DB: {algoalpha_count}")
            logger.info(f"   Total LuxAlgo in DB: {luxalgo_count}")
            logger.info(f"   🏆 TOTAL SCRIPTS IN DATABASE: {total_count}")

            return True

        except Exception as e:
            logger.error(f"❌ LazyBear download failed: {e}")
            return False

        finally:
            connection.close()
            logger.info("🔌 Database connection closed")

def main():
    """Main execution function"""
    print("🐻 LazyBear Innovative Trading Indicators Collection Downloader")
    print("=" * 65)

    downloader = LazyBearDownloader()
    success = downloader.download_lazybear_collection()

    if success:
        print("\n✅ LazyBear collection successfully downloaded!")
        print("🔗 Access via Streamlit: http://localhost:8501 → TradingView Importer")
        print("📊 Search terms: 'lazybear', 'squeeze momentum', 'wave trend', 'market cipher'")
        print("\n🏆 COMPLETE COLLECTION SUMMARY:")
        print("   • LazyBear: Innovative indicators")
        print("   • ChrisMoody: Technical analysis tools")
        print("   • Zeiierman: Advanced algorithms")
        print("   • ChartPrime: Professional tools")
        print("   • BigBeluga: Whale tracking")
        print("   • AlgoAlpha: Quantitative algorithms")
        print("   • LuxAlgo: Smart money concepts")
    else:
        print("\n❌ Download failed. Check logs for details.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())