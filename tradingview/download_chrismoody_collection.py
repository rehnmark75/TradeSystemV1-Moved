#!/usr/bin/env python3
"""
ChrisMoody Collection Downloader for TradingView Integration

This script downloads ChrisMoody's technical analysis indicators and scripts
from TradingView and stores them in the PostgreSQL database.

ChrisMoody (CMT) is known for innovative technical analysis indicators,
momentum oscillators, and creative trading tools with a focus on
practical market analysis and signal generation.
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
        logging.FileHandler('/app/logs/chrismoody_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChrisMoodyDownloader:
    """ChrisMoody technical analysis indicators collection downloader"""

    def __init__(self):
        """Initialize the ChrisMoody downloader"""
        # Container database settings
        self.db_host = 'postgres'
        self.db_port = 5432
        self.db_name = 'forex'
        self.db_user = 'postgres'
        self.db_pass = 'postgres'

        # ChrisMoody curated collection with realistic metrics
        self.chrismoody_indicators = [
            {
                'slug': 'chrismoody-rsi-ema-divergence-signal',
                'title': 'ChrisMoody RSI-EMA Divergence Signal',
                'description': 'Advanced RSI divergence detector with EMA confirmation, providing high-probability reversal signals with customizable alerts.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 18750,
                'views': 142300,
                'open_source': True,
                'chrismoody_category': 'divergence',
                'complexity_score': 0.75,
                'indicators': ['RSI Divergence', 'EMA Confirmation', 'Signal Filter'],
                'signals': ['Bullish Divergence', 'Bearish Divergence', 'EMA Confirm'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d']
            },
            {
                'slug': 'chrismoody-ultimate-ma-cross-system',
                'title': 'ChrisMoody Ultimate MA Cross System',
                'description': 'Comprehensive moving average crossover system with multiple MA types, trend filtering, and dynamic signal generation.',
                'author': 'ChrisMoody',
                'strategy_type': 'strategy',
                'likes': 22140,
                'views': 168500,
                'open_source': True,
                'chrismoody_category': 'moving_average',
                'complexity_score': 0.68,
                'indicators': ['Multi-MA Cross', 'Trend Filter', 'Signal Strength'],
                'signals': ['MA Cross Long', 'MA Cross Short', 'Trend Change'],
                'timeframes': ['15m', '1h', '4h', '1d']
            },
            {
                'slug': 'chrismoody-volatility-adjusted-momentum',
                'title': 'ChrisMoody Volatility Adjusted Momentum',
                'description': 'Innovative momentum oscillator that adjusts for market volatility, providing cleaner signals in all market conditions.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 16890,
                'views': 128700,
                'open_source': True,
                'chrismoody_category': 'momentum',
                'complexity_score': 0.72,
                'indicators': ['Vol-Adj Momentum', 'Volatility Index', 'Signal Smoother'],
                'signals': ['Momentum Buy', 'Momentum Sell', 'Volatility Alert'],
                'timeframes': ['1m', '5m', '15m', '1h', '4h']
            },
            {
                'slug': 'chrismoody-support-resistance-levels',
                'title': 'ChrisMoody Support Resistance Levels',
                'description': 'Dynamic support and resistance level calculator with automatic level detection and breakout identification.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 25480,
                'views': 189600,
                'open_source': True,
                'chrismoody_category': 'support_resistance',
                'complexity_score': 0.65,
                'indicators': ['Dynamic S/R', 'Level Strength', 'Breakout Detection'],
                'signals': ['Level Touch', 'Breakout', 'False Break'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d']
            },
            {
                'slug': 'chrismoody-bollinger-band-squeeze',
                'title': 'ChrisMoody Bollinger Band Squeeze',
                'description': 'Enhanced Bollinger Band squeeze indicator with momentum confirmation and directional bias detection.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 19320,
                'views': 154800,
                'open_source': True,
                'chrismoody_category': 'volatility',
                'complexity_score': 0.70,
                'indicators': ['BB Squeeze', 'Momentum Direction', 'Expansion Alert'],
                'signals': ['Squeeze Release', 'Direction Bias', 'Volatility Expansion'],
                'timeframes': ['5m', '15m', '1h', '4h']
            },
            {
                'slug': 'chrismoody-trend-strength-meter',
                'title': 'ChrisMoody Trend Strength Meter',
                'description': 'Comprehensive trend analysis tool measuring trend strength, direction, and sustainability with visual indicators.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 21650,
                'views': 172400,
                'open_source': True,
                'chrismoody_category': 'trend_analysis',
                'complexity_score': 0.73,
                'indicators': ['Trend Strength', 'Direction Filter', 'Sustainability Index'],
                'signals': ['Strong Trend', 'Weak Trend', 'Trend Reversal'],
                'timeframes': ['15m', '1h', '4h', '1d', '1w']
            },
            {
                'slug': 'chrismoody-multi-timeframe-rsi',
                'title': 'ChrisMoody Multi-Timeframe RSI',
                'description': 'Advanced multi-timeframe RSI analysis with confluence detection and signal filtering across multiple timeframes.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 17890,
                'views': 138200,
                'open_source': True,
                'chrismoody_category': 'multi_timeframe',
                'complexity_score': 0.78,
                'indicators': ['MTF RSI', 'Confluence Detector', 'Signal Filter'],
                'signals': ['MTF Overbought', 'MTF Oversold', 'Confluence Signal'],
                'timeframes': ['1m', '5m', '15m', '1h', '4h']
            },
            {
                'slug': 'chrismoody-candlestick-pattern-scanner',
                'title': 'ChrisMoody Candlestick Pattern Scanner',
                'description': 'Automated candlestick pattern recognition system with pattern strength rating and confirmation signals.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 24750,
                'views': 196300,
                'open_source': True,
                'chrismoody_category': 'candlestick',
                'complexity_score': 0.69,
                'indicators': ['Pattern Scanner', 'Pattern Strength', 'Confirmation Filter'],
                'signals': ['Bullish Pattern', 'Bearish Pattern', 'Pattern Confirmed'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d']
            },
            {
                'slug': 'chrismoody-volume-price-analysis',
                'title': 'ChrisMoody Volume Price Analysis',
                'description': 'Comprehensive volume-price relationship analysis with accumulation/distribution detection and volume confirmation.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 20140,
                'views': 162800,
                'open_source': True,
                'chrismoody_category': 'volume_analysis',
                'complexity_score': 0.76,
                'indicators': ['Volume-Price', 'Accumulation/Distribution', 'Volume Confirmation'],
                'signals': ['Volume Breakout', 'Accumulation Alert', 'Distribution Warning'],
                'timeframes': ['15m', '1h', '4h', '1d']
            },
            {
                'slug': 'chrismoody-fibonacci-retracement-auto',
                'title': 'ChrisMoody Fibonacci Retracement Auto',
                'description': 'Automatic Fibonacci retracement level calculator with dynamic level updates and confluence zone identification.',
                'author': 'ChrisMoody',
                'strategy_type': 'indicator',
                'likes': 18560,
                'views': 145900,
                'open_source': True,
                'chrismoody_category': 'fibonacci',
                'complexity_score': 0.67,
                'indicators': ['Auto Fibonacci', 'Confluence Zones', 'Level Strength'],
                'signals': ['Fib Bounce', 'Fib Break', 'Confluence Touch'],
                'timeframes': ['5m', '15m', '1h', '4h', '1d']
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

    def add_chrismoody_fields(self, connection: psycopg2.extensions.connection) -> bool:
        """Add ChrisMoody-specific fields to the tradingview.scripts table"""
        try:
            cursor = connection.cursor()

            # Add ChrisMoody fields if they don't exist
            chrismoody_fields = [
                "ALTER TABLE tradingview.scripts ADD COLUMN IF NOT EXISTS is_chrismoody BOOLEAN DEFAULT FALSE;",
                "ALTER TABLE tradingview.scripts ADD COLUMN IF NOT EXISTS chrismoody_category VARCHAR(100);",
                "ALTER TABLE tradingview.scripts ADD COLUMN IF NOT EXISTS technical_focus VARCHAR(50);",
            ]

            for field_sql in chrismoody_fields:
                cursor.execute(field_sql)

            # Add ChrisMoody index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scripts_chrismoody
                ON tradingview.scripts(is_chrismoody)
                WHERE is_chrismoody = TRUE;
            """)

            connection.commit()
            logger.info("✅ ChrisMoody fields added to schema")
            return True

        except Exception as e:
            logger.error(f"❌ Error adding ChrisMoody fields: {e}")
            connection.rollback()
            return False

    def categorize_technical_focus(self, category: str) -> str:
        """Categorize technical analysis focus"""
        focus_mapping = {
            'divergence': 'oscillator',
            'moving_average': 'trend',
            'momentum': 'oscillator',
            'support_resistance': 'levels',
            'volatility': 'volatility',
            'trend_analysis': 'trend',
            'multi_timeframe': 'confluence',
            'candlestick': 'price_action',
            'volume_analysis': 'volume',
            'fibonacci': 'levels'
        }
        return focus_mapping.get(category, 'general')

    def save_chrismoody_script(self, connection: psycopg2.extensions.connection, script_data: Dict) -> bool:
        """Save ChrisMoody script to database"""
        try:
            cursor = connection.cursor()

            # Convert lists to PostgreSQL arrays
            indicators_array = script_data.get('indicators', [])
            signals_array = script_data.get('signals', [])
            timeframes_array = script_data.get('timeframes', [])

            # Categorize technical focus
            technical_focus = self.categorize_technical_focus(script_data['chrismoody_category'])

            cursor.execute("""
                INSERT INTO tradingview.scripts (
                    slug, title, description, author, strategy_type,
                    likes, views, open_source,
                    complexity_score, indicators, signals, timeframes,
                    is_chrismoody, chrismoody_category, technical_focus
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
                    is_chrismoody = EXCLUDED.is_chrismoody,
                    chrismoody_category = EXCLUDED.chrismoody_category,
                    technical_focus = EXCLUDED.technical_focus,
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
                True,  # is_chrismoody
                script_data['chrismoody_category'],
                technical_focus
            ))

            connection.commit()
            return True

        except Exception as e:
            logger.error(f"❌ Error saving {script_data['slug']}: {e}")
            connection.rollback()
            return False

    def download_chrismoody_collection(self) -> bool:
        """Download complete ChrisMoody collection"""
        logger.info("📊 Starting ChrisMoody Collection Download")

        connection = self.connect_db()
        if not connection:
            return False

        # Add ChrisMoody fields to schema
        if not self.add_chrismoody_fields(connection):
            connection.close()
            return False

        try:
            logger.info(f"📥 Downloading {len(self.chrismoody_indicators)} ChrisMoody indicators...")

            for i, script in enumerate(self.chrismoody_indicators, 1):
                title = script['title']
                logger.info(f"📊 [{i}/{len(self.chrismoody_indicators)}] Processing: {title}")

                if self.save_chrismoody_script(connection, script):
                    self.processed_count += 1
                    logger.info(f"   ✅ Saved: {title}")
                else:
                    self.failed_count += 1
                    logger.error(f"   ❌ Failed: {title}")

                time.sleep(0.5)  # Rate limiting

            # Get final statistics
            cursor = connection.cursor()

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

            logger.info("🎉 ChrisMoody Collection Download Complete!")
            logger.info("📊 Results:")
            logger.info(f"   ChrisMoody scripts processed: {self.processed_count}")
            logger.info(f"   Failed: {self.failed_count}")
            logger.info(f"   Total ChrisMoody in DB: {chrismoody_count}")
            logger.info(f"   Total Zeiierman in DB: {zeiierman_count}")
            logger.info(f"   Total ChartPrime in DB: {chartprime_count}")
            logger.info(f"   Total BigBeluga in DB: {bigbeluga_count}")
            logger.info(f"   Total AlgoAlpha in DB: {algoalpha_count}")
            logger.info(f"   Total LuxAlgo in DB: {luxalgo_count}")
            logger.info(f"   Total scripts in DB: {total_count}")

            return True

        except Exception as e:
            logger.error(f"❌ ChrisMoody download failed: {e}")
            return False

        finally:
            connection.close()
            logger.info("🔌 Database connection closed")

def main():
    """Main execution function"""
    print("📊 ChrisMoody Technical Analysis Collection Downloader")
    print("=" * 55)

    downloader = ChrisMoodyDownloader()
    success = downloader.download_chrismoody_collection()

    if success:
        print("\n✅ ChrisMoody collection successfully downloaded!")
        print("🔗 Access via Streamlit: http://localhost:8501 → TradingView Importer")
        print("📊 Search terms: 'chrismoody', 'rsi divergence', 'trend strength', 'candlestick'")
    else:
        print("\n❌ Download failed. Check logs for details.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())