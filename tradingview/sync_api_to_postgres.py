#!/usr/bin/env python3
"""
Sync TradingView API Data to PostgreSQL

This script syncs the existing TradingView API database (SQLite) to PostgreSQL
so we can search and analyze the data with our LuxAlgo identification algorithms.
"""

import os
import sys
import time
import logging
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/sync_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIToPostgreSQLSyncer:
    """Sync TradingView API data to PostgreSQL"""

    def __init__(self):
        """Initialize the syncer"""
        # Database configuration
        self.db_host = 'postgres'
        self.db_port = 5432
        self.db_name = 'forex'
        self.db_user = 'postgres'
        self.db_pass = 'postgres'

        # API configuration
        self.tv_api_base = "http://localhost:8080"

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

    def get_all_scripts_from_api(self) -> List[Dict]:
        """Get all scripts from the TradingView API"""
        all_scripts = []

        # First, get overall stats
        try:
            response = requests.get(f"{self.tv_api_base}/api/tvscripts/stats", timeout=30)
            if response.status_code == 200:
                stats = response.json()
                total_scripts = stats.get('total_scripts', 0)
                logger.info(f"📊 API contains {total_scripts} scripts")
            else:
                logger.warning("Could not get API stats")
                total_scripts = 100  # fallback
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            total_scripts = 100

        # Search with common terms to get all scripts
        search_terms = [
            'strategy', 'indicator', 'EMA', 'RSI', 'MACD', 'Ichimoku', 'VWAP',
            'trend', 'momentum', 'oscillator', 'volume', 'volatility'
        ]

        all_slugs = set()

        for term in search_terms:
            try:
                logger.info(f"🔍 Searching API for: '{term}'")
                response = requests.post(
                    f"{self.tv_api_base}/api/tvscripts/search",
                    params={'query': term, 'limit': 50},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    logger.info(f"   Found {len(results)} results")

                    for script in results:
                        slug = script.get('slug')
                        if slug and slug not in all_slugs:
                            all_scripts.append(script)
                            all_slugs.add(slug)
                else:
                    logger.warning(f"Search failed for '{term}': HTTP {response.status_code}")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error searching for '{term}': {e}")

        logger.info(f"📊 Total unique scripts collected: {len(all_scripts)}")
        return all_scripts

    def is_luxalgo_style(self, script: Dict) -> bool:
        """Check if script matches LuxAlgo methodology"""
        author = script.get('author', '').lower()
        title = script.get('title', '').lower()
        description = script.get('description', '').lower()

        # Direct LuxAlgo references
        direct_terms = ['luxalgo', 'lux algo', 'lux-algo']
        if any(term in author or term in title for term in direct_terms):
            return True

        # LuxAlgo methodology indicators
        content = f"{title} {description}".lower()
        luxalgo_concepts = [
            'smart money', 'order block', 'liquidity', 'premium',
            'market structure', 'institutional', 'algo trading',
            'volume profile', 'support resistance'
        ]

        return any(concept in content for concept in luxalgo_concepts)

    def categorize_script(self, script: Dict) -> str:
        """Categorize script type"""
        title = script.get('title', '').lower()
        description = script.get('description', '').lower()
        content = f"{title} {description}".lower()

        if any(term in content for term in ['oscillator', 'rsi', 'stochastic']):
            return 'oscillator'
        elif any(term in content for term in ['ema', 'sma', 'moving average']):
            return 'moving_average'
        elif any(term in content for term in ['macd', 'momentum']):
            return 'momentum'
        elif any(term in content for term in ['volume', 'vwap']):
            return 'volume'
        elif any(term in content for term in ['ichimoku', 'cloud']):
            return 'ichimoku'
        elif any(term in content for term in ['support', 'resistance']):
            return 'support_resistance'
        else:
            return 'general'

    def create_tables(self, connection: psycopg2.extensions.connection):
        """Create/update database tables"""
        try:
            cursor = connection.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tvscripts (
                    id SERIAL PRIMARY KEY,
                    slug VARCHAR(255) UNIQUE NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    author VARCHAR(255),
                    strategy_type VARCHAR(100),
                    script_type VARCHAR(50),
                    pine_version VARCHAR(20),
                    source_code TEXT,
                    likes INTEGER DEFAULT 0,
                    views INTEGER DEFAULT 0,
                    uses INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    open_source BOOLEAN DEFAULT FALSE,
                    published_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    tags TEXT[],
                    categories TEXT[],
                    source_url VARCHAR(500),
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Analysis fields
                    is_luxalgo BOOLEAN DEFAULT FALSE,
                    luxalgo_category VARCHAR(100),
                    complexity_score FLOAT DEFAULT 0.5,
                    indicators TEXT[],
                    signals TEXT[],
                    timeframes TEXT[],

                    -- Search
                    search_vector TSVECTOR
                );
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tvscripts_author ON tvscripts(author);
                CREATE INDEX IF NOT EXISTS idx_tvscripts_luxalgo ON tvscripts(is_luxalgo) WHERE is_luxalgo = TRUE;
                CREATE INDEX IF NOT EXISTS idx_tvscripts_strategy_type ON tvscripts(strategy_type);
                CREATE INDEX IF NOT EXISTS idx_tvscripts_script_type ON tvscripts(script_type);
                CREATE INDEX IF NOT EXISTS idx_tvscripts_open_source ON tvscripts(open_source);
            """)

            connection.commit()
            logger.info("✅ Database tables ready")

        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            connection.rollback()

    def save_script(self, connection: psycopg2.extensions.connection, script: Dict) -> bool:
        """Save script to PostgreSQL"""
        try:
            cursor = connection.cursor()

            slug = script.get('slug', '')
            title = script.get('title', 'Unknown')
            author = script.get('author', 'Unknown')
            description = script.get('description', '')

            # Analyze for LuxAlgo content
            is_luxalgo = self.is_luxalgo_style(script)
            luxalgo_category = self.categorize_script(script) if is_luxalgo else None

            cursor.execute("""
                INSERT INTO tvscripts (
                    slug, title, description, author, strategy_type, script_type,
                    likes, views, open_source, source_url,
                    is_luxalgo, luxalgo_category, indicators, signals, timeframes
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (slug) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    likes = EXCLUDED.likes,
                    views = EXCLUDED.views,
                    is_luxalgo = EXCLUDED.is_luxalgo,
                    luxalgo_category = EXCLUDED.luxalgo_category,
                    scraped_at = CURRENT_TIMESTAMP
            """, (
                slug, title, description, author,
                script.get('strategy_type', 'indicator'),
                script.get('script_type', 'indicator'),
                script.get('likes', 0),
                script.get('views', 0),
                script.get('open_source', False),
                script.get('url', ''),
                is_luxalgo, luxalgo_category,
                script.get('indicators', []),
                script.get('signals', []),
                script.get('timeframes', [])
            ))

            connection.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving script {slug}: {e}")
            connection.rollback()
            return False

    def sync_api_data(self):
        """Main sync process"""
        logger.info("🔄 Starting API to PostgreSQL sync")

        connection = self.connect_db()
        if not connection:
            return False

        self.create_tables(connection)

        try:
            # Get all scripts from API
            scripts = self.get_all_scripts_from_api()

            if not scripts:
                logger.warning("No scripts found in API")
                return False

            # Save to PostgreSQL
            logger.info("📥 Saving scripts to PostgreSQL...")
            saved_count = 0
            luxalgo_count = 0

            for i, script in enumerate(scripts, 1):
                if self.save_script(connection, script):
                    saved_count += 1
                    if self.is_luxalgo_style(script):
                        luxalgo_count += 1
                        logger.info(f"   📈 LuxAlgo-style script: {script.get('title')}")

                if i % 5 == 0:
                    logger.info(f"   Progress: {i}/{len(scripts)} scripts processed")

            # Final summary
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM tvscripts")
            total_in_db = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tvscripts WHERE is_luxalgo = TRUE")
            luxalgo_in_db = cursor.fetchone()[0]

            logger.info("🎉 Sync completed!")
            logger.info(f"📊 Results:")
            logger.info(f"   Scripts in API: {len(scripts)}")
            logger.info(f"   Scripts saved: {saved_count}")
            logger.info(f"   Total in DB: {total_in_db}")
            logger.info(f"   LuxAlgo-style found: {luxalgo_in_db}")

            return True

        except Exception as e:
            logger.error(f"❌ Sync process failed: {e}")
            return False

        finally:
            connection.close()

def main():
    """Main execution"""
    print("🔄 TradingView API to PostgreSQL Sync")
    print("=" * 40)

    syncer = APIToPostgreSQLSyncer()
    success = syncer.sync_api_data()

    if success:
        print("\n✅ Sync completed successfully!")
        print("🔗 Data now available in PostgreSQL for LuxAlgo analysis")
    else:
        print("\n❌ Sync failed. Check logs for details.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())