#!/usr/bin/env python3
"""
LuxAlgo Indicators Downloader for TradingView Integration

This script searches for and downloads all available LuxAlgo indicators and strategies
from TradingView, storing them in the PostgreSQL database for use with TradeSystemV1.

LuxAlgo is known for premium algorithmic trading indicators and tools.
This script focuses on their open-source contributions to the TradingView community.
"""

import os
import sys
import time
import logging
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/hr/Projects/TradeSystemV1/logs/tradingview/luxalgo_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LuxAlgoDownloader:
    """Specialized downloader for LuxAlgo indicators and strategies"""

    def __init__(self):
        """Initialize the LuxAlgo downloader"""
        self.db_host = 'localhost'  # External connection
        self.db_port = 5432
        self.db_name = 'forex'
        self.db_user = 'postgres'
        self.db_pass = 'postgres'

        # TradingView API configuration
        self.tv_api_base = "http://localhost:8080"

        # LuxAlgo search terms and variations
        self.luxalgo_terms = [
            'luxalgo',
            'lux algo',
            'lux-algo',
            'LuxAlgo',
            'LUX ALGO',
            'luxalgotm',  # TradingView username variations
            'author:LuxAlgo',
            'smart money concepts',
            'liquidity',
            'order blocks',
            'premium oscillator',
            'support resistance',
            'volume profile'
        ]

        # Common LuxAlgo indicator names/patterns
        self.luxalgo_indicators = [
            'premium',
            'oscillator',
            'liquidity',
            'volume profile',
            'order block',
            'market structure',
            'support resistance',
            'trend strength',
            'volatility',
            'momentum',
            'smart money',
            'institutional',
            'algo trading'
        ]

        self.downloaded_scripts = set()
        self.failed_downloads = []
        self.processed_count = 0

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

    def create_tables_if_not_exist(self, connection: psycopg2.extensions.connection):
        """Create necessary tables for storing TradingView scripts"""
        try:
            cursor = connection.cursor()

            # Create tvscripts table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tvscripts (
                    id SERIAL PRIMARY KEY,
                    slug VARCHAR(255) UNIQUE NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    author VARCHAR(255),
                    strategy_type VARCHAR(100),
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

                    -- LuxAlgo specific fields
                    is_luxalgo BOOLEAN DEFAULT FALSE,
                    luxalgo_category VARCHAR(100),
                    complexity_score FLOAT,

                    -- Search indexes
                    search_vector TSVECTOR
                );
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tvscripts_author ON tvscripts(author);
                CREATE INDEX IF NOT EXISTS idx_tvscripts_luxalgo ON tvscripts(is_luxalgo) WHERE is_luxalgo = TRUE;
                CREATE INDEX IF NOT EXISTS idx_tvscripts_search ON tvscripts USING gin(search_vector);
                CREATE INDEX IF NOT EXISTS idx_tvscripts_strategy_type ON tvscripts(strategy_type);
                CREATE INDEX IF NOT EXISTS idx_tvscripts_open_source ON tvscripts(open_source);
            """)

            connection.commit()
            logger.info("✅ Database tables ready")

        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            connection.rollback()

    def search_luxalgo_scripts(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for LuxAlgo scripts using the TradingView API"""
        try:
            search_url = f"{self.tv_api_base}/api/tvscripts/search"
            params = {
                'query': query,
                'limit': limit
            }

            logger.info(f"🔍 Searching for: '{query}' (limit: {limit})")
            response = requests.post(search_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                logger.info(f"   Found {len(results)} results")
                return results
            else:
                logger.error(f"   Search failed: HTTP {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"   Search error for '{query}': {e}")
            return []

    def get_script_details(self, slug: str) -> Optional[Dict]:
        """Get detailed script information"""
        try:
            details_url = f"{self.tv_api_base}/api/tvscripts/script/{slug}"
            response = requests.get(details_url, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"   Failed to get details for {slug}: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"   Details error for {slug}: {e}")
            return None

    def analyze_script(self, slug: str) -> Optional[Dict]:
        """Analyze script for patterns and complexity"""
        try:
            analyze_url = f"{self.tv_api_base}/api/tvscripts/analyze"
            params = {'slug': slug}
            response = requests.post(analyze_url, params=params, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"   Analysis failed for {slug}: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"   Analysis error for {slug}: {e}")
            return None

    def is_luxalgo_script(self, script: Dict) -> bool:
        """Determine if a script is from LuxAlgo or uses their methodology"""
        author = script.get('author', '').lower()
        title = script.get('title', '').lower()
        description = script.get('description', '').lower()

        # Direct LuxAlgo authorship
        luxalgo_authors = ['luxalgo', 'lux algo', 'lux-algo', 'luxalgotm']
        if any(luxalgo_author in author for luxalgo_author in luxalgo_authors):
            return True

        # LuxAlgo methodology indicators
        luxalgo_keywords = [
            'luxalgo', 'lux algo', 'smart money', 'order block',
            'liquidity', 'premium oscillator', 'market structure',
            'institutional', 'algorithmic trading'
        ]

        content = f"{title} {description}".lower()
        return any(keyword in content for keyword in luxalgo_keywords)

    def categorize_luxalgo_script(self, script: Dict, analysis: Optional[Dict] = None) -> str:
        """Categorize LuxAlgo script by type"""
        title = script.get('title', '').lower()
        description = script.get('description', '').lower()
        content = f"{title} {description}".lower()

        if any(term in content for term in ['oscillator', 'momentum', 'rsi', 'stochastic']):
            return 'oscillator'
        elif any(term in content for term in ['support', 'resistance', 'level', 'zone']):
            return 'support_resistance'
        elif any(term in content for term in ['volume', 'liquidity', 'flow']):
            return 'volume'
        elif any(term in content for term in ['trend', 'direction', 'bias']):
            return 'trend'
        elif any(term in content for term in ['volatility', 'atr', 'range']):
            return 'volatility'
        elif any(term in content for term in ['order', 'block', 'institutional', 'smart money']):
            return 'order_flow'
        else:
            return 'general'

    def save_script_to_db(self, connection: psycopg2.extensions.connection, script: Dict,
                          details: Optional[Dict] = None, analysis: Optional[Dict] = None) -> bool:
        """Save script to database"""
        try:
            cursor = connection.cursor()

            # Prepare script data
            slug = script.get('slug', '')
            title = script.get('title', 'Unknown Title')
            author = script.get('author', 'Unknown')
            description = script.get('description', '')

            # Enhanced data from details
            source_code = ''
            pine_version = 'v5'
            published_at = None
            updated_at = None
            source_url = ''

            if details:
                source_code = details.get('source_code', '')
                pine_version = details.get('pine_version', 'v5')
                published_at = details.get('published_at')
                updated_at = details.get('updated_at')
                source_url = details.get('source_url', '')

            # Analysis data
            complexity_score = 0.5
            if analysis and 'signals' in analysis:
                complexity_score = analysis['signals'].get('complexity_score', 0.5)

            # LuxAlgo classification
            is_luxalgo = self.is_luxalgo_script(script)
            luxalgo_category = self.categorize_luxalgo_script(script, analysis) if is_luxalgo else None

            # Insert or update script
            cursor.execute("""
                INSERT INTO tvscripts (
                    slug, title, description, author, strategy_type,
                    pine_version, source_code, likes, views, uses, comments,
                    open_source, published_at, updated_at, tags, categories,
                    source_url, is_luxalgo, luxalgo_category, complexity_score
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (slug) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    author = EXCLUDED.author,
                    source_code = EXCLUDED.source_code,
                    likes = EXCLUDED.likes,
                    views = EXCLUDED.views,
                    uses = EXCLUDED.uses,
                    comments = EXCLUDED.comments,
                    updated_at = EXCLUDED.updated_at,
                    is_luxalgo = EXCLUDED.is_luxalgo,
                    luxalgo_category = EXCLUDED.luxalgo_category,
                    complexity_score = EXCLUDED.complexity_score,
                    scraped_at = CURRENT_TIMESTAMP
            """, (
                slug, title, description, author, script.get('strategy_type', 'indicator'),
                pine_version, source_code, script.get('likes', 0), script.get('views', 0),
                script.get('uses', 0), script.get('comments', 0), script.get('open_source', False),
                published_at, updated_at, script.get('tags', []), script.get('categories', []),
                source_url, is_luxalgo, luxalgo_category, complexity_score
            ))

            connection.commit()
            return True

        except Exception as e:
            logger.error(f"   Database save error for {slug}: {e}")
            connection.rollback()
            return False

    def download_comprehensive_luxalgo_collection(self):
        """Download comprehensive collection of LuxAlgo indicators and strategies"""
        logger.info("🚀 Starting comprehensive LuxAlgo collection download")

        connection = self.connect_db()
        if not connection:
            return False

        self.create_tables_if_not_exist(connection)

        try:
            all_scripts = []

            # Phase 1: Direct LuxAlgo searches
            logger.info("📊 Phase 1: Direct LuxAlgo searches")
            for term in self.luxalgo_terms:
                scripts = self.search_luxalgo_scripts(term, limit=100)
                for script in scripts:
                    if script.get('slug') not in self.downloaded_scripts:
                        all_scripts.append(script)
                        self.downloaded_scripts.add(script.get('slug'))

                time.sleep(1)  # Rate limiting

            # Phase 2: LuxAlgo methodology searches
            logger.info("📊 Phase 2: LuxAlgo methodology indicators")
            methodology_terms = [
                'smart money concepts',
                'order blocks',
                'liquidity zones',
                'market structure',
                'institutional trading',
                'premium oscillator',
                'support resistance levels',
                'volume profile analysis',
                'trend strength indicator',
                'volatility analysis'
            ]

            for term in methodology_terms:
                scripts = self.search_luxalgo_scripts(term, limit=50)
                for script in scripts:
                    if script.get('slug') not in self.downloaded_scripts:
                        if self.is_luxalgo_script(script):  # Filter for LuxAlgo-style scripts
                            all_scripts.append(script)
                            self.downloaded_scripts.add(script.get('slug'))

                time.sleep(1)  # Rate limiting

            logger.info(f"📊 Total unique scripts found: {len(all_scripts)}")

            # Phase 3: Process and download each script
            logger.info("📊 Phase 3: Processing and downloading scripts")

            luxalgo_count = 0
            for i, script in enumerate(all_scripts, 1):
                slug = script.get('slug', '')
                title = script.get('title', 'Unknown')
                author = script.get('author', 'Unknown')

                logger.info(f"📥 [{i}/{len(all_scripts)}] Processing: {title} by {author}")

                # Get detailed information
                details = self.get_script_details(slug)
                analysis = self.analyze_script(slug)

                # Save to database
                if self.save_script_to_db(connection, script, details, analysis):
                    self.processed_count += 1
                    if self.is_luxalgo_script(script):
                        luxalgo_count += 1
                        logger.info(f"   ✅ LuxAlgo script saved")
                    else:
                        logger.info(f"   ✅ Related script saved")
                else:
                    self.failed_downloads.append(slug)
                    logger.error(f"   ❌ Failed to save")

                # Rate limiting and progress
                time.sleep(0.5)
                if i % 20 == 0:
                    logger.info(f"📊 Progress: {i}/{len(all_scripts)} processed")

            # Final summary
            logger.info("🎉 LuxAlgo collection download completed!")
            logger.info(f"📊 Statistics:")
            logger.info(f"   Total scripts processed: {self.processed_count}")
            logger.info(f"   LuxAlgo scripts found: {luxalgo_count}")
            logger.info(f"   Failed downloads: {len(self.failed_downloads)}")

            if self.failed_downloads:
                logger.warning(f"⚠️  Failed slugs: {', '.join(self.failed_downloads[:10])}")

            return True

        except Exception as e:
            logger.error(f"❌ Download process failed: {e}")
            return False

        finally:
            connection.close()
            logger.info("🔌 Database connection closed")

    def get_collection_stats(self) -> Dict:
        """Get statistics about the downloaded LuxAlgo collection"""
        connection = self.connect_db()
        if not connection:
            return {}

        try:
            cursor = connection.cursor()

            # Get comprehensive stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_scripts,
                    COUNT(CASE WHEN is_luxalgo = TRUE THEN 1 END) as luxalgo_scripts,
                    COUNT(CASE WHEN open_source = TRUE THEN 1 END) as open_source_scripts,
                    COUNT(CASE WHEN is_luxalgo = TRUE AND open_source = TRUE THEN 1 END) as luxalgo_open_source,
                    AVG(complexity_score) as avg_complexity,
                    SUM(likes) as total_likes,
                    SUM(views) as total_views
                FROM tvscripts;
            """)

            stats = cursor.fetchone()

            # Get LuxAlgo categories
            cursor.execute("""
                SELECT luxalgo_category, COUNT(*)
                FROM tvscripts
                WHERE is_luxalgo = TRUE AND luxalgo_category IS NOT NULL
                GROUP BY luxalgo_category
                ORDER BY COUNT(*) DESC;
            """)

            categories = dict(cursor.fetchall())

            # Get top authors
            cursor.execute("""
                SELECT author, COUNT(*)
                FROM tvscripts
                WHERE is_luxalgo = TRUE
                GROUP BY author
                ORDER BY COUNT(*) DESC
                LIMIT 10;
            """)

            top_authors = dict(cursor.fetchall())

            return {
                'total_scripts': stats[0] or 0,
                'luxalgo_scripts': stats[1] or 0,
                'open_source_scripts': stats[2] or 0,
                'luxalgo_open_source': stats[3] or 0,
                'avg_complexity': float(stats[4] or 0),
                'total_likes': stats[5] or 0,
                'total_views': stats[6] or 0,
                'categories': categories,
                'top_authors': top_authors
            }

        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {}

        finally:
            connection.close()

def main():
    """Main execution function"""
    print("🔥 LuxAlgo Indicators Downloader for TradeSystemV1")
    print("=" * 50)

    downloader = LuxAlgoDownloader()

    # Download comprehensive LuxAlgo collection
    success = downloader.download_comprehensive_luxalgo_collection()

    if success:
        print("\n📊 Getting collection statistics...")
        stats = downloader.get_collection_stats()

        print(f"\n🎉 LuxAlgo Collection Statistics:")
        print(f"   Total Scripts: {stats.get('total_scripts', 0):,}")
        print(f"   LuxAlgo Scripts: {stats.get('luxalgo_scripts', 0):,}")
        print(f"   Open Source: {stats.get('open_source_scripts', 0):,}")
        print(f"   LuxAlgo Open Source: {stats.get('luxalgo_open_source', 0):,}")
        print(f"   Average Complexity: {stats.get('avg_complexity', 0):.2f}")
        print(f"   Total Likes: {stats.get('total_likes', 0):,}")
        print(f"   Total Views: {stats.get('total_views', 0):,}")

        if stats.get('categories'):
            print(f"\n📋 LuxAlgo Categories:")
            for category, count in list(stats['categories'].items())[:5]:
                print(f"   {category}: {count}")

        if stats.get('top_authors'):
            print(f"\n👥 Top Authors:")
            for author, count in list(stats['top_authors'].items())[:5]:
                print(f"   {author}: {count}")

        print(f"\n✅ LuxAlgo collection successfully downloaded!")
        print(f"🔗 Access via Streamlit: http://localhost:8501 → TradingView Importer")

    else:
        print(f"\n❌ Download failed. Check logs for details.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())