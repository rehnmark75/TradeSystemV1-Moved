#!/usr/bin/env python3
"""
TradingView Scripts PostgreSQL Migration

Migrates TradingView scripts from SQLite to PostgreSQL and sets up
the proper schema for integration with TradeSystemV1.
"""

import os
import sys
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import uuid

# Database imports
try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:
    print("❌ PostgreSQL adapter not available. Installing...")
    os.system("pip install psycopg2-binary")
    import psycopg2
    from psycopg2.extras import Json, execute_values

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingViewPostgreSQLMigrator:
    """Handles migration of TradingView data to PostgreSQL"""
    
    def __init__(self, pg_url: str = None, sqlite_path: str = None):
        self.pg_url = pg_url or os.getenv(
            'DATABASE_URL', 
            'postgresql://postgres:postgres@localhost:5432/forex'
        )
        self.sqlite_path = sqlite_path or 'data/tvscripts.db'
        self.pg_conn = None
        self.sqlite_conn = None
    
    def connect_postgresql(self):
        """Connect to PostgreSQL"""
        try:
            self.pg_conn = psycopg2.connect(self.pg_url)
            self.pg_conn.autocommit = False
            logger.info("✅ Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False
    
    def connect_sqlite(self):
        """Connect to SQLite"""
        try:
            if not Path(self.sqlite_path).exists():
                logger.warning(f"⚠️ SQLite database not found at {self.sqlite_path}")
                return False
            
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            logger.info(f"✅ Connected to SQLite: {self.sqlite_path}")
            return True
        except Exception as e:
            logger.error(f"❌ SQLite connection failed: {e}")
            return False
    
    def setup_postgresql_schema(self):
        """Set up PostgreSQL schema"""
        try:
            sql_file = Path(__file__).parent / 'sql' / 'init_tradingview_tables.sql'
            
            if not sql_file.exists():
                logger.error(f"❌ SQL file not found: {sql_file}")
                return False
            
            with open(sql_file, 'r') as f:
                sql_content = f.read()
            
            with self.pg_conn.cursor() as cursor:
                cursor.execute(sql_content)
                self.pg_conn.commit()
            
            logger.info("✅ PostgreSQL schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema setup failed: {e}")
            self.pg_conn.rollback()
            return False
    
    def migrate_scripts_data(self):
        """Migrate scripts from SQLite to PostgreSQL"""
        try:
            # Get data from SQLite
            sqlite_cursor = self.sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT slug, title, author, description, code, open_source,
                       likes, views, strategy_type, indicators, signals, 
                       timeframes, source_url
                FROM scripts
                ORDER BY likes DESC
            """)
            
            sqlite_data = sqlite_cursor.fetchall()
            logger.info(f"📊 Found {len(sqlite_data)} scripts in SQLite")
            
            if not sqlite_data:
                logger.warning("⚠️ No scripts found in SQLite")
                return True
            
            # Prepare data for PostgreSQL
            pg_data = []
            for row in sqlite_data:
                try:
                    # Parse JSON-like strings back to arrays
                    indicators = self._parse_array_string(row[9]) if row[9] else []
                    signals = self._parse_array_string(row[10]) if row[10] else []
                    timeframes = self._parse_array_string(row[11]) if row[11] else []
                    
                    # Determine script type
                    script_type = 'indicator' if row[8] == 'indicator' else 'strategy'
                    
                    pg_data.append((
                        str(uuid.uuid4()),  # id
                        row[0],  # slug
                        row[1],  # title
                        row[2],  # author
                        row[3],  # description
                        row[4],  # code
                        bool(row[5]),  # open_source
                        int(row[6]) if row[6] else 0,  # likes
                        int(row[7]) if row[7] else 0,  # views
                        script_type,  # script_type
                        row[8],  # strategy_type
                        indicators,  # indicators array
                        signals,  # signals array
                        timeframes,  # timeframes array
                        Json({}),  # parameters (empty for now)
                        Json({'migrated_from': 'sqlite'}),  # metadata
                        row[12] if row[12] else f"https://www.tradingview.com/script/{row[0]}/"  # source_url
                    ))
                    
                except Exception as e:
                    logger.warning(f"⚠️ Skipping malformed row {row[0]}: {e}")
                    continue
            
            # Insert into PostgreSQL
            with self.pg_conn.cursor() as cursor:
                insert_sql = """
                    INSERT INTO tradingview.scripts (
                        id, slug, title, author, description, code, open_source,
                        likes, views, script_type, strategy_type, indicators, 
                        signals, timeframes, parameters, metadata, source_url
                    ) VALUES %s
                    ON CONFLICT (slug) DO UPDATE SET
                        title = EXCLUDED.title,
                        author = EXCLUDED.author,
                        description = EXCLUDED.description,
                        code = EXCLUDED.code,
                        likes = EXCLUDED.likes,
                        views = EXCLUDED.views,
                        updated_at = CURRENT_TIMESTAMP
                """
                
                execute_values(cursor, insert_sql, pg_data, page_size=100)
                self.pg_conn.commit()
            
            logger.info(f"✅ Migrated {len(pg_data)} scripts to PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            self.pg_conn.rollback()
            return False
    
    def _parse_array_string(self, array_str: str) -> List[str]:
        """Parse array-like strings from SQLite"""
        try:
            if not array_str:
                return []
            
            # Remove brackets and quotes, split by comma
            cleaned = array_str.strip("[]'\"")
            if not cleaned:
                return []
            
            items = [item.strip().strip("'\"") for item in cleaned.split(',')]
            return [item for item in items if item]
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse array string '{array_str}': {e}")
            return []
    
    def verify_migration(self):
        """Verify the migration was successful"""
        try:
            with self.pg_conn.cursor() as cursor:
                # Count scripts
                cursor.execute("SELECT COUNT(*) FROM tradingview.scripts")
                script_count = cursor.fetchone()[0]
                
                # Get categories
                cursor.execute("""
                    SELECT strategy_type, script_type, COUNT(*) 
                    FROM tradingview.scripts 
                    GROUP BY strategy_type, script_type
                    ORDER BY COUNT(*) DESC
                """)
                categories = cursor.fetchall()
                
                # Get top scripts
                cursor.execute("""
                    SELECT title, author, likes, views 
                    FROM tradingview.scripts 
                    ORDER BY likes DESC 
                    LIMIT 5
                """)
                top_scripts = cursor.fetchall()
                
                logger.info(f"📊 Migration verification:")
                logger.info(f"   Total scripts: {script_count}")
                logger.info(f"   Categories:")
                for strategy_type, script_type, count in categories:
                    logger.info(f"     {script_type}/{strategy_type}: {count}")
                
                logger.info(f"   Top scripts:")
                for title, author, likes, views in top_scripts:
                    logger.info(f"     {title} by {author} ({likes:,} likes)")
                
                return script_count > 0
                
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def create_sample_data_if_empty(self):
        """Create sample data if no SQLite data exists"""
        try:
            with self.pg_conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM tradingview.scripts")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    logger.info(f"✅ Database already has {count} scripts")
                    return True
            
            logger.info("📋 Creating sample TradingView data...")
            
            # Create sample data directly in PostgreSQL
            sample_scripts = self._get_sample_scripts()
            
            with self.pg_conn.cursor() as cursor:
                for script in sample_scripts:
                    cursor.execute("""
                        INSERT INTO tradingview.scripts (
                            id, slug, title, author, description, code, open_source,
                            likes, views, script_type, strategy_type, indicators,
                            signals, timeframes, source_url, parameters, metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) ON CONFLICT (slug) DO NOTHING
                    """, (
                        str(uuid.uuid4()),
                        script['slug'],
                        script['title'],
                        script['author'],
                        script['description'],
                        script['code'],
                        script['open_source'],
                        script['likes'],
                        script['views'],
                        script.get('script_type', 'strategy'),
                        script.get('strategy_type', 'trending'),
                        script.get('indicators', []),
                        script.get('signals', []),
                        script.get('timeframes', []),
                        script.get('source_url', f"https://www.tradingview.com/script/{script['slug']}/"),
                        Json(script.get('parameters', {})),
                        Json({'source': 'sample_data'})
                    ))
                
                self.pg_conn.commit()
            
            logger.info(f"✅ Created {len(sample_scripts)} sample scripts")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sample data creation failed: {e}")
            self.pg_conn.rollback()
            return False
    
    def _get_sample_scripts(self):
        """Get sample TradingView scripts"""
        return [
            {
                'slug': 'volume-weighted-average-price-vwap',
                'title': 'Volume Weighted Average Price (VWAP)',
                'author': 'TradingView',
                'description': 'The Volume Weighted Average Price (VWAP) is a trading benchmark used especially in pension funds',
                'code': '// VWAP Sample Code\nvwap_value = ta.vwap(hlc3)\nplot(vwap_value, color=color.blue)',
                'open_source': True,
                'likes': 15420,
                'views': 890000,
                'script_type': 'indicator',
                'strategy_type': 'volume',
                'indicators': ['VWAP'],
                'signals': ['institutional_level'],
                'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d']
            },
            {
                'slug': 'relative-strength-index-rsi',
                'title': 'Relative Strength Index (RSI)',
                'author': 'TradingView',
                'description': 'RSI is a momentum oscillator that measures the speed and magnitude of price changes',
                'code': '// RSI Sample Code\nrsi_value = ta.rsi(close, 14)\nplot(rsi_value, "RSI")',
                'open_source': True,
                'likes': 12800,
                'views': 750000,
                'script_type': 'indicator',
                'strategy_type': 'momentum',
                'indicators': ['RSI'],
                'signals': ['overbought', 'oversold'],
                'timeframes': ['5m', '15m', '30m', '1h', '4h', '1d']
            },
            {
                'slug': 'triple-ema-system',
                'title': 'Triple EMA System',
                'author': 'EMAExpert',
                'description': 'Advanced triple EMA system with trend confirmation',
                'code': '// Triple EMA Sample Code\nema1 = ta.ema(close, 8)\nema2 = ta.ema(close, 21)\nema3 = ta.ema(close, 55)',
                'open_source': True,
                'likes': 420,
                'views': 8500,
                'script_type': 'strategy',
                'strategy_type': 'trending',
                'indicators': ['EMA'],
                'signals': ['crossover', 'trend_confirmation'],
                'timeframes': ['15m', '1h', '4h']
            }
        ]
    
    def close_connections(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.pg_conn:
            self.pg_conn.close()

def main():
    """Main migration function"""
    logger.info("🚀 Starting TradingView PostgreSQL migration...")
    
    migrator = TradingViewPostgreSQLMigrator()
    
    try:
        # Connect to databases
        if not migrator.connect_postgresql():
            return False
        
        # Setup PostgreSQL schema
        if not migrator.setup_postgresql_schema():
            return False
        
        # Try to migrate from SQLite first
        sqlite_available = migrator.connect_sqlite()
        
        if sqlite_available:
            logger.info("📋 Migrating from existing SQLite database...")
            if not migrator.migrate_scripts_data():
                return False
        else:
            logger.info("📋 No SQLite database found, creating sample data...")
            if not migrator.create_sample_data_if_empty():
                return False
        
        # Verify migration
        if not migrator.verify_migration():
            return False
        
        logger.info("🎉 TradingView PostgreSQL migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        return False
        
    finally:
        migrator.close_connections()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)