#!/usr/bin/env python3
"""
Simple Database Setup Script for Alert History System
Works with DATABASE_URL configuration (your current setup)
"""

import os
import sys
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_url():
    """Get database URL from environment or config"""
    # Try environment variable first
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        logger.info("✅ Found DATABASE_URL in environment")
        return database_url
    
    # Try importing from config
    try:
        import config
        if hasattr(config, 'DATABASE_URL') and config.DATABASE_URL:
            logger.info("✅ Found DATABASE_URL in config")
            return config.DATABASE_URL
    except ImportError:
        pass
    
    # Prompt user for database URL
    logger.warning("⚠️ DATABASE_URL not found")
    print("\nPlease provide your PostgreSQL DATABASE_URL:")
    print("Format: postgresql://username:password@host:port/database")
    print("Example: postgresql://postgres:password@localhost:5432/forex_scanner")
    
    database_url = input("\nDATABASE_URL: ").strip()
    
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    
    return database_url


def test_connection(database_url):
    """Test database connection"""
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Database connection successful")
        logger.info(f"   PostgreSQL version: {version}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def create_tables(database_url):
    """Create the alert history tables"""
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        logger.info("🔧 Creating alert_history table...")
        
        # Main alert history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id SERIAL PRIMARY KEY,
                alert_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                epic VARCHAR(50) NOT NULL,
                pair VARCHAR(10) NOT NULL,
                signal_type VARCHAR(10) NOT NULL,
                strategy VARCHAR(50) NOT NULL,
                confidence_score DECIMAL(5,4) NOT NULL,
                price DECIMAL(10,5) NOT NULL,
                bid_price DECIMAL(10,5),
                ask_price DECIMAL(10,5),
                spread_pips DECIMAL(5,2),
                timeframe VARCHAR(10) NOT NULL,
                
                -- Generic strategy data (JSON)
                strategy_config JSON,
                strategy_indicators JSON,
                strategy_metadata JSON,
                
                -- Common technical indicators
                ema_short DECIMAL(10,5),
                ema_long DECIMAL(10,5), 
                ema_trend DECIMAL(10,5),
                
                macd_line DECIMAL(10,6),
                macd_signal DECIMAL(10,6),
                macd_histogram DECIMAL(10,6),
                
                volume DECIMAL(15,2),
                volume_ratio DECIMAL(8,4),
                volume_confirmation BOOLEAN DEFAULT FALSE,
                
                nearest_support DECIMAL(10,5),
                nearest_resistance DECIMAL(10,5),
                distance_to_support_pips DECIMAL(8,2),
                distance_to_resistance_pips DECIMAL(8,2),
                risk_reward_ratio DECIMAL(8,4),
                
                market_session VARCHAR(20),
                is_market_hours BOOLEAN DEFAULT TRUE,
                market_regime VARCHAR(30),
                
                signal_trigger VARCHAR(50),
                signal_conditions JSON,
                crossover_type VARCHAR(50),
                
                claude_analysis TEXT,
                alert_message TEXT,
                alert_level VARCHAR(20) DEFAULT 'INFO',
                
                status VARCHAR(20) DEFAULT 'ACTIVE',
                processed_at TIMESTAMP,
                notes TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        logger.info("🔧 Creating strategy_summary table...")
        
        # Strategy summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_summary (
                id SERIAL PRIMARY KEY,
                date_tracked DATE NOT NULL,
                epic VARCHAR(50) NOT NULL,
                strategy VARCHAR(50) NOT NULL,
                strategy_config_hash VARCHAR(64),
                signal_type VARCHAR(10) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                total_alerts INTEGER DEFAULT 0,
                avg_confidence DECIMAL(5,4),
                min_confidence DECIMAL(5,4),
                max_confidence DECIMAL(5,4),
                avg_price DECIMAL(10,5),
                avg_spread_pips DECIMAL(5,2),
                signal_triggers JSON,
                market_sessions JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date_tracked, epic, strategy, strategy_config_hash, signal_type, timeframe)
            )
        ''')
        
        logger.info("🔧 Creating strategy_configs table...")
        
        # Strategy configs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_configs (
                id SERIAL PRIMARY KEY,
                strategy VARCHAR(50) NOT NULL,
                config_hash VARCHAR(64) NOT NULL,
                config_name VARCHAR(50),
                config_data JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy, config_hash)
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ Tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return False


def create_indexes(database_url):
    """Create performance indexes"""
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        logger.info("🔧 Creating indexes...")
        
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_alert_history_timestamp ON alert_history(alert_timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_alert_history_epic ON alert_history(epic)',
            'CREATE INDEX IF NOT EXISTS idx_alert_history_strategy ON alert_history(strategy)',
            'CREATE INDEX IF NOT EXISTS idx_alert_history_signal_type ON alert_history(signal_type)',
            'CREATE INDEX IF NOT EXISTS idx_alert_history_confidence ON alert_history(confidence_score)',
            'CREATE INDEX IF NOT EXISTS idx_alert_history_status ON alert_history(status)',
            'CREATE INDEX IF NOT EXISTS idx_alert_history_trigger ON alert_history(signal_trigger)',
            'CREATE INDEX IF NOT EXISTS idx_strategy_summary_date ON strategy_summary(date_tracked)',
            'CREATE INDEX IF NOT EXISTS idx_strategy_configs_hash ON strategy_configs(config_hash)',
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ Indexes created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating indexes: {e}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return False


def insert_test_data(database_url):
    """Insert test data to verify everything works"""
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        logger.info("🧪 Inserting test data...")
        
        test_alert = {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'pair': 'EURUSD',
            'signal_type': 'BULL',
            'strategy': 'enhanced_simple_ema',
            'confidence_score': 0.85,
            'price': 1.1234,
            'timeframe': '15m',
            'ema_short': 1.1235,
            'ema_long': 1.1230,
            'ema_trend': 1.1220,
            'signal_trigger': 'price_above_short',
            'crossover_type': 'ema_crossover',
            'strategy_config': '{"ema_config": "aggressive", "ema_short_period": 5}',
            'market_session': 'london',
            'alert_level': 'HIGH',
            'alert_message': 'Test EMA signal'
        }
        
        cursor.execute('''
            INSERT INTO alert_history (
                epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                ema_short, ema_long, ema_trend, signal_trigger, crossover_type,
                strategy_config, market_session, alert_level, alert_message
            ) VALUES (
                %(epic)s, %(pair)s, %(signal_type)s, %(strategy)s, %(confidence_score)s, %(price)s, %(timeframe)s,
                %(ema_short)s, %(ema_long)s, %(ema_trend)s, %(signal_trigger)s, %(crossover_type)s,
                %(strategy_config)s, %(market_session)s, %(alert_level)s, %(alert_message)s
            )
        ''', test_alert)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ Test data inserted successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inserting test data: {e}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return False


def verify_setup(database_url):
    """Verify the database setup"""
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        logger.info("🔍 Verifying setup...")
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('alert_history', 'strategy_summary', 'strategy_configs')
        """)
        
        tables = [row['table_name'] for row in cursor.fetchall()]
        
        if len(tables) == 3:
            logger.info("✅ All tables created successfully")
        else:
            logger.warning(f"⚠️ Only found {len(tables)} tables: {tables}")
        
        # Check test data
        cursor.execute("SELECT COUNT(*) as count FROM alert_history")
        count = cursor.fetchone()['count']
        
        if count > 0:
            logger.info(f"✅ Found {count} alert(s) in database")
        else:
            logger.info("📭 No alerts in database yet")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        if 'conn' in locals():
            cursor.close()
            conn.close()
        return False


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Alert History Database (Simple Version)')
    parser.add_argument('--test-data', action='store_true', help='Insert test data')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing setup')
    
    args = parser.parse_args()
    
    try:
        # Get database URL
        logger.info("🔌 Getting database configuration...")
        database_url = get_database_url()
        
        # Test connection
        if not test_connection(database_url):
            logger.error("❌ Cannot connect to database. Please check your DATABASE_URL")
            return 1
        
        if args.verify_only:
            logger.info("🔍 Verification mode...")
            if verify_setup(database_url):
                logger.info("🎉 Database setup is correct!")
                return 0
            else:
                logger.error("❌ Database setup has issues")
                return 1
        
        # Create tables
        logger.info("🚀 Setting up database...")
        
        if not create_tables(database_url):
            logger.error("❌ Failed to create tables")
            return 1
        
        if not create_indexes(database_url):
            logger.error("❌ Failed to create indexes")
            return 1
        
        # Insert test data if requested
        if args.test_data:
            if not insert_test_data(database_url):
                logger.error("❌ Failed to insert test data")
                return 1
        
        # Verify setup
        if not verify_setup(database_url):
            logger.error("❌ Setup verification failed")
            return 1
        
        # Success!
        logger.info("\n" + "="*60)
        logger.info("🎉 DATABASE SETUP COMPLETE!")
        logger.info("="*60)
        logger.info("Your alert history database is ready!")
        
        logger.info("\n📊 Next steps:")
        logger.info("1. Update your scanner.py to use AlertHistoryManager")
        logger.info("2. Start your forex scanner")
        logger.info("3. Signals will be automatically saved to the database")
        
        if args.test_data:
            logger.info("\n🧪 Test query:")
            logger.info("SELECT * FROM alert_history ORDER BY alert_timestamp DESC LIMIT 5;")
        
        logger.info("\n📈 Monitor your alerts:")
        logger.info("• Use the analytics functions to analyze performance")
        logger.info("• Export data for external analysis")
        logger.info("• Track which strategies work best")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)