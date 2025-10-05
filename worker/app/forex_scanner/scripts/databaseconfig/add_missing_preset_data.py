#!/usr/bin/env python3
"""
Script to populate missing config_presets table data
Run this to add the default presets: Conservative, Aggressive, Scalping
"""

import psycopg2
import json
from datetime import datetime

def get_preset_configurations():
    """Define the standard configuration presets"""
    
    presets = {
        "Conservative": {
            "description": "Conservative trading approach with higher confidence thresholds and lower risk",
            "config_data": {
                # Trading Parameters
                "MIN_CONFIDENCE": 0.80,
                "SPREAD_PIPS": 1.5,
                "DEFAULT_TIMEFRAME": "15m",
                "SCAN_INTERVAL": 120,
                
                # Strategy Settings
                "SIMPLE_EMA_STRATEGY": True,
                "MACD_EMA_STRATEGY": True,
                "COMBINED_STRATEGY_MODE": "consensus",
                "STRATEGY_WEIGHT_EMA": 0.5,
                "STRATEGY_WEIGHT_MACD": 0.5,
                "STRATEGY_WEIGHT_VOLUME": 0.0,
                "STRATEGY_WEIGHT_BEHAVIOR": 0.0,
                
                # Risk Management
                "DEFAULT_STOP_DISTANCE": 30,
                "DEFAULT_RISK_REWARD": 3.0,
                "MAX_CONCURRENT_POSITIONS": 3,
                "RISK_PER_TRADE_PERCENT": 0.5,
                
                # Market Intelligence
                "ENABLE_CLAUDE_ANALYSIS": True,
                "MARKET_ANALYSIS_DEPTH": "deep",
                
                # Pair Management
                "EPIC_LIST": [
                    "CS.D.EURUSD.CEEM.IP",
                    "CS.D.GBPUSD.MINI.IP",
                    "CS.D.USDJPY.MINI.IP"
                ],
                "MAJOR_PAIRS_ONLY": True,
                "EXCLUDED_PAIRS": []
            }
        },
        
        "Aggressive": {
            "description": "Aggressive trading approach with lower confidence thresholds and higher risk tolerance",
            "config_data": {
                # Trading Parameters
                "MIN_CONFIDENCE": 0.60,
                "SPREAD_PIPS": 2.0,
                "DEFAULT_TIMEFRAME": "5m",
                "SCAN_INTERVAL": 30,
                
                # Strategy Settings
                "SIMPLE_EMA_STRATEGY": True,
                "MACD_EMA_STRATEGY": True,
                "COMBINED_STRATEGY_MODE": "confirmation",
                "STRATEGY_WEIGHT_EMA": 0.3,
                "STRATEGY_WEIGHT_MACD": 0.4,
                "STRATEGY_WEIGHT_VOLUME": 0.2,
                "STRATEGY_WEIGHT_BEHAVIOR": 0.1,
                
                # Risk Management
                "DEFAULT_STOP_DISTANCE": 15,
                "DEFAULT_RISK_REWARD": 1.5,
                "MAX_CONCURRENT_POSITIONS": 8,
                "RISK_PER_TRADE_PERCENT": 2.0,
                
                # Market Intelligence
                "ENABLE_CLAUDE_ANALYSIS": True,
                "MARKET_ANALYSIS_DEPTH": "standard",
                
                # Pair Management
                "EPIC_LIST": [
                    "CS.D.EURUSD.CEEM.IP",
                    "CS.D.GBPUSD.MINI.IP",
                    "CS.D.USDJPY.MINI.IP",
                    "CS.D.USDCHF.MINI.IP",
                    "CS.D.AUDUSD.MINI.IP",
                    "CS.D.USDCAD.MINI.IP",
                    "CS.D.NZDUSD.MINI.IP"
                ],
                "MAJOR_PAIRS_ONLY": False,
                "EXCLUDED_PAIRS": []
            }
        },
        
        "Scalping": {
            "description": "High-frequency scalping approach with very short timeframes and quick exits",
            "config_data": {
                # Trading Parameters
                "MIN_CONFIDENCE": 0.65,
                "SPREAD_PIPS": 1.0,
                "DEFAULT_TIMEFRAME": "1m",
                "SCAN_INTERVAL": 15,
                
                # Strategy Settings
                "SIMPLE_EMA_STRATEGY": True,
                "MACD_EMA_STRATEGY": False,  # MACD too slow for scalping
                "COMBINED_STRATEGY_MODE": "hierarchy",
                "STRATEGY_WEIGHT_EMA": 0.6,
                "STRATEGY_WEIGHT_MACD": 0.0,
                "STRATEGY_WEIGHT_VOLUME": 0.3,
                "STRATEGY_WEIGHT_BEHAVIOR": 0.1,
                
                # Risk Management
                "DEFAULT_STOP_DISTANCE": 8,
                "DEFAULT_RISK_REWARD": 1.0,
                "MAX_CONCURRENT_POSITIONS": 10,
                "RISK_PER_TRADE_PERCENT": 0.5,
                
                # Market Intelligence
                "ENABLE_CLAUDE_ANALYSIS": False,  # Too slow for scalping
                "MARKET_ANALYSIS_DEPTH": "basic",
                
                # Pair Management
                "EPIC_LIST": [
                    "CS.D.EURUSD.CEEM.IP",
                    "CS.D.GBPUSD.MINI.IP",
                    "CS.D.USDJPY.MINI.IP"
                ],
                "MAJOR_PAIRS_ONLY": True,  # Focus on most liquid pairs
                "EXCLUDED_PAIRS": []
            }
        },
        
        "Default": {
            "description": "Balanced default configuration suitable for most trading conditions",
            "config_data": {
                # Trading Parameters
                "MIN_CONFIDENCE": 0.70,
                "SPREAD_PIPS": 1.5,
                "DEFAULT_TIMEFRAME": "15m",
                "SCAN_INTERVAL": 60,
                
                # Strategy Settings
                "SIMPLE_EMA_STRATEGY": True,
                "MACD_EMA_STRATEGY": True,
                "COMBINED_STRATEGY_MODE": "consensus",
                "STRATEGY_WEIGHT_EMA": 0.4,
                "STRATEGY_WEIGHT_MACD": 0.3,
                "STRATEGY_WEIGHT_VOLUME": 0.2,
                "STRATEGY_WEIGHT_BEHAVIOR": 0.1,
                
                # Risk Management
                "DEFAULT_STOP_DISTANCE": 20,
                "DEFAULT_RISK_REWARD": 2.0,
                "MAX_CONCURRENT_POSITIONS": 5,
                "RISK_PER_TRADE_PERCENT": 1.0,
                
                # Market Intelligence
                "ENABLE_CLAUDE_ANALYSIS": True,
                "MARKET_ANALYSIS_DEPTH": "standard",
                
                # Pair Management
                "EPIC_LIST": [
                    "CS.D.EURUSD.CEEM.IP",
                    "CS.D.GBPUSD.MINI.IP",
                    "CS.D.USDJPY.MINI.IP",
                    "CS.D.USDCHF.MINI.IP",
                    "CS.D.AUDUSD.MINI.IP"
                ],
                "MAJOR_PAIRS_ONLY": True,
                "EXCLUDED_PAIRS": []
            }
        }
    }
    
    return presets

def check_existing_presets(connection_string):
    """Check what presets already exist in the database"""
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        cursor.execute("SELECT preset_name FROM config_presets")
        existing_presets = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return existing_presets
        
    except Exception as e:
        print(f"❌ Error checking existing presets: {e}")
        return []

def insert_preset(connection_string, preset_name, description, config_data, created_by="system"):
    """Insert a single preset into the database"""
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check if preset already exists
        cursor.execute("SELECT id FROM config_presets WHERE preset_name = %s", (preset_name,))
        if cursor.fetchone():
            print(f"⚠️  Preset '{preset_name}' already exists, skipping...")
            cursor.close()
            conn.close()
            return False
        
        # Insert new preset
        cursor.execute("""
            INSERT INTO config_presets (preset_name, description, config_data, created_by, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            preset_name,
            description,
            json.dumps(config_data),
            created_by,
            datetime.now()
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Inserted preset '{preset_name}'")
        return True
        
    except Exception as e:
        print(f"❌ Error inserting preset '{preset_name}': {e}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return False

def verify_table_structure(connection_string):
    """Verify the config_presets table exists and has the correct structure"""
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'config_presets'
        """)
        
        if not cursor.fetchone():
            print("❌ config_presets table does not exist!")
            cursor.close()
            conn.close()
            return False
        
        # Check table structure
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'config_presets'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        print(f"📋 config_presets table structure ({len(columns)} columns):")
        for column_name, data_type in columns:
            print(f"   {column_name}: {data_type}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error checking table structure: {e}")
        return False

def create_table_if_missing(connection_string):
    """Create the config_presets table if it doesn't exist"""
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_presets (
                id SERIAL PRIMARY KEY,
                preset_name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                config_data JSON NOT NULL,
                created_by VARCHAR(100) DEFAULT 'system',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ config_presets table verified/created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return False

def populate_presets(connection_string):
    """Main function to populate config presets"""
    
    print("🚀 Populating config_presets table...")
    print("=" * 50)
    
    # Verify/create table
    if not create_table_if_missing(connection_string):
        return False
    
    # Verify table structure
    if not verify_table_structure(connection_string):
        return False
    
    # Check existing presets
    existing_presets = check_existing_presets(connection_string)
    print(f"📊 Existing presets: {existing_presets}")
    
    # Get preset configurations
    presets = get_preset_configurations()
    
    # Insert each preset
    inserted_count = 0
    for preset_name, preset_data in presets.items():
        if insert_preset(
            connection_string,
            preset_name,
            preset_data["description"],
            preset_data["config_data"]
        ):
            inserted_count += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Successfully inserted {inserted_count}/{len(presets)} presets")
    
    # Verify final state
    final_presets = check_existing_presets(connection_string)
    print(f"📊 Total presets now: {len(final_presets)}")
    print(f"🎯 Available presets: {', '.join(final_presets)}")
    
    return True

def main():
    """Main entry point"""
    import os
    
    # Get database connection string
    connection_string = os.getenv('CONFIG_DATABASE_URL')
    
    if not connection_string:
        # Try alternative environment variables
        connection_string = os.getenv('DATABASE_URL')
        
    if not connection_string:
        # Default for local development
        connection_string = "postgresql://postgres:postgres@localhost:5433/forex_config"
        print("⚠️  Using default local database connection")
    
    print(f"🔌 Connecting to: {connection_string[:30]}...")
    
    try:
        # Test connection
        conn = psycopg2.connect(connection_string)
        conn.close()
        print("✅ Database connection successful")
        
        # Populate presets
        success = populate_presets(connection_string)
        
        if success:
            print("\n🎉 Config presets population completed successfully!")
            print("\n💡 You can now use these presets in your configuration management system:")
            print("   • Conservative - Safe trading with higher confidence thresholds")
            print("   • Aggressive - High-risk trading with more opportunities")
            print("   • Scalping - High-frequency trading with quick exits")
            print("   • Default - Balanced configuration for general use")
        else:
            print("\n❌ Failed to populate config presets")
            return 1
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Make sure your database is running and accessible:")
        print("   • Check DATABASE_URL or CONFIG_DATABASE_URL environment variable")
        print("   • Verify PostgreSQL is running on the specified port")
        print("   • Ensure the forex_config database exists")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())