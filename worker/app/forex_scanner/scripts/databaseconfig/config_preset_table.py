#!/usr/bin/env python3
"""
Quick script to check what's in your config_presets table
"""

import psycopg2
import json
from datetime import datetime

def check_config_presets_table(connection_string):
    """Check the current state of config_presets table"""
    
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        print("🔍 Checking config_presets table...")
        print("=" * 50)
        
        # Check if table exists
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'config_presets'
        """)
        
        if not cursor.fetchone():
            print("❌ config_presets table does not exist!")
            print("💡 You need to create the table first.")
            return False
        
        print("✅ config_presets table exists")
        
        # Check table structure
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'config_presets'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        print(f"\n📋 Table Structure ({len(columns)} columns):")
        for column_name, data_type, nullable, default in columns:
            nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
            default_str = f", default: {default}" if default else ""
            print(f"   {column_name}: {data_type} ({nullable_str}{default_str})")
        
        # Check current data
        cursor.execute("SELECT COUNT(*) FROM config_presets")
        total_count = cursor.fetchone()[0]
        
        print(f"\n📊 Current Data:")
        print(f"   Total presets: {total_count}")
        
        if total_count > 0:
            # Get all presets with basic info
            cursor.execute("""
                SELECT id, preset_name, description, created_by, created_at,
                       LENGTH(config_data::text) as config_size
                FROM config_presets 
                ORDER BY created_at DESC
            """)
            
            presets = cursor.fetchall()
            
            print(f"\n📝 Existing Presets:")
            for preset_id, name, desc, created_by, created_at, config_size in presets:
                print(f"   {preset_id}. {name}")
                print(f"      Description: {desc[:80]}{'...' if len(desc) > 80 else ''}")
                print(f"      Created by: {created_by} on {created_at}")
                print(f"      Config size: {config_size} characters")
                print()
            
            # Show a sample configuration
            cursor.execute("""
                SELECT preset_name, config_data 
                FROM config_presets 
                LIMIT 1
            """)
            
            sample = cursor.fetchone()
            if sample:
                name, config_data = sample
                print(f"📄 Sample Configuration ({name}):")
                
                # Pretty print the JSON config
                if isinstance(config_data, str):
                    config_dict = json.loads(config_data)
                else:
                    config_dict = config_data
                
                # Show first few settings
                count = 0
                for key, value in config_dict.items():
                    if count < 5:  # Show first 5 settings
                        print(f"   {key}: {value}")
                        count += 1
                    else:
                        break
                
                if len(config_dict) > 5:
                    print(f"   ... and {len(config_dict) - 5} more settings")
        
        else:
            print("   No presets found in table")
            print("\n❌ Missing Data!")
            print("💡 Expected presets: Conservative, Aggressive, Scalping, Default")
            print("🔧 Run the population script to add missing presets")
        
        cursor.close()
        conn.close()
        
        return total_count > 0
        
    except Exception as e:
        print(f"❌ Error checking table: {e}")
        return False

def main():
    """Main function"""
    import os
    
    # Get database connection
    connection_string = os.getenv('CONFIG_DATABASE_URL') or os.getenv('DATABASE_URL')
    
    if not connection_string:
        connection_string = "postgresql://postgres:postgres@localhost:5433/forex_config"
        print("⚠️  Using default connection string")
    
    print(f"🔌 Connecting to: {connection_string[:30]}...")
    
    try:
        # Test basic connection
        conn = psycopg2.connect(connection_string)
        conn.close()
        print("✅ Database connection successful\n")
        
        # Check the table
        has_data = check_config_presets_table(connection_string)
        
        print("\n" + "=" * 50)
        
        if has_data:
            print("✅ config_presets table has data")
        else:
            print("❌ config_presets table is missing data")
            print("\n🔧 To fix this, run:")
            print("   python populate_config_presets.py")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Check your database configuration:")
        print("   • Is PostgreSQL running?")
        print("   • Does the forex_config database exist?")
        print("   • Are the connection credentials correct?")

if __name__ == "__main__":
    main()