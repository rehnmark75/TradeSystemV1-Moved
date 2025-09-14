#!/usr/bin/env python3
"""
Simple TradingView Data Migration

Extracts data from SQLite and generates SQL INSERT statements
for PostgreSQL. Can be run without PostgreSQL dependencies.
"""

import sqlite3
import json
import uuid
from pathlib import Path

def extract_sqlite_data(db_path: str = "data/tvscripts.db"):
    """Extract data from SQLite and format for PostgreSQL"""
    
    if not Path(db_path).exists():
        print(f"❌ SQLite database not found at {db_path}")
        return []
    
    print(f"📊 Reading data from {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all data
    cursor.execute("""
        SELECT slug, title, author, description, code, open_source,
               likes, views, strategy_type, indicators, signals, 
               timeframes, source_url
        FROM scripts
        ORDER BY likes DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"✅ Found {len(rows)} scripts")
    
    # Convert to PostgreSQL format
    pg_data = []
    for row in rows:
        try:
            # Parse string arrays back to Python lists
            indicators = parse_array_string(row[9]) if row[9] else []
            signals = parse_array_string(row[10]) if row[10] else []
            timeframes = parse_array_string(row[11]) if row[11] else []
            
            # Determine script type
            script_type = 'indicator' if row[8] == 'indicator' else 'strategy'
            
            pg_data.append({
                'id': str(uuid.uuid4()),
                'slug': row[0],
                'title': row[1],
                'author': row[2],
                'description': row[3] or '',
                'code': row[4] or '',
                'open_source': bool(row[5]),
                'likes': int(row[6]) if row[6] else 0,
                'views': int(row[7]) if row[7] else 0,
                'script_type': script_type,
                'strategy_type': row[8] or 'unknown',
                'indicators': indicators,
                'signals': signals,
                'timeframes': timeframes,
                'source_url': row[12] or f"https://www.tradingview.com/script/{row[0]}/"
            })
            
        except Exception as e:
            print(f"⚠️ Skipping malformed row {row[0]}: {e}")
            continue
    
    return pg_data

def parse_array_string(array_str: str) -> list:
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
        
    except:
        return []

def generate_sql_inserts(pg_data: list) -> str:
    """Generate PostgreSQL INSERT statements"""
    
    if not pg_data:
        return ""
    
    sql_statements = []
    
    # Schema creation
    sql_statements.append("-- TradingView Scripts Migration")
    sql_statements.append("-- Run this against your PostgreSQL database")
    sql_statements.append("")
    sql_statements.append("-- Create extension and schema")
    sql_statements.append('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    sql_statements.append("CREATE SCHEMA IF NOT EXISTS tradingview;")
    sql_statements.append("")
    
    # Table creation (simplified version)
    table_sql = """
-- Create scripts table
CREATE TABLE IF NOT EXISTS tradingview.scripts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(255) NOT NULL,
    description TEXT,
    code TEXT,
    open_source BOOLEAN DEFAULT TRUE,
    likes INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source_url TEXT,
    script_type VARCHAR(50) DEFAULT 'strategy',
    strategy_type VARCHAR(50),
    indicators TEXT[],
    signals TEXT[],
    timeframes TEXT[],
    parameters JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_scripts_slug ON tradingview.scripts(slug);
CREATE INDEX IF NOT EXISTS idx_scripts_strategy_type ON tradingview.scripts(strategy_type);
CREATE INDEX IF NOT EXISTS idx_scripts_likes ON tradingview.scripts(likes DESC);
CREATE INDEX IF NOT EXISTS idx_scripts_title_fts ON tradingview.scripts USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_scripts_description_fts ON tradingview.scripts USING gin(to_tsvector('english', description));
"""
    sql_statements.append(table_sql)
    sql_statements.append("")
    
    # Data inserts
    sql_statements.append("-- Insert TradingView scripts")
    sql_statements.append("INSERT INTO tradingview.scripts (")
    sql_statements.append("    id, slug, title, author, description, code, open_source,")
    sql_statements.append("    likes, views, script_type, strategy_type, indicators,")
    sql_statements.append("    signals, timeframes, source_url, parameters, metadata")
    sql_statements.append(") VALUES")
    
    # Generate values
    for i, script in enumerate(pg_data):
        # Escape single quotes in strings
        def escape_sql_string(s):
            if s is None:
                return 'NULL'
            return "'" + str(s).replace("'", "''") + "'"
        
        # Format arrays
        def format_array(arr):
            if not arr:
                return "ARRAY[]::TEXT[]"
            escaped_items = [escape_sql_string(item).strip("'") for item in arr]
            return "ARRAY[" + ",".join(f"'{item}'" for item in escaped_items) + "]"
        
        values = f"""    (
        '{script['id']}',
        {escape_sql_string(script['slug'])},
        {escape_sql_string(script['title'])},
        {escape_sql_string(script['author'])},
        {escape_sql_string(script['description'])},
        {escape_sql_string(script['code'])},
        {script['open_source']},
        {script['likes']},
        {script['views']},
        {escape_sql_string(script['script_type'])},
        {escape_sql_string(script['strategy_type'])},
        {format_array(script['indicators'])},
        {format_array(script['signals'])},
        {format_array(script['timeframes'])},
        {escape_sql_string(script['source_url'])},
        '{{}}',
        '{{"migrated_from": "sqlite"}}'
    )"""
        
        if i < len(pg_data) - 1:
            values += ","
        else:
            values += ";"
        
        sql_statements.append(values)
    
    sql_statements.append("")
    sql_statements.append("-- Verify migration")
    sql_statements.append("SELECT COUNT(*) as total_scripts FROM tradingview.scripts;")
    sql_statements.append("SELECT script_type, COUNT(*) FROM tradingview.scripts GROUP BY script_type;")
    
    return "\n".join(sql_statements)

def main():
    """Main migration function"""
    print("🚀 TradingView SQLite to PostgreSQL Migration")
    print("=" * 50)
    
    # Extract data from SQLite
    pg_data = extract_sqlite_data()
    
    if not pg_data:
        print("❌ No data to migrate")
        return False
    
    # Generate SQL file
    sql_content = generate_sql_inserts(pg_data)
    
    # Write SQL file
    output_file = "tradingview_migration.sql"
    with open(output_file, 'w') as f:
        f.write(sql_content)
    
    print(f"✅ Generated migration SQL: {output_file}")
    print(f"📊 Migrating {len(pg_data)} scripts")
    
    print("\n🔧 To complete the migration:")
    print(f"1. Run this SQL file against your PostgreSQL database:")
    print(f"   psql -h localhost -U postgres -d forex -f {output_file}")
    print("2. Or copy/paste the SQL into pgAdmin")
    print("3. Then restart the TradingView container")
    
    return True

if __name__ == "__main__":
    main()