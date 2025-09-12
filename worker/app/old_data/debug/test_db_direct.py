#!/usr/bin/env python3
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

print('🔍 Testing direct database connection...')

try:
    # Use the database URL directly (from docker-compose environment)
    import os
    
    # Try to build database URL from environment
    db_host = os.getenv('POSTGRES_HOST', 'postgres')  
    db_name = os.getenv('POSTGRES_DB', 'forex')
    db_user = os.getenv('POSTGRES_USER', 'postgres')
    db_pass = os.getenv('POSTGRES_PASSWORD', 'password')
    
    db_url = f'postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}'
    print(f'📊 Connecting to: postgresql://{db_user}:***@{db_host}:5432/{db_name}')
    
    # Create engine
    engine = create_engine(db_url)
    
    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1 as test_value'))
        row = result.fetchone()
        print(f'✅ Connection test passed: {row[0]}')
    
    # Test ig_candles table query (similar to validation system)
    query = text('''
        SELECT start_time, epic, open, high, low, close, volume
        FROM ig_candles 
        WHERE epic = :epic 
        AND timeframe = :timeframe
        ORDER BY start_time DESC
        LIMIT 10
    ''')
    
    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                'epic': 'CS.D.EURUSD.MINI.IP',
                'timeframe': 15
            }
        )
    
    print(f'✅ Candle query successful: {len(df)} rows')
    if len(df) > 0:
        print(f'   Latest candle: {df.iloc[0]["start_time"]} - {df.iloc[0]["epic"]} - Close: {df.iloc[0]["close"]}')
    
    print('🎉 Direct database connection test PASSED!')
    print('✅ The database connection fix should work for validation system')
    
except Exception as e:
    print(f'❌ Database connection test failed: {e}')
    import traceback
    traceback.print_exc()
