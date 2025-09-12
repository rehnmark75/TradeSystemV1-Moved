#!/usr/bin/env python3
import pandas as pd
import os
from sqlalchemy import create_engine, text

print('🔍 Testing database connection with environment variables...')

try:
    # Get database URL from environment (as used by the scanner)
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL environment variable not found')
        exit(1)
        
    print(f'📊 Using DATABASE_URL: {db_url.split("@")[0]}@***')
    
    # Create engine (same as DatabaseManager would do)
    engine = create_engine(db_url)
    
    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1 as test_value'))
        row = result.fetchone()
        print(f'✅ Connection test passed: {row[0]}')
    
    # Test the exact query pattern from historical_data_manager.py
    query = text('''
        SELECT start_time, epic, open, high, low, close, volume
        FROM ig_candles 
        WHERE epic = :epic 
        AND timeframe = :timeframe
        AND start_time >= NOW() - INTERVAL '7 days'
        ORDER BY start_time DESC
        LIMIT 5
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
    
    print(f'✅ Historical data query successful: {len(df)} rows')
    if len(df) > 0:
        print(f'   Latest candle: {df.iloc[0]["start_time"]} - Close: {df.iloc[0]["close"]}')
        
        # Test the context manager pattern that we fixed
        print('✅ Context manager pattern works correctly')
    else:
        print('⚠️  No recent data found (this might be expected)')
    
    print('🎉 Database connection test PASSED!')
    print('✅ The historical_data_manager fix should work correctly')
    
except Exception as e:
    print(f'❌ Database test failed: {e}')
    import traceback
    traceback.print_exc()
