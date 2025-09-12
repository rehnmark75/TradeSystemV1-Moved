#!/usr/bin/env python3
import os
from sqlalchemy import create_engine, text
import pandas as pd

print('🔍 Checking available historical data...')

try:
    db_url = os.getenv('DATABASE_URL')
    engine = create_engine(db_url)
    
    # Check what epics are available
    print('\n📊 Available epics:')
    with engine.connect() as conn:
        result = conn.execute(text('''
            SELECT epic, COUNT(*) as candle_count, 
                   MIN(start_time) as earliest, 
                   MAX(start_time) as latest,
                   ARRAY_AGG(DISTINCT timeframe ORDER BY timeframe) as timeframes
            FROM ig_candles 
            GROUP BY epic 
            ORDER BY candle_count DESC 
            LIMIT 10
        '''))
        
        for row in result:
            print(f'  {row[0]}: {row[1]} candles ({row[2]} → {row[3]}) - TF: {row[4]}')
    
    # Check recent data for common pairs
    common_pairs = ['CS.D.EURUSD.MINI.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP']
    
    print('\n📈 Recent data for common pairs:')
    for epic in common_pairs:
        with engine.connect() as conn:
            result = conn.execute(text('''
                SELECT COUNT(*) as count, MAX(start_time) as latest
                FROM ig_candles 
                WHERE epic = :epic AND timeframe = 15
            '''), {'epic': epic})
            
            row = result.fetchone()
            if row[0] > 0:
                print(f'  {epic}: {row[0]} candles (latest: {row[1]})')
            else:
                print(f'  {epic}: No data available')
    
    # Check if there's any recent data at all
    print('\n🕒 Most recent candles across all epics:')
    with engine.connect() as conn:
        df = pd.read_sql(text('''
            SELECT epic, start_time, timeframe, close
            FROM ig_candles 
            ORDER BY start_time DESC 
            LIMIT 5
        '''), conn)
        
        if len(df) > 0:
            for _, row in df.iterrows():
                print(f'  {row["epic"]} - {row["start_time"]} (TF: {row["timeframe"]}) - Close: {row["close"]}')
        else:
            print('  No candle data found in database')
    
    # Check alert_history table for comparison
    print('\n🚨 Alert history data:')
    try:
        with engine.connect() as conn:
            result = conn.execute(text('''
                SELECT COUNT(*) as alert_count, 
                       MIN(timestamp) as earliest_alert,
                       MAX(timestamp) as latest_alert
                FROM alert_history
            '''))
            
            row = result.fetchone()
            if row[0] > 0:
                print(f'  {row[0]} alerts found ({row[1]} → {row[2]})')
                
                # Get recent alerts
                recent_alerts = conn.execute(text('''
                    SELECT epic, timestamp, signal_type 
                    FROM alert_history 
                    ORDER BY timestamp DESC 
                    LIMIT 3
                ''')).fetchall()
                
                for alert in recent_alerts:
                    print(f'    {alert[0]} - {alert[1]} - {alert[2]}')
            else:
                print('  No alert history found')
    except Exception as e:
        print(f'  Alert history table not accessible: {e}')
        
except Exception as e:
    print(f'❌ Error checking data: {e}')
    import traceback
    traceback.print_exc()
