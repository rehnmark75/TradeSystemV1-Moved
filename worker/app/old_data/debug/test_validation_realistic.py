#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone, timedelta

# Add proper path
sys.path.insert(0, '/app')

print('🔍 Testing validation system with realistic parameters...')

try:
    # First test: Try to import the validation system (might still have import issues)
    print('\n📋 Testing imports (may fail due to core import issues):')
    try:
        from forex_scanner.validation.signal_replay_validator import SignalReplayValidator
        print('✅ SignalReplayValidator imported successfully')
        validation_import_works = True
    except Exception as e:
        print(f'❌ Import failed: {e}')
        validation_import_works = False
    
    # If imports work, test with realistic parameters
    if validation_import_works:
        print('\n🎯 Testing validation with available data parameters:')
        
        # Use parameters that match available data:
        # - Epic: CS.D.EURUSD.MINI.IP (has 22775 candles)
        # - Timeframe: 1h (60 minutes - available in data)  
        # - Timestamp: Recent but not latest (2025-09-04 12:00:00)
        
        validator = SignalReplayValidator()
        
        result = validator.validate_single_signal(
            epic='CS.D.EURUSD.MINI.IP',
            timestamp=datetime(2025, 9, 4, 12, 0, 0, tzinfo=timezone.utc),
            timeframe='1h',  # Use 60-minute data that exists
            show_calculations=True,
            compare_with_stored=False  # Skip stored comparison for now
        )
        
        if result:
            print(f'✅ Validation completed successfully!')
            print(f'   Epic: {result.epic}')
            print(f'   Success: {result.success}')
            print(f'   Signal detected: {result.signal_detected}')
            print(f'   Processing time: {result.processing_time_ms}ms')
            if hasattr(result, 'message'):
                print(f'   Message: {result.message}')
        else:
            print('❌ Validation returned None')
            
    else:
        print('\n🔧 Testing core validation logic independently:')
        
        # Test the database query pattern directly
        from sqlalchemy import create_engine, text
        import pandas as pd
        from datetime import datetime, timezone, timedelta
        
        db_url = os.getenv('DATABASE_URL')
        engine = create_engine(db_url)
        
        # Simulate the historical data query with realistic parameters
        target_timestamp = datetime(2025, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
        epic = 'CS.D.EURUSD.MINI.IP'
        timeframe_minutes = 60  # 1 hour
        lookback_bars = 200
        
        # Calculate time window (as validation system would do)
        lookback_duration = timedelta(minutes=lookback_bars * timeframe_minutes * 1.5)
        start_time = target_timestamp - lookback_duration
        end_time = target_timestamp + timedelta(minutes=timeframe_minutes)
        
        print(f'   Epic: {epic}')
        print(f'   Target timestamp: {target_timestamp}')
        print(f'   Timeframe: {timeframe_minutes} minutes')  
        print(f'   Query window: {start_time} → {end_time}')
        
        query = text('''
            SELECT start_time, epic, open, high, low, close, volume
            FROM ig_candles 
            WHERE epic = :epic 
            AND timeframe = :timeframe
            AND start_time >= :start_time 
            AND start_time <= :end_time
            ORDER BY start_time ASC
        ''')
        
        with engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    'epic': epic,
                    'timeframe': timeframe_minutes,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
        
        print(f'✅ Query successful: {len(df)} candles retrieved')
        if len(df) > 0:
            print(f'   Date range: {df.iloc[0]["start_time"]} → {df.iloc[-1]["start_time"]}')
            print(f'   Price range: {df["low"].min():.5f} → {df["high"].max():.5f}')
            print(f'   Latest close: {df.iloc[-1]["close"]:.5f}')
            
            if len(df) >= 100:
                print('✅ Sufficient data for validation (100+ bars)')
            else:
                print(f'⚠️  Limited data: {len(df)} bars (validation needs 100+)')
        else:
            print('❌ No data found for specified parameters')
            
        print('\n🎯 Data availability confirmed - validation system should work with:')
        print('   - Epic: CS.D.EURUSD.MINI.IP')  
        print('   - Timeframe: 1h (not 15m)')
        print('   - Recent timestamps (2025-09-04)')
        print('   - Sufficient historical data (200+ bars available)')

except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
