#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

try:
    from forex_scanner.validation.historical_data_manager import HistoricalDataManager
    from forex_scanner.core.database import DatabaseManager  
    from forex_scanner import config
    from datetime import datetime, timezone
    
    print('✅ Imports successful')
    
    db_manager = DatabaseManager(getattr(config, 'DATABASE_URL'))
    print('✅ Database manager created')
    
    data_manager = HistoricalDataManager(db_manager, 'Europe/Stockholm')
    print('✅ Data manager created')
    
    target_time = datetime(2025, 9, 4, 10, 0, 0, tzinfo=timezone.utc)
    print(f'✅ Target time: {target_time}')
    
    print('Calling get_historical_data...')
    
    # Call the basic method first
    data = data_manager.get_historical_data(
        epic='CS.D.EURUSD.MINI.IP',
        target_timestamp=target_time,
        timeframe='1h'
    )
    
    if data is not None:
        print(f'✅ Basic data retrieval successful: {len(data)} rows')
    else:
        print('❌ Basic data retrieval failed')
        
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
