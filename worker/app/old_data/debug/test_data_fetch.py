#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

from forex_scanner.validation.historical_data_manager import HistoricalDataManager
from forex_scanner.core.database import DatabaseManager  
from forex_scanner import config
from datetime import datetime, timezone

try:
    print('🔍 Testing data fetch directly...')
    
    db_manager = DatabaseManager(getattr(config, 'DATABASE_URL'))
    data_manager = HistoricalDataManager(db_manager, 'Europe/Stockholm')
    
    target_time = datetime(2025, 9, 4, 10, 0, 0, tzinfo=timezone.utc)
    
    print(f'Fetching data for CS.D.EURUSD.MINI.IP at {target_time}')
    
    # Try the exact same call as the validation system
    data = data_manager.get_enhanced_historical_data(
        epic='CS.D.EURUSD.MINI.IP',
        target_timestamp=target_time,
        timeframe='1h',
        strategy=None,
        indicators_needed=None
    )
    
    if data is not None:
        print(f'✅ SUCCESS: Retrieved {len(data)} candles')
        print(f'Date range: {data.iloc[0].get("start_time", "N/A")} → {data.iloc[-1].get("start_time", "N/A")}')
        print(f'Columns: {list(data.columns)}')
    else:
        print('❌ FAILED: No data returned')
        
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
