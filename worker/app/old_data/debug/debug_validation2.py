#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

try:
    from forex_scanner.validation.historical_data_manager import HistoricalDataManager
    from forex_scanner.core.database import DatabaseManager  
    from forex_scanner import config
    from datetime import datetime, timezone
    
    db_manager = DatabaseManager(getattr(config, 'DATABASE_URL'))
    data_manager = HistoricalDataManager(db_manager, 'Europe/Stockholm')
    target_time = datetime(2025, 9, 4, 10, 0, 0, tzinfo=timezone.utc)
    
    print('Testing get_historical_candles...')
    
    # Test the basic candle retrieval
    data = data_manager.get_historical_candles(
        epic='CS.D.EURUSD.MINI.IP',
        target_timestamp=target_time,
        timeframe='1h',
        lookback_bars=100
    )
    
    if data is not None:
        print(f'✅ SUCCESS: Retrieved {len(data)} candles')
        print(f'Date range: {data.iloc[0].get("start_time", "N/A")} → {data.iloc[-1].get("start_time", "N/A")}')
        print(f'Sample prices: open={data.iloc[-1].get("open", "N/A")}, close={data.iloc[-1].get("close", "N/A")}')
    else:
        print('❌ No data returned')
        
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
