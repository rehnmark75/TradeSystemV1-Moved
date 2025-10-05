
# Test the fixed AlertHistoryManager
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

from core.database import DatabaseManager
from alerts.alert_history import AlertHistoryManager
import config
from datetime import datetime

# Test signal with deduplication metadata
test_signal = {
    'epic': 'CS.D.EURUSD.CEEM.IP',
    'signal_type': 'BUY',
    'strategy': 'MINIMAL_TEST',
    'price': 1.0950,
    'confidence_score': 0.85,
    'timeframe': '1H',
    'signal_hash': 'test_hash_12345',
    'cooldown_key': 'CS.D.EURUSD.CEEM.IP:BUY:MINIMAL_TEST',
    'data_source': 'minimal_test',
    'market_timestamp': datetime.now().isoformat()
}

print("Testing AlertHistoryManager with metadata...")
print(f"Signal hash: {test_signal['signal_hash']}")
print(f"Cooldown key: {test_signal['cooldown_key']}")

# Initialize and test
db_manager = DatabaseManager(config.DATABASE_URL)
alert_manager = AlertHistoryManager(db_manager)

alert_id = alert_manager.save_alert(test_signal, "Minimal test alert", 'INFO')

if alert_id:
    print(f"✅ Alert saved with ID: {alert_id}")
    
    # Verify metadata was saved
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT signal_hash, data_source, cooldown_key, market_timestamp
        FROM alert_history WHERE id = %s
    """, (alert_id,))
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result:
        signal_hash, data_source, cooldown_key, market_timestamp = result
        print(f"Saved metadata:")
        print(f"  signal_hash: {signal_hash} {'✅' if signal_hash else '❌'}")
        print(f"  data_source: {data_source} {'✅' if data_source else '❌'}")  
        print(f"  cooldown_key: {cooldown_key} {'✅' if cooldown_key else '❌'}")
        print(f"  market_timestamp: {market_timestamp} {'✅' if market_timestamp else '❌'}")
        
        if signal_hash and cooldown_key and data_source:
            print("🎉 SUCCESS! Deduplication metadata is working!")
        else:
            print("❌ Some metadata is still missing")
    else:
        print("❌ Could not retrieve saved alert")
else:
    print("❌ Alert save failed")
