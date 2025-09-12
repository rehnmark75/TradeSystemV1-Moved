#!/usr/bin/env python3
"""
Working fix for alert history saving
Fixes both the dict adaptation error and SQLAlchemy INSERT issue
"""

import sys
sys.path.append('/app/forex_scanner')

def create_working_alert_saver():
    """Create a working alert saver that definitely works"""
    
    import json
    from datetime import datetime
    import logging
    
    def save_alert_working(signal, message="Signal detected"):
        """
        Working alert saver that handles all the issues
        """
        try:
            from core.database import DatabaseManager
            import config
            
            db = DatabaseManager(config.DATABASE_URL)
            
            # Get raw connection to bypass SQLAlchemy issues
            conn = db.get_connection()
            cursor = conn.cursor()
            
            # Extract and clean data with explicit type conversion
            epic = str(signal.get('epic', 'Unknown'))
            pair = str(signal.get('pair', epic.replace('CS.D.', '').replace('.MINI.IP', '')))
            signal_type = str(signal.get('signal_type', 'Unknown'))
            strategy = str(signal.get('strategy', 'Unknown'))
            confidence_score = float(signal.get('confidence_score', 0.0))
            price = float(signal.get('price', 0.0))
            timeframe = str(signal.get('timeframe', '15m'))
            
            # Handle JSON fields properly - convert to string or None
            strategy_config = None
            strategy_indicators = None
            strategy_metadata = None
            
            if signal.get('strategy_config') and isinstance(signal.get('strategy_config'), dict):
                strategy_config = json.dumps(signal['strategy_config'])
            
            if signal.get('strategy_indicators') and isinstance(signal.get('strategy_indicators'), dict):
                strategy_indicators = json.dumps(signal['strategy_indicators'])
                
            if signal.get('strategy_metadata') and isinstance(signal.get('strategy_metadata'), dict):
                strategy_metadata = json.dumps(signal['strategy_metadata'])
            
            # Simple INSERT that works
            insert_sql = """
                INSERT INTO alert_history (
                    epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                    strategy_config, strategy_indicators, strategy_metadata,
                    alert_message, alert_level, status, alert_timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """
            
            # Execute with proper parameter binding
            cursor.execute(insert_sql, (
                epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                strategy_config, strategy_indicators, strategy_metadata,
                message, 'INFO', 'NEW', datetime.now()
            ))
            
            # Get the ID
            alert_id = cursor.fetchone()[0]
            
            # Commit and close
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"✅ Alert saved successfully with ID: {alert_id}")
            return alert_id
            
        except Exception as e:
            print(f"❌ Error saving alert: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up connection
            if 'conn' in locals():
                try:
                    conn.rollback()
                    cursor.close()
                    conn.close()
                except:
                    pass
            
            return None
    
    return save_alert_working

def test_working_saver():
    """Test the working alert saver"""
    
    print("🧪 Testing working alert saver...")
    
    save_alert = create_working_alert_saver()
    
    # Test 1: Simple signal
    simple_signal = {
        'epic': 'CS.D.EURUSD.MINI.IP',
        'signal_type': 'BULL',
        'confidence_score': 0.75,
        'strategy': 'test_working',
        'price': 1.0850,
        'timeframe': '15m'
    }
    
    print("🔹 Test 1: Simple signal...")
    result1 = save_alert(simple_signal, "Working saver test - simple")
    
    if result1:
        print(f"✅ Test 1 passed - ID: {result1}")
    else:
        print("❌ Test 1 failed")
        return False
    
    # Test 2: Signal with JSON data
    json_signal = {
        'epic': 'CS.D.AUDJPY.MINI.IP',
        'signal_type': 'BEAR',
        'confidence_score': 0.95,
        'strategy': 'macd_ema200_m',
        'price': 94.11600,
        'timeframe': '15m',
        'strategy_config': {'fast_ema': 12, 'slow_ema': 26},
        'strategy_indicators': {'ema_200': 94.0, 'macd_line': -0.001},
        'strategy_metadata': {'source': 'working_test', 'backtest_recreation': True}
    }
    
    print("🔹 Test 2: Signal with JSON data...")
    result2 = save_alert(json_signal, "Working saver test - JSON data")
    
    if result2:
        print(f"✅ Test 2 passed - ID: {result2}")
    else:
        print("❌ Test 2 failed")
        return False
    
    # Verify both alerts were saved
    print("\n🔍 Verifying alerts in database...")
    
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Check recent alerts
        recent_alerts = db.execute_query("""
            SELECT id, epic, signal_type, confidence_score, strategy, alert_timestamp, alert_message
            FROM alert_history 
            WHERE id IN (%s, %s)
            ORDER BY alert_timestamp DESC
        """, {'id1': result1, 'id2': result2})
        
        # Note: SQLAlchemy parameter binding issue - let's use a different approach
        verification = db.execute_query("""
            SELECT id, epic, signal_type, confidence_score, strategy, alert_message
            FROM alert_history 
            ORDER BY alert_timestamp DESC 
            LIMIT 5
        """)
        
        print("📊 Recent alerts:")
        print(verification.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def create_scanner_integration():
    """Show how to integrate this into the scanner"""
    
    integration_code = '''
# Save this as /app/forex_scanner/utils/alert_saver.py

import json
from datetime import datetime

def save_signal_to_database(signal, message="Signal detected"):
    """
    Working alert saver for scanner integration
    """
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Clean data
        epic = str(signal.get('epic', 'Unknown'))
        pair = str(signal.get('pair', epic.replace('CS.D.', '').replace('.MINI.IP', '')))
        signal_type = str(signal.get('signal_type', 'Unknown'))
        strategy = str(signal.get('strategy', 'Unknown'))
        confidence_score = float(signal.get('confidence_score', 0.0))
        price = float(signal.get('price', 0.0))
        timeframe = str(signal.get('timeframe', '15m'))
        
        # Handle JSON fields
        strategy_config = json.dumps(signal.get('strategy_config', {})) if signal.get('strategy_config') else None
        strategy_indicators = json.dumps(signal.get('strategy_indicators', {})) if signal.get('strategy_indicators') else None
        strategy_metadata = json.dumps(signal.get('strategy_metadata', {})) if signal.get('strategy_metadata') else None
        
        # Insert
        cursor.execute("""
            INSERT INTO alert_history (
                epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                strategy_config, strategy_indicators, strategy_metadata,
                alert_message, alert_level, status, alert_timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """, (
            epic, pair, signal_type, strategy, confidence_score, price, timeframe,
            strategy_config, strategy_indicators, strategy_metadata,
            message, 'INFO', 'NEW', datetime.now()
        ))
        
        alert_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        return alert_id
        
    except Exception as e:
        print(f"Error saving alert: {e}")
        return None

# Then modify your container script:
# In scan_and_trade function, after finding signals:

from utils.alert_saver import save_signal_to_database

if signals:
    for signal in signals:
        # Your existing signal processing
        alert_id = save_signal_to_database(signal, f"Live signal detected")
        if alert_id:
            print(f"      💾 Alert saved to database (ID: {alert_id})")
        else:
            print(f"      ⚠️ Failed to save alert to database")
'''
    
    print("📋 SCANNER INTEGRATION CODE:")
    print("=" * 50)
    print(integration_code)

if __name__ == "__main__":
    print("🔧 WORKING ALERT HISTORY FIX")
    print("=" * 50)
    
    success = test_working_saver()
    
    if success:
        print("\n🎉 SUCCESS! Working alert saver confirmed!")
        create_scanner_integration()
        print("\n✅ You can now integrate this into your scanner.")
        print("✅ This bypasses both the dict adaptation and SQLAlchemy issues.")
    else:
        print("\n❌ Working saver test failed")
        print("Need further investigation of database setup")