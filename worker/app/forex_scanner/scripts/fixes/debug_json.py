#!/usr/bin/env python3
"""
Debug the specific JSON/dict issue in AlertHistoryManager
"""

import sys
sys.path.append('/app/forex_scanner')

def debug_exact_error():
    """Debug exactly where the 'can't adapt type dict' error occurs"""
    
    print("🔍 Debugging the exact JSON error...")
    
    try:
        from alerts.alert_history import AlertHistoryManager
        from core.database import DatabaseManager
        import config
        import json
        
        db = DatabaseManager(config.DATABASE_URL)
        alert_mgr = AlertHistoryManager(db)
        
        # Create a test signal that matches what your scanner produces
        test_signal = {
            'epic': 'CS.D.AUDJPY.MINI.IP',
            'signal_type': 'BEAR',
            'confidence_score': 0.95,
            'strategy': 'macd_ema200_m',
            'price': 94.11600,
            'timestamp': '2025-06-30 12:30:00',
            'timeframe': '15m'
        }
        
        print("🧪 Testing with basic signal (no JSON fields)...")
        try:
            alert_id = alert_mgr.save_alert(test_signal, "Basic test")
            print(f"✅ Basic signal saved: {alert_id}")
        except Exception as e:
            print(f"❌ Basic signal failed: {e}")
            return
        
        # Now test with JSON fields
        test_signal_with_json = {
            'epic': 'CS.D.AUDJPY.MINI.IP',
            'signal_type': 'BEAR',
            'confidence_score': 0.95,
            'strategy': 'macd_ema200_m',
            'price': 94.11600,
            'timestamp': '2025-06-30 12:30:00',
            'timeframe': '15m',
            'strategy_config': {'fast_ema': 12, 'slow_ema': 26},  # This is a dict
            'strategy_indicators': {'ema_200': 94.0, 'macd': -0.001},  # This is a dict
            'strategy_metadata': {'source': 'test'}  # This is a dict
        }
        
        print("🧪 Testing with JSON fields...")
        try:
            alert_id = alert_mgr.save_alert(test_signal_with_json, "JSON test")
            print(f"✅ JSON signal saved: {alert_id}")
        except Exception as e:
            print(f"❌ JSON signal failed: {e}")
            print(f"   Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            # Let's check what the alert_history.py is actually doing
            print("\n🔍 Debugging the save_alert method...")
            debug_save_alert_method(test_signal_with_json)
        
    except Exception as e:
        print(f"❌ Debug setup failed: {e}")
        import traceback
        traceback.print_exc()

def debug_save_alert_method(signal):
    """Debug what happens inside save_alert method"""
    
    print("🔍 Checking what save_alert method receives...")
    
    # Extract the same data that save_alert method would extract
    strategy_config = signal.get('strategy_config')
    strategy_indicators = signal.get('strategy_indicators')
    strategy_metadata = signal.get('strategy_metadata')
    
    print(f"strategy_config type: {type(strategy_config)} = {strategy_config}")
    print(f"strategy_indicators type: {type(strategy_indicators)} = {strategy_indicators}")
    print(f"strategy_metadata type: {type(strategy_metadata)} = {strategy_metadata}")
    
    # Test the JSON conversion
    import json
    
    try:
        converted_config = json.dumps(strategy_config) if strategy_config else None
        converted_indicators = json.dumps(strategy_indicators) if strategy_indicators else None
        converted_metadata = json.dumps(strategy_metadata) if strategy_metadata else None
        
        print(f"✅ JSON conversions successful:")
        print(f"   config: {converted_config}")
        print(f"   indicators: {converted_indicators}")
        print(f"   metadata: {converted_metadata}")
        
    except Exception as e:
        print(f"❌ JSON conversion failed: {e}")

def check_alert_history_file():
    """Check the actual content of alert_history.py"""
    
    print("\n🔍 Checking alert_history.py file...")
    
    try:
        with open('/app/forex_scanner/alerts/alert_history.py', 'r') as f:
            content = f.read()
        
        # Check for json import
        if 'import json' in content:
            print("✅ json module imported")
        else:
            print("❌ json module NOT imported")
            
        # Check for the JSON conversion lines
        json_lines = [
            "json.dumps(strategy_config)",
            "json.dumps(strategy_indicators)", 
            "json.dumps(strategy_metadata)"
        ]
        
        for line in json_lines:
            if line in content:
                print(f"✅ Found: {line}")
            else:
                print(f"❌ Missing: {line}")
        
        # Check for the specific error-causing pattern
        if "signal.get('strategy_config')" in content and "json.dumps" not in content:
            print("❌ Found problematic pattern: direct dict assignment without JSON conversion")
        
        # Show the save_alert method signature area
        if 'def save_alert(' in content:
            start = content.find('def save_alert(')
            # Find the first 50 lines of the method
            method_part = content[start:start+2000]
            print("\n📄 save_alert method beginning:")
            print("=" * 50)
            print(method_part[:1000] + "..." if len(method_part) > 1000 else method_part)
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def test_database_connection():
    """Test if the database connection itself is the issue"""
    
    print("\n🔍 Testing database connection...")
    
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Test basic query
        result = db.execute_query("SELECT 1 as test")
        print(f"✅ Basic query works: {result.iloc[0]['test']}")
        
        # Test JSON insertion directly
        test_json = '{"test": "value"}'
        result = db.execute_query("SELECT %s::json as json_test", {'test_json': test_json})
        print(f"✅ JSON query works")
        
        # Test getting a raw connection (what AlertHistoryManager uses)
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        print(f"✅ Raw connection works: {result}")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 DEBUGGING JSON/DICT ADAPTATION ERROR")
    print("=" * 60)
    
    debug_exact_error()
    check_alert_history_file()
    test_database_connection()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print("If the issue persists, it might be:")
    print("1. A different field causing the error (not strategy_* fields)")
    print("2. The database driver version")
    print("3. The way the cursor.execute() is called")
    print("4. A different dict field we haven't identified")