#!/usr/bin/env python3
"""
Fix the parameter binding issue in AlertHistoryManager
The issue is likely with how psycopg2 parameters are passed
"""

def fix_alert_history_parameter_binding():
    """
    The issue is likely that some field is still being passed as a dict
    or there's a parameter binding problem with psycopg2
    """
    
    import sys
    sys.path.append('/app/forex_scanner')
    
    print("🔍 Testing parameter binding issue...")
    
    try:
        from alerts.alert_history import AlertHistoryManager
        from core.database import DatabaseManager
        import config
        import json
        import psycopg2.extras
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Test with raw psycopg2 connection
        conn = db.get_connection()
        cursor = conn.cursor()
        
        print("✅ Raw psycopg2 connection obtained")
        
        # Test a simple insert to see parameter format
        test_values = (
            'TEST_EPIC',           # epic
            'EURUSD',              # pair  
            'BULL',                # signal_type
            'test_strategy',       # strategy
            0.75,                  # confidence_score
            1.0850,                # price
            None,                  # bid_price
            None,                  # ask_price
            None,                  # spread_pips
            '15m',                 # timeframe
            None,                  # strategy_config (JSON)
            None,                  # strategy_indicators (JSON)
            None,                  # strategy_metadata (JSON)
            None,                  # ema_short
            None,                  # ema_long
            None,                  # ema_trend
            None,                  # macd_line
            None,                  # macd_signal
            None,                  # macd_histogram
            None,                  # volume
            None,                  # volume_ratio
            False,                 # volume_confirmation
            None,                  # nearest_support
            None,                  # nearest_resistance
            None,                  # distance_to_support_pips
            None,                  # distance_to_resistance_pips
            None,                  # risk_reward_ratio
            None,                  # market_session
            True,                  # is_market_hours
            None,                  # market_regime
            None,                  # signal_trigger
            None,                  # signal_conditions (JSON)
            None,                  # crossover_type
            None,                  # claude_analysis
            'Test alert',          # alert_message
            'INFO',                # alert_level
            'NEW',                 # status
            'NOW()'                # alert_timestamp
        )
        
        # Test basic insert
        insert_sql = """
            INSERT INTO alert_history (
                epic, pair, signal_type, strategy, confidence_score, price,
                bid_price, ask_price, spread_pips, timeframe,
                strategy_config, strategy_indicators, strategy_metadata,
                ema_short, ema_long, ema_trend,
                macd_line, macd_signal, macd_histogram,
                volume, volume_ratio, volume_confirmation,
                nearest_support, nearest_resistance, 
                distance_to_support_pips, distance_to_resistance_pips,
                risk_reward_ratio, market_session, is_market_hours,
                market_regime, signal_trigger, signal_conditions,
                crossover_type, claude_analysis, alert_message, alert_level,
                status, alert_timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, NOW()
            ) RETURNING id
        """
        
        print("🧪 Testing basic parameter binding...")
        
        # Prepare safer values (remove the NOW() from values)
        safe_values = test_values[:-1]  # Remove the NOW() string
        
        cursor.execute(insert_sql, safe_values)
        alert_id = cursor.fetchone()[0]
        
        print(f"✅ Basic insert successful! ID: {alert_id}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Now test the AlertHistoryManager with the same simple data
        print("\n🧪 Testing AlertHistoryManager with simple signal...")
        
        alert_mgr = AlertHistoryManager(db)
        
        simple_signal = {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'signal_type': 'BULL',
            'confidence_score': 0.75,
            'strategy': 'test_strategy',
            'price': 1.0850,
            'timeframe': '15m'
        }
        
        try:
            mgr_alert_id = alert_mgr.save_alert(simple_signal, "Manager test")
            print(f"✅ AlertHistoryManager simple test successful! ID: {mgr_alert_id}")
            
            # Now test with JSON data
            print("\n🧪 Testing AlertHistoryManager with JSON data...")
            
            json_signal = {
                'epic': 'CS.D.EURUSD.MINI.IP',
                'signal_type': 'BULL',
                'confidence_score': 0.75,
                'strategy': 'test_strategy',
                'price': 1.0850,
                'timeframe': '15m',
                'strategy_config': {'test': 'value'},  # This should cause the error if not handled
                'strategy_indicators': {'ema_9': 1.0850},
                'strategy_metadata': {'source': 'test'}
            }
            
            mgr_json_id = alert_mgr.save_alert(json_signal, "Manager JSON test")
            print(f"✅ AlertHistoryManager JSON test successful! ID: {mgr_json_id}")
            
        except Exception as e:
            print(f"❌ AlertHistoryManager failed: {e}")
            print(f"   Error type: {type(e)}")
            
            # Let's see the exact line that's failing
            import traceback
            tb = traceback.format_exc()
            print("📋 Full traceback:")
            print(tb)
            
            # Check if it's a specific field
            print("\n🔍 Investigating which field causes the issue...")
            investigate_problematic_field(alert_mgr, simple_signal)
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

def investigate_problematic_field(alert_mgr, base_signal):
    """Test each field individually to find which one causes the dict error"""
    
    problematic_fields = [
        ('strategy_config', {'fast': 12, 'slow': 26}),
        ('strategy_indicators', {'ema_9': 1.0850, 'ema_21': 1.0840}),
        ('strategy_metadata', {'source': 'test', 'confidence': 'high'}),
        ('signal_conditions', {'crossover': True, 'volume_ok': True}),
        ('timestamp', {'hour': 12, 'minute': 30}),  # Sometimes timestamp is passed as dict
    ]
    
    for field_name, field_value in problematic_fields:
        print(f"\n🧪 Testing field: {field_name}")
        
        test_signal = base_signal.copy()
        test_signal[field_name] = field_value
        
        try:
            alert_id = alert_mgr.save_alert(test_signal, f"Test {field_name}")
            print(f"   ✅ {field_name} works fine")
        except Exception as e:
            print(f"   ❌ {field_name} causes error: {e}")
            if "can't adapt type 'dict'" in str(e):
                print(f"   🎯 FOUND THE CULPRIT: {field_name}")
                return field_name
    
    return None

if __name__ == "__main__":
    print("🔧 FIXING PARAMETER BINDING ISSUE")
    print("=" * 50)
    
    fix_alert_history_parameter_binding()