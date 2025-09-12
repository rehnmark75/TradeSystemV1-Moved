# test_enhanced_alert_saving_fixed.py
import sys
sys.path.append('/app/forex_scanner')

from alerts.alert_history import AlertHistoryManager
from core.database import DatabaseManager
import config
from datetime import datetime

def test_enhanced_alert_saving():
    """Test the enhanced alert saving with all required fields"""
    
    # Create test signal with ALL possible fields that the alert_history.py might expect
    rich_signal = {
        # Basic data (working)
        'epic': 'CS.D.EURUSD.MINI.IP',
        'pair': 'EURUSD',
        'signal_type': 'BULL',
        'strategy': 'combined_strategy',
        'confidence_score': 0.87,
        'price': 1.0845,
        'bid_price': 1.0844,
        'ask_price': 1.0846,
        'spread_pips': 2.0,
        'timeframe': '15m',
        
        # Technical indicators
        'ema_9': 1.0840,
        'ema_21': 1.0835,
        'ema_200': 1.0820,
        'ema_short': 1.0840,
        'ema_long': 1.0835,
        'ema_trend': 1.0820,
        'macd_line': 0.0023,
        'macd_signal': 0.0019,
        'macd_histogram': 0.0004,
        
        # Volume
        'volume': 1500000,
        'volume_ratio': 1.25,
        'volume_confirmation': True,
        
        # Support/Resistance
        'nearest_support': 1.0820,
        'nearest_resistance': 1.0870,
        'distance_to_support_pips': 25,
        'distance_to_resistance_pips': 25,
        'risk_reward_ratio': 1.0,
        
        # Market conditions
        'market_session': 'London',
        'is_market_hours': True,
        'market_regime': 'trending',
        
        # Signal metadata - ADD THE MISSING FIELDS
        'signal_trigger': 'ema_crossover',
        'crossover_type': 'ema_cross',  # THIS WAS MISSING!
        'signal_conditions': {'ema_alignment': True, 'volume_ok': True},
        'signal_strength': 0.85,
        'signal_quality': 'high',
        'technical_score': 0.89,
        
        # Strategy data
        'strategy_config': {
            'ema_periods': [9, 21, 200],
            'macd_params': [12, 26, 9],
            'mode': 'consensus'
        },
        'strategy_indicators': {
            'ema_trend': 'bullish',
            'macd_trend': 'bullish',
            'volume_trend': 'increasing'
        },
        'strategy_metadata': {
            'version': '2.0',
            'confidence_factors': ['ema_alignment', 'macd_confirmation'],
            'market_conditions': 'favorable'
        },
        
        # Deduplication - ADD THESE TOO
        'signal_hash': 'abc123def456',
        'data_source': 'live_scanner',
        'market_timestamp': datetime.now(),
        'cooldown_key': 'EURUSD_BULL_15m',
        
        # Additional fields that might be expected
        'timestamp': datetime.now(),
        'execution_price': 1.0846,
        'config_name': 'default',
        'config_mode': 'static',
        'ema_config': {'short': 9, 'long': 21, 'trend': 200}
    }
    
    # Test saving
    try:
        db = DatabaseManager(config.DATABASE_URL)
        alert_mgr = AlertHistoryManager(db)
        
        print("🧪 Testing enhanced alert saving with ALL fields...")
        alert_id = alert_mgr.save_alert(
            signal=rich_signal,
            alert_message="Test enhanced signal with all fields",
            alert_level='INFO'
        )
        
        if alert_id:
            print(f"✅ Enhanced alert saved successfully (ID: {alert_id})")
            
            # Verify data was saved correctly
            conn = db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT epic, timeframe, ema_short, macd_line, volume, 
                       strategy_config, signal_strength, data_source,
                       crossover_type, signal_trigger
                FROM alert_history 
                WHERE id = %s
            """, (alert_id,))
            
            result = cursor.fetchone()
            if result:
                print("📊 Verification - Fields saved:")
                fields = ['epic', 'timeframe', 'ema_short', 'macd_line', 'volume', 
                         'strategy_config', 'signal_strength', 'data_source',
                         'crossover_type', 'signal_trigger']
                for i, field in enumerate(fields):
                    value = result[i]
                    status = "✅" if value is not None else "❌"
                    print(f"   {status} {field}: {value}")
            
            cursor.close()
            conn.close()
            
        else:
            print("❌ Enhanced alert save failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_minimal_signal():
    """Test with just the basic required fields to see what's truly needed"""
    print("\n🧪 Testing minimal signal to identify required fields...")
    
    minimal_signal = {
        'epic': 'CS.D.EURUSD.MINI.IP',
        'signal_type': 'BULL',
        'strategy': 'test_strategy',
        'confidence_score': 0.75,
        'price': 1.0850,
        'timeframe': '15m'
    }
    
    try:
        db = DatabaseManager(config.DATABASE_URL)
        alert_mgr = AlertHistoryManager(db)
        
        alert_id = alert_mgr.save_alert(minimal_signal, "Minimal test")
        print(f"✅ Minimal signal saved (ID: {alert_id})")
        
    except Exception as e:
        print(f"❌ Minimal signal failed: {e}")
        print("This tells us which field is required but missing!")
        
        # Add the missing field and try again
        if "'crossover_type'" in str(e):
            print("🔍 Adding missing crossover_type field...")
            minimal_signal['crossover_type'] = 'ema_cross'
            
        if "'signal_trigger'" in str(e):
            print("🔍 Adding missing signal_trigger field...")
            minimal_signal['signal_trigger'] = 'ema_crossover'
            
        if "'data_source'" in str(e):
            print("🔍 Adding missing data_source field...")
            minimal_signal['data_source'] = 'live_scanner'
            
        # Try again with the added field
        try:
            alert_id = alert_mgr.save_alert(minimal_signal, "Minimal test with fix")
            print(f"✅ Fixed minimal signal saved (ID: {alert_id})")
        except Exception as e2:
            print(f"❌ Still failing: {e2}")

if __name__ == "__main__":
    test_minimal_signal()
    test_enhanced_alert_saving()