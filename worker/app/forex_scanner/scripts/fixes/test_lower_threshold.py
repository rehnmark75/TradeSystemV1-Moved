# test_macd_integration.py
import sys
sys.path.append('/app/forex_scanner')

def test_macd_integration():
    print("🔍 TESTING MACD STRATEGY INTEGRATION")
    print("=" * 50)
    
    from core.signal_detector import SignalDetector
    from core.database import DatabaseManager
    import config
    
    try:
        db = DatabaseManager(config.DATABASE_URL)
        detector = SignalDetector(db)
        
        print(f"📊 Configuration:")
        print(f"   MACD_EMA_STRATEGY: {getattr(config, 'MACD_EMA_STRATEGY', 'NOT_SET')}")
        print(f"   Has macd_strategy: {hasattr(detector, 'macd_strategy')}")
        
        # Look for MACD-related methods
        macd_methods = [attr for attr in dir(detector) if 'macd' in attr.lower()]
        print(f"   MACD methods: {macd_methods}")
        
        # Test if MACD is called in detect_signals_bid_adjusted
        epic = 'CS.D.EURUSD.MINI.IP'
        pair = 'EURUSD'
        
        print(f"\n🧪 Testing MACD through different methods:")
        
        # Check your detect_signals_bid_adjusted method
        print(f"1️⃣ Via detect_signals_bid_adjusted...")
        # This should try MACD if EMA fails, based on your config
        
        # Check if there are specific MACD methods
        if hasattr(detector, 'detect_macd_ema_signals'):
            print(f"2️⃣ Testing detect_macd_ema_signals...")
            signal = detector.detect_macd_ema_signals(epic, pair, 1.5, '15m')
            if signal:
                print(f"   ✅ MACD method found signal: {signal}")
            else:
                print(f"   ❌ MACD method found no signal")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_macd_integration()