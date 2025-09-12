# diagnostic_check.py
# Check if enhanced implementation is working

def check_implementation():
    """Check if enhanced EMA logic is properly implemented"""
    
    print("🔍 ENHANCED IMPLEMENTATION DIAGNOSTIC")
    print("=" * 50)
    
    # Check 1: Configuration
    print("1️⃣ Configuration Check:")
    try:
        import config
        
        enhanced_logic = getattr(config, 'USE_ENHANCED_EMA_LOGIC', 'NOT SET')
        epsilon = getattr(config, 'EMA_EPSILON', 'NOT SET')
        ema21_req = getattr(config, 'REQUIRE_PRICE_VS_EMA21', 'NOT SET')
        
        print(f"   USE_ENHANCED_EMA_LOGIC: {enhanced_logic}")
        print(f"   EMA_EPSILON: {epsilon}")
        print(f"   REQUIRE_PRICE_VS_EMA21: {ema21_req}")
        
        if enhanced_logic == 'NOT SET':
            print("   ❌ Enhanced config NOT found in config.py")
            print("   💡 You need to add the enhanced settings to config.py")
            return False
        elif enhanced_logic == False:
            print("   ⚠️ Enhanced logic is DISABLED in config")
            print("   💡 Set USE_ENHANCED_EMA_LOGIC = True in config.py")
            return False
        else:
            print("   ✅ Enhanced config found and enabled")
            
    except Exception as e:
        print(f"   ❌ Config check failed: {e}")
        return False
    
    print()
    
    # Check 2: Method Implementation
    print("2️⃣ Method Implementation Check:")
    try:
        from core.database import DatabaseManager
        from core.signal_detector import SignalDetector
        
        db = DatabaseManager(config.DATABASE_URL)
        detector = SignalDetector(db, 'Europe/Stockholm')
        
        # Check if enhanced methods exist
        methods = [
            '_detect_enhanced_ema_signal',
            '_detect_original_simple_signal',
            '_calculate_enhanced_confidence'
        ]
        
        all_methods_exist = True
        for method in methods:
            if hasattr(detector, method):
                print(f"   ✅ {method}: Found")
            else:
                print(f"   ❌ {method}: Missing")
                all_methods_exist = False
        
        if not all_methods_exist:
            print("   💡 You need to add the enhanced methods to SignalDetector class")
            return False
        else:
            print("   ✅ All enhanced methods found")
            
    except Exception as e:
        print(f"   ❌ Method check failed: {e}")
        return False
    
    print()
    
    # Check 3: Debug Output Format
    print("3️⃣ Debug Output Format Check:")
    try:
        debug_info = detector.debug_signal_detection('CS.D.EURUSD.MINI.IP', 'EURUSD')
        
        if 'error' in debug_info:
            print(f"   ❌ Debug failed: {debug_info['error']}")
            return False
        
        # Check for enhanced debug sections
        enhanced_sections = [
            'enhanced_settings',
            'crossover_analysis',
            'ema_relationships',
            'enhanced_bull_conditions'
        ]
        
        found_enhanced = False
        for section in enhanced_sections:
            if section in debug_info:
                print(f"   ✅ {section}: Found")
                found_enhanced = True
            else:
                print(f"   ❌ {section}: Missing")
        
        if not found_enhanced:
            print("   ⚠️ No enhanced debug sections found")
            print("   💡 Debug method may not be using enhanced logic")
        else:
            print("   ✅ Enhanced debug sections found")
        
        # Check strategy used
        strategy = debug_info.get('strategy', 'unknown')
        print(f"   Strategy detected: {strategy}")
        
        if strategy == 'enhanced_simple_ema':
            print("   ✅ Enhanced strategy is being used")
        elif strategy == 'simple_ema':
            print("   ⚠️ Original simple strategy is being used")
            print("   💡 Enhanced logic may not be reaching the signal detection")
        else:
            print(f"   ❌ Unknown strategy: {strategy}")
            
    except Exception as e:
        print(f"   ❌ Debug output check failed: {e}")
        return False
    
    print()
    
    # Check 4: Signal Detection Path
    print("4️⃣ Signal Detection Path Check:")
    try:
        # Test if the enhanced path is being called
        signal = detector.detect_signals_bid_adjusted('CS.D.EURUSD.MINI.IP', 'EURUSD')
        
        if signal:
            strategy = signal.get('strategy', 'unknown')
            print(f"   Signal strategy: {strategy}")
            
            if strategy == 'enhanced_simple_ema':
                print("   ✅ Enhanced signal detection working")
                return True
            elif strategy == 'simple_ema':
                print("   ⚠️ Falling back to original logic")
                print("   💡 Enhanced conditions may be too restrictive")
            else:
                print(f"   ❌ Unexpected strategy: {strategy}")
        else:
            print("   ℹ️ No signal detected (this is normal)")
            print("   💡 Enhanced logic may be working but finding no valid signals")
            
    except Exception as e:
        print(f"   ❌ Signal detection check failed: {e}")
        return False
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if enhanced_logic != True:
        print("   1. Add enhanced settings to config.py")
    if not all_methods_exist:
        print("   2. Add enhanced methods to SignalDetector class")
    if not found_enhanced:
        print("   3. Update debug_signal_detection method")
    if strategy != 'enhanced_simple_ema':
        print("   4. Check why enhanced logic is not being used")
        print("   5. Consider temporarily setting USE_ENHANCED_EMA_LOGIC = False to test")
    
    return True

if __name__ == "__main__":
    check_implementation()