#!/usr/bin/env python3
"""
Diagnose exactly why no strategies are executing
"""

import sys
sys.path.append('/app/forex_scanner')

def diagnose_strategy_execution():
    """Test which strategies are configured and which are failing"""
    print("🔍 STRATEGY EXECUTION DIAGNOSIS")
    print("=" * 60)
    
    try:
        import config
        
        # Check configuration
        print("📋 CONFIGURATION CHECK:")
        print("-" * 30)
        
        simple_ema = getattr(config, 'SIMPLE_EMA_STRATEGY', 'NOT_SET')
        macd_ema = getattr(config, 'MACD_EMA_STRATEGY', 'NOT_SET')
        combined_mode = getattr(config, 'COMBINED_STRATEGY_MODE', 'NOT_SET')
        
        print(f"   SIMPLE_EMA_STRATEGY: {simple_ema}")
        print(f"   MACD_EMA_STRATEGY: {macd_ema}")
        print(f"   COMBINED_STRATEGY_MODE: {combined_mode}")
        
        # Check if combined mode should be triggered
        combined_enabled = (
            simple_ema is True and 
            macd_ema is True and
            hasattr(config, 'COMBINED_STRATEGY_MODE')
        )
        
        print(f"\n🔄 STRATEGY SELECTION LOGIC:")
        print(f"   Should use combined mode: {combined_enabled}")
        print(f"   Criteria:")
        print(f"     - SIMPLE_EMA_STRATEGY = True: {'✅' if simple_ema is True else '❌'}")
        print(f"     - MACD_EMA_STRATEGY = True: {'✅' if macd_ema is True else '❌'}")
        print(f"     - Has COMBINED_STRATEGY_MODE: {'✅' if hasattr(config, 'COMBINED_STRATEGY_MODE') else '❌'}")
        
        return combined_enabled
        
    except Exception as e:
        print(f"❌ Config check failed: {e}")
        return False

def test_individual_strategies():
    """Test individual strategies to see which ones work"""
    print("\n🧪 INDIVIDUAL STRATEGY TESTS")
    print("-" * 40)
    
    try:
        from core.signal_detector import SignalDetector
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        detector = SignalDetector(db, 'Europe/Stockholm')
        
        epic = 'CS.D.EURUSD.CEEM.IP'
        pair = 'EURUSD'
        spread_pips = 1.5
        timeframe = '15m'
        
        print(f"Testing with: {epic}")
        
        # Test 1: EMA Strategy Only
        print("\n🔵 TEST 1: EMA Strategy")
        try:
            # Temporarily disable MACD to force EMA only
            original_macd = getattr(config, 'MACD_EMA_STRATEGY', True)
            config.MACD_EMA_STRATEGY = False
            
            ema_signal = detector.detect_signals_all_strategies(epic, pair, spread_pips, timeframe)
            
            # Restore original setting
            config.MACD_EMA_STRATEGY = original_macd
            
            print(f"   📊 EMA Strategy result: {len(ema_signal)} signals")
            if ema_signal:
                for signal in ema_signal:
                    print(f"      Strategy: {signal.get('strategy', 'unknown')}")
                    print(f"      Type: {signal.get('signal_type', 'unknown')}")
                    print(f"      Confidence: {signal.get('confidence_score', 0):.1%}")
            
        except Exception as e:
            print(f"   ❌ EMA Strategy failed: {e}")
        
        # Test 2: MACD Strategy Only
        print("\n🟡 TEST 2: MACD Strategy")
        try:
            # Temporarily disable EMA to force MACD only
            original_ema = getattr(config, 'SIMPLE_EMA_STRATEGY', True)
            config.SIMPLE_EMA_STRATEGY = False
            
            macd_signal = detector.detect_signals_all_strategies(epic, pair, spread_pips, timeframe)
            
            # Restore original setting
            config.SIMPLE_EMA_STRATEGY = original_ema
            
            print(f"   📊 MACD Strategy result: {len(macd_signal)} signals")
            if macd_signal:
                for signal in macd_signal:
                    print(f"      Strategy: {signal.get('strategy', 'unknown')}")
                    print(f"      Type: {signal.get('signal_type', 'unknown')}")
                    print(f"      Confidence: {signal.get('confidence_score', 0):.1%}")
            
        except Exception as e:
            print(f"   ❌ MACD Strategy failed: {e}")
        
        # Test 3: Combined Strategy
        print("\n🟢 TEST 3: Combined Strategy")
        try:
            # Ensure both are enabled for combined test
            config.SIMPLE_EMA_STRATEGY = True
            config.MACD_EMA_STRATEGY = True
            
            combined_signal = detector.detect_signals_all_strategies(epic, pair, spread_pips, timeframe)
            
            print(f"   📊 Combined Strategy result: {len(combined_signal)} signals")
            if combined_signal:
                for signal in combined_signal:
                    print(f"      Strategy: {signal.get('strategy', 'unknown')}")
                    print(f"      Type: {signal.get('signal_type', 'unknown')}")
                    print(f"      Confidence: {signal.get('confidence_score', 0):.1%}")
            else:
                print("   ❌ No signals from combined strategy")
                print("   💡 This might be why debug shows 'Strategy: None'")
            
        except Exception as e:
            print(f"   ❌ Combined Strategy failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Strategy testing failed: {e}")

def test_debug_commands():
    """Test the specific debug commands that should work"""
    print("\n🔬 DEBUG COMMAND TESTS")
    print("-" * 30)
    
    print("Try these commands to get more specific information:")
    print("   python main.py debug-combined --epic CS.D.EURUSD.CEEM.IP")
    print("   python main.py debug-macd --epic CS.D.EURUSD.CEEM.IP")
    
    # Test if combined debug works
    try:
        from commands.debug_commands import DebugCommands
        debug_cmd = DebugCommands()
        
        print("\n🔄 Testing combined strategy debug...")
        success = debug_cmd.debug_combined_strategies('CS.D.EURUSD.CEEM.IP')
        
        if success:
            print("✅ Combined strategy debug completed")
        else:
            print("❌ Combined strategy debug failed")
            
    except Exception as e:
        print(f"❌ Debug command test failed: {e}")

def main():
    """Run complete strategy diagnosis"""
    print("Diagnosing why 'Strategy: None' appears in debug output...\n")
    
    # Check configuration
    combined_enabled = diagnose_strategy_execution()
    
    # Test individual strategies
    test_individual_strategies()
    
    # Test debug commands
    test_debug_commands()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if combined_enabled:
        print("✅ Configuration suggests combined strategy should work")
        print("❌ But combined strategy is failing to produce signals")
        print("\n💡 MOST LIKELY ISSUES:")
        print("   1. Combined strategy logic is too restrictive")
        print("   2. Market conditions don't meet combined criteria")
        print("   3. Missing indicators for combined strategy")
        print("   4. Confidence threshold too high in combined mode")
    else:
        print("❌ Configuration has issues preventing strategy execution")
        print("\n💡 CONFIGURATION FIXES NEEDED:")
        print("   1. Ensure COMBINED_STRATEGY_MODE is defined in config.py")
        print("   2. Check strategy enable/disable settings")
    
    print("\n📋 RECOMMENDED NEXT STEPS:")
    print("   1. Run: python main.py debug-combined --epic CS.D.EURUSD.CEEM.IP")
    print("   2. Temporarily disable combined mode to test individual strategies")
    print("   3. Lower confidence thresholds for testing")
    print("   4. Check if missing required config settings")

if __name__ == "__main__":
    main()