#!/usr/bin/env python3
"""
Debug why live scanner doesn't find the same signals as backtest
"""

import sys
sys.path.append('/app/forex_scanner')

def compare_live_vs_backtest():
    """Compare live scanner vs backtest detection"""
    
    print("🔍 DEBUGGING LIVE VS BACKTEST SIGNAL DETECTION")
    print("=" * 70)
    
    try:
        from core.scanner import IntelligentForexScanner
        from core.database import DatabaseManager
        from commands.backtest_commands import BacktestCommands
        import config
        from datetime import datetime, timedelta
        
        # Initialize components
        db_manager = DatabaseManager(config.DATABASE_URL)
        
        # Test the specific pairs from your backtest
        test_pairs = ['CS.D.EURUSD.MINI.IP', 'CS.D.EURJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP']
        
        print("📊 Testing specific pairs that showed signals in backtest:")
        print(f"   Pairs: {test_pairs}")
        print(f"   Target signals: BULL EURUSD/EURJPY, BEAR AUDJPY")
        print(f"   Expected confidence: 82.3%, 95.0%, 95.0%")
        
        # 1. TEST LIVE SCANNER
        print("\n🔴 LIVE SCANNER TEST:")
        print("-" * 40)
        
        live_scanner = IntelligentForexScanner(
            db_manager=db_manager,
            epic_list=test_pairs,
            claude_api_key=getattr(config, 'CLAUDE_API_KEY', None),
            enable_claude_analysis=getattr(config, 'ENABLE_CLAUDE_ANALYSIS', False),
            use_bid_adjustment=getattr(config, 'USE_BID_ADJUSTMENT', True),
            spread_pips=getattr(config, 'SPREAD_PIPS', 1.5),
            min_confidence=getattr(config, 'MIN_CONFIDENCE', 0.6),
            user_timezone=getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm')
        )
        
        print("🔍 Running live scanner scan...")
        live_signals = live_scanner.scan_once()
        
        print(f"📊 Live scanner results: {len(live_signals)} signals")
        if live_signals:
            for i, signal in enumerate(live_signals, 1):
                epic = signal.get('epic', 'Unknown')
                signal_type = signal.get('signal_type', 'Unknown')
                confidence = signal.get('confidence_score', 0)
                strategy = signal.get('strategy', 'unknown')
                timestamp = signal.get('timestamp', 'unknown')
                print(f"   {i}. {signal_type} {epic} ({strategy}) - {confidence:.1%} at {timestamp}")
        else:
            print("   ❌ No signals found by live scanner")
        
        # 2. TEST BACKTEST
        print("\n🔵 BACKTEST TEST:")
        print("-" * 40)
        
        backtest_commands = BacktestCommands()
        
        print("🔍 Running backtest for same timeframe...")
        
        # Run backtest for a short recent period
        current_time = datetime.now()
        
        for epic in test_pairs:
            print(f"\n📈 Backtesting {epic}:")
            try:
                # This might not work exactly the same way, but let's try
                success = backtest_commands.run_backtest(
                    epic=epic,
                    days=1,  # Just last day
                    timeframe='15m',
                    show_signals=True
                )
                
                if success:
                    print(f"   ✅ Backtest completed for {epic}")
                else:
                    print(f"   ⚠️ Backtest had issues for {epic}")
                    
            except Exception as e:
                print(f"   ❌ Backtest failed for {epic}: {e}")
        
        # 3. ANALYZE DIFFERENCES
        print("\n🔍 ANALYZING DIFFERENCES:")
        print("-" * 40)
        
        analyze_scanner_configuration(live_scanner)
        check_intelligence_filtering()
        check_time_sensitivity()
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_scanner_configuration(scanner):
    """Analyze the scanner configuration for filtering"""
    
    print("🔧 Scanner Configuration Analysis:")
    
    # Check if intelligence filtering is enabled
    if hasattr(scanner, 'intelligence_engine'):
        print("   ✅ Intelligence engine active")
        
        # Check market intelligence state
        if hasattr(scanner.intelligence_engine, 'market_intelligence'):
            intel = scanner.intelligence_engine.market_intelligence
            print(f"   📊 Market regime: {intel.get('regime', 'unknown')}")
            print(f"   📊 Confidence: {intel.get('confidence', 'unknown')}")
            print(f"   📊 Risk level: {intel.get('risk_level', 'unknown')}")
            
            # Check if regime affects filtering
            if intel.get('regime') in ['ranging', 'low_volatility']:
                print("   ⚠️ Low volatility regime might filter out signals")
            
    else:
        print("   ❌ No intelligence engine found")
    
    # Check confidence thresholds
    min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.6)
    print(f"   🎯 Minimum confidence threshold: {min_confidence:.1%}")
    
    # Check timeframe settings
    timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '5m')
    print(f"   ⏰ Default timeframe: {timeframe}")
    
    # Check strategy configuration
    strategies_enabled = []
    if getattr(config, 'SIMPLE_EMA_STRATEGY', True):
        strategies_enabled.append('EMA')
    if getattr(config, 'MACD_EMA_STRATEGY', False):
        strategies_enabled.append('MACD')
    
    print(f"   🎯 Enabled strategies: {', '.join(strategies_enabled)}")

def check_intelligence_filtering():
    """Check if intelligence filtering is too restrictive"""
    
    print("\n🧠 Intelligence Filtering Analysis:")
    
    try:
        import config
        
        # Check intelligence settings
        intelligence_settings = [
            ('MARKET_INTELLIGENCE_ENABLED', getattr(config, 'MARKET_INTELLIGENCE_ENABLED', 'Not set')),
            ('INTELLIGENT_FILTERING', getattr(config, 'INTELLIGENT_FILTERING', 'Not set')),
            ('ADAPTIVE_CONFIDENCE_THRESHOLD', getattr(config, 'ADAPTIVE_CONFIDENCE_THRESHOLD', 'Not set')),
            ('REGIME_BASED_FILTERING', getattr(config, 'REGIME_BASED_FILTERING', 'Not set')),
        ]
        
        for setting, value in intelligence_settings:
            status = "✅" if value not in ['Not set', False] else "❌"
            print(f"   {status} {setting}: {value}")
            
            if setting == 'ADAPTIVE_CONFIDENCE_THRESHOLD' and value is True:
                print("      ⚠️ Adaptive thresholds might be raising confidence requirements")
            
            if setting == 'REGIME_BASED_FILTERING' and value is True:
                print("      ⚠️ Regime filtering might be excluding signals")
        
    except Exception as e:
        print(f"   ❌ Error checking intelligence settings: {e}")

def check_time_sensitivity():
    """Check if there's a time sensitivity issue"""
    
    print("\n⏰ Time Sensitivity Analysis:")
    
    try:
        from datetime import datetime
        import pytz
        
        current_time = datetime.now()
        print(f"   🕐 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if we're in a different timezone
        tz = pytz.timezone('Europe/Stockholm')
        stockholm_time = datetime.now(tz)
        print(f"   🕐 Stockholm time: {stockholm_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # The backtest signals were at:
        # 2025-06-30 13:30 - EURUSD BULL, EURJPY BULL  
        # 2025-06-30 12:30 - AUDJPY BEAR
        
        print("\n   📊 Backtest signal times:")
        print("      13:30 - EURUSD BULL (82.3%), EURJPY BULL (95.0%)")
        print("      12:30 - AUDJPY BEAR (95.0%)")
        
        # Check how much time has passed
        if current_time.hour > 13:
            time_diff = current_time.hour - 13
            print(f"   ⚠️ {time_diff} hours have passed since the 13:30 signals")
            print("   💡 Market conditions may have changed")
        
        if current_time.minute > 30:
            print("   ⚠️ Market has moved since the signal timestamps")
            print("   💡 Signals may no longer be valid")
            
    except Exception as e:
        print(f"   ❌ Time analysis error: {e}")

def test_manual_signal_detection():
    """Manually test signal detection for the specific pairs"""
    
    print("\n🔧 MANUAL SIGNAL DETECTION TEST:")
    print("-" * 40)
    
    try:
        from core.signal_detector import SignalDetector
        from core.database import DatabaseManager
        import config
        
        db_manager = DatabaseManager(config.DATABASE_URL)
        detector = SignalDetector(db_manager, 'Europe/Stockholm')
        
        # Test each pair individually
        test_pairs = [
            ('CS.D.EURUSD.MINI.IP', 'EURUSD'),
            ('CS.D.EURJPY.MINI.IP', 'EURJPY'), 
            ('CS.D.AUDJPY.MINI.IP', 'AUDJPY')
        ]
        
        for epic, pair in test_pairs:
            print(f"\n🔍 Testing {epic} ({pair}):")
            
            try:
                # Test different strategies
                strategies = [
                    ('EMA', lambda: detector.detect_signals_bid_adjusted(epic, pair, 1.5, '15m')),
                    ('MACD', lambda: detector.detect_macd_signals(epic, pair, 1.5, '15m')),
                    ('Combined', lambda: detector.detect_combined_signals(epic, pair, 1.5, '15m'))
                ]
                
                for strategy_name, strategy_func in strategies:
                    try:
                        signal = strategy_func()
                        if signal:
                            confidence = signal.get('confidence_score', 0)
                            signal_type = signal.get('signal_type', 'Unknown')
                            print(f"   ✅ {strategy_name}: {signal_type} ({confidence:.1%})")
                        else:
                            print(f"   ❌ {strategy_name}: No signal")
                    except Exception as e:
                        print(f"   ❌ {strategy_name}: Error - {e}")
                        
            except Exception as e:
                print(f"   ❌ Failed to test {epic}: {e}")
                
    except Exception as e:
        print(f"❌ Manual detection test failed: {e}")

if __name__ == "__main__":
    compare_live_vs_backtest()
    test_manual_signal_detection()
    
    print("\n" + "=" * 70)
    print("🎯 LIKELY CAUSES:")
    print("1. Intelligence filtering is too restrictive")
    print("2. Time sensitivity - signals expired")
    print("3. Different confidence thresholds")
    print("4. Market regime filtering")
    print("5. Different strategy weights in live vs backtest")
    print("\n💡 RECOMMENDATIONS:")
    print("1. Check MIN_CONFIDENCE setting")
    print("2. Temporarily disable intelligence filtering")
    print("3. Compare exact timestamps")
    print("4. Check if regime filtering is active")