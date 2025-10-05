#!/usr/bin/env python3
"""
Simple Scanner Diagnostic for Static Config
Works with your current config.py setup
"""

import sys
sys.path.append('/app/forex_scanner')

def check_current_config():
    """Check your actual config.py settings"""
    print("⚙️ CHECKING CURRENT CONFIGURATION")
    print("-" * 50)
    
    try:
        import config
        
        settings = {
            'SCAN_INTERVAL': getattr(config, 'SCAN_INTERVAL', 'NOT_SET'),
            'SPREAD_PIPS': getattr(config, 'SPREAD_PIPS', 'NOT_SET'),
            'MIN_CONFIDENCE': getattr(config, 'MIN_CONFIDENCE', 'NOT_SET'),
            'USE_BID_ADJUSTMENT': getattr(config, 'USE_BID_ADJUSTMENT', 'NOT_SET'),
            'DEFAULT_TIMEFRAME': getattr(config, 'DEFAULT_TIMEFRAME', 'NOT_SET'),
            'DATABASE_URL': 'FOUND' if hasattr(config, 'DATABASE_URL') else 'NOT_FOUND',
            'EPIC_LIST': f"{len(getattr(config, 'EPIC_LIST', []))} pairs" if hasattr(config, 'EPIC_LIST') else 'NOT_SET'
        }
        
        for key, value in settings.items():
            if key == 'MIN_CONFIDENCE' and isinstance(value, (int, float)):
                status = "⚠️ HIGH" if value > 0.7 else "✅ OK" if value <= 0.6 else "⚠️ MODERATE"
                print(f"   {status} {key}: {value:.1%}")
            else:
                status = "✅" if value != 'NOT_SET' and value != 'NOT_FOUND' else "❌"
                print(f"   {status} {key}: {value}")
        
        return config
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return None

def test_database_connection(config):
    """Test if database connection works"""
    print("\n🗄️ TESTING DATABASE CONNECTION")
    print("-" * 50)
    
    try:
        if not hasattr(config, 'DATABASE_URL'):
            print("❌ DATABASE_URL not found in config.py")
            return False
        
        from core.database import DatabaseManager
        db = DatabaseManager(config.DATABASE_URL)
        
        # Test basic connection
        result = db.execute_query("SELECT 1 as test")
        print(f"✅ Database connection works")
        
        # Check for data
        pairs_to_check = ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP']
        
        for epic in pairs_to_check:
            try:
                data_check = db.execute_query("""
                    SELECT COUNT(*) as count, MAX(start_time) as latest
                    FROM ig_candles 
                    WHERE epic = %s
                """, {'epic': epic})
                
                if len(data_check) > 0:
                    count = data_check.iloc[0]['count']
                    latest = data_check.iloc[0]['latest']
                    epic_short = epic.replace('CS.D.', '').replace('.MINI.IP', '')
                    print(f"   📊 {epic_short}: {count} rows, latest: {latest}")
                else:
                    print(f"   📭 {epic}: No data")
                    
            except Exception as e:
                print(f"   ❌ {epic}: Query error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_confidence_thresholds(config):
    """Test scanner with different confidence levels"""
    print("\n🎯 TESTING CONFIDENCE THRESHOLDS")
    print("-" * 50)
    
    try:
        from core.scanner import IntelligentForexScanner
        from core.database import DatabaseManager
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Test pairs from your config or use defaults
        test_pairs = getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'])[:3]
        
        print(f"Testing with pairs: {[p.replace('CS.D.', '').replace('.MINI.IP', '') for p in test_pairs]}")
        
        # Test different confidence levels
        confidence_levels = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        
        for min_conf in confidence_levels:
            print(f"\n🔍 Testing confidence: {min_conf:.1%}")
            
            try:
                scanner = IntelligentForexScanner(
                    db_manager=db,
                    epic_list=test_pairs,
                    min_confidence=min_conf,
                    enable_claude_analysis=False,  # Disable for speed
                    spread_pips=config.SPREAD_PIPS,
                    use_bid_adjustment=config.USE_BID_ADJUSTMENT
                )
                
                signals = scanner.scan_once()
                print(f"   📊 Found {len(signals)} signals")
                
                if signals:
                    for signal in signals[:3]:  # Show first 3
                        epic = signal.get('epic', 'Unknown').replace('CS.D.', '').replace('.MINI.IP', '')
                        signal_type = signal.get('signal_type', 'Unknown')
                        confidence = signal.get('confidence_score', 0)
                        strategy = signal.get('strategy', 'unknown')
                        print(f"      📈 {epic}: {signal_type} ({strategy}) - {confidence:.1%}")
                    
                    if len(signals) > 3:
                        print(f"      ... and {len(signals) - 3} more signals")
                    
                    # If we found signals at reasonable confidence, recommend this level
                    if min_conf >= 0.5:
                        print(f"✅ RECOMMENDATION: Set MIN_CONFIDENCE to {min_conf:.1%}")
                        return min_conf
                else:
                    print("   📭 No signals found")
                    
            except Exception as e:
                print(f"   ❌ Scanner error: {e}")
                
    except Exception as e:
        print(f"❌ Confidence test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def test_live_vs_backtest(config):
    """Compare live scanning vs backtest results"""
    print("\n⚖️ TESTING LIVE VS BACKTEST")
    print("-" * 50)
    
    try:
        # Test live scan first
        print("🔴 LIVE SCAN:")
        from core.scanner import IntelligentForexScanner
        from core.database import DatabaseManager
        
        db = DatabaseManager(config.DATABASE_URL)
        test_epic = 'CS.D.EURUSD.CEEM.IP'
        
        scanner = IntelligentForexScanner(
            db_manager=db,
            epic_list=[test_epic],
            min_confidence=0.4,  # Low threshold for testing
            enable_claude_analysis=False
        )
        
        live_signals = scanner.scan_once()
        print(f"   📊 Live scan found: {len(live_signals)} signals")
        
        # Test backtest using main.py command
        print("\n🔵 BACKTEST (using project commands):")
        print("   Run this command manually to compare:")
        print(f"   python main.py backtest --epic {test_epic} --days 3 --timeframe 15m")
        print("   Expected: Should find multiple signals if system is working")
        
        return len(live_signals)
        
    except Exception as e:
        print(f"❌ Live vs backtest test failed: {e}")
        return 0

def test_debug_commands():
    """Test the project's built-in debug commands"""
    print("\n🔬 TESTING DEBUG COMMANDS")
    print("-" * 50)
    
    print("Available debug commands to try manually:")
    print("   python main.py debug --epic CS.D.EURUSD.CEEM.IP")
    print("   python main.py debug-combined --epic CS.D.EURUSD.CEEM.IP")
    print("   python main.py debug-macd --epic CS.D.EURUSD.CEEM.IP")
    print("   python main.py scan --config-check")
    
    try:
        from commands.debug_commands import DebugCommands
        debug_cmd = DebugCommands()
        print("✅ Debug commands module loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Debug commands failed to load: {e}")
        return False

def main():
    """Run complete diagnostic"""
    print("🔍 FOREX SCANNER DIAGNOSTIC - STATIC CONFIG")
    print("=" * 60)
    
    # Check configuration
    config = check_current_config()
    if not config:
        print("❌ Cannot proceed without valid configuration")
        return
    
    # Test database
    db_ok = test_database_connection(config)
    if not db_ok:
        print("❌ Cannot proceed without database connection")
        return
    
    # Test confidence levels
    optimal_confidence = test_confidence_thresholds(config)
    
    # Test live vs backtest
    live_signal_count = test_live_vs_backtest(config)
    
    # Test debug commands
    debug_ok = test_debug_commands()
    
    # Summary and recommendations
    print("\n" + "=" * 60)
    print("🎯 SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    current_confidence = getattr(config, 'MIN_CONFIDENCE', 0.6)
    
    if optimal_confidence and optimal_confidence != current_confidence:
        print(f"💡 MAIN ISSUE: Confidence threshold too high!")
        print(f"   Current: {current_confidence:.1%}")
        print(f"   Recommended: {optimal_confidence:.1%}")
        print(f"   Fix: Edit config.py and change MIN_CONFIDENCE = {optimal_confidence}")
    
    elif live_signal_count == 0:
        print("💡 MAIN ISSUE: No signals found even at low confidence")
        print("   Possible causes:")
        print("   1. Market conditions (outside trading hours?)")
        print("   2. Data is stale (no recent market data)")
        print("   3. Strategy configuration issue")
        print("   4. Intelligence/filtering is too restrictive")
    
    else:
        print("✅ System appears to be working!")
        print(f"   Found {live_signal_count} signals in live scan")
    
    print("\n📋 NEXT STEPS:")
    print("1. Try manual debug commands listed above")
    print("2. Run a backtest to verify: python main.py backtest --epic CS.D.EURUSD.CEEM.IP --days 3")
    print("3. If backtest works but live doesn't, check market hours/data freshness")
    print("4. Consider temporarily lowering MIN_CONFIDENCE for testing")

if __name__ == "__main__":
    main()