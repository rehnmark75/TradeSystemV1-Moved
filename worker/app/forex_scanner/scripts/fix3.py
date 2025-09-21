#!/usr/bin/env python3
"""
MACD Signal Rejection Debug - Find exactly where signals are being filtered out
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime, timedelta

def setup_paths():
    """Setup Python paths for imports"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_project_roots = [
        os.path.dirname(script_dir),  # scripts/.. (parent of scripts)
        os.path.dirname(os.path.dirname(script_dir)),  # scripts/../.. (grandparent)
        script_dir,  # scripts/ (current dir)
    ]
    
    for root in possible_project_roots:
        config_path = os.path.join(root, 'config.py')
        core_path = os.path.join(root, 'core')
        
        if os.path.exists(config_path) and os.path.exists(core_path):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None

def debug_specific_crossover():
    """Debug why a specific known crossover isn't generating a signal"""
    
    print("🔍 DEBUGGING SPECIFIC MACD CROSSOVER REJECTION")
    print("=" * 60)
    
    # Setup
    setup_paths()
    import config
    from core.database import DatabaseManager
    from core.strategies.macd_strategy import MACDStrategy
    from core.data_fetcher import DataFetcher
    
    # Apply emergency config
    config.EMERGENCY_MACD_MODE = True
    config.MACD_ENHANCED_FILTERS_ENABLED = False
    config.MACD_REQUIRE_EMA200_ALIGNMENT = False
    config.MACD_DISABLE_EMA200_FILTER = True
    config.MIN_CONFIDENCE = 0.10  # Very low
    config.INTELLIGENCE_MODE = 'disabled'
    
    # Set up detailed logging for MACD
    logging.basicConfig(level=logging.DEBUG)
    macd_logger = logging.getLogger('core.strategies.macd_strategy')
    macd_logger.setLevel(logging.DEBUG)
    
    # Initialize components
    db_manager = DatabaseManager(config.DATABASE_URL)
    data_fetcher = DataFetcher(db_manager, getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm'))
    strategy = MACDStrategy(data_fetcher=data_fetcher)
    
    # Get data
    epic = "CS.D.EURUSD.CEEM.IP"
    df = data_fetcher.get_enhanced_data(epic, "EURUSD", "15m", lookback_hours=168)
    
    print(f"📊 Testing known crossover at bar 437 (Bullish)")
    print(f"   Data shape: {df.shape}")
    
    # Test the specific crossover we know exists (bar 437)
    crossover_index = 437
    current_data = df.iloc[:crossover_index+1].copy()
    
    print(f"\n🔍 DETAILED CROSSOVER ANALYSIS:")
    latest = current_data.iloc[-1]
    previous = current_data.iloc[-2]
    
    print(f"   Bar Index: {crossover_index}")
    print(f"   Date: {current_data.index[-1]}")
    print(f"   Previous MACD Histogram: {previous.get('macd_histogram', 'Missing')}")
    print(f"   Current MACD Histogram: {latest.get('macd_histogram', 'Missing')}")
    print(f"   Crossover: {previous.get('macd_histogram', 0)} → {latest.get('macd_histogram', 0)}")
    print(f"   Price: {latest.get('close', 'Missing')}")
    print(f"   EMA200: {latest.get('ema_200', 'Missing')}")
    
    # Test emergency mode detection with debugging
    print(f"\n🚨 TESTING EMERGENCY MODE DETECTION:")
    print(f"   Emergency Mode: {getattr(config, 'EMERGENCY_MACD_MODE', False)}")
    print(f"   Enhanced Filters: {getattr(config, 'MACD_ENHANCED_FILTERS_ENABLED', True)}")
    print(f"   EMA200 Filter: {getattr(config, 'MACD_DISABLE_EMA200_FILTER', False)}")
    print(f"   Min Confidence: {getattr(config, 'MIN_CONFIDENCE', 0.7)}")
    
    # Call the detect_signal method with debugging
    try:
        signal = strategy.detect_signal(current_data, epic, 1.5, "15m")
        
        if signal:
            print(f"\n✅ SIGNAL DETECTED!")
            print(f"   Type: {signal.get('signal_type', 'Unknown')}")
            print(f"   Confidence: {signal.get('confidence_score', 0):.1%}")
            print(f"   Emergency Mode Used: {signal.get('emergency_mode', False)}")
            print(f"   Strategy: {signal.get('strategy', 'Unknown')}")
        else:
            print(f"\n❌ NO SIGNAL DETECTED")
            print(f"   This crossover should have generated a signal!")
            
            # Let's debug step by step
            print(f"\n🔧 STEP-BY-STEP DEBUG:")
            
            # Check if emergency mode is being used
            if hasattr(strategy, '_detect_signals_emergency_mode'):
                print(f"   Testing emergency mode detection...")
                emergency_signals = strategy._detect_signals_emergency_mode(
                    current_data, epic, "15m", latest, previous
                )
                print(f"   Emergency mode signals: {len(emergency_signals)}")
                if emergency_signals:
                    emergency_signal = emergency_signals[0]
                    print(f"   Emergency signal type: {emergency_signal.get('signal_type', 'Unknown')}")
                    print(f"   Emergency confidence: {emergency_signal.get('confidence_score', 0)}")
            
            # Check minimum bars
            min_bars = getattr(config, 'MACD_MIN_BARS_REQUIRED', 50)
            print(f"   Data length: {len(current_data)} vs min required: {min_bars}")
            
            # Check if strategy is enabled
            print(f"   MACD_EMA_STRATEGY: {getattr(config, 'MACD_EMA_STRATEGY', False)}")
            
            # Test manual crossover detection
            hist_current = latest.get('macd_histogram', 0)
            hist_prev = previous.get('macd_histogram', 0)
            print(f"   Manual crossover check:")
            print(f"     Previous <= 0: {hist_prev <= 0}")
            print(f"     Current > 0: {hist_current > 0}")
            print(f"     Is bullish crossover: {hist_prev <= 0 and hist_current > 0}")
            
            # Check confidence calculation
            if hist_prev <= 0 and hist_current > 0:
                print(f"   Crossover detected manually, checking why no signal...")
                print(f"   Histogram change: {abs(hist_current - hist_prev):.8f}")
                
                # Check against various thresholds
                emergency_threshold = 0.000001
                print(f"   Emergency threshold: {emergency_threshold:.8f}")
                print(f"   Above emergency threshold: {abs(hist_current - hist_prev) >= emergency_threshold}")
                
    except Exception as e:
        print(f"❌ Error during signal detection: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
    
    # Test with multiple known crossovers
    print(f"\n🔄 TESTING MULTIPLE KNOWN CROSSOVERS:")
    known_crossovers = [437, 449, 476]  # From our crossover detection
    
    for idx in known_crossovers:
        if idx < len(df):
            test_data = df.iloc[:idx+1].copy()
            signal = strategy.detect_signal(test_data, epic, 1.5, "15m")
            
            crossover_type = "BULLISH" if df.iloc[idx]['macd_histogram'] > 0 and df.iloc[idx-1]['macd_histogram'] <= 0 else "BEARISH"
            
            print(f"   Bar {idx} ({crossover_type}): {'✅ SIGNAL' if signal else '❌ NO SIGNAL'}")
            
            if signal:
                print(f"     Type: {signal.get('signal_type', 'Unknown')}")
                print(f"     Confidence: {signal.get('confidence_score', 0):.1%}")
                break  # Found one working signal

def check_strategy_methods():
    """Check which detection method is being used"""
    
    print(f"\n🔍 CHECKING STRATEGY METHODS:")
    
    setup_paths()
    import config
    from core.strategies.macd_strategy import MACDStrategy
    from core.data_fetcher import DataFetcher
    from core.database import DatabaseManager
    
    db_manager = DatabaseManager(config.DATABASE_URL)
    data_fetcher = DataFetcher(db_manager, getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm'))
    strategy = MACDStrategy(data_fetcher=data_fetcher)
    
    print(f"   Strategy class: {type(strategy).__name__}")
    print(f"   Has detect_signal: {hasattr(strategy, 'detect_signal')}")
    print(f"   Has detect_signals: {hasattr(strategy, 'detect_signals')}")
    print(f"   Has emergency mode: {hasattr(strategy, '_detect_signals_emergency_mode')}")
    print(f"   Has debug method: {hasattr(strategy, 'debug_signal_detection')}")
    
    # Check config values
    print(f"\n🎛️ CURRENT CONFIG VALUES:")
    config_values = [
        'EMERGENCY_MACD_MODE',
        'MACD_ENHANCED_FILTERS_ENABLED', 
        'MACD_REQUIRE_EMA200_ALIGNMENT',
        'MACD_DISABLE_EMA200_FILTER',
        'MIN_CONFIDENCE',
        'MACD_EMA_STRATEGY',
        'INTELLIGENCE_MODE'
    ]
    
    for value in config_values:
        print(f"   {value}: {getattr(config, value, 'NOT SET')}")

def run_comprehensive_debug():
    """Run comprehensive debugging"""
    
    print("🚨 COMPREHENSIVE MACD DEBUG")
    print("=" * 50)
    
    check_strategy_methods()
    debug_specific_crossover()
    
    print(f"\n📝 SUMMARY:")
    print("1. MACD crossovers exist in the data ✅")
    print("2. Emergency config is applied ✅") 
    print("3. Signal detection is being called ✅")
    print("4. BUT signals are being filtered out somewhere ❌")
    print()
    print("🔧 LIKELY CAUSES:")
    print("- Confidence calculation returning values below MIN_CONFIDENCE")
    print("- Emergency mode not being triggered correctly")
    print("- Additional validation filters rejecting signals")
    print("- Wrong method being called (detect_signal vs detect_signals)")

if __name__ == "__main__":
    run_comprehensive_debug()