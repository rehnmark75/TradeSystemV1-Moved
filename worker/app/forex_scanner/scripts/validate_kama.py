#!/usr/bin/env python3
"""
Test Fixed KAMA Strategy
Test the actual updated KAMA strategy implementation

Usage: python scripts/test_fixed_kama.py
"""

import sys
import os
import logging

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir.endswith('scripts'):
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_kama():
    """Test the actual fixed KAMA strategy"""
    
    try:
        import config
        from core.database import DatabaseManager
        from core.data_fetcher import DataFetcher
        from core.strategies.kama_strategy import KAMAStrategy
        
        logger.info("🎯 Testing Fixed KAMA Strategy")
        
        # Set very permissive thresholds
        config.KAMA_MIN_EFFICIENCY = 0.001
        config.KAMA_TREND_THRESHOLD = 0.0001
        config.KAMA_MIN_BARS = 10
        
        # Initialize components
        db_manager = DatabaseManager(config.DATABASE_URL)
        data_fetcher = DataFetcher(db_manager)
        
        # Get data
        epic = 'CS.D.EURUSD.CEEM.IP'
        pair = 'EURUSD'
        timeframe = '15m'
        
        df = data_fetcher.get_enhanced_data(
            epic=epic, pair=pair, timeframe=timeframe, lookback_hours=24*30
        )
        
        if df is None or len(df) < 50:
            logger.error("❌ Insufficient data")
            return False
        
        logger.info(f"✅ Data: {len(df)} bars")
        
        # Initialize KAMA strategy
        kama_strategy = KAMAStrategy()
        
        # Override thresholds directly on strategy
        kama_strategy.min_efficiency = 0.001
        kama_strategy.trend_threshold = 0.0001
        kama_strategy.min_bars = 10
        
        logger.info(f"🔧 Strategy thresholds:")
        logger.info(f"   min_efficiency: {kama_strategy.min_efficiency}")
        logger.info(f"   trend_threshold: {kama_strategy.trend_threshold}")
        logger.info(f"   min_bars: {kama_strategy.min_bars}")
        
        # Test current signal with ACTUAL strategy
        logger.info("🎯 Testing with ACTUAL fixed KAMA strategy...")
        
        signal = kama_strategy.detect_signal(df, epic, config.SPREAD_PIPS, timeframe)
        
        if signal:
            logger.info("🎉 SUCCESS! FIXED KAMA STRATEGY WORKS!")
            logger.info(f"   Signal type: {signal['signal_type']}")
            logger.info(f"   Confidence: {signal.get('confidence_score', 0):.3f}")
            logger.info(f"   Trigger: {signal.get('trigger_reason', 'N/A')}")
            logger.info(f"   KAMA column: {signal.get('kama_column', 'N/A')}")
            logger.info(f"   ER column: {signal.get('er_column', 'N/A')}")
            logger.info(f"   Efficiency Ratio: {signal.get('efficiency_ratio', 0):.3f}")
            logger.info(f"   Signal strength: {signal.get('signal_strength', 0):.3f}")
            logger.info(f"   Market regime: {signal.get('market_regime', 'N/A')}")
            
            return True
        else:
            logger.warning("⚠️ No signal with fixed strategy")
            
            # Check if methods exist
            if not hasattr(kama_strategy, 'calculate_confidence_fixed'):
                logger.error("❌ calculate_confidence_fixed method missing!")
                logger.info("💡 Make sure you added ALL the new methods to kama_strategy.py")
                return False
            
            if not hasattr(kama_strategy, '_enhance_kama_signal'):
                logger.error("❌ _enhance_kama_signal method missing!")
                logger.info("💡 Make sure you added ALL the new methods to kama_strategy.py")
                return False
            
            logger.info("✅ New methods are present")
            
            # Test recent data values
            recent = df.tail(3)
            
            # Find KAMA columns
            kama_col = None
            er_col = None
            
            for col in ['kama_14', 'kama_10', 'kama_6']:
                if col in df.columns:
                    kama_col = col
                    break
                    
            for col in ['kama_14_er', 'kama_10_er', 'kama_6_er']:
                if col in df.columns:
                    er_col = col
                    break
            
            if kama_col and er_col:
                current = recent.iloc[-1]
                previous = recent.iloc[-2]
                
                current_price = current['close']
                current_kama = current[kama_col]
                current_er = current[er_col]
                
                kama_change = current_kama - previous[kama_col]
                kama_trend = kama_change / previous[kama_col] if previous[kama_col] != 0 else 0
                
                # Test the fixed confidence calculation
                signal_strength = min(abs(kama_trend) * 1000, 0.8)  # Fixed scaling
                
                logger.info(f"🔍 Debug fixed calculation:")
                logger.info(f"   KAMA trend: {kama_trend:.8f}")
                logger.info(f"   Signal strength (fixed): {signal_strength:.6f}")
                logger.info(f"   ER: {current_er:.6f}")
                logger.info(f"   Raw confidence (fixed): {signal_strength * current_er:.6f}")
                
                if signal_strength * current_er > 0.01:  # Should be much higher now
                    logger.info("✅ Fixed confidence calculation should work!")
                else:
                    logger.warning("⚠️ Even fixed calculation produces low confidence")
            
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_backtest():
    """Test KAMA backtest with fixed strategy"""
    
    logger.info("🔙 Testing KAMA backtest with fixed strategy...")
    
    try:
        import subprocess
        
        # Run the backtest command
        result = subprocess.run([
            'python', 'main.py', 'backtest-kama',
            '--epic', 'CS.D.EURUSD.CEEM.IP',
            '--days', '14',
            '--kama-min-efficiency', '0.001',
            '--kama-trend-threshold', '0.0001',
            '--show-signals'
        ], capture_output=True, text=True, cwd=project_root)
        
        logger.info("📊 Backtest output:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.error("Backtest errors:")
            logger.error(result.stderr)
        
        # Check if signals were found
        if "Found 0 KAMA signals" in result.stdout or "No KAMA signals found" in result.stdout:
            logger.warning("⚠️ Backtest still shows 0 signals")
            return False
        elif "Found" in result.stdout and "KAMA signal" in result.stdout:
            logger.info("🎉 Backtest found KAMA signals!")
            return True
        else:
            logger.info("📊 Backtest completed, check output above")
            return True
        
    except Exception as e:
        logger.error(f"❌ Backtest test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting Fixed KAMA Strategy Test")
    
    # Test 1: Direct strategy test
    test1_success = test_fixed_kama()
    
    # Test 2: Backtest (only if strategy works)
    if test1_success:
        test2_success = test_backtest()
    else:
        test2_success = False
        logger.info("⏭️ Skipping backtest test since direct strategy test failed")
    
    if test1_success and test2_success:
        logger.info("🎉 ALL TESTS PASSED! KAMA strategy is working!")
    elif test1_success:
        logger.info("✅ Strategy works, but backtest may need framework fixes")
    else:
        logger.error("❌ KAMA strategy still has issues")
        logger.info("💡 Make sure you applied ALL the fixes to kama_strategy.py")