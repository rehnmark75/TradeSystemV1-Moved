# direct_bb_test.py
"""
Direct BB+Supertrend Strategy Test
Bypasses the backtest system and directly tests the strategy
Run from: forex_scanner/scripts/
"""

import sys
import os
import logging

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bb_supertrend_directly():
    """Test BB+Supertrend strategy directly without backtest framework"""
    try:
        import config
        from core.database import DatabaseManager
        from core.signal_detector import SignalDetector
        
        # Override config parameters
        config.BB_PERIOD = 14
        config.BB_STD_DEV = 1.8
        config.SUPERTREND_PERIOD = 8
        config.SUPERTREND_MULTIPLIER = 2.5
        config.BB_SUPERTREND_BASE_CONFIDENCE = 0.5
        config.MIN_CONFIDENCE = 0.65
        
        logger.info("🔧 Forcing config parameters:")
        logger.info(f"   BB_PERIOD = {config.BB_PERIOD}")
        logger.info(f"   BB_STD_DEV = {config.BB_STD_DEV}")
        logger.info(f"   SUPERTREND_PERIOD = {config.SUPERTREND_PERIOD}")
        logger.info(f"   SUPERTREND_MULTIPLIER = {config.SUPERTREND_MULTIPLIER}")
        
        # Initialize components
        db_manager = DatabaseManager(config.DATABASE_URL)
        detector = SignalDetector(db_manager, getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm'))
        
        if not detector.bb_supertrend_strategy:
            logger.error("❌ BB+Supertrend strategy not available")
            return False
        
        # Get data for last 7 days
        epic = 'CS.D.EURUSD.CEEM.IP'
        pair = 'EURUSD'
        
        df = detector.data_fetcher.get_enhanced_data(epic, pair, timeframe='15m', lookback_hours=7*24)
        
        if df is None or len(df) < 50:
            logger.error(f"❌ Insufficient data: {len(df) if df is not None else 0} bars")
            return False
        
        logger.info(f"📊 Testing {len(df)} bars of data")
        
        # Test current signal detection
        logger.info("\n🔍 Testing current signal detection...")
        signal = detector.bb_supertrend_strategy.detect_signal(df, epic, 1.5, '15m')
        
        if signal:
            logger.info("🎯 CURRENT SIGNAL DETECTED!")
            logger.info(f"   Type: {signal.get('signal_type')}")
            logger.info(f"   Confidence: {signal.get('confidence_score', 0):.3f}")
            logger.info(f"   Entry Price: {signal.get('entry_price', 0):.5f}")
        else:
            logger.info("ℹ️ No current signal")
        
        # Scan historical data for signals
        logger.info("\n📈 Scanning historical data for signals...")
        signals_found = []
        
        for i in range(50, len(df)):
            # Create subset of data up to this point
            test_df = df.iloc[:i+1].copy()
            
            # Test signal detection
            test_signal = detector.bb_supertrend_strategy.detect_signal(test_df, epic, 1.5, '15m')
            
            if test_signal:
                timestamp = df.iloc[i].get('start_time', f'Bar {i}')
                signals_found.append({
                    'timestamp': timestamp,
                    'signal_type': test_signal.get('signal_type'),
                    'confidence': test_signal.get('confidence_score', 0),
                    'entry_price': test_signal.get('entry_price', 0)
                })
                
                logger.info(f"   📍 {timestamp}: {test_signal.get('signal_type')} "
                          f"(confidence: {test_signal.get('confidence_score', 0):.3f})")
        
        logger.info(f"\n📊 RESULTS:")
        logger.info(f"   Total bars analyzed: {len(df) - 50}")
        logger.info(f"   Signals found: {len(signals_found)}")
        
        if signals_found:
            avg_confidence = sum(s['confidence'] for s in signals_found) / len(signals_found)
            logger.info(f"   Average confidence: {avg_confidence:.3f}")
            
            signal_types = {}
            for s in signals_found:
                signal_types[s['signal_type']] = signal_types.get(s['signal_type'], 0) + 1
            
            for signal_type, count in signal_types.items():
                logger.info(f"   {signal_type} signals: {count}")
        else:
            logger.warning("⚠️ No signals found in historical data")
            logger.info("💡 Try lowering MIN_CONFIDENCE or adjusting parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_bb_supertrend_directly()