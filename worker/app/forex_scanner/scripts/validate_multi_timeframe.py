# scripts/validate_multi_timeframe.py
"""
Validation script for multi-timeframe analysis implementation
Tests that all required methods and data fetching work correctly
"""

import sys
import os
sys.path.insert(0, '/app/forex_scanner')

import config
from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.signal_detector import SignalDetector
from analysis.multi_timeframe import MultiTimeframeAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data_fetcher_methods():
    """Validate that DataFetcher has required methods with correct signatures"""
    
    logger.info("🔍 Validating DataFetcher methods...")
    
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        data_fetcher = DataFetcher(db_manager, config.USER_TIMEZONE)
        
        # Check if get_enhanced_data method exists
        if hasattr(data_fetcher, 'get_enhanced_data'):
            logger.info("✅ get_enhanced_data method exists")
            
            # Check method signature
            import inspect
            sig = inspect.signature(data_fetcher.get_enhanced_data)
            params = list(sig.parameters.keys())
            
            required_params = ['epic', 'pair']
            optional_params = ['timeframe', 'lookback_hours']
            
            has_required = all(param in params for param in required_params)
            has_optional = any(param in params for param in optional_params)
            
            if has_required:
                logger.info(f"✅ Required parameters present: {required_params}")
            else:
                logger.warning(f"⚠️ Missing required parameters. Found: {params}")
            
            if has_optional:
                logger.info(f"✅ Optional parameters available: {optional_params}")
            
            # Test method call
            test_epic = 'CS.D.EURUSD.CEEM.IP'
            test_pair = 'EURUSD'
            
            logger.info(f"🧪 Testing get_enhanced_data with {test_epic}...")
            
            try:
                df = data_fetcher.get_enhanced_data(
                    epic=test_epic,
                    pair=test_pair,
                    timeframe='5m',
                    lookback_hours=48
                )
                
                if df is not None and len(df) > 0:
                    logger.info(f"✅ get_enhanced_data works: {len(df)} rows returned")
                    logger.info(f"   Columns: {list(df.columns)}")
                    return True
                else:
                    logger.error("❌ get_enhanced_data returned empty data")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ get_enhanced_data failed: {e}")
                return False
        else:
            logger.error("❌ get_enhanced_data method not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ DataFetcher validation failed: {e}")
        return False


def validate_multi_timeframe_analyzer():
    """Validate MultiTimeframeAnalyzer implementation"""
    
    logger.info("🔍 Validating MultiTimeframeAnalyzer...")
    
    try:
        # Initialize components
        db_manager = DatabaseManager(config.DATABASE_URL)
        data_fetcher = DataFetcher(db_manager, config.USER_TIMEZONE)
        mt_analyzer = MultiTimeframeAnalyzer(data_fetcher)
        
        # Test data for validation
        test_epic = 'CS.D.EURUSD.CEEM.IP'
        test_pair = 'EURUSD'
        
        logger.info("🧪 Testing multi-timeframe data fetching...")
        
        # Get data for multiple timeframes
        df_5m = data_fetcher.get_enhanced_data(test_epic, test_pair, '5m', lookback_hours=48)
        df_15m = data_fetcher.get_enhanced_data(test_epic, test_pair, '15m', lookback_hours=168)
        df_1h = data_fetcher.get_enhanced_data(test_epic, test_pair, '1h', lookback_hours=720)
        
        if all(df is not None and len(df) > 50 for df in [df_5m, df_15m, df_1h]):
            logger.info("✅ Multi-timeframe data fetching successful")
            logger.info(f"   5m: {len(df_5m)} bars")
            logger.info(f"   15m: {len(df_15m)} bars") 
            logger.info(f"   1h: {len(df_1h)} bars")
        else:
            logger.error("❌ Insufficient data for multi-timeframe analysis")
            return False
        
        # Test trend determination
        logger.info("🧪 Testing trend determination...")
        trend_5m = mt_analyzer._determine_trend(df_5m, period=20)
        trend_15m = mt_analyzer._determine_trend(df_15m, period=20)
        trend_1h = mt_analyzer._determine_trend(df_1h, period=20)
        
        logger.info(f"✅ Trend analysis results:")
        logger.info(f"   5m trend: {trend_5m}")
        logger.info(f"   15m trend: {trend_15m}")
        logger.info(f"   1h trend: {trend_1h}")
        
        # Test full multi-timeframe analysis
        logger.info("🧪 Testing full multi-timeframe analysis...")
        df_5m_enhanced, df_15m_enhanced, df_1h_enhanced = mt_analyzer.add_multi_timeframe_analysis(
            df_5m, df_15m, df_1h, test_pair
        )
        
        # Check if trend columns were added
        trend_columns = [col for col in df_5m_enhanced.columns if 'trend_' in col]
        if trend_columns:
            logger.info(f"✅ Trend columns added: {trend_columns}")
        else:
            logger.warning("⚠️ No trend columns found in enhanced data")
        
        # Test confluence scoring
        logger.info("🧪 Testing confluence scoring...")
        test_signal = {
            'signal_type': 'BULL',
            'trend_alignment': 'bullish',
            'trend_strength_score': 0.7
        }
        
        confluence_score = mt_analyzer.get_confluence_score(test_signal)
        logger.info(f"✅ Confluence score calculation: {confluence_score:.3f}")
        
        # Test signal confluence analysis
        if hasattr(mt_analyzer, 'analyze_signal_confluence'):
            logger.info("🧪 Testing signal confluence analysis...")
            confluence_result = mt_analyzer.analyze_signal_confluence(
                df_5m, test_pair, spread_pips=1.5, timeframe='5m'
            )
            
            logger.info(f"✅ Signal confluence analysis:")
            logger.info(f"   Dominant direction: {confluence_result['dominant_direction']}")
            logger.info(f"   Confluence score: {confluence_result['confluence_score']:.3f}")
            logger.info(f"   Agreement level: {confluence_result['agreement_level']}")
            logger.info(f"   Strategies tested: {confluence_result.get('strategies_tested', [])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MultiTimeframeAnalyzer validation failed: {e}")
        return False


def validate_signal_detector_integration():
    """Validate SignalDetector multi-timeframe integration"""
    
    logger.info("🔍 Validating SignalDetector integration...")
    
    try:
        # Initialize components
        db_manager = DatabaseManager(config.DATABASE_URL)
        signal_detector = SignalDetector(db_manager, config.USER_TIMEZONE)
        
        test_epic = 'CS.D.EURUSD.CEEM.IP'
        test_pair = 'EURUSD'
        
        # Test if detect_signals_multi_timeframe exists
        if hasattr(signal_detector, 'detect_signals_multi_timeframe'):
            logger.info("✅ detect_signals_multi_timeframe method exists")
            
            logger.info("🧪 Testing multi-timeframe signal detection...")
            signal = signal_detector.detect_signals_multi_timeframe(
                epic=test_epic,
                pair=test_pair,
                spread_pips=1.5,
                primary_timeframe='5m'
            )
            
            if signal:
                logger.info("✅ Multi-timeframe signal detection successful")
                logger.info(f"   Signal type: {signal['signal_type']}")
                logger.info(f"   Confidence: {signal.get('confidence_score', 0):.3f}")
                logger.info(f"   Confluence score: {signal.get('confluence_score', 0):.3f}")
                logger.info(f"   Multi-timeframe: {signal.get('multi_timeframe_analysis', False)}")
            else:
                logger.info("ℹ️ No multi-timeframe signal found (normal)")
        else:
            logger.error("❌ detect_signals_multi_timeframe method not found")
            return False
        
        # Test analyze_signal_confluence method
        if hasattr(signal_detector, 'analyze_signal_confluence'):
            logger.info("✅ analyze_signal_confluence method exists")
            
            # Get test data
            df = signal_detector.data_fetcher.get_enhanced_data(
                test_epic, test_pair, timeframe='5m', lookback_hours=48
            )
            
            if df is not None and len(df) > 50:
                logger.info("🧪 Testing signal confluence analysis...")
                confluence = signal_detector.analyze_signal_confluence(
                    df, test_pair, spread_pips=1.5, timeframe='5m'
                )
                
                logger.info(f"✅ Signal confluence analysis:")
                logger.info(f"   Strategies tested: {confluence.get('strategies_tested', [])}")
                logger.info(f"   Bull signals: {len(confluence.get('bull_signals', []))}")
                logger.info(f"   Bear signals: {len(confluence.get('bear_signals', []))}")
                logger.info(f"   Confluence score: {confluence.get('confluence_score', 0):.3f}")
            else:
                logger.error("❌ Insufficient data for confluence testing")
                return False
        else:
            logger.error("❌ analyze_signal_confluence method not found")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SignalDetector integration validation failed: {e}")
        return False


def validate_configuration():
    """Validate multi-timeframe configuration settings"""
    
    logger.info("🔍 Validating configuration...")
    
    # Check required config settings
    required_configs = [
        'ENABLE_MULTI_TIMEFRAME_ANALYSIS',
        'MIN_CONFLUENCE_SCORE',
        'CONFLUENCE_TIMEFRAMES',
        'TIMEFRAME_WEIGHTS'
    ]
    
    missing_configs = []
    for config_name in required_configs:
        if not hasattr(config, config_name):
            missing_configs.append(config_name)
        else:
            value = getattr(config, config_name)
            logger.info(f"✅ {config_name}: {value}")
    
    if missing_configs:
        logger.warning(f"⚠️ Missing config settings: {missing_configs}")
        logger.info("Add these to your config.py:")
        for missing in missing_configs:
            if missing == 'ENABLE_MULTI_TIMEFRAME_ANALYSIS':
                logger.info(f"   {missing} = True")
            elif missing == 'MIN_CONFLUENCE_SCORE':
                logger.info(f"   {missing} = 0.3")
            elif missing == 'CONFLUENCE_TIMEFRAMES':
                logger.info(f"   {missing} = ['5m', '15m', '1h']")
            elif missing == 'TIMEFRAME_WEIGHTS':
                logger.info(f"   {missing} = {{'5m': 0.2, '15m': 0.4, '1h': 0.4}}")
        
        return False
    
    return True


def run_complete_validation():
    """Run complete validation of multi-timeframe implementation"""
    
    logger.info("🚀 Starting complete multi-timeframe validation...")
    
    results = {
        'configuration': False,
        'data_fetcher': False,
        'multi_timeframe_analyzer': False,
        'signal_detector_integration': False
    }
    
    # Run all validations
    results['configuration'] = validate_configuration()
    results['data_fetcher'] = validate_data_fetcher_methods()
    results['multi_timeframe_analyzer'] = validate_multi_timeframe_analyzer()
    results['signal_detector_integration'] = validate_signal_detector_integration()
    
    # Summary
    logger.info("📊 Validation Summary:")
    all_passed = True
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {component}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 All validations passed! Multi-timeframe analysis is ready for use.")
    else:
        logger.error("❌ Some validations failed. Check the errors above and fix before using.")
    
    return all_passed


if __name__ == "__main__":
    run_complete_validation()