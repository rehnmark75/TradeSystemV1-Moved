#!/usr/bin/env python3
"""
Test script for ADX Trend Strength Filter (Phase 2)
Tests that signals are filtered based on ADX trend strength
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data_with_adx(adx_values: List[float], trend_strength: str) -> pd.DataFrame:
    """Create test data with specific ADX values"""
    
    num_bars = len(adx_values)
    dates = pd.date_range(start='2025-08-27 10:00', periods=num_bars, freq='5min')
    
    # Create price data that matches the trend strength
    if trend_strength == 'strong_uptrend':
        # Strong uptrend: prices rising consistently
        base_price = 1.0850
        prices = [base_price + (i * 0.0002) for i in range(num_bars)]
        highs = [p + 0.0005 for p in prices]
        lows = [p - 0.0003 for p in prices]
    elif trend_strength == 'ranging':
        # Ranging market: prices oscillating
        base_price = 1.0850
        prices = [base_price + 0.0001 * np.sin(i * 0.5) for i in range(num_bars)]
        highs = [p + 0.0002 for p in prices]
        lows = [p - 0.0002 for p in prices]
    else:
        # Default: neutral trend
        base_price = 1.0850
        prices = [base_price] * num_bars
        highs = [p + 0.0003 for p in prices]
        lows = [p - 0.0003 for p in prices]
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'mid_o': prices,
        'mid_h': highs,
        'mid_l': lows,
        'mid_c': prices,
        'volume': [1000] * num_bars,
        
        # MACD data - simulate a crossover signal
        'macd_histogram': [0.00008] * num_bars,  # Above EURUSD threshold
        'macd_line': [0.00012] * num_bars,
        'macd_signal_line': [0.00004] * num_bars,
        'ema_200': [1.0840] * num_bars,
        
        # ADX data
        'ADX': adx_values,
        'DI_plus': [20.0] * num_bars,
        'DI_minus': [15.0] * num_bars
    })
    
    df.set_index('timestamp', inplace=True)
    return df

def test_adx_filter():
    """Test the ADX filter functionality"""
    
    logger.info("=" * 80)
    logger.info("TESTING ADX TREND STRENGTH FILTER (PHASE 2)")
    logger.info("=" * 80)
    
    # Import required modules
    try:
        from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
        from core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
        from core.strategies.helpers.adx_calculator import ADXCalculator
        import config
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Verify configuration
    adx_enabled = getattr(config, 'ADX_FILTER_ENABLED', False)
    adx_mode = getattr(config, 'ADX_FILTER_MODE', 'moderate')
    adx_thresholds = getattr(config, 'ADX_THRESHOLDS', {})
    
    logger.info(f"Configuration:")
    logger.info(f"  ADX filter enabled: {adx_enabled}")
    logger.info(f"  ADX filter mode: {adx_mode}")
    logger.info(f"  ADX thresholds: {adx_thresholds}")
    
    if not adx_enabled:
        logger.warning("⚠️ ADX_FILTER_ENABLED is False - tests may not work correctly")
    
    # Initialize components
    forex_optimizer = MACDForexOptimizer(logger=logger)
    signal_detector = MACDSignalDetector(
        logger=logger,
        forex_optimizer=forex_optimizer
    )
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Strong trending market (ADX 30)',
            'adx_values': [30.0] * 20,
            'trend_strength': 'strong_uptrend',
            'expected_result': True,  # Should allow signals
            'expected_strength': 'STRONG'
        },
        {
            'name': 'Moderate trending market (ADX 22)',
            'adx_values': [22.0] * 20,
            'trend_strength': 'strong_uptrend',
            'expected_result': True if adx_mode in ['moderate', 'permissive'] else False,
            'expected_strength': 'MODERATE'
        },
        {
            'name': 'Weak trend/ranging market (ADX 18)',
            'adx_values': [18.0] * 20,
            'trend_strength': 'ranging',
            'expected_result': True if adx_mode == 'permissive' else False,
            'expected_strength': 'WEAK'
        },
        {
            'name': 'Very weak trend (ADX 12)',
            'adx_values': [12.0] * 20,
            'trend_strength': 'ranging',
            'expected_result': False,  # Should reject in all modes
            'expected_strength': 'VERY_WEAK'
        },
        {
            'name': 'ADX building strength (15 to 26)',
            'adx_values': list(np.linspace(15, 26, 20)),
            'trend_strength': 'strong_uptrend',
            'expected_result': True,  # Latest ADX (26) is strong
            'expected_strength': 'STRONG'
        }
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    logger.info("\\n" + "=" * 60)
    logger.info("RUNNING ADX FILTER TEST SCENARIOS")
    logger.info("=" * 60)
    
    for scenario in test_scenarios:
        logger.info(f"\\n{'='*50}")
        logger.info(f"SCENARIO: {scenario['name']}")
        logger.info(f"{'='*50}")
        
        # Create test data
        df = create_test_data_with_adx(scenario['adx_values'], scenario['trend_strength'])
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        logger.info(f"  ADX value: {latest['ADX']:.1f}")
        logger.info(f"  Expected strength: {scenario['expected_strength']}")
        logger.info(f"  Expected result: {'ALLOW' if scenario['expected_result'] else 'REJECT'}")
        
        # Test ADX validation directly first
        adx_validation = signal_detector._validate_adx_trend_strength(
            df_enhanced=df,
            epic='CS.D.EURUSD.CEEM.IP',
            signal_type='BULL'
        )
        
        logger.info(f"  ADX validation result:")
        logger.info(f"    Is valid: {adx_validation['is_valid']}")
        logger.info(f"    Trend strength: {adx_validation['trend_strength']}")
        logger.info(f"    Reason: {adx_validation['reason']}")
        
        # Check if result matches expectation
        if adx_validation['is_valid'] == scenario['expected_result']:
            logger.info(f"  ✅ TEST PASSED: Got expected result")
            passed_tests += 1
        else:
            logger.error(f"  ❌ TEST FAILED: Expected {'ALLOW' if scenario['expected_result'] else 'REJECT'}, got {'ALLOW' if adx_validation['is_valid'] else 'REJECT'}")
            failed_tests += 1
        
        # Test full signal detection with ADX filter
        logger.info(f"  Testing full signal detection...")
        
        try:
            signal = signal_detector.detect_enhanced_macd_signal(
                latest=latest,
                previous=previous,
                epic='CS.D.EURUSD.CEEM.IP',
                timeframe='5m',
                df_enhanced=df,
                forex_optimizer=forex_optimizer
            )
            
            if signal is not None and scenario['expected_result']:
                logger.info(f"    ✅ Signal generated (as expected)")
                if 'adx_validation' in signal:
                    logger.info(f"    ADX data in signal: {signal['adx_validation']['trend_strength']}")
            elif signal is None and not scenario['expected_result']:
                logger.info(f"    ✅ Signal blocked by ADX filter (as expected)")
            elif signal is not None and not scenario['expected_result']:
                logger.error(f"    ❌ Signal should have been blocked by ADX filter")
                failed_tests += 1
                passed_tests -= 1  # Adjust counts
            elif signal is None and scenario['expected_result']:
                logger.error(f"    ❌ Signal should have been allowed through ADX filter")
                failed_tests += 1
                passed_tests -= 1  # Adjust counts
                
        except Exception as e:
            logger.error(f"    ⚠️ Error during signal detection: {e}")
    
    # Summary
    logger.info("\\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tests: {len(test_scenarios)}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    
    # Overall success
    success = failed_tests == 0
    
    if success:
        logger.info("\\n🎉 ALL ADX FILTER TESTS PASSED!")
        logger.info("The ADX trend strength filter is working correctly.")
    else:
        logger.error(f"\\n⚠️ {failed_tests} ADX FILTER TESTS FAILED!")
        logger.error("The ADX trend strength filter needs adjustment.")
    
    return success

def test_adx_configuration():
    """Test ADX filter configuration options"""
    
    logger.info("\\n" + "=" * 60)
    logger.info("TESTING ADX CONFIGURATION")
    logger.info("=" * 60)
    
    try:
        from core.strategies.helpers.adx_calculator import ADXCalculator
        import config
        
        adx_calc = ADXCalculator(period=14, logger=logger)
        
        # Test different filter modes
        modes = ['strict', 'moderate', 'permissive', 'disabled']
        adx_value = 22.0  # Moderate trend
        
        for mode in modes:
            result = adx_calc.validate_adx_signal(
                adx_value=adx_value,
                epic='CS.D.EURUSD.CEEM.IP',
                filter_mode=mode
            )
            
            logger.info(f"Mode '{mode}': ADX {adx_value} → {'ALLOW' if result['is_valid'] else 'REJECT'}")
            logger.info(f"  Reason: {result['reason']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_adx_filter()
    success2 = test_adx_configuration()
    
    overall_success = success1 and success2
    logger.info(f"\\n{'='*80}")
    logger.info(f"OVERALL TEST RESULT: {'PASS' if overall_success else 'FAIL'}")
    logger.info(f"{'='*80}")
    
    sys.exit(0 if overall_success else 1)