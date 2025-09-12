#!/usr/bin/env python3
"""
Test script to verify enhanced decision logging for MACD signals
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict

# Set up logging to capture all levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_logging_scenarios():
    """Test various scenarios to verify enhanced logging works"""
    
    logger.info("=" * 80)
    logger.info("TESTING ENHANCED DECISION LOGGING")
    logger.info("=" * 80)
    
    # Import required modules
    try:
        from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
        from core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
        import config
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Initialize components
    forex_optimizer = MACDForexOptimizer(logger=logger)
    signal_detector = MACDSignalDetector(
        logger=logger,
        forex_optimizer=forex_optimizer
    )
    
    # Test scenarios
    scenarios = [
        {
            'name': 'USDJPY - Should be REJECTED (like your alert)',
            'epic': 'CS.D.USDJPY.MINI.IP',
            'histogram_change': 0.00306,  # Just above threshold but below buffer
            'expected_decision': 'REJECTED'
        },
        {
            'name': 'USDJPY - Should be APPROVED (well above threshold)',
            'epic': 'CS.D.USDJPY.MINI.IP',
            'histogram_change': 0.0035,  # Well above threshold + buffer
            'expected_decision': 'APPROVED'
        },
        {
            'name': 'EURUSD - Should be REJECTED (at boundary)',
            'epic': 'CS.D.EURUSD.MINI.IP',
            'histogram_change': 0.00005,  # Exactly at threshold
            'expected_decision': 'REJECTED'
        },
        {
            'name': 'EURUSD - Should be APPROVED (above buffer)',
            'epic': 'CS.D.EURUSD.MINI.IP',
            'histogram_change': 0.00006,  # Above threshold + buffer
            'expected_decision': 'APPROVED'
        }
    ]
    
    logger.info("\\n" + "=" * 60)
    logger.info("RUNNING LOGGING TEST SCENARIOS")
    logger.info("=" * 60)
    
    for scenario in scenarios:
        logger.info(f"\\n{'='*50}")
        logger.info(f"SCENARIO: {scenario['name']}")
        logger.info(f"{'='*50}")
        
        # Test threshold validation (this should trigger our enhanced logging)
        result = signal_detector._validate_normalized_crossover_threshold_accurate(
            histogram_change=scenario['histogram_change'],
            signal_type='BULL',
            epic=scenario['epic'],
            df_enhanced=None,  # Will use fallback method
            forex_optimizer=forex_optimizer
        )
        
        actual_decision = 'APPROVED' if result['is_valid'] else 'REJECTED'
        logger.info(f"Expected: {scenario['expected_decision']}, Actual: {actual_decision}")
        
        if actual_decision == scenario['expected_decision']:
            logger.info("✅ Test result matches expectation")
        else:
            logger.error("❌ Test result differs from expectation")
    
    # Test momentum confirmation logging
    logger.info("\\n" + "=" * 60)
    logger.info("TESTING MOMENTUM CONFIRMATION LOGGING")
    logger.info("=" * 60)
    
    # Create test data for momentum confirmation
    dates = pd.date_range(start='2025-08-27 10:00', periods=5, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [147.50] * 5,
        'high': [147.60] * 5,
        'low': [147.40] * 5,
        'close': [147.55] * 5,
        'mid_c': [147.55] * 5,
        'macd_histogram': [0.002, 0.0025, 0.003, 0.0032, 0.0035],  # Building momentum
        'macd_line': [0.05] * 5,
        'macd_signal_line': [0.048] * 5,
        'ema_200': [147.40] * 5
    })
    df.set_index('timestamp', inplace=True)
    
    # Test momentum confirmation check
    latest = df.iloc[-1]
    momentum_signal = signal_detector.check_momentum_confirmation_signals(
        epic='CS.D.USDJPY.MINI.IP',
        timeframe='5m',
        df_enhanced=df,
        latest=latest,
        forex_optimizer=forex_optimizer
    )
    
    if momentum_signal:
        logger.info("✅ Momentum confirmation signal generated (with detailed logging)")
    else:
        logger.info("ℹ️ No momentum confirmation signal (this is expected for this test)")
    
    logger.info("\\n" + "=" * 80)
    logger.info("LOGGING TEST COMPLETED")
    logger.info("=" * 80)
    logger.info("Check the output above to verify that you see detailed decision logging including:")
    logger.info("- 🎯 [SIGNAL APPROVED/REJECTED] messages")
    logger.info("- Histogram change values")
    logger.info("- Original vs effective thresholds")
    logger.info("- Buffer multiplier information")
    logger.info("- Detailed comparison results")
    
    return True

if __name__ == "__main__":
    test_logging_scenarios()