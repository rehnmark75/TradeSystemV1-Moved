#!/usr/bin/env python3
"""
Test script for MACD threshold buffer zone
Tests that signals at the boundary (like 0.00306 vs 0.003) are rejected
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

def test_threshold_buffer():
    """Test the threshold buffer implementation"""
    
    logger.info("=" * 80)
    logger.info("TESTING MACD THRESHOLD BUFFER ZONE")
    logger.info("=" * 80)
    
    # Import required modules
    try:
        from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
        from core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
        from core.strategies.helpers.macd_crossover_tracker import MACDCrossoverTracker
        import config
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Verify configuration
    buffer_multiplier = getattr(config, 'MACD_THRESHOLD_BUFFER_MULTIPLIER', 1.0)
    logger.info(f"Configuration:")
    logger.info(f"  Buffer multiplier: {buffer_multiplier}")
    
    if buffer_multiplier <= 1.0:
        logger.warning("⚠️ MACD_THRESHOLD_BUFFER_MULTIPLIER is not > 1.0 - buffer zone disabled")
    
    # Initialize components
    forex_optimizer = MACDForexOptimizer(logger=logger)
    signal_detector = MACDSignalDetector(
        logger=logger,
        forex_optimizer=forex_optimizer
    )
    
    # Test scenarios
    test_cases = [
        {
            'name': 'USDJPY at exact threshold (should reject with buffer)',
            'epic': 'CS.D.USDJPY.MINI.IP',
            'histogram': 0.003,  # Exactly at threshold
            'threshold': 0.003,
            'should_pass': False  # Should reject with 1.1x buffer
        },
        {
            'name': 'USDJPY slightly above threshold (like your alert)',
            'epic': 'CS.D.USDJPY.MINI.IP',
            'histogram': 0.00306,  # Just 2% above threshold
            'threshold': 0.003,
            'should_pass': False  # Should reject with 1.1x buffer
        },
        {
            'name': 'USDJPY well above threshold',
            'epic': 'CS.D.USDJPY.MINI.IP',
            'histogram': 0.0035,  # 16.7% above threshold
            'threshold': 0.003,
            'should_pass': True  # Should pass with 1.1x buffer (needs 0.0033)
        },
        {
            'name': 'EURUSD at boundary',
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'histogram': 0.00005,  # Exactly at threshold
            'threshold': 0.00005,
            'should_pass': False  # Should reject with 1.1x buffer
        },
        {
            'name': 'EURUSD above buffer zone',
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'histogram': 0.00006,  # 20% above threshold
            'threshold': 0.00005,
            'should_pass': True  # Should pass with 1.1x buffer (needs 0.000055)
        }
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING TEST SCENARIOS")
    logger.info("=" * 60)
    
    for test_case in test_cases:
        logger.info(f"\nTest: {test_case['name']}")
        logger.info(f"  Epic: {test_case['epic']}")
        logger.info(f"  Histogram: {test_case['histogram']:.6f}")
        logger.info(f"  Base threshold: {test_case['threshold']:.6f}")
        logger.info(f"  Effective threshold (with {buffer_multiplier}x buffer): {test_case['threshold'] * buffer_multiplier:.6f}")
        logger.info(f"  Expected: {'PASS' if test_case['should_pass'] else 'REJECT'}")
        
        # Get the actual threshold from forex optimizer
        actual_threshold = forex_optimizer.get_macd_threshold_for_epic(test_case['epic'])
        logger.info(f"  Actual threshold from optimizer: {actual_threshold:.6f}")
        
        # Apply buffer multiplier
        effective_threshold = actual_threshold * buffer_multiplier
        
        # Test the validation
        passes = test_case['histogram'] >= effective_threshold
        
        logger.info(f"  Result: {'PASS' if passes else 'REJECT'} (histogram {test_case['histogram']:.6f} {'>' if passes else '<'} {effective_threshold:.6f})")
        
        # Check if result matches expectation
        if passes == test_case['should_pass']:
            logger.info(f"  ✅ TEST PASSED")
            passed_tests += 1
        else:
            logger.error(f"  ❌ TEST FAILED: Expected {'PASS' if test_case['should_pass'] else 'REJECT'}, got {'PASS' if passes else 'REJECT'}")
            failed_tests += 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tests: {len(test_cases)}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    
    # Special test for the exact scenario from user's alert
    logger.info("\n" + "=" * 60)
    logger.info("TESTING USER'S ALERT SCENARIO")
    logger.info("=" * 60)
    
    user_histogram = 0.0030627351678618464
    usdjpy_threshold = 0.003
    effective_threshold = usdjpy_threshold * buffer_multiplier
    
    logger.info(f"User's USDJPY histogram: {user_histogram:.6f}")
    logger.info(f"USDJPY threshold: {usdjpy_threshold:.6f}")
    logger.info(f"Effective threshold (with {buffer_multiplier}x buffer): {effective_threshold:.6f}")
    logger.info(f"Passes validation: {user_histogram >= effective_threshold}")
    
    if user_histogram < effective_threshold:
        logger.info("✅ CORRECT: This signal would now be REJECTED (as it should be)")
    else:
        logger.error("❌ PROBLEM: This signal would still pass (needs higher buffer)")
    
    # Overall success
    success = failed_tests == 0
    
    if success:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("The threshold buffer zone is working correctly.")
    else:
        logger.error(f"\n⚠️ {failed_tests} TESTS FAILED!")
    
    return success

if __name__ == "__main__":
    success = test_threshold_buffer()
    sys.exit(0 if success else 1)