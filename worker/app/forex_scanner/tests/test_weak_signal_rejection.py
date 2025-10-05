#!/usr/bin/env python3
"""
Test script to verify weak MACD signals are properly rejected
Specifically testing that 0.00002 EURUSD histogram is rejected
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_weak_signal_rejection():
    """Test that weak signals like 0.00002 EURUSD are properly rejected"""
    
    logger.info("=" * 80)
    logger.info("TESTING WEAK SIGNAL REJECTION")
    logger.info("=" * 80)
    
    # Import required modules
    from core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
    from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
    
    # Initialize components
    forex_optimizer = MACDForexOptimizer(logger=logger)
    signal_detector = MACDSignalDetector(
        logger=logger,
        forex_optimizer=forex_optimizer,
        validator=None,
        db_manager=None,
        data_fetcher=None
    )
    
    # Test cases with weak histograms that should be rejected
    test_cases = [
        {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'current_histogram': 0.00002,  # This was incorrectly passing through
            'previous_histogram': 0.00001,
            'expected': 'REJECT',
            'reason': 'Histogram too weak (0.00002 < 0.00005 threshold)'
        },
        {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'current_histogram': 0.00003,
            'previous_histogram': 0.00001,
            'expected': 'REJECT',
            'reason': 'Still below threshold (0.00003 < 0.00005)'
        },
        {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'current_histogram': 0.00006,
            'previous_histogram': 0.00001,
            'expected': 'PASS',
            'reason': 'Above threshold (0.00006 > 0.00005)'
        },
        {
            'epic': 'CS.D.GBPUSD.MINI.IP',
            'current_histogram': 0.00005,
            'previous_histogram': 0.00001,
            'expected': 'REJECT',
            'reason': 'Below GBPUSD threshold (0.00005 < 0.00008)'
        },
        {
            'epic': 'CS.D.USDJPY.MINI.IP',
            'current_histogram': 0.005,
            'previous_histogram': 0.001,
            'expected': 'REJECT',
            'reason': 'Below JPY threshold (0.005 < 0.008)'
        },
        {
            'epic': 'CS.D.USDJPY.MINI.IP',
            'current_histogram': 0.010,
            'previous_histogram': 0.001,
            'expected': 'PASS',
            'reason': 'Above JPY threshold (0.010 > 0.008)'
        }
    ]
    
    results = []
    
    for test in test_cases:
        epic = test['epic']
        current_hist = test['current_histogram']
        prev_hist = test['previous_histogram']
        
        logger.info(f"\nTesting {epic}:")
        logger.info(f"  Current histogram: {current_hist:.6f}")
        logger.info(f"  Previous histogram: {prev_hist:.6f}")
        logger.info(f"  Change: {abs(current_hist - prev_hist):.6f}")
        
        # Get threshold
        threshold = forex_optimizer.get_macd_threshold_for_epic(epic)
        logger.info(f"  Threshold: {threshold:.6f}")
        
        # Test threshold validation
        is_valid, threshold_used, reason = signal_detector._validate_normalized_macd_threshold(
            current_histogram=current_hist,
            previous_histogram=prev_hist,
            epic=epic,
            signal_type='BULL',
            df_enhanced=None,
            latest=None,
            forex_optimizer=forex_optimizer
        )
        
        # Check result
        if test['expected'] == 'REJECT':
            if is_valid:
                logger.error(f"  ❌ FAILED: Signal was ACCEPTED but should be REJECTED")
                logger.error(f"     Reason: {test['reason']}")
                results.append(('FAIL', epic, f"Should reject but accepted: {reason}"))
            else:
                logger.info(f"  ✅ PASSED: Signal correctly REJECTED")
                logger.info(f"     Reason: {reason}")
                results.append(('PASS', epic, f"Correctly rejected: {reason}"))
        else:  # expected == 'PASS'
            if is_valid:
                logger.info(f"  ✅ PASSED: Signal correctly ACCEPTED")
                logger.info(f"     Reason: {reason}")
                results.append(('PASS', epic, f"Correctly accepted: {reason}"))
            else:
                logger.error(f"  ❌ FAILED: Signal was REJECTED but should be ACCEPTED")
                logger.error(f"     Reason: {test['reason']}")
                results.append(('FAIL', epic, f"Should accept but rejected: {reason}"))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for r in results if r[0] == 'PASS')
    failed = sum(1 for r in results if r[0] == 'FAIL')
    
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("\n🎉 ALL TESTS PASSED! Weak signals are properly rejected.")
    else:
        logger.error("\n⚠️ SOME TESTS FAILED! Weak signals may still pass through.")
        logger.error("Failed tests:")
        for result in results:
            if result[0] == 'FAIL':
                logger.error(f"  - {result[1]}: {result[2]}")
    
    # Test specific case that was problematic
    logger.info("\n" + "=" * 80)
    logger.info("SPECIFIC TEST: 0.00002 EURUSD (the problematic case)")
    logger.info("=" * 80)
    
    epic = 'CS.D.EURUSD.CEEM.IP'
    weak_histogram = 0.00002
    
    threshold = forex_optimizer.get_macd_threshold_for_epic(epic)
    logger.info(f"EURUSD threshold: {threshold:.6f}")
    logger.info(f"Weak histogram: {weak_histogram:.6f}")
    
    if weak_histogram < threshold:
        logger.info(f"✅ CORRECT: {weak_histogram:.6f} < {threshold:.6f} - Signal would be REJECTED")
    else:
        logger.error(f"❌ PROBLEM: {weak_histogram:.6f} >= {threshold:.6f} - Signal would be ACCEPTED!")
    
    return failed == 0

if __name__ == "__main__":
    success = test_weak_signal_rejection()
    sys.exit(0 if success else 1)