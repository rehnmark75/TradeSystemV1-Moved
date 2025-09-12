#!/usr/bin/env python3
"""
Test script for EMA200 distance validation
Tests that signals are rejected when price is too close to EMA200
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

def create_test_scenarios() -> List[Dict]:
    """Create test scenarios for EMA200 distance validation"""
    
    scenarios = []
    
    # Scenario 1: Price too close to EMA200 (should reject)
    scenarios.append({
        'name': 'Price 2 pips above EMA200 (too close)',
        'epic': 'CS.D.EURUSD.MINI.IP',
        'current_price': 1.0852,  # 2 pips above EMA200
        'ema_200': 1.0850,
        'signal_type': 'BULL',
        'macd_histogram': 0.00006,  # Above threshold but not strong
        'expected_result': False,  # Should reject (min 5 pips for EURUSD)
        'expected_distance': 2.0
    })
    
    # Scenario 2: Price at minimum distance (should pass)
    scenarios.append({
        'name': 'Price 5 pips above EMA200 (at minimum)',
        'epic': 'CS.D.EURUSD.MINI.IP',
        'current_price': 1.0855,  # 5 pips above EMA200
        'ema_200': 1.0850,
        'signal_type': 'BULL',
        'macd_histogram': 0.00006,
        'expected_result': True,  # Should pass (exactly 5 pips)
        'expected_distance': 5.0
    })
    
    # Scenario 3: Strong momentum reduces distance requirement
    scenarios.append({
        'name': 'Price 4 pips above EMA200 with strong momentum',
        'epic': 'CS.D.EURUSD.MINI.IP',
        'current_price': 1.0854,  # 4 pips above EMA200
        'ema_200': 1.0850,
        'signal_type': 'BULL',
        'macd_histogram': 0.00012,  # Very strong momentum (>2x threshold)
        'expected_result': True,  # Should pass (4 pips > 5*0.8=4.0 with momentum)
        'expected_distance': 4.0
    })
    
    # Scenario 4: GBPUSD requires more distance
    scenarios.append({
        'name': 'GBPUSD 6 pips below EMA200 (too close)',
        'epic': 'CS.D.GBPUSD.MINI.IP',
        'current_price': 1.2644,  # 6 pips below EMA200
        'ema_200': 1.2650,
        'signal_type': 'BEAR',
        'macd_histogram': -0.00008,
        'expected_result': False,  # Should reject (min 8 pips for GBPUSD)
        'expected_distance': 6.0
    })
    
    # Scenario 5: USDJPY with different scale
    scenarios.append({
        'name': 'USDJPY 30 pips above EMA200 (too close)',
        'epic': 'CS.D.USDJPY.MINI.IP',
        'current_price': 145.30,  # 30 pips above EMA200
        'ema_200': 145.00,
        'signal_type': 'BULL',
        'macd_histogram': 0.003,
        'expected_result': False,  # Should reject (min 50 pips for USDJPY)
        'expected_distance': 30.0
    })
    
    # Scenario 6: USDJPY at minimum distance
    scenarios.append({
        'name': 'USDJPY 60 pips above EMA200 (passes)',
        'epic': 'CS.D.USDJPY.MINI.IP',
        'current_price': 145.60,  # 60 pips above EMA200
        'ema_200': 145.00,
        'signal_type': 'BULL',
        'macd_histogram': 0.003,
        'expected_result': True,  # Should pass (60 > 50 pips minimum)
        'expected_distance': 60.0
    })
    
    return scenarios

def test_ema200_distance_validation():
    """Test the EMA200 distance validation system"""
    
    logger.info("=" * 80)
    logger.info("TESTING EMA200 DISTANCE VALIDATION")
    logger.info("=" * 80)
    
    # Import required modules
    try:
        from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
        from core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
        import config
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Verify configuration is enabled
    distance_enabled = getattr(config, 'EMA200_MIN_DISTANCE_ENABLED', False)
    distance_config = getattr(config, 'EMA200_MIN_DISTANCE_PIPS', {})
    momentum_multiplier = getattr(config, 'EMA200_DISTANCE_MOMENTUM_MULTIPLIER', 0.8)
    
    logger.info(f"Configuration:")
    logger.info(f"  Distance validation enabled: {distance_enabled}")
    logger.info(f"  Momentum multiplier: {momentum_multiplier}")
    logger.info(f"  Configured distances: {distance_config}")
    
    if not distance_enabled:
        logger.warning("⚠️ EMA200_MIN_DISTANCE_ENABLED is False - tests may not work correctly")
    
    # Initialize components
    forex_optimizer = MACDForexOptimizer(logger=logger)
    signal_detector = MACDSignalDetector(
        logger=logger,
        forex_optimizer=forex_optimizer
    )
    
    # Get test scenarios
    scenarios = create_test_scenarios()
    
    # Track results
    passed_tests = 0
    failed_tests = 0
    
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING TEST SCENARIOS")
    logger.info("=" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\nScenario {i}: {scenario['name']}")
        logger.info(f"  Epic: {scenario['epic']}")
        logger.info(f"  Price: {scenario['current_price']:.5f}")
        logger.info(f"  EMA200: {scenario['ema_200']:.5f}")
        logger.info(f"  Signal Type: {scenario['signal_type']}")
        logger.info(f"  MACD Histogram: {scenario['macd_histogram']:.6f}")
        logger.info(f"  Expected Distance: {scenario['expected_distance']:.1f} pips")
        logger.info(f"  Expected Result: {'PASS' if scenario['expected_result'] else 'REJECT'}")
        
        # Test the validation
        result = signal_detector.validate_ema_position_enhanced(
            current_price=scenario['current_price'],
            ema_200=scenario['ema_200'],
            signal_type=scenario['signal_type'],
            epic=scenario['epic'],
            macd_histogram=scenario['macd_histogram'],
            require_alignment=True,
            filter_mode='standard'
        )
        
        # Check if result matches expectation
        if result == scenario['expected_result']:
            logger.info(f"  ✅ TEST PASSED: Got expected result {'PASS' if result else 'REJECT'}")
            passed_tests += 1
        else:
            logger.error(f"  ❌ TEST FAILED: Expected {'PASS' if scenario['expected_result'] else 'REJECT'}, got {'PASS' if result else 'REJECT'}")
            failed_tests += 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tests: {len(scenarios)}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    
    # Test the helper method directly
    logger.info("\n" + "=" * 60)
    logger.info("TESTING HELPER METHOD")
    logger.info("=" * 60)
    
    test_pairs = [
        ('CS.D.EURUSD.MINI.IP', 0.00005, 5.0),   # Normal threshold
        ('CS.D.EURUSD.MINI.IP', 0.00012, 4.0),   # Strong momentum (5.0 * 0.8)
        ('CS.D.GBPUSD.MINI.IP', 0.00008, 8.0),   # GBP higher requirement
        ('CS.D.USDJPY.MINI.IP', 0.003, 50.0),    # JPY different scale
    ]
    
    for epic, histogram, expected_distance in test_pairs:
        min_distance = signal_detector._get_ema200_min_distance(epic, 'BULL', histogram)
        logger.info(f"{epic}: min_distance={min_distance:.1f} (expected={expected_distance:.1f})")
        if abs(min_distance - expected_distance) < 0.1:
            logger.info(f"  ✅ Correct distance calculated")
        else:
            logger.error(f"  ❌ Incorrect distance: got {min_distance:.1f}, expected {expected_distance:.1f}")
    
    # Overall success
    success = failed_tests == 0
    
    if success:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("The EMA200 distance validation is working correctly.")
    else:
        logger.error(f"\n⚠️ {failed_tests} TESTS FAILED!")
        logger.error("The EMA200 distance validation needs adjustment.")
    
    return success

if __name__ == "__main__":
    success = test_ema200_distance_validation()
    sys.exit(0 if success else 1)