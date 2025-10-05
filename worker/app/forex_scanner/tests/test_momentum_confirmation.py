#!/usr/bin/env python3
"""
Test script for MACD momentum confirmation system
Tests both immediate signals and delayed momentum confirmation
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

def create_test_data() -> pd.DataFrame:
    """Create test data with MACD crossover scenario"""
    
    # Create 20 bars of test data
    dates = pd.date_range(start='2025-08-26 10:00', periods=20, freq='5min')
    
    # Simulate a bullish crossover at bar 10 that starts weak but gains momentum
    macd_histogram_values = [
        -0.00010, -0.00008, -0.00006, -0.00004, -0.00003,  # Bars 0-4: Bearish, getting weaker
        -0.00002, -0.00001, -0.000005, 0.000005, 0.00001,  # Bars 5-9: Approaching crossover
        0.00002,   # Bar 10: BULLISH CROSSOVER but weak (below 0.00005 threshold)
        0.00003,   # Bar 11: Still building
        0.00006,   # Bar 12: NOW exceeds EURUSD threshold (0.00005)! -> Should trigger momentum confirmation
        0.00008,   # Bar 13: Stronger momentum
        0.00012,   # Bar 14: Very strong momentum -> Should trigger continuation
        0.00010, 0.00008, 0.00006, 0.00004, 0.00002       # Bars 15-19: Weakening
    ]
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [1.0850] * 20,
        'high': [1.0860] * 20,
        'low': [1.0840] * 20,
        'close': [1.0855] * 20,
        'mid_o': [1.0850] * 20,
        'mid_h': [1.0860] * 20,
        'mid_l': [1.0840] * 20,
        'mid_c': [1.0855] * 20,
        'volume': [1000] * 20,
        'macd_histogram': macd_histogram_values,
        'macd_line': [0.00001] * 20,
        'macd_signal_line': [-0.00001] * 20,
        'ema_200': [1.0840] * 20
    })
    
    df.set_index('timestamp', inplace=True)
    return df

def test_momentum_confirmation():
    """Test the momentum confirmation system"""
    
    logger.info("=" * 80)
    logger.info("TESTING MACD MOMENTUM CONFIRMATION SYSTEM")
    logger.info("=" * 80)
    
    # Import required modules
    try:
        from core.strategies.helpers.macd_crossover_tracker import MACDCrossoverTracker
        from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
        from core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
        from core.strategies.macd_strategy import MACDStrategy
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Create test data
    df = create_test_data()
    epic = 'CS.D.EURUSD.CEEM.IP'
    timeframe = '5m'
    
    logger.info(f"Created test data with {len(df)} bars")
    logger.info("Scenario: Weak crossover at bar 10, momentum builds at bar 12")
    
    # Initialize components
    forex_optimizer = MACDForexOptimizer(logger=logger)
    signal_detector = MACDSignalDetector(
        logger=logger,
        forex_optimizer=forex_optimizer
    )
    
    # Verify EURUSD threshold
    threshold = forex_optimizer.get_macd_threshold_for_epic(epic)
    logger.info(f"EURUSD threshold: {threshold:.6f}")
    
    # Test scenario
    signals_detected = []
    
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING SIMULATION (bar by bar)")
    logger.info("=" * 60)
    
    for i in range(1, len(df)):
        current_bar = df.iloc[i]
        previous_bar = df.iloc[i-1]
        
        logger.info(f"\nBar {i}: {current_bar.name}")
        logger.info(f"  Histogram: {current_bar['macd_histogram']:.6f}")
        logger.info(f"  Previous:  {previous_bar['macd_histogram']:.6f}")
        
        # Check for crossover signals (immediate)
        if hasattr(signal_detector, 'enhanced_macd_signal_detection'):
            signal = signal_detector.enhanced_macd_signal_detection(
                latest=current_bar,
                previous=previous_bar,
                epic=epic,
                timeframe=timeframe,
                df_enhanced=df.iloc[:i+1],
                forex_optimizer=forex_optimizer
            )
            
            if signal:
                signals_detected.append({
                    'bar': i,
                    'type': 'immediate',
                    'signal': signal
                })
                logger.info(f"  -> ✅ IMMEDIATE SIGNAL: {signal.get('signal_type')} ({signal.get('trigger_reason')})")
        
        # Check for momentum confirmation signals
        momentum_signal = signal_detector.check_momentum_confirmation_signals(
            epic=epic,
            timeframe=timeframe,
            df_enhanced=df.iloc[:i+1],
            latest=current_bar,
            forex_optimizer=forex_optimizer
        )
        
        if momentum_signal:
            signals_detected.append({
                'bar': i,
                'type': 'momentum_confirmation',
                'signal': momentum_signal
            })
            logger.info(f"  -> 🎯 MOMENTUM SIGNAL: {momentum_signal.get('signal_type')} ({momentum_signal.get('trigger_reason')})")
            
            # Log momentum details
            if 'momentum_confirmation' in momentum_signal:
                mc = momentum_signal['momentum_confirmation']
                logger.info(f"     Initial histogram: {mc['initial_histogram']:.6f}")
                logger.info(f"     Bars since cross: {mc['bars_since_crossover']}")
        
        # Get tracker stats
        if hasattr(signal_detector, 'get_tracker_statistics'):
            stats = signal_detector.get_tracker_statistics()
            if stats and stats.get('active_crossovers', 0) > 0:
                logger.info(f"  -> 📌 Tracking {stats['active_crossovers']} crossover(s)")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    immediate_signals = [s for s in signals_detected if s['type'] == 'immediate']
    momentum_signals = [s for s in signals_detected if s['type'] == 'momentum_confirmation']
    continuation_signals = [s for s in signals_detected if s['type'] == 'momentum_continuation']
    
    logger.info(f"Total signals detected: {len(signals_detected)}")
    logger.info(f"  - Immediate signals: {len(immediate_signals)}")
    logger.info(f"  - Momentum confirmations: {len(momentum_signals)}")
    logger.info(f"  - Momentum continuations: {len(continuation_signals)}")
    
    # Detailed results
    for signal_info in signals_detected:
        bar = signal_info['bar']
        signal_type = signal_info['type']
        signal = signal_info['signal']
        
        logger.info(f"\nBar {bar} - {signal_type.upper()}:")
        logger.info(f"  Signal type: {signal.get('signal_type')}")
        logger.info(f"  Trigger: {signal.get('trigger_reason')}")
        logger.info(f"  Histogram: {signal.get('macd_histogram', 0):.6f}")
        logger.info(f"  Confidence: {signal.get('confidence', 0):.2f}")
    
    # Validation checks
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 60)
    
    success = True
    
    # Check 1: Should NOT have immediate signal at bar 10 (weak crossover)
    bar_10_immediate = [s for s in immediate_signals if s['bar'] == 10]
    if bar_10_immediate:
        logger.error("❌ FAIL: Bar 10 should NOT trigger immediate signal (weak crossover)")
        success = False
    else:
        logger.info("✅ PASS: Bar 10 correctly did not trigger immediate signal")
    
    # Check 2: Should have momentum confirmation around bar 12
    momentum_bars = [s['bar'] for s in momentum_signals]
    if 12 in momentum_bars or 13 in momentum_bars:
        logger.info("✅ PASS: Momentum confirmation signal detected around bar 12-13")
    else:
        logger.error("❌ FAIL: No momentum confirmation signal around bar 12-13")
        success = False
    
    # Check 3: Should have at least one signal total
    if len(signals_detected) == 0:
        logger.error("❌ FAIL: No signals detected at all")
        success = False
    else:
        logger.info(f"✅ PASS: {len(signals_detected)} total signals detected")
    
    # Get final tracker statistics
    if hasattr(signal_detector, 'get_tracker_statistics'):
        final_stats = signal_detector.get_tracker_statistics()
        if final_stats:
            logger.info(f"\nTracker Statistics:")
            for key, value in final_stats.items():
                logger.info(f"  {key}: {value}")
    
    if success:
        logger.info("\n🎉 ALL VALIDATION CHECKS PASSED!")
        logger.info("The momentum confirmation system is working correctly.")
    else:
        logger.error("\n⚠️ SOME VALIDATION CHECKS FAILED!")
        logger.error("The momentum confirmation system needs adjustment.")
    
    return success

def test_weak_crossover_scenario():
    """Test specific scenario: weak crossover that never develops momentum"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TESTING WEAK CROSSOVER (NO CONFIRMATION)")
    logger.info("=" * 80)
    
    # Create scenario where crossover happens but never gains momentum
    dates = pd.date_range(start='2025-08-26 11:00', periods=10, freq='5min')
    
    # Weak crossover that stays weak
    histogram_values = [
        -0.00002, -0.00001, 0.00001,  # Bars 0-2: Crossover happens
        0.00002, 0.00002, 0.00001,    # Bars 3-5: Stays weak
        0.000005, -0.000005, -0.00001, -0.00002  # Bars 6-9: Reverses
    ]
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [1.0850] * 10,
        'high': [1.0860] * 10, 
        'low': [1.0840] * 10,
        'close': [1.0855] * 10,
        'mid_c': [1.0855] * 10,
        'macd_histogram': histogram_values,
        'macd_line': [0.00001] * 10,
        'macd_signal_line': [-0.00001] * 10,
        'ema_200': [1.0840] * 10
    })
    df.set_index('timestamp', inplace=True)
    
    # Test the scenario
    from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
    from core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
    
    forex_optimizer = MACDForexOptimizer(logger=logger)
    signal_detector = MACDSignalDetector(logger=logger, forex_optimizer=forex_optimizer)
    
    epic = 'CS.D.EURUSD.CEEM.IP'
    timeframe = '5m'
    
    signals_detected = []
    
    for i in range(1, len(df)):
        current_bar = df.iloc[i]
        
        # Check for momentum confirmation
        momentum_signal = signal_detector.check_momentum_confirmation_signals(
            epic=epic,
            timeframe=timeframe,
            df_enhanced=df.iloc[:i+1],
            latest=current_bar,
            forex_optimizer=forex_optimizer
        )
        
        if momentum_signal:
            signals_detected.append(momentum_signal)
            logger.info(f"Bar {i}: Signal detected - {momentum_signal.get('trigger_reason')}")
    
    # Should have no signals
    if len(signals_detected) == 0:
        logger.info("✅ CORRECT: No signals generated for weak crossover that doesn't develop")
        return True
    else:
        logger.error(f"❌ INCORRECT: {len(signals_detected)} signals generated for weak crossover")
        return False

if __name__ == "__main__":
    success1 = test_momentum_confirmation()
    success2 = test_weak_crossover_scenario()
    
    overall_success = success1 and success2
    logger.info(f"\n{'='*80}")
    logger.info(f"OVERALL TEST RESULT: {'PASS' if overall_success else 'FAIL'}")
    logger.info(f"{'='*80}")
    
    sys.exit(0 if overall_success else 1)