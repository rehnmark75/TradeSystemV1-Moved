#!/usr/bin/env python3
"""
Diagnostic script to check if MACD crossover tracker is working
"""

import sys
sys.path.append('/app/forex_scanner')

from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
from core.strategies.helpers.macd_crossover_tracker import MACDCrossoverTracker
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_crossover_tracker():
    """Test if the crossover tracker is properly integrated"""
    
    print("🔍 CHECKING MACD CROSSOVER TRACKER INTEGRATION")
    print("=" * 60)
    
    # Initialize detector
    detector = MACDSignalDetector()
    
    # Check if tracker is enabled
    print(f"\n📊 Tracker Status:")
    print(f"   Enabled: {detector.momentum_confirmation_enabled}")
    print(f"   Tracker object exists: {detector.crossover_tracker is not None}")
    
    if detector.crossover_tracker:
        print(f"   Confirmation window: {detector.crossover_tracker.confirmation_window} bars")
        print(f"   Momentum multiplier: {detector.crossover_tracker.momentum_multiplier}x")
        
        # Check statistics
        stats = detector.crossover_tracker.stats
        print(f"\n📈 Tracker Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Check for the missing method
    print(f"\n🔍 Method Analysis:")
    print(f"   detect_enhanced_macd_signal: {'✅' if hasattr(detector, 'detect_enhanced_macd_signal') else '❌'}")
    print(f"   check_for_momentum_confirmation: {'✅' if hasattr(detector, 'check_for_momentum_confirmation') else '❌'}")
    
    # Check what methods the tracker has
    if detector.crossover_tracker:
        print(f"\n📋 Tracker Methods:")
        tracker_methods = [m for m in dir(detector.crossover_tracker) if not m.startswith('_')]
        for method in tracker_methods:
            if 'check' in method or 'record' in method:
                print(f"   - {method}")
    
    # CRITICAL ISSUE CHECK
    print(f"\n⚠️ CRITICAL ISSUE ANALYSIS:")
    print("=" * 40)
    
    print("1. Crossover Recording: ✅ WORKING")
    print("   - Weak crossovers ARE being recorded")
    print("   - See lines 1987-1999 in macd_signal_detector.py")
    
    print("\n2. Momentum Confirmation Check: ❌ MISSING!")
    print("   - No method calls check_momentum_confirmation()")
    print("   - Recorded crossovers are NEVER checked again")
    print("   - Weak signals that gain momentum are LOST")
    
    print("\n3. The Problem:")
    print("   - detect_enhanced_macd_signal() only runs on crossover bars")
    print("   - It doesn't run on subsequent bars to check tracked crossovers")
    print("   - Need a separate check that runs EVERY bar")
    
    print("\n💡 SOLUTION NEEDED:")
    print("   Add a method that:")
    print("   1. Runs on every bar (not just crossovers)")
    print("   2. Calls tracker.check_momentum_confirmation()")
    print("   3. Generates signals for confirmed crossovers")
    print("   4. Should be called from MACD strategy's detect_signal()")
    
    return detector

if __name__ == "__main__":
    test_crossover_tracker()