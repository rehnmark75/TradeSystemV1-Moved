#!/usr/bin/env python3
"""
Simple threshold check without heavy dependencies
"""
import sys
import os

# Add path
sys.path.append('/app/forex_scanner')

# Try importing just what we need
try:
    # Import the forex optimizer thresholds directly
    from core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
    import logging
    
    # Simple logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = MACDForexOptimizer(logger=logger)
    
    # Check the hardcoded thresholds
    epic = 'CS.D.EURUSD.MINI.IP'
    
    print("=== Direct Threshold Check ===")
    print(f"Hardcoded thresholds in forex_optimizer:")
    print(f"  {epic}: {optimizer.forex_macd_thresholds.get(epic, 'NOT_FOUND')}")
    print()
    
    # Try the method
    try:
        # Check session first
        session = optimizer.get_current_market_session()
        multiplier = optimizer.session_multipliers.get(session, 1.0)
        print(f"Current session: {session}")
        print(f"Session multiplier: {multiplier}")
        print()
        
        threshold = optimizer.get_macd_threshold_for_epic(epic)
        print(f"get_macd_threshold_for_epic('{epic}'): {threshold}")
        print(f"Expected: 0.00005")
        print(f"Actual:   {threshold:.8f}")
        
        # Calculate expected with session
        base = 0.00005
        expected_with_session = base * multiplier
        print(f"Expected with session ({session}): {expected_with_session:.8f}")
        
        if abs(threshold - expected_with_session) < 0.000001:
            print("✅ Correct threshold with session multiplier")
        elif abs(threshold - 0.00005) < 0.000001:
            print("✅ Correct base threshold (no session applied)")
        else:
            print("❌ Wrong threshold - something else is modifying it!")
            print(f"   Difference from base: {threshold - 0.00005:.10f}")
            print(f"   Difference from session-adjusted: {threshold - expected_with_session:.10f}")
            print(f"   Ratio to base: {threshold / 0.00005:.6f}")
    except Exception as method_error:
        print(f"Method error: {method_error}")
        
except Exception as e:
    print(f"Import error: {e}")
    
    # Fallback: just check the raw file
    print("\n=== Fallback: Check raw threshold values ===")
    try:
        # Read the forex optimizer file directly
        with open('/app/forex_scanner/core/strategies/helpers/macd_forex_optimizer.py', 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if 'CS.D.EURUSD.MINI.IP' in line and '0.0000' in line:
                print(f"Line {i+1}: {line.strip()}")
    except:
        print("Could not read file")