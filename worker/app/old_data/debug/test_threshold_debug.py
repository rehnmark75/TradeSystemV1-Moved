#!/usr/bin/env python3
"""
Debug script to check what threshold is being returned by forex optimizer
"""

import sys
import os
sys.path.append('/datadrive/Trader/TradeSystemV1/worker/app')
sys.path.append('/datadrive/Trader/TradeSystemV1/worker/app/forex_scanner')

try:
    from forex_scanner.core.strategies.helpers.macd_forex_optimizer import MACDForexOptimizer
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create forex optimizer
    forex_optimizer = MACDForexOptimizer(logger=logger)
    
    # Test EURUSD threshold
    epic = 'CS.D.EURUSD.MINI.IP'
    threshold = forex_optimizer.get_macd_threshold_for_epic(epic)
    
    print(f"\n=== THRESHOLD DEBUG ===")
    print(f"Epic: {epic}")
    print(f"Threshold from forex_optimizer: {threshold:.8f}")
    print(f"Expected from config: 0.00005000")
    print(f"Match: {'✅ YES' if abs(threshold - 0.00005) < 0.000001 else '❌ NO'}")
    
    # Check hardcoded thresholds
    print(f"\n=== HARDCODED THRESHOLDS ===")
    hardcoded = forex_optimizer.forex_macd_thresholds.get(epic)
    print(f"Hardcoded threshold: {hardcoded:.8f}")
    
    # Check session multipliers
    session = forex_optimizer.get_current_market_session()
    multiplier = forex_optimizer.session_multipliers.get(session, 1.0)
    expected_with_session = hardcoded * multiplier
    print(f"Current session: {session}")
    print(f"Session multiplier: {multiplier}")
    print(f"Expected with session: {expected_with_session:.8f}")
    
    print(f"\nIf the returned threshold is 0.000055, then something is overriding the forex optimizer!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()