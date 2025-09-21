#!/usr/bin/env python3
"""
Compare EMA strategy initialization between debug script and unified backtest
"""

import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
forex_scanner_dir = os.path.dirname(script_dir)
app_dir = os.path.dirname(forex_scanner_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, forex_scanner_dir)
sys.path.insert(0, app_dir)

try:
    from core.strategies.ema_strategy import EMAStrategy
    import config
except ImportError as e:
    try:
        from forex_scanner.core.strategies.ema_strategy import EMAStrategy
        from forex_scanner import config
    except ImportError as e2:
        print(f"Import error: {e2}")
        sys.exit(1)

def compare_ema_initializations():
    """Compare EMA strategy initialization methods"""
    print("🔍 Comparing EMA Strategy Initializations")
    print("=" * 60)

    # Method 1: Debug script way (no parameters)
    print("\n📊 Method 1: Debug script way (no parameters)")
    strategy1 = EMAStrategy()
    print(f"   backtest_mode: {strategy1.backtest_mode}")
    print(f"   enhanced_validation: {strategy1.enhanced_validation}")
    print(f"   ema_config: {strategy1.ema_config}")
    print(f"   min_confidence: {strategy1.min_confidence}")

    # Method 2: Unified backtest way (with backtest_mode=True)
    print("\n📊 Method 2: Unified backtest way (backtest_mode=True)")
    strategy2 = EMAStrategy(
        backtest_mode=True,
        epic='CS.D.EURUSD.CEEM.IP',
        use_optimal_parameters=True
    )
    print(f"   backtest_mode: {strategy2.backtest_mode}")
    print(f"   enhanced_validation: {strategy2.enhanced_validation}")
    print(f"   ema_config: {strategy2.ema_config}")
    print(f"   min_confidence: {strategy2.min_confidence}")

    # Method 3: Check if breakout_validator differs
    print("\n📊 Breakout Validator Comparison:")
    print(f"   Strategy1 has breakout_validator: {hasattr(strategy1, 'breakout_validator') and strategy1.breakout_validator is not None}")
    print(f"   Strategy2 has breakout_validator: {hasattr(strategy2, 'breakout_validator') and strategy2.breakout_validator is not None}")

    # Check configuration differences
    print("\n📊 Configuration Comparison:")
    print(f"   Strategy1 epic: {getattr(strategy1, 'epic', 'None')}")
    print(f"   Strategy2 epic: {getattr(strategy2, 'epic', 'None')}")
    print(f"   Strategy1 use_optimal_parameters: {getattr(strategy1, 'use_optimal_parameters', 'None')}")
    print(f"   Strategy2 use_optimal_parameters: {getattr(strategy2, 'use_optimal_parameters', 'None')}")

    return strategy1, strategy2

if __name__ == "__main__":
    compare_ema_initializations()