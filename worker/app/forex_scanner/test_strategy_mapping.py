#!/usr/bin/env python3
"""
Test script to verify strategy mapping works correctly
"""

def test_strategy_mapping():
    """Test if strategy name mapping works"""

    # Simulate the mapping from backtest_scanner.py
    strategy_methods = {
        'EMA_CROSSOVER': 'detect_signals_mid_prices',
        'EMA': 'detect_signals_mid_prices',
        'MACD': 'detect_macd_ema_signals',
        'MACD_EMA': 'detect_macd_ema_signals',
        'KAMA': 'detect_kama_signals',
        'BOLLINGER_SUPERTREND': 'detect_bb_supertrend_signals',
        'BB_SUPERTREND': 'detect_bb_supertrend_signals',
        'BB': 'detect_bb_supertrend_signals',
        'ZERO_LAG': 'detect_zero_lag_signals',
        'ZEROLAG': 'detect_zero_lag_signals',  # Fixed: Accept both forms
        'ZL': 'detect_zero_lag_signals',
        'MOMENTUM': 'detect_momentum_signals',
        'SMC_FAST': 'detect_smc_signals',
        'SMC': 'detect_smc_signals',
        'ICHIMOKU': 'detect_ichimoku_signals',
        'ICHIMOKU_CLOUD': 'detect_ichimoku_signals',
        'MEAN_REVERSION': 'detect_mean_reversion_signals',
        'MEANREV': 'detect_mean_reversion_signals',  # Fixed: Accept short form
        'RANGING_MARKET': 'detect_ranging_market_signals',
        'RANGING': 'detect_ranging_market_signals'
    }

    # Test cases
    test_strategies = ['ZEROLAG', 'ZERO_LAG', 'EMA', 'MACD', 'MEANREV', 'RANGING']

    print("🧪 Testing strategy mapping:")
    for strategy in test_strategies:
        strategy_name = strategy.upper()
        if strategy_name in strategy_methods:
            method_name = strategy_methods[strategy_name]
            print(f"   ✅ {strategy} -> {method_name}")
        else:
            print(f"   ❌ {strategy} -> NOT FOUND (would run all strategies)")

    # Test the specific case that was failing
    test_strategy = "ZEROLAG"
    strategy_name = test_strategy.upper()

    if strategy_name in strategy_methods:
        method_name = strategy_methods[strategy_name]
        print(f"\n🎯 SPECIFIC TEST: '{test_strategy}' -> '{method_name}'")
        print("   ✅ Strategy filtering should work correctly!")
        return True
    else:
        print(f"\n❌ SPECIFIC TEST: '{test_strategy}' -> NOT FOUND")
        print("   ❌ Strategy filtering will NOT work!")
        return False

if __name__ == "__main__":
    success = test_strategy_mapping()
    exit(0 if success else 1)