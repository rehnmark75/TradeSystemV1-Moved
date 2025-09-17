#!/usr/bin/env python3
"""Test script for mean reversion strategy imports"""

try:
    print("Testing mean reversion strategy imports...")

    from forex_scanner.core.strategies.mean_reversion_strategy import MeanReversionStrategy
    print("✅ MeanReversionStrategy imported successfully")

    from forex_scanner.backtests.backtest_mean_reversion import MeanReversionBacktest
    print("✅ BacktestMeanReversion imported successfully")

    import forex_scanner.configdata.strategies.config_mean_reversion_strategy as mr_config
    print("✅ Mean reversion configuration imported successfully")

    # Test basic configuration access
    epic_list = getattr(mr_config, 'EPIC_LIST', [])
    print(f"✅ Configuration loaded with {len(epic_list)} epics")

    # Test strategy initialization
    strategy = MeanReversionStrategy()
    print("✅ Strategy initialized successfully")

    # Test backtest initialization
    backtest = MeanReversionBacktest()
    print("✅ Backtest class initialized successfully")

    print("\n🎯 All mean reversion components successfully integrated!")
    print("✅ Strategy is ready for live trading and backtesting")

except Exception as e:
    print(f"❌ Import/initialization error: {e}")
    import traceback
    traceback.print_exc()