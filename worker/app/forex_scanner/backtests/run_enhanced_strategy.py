#!/usr/bin/env python3
"""
Unified Enhanced Strategy Runner
Runs any enhanced strategy with consistent interface
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description='Unified Enhanced Strategy Runner')
    parser.add_argument('strategy', choices=[
        'ema', 'macd', 'ichimoku', 'bb_supertrend', 'kama',
        'smc', 'zero_lag', 'scalping', 'combined', 'mean_reversion'
    ], help='Strategy to run')
    parser.add_argument('--epic', type=str, help='Epic to backtest')
    parser.add_argument('--days', type=int, default=7, help='Number of days')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe')
    parser.add_argument('--show-signals', action='store_true', help='Show signals')
    parser.add_argument('--no-optimization', action='store_true', help='Disable optimization')

    args = parser.parse_args()

    # Import and run the specified strategy
    try:
        if args.strategy == 'mean_reversion':
            from backtests.backtest_mean_reversion import MeanReversionBacktest as BacktestClass
        else:
            module_name = f"backtests.backtest_{args.strategy}_enhanced"
            class_name = f"Enhanced{args.strategy.title().replace('_', '')}Backtest"

            module = __import__(module_name, fromlist=[class_name])
            BacktestClass = getattr(module, class_name)

        # Create and run backtest
        backtest = BacktestClass(use_optimal_parameters=not args.no_optimization)

        result = backtest.run_backtest(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=args.show_signals
        )

        if result.success:
            print(f"\n🎯 {args.strategy.title()} Backtest Completed!")
            print(f"   Signals: {result.total_signals}")
            print(f"   Time: {result.execution_time:.2f}s")
        else:
            print(f"\n❌ Backtest failed: {result.error_message}")

    except Exception as e:
        print(f"\n❌ Error running {args.strategy}: {e}")

if __name__ == "__main__":
    main()
