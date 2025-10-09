#!/usr/bin/env python3
"""
Quick test of MACD signal export functionality
"""

import logging
import sys
import os
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from run_realistic_backtest import RealisticMACDBacktest

def main():
    """Quick test with minimal configuration"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Minimal test configuration
    test_config = {
        'start_date': datetime(2024, 1, 1),
        'end_date': datetime(2024, 1, 3),  # Just 2 days for quick test
        'timeframe': '15m',
        'test_pairs': [
            'CS.D.EURUSD.CEEM.IP'  # Just one pair for speed
        ],
        'initial_balance': 10000.0,
        'max_trades_per_day': 3,
        'signal_spacing_hours': 4
    }

    try:
        print("🚀 Starting quick MACD signal export test...")

        # Initialize and run backtest
        backtest = RealisticMACDBacktest(test_config)

        # Run backtest for one pair
        results = backtest.run_full_backtest()

        # Generate report
        report = backtest.generate_report(results)
        print(report)

        # Test signal export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signals_file = f"/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/backtests/macd/reports/test_signals_{timestamp}.csv"

        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(signals_file), exist_ok=True)

        # Export signals
        backtest.export_detailed_signals(results, signals_file)

        print(f"\n✅ Quick test completed!")
        print(f"📊 Signal export file: {signals_file}")

        return results

    except Exception as e:
        logging.error(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()