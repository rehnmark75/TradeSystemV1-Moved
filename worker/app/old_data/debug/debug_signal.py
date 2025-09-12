#!/usr/bin/env python3

import sys
sys.path.append("/app")

from forex_scanner.backtests.backtest_ema import EMABacktestFixed
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    try:
        backtest = EMABacktestFixed()
        
        # Get some test data first
        from forex_scanner.core.data_fetcher import EnhancedDataFetcher
        from forex_scanner.core.database import DatabaseManager
        import config
        
        db_manager = DatabaseManager()
        data_fetcher = EnhancedDataFetcher(db_manager)
        
        # Get data
        df = data_fetcher.get_enhanced_data("CS.D.AUDUSD.MINI.IP", "AUDUSD", "15m")
        if df is None or len(df) < 100:
            print("No data available")
            return
            
        print(f"Data available: {len(df)} rows")
        
        # Run backtest
        signals = backtest._run_ema_backtest_fixed(df, "CS.D.AUDUSD.MINI.IP", "15m")
        print(f"Generated {len(signals)} signals")
        
        if signals:
            first_signal = signals[0]
            print("\\nFirst signal keys:")
            for key in sorted(first_signal.keys()):
                value = first_signal[key]
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {str(value)[:50]}...")
            
            # Check for performance metrics specifically
            performance_fields = ['max_profit_pips', 'max_loss_pips', 'profit_loss_ratio', 
                                'is_winner', 'is_loser', 'trade_outcome', 'exit_reason']
            print("\\nPerformance metric fields:")
            for field in performance_fields:
                if field in first_signal:
                    print(f"  ✅ {field}: {first_signal[field]}")
                else:
                    print(f"  ❌ {field}: MISSING")
        else:
            print("No signals generated")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()