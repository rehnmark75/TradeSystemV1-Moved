#!/usr/bin/env python3
"""Export SMC_STRUCTURE backtest signals to CSV"""

import sys
import pandas as pd

# Use same import pattern as enhanced_backtest_commands.py
try:
    import config
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager

def main():
    print("🔍 Connecting to database...")
    db = DatabaseManager(config.DATABASE_URL)
    
    # Get most recent SMC_STRUCTURE backtest (fixed column names)
    exec_query = """
    SELECT id, execution_name, strategy_name, created_at
    FROM backtest_executions
    WHERE strategy_name = 'SMC_STRUCTURE'
    ORDER BY created_at DESC
    LIMIT 1
    """
    
    executions = db.execute_query(exec_query, {})
    
    if len(executions) == 0:
        print("❌ No SMC_STRUCTURE backtests found")
        return 1
    
    execution_id = executions.iloc[0]['id']
    exec_name = executions.iloc[0]['execution_name']
    created_at = executions.iloc[0]['created_at']
    
    print(f"\n📊 Found backtest:")
    print(f"   ID: {execution_id}")
    print(f"   Name: {exec_name}")
    print(f"   Created: {created_at}")
    
    # Get all signals
    signals_query = """
    SELECT *
    FROM backtest_signals
    WHERE execution_id = :execution_id
    ORDER BY signal_timestamp DESC
    """
    
    signals = db.execute_query(signals_query, {'execution_id': int(execution_id)})
    
    if len(signals) == 0:
        print("❌ No signals found")
        return 1
    
    print(f"\n📈 Total signals: {len(signals)}")
    
    # Calculate stats
    winners = signals[signals['trade_result'] == 'win']
    losers = signals[signals['trade_result'] == 'loss']
    breakeven = signals[signals['trade_result'] == 'breakeven']
    bull = signals[signals['signal_type'] == 'BULL']
    bear = signals[signals['signal_type'] == 'BEAR']
    
    print(f"   📈 BULL signals: {len(bull)}")
    print(f"   📉 BEAR signals: {len(bear)}")
    print(f"   ✅ Winners: {len(winners)}")
    print(f"   ❌ Losers: {len(losers)}")
    print(f"   ➖ Breakeven: {len(breakeven)}")
    
    if len(winners) + len(losers) > 0:
        win_rate = len(winners) / (len(winners) + len(losers)) * 100
        print(f"   🎯 Win Rate: {win_rate:.1f}%")
    
    # Export to CSV
    csv_path = '/tmp/30daytestwithimprovements.csv'
    signals.to_csv(csv_path, index=False)
    
    print(f"\n✅ Exported to: {csv_path}")
    print(f"📋 Columns: {list(signals.columns)}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
