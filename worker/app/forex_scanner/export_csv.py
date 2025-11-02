import sys
sys.path.insert(0, "/app/forex_scanner")

from core.database.database_manager import DatabaseManager
import config
import pandas as pd

db = DatabaseManager(config.DATABASE_URL)
query = """
SELECT * FROM backtest_signals 
WHERE execution_id = (
    SELECT id FROM backtest_executions 
    WHERE strategy_name = 'SMC_STRUCTURE' 
    ORDER BY created_at DESC LIMIT 1
)
ORDER BY signal_timestamp DESC
"""
df = db.execute_query(query, {})
df.to_csv("/tmp/30daytestwithimprovements.csv", index=False)
print(f"✅ Exported {len(df)} signals to /tmp/30daytestwithimprovements.csv")
winners = df[df["trade_result"] == "win"]
losers = df[df["trade_result"] == "loss"]
print(f"   Winners: {len(winners)}, Losers: {len(losers)}")
