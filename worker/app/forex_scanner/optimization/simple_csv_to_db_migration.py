#!/usr/bin/env python3
"""
Simple SMC CSV to Database Migration

Simplified script to migrate SMC CSV data to database tables.
"""

import sys
import pandas as pd
import logging
from datetime import datetime

sys.path.append('/app/forex_scanner')

from core.database import DatabaseManager
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run simple migration."""
    
    print("🔄 Simple SMC CSV to Database Migration")
    print("=" * 50)
    
    # Initialize database
    db = DatabaseManager(config.DATABASE_URL)
    
    # Load CSV
    csv_path = '/app/forex_scanner/optimization/results/smc_optimization_results.csv'
    df = pd.read_csv(csv_path)
    
    print(f"📊 Loaded {len(df)} records from CSV")
    
    # Get unique epics and their best configurations
    best_configs = {}
    for epic in df['epic'].unique():
        epic_data = df[df['epic'] == epic]
        best_row = epic_data.loc[epic_data['performance_score'].idxmax()]
        best_configs[epic] = best_row
    
    print(f"🎯 Found {len(best_configs)} epics with best configurations")
    
    # Insert best parameters directly using simple queries
    inserted_count = 0
    
    for epic, row in best_configs.items():
        try:
            # Direct SQL execution
            query = f"""
            INSERT INTO smc_best_parameters (
                epic, best_smc_config, best_confidence_level, best_timeframe, 
                optimal_stop_loss_pips, optimal_take_profit_pips, optimal_risk_reward_ratio,
                best_win_rate, best_profit_factor, best_net_pips, best_performance_score,
                confluence_accuracy, optimization_days_used, total_combinations_tested
            ) VALUES (
                '{epic}', '{row['smc_config']}', {row['confidence_level']}, '{row['timeframe']}',
                {row['stop_loss_pips']}, {row['take_profit_pips']}, {row['risk_reward_ratio']},
                {row['win_rate']}, {row['profit_factor']}, {row['net_pips']}, {row['performance_score']},
                {row['confluence_accuracy']}, 30, {len(df[df['epic'] == epic])}
            ) ON CONFLICT (epic) DO UPDATE SET
                best_smc_config = EXCLUDED.best_smc_config,
                best_confidence_level = EXCLUDED.best_confidence_level,
                best_timeframe = EXCLUDED.best_timeframe,
                optimal_stop_loss_pips = EXCLUDED.optimal_stop_loss_pips,
                optimal_take_profit_pips = EXCLUDED.optimal_take_profit_pips,
                optimal_risk_reward_ratio = EXCLUDED.optimal_risk_reward_ratio,
                best_win_rate = EXCLUDED.best_win_rate,
                best_profit_factor = EXCLUDED.best_profit_factor,
                best_net_pips = EXCLUDED.best_net_pips,
                best_performance_score = EXCLUDED.best_performance_score,
                confluence_accuracy = EXCLUDED.confluence_accuracy,
                last_updated = CURRENT_TIMESTAMP
            """
            
            result = db.execute_query(query)
            inserted_count += 1
            print(f"✅ {epic}: {row['smc_config']} (Win Rate: {row['win_rate']:.1f}%, Score: {row['performance_score']:.1f})")
            
        except Exception as e:
            print(f"❌ Failed to insert {epic}: {e}")
    
    # Verify results
    print(f"\n📋 Verification:")
    count_result = db.execute_query("SELECT COUNT(*) FROM smc_best_parameters")
    if count_result:
        print(f"   Total records in smc_best_parameters: {count_result[0][0]}")
    
    top_result = db.execute_query("""
        SELECT epic, best_smc_config, best_win_rate, best_performance_score 
        FROM smc_best_parameters 
        ORDER BY best_performance_score DESC 
        LIMIT 3
    """)
    
    if top_result:
        print(f"   Top 3 performers:")
        for row in top_result:
            print(f"     {row[0]}: {row[1]} (Win Rate: {row[2]:.1f}%, Score: {row[3]:.1f})")
    
    print(f"\n✅ Migration completed! Inserted/updated {inserted_count} records.")
    return True

if __name__ == "__main__":
    main()