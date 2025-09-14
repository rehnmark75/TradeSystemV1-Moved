#!/usr/bin/env python3
"""
Focused SL/TP Optimization for Enhanced EMA Strategy
Tests key SL/TP combinations to find optimal levels for 58.2% win rate strategy
"""

import sys
import os
import logging
import subprocess
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_backtest_with_sl_tp(epic, sl_pips, tp_pips, days=7, timeframe="15m"):
    """Run backtest with specific SL/TP levels by temporarily modifying config"""
    
    logger.info(f"🧪 Testing {epic} with SL: {sl_pips} pips, TP: {tp_pips} pips")
    
    # Read current config
    config_path = "/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/configdata/strategies/config_ema_strategy.py"
    
    # Backup original config
    backup_path = config_path + ".backup"
    subprocess.run(f"cp '{config_path}' '{backup_path}'", shell=True, check=True)
    
    try:
        # Modify config with new SL/TP
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace SL/TP values (find the lines and replace them)
        lines = config_content.split('\n')
        modified_lines = []
        
        for line in lines:
            if 'EMA_STOP_LOSS_PIPS' in line and '=' in line:
                modified_lines.append(f"EMA_STOP_LOSS_PIPS = {sl_pips}")
            elif 'EMA_TAKE_PROFIT_PIPS' in line and '=' in line:
                modified_lines.append(f"EMA_TAKE_PROFIT_PIPS = {tp_pips}")
            else:
                modified_lines.append(line)
        
        # Write modified config
        with open(config_path, 'w') as f:
            f.write('\n'.join(modified_lines))
        
        # Run backtest
        cmd = f"docker exec task-worker python3 forex_scanner/backtests/backtest_ema.py --epic {epic} --days {days} --timeframe {timeframe}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Parse results from output
        output_lines = result.stdout.split('\n')
        
        # Extract performance metrics
        performance = {}
        for line in output_lines:
            if "Total Signals:" in line:
                performance['total_signals'] = int(line.split(': ')[1])
            elif "Win Rate:" in line:
                performance['win_rate'] = float(line.split(': ')[1].replace('%', ''))
            elif "Average Profit:" in line:
                performance['avg_profit'] = float(line.split(': ')[1].split(' ')[0])
            elif "Average Loss:" in line:
                performance['avg_loss'] = float(line.split(': ')[1].split(' ')[0])
            elif "Winners:" in line:
                performance['winners'] = int(line.split(': ')[1].split(' ')[0])
            elif "Losers:" in line:
                performance['losers'] = int(line.split(': ')[1].split(' ')[0])
        
        # Calculate additional metrics
        if performance:
            risk_reward = tp_pips / sl_pips if sl_pips > 0 else 0
            net_pips = (performance.get('winners', 0) * tp_pips) - (performance.get('losers', 0) * sl_pips)
            
            performance.update({
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'risk_reward_ratio': risk_reward,
                'net_pips': net_pips,
                'profit_factor': (performance.get('winners', 0) * tp_pips) / (performance.get('losers', 1) * sl_pips)
            })
        
        return performance
        
    finally:
        # Restore original config
        subprocess.run(f"cp '{backup_path}' '{config_path}'", shell=True, check=True)
        subprocess.run(f"rm '{backup_path}'", shell=True, check=True)

def main():
    """Test key SL/TP combinations"""
    
    logger.info("🎯 EMA Strategy SL/TP Optimization")
    logger.info("=" * 50)
    
    # Test epic
    epic = "CS.D.GBPUSD.MINI.IP"
    
    # SL/TP combinations to test (based on common ratios and our current performance)
    sl_tp_combinations = [
        (5, 15),   # 3:1 ratio - tight scalping
        (8, 16),   # 2:1 ratio - similar to current
        (8, 24),   # 3:1 ratio - current SL with higher TP
        (10, 20),  # 2:1 ratio - balanced
        (10, 30),  # 3:1 ratio - balanced with higher TP
        (12, 24),  # 2:1 ratio - wider SL
        (15, 30),  # 2:1 ratio - swing approach
        (15, 45),  # 3:1 ratio - swing with high TP
    ]
    
    results = []
    
    for sl_pips, tp_pips in sl_tp_combinations:
        try:
            performance = run_backtest_with_sl_tp(epic, sl_pips, tp_pips)
            if performance:
                results.append(performance)
                
                logger.info(f"✅ SL: {sl_pips}, TP: {tp_pips} - Signals: {performance.get('total_signals', 0)}, "
                           f"Win Rate: {performance.get('win_rate', 0):.1f}%, "
                           f"Net Pips: {performance.get('net_pips', 0):.1f}, "
                           f"R:R: {performance.get('risk_reward_ratio', 0):.1f}")
            else:
                logger.warning(f"❌ No results for SL: {sl_pips}, TP: {tp_pips}")
                
        except Exception as e:
            logger.error(f"Error testing SL: {sl_pips}, TP: {tp_pips} - {e}")
    
    # Find best configuration
    if results:
        logger.info("\n🏆 OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        
        # Sort by net pips (profitability)
        best_by_pips = max(results, key=lambda x: x.get('net_pips', 0))
        
        # Sort by win rate
        best_by_winrate = max(results, key=lambda x: x.get('win_rate', 0))
        
        # Sort by profit factor
        best_by_pf = max(results, key=lambda x: x.get('profit_factor', 0))
        
        logger.info(f"📈 Best Net Pips: SL {best_by_pips.get('sl_pips')} / TP {best_by_pips.get('tp_pips')} "
                   f"= {best_by_pips.get('net_pips', 0):.1f} pips ({best_by_pips.get('win_rate', 0):.1f}% WR)")
        
        logger.info(f"🎯 Best Win Rate: SL {best_by_winrate.get('sl_pips')} / TP {best_by_winrate.get('tp_pips')} "
                   f"= {best_by_winrate.get('win_rate', 0):.1f}% ({best_by_winrate.get('net_pips', 0):.1f} pips)")
        
        logger.info(f"⚡ Best Profit Factor: SL {best_by_pf.get('sl_pips')} / TP {best_by_pf.get('tp_pips')} "
                   f"= {best_by_pf.get('profit_factor', 0):.2f} ({best_by_pf.get('win_rate', 0):.1f}% WR)")
        
        # Detailed results table
        logger.info("\n📊 DETAILED RESULTS:")
        logger.info("SL  | TP  | Signals | WR%   | Net Pips | R:R | PF")
        logger.info("----|-----|---------|-------|----------|-----|----")
        
        for result in sorted(results, key=lambda x: x.get('net_pips', 0), reverse=True):
            logger.info(f"{result.get('sl_pips', 0):2d}  | "
                       f"{result.get('tp_pips', 0):2d}  | "
                       f"{result.get('total_signals', 0):7d} | "
                       f"{result.get('win_rate', 0):5.1f} | "
                       f"{result.get('net_pips', 0):8.1f} | "
                       f"{result.get('risk_reward_ratio', 0):3.1f} | "
                       f"{result.get('profit_factor', 0):4.2f}")
    
    else:
        logger.error("❌ No results obtained")

if __name__ == "__main__":
    main()