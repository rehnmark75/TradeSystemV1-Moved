#!/usr/bin/env python3
"""
GBPUSD Parameter Optimization - Automated parameter sweep
Runs multiple backtests without user input, presents results at end

Usage:
    docker exec task-worker python /app/forex_scanner/optimize_gbpusd.py
"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

# Suppress ALL logging before any imports
import logging
logging.disable(logging.CRITICAL)

import os
os.environ['LOG_LEVEL'] = 'ERROR'

from datetime import datetime, timedelta
from typing import Dict, List, Any
from itertools import product

from forex_scanner.commands.enhanced_backtest_commands import EnhancedBacktestCommands
from forex_scanner.core.database import DatabaseManager
from forex_scanner import config

# REDUCED parameter grid for faster testing (key combinations only)
# 8 total combinations with R:R >= 1.5 filter
PARAMETER_GRID = {
    'fixed_stop_loss_pips': [8, 10],
    'fixed_take_profit_pips': [15, 20],
    'min_confidence': [0.50, 0.55],
}


class GBPUSDOptimizer:
    """Automated parameter optimizer for GBPUSD"""

    def __init__(self):
        self.backtest_cmd = EnhancedBacktestCommands()
        self.db = DatabaseManager(config.DATABASE_URL)
        self.results = []
        self.logger = logging.getLogger(__name__)

    def generate_combinations(self) -> List[Dict]:
        """Generate valid parameter combinations (R:R >= 1.5)"""
        combinations = []
        keys = list(PARAMETER_GRID.keys())

        for values in product(*PARAMETER_GRID.values()):
            params = dict(zip(keys, values))

            # Filter: R:R must be >= 1.5
            rr = params['fixed_take_profit_pips'] / params['fixed_stop_loss_pips']
            if rr >= 1.5:
                params['rr_ratio'] = rr  # Store for reference
                combinations.append(params)

        return combinations

    def _suppress_logging(self):
        """Aggressively suppress all logging"""
        logging.disable(logging.CRITICAL)
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.CRITICAL)
            logging.getLogger(name).disabled = True

    def run_single_backtest(self, params: Dict, epic: str,
                           start_date: datetime, end_date: datetime) -> Dict:
        """Run a single backtest with given parameters"""

        # Suppress logging before each backtest
        self._suppress_logging()

        # Prepare override dict (exclude rr_ratio - it's calculated, not a param)
        override = {k: v for k, v in params.items() if k != 'rr_ratio'}

        try:
            success = self.backtest_cmd.run_enhanced_backtest(
                epic=epic,
                start_date=start_date,
                end_date=end_date,
                strategy='SMC_SIMPLE',
                config_override=override,
                use_historical_intelligence=False,  # DISABLED as requested
                pipeline=False,  # Fast mode for optimization
                show_signals=False
            )

            if success:
                return self._get_latest_execution_results(epic)
            else:
                return {'status': 'FAILED', 'error': 'Backtest returned False'}

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def _get_latest_execution_results(self, epic: str) -> Dict:
        """Get results from the most recent backtest execution"""

        query = """
            SELECT
                be.id as execution_id,
                COUNT(bs.id) as total_signals,
                SUM(CASE WHEN bs.trade_result = 'win' THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN bs.trade_result = 'loss' THEN 1 ELSE 0 END) as losers,
                SUM(bs.pips_gained) as total_pips,
                AVG(bs.confidence_score) as avg_confidence,
                SUM(CASE WHEN bs.trade_result = 'win' THEN bs.pips_gained ELSE 0 END) as gross_profit,
                SUM(CASE WHEN bs.trade_result = 'loss' THEN ABS(bs.pips_gained) ELSE 0 END) as gross_loss
            FROM backtest_executions be
            LEFT JOIN backtest_signals bs ON be.id = bs.execution_id
            WHERE be.status = 'COMPLETED'
            GROUP BY be.id, be.created_at
            ORDER BY be.created_at DESC
            LIMIT 1
        """

        result = self.db.execute_query(query)

        if result.empty:
            return {'status': 'FAILED', 'error': 'No results found'}

        row = result.iloc[0]
        winners = int(row['winners'] or 0)
        losers = int(row['losers'] or 0)
        total = winners + losers

        return {
            'status': 'SUCCESS',
            'execution_id': int(row['execution_id']),
            'total_signals': int(row['total_signals'] or 0),
            'winners': winners,
            'losers': losers,
            'win_rate': winners / max(total, 1),
            'total_pips': float(row['total_pips'] or 0),
            'avg_confidence': float(row['avg_confidence'] or 0),
            'profit_factor': (float(row['gross_profit'] or 0) /
                            max(float(row['gross_loss'] or 1), 0.01))
        }

    def run_optimization(self, days: int = 30):
        """Main optimization loop"""

        epic = 'CS.D.GBPUSD.MINI.IP'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        combinations = self.generate_combinations()
        total = len(combinations)

        # Suppress logging before starting
        self._suppress_logging()

        print(f"\n{'='*80}", flush=True)
        print(f"🚀 GBPUSD PARAMETER OPTIMIZATION", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Epic: {epic}", flush=True)
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)", flush=True)
        print(f"Parameter combinations: {total}", flush=True)
        print(f"Historical Intelligence: DISABLED", flush=True)
        print(f"{'='*80}\n", flush=True)

        for idx, params in enumerate(combinations, 1):
            print(f"[{idx}/{total}] Testing: SL={params['fixed_stop_loss_pips']}, "
                  f"TP={params['fixed_take_profit_pips']}, Conf={params['min_confidence']}",
                  flush=True)

            result = self.run_single_backtest(params, epic, start_date, end_date)
            result['params'] = params
            self.results.append(result)

            if result['status'] == 'SUCCESS':
                print(f"   ✅ Signals: {result['total_signals']}, "
                      f"Win: {result['win_rate']:.1%}, "
                      f"PF: {result['profit_factor']:.2f}, "
                      f"Pips: {result['total_pips']:.1f}", flush=True)
            else:
                print(f"   ❌ {result.get('error', 'Unknown error')}", flush=True)

        self._print_final_report()

    def _print_final_report(self):
        """Generate and print the final optimization report"""

        successful = [r for r in self.results if r['status'] == 'SUCCESS']

        print(f"\n{'='*80}")
        print(f"📊 OPTIMIZATION RESULTS - GBPUSD 30 Days")
        print(f"{'='*80}")
        print(f"Total tests: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(self.results) - len(successful)}")

        if not successful:
            print("❌ No successful backtests!")
            return

        # Sort by composite score: (win_rate * profit_factor) + normalized pips
        max_pips = max(abs(r['total_pips']) for r in successful) or 1

        for r in successful:
            r['score'] = (r['win_rate'] * r['profit_factor']) + (r['total_pips'] / max_pips)

        sorted_results = sorted(successful, key=lambda x: x['score'], reverse=True)

        # Top 10 configurations
        print(f"\n🏆 TOP 10 CONFIGURATIONS:")
        print(f"{'Rank':<5} {'SL':<4} {'TP':<4} {'R:R':<5} {'Conf':<6} "
              f"{'Signals':<8} {'Win%':<7} {'PF':<6} {'Pips':<10}")
        print("-" * 70)

        for i, r in enumerate(sorted_results[:10], 1):
            p = r['params']
            print(f"{i:<5} {p['fixed_stop_loss_pips']:<4} {p['fixed_take_profit_pips']:<4} "
                  f"{p['rr_ratio']:<5.2f} {p['min_confidence']:<6.0%} "
                  f"{r['total_signals']:<8} {r['win_rate']:<7.1%} "
                  f"{r['profit_factor']:<6.2f} {r['total_pips']:<10.1f}")

        # Best configuration details
        best = sorted_results[0]
        bp = best['params']

        print(f"\n{'='*80}")
        print(f"🥇 BEST CONFIGURATION FOR GBPUSD:")
        print(f"{'='*80}")
        print(f"   Stop Loss:     {bp['fixed_stop_loss_pips']} pips")
        print(f"   Take Profit:   {bp['fixed_take_profit_pips']} pips")
        print(f"   Risk:Reward:   {bp['rr_ratio']:.2f}")
        print(f"   Min Confidence: {bp['min_confidence']:.0%}")
        print(f"\n   Performance:")
        print(f"   - Total Signals: {best['total_signals']}")
        print(f"   - Win Rate:      {best['win_rate']:.1%}")
        print(f"   - Profit Factor: {best['profit_factor']:.2f}")
        print(f"   - Total Pips:    {best['total_pips']:.1f}")
        print(f"{'='*80}")

        # Command to apply this configuration
        print(f"\n📝 To use this configuration:")
        print(f"   python bt.py GBPUSD 30 \\")
        print(f"     --override fixed_stop_loss_pips={bp['fixed_stop_loss_pips']} \\")
        print(f"     --override fixed_take_profit_pips={bp['fixed_take_profit_pips']} \\")
        print(f"     --override min_confidence={bp['min_confidence']} \\")
        print(f"     --no-historical-intelligence")

        # Save as snapshot suggestion
        print(f"\n💾 To save as a snapshot:")
        print(f"   python snapshot_cli.py create gbpusd_optimized \\")
        print(f"     --set fixed_stop_loss_pips={bp['fixed_stop_loss_pips']} \\")
        print(f"     --set fixed_take_profit_pips={bp['fixed_take_profit_pips']} \\")
        print(f"     --set min_confidence={bp['min_confidence']} \\")
        print(f"     --desc 'Optimized GBPUSD config from 7-day backtest'")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    optimizer = GBPUSDOptimizer()
    optimizer.run_optimization(days=7)  # Use 7 days for faster testing (~3 min per test)
