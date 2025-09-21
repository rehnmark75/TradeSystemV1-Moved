#!/usr/bin/env python3
"""
All Strategies Comparison Backtest - Master Module
Run: python backtest_all.py --epic CS.D.EURUSD.CEEM.IP --days 7 --timeframe 15m
"""

import sys
import os
import argparse
import logging
import subprocess
import time
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
except ImportError:
    from forex_scanner import config


class AllStrategiesBacktest:
    """Master backtest runner for all strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger('all_strategies_backtest')
        self.setup_logging()
        
        # Available strategy backtests
        self.available_strategies = {
            'ema': 'backtest_ema.py',
            'macd': 'backtest_macd.py', 
            'kama': 'backtest_kama.py',
            'bb_supertrend': 'backtest_bb_supertrend.py',
            'zero_lag': 'backtest_zero_lag.py',
            'combined': 'backtest_combined.py',
            'scalping': 'backtest_scalping.py'
        }
        
        # Strategy enabled status
        self.strategy_enabled = {
            'ema': getattr(config, 'SIMPLE_EMA_STRATEGY', True),
            'macd': getattr(config, 'MACD_EMA_STRATEGY', True),
            'kama': getattr(config, 'KAMA_STRATEGY', True),
            'bb_supertrend': getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', True),
            'zero_lag': getattr(config, 'ZERO_LAG_STRATEGY', False),
            'combined': True,  # Always available
            'scalping': getattr(config, 'SCALPING_STRATEGY_ENABLED', True)
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def run_comparison_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        strategies: List[str] = None,
        parallel: bool = False,
        show_details: bool = False
    ) -> bool:
        """Run comparison backtest across all or selected strategies"""
        
        self.logger.info("🏆 ALL STRATEGIES COMPARISON BACKTEST")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Epic: {epic or 'All configured pairs'}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🔄 Parallel execution: {parallel}")
        
        # Determine which strategies to test
        if strategies:
            test_strategies = [s for s in strategies if s in self.available_strategies]
            if not test_strategies:
                self.logger.error(f"❌ No valid strategies specified. Available: {list(self.available_strategies.keys())}")
                return False
        else:
            # Test all enabled strategies
            test_strategies = [s for s, enabled in self.strategy_enabled.items() if enabled]
        
        self.logger.info(f"🧪 Testing strategies: {', '.join(test_strategies)}")
        
        # Run backtests
        if parallel:
            results = self._run_parallel_backtests(test_strategies, epic, days, timeframe)
        else:
            results = self._run_sequential_backtests(test_strategies, epic, days, timeframe, show_details)
        
        # Display comparison results
        self._display_comparison_results(results, epic, days, timeframe)
        
        return True
    
    def _run_sequential_backtests(
        self, 
        strategies: List[str], 
        epic: str, 
        days: int, 
        timeframe: str,
        show_details: bool
    ) -> Dict:
        """Run backtests sequentially"""
        
        results = {}
        
        for i, strategy in enumerate(strategies, 1):
            self.logger.info(f"\n🧪 [{i}/{len(strategies)}] Testing {strategy.upper()} Strategy")
            self.logger.info("-" * 40)
            
            start_time = time.time()
            
            try:
                # Build command
                script_path = self.available_strategies[strategy]
                cmd = ['python', script_path]
                
                # Add arguments
                if epic:
                    cmd.extend(['--epic', epic])
                cmd.extend(['--days', str(days)])
                cmd.extend(['--timeframe', timeframe])
                
                # Add strategy-specific arguments
                if strategy == 'combined' and show_details:
                    cmd.append('--show-details')
                
                # Check if script exists
                if not os.path.exists(script_path):
                    self.logger.warning(f"⚠️ Script not found: {script_path} - Skipping {strategy}")
                    results[strategy] = {
                        'success': False,
                        'error': 'Script not found',
                        'duration': 0
                    }
                    continue
                
                # Run the backtest
                self.logger.info(f"   Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    # Parse output for signal count
                    signal_count = self._parse_signal_count(result.stdout)
                    
                    results[strategy] = {
                        'success': True,
                        'signal_count': signal_count,
                        'duration': duration,
                        'output': result.stdout
                    }
                    
                    self.logger.info(f"   ✅ {strategy} completed: {signal_count} signals ({duration:.1f}s)")
                    
                else:
                    results[strategy] = {
                        'success': False,
                        'error': result.stderr or 'Unknown error',
                        'duration': duration,
                        'output': result.stdout
                    }
                    
                    self.logger.error(f"   ❌ {strategy} failed: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                results[strategy] = {
                    'success': False,
                    'error': 'Timeout (>5 minutes)',
                    'duration': duration
                }
                self.logger.error(f"   ⏰ {strategy} timed out after {duration:.1f}s")
                
            except Exception as e:
                duration = time.time() - start_time
                results[strategy] = {
                    'success': False,
                    'error': str(e),
                    'duration': duration
                }
                self.logger.error(f"   💥 {strategy} crashed: {e}")
        
        return results
    
    def _run_parallel_backtests(
        self, 
        strategies: List[str], 
        epic: str, 
        days: int, 
        timeframe: str
    ) -> Dict:
        """Run backtests in parallel (basic implementation)"""
        
        self.logger.info("🚀 Running backtests in parallel...")
        
        # For now, fall back to sequential
        # In a future enhancement, could use multiprocessing
        self.logger.warning("⚠️ Parallel execution not yet implemented, falling back to sequential")
        
        return self._run_sequential_backtests(strategies, epic, days, timeframe, False)
    
    def _parse_signal_count(self, output: str) -> int:
        """Parse signal count from backtest output"""
        try:
            # Look for patterns like "TOTAL SIGNALS: 25" or "signals found: 15"
            lines = output.split('\n')
            
            for line in lines:
                if 'TOTAL' in line.upper() and 'SIGNAL' in line.upper():
                    # Extract number from lines like "TOTAL EMA SIGNALS: 25"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        number_part = parts[-1].strip()
                        # Extract first number found
                        import re
                        numbers = re.findall(r'\d+', number_part)
                        if numbers:
                            return int(numbers[0])
                
                elif 'signals found' in line.lower():
                    # Extract from "25 signals found"
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        return int(numbers[0])
            
            return 0
            
        except Exception:
            return 0
    
    def _display_comparison_results(
        self, 
        results: Dict, 
        epic: str, 
        days: int, 
        timeframe: str
    ) -> None:
        """Display comprehensive comparison results"""
        
        self.logger.info("\n🏆 STRATEGY COMPARISON RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Epic: {epic or 'All pairs'} | Days: {days} | Timeframe: {timeframe}")
        self.logger.info("=" * 60)
        
        # Sort results by success then by signal count
        sorted_results = sorted(
            results.items(),
            key=lambda x: (x[1]['success'], x[1].get('signal_count', 0)),
            reverse=True
        )
        
        successful_strategies = []
        failed_strategies = []
        
        # Display individual results
        self.logger.info("\n📊 INDIVIDUAL RESULTS:")
        self.logger.info("-" * 40)
        
        for strategy, result in sorted_results:
            strategy_name = strategy.upper().ljust(12)
            
            if result['success']:
                signal_count = result.get('signal_count', 0)
                duration = result.get('duration', 0)
                
                self.logger.info(f"✅ {strategy_name} | {signal_count:3d} signals | {duration:5.1f}s")
                successful_strategies.append((strategy, signal_count))
                
            else:
                error = result.get('error', 'Unknown error')
                duration = result.get('duration', 0)
                
                self.logger.info(f"❌ {strategy_name} | FAILED: {error} | {duration:5.1f}s")
                failed_strategies.append((strategy, error))
        
        # Summary statistics
        self._display_summary_statistics(successful_strategies, failed_strategies)
        
        # Performance ranking
        if successful_strategies:
            self._display_performance_ranking(successful_strategies)
        
        # Recommendations
        self._display_recommendations(successful_strategies, failed_strategies, epic, days)
    
    def _display_summary_statistics(
        self, 
        successful: List[Tuple[str, int]], 
        failed: List[Tuple[str, str]]
    ) -> None:
        """Display summary statistics"""
        
        total_strategies = len(successful) + len(failed)
        total_signals = sum(count for _, count in successful)
        
        self.logger.info(f"\n📈 SUMMARY STATISTICS:")
        self.logger.info("-" * 30)
        self.logger.info(f"   Strategies tested: {total_strategies}")
        self.logger.info(f"   Successful: {len(successful)}")
        self.logger.info(f"   Failed: {len(failed)}")
        self.logger.info(f"   Total signals: {total_signals}")
        
        if successful:
            avg_signals = total_signals / len(successful)
            self.logger.info(f"   Average signals per strategy: {avg_signals:.1f}")
    
    def _display_performance_ranking(self, successful: List[Tuple[str, int]]) -> None:
        """Display performance ranking"""
        
        self.logger.info(f"\n🥇 PERFORMANCE RANKING:")
        self.logger.info("-" * 25)
        
        # Sort by signal count (highest first)
        ranked = sorted(successful, key=lambda x: x[1], reverse=True)
        
        medals = ['🥇', '🥈', '🥉']
        
        for i, (strategy, signal_count) in enumerate(ranked):
            medal = medals[i] if i < len(medals) else f"{i+1:2d}."
            self.logger.info(f"   {medal} {strategy.upper()}: {signal_count} signals")
    
    def _display_recommendations(
        self, 
        successful: List[Tuple[str, int]], 
        failed: List[Tuple[str, str]], 
        epic: str, 
        days: int
    ) -> None:
        """Display recommendations based on results"""
        
        self.logger.info(f"\n💡 RECOMMENDATIONS:")
        self.logger.info("-" * 20)
        
        if not successful:
            self.logger.info("   ⚠️ No strategies generated signals!")
            self.logger.info("   Consider:")
            self.logger.info("     • Lowering confidence thresholds")
            self.logger.info("     • Increasing backtest period")
            self.logger.info("     • Checking market volatility")
            return
        
        # Find best performing strategy
        best_strategy, best_count = max(successful, key=lambda x: x[1])
        
        if best_count == 0:
            self.logger.info("   ⚠️ All strategies found 0 signals")
            self.logger.info("   Market may be in consolidation phase")
        elif best_count < 5:
            self.logger.info(f"   📊 Low signal activity ({best_count} max)")
            self.logger.info("   Consider extending backtest period")
        else:
            self.logger.info(f"   ✅ {best_strategy.upper()} performed best ({best_count} signals)")
            
            # Check if combined strategy is among top performers
            combined_signals = next((count for strategy, count in successful if strategy == 'combined'), 0)
            
            if combined_signals > 0:
                self.logger.info(f"   🎯 Combined strategy: {combined_signals} signals")
                if combined_signals >= best_count * 0.8:  # Within 80% of best
                    self.logger.info("   👍 Combined strategy is competitive")
                else:
                    self.logger.info("   💭 Consider adjusting combined strategy weights")
        
        # Check for failed strategies
        if failed:
            self.logger.info(f"\n   ⚠️ {len(failed)} strategies failed:")
            for strategy, error in failed:
                self.logger.info(f"     • {strategy}: {error}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='All Strategies Comparison Backtest')
    
    # Required arguments
    parser.add_argument('--epic', help='Epic to backtest (e.g., CS.D.EURUSD.CEEM.IP)')
    parser.add_argument('--days', type=int, default=7, help='Days to backtest (default: 7)')
    parser.add_argument('--timeframe', default='15m', help='Timeframe (default: 15m)')
    
    # Optional arguments
    parser.add_argument('--strategies', nargs='+', 
                       help='Specific strategies to test (ema, macd, kama, bb_supertrend, combined)')
    parser.add_argument('--parallel', action='store_true', 
                       help='Run backtests in parallel (experimental)')
    parser.add_argument('--show-details', action='store_true', 
                       help='Show detailed output for supported strategies')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate strategies if specified
    available_strategies = ['ema', 'macd', 'kama', 'bb_supertrend', 'zero_lag', 'combined', 'scalping']
    if args.strategies:
        invalid_strategies = [s for s in args.strategies if s not in available_strategies]
        if invalid_strategies:
            print(f"❌ Invalid strategies: {', '.join(invalid_strategies)}")
            print(f"Available: {', '.join(available_strategies)}")
            sys.exit(1)
    
    # Run comparison backtest
    backtest = AllStrategiesBacktest()
    
    success = backtest.run_comparison_backtest(
        epic=args.epic,
        days=args.days,
        timeframe=args.timeframe,
        strategies=args.strategies,
        parallel=args.parallel,
        show_details=args.show_details
    )
    
    if success:
        print("\n✅ Strategy comparison completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Strategy comparison failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()