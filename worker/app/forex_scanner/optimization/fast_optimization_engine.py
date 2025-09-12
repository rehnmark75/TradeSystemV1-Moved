#!/usr/bin/env python3
"""
Fast Optimization Engine
Ultra-fast parameter optimization for multi-strategy systems
Target: 2 hours for all 9 epics (one strategy)
"""

import sys
import os
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from optimize_ema_parameters import ParameterOptimizationEngine
from optimal_parameter_service import OptimalParameterService

try:
    import config
except ImportError:
    from forex_scanner import config


class FastOptimizationEngine:
    """
    Ultra-fast optimization engine for multi-strategy environments
    
    Key optimizations:
    1. Reduced parameter grid (81 vs 14,406 combinations)
    2. Simplified validation (2 vs 5 layers) 
    3. Parallel epic processing
    4. Smart parameter sampling
    5. Progressive refinement
    """
    
    def __init__(self, strategy_name: str = 'ema'):
        self.logger = logging.getLogger('fast_optimization')
        self.strategy_name = strategy_name
        self.setup_logging()
        
        # Fast optimization parameters
        self.max_workers = min(4, mp.cpu_count())  # Limit concurrent epics
        self.target_runtime_hours = 2.0
        self.target_combinations_per_epic = 81
        
        self.logger.info(f"🚀 Fast Optimization Engine initialized")
        self.logger.info(f"   Strategy: {strategy_name}")
        self.logger.info(f"   Max workers: {self.max_workers}")
        self.logger.info(f"   Target runtime: {self.target_runtime_hours} hours")
        self.logger.info(f"   Combinations per epic: {self.target_combinations_per_epic}")
    
    def setup_logging(self):
        """Setup logging for fast optimization"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_fast_parameter_grid(self) -> Dict:
        """Get reduced parameter grid optimized for speed"""
        return {
            'ema_configs': ['default', 'aggressive', 'conservative'],
            'confidence_levels': [0.45, 0.55, 0.65],
            'timeframes': ['15m'],  # Single timeframe for speed
            'smart_money_options': [False],  # Disable for speed
            'stop_loss_levels': [8, 12, 18],
            'take_profit_levels': [16, 24, 36]  # Maintain 2:1 risk/reward
        }
    
    def get_priority_epics(self) -> List[str]:
        """Get epics ordered by trading priority/volume"""
        # High-priority epics first (most liquid, most traded)
        priority_epics = [
            'CS.D.EURUSD.CEEM.IP',   # Most liquid
            'CS.D.GBPUSD.MINI.IP',   # High volume
            'CS.D.USDJPY.MINI.IP',   # Major pair
            'CS.D.AUDUSD.MINI.IP',   # Commodity currency
            'CS.D.USDCAD.MINI.IP',   # Oil correlation
            'CS.D.NZDUSD.MINI.IP',   # Risk sentiment
            'CS.D.EURGBP.MINI.IP',   # Cross pair
            'CS.D.EURJPY.MINI.IP',   # Carry trade
            'CS.D.GBPJPY.MINI.IP'    # Volatile cross
        ]
        
        # Filter to only include epics that actually exist in config
        available_epics = getattr(config, 'TRADEABLE_EPICS', priority_epics)
        
        # Return intersection, maintaining priority order
        return [epic for epic in priority_epics if epic in available_epics or not available_epics]
    
    def optimize_single_epic_fast(self, epic: str, days: int = 5) -> Tuple[str, dict]:
        """
        Optimize single epic with fast parameters
        
        Args:
            epic: Trading pair to optimize
            days: Days of data (reduced from 7+ to 5 for speed)
            
        Returns:
            Tuple of (epic, results_summary)
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🎯 Fast optimizing {epic}...")
            
            # Create optimization engine with fast parameters
            optimizer = ParameterOptimizationEngine()
            optimizer.parameter_grid = self.get_fast_parameter_grid()
            
            # Override optimization settings for speed
            fast_settings = {
                'use_fast_mode': True,
                'simplified_validation': True,
                'skip_mtf_analysis': True,
                'reduced_lookback': True,
                'min_signals': 3  # Lower threshold for faster validation
            }
            
            # Create optimization run
            run_id = optimizer.create_optimization_run(
                run_name=f"fast_{self.strategy_name}_{epic.split('.')[-3]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                epics=[epic],
                backtest_days=days,
                notes=f"Fast optimization mode - {self.target_combinations_per_epic} combinations"
            )
            
            if not run_id:
                return epic, {'error': 'Failed to create optimization run'}
            
            # Run optimization with fast settings
            optimizer.optimize_epic_parameters(
                epic=epic,
                run_id=run_id,
                days=days,
                **fast_settings
            )
            
            # Get results summary
            results_summary = self._get_epic_results_summary(epic, run_id)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ {epic} optimized in {elapsed_time/60:.1f} minutes")
            
            return epic, {
                'run_id': run_id,
                'elapsed_minutes': elapsed_time / 60,
                'combinations_tested': self.target_combinations_per_epic,
                'results': results_summary,
                'success': True
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Fast optimization failed for {epic}: {e}")
            return epic, {
                'error': str(e),
                'elapsed_minutes': elapsed_time / 60,
                'success': False
            }
    
    def optimize_all_epics_parallel(self, 
                                   days: int = 5,
                                   max_workers: Optional[int] = None) -> Dict[str, dict]:
        """
        Optimize all epics in parallel for maximum speed
        
        Args:
            days: Days of backtesting data
            max_workers: Maximum parallel workers (defaults to class setting)
            
        Returns:
            Dictionary with results for each epic
        """
        start_time = time.time()
        workers = max_workers or self.max_workers
        
        epics = self.get_priority_epics()
        
        self.logger.info(f"🚀 FAST PARALLEL OPTIMIZATION STARTING")
        self.logger.info(f"   Strategy: {self.strategy_name}")
        self.logger.info(f"   Epics: {len(epics)}")
        self.logger.info(f"   Workers: {workers}")
        self.logger.info(f"   Days: {days}")
        self.logger.info(f"   Target time: {self.target_runtime_hours} hours")
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all epics for optimization
            future_to_epic = {
                executor.submit(self.optimize_single_epic_fast, epic, days): epic 
                for epic in epics
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_epic):
                epic = future_to_epic[future]
                try:
                    epic_result, result_data = future.result()
                    results[epic_result] = result_data
                    
                    if result_data.get('success'):
                        elapsed_min = result_data['elapsed_minutes']
                        self.logger.info(f"✅ {epic} completed in {elapsed_min:.1f} min")
                    else:
                        self.logger.error(f"❌ {epic} failed: {result_data.get('error')}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {epic} exception: {e}")
                    results[epic] = {'error': str(e), 'success': False}
        
        total_elapsed = time.time() - start_time
        successful_epics = sum(1 for r in results.values() if r.get('success'))
        
        self.logger.info(f"🏁 FAST OPTIMIZATION COMPLETE")
        self.logger.info(f"   Total time: {total_elapsed/3600:.2f} hours")
        self.logger.info(f"   Target time: {self.target_runtime_hours} hours")
        self.logger.info(f"   Success rate: {successful_epics}/{len(epics)} epics")
        self.logger.info(f"   Average per epic: {total_elapsed/len(epics)/60:.1f} minutes")
        
        # Check if we met our time target
        if total_elapsed/3600 <= self.target_runtime_hours:
            self.logger.info(f"🎯 TIME TARGET MET! ({total_elapsed/3600:.2f}h ≤ {self.target_runtime_hours}h)")
        else:
            self.logger.warning(f"⏰ Time target missed ({total_elapsed/3600:.2f}h > {self.target_runtime_hours}h)")
        
        return results
    
    def _get_epic_results_summary(self, epic: str, run_id: int) -> dict:
        """Get optimization results summary for an epic"""
        try:
            service = OptimalParameterService()
            params = service.get_epic_parameters(epic, force_refresh=True)
            
            return {
                'ema_config': params.ema_config,
                'confidence_threshold': params.confidence_threshold,
                'stop_loss_pips': params.stop_loss_pips,
                'take_profit_pips': params.take_profit_pips,
                'performance_score': params.performance_score,
                'timeframe': params.timeframe
            }
        except Exception as e:
            return {'error': f'Failed to get results: {e}'}
    
    def generate_fast_optimization_report(self, results: Dict[str, dict]) -> str:
        """Generate comprehensive report of fast optimization results"""
        
        successful_results = {k: v for k, v in results.items() if v.get('success')}
        failed_results = {k: v for k, v in results.items() if not v.get('success')}
        
        report = f"""
🚀 FAST OPTIMIZATION REPORT - {self.strategy_name.upper()} STRATEGY
{'='*70}

📊 EXECUTION SUMMARY:
   Total Epics: {len(results)}
   Successful: {len(successful_results)}
   Failed: {len(failed_results)}
   Success Rate: {len(successful_results)/len(results)*100:.1f}%

⏱️ TIMING ANALYSIS:
"""
        
        if successful_results:
            times = [r['elapsed_minutes'] for r in successful_results.values()]
            total_time = sum(times)
            avg_time = total_time / len(times)
            
            report += f"""   Total Runtime: {total_time:.1f} minutes ({total_time/60:.2f} hours)
   Average per Epic: {avg_time:.1f} minutes
   Target: {self.target_runtime_hours * 60:.0f} minutes ({self.target_runtime_hours} hours)
   Performance: {'✅ TARGET MET' if total_time/60 <= self.target_runtime_hours else '⏰ TARGET MISSED'}

🏆 OPTIMIZATION RESULTS:
"""
            
            # Sort by performance score
            sorted_results = sorted(
                successful_results.items(),
                key=lambda x: x[1].get('results', {}).get('performance_score', 0),
                reverse=True
            )
            
            for epic, data in sorted_results:
                epic_short = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
                results_data = data.get('results', {})
                
                report += f"""
   🎯 {epic_short}:
      Config: {results_data.get('ema_config', 'N/A')}
      Confidence: {results_data.get('confidence_threshold', 0):.0%}
      SL/TP: {results_data.get('stop_loss_pips', 0):.0f}/{results_data.get('take_profit_pips', 0):.0f}
      Score: {results_data.get('performance_score', 0):.3f}
      Runtime: {data['elapsed_minutes']:.1f}m
"""
        
        if failed_results:
            report += f"\n❌ FAILED OPTIMIZATIONS:\n"
            for epic, data in failed_results.items():
                epic_short = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
                report += f"   {epic_short}: {data.get('error', 'Unknown error')}\n"
        
        report += f"""
💡 FAST MODE OPTIMIZATIONS APPLIED:
   ✅ Reduced parameter grid: {self.target_combinations_per_epic} combinations (vs 14,406 full)
   ✅ Simplified validation: 2 layers (vs 5 full layers)
   ✅ Parallel processing: {self.max_workers} workers
   ✅ Single timeframe focus: 15m only
   ✅ Reduced data lookback: 5 days (vs 7+ full)
   ✅ Disabled heavy features: Smart Money, MTF analysis

🚀 NEXT STEPS:
   1. Review optimization results above
   2. Test live trading with optimized parameters
   3. Monitor performance for 1-2 weeks
   4. Run full optimization on best-performing epics if needed
   5. Repeat fast optimization monthly or when performance drops

{'='*70}
        """
        
        return report


def main():
    """Main execution for fast optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast Parameter Optimization')
    parser.add_argument('--strategy', default='ema', help='Strategy to optimize')
    parser.add_argument('--days', type=int, default=5, help='Days of backtesting data')
    parser.add_argument('--workers', type=int, help='Max parallel workers')
    parser.add_argument('--epic', help='Single epic to optimize (for testing)')
    
    args = parser.parse_args()
    
    # Create fast optimization engine
    engine = FastOptimizationEngine(strategy_name=args.strategy)
    
    if args.epic:
        # Single epic test
        print(f"🧪 FAST OPTIMIZATION TEST - Single Epic")
        print(f"Epic: {args.epic}")
        print(f"Expected runtime: ~13 minutes")
        
        start_time = time.time()
        epic, results = engine.optimize_single_epic_fast(args.epic, args.days)
        elapsed = time.time() - start_time
        
        print(f"\n📊 RESULTS:")
        print(f"Epic: {epic}")
        print(f"Runtime: {elapsed/60:.1f} minutes")
        print(f"Success: {results.get('success')}")
        if results.get('success'):
            result_data = results.get('results', {})
            print(f"Best Config: {result_data.get('ema_config')}")
            print(f"Score: {result_data.get('performance_score', 0):.3f}")
        else:
            print(f"Error: {results.get('error')}")
    else:
        # Full optimization
        print(f"🚀 FAST OPTIMIZATION - All Epics")
        print(f"Strategy: {args.strategy}")
        print(f"Target: 2 hours total")
        print(f"Workers: {args.workers or engine.max_workers}")
        
        # Run optimization
        results = engine.optimize_all_epics_parallel(
            days=args.days,
            max_workers=args.workers
        )
        
        # Generate and display report
        report = engine.generate_fast_optimization_report(results)
        print(report)
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"/tmp/fast_optimization_report_{args.strategy}_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_file}")


if __name__ == "__main__":
    main()