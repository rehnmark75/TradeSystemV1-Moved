#!/usr/bin/env python3
"""
Epic Parameter Optimizer
Comprehensive EMA strategy parameter optimization system with risk management
"""

import sys
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import itertools
import pandas as pd
from decimal import Decimal
import json

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.strategies.ema_strategy import EMAStrategy
from forex_scanner.backtests.performance_analyzer import PerformanceAnalyzer

# Import backtest engine
from backtests.backtest_ema import EMABacktest

# Configuration imports
from configdata import config as strategy_config
try:
    import config
except ImportError:
    from forex_scanner import config


class EpicParameterOptimizer:
    """
    Comprehensive parameter optimization system for EMA strategy
    Tests all parameter combinations to find optimal settings per epic
    """
    
    def __init__(self, database_url: str = None, timezone: str = 'UTC'):
        """Initialize optimizer with database connection"""
        self.logger = logging.getLogger('epic_optimizer')
        self.setup_logging()
        
        # Database connection
        db_url = database_url or config.DATABASE_URL
        self.db_manager = DatabaseManager(db_url)
        
        # Initialize backtest engine
        self.backtest_engine = EMABacktest()
        
        # Optimization state
        self.current_run_id = None
        self.optimization_stats = {
            'start_time': None,
            'total_combinations': 0,
            'completed_combinations': 0,
            'failed_combinations': 0,
            'best_results': {}
        }
        
        self.logger.info("✅ Epic Parameter Optimizer initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def create_optimization_run(self, run_name: str, description: str, epics: List[str], 
                              backtest_days: int = 30) -> int:
        """Create new optimization run record"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate total combinations
                total_combinations = self._calculate_total_combinations(epics)
                
                cursor.execute("""
                    INSERT INTO ema_optimization_runs 
                    (run_name, description, total_combinations, epics_tested, backtest_days, status)
                    VALUES (%s, %s, %s, %s, %s, 'running')
                    RETURNING id
                """, (run_name, description, total_combinations, epics, backtest_days))
                
                run_id = cursor.fetchone()[0]
                conn.commit()
                
                self.current_run_id = run_id
                self.logger.info(f"📊 Created optimization run {run_id}: {run_name}")
                self.logger.info(f"   Total combinations to test: {total_combinations:,}")
                
                return run_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create optimization run: {e}")
            raise
    
    def get_parameter_grid(self) -> Dict:
        """Define comprehensive parameter grid for testing"""
        return {
            # Strategy Parameters
            'ema_configs': ['default', 'conservative', 'aggressive', 'scalping', 'swing', 'news_safe', 'crypto'],
            'confidence_levels': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            'timeframes': ['5m', '15m', '1h'],
            'smart_money_options': [True, False],
            
            # Risk Management Parameters
            'stop_loss_levels': [5, 8, 10, 12, 15, 20, 25],  # pips
            'take_profit_levels': [10, 15, 20, 25, 30, 40, 50],  # pips
            'risk_reward_ratios': [1.5, 2.0, 2.5, 3.0]  # R:R ratios
        }
    
    def _calculate_total_combinations(self, epics: List[str]) -> int:
        """Calculate total parameter combinations to test"""
        grid = self.get_parameter_grid()
        
        # Calculate combinations for strategy parameters
        strategy_combinations = (
            len(grid['ema_configs']) * 
            len(grid['confidence_levels']) * 
            len(grid['timeframes']) * 
            len(grid['smart_money_options'])
        )
        
        # Calculate risk management combinations
        risk_combinations = (
            len(grid['stop_loss_levels']) * 
            len(grid['take_profit_levels'])
        )
        
        # Total per epic
        per_epic_total = strategy_combinations * risk_combinations
        
        # Total across all epics
        total = len(epics) * per_epic_total
        
        self.logger.info(f"📊 Parameter grid calculations:")
        self.logger.info(f"   Strategy combinations: {strategy_combinations:,}")
        self.logger.info(f"   Risk management combinations: {risk_combinations:,}")
        self.logger.info(f"   Per epic total: {per_epic_total:,}")
        self.logger.info(f"   Total across {len(epics)} epics: {total:,}")
        
        return total
    
    def optimize_epic(self, epic: str, backtest_days: int = 30, 
                     min_signals_threshold: int = 20) -> Dict:
        """
        Run comprehensive optimization for a single epic
        
        Args:
            epic: Trading pair to optimize
            backtest_days: Number of days to backtest
            min_signals_threshold: Minimum signals required for valid result
            
        Returns:
            Dictionary with best parameters and performance metrics
        """
        self.logger.info(f"\n🎯 OPTIMIZING {epic}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        grid = self.get_parameter_grid()
        epic_results = []
        
        # Generate all parameter combinations
        combinations = list(itertools.product(
            grid['ema_configs'],
            grid['confidence_levels'],
            grid['timeframes'],
            grid['smart_money_options'],
            grid['stop_loss_levels'],
            grid['take_profit_levels']
        ))
        
        total_combinations = len(combinations)
        self.logger.info(f"📊 Testing {total_combinations:,} parameter combinations for {epic}")
        
        # Test each combination
        for i, (ema_config, confidence, timeframe, smart_money, stop_loss, take_profit) in enumerate(combinations):
            try:
                # Calculate risk-reward ratio
                risk_reward_ratio = take_profit / stop_loss if stop_loss > 0 else 0
                
                # Skip if R:R is too low
                if risk_reward_ratio < 1.2:
                    continue
                
                # Show progress
                if (i + 1) % 100 == 0 or i == 0:
                    progress = ((i + 1) / total_combinations) * 100
                    self.logger.info(f"   Progress: {progress:.1f}% ({i+1:,}/{total_combinations:,})")
                
                # Run single parameter test
                result = self._run_single_test(
                    epic=epic,
                    ema_config=ema_config,
                    confidence_threshold=confidence,
                    timeframe=timeframe,
                    smart_money_enabled=smart_money,
                    stop_loss_pips=stop_loss,
                    take_profit_pips=take_profit,
                    backtest_days=backtest_days
                )
                
                if result and result.get('total_signals', 0) >= min_signals_threshold:
                    # Store result in database
                    self._store_optimization_result(result)
                    epic_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error testing combination {i+1}: {e}")
                self.optimization_stats['failed_combinations'] += 1
                continue
        
        # Find best result for this epic
        best_result = self._find_best_result(epic_results)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"\n✅ {epic} optimization completed in {elapsed_time/60:.1f} minutes")
        self.logger.info(f"   Valid results: {len(epic_results)}")
        
        if best_result:
            self.logger.info(f"   🏆 Best configuration:")
            self.logger.info(f"      EMA Config: {best_result['ema_config']}")
            self.logger.info(f"      Confidence: {best_result['confidence_threshold']:.1%}")
            self.logger.info(f"      Timeframe: {best_result['timeframe']}")
            self.logger.info(f"      Stop Loss: {best_result['stop_loss_pips']} pips")
            self.logger.info(f"      Take Profit: {best_result['take_profit_pips']} pips")
            self.logger.info(f"      Win Rate: {best_result['win_rate']:.1%}")
            self.logger.info(f"      Profit Factor: {best_result['profit_factor']:.2f}")
            self.logger.info(f"      Net Pips: {best_result['net_pips']:.1f}")
            
            # Store best parameters
            self._store_best_parameters(epic, best_result)
        
        return best_result or {}
    
    def _run_single_test(self, epic: str, ema_config: str, confidence_threshold: float,
                        timeframe: str, smart_money_enabled: bool, stop_loss_pips: float,
                        take_profit_pips: float, backtest_days: int) -> Optional[Dict]:
        """Run backtest for single parameter combination"""
        try:
            # Create modified backtest with specific risk parameters
            backtest = EMABacktest()
            
            # Temporarily override confidence threshold
            original_confidence = getattr(config, 'MIN_CONFIDENCE', 0.45)
            config.MIN_CONFIDENCE = confidence_threshold
            
            # Initialize EMA strategy with specific config
            backtest.initialize_ema_strategy(ema_config)
            
            # Run backtest with Smart Money if enabled
            success = backtest.run_backtest(
                epic=epic,
                days=backtest_days,
                timeframe=timeframe,
                show_signals=False,
                ema_config=ema_config,
                min_confidence=confidence_threshold,
                enable_smart_money=smart_money_enabled
            )
            
            # Restore original confidence
            config.MIN_CONFIDENCE = original_confidence
            
            if not success:
                return None
            
            # Extract signals from backtest (this would need to be added to EMABacktest)
            signals = getattr(backtest, '_last_signals', [])
            
            if not signals:
                return None
            
            # Calculate performance with custom risk parameters
            performance = self._calculate_performance_metrics(
                signals=signals,
                epic=epic,
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips
            )
            
            # Combine parameters and performance
            result = {
                'epic': epic,
                'ema_config': ema_config,
                'confidence_threshold': confidence_threshold,
                'timeframe': timeframe,
                'smart_money_enabled': smart_money_enabled,
                'stop_loss_pips': stop_loss_pips,
                'take_profit_pips': take_profit_pips,
                'risk_reward_ratio': take_profit_pips / stop_loss_pips,
                **performance
            }
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Single test failed for {epic}: {e}")
            return None
    
    def _calculate_performance_metrics(self, signals: List[Dict], epic: str, 
                                     stop_loss_pips: float, take_profit_pips: float) -> Dict:
        """Calculate comprehensive performance metrics with custom risk parameters"""
        if not signals:
            return self._empty_performance_metrics()
        
        # Recalculate performance with custom stop/target levels
        total_signals = len(signals)
        winners = []
        losers = []
        total_profit = 0.0
        total_loss = 0.0
        
        # Adjust for JPY pairs
        is_jpy_pair = any(jpy in epic.upper() for jpy in ['JPY'])
        pip_multiplier = 100 if is_jpy_pair else 10000
        
        for signal in signals:
            # Get max profit/loss from signal (already calculated in backtest)
            max_profit = signal.get('max_profit_pips', 0)
            max_loss = signal.get('max_loss_pips', 0)
            
            # Determine if this signal would hit stop or target with our custom levels
            if max_profit >= take_profit_pips:
                # Would hit take profit
                winners.append(signal)
                total_profit += take_profit_pips
            elif max_loss >= stop_loss_pips:
                # Would hit stop loss
                losers.append(signal)
                total_loss += stop_loss_pips
            else:
                # Would timeout - use actual result
                if max_profit > max_loss:
                    winners.append(signal)
                    total_profit += max_profit
                else:
                    losers.append(signal)
                    total_loss += max_loss
        
        # Calculate metrics
        win_rate = len(winners) / total_signals if total_signals > 0 else 0
        loss_rate = len(losers) / total_signals if total_signals > 0 else 0
        avg_profit = total_profit / len(winners) if winners else 0
        avg_loss = total_loss / len(losers) if losers else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        net_pips = total_profit - total_loss
        expectancy = net_pips / total_signals if total_signals > 0 else 0
        
        # Calculate composite score
        composite_score = win_rate * profit_factor * (net_pips / 100.0) if profit_factor > 0 else 0
        
        return {
            'total_signals': total_signals,
            'valid_signals': total_signals,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_profit_pips': avg_profit,
            'avg_loss_pips': avg_loss,
            'total_profit_pips': total_profit,
            'total_loss_pips': total_loss,
            'net_pips': net_pips,
            'profit_factor': profit_factor,
            'expectancy_per_trade': expectancy,
            'risk_reward_achieved': avg_profit / avg_loss if avg_loss > 0 else 0,
            'composite_score': composite_score,
            'bull_signals': len([s for s in signals if s.get('signal_type', '').upper() in ['BUY', 'BULL', 'LONG']]),
            'bear_signals': len([s for s in signals if s.get('signal_type', '').upper() in ['SELL', 'BEAR', 'SHORT']]),
            'profit_target_exits': len(winners),
            'stop_loss_exits': len(losers)
        }
    
    def _empty_performance_metrics(self) -> Dict:
        """Return empty performance metrics structure"""
        return {
            'total_signals': 0,
            'valid_signals': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'avg_profit_pips': 0.0,
            'avg_loss_pips': 0.0,
            'total_profit_pips': 0.0,
            'total_loss_pips': 0.0,
            'net_pips': 0.0,
            'profit_factor': 0.0,
            'expectancy_per_trade': 0.0,
            'risk_reward_achieved': 0.0,
            'composite_score': 0.0,
            'bull_signals': 0,
            'bear_signals': 0,
            'profit_target_exits': 0,
            'stop_loss_exits': 0
        }
    
    def _store_optimization_result(self, result: Dict) -> None:
        """Store single optimization result in database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ema_optimization_results (
                        run_id, epic, ema_config, confidence_threshold, timeframe, smart_money_enabled,
                        stop_loss_pips, take_profit_pips, risk_reward_ratio,
                        total_signals, win_rate, profit_factor, net_pips, composite_score,
                        avg_profit_pips, avg_loss_pips, total_profit_pips, total_loss_pips,
                        expectancy_per_trade, risk_reward_achieved,
                        bull_signals, bear_signals, profit_target_exits, stop_loss_exits
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    self.current_run_id,
                    result['epic'],
                    result['ema_config'],
                    result['confidence_threshold'],
                    result['timeframe'],
                    result['smart_money_enabled'],
                    result['stop_loss_pips'],
                    result['take_profit_pips'],
                    result['risk_reward_ratio'],
                    result['total_signals'],
                    result['win_rate'],
                    result['profit_factor'],
                    result['net_pips'],
                    result['composite_score'],
                    result['avg_profit_pips'],
                    result['avg_loss_pips'],
                    result['total_profit_pips'],
                    result['total_loss_pips'],
                    result['expectancy_per_trade'],
                    result['risk_reward_achieved'],
                    result['bull_signals'],
                    result['bear_signals'],
                    result['profit_target_exits'],
                    result['stop_loss_exits']
                ))
                
                conn.commit()
                self.optimization_stats['completed_combinations'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store result: {e}")
    
    def _find_best_result(self, results: List[Dict]) -> Optional[Dict]:
        """Find best result based on composite score"""
        if not results:
            return None
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        return sorted_results[0]
    
    def _store_best_parameters(self, epic: str, best_result: Dict) -> None:
        """Store best parameters for epic"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ema_best_parameters (
                        epic, best_ema_config, best_confidence_threshold, best_timeframe,
                        optimal_stop_loss_pips, optimal_take_profit_pips,
                        best_win_rate, best_profit_factor, best_net_pips
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (epic) DO UPDATE SET
                        best_ema_config = EXCLUDED.best_ema_config,
                        best_confidence_threshold = EXCLUDED.best_confidence_threshold,
                        best_timeframe = EXCLUDED.best_timeframe,
                        optimal_stop_loss_pips = EXCLUDED.optimal_stop_loss_pips,
                        optimal_take_profit_pips = EXCLUDED.optimal_take_profit_pips,
                        best_win_rate = EXCLUDED.best_win_rate,
                        best_profit_factor = EXCLUDED.best_profit_factor,
                        best_net_pips = EXCLUDED.best_net_pips,
                        last_updated = NOW()
                """, (
                    epic,
                    best_result['ema_config'],
                    best_result['confidence_threshold'],
                    best_result['timeframe'],
                    best_result['stop_loss_pips'],
                    best_result['take_profit_pips'],
                    best_result['win_rate'],
                    best_result['profit_factor'],
                    best_result['net_pips']
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store best parameters: {e}")
    
    def optimize_all_epics(self, epics: List[str] = None, backtest_days: int = 30,
                          run_name: str = None) -> Dict:
        """
        Run optimization for all specified epics
        
        Args:
            epics: List of epics to optimize (defaults to config.EPIC_LIST)
            backtest_days: Days to backtest
            run_name: Name for this optimization run
            
        Returns:
            Dictionary with optimization summary
        """
        # Use default epics if not specified
        if epics is None:
            epics = config.EPIC_LIST
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"full_optimization_{timestamp}"
        
        self.logger.info("🚀 STARTING FULL EPIC OPTIMIZATION")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Epics to optimize: {len(epics)}")
        self.logger.info(f"📅 Backtest period: {backtest_days} days")
        self.logger.info(f"📋 Run name: {run_name}")
        
        # Create optimization run
        run_id = self.create_optimization_run(
            run_name=run_name,
            description=f"Full optimization of {len(epics)} epics over {backtest_days} days",
            epics=epics,
            backtest_days=backtest_days
        )
        
        # Track overall results
        optimization_summary = {
            'run_id': run_id,
            'start_time': datetime.now(),
            'epics_optimized': {},
            'total_epics': len(epics),
            'completed_epics': 0,
            'failed_epics': 0
        }
        
        # Optimize each epic
        for i, epic in enumerate(epics):
            try:
                self.logger.info(f"\n📈 Epic {i+1}/{len(epics)}: {epic}")
                
                # Run optimization for this epic
                best_result = self.optimize_epic(
                    epic=epic,
                    backtest_days=backtest_days,
                    min_signals_threshold=20
                )
                
                optimization_summary['epics_optimized'][epic] = best_result
                optimization_summary['completed_epics'] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Failed to optimize {epic}: {e}")
                optimization_summary['failed_epics'] += 1
                optimization_summary['epics_optimized'][epic] = {'error': str(e)}
        
        # Complete optimization run
        self._complete_optimization_run(run_id)
        
        optimization_summary['end_time'] = datetime.now()
        optimization_summary['total_duration'] = optimization_summary['end_time'] - optimization_summary['start_time']
        
        # Log final summary
        self.logger.info("\n🏁 OPTIMIZATION COMPLETE!")
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Successfully optimized: {optimization_summary['completed_epics']}/{optimization_summary['total_epics']} epics")
        self.logger.info(f"❌ Failed optimizations: {optimization_summary['failed_epics']}")
        self.logger.info(f"⏱️ Total duration: {optimization_summary['total_duration']}")
        
        return optimization_summary
    
    def _complete_optimization_run(self, run_id: int) -> None:
        """Mark optimization run as completed"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE ema_optimization_runs 
                    SET end_time = NOW(), status = 'completed'
                    WHERE id = %s
                """, (run_id,))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to complete optimization run: {e}")


def main():
    """Main execution for standalone optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Epic Parameter Optimizer')
    parser.add_argument('--epic', help='Single epic to optimize')
    parser.add_argument('--epics', nargs='+', help='Multiple epics to optimize')
    parser.add_argument('--days', type=int, default=30, help='Backtest days')
    parser.add_argument('--run-name', help='Custom run name')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize optimizer
    optimizer = EpicParameterOptimizer()
    
    try:
        if args.epic:
            # Optimize single epic
            result = optimizer.optimize_epic(args.epic, args.days)
            print(f"\n✅ Optimization complete for {args.epic}")
            
        elif args.epics:
            # Optimize specified epics
            summary = optimizer.optimize_all_epics(args.epics, args.days, args.run_name)
            print(f"\n✅ Optimization complete for {len(args.epics)} epics")
            
        else:
            # Optimize all epics
            summary = optimizer.optimize_all_epics(backtest_days=args.days, run_name=args.run_name)
            print(f"\n✅ Full optimization complete")
            
    except KeyboardInterrupt:
        print("\n⚠️ Optimization cancelled by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()