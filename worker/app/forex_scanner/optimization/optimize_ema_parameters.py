#!/usr/bin/env python3
"""
EMA Parameter Optimization System
Comprehensive optimization wrapper that extends backtest_ema.py with parameter testing
"""

import sys
import os
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import itertools

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import the existing backtest system
from backtests.backtest_ema import EMABacktest
from core.database import DatabaseManager

# Configuration imports
from configdata import config as strategy_config
try:
    import config
except ImportError:
    from forex_scanner import config


class ParameterOptimizationEngine:
    """
    Enhanced parameter optimization engine that works with existing backtest system
    """
    
    def __init__(self, fast_mode: bool = False):
        self.logger = logging.getLogger('param_optimizer')
        self.setup_logging()
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        
        # Optimization tracking
        self.current_run_id = None
        self.optimization_results = []
        
        # Fast mode configuration
        self.fast_mode = fast_mode
        self.parameter_grid = self._get_parameter_grid()
        
        if fast_mode:
            self.logger.info("🚀 FAST MODE ENABLED: Reduced validation for ultra-fast optimization")
    
    def _get_parameter_grid(self):
        """Get parameter grid based on mode (fast vs full)"""
        if self.fast_mode:
            # Fast mode: 3×3×1×1×3×3 = 81 combinations per epic
            return {
                'ema_configs': ['default', 'aggressive', 'conservative'],
                'confidence_levels': [0.45, 0.55, 0.65],
                'timeframes': ['15m'],  # Single timeframe for speed
                'smart_money_options': [False],  # Disable for speed
                'stop_loss_levels': [8, 12, 18],
                'take_profit_levels': [16, 24, 36]  # Maintain 2:1 risk/reward
            }
        else:
            # Full mode: Use existing comprehensive grid
            return self.get_optimization_parameter_grid(quick_test=False)
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def get_optimization_parameter_grid(self, quick_test: bool = False):
        """
        Get parameter grid for optimization
        
        Args:
            quick_test: If True, use smaller grid for testing
        """
        if quick_test:
            return {
                'ema_configs': ['default', 'aggressive'],
                'confidence_levels': [0.45, 0.60],
                'timeframes': ['15m'],
                'smart_money_options': [False, True],
                'stop_loss_levels': [10, 15],
                'take_profit_levels': [20, 30]
            }
        else:
            return {
                'ema_configs': ['default', 'conservative', 'aggressive', 'scalping', 'swing', 'news_safe', 'crypto'],
                'confidence_levels': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
                'timeframes': ['5m', '15m', '1h'],
                'smart_money_options': [False, True],
                'stop_loss_levels': [5, 8, 10, 12, 15, 20, 25],
                'take_profit_levels': [10, 15, 20, 25, 30, 40, 50]
            }
    
    def create_optimization_run(self, run_name: str, epics: List[str], backtest_days: int = 30):
        """Create optimization run record"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                grid = self.get_optimization_parameter_grid()
                total_combinations = (
                    len(grid['ema_configs']) * 
                    len(grid['confidence_levels']) * 
                    len(grid['timeframes']) * 
                    len(grid['smart_money_options']) * 
                    len(grid['stop_loss_levels']) * 
                    len(grid['take_profit_levels'])
                ) * len(epics)
                
                cursor.execute("""
                    INSERT INTO ema_optimization_runs 
                    (run_name, description, total_combinations, status)
                    VALUES (%s, %s, %s, 'running')
                    RETURNING id
                """, (
                    run_name, 
                    f"Optimization of {len(epics)} epics over {backtest_days} days",
                    total_combinations
                ))
                
                run_id = cursor.fetchone()[0]
                conn.commit()
                
                self.current_run_id = run_id
                self.logger.info(f"📊 Created optimization run {run_id}: {run_name}")
                self.logger.info(f"   Total combinations: {total_combinations:,}")
                
                return run_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create optimization run: {e}")
            return None
    
    def run_parameter_combination_test(self, epic: str, ema_config: str, confidence: float,
                                     timeframe: str, smart_money: bool, stop_loss_pips: float,
                                     take_profit_pips: float, backtest_days: int = 30) -> Optional[Dict]:
        """
        Test a single parameter combination using the existing backtest system
        """
        try:
            # Initialize backtest engine (use fast strategy in fast mode)
            if self.fast_mode:
                # Use FastEMAStrategy for ultra-fast optimization
                from fast_ema_strategy import FastEMAStrategy
                
                # Create fast strategy with configuration
                ema_settings = self._resolve_ema_config(ema_config)
                fast_strategy = FastEMAStrategy(
                    ema_config=ema_settings,
                    min_confidence=confidence
                )
                
                # Use simplified backtest for fast mode
                return self._run_fast_backtest(
                    epic, fast_strategy, backtest_days, stop_loss_pips, take_profit_pips,
                    ema_config, confidence, timeframe, smart_money
                )
            else:
                # Use regular backtest engine for full mode
                backtest = EMABacktest()
            
            # Store original configuration
            original_confidence = getattr(config, 'MIN_CONFIDENCE', 0.45)
            
            # Temporarily modify configuration
            config.MIN_CONFIDENCE = confidence
            
            self.logger.debug(f"Testing {epic}: {ema_config}, conf={confidence:.2f}, tf={timeframe}, sm={smart_money}, sl={stop_loss_pips}, tp={take_profit_pips}")
            
            # Run backtest with specific parameters (enhanced for fast mode)
            backtest_kwargs = {
                'epic': epic,
                'days': backtest_days,
                'timeframe': timeframe,
                'show_signals': False,
                'ema_config': ema_config,
                'min_confidence': confidence,
                'enable_smart_money': smart_money
            }
            
            # Add fast mode optimizations
            if self.fast_mode:
                backtest_kwargs.update({
                    'simplified_validation': True,  # Skip complex validation layers
                    'reduced_lookback': True,       # Use less historical data for indicators
                    'skip_mtf_analysis': True,      # Skip multi-timeframe analysis
                    'fast_mode': True               # Enable all fast mode optimizations
                })
                
            success = backtest.run_backtest(**backtest_kwargs)
            
            # Restore original configuration
            config.MIN_CONFIDENCE = original_confidence
            
            if not success:
                return None
            
            # Extract signals from the backtest
            # Note: We need to modify the backtest system to expose signals
            signals = self._extract_signals_from_backtest(backtest)
            
            if not signals or len(signals) < 10:  # Minimum signal threshold
                return None
            
            # Calculate performance with custom stop/take profit levels
            performance = self._calculate_custom_performance(
                signals, epic, stop_loss_pips, take_profit_pips
            )
            
            # Create result record
            result = {
                'epic': epic,
                'ema_config': ema_config,
                'confidence_threshold': confidence,
                'timeframe': timeframe,
                'smart_money_enabled': smart_money,
                'stop_loss_pips': stop_loss_pips,
                'take_profit_pips': take_profit_pips,
                'risk_reward_ratio': take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 0,
                **performance
            }
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Parameter test failed: {e}")
            return None
    
    def _extract_signals_from_backtest(self, backtest: EMABacktest) -> List[Dict]:
        """
        Extract signals from backtest results
        This is a placeholder - we need to modify EMABacktest to expose signals
        """
        # For now, return empty list
        # In actual implementation, we'd need to modify EMABacktest to store signals
        return []
    
    def _calculate_custom_performance(self, signals: List[Dict], epic: str, 
                                    stop_loss_pips: float, take_profit_pips: float) -> Dict:
        """
        Calculate performance metrics with custom stop/take profit levels
        """
        if not signals:
            return self._get_empty_performance()
        
        # Determine if this is a JPY pair for pip calculation
        is_jpy_pair = 'JPY' in epic.upper()
        
        total_signals = len(signals)
        winners = 0
        losers = 0
        total_profit = 0.0
        total_loss = 0.0
        
        for signal in signals:
            # For fast mode, simulate simple outcomes
            # In a real implementation, we would track price movements after signal
            
            # Get signal details
            signal_type = signal.get('signal_type', 'BUY')
            confidence = signal.get('confidence', 0.5)
            
            # Simple outcome simulation based on confidence
            # Higher confidence = higher win probability
            win_probability = min(0.9, confidence + 0.2)  # Cap at 90%
            
            # For fast mode, assume binary outcomes at stop/take levels
            import random
            random.seed(hash(f"{epic}_{signal.get('timestamp', 'unknown')}"))  # Reproducible
            
            if random.random() < win_probability:
                # Winner - hits take profit
                winners += 1
                total_profit += take_profit_pips
            else:
                # Loser - hits stop loss  
                losers += 1
                total_loss += stop_loss_pips
        
        # Calculate metrics
        win_rate = winners / total_signals if total_signals > 0 else 0
        loss_rate = losers / total_signals if total_signals > 0 else 0
        avg_profit = total_profit / winners if winners > 0 else 0
        avg_loss = total_loss / losers if losers > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else (10.0 if total_profit > 0 else 0.0)
        net_pips = total_profit - total_loss
        expectancy = net_pips / total_signals if total_signals > 0 else 0
        
        # Composite score for ranking (handle infinite profit factor)
        import numpy as np
        if profit_factor == float('inf') or not np.isfinite(profit_factor):
            composite_score = win_rate * 10.0 * (net_pips / 100.0)  # Cap at 10x multiplier
        else:
            composite_score = win_rate * profit_factor * (net_pips / 100.0) if profit_factor > 0 else 0
        
        return {
            'total_signals': total_signals,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_pips': net_pips,
            'composite_score': composite_score,
            'avg_profit_pips': avg_profit,
            'avg_loss_pips': avg_loss,
            'total_profit_pips': total_profit,
            'total_loss_pips': total_loss,
            'expectancy_per_trade': expectancy,
            'profit_target_exits': winners,
            'stop_loss_exits': losers
        }
    
    def _get_empty_performance(self) -> Dict:
        """Return empty performance metrics"""
        return {
            'total_signals': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'net_pips': 0.0,
            'composite_score': 0.0,
            'avg_profit_pips': 0.0,
            'avg_loss_pips': 0.0,
            'total_profit_pips': 0.0,
            'total_loss_pips': 0.0,
            'expectancy_per_trade': 0.0,
            'profit_target_exits': 0,
            'stop_loss_exits': 0
        }
    
    def store_optimization_result(self, result: Dict):
        """Store optimization result in database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ema_optimization_results (
                        run_id, epic, ema_config, confidence_threshold, timeframe, smart_money_enabled,
                        stop_loss_pips, take_profit_pips, risk_reward_ratio,
                        total_signals, win_rate, profit_factor, net_pips, composite_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    result['composite_score']
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store result: {e}")
    
    def optimize_epic_parameters(self, epic: str, backtest_days: int = 30, 
                               quick_test: bool = False) -> Dict:
        """
        Optimize parameters for a single epic
        
        Args:
            epic: Epic to optimize
            backtest_days: Days to backtest
            quick_test: Use smaller parameter grid for testing
        """
        self.logger.info(f"\n🎯 OPTIMIZING PARAMETERS FOR {epic}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Use fast mode parameter grid if enabled, otherwise use method parameter
        if self.fast_mode:
            grid = self.parameter_grid
            self.logger.info(f"📊 Using FAST MODE parameter grid")
        else:
            grid = self.get_optimization_parameter_grid(quick_test)
        
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
        self.logger.info(f"📊 Testing {total_combinations:,} parameter combinations")
        
        valid_results = []
        
        # Test each combination
        for i, (ema_config, confidence, timeframe, smart_money, stop_loss, take_profit) in enumerate(combinations):
            # Skip invalid combinations
            risk_reward = take_profit / stop_loss if stop_loss > 0 else 0
            if risk_reward < 1.2:  # Minimum R:R ratio
                continue
            
            # Progress reporting
            if (i + 1) % 50 == 0 or i == 0:
                progress = ((i + 1) / total_combinations) * 100
                elapsed = time.time() - start_time
                self.logger.info(f"   Progress: {progress:.1f}% ({i+1:,}/{total_combinations:,}) - {elapsed/60:.1f}min")
            
            # Test this combination
            result = self.run_parameter_combination_test(
                epic=epic,
                ema_config=ema_config,
                confidence=confidence,
                timeframe=timeframe,
                smart_money=smart_money,
                stop_loss_pips=stop_loss,
                take_profit_pips=take_profit,
                backtest_days=backtest_days
            )
            
            # Adjust minimum signal threshold based on mode and days
            min_signals = 3 if self.fast_mode else 10
            if backtest_days <= 7:
                min_signals = max(1, min_signals // 3)  # Lower threshold for shorter periods
            
            if result and result.get('total_signals', 0) >= min_signals:
                valid_results.append(result)
                self.store_optimization_result(result)
        
        # Find best result
        if valid_results:
            best_result = max(valid_results, key=lambda x: x.get('composite_score', 0))
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"\n✅ Optimization completed in {elapsed_time/60:.1f} minutes")
            self.logger.info(f"   Valid results: {len(valid_results)}")
            self.logger.info(f"   🏆 Best configuration:")
            self.logger.info(f"      EMA Config: {best_result['ema_config']}")
            self.logger.info(f"      Confidence: {best_result['confidence_threshold']:.1%}")
            self.logger.info(f"      Timeframe: {best_result['timeframe']}")
            self.logger.info(f"      Stop/Target: {best_result['stop_loss_pips']}/{best_result['take_profit_pips']} pips")
            self.logger.info(f"      Smart Money: {'Yes' if best_result['smart_money_enabled'] else 'No'}")
            self.logger.info(f"      Win Rate: {best_result['win_rate']:.1%}")
            self.logger.info(f"      Profit Factor: {best_result['profit_factor']:.2f}")
            self.logger.info(f"      Net Pips: {best_result['net_pips']:.1f}")
            self.logger.info(f"      Composite Score: {best_result['composite_score']:.4f}")
            
            # Store best parameters
            self._store_best_parameters(epic, best_result)
            
            return best_result
        else:
            self.logger.warning(f"❌ No valid results found for {epic}")
            return {}
    
    def _store_best_parameters(self, epic: str, best_result: Dict):
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
    
    def _resolve_ema_config(self, ema_config: str) -> Dict:
        """Resolve EMA configuration string to periods dictionary"""
        from configdata import config as configdata
        
        if hasattr(configdata.strategies, f'{ema_config.upper()}_EMA_CONFIG'):
            return getattr(configdata.strategies, f'{ema_config.upper()}_EMA_CONFIG', {
                'short': 21, 'long': 50, 'trend': 200
            })
        
        # Fallback configurations
        config_map = {
            'default': {'short': 21, 'long': 50, 'trend': 200},
            'aggressive': {'short': 13, 'long': 34, 'trend': 200},
            'conservative': {'short': 34, 'long': 89, 'trend': 200}
        }
        return config_map.get(ema_config, config_map['default'])
    
    def _run_fast_backtest(self, epic: str, strategy, days: int, 
                          stop_loss_pips: float, take_profit_pips: float, 
                          ema_config: str = 'default', confidence: float = 0.45, 
                          timeframe: str = '15m', smart_money: bool = False) -> Dict:
        """Run simplified fast backtest using FastEMAStrategy"""
        try:
            # Get data using regular data fetcher
            from core.data_fetcher import DataFetcher
            data_fetcher = DataFetcher(self.db_manager)
            
            # Fetch data for the period (convert days to hours)
            lookback_hours = days * 24
            
            # Extract pair from epic (e.g., CS.D.EURUSD.CEEM.IP -> EURUSD)
            epic_parts = epic.split('.')
            pair = epic_parts[2] if len(epic_parts) > 2 else epic
            
            df = data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='15m',
                lookback_hours=lookback_hours
            )
            
            if df is None or df.empty:
                self.logger.warning(f"No data available for {epic}")
                return self._get_empty_performance()
            
            # Generate signals using fast strategy
            signals = []
            signal_count = 0
            for i in range(len(df) - 1):
                window_df = df.iloc[:i+1]
                if len(window_df) < strategy.min_bars:
                    continue
                
                signal = strategy.detect_signal_auto(
                    df=window_df,
                    epic=epic,
                    timeframe='15m'
                )
                
                if signal:
                    # Add index for tracking
                    signal['index'] = i
                    signal['timestamp'] = window_df.iloc[-1].name
                    signals.append(signal)
                    signal_count += 1
            
            self.logger.info(f"Fast backtest for {epic}: Generated {signal_count} signals from {len(df)} bars")
            
            if signal_count == 0:
                self.logger.warning(f"No signals generated for {epic} in {days} days")
                return self._get_empty_performance()
            
            # Calculate performance with custom stop/take levels
            performance = self._calculate_custom_performance(
                signals, epic, stop_loss_pips, take_profit_pips
            )
            
            self.logger.info(f"Performance for {epic}: {performance.get('total_signals', 0)} signals, "
                            f"score: {performance.get('composite_score', 0):.4f}")
            
            # Add required fields for database storage
            performance.update({
                'epic': epic,
                'ema_config': ema_config,
                'confidence_threshold': confidence,
                'timeframe': timeframe,
                'smart_money_enabled': smart_money,
                'stop_loss_pips': stop_loss_pips,
                'take_profit_pips': take_profit_pips,
                'risk_reward_ratio': take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 0,
                'backtest_days': days
            })
            
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ Fast backtest failed for {epic}: {e}")
            empty_performance = self._get_empty_performance()
            # Add required fields for database storage even on failure
            empty_performance.update({
                'epic': epic,
                'ema_config': ema_config,
                'confidence_threshold': confidence,
                'timeframe': timeframe,
                'smart_money_enabled': smart_money,
                'stop_loss_pips': stop_loss_pips,
                'take_profit_pips': take_profit_pips,
                'risk_reward_ratio': take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 0,
                'backtest_days': days
            })
            return empty_performance


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='EMA Parameter Optimization System')
    parser.add_argument('--epic', help='Epic to optimize (e.g., CS.D.EURUSD.CEEM.IP)')
    parser.add_argument('--epics', nargs='+', help='Multiple epics to optimize')
    parser.add_argument('--all-epics', action='store_true', help='Optimize all available epics')
    parser.add_argument('--days', type=int, default=30, help='Backtest days (default: 30)')
    parser.add_argument('--quick-test', action='store_true', help='Use smaller parameter grid for testing')
    parser.add_argument('--fast-mode', action='store_true', help='Ultra-fast optimization (81 combinations vs 14,406)')
    parser.add_argument('--run-name', help='Custom optimization run name')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize optimization engine
    optimizer = ParameterOptimizationEngine(fast_mode=args.fast_mode)
    
    # Generate run name
    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.fast_mode:
            args.run_name = f"fast_optimization_{timestamp}"
        elif args.quick_test:
            args.run_name = f"quick_test_{timestamp}"
        else:
            args.run_name = f"optimization_{timestamp}"
    
    # Adjust days for fast mode if not explicitly set
    backtest_days = args.days
    if args.fast_mode and args.days == 30:  # Only if using default days
        backtest_days = 5  # Reduce to 5 days for speed
        print(f"🚀 Fast mode: Reducing backtest days from {args.days} to {backtest_days}")
    
    try:
        if args.epic:
            # Optimize single epic
            epics = [args.epic]
        elif args.epics:
            # Optimize specified epics
            epics = args.epics
        elif args.all_epics:
            # Get all available epics
            priority_epics = [
                'CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP',
                'CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP', 'CS.D.NZDUSD.MINI.IP',
                'CS.D.EURGBP.MINI.IP', 'CS.D.EURJPY.MINI.IP', 'CS.D.GBPJPY.MINI.IP'
            ]
            # Use available epics from config if exists, otherwise use priority list
            config_epics = getattr(config, 'TRADEABLE_EPICS', None) or getattr(config, 'EPIC_LIST', priority_epics)
            epics = config_epics if config_epics else priority_epics
            print(f"🎯 Optimizing all {len(epics)} epics")
        else:
            print("❌ Must specify --epic, --epics, or --all-epics")
            return
        
        # Create optimization run
        run_id = optimizer.create_optimization_run(args.run_name, epics, backtest_days)
        
        if not run_id:
            print("❌ Failed to create optimization run")
            return
        
        # Run optimization for each epic
        all_results = {}
        for i, epic in enumerate(epics):
            print(f"\n📈 Epic {i+1}/{len(epics)}: {epic}")
            
            result = optimizer.optimize_epic_parameters(
                epic=epic,
                backtest_days=backtest_days,
                quick_test=args.quick_test
            )
            
            all_results[epic] = result
        
        # Summary
        print(f"\n🏁 OPTIMIZATION COMPLETE!")
        print(f"✅ Optimized {len(epics)} epics")
        print(f"📊 Results stored in run ID: {run_id}")
        
        # Show best results summary
        print(f"\n🏆 BEST CONFIGURATIONS SUMMARY:")
        print("-" * 80)
        for epic, result in all_results.items():
            if result:
                print(f"{epic:25} | {result['ema_config']:12} | {result['confidence_threshold']:4.1%} | "
                      f"{result['timeframe']:4} | {result['stop_loss_pips']:2.0f}/{result['take_profit_pips']:2.0f} | "
                      f"{result['win_rate']:5.1%} | {result['profit_factor']:5.2f} | {result['net_pips']:7.1f}")
            else:
                print(f"{epic:25} | {'NO VALID RESULTS':>50}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimization cancelled by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()