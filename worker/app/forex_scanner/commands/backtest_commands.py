# commands/backtest_commands.py
"""
Backtest Commands Module
Handles backtesting operations, performance analysis, and EMA configuration comparisons
"""

import logging
from typing import List, Dict, Optional

try:
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector
    from forex_scanner.backtests.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.backtests.signal_analyzer import SignalAnalyzer
    import config
except ImportError:
    try:
        from forex_scanner.core.database import DatabaseManager
        from forex_scanner.core.signal_detector import SignalDetector
        from forex_scanner.core.data_fetcher import DataFetcher
        from forex_scanner.core.scanner import IntelligentForexScanner as ForexScanner
        from forex_scanner import config
    except ImportError as e:
        import sys
        print(f"Warning: Import fallback failed for {sys.modules[__name__]}: {e}")
        pass

class BacktestCommands:
    """Backtest command implementations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        show_signals: bool = False,
        timeframe: str = '15m',
        bb_analysis: bool = False,
        ema_config: str = None
    ) -> bool:
        """Run comprehensive backtesting"""
        
        # Setup epic list
        if epic:
            epic_list = [epic]
            self.logger.info(f"📊 Running backtest for {epic}")
        else:
            epic_list = config.EPIC_LIST
            self.logger.info(f"📊 Running backtest for {len(epic_list)} pairs")
        
        self.logger.info(f"   Timeframe: {timeframe}")
        self.logger.info(f"   Days: {days}")
        self.logger.info(f"   Show signals: {show_signals}")
        
        try:
            # Temporarily switch EMA config if specified
            original_config = None
            if ema_config and hasattr(config, 'ACTIVE_EMA_CONFIG'):
                original_config = config.ACTIVE_EMA_CONFIG
                config.ACTIVE_EMA_CONFIG = ema_config
                self.logger.info(f"   Using EMA config: {ema_config}")
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            analyzer = SignalAnalyzer()
            performance_analyzer = PerformanceAnalyzer()
            
            # Run backtest
            results = detector.backtest_signals(
                epic_list=epic_list,
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe=timeframe
            )
            
            if not results:
                self.logger.warning("❌ No signals found in backtest")
                return False
            
            # Display results
            self.logger.info(f"✅ Backtest complete: {len(results)} signals found")
            
            # Performance analysis
            performance = performance_analyzer.analyze_performance(results)
            
            # Display summary by strategy
            analyzer.display_signal_summary_by_strategy(results)
            
            # Display summary by pair
            analyzer.display_signal_summary_by_pair(results)
            
            # Show detailed signal list if requested
            if show_signals:
                analyzer.display_signal_list(results, max_signals=50)
            
            # Bollinger Band analysis if requested
            if bb_analysis:
                self._analyze_bb_performance(results)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Restore original EMA config
            if original_config and hasattr(config, 'ACTIVE_EMA_CONFIG'):
                config.ACTIVE_EMA_CONFIG = original_config
    
    def compare_ema_configs(
        self, 
        epic: str = None, 
        days: int = 7, 
        timeframe: str = '15m',
        configs: List[str] = None
    ) -> bool:
        """Compare performance across different EMA configurations"""
        
        if not hasattr(config, 'EMA_STRATEGY_CONFIG'):
            self.logger.error("❌ EMA_STRATEGY_CONFIG not found in config")
            return False
        
        # Use provided configs or test all available
        test_configs = configs or list(config.EMA_STRATEGY_CONFIG.keys())
        
        epic_list = [epic] if epic else ['CS.D.EURUSD.CEEM.IP']  # Default to EURUSD
        
        self.logger.info(f"🔄 Comparing EMA configurations: {test_configs}")
        self.logger.info(f"   Epic: {epic_list[0]}")
        self.logger.info(f"   Timeframe: {timeframe}")
        self.logger.info(f"   Days: {days}")
        
        try:
            # Store original config
            original_config = getattr(config, 'ACTIVE_EMA_CONFIG', 'default')
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            performance_analyzer = PerformanceAnalyzer()
            
            comparison_results = {}
            
            # Test each configuration
            for ema_config in test_configs:
                self.logger.info(f"\n📊 Testing {ema_config} configuration...")
                
                # Switch to this config
                config.ACTIVE_EMA_CONFIG = ema_config
                ema_settings = config.EMA_STRATEGY_CONFIG[ema_config]
                
                self.logger.info(f"   EMAs: {ema_settings['short']}/{ema_settings['long']}/{ema_settings['trend']}")
                
                # Run backtest with this config
                results = detector.backtest_signals(
                    epic_list=epic_list,
                    lookback_days=days,
                    use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                    spread_pips=config.SPREAD_PIPS,
                    timeframe=timeframe
                )
                
                # Analyze performance
                performance = performance_analyzer.analyze_performance(results)
                
                comparison_results[ema_config] = {
                    'signals': results,
                    'performance': performance,
                    'ema_settings': ema_settings
                }
                
                # Quick summary
                signal_count = performance.get('total_signals', 0)
                avg_confidence = performance.get('average_confidence', 0)
                win_rate = performance.get('win_rate', 0)
                
                self.logger.info(f"   Results: {signal_count} signals, {avg_confidence:.1%} avg confidence, {win_rate:.1%} win rate")
            
            # Display comparison
            self._display_ema_comparison(comparison_results)
            
            # Restore original config
            config.ACTIVE_EMA_CONFIG = original_config
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ EMA config comparison failed: {e}")
            # Restore original config on error
            if 'original_config' in locals():
                config.ACTIVE_EMA_CONFIG = original_config
            return False
    
    def run_strategy_comparison(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m'
    ) -> bool:
        """Compare performance between individual strategies"""
        
        epic_list = [epic] if epic else ['CS.D.EURUSD.CEEM.IP']
        
        self.logger.info(f"🎯 Comparing strategy performance")
        self.logger.info(f"   Epic: {epic_list[0]}")
        self.logger.info(f"   Days: {days}")
        
        try:
            # Store original settings
            original_ema = getattr(config, 'SIMPLE_EMA_STRATEGY', True)
            original_macd = getattr(config, 'MACD_EMA_STRATEGY', False)
            original_combined = getattr(config, 'COMBINED_STRATEGY_MODE', None)
            
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            performance_analyzer = PerformanceAnalyzer()
            
            strategy_results = {}
            
            # Test EMA only
            self.logger.info(f"\n📈 Testing EMA strategy only...")
            config.SIMPLE_EMA_STRATEGY = True
            config.MACD_EMA_STRATEGY = False
            if hasattr(config, 'COMBINED_STRATEGY_MODE'):
                delattr(config, 'COMBINED_STRATEGY_MODE')
            
            ema_results = detector.backtest_signals(
                epic_list=epic_list,
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe=timeframe
            )
            
            strategy_results['ema_only'] = {
                'signals': ema_results,
                'performance': performance_analyzer.analyze_performance(ema_results)
            }
            
            # Test MACD only
            self.logger.info(f"\n📊 Testing MACD strategy only...")
            config.SIMPLE_EMA_STRATEGY = False
            config.MACD_EMA_STRATEGY = True
            
            macd_results = detector.backtest_signals(
                epic_list=epic_list,
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe=timeframe
            )
            
            strategy_results['macd_only'] = {
                'signals': macd_results,
                'performance': performance_analyzer.analyze_performance(macd_results)
            }
            
            # Test Combined
            self.logger.info(f"\n🎯 Testing Combined strategy...")
            config.SIMPLE_EMA_STRATEGY = True
            config.MACD_EMA_STRATEGY = True
            config.COMBINED_STRATEGY_MODE = 'consensus'
            
            combined_results = detector.backtest_signals(
                epic_list=epic_list,
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe=timeframe
            )
            
            strategy_results['combined'] = {
                'signals': combined_results,
                'performance': performance_analyzer.analyze_performance(combined_results)
            }
            
            # Display comparison
            self._display_strategy_comparison(strategy_results)
            
            # Restore original settings
            config.SIMPLE_EMA_STRATEGY = original_ema
            config.MACD_EMA_STRATEGY = original_macd
            if original_combined:
                config.COMBINED_STRATEGY_MODE = original_combined
            elif hasattr(config, 'COMBINED_STRATEGY_MODE'):
                delattr(config, 'COMBINED_STRATEGY_MODE')
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Strategy comparison failed: {e}")
            return False
    
    def _analyze_bb_performance(self, results: List[Dict]):
        """Analyze Bollinger Band filter performance"""
        self.logger.info(f"\n📊 BOLLINGER BAND ANALYSIS:")
        
        bb_signals = [s for s in results if 'bb_position' in s]
        if not bb_signals:
            self.logger.warning("   No BB data available in signals")
            return
        
        # Analyze BB position distribution
        bb_positions = [s['bb_position'] for s in bb_signals if s.get('bb_position') is not None]
        
        if bb_positions:
            avg_bb_position = sum(bb_positions) / len(bb_positions)
            self.logger.info(f"   Average BB Position: {avg_bb_position:.2f}")
            
            # Count signals by BB zones
            lower_zone = len([p for p in bb_positions if p < 0.3])  # Lower 30%
            middle_zone = len([p for p in bb_positions if 0.3 <= p <= 0.7])  # Middle 40%
            upper_zone = len([p for p in bb_positions if p > 0.7])  # Upper 30%
            
            self.logger.info(f"   BB Zone Distribution:")
            self.logger.info(f"     Lower (0-30%): {lower_zone} signals")
            self.logger.info(f"     Middle (30-70%): {middle_zone} signals")
            self.logger.info(f"     Upper (70-100%): {upper_zone} signals")
    
    def _display_ema_comparison(self, comparison_results: Dict):
        """Display EMA configuration comparison results"""
        self.logger.info(f"\n📊 EMA CONFIGURATION COMPARISON:")
        self.logger.info("=" * 80)
        
        # Header
        header = f"{'Config':<12} {'EMAs':<12} {'Signals':<8} {'Avg Conf':<9} {'Win Rate':<9} {'Avg Profit':<11} {'Avg Loss':<9}"
        self.logger.info(header)
        self.logger.info("-" * 80)
        
        # Sort by signal count
        sorted_configs = sorted(
            comparison_results.items(), 
            key=lambda x: x[1]['performance'].get('total_signals', 0), 
            reverse=True
        )
        
        for config_name, data in sorted_configs:
            performance = data['performance']
            ema_settings = data['ema_settings']
            
            ema_str = f"{ema_settings['short']}/{ema_settings['long']}/{ema_settings['trend']}"
            
            row = (f"{config_name:<12} {ema_str:<12} "
                   f"{performance.get('total_signals', 0):<8} "
                   f"{performance.get('average_confidence', 0):<9.1%} "
                   f"{performance.get('win_rate', 0):<9.1%} "
                   f"{performance.get('average_profit_pips', 0):<11.1f} "
                   f"{performance.get('average_loss_pips', 0):<9.1f}")
            
            self.logger.info(row)
        
        self.logger.info("=" * 80)
        
        # Recommendations
        if sorted_configs:
            best_config = sorted_configs[0]
            self.logger.info(f"🏆 Best performing: {best_config[0]} ({best_config[1]['performance']['total_signals']} signals)")
    
    def _display_strategy_comparison(self, strategy_results: Dict):
        """Display strategy comparison results"""
        self.logger.info(f"\n🎯 STRATEGY PERFORMANCE COMPARISON:")
        self.logger.info("=" * 70)
        
        # Header
        header = f"{'Strategy':<15} {'Signals':<8} {'Avg Conf':<9} {'Win Rate':<9} {'Avg Profit':<11}"
        self.logger.info(header)
        self.logger.info("-" * 70)
        
        for strategy_name, data in strategy_results.items():
            performance = data['performance']
            
            row = (f"{strategy_name:<15} "
                   f"{performance.get('total_signals', 0):<8} "
                   f"{performance.get('average_confidence', 0):<9.1%} "
                   f"{performance.get('win_rate', 0):<9.1%} "
                   f"{performance.get('average_profit_pips', 0):<11.1f}")
            
            self.logger.info(row)
        
        self.logger.info("=" * 70)
        
        # Analysis
        total_signals = {name: data['performance'].get('total_signals', 0) for name, data in strategy_results.items()}
        best_strategy = max(total_signals.items(), key=lambda x: x[1])
        
        self.logger.info(f"🏆 Most active strategy: {best_strategy[0]} ({best_strategy[1]} signals)")
    
    def quick_backtest(self, epic: str, hours: int = 24) -> bool:
        """Quick backtest for recent signals"""
        self.logger.info(f"⚡ Quick backtest: {epic} last {hours} hours")
        
        try:
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Convert hours to days for backtest
            days = max(1, hours // 24)
            
            results = detector.backtest_signals(
                epic_list=[epic],
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe='5m'
            )
            
            if results:
                self.logger.info(f"✅ Found {len(results)} recent signals")
                
                # Quick summary
                for signal in results[-5:]:  # Show last 5
                    timestamp = signal.get('timestamp', 'Unknown')
                    signal_type = signal.get('signal_type', 'Unknown')
                    confidence = signal.get('confidence_score', 0)
                    strategy = signal.get('strategy', 'unknown')
                    
                    self.logger.info(f"   {timestamp}: {signal_type} ({strategy}, {confidence:.1%})")
            else:
                self.logger.info("❌ No recent signals found")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Quick backtest failed: {e}")
            return False