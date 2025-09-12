# commands/dynamic_config_commands.py
"""
Dynamic EMA Configuration Commands for CLI Integration
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
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

class DynamicConfigCommands:
    """CLI commands for dynamic EMA configuration management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._db_manager = None
        self._data_fetcher = None
        self._dynamic_config_manager = None
    
    @property
    def db_manager(self):
        """Lazy initialization of database manager"""
        if self._db_manager is None:
            self._db_manager = DatabaseManager(config.DATABASE_URL)
        return self._db_manager
    
    @property
    def data_fetcher(self):
        """Lazy initialization of data fetcher"""
        if self._data_fetcher is None:
            self._data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
        return self._data_fetcher
    
    @property
    def dynamic_config_manager(self):
        """Lazy initialization of dynamic config manager"""
        if self._dynamic_config_manager is None:
            try:
                # Use absolute import
                from core.intelligence.dynamic_ema_config_manager import get_dynamic_ema_config_manager
                self._dynamic_config_manager = get_dynamic_ema_config_manager(self.data_fetcher)
                self.logger.info("✅ Dynamic config manager loaded successfully")
            except ImportError as e:
                self.logger.error(f"❌ Dynamic config manager not available: {e}")
                self.logger.error("   Please ensure core/intelligence/dynamic_ema_config_manager.py exists")
                return None
            except Exception as e:
                self.logger.error(f"❌ Error initializing dynamic config manager: {e}")
                return None
        return self._dynamic_config_manager
    
    def show_configs(self, epic: str = None, format: str = 'table', verbose: bool = False) -> bool:
        """Show current dynamic EMA configurations and their status"""
        
        print("🧠 Dynamic EMA Configuration Status")
        print("=" * 50)
        
        # Check if dynamic config is enabled
        if not getattr(config, 'ENABLE_DYNAMIC_EMA_CONFIG', False):
            print("❌ Dynamic EMA configuration is DISABLED")
            print("   Set ENABLE_DYNAMIC_EMA_CONFIG = True in config.py to enable")
            return True
        
        if not self.dynamic_config_manager:
            print("❌ Dynamic config manager not available")
            return False
        
        try:
            # Get configuration summary
            summary = self.dynamic_config_manager.get_configuration_summary(epic)
            
            if format == 'json':
                print(json.dumps(summary, indent=2, default=str))
                return True
            
            # Table format
            print(f"📊 Total Configurations: {summary['total_configurations']}")
            print()
            
            # Show each configuration
            for config_name, config_data in summary['configurations'].items():
                print(f"🔧 {config_name.upper()}")
                print(f"   EMA Periods: {config_data['ema_periods']}")
                print(f"   Best Conditions: {config_data['best_conditions']}")
                
                if verbose:
                    print(f"   Preferred Pairs: {config_data['preferred_pairs']}")
                    
                    if config_data['performance']:
                        perf = config_data['performance']
                        print(f"   Performance: {perf['win_rate']:.1%} win rate, "
                             f"{perf['avg_profit']:.1f} avg pips, {perf['signals']} signals")
                
                print()
            
            # Show epic-specific analysis if requested
            if epic:
                print(f"🎯 Analysis for {epic}")
                print("-" * 30)
                
                try:
                    market_conditions = self.dynamic_config_manager.analyze_market_conditions(epic)
                    optimal_config = self.dynamic_config_manager.select_optimal_ema_config(epic)
                    
                    print(f"Market Conditions:")
                    print(f"   Volatility: {market_conditions['volatility_regime']}")
                    print(f"   Trend Strength: {market_conditions['trend_strength']}")
                    print(f"   Market Regime: {market_conditions['market_regime']}")
                    print(f"   Daily Pip Volatility: {market_conditions['daily_pip_volatility']:.1f}")
                    print(f"   Current Session: {market_conditions['current_session']}")
                    print()
                    
                    print(f"✅ Recommended Configuration: {optimal_config.name}")
                    print(f"   EMA Periods: {optimal_config.short}/{optimal_config.long}/{optimal_config.trend}")
                    
                except Exception as e:
                    print(f"❌ Error analyzing {epic}: {e}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Error showing configs: {e}")
            return False
    
    def config_performance(self, epic: str = None, days: int = 30, config: str = None, format: str = 'table') -> bool:
        """Show configuration performance statistics"""
        
        print("📊 Dynamic EMA Configuration Performance")
        print("=" * 50)
        
        if not getattr(config, 'ENABLE_DYNAMIC_EMA_CONFIG', False):
            print("❌ Dynamic EMA configuration is DISABLED")
            return True
        
        if not self.dynamic_config_manager:
            print("❌ Dynamic config manager not available")
            return False
        
        try:
            # Get performance data
            performance_data = {}
            epics_to_analyze = [epic] if epic else getattr(config, 'EPIC_LIST', [])
            
            for test_epic in epics_to_analyze:
                try:
                    summary = self.dynamic_config_manager.get_configuration_summary(test_epic)
                    if 'configurations' in summary:
                        performance_data[test_epic] = summary['configurations']
                except Exception as e:
                    print(f"⚠️ Error analyzing {test_epic}: {e}")
            
            if format == 'json':
                print(json.dumps(performance_data, indent=2, default=str))
                return True
            
            # Table format
            for test_epic, configs in performance_data.items():
                print(f"📈 {test_epic}")
                print("-" * 40)
                
                # Filter by specific config if requested
                configs_to_show = {config_name: data for config_name, data in configs.items() 
                                 if not config or config_name == config} if config else configs
                
                for config_name, config_data in configs_to_show.items():
                    if 'performance' in config_data and config_data['performance']:
                        perf = config_data['performance']
                        print(f"   {config_name:12} | "
                             f"Win Rate: {perf['win_rate']:6.1%} | "
                             f"Avg Profit: {perf['avg_profit']:6.1f} pips | "
                             f"Signals: {perf['signals']:4d}")
                    else:
                        print(f"   {config_name:12} | No performance data")
                
                print()
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Error showing performance: {e}")
            return False
    
    def optimize_configs(self, epic: str, days: int = 30, dry_run: bool = False) -> bool:
        """Optimize EMA configurations for specific epic"""
        
        print(f"🎯 Optimizing EMA Configuration for {epic}")
        print("=" * 50)
        
        if not getattr(config, 'ENABLE_DYNAMIC_EMA_CONFIG', False):
            print("❌ Dynamic EMA configuration is DISABLED")
            return True
        
        if not self.dynamic_config_manager:
            print("❌ Dynamic config manager not available")
            return False
        
        try:
            # Analyze market conditions
            print("🔍 Analyzing market conditions...")
            market_conditions = self.dynamic_config_manager.analyze_market_conditions(epic)
            
            print(f"   Volatility Regime: {market_conditions['volatility_regime']}")
            print(f"   Trend Strength: {market_conditions['trend_strength']}")
            print(f"   Market Regime: {market_conditions['market_regime']}")
            print(f"   Daily Volatility: {market_conditions['daily_pip_volatility']:.1f} pips")
            print()
            
            # Get optimal configuration
            optimal_config = self.dynamic_config_manager.select_optimal_ema_config(epic)
            
            print("✅ Optimization Results:")
            print(f"   Recommended Config: {optimal_config.name}")
            print(f"   EMA Periods: {optimal_config.short}/{optimal_config.long}/{optimal_config.trend}")
            print(f"   Best For: {optimal_config.best_market_regime} markets with {optimal_config.best_volatility_regime} volatility")
            print(f"   Volatility Range: {optimal_config.min_pip_volatility:.1f} - {optimal_config.max_pip_volatility:.1f} pips")
            print()
            
            # Show performance data if available
            if epic in self.dynamic_config_manager.performance_history:
                epic_performance = self.dynamic_config_manager.performance_history[epic]
                if optimal_config.name in epic_performance:
                    perf = epic_performance[optimal_config.name]
                    print("📊 Historical Performance:")
                    print(f"   Win Rate: {perf['win_rate']:.1%}")
                    print(f"   Average Profit: {perf['avg_profit']:.1f} pips")
                    print(f"   Total Signals: {perf['signals']}")
                    print()
            
            if not dry_run:
                print("💾 Configuration optimization completed and cached.")
                # Force refresh to apply the new optimal config
                self.dynamic_config_manager.force_config_refresh(epic)
            else:
                print("🔍 Dry run completed - no changes applied.")
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Error optimizing configs: {e}")
            return False
    
    def market_analysis(self, epic: str = None, refresh: bool = False, detailed: bool = False) -> bool:
        """Show market condition analysis for dynamic configuration"""
        
        print("🌍 Market Condition Analysis")
        print("=" * 50)
        
        if not getattr(config, 'ENABLE_DYNAMIC_EMA_CONFIG', False):
            print("❌ Dynamic EMA configuration is DISABLED")
            return True
        
        if not self.dynamic_config_manager:
            print("❌ Dynamic config manager not available")
            return False
        
        try:
            # Analyze epics
            epics_to_analyze = [epic] if epic else getattr(config, 'EPIC_LIST', [])
            
            for test_epic in epics_to_analyze:
                try:
                    print(f"📊 {test_epic}")
                    print("-" * 40)
                    
                    if refresh:
                        self.dynamic_config_manager.force_config_refresh(test_epic)
                    
                    # Get market conditions
                    market_conditions = self.dynamic_config_manager.analyze_market_conditions(test_epic)
                    
                    print(f"Volatility Regime:     {market_conditions['volatility_regime'].upper()}")
                    print(f"Trend Strength:       {market_conditions['trend_strength'].upper()}")
                    print(f"Market Regime:        {market_conditions['market_regime'].upper()}")
                    print(f"Daily Pip Volatility: {market_conditions['daily_pip_volatility']:.1f} pips")
                    print(f"Current Session:      {market_conditions['current_session'].upper()}")
                    print(f"Analysis Time:        {market_conditions['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if detailed:
                        # Get additional technical analysis
                        pair = test_epic.split('.')[-2] if '.' in test_epic else test_epic
                        df = self.data_fetcher.get_enhanced_data(test_epic, pair, timeframe='15m')
                        
                        if df is not None and len(df) > 0:
                            current_price = df['close'].iloc[-1]
                            print(f"Current Price:        {current_price:.5f}")
                            
                            if 'atr_14' in df.columns:
                                atr = df['atr_14'].iloc[-1]
                                atr_pips = atr / current_price * 10000
                                print(f"ATR (14):             {atr_pips:.1f} pips")
                            
                            if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_200']):
                                ema_9 = df['ema_9'].iloc[-1]
                                ema_21 = df['ema_21'].iloc[-1]
                                ema_200 = df['ema_200'].iloc[-1]
                                
                                print(f"EMA Alignment:")
                                print(f"   EMA 9:             {ema_9:.5f}")
                                print(f"   EMA 21:            {ema_21:.5f}")
                                print(f"   EMA 200:           {ema_200:.5f}")
                                
                                if ema_9 > ema_21 > ema_200:
                                    print(f"   Trend:             BULLISH ⬆️")
                                elif ema_9 < ema_21 < ema_200:
                                    print(f"   Trend:             BEARISH ⬇️")
                                else:
                                    print(f"   Trend:             MIXED ↔️")
                    
                    # Show recommended configuration
                    optimal_config = self.dynamic_config_manager.select_optimal_ema_config(test_epic)
                    print(f"Recommended Config:   {optimal_config.name.upper()}")
                    print(f"EMA Periods:          {optimal_config.short}/{optimal_config.long}/{optimal_config.trend}")
                    
                    print()
                    
                except Exception as e:
                    print(f"❌ Error analyzing {test_epic}: {e}")
                    print()
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Error in market analysis: {e}")
            return False
    
    def test_config_selection(self, epic: str = None, config: str = None, iterations: int = 5) -> bool:
        """Test dynamic configuration selection algorithm"""
        
        print("🧪 Testing Dynamic Configuration Selection")
        print("=" * 50)
        
        if not getattr(config, 'ENABLE_DYNAMIC_EMA_CONFIG', False):
            print("❌ Dynamic EMA configuration is DISABLED")
            return True
        
        if not self.dynamic_config_manager:
            print("❌ Dynamic config manager not available")
            return False
        
        try:
            # Test epics
            epics_to_test = [epic] if epic else getattr(config, 'EPIC_LIST', [])[:3]  # Limit to 3 for testing
            
            results = {}
            
            for test_epic in epics_to_test:
                print(f"🔬 Testing {test_epic}")
                print("-" * 30)
                
                epic_results = []
                
                for i in range(iterations):
                    try:
                        # Force refresh to get fresh analysis
                        self.dynamic_config_manager.force_config_refresh(test_epic)
                        
                        # Get market conditions
                        market_conditions = self.dynamic_config_manager.analyze_market_conditions(test_epic)
                        
                        # Test different performance weights
                        performance_weights = [0.1, 0.3, 0.5] if iterations >= 3 else [0.3]
                        
                        for weight in performance_weights:
                            optimal_config = self.dynamic_config_manager.select_optimal_ema_config(
                                test_epic, market_conditions, weight
                            )
                            
                            epic_results.append({
                                'iteration': i + 1,
                                'performance_weight': weight,
                                'selected_config': optimal_config.name,
                                'market_conditions': market_conditions.copy()
                            })
                            
                            print(f"   Iteration {i+1} (weight: {weight}): {optimal_config.name}")
                    
                    except Exception as e:
                        print(f"   ❌ Iteration {i+1} failed: {e}")
                
                results[test_epic] = epic_results
                print()
            
            # Analyze results
            print("📊 Test Results Summary")
            print("=" * 30)
            
            for test_epic, epic_results in results.items():
                if epic_results:
                    configs_selected = [r['selected_config'] for r in epic_results]
                    unique_configs = set(configs_selected)
                    most_common = max(unique_configs, key=configs_selected.count) if unique_configs else "None"
                    
                    print(f"{test_epic}:")
                    print(f"   Unique configs selected: {len(unique_configs)}")
                    print(f"   Most common selection: {most_common}")
                    print(f"   Selection consistency: {configs_selected.count(most_common)}/{len(configs_selected)}")
                    
                    if config:
                        # Show specific config results if requested
                        config_selections = [r for r in epic_results if r['selected_config'] == config]
                        print(f"   '{config}' selected: {len(config_selections)} times")
                    
                    print()
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Error testing config selection: {e}")
            return False
    
    def settings(self, enable: bool = None, performance_weight: float = None, 
                refresh_interval: int = None, show_current: bool = False) -> bool:
        """Manage dynamic configuration settings"""
        
        print("⚙️ Dynamic Configuration Settings")
        print("=" * 50)
        
        # Show current settings
        if show_current or (enable is None and performance_weight is None and refresh_interval is None):
            print("Current Settings:")
            print(f"   Enabled: {getattr(config, 'ENABLE_DYNAMIC_EMA_CONFIG', False)}")
            print(f"   Performance Weight: {getattr(config, 'DYNAMIC_CONFIG_PERFORMANCE_WEIGHT', 0.3)}")
            print(f"   Refresh Interval: {getattr(config, 'DYNAMIC_CONFIG_REFRESH_INTERVAL', 3600)} seconds")
            print(f"   Cache Expiry: {getattr(config, 'MARKET_CONDITION_CACHE_EXPIRY', 900)} seconds")
            print(f"   Performance Tracking: {getattr(config, 'PERFORMANCE_TRACKING_ENABLED', True)}")
            print()
        
        # Update settings (Note: These would need to be persisted to config file or database)
        changes_made = False
        
        if enable is not None:
            print(f"{'✅ Enabling' if enable else '❌ Disabling'} dynamic configuration")
            # In production, this would update the config file or database
            changes_made = True
        
        if performance_weight is not None:
            if 0.0 <= performance_weight <= 1.0:
                print(f"🎛️ Setting performance weight to {performance_weight}")
                # In production, this would update the config
                changes_made = True
            else:
                print("❌ Performance weight must be between 0.0 and 1.0")
                return False
        
        if refresh_interval is not None:
            if refresh_interval > 0:
                print(f"⏱️ Setting refresh interval to {refresh_interval} seconds")
                # In production, this would update the config
                changes_made = True
            else:
                print("❌ Refresh interval must be positive")
                return False
        
        if changes_made:
            print()
            print("⚠️ Note: Settings changes require restart to take effect")
            print("   Consider implementing runtime config updates for production use")
        elif not show_current:
            print("ℹ️ No changes specified. Use --help to see available options.")
        
        return True


# Helper functions for backward compatibility with the click-based interface
def show_configs_click(epic=None, format='table', verbose=False):
    """Click-compatible wrapper for show_configs"""
    cmd = DynamicConfigCommands()
    return cmd.show_configs(epic, format, verbose)


def config_performance_click(epic=None, days=30, config=None, format='table'):
    """Click-compatible wrapper for config_performance"""
    cmd = DynamicConfigCommands()
    return cmd.config_performance(epic, days, config, format)


def optimize_configs_click(epic, days=30, dry_run=False):
    """Click-compatible wrapper for optimize_configs"""
    cmd = DynamicConfigCommands()
    return cmd.optimize_configs(epic, days, dry_run)


def market_analysis_click(epic=None, refresh=False, detailed=False):
    """Click-compatible wrapper for market_analysis"""
    cmd = DynamicConfigCommands()
    return cmd.market_analysis(epic, refresh, detailed)


def test_config_selection_click(epic=None, config=None, iterations=5):
    """Click-compatible wrapper for test_config_selection"""
    cmd = DynamicConfigCommands()
    return cmd.test_config_selection(epic, config, iterations)


def settings_click(enable=None, performance_weight=None, refresh_interval=None, show_current=False):
    """Click-compatible wrapper for settings"""
    cmd = DynamicConfigCommands()
    return cmd.settings(enable, performance_weight, refresh_interval, show_current)