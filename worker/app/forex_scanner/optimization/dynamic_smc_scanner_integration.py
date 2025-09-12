#!/usr/bin/env python3
"""
Dynamic SMC Scanner Integration

Demonstrates the complete SMC optimization system in action:
- Creates SMC strategies with optimal parameters for each epic
- Shows optimization vs static configuration comparison
- Provides system status and readiness reporting

Usage:
    python dynamic_smc_scanner_integration.py
"""

import sys
import os
import logging
from typing import Dict, List
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from forex_scanner.core.database.database_manager import DatabaseManager

class DynamicSMCScanner:
    """
    Dynamic SMC scanner that automatically uses optimized parameters
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        self.smc_strategies_cache = {}  # Cache SMC strategies by epic
    
    def _get_smc_strategy_for_epic(self, epic: str):
        """Get SMC strategy instance for specific epic with optimal parameters"""
        
        if epic in self.smc_strategies_cache:
            return self.smc_strategies_cache[epic]
        
        try:
            # Try to import SMC strategy
            from forex_scanner.core.strategies.smc_strategy import SMCStrategy
            
            # Create strategy with epic-specific optimized parameters
            strategy = SMCStrategy(
                epic=epic,
                use_optimized_parameters=True,
                backtest_mode=False
            )
            
            # Cache the strategy
            self.smc_strategies_cache[epic] = strategy
            
            self.logger.info(f"✅ Created SMC strategy for {epic}")
            return strategy
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create SMC strategy for {epic}: {e}")
            
            # Fallback: Create basic strategy without optimization
            try:
                from forex_scanner.core.strategies.smc_strategy import SMCStrategy
                
                fallback_strategy = SMCStrategy(
                    smc_config_name='moderate',
                    use_optimized_parameters=False
                )
                
                self.smc_strategies_cache[epic] = fallback_strategy
                self.logger.warning(f"⚠️ Using fallback SMC strategy for {epic}")
                return fallback_strategy
                
            except Exception as fallback_error:
                self.logger.error(f"❌ Failed to create fallback SMC strategy: {fallback_error}")
                return None
    
    def scan_all_optimized_epics(self) -> Dict[str, any]:
        """Scan all epics with optimized SMC parameters"""
        
        try:
            from optimization.optimal_parameter_service import (
                get_all_optimized_smc_epics,
                get_smc_system_readiness,
                get_smc_optimal_parameters
            )
            
            # Get list of optimized epics
            optimized_epics = get_all_optimized_smc_epics()
            
            if not optimized_epics:
                self.logger.warning("⚠️ No SMC optimized epics found")
                return {
                    'total_epics': 0,
                    'strategies_created': 0,
                    'optimization_data': [],
                    'system_status': 'no_optimization_data'
                }
            
            self.logger.info(f"🎯 Found {len(optimized_epics)} SMC optimized epics")
            
            strategies_created = 0
            optimization_data = []
            
            # Create strategies for each optimized epic
            for epic in optimized_epics:
                try:
                    # Get optimal parameters
                    optimal_params = get_smc_optimal_parameters(epic)
                    
                    # Create strategy
                    strategy = self._get_smc_strategy_for_epic(epic)
                    
                    if strategy:
                        strategies_created += 1
                        
                        optimization_data.append({
                            'epic': epic,
                            'smc_config': optimal_params.smc_config,
                            'confidence_threshold': optimal_params.confidence_threshold,
                            'timeframe': optimal_params.timeframe,
                            'performance_score': optimal_params.performance_score,
                            'win_rate': optimal_params.win_rate,
                            'confluence_accuracy': optimal_params.confluence_accuracy,
                            'strategy_created': True,
                            'uses_optimization': strategy.smc_config.get('_optimized', False),
                            'stop_loss_pips': optimal_params.stop_loss_pips,
                            'take_profit_pips': optimal_params.take_profit_pips,
                            'last_optimized': optimal_params.last_optimized.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                        self.logger.info(f"   ✅ {epic}: {optimal_params.smc_config} config, "
                                       f"Score: {optimal_params.performance_score:.6f}, "
                                       f"Win Rate: {optimal_params.win_rate:.1f}%")
                    else:
                        optimization_data.append({
                            'epic': epic,
                            'strategy_created': False,
                            'error': 'Failed to create strategy'
                        })
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to process {epic}: {e}")
                    optimization_data.append({
                        'epic': epic,
                        'strategy_created': False,
                        'error': str(e)
                    })
            
            return {
                'total_epics': len(optimized_epics),
                'strategies_created': strategies_created,
                'optimization_data': optimization_data,
                'system_status': 'operational' if strategies_created > 0 else 'degraded'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to scan optimized epics: {e}")
            return {
                'total_epics': 0,
                'strategies_created': 0,
                'optimization_data': [],
                'system_status': 'error',
                'error': str(e)
            }
    
    def print_optimization_status(self):
        """Print comprehensive optimization status report"""
        
        try:
            from optimization.optimal_parameter_service import get_smc_system_readiness
            from configdata.strategies.config_smc_strategy import validate_smc_config
            import config
            
            print("\n" + "="*80)
            print("🧠 SMC OPTIMIZATION SYSTEM STATUS")
            print("="*80)
            
            # SMC Configuration validation
            config_validation = validate_smc_config()
            print(f"\n📋 SMC CONFIGURATION:")
            if config_validation.get('valid'):
                print(f"   ✅ Valid: {config_validation.get('config_count')} configurations available")
                print(f"   🔧 Active Config: {config_validation.get('active_config')}")
                print(f"   🎛️ Features Enabled: {config_validation.get('features_enabled')}")
            else:
                print(f"   ❌ Invalid: {config_validation.get('error')}")
            
            # System readiness
            readiness = get_smc_system_readiness()
            print(f"\n📊 SYSTEM READINESS:")
            print(f"   🎯 Configured Epics: {readiness.get('total_configured', 0)}")
            print(f"   ⚡ Optimized Epics: {readiness.get('total_optimized', 0)}")
            print(f"   📈 Coverage: {readiness.get('optimization_coverage', 0):.1f}%")
            print(f"   🚀 Production Ready: {'✅ Yes' if readiness.get('ready_for_production') else '❌ No'}")
            
            if readiness.get('missing_epics'):
                print(f"   ⚠️ Missing Optimization: {len(readiness['missing_epics'])} epics")
                print(f"      {', '.join(readiness['missing_epics'][:3])}{'...' if len(readiness['missing_epics']) > 3 else ''}")
            
            # Scan optimization data
            scan_results = self.scan_all_optimized_epics()
            print(f"\n🔍 SCANNING RESULTS:")
            print(f"   📊 Total Epics Scanned: {scan_results.get('total_epics', 0)}")
            print(f"   ✅ Strategies Created: {scan_results.get('strategies_created', 0)}")
            print(f"   🎯 System Status: {scan_results.get('system_status', 'unknown').upper()}")
            
            # Show top performers
            optimization_data = scan_results.get('optimization_data', [])
            successful_optimizations = [data for data in optimization_data 
                                       if data.get('strategy_created') and 'performance_score' in data]
            
            if successful_optimizations:
                print(f"\n🌟 TOP PERFORMERS:")
                # Sort by performance score
                top_performers = sorted(successful_optimizations, 
                                      key=lambda x: x.get('performance_score', 0), 
                                      reverse=True)[:5]
                
                for i, performer in enumerate(top_performers, 1):
                    print(f"   {i}. {performer['epic']}: {performer['smc_config']} | "
                          f"Score: {performer['performance_score']:.6f} | "
                          f"Win: {performer['win_rate']:.1f}% | "
                          f"Confluence: {performer['confluence_accuracy']:.1f}% | "
                          f"SL/TP: {performer['stop_loss_pips']:.0f}/{performer['take_profit_pips']:.0f}")
            
            # Next steps
            print(f"\n🚀 NEXT STEPS:")
            if readiness.get('ready_for_production'):
                print(f"   ✅ System is ready for production trading")
                print(f"   📈 Run live scanner with optimized parameters")
                print(f"   📊 Monitor performance and re-optimize periodically")
            else:
                missing_count = len(readiness.get('missing_epics', []))
                print(f"   🔧 Optimize {missing_count} remaining epics:")
                print(f"      python optimize_smc_parameters.py --all-epics --smart-presets --days 30")
                print(f"   📊 Re-run system status after optimization")
            
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"❌ Failed to print optimization status: {e}")
    
    def compare_optimization_vs_static(self, epic: str = None):
        """Compare optimized vs static SMC parameters for an epic"""
        
        try:
            from optimization.optimal_parameter_service import (
                get_smc_optimal_parameters,
                is_epic_smc_optimized
            )
            from configdata.strategies.config_smc_strategy import SMC_STRATEGY_CONFIG, ACTIVE_SMC_CONFIG
            
            test_epic = epic or 'CS.D.EURUSD.MINI.IP'
            
            print(f"\n🔍 OPTIMIZATION vs STATIC COMPARISON: {test_epic}")
            print("="*80)
            
            # Get static configuration
            static_config = SMC_STRATEGY_CONFIG.get(ACTIVE_SMC_CONFIG, {})
            
            print(f"📋 STATIC CONFIGURATION ({ACTIVE_SMC_CONFIG}):")
            print(f"   Swing Length: {static_config.get('swing_length', 'N/A')}")
            print(f"   Structure Confirmation: {static_config.get('structure_confirmation', 'N/A')}")
            print(f"   Order Block Length: {static_config.get('order_block_length', 'N/A')}")
            print(f"   FVG Min Size: {static_config.get('fvg_min_size', 'N/A')}")
            print(f"   Confluence Required: {static_config.get('confluence_required', 'N/A')}")
            print(f"   Min Risk:Reward: {static_config.get('min_risk_reward', 'N/A')}")
            print(f"   Performance Score: N/A (static config)")
            print(f"   Win Rate: N/A (static config)")
            
            # Get optimized parameters
            if is_epic_smc_optimized(test_epic):
                print(f"\n⚡ OPTIMIZED CONFIGURATION:")
                optimal_params = get_smc_optimal_parameters(test_epic)
                
                print(f"   SMC Config: {optimal_params.smc_config}")
                print(f"   Swing Length: {optimal_params.swing_length}")
                print(f"   Structure Confirmation: {optimal_params.structure_confirmation}")
                print(f"   Order Block Length: {optimal_params.order_block_length}")
                print(f"   FVG Min Size: {optimal_params.fvg_min_size}")
                print(f"   Confluence Required: {optimal_params.confluence_required}")
                print(f"   Min Risk:Reward: {optimal_params.min_risk_reward}")
                print(f"   Performance Score: {optimal_params.performance_score:.6f}")
                print(f"   Win Rate: {optimal_params.win_rate:.1f}%")
                print(f"   Confluence Accuracy: {optimal_params.confluence_accuracy:.1f}%")
                print(f"   Last Optimized: {optimal_params.last_optimized.strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"\n💡 OPTIMIZATION BENEFITS:")
                print(f"   🎯 Data-driven parameter selection")
                print(f"   📊 Proven performance metrics (Score: {optimal_params.performance_score:.6f})")
                print(f"   📈 Win rate: {optimal_params.win_rate:.1f}%")
                print(f"   🧠 Epic-specific calibration")
                print(f"   ⚡ Automatic parameter updates from backtesting")
                
            else:
                print(f"\n⚠️ NO OPTIMIZATION DATA:")
                print(f"   Epic {test_epic} has not been optimized yet")
                print(f"   Currently using static '{ACTIVE_SMC_CONFIG}' configuration")
                print(f"   Run optimization: python optimize_smc_parameters.py --epic {test_epic} --smart-presets --days 30")
            
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"❌ Failed to compare configurations: {e}")


def main():
    """Main demonstration function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dynamic_smc_scanner.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create dynamic scanner
        scanner = DynamicSMCScanner()
        
        # Print optimization status
        scanner.print_optimization_status()
        
        # Show comparison between optimization vs static
        scanner.compare_optimization_vs_static('CS.D.EURUSD.MINI.IP')
        
        # Demonstrate scanning with optimized parameters
        print("\n🔍 SCANNING WITH OPTIMIZED PARAMETERS:")
        print("="*80)
        
        scan_results = scanner.scan_all_optimized_epics()
        
        if scan_results.get('total_epics', 0) > 0:
            print(f"✅ Successfully scanned {scan_results['total_epics']} epics")
            print(f"⚡ Created {scan_results['strategies_created']} optimized strategies")
            
            # Show details for first few epics
            optimization_data = scan_results.get('optimization_data', [])[:3]
            for data in optimization_data:
                if data.get('strategy_created'):
                    print(f"   📊 {data['epic']}: {data['smc_config']} config, "
                          f"Score: {data.get('performance_score', 0):.6f}, "
                          f"Win Rate: {data.get('win_rate', 0):.1f}%")
        else:
            print("⚠️ No optimized epics found - run SMC parameter optimization first")
            print("   Command: python optimize_smc_parameters.py --all-epics --smart-presets --days 30")
        
        print("="*80 + "\n")
        
        logger.info("✅ Dynamic SMC scanner demonstration completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)