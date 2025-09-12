#!/usr/bin/env python3
"""
Dynamic Zero-Lag Scanner Integration
Intelligent scanning system that uses optimized parameters per epic from database
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.database import DatabaseManager
from optimization.optimal_parameter_service import get_zerolag_optimal_parameters, get_all_optimized_zerolag_epics
from core.strategies.zero_lag_strategy import create_optimized_zero_lag_strategy

try:
    import config
except ImportError:
    from forex_scanner import config


@dataclass
class ZeroLagScannerConfig:
    """Configuration for dynamic zero-lag scanner"""
    use_optimization_data: bool = True
    fallback_to_config: bool = True
    min_optimization_age_days: int = 90
    min_performance_score: float = 0.1
    preferred_timeframe: str = '15m'
    max_epics_per_scan: int = 20


class DynamicZeroLagScanner:
    """
    Intelligent zero-lag scanner that automatically uses optimal parameters per epic
    """
    
    def __init__(self, config: ZeroLagScannerConfig = None):
        self.logger = logging.getLogger('dynamic_zerolag_scanner')
        self.setup_logging()
        
        self.config = config or ZeroLagScannerConfig()
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        
        # Track scanner state
        self.optimized_strategies = {}  # epic -> strategy instance
        self.fallback_strategies = {}   # epic -> strategy instance
        self.scanner_stats = {
            'optimized_epics': 0,
            'fallback_epics': 0,
            'total_scanned': 0,
            'optimization_failures': 0
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def initialize_optimized_strategies(self) -> Dict[str, bool]:
        """Initialize strategy instances with optimal parameters for each epic"""
        self.logger.info("🚀 Initializing Dynamic Zero-Lag Scanner...")
        
        initialization_results = {}
        
        try:
            # Get all epics with optimization data
            optimized_epics = get_all_optimized_zerolag_epics()
            
            # Get fallback epic list if needed
            fallback_epics = getattr(config, 'TRADEABLE_EPICS', None) or getattr(config, 'EPIC_LIST', [])
            
            all_epics = list(set(optimized_epics + fallback_epics))[:self.config.max_epics_per_scan]
            
            self.logger.info(f"📊 Processing {len(all_epics)} epics:")
            self.logger.info(f"   ✅ Optimized: {len(optimized_epics)}")
            self.logger.info(f"   📋 Fallback: {len(set(fallback_epics) - set(optimized_epics))}")
            
            for epic in all_epics:
                try:
                    if epic in optimized_epics:
                        # Create optimized strategy
                        strategy = create_optimized_zero_lag_strategy(epic)
                        self.optimized_strategies[epic] = strategy
                        initialization_results[epic] = True
                        self.scanner_stats['optimized_epics'] += 1
                        
                        # Get metadata for logging
                        metadata = strategy.get_strategy_metadata()
                        opt_data = metadata['optimization_data']
                        self.logger.info(f"   ✅ {epic}: Optimized (Score: {opt_data['performance_score']:.3f})")
                        
                    else:
                        # Create fallback strategy
                        from core.strategies.zero_lag_strategy import create_zero_lag_strategy
                        strategy = create_zero_lag_strategy(epic=epic)
                        self.fallback_strategies[epic] = strategy
                        initialization_results[epic] = False
                        self.scanner_stats['fallback_epics'] += 1
                        
                        self.logger.info(f"   📋 {epic}: Fallback config")
                        
                except Exception as e:
                    self.logger.error(f"   ❌ {epic}: Failed to initialize - {e}")
                    initialization_results[epic] = False
                    self.scanner_stats['optimization_failures'] += 1
            
            self.scanner_stats['total_scanned'] = len(all_epics)
            
            # Summary
            self.logger.info(f"\n📈 Dynamic Scanner Initialization Complete:")
            self.logger.info(f"   🎯 Optimized Strategies: {self.scanner_stats['optimized_epics']}")
            self.logger.info(f"   📋 Fallback Strategies: {self.scanner_stats['fallback_epics']}")
            self.logger.info(f"   ❌ Failures: {self.scanner_stats['optimization_failures']}")
            self.logger.info(f"   📊 Success Rate: {((len(all_epics) - self.scanner_stats['optimization_failures']) / len(all_epics) * 100):.1f}%")
            
            return initialization_results
            
        except Exception as e:
            self.logger.error(f"❌ Scanner initialization failed: {e}")
            return {}
    
    def scan_all_optimized_epics(self) -> List[Dict]:
        """Scan all epics using their optimal parameters"""
        if not self.optimized_strategies and not self.fallback_strategies:
            self.initialize_optimized_strategies()
        
        all_signals = []
        
        self.logger.info(f"🔍 Scanning {len(self.optimized_strategies) + len(self.fallback_strategies)} epics...")
        
        # Scan optimized strategies
        for epic, strategy in self.optimized_strategies.items():
            try:
                signals = self._scan_epic_with_strategy(epic, strategy, is_optimized=True)
                all_signals.extend(signals)
            except Exception as e:
                self.logger.error(f"❌ Scan failed for optimized {epic}: {e}")
        
        # Scan fallback strategies
        for epic, strategy in self.fallback_strategies.items():
            try:
                signals = self._scan_epic_with_strategy(epic, strategy, is_optimized=False)
                all_signals.extend(signals)
            except Exception as e:
                self.logger.error(f"❌ Scan failed for fallback {epic}: {e}")
        
        self.logger.info(f"✅ Scan complete: {len(all_signals)} signals detected")
        
        return all_signals
    
    def _scan_epic_with_strategy(self, epic: str, strategy, is_optimized: bool) -> List[Dict]:
        """Scan a specific epic with its strategy"""
        # This is a placeholder for actual scanning logic
        # In practice, this would integrate with the data fetcher and signal detection system
        
        signals = []
        
        try:
            # Get strategy metadata for context
            metadata = strategy.get_strategy_metadata()
            
            # Simulate signal detection (in practice, this would call the actual strategy)
            # signal = strategy.detect_signal(df, epic, spread_pips, timeframe)
            # if signal:
            #     signals.append(signal)
            
            # For now, log the strategy status
            config_type = "optimized" if is_optimized else "fallback"
            config_data = metadata['configuration']
            
            self.logger.debug(f"🔍 Scanning {epic} ({config_type}): "
                            f"ZL={config_data['zl_length']}, "
                            f"Band={config_data['band_multiplier']:.2f}, "
                            f"Conf={config_data['min_confidence']:.1%}")
            
        except Exception as e:
            self.logger.debug(f"❌ Error scanning {epic}: {e}")
        
        return signals
    
    def get_epic_strategy_info(self, epic: str) -> Dict:
        """Get detailed information about strategy for specific epic"""
        try:
            if epic in self.optimized_strategies:
                strategy = self.optimized_strategies[epic]
                metadata = strategy.get_strategy_metadata()
                
                return {
                    'epic': epic,
                    'strategy_type': 'optimized',
                    'metadata': metadata,
                    'optimization_status': 'active'
                }
                
            elif epic in self.fallback_strategies:
                strategy = self.fallback_strategies[epic]
                metadata = strategy.get_strategy_metadata()
                
                return {
                    'epic': epic,
                    'strategy_type': 'fallback',
                    'metadata': metadata,
                    'optimization_status': 'not_optimized'
                }
                
            else:
                return {
                    'epic': epic,
                    'strategy_type': 'not_loaded',
                    'optimization_status': 'unknown'
                }
                
        except Exception as e:
            return {
                'epic': epic,
                'error': str(e),
                'optimization_status': 'error'
            }
    
    def print_optimization_status(self):
        """Print detailed optimization status report"""
        print("\n" + "=" * 80)
        print("🔍 DYNAMIC ZERO-LAG SCANNER STATUS REPORT")
        print("=" * 80)
        print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # Overall statistics
        total_strategies = len(self.optimized_strategies) + len(self.fallback_strategies)
        if total_strategies > 0:
            optimization_rate = (len(self.optimized_strategies) / total_strategies) * 100
            print(f"📊 SCANNER STATISTICS:")
            print(f"   🎯 Total Epics: {total_strategies}")
            print(f"   ✅ Optimized: {len(self.optimized_strategies)} ({optimization_rate:.1f}%)")
            print(f"   📋 Fallback: {len(self.fallback_strategies)} ({100-optimization_rate:.1f}%)")
            print(f"   ❌ Failures: {self.scanner_stats['optimization_failures']}")
            print("")
        
        # Optimized strategies details
        if self.optimized_strategies:
            print("🎯 OPTIMIZED STRATEGIES:")
            print("-" * 60)
            print(f"{'Epic':25} | {'Score':8} | {'ZL':3} | {'Band':5} | {'Conf':5} | {'SL/TP':6}")
            print("-" * 60)
            
            for epic, strategy in self.optimized_strategies.items():
                try:
                    metadata = strategy.get_strategy_metadata()
                    config_data = metadata['configuration']
                    opt_data = metadata['optimization_data']
                    
                    score = opt_data.get('performance_score', 0.0)
                    zl_length = config_data.get('zl_length', 0)
                    band_mult = config_data.get('band_multiplier', 0.0)
                    confidence = config_data.get('min_confidence', 0.0)
                    sl_pips = opt_data.get('optimal_stop_loss_pips', 0)
                    tp_pips = opt_data.get('optimal_take_profit_pips', 0)
                    
                    sl_tp_str = f"{sl_pips:.0f}/{tp_pips:.0f}" if sl_pips and tp_pips else "N/A"
                    
                    print(f"{epic:25} | {score:8.3f} | {zl_length:3} | {band_mult:5.2f} | {confidence:4.1%} | {sl_tp_str:>6}")
                    
                except Exception as e:
                    print(f"{epic:25} | ERROR: {str(e)[:30]}")
            print("")
        
        # Fallback strategies
        if self.fallback_strategies:
            print("📋 FALLBACK STRATEGIES:")
            print("-" * 40)
            for epic in sorted(self.fallback_strategies.keys()):
                print(f"   📋 {epic}")
            print("")
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        if len(self.optimized_strategies) == 0:
            print("   ⚠️  No optimized strategies found - consider running optimization")
        elif len(self.optimized_strategies) < 5:
            print("   📈 Limited optimization data - consider optimizing more epics")
        else:
            print("   ✅ Good optimization coverage")
        
        if self.scanner_stats['optimization_failures'] > 0:
            print(f"   🔧 {self.scanner_stats['optimization_failures']} initialization failures - check logs")
        
        print("")
        print("📋 To optimize more epics, run:")
        print("   docker exec task-worker python forex_scanner/optimization/optimize_zerolag_parameters.py --all-epics")
        print("")
        print("=" * 80)
    
    def get_scanner_statistics(self) -> Dict:
        """Get scanner statistics as dictionary"""
        total_strategies = len(self.optimized_strategies) + len(self.fallback_strategies)
        
        return {
            'total_epics': total_strategies,
            'optimized_count': len(self.optimized_strategies),
            'fallback_count': len(self.fallback_strategies),
            'optimization_rate': (len(self.optimized_strategies) / total_strategies * 100) if total_strategies > 0 else 0,
            'initialization_failures': self.scanner_stats['optimization_failures'],
            'optimized_epics': list(self.optimized_strategies.keys()),
            'fallback_epics': list(self.fallback_strategies.keys()),
            'last_updated': datetime.now()
        }
    
    def refresh_optimization_data(self):
        """Refresh optimization data and reinitialize strategies"""
        self.logger.info("🔄 Refreshing optimization data...")
        
        # Clear existing strategies
        self.optimized_strategies.clear()
        self.fallback_strategies.clear()
        
        # Reset statistics
        self.scanner_stats = {
            'optimized_epics': 0,
            'fallback_epics': 0,
            'total_scanned': 0,
            'optimization_failures': 0
        }
        
        # Reinitialize
        results = self.initialize_optimized_strategies()
        
        self.logger.info(f"✅ Refresh complete - {len([r for r in results.values() if r])} strategies reinitialized")
        
        return results
    
    def suggest_optimization_candidates(self) -> List[str]:
        """Suggest epics that would benefit from optimization"""
        candidates = []
        
        # Epics using fallback configuration
        candidates.extend(list(self.fallback_strategies.keys()))
        
        # Check for old optimization data
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.min_optimization_age_days)
            
            for epic, strategy in self.optimized_strategies.items():
                metadata = strategy.get_strategy_metadata()
                opt_data = metadata['optimization_data']
                
                last_opt = opt_data.get('last_optimized')
                if last_opt and isinstance(last_opt, datetime) and last_opt < cutoff_date:
                    candidates.append(f"{epic} (optimization aged)")
                
                score = opt_data.get('performance_score', 0.0)
                if score < self.config.min_performance_score:
                    candidates.append(f"{epic} (low performance: {score:.3f})")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Could not analyze optimization candidates: {e}")
        
        return candidates


def main():
    """Main execution for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dynamic Zero-Lag Scanner')
    parser.add_argument('--initialize', action='store_true', help='Initialize optimized strategies')
    parser.add_argument('--scan', action='store_true', help='Run full scan')
    parser.add_argument('--status', action='store_true', help='Show optimization status')
    parser.add_argument('--refresh', action='store_true', help='Refresh optimization data')
    parser.add_argument('--epic', help='Get info for specific epic')
    parser.add_argument('--suggest', action='store_true', help='Suggest optimization candidates')
    
    args = parser.parse_args()
    
    if not any([args.initialize, args.scan, args.status, args.refresh, args.epic, args.suggest]):
        print("❌ Must specify at least one action")
        parser.print_help()
        return
    
    scanner = DynamicZeroLagScanner()
    
    try:
        if args.initialize:
            print("🚀 Initializing Dynamic Zero-Lag Scanner...")
            results = scanner.initialize_optimized_strategies()
            print(f"✅ Initialization complete: {len(results)} epics processed")
        
        if args.scan:
            print("🔍 Running full zero-lag scan...")
            signals = scanner.scan_all_optimized_epics()
            print(f"✅ Scan complete: {len(signals)} signals detected")
        
        if args.status:
            scanner.print_optimization_status()
        
        if args.epic:
            print(f"\n📊 Strategy info for {args.epic}:")
            info = scanner.get_epic_strategy_info(args.epic)
            for key, value in info.items():
                print(f"   {key}: {value}")
        
        if args.refresh:
            print("🔄 Refreshing optimization data...")
            scanner.refresh_optimization_data()
            print("✅ Refresh complete")
        
        if args.suggest:
            print("💡 Optimization candidates:")
            candidates = scanner.suggest_optimization_candidates()
            if candidates:
                for candidate in candidates:
                    print(f"   📈 {candidate}")
            else:
                print("   ✅ All strategies are well optimized")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()