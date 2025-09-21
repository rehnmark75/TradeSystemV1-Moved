#!/usr/bin/env python3
"""
Dynamic MACD Scanner Integration
Integrates optimized MACD parameters with the main scanner system
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.database import DatabaseManager
# MACDStrategy would be imported here when implemented
# from core.strategies.macd_strategy import MACDStrategy
from optimization.optimal_parameter_service import OptimalParameterService

try:
    import config
except ImportError:
    from forex_scanner import config


class DynamicMACDScanner:
    """
    Dynamic MACD scanner that uses optimized parameters from database
    """
    
    def __init__(self):
        self.logger = logging.getLogger('dynamic_macd_scanner')
        self.setup_logging()
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.parameter_service = OptimalParameterService()
        self.optimized_strategies = {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def get_optimized_macd_parameters(self, epic: str) -> Optional[Dict]:
        """Get optimized MACD parameters for specific epic"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        best_fast_ema, best_slow_ema, best_signal_ema,
                        best_confidence_threshold, best_timeframe,
                        best_histogram_threshold, best_zero_line_filter,
                        best_rsi_filter_enabled, best_momentum_confirmation,
                        best_mtf_enabled, best_mtf_timeframes,
                        best_smart_money_enabled, optimal_stop_loss_pips,
                        optimal_take_profit_pips, best_composite_score,
                        best_win_rate, last_updated
                    FROM macd_best_parameters 
                    WHERE epic = %s
                """, (epic,))
                
                result = cursor.fetchone()
                if not result:
                    self.logger.warning(f"No optimized MACD parameters found for {epic}")
                    return None
                
                return {
                    'fast_ema': int(result[0]),
                    'slow_ema': int(result[1]),
                    'signal_ema': int(result[2]),
                    'confidence_threshold': float(result[3]),
                    'timeframe': result[4],
                    'histogram_threshold': float(result[5]),
                    'zero_line_filter': result[6],
                    'rsi_filter_enabled': result[7],
                    'momentum_confirmation': result[8],
                    'mtf_enabled': result[9],
                    'mtf_timeframes': result[10],
                    'smart_money_enabled': result[11],
                    'stop_loss_pips': float(result[12]),
                    'take_profit_pips': float(result[13]),
                    'composite_score': float(result[14]),
                    'win_rate': float(result[15]),
                    'last_updated': result[16],
                    'source': 'database_optimization'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get optimized parameters for {epic}: {e}")
            return None
    
    def create_optimized_macd_strategy(self, epic: str) -> Optional[object]:
        """Create MACD strategy instance with optimized parameters"""
        params = self.get_optimized_macd_parameters(epic)
        if not params:
            # Fallback to config-based parameters
            return self.create_fallback_macd_strategy(epic)
        
        # For now, return a mock strategy object since MACDStrategy is not implemented yet
        # In production, this would create actual MACDStrategy instances
        class MockMACDStrategy:
            def __init__(self, epic, **params):
                self.epic = epic
                self.params = params
            
            def scan_for_signals(self):
                return []  # Return empty signals for now
        
        try:
            strategy = MockMACDStrategy(
                epic=epic,
                fast_ema=params['fast_ema'],
                slow_ema=params['slow_ema'],
                signal_ema=params['signal_ema'],
                confidence_threshold=params['confidence_threshold'],
                timeframe=params['timeframe'],
                histogram_threshold=params['histogram_threshold'],
                use_optimized_parameters=True
            )
            
            self.logger.info(f"✅ Created optimized MACD strategy for {epic} "
                           f"({params['fast_ema']}/{params['slow_ema']}/{params['signal_ema']}, "
                           f"Score: {params['composite_score']:.6f})")
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Failed to create optimized MACD strategy for {epic}: {e}")
            return self.create_fallback_macd_strategy(epic)
    
    def create_fallback_macd_strategy(self, epic: str) -> object:
        """Create fallback MACD strategy using config parameters"""
        self.logger.warning(f"Using fallback MACD parameters for {epic}")
        
        from configdata.strategies.config_macd_strategy import MACD_PERIODS
        
        # Mock strategy for testing
        class MockMACDStrategy:
            def __init__(self, epic, **params):
                self.epic = epic
                self.params = params
            
            def scan_for_signals(self):
                return []
        
        strategy = MockMACDStrategy(
            epic=epic,
            fast_ema=MACD_PERIODS['fast_ema'],
            slow_ema=MACD_PERIODS['slow_ema'],
            signal_ema=MACD_PERIODS['signal_ema'],
            confidence_threshold=0.55,
            timeframe='15m',
            use_optimized_parameters=False
        )
        
        return strategy
    
    def scan_all_optimized_epics(self) -> List[Dict]:
        """Scan all epics with optimized MACD parameters"""
        signals = []
        
        # Get list of optimized epics
        optimized_epics = self.get_optimized_epics_list()
        
        if not optimized_epics:
            self.logger.warning("No optimized MACD epics found. Run optimization first.")
            return signals
        
        self.logger.info(f"🎯 Scanning {len(optimized_epics)} optimized MACD epics...")
        
        for epic in optimized_epics:
            try:
                strategy = self.create_optimized_macd_strategy(epic)
                if strategy:
                    epic_signals = strategy.scan_for_signals()
                    if epic_signals:
                        signals.extend(epic_signals)
                        self.logger.info(f"📊 {epic}: Found {len(epic_signals)} MACD signals")
                    
            except Exception as e:
                self.logger.error(f"Failed to scan {epic}: {e}")
        
        self.logger.info(f"🎉 MACD Scan completed: {len(signals)} total signals from {len(optimized_epics)} epics")
        return signals
    
    def get_optimized_epics_list(self) -> List[str]:
        """Get list of epics with optimized parameters"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT epic FROM macd_best_parameters 
                    ORDER BY best_composite_score DESC
                """)
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Failed to get optimized epics list: {e}")
            return []
    
    def get_optimization_status(self) -> Dict:
        """Get optimization status across all configured epics"""
        try:
            configured_epics = set(config.EPIC_LIST)
            optimized_epics = set(self.get_optimized_epics_list())
            
            missing_epics = configured_epics - optimized_epics
            extra_epics = optimized_epics - configured_epics
            
            return {
                'total_configured': len(configured_epics),
                'total_optimized': len(optimized_epics),
                'optimization_coverage': len(optimized_epics) / len(configured_epics) * 100,
                'missing_epics': list(missing_epics),
                'extra_epics': list(extra_epics),
                'ready_for_production': len(missing_epics) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization status: {e}")
            return {}
    
    def print_optimization_status(self):
        """Print detailed optimization status report"""
        status = self.get_optimization_status()
        
        print("\n" + "="*80)
        print("🎯 DYNAMIC MACD SCANNER OPTIMIZATION STATUS")
        print("="*80)
        
        if not status:
            print("❌ Unable to retrieve optimization status")
            return
        
        print(f"📊 EPIC COVERAGE:")
        print(f"   • Configured Epics: {status['total_configured']}")
        print(f"   • Optimized Epics: {status['total_optimized']}")
        print(f"   • Coverage: {status['optimization_coverage']:.1f}%")
        
        if status['ready_for_production']:
            print(f"   ✅ System ready for production MACD scanning!")
        else:
            print(f"   ⚠️  Missing optimization for {len(status['missing_epics'])} epics")
        
        if status['missing_epics']:
            print(f"\n❌ MISSING OPTIMIZATION:")
            for epic in status['missing_epics'][:10]:  # Show first 10
                print(f"   • {epic}")
            if len(status['missing_epics']) > 10:
                print(f"   • ... and {len(status['missing_epics']) - 10} more")
        
        if status['extra_epics']:
            print(f"\n💡 EXTRA OPTIMIZED EPICS:")
            for epic in status['extra_epics'][:5]:  # Show first 5
                print(f"   • {epic}")
            if len(status['extra_epics']) > 5:
                print(f"   • ... and {len(status['extra_epics']) - 5} more")
        
        # Show top performing optimized strategies
        top_performers = self.get_top_performing_strategies(5)
        if top_performers:
            print(f"\n🏆 TOP PERFORMING MACD STRATEGIES:")
            print(f"{'Epic':<20} {'MACD':<12} {'Win%':<6} {'Score':<10} {'Updated':<12}")
            print("-" * 70)
            
            for performer in top_performers:
                epic = performer['epic'][:18]
                macd_config = f"{performer['fast_ema']}/{performer['slow_ema']}/{performer['signal_ema']}"
                win_rate = f"{performer['win_rate']:.1%}"
                score = f"{performer['composite_score']:.6f}"
                updated = performer['last_updated'].strftime('%Y-%m-%d') if performer['last_updated'] else 'N/A'
                
                print(f"{epic:<20} {macd_config:<12} {win_rate:<6} {score:<10} {updated:<12}")
        
        print("="*80)
    
    def get_top_performing_strategies(self, limit: int = 10) -> List[Dict]:
        """Get top performing MACD strategies by composite score"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        epic, best_fast_ema, best_slow_ema, best_signal_ema,
                        best_win_rate, best_composite_score, last_updated
                    FROM macd_best_parameters 
                    ORDER BY best_composite_score DESC
                    LIMIT %s
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'epic': row[0],
                        'fast_ema': int(row[1]),
                        'slow_ema': int(row[2]),
                        'signal_ema': int(row[3]),
                        'win_rate': float(row[4]) if row[4] else 0,
                        'composite_score': float(row[5]) if row[5] else 0,
                        'last_updated': row[6]
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get top performing strategies: {e}")
            return []
    
    def validate_system_readiness(self) -> Dict:
        """Validate if the dynamic MACD system is ready for production"""
        try:
            # Check database connectivity
            db_ready = self.db_manager.test_connection()
            
            # Check optimization coverage
            status = self.get_optimization_status()
            optimization_ready = status.get('ready_for_production', False)
            
            # Check parameter quality
            top_performers = self.get_top_performing_strategies(5)
            quality_threshold = 0.1  # Minimum composite score
            quality_ready = any(p['composite_score'] > quality_threshold for p in top_performers)
            
            # Overall system readiness
            system_ready = db_ready and optimization_ready and quality_ready
            
            return {
                'database_ready': db_ready,
                'optimization_ready': optimization_ready,
                'quality_ready': quality_ready,
                'system_ready': system_ready,
                'coverage_percentage': status.get('optimization_coverage', 0),
                'top_score': max([p['composite_score'] for p in top_performers], default=0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to validate system readiness: {e}")
            return {
                'database_ready': False,
                'optimization_ready': False,
                'quality_ready': False,
                'system_ready': False,
                'coverage_percentage': 0,
                'top_score': 0
            }


def main():
    """Main function for testing dynamic MACD scanner"""
    scanner = DynamicMACDScanner()
    
    print("🎯 Testing Dynamic MACD Scanner Integration")
    
    # Print optimization status
    scanner.print_optimization_status()
    
    # Validate system readiness
    readiness = scanner.validate_system_readiness()
    
    print(f"\n🔍 SYSTEM READINESS VALIDATION:")
    print(f"   • Database: {'✅' if readiness['database_ready'] else '❌'}")
    print(f"   • Optimization: {'✅' if readiness['optimization_ready'] else '❌'}")
    print(f"   • Quality: {'✅' if readiness['quality_ready'] else '❌'}")
    print(f"   • Overall: {'✅ READY' if readiness['system_ready'] else '❌ NOT READY'}")
    
    # Test strategy creation for a sample epic
    test_epic = "CS.D.EURUSD.CEEM.IP"
    print(f"\n🧪 Testing strategy creation for {test_epic}...")
    
    strategy = scanner.create_optimized_macd_strategy(test_epic)
    if strategy:
        print(f"✅ Successfully created optimized MACD strategy for {test_epic}")
    else:
        print(f"❌ Failed to create strategy for {test_epic}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()