#!/usr/bin/env python3
"""
Dynamic Scanner Integration
Integrates optimal parameters from optimization results into the forex scanner
"""

import sys
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from optimal_parameter_service import OptimalParameterService, MarketConditions
from core.strategies.ema_strategy import EMAStrategy
from core.data_fetcher import DataFetcher
from core.database import DatabaseManager

try:
    import config
except ImportError:
    from forex_scanner import config


class DynamicEMAScanner:
    """
    Enhanced EMA Scanner that automatically uses optimal parameters for each epic
    """
    
    def __init__(self):
        self.logger = logging.getLogger('dynamic_ema_scanner')
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager)
        self.parameter_service = OptimalParameterService()
        
        # Get all epics with optimal parameters
        self.optimal_epics = self._get_optimized_epics()
        self.logger.info(f"🎯 Dynamic scanner initialized with {len(self.optimal_epics)} optimized epics")
    
    def _get_optimized_epics(self) -> Dict[str, dict]:
        """Get all epics that have optimization data"""
        try:
            all_params = self.parameter_service.get_all_epic_parameters()
            optimized_epics = {}
            
            for epic, params in all_params.items():
                # Only include epics with actual optimization data (not fallbacks)
                if params.performance_score > 0.0:
                    optimized_epics[epic] = {
                        'epic': epic,
                        'ema_config': params.ema_config,
                        'confidence_threshold': params.confidence_threshold,
                        'timeframe': params.timeframe,
                        'smart_money_enabled': params.smart_money_enabled,
                        'stop_loss_pips': params.stop_loss_pips,
                        'take_profit_pips': params.take_profit_pips,
                        'performance_score': params.performance_score,
                        'last_optimized': params.last_optimized
                    }
            
            return optimized_epics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get optimized epics: {e}")
            return {}
    
    def create_optimized_strategy(self, epic: str, market_conditions: Optional[MarketConditions] = None) -> EMAStrategy:
        """Create EMA strategy with optimal parameters for specific epic"""
        try:
            # Create strategy with dynamic parameters enabled
            strategy = EMAStrategy(
                data_fetcher=self.data_fetcher,
                epic=epic,
                use_optimal_parameters=True,
                backtest_mode=False  # Live scanning mode
            )
            
            self.logger.info(f"✅ Created optimized EMA strategy for {epic}")
            return strategy
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create optimized strategy for {epic}: {e}")
            # Fallback to default strategy
            return EMAStrategy(data_fetcher=self.data_fetcher, epic=epic, use_optimal_parameters=False)
    
    def scan_all_optimized_epics(self, 
                                 timeframe: str = '15m', 
                                 market_conditions: Optional[MarketConditions] = None) -> Dict[str, List[dict]]:
        """
        Scan all epics with optimal parameters for signals
        
        Returns:
            Dict with epic as key and list of signals as value
        """
        all_signals = {}
        
        for epic in self.optimal_epics.keys():
            try:
                self.logger.info(f"🔍 Scanning {epic} with optimal parameters...")
                
                # Create optimized strategy for this epic
                strategy = self.create_optimized_strategy(epic, market_conditions)
                
                # Get optimal timeframe for this epic (or use provided)
                optimal_tf = strategy.get_optimal_timeframe() or timeframe
                
                # Fetch data for the optimal timeframe
                df = self.data_fetcher.get_price_data(
                    epic=epic,
                    timeframe=optimal_tf,
                    days=3  # Recent data for signal detection
                )
                
                if df is None or df.empty:
                    self.logger.warning(f"⚠️ No data available for {epic}")
                    continue
                
                # Detect signals using optimal parameters
                signal = strategy.detect_signal_auto(
                    df=df,
                    epic=epic,
                    timeframe=optimal_tf
                )
                
                if signal:
                    # Enhance signal with optimal parameter info
                    enhanced_signal = self._enhance_signal_with_optimal_params(signal, epic)
                    all_signals[epic] = [enhanced_signal]
                    self.logger.info(f"🎯 Signal detected for {epic}: {signal['signal_type']} at {signal.get('price', 'N/A')}")
                else:
                    all_signals[epic] = []
                
            except Exception as e:
                self.logger.error(f"❌ Error scanning {epic}: {e}")
                all_signals[epic] = []
        
        # Summary
        total_signals = sum(len(signals) for signals in all_signals.values())
        self.logger.info(f"🏁 Scan complete: {total_signals} signals across {len(self.optimal_epics)} epics")
        
        return all_signals
    
    def _enhance_signal_with_optimal_params(self, signal: dict, epic: str) -> dict:
        """Add optimal parameter information to signal"""
        optimal_params = self.parameter_service.get_epic_parameters(epic)
        
        enhanced_signal = signal.copy()
        enhanced_signal.update({
            'optimal_stop_loss_pips': optimal_params.stop_loss_pips,
            'optimal_take_profit_pips': optimal_params.take_profit_pips,
            'optimal_risk_reward_ratio': optimal_params.risk_reward_ratio,
            'optimization_performance_score': optimal_params.performance_score,
            'optimization_last_updated': optimal_params.last_optimized.isoformat(),
            'using_optimal_parameters': True
        })
        
        return enhanced_signal
    
    def get_optimization_status_report(self) -> Dict[str, any]:
        """Generate report on optimization status across all epics"""
        try:
            # Get all configured epics (from config)
            all_epics = getattr(config, 'TRADEABLE_EPICS', [])
            
            optimized_count = len(self.optimal_epics)
            unoptimized_epics = [epic for epic in all_epics if epic not in self.optimal_epics]
            
            # Calculate performance statistics
            if self.optimal_epics:
                scores = [params['performance_score'] for params in self.optimal_epics.values()]
                avg_performance = sum(scores) / len(scores)
                
                # Get oldest optimization date
                oldest_optimization = min(
                    params['last_optimized'] for params in self.optimal_epics.values()
                )
            else:
                avg_performance = 0.0
                oldest_optimization = None
            
            report = {
                'total_epics': len(all_epics),
                'optimized_epics': optimized_count,
                'unoptimized_epics': len(unoptimized_epics),
                'optimization_coverage': (optimized_count / len(all_epics)) * 100 if all_epics else 0,
                'average_performance_score': avg_performance,
                'oldest_optimization_date': oldest_optimization.isoformat() if oldest_optimization else None,
                'unoptimized_epic_list': unoptimized_epics,
                'optimized_epic_details': self.optimal_epics
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate optimization status report: {e}")
            return {'error': str(e)}
    
    def print_optimization_status(self):
        """Print formatted optimization status report"""
        report = self.get_optimization_status_report()
        
        if 'error' in report:
            print(f"❌ Error generating report: {report['error']}")
            return
        
        print("\n🎯 DYNAMIC SCANNER OPTIMIZATION STATUS")
        print("=" * 60)
        print(f"📊 Total Epics: {report['total_epics']}")
        print(f"✅ Optimized: {report['optimized_epics']} ({report['optimization_coverage']:.1f}%)")
        print(f"⚠️ Unoptimized: {report['unoptimized_epics']}")
        print(f"📈 Average Performance Score: {report['average_performance_score']:.3f}")
        
        if report['oldest_optimization_date']:
            print(f"📅 Oldest Optimization: {report['oldest_optimization_date'][:10]}")
        
        if report['unoptimized_epic_list']:
            print(f"\n⚠️ Unoptimized Epics:")
            for epic in report['unoptimized_epic_list']:
                print(f"   - {epic}")
        
        if report['optimized_epic_details']:
            print(f"\n✅ Optimized Epics (Top 5 by Performance):")
            sorted_epics = sorted(
                report['optimized_epic_details'].items(),
                key=lambda x: x[1]['performance_score'],
                reverse=True
            )[:5]
            
            for epic, params in sorted_epics:
                epic_short = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
                print(f"   🏆 {epic_short:<10} | {params['ema_config']:<12} | "
                      f"{params['confidence_threshold']:<5.0%} | "
                      f"{params['stop_loss_pips']:.0f}/{params['take_profit_pips']:.0f} | "
                      f"Score: {params['performance_score']:.3f}")
    
    def recommend_optimizations(self) -> List[str]:
        """Generate recommendations for improving the optimization system"""
        recommendations = []
        
        report = self.get_optimization_status_report()
        
        if report['optimization_coverage'] < 100:
            unoptimized = report['unoptimized_epics']
            recommendations.append(
                f"🎯 Run optimization for {unoptimized} unoptimized epics: "
                f"{', '.join(report['unoptimized_epic_list'][:3])}"
                f"{'...' if len(report['unoptimized_epic_list']) > 3 else ''}"
            )
        
        if report['oldest_optimization_date']:
            oldest_date = datetime.fromisoformat(report['oldest_optimization_date'])
            days_old = (datetime.now() - oldest_date).days
            
            if days_old > 30:
                recommendations.append(
                    f"🔄 Re-optimize old parameters (oldest: {days_old} days ago)"
                )
            
            if days_old > 7:
                recommendations.append(
                    f"📊 Consider running parameter sensitivity analysis"
                )
        
        if report['average_performance_score'] < 1.5:
            recommendations.append(
                f"⚠️ Low average performance score ({report['average_performance_score']:.3f}) - "
                f"consider expanding optimization parameter grid"
            )
        
        if not recommendations:
            recommendations.append("✅ Optimization system is up to date and performing well!")
        
        return recommendations


def main():
    """Main function to demonstrate dynamic scanner"""
    scanner = DynamicEMAScanner()
    
    # Print status report
    scanner.print_optimization_status()
    
    # Show recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    recommendations = scanner.recommend_optimizations()
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n🚀 Ready to scan with optimal parameters!")
    print(f"   Use: scanner.scan_all_optimized_epics() to scan all optimized epics")


if __name__ == "__main__":
    main()