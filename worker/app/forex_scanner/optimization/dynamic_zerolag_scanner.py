#!/usr/bin/env python3
"""
Dynamic Zero-Lag Scanner
Automatically uses optimal parameters per epic from optimization results
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

from core.strategies.zero_lag_strategy import ZeroLagStrategy
from core.data_fetcher import DataFetcher
from optimization.zerolag_parameter_service import get_zerolag_parameter_service

try:
    import config
except ImportError:
    from forex_scanner import config

logger = logging.getLogger(__name__)

class DynamicZeroLagScanner:
    """
    Smart scanner that automatically uses optimal Zero-Lag parameters per epic
    """
    
    def __init__(self):
        self.logger = logging.getLogger('dynamic_zl_scanner')
        
        # Initialize database manager and data fetcher
        from core.database import DatabaseManager
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager)
        self.parameter_service = get_zerolag_parameter_service()
        
        # Initialize results tracking
        self.scan_results = {}
        self.optimization_status = {}
        
        self.logger.info("🚀 Dynamic Zero-Lag Scanner initialized")
    
    def scan_optimized_epic(self, epic: str, timeframe: str = '15m') -> Optional[Dict]:
        """
        Scan single epic using optimal parameters if available
        """
        try:
            # Check if we have optimal parameters
            optimal_config = None
            if self.parameter_service:
                optimal_config = self.parameter_service.get_optimal_parameters(epic)
            
            if optimal_config:
                # Use optimal parameters
                strategy = ZeroLagStrategy(
                    data_fetcher=self.data_fetcher,
                    epic=epic,
                    use_optimal_parameters=True
                )
                
                self.logger.info(f"✅ Using optimal parameters for {epic}")
                self.logger.info(f"   Score: {optimal_config.composite_score:.2f} | Win Rate: {optimal_config.win_rate:.1%}")
                
            else:
                # Fall back to static configuration
                strategy = ZeroLagStrategy(
                    data_fetcher=self.data_fetcher,
                    epic=epic,
                    use_optimal_parameters=False
                )
                
                self.logger.warning(f"⚠️ No optimal parameters for {epic} - using static config")
            
            # Perform scan - get recent data for analysis
            # Extract pair from epic (e.g., CS.D.USDJPY.MINI.IP -> USDJPY)
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe=timeframe)
            
            if df.empty:
                self.logger.warning(f"❌ No data available for {epic}")
                return None
            
            # Detect signal using the correct method
            signal = strategy.detect_signal(df, epic, spread_pips=2.0, timeframe=timeframe)
            
            if not signal:
                self.logger.debug(f"📊 No signals found for {epic}")
                return None
            
            # Enhance signal with metadata
            signal['epic'] = epic
            signal['optimized'] = optimal_config is not None
            signal['timestamp'] = datetime.now()
            
            # Add optimization metadata if available
            if optimal_config:
                signal['optimization_metadata'] = {
                    'performance_score': optimal_config.composite_score,
                    'win_rate': optimal_config.win_rate,
                    'net_pips': optimal_config.net_pips,
                    'optimal_sl_pips': optimal_config.stop_loss_pips,
                    'optimal_tp_pips': optimal_config.take_profit_pips
                }
            
            self.logger.info(f"🎯 {epic} signal: {signal.get('signal_type', 'UNKNOWN')}")
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning {epic}: {e}")
            return None
    
    def scan_all_optimized_epics(self) -> Dict[str, Dict]:
        """
        Scan all epics that have optimal parameters
        """
        if not self.parameter_service:
            self.logger.error("❌ Parameter service not available")
            return {}
        
        # Get all optimized epics
        optimized_epics = self.parameter_service.get_all_optimized_epics()
        
        if not optimized_epics:
            self.logger.warning("⚠️ No optimized epics found")
            return {}
        
        self.logger.info(f"🔍 Scanning {len(optimized_epics)} optimized epics")
        
        results = {}
        for epic in optimized_epics.keys():
            signal = self.scan_optimized_epic(epic)
            if signal:
                results[epic] = signal
        
        self.logger.info(f"✅ Found signals for {len(results)} out of {len(optimized_epics)} optimized epics")
        return results
    
    def scan_mixed_portfolio(self, target_epics: List[str]) -> Dict[str, Dict]:
        """
        Scan a mixed portfolio of epics (optimized and non-optimized)
        """
        results = {}
        
        for epic in target_epics:
            signal = self.scan_optimized_epic(epic)
            if signal:
                results[epic] = signal
        
        return results
    
    def print_optimization_status(self):
        """Print optimization status for common epics"""
        if not self.parameter_service:
            print("❌ Parameter service not available")
            return
        
        self.parameter_service.print_optimization_status()
    
    def get_optimization_recommendations(self) -> Dict[str, str]:
        """Get recommendations for optimization based on current status"""
        if not self.parameter_service:
            return {}
        
        optimized_epics = self.parameter_service.get_all_optimized_epics()
        
        common_epics = [
            'CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP',
            'CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP', 'CS.D.NZDUSD.MINI.IP',
            'CS.D.EURJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP', 'CS.D.USDCHF.MINI.IP'
        ]
        
        recommendations = {}
        
        for epic in common_epics:
            if epic in optimized_epics:
                config = optimized_epics[epic]
                if config.composite_score > 20:
                    recommendations[epic] = f"✅ Ready for production (Score: {config.composite_score:.1f})"
                elif config.composite_score > 10:
                    recommendations[epic] = f"⚡ Good performance (Score: {config.composite_score:.1f})"
                else:
                    recommendations[epic] = f"⚠️ Re-optimize with longer timeframe (Score: {config.composite_score:.1f})"
            else:
                recommendations[epic] = "❌ Needs optimization - run smart-presets test first"
        
        return recommendations
    
    def suggest_optimization_commands(self) -> List[str]:
        """Suggest optimization commands for missing epics"""
        if not self.parameter_service:
            return []
        
        optimized_epics = self.parameter_service.get_all_optimized_epics()
        
        common_epics = [
            'CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP',
            'CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP', 'CS.D.NZDUSD.MINI.IP',
            'CS.D.EURJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP', 'CS.D.USDCHF.MINI.IP'
        ]
        
        unoptimized = [epic for epic in common_epics if epic not in optimized_epics]
        
        commands = []
        for epic in unoptimized[:3]:  # Limit to first 3
            commands.append(f"docker exec task-worker python forex_scanner/optimization/optimize_zerolag_parameters.py --epic {epic} --smart-presets")
        
        return commands


def main():
    """CLI interface for dynamic scanning"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dynamic Zero-Lag Scanner with Optimal Parameters')
    parser.add_argument('--epic', help='Scan specific epic')
    parser.add_argument('--all-optimized', action='store_true', help='Scan all optimized epics')
    parser.add_argument('--status', action='store_true', help='Show optimization status')
    parser.add_argument('--recommendations', action='store_true', help='Show optimization recommendations')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = DynamicZeroLagScanner()
    
    if args.status:
        scanner.print_optimization_status()
        return
    
    if args.recommendations:
        print("\n" + "="*60)
        print("💡 OPTIMIZATION RECOMMENDATIONS")
        print("="*60)
        
        recommendations = scanner.get_optimization_recommendations()
        for epic, rec in recommendations.items():
            print(f"{epic:<25} | {rec}")
        
        print("\n🚀 SUGGESTED COMMANDS:")
        commands = scanner.suggest_optimization_commands()
        for cmd in commands:
            print(f"   {cmd}")
        return
    
    if args.epic:
        # Scan specific epic
        signal = scanner.scan_optimized_epic(args.epic)
        if signal:
            print(f"✅ Signal found for {args.epic}:")
            print(f"   Type: {signal.get('signal_type', 'UNKNOWN')}")
            print(f"   Confidence: {signal.get('confidence_score', 0):.1%}")
            print(f"   Optimized: {signal.get('optimized', False)}")
        else:
            print(f"❌ No signal found for {args.epic}")
        return
    
    if args.all_optimized:
        # Scan all optimized epics
        results = scanner.scan_all_optimized_epics()
        
        print(f"\n📊 SCAN RESULTS ({len(results)} signals found):")
        print("="*60)
        
        for epic, signal in results.items():
            opt_flag = "🎯" if signal.get('optimized') else "📊"
            print(f"{opt_flag} {epic:<25} | {signal.get('signal_type', 'UNKNOWN'):<6} | {signal.get('confidence_score', 0):.1%}")
        
        return
    
    # Default: show status
    scanner.print_optimization_status()


if __name__ == "__main__":
    main()