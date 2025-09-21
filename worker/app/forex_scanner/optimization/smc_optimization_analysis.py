#!/usr/bin/env python3
"""
SMC Optimization Analysis Tools

Provides comprehensive analysis capabilities for SMC strategy optimization results.
Similar to EMA and MACD analysis tools but tailored for Smart Money Concepts metrics.

Features:
- Performance analysis by epic and configuration
- SMC-specific metrics analysis (structure breaks, order blocks, FVGs)
- Parameter impact analysis and recommendations
- Confluence accuracy analysis
- Best configuration identification and comparison
- Optimization history and trend analysis

Usage:
    # Summary analysis for all epics
    python smc_optimization_analysis.py --summary
    
    # Detailed analysis for specific epic
    python smc_optimization_analysis.py --epic CS.D.EURUSD.CEEM.IP --detailed
    
    # Top performing configurations
    python smc_optimization_analysis.py --top-performers --limit 10
    
    # Parameter impact analysis
    python smc_optimization_analysis.py --parameter-analysis --epic CS.D.EURUSD.CEEM.IP
    
    # Compare configurations
    python smc_optimization_analysis.py --compare-configs --epic CS.D.EURUSD.CEEM.IP
"""

import argparse
import logging
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

# Add the worker/app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from forex_scanner.core.database.database_manager import DatabaseManager

class SMCOptimizationAnalyzer:
    """Analyzes SMC optimization results and provides insights"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
    
    def get_summary_statistics(self) -> Dict:
        """Get overall optimization summary statistics"""
        
        try:
            query = """
            SELECT 
                COUNT(DISTINCT epic) as total_epics_optimized,
                COUNT(DISTINCT run_id) as total_optimization_runs,
                COUNT(*) as total_parameter_tests,
                AVG(win_rate) as avg_win_rate,
                MAX(win_rate) as max_win_rate,
                AVG(profit_factor) as avg_profit_factor,
                MAX(profit_factor) as max_profit_factor,
                AVG(performance_score) as avg_performance_score,
                MAX(performance_score) as max_performance_score,
                AVG(confluence_accuracy) as avg_confluence_accuracy,
                MAX(confluence_accuracy) as max_confluence_accuracy,
                AVG(structure_breaks_detected) as avg_structure_breaks,
                AVG(order_block_reactions) as avg_order_block_reactions,
                AVG(fvg_reactions) as avg_fvg_reactions
            FROM smc_optimization_results
            """
            
            results = self.db_manager.execute_query(query, fetch_results=True)
            return results[0] if results else {}
            
        except Exception as e:
            self.logger.error(f"Failed to get summary statistics: {e}")
            return {}
    
    def get_epic_analysis(self, epic: str, detailed: bool = False) -> Dict:
        """Get detailed analysis for a specific epic"""
        
        try:
            # Basic epic statistics
            basic_query = """
            SELECT 
                COUNT(*) as total_tests,
                AVG(win_rate) as avg_win_rate,
                MAX(win_rate) as max_win_rate,
                MIN(win_rate) as min_win_rate,
                AVG(profit_factor) as avg_profit_factor,
                MAX(profit_factor) as max_profit_factor,
                AVG(net_pips) as avg_net_pips,
                MAX(net_pips) as max_net_pips,
                AVG(performance_score) as avg_performance_score,
                MAX(performance_score) as max_performance_score,
                AVG(confluence_accuracy) as avg_confluence_accuracy,
                MAX(confluence_accuracy) as max_confluence_accuracy,
                AVG(structure_breaks_detected) as avg_structure_breaks,
                AVG(order_block_reactions) as avg_order_block_reactions,
                AVG(fvg_reactions) as avg_fvg_reactions,
                COUNT(DISTINCT smc_config) as configs_tested,
                COUNT(DISTINCT timeframe) as timeframes_tested
            FROM smc_optimization_results
            WHERE epic = %s
            """
            
            basic_results = self.db_manager.execute_query(basic_query, (epic,), fetch_results=True)
            analysis = basic_results[0] if basic_results else {}
            
            # Best configuration
            best_query = """
            SELECT 
                smc_config, confidence_level, timeframe, stop_loss_pips, take_profit_pips,
                win_rate, profit_factor, net_pips, performance_score,
                confluence_accuracy, structure_breaks_detected, order_block_reactions, fvg_reactions,
                swing_length, structure_confirmation, order_block_length, fvg_min_size,
                confluence_required, min_risk_reward
            FROM smc_optimization_results
            WHERE epic = %s
            ORDER BY performance_score DESC
            LIMIT 1
            """
            
            best_results = self.db_manager.execute_query(best_query, (epic,), fetch_results=True)
            analysis['best_configuration'] = best_results[0] if best_results else {}
            
            if detailed:
                # Configuration performance breakdown
                config_query = """
                SELECT 
                    smc_config,
                    COUNT(*) as tests,
                    AVG(win_rate) as avg_win_rate,
                    MAX(win_rate) as max_win_rate,
                    AVG(profit_factor) as avg_profit_factor,
                    AVG(performance_score) as avg_performance_score,
                    AVG(confluence_accuracy) as avg_confluence_accuracy
                FROM smc_optimization_results
                WHERE epic = %s
                GROUP BY smc_config
                ORDER BY avg_performance_score DESC
                """
                
                config_results = self.db_manager.execute_query(config_query, (epic,), fetch_results=True)
                analysis['config_breakdown'] = config_results or []
                
                # Timeframe analysis
                timeframe_query = """
                SELECT 
                    timeframe,
                    COUNT(*) as tests,
                    AVG(win_rate) as avg_win_rate,
                    AVG(profit_factor) as avg_profit_factor,
                    AVG(performance_score) as avg_performance_score,
                    AVG(confluence_accuracy) as avg_confluence_accuracy
                FROM smc_optimization_results
                WHERE epic = %s
                GROUP BY timeframe
                ORDER BY avg_performance_score DESC
                """
                
                timeframe_results = self.db_manager.execute_query(timeframe_query, (epic,), fetch_results=True)
                analysis['timeframe_breakdown'] = timeframe_results or []
                
                # Risk/Reward analysis
                rr_query = """
                SELECT 
                    risk_reward_ratio,
                    COUNT(*) as tests,
                    AVG(win_rate) as avg_win_rate,
                    AVG(profit_factor) as avg_profit_factor,
                    AVG(performance_score) as avg_performance_score
                FROM smc_optimization_results
                WHERE epic = %s
                GROUP BY risk_reward_ratio
                ORDER BY avg_performance_score DESC
                """
                
                rr_results = self.db_manager.execute_query(rr_query, (epic,), fetch_results=True)
                analysis['risk_reward_breakdown'] = rr_results or []
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to get epic analysis for {epic}: {e}")
            return {}
    
    def get_top_performers(self, limit: int = 10) -> List[Dict]:
        """Get top performing configurations across all epics"""
        
        try:
            query = """
            SELECT 
                epic, smc_config, confidence_level, timeframe,
                stop_loss_pips, take_profit_pips, risk_reward_ratio,
                win_rate, profit_factor, net_pips, performance_score,
                confluence_accuracy, structure_breaks_detected, 
                order_block_reactions, fvg_reactions,
                tested_at
            FROM smc_optimization_results
            ORDER BY performance_score DESC
            LIMIT %s
            """
            
            results = self.db_manager.execute_query(query, (limit,), fetch_results=True)
            return results or []
            
        except Exception as e:
            self.logger.error(f"Failed to get top performers: {e}")
            return []
    
    def analyze_parameter_impact(self, epic: str = None) -> Dict:
        """Analyze the impact of different parameters on performance"""
        
        try:
            base_where = "WHERE epic = %s" if epic else ""
            params = (epic,) if epic else ()
            
            analysis = {}
            
            # SMC Configuration impact
            config_query = f"""
            SELECT 
                smc_config,
                COUNT(*) as tests,
                AVG(performance_score) as avg_score,
                AVG(win_rate) as avg_win_rate,
                AVG(confluence_accuracy) as avg_confluence_accuracy,
                STDDEV(performance_score) as score_stddev
            FROM smc_optimization_results
            {base_where}
            GROUP BY smc_config
            ORDER BY avg_score DESC
            """
            
            analysis['config_impact'] = self.db_manager.execute_query(config_query, params, fetch_results=True) or []
            
            # Confidence level impact
            confidence_query = f"""
            SELECT 
                confidence_level,
                COUNT(*) as tests,
                AVG(performance_score) as avg_score,
                AVG(win_rate) as avg_win_rate,
                AVG(confluence_accuracy) as avg_confluence_accuracy
            FROM smc_optimization_results
            {base_where}
            GROUP BY confidence_level
            ORDER BY avg_score DESC
            """
            
            analysis['confidence_impact'] = self.db_manager.execute_query(confidence_query, params, fetch_results=True) or []
            
            # Timeframe impact
            timeframe_query = f"""
            SELECT 
                timeframe,
                COUNT(*) as tests,
                AVG(performance_score) as avg_score,
                AVG(win_rate) as avg_win_rate,
                AVG(confluence_accuracy) as avg_confluence_accuracy
            FROM smc_optimization_results
            {base_where}
            GROUP BY timeframe
            ORDER BY avg_score DESC
            """
            
            analysis['timeframe_impact'] = self.db_manager.execute_query(timeframe_query, params, fetch_results=True) or []
            
            # Confluence requirement impact
            confluence_query = f"""
            SELECT 
                confluence_required,
                COUNT(*) as tests,
                AVG(performance_score) as avg_score,
                AVG(win_rate) as avg_win_rate,
                AVG(confluence_accuracy) as avg_confluence_accuracy
            FROM smc_optimization_results
            {base_where}
            GROUP BY confluence_required
            ORDER BY avg_score DESC
            """
            
            analysis['confluence_impact'] = self.db_manager.execute_query(confluence_query, params, fetch_results=True) or []
            
            # Risk/Reward impact
            rr_query = f"""
            SELECT 
                risk_reward_ratio,
                COUNT(*) as tests,
                AVG(performance_score) as avg_score,
                AVG(win_rate) as avg_win_rate,
                AVG(profit_factor) as avg_profit_factor
            FROM smc_optimization_results
            {base_where}
            GROUP BY risk_reward_ratio
            ORDER BY avg_score DESC
            """
            
            analysis['risk_reward_impact'] = self.db_manager.execute_query(rr_query, params, fetch_results=True) or []
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze parameter impact: {e}")
            return {}
    
    def compare_configurations(self, epic: str = None) -> Dict:
        """Compare different SMC configurations side by side"""
        
        try:
            base_where = "WHERE epic = %s" if epic else ""
            params = (epic,) if epic else ()
            
            query = f"""
            SELECT 
                smc_config,
                COUNT(*) as total_tests,
                AVG(win_rate) as avg_win_rate,
                MAX(win_rate) as max_win_rate,
                AVG(profit_factor) as avg_profit_factor,
                MAX(profit_factor) as max_profit_factor,
                AVG(net_pips) as avg_net_pips,
                MAX(net_pips) as max_net_pips,
                AVG(performance_score) as avg_performance_score,
                MAX(performance_score) as max_performance_score,
                AVG(confluence_accuracy) as avg_confluence_accuracy,
                MAX(confluence_accuracy) as max_confluence_accuracy,
                AVG(structure_breaks_detected) as avg_structure_breaks,
                AVG(order_block_reactions) as avg_order_block_reactions,
                AVG(fvg_reactions) as avg_fvg_reactions,
                STDDEV(performance_score) as performance_consistency
            FROM smc_optimization_results
            {base_where}
            GROUP BY smc_config
            ORDER BY max_performance_score DESC
            """
            
            results = self.db_manager.execute_query(query, params, fetch_results=True)
            return {'comparisons': results or []}
            
        except Exception as e:
            self.logger.error(f"Failed to compare configurations: {e}")
            return {}
    
    def get_optimization_history(self, days: int = 30) -> List[Dict]:
        """Get optimization history for the last N days"""
        
        try:
            query = """
            SELECT 
                run_id, epic, optimization_mode, status,
                total_combinations, completed_combinations,
                best_score, start_time, end_time,
                EXTRACT(EPOCH FROM (end_time - start_time))/60 as duration_minutes
            FROM smc_optimization_runs
            WHERE start_time >= %s
            ORDER BY start_time DESC
            """
            
            cutoff_date = datetime.now() - timedelta(days=days)
            results = self.db_manager.execute_query(query, (cutoff_date,), fetch_results=True)
            return results or []
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization history: {e}")
            return []
    
    def print_summary_report(self):
        """Print comprehensive summary report"""
        
        try:
            print("\n" + "="*80)
            print("🧠 SMC OPTIMIZATION ANALYSIS SUMMARY")
            print("="*80)
            
            # Overall statistics
            summary = self.get_summary_statistics()
            if summary:
                print(f"\n📊 OVERALL STATISTICS:")
                print(f"   🎯 Epics Optimized: {summary.get('total_epics_optimized', 0)}")
                print(f"   🔄 Optimization Runs: {summary.get('total_optimization_runs', 0)}")
                print(f"   🧪 Parameter Tests: {summary.get('total_parameter_tests', 0):,}")
                print(f"   📈 Avg Win Rate: {summary.get('avg_win_rate', 0):.1f}% (Max: {summary.get('max_win_rate', 0):.1f}%)")
                print(f"   💹 Avg Profit Factor: {summary.get('avg_profit_factor', 0):.3f} (Max: {summary.get('max_profit_factor', 0):.3f})")
                print(f"   🏆 Avg Performance Score: {summary.get('avg_performance_score', 0):.6f} (Max: {summary.get('max_performance_score', 0):.6f})")
                print(f"   🧠 Avg Confluence Accuracy: {summary.get('avg_confluence_accuracy', 0):.1f}% (Max: {summary.get('max_confluence_accuracy', 0):.1f}%)")
                print(f"   📊 Avg Structure Breaks: {summary.get('avg_structure_breaks', 0):.1f}")
                print(f"   🏦 Avg Order Block Reactions: {summary.get('avg_order_block_reactions', 0):.1f}")
                print(f"   📐 Avg FVG Reactions: {summary.get('avg_fvg_reactions', 0):.1f}")
            
            # Top performers
            print(f"\n🌟 TOP 5 PERFORMERS:")
            top_performers = self.get_top_performers(5)
            for i, performer in enumerate(top_performers, 1):
                print(f"   {i}. {performer['epic']}: {performer['smc_config']} | "
                      f"TF: {performer['timeframe']} | "
                      f"Score: {performer['performance_score']:.6f} | "
                      f"Win: {performer['win_rate']:.1f}% | "
                      f"PF: {performer['profit_factor']:.3f} | "
                      f"Confluence: {performer['confluence_accuracy']:.1f}%")
            
            # Configuration comparison
            print(f"\n🔧 CONFIGURATION PERFORMANCE:")
            config_comparison = self.compare_configurations()
            for config in config_comparison.get('comparisons', []):
                print(f"   {config['smc_config']}: "
                      f"Tests: {config['total_tests']:,} | "
                      f"Max Score: {config['max_performance_score']:.6f} | "
                      f"Avg Win: {config['avg_win_rate']:.1f}% | "
                      f"Avg Confluence: {config['avg_confluence_accuracy']:.1f}%")
            
            print("="*80 + "\n")
            
        except Exception as e:
            self.logger.error(f"Failed to print summary report: {e}")
    
    def print_epic_report(self, epic: str, detailed: bool = False):
        """Print detailed epic analysis report"""
        
        try:
            print(f"\n" + "="*80)
            print(f"🎯 SMC OPTIMIZATION ANALYSIS: {epic}")
            print("="*80)
            
            analysis = self.get_epic_analysis(epic, detailed)
            
            # Basic statistics
            print(f"\n📊 BASIC STATISTICS:")
            print(f"   🧪 Total Tests: {analysis.get('total_tests', 0):,}")
            print(f"   📈 Win Rate: Avg {analysis.get('avg_win_rate', 0):.1f}% | Max {analysis.get('max_win_rate', 0):.1f}% | Min {analysis.get('min_win_rate', 0):.1f}%")
            print(f"   💹 Profit Factor: Avg {analysis.get('avg_profit_factor', 0):.3f} | Max {analysis.get('max_profit_factor', 0):.3f}")
            print(f"   💰 Net Pips: Avg {analysis.get('avg_net_pips', 0):.1f} | Max {analysis.get('max_net_pips', 0):.1f}")
            print(f"   🏆 Performance Score: Avg {analysis.get('avg_performance_score', 0):.6f} | Max {analysis.get('max_performance_score', 0):.6f}")
            print(f"   🧠 Confluence Accuracy: Avg {analysis.get('avg_confluence_accuracy', 0):.1f}% | Max {analysis.get('max_confluence_accuracy', 0):.1f}%")
            print(f"   🔧 Configurations Tested: {analysis.get('configs_tested', 0)}")
            print(f"   ⏰ Timeframes Tested: {analysis.get('timeframes_tested', 0)}")
            
            # Best configuration
            best = analysis.get('best_configuration', {})
            if best:
                print(f"\n🏆 BEST CONFIGURATION:")
                print(f"   Config: {best['smc_config']} | Confidence: {best['confidence_level']} | TF: {best['timeframe']}")
                print(f"   Performance: {best['performance_score']:.6f} | Win Rate: {best['win_rate']:.1f}% | PF: {best['profit_factor']:.3f}")
                print(f"   Net Pips: {best['net_pips']:.1f} | SL/TP: {best['stop_loss_pips']}/{best['take_profit_pips']}")
                print(f"   SMC Metrics: Structures: {best['structure_breaks_detected']}, OB: {best['order_block_reactions']}, FVG: {best['fvg_reactions']}")
                print(f"   Confluence Accuracy: {best['confluence_accuracy']:.1f}%")
                print(f"   Parameters: Swing: {best['swing_length']}, Confirm: {best['structure_confirmation']}, OB Len: {best['order_block_length']}")
                print(f"   FVG Size: {best['fvg_min_size']}, Confluence: {best['confluence_required']}, Min R:R: {best['min_risk_reward']}")
            
            if detailed:
                # Configuration breakdown
                print(f"\n🔧 CONFIGURATION BREAKDOWN:")
                for config in analysis.get('config_breakdown', []):
                    print(f"   {config['smc_config']}: "
                          f"Tests: {config['tests']:,} | "
                          f"Avg Score: {config['avg_performance_score']:.6f} | "
                          f"Max Win: {config['max_win_rate']:.1f}% | "
                          f"Confluence: {config['avg_confluence_accuracy']:.1f}%")
                
                # Timeframe breakdown
                print(f"\n⏰ TIMEFRAME BREAKDOWN:")
                for tf in analysis.get('timeframe_breakdown', []):
                    print(f"   {tf['timeframe']}: "
                          f"Tests: {tf['tests']:,} | "
                          f"Avg Score: {tf['avg_performance_score']:.6f} | "
                          f"Win: {tf['avg_win_rate']:.1f}% | "
                          f"Confluence: {tf['avg_confluence_accuracy']:.1f}%")
                
                # Risk/Reward breakdown
                print(f"\n💰 RISK/REWARD BREAKDOWN:")
                for rr in analysis.get('risk_reward_breakdown', []):
                    print(f"   R:R {rr['risk_reward_ratio']}: "
                          f"Tests: {rr['tests']:,} | "
                          f"Avg Score: {rr['avg_performance_score']:.6f} | "
                          f"Win: {rr['avg_win_rate']:.1f}% | "
                          f"PF: {rr['avg_profit_factor']:.3f}")
            
            print("="*80 + "\n")
            
        except Exception as e:
            self.logger.error(f"Failed to print epic report for {epic}: {e}")
    
    def print_parameter_impact_report(self, epic: str = None):
        """Print parameter impact analysis report"""
        
        try:
            scope = f"for {epic}" if epic else "across all epics"
            print(f"\n" + "="*80)
            print(f"🔬 SMC PARAMETER IMPACT ANALYSIS {scope.upper()}")
            print("="*80)
            
            impact = self.analyze_parameter_impact(epic)
            
            # Configuration impact
            print(f"\n🔧 SMC CONFIGURATION IMPACT:")
            for config in impact.get('config_impact', []):
                consistency = "High" if config.get('score_stddev', 0) < 0.01 else "Medium" if config.get('score_stddev', 0) < 0.02 else "Low"
                print(f"   {config['smc_config']}: "
                      f"Score: {config['avg_score']:.6f} | "
                      f"Win: {config['avg_win_rate']:.1f}% | "
                      f"Confluence: {config['avg_confluence_accuracy']:.1f}% | "
                      f"Tests: {config['tests']:,} | "
                      f"Consistency: {consistency}")
            
            # Confidence level impact
            print(f"\n🎯 CONFIDENCE LEVEL IMPACT:")
            for conf in impact.get('confidence_impact', []):
                print(f"   {conf['confidence_level']}: "
                      f"Score: {conf['avg_score']:.6f} | "
                      f"Win: {conf['avg_win_rate']:.1f}% | "
                      f"Confluence: {conf['avg_confluence_accuracy']:.1f}% | "
                      f"Tests: {conf['tests']:,}")
            
            # Timeframe impact
            print(f"\n⏰ TIMEFRAME IMPACT:")
            for tf in impact.get('timeframe_impact', []):
                print(f"   {tf['timeframe']}: "
                      f"Score: {tf['avg_score']:.6f} | "
                      f"Win: {tf['avg_win_rate']:.1f}% | "
                      f"Confluence: {tf['avg_confluence_accuracy']:.1f}% | "
                      f"Tests: {tf['tests']:,}")
            
            # Confluence requirement impact
            print(f"\n🧠 CONFLUENCE REQUIREMENT IMPACT:")
            for conf in impact.get('confluence_impact', []):
                print(f"   {conf['confluence_required']}: "
                      f"Score: {conf['avg_score']:.6f} | "
                      f"Win: {conf['avg_win_rate']:.1f}% | "
                      f"Confluence: {conf['avg_confluence_accuracy']:.1f}% | "
                      f"Tests: {conf['tests']:,}")
            
            # Risk/Reward impact
            print(f"\n💰 RISK/REWARD IMPACT:")
            for rr in impact.get('risk_reward_impact', []):
                print(f"   R:R {rr['risk_reward_ratio']}: "
                      f"Score: {rr['avg_score']:.6f} | "
                      f"Win: {rr['avg_win_rate']:.1f}% | "
                      f"PF: {rr['avg_profit_factor']:.3f} | "
                      f"Tests: {rr['tests']:,}")
            
            print("="*80 + "\n")
            
        except Exception as e:
            self.logger.error(f"Failed to print parameter impact report: {e}")


def main():
    """Main analysis function"""
    
    parser = argparse.ArgumentParser(description='SMC Optimization Analysis Tools')
    parser.add_argument('--summary', action='store_true', help='Show summary analysis')
    parser.add_argument('--epic', type=str, help='Analyze specific epic')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    parser.add_argument('--top-performers', action='store_true', help='Show top performers')
    parser.add_argument('--limit', type=int, default=10, help='Limit for top performers')
    parser.add_argument('--parameter-analysis', action='store_true', help='Analyze parameter impact')
    parser.add_argument('--compare-configs', action='store_true', help='Compare configurations')
    parser.add_argument('--history', action='store_true', help='Show optimization history')
    parser.add_argument('--days', type=int, default=30, help='Days for history analysis')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        analyzer = SMCOptimizationAnalyzer()
        
        if args.summary:
            analyzer.print_summary_report()
        
        if args.epic:
            analyzer.print_epic_report(args.epic, args.detailed)
        
        if args.top_performers:
            print(f"\n🌟 TOP {args.limit} PERFORMERS:")
            print("="*80)
            performers = analyzer.get_top_performers(args.limit)
            for i, performer in enumerate(performers, 1):
                print(f"{i:2d}. {performer['epic']} | {performer['smc_config']} | "
                      f"TF: {performer['timeframe']} | Conf: {performer['confidence_level']} | "
                      f"Score: {performer['performance_score']:.6f} | "
                      f"Win: {performer['win_rate']:.1f}% | PF: {performer['profit_factor']:.3f} | "
                      f"Confluence: {performer['confluence_accuracy']:.1f}% | "
                      f"SL/TP: {performer['stop_loss_pips']}/{performer['take_profit_pips']}")
        
        if args.parameter_analysis:
            analyzer.print_parameter_impact_report(args.epic)
        
        if args.compare_configs:
            scope = f" for {args.epic}" if args.epic else ""
            print(f"\n🔧 CONFIGURATION COMPARISON{scope.upper()}:")
            print("="*80)
            comparison = analyzer.compare_configurations(args.epic)
            for config in comparison.get('comparisons', []):
                print(f"{config['smc_config']:<12} | "
                      f"Tests: {config['total_tests']:>6,} | "
                      f"Max Score: {config['max_performance_score']:>8.6f} | "
                      f"Avg Win: {config['avg_win_rate']:>5.1f}% | "
                      f"Max Win: {config['max_win_rate']:>5.1f}% | "
                      f"Confluence: {config['avg_confluence_accuracy']:>5.1f}% | "
                      f"Consistency: {config.get('performance_consistency', 0):>6.4f}")
        
        if args.history:
            print(f"\n📊 OPTIMIZATION HISTORY (Last {args.days} days):")
            print("="*80)
            history = analyzer.get_optimization_history(args.days)
            for run in history:
                status_emoji = "✅" if run['status'] == 'completed' else "🔄" if run['status'] == 'running' else "❌"
                duration = f"{run.get('duration_minutes', 0):.1f}min" if run.get('duration_minutes') else "N/A"
                print(f"{status_emoji} Run {run['run_id']} | {run['epic']} | "
                      f"Mode: {run['optimization_mode']} | "
                      f"Progress: {run['completed_combinations']:,}/{run['total_combinations']:,} | "
                      f"Best: {run.get('best_score', 0):.6f} | "
                      f"Duration: {duration} | "
                      f"Started: {run['start_time']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)