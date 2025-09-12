#!/usr/bin/env python3
"""
MACD Optimization Analysis Tools
Comprehensive analysis and reporting for MACD parameter optimization results
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.database import DatabaseManager
try:
    import config
except ImportError:
    from forex_scanner import config


class MACDOptimizationAnalyzer:
    """
    Advanced analysis tools for MACD optimization results
    """
    
    def __init__(self):
        self.logger = logging.getLogger('macd_analyzer')
        self.setup_logging()
        self.db_manager = DatabaseManager(config.DATABASE_URL)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def get_optimization_summary(self) -> Dict:
        """Get overall optimization summary statistics"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get run statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_runs,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_runs,
                        COUNT(CASE WHEN status = 'running' THEN 1 END) as running_runs,
                        MAX(start_time) as last_run_time
                    FROM macd_optimization_runs
                """)
                run_stats = cursor.fetchone()
                
                # Get results statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_results,
                        COUNT(DISTINCT epic) as unique_epics,
                        AVG(win_rate) as avg_win_rate,
                        AVG(profit_factor) as avg_profit_factor,
                        AVG(composite_score) as avg_composite_score,
                        MAX(composite_score) as max_composite_score,
                        COUNT(CASE WHEN total_signals >= 3 THEN 1 END) as valid_results
                    FROM macd_optimization_results
                """)
                result_stats = cursor.fetchone()
                
                # Get best parameters count
                cursor.execute("SELECT COUNT(*) FROM macd_best_parameters")
                best_params_count = cursor.fetchone()[0]
                
                return {
                    'total_runs': run_stats[0],
                    'completed_runs': run_stats[1],
                    'running_runs': run_stats[2],
                    'last_run_time': run_stats[3],
                    'total_results': result_stats[0],
                    'unique_epics': result_stats[1],
                    'avg_win_rate': result_stats[2],
                    'avg_profit_factor': result_stats[3],
                    'avg_composite_score': result_stats[4],
                    'max_composite_score': result_stats[5],
                    'valid_results': result_stats[6],
                    'optimized_epics_count': best_params_count
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get optimization summary: {e}")
            return {}
    
    def get_epic_analysis(self, epic: str, top_n: int = 10) -> Dict:
        """Get detailed analysis for specific epic"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get best parameters for the epic
                cursor.execute("""
                    SELECT * FROM macd_best_parameters 
                    WHERE epic = %s
                """, (epic,))
                best_params = cursor.fetchone()
                
                if not best_params:
                    return {'error': f'No optimization data found for {epic}'}
                
                # Convert to dict
                columns = [desc[0] for desc in cursor.description]
                best_params_dict = dict(zip(columns, best_params))
                
                # Get top N results for the epic
                cursor.execute("""
                    SELECT 
                        fast_ema, slow_ema, signal_ema, confidence_threshold,
                        timeframe, macd_histogram_threshold, total_signals,
                        win_rate, profit_factor, net_pips, composite_score,
                        crossover_signals, momentum_confirmed_signals,
                        false_signal_rate, signal_delay_avg_bars
                    FROM macd_optimization_results 
                    WHERE epic = %s AND total_signals >= 3
                    ORDER BY composite_score DESC, win_rate DESC
                    LIMIT %s
                """, (epic, top_n))
                
                top_results = cursor.fetchall()
                
                # Get parameter distribution statistics
                cursor.execute("""
                    SELECT 
                        AVG(fast_ema) as avg_fast_ema,
                        AVG(slow_ema) as avg_slow_ema,
                        AVG(signal_ema) as avg_signal_ema,
                        AVG(confidence_threshold) as avg_confidence,
                        COUNT(DISTINCT timeframe) as timeframe_variants,
                        AVG(win_rate) as avg_win_rate,
                        AVG(profit_factor) as avg_profit_factor,
                        AVG(total_signals) as avg_signals,
                        MAX(composite_score) as best_score
                    FROM macd_optimization_results 
                    WHERE epic = %s AND total_signals >= 3
                """, (epic,))
                
                stats = cursor.fetchone()
                
                return {
                    'epic': epic,
                    'best_parameters': best_params_dict,
                    'top_results': top_results,
                    'statistics': {
                        'avg_fast_ema': stats[0],
                        'avg_slow_ema': stats[1],
                        'avg_signal_ema': stats[2],
                        'avg_confidence': stats[3],
                        'timeframe_variants': stats[4],
                        'avg_win_rate': stats[5],
                        'avg_profit_factor': stats[6],
                        'avg_signals': stats[7],
                        'best_score': stats[8]
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze epic {epic}: {e}")
            return {'error': str(e)}
    
    def get_parameter_effectiveness_analysis(self) -> Dict:
        """Analyze which parameter ranges perform best across all epics"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Analyze fast EMA effectiveness
                cursor.execute("""
                    SELECT 
                        fast_ema,
                        COUNT(*) as frequency,
                        AVG(win_rate) as avg_win_rate,
                        AVG(profit_factor) as avg_profit_factor,
                        AVG(composite_score) as avg_composite_score
                    FROM macd_optimization_results 
                    WHERE total_signals >= 3
                    GROUP BY fast_ema
                    ORDER BY avg_composite_score DESC
                """)
                fast_ema_analysis = cursor.fetchall()
                
                # Analyze slow EMA effectiveness
                cursor.execute("""
                    SELECT 
                        slow_ema,
                        COUNT(*) as frequency,
                        AVG(win_rate) as avg_win_rate,
                        AVG(profit_factor) as avg_profit_factor,
                        AVG(composite_score) as avg_composite_score
                    FROM macd_optimization_results 
                    WHERE total_signals >= 3
                    GROUP BY slow_ema
                    ORDER BY avg_composite_score DESC
                """)
                slow_ema_analysis = cursor.fetchall()
                
                # Analyze signal EMA effectiveness
                cursor.execute("""
                    SELECT 
                        signal_ema,
                        COUNT(*) as frequency,
                        AVG(win_rate) as avg_win_rate,
                        AVG(profit_factor) as avg_profit_factor,
                        AVG(composite_score) as avg_composite_score
                    FROM macd_optimization_results 
                    WHERE total_signals >= 3
                    GROUP BY signal_ema
                    ORDER BY avg_composite_score DESC
                """)
                signal_ema_analysis = cursor.fetchall()
                
                # Analyze timeframe effectiveness
                cursor.execute("""
                    SELECT 
                        timeframe,
                        COUNT(*) as frequency,
                        AVG(win_rate) as avg_win_rate,
                        AVG(profit_factor) as avg_profit_factor,
                        AVG(composite_score) as avg_composite_score
                    FROM macd_optimization_results 
                    WHERE total_signals >= 3
                    GROUP BY timeframe
                    ORDER BY avg_composite_score DESC
                """)
                timeframe_analysis = cursor.fetchall()
                
                return {
                    'fast_ema_effectiveness': fast_ema_analysis,
                    'slow_ema_effectiveness': slow_ema_analysis,
                    'signal_ema_effectiveness': signal_ema_analysis,
                    'timeframe_effectiveness': timeframe_analysis
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze parameter effectiveness: {e}")
            return {}
    
    def get_optimization_recommendations(self) -> List[Dict]:
        """Get recommendations for improving optimization results"""
        recommendations = []
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check for epics with low signal counts
                cursor.execute("""
                    SELECT epic, AVG(total_signals) as avg_signals
                    FROM macd_optimization_results 
                    GROUP BY epic
                    HAVING AVG(total_signals) < 5
                    ORDER BY avg_signals ASC
                """)
                
                low_signal_epics = cursor.fetchall()
                if low_signal_epics:
                    recommendations.append({
                        'type': 'low_signals',
                        'priority': 'high',
                        'description': 'Some epics have very few signals',
                        'epics': low_signal_epics,
                        'suggestion': 'Consider using longer backtest periods or more sensitive parameters'
                    })
                
                # Check for epics with poor performance
                cursor.execute("""
                    SELECT epic, best_composite_score, best_win_rate
                    FROM macd_best_parameters 
                    WHERE best_composite_score < 0.1 OR best_win_rate < 0.4
                    ORDER BY best_composite_score ASC
                """)
                
                poor_performers = cursor.fetchall()
                if poor_performers:
                    recommendations.append({
                        'type': 'poor_performance',
                        'priority': 'medium',
                        'description': 'Some epics have poor optimization results',
                        'epics': poor_performers,
                        'suggestion': 'Consider different parameter ranges or additional filters'
                    })
                
                # Check for unoptimized epics
                cursor.execute("""
                    SELECT COUNT(*) FROM macd_best_parameters
                """)
                optimized_count = cursor.fetchone()[0]
                
                if optimized_count < len(config.EPIC_LIST):
                    recommendations.append({
                        'type': 'incomplete_optimization',
                        'priority': 'medium',
                        'description': f'Only {optimized_count} out of {len(config.EPIC_LIST)} epics optimized',
                        'suggestion': 'Run optimization for remaining epics'
                    })
                
                return recommendations
                
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    def print_summary_report(self):
        """Print comprehensive summary report"""
        summary = self.get_optimization_summary()
        
        print("\n" + "="*80)
        print("🎯 MACD OPTIMIZATION ANALYSIS REPORT")
        print("="*80)
        
        if not summary:
            print("❌ No optimization data available")
            return
        
        print(f"📊 OPTIMIZATION RUNS:")
        print(f"   • Total Runs: {summary['total_runs']}")
        print(f"   • Completed: {summary['completed_runs']}")
        print(f"   • Running: {summary['running_runs']}")
        if summary['last_run_time']:
            print(f"   • Last Run: {summary['last_run_time']}")
        
        print(f"\n📈 RESULTS OVERVIEW:")
        print(f"   • Total Results: {summary['total_results']:,}")
        print(f"   • Valid Results (≥3 signals): {summary['valid_results']:,}")
        print(f"   • Unique Epics Tested: {summary['unique_epics']}")
        print(f"   • Optimized Epics: {summary['optimized_epics_count']}")
        
        if summary['avg_win_rate']:
            print(f"\n🎯 PERFORMANCE METRICS:")
            print(f"   • Average Win Rate: {summary['avg_win_rate']:.1%}")
            print(f"   • Average Profit Factor: {summary['avg_profit_factor']:.2f}")
            print(f"   • Average Composite Score: {summary['avg_composite_score']:.6f}")
            print(f"   • Maximum Composite Score: {summary['max_composite_score']:.6f}")
        
        # Show parameter effectiveness
        effectiveness = self.get_parameter_effectiveness_analysis()
        if effectiveness:
            print(f"\n🏆 TOP PERFORMING PARAMETERS:")
            
            if effectiveness['fast_ema_effectiveness']:
                best_fast = effectiveness['fast_ema_effectiveness'][0]
                print(f"   • Best Fast EMA: {int(best_fast[0])} (Score: {best_fast[4]:.6f})")
            
            if effectiveness['slow_ema_effectiveness']:
                best_slow = effectiveness['slow_ema_effectiveness'][0]
                print(f"   • Best Slow EMA: {int(best_slow[0])} (Score: {best_slow[4]:.6f})")
            
            if effectiveness['timeframe_effectiveness']:
                best_timeframe = effectiveness['timeframe_effectiveness'][0]
                print(f"   • Best Timeframe: {best_timeframe[0]} (Score: {best_timeframe[4]:.6f})")
        
        # Show recommendations
        recommendations = self.get_optimization_recommendations()
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡"
                print(f"   {priority_emoji} {rec['description']}")
                print(f"      → {rec['suggestion']}")
        
        print("\n" + "="*80)
    
    def print_epic_report(self, epic: str, top_n: int = 10):
        """Print detailed report for specific epic"""
        analysis = self.get_epic_analysis(epic, top_n)
        
        print(f"\n" + "="*80)
        print(f"🎯 MACD OPTIMIZATION REPORT FOR {epic}")
        print("="*80)
        
        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
            return
        
        best = analysis['best_parameters']
        print(f"🏆 OPTIMAL CONFIGURATION:")
        print(f"   • MACD Periods: {int(best['best_fast_ema'])}/{int(best['best_slow_ema'])}/{int(best['best_signal_ema'])}")
        print(f"   • Confidence Threshold: {best['best_confidence_threshold']:.3f}")
        print(f"   • Timeframe: {best['best_timeframe']}")
        print(f"   • Histogram Threshold: {best['best_histogram_threshold']:.8f}")
        print(f"   • Stop Loss: {best['optimal_stop_loss_pips']} pips")
        print(f"   • Take Profit: {best['optimal_take_profit_pips']} pips")
        
        print(f"\n🎯 FILTERS & ENHANCEMENTS:")
        print(f"   • RSI Filter: {'✅' if best['best_rsi_filter_enabled'] else '❌'}")
        print(f"   • Momentum Confirmation: {'✅' if best['best_momentum_confirmation'] else '❌'}")
        print(f"   • Zero Line Filter: {'✅' if best['best_zero_line_filter'] else '❌'}")
        print(f"   • Multi-Timeframe: {'✅' if best['best_mtf_enabled'] else '❌'}")
        print(f"   • Smart Money: {'✅' if best['best_smart_money_enabled'] else '❌'}")
        
        print(f"\n📊 BEST PERFORMANCE:")
        print(f"   • Win Rate: {best['best_win_rate']:.1%}")
        print(f"   • Profit Factor: {best['best_profit_factor']:.2f}")
        print(f"   • Net Pips: {best['best_net_pips']:.1f}")
        print(f"   • Composite Score: {best['best_composite_score']:.6f}")
        
        stats = analysis['statistics']
        print(f"\n📈 OPTIMIZATION STATISTICS:")
        print(f"   • Average Fast EMA: {stats['avg_fast_ema']:.1f}")
        print(f"   • Average Slow EMA: {stats['avg_slow_ema']:.1f}")
        print(f"   • Average Signal EMA: {stats['avg_signal_ema']:.1f}")
        print(f"   • Average Win Rate: {stats['avg_win_rate']:.1%}")
        print(f"   • Average Signals per Test: {stats['avg_signals']:.1f}")
        print(f"   • Timeframe Variants Tested: {int(stats['timeframe_variants'])}")
        
        print(f"\n🔝 TOP {min(top_n, len(analysis['top_results']))} CONFIGURATIONS:")
        print(f"{'Rank':<4} {'MACD':<12} {'Conf':<5} {'TF':<4} {'Signals':<7} {'Win%':<6} {'PF':<6} {'Pips':<8} {'Score':<10}")
        print("-" * 80)
        
        for i, result in enumerate(analysis['top_results'], 1):
            macd_periods = f"{int(result[0])}/{int(result[1])}/{int(result[2])}"
            print(f"{i:<4} {macd_periods:<12} {result[3]:<5.2f} {result[4]:<4} {int(result[6]):<7} "
                  f"{result[7]:<6.1%} {result[8]:<6.2f} {result[9]:<8.1f} {result[10]:<10.6f}")
        
        print("="*80)
    
    def export_results_to_csv(self, output_dir: str = "/tmp"):
        """Export optimization results to CSV files"""
        try:
            with self.db_manager.get_connection() as conn:
                # Export optimization results
                results_df = pd.read_sql("""
                    SELECT 
                        epic, fast_ema, slow_ema, signal_ema, confidence_threshold,
                        timeframe, total_signals, win_rate, profit_factor, net_pips,
                        composite_score, created_at
                    FROM macd_optimization_results 
                    WHERE total_signals >= 3
                    ORDER BY composite_score DESC
                """, conn)
                
                results_file = os.path.join(output_dir, f"macd_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                results_df.to_csv(results_file, index=False)
                
                # Export best parameters
                best_params_df = pd.read_sql("""
                    SELECT * FROM macd_best_parameters 
                    ORDER BY best_composite_score DESC
                """, conn)
                
                best_params_file = os.path.join(output_dir, f"macd_best_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                best_params_df.to_csv(best_params_file, index=False)
                
                self.logger.info(f"✅ Exported results to {results_file}")
                self.logger.info(f"✅ Exported best parameters to {best_params_file}")
                
                return [results_file, best_params_file]
                
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(description='MACD Optimization Analysis Tools')
    parser.add_argument('--summary', action='store_true', help='Show optimization summary')
    parser.add_argument('--epic', type=str, help='Analyze specific epic')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top results to show (default: 10)')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV files')
    parser.add_argument('--recommendations', action='store_true', help='Show optimization recommendations')
    
    args = parser.parse_args()
    
    analyzer = MACDOptimizationAnalyzer()
    
    if args.summary:
        analyzer.print_summary_report()
    
    if args.epic:
        analyzer.print_epic_report(args.epic, args.top_n)
    
    if args.export_csv:
        files = analyzer.export_results_to_csv()
        if files:
            print(f"✅ Results exported to {len(files)} CSV files")
    
    if args.recommendations:
        recommendations = analyzer.get_optimization_recommendations()
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for rec in recommendations:
            priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡"
            print(f"   {priority_emoji} {rec['description']}")
            print(f"      → {rec['suggestion']}")
    
    if not any([args.summary, args.epic, args.export_csv, args.recommendations]):
        print("Please specify at least one analysis option. Use --help for details.")


if __name__ == "__main__":
    main()