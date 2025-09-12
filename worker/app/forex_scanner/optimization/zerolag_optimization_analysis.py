#!/usr/bin/env python3
"""
Zero-Lag Optimization Analysis and Reporting Tools
Comprehensive analysis of zero-lag optimization results with detailed reporting
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.database import DatabaseManager

try:
    import config
except ImportError:
    from forex_scanner import config


class ZeroLagOptimizationAnalyzer:
    """
    Comprehensive analysis tool for zero-lag optimization results
    """
    
    def __init__(self):
        self.logger = logging.getLogger('zerolag_optimization_analyzer')
        self.setup_logging()
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def get_optimization_summary(self, days: int = 30) -> Dict:
        """Get comprehensive optimization summary"""
        try:
            with self.db_manager.get_connection() as conn:
                # Get basic statistics
                summary_query = """
                    SELECT 
                        COUNT(DISTINCT epic) as total_epics,
                        COUNT(*) as total_runs,
                        AVG(best_composite_score) as avg_score,
                        MAX(best_composite_score) as max_score,
                        MIN(last_updated) as first_optimization,
                        MAX(last_updated) as last_optimization,
                        COUNT(CASE WHEN last_updated >= NOW() - INTERVAL '%s days' THEN 1 END) as recent_optimizations
                    FROM zerolag_best_parameters
                """
                
                summary_df = pd.read_sql_query(summary_query, conn, params=[days])
                summary_stats = summary_df.iloc[0].to_dict()
                
                # Get parameter distribution
                param_query = """
                    SELECT 
                        best_zl_length,
                        best_band_multiplier,
                        best_confidence_threshold,
                        best_timeframe,
                        best_smart_money_enabled,
                        best_mtf_validation_enabled,
                        COUNT(*) as frequency
                    FROM zerolag_best_parameters
                    GROUP BY best_zl_length, best_band_multiplier, best_confidence_threshold, 
                             best_timeframe, best_smart_money_enabled, best_mtf_validation_enabled
                    ORDER BY frequency DESC
                """
                
                param_df = pd.read_sql_query(param_query, conn)
                
                # Get top performers
                top_performers_query = """
                    SELECT epic, best_composite_score, best_win_rate, best_profit_factor,
                           best_net_pips, best_zl_length, best_band_multiplier,
                           best_confidence_threshold, best_timeframe
                    FROM zerolag_best_parameters
                    ORDER BY best_composite_score DESC
                    LIMIT 10
                """
                
                top_performers_df = pd.read_sql_query(top_performers_query, conn)
                
                return {
                    'summary_stats': summary_stats,
                    'parameter_distribution': param_df,
                    'top_performers': top_performers_df,
                    'analysis_timestamp': datetime.now()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get optimization summary: {e}")
            return {}
    
    def analyze_epic_performance(self, epic: str, days: int = 30) -> Dict:
        """Detailed analysis of a specific epic's optimization results"""
        try:
            with self.db_manager.get_connection() as conn:
                # Get current best parameters
                best_params_query = """
                    SELECT * FROM zerolag_best_parameters 
                    WHERE epic = %s
                """
                
                best_params_df = pd.read_sql_query(best_params_query, conn, params=[epic])
                
                if best_params_df.empty:
                    return {'error': f'No optimization data found for {epic}'}
                
                # Get historical optimization results
                history_query = """
                    SELECT 
                        zl_length, band_multiplier, confidence_threshold, timeframe,
                        bb_length, bb_mult, kc_length, kc_mult,
                        smart_money_enabled, mtf_validation_enabled,
                        stop_loss_pips, take_profit_pips, risk_reward_ratio,
                        total_signals, win_rate, profit_factor, net_pips, 
                        composite_score, created_at
                    FROM zerolag_optimization_results
                    WHERE epic = %s 
                        AND created_at >= NOW() - INTERVAL '%s days'
                        AND total_signals >= 3
                    ORDER BY composite_score DESC
                    LIMIT 100
                """
                
                history_df = pd.read_sql_query(history_query, conn, params=[epic, days])
                
                # Calculate parameter correlations with performance
                correlations = {}
                if not history_df.empty:
                    numeric_cols = ['zl_length', 'band_multiplier', 'confidence_threshold',
                                  'bb_length', 'bb_mult', 'kc_length', 'kc_mult',
                                  'stop_loss_pips', 'take_profit_pips']
                    
                    for col in numeric_cols:
                        if col in history_df.columns:
                            correlation = history_df[col].corr(history_df['composite_score'])
                            correlations[col] = correlation if not pd.isna(correlation) else 0
                
                # Performance analysis over time
                if not history_df.empty:
                    history_df['created_at'] = pd.to_datetime(history_df['created_at'])
                    time_analysis = history_df.groupby(history_df['created_at'].dt.date).agg({
                        'composite_score': ['mean', 'max', 'count'],
                        'win_rate': 'mean',
                        'total_signals': 'sum'
                    }).round(4)
                else:
                    time_analysis = pd.DataFrame()
                
                return {
                    'epic': epic,
                    'best_parameters': best_params_df.iloc[0].to_dict(),
                    'optimization_history': history_df,
                    'parameter_correlations': correlations,
                    'time_analysis': time_analysis,
                    'total_tests': len(history_df)
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze epic {epic}: {e}")
            return {'error': str(e)}
    
    def compare_strategies(self, epics: List[str]) -> Dict:
        """Compare optimization results across multiple epics"""
        try:
            with self.db_manager.get_connection() as conn:
                epic_list = "', '".join(epics)
                comparison_query = f"""
                    SELECT 
                        epic,
                        best_zl_length,
                        best_band_multiplier, 
                        best_confidence_threshold,
                        best_timeframe,
                        best_bb_length,
                        best_bb_mult,
                        best_kc_length,
                        best_kc_mult,
                        best_smart_money_enabled,
                        best_mtf_validation_enabled,
                        optimal_stop_loss_pips,
                        optimal_take_profit_pips,
                        best_win_rate,
                        best_profit_factor,
                        best_net_pips,
                        best_composite_score,
                        last_updated
                    FROM zerolag_best_parameters 
                    WHERE epic IN ('{epic_list}')
                    ORDER BY best_composite_score DESC
                """
                
                comparison_df = pd.read_sql_query(comparison_query, conn)
                
                if comparison_df.empty:
                    return {'error': 'No optimization data found for specified epics'}
                
                # Calculate statistics across epics
                numeric_cols = ['best_zl_length', 'best_band_multiplier', 'best_confidence_threshold',
                               'best_bb_length', 'best_bb_mult', 'best_kc_length', 'best_kc_mult',
                               'optimal_stop_loss_pips', 'optimal_take_profit_pips',
                               'best_win_rate', 'best_profit_factor', 'best_net_pips', 'best_composite_score']
                
                stats_summary = comparison_df[numeric_cols].describe().round(4)
                
                # Find best performing configurations
                best_overall = comparison_df.loc[comparison_df['best_composite_score'].idxmax()]
                highest_winrate = comparison_df.loc[comparison_df['best_win_rate'].idxmax()]
                highest_pf = comparison_df.loc[comparison_df['best_profit_factor'].idxmax()]
                
                return {
                    'comparison_data': comparison_df,
                    'statistical_summary': stats_summary,
                    'best_performers': {
                        'overall_best': best_overall.to_dict(),
                        'highest_win_rate': highest_winrate.to_dict(),
                        'highest_profit_factor': highest_pf.to_dict()
                    },
                    'epic_count': len(comparison_df)
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to compare strategies: {e}")
            return {'error': str(e)}
    
    def get_parameter_recommendations(self, epic: str = None) -> Dict:
        """Get parameter recommendations based on optimization results"""
        try:
            with self.db_manager.get_connection() as conn:
                if epic:
                    # Epic-specific recommendations
                    query = """
                        SELECT 
                            zl_length, band_multiplier, confidence_threshold,
                            bb_length, bb_mult, kc_length, kc_mult,
                            smart_money_enabled, mtf_validation_enabled,
                            AVG(composite_score) as avg_score,
                            COUNT(*) as frequency,
                            AVG(win_rate) as avg_win_rate,
                            AVG(profit_factor) as avg_profit_factor
                        FROM zerolag_optimization_results
                        WHERE epic = %s AND total_signals >= 5
                        GROUP BY zl_length, band_multiplier, confidence_threshold,
                                 bb_length, bb_mult, kc_length, kc_mult,
                                 smart_money_enabled, mtf_validation_enabled
                        HAVING COUNT(*) >= 2
                        ORDER BY avg_score DESC
                        LIMIT 10
                    """
                    params = [epic]
                else:
                    # Global recommendations
                    query = """
                        SELECT 
                            zl_length, band_multiplier, confidence_threshold,
                            bb_length, bb_mult, kc_length, kc_mult,
                            smart_money_enabled, mtf_validation_enabled,
                            AVG(composite_score) as avg_score,
                            COUNT(*) as frequency,
                            COUNT(DISTINCT epic) as epic_count,
                            AVG(win_rate) as avg_win_rate,
                            AVG(profit_factor) as avg_profit_factor
                        FROM zerolag_optimization_results
                        WHERE total_signals >= 5
                        GROUP BY zl_length, band_multiplier, confidence_threshold,
                                 bb_length, bb_mult, kc_length, kc_mult,
                                 smart_money_enabled, mtf_validation_enabled
                        HAVING COUNT(*) >= 5 AND COUNT(DISTINCT epic) >= 2
                        ORDER BY avg_score DESC
                        LIMIT 15
                    """
                    params = []
                
                recommendations_df = pd.read_sql_query(query, conn, params=params)
                
                if recommendations_df.empty:
                    return {'error': 'No sufficient data for recommendations'}
                
                # Analyze most successful parameter ranges
                parameter_analysis = {}
                numeric_params = ['zl_length', 'band_multiplier', 'confidence_threshold',
                                'bb_length', 'bb_mult', 'kc_length', 'kc_mult']
                
                for param in numeric_params:
                    if param in recommendations_df.columns:
                        # Weighted analysis based on frequency and performance
                        weighted_stats = {
                            'best_value': recommendations_df.loc[0, param],  # From best performing config
                            'avg_value': (recommendations_df[param] * recommendations_df['frequency']).sum() / recommendations_df['frequency'].sum(),
                            'range_low': recommendations_df[param].quantile(0.25),
                            'range_high': recommendations_df[param].quantile(0.75),
                            'most_frequent': recommendations_df[param].mode().iloc[0] if not recommendations_df[param].mode().empty else recommendations_df[param].median()
                        }
                        parameter_analysis[param] = weighted_stats
                
                return {
                    'epic': epic or 'global',
                    'top_configurations': recommendations_df,
                    'parameter_analysis': parameter_analysis,
                    'recommendations_count': len(recommendations_df)
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get recommendations: {e}")
            return {'error': str(e)}
    
    def generate_optimization_report(self, epic: str = None, days: int = 30) -> str:
        """Generate comprehensive optimization report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("🔍 ZERO-LAG OPTIMIZATION ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"📊 Analysis Period: Last {days} days")
            report.append("")
            
            if epic:
                # Epic-specific report
                report.append(f"🎯 EPIC: {epic}")
                report.append("-" * 40)
                
                analysis = self.analyze_epic_performance(epic, days)
                if 'error' in analysis:
                    report.append(f"❌ Error: {analysis['error']}")
                    return "\n".join(report)
                
                best_params = analysis['best_parameters']
                
                report.append("🏆 OPTIMAL PARAMETERS:")
                report.append(f"   ⚡ ZL Length: {best_params['best_zl_length']}")
                report.append(f"   📈 Band Multiplier: {best_params['best_band_multiplier']:.2f}")
                report.append(f"   🎯 Confidence: {best_params['best_confidence_threshold']:.1%}")
                report.append(f"   ⏰ Timeframe: {best_params['best_timeframe']}")
                report.append(f"   🔍 Squeeze BB: {best_params['best_bb_length']}/{best_params['best_bb_mult']:.1f}")
                report.append(f"   🔍 Squeeze KC: {best_params['best_kc_length']}/{best_params['best_kc_mult']:.1f}")
                report.append(f"   🧠 Smart Money: {'✓' if best_params['best_smart_money_enabled'] else '✗'}")
                report.append(f"   📊 MTF Validation: {'✓' if best_params['best_mtf_validation_enabled'] else '✗'}")
                report.append("")
                
                report.append("📈 PERFORMANCE METRICS:")
                report.append(f"   🎯 Composite Score: {best_params['best_composite_score']:.4f}")
                report.append(f"   📊 Win Rate: {best_params['best_win_rate']:.1%}")
                report.append(f"   💰 Profit Factor: {best_params['best_profit_factor']:.2f}")
                report.append(f"   📏 Net Pips: {best_params['best_net_pips']:.1f}")
                report.append(f"   🛑 Stop Loss: {best_params['optimal_stop_loss_pips']:.0f} pips")
                report.append(f"   🎯 Take Profit: {best_params['optimal_take_profit_pips']:.0f} pips")
                report.append("")
                
                # Parameter correlations
                if analysis['parameter_correlations']:
                    report.append("🔗 PARAMETER CORRELATIONS (with performance):")
                    correlations = analysis['parameter_correlations']
                    for param, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                        correlation_strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                        direction = "Positive" if corr > 0 else "Negative"
                        report.append(f"   {param:20} {corr:+.3f} ({correlation_strength} {direction})")
                    report.append("")
                
                report.append(f"📋 Total optimization tests: {analysis['total_tests']}")
                
            else:
                # Global summary report
                summary = self.get_optimization_summary(days)
                if not summary:
                    report.append("❌ Failed to generate summary")
                    return "\n".join(report)
                
                stats = summary['summary_stats']
                
                report.append("📊 GLOBAL OPTIMIZATION SUMMARY:")
                report.append(f"   📈 Total Epics Optimized: {int(stats['total_epics'])}")
                report.append(f"   🔄 Total Optimization Runs: {int(stats['total_runs'])}")
                report.append(f"   📊 Average Score: {stats['avg_score']:.4f}")
                report.append(f"   🏆 Best Score: {stats['max_score']:.4f}")
                report.append(f"   🕐 First Optimization: {stats['first_optimization']}")
                report.append(f"   🕐 Last Optimization: {stats['last_optimization']}")
                report.append(f"   📅 Recent Optimizations: {int(stats['recent_optimizations'])}")
                report.append("")
                
                # Top performers
                top_performers = summary['top_performers']
                if not top_performers.empty:
                    report.append("🏆 TOP PERFORMING EPICS:")
                    for idx, row in top_performers.head(5).iterrows():
                        report.append(f"   {idx+1}. {row['epic']} - Score: {row['best_composite_score']:.4f} "
                                    f"(WR: {row['best_win_rate']:.1%}, PF: {row['best_profit_factor']:.2f})")
                    report.append("")
                
                # Most common parameters
                param_dist = summary['parameter_distribution']
                if not param_dist.empty:
                    report.append("🔧 MOST SUCCESSFUL PARAMETER COMBINATIONS:")
                    for idx, row in param_dist.head(3).iterrows():
                        report.append(f"   {idx+1}. ZL:{row['best_zl_length']}, Band:{row['best_band_multiplier']:.1f}, "
                                    f"Conf:{row['best_confidence_threshold']:.1%} ({row['frequency']} epics)")
                    report.append("")
            
            report.append("=" * 80)
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            return f"❌ Report generation failed: {e}"
    
    def export_results(self, output_file: str, epic: str = None, days: int = 30):
        """Export optimization results to CSV file"""
        try:
            if epic:
                analysis = self.analyze_epic_performance(epic, days)
                if 'error' in analysis:
                    self.logger.error(f"❌ Cannot export - {analysis['error']}")
                    return False
                
                # Export epic-specific results
                analysis['optimization_history'].to_csv(f"{output_file}_{epic}_history.csv", index=False)
                
                # Export best parameters
                best_params_df = pd.DataFrame([analysis['best_parameters']])
                best_params_df.to_csv(f"{output_file}_{epic}_best.csv", index=False)
                
                self.logger.info(f"✅ Exported {epic} results to {output_file}_*")
                
            else:
                # Export global summary
                summary = self.get_optimization_summary(days)
                if not summary:
                    self.logger.error("❌ Cannot export - failed to get summary")
                    return False
                
                summary['top_performers'].to_csv(f"{output_file}_top_performers.csv", index=False)
                summary['parameter_distribution'].to_csv(f"{output_file}_param_distribution.csv", index=False)
                
                self.logger.info(f"✅ Exported global results to {output_file}_*")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")
            return False


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Zero-Lag Optimization Analysis Tool')
    parser.add_argument('--epic', help='Analyze specific epic (e.g., CS.D.EURUSD.CEEM.IP)')
    parser.add_argument('--summary', action='store_true', help='Show global optimization summary')
    parser.add_argument('--compare', nargs='+', help='Compare multiple epics')
    parser.add_argument('--recommendations', action='store_true', help='Get parameter recommendations')
    parser.add_argument('--days', type=int, default=30, help='Analysis period in days (default: 30)')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--export', help='Export results to CSV (base filename)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top results to show')
    
    args = parser.parse_args()
    
    if not any([args.epic, args.summary, args.compare, args.recommendations, args.report]):
        print("❌ Must specify at least one analysis option")
        parser.print_help()
        return
    
    analyzer = ZeroLagOptimizationAnalyzer()
    
    try:
        if args.summary or args.report:
            if args.epic:
                print(analyzer.generate_optimization_report(args.epic, args.days))
            else:
                print(analyzer.generate_optimization_report(days=args.days))
        
        if args.epic and not args.report:
            print(f"\n📊 ANALYZING EPIC: {args.epic}")
            analysis = analyzer.analyze_epic_performance(args.epic, args.days)
            if 'error' in analysis:
                print(f"❌ Error: {analysis['error']}")
                return
            
            print(f"✅ Analysis complete - {analysis['total_tests']} optimization tests found")
        
        if args.compare:
            print(f"\n🔄 COMPARING EPICS: {', '.join(args.compare)}")
            comparison = analyzer.compare_strategies(args.compare)
            if 'error' in comparison:
                print(f"❌ Error: {comparison['error']}")
                return
            
            print(f"✅ Comparison complete - {comparison['epic_count']} epics analyzed")
        
        if args.recommendations:
            print(f"\n💡 PARAMETER RECOMMENDATIONS{'for ' + args.epic if args.epic else ' (Global)'}:")
            recommendations = analyzer.get_parameter_recommendations(args.epic)
            if 'error' in recommendations:
                print(f"❌ Error: {recommendations['error']}")
                return
            
            print(f"✅ Generated {recommendations['recommendations_count']} recommendations")
        
        if args.export:
            print(f"\n📤 EXPORTING RESULTS...")
            success = analyzer.export_results(args.export, args.epic, args.days)
            if success:
                print("✅ Export completed successfully")
            else:
                print("❌ Export failed")
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis cancelled by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()