#!/usr/bin/env python3
"""
Optimization Results Analysis
Tools for analyzing and reporting on EMA parameter optimization results
"""

import sys
import os
import logging
from typing import Dict, List, Optional
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


class OptimizationAnalyzer:
    """
    Analyze and report optimization results
    """
    
    def __init__(self):
        self.logger = logging.getLogger('optimization_analyzer')
        self.setup_logging()
        self.db_manager = DatabaseManager(config.DATABASE_URL)
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def get_best_parameters_summary(self) -> pd.DataFrame:
        """Get summary of best parameters for all epics"""
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                    SELECT 
                        epic,
                        best_ema_config,
                        best_confidence_threshold,
                        best_timeframe,
                        optimal_stop_loss_pips,
                        optimal_take_profit_pips,
                        ROUND(optimal_take_profit_pips / optimal_stop_loss_pips, 2) as risk_reward,
                        ROUND(best_win_rate * 100, 1) as win_rate_pct,
                        ROUND(best_profit_factor, 2) as profit_factor,
                        ROUND(best_net_pips, 1) as net_pips,
                        last_updated
                    FROM ema_best_parameters
                    ORDER BY best_net_pips DESC
                """
                
                df = pd.read_sql_query(query, conn)
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get best parameters: {e}")
            return pd.DataFrame()
    
    def get_optimization_run_summary(self, run_id: int = None) -> pd.DataFrame:
        """Get summary of optimization runs"""
        try:
            with self.db_manager.get_connection() as conn:
                if run_id:
                    query = """
                        SELECT * FROM ema_optimization_runs 
                        WHERE id = %s
                        ORDER BY start_time DESC
                    """
                    df = pd.read_sql_query(query, conn, params=[run_id])
                else:
                    query = """
                        SELECT * FROM ema_optimization_runs 
                        ORDER BY start_time DESC
                        LIMIT 10
                    """
                    df = pd.read_sql_query(query, conn)
                
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get run summary: {e}")
            return pd.DataFrame()
    
    def get_epic_optimization_results(self, epic: str, top_n: int = 10) -> pd.DataFrame:
        """Get top optimization results for specific epic"""
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                    SELECT 
                        epic,
                        ema_config,
                        ROUND(confidence_threshold * 100, 1) as confidence_pct,
                        timeframe,
                        smart_money_enabled,
                        stop_loss_pips,
                        take_profit_pips,
                        ROUND(risk_reward_ratio, 2) as risk_reward,
                        total_signals,
                        ROUND(win_rate * 100, 1) as win_rate_pct,
                        ROUND(profit_factor, 2) as profit_factor,
                        ROUND(net_pips, 1) as net_pips,
                        ROUND(composite_score, 4) as composite_score,
                        created_at
                    FROM ema_optimization_results
                    WHERE epic = %s AND total_signals >= 10
                    ORDER BY composite_score DESC NULLS LAST
                    LIMIT %s
                """
                
                df = pd.read_sql_query(query, conn, params=[epic, top_n])
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get epic results: {e}")
            return pd.DataFrame()
    
    def get_parameter_sensitivity_analysis(self, epic: str) -> Dict[str, pd.DataFrame]:
        """Analyze parameter sensitivity for an epic"""
        try:
            with self.db_manager.get_connection() as conn:
                results = {}
                
                # EMA Config sensitivity
                query = """
                    SELECT 
                        ema_config,
                        COUNT(*) as test_count,
                        ROUND(AVG(win_rate) * 100, 1) as avg_win_rate_pct,
                        ROUND(AVG(profit_factor), 2) as avg_profit_factor,
                        ROUND(AVG(net_pips), 1) as avg_net_pips,
                        ROUND(AVG(composite_score), 4) as avg_composite_score,
                        ROUND(MAX(composite_score), 4) as best_composite_score
                    FROM ema_optimization_results
                    WHERE epic = %s AND total_signals >= 10
                    GROUP BY ema_config
                    ORDER BY avg_composite_score DESC NULLS LAST
                """
                results['ema_config'] = pd.read_sql_query(query, conn, params=[epic])
                
                # Confidence threshold sensitivity  
                query = """
                    SELECT 
                        confidence_threshold,
                        COUNT(*) as test_count,
                        ROUND(AVG(win_rate) * 100, 1) as avg_win_rate_pct,
                        ROUND(AVG(profit_factor), 2) as avg_profit_factor,
                        ROUND(AVG(net_pips), 1) as avg_net_pips,
                        ROUND(AVG(composite_score), 4) as avg_composite_score
                    FROM ema_optimization_results
                    WHERE epic = %s AND total_signals >= 10
                    GROUP BY confidence_threshold
                    ORDER BY avg_composite_score DESC NULLS LAST
                """
                results['confidence_threshold'] = pd.read_sql_query(query, conn, params=[epic])
                
                # Timeframe sensitivity
                query = """
                    SELECT 
                        timeframe,
                        COUNT(*) as test_count,
                        ROUND(AVG(win_rate) * 100, 1) as avg_win_rate_pct,
                        ROUND(AVG(profit_factor), 2) as avg_profit_factor,
                        ROUND(AVG(net_pips), 1) as avg_net_pips,
                        ROUND(AVG(composite_score), 4) as avg_composite_score
                    FROM ema_optimization_results
                    WHERE epic = %s AND total_signals >= 10
                    GROUP BY timeframe
                    ORDER BY avg_composite_score DESC NULLS LAST
                """
                results['timeframe'] = pd.read_sql_query(query, conn, params=[epic])
                
                # Risk management sensitivity
                query = """
                    SELECT 
                        stop_loss_pips,
                        take_profit_pips,
                        ROUND(risk_reward_ratio, 2) as risk_reward,
                        COUNT(*) as test_count,
                        ROUND(AVG(win_rate) * 100, 1) as avg_win_rate_pct,
                        ROUND(AVG(profit_factor), 2) as avg_profit_factor,
                        ROUND(AVG(net_pips), 1) as avg_net_pips,
                        ROUND(AVG(composite_score), 4) as avg_composite_score
                    FROM ema_optimization_results
                    WHERE epic = %s AND total_signals >= 10
                    GROUP BY stop_loss_pips, take_profit_pips, risk_reward_ratio
                    ORDER BY avg_composite_score DESC NULLS LAST
                    LIMIT 20
                """
                results['risk_management'] = pd.read_sql_query(query, conn, params=[epic])
                
                return results
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get sensitivity analysis: {e}")
            return {}
    
    def print_best_parameters_report(self):
        """Print formatted report of best parameters"""
        print("\n🏆 BEST EMA PARAMETERS BY EPIC")
        print("=" * 120)
        
        df = self.get_best_parameters_summary()
        
        if df.empty:
            print("❌ No optimization results found")
            return
        
        # Print header
        print(f"{'EPIC':<25} {'EMA CONFIG':<12} {'CONF':<5} {'TF':<4} {'SL/TP':<8} {'R:R':<5} {'WIN%':<6} {'PF':<6} {'NET PIPS':<10}")
        print("-" * 120)
        
        # Print each epic's best parameters
        for _, row in df.iterrows():
            epic_short = row['epic'].replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
            sl_tp = f"{row['optimal_stop_loss_pips']:.0f}/{row['optimal_take_profit_pips']:.0f}"
            
            print(f"{epic_short:<25} {row['best_ema_config']:<12} {row['best_confidence_threshold']:<5.0%} "
                  f"{row['best_timeframe']:<4} {sl_tp:<8} {row['risk_reward']:<5} "
                  f"{row['win_rate_pct']:<6}% {row['profit_factor']:<6} {row['net_pips']:<10}")
        
        print("-" * 120)
        print(f"📊 Total optimized epics: {len(df)}")
        print(f"💰 Average net pips: {df['net_pips'].mean():.1f}")
        print(f"🏆 Average win rate: {df['win_rate_pct'].mean():.1f}%")
        print(f"📈 Average profit factor: {df['profit_factor'].mean():.2f}")
    
    def print_epic_analysis(self, epic: str, top_n: int = 10):
        """Print detailed analysis for specific epic"""
        print(f"\n🎯 DETAILED ANALYSIS: {epic}")
        print("=" * 80)
        
        # Get top results
        results_df = self.get_epic_optimization_results(epic, top_n)
        
        if results_df.empty:
            print("❌ No results found for this epic")
            return
        
        print(f"\n🏆 TOP {top_n} CONFIGURATIONS:")
        print("-" * 80)
        print(f"{'RANK':<5} {'EMA':<12} {'CONF':<5} {'TF':<4} {'SL/TP':<8} {'SIGS':<5} {'WIN%':<6} {'PF':<6} {'NET':<8} {'SCORE':<8}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            sl_tp = f"{row['stop_loss_pips']:.0f}/{row['take_profit_pips']:.0f}"
            
            print(f"{i:<5} {row['ema_config']:<12} {row['confidence_pct']:<5.0f}% "
                  f"{row['timeframe']:<4} {sl_tp:<8} {row['total_signals']:<5} "
                  f"{row['win_rate_pct']:<6}% {row['profit_factor']:<6} "
                  f"{row['net_pips']:<8} {row['composite_score']:<8}")
        
        # Parameter sensitivity analysis
        sensitivity = self.get_parameter_sensitivity_analysis(epic)
        
        if sensitivity:
            print(f"\n📊 PARAMETER SENSITIVITY ANALYSIS:")
            
            # EMA Config performance
            if not sensitivity['ema_config'].empty:
                print(f"\n📈 EMA Configuration Performance:")
                print(sensitivity['ema_config'].to_string(index=False))
            
            # Timeframe performance
            if not sensitivity['timeframe'].empty:
                print(f"\n⏱️ Timeframe Performance:")
                print(sensitivity['timeframe'].to_string(index=False))
            
            # Risk management performance
            if not sensitivity['risk_management'].empty:
                print(f"\n🎯 Risk Management Performance (Top 10):")
                print(sensitivity['risk_management'].to_string(index=False))
    
    def generate_optimization_insights(self, epic: str = None) -> List[str]:
        """Generate optimization insights and recommendations"""
        insights = []
        
        if epic:
            # Epic-specific insights
            df = self.get_epic_optimization_results(epic, 100)
            
            if not df.empty:
                # Best EMA config
                best_ema = df.groupby('ema_config')['composite_score'].mean().idxmax()
                insights.append(f"Best EMA configuration for {epic}: {best_ema}")
                
                # Best timeframe
                best_tf = df.groupby('timeframe')['composite_score'].mean().idxmax()
                insights.append(f"Best timeframe for {epic}: {best_tf}")
                
                # Risk preferences
                avg_sl = df['stop_loss_pips'].mean()
                avg_tp = df['take_profit_pips'].mean()
                insights.append(f"Optimal risk levels for {epic}: ~{avg_sl:.0f} SL / ~{avg_tp:.0f} TP")
                
                # Smart Money impact
                sm_performance = df.groupby('smart_money_enabled')['composite_score'].mean()
                if len(sm_performance) > 1:
                    best_sm = sm_performance.idxmax()
                    insights.append(f"Smart Money analysis for {epic}: {'Beneficial' if best_sm else 'Not beneficial'}")
        
        else:
            # Global insights
            df = self.get_best_parameters_summary()
            
            if not df.empty:
                # Most common best EMA config
                common_ema = df['best_ema_config'].mode().iloc[0] if not df['best_ema_config'].mode().empty else 'N/A'
                insights.append(f"Most successful EMA configuration overall: {common_ema}")
                
                # Most common timeframe
                common_tf = df['best_timeframe'].mode().iloc[0] if not df['best_timeframe'].mode().empty else 'N/A'
                insights.append(f"Most successful timeframe overall: {common_tf}")
                
                # Average risk preferences
                avg_sl = df['optimal_stop_loss_pips'].mean()
                avg_tp = df['optimal_take_profit_pips'].mean()
                insights.append(f"Average optimal risk levels: {avg_sl:.1f} SL / {avg_tp:.1f} TP")
        
        return insights


def main():
    """Main execution for optimization analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimization Results Analysis')
    parser.add_argument('--epic', help='Analyze specific epic')
    parser.add_argument('--summary', action='store_true', help='Show best parameters summary')
    parser.add_argument('--runs', action='store_true', help='Show optimization runs')
    parser.add_argument('--insights', action='store_true', help='Generate insights')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top results to show')
    
    args = parser.parse_args()
    
    analyzer = OptimizationAnalyzer()
    
    try:
        if args.summary or (not args.epic and not args.runs and not args.insights):
            # Default: show best parameters summary
            analyzer.print_best_parameters_report()
        
        if args.epic:
            # Analyze specific epic
            analyzer.print_epic_analysis(args.epic, args.top_n)
        
        if args.runs:
            # Show optimization runs
            runs_df = analyzer.get_optimization_run_summary()
            print("\n📊 OPTIMIZATION RUNS:")
            print(runs_df.to_string(index=False))
        
        if args.insights:
            # Generate insights
            print("\n💡 OPTIMIZATION INSIGHTS:")
            insights = analyzer.generate_optimization_insights(args.epic)
            for i, insight in enumerate(insights, 1):
                print(f"   {i}. {insight}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()