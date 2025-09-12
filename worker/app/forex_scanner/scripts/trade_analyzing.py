#!/usr/bin/env python3
"""
Trading Performance Analysis Script
🚨 EMERGENCY: Analyze poor trading performance from alert_history table

This script will help identify why the trading system is performing poorly
by analyzing recent signals, their characteristics, and patterns.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
try:
    from alerts.alert_history import AlertHistoryManager
    from core.database import DatabaseManager
    from config import DATABASE_URL
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the forex_scanner directory")
    sys.exit(1)


class TradingPerformanceAnalyzer:
    """
    🚨 EMERGENCY: Comprehensive trading performance analyzer
    """
    
    def __init__(self):
        print("🔍 Initializing Trading Performance Analyzer...")
        
        # Try to initialize database components with fallbacks
        self.db_manager = None
        self.alert_manager = None
        
        # Method 1: Try full initialization
        try:
            from core.database import DatabaseManager
            from config import DATABASE_URL
            
            self.db_manager = DatabaseManager(DATABASE_URL)
            print("✅ Database manager initialized")
            
            # Try to initialize alert manager
            try:
                self.alert_manager = AlertHistoryManager(self.db_manager)
                print("✅ Alert history manager initialized")
            except Exception as e:
                print(f"⚠️ Alert history manager failed: {e}")
                print("🔄 Will use direct database queries instead")
                
        except Exception as e:
            print(f"⚠️ Database manager failed: {e}")
            print("🔄 Will use direct database connection for analysis")
        
        # Test database connection
        if not self._test_database_connection():
            print("❌ No database connection available - analysis cannot proceed")
            raise ConnectionError("Database connection required for analysis")
        
        self.analysis_results = {}
    
    def _test_database_connection(self) -> bool:
        """Test if we can connect to the database"""
        try:
            import psycopg2
            from config import DATABASE_URL
            
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alert_history LIMIT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            
            print("✅ Database connection test successful")
            return True
            
        except Exception as e:
            print(f"❌ Database connection test failed: {e}")
            return False
        
    def analyze_recent_performance(self, hours: int = 24) -> Dict:
        """
        🚨 MAIN ANALYSIS: Analyze recent trading performance
        """
        print(f"\n📊 ANALYZING LAST {hours} HOURS OF TRADING...")
        print("=" * 60)
        
        # Get recent signals
        recent_signals = self.get_recent_signals(hours)
        
        if recent_signals.empty:
            print(f"❌ No signals found in last {hours} hours")
            return {"error": "No recent signals"}
        
        print(f"📈 Found {len(recent_signals)} signals in last {hours} hours")
        
        # Perform comprehensive analysis
        results = {
            "signal_overview": self.analyze_signal_overview(recent_signals),
            "confidence_analysis": self.analyze_confidence_distribution(recent_signals),
            "strategy_performance": self.analyze_strategy_performance(recent_signals),
            "pair_analysis": self.analyze_pair_performance(recent_signals),
            "timing_analysis": self.analyze_timing_patterns(recent_signals),
            "technical_analysis": self.analyze_technical_indicators(recent_signals),
            "quality_issues": self.identify_quality_issues(recent_signals),
            "recommendations": self.generate_recommendations(recent_signals)
        }
        
        self.analysis_results = results
        return results
    
    def get_recent_signals(self, hours: int) -> pd.DataFrame:
        """Get recent signals from alert_history table"""
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Method 1: Try using AlertHistoryManager
            if hasattr(self, 'alert_manager') and self.alert_manager:
                try:
                    df = self.alert_manager.get_recent_alerts(
                        days=max(1, hours // 24 + 1),  # Convert hours to days, minimum 1
                        strategy=None,  # Get all strategies
                        include_claude=True,
                        limit=1000
                    )
                    
                    if not df.empty:
                        # Filter to exact time window
                        df['alert_timestamp'] = pd.to_datetime(df['alert_timestamp'])
                        df = df[df['alert_timestamp'] >= cutoff_time]
                        print(f"✅ Method 1: AlertHistoryManager returned {len(df)} signals")
                        return df
                    else:
                        print("⚠️ Method 1: AlertHistoryManager returned empty DataFrame")
                except Exception as e:
                    print(f"⚠️ Method 1 failed: {e}")
            
            # Method 2: Direct database query as fallback
            print("🔄 Trying Method 2: Direct database query...")
            return self._get_signals_direct_query(hours)
            
        except Exception as e:
            print(f"❌ Error getting recent signals: {e}")
            return pd.DataFrame()
    
    def _get_signals_direct_query(self, hours: int) -> pd.DataFrame:
        """Fallback: Direct database query without AlertHistoryManager"""
        try:
            import psycopg2
            from config import DATABASE_URL
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Connect directly
            conn = psycopg2.connect(DATABASE_URL)
            
            # Query for recent signals
            query = """
                SELECT * FROM alert_history 
                WHERE alert_timestamp > %s
                ORDER BY alert_timestamp DESC
                LIMIT 1000
            """
            
            # Use pandas to read directly
            df = pd.read_sql_query(query, conn, params=[cutoff_time])
            conn.close()
            
            print(f"✅ Method 2: Direct query returned {len(df)} signals")
            return df
            
        except Exception as e:
            print(f"❌ Direct query failed: {e}")
            return pd.DataFrame()
    
    def analyze_signal_overview(self, df: pd.DataFrame) -> Dict:
        """Basic signal overview analysis"""
        try:
            overview = {
                "total_signals": len(df),
                "bull_signals": len(df[df['signal_type'] == 'BULL']),
                "bear_signals": len(df[df['signal_type'] == 'BEAR']),
                "avg_confidence": df['confidence_score'].mean() if not df.empty else 0,
                "min_confidence": df['confidence_score'].min() if not df.empty else 0,
                "max_confidence": df['confidence_score'].max() if not df.empty else 0,
                "unique_pairs": df['epic'].nunique() if not df.empty else 0,
                "strategies_used": df['strategy'].unique().tolist() if not df.empty else [],
                "time_range": {
                    "start": df['alert_timestamp'].min().isoformat() if not df.empty else None,
                    "end": df['alert_timestamp'].max().isoformat() if not df.empty else None
                }
            }
            
            print(f"📊 SIGNAL OVERVIEW:")
            print(f"   Total signals: {overview['total_signals']}")
            print(f"   BULL/BEAR: {overview['bull_signals']}/{overview['bear_signals']}")
            print(f"   Average confidence: {overview['avg_confidence']:.1%}")
            print(f"   Confidence range: {overview['min_confidence']:.1%} - {overview['max_confidence']:.1%}")
            print(f"   Unique pairs: {overview['unique_pairs']}")
            print(f"   Strategies: {', '.join(overview['strategies_used'])}")
            
            return overview
            
        except Exception as e:
            print(f"❌ Error in signal overview: {e}")
            return {"error": str(e)}
    
    def analyze_confidence_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze confidence score distribution"""
        try:
            if df.empty:
                return {"error": "No data"}
            
            confidence_bins = {
                "very_high (90-100%)": len(df[df['confidence_score'] >= 0.9]),
                "high (80-90%)": len(df[(df['confidence_score'] >= 0.8) & (df['confidence_score'] < 0.9)]),
                "medium (70-80%)": len(df[(df['confidence_score'] >= 0.7) & (df['confidence_score'] < 0.8)]),
                "low (60-70%)": len(df[(df['confidence_score'] >= 0.6) & (df['confidence_score'] < 0.7)]),
                "very_low (<60%)": len(df[df['confidence_score'] < 0.6])
            }
            
            analysis = {
                "distribution": confidence_bins,
                "avg_confidence": df['confidence_score'].mean(),
                "std_confidence": df['confidence_score'].std(),
                "low_confidence_signals": len(df[df['confidence_score'] < 0.7]),
                "high_confidence_signals": len(df[df['confidence_score'] >= 0.8])
            }
            
            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"   Average: {analysis['avg_confidence']:.1%}")
            print(f"   Std deviation: {analysis['std_confidence']:.1%}")
            print(f"   Low confidence (<70%): {analysis['low_confidence_signals']}")
            print(f"   High confidence (≥80%): {analysis['high_confidence_signals']}")
            
            for range_name, count in confidence_bins.items():
                if count > 0:
                    print(f"   {range_name}: {count}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error in confidence analysis: {e}")
            return {"error": str(e)}
    
    def analyze_strategy_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by strategy"""
        try:
            if df.empty:
                return {"error": "No data"}
            
            strategy_stats = {}
            
            for strategy in df['strategy'].unique():
                strategy_df = df[df['strategy'] == strategy]
                
                strategy_stats[strategy] = {
                    "signal_count": len(strategy_df),
                    "avg_confidence": strategy_df['confidence_score'].mean(),
                    "bull_bear_ratio": len(strategy_df[strategy_df['signal_type'] == 'BULL']) / len(strategy_df),
                    "avg_macd_histogram": strategy_df['macd_histogram'].mean() if 'macd_histogram' in strategy_df.columns else None,
                    "unique_pairs": strategy_df['epic'].nunique()
                }
            
            print(f"\n📈 STRATEGY PERFORMANCE:")
            for strategy, stats in strategy_stats.items():
                print(f"   {strategy}:")
                print(f"     Signals: {stats['signal_count']}")
                print(f"     Avg confidence: {stats['avg_confidence']:.1%}")
                print(f"     Bull ratio: {stats['bull_bear_ratio']:.1%}")
                if stats['avg_macd_histogram']:
                    print(f"     Avg MACD histogram: {stats['avg_macd_histogram']:.6f}")
            
            return strategy_stats
            
        except Exception as e:
            print(f"❌ Error in strategy analysis: {e}")
            return {"error": str(e)}
    
    def analyze_pair_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by currency pair"""
        try:
            if df.empty:
                return {"error": "No data"}
            
            pair_stats = {}
            
            for epic in df['epic'].unique():
                pair_df = df[df['epic'] == epic]
                
                pair_stats[epic] = {
                    "signal_count": len(pair_df),
                    "avg_confidence": pair_df['confidence_score'].mean(),
                    "bull_signals": len(pair_df[pair_df['signal_type'] == 'BULL']),
                    "bear_signals": len(pair_df[pair_df['signal_type'] == 'BEAR']),
                    "strategies_used": pair_df['strategy'].unique().tolist(),
                    "avg_spread_pips": pair_df['spread_pips'].mean() if 'spread_pips' in pair_df.columns else None
                }
            
            print(f"\n💱 PAIR ANALYSIS:")
            for epic, stats in pair_stats.items():
                pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '')
                print(f"   {pair_name}:")
                print(f"     Signals: {stats['signal_count']} (B:{stats['bull_signals']}, S:{stats['bear_signals']})")
                print(f"     Avg confidence: {stats['avg_confidence']:.1%}")
                print(f"     Strategies: {', '.join(stats['strategies_used'])}")
            
            return pair_stats
            
        except Exception as e:
            print(f"❌ Error in pair analysis: {e}")
            return {"error": str(e)}
    
    def analyze_timing_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze timing patterns in signals"""
        try:
            if df.empty:
                return {"error": "No data"}
            
            df['hour'] = df['alert_timestamp'].dt.hour
            df['minute'] = df['alert_timestamp'].dt.minute
            
            # Calculate time between signals
            df_sorted = df.sort_values('alert_timestamp')
            time_diffs = df_sorted['alert_timestamp'].diff().dt.total_seconds() / 60  # minutes
            
            timing_analysis = {
                "signals_by_hour": df['hour'].value_counts().to_dict(),
                "avg_time_between_signals_minutes": time_diffs.mean(),
                "min_time_between_signals_minutes": time_diffs.min(),
                "signals_too_close": len(time_diffs[time_diffs < 30]),  # Less than 30 minutes apart
                "overtrading_periods": []
            }
            
            # Identify overtrading periods (more than 3 signals per hour)
            hourly_counts = df['hour'].value_counts()
            overtrading_hours = hourly_counts[hourly_counts > 3].index.tolist()
            timing_analysis["overtrading_hours"] = overtrading_hours
            
            print(f"\n⏰ TIMING ANALYSIS:")
            print(f"   Avg time between signals: {timing_analysis['avg_time_between_signals_minutes']:.1f} minutes")
            print(f"   Signals too close (<30min): {timing_analysis['signals_too_close']}")
            if overtrading_hours:
                print(f"   Overtrading hours (>3 signals/hour): {overtrading_hours}")
            
            return timing_analysis
            
        except Exception as e:
            print(f"❌ Error in timing analysis: {e}")
            return {"error": str(e)}
    
    def analyze_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyze technical indicator patterns"""
        try:
            if df.empty:
                return {"error": "No data"}
            
            technical_analysis = {}
            
            # MACD analysis
            if 'macd_histogram' in df.columns:
                macd_data = df.dropna(subset=['macd_histogram'])
                if not macd_data.empty:
                    technical_analysis['macd'] = {
                        "avg_histogram": macd_data['macd_histogram'].mean(),
                        "std_histogram": macd_data['macd_histogram'].std(),
                        "weak_signals": len(macd_data[abs(macd_data['macd_histogram']) < 0.0001]),
                        "strong_signals": len(macd_data[abs(macd_data['macd_histogram']) >= 0.0002])
                    }
            
            # EMA analysis
            if 'ema_short' in df.columns and 'ema_long' in df.columns:
                ema_data = df.dropna(subset=['ema_short', 'ema_long'])
                if not ema_data.empty:
                    ema_data['ema_spread'] = abs(ema_data['ema_short'] - ema_data['ema_long'])
                    technical_analysis['ema'] = {
                        "avg_spread": ema_data['ema_spread'].mean(),
                        "narrow_spreads": len(ema_data[ema_data['ema_spread'] < 0.0005])
                    }
            
            # Volume analysis
            if 'volume_ratio' in df.columns:
                volume_data = df.dropna(subset=['volume_ratio'])
                if not volume_data.empty:
                    technical_analysis['volume'] = {
                        "avg_ratio": volume_data['volume_ratio'].mean(),
                        "low_volume_signals": len(volume_data[volume_data['volume_ratio'] < 1.0])
                    }
            
            print(f"\n🔧 TECHNICAL ANALYSIS:")
            if 'macd' in technical_analysis:
                macd = technical_analysis['macd']
                print(f"   MACD histogram avg: {macd['avg_histogram']:.6f}")
                print(f"   Weak MACD signals: {macd['weak_signals']}")
                print(f"   Strong MACD signals: {macd['strong_signals']}")
            
            return technical_analysis
            
        except Exception as e:
            print(f"❌ Error in technical analysis: {e}")
            return {"error": str(e)}
    
    def identify_quality_issues(self, df: pd.DataFrame) -> Dict:
        """Identify potential signal quality issues"""
        try:
            if df.empty:
                return {"error": "No data"}
            
            issues = {
                "low_confidence_signals": [],
                "weak_technical_signals": [],
                "overtrading_pairs": [],
                "suspicious_patterns": []
            }
            
            # Low confidence signals
            low_conf = df[df['confidence_score'] < 0.7]
            if not low_conf.empty:
                issues["low_confidence_signals"] = [
                    {
                        "timestamp": row['alert_timestamp'].isoformat(),
                        "epic": row['epic'],
                        "confidence": row['confidence_score'],
                        "strategy": row['strategy']
                    }
                    for _, row in low_conf.head(10).iterrows()
                ]
            
            # Weak technical signals (if MACD data available)
            if 'macd_histogram' in df.columns:
                weak_macd = df[abs(df['macd_histogram']) < 0.00005]
                if not weak_macd.empty:
                    issues["weak_technical_signals"] = [
                        {
                            "timestamp": row['alert_timestamp'].isoformat(),
                            "epic": row['epic'],
                            "macd_histogram": row['macd_histogram']
                        }
                        for _, row in weak_macd.head(10).iterrows()
                    ]
            
            # Overtrading pairs
            pair_counts = df['epic'].value_counts()
            overtraded = pair_counts[pair_counts > 3]  # More than 3 signals per pair in time period
            if not overtraded.empty:
                issues["overtrading_pairs"] = overtraded.to_dict()
            
            print(f"\n🚨 QUALITY ISSUES IDENTIFIED:")
            print(f"   Low confidence signals: {len(issues['low_confidence_signals'])}")
            print(f"   Weak technical signals: {len(issues['weak_technical_signals'])}")
            print(f"   Overtraded pairs: {len(issues['overtrading_pairs'])}")
            
            return issues
            
        except Exception as e:
            print(f"❌ Error identifying quality issues: {e}")
            return {"error": str(e)}
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if df.empty:
            return ["❌ No data to analyze - check signal generation"]
        
        # Analyze confidence levels
        avg_confidence = df['confidence_score'].mean()
        if avg_confidence < 0.8:
            recommendations.append(f"🎯 URGENT: Raise minimum confidence threshold - current avg {avg_confidence:.1%} is too low")
        
        # Check for overtrading
        if len(df) > 10:  # More than 10 signals in time period
            recommendations.append("⏰ URGENT: Add signal cooldown - too many signals generated")
        
        # Check time clustering
        df['hour'] = df['alert_timestamp'].dt.hour
        hourly_max = df['hour'].value_counts().max()
        if hourly_max > 3:
            recommendations.append("🚫 URGENT: Limit signals per hour - detected overtrading")
        
        # Check technical strength
        if 'macd_histogram' in df.columns:
            weak_macd_count = len(df[abs(df['macd_histogram']) < 0.0001])
            if weak_macd_count > len(df) * 0.5:
                recommendations.append("🔧 Fix MACD thresholds - too many weak signals")
        
        # Check strategy distribution
        strategy_counts = df['strategy'].value_counts()
        if len(strategy_counts) == 1:
            recommendations.append("⚖️ Enable multiple strategies for better diversification")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return recommendations
    
    def export_analysis_report(self, filename: str = None) -> str:
        """Export detailed analysis report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_analysis_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write("TRADING PERFORMANCE ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                # Write all analysis results
                for section, data in self.analysis_results.items():
                    f.write(f"\n{section.upper().replace('_', ' ')}:\n")
                    f.write("-" * 30 + "\n")
                    f.write(str(data) + "\n")
            
            print(f"\n📁 Analysis report saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return ""


def main():
    """Main execution function"""
    print("🚨 EMERGENCY TRADING PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    analyzer = TradingPerformanceAnalyzer()
    
    # Analyze recent performance (last 24 hours by default)
    hours = 24
    if len(sys.argv) > 1:
        try:
            hours = int(sys.argv[1])
        except ValueError:
            print(f"Invalid hours parameter: {sys.argv[1]}, using default 24")
    
    results = analyzer.analyze_recent_performance(hours)
    
    if "error" not in results:
        # Export detailed report
        analyzer.export_analysis_report()
        
        print(f"\n🎯 ANALYSIS COMPLETE")
        print("Check the generated report file for detailed analysis")
    else:
        print(f"❌ Analysis failed: {results['error']}")


if __name__ == "__main__":
    main()