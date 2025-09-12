#!/usr/bin/env python3
"""
Stream Validation Monitoring Dashboard
Shows real-time statistics of stream vs API validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from core.database import DatabaseManager
try:
    import config
except ImportError:
    from forex_scanner import config

logger = logging.getLogger(__name__)

class StreamValidationMonitor:
    """Monitor stream validation performance and results"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def get_validation_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get validation summary for the specified time period"""
        
        since_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Get validation statistics
        validation_stats_query = """
            SELECT 
                data_source,
                severity,
                COUNT(*) as count,
                AVG(price_difference_pips) as avg_pips_diff,
                MAX(price_difference_pips) as max_pips_diff,
                MIN(timestamp) as first_validation,
                MAX(timestamp) as last_validation
            FROM price_validation_log
            WHERE validation_type = 'STREAM_API_VALIDATION'
              AND timestamp >= %s
            GROUP BY data_source, severity
            ORDER BY count DESC
        """
        
        validation_stats = self.db_manager.execute_query(
            validation_stats_query, 
            params=[since_time]
        )
        
        # Get discrepancy trends by epic
        epic_trends_query = """
            SELECT 
                epic,
                COUNT(*) as total_validations,
                COUNT(CASE WHEN severity = 'CRITICAL' THEN 1 END) as critical_count,
                COUNT(CASE WHEN severity = 'WARNING' THEN 1 END) as warning_count,
                AVG(price_difference_pips) as avg_pips_diff,
                MAX(price_difference_pips) as max_pips_diff
            FROM price_validation_log
            WHERE validation_type = 'STREAM_API_VALIDATION'
              AND timestamp >= %s
              AND epic NOT LIKE '%ETH%'
              AND epic NOT LIKE '%BTC%' 
            GROUP BY epic
            ORDER BY critical_count DESC, avg_pips_diff DESC
        """
        
        epic_trends = self.db_manager.execute_query(
            epic_trends_query,
            params=[since_time] 
        )
        
        # Get recent critical discrepancies
        recent_critical_query = """
            SELECT 
                timestamp,
                epic,
                old_value as stream_price,
                new_value as api_price,
                price_difference_pips,
                message,
                resolution
            FROM price_validation_log
            WHERE validation_type = 'STREAM_API_VALIDATION'
              AND severity = 'CRITICAL'
              AND timestamp >= %s
            ORDER BY timestamp DESC
            LIMIT 20
        """
        
        recent_critical = self.db_manager.execute_query(
            recent_critical_query,
            params=[since_time]
        )
        
        # Calculate overall health score
        health_score = self._calculate_validation_health_score(validation_stats, epic_trends)
        
        return {
            'timestamp': datetime.utcnow(),
            'time_period_hours': hours_back,
            'validation_stats': validation_stats,
            'epic_trends': epic_trends,
            'recent_critical': recent_critical,
            'health_score': health_score,
            'total_validations': validation_stats['count'].sum() if isinstance(validation_stats, pd.DataFrame) and not validation_stats.empty else 0
        }
    
    def _calculate_validation_health_score(self, validation_stats, epic_trends) -> Dict[str, Any]:
        """Calculate overall validation system health score"""
        
        try:
            if not isinstance(validation_stats, pd.DataFrame) or validation_stats.empty:
                return {
                    'score': 100,
                    'status': 'UNKNOWN',
                    'description': 'No validation data available'
                }
            
            total_validations = validation_stats['count'].sum()
            critical_validations = validation_stats[validation_stats['severity'] == 'CRITICAL']['count'].sum() if 'CRITICAL' in validation_stats['severity'].values else 0
            
            if total_validations == 0:
                return {
                    'score': 100,
                    'status': 'NO_DATA',
                    'description': 'No validations performed'
                }
            
            # Calculate health score (100 = perfect, 0 = critical issues)
            critical_rate = critical_validations / total_validations
            score = max(0, 100 - (critical_rate * 100))
            
            # Determine status
            if score >= 95:
                status = 'EXCELLENT'
                description = 'Stream data highly accurate'
            elif score >= 85:
                status = 'GOOD' 
                description = 'Stream data mostly accurate with minor discrepancies'
            elif score >= 70:
                status = 'ACCEPTABLE'
                description = 'Stream data acceptable but monitoring needed'
            elif score >= 50:
                status = 'DEGRADED'
                description = 'Stream data quality degraded - investigation needed'
            else:
                status = 'CRITICAL'
                description = 'Stream data integrity compromised - immediate action required'
            
            return {
                'score': round(score, 1),
                'status': status,
                'description': description,
                'total_validations': int(total_validations),
                'critical_issues': int(critical_validations),
                'critical_rate_pct': round(critical_rate * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return {
                'score': 0,
                'status': 'ERROR',
                'description': f'Error calculating health: {str(e)}'
            }
    
    def generate_validation_report(self, hours_back: int = 24) -> str:
        """Generate human-readable validation report"""
        
        try:
            summary = self.get_validation_summary(hours_back)
            
            report = [
                f"📊 STREAM VALIDATION REPORT",
                f"Generated: {summary['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"Period: Last {hours_back} hours",
                "=" * 60,
                ""
            ]
            
            # Health score section
            health = summary['health_score']
            status_emoji = {
                'EXCELLENT': '🟢',
                'GOOD': '🟢', 
                'ACCEPTABLE': '🟡',
                'DEGRADED': '🟠',
                'CRITICAL': '🔴',
                'ERROR': '⚠️',
                'UNKNOWN': '⚪',
                'NO_DATA': '⚪'
            }
            
            emoji = status_emoji.get(health['status'], '⚪')
            report.extend([
                f"{emoji} OVERALL HEALTH: {health['status']} (Score: {health['score']}/100)",
                f"Description: {health['description']}",
                ""
            ])
            
            # Validation statistics
            validation_stats = summary['validation_stats']
            if isinstance(validation_stats, pd.DataFrame) and not validation_stats.empty:
                report.append("📈 VALIDATION STATISTICS:")
                report.append("-" * 30)
                
                for _, row in validation_stats.iterrows():
                    severity_emoji = "🔴" if row['severity'] == 'CRITICAL' else "🟡" if row['severity'] == 'WARNING' else "ℹ️"
                    report.append(
                        f"{severity_emoji} {row['severity']}: {row['count']} validations "
                        f"(avg: {row['avg_pips_diff']:.2f} pips, max: {row['max_pips_diff']:.2f} pips)"
                    )
                report.append("")
            
            # Epic-specific trends
            epic_trends = summary['epic_trends']
            if isinstance(epic_trends, pd.DataFrame) and not epic_trends.empty:
                report.append("🎯 VALIDATION BY PAIR:")
                report.append("-" * 30)
                
                for _, row in epic_trends.head(10).iterrows():
                    pair_status = "🔴" if row['critical_count'] > 0 else "🟡" if row['warning_count'] > 0 else "🟢"
                    report.append(
                        f"{pair_status} {row['epic']}: {row['total_validations']} validations, "
                        f"{row['critical_count']} critical, avg {row['avg_pips_diff']:.2f} pips diff"
                    )
                report.append("")
            
            # Recent critical issues
            recent_critical = summary['recent_critical']
            if isinstance(recent_critical, pd.DataFrame) and not recent_critical.empty:
                report.append("🚨 RECENT CRITICAL DISCREPANCIES:")
                report.append("-" * 40)
                
                for _, row in recent_critical.head(5).iterrows():
                    timestamp = row['timestamp'].strftime('%H:%M:%S') if hasattr(row['timestamp'], 'strftime') else str(row['timestamp'])
                    report.append(
                        f"🔴 {timestamp} {row['epic']}: "
                        f"Stream={row['stream_price']:.5f} vs API={row['api_price']:.5f} "
                        f"({row['price_difference_pips']:.1f} pips)"
                    )
                report.append("")
            
            # Recommendations
            report.extend([
                "🔧 RECOMMENDATIONS:",
                self._get_validation_recommendations(health),
                "",
                "=" * 60,
                f"Report generated by Stream Validation Monitor v1.0"
            ])
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return f"❌ Error generating validation report: {str(e)}"
    
    def _get_validation_recommendations(self, health: Dict[str, Any]) -> str:
        """Generate recommendations based on validation health"""
        
        status = health.get('status', 'UNKNOWN')
        
        recommendations = {
            'EXCELLENT': [
                "✅ Stream validation performing excellently",
                "✅ Continue current monitoring schedule", 
                "✅ System ready for production trading"
            ],
            'GOOD': [
                "✅ Stream validation performing well",
                "📊 Continue monitoring for trends",
                "✅ Safe for production trading"
            ],
            'ACCEPTABLE': [
                "⚠️ Stream validation showing some discrepancies",
                "📊 Increase monitoring frequency",
                "🔍 Investigate patterns in validation failures"
            ],
            'DEGRADED': [
                "⚠️ Stream data quality degrading",
                "🔄 Consider restarting streaming services",
                "📞 Alert development team",
                "⚖️ Consider reducing position sizes"
            ],
            'CRITICAL': [
                "🛑 HALT TRADING immediately",
                "🔄 Restart all streaming services",
                "📞 Escalate to operations team",
                "🔍 Investigate data source connectivity"
            ]
        }
        
        default_recommendations = [
            "📊 Monitor validation system health",
            "🔍 Investigate any recurring issues"
        ]
        
        recs = recommendations.get(status, default_recommendations)
        return "\n".join(f"  {rec}" for rec in recs)
    
    def run_validation_check(self, hours_back: int = 24) -> bool:
        """Run validation check and log results"""
        
        try:
            logger.info(f"🔍 Checking stream validation health for last {hours_back} hours...")
            
            # Generate report
            report = self.generate_validation_report(hours_back)
            
            # Get health status for logging level
            summary = self.get_validation_summary(hours_back)
            health_status = summary['health_score']['status']
            
            # Log based on health status
            if health_status in ['CRITICAL', 'ERROR']:
                logger.critical(f"\n{report}")
                return False
            elif health_status == 'DEGRADED':
                logger.warning(f"\n{report}")
                return True
            else:
                logger.info("✅ Stream validation health check passed")
                logger.debug(f"\n{report}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Validation health check failed: {e}")
            return False

def main():
    """CLI entry point for stream validation monitoring"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize database manager
        db_url = getattr(config, 'DATABASE_URL', 'postgresql://postgres:@localhost:5432/forex')
        db_manager = DatabaseManager(db_url)
        
        # Run monitoring check
        monitor = StreamValidationMonitor(db_manager)
        
        # Default to 6 hours for more frequent checks
        hours_back = int(sys.argv[1]) if len(sys.argv) > 1 else 6
        success = monitor.run_validation_check(hours_back)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"❌ Validation monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()