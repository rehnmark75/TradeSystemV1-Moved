#!/usr/bin/env python3
"""
Data Quality Monitoring Script
Monitors price data integrity and sends alerts for critical issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from core.database import DatabaseManager
try:
    import config
except ImportError:
    from forex_scanner import config

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """Monitor data quality and generate alerts for trading safety"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.alert_thresholds = {
            'critical_pips': getattr(config, 'MAX_PRICE_DISCREPANCY_PIPS', 10.0),
            'min_quality': getattr(config, 'MIN_QUALITY_SCORE_FOR_TRADING', 0.5),
            'max_stale_minutes': 60,
            'max_critical_issues_per_hour': 5
        }
    
    def check_current_data_quality(self) -> Dict[str, Any]:
        """Check current data quality status across all forex pairs"""
        
        # Get current data quality dashboard
        dashboard_query = """
            SELECT * FROM data_quality_dashboard 
            WHERE epic NOT LIKE '%ETH%' 
              AND epic NOT LIKE '%BTC%' 
              AND epic NOT LIKE '%CFE%'
            ORDER BY minutes_stale DESC, critical_count DESC
        """
        
        dashboard = self.db_manager.execute_query(dashboard_query)
        
        # Get recent critical discrepancies
        critical_query = """
            SELECT 
                epic,
                COUNT(*) as issue_count,
                MAX(pips_diff) as max_pips,
                MAX(start_time) as latest_issue
            FROM price_discrepancies 
            WHERE epic NOT LIKE '%ETH%' 
              AND epic NOT LIKE '%BTC%'
              AND epic NOT LIKE '%CFE%'
              AND start_time >= NOW() - INTERVAL '1 hour'
              AND pips_diff > %s
            GROUP BY epic
            ORDER BY max_pips DESC
        """
        
        critical_issues = self.db_manager.execute_query(
            critical_query, 
            params=[self.alert_thresholds['critical_pips']]
        )
        
        # Get recent validation logs
        validation_query = """
            SELECT 
                data_source,
                severity,
                COUNT(*) as count,
                MAX(timestamp) as latest
            FROM price_validation_log
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
              AND epic NOT LIKE '%ETH%'
            GROUP BY data_source, severity
            ORDER BY count DESC
        """
        
        validation_logs = self.db_manager.execute_query(validation_query)
        
        return {
            'timestamp': datetime.now(),
            'dashboard': dashboard,
            'critical_issues': critical_issues,
            'validation_logs': validation_logs,
            'alert_level': self._calculate_alert_level(dashboard, critical_issues, validation_logs)
        }
    
    def _calculate_alert_level(self, dashboard, critical_issues, validation_logs) -> str:
        """Calculate overall system alert level"""
        
        if not isinstance(dashboard, pd.DataFrame):
            return 'UNKNOWN'
        
        # Check for stale data (no updates > 1 hour)
        if not dashboard.empty and dashboard['minutes_stale'].max() > 60:
            return 'CRITICAL'
        
        # Check for multiple critical quality issues
        if not isinstance(critical_issues, pd.DataFrame):
            critical_count = 0
        else:
            critical_count = len(critical_issues) if not critical_issues.empty else 0
        
        if critical_count > self.alert_thresholds['max_critical_issues_per_hour']:
            return 'CRITICAL'
        elif critical_count > 0:
            return 'WARNING'
        
        # Check for low quality scores
        if not dashboard.empty:
            min_quality = dashboard['min_quality'].min()
            if min_quality < self.alert_thresholds['min_quality']:
                return 'WARNING'
        
        return 'HEALTHY'
    
    def generate_alert_report(self, status: Dict[str, Any]) -> str:
        """Generate human-readable alert report"""
        
        alert_level = status['alert_level']
        timestamp = status['timestamp']
        
        report = [
            f"🚨 DATA QUALITY ALERT - {alert_level} 🚨",
            f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 60,
            ""
        ]
        
        # Dashboard summary
        dashboard = status['dashboard']
        if isinstance(dashboard, pd.DataFrame) and not dashboard.empty:
            report.append("📊 CURRENT STATUS BY PAIR:")
            for _, row in dashboard.head(10).iterrows():
                status_emoji = "🔴" if row['minutes_stale'] > 60 else "🟡" if row['critical_count'] > 0 else "🟢"
                report.append(
                    f"  {status_emoji} {row['epic']} ({row['timeframe']}m): "
                    f"Quality={row['avg_quality']:.3f}, "
                    f"Stale={row['minutes_stale']:.1f}min, "
                    f"Critical={row['critical_count']}"
                )
            report.append("")
        
        # Critical issues
        critical_issues = status['critical_issues']
        if isinstance(critical_issues, pd.DataFrame) and not critical_issues.empty:
            report.append("⚠️ CRITICAL PRICE DISCREPANCIES (LAST HOUR):")
            for _, row in critical_issues.iterrows():
                report.append(
                    f"  🚨 {row['epic']}: {row['max_pips']:.1f} pips difference "
                    f"({row['issue_count']} occurrences)"
                )
            report.append("")
        
        # Validation summary
        validation_logs = status['validation_logs']
        if isinstance(validation_logs, pd.DataFrame) and not validation_logs.empty:
            report.append("🔍 VALIDATION SUMMARY (LAST HOUR):")
            for _, row in validation_logs.iterrows():
                severity_emoji = "🔴" if row['severity'] == 'CRITICAL' else "🟡" if row['severity'] == 'WARNING' else "ℹ️"
                report.append(
                    f"  {severity_emoji} {row['data_source']}: {row['count']} {row['severity']} issues"
                )
            report.append("")
        
        # Recommendations
        report.extend([
            "🔧 RECOMMENDED ACTIONS:",
            self._get_recommendations(status),
            "",
            "=" * 60,
            f"Report generated by Data Quality Monitor v1.0"
        ])
        
        return "\n".join(report)
    
    def _get_recommendations(self, status: Dict[str, Any]) -> str:
        """Generate specific recommendations based on current issues"""
        
        alert_level = status['alert_level']
        recommendations = []
        
        if alert_level == 'CRITICAL':
            recommendations.extend([
                "  🛑 HALT ALL TRADING immediately until data issues resolved",
                "  🔧 Restart streaming services (fastapi-stream container)",
                "  📞 Contact system administrator",
                "  🔍 Investigate data source connectivity"
            ])
        elif alert_level == 'WARNING':
            recommendations.extend([
                "  ⚠️ Review trading positions for affected pairs",
                "  📊 Monitor data quality closely for next 30 minutes",
                "  🔄 Consider reducing position sizes until issues resolve",
                "  📋 Document issues for post-incident analysis"
            ])
        else:
            recommendations.append("  ✅ System operating normally - continue monitoring")
        
        return "\n".join(recommendations)
    
    def run_monitoring_check(self) -> bool:
        """Run complete monitoring check and log results"""
        
        try:
            logger.info("🔍 Starting data quality monitoring check...")
            
            # Get current status
            status = self.check_current_data_quality()
            
            # Generate report
            report = self.generate_alert_report(status)
            
            # Log based on alert level
            alert_level = status['alert_level']
            if alert_level == 'CRITICAL':
                logger.critical(f"\n{report}")
            elif alert_level == 'WARNING':
                logger.warning(f"\n{report}")
            else:
                logger.info(f"✅ Data quality check passed - system healthy")
                logger.debug(f"\n{report}")
            
            return alert_level in ['HEALTHY', 'WARNING']
            
        except Exception as e:
            logger.error(f"❌ Data quality monitoring failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

def main():
    """CLI entry point for data quality monitoring"""
    
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
        monitor = DataQualityMonitor(db_manager)
        success = monitor.run_monitoring_check()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"❌ Monitoring script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()