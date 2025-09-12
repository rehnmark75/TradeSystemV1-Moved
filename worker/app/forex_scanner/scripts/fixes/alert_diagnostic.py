#!/usr/bin/env python3
"""
Alert System Diagnostic Script
Analyzes current alert patterns and provides enhancement recommendations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_current_alerts():
    """Analyze current alert patterns in the database"""
    
    print("🔍 ALERT SYSTEM DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Check alert table structure
        print("\n📋 1. ALERT TABLE STRUCTURE")
        print("-" * 30)
        
        columns_query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'alert_history'
        ORDER BY ordinal_position
        """
        
        columns = db.execute_query(columns_query)
        print(f"Table has {len(columns)} columns:")
        for _, col in columns.iterrows():
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            print(f"  • {col['column_name']: <25} {col['data_type']: <15} {nullable}")
        
        # Analyze recent alerts
        print("\n📊 2. RECENT ALERT ANALYSIS (Last 24 hours)")
        print("-" * 45)
        
        recent_alerts_query = """
        SELECT 
            COUNT(*) as total_alerts,
            COUNT(DISTINCT epic) as unique_pairs,
            COUNT(DISTINCT signal_type) as signal_types,
            COUNT(DISTINCT strategy) as strategies,
            MIN(alert_timestamp) as earliest_alert,
            MAX(alert_timestamp) as latest_alert
        FROM alert_history 
        WHERE alert_timestamp >= NOW() - INTERVAL '24 hours'
        """
        
        summary = db.execute_query(recent_alerts_query)
        if len(summary) > 0:
            row = summary.iloc[0]
            print(f"Total Alerts: {row['total_alerts']}")
            print(f"Unique Pairs: {row['unique_pairs']}")
            print(f"Signal Types: {row['signal_types']}")
            print(f"Strategies: {row['strategies']}")
            print(f"Time Range: {row['earliest_alert']} → {row['latest_alert']}")
        
        # Signal type distribution
        print("\n📈 3. SIGNAL TYPE DISTRIBUTION")
        print("-" * 35)
        
        signal_dist_query = """
        SELECT 
            signal_type,
            COUNT(*) as count,
            ROUND(AVG(confidence_score), 3) as avg_confidence,
            ROUND(MIN(confidence_score), 3) as min_confidence,
            ROUND(MAX(confidence_score), 3) as max_confidence
        FROM alert_history 
        WHERE alert_timestamp >= NOW() - INTERVAL '24 hours'
        GROUP BY signal_type
        ORDER BY count DESC
        """
        
        signal_dist = db.execute_query(signal_dist_query)
        total_recent = sum(signal_dist['count']) if len(signal_dist) > 0 else 0
        
        for _, row in signal_dist.iterrows():
            percentage = (row['count'] / total_recent * 100) if total_recent > 0 else 0
            print(f"  {row['signal_type']: <8} {row['count']: >4} ({percentage: >5.1f}%) "
                  f"Conf: {row['avg_confidence']} ({row['min_confidence']}-{row['max_confidence']})")
        
        # Strategy distribution
        print("\n🎯 4. STRATEGY DISTRIBUTION")
        print("-" * 30)
        
        strategy_dist_query = """
        SELECT 
            strategy,
            COUNT(*) as count,
            ROUND(AVG(confidence_score), 3) as avg_confidence
        FROM alert_history 
        WHERE alert_timestamp >= NOW() - INTERVAL '24 hours'
        GROUP BY strategy
        ORDER BY count DESC
        """
        
        strategy_dist = db.execute_query(strategy_dist_query)
        for _, row in strategy_dist.iterrows():
            percentage = (row['count'] / total_recent * 100) if total_recent > 0 else 0
            print(f"  {row['strategy']: <15} {row['count']: >4} ({percentage: >5.1f}%) "
                  f"Avg Conf: {row['avg_confidence']}")
        
        # Timestamp analysis
        print("\n⏰ 5. TIMESTAMP ANALYSIS")
        print("-" * 25)
        
        timestamp_query = """
        SELECT 
            alert_timestamp,
            strategy_metadata,
            signal_type,
            epic,
            confidence_score
        FROM alert_history 
        WHERE alert_timestamp >= NOW() - INTERVAL '6 hours'
        ORDER BY alert_timestamp DESC
        LIMIT 10
        """
        
        recent_timestamps = db.execute_query(timestamp_query)
        
        print("Recent alerts with timestamp comparison:")
        for _, row in recent_timestamps.iterrows():
            alert_time = row['alert_timestamp']
            
            # Extract signal timestamp from metadata
            signal_time = "N/A"
            if row['strategy_metadata']:
                try:
                    metadata = json.loads(row['strategy_metadata'])
                    signal_time = metadata.get('signal_timestamp', 'N/A')
                except:
                    pass
            
            print(f"  {row['epic']: <20} {row['signal_type']: <6} Alert: {alert_time}")
            print(f"  {' ': <20} {' ': <6} Signal: {signal_time}")
            print(f"  {' ': <20} {' ': <6} Confidence: {row['confidence_score']:.3f}")
            print()
        
        # Detect patterns and issues
        print("\n🚨 6. DETECTED ISSUES & RECOMMENDATIONS")
        print("-" * 40)
        
        issues = detect_issues(signal_dist, strategy_dist, recent_timestamps)
        
        if issues:
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            print("  ✅ No major issues detected")
        
        # Enhancement recommendations
        print("\n💡 7. ENHANCEMENT RECOMMENDATIONS")
        print("-" * 35)
        
        recommendations = generate_recommendations(signal_dist, strategy_dist, total_recent)
        
        for rec in recommendations:
            print(f"  📌 {rec}")
        
        # Configuration suggestions
        print("\n⚙️ 8. CONFIGURATION SUGGESTIONS")
        print("-" * 32)
        
        config_suggestions = generate_config_suggestions(signal_dist, strategy_dist)
        
        print("Add/modify these settings in your config.py:")
        print()
        for setting in config_suggestions:
            print(f"  {setting}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Diagnostic analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def detect_issues(signal_dist, strategy_dist, recent_timestamps) -> List[str]:
    """Detect potential issues in alert patterns"""
    issues = []
    
    # Check signal type imbalance
    total_signals = sum(signal_dist['count']) if len(signal_dist) > 0 else 0
    if total_signals > 0:
        bull_signals = signal_dist[signal_dist['signal_type'] == 'BULL']['count'].sum()
        bear_signals = signal_dist[signal_dist['signal_type'] == 'BEAR']['count'].sum()
        
        bull_ratio = bull_signals / total_signals
        bear_ratio = bear_signals / total_signals
        
        if bull_ratio > 0.9:
            issues.append(f"Extreme bull bias: {bull_ratio:.1%} bull signals (should be ~50%)")
        elif bear_ratio > 0.9:
            issues.append(f"Extreme bear bias: {bear_ratio:.1%} bear signals (should be ~50%)")
        elif bull_ratio > 0.8:
            issues.append(f"High bull bias: {bull_ratio:.1%} bull signals")
        elif bear_ratio > 0.8:
            issues.append(f"High bear bias: {bear_ratio:.1%} bear signals")
    
    # Check strategy dominance
    if len(strategy_dist) > 0:
        total_strategy_signals = sum(strategy_dist['count'])
        max_strategy_count = max(strategy_dist['count'])
        dominance_ratio = max_strategy_count / total_strategy_signals if total_strategy_signals > 0 else 0
        
        if dominance_ratio > 0.95:
            dominant_strategy = strategy_dist.iloc[0]['strategy']
            issues.append(f"Single strategy dominance: {dominant_strategy} ({dominance_ratio:.1%})")
    
    # Check timestamp patterns
    if len(recent_timestamps) > 5:
        # Look for identical timestamps in strategy_metadata
        metadata_timestamps = []
        for _, row in recent_timestamps.iterrows():
            if row['strategy_metadata']:
                try:
                    metadata = json.loads(row['strategy_metadata'])
                    signal_time = metadata.get('signal_timestamp')
                    if signal_time:
                        metadata_timestamps.append(signal_time)
                except:
                    pass
        
        if len(set(metadata_timestamps)) < len(metadata_timestamps) * 0.5:
            issues.append("Many signals have identical timestamps in strategy_metadata")
        
        # Check if all signal timestamps are very recent (indicating real-time detection)
        try:
            current_time = datetime.now()
            recent_count = 0
            for timestamp_str in metadata_timestamps:
                if timestamp_str and timestamp_str != 'N/A':
                    # Parse timestamp
                    signal_time = datetime.fromisoformat(timestamp_str.replace('+00:00', ''))
                    time_diff = abs((current_time - signal_time).total_seconds())
                    if time_diff < 3600:  # Less than 1 hour old
                        recent_count += 1
            
            if recent_count == len(metadata_timestamps) and len(metadata_timestamps) > 3:
                issues.append("All signals appear to be real-time (no historical signal detection)")
        except:
            pass
    
    # Check confidence score patterns
    if len(signal_dist) > 0:
        confidence_variance = signal_dist['avg_confidence'].var()
        if confidence_variance < 0.01:  # Very low variance
            issues.append("Very similar confidence scores across signals (indicates poor calibration)")
    
    return issues

def generate_recommendations(signal_dist, strategy_dist, total_alerts) -> List[str]:
    """Generate enhancement recommendations"""
    recommendations = []
    
    # Alert frequency recommendations
    if total_alerts > 100:
        recommendations.append("High alert frequency detected - consider implementing cooldown periods")
    elif total_alerts < 5:
        recommendations.append("Low alert frequency - consider lowering confidence thresholds")
    
    # Signal balance recommendations
    if len(signal_dist) > 0:
        bull_count = signal_dist[signal_dist['signal_type'] == 'BULL']['count'].sum()
        bear_count = signal_dist[signal_dist['signal_type'] == 'BEAR']['count'].sum()
        
        if bull_count > bear_count * 3:
            recommendations.append("Calibrate bear signal detection - thresholds may be too strict")
        elif bear_count > bull_count * 3:
            recommendations.append("Calibrate bull signal detection - thresholds may be too strict")
    
    # Strategy diversity recommendations
    if len(strategy_dist) == 1:
        recommendations.append("Enable additional strategies for better signal diversity")
    elif len(strategy_dist) > 1:
        max_strategy_ratio = max(strategy_dist['count']) / total_alerts if total_alerts > 0 else 0
        if max_strategy_ratio > 0.8:
            recommendations.append("Balance strategy weights - one strategy is dominating")
    
    # Timestamp recommendations
    recommendations.append("Implement enhanced timestamp tracking to separate market time vs scan time")
    recommendations.append("Add signal deduplication based on market timestamp + epic + signal type")
    recommendations.append("Consider implementing signal cooldown periods per epic (5-15 minutes)")
    
    # Quality improvements
    recommendations.append("Add market session analysis to strategy_metadata")
    recommendations.append("Include data age tracking to distinguish live vs historical signals")
    recommendations.append("Implement anomaly detection for unusual alert patterns")
    
    return recommendations

def generate_config_suggestions(signal_dist, strategy_dist) -> List[str]:
    """Generate configuration suggestions"""
    suggestions = []
    
    # Basic alert control settings
    suggestions.append("# Alert frequency control")
    suggestions.append("MIN_SIGNAL_INTERVAL_SECONDS = 300  # 5 minutes between signals per epic")
    suggestions.append("MAX_ALERTS_PER_HOUR = 20")
    suggestions.append("ENABLE_SIGNAL_DEDUPLICATION = True")
    suggestions.append("")
    
    # Strategy balance settings
    suggestions.append("# Strategy calibration")
    if len(signal_dist) > 0:
        bull_count = signal_dist[signal_dist['signal_type'] == 'BULL']['count'].sum()
        bear_count = signal_dist[signal_dist['signal_type'] == 'BEAR']['count'].sum()
        
        if bull_count > bear_count * 2:
            suggestions.append("# Adjust for bear signal detection")
            suggestions.append("EMA_BEAR_THRESHOLD = 0.60  # Lower from 0.70 for more bear signals")
            suggestions.append("MACD_BEAR_SENSITIVITY = 1.2  # Increase sensitivity")
        elif bear_count > bull_count * 2:
            suggestions.append("# Adjust for bull signal detection")
            suggestions.append("EMA_BULL_THRESHOLD = 0.60  # Lower from 0.70 for more bull signals")
            suggestions.append("MACD_BULL_SENSITIVITY = 1.2  # Increase sensitivity")
        else:
            suggestions.append("EMA_BULL_THRESHOLD = 0.65")
            suggestions.append("EMA_BEAR_THRESHOLD = 0.65")
    
    suggestions.append("")
    
    # Enhanced metadata settings
    suggestions.append("# Enhanced alert metadata")
    suggestions.append("ENABLE_MARKET_SESSION_TRACKING = True")
    suggestions.append("ENABLE_DATA_AGE_TRACKING = True")
    suggestions.append("ENABLE_VOLATILITY_ASSESSMENT = True")
    suggestions.append("USER_TIMEZONE = 'Europe/Stockholm'")
    suggestions.append("")
    
    # Monitoring settings
    suggestions.append("# Alert monitoring")
    suggestions.append("ENABLE_ALERT_ANOMALY_DETECTION = True")
    suggestions.append("ALERT_ANALYSIS_WINDOW_HOURS = 24")
    suggestions.append("LOG_DETAILED_ALERT_INFO = True")
    
    return suggestions

def test_enhanced_alert_system():
    """Test the enhanced alert system"""
    print("\n🧪 TESTING ENHANCED ALERT SYSTEM")
    print("=" * 40)
    
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        
        # Test signal with proper timestamps
        test_signal = {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'signal_type': 'BULL',
            'strategy': 'enhanced_ema',
            'confidence_score': 0.85,
            'price': 1.08450,
            'timeframe': '15m',
            'timestamp': datetime(2025, 7, 11, 10, 30, 0),  # Market data timestamp
            'scan_time': datetime.now(),  # Current scan time
            'strategy_config': {
                'ema_fast': 9,
                'ema_slow': 21,
                'ema_trend': 200
            },
            'strategy_indicators': {
                'ema_9': 1.08445,
                'ema_21': 1.08425,
                'ema_200': 1.08200
            },
            'volume_ratio': 1.5,
            'market_session': 'london'
        }
        
        print("📊 Test Signal Structure:")
        print(f"  Epic: {test_signal['epic']}")
        print(f"  Signal Type: {test_signal['signal_type']}")
        print(f"  Market Time: {test_signal['timestamp']}")
        print(f"  Scan Time: {test_signal['scan_time']}")
        print(f"  Confidence: {test_signal['confidence_score']:.1%}")
        
        # Test with enhanced alert manager if available
        try:
            from alerts.enhanced_alert_history import EnhancedAlertHistoryManager
            
            enhanced_manager = EnhancedAlertHistoryManager(db)
            alert_id = enhanced_manager.save_alert(test_signal, "Test enhanced alert")
            
            if alert_id:
                print(f"✅ Enhanced alert saved successfully (ID: {alert_id})")
                
                # Get statistics
                stats = enhanced_manager.get_alert_statistics()
                print(f"📈 Alert Statistics: {stats}")
            else:
                print("❌ Enhanced alert save failed")
                
        except ImportError:
            print("⚠️ Enhanced alert manager not available - using standard system")
            
            # Test with standard alert system
            from alerts.alert_history import AlertHistoryManager
            
            standard_manager = AlertHistoryManager(db)
            alert_id = standard_manager.save_alert(test_signal, "Test standard alert")
            
            if alert_id:
                print(f"✅ Standard alert saved successfully (ID: {alert_id})")
            else:
                print("❌ Standard alert save failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced alert system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_migration_script():
    """Generate SQL migration script for enhanced alert system"""
    print("\n📝 ALERT SYSTEM MIGRATION SCRIPT")
    print("=" * 38)
    
    migration_sql = """
-- Alert System Enhancement Migration Script
-- Run this to add enhanced columns to your alert_history table

-- Add enhanced columns if they don't exist
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS signal_hash VARCHAR(32);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS data_source VARCHAR(20);
ALTER TABLE alert_history ADD COLUMN IF NOT EXISTS market_timestamp TIMESTAMP;

-- Add enhanced indexes
CREATE INDEX IF NOT EXISTS idx_alert_history_signal_hash ON alert_history(signal_hash);
CREATE INDEX IF NOT EXISTS idx_alert_history_data_source ON alert_history(data_source);
CREATE INDEX IF NOT EXISTS idx_alert_history_market_timestamp ON alert_history(market_timestamp);
CREATE INDEX IF NOT EXISTS idx_alert_history_epic_signal_time ON alert_history(epic, signal_type, alert_timestamp);

-- Create alert analysis view
CREATE OR REPLACE VIEW alert_analysis AS
SELECT 
    DATE_TRUNC('hour', alert_timestamp) as hour_bucket,
    signal_type,
    strategy,
    data_source,
    COUNT(*) as alert_count,
    AVG(confidence_score) as avg_confidence,
    COUNT(DISTINCT epic) as unique_pairs,
    MIN(alert_timestamp) as first_alert,
    MAX(alert_timestamp) as last_alert
FROM alert_history 
GROUP BY hour_bucket, signal_type, strategy, data_source
ORDER BY hour_bucket DESC;

-- Create anomaly detection function
CREATE OR REPLACE FUNCTION detect_alert_anomalies(hours_back INTEGER DEFAULT 24)
RETURNS TABLE(
    anomaly_type TEXT,
    severity TEXT,
    description TEXT,
    metric_value NUMERIC
) AS $
BEGIN
    -- Signal type imbalance
    RETURN QUERY
    WITH signal_stats AS (
        SELECT 
            signal_type,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
        FROM alert_history 
        WHERE alert_timestamp >= NOW() - INTERVAL '1 hour' * hours_back
        GROUP BY signal_type
    )
    SELECT 
        'signal_imbalance'::TEXT,
        CASE WHEN percentage > 90 THEN 'HIGH' ELSE 'MEDIUM' END::TEXT,
        'Signal type ' || signal_type || ' dominance: ' || ROUND(percentage, 1) || '%',
        percentage
    FROM signal_stats 
    WHERE percentage > 80;
    
    -- Strategy dominance
    RETURN QUERY
    WITH strategy_stats AS (
        SELECT 
            strategy,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
        FROM alert_history 
        WHERE alert_timestamp >= NOW() - INTERVAL '1 hour' * hours_back
        GROUP BY strategy
    )
    SELECT 
        'strategy_dominance'::TEXT,
        CASE WHEN percentage > 95 THEN 'HIGH' ELSE 'MEDIUM' END::TEXT,
        'Strategy ' || strategy || ' dominance: ' || ROUND(percentage, 1) || '%',
        percentage
    FROM strategy_stats 
    WHERE percentage > 90;
    
END;
$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON alert_analysis TO your_user;
-- GRANT EXECUTE ON FUNCTION detect_alert_anomalies TO your_user;

COMMENT ON VIEW alert_analysis IS 'Aggregated view for alert pattern analysis';
COMMENT ON FUNCTION detect_alert_anomalies IS 'Detects anomalies in alert patterns over specified time period';
"""
    
    print("Save this SQL to a file and run it against your database:")
    print()
    print(migration_sql)
    
    return migration_sql

def main():
    """Main diagnostic function"""
    print("🚀 STARTING COMPREHENSIVE ALERT SYSTEM ANALYSIS")
    print("=" * 55)
    
    # Run main analysis
    analysis_success = analyze_current_alerts()
    
    if analysis_success:
        print("\n" + "=" * 55)
        
        # Test enhanced system
        test_success = test_enhanced_alert_system()
        
        # Generate migration script
        generate_migration_script()
        
        print("\n✅ DIAGNOSTIC ANALYSIS COMPLETE")
        print("\n📋 NEXT STEPS:")
        print("1. Review the detected issues and recommendations above")
        print("2. Update your config.py with the suggested settings")
        print("3. Run the migration SQL script to enhance your database")
        print("4. Deploy the enhanced alert system code")
        print("5. Monitor alert patterns using the new analysis tools")
        print("\n🎯 KEY IMPROVEMENTS:")
        print("• Proper timestamp separation (market vs scan vs alert time)")
        print("• Signal deduplication and cooldown periods")
        print("• Enhanced metadata with market context")
        print("• Anomaly detection and pattern analysis")
        print("• Better strategy calibration recommendations")
        
    else:
        print("\n❌ DIAGNOSTIC ANALYSIS FAILED")
        print("Please check your database connection and configuration.")

if __name__ == "__main__":
    main()