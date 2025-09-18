"""
Simple Log Intelligence Interface - Lightweight version without external dependencies
Real-time signal intelligence and system health monitoring
"""

import streamlit as st
import time
from datetime import datetime, timedelta
import sys
import os
import re
from collections import defaultdict, Counter
import json

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

try:
    from simple_log_intelligence import SimpleLogParser
except ImportError as e:
    st.error(f"Failed to import simple log intelligence: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Signal Intelligence Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4037 0%, #99f2c8 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }

    .signal-card {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .status-healthy { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-critical { color: #dc3545; font-weight: bold; }

    .signal-detected { border-left: 4px solid #28a745; }
    .signal-rejected { border-left: 4px solid #dc3545; }

    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .alert-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .filter-section {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .quick-stat {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 0.8rem;
        border-radius: 6px;
        color: white;
        text-align: center;
        margin: 0.3rem;
        font-size: 0.9rem;
    }

    .analytics-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .metric-big {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }

    .epic-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }

    .performance-good { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); }
    .performance-average { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .performance-poor { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }

    .time-analysis {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #007bff;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .insight-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .search-result {
        background: #fff;
        padding: 1rem;
        border-left: 4px solid #007bff;
        border-radius: 4px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }

    .search-result-error {
        border-left-color: #dc3545;
        background: #fff5f5;
    }

    .search-result-warning {
        border-left-color: #ffc107;
        background: #fffcf0;
    }

    .search-result-signal {
        border-left-color: #28a745;
        background: #f0fff4;
    }

    .search-meta {
        color: #6c757d;
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .search-content {
        color: #333;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .highlight {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_log_parser():
    """Get cached log parser instance"""
    return SimpleLogParser()

def get_log_parser_fresh():
    """Get fresh log parser instance (for testing new methods)"""
    return SimpleLogParser()

def format_confidence(confidence):
    """Format confidence percentage"""
    if confidence is None or confidence == 0:
        return "N/A"
    return f"{confidence:.1%}"

def get_status_color(status):
    """Get color class for status"""
    status_map = {
        'healthy': 'status-healthy',
        'warning': 'status-warning',
        'critical': 'status-critical',
        'unknown': 'status-warning'
    }
    return status_map.get(status, 'status-warning')

def check_alerts(signal_data, health_data):
    """Check for alert conditions"""
    alerts = []

    # Check for critical system issues
    if health_data['overall_status'] == 'critical':
        alerts.append({
            'type': 'critical',
            'message': f"🚨 System Critical: {health_data['error_count_24h']} errors in 24h"
        })

    # Check for low signal activity
    if signal_data['total_signals'] < 10:  # Less than 10 signals in selected timeframe
        alerts.append({
            'type': 'warning',
            'message': f"⚠️ Low Signal Activity: Only {signal_data['total_signals']} signals detected"
        })

    # Check for low confidence
    if signal_data['avg_confidence'] < 0.8 and signal_data['total_signals'] > 0:
        alerts.append({
            'type': 'warning',
            'message': f"📉 Low Confidence: Average {signal_data['avg_confidence']:.1%}"
        })

    # Check for high rejection rate
    if signal_data['total_signals'] > 0:
        rejection_rate = signal_data['signals_rejected'] / signal_data['total_signals']
        if rejection_rate > 0.5:
            alerts.append({
                'type': 'warning',
                'message': f"🚫 High Rejection Rate: {rejection_rate:.1%} signals rejected"
            })

    # Check for scanner/stream health
    if health_data['forex_scanner_health'] == 'unknown':
        alerts.append({
            'type': 'warning',
            'message': "🤖 Scanner Status Unknown - Check scanner service"
        })

    if health_data['stream_health'] == 'unknown':
        alerts.append({
            'type': 'warning',
            'message': "📡 Stream Status Unknown - Check stream service"
        })

    return alerts

def get_quick_stats(parser, hours_back):
    """Get quick statistics for different timeframes"""
    stats = {}

    for period in [1, 6, 24]:
        if period <= hours_back:
            data = parser.get_recent_signal_data(hours_back=period)
            stats[f"{period}h"] = {
                'total': data['total_signals'],
                'detected': data['signals_detected'],
                'confidence': data['avg_confidence']
            }

    return stats

def filter_activities(activities, epic_filter=None, signal_type_filter=None, confidence_filter=None):
    """Filter activities based on criteria"""
    filtered = activities

    if epic_filter and epic_filter != "All":
        filtered = [a for a in filtered if a.get('epic') == epic_filter]

    if signal_type_filter and signal_type_filter != "All":
        if signal_type_filter == "Detected":
            filtered = [a for a in filtered if a['type'] == 'signal_detected']
        elif signal_type_filter == "Rejected":
            filtered = [a for a in filtered if a['type'] == 'signal_rejected']
        elif signal_type_filter == "Trade Events":
            filtered = [a for a in filtered if a['type'] in ['trade_opened', 'trade_monitoring', 'trade_adjustment']]
        elif signal_type_filter == "Trailing Events":
            trailing_types = [
                'trailing_breakeven_trigger', 'trailing_stage2_trigger', 'trailing_stage3_trigger',
                'trailing_breakeven_executed', 'trailing_stage2_executed', 'trailing_stage3_executed',
                'trailing_success', 'trailing_progressive_stage', 'trailing_percentage_calculation',
                'trailing_intelligent_calculation', 'trailing_config', 'trailing_safe_distance',
                'trailing_stage_config', 'trailing_api_details'
            ]
            filtered = [a for a in filtered if a['type'] in trailing_types]

    if confidence_filter:
        min_conf, max_conf = confidence_filter
        filtered = [a for a in filtered
                   if a.get('confidence') is not None and
                   min_conf <= a['confidence'] <= max_conf]

    return filtered

def analyze_epic_performance(parser, hours_back=24):
    """Analyze performance by currency pair"""
    epic_stats = defaultdict(lambda: {"detected": 0, "rejected": 0, "confidences": []})

    cutoff_time = datetime.now() - timedelta(hours=hours_back)

    for log_file in parser.log_files['forex_scanner']:
        if parser.base_log_dir == "":
            file_path = log_file
        else:
            file_path = os.path.join(parser.base_log_dir, log_file)

        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if not timestamp_match:
                        continue

                    try:
                        log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        if log_time < cutoff_time:
                            continue
                    except ValueError:
                        continue

                    epic_match = re.search(r'CS\.D\.([A-Z]{6})\.MINI\.IP', line)
                    if epic_match:
                        epic = epic_match.group(1)

                        if re.search(r'📊.*CS\.D\.[A-Z]{6}\.MINI\.IP.*(BULL|BEAR)', line):
                            epic_stats[epic]["detected"] += 1
                            conf_match = re.search(r'\((\d+\.?\d*)%\)', line)
                            if conf_match:
                                epic_stats[epic]["confidences"].append(float(conf_match.group(1)))
                        elif re.search(r'🚫.*REJECTED.*CS\.D\.[A-Z]{6}\.MINI\.IP', line):
                            epic_stats[epic]["rejected"] += 1

        except Exception:
            continue

    return epic_stats

def analyze_hourly_patterns(parser, hours_back=48):
    """Analyze signal patterns by hour of day"""
    hourly_data = defaultdict(lambda: {"detected": 0, "rejected": 0})

    cutoff_time = datetime.now() - timedelta(hours=hours_back)

    for log_file in parser.log_files['forex_scanner']:
        if parser.base_log_dir == "":
            file_path = log_file
        else:
            file_path = os.path.join(parser.base_log_dir, log_file)

        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if not timestamp_match:
                        continue

                    try:
                        log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        if log_time < cutoff_time:
                            continue
                        hour = log_time.hour
                    except ValueError:
                        continue

                    if re.search(r'📊.*CS\.D\.[A-Z]{6}\.MINI\.IP.*(BULL|BEAR)', line):
                        hourly_data[hour]["detected"] += 1
                    elif re.search(r'🚫.*REJECTED.*CS\.D\.[A-Z]{6}\.MINI\.IP', line):
                        hourly_data[hour]["rejected"] += 1

        except Exception:
            continue

    return hourly_data

def get_performance_insights(epic_stats, hourly_data, signal_data):
    """Generate performance insights"""
    insights = []

    if epic_stats:
        best_epic = max(epic_stats.items(), key=lambda x: x[1]["detected"])
        worst_epic = min([item for item in epic_stats.items() if item[1]["detected"] > 0],
                        key=lambda x: x[1]["detected"] / max(1, x[1]["detected"] + x[1]["rejected"]))

        insights.append(f"🏆 **Best Performer**: {best_epic[0]} with {best_epic[1]['detected']} signals")
        insights.append(f"⚠️ **Needs Attention**: {worst_epic[0]} with low success rate")

    if hourly_data:
        peak_hour = max(hourly_data.items(), key=lambda x: x[1]["detected"])
        insights.append(f"⏰ **Peak Activity**: {peak_hour[0]:02d}:00 with {peak_hour[1]['detected']} signals")

    if signal_data['avg_confidence'] > 0.95:
        insights.append("🎯 **High Confidence**: Average confidence above 95%")
    elif signal_data['avg_confidence'] < 0.80:
        insights.append("⚠️ **Low Confidence**: Consider reviewing strategy parameters")

    return insights

def render_signal_activity_card(activity):
    """Render a signal activity card"""
    if activity['type'] == 'signal_detected':
        icon = "🚀"
        status_text = "SIGNAL DETECTED"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']} | **Type:** {activity.get('signal_type', 'N/A')}"
        if activity.get('confidence'):
            details += f" | **Confidence:** {format_confidence(activity['confidence'])}"

    elif activity['type'] == 'signal_rejected':
        icon = "🚫"
        status_text = "SIGNAL REJECTED"
        card_class = "signal-rejected"

        details = f"**Epic:** {activity['epic']}"
        if activity.get('reason'):
            details += f" | **Reason:** {activity['reason'][:50]}..."

    elif activity['type'] == 'trade_opened':
        icon = "💰"
        status_text = "TRADE OPENED"
        card_class = "signal-detected"  # Use green styling

        details = f"**Epic:** {activity['epic']} | **Direction:** {activity.get('direction', 'N/A')}"
        if activity.get('entry_price'):
            details += f" | **Entry Price:** {activity['entry_price']:.5f}"
        if activity.get('deal_reference'):
            details += f"<br>**Deal Ref:** {activity['deal_reference']}"

    elif activity['type'] == 'trade_monitoring':
        icon = "📊"
        status_text = f"TRADE MONITORING #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-card"  # Use neutral styling

        details = f"**Epic:** {activity['epic']} | **Direction:** {activity.get('direction', 'N/A')}"

        # Enhanced profit display with rich information
        if activity.get('profit_pts') is not None:
            profit_pts = activity['profit_pts']
            profit_color = "green" if profit_pts > 0 else "red" if profit_pts < 0 else "gray"
            details += f" | **Profit:** <span style='color:{profit_color}'>{profit_pts}pts</span>"

        # Add entry and current prices
        if activity.get('entry_price') and activity.get('current_price'):
            details += f"<br>**Entry:** {activity['entry_price']:.5f} | **Current:** {activity['current_price']:.5f}"

        # Add trigger progress
        if activity.get('trigger_pts') and activity.get('progress_to_trigger'):
            details += f" | **Trigger:** {activity['trigger_pts']}pts ({activity['progress_to_trigger']}%)"

        # Add percentage move if available
        if activity.get('price_move_pct') is not None:
            move_color = "green" if activity['price_move_pct'] > 0 else "red" if activity['price_move_pct'] < 0 else "gray"
            details += f"<br>**Price Move:** <span style='color:{move_color}'>{activity['price_move_pct']:+.4f}%</span>"

    elif activity['type'] == 'trade_adjustment':
        icon = "🔧"
        status_text = "TRADE ADJUSTMENT"
        card_class = "signal-card"  # Use neutral styling

        details = f"**Epic:** {activity['epic']} | **Direction:** {activity.get('direction', 'N/A')}"
        if activity.get('adjustment_type'):
            details += f" | **Type:** {activity['adjustment_type'].replace('_', ' ').title()}"

        # Add stop level changes
        if activity.get('old_stop_level') and activity.get('new_stop_level'):
            old_stop = activity['old_stop_level']
            new_stop = activity['new_stop_level']
            change = new_stop - old_stop
            change_color = "green" if change > 0 else "red" if change < 0 else "gray"
            details += f"<br>**Stop:** {old_stop:.5f} → {new_stop:.5f} "
            details += f"(<span style='color:{change_color}'>{change:+.5f}</span>)"

        # Add limit level changes
        if activity.get('old_limit_level') and activity.get('new_limit_level'):
            old_limit = activity['old_limit_level']
            new_limit = activity['new_limit_level']
            change = new_limit - old_limit
            change_color = "green" if change > 0 else "red" if change < 0 else "gray"
            details += f"<br>**Limit:** {old_limit:.5f} → {new_limit:.5f} "
            details += f"(<span style='color:{change_color}'>{change:+.5f}</span>)"

    # Trailing Events
    elif activity['type'] == 'trailing_breakeven_trigger':
        icon = "🎯"
        status_text = f"BREAK-EVEN TRIGGER - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']} | **Stage:** Stage 1 (Break-even)"
        if activity.get('profit_pts') and activity.get('trigger_pts'):
            details += f"<br>**Profit:** <span style='color:green'>{activity['profit_pts']}pts</span> | **Trigger:** {activity['trigger_pts']}pts"

    elif activity['type'] == 'trailing_stage2_trigger':
        icon = "💰"
        status_text = f"STAGE 2 TRIGGER - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']} | **Stage:** Stage 2 (Profit Lock)"
        if activity.get('profit_pts') and activity.get('trigger_pts'):
            details += f"<br>**Profit:** <span style='color:green'>{activity['profit_pts']}pts</span> | **Trigger:** {activity['trigger_pts']}pts"

    elif activity['type'] == 'trailing_stage3_trigger':
        icon = "🚀"
        status_text = f"STAGE 3 TRIGGER - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']} | **Stage:** Stage 3 (Percentage Trailing)"
        if activity.get('profit_pts') and activity.get('trigger_pts'):
            details += f"<br>**Profit:** <span style='color:green'>{activity['profit_pts']}pts</span> | **Trigger:** {activity['trigger_pts']}pts"

    elif activity['type'] == 'trailing_breakeven_executed':
        icon = "🎉"
        status_text = f"BREAK-EVEN EXECUTED - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']} | **Stage:** Stage 1 Complete"
        if activity.get('new_stop_level'):
            details += f"<br>**New Stop:** {activity['new_stop_level']:.5f} (Break-even + 1pt)"

    elif activity['type'] == 'trailing_stage2_executed':
        icon = "💎"
        status_text = f"PROFIT LOCKED - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']} | **Stage:** Stage 2 Complete"
        if activity.get('new_stop_level') and activity.get('lock_points'):
            details += f"<br>**New Stop:** {activity['new_stop_level']:.5f} (+{activity['lock_points']}pts locked)"

    elif activity['type'] == 'trailing_stage3_executed':
        icon = "🎯"
        status_text = f"PERCENTAGE TRAILING - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']} | **Stage:** Stage 3 Active"
        if activity.get('new_stop_level'):
            details += f"<br>**New Stop:** {activity['new_stop_level']:.5f} (Percentage-based)"

    elif activity['type'] == 'trailing_success':
        icon = "🎯"
        status_text = f"TRAILING SUCCESS - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']}"
        if activity.get('new_stop_level') and activity.get('adjustment_pts'):
            details += f"<br>**New Stop:** {activity['new_stop_level']:.5f} ({activity['adjustment_pts']}pts adjustment)"

    elif activity['type'] == 'trailing_progressive_stage':
        icon = "⚡"
        status_text = f"PROGRESSIVE STAGE {activity.get('stage_number', '?')} - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-card"

        details = f"**Epic:** {activity['epic']} | **Stage:** Progressive Stage {activity.get('stage_number', '?')}"
        if activity.get('profit_pts') and activity.get('trail_level'):
            details += f"<br>**Profit:** <span style='color:green'>{activity['profit_pts']}pts</span> | **Trail Level:** {activity['trail_level']:.5f}"

    elif activity['type'] == 'trailing_percentage_calculation':
        icon = "📊"
        status_text = f"PERCENTAGE CALC - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-card"

        details = f"**Epic:** {activity['epic']}"
        if activity.get('profit_pts') and activity.get('retracement_percentage'):
            details += f"<br>**Profit:** <span style='color:green'>{activity['profit_pts']:.1f}pts</span> | **Retracement:** {activity['retracement_percentage']}%"
        if activity.get('trail_distance_pts'):
            details += f" | **Trail Distance:** {activity['trail_distance_pts']:.1f}pts"

    elif activity['type'] == 'trailing_intelligent_calculation':
        icon = "🧠"
        status_text = "INTELLIGENT TRAIL CALC"
        card_class = "signal-card"

        details = f"**Epic:** {activity['epic']} | **Direction:** {activity.get('direction', 'N/A')}"
        if activity.get('current_price') and activity.get('calculated_trail_level'):
            details += f"<br>**Current:** {activity['current_price']:.5f} | **Calculated Trail:** {activity['calculated_trail_level']:.5f}"
        if activity.get('distance_from_current_pts'):
            details += f" | **Distance:** {activity['distance_from_current_pts']:.1f}pts"

    elif activity['type'] == 'trailing_config':
        icon = "⚙️"
        status_text = "TRAILING CONFIG"
        card_class = "signal-card"

        details = f"**Epic:** {activity['epic']} | **Stage:** Configuration Setup"

    elif activity['type'] == 'trailing_safe_distance':
        icon = "📏"
        status_text = f"SAFE DISTANCE CALC - Trade #{activity.get('trade_id', 'N/A')}"
        card_class = "signal-card"

        details = f"**Epic:** {activity['epic']}"
        if activity.get('safe_distance_pts'):
            details += f"<br>**Safe Distance:** {activity['safe_distance_pts']:.1f}pts for trailing"

    elif activity['type'] == 'trailing_stage_config':
        icon = "🎛️"
        status_text = f"STAGE {activity.get('stage_number', '?')} CONFIG"
        card_class = "signal-card"

        details = f"**Epic:** {activity['epic']} | **Stage:** Stage {activity.get('stage_number', '?')} Configuration"

    elif activity['type'] == 'trailing_api_details':
        icon = "🔗"
        status_text = "TRAILING API DETAILS"
        card_class = "signal-card"

        details = f"**Epic:** {activity['epic']}"
        if activity.get('trailing_stop_distance'):
            details += f"<br>**Stop Distance:** {activity['trailing_stop_distance']:.1f}"
        if activity.get('trailing_step'):
            details += f" | **Step:** {activity['trailing_step']:.1f}"

    else:
        icon = "ℹ️"
        status_text = activity['type'].upper().replace('_', ' ')
        card_class = "signal-card"
        details = f"**Epic:** {activity.get('epic', 'N/A')}"

    st.markdown(f"""
    <div class="signal-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-weight: bold; font-size: 1.1rem;">
                {icon} {status_text} - {activity['epic']}
            </div>
            <div style="color: #6c757d; font-size: 0.9rem;">
                {activity['timestamp'].strftime("%H:%M:%S")}
            </div>
        </div>
        <div style="font-size: 0.85rem; margin-bottom: 0.5rem;">
            {details}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard():
    """Render main dashboard"""
    st.markdown('<div class="main-header">🧠 Signal Intelligence Dashboard</div>', unsafe_allow_html=True)

    # Initialize parser
    try:
        parser = get_log_parser_fresh()  # Use fresh parser to avoid cache issues
    except Exception as e:
        st.error(f"Failed to initialize log parser: {e}")
        return

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        time_range_hours = st.selectbox(
            "📅 Time Range",
            options=[1, 4, 12, 24, 48],
            format_func=lambda x: f"Last {x} hour{'s' if x > 1 else ''}",
            index=1  # Default to 4 hours
        )

    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)

    with col3:
        if st.button("🔄 Refresh Now"):
            # Clear cache and rerun
            st.cache_resource.clear()
            st.rerun()

    # Get data
    try:
        signal_data = parser.get_recent_signal_data(hours_back=time_range_hours)
        health_data = parser.get_system_health(hours_back=time_range_hours)
        recent_activity = parser.get_recent_activity(hours_back=2, max_entries=20)
        alerts = check_alerts(signal_data, health_data)
        quick_stats = get_quick_stats(parser, time_range_hours)

        # Get trade data with fallback
        try:
            trade_data = parser.get_recent_trade_events(hours_back=time_range_hours)
        except Exception as trade_error:
            st.warning(f"Trade data unavailable: {trade_error}")
            trade_data = {
                'trade_opened': 0,
                'trade_closed': 0,
                'trade_monitoring': 0,
                'trade_adjustments': 0,
                'active_trades': 0,
                'active_trade_pairs': []
            }

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Display alerts
    if alerts:
        st.subheader("🚨 System Alerts")
        for alert in alerts:
            if alert['type'] == 'critical':
                st.markdown(f'<div class="alert-box">🚨 {alert["message"]}</div>', unsafe_allow_html=True)
            else:
                st.warning(f'{alert["message"]}')
    else:
        st.markdown('<div class="success-box">✅ All Systems Operating Normally</div>', unsafe_allow_html=True)

    # Key Metrics Row
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "🎯 Total Signals",
            signal_data['total_signals']
        )

    with col2:
        st.metric(
            "✅ Detected",
            signal_data['signals_detected']
        )

    with col3:
        st.metric(
            "🚫 Rejected",
            signal_data['signals_rejected']
        )

    with col4:
        st.metric(
            "📈 Avg Confidence",
            format_confidence(signal_data['avg_confidence'])
        )

    with col5:
        st.metric(
            "🏆 Success Est.",
            format_confidence(signal_data['success_rate'])
        )

    # Trade Activity Row
    st.subheader("💼 Trade Activity")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "🚀 Trades Opened",
            trade_data['trade_opened']
        )

    with col2:
        st.metric(
            "📊 Trade Monitoring",
            trade_data['trade_monitoring']
        )

    with col3:
        st.metric(
            "🔧 Adjustments",
            trade_data['trade_adjustments']
        )

    with col4:
        st.metric(
            "📈 Active Trades",
            trade_data['active_trades']
        )

    with col5:
        st.metric(
            "💰 Trade Closed",
            trade_data['trade_closed']
        )

    # System Health Row
    st.subheader("🔧 System Health")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        status_class = get_status_color(health_data['overall_status'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="{status_class}" style="font-size: 1.2rem;">
                Overall: {health_data['overall_status'].title()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        scanner_class = get_status_color(health_data['forex_scanner_health'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="{scanner_class}">
                Scanner: {health_data['forex_scanner_health'].title()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        stream_class = get_status_color(health_data['stream_health'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="{stream_class}">
                Stream: {health_data['stream_health'].title()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        trade_monitor_class = get_status_color(health_data['trade_monitor_health'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="{trade_monitor_class}">
                Trade Monitor: {health_data['trade_monitor_health'].title()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        fastapi_class = get_status_color(health_data['fastapi_health'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="{fastapi_class}">
                FastAPI: {health_data['fastapi_health'].title()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.metric(
            "❌ Errors (24h)",
            health_data['error_count_24h']
        )

    # Quick Stats Comparison
    if quick_stats:
        st.subheader("⚡ Quick Stats Comparison")
        stats_cols = st.columns(len(quick_stats))

        for idx, (period, stats) in enumerate(quick_stats.items()):
            with stats_cols[idx]:
                st.markdown(f"""
                <div class="quick-stat">
                    <strong>{period}</strong><br>
                    📊 {stats['total']} signals<br>
                    ✅ {stats['detected']} detected<br>
                    🎯 {stats['confidence']:.1%} conf
                </div>
                """, unsafe_allow_html=True)

    # Additional stats
    if signal_data['top_epic']:
        st.info(f"🏆 **Most Active Pair:** {signal_data['top_epic']} | **Active Pairs:** {signal_data['active_pairs']}")

    # Active trade information
    if trade_data['active_trade_pairs']:
        st.info(f"💼 **Active Trade Pairs:** {', '.join(trade_data['active_trade_pairs'])} | **Total Active:** {trade_data['active_trades']}")

    # Real-time Activity Feed with Filters
    st.subheader("🚀 Recent Signal Activity")

    if recent_activity:
        # Activity filters
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.write("**🔧 Activity Filters**")

        filter_cols = st.columns(4)

        with filter_cols[0]:
            # Get unique epics from activities
            unique_epics = list(set([a.get('epic', 'Unknown') for a in recent_activity if a.get('epic')]))
            epic_filter = st.selectbox("🌍 Currency Pair", ["All"] + sorted(unique_epics))

        with filter_cols[1]:
            signal_type_filter = st.selectbox("📊 Event Type", ["All", "Detected", "Rejected", "Trade Events", "Trailing Events"])

        with filter_cols[2]:
            confidence_range = st.slider("🎯 Confidence Range", 0.0, 1.0, (0.0, 1.0), step=0.05)

        with filter_cols[3]:
            max_activities = st.slider("📋 Max Results", 5, 50, 15, step=5)

        st.markdown('</div>', unsafe_allow_html=True)

        # Apply filters
        filtered_activities = filter_activities(
            recent_activity,
            epic_filter if epic_filter != "All" else None,
            signal_type_filter,
            confidence_range if confidence_range != (0.0, 1.0) else None
        )

        # Limit results
        filtered_activities = filtered_activities[:max_activities]

        if filtered_activities:
            st.write(f"**Showing {len(filtered_activities)} of {len(recent_activity)} activities**")

            for activity in filtered_activities:
                render_signal_activity_card(activity)
        else:
            st.info("No activities match the selected filters. Try adjusting the filter criteria.")
    else:
        st.info("No recent signal activity found. System is monitoring...")

    # Debug info (optional)
    if st.checkbox("🔍 Show Debug Info"):
        st.subheader("Debug Information")

        col1, col2 = st.columns(2)

        with col1:
            st.json({
                "signal_data": signal_data,
                "recent_activities": len(recent_activity)
            })

        with col2:
            st.json({
                "health_data": health_data
            })

    # Auto-refresh logic - much less frequent
    if auto_refresh:
        time.sleep(30)  # Refresh every 30 seconds instead of 5
        st.rerun()

def render_health_details():
    """Render detailed health monitoring"""
    st.markdown('<div class="main-header">🔧 System Health Monitor</div>', unsafe_allow_html=True)

    try:
        parser = get_log_parser()
        health = parser.get_system_health(hours_back=24)
    except Exception as e:
        st.error(f"Error fetching health data: {e}")
        return

    # Overall Status
    st.subheader("🎯 Overall System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        status_class = get_status_color(health['overall_status'])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top: 0;">Overall Status</h3>
            <div class="{status_class}" style="font-size: 1.5rem;">
                {health['overall_status'].upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("❌ Errors (24h)", health['error_count_24h'])
        st.metric("⚠️ Warnings (24h)", health['warning_count_24h'])

    with col3:
        st.metric("🤖 Scanner Indicators", health['scanner_indicators'])
        st.metric("📡 Stream Indicators", health['stream_indicators'])

    # Service Status
    st.subheader("🔧 Service Status")

    services = [
        ("🤖 Forex Scanner", health['forex_scanner_health']),
        ("📡 Stream Service", health['stream_health'])
    ]

    for service_name, status in services:
        status_class = get_status_color(status)
        st.markdown(f'<div class="{status_class}">• {service_name}: {status.title()}</div>', unsafe_allow_html=True)

    # Recent Issues
    st.subheader("🚨 Recent Issues")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Last Error:**")
        if health['last_error']:
            st.error(f"""
            **Time:** {health['last_error']['time']}
            **Message:** {health['last_error']['message'][:100]}...
            """)
        else:
            st.success("No recent errors! 🎉")

    with col2:
        st.write("**Last Warning:**")
        if health['last_warning']:
            st.warning(f"""
            **Time:** {health['last_warning']['time']}
            **Message:** {health['last_warning']['message'][:100]}...
            """)
        else:
            st.success("No recent warnings! 👍")

def search_logs(parser, search_term, log_types, start_date, end_date, regex_mode=False, case_sensitive=False, max_results=500):
    """Search through log files with advanced filtering"""
    results = []

    # Prepare search pattern
    if regex_mode:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(search_term, flags)
        except re.error as e:
            st.error(f"Invalid regex pattern: {e}")
            return []
    else:
        if case_sensitive:
            search_func = lambda text: search_term in text
        else:
            search_func = lambda text: search_term.lower() in text.lower()

    # Define log file mappings
    log_files_to_search = []
    if 'forex_scanner' in log_types:
        log_files_to_search.extend(parser.log_files['forex_scanner'])
    if 'stream_service' in log_types:
        log_files_to_search.extend(parser.log_files['stream_service'])
    if 'trade_monitor' in log_types:
        log_files_to_search.extend(parser.log_files['trade_monitor'])
    if 'fastapi_dev' in log_types:
        log_files_to_search.extend(parser.log_files.get('fastapi_dev', []))
    if 'dev_trade' in log_types:
        log_files_to_search.extend(parser.log_files.get('dev_trade', []))

    for log_file in log_files_to_search:
        if parser.base_log_dir == "":
            file_path = log_file
        else:
            file_path = os.path.join(parser.base_log_dir, log_file)

        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_number = 0
                for line in f:
                    line_number += 1

                    # Parse timestamp for filtering
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')

                            # Filter by date range
                            if log_time.date() < start_date or log_time.date() > end_date:
                                continue
                        except ValueError:
                            continue
                    else:
                        continue

                    # Search in line
                    match_found = False
                    if regex_mode:
                        match = pattern.search(line)
                        if match:
                            match_found = True
                    else:
                        if search_func(line):
                            match_found = True

                    if match_found:
                        # Determine log type with more granularity
                        log_type = 'info'
                        if ' - ERROR - ' in line:
                            log_type = 'error'
                        elif ' - WARNING - ' in line:
                            log_type = 'warning'
                        elif '📊 [PROFIT]' in line:
                            log_type = 'trade_monitoring'
                        elif '✅ Trade logged' in line:
                            log_type = 'trade_opened'
                        elif '[ADJUST-STOP]' in line:
                            log_type = 'trade_adjustment'
                        elif '📊' in line or 'signal' in line.lower():
                            log_type = 'signal'
                        elif 'trade' in line.lower():
                            log_type = 'trade'

                        results.append({
                            'file': os.path.basename(file_path),
                            'line_number': line_number,
                            'timestamp': log_time if timestamp_match else None,
                            'content': line.strip(),
                            'log_type': log_type
                        })

                        if len(results) >= max_results:
                            break

        except Exception as e:
            st.warning(f"Error reading {file_path}: {e}")
            continue

        if len(results) >= max_results:
            break

    return results

def highlight_search_term(text, search_term, regex_mode=False, case_sensitive=False):
    """Highlight search term in text"""
    if not search_term:
        return text

    if regex_mode:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(f'({search_term})', flags)
            return pattern.sub(r'<span class="highlight">\1</span>', text)
        except re.error:
            return text
    else:
        if case_sensitive:
            return text.replace(search_term, f'<span class="highlight">{search_term}</span>')
        else:
            # Case insensitive replacement
            pattern = re.compile(re.escape(search_term), re.IGNORECASE)
            return pattern.sub(lambda m: f'<span class="highlight">{m.group()}</span>', text)

def render_search_interface():
    """Render search interface within logs page"""
    st.markdown('<div class="main-header">🔍 Advanced Log Search</div>', unsafe_allow_html=True)

    # Initialize parser
    try:
        parser = get_log_parser_fresh()
    except Exception as e:
        st.error(f"Failed to initialize log parser: {e}")
        return

    # Search Controls
    st.subheader("🎯 Search Configuration")

    col1, col2 = st.columns([3, 1])

    with col1:
        search_term = st.text_input(
            "🔍 Search Term",
            placeholder="Enter search term or regex pattern...",
            help="Enter text to search for, or enable regex mode for pattern matching"
        )

    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Advanced Filters
    st.markdown("**🔧 Advanced Filters**")

    filter_cols = st.columns(4)

    with filter_cols[0]:
        log_types = st.multiselect(
            "📁 Log Sources",
            options=['forex_scanner', 'stream_service', 'trade_monitor', 'fastapi_dev', 'dev_trade'],
            default=['forex_scanner', 'fastapi_dev'],
            help="Select which log sources to search"
        )

    with filter_cols[1]:
        regex_mode = st.checkbox("🔧 Regex Mode", help="Enable regular expression patterns")
        case_sensitive = st.checkbox("🔤 Case Sensitive", help="Case sensitive search")

    with filter_cols[2]:
        start_date = st.date_input(
            "📅 Start Date",
            value=datetime.now().date() - timedelta(days=1),
            help="Search from this date"
        )

    with filter_cols[3]:
        end_date = st.date_input(
            "📅 End Date",
            value=datetime.now().date(),
            help="Search until this date"
        )

    max_results = st.slider("📊 Max Results", min_value=50, max_value=500, value=100, step=25)

    # Quick Search Buttons
    st.write("**⚡ Quick Searches**")
    quick_cols = st.columns(8)

    quick_searches = [
        ("🚀 Signals", "📊.*CS\\.D\\.[A-Z]{6}\\.MINI\\.IP", True),
        ("❌ Errors", "ERROR", False),
        ("⚠️ Warnings", "WARNING", False),
        ("🚫 Rejected", "REJECTED", False),
        ("🎯 High Confidence", "\\(9[0-9]\\.[0-9]%\\)", True),
        ("💰 Trade Opened", "✅ Trade logged", False),
        ("📊 Trade Monitoring", "\\[PROFIT\\] Trade", True),
        ("🔧 Adjustments", "ADJUST-STOP", False)
    ]

    for idx, (label, term, is_regex) in enumerate(quick_searches):
        with quick_cols[idx]:
            if st.button(label, use_container_width=True, key=f"quick_{idx}"):
                st.session_state.search_term = term
                st.session_state.regex_mode = is_regex
                st.rerun()

    # Update search term from session state
    if 'search_term' in st.session_state:
        search_term = st.session_state.search_term
        regex_mode = st.session_state.get('regex_mode', False)

    # Perform search
    results = []
    if search_button and search_term:
        with st.spinner("🔍 Searching logs..."):
            results = search_logs(
                parser, search_term, log_types, start_date, end_date,
                regex_mode, case_sensitive, max_results
            )

        # Display search statistics
        if results:
            st.success(f"📊 Found {len(results)} matches for '{search_term}'")

            # Filter results by type
            type_filter = st.selectbox(
                "Filter by log type:",
                options=["all", "signal", "trade_monitoring", "trade_opened", "trade_adjustment", "trade", "error", "warning", "info"],
                index=0
            )

            if type_filter != "all":
                results = [r for r in results if r['log_type'] == type_filter]

            # Display results
            st.subheader(f"📋 Search Results ({len(results)} items)")

            for idx, result in enumerate(results):
                # Determine CSS class based on log type
                css_class_map = {
                    'error': 'search-result-error',
                    'warning': 'search-result-warning',
                    'signal': 'search-result-signal',
                    'trade': 'search-result-signal'
                }
                css_class = css_class_map.get(result['log_type'], 'search-result')

                # Highlight search term
                highlighted_content = highlight_search_term(
                    result['content'], search_term, regex_mode, case_sensitive
                )

                timestamp_str = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if result['timestamp'] else 'Unknown'

                st.markdown(f"""
                <div class="search-result {css_class}">
                    <div class="search-meta">
                        📁 {result['file']} | 📍 Line {result['line_number']} |
                        🕒 {timestamp_str} | 🏷️ {result['log_type'].upper()}
                    </div>
                    <div class="search-content">{highlighted_content}</div>
                </div>
                """, unsafe_allow_html=True)

                # Add separator every 10 results for better readability
                if (idx + 1) % 10 == 0 and idx + 1 < len(results):
                    st.markdown("---")

        else:
            st.info(f"No results found for '{search_term}' in the selected date range and log sources.")

    elif search_term and not search_button:
        st.info("👆 Click the Search button to start searching")

def render_analytics_dashboard():
    """Render analytics dashboard with deep insights"""
    st.markdown('<div class="analytics-header">📈 Advanced Signal Analytics</div>', unsafe_allow_html=True)

    # Initialize parser
    try:
        parser = get_log_parser()
    except Exception as e:
        st.error(f"Failed to initialize log parser: {e}")
        return

    # Time range selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analysis_hours = st.selectbox(
            "📊 Analysis Period",
            options=[6, 12, 24, 48, 72],
            format_func=lambda x: f"Last {x} hours",
            index=2  # Default to 24 hours
        )

    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)

    with col3:
        if st.button("🔄 Refresh Analytics"):
            st.rerun()

    # Get analytics data
    try:
        signal_data = parser.get_recent_signal_data(hours_back=analysis_hours)
        epic_stats = analyze_epic_performance(parser, hours_back=analysis_hours)
        hourly_data = analyze_hourly_patterns(parser, hours_back=min(48, analysis_hours * 2))
        insights = get_performance_insights(epic_stats, hourly_data, signal_data)
    except Exception as e:
        st.error(f"Error analyzing data: {e}")
        return

    # Key Performance Indicators
    st.subheader("🎯 Key Performance Indicators")

    kpi_cols = st.columns(4)

    with kpi_cols[0]:
        st.markdown(f"""
        <div class="metric-big performance-good">
            <div class="metric-value">{signal_data['signals_detected']}</div>
            <div class="metric-label">Signals Detected</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[1]:
        success_rate = signal_data['signals_detected'] / max(1, signal_data['total_signals'])
        perf_class = "performance-good" if success_rate > 0.8 else "performance-average" if success_rate > 0.6 else "performance-poor"
        st.markdown(f"""
        <div class="metric-big {perf_class}">
            <div class="metric-value">{success_rate:.1%}</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[2]:
        conf_class = "performance-good" if signal_data['avg_confidence'] > 0.9 else "performance-average"
        st.markdown(f"""
        <div class="metric-big {conf_class}">
            <div class="metric-value">{signal_data['avg_confidence']:.1%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[3]:
        st.markdown(f"""
        <div class="metric-big performance-good">
            <div class="metric-value">{signal_data['active_pairs']}</div>
            <div class="metric-label">Active Pairs</div>
        </div>
        """, unsafe_allow_html=True)

    # Epic Performance Analysis
    st.subheader("🌍 Currency Pair Performance")

    if epic_stats:
        epic_cols = st.columns(min(4, len(epic_stats)))

        for idx, (epic, stats) in enumerate(sorted(epic_stats.items(),
                                                  key=lambda x: x[1]["detected"], reverse=True)[:4]):
            with epic_cols[idx]:
                total_signals = stats["detected"] + stats["rejected"]
                success_rate = stats["detected"] / max(1, total_signals)
                avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0

                perf_class = "performance-good" if success_rate > 0.8 else "performance-average" if success_rate > 0.6 else "performance-poor"

                st.markdown(f"""
                <div class="epic-card {perf_class}">
                    <h3 style="margin: 0; font-size: 1.2rem;">{epic}</h3>
                    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
                        📊 {stats["detected"]} detected<br>
                        🚫 {stats["rejected"]} rejected<br>
                        🎯 {success_rate:.1%} success<br>
                        📈 {avg_conf:.1%} confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No currency pair data available for the selected period")

    # Hourly Activity Pattern
    st.subheader("⏰ Hourly Activity Patterns")

    if hourly_data:
        # Create hourly chart data
        hours = list(range(24))
        detected_counts = [hourly_data.get(h, {}).get("detected", 0) for h in hours]
        rejected_counts = [hourly_data.get(h, {}).get("rejected", 0) for h in hours]

        chart_data = {
            "Hour": [f"{h:02d}:00" for h in hours],
            "Detected": detected_counts,
            "Rejected": rejected_counts
        }

        st.bar_chart(chart_data, x="Hour", y=["Detected", "Rejected"])

        # Peak activity analysis
        peak_detected = max(detected_counts)
        peak_hour = detected_counts.index(peak_detected)

        st.markdown(f"""
        <div class="time-analysis">
            <strong>📊 Activity Analysis:</strong><br>
            • Peak activity: {peak_hour:02d}:00 with {peak_detected} signals<br>
            • Total active hours: {sum(1 for x in detected_counts if x > 0)}<br>
            • Average signals per active hour: {sum(detected_counts) / max(1, sum(1 for x in detected_counts if x > 0)):.1f}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No hourly pattern data available")

    # Performance Insights
    st.subheader("💡 Performance Insights")

    if insights:
        insights_text = "<br>".join([f"• {insight}" for insight in insights])
        st.markdown(f"""
        <div class="insight-box">
            <h4 style="margin-top: 0;">🔍 Key Insights:</h4>
            {insights_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Gathering insights... Check back with more data")

    # Detailed Epic Statistics Table
    if epic_stats:
        st.subheader("📋 Detailed Statistics")

        table_data = []
        for epic, stats in sorted(epic_stats.items(), key=lambda x: x[1]["detected"], reverse=True):
            total = stats["detected"] + stats["rejected"]
            success_rate = stats["detected"] / max(1, total)
            avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0

            table_data.append({
                "Epic": epic,
                "Detected": stats["detected"],
                "Rejected": stats["rejected"],
                "Total": total,
                "Success Rate": f"{success_rate:.1%}",
                "Avg Confidence": f"{avg_conf:.1%}" if avg_conf > 0 else "N/A",
                "Max Confidence": f"{max(stats['confidences']):.1%}" if stats["confidences"] else "N/A"
            })

        st.dataframe(table_data, use_container_width=True)

    # Auto-refresh logic - much less frequent
    if auto_refresh:
        time.sleep(60)  # Refresh every 60 seconds for analytics
        st.rerun()

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("🧠 Signal Intelligence")

    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "🔧 System Health", "📈 Analytics", "🔍 Search Logs"]
    )

    # Additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Auto-Refresh")
    st.sidebar.info("Dashboard auto-refreshes every 30 seconds when enabled (Analytics: 60s)")

    st.sidebar.markdown("### 📊 Data Sources")
    st.sidebar.info("""
    - **Forex Scanner Logs**
    - **Stream Service Logs**
    - **FastAPI Dev Logs**
    - **Trade Monitor Logs**
    - **Dev Trade Logs**
    """)

    st.sidebar.markdown("### 🎯 Features")
    st.sidebar.success("""
    ✅ Real-time monitoring
    ✅ Signal & trade intelligence
    ✅ System health tracking
    ✅ Advanced log search
    ✅ Analytics & insights
    """)

    # Route to appropriate page
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🔧 System Health":
        render_health_details()
    elif page == "📈 Analytics":
        render_analytics_dashboard()
    elif page == "🔍 Search Logs":
        render_search_interface()

if __name__ == "__main__":
    main()