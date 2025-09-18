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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_log_parser():
    """Get cached log parser instance"""
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
        status_text = "DETECTED"
        card_class = "signal-detected"

        details = f"**Epic:** {activity['epic']} | **Type:** {activity.get('signal_type', 'N/A')}"
        if activity.get('confidence'):
            details += f" | **Confidence:** {format_confidence(activity['confidence'])}"

    else:  # signal_rejected
        icon = "🚫"
        status_text = "REJECTED"
        card_class = "signal-rejected"

        details = f"**Epic:** {activity['epic']}"
        if activity.get('reason'):
            details += f" | **Reason:** {activity['reason'][:50]}..."

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
        parser = get_log_parser()
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
            st.rerun()

    # Get data
    try:
        signal_data = parser.get_recent_signal_data(hours_back=time_range_hours)
        health_data = parser.get_system_health(hours_back=time_range_hours)
        recent_activity = parser.get_recent_activity(hours_back=2, max_entries=20)
        alerts = check_alerts(signal_data, health_data)
        quick_stats = get_quick_stats(parser, time_range_hours)
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

    # System Health Row
    st.subheader("🔧 System Health")
    col1, col2, col3, col4 = st.columns(4)

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
            signal_type_filter = st.selectbox("📊 Signal Type", ["All", "Detected", "Rejected"])

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

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(5)
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

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(10)
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
    st.sidebar.info("Dashboard auto-refreshes every 5 seconds when enabled")

    st.sidebar.markdown("### 📊 Data Sources")
    st.sidebar.info("""
    - **Forex Scanner Logs**
    - **Stream Service Logs**
    - **Real-time Signal Feed**
    """)

    st.sidebar.markdown("### 🎯 Features")
    st.sidebar.success("""
    ✅ Real-time monitoring
    ✅ Signal intelligence
    ✅ System health tracking
    ✅ Lightweight & fast
    """)

    # Route to appropriate page
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🔧 System Health":
        render_health_details()
    elif page == "📈 Analytics":
        render_analytics_dashboard()
    elif page == "🔍 Search Logs":
        st.info("🔗 **Search Page Available** - Navigate to the Search page from the main Streamlit sidebar for advanced log exploration.")

if __name__ == "__main__":
    main()