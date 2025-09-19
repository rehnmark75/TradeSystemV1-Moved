"""
System Status Page - Monitor IG Streaming and Backfill Processes
Monitors the fastapi-stream service health, candle data quality, and stream status
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import sys
import os
import requests
from typing import Dict, Any, List

# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

try:
    from stream_monitor import StreamMonitor
    MONITOR_AVAILABLE = True
    # Initialize the monitor
    stream_monitor = StreamMonitor()
except ImportError as e:
    MONITOR_AVAILABLE = False
    stream_monitor = None
    st.error(f"Stream monitor service not available: {e}")
except Exception as e:
    MONITOR_AVAILABLE = False
    stream_monitor = None
    st.error(f"Error initializing stream monitor: {e}")

# Page configuration
st.set_page_config(
    page_title="System Status - IG Stream Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
STREAM_API_BASE = "http://fastapi-stream:8000"  # Docker service name with internal port
MAIN_API_BASE = "http://fastapi-dev:8000"       # Main API service with internal port
REFRESH_INTERVAL = 30  # seconds

def safe_api_call(url: str, timeout: int = 5) -> Dict[str, Any]:
    """Safely make API call with error handling"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Service unavailable", "status": "offline"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout", "status": "timeout"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}", "status": "error"}
    except Exception as e:
        return {"error": str(e), "status": "unknown"}

def get_backfill_status() -> Dict[str, Any]:
    """Get backfill service status"""
    return safe_api_call(f"{STREAM_API_BASE}/backfill/status")

def get_backfill_gaps() -> Dict[str, Any]:
    """Get current data gaps"""
    return safe_api_call(f"{STREAM_API_BASE}/backfill/gaps")

def get_stream_status() -> Dict[str, Any]:
    """Get streaming service status"""
    return safe_api_call(f"{STREAM_API_BASE}/stream/status")

def get_candle_health(epic: str, timeframe: int = 5) -> Dict[str, Any]:
    """Get recent candle data to check health"""
    url = f"{STREAM_API_BASE}/stream/candles/{epic}?timeframe={timeframe}"
    return safe_api_call(url)

def get_latest_candle(epic: str, timeframe: int = 5) -> Dict[str, Any]:
    """Get the latest candle for an epic"""
    url = f"{STREAM_API_BASE}/stream/candle/latest/{epic}?timeframe={timeframe}"
    return safe_api_call(url)

def get_system_health_summary() -> Dict[str, Any]:
    """Get comprehensive system health summary"""
    try:
        health_data = safe_api_call(f"{STREAM_API_BASE}/stream/system/summary")
        if "error" not in health_data:
            return health_data
        else:
            return {"status": "error", "error": health_data["error"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def render_service_status():
    """Render overall service status with real data"""
    st.header("🔧 Service Status Overview")
    
    # Get comprehensive system health
    system_health = get_system_health_summary()
    
    # Get status of individual services
    backfill_status = get_backfill_status()
    stream_status = get_stream_status()
    
    # Create status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "error" not in backfill_status:
            st.metric(
                "📈 Backfill Service",
                "Online",
                delta="✅ Healthy"
            )
        else:
            st.metric(
                "📈 Backfill Service", 
                "Offline",
                delta="❌ Error"
            )
    
    with col2:
        # Use system health data if available
        if "error" not in system_health:
            status = system_health.get("status", "unknown")
            if status == "healthy":
                st.metric(
                    "📡 Stream Service",
                    "Online", 
                    delta="✅ Healthy"
                )
            elif status == "issues":
                st.metric(
                    "📡 Stream Service",
                    "Issues Detected",
                    delta="⚠️ Warning"
                )
            else:
                st.metric(
                    "📡 Stream Service",
                    "Unknown",
                    delta="❓ Status Unknown"
                )
        elif "error" not in stream_status:
            st.metric(
                "📡 Stream Service",
                "Online", 
                delta="✅ Healthy"
            )
        else:
            st.metric(
                "📡 Stream Service",
                "Offline",
                delta="❌ Error"
            )
    
    with col3:
        # Use system health data for monitored pairs
        if "error" not in system_health:
            total_streams = system_health.get("total_streams", 0)
            market_open = system_health.get("market_open", True)
            
            delta_text = "Market Open" if market_open else "Market Closed"
            st.metric(
                "📊 Active Streams",
                str(total_streams),
                delta=delta_text
            )
        else:
            # Fallback to backfill status
            if "error" not in backfill_status:
                total_epics = backfill_status.get("monitored_epics", 
                             backfill_status.get("statistics", {}).get("total_epics_monitored", 
                             backfill_status.get("total_epics", "N/A")))
            else:
                total_epics = "N/A"
            
            st.metric(
                "📊 Monitored Pairs",
                total_epics if total_epics != "N/A" else "0",
                delta="Active" if total_epics != "N/A" and total_epics > 0 else None
            )
    
    with col4:
        # Show system activity summary
        if "error" not in system_health and "recent_activity" in system_health:
            activity = system_health["recent_activity"]
            errors = activity.get("errors_last_hour", 0)
            
            if errors == 0:
                st.metric(
                    "🕐 System Health",
                    "Healthy",
                    delta="✅ No Errors"
                )
            else:
                st.metric(
                    "🕐 System Health", 
                    f"{errors} Errors",
                    delta="⚠️ Issues"
                )
        else:
            # Last update time fallback
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric(
                "🕐 Last Update",
                current_time,
                delta="Live"
            )
    
    # Add a status summary bar if we have system health data
    if "error" not in system_health:
        health_indicators = system_health.get("health_indicators", {})
        stream_health = health_indicators.get("stream_health", "unknown")
        gap_status = health_indicators.get("gap_status", "unknown")
        
        # Create a status summary
        status_items = []
        if stream_health == "healthy":
            status_items.append("🟢 Streams Healthy")
        elif stream_health == "issues":
            status_items.append("🟡 Stream Issues")
        
        if gap_status == "no_gaps":
            status_items.append("🟢 No Data Gaps")
        elif gap_status == "gaps_found":
            status_items.append("🟡 Gaps Detected")
        
        if status_items:
            st.info(" | ".join(status_items))

def render_backfill_status():
    """Render detailed backfill status"""
    st.header("📈 Auto-Backfill Service Status")
    
    backfill_status = get_backfill_status()
    
    if "error" in backfill_status:
        st.error(f"❌ Backfill Service Error: {backfill_status['error']}")
        return
    
    # Display backfill statistics
    stats = backfill_status.get("statistics", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Service Statistics")
        if stats:
            for key, value in stats.items():
                if key.startswith("total_"):
                    display_key = key.replace("total_", "").replace("_", " ").title()
                    st.write(f"**{display_key}:** {value}")
    
    with col2:
        st.subheader("⏰ Timing Information")
        if "last_run" in backfill_status:
            st.write(f"**Last Run:** {backfill_status['last_run']}")
        if "next_run" in backfill_status:
            st.write(f"**Next Run:** {backfill_status['next_run']}")
        if "run_interval" in backfill_status:
            st.write(f"**Run Interval:** {backfill_status['run_interval']} minutes")
    
    # Show recent gaps
    st.subheader("🕳️ Current Data Gaps")
    
    if st.button("🔍 Check for Gaps Now"):
        with st.spinner("Checking for data gaps..."):
            gaps_data = get_backfill_gaps()
            
            if "error" in gaps_data:
                st.error(f"Error checking gaps: {gaps_data['error']}")
            else:
                gaps_report = gaps_data.get("report", {})
                gaps_stats = gaps_data.get("statistics", {})
                
                # Handle case where report is a string (no gaps) vs dict (gaps found)
                if isinstance(gaps_report, str):
                    st.success(f"✅ {gaps_report}")
                elif isinstance(gaps_report, dict) and gaps_report:
                    st.write("**Found Gaps:**")
                    for epic, gaps in gaps_report.items():
                        if gaps:
                            st.write(f"- **{epic}:** {len(gaps)} gaps")
                            for gap in gaps[:3]:  # Show first 3 gaps
                                st.write(f"  - {gap}")
                        else:
                            st.write(f"- **{epic}:** No gaps ✅")
                else:
                    st.success("✅ No data gaps found!")
                
                if gaps_stats:
                    st.write("**Gap Statistics:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Gaps", gaps_stats.get("total_gaps", 0))
                        st.metric("Missing Candles", gaps_stats.get("total_missing_candles", 0))
                    with col2:
                        st.metric("Recent Gaps", gaps_stats.get("recent_gaps", 0))
                        st.metric("Largest Gap (min)", gaps_stats.get("largest_gap_minutes", 0))

def render_streaming_status():
    """Render streaming service status"""
    st.header("📡 IG Streaming Service Status")
    
    stream_status = get_stream_status()
    
    if "error" in stream_status:
        st.error(f"❌ Stream Service Error: {stream_status['error']}")
        return
    
    # Display streaming status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔌 Connection Status")
        
        # Check if data is being received recently (better indicator than the status endpoint)
        if MONITOR_AVAILABLE and stream_monitor:
            db_health = stream_monitor.get_database_health()
            if not isinstance(db_health, dict) or "error" not in db_health:
                candle_stats = db_health.get("candle_stats", {})
                recent_candles = candle_stats.get("recent_candles", 0)
                
                if recent_candles > 0:
                    st.success(f"✅ Streaming is active ({recent_candles} candles in last hour)")
                else:
                    st.warning("⚠️ No recent streaming data")
        else:
            # Fallback to original status check
            if "running" in stream_status:
                if stream_status["running"]:
                    st.success("✅ Streaming is active")
                else:
                    st.warning("⚠️ Streaming is not active")
        
        if "last_status" in stream_status:
            st.write(f"**Last Status:** {stream_status['last_status']}")
        
        if "connected_epics" in stream_status:
            st.write(f"**Connected Epics:** {len(stream_status['connected_epics'])}")
    
    with col2:
        st.subheader("📊 Stream Statistics")
        
        # Show meaningful statistics from database activity
        if MONITOR_AVAILABLE and stream_monitor:
            db_health = stream_monitor.get_database_health()
            if not isinstance(db_health, dict) or "error" not in db_health:
                candle_stats = db_health.get("candle_stats", {})
                
                st.metric("Active Epics", candle_stats.get("unique_epics", 0))
                st.metric("Recent Updates", f"{candle_stats.get('recent_candles', 0)} candles/hour")
                
                newest = candle_stats.get("newest_candle")
                if newest:
                    if isinstance(newest, str):
                        newest_dt = datetime.fromisoformat(newest)
                    else:
                        newest_dt = newest
                    
                    age_seconds = (datetime.now() - newest_dt.replace(tzinfo=None)).total_seconds()
                    if age_seconds < 60:
                        st.write(f"**Last Update:** {int(age_seconds)} seconds ago")
                    else:
                        st.write(f"**Last Update:** {int(age_seconds/60)} minutes ago")
        else:
            # Fallback to original status display
            for key, value in stream_status.items():
                if key not in ["running", "last_status", "connected_epics", "error"]:
                    display_key = key.replace("_", " ").title()
                    st.write(f"**{display_key}:** {value}")

def render_candle_health():
    """Render candle data health check"""
    st.header("🕯️ Candle Data Health Check")
    
    # Major forex pairs to check (aligned with stream-app EPICS)
    major_pairs = [
        "CS.D.EURUSD.CEEM.IP",  # Updated to new EURUSD epic name
        "CS.D.GBPUSD.MINI.IP", 
        "CS.D.USDJPY.MINI.IP",
        "CS.D.AUDUSD.MINI.IP",
        "CS.D.USDCAD.MINI.IP",
        "CS.D.EURJPY.MINI.IP",
        "CS.D.AUDJPY.MINI.IP",  # Added missing pair
        "CS.D.NZDUSD.MINI.IP",
        "CS.D.USDCHF.MINI.IP"
    ]
    
    # Timeframe selection
    timeframe = st.selectbox("Select Timeframe (minutes):", [1, 5, 15, 30], index=1)
    
    # Create health check results
    health_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def extract_clean_epic_name(epic_name):
        """Extract clean epic name, handling different IG naming patterns"""
        clean_name = epic_name.replace("CS.D.", "")
        clean_name = clean_name.replace(".MINI.IP", "").replace(".CEEM.IP", "").replace(".CFE.IP", "")
        return clean_name
    
    for i, epic in enumerate(major_pairs):
        progress = (i + 1) / len(major_pairs)
        progress_bar.progress(progress)
        status_text.text(f"Checking {epic}...")
        
        # Get latest candle from database if monitor is available, otherwise use API
        if MONITOR_AVAILABLE and stream_monitor:
            db_health = stream_monitor.get_candle_health_from_db(epic, timeframe, hours_back=1)
            if "error" not in db_health and db_health.get("latest_candle"):
                latest_candle = db_health["latest_candle"]
            else:
                latest_candle = {"error": f"No database data: {db_health.get('error', 'No candles found')}"}
        else:
            # Fallback to API
            latest_candle = get_latest_candle(epic, timeframe)
        
        if "error" not in latest_candle:
            # Calculate health metrics
            candle_time = latest_candle.get("time")
            if candle_time:
                try:
                    # Handle both datetime objects and strings
                    if isinstance(candle_time, datetime):
                        candle_dt = candle_time
                    elif isinstance(candle_time, str):
                        candle_dt = datetime.fromisoformat(candle_time.replace('Z', '+00:00'))
                    else:
                        candle_dt = datetime.fromisoformat(str(candle_time).replace('Z', '+00:00'))
                    
                    # Remove timezone info for comparison
                    if hasattr(candle_dt, 'tzinfo') and candle_dt.tzinfo:
                        candle_dt = candle_dt.replace(tzinfo=None)
                    
                    age_minutes = (datetime.now() - candle_dt).total_seconds() / 60
                    
                    # Determine health status
                    if age_minutes <= timeframe * 2:  # Within 2 timeframe periods
                        health_status = "✅ Healthy"
                        health_color = "green"
                    elif age_minutes <= timeframe * 5:  # Within 5 timeframe periods
                        health_status = "⚠️ Stale"
                        health_color = "orange"
                    else:
                        health_status = "❌ Very Stale"
                        health_color = "red"
                    
                    # Ensure Price and Volume are strings for consistent DataFrame handling
                    price_value = latest_candle.get("close", "N/A")
                    volume_value = latest_candle.get("volume", "N/A")
                    
                    health_results.append({
                        "Epic": extract_clean_epic_name(epic),
                        "Latest Candle": candle_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "Age (minutes)": f"{age_minutes:.1f}",
                        "Status": health_status,
                        "Price": str(price_value) if price_value != "N/A" else "N/A",
                        "Volume": str(volume_value) if volume_value != "N/A" else "N/A"
                    })
                    
                except Exception as e:
                    # Log the error for debugging
                    st.error(f"Error parsing candle for {epic}: {str(e)}")
                    st.write(f"Candle data type: {type(candle_time)}, value: {candle_time}")
                    health_results.append({
                        "Epic": extract_clean_epic_name(epic),
                        "Latest Candle": "Parse Error",
                        "Age (minutes)": "N/A",
                        "Status": "❌ Error",
                        "Price": "N/A",
                        "Volume": "N/A"
                    })
        else:
            health_results.append({
                "Epic": extract_clean_epic_name(epic),
                "Latest Candle": "No Data",
                "Age (minutes)": "N/A", 
                "Status": "❌ Offline",
                "Price": "N/A",
                "Volume": "N/A"
            })
    
    progress_bar.progress(1.0)
    status_text.text("Health check complete!")
    
    # Display results
    if health_results:
        df = pd.DataFrame(health_results)
        st.dataframe(df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            healthy_count = len([r for r in health_results if "✅" in r["Status"]])
            st.metric("Healthy Streams", f"{healthy_count}/{len(health_results)}")
        
        with col2:
            stale_count = len([r for r in health_results if "⚠️" in r["Status"]])
            st.metric("Stale Streams", stale_count)
        
        with col3:
            error_count = len([r for r in health_results if "❌" in r["Status"]])
            st.metric("Error Streams", error_count)

def get_real_alerts() -> List[Dict[str, Any]]:
    """Get real alerts from the stream API"""
    try:
        alerts_data = safe_api_call(f"{STREAM_API_BASE}/stream/alerts/recent?hours_back=6")
        if "error" not in alerts_data:
            return alerts_data.get("alerts", [])
        else:
            st.warning(f"Could not fetch alerts: {alerts_data['error']}")
            return []
    except Exception as e:
        st.error(f"Error fetching alerts: {e}")
        return []

def get_real_operations() -> List[Dict[str, Any]]:
    """Get real operations from the stream API"""
    try:
        operations_data = safe_api_call(f"{STREAM_API_BASE}/stream/operations/recent?hours_back=6")
        if "error" not in operations_data:
            return operations_data.get("operations", [])
        else:
            st.warning(f"Could not fetch operations: {operations_data['error']}")
            return []
    except Exception as e:
        st.error(f"Error fetching operations: {e}")
        return []

def render_recent_activity():
    """Render recent activity and logs with real data"""
    st.header("📋 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Recent System Operations")
        
        # Get real operations data
        operations = get_real_operations()
        
        if operations:
            # Convert to DataFrame for display
            operations_df = pd.DataFrame(operations)
            
            # Rename columns for better display
            if not operations_df.empty:
                operations_df = operations_df.rename(columns={
                    'time': 'Time',
                    'epic': 'Epic', 
                    'action': 'Action',
                    'status': 'Status'
                })
                
                # Select only the columns we want to show
                display_columns = ['Time', 'Epic', 'Action', 'Status']
                display_df = operations_df[display_columns]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No recent operations found")
        else:
            st.warning("Could not load recent operations - showing sample data")
            # Fallback to sample data
            fallback_operations = [
                {"Time": "Loading...", "Epic": "SYSTEM", "Action": "Fetching data", "Status": "⏳"},
            ]
            fallback_df = pd.DataFrame(fallback_operations)
            st.dataframe(fallback_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("⚠️ Alerts & Warnings")
        
        # Get real alerts data
        alerts = get_real_alerts()
        
        if alerts:
            # Show the most recent alerts
            for alert in alerts[:8]:  # Show up to 8 recent alerts
                # Map severity levels to colors
                level_color = {
                    "INFO": "blue",
                    "WARNING": "orange", 
                    "ERROR": "red",
                    "CRITICAL": "darkred"
                }.get(alert.get("severity", "INFO"), "gray")
                
                # Format the alert message
                time_str = alert.get("time", "Unknown")
                severity = alert.get("severity", "INFO")
                message = alert.get("message", "No message")
                
                # Truncate long messages
                if len(message) > 80:
                    message = message[:77] + "..."
                
                st.markdown(f"""
                <div style="padding: 6px; margin: 3px 0; border-left: 3px solid {level_color}; background: #f8f9fa; font-size: 13px;">
                    <strong>{time_str}</strong> [{severity}] {message}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Could not load recent alerts - showing sample data")
            # Fallback to sample alerts
            fallback_alerts = [
                {"time": "Loading...", "severity": "INFO", "message": "Fetching real-time alerts..."},
            ]
            
            for alert in fallback_alerts:
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 4px solid gray; background: #f0f2f6;">
                    <strong>{alert['time']}</strong> [{alert['severity']}] {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        
        # Add refresh button for alerts
        if st.button("🔄 Refresh Alerts", key="refresh_alerts"):
            st.rerun()

def render_database_health():
    """Render database health section"""
    st.header("🗄️ Database Health")
    
    if not MONITOR_AVAILABLE or not stream_monitor:
        st.warning("⚠️ Stream monitor not available - using API-only mode")
        return
    
    db_health = stream_monitor.get_database_health()
    
    if "error" in db_health:
        st.error(f"❌ Database Error: {db_health['error']}")
        return
    
    # Database connection status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Database Status", "✅ Connected" if db_health.get("status") == "healthy" else "❌ Error")
    
    with col2:
        candle_stats = db_health.get("candle_stats", {})
        total_candles = candle_stats.get("total_candles", 0)
        st.metric("Total Candles", f"{total_candles:,}")
    
    with col3:
        unique_epics = candle_stats.get("unique_epics", 0)
        st.metric("Monitored Epics", unique_epics)
    
    # Table statistics
    if "table_stats" in db_health:
        st.subheader("📊 Table Statistics")
        table_df = pd.DataFrame(db_health["table_stats"])
        if not table_df.empty:
            st.dataframe(table_df, use_container_width=True)
    
    # Recent activity metrics
    if candle_stats:
        st.subheader("📈 Recent Activity")
        col1, col2 = st.columns(2)
        
        with col1:
            recent_candles = candle_stats.get("recent_candles", 0)
            st.metric("Candles (Last Hour)", recent_candles)
        
        with col2:
            oldest_candle = candle_stats.get("oldest_candle")
            if oldest_candle:
                if isinstance(oldest_candle, str):
                    oldest_dt = datetime.fromisoformat(oldest_candle)
                else:
                    oldest_dt = oldest_candle
                days_of_data = (datetime.now() - oldest_dt).days
                st.metric("Data History", f"{days_of_data} days")

def main():
    """Main application"""
    st.title("📡 System Status - IG Stream Monitor")
    st.markdown("*Real-time monitoring of IG streaming and backfill processes*")
    
    # Add real-time status indicator
    system_health = get_system_health_summary()
    if "error" not in system_health:
        overall_status = system_health.get("status", "unknown")
        if overall_status == "healthy":
            st.success("🟢 System Status: All services operational")
        elif overall_status == "issues":
            st.warning("🟡 System Status: Some issues detected")
        else:
            st.info("🔵 System Status: Monitoring active")
        
        # Show last update time
        last_update = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Last updated: {last_update} | Data refreshes automatically every 30 seconds")
    else:
        st.error("🔴 System Status: Cannot connect to monitoring services")
        st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")
    
    # Monitor status
    if MONITOR_AVAILABLE and stream_monitor:
        st.session_state.monitor = stream_monitor
    elif MONITOR_AVAILABLE:
        st.error("❌ Failed to initialize stream monitor")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        if 'monitor' in st.session_state:
            # Clear the monitor to reinitialize
            del st.session_state.monitor
        st.rerun()
    
    # Monitoring options
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Monitoring Options")
    
    show_db_health = st.sidebar.checkbox("Show Database Health", value=True)
    show_api_health = st.sidebar.checkbox("Show API Health", value=True)
    show_detailed_gaps = st.sidebar.checkbox("Show Detailed Gap Analysis", value=False)
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox("Candle Timeframe (min):", [1, 5, 15, 30], index=1)
    
    # Last updated timestamp
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.markdown(f"**Monitor Available:** {'✅ Yes' if MONITOR_AVAILABLE else '❌ No'}")
    
    # Main content
    render_service_status()
    st.markdown("---")
    
    if show_db_health and MONITOR_AVAILABLE:
        render_database_health()
        st.markdown("---")
    
    render_backfill_status()
    st.markdown("---")
    
    render_streaming_status()
    st.markdown("---")
    
    render_candle_health()
    st.markdown("---")
    
    if show_detailed_gaps and MONITOR_AVAILABLE:
        render_detailed_gap_analysis(timeframe)
        st.markdown("---")
    
    render_recent_activity()
    
    # Auto-refresh logic
    if auto_refresh:
        # Use placeholder to show countdown
        placeholder = st.empty()
        for i in range(REFRESH_INTERVAL, 0, -1):
            placeholder.markdown(f"🔄 Auto-refresh in {i} seconds...")
            time.sleep(1)
        placeholder.empty()
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; font-size: 12px;'>
        System Status Monitor v1.0 | 
        Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
        Monitoring: IG Streaming & Backfill Services |
        Mode: {'Database + API' if MONITOR_AVAILABLE else 'API Only'}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_detailed_gap_analysis(timeframe: int):
    """Render detailed gap analysis using database queries"""
    st.header("🕳️ Detailed Gap Analysis")
    
    if 'monitor' not in st.session_state:
        st.warning("Monitor not initialized")
        return
    
    monitor = st.session_state.monitor
    
    # Major pairs for analysis
    major_pairs = [
        "CS.D.EURUSD.CEEM.IP",
        "CS.D.GBPUSD.MINI.IP",
        "CS.D.USDJPY.MINI.IP",
        "CS.D.USDCHF.MINI.IP"
    ]
    
    selected_epics = st.multiselect(
        "Select epics for gap analysis:",
        major_pairs,
        default=major_pairs[:2]
    )
    
    hours_back = st.slider("Hours to analyze:", 1, 72, 24)
    
    if selected_epics and st.button("🔍 Analyze Gaps"):
        with st.spinner("Analyzing gaps..."):
            for epic in selected_epics:
                st.subheader(f"📊 {epic.replace('CS.D.', '').replace('.MINI.IP', '')}")
                
                gaps_data = monitor.get_data_gaps_from_db(epic, timeframe, hours_back)
                
                if "error" in gaps_data:
                    st.error(f"Error analyzing {epic}: {gaps_data['error']}")
                    continue
                
                total_gaps = gaps_data.get("total_gaps", 0)
                gaps = gaps_data.get("gaps", [])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Gaps", total_gaps)
                
                with col2:
                    if gaps:
                        avg_gap = sum(gap["gap_minutes"] for gap in gaps) / len(gaps)
                        st.metric("Avg Gap Size", f"{avg_gap:.1f} min")
                
                if gaps:
                    st.write("**Recent Gaps:**")
                    gaps_df = pd.DataFrame(gaps)
                    st.dataframe(gaps_df, use_container_width=True)

if __name__ == "__main__":
    main()