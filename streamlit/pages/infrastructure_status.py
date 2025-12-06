"""
Infrastructure Status - Container monitoring dashboard.
Real-time visualization of all Docker containers with drill-down capability.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

from infrastructure_service import (
    get_infrastructure_service,
    format_bytes,
    format_uptime,
    get_status_color,
    get_severity_color,
)

# Page configuration
st.set_page_config(
    page_title="Infrastructure Status",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }

    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }

    .container-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .container-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }

    .status-running { background-color: #28a745; }
    .status-stopped { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; color: #333; }
    .status-unknown { background-color: #6c757d; }

    .alert-item {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }

    .alert-critical {
        background: #fff5f5;
        border-left-color: #dc3545;
    }

    .alert-warning {
        background: #fffbeb;
        border-left-color: #ffc107;
    }

    .alert-info {
        background: #f0f9ff;
        border-left-color: #17a2b8;
    }

    .health-bar {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        overflow: hidden;
    }

    .health-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .log-viewer {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render page header."""
    st.markdown("""
    <div class="main-header">
        <h1>🖥️ Infrastructure Status</h1>
        <p>Real-time container monitoring and system health</p>
    </div>
    """, unsafe_allow_html=True)


def render_service_unavailable():
    """Render message when system monitor is unavailable."""
    st.error("⚠️ System Monitor service is not available")
    st.info("""
    The System Monitor container may not be running. To start it:

    ```bash
    docker-compose up -d system-monitor
    ```

    Or check if there are connection issues:
    ```bash
    docker logs system-monitor
    ```
    """)


def render_status_overview(status: dict):
    """Render the status overview cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        running = status.get("running_containers", 0)
        total = status.get("total_containers", 0)
        st.metric(
            label="🟢 Running",
            value=f"{running}/{total}",
            delta=None
        )

    with col2:
        stopped = status.get("stopped_containers", 0)
        st.metric(
            label="🔴 Stopped",
            value=stopped,
            delta=None if stopped == 0 else f"+{stopped}",
            delta_color="inverse"
        )

    with col3:
        unhealthy = status.get("unhealthy_containers", 0)
        st.metric(
            label="⚠️ Unhealthy",
            value=unhealthy,
            delta=None if unhealthy == 0 else f"+{unhealthy}",
            delta_color="inverse"
        )

    with col4:
        alerts = status.get("active_alerts", 0)
        st.metric(
            label="🔔 Alerts",
            value=alerts,
            delta=None if alerts == 0 else f"+{alerts}",
            delta_color="inverse"
        )

    with col5:
        score = status.get("health_score", 0)
        st.metric(
            label="📊 Health Score",
            value=f"{score:.0f}%",
            delta=None
        )

    # Health bar
    score_color = "#28a745" if score >= 80 else "#ffc107" if score >= 60 else "#dc3545"
    st.markdown(f"""
    <div class="health-bar">
        <div class="health-bar-fill" style="width: {score}%; background-color: {score_color};"></div>
    </div>
    """, unsafe_allow_html=True)


def render_container_grid(containers: list):
    """Render the container status grid."""
    st.subheader("📦 Container Status")

    # Sort containers: critical first, then by status
    def sort_key(c):
        status_order = {"stopped": 0, "exited": 0, "unhealthy": 1, "restarting": 2, "running": 3}
        return (
            0 if c.get("is_critical") else 1,
            status_order.get(c.get("status", ""), 5),
            c.get("name", "")
        )

    sorted_containers = sorted(containers, key=sort_key)

    # Create grid
    cols = st.columns(4)
    for idx, container in enumerate(sorted_containers):
        col = cols[idx % 4]
        with col:
            render_container_card(container)


def render_container_card(container: dict):
    """Render a single container card using native Streamlit components."""
    name = container.get("name", "Unknown")
    status = container.get("status", "unknown")
    status_emoji = container.get("status_emoji", "⚪")
    is_critical = container.get("is_critical", False)
    uptime = container.get("uptime_human", "N/A")
    description = container.get("description", "")

    warnings = container.get("warnings", [])
    errors = container.get("errors", [])

    # Determine status indicator
    if errors or status != "running":
        status_icon = "🔴"
    elif warnings:
        status_icon = "🟡"
    else:
        status_icon = "🟢"

    # Build title with critical star
    title = f"{status_icon} {name}"
    if is_critical:
        title += " ⭐"

    # Use native Streamlit container with border
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"Uptime: {uptime}")
        if description:
            st.caption(description)

        # Show warnings/errors if any
        if errors:
            for error in errors:
                st.error(error, icon="🚨")
        if warnings:
            for warning in warnings:
                st.warning(warning, icon="⚠️")


def render_container_detail(service: object, container_name: str):
    """Render detailed view for a specific container."""
    detail = service.get_container_detail(container_name)

    if not detail:
        st.error(f"Could not fetch details for {container_name}")
        return

    st.subheader(f"📦 {container_name} Details")

    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", f"{detail.get('status_emoji', '')} {detail.get('status', 'N/A')}")
    with col2:
        st.metric("Health", detail.get("health_status", "N/A"))
    with col3:
        st.metric("Uptime", detail.get("uptime_human", "N/A"))
    with col4:
        st.metric("Restarts", detail.get("restart_count", 0))

    # Container info
    st.markdown("### Container Info")
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"Container ID: {detail.get('container_id', 'N/A')}")
        st.text(f"Image: {detail.get('image', 'N/A')}")
    with col2:
        ports = detail.get("ports", {})
        if ports:
            port_str = ", ".join([f"{k} → {v}" for k, v in ports.items()])
            st.text(f"Ports: {port_str}")
        st.text(f"Critical: {'Yes' if detail.get('is_critical') else 'No'}")

    # Logs
    st.markdown("### Recent Logs")
    if st.button("📜 Load Logs", key=f"logs_{container_name}"):
        logs_data = service.get_container_logs(container_name, lines=100)
        if logs_data and logs_data.get("logs"):
            st.markdown(f"""
            <div class="log-viewer">
{logs_data['logs']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No logs available")

    # Actions
    st.markdown("### Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Restart Container", key=f"restart_{container_name}"):
            result = service.restart_container(container_name)
            if result and result.get("success"):
                st.success(f"Container {container_name} restarted successfully!")
                st.rerun()
            else:
                st.error(f"Failed to restart container: {result}")


def render_alerts(service: object):
    """Render the alerts section."""
    st.subheader("🔔 Recent Alerts")

    alerts_data = service.get_alerts(limit=20, active_only=False)

    if not alerts_data or not alerts_data.get("alerts"):
        st.info("No recent alerts")
        return

    alerts = alerts_data["alerts"]

    # Filter options
    col1, col2 = st.columns([1, 4])
    with col1:
        show_active_only = st.checkbox("Active only", value=False)

    if show_active_only:
        alerts = [a for a in alerts if not a.get("resolved_at")]

    for alert in alerts:
        severity = alert.get("severity", "info")
        alert_class = f"alert-{severity}"

        emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(severity, "📢")

        created = alert.get("created_at", "")[:19] if alert.get("created_at") else "N/A"
        container = alert.get("container_name", "System")
        title = alert.get("title", "Unknown alert")
        message = alert.get("message", "")

        resolved = "✅ Resolved" if alert.get("resolved_at") else ""
        acknowledged = "👁️ Acknowledged" if alert.get("acknowledged_at") else ""

        st.markdown(f"""
        <div class="{alert_class}" style="
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            border-left: 4px solid {get_severity_color(severity)};
            background: {'#fff5f5' if severity == 'critical' else '#fffbeb' if severity == 'warning' else '#f0f9ff'};
        ">
            <div style="display: flex; justify-content: space-between;">
                <strong>{emoji} {title}</strong>
                <span style="color: #666; font-size: 0.85rem;">{created}</span>
            </div>
            <div style="margin-top: 0.5rem; color: #666;">
                📦 {container} {resolved} {acknowledged}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Acknowledge/Resolve buttons for active alerts
        if not alert.get("resolved_at"):
            col1, col2, col3 = st.columns([1, 1, 8])
            alert_id = alert.get("id")
            with col1:
                if not alert.get("acknowledged_at"):
                    if st.button("Ack", key=f"ack_{alert_id}"):
                        service.acknowledge_alert(alert_id)
                        st.rerun()
            with col2:
                if st.button("Resolve", key=f"resolve_{alert_id}"):
                    service.resolve_alert(alert_id)
                    st.rerun()


def render_health_checks(service: object):
    """Render health check results."""
    st.subheader("🏥 Service Health Checks")

    health_data = service.get_health_checks()

    if not health_data or not health_data.get("services"):
        st.info("No health check data available")
        return

    services = health_data["services"]

    # Create table
    data = []
    for name, info in services.items():
        status = info.get("status", "unknown")
        response_time = info.get("response_time_ms", 0)
        failures = info.get("consecutive_failures", 0)

        status_emoji = "🟢" if status == "healthy" else "🔴" if status == "unhealthy" else "⚪"

        data.append({
            "Service": name,
            "Status": f"{status_emoji} {status}",
            "Response Time": f"{response_time:.0f}ms",
            "Failures": failures,
            "Error": info.get("error") or "-"
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    """Main page function."""
    render_header()

    # Get service
    service = get_infrastructure_service()

    # Check availability
    if not service.is_available():
        render_service_unavailable()
        return

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Controls")

        if st.button("🔄 Refresh"):
            st.rerun()

        st.markdown("---")

        # Container selection for detail view
        containers_data = service.get_all_containers()
        if containers_data and containers_data.get("containers"):
            container_names = ["Overview"] + [c["name"] for c in containers_data["containers"]]
            selected = st.selectbox("View", container_names)
        else:
            selected = "Overview"

        st.markdown("---")

        # Notification test
        st.markdown("### 📬 Notifications")
        if st.button("Test Telegram"):
            result = service.test_notification("telegram")
            if result and result.get("success"):
                st.success("Telegram test sent!")
            else:
                st.error(f"Failed: {result.get('message', 'Unknown error')}")

        st.markdown("---")

        # Last update time
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Main content
    if selected == "Overview":
        # Status overview
        status = service.get_system_status()
        if status:
            render_status_overview(status)

        st.markdown("---")

        # Container grid
        if containers_data and containers_data.get("containers"):
            render_container_grid(containers_data["containers"])

        st.markdown("---")

        # Tabs for alerts and health checks
        tab1, tab2 = st.tabs(["🔔 Alerts", "🏥 Health Checks"])

        with tab1:
            render_alerts(service)

        with tab2:
            render_health_checks(service)

    else:
        # Container detail view
        render_container_detail(service, selected)

        if st.button("← Back to Overview"):
            st.rerun()


if __name__ == "__main__":
    main()
