"""
Alert History Tab Component

Renders the Alert History tab with Claude Vision analysis status.
Sub-features:
- Alert list with expandable details
- Filter by status, strategy, pair
- Summary metrics
- Chart image display where available
"""

import streamlit as st
import pandas as pd
import os
import glob
from typing import Optional

from services.alert_history_service import AlertHistoryService


def render_alert_history_tab():
    """Render Alert History tab with Claude Vision analysis status"""
    service = AlertHistoryService()

    # Header with refresh button
    header_col1, header_col2 = st.columns([6, 1])
    with header_col1:
        st.header("Alert History")
    with header_col2:
        if st.button("Refresh", key="alert_history_refresh", help="Refresh alert data"):
            st.rerun()

    st.markdown("View all trading signals with Claude AI analysis status")

    # Get filter options from service (cached)
    filter_options = service.get_filter_options()
    strategies = filter_options['strategies']
    pairs = filter_options['pairs']

    # Filters row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        days_filter = st.selectbox("Time Period", [1, 3, 7, 14, 30], index=0, key="alert_history_days")
    with col2:
        status_filter = st.selectbox("Claude Status", ["All", "Approved", "Rejected"], key="alert_history_status")
    with col3:
        strategy_filter = st.selectbox("Strategy", strategies, key="alert_history_strategy")
    with col4:
        pair_filter = st.selectbox("Pair", pairs, key="alert_history_pair")

    # Fetch data from service (cached)
    df = service.fetch_alert_history(days_filter, status_filter, strategy_filter, pair_filter)

    if df.empty:
        st.info("No alerts found for the selected filters.")
        return

    # Summary metrics
    st.markdown("---")
    total_alerts = len(df)
    approved_count = len(df[df['claude_approved'] == True]) if 'claude_approved' in df.columns else 0
    rejected_count = len(df[(df['claude_approved'] == False) | (df['alert_level'] == 'REJECTED')]) if 'claude_approved' in df.columns else 0
    approval_rate = (approved_count / total_alerts * 100) if total_alerts > 0 else 0
    avg_score = df['claude_score'].mean() if 'claude_score' in df.columns and not df['claude_score'].isna().all() else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Alerts", total_alerts)
    with col2:
        st.metric("Approved", approved_count, delta=None)
    with col3:
        st.metric("Rejected", rejected_count, delta=None)
    with col4:
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    with col5:
        st.metric("Avg Score", f"{avg_score:.1f}/10" if avg_score > 0 else "N/A")

    st.markdown("---")

    # Pagination settings
    ITEMS_PER_PAGE = 25
    total_pages = (len(df) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE

    # Page selector
    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
    with page_col2:
        if total_pages > 1:
            current_page = st.selectbox(
                f"Page (showing {ITEMS_PER_PAGE} per page)",
                range(1, total_pages + 1),
                key="alert_history_page",
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
        else:
            current_page = 1

    # Calculate slice for current page
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, len(df))

    st.caption(f"Showing alerts {start_idx + 1}-{end_idx} of {len(df)}")

    # Display only current page of alerts
    page_df = df.iloc[start_idx:end_idx]
    for idx, row in page_df.iterrows():
        _render_alert_row(row)


def _render_alert_row(row: pd.Series):
    """Render a single alert row with expandable details"""
    # Determine status icon and color
    is_approved = row.get('claude_approved', None)
    claude_decision = row.get('claude_decision', '')
    alert_level = row.get('alert_level', '')

    if is_approved == True or claude_decision == 'APPROVE':
        status_icon = "✅"
        status_text = "APPROVED"
    elif is_approved == False or claude_decision == 'REJECT' or alert_level == 'REJECTED':
        status_icon = "❌"
        status_text = "REJECTED"
    else:
        status_icon = "⚪"
        status_text = "PENDING"

    # Format timestamp
    timestamp = row.get('alert_timestamp', '')
    if isinstance(timestamp, pd.Timestamp):
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
    else:
        timestamp_str = str(timestamp)[:16] if timestamp else 'N/A'

    # Get values with defaults
    pair = row.get('pair', row.get('epic', 'N/A'))
    if pair == 'N/A' or pd.isna(pair):
        epic = row.get('epic', '')
        if epic:
            # Extract pair from epic like CS.D.EURUSD.CEEM.IP
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2][:6] if len(parts[2]) >= 6 else parts[2]

    strategy = row.get('strategy', 'N/A')
    signal_type = row.get('signal_type', 'N/A')
    price = row.get('price', 0)
    price_str = f"{price:.5f}" if price and not pd.isna(price) else 'N/A'
    session = row.get('market_session', 'N/A')
    if pd.isna(session):
        session = 'N/A'
    score = row.get('claude_score', 0)
    score_str = f"{int(score)}/10" if score and not pd.isna(score) else 'N/A'

    # Create expander for each row
    expander_title = f"{status_icon} {timestamp_str} | {pair} | {strategy} | {signal_type} | {price_str} | {session} | Score: {score_str}"

    with st.expander(expander_title, expanded=False):
        # Two columns: details and chart
        detail_col, chart_col = st.columns([1, 1])

        with detail_col:
            st.markdown("**Signal Details:**")
            st.write(f"- **Status:** {status_icon} {status_text}")
            st.write(f"- **Pair:** {pair}")
            st.write(f"- **Strategy:** {strategy}")
            st.write(f"- **Signal:** {signal_type}")
            st.write(f"- **Price:** {price_str}")
            st.write(f"- **Session:** {session}")
            st.write(f"- **Claude Score:** {score_str}")
            st.write(f"- **Claude Mode:** {row.get('claude_mode', 'N/A')}")

            # Claude reason
            reason = row.get('claude_reason', '')
            if reason and not pd.isna(reason):
                st.markdown("**Claude Reason:**")
                st.info(reason)

        with chart_col:
            st.markdown("**Chart Image:**")

            # Priority 1: Use MinIO URL from database if available
            chart_url = row.get('vision_chart_url', None)
            if chart_url and not pd.isna(chart_url) and not chart_url.startswith('file://'):
                try:
                    # Fetch image server-side since minio:9000 is not accessible from browser
                    import urllib.request
                    with urllib.request.urlopen(chart_url, timeout=10) as response:
                        image_bytes = response.read()
                    st.image(image_bytes, caption="Vision Analysis Chart", use_container_width=True)
                except Exception as e:
                    # URL might have expired (30-day retention) - show fallback message
                    st.warning(f"Chart expired or unavailable: {e}")

            # Priority 2: Fallback to disk storage for older records
            else:
                alert_id = row.get('id', None)
                chart_path = _get_vision_chart_path(
                    row.get('epic', ''),
                    row.get('alert_timestamp'),
                    alert_id
                )

                if chart_path and os.path.exists(chart_path):
                    st.image(chart_path, caption="Vision Analysis Chart", use_container_width=True)
                elif chart_url and chart_url.startswith('file://'):
                    # Handle file:// URL format from disk fallback
                    local_path = chart_url.replace('file://', '')
                    if os.path.exists(local_path):
                        st.image(local_path, caption="Vision Analysis Chart", use_container_width=True)
                    else:
                        st.info("Chart not available (file not found)")
                else:
                    st.info("No chart available (vision analysis not used or chart expired)")

        # Full raw response in a separate section
        raw_response = row.get('claude_raw_response', '')
        if raw_response and not pd.isna(raw_response):
            st.markdown("---")
            st.markdown("**Full Claude Raw Response:**")
            st.code(raw_response, language=None)


def _get_vision_chart_path(epic: str, timestamp, alert_id: int = None) -> Optional[str]:
    """
    Find the chart image path for an alert.

    Args:
        epic: Trading instrument epic
        timestamp: Alert timestamp
        alert_id: Optional alert ID

    Returns:
        Path to chart image or None if not found
    """
    # Vision artifacts directory (check both local and Docker paths)
    vision_dirs = [
        "claude_analysis_enhanced/vision_analysis",
        "/app/claude_analysis_enhanced/vision_analysis",
        "../worker/app/claude_analysis_enhanced/vision_analysis"
    ]

    # Clean epic for filename matching
    epic_clean = epic.replace('.', '_') if epic else ""

    # Format timestamp for matching
    if isinstance(timestamp, str):
        try:
            ts = pd.to_datetime(timestamp)
            timestamp_str = ts.strftime('%Y%m%d_%H%M')
        except:
            timestamp_str = ""
    elif hasattr(timestamp, 'strftime'):
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M')
    else:
        timestamp_str = ""

    for vision_dir in vision_dirs:
        if not os.path.exists(vision_dir):
            continue

        # Try to find matching chart file
        patterns = [
            f"{vision_dir}/{alert_id}_{epic_clean}*_chart.png" if alert_id else None,
            f"{vision_dir}/{epic_clean}_{timestamp_str}*_chart.png",
            f"{vision_dir}/*{epic_clean}*_chart.png"
        ]

        for pattern in patterns:
            if pattern:
                matches = glob.glob(pattern)
                if matches:
                    # Return most recent match
                    return sorted(matches)[-1]

    return None
