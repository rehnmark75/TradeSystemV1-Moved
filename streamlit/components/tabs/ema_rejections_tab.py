"""
EMA Double Rejections Tab Component

Renders the EMA Double Confirmation strategy rejection analysis tab with sub-tabs:
- Stage Breakdown
- Time Analysis
- Market Context
- Near-Misses
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any

from services.rejection_analytics_service import RejectionAnalyticsService


def render_ema_double_rejections_tab():
    """Render EMA Double Confirmation Rejection Analysis tab"""
    service = RejectionAnalyticsService()

    # Header with refresh button
    header_col1, header_col2 = st.columns([6, 1])
    with header_col1:
        st.header("EMA Double Confirmation Rejection Analysis")
    with header_col2:
        if st.button("Refresh", key="ema_double_rejections_refresh", help="Refresh rejection data"):
            st.rerun()

    st.markdown("Analyze why EMA Double Confirmation strategy signals were rejected to improve strategy parameters")

    # Get filter options from service (cached)
    filter_options = service.get_ema_filter_options()

    if not filter_options.get('table_exists', True):
        st.warning("EMA Double Rejections table not yet created. Run the database migration:")
        st.code("""
docker exec -it postgres psql -U postgres -d trading -f /app/forex_scanner/migrations/create_ema_double_rejections_table.sql
        """)
        return

    stages = filter_options['stages']
    pairs = filter_options['pairs']
    sessions = filter_options['sessions']

    # Filters row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        days_filter = st.selectbox("Time Period", [3, 7, 14, 30, 60, 90], index=0, key="ema_double_rej_days")
    with col2:
        stage_filter = st.selectbox("Rejection Stage", stages, key="ema_double_rej_stage")
    with col3:
        pair_filter = st.selectbox("Pair", pairs, key="ema_double_rej_pair")
    with col4:
        session_filter = st.selectbox("Session", sessions, key="ema_double_rej_session")

    # Fetch statistics from service (cached)
    stats = service.fetch_ema_double_rejection_stats(days_filter)

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Rejections", f"{stats.get('total', 0):,}")
    with col2:
        st.metric("Unique Pairs", stats.get('unique_pairs', 0))
    with col3:
        top_stage = max(stats.get('by_stage', {'N/A': 0}).items(), key=lambda x: x[1])[0] if stats.get('by_stage') else 'N/A'
        st.metric("Top Stage", top_stage)
    with col4:
        st.metric("Near-Misses", stats.get('near_misses', 0), help="Signals that reached confidence stage but were rejected")
    with col5:
        st.metric("Most Rejected", stats.get('most_rejected_pair', 'N/A'))
    with col6:
        st.metric("Avg Crossovers", f"{stats.get('avg_crossover_count', 0):.1f}")

    st.markdown("---")

    # Sub-tabs for different analysis views
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Stage Breakdown", "Time Analysis", "Market Context", "Near-Misses"
    ])

    # Fetch data from service (cached)
    df = service.fetch_ema_double_rejections(days_filter, stage_filter, pair_filter, session_filter)

    if df.empty:
        st.info("No rejections found for the selected filters.")
        return

    with sub_tab1:
        _render_stage_breakdown(df, stats)

    with sub_tab2:
        _render_time_analysis(df)

    with sub_tab3:
        _render_market_context(df)

    with sub_tab4:
        _render_near_misses(df, days_filter)


def _render_stage_breakdown(df: pd.DataFrame, stats: dict):
    """Render stage breakdown sub-tab for EMA Double strategy"""
    st.subheader("Rejections by Stage")

    if df.empty:
        st.info("No rejection data available.")
        return

    # Pie chart for stage distribution
    stage_counts = df['rejection_stage'].value_counts().reset_index()
    stage_counts.columns = ['Stage', 'Count']

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_pie = px.pie(
            stage_counts,
            values='Count',
            names='Stage',
            title="Rejection Distribution by Stage",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar chart
        fig_bar = px.bar(
            stage_counts,
            x='Stage',
            y='Count',
            title="Rejection Counts by Stage",
            color='Count',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed reason breakdown
    st.subheader("Top Rejection Reasons")
    reason_counts = df.groupby(['rejection_stage', 'rejection_reason']).size().reset_index(name='Count')
    reason_counts = reason_counts.sort_values('Count', ascending=False).head(20)

    st.dataframe(
        reason_counts,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rejection_stage": st.column_config.TextColumn("Stage"),
            "rejection_reason": st.column_config.TextColumn("Reason"),
            "Count": st.column_config.NumberColumn("Count", format="%d")
        }
    )


def _render_time_analysis(df: pd.DataFrame):
    """Render time analysis sub-tab for EMA Double strategy"""
    st.subheader("Rejection Patterns Over Time")

    if df.empty or 'market_hour' not in df.columns:
        st.info("No time data available.")
        return

    # Heatmap: Hour vs Stage
    if 'market_hour' in df.columns and df['market_hour'].notna().any():
        st.markdown("#### Rejections by Hour and Stage")
        pivot = df.groupby(['market_hour', 'rejection_stage']).size().unstack(fill_value=0)

        fig_heat = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            labels=dict(x="Rejection Stage", y="Hour (UTC)", color="Count"),
            title="Rejection Heatmap: Hour vs Stage",
            color_continuous_scale='YlOrBr'
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Rejections by hour
    col1, col2 = st.columns(2)

    with col1:
        if 'market_hour' in df.columns:
            hour_counts = df['market_hour'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['Hour', 'Count']

            fig_hour = px.bar(
                hour_counts,
                x='Hour',
                y='Count',
                title="Rejections by Hour (UTC)",
                color='Count',
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig_hour, use_container_width=True)

    with col2:
        # By session
        if 'market_session' in df.columns and df['market_session'].notna().any():
            session_counts = df['market_session'].value_counts().reset_index()
            session_counts.columns = ['Session', 'Count']

            fig_session = px.bar(
                session_counts,
                x='Session',
                y='Count',
                title="Rejections by Market Session",
                color='Session',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_session, use_container_width=True)

    # Rejections over time (daily trend)
    st.markdown("#### Daily Rejection Trend")
    if 'scan_timestamp' in df.columns:
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['scan_timestamp']).dt.date
        daily_counts = df_copy.groupby(['date', 'rejection_stage']).size().reset_index(name='Count')

        fig_trend = px.line(
            daily_counts,
            x='date',
            y='Count',
            color='rejection_stage',
            title="Daily Rejections by Stage",
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)


def _render_market_context(df: pd.DataFrame):
    """Render market context sub-tab for EMA Double strategy"""
    st.subheader("Market Context Analysis")

    if df.empty:
        st.info("No market context data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Crossover count distribution by stage
        if 'successful_crossover_count' in df.columns and df['successful_crossover_count'].notna().any():
            st.markdown("#### Successful Crossovers at Rejection")
            fig_xover = px.box(
                df[df['successful_crossover_count'].notna()],
                x='rejection_stage',
                y='successful_crossover_count',
                title="Crossover Count by Rejection Stage",
                color='rejection_stage'
            )
            st.plotly_chart(fig_xover, use_container_width=True)

    with col2:
        # ADX distribution by stage
        if 'adx_value' in df.columns and df['adx_value'].notna().any():
            st.markdown("#### ADX at Rejection Points")
            fig_adx = px.histogram(
                df[df['adx_value'].notna()],
                x='adx_value',
                color='rejection_stage',
                title="ADX Distribution",
                nbins=30
            )
            st.plotly_chart(fig_adx, use_container_width=True)

    # EMA separation analysis
    if 'ema_fast_slow_separation_pips' in df.columns and df['ema_fast_slow_separation_pips'].notna().any():
        st.markdown("#### EMA Separation at Rejection Points")
        fig_ema = px.scatter(
            df[df['ema_fast_slow_separation_pips'].notna()],
            x='ema_fast_slow_separation_pips',
            y='adx_value' if 'adx_value' in df.columns else 'successful_crossover_count',
            color='rejection_stage',
            title="EMA Separation vs ADX",
            hover_data=['pair', 'rejection_reason']
        )
        st.plotly_chart(fig_ema, use_container_width=True)

    # RSI analysis
    if 'rsi_value' in df.columns and df['rsi_value'].notna().any():
        st.markdown("#### RSI Distribution")
        fig_rsi = px.histogram(
            df[df['rsi_value'].notna()],
            x='rsi_value',
            color='attempted_direction',
            title="RSI at Rejection Points",
            nbins=20
        )
        st.plotly_chart(fig_rsi, use_container_width=True)


def _render_near_misses(df: pd.DataFrame, days: int):
    """Render near-misses sub-tab for EMA Double strategy"""
    st.subheader("Near-Miss Signals")
    st.caption("Signals that almost passed - reached the confidence stage but were rejected")

    # Filter to confidence stage rejections
    near_miss_df = df[df['rejection_stage'] == 'CONFIDENCE'].copy()

    if near_miss_df.empty:
        st.info("No near-miss signals (confidence-stage rejections) found for the selected period.")
        return

    # Sort by confidence score descending
    if 'confidence_score' in near_miss_df.columns:
        near_miss_df = near_miss_df.sort_values('confidence_score', ascending=False)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Near-Misses", len(near_miss_df))
    with col2:
        avg_conf = near_miss_df['confidence_score'].mean() * 100 if 'confidence_score' in near_miss_df.columns and near_miss_df['confidence_score'].notna().any() else 0
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    with col3:
        avg_rr = near_miss_df['potential_rr_ratio'].mean() if 'potential_rr_ratio' in near_miss_df.columns and near_miss_df['potential_rr_ratio'].notna().any() else 0
        st.metric("Avg R:R", f"{avg_rr:.2f}")
    with col4:
        bull_count = len(near_miss_df[near_miss_df['attempted_direction'] == 'BULL']) if 'attempted_direction' in near_miss_df.columns else 0
        bear_count = len(near_miss_df[near_miss_df['attempted_direction'] == 'BEAR']) if 'attempted_direction' in near_miss_df.columns else 0
        st.metric("Direction Split", f"{bull_count} Bull / {bear_count} Bear")

    st.markdown("---")

    # Display near-misses with expandable details
    for idx, row in near_miss_df.head(20).iterrows():
        timestamp = row.get('scan_timestamp', '')
        if isinstance(timestamp, pd.Timestamp):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
        else:
            timestamp_str = str(timestamp)[:16] if timestamp else 'N/A'

        pair = row.get('pair', 'N/A')
        direction = row.get('attempted_direction', 'N/A')
        direction_icon = "+" if direction == 'BULL' else "-" if direction == 'BEAR' else "o"
        confidence = row.get('confidence_score', 0)
        conf_str = f"{confidence*100:.0f}%" if confidence else 'N/A'
        crossovers = row.get('successful_crossover_count', 0)
        session = row.get('market_session', 'N/A')
        reason = row.get('rejection_reason', 'N/A')

        expander_title = f"{direction_icon} {timestamp_str} | {pair} | {direction} | Conf: {conf_str} | Crossovers: {crossovers} | {session}"

        with st.expander(expander_title, expanded=False):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Signal Details:**")
                st.write(f"- **Pair:** {pair}")
                st.write(f"- **Direction:** {direction}")
                st.write(f"- **Confidence:** {conf_str}")
                st.write(f"- **Session:** {session}")
                st.write(f"- **Prior Crossovers:** {crossovers}")
                adx = row.get('adx_value', None)
                if adx:
                    st.write(f"- **ADX:** {adx:.1f}")

            with detail_col2:
                st.markdown("**EMA Context:**")
                ema_fast = row.get('ema_fast_value', None)
                ema_slow = row.get('ema_slow_value', None)
                separation = row.get('ema_fast_slow_separation_pips', None)
                if ema_fast:
                    st.write(f"- **EMA Fast:** {ema_fast:.5f}")
                if ema_slow:
                    st.write(f"- **EMA Slow:** {ema_slow:.5f}")
                if separation:
                    st.write(f"- **Separation:** {separation:.1f} pips")
                htf_pos = row.get('htf_price_position', None)
                if htf_pos:
                    st.write(f"- **HTF Position:** {htf_pos}")

            # Rejection reason
            st.markdown("**Rejection Reason:**")
            st.warning(reason)

    # Export button
    st.markdown("---")
    csv = near_miss_df.to_csv(index=False)
    st.download_button(
        label="Export Near-Misses to CSV",
        data=csv,
        file_name=f"ema_double_near_misses_{days}days.csv",
        mime="text/csv"
    )
