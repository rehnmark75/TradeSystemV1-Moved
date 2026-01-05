"""
Scan Performance Snapshot Tab Component

Renders the Scan Performance Snapshot analysis tab with:
- Scan overview metrics
- Signal generation tracking
- Rejection analysis
- Market regime and session distribution
- Per-epic performance breakdown
- Indicator comparisons
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict

from services.performance_snapshot_service import PerformanceSnapshotService


def render_performance_snapshot_tab():
    """Render the Scan Performance Snapshot analysis tab"""
    service = PerformanceSnapshotService()

    st.header("Scan Performance Snapshot Analysis")
    st.markdown("*Analyze scan performance, signal generation, and rejection patterns*")

    # Date range and controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=1),
            key="sps_start_date"
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key="sps_end_date"
        )

    with col3:
        if st.button("Refresh", key="sps_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Fetch summary data
    summary = service.get_scan_summary(start_date, end_date)

    if not summary or summary.get('total_scans', 0) == 0:
        st.warning("No scan performance snapshot data found for the selected period.")
        st.info("Scan performance snapshots are recorded during live scanner operations.")
        return

    # Overview Metrics
    _render_overview_metrics(summary)

    # Scan Timeline
    timeline = service.get_scan_timeline(start_date, end_date)
    if not timeline.empty:
        _render_scan_timeline(timeline)

    # Regime and Session Distribution
    col1, col2 = st.columns(2)

    with col1:
        regime_dist = service.get_regime_distribution(start_date, end_date)
        if not regime_dist.empty:
            _render_regime_distribution(regime_dist)

    with col2:
        session_dist = service.get_session_distribution(start_date, end_date)
        if not session_dist.empty:
            _render_session_distribution(session_dist)

    # Rejection Analysis
    rejection_data = service.get_rejection_analysis(start_date, end_date)
    if not rejection_data.empty:
        _render_rejection_analysis(rejection_data)

    # Epic Performance
    epic_summary = service.get_epic_summary(start_date, end_date)
    if not epic_summary.empty:
        _render_epic_performance(epic_summary)

    # Indicator Comparison
    indicator_stats = service.get_indicator_stats(start_date, end_date)
    if indicator_stats:
        _render_indicator_comparison(indicator_stats)

    # Raw Data Export
    _render_export_section(service, start_date, end_date)


def _render_overview_metrics(summary: Dict):
    """Render overview metrics cards"""
    st.subheader("Scan Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Scans",
            f"{summary.get('total_scans', 0):,}"
        )

    with col2:
        st.metric(
            "Scan Cycles",
            f"{summary.get('scan_cycles', 0):,}"
        )

    with col3:
        st.metric(
            "Unique Epics",
            f"{summary.get('unique_epics', 0)}"
        )

    with col4:
        signals = summary.get('signals_generated', 0)
        st.metric(
            "Signals Generated",
            f"{signals}",
            delta=f"{summary.get('buy_signals', 0)} BUY / {summary.get('sell_signals', 0)} SELL"
        )

    with col5:
        signal_rate = summary.get('signal_rate', 0) * 100
        st.metric(
            "Signal Rate",
            f"{signal_rate:.2f}%"
        )

    # Second row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Avg Raw Confidence",
            f"{summary.get('avg_raw_confidence', 0):.3f}"
        )

    with col2:
        st.metric(
            "Avg Final Confidence",
            f"{summary.get('avg_final_confidence', 0):.3f}"
        )

    with col3:
        st.metric(
            "Confidence Rejections",
            f"{summary.get('confidence_rejections', 0):,}"
        )

    with col4:
        st.metric(
            "Dedup Rejections",
            f"{summary.get('dedup_rejections', 0):,}"
        )


def _render_scan_timeline(timeline: pd.DataFrame):
    """Render scan timeline chart"""
    st.subheader("Scan Activity Timeline")

    fig = go.Figure()

    # Total scans as area
    fig.add_trace(go.Scatter(
        x=timeline['hour'],
        y=timeline['total_scans'],
        mode='lines',
        name='Total Scans',
        fill='tozeroy',
        fillcolor='rgba(0, 123, 255, 0.2)',
        line=dict(color='#007bff', width=2)
    ))

    # Signals as bars
    fig.add_trace(go.Bar(
        x=timeline['hour'],
        y=timeline['signals'],
        name='Signals Generated',
        marker_color='#28a745',
        opacity=0.7
    ))

    fig.update_layout(
        title="Scan Activity Over Time",
        xaxis_title="Time",
        yaxis_title="Count",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        barmode='overlay'
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_regime_distribution(regime_dist: pd.DataFrame):
    """Render market regime distribution"""
    st.subheader("Market Regime Distribution")

    # Convert numeric columns
    for col in ['count', 'signals', 'avg_confidence', 'avg_signal_confidence']:
        if col in regime_dist.columns:
            regime_dist[col] = pd.to_numeric(regime_dist[col], errors='coerce')

    fig = px.pie(
        regime_dist,
        values='count',
        names='market_regime',
        title="Scans by Market Regime",
        color='market_regime',
        color_discrete_map={
            'trending': '#28a745',
            'ranging': '#007bff',
            'breakout': '#ff6b6b',
            'reversal': '#ffc107',
            'low_volatility': '#6c757d',
            'high_volatility': '#e83e8c'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # Signal rate by regime
    regime_dist['signal_rate'] = regime_dist['signals'] / regime_dist['count'] * 100
    fig_rate = px.bar(
        regime_dist,
        x='market_regime',
        y='signal_rate',
        title="Signal Rate by Regime (%)",
        color='signal_rate',
        color_continuous_scale='RdYlGn'
    )
    fig_rate.update_layout(yaxis_title="Signal Rate (%)")
    st.plotly_chart(fig_rate, use_container_width=True)


def _render_session_distribution(session_dist: pd.DataFrame):
    """Render trading session distribution"""
    st.subheader("Session Distribution")

    # Convert numeric columns before aggregation
    numeric_cols = ['count', 'signals', 'avg_confidence', 'avg_atr_pips', 'avg_spread']
    for col in numeric_cols:
        if col in session_dist.columns:
            session_dist[col] = pd.to_numeric(session_dist[col], errors='coerce')

    # Aggregate by session only
    session_agg = session_dist.groupby('session').agg({
        'count': 'sum',
        'signals': 'sum',
        'avg_confidence': 'mean',
        'avg_atr_pips': 'mean',
        'avg_spread': 'mean'
    }).reset_index()

    fig = px.bar(
        session_agg,
        x='session',
        y='count',
        title="Scans by Trading Session",
        color='signals',
        color_continuous_scale='Greens',
        text='signals'
    )
    fig.update_traces(texttemplate='%{text} signals', textposition='outside')
    fig.update_layout(yaxis_title="Total Scans")
    st.plotly_chart(fig, use_container_width=True)

    # Session metrics table - ensure numeric types before rounding
    session_agg['signal_rate'] = pd.to_numeric(session_agg['signals'], errors='coerce') / pd.to_numeric(session_agg['count'], errors='coerce') * 100
    session_agg['signal_rate'] = session_agg['signal_rate'].round(2)
    session_agg['avg_confidence'] = pd.to_numeric(session_agg['avg_confidence'], errors='coerce').round(4)
    session_agg['avg_atr_pips'] = pd.to_numeric(session_agg['avg_atr_pips'], errors='coerce').round(2)
    session_agg['avg_spread'] = pd.to_numeric(session_agg['avg_spread'], errors='coerce').round(2)

    st.dataframe(
        session_agg.rename(columns={
            'session': 'Session',
            'count': 'Total Scans',
            'signals': 'Signals',
            'signal_rate': 'Signal Rate %',
            'avg_confidence': 'Avg Confidence',
            'avg_atr_pips': 'Avg ATR (pips)',
            'avg_spread': 'Avg Spread'
        }),
        use_container_width=True,
        hide_index=True
    )


def _render_rejection_analysis(rejection_data: pd.DataFrame):
    """Render rejection analysis"""
    st.subheader("Rejection Analysis")

    # Convert numeric columns
    numeric_cols = ['count', 'avg_raw_confidence', 'avg_final_confidence', 'avg_threshold', 'affected_epics']
    for col in numeric_cols:
        if col in rejection_data.columns:
            rejection_data[col] = pd.to_numeric(rejection_data[col], errors='coerce')

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            rejection_data,
            values='count',
            names='rejection_reason',
            title="Rejections by Reason",
            color='rejection_reason',
            color_discrete_map={
                'confidence': '#ffc107',
                'dedup': '#17a2b8',
                'validation': '#dc3545'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Confidence gap analysis
        rejection_data['confidence_gap'] = rejection_data['avg_threshold'] - rejection_data['avg_final_confidence']

        fig = px.bar(
            rejection_data,
            x='rejection_reason',
            y=['avg_final_confidence', 'avg_threshold'],
            title="Confidence vs Threshold by Rejection Reason",
            barmode='group',
            labels={'value': 'Confidence', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rejection details table - select only numeric columns for rounding
    display_df = rejection_data.copy()
    for col in ['avg_raw_confidence', 'avg_final_confidence', 'avg_threshold', 'confidence_gap']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(4)

    st.dataframe(
        display_df.rename(columns={
            'rejection_reason': 'Reason',
            'count': 'Count',
            'avg_raw_confidence': 'Avg Raw Conf',
            'avg_final_confidence': 'Avg Final Conf',
            'avg_threshold': 'Avg Threshold',
            'affected_epics': 'Affected Epics'
        }),
        use_container_width=True,
        hide_index=True
    )


def _render_epic_performance(epic_summary: pd.DataFrame):
    """Render per-epic performance breakdown"""
    st.subheader("Epic (Pair) Performance")

    # Convert numeric columns
    numeric_cols = ['total_scans', 'signals', 'buy_signals', 'sell_signals', 'signal_rate',
                   'avg_raw_confidence', 'avg_final_confidence', 'avg_rsi', 'avg_adx',
                   'avg_atr_pips', 'avg_spread', 'confidence_rejections', 'dedup_rejections']
    for col in numeric_cols:
        if col in epic_summary.columns:
            epic_summary[col] = pd.to_numeric(epic_summary[col], errors='coerce')

    col1, col2 = st.columns(2)

    with col1:
        # Top epics by signal count
        top_epics = epic_summary.head(10)
        fig = px.bar(
            top_epics,
            x='pair_name',
            y='signals',
            title="Top 10 Pairs by Signal Count",
            color='signal_rate',
            color_continuous_scale='RdYlGn',
            text='signals'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Signal rate distribution
        fig = px.scatter(
            epic_summary,
            x='total_scans',
            y='signal_rate',
            size='signals',
            color='avg_adx',
            hover_name='pair_name',
            title="Signal Rate vs Scan Count",
            labels={
                'total_scans': 'Total Scans',
                'signal_rate': 'Signal Rate',
                'avg_adx': 'Avg ADX'
            },
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

    # Epic performance table
    with st.expander("View All Epic Performance Data"):
        display_cols = ['pair_name', 'total_scans', 'signals', 'buy_signals', 'sell_signals',
                       'signal_rate', 'avg_raw_confidence', 'avg_rsi', 'avg_adx',
                       'dominant_regime', 'dominant_volatility']
        available_cols = [c for c in display_cols if c in epic_summary.columns]

        display_df = epic_summary[available_cols].copy()
        # Convert signal_rate to percentage
        if 'signal_rate' in display_df.columns:
            display_df['signal_rate'] = (pd.to_numeric(display_df['signal_rate'], errors='coerce') * 100).round(2)

        # Round numeric columns
        for col in display_df.columns:
            if display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].round(3)

        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_indicator_comparison(indicator_stats: Dict):
    """Render indicator comparison between signals and non-signals"""
    st.subheader("Indicator Analysis: Signals vs Non-Signals")

    signals = indicator_stats.get('signals', {})
    non_signals = indicator_stats.get('non_signals', {})

    if not signals and not non_signals:
        st.info("No indicator comparison data available")
        return

    # Build comparison dataframe
    metrics = ['avg_rsi', 'avg_adx', 'avg_er', 'avg_atr', 'avg_smc_score',
               'avg_mtf_score', 'avg_entry_quality']

    comparison_data = []
    for metric in metrics:
        comparison_data.append({
            'Metric': metric.replace('avg_', '').upper(),
            'Signals': signals.get(metric, 0),
            'Non-Signals': non_signals.get(metric, 0),
            'Difference': signals.get(metric, 0) - non_signals.get(metric, 0)
        })

    comparison_df = pd.DataFrame(comparison_data)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            comparison_df,
            x='Metric',
            y=['Signals', 'Non-Signals'],
            title="Indicator Values: Signals vs Non-Signals",
            barmode='group',
            color_discrete_map={'Signals': '#28a745', 'Non-Signals': '#6c757d'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Show count comparison
        st.metric(
            "Signals Count",
            signals.get('count', 0),
            delta=f"{signals.get('count', 0) / (signals.get('count', 0) + non_signals.get('count', 1)) * 100:.2f}% of scans"
        )
        st.metric(
            "Non-Signals Count",
            non_signals.get('count', 0)
        )

        # Key differentiators
        st.markdown("**Key Indicator Differentiators:**")
        for _, row in comparison_df.iterrows():
            diff = row['Difference']
            color = '#28a745' if diff > 0 else '#dc3545' if diff < 0 else '#6c757d'
            arrow = '+' if diff > 0 else ''
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 4px 8px;
                        border-left: 3px solid {color}; margin: 2px 0; background: #f8f9fa;">
                <span>{row['Metric']}</span>
                <span style="color: {color}; font-weight: bold;">{arrow}{diff:.3f}</span>
            </div>
            """, unsafe_allow_html=True)


def _render_export_section(service: PerformanceSnapshotService, start_date, end_date):
    """Render data export section"""
    st.subheader("Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export All Scans", key="export_all"):
            with st.spinner("Fetching data..."):
                data = service.fetch_scan_snapshots(start_date, end_date)
                if not data.empty:
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download All Scans CSV",
                        data=csv,
                        file_name=f"scan_snapshots_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")

    with col2:
        if st.button("Export Signals Only", key="export_signals"):
            with st.spinner("Fetching data..."):
                data = service.fetch_signals_only(start_date, end_date)
                if not data.empty:
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download Signals CSV",
                        data=csv,
                        file_name=f"scan_signals_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No signals to export")

    with col3:
        if st.button("Export Rejections", key="export_rejections"):
            with st.spinner("Fetching data..."):
                data = service.fetch_rejections(start_date, end_date)
                if not data.empty:
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download Rejections CSV",
                        data=csv,
                        file_name=f"scan_rejections_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No rejections to export")
