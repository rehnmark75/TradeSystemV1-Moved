"""
Market Intelligence Tab Component

Renders the Market Intelligence analysis tab with:
- Comprehensive and signal-based data views
- Market regime distribution charts
- Session and volatility analysis
- Individual epic regime analysis
- Strategy recommendations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json

from services.market_intelligence_analytics_service import MarketIntelligenceAnalyticsService


def render_market_intelligence_tab():
    """Render the Market Intelligence analysis tab"""
    service = MarketIntelligenceAnalyticsService()

    st.header("Market Intelligence Analysis")
    st.markdown("*Analyze market conditions and regime patterns from comprehensive market scans*")

    # Data source selector and date range
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=7),
            key="mi_start_date"
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key="mi_end_date"
        )

    with col3:
        data_source = st.selectbox(
            "Data Source",
            options=["Comprehensive Scans", "Signal-Based", "Both"],
            index=0,
            help="Choose data source: Comprehensive = all scans, Signal-Based = only when signals generated"
        )

    with col4:
        if st.button("Refresh", key="mi_refresh"):
            st.rerun()

    # Fetch data based on source
    if data_source == "Comprehensive Scans":
        mi_data = service.fetch_comprehensive_intelligence(start_date, end_date)
        scan_data = mi_data
        signal_data = pd.DataFrame()
    elif data_source == "Signal-Based":
        mi_data = service.fetch_signal_based_intelligence(start_date, end_date)
        scan_data = pd.DataFrame()
        signal_data = mi_data
    else:  # Both
        scan_data = service.fetch_comprehensive_intelligence(start_date, end_date)
        signal_data = service.fetch_signal_based_intelligence(start_date, end_date)
        mi_data = scan_data  # Use comprehensive data as primary

    if mi_data.empty and scan_data.empty and signal_data.empty:
        st.warning("No market intelligence data found for the selected period.")
        st.info("Market intelligence data is captured automatically during forex scanner operations.")
        return

    # Market Intelligence Overview
    st.subheader("Market Intelligence Overview")

    # Show data source information
    if data_source == "Comprehensive Scans":
        st.info(f"Displaying comprehensive market intelligence from {len(mi_data)} scan cycles")
    elif data_source == "Signal-Based":
        st.info(f"Displaying market intelligence from {len(mi_data)} signals")
    else:
        st.info(f"Displaying combined data: {len(scan_data)} scans + {len(signal_data)} signals")

    # Render metrics based on data source
    _render_metrics(mi_data, data_source)

    # Market Regime Analysis
    if 'regime' in mi_data.columns:
        _render_regime_analysis(mi_data, data_source)

    # Session and Volatility Analysis
    _render_session_volatility_analysis(mi_data)

    # Strategy Performance Analysis
    _render_strategy_analysis(mi_data, data_source)

    # Intelligence Source Analysis
    if 'intelligence_source' in mi_data.columns:
        _render_intelligence_source_analysis(mi_data)

    # Enhanced Comprehensive Market Intelligence Visualizations
    if data_source == "Comprehensive Scans" or (data_source == "Both" and not scan_data.empty):
        _render_comprehensive_charts(mi_data)

    # Market Intelligence Search
    _render_search_filter(mi_data, data_source, start_date, end_date)


def _render_metrics(mi_data: pd.DataFrame, data_source: str):
    """Render overview metrics"""
    if data_source == "Comprehensive Scans" or data_source == "Both":
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Market Scans", len(mi_data))

        with col2:
            avg_epics = mi_data['epic_count'].mean() if 'epic_count' in mi_data.columns else 0
            st.metric("Avg Epics per Scan", f"{avg_epics:.1f}")

        with col3:
            unique_regimes = mi_data['regime'].nunique() if 'regime' in mi_data.columns else 0
            st.metric("Market Regimes Detected", unique_regimes)

        with col4:
            avg_confidence = mi_data['regime_confidence'].mean() if 'regime_confidence' in mi_data.columns else 0
            st.metric("Avg Regime Confidence", f"{avg_confidence:.1%}")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Signals with Intelligence", len(mi_data))

        with col2:
            unique_strategies = mi_data['strategy'].nunique() if 'strategy' in mi_data.columns else 0
            st.metric("Strategies Covered", unique_strategies)

        with col3:
            unique_regimes = mi_data['regime'].nunique() if 'regime' in mi_data.columns else 0
            st.metric("Market Regimes Detected", unique_regimes)

        with col4:
            avg_confidence = mi_data['regime_confidence'].mean() if 'regime_confidence' in mi_data.columns else 0
            st.metric("Avg Regime Confidence", f"{avg_confidence:.1%}")


def _render_regime_analysis(mi_data: pd.DataFrame, data_source: str):
    """Render market regime distribution analysis"""
    st.subheader("Market Regime Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Regime distribution pie chart
        regime_counts = mi_data['regime'].value_counts()
        fig_regime = px.pie(
            values=regime_counts.values,
            names=regime_counts.index,
            title="Market Regime Distribution"
        )
        st.plotly_chart(fig_regime, use_container_width=True)

    with col2:
        # Regime confidence by strategy/session
        if 'regime_confidence' in mi_data.columns:
            if 'strategy' in mi_data.columns:
                regime_conf_by_strategy = mi_data.groupby(['strategy', 'regime'])['regime_confidence'].mean().reset_index()
                fig_conf = px.bar(
                    regime_conf_by_strategy,
                    x='strategy',
                    y='regime_confidence',
                    color='regime',
                    title="Average Regime Confidence by Strategy",
                    labels={'regime_confidence': 'Confidence'}
                )
                fig_conf.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_conf, use_container_width=True)
            elif 'session' in mi_data.columns:
                regime_conf_by_session = mi_data.groupby(['session', 'regime'])['regime_confidence'].mean().reset_index()
                fig_conf = px.bar(
                    regime_conf_by_session,
                    x='session',
                    y='regime_confidence',
                    color='regime',
                    title="Average Regime Confidence by Session",
                    labels={'regime_confidence': 'Confidence'}
                )
                fig_conf.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_conf, use_container_width=True)


def _render_session_volatility_analysis(mi_data: pd.DataFrame):
    """Render session and volatility analysis"""
    col1, col2 = st.columns(2)

    with col1:
        if 'session' in mi_data.columns:
            st.subheader("Trading Session Analysis")
            session_counts = mi_data['session'].value_counts()
            session_df = pd.DataFrame({
                'session': session_counts.index,
                'count': session_counts.values
            })
            fig_session = px.bar(
                session_df,
                x='session',
                y='count',
                title="Signals by Trading Session",
                labels={'count': 'Signal Count', 'session': 'Session'}
            )
            st.plotly_chart(fig_session, use_container_width=True)

    with col2:
        if 'volatility_level' in mi_data.columns:
            st.subheader("Volatility Level Distribution")
            vol_counts = mi_data['volatility_level'].value_counts()
            vol_df = pd.DataFrame({
                'volatility': vol_counts.index,
                'count': vol_counts.values
            })
            fig_vol = px.bar(
                vol_df,
                x='volatility',
                y='count',
                title="Signals by Volatility Level",
                labels={'count': 'Signal Count', 'volatility': 'Volatility Level'},
                color='volatility',
                color_discrete_map={'high': 'red', 'medium': 'orange', 'low': 'green'}
            )
            st.plotly_chart(fig_vol, use_container_width=True)


def _render_strategy_analysis(mi_data: pd.DataFrame, data_source: str):
    """Render strategy performance analysis"""
    if data_source != "Comprehensive Scans":
        st.subheader("Strategy Performance by Market Conditions")

        if 'confidence_score' in mi_data.columns and 'regime' in mi_data.columns and 'strategy' in mi_data.columns:
            strategy_performance = mi_data.groupby(['strategy', 'regime']).agg({
                'confidence_score': ['mean', 'count'],
                'regime_confidence': 'mean' if 'regime_confidence' in mi_data.columns else lambda x: 0
            }).round(3)

            strategy_performance.columns = ['Avg_Signal_Confidence', 'Signal_Count', 'Avg_Regime_Confidence']
            strategy_performance = strategy_performance.reset_index()

            st.dataframe(strategy_performance, use_container_width=True)
    else:
        if 'recommended_strategy' in mi_data.columns and 'regime' in mi_data.columns:
            st.subheader("Recommended Strategy by Market Conditions")

            strategy_performance = mi_data.groupby(['recommended_strategy', 'regime']).agg({
                'regime_confidence': ['mean', 'count']
            }).round(3)

            strategy_performance.columns = ['Avg_Regime_Confidence', 'Scan_Count']
            strategy_performance = strategy_performance.reset_index()

            st.dataframe(strategy_performance, use_container_width=True)


def _render_intelligence_source_analysis(mi_data: pd.DataFrame):
    """Render intelligence source analysis"""
    st.subheader("Intelligence Source Analysis")

    col1, col2 = st.columns(2)

    with col1:
        source_counts = mi_data['intelligence_source'].value_counts()
        st.write("**Intelligence Sources:**")
        for source, count in source_counts.items():
            source_type = "Strategy-Specific" if "MarketIntelligenceEngine" in str(source) else "Universal Capture"
            st.write(f"{source_type}: {count} signals")

    with col2:
        if 'strategy' in mi_data.columns:
            source_by_strategy = mi_data.groupby('strategy')['intelligence_source'].value_counts().reset_index()
            fig_source = px.bar(
                source_by_strategy,
                x='strategy',
                y='count',
                color='intelligence_source',
                title="Intelligence Source by Strategy",
                labels={'count': 'Signal Count'}
            )
            st.plotly_chart(fig_source, use_container_width=True)
        elif 'session' in mi_data.columns:
            source_by_session = mi_data.groupby('session')['intelligence_source'].value_counts().reset_index()
            fig_source = px.bar(
                source_by_session,
                x='session',
                y='count',
                color='intelligence_source',
                title="Intelligence Source by Session",
                labels={'count': 'Scan Count'}
            )
            st.plotly_chart(fig_source, use_container_width=True)


def _render_comprehensive_charts(mi_data: pd.DataFrame):
    """Render enhanced charts for comprehensive market intelligence data"""
    st.subheader("Enhanced Market Intelligence Analytics")
    st.markdown("*Advanced visualizations from comprehensive market scan data*")

    # Time series analysis
    if 'scan_timestamp' in mi_data.columns:
        st.subheader("Market Regime Evolution Over Time")

        mi_data_time = mi_data.copy()
        mi_data_time['scan_timestamp'] = pd.to_datetime(mi_data_time['scan_timestamp'])
        mi_data_time = mi_data_time.sort_values('scan_timestamp')

        fig_timeline = px.scatter(
            mi_data_time,
            x='scan_timestamp',
            y='regime_confidence',
            color='regime',
            size='epic_count',
            title="Market Regime Confidence Timeline",
            labels={
                'scan_timestamp': 'Time',
                'regime_confidence': 'Confidence',
                'epic_count': 'Epics Analyzed'
            },
            hover_data=['session', 'market_bias'] if 'market_bias' in mi_data_time.columns else ['session']
        )
        fig_timeline.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Market bias and risk sentiment analysis
    col1, col2 = st.columns(2)

    with col1:
        if 'market_bias' in mi_data.columns:
            st.subheader("Market Bias Distribution")
            bias_counts = mi_data['market_bias'].value_counts()
            fig_bias = px.pie(
                values=bias_counts.values,
                names=bias_counts.index,
                title="Market Bias Distribution",
                color_discrete_map={
                    'bullish': '#28a745',
                    'bearish': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_bias, use_container_width=True)

    with col2:
        if 'risk_sentiment' in mi_data.columns:
            st.subheader("Risk Sentiment Analysis")
            risk_counts = mi_data['risk_sentiment'].value_counts()
            fig_risk = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Sentiment Distribution",
                labels={'x': 'Risk Sentiment', 'y': 'Count'},
                color=risk_counts.index,
                color_discrete_map={
                    'risk_on': '#28a745',
                    'risk_off': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_risk, use_container_width=True)

    # Advanced regime analysis with individual scores
    _render_regime_scores(mi_data)

    # Market strength indicators
    _render_market_strength(mi_data)

    # Session vs Regime Matrix
    _render_session_regime_matrix(mi_data)

    # Individual Epic Regimes Analysis
    _render_individual_epic_regimes(mi_data)

    # Summary statistics
    _render_summary_stats(mi_data)


def _render_regime_scores(mi_data: pd.DataFrame):
    """Render detailed regime score analysis"""
    regime_score_columns = [
        'regime_trending_score', 'regime_ranging_score', 'regime_breakout_score',
        'regime_reversal_score', 'regime_high_vol_score', 'regime_low_vol_score'
    ]

    available_score_columns = [col for col in regime_score_columns if col in mi_data.columns]

    if available_score_columns:
        st.subheader("Detailed Regime Score Analysis")

        regime_scores_data = []
        for _, row in mi_data.iterrows():
            timestamp = row.get('scan_timestamp', 'Unknown')
            for score_col in available_score_columns:
                if pd.notnull(row.get(score_col)):
                    regime_scores_data.append({
                        'timestamp': timestamp,
                        'regime_type': score_col.replace('regime_', '').replace('_score', ''),
                        'score': row[score_col],
                        'dominant_regime': row.get('regime', 'unknown')
                    })

        if regime_scores_data:
            scores_df = pd.DataFrame(regime_scores_data)

            fig_scores = px.box(
                scores_df,
                x='regime_type',
                y='score',
                color='dominant_regime',
                title="Regime Score Distribution by Dominant Regime",
                labels={'score': 'Score', 'regime_type': 'Regime Type'}
            )
            fig_scores.update_layout(yaxis_tickformat='.2f')
            st.plotly_chart(fig_scores, use_container_width=True)


def _render_market_strength(mi_data: pd.DataFrame):
    """Render market strength indicators"""
    strength_columns = ['average_trend_strength', 'average_volatility']
    available_strength_columns = [col for col in strength_columns if col in mi_data.columns]

    if available_strength_columns:
        st.subheader("Market Strength Indicators")

        col1, col2 = st.columns(2)

        with col1:
            if 'average_trend_strength' in mi_data.columns and 'session' in mi_data.columns:
                strength_by_session = mi_data.groupby('session')['average_trend_strength'].mean().reset_index()
                fig_strength = px.bar(
                    strength_by_session,
                    x='session',
                    y='average_trend_strength',
                    title="Average Trend Strength by Session",
                    labels={'average_trend_strength': 'Trend Strength'}
                )
                fig_strength.update_layout(yaxis_tickformat='.2f')
                st.plotly_chart(fig_strength, use_container_width=True)

        with col2:
            if 'average_volatility' in mi_data.columns:
                fig_vol_dist = px.histogram(
                    mi_data,
                    x='average_volatility',
                    nbins=20,
                    title="Market Volatility Distribution",
                    labels={'average_volatility': 'Volatility', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_vol_dist, use_container_width=True)


def _render_session_regime_matrix(mi_data: pd.DataFrame):
    """Render session vs regime matrix"""
    if 'session' in mi_data.columns and 'regime' in mi_data.columns:
        st.subheader("Session vs Regime Matrix")

        session_regime_matrix = pd.crosstab(mi_data['session'], mi_data['regime'])
        session_regime_pct = session_regime_matrix.div(session_regime_matrix.sum(axis=1), axis=0) * 100

        fig_matrix = px.imshow(
            session_regime_pct.values,
            labels=dict(x="Market Regime", y="Trading Session", color="Percentage"),
            x=session_regime_pct.columns,
            y=session_regime_pct.index,
            title="Session vs Regime Distribution (%)",
            color_continuous_scale="RdYlGn"
        )
        fig_matrix.update_layout(width=700, height=400)
        st.plotly_chart(fig_matrix, use_container_width=True)


def _render_individual_epic_regimes(mi_data: pd.DataFrame):
    """Render individual epic regime analysis"""
    if mi_data.empty:
        return

    if 'individual_epic_regimes' not in mi_data.columns:
        st.info("Individual epic regime analysis will be available after the next market intelligence scan with the enhanced system.")
        st.markdown("*This feature shows detailed regime analysis for each currency pair individually*")
        return

    st.subheader("Individual Epic Regime Analysis")
    st.markdown("*Detailed regime analysis for each currency pair across all market scans*")

    try:
        # Process individual epic regimes data
        all_epic_regimes = {}
        regime_timeline = []

        for idx, row in mi_data.iterrows():
            if pd.notna(row['individual_epic_regimes']):
                try:
                    epic_regimes = json.loads(row['individual_epic_regimes']) if isinstance(row['individual_epic_regimes'], str) else row['individual_epic_regimes']
                    timestamp = row['scan_timestamp']

                    for epic, data in epic_regimes.items():
                        if epic not in all_epic_regimes:
                            all_epic_regimes[epic] = {'regimes': [], 'confidences': [], 'timestamps': []}

                        all_epic_regimes[epic]['regimes'].append(data.get('regime', 'unknown'))
                        all_epic_regimes[epic]['confidences'].append(data.get('confidence', 0.5))
                        all_epic_regimes[epic]['timestamps'].append(timestamp)

                        regime_timeline.append({
                            'epic': epic,
                            'regime': data.get('regime', 'unknown'),
                            'confidence': data.get('confidence', 0.5),
                            'timestamp': timestamp,
                            'scan_id': row['id']
                        })
                except (json.JSONDecodeError, TypeError):
                    continue

        if not all_epic_regimes:
            st.info("Individual epic regime data not available for selected period.")
            return

        # Epic regime distribution overview
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Current Epic Regimes")

            latest_regimes = {}
            for epic, data in all_epic_regimes.items():
                if data['regimes']:
                    latest_regimes[epic] = {
                        'regime': data['regimes'][-1],
                        'confidence': data['confidences'][-1]
                    }

            for epic, regime_data in sorted(latest_regimes.items()):
                regime = regime_data['regime']
                confidence = regime_data['confidence']

                color_map = {
                    'trending': '#28a745',
                    'ranging': '#007bff',
                    'breakout': '#ff6b6b',
                    'reversal': '#ffc107',
                    'low_volatility': '#6c757d',
                    'high_volatility': '#e83e8c'
                }
                color = color_map.get(regime, '#6c757d')

                st.markdown(f"""
                <div style="
                    background: {color};
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    margin: 4px 0;
                    display: inline-block;
                    font-weight: bold;
                    min-width: 120px;
                    text-align: center;
                ">
                    {epic}: {regime.upper()} ({confidence:.1%})
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if latest_regimes:
                regime_counts = {}
                for data in latest_regimes.values():
                    regime = data['regime']
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1

                fig_pie = px.pie(
                    values=list(regime_counts.values()),
                    names=list(regime_counts.keys()),
                    title="Current Regime Distribution",
                    color_discrete_map={
                        'trending': '#28a745',
                        'ranging': '#007bff',
                        'breakout': '#ff6b6b',
                        'reversal': '#ffc107',
                        'low_volatility': '#6c757d',
                        'high_volatility': '#e83e8c'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Epic regime timeline
        if regime_timeline:
            st.subheader("Epic Regime Evolution Timeline")

            timeline_df = pd.DataFrame(regime_timeline)
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])

            selected_epics = st.multiselect(
                "Select Epics to View",
                options=sorted(timeline_df['epic'].unique()),
                default=sorted(timeline_df['epic'].unique())[:5],
                help="Select which currency pairs to display in the timeline"
            )

            if selected_epics:
                filtered_timeline = timeline_df[timeline_df['epic'].isin(selected_epics)]

                fig_timeline = px.scatter(
                    filtered_timeline,
                    x='timestamp',
                    y='epic',
                    color='regime',
                    size='confidence',
                    title="Epic Regime Timeline",
                    labels={
                        'timestamp': 'Time',
                        'epic': 'Currency Pair',
                        'confidence': 'Confidence',
                        'regime': 'Market Regime'
                    },
                    hover_data=['confidence'],
                    color_discrete_map={
                        'trending': '#28a745',
                        'ranging': '#007bff',
                        'breakout': '#ff6b6b',
                        'reversal': '#ffc107',
                        'low_volatility': '#6c757d',
                        'high_volatility': '#e83e8c'
                    }
                )
                fig_timeline.update_layout(height=400)
                st.plotly_chart(fig_timeline, use_container_width=True)

        # Epic-specific strategy recommendations
        _render_strategy_recommendations(latest_regimes)

    except Exception as e:
        st.error(f"Error rendering individual epic regimes: {e}")


def _render_strategy_recommendations(latest_regimes: dict):
    """Render epic-specific strategy recommendations"""
    st.subheader("Epic-Specific Strategy Recommendations")

    strategy_recommendations = {
        'trending': {
            'strategy': 'Trend Following',
            'description': 'Use momentum strategies, trend confirmations, and trailing stops',
            'icon': '+',
            'color': '#28a745'
        },
        'ranging': {
            'strategy': 'Range Trading',
            'description': 'Trade support/resistance bounces, use mean reversion strategies',
            'icon': '=',
            'color': '#007bff'
        },
        'breakout': {
            'strategy': 'Breakout Trading',
            'description': 'Monitor for volume confirmation, use wider stops, expect volatility',
            'icon': '>',
            'color': '#ff6b6b'
        },
        'reversal': {
            'strategy': 'Reversal Trading',
            'description': 'Look for reversal patterns, use conservative position sizing',
            'icon': '<',
            'color': '#ffc107'
        },
        'low_volatility': {
            'strategy': 'Conservative Trading',
            'description': 'Use smaller position sizes, tighter stops, range strategies',
            'icon': '-',
            'color': '#6c757d'
        },
        'high_volatility': {
            'strategy': 'Volatility Trading',
            'description': 'Reduce position size, use wider stops, consider straddles',
            'icon': '!',
            'color': '#e83e8c'
        }
    }

    regime_groups = {}
    for epic, regime_data in latest_regimes.items():
        regime = regime_data['regime']
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(epic)

    for regime, epics in regime_groups.items():
        if regime in strategy_recommendations:
            rec = strategy_recommendations[regime]

            st.markdown(f"""
            <div style="
                border-left: 4px solid {rec['color']};
                background: #f8f9fa;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 4px;
            ">
                <h4 style="margin: 0; color: {rec['color']};">
                    {rec['icon']} {rec['strategy']} - {regime.upper()} Regime
                </h4>
                <p style="margin: 0.5rem 0; color: #6c757d;">
                    <strong>Pairs:</strong> {', '.join(epics)}
                </p>
                <p style="margin: 0; color: #333;">
                    {rec['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)


def _render_summary_stats(mi_data: pd.DataFrame):
    """Render summary statistics table"""
    if len(mi_data) > 0:
        st.subheader("Comprehensive Market Intelligence Summary")

        summary_stats = {
            'Total Scan Cycles': len(mi_data),
            'Date Range': f"{mi_data['scan_timestamp'].min().date()} to {mi_data['scan_timestamp'].max().date()}" if 'scan_timestamp' in mi_data.columns else "N/A",
            'Most Common Regime': mi_data['regime'].mode().iloc[0] if 'regime' in mi_data.columns and len(mi_data['regime'].mode()) > 0 else "N/A",
            'Average Confidence': f"{mi_data['regime_confidence'].mean():.1%}" if 'regime_confidence' in mi_data.columns else "N/A",
            'Most Active Session': mi_data['session'].mode().iloc[0] if 'session' in mi_data.columns and len(mi_data['session'].mode()) > 0 else "N/A",
            'Average Epics per Scan': f"{mi_data['epic_count'].mean():.1f}" if 'epic_count' in mi_data.columns else "N/A"
        }

        if 'average_trend_strength' in mi_data.columns:
            summary_stats['Average Trend Strength'] = f"{mi_data['average_trend_strength'].mean():.3f}"
        if 'market_bias' in mi_data.columns:
            summary_stats['Most Common Bias'] = mi_data['market_bias'].mode().iloc[0] if len(mi_data['market_bias'].mode()) > 0 else "N/A"

        col1, col2, col3 = st.columns(3)

        stats_items = list(summary_stats.items())
        for i, (key, value) in enumerate(stats_items):
            col = [col1, col2, col3][i % 3]
            with col:
                st.metric(key, value)


def _render_search_filter(mi_data: pd.DataFrame, data_source: str, start_date, end_date):
    """Render market intelligence search and filter"""
    st.subheader("Market Intelligence Search")

    col1, col2, col3 = st.columns(3)

    with col1:
        if 'strategy' in mi_data.columns:
            search_strategy = st.selectbox(
                "Filter by Strategy",
                options=['All'] + sorted(mi_data['strategy'].unique().tolist()),
                key="mi_search_strategy"
            )
        elif 'recommended_strategy' in mi_data.columns:
            search_strategy = st.selectbox(
                "Filter by Recommended Strategy",
                options=['All'] + sorted(mi_data['recommended_strategy'].unique().tolist()),
                key="mi_search_strategy"
            )
        else:
            search_strategy = 'All'
            st.info("Strategy filtering not available")

    with col2:
        search_regime = st.selectbox(
            "Filter by Regime",
            options=['All'] + sorted(mi_data['regime'].unique().tolist()) if 'regime' in mi_data.columns else ['All'],
            key="mi_search_regime"
        )

    with col3:
        search_session = st.selectbox(
            "Filter by Session",
            options=['All'] + sorted(mi_data['session'].unique().tolist()) if 'session' in mi_data.columns else ['All'],
            key="mi_search_session"
        )

    # Apply filters
    filtered_data = mi_data.copy()

    if search_strategy != 'All':
        if 'strategy' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['strategy'] == search_strategy]
        elif 'recommended_strategy' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['recommended_strategy'] == search_strategy]

    if search_regime != 'All' and 'regime' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['regime'] == search_regime]
    if search_session != 'All' and 'session' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['session'] == search_session]

    if data_source == "Comprehensive Scans":
        st.write(f"**Showing {len(filtered_data)} scan cycles matching filters:**")
    else:
        st.write(f"**Showing {len(filtered_data)} signals matching filters:**")

    if not filtered_data.empty:
        if data_source == "Comprehensive Scans" or 'scan_timestamp' in filtered_data.columns:
            display_columns = ['scan_timestamp', 'scan_cycle_id', 'epic_count']
            if 'regime' in filtered_data.columns:
                display_columns.extend(['regime', 'regime_confidence'])
            if 'session' in filtered_data.columns:
                display_columns.append('session')
            if 'market_bias' in filtered_data.columns:
                display_columns.append('market_bias')
            if 'risk_sentiment' in filtered_data.columns:
                display_columns.append('risk_sentiment')
            if 'recommended_strategy' in filtered_data.columns:
                display_columns.append('recommended_strategy')
            sort_column = 'scan_timestamp'
        else:
            display_columns = ['alert_timestamp', 'epic', 'strategy', 'signal_type', 'confidence_score']
            if 'regime' in filtered_data.columns:
                display_columns.extend(['regime', 'regime_confidence'])
            if 'session' in filtered_data.columns:
                display_columns.append('session')
            if 'volatility_level' in filtered_data.columns:
                display_columns.append('volatility_level')
            sort_column = 'alert_timestamp'

        available_columns = [col for col in display_columns if col in filtered_data.columns]

        if sort_column in filtered_data.columns:
            sorted_data = filtered_data[available_columns].sort_values(sort_column, ascending=False)
        else:
            sorted_data = filtered_data[available_columns]

        st.dataframe(sorted_data, use_container_width=True)

        # Export functionality
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Export Filtered Data as CSV",
            data=csv,
            file_name=f"market_intelligence_{start_date}_{end_date}.csv",
            mime="text/csv"
        )
    else:
        st.info("No signals match the selected filters.")
