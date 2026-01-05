"""
SMC Rejections Tab Component

Renders the SMC Simple strategy rejection analysis tab with multiple sub-tabs:
- Stage Breakdown
- Outcome Analysis
- S/R Path Blocking
- Time Analysis
- Market Context
- Near-Misses
- Scanner Efficiency
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from typing import Dict, Any

from services.rejection_analytics_service import RejectionAnalyticsService
from services.db_utils import get_psycopg2_connection


def render_smc_rejections_tab():
    """Render SMC Simple Rejection Analysis tab"""
    service = RejectionAnalyticsService()

    # Header with refresh button
    header_col1, header_col2 = st.columns([6, 1])
    with header_col1:
        st.header("SMC Simple Rejection Analysis")
    with header_col2:
        if st.button("Refresh", key="smc_rejections_refresh", help="Refresh rejection data"):
            st.rerun()

    st.markdown("Analyze why SMC Simple strategy signals were rejected to improve strategy parameters")

    # Get filter options from service (cached)
    filter_options = service.get_smc_filter_options()

    if not filter_options.get('table_exists', True):
        st.warning("SMC Rejections table not yet created. Run the database migration:")
        st.code("""
docker exec -it postgres psql -U postgres -d trading -f /path/to/create_smc_simple_rejections_table.sql
        """)
        return

    stages = filter_options['stages']
    pairs = filter_options['pairs']
    sessions = filter_options['sessions']

    # Filters row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        days_filter = st.selectbox("Time Period", [3, 7, 14, 30, 60, 90], index=0, key="smc_rej_days")
    with col2:
        stage_filter = st.selectbox("Rejection Stage", stages, key="smc_rej_stage")
    with col3:
        pair_filter = st.selectbox("Pair", pairs, key="smc_rej_pair")
    with col4:
        session_filter = st.selectbox("Session", sessions, key="smc_rej_session")

    # Fetch statistics from service (cached)
    stats = service.fetch_smc_rejection_stats(days_filter)

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rejections", f"{stats.get('total', 0):,}")
    with col2:
        st.metric("Unique Pairs", stats.get('unique_pairs', 0))
    with col3:
        top_stage = max(stats.get('by_stage', {'N/A': 0}).items(), key=lambda x: x[1])[0] if stats.get('by_stage') else 'N/A'
        st.metric("Top Rejection Stage", top_stage)
    with col4:
        st.metric("Near-Misses", stats.get('near_misses', 0), help="Signals that reached confidence stage but were rejected")
    with col5:
        st.metric("SMC Conflicts", stats.get('smc_conflicts', 0), help="Signals rejected due to SMC data conflicts")

    st.markdown("---")

    # Sub-tabs for different analysis views
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6, sub_tab7, sub_tab8 = st.tabs([
        "Stage Breakdown", "SMC Conflicts", "Outcome Analysis", "S/R Path Blocking", "Time Analysis", "Market Context", "Near-Misses", "Scanner Efficiency"
    ])

    # Fetch data from service (cached)
    df = service.fetch_smc_rejections(days_filter, stage_filter, pair_filter, session_filter)

    if df.empty:
        st.info("No rejections found for the selected filters.")
        return

    with sub_tab1:
        _render_stage_breakdown(df, stats)

    with sub_tab2:
        _render_smc_conflict_analysis(service, days_filter)

    with sub_tab3:
        _render_rejection_outcome_analysis(days_filter)

    with sub_tab4:
        _render_sr_path_blocking(df, days_filter)

    with sub_tab5:
        _render_time_analysis(df)

    with sub_tab6:
        _render_market_context(df)

    with sub_tab7:
        _render_near_misses(df, days_filter)

    with sub_tab8:
        _render_scanner_efficiency(days_filter)


def _render_stage_breakdown(df: pd.DataFrame, stats: dict):
    """Render stage breakdown sub-tab"""
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
            color_discrete_sequence=px.colors.qualitative.Set2
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
            color_continuous_scale='Reds'
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


def _render_rejection_outcome_analysis(days_filter: int):
    """Render rejection outcome analysis sub-tab."""
    st.subheader("Rejection Outcome Analysis")

    st.info("""
    **What is Rejection Outcome Analysis?**

    This analysis tracks what would have happened if rejected signals were actually executed.
    Using fixed SL=9 pips and TP=15 pips, we monitor price movement after each rejection to determine:
    - **Would-be Winners**: Signals that would have hit TP before SL
    - **Would-be Losers**: Signals that would have hit SL before TP

    This helps identify if rejection filters are too aggressive (missing profitable trades) or
    working correctly (filtering out losing trades).
    """)

    # Fetch data from FastAPI endpoint
    fastapi_base_url = "http://fastapi-dev:8000"
    headers = {
        "X-APIM-Gateway": "verified",
        "X-API-KEY": "436abe054a074894a0517e5172f0e5b6"
    }

    try:
        # Fetch summary
        summary_response = requests.get(
            f"{fastapi_base_url}/api/rejection-outcomes/summary",
            params={"days": days_filter},
            headers=headers,
            timeout=30
        )

        if summary_response.status_code != 200:
            st.warning("Could not fetch outcome data. Run the rejection outcome analyzer first.")
            st.code("""
# Run the analyzer to populate outcome data:
docker exec -it task-worker python /app/forex_scanner/monitoring/rejection_outcome_analyzer.py --days 7
            """)
            return

        summary = summary_response.json()

        if summary.get('total_analyzed', 0) == 0:
            st.warning("No outcome data available yet. Run the analyzer to populate data.")
            st.code("""
# First, run the database migration:
docker exec postgres psql -U postgres -d forex -f /app/forex_scanner/migrations/create_smc_rejection_outcomes_table.sql

# Then run the analyzer:
docker exec -it task-worker python /app/forex_scanner/monitoring/rejection_outcome_analyzer.py --days 7
            """)
            return

        # Summary metrics row
        st.markdown("### Overall Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Analyzed", f"{summary.get('total_analyzed', 0):,}")

        with col2:
            win_rate = summary.get('would_be_win_rate', 0)
            delta_color = "normal" if win_rate >= 50 else "inverse"
            st.metric(
                "Would-Be Win Rate",
                f"{win_rate:.1f}%",
                delta=f"{win_rate - 50:.1f}% vs break-even" if win_rate else None,
                delta_color=delta_color
            )

        with col3:
            st.metric(
                "Would-Be Winners",
                summary.get('winners', 0),
                help="Rejected signals that would have hit TP"
            )

        with col4:
            missed_pips = summary.get('total_missed_pips', 0)
            st.metric(
                "Missed Profit",
                f"{missed_pips:.0f} pips",
                help="Potential profit from would-be winners"
            )

        with col5:
            avoided_loss = summary.get('avoided_loss_pips', 0)
            st.metric(
                "Avoided Loss",
                f"{avoided_loss:.0f} pips",
                help="Loss avoided by rejecting would-be losers"
            )

        st.markdown("---")

        # Win rate by stage analysis
        st.markdown("### Win Rate by Rejection Stage")

        stage_response = requests.get(
            f"{fastapi_base_url}/api/rejection-outcomes/win-rate-by-stage",
            params={"days": days_filter},
            headers=headers,
            timeout=30
        )

        if stage_response.status_code == 200:
            stage_data = stage_response.json()

            if stage_data:
                stage_df = pd.DataFrame(stage_data)

                # Bar chart
                fig = px.bar(
                    stage_df,
                    x='rejection_stage',
                    y='would_be_win_rate',
                    color='would_be_win_rate',
                    color_continuous_scale='RdYlGn',
                    title='Would-Be Win Rate by Rejection Stage',
                    labels={'would_be_win_rate': 'Win Rate (%)', 'rejection_stage': 'Stage'}
                )
                fig.add_hline(y=50, line_dash="dash", line_color="gray",
                              annotation_text="Break-even (50%)")
                fig.update_layout(coloraxis_colorbar=dict(title="Win Rate %"))
                st.plotly_chart(fig, use_container_width=True)

                # Interpretation
                st.markdown("### Stage Interpretation")

                for _, row in stage_df.iterrows():
                    stage = row.get('rejection_stage', 'UNKNOWN')
                    win_rate = row.get('would_be_win_rate', 0) or 0
                    total = row.get('total_analyzed', 0)
                    missed = row.get('missed_profit_pips', 0) or 0
                    avoided = row.get('avoided_loss_pips', 0) or 0

                    if win_rate > 60:
                        st.warning(
                            f"**{stage}**: {win_rate:.0f}% win rate ({total} signals) - "
                            f"Missing {missed:.0f} pips of profit. Consider relaxing this filter."
                        )
                    elif win_rate < 40:
                        st.success(
                            f"**{stage}**: {win_rate:.0f}% win rate ({total} signals) - "
                            f"Correctly avoided {avoided:.0f} pips of loss. Keep current settings."
                        )
                    else:
                        st.info(
                            f"**{stage}**: {win_rate:.0f}% win rate ({total} signals) - "
                            f"Neutral performance. Net: {missed - avoided:.0f} pips."
                        )

                # Data table
                st.markdown("### Detailed Stage Metrics")
                display_cols = [
                    'rejection_stage', 'total_analyzed', 'would_be_winners',
                    'would_be_losers', 'would_be_win_rate', 'missed_profit_pips',
                    'avoided_loss_pips', 'avg_mfe_pips', 'avg_mae_pips'
                ]
                available_cols = [c for c in display_cols if c in stage_df.columns]
                if available_cols:
                    st.dataframe(
                        stage_df[available_cols].rename(columns={
                            'rejection_stage': 'Stage',
                            'total_analyzed': 'Total',
                            'would_be_winners': 'Winners',
                            'would_be_losers': 'Losers',
                            'would_be_win_rate': 'Win Rate %',
                            'missed_profit_pips': 'Missed Pips',
                            'avoided_loss_pips': 'Avoided Pips',
                            'avg_mfe_pips': 'Avg MFE',
                            'avg_mae_pips': 'Avg MAE'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

        st.markdown("---")

        # Parameter suggestions
        st.markdown("### Parameter Suggestions")

        suggestions_response = requests.get(
            f"{fastapi_base_url}/api/rejection-outcomes/parameter-suggestions",
            params={"days": days_filter},
            headers=headers,
            timeout=30
        )

        if suggestions_response.status_code == 200:
            suggestions = suggestions_response.json()

            if suggestions.get('overall_assessment'):
                st.markdown(f"**Overall Assessment:** {suggestions['overall_assessment']}")

            if suggestions.get('stage_adjustments'):
                with st.expander("Stage-Specific Recommendations", expanded=True):
                    for adj in suggestions['stage_adjustments']:
                        issue = adj.get('issue', '')
                        if issue == 'TOO_AGGRESSIVE':
                            st.warning(f"{adj.get('recommendation', '')}")
                        elif issue == 'WORKING_WELL':
                            st.success(f"{adj.get('recommendation', '')}")
                        else:
                            st.info(f"{adj.get('recommendation', '')}")

            if suggestions.get('session_patterns'):
                with st.expander("Session Patterns"):
                    session_df = pd.DataFrame(suggestions['session_patterns'])
                    if not session_df.empty:
                        st.dataframe(session_df, use_container_width=True, hide_index=True)

            if suggestions.get('pair_insights'):
                _render_pair_insights(suggestions['pair_insights'])

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to FastAPI backend: {e}")
        st.info("Make sure the fastapi-dev container is running.")
    except Exception as e:
        st.error(f"Error rendering outcome analysis: {e}")


def _render_pair_insights(pair_insights: list):
    """Render per-pair recommendations."""
    st.markdown("---")
    st.markdown("### Per-Pair Recommendations")

    pair_df = pd.DataFrame(pair_insights)

    if pair_df.empty:
        return

    # Create color-coded bar chart
    fig = px.bar(
        pair_df,
        x='pair',
        y='win_rate',
        color='status',
        color_discrete_map={
            'TOO_AGGRESSIVE': '#ff6b6b',
            'WORKING_WELL': '#51cf66',
            'NEUTRAL': '#ffd43b'
        },
        title='Would-Be Win Rate by Currency Pair',
        labels={
            'win_rate': 'Win Rate (%)',
            'pair': 'Currency Pair',
            'status': 'Filter Status'
        },
        hover_data=['total_analyzed', 'missed_profit_pips', 'avoided_loss_pips', 'net_pips']
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Break-even (50%)")
    fig.add_hline(y=60, line_dash="dot", line_color="red", annotation_text="Too Aggressive (60%)")
    fig.add_hline(y=40, line_dash="dot", line_color="green", annotation_text="Working Well (40%)")
    st.plotly_chart(fig, use_container_width=True)

    # Pair-specific recommendations
    st.markdown("#### Pair-Specific Action Items")

    pair_df_sorted = pair_df.sort_values('net_pips', ascending=False)

    for _, row in pair_df_sorted.iterrows():
        pair = row.get('pair', 'UNKNOWN')
        epic = row.get('epic', '')
        win_rate = row.get('win_rate', 0) or 0
        total = row.get('total_analyzed', 0)
        status = row.get('status', 'NEUTRAL')
        recommendation = row.get('recommendation', '')
        net_pips = row.get('net_pips', 0) or 0

        if status == 'TOO_AGGRESSIVE':
            st.warning(
                f"**{pair}** ({epic}): {win_rate:.0f}% win rate from {total} rejections. "
                f"Net missed: {net_pips:.0f} pips. {recommendation}"
            )
        elif status == 'WORKING_WELL':
            st.success(
                f"**{pair}** ({epic}): {win_rate:.0f}% win rate from {total} rejections. "
                f"Net saved: {abs(net_pips):.0f} pips. {recommendation}"
            )
        else:
            st.info(
                f"**{pair}** ({epic}): {win_rate:.0f}% win rate from {total} rejections. "
                f"Net impact: {net_pips:.0f} pips. {recommendation}"
            )

    # Detailed pair metrics table
    with st.expander("Detailed Pair Metrics"):
        display_cols = [
            'pair', 'epic', 'total_analyzed', 'win_rate',
            'missed_profit_pips', 'avoided_loss_pips', 'net_pips', 'status'
        ]
        available_cols = [c for c in display_cols if c in pair_df.columns]
        if available_cols:
            display_df = pair_df[available_cols].rename(columns={
                'pair': 'Pair',
                'epic': 'Epic',
                'total_analyzed': 'Total',
                'win_rate': 'Win Rate %',
                'missed_profit_pips': 'Missed Pips',
                'avoided_loss_pips': 'Avoided Pips',
                'net_pips': 'Net Pips',
                'status': 'Status'
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_sr_path_blocking(df: pd.DataFrame, days_filter: int):
    """Render S/R Path Blocking analysis sub-tab"""
    st.subheader("S/R Path Blocking Analysis")

    st.info("""
    **What is S/R Path Blocking?**

    This analysis shows trades rejected because a Support/Resistance level was blocking
    the path from entry to take profit target. For example, if you're buying at 1.2500
    with a TP at 1.2560, but there's resistance at 1.2510, that resistance blocks 83%
    of your profit path - a high-risk trade that should be avoided.

    **Critical threshold:** S/R blocking >75% of path = Auto-reject
    **Warning threshold:** S/R blocking >50% of path = Flagged for review
    """)

    # Filter for SR_PATH_BLOCKED rejections
    sr_blocked_df = df[df['rejection_stage'] == 'SR_PATH_BLOCKED'].copy() if 'rejection_stage' in df.columns else pd.DataFrame()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sr_blocked_count = len(sr_blocked_df)
        total_rejections = len(df)
        pct_of_total = (sr_blocked_count / total_rejections * 100) if total_rejections > 0 else 0
        st.metric(
            "S/R Path Blocked Rejections",
            f"{sr_blocked_count:,}",
            f"{pct_of_total:.1f}% of all rejections"
        )

    with col2:
        if 'sr_path_blocked_pct' in sr_blocked_df.columns and not sr_blocked_df.empty:
            avg_blocked = sr_blocked_df['sr_path_blocked_pct'].mean()
            st.metric("Avg Path Blocked %", f"{avg_blocked:.1f}%")
        else:
            st.metric("Avg Path Blocked %", "N/A")

    with col3:
        if not sr_blocked_df.empty and 'pair' in sr_blocked_df.columns:
            affected_pairs = sr_blocked_df['pair'].nunique()
            st.metric("Affected Pairs", affected_pairs)
        else:
            st.metric("Affected Pairs", "0")

    with col4:
        if not sr_blocked_df.empty and 'target_distance_pips' in sr_blocked_df.columns:
            avg_target = sr_blocked_df['target_distance_pips'].mean()
            st.metric("Avg Target Distance", f"{avg_target:.1f} pips")
        else:
            st.metric("Avg Target Distance", "N/A")

    if sr_blocked_df.empty:
        st.info("No S/R path blocking rejections found for the selected period. This is good - it means no trades were rejected due to S/R blocking the path to target!")

        # Show alternative: Check if there are S/R related issues in other stages
        sr_related = df[df['rejection_reason'].str.contains('S/R|support|resistance|level', case=False, na=False)]
        if not sr_related.empty:
            st.warning(f"However, {len(sr_related)} rejections mention S/R in other stages:")
            st.dataframe(
                sr_related[['scan_timestamp', 'pair', 'rejection_stage', 'rejection_reason']].head(10),
                use_container_width=True,
                hide_index=True
            )
        return

    st.markdown("---")

    # Breakdown by pair
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### S/R Blocking by Currency Pair")
        if 'pair' in sr_blocked_df.columns:
            pair_counts = sr_blocked_df['pair'].value_counts().reset_index()
            pair_counts.columns = ['Pair', 'Count']

            fig_pair = px.bar(
                pair_counts,
                x='Pair',
                y='Count',
                title="S/R Path Blocking Rejections by Pair",
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_pair, use_container_width=True)

    with col2:
        st.markdown("#### S/R Blocking by Type")
        if 'sr_blocking_type' in sr_blocked_df.columns and sr_blocked_df['sr_blocking_type'].notna().any():
            type_counts = sr_blocked_df['sr_blocking_type'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']

            fig_type = px.pie(
                type_counts,
                values='Count',
                names='Type',
                title="Blocking by S/R Type (Support vs Resistance)",
                color_discrete_map={'resistance': '#f44336', 'support': '#4caf50'}
            )
            st.plotly_chart(fig_type, use_container_width=True)

    # Path blocked percentage distribution
    if 'sr_path_blocked_pct' in sr_blocked_df.columns and sr_blocked_df['sr_path_blocked_pct'].notna().any():
        st.markdown("#### Path Blocked Percentage Distribution")
        fig_hist = px.histogram(
            sr_blocked_df,
            x='sr_path_blocked_pct',
            nbins=20,
            title="Distribution of Path Blocked Percentage",
            labels={'sr_path_blocked_pct': 'Path Blocked %'},
            color_discrete_sequence=['#ff6b6b']
        )
        fig_hist.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="Critical (75%)")
        fig_hist.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Warning (50%)")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Detailed rejection table
    st.markdown("#### Recent S/R Path Blocking Rejections")
    display_cols = ['scan_timestamp', 'pair', 'attempted_direction', 'current_price',
                   'potential_take_profit', 'sr_blocking_level', 'sr_blocking_type',
                   'sr_blocking_distance_pips', 'sr_path_blocked_pct', 'target_distance_pips']
    available_cols = [c for c in display_cols if c in sr_blocked_df.columns]

    if available_cols:
        st.dataframe(
            sr_blocked_df[available_cols].head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "scan_timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD, HH:mm"),
                "pair": st.column_config.TextColumn("Pair"),
                "attempted_direction": st.column_config.TextColumn("Direction"),
                "current_price": st.column_config.NumberColumn("Entry", format="%.5f"),
                "potential_take_profit": st.column_config.NumberColumn("TP Target", format="%.5f"),
                "sr_blocking_level": st.column_config.NumberColumn("S/R Level", format="%.5f"),
                "sr_blocking_type": st.column_config.TextColumn("S/R Type"),
                "sr_blocking_distance_pips": st.column_config.NumberColumn("Dist to S/R", format="%.1f pips"),
                "sr_path_blocked_pct": st.column_config.ProgressColumn("Path Blocked", min_value=0, max_value=100, format="%.1f%%"),
                "target_distance_pips": st.column_config.NumberColumn("Target Dist", format="%.1f pips")
            }
        )
    else:
        st.dataframe(sr_blocked_df.head(20), use_container_width=True, hide_index=True)

    # Analysis insights
    st.markdown("---")
    st.markdown("#### Insights")

    if 'pair' in sr_blocked_df.columns and not sr_blocked_df.empty:
        most_blocked_pair = sr_blocked_df['pair'].value_counts().idxmax()
        most_blocked_count = sr_blocked_df['pair'].value_counts().max()

        st.markdown(f"""
        - **Most affected pair:** {most_blocked_pair} ({most_blocked_count} rejections)
        - This pair may have significant S/R levels that frequently block trade targets
        - Consider adjusting TP targets or entry timing for this pair
        """)

    if 'sr_blocking_type' in sr_blocked_df.columns and sr_blocked_df['sr_blocking_type'].notna().any():
        type_mode = sr_blocked_df['sr_blocking_type'].mode()
        if len(type_mode) > 0:
            dominant_type = type_mode.iloc[0]
            st.markdown(f"""
            - **Dominant blocking type:** {dominant_type.upper()}
            - {'BUY signals are being blocked by resistance levels' if dominant_type == 'resistance' else 'SELL signals are being blocked by support levels'}
            """)


def _render_time_analysis(df: pd.DataFrame):
    """Render time analysis sub-tab"""
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
            color_continuous_scale='YlOrRd'
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
                color_continuous_scale='Blues'
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
                color_discrete_sequence=px.colors.qualitative.Pastel
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


def _render_macd_analysis(df: pd.DataFrame):
    """Render MACD momentum analysis section within Market Context tab"""
    st.markdown("#### MACD Momentum Analysis")

    # Check if MACD columns exist
    has_macd = 'macd_line' in df.columns and df['macd_line'].notna().any()
    has_macd_aligned = 'macd_aligned' in df.columns and df['macd_aligned'].notna().any()

    if not has_macd:
        st.info("No MACD data available yet. MACD data will populate as new rejections are logged by the scanner. "
                "Existing rejections recorded before v2.10.0 will not have MACD data.")
        return

    # Filter for rows with MACD data
    macd_df = df[df['macd_line'].notna()].copy()

    if macd_df.empty:
        st.info("No rejections with MACD data found.")
        return

    # Calculate MACD momentum direction if not already present
    if 'macd_momentum' not in macd_df.columns or macd_df['macd_momentum'].isna().all():
        macd_df['macd_momentum'] = macd_df.apply(
            lambda r: 'bullish' if r['macd_line'] > r.get('macd_signal', 0) else 'bearish',
            axis=1
        )

    # Calculate alignment if not present
    if not has_macd_aligned:
        macd_df['macd_aligned'] = macd_df.apply(
            lambda r: (r.get('attempted_direction') == 'BULL' and r['macd_momentum'] == 'bullish') or
                      (r.get('attempted_direction') == 'BEAR' and r['macd_momentum'] == 'bearish'),
            axis=1
        )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_with_macd = len(macd_df)
        st.metric("Rejections with MACD", f"{total_with_macd:,}")

    with col2:
        macd_misaligned = len(df[df['rejection_stage'] == 'MACD_MISALIGNED']) if 'rejection_stage' in df.columns else 0
        st.metric("MACD Misaligned Rejections", f"{macd_misaligned:,}",
                  help="Signals rejected specifically due to MACD momentum opposing trade direction")

    with col3:
        if has_macd_aligned or 'macd_aligned' in macd_df.columns:
            aligned_count = macd_df['macd_aligned'].sum()
            aligned_pct = (aligned_count / len(macd_df) * 100) if len(macd_df) > 0 else 0
            st.metric("MACD Aligned %", f"{aligned_pct:.1f}%",
                      help="% of rejected signals where MACD momentum matched trade direction")

    with col4:
        bullish_count = len(macd_df[macd_df['macd_momentum'] == 'bullish'])
        bearish_count = len(macd_df[macd_df['macd_momentum'] == 'bearish'])
        st.metric("Momentum Split", f"{bullish_count} Bull / {bearish_count} Bear")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        # MACD alignment by rejection stage
        if 'macd_aligned' in macd_df.columns and 'rejection_stage' in macd_df.columns:
            alignment_by_stage = macd_df.groupby(['rejection_stage', 'macd_aligned']).size().reset_index(name='count')
            alignment_by_stage['macd_aligned'] = alignment_by_stage['macd_aligned'].map({True: 'Aligned', False: 'Misaligned'})

            fig_align = px.bar(
                alignment_by_stage,
                x='rejection_stage',
                y='count',
                color='macd_aligned',
                title="MACD Alignment by Rejection Stage",
                barmode='group',
                color_discrete_map={'Aligned': '#51cf66', 'Misaligned': '#ff6b6b'}
            )
            fig_align.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_align, use_container_width=True)

    with col2:
        # MACD histogram distribution
        if 'macd_histogram' in macd_df.columns and macd_df['macd_histogram'].notna().any():
            fig_hist = px.histogram(
                macd_df[macd_df['macd_histogram'].notna()],
                x='macd_histogram',
                color='attempted_direction' if 'attempted_direction' in macd_df.columns else None,
                title="MACD Histogram Distribution at Rejection",
                nbins=40,
                color_discrete_map={'BULL': '#51cf66', 'BEAR': '#ff6b6b'}
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Zero Line")
            st.plotly_chart(fig_hist, use_container_width=True)

    # MACD misalignment detail table
    if macd_misaligned > 0:
        st.markdown("#### Recent MACD Misaligned Rejections")
        misaligned_df = df[df['rejection_stage'] == 'MACD_MISALIGNED'].copy()

        if not misaligned_df.empty:
            display_cols = ['scan_timestamp', 'pair', 'attempted_direction', 'macd_line', 'macd_signal',
                            'macd_histogram', 'macd_momentum', 'confidence_score', 'potential_rr_ratio']
            available_cols = [c for c in display_cols if c in misaligned_df.columns]

            if available_cols:
                st.dataframe(
                    misaligned_df[available_cols].head(20),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "scan_timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD, HH:mm"),
                        "pair": st.column_config.TextColumn("Pair"),
                        "attempted_direction": st.column_config.TextColumn("Signal"),
                        "macd_line": st.column_config.NumberColumn("MACD Line", format="%.6f"),
                        "macd_signal": st.column_config.NumberColumn("Signal Line", format="%.6f"),
                        "macd_histogram": st.column_config.NumberColumn("Histogram", format="%.6f"),
                        "macd_momentum": st.column_config.TextColumn("Momentum"),
                        "confidence_score": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
                        "potential_rr_ratio": st.column_config.NumberColumn("R:R", format="%.2f")
                    }
                )

    # Insights
    st.markdown("#### MACD Insights")

    if 'macd_aligned' in macd_df.columns:
        aligned_pct = (macd_df['macd_aligned'].sum() / len(macd_df) * 100) if len(macd_df) > 0 else 0

        if aligned_pct < 50:
            st.warning(f"""
            **Low MACD alignment ({aligned_pct:.0f}%)**: Most rejected signals had MACD momentum opposing trade direction.
            This suggests the MACD filter is catching many counter-trend attempts.
            """)
        elif aligned_pct > 80:
            st.success(f"""
            **High MACD alignment ({aligned_pct:.0f}%)**: Most rejected signals had MACD momentum supporting trade direction.
            Rejections are due to other factors (confidence, volume, R:R, etc.), not MACD misalignment.
            """)
        else:
            st.info(f"""
            **Moderate MACD alignment ({aligned_pct:.0f}%)**: Mix of aligned and misaligned rejections.
            The MACD filter is contributing to signal filtering alongside other criteria.
            """)

    # Show MACD misalignment impact if data available
    if macd_misaligned > 0:
        total_rejections = len(df)
        macd_pct = (macd_misaligned / total_rejections * 100) if total_rejections > 0 else 0
        st.info(f"""
        **MACD Filter Impact**: {macd_misaligned} signals ({macd_pct:.1f}% of rejections) were blocked
        specifically due to MACD momentum opposing the trade direction. Based on historical analysis,
        these counter-momentum trades had only 38% win rate vs 60% for aligned trades.
        """)


def _render_market_context(df: pd.DataFrame):
    """Render market context sub-tab"""
    st.subheader("Market Context Analysis")

    if df.empty:
        st.info("No market context data available.")
        return

    # MACD Analysis Section
    _render_macd_analysis(df)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # ATR distribution by stage
        if 'atr_15m' in df.columns and df['atr_15m'].notna().any():
            st.markdown("#### ATR at Rejection Points")
            fig_atr = px.box(
                df[df['atr_15m'].notna()],
                x='rejection_stage',
                y='atr_15m',
                title="ATR Distribution by Rejection Stage",
                color='rejection_stage'
            )
            st.plotly_chart(fig_atr, use_container_width=True)

    with col2:
        # EMA distance distribution
        if 'ema_distance_pips' in df.columns and df['ema_distance_pips'].notna().any():
            st.markdown("#### EMA Distance at Rejection Points")
            fig_ema = px.histogram(
                df[df['ema_distance_pips'].notna()],
                x='ema_distance_pips',
                color='rejection_stage',
                title="EMA Distance Distribution (pips)",
                nbins=30
            )
            st.plotly_chart(fig_ema, use_container_width=True)

    # Volume ratio analysis
    if 'volume_ratio' in df.columns and df['volume_ratio'].notna().any():
        st.markdown("#### Volume Ratio at Rejection Points")
        fig_vol = px.scatter(
            df[df['volume_ratio'].notna()],
            x='ema_distance_pips' if 'ema_distance_pips' in df.columns else 'market_hour',
            y='volume_ratio',
            color='rejection_stage',
            title="Volume Ratio vs EMA Distance",
            hover_data=['pair', 'rejection_reason']
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # Pullback depth analysis
    if 'pullback_depth' in df.columns and df['pullback_depth'].notna().any():
        st.markdown("#### Pullback Depth Distribution")
        pullback_df = df[df['pullback_depth'].notna()].copy()
        pullback_df['pullback_pct'] = pullback_df['pullback_depth'] * 100

        fig_pb = px.histogram(
            pullback_df,
            x='pullback_pct',
            color='rejection_stage',
            title="Pullback Depth Distribution (%)",
            nbins=40,
            labels={'pullback_pct': 'Pullback Depth (%)'}
        )
        st.plotly_chart(fig_pb, use_container_width=True)


def _render_near_misses(df: pd.DataFrame, days: int):
    """Render near-misses sub-tab"""
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
        rr = row.get('potential_rr_ratio', 0)
        rr_str = f"{rr:.2f}" if rr else 'N/A'
        session = row.get('market_session', 'N/A')
        reason = row.get('rejection_reason', 'N/A')

        expander_title = f"{direction_icon} {timestamp_str} | {pair} | {direction} | Conf: {conf_str} | R:R: {rr_str} | {session}"

        with st.expander(expander_title, expanded=False):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Signal Details:**")
                st.write(f"- **Pair:** {pair}")
                st.write(f"- **Direction:** {direction}")
                st.write(f"- **Confidence:** {conf_str}")
                st.write(f"- **Session:** {session}")
                entry = row.get('potential_entry', None)
                if entry:
                    st.write(f"- **Entry:** {entry:.5f}")
                sl = row.get('potential_stop_loss', None)
                if sl:
                    st.write(f"- **Stop Loss:** {sl:.5f}")
                tp = row.get('potential_take_profit', None)
                if tp:
                    st.write(f"- **Take Profit:** {tp:.5f}")

            with detail_col2:
                st.markdown("**Risk/Reward:**")
                risk_pips = row.get('potential_risk_pips', None)
                reward_pips = row.get('potential_reward_pips', None)
                if risk_pips:
                    st.write(f"- **Risk:** {risk_pips:.1f} pips")
                if reward_pips:
                    st.write(f"- **Reward:** {reward_pips:.1f} pips")
                st.write(f"- **R:R Ratio:** {rr_str}")

                st.markdown("**Market Context:**")
                ema_dist = row.get('ema_distance_pips', None)
                if ema_dist:
                    st.write(f"- **EMA Distance:** {ema_dist:.1f} pips")
                pullback = row.get('pullback_depth', None)
                if pullback:
                    st.write(f"- **Pullback Depth:** {pullback*100:.1f}%")
                fib_zone = row.get('fib_zone', None)
                if fib_zone:
                    st.write(f"- **Fib Zone:** {fib_zone}")

                # MACD data
                macd_line = row.get('macd_line', None)
                macd_signal = row.get('macd_signal', None)
                macd_momentum = row.get('macd_momentum', None)
                macd_aligned = row.get('macd_aligned', None)
                if macd_line is not None and pd.notna(macd_line):
                    st.markdown("**MACD:**")
                    momentum_dir = macd_momentum if macd_momentum else ('bullish' if macd_line > (macd_signal or 0) else 'bearish')
                    momentum_icon = "🟢" if momentum_dir == 'bullish' else "🔴"
                    st.write(f"- **Momentum:** {momentum_icon} {momentum_dir}")
                    if macd_aligned is not None:
                        aligned_icon = "✅" if macd_aligned else "❌"
                        st.write(f"- **Aligned with Signal:** {aligned_icon}")

            # Rejection reason
            st.markdown("**Rejection Reason:**")
            st.warning(reason)

    # Export button
    st.markdown("---")
    csv = near_miss_df.to_csv(index=False)
    st.download_button(
        label="Export Near-Misses to CSV",
        data=csv,
        file_name=f"smc_near_misses_{days}days.csv",
        mime="text/csv"
    )


def _render_scanner_efficiency(days: int):
    """Render scanner redundancy analysis sub-tab"""
    st.subheader("Scan Redundancy Analysis")
    st.caption("Understand how often candles are re-scanned and whether decisions change")

    conn = get_psycopg2_connection("trading")
    if not conn:
        st.error("Database connection failed")
        return

    try:
        # Query for redundancy analysis
        redundancy_query = """
        WITH candle_scans AS (
            SELECT
                scan_timestamp,
                epic,
                pair,
                rejection_stage,
                rejection_reason,
                created_at,
                ROW_NUMBER() OVER (PARTITION BY scan_timestamp, epic ORDER BY created_at ASC) as scan_num,
                COUNT(*) OVER (PARTITION BY scan_timestamp, epic) as total_scans,
                LAG(rejection_stage) OVER (PARTITION BY scan_timestamp, epic ORDER BY created_at) as prev_stage
            FROM smc_simple_rejections
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
        )
        SELECT * FROM candle_scans
        ORDER BY scan_timestamp DESC, epic, created_at
        """
        df = pd.read_sql_query(redundancy_query, conn, params=[days])

        if df.empty:
            st.info("No data available for redundancy analysis.")
            return

        # Calculate key metrics
        unique_candles = df.groupby(['scan_timestamp', 'epic']).ngroups
        total_scans = len(df)
        redundant_scans = total_scans - unique_candles

        # State changes
        df['stage_changed'] = (df['rejection_stage'] != df['prev_stage']) & df['prev_stage'].notna()
        state_changes = df['stage_changed'].sum()
        candles_with_changes = df[df['stage_changed']].groupby(['scan_timestamp', 'epic']).ngroups

        # Average scans per candle
        scans_per_candle = df.groupby(['scan_timestamp', 'epic'])['total_scans'].first()
        avg_scans = scans_per_candle.mean()
        max_scans = scans_per_candle.max()

        # Summary metrics
        st.markdown("### Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Unique Candles", f"{unique_candles:,}", help="Number of distinct candle+pair combinations analyzed")

        with col2:
            st.metric("Avg Scans/Candle", f"{avg_scans:.1f}", help="How many times each candle is scanned on average")

        with col3:
            redundant_pct = (redundant_scans / total_scans * 100) if total_scans > 0 else 0
            st.metric("Redundant Scans", f"{redundant_pct:.0f}%", help="% of scans that re-analyzed the same candle")

        with col4:
            st.metric(
                "State Changes",
                f"{state_changes:,}",
                delta=f"{candles_with_changes} candles" if candles_with_changes > 0 else None,
                delta_color="normal",
                help="Times the rejection stage changed between scans of the same candle"
            )

        with col5:
            change_rate = (candles_with_changes / unique_candles * 100) if unique_candles > 0 else 0
            st.metric("Change Rate", f"{change_rate:.1f}%", help="% of candles where the rejection decision changed")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Scans Per Candle Distribution")
            scans_df = scans_per_candle.reset_index(name='scan_count')
            scans_df.columns = ['scan_timestamp', 'epic', 'scan_count']

            fig_scans = px.histogram(
                scans_df,
                x='scan_count',
                nbins=min(int(max_scans) + 1, 20),
                title="How Many Times Each Candle Was Re-Scanned",
                labels={'scan_count': 'Scans per Candle', 'count': 'Number of Candles'}
            )
            fig_scans.add_vline(x=avg_scans, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_scans:.1f}")
            fig_scans.add_vline(x=7, line_dash="dot", line_color="green", annotation_text="Expected: ~7")
            st.plotly_chart(fig_scans, use_container_width=True)

        with col2:
            st.markdown("#### State Changes by Stage")
            if state_changes > 0:
                changes_df = df[df['stage_changed']].groupby('rejection_stage').size().reset_index(name='changes')
                fig_changes = px.bar(
                    changes_df,
                    x='rejection_stage',
                    y='changes',
                    title="Which Stages Had Decision Changes",
                    labels={'rejection_stage': 'New Stage', 'changes': 'Number of Changes'}
                )
                st.plotly_chart(fig_changes, use_container_width=True)
            else:
                st.info("No state changes detected - rejection decisions are stable within candles.")

        # State Changes Detail Table
        if state_changes > 0:
            st.markdown("---")
            st.markdown("#### Candles with State Changes")
            st.caption("These are edge cases where market conditions changed during a candle's lifetime")

            change_details = df[df['stage_changed']][
                ['scan_timestamp', 'epic', 'pair', 'prev_stage', 'rejection_stage', 'rejection_reason', 'created_at']
            ].copy()
            change_details.columns = ['Candle Time', 'Epic', 'Pair', 'Previous Stage', 'New Stage', 'New Reason', 'Changed At']
            change_details = change_details.sort_values('Changed At', ascending=False).head(50)

            st.dataframe(change_details, use_container_width=True, hide_index=True)

        # Interpretation
        st.markdown("---")
        st.markdown("#### Interpretation")

        if avg_scans <= 1.5:
            st.success(f"**Low redundancy** ({avg_scans:.1f} scans/candle): Most candles are only scanned once.")
        elif avg_scans <= 4:
            st.info(f"**Moderate redundancy** ({avg_scans:.1f} scans/candle): Candles are re-scanned a few times. Normal for early-stage rejections.")
        elif avg_scans <= 8:
            st.warning(f"**Expected redundancy** ({avg_scans:.1f} scans/candle): Close to theoretical maximum for 2-min scanner on 15-min candles.")
        else:
            st.error(f"**High redundancy** ({avg_scans:.1f} scans/candle): More scans than expected. Check scanner frequency.")

        if change_rate > 10:
            st.warning(f"**High state change rate** ({change_rate:.1f}%): Many candles changed rejection stage. Market may be volatile.")
        elif change_rate > 2:
            st.info(f"**Normal state change rate** ({change_rate:.1f}%): Some edge cases worth investigating.")
        else:
            st.success(f"**Stable decisions** ({change_rate:.1f}%): Rejection decisions are consistent within candles.")

    except Exception as e:
        if "does not exist" in str(e):
            st.info("Redundancy data not yet available. Run the scanner to collect data.")
        else:
            st.error(f"Error fetching redundancy data: {e}")
    finally:
        conn.close()


def _render_smc_conflict_analysis(service, days_filter: int):
    """Render SMC Conflict Analysis sub-tab"""
    st.subheader("SMC Conflict Analysis")

    st.info("""
    **What is SMC Conflict Rejection?**

    Signals are rejected when Smart Money Concepts (SMC) data conflicts with the signal direction:
    - **Order Flow Conflict**: Order flow bias is opposite to signal direction (e.g., BEARISH flow with BULL signal)
    - **Ranging Structure**: Market structure shows no clear directional bias (RANGING)
    - **Low Directional Consensus**: Multiple SMC indicators disagree on direction
    - **Low Structure Score**: Structure score below threshold with no clear bias

    These filters help avoid trades where institutional activity doesn't support the signal direction.
    """)

    # Fetch SMC conflict specific stats
    conflict_stats = service.fetch_smc_conflict_stats(days_filter)

    if conflict_stats.get('total', 0) == 0:
        st.warning("No SMC conflict rejections found for the selected period. This could mean:")
        st.markdown("""
        - The SMC Conflict filter is not yet enabled
        - No signals had conflicting SMC data
        - The filter thresholds are not triggering rejections
        """)

        # Show config info
        with st.expander("SMC Conflict Filter Configuration"):
            st.code("""
# In worker/app/forex_scanner/config.py:
SMC_CONFLICT_FILTER_ENABLED = True   # Enable SMC conflict-based signal rejection
SMC_MIN_DIRECTIONAL_CONSENSUS = 0.3  # Minimum directional consensus (0.0-1.0)
SMC_REJECT_ORDER_FLOW_CONFLICT = True  # Reject when order flow opposes signal
SMC_REJECT_RANGING_STRUCTURE = True    # Reject when structure is RANGING
SMC_MIN_STRUCTURE_SCORE = 0.5          # Minimum structure score
            """)
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total SMC Conflicts", f"{conflict_stats.get('total', 0):,}")
    with col2:
        st.metric("Pairs Affected", conflict_stats.get('unique_pairs', 0))
    with col3:
        st.metric("Sessions Affected", conflict_stats.get('sessions_affected', 0))
    with col4:
        # Calculate most common conflict
        top_reasons = conflict_stats.get('top_reasons', [])
        if top_reasons:
            top_reason = top_reasons[0].get('rejection_reason', 'N/A')[:30]
            st.metric("Top Conflict Type", top_reason + "...")
        else:
            st.metric("Top Conflict Type", "N/A")

    st.markdown("---")

    # Breakdown charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### SMC Conflicts by Currency Pair")
        by_pair = conflict_stats.get('by_pair', {})
        if by_pair:
            pair_df = pd.DataFrame(list(by_pair.items()), columns=['Pair', 'Count'])
            fig_pair = px.bar(
                pair_df,
                x='Pair',
                y='Count',
                title="SMC Conflict Rejections by Pair",
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_pair, use_container_width=True)
        else:
            st.info("No pair breakdown available.")

    with col2:
        st.markdown("#### SMC Conflicts by Session")
        by_session = conflict_stats.get('by_session', {})
        if by_session:
            session_df = pd.DataFrame(list(by_session.items()), columns=['Session', 'Count'])
            fig_session = px.pie(
                session_df,
                values='Count',
                names='Session',
                title="SMC Conflict Rejections by Session",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_session, use_container_width=True)
        else:
            st.info("No session breakdown available.")

    # Top conflict reasons
    st.markdown("---")
    st.markdown("#### Top Conflict Reasons")
    top_reasons = conflict_stats.get('top_reasons', [])
    if top_reasons:
        reason_df = pd.DataFrame(top_reasons)
        st.dataframe(
            reason_df.rename(columns={'rejection_reason': 'Conflict Reason', 'count': 'Count'}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No conflict reason breakdown available.")

    # Fetch detailed SMC conflict data
    st.markdown("---")
    st.markdown("#### Recent SMC Conflict Rejections")

    conflict_df = service.fetch_smc_conflict_details(days_filter)

    if conflict_df.empty:
        st.info("No detailed conflict data available.")
        return

    # Show detailed table with parsed SMC data
    display_cols = ['scan_timestamp', 'pair', 'attempted_direction', 'order_flow_bias',
                   'structure_bias', 'structure_score', 'directional_consensus',
                   'confidence_score', 'potential_rr_ratio']
    available_cols = [c for c in display_cols if c in conflict_df.columns]

    if available_cols:
        st.dataframe(
            conflict_df[available_cols].head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                "scan_timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD, HH:mm"),
                "pair": st.column_config.TextColumn("Pair"),
                "attempted_direction": st.column_config.TextColumn("Signal"),
                "order_flow_bias": st.column_config.TextColumn("Order Flow"),
                "structure_bias": st.column_config.TextColumn("Structure"),
                "structure_score": st.column_config.NumberColumn("Struct Score", format="%.2f"),
                "directional_consensus": st.column_config.NumberColumn("Dir Consensus", format="%.2f"),
                "confidence_score": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
                "potential_rr_ratio": st.column_config.NumberColumn("R:R", format="%.2f")
            }
        )
    else:
        st.dataframe(conflict_df.head(50), use_container_width=True, hide_index=True)

    # Analysis insights
    st.markdown("---")
    st.markdown("#### Insights & Recommendations")

    # Check for patterns
    if 'order_flow_bias' in conflict_df.columns:
        order_flow_conflicts = conflict_df[conflict_df['order_flow_bias'].isin(['BEARISH', 'BULLISH'])]
        if len(order_flow_conflicts) > 0:
            of_pct = len(order_flow_conflicts) / len(conflict_df) * 100
            st.markdown(f"- **{of_pct:.0f}%** of conflicts involve order flow opposing the signal direction")

    if 'structure_bias' in conflict_df.columns:
        ranging_conflicts = conflict_df[conflict_df['structure_bias'].str.upper() == 'RANGING']
        if len(ranging_conflicts) > 0:
            ranging_pct = len(ranging_conflicts) / len(conflict_df) * 100
            st.markdown(f"- **{ranging_pct:.0f}%** of conflicts occur during RANGING market structure")

    if 'directional_consensus' in conflict_df.columns and conflict_df['directional_consensus'].notna().any():
        # Convert to numeric (may be stored as string in JSON)
        avg_consensus = pd.to_numeric(conflict_df['directional_consensus'], errors='coerce').mean()
        if pd.notna(avg_consensus):
            st.markdown(f"- Average directional consensus at rejection: **{avg_consensus:.2f}** (threshold: 0.3)")

    # Recommendations based on data
    if conflict_stats.get('total', 0) > 20:
        st.success("""
        **The SMC Conflict filter is actively protecting against trades with conflicting institutional activity.**

        Monitor the Outcome Analysis tab to see if these rejections are avoiding losses (would-be losers)
        or missing profits (would-be winners). Adjust thresholds if needed.
        """)
    else:
        st.info("""
        **Low conflict rejection count.** This could indicate:
        - Market conditions are aligned (good for trading)
        - Thresholds may be too relaxed
        - Filter may need more time to collect data
        """)

    # Export option
    st.markdown("---")
    csv = conflict_df.to_csv(index=False)
    st.download_button(
        label="Export SMC Conflicts to CSV",
        data=csv,
        file_name=f"smc_conflicts_{days_filter}days.csv",
        mime="text/csv"
    )
