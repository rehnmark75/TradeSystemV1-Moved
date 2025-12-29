"""
Unfilled Orders Tab Component

Renders the Unfilled Orders Analysis tab with:
- Summary statistics
- Detailed order analysis table
- Per-epic breakdown with charts
- Optimization insights
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from services.unfilled_orders_service import UnfilledOrdersService


def render_unfilled_orders_tab():
    """Render Unfilled Orders Analysis tab."""
    service = UnfilledOrdersService()

    st.header("Unfilled Order Analysis")
    st.markdown("""
    Analyze stop-entry orders that expired without filling. Determines if the signal was:
    - **GOOD_SIGNAL**: Would have hit take profit if filled
    - **BAD_SIGNAL**: Would have hit stop loss if filled
    - **INCONCLUSIVE**: Neither TP nor SL reached in 24h
    """)

    # Check if view exists
    if not service.check_view_exists():
        st.error("Database view not found")
        st.info("Make sure the v_unfilled_order_analysis view exists. Run the migration if needed.")
        return

    # Fetch data from service (cached)
    summary_df = service.fetch_summary()
    detail_df = service.fetch_detailed_analysis()
    epic_df = service.fetch_epic_breakdown()

    if detail_df.empty:
        st.info("No unfilled orders found. This is good - all orders are filling!")
        return

    # Summary metrics
    st.subheader("Summary")
    if not summary_df.empty:
        row = summary_df.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Unfilled", int(row.get('total_unfilled', 0)))
        with col2:
            st.metric("Would Fill (4h)", int(row.get('would_fill_4h', 0)))
        with col3:
            good = int(row.get('good_signals', 0))
            bad = int(row.get('bad_signals', 0))
            st.metric("Good Signals", good, delta=f"{good} +")
        with col4:
            st.metric("Bad Signals", bad, delta=f"-{bad}" if bad > 0 else "0", delta_color="inverse")
        with col5:
            win_rate = row.get('win_rate_pct', 0)
            if win_rate and not pd.isna(win_rate):
                st.metric("Win Rate (if filled)", f"{win_rate:.0f}%")
            else:
                st.metric("Win Rate (if filled)", "N/A")

    # Sub-tabs
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Detailed Analysis", "Per-Epic Breakdown", "Insights"
    ])

    with sub_tab1:
        _render_detailed_analysis(detail_df)

    with sub_tab2:
        _render_epic_breakdown(epic_df)

    with sub_tab3:
        _render_insights(epic_df)


def _render_detailed_analysis(detail_df: pd.DataFrame):
    """Render detailed unfilled order analysis"""
    st.subheader("Unfilled Order Details")

    # Format the dataframe for display
    display_df = detail_df.copy()
    display_df['order_time'] = pd.to_datetime(display_df['order_time']).dt.strftime('%Y-%m-%d %H:%M')

    # Rename columns for clarity
    display_df = display_df.rename(columns={
        'symbol': 'Epic',
        'direction': 'Dir',
        'order_time': 'Order Time',
        'gap_to_entry_pips': 'Gap (pips)',
        'would_fill_4h': 'Fill 4h',
        'outcome_4h': '4h Outcome',
        'would_fill_24h': 'Fill 24h',
        'outcome_24h': '24h Outcome',
        'signal_quality': 'Quality',
        'max_favorable_pips': 'Favorable Move',
        'max_adverse_pips': 'Adverse Move'
    })

    # Select columns to display
    cols_to_show = ['id', 'Epic', 'Dir', 'Order Time', 'Gap (pips)',
                  'Fill 4h', '4h Outcome', 'Fill 24h', '24h Outcome',
                  'Quality', 'Favorable Move', 'Adverse Move']
    display_df = display_df[[c for c in cols_to_show if c in display_df.columns]]

    # Color-code signal quality
    def highlight_quality(val):
        if val == 'GOOD_SIGNAL':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'BAD_SIGNAL':
            return 'background-color: #f8d7da; color: #721c24'
        return 'background-color: #fff3cd; color: #856404'

    st.dataframe(
        display_df.style.applymap(
            highlight_quality, subset=['Quality'] if 'Quality' in display_df.columns else []
        ),
        use_container_width=True,
        height=400
    )

    # Expandable details for each order
    st.markdown("---")
    st.markdown("**Click to expand individual order details:**")
    for idx, row in detail_df.iterrows():
        quality = row.get('signal_quality', 'UNKNOWN')
        icon = "+" if quality == 'GOOD_SIGNAL' else "-" if quality == 'BAD_SIGNAL' else "o"
        with st.expander(f"{icon} {row['symbol']} {row['direction']} @ {row['order_time']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Entry Details:**")
                st.write(f"- Entry Level: {row.get('entry_level', 'N/A')}")
                st.write(f"- Stop Loss: {row.get('stop_loss', 'N/A')}")
                st.write(f"- Take Profit: {row.get('take_profit', 'N/A')}")
            with col2:
                st.markdown("**At Expiry:**")
                st.write(f"- Price: {row.get('price_at_expiry', 'N/A')}")
                st.write(f"- Gap to Entry: {row.get('gap_to_entry_pips', 'N/A')} pips")
                st.write(f"- Expiry Time: {row.get('expiry_time', 'N/A')}")
            with col3:
                st.markdown("**Post-Expiry Analysis:**")
                st.write(f"- Would Fill 24h: {'Yes' if row.get('would_fill_24h') else 'No'}")
                st.write(f"- 24h Outcome: {row.get('outcome_24h', 'N/A')}")
                st.write(f"- Max Favorable: {row.get('max_favorable_pips', 'N/A')} pips")
                st.write(f"- Max Adverse: {row.get('max_adverse_pips', 'N/A')} pips")

            # Signal quality explanation
            if quality == 'GOOD_SIGNAL':
                st.success("+ GOOD SIGNAL: Would have hit take profit if filled. Consider longer expiry or tighter entry offset.")
            elif quality == 'BAD_SIGNAL':
                st.error("- BAD SIGNAL: Would have hit stop loss. The direction was wrong - review entry criteria.")
            else:
                st.warning("o INCONCLUSIVE: Neither TP nor SL hit in 24h. More data needed.")


def _render_epic_breakdown(epic_df: pd.DataFrame):
    """Render per-epic performance breakdown"""
    st.subheader("Per-Epic Performance")

    if epic_df.empty:
        st.info("No data available for epic breakdown.")
        return

    # Add win rate column
    epic_df = epic_df.copy()
    epic_df['decisive'] = epic_df['good'] + epic_df['bad']
    epic_df['win_rate'] = epic_df.apply(
        lambda r: f"{100*r['good']/r['decisive']:.0f}%" if r['decisive'] > 0 else "N/A",
        axis=1
    )

    # Display table
    st.dataframe(
        epic_df.rename(columns={
            'symbol': 'Epic',
            'total_unfilled': 'Total Unfilled',
            'good': 'Good Signals',
            'bad': 'Bad Signals',
            'inconclusive': 'Inconclusive',
            'avg_gap_pips': 'Avg Gap (pips)',
            'avg_favorable': 'Avg Favorable',
            'avg_adverse': 'Avg Adverse',
            'win_rate': 'Win Rate'
        }),
        use_container_width=True
    )

    # Bar chart of signal quality by epic
    if len(epic_df) > 0:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Good Signals',
            x=epic_df['symbol'],
            y=epic_df['good'],
            marker_color='#28a745'
        ))
        fig.add_trace(go.Bar(
            name='Bad Signals',
            x=epic_df['symbol'],
            y=epic_df['bad'],
            marker_color='#dc3545'
        ))
        fig.add_trace(go.Bar(
            name='Inconclusive',
            x=epic_df['symbol'],
            y=epic_df['inconclusive'],
            marker_color='#ffc107'
        ))
        fig.update_layout(
            barmode='stack',
            title='Unfilled Order Quality by Epic',
            xaxis_title='Epic',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_insights(epic_df: pd.DataFrame):
    """Render optimization insights"""
    st.subheader("Optimization Insights")

    st.markdown("""
    ### Key Questions to Answer:
    1. **Are entries too far from market?** High gap = entries never reach
    2. **Are we picking wrong direction?** High bad signals = strategy issue
    3. **Is expiry too short?** Good signals that would fill later = extend expiry
    """)

    if not epic_df.empty:
        # Identify epics needing attention
        st.markdown("### Per-Epic Recommendations")

        for _, row in epic_df.iterrows():
            epic = row['symbol']
            good = row['good']
            bad = row['bad']
            total = row['total_unfilled']
            avg_gap = row['avg_gap_pips']

            if total < 2:
                continue  # Not enough data

            with st.expander(f"{epic} ({total} unfilled)"):
                issues = []
                recommendations = []

                # Analyze issues
                if avg_gap and avg_gap > 5:
                    issues.append(f"! High avg gap to entry: {avg_gap} pips")
                    recommendations.append("Consider reducing stop-entry offset or extending expiry")

                if bad > good and (good + bad) > 0:
                    issues.append(f"! More bad signals ({bad}) than good ({good})")
                    recommendations.append("Review entry direction logic - signals often wrong")

                if good > 0 and bad == 0:
                    issues.append(f"+ All decisive signals were good ({good})")
                    recommendations.append("Consider extending expiry time - good signals not filling")

                if avg_gap and avg_gap < 3:
                    issues.append(f"+ Entries close to market ({avg_gap} pips)")

                # Display
                if issues:
                    for issue in issues:
                        st.write(issue)
                    st.markdown("**Recommendations:**")
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("No significant issues detected. Continue monitoring.")

    st.markdown("---")
    st.info("Tip: Run for several more days to gather statistically significant data, then tune parameters per epic.")
