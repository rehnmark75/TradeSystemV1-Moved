"""
HTF (Higher Timeframe) Analysis Tab Component

Analyzes the correlation between 4H candle direction at signal time
and trade outcomes to help determine if HTF momentum impacts performance.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.htf_analysis_service import HTFAnalysisService


def render_htf_analysis_tab():
    """Render the HTF Analysis tab."""
    st.header("4H Candle Direction Analysis")
    st.markdown("*Analyze correlation between higher timeframe momentum and trade outcomes*")

    service = HTFAnalysisService()

    # Time period selector
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        days = st.selectbox(
            "Time Period (days)",
            options=[7, 14, 30, 60, 90],
            index=2,
            key="htf_analysis_days"
        )

    with col2:
        if st.button("Refresh Data", key="htf_refresh"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # Main metrics row
    _render_alignment_summary(service, days)

    st.markdown("---")

    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Alignment Analysis",
        "Two-Candle Patterns",
        "By Pair",
        "Distribution"
    ])

    with tab1:
        _render_alignment_analysis(service, days)

    with tab2:
        _render_pattern_analysis(service, days)

    with tab3:
        _render_pair_analysis(service, days)

    with tab4:
        _render_distribution_analysis(service, days)


def _render_alignment_summary(service: HTFAnalysisService, days: int):
    """Render the summary metrics for alignment analysis."""
    df = service.get_alignment_analysis(days)

    if df.empty:
        st.warning("No data available for the selected period.")
        return

    # Calculate summary metrics
    aligned = df[df['alignment'] == 'ALIGNED']
    counter = df[df['alignment'] == 'COUNTER']

    aligned_signals = int(aligned['total_signals'].sum()) if not aligned.empty else 0
    aligned_trades = int(aligned['total_trades'].sum()) if not aligned.empty else 0
    aligned_wins = int(aligned['wins'].sum()) if not aligned.empty else 0
    aligned_losses = int(aligned['losses'].sum()) if not aligned.empty else 0
    aligned_pnl = float(aligned['total_pnl'].sum()) if not aligned.empty else 0

    counter_signals = int(counter['total_signals'].sum()) if not counter.empty else 0
    counter_trades = int(counter['total_trades'].sum()) if not counter.empty else 0
    counter_wins = int(counter['wins'].sum()) if not counter.empty else 0
    counter_losses = int(counter['losses'].sum()) if not counter.empty else 0
    counter_pnl = float(counter['total_pnl'].sum()) if not counter.empty else 0

    # Calculate win rates
    aligned_wr = (aligned_wins / (aligned_wins + aligned_losses) * 100) if (aligned_wins + aligned_losses) > 0 else 0
    counter_wr = (counter_wins / (counter_wins + counter_losses) * 100) if (counter_wins + counter_losses) > 0 else 0

    # Display metrics
    st.subheader("Summary: Aligned vs Counter-Trend Signals")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Aligned Signals",
            aligned_signals,
            help="Signals where direction matches 4H candle (BULL+BULLISH or BEAR+BEARISH)"
        )

    with col2:
        st.metric(
            "Counter Signals",
            counter_signals,
            help="Signals against 4H candle direction (BULL+BEARISH or BEAR+BULLISH)"
        )

    with col3:
        delta = f"{aligned_wr - counter_wr:+.1f}%" if counter_wr > 0 else None
        st.metric(
            "Aligned Win Rate",
            f"{aligned_wr:.1f}%",
            delta=delta,
            help="Win rate when signal aligns with 4H candle"
        )

    with col4:
        st.metric(
            "Counter Win Rate",
            f"{counter_wr:.1f}%",
            help="Win rate when signal is against 4H candle"
        )

    with col5:
        pnl_diff = aligned_pnl - counter_pnl
        st.metric(
            "P&L Difference",
            f"${pnl_diff:,.2f}",
            delta=f"Aligned: ${aligned_pnl:,.2f}" if aligned_pnl != 0 else None,
            help="Total P&L difference (Aligned - Counter)"
        )

    # Key insight
    if aligned_wr > counter_wr and (aligned_wins + aligned_losses) >= 5:
        st.success(f"Aligned signals outperform by {aligned_wr - counter_wr:.1f}% win rate. Consider prioritizing trades aligned with 4H momentum.")
    elif counter_wr > aligned_wr and (counter_wins + counter_losses) >= 5:
        st.warning(f"Counter-trend signals have {counter_wr - aligned_wr:.1f}% higher win rate. This may indicate mean-reversion opportunities.")
    else:
        st.info("Insufficient data to draw conclusions. Need more completed trades.")


def _render_alignment_analysis(service: HTFAnalysisService, days: int):
    """Render detailed alignment analysis."""
    st.subheader("Alignment Performance Details")

    df = service.get_alignment_analysis(days)

    if df.empty:
        st.warning("No data available.")
        return

    # Display table
    display_df = df.copy()
    display_df.columns = ['Alignment', 'Signals', 'Trades', 'Wins', 'Losses', 'BE', 'Win Rate %', 'Total P&L', 'Avg P&L']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Win Rate %": st.column_config.NumberColumn(format="%.1f%%"),
            "Total P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Avg P&L": st.column_config.NumberColumn(format="$%.2f"),
        }
    )

    # Win rate comparison chart
    if len(df) > 1:
        fig = px.bar(
            df,
            x='alignment',
            y='win_rate',
            color='alignment',
            color_discrete_map={'ALIGNED': '#28a745', 'COUNTER': '#dc3545', 'NEUTRAL': '#6c757d'},
            title="Win Rate by Alignment Type",
            labels={'alignment': 'Alignment', 'win_rate': 'Win Rate (%)'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def _render_pattern_analysis(service: HTFAnalysisService, days: int):
    """Render two-candle pattern analysis."""
    st.subheader("Two-Candle Pattern Performance")
    st.markdown("*Pattern format: Current_Previous (e.g., BULLISH_BEARISH = current 4H bullish, previous 4H bearish)*")

    df = service.get_two_candle_pattern_analysis(days)

    if df.empty:
        st.warning("No pattern data available.")
        return

    # Display table
    display_df = df.copy()
    display_df.columns = ['Pattern', 'Signal', 'Signals', 'Trades', 'Wins', 'Losses', 'Win Rate %', 'Total P&L', 'Avg P&L']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Win Rate %": st.column_config.NumberColumn(format="%.1f%%"),
            "Total P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Avg P&L": st.column_config.NumberColumn(format="$%.2f"),
        }
    )

    # Pattern performance chart (if enough data)
    if len(df) >= 2:
        # Filter to patterns with at least 1 trade
        chart_df = df[df['total_trades'] > 0].copy()
        if not chart_df.empty:
            chart_df['label'] = chart_df['pattern'] + ' (' + chart_df['signal_type'] + ')'

            fig = px.bar(
                chart_df,
                x='label',
                y='win_rate',
                color='signal_type',
                color_discrete_map={'BULL': '#28a745', 'BEAR': '#dc3545'},
                title="Win Rate by Pattern",
                labels={'label': 'Pattern (Signal)', 'win_rate': 'Win Rate (%)'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # Interpretation guide
    with st.expander("Pattern Interpretation Guide"):
        st.markdown("""
        **Momentum Continuation Patterns:**
        - `BULLISH_BULLISH` + BULL signal = Strong bullish momentum
        - `BEARISH_BEARISH` + BEAR signal = Strong bearish momentum

        **Potential Reversal Patterns:**
        - `BULLISH_BEARISH` = Reversal from bearish to bullish
        - `BEARISH_BULLISH` = Reversal from bullish to bearish

        **Key Insight:** Compare win rates to identify which patterns work best for your strategy.
        """)


def _render_pair_analysis(service: HTFAnalysisService, days: int):
    """Render per-pair HTF analysis."""
    st.subheader("HTF Alignment by Currency Pair")

    df = service.get_pair_htf_performance(days)

    if df.empty:
        st.warning("No pair data available.")
        return

    # Pivot for better display
    pivot_df = df.pivot_table(
        index='pair',
        columns='alignment',
        values=['total_signals', 'total_trades', 'wins', 'losses', 'win_rate', 'total_pnl'],
        aggfunc='first'
    ).fillna(0)

    # Flatten column names
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Display simplified view
    simple_cols = ['pair']
    for align in ['ALIGNED', 'COUNTER']:
        if f'{align}_total_trades' in pivot_df.columns:
            simple_cols.extend([f'{align}_total_trades', f'{align}_win_rate', f'{align}_total_pnl'])

    if len(simple_cols) > 1:
        display_df = pivot_df[simple_cols].copy()

        # Rename columns for clarity
        col_rename = {
            'pair': 'Pair',
            'ALIGNED_total_trades': 'Aligned Trades',
            'ALIGNED_win_rate': 'Aligned WR%',
            'ALIGNED_total_pnl': 'Aligned P&L',
            'COUNTER_total_trades': 'Counter Trades',
            'COUNTER_win_rate': 'Counter WR%',
            'COUNTER_total_pnl': 'Counter P&L'
        }
        display_df = display_df.rename(columns=col_rename)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    # Comparison chart
    if not df.empty and len(df) >= 2:
        fig = px.bar(
            df,
            x='pair',
            y='win_rate',
            color='alignment',
            barmode='group',
            color_discrete_map={'ALIGNED': '#28a745', 'COUNTER': '#dc3545'},
            title="Win Rate by Pair and Alignment",
            labels={'pair': 'Currency Pair', 'win_rate': 'Win Rate (%)', 'alignment': 'Alignment'}
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_distribution_analysis(service: HTFAnalysisService, days: int):
    """Render 4H direction distribution."""
    st.subheader("4H Candle Direction Distribution")

    df = service.get_htf_direction_distribution(days)

    if df.empty:
        st.warning("No distribution data available.")
        return

    # Pie chart for overall distribution
    overall = df.groupby('direction')['count'].sum().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            overall,
            values='count',
            names='direction',
            title="Overall 4H Direction at Signal Time",
            color='direction',
            color_discrete_map={'BULLISH': '#28a745', 'BEARISH': '#dc3545', 'NEUTRAL': '#6c757d'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # By signal type
        fig = px.bar(
            df,
            x='signal_type',
            y='count',
            color='direction',
            barmode='group',
            title="4H Direction by Signal Type",
            color_discrete_map={'BULLISH': '#28a745', 'BEARISH': '#dc3545', 'NEUTRAL': '#6c757d'},
            labels={'signal_type': 'Signal Type', 'count': 'Count', 'direction': '4H Direction'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    total = overall['count'].sum()
    st.markdown(f"**Total Signals Analyzed:** {total}")

    for _, row in overall.iterrows():
        pct = row['count'] / total * 100 if total > 0 else 0
        st.markdown(f"- **{row['direction']}:** {row['count']} ({pct:.1f}%)")
