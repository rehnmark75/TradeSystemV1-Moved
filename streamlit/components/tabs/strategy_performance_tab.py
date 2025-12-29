"""
Strategy Performance Tab Component

Renders the strategy performance analysis tab with:
- Strategy comparison table
- P&L and win rate charts by strategy
- Detailed strategy breakdown
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from services.trading_analytics_service import TradingAnalyticsService


def render_strategy_performance_tab():
    """Render the strategy analysis tab"""
    service = TradingAnalyticsService()

    st.header("Strategy Performance Analysis")

    # Controls
    col1, col2 = st.columns([2, 1])

    with col1:
        days_back = st.selectbox(
            "Analysis Period",
            [7, 30, 90],
            index=0,
            format_func=lambda x: f"{x} days",
            key="strategy_days"
        )

    with col2:
        if st.button("Refresh", key="strategy_refresh"):
            st.rerun()

    # Fetch strategy data from service (cached)
    strategy_df = service.fetch_strategy_performance(days_back)

    if strategy_df.empty:
        st.warning(f"No strategy data found for the last {days_back} days")
        return

    # Strategy performance table
    st.subheader("Strategy Performance Summary")

    st.dataframe(
        strategy_df[['strategy', 'total_trades', 'wins', 'losses', 'win_rate', 'total_pnl', 'avg_pnl', 'avg_confidence', 'pairs_traded']],
        column_config={
            "strategy": "Strategy",
            "total_trades": "Total Trades",
            "wins": "Wins",
            "losses": "Losses",
            "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
            "total_pnl": st.column_config.NumberColumn("Total P&L", format="%.2f"),
            "avg_pnl": st.column_config.NumberColumn("Avg P&L", format="%.2f"),
            "avg_confidence": st.column_config.NumberColumn("Avg Confidence", format="%.1%"),
            "pairs_traded": "Pairs Traded"
        },
        use_container_width=True,
        hide_index=True
    )

    # Strategy comparison charts
    st.subheader("Strategy Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # P&L by strategy
        fig_pnl = px.bar(
            strategy_df,
            x='strategy',
            y='total_pnl',
            title="Total P&L by Strategy",
            color='total_pnl',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_pnl, use_container_width=True)

    with col2:
        # Win rate comparison
        fig_win = px.bar(
            strategy_df,
            x='strategy',
            y='win_rate',
            title="Win Rate by Strategy",
            color='win_rate',
            color_continuous_scale='RdYlGn'
        )
        fig_win.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Break Even")
        st.plotly_chart(fig_win, use_container_width=True)

    # Strategy details
    st.subheader("Strategy Details")

    for _, strategy in strategy_df.iterrows():
        with st.expander(f"{strategy['strategy']} Strategy Details"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Trades", strategy['total_trades'])
                st.metric("Wins", strategy['wins'])

            with col2:
                st.metric("Win Rate", f"{strategy['win_rate']:.1f}%")
                st.metric("Losses", strategy['losses'])

            with col3:
                st.metric("Total P&L", f"{strategy['total_pnl']:.2f}")
                st.metric("Avg P&L", f"{strategy['avg_pnl']:.2f}")

            with col4:
                st.metric("Best Trade", f"{strategy['best_trade']:.2f}")
                st.metric("Worst Trade", f"{strategy['worst_trade']:.2f}")

    # Pair performance section
    st.subheader("Pair Performance")
    pair_df = service.fetch_pair_performance(days_back)

    if not pair_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Top performing pairs
            fig_pair_pnl = px.bar(
                pair_df.head(10),
                x='symbol',
                y='total_pnl',
                title="Top 10 Pairs by P&L",
                color='total_pnl',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig_pair_pnl, use_container_width=True)

        with col2:
            # Win rate by pair
            fig_pair_win = px.bar(
                pair_df.head(10),
                x='symbol',
                y='win_rate',
                title="Win Rate by Pair",
                color='win_rate',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_pair_win, use_container_width=True)

        # Full pair table
        st.dataframe(
            pair_df[['symbol', 'total_trades', 'wins', 'losses', 'win_rate', 'total_pnl', 'avg_pnl']],
            column_config={
                "symbol": "Pair",
                "total_trades": "Trades",
                "wins": "Wins",
                "losses": "Losses",
                "win_rate": st.column_config.NumberColumn("Win Rate %", format="%.1f"),
                "total_pnl": st.column_config.NumberColumn("Total P&L", format="%.2f"),
                "avg_pnl": st.column_config.NumberColumn("Avg P&L", format="%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
