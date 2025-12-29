"""
Overview Tab Component

Renders the main trading overview tab with:
- Key performance metrics
- Cumulative P&L chart
- Win/Loss pie chart
- Recent trades table
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from services.trading_analytics_service import TradingAnalyticsService, TradingStatistics


def render_overview_tab():
    """Render the overview tab with key metrics and charts"""
    service = TradingAnalyticsService()

    st.header("Trading Overview")

    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        timeframe = st.selectbox(
            "Analysis Period",
            options=['1_day', '7_days', '30_days', '90_days'],
            format_func=lambda x: {
                '1_day': '24 Hours',
                '7_days': '7 Days',
                '30_days': '30 Days',
                '90_days': '90 Days'
            }[x],
            key='overview_timeframe',
            index=1  # Default to 7 days
        )

    with col2:
        if st.button("Refresh Data", key="overview_refresh"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    with col3:
        show_debug = st.checkbox("Debug Mode", key="overview_debug")

    # Map timeframe to days
    timeframe_map = {'1_day': 1, '7_days': 7, '30_days': 30, '90_days': 90}
    days_back = timeframe_map[timeframe]

    # Fetch data from service (cached)
    stats = service.fetch_trading_statistics(days_back)
    trades_df = service.fetch_trades_dataframe(days_back)

    if not stats:
        st.error("Unable to fetch trading statistics. Please check database connection.")
        return

    # Key metrics
    _render_key_metrics(stats)

    # Performance charts
    if not trades_df.empty:
        st.subheader("Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Cumulative P&L Chart
            if 'timestamp' in trades_df.columns and 'profit_loss' in trades_df.columns:
                trades_sorted = trades_df[trades_df['profit_loss'].notna()].sort_values('timestamp')
                trades_sorted['cumulative_pnl'] = trades_sorted['profit_loss'].cumsum()

                fig_cumulative = px.line(
                    trades_sorted,
                    x='timestamp',
                    y='cumulative_pnl',
                    title="Cumulative P&L Over Time",
                    labels={'cumulative_pnl': 'Cumulative P&L (SEK)', 'timestamp': 'Date'}
                )
                fig_cumulative.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_cumulative, use_container_width=True)

        with col2:
            # Win/Loss Pie Chart
            win_loss_data = pd.DataFrame({
                'Result': ['Wins', 'Losses', 'Pending'],
                'Count': [stats.winning_trades, stats.losing_trades, stats.pending_trades]
            })

            fig_pie = px.pie(
                win_loss_data,
                values='Count',
                names='Result',
                title="Trade Outcomes",
                color='Result',
                color_discrete_map={'Wins': '#28a745', 'Losses': '#dc3545', 'Pending': '#6c757d'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Recent trades summary
    if not trades_df.empty:
        st.subheader("Recent Trades Summary")
        recent_trades = trades_df.head(10)

        display_columns = ['timestamp', 'symbol', 'strategy', 'direction', 'profit_loss_formatted', 'trade_result']
        display_columns = [col for col in display_columns if col in recent_trades.columns]

        st.dataframe(
            recent_trades[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time", format="MM/DD HH:mm"),
                "symbol": "Pair",
                "strategy": "Strategy",
                "direction": "Direction",
                "profit_loss_formatted": "P&L",
                "trade_result": "Result"
            }
        )


def _render_key_metrics(stats: TradingStatistics):
    """Render key trading metrics cards"""
    st.subheader("Key Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        profit_class = "profit-positive" if stats.total_profit_loss > 0 else "profit-negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total P&L</h4>
            <h2 class="{profit_class}">{stats.total_profit_loss:.2f} SEK</h2>
            <small>{stats.total_trades} total trades</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        win_rate_class = "win-rate-high" if stats.win_rate >= 60 else "win-rate-medium" if stats.win_rate >= 40 else "win-rate-low"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Win Rate</h4>
            <h2 class="{win_rate_class}">{stats.win_rate:.1f}%</h2>
            <small>{stats.winning_trades}W / {stats.losing_trades}L</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Profit Factor</h4>
            <h2>{stats.profit_factor:.2f}</h2>
            <small>Avg Win: {stats.avg_profit:.2f}</small>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Best Trade</h4>
            <h2 class="profit-positive">+{stats.largest_win:.2f}</h2>
            <small>Largest single win</small>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Worst Trade</h4>
            <h2 class="profit-negative">{stats.largest_loss:.2f}</h2>
            <small>Largest single loss</small>
        </div>
        """, unsafe_allow_html=True)
