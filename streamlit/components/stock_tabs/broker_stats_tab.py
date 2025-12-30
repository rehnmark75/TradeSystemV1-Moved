"""
Stock Scanner Broker Stats Tab

Real trading performance from RoboMarkets:
- Account balance and trend
- Key trading metrics (net profit, win rate, profit factor)
- Open positions
- Performance by side (long/short)
- Equity curve
- Daily P&L
- Performance by ticker
- Recent closed trades
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_broker_stats_tab():
    """Render the Broker Statistics tab with real trading performance from RoboMarkets."""
    from services.broker_analytics_service import get_broker_service

    broker_service = get_broker_service()

    if not broker_service.is_configured:
        st.warning("Database connection not available. Check PostgreSQL configuration.")
        return

    st.markdown("""
    <div class="main-header">
        <h2>Broker Trading Statistics</h2>
        <p>Performance data from RoboMarkets account (synced to database)</p>
    </div>
    """, unsafe_allow_html=True)

    # Account Balance Section
    balance_data = broker_service.get_account_balance()
    trend_data = broker_service.get_account_balance_trend(days=7)

    if not balance_data.get("error"):
        _render_account_overview(balance_data, trend_data)
        st.markdown("---")

    # Period selector
    col1, col2 = st.columns([1, 3])
    with col1:
        days = st.selectbox("Analysis Period", [7, 14, 30, 60, 90], index=2)

    # Fetch data
    with st.spinner("Fetching broker data..."):
        stats = broker_service.calculate_statistics(days=days)
        positions_data = broker_service.get_open_positions()
        trades_data = broker_service.get_closed_trades(days=days)

    if stats.error:
        st.error(f"Failed to fetch broker data: {stats.error}")
        st.info("Sync data with: `docker exec task-worker python -m stock_scanner.main broker-sync`")
        return

    # Show last sync time
    if stats.last_sync:
        st.caption(f"Last synced: {stats.last_sync.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    else:
        st.warning("No sync history found. Run: `docker exec task-worker python -m stock_scanner.main broker-sync`")

    # Key metrics
    _render_key_metrics(stats, days)

    st.markdown("---")

    # Open positions and side breakdown
    _render_positions_and_side_breakdown(positions_data, stats)

    st.markdown("---")

    # Equity Curve
    if stats.equity_curve:
        _render_equity_curve(stats.equity_curve)

    # Daily P&L
    if stats.by_day:
        _render_daily_pnl(stats.by_day)

    # Performance by ticker
    if stats.by_ticker:
        _render_performance_by_ticker(stats.by_ticker)

    # Recent trades
    _render_recent_trades(trades_data, days)


def _render_account_overview(balance_data, trend_data):
    """Render account overview section."""
    st.subheader("Account Overview")
    bal_col1, bal_col2, bal_col3, bal_col4 = st.columns(4)

    with bal_col1:
        if trend_data.get("trend") == "up":
            delta_color = "normal"
            trend_arrow = "+"
        elif trend_data.get("trend") == "down":
            delta_color = "inverse"
            trend_arrow = ""
        else:
            delta_color = "off"
            trend_arrow = ""

        st.metric(
            "Total Account Value",
            f"${balance_data['total_value']:,.2f}",
            delta=f"{trend_arrow}${trend_data.get('change', 0):,.2f} ({trend_data.get('change_pct', 0):+.2f}%)" if trend_data.get("change") else None,
            delta_color=delta_color
        )

    with bal_col2:
        st.metric(
            "Invested",
            f"${balance_data['invested']:,.2f}",
            delta=f"{balance_data['invested']/balance_data['total_value']*100:.1f}% of total" if balance_data['total_value'] > 0 else None
        )

    with bal_col3:
        st.metric(
            "Available Cash",
            f"${balance_data['available']:,.2f}",
            delta=f"{balance_data['available']/balance_data['total_value']*100:.1f}% of total" if balance_data['total_value'] > 0 else None
        )

    with bal_col4:
        if trend_data.get("data_points", 0) > 1:
            trend_text = {
                "up": "Increasing",
                "down": "Decreasing",
                "neutral": "Stable"
            }.get(trend_data.get("trend", "neutral"), "N/A")

            trend_emoji = {
                "up": "",
                "down": "",
                "neutral": ""
            }.get(trend_data.get("trend", "neutral"), "")

            st.metric(
                "7-Day Trend",
                f"{trend_emoji} {trend_text}",
                delta=f"Based on {trend_data.get('data_points', 0)} snapshots"
            )
        else:
            st.metric(
                "7-Day Trend",
                "N/A",
                delta="Need more data points"
            )


def _render_key_metrics(stats, days):
    """Render key trading metrics."""
    # Row 1
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Net Profit", f"${stats.net_profit:,.2f}", delta=None)

    with col2:
        st.metric("Win Rate", f"{stats.win_rate:.1f}%", delta=f"{stats.winning_trades}W / {stats.losing_trades}L")

    with col3:
        st.metric(
            "Profit Factor",
            f"{stats.profit_factor:.2f}",
            delta="Good" if stats.profit_factor > 1.5 else ("Fair" if stats.profit_factor > 1 else "Poor")
        )

    with col4:
        st.metric("Total Trades", stats.total_trades, delta=f"Last {days} days")

    # Row 2
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Win", f"${stats.avg_profit:,.2f}", delta=f"+{stats.avg_profit_pct:.1f}%")

    with col2:
        st.metric("Avg Loss", f"${stats.avg_loss:,.2f}", delta=f"-{stats.avg_loss_pct:.1f}%")

    with col3:
        st.metric("Largest Win", f"${stats.largest_win:,.2f}")

    with col4:
        st.metric("Largest Loss", f"${stats.largest_loss:,.2f}")

    # Row 3 - Risk & Streaks
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Expectancy", f"${stats.expectancy:,.2f}", help="Expected value per trade")

    with col2:
        st.metric("Max Drawdown", f"${stats.max_drawdown:,.2f}", delta=f"-{stats.max_drawdown_pct:.1f}%")

    with col3:
        st.metric("Win Streak (max)", stats.max_consecutive_wins)

    with col4:
        st.metric("Loss Streak (max)", stats.max_consecutive_losses)


def _render_positions_and_side_breakdown(positions_data, stats):
    """Render open positions and performance by side."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Open Positions")
        if positions_data.get("positions"):
            pos_df = pd.DataFrame(positions_data["positions"])
            if not pos_df.empty:
                display_df = pos_df[["ticker", "side", "quantity", "entry_price", "current_price", "unrealized_pnl", "profit_pct"]].copy()
                display_df.columns = ["Ticker", "Side", "Qty", "Entry", "Current", "P&L ($)", "P&L (%)"]

                st.dataframe(
                    display_df.style.format({
                        "Entry": "${:.2f}",
                        "Current": "${:.2f}",
                        "P&L ($)": "${:+,.2f}",
                        "P&L (%)": "{:+.2f}%"
                    }).map(
                        lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                        subset=["P&L ($)", "P&L (%)"]
                    ),
                    use_container_width=True,
                    height=250
                )

                st.metric("Total Unrealized P&L", f"${positions_data['total_unrealized_pnl']:+,.2f}")
        else:
            st.info("No open positions")

    with col2:
        st.subheader("Performance by Side")

        side_data = {
            "Metric": ["Trades", "Win Rate", "Profit"],
            "Long": [
                stats.long_trades,
                f"{stats.long_win_rate:.1f}%",
                f"${stats.long_profit:,.2f}"
            ],
            "Short": [
                stats.short_trades,
                f"{stats.short_win_rate:.1f}%",
                f"${stats.short_profit:,.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(side_data), use_container_width=True, hide_index=True)

        st.subheader("Trade Duration")
        st.metric("Avg Hold Time", f"{stats.avg_trade_duration_hours:.1f} hours")


def _render_equity_curve(equity_curve):
    """Render equity curve chart."""
    st.subheader("Equity Curve")

    equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"])
    equity_df["Date"] = pd.to_datetime(equity_df["Date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["Date"],
        y=equity_df["Equity"],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(26, 95, 122, 0.2)',
        line=dict(color='#1a5f7a', width=2),
        name='Equity'
    ))

    fig.update_layout(
        title="Account Equity Over Time",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_daily_pnl(by_day):
    """Render daily P&L chart."""
    st.subheader("Daily Profit/Loss")

    daily_df = pd.DataFrame([
        {"Date": day, "Profit": data["profit"], "Trades": data["trades"]}
        for day, data in sorted(by_day.items())
    ])

    fig = go.Figure()
    colors = ['#28a745' if x >= 0 else '#dc3545' for x in daily_df["Profit"]]

    fig.add_trace(go.Bar(
        x=daily_df["Date"],
        y=daily_df["Profit"],
        marker_color=colors,
        name='Daily P&L',
        hovertemplate='%{x}<br>P&L: $%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title="Daily Profit/Loss",
        xaxis_title="Date",
        yaxis_title="Profit ($)",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_performance_by_ticker(by_ticker):
    """Render performance by ticker table."""
    st.subheader("Performance by Ticker")

    ticker_df = pd.DataFrame([
        {
            "Ticker": ticker,
            "Trades": data["trades"],
            "Wins": data["wins"],
            "Losses": data["losses"],
            "Win Rate": f"{data.get('win_rate', 0):.1f}%",
            "Profit": data["profit"]
        }
        for ticker, data in sorted(by_ticker.items(), key=lambda x: x[1]["profit"], reverse=True)
    ])

    st.dataframe(
        ticker_df.style.format({"Profit": "${:,.2f}"}).map(
            lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
            subset=["Profit"]
        ),
        use_container_width=True,
        height=300
    )


def _render_recent_trades(trades_data, days):
    """Render recent closed trades table."""
    st.subheader("Recent Closed Trades")

    if trades_data.get("trades"):
        trades_df = pd.DataFrame(trades_data["trades"][:50])
        if not trades_df.empty:
            display_trades = trades_df[[
                "ticker", "side", "open_price", "close_price", "profit", "profit_pct", "duration_hours", "close_time"
            ]].copy()
            display_trades.columns = ["Ticker", "Side", "Open", "Close", "Profit ($)", "Profit (%)", "Duration (h)", "Closed"]

            st.dataframe(
                display_trades.style.format({
                    "Open": "${:.2f}",
                    "Close": "${:.2f}",
                    "Profit ($)": "${:+,.2f}",
                    "Profit (%)": "{:+.2f}%",
                    "Duration (h)": "{:.1f}"
                }).map(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                    subset=["Profit ($)", "Profit (%)"]
                ),
                use_container_width=True,
                height=400
            )
    else:
        st.info(f"No closed trades in the last {days} days")
