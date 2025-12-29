"""
Trade Performance Tab Component

Renders the trade performance analysis tab with:
- Performance metrics (P&L, win rate, etc.)
- Filterable trade details table
- Export functionality
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict

from services.trading_analytics_service import TradingAnalyticsService


def render_trade_performance_tab():
    """Render the trade performance tab"""
    service = TradingAnalyticsService()

    st.header("Trade Performance Analysis")

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        date_option = st.selectbox(
            "Date Filter",
            ["7_days", "30_days", "90_days"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="trade_date_filter"
        )

    with col2:
        if st.button("Refresh", key="trade_refresh"):
            st.rerun()

    with col3:
        export_enabled = st.checkbox("Enable Export", key="export_checkbox")

    # Map to days
    days_map = {"7_days": 7, "30_days": 30, "90_days": 90}
    days_back = days_map[date_option]

    # Fetch data from service (cached)
    trades_df = service.fetch_trades_dataframe(days_back)

    if trades_df.empty:
        st.warning(f"No trade data found for the last {days_back} days")
        return

    # Calculate metrics
    metrics = _calculate_simple_metrics(trades_df)

    # Display metrics
    st.subheader("Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        pnl = metrics.get('total_pnl', 0)
        pnl_class = 'profit' if pnl > 0 else 'loss' if pnl < 0 else ''
        st.markdown(f"""
        <div class="metric-box">
            <h3 class="{pnl_class}">{pnl:+.2f}</h3>
            <p>Total P&L</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        win_rate = metrics.get('win_rate', 0)
        st.markdown(f"""
        <div class="metric-box">
            <h3>{win_rate:.1f}%</h3>
            <p>Win Rate</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.metric("Completed", metrics.get('completed_trades', 0))

    with col4:
        # Show pending and open trades
        pending = metrics.get('pending_trades', 0)
        open_trades = metrics.get('open_trades', 0)
        st.metric("Pending/Open", f"{pending}/{open_trades}",
                 help="Pending limit orders / Currently open positions")

    with col5:
        # Show expired and rejected trades
        expired = metrics.get('expired_trades', 0)
        rejected = metrics.get('rejected_trades', 0)
        st.metric("Expired/Rejected", f"{expired}/{rejected}",
                 help="Unfilled limit orders / Rejected or cancelled orders")

    # Detailed trades table
    st.subheader("Trade Details")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Get unique trade results from data, with sensible ordering
        result_order = ['WIN', 'LOSS', 'BREAKEVEN', 'OPEN', 'PENDING', 'EXPIRED', 'REJECTED', 'CANCELLED']
        available_results = trades_df['trade_result'].unique() if 'trade_result' in trades_df.columns else []
        ordered_results = [r for r in result_order if r in available_results]
        # Add any results not in our predefined order
        ordered_results += [r for r in available_results if r not in result_order]

        result_filter = st.multiselect(
            "Filter by Result",
            options=ordered_results,
            default=ordered_results,
            key="result_filter"
        )

    with col2:
        direction_filter = st.multiselect(
            "Filter by Direction",
            options=trades_df['direction'].unique() if 'direction' in trades_df.columns else [],
            default=trades_df['direction'].unique() if 'direction' in trades_df.columns else [],
            key="direction_filter"
        )

    with col3:
        symbol_filter = st.multiselect(
            "Filter by Symbol",
            options=trades_df['symbol'].unique() if 'symbol' in trades_df.columns else [],
            default=trades_df['symbol'].unique() if 'symbol' in trades_df.columns else [],
            key="symbol_filter"
        )

    with col4:
        strategy_filter = st.multiselect(
            "Filter by Strategy",
            options=trades_df['strategy'].dropna().unique() if 'strategy' in trades_df.columns else [],
            default=trades_df['strategy'].dropna().unique() if 'strategy' in trades_df.columns else [],
            key="strategy_filter"
        )

    # Apply filters
    filtered_df = trades_df.copy()
    if result_filter and 'trade_result' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['trade_result'].isin(result_filter)]
    if direction_filter and 'direction' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['direction'].isin(direction_filter)]
    if symbol_filter and 'symbol' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['symbol'].isin(symbol_filter)]
    if strategy_filter and 'strategy' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['strategy'].isin(strategy_filter)]

    # Display table
    if not filtered_df.empty:
        display_columns = ['timestamp', 'symbol', 'strategy', 'direction', 'entry_price', 'profit_loss_formatted', 'trade_result', 'status']
        display_columns = [col for col in display_columns if col in filtered_df.columns]

        st.dataframe(
            filtered_df[display_columns].head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Trade Time", format="DD/MM/YY HH:mm"),
                "symbol": "Symbol",
                "strategy": "Strategy",
                "direction": "Direction",
                "entry_price": st.column_config.NumberColumn("Entry Price", format="%.5f"),
                "profit_loss_formatted": "P&L",
                "trade_result": "Result",
                "status": "Status"
            }
        )

        # Export functionality
        if export_enabled:
            st.subheader("Export Data")
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trades_{date_option}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No trades match the selected filters.")


def _calculate_simple_metrics(df: pd.DataFrame) -> Dict:
    """Calculate simple performance metrics from trades dataframe"""
    if df.empty:
        return {'has_data': False}

    # Basic counts
    total_trades = len(df)
    trades_with_pnl = df[df['profit_loss'].notna()]

    # Count truly pending trades (only pending_limit and pending statuses)
    # Exclude limit_not_filled, limit_rejected, limit_cancelled as they are terminal states
    pending_statuses = ('pending', 'pending_limit')
    if 'status' in df.columns:
        pending_trades = df[df['status'].isin(pending_statuses)]
        open_trades = df[df['status'] == 'tracking']
        expired_trades = df[df['status'] == 'limit_not_filled']
        rejected_trades = df[df['status'].isin(['limit_rejected', 'limit_cancelled'])]
    else:
        # Fallback if status column not available
        pending_trades = df[df['profit_loss'].isna()]
        open_trades = pd.DataFrame()
        expired_trades = pd.DataFrame()
        rejected_trades = pd.DataFrame()

    if trades_with_pnl.empty and pending_trades.empty and open_trades.empty:
        return {
            'total_trades': total_trades,
            'pending_trades': len(pending_trades),
            'open_trades': len(open_trades),
            'expired_trades': len(expired_trades),
            'rejected_trades': len(rejected_trades),
            'completed_trades': 0,
            'has_data': False
        }

    # P&L analysis
    winning_trades = trades_with_pnl[trades_with_pnl['profit_loss'] > 0]
    losing_trades = trades_with_pnl[trades_with_pnl['profit_loss'] < 0]

    total_pnl = trades_with_pnl['profit_loss'].sum()
    win_rate = len(winning_trades) / len(trades_with_pnl) * 100 if len(trades_with_pnl) > 0 else 0

    return {
        'total_trades': total_trades,
        'completed_trades': len(trades_with_pnl),
        'pending_trades': len(pending_trades),
        'open_trades': len(open_trades),
        'expired_trades': len(expired_trades),
        'rejected_trades': len(rejected_trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'has_data': True
    }
