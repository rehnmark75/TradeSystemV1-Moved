"""
Backtest Results Tab Component

Renders backtest history with performance metrics and chart visualization.
Features:
- Backtest execution list with filters
- Performance metrics display
- Chart image display from MinIO
- Signal details table
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional
import urllib.request

from services.backtest_service import BacktestService


def render_backtest_results_tab():
    """Render Backtest Results tab with execution history and charts"""
    service = BacktestService()

    # Header with refresh button
    header_col1, header_col2 = st.columns([6, 1])
    with header_col1:
        st.header("Backtest Results")
    with header_col2:
        if st.button("Refresh", key="backtest_refresh", help="Refresh backtest data"):
            # Clear only backtest-related cache (not all cache)
            service.fetch_backtest_executions.clear()
            service.fetch_backtest_signals.clear()
            service.get_filter_options.clear()
            st.rerun()

    st.markdown("View backtest execution history with performance metrics and charts")

    # Get filter options
    filter_options = service.get_filter_options()
    strategies = filter_options.get('strategies', ['All'])
    epics = filter_options.get('epics', ['All'])

    # Filters row
    col1, col2, col3 = st.columns(3)
    with col1:
        days_filter = st.selectbox(
            "Time Period",
            [7, 14, 30, 60, 90],
            index=1,
            key="backtest_days"
        )
    with col2:
        strategy_filter = st.selectbox(
            "Strategy",
            strategies,
            key="backtest_strategy"
        )
    with col3:
        epic_filter = st.selectbox(
            "Epic/Pair",
            epics,
            key="backtest_epic"
        )

    # Fetch executions
    df = service.fetch_backtest_executions(days_filter, strategy_filter)

    # Apply epic filter client-side (epics_tested is an array)
    if epic_filter != "All" and not df.empty:
        df = df[df['epics_tested'].apply(lambda x: epic_filter in x if isinstance(x, list) else False)]

    if df.empty:
        st.info("No backtest executions found for the selected filters.")
        return

    # Summary metrics
    st.markdown("---")
    total_backtests = len(df)
    completed = len(df[df['status'] == 'completed'])
    with_charts = len(df[df['chart_url'].notna()])
    total_signals = df['signal_count'].sum()
    avg_win_rate = (df['win_count'].sum() / df['signal_count'].sum() * 100) if df['signal_count'].sum() > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Backtests", total_backtests)
    with col2:
        st.metric("Completed", completed)
    with col3:
        st.metric("With Charts", with_charts)
    with col4:
        st.metric("Total Signals", int(total_signals))
    with col5:
        st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")

    st.markdown("---")

    # Pagination settings
    ITEMS_PER_PAGE = 25
    total_pages = (len(df) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE

    # Page selector
    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
    with page_col2:
        if total_pages > 1:
            current_page = st.selectbox(
                f"Page (showing {ITEMS_PER_PAGE} per page)",
                range(1, total_pages + 1),
                key="backtest_page",
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
        else:
            current_page = 1

    # Calculate slice for current page
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, len(df))

    st.caption(f"Showing backtests {start_idx + 1}-{end_idx} of {len(df)}")

    # Display only current page of backtests
    page_df = df.iloc[start_idx:end_idx]
    for idx, row in page_df.iterrows():
        _render_backtest_row(row, service)


def _render_backtest_row(row: pd.Series, service: BacktestService):
    """Render a single backtest execution row with expandable details"""
    # Determine status icon
    status = row.get('status', 'unknown')
    if status == 'completed':
        status_icon = "✅"
    elif status == 'running':
        status_icon = "🔄"
    elif status == 'failed':
        status_icon = "❌"
    else:
        status_icon = "⚪"

    # Format timestamp
    start_time = row.get('start_time', '')
    if isinstance(start_time, pd.Timestamp):
        timestamp_str = start_time.strftime('%Y-%m-%d %H:%M')
    else:
        timestamp_str = str(start_time)[:16] if start_time else 'N/A'

    # Get values
    execution_id = row.get('id', 0)
    strategy = row.get('strategy_name', 'N/A')
    signal_count = row.get('signal_count', 0)
    win_count = row.get('win_count', 0)
    loss_count = row.get('loss_count', 0)
    total_pips = row.get('total_pips', 0)
    has_chart = pd.notna(row.get('chart_url'))

    # Calculate win rate
    win_rate = (win_count / signal_count * 100) if signal_count > 0 else 0

    # Format epics
    epics = row.get('epics_tested', [])
    if isinstance(epics, list):
        # Extract pair names from epics
        pairs = []
        for epic in epics[:3]:  # Show first 3
            parts = str(epic).split('.')
            if len(parts) >= 3:
                pairs.append(parts[2][:6])
            else:
                pairs.append(str(epic)[:6])
        epics_str = ', '.join(pairs)
        if len(epics) > 3:
            epics_str += f" +{len(epics)-3}"
    else:
        epics_str = str(epics)[:20] if epics else 'N/A'

    # Chart indicator
    chart_icon = "📊" if has_chart else ""

    # Create expander title
    pips_color = "+" if total_pips >= 0 else ""
    expander_title = (
        f"{status_icon} {timestamp_str} | {strategy} | {epics_str} | "
        f"{signal_count} signals | {win_rate:.0f}% WR | {pips_color}{total_pips:.1f} pips {chart_icon}"
    )

    with st.expander(expander_title, expanded=False):
        # Two columns: details and chart
        detail_col, chart_col = st.columns([1, 1])

        with detail_col:
            st.markdown("**Execution Details:**")
            st.write(f"- **ID:** {execution_id}")
            st.write(f"- **Strategy:** {strategy}")
            st.write(f"- **Status:** {status_icon} {status}")
            st.write(f"- **Started:** {timestamp_str}")

            # Data period
            data_start = row.get('data_start_date')
            data_end = row.get('data_end_date')
            if data_start and data_end:
                if isinstance(data_start, pd.Timestamp):
                    period = f"{data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}"
                else:
                    period = f"{str(data_start)[:10]} to {str(data_end)[:10]}"
                st.write(f"- **Data Period:** {period}")

            # Pairs tested
            if isinstance(epics, list) and epics:
                st.write(f"- **Pairs Tested:** {len(epics)}")

            st.markdown("---")
            st.markdown("**Performance:**")
            st.write(f"- **Signals:** {signal_count}")
            st.write(f"- **Wins:** {win_count}")
            st.write(f"- **Losses:** {loss_count}")
            st.write(f"- **Win Rate:** {win_rate:.1f}%")
            st.write(f"- **Total Pips:** {pips_color}{total_pips:.1f}")

            # Profit factor (pre-computed in main query to avoid N+1)
            if win_count > 0 and loss_count > 0:
                win_pips = row.get('win_pips', 0)
                loss_pips = row.get('loss_pips', 0)
                if loss_pips > 0:
                    pf = win_pips / loss_pips
                    st.write(f"- **Profit Factor:** {pf:.2f}")
                elif win_pips > 0:
                    st.write("- **Profit Factor:** ∞")

                # Expectancy
                if signal_count > 0:
                    expectancy = total_pips / signal_count
                    st.write(f"- **Expectancy:** {expectancy:.2f} pips/trade")

        with chart_col:
            st.markdown("**Chart:**")

            chart_url = row.get('chart_url')
            if chart_url and pd.notna(chart_url):
                # Lazy load chart only when user clicks
                if st.button("Load Chart", key=f"load_chart_{execution_id}"):
                    try:
                        # Fetch image server-side with shorter timeout
                        with urllib.request.urlopen(chart_url, timeout=5) as response:
                            image_bytes = response.read()
                        st.image(image_bytes, caption="Backtest Chart", width="stretch")
                    except Exception as e:
                        st.warning(f"Chart unavailable: {e}")
            else:
                st.info("No chart available for this backtest")

        # Signals table (collapsed by default)
        if signal_count > 0:
            st.markdown("---")
            show_signals = st.checkbox(
                f"Show {signal_count} signals",
                key=f"show_signals_{execution_id}",
                value=False
            )

            if show_signals:
                signals_df = service.fetch_backtest_signals(execution_id)
                if not signals_df.empty:
                    # Format for display
                    display_df = signals_df[[
                        'signal_timestamp', 'epic', 'signal_type',
                        'entry_price', 'pips_gained', 'trade_result'
                    ]].copy()

                    # Clean up epic names
                    display_df['epic'] = display_df['epic'].apply(
                        lambda x: x.split('.')[2][:6] if '.' in str(x) else str(x)[:6]
                    )

                    # Format timestamp
                    display_df['signal_timestamp'] = pd.to_datetime(
                        display_df['signal_timestamp']
                    ).dt.strftime('%Y-%m-%d %H:%M')

                    # Rename columns
                    display_df.columns = ['Time', 'Pair', 'Type', 'Entry', 'Pips', 'Result']

                    st.dataframe(
                        display_df,
                        width="stretch",
                        hide_index=True
                    )
                else:
                    st.info("No signal details available")
