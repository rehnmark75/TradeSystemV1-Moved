"""
Stock Scanner Watchlists Tab

5 predefined technical watchlists:
- EMA 50 Crossover
- EMA 20 Crossover
- MACD Bullish Cross
- Gap Up Continuation
- RSI Oversold Bounce
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date

# RS Percentile color helpers
def _get_rs_color(percentile):
    """Get color for RS percentile value."""
    if percentile is None or pd.isna(percentile):
        return '#6c757d'
    if percentile >= 90:
        return '#28a745'  # Elite - green
    elif percentile >= 70:
        return '#17a2b8'  # Strong - blue
    elif percentile >= 40:
        return '#ffc107'  # Average - yellow
    else:
        return '#dc3545'  # Weak - red

RS_TREND_ICONS = {
    'improving': '↗️',
    'stable': '➡️',
    'deteriorating': '↘️',
}


# Watchlist definitions - matches worker/app/stock_scanner/scanners/watchlist_scanner.py
WATCHLIST_DEFINITIONS = {
    'ema_50_crossover': {
        'name': 'EMA 50 Crossover',
        'description': 'Price > EMA 200, Price crosses above EMA 50, Volume > 1M/day',
        'icon': '📈',
    },
    'ema_20_crossover': {
        'name': 'EMA 20 Crossover',
        'description': 'Price > EMA 200, Price crosses above EMA 20, Volume > 1M/day',
        'icon': '📊',
    },
    'macd_bullish_cross': {
        'name': 'MACD Bullish Cross',
        'description': 'MACD crosses from negative to positive, Price > EMA 200, Volume > 1M/day',
        'icon': '🔄',
    },
    'gap_up_continuation': {
        'name': 'Gap Up Continuation',
        'description': 'Gap up > 2% today, Closing above open, Price > EMA 200, Volume > 1M/day',
        'icon': '🚀',
    },
    'rsi_oversold_bounce': {
        'name': 'RSI Oversold Bounce',
        'description': 'RSI(14) < 30, Price > EMA 200, Bullish candle, Volume > 1M/day',
        'icon': '💪',
    },
}

# Define which are crossover vs event watchlists
CROSSOVER_WATCHLISTS = {'ema_50_crossover', 'ema_20_crossover', 'macd_bullish_cross'}
EVENT_WATCHLISTS = {'gap_up_continuation', 'rsi_oversold_bounce'}


def render_watchlists_tab(service):
    """Render the Watchlists tab with 5 predefined technical screens."""
    st.markdown("""
    <div class="main-header">
        <h2>Technical Watchlists</h2>
        <p>Crossover watchlists track days since signal - Event watchlists show daily occurrences</p>
    </div>
    """, unsafe_allow_html=True)

    # Watchlist selector and date (date only for event watchlists)
    watchlist_options = list(WATCHLIST_DEFINITIONS.keys())
    watchlist_labels = [f"{WATCHLIST_DEFINITIONS[k]['icon']} {WATCHLIST_DEFINITIONS[k]['name']}" for k in watchlist_options]

    col_watchlist, col_date = st.columns([3, 1])

    with col_watchlist:
        selected_idx = st.selectbox(
            "Select Watchlist",
            range(len(watchlist_options)),
            format_func=lambda i: watchlist_labels[i],
            key="watchlist_selector"
        )

    selected_watchlist = watchlist_options[selected_idx]
    watchlist_info = WATCHLIST_DEFINITIONS[selected_watchlist]

    # Date picker only for event watchlists
    selected_date = None
    with col_date:
        if selected_watchlist in EVENT_WATCHLISTS:
            available_dates = service.get_watchlist_available_dates(selected_watchlist)
            if available_dates:
                selected_date = st.selectbox(
                    "Event Date",
                    available_dates,
                    index=0,
                    key="watchlist_date_selector",
                    help="Select date for gap/RSI events"
                )
            else:
                st.info("No data")
        else:
            st.markdown("""
            <div style="padding: 0.5rem; background: #e8f4f8; border-radius: 5px; font-size: 0.85rem; text-align: center; margin-top: 1.5rem;">
                📊 <b>Live tracking</b><br>
                <span style="color: #666; font-size: 0.75rem;">Days since crossover</span>
            </div>
            """, unsafe_allow_html=True)

    # Get watchlist stats for selected date
    with st.spinner("Loading watchlist data..."):
        watchlist_stats = service.get_watchlist_stats(selected_date)

    # Watchlist info box
    last_scan = watchlist_stats.get('last_scan', 'Not available')
    if isinstance(last_scan, datetime):
        last_scan = last_scan.strftime('%Y-%m-%d')
    elif isinstance(last_scan, date):
        last_scan = str(last_scan)

    total_stocks = watchlist_stats.get('total_stocks_scanned', 0)
    result_count = watchlist_stats.get('counts', {}).get(selected_watchlist, 0)

    is_crossover = selected_watchlist in CROSSOVER_WATCHLISTS
    tracking_info = "Tracks days since crossover (up to 30 days)" if is_crossover else "Single-day events"

    st.markdown(f"""
    <div class="watchlist-info">
        <div style="font-weight: bold; font-size: 1.1rem;">{watchlist_info['icon']} {watchlist_info['name']}</div>
        <div style="color: #555; margin: 0.3rem 0;">{watchlist_info['description']}</div>
        <div style="font-size: 0.85rem; color: #888;">
            Last scan: {last_scan} | {total_stocks:,} stocks scanned | <b>{result_count} active</b> | {tracking_info}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick summary of all watchlists
    st.markdown("### Watchlist Summary")
    summary_cols = st.columns(5)
    for i, (wl_key, wl_info) in enumerate(WATCHLIST_DEFINITIONS.items()):
        with summary_cols[i]:
            count = watchlist_stats.get('counts', {}).get(wl_key, 0)
            is_selected = wl_key == selected_watchlist
            bg_color = '#e8f4f8' if is_selected else '#f8f9fa'
            border = '2px solid #1a5f7a' if is_selected else '1px solid #dee2e6'
            st.markdown(f"""
            <div style="background: {bg_color}; padding: 0.5rem; border-radius: 8px; text-align: center; border: {border};">
                <div style="font-size: 1.5rem;">{wl_info['icon']}</div>
                <div style="font-size: 0.75rem; color: #555;">{wl_info['name'].split()[0]}</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #1a5f7a;">{count}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Get results for selected watchlist and date
    with st.spinner(f"Loading {watchlist_info['name']} results..."):
        df = service.get_watchlist_results(selected_watchlist, selected_date)

    if df.empty:
        st.info(f"No stocks currently match the {watchlist_info['name']} criteria. This scan runs daily.")
        return

    # Results header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {result_count} Stocks Matching Criteria")
    with col2:
        csv = df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, f"watchlist_{selected_watchlist}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    # Format dataframe for display
    display_df = df.copy()
    display_df['Price'] = display_df['price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else '-')
    display_df['Volume'] = display_df['volume'].apply(lambda x: f"{x/1e6:.1f}M" if pd.notnull(x) else '-')
    display_df['Avg Vol'] = display_df['avg_volume'].apply(lambda x: f"{x/1e6:.1f}M" if pd.notnull(x) else '-')
    display_df['EMA 20'] = display_df['ema_20'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else '-')
    display_df['EMA 50'] = display_df['ema_50'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else '-')
    display_df['EMA 200'] = display_df['ema_200'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else '-')
    display_df['RSI'] = display_df['rsi_14'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else '-')
    display_df['1D Chg'] = display_df['price_change_1d'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else '-')
    display_df['Avg/Day'] = display_df['avg_daily_change_5d'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else '-')

    # Days column - for crossover watchlists this is days since crossover
    display_df['Days'] = display_df['days_on_list'].apply(lambda x: f"{int(x)}d" if pd.notnull(x) else '1d')

    # RS Percentile column - color-coded relative strength
    if 'rs_percentile' in display_df.columns:
        def format_rs(row):
            rs = row.get('rs_percentile')
            trend = row.get('rs_trend', '')
            if pd.isna(rs) or rs is None:
                return '-'
            icon = RS_TREND_ICONS.get(trend, '')
            return f"{int(rs)}{icon}"
        display_df['RS'] = display_df.apply(format_rs, axis=1)
    else:
        display_df['RS'] = '-'

    # In Trade column - shows if stock has open broker position with profit
    def format_trade_status(row):
        if row.get('in_trade'):
            profit = row.get('trade_profit', 0) or 0
            side = row.get('trade_side', 'BUY')
            side_icon = '📈' if side == 'BUY' else '📉'
            if profit >= 0:
                return f"{side_icon} +${profit:.0f}"
            else:
                return f"{side_icon} -${abs(profit):.0f}"
        return ''

    display_df['Trade'] = display_df.apply(format_trade_status, axis=1)

    # Crossover date formatting for crossover watchlists
    if 'crossover_date' in display_df.columns and is_crossover:
        display_df['Crossover'] = display_df['crossover_date'].apply(
            lambda x: x.strftime('%m/%d') if pd.notnull(x) else '-'
        )

    # Conditional columns based on watchlist type
    # Include Trade column if any stocks are in trade
    has_trades = display_df['Trade'].any()

    if selected_watchlist == 'gap_up_continuation':
        display_df['Gap %'] = display_df['gap_pct'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else '-')
        cols = ['ticker', 'RS', 'Price', 'Gap %', 'Volume', 'RSI', '1D Chg', 'Avg/Day']
        if has_trades:
            cols.insert(1, 'Trade')
        result_df = display_df[cols].rename(columns={'ticker': 'Ticker'})
    elif selected_watchlist == 'rsi_oversold_bounce':
        cols = ['ticker', 'RS', 'Price', 'RSI', 'Volume', '1D Chg', 'Avg/Day']
        if has_trades:
            cols.insert(1, 'Trade')
        result_df = display_df[cols].rename(columns={'ticker': 'Ticker'})
    elif selected_watchlist == 'macd_bullish_cross':
        display_df['MACD'] = display_df['macd'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else '-')
        cols = ['ticker', 'RS', 'Days', 'Crossover', 'Price', 'MACD', 'Volume', 'RSI', '1D Chg', 'Avg/Day']
        if has_trades:
            cols.insert(1, 'Trade')
        result_df = display_df[cols].rename(columns={'ticker': 'Ticker'})
    else:
        # EMA crossover watchlists
        cols = ['ticker', 'RS', 'Days', 'Crossover', 'Price', 'Volume', 'RSI', '1D Chg', 'Avg/Day']
        if has_trades:
            cols.insert(1, 'Trade')
        result_df = display_df[cols].rename(columns={'ticker': 'Ticker'})

    # Style functions
    def style_change(val):
        if '+' in str(val):
            return 'color: #28a745; font-weight: bold;'
        elif '-' in str(val) and val != '-':
            return 'color: #dc3545;'
        return ''

    def style_rsi(val):
        try:
            rsi = float(val)
            if rsi < 30:
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif rsi > 70:
                return 'background-color: #f8d7da; color: #721c24;'
        except (ValueError, TypeError):
            pass
        return ''

    def style_days(val):
        """Highlight stocks that appear frequently (3+ days = pattern forming)"""
        try:
            days = int(val.replace('d', ''))
            if days >= 5:
                return 'background-color: #cce5ff; color: #004085; font-weight: bold;'
            elif days >= 3:
                return 'background-color: #fff3cd; color: #856404;'
        except (ValueError, TypeError, AttributeError):
            pass
        return ''

    def style_trade(val):
        """Style the trade column - green for profit, red for loss"""
        if not val:
            return ''
        if '+$' in str(val):
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif '-$' in str(val):
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        return ''

    def style_rs(val):
        """Style the RS column based on percentile value"""
        if val == '-' or not val:
            return ''
        try:
            # Extract number from string like "85↗️"
            rs = int(''.join(filter(str.isdigit, str(val))))
            if rs >= 90:
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'  # Elite - green
            elif rs >= 70:
                return 'background-color: #d1ecf1; color: #0c5460; font-weight: bold;'  # Strong - blue
            elif rs >= 40:
                return 'background-color: #fff3cd; color: #856404;'  # Average - yellow
            else:
                return 'background-color: #f8d7da; color: #721c24;'  # Weak - red
        except (ValueError, TypeError):
            pass
        return ''

    styled_df = result_df.style.map(style_change, subset=['1D Chg'])
    if 'Days' in result_df.columns:
        styled_df = styled_df.map(style_days, subset=['Days'])
    if 'RSI' in result_df.columns:
        styled_df = styled_df.map(style_rsi, subset=['RSI'])
    if 'Gap %' in result_df.columns:
        styled_df = styled_df.map(style_change, subset=['Gap %'])
    if 'Trade' in result_df.columns:
        styled_df = styled_df.map(style_trade, subset=['Trade'])
    if 'RS' in result_df.columns:
        styled_df = styled_df.map(style_rs, subset=['RS'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

    # Quick actions
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Quick Actions")
        ticker_list = df['ticker'].tolist()
        tickers_str = ', '.join(ticker_list[:20])
        if len(ticker_list) > 20:
            tickers_str += f" ... and {len(ticker_list) - 20} more"

        st.text_area("Copy Tickers for TradingView", value=','.join(ticker_list), height=80, key="tv_tickers")
        st.caption("Paste into TradingView watchlist")

    with col2:
        st.markdown("### Open Chart & Analysis")
        if ticker_list:
            selected_ticker = st.selectbox("Select stock for analysis", ticker_list, key="watchlist_deep_dive")
            if st.button("📈 Open Chart & Analysis", key="open_chart_analysis"):
                st.session_state.chart_ticker = selected_ticker
                st.session_state.current_chart_ticker = selected_ticker
                st.info(f"Switch to **Chart** tab to see chart and analysis for {selected_ticker}")
