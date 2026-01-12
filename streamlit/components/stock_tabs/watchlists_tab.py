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
from typing import Dict, Any, Optional

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

# DAQ component weights (for display)
DAQ_WEIGHTS = {
    'mtf': 20,      # Multi-timeframe confluence
    'volume': 10,   # Volume analysis
    'smc': 15,      # Smart Money Concepts
    'quality': 15,  # Financial quality
    'catalyst': 10, # Catalyst timing
    'news': 10,     # News sentiment
    'regime': 10,   # Market regime
    'sector': 10,   # Sector rotation
}


def _render_score_bar(score: Optional[int], max_points: int, label: str) -> str:
    """Render a visual score bar for DAQ components."""
    if score is None or pd.isna(score):
        return f'<div style="color: #999;">{label}: N/A</div>'

    # Calculate weighted contribution (score is 0-100, weight is max points)
    weighted = int((score / 100) * max_points)
    pct = score  # Already 0-100

    # Color based on score
    if pct >= 70:
        color = '#28a745'  # Green
    elif pct >= 50:
        color = '#ffc107'  # Yellow
    else:
        color = '#dc3545'  # Red

    bar_width = min(pct, 100)
    return f'''
    <div style="margin: 2px 0;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 90px; font-size: 0.8rem; color: #555;">{label}</span>
            <div style="flex: 1; background: #e9ecef; border-radius: 4px; height: 12px; max-width: 150px;">
                <div style="width: {bar_width}%; background: {color}; height: 100%; border-radius: 4px;"></div>
            </div>
            <span style="font-size: 0.8rem; font-weight: bold; width: 50px;">{weighted}/{max_points}</span>
        </div>
    </div>
    '''


def _get_risk_badges(row: Dict[str, Any]) -> str:
    """Generate risk flag badges HTML."""
    badges = []
    if row.get('daq_earnings_risk'):
        badges.append('<span style="background: #dc3545; color: white; padding: 1px 6px; border-radius: 10px; font-size: 0.7rem; margin-right: 4px;" title="Earnings within 7 days">EARNINGS</span>')
    if row.get('daq_high_short_interest'):
        badges.append('<span style="background: #fd7e14; color: white; padding: 1px 6px; border-radius: 10px; font-size: 0.7rem; margin-right: 4px;" title="High short interest >20%">HIGH SI</span>')
    if row.get('daq_sector_underperforming'):
        badges.append('<span style="background: #6c757d; color: white; padding: 1px 6px; border-radius: 10px; font-size: 0.7rem;" title="Sector underperforming SPY">SECTOR WEAK</span>')
    return ''.join(badges) if badges else ''


def _render_daq_detail(row: Dict[str, Any]) -> None:
    """Render DAQ detail breakdown inside an expander."""
    daq_score = row.get('daq_score')
    daq_grade = row.get('daq_grade', '')

    if pd.isna(daq_score) or daq_score is None:
        st.info("No DAQ analysis available for this stock")
        return

    # Grade color
    grade_colors = {
        'A+': '#1e7e34', 'A': '#28a745', 'B': '#17a2b8',
        'C': '#ffc107', 'D': '#6c757d'
    }
    grade_color = grade_colors.get(daq_grade, '#6c757d')

    # Risk badges
    risk_html = _get_risk_badges(row)

    # Header with score and grade
    st.markdown(f'''
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <div style="background: {grade_color}; color: white; padding: 8px 16px; border-radius: 8px; font-size: 1.2rem; font-weight: bold;">
            {int(daq_score)} {daq_grade}
        </div>
        <div>{risk_html}</div>
    </div>
    ''', unsafe_allow_html=True)

    # Three columns for component categories
    col1, col2, col3 = st.columns(3)

    with col1:
        # Calculate technical total
        tech_total = sum([
            int((row.get('daq_mtf_score', 0) or 0) / 100 * 20),
            int((row.get('daq_volume_score', 0) or 0) / 100 * 10),
            int((row.get('daq_smc_score', 0) or 0) / 100 * 15),
        ])
        st.markdown(f'<div style="font-weight: bold; margin-bottom: 6px; color: #1a5f7a;">TECHNICAL ({tech_total}/45)</div>', unsafe_allow_html=True)
        st.markdown(_render_score_bar(row.get('daq_mtf_score'), 20, 'MTF'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(row.get('daq_volume_score'), 10, 'Volume'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(row.get('daq_smc_score'), 15, 'SMC'), unsafe_allow_html=True)

    with col2:
        # Calculate fundamental total
        fund_total = sum([
            int((row.get('daq_quality_score', 0) or 0) / 100 * 15),
            int((row.get('daq_catalyst_score', 0) or 0) / 100 * 10),
        ])
        st.markdown(f'<div style="font-weight: bold; margin-bottom: 6px; color: #1a5f7a;">FUNDAMENTAL ({fund_total}/25)</div>', unsafe_allow_html=True)
        st.markdown(_render_score_bar(row.get('daq_quality_score'), 15, 'Quality'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(row.get('daq_catalyst_score'), 10, 'Catalyst'), unsafe_allow_html=True)

    with col3:
        # Calculate contextual total
        ctx_total = sum([
            int((row.get('daq_news_score', 0) or 0) / 100 * 10),
            int((row.get('daq_regime_score', 0) or 0) / 100 * 10),
            int((row.get('daq_sector_score', 0) or 0) / 100 * 10),
        ])
        st.markdown(f'<div style="font-weight: bold; margin-bottom: 6px; color: #1a5f7a;">CONTEXTUAL ({ctx_total}/30)</div>', unsafe_allow_html=True)
        st.markdown(_render_score_bar(row.get('daq_news_score'), 10, 'News'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(row.get('daq_regime_score'), 10, 'Regime'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(row.get('daq_sector_score'), 10, 'Sector'), unsafe_allow_html=True)


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

    # DAQ Score column - Deep Analysis Quality score if available
    if 'daq_score' in display_df.columns:
        def format_daq(row):
            score = row.get('daq_score')
            grade = row.get('daq_grade', '')
            if pd.isna(score) or score is None:
                return '-'
            return f"{int(score)} {grade}" if grade else f"{int(score)}"
        display_df['DAQ'] = display_df.apply(format_daq, axis=1)
    else:
        display_df['DAQ'] = '-'

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
    # Include DAQ column if any stocks have DAQ scores
    has_daq = 'daq_score' in df.columns and df['daq_score'].notna().any()

    if selected_watchlist == 'gap_up_continuation':
        display_df['Gap %'] = display_df['gap_pct'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else '-')
        cols = ['ticker', 'RS', 'Price', 'Gap %', 'Volume', 'RSI', '1D Chg', 'Avg/Day']
        if has_daq:
            cols.insert(2, 'DAQ')
        if has_trades:
            cols.insert(1, 'Trade')
        result_df = display_df[cols].rename(columns={'ticker': 'Ticker'})
    elif selected_watchlist == 'rsi_oversold_bounce':
        cols = ['ticker', 'RS', 'Price', 'RSI', 'Volume', '1D Chg', 'Avg/Day']
        if has_daq:
            cols.insert(2, 'DAQ')
        if has_trades:
            cols.insert(1, 'Trade')
        result_df = display_df[cols].rename(columns={'ticker': 'Ticker'})
    elif selected_watchlist == 'macd_bullish_cross':
        display_df['MACD'] = display_df['macd'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else '-')
        cols = ['ticker', 'RS', 'Days', 'Crossover', 'Price', 'MACD', 'Volume', 'RSI', '1D Chg', 'Avg/Day']
        if has_daq:
            cols.insert(2, 'DAQ')
        if has_trades:
            cols.insert(1, 'Trade')
        result_df = display_df[cols].rename(columns={'ticker': 'Ticker'})
    else:
        # EMA crossover watchlists
        cols = ['ticker', 'RS', 'Days', 'Crossover', 'Price', 'Volume', 'RSI', '1D Chg', 'Avg/Day']
        if has_daq:
            cols.insert(2, 'DAQ')
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

    def style_daq(val):
        """Style the DAQ column based on score and grade"""
        if val == '-' or not val:
            return ''
        try:
            # Extract score from string like "75 A" or "85 A+"
            score = int(''.join(filter(str.isdigit, str(val).split()[0])))
            if score >= 85:  # A+
                return 'background-color: #1e7e34; color: white; font-weight: bold;'
            elif score >= 70:  # A
                return 'background-color: #28a745; color: white; font-weight: bold;'
            elif score >= 60:  # B
                return 'background-color: #17a2b8; color: white;'
            elif score >= 50:  # C
                return 'background-color: #ffc107; color: #212529;'
            else:  # D
                return 'background-color: #6c757d; color: white;'
        except (ValueError, TypeError, IndexError):
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
    if 'DAQ' in result_df.columns:
        styled_df = styled_df.map(style_daq, subset=['DAQ'])

    # View mode toggle - Table vs Detail (with DAQ breakdown)
    view_col1, view_col2 = st.columns([3, 1])
    with view_col2:
        view_mode = st.radio(
            "View",
            ["Table", "Detail"],
            horizontal=True,
            key="watchlist_view_mode",
            help="Detail view shows DAQ score breakdown for each stock"
        )

    if view_mode == "Table":
        # Standard table view
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
    else:
        # Detail view with expandable DAQ breakdown
        st.markdown('<div style="max-height: 600px; overflow-y: auto;">', unsafe_allow_html=True)

        for idx, row in df.iterrows():
            ticker = row['ticker']
            price = row.get('price', 0)
            rs = row.get('rs_percentile')
            rs_trend = row.get('rs_trend', '')
            daq_score = row.get('daq_score')
            daq_grade = row.get('daq_grade', '')
            change_1d = row.get('price_change_1d', 0)

            # Build summary line
            rs_icon = RS_TREND_ICONS.get(rs_trend, '')
            rs_str = f"RS {int(rs)}{rs_icon}" if rs and not pd.isna(rs) else ""

            # DAQ badge with color
            if daq_score and not pd.isna(daq_score):
                grade_colors = {'A+': '#1e7e34', 'A': '#28a745', 'B': '#17a2b8', 'C': '#ffc107', 'D': '#6c757d'}
                daq_color = grade_colors.get(daq_grade, '#6c757d')
                daq_str = f'<span style="background: {daq_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem;">{int(daq_score)} {daq_grade}</span>'
            else:
                daq_str = ""

            # Risk badges
            risk_badges = _get_risk_badges(row.to_dict())

            # Change string for expander title (plain text)
            if change_1d and not pd.isna(change_1d):
                change_prefix = "+" if change_1d >= 0 else ""
                change_str_plain = f"({change_prefix}{change_1d:.1f}%)"
                change_color = '#28a745' if change_1d >= 0 else '#dc3545'
                change_str_html = f'<span style="color: {change_color}; font-weight: bold;">{change_1d:+.1f}%</span>'
            else:
                change_str_plain = ""
                change_str_html = ""

            # Build expander label with DAQ indicator
            daq_indicator = f" | DAQ {int(daq_score)}{daq_grade}" if daq_score and not pd.isna(daq_score) else ""
            risk_indicator = ""
            if row.get('daq_earnings_risk'):
                risk_indicator += " ⚠️"
            if row.get('daq_high_short_interest'):
                risk_indicator += " 🩳"

            # Expander title with key metrics
            with st.expander(f"{ticker} - ${price:.2f} {change_str_plain}{daq_indicator}{risk_indicator}", expanded=False):
                # Summary row
                st.markdown(f'''
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap;">
                    <span style="font-size: 1.1rem; font-weight: bold;">{ticker}</span>
                    <span style="color: #555;">${price:.2f}</span>
                    {change_str_html}
                    <span style="background: #e8f4f8; padding: 2px 8px; border-radius: 4px;">{rs_str}</span>
                    {daq_str}
                    {risk_badges}
                </div>
                ''', unsafe_allow_html=True)

                # DAQ detail breakdown (only if DAQ data exists)
                if daq_score and not pd.isna(daq_score):
                    st.markdown("---")
                    _render_daq_detail(row.to_dict())
                else:
                    st.caption("No DAQ analysis available - run deep analysis to see breakdown")

        st.markdown('</div>', unsafe_allow_html=True)

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
