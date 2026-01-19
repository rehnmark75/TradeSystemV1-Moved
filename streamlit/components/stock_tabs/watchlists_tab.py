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
import streamlit.components.v1 as components
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


def _render_trade_plan(row: Dict[str, Any]) -> None:
    """Render the Trade Plan section with entry, stop loss, targets, and R:R."""
    price = row.get('price', 0) or 0
    entry_low = row.get('suggested_entry_low')
    entry_high = row.get('suggested_entry_high')
    stop_loss = row.get('suggested_stop_loss')
    target_1 = row.get('suggested_target_1')
    target_2 = row.get('suggested_target_2')
    risk_reward = row.get('risk_reward_ratio')
    risk_percent = row.get('risk_percent')
    atr_14 = row.get('atr_14')
    atr_percent = row.get('atr_percent')
    swing_high = row.get('swing_high')
    swing_low = row.get('swing_low')
    swing_high_date = row.get('swing_high_date')
    swing_low_date = row.get('swing_low_date')
    volume_trend = row.get('volume_trend', '')
    relative_volume = row.get('relative_volume')
    rs_percentile = row.get('rs_percentile')
    rs_trend = row.get('rs_trend', '')
    earnings_date = row.get('earnings_date')
    days_to_earnings = row.get('days_to_earnings')
    daq_earnings_risk = row.get('daq_earnings_risk', False)

    # Check if we have enough data to display
    has_trade_plan = any([entry_low, stop_loss, target_1])
    has_sr_levels = any([swing_high, swing_low])
    has_volatility = any([atr_14, atr_percent])

    if not has_trade_plan and not has_sr_levels and not has_volatility:
        st.caption("No trade plan data available")
        return

    # Trade Plan header
    st.markdown("""
    <div style="font-weight: bold; font-size: 1rem; color: #1a5f7a; margin: 10px 0 8px 0; border-bottom: 2px solid #1a5f7a; padding-bottom: 4px;">
        Trade Plan
    </div>
    """, unsafe_allow_html=True)

    # Earnings warning banner (if applicable)
    if days_to_earnings is not None and not pd.isna(days_to_earnings):
        days_int = int(days_to_earnings)
        earnings_str = earnings_date.strftime('%m/%d') if earnings_date and not pd.isna(earnings_date) else ''
        if days_int <= 3:
            # Critical - earnings imminent
            st.markdown(f"""
            <div style="background: #dc3545; color: white; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9rem;">
                <strong>EARNINGS IMMINENT</strong> - Reporting on {earnings_str} ({days_int} day{"s" if days_int != 1 else ""} away). High volatility expected. Consider reducing position size or avoiding.
            </div>
            """, unsafe_allow_html=True)
        elif days_int <= 7:
            # Warning - earnings within a week
            st.markdown(f"""
            <div style="background: #fd7e14; color: white; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9rem;">
                <strong>EARNINGS WARNING</strong> - Reports {earnings_str} ({days_int} days away). Plan exit or hedge before announcement.
            </div>
            """, unsafe_allow_html=True)
        elif days_int <= 14:
            # Info - earnings upcoming
            st.markdown(f"""
            <div style="background: #ffc107; color: #212529; padding: 6px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.85rem;">
                Earnings: {earnings_str} ({days_int} days) - Monitor for pre-earnings run or positioning
            </div>
            """, unsafe_allow_html=True)
    elif daq_earnings_risk:
        # Fallback to DAQ flag if no specific date
        st.markdown("""
        <div style="background: #dc3545; color: white; padding: 6px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.85rem;">
            <strong>EARNINGS RISK</strong> - Earnings expected within 7 days. High volatility possible.
        </div>
        """, unsafe_allow_html=True)

    # Create three columns for layout
    col1, col2, col3 = st.columns(3)

    with col1:
        # Entry Zone and Stop Loss
        st.markdown('<div style="font-weight: bold; color: #495057; margin-bottom: 4px;">Entry & Risk</div>', unsafe_allow_html=True)

        if entry_low and entry_high and not pd.isna(entry_low) and not pd.isna(entry_high):
            entry_pct_low = ((entry_low - price) / price * 100) if price > 0 else 0
            entry_pct_high = ((entry_high - price) / price * 100) if price > 0 else 0
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Entry Zone:</span>
                <span style="color: #28a745; font-weight: bold;">${entry_low:.2f} - ${entry_high:.2f}</span>
                <span style="color: #888; font-size: 0.75rem;">({entry_pct_low:+.1f}% to {entry_pct_high:+.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

        if stop_loss and not pd.isna(stop_loss):
            sl_pct = ((stop_loss - price) / price * 100) if price > 0 else 0
            sl_color = '#dc3545'
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Stop Loss:</span>
                <span style="color: {sl_color}; font-weight: bold;">${stop_loss:.2f}</span>
                <span style="color: #888; font-size: 0.75rem;">({sl_pct:+.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

        if risk_percent and not pd.isna(risk_percent):
            risk_color = '#dc3545' if risk_percent > 5 else '#ffc107' if risk_percent > 3 else '#28a745'
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Risk:</span>
                <span style="color: {risk_color}; font-weight: bold;">{risk_percent:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Targets
        st.markdown('<div style="font-weight: bold; color: #495057; margin-bottom: 4px;">Targets</div>', unsafe_allow_html=True)

        if target_1 and not pd.isna(target_1):
            t1_pct = ((target_1 - price) / price * 100) if price > 0 else 0
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Target 1:</span>
                <span style="color: #28a745; font-weight: bold;">${target_1:.2f}</span>
                <span style="color: #888; font-size: 0.75rem;">(+{t1_pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

        if target_2 and not pd.isna(target_2):
            t2_pct = ((target_2 - price) / price * 100) if price > 0 else 0
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Target 2:</span>
                <span style="color: #17a2b8; font-weight: bold;">${target_2:.2f}</span>
                <span style="color: #888; font-size: 0.75rem;">(+{t2_pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

        if risk_reward and not pd.isna(risk_reward):
            rr_color = '#28a745' if risk_reward >= 2 else '#ffc107' if risk_reward >= 1.5 else '#dc3545'
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">R:R Ratio:</span>
                <span style="color: {rr_color}; font-weight: bold;">1:{risk_reward:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        # Support/Resistance and Volatility
        st.markdown('<div style="font-weight: bold; color: #495057; margin-bottom: 4px;">Key Levels</div>', unsafe_allow_html=True)

        if swing_high and not pd.isna(swing_high):
            sh_pct = ((swing_high - price) / price * 100) if price > 0 else 0
            date_str = swing_high_date.strftime('%m/%d') if swing_high_date and not pd.isna(swing_high_date) else ''
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Resistance:</span>
                <span style="color: #dc3545; font-weight: bold;">${swing_high:.2f}</span>
                <span style="color: #888; font-size: 0.75rem;">(+{sh_pct:.1f}% {date_str})</span>
            </div>
            """, unsafe_allow_html=True)

        if swing_low and not pd.isna(swing_low):
            sl_pct = ((swing_low - price) / price * 100) if price > 0 else 0
            date_str = swing_low_date.strftime('%m/%d') if swing_low_date and not pd.isna(swing_low_date) else ''
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Support:</span>
                <span style="color: #28a745; font-weight: bold;">${swing_low:.2f}</span>
                <span style="color: #888; font-size: 0.75rem;">({sl_pct:+.1f}% {date_str})</span>
            </div>
            """, unsafe_allow_html=True)

        if atr_percent and not pd.isna(atr_percent):
            atr_color = '#dc3545' if atr_percent > 5 else '#ffc107' if atr_percent > 3 else '#28a745'
            st.markdown(f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">ATR%:</span>
                <span style="color: {atr_color}; font-weight: bold;">{atr_percent:.1f}%</span>
                <span style="color: #888; font-size: 0.75rem;">(${atr_14:.2f})</span>
            </div>
            """, unsafe_allow_html=True) if atr_14 and not pd.isna(atr_14) else None

    # Second row: Volume and RS context
    if volume_trend or relative_volume or rs_percentile:
        st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
        vol_col, rs_col = st.columns(2)

        with vol_col:
            if volume_trend:
                vol_icons = {
                    'accumulation': ('Accumulation', '#28a745', 'Buying pressure'),
                    'distribution': ('Distribution', '#dc3545', 'Selling pressure'),
                    'neutral': ('Neutral', '#6c757d', 'Balanced flow')
                }
                vol_label, vol_color, vol_desc = vol_icons.get(volume_trend, ('Unknown', '#6c757d', ''))
                rvol_str = f" ({relative_volume:.1f}x avg)" if relative_volume and not pd.isna(relative_volume) else ""
                st.markdown(f"""
                <div style="font-size: 0.85rem;">
                    <span style="color: #666;">Volume:</span>
                    <span style="color: {vol_color}; font-weight: bold;">{vol_label}</span>
                    <span style="color: #888; font-size: 0.75rem;">{rvol_str} - {vol_desc}</span>
                </div>
                """, unsafe_allow_html=True)

        with rs_col:
            if rs_percentile and not pd.isna(rs_percentile):
                rs_val = int(rs_percentile)
                rs_icon = RS_TREND_ICONS.get(rs_trend, '')
                rs_color = _get_rs_color(rs_val)
                trend_desc = {'improving': 'gaining strength', 'stable': 'holding steady', 'deteriorating': 'weakening'}.get(rs_trend, '')
                st.markdown(f"""
                <div style="font-size: 0.85rem;">
                    <span style="color: #666;">Relative Strength:</span>
                    <span style="color: {rs_color}; font-weight: bold;">{rs_val}th %ile {rs_icon}</span>
                    <span style="color: #888; font-size: 0.75rem;">({trend_desc})</span>
                </div>
                """, unsafe_allow_html=True)


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

    # Initialize selected watchlist in session state if not present
    if 'selected_watchlist_key' not in st.session_state:
        st.session_state.selected_watchlist_key = watchlist_options[0]

    # Sync the dropdown widget state with our session state (for button clicks)
    current_idx = watchlist_options.index(st.session_state.selected_watchlist_key) if st.session_state.selected_watchlist_key in watchlist_options else 0

    # Initialize or sync dropdown state BEFORE widget renders
    if 'watchlist_selector_dropdown' not in st.session_state:
        st.session_state.watchlist_selector_dropdown = current_idx
    elif st.session_state.watchlist_selector_dropdown != current_idx:
        # Button was clicked - sync dropdown to match
        st.session_state.watchlist_selector_dropdown = current_idx

    # Callback for when dropdown changes
    def on_watchlist_dropdown_change():
        dropdown_idx = st.session_state.watchlist_selector_dropdown
        st.session_state.selected_watchlist_key = watchlist_options[dropdown_idx]

    col_watchlist, col_date = st.columns([3, 1])

    with col_watchlist:
        st.selectbox(
            "Select Watchlist",
            range(len(watchlist_options)),
            format_func=lambda i: watchlist_labels[i],
            key="watchlist_selector_dropdown",
            on_change=on_watchlist_dropdown_change
        )

    selected_watchlist = st.session_state.selected_watchlist_key
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

    # Quick summary of all watchlists - clickable boxes
    st.markdown("### Watchlist Summary")

    # Add custom CSS for clickable summary boxes
    st.markdown("""
    <style>
    .watchlist-summary-box {
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .watchlist-summary-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

    summary_cols = st.columns(5)
    watchlist_keys = list(WATCHLIST_DEFINITIONS.keys())
    for i, (wl_key, wl_info) in enumerate(WATCHLIST_DEFINITIONS.items()):
        with summary_cols[i]:
            count = watchlist_stats.get('counts', {}).get(wl_key, 0)
            is_selected = wl_key == selected_watchlist
            bg_color = '#e8f4f8' if is_selected else '#f8f9fa'
            border = '2px solid #1a5f7a' if is_selected else '1px solid #dee2e6'

            # Create a clickable button styled as a summary box
            button_key = f"wl_summary_{wl_key}"
            if st.button(
                f"{wl_info['icon']} {wl_info['name'].split()[0]}: {count}",
                key=button_key,
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                help=f"Click to view {wl_info['name']} ({count} stocks)"
            ):
                # Update session state and rerun to switch watchlist
                st.session_state.selected_watchlist_key = wl_key
                st.rerun()

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

    # View mode toggle - Expandable vs Table (Expandable is default)
    view_col1, view_col2 = st.columns([3, 1])
    with view_col2:
        view_mode = st.radio(
            "View",
            ["Expandable", "Table"],
            horizontal=True,
            key="watchlist_view_mode",
            help="Expandable view lets you click rows to see DAQ breakdown"
        )

    if view_mode == "Table":
        # Standard table view
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
    else:
        # Expandable view - table-like rows that expand to show DAQ details
        # Add CSS for table-like styling
        st.markdown("""
        <style>
        .expandable-header {
            display: grid;
            grid-template-columns: 80px 60px 70px 60px 80px 80px 60px 70px 70px;
            gap: 8px;
            padding: 8px 12px;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            font-weight: bold;
            font-size: 0.85rem;
            color: #495057;
        }
        .expandable-row {
            display: grid;
            grid-template-columns: 80px 60px 70px 60px 80px 80px 60px 70px 70px;
            gap: 8px;
            padding: 6px 12px;
            align-items: center;
            font-size: 0.9rem;
        }
        .expandable-row:hover {
            background: #f8f9fa;
        }
        .ticker-cell { font-weight: bold; }
        .price-cell { color: #212529; }
        .change-positive { color: #28a745; font-weight: bold; }
        .change-negative { color: #dc3545; font-weight: bold; }
        .rs-badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .rs-elite { background: #d4edda; color: #155724; }
        .rs-strong { background: #d1ecf1; color: #0c5460; }
        .rs-average { background: #fff3cd; color: #856404; }
        .rs-weak { background: #f8d7da; color: #721c24; }
        .daq-badge {
            padding: 2px 8px;
            border-radius: 4px;
            color: white;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .daq-aplus { background: #1e7e34; }
        .daq-a { background: #28a745; }
        .daq-b { background: #17a2b8; }
        .daq-c { background: #ffc107; color: #212529; }
        .daq-d { background: #6c757d; }
        </style>
        """, unsafe_allow_html=True)

        # Sort by crossover date (most recent first) then by days on list
        if 'crossover_date' in df.columns:
            df_sorted = df.sort_values(['crossover_date', 'days_on_list'], ascending=[False, True])
        elif 'days_on_list' in df.columns:
            df_sorted = df.sort_values('days_on_list', ascending=True)
        else:
            df_sorted = df

        # Rows with expanders
        for idx, row in df_sorted.iterrows():
            ticker = row['ticker']
            price = row.get('price', 0) or 0
            volume = row.get('volume', 0) or 0
            rsi = row.get('rsi_14')
            rs = row.get('rs_percentile')
            rs_trend = row.get('rs_trend', '')
            daq_score = row.get('daq_score')
            daq_grade = row.get('daq_grade', '')
            change_1d = row.get('price_change_1d', 0) or 0
            avg_daily = row.get('avg_daily_change_5d', 0) or 0
            days_on_list = row.get('days_on_list', 1) or 1

            # Format values
            price_str = f"${price:.2f}"
            volume_str = f"{volume/1e6:.1f}M" if volume else "-"
            rsi_str = f"{int(rsi)}" if rsi and not pd.isna(rsi) else "-"
            days_str = f"{int(days_on_list)}d"
            avg_str = f"{avg_daily:.1f}%" if avg_daily else "-"

            # RS formatting
            rs_icon = RS_TREND_ICONS.get(rs_trend, '')
            if rs and not pd.isna(rs):
                rs_val = int(rs)
                if rs_val >= 90:
                    rs_class = "rs-elite"
                elif rs_val >= 70:
                    rs_class = "rs-strong"
                elif rs_val >= 40:
                    rs_class = "rs-average"
                else:
                    rs_class = "rs-weak"
                rs_html = f'<span class="rs-badge {rs_class}">{rs_val}{rs_icon}</span>'
            else:
                rs_html = "-"

            # DAQ formatting
            if daq_score and not pd.isna(daq_score):
                daq_val = int(daq_score)
                if daq_val >= 85:
                    daq_class = "daq-aplus"
                elif daq_val >= 70:
                    daq_class = "daq-a"
                elif daq_val >= 60:
                    daq_class = "daq-b"
                elif daq_val >= 50:
                    daq_class = "daq-c"
                else:
                    daq_class = "daq-d"
                daq_html = f'<span class="daq-badge {daq_class}">{daq_val} {daq_grade}</span>'
            else:
                daq_html = "-"

            # Change formatting
            if change_1d >= 0:
                change_html = f'<span class="change-positive">+{change_1d:.1f}%</span>'
            else:
                change_html = f'<span class="change-negative">{change_1d:.1f}%</span>'

            # Risk indicators for expander title
            risk_indicator = ""
            if row.get('daq_earnings_risk'):
                risk_indicator += " ⚠️"
            if row.get('daq_high_short_interest'):
                risk_indicator += " 🩳"

            # Build expander label with key metrics - more comprehensive header
            # Format: TICKER | Xd | RS XX | DAQ XX G | $price (+X.X%)
            days_label = f"{int(days_on_list)}d"
            rs_label = f"RS {int(rs)}" if rs and not pd.isna(rs) else "RS -"
            daq_label = f"DAQ {int(daq_score)} {daq_grade}" if daq_score and not pd.isna(daq_score) else "DAQ -"
            change_sign = "+" if change_1d >= 0 else ""

            # Get crossover date for display
            crossover_date = row.get('crossover_date')
            if crossover_date and not pd.isna(crossover_date):
                crossover_str = f" | {crossover_date.strftime('%m/%d')}"
            else:
                crossover_str = ""

            # Determine if this is an "interesting" stock worth highlighting
            # Criteria: RS >= 70 AND DAQ >= 60 (grade B or better) AND recent crossover (≤5 days)
            rs_val_check = int(rs) if rs and not pd.isna(rs) else 0
            daq_val_check = int(daq_score) if daq_score and not pd.isna(daq_score) else 0
            is_interesting = (rs_val_check >= 70 and daq_val_check >= 60 and days_on_list <= 5)
            is_very_interesting = (rs_val_check >= 80 and daq_val_check >= 70 and days_on_list <= 3)

            # Add highlight indicator to label
            if is_very_interesting:
                highlight_prefix = "🟢 "  # Green circle for very interesting
            elif is_interesting:
                highlight_prefix = "🟡 "  # Yellow circle for interesting
            else:
                highlight_prefix = ""

            with st.expander(f"{highlight_prefix}**{ticker}** | {days_label}{crossover_str} | {rs_label} | {daq_label} | ${price:.2f} ({change_sign}{change_1d:.1f}%){risk_indicator}", expanded=False):
                # Show row data in a formatted grid
                st.markdown(f"""
                <div class="expandable-row" style="background: #f8f9fa; border-radius: 4px; margin-bottom: 8px;">
                    <span class="ticker-cell">{ticker}</span>
                    <span>{rs_html}</span>
                    <span>{daq_html}</span>
                    <span>{days_str}</span>
                    <span class="price-cell">{price_str}</span>
                    <span>{volume_str}</span>
                    <span>{rsi_str}</span>
                    <span>{change_html}</span>
                    <span>{avg_str}</span>
                </div>
                """, unsafe_allow_html=True)

                # Trade Plan section - always show first as it's most actionable
                _render_trade_plan(row.to_dict())

                # DAQ detail breakdown (only if DAQ data exists)
                if daq_score and not pd.isna(daq_score):
                    _render_daq_detail(row.to_dict())

                # Button to open chart and analysis for this stock
                if st.button(f"📈 Open {ticker} Chart & Analysis", key=f"chart_btn_{ticker}"):
                    st.session_state.chart_ticker = ticker
                    st.session_state.current_chart_ticker = ticker
                    # Use components.html to inject JavaScript that clicks the Chart tab
                    components.html("""
                    <script>
                        // Find and click the Chart & Analysis tab (index 7)
                        const tabs = parent.document.querySelectorAll('[data-baseweb="tab"]');
                        if (tabs.length >= 8) {
                            tabs[7].click();
                        }
                    </script>
                    """, height=0)

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
