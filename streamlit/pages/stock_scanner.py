"""
Stock Scanner Dashboard

Consolidated dashboard for the stock scanner system with multiple views:
- Overview: Key metrics, tier distribution, top opportunities
- Watchlist: Browse and filter stocks with advanced filtering
- Signals: View and track ZLMA trading signals
- Deep Dive: Detailed analysis for individual stocks
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import service
import sys
sys.path.insert(0, '/app')
from services.stock_analytics_service import get_stock_service

# Custom CSS
st.markdown("""
<style>
    /* Main header styles */
    .main-header {
        background: linear-gradient(90deg, #1a5f7a 0%, #57c5b6 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .main-header h2 { margin: 0; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1a5f7a;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a5f7a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }

    /* Tier badges */
    .tier-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .tier-1 { background: #28a745; color: white; }
    .tier-2 { background: #17a2b8; color: white; }
    .tier-3 { background: #ffc107; color: black; }
    .tier-4 { background: #6c757d; color: white; }

    /* Signal styles */
    .signal-buy { color: #28a745; font-weight: bold; }
    .signal-sell { color: #dc3545; font-weight: bold; }
    .signal-card {
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .signal-card-buy {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
    }
    .signal-card-sell {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
    }

    /* Trend styles */
    .trend-up { color: #28a745; }
    .trend-down { color: #dc3545; }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e9ecef;
    }

    /* Score bar */
    .score-bar {
        height: 6px;
        background: #e9ecef;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.2rem;
    }
    .score-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* Filter section */
    .filter-section {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# OVERVIEW TAB
# =============================================================================

def render_overview_tab(service):
    """Render the Overview tab."""
    with st.spinner("Loading overview data..."):
        stats = service.get_overview_stats()
        top_opps = service.get_top_opportunities(limit=10)
        recent_signals = service.get_recent_signals(hours=24)

    if not stats:
        st.error("Failed to load data. Please check database connection.")
        return

    # Header with date
    latest_date = stats.get('latest_date', 'Unknown')
    if isinstance(latest_date, datetime):
        latest_date = latest_date.strftime('%Y-%m-%d')

    st.markdown(f"""
    <div class="main-header">
        <h2>📊 Stock Scanner Overview</h2>
        <p>Daily analysis of {stats.get('total_with_metrics', 0):,} US stocks | Last updated: {latest_date}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Stocks Analyzed", f"{stats.get('total_with_metrics', 0):,}")

    with col2:
        st.metric("In Watchlist", f"{stats.get('total_watchlist', 0):,}")

    with col3:
        st.metric("Active Signals (7d)", stats.get('total_signals', 0))

    with col4:
        buy = stats.get('buy_signals', 0)
        sell = stats.get('sell_signals', 0)
        st.metric("BUY / SELL", f"{buy} / {sell}")

    st.markdown("---")

    # Two columns: Tier distribution and Recent signals
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">Tier Distribution</div>', unsafe_allow_html=True)
        tier_stats = stats.get('tier_stats', {})

        if tier_stats:
            tiers = []
            for tier_num in [1, 2, 3, 4]:
                if tier_num in tier_stats:
                    t = tier_stats[tier_num]
                    tiers.append({
                        'Tier': f"Tier {tier_num}",
                        'Count': t['count'],
                        'Avg Score': float(t['avg_score'] or 0),
                        'Avg ATR%': float(t['avg_atr'] or 0)
                    })

            df = pd.DataFrame(tiers)
            colors = ['#28a745', '#17a2b8', '#ffc107', '#6c757d']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['Tier'],
                y=df['Count'],
                text=df['Count'],
                textposition='outside',
                marker_color=colors[:len(df)],
                hovertemplate="<b>%{x}</b><br>Stocks: %{y}<extra></extra>"
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="Number of Stocks",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats summary
            cols = st.columns(4)
            for i, tier_num in enumerate([1, 2, 3, 4]):
                if tier_num in tier_stats:
                    t = tier_stats[tier_num]
                    with cols[i]:
                        st.markdown(f"**Tier {tier_num}**: {t['count']} stocks")
                        st.caption(f"Avg Score: {t['avg_score']} | ATR: {t['avg_atr']}%")

    with col2:
        st.markdown('<div class="section-header">Recent Signals (24h)</div>', unsafe_allow_html=True)

        if recent_signals.empty:
            st.info("No recent signals")
        else:
            for _, row in recent_signals.head(8).iterrows():
                sig_type = row['signal_type']
                sig_class = "signal-card-buy" if sig_type == "BUY" else "signal-card-sell"
                sig_color = "#28a745" if sig_type == "BUY" else "#dc3545"

                timestamp = row['signal_timestamp']
                time_str = timestamp.strftime('%H:%M') if isinstance(timestamp, datetime) else str(timestamp)[:5]

                st.markdown(f"""
                <div class="signal-card {sig_class}">
                    <div style="display: flex; justify-content: space-between;">
                        <span><b>{row['ticker']}</b> <span style="color: {sig_color};">{sig_type}</span></span>
                        <span style="color: #666; font-size: 0.85rem;">{time_str}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #555;">
                        Entry: ${row['entry_price']:.2f} | Conf: {row['confidence']:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Top opportunities
    st.markdown('<div class="section-header">Top 10 Opportunities</div>', unsafe_allow_html=True)

    if not top_opps.empty:
        display_data = []
        for _, row in top_opps.iterrows():
            display_data.append({
                'Rank': row['rank_overall'],
                'Tier': row['tier'],
                'Ticker': row['ticker'],
                'Name': row.get('name', '')[:25] + '...' if len(row.get('name', '')) > 25 else row.get('name', ''),
                'Score': f"{row['score']:.0f}",
                'Price': f"${row['current_price']:.2f}",
                'ATR%': f"{row['atr_percent']:.1f}%",
                '$Vol(M)': f"${row['dollar_vol_m']:.0f}M",
                'Trend': row['trend_strength'].replace('_', ' ').title() if row['trend_strength'] else '-',
                'Signal': row.get('signal_type', '') or '-'
            })

        result_df = pd.DataFrame(display_data)

        def style_tier(val):
            colors = {1: '#28a745', 2: '#17a2b8', 3: '#ffc107', 4: '#6c757d'}
            text_colors = {1: 'white', 2: 'white', 3: 'black', 4: 'white'}
            return f'background-color: {colors.get(val, "#fff")}; color: {text_colors.get(val, "black")}; font-weight: bold;'

        def style_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            return ''

        styled_df = result_df.style.map(style_tier, subset=['Tier']).map(style_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)


# =============================================================================
# WATCHLIST TAB
# =============================================================================

def render_watchlist_sidebar_filters():
    """Render sidebar filters for watchlist."""
    filters = {}

    st.sidebar.markdown("### Watchlist Filters")

    # Tier filter
    tier_options = st.sidebar.multiselect(
        "Tier",
        options=[1, 2, 3, 4],
        default=[1, 2],
        format_func=lambda x: f"Tier {x}"
    )
    filters['tiers'] = tier_options if tier_options else None

    # Score range
    score_range = st.sidebar.slider("Score Range", 0, 100, (40, 100))
    filters['min_score'] = score_range[0]
    filters['max_score'] = score_range[1]

    # ATR% range
    atr_range = st.sidebar.slider("ATR %", 0.0, 15.0, (1.5, 10.0), step=0.5)
    filters['min_atr'] = atr_range[0]
    filters['max_atr'] = atr_range[1]

    # Dollar Volume
    vol_options = {'$1M+': 1, '$10M+': 10, '$25M+': 25, '$50M+': 50, '$100M+': 100}
    selected_vol = st.sidebar.radio("Min Dollar Volume", list(vol_options.keys()), index=1)
    filters['min_dollar_vol'] = vol_options[selected_vol]

    # Trend filter
    trend_options = st.sidebar.multiselect(
        "Trend Strength",
        options=['strong_up', 'up', 'neutral', 'down', 'strong_down'],
        default=['strong_up', 'up'],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    filters['trends'] = trend_options if trend_options else None

    # RSI Range
    rsi_range = st.sidebar.slider("RSI Range", 0, 100, (20, 80))
    filters['min_rsi'] = rsi_range[0]
    filters['max_rsi'] = rsi_range[1]

    # Additional filters
    st.sidebar.markdown("---")
    has_signal = st.sidebar.checkbox("Has active signal", value=False)
    filters['has_signal'] = True if has_signal else None

    is_new = st.sidebar.checkbox("New to tier", value=False)
    filters['is_new_to_tier'] = True if is_new else None

    # Set default values for filters not exposed in UI
    filters['ma_alignments'] = None
    filters['min_rvol'] = 0
    filters['max_rvol'] = 100

    return filters


def render_watchlist_tab(service):
    """Render the Watchlist tab."""
    st.markdown("""
    <div class="main-header">
        <h2>📋 Watchlist Explorer</h2>
        <p>Browse and filter stocks by tier, score, volatility, and more</p>
    </div>
    """, unsafe_allow_html=True)

    filters = render_watchlist_sidebar_filters()

    with st.spinner("Loading watchlist..."):
        df = service.get_watchlist(
            tiers=filters['tiers'],
            min_score=filters['min_score'],
            max_score=filters['max_score'],
            min_atr=filters['min_atr'],
            max_atr=filters['max_atr'],
            min_dollar_vol=filters['min_dollar_vol'],
            trends=filters['trends'],
            ma_alignments=filters['ma_alignments'],
            min_rsi=filters['min_rsi'],
            max_rsi=filters['max_rsi'],
            min_rvol=filters['min_rvol'],
            max_rvol=filters['max_rvol'],
            has_signal=filters['has_signal'],
            is_new_to_tier=filters['is_new_to_tier']
        )

    # Results header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{len(df):,}** stocks match your filters")
    with col2:
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button("Export CSV", csv, f"watchlist_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    if df.empty:
        st.warning("No stocks match your filter criteria. Try adjusting the filters.")
        return

    # Format dataframe
    display_df = df.copy()
    display_df['Score'] = display_df['score'].apply(lambda x: f"{x:.0f}")
    display_df['Price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
    display_df['ATR%'] = display_df['atr_percent'].apply(lambda x: f"{x:.1f}%")
    display_df['$Vol(M)'] = display_df['dollar_vol_m'].apply(lambda x: f"${x:.0f}M")
    display_df['RVol'] = display_df['relative_volume'].apply(lambda x: f"{x:.1f}x" if pd.notnull(x) else '-')
    display_df['1D%'] = display_df['price_change_1d'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else '-')
    display_df['5D%'] = display_df['price_change_5d'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else '-')
    display_df['RSI'] = display_df['rsi_14'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else '-')
    display_df['Trend'] = display_df['trend_strength'].apply(lambda x: x.replace('_', ' ').title() if x else '-')
    display_df['Signal'] = display_df.apply(lambda row: row['latest_signal'] if row['signal_count'] > 0 else '-', axis=1)

    result_df = display_df[['rank_overall', 'tier', 'ticker', 'name', 'Score', 'Price', 'ATR%', '$Vol(M)', 'RVol', '1D%', '5D%', 'RSI', 'Trend', 'Signal']].rename(columns={
        'rank_overall': 'Rank', 'tier': 'Tier', 'ticker': 'Ticker', 'name': 'Name'
    })
    result_df['Name'] = result_df['Name'].apply(lambda x: x[:20] + '...' if len(str(x)) > 20 else x)

    # Style functions
    def style_tier(val):
        colors = {1: '#28a745', 2: '#17a2b8', 3: '#ffc107', 4: '#6c757d'}
        text_colors = {1: 'white', 2: 'white', 3: 'black', 4: 'white'}
        return f'background-color: {colors.get(val, "#fff")}; color: {text_colors.get(val, "black")}; font-weight: bold;'

    def style_signal(val):
        if val == 'BUY':
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif val == 'SELL':
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        return ''

    def style_change(val):
        if '+' in str(val):
            return 'color: #28a745;'
        elif '-' in str(val) and val != '-':
            return 'color: #dc3545;'
        return ''

    styled_df = result_df.style.map(style_tier, subset=['Tier']).map(style_signal, subset=['Signal']).map(style_change, subset=['1D%', '5D%'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)

    # Stock detail
    st.markdown("---")
    st.markdown("### Quick View")

    ticker_options = df['ticker'].tolist()
    selected_ticker = st.selectbox("Select a stock for details", options=ticker_options, index=0 if ticker_options else None)

    if selected_ticker:
        selected_row = df[df['ticker'] == selected_ticker].iloc[0]

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown(f"### {selected_ticker}")
            st.markdown(f"**{selected_row.get('name', '')}**")
            st.markdown(f"Tier {selected_row['tier']} | Rank #{selected_row['rank_overall']}")

        with col2:
            st.markdown("#### Score Breakdown")
            cols = st.columns(4)
            components = [
                ('Volume', float(selected_row.get('volume_score', 0) or 0), 30),
                ('Volatility', float(selected_row.get('volatility_score', 0) or 0), 25),
                ('Momentum', float(selected_row.get('momentum_score', 0) or 0), 30),
                ('Rel Strength', float(selected_row.get('relative_strength_score', 0) or 0), 15)
            ]
            for i, (name, score, max_score) in enumerate(components):
                with cols[i]:
                    st.metric(name, f"{score:.0f}/{max_score}")
                    st.progress(score / max_score)


# =============================================================================
# SIGNALS TAB
# =============================================================================

def render_signals_sidebar_filters():
    """Render sidebar filters for signals."""
    filters = {}

    st.sidebar.markdown("### Signal Filters")

    # Signal type
    signal_type = st.sidebar.radio("Signal Type", ["All", "BUY", "SELL"], index=0)
    filters['signal_type'] = None if signal_type == "All" else signal_type

    # Days back
    days_options = {'Last 24h': 1, 'Last 3 days': 3, 'Last 7 days': 7, 'Last 14 days': 14, 'Last 30 days': 30}
    selected_days = st.sidebar.radio("Time Range", list(days_options.keys()), index=2)
    filters['days_back'] = days_options[selected_days]

    # Minimum confidence
    min_conf = st.sidebar.slider("Min Confidence", 0, 100, 30, step=5)
    filters['min_confidence'] = min_conf

    # Tier filter
    tier_options = st.sidebar.multiselect("Tier Filter", [1, 2, 3, 4], default=[], format_func=lambda x: f"Tier {x}")
    filters['tiers'] = tier_options if tier_options else None

    return filters


def render_signals_tab(service):
    """Render the Signals tab."""
    st.markdown("""
    <div class="main-header">
        <h2>📡 Signal Monitor</h2>
        <p>Track ZLMA trading signals with entry, stop loss, and take profit levels</p>
    </div>
    """, unsafe_allow_html=True)

    filters = render_signals_sidebar_filters()

    with st.spinner("Loading signals..."):
        df = service.get_all_signals(
            signal_type=filters['signal_type'],
            days_back=filters['days_back'],
            min_confidence=filters['min_confidence']
        )

    if filters['tiers'] and not df.empty:
        df = df[df['tier'].isin(filters['tiers'])]

    # Stats
    total = len(df)
    buy_count = len(df[df['signal_type'] == 'BUY']) if not df.empty else 0
    sell_count = len(df[df['signal_type'] == 'SELL']) if not df.empty else 0
    avg_conf = df['confidence'].mean() if not df.empty else 0
    avg_rr = df['risk_reward'].mean() if 'risk_reward' in df.columns and not df.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Signals", total)
    col2.metric("BUY Signals", buy_count)
    col3.metric("SELL Signals", sell_count)
    col4.metric("Avg Confidence", f"{avg_conf:.0f}%")
    col5.metric("Avg R:R", f"{avg_rr:.1f}:1" if avg_rr else "-")

    st.markdown("---")

    if df.empty:
        st.warning("No signals match your filter criteria.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Latest Signals")
        for _, row in df.head(5).iterrows():
            sig_type = row['signal_type']
            sig_class = "signal-card-buy" if sig_type == "BUY" else "signal-card-sell"
            sig_color = "#28a745" if sig_type == "BUY" else "#dc3545"

            tier = row.get('tier')
            tier_str = f"T{int(tier)}" if pd.notnull(tier) else ''

            timestamp = pd.to_datetime(row['signal_timestamp'])
            time_str = timestamp.strftime('%Y-%m-%d %H:%M')

            rr = row.get('risk_reward', 0)
            rr_str = f"{rr:.1f}:1" if rr and rr > 0 else '-'

            st.markdown(f"""
            <div class="signal-card {sig_class}">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>{row['ticker']}</b> <span style="color: {sig_color};">{sig_type}</span> {tier_str}</span>
                    <span style="font-size: 0.8rem; color: #666;">{time_str}</span>
                </div>
                <div style="margin-top: 0.3rem; font-size: 0.85rem;">
                    Entry: ${row['entry_price']:.2f} | SL: ${row['stop_loss']:.2f} | TP: ${row['take_profit']:.2f} | R:R: {rr_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### All Signals")

        display_df = df.copy()
        display_df['Time'] = pd.to_datetime(display_df['signal_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['Entry'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
        display_df['SL'] = display_df['stop_loss'].apply(lambda x: f"${x:.2f}")
        display_df['TP'] = display_df['take_profit'].apply(lambda x: f"${x:.2f}")
        display_df['R:R'] = display_df['risk_reward'].apply(lambda x: f"{x:.1f}:1" if pd.notnull(x) and x > 0 else '-')
        display_df['Conf'] = display_df['confidence'].apply(lambda x: f"{x:.0f}%")
        display_df['Tier'] = display_df['tier'].apply(lambda x: f"T{int(x)}" if pd.notnull(x) else '-')

        result_df = display_df[['Time', 'ticker', 'signal_type', 'Entry', 'SL', 'TP', 'R:R', 'Conf', 'Tier']].rename(columns={'ticker': 'Ticker', 'signal_type': 'Type'})

        def style_type(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            return ''

        styled_df = result_df.style.map(style_type, subset=['Type'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

        csv = df.to_csv(index=False)
        st.download_button("Export Signals CSV", csv, f"signals_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


# =============================================================================
# DEEP DIVE TAB
# =============================================================================

def render_deep_dive_tab(service):
    """Render the Deep Dive tab."""
    st.markdown("""
    <div class="main-header">
        <h2>🔍 Stock Deep Dive</h2>
        <p>Comprehensive analysis for individual stocks</p>
    </div>
    """, unsafe_allow_html=True)

    # Search
    col1, col2 = st.columns([2, 1])

    with col1:
        search_term = st.text_input("Search for a stock", placeholder="Enter ticker or company name...")

    ticker = None

    if search_term:
        with st.spinner("Searching..."):
            results = service.get_ticker_search(search_term, limit=10)

        if results:
            with col2:
                options = [f"{r['ticker']} - {r.get('name', '')[:30]}" for r in results]
                selected = st.selectbox("Select stock", options, label_visibility="collapsed")
                if selected:
                    ticker = selected.split(" - ")[0]
        else:
            st.warning("No stocks found matching your search.")

    if not ticker:
        # Show quick picks
        st.markdown("### Quick Picks")
        with st.spinner("Loading top stocks..."):
            top_stocks = service.get_top_opportunities(limit=8)

        if not top_stocks.empty:
            cols = st.columns(8)
            for i, row in top_stocks.iterrows():
                with cols[i % 8]:
                    if st.button(f"{row['ticker']}\nT{row['tier']}", key=f"quick_{row['ticker']}"):
                        ticker = row['ticker']
                        st.rerun()
        return

    # Fetch stock data
    with st.spinner(f"Loading data for {ticker}..."):
        data = service.get_stock_details(ticker)
        candles = service.get_daily_candles(ticker, days=60)

    if not data or not data.get('instrument'):
        st.error(f"Stock '{ticker}' not found in database.")
        return

    instrument = data.get('instrument', {})
    metrics = data.get('metrics', {})
    watchlist = data.get('watchlist', {})
    signals = data.get('signals', [])

    # Header
    tier = watchlist.get('tier')
    score = watchlist.get('score', 0)
    rank = watchlist.get('rank_overall', '-')

    tier_badge = f'<span class="tier-badge tier-{tier}">Tier {tier}</span>' if tier else ''

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #0d6efd 0%, #6610f2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.8rem; font-weight: bold;">{ticker}</span> {tier_badge}
                <div style="opacity: 0.9;">{instrument.get('name', '')}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.8rem; font-weight: bold;">{score:.0f}</div>
                <div>Score | Rank #{rank}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        price = metrics.get('current_price', 0)
        change_1d = metrics.get('price_change_1d', 0) or 0
        st.metric("Price", f"${price:.2f}", f"{change_1d:+.1f}%")

    with col2:
        atr_pct = watchlist.get('atr_percent', 0) or 0
        st.metric("ATR %", f"{atr_pct:.2f}%")

    with col3:
        dollar_vol = watchlist.get('avg_dollar_volume', 0) or 0
        st.metric("Avg $ Volume", f"${dollar_vol / 1_000_000:.1f}M")

    with col4:
        rsi = metrics.get('rsi_14', 50) or 50
        st.metric("RSI (14)", f"{rsi:.0f}")

    # Score breakdown
    if watchlist:
        st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        components = [
            ('Volume', float(watchlist.get('volume_score', 0) or 0), 30),
            ('Volatility', float(watchlist.get('volatility_score', 0) or 0), 25),
            ('Momentum', float(watchlist.get('momentum_score', 0) or 0), 30),
            ('Rel Strength', float(watchlist.get('relative_strength_score', 0) or 0), 15)
        ]
        for i, (name, score_val, max_score) in enumerate(components):
            with cols[i]:
                pct = score_val / max_score if max_score > 0 else 0
                color = "#28a745" if pct >= 0.7 else "#17a2b8" if pct >= 0.5 else "#ffc107" if pct >= 0.3 else "#dc3545"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{score_val:.0f}/{max_score}</div>
                    <div class="metric-label">{name}</div>
                    <div class="score-bar"><div class="score-fill" style="width: {pct*100}%; background: {color};"></div></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Chart and signals
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Price Chart (Daily)</div>', unsafe_allow_html=True)

        if not candles.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

            fig.add_trace(go.Candlestick(
                x=candles['timestamp'],
                open=candles['open'],
                high=candles['high'],
                low=candles['low'],
                close=candles['close'],
                name='Price'
            ), row=1, col=1)

            # Add signals
            for sig in signals:
                timestamp = sig.get('signal_timestamp')
                entry = sig.get('entry_price')
                sig_type = sig.get('signal_type')

                if timestamp and entry:
                    marker_color = '#28a745' if sig_type == 'BUY' else '#dc3545'
                    marker_symbol = 'triangle-up' if sig_type == 'BUY' else 'triangle-down'

                    fig.add_trace(go.Scatter(
                        x=[timestamp],
                        y=[entry],
                        mode='markers',
                        marker=dict(size=12, color=marker_color, symbol=marker_symbol),
                        name=sig_type,
                        hovertemplate=f"{sig_type}<br>${entry:.2f}<extra></extra>"
                    ), row=1, col=1)

            # Volume
            colors = ['#28a745' if c >= o else '#dc3545' for c, o in zip(candles['close'], candles['open'])]
            fig.add_trace(go.Bar(x=candles['timestamp'], y=candles['volume'], marker_color=colors, name='Volume', opacity=0.7), row=2, col=1)

            fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), showlegend=False, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available")

    with col2:
        st.markdown('<div class="section-header">Signal History</div>', unsafe_allow_html=True)

        if signals:
            for sig in signals[:8]:
                sig_type = sig.get('signal_type', '')
                sig_class = "signal-card-buy" if sig_type == "BUY" else "signal-card-sell"
                sig_color = "#28a745" if sig_type == "BUY" else "#dc3545"

                timestamp = sig.get('signal_timestamp')
                time_str = timestamp.strftime('%Y-%m-%d') if isinstance(timestamp, datetime) else str(timestamp)[:10]

                st.markdown(f"""
                <div class="signal-card {sig_class}">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: {sig_color}; font-weight: bold;">{sig_type}</span>
                        <span style="font-size: 0.8rem; color: #666;">{time_str}</span>
                    </div>
                    <div style="font-size: 0.85rem;">
                        Entry: ${sig.get('entry_price', 0):.2f} | Conf: {sig.get('confidence', 0):.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No signals for this stock")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    service = get_stock_service()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Watchlist", "📡 Signals", "🔍 Deep Dive"])

    with tab1:
        render_overview_tab(service)

    with tab2:
        render_watchlist_tab(service)

    with tab3:
        render_signals_tab(service)

    with tab4:
        render_deep_dive_tab(service)


if __name__ == "__main__":
    main()
