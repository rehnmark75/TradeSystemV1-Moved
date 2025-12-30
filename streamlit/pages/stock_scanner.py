"""
Stock Scanner Dashboard

Unified dashboard for the stock scanner system with 6 tabs:
- Dashboard: Quick snapshot with scanner leaderboard, top signals, quality distribution
- Signals: Unified signal browser with filters (scanner, tier, Claude grade, signal type)
- Watchlists: 5 predefined technical screens (EMA crossovers, MACD, Gap Up, RSI)
- Scanner Performance: Per-scanner backtest metrics and performance analysis
- Deep Dive: Comprehensive analysis for individual stocks (Claude AI, charts, technicals)
- Broker Stats: Real trading performance from RoboMarkets
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
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

    /* Quality tier badges (A+, A, B, C, D) */
    .tier-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .tier-a-plus { background: #1e7e34; color: white; }
    .tier-a { background: #28a745; color: white; }
    .tier-b { background: #17a2b8; color: white; }
    .tier-c { background: #ffc107; color: black; }
    .tier-d { background: #6c757d; color: white; }

    /* Scanner leaderboard */
    .leaderboard-item {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.3rem;
        background: #f8f9fa;
    }
    .leaderboard-rank {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1a5f7a;
        width: 30px;
    }
    .leaderboard-name {
        flex-grow: 1;
        font-weight: 500;
    }
    .leaderboard-pf {
        font-weight: bold;
        color: #28a745;
    }

    /* Signal comparison */
    .comparison-card {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }
    .comparison-selected {
        border-color: #1a5f7a;
        background: #e8f4f8;
    }

    /* Watchlist selector */
    .watchlist-info {
        background: #e8f4f8;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1a5f7a;
    }

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
# DASHBOARD TAB
# =============================================================================

def render_dashboard_tab(service):
    """Render the Dashboard tab - quick snapshot of scanner system."""
    with st.spinner("Loading dashboard..."):
        stats = service.get_dashboard_stats()
        leaderboard = service.get_scanner_leaderboard()
        top_signals = service.get_top_signals(limit=5, min_tier='A')

    if not stats:
        st.info("No dashboard data available. Run the scanner to generate signals.")
        return

    # Header
    last_scan = stats.get('last_scan', 'Unknown')
    if isinstance(last_scan, datetime):
        last_scan = last_scan.strftime('%Y-%m-%d %H:%M')

    st.markdown(f"""
    <div class="main-header">
        <h2>📊 Stock Scanner Dashboard</h2>
        <p>Scanning {stats.get('total_stocks', 3454):,} US stocks via 7 backtested strategies | Last scan: {last_scan}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics - Row 1
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Signals", stats.get('active_signals', 0))

    with col2:
        st.metric("High Quality (A+/A)", stats.get('high_quality', 0),
                  help="Signals with quality tier A+ or A")

    with col3:
        st.metric("Claude Analyzed", stats.get('claude_analyzed', 0),
                  help="Signals with AI analysis complete")

    with col4:
        st.metric("Today's Signals", stats.get('today_signals', 0))

    st.markdown("---")

    # Two columns: Scanner Leaderboard and Top Signals
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">🏆 Scanner Leaderboard (by Profit Factor)</div>', unsafe_allow_html=True)

        if leaderboard:
            # Create horizontal bar chart
            scanner_names = [s['scanner_name'].replace('_', ' ').title() for s in leaderboard]
            profit_factors = [float(s.get('profit_factor', 1.0)) for s in leaderboard]
            win_rates = [float(s.get('win_rate', 0)) for s in leaderboard]

            # Reverse for horizontal bar chart (top scanner at top)
            scanner_names = scanner_names[::-1]
            profit_factors = profit_factors[::-1]

            # Color gradient based on PF
            colors = ['#28a745' if pf >= 2.0 else '#17a2b8' if pf >= 1.5 else '#ffc107' if pf >= 1.0 else '#dc3545'
                      for pf in profit_factors]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=scanner_names,
                x=profit_factors,
                orientation='h',
                text=[f'{pf:.2f}' for pf in profit_factors],
                textposition='outside',
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Profit Factor: %{x:.2f}<extra></extra>"
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=60, t=20, b=20),
                xaxis_title="Profit Factor",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

            # Leaderboard details
            st.markdown("**Backtest Performance:**")
            for i, scanner in enumerate(leaderboard[:5]):
                name = scanner['scanner_name'].replace('_', ' ').title()
                pf = float(scanner.get('profit_factor', 1.0))
                wr = float(scanner.get('win_rate', 0))
                signals = scanner.get('total_signals', 0)

                pf_color = '#28a745' if pf >= 2.0 else '#17a2b8' if pf >= 1.5 else '#ffc107'
                st.markdown(f"{i+1}. **{name}** - PF: <span style='color:{pf_color}'>{pf:.2f}</span> | WR: {wr:.0f}% | Signals: {signals}", unsafe_allow_html=True)
        else:
            st.info("No scanner performance data available. Run backtests to generate metrics.")

    with col2:
        st.markdown('<div class="section-header">⭐ Top A+/A Signals</div>', unsafe_allow_html=True)

        if top_signals:
            for signal in top_signals:
                ticker = signal.get('ticker', 'N/A')
                sig_type = signal.get('signal_type', 'BUY')
                quality_tier = signal.get('quality_tier', 'B')
                score = signal.get('composite_score', 0)
                entry = signal.get('entry_price', 0)
                scanner = signal.get('scanner_name', '').replace('_', ' ').title()

                sig_class = "signal-card-buy" if sig_type == "BUY" else "signal-card-sell"
                sig_color = "#28a745" if sig_type == "BUY" else "#dc3545"
                tier_class = f"tier-{quality_tier.lower().replace('+', '-plus')}"

                # Claude analysis badge
                claude_grade = signal.get('claude_grade')
                claude_badge = ""
                if claude_grade:
                    grade_colors = {'A+': '#1e7e34', 'A': '#28a745', 'B': '#17a2b8', 'C': '#ffc107', 'D': '#6c757d'}
                    grade_color = grade_colors.get(claude_grade, '#6c757d')
                    claude_badge = f'<span style="background-color: {grade_color}; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.7rem; margin-left: 5px;">{claude_grade}</span>'

                st.markdown(f"""
                <div class="signal-card {sig_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><b>{ticker}</b> <span style="color: {sig_color};">{sig_type}</span>{claude_badge}</span>
                        <span class="tier-badge {tier_class}">{quality_tier}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #555;">
                        Entry: ${entry:.2f} | Score: {score}
                    </div>
                    <div style="font-size: 0.75rem; color: #888;">
                        {scanner}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high-quality signals available.")

    st.markdown("---")

    # Bottom row: Quality Distribution and Scanner Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Quality Tier Distribution</div>', unsafe_allow_html=True)
        tier_dist = stats.get('tier_distribution', {})

        if tier_dist:
            tiers = ['A+', 'A', 'B', 'C', 'D']
            counts = [tier_dist.get(t, 0) for t in tiers]
            colors = ['#1e7e34', '#28a745', '#17a2b8', '#ffc107', '#6c757d']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tiers,
                y=counts,
                text=counts,
                textposition='outside',
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>Signals: %{y}<extra></extra>"
            ))
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=40),
                yaxis_title="Number of Signals",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tier distribution data available.")

    with col2:
        st.markdown('<div class="section-header">Signals by Scanner</div>', unsafe_allow_html=True)
        scanner_dist = stats.get('scanner_distribution', {})

        if scanner_dist:
            scanner_names = [k.replace('_', ' ').title() for k in scanner_dist.keys()]
            scanner_counts = list(scanner_dist.values())

            fig = go.Figure(data=[go.Pie(
                labels=scanner_names,
                values=scanner_counts,
                hole=0.4,
                textinfo='label+value',
                marker_colors=px.colors.qualitative.Set2[:len(scanner_names)]
            )])
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scanner distribution data available.")


# =============================================================================
# WATCHLISTS TAB (NEW - Predefined Technical Screens)
# =============================================================================

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


def render_watchlists_tab(service):
    """Render the Watchlists tab with 5 predefined technical screens."""
    st.markdown("""
    <div class="main-header">
        <h2>📋 Technical Watchlists</h2>
        <p>Crossover watchlists track days since signal • Event watchlists show daily occurrences</p>
    </div>
    """, unsafe_allow_html=True)

    # Watchlist selector and date (date only for event watchlists)
    watchlist_options = list(WATCHLIST_DEFINITIONS.keys())
    watchlist_labels = [f"{WATCHLIST_DEFINITIONS[k]['icon']} {WATCHLIST_DEFINITIONS[k]['name']}" for k in watchlist_options]

    # Define which are crossover vs event watchlists
    crossover_watchlists = {'ema_50_crossover', 'ema_20_crossover', 'macd_bullish_cross'}
    event_watchlists = {'gap_up_continuation', 'rsi_oversold_bounce'}

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
        if selected_watchlist in event_watchlists:
            # Get available dates for event watchlists
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

    # Show different info based on watchlist type
    is_crossover = selected_watchlist in crossover_watchlists
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

    # Days column - for crossover watchlists this is days since crossover
    display_df['Days'] = display_df['days_on_list'].apply(lambda x: f"{int(x)}d" if pd.notnull(x) else '1d')

    # Crossover date formatting for crossover watchlists
    if 'crossover_date' in display_df.columns and is_crossover:
        display_df['Crossover'] = display_df['crossover_date'].apply(
            lambda x: x.strftime('%m/%d') if pd.notnull(x) else '-'
        )

    # Conditional columns based on watchlist type (no EMA columns - too cluttered)
    if selected_watchlist == 'gap_up_continuation':
        display_df['Gap %'] = display_df['gap_pct'].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else '-')
        result_df = display_df[['ticker', 'Price', 'Gap %', 'Volume', 'RSI', '1D Chg']].rename(
            columns={'ticker': 'Ticker'}
        )
    elif selected_watchlist == 'rsi_oversold_bounce':
        result_df = display_df[['ticker', 'Price', 'RSI', 'Volume', '1D Chg']].rename(
            columns={'ticker': 'Ticker'}
        )
    elif selected_watchlist == 'macd_bullish_cross':
        display_df['MACD'] = display_df['macd'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else '-')
        result_df = display_df[['ticker', 'Days', 'Crossover', 'Price', 'MACD', 'Volume', 'RSI', '1D Chg']].rename(
            columns={'ticker': 'Ticker'}
        )
    else:
        # EMA crossover watchlists
        result_df = display_df[['ticker', 'Days', 'Crossover', 'Price', 'Volume', 'RSI', '1D Chg']].rename(
            columns={'ticker': 'Ticker'}
        )

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
                return 'background-color: #cce5ff; color: #004085; font-weight: bold;'  # Blue - very frequent
            elif days >= 3:
                return 'background-color: #fff3cd; color: #856404;'  # Yellow - noteworthy
        except (ValueError, TypeError, AttributeError):
            pass
        return ''

    styled_df = result_df.style.map(style_change, subset=['1D Chg'])
    if 'Days' in result_df.columns:
        styled_df = styled_df.map(style_days, subset=['Days'])
    if 'RSI' in result_df.columns:
        styled_df = styled_df.map(style_rsi, subset=['RSI'])
    if 'Gap %' in result_df.columns:
        styled_df = styled_df.map(style_change, subset=['Gap %'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

    # Quick actions
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Quick Actions")
        ticker_list = df['ticker'].tolist()
        tickers_str = ', '.join(ticker_list[:20])  # Limit to 20 for display
        if len(ticker_list) > 20:
            tickers_str += f" ... and {len(ticker_list) - 20} more"

        st.text_area("Copy Tickers for TradingView", value=','.join(ticker_list), height=80, key="tv_tickers")
        st.caption("Paste into TradingView watchlist")

    with col2:
        st.markdown("### Open in Deep Dive")
        if ticker_list:
            selected_ticker = st.selectbox("Select stock for analysis", ticker_list, key="watchlist_deep_dive")
            if st.button("🔍 Open Deep Dive", key="open_deep_dive"):
                st.session_state.deep_dive_ticker = selected_ticker
                st.session_state.current_deep_dive_ticker = selected_ticker
                st.info(f"Switch to Deep Dive tab to see analysis for {selected_ticker}")


# =============================================================================
# DEEP DIVE TAB
# =============================================================================

def render_deep_dive_tab(service):
    """Render the Deep Dive tab with comprehensive stock analysis."""
    st.markdown("""
    <div class="main-header">
        <h2>🔍 Stock Deep Dive</h2>
        <p>Comprehensive analysis for individual stocks</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for Claude analysis
    if 'deep_dive_claude_analysis' not in st.session_state:
        st.session_state.deep_dive_claude_analysis = {}

    # Check if we have a ticker from navigation (e.g., from Top Picks table)
    # or a currently active deep dive ticker
    navigated_ticker = st.session_state.get('deep_dive_ticker', None)
    current_deep_dive = st.session_state.get('current_deep_dive_ticker', None)

    if navigated_ticker:
        # New navigation - use this ticker and persist it
        st.session_state.current_deep_dive_ticker = navigated_ticker
        st.session_state.deep_dive_ticker = None  # Clear navigation flag
        ticker = navigated_ticker
        st.info(f"Analyzing **{ticker}** from Top Picks")
    elif current_deep_dive:
        # Continue viewing the same ticker (e.g., after Fetch News rerun)
        ticker = current_deep_dive
    else:
        ticker = None

    # Search section - show if no ticker OR if user wants to search for another
    show_search = not ticker
    if ticker:
        # Show a "Search Another" button to allow searching for different stock
        col_ticker, col_clear = st.columns([4, 1])
        with col_clear:
            if st.button("🔍 Search Another", key="clear_deep_dive"):
                st.session_state.current_deep_dive_ticker = None
                st.rerun()

    if show_search:
        col1, col2 = st.columns([2, 1])

        with col1:
            search_term = st.text_input("Search for a stock", placeholder="Enter ticker or company name...")

        if search_term:
            with st.spinner("Searching..."):
                results = service.get_ticker_search(search_term, limit=10)

            if results:
                with col2:
                    options = [f"{r['ticker']} - {r.get('name', '')[:30]}" for r in results]
                    selected = st.selectbox("Select stock", options, label_visibility="collapsed")
                    if selected:
                        ticker = selected.split(" - ")[0]
                        # Persist ticker for actions like Fetch News that trigger rerun
                        st.session_state.current_deep_dive_ticker = ticker
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
                        st.session_state.current_deep_dive_ticker = ticker
                        st.rerun()
        return

    # Fetch all stock data
    with st.spinner(f"Loading comprehensive data for {ticker}..."):
        data = service.get_stock_details(ticker)
        candles = service.get_daily_candles(ticker, days=90)  # Extended for better MAs
        scanner_signals = service.get_scanner_signals_for_ticker(ticker, limit=10)
        fundamentals = service.get_full_fundamentals(ticker)

    if not data or not data.get('instrument'):
        st.error(f"Stock '{ticker}' not found in database.")
        return

    instrument = data.get('instrument', {})
    metrics = data.get('metrics', {})
    watchlist = data.get('watchlist', {})

    # Get the most recent active signal (if any)
    active_signal = scanner_signals[0] if scanner_signals else None

    # ==========================================================================
    # SECTION 1: Enhanced Header with Sector/Industry
    # ==========================================================================
    tier = watchlist.get('tier')
    score = watchlist.get('score', 0)
    rank = watchlist.get('rank_overall', '-')
    sector = instrument.get('sector', '') or fundamentals.get('sector', '')
    industry = instrument.get('industry', '') or fundamentals.get('industry', '')

    tier_badge = f'<span class="tier-badge tier-{tier}">Tier {tier}</span>' if tier else ''
    sector_info = f"{sector} | {industry}" if sector and industry else sector or industry or ""

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #0d6efd 0%, #6610f2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.8rem; font-weight: bold;">{ticker}</span> {tier_badge}
                <div style="opacity: 0.9; font-size: 1.1rem;">{instrument.get('name', '')}</div>
                <div style="opacity: 0.7; font-size: 0.85rem; margin-top: 0.2rem;">{sector_info}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.8rem; font-weight: bold;">{score:.0f}</div>
                <div>Score | Rank #{rank}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        price = metrics.get('current_price', 0)
        change_1d = metrics.get('price_change_1d', 0) or 0
        st.metric("Price", f"${price:.2f}", f"{change_1d:+.1f}%")

    with col2:
        atr_pct = watchlist.get('atr_percent', 0) or 0
        avg_daily = watchlist.get('avg_daily_change_5d', 0) or metrics.get('avg_daily_change_5d', 0) or 0
        st.metric("ATR %", f"{atr_pct:.2f}%", f"Avg {avg_daily:.1f}%/day" if avg_daily else None)

    with col3:
        dollar_vol = watchlist.get('avg_dollar_volume', 0) or 0
        st.metric("Avg $ Volume", f"${dollar_vol / 1_000_000:.1f}M")

    with col4:
        rsi = metrics.get('rsi_14', 50) or 50
        rsi_delta = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else None
        st.metric("RSI (14)", f"{rsi:.0f}", rsi_delta)

    with col5:
        # market_cap is stored as formatted string (e.g., "150.5B", "2.3T")
        market_cap = fundamentals.get('market_cap')
        if market_cap:
            cap_str = f"${market_cap}"
        else:
            cap_str = "N/A"
        st.metric("Market Cap", cap_str)

    st.markdown("---")

    # ==========================================================================
    # SECTION 2: Claude AI Analysis
    # ==========================================================================
    st.markdown('<div class="section-header">🤖 AI Analysis</div>', unsafe_allow_html=True)

    # Check multiple sources for Claude analysis (in priority order):
    # 1. Session state (fresh analysis from this session)
    # 2. Scanner signals table (analysis attached to a signal)
    # 3. Watchlist table (analysis from Top Picks)
    existing_analysis = None
    analysis_source = None

    # Source 1: Check session state for fresh analysis
    session_key = f"claude_analysis_{ticker}"
    if session_key in st.session_state.deep_dive_claude_analysis:
        existing_analysis = st.session_state.deep_dive_claude_analysis[session_key]
        analysis_source = 'session'

    # Source 2: Check scanner signals table
    if not existing_analysis and active_signal and active_signal.get('claude_grade'):
        existing_analysis = {
            'rating': active_signal.get('claude_grade'),
            'confidence_score': active_signal.get('claude_score'),
            'recommendation': active_signal.get('claude_action'),
            'conviction': active_signal.get('claude_conviction'),
            'thesis': active_signal.get('claude_thesis'),
            'key_factors': active_signal.get('claude_key_strengths'),
            'risk_assessment': active_signal.get('claude_key_risks'),
            'position_sizing': active_signal.get('claude_position_rec'),
            'time_horizon': active_signal.get('claude_time_horizon'),
            'stop_adjustment': active_signal.get('claude_stop_adjustment'),
            'analyzed_at': active_signal.get('claude_analyzed_at'),
            'signal_id': active_signal.get('id')
        }
        analysis_source = 'signal'

    # Source 3: Check watchlist table (Top Picks analysis)
    if not existing_analysis:
        watchlist_analysis = service.get_latest_claude_analysis_from_watchlist(ticker)
        if watchlist_analysis and watchlist_analysis.get('rating'):
            existing_analysis = watchlist_analysis
            analysis_source = 'top_picks'

    # Handle both fresh analysis (grade/score/action) and DB format (rating/confidence_score/recommendation)
    if existing_analysis and (existing_analysis.get('rating') or existing_analysis.get('grade')):
        # Display existing analysis - handle both API response and DB format
        rating = existing_analysis.get('rating') or existing_analysis.get('grade', 'N/A')
        conf_score = existing_analysis.get('confidence_score') or existing_analysis.get('score', 0) or 0
        recommendation = existing_analysis.get('recommendation') or existing_analysis.get('action', 'N/A')
        conviction = existing_analysis.get('conviction', 'MEDIUM')
        thesis = existing_analysis.get('thesis', '')
        key_factors = existing_analysis.get('key_factors') or existing_analysis.get('key_strengths', []) or []
        risk_assessment = existing_analysis.get('risk_assessment') or existing_analysis.get('key_risks', 'N/A')
        position_sizing = existing_analysis.get('position_sizing') or existing_analysis.get('position_recommendation', 'Standard')
        time_horizon = existing_analysis.get('time_horizon', 'N/A')
        stop_adjustment = existing_analysis.get('stop_adjustment', 'Keep')
        analyzed_at = existing_analysis.get('analyzed_at')

        # Rating color
        rating_colors = {
            'A+': '#28a745', 'A': '#28a745', 'A-': '#5cb85c',
            'B+': '#17a2b8', 'B': '#17a2b8', 'B-': '#5bc0de',
            'C+': '#ffc107', 'C': '#ffc107', 'C-': '#f0ad4e',
            'D': '#dc3545', 'F': '#dc3545'
        }
        rating_color = rating_colors.get(rating, '#6c757d')

        # Recommendation styling
        rec_colors = {
            'STRONG BUY': '#28a745', 'BUY': '#5cb85c',
            'HOLD': '#ffc107', 'NEUTRAL': '#6c757d',
            'SELL': '#dc3545', 'STRONG SELL': '#c9302c'
        }
        rec_color = rec_colors.get(recommendation.upper() if recommendation else '', '#6c757d')

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown(f"""
            <div style="background: {rating_color}; color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2.5rem; font-weight: bold;">{rating}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">AI Rating</div>
                <div style="margin-top: 0.5rem; font-size: 1.2rem;">{conf_score:.1f}/10</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Conviction color
            conviction_colors = {'HIGH': '#28a745', 'MEDIUM': '#ffc107', 'LOW': '#dc3545'}
            conv_color = conviction_colors.get(conviction.upper() if conviction else 'MEDIUM', '#6c757d')

            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid {rec_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.2rem; font-weight: bold; color: {rec_color};">{recommendation}</span>
                    <span style="background: {conv_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">{conviction} Conviction</span>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                    <strong>Position:</strong> {position_sizing} |
                    <strong>Horizon:</strong> {time_horizon} |
                    <strong>Stop:</strong> {stop_adjustment}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Investment thesis
            if thesis:
                st.markdown(f"""
                <div style="margin-top: 0.5rem; padding: 0.5rem; background: #e7f3ff; border-radius: 4px; font-size: 0.9rem;">
                    💡 <strong>Thesis:</strong> {thesis}
                </div>
                """, unsafe_allow_html=True)

            # Key strengths as badges
            if key_factors:
                factors_list = key_factors if isinstance(key_factors, list) else [key_factors]
                factors_html = " ".join([
                    f'<span style="background: #d4edda; padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.3rem; font-size: 0.8rem;">✓ {f}</span>'
                    for f in factors_list[:5]
                ])
                st.markdown(f"""
                <div style="margin-top: 0.5rem;">
                    <strong style="font-size: 0.85rem;">Strengths:</strong> {factors_html}
                </div>
                """, unsafe_allow_html=True)

            # Risk assessment (handle both string and list)
            if risk_assessment:
                if isinstance(risk_assessment, list):
                    risks_html = " ".join([
                        f'<span style="background: #f8d7da; padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.3rem; font-size: 0.8rem;">⚠️ {r}</span>'
                        for r in risk_assessment[:3]
                    ])
                    st.markdown(f"""
                    <div style="margin-top: 0.5rem;">
                        <strong style="font-size: 0.85rem;">Risks:</strong> {risks_html}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="margin-top: 0.5rem; padding: 0.5rem; background: #fff3cd; border-radius: 4px; font-size: 0.85rem;">
                        ⚠️ <strong>Risk:</strong> {risk_assessment}
                    </div>
                    """, unsafe_allow_html=True)

        # Show when analyzed and source with staleness indicator
        if analyzed_at:
            if isinstance(analyzed_at, datetime):
                # Handle timezone-aware datetimes
                now = datetime.now(analyzed_at.tzinfo) if analyzed_at.tzinfo else datetime.now()
                time_ago = now - analyzed_at

                # Format the actual date/time
                date_str = analyzed_at.strftime("%b %d, %Y %H:%M")

                # Calculate relative time
                if time_ago.days > 0:
                    ago_str = f"{time_ago.days}d ago"
                elif time_ago.seconds >= 3600:
                    ago_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    ago_str = f"{time_ago.seconds // 60}m ago"

                # Check if stale (more than 7 days old)
                is_stale = time_ago.days > 7
            else:
                date_str = str(analyzed_at)[:16]
                ago_str = ""
                is_stale = False

            # Show source of analysis
            source_label = {
                'session': 'This session',
                'signal': 'Scanner Signal',
                'top_picks': 'Top Picks'
            }.get(analysis_source, '')
            source_text = f" | Source: {source_label}" if source_label else ""

            # Build the display string with staleness warning
            if is_stale:
                st.warning(f"⚠️ Analysis is {time_ago.days} days old - consider refreshing")
            st.caption(f"📅 {date_str} ({ago_str}){source_text}")

        # Re-analyze button
        if st.button("🔄 Re-analyze with Claude", key="reanalyze_claude"):
            _run_claude_analysis(service, ticker, active_signal, metrics, fundamentals, candles, session_key)

    else:
        # No analysis - show analyze button
        st.info("No AI analysis available for this stock yet.")

        # Allow analysis even without a scanner signal - use metrics data
        if st.button("🤖 Analyze with Claude (with Chart Vision)", key="analyze_claude", type="primary"):
            # Create a synthetic signal from metrics if no active signal
            signal_to_analyze = active_signal if active_signal else {
                'ticker': ticker,
                'signal_type': 'ANALYSIS',  # Not a real signal, just analysis
                'scanner_name': 'Deep Dive',
                'entry_price': metrics.get('current_price', 0),
                'composite_score': watchlist.get('score', 50) if watchlist else 50,
                'quality_tier': f"T{watchlist.get('tier', 3)}" if watchlist else 'T3',
            }
            _run_claude_analysis(service, ticker, signal_to_analyze, metrics, fundamentals, candles, session_key)

    st.markdown("---")

    # ==========================================================================
    # SECTION 2.5: News Sentiment
    # ==========================================================================
    st.markdown('<div class="section-header">📰 News Sentiment</div>', unsafe_allow_html=True)

    # Get news data for this ticker from multiple sources:
    # 1. Session state (freshly fetched this session)
    # 2. Active signal in database
    # 3. Other signals for this ticker
    news_data = None
    news_signal_id = None

    # Source 1: Check session state for freshly fetched news
    session_news_key = f"news_sentiment_{ticker}"
    if session_news_key in st.session_state:
        news_data = st.session_state[session_news_key]

    # Source 2: Check if active signal has news data
    if not news_data and active_signal and active_signal.get('news_sentiment_score') is not None:
        news_data = {
            'score': active_signal.get('news_sentiment_score'),
            'level': active_signal.get('news_sentiment_level'),
            'count': active_signal.get('news_headlines_count', 0),
            'factors': active_signal.get('news_factors', []),
            'analyzed_at': active_signal.get('news_analyzed_at')
        }
        news_signal_id = active_signal.get('id')

    # Source 3: If no news from active signal, check other signals for this ticker
    if not news_data and scanner_signals:
        for sig in scanner_signals:
            if sig.get('news_sentiment_score') is not None:
                news_data = {
                    'score': sig.get('news_sentiment_score'),
                    'level': sig.get('news_sentiment_level'),
                    'count': sig.get('news_headlines_count', 0),
                    'factors': sig.get('news_factors', []),
                    'analyzed_at': sig.get('news_analyzed_at')
                }
                news_signal_id = sig.get('id')
                break

    # If we have signals but no news_signal_id yet, use first signal for enrichment
    if not news_signal_id and scanner_signals:
        news_signal_id = scanner_signals[0].get('id')

    col1, col2 = st.columns([3, 1])

    with col2:
        # Fetch/Refresh News button
        button_label = "🔄 Refresh News" if news_data else "📰 Fetch News"
        if st.button(button_label, key=f"fetch_news_{ticker}"):
            with st.spinner(f"Fetching news for {ticker}..."):
                # If we have a signal, enrich that signal
                if news_signal_id:
                    result = service.enrich_signal_with_news(news_signal_id)
                else:
                    # No signal - fetch news directly for display (won't persist without signal)
                    result = _fetch_news_for_ticker(ticker)

                if result.get('success'):
                    st.success(f"✅ {result.get('message', 'News fetched!')}")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Failed to fetch news')}")

    with col1:
        if news_data:
            # Format analyzed timestamp
            news_time_str = None
            news_ago_str = None
            analyzed_at = news_data.get('analyzed_at')
            if analyzed_at:
                if isinstance(analyzed_at, datetime):
                    news_time_str = analyzed_at.strftime('%b %d, %Y %H:%M')
                    now = datetime.now(analyzed_at.tzinfo) if analyzed_at.tzinfo else datetime.now()
                    time_ago = now - analyzed_at
                    if time_ago.days > 0:
                        news_ago_str = f"{time_ago.days}d ago"
                    elif time_ago.seconds >= 3600:
                        news_ago_str = f"{time_ago.seconds // 3600}h ago"
                    else:
                        news_ago_str = f"{time_ago.seconds // 60}m ago"
                else:
                    news_time_str = str(analyzed_at)[:16]

            # News sentiment colors
            sentiment_colors = {
                'very_bullish': '#28a745',
                'bullish': '#5cb85c',
                'neutral': '#6c757d',
                'bearish': '#f0ad4e',
                'very_bearish': '#dc3545'
            }
            sentiment_icons = {
                'very_bullish': '🟢🟢',
                'bullish': '🟢',
                'neutral': '⚪',
                'bearish': '🟠',
                'very_bearish': '🔴🔴'
            }

            level = (news_data.get('level') or 'neutral').lower()
            score = news_data.get('score', 0) or 0
            count = news_data.get('count', 0) or 0
            factors = news_data.get('factors', [])

            sent_color = sentiment_colors.get(level, '#6c757d')
            sent_icon = sentiment_icons.get(level, '⚪')
            level_display = level.replace('_', ' ').title()

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid {sent_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div>
                        <span style="font-size: 1.3rem;">{sent_icon}</span>
                        <span style="font-size: 1.2rem; font-weight: bold; color: {sent_color}; margin-left: 0.5rem;">{level_display}</span>
                        <span style="margin-left: 1rem; font-size: 0.9rem; color: #666;">Score: {score:.2f}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">
                        {count} articles analyzed
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show factors
            if factors:
                factors_list = factors if isinstance(factors, list) else [factors]
                if factors_list:
                    with st.expander("📋 Key News Factors", expanded=False):
                        for factor in factors_list[:5]:
                            if level in ['very_bullish', 'bullish']:
                                st.markdown(f"- :green[+] {factor}")
                            elif level in ['very_bearish', 'bearish']:
                                st.markdown(f"- :red[-] {factor}")
                            else:
                                st.markdown(f"- {factor}")

            # Show timestamp
            if news_time_str:
                ago_text = f" ({news_ago_str})" if news_ago_str else ""
                st.caption(f"📅 {news_time_str}{ago_text}")
        else:
            st.info("📰 No news sentiment data available. Click 'Fetch News' to analyze recent news for this stock.")

    st.markdown("---")

    # ==========================================================================
    # SECTION 3: Active Signal Card
    # ==========================================================================
    if active_signal:
        st.markdown('<div class="section-header">📈 Active Signal</div>', unsafe_allow_html=True)

        sig_type = active_signal.get('signal_type', 'BUY')
        scanner_name = active_signal.get('scanner_name', 'Unknown')
        entry_price = active_signal.get('entry_price', 0)
        stop_loss = active_signal.get('stop_loss', 0)
        take_profit_1 = active_signal.get('take_profit_1', 0)
        take_profit_2 = active_signal.get('take_profit_2', 0)
        quality_tier = active_signal.get('quality_tier', '-')
        composite_score = active_signal.get('composite_score', 0)
        signal_timestamp = active_signal.get('signal_timestamp')

        # Calculate P&L and R:R
        current_price = metrics.get('current_price', entry_price)
        if sig_type == 'BUY':
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            risk_pct = ((entry_price - stop_loss) / entry_price * 100) if entry_price > 0 and stop_loss > 0 else 0
            reward_pct = ((take_profit_1 - entry_price) / entry_price * 100) if entry_price > 0 and take_profit_1 > 0 else 0
        else:
            pnl_pct = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
            risk_pct = ((stop_loss - entry_price) / entry_price * 100) if entry_price > 0 and stop_loss > 0 else 0
            reward_pct = ((entry_price - take_profit_1) / entry_price * 100) if entry_price > 0 and take_profit_1 > 0 else 0

        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0

        # Days since signal
        if signal_timestamp:
            if isinstance(signal_timestamp, datetime):
                # Handle timezone-aware datetimes
                now = datetime.now(signal_timestamp.tzinfo) if signal_timestamp.tzinfo else datetime.now()
                days_active = (now - signal_timestamp).days
            else:
                days_active = 0
        else:
            days_active = 0

        sig_color = "#28a745" if sig_type == "BUY" else "#dc3545"
        pnl_color = "#28a745" if pnl_pct >= 0 else "#dc3545"
        tier_color = "#28a745" if quality_tier in ['A+', 'A'] else "#17a2b8" if quality_tier in ['A-', 'B+', 'B'] else "#ffc107"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid {sig_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <div>
                    <span style="font-size: 1.3rem; font-weight: bold; color: {sig_color};">{sig_type}</span>
                    <span style="background: #e9ecef; padding: 0.2rem 0.5rem; border-radius: 4px; margin-left: 0.5rem; font-size: 0.85rem;">{scanner_name}</span>
                </div>
                <div>
                    <span style="background: {tier_color}; color: white; padding: 0.3rem 0.6rem; border-radius: 4px; font-weight: bold;">{quality_tier}</span>
                    <span style="margin-left: 0.5rem; font-size: 0.9rem;">Score: {composite_score:.1f}</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; text-align: center;">
                <div>
                    <div style="font-size: 0.8rem; color: #666;">Entry</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">${entry_price:.2f}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #666;">Stop Loss</div>
                    <div style="font-size: 1.1rem; font-weight: bold; color: #dc3545;">${stop_loss:.2f}</div>
                    <div style="font-size: 0.75rem; color: #666;">-{risk_pct:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #666;">Target 1</div>
                    <div style="font-size: 1.1rem; font-weight: bold; color: #28a745;">${take_profit_1:.2f}</div>
                    <div style="font-size: 0.75rem; color: #666;">+{reward_pct:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #666;">R:R Ratio</div>
                    <div style="font-size: 1.1rem; font-weight: bold;">{rr_ratio:.1f}:1</div>
                </div>
            </div>
            <div style="margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid #dee2e6; display: flex; justify-content: space-between;">
                <div>
                    <span style="font-size: 0.85rem;">Current: </span>
                    <span style="font-weight: bold;">${current_price:.2f}</span>
                    <span style="color: {pnl_color}; font-weight: bold; margin-left: 0.3rem;">({pnl_pct:+.1f}%)</span>
                </div>
                <div style="font-size: 0.85rem; color: #666;">
                    {days_active} days active
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

    # ==========================================================================
    # SECTION 4: Technical + SMC Analysis (side by side)
    # ==========================================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Technical Analysis</div>', unsafe_allow_html=True)

        # RSI with visual bar
        rsi = metrics.get('rsi_14', 50) or 50
        rsi_color = "#dc3545" if rsi > 70 else "#28a745" if rsi < 30 else "#17a2b8"
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between;">
                <span><strong>RSI(14)</strong></span>
                <span style="color: {rsi_color}; font-weight: bold;">{rsi:.0f} - {rsi_status}</span>
            </div>
            <div class="score-bar" style="margin-top: 0.3rem;">
                <div class="score-fill" style="width: {rsi}%; background: {rsi_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # MACD
        macd = metrics.get('macd', 0) or 0
        macd_signal = metrics.get('macd_signal', 0) or 0
        macd_hist = metrics.get('macd_histogram', 0) or 0
        macd_status = "Bullish ✓" if macd > macd_signal else "Bearish ✗"
        macd_color = "#28a745" if macd > macd_signal else "#dc3545"

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between;">
                <span><strong>MACD</strong></span>
                <span style="color: {macd_color}; font-weight: bold;">{macd_status}</span>
            </div>
            <div style="font-size: 0.85rem; color: #666; margin-top: 0.2rem;">
                Line: {macd:.3f} | Signal: {macd_signal:.3f} | Hist: {macd_hist:+.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Moving averages
        sma_20 = metrics.get('sma_20', 0) or 0
        sma_50 = metrics.get('sma_50', 0) or 0
        sma_200 = metrics.get('sma_200', 0) or 0
        current_price = metrics.get('current_price', 0) or 0

        # Calculate price vs MAs
        def pct_from_ma(price, ma):
            return ((price - ma) / ma * 100) if ma > 0 else 0

        pct_20 = pct_from_ma(current_price, sma_20)
        pct_50 = pct_from_ma(current_price, sma_50)
        pct_200 = pct_from_ma(current_price, sma_200)

        # Trend determination
        if sma_20 > sma_50 > sma_200 and current_price > sma_20:
            trend = "Strong Uptrend ↗"
            trend_color = "#28a745"
        elif sma_20 < sma_50 < sma_200 and current_price < sma_20:
            trend = "Strong Downtrend ↘"
            trend_color = "#dc3545"
        elif current_price > sma_50:
            trend = "Uptrend ↗"
            trend_color = "#5cb85c"
        elif current_price < sma_50:
            trend = "Downtrend ↘"
            trend_color = "#f0ad4e"
        else:
            trend = "Sideways →"
            trend_color = "#6c757d"

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between;">
                <span><strong>Trend</strong></span>
                <span style="color: {trend_color}; font-weight: bold;">{trend}</span>
            </div>
            <div style="font-size: 0.85rem; margin-top: 0.3rem;">
                <div>SMA20: ${sma_20:.2f} <span style="color: {'#28a745' if pct_20 > 0 else '#dc3545'}">({pct_20:+.1f}%)</span></div>
                <div>SMA50: ${sma_50:.2f} <span style="color: {'#28a745' if pct_50 > 0 else '#dc3545'}">({pct_50:+.1f}%)</span></div>
                <div>SMA200: ${sma_200:.2f} <span style="color: {'#28a745' if pct_200 > 0 else '#dc3545'}">({pct_200:+.1f}%)</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">🎯 SMC Analysis</div>', unsafe_allow_html=True)

        # SMC data priority: metrics table > derived from price action
        # The metrics table has smc_trend, smc_bias, smc_confluence_score from daily calculations
        smc_trend = metrics.get('smc_trend')
        smc_bias = metrics.get('smc_bias')
        smc_confluence = metrics.get('smc_confluence_score', 0) or 0

        # If no SMC data in metrics, derive from price action
        if not smc_trend:
            if sma_20 > sma_50 > sma_200 and current_price > sma_20:
                smc_trend = 'Bullish'
                smc_bias = 'Bullish'
            elif sma_20 < sma_50 < sma_200 and current_price < sma_20:
                smc_trend = 'Bearish'
                smc_bias = 'Bearish'
            elif current_price > sma_50:
                smc_trend = 'Bullish'
                smc_bias = 'Bullish'
            elif current_price < sma_50:
                smc_trend = 'Bearish'
                smc_bias = 'Bearish'
            else:
                smc_trend = 'Neutral'
                smc_bias = 'Neutral'

        # Zone from price position in 52-week range
        high_52w = fundamentals.get('fifty_two_week_high', 0) or fundamentals.get('week_52_high', 0) or 0
        low_52w = fundamentals.get('fifty_two_week_low', 0) or fundamentals.get('week_52_low', 0) or 0
        if high_52w and low_52w and high_52w > low_52w:
            range_pct = (current_price - low_52w) / (high_52w - low_52w) * 100
            if range_pct < 33:
                smc_zone = 'Discount'
            elif range_pct > 67:
                smc_zone = 'Premium'
            else:
                smc_zone = 'Equilibrium'
        else:
            smc_zone = 'N/A'

        # If no confluence score from metrics, calculate from multiple factors
        if smc_confluence == 0:
            conf_factors = 0
            if smc_trend == 'Bullish':
                if rsi < 70: conf_factors += 2  # Not overbought
                if macd > macd_signal: conf_factors += 2  # MACD bullish
                if current_price > sma_20: conf_factors += 1.5
                if current_price > sma_50: conf_factors += 1.5
                if current_price > sma_200: conf_factors += 1.5
            elif smc_trend == 'Bearish':
                if rsi > 30: conf_factors += 2  # Not oversold
                if macd < macd_signal: conf_factors += 2  # MACD bearish
                if current_price < sma_20: conf_factors += 1.5
                if current_price < sma_50: conf_factors += 1.5
                if current_price < sma_200: conf_factors += 1.5
            smc_confluence = min(conf_factors, 10)

        bos_direction = smc_trend

        # Trend styling
        trend_colors = {'Bullish': '#28a745', 'Bearish': '#dc3545', 'Neutral': '#6c757d'}
        trend_color = trend_colors.get(smc_trend, '#6c757d')

        # Zone styling
        zone_pct = 50  # Default
        if smc_zone == 'Discount':
            zone_pct = 25
            zone_color = "#28a745"
        elif smc_zone == 'Premium':
            zone_pct = 75
            zone_color = "#dc3545"
        else:
            zone_color = "#ffc107"

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between;">
                <span><strong>HTF Trend</strong></span>
                <span style="color: {trend_color}; font-weight: bold;">{smc_trend}</span>
            </div>
            <div style="font-size: 0.85rem; color: #666; margin-top: 0.2rem;">
                Bias: {smc_bias} | Direction: {bos_direction}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between;">
                <span><strong>Price Zone</strong></span>
                <span style="color: {zone_color}; font-weight: bold;">{smc_zone}</span>
            </div>
            <div style="position: relative; height: 20px; background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%); border-radius: 4px; margin-top: 0.3rem;">
                <div style="position: absolute; left: {zone_pct}%; top: -2px; width: 4px; height: 24px; background: #000; border-radius: 2px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #666; margin-top: 0.2rem;">
                <span>Discount (Buy)</span>
                <span>Equilibrium</span>
                <span>Premium (Sell)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confluence score
        conf_color = "#28a745" if smc_confluence >= 7 else "#17a2b8" if smc_confluence >= 5 else "#ffc107"
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between;">
                <span><strong>Confluence Score</strong></span>
                <span style="color: {conf_color}; font-weight: bold;">{smc_confluence:.1f}/10</span>
            </div>
            <div class="score-bar" style="margin-top: 0.3rem;">
                <div class="score-fill" style="width: {smc_confluence * 10}%; background: {conf_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not active_signal:
            st.caption("📊 Derived from technical indicators (no active signal)")

    st.markdown("---")

    # ==========================================================================
    # SECTION 5: Fundamentals (Tiered - Key Metrics + Expandable Full)
    # ==========================================================================
    st.markdown('<div class="section-header">📊 Fundamentals</div>', unsafe_allow_html=True)

    if fundamentals:
        # Key metrics row (always visible)
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            pe = fundamentals.get('pe_trailing', fundamentals.get('pe_ratio'))
            st.metric("P/E", f"{pe:.1f}" if pe else "N/A")

        with col2:
            pb = fundamentals.get('pb_ratio', fundamentals.get('price_to_book'))
            st.metric("P/B", f"{pb:.2f}" if pb else "N/A")

        with col3:
            peg = fundamentals.get('peg_ratio')
            st.metric("PEG", f"{peg:.2f}" if peg else "N/A")

        with col4:
            beta = fundamentals.get('beta')
            st.metric("Beta", f"{beta:.2f}" if beta else "N/A")

        with col5:
            profit_margin = fundamentals.get('profit_margin', fundamentals.get('net_margin'))
            st.metric("Margin", f"{profit_margin*100:.1f}%" if profit_margin else "N/A")

        with col6:
            roe = fundamentals.get('roe', fundamentals.get('return_on_equity'))
            st.metric("ROE", f"{roe*100:.1f}%" if roe else "N/A")

        # Second row of key metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            rev_growth = fundamentals.get('revenue_growth', fundamentals.get('rev_growth_yoy'))
            st.metric("Rev Growth", f"{rev_growth*100:+.1f}%" if rev_growth else "N/A")

        with col2:
            short_pct = fundamentals.get('short_percent', fundamentals.get('short_percent_of_float'))
            st.metric("Short %", f"{short_pct*100:.1f}%" if short_pct else "N/A")

        with col3:
            inst_pct = fundamentals.get('institutional_percent', fundamentals.get('held_percent_institutions'))
            st.metric("Inst %", f"{inst_pct*100:.1f}%" if inst_pct else "N/A")

        with col4:
            insider_pct = fundamentals.get('insider_percent', fundamentals.get('held_percent_insiders'))
            st.metric("Insider %", f"{insider_pct*100:.1f}%" if insider_pct else "N/A")

        with col5:
            div_yield = fundamentals.get('dividend_yield', fundamentals.get('forward_dividend_yield'))
            st.metric("Div Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")

        with col6:
            earnings_date = fundamentals.get('earnings_date', fundamentals.get('next_earnings'))
            if earnings_date:
                if isinstance(earnings_date, datetime):
                    days_to = (earnings_date - datetime.now()).days
                    st.metric("Earnings", earnings_date.strftime('%b %d'), f"{days_to}d" if days_to > 0 else "Past")
                else:
                    st.metric("Earnings", str(earnings_date)[:10])
            else:
                st.metric("Earnings", "N/A")

        # Expandable full fundamentals
        with st.expander("📋 View All Fundamentals"):
            # Organize into categories
            valuation_cols = ['pe_trailing', 'pe_forward', 'pb_ratio', 'ps_ratio', 'peg_ratio',
                              'ev_to_ebitda', 'ev_to_revenue', 'price_to_sales', 'enterprise_value']
            growth_cols = ['revenue_growth', 'earnings_growth', 'quarterly_revenue_growth',
                           'quarterly_earnings_growth', 'eps_growth_yoy']
            profitability_cols = ['profit_margin', 'operating_margin', 'gross_margin', 'roe', 'roa', 'roic']
            health_cols = ['debt_to_equity', 'current_ratio', 'quick_ratio', 'total_debt', 'total_cash']
            dividend_cols = ['dividend_yield', 'dividend_rate', 'payout_ratio', 'ex_dividend_date']

            def render_fundamental_section(title, cols, data):
                items = []
                for col in cols:
                    val = data.get(col)
                    if val is not None:
                        # Format value
                        if isinstance(val, float):
                            if 'percent' in col or 'margin' in col or 'yield' in col or 'growth' in col or col in ['roe', 'roa', 'roic']:
                                formatted = f"{val*100:.2f}%"
                            elif val > 1_000_000_000:
                                formatted = f"${val/1_000_000_000:.2f}B"
                            elif val > 1_000_000:
                                formatted = f"${val/1_000_000:.2f}M"
                            else:
                                formatted = f"{val:.2f}"
                        else:
                            formatted = str(val)
                        # Clean column name
                        clean_name = col.replace('_', ' ').title()
                        items.append(f"**{clean_name}:** {formatted}")
                if items:
                    st.markdown(f"**{title}**")
                    st.markdown(" | ".join(items[:6]))
                    if len(items) > 6:
                        st.markdown(" | ".join(items[6:]))

            render_fundamental_section("💰 Valuation", valuation_cols, fundamentals)
            render_fundamental_section("📈 Growth", growth_cols, fundamentals)
            render_fundamental_section("💵 Profitability", profitability_cols, fundamentals)
            render_fundamental_section("🏦 Financial Health", health_cols, fundamentals)
            render_fundamental_section("💸 Dividends", dividend_cols, fundamentals)

    else:
        st.info("No fundamental data available")

    st.markdown("---")

    # ==========================================================================
    # SECTION 6: Score Breakdown (keep existing)
    # ==========================================================================
    if watchlist:
        st.markdown('<div class="section-header">🎯 Score Breakdown</div>', unsafe_allow_html=True)
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

    # ==========================================================================
    # SECTION 7: Enhanced Price Chart with MAs, RSI subplot, Signal Levels
    # ==========================================================================
    st.markdown('<div class="section-header">📈 Price Chart (90 Days)</div>', unsafe_allow_html=True)

    if not candles.empty:
        # Calculate indicators
        candles = candles.copy()
        candles['sma20'] = candles['close'].rolling(20).mean()
        candles['sma50'] = candles['close'].rolling(50).mean()
        candles['sma200'] = candles['close'].rolling(200).mean()

        # RSI calculation
        delta = candles['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        candles['rsi'] = 100 - (100 / (1 + rs))

        # Volume average
        candles['vol_avg'] = candles['volume'].rolling(20).mean()

        # Create figure with 3 subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('', 'RSI(14)', 'Volume')
        )

        # Row 1: Candlesticks
        fig.add_trace(go.Candlestick(
            x=candles['timestamp'],
            open=candles['open'],
            high=candles['high'],
            low=candles['low'],
            close=candles['close'],
            name='Price',
            increasing_line_color='#28a745',
            decreasing_line_color='#dc3545'
        ), row=1, col=1)

        # Moving averages
        fig.add_trace(go.Scatter(
            x=candles['timestamp'], y=candles['sma20'],
            name='SMA20', line=dict(color='#2196F3', width=1),
            hovertemplate='SMA20: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=candles['timestamp'], y=candles['sma50'],
            name='SMA50', line=dict(color='#FF9800', width=1),
            hovertemplate='SMA50: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=candles['timestamp'], y=candles['sma200'],
            name='SMA200', line=dict(color='#9C27B0', width=1.5),
            hovertemplate='SMA200: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # Signal levels (if active signal)
        if active_signal:
            entry_price = active_signal.get('entry_price', 0)
            stop_loss = active_signal.get('stop_loss', 0)
            take_profit_1 = active_signal.get('take_profit_1', 0)

            if entry_price > 0:
                fig.add_hline(y=entry_price, line_dash="dash", line_color="#28a745",
                              annotation_text=f"Entry ${entry_price:.2f}", row=1, col=1)
            if stop_loss > 0:
                fig.add_hline(y=stop_loss, line_dash="dash", line_color="#dc3545",
                              annotation_text=f"Stop ${stop_loss:.2f}", row=1, col=1)
            if take_profit_1 > 0:
                fig.add_hline(y=take_profit_1, line_dash="dash", line_color="#2196F3",
                              annotation_text=f"Target ${take_profit_1:.2f}", row=1, col=1)

        # Signal markers
        for sig in scanner_signals[:5]:
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
                    marker=dict(size=14, color=marker_color, symbol=marker_symbol, line=dict(width=1, color='white')),
                    name=f'{sig_type} Signal',
                    hovertemplate=f"{sig_type}<br>${entry:.2f}<extra></extra>",
                    showlegend=False
                ), row=1, col=1)

        # Row 2: RSI
        fig.add_trace(go.Scatter(
            x=candles['timestamp'], y=candles['rsi'],
            name='RSI', line=dict(color='#9C27B0', width=1.5),
            fill='tozeroy', fillcolor='rgba(156, 39, 176, 0.1)'
        ), row=2, col=1)

        fig.add_hline(y=70, line_dash="dot", line_color="#dc3545", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#28a745", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#6c757d", row=2, col=1)

        # Row 3: Volume
        colors = ['#28a745' if c >= o else '#dc3545' for c, o in zip(candles['close'], candles['open'])]
        fig.add_trace(go.Bar(
            x=candles['timestamp'], y=candles['volume'],
            marker_color=colors, name='Volume', opacity=0.7
        ), row=3, col=1)

        # Volume average line
        fig.add_trace(go.Scatter(
            x=candles['timestamp'], y=candles['vol_avg'],
            name='Vol Avg', line=dict(color='#FF9800', width=1, dash='dash')
        ), row=3, col=1)

        # Layout
        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price data available")

    st.markdown("---")

    # ==========================================================================
    # SECTION 8: Scanner Signal History (replacing ZLMA signals)
    # ==========================================================================
    st.markdown('<div class="section-header">📋 Scanner Signal History</div>', unsafe_allow_html=True)

    if scanner_signals:
        # Create a more detailed signal history table
        signal_data = []
        for sig in scanner_signals:
            sig_type = sig.get('signal_type', '')
            scanner = sig.get('scanner_name', 'Unknown')
            entry = sig.get('entry_price', 0)
            quality = sig.get('quality_tier', '-')
            score = sig.get('composite_score', 0)
            claude_rating = sig.get('claude_grade', '-')
            timestamp = sig.get('signal_timestamp')

            time_str = timestamp.strftime('%Y-%m-%d') if isinstance(timestamp, datetime) else str(timestamp)[:10]

            signal_data.append({
                'Date': time_str,
                'Type': sig_type,
                'Scanner': scanner,
                'Entry': f"${entry:.2f}",
                'Quality': quality,
                'Score': f"{score:.1f}",
                'Claude': claude_rating if claude_rating else '-'
            })

        if signal_data:
            df = pd.DataFrame(signal_data)

            # Style the dataframe
            def style_type(val):
                color = '#28a745' if val == 'BUY' else '#dc3545' if val == 'SELL' else '#6c757d'
                return f'color: {color}; font-weight: bold;'

            def style_quality(val):
                colors = {'A+': '#28a745', 'A': '#28a745', 'A-': '#5cb85c',
                          'B+': '#17a2b8', 'B': '#17a2b8', 'B-': '#5bc0de',
                          'C+': '#ffc107', 'C': '#ffc107', 'C-': '#f0ad4e'}
                color = colors.get(val, '#6c757d')
                return f'color: {color}; font-weight: bold;'

            styled_df = df.style.map(style_type, subset=['Type']).map(style_quality, subset=['Quality', 'Claude'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No scanner signals for this stock")


def _run_claude_analysis(service, ticker, signal, metrics, fundamentals, candles, session_key):
    """Helper function to run Claude analysis with chart vision."""
    with st.spinner("🤖 Analyzing with Claude (generating chart + vision analysis)..."):
        try:
            analysis = service.analyze_stock_with_claude(
                ticker=ticker,
                signal=signal,
                metrics=metrics,
                fundamentals=fundamentals,
                candles=candles
            )

            if analysis and analysis.get('grade'):
                st.session_state.deep_dive_claude_analysis[session_key] = analysis
                st.success("✅ Analysis complete!")
                st.rerun()
            elif analysis and analysis.get('error'):
                st.error(f"Analysis failed: {analysis.get('error')}")
            else:
                st.error("Analysis failed - no grade returned")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")


def _fetch_news_for_ticker(ticker: str) -> Dict[str, Any]:
    """
    Fetch and analyze news for a ticker without requiring a signal.
    Results are stored in session state for display but not persisted to database.
    """
    import os
    import requests
    from datetime import datetime, timedelta

    try:
        # Get Finnhub API key
        finnhub_api_key = os.getenv('FINNHUB_API_KEY', '')
        if not finnhub_api_key:
            return {'success': False, 'error': 'FINNHUB_API_KEY not configured'}

        # Fetch news from Finnhub
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': ticker.upper(),
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': finnhub_api_key
        }

        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            return {'success': False, 'error': f'Finnhub API error: {response.status_code}'}

        articles = response.json()

        if not articles:
            return {
                'success': True,
                'message': f'No news found for {ticker}',
                'ticker': ticker,
                'articles_count': 0
            }

        # Analyze sentiment using VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()

            scores = []
            for article in articles[:50]:
                headline = article.get('headline', '')
                summary = article.get('summary', '')
                text = f"{headline} {summary}"

                if text.strip():
                    sentiment = analyzer.polarity_scores(text)
                    scores.append(sentiment['compound'])

            if scores:
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = 0.0

            # Classify sentiment level
            if avg_score >= 0.5:
                level = 'very_bullish'
            elif avg_score >= 0.15:
                level = 'bullish'
            elif avg_score <= -0.5:
                level = 'very_bearish'
            elif avg_score <= -0.15:
                level = 'bearish'
            else:
                level = 'neutral'

            # Build factors
            factors = []
            level_labels = {
                'very_bullish': 'Strong positive',
                'bullish': 'Positive',
                'neutral': 'Neutral',
                'bearish': 'Negative',
                'very_bearish': 'Strong negative'
            }
            factors.append(f"{level_labels[level]} news sentiment ({avg_score:.2f})")
            factors.append(f"{len(scores)} news articles analyzed")

            if articles:
                top_headline = articles[0].get('headline', '')[:80]
                if top_headline:
                    factors.append(f'Key: "{top_headline}..."')

            # Store in session state for display
            session_key = f"news_sentiment_{ticker}"
            st.session_state[session_key] = {
                'score': avg_score,
                'level': level,
                'count': len(scores),
                'factors': factors,
                'analyzed_at': datetime.now()
            }

            return {
                'success': True,
                'message': f'News analysis completed for {ticker}',
                'ticker': ticker,
                'sentiment_score': avg_score,
                'sentiment_level': level,
                'articles_count': len(scores)
            }

        except ImportError:
            return {'success': False, 'error': 'VADER sentiment analyzer not installed'}

    except requests.Timeout:
        return {'success': False, 'error': 'Finnhub API request timed out'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# =============================================================================
# SCANNER SIGNALS TAB
# =============================================================================

def render_scanner_signals_tab(service):
    """Render the All Signals tab - unified view of all scanner signals."""
    st.markdown("""
    <div class="main-header">
        <h2>📡 All Signals</h2>
        <p>Unified view of all trading signals from all scanners - with Claude AI Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Get scanner stats
    stats = service.get_scanner_stats()

    if not stats:
        st.info("No scanner signals available yet. Run the scanner to generate signals.")
        return

    # Top metrics - Row 1: Scanner stats
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Active Signals", stats.get('active_signals', 0))
    col2.metric("High Quality (A/A+)", stats.get('high_quality', 0))
    col3.metric("Today's Signals", stats.get('today_signals', 0))
    col4.metric("Total Signals", stats.get('total_signals', 0))

    # Claude AI stats - Row 2
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Claude Analyzed", stats.get('claude_analyzed', 0), help="Signals analyzed by Claude AI")
    col2.metric("Claude A/A+", stats.get('claude_high_grade', 0), help="Claude high-grade signals")
    col3.metric("Strong Buys", stats.get('claude_strong_buys', 0), help="Claude STRONG BUY recommendations")
    col4.metric("Awaiting Analysis", stats.get('awaiting_analysis', 0), help="Active signals not yet analyzed")

    st.markdown("---")

    # Filters - Row 1
    col1, col2, col3, col4 = st.columns(4)

    # Build scanner list dynamically from database
    by_scanner = stats.get('by_scanner', [])
    scanner_names = ["All Scanners"] + [s['scanner_name'] for s in by_scanner]

    with col1:
        scanner_filter = st.selectbox(
            "Scanner",
            scanner_names,
            format_func=lambda x: x.replace('_', ' ').title() if x != "All Scanners" else x
        )

    with col2:
        tier_filter = st.selectbox(
            "Quality Tier",
            ["All Tiers", "A+", "A", "B", "C"]
        )

    with col3:
        status_filter = st.selectbox(
            "Status",
            ["active", "triggered", "closed", "expired", "All"]
        )

    with col4:
        claude_filter = st.selectbox(
            "Claude Analysis",
            ["All Signals", "Claude Analyzed Only", "A+ Grade", "A Grade", "B Grade", "STRONG BUY", "BUY"]
        )

    # Filters - Row 2: Date range filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Default to today
        date_from = st.date_input(
            "Signal Date From",
            value=datetime.now().date(),
            help="Filter signals detected from this date"
        )

    with col2:
        date_to = st.date_input(
            "Signal Date To",
            value=datetime.now().date(),
            help="Filter signals detected up to this date"
        )

    with col3:
        st.empty()  # Placeholder for alignment

    with col4:
        st.empty()  # Placeholder for alignment

    # Get filtered signals
    scanner_name = None if scanner_filter == "All Scanners" else scanner_filter
    status = None if status_filter == "All" else status_filter
    min_score = {'A+': 85, 'A': 70, 'B': 60, 'C': 50}.get(tier_filter)

    # Claude filter mapping
    claude_analyzed_only = claude_filter in ["Claude Analyzed Only", "A+ Grade", "A Grade", "B Grade", "STRONG BUY", "BUY"]
    min_claude_grade = None
    if claude_filter in ["A+ Grade"]:
        min_claude_grade = "A+"
    elif claude_filter in ["A Grade"]:
        min_claude_grade = "A"
    elif claude_filter in ["B Grade"]:
        min_claude_grade = "B"

    signals = service.get_scanner_signals(
        scanner_name=scanner_name,
        status=status,
        min_score=min_score,
        min_claude_grade=min_claude_grade,
        claude_analyzed_only=claude_analyzed_only,
        signal_date_from=str(date_from) if date_from else None,
        signal_date_to=str(date_to) if date_to else None,
        limit=100
    )

    if not signals:
        st.info("No signals match the current filters.")
        return

    # Initialize comparison session state
    if 'compare_signals' not in st.session_state:
        st.session_state.compare_signals = []

    # Comparison mode controls
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"### Showing {len(signals)} Signals")

    with col2:
        compare_count = len(st.session_state.compare_signals)
        if compare_count > 0:
            st.info(f"{compare_count}/3 selected for comparison")

    with col3:
        if compare_count >= 2:
            if st.button("🔄 Compare Selected", type="primary"):
                st.session_state.show_comparison = True
        if compare_count > 0:
            if st.button("Clear Selection"):
                st.session_state.compare_signals = []
                st.session_state.show_comparison = False
                st.rerun()

    # Show comparison panel if active
    if st.session_state.get('show_comparison') and len(st.session_state.compare_signals) >= 2:
        _render_signal_comparison(st.session_state.compare_signals, signals)
        st.markdown("---")

    # Signal cards layout with comparison checkboxes
    for i, signal in enumerate(signals):
        signal_id = signal.get('id')
        col_check, col_card = st.columns([0.05, 0.95])

        with col_check:
            is_selected = signal_id in st.session_state.compare_signals
            can_select = len(st.session_state.compare_signals) < 3 or is_selected

            if can_select:
                if st.checkbox("", value=is_selected, key=f"compare_{signal_id}", label_visibility="collapsed"):
                    if signal_id not in st.session_state.compare_signals:
                        st.session_state.compare_signals.append(signal_id)
                else:
                    if signal_id in st.session_state.compare_signals:
                        st.session_state.compare_signals.remove(signal_id)

        with col_card:
            _render_signal_card(signal, service=service)

    # Export section
    st.markdown("---")
    st.markdown("### Export")

    col1, col2 = st.columns(2)

    with col1:
        # CSV export
        if signals:
            csv_data = _signals_to_csv(signals)
            st.download_button(
                "Download CSV (TradingView)",
                csv_data,
                f"scanner_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

    with col2:
        # Symbol list export
        if signals:
            symbols = "\n".join([f"NASDAQ:{s['ticker']}" for s in signals])
            st.download_button(
                "Download Symbol List",
                symbols,
                f"watchlist_{datetime.now().strftime('%Y%m%d')}.txt",
                "text/plain"
            )


def _render_signal_comparison(selected_ids: List[int], all_signals: List[Dict[str, Any]]):
    """Render side-by-side comparison of selected signals."""
    # Get the selected signals from the full list
    selected_signals = [s for s in all_signals if s.get('id') in selected_ids]

    if len(selected_signals) < 2:
        st.warning("Select at least 2 signals to compare")
        return

    st.markdown("### 🔄 Signal Comparison")

    # Create columns based on number of signals (2-3)
    cols = st.columns(len(selected_signals))

    for col_idx, signal in enumerate(selected_signals):
        with cols[col_idx]:
            ticker = signal.get('ticker', 'N/A')
            tier = signal.get('quality_tier', 'B')
            scanner = signal.get('scanner_name', '').replace('_', ' ').title()
            score = signal.get('composite_score', 0)
            entry = float(signal.get('entry_price', 0))
            stop = float(signal.get('stop_loss', 0))
            tp1 = float(signal.get('take_profit_1', 0))
            risk_pct = float(signal.get('risk_percent', 0))
            rr = float(signal.get('risk_reward_ratio', 0))

            # Claude data
            claude_grade = signal.get('claude_grade', '-')
            claude_action = signal.get('claude_action', '-')
            claude_thesis = signal.get('claude_thesis', '')

            # Tier colors
            tier_colors = {'A+': '#1e7e34', 'A': '#28a745', 'B': '#17a2b8', 'C': '#ffc107', 'D': '#6c757d'}
            tier_color = tier_colors.get(tier, '#6c757d')

            # Card header
            st.markdown(f"""
            <div class="comparison-card comparison-selected">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.3rem; font-weight: bold;">{ticker}</span>
                    <span style="background: {tier_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: bold;">{tier}</span>
                </div>
                <div style="font-size: 0.85rem; color: #555; margin-bottom: 0.5rem;">{scanner}</div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            st.metric("Composite Score", f"{score}")
            st.metric("Entry Price", f"${entry:.2f}")
            st.metric("Stop Loss", f"${stop:.2f}")
            st.metric("Take Profit", f"${tp1:.2f}")
            st.metric("Risk %", f"{risk_pct:.1f}%")
            st.metric("R:R Ratio", f"{rr:.2f}")

            # Claude Analysis
            st.markdown("---")
            st.markdown("**Claude Analysis**")

            if claude_grade and claude_grade != '-':
                action_colors = {'STRONG BUY': '#28a745', 'BUY': '#28a745', 'HOLD': '#ffc107', 'AVOID': '#dc3545'}
                action_color = action_colors.get(claude_action, '#6c757d')

                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <span style="background: {tier_colors.get(claude_grade, '#6c757d')}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.3rem;">{claude_grade}</span>
                    <span style="background: {action_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px;">{claude_action}</span>
                </div>
                """, unsafe_allow_html=True)

                if claude_thesis:
                    st.caption(claude_thesis[:150] + "..." if len(claude_thesis) > 150 else claude_thesis)
            else:
                st.caption("Not analyzed yet")

            # Score breakdown
            st.markdown("---")
            st.markdown("**Score Breakdown**")
            trend_score = float(signal.get('trend_score', 0))
            momentum_score = float(signal.get('momentum_score', 0))
            volume_score = float(signal.get('volume_score', 0))
            pattern_score = float(signal.get('pattern_score', 0))

            breakdown_data = {
                'Component': ['Trend', 'Momentum', 'Volume', 'Pattern'],
                'Score': [trend_score, momentum_score, volume_score, pattern_score]
            }
            st.bar_chart(pd.DataFrame(breakdown_data).set_index('Component'), height=150)

    # Close comparison button
    if st.button("✕ Close Comparison"):
        st.session_state.show_comparison = False
        st.rerun()


def _render_signal_card(signal: Dict[str, Any], service=None):
    """Render a single signal card using native Streamlit components."""
    signal_id = signal.get('id')
    tier = signal.get('quality_tier', 'D')
    ticker = signal.get('ticker', 'N/A')
    scanner = signal.get('scanner_name', '').replace('_', ' ').title()
    score = signal.get('composite_score', 0)
    entry = float(signal.get('entry_price', 0))
    stop = float(signal.get('stop_loss', 0))
    tp1 = float(signal.get('take_profit_1', 0))
    tp2 = float(signal.get('take_profit_2', 0)) if signal.get('take_profit_2') else None
    risk_pct = float(signal.get('risk_percent', 0))
    rr = float(signal.get('risk_reward_ratio', 0))
    setup = signal.get('setup_description', '')[:100]
    factors = signal.get('confluence_factors', [])
    if isinstance(factors, list):
        # Show all factors, separating technical and fundamental
        factors_str = ', '.join(factors)
    else:
        factors_str = str(factors) if factors else ''

    # Signal timestamp - format for header display
    signal_timestamp = signal.get('signal_timestamp')
    if signal_timestamp:
        if isinstance(signal_timestamp, datetime):
            signal_time_str = signal_timestamp.strftime('%b %d %H:%M')
        else:
            signal_time_str = str(signal_timestamp)[:16]
    else:
        signal_time_str = ''

    # Claude analysis data
    claude_grade = signal.get('claude_grade')
    claude_score = signal.get('claude_score')
    claude_action = signal.get('claude_action')
    claude_conviction = signal.get('claude_conviction')
    claude_thesis = signal.get('claude_thesis')
    claude_strengths = signal.get('claude_key_strengths', [])
    claude_risks = signal.get('claude_key_risks', [])
    claude_position = signal.get('claude_position_rec')
    claude_analyzed_at = signal.get('claude_analyzed_at')
    has_claude = claude_grade is not None

    # News sentiment data
    news_sentiment_score = signal.get('news_sentiment_score')
    news_sentiment_level = signal.get('news_sentiment_level')
    news_headlines_count = signal.get('news_headlines_count', 0)
    news_factors = signal.get('news_factors', [])
    news_analyzed_at = signal.get('news_analyzed_at')
    has_news = news_sentiment_score is not None

    # Format claude_analyzed_at timestamp with staleness check
    analyzed_time_str = None
    analyzed_ago_str = None
    is_analysis_stale = False
    if claude_analyzed_at:
        if isinstance(claude_analyzed_at, datetime):
            analyzed_time_str = claude_analyzed_at.strftime('%b %d, %Y %H:%M')
            # Calculate time ago
            now = datetime.now(claude_analyzed_at.tzinfo) if claude_analyzed_at.tzinfo else datetime.now()
            time_ago = now - claude_analyzed_at
            if time_ago.days > 0:
                analyzed_ago_str = f"{time_ago.days}d ago"
            elif time_ago.seconds >= 3600:
                analyzed_ago_str = f"{time_ago.seconds // 3600}h ago"
            else:
                analyzed_ago_str = f"{time_ago.seconds // 60}m ago"
            is_analysis_stale = time_ago.days > 7
        else:
            analyzed_time_str = str(claude_analyzed_at)[:16]

    # Tier styling
    tier_colors = {
        'A+': 'green', 'A': 'blue', 'B': 'orange', 'C': 'gray', 'D': 'red'
    }
    tier_color = tier_colors.get(tier, 'gray')

    # Scanner styling
    scanner_icons = {
        'Trend Momentum': '📈',
        'Breakout Confirmation': '🚀',
        'Mean Reversion': '🔄',
        'Gap And Go': '⚡',
        'Sector Rotation': '🔀',
        'Zlma Trend': '〰️',
        # Forex-adapted strategies
        'Smc Ema Trend': '🎯',
        'Ema Crossover': '📊',
        'Macd Momentum': '📉',
        # AlphaSuite-adapted strategies
        'Selling Climax': '💥',
        'Rsi Divergence': '↗️',
        'Wyckoff Spring': '🌱',
    }
    scanner_icon = scanner_icons.get(scanner, '📊')

    # Claude action colors
    claude_action_colors = {
        'STRONG BUY': 'green',
        'BUY': 'blue',
        'HOLD': 'orange',
        'AVOID': 'red'
    }

    # News sentiment colors
    news_sentiment_colors = {
        'very_bullish': 'green',
        'bullish': 'blue',
        'neutral': 'gray',
        'bearish': 'orange',
        'very_bearish': 'red'
    }
    news_sentiment_icons = {
        'very_bullish': '📰🟢',
        'bullish': '📰🔵',
        'neutral': '📰⚪',
        'bearish': '📰🟠',
        'very_bearish': '📰🔴'
    }

    # Build title with Claude info if available
    claude_badge = ""
    if has_claude:
        action_color = claude_action_colors.get(claude_action, 'gray')
        claude_badge = f" | 🤖 :{action_color}[{claude_action}] ({claude_grade})"

    # Build news badge if available
    news_badge = ""
    if has_news:
        news_level = news_sentiment_level or 'neutral'
        news_color = news_sentiment_colors.get(news_level.lower(), 'gray')
        news_icon = news_sentiment_icons.get(news_level.lower(), '📰')
        news_label = news_level.replace('_', ' ').title()
        news_badge = f" | {news_icon} :{news_color}[{news_label}]"

    # Build header with timestamp
    timestamp_part = f" | 📅 {signal_time_str}" if signal_time_str else ""

    # Use expander for each signal card (collapsed by default for easier browsing)
    with st.expander(f"**{ticker}** | :{tier_color}[{tier}] | Score: {score} | {scanner_icon} {scanner}{claude_badge}{news_badge}{timestamp_part}", expanded=False):

        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Entry", f"${entry:.2f}")
        with col2:
            st.metric("Stop Loss", f"${stop:.2f}")
        with col3:
            st.metric("Target 1", f"${tp1:.2f}")
        with col4:
            st.metric("Risk", f"{risk_pct:.1f}%")
        with col5:
            st.metric("R:R", f"{rr:.1f}:1")

        # Setup description
        if setup:
            st.caption(f"**Setup:** {setup}")

        # Confluence factors
        if factors_str:
            st.caption(f"**Factors:** {factors_str}")

        # Claude AI Analysis Section
        if has_claude:
            st.markdown("---")

            # Claude header with timestamp and re-analyze button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("#### 🤖 Claude AI Analysis")
                if analyzed_time_str:
                    ago_text = f" ({analyzed_ago_str})" if analyzed_ago_str else ""
                    if is_analysis_stale:
                        st.warning(f"⚠️ Analysis is stale - consider refreshing")
                    st.caption(f"📅 {analyzed_time_str}{ago_text}")
            with col2:
                # Re-analyze button
                if signal_id and service:
                    if st.button("🔄 Re-analyze", key=f"reanalyze_{signal_id}", help="Run fresh Claude AI analysis"):
                        with st.spinner(f"Re-analyzing {ticker}..."):
                            result = service.reanalyze_signal(signal_id)
                            if result.get('success'):
                                st.success(f"✅ {result.get('message', 'Analysis complete!')}")
                                st.rerun()
                            else:
                                st.error(f"❌ {result.get('error', 'Analysis failed')}")

            # Claude metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                grade_color = tier_colors.get(claude_grade, 'gray')
                st.markdown(f"**Grade:** :{grade_color}[{claude_grade}]")
            with col2:
                st.markdown(f"**Score:** {claude_score}/10")
            with col3:
                action_color = claude_action_colors.get(claude_action, 'gray')
                st.markdown(f"**Action:** :{action_color}[{claude_action}]")
            with col4:
                st.markdown(f"**Position:** {claude_position or '-'}")

            # Thesis
            if claude_thesis:
                st.markdown(f"**Thesis:** {claude_thesis}")

            # Strengths and Risks in columns
            if claude_strengths or claude_risks:
                col1, col2 = st.columns(2)
                with col1:
                    if claude_strengths:
                        st.markdown("**Key Strengths:**")
                        for s in claude_strengths[:3]:
                            st.markdown(f"- :green[+] {s}")
                with col2:
                    if claude_risks:
                        st.markdown("**Key Risks:**")
                        for r in claude_risks[:3]:
                            st.markdown(f"- :red[-] {r}")
        else:
            # Show pending analysis indicator
            st.caption("⏳ *Awaiting Claude AI analysis*")

        # News Sentiment Section
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("#### 📰 News Sentiment")
        with col2:
            # Enrich with News button
            if signal_id and service:
                button_label = "🔄 Refresh News" if has_news else "📰 Enrich with News"
                if st.button(button_label, key=f"news_{signal_id}", help="Fetch and analyze recent news"):
                    with st.spinner(f"Fetching news for {ticker}..."):
                        result = service.enrich_signal_with_news(signal_id)
                        if result.get('success'):
                            st.success(f"✅ {result.get('message', 'News enrichment complete!')}")
                            st.rerun()
                        else:
                            st.error(f"❌ {result.get('error', 'News enrichment failed')}")

        if has_news:
            # Format news_analyzed_at timestamp
            news_time_str = None
            news_ago_str = None
            if news_analyzed_at:
                if isinstance(news_analyzed_at, datetime):
                    news_time_str = news_analyzed_at.strftime('%b %d, %Y %H:%M')
                    now = datetime.now(news_analyzed_at.tzinfo) if news_analyzed_at.tzinfo else datetime.now()
                    time_ago = now - news_analyzed_at
                    if time_ago.days > 0:
                        news_ago_str = f"{time_ago.days}d ago"
                    elif time_ago.seconds >= 3600:
                        news_ago_str = f"{time_ago.seconds // 3600}h ago"
                    else:
                        news_ago_str = f"{time_ago.seconds // 60}m ago"
                else:
                    news_time_str = str(news_analyzed_at)[:16]

            if news_time_str:
                ago_text = f" ({news_ago_str})" if news_ago_str else ""
                st.caption(f"📅 {news_time_str}{ago_text}")

            # News metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                news_level = news_sentiment_level or 'neutral'
                news_color = news_sentiment_colors.get(news_level.lower(), 'gray')
                news_label = news_level.replace('_', ' ').title()
                st.markdown(f"**Sentiment:** :{news_color}[{news_label}]")
            with col2:
                score_display = f"{news_sentiment_score:.2f}" if news_sentiment_score else "N/A"
                st.markdown(f"**Score:** {score_display}")
            with col3:
                st.markdown(f"**Articles:** {news_headlines_count or 0}")

            # News factors
            if news_factors:
                if isinstance(news_factors, list):
                    factors_display = news_factors
                else:
                    factors_display = [news_factors] if news_factors else []

                if factors_display:
                    st.markdown("**Key Factors:**")
                    for factor in factors_display[:3]:
                        # Color-code based on sentiment level
                        if news_level.lower() in ['very_bullish', 'bullish']:
                            st.markdown(f"- :green[+] {factor}")
                        elif news_level.lower() in ['very_bearish', 'bearish']:
                            st.markdown(f"- :red[-] {factor}")
                        else:
                            st.markdown(f"- :gray[•] {factor}")
        else:
            st.caption("📰 *No news data yet - click 'Enrich with News' to fetch*")


def _signals_to_csv(signals: List[Dict]) -> str:
    """Convert signals to CSV format including Claude analysis and news sentiment."""
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    # Header with Claude and News columns
    writer.writerow([
        'Symbol', 'Side', 'Entry', 'Stop', 'TP1', 'TP2',
        'Risk%', 'R:R', 'Score', 'Tier', 'Scanner', 'Setup',
        'Claude_Grade', 'Claude_Score', 'Claude_Action', 'Claude_Conviction', 'Claude_Thesis',
        'News_Sentiment', 'News_Score', 'News_Articles'
    ])

    # Data
    for s in signals:
        writer.writerow([
            s.get('ticker', ''),
            s.get('signal_type', 'BUY'),
            f"{float(s.get('entry_price', 0)):.2f}",
            f"{float(s.get('stop_loss', 0)):.2f}",
            f"{float(s.get('take_profit_1', 0)):.2f}",
            f"{float(s.get('take_profit_2', 0)):.2f}" if s.get('take_profit_2') else '',
            f"{float(s.get('risk_percent', 0)):.1f}",
            f"{float(s.get('risk_reward_ratio', 0)):.1f}",
            s.get('composite_score', 0),
            s.get('quality_tier', 'D'),
            s.get('scanner_name', ''),
            (s.get('setup_description', '') or '')[:50],
            s.get('claude_grade', ''),
            s.get('claude_score', ''),
            s.get('claude_action', ''),
            s.get('claude_conviction', ''),
            (s.get('claude_thesis', '') or '')[:100],
            s.get('news_sentiment_level', ''),
            s.get('news_sentiment_score', ''),
            s.get('news_headlines_count', '')
        ])

    return output.getvalue()


# =============================================================================
# SCANNER ANALYSIS TAB
# =============================================================================

def render_scanner_analysis_tab(service):
    """
    Scanner Analysis Tab - Individual scanner performance drill-down.

    Provides detailed analysis of each scanner's performance including:
    - Signal count and quality distribution
    - Win/loss rates (if tracked)
    - Historical performance trends
    - Scanner-specific configuration info
    """
    st.header("Scanner Analysis")
    st.markdown("*Drill down into individual scanner performance and statistics*")

    # Get scanner stats
    stats = service.get_scanner_stats()
    by_scanner = stats.get('by_scanner', [])

    if not by_scanner:
        st.info("No scanner data available yet. Run scanners to generate signals.")
        return

    # Scanner selector and date filter
    from datetime import datetime, timedelta

    col_scanner, col_date_from, col_date_to = st.columns([2, 1, 1])

    with col_scanner:
        scanner_names = [s['scanner_name'] for s in by_scanner]
        selected_scanner = st.selectbox(
            "Select Scanner to Analyze",
            scanner_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col_date_from:
        # Default to 30 days ago
        default_from = datetime.now().date() - timedelta(days=30)
        date_from = st.date_input(
            "From Date",
            value=default_from,
            max_value=datetime.now().date()
        )

    with col_date_to:
        date_to = st.date_input(
            "To Date",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )

    st.markdown("---")

    # Find selected scanner stats
    scanner_stats = next((s for s in by_scanner if s['scanner_name'] == selected_scanner), None)

    if scanner_stats:
        # Scanner overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Signals", scanner_stats.get('signal_count', 0))
        with col2:
            st.metric("Active Signals", scanner_stats.get('active_count', 0))
        with col3:
            st.metric("Avg Score", f"{scanner_stats.get('avg_score', 0):.1f}")
        with col4:
            # Calculate quality ratio
            total = scanner_stats.get('signal_count', 0)
            active = scanner_stats.get('active_count', 0)
            active_pct = (active / total * 100) if total > 0 else 0
            st.metric("Active %", f"{active_pct:.0f}%")

        st.markdown("---")

        # Get signals for this scanner with date filter, ordered by most recent first
        signals = service.get_scanner_signals(
            scanner_name=selected_scanner,
            status=None,  # All statuses
            signal_date_from=str(date_from),
            signal_date_to=str(date_to),
            limit=200,
            order_by='timestamp'  # Show latest signals first
        )

        if signals:
            # Quality distribution
            st.subheader("Quality Distribution")

            tier_counts = {}
            for s in signals:
                tier = s.get('quality_tier', 'Unknown')
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            # Display as horizontal bar chart using columns
            tier_order = ['A+', 'A', 'B', 'C', 'D']
            tier_colors = {'A+': '🟢', 'A': '🔵', 'B': '🟡', 'C': '🟠', 'D': '🔴'}

            for tier in tier_order:
                if tier in tier_counts:
                    count = tier_counts[tier]
                    pct = count / len(signals) * 100
                    st.write(f"{tier_colors.get(tier, '⚪')} **{tier}**: {count} signals ({pct:.1f}%)")

            st.markdown("---")

            # Signal type distribution
            st.subheader("Signal Direction")
            buy_count = sum(1 for s in signals if s.get('signal_type') == 'BUY')
            sell_count = sum(1 for s in signals if s.get('signal_type') == 'SELL')

            col1, col2 = st.columns(2)
            with col1:
                st.metric("BUY Signals", buy_count, delta=None)
            with col2:
                st.metric("SELL Signals", sell_count, delta=None)

            st.markdown("---")

            # Claude Analysis Summary (if any)
            claude_analyzed = [s for s in signals if s.get('claude_analyzed_at')]
            if claude_analyzed:
                st.subheader("Claude AI Analysis Summary")

                # Claude action distribution
                action_counts = {}
                for s in claude_analyzed:
                    action = s.get('claude_action', 'Unknown')
                    action_counts[action] = action_counts.get(action, 0) + 1

                cols = st.columns(len(action_counts))
                for i, (action, count) in enumerate(sorted(action_counts.items())):
                    with cols[i % len(cols)]:
                        color = {'STRONG BUY': '🟢', 'BUY': '🔵', 'HOLD': '🟡', 'AVOID': '🔴'}.get(action, '⚪')
                        st.metric(f"{color} {action}", count)

                # Average Claude score
                claude_scores = [s.get('claude_score', 0) for s in claude_analyzed if s.get('claude_score')]
                if claude_scores:
                    avg_claude_score = sum(claude_scores) / len(claude_scores)
                    st.metric("Avg Claude Score", f"{avg_claude_score:.1f}/100")

            st.markdown("---")

            # Recent signals table with date range
            st.subheader(f"{selected_scanner.replace('_', ' ').title()} Signals ({date_from} to {date_to})")

            # Prepare data for table with SL/TP levels
            table_data = []
            for s in signals[:50]:  # Show up to 50 signals in date range
                entry = float(s.get('entry_price', 0))
                stop_loss = float(s.get('stop_loss', 0)) if s.get('stop_loss') else None
                tp1 = float(s.get('take_profit_1', 0)) if s.get('take_profit_1') else None
                tp2 = float(s.get('take_profit_2', 0)) if s.get('take_profit_2') else None
                rr = float(s.get('risk_reward_ratio', 0)) if s.get('risk_reward_ratio') else None
                risk_pct = float(s.get('risk_percent', 0)) if s.get('risk_percent') else None

                table_data.append({
                    'Ticker': s.get('ticker', ''),
                    'Type': s.get('signal_type', ''),
                    'Entry': f"${entry:.2f}",
                    'Stop Loss': f"${stop_loss:.2f}" if stop_loss else '-',
                    'TP1': f"${tp1:.2f}" if tp1 else '-',
                    'TP2': f"${tp2:.2f}" if tp2 else '-',
                    'R:R': f"{rr:.1f}:1" if rr else '-',
                    'Risk%': f"{risk_pct:.1f}%" if risk_pct else '-',
                    'Score': s.get('composite_score', 0),
                    'Tier': s.get('quality_tier', ''),
                    'Claude': s.get('claude_action', '-'),
                    'Date': str(s.get('signal_timestamp', ''))[:10] if s.get('signal_timestamp') else ''
                })

            if table_data:
                import pandas as pd
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No signals found for {selected_scanner.replace('_', ' ').title()}")

    # Scanner descriptions
    st.markdown("---")
    st.subheader("Scanner Information")

    # Scanner descriptions for the 7 active scanners
    scanner_info = {
        'reversal_scanner': {
            'description': 'Capitulation + mean reversion + Wyckoff spring patterns. Best performing scanner (PF 2.44).',
            'best_for': 'Oversold bounces, selling climax reversals, capitulation plays',
            'criteria': 'RSI oversold, volume spike, bullish reversal candles, support tests'
        },
        'ema_pullback': {
            'description': 'EMA trend pullback strategy with optimized filters for high-probability entries.',
            'best_for': 'Strong uptrends with healthy pullbacks',
            'criteria': 'EMA 20/50/100/200 alignment, ADX>20, MACD>0, RSI 40-60, Volume>=1.2x avg'
        },
        'macd_momentum': {
            'description': 'MACD momentum confluence with price structure.',
            'best_for': 'Momentum confirmation trades',
            'criteria': 'MACD crossover, histogram expansion, price structure alignment'
        },
        'zlma_trend': {
            'description': 'Zero-Lag Moving Average crossover with EMA confirmation.',
            'best_for': 'Trend-following with reduced lag',
            'criteria': 'ZLMA/EMA crossover, ATR-based stops, trend alignment'
        },
        'breakout_confirmation': {
            'description': 'Identifies volume-confirmed breakouts above 52-week highs.',
            'best_for': 'Range breakouts, new highs momentum',
            'criteria': 'Price break + volume surge + trend alignment'
        },
        'gap_and_go': {
            'description': 'Gap continuation plays with momentum confirmation.',
            'best_for': 'Opening momentum, news-driven moves',
            'criteria': 'Gap up >2% + volume + VWAP hold + trend continuation'
        },
        'rsi_divergence': {
            'description': 'Price/RSI divergence reversals with confirmation.',
            'best_for': 'Counter-trend entries, exhaustion plays',
            'criteria': 'RSI bullish divergence, support test, reversal candle'
        }
    }

    info = scanner_info.get(selected_scanner, {})
    if info:
        st.markdown(f"**Description:** {info.get('description', 'N/A')}")
        st.markdown(f"**Best For:** {info.get('best_for', 'N/A')}")
        st.markdown(f"**Entry Criteria:** {info.get('criteria', 'N/A')}")


# =============================================================================
# BROKER STATISTICS TAB
# =============================================================================

def render_broker_stats_tab():
    """Render the Broker Statistics tab with real trading performance from RoboMarkets."""
    from services.broker_analytics_service import get_broker_service

    broker_service = get_broker_service()

    # Check if configured
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
        st.subheader("Account Overview")
        bal_col1, bal_col2, bal_col3, bal_col4 = st.columns(4)

        with bal_col1:
            # Determine trend indicator
            if trend_data.get("trend") == "up":
                delta_color = "normal"  # Green
                trend_arrow = "+"
            elif trend_data.get("trend") == "down":
                delta_color = "inverse"  # Red shows as negative
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
            # 7-day trend indicator
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

    # Key metrics row 1
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Net Profit",
            f"${stats.net_profit:,.2f}",
            delta=None
        )

    with col2:
        st.metric(
            "Win Rate",
            f"{stats.win_rate:.1f}%",
            delta=f"{stats.winning_trades}W / {stats.losing_trades}L"
        )

    with col3:
        st.metric(
            "Profit Factor",
            f"{stats.profit_factor:.2f}",
            delta="Good" if stats.profit_factor > 1.5 else ("Fair" if stats.profit_factor > 1 else "Poor")
        )

    with col4:
        st.metric(
            "Total Trades",
            stats.total_trades,
            delta=f"Last {days} days"
        )

    # Key metrics row 2
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Win", f"${stats.avg_profit:,.2f}", delta=f"+{stats.avg_profit_pct:.1f}%")

    with col2:
        st.metric("Avg Loss", f"${stats.avg_loss:,.2f}", delta=f"-{stats.avg_loss_pct:.1f}%")

    with col3:
        st.metric("Largest Win", f"${stats.largest_win:,.2f}")

    with col4:
        st.metric("Largest Loss", f"${stats.largest_loss:,.2f}")

    # Risk & Streaks row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Expectancy", f"${stats.expectancy:,.2f}", help="Expected value per trade")

    with col2:
        st.metric("Max Drawdown", f"${stats.max_drawdown:,.2f}", delta=f"-{stats.max_drawdown_pct:.1f}%")

    with col3:
        st.metric("Win Streak (max)", stats.max_consecutive_wins)

    with col4:
        st.metric("Loss Streak (max)", stats.max_consecutive_losses)

    st.markdown("---")

    # Two column layout for open positions and side breakdown
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Open Positions")
        if positions_data.get("positions"):
            pos_df = pd.DataFrame(positions_data["positions"])
            if not pos_df.empty:
                # Format for display
                display_df = pos_df[["ticker", "side", "quantity", "entry_price", "current_price", "unrealized_pnl", "profit_pct"]].copy()
                display_df.columns = ["Ticker", "Side", "Qty", "Entry", "Current", "P&L ($)", "P&L (%)"]

                # Style the dataframe
                st.dataframe(
                    display_df.style.format({
                        "Entry": "${:.2f}",
                        "Current": "${:.2f}",
                        "P&L ($)": "${:+,.2f}",
                        "P&L (%)": "{:+.2f}%"
                    }).applymap(
                        lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                        subset=["P&L ($)", "P&L (%)"]
                    ),
                    use_container_width=True,
                    height=250
                )

                # Total unrealized
                st.metric("Total Unrealized P&L", f"${positions_data['total_unrealized_pnl']:+,.2f}")
        else:
            st.info("No open positions")

    with col2:
        st.subheader("Performance by Side")

        # Long vs Short stats
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

        # Duration stats
        st.subheader("Trade Duration")
        st.metric("Avg Hold Time", f"{stats.avg_trade_duration_hours:.1f} hours")

    st.markdown("---")

    # Equity Curve Chart
    if stats.equity_curve:
        st.subheader("Equity Curve")

        equity_df = pd.DataFrame(stats.equity_curve, columns=["Date", "Equity"])
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

    # Daily P&L chart
    if stats.by_day:
        st.subheader("Daily Profit/Loss")

        daily_df = pd.DataFrame([
            {"Date": day, "Profit": data["profit"], "Trades": data["trades"]}
            for day, data in sorted(stats.by_day.items())
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

    # Ticker breakdown
    if stats.by_ticker:
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
            for ticker, data in sorted(stats.by_ticker.items(), key=lambda x: x[1]["profit"], reverse=True)
        ])

        st.dataframe(
            ticker_df.style.format({"Profit": "${:,.2f}"}).applymap(
                lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                subset=["Profit"]
            ),
            use_container_width=True,
            height=300
        )

    # Recent trades table
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
                }).applymap(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                    subset=["Profit ($)", "Profit (%)"]
                ),
                use_container_width=True,
                height=400
            )
    else:
        st.info(f"No closed trades in the last {days} days")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    service = get_stock_service()

    # Create tabs - new 6-tab structure
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "📡 Signals",
        "📋 Watchlists",
        "📈 Scanner Performance",
        "🔍 Deep Dive",
        "💹 Broker Stats"
    ])

    with tab1:
        render_dashboard_tab(service)

    with tab2:
        # Unified signal browser with all scanner signals
        render_scanner_signals_tab(service)

    with tab3:
        # 5 predefined technical watchlists
        render_watchlists_tab(service)

    with tab4:
        # Per-scanner performance metrics from backtests
        render_scanner_analysis_tab(service)

    with tab5:
        render_deep_dive_tab(service)

    with tab6:
        render_broker_stats_tab()


if __name__ == "__main__":
    main()
