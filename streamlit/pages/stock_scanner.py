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

    # Key metrics - Row 1
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

    # SMC metrics - Row 2
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        bullish = stats.get('smc_bullish', 0)
        st.metric("SMC Bullish", bullish, delta=None, help="Stocks with bullish market structure")

    with col2:
        bearish = stats.get('smc_bearish', 0)
        st.metric("SMC Bearish", bearish, delta=None, help="Stocks with bearish market structure")

    with col3:
        neutral = stats.get('smc_neutral', 0)
        st.metric("SMC Neutral", neutral, delta=None, help="Stocks with neutral/ranging structure")

    with col4:
        total_smc = bullish + bearish + neutral
        if total_smc > 0:
            bull_pct = round(bullish / total_smc * 100)
            st.metric("Bull/Bear Ratio", f"{bull_pct}% / {100-bull_pct-round(neutral/total_smc*100)}%")
        else:
            st.metric("Bull/Bear Ratio", "N/A")

    # Fundamentals metrics - Row 3
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        earnings_count = stats.get('upcoming_earnings', 0)
        st.metric("Earnings (14d)", earnings_count, delta=None, help="Stocks with earnings in next 14 days")

    with col2:
        high_short = stats.get('high_short_interest', 0)
        st.metric("High Short Interest", high_short, delta=None, help="Stocks with >15% short float (squeeze potential)")

    with col3:
        st.metric("Fundamentals", "Weekly", delta=None, help="Fundamental data refreshed weekly on Sundays")

    with col4:
        st.metric("Data Source", "yfinance", delta=None, help="Earnings, beta, short interest from yfinance")

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

    # SMC Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("**SMC Filters**")

    smc_trend_options = st.sidebar.multiselect(
        "SMC Trend",
        options=['Bullish', 'Bearish', 'Neutral'],
        default=[],
        help="Filter by Smart Money Concepts market structure"
    )
    filters['smc_trends'] = smc_trend_options if smc_trend_options else None

    smc_zone_options = st.sidebar.multiselect(
        "Price Zone",
        options=['Extreme Discount', 'Discount', 'Equilibrium', 'Premium', 'Extreme Premium'],
        default=[],
        help="Filter by premium/discount zone"
    )
    filters['smc_zones'] = smc_zone_options if smc_zone_options else None

    # Fundamentals filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Fundamentals**")

    earnings_filter = st.sidebar.checkbox("Has earnings in next 14 days", value=False)
    filters['earnings_within_days'] = 14 if earnings_filter else None

    short_filter = st.sidebar.checkbox("High short interest (>10%)", value=False)
    filters['min_short_interest'] = 10.0 if short_filter else None

    # Enhanced Signal Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Technical Signals**")

    rsi_signal_options = st.sidebar.multiselect(
        "RSI Signal",
        options=['oversold_extreme', 'oversold', 'neutral', 'overbought', 'overbought_extreme'],
        default=[],
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Filter by RSI classification"
    )
    filters['rsi_signals'] = rsi_signal_options if rsi_signal_options else None

    sma_cross_options = st.sidebar.multiselect(
        "SMA Cross",
        options=['golden_cross', 'death_cross', 'bullish', 'bearish'],
        default=[],
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Golden Cross = SMA50 crossed above SMA200"
    )
    filters['sma_cross_signals'] = sma_cross_options if sma_cross_options else None

    macd_cross_options = st.sidebar.multiselect(
        "MACD Signal",
        options=['bullish_cross', 'bearish_cross', 'bullish', 'bearish'],
        default=[],
        format_func=lambda x: x.replace('_', ' ').title(),
        help="MACD histogram direction"
    )
    filters['macd_cross_signals'] = macd_cross_options if macd_cross_options else None

    high_low_options = st.sidebar.multiselect(
        "52W Position",
        options=['new_high', 'near_high', 'neutral', 'near_low', 'new_low'],
        default=[],
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Position relative to 52-week high/low"
    )
    filters['high_low_signals'] = high_low_options if high_low_options else None

    pattern_options = st.sidebar.multiselect(
        "Candlestick Pattern",
        options=['hammer', 'inverted_hammer', 'bullish_engulfing', 'bearish_engulfing',
                 'doji', 'dragonfly_doji', 'gravestone_doji', 'hanging_man', 'shooting_star',
                 'strong_bullish', 'strong_bearish', 'bullish_marubozu', 'bearish_marubozu'],
        default=[],
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Latest candlestick pattern"
    )
    filters['candlestick_patterns'] = pattern_options if pattern_options else None

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
            smc_trends=filters['smc_trends'],
            smc_zones=filters['smc_zones'],
            min_rsi=filters['min_rsi'],
            max_rsi=filters['max_rsi'],
            min_rvol=filters['min_rvol'],
            max_rvol=filters['max_rvol'],
            has_signal=filters['has_signal'],
            is_new_to_tier=filters['is_new_to_tier'],
            earnings_within_days=filters['earnings_within_days'],
            min_short_interest=filters['min_short_interest'],
            # Technical signal filters
            rsi_signals=filters.get('rsi_signals'),
            sma_cross_signals=filters.get('sma_cross_signals'),
            macd_cross_signals=filters.get('macd_cross_signals'),
            high_low_signals=filters.get('high_low_signals'),
            candlestick_patterns=filters.get('candlestick_patterns')
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
    display_df['SMC'] = display_df['smc_trend'].apply(lambda x: x if x else '-')
    display_df['Zone'] = display_df['smc_zone'].apply(lambda x: x.replace('Extreme ', 'X-').replace('Equilibrium', 'EQ') if x else '-')
    # Fundamentals columns (with safety checks for missing columns)
    if 'earnings_date' in display_df.columns:
        display_df['Earn'] = display_df['earnings_date'].apply(
            lambda x: x.strftime('%m/%d') if pd.notnull(x) else '-'
        )
    else:
        display_df['Earn'] = '-'

    if 'beta' in display_df.columns:
        display_df['Beta'] = display_df['beta'].apply(
            lambda x: f"{x:.1f}" if pd.notnull(x) else '-'
        )
    else:
        display_df['Beta'] = '-'

    if 'short_percent_float' in display_df.columns:
        display_df['Short%'] = display_df['short_percent_float'].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) and x > 0 else '-'
        )
    else:
        display_df['Short%'] = '-'

    # Enhanced signal columns (with safety checks)
    if 'rsi_signal' in display_df.columns:
        display_df['RSI_Sig'] = display_df['rsi_signal'].apply(
            lambda x: x.replace('_', ' ').title() if x else '-'
        )
    else:
        display_df['RSI_Sig'] = '-'

    if 'sma_cross_signal' in display_df.columns:
        display_df['SMA_X'] = display_df['sma_cross_signal'].apply(
            lambda x: x.replace('_', ' ').title() if x else '-'
        )
    else:
        display_df['SMA_X'] = '-'

    if 'macd_cross_signal' in display_df.columns:
        display_df['MACD_X'] = display_df['macd_cross_signal'].apply(
            lambda x: x.replace('_', ' ').title() if x else '-'
        )
    else:
        display_df['MACD_X'] = '-'

    if 'high_low_signal' in display_df.columns:
        display_df['52W'] = display_df['high_low_signal'].apply(
            lambda x: x.replace('_', ' ').title() if x else '-'
        )
    else:
        display_df['52W'] = '-'

    if 'candlestick_pattern' in display_df.columns:
        display_df['Pattern'] = display_df['candlestick_pattern'].apply(
            lambda x: x.replace('_', ' ').title() if x else '-'
        )
    else:
        display_df['Pattern'] = '-'

    result_df = display_df[['rank_overall', 'tier', 'ticker', 'name', 'Score', 'Price', 'ATR%', '1D%', 'RSI', 'RSI_Sig', 'SMA_X', 'MACD_X', '52W', 'Pattern', 'SMC', 'Earn', 'Short%', 'Signal']].rename(columns={
        'rank_overall': 'Rank', 'tier': 'Tier', 'ticker': 'Ticker', 'name': 'Name'
    })
    result_df['Name'] = result_df['Name'].apply(lambda x: x[:18] + '...' if len(str(x)) > 18 else x)

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

    def style_smc(val):
        if val == 'Bullish':
            return 'color: #28a745; font-weight: bold;'
        elif val == 'Bearish':
            return 'color: #dc3545; font-weight: bold;'
        return ''

    def style_zone(val):
        if 'Discount' in str(val):
            return 'background-color: #d4edda; color: #155724;'
        elif 'Premium' in str(val):
            return 'background-color: #f8d7da; color: #721c24;'
        return ''

    def style_change(val):
        if '+' in str(val):
            return 'color: #28a745;'
        elif '-' in str(val) and val != '-':
            return 'color: #dc3545;'
        return ''

    def style_short(val):
        """Highlight high short interest stocks (squeeze potential)."""
        if val != '-':
            try:
                pct = float(val.replace('%', ''))
                if pct >= 15:
                    return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                elif pct >= 10:
                    return 'background-color: #ffeeba; color: #856404;'
            except Exception:
                pass
        return ''

    def style_earn(val):
        """Highlight stocks with upcoming earnings."""
        if val != '-':
            return 'background-color: #d1ecf1; color: #0c5460; font-weight: bold;'
        return ''

    def style_rsi_signal(val):
        """Color RSI signals - green for oversold (buy opportunity), red for overbought (sell opportunity)."""
        val_lower = str(val).lower()
        if 'oversold' in val_lower:
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif 'overbought' in val_lower:
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        return ''

    def style_cross_signal(val):
        """Color cross signals - green for bullish, red for bearish."""
        val_lower = str(val).lower()
        if 'golden' in val_lower or 'bullish' in val_lower:
            return 'color: #28a745; font-weight: bold;'
        elif 'death' in val_lower or 'bearish' in val_lower:
            return 'color: #dc3545; font-weight: bold;'
        return ''

    def style_52w_signal(val):
        """Color 52-week position signals."""
        val_lower = str(val).lower()
        if 'new high' in val_lower or 'near high' in val_lower:
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif 'new low' in val_lower or 'near low' in val_lower:
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        return ''

    def style_pattern(val):
        """Color candlestick patterns - bullish green, bearish red."""
        val_lower = str(val).lower()
        bullish_patterns = ['hammer', 'bullish engulfing', 'morning star', 'piercing', 'three white']
        bearish_patterns = ['hanging man', 'bearish engulfing', 'evening star', 'dark cloud', 'three black']
        if any(p in val_lower for p in bullish_patterns):
            return 'color: #28a745; font-weight: bold;'
        elif any(p in val_lower for p in bearish_patterns):
            return 'color: #dc3545; font-weight: bold;'
        return ''

    styled_df = (result_df.style
        .map(style_tier, subset=['Tier'])
        .map(style_signal, subset=['Signal'])
        .map(style_smc, subset=['SMC'])
        .map(style_change, subset=['1D%'])
        .map(style_short, subset=['Short%'])
        .map(style_earn, subset=['Earn'])
        .map(style_rsi_signal, subset=['RSI_Sig'])
        .map(style_cross_signal, subset=['SMA_X', 'MACD_X'])
        .map(style_52w_signal, subset=['52W'])
        .map(style_pattern, subset=['Pattern'])
    )

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

            smc_trend = row.get('smc_trend', '')
            smc_align = row.get('smc_alignment', '')
            align_icon = "✓" if smc_align == "Aligned" else ("✗" if smc_align == "Divergent" else "?")
            align_color = "#28a745" if smc_align == "Aligned" else ("#dc3545" if smc_align == "Divergent" else "#666")

            st.markdown(f"""
            <div class="signal-card {sig_class}">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>{row['ticker']}</b> <span style="color: {sig_color};">{sig_type}</span> {tier_str}</span>
                    <span style="font-size: 0.8rem; color: #666;">{time_str}</span>
                </div>
                <div style="margin-top: 0.3rem; font-size: 0.85rem;">
                    Entry: ${row['entry_price']:.2f} | SL: ${row['stop_loss']:.2f} | TP: ${row['take_profit']:.2f} | R:R: {rr_str}
                </div>
                <div style="margin-top: 0.2rem; font-size: 0.8rem; color: #555;">
                    SMC: {smc_trend if smc_trend else '-'} <span style="color: {align_color};">{align_icon}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### All Signals")

        display_df = df.copy()
        display_df['Time'] = pd.to_datetime(display_df['signal_timestamp']).dt.strftime('%m-%d %H:%M')
        display_df['Entry'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
        display_df['SL'] = display_df['stop_loss'].apply(lambda x: f"${x:.2f}")
        display_df['TP'] = display_df['take_profit'].apply(lambda x: f"${x:.2f}")
        display_df['R:R'] = display_df['risk_reward'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) and x > 0 else '-')
        display_df['Conf'] = display_df['confidence'].apply(lambda x: f"{x:.0f}%")
        display_df['Tier'] = display_df['tier'].apply(lambda x: f"T{int(x)}" if pd.notnull(x) else '-')
        display_df['SMC'] = display_df['smc_trend'].apply(lambda x: x[:4] if x else '-')
        display_df['Zone'] = display_df['smc_zone'].apply(lambda x: x.replace('Extreme ', 'X-').replace('Equilibrium', 'EQ')[:8] if x else '-')
        display_df['Align'] = display_df['smc_alignment'].apply(lambda x: '✓' if x == 'Aligned' else ('✗' if x == 'Divergent' else '-'))

        result_df = display_df[['Time', 'ticker', 'signal_type', 'Entry', 'R:R', 'Conf', 'Tier', 'SMC', 'Zone', 'Align']].rename(columns={'ticker': 'Ticker', 'signal_type': 'Type'})

        def style_type(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            return ''

        def style_align(val):
            if val == '✓':
                return 'color: #28a745; font-weight: bold;'
            elif val == '✗':
                return 'color: #dc3545; font-weight: bold;'
            return ''

        styled_df = result_df.style.map(style_type, subset=['Type']).map(style_align, subset=['Align'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

        csv = df.to_csv(index=False)
        st.download_button("Export Signals CSV", csv, f"signals_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


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
    navigated_ticker = st.session_state.get('deep_dive_ticker', None)
    if navigated_ticker:
        # Clear it after use to prevent sticky behavior
        st.session_state.deep_dive_ticker = None
        ticker = navigated_ticker
        st.info(f"Analyzing **{ticker}** from Top Picks")
    else:
        ticker = None

    # Search (only show if no navigated ticker)
    if not ticker:
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
        st.metric("ATR %", f"{atr_pct:.2f}%")

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


# =============================================================================
# TOP PICKS TAB
# =============================================================================

def render_top_picks_tab(service):
    """Render the Daily Top Picks tab."""
    st.markdown("""
    <div class="main-header">
        <h2>Daily Top Picks</h2>
        <p>High-quality BUY candidates ranked by signal confluence and quality metrics</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating top picks..."):
        picks_data = service.get_daily_top_picks()

    if not picks_data or picks_data.get('total_picks', 0) == 0:
        st.warning("No top picks available. Please ensure the metrics have been calculated.")
        return

    # Initialize Claude analysis session state
    if 'claude_top_picks_analysis' not in st.session_state:
        st.session_state.claude_top_picks_analysis = {}

    # Header stats
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Picks", picks_data['total_picks'])
    col2.metric("Momentum", len(picks_data.get('momentum', [])))
    col3.metric("Breakout", len(picks_data.get('breakout', [])))
    col4.metric("Bounce Plays", len(picks_data.get('mean_reversion', [])))
    col5.metric("Data Date", picks_data.get('date', 'N/A'))

    # Claude Analysis section
    st.markdown("---")
    st.markdown("### Claude AI Analysis")

    all_picks = (
        picks_data.get('momentum', []) +
        picks_data.get('breakout', []) +
        picks_data.get('mean_reversion', [])
    )

    # Find picks without Claude analysis (not in DB and not in session)
    unanalyzed_picks = []
    for pick in all_picks:
        ticker = pick['ticker']
        has_db_analysis = pick.get('claude_grade') is not None
        has_session_analysis = st.session_state.claude_top_picks_analysis.get(ticker, {}).get('success', False)
        if not has_db_analysis and not has_session_analysis:
            unanalyzed_picks.append(pick)

    # Sort unanalyzed by score
    unanalyzed_picks.sort(key=lambda x: x['total_score'], reverse=True)
    unanalyzed_count = len(unanalyzed_picks)

    col_btn1, col_btn2, col_info = st.columns([1, 1, 2])
    with col_btn1:
        btn_label = f"Analyze {unanalyzed_count} Unanalyzed" if unanalyzed_count > 0 else "All Analyzed"
        btn_disabled = unanalyzed_count == 0
        if st.button(btn_label, type="primary", key="analyze_unanalyzed", disabled=btn_disabled):
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, pick in enumerate(unanalyzed_picks):
                ticker = pick['ticker']
                status_text.text(f"Analyzing {ticker}... ({i+1}/{unanalyzed_count})")
                result = service.analyze_top_pick_with_claude(pick)
                st.session_state.claude_top_picks_analysis[ticker] = result
                progress_bar.progress((i + 1) / unanalyzed_count)
            status_text.text("Analysis complete!")
            # Clear cache to reload picks with fresh Claude data
            service.get_daily_top_picks.clear()
            st.rerun()

    with col_btn2:
        if st.button("Clear Session Analysis", key="clear_claude"):
            st.session_state.claude_top_picks_analysis = {}
            st.rerun()

    with col_info:
        total_picks = len(all_picks)
        analyzed_in_db = len([p for p in all_picks if p.get('claude_grade') is not None])
        analyzed_in_session = len([a for a in st.session_state.claude_top_picks_analysis.values() if a.get('success')])
        total_analyzed = analyzed_in_db + analyzed_in_session - len([
            p for p in all_picks
            if p.get('claude_grade') is not None and st.session_state.claude_top_picks_analysis.get(p['ticker'], {}).get('success', False)
        ])
        st.info(f"{total_analyzed}/{total_picks} picks analyzed | {unanalyzed_count} remaining")

    st.markdown("---")

    # Render each category with Claude analysis data
    col1, col2, col3 = st.columns(3)

    with col1:
        _render_picks_category(
            "Momentum Riders",
            "Trending stocks with bullish confirmation signals",
            picks_data.get('momentum', []),
            "#1a5f7a",
            service,
            st.session_state.claude_top_picks_analysis
        )

    with col2:
        _render_picks_category(
            "Breakout Watch",
            "Stocks near 52W highs with volume surge",
            picks_data.get('breakout', []),
            "#28a745",
            service,
            st.session_state.claude_top_picks_analysis
        )

    with col3:
        _render_picks_category(
            "Bounce Plays",
            "Oversold stocks showing reversal patterns",
            picks_data.get('mean_reversion', []),
            "#6f42c1",
            service,
            st.session_state.claude_top_picks_analysis
        )

    # Detailed table view
    st.markdown("---")
    st.markdown("### All Picks - Detailed View")
    st.caption("Click a stock name to open Deep Dive analysis")

    # Re-gather all picks for the table (already defined above but scoped differently)
    all_picks_table = (
        picks_data.get('momentum', []) +
        picks_data.get('breakout', []) +
        picks_data.get('mean_reversion', [])
    )

    if all_picks_table:
        # Sort by total_score descending
        all_picks_table.sort(key=lambda x: x['total_score'], reverse=True)

        # Get Claude analysis from session state
        claude_analysis = st.session_state.get('claude_top_picks_analysis', {})

        # Initialize deep dive ticker in session state if not present
        if 'deep_dive_ticker' not in st.session_state:
            st.session_state.deep_dive_ticker = None

        # Create header row
        header_cols = st.columns([0.8, 2, 1.2, 0.6, 0.5, 0.8, 0.7, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8])
        headers = ['Ticker', 'Name', 'Category', 'Score', 'Tier', 'Price', '1D%', 'RVol', 'ATR%', 'Stop%', 'R/R', 'AI Grade', 'AI Action']
        for col, header in zip(header_cols, headers):
            col.markdown(f"**{header}**")

        st.markdown("---")

        # Build rows with clickable names
        df_data = []  # Keep for CSV export
        for idx, p in enumerate(all_picks_table):
            ticker = p['ticker']

            # Check for Claude analysis from both DB and session
            has_claude_from_db = p.get('claude_grade') is not None
            claude_data_from_session = claude_analysis.get(ticker, {})
            has_claude_from_session = claude_data_from_session.get('success', False)
            has_claude = has_claude_from_db or has_claude_from_session

            # Get Claude values (prefer session, fallback to DB)
            if has_claude_from_session:
                ai_grade = claude_data_from_session.get('claude_grade', '-')
                ai_score = claude_data_from_session.get('claude_score', 0)
                ai_action = claude_data_from_session.get('claude_action', '-')
            elif has_claude_from_db:
                ai_grade = p.get('claude_grade', '-')
                ai_score = p.get('claude_score', 0)
                ai_action = p.get('claude_action', '-')
            else:
                ai_grade = '-'
                ai_score = 0
                ai_action = '-'

            # Category colors
            cat_colors = {'Momentum': '#1a5f7a', 'Breakout': '#28a745', 'Mean Reversion': '#6f42c1'}
            cat_color = cat_colors.get(p['category'], '#666')

            # Tier colors
            tier_colors = {1: '#28a745', 2: '#17a2b8', 3: '#ffc107'}
            tier_color = tier_colors.get(p['tier'], '#666')

            # AI action colors
            action_colors = {'STRONG BUY': '#28a745', 'BUY': '#28a745', 'HOLD': '#ffc107', 'AVOID': '#dc3545'}
            action_color = action_colors.get(ai_action, '#666')

            # 1D% color
            change_1d = p.get('price_change_1d', 0) or 0
            change_color = '#28a745' if change_1d >= 0 else '#dc3545'

            # Create row
            row_cols = st.columns([0.8, 2, 1.2, 0.6, 0.5, 0.8, 0.7, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8])

            row_cols[0].write(ticker)

            # Clickable name - button styled as link (opens inline deep dive below table)
            name_display = p.get('name', '')[:18] + '...' if len(p.get('name', '')) > 18 else p.get('name', '')
            if row_cols[1].button(f"🔍 {name_display}", key=f"dive_{ticker}_{idx}", help=f"Quick view for {ticker}"):
                st.session_state.inline_deep_dive_ticker = ticker
                st.rerun()

            row_cols[2].markdown(f"<span style='background-color:{cat_color};color:white;padding:2px 6px;border-radius:4px;font-size:0.8em'>{p['category']}</span>", unsafe_allow_html=True)
            row_cols[3].write(f"{p['total_score']:.0f}")
            row_cols[4].markdown(f"<span style='background-color:{tier_color};color:white;padding:2px 6px;border-radius:4px'>{p['tier']}</span>", unsafe_allow_html=True)
            row_cols[5].write(f"${p['current_price']:.2f}")
            row_cols[6].markdown(f"<span style='color:{change_color}'>{change_1d:+.1f}%</span>", unsafe_allow_html=True)
            row_cols[7].write(f"{p['relative_volume']:.1f}x" if p.get('relative_volume') else '-')
            row_cols[8].write(f"{p['atr_percent']:.1f}%")
            row_cols[9].write(f"-{p['suggested_stop_pct']:.1f}%")
            row_cols[10].write(f"{p['risk_reward_ratio']:.1f}:1" if p.get('risk_reward_ratio') else '-')

            if has_claude:
                grade_colors = {'A+': '#28a745', 'A': '#28a745', 'B': '#17a2b8', 'C': '#ffc107', 'D': '#dc3545'}
                grade_color = grade_colors.get(ai_grade, '#666')
                row_cols[11].markdown(f"<span style='background-color:{grade_color};color:white;padding:2px 6px;border-radius:4px'>{ai_grade}</span>", unsafe_allow_html=True)
                row_cols[12].markdown(f"<span style='background-color:{action_color};color:white;padding:2px 6px;border-radius:4px;font-size:0.8em'>{ai_action}</span>", unsafe_allow_html=True)
            else:
                row_cols[11].write('-')
                row_cols[12].write('-')

            # Store for CSV export
            df_data.append({
                'Ticker': ticker,
                'Name': p.get('name', ''),
                'Category': p['category'],
                'Score': f"{p['total_score']:.0f}",
                'Tier': p['tier'],
                'Price': f"${p['current_price']:.2f}",
                '1D%': f"{change_1d:+.1f}%",
                'RVol': f"{p['relative_volume']:.1f}x" if p.get('relative_volume') else '-',
                'ATR%': f"{p['atr_percent']:.1f}%",
                'Stop%': f"-{p['suggested_stop_pct']:.1f}%",
                'R/R': f"{p['risk_reward_ratio']:.1f}:1" if p.get('risk_reward_ratio') else '-',
                'AI Grade': ai_grade if has_claude else '-',
                'AI Score': f"{ai_score}/10" if has_claude else '-',
                'AI Action': ai_action if has_claude else '-',
            })

        df = pd.DataFrame(df_data)

        # Export and Quick Deep Dive section
        st.markdown("---")
        col_export, col_dive_label, col_dive_select, col_dive_btn = st.columns([1, 0.8, 1.5, 0.8])

        with col_export:
            csv = df.to_csv(index=False)
            st.download_button(
                "Export CSV",
                csv,
                f"top_picks_{picks_data.get('date', 'unknown')}.csv",
                "text/csv"
            )

        with col_dive_label:
            st.markdown("**Quick Deep Dive:**")

        with col_dive_select:
            # Create options with ticker and name
            dive_options = [f"{p['ticker']} - {p.get('name', '')[:20]}" for p in all_picks_table]
            selected_dive = st.selectbox(
                "Select stock",
                dive_options,
                label_visibility="collapsed",
                key="quick_dive_select"
            )

        with col_dive_btn:
            if st.button("🔍 Open Deep Dive", key="quick_dive_btn", type="primary"):
                if selected_dive:
                    ticker_to_dive = selected_dive.split(" - ")[0]
                    st.session_state.inline_deep_dive_ticker = ticker_to_dive
                    st.rerun()

        # Inline Deep Dive section (renders below the table when a stock is selected)
        inline_ticker = st.session_state.get('inline_deep_dive_ticker')
        if inline_ticker:
            st.markdown("---")
            st.markdown(f"### 🔍 Quick Deep Dive: {inline_ticker}")

            col_close, col_spacer = st.columns([1, 5])
            with col_close:
                if st.button("✕ Close", key="close_inline_dive"):
                    st.session_state.inline_deep_dive_ticker = None
                    st.rerun()

            # Fetch and display stock data inline
            with st.spinner(f"Loading data for {inline_ticker}..."):
                data = service.get_stock_details(inline_ticker)
                candles = service.get_daily_candles(inline_ticker, days=90)

            if data and data.get('instrument'):
                inst = data['instrument']
                metrics = data.get('metrics', {}) or {}
                watchlist = data.get('watchlist', {}) or {}

                # Quick metrics row
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Price", f"${float(metrics.get('current_price', 0)):.2f}")
                col2.metric("1D Change", f"{float(metrics.get('price_change_1d', 0)):+.1f}%")
                col3.metric("RSI", f"{float(metrics.get('rsi_14', 0)):.1f}")
                col4.metric("ATR%", f"{float(metrics.get('atr_percent', 0)):.1f}%")
                col5.metric("Tier", watchlist.get('tier', '-'))

                # Chart
                if candles is not None and not candles.empty:
                    import plotly.graph_objects as go

                    # Use 'timestamp' column (from get_daily_candles) or 'date' as fallback
                    date_col = 'timestamp' if 'timestamp' in candles.columns else 'date'

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=candles[date_col],
                        open=candles['open'],
                        high=candles['high'],
                        low=candles['low'],
                        close=candles['close'],
                        name='Price'
                    ))
                    # Add SMAs if available
                    if 'sma_20' in candles.columns:
                        fig.add_trace(go.Scatter(x=candles[date_col], y=candles['sma_20'], name='SMA20', line=dict(color='orange', width=1)))
                    if 'sma_50' in candles.columns:
                        fig.add_trace(go.Scatter(x=candles[date_col], y=candles['sma_50'], name='SMA50', line=dict(color='blue', width=1)))

                    fig.update_layout(
                        title=f"{inline_ticker} - {inst.get('name', '')}",
                        xaxis_rangeslider_visible=False,
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Signals summary
                signals = []
                if watchlist.get('sma_cross_signal'):
                    signals.append(f"SMA: {watchlist['sma_cross_signal']}")
                if watchlist.get('rsi_signal'):
                    signals.append(f"RSI: {watchlist['rsi_signal']}")
                if watchlist.get('macd_cross_signal'):
                    signals.append(f"MACD: {watchlist['macd_cross_signal']}")
                if watchlist.get('trend_strength'):
                    signals.append(f"Trend: {watchlist['trend_strength']}")

                if signals:
                    st.markdown(f"**Signals:** {' | '.join(signals)}")

                # Link to full Deep Dive
                st.markdown("---")
                if st.button(f"📊 Open Full Deep Dive for {inline_ticker}", key="full_dive_from_inline"):
                    st.session_state.deep_dive_ticker = inline_ticker
                    st.session_state.inline_deep_dive_ticker = None
                    st.info(f"📌 **{inline_ticker}** selected - click the **🔍 Deep Dive** tab above for full analysis")
            else:
                st.warning(f"Could not load data for {inline_ticker}")

    # Stats section
    if picks_data.get('stats'):
        st.markdown("---")
        st.markdown("### Analysis Summary")
        stats = picks_data['stats']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates Analyzed", stats.get('candidates_analyzed', 0))
        col2.metric("Avg Momentum Score", stats.get('avg_score_momentum', 0))
        col3.metric("Avg Breakout Score", stats.get('avg_score_breakout', 0))
        col4.metric("Avg Bounce Score", stats.get('avg_score_reversion', 0))


def _render_picks_category(title: str, description: str, picks: List[Dict], color: str, service=None, claude_analysis: Dict = None):
    """Render a single category of picks as cards with optional Claude analysis."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%); padding: 0.8rem; border-radius: 10px; color: white; margin-bottom: 0.8rem;">
        <h4 style="margin: 0;">{title}</h4>
        <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.9;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

    if not picks:
        st.info("No picks in this category today")
        return

    claude_analysis = claude_analysis or {}

    for pick in picks:
        ticker = pick['ticker']
        change_color = "#28a745" if pick['price_change_1d'] >= 0 else "#dc3545"
        tier_colors = {1: '#28a745', 2: '#17a2b8', 3: '#ffc107'}
        tier_color = tier_colors.get(pick['tier'], '#6c757d')

        # Check if this pick has Claude analysis from:
        # 1. Database (stored in pick from query)
        # 2. Session state (from recent API call)
        has_claude_from_db = pick.get('claude_grade') is not None
        claude_data_from_session = claude_analysis.get(ticker, {})
        has_claude_from_session = claude_data_from_session.get('success', False)
        has_claude = has_claude_from_db or has_claude_from_session

        # Get Claude data if available
        grade, score, action, pos_rec, thesis, strengths, risks, analyzed_at = '', 0, '', '', '', [], [], None
        if has_claude:
            if has_claude_from_session:
                grade = claude_data_from_session.get('claude_grade', 'N/A')
                score = claude_data_from_session.get('claude_score', 0)
                action = claude_data_from_session.get('claude_action', 'N/A')
                pos_rec = claude_data_from_session.get('claude_position_rec', '')
                thesis = claude_data_from_session.get('claude_thesis', '')
                strengths = claude_data_from_session.get('claude_key_strengths', [])
                risks = claude_data_from_session.get('claude_key_risks', [])
                analyzed_at = None  # Session analysis is just done now
            else:
                grade = pick.get('claude_grade', 'N/A')
                score = pick.get('claude_score', 0)
                action = pick.get('claude_action', 'N/A')
                pos_rec = pick.get('claude_position_rec', '')
                thesis = pick.get('claude_thesis', '')
                strengths = pick.get('claude_key_strengths', []) or []
                risks = pick.get('claude_key_risks', []) or []
                analyzed_at = pick.get('claude_analyzed_at')

        # Render the main card (without Claude badges - those go in separate markdown)
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.7rem; border-radius: 8px; margin-bottom: 0.3rem; border-left: 4px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; font-size: 1.1rem;">{ticker}</span>
                <span style="background: {tier_color}; color: white; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.75rem;">T{pick['tier']}</span>
            </div>
            <div style="font-size: 0.85rem; color: #555; margin-top: 0.2rem;">
                ${pick['current_price']:.2f} <span style="color: {change_color};">({pick['price_change_1d']:+.1f}%)</span>
            </div>
            <div style="font-size: 0.8rem; margin-top: 0.3rem;">
                <span style="background: #e9ecef; padding: 0.1rem 0.3rem; border-radius: 3px;">Score: {pick['total_score']:.0f}</span>
                <span style="background: #e9ecef; padding: 0.1rem 0.3rem; border-radius: 3px; margin-left: 0.3rem;">Vol: {pick['relative_volume']:.1f}x</span>
            </div>
            <div style="font-size: 0.75rem; color: #666; margin-top: 0.3rem;">
                {pick.get('signals_summary', '')[:50]}
            </div>
            <div style="font-size: 0.75rem; color: #888; margin-top: 0.2rem;">
                Stop: -{pick['suggested_stop_pct']:.1f}% | R/R: {pick['risk_reward_ratio']:.1f}:1
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Render Claude badges in separate markdown call if has Claude
        if has_claude:
            grade_colors = {'A+': '#28a745', 'A': '#28a745', 'B': '#17a2b8', 'C': '#ffc107', 'D': '#dc3545'}
            action_colors = {'STRONG BUY': '#28a745', 'BUY': '#28a745', 'HOLD': '#ffc107', 'AVOID': '#dc3545'}
            pos_colors = {'Full': '#28a745', 'Half': '#17a2b8', 'Quarter': '#ffc107', 'Skip': '#dc3545'}

            grade_bg = grade_colors.get(grade, '#6c757d')
            action_bg = action_colors.get(action, '#6c757d')
            pos_bg = pos_colors.get(pos_rec, '#6c757d')

            st.markdown(f"""
            <div style="display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: -0.2rem; margin-bottom: 0.5rem; padding: 0.3rem; background: #f0f0f0; border-radius: 0 0 8px 8px;">
                <span style="background: {grade_bg}; color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; font-weight: bold;">AI: {grade}</span>
                <span style="background: {action_bg}; color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem;">{action}</span>
                <span style="background: #6c757d; color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem;">{score}/10</span>
                <span style="background: {pos_bg}; color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem;">{pos_rec}</span>
            </div>
            """, unsafe_allow_html=True)

        # Expandable section for detailed Claude analysis
        if has_claude and (thesis or strengths or risks):
            with st.expander(f"📊 View AI Analysis", expanded=False):
                # Show analysis timestamp with staleness warning
                if analyzed_at:
                    if isinstance(analyzed_at, datetime):
                        now = datetime.now(analyzed_at.tzinfo) if analyzed_at.tzinfo else datetime.now()
                        time_ago = now - analyzed_at
                        date_str = analyzed_at.strftime("%b %d, %Y %H:%M")
                        if time_ago.days > 0:
                            ago_str = f"{time_ago.days}d ago"
                        elif time_ago.seconds >= 3600:
                            ago_str = f"{time_ago.seconds // 3600}h ago"
                        else:
                            ago_str = f"{time_ago.seconds // 60}m ago"
                        if time_ago.days > 7:
                            st.warning(f"⚠️ Analysis is {time_ago.days} days old - consider refreshing")
                        st.caption(f"📅 {date_str} ({ago_str})")
                    else:
                        st.caption(f"📅 {str(analyzed_at)[:16]}")
                else:
                    st.caption("📅 Just analyzed")

                if thesis:
                    st.markdown(f"**💡 Investment Thesis:**")
                    st.info(thesis)

                col1, col2 = st.columns(2)
                with col1:
                    if strengths:
                        st.markdown("**✅ Key Strengths:**")
                        for s in strengths[:3]:
                            st.success(f"{s}")
                with col2:
                    if risks:
                        st.markdown("**⚠️ Key Risks:**")
                        for r in risks[:3]:
                            st.error(f"{r}")

        # Add individual analyze button if service is provided and no analysis yet
        if service and not has_claude:
            if st.button(f"Analyze {ticker}", key=f"analyze_{ticker}", type="secondary"):
                with st.spinner(f"Analyzing {ticker}..."):
                    result = service.analyze_top_pick_with_claude(pick)
                    st.session_state.claude_top_picks_analysis[ticker] = result
                    # Clear cache to reload picks with fresh Claude data
                    service.get_daily_top_picks.clear()
                st.rerun()
                return  # Stop execution after rerun


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

    with col1:
        scanner_filter = st.selectbox(
            "Scanner",
            ["All Scanners", "trend_momentum", "breakout_confirmation", "mean_reversion", "gap_and_go",
             "zlma_trend", "smc_ema_trend", "ema_crossover", "macd_momentum"]
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
        # Default to last 7 days
        default_from = datetime.now().date() - timedelta(days=7)
        date_from = st.date_input(
            "Signal Date From",
            value=default_from,
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

    # Display signals
    st.markdown(f"### Showing {len(signals)} Signals")

    # Signal cards layout
    for i, signal in enumerate(signals):
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
    }
    scanner_icon = scanner_icons.get(scanner, '📊')

    # Claude action colors
    claude_action_colors = {
        'STRONG BUY': 'green',
        'BUY': 'blue',
        'HOLD': 'orange',
        'AVOID': 'red'
    }

    # Build title with Claude info if available
    claude_badge = ""
    if has_claude:
        action_color = claude_action_colors.get(claude_action, 'gray')
        claude_badge = f" | 🤖 :{action_color}[{claude_action}] ({claude_grade})"

    # Build header with timestamp
    timestamp_part = f" | 📅 {signal_time_str}" if signal_time_str else ""

    # Use expander for each signal card (collapsed by default for easier browsing)
    with st.expander(f"**{ticker}** | :{tier_color}[{tier}] | Score: {score} | {scanner_icon} {scanner}{claude_badge}{timestamp_part}", expanded=False):

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


def _signals_to_csv(signals: List[Dict]) -> str:
    """Convert signals to CSV format including Claude analysis."""
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    # Header with Claude columns
    writer.writerow([
        'Symbol', 'Side', 'Entry', 'Stop', 'TP1', 'TP2',
        'Risk%', 'R:R', 'Score', 'Tier', 'Scanner', 'Setup',
        'Claude_Grade', 'Claude_Score', 'Claude_Action', 'Claude_Conviction', 'Claude_Thesis'
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
            (s.get('claude_thesis', '') or '')[:100]
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

    # Scanner selector
    scanner_names = [s['scanner_name'] for s in by_scanner]
    selected_scanner = st.selectbox(
        "Select Scanner to Analyze",
        scanner_names,
        format_func=lambda x: x.replace('_', ' ').title()
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

        # Get signals for this scanner
        signals = service.get_scanner_signals(
            scanner_name=selected_scanner,
            status=None,  # All statuses
            limit=100
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

            # Recent signals table
            st.subheader(f"Recent {selected_scanner.replace('_', ' ').title()} Signals")

            # Prepare data for table
            table_data = []
            for s in signals[:20]:  # Show last 20
                table_data.append({
                    'Ticker': s.get('ticker', ''),
                    'Type': s.get('signal_type', ''),
                    'Entry': f"${float(s.get('entry_price', 0)):.2f}",
                    'Score': s.get('composite_score', 0),
                    'Tier': s.get('quality_tier', ''),
                    'Status': s.get('status', ''),
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

    scanner_info = {
        'trend_momentum': {
            'description': 'Finds pullback entry opportunities in established uptrends.',
            'best_for': 'Trending markets, continuation plays',
            'criteria': 'Above SMA20/50, RSI 35-65, MACD bullish'
        },
        'breakout_confirmation': {
            'description': 'Identifies volume-confirmed breakouts above resistance.',
            'best_for': 'Range breakouts, momentum starts',
            'criteria': 'Price break + volume surge + trend alignment'
        },
        'mean_reversion': {
            'description': 'Spots oversold bounces with reversal patterns.',
            'best_for': 'Counter-trend entries, support bounces',
            'criteria': 'RSI oversold, bullish divergence, support test'
        },
        'gap_and_go': {
            'description': 'Gap continuation plays with momentum confirmation.',
            'best_for': 'Opening momentum, news-driven moves',
            'criteria': 'Gap up + volume + trend continuation'
        },
        'zlma_trend': {
            'description': 'Zero-Lag Moving Average crossover with EMA confirmation.',
            'best_for': 'Trend-following with reduced lag',
            'criteria': 'ZLMA/EMA crossover, ATR-based stops'
        },
        'smc_ema_trend': {
            'description': 'SMC-style EMA trend following with swing structure analysis.',
            'best_for': 'Institutional order flow alignment',
            'criteria': 'EMA stack, swing structure, volume profile'
        },
        'ema_crossover': {
            'description': 'EMA cascade crossover with multi-timeframe trend alignment.',
            'best_for': 'Clear trend transitions',
            'criteria': 'EMA 9/21/50 cascade, trend alignment'
        },
        'macd_momentum': {
            'description': 'MACD momentum confluence with price structure.',
            'best_for': 'Momentum confirmation trades',
            'criteria': 'MACD crossover, histogram expansion, price structure'
        }
    }

    info = scanner_info.get(selected_scanner, {})
    if info:
        st.markdown(f"**Description:** {info.get('description', 'N/A')}")
        st.markdown(f"**Best For:** {info.get('best_for', 'N/A')}")
        st.markdown(f"**Entry Criteria:** {info.get('criteria', 'N/A')}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    service = get_stock_service()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📋 Watchlist", "📡 All Signals",
        "📈 Scanner Analysis", "🎯 Top Picks", "🔍 Deep Dive"
    ])

    with tab1:
        render_overview_tab(service)

    with tab2:
        render_watchlist_tab(service)

    with tab3:
        # Consolidated signals tab - shows all scanner signals including ZLMA
        render_scanner_signals_tab(service)

    with tab4:
        # Individual scanner drill-down and analysis
        render_scanner_analysis_tab(service)

    with tab5:
        render_top_picks_tab(service)

    with tab6:
        render_deep_dive_tab(service)


if __name__ == "__main__":
    main()
