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

    # Header stats
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Picks", picks_data['total_picks'])
    col2.metric("Momentum", len(picks_data.get('momentum', [])))
    col3.metric("Breakout", len(picks_data.get('breakout', [])))
    col4.metric("Bounce Plays", len(picks_data.get('mean_reversion', [])))
    col5.metric("Data Date", picks_data.get('date', 'N/A'))

    st.markdown("---")

    # Render each category
    col1, col2, col3 = st.columns(3)

    with col1:
        _render_picks_category(
            "Momentum Riders",
            "Trending stocks with bullish confirmation signals",
            picks_data.get('momentum', []),
            "#1a5f7a"
        )

    with col2:
        _render_picks_category(
            "Breakout Watch",
            "Stocks near 52W highs with volume surge",
            picks_data.get('breakout', []),
            "#28a745"
        )

    with col3:
        _render_picks_category(
            "Bounce Plays",
            "Oversold stocks showing reversal patterns",
            picks_data.get('mean_reversion', []),
            "#6f42c1"
        )

    # Detailed table view
    st.markdown("---")
    st.markdown("### All Picks - Detailed View")

    all_picks = (
        picks_data.get('momentum', []) +
        picks_data.get('breakout', []) +
        picks_data.get('mean_reversion', [])
    )

    if all_picks:
        # Sort by total_score descending
        all_picks.sort(key=lambda x: x['total_score'], reverse=True)

        df_data = []
        for p in all_picks:
            df_data.append({
                'Ticker': p['ticker'],
                'Name': p.get('name', '')[:20] + '...' if len(p.get('name', '')) > 20 else p.get('name', ''),
                'Category': p['category'],
                'Score': f"{p['total_score']:.0f}",
                'Tier': p['tier'],
                'Price': f"${p['current_price']:.2f}",
                '1D%': f"{p['price_change_1d']:+.1f}%" if p['price_change_1d'] else '-',
                'RVol': f"{p['relative_volume']:.1f}x" if p['relative_volume'] else '-',
                'ATR%': f"{p['atr_percent']:.1f}%",
                'Signals': p.get('signals_summary', '-')[:30],
                'Stop%': f"-{p['suggested_stop_pct']:.1f}%",
                'R/R': f"{p['risk_reward_ratio']:.1f}:1" if p['risk_reward_ratio'] else '-'
            })

        df = pd.DataFrame(df_data)

        def style_category(val):
            colors = {
                'Momentum': '#1a5f7a',
                'Breakout': '#28a745',
                'Mean Reversion': '#6f42c1'
            }
            return f'background-color: {colors.get(val, "#fff")}; color: white; font-weight: bold;'

        def style_tier(val):
            colors = {1: '#28a745', 2: '#17a2b8', 3: '#ffc107'}
            text_colors = {1: 'white', 2: 'white', 3: 'black'}
            return f'background-color: {colors.get(val, "#fff")}; color: {text_colors.get(val, "black")}; font-weight: bold;'

        def style_change(val):
            if '+' in str(val):
                return 'color: #28a745;'
            elif '-' in str(val) and val != '-':
                return 'color: #dc3545;'
            return ''

        styled_df = df.style.map(style_category, subset=['Category']).map(style_tier, subset=['Tier']).map(style_change, subset=['1D%'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            "Export Top Picks CSV",
            csv,
            f"top_picks_{picks_data.get('date', 'unknown')}.csv",
            "text/csv"
        )

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


def _render_picks_category(title: str, description: str, picks: List[Dict], color: str):
    """Render a single category of picks as cards."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%); padding: 0.8rem; border-radius: 10px; color: white; margin-bottom: 0.8rem;">
        <h4 style="margin: 0;">{title}</h4>
        <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.9;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

    if not picks:
        st.info("No picks in this category today")
        return

    for pick in picks:
        change_color = "#28a745" if pick['price_change_1d'] >= 0 else "#dc3545"
        tier_colors = {1: '#28a745', 2: '#17a2b8', 3: '#ffc107'}
        tier_color = tier_colors.get(pick['tier'], '#6c757d')

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.7rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; font-size: 1.1rem;">{pick['ticker']}</span>
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
                {pick.get('signals_summary', '')[:40]}
            </div>
            <div style="font-size: 0.75rem; color: #888; margin-top: 0.2rem;">
                Stop: -{pick['suggested_stop_pct']:.1f}% | R/R: {pick['risk_reward_ratio']:.1f}:1
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SCANNER SIGNALS TAB
# =============================================================================

def render_scanner_signals_tab(service):
    """Render the Scanner Signals tab - automated signal scanner results."""
    st.markdown("""
    <div class="main-header">
        <h2>Signal Scanners</h2>
        <p>Automated trading signals with entry, stop-loss, and take-profit levels + Claude AI Analysis</p>
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
            ["All Scanners", "trend_momentum", "breakout_confirmation", "mean_reversion", "gap_and_go"]
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

    # Signal timestamp
    signal_timestamp = signal.get('signal_timestamp')
    if signal_timestamp:
        if isinstance(signal_timestamp, datetime):
            signal_time_str = signal_timestamp.strftime('%Y-%m-%d %H:%M')
        else:
            signal_time_str = str(signal_timestamp)[:16]
    else:
        signal_time_str = 'Unknown'

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

    # Format claude_analyzed_at timestamp
    if claude_analyzed_at:
        if isinstance(claude_analyzed_at, datetime):
            analyzed_time_str = claude_analyzed_at.strftime('%Y-%m-%d %H:%M')
        else:
            analyzed_time_str = str(claude_analyzed_at)[:16]
    else:
        analyzed_time_str = None

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
        'Gap And Go': '⚡'
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

    # Use expander for each signal card
    with st.expander(f"**{ticker}** | :{tier_color}[{tier}] | Score: {score} | {scanner_icon} {scanner}{claude_badge}", expanded=True):
        # Signal detected timestamp
        st.caption(f"📅 **Signal Detected:** {signal_time_str}")

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
                    st.caption(f"✅ Analyzed: {analyzed_time_str}")
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
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    service = get_stock_service()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📋 Watchlist", "📡 Signals",
        "🎯 Top Picks", "🔎 Scanners", "🔍 Deep Dive"
    ])

    with tab1:
        render_overview_tab(service)

    with tab2:
        render_watchlist_tab(service)

    with tab3:
        render_signals_tab(service)

    with tab4:
        render_top_picks_tab(service)

    with tab5:
        render_scanner_signals_tab(service)

    with tab6:
        render_deep_dive_tab(service)


if __name__ == "__main__":
    main()
