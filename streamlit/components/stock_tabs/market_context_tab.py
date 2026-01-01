"""
Market Context Tab

Provides market-level context for stock selection:
- Market Regime indicator (Bull/Bear classification)
- Sector rotation heat map
- Breadth indicators
- RS leaders by sector
- Strategy recommendations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Any


def render_market_context_tab(service: Any) -> None:
    """Render the Market Context tab with regime and sector analysis."""

    # Get market data
    regime_data = service.get_market_regime()
    sector_data = service.get_sector_analysis()

    # Market Regime Header
    st.markdown("### Market Regime")

    if regime_data:
        _render_regime_indicator(regime_data)
        _render_breadth_indicators(regime_data)
    else:
        st.info("Market regime data not yet calculated. Run the RS calculator to populate this data.")

    st.divider()

    # Sector Heat Map
    st.markdown("### Sector Rotation Heat Map")

    if not sector_data.empty:
        _render_sector_heatmap(sector_data)
        _render_sector_details(sector_data, service)
    else:
        st.info("Sector analysis data not yet available. Run the sector analyzer to populate this data.")

    st.divider()

    # RS Leaders
    st.markdown("### Relative Strength Leaders")
    _render_rs_leaders(service)


def _render_regime_indicator(data: dict) -> None:
    """Render the market regime indicator with large visual."""
    regime = data.get('market_regime', 'unknown')
    spy_price = data.get('spy_price', 0)
    spy_vs_200 = data.get('spy_vs_sma200_pct', 0)

    # Regime colors and icons
    regime_config = {
        'bull_confirmed': {'color': '#00C853', 'icon': '🟢', 'label': 'BULL CONFIRMED', 'bg': 'rgba(0, 200, 83, 0.1)'},
        'bull_weakening': {'color': '#FFB300', 'icon': '🟡', 'label': 'BULL WEAKENING', 'bg': 'rgba(255, 179, 0, 0.1)'},
        'bear_weakening': {'color': '#FF9800', 'icon': '🟠', 'label': 'BEAR WEAKENING', 'bg': 'rgba(255, 152, 0, 0.1)'},
        'bear_confirmed': {'color': '#F44336', 'icon': '🔴', 'label': 'BEAR CONFIRMED', 'bg': 'rgba(244, 67, 54, 0.1)'},
    }

    config = regime_config.get(regime, {'color': '#9E9E9E', 'icon': '⚪', 'label': 'UNKNOWN', 'bg': 'rgba(158, 158, 158, 0.1)'})

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="background: {config['bg']}; border: 2px solid {config['color']}; border-radius: 10px; padding: 20px; text-align: center;">
            <div style="font-size: 48px;">{config['icon']}</div>
            <div style="font-size: 24px; font-weight: bold; color: {config['color']};">{config['label']}</div>
            <div style="font-size: 14px; color: #888; margin-top: 5px;">
                SPY: ${spy_price:.2f} ({spy_vs_200:+.1f}% vs SMA200)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric(
            "SPY vs SMA50",
            f"{data.get('spy_vs_sma50_pct', 0):+.1f}%",
            delta=None
        )

    with col3:
        st.metric(
            "SPY Trend",
            data.get('spy_trend', 'N/A').title(),
            delta=None
        )

    with col4:
        vol_regime = data.get('volatility_regime', 'normal')
        vol_color = {'low': 'green', 'normal': 'blue', 'high': 'orange', 'extreme': 'red'}.get(vol_regime, 'gray')
        st.metric(
            "Volatility",
            vol_regime.title(),
            delta=f"ATR: {data.get('avg_atr_pct', 0):.1f}%"
        )

    # Strategy Recommendations
    strategies = data.get('recommended_strategies', {})
    if strategies:
        st.markdown("#### Strategy Weights")
        cols = st.columns(4)
        for i, (strategy, weight) in enumerate(strategies.items()):
            with cols[i % 4]:
                color = '#00C853' if weight >= 0.6 else '#FFB300' if weight >= 0.4 else '#F44336'
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: rgba(128,128,128,0.1); border-radius: 5px;">
                    <div style="font-size: 12px; color: #888;">{strategy.replace('_', ' ').title()}</div>
                    <div style="font-size: 20px; font-weight: bold; color: {color};">{weight*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)


def _render_breadth_indicators(data: dict) -> None:
    """Render market breadth gauges."""
    st.markdown("#### Market Breadth")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pct_200 = data.get('pct_above_sma200', 0) or 0
        color = '#00C853' if pct_200 > 60 else '#FFB300' if pct_200 > 40 else '#F44336'
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: rgba(128,128,128,0.1); border-radius: 8px;">
            <div style="font-size: 12px; color: #888;">% Above SMA200</div>
            <div style="font-size: 28px; font-weight: bold; color: {color};">{pct_200:.0f}%</div>
            <div style="font-size: 10px; color: #666;">Market Health</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pct_50 = data.get('pct_above_sma50', 0) or 0
        color = '#00C853' if pct_50 > 60 else '#FFB300' if pct_50 > 40 else '#F44336'
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: rgba(128,128,128,0.1); border-radius: 8px;">
            <div style="font-size: 12px; color: #888;">% Above SMA50</div>
            <div style="font-size: 28px; font-weight: bold; color: {color};">{pct_50:.0f}%</div>
            <div style="font-size: 10px; color: #666;">Intermediate Trend</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        hl_ratio = data.get('high_low_ratio', 1) or 1
        color = '#00C853' if hl_ratio > 2 else '#FFB300' if hl_ratio > 1 else '#F44336'
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: rgba(128,128,128,0.1); border-radius: 8px;">
            <div style="font-size: 12px; color: #888;">New Highs/Lows</div>
            <div style="font-size: 28px; font-weight: bold; color: {color};">{hl_ratio:.1f}x</div>
            <div style="font-size: 10px; color: #666;">{data.get('new_highs_count', 0)} / {data.get('new_lows_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        ad_ratio = data.get('ad_ratio', 1) or 1
        color = '#00C853' if ad_ratio > 1.5 else '#FFB300' if ad_ratio > 0.8 else '#F44336'
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: rgba(128,128,128,0.1); border-radius: 8px;">
            <div style="font-size: 12px; color: #888;">Advance/Decline</div>
            <div style="font-size: 28px; font-weight: bold; color: {color};">{ad_ratio:.2f}</div>
            <div style="font-size: 10px; color: #666;">{data.get('advancing_count', 0)} / {data.get('declining_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_sector_heatmap(df: pd.DataFrame) -> None:
    """Render sector heat map sorted by RS."""
    if df.empty:
        return

    # Create horizontal bar chart for sector RS
    fig = go.Figure()

    # Sort by RS
    df_sorted = df.sort_values('rs_vs_spy', ascending=True)

    # Color based on RS value
    colors = []
    for rs in df_sorted['rs_vs_spy']:
        if rs is None:
            colors.append('#9E9E9E')
        elif rs > 1.1:
            colors.append('#00C853')
        elif rs > 1.0:
            colors.append('#4CAF50')
        elif rs > 0.9:
            colors.append('#FFB300')
        else:
            colors.append('#F44336')

    fig.add_trace(go.Bar(
        x=df_sorted['rs_vs_spy'],
        y=df_sorted['sector'],
        orientation='h',
        marker_color=colors,
        text=[f"{rs:.2f}" if rs else "N/A" for rs in df_sorted['rs_vs_spy']],
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>RS vs SPY: %{x:.2f}<br>Stage: %{customdata}<extra></extra>",
        customdata=df_sorted['sector_stage']
    ))

    # Add vertical line at RS = 1.0
    fig.add_vline(x=1.0, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        title="Sector Relative Strength vs SPY",
        xaxis_title="RS vs SPY (>1.0 = Outperforming)",
        yaxis_title="",
        height=400,
        margin=dict(l=150, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_sector_details(df: pd.DataFrame, service: Any) -> None:
    """Render detailed sector table with expandable rows."""
    if df.empty:
        return

    # Create styled table
    st.markdown("#### Sector Details")

    # Display columns
    cols = ['sector', 'sector_etf', 'rs_vs_spy', 'rs_percentile', 'rs_trend', 'sector_return_20d', 'sector_stage', 'stocks_in_sector']

    display_df = df[cols].copy()
    display_df.columns = ['Sector', 'ETF', 'RS vs SPY', 'RS %ile', 'Trend', '20D Return', 'Stage', 'Stocks']

    # Format numeric columns
    display_df['RS vs SPY'] = display_df['RS vs SPY'].apply(lambda x: f"{x:.2f}" if x else "N/A")
    display_df['20D Return'] = display_df['20D Return'].apply(lambda x: f"{x:+.1f}%" if x else "N/A")

    # Style the dataframe
    def style_rs_trend(val):
        if val == 'improving':
            return 'color: #00C853'
        elif val == 'deteriorating':
            return 'color: #F44336'
        return ''

    def style_stage(val):
        stages = {
            'leading': 'color: #00C853; font-weight: bold',
            'weakening': 'color: #FFB300',
            'lagging': 'color: #F44336',
            'improving': 'color: #2196F3'
        }
        return stages.get(val, '')

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Show top stocks for selected sector
    st.markdown("#### Top Stocks by Sector")
    selected_sector = st.selectbox("Select Sector", df['sector'].tolist(), key="sector_select")

    if selected_sector:
        sector_row = df[df['sector'] == selected_sector].iloc[0]
        top_stocks = sector_row.get('top_stocks', [])

        if top_stocks:
            st.markdown(f"**Top RS Stocks in {selected_sector}:**")
            cols = st.columns(min(len(top_stocks), 5))
            for i, stock in enumerate(top_stocks[:5]):
                with cols[i]:
                    ticker = stock.get('ticker', '')
                    rs_pct = stock.get('rs_percentile', 0)
                    color = '#00C853' if rs_pct >= 80 else '#4CAF50' if rs_pct >= 60 else '#FFB300'
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background: rgba(128,128,128,0.1); border-radius: 5px; border-left: 3px solid {color};">
                        <div style="font-size: 16px; font-weight: bold;">{ticker}</div>
                        <div style="font-size: 12px; color: {color};">RS: {rs_pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)


def _render_rs_leaders(service: Any) -> None:
    """Render RS leaders table with filtering."""
    col1, col2 = st.columns([1, 3])

    with col1:
        min_rs = st.slider("Min RS Percentile", 50, 95, 80, 5, key="rs_min_slider")

    rs_leaders = service.get_rs_leaders(min_rs_percentile=min_rs, limit=30)

    if rs_leaders.empty:
        st.info("No stocks found meeting the RS criteria. Try lowering the minimum percentile.")
        return

    # Display table
    display_cols = ['ticker', 'name', 'sector', 'current_price', 'rs_percentile', 'rs_trend', 'price_change_20d', 'trend_strength', 'pct_from_52w_high']
    display_df = rs_leaders[display_cols].copy()
    display_df.columns = ['Ticker', 'Name', 'Sector', 'Price', 'RS %ile', 'RS Trend', '20D Chg', 'Trend', '% from 52W High']

    # Format
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if x else "N/A")
    display_df['20D Chg'] = display_df['20D Chg'].apply(lambda x: f"{x:+.1f}%" if x else "N/A")
    display_df['% from 52W High'] = display_df['% from 52W High'].apply(lambda x: f"{x:.1f}%" if x else "N/A")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Summary stats
    st.markdown(f"**Found {len(rs_leaders)} stocks with RS >= {min_rs}%ile**")
