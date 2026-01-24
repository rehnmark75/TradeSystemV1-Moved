"""
TradingView-Style Technical Indicator Gauges

Renders oscillator and moving average summary gauges matching TradingView's indicator display.
Shows Buy/Sell/Neutral signal counts with visual gauges and expandable detail tables.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional


def render_tv_summary_section(ticker_data: Dict[str, Any]) -> None:
    """
    Render TradingView-style technical summary section.

    Shows three gauges:
    1. Oscillators (11 indicators)
    2. Moving Averages (14 indicators)
    3. Overall Summary (combined)

    Args:
        ticker_data: Dictionary containing TV indicator summary counts and values
    """
    st.markdown("---")
    st.markdown("### Technical Summary")

    # Three-column layout for gauges
    col1, col2, col3 = st.columns(3)

    with col1:
        render_gauge(
            title="Oscillators",
            buy=ticker_data.get('tv_osc_buy', 0),
            sell=ticker_data.get('tv_osc_sell', 0),
            neutral=ticker_data.get('tv_osc_neutral', 0)
        )

    with col2:
        render_gauge(
            title="Moving Averages",
            buy=ticker_data.get('tv_ma_buy', 0),
            sell=ticker_data.get('tv_ma_sell', 0),
            neutral=ticker_data.get('tv_ma_neutral', 0)
        )

    with col3:
        render_overall_gauge(
            signal=ticker_data.get('tv_overall_signal', 'NEUTRAL'),
            score=ticker_data.get('tv_overall_score', 0.0)
        )

    # Expandable detail tables
    with st.expander("📊 Oscillator Details"):
        render_oscillator_details(ticker_data)

    with st.expander("📈 Moving Average Details"):
        render_ma_details(ticker_data)


def render_gauge(title: str, buy: int, sell: int, neutral: int) -> None:
    """
    Render a semicircular gauge showing Buy/Sell/Neutral distribution.

    Args:
        title: Gauge title (e.g., "Oscillators")
        buy: Number of BUY signals
        sell: Number of SELL signals
        neutral: Number of NEUTRAL signals
    """
    total = buy + sell + neutral

    if total == 0:
        st.markdown(f"**{title}**")
        st.info("No data available")
        return

    # Calculate score (-100 to +100)
    score = ((buy - sell) / total * 100) if total > 0 else 0

    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 14, 'color': '#333'}},
        number={'suffix': "", 'font': {'size': 0}},  # Hide number, show in summary
        gauge={
            'axis': {
                'range': [-100, 100],
                'tickwidth': 1,
                'tickcolor': "#999",
                'tickfont': {'size': 10}
            },
            'bar': {'color': "#1f77b4", 'thickness': 0.3},
            'steps': [
                {'range': [-100, -60], 'color': "#dc3545"},      # Strong Sell (red)
                {'range': [-60, -20], 'color': "#f8d7da"},       # Sell (light red)
                {'range': [-20, 20], 'color': "#ffc107"},        # Neutral (yellow)
                {'range': [20, 60], 'color': "#d4edda"},         # Buy (light green)
                {'range': [60, 100], 'color': "#28a745"},        # Strong Buy (green)
            ],
            'threshold': {
                'line': {'color': "#000", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=180,
        margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary counts below gauge
    st.markdown(f"""
        <div style="text-align:center; font-size:0.85rem; margin-top:-10px;">
            <span style="color:#dc3545; font-weight:500;">Sell: {sell}</span> |
            <span style="color:#6c757d; font-weight:500;">Neutral: {neutral}</span> |
            <span style="color:#28a745; font-weight:500;">Buy: {buy}</span>
        </div>
    """, unsafe_allow_html=True)


def render_overall_gauge(signal: str, score: float) -> None:
    """
    Render the overall summary gauge.

    Args:
        signal: Overall signal (STRONG BUY, BUY, NEUTRAL, SELL, STRONG SELL)
        score: Overall score (-100 to +100)
    """
    # Color mapping
    signal_colors = {
        'STRONG BUY': '#28a745',
        'BUY': '#d4edda',
        'NEUTRAL': '#ffc107',
        'SELL': '#f8d7da',
        'STRONG SELL': '#dc3545'
    }

    signal_text_colors = {
        'STRONG BUY': '#ffffff',
        'BUY': '#155724',
        'NEUTRAL': '#856404',
        'SELL': '#721c24',
        'STRONG SELL': '#ffffff'
    }

    bg_color = signal_colors.get(signal, '#ffc107')
    text_color = signal_text_colors.get(signal, '#000')

    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Overall", 'font': {'size': 14, 'color': '#333'}},
        number={'suffix': "", 'font': {'size': 0}},
        gauge={
            'axis': {
                'range': [-100, 100],
                'tickwidth': 1,
                'tickcolor': "#999",
                'tickfont': {'size': 10}
            },
            'bar': {'color': "#1f77b4", 'thickness': 0.3},
            'steps': [
                {'range': [-100, -60], 'color': "#dc3545"},
                {'range': [-60, -20], 'color': "#f8d7da"},
                {'range': [-20, 20], 'color': "#ffc107"},
                {'range': [20, 60], 'color': "#d4edda"},
                {'range': [60, 100], 'color': "#28a745"},
            ],
            'threshold': {
                'line': {'color': "#000", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=180,
        margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Signal badge
    st.markdown(f"""
        <div style="text-align:center; margin-top:-10px;">
            <div style="
                background-color:{bg_color};
                color:{text_color};
                padding:4px 12px;
                border-radius:12px;
                font-size:0.85rem;
                font-weight:600;
                display:inline-block;
            ">{signal}</div>
            <div style="color:#6c757d; font-size:0.75rem; margin-top:4px;">Score: {score:.1f}</div>
        </div>
    """, unsafe_allow_html=True)


def render_oscillator_details(ticker_data: Dict[str, Any]) -> None:
    """Render expandable table showing individual oscillator values and signals."""

    # Build oscillator data
    oscillators = [
        {
            'name': 'RSI (14)',
            'value': ticker_data.get('rsi_14'),
            'signal': _classify_rsi(ticker_data.get('rsi_14'))
        },
        {
            'name': 'Stochastic %K (14,3,3)',
            'value': ticker_data.get('stoch_k'),
            'signal': _classify_stochastic(ticker_data.get('stoch_k'))
        },
        {
            'name': 'CCI (20)',
            'value': ticker_data.get('cci_20'),
            'signal': _classify_cci(ticker_data.get('cci_20'))
        },
        {
            'name': 'ADX (14)',
            'value': ticker_data.get('adx_14'),
            'signal': _classify_adx(
                ticker_data.get('adx_14'),
                ticker_data.get('plus_di'),
                ticker_data.get('minus_di')
            )
        },
        {
            'name': 'Awesome Oscillator',
            'value': ticker_data.get('ao_value'),
            'signal': _classify_ao(ticker_data.get('ao_value'))
        },
        {
            'name': 'Momentum (10)',
            'value': ticker_data.get('momentum_10'),
            'signal': _classify_momentum(ticker_data.get('momentum_10'))
        },
        {
            'name': 'MACD Level (12,26)',
            'value': ticker_data.get('macd'),
            'signal': _classify_macd(
                ticker_data.get('macd'),
                ticker_data.get('macd_signal')
            )
        },
        {
            'name': 'Stochastic RSI (3,3,14,14)',
            'value': ticker_data.get('stoch_rsi_k'),
            'signal': _classify_stochastic(ticker_data.get('stoch_rsi_k'))
        },
        {
            'name': 'Williams %R (14)',
            'value': ticker_data.get('williams_r'),
            'signal': _classify_williams(ticker_data.get('williams_r'))
        },
        {
            'name': 'Bull Bear Power',
            'value': ticker_data.get('bull_power'),
            'signal': _classify_bbp(
                ticker_data.get('bull_power'),
                ticker_data.get('bear_power')
            )
        },
        {
            'name': 'Ultimate Oscillator (7,14,28)',
            'value': ticker_data.get('ultimate_osc'),
            'signal': _classify_uo(ticker_data.get('ultimate_osc'))
        },
    ]

    # Render table
    for osc in oscillators:
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"**{osc['name']}**")

        with col2:
            value = osc['value']
            if value is not None:
                st.text(f"{value:.2f}")
            else:
                st.text("N/A")

        with col3:
            signal = osc['signal']
            signal_colors = {'BUY': '#28a745', 'SELL': '#dc3545', 'NEUTRAL': '#6c757d'}
            color = signal_colors.get(signal, '#6c757d')
            st.markdown(f'<span style="color:{color}; font-weight:600;">{signal}</span>',
                       unsafe_allow_html=True)


def render_ma_details(ticker_data: Dict[str, Any]) -> None:
    """Render expandable table showing individual MA values and signals."""

    price = ticker_data.get('price') or ticker_data.get('current_price', 0)

    # Build MA data
    mas = [
        ('EMA (10)', ticker_data.get('ema_10')),
        ('SMA (10)', ticker_data.get('sma_10')),
        ('EMA (20)', ticker_data.get('ema_20')),
        ('SMA (20)', ticker_data.get('sma_20')),
        ('EMA (30)', ticker_data.get('ema_30')),
        ('SMA (30)', ticker_data.get('sma_30')),
        ('EMA (50)', ticker_data.get('ema_50')),
        ('SMA (50)', ticker_data.get('sma_50')),
        ('EMA (100)', ticker_data.get('ema_100')),
        ('SMA (100)', ticker_data.get('sma_100')),
        ('EMA (200)', ticker_data.get('ema_200')),
        ('SMA (200)', ticker_data.get('sma_200')),
        ('Ichimoku Base (26)', ticker_data.get('ichimoku_base')),
        ('VWMA (20)', ticker_data.get('vwma_20')),
    ]

    # Render table
    for name, value in mas:
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"**{name}**")

        with col2:
            if value is not None:
                st.text(f"{value:.2f}")
            else:
                st.text("N/A")

        with col3:
            signal = _classify_ma(price, value)
            signal_colors = {'BUY': '#28a745', 'SELL': '#dc3545', 'NEUTRAL': '#6c757d'}
            color = signal_colors.get(signal, '#6c757d')
            st.markdown(f'<span style="color:{color}; font-weight:600;">{signal}</span>',
                       unsafe_allow_html=True)


# ===========================
# SIGNAL CLASSIFICATION HELPERS
# ===========================

def _classify_rsi(rsi: Optional[float]) -> str:
    """RSI: <30 Buy, >70 Sell."""
    if rsi is None:
        return 'NEUTRAL'
    if rsi < 30:
        return 'BUY'
    elif rsi > 70:
        return 'SELL'
    return 'NEUTRAL'


def _classify_stochastic(stoch: Optional[float]) -> str:
    """Stochastic: <20 Buy, >80 Sell."""
    if stoch is None:
        return 'NEUTRAL'
    if stoch < 20:
        return 'BUY'
    elif stoch > 80:
        return 'SELL'
    return 'NEUTRAL'


def _classify_cci(cci: Optional[float]) -> str:
    """CCI: <-100 Buy, >100 Sell."""
    if cci is None:
        return 'NEUTRAL'
    if cci < -100:
        return 'BUY'
    elif cci > 100:
        return 'SELL'
    return 'NEUTRAL'


def _classify_adx(adx: Optional[float], plus_di: Optional[float], minus_di: Optional[float]) -> str:
    """ADX: +DI > -DI with strong ADX = Buy."""
    if adx is None or plus_di is None or minus_di is None:
        return 'NEUTRAL'
    if adx < 25:
        return 'NEUTRAL'
    if plus_di > minus_di:
        return 'BUY'
    elif minus_di > plus_di:
        return 'SELL'
    return 'NEUTRAL'


def _classify_ao(ao: Optional[float]) -> str:
    """AO: >0 Buy, <0 Sell."""
    if ao is None:
        return 'NEUTRAL'
    if ao > 0:
        return 'BUY'
    elif ao < 0:
        return 'SELL'
    return 'NEUTRAL'


def _classify_momentum(momentum: Optional[float]) -> str:
    """Momentum: >0 Buy, <0 Sell."""
    if momentum is None:
        return 'NEUTRAL'
    if momentum > 0:
        return 'BUY'
    elif momentum < 0:
        return 'SELL'
    return 'NEUTRAL'


def _classify_macd(macd: Optional[float], macd_signal: Optional[float]) -> str:
    """MACD: MACD > Signal = Buy."""
    if macd is None or macd_signal is None:
        return 'NEUTRAL'
    if macd > macd_signal:
        return 'BUY'
    elif macd < macd_signal:
        return 'SELL'
    return 'NEUTRAL'


def _classify_williams(williams: Optional[float]) -> str:
    """Williams %R: <-80 Buy, >-20 Sell."""
    if williams is None:
        return 'NEUTRAL'
    if williams < -80:
        return 'BUY'
    elif williams > -20:
        return 'SELL'
    return 'NEUTRAL'


def _classify_bbp(bull: Optional[float], bear: Optional[float]) -> str:
    """Bull Bear Power: Bull > 0 & dominant = Buy."""
    if bull is None or bear is None:
        return 'NEUTRAL'
    if bull > 0 and abs(bull) > abs(bear):
        return 'BUY'
    elif bear < 0 and abs(bear) > abs(bull):
        return 'SELL'
    return 'NEUTRAL'


def _classify_uo(uo: Optional[float]) -> str:
    """Ultimate Oscillator: <30 Buy, >70 Sell."""
    if uo is None:
        return 'NEUTRAL'
    if uo < 30:
        return 'BUY'
    elif uo > 70:
        return 'SELL'
    return 'NEUTRAL'


def _classify_ma(price: float, ma: Optional[float]) -> str:
    """MA: Price > MA = Buy, Price < MA = Sell."""
    if ma is None or price == 0:
        return 'NEUTRAL'

    # Within 0.1% = Neutral
    threshold = ma * 0.001
    if price > ma + threshold:
        return 'BUY'
    elif price < ma - threshold:
        return 'SELL'
    return 'NEUTRAL'
