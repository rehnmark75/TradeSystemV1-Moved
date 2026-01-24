"""
Stock Scanner Signals Tab

Unified view of all scanner signals with:
- Filter by scanner, tier, status, Claude analysis
- Signal cards with expandable details
- Signal comparison feature
- CSV export functionality
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from .tradingview_gauge import render_tv_summary_section
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


# Tier and action color mappings
TIER_COLORS = {
    'A+': 'green', 'A': 'blue', 'B': 'orange', 'C': 'gray', 'D': 'red'
}

TIER_HEX_COLORS = {
    'A+': '#1e7e34', 'A': '#28a745', 'B': '#17a2b8', 'C': '#ffc107', 'D': '#6c757d'
}

CLAUDE_ACTION_COLORS = {
    'STRONG BUY': 'green',
    'BUY': 'blue',
    'HOLD': 'orange',
    'AVOID': 'red'
}

NEWS_SENTIMENT_COLORS = {
    'very_bullish': 'green',
    'bullish': 'blue',
    'neutral': 'gray',
    'bearish': 'orange',
    'very_bearish': 'red'
}

NEWS_SENTIMENT_ICONS = {
    'very_bullish': '🟢',
    'bullish': '🔵',
    'neutral': '⚪',
    'bearish': '🟠',
    'very_bearish': '🔴'
}

SCANNER_ICONS = {
    'Trend Momentum': '📈',
    'Breakout Confirmation': '🚀',
    'Mean Reversion': '🔄',
    'Gap And Go': '⚡',
    'Sector Rotation': '🔀',
    'Zlma Trend': '〰️',
    'Smc Ema Trend': '🎯',
    'Ema Crossover': '📊',
    'Macd Momentum': '📉',
    'Selling Climax': '💥',
    'Rsi Divergence': '↗️',
    'Wyckoff Spring': '🌱',
}

# RS Percentile color coding
RS_COLORS = {
    'elite': '#28a745',      # 90+ green
    'strong': '#17a2b8',     # 70-89 blue
    'average': '#ffc107',    # 40-69 yellow
    'weak': '#dc3545',       # <40 red
}

RS_TREND_ICONS = {
    'improving': '↗️',
    'stable': '➡️',
    'deteriorating': '↘️',
}

# DAQ (Deep Analysis Quality) color coding
DAQ_COLORS = {
    'A+': '#1e7e34',  # Dark green
    'A': '#28a745',   # Green
    'B': '#17a2b8',   # Blue
    'C': '#ffc107',   # Yellow
    'D': '#dc3545',   # Red
}

def _get_daq_color(grade: str) -> str:
    """Get color for DAQ grade."""
    return DAQ_COLORS.get(grade, '#6c757d')


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
            <span style="width: 70px; font-size: 0.75rem; color: #555;">{label}</span>
            <div style="flex: 1; background: #e9ecef; border-radius: 4px; height: 10px; max-width: 120px;">
                <div style="width: {bar_width}%; background: {color}; height: 100%; border-radius: 4px;"></div>
            </div>
            <span style="font-size: 0.75rem; font-weight: bold; width: 40px;">{weighted}/{max_points}</span>
        </div>
    </div>
    '''


def _get_risk_badges_html(earnings_risk: bool, high_short: bool, sector_weak: bool) -> str:
    """Generate risk flag badges HTML for signals."""
    badges = []
    if earnings_risk:
        badges.append('<span style="background: #dc3545; color: white; padding: 1px 6px; border-radius: 10px; font-size: 0.7rem; margin-right: 4px;" title="Earnings within 7 days">EARNINGS</span>')
    if high_short:
        badges.append('<span style="background: #fd7e14; color: white; padding: 1px 6px; border-radius: 10px; font-size: 0.7rem; margin-right: 4px;" title="High short interest >20%">HIGH SI</span>')
    if sector_weak:
        badges.append('<span style="background: #6c757d; color: white; padding: 1px 6px; border-radius: 10px; font-size: 0.7rem;" title="Sector underperforming SPY">SECTOR WEAK</span>')
    return ''.join(badges) if badges else ''


def _render_signal_trade_plan(signal: Dict[str, Any]) -> None:
    """Render the Trade Plan section for a signal with fixed 3% SL / 5% TP.

    Uses entry_price from the signal to calculate:
    - Entry Zone: entry_price -0.5% to +1%
    - Stop Loss: 3% below entry
    - Target 1: 5% above entry
    - Target 2: 10% above entry
    - R:R Ratio: 1.67 (fixed)

    Also displays support/resistance levels and ATR for context.
    """
    entry = float(signal.get('entry_price', 0) or 0)
    if entry <= 0:
        return

    # Fixed trade parameters (learning phase)
    stop_loss = entry * 0.97  # 3% below
    target_1 = entry * 1.05   # 5% above
    target_2 = entry * 1.10   # 10% above
    entry_low = entry * 0.995
    entry_high = entry * 1.01
    risk_pct = 3.0
    rr_ratio = 1.67

    # Key levels from screening metrics (convert Decimal to float for arithmetic)
    swing_high = float(signal.get('swing_high')) if signal.get('swing_high') is not None else None
    swing_low = float(signal.get('swing_low')) if signal.get('swing_low') is not None else None
    swing_high_date = signal.get('swing_high_date')
    swing_low_date = signal.get('swing_low_date')
    atr_14 = signal.get('atr_14')
    atr_percent = signal.get('atr_percent')
    relative_volume = signal.get('relative_volume')
    rs_percentile = signal.get('rs_percentile')
    rs_trend = signal.get('rs_trend')

    # Earnings data
    earnings_date = signal.get('earnings_date')
    days_to_earnings = signal.get('days_to_earnings')
    earnings_risk = signal.get('earnings_within_7d', False)

    # Trade Plan header
    st.markdown("""
    <div style="font-weight: bold; font-size: 1rem; color: #1a5f7a; margin: 10px 0 8px 0; border-bottom: 2px solid #1a5f7a; padding-bottom: 4px;">
        Trade Plan (3% SL / 5% TP)
    </div>
    """, unsafe_allow_html=True)

    # Earnings warning banner
    if days_to_earnings is not None and not pd.isna(days_to_earnings):
        days_int = int(days_to_earnings)
        earnings_str = earnings_date.strftime('%m/%d') if earnings_date and not pd.isna(earnings_date) else ''
        if days_int <= 3:
            st.markdown(f"""
            <div style="background: #dc3545; color: white; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9rem;">
                <strong>EARNINGS IMMINENT</strong> - Reporting on {earnings_str} ({days_int} day{"s" if days_int != 1 else ""} away). High volatility expected.
            </div>
            """, unsafe_allow_html=True)
        elif days_int <= 7:
            st.markdown(f"""
            <div style="background: #fd7e14; color: white; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9rem;">
                <strong>EARNINGS WARNING</strong> - Reports {earnings_str} ({days_int} days away). Plan exit before announcement.
            </div>
            """, unsafe_allow_html=True)
        elif days_int <= 14:
            st.markdown(f"""
            <div style="background: #ffc107; color: #212529; padding: 6px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.85rem;">
                Earnings: {earnings_str} ({days_int} days) - Monitor for pre-earnings positioning
            </div>
            """, unsafe_allow_html=True)
    elif earnings_risk:
        st.markdown("""
        <div style="background: #dc3545; color: white; padding: 6px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.85rem;">
            <strong>EARNINGS RISK</strong> - Earnings expected within 7 days.
        </div>
        """, unsafe_allow_html=True)

    # Three columns layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div style="font-weight: bold; color: #495057; margin-bottom: 4px;">Entry & Risk</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 0.85rem; margin: 2px 0;">
            <span style="color: #666;">Entry Zone:</span>
            <span style="color: #28a745; font-weight: bold;">${entry_low:.2f} - ${entry_high:.2f}</span>
        </div>
        <div style="font-size: 0.85rem; margin: 2px 0;">
            <span style="color: #666;">Stop Loss:</span>
            <span style="color: #dc3545; font-weight: bold;">${stop_loss:.2f}</span>
            <span style="color: #888; font-size: 0.75rem;">(-{risk_pct:.0f}%)</span>
        </div>
        <div style="font-size: 0.85rem; margin: 2px 0;">
            <span style="color: #666;">Risk:</span>
            <span style="color: #28a745; font-weight: bold;">{risk_pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="font-weight: bold; color: #495057; margin-bottom: 4px;">Targets</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 0.85rem; margin: 2px 0;">
            <span style="color: #666;">Target 1:</span>
            <span style="color: #28a745; font-weight: bold;">${target_1:.2f}</span>
            <span style="color: #888; font-size: 0.75rem;">(+5%)</span>
        </div>
        <div style="font-size: 0.85rem; margin: 2px 0;">
            <span style="color: #666;">Target 2:</span>
            <span style="color: #17a2b8; font-weight: bold;">${target_2:.2f}</span>
            <span style="color: #888; font-size: 0.75rem;">(+10%)</span>
        </div>
        <div style="font-size: 0.85rem; margin: 2px 0;">
            <span style="color: #666;">R:R Ratio:</span>
            <span style="color: #ffc107; font-weight: bold;">1:{rr_ratio:.1f}</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown('<div style="font-weight: bold; color: #495057; margin-bottom: 4px;">Key Levels</div>', unsafe_allow_html=True)
        levels_html = ""
        if swing_high and not pd.isna(swing_high):
            sh_pct = ((swing_high - entry) / entry * 100) if entry > 0 else 0
            date_str = swing_high_date.strftime('%m/%d') if swing_high_date and not pd.isna(swing_high_date) else ''
            levels_html += f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Resistance:</span>
                <span style="color: #dc3545; font-weight: bold;">${swing_high:.2f}</span>
                <span style="color: #888; font-size: 0.75rem;">(+{sh_pct:.1f}% {date_str})</span>
            </div>
            """
        if swing_low and not pd.isna(swing_low):
            sl_pct = ((swing_low - entry) / entry * 100) if entry > 0 else 0
            date_str = swing_low_date.strftime('%m/%d') if swing_low_date and not pd.isna(swing_low_date) else ''
            levels_html += f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">Support:</span>
                <span style="color: #28a745; font-weight: bold;">${swing_low:.2f}</span>
                <span style="color: #888; font-size: 0.75rem;">({sl_pct:+.1f}% {date_str})</span>
            </div>
            """
        if atr_percent and not pd.isna(atr_percent):
            atr_color = '#dc3545' if atr_percent > 5 else '#ffc107' if atr_percent > 3 else '#28a745'
            atr_val = f"${atr_14:.2f}" if atr_14 and not pd.isna(atr_14) else ""
            levels_html += f"""
            <div style="font-size: 0.85rem; margin: 2px 0;">
                <span style="color: #666;">ATR%:</span>
                <span style="color: {atr_color}; font-weight: bold;">{atr_percent:.1f}%</span>
                <span style="color: #888; font-size: 0.75rem;">({atr_val})</span>
            </div>
            """
        if levels_html:
            st.markdown(levels_html, unsafe_allow_html=True)
        else:
            st.caption("No key levels data")

    # Volume and RS context row
    if relative_volume or rs_percentile:
        st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
        vol_col, rs_col = st.columns(2)

        with vol_col:
            if relative_volume and not pd.isna(relative_volume):
                if relative_volume >= 1.5:
                    vol_label, vol_color = 'Accumulation', '#28a745'
                elif relative_volume <= 0.7:
                    vol_label, vol_color = 'Distribution', '#dc3545'
                else:
                    vol_label, vol_color = 'Neutral', '#6c757d'
                st.markdown(f"""
                <div style="font-size: 0.85rem;">
                    <span style="color: #666;">Volume:</span>
                    <span style="color: {vol_color}; font-weight: bold;">{vol_label}</span>
                    <span style="color: #888; font-size: 0.75rem;">({relative_volume:.1f}x avg)</span>
                </div>
                """, unsafe_allow_html=True)

        with rs_col:
            if rs_percentile and not pd.isna(rs_percentile):
                rs_val = int(rs_percentile)
                rs_icon = RS_TREND_ICONS.get(rs_trend, '')
                rs_color = _get_rs_color(rs_val)
                st.markdown(f"""
                <div style="font-size: 0.85rem;">
                    <span style="color: #666;">Relative Strength:</span>
                    <span style="color: {rs_color}; font-weight: bold;">{rs_val}th %ile {rs_icon}</span>
                </div>
                """, unsafe_allow_html=True)


def _render_daq_visual_breakdown(
    daq_score: int,
    daq_grade: str,
    mtf_score: Optional[int],
    volume_score: Optional[int],
    smc_score: Optional[int],
    quality_score: Optional[int],
    catalyst_score: Optional[int],
    news_score: Optional[int],
    regime_score: Optional[int],
    sector_score: Optional[int],
    earnings_risk: bool = False,
    high_short: bool = False,
    sector_weak: bool = False
) -> None:
    """Render visual DAQ breakdown with score bars."""
    grade_color = _get_daq_color(daq_grade)
    risk_html = _get_risk_badges_html(earnings_risk, high_short, sector_weak)

    # Header with score, grade, and risk badges
    st.markdown(f'''
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <div style="background: {grade_color}; color: white; padding: 6px 14px; border-radius: 6px; font-size: 1.1rem; font-weight: bold;">
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
            int((mtf_score or 0) / 100 * 20),
            int((volume_score or 0) / 100 * 10),
            int((smc_score or 0) / 100 * 15),
        ])
        st.markdown(f'<div style="font-weight: bold; margin-bottom: 4px; color: #1a5f7a; font-size: 0.85rem;">TECHNICAL ({tech_total}/45)</div>', unsafe_allow_html=True)
        st.markdown(_render_score_bar(mtf_score, 20, 'MTF'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(volume_score, 10, 'Volume'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(smc_score, 15, 'SMC'), unsafe_allow_html=True)

    with col2:
        # Calculate fundamental total
        fund_total = sum([
            int((quality_score or 0) / 100 * 15),
            int((catalyst_score or 0) / 100 * 10),
        ])
        st.markdown(f'<div style="font-weight: bold; margin-bottom: 4px; color: #1a5f7a; font-size: 0.85rem;">FUNDAMENTAL ({fund_total}/25)</div>', unsafe_allow_html=True)
        st.markdown(_render_score_bar(quality_score, 15, 'Quality'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(catalyst_score, 10, 'Catalyst'), unsafe_allow_html=True)

    with col3:
        # Calculate contextual total
        ctx_total = sum([
            int((news_score or 0) / 100 * 10),
            int((regime_score or 0) / 100 * 10),
            int((sector_score or 0) / 100 * 10),
        ])
        st.markdown(f'<div style="font-weight: bold; margin-bottom: 4px; color: #1a5f7a; font-size: 0.85rem;">CONTEXTUAL ({ctx_total}/30)</div>', unsafe_allow_html=True)
        st.markdown(_render_score_bar(news_score, 10, 'News'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(regime_score, 10, 'Regime'), unsafe_allow_html=True)
        st.markdown(_render_score_bar(sector_score, 10, 'Sector'), unsafe_allow_html=True)

def _get_rs_color(percentile: int) -> str:
    """Get color for RS percentile value."""
    if percentile is None:
        return '#6c757d'
    if percentile >= 90:
        return RS_COLORS['elite']
    elif percentile >= 70:
        return RS_COLORS['strong']
    elif percentile >= 40:
        return RS_COLORS['average']
    else:
        return RS_COLORS['weak']

def _get_rs_label(percentile: int) -> str:
    """Get label for RS percentile value."""
    if percentile is None:
        return '-'
    if percentile >= 90:
        return 'Elite'
    elif percentile >= 70:
        return 'Strong'
    elif percentile >= 40:
        return 'Average'
    else:
        return 'Weak'


def render_signals_tab(service):
    """Render the All Signals tab - unified view of all scanner signals."""
    st.markdown("""
    <div class="main-header">
        <h2>All Signals</h2>
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

    # Deep Analysis (DAQ) stats - Row 3
    daq_stats = service.get_deep_analysis_summary(days=7)
    if daq_stats and daq_stats.get('total', 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Deep Analyzed", daq_stats.get('total', 0), help="Signals with Deep Analysis Quality (DAQ) score")
        col2.metric("Avg DAQ Score", f"{daq_stats.get('avg_daq', 0)}/100", help="Average DAQ score (0-100)")
        grade_summary = f"A+:{daq_stats.get('grade_a_plus', 0)} A:{daq_stats.get('grade_a', 0)} B:{daq_stats.get('grade_b', 0)}"
        col3.metric("DAQ A+/A/B", grade_summary, help="DAQ grades breakdown")
        col4.metric("DAQ C/D", f"{daq_stats.get('grade_c', 0)}/{daq_stats.get('grade_d', 0)}", help="Lower DAQ grades")

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
            ["All", "active", "triggered", "closed", "expired"]
        )

    with col4:
        claude_filter = st.selectbox(
            "Claude Analysis",
            ["All Signals", "Claude Analyzed Only", "A+ Grade", "A Grade", "B Grade", "STRONG BUY", "BUY"]
        )

    # Filters - Row 2: Date range and RS filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        date_from = st.date_input(
            "Signal Date From",
            value=datetime.now().date() - timedelta(days=7),
            help="Filter signals detected from this date (default: last 7 days)"
        )

    with col2:
        date_to = st.date_input(
            "Signal Date To",
            value=datetime.now().date(),
            help="Filter signals detected up to this date"
        )

    with col3:
        rs_filter = st.selectbox(
            "Relative Strength",
            ["All RS", "Elite (90+)", "Strong (70+)", "Average (40+)", "Weak (<40)"],
            help="Filter by Relative Strength percentile vs SPY"
        )

    with col4:
        rs_trend_filter = st.selectbox(
            "RS Trend",
            ["All Trends", "Improving", "Stable", "Deteriorating"],
            help="Filter by RS momentum direction"
        )

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

    # RS filter mapping
    min_rs_percentile = None
    max_rs_percentile = None
    if rs_filter == "Elite (90+)":
        min_rs_percentile = 90
    elif rs_filter == "Strong (70+)":
        min_rs_percentile = 70
    elif rs_filter == "Average (40+)":
        min_rs_percentile = 40
    elif rs_filter == "Weak (<40)":
        max_rs_percentile = 39

    # RS trend filter
    rs_trend = None
    if rs_trend_filter != "All Trends":
        rs_trend = rs_trend_filter.lower()

    signals = service.get_scanner_signals(
        scanner_name=scanner_name,
        status=status,
        min_score=min_score,
        min_claude_grade=min_claude_grade,
        claude_analyzed_only=claude_analyzed_only,
        signal_date_from=str(date_from) if date_from else None,
        signal_date_to=str(date_to) if date_to else None,
        min_rs_percentile=min_rs_percentile,
        max_rs_percentile=max_rs_percentile,
        rs_trend=rs_trend,
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
        if signals:
            csv_data = _signals_to_csv(signals)
            st.download_button(
                "Download CSV (TradingView)",
                csv_data,
                f"scanner_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

    with col2:
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
    selected_signals = [s for s in all_signals if s.get('id') in selected_ids]

    if len(selected_signals) < 2:
        st.warning("Select at least 2 signals to compare")
        return

    st.markdown("### 🔄 Signal Comparison")

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

            tier_color = TIER_HEX_COLORS.get(tier, '#6c757d')

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
                action_colors_hex = {'STRONG BUY': '#28a745', 'BUY': '#28a745', 'HOLD': '#ffc107', 'AVOID': '#dc3545'}
                action_color = action_colors_hex.get(claude_action, '#6c757d')

                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <span style="background: {TIER_HEX_COLORS.get(claude_grade, '#6c757d')}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.3rem;">{claude_grade}</span>
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
        factors_str = ', '.join(factors)
    else:
        factors_str = str(factors) if factors else ''

    # Relative Strength data
    rs_percentile = signal.get('rs_percentile')
    rs_trend = signal.get('rs_trend')
    sector = signal.get('sector')
    sector_stage = signal.get('sector_stage')

    # Signal timestamp
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

    # Deep Analysis Quality (DAQ) data
    daq_score = signal.get('daq_score')
    daq_grade = signal.get('daq_grade')
    daq_mtf_score = signal.get('mtf_score')
    daq_volume_score = signal.get('daq_volume_score')
    daq_smc_score = signal.get('daq_smc_score')
    daq_quality_score = signal.get('daq_quality_score')
    daq_catalyst_score = signal.get('daq_catalyst_score')
    daq_news_score = signal.get('daq_news_score')
    daq_regime_score = signal.get('daq_regime_score')
    daq_sector_score = signal.get('daq_sector_score')
    daq_earnings_risk = signal.get('earnings_within_7d', False)
    daq_high_short = signal.get('high_short_interest', False)
    daq_sector_weak = signal.get('sector_underperforming', False)
    has_daq = daq_score is not None

    # Stock metrics
    avg_daily_change = signal.get('avg_daily_change_5d', 0) or 0

    # Format claude_analyzed_at timestamp with staleness check
    analyzed_time_str = None
    analyzed_ago_str = None
    is_analysis_stale = False
    if claude_analyzed_at:
        if isinstance(claude_analyzed_at, datetime):
            analyzed_time_str = claude_analyzed_at.strftime('%b %d, %Y %H:%M')
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

    tier_color = TIER_COLORS.get(tier, 'gray')
    scanner_icon = SCANNER_ICONS.get(scanner, '📊')

    # Build title with Claude info if available
    claude_badge = ""
    if has_claude:
        action_color = CLAUDE_ACTION_COLORS.get(claude_action, 'gray')
        claude_badge = f" | 🤖 :{action_color}[{claude_action}] ({claude_grade})"

    # Build news badge if available
    news_badge = ""
    if has_news:
        news_level = news_sentiment_level or 'neutral'
        news_color = NEWS_SENTIMENT_COLORS.get(news_level.lower(), 'gray')
        news_icon = NEWS_SENTIMENT_ICONS.get(news_level.lower(), '📰')
        news_label = news_level.replace('_', ' ').title()
        news_badge = f" | {news_icon} :{news_color}[{news_label}]"

    timestamp_part = f" | 📅 {signal_time_str}" if signal_time_str else ""

    # Days active badge - shows persistence (signal firing multiple days)
    days_active = signal.get('days_active', 1)
    days_badge = ""
    if days_active and days_active > 1:
        days_badge = f" | 🔥 {days_active}d"

    # In trade badge - shows if stock has an open broker position
    in_trade = signal.get('in_trade', False)
    trade_badge = ""
    if in_trade:
        trade_profit = signal.get('trade_profit', 0) or 0
        profit_color = 'green' if trade_profit >= 0 else 'red'
        profit_sign = '+' if trade_profit >= 0 else ''
        trade_badge = f" | 💼 :{profit_color}[{profit_sign}${trade_profit:.2f}]"

    # RS badge - shows relative strength percentile and trend
    rs_badge = ""
    if rs_percentile is not None:
        rs_label = _get_rs_label(rs_percentile)
        rs_trend_icon = RS_TREND_ICONS.get(rs_trend, '')
        if rs_percentile >= 70:
            rs_badge = f" | RS: :green[{rs_percentile}]{rs_trend_icon}"
        elif rs_percentile >= 40:
            rs_badge = f" | RS: :orange[{rs_percentile}]{rs_trend_icon}"
        else:
            rs_badge = f" | RS: :red[{rs_percentile}]{rs_trend_icon}"

    # DAQ badge - shows Deep Analysis Quality score
    daq_badge = ""
    if has_daq:
        daq_color = 'green' if daq_score >= 70 else 'orange' if daq_score >= 50 else 'red'
        daq_badge = f" | DAQ: :{daq_color}[{daq_score}]({daq_grade})"

    with st.expander(f"**{ticker}** | :{tier_color}[{tier}] | Score: {score} | {scanner_icon} {scanner}{rs_badge}{daq_badge}{claude_badge}{news_badge}{days_badge}{trade_badge}{timestamp_part}", expanded=False):

        # Metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
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
        with col6:
            st.metric("Avg Move", f"{avg_daily_change:.1f}%/d" if avg_daily_change else "-")

        if setup:
            st.caption(f"**Setup:** {setup}")

        if factors_str:
            st.caption(f"**Factors:** {factors_str}")

        # Trade Plan Section (fixed 3% SL / 5% TP for learning)
        _render_signal_trade_plan(signal)

        # Relative Strength Section (if data available)
        if rs_percentile is not None:
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                rs_color = _get_rs_color(rs_percentile)
                st.markdown(f"**RS Percentile:** <span style='color: {rs_color}; font-weight: bold;'>{rs_percentile}</span>", unsafe_allow_html=True)
            with col2:
                trend_icon = RS_TREND_ICONS.get(rs_trend, '')
                st.markdown(f"**RS Trend:** {trend_icon} {rs_trend or 'N/A'}")
            with col3:
                st.markdown(f"**Sector:** {sector or 'N/A'}")
            with col4:
                stage_color = {'leading': 'green', 'improving': 'blue', 'weakening': 'orange', 'lagging': 'red'}.get(sector_stage, 'gray')
                st.markdown(f"**Sector Stage:** :{stage_color}[{sector_stage or 'N/A'}]")

        # Deep Analysis Quality (DAQ) Section - Visual Breakdown
        if has_daq:
            st.markdown("---")
            st.markdown("#### 📊 Deep Analysis Quality")
            _render_daq_visual_breakdown(
                daq_score=daq_score,
                daq_grade=daq_grade,
                mtf_score=daq_mtf_score,
                volume_score=daq_volume_score,
                smc_score=daq_smc_score,
                quality_score=daq_quality_score,
                catalyst_score=daq_catalyst_score,
                news_score=daq_news_score,
                regime_score=daq_regime_score,
                sector_score=daq_sector_score,
                earnings_risk=daq_earnings_risk,
                high_short=daq_high_short,
                sector_weak=daq_sector_weak
            )

            # TradingView Technical Summary
            render_tv_summary_section(signal)

        # Claude AI Analysis Section
        if has_claude:
            st.markdown("---")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("#### 🤖 Claude AI Analysis")
                if analyzed_time_str:
                    ago_text = f" ({analyzed_ago_str})" if analyzed_ago_str else ""
                    if is_analysis_stale:
                        st.warning(f"⚠️ Analysis is stale - consider refreshing")
                    st.caption(f"📅 {analyzed_time_str}{ago_text}")
            with col2:
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
                grade_color = TIER_COLORS.get(claude_grade, 'gray')
                st.markdown(f"**Grade:** :{grade_color}[{claude_grade}]")
            with col2:
                st.markdown(f"**Score:** {claude_score}/10")
            with col3:
                action_color = CLAUDE_ACTION_COLORS.get(claude_action, 'gray')
                st.markdown(f"**Action:** :{action_color}[{claude_action}]")
            with col4:
                st.markdown(f"**Position:** {claude_position or '-'}")

            if claude_thesis:
                st.markdown(f"**Thesis:** {claude_thesis}")

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
            st.caption("⏳ *Awaiting Claude AI analysis*")

        # News Sentiment Section
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("#### 📰 News Sentiment")
        with col2:
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

            col1, col2, col3 = st.columns(3)
            with col1:
                news_level = news_sentiment_level or 'neutral'
                news_color = NEWS_SENTIMENT_COLORS.get(news_level.lower(), 'gray')
                news_label = news_level.replace('_', ' ').title()
                st.markdown(f"**Sentiment:** :{news_color}[{news_label}]")
            with col2:
                score_display = f"{news_sentiment_score:.2f}" if news_sentiment_score else "N/A"
                st.markdown(f"**Score:** {score_display}")
            with col3:
                st.markdown(f"**Articles:** {news_headlines_count or 0}")

            if news_factors:
                if isinstance(news_factors, list):
                    factors_display = news_factors
                else:
                    factors_display = [news_factors] if news_factors else []

                if factors_display:
                    st.markdown("**Key Factors:**")
                    for factor in factors_display[:3]:
                        if news_level.lower() in ['very_bullish', 'bullish']:
                            st.markdown(f"- :green[+] {factor}")
                        elif news_level.lower() in ['very_bearish', 'bearish']:
                            st.markdown(f"- :red[-] {factor}")
                        else:
                            st.markdown(f"- :gray[•] {factor}")
        else:
            st.caption("📰 *No news data yet - click 'Enrich with News' to fetch*")

        # Chart & Analysis navigation button
        st.markdown("---")
        if st.button(f"📈 Open {ticker} Chart & Analysis", key=f"chart_btn_{signal_id}"):
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

        # Position Calculator Section
        if st.checkbox(f"📐 Calculate Position Size", key=f"pos_calc_{signal_id}", value=False):
            _render_inline_position_calculator(ticker, entry, stop, signal_id)


def _render_inline_position_calculator(ticker: str, entry: float, stop: float, signal_id: int) -> None:
    """Render an inline position calculator for a signal."""
    # Initialize session state for account size if not already set
    if 'account_size' not in st.session_state:
        st.session_state.account_size = 25000.0
    if 'risk_percent' not in st.session_state:
        st.session_state.risk_percent = 1.0

    col1, col2 = st.columns(2)

    with col1:
        account_size = st.number_input(
            "Account Size ($)",
            min_value=1000.0,
            max_value=10000000.0,
            value=st.session_state.account_size,
            step=1000.0,
            format="%.0f",
            key=f"account_{signal_id}"
        )
        st.session_state.account_size = account_size

    with col2:
        risk_pct = st.number_input(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.risk_percent,
            step=0.1,
            format="%.1f",
            key=f"risk_{signal_id}"
        )
        st.session_state.risk_percent = risk_pct

    # Calculate position size
    if entry > 0 and stop > 0 and entry != stop:
        risk_per_share = abs(entry - stop)
        risk_dollars = account_size * (risk_pct / 100)
        shares = int(risk_dollars / risk_per_share)
        position_value = shares * entry
        position_pct = (position_value / account_size) * 100

        # Determine trade direction
        is_long = stop < entry
        target_price = entry + (2 * risk_per_share) if is_long else entry - (2 * risk_per_share)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Shares to Buy", f"{shares:,}")
        with col2:
            st.metric("Position Value", f"${position_value:,.0f}")
        with col3:
            st.metric("Dollar Risk", f"${risk_dollars:,.0f}")
        with col4:
            st.metric("Position %", f"{position_pct:.1f}%")

        # Warnings
        if position_pct > 25:
            st.error("Position > 25% of account - HIGH CONCENTRATION")
        elif position_pct > 15:
            st.warning("Position > 15% of account - moderate concentration")

        if shares < 1:
            st.error("Cannot buy even 1 share with this risk amount")

        # Trade summary
        st.success(f"**{ticker}**: Buy {shares:,} shares @ ${entry:.2f} | Stop ${stop:.2f} | Target ${target_price:.2f} (2R)")


def _signals_to_csv(signals: List[Dict]) -> str:
    """Convert signals to CSV format including Claude analysis and news sentiment."""
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        'Symbol', 'Side', 'Entry', 'Stop', 'TP1', 'TP2',
        'Risk%', 'R:R', 'Score', 'Tier', 'Scanner', 'Setup',
        'Claude_Grade', 'Claude_Score', 'Claude_Action', 'Claude_Conviction', 'Claude_Thesis',
        'News_Sentiment', 'News_Score', 'News_Articles'
    ])

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
