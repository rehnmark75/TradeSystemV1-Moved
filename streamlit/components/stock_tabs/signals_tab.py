"""
Stock Scanner Signals Tab

Unified view of all scanner signals with:
- Filter by scanner, tier, status, Claude analysis
- Signal cards with expandable details
- Signal comparison feature
- CSV export functionality
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List


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

    # Filters - Row 2: Date range and RS filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        date_from = st.date_input(
            "Signal Date From",
            value=datetime.now().date() - timedelta(days=1),
            help="Filter signals detected from this date"
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

    with st.expander(f"**{ticker}** | :{tier_color}[{tier}] | Score: {score} | {scanner_icon} {scanner}{rs_badge}{claude_badge}{news_badge}{days_badge}{trade_badge}{timestamp_part}", expanded=False):

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

        # Position Calculator Section
        st.markdown("---")
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
