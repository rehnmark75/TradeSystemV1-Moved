"""
Stock Scanner Performance Tab

Individual scanner performance drill-down:
- Signal count and quality distribution
- Win/loss rates (if tracked)
- Historical performance trends
- Scanner-specific configuration info
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


# Scanner descriptions for the 7 active scanners
SCANNER_INFO = {
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


def render_scanner_performance_tab(service):
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
    col_scanner, col_date_from, col_date_to = st.columns([2, 1, 1])

    with col_scanner:
        scanner_names = [s['scanner_name'] for s in by_scanner]
        selected_scanner = st.selectbox(
            "Select Scanner to Analyze",
            scanner_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col_date_from:
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
            total = scanner_stats.get('signal_count', 0)
            active = scanner_stats.get('active_count', 0)
            active_pct = (active / total * 100) if total > 0 else 0
            st.metric("Active %", f"{active_pct:.0f}%")

        st.markdown("---")

        # Get signals for this scanner with date filter
        signals = service.get_scanner_signals(
            scanner_name=selected_scanner,
            status=None,
            signal_date_from=str(date_from),
            signal_date_to=str(date_to),
            limit=200,
            order_by='timestamp'
        )

        if signals:
            # Quality distribution
            st.subheader("Quality Distribution")

            tier_counts = {}
            for s in signals:
                tier = s.get('quality_tier', 'Unknown')
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

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

                action_counts = {}
                for s in claude_analyzed:
                    action = s.get('claude_action', 'Unknown')
                    action_counts[action] = action_counts.get(action, 0) + 1

                cols = st.columns(len(action_counts))
                for i, (action, count) in enumerate(sorted(action_counts.items())):
                    with cols[i % len(cols)]:
                        color = {'STRONG BUY': '🟢', 'BUY': '🔵', 'HOLD': '🟡', 'AVOID': '🔴'}.get(action, '⚪')
                        st.metric(f"{color} {action}", count)

                claude_scores = [s.get('claude_score', 0) for s in claude_analyzed if s.get('claude_score')]
                if claude_scores:
                    avg_claude_score = sum(claude_scores) / len(claude_scores)
                    st.metric("Avg Claude Score", f"{avg_claude_score:.1f}/100")

            st.markdown("---")

            # Recent signals table with date range
            st.subheader(f"{selected_scanner.replace('_', ' ').title()} Signals ({date_from} to {date_to})")

            table_data = []
            for s in signals[:50]:
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
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No signals found for {selected_scanner.replace('_', ' ').title()}")

    # Scanner descriptions
    st.markdown("---")
    st.subheader("Scanner Information")

    info = SCANNER_INFO.get(selected_scanner, {})
    if info:
        st.markdown(f"**Description:** {info.get('description', 'N/A')}")
        st.markdown(f"**Best For:** {info.get('best_for', 'N/A')}")
        st.markdown(f"**Entry Criteria:** {info.get('criteria', 'N/A')}")
