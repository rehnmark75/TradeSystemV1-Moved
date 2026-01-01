"""
Stock Scanner Deep Dive Tab

Comprehensive analysis for individual stocks:
- Stock search and quick picks
- Claude AI analysis with chart vision
- News sentiment analysis
- Active signal card with P&L
- Technical analysis (RSI, MACD, MAs, trend)
- SMC analysis
- Fundamentals display
- Price chart with indicators
- Scanner signal history
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import requests


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

    # Check if we have a ticker from navigation
    navigated_ticker = st.session_state.get('deep_dive_ticker', None)
    current_deep_dive = st.session_state.get('current_deep_dive_ticker', None)

    if navigated_ticker:
        st.session_state.current_deep_dive_ticker = navigated_ticker
        st.session_state.deep_dive_ticker = None
        ticker = navigated_ticker
        st.info(f"Analyzing **{ticker}** from Top Picks")
    elif current_deep_dive:
        ticker = current_deep_dive
    else:
        ticker = None

    # Search section
    show_search = not ticker
    if ticker:
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
        candles = service.get_daily_candles(ticker, days=90)
        scanner_signals = service.get_scanner_signals_for_ticker(ticker, limit=10)
        fundamentals = service.get_full_fundamentals(ticker)

    if not data or not data.get('instrument'):
        st.error(f"Stock '{ticker}' not found in database.")
        return

    instrument = data.get('instrument', {})
    metrics = data.get('metrics', {})
    watchlist = data.get('watchlist', {})

    active_signal = scanner_signals[0] if scanner_signals else None

    # ==========================================================================
    # SECTION 1: Header with Sector/Industry
    # ==========================================================================
    _render_stock_header(ticker, instrument, watchlist, fundamentals, metrics)

    st.markdown("---")

    # ==========================================================================
    # SECTION 2: Claude AI Analysis
    # ==========================================================================
    _render_claude_analysis_section(service, ticker, active_signal, metrics, fundamentals, candles)

    st.markdown("---")

    # ==========================================================================
    # SECTION 2.5: News Sentiment
    # ==========================================================================
    _render_news_sentiment_section(service, ticker, active_signal, scanner_signals)

    st.markdown("---")

    # ==========================================================================
    # SECTION 3: Active Signal Card
    # ==========================================================================
    if active_signal:
        _render_active_signal_section(active_signal, metrics)
        st.markdown("")

    # ==========================================================================
    # SECTION 4: Technical + SMC Analysis
    # ==========================================================================
    _render_technical_and_smc_analysis(metrics, fundamentals, active_signal)

    st.markdown("---")

    # ==========================================================================
    # SECTION 5: Fundamentals
    # ==========================================================================
    _render_fundamentals_section(fundamentals)

    st.markdown("---")

    # ==========================================================================
    # SECTION 6: Score Breakdown
    # ==========================================================================
    if watchlist:
        _render_score_breakdown(watchlist)

    st.markdown("---")

    # ==========================================================================
    # SECTION 7: Price Chart
    # ==========================================================================
    _render_price_chart(candles, active_signal, scanner_signals)

    st.markdown("---")

    # ==========================================================================
    # SECTION 8: Scanner Signal History
    # ==========================================================================
    _render_signal_history(scanner_signals)


def _render_stock_header(ticker, instrument, watchlist, fundamentals, metrics):
    """Render the stock header with tier, score, and sector info."""
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

    # Key metrics row 1
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
        market_cap = fundamentals.get('market_cap')
        cap_str = f"${market_cap}" if market_cap else "N/A"
        st.metric("Market Cap", cap_str)

    # Key metrics row 2 - Relative Strength
    rs_percentile = metrics.get('rs_percentile')
    rs_trend = metrics.get('rs_trend')
    rs_vs_spy = metrics.get('rs_vs_spy')

    if rs_percentile is not None:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            # RS Percentile with color-coded delta
            rs_label = "Elite" if rs_percentile >= 90 else "Strong" if rs_percentile >= 70 else "Average" if rs_percentile >= 40 else "Weak"
            st.metric("RS Percentile", f"{rs_percentile}", rs_label)

        with col2:
            # RS vs SPY ratio
            if rs_vs_spy:
                rs_delta = "Outperforming" if rs_vs_spy > 1.0 else "Underperforming" if rs_vs_spy < 1.0 else None
                st.metric("RS vs SPY", f"{rs_vs_spy:.2f}", rs_delta)
            else:
                st.metric("RS vs SPY", "N/A")

        with col3:
            # RS Trend
            trend_icon = "↑" if rs_trend == 'improving' else "↓" if rs_trend == 'deteriorating' else "→"
            trend_label = rs_trend.capitalize() if rs_trend else "N/A"
            st.metric("RS Trend", f"{trend_icon} {trend_label}")

        with col4:
            # 20-day price change
            change_20d = metrics.get('price_change_20d', 0) or 0
            st.metric("20D Change", f"{change_20d:+.1f}%")

        with col5:
            # Trend strength
            trend_strength = metrics.get('trend_strength', '') or watchlist.get('trend_strength', '')
            st.metric("Trend", trend_strength or "N/A")


def _render_claude_analysis_section(service, ticker, active_signal, metrics, fundamentals, candles):
    """Render Claude AI analysis section."""
    st.markdown('<div class="section-header">🤖 AI Analysis</div>', unsafe_allow_html=True)

    # Check multiple sources for Claude analysis
    existing_analysis = None
    analysis_source = None

    # Source 1: Check session state
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

    # Source 3: Check watchlist table
    if not existing_analysis:
        watchlist_analysis = service.get_latest_claude_analysis_from_watchlist(ticker)
        if watchlist_analysis and watchlist_analysis.get('rating'):
            existing_analysis = watchlist_analysis
            analysis_source = 'top_picks'

    if existing_analysis and (existing_analysis.get('rating') or existing_analysis.get('grade')):
        _display_existing_claude_analysis(existing_analysis, analysis_source, service, ticker, active_signal, metrics, fundamentals, candles, session_key)
    else:
        st.info("No AI analysis available for this stock yet.")
        if st.button("🤖 Analyze with Claude (with Chart Vision)", key="analyze_claude", type="primary"):
            signal_to_analyze = active_signal if active_signal else {
                'ticker': ticker,
                'signal_type': 'ANALYSIS',
                'scanner_name': 'Deep Dive',
                'entry_price': metrics.get('current_price', 0),
                'composite_score': 50,
                'quality_tier': 'T3',
            }
            _run_claude_analysis(service, ticker, signal_to_analyze, metrics, fundamentals, candles, session_key)


def _display_existing_claude_analysis(existing_analysis, analysis_source, service, ticker, active_signal, metrics, fundamentals, candles, session_key):
    """Display existing Claude analysis."""
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

    rating_colors = {
        'A+': '#28a745', 'A': '#28a745', 'A-': '#5cb85c',
        'B+': '#17a2b8', 'B': '#17a2b8', 'B-': '#5bc0de',
        'C+': '#ffc107', 'C': '#ffc107', 'C-': '#f0ad4e',
        'D': '#dc3545', 'F': '#dc3545'
    }
    rating_color = rating_colors.get(rating, '#6c757d')

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

        if thesis:
            st.markdown(f"""
            <div style="margin-top: 0.5rem; padding: 0.5rem; background: #e7f3ff; border-radius: 4px; font-size: 0.9rem;">
                💡 <strong>Thesis:</strong> {thesis}
            </div>
            """, unsafe_allow_html=True)

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

    # Show when analyzed and source
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

            is_stale = time_ago.days > 7
        else:
            date_str = str(analyzed_at)[:16]
            ago_str = ""
            is_stale = False

        source_label = {
            'session': 'This session',
            'signal': 'Scanner Signal',
            'top_picks': 'Top Picks'
        }.get(analysis_source, '')
        source_text = f" | Source: {source_label}" if source_label else ""

        if is_stale:
            st.warning(f"⚠️ Analysis is {time_ago.days} days old - consider refreshing")
        st.caption(f"📅 {date_str} ({ago_str}){source_text}")

    if st.button("🔄 Re-analyze with Claude", key="reanalyze_claude"):
        _run_claude_analysis(service, ticker, active_signal, metrics, fundamentals, candles, session_key)


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


def _render_news_sentiment_section(service, ticker, active_signal, scanner_signals):
    """Render news sentiment section."""
    st.markdown('<div class="section-header">📰 News Sentiment</div>', unsafe_allow_html=True)

    news_data = None
    news_signal_id = None

    # Source 1: Check session state
    session_news_key = f"news_sentiment_{ticker}"
    if session_news_key in st.session_state:
        news_data = st.session_state[session_news_key]

    # Source 2: Check active signal
    if not news_data and active_signal and active_signal.get('news_sentiment_score') is not None:
        news_data = {
            'score': active_signal.get('news_sentiment_score'),
            'level': active_signal.get('news_sentiment_level'),
            'count': active_signal.get('news_headlines_count', 0),
            'factors': active_signal.get('news_factors', []),
            'analyzed_at': active_signal.get('news_analyzed_at')
        }
        news_signal_id = active_signal.get('id')

    # Source 3: Check other signals
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

    if not news_signal_id and scanner_signals:
        news_signal_id = scanner_signals[0].get('id')

    col1, col2 = st.columns([3, 1])

    with col2:
        button_label = "🔄 Refresh News" if news_data else "📰 Fetch News"
        if st.button(button_label, key=f"fetch_news_{ticker}"):
            with st.spinner(f"Fetching news for {ticker}..."):
                if news_signal_id:
                    result = service.enrich_signal_with_news(news_signal_id)
                else:
                    result = _fetch_news_for_ticker(ticker)

                if result.get('success'):
                    st.success(f"✅ {result.get('message', 'News fetched!')}")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Failed to fetch news')}")

    with col1:
        if news_data:
            _display_news_sentiment(news_data)
        else:
            st.info("📰 No news sentiment data available. Click 'Fetch News' to analyze recent news for this stock.")


def _display_news_sentiment(news_data):
    """Display news sentiment data."""
    analyzed_at = news_data.get('analyzed_at')
    news_time_str = None
    news_ago_str = None

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

    if news_time_str:
        ago_text = f" ({news_ago_str})" if news_ago_str else ""
        st.caption(f"📅 {news_time_str}{ago_text}")


def _fetch_news_for_ticker(ticker: str) -> Dict[str, Any]:
    """Fetch and analyze news for a ticker without requiring a signal."""
    try:
        finnhub_api_key = os.getenv('FINNHUB_API_KEY', '')
        if not finnhub_api_key:
            return {'success': False, 'error': 'FINNHUB_API_KEY not configured'}

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

            avg_score = sum(scores) / len(scores) if scores else 0.0

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


def _render_active_signal_section(active_signal, metrics):
    """Render active signal card."""
    st.markdown('<div class="section-header">📈 Active Signal</div>', unsafe_allow_html=True)

    sig_type = active_signal.get('signal_type', 'BUY')
    scanner_name = active_signal.get('scanner_name', 'Unknown')
    entry_price = active_signal.get('entry_price', 0)
    stop_loss = active_signal.get('stop_loss', 0)
    take_profit_1 = active_signal.get('take_profit_1', 0)
    quality_tier = active_signal.get('quality_tier', '-')
    composite_score = active_signal.get('composite_score', 0)
    signal_timestamp = active_signal.get('signal_timestamp')

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

    if signal_timestamp:
        if isinstance(signal_timestamp, datetime):
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


def _render_technical_and_smc_analysis(metrics, fundamentals, active_signal):
    """Render technical and SMC analysis side by side."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Technical Analysis</div>', unsafe_allow_html=True)

        # RSI
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

        def pct_from_ma(price, ma):
            return ((price - ma) / ma * 100) if ma > 0 else 0

        pct_20 = pct_from_ma(current_price, sma_20)
        pct_50 = pct_from_ma(current_price, sma_50)
        pct_200 = pct_from_ma(current_price, sma_200)

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

        smc_trend = metrics.get('smc_trend')
        smc_bias = metrics.get('smc_bias')
        smc_confluence = metrics.get('smc_confluence_score', 0) or 0

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

        if smc_confluence == 0:
            conf_factors = 0
            if smc_trend == 'Bullish':
                if rsi < 70: conf_factors += 2
                if macd > macd_signal: conf_factors += 2
                if current_price > sma_20: conf_factors += 1.5
                if current_price > sma_50: conf_factors += 1.5
                if current_price > sma_200: conf_factors += 1.5
            elif smc_trend == 'Bearish':
                if rsi > 30: conf_factors += 2
                if macd < macd_signal: conf_factors += 2
                if current_price < sma_20: conf_factors += 1.5
                if current_price < sma_50: conf_factors += 1.5
                if current_price < sma_200: conf_factors += 1.5
            smc_confluence = min(conf_factors, 10)

        bos_direction = smc_trend

        trend_colors_map = {'Bullish': '#28a745', 'Bearish': '#dc3545', 'Neutral': '#6c757d'}
        trend_color = trend_colors_map.get(smc_trend, '#6c757d')

        zone_pct = 50
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


def _render_fundamentals_section(fundamentals):
    """Render fundamentals section."""
    st.markdown('<div class="section-header">📊 Fundamentals</div>', unsafe_allow_html=True)

    if fundamentals:
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

        with st.expander("📋 View All Fundamentals"):
            _render_full_fundamentals(fundamentals)
    else:
        st.info("No fundamental data available")


def _render_full_fundamentals(fundamentals):
    """Render full fundamentals in expander."""
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


def _render_score_breakdown(watchlist):
    """Render score breakdown section."""
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


def _render_price_chart(candles, active_signal, scanner_signals):
    """Render price chart with indicators."""
    st.markdown('<div class="section-header">📈 Price Chart (90 Days)</div>', unsafe_allow_html=True)

    if candles.empty:
        st.info("No price data available")
        return

    candles = candles.copy()
    candles['sma20'] = candles['close'].rolling(20).mean()
    candles['sma50'] = candles['close'].rolling(50).mean()
    candles['sma200'] = candles['close'].rolling(200).mean()

    delta = candles['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    candles['rsi'] = 100 - (100 / (1 + rs))

    candles['vol_avg'] = candles['volume'].rolling(20).mean()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('', 'RSI(14)', 'Volume')
    )

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

    fig.add_trace(go.Scatter(
        x=candles['timestamp'], y=candles['rsi'],
        name='RSI', line=dict(color='#9C27B0', width=1.5),
        fill='tozeroy', fillcolor='rgba(156, 39, 176, 0.1)'
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash="dot", line_color="#dc3545", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#28a745", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#6c757d", row=2, col=1)

    colors = ['#28a745' if c >= o else '#dc3545' for c, o in zip(candles['close'], candles['open'])]
    fig.add_trace(go.Bar(
        x=candles['timestamp'], y=candles['volume'],
        marker_color=colors, name='Volume', opacity=0.7
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=candles['timestamp'], y=candles['vol_avg'],
        name='Vol Avg', line=dict(color='#FF9800', width=1, dash='dash')
    ), row=3, col=1)

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


def _render_signal_history(scanner_signals):
    """Render scanner signal history table."""
    st.markdown('<div class="section-header">📋 Scanner Signal History</div>', unsafe_allow_html=True)

    if scanner_signals:
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
