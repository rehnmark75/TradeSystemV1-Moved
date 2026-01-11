"""
Stock Scanner Chart Tab

Interactive stock charting with Deep Dive analysis:
- Stock selection from signals, watchlists, or manual entry
- Lightweight Charts (TradingView-like) visualization
- EMAs (20, 50, 100, 200)
- MACD with histogram
- Volume subplot
- Deep Dive analysis sections below chart:
  - Claude AI Analysis
  - News Sentiment
  - Active Signal Card
  - Technical + SMC Analysis
  - Fundamentals
  - Score Breakdown
  - Signal History
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_lightweight_charts_ntf import renderLightweightCharts

# Import deep dive helper functions
from .deep_dive_tab import (
    _render_stock_header,
    _render_claude_analysis_section,
    _render_news_sentiment_section,
    _render_active_signal_section,
    _render_technical_and_smc_analysis,
    _render_fundamentals_section,
    _render_score_breakdown,
    _render_signal_history,
)
from .position_calculator import render_inline_position_calculator


def render_chart_tab(service):
    """
    Render the Chart tab with interactive stock charting and Deep Dive analysis.

    Features:
    - Stock selection from signals, watchlists, or manual entry
    - Candlestick chart with EMAs (20, 50, 100, 200)
    - MACD indicator panel
    - Volume panel
    - Deep Dive analysis sections below chart
    """
    st.header("Stock Chart & Analysis")
    st.markdown("*Interactive charting with comprehensive stock analysis*")

    # Initialize session state for Claude analysis (needed for deep dive sections)
    if 'deep_dive_claude_analysis' not in st.session_state:
        st.session_state.deep_dive_claude_analysis = {}

    # Check if navigated from another tab (watchlist, signals, etc.)
    navigated_ticker = st.session_state.get('chart_ticker', None)
    if navigated_ticker:
        st.session_state.current_chart_ticker = navigated_ticker
        st.session_state.chart_ticker = None  # Clear after use

    # Stock Selection Section
    st.subheader("Select Stock")

    source_col, ticker_col, clear_col = st.columns([1, 2, 0.5])

    # Check if we have a pre-selected ticker from navigation
    current_ticker = st.session_state.get('current_chart_ticker', None)

    with clear_col:
        if current_ticker:
            if st.button("🔄 Clear", key="clear_chart_ticker"):
                st.session_state.current_chart_ticker = None
                st.rerun()

    with source_col:
        source = st.radio(
            "Source",
            ["Signals", "Watchlist", "Manual Entry"],
            horizontal=True,
            key="chart_source"
        )

    selected_ticker = current_ticker  # Start with navigated ticker if available

    with ticker_col:
        if current_ticker:
            st.info(f"Viewing: **{current_ticker}** (from navigation)")
        elif source == "Signals":
            # Date filter for signals
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                signal_date_from = st.date_input(
                    "From date",
                    value=datetime.now() - timedelta(days=30),
                    key="chart_signal_date_from"
                )
            with date_col2:
                signal_date_to = st.date_input(
                    "To date",
                    value=datetime.now(),
                    key="chart_signal_date_to"
                )

            # Get active signal tickers with date filter
            signal_tickers = _get_signal_tickers(
                service,
                date_from=signal_date_from.strftime('%Y-%m-%d') if signal_date_from else None,
                date_to=signal_date_to.strftime('%Y-%m-%d') if signal_date_to else None
            )
            if signal_tickers:
                selected_ticker = st.selectbox(
                    "Select from active signals",
                    signal_tickers,
                    key="chart_signal_ticker"
                )
            else:
                st.info("No active signals available for selected date range")

        elif source == "Watchlist":
            # Date filter for watchlist
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                watchlist_date_from = st.date_input(
                    "From date",
                    value=datetime.now() - timedelta(days=30),
                    key="chart_watchlist_date_from"
                )
            with date_col2:
                watchlist_date_to = st.date_input(
                    "To date",
                    value=datetime.now(),
                    key="chart_watchlist_date_to"
                )

            # Watchlist type selector
            watchlist_types = [
                "EMA 50 Cross Over",
                "EMA 20 Cross Over",
                "MACD Bullish Cross",
                "Gap Up Continuation",
                "RSI Oversold Bounce",
                "All Tickers"
            ]
            watchlist_type = st.selectbox(
                "Watchlist Type",
                watchlist_types,
                key="chart_watchlist_type"
            )

            # Get tickers for selected watchlist with date filter
            watchlist_tickers = _get_watchlist_tickers(
                service,
                watchlist_type,
                date_from=watchlist_date_from.strftime('%Y-%m-%d') if watchlist_date_from else None,
                date_to=watchlist_date_to.strftime('%Y-%m-%d') if watchlist_date_to else None
            )
            if watchlist_tickers:
                selected_ticker = st.selectbox(
                    "Select ticker",
                    watchlist_tickers,
                    key="chart_watchlist_ticker"
                )
            else:
                st.info(f"No tickers in {watchlist_type} watchlist for selected date range")

        else:  # Manual Entry
            manual_input = st.text_input(
                "Enter ticker symbol",
                placeholder="e.g., AAPL, MSFT, GOOGL",
                key="chart_manual_ticker"
            ).upper().strip()

            if manual_input:
                # Validate ticker exists
                all_tickers = service.get_all_tickers()
                if manual_input in all_tickers:
                    selected_ticker = manual_input
                else:
                    # Search for partial matches
                    matches = [t for t in all_tickers if manual_input in t]
                    if matches:
                        st.warning(f"'{manual_input}' not found. Did you mean: {', '.join(matches[:5])}?")
                    else:
                        st.warning(f"Ticker '{manual_input}' not found in database")

    if not selected_ticker:
        st.info("Select a stock to view chart and analysis")
        return

    st.markdown("---")

    # Chart Configuration
    config_col1, config_col2 = st.columns(2)

    with config_col1:
        lookback_days = st.radio(
            "Lookback Period",
            [30, 60, 90, 180],
            index=2,  # Default to 90 days
            horizontal=True,
            key="chart_lookback"
        )

    with config_col2:
        show_volume = st.checkbox("Show Volume", value=True, key="chart_show_volume")
        show_macd = st.checkbox("Show MACD", value=True, key="chart_show_macd")
        show_smc = st.checkbox("Show SMC (Smart Money)", value=False, key="chart_show_smc")
        show_mtf = st.checkbox("Show Multi-Timeframe", value=False, key="chart_show_mtf")
        show_markers = st.checkbox("Show Signal/Watchlist Markers", value=False, key="chart_show_markers")

    # Fetch candle data (extra for EMA 200 warmup)
    with st.spinner(f"Loading {selected_ticker} chart data..."):
        df = service.get_daily_candles(selected_ticker, days=lookback_days + 200)

    if df is None or df.empty:
        st.warning(f"No candle data available for {selected_ticker}")
        return

    # Get stock details for title
    stock_details = service.get_stock_details(selected_ticker)
    stock_name = stock_details.get('name', selected_ticker) if stock_details else selected_ticker

    # Display stock header
    st.subheader(f"{selected_ticker} - {stock_name}")

    # Calculate indicators
    df = _calculate_indicators(df)

    # Trim to lookback period after indicator calculation
    df = df.tail(lookback_days).reset_index(drop=True)

    if df.empty:
        st.warning("Not enough data after indicator calculation")
        return

    # Fetch SMC data if enabled
    smc_data = None
    if show_smc:
        with st.spinner("Loading SMC zones..."):
            smc_data = service.get_smc_zones_for_ticker(selected_ticker)
            if smc_data:
                # Show SMC summary above chart
                _render_smc_summary(smc_data)

    # Fetch markers data if enabled
    markers_data = None
    if show_markers:
        with st.spinner("Loading signal/watchlist markers..."):
            markers_data = service.get_chart_markers_for_ticker(selected_ticker, days=lookback_days)
            if markers_data:
                # Show markers summary above chart
                _render_markers_summary(markers_data)

    # Chart Legend (color-coded)
    legend_parts = [
        '<span style="color: #2196F3;">━</span> EMA 20 &nbsp;&nbsp;',
        '<span style="color: #FF9800;">━</span> EMA 50 &nbsp;&nbsp;',
        '<span style="color: #9C27B0;">━</span> EMA 100 &nbsp;&nbsp;',
        '<span style="color: #F44336;">━</span> EMA 200',
    ]
    if show_macd:
        legend_parts.append(
            '&nbsp;&nbsp;│&nbsp;&nbsp;'
            '<span style="color: #2962FF;">━</span> MACD &nbsp;&nbsp;'
            '<span style="color: #FF6D00;">━</span> Signal &nbsp;&nbsp;'
            '<span style="color: #26a69a;">▮</span>/<span style="color: #ef5350;">▮</span> Histogram'
        )
    st.markdown(
        f'<div style="margin-bottom: 8px; font-size: 12px;">{"".join(legend_parts)}</div>',
        unsafe_allow_html=True
    )

    # Build and render charts
    if show_mtf:
        # Multi-timeframe layout
        _render_mtf_charts(service, selected_ticker, df, lookback_days, show_macd, show_volume, smc_data, markers_data)
    else:
        # Standard single chart layout
        try:
            charts = _build_charts(df, show_macd, show_volume, smc_data, markers_data)
            renderLightweightCharts(charts, f"stock-chart-{selected_ticker}")
        except Exception as chart_error:
            # If chart fails with SMC or markers, try without overlays
            if smc_data or markers_data:
                st.warning("Overlays caused a rendering issue. Showing chart without extra overlays.")
                charts = _build_charts(df, show_macd, show_volume, None, None)
                renderLightweightCharts(charts, f"stock-chart-{selected_ticker}-fallback")
            else:
                st.error(f"Chart rendering failed: {chart_error}")

    # ==========================================================================
    # DEEP DIVE ANALYSIS SECTIONS
    # ==========================================================================
    st.markdown("---")
    st.markdown("## 🔍 Stock Analysis")

    # Fetch all additional stock data for deep dive analysis
    with st.spinner(f"Loading analysis data for {selected_ticker}..."):
        data = service.get_stock_details(selected_ticker)
        candles_90d = service.get_daily_candles(selected_ticker, days=90)
        scanner_signals = service.get_scanner_signals_for_ticker(selected_ticker, limit=10)
        fundamentals = service.get_full_fundamentals(selected_ticker)

    if not data or not data.get('instrument'):
        st.warning(f"Limited analysis data available for '{selected_ticker}'")
    else:
        instrument = data.get('instrument', {})
        metrics = data.get('metrics', {})
        watchlist = data.get('watchlist', {})
        active_signal = scanner_signals[0] if scanner_signals else None

        # SECTION 1: Header with Sector/Industry
        _render_stock_header(selected_ticker, instrument, watchlist, fundamentals, metrics)

        st.markdown("---")

        # SECTION 2: Claude AI Analysis
        _render_claude_analysis_section(service, selected_ticker, active_signal, metrics, fundamentals, candles_90d)

        st.markdown("---")

        # SECTION 3: News Sentiment
        _render_news_sentiment_section(service, selected_ticker, active_signal, scanner_signals)

        st.markdown("---")

        # SECTION 4: Active Signal Card
        if active_signal:
            _render_active_signal_section(active_signal, metrics)
            st.markdown("")

        # SECTION 5: Position Calculator
        _render_position_calculator_section(selected_ticker, metrics, active_signal)

        st.markdown("---")

        # SECTION 6: Technical + SMC Analysis
        _render_technical_and_smc_analysis(metrics, fundamentals, active_signal)

        st.markdown("---")

        # SECTION 7: Fundamentals
        _render_fundamentals_section(fundamentals)

        st.markdown("---")

        # SECTION 8: Score Breakdown
        if watchlist:
            _render_score_breakdown(watchlist)
            st.markdown("---")

        # SECTION 9: Scanner Signal History
        _render_signal_history(scanner_signals)


def _render_position_calculator_section(ticker: str, metrics: dict, active_signal: dict = None) -> None:
    """
    Render the position calculator section in the chart analysis.

    Uses total account value from session state and either:
    - Active signal's entry/stop if available
    - Current price with ATR-based stop as defaults

    Args:
        ticker: Stock ticker symbol
        metrics: Dict with current_price, atr_percent, etc.
        active_signal: Optional active signal with entry_price, stop_loss
    """
    # Get values from signal if available, otherwise from metrics
    if active_signal:
        suggested_entry = float(active_signal.get('entry_price', 0)) or None
        suggested_stop = float(active_signal.get('stop_loss', 0)) or None
    else:
        suggested_entry = None
        suggested_stop = None

    current_price = float(metrics.get('current_price', 0)) if metrics.get('current_price') else None
    atr_percent = float(metrics.get('atr_percent', 0)) if metrics.get('atr_percent') else None

    # Render the inline calculator
    render_inline_position_calculator(
        ticker=ticker,
        current_price=current_price,
        suggested_entry=suggested_entry,
        suggested_stop=suggested_stop,
        atr_percent=atr_percent
    )


def _get_signal_tickers(service, date_from: str = None, date_to: str = None) -> list:
    """Get unique tickers from active signals, optionally filtered by date range."""
    try:
        signals = service.get_scanner_signals(
            status='active',
            limit=100,
            signal_date_from=date_from,
            signal_date_to=date_to
        )
        if signals:
            tickers = list(set(s.get('ticker') for s in signals if s.get('ticker')))
            return sorted(tickers)
    except Exception:
        pass
    return []


def _get_watchlist_tickers(service, watchlist_type: str, date_from: str = None, date_to: str = None) -> list:
    """Get tickers for a specific watchlist type, optionally filtered by date range."""
    try:
        if watchlist_type == "All Tickers":
            return service.get_all_tickers()

        # Map display names to database watchlist names
        watchlist_name_map = {
            "EMA 50 Cross Over": "ema_50_crossover",
            "EMA 20 Cross Over": "ema_20_crossover",
            "MACD Bullish Cross": "macd_bullish_cross",
            "Gap Up Continuation": "gap_up_continuation",
            "RSI Oversold Bounce": "rsi_oversold_bounce",
        }

        # Crossover watchlists use crossover_date, event watchlists use scan_date
        crossover_watchlists = {"ema_50_crossover", "ema_20_crossover", "macd_bullish_cross"}

        watchlist_name = watchlist_name_map.get(watchlist_type)
        if watchlist_name:
            # get_watchlist_results returns a DataFrame
            df = service.get_watchlist_results(watchlist_name)
            if df is not None and not df.empty and 'ticker' in df.columns:
                # Apply date filter based on watchlist type
                date_col = 'crossover_date' if watchlist_name in crossover_watchlists else 'scan_date'

                if date_col in df.columns and (date_from or date_to):
                    # Convert date column to datetime for comparison
                    df[date_col] = pd.to_datetime(df[date_col])

                    if date_from:
                        date_from_dt = pd.to_datetime(date_from)
                        df = df[df[date_col] >= date_from_dt]

                    if date_to:
                        date_to_dt = pd.to_datetime(date_to)
                        df = df[df[date_col] <= date_to_dt]

                if not df.empty:
                    return sorted(df['ticker'].unique().tolist())
    except Exception:
        pass

    return []


def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMAs and MACD indicators."""
    df = df.copy()

    # Ensure numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate EMAs
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Calculate MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    return df


def _build_charts(df: pd.DataFrame, show_macd: bool, show_volume: bool, smc_data: dict = None, markers_data: dict = None) -> list:
    """Build chart configurations for Lightweight Charts."""
    charts = []

    # Price chart with candlesticks and EMAs
    price_chart = _build_price_chart(df, smc_data, markers_data)
    charts.append(price_chart)

    # MACD chart
    if show_macd:
        macd_chart = _build_macd_chart(df)
        charts.append(macd_chart)

    # Volume chart
    if show_volume:
        volume_chart = _build_volume_chart(df)
        charts.append(volume_chart)

    return charts


def _build_price_chart(df: pd.DataFrame, smc_data: dict = None, markers_data: dict = None) -> dict:
    """Build the main price chart with candlesticks, EMAs, optional SMC overlays, and event markers."""
    # Prepare candle data and build a timestamp->price lookup for markers
    candles = []
    date_to_price = {}  # Map date strings to OHLC data for marker placement
    for row in df.itertuples():
        if pd.notna(row.timestamp):
            ts = int(pd.Timestamp(row.timestamp).timestamp())
            date_str = pd.Timestamp(row.timestamp).strftime('%Y-%m-%d')
            candles.append({
                "time": ts,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close)
            })
            date_to_price[date_str] = {
                "time": ts,
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close)
            }

    # Build series list
    series = [
        {
            "type": "Candlestick",
            "data": candles,
            "options": {
                "upColor": "#26a69a",
                "downColor": "#ef5350",
                "borderVisible": False,
                "wickUpColor": "#26a69a",
                "wickDownColor": "#ef5350"
            }
        }
    ]

    # Add SMC overlays if data available
    if smc_data:
        try:
            smc_series = _build_smc_overlays(df, smc_data)
            if smc_series:
                series.extend(smc_series)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to add SMC overlays: {e}")

    # Add signal and watchlist markers to candlestick series if data available
    event_markers = []
    if markers_data:
        try:
            event_markers = _build_event_markers(markers_data, date_to_price)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to build event markers: {e}")

    # Add markers to the candlestick series if we have any
    if event_markers and series:
        series[0]["markers"] = event_markers

    # Add EMA lines
    ema_configs = [
        ("ema_20", "#2196F3", "EMA 20"),   # Blue
        ("ema_50", "#FF9800", "EMA 50"),   # Orange
        ("ema_100", "#9C27B0", "EMA 100"), # Purple
        ("ema_200", "#F44336", "EMA 200"), # Red
    ]

    for col, color, title in ema_configs:
        if col in df.columns:
            ema_data = []
            for row in df.itertuples():
                val = getattr(row, col, None)
                if pd.notna(row.timestamp) and pd.notna(val):
                    ts = int(pd.Timestamp(row.timestamp).timestamp())
                    ema_data.append({"time": ts, "value": float(val)})

            if ema_data:
                series.append({
                    "type": "Line",
                    "data": ema_data,
                    "options": {
                        "color": color,
                        "lineWidth": 1,
                        "priceLineVisible": False,
                        "lastValueVisible": False,
                        "crosshairMarkerVisible": True
                    }
                })

    # Chart configuration
    chart_config = {
        "height": 400,
        "layout": {
            "background": {"color": "#ffffff"},
            "textColor": "#333333"
        },
        "rightPriceScale": {
            "scaleMargins": {"top": 0.1, "bottom": 0.1},
            "borderVisible": True
        },
        "timeScale": {
            "timeVisible": True,
            "secondsVisible": False,
            "borderVisible": True
        },
        "grid": {
            "vertLines": {"color": "#e0e0e0", "style": 1},
            "horzLines": {"color": "#e0e0e0", "style": 1}
        },
        "crosshair": {
            "mode": 0,
            "vertLine": {"visible": True, "labelVisible": True},
            "horzLine": {"visible": True, "labelVisible": True}
        }
    }

    return {"chart": chart_config, "series": series}


def _build_macd_chart(df: pd.DataFrame) -> dict:
    """Build the MACD indicator chart."""
    macd_line_data = []
    signal_line_data = []
    histogram_data = []

    for row in df.itertuples():
        if pd.notna(row.timestamp):
            ts = int(pd.Timestamp(row.timestamp).timestamp())

            if pd.notna(row.macd_line):
                macd_line_data.append({"time": ts, "value": float(row.macd_line)})

            if pd.notna(row.macd_signal):
                signal_line_data.append({"time": ts, "value": float(row.macd_signal)})

            if pd.notna(row.macd_hist):
                hist_val = float(row.macd_hist)
                histogram_data.append({
                    "time": ts,
                    "value": hist_val,
                    "color": "#26a69a" if hist_val >= 0 else "#ef5350"
                })

    series = [
        {
            "type": "Histogram",
            "data": histogram_data,
            "options": {
                "priceLineVisible": False,
                "lastValueVisible": False
            }
        },
        {
            "type": "Line",
            "data": macd_line_data,
            "options": {
                "color": "#2962FF",
                "lineWidth": 2,
                "priceLineVisible": False,
                "lastValueVisible": False,
                "crosshairMarkerVisible": True
            }
        },
        {
            "type": "Line",
            "data": signal_line_data,
            "options": {
                "color": "#FF6D00",
                "lineWidth": 2,
                "priceLineVisible": False,
                "lastValueVisible": False,
                "crosshairMarkerVisible": True
            }
        }
    ]

    chart_config = {
        "height": 150,
        "layout": {
            "background": {"color": "#ffffff"},
            "textColor": "#333333"
        },
        "rightPriceScale": {
            "scaleMargins": {"top": 0.1, "bottom": 0.1},
            "borderVisible": True
        },
        "timeScale": {
            "timeVisible": True,
            "secondsVisible": False,
            "borderVisible": True,
            "visible": True
        },
        "grid": {
            "vertLines": {"color": "#e0e0e0", "style": 1},
            "horzLines": {"color": "#e0e0e0", "style": 1}
        }
    }

    return {"chart": chart_config, "series": series}


def _build_volume_chart(df: pd.DataFrame) -> dict:
    """Build the volume chart."""
    volume_data = []

    for row in df.itertuples():
        if pd.notna(row.timestamp) and pd.notna(row.volume):
            ts = int(pd.Timestamp(row.timestamp).timestamp())
            vol = float(row.volume)

            # Color based on close vs open
            is_up = row.close >= row.open if pd.notna(row.close) and pd.notna(row.open) else True
            color = "#26a69a" if is_up else "#ef5350"

            volume_data.append({
                "time": ts,
                "value": vol,
                "color": color
            })

    series = [
        {
            "type": "Histogram",
            "data": volume_data,
            "options": {
                "priceLineVisible": False,
                "lastValueVisible": False
            }
        }
    ]

    chart_config = {
        "height": 100,
        "layout": {
            "background": {"color": "#ffffff"},
            "textColor": "#333333"
        },
        "rightPriceScale": {
            "scaleMargins": {"top": 0.1, "bottom": 0.1},
            "borderVisible": True
        },
        "timeScale": {
            "timeVisible": True,
            "secondsVisible": False,
            "borderVisible": True,
            "visible": True
        },
        "grid": {
            "vertLines": {"color": "#e0e0e0", "style": 1},
            "horzLines": {"color": "#e0e0e0", "style": 1}
        }
    }

    return {"chart": chart_config, "series": series}


# =============================================================================
# SMC (Smart Money Concepts) Visualization
# =============================================================================

def _render_smc_summary(smc_data: dict) -> None:
    """Render SMC analysis summary above the chart."""
    if not smc_data:
        return

    trend = smc_data.get('trend', 'Unknown')
    bias = smc_data.get('bias', 'Unknown')
    confluence_score = smc_data.get('confluence_score', 0)
    zone = smc_data.get('premium_discount_zone', 'Unknown')
    order_blocks = smc_data.get('order_blocks', [])
    bos_choch_events = smc_data.get('bos_choch_events', [])

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("SMC Trend", trend)

    with col2:
        st.metric("Bias", bias)

    with col3:
        st.metric("Zone", zone)

    with col4:
        st.metric("Confluence", f"{confluence_score:.0f}" if confluence_score else "N/A")

    with col5:
        st.metric("Order Blocks", len(order_blocks) if order_blocks else 0)

    # Show recent BOS/CHOCH events
    if bos_choch_events:
        recent_events = bos_choch_events[:3]
        events_text = " | ".join([f"{e.get('event_type', 'BOS')} @ ${e.get('price', 0):.2f}" for e in recent_events])
        st.caption(f"Recent Events: {events_text}")


def _build_smc_overlays(df: pd.DataFrame, smc_data: dict) -> list:
    """
    Build SMC overlay series for the chart.

    Adds order block zones as horizontal lines to visualize supply/demand areas.
    Uses multiple data points along the line to ensure proper rendering.
    """
    series = []

    if not smc_data:
        return series

    # Get the chart's time range
    if df.empty:
        return series

    try:
        # Ensure we have valid timestamps
        timestamps = df['timestamp'].dropna()
        if timestamps.empty:
            return series

        # Get all timestamps from the dataframe for the horizontal lines
        # Using actual candle timestamps ensures proper rendering
        time_points = []
        for ts in timestamps:
            try:
                time_val = int(pd.Timestamp(ts).timestamp())
                if time_val > 0:
                    time_points.append(time_val)
            except (ValueError, TypeError):
                continue

        if len(time_points) < 2:
            return series

        # Sort and deduplicate time points
        time_points = sorted(set(time_points))

        # Sample every 5th point to reduce data size while keeping line smooth
        if len(time_points) > 20:
            sampled_times = time_points[::5]
            # Always include first and last
            if time_points[0] not in sampled_times:
                sampled_times.insert(0, time_points[0])
            if time_points[-1] not in sampled_times:
                sampled_times.append(time_points[-1])
            time_points = sorted(sampled_times)

        # Add Order Blocks as horizontal line series
        order_blocks = smc_data.get('order_blocks', [])

        if order_blocks:
            for i, ob in enumerate(order_blocks[:6]):  # Limit to 6 order blocks (3 zones * 2 lines)
                ob_type = ob.get('type', 'bullish')

                # Safely get and validate price values
                try:
                    price_high = ob.get('price_high')
                    price_low = ob.get('price_low')

                    if price_high is None or price_low is None:
                        continue

                    price_top = float(price_high)
                    price_bottom = float(price_low)

                    # Validate prices are positive and reasonable
                    if price_top <= 0 or price_bottom <= 0:
                        continue
                    if pd.isna(price_top) or pd.isna(price_bottom):
                        continue

                except (ValueError, TypeError):
                    continue

                # Color based on type (bullish = demand zone, bearish = supply zone)
                # Use solid colors (not semi-transparent) for better Linux compatibility
                line_color = '#26a69a' if ob_type.lower() == 'bullish' else '#ef5350'

                # Build horizontal line data with multiple points (more robust rendering)
                ob_top_data = [{"time": t, "value": price_top} for t in time_points]
                ob_bottom_data = [{"time": t, "value": price_bottom} for t in time_points]

                # Add top line (supply/demand zone upper boundary)
                # Note: Removed lineStyle to avoid Linux rendering issues
                series.append({
                    "type": "Line",
                    "data": ob_top_data,
                    "options": {
                        "color": line_color,
                        "lineWidth": 1,
                        "priceLineVisible": False,
                        "lastValueVisible": False
                    }
                })

                # Add bottom line (supply/demand zone lower boundary)
                series.append({
                    "type": "Line",
                    "data": ob_bottom_data,
                    "options": {
                        "color": line_color,
                        "lineWidth": 1,
                        "priceLineVisible": False,
                        "lastValueVisible": False
                    }
                })

    except Exception as e:
        # Log error but don't crash the chart
        import logging
        logging.getLogger(__name__).warning(f"Error building SMC overlays: {e}")
        return []

    return series


# =============================================================================
# Signal & Watchlist Event Markers
# =============================================================================

def _render_markers_summary(markers_data: dict) -> None:
    """Render summary of signal and watchlist events above the chart."""
    if not markers_data:
        return

    signals = markers_data.get('signals', [])
    watchlist_events = markers_data.get('watchlist_events', [])

    # Count unique events
    signal_count = len(signals)
    watchlist_count = len(watchlist_events)

    # Get unique watchlist names with their event dates
    watchlist_info = {}
    for event in watchlist_events:
        wl_name = event.get('watchlist_name', '')
        event_date = event.get('event_date') or event.get('crossover_date') or event.get('scan_date')
        if wl_name:
            display_name = wl_name.replace('_', ' ').title()
            date_str = str(event_date)[:10] if event_date else ''
            if display_name not in watchlist_info:
                watchlist_info[display_name] = date_str

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Signals", signal_count)

    with col2:
        st.metric("Watchlist Events", watchlist_count)

    with col3:
        if watchlist_info:
            # Show watchlist names with dates
            wl_texts = [f"{name} ({date})" for name, date in watchlist_info.items()]
            st.caption(f"Watchlists: {', '.join(sorted(wl_texts))}")
        else:
            st.caption("No watchlist events")

    # Show recent signal details
    if signals:
        recent_signals = signals[:3]
        signal_texts = []
        for sig in recent_signals:
            sig_type = sig.get('signal_type', '?')
            tier = sig.get('quality_tier', '')
            date = sig.get('event_date')
            date_str = str(date)[:10] if date else ''
            signal_texts.append(f"{sig_type} ({tier}) {date_str}")
        st.caption(f"Recent Signals: {' | '.join(signal_texts)}")


def _build_event_markers(markers_data: dict, date_to_price: dict) -> list:
    """
    Build markers list for signals and watchlist events.

    Returns a list of marker dicts that can be attached to the candlestick series.
    This follows the pattern used in tvchart.py for trade markers.

    Signals appear as triangles (up for BUY, down for SELL).
    Watchlist events appear as circles with different colors per watchlist type.
    """
    markers = []

    if not markers_data or not date_to_price:
        return markers

    # Build signal markers - shown as arrows above/below candles
    signals = markers_data.get('signals', [])
    for sig in signals:
        event_date = sig.get('event_date')
        if not event_date:
            continue

        date_str = str(event_date)[:10]
        price_data = date_to_price.get(date_str)
        if not price_data:
            continue

        sig_type = sig.get('signal_type', 'BUY')
        tier = sig.get('quality_tier', '')
        scanner = sig.get('scanner_name', '')

        # Build label: "BUY A+" or "SELL B"
        label = f"{sig_type}"
        if tier:
            label += f" {tier}"

        if sig_type == 'BUY':
            markers.append({
                "time": price_data['time'],
                "position": "belowBar",
                "color": "#00C853",  # Green
                "shape": "arrowUp",
                "text": label
            })
        else:
            markers.append({
                "time": price_data['time'],
                "position": "aboveBar",
                "color": "#FF1744",  # Red
                "shape": "arrowDown",
                "text": label
            })

    # Build watchlist event markers - shown as circles
    watchlist_events = markers_data.get('watchlist_events', [])
    if watchlist_events:
        watchlist_colors = {
            'ema_50_crossover': '#2196F3',      # Blue
            'ema_20_crossover': '#9C27B0',      # Purple
            'macd_bullish_cross': '#FF9800',    # Orange
            'gap_up_continuation': '#4CAF50',   # Green
            'rsi_oversold_bounce': '#E91E63',   # Pink
        }

        watchlist_labels = {
            'ema_50_crossover': 'E50',
            'ema_20_crossover': 'E20',
            'macd_bullish_cross': 'MCD',
            'gap_up_continuation': 'GAP',
            'rsi_oversold_bounce': 'RSI',
        }

        # Track which (watchlist, date) combos we've processed to avoid duplicates
        processed = set()

        for event in watchlist_events:
            wl_name = event.get('watchlist_name', '')
            crossover_date = event.get('crossover_date')
            scan_date = event.get('scan_date')

            # Use crossover_date for crossover watchlists, scan_date for event watchlists
            if wl_name in ['ema_50_crossover', 'ema_20_crossover', 'macd_bullish_cross']:
                event_date = crossover_date or scan_date
            else:
                event_date = scan_date

            if not event_date:
                continue

            date_str = str(event_date)[:10]
            key = (wl_name, date_str)

            # Skip duplicates
            if key in processed:
                continue
            processed.add(key)

            price_data = date_to_price.get(date_str)
            if not price_data:
                continue

            color = watchlist_colors.get(wl_name, '#9E9E9E')
            label = watchlist_labels.get(wl_name, wl_name[:3].upper())

            markers.append({
                "time": price_data['time'],
                "position": "inBar",
                "color": color,
                "shape": "circle",
                "text": label
            })

    return markers


# =============================================================================
# Multi-Timeframe Charts
# =============================================================================

def _render_mtf_charts(service, ticker: str, daily_df: pd.DataFrame, lookback_days: int, show_macd: bool, show_volume: bool, smc_data: dict = None, markers_data: dict = None) -> None:
    """
    Render multi-timeframe chart layout with Daily and Weekly views plus trend alignment.
    """
    # Fetch weekly data
    with st.spinner("Loading multi-timeframe data..."):
        weekly_df = service.get_weekly_candles(ticker, weeks=52)

    if weekly_df is None or weekly_df.empty:
        st.warning("Weekly data not available - showing daily chart only")
        try:
            charts = _build_charts(daily_df, show_macd, show_volume, smc_data, markers_data)
            renderLightweightCharts(charts, f"stock-chart-{ticker}")
        except Exception as e:
            if smc_data or markers_data:
                st.warning("Overlays caused a rendering issue. Showing chart without extra overlays.")
                charts = _build_charts(daily_df, show_macd, show_volume, None, None)
                renderLightweightCharts(charts, f"stock-chart-{ticker}-fallback")
        return

    # Calculate indicators for weekly
    weekly_df = _calculate_indicators(weekly_df)
    weekly_df = weekly_df.tail(int(lookback_days / 5)).reset_index(drop=True)  # Approximate weeks

    # Create two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Daily Chart")
        try:
            daily_charts = _build_charts(daily_df, show_macd, show_volume, smc_data, markers_data)
            renderLightweightCharts(daily_charts, f"stock-chart-daily-{ticker}")
        except Exception as e:
            if smc_data or markers_data:
                st.warning("Overlays caused a rendering issue. Showing chart without extra overlays.")
                daily_charts = _build_charts(daily_df, show_macd, show_volume, None, None)
                renderLightweightCharts(daily_charts, f"stock-chart-daily-{ticker}-fallback")

    with col2:
        st.markdown("#### Weekly Chart")
        weekly_charts = _build_charts(weekly_df, False, False, None, None)  # Simplified weekly chart
        renderLightweightCharts(weekly_charts, f"stock-chart-weekly-{ticker}")

    # Trend Alignment Summary
    st.markdown("---")
    _render_trend_alignment(daily_df, weekly_df, ticker)


def _render_trend_alignment(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, ticker: str) -> None:
    """Render trend alignment indicator across timeframes."""
    st.markdown("#### Trend Alignment")

    # Calculate trend direction for each timeframe
    def get_trend(df: pd.DataFrame, ema_col: str) -> str:
        if df.empty or ema_col not in df.columns:
            return "Unknown"
        last_price = df['close'].iloc[-1] if not df.empty else 0
        last_ema = df[ema_col].iloc[-1] if ema_col in df.columns and not pd.isna(df[ema_col].iloc[-1]) else 0
        if last_price > last_ema * 1.02:
            return "Bullish"
        elif last_price < last_ema * 0.98:
            return "Bearish"
        else:
            return "Neutral"

    # Daily trends
    daily_ema20_trend = get_trend(daily_df, 'ema_20')
    daily_ema50_trend = get_trend(daily_df, 'ema_50')
    daily_ema200_trend = get_trend(daily_df, 'ema_200')

    # Weekly trends
    weekly_ema20_trend = get_trend(weekly_df, 'ema_20')
    weekly_ema50_trend = get_trend(weekly_df, 'ema_50')

    # Trend colors
    def trend_color(trend: str) -> str:
        return 'green' if trend == 'Bullish' else ('red' if trend == 'Bearish' else 'orange')

    def trend_icon(trend: str) -> str:
        return '↑' if trend == 'Bullish' else ('↓' if trend == 'Bearish' else '→')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        tc = trend_color(daily_ema20_trend)
        st.markdown(f"**Daily EMA20:** :{tc}[{trend_icon(daily_ema20_trend)} {daily_ema20_trend}]")

    with col2:
        tc = trend_color(daily_ema50_trend)
        st.markdown(f"**Daily EMA50:** :{tc}[{trend_icon(daily_ema50_trend)} {daily_ema50_trend}]")

    with col3:
        tc = trend_color(daily_ema200_trend)
        st.markdown(f"**Daily EMA200:** :{tc}[{trend_icon(daily_ema200_trend)} {daily_ema200_trend}]")

    with col4:
        tc = trend_color(weekly_ema20_trend)
        st.markdown(f"**Weekly EMA20:** :{tc}[{trend_icon(weekly_ema20_trend)} {weekly_ema20_trend}]")

    with col5:
        tc = trend_color(weekly_ema50_trend)
        st.markdown(f"**Weekly EMA50:** :{tc}[{trend_icon(weekly_ema50_trend)} {weekly_ema50_trend}]")

    # Overall alignment score
    trends = [daily_ema20_trend, daily_ema50_trend, daily_ema200_trend, weekly_ema20_trend, weekly_ema50_trend]
    bullish_count = sum(1 for t in trends if t == 'Bullish')
    bearish_count = sum(1 for t in trends if t == 'Bearish')

    if bullish_count >= 4:
        st.success(f"**{ticker}**: Strong bullish alignment ({bullish_count}/5 timeframes bullish)")
    elif bearish_count >= 4:
        st.error(f"**{ticker}**: Strong bearish alignment ({bearish_count}/5 timeframes bearish)")
    elif bullish_count >= 3:
        st.info(f"**{ticker}**: Moderate bullish bias ({bullish_count}/5 timeframes bullish)")
    elif bearish_count >= 3:
        st.warning(f"**{ticker}**: Moderate bearish bias ({bearish_count}/5 timeframes bearish)")
    else:
        st.caption(f"**{ticker}**: Mixed signals - no clear alignment")
