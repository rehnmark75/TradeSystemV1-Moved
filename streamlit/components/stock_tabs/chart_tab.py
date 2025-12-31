"""
Stock Scanner Chart Tab

Interactive stock charting with:
- Stock selection from signals, watchlists, or manual entry
- Lightweight Charts (TradingView-like) visualization
- EMAs (20, 50, 100, 200)
- MACD with histogram
- Volume subplot
- 90-day default lookback
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_lightweight_charts_ntf import renderLightweightCharts


def render_chart_tab(service):
    """
    Render the Chart tab with interactive stock charting.

    Features:
    - Stock selection from signals, watchlists, or manual entry
    - Candlestick chart with EMAs (20, 50, 100, 200)
    - MACD indicator panel
    - Volume panel
    """
    st.header("Stock Chart")
    st.markdown("*Interactive charting with technical indicators*")

    # Stock Selection Section
    st.subheader("Select Stock")

    source_col, ticker_col = st.columns([1, 2])

    with source_col:
        source = st.radio(
            "Source",
            ["Signals", "Watchlist", "Manual Entry"],
            horizontal=True,
            key="chart_source"
        )

    selected_ticker = None

    with ticker_col:
        if source == "Signals":
            # Get active signal tickers
            signal_tickers = _get_signal_tickers(service)
            if signal_tickers:
                selected_ticker = st.selectbox(
                    "Select from active signals",
                    signal_tickers,
                    key="chart_signal_ticker"
                )
            else:
                st.info("No active signals available")

        elif source == "Watchlist":
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

            # Get tickers for selected watchlist
            watchlist_tickers = _get_watchlist_tickers(service, watchlist_type)
            if watchlist_tickers:
                selected_ticker = st.selectbox(
                    "Select ticker",
                    watchlist_tickers,
                    key="chart_watchlist_ticker"
                )
            else:
                st.info(f"No tickers in {watchlist_type} watchlist")

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
        st.info("Select a stock to view chart")
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

    # Build and render charts
    charts = _build_charts(df, show_macd, show_volume)

    # Render with unique key
    renderLightweightCharts(charts, f"stock-chart-{selected_ticker}")

    # Display current price info
    _render_price_info(df, selected_ticker)


def _get_signal_tickers(service) -> list:
    """Get unique tickers from active signals."""
    try:
        signals = service.get_scanner_signals(status='active', limit=100)
        if signals:
            tickers = list(set(s.get('ticker') for s in signals if s.get('ticker')))
            return sorted(tickers)
    except Exception:
        pass
    return []


def _get_watchlist_tickers(service, watchlist_type: str) -> list:
    """Get tickers for a specific watchlist type."""
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

        watchlist_name = watchlist_name_map.get(watchlist_type)
        if watchlist_name:
            # get_watchlist_results returns a DataFrame
            df = service.get_watchlist_results(watchlist_name)
            if df is not None and not df.empty and 'ticker' in df.columns:
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


def _build_charts(df: pd.DataFrame, show_macd: bool, show_volume: bool) -> list:
    """Build chart configurations for Lightweight Charts."""
    charts = []

    # Price chart with candlesticks and EMAs
    price_chart = _build_price_chart(df)
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


def _build_price_chart(df: pd.DataFrame) -> dict:
    """Build the main price chart with candlesticks and EMAs."""
    # Prepare candle data
    candles = []
    for row in df.itertuples():
        if pd.notna(row.timestamp):
            ts = int(pd.Timestamp(row.timestamp).timestamp())
            candles.append({
                "time": ts,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close)
            })

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
                        "title": title,
                        "priceLineVisible": False,
                        "lastValueVisible": False
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
                "title": "MACD",
                "priceLineVisible": False
            }
        },
        {
            "type": "Line",
            "data": signal_line_data,
            "options": {
                "color": "#FF6D00",
                "lineWidth": 2,
                "title": "Signal",
                "priceLineVisible": False
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


def _render_price_info(df: pd.DataFrame, ticker: str):
    """Render current price information below the chart."""
    if df.empty:
        return

    latest = df.iloc[-1]

    st.markdown("---")
    st.subheader("Current Data")

    # OHLCV row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Open", f"${latest['open']:.2f}")
    with col2:
        st.metric("High", f"${latest['high']:.2f}")
    with col3:
        st.metric("Low", f"${latest['low']:.2f}")
    with col4:
        st.metric("Close", f"${latest['close']:.2f}")
    with col5:
        change = latest['close'] - latest['open']
        change_pct = (change / latest['open'] * 100) if latest['open'] != 0 else 0
        st.metric("Change", f"${change:+.2f}", f"{change_pct:+.2f}%")
    with col6:
        vol = latest.get('volume', 0)
        if vol >= 1_000_000:
            vol_str = f"{vol/1_000_000:.1f}M"
        elif vol >= 1_000:
            vol_str = f"{vol/1_000:.1f}K"
        else:
            vol_str = f"{vol:.0f}"
        st.metric("Volume", vol_str)

    # EMA values row
    st.markdown("**Moving Averages**")
    ema_col1, ema_col2, ema_col3, ema_col4 = st.columns(4)

    with ema_col1:
        if pd.notna(latest.get('ema_20')):
            st.metric("EMA 20", f"${latest['ema_20']:.2f}")
    with ema_col2:
        if pd.notna(latest.get('ema_50')):
            st.metric("EMA 50", f"${latest['ema_50']:.2f}")
    with ema_col3:
        if pd.notna(latest.get('ema_100')):
            st.metric("EMA 100", f"${latest['ema_100']:.2f}")
    with ema_col4:
        if pd.notna(latest.get('ema_200')):
            st.metric("EMA 200", f"${latest['ema_200']:.2f}")

    # MACD values row
    st.markdown("**MACD Indicator**")
    macd_col1, macd_col2, macd_col3 = st.columns(3)

    with macd_col1:
        if pd.notna(latest.get('macd_line')):
            st.metric("MACD Line", f"{latest['macd_line']:.4f}")
    with macd_col2:
        if pd.notna(latest.get('macd_signal')):
            st.metric("Signal Line", f"{latest['macd_signal']:.4f}")
    with macd_col3:
        if pd.notna(latest.get('macd_hist')):
            hist_val = latest['macd_hist']
            hist_trend = "Bullish" if hist_val > 0 else "Bearish"
            st.metric("Histogram", f"{hist_val:.4f}", hist_trend)
