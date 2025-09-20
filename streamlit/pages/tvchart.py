import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
import os
from sqlalchemy import create_engine
from streamlit_lightweight_charts_ntf import renderLightweightCharts

from services.data import get_candle_data, get_trade_logs, get_epics
from services.indicators import apply_indicators, calculate_two_pole_oscillator, calculate_zero_lag_ema, calculate_support_resistance
# Use the simple service directly for now since worker files aren't available in container
from services.ema_config_simple import get_ema_config_service_simple as get_ema_config_service, get_ema_config_summary_simple as get_ema_config_summary
from services.smc_structure import (
    detect_pivots, classify_pivots, convert_swings_to_plot_shapes
)
from services.market_intelligence_service import get_intelligence_service
from utils.helpers import synthesize_15m_from_5m

st.set_page_config(page_title="TradingView Candlestick Viewer", layout="wide")
LOCAL_TZ = pytz.timezone("UTC")
DISPLAY_CANDLES = 1000

@st.cache_data
def robust_clean_candle_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    numeric_cols = ["open", "high", "low", "close"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=numeric_cols, inplace=True)
    df[numeric_cols] = df[numeric_cols].astype("float64")

    is_flat = (
        np.isclose(df["open"], df["high"]) &
        np.isclose(df["high"], df["low"]) &
        np.isclose(df["low"], df["close"])
    )
    df = df[~is_flat]

    df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("UTC")
    df.sort_values("start_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Database connection
engine = create_engine(os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex"))
epics = get_epics(engine)

# Sidebar controls
st.sidebar.header("Chart Settings")
selected_epic = st.sidebar.selectbox("Select Symbol (Epic)", epics)

# Load dynamic EMA configuration for selected epic
ema_config = None
ema_config_summary = None

try:
    ema_config_service = get_ema_config_service()
    ema_config = ema_config_service.get_ema_config(selected_epic)
    ema_config_summary = get_ema_config_summary(selected_epic)

    # Cache the configuration for use in indicators
    st.session_state.current_ema_config = ema_config
    st.session_state.current_ema_summary = ema_config_summary

    # Debug info
    st.sidebar.success(f"✅ Config loaded: {ema_config.source} - {ema_config.short_period}/{ema_config.long_period}/{ema_config.trend_period}")

except Exception as e:
    st.sidebar.error(f"Error loading EMA config: {str(e)}")
    import traceback
    st.sidebar.text("Full error:")
    st.sidebar.code(traceback.format_exc())
    # Fallback to default values
    ema_config = None
    ema_config_summary = None

# Timeframe selector for both 5m and 15m
timeframe_options = ["5m", "15m"]
selected_tf = st.sidebar.radio("Chart Timeframe", timeframe_options, index=1)
timeframe_minutes = {"5m": 5, "15m": 15}
timeframe = timeframe_minutes[selected_tf]

# EMA toggles - now shows actual periods for selected epic
st.sidebar.subheader("EMA Indicators")

# Get actual EMA periods to display
if ema_config:
    short_label = f"Show EMA {ema_config.short_period}"
    long_label = f"Show EMA {ema_config.long_period}"
    trend_label = f"Show EMA {ema_config.trend_period}"

    # Show configuration source
    if ema_config.source == "optimal":
        config_status = "🎯 Optimized"
    elif ema_config.source == "config":
        config_status = "⚙️ Configured"
    else:
        config_status = "📋 Default"

    st.sidebar.caption(f"Configuration: {config_status}")
else:
    # Fallback labels
    short_label = "Show EMA 21 (default)"
    long_label = "Show EMA 50 (default)"
    trend_label = "Show EMA 200 (default)"

show_ema21 = st.sidebar.checkbox(short_label, value=True)
show_ema50 = st.sidebar.checkbox(long_label, value=True)
show_ema200 = st.sidebar.checkbox(trend_label, value=True)

# Add configuration details section
with st.sidebar.expander("🔧 EMA Configuration Details", expanded=False):
    if ema_config and ema_config_summary:
        st.write("**Current Configuration:**")
        st.write(f"• Short EMA: {ema_config.short_period}")
        st.write(f"• Long EMA: {ema_config.long_period}")
        st.write(f"• Trend EMA: {ema_config.trend_period}")
        st.write(f"• Source: {ema_config.source.title()}")
        st.write(f"• Config: {ema_config.config_name}")

        if ema_config.description:
            st.caption(f"*{ema_config.description}*")

        # Performance metrics if available
        if 'performance' in ema_config_summary:
            perf = ema_config_summary['performance']
            st.write("**Performance Metrics:**")
            if perf.get('performance_score'):
                st.write(f"• Score: {perf['performance_score']:.3f}")
            if perf.get('confidence_threshold'):
                st.write(f"• Confidence: {perf['confidence_threshold']:.1%}")
            if perf.get('risk_reward_ratio'):
                st.write(f"• Risk/Reward: {perf['risk_reward_ratio']:.2f}")

        # Last updated info
        if ema_config.last_updated:
            st.caption(f"Last updated: {ema_config.last_updated.strftime('%Y-%m-%d %H:%M')}")

        # Refresh button
        if st.button("🔄 Refresh Config"):
            ema_config_service.refresh_cache(selected_epic)
            st.rerun()

    else:
        st.write("**Using Default Configuration**")
        st.write("• Short EMA: 21")
        st.write("• Long EMA: 50")
        st.write("• Trend EMA: 200")
        st.caption("*Standard EMA periods*")

# Zero Lag EMA toggle
st.sidebar.subheader("Zero Lag EMA")
show_zlema = st.sidebar.checkbox("Show Zero Lag EMA", value=False)
show_zlema_bands = st.sidebar.checkbox("Show ZLEMA Bands", value=False)

# MACD toggle
st.sidebar.subheader("MACD Indicator")
show_macd = st.sidebar.checkbox("Show MACD", value=False)

# Two-Pole Oscillator toggle
st.sidebar.subheader("Two-Pole Oscillator")
show_two_pole = st.sidebar.checkbox("Show Two-Pole Oscillator", value=False)

# Support/Resistance toggle
st.sidebar.subheader("Support & Resistance")
show_sr_levels = st.sidebar.checkbox("Show S/R Levels", value=False)
show_sr_breaks = st.sidebar.checkbox("Show Break Markers", value=False)
if show_sr_levels or show_sr_breaks:
    sr_left_bars = st.sidebar.number_input("Left Bars", value=15, min_value=5, max_value=50, step=1)
    sr_right_bars = st.sidebar.number_input("Right Bars", value=15, min_value=5, max_value=50, step=1)
    sr_volume_thresh = st.sidebar.slider("Volume Threshold %", 0, 50, 20)
    sr_max_levels = st.sidebar.number_input("Max Levels", value=5, min_value=1, max_value=10, step=1)
else:
    # Default values when not shown
    sr_left_bars = 15
    sr_right_bars = 15
    sr_volume_thresh = 20
    sr_max_levels = 5

# Market Intelligence toggle
st.sidebar.subheader("🧠 Market Intelligence")
show_market_regime = st.sidebar.checkbox("Show Market Regime", value=False)
show_session_analysis = st.sidebar.checkbox("Show Session Info", value=True)
show_trade_context = st.sidebar.checkbox("Analyze Trade Context", value=False)
if show_market_regime or show_trade_context:
    intelligence_lookback = st.sidebar.slider("Analysis Lookback (hours)", 6, 48, 24, step=2)
else:
    intelligence_lookback = 24

# Trade Markers toggle
st.sidebar.subheader("📈 Trade Markers")
show_stop_levels = st.sidebar.checkbox("Show Stop Loss Levels", value=False)
show_tp_levels = st.sidebar.checkbox("Show Take Profit Levels", value=False)
show_trailing_info = st.sidebar.checkbox("Show Trailing/Breakeven", value=False)

# Other settings
lookback_days = st.sidebar.slider("Lookback period (days)", 1, 5, 2)
candles_per_day = (24 * 60) // timeframe
lookback_candles = lookback_days * candles_per_day

refresh_minutes = st.sidebar.selectbox("Auto-refresh interval (minutes)", [0, 1, 2, 5, 10], index=2)
show_swings = st.sidebar.checkbox("Show Swing Labels", value=False)

st.title("📈 TradingView Style Chart")
st.caption(f"Showing {selected_epic} - {selected_tf} timeframe (UTC)")

# Display EMA configuration summary prominently
if ema_config:
    config_color = {
        'optimal': 'success',
        'config': 'info',
        'default': 'secondary'
    }.get(ema_config.source, 'secondary')

    config_icon = {
        'optimal': '🎯',
        'config': '⚙️',
        'default': '📋'
    }.get(ema_config.source, '📋')

    config_message = (
        f"{config_icon} **EMA Configuration:** "
        f"{ema_config.short_period}/{ema_config.long_period}/{ema_config.trend_period} "
        f"({ema_config.source.title()} - {ema_config.config_name})"
    )

    if config_color == 'success':
        st.success(config_message)
    elif config_color == 'info':
        st.info(config_message)
    else:
        st.info(config_message)

# Add legend for trade markers
with st.expander("📊 Chart Legend", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **Trade Arrows:**
        - ⬆️ Arrow Up = BUY signal
        - ⬇️ Arrow Down = SELL signal
        """)
    with col2:
        st.markdown("""
        **Arrow Colors:**
        - 🟢 Green = Profitable trade
        - 🔴 Red = Loss trade
        - 🟡 Yellow = Still running
        - ⚫ Gray = Breakeven
        """)
    with col3:
        st.markdown("""
        **Risk Management:**
        - 🔴 Circle = Stop Loss Level
        - 🟢 Circle = Take Profit Level
        - 🟡 Circle = Moved to Breakeven
        - 🟣 Circle = Trailing Active
        """)
    with col4:
        ema_periods = (ema_config.short_period, ema_config.long_period, ema_config.trend_period) if ema_config else (21, 50, 200)
        st.markdown(f"""
        **EMA Indicators:**
        - 🟢 Green Line = EMA {ema_periods[0]}
        - 🟠 Orange Line = EMA {ema_periods[1]}
        - 🔵 Blue Line = EMA {ema_periods[2]}
        - *Periods adapt per epic*
        """)

# Fetch candle data based on selected timeframe
if selected_tf == "5m":
    df_raw = get_candle_data(engine, 5, selected_epic, limit=lookback_candles * 2)
    df_full = robust_clean_candle_data(df_raw)
else:  # 15m
    # Try to get 15m candles directly
    df_raw = get_candle_data(engine, 15, selected_epic, limit=lookback_candles * 2)
    df_full = robust_clean_candle_data(df_raw)
    
    # If not enough 15m candles, synthesize from 5m
    if len(df_full) < lookback_candles:
        df_full = synthesize_15m_from_5m(engine, selected_epic, lookback_candles)

# Display subset of candles
df_display = df_full.tail(DISPLAY_CANDLES).reset_index(drop=True)
if df_display.empty:
    st.warning("No candles available to display.")
    st.stop()

# Get the earliest timestamp for trade data
earliest_time = df_full["start_time"].min() if not df_full.empty else pd.Timestamp.now(tz=LOCAL_TZ)

# Determine precision based on the selected epic (needed for chart configuration)
if "JPY" in selected_epic:
    price_precision = 3
else:
    price_precision = 5

# Apply indicators based on toggles
indicators_to_apply = []
if show_ema21:
    indicators_to_apply.append("EMA21")
if show_ema50:
    indicators_to_apply.append("EMA50")
if show_ema200:
    indicators_to_apply.append("EMA200")
if show_macd:
    indicators_to_apply.append("MACD")

# Apply indicators with dynamic EMA configuration
if ema_config:
    df_display = apply_indicators(
        df_display,
        ema_short=ema_config.short_period,
        ema_long=ema_config.long_period,
        ema_trend=ema_config.trend_period,
        epic=selected_epic,
        indicators=indicators_to_apply
    )
else:
    # Fallback to legacy behavior
    df_display = apply_indicators(df_display, indicators=indicators_to_apply)

# Calculate Two-Pole Oscillator if enabled
if show_two_pole:
    df_display = calculate_two_pole_oscillator(df_display)

# Calculate Zero Lag EMA if enabled
if show_zlema or show_zlema_bands:
    df_display = calculate_zero_lag_ema(df_display)

# Calculate Support/Resistance levels if enabled
sr_data = None
if show_sr_levels or show_sr_breaks:
    sr_data = calculate_support_resistance(
        df_display,
        left_bars=sr_left_bars,
        right_bars=sr_right_bars,
        volume_threshold=sr_volume_thresh,
        max_levels=sr_max_levels
    )

# Get Market Intelligence analysis if enabled
intelligence_report = None
session_analysis = None
regime_data = None

if show_market_regime or show_session_analysis or show_trade_context:
    try:
        intelligence_service = get_intelligence_service()
        
        # Set up the data fetcher with the database engine
        intelligence_service.data_fetcher.engine = engine
        
        if show_session_analysis:
            session_analysis = intelligence_service.get_session_analysis()
        
        if show_market_regime or show_trade_context:
            # Get regime analysis for current epic
            regime_data = intelligence_service.get_regime_for_timeframe(
                selected_epic, 
                f"{timeframe}m" if timeframe < 60 else "1h",
                intelligence_lookback
            )
            
            # Get full intelligence report if analyzing multiple aspects
            if show_trade_context:
                epic_list = [selected_epic]  # Could expand to analyze related pairs
                intelligence_report = intelligence_service.get_market_intelligence_report(
                    epic_list, 
                    intelligence_lookback
                )
    except Exception as e:
        st.sidebar.error(f"Intelligence analysis error: {str(e)[:100]}...")
        intelligence_report = None
        session_analysis = None
        regime_data = None

# Get trade logs with strategy information
trades_df = get_trade_logs(engine, selected_epic, earliest_time)

# Debug: Show trade data info
if not trades_df.empty:
    st.sidebar.write(f"Found {len(trades_df)} trades")
else:
    st.sidebar.write("No trades found for this symbol")

# Prepare candle data for chart with proper precision
candles = [
    {
        "time": int(row.start_time.timestamp()),  # Must be UNIX timestamp (in seconds)
        "open": float(row.open),
        "high": float(row.high),
        "low": float(row.low),
        "close": float(row.close)
    }
    for row in df_display.itertuples()
]

# Debug: Show sample data precision to verify we have full precision
if candles and selected_epic:
    sample_candle = candles[-1]
    st.sidebar.write(f"Data Precision Check:")
    st.sidebar.write(f"Raw DB Close: {df_display.iloc[-1]['close']}")
    st.sidebar.write(f"Chart Close: {sample_candle['close']}")
    st.sidebar.write(f"Full precision: {sample_candle['close']:.10f}")

# Prepare trade markers with strategy labels
trade_markers = []
stop_loss_markers = []
take_profit_markers = []
trailing_markers = []

for row in trades_df.itertuples():
    # Handle timezone for trade timestamp - keep in UTC
    trade_time = pd.Timestamp(row.timestamp)
    if trade_time.tz is None:
        trade_time = trade_time.tz_localize('UTC')
    else:
        trade_time = trade_time.tz_convert('UTC')

    timestamp = int(trade_time.timestamp())  # ensure it's in seconds
    direction = row.direction.upper() if row.direction else "UNKNOWN"

    # Get strategy name, use short version if available
    strategy_name = ""
    if hasattr(row, 'strategy') and row.strategy:
        # Shorten common strategy names for better display
        strategy_map = {
            "enhanced_ema_price_crossover": "EMA",
            "enhanced_ema_ema_crossover": "EMA-X",
            "enhanced_ema_simple_forex": "EMA-S",
            "zero_lag_ema_entry": "ZeroLag",
            "zero_lag_ema Entry": "ZeroLag",
            "zero_lag_squeeze": "ZL-Squeeze",
            "macd": "MACD",
            "MACD_MODULAR": "MACD-M",
            "macd_ema200": "MACD-200",
            "kama": "KAMA",
            "bollinger_supertrend": "BB-ST",
            "combined": "Combined",
            "combined_dynamic_all": "Combo-All",
            "combined_weighted_1": "Combo-W1",
            "combined_weighted_2": "Combo-W2",
            "ema": "EMA"
        }
        strategy_name = strategy_map.get(row.strategy, row.strategy[:10])  # Limit to 10 chars if not mapped

    # Determine if it's a bull or bear signal
    is_bull = direction == "BUY"

    # Determine color based on profit/loss
    # profit_loss > 0: green (profitable)
    # profit_loss < 0: red (loss)
    # profit_loss is None/NaN: yellow (still running)
    marker_color = "yellow"  # Default for running trades
    if hasattr(row, 'profit_loss') and row.profit_loss is not None:
        try:
            pnl_value = float(row.profit_loss)
            if pnl_value > 0:
                marker_color = "green"
            elif pnl_value < 0:
                marker_color = "red"
            else:
                marker_color = "gray"  # Exactly zero (breakeven)
        except (ValueError, TypeError):
            marker_color = "yellow"  # Can't determine, treat as running

    # Add P&L to text if available
    text_label = f"{strategy_name}" if strategy_name else ("BUY" if is_bull else "SELL")
    if hasattr(row, 'profit_loss') and row.profit_loss is not None:
        try:
            pnl_value = float(row.profit_loss)
            if pnl_value != 0:
                text_label += f" ({pnl_value:+.2f})"
        except (ValueError, TypeError):
            pass

    # Main trade marker (arrow) - keep this clean with just strategy and P&L
    marker = {
        "time": timestamp,
        "position": "aboveBar" if is_bull else "belowBar",
        "color": marker_color,
        "shape": "arrowUp" if is_bull else "arrowDown",
        "text": text_label
    }
    trade_markers.append(marker)

    # Add separate circle markers for risk management
    # Use small time offsets to avoid exact timestamp conflicts

    # Add Stop Loss circle marker if enabled and available
    if show_stop_levels and hasattr(row, 'sl_price') and row.sl_price is not None:
        try:
            sl_price = float(row.sl_price)
            # Offset by 30 seconds to avoid conflicts
            sl_marker = {
                "time": timestamp + 30,
                "position": "belowBar" if is_bull else "aboveBar",
                "color": "#FF4444",  # Red for stop loss
                "shape": "circle",
                "text": f"SL: {sl_price:.{price_precision}f}",
                "size": 1
            }
            stop_loss_markers.append(sl_marker)
        except (ValueError, TypeError):
            pass

    # Add Take Profit circle marker if enabled and available
    if show_tp_levels and hasattr(row, 'tp_price') and row.tp_price is not None:
        try:
            tp_price = float(row.tp_price)
            # Offset by 60 seconds to avoid conflicts
            tp_marker = {
                "time": timestamp + 60,
                "position": "aboveBar" if is_bull else "belowBar",
                "color": "#44FF44",  # Green for take profit
                "shape": "circle",
                "text": f"TP: {tp_price:.{price_precision}f}",
                "size": 1
            }
            take_profit_markers.append(tp_marker)
        except (ValueError, TypeError):
            pass

    # Add Trailing/Breakeven circle marker if enabled and available
    if show_trailing_info:
        if hasattr(row, 'moved_to_breakeven') and row.moved_to_breakeven:
            try:
                # Use entry price as breakeven level
                entry_price = float(row.entry_price)
                # Offset by 90 seconds to avoid conflicts
                be_marker = {
                    "time": timestamp + 90,
                    "position": "inBar",  # Center position for breakeven
                    "color": "#FFB000",  # Orange/yellow for breakeven
                    "shape": "circle",
                    "text": f"BE: {entry_price:.{price_precision}f}",
                    "size": 1
                }
                trailing_markers.append(be_marker)
            except (ValueError, TypeError):
                pass
        elif hasattr(row, 'trigger_distance') and row.trigger_distance is not None:
            try:
                trigger_distance = float(row.trigger_distance)
                # Only show if there's an actual trailing distance set
                if trigger_distance > 0:
                    # Offset by 120 seconds to avoid conflicts
                    trail_marker = {
                        "time": timestamp + 120,
                        "position": "inBar",
                        "color": "#9966FF",  # Purple for trailing info
                        "shape": "circle",
                        "text": f"Trail: {trigger_distance:.0f}pts",
                        "size": 1
                    }
                    trailing_markers.append(trail_marker)
            except (ValueError, TypeError):
                pass

# Build chart series with OHLC display enabled and proper precision
series = [
    {
        "type": "Candlestick",
        "data": candles,
        "options": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderVisible": False,
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350",
            "priceFormat": {
                "type": "price",
                "precision": price_precision,
                "minMove": 1 / (10 ** price_precision)
            }
        }
    }
]

# Add EMAs as line series based on toggles - now with dynamic periods and labels
if show_ema21 and "ema21" in df_display.columns:
    ema21_data = [
        {"time": int(row.start_time.timestamp()), "value": row.ema21}
        for row in df_display.itertuples() if not pd.isna(row.ema21)
    ]

    # Dynamic label based on actual period
    ema_short_period = ema_config.short_period if ema_config else 21
    series.append({
        "type": "Line",
        "data": ema21_data,
        "options": {
            "color": "#4caf50",  # Green for EMA short
            "lineWidth": 2,
            "title": f"EMA {ema_short_period}"
        }
    })

if show_ema50 and "ema50" in df_display.columns:
    ema50_data = [
        {"time": int(row.start_time.timestamp()), "value": row.ema50}
        for row in df_display.itertuples() if not pd.isna(row.ema50)
    ]

    # Dynamic label based on actual period
    ema_long_period = ema_config.long_period if ema_config else 50
    series.append({
        "type": "Line",
        "data": ema50_data,
        "options": {
            "color": "#ff9800",  # Orange for EMA long
            "lineWidth": 2,
            "title": f"EMA {ema_long_period}"
        }
    })

if show_ema200 and "ema200" in df_display.columns:
    ema200_data = [
        {"time": int(row.start_time.timestamp()), "value": row.ema200}
        for row in df_display.itertuples() if not pd.isna(row.ema200)
    ]

    # Dynamic label based on actual period
    ema_trend_period = ema_config.trend_period if ema_config else 200
    series.append({
        "type": "Line",
        "data": ema200_data,
        "options": {
            "color": "#2196f3",  # Blue for EMA trend
            "lineWidth": 2,
            "title": f"EMA {ema_trend_period}"
        }
    })

# Add Zero Lag EMA if enabled with color coding
if show_zlema and "zlema" in df_display.columns:
    # Create colored segments based on trend (price vs ZLEMA)
    current_trend = None
    current_segment = []
    all_segments = []
    
    for row in df_display.itertuples():
        if not pd.isna(row.zlema):
            # Determine trend: bullish if price > ZLEMA, bearish if price < ZLEMA  
            is_bullish = row.close > row.zlema
            data_point = {"time": int(row.start_time.timestamp()), "value": row.zlema}
            
            if current_trend is None:
                # First point
                current_trend = is_bullish
                current_segment = [data_point]
            elif current_trend == is_bullish:
                # Same trend, add to current segment
                current_segment.append(data_point)
            else:
                # Trend changed, save current segment and start new one
                if len(current_segment) > 1:
                    all_segments.append({
                        "data": current_segment.copy(),
                        "color": "#26a69a" if current_trend else "#ef5350",  # Green for bull, red for bear
                        "title": "ZLEMA Bull" if current_trend else "ZLEMA Bear"
                    })
                
                # Start new segment with the transition point from previous segment
                current_segment = [current_segment[-1] if current_segment else data_point, data_point]
                current_trend = is_bullish
    
    # Add the final segment
    if current_segment and len(current_segment) > 1:
        all_segments.append({
            "data": current_segment,
            "color": "#26a69a" if current_trend else "#ef5350",
            "title": "ZLEMA Bull" if current_trend else "ZLEMA Bear"
        })
    
    # Add all segments as separate line series with hidden price lines
    for i, segment in enumerate(all_segments):
        series.append({
            "type": "Line",
            "data": segment["data"],
            "options": {
                "color": segment["color"],
                "lineWidth": 2,
                "priceLineVisible": False,  # Hide price line on right axis
                "lastValueVisible": False,  # Hide last value label
                "title": "ZLEMA" if i == 0 else ""  # Only show title for first segment
            }
        })
    
    # Add upper and lower bands if enabled
    if show_zlema_bands:
        zlema_upper_data = [
            {"time": int(row.start_time.timestamp()), "value": row.zlema_upper}
            for row in df_display.itertuples() if not pd.isna(row.zlema_upper)
        ]
        zlema_lower_data = [
            {"time": int(row.start_time.timestamp()), "value": row.zlema_lower}
            for row in df_display.itertuples() if not pd.isna(row.zlema_lower)
        ]
        
        series.append({
            "type": "Line",
            "data": zlema_upper_data,
            "options": {
                "color": "#e91e63",
                "lineWidth": 1,
                "lineStyle": 2,  # Dashed
                "title": "ZLEMA Upper"
            }
        })
        
        series.append({
            "type": "Line",
            "data": zlema_lower_data,
            "options": {
                "color": "#e91e63",
                "lineWidth": 1,
                "lineStyle": 2,  # Dashed
                "title": "ZLEMA Lower"
            }
        })

# Add Support/Resistance levels if enabled
if show_sr_levels and sr_data:
    # Get the time range of visible candles
    if candles:
        first_time = candles[0]["time"]
        last_time = candles[-1]["time"]
        
        # Add resistance levels (red lines)
        for i, resistance_level in enumerate(sr_data.get('resistance_levels', [])):
            # Create a horizontal line by using the first and last time points
            resistance_line_data = [
                {"time": first_time, "value": resistance_level},
                {"time": last_time, "value": resistance_level}
            ]
            series.append({
                "type": "Line",
                "data": resistance_line_data,
                "options": {
                    "color": "#FF0000",  # Red for resistance
                    "lineWidth": 2 if i == 0 else 1,  # Thicker line for strongest level
                    "lineStyle": 0,  # Solid line
                    "priceLineVisible": False,
                    "lastValueVisible": False,
                    "title": f"R{i+1}" if i == 0 else ""  # Only label first resistance
                }
            })
        
        # Add support levels (blue lines)
        for i, support_level in enumerate(sr_data.get('support_levels', [])):
            # Create a horizontal line by using the first and last time points
            support_line_data = [
                {"time": first_time, "value": support_level},
                {"time": last_time, "value": support_level}
            ]
            series.append({
                "type": "Line",
                "data": support_line_data,
                "options": {
                    "color": "#233dee",  # Blue for support
                    "lineWidth": 2 if i == 0 else 1,  # Thicker line for strongest level
                    "lineStyle": 0,  # Solid line
                    "priceLineVisible": False,
                    "lastValueVisible": False,
                    "title": f"S{i+1}" if i == 0 else ""  # Only label first support
                }
            })

# Add swing markers if enabled
markers = []
if show_swings:
    internal_swings = classify_pivots(detect_pivots(df_full, lookback=5))
    swing_swings = classify_pivots(detect_pivots(df_full, lookback=50))
    swings = convert_swings_to_plot_shapes(internal_swings + swing_swings)

    for swing in swings:
        ts = int(pd.Timestamp(swing["x"]).timestamp())
        markers.append({
            "time": ts,
            "position": "aboveBar" if swing["text"] in ["HH", "LH"] else "belowBar",
            "color": swing["label_color"],
            "shape": "circle",
            "text": swing["text"]
        })

# Add S/R break markers if enabled
sr_break_markers = []
if show_sr_breaks and sr_data:
    # Add resistance break markers
    for break_event in sr_data.get('resistance_breaks', []):
        timestamp = int(break_event['time'].timestamp())
        sr_break_markers.append({
            "time": timestamp,
            "position": "belowBar",  # Resistance break shows below (bullish)
            "color": "green",
            "shape": "text",
            "text": break_event['label']  # "B" or "Bull Wick"
        })
    
    # Add support break markers
    for break_event in sr_data.get('support_breaks', []):
        timestamp = int(break_event['time'].timestamp())
        sr_break_markers.append({
            "time": timestamp,
            "position": "aboveBar",  # Support break shows above (bearish)
            "color": "red",
            "shape": "text",
            "text": break_event['label']  # "B" or "Bear Wick"
        })

# Filter markers to visible range
visible_candle_times = set(c["time"] for c in candles)

# For trade markers, we need to align them to the nearest candle time
# since trades can happen at any time but candles are at fixed intervals
aligned_trade_markers = []
if trade_markers and candles:
    candle_times_list = sorted([c["time"] for c in candles])
    
    for marker in trade_markers:
        trade_time = marker["time"]
        
        # Find the candle that contains this trade time
        # For a 5m candle starting at 10:00, it includes trades from 10:00:00 to 10:04:59
        for i, candle_time in enumerate(candle_times_list):
            # Check if trade falls within this candle's time range
            if i < len(candle_times_list) - 1:
                next_candle_time = candle_times_list[i + 1]
                if candle_time <= trade_time < next_candle_time:
                    # Align trade to this candle's time
                    aligned_marker = marker.copy()
                    aligned_marker["time"] = candle_time
                    aligned_trade_markers.append(aligned_marker)
                    break
            else:
                # Last candle - check if trade is within timeframe minutes of it
                time_diff = trade_time - candle_time
                if 0 <= time_diff < (timeframe * 60):  # timeframe in minutes, convert to seconds
                    aligned_marker = marker.copy()
                    aligned_marker["time"] = candle_time
                    aligned_trade_markers.append(aligned_marker)

# Combine all risk management markers for visibility filtering
all_risk_markers = stop_loss_markers + take_profit_markers + trailing_markers

# Filter all markers to visible range
visible_swing_markers = [m for m in markers if m["time"] in visible_candle_times]
visible_sr_break_markers = [m for m in sr_break_markers if m["time"] in visible_candle_times]

# For risk management markers, align them to candle times like trade markers
aligned_risk_markers = []
if all_risk_markers and candles:
    candle_times_list = sorted([c["time"] for c in candles])

    for marker in all_risk_markers:
        marker_time = marker["time"]

        # Find the candle that contains this marker time
        for i, candle_time in enumerate(candle_times_list):
            if i < len(candle_times_list) - 1:
                next_candle_time = candle_times_list[i + 1]
                if candle_time <= marker_time < next_candle_time:
                    aligned_marker = marker.copy()
                    aligned_marker["time"] = candle_time
                    aligned_risk_markers.append(aligned_marker)
                    break
            else:
                # Last candle - check if marker is within timeframe minutes of it
                time_diff = marker_time - candle_time
                if 0 <= time_diff < (timeframe * 60):
                    aligned_marker = marker.copy()
                    aligned_marker["time"] = candle_time
                    aligned_risk_markers.append(aligned_marker)

visible_markers = visible_swing_markers + aligned_trade_markers + visible_sr_break_markers + aligned_risk_markers

# Debug info
if trade_markers:
    st.sidebar.write(f"Total trade markers: {len(trade_markers)}")
    st.sidebar.write(f"Aligned trade markers: {len(aligned_trade_markers)}")
    if aligned_trade_markers:
        st.sidebar.write(f"First marker: {aligned_trade_markers[0]['text']}")

    # Show risk management markers info
    if show_stop_levels or show_tp_levels or show_trailing_info:
        st.sidebar.write("--- Risk Management ---")
        if show_stop_levels:
            st.sidebar.write(f"Stop Loss markers: {len(stop_loss_markers)}")
        if show_tp_levels:
            st.sidebar.write(f"Take Profit markers: {len(take_profit_markers)}")
        if show_trailing_info:
            st.sidebar.write(f"Trailing/BE markers: {len(trailing_markers)}")
        st.sidebar.write(f"Total risk markers: {len(aligned_risk_markers)}")

    # Show timestamp comparison
    if trade_markers and candles:
        st.sidebar.write("--- Debug Info ---")
        # Show first and last candle times
        candle_times_sorted = sorted(visible_candle_times)
        if candle_times_sorted:
            first_candle = min(candle_times_sorted)
            last_candle = max(candle_times_sorted)
            st.sidebar.write(f"Candle range: {pd.Timestamp(first_candle, unit='s').strftime('%Y-%m-%d %H:%M')} to {pd.Timestamp(last_candle, unit='s').strftime('%Y-%m-%d %H:%M')} UTC")

        # Show first trade marker time
        if trade_markers:
            first_trade = trade_markers[0]
            st.sidebar.write(f"First trade: {pd.Timestamp(first_trade['time'], unit='s').strftime('%Y-%m-%d %H:%M')} UTC")

# Configure chart options with explicit precision for forex pairs
chart_config = {
    "width": None,
    "height": 700,
    "layout": {
        "background": {"color": "#ffffff"},
        "textColor": "#333333"
    },
    "rightPriceScale": {
        "scaleMargins": {"top": 0.1, "bottom": 0.1},
        "borderVisible": True,
        "entireTextOnly": False,
        "visible": True
    },
    "crosshair": {
        "mode": 0,  # Normal crosshair mode
        "vertLine": {
            "width": 1,
            "color": "#758696",
            "style": 0
        },
        "horzLine": {
            "width": 1,
            "color": "#758696", 
            "style": 0
        }
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
        "mode": 0,  # Normal mode (0) instead of magnet mode  
        "vertLine": {
            "visible": True,
            "labelVisible": True,
            "style": 0,
            "width": 1,
            "color": "#758696"
        },
        "horzLine": {
            "visible": True,
            "labelVisible": True,
            "style": 0,
            "width": 1,
            "color": "#758696"
        }
    },
    "trackingMode": {
        "exitOnScrollOrScale": False
    }
}

# Prepare chart data with markers attached to the candlestick series
# The library might expect markers to be part of the series
series_with_markers = []
for s in series:
    if s["type"] == "Candlestick":
        # Add markers to the candlestick series
        s_copy = s.copy()
        if visible_markers:
            s_copy["markers"] = visible_markers
        series_with_markers.append(s_copy)
    else:
        series_with_markers.append(s)

# Prepare charts list - main price chart always first
charts_to_render = [
    {
        "chart": chart_config,
        "series": series_with_markers
    }
]

# Add MACD chart if enabled
if show_macd and "macd_line" in df_display.columns:
    # Prepare MACD data
    macd_line_data = [
        {"time": int(row.start_time.timestamp()), "value": row.macd_line}
        for row in df_display.itertuples() if not pd.isna(row.macd_line)
    ]
    
    macd_signal_data = [
        {"time": int(row.start_time.timestamp()), "value": row.macd_signal}
        for row in df_display.itertuples() if not pd.isna(row.macd_signal)
    ]
    
    macd_histogram_data = [
        {
            "time": int(row.start_time.timestamp()), 
            "value": row.macd_histogram,
            "color": "#26a69a" if row.macd_histogram >= 0 else "#ef5350"
        }
        for row in df_display.itertuples() if not pd.isna(row.macd_histogram)
    ]
    
    # MACD chart configuration
    macd_chart_config = {
        "width": None,
        "height": 250,
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
    
    # MACD series
    macd_series = [
        {
            "type": "Histogram",
            "data": macd_histogram_data,
            "options": {
                "priceLineVisible": False,
                "lastValueVisible": False
            }
        },
        {
            "type": "Line",
            "data": macd_line_data,
            "options": {
                "color": "#2196f3",
                "lineWidth": 2,
                "title": "MACD"
            }
        },
        {
            "type": "Line",
            "data": macd_signal_data,
            "options": {
                "color": "#ff9800",
                "lineWidth": 2,
                "title": "Signal"
            }
        }
    ]
    
    charts_to_render.append({
        "chart": macd_chart_config,
        "series": macd_series
    })

# Add Two-Pole Oscillator chart if enabled
if show_two_pole and "two_pole_osc" in df_display.columns:
    # Prepare Two-Pole data
    two_pole_data = [
        {"time": int(row.start_time.timestamp()), "value": row.two_pole_osc}
        for row in df_display.itertuples() if not pd.isna(row.two_pole_osc)
    ]
    
    two_pole_delayed_data = [
        {"time": int(row.start_time.timestamp()), "value": row.two_pole_osc_delayed}
        for row in df_display.itertuples() if not pd.isna(row.two_pole_osc_delayed)
    ]
    
    # Create histogram data with colors based on oscillator state
    two_pole_histogram_data = []
    for row in df_display.itertuples():
        if not pd.isna(row.two_pole_osc):
            # Color based on whether oscillator is rising (green) or falling (purple)
            color = "#26a69a" if getattr(row, 'two_pole_is_green', False) else "#9c27b0"
            two_pole_histogram_data.append({
                "time": int(row.start_time.timestamp()),
                "value": row.two_pole_osc,
                "color": color
            })
    
    # Two-Pole chart configuration
    two_pole_chart_config = {
        "width": None,
        "height": 250,
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
    
    # Add zero line reference
    zero_line_data = [
        {"time": t["time"], "value": 0}
        for t in two_pole_data
    ]
    
    # Two-Pole series
    two_pole_series = [
        {
            "type": "Histogram",
            "data": two_pole_histogram_data,
            "options": {
                "priceLineVisible": False,
                "lastValueVisible": False
            }
        },
        {
            "type": "Line",
            "data": two_pole_data,
            "options": {
                "color": "#2196f3",
                "lineWidth": 2,
                "title": "Two-Pole Osc"
            }
        },
        {
            "type": "Line",
            "data": two_pole_delayed_data,
            "options": {
                "color": "#ff9800",
                "lineWidth": 1,
                "lineStyle": 2,  # Dashed line
                "title": "Signal"
            }
        },
        {
            "type": "Line",
            "data": zero_line_data,
            "options": {
                "color": "#666666",
                "lineWidth": 1,
                "lineStyle": 1,  # Dotted line
                "title": "Zero"
            }
        }
    ]
    
    charts_to_render.append({
        "chart": two_pole_chart_config,
        "series": two_pole_series
    })

# Display Session Analysis if enabled
if show_session_analysis and session_analysis:
    session_name = session_analysis.get('current_session', 'Unknown')
    active_sessions = session_analysis.get('active_sessions', [])
    session_chars = session_analysis.get('session_characteristics', [])
    
    # Determine volatility based on session
    if 'London' in active_sessions and 'New York' in active_sessions:
        volatility = 'Peak'
        session_type = 'Overlap'
    elif 'London' in active_sessions:
        volatility = 'High'
        session_type = 'London'
    elif 'New York' in active_sessions:
        volatility = 'High'
        session_type = 'New York'
    elif 'Asian' in active_sessions:
        volatility = 'Low'
        session_type = 'Asian'
    else:
        volatility = 'Minimal'
        session_type = 'Off Hours'
    
    # Get first characteristic or default
    characteristics = session_chars[0] if session_chars else f"{session_type} market conditions"
    
    # Create session indicator bar
    session_color = {
        'Asian': '🟡',
        'London': '🟢', 
        'New York': '🔵',
        'London/New York': '🟣',
        'Off Hours': '⚪'
    }.get(session_name, '⚪')
    
    st.info(f"{session_color} **{session_name.upper()} SESSION** | {volatility} Volatility | {characteristics}")

# Display Market Regime if enabled  
if show_market_regime and regime_data:
    regime_scores = regime_data.get('regime_scores', {})
    if regime_scores:
        # Find dominant regime
        dominant_regime = max(regime_scores.items(), key=lambda x: x[1])
        regime_name, confidence = dominant_regime
        
        # Create regime indicator
        regime_colors = {
            'trending': '🟢',
            'ranging': '🟡', 
            'breakout': '🟠',
            'reversal': '🔴',
            'high_volatility': '🔴',
            'low_volatility': '🔵'
        }
        
        regime_icon = regime_colors.get(regime_name, '⚪')
        volatility_pct = regime_data.get('volatility_percentile', 50)
        
        st.warning(f"{regime_icon} **{regime_name.upper()} REGIME** ({confidence:.1%} confidence) | Volatility: {volatility_pct:.0f}th percentile")

# Render all charts
renderLightweightCharts(charts_to_render, "tvchart-main")

# Display current candle information (last candle OHLC data)
if not df_display.empty:
    latest_candle = df_display.iloc[-1]
    
    # OHLC Data with appropriate precision for different pairs
    st.subheader(f"Current Candle Data - {latest_candle['start_time'].strftime('%Y-%m-%d %H:%M')} UTC")
    
    # Determine decimal places based on pair type
    # JPY pairs typically use 3 decimals, others use 5
    if "JPY" in selected_epic:
        decimal_places = 3
        change_decimal_places = 3
    else:
        decimal_places = 5
        change_decimal_places = 5
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Open", f"{latest_candle['open']:.{decimal_places}f}")
    with col2:
        st.metric("High", f"{latest_candle['high']:.{decimal_places}f}")
    with col3:
        st.metric("Low", f"{latest_candle['low']:.{decimal_places}f}")
    with col4:
        st.metric("Close", f"{latest_candle['close']:.{decimal_places}f}")
    with col5:
        change = latest_candle['close'] - latest_candle['open']
        change_pct = (change / latest_candle['open']) * 100 if latest_candle['open'] != 0 else 0
        st.metric("Change", f"{change:+.{change_decimal_places}f}", f"{change_pct:+.2f}%")
    
    # Technical Indicators (if available)
    has_indicators = any([
        'ema21' in latest_candle and not pd.isna(latest_candle['ema21']),
        'ema50' in latest_candle and not pd.isna(latest_candle['ema50']),
        'ema200' in latest_candle and not pd.isna(latest_candle['ema200']),
        'macd_line' in latest_candle and not pd.isna(latest_candle['macd_line']),
        'zlema' in latest_candle and not pd.isna(latest_candle['zlema'])
    ])
    
    if has_indicators:
        st.subheader("Technical Indicators")
        
        # EMAs row - now with dynamic periods
        if any([show_ema21, show_ema50, show_ema200]):
            ema_cols = st.columns(4)

            # Get dynamic periods for display
            ema_short_period = ema_config.short_period if ema_config else 21
            ema_long_period = ema_config.long_period if ema_config else 50
            ema_trend_period = ema_config.trend_period if ema_config else 200

            if show_ema21 and 'ema21' in latest_candle and not pd.isna(latest_candle['ema21']):
                with ema_cols[0]:
                    st.metric(f"EMA {ema_short_period}", f"{latest_candle['ema21']:.{decimal_places}f}")
            if show_ema50 and 'ema50' in latest_candle and not pd.isna(latest_candle['ema50']):
                with ema_cols[1]:
                    st.metric(f"EMA {ema_long_period}", f"{latest_candle['ema50']:.{decimal_places}f}")
            if show_ema200 and 'ema200' in latest_candle and not pd.isna(latest_candle['ema200']):
                with ema_cols[2]:
                    st.metric(f"EMA {ema_trend_period}", f"{latest_candle['ema200']:.{decimal_places}f}")
            if show_zlema and 'zlema' in latest_candle and not pd.isna(latest_candle['zlema']):
                with ema_cols[3]:
                    zlema_trend = "↗️" if latest_candle['close'] > latest_candle['zlema'] else "↘️"
                    st.metric(f"ZLEMA {zlema_trend}", f"{latest_candle['zlema']:.{decimal_places}f}")
        
        # MACD row (use higher precision for MACD values)
        if show_macd and 'macd_line' in latest_candle and not pd.isna(latest_candle['macd_line']):
            macd_precision = 7 if "JPY" not in selected_epic else 6
            macd_cols = st.columns(3)
            with macd_cols[0]:
                st.metric("MACD", f"{latest_candle['macd_line']:.{macd_precision}f}")
            with macd_cols[1]:
                st.metric("Signal", f"{latest_candle['macd_signal']:.{macd_precision}f}")
            with macd_cols[2]:
                histogram_trend = "↗️" if latest_candle['macd_histogram'] > 0 else "↘️"
                st.metric(f"Histogram {histogram_trend}", f"{latest_candle['macd_histogram']:.{macd_precision}f}")
        
        # Two-Pole row
        if show_two_pole and 'two_pole_osc' in latest_candle and not pd.isna(latest_candle['two_pole_osc']):
            pole_cols = st.columns(2)
            with pole_cols[0]:
                pole_color = "🟢" if getattr(latest_candle, 'two_pole_is_green', False) else "🟣"
                st.metric(f"Two-Pole {pole_color}", f"{latest_candle['two_pole_osc']:.6f}")
            with pole_cols[1]:
                zone = "Overbought" if latest_candle['two_pole_osc'] > 0 else "Oversold"
                st.metric("Zone", zone)

# Display trade statistics
if not trades_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", len(trades_df))
    with col2:
        buy_count = len(trades_df[trades_df["direction"].str.upper() == "BUY"])
        st.metric("Buy Signals", buy_count)
    with col3:
        sell_count = len(trades_df[trades_df["direction"].str.upper() == "SELL"])
        st.metric("Sell Signals", sell_count)
    with col4:
        # Count trades by P&L status
        if "profit_loss" in trades_df.columns:
            profitable = len(trades_df[trades_df["profit_loss"] > 0])
            losses = len(trades_df[trades_df["profit_loss"] < 0])
            running = len(trades_df[trades_df["profit_loss"].isna()])
            st.metric("Win/Loss/Running", f"{profitable}/{losses}/{running}")
    
    # Show strategy distribution
    if "strategy" in trades_df.columns:
        strategy_counts = trades_df["strategy"].value_counts()
        if not strategy_counts.empty:
            st.subheader("Strategy Distribution")
            st.bar_chart(strategy_counts)

# Display Trade Context Analysis if enabled
if show_trade_context and not trades_df.empty:
    st.subheader("📊 Trade Context Analysis")
    st.write("Understanding market conditions when trades occurred")
    
    # Get intelligence service
    try:
        intelligence_service = get_intelligence_service()
        
        if intelligence_service.is_available():
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Individual Trades", "Regime Performance", "Trade Summary"])
            
            with tab1:
                st.write("**Trade-by-Trade Analysis**")
                
                trade_contexts = []
                for idx, trade in trades_df.head(10).iterrows():  # Limit to 10 for performance
                    try:
                        trade_data = {
                            'entry_time': trade['timestamp'],
                            'direction': trade['direction'], 
                            'pnl': trade.get('profit_loss', 0),
                            'strategy': trade.get('strategy', 'Unknown')
                        }
                        
                        context = intelligence_service.analyze_trade_context(trade_data, selected_epic)
                        trade_contexts.append(context)
                        
                        # Display individual trade analysis
                        success_color = "🟢" if context.get('success', False) else "🔴"
                        regime = context.get('dominant_regime', 'unknown')
                        quality_score = context.get('trade_quality_score', 0) * 100
                        
                        with st.expander(f"{success_color} {trade_data['strategy']} {trade_data['direction']} - PnL: {trade_data['pnl']:.2f} ({quality_score:.0f}% quality)", expanded=False):
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Market Conditions:**")
                                st.write(f"• Regime: {regime.title()}")
                                st.write(f"• Session: {context.get('current_session', 'unknown').title()}")
                                st.write(f"• Volatility: {context.get('volatility_percentile', 50):.0f}th percentile")
                                st.write(f"• Volume: {context.get('volume_relative', 1):.1f}x normal")
                            
                            with col2:
                                st.write("**Trade Analysis:**")
                                regime_align = context.get('regime_alignment', {})
                                session_align = context.get('session_alignment', {})
                                st.write(f"• Regime Alignment: {regime_align.get('alignment', 'unknown').title()}")
                                st.write(f"• Quality Score: {quality_score:.0f}%")
                                st.write(f"• Entry Price: {context.get('entry_price', 0):.5f}")
                                st.write(f"• Regime Score: {regime_align.get('score', 0):.1%}")
                                st.write(f"• Session Score: {session_align.get('score', 0):.1%}")
                            
                            # Success factors and improvements
                            st.write("**Key Insights:**")
                            for factor in context.get('success_factors', [])[:3]:  # Limit to top 3
                                st.write(f"✓ {factor}")
                            
                            for suggestion in context.get('improvement_suggestions', [])[:2]:  # Limit to top 2
                                st.write(f"💡 {suggestion}")
                            
                    except Exception as e:
                        st.error(f"Error analyzing trade {idx}: {str(e)[:100]}")
                        continue
            
            with tab2:
                st.write("**Performance by Market Regime**")
                
                if 'trade_contexts' in locals() and trade_contexts:
                    # Create regime performance summary
                    regime_performance = {}
                    
                    for context in trade_contexts:
                        regime = context.get('dominant_regime', 'unknown')
                        success = context.get('success', False)
                        pnl = context.get('pnl', 0)
                        
                        if regime not in regime_performance:
                            regime_performance[regime] = {
                                'total_trades': 0,
                                'winning_trades': 0,
                                'total_pnl': 0,
                                'avg_quality': 0
                            }
                        
                        regime_performance[regime]['total_trades'] += 1
                        if success:
                            regime_performance[regime]['winning_trades'] += 1
                        regime_performance[regime]['total_pnl'] += pnl
                        regime_performance[regime]['avg_quality'] += context.get('trade_quality_score', 0)
                    
                    # Display regime performance table
                    regime_data = []
                    for regime, stats in regime_performance.items():
                        win_rate = (stats['winning_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
                        avg_quality = (stats['avg_quality'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
                        
                        regime_data.append({
                            'Regime': regime.title(),
                            'Total Trades': stats['total_trades'],
                            'Win Rate': f"{win_rate:.1f}%", 
                            'Total PnL': f"{stats['total_pnl']:.2f}",
                            'Avg Quality': f"{avg_quality:.0f}%"
                        })
                    
                    if regime_data:
                        st.dataframe(regime_data, use_container_width=True)
                        
                        # Highlight best and worst performing regimes
                        if len(regime_performance) > 1:
                            best_regime = max(regime_performance.items(), key=lambda x: x[1]['total_pnl'])
                            worst_regime = min(regime_performance.items(), key=lambda x: x[1]['total_pnl'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"🏆 Best: **{best_regime[0].title()}** ({best_regime[1]['total_pnl']:+.2f} PnL)")
                            with col2:
                                st.error(f"⚠️ Worst: **{worst_regime[0].title()}** ({worst_regime[1]['total_pnl']:+.2f} PnL)")
            
            with tab3:
                st.write("**Overall Trading Summary**")
                
                if 'trade_contexts' in locals() and trade_contexts:
                    # Overall statistics
                    total_trades = len(trade_contexts)
                    successful_trades = sum(1 for ctx in trade_contexts if ctx.get('success', False))
                    total_pnl = sum(ctx.get('pnl', 0) for ctx in trade_contexts)
                    avg_quality = sum(ctx.get('trade_quality_score', 0) for ctx in trade_contexts) / total_trades * 100 if total_trades > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Analyzed Trades", total_trades)
                    with col2:
                        st.metric("Win Rate", f"{(successful_trades/total_trades)*100:.1f}%" if total_trades > 0 else "0%")
                    with col3:
                        st.metric("Total PnL", f"{total_pnl:+.2f}")
                    with col4:
                        st.metric("Avg Quality Score", f"{avg_quality:.0f}%")
                    
                    # Key insights
                    st.write("**Key Insights:**")
                    if avg_quality < 60:
                        st.warning("• Consider waiting for higher quality setups")
                    if total_pnl < 0:
                        st.error("• Review losing trades for improvement opportunities")
                    else:
                        st.success("• Trading approach is generating positive results")
        else:
            st.warning("⚠️ Market Intelligence Engine not available. Trade context analysis requires the intelligence system to be properly configured.")
            
    except Exception as e:
        st.error(f"Error initializing trade context analysis: {str(e)[:100]}")

elif show_trade_context and trades_df.empty:
    st.info("No trades found for context analysis. Generate some trades to see intelligent analysis!")

# Auto-refresh functionality
if refresh_minutes > 0:
    total_seconds = refresh_minutes * 60
    with st.empty():
        for i in range(total_seconds, 0, -1):
            mins, secs = divmod(i, 60)
            st.info(f"⏳ Refreshing in {mins:02}:{secs:02} minutes")
            time.sleep(1)
        st.rerun()