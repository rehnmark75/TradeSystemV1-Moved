import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
import os
from sqlalchemy import create_engine
import plotly.graph_objects as go

from services.data import get_candle_data, get_trade_logs, get_epics
from services.indicators import apply_indicators
from services.smc_structure import (
    detect_pivots, classify_pivots, calculate_zones, determine_bias,
    detect_structure_signals_luxalgo, TrailingExtremes,
    convert_swings_to_plot_shapes, get_recent_trailing_extremes
)
from utils.plotting import build_candlestick_chart
from utils.helpers import synthesize_15m_from_5m

# --- Config ---
st.set_page_config(page_title="Candlestick Viewer", layout="wide")
LOCAL_TZ = pytz.timezone("Europe/Stockholm")
DISPLAY_CANDLES = 120 if "5m" in st.session_state.get("selected_tf", "") else 80

# --- Robust candle cleaning ---
def robust_clean_candle_data(df: pd.DataFrame, tz) -> pd.DataFrame:
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

    df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("UTC").dt.tz_convert(tz)
    df.sort_values("start_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- Sidebar ---
engine = create_engine(os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex"))
epics = get_epics(engine)
timeframe_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
selected_tf = st.sidebar.selectbox("Chart timeframe", options=["5m", "15m", "1h"], index=1)
st.session_state["selected_tf"] = selected_tf
timeframe = timeframe_minutes[selected_tf]
candles_per_day = (24 * 60) // timeframe
lookback_days = st.sidebar.slider("Lookback period (days)", 1, 5, 2)
lookback_candles = lookback_days * candles_per_day

with st.sidebar:
    st.header("⚙️ Settings")
    show_premium = st.checkbox("Show Premium zone", value=False)
    show_discount = st.checkbox("Show Discount zone", value=False)
    show_equilibrium = st.checkbox("Show Equilibrium zone", value=True)
    selected_epic = st.selectbox("Select Symbol (Epic)", epics)
    refresh_minutes = st.selectbox("Auto-refresh interval (minutes)", [0, 1, 2, 5, 10], index=2)
    selected_directions = st.multiselect("Filter trades by direction", ["BUY", "SELL"], default=["BUY", "SELL"])
    indicators = st.multiselect("Indicators", options=["EMA1", "EMA2"], default=["EMA1", "EMA2"])
    ema1_period = st.slider("EMA1 Period", 5, 50, 12) if "EMA1" in indicators else None
    ema2_period = st.slider("EMA2 Period", 5, 50, 50) if "EMA2" in indicators else None
    show_swings = st.checkbox("Show HH/HL/LH/LL labels", value=True)
    show_structure = st.checkbox("Show BOS/CHoCH markers", value=True)

if st.button("🔁 Refresh now"):
    st.cache_data.clear()
    st.rerun()

st.title("📊 Live Candlestick Chart")

if "last_epic" not in st.session_state:
    st.session_state.last_epic = selected_epic
elif st.session_state.last_epic != selected_epic:
    st.session_state.last_epic = selected_epic
    st.session_state.last_top = None
    st.session_state.last_bottom = None

if "last_top" not in st.session_state:
    st.session_state.last_top = None
if "last_bottom" not in st.session_state:
    st.session_state.last_bottom = None

# --- Load and clean candles ---
df_raw = get_candle_data(engine, timeframe, selected_epic, limit=lookback_candles * 2)
df_full = robust_clean_candle_data(df_raw, LOCAL_TZ)

if len(df_full) < DISPLAY_CANDLES and timeframe == 15:
    df_full = synthesize_15m_from_5m(engine, selected_epic, lookback_candles, LOCAL_TZ)

df_display = df_full.tail(DISPLAY_CANDLES).reset_index(drop=True)

if df_display.empty:
    st.warning("No candles available to display.")
    st.stop()

visible_start, visible_end = df_display["start_time"].min(), df_display["start_time"].max()

internal_swings = classify_pivots(detect_pivots(df_full, lookback=5))
swing_swings = classify_pivots(detect_pivots(df_full, lookback=50))
all_classified = sorted(internal_swings + swing_swings, key=lambda x: x["time"])

internal_swings = [s for s in internal_swings if visible_start <= s["time"] <= visible_end]
swing_swings = [s for s in swing_swings if visible_start <= s["time"] <= visible_end]

structure_signals = detect_structure_signals_luxalgo(all_classified, df_full) if show_structure else []
structure_signals = [s for s in structure_signals if visible_start <= s.get("confirmation_time", s["time"]) <= visible_end]
bias = determine_bias(structure_signals)

trades_df = get_trade_logs(engine, selected_epic, df_full["start_time"].min())
trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
trades_df = trades_df[trades_df["direction"].str.upper().isin(selected_directions)]
trades_df = trades_df[(trades_df["timestamp"] >= visible_start) & (trades_df["timestamp"] <= visible_end)]

extremes = get_recent_trailing_extremes(all_classified, last_top=st.session_state.last_top, last_bottom=st.session_state.last_bottom)
if extremes:
    st.session_state.last_top = extremes["top_source"]
    st.session_state.last_bottom = extremes["bottom_source"]
zones = calculate_zones(extremes["top"], extremes["bottom"]) if extremes else None
zone_x0 = extremes["bar_time"] if extremes else visible_start
zone_x1 = visible_end

df_display = apply_indicators(df_display, ema1_period, ema2_period, indicators) if not df_display.empty else df_display

fig = build_candlestick_chart(df_display, trades_df, internal_swings, swing_swings, structure_signals,
                              TrailingExtremes(), zones, show_premium, show_discount, show_equilibrium,
                              indicators, ema1_period, ema2_period, show_swings, show_structure, bias,
                              zone_x0=zone_x0, zone_x1=zone_x1)

fig.update_layout(
    xaxis=dict(
        type="date",
        tickformat="%H:%M\n%b %d",
        showgrid=True,
        autorange=False,
        range=[df_display["start_time"].min(), df_display["start_time"].max()],
        rangeslider=dict(visible=False),
        rangebreaks=[dict(pattern="day of week", bounds=[6, 1])]
    )
)

st.plotly_chart(fig, use_container_width=True)

if bias is not None:
    st.sidebar.info(f"Market Structure Bias: {'Bullish' if bias == 1 else 'Bearish'}")
if "overall_trend" in locals():
    st.sidebar.info(f"1H Trend Bias: {'Bullish' if overall_trend == 'uptrend' else 'Bearish'}")

if refresh_minutes > 0:
    total_seconds = refresh_minutes * 60
    with st.empty():
        for i in range(total_seconds, 0, -1):
            mins, secs = divmod(i, 60)
            st.info(f"⏳ Refreshing in {mins:02}:{secs:02} minutes")
            time.sleep(1)
        st.rerun()
