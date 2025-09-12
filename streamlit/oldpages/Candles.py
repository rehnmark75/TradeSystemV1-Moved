import streamlit as st
import pandas as pd
import os
import time
import pytz
import numpy as np
from sqlalchemy import create_engine
import plotly.graph_objects as go

from services.data import get_candle_data, get_trade_logs, get_epics
from services.indicators import detect_fvg
from services.smc_structure import (
    detect_pivots, classify_pivots, get_recent_trailing_extremes,
    calculate_zones, convert_swings_to_plot_shapes, determine_bias,
    detect_structure_signals_luxalgo, TrailingExtremes
)

# --- Config ---
st.set_page_config(page_title="Candlestick Viewer", layout="wide")
LOCAL_TZ = pytz.timezone("Europe/Stockholm")
DISPLAY_CANDLES = 200

# --- Sidebar ---
engine = create_engine(os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex"))
epics = get_epics(engine)

# Timeframe selector (optional)
selected_tf = st.sidebar.selectbox("Chart timeframe", options=["5m", "15m", "1h"], index=0)

# Timeframe mapping
timeframe_minutes = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60
}

minutes_per_candle = timeframe_minutes[selected_tf]
candles_per_day = (24 * 60) // minutes_per_candle

# Lookback days slider
lookback_days = st.sidebar.slider("Lookback period (days)", 1, 5, 2)
lookback_candles = lookback_days * candles_per_day

st.sidebar.write(f"{lookback_days} day(s) → {lookback_candles} candles")

with st.sidebar:
    st.header("⚙️ Settings")
    show_premium = st.checkbox("Show Premium zone", value=False)
    show_discount = st.checkbox("Show Discount zone", value=False)
    show_equilibrium = st.checkbox("Show Equilibrium zone", value=True)
    selected_epic = st.selectbox("Select Symbol (Epic)", epics)
    timeframe = st.selectbox("Select timeframe", [1, 5, 15, 60], index=2)
    refresh_minutes = st.selectbox("Auto-refresh interval (minutes)", [0, 1, 2, 5, 10], index=2)
    selected_directions = st.multiselect("Filter trades by direction", ["BUY", "SELL"], default=["BUY", "SELL"])
    indicators = st.multiselect("Indicators", options=["EMA1", "EMA2"], default=["EMA1", "EMA2"])
    ema1_period = None
    ema2_period = None
    if "EMA1" in indicators:
        ema1_period = st.slider("EMA1 Period", min_value=5, max_value=50, value=12)
    if "EMA2" in indicators:
        ema2_period = st.slider("EMA2 Period", min_value=5, max_value=50, value=50)
    show_swings = st.checkbox("Show HH/HL/LH/LL labels", value=True)
    show_structure = st.checkbox("Show BOS/CHoCH markers", value=True)

if st.button("🔁 Refresh now"):
    st.cache_data.clear()
    st.rerun()

st.title("📊 Live Candlestick Chart")

@st.cache_data(ttl=10)
def cached_get_candles(tf, epic, max_rows=500):
    return get_candle_data(engine, tf, epic, limit=max_rows * 2)

@st.cache_data(ttl=10)
def cached_get_trades(epic, min_time):
    return get_trade_logs(engine, epic, min_time)

# --- Load candles ---
df_full = cached_get_candles(timeframe, selected_epic, max_rows=lookback_candles)

# Convert and clean numeric columns
numeric_cols = ["open", "high", "low", "close"]

# Convert to numeric and coerce non-numeric to NaN
for col in numeric_cols:
    df_full[col] = pd.to_numeric(df_full[col], errors="coerce")

# Drop rows with NaNs in any price columns
df_full.dropna(subset=numeric_cols, inplace=True)

# Now force float64 explicitly
df_full[numeric_cols] = df_full[numeric_cols].astype("float64")

# Safely remove flat candles
is_flat = (
    np.isclose(df_full["open"].values, df_full["high"].values) &
    np.isclose(df_full["high"].values, df_full["low"].values) &
    np.isclose(df_full["low"].values, df_full["close"].values)
)
df_full = df_full[~is_flat]

df_full["start_time"] = pd.to_datetime(df_full["start_time"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)

# Attempt 15m fallback using 5m aggregation if too few candles
if len(df_full) < DISPLAY_CANDLES and timeframe == 15:
    #st.warning("Not enough 15m candles available. Trying to synthesize 15m candles from 5m data.")
    df_5m = get_candle_data(engine, 5, selected_epic, limit=lookback_candles * 3)
    df_5m[numeric_cols] = df_5m[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df_5m.dropna(subset=numeric_cols, inplace=True)
    df_5m["start_time"] = pd.to_datetime(df_5m["start_time"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
    df_5m.set_index("start_time", inplace=True)
    
    df_full = df_5m.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()

df_full.sort_values("start_time", inplace=True)
df_full.reset_index(drop=True, inplace=True)

if len(df_full) < DISPLAY_CANDLES:
   #st.warning(f"Only {len(df_full)} valid candles available (showing all).")
    df = df_full.reset_index(drop=True)
else:
    df = df_full.tail(DISPLAY_CANDLES).reset_index(drop=True)


# --- Trim for display ---
zone_df = df_full.tail(DISPLAY_CANDLES + 100).reset_index(drop=True)  # More bars for trailing zone calc
df = zone_df.tail(DISPLAY_CANDLES).reset_index(drop=True)  

trailing_extremes = TrailingExtremes()

for _, row in zone_df.iterrows():
    update_trailing_extremes(trailing_extremes, row)

# --- Trades ---
trades_df = cached_get_trades(selected_epic, df_full["start_time"].min())
trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
trades_df = trades_df[trades_df["direction"].str.upper().isin(selected_directions)]

# --- Indicators ---
if "EMA1" in indicators and ema1_period:
    df["ema1"] = df["close"].rolling(window=ema1_period).mean()
if "EMA2" in indicators and ema2_period:
    df["ema2"] = df["close"].ewm(span=ema2_period, adjust=False).mean()

# --- Load 1H and 4H candles ---
df_4h = get_candle_data(engine, 240, selected_epic, limit=100)
df_1h = get_candle_data(engine, 60, selected_epic, limit=100)

for df_tmp in [df_4h, df_1h]:
    df_tmp[numeric_cols] = df_tmp[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df_tmp.dropna(subset=numeric_cols, inplace=True)
    df_tmp["start_time"] = pd.to_datetime(df_tmp["start_time"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
    df_tmp.set_index("start_time", inplace=True)

# 1H/4H trend bias
trend_1h = trend_4h = overall_trend = None
if not df_4h.empty:
    df_4h["ema_50"] = df_4h["close"].ewm(span=50, adjust=False).mean()
    trend_4h = "up" if df_4h["close"].iloc[-1] > df_4h["ema_50"].iloc[-1] else "down"
else:
    st.warning("⚠️ No 4H data available.")
if not df_1h.empty:
    df_1h["ema_50"] = df_1h["close"].ewm(span=50, adjust=False).mean()
    trend_1h = "up" if df_1h["close"].iloc[-1] > df_1h["ema_50"].iloc[-1] else "down"
    overall_trend = "uptrend" if trend_1h == "up" else "downtrend"
else:
    st.warning("⚠️ No 1H data available.")

# --- Swings and Structure ---
visible_start = df["start_time"].min()
visible_end = df["start_time"].max()
internal_swings = classify_pivots(detect_pivots(df_full, lookback=5))
swing_swings = classify_pivots(detect_pivots(df_full, lookback=50))
all_classified = sorted(internal_swings + swing_swings, key=lambda x: x["time"])

internal_swings_visible = [s for s in internal_swings if visible_start <= s["time"] <= visible_end]
swing_swings_visible = [s for s in swing_swings if visible_start <= s["time"] <= visible_end]
all_classified = sorted(internal_swings + swing_swings, key=lambda x: x["time"])
structure_signals = detect_structure_signals_luxalgo(all_classified, df_full) if show_structure else []
bias = determine_bias(structure_signals)

swings_input = swing_swings if swing_swings else internal_swings
visible_swings = [s for s in swings_input if visible_start <= s["time"] <= visible_end]

zones = calculate_zones(trailing_extremes.top, trailing_extremes.bottom) if trailing_extremes.top and trailing_extremes.bottom else None


# --- Plotting ---
fig = go.Figure()

zone_x0 = trailing_extremes.bar_time or df["start_time"].min()
zone_x1 = df["start_time"].max()

if zones:
    st.sidebar.success("✅ Premium/Discount zones plotted.")
    if show_premium:
        fig.add_shape(type="rect", x0=zone_x0, x1=zone_x1,
                    y0=zones["premium"], y1=trailing_extremes.top,
                    fillcolor="rgba(255,0,0,0.08)", line=dict(width=0), layer="below")

    if show_discount:
        fig.add_shape(type="rect", x0=zone_x0, x1=zone_x1,
                    y0=zones["discount"], y1=trailing_extremes.bottom,
                    fillcolor="rgba(0,255,0,0.08)", line=dict(width=0), layer="below")

    if show_equilibrium:
        fig.add_shape(type="rect", x0=zone_x0, x1=zone_x1,
                    y0=zones["equilibrium_bottom"], y1=zones["equilibrium_top"],
                    fillcolor="rgba(128,128,128,0.15)", line=dict(width=0), layer="below")
        
    # Optional midpoint label
    eq_mid = (zones["equilibrium_bottom"] + zones["equilibrium_top"]) / 2
    fig.add_annotation(x=zone_x1, y=eq_mid, text="Equi",
                       showarrow=False, font=dict(size=10, color="gray"),
                       xanchor="right", yanchor="bottom")

fig.add_trace(go.Candlestick(x=df["start_time"], open=df["open"], high=df["high"],
                             low=df["low"], close=df["close"], name="Candles",
                             increasing_line_color="green", decreasing_line_color="red"))

if "EMA1" in indicators:
    fig.add_trace(go.Scatter(x=df["start_time"], y=df["ema1"], mode="lines", name=f"EMA1 ({ema1_period})", line=dict(color="blue")))
if "EMA2" in indicators:
    fig.add_trace(go.Scatter(x=df["start_time"], y=df["ema2"], mode="lines", name=f"EMA2 ({ema2_period})", line=dict(color="orange")))

for _, trade in trades_df.iterrows():
    fig.add_trace(go.Scatter(x=[trade["timestamp"]], y=[trade["entry_price"]], mode="markers", name=f"{trade['direction']} Trade", showlegend=False,
                             marker=dict(size=14, symbol="triangle-up" if trade["direction"] == "BUY" else "triangle-down",
                                         color="green" if trade["direction"] == "BUY" else "red", line=dict(width=1, color="black"))))

if show_swings:
    for swing in convert_swings_to_plot_shapes(internal_swings_visible + swing_swings_visible):
        fig.add_trace(go.Scatter(x=[swing["x"]], y=[swing["y"]], mode="text", text=[swing["text"]],
                                 textfont=dict(size=11, color=swing["label_color"]), showlegend=False))

if show_structure:
    for signal in structure_signals:
        fig.add_trace(go.Scatter(
            x=[signal.get("confirmation_time", signal["time"])],
            y=[signal["price"]],
            mode="markers",
            marker=dict(symbol="diamond", size=12,
                        color="green" if signal["direction"] == "bullish" else "red",
                        line=dict(color="black", width=1)),
            name="BOS/CHoCH Confirm",
            showlegend=False
        ))

y_vals = [df["low"].min(), df["high"].max()]
if zones:
    y_vals.extend([zones["discount"], zones["premium"], zones["equilibrium_top"], zones["equilibrium_bottom"]])
yaxis_range = [min(y_vals) * 0.999, max(y_vals) * 1.001]


# Get the last confirmed swing high and low in the visible range
visible_swings = [s for s in all_classified if visible_start <= s["time"] <= visible_end]
# Get the most recent HH and LL only — not LH/HL
last_hh = next((s for s in reversed(visible_swings) if s["label"] == "HH"), None)
last_ll = next((s for s in reversed(visible_swings) if s["label"] == "LL"), None)


# Strong/Weak High from last HH or LH
if last_hh:
    fig.add_shape(type="line",
        x0=last_hh["time"], x1=df["start_time"].max(),
        y0=last_hh["price"], y1=last_hh["price"],
        line=dict(color="red", width=1, dash="solid"),
        layer="below"
    )
    fig.add_annotation(x=df["start_time"].max(), y=last_hh["price"],
        text="Weak High" if bias == 1 else "Strong High",
        showarrow=False, font=dict(size=10, color="red"),
        xanchor="right", yanchor="bottom"
    )

if last_ll:
    fig.add_shape(type="line",
        x0=last_ll["time"], x1=df["start_time"].max(),
        y0=last_ll["price"], y1=last_ll["price"],
        line=dict(color="green", width=1, dash="solid"),
        layer="below"
    )
    fig.add_annotation(x=df["start_time"].max(), y=last_ll["price"],
        text="Strong Low" if bias == 1 else "Weak Low",
        showarrow=False, font=dict(size=10, color="green"),
        xanchor="right", yanchor="top"
    )


fig.update_layout(
    height=700,
    title=f"{timeframe}m Candle Chart – {selected_epic}",
    yaxis=dict(range=yaxis_range),
    xaxis_title="Time",
    yaxis_title="Price",
    xaxis=dict(
    type="date",
    tickformat="%H:%M\n%b %d",
    showgrid=True,
    rangeslider=dict(visible=False),
    rangebreaks=[
            # Skip weekends explicitly (Saturday and most of Sunday)
            dict(pattern="day of week", bounds=[6, 1]),  # 6=Saturday, 1=Monday

            # Skip the specific break: Fri 22:45 to Sun 22:00
            dict(bounds=["2025-05-23 23:00", "2025-05-25 22:00"])  # Use your actual dates
        ]
    ),
    margin=dict(l=40, r=40, t=50, b=40),
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

if bias is not None:
    st.markdown(f"### Market Structure Bias: {'🟢 Bullish' if bias == 1 else '🔴 Bearish'}")
if overall_trend is not None:
    st.markdown(f"### 1h Trend Bias: {'🟢 Bullish' if overall_trend == 'uptrend' else '🔴 Bearish'}")

if refresh_minutes > 0:
    total_seconds = refresh_minutes * 60
    with st.empty():
        for i in range(total_seconds, 0, -1):
            mins, secs = divmod(i, 60)
            st.info(f"⏳ Refreshing in {mins:02}:{secs:02} minutes")
            time.sleep(1)
        st.rerun()
