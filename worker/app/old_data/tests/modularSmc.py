import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas_ta as ta
import pytz
import plotly.graph_objects as go
from PIL import Image
from scipy.stats import linregress
from typing import Optional
import sys
import os
import time
from datetime import timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.smc_structure import detect_swing_pivots, detect_structure_signals_luxalgo, mark_confirmed_swings, get_most_recent_confirmed_hh_ll, calculate_zones,calculate_premium_discount_zones, determine_internal_trend, get_most_recent_hh_ll_from_swings, select_recent_hh_ll_structure



# --- Config ---
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

stockholm = pytz.timezone("Europe/Stockholm")
LOCAL_TZ = pytz.timezone("Europe/Stockholm")

# --- Fetch candles ---
def fetch_candle_data(engine, epic, timeframe=15, lookback_hours=500):
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    source_tf = 5 if timeframe == 15 else timeframe

    query = text("""
        SELECT start_time, open, high, low, close
        FROM ig_candles
        WHERE epic = :epic
          AND timeframe = :timeframe
          AND start_time >= :since
        ORDER BY start_time ASC
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {
            "epic": epic,
            "timeframe": source_tf,
            "since": since
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if df.empty:
        raise ValueError(f"No data returned for epic={epic}, timeframe={source_tf}")

    df['start_time'] = pd.to_datetime(df['start_time'])

    if timeframe == 15 and source_tf == 5:
        df.set_index("start_time", inplace=True)
        df = df.resample("15min", label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()

    return df.reset_index(drop=True)

def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return ta.ema(series, length=length)

def compute_macd(series: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame with 'macd_line', 'signal_line', and 'histogram'.
    """
    macd_df = ta.macd(series)
    return macd_df.rename(columns={
        f"MACD_12_26_9": "macd_line",
        f"MACDs_12_26_9": "signal_line",
        f"MACDh_12_26_9": "histogram"
    })

def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    return ta.rsi(series, length=length)

def detect_ema_alerts_from_indicators(df: pd.DataFrame, epic: str = "", eps: float = 1e-8) -> pd.DataFrame:
    df = df.sort_values('start_time').reset_index(drop=True)

    df["prev_close"] = df["close"].shift(1)
    df["prev_ema_9"] = df["ema_9"].shift(1)

    # Bull logic
    df['bull_cross'] = (df['prev_close'] < df['prev_ema_9'] - eps) & (df['close'] > df['ema_9'] + eps)
    df['bull_condition'] = (df['close'] > df['ema_21'] + eps) & (df['ema_9'] > df['ema_21'] + eps)

    df['bull_alert'] = df['bull_cross'] & df['bull_condition']

    # Bear logic
    df['bear_cross'] = (df['prev_close'] > df['prev_ema_9'] + eps) & (df['close'] < df['ema_9'] - eps)
    df['bear_condition'] = (df['close'] < df['ema_21'] - eps) & (df['ema_9'] < df['ema_21'] - eps) 
    df['bear_alert'] = df['bear_cross'] & df['bear_condition']

    return df

def detect_ema9_21_cross_confirmed(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    df = df.sort_values("start_time").reset_index(drop=True)

    # Previous values to detect the crossover event
    df["prev_ema_9"] = df["ema_9"].shift(1)
    df["prev_ema_21"] = df["ema_21"].shift(1)
    df["next_close"] = df["close"].shift(-1)

    # === Bullish: EMA12 crosses above EMA50, next close confirms
    df["bull_ema_cross"] = (
        (df["prev_ema_9"] < df["prev_ema_21"] - eps) &
        (df["ema_9"] > df["ema_21"] + eps) &
        (df["next_close"] > df["ema_9"] + eps)
    )

    # === Bearish: EMA12 crosses below EMA50, next close confirms
    df["bear_ema_cross"] = (
        (df["prev_ema_9"] > df["prev_ema_21"] + eps) &
        (df["ema_9"] < df["ema_21"] - eps) &
        (df["next_close"] < df["ema_9"] - eps)
    )

    return df

def compute_alligator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Williams Alligator indicator and a trend filter.
    
    Adds the following columns to df:
    - alligator_jaw (SMMA 13 shifted 8)
    - alligator_teeth (SMMA 8 shifted 5)
    - alligator_lips (SMMA 5 shifted 3)
    - is_trending (True if lips > teeth > jaw or lips < teeth < jaw)
    """
    # Calculate SMMA (equivalent to EMA with smoothing) with shift
    df["alligator_jaw"] = df["close"].ewm(span=13, adjust=False).mean().shift(8)
    df["alligator_teeth"] = df["close"].ewm(span=8, adjust=False).mean().shift(5)
    df["alligator_lips"] = df["close"].ewm(span=5, adjust=False).mean().shift(3)

    # Determine if the market is trending
    df["is_trending"] = (
        (df["alligator_lips"] > df["alligator_teeth"]) & 
        (df["alligator_teeth"] > df["alligator_jaw"])
    ) | (
        (df["alligator_lips"] < df["alligator_teeth"]) & 
        (df["alligator_teeth"] < df["alligator_jaw"])
    )

    return df

def compute_macd(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df[f"macd_line{suffix}"] = macd["MACD_12_26_9"]
    df[f"signal_line{suffix}"] = macd["MACDs_12_26_9"]
    df[f"macd_histogram{suffix}"] = macd["MACDh_12_26_9"]
    return df

# --- Plot and Save ---
def plot_alert_chart(df, epic, timeframe, image_path, zones=None, position_note=None, last_alert=None):
    print("time to plot the chart")
    offset = 0.01 if "JPY" in epic else 0.0001

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(x=df['start_time'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'],
                                 increasing_line_color="green", decreasing_line_color="red"))

    # EMA lines
    fig.add_trace(go.Scatter(x=df['start_time'], y=df['ema_12'], mode="lines", name="EMA12", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df['start_time'], y=df['ema_50'], mode="lines", name="EMA50", line=dict(color="orange")))


    # Single alert marker from last_alert
    if last_alert:
        condition = last_alert.get("condition", "")
        alert_type = condition.replace("_alert", "")
        color = "green" if alert_type == "bull" else "red"
        symbol = "triangle-up" if alert_type == "bull" else "triangle-down"
        y = last_alert["price"] + offset if alert_type == "bull" else last_alert["price"] - offset

        fig.add_trace(go.Scatter(
            x=[last_alert["timestamp"]],
            y=[y],
            mode="markers+text",
            marker=dict(color=color, size=12, symbol=symbol),
            textposition="top center",
            name="Alert"
        ))

    # Layout
    fig.update_layout(
        title=f"{epic} - {timeframe}m EMA Alerts",
        xaxis_title="Time",
        yaxis_title="Price",
        plot_bgcolor="white",
        width=1000,
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangebreaks=[
                dict(pattern="day of week", bounds=[6, 1]),
                dict(bounds=["2025-05-23 23:00", "2025-05-25 22:00"])
            ]
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    # Save chart
    #fig.write_image(image_path)
    print(f"📈 Chart saved to {image_path}")


    # Save the image
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    img_bytes = fig.to_image(format="png", scale=2)
    with open(image_path, "wb") as f:
        f.write(img_bytes)
    print(f"✅ Chart saved to: {image_path}")
    return image_path

def get_last_alert_type(valid_alerts):
    if not valid_alerts:
        return None
    return valid_alerts[-1]

def is_exhaustion_candle(open_, high, low, close, wick_ratio=2.0):
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    return body > 0 and upper_wick > wick_ratio * body

def is_overextended(close, ema, threshold_pct=1.5):
    distance = abs(close - ema)
    return (distance / ema) * 100 > threshold_pct

def detect_ranges(df: pd.DataFrame, length: int = 20, atr_len: int = 14, mult: float = 1.0) -> pd.DataFrame:
    """
    Adds range_top, range_bottom, in_range, and breakout columns using pandas_ta ATR.
    """
    df = df.copy()

    # Moving average of close
    df["ma"] = df["close"].rolling(window=length).mean()

    # ATR using pandas_ta
    df["atr"] = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=atr_len)

    # Define range
    df["range_top"] = df["ma"] + df["atr"] * mult
    df["range_bottom"] = df["ma"] - df["atr"] * mult

    # Range conditions
    df["in_range"] = (df["close"] <= df["range_top"]) & (df["close"] >= df["range_bottom"])

    # Breakout signal
    df["breakout"] = 0
    df.loc[df["close"] > df["range_top"], "breakout"] = 1
    df.loc[df["close"] < df["range_bottom"], "breakout"] = -1

    return df


def slope_diff_1(df: pd.DataFrame, col: str, out_col: str = "slope_1") -> pd.DataFrame:
    df[out_col] = df[col].diff()
    df[out_col + "_falling"] = df[out_col] < 0
    df[out_col + "_rising"] = df[out_col] > 0
    return df

def slope_diff_2(df: pd.DataFrame, col: str, out_col: str = "slope_2") -> pd.DataFrame:
    df[out_col] = df[col] - df[col].shift(2)
    df[out_col + "_falling"] = df[out_col] < 0
    df[out_col + "_rising"] = df[out_col] > 0
    return df

def is_valid_structure(row: pd.Series, direction: str) -> bool:
    """
    Returns True if the structure is valid for the given direction.
    
    For 'bull':
        - close > EMA12
        - EMA12 > EMA50

    For 'bear':
        - close < EMA12
        - EMA12 < EMA50
    """
    if direction == "bull":
        return row["close"] > row["ema_12"] and row["ema_12"] > row["ema_50"]
    elif direction == "bear":
        return row["close"] < row["ema_12"] and row["ema_12"] < row["ema_50"]
    else:
        raise ValueError("Direction must be 'bull' or 'bear'")

def slope_linregress(
    df: pd.DataFrame,
    col: str,
    window: int = 3,
    out_col: str = "slope_lr"
) -> pd.DataFrame:
    """
    Compute the linear regression slope over a rolling window for a given column.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - col (str): The name of the column to compute the slope on (e.g., "ema_12").
    - window (int): Number of candles to use for the linear regression. Default is 3.
    - out_col (str): Base name for the output slope column (default "slope_lr").
    
    Returns:
    - pd.DataFrame: The original dataframe with three new columns:
        - out_col: the raw slope value at each row
        - out_col + "_falling": True if slope < 0 (downward trend)
        - out_col + "_rising": True if slope > 0 (upward trend)
    
    Notes:
    - The first `window` rows will contain NaNs in the slope output.
    - Slope is calculated using scipy.stats.linregress over the window.
    """
    slopes = [None] * window  # Pad the beginning with None to preserve index alignment

    for i in range(window, len(df)):
        y = df[col].iloc[i - window:i]      # Y-axis: actual EMA values
        x = range(window)                   # X-axis: 0, 1, 2, ..., window-1
        slope, *_ = linregress(x, y)        # Compute slope of best-fit line
        slopes.append(slope)                # Store the slope

    df[out_col] = slopes
    df[out_col + "_falling"] = df[out_col] < 0
    df[out_col + "_rising"] = df[out_col] > 0

    return df

def is_near_local_extreme(df, idx, direction: str, window=20, atr_col="atr", atr_mult=1.0, fallback_pct=0.5) -> bool:
    if idx < window or idx >= len(df):
        return False

    row = df.iloc[idx]
    close = row["close"]
    atr = row.get(atr_col, None)
    threshold = atr * atr_mult if pd.notnull(atr) and atr > 0 else close * (fallback_pct / 100.0)

    if direction == "bull":
        recent_high = df["high"].iloc[idx - window:idx].max()
        is_near = (recent_high - close) <= threshold
        print(f"[{row['start_time']}] BULL | Close={close:.5f}, RecentHigh={recent_high:.5f}, Threshold={threshold:.5f}, ATR={atr if pd.notnull(atr) else 'N/A'} → Near={is_near}")
        return is_near

    elif direction == "bear":
        recent_low = df["low"].iloc[idx - window:idx].min()
        is_near = (close - recent_low) <= threshold
        print(f"[{row['start_time']}] BEAR | Close={close:.5f}, RecentLow={recent_low:.5f}, Threshold={threshold:.5f}, ATR={atr if pd.notnull(atr) else 'N/A'} → Near={is_near}")
        return is_near

    else:
        raise ValueError("Direction must be 'bull' or 'bear'")

def is_ema_distance_exceeded(row, ema_fast_col="ema_12", ema_slow_col="ema_50", max_pct: float = 0.05) -> bool:
    """
    Checks if the distance between fast and slow EMA is too large.

    Parameters:
    - row (pd.Series): A row of the DataFrame.
    - ema_fast_col (str): Column name of the fast EMA.
    - ema_slow_col (str): Column name of the slow EMA.
    - max_pct (float): Max allowed EMA distance as a fraction (e.g., 0.05 = 5%).

    Returns:
    - bool: True if distance exceeds the allowed threshold.
    """
    fast = row.get(ema_fast_col)
    slow = row.get(ema_slow_col)

    if fast is None or slow is None or slow == 0:
        return False  # Fail-safe

    distance_pct = abs(fast - slow) / slow * 100
    if distance_pct > (max_pct * 100):
        print(f"[{row['start_time']}] 🚫 EMA distance too wide → EMA12={fast:.5f}, EMA50={slow:.5f}, Dist={distance_pct:.2f}%")
        return True
    return False


def confirms_strong_direction(df: pd.DataFrame, idx: int, direction: str, min_body_pct=0.2) -> bool:
    if idx + 1 >= len(df): return False

    row = df.iloc[idx]
    next_candle = df.iloc[idx + 1]

    body = abs(next_candle["close"] - next_candle["open"])
    candle_range = next_candle["high"] - next_candle["low"]
    body_pct = body / (candle_range + 1e-6)

    if direction == "bull":
        return next_candle["close"] > row["close"] and body_pct >= min_body_pct
    elif direction == "bear":
        return next_candle["close"] < row["close"] and body_pct >= min_body_pct
    return False


def detect_valid_alerts(
    df: pd.DataFrame,
    use_ema_alert=False,
    use_ema_cross_confirm_filter=False,
    use_trend_filter=False,
    use_macd_5m_filter=False,
    use_macd_15m_filter=False,
    use_macd_1h_filter=False,
    use_exhaustion_filter=False,
    use_overextension_filter=False,
    use_range_filter=False,
    use_slope_filter=False,
    slope_method="diff_2",  # "diff_1", "diff_2", or "linreg"
    use_local_extreme_filter=False,
    atr_mult_extreme=1.0,
    use_ema_distance_filter=False,
    max_ema_distance_pct=0.5,
    use_follow_through_filter=False,
    use_zone_filter: bool = False,
    zones: Optional[dict] = None,
    debug=False
) -> list[dict]:
    alerts = []

    # Select correct slope column
    base_slope_col = (
        f"ema12_slope{'1' if slope_method == 'diff_1' else '2' if slope_method == 'diff_2' else '_lr'}"
    )
    slope_col_bull = base_slope_col + "_rising"
    slope_col_bear = base_slope_col + "_falling"

    for idx, row in df.iterrows():
        alert_time = row["start_time"]
        alert_price = row["close"]
        alert_type, condition = None, None

        # === Bullish logic (accept either alert method) ===
        bull_trigger = (
            (use_ema_alert and row.get("bull_alert", False)) or
            (use_ema_cross_confirm_filter and row.get("bull_ema_cross", False)) or
            ((not use_ema_alert and not use_ema_cross_confirm_filter) and row.get("bull_alert", False))
        )

        if bull_trigger:
            if use_zone_filter and zones and alert_price > zones["equilibrium_bottom"]:
                if debug: print(f"[{alert_time}] 🚫 Bull: above equilibrium zone")
                continue
            if use_ema_distance_filter and is_ema_distance_exceeded(row, "ema_12", "ema_50", max_pct=max_ema_distance_pct):
                if debug: print(f"[{alert_time}] 🚫 BULL: EMA distance too wide")
                continue
            if use_local_extreme_filter and is_near_local_extreme(df, idx, direction="bull", atr_mult=atr_mult_extreme):
                if debug: print(f"[{alert_time}] 🚫 Bull: too close to recent local high")
                continue
            if use_range_filter and row.get("in_range", False):
                if debug: print(f"[{alert_time}] 🚫 Bull: inside range")
                continue
            if use_trend_filter and not row.get("trend", False):
                if debug: print(f"[{alert_time}] ❌ Bull: 1H trend not bullish")
                continue
            if use_macd_5m_filter and row.get("macd_histogram_5m", 0) <= 0:
                if debug: print(f"[{alert_time}] ❌ Bull: 5m MACD ≤ 0")
                continue
            if use_macd_15m_filter and row.get("macd_histogram_15m", 0) <= 0:
                if debug: print(f"[{alert_time}] ❌ Bull: 15m MACD ≤ 0")
                continue
            if use_macd_1h_filter and row.get("macd_histogram_1h", 0) <= 0:
                if debug: print(f"[{alert_time}] ❌ Bull: 1H MACD ≤ 0")
                continue
            if use_exhaustion_filter and is_exhaustion_candle(row["open"], row["high"], row["low"], row["close"]):
                if debug: print(f"[{alert_time}] 🚫 Bull: exhaustion candle")
                continue
            if use_overextension_filter and is_overextended(row["close"], row["ema_12"], threshold_pct=1.5):
                if debug: print(f"[{alert_time}] 🚫 Bull: overextended from EMA12")
                continue
            if use_slope_filter and not row.get(slope_col_bull, False):
                if debug: print(f"[{alert_time}] ❌ Bull: EMA12 slope not rising ({slope_method})")
                continue
            if use_follow_through_filter and not confirms_strong_direction(df, idx, "bull"):
                if debug: print(f"[{alert_time}] 🚫 Bull: no bullish follow-through on next candle")
                continue


            alert_type, condition = "bull", "bull_alert"

        # === Bearish logic (accept either alert method) ===
        bear_trigger = (
            (use_ema_alert and row.get("bear_alert", False)) or
            (use_ema_cross_confirm_filter and row.get("bear_ema_cross", False)) or
            ((not use_ema_alert and not use_ema_cross_confirm_filter) and row.get("bear_alert", False))
        )

        if bear_trigger:
            if use_zone_filter and zones and alert_price < zones["equilibrium_top"]:
                if debug: print(f"[{alert_time}] 🚫 Bear: below equilibrium zone")
                continue
            if use_ema_distance_filter and is_ema_distance_exceeded(row, "ema_12", "ema_50", max_pct=max_ema_distance_pct):
                if debug: print(f"[{alert_time}] 🚫 BEAR: EMA distance too wide")
                continue
            if use_local_extreme_filter and is_near_local_extreme(df, idx, direction="bear", atr_mult=atr_mult_extreme):
                if debug: print(f"[{alert_time}] 🚫 Bear: too close to recent local low")
                continue
            if use_range_filter and row.get("in_range", False):
                if debug: print(f"[{alert_time}] 🚫 Bear: inside range")
                continue
            if use_trend_filter and row.get("trend", True):
                if debug: print(f"[{alert_time}] ❌ Bear: 1H trend still bullish")
                continue
            if use_macd_5m_filter and row.get("macd_histogram_5m", 0) >= 0:
                if debug: print(f"[{alert_time}] ❌ Bear: 5m MACD ≥ 0")
                continue
            if use_macd_15m_filter and row.get("macd_histogram_15m", 0) >= 0:
                if debug: print(f"[{alert_time}] ❌ Bear: 15m MACD ≥ 0")
                continue
            if use_macd_1h_filter and row.get("macd_histogram_1h", 0) >= 0:
                if debug: print(f"[{alert_time}] ❌ Bear: 1H MACD ≥ 0")
                continue
            if use_exhaustion_filter and is_exhaustion_candle(row["open"], row["low"], row["high"], row["close"]):
                if debug: print(f"[{alert_time}] 🚫 Bear: exhaustion candle")
                continue
            if use_overextension_filter and is_overextended(row["close"], row["ema_12"], threshold_pct=1.5):
                if debug: print(f"[{alert_time}] 🚫 Bear: overextended from EMA12")
                continue
            if use_slope_filter and not row.get(slope_col_bear, False):
                if debug: print(f"[{alert_time}] ❌ Bear: EMA12 slope not falling ({slope_method})")
                continue
            if use_follow_through_filter and not confirms_strong_direction(df, idx, "bear"):
                if debug: print(f"[{alert_time}] 🚫 Bear: no bearish follow-through on next candle")
                continue


            alert_type, condition = "bear", "bear_alert"

        # === Finalize
        if alert_type:
            alerts.append({
                "timestamp": alert_time,
                "type": alert_type,
                "price": alert_price,
                "condition": condition
            })

    return alerts

def run_alert_pipeline(epic: str, engine, debug=False) -> tuple[list[dict], pd.DataFrame]:
    df_5m = fetch_candle_data(engine, epic, 5, 1000)
    df_15m = fetch_candle_data(engine, epic, 15, 1000)
    df_1h = fetch_candle_data(engine, epic, 60, lookback_hours=200)

    for df in [df_5m, df_15m, df_1h]:
        df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)

    # Indicators
    df_5m["ema_12_5m"] = compute_ema(df_5m["close"], 12)
    df_5m["ema_50_5m"] = compute_ema(df_5m["close"], 50)
    df_5m = compute_macd(df_5m, suffix="_5m")

    df_15m["ema_9"] = compute_ema(df_15m["close"], 9)
    df_15m["ema_21"] = compute_ema(df_15m["close"], 21)
    df_15m = compute_alligator(df_15m)
    df_15m = compute_macd(df_15m, suffix="_15m")

    # detect EMA alerts
    df_15m = detect_ema_alerts_from_indicators(df_15m)
    df_15m = detect_ema9_21_cross_confirmed(df_15m)

    df_15m = slope_diff_1(df_15m, "ema_9", out_col="ema9_slope1")
    df_15m = slope_diff_2(df_15m, "ema_9", out_col="ema9_slope2")
    df_15m = slope_linregress(df_15m, "ema_9", window=3, out_col="ema9_slope_lr") 

    df_1h["ema_12_1h"] = compute_ema(df_1h["close"], 12)
    df_1h["ema_50_1h"] = compute_ema(df_1h["close"], 50)
    df_1h["ema_100_1h"] = compute_ema(df_1h["close"], 100)
    df_1h = compute_macd(df_1h, suffix="_1h")
    df_1h = compute_alligator(df_1h)
    df_1h["trend"] = df_1h["close"] > df_1h["ema_100_1h"]

    # Merge into 15m base
    df_15m = pd.merge_asof(
        df_15m.sort_values("start_time"),
        df_5m[["start_time", "macd_histogram_5m", "ema_12_5m", "ema_50_5m"]].sort_values("start_time"),
        on="start_time", direction="backward", tolerance=pd.Timedelta("15min")
    )
    df_15m = pd.merge_asof(
        df_15m.sort_values("start_time"),
        df_1h[["start_time", "macd_histogram_1h", "trend", "ema_12_1h", "ema_50_1h"]].sort_values("start_time"),
        on="start_time", direction="backward", tolerance=pd.Timedelta("1h")
    )
    df_15m["trend"] = df_15m["trend"].where(pd.notnull(df_15m["trend"]), False).astype(bool)

    # add ATR to DF
    df_15m["atr"] = ta.atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14)

    # 🔍 Add range detection
    df_15m = detect_ranges(df_15m, length=20, atr_len=500, mult=1.0)

    return df_15m

def select_recent_hh_ll_structure(swings, bar_limit=192, max_age_minutes=2880, debug=False):
    now = pd.Timestamp.utcnow().tz_convert("Europe/Stockholm")
    recent_swings = swings.copy()
    recent_swings["time_utc"] = recent_swings["time"].dt.tz_convert("Europe/Stockholm")
    recent_swings = recent_swings.sort_values(by="time_utc", ascending=False).head(bar_limit)
    recent_swings = recent_swings[recent_swings["time_utc"] >= now - timedelta(minutes=max_age_minutes)]

    hh_candidates = recent_swings[recent_swings["label"] == "HH"].sort_values(by="time_utc", ascending=False)
    ll_candidates = recent_swings[recent_swings["label"] == "LL"].sort_values(by="time_utc", ascending=False)

    for hh in hh_candidates.itertuples():
        for ll in ll_candidates.itertuples():
            if hh.time_utc > ll.time_utc and hh.price > ll.price:
                if debug:
                    print("✅ Found valid HH > LL within window:")
                    print(f"  HH: {hh.price} @ {hh.time}")
                    print(f"  LL: {ll.price} @ {ll.time}")
                return {
                    "top": hh.price,
                    "bottom": ll.price,
                    "top_source": {"price": hh.price, "time": hh.time},
                    "bottom_source": {"price": ll.price, "time": ll.time}
                }

    if debug:
        print("❌ No valid HH > LL pair found within constraints.")
    return None

def detect_confirmed_swing_highs_lows(df, left=3, right=3):
    highs = []
    lows = []
    for i in range(left, len(df) - right):
        window_high = df["high"].iloc[i - left : i + right + 1]
        center_high = df["high"].iloc[i]
        if center_high == max(window_high):
            highs.append({
                "type": "high",
                "label": "HH",
                "price": center_high,
                "time": df["start_time"].iloc[i]
            })

        window_low = df["low"].iloc[i - left : i + right + 1]
        center_low = df["low"].iloc[i]
        if center_low == min(window_low):
            lows.append({
                "type": "low",
                "label": "LL",
                "price": center_low,
                "time": df["start_time"].iloc[i]
            })

    return sorted(highs + lows, key=lambda x: x["time"])

def select_highest_HH_with_time_fallback(swings_df: pd.DataFrame, bar_limit: int = 192, max_age_minutes: int = 2880, debug: bool = False):
    recent_swings = swings_df.tail(bar_limit).copy()
    recent_swings["time"] = pd.to_datetime(recent_swings["time"])
    now = recent_swings["time"].max()
    recent_swings["age_minutes"] = (now - recent_swings["time"]).dt.total_seconds() / 60

    hh_swings = recent_swings[(recent_swings["label"] == "HH") & (recent_swings["age_minutes"] <= max_age_minutes)]
    if not hh_swings.empty:
        highest = hh_swings.loc[hh_swings["price"].idxmax()]
        if debug:
            print(f"✅ Using HH within time limit: {highest['price']} @ {highest['time']}")
        return highest.to_dict()

    # fallback: just take highest HH in last bars regardless of age
    hh_fallback = recent_swings[recent_swings["label"] == "HH"]
    if not hh_fallback.empty:
        highest = hh_fallback.loc[hh_fallback["price"].idxmax()]
        if debug:
            print(f"⚠️ Using fallback HH (ignoring age): {highest['price']} @ {highest['time']}")
        return highest.to_dict()

    return None

def select_lowest_LL_with_time_fallback(swings_df: pd.DataFrame, bar_limit: int = 192, max_age_minutes: int = 2880, debug: bool = False):
    recent_swings = swings_df.tail(bar_limit).copy()
    recent_swings["time"] = pd.to_datetime(recent_swings["time"])
    now = recent_swings["time"].max()
    recent_swings["age_minutes"] = (now - recent_swings["time"]).dt.total_seconds() / 60

    ll_swings = recent_swings[(recent_swings["label"] == "LL") & (recent_swings["age_minutes"] <= max_age_minutes)]
    if not ll_swings.empty:
        lowest = ll_swings.loc[ll_swings["price"].idxmin()]
        if debug:
            print(f"✅ Using LL within time limit: {lowest['price']} @ {lowest['time']}")
        return lowest.to_dict()

    # fallback: just take lowest LL in last bars regardless of age
    ll_fallback = recent_swings[recent_swings["label"] == "LL"]
    if not ll_fallback.empty:
        lowest = ll_fallback.loc[ll_fallback["price"].idxmin()]
        if debug:
            print(f"⚠️ Using fallback LL (ignoring age): {lowest['price']} @ {lowest['time']}")
        return lowest.to_dict()

    return None

def calculate_zones(hh_price: float, ll_price: float):
    zone_height = hh_price - ll_price
    third = zone_height / 3
    return {
        "premium_top": hh_price,
        "premium_bottom": hh_price - third,
        "equilibrium_top": hh_price - third,
        "equilibrium_bottom": ll_price + third,
        "discount_top": ll_price + third,
        "discount_bottom": ll_price,
    }


# === Entry Point ===
if __name__ == "__main__":
    
    
    epic = "CS.D.GBPUSD.MINI.IP"
    image_path = f"/app/tradesetups/trade_snapshot_{epic.replace('.', '_')}.png"
    df_result = run_alert_pipeline(epic, engine, debug=True)

    # Step 1: Detect swing pivots using the LuxAlgo-style function
    swings = detect_confirmed_swing_highs_lows(df_result, left=3, right=3)
    
    # Step 2: Confirm structure signals (BOS/CHoCH)
    signals = detect_structure_signals_luxalgo(swings, df_result)
    swings_df = pd.DataFrame(swings)
    swings_df["time"] = pd.to_datetime(swings_df["time"]).dt.tz_convert("Europe/Stockholm")
    top = select_highest_HH_with_time_fallback(swings_df, bar_limit=96, max_age_minutes=1440, debug=True)
    bottom = select_lowest_LL_with_time_fallback(swings_df, bar_limit=96, max_age_minutes=1440, debug=True)
    if top and bottom:
        zones = calculate_zones(top["price"], bottom["price"])
        print(f"📍 HH used: {top['price']} @ {top['time']}")
        print(f"📍 LL used: {bottom['price']} @ {bottom['time']}")
        print(zones)
    else:
        zones = None
        print("❌ No valid HH/LL found.")
    
    
    
    trend_bias = determine_internal_trend(signals, debug=True)
    
    
    alerts = detect_valid_alerts(df_result , debug=True,use_macd_5m_filter=False, use_macd_15m_filter=False, use_macd_1h_filter=False,
                                 use_trend_filter=False, use_slope_filter=False, 
                                 slope_method="linreg",use_ema_distance_filter=False, max_ema_distance_pct=0.0015,
                                 use_follow_through_filter=False,use_zone_filter=True,zones=zones)
    chart_df = df_result.tail(100)
    

    if alerts:
        print(f"✅ {len(alerts)} alert(s) found:")
        for alert in alerts:
            print(f"🔔 {alert['type'].upper()} at {alert['price']} ({alert['timestamp']})")
        
        last = alerts[-1]
        print(f"\n📌 Last alert: {last['type'].upper()} at {last['price']} ({last['timestamp']})")
        
        #img_path = plot_alert_chart(chart_df, epic, 15, image_path,last_alert=last)
    else:
        print("🚫 No confirmed alerts found.")