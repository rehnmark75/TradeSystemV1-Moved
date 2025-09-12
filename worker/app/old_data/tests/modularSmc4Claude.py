import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas_ta as ta
import pytz
import plotly.graph_objects as go
from PIL import Image
from scipy.stats import linregress
from typing import List, Dict, Optional, Any
import sys
import os
import time
from datetime import timedelta
import warnings
import numpy as np

warnings.simplefilter("error", RuntimeWarning)
# --- Config ---
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

stockholm = pytz.timezone("Europe/Stockholm")
LOCAL_TZ = pytz.timezone("Europe/Stockholm")

def safe_gt(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    result = pd.Series(False, index=series_a.index, dtype=bool)
    mask = series_a.notnull() & series_b.notnull()
    result.loc[mask] = (series_a[mask] > series_b[mask]).astype(bool)
    return result

def safe_lt(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    result = pd.Series(False, index=series_a.index, dtype=bool)
    mask = series_a.notnull() & series_b.notnull()
    result.loc[mask] = (series_a[mask] < series_b[mask]).astype(bool)
    return result


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

def detect_ema_alerts_from_indicators(
    df: pd.DataFrame,
    fast_ema_col: str,
    slow_ema_col: str,
    eps: float = 1e-8
) -> pd.DataFrame:
    df = df.sort_values('start_time').reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    prev_fast_col = f"prev_{fast_ema_col}"
    df[prev_fast_col] = df[fast_ema_col].shift(1)

    df["bull_cross"] = (
        safe_lt(df["prev_close"], df[prev_fast_col] - eps) &
        safe_gt(df["close"], df[fast_ema_col] + eps)
    )

    df["bull_condition"] = (
        safe_gt(df["close"], df[slow_ema_col] + eps) &
        safe_gt(df[fast_ema_col], df[slow_ema_col] + eps)
    )

    df["bull_alert"] = df["bull_cross"] & df["bull_condition"]

    df["bear_cross"] = (
        safe_gt(df["prev_close"], df[prev_fast_col] + eps) &
        safe_lt(df["close"], df[fast_ema_col] - eps)
    )

    df["bear_condition"] = (
        safe_lt(df["close"], df[slow_ema_col] - eps) &
        safe_lt(df[fast_ema_col], df[slow_ema_col] - eps)
    )

    df["bear_alert"] = df["bear_cross"] & df["bear_condition"]

    return df


def detect_ema_cross_confirmed(
    df: pd.DataFrame,
    fast_ema_col: str,
    slow_ema_col: str,
    eps: float = 1e-8
) -> pd.DataFrame:
    df = df.sort_values("start_time").reset_index(drop=True)

    prev_fast_col = f"prev_{fast_ema_col}"
    prev_slow_col = f"prev_{slow_ema_col}"

    df[prev_fast_col] = df[fast_ema_col].shift(1)
    df[prev_slow_col] = df[slow_ema_col].shift(1)

    # Initialize with safe defaults
    df["bull_ema_cross"] = False
    df["bear_ema_cross"] = False

    # Bullish crossover: fast crosses above slow
    bull_mask = safe_lt(df[prev_fast_col], df[prev_slow_col] - eps) & safe_gt(df[fast_ema_col], df[slow_ema_col] + eps)
    df["bull_ema_cross"] = bull_mask

    # Bearish crossover: fast crosses below slow
    bear_mask = safe_gt(df[prev_fast_col], df[prev_slow_col] + eps) & safe_lt(df[fast_ema_col], df[slow_ema_col] - eps)
    df["bear_ema_cross"] = bear_mask

    return df

def compute_atr(df: pd.DataFrame, period=14, suffix: str = "") -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    col_name = f"atr{suffix}" if suffix else "atr"
    df[col_name] = atr
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
def plot_alert_chart(df, epic, timeframe, image_path, zones=None, position_note=None, last_alert=None, ema_columns=None):
    print("time to plot the chart")
    offset = 0.01 if "JPY" in epic else 0.0001

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=df['start_time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color="green", decreasing_line_color="red"
    ))

    # EMA lines
    # Dynamically plot all EMA lines if available
    for ema_col in ema_columns:
        if ema_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['start_time'],
                y=df[ema_col],
                mode="lines+text",
                name=ema_col.upper(),
                text=[ema_col.upper()] + [""] * (len(df) - 1),
                textposition="bottom right",
                line=dict(width=1, dash="dot")
            ))
        else:
            print(f"⚠️ EMA column missing in df: {ema_col}")

    # Zones (dotted lines)
    if zones:
        for name, value in zones.items():
            fig.add_trace(go.Scatter(
                x=[df["start_time"].iloc[0], df["start_time"].iloc[-1]],
                y=[value, value],
                mode="lines",
                name=name.replace("_", " ").title(),
                line=dict(dash="dot", width=1),
                showlegend=True
            ))

    # Alert marker
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
    fast_ema_length: int = 12,
    slow_ema_length: int = 50,
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
    slope_method="diff_2",
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

    fast_ema_col = f"ema_{fast_ema_length}"
    slow_ema_col = f"ema_{slow_ema_length}"
    slope_base = f"{fast_ema_col}_slope{'1' if slope_method == 'diff_1' else '2' if slope_method == 'diff_2' else '_lr'}"
    slope_col_bull = f"{slope_base}_rising"
    slope_col_bear = f"{slope_base}_falling"

    for idx, row in df.iterrows():
        alert_time = row["start_time"]
        alert_price = row["close"]
        alert_type, condition = None, None

        bull_trigger = (
            (use_ema_alert and row.get("bull_alert", False)) or
            (use_ema_cross_confirm_filter and row.get("bull_ema_cross", False)) or
            ((not use_ema_alert and not use_ema_cross_confirm_filter) and row.get("bull_alert", False))
        )

        if bull_trigger:
            if use_zone_filter and zones and alert_price > zones.get("equilibrium_bottom", float("inf")):
                if debug: print(f"[{alert_time}] 🚫 Bull: above equilibrium zone")
                continue

            fast_val = row.get(fast_ema_col)
            slow_val = row.get(slow_ema_col)

            if use_ema_distance_filter and pd.notnull(fast_val) and pd.notnull(slow_val):
                if is_ema_distance_exceeded(row, fast_ema_col, slow_ema_col, max_pct=max_ema_distance_pct):
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
            if use_overextension_filter and pd.notnull(fast_val) and is_overextended(row["close"], fast_val, threshold_pct=1.5):
                if debug: print(f"[{alert_time}] 🚫 Bull: overextended from {fast_ema_col}")
                continue
            if use_slope_filter and not row.get(slope_col_bull, False):
                if debug: print(f"[{alert_time}] ❌ Bull: EMA slope not rising ({slope_method})")
                continue
            if use_follow_through_filter and not confirms_strong_direction(df, idx, "bull"):
                if debug: print(f"[{alert_time}] 🚫 Bull: no bullish follow-through on next candle")
                continue

            alert_type, condition = "bull", "bull_alert"

        bear_trigger = (
            (use_ema_alert and row.get("bear_alert", False)) or
            (use_ema_cross_confirm_filter and row.get("bear_ema_cross", False)) or
            ((not use_ema_alert and not use_ema_cross_confirm_filter) and row.get("bear_alert", False))
        )

        if bear_trigger:
            if use_zone_filter and zones and alert_price < zones.get("equilibrium_top", -float("inf")):
                if debug: print(f"[{alert_time}] 🚫 Bear: below equilibrium zone")
                continue

            fast_val = row.get(fast_ema_col)
            slow_val = row.get(slow_ema_col)

            if use_ema_distance_filter and pd.notnull(fast_val) and pd.notnull(slow_val):
                if is_ema_distance_exceeded(row, fast_ema_col, slow_ema_col, max_pct=max_ema_distance_pct):
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
            if use_overextension_filter and pd.notnull(fast_val) and is_overextended(row["close"], fast_val, threshold_pct=1.5):
                if debug: print(f"[{alert_time}] 🚫 Bear: overextended from {fast_ema_col}")
                continue
            if use_slope_filter and not row.get(slope_col_bear, False):
                if debug: print(f"[{alert_time}] ❌ Bear: EMA slope not falling ({slope_method})")
                continue
            if use_follow_through_filter and not confirms_strong_direction(df, idx, "bear"):
                if debug: print(f"[{alert_time}] 🚫 Bear: no bearish follow-through on next candle")
                continue

            alert_type, condition = "bear", "bear_alert"

        if alert_type:
            alerts.append({
                "timestamp": alert_time,
                "type": alert_type,
                "price": alert_price,
                "condition": condition
            })

    return alerts


def run_alert_pipeline(
    epic: str,
    engine,
    debug=False,
    ema_config: dict = None,
    base_timeframe: str = "15m"
) -> pd.DataFrame:
    if ema_config is None:
        ema_config = {
            "5m": [12, 50],
            "15m": [9, 21],
            "1h": [12, 50, 100]
        }

    df_5m = fetch_candle_data(engine, epic, 5, 1000)
    df_15m = fetch_candle_data(engine, epic, 15, 1000)
    df_1h = fetch_candle_data(engine, epic, 60, lookback_hours=200)

    for df in [df_5m, df_15m, df_1h]:
        df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)

    def apply_emas(df, timeframe_key, suffix):
        ema_cols = []
        for length in ema_config.get(timeframe_key, []):
            col = f"ema_{length}_{suffix}"
            df[col] = compute_ema(df["close"], length)
            ema_cols.append(length)
        return df, ema_cols

    df_5m, ema_5m_lengths = apply_emas(df_5m, "5m", "5m")
    df_5m = compute_macd(df_5m, suffix="_5m")

    df_15m, ema_15m_lengths = apply_emas(df_15m, "15m", "15m")
    df_15m = compute_macd(df_15m, suffix="_15m")
    df_15m = compute_alligator(df_15m)

    df_1h, ema_1h_lengths = apply_emas(df_1h, "1h", "1h")
    df_1h = compute_macd(df_1h, suffix="_1h")
    df_1h = compute_alligator(df_1h)
    df_1h = compute_atr(df_1h, period=14, suffix="_1h")

    for tf_key, df_var, ema_lengths, suffix in [
        ("5m", df_5m, ema_5m_lengths, "5m"),
        ("15m", df_15m, ema_15m_lengths, "15m"),
        ("1h", df_1h, ema_1h_lengths, "1h"),
    ]:
        if len(ema_lengths) >= 2:
            fast_len, slow_len = sorted(ema_lengths)[:2]
            fast_col = f"ema_{fast_len}_{suffix}"
            slow_col = f"ema_{slow_len}_{suffix}"
            df_var = detect_ema_alerts_from_indicators(df_var, fast_col, slow_col)
            df_var = detect_ema_cross_confirmed(df_var, fast_col, slow_col)

        if ema_lengths:
            ema_key = f"ema_{ema_lengths[0]}_{suffix}"
            df_var = slope_diff_1(df_var, ema_key, out_col=f"{ema_key}_slope1")
            df_var = slope_diff_2(df_var, ema_key, out_col=f"{ema_key}_slope2")
            df_var = slope_linregress(df_var, ema_key, window=3, out_col=f"{ema_key}_slope_lr")

        if tf_key == "5m":
            df_5m = df_var
        elif tf_key == "15m":
            df_15m = df_var
        elif tf_key == "1h":
            df_1h = df_var

    df_1h["trend"] = False
    if ema_1h_lengths:
        longest_ema_col = f"ema_{max(ema_1h_lengths)}_1h"
        if longest_ema_col in df_1h.columns:
            df_1h["trend"] = safe_gt(df_1h["close"], df_1h[longest_ema_col])

    dataframes = {"5m": df_5m, "15m": df_15m, "1h": df_1h}
    base_df = dataframes.get(base_timeframe, df_15m).sort_values("start_time")

    if base_timeframe != "5m":
        merge_cols_5m = [c for c in df_5m.columns if c.startswith("ema_") or "macd_histogram_5m" in c]
        df_5m_merge = df_5m[["start_time"] + merge_cols_5m].sort_values("start_time")
        base_df = pd.merge_asof(
            base_df, df_5m_merge, on="start_time", direction="backward", tolerance=pd.Timedelta("15min")
        )

    if base_timeframe != "1h":
        merge_cols_1h = [
            c for c in df_1h.columns
            if c.startswith("ema_") or "macd_histogram_1h" in c or c in {"trend", "atr_1h"}
        ]
        df_1h_merge = df_1h[["start_time"] + merge_cols_1h].sort_values("start_time")
        base_df = pd.merge_asof(
            base_df, df_1h_merge, on="start_time", direction="backward", tolerance=pd.Timedelta("1h")
        )

    base_df["trend"] = base_df["trend"].where(pd.notnull(base_df["trend"]), False).astype(bool)

    if base_timeframe == "15m":
        base_df["atr"] = ta.atr(base_df["high"], base_df["low"], base_df["close"], length=14)
        base_df = detect_ranges(base_df, length=20, atr_len=500, mult=1.0)

    return base_df, base_timeframe



# SMC functions


from dataclasses import dataclass
from enum import Enum

class SwingType(Enum):
    HIGH = "high"
    LOW = "low"

class SignalType(Enum):
    BOS = "BOS"  # Break of Structure
    CHOCH = "CHoCH"  # Change of Character

@dataclass
class SwingPoint:
    swing_type: SwingType
    price: float
    time: pd.Timestamp
    label: str
    strength: float = 0.0  # Added strength indicator
    
    def to_dict(self) -> Dict:
        return {
            "type": self.swing_type.value,
            "label": self.label,
            "price": self.price,
            "time": self.time,
            "strength": self.strength
        }

@dataclass
class TradingZones:
    premium_top: float
    premium_bottom: float
    equilibrium_top: float
    equilibrium_bottom: float
    discount_top: float
    discount_bottom: float
    
    def get_zone_for_price(self, price: float) -> str:
        """Determine which zone a price falls into"""
        if price >= self.premium_bottom:
            return "premium"
        elif price <= self.discount_top:
            return "discount"
        else:
            return "equilibrium"

class EnhancedSwingDetector:
    """Enhanced swing detection with multiple algorithms and filtering"""
    
    def __init__(self, left: int = 3, right: int = 3, min_strength: float = 0.0):
        self.left = left
        self.right = right
        self.min_strength = min_strength
    
    def detect_swings_vectorized(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Vectorized swing detection for better performance"""
        if len(df) < self.left + self.right + 1:
            return []
        
        # Normalize timezone for consistent handling
        df_work = df.copy()
        df_work['start_time_normalized'] = self._normalize_timestamps(df_work['start_time'])
        
        highs = df_work['high'].values
        lows = df_work['low'].values
        times = df_work['start_time_normalized'].values
        
        swing_points = []
        
        # Vectorized approach using rolling windows
        for i in range(self.left, len(df_work) - self.right):
            # Check for swing high
            window_high = highs[i - self.left:i + self.right + 1]
            center_high = highs[i]
            
            if center_high == np.max(window_high):
                # Calculate strength based on price differential
                strength = (center_high - np.mean(window_high)) / center_high
                
                if strength >= self.min_strength:
                    swing_points.append(SwingPoint(
                        swing_type=SwingType.HIGH,
                        price=center_high,
                        time=times[i],
                        label="HH",
                        strength=strength
                    ))
            
            # Check for swing low
            window_low = lows[i - self.left:i + self.right + 1]
            center_low = lows[i]
            
            if center_low == np.min(window_low):
                # Calculate strength based on price differential
                strength = (np.mean(window_low) - center_low) / center_low
                
                if strength >= self.min_strength:
                    swing_points.append(SwingPoint(
                        swing_type=SwingType.LOW,
                        price=center_low,
                        time=times[i],
                        label="LL",
                        strength=strength
                    ))
        
        return sorted(swing_points, key=lambda x: x.time)
    
    def _normalize_timestamps(self, timestamp_series):
        """Normalize timestamps to timezone-naive for consistent comparison"""
        if hasattr(timestamp_series.iloc[0], 'tz') and timestamp_series.iloc[0].tz is not None:
            return timestamp_series.dt.tz_convert('UTC').dt.tz_localize(None)
        return timestamp_series
    
    def classify_swing_structure(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """Classify swings as Higher Highs, Lower Lows, etc."""
        if len(swings) < 2:
            return swings
        
        classified_swings = []
        
        for i, swing in enumerate(swings):
            if i == 0:
                classified_swings.append(swing)
                continue
            
            # Find previous swing of same type
            prev_same_type = None
            for j in range(i - 1, -1, -1):
                if swings[j].swing_type == swing.swing_type:
                    prev_same_type = swings[j]
                    break
            
            if prev_same_type is None:
                classified_swings.append(swing)
                continue
            
            # Classify based on price relationship
            if swing.swing_type == SwingType.HIGH:
                if swing.price > prev_same_type.price:
                    swing.label = "HH"  # Higher High
                else:
                    swing.label = "LH"  # Lower High
            else:  # SwingType.LOW
                if swing.price < prev_same_type.price:
                    swing.label = "LL"  # Lower Low
                else:
                    swing.label = "HL"  # Higher Low
            
            classified_swings.append(swing)
        
        return classified_swings

class ZoneCalculator:
    """Enhanced zone calculation with multiple methods"""
    
    @staticmethod
    def calculate_fibonacci_zones(hh_price: float, ll_price: float) -> TradingZones:
        """Calculate zones using Fibonacci levels"""
        zone_height = hh_price - ll_price
        
        # Fibonacci retracement levels
        fib_618 = zone_height * 0.618
        fib_382 = zone_height * 0.382
        
        return TradingZones(
            premium_top=hh_price,
            premium_bottom=hh_price - fib_382,
            equilibrium_top=hh_price - fib_382,
            equilibrium_bottom=ll_price + fib_382,
            discount_top=ll_price + fib_382,
            discount_bottom=ll_price
        )
    
    @staticmethod
    def calculate_equal_zones(hh_price: float, ll_price: float) -> TradingZones:
        """Calculate equal third zones (your original method)"""
        zone_height = hh_price - ll_price
        third = zone_height / 3
        
        return TradingZones(
            premium_top=hh_price,
            premium_bottom=hh_price - third,
            equilibrium_top=hh_price - third,
            equilibrium_bottom=ll_price + third,
            discount_top=ll_price + third,
            discount_bottom=ll_price
        )

class StructureAnalyzer:
    """Enhanced structure analysis with trend detection"""
    
    def __init__(self):
        self.trend_bias = None
    
    def detect_structure_breaks(self, swings: List[SwingPoint], df: pd.DataFrame) -> List[Dict]:
        """Enhanced structure break detection"""
        if len(swings) < 2:
            return []
        
        # Normalize DataFrame timestamps for consistent comparison
        df_work = df.copy()
        df_work['start_time_normalized'] = self._normalize_timestamps(df_work['start_time'])
        
        signals = []
        
        for i in range(1, len(swings)):
            prev_swing = swings[i - 1]
            curr_swing = swings[i]
            
            # Only compare swings of same type
            if prev_swing.swing_type != curr_swing.swing_type:
                continue
            
            # Get segment between swings for confirmation
            segment_mask = (df_work['start_time_normalized'] > prev_swing.time) & (df_work['start_time_normalized'] <= curr_swing.time)
            segment = df_work[segment_mask].copy()
            
            if segment.empty:
                continue
            
            confirmation_row = self._find_confirmation(prev_swing, curr_swing, segment)
            
            if confirmation_row is None:
                continue
            
            signal = self._create_signal(prev_swing, curr_swing, confirmation_row)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _normalize_timestamps(self, timestamp_series):
        """Normalize timestamps to timezone-naive for consistent comparison"""
        if hasattr(timestamp_series.iloc[0], 'tz') and timestamp_series.iloc[0].tz is not None:
            return timestamp_series.dt.tz_convert('UTC').dt.tz_localize(None)
        return timestamp_series
    
    def _find_confirmation(self, prev_swing: SwingPoint, curr_swing: SwingPoint, segment: pd.DataFrame) -> Optional[pd.Series]:
        """Find confirmation candle that breaks structure"""
        if curr_swing.swing_type == SwingType.HIGH:
            # Look for close above previous high
            confirmation_mask = segment['close'] > prev_swing.price
        else:
            # Look for close below previous low
            confirmation_mask = segment['close'] < prev_swing.price
        
        confirmation_rows = segment[confirmation_mask]
        return confirmation_rows.iloc[0] if not confirmation_rows.empty else None
    
    def _create_signal(self, prev_swing: SwingPoint, curr_swing: SwingPoint, confirmation_row: pd.Series) -> Optional[Dict]:
        """Create structure signal"""
        direction = "bullish" if curr_swing.swing_type == SwingType.HIGH else "bearish"
        new_bias = 1 if direction == "bullish" else -1
        
        # Determine if this is BOS or CHoCH
        signal_type = SignalType.BOS if self.trend_bias != new_bias else SignalType.CHOCH
        
        self.trend_bias = new_bias
        
        # Use normalized timestamp column if available, otherwise use original
        confirmation_time = confirmation_row.get('start_time_normalized', confirmation_row['start_time'])
        
        return {
            "label": signal_type.value,
            "direction": direction,
            "price": prev_swing.price,
            "time": confirmation_time,
            "from_time": prev_swing.time,
            "to_time": curr_swing.time,
            "confirmation_time": confirmation_time,
            "strength": curr_swing.strength
        }

class TradingSystemAnalyzer:
    """Main class that orchestrates the analysis"""
    
    def __init__(self, swing_params: Dict = None, zone_method: str = "fibonacci"):
        self.swing_params = swing_params or {"left": 3, "right": 3, "min_strength": 0.001}
        self.zone_method = zone_method
        
        self.swing_detector = EnhancedSwingDetector(**self.swing_params)
        self.structure_analyzer = StructureAnalyzer()
    
    def analyze_market_structure(self, df: pd.DataFrame, bar_limit: int = 192) -> Dict:
        """Complete market structure analysis"""
        # Detect swings
        swings = self.swing_detector.detect_swings_vectorized(df)
        
        if not swings:
            return {"error": "No swings detected"}
        
        # Classify swing structure
        classified_swings = self.swing_detector.classify_swing_structure(swings)
        
    def analyze_market_structure(self, df: pd.DataFrame, bar_limit: int = 192) -> Dict:
        """Complete market structure analysis"""
        # Detect swings
        swings = self.swing_detector.detect_swings_vectorized(df)
        
        if not swings:
            return {"error": "No swings detected"}
        
        # Classify swing structure
        classified_swings = self.swing_detector.classify_swing_structure(swings)
        
        # Filter to recent bars if specified
        if bar_limit and len(df) > bar_limit:
            cutoff_time = df['start_time'].iloc[-bar_limit]
            
            # Normalize cutoff time
            if hasattr(cutoff_time, 'tz') and cutoff_time.tz is not None:
                cutoff_naive = cutoff_time.tz_convert('UTC').tz_localize(None)
            else:
                cutoff_naive = cutoff_time
            
            # Filter swings (swing times are already normalized from detection)
            classified_swings = [s for s in classified_swings if s.time >= cutoff_naive]
        
        if not classified_swings:
            return {"error": "No recent swings found"}
        
        # Find key levels
        highest_hh = self._select_extreme_swing(classified_swings, "HH", max_func=True)
        lowest_ll = self._select_extreme_swing(classified_swings, "LL", max_func=False)
        
        if not highest_hh or not lowest_ll:
            return {"error": "Could not find HH and LL for zone calculation"}
        
        # Calculate zones
        if self.zone_method == "fibonacci":
            zones = ZoneCalculator.calculate_fibonacci_zones(highest_hh.price, lowest_ll.price)
        else:
            zones = ZoneCalculator.calculate_equal_zones(highest_hh.price, lowest_ll.price)
        
        # Detect structure signals
        structure_signals = self.structure_analyzer.detect_structure_breaks(classified_swings, df)
        
        return {
            "swings": [s.to_dict() for s in classified_swings],
            "highest_hh": highest_hh.to_dict(),
            "lowest_ll": lowest_ll.to_dict(),
            "zones": zones.__dict__,
            "structure_signals": structure_signals,
            "current_zone": zones.get_zone_for_price(df['close'].iloc[-1])
        }
    
    def _select_extreme_swing(self, swings: List[SwingPoint], label: str, max_func: bool) -> Optional[SwingPoint]:
        """Select the most extreme swing of given type"""
        filtered_swings = [s for s in swings if s.label == label]
        
        if not filtered_swings:
            return None
        
        if max_func:
            return max(filtered_swings, key=lambda x: x.price)
        else:
            return min(filtered_swings, key=lambda x: x.price)




# === Entry Point ===
if __name__ == "__main__":
    ema_config = {
        "5m": [9, 21],
        "15m": [9, 21],
        "1h": [50, 200]
    }

    zone_config = {
        "5m": 288,   # 5m * 288 = 1440 minutes = 1 day
        "15m": 240,   # 15m * 96 = 1440 minutes = 1 day
        "1h": 24     # 1h * 24 = 1 day
    }

    tf_input = "15m"
    epic = "CS.D.USDJPY.MINI.IP"
    image_path = f"/app/tradesetups/trade_snapshot_{epic.replace('.', '_')}.png"
    df_result, tf = run_alert_pipeline(epic, engine, base_timeframe=tf_input, ema_config=ema_config)
    zone_lookback = zone_config.get(tf, 96)
    
    analyzer = TradingSystemAnalyzer(
        swing_params={"left": 3, "right": 3, "min_strength": 0.002},
        zone_method="fibonacci"
    )
    
    # Analyze market structure
    results = analyzer.analyze_market_structure(df_result, bar_limit=500)
    
    print("Analysis Results:")
    print(f"- Found {len(results.get('swings', []))} swings")
    print(f"- Current zone: {results.get('current_zone', 'Unknown')}")
    print(f"- Structure signals: {len(results.get('structure_signals', []))}")
    # Zones
    zones = results['zones']
    print(f"\nTrading Zones:")
    print(f"  🔴 PREMIUM:     {zones['premium_bottom']:.5f} - {zones['premium_top']:.5f}")
    print(f"  ⚪ EQUILIBRIUM: {zones['equilibrium_bottom']:.5f} - {zones['equilibrium_top']:.5f}")
    print(f"  🟢 DISCOUNT:    {zones['discount_bottom']:.5f} - {zones['discount_top']:.5f}")

    
    

    
        
        
    # preapare data for image
    timeframe_key = tf_input if tf_input in ema_config else f"{int(tf_input)}m"
    ema_columns = [f"ema_{n}_{timeframe_key}" for n in ema_config.get(timeframe_key, [])]
    #img_path = plot_alert_chart(chart_df, epic, 15, image_path,last_alert=last, zones=zones,ema_columns=ema_columns)
    


# hämta data med db function direkt i zone calc, alt använd tail antal DF som ska vara relevanta