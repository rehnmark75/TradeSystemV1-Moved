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
import logging
from logging.handlers import TimedRotatingFileHandler
import requests
import schedule


warnings.simplefilter("error", RuntimeWarning)
# --- Config ---

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

stockholm = pytz.timezone("Europe/Stockholm")
LOCAL_TZ = pytz.timezone("Europe/Stockholm")

ORDER_API_URL = "http://fastapi-dev:8000/orders/place-order"  # Update if hosted elsewhere
API_SUBSCRIPTION_KEY = "436abe054a074894a0517e5172f0e5b6"

EPIC_MAP = {
    "CS.D.EURUSD.MINI.IP": "EURUSD.1.MINI",
    "CS.D.GBPUSD.MINI.IP": "GBPUSD.1.MINI",
    "CS.D.USDJPY.MINI.IP": "USDJPY.100.MINI",
    "CS.D.AUDUSD.MINI.IP": "AUDUSD.1.MINI",
    "CS.D.USDCAD.MINI.IP": "USDCAD.1.MINI",
    "CS.D.EURJPY.MINI.IP": "EURJPY.100.MINI",
    "CS.D.AUDJPY.MINI.IP": "AUDJPY.100.MINI",
    "CS.D.NZDUSD.MINI.IP": "NZDUSD.1.MINI",
    "CS.D.USDCHF.MINI.IP": "USDCHF.1.MINI"
}

class LocalTimeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz or pytz.UTC

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        return dt.strftime(datefmt) if datefmt else dt.isoformat()

formatter = LocalTimeFormatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    tz=stockholm
)

# Create rotating log handler — rotates daily
# --- File handler (rotating daily)
log_handler = TimedRotatingFileHandler("/app/logs/trade_scan_smc1_alert.log", when="midnight", interval=1, backupCount=7)
log_handler.suffix = "%Y-%m-%d"
log_handler.setFormatter(formatter)

# --- Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# --- Setup logger manually
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)
log.addHandler(console_handler)
log.propagate = False

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

def safe_le(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    result = pd.Series(False, index=series_a.index, dtype=bool)
    mask = series_a.notnull() & series_b.notnull()
    result.loc[mask] = (series_a[mask] <= series_b[mask]).astype(bool)
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

def log_alert_to_db(engine, epic, start_time, direction, price, alert_type):
    # Normalize to naive, second-level precision
    start_time = pd.to_datetime(start_time).tz_localize(None).floor("S")

    query_check = text("""
        SELECT 1 FROM alerts
        WHERE epic = :epic AND start_time = :start_time AND alert_type = :alert_type
        LIMIT 1
    """)
    
    query_insert = text("""
        INSERT INTO alerts (epic, start_time, direction, price, alert_type, created_at)
        VALUES (:epic, :start_time, :direction, :price, :alert_type, NOW())
    """)

    with engine.begin() as conn:
        exists = conn.execute(query_check, {
            "epic": epic,
            "start_time": start_time,
            "alert_type": alert_type
        }).fetchone()
        
        if exists:
            print(f"⚠️ Alert already exists in DB for {epic} at {start_time} ({alert_type})")
            return  # Do not insert

        conn.execute(query_insert, {
            "epic": epic,
            "start_time": start_time,
            "direction": direction,
            "price": price,
            "alert_type": alert_type
        })

def alert_exists_in_db(engine, epic, start_time, alert_type):
    
    start_time = pd.to_datetime(start_time).replace(tzinfo=None).floor("S")
    query = text("""
        SELECT 1 FROM alerts
        WHERE epic = :epic AND start_time = :start_time AND alert_type = :alert_type
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {
            "epic": epic,
            "start_time": start_time,
            "alert_type": alert_type
        })
        return result.scalar() is not None

def send_order(external_epic, direction, stop_distance, limit_distance=None, size=None, custom_label=None, risk_reward=2.0):
    # Reverse lookup for internal epic name
    internal_epic = EPIC_MAP.get(external_epic)
    if not internal_epic:
        log.info(f"send_order [WARN] No reverse mapping found for: {external_epic}")
        return

    # Build order payload
    body = {
        "epic": internal_epic,
        "direction": direction,
        "stop_distance": stop_distance
    }

    # Optional fields
    if limit_distance is not None:
        body["limit_distance"] = limit_distance
    if size is not None:
        body["size"] = size
    if custom_label is not None:
        body["custom_label"] = custom_label
    if risk_reward is not None:
        body["risk_reward"] = risk_reward

    headers = {
        "x-apim-gateway": "verified",
        "Content-Type": "application/json"
    }

    params = {
        "subscription-key": API_SUBSCRIPTION_KEY
    }

    try:
        response = requests.post(ORDER_API_URL, json=body, headers=headers, params=params)
        if response.status_code == 200:
            log.info(f"send_order: [ORDER SENT] {internal_epic} -> {direction}")
        else:
            log.info(f"send_order: [ERROR] Failed to send order: {response.status_code} | {response.text}")
    except Exception as e:
        log.info(f"send_order: [EXCEPTION] Error sending order: {e}")


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
    eps: float = 1e-4 
) -> pd.DataFrame:
    df = df.sort_values('start_time').reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    prev_fast_col = f"prev_{fast_ema_col}"
    df[prev_fast_col] = df[fast_ema_col].shift(1)

    suffix = fast_ema_col.split('_')[-1]
    ema_200_col = f"ema_200_{suffix}"
    has_ema_200 = ema_200_col in df.columns

    df["bull_cross"] = (
        safe_lt(df["prev_close"], df[prev_fast_col] - eps) &
        safe_gt(df["close"], df[fast_ema_col] + eps)
    )

    df["bear_cross"] = (
        safe_gt(df["prev_close"], df[prev_fast_col] + eps) &
        safe_lt(df["close"], df[fast_ema_col] - eps)
    )

    slow_minus_eps = df[slow_ema_col] - eps
    df["bull_condition"] = (
        safe_gt(df["close"], slow_minus_eps) &
        safe_le(df[slow_ema_col] - eps, df[fast_ema_col]) &
        safe_gt(df["close"], df[ema_200_col] + eps) &
        safe_gt(df[fast_ema_col], df[ema_200_col] + eps) &
        safe_gt(df[slow_ema_col], df[ema_200_col] + eps)
    )

    slow_plus_eps = df[slow_ema_col] + eps
    df["bear_condition"] = (
        safe_lt(df["close"], df[slow_ema_col] - eps) &
        safe_le(df[fast_ema_col], slow_plus_eps) &
        safe_lt(df["close"], df[ema_200_col] - eps) &
        safe_lt(df[fast_ema_col], df[ema_200_col] - eps) &
        safe_lt(df[slow_ema_col], df[ema_200_col] - eps)
    )

    # Add 200 EMA filter if available
    if has_ema_200:
        df["bull_condition"] &= (
            safe_gt(df["close"], df[ema_200_col] + eps) &
            safe_gt(df[fast_ema_col], df[ema_200_col] + eps) &
            safe_gt(df[slow_ema_col], df[ema_200_col] + eps)
        )

        df["bear_condition"] &= (
            safe_lt(df["close"], df[ema_200_col] - eps) &
            safe_lt(df[fast_ema_col], df[ema_200_col] - eps) &
            safe_lt(df[slow_ema_col], df[ema_200_col] - eps)
        )

    df["bull_alert"] = df["bull_cross"] & df["bull_condition"]
    df["bear_alert"] = df["bear_cross"] & df["bear_condition"]

        # Debug helper for troubleshooting missed alerts
    debug_rows = df[df["bear_cross"] | df["bull_cross"]].copy()
    suffix = fast_ema_col.split('_')[-1]
    ema_200_col = f"ema_200_{suffix}"

    """ if not debug_rows.empty:
        print(f"\n--- DEBUG: Crosses with potential alerts ({suffix}) ---")
        for i, row in debug_rows.tail(5).iterrows():
            print(f"\nCandle: {row['start_time']}")
            print(f"  Close:        {row['close']:.5f}")
            print(f"  Fast EMA:     {row[fast_ema_col]:.5f}")
            print(f"  Slow EMA:     {row[slow_ema_col]:.5f}")
            print(f"  200 EMA:      {row.get(ema_200_col, float('nan')):.5f}")
            print(f"  Bear Cross:   {row['bear_cross']}")

            if ema_200_col in row:
                conds = {
                    "close < slow": row["close"] < row[slow_ema_col],
                    "fast < slow": row[fast_ema_col] < row[slow_ema_col],
                    "close < 200": row["close"] < row[ema_200_col],
                    "fast < 200": row[fast_ema_col] < row[ema_200_col],
                    "slow < 200": row[slow_ema_col] < row[ema_200_col]
                }
                for label, passed in conds.items():
                    print(f"    {label:<13}: {passed}")
                print(f"  Bear Condition: {row['bear_condition']} → Bear Alert: {row['bear_alert']}")
            else:
                print("  200 EMA column missing – skipping condition check.") """


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

def run_alert_pipeline(
    epic: str,
    engine,
    debug=False,
    ema_config: dict = None,
    base_timeframe: str = "15m"
) -> pd.DataFrame:
    if ema_config is None:
        ema_config = {
            "5m": [12, 50, 200],
            "15m": [9, 21, 200],
            "1h": [12, 50, 100, 200]
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
            if use_zone_filter and zones and alert_price > zones.get("equilibrium_top", float("inf")):
                if debug: print(f"[{alert_time}] 🚫 Bull: price not in discount zone")
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
            if use_zone_filter and zones and alert_price < zones.get("equilibrium_bottom", -float("inf")):
                if debug: print(f"[{alert_time}] 🚫 Bear: price not in premium zone")
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




    if base_timeframe != "5m":
        merge_cols_5m = [c for c in df_5m.columns if c.startswith("ema_") or "macd_histogram_5m" in c]
        df_5m_merge = df_5m[["start_time"] + merge_cols_5m].sort_values("start_time")
        base_df = pd.merge_asof(
            base_df, df_5m_merge, on="start_time", direction="backward", tolerance=pd.Timedelta("15min")
        )

    if base_timeframe != "1h":
        merge_cols_1h = [c for c in df_1h.columns if c.startswith("ema_") or "macd_histogram_1h" in c or c == "trend"]
        df_1h_merge = df_1h[["start_time"] + merge_cols_1h].sort_values("start_time")
        base_df = pd.merge_asof(
            base_df, df_1h_merge, on="start_time", direction="backward", tolerance=pd.Timedelta("1h")
        )

    base_df["trend"] = base_df["trend"].where(pd.notnull(base_df["trend"]), False).astype(bool)

    if base_timeframe == "15m":
        base_df["atr"] = ta.atr(base_df["high"], base_df["low"], base_df["close"], length=14)
        base_df = detect_ranges(base_df, length=20, atr_len=500, mult=1.0)

    return base_df, base_timeframe

def atr_to_ig_points(atr: float, epic: str) -> int:
    if "JPY" in epic:
        point_value = 0.01
    elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        point_value = 0.0001
    else:
        point_value = 1.0
    return max(5, int(atr / point_value))  # Minimum stop of 5 IG points

# SMC functions
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

def select_highest_HH(swings_df: pd.DataFrame, debug: bool = False):
    hh_swings = swings_df[swings_df["label"] == "HH"]

    if not hh_swings.empty:
        highest = hh_swings.loc[hh_swings["price"].idxmax()]
        if debug:
            print(f"✅ Selected HH: {highest['price']} @ {highest['time']}")
        return highest.to_dict()

    if debug:
        print("⚠️ No HH found in the recent swings.")
    return None

def select_lowest_LL(swings_df: pd.DataFrame, debug: bool = False):
    ll_swings = swings_df[swings_df["label"] == "LL"]

    if not ll_swings.empty:
        lowest = ll_swings.loc[ll_swings["price"].idxmin()]
        if debug:
            print(f"✅ Selected LL: {lowest['price']} @ {lowest['time']}")
        return lowest.to_dict()

    if debug:
        print("⚠️ No LL found in the recent swings.")
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

def detect_structure_signals_luxalgo(swings: List[Dict], df: pd.DataFrame) -> List[Dict]:
    signals = []
    trend_bias = None

    for i in range(1, len(swings)):
        prev = swings[i - 1]
        curr = swings[i]

        if prev["type"] != curr["type"]:
            continue

        segment = df[(df["start_time"] > prev["time"]) & (df["start_time"] <= curr["time"])]
        confirmation_row = None

        if curr["type"] == "high":
            confirmation_row = segment[segment["close"] > prev["price"]].head(1)
        elif curr["type"] == "low":
            confirmation_row = segment[segment["close"] < prev["price"]].head(1)

        if confirmation_row is None or confirmation_row.empty:
            continue

        confirm_time = confirmation_row.iloc[0]["start_time"]
        label = "BOS" if trend_bias != (-1 if curr["type"] == "high" else 1) else "CHoCH"
        direction = "bullish" if curr["type"] == "high" else "bearish"
        trend_bias = 1 if direction == "bullish" else -1

        signals.append({
            "label": label,
            "direction": direction,
            "price": prev["price"],
            "time": confirm_time,
            "from_time": prev["time"],
            "to_time": curr["time"],
            "confirmation_time": confirm_time
        })

    return signals

def filter_swings_to_recent_bars(swings_df: pd.DataFrame, df_result: pd.DataFrame, bar_limit: int) -> pd.DataFrame:
    # Determine time threshold from last N bars in df_result
    cutoff_time = df_result["start_time"].iloc[-bar_limit]
    filtered = swings_df[swings_df["time"] >= cutoff_time]
    return filtered

    
def main():    
    epics = [
    "CS.D.EURUSD.MINI.IP",
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP",
    "CS.D.USDCAD.MINI.IP",
    "CS.D.EURJPY.MINI.IP",
    "CS.D.AUDJPY.MINI.IP",
    "CS.D.NZDUSD.MINI.IP"
    ]

    for epic in epics:
            
        ema_config = {
            "5m": [9, 21, 200],
            "15m": [9, 21, 200],
            "1h": [50, 200]
        }

        zone_config = {
            "5m": 288,   # 5m * 288 = 1440 minutes = 1 day
            "15m": 192,   # 15m * 96 = 1440 minutes = 1 day
            "1h": 24     # 1h * 24 = 1 day
        }

        tf_input = "15m"
        #epic = "CS.D.AUDUSD.MINI.IP"
        image_path = f"/app/tradesetups/trade_snapshot_{epic.replace('.', '_')}.png"
        df_result, tf = run_alert_pipeline(epic, engine, base_timeframe=tf_input, ema_config=ema_config)
        zone_lookback = zone_config.get(tf, 96)

        #Step 1: Detect swing pivots using the LuxAlgo-style function
        swings = detect_confirmed_swing_highs_lows(df_result, left=3, right=3)

        # Step 2: Confirm structure signals (BOS/CHoCH)
        signals = detect_structure_signals_luxalgo(swings, df_result)
        
        swings_df = pd.DataFrame(swings)
        
        swings_df["time"] = pd.to_datetime(swings_df["time"]).dt.tz_convert("Europe/Stockholm")
        swings_df = filter_swings_to_recent_bars(swings_df, df_result, bar_limit=zone_lookback)


        top = select_highest_HH(swings_df, debug=True)
        bottom = select_lowest_LL(swings_df, debug=True)

        if top and bottom:
            zones = calculate_zones(top["price"], bottom["price"])
            print(f"📍 HH used: {top['price']} @ {top['time']}")
            print(f"📍 LL used: {bottom['price']} @ {bottom['time']}")
            print(zones)
        else:
            zones = None
            print("❌ No valid HH/LL found.")

        alerts = detect_valid_alerts(df_result , debug=False,use_macd_5m_filter=False, use_macd_15m_filter=False, use_macd_1h_filter=False,
                                    use_trend_filter=False, use_slope_filter=False, 
                                    slope_method="linreg",use_ema_distance_filter=False, max_ema_distance_pct=0.0015,
                                    use_follow_through_filter=False,use_zone_filter=False,zones=None)
    
        if not alerts:
            log.info("🚫 No confirmed alerts found.")
        else:
            # Expect only one alert
            alert = alerts[-1]  # or alerts[0] if sorted from newest to oldest

            alert_time = pd.to_datetime(alert["timestamp"])
            if alert_time.tzinfo is None:
                alert_time = alert_time.replace(tzinfo=pytz.utc)
            alert_time = alert_time.astimezone(LOCAL_TZ).floor("min")
            now = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(LOCAL_TZ)
            log.info(f"Alert to check {epic}, now: {now} and alerttime is: {alert_time}")
            if now - alert_time > timedelta(minutes=30):
                log.info(f"⏩ Alert for {epic} is older than 30 minutes ({alert_time}), skipping.")
            elif alert_exists_in_db(engine, epic, alert_time, alert["type"]):
                log.info(f"⏩ Alert for {epic} at {alert_time} ({alert['type']}) already exists in DB, skipping.")
            else:
                direction = "BUY" if alert["type"] == "bull" else "SELL"
                log.info(f"\n📌 New alert: {direction} at {alert['price']} ({alert_time})")

                try:
                    log_alert_to_db(
                        engine,
                        epic,
                        alert_time,
                        direction,
                        float(alert["price"]),
                        alert_type=alert["type"]
                    )
                except Exception as e:
                    log.warning(f"⚠️ Failed to log alert to DB: {e}")
                else:
                    atr_1h = df_result["atr_1h"].iloc[-1]
                    ig_stop_points = atr_to_ig_points(atr_1h, epic)
                    log.info(f"📏 Using ATR-based stop: {ig_stop_points} IG points")

                    log.info(f"📤 Sending order: {epic} {direction} @ {alert['price']} with stop {ig_stop_points}")
                    # send_order(epic, direction, ig_stop_points)

                    log.info(f"HH used: {top['price']} @ {top['time']}")
                    log.info(f"LL used: {bottom['price']} @ {bottom['time']}")

def scan_and_trade():
    log.info(f"[{datetime.utcnow()}] Running scheduled scan...")
    main()


if __name__ == "__main__":

        print("⏳ Scheduler started... Running every 60 seconds.")
        scan_and_trade()  # Optional initial run

        schedule.every(2).minutes.do(scan_and_trade)
        while True:
            schedule.run_pending()
            time.sleep(1)