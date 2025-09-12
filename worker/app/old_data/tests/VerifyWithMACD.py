import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from PIL import Image
import google.generativeai as genai
from services.keyvault import get_secret
import json
import re
import logging
from logging.handlers import TimedRotatingFileHandler
import requests
import schedule
import time
import pytz
from datetime import datetime
from services.smc_structure import (
    detect_pivots, classify_pivots, get_recent_trailing_extremes,
    calculate_zones, convert_swings_to_plot_shapes, determine_bias, mark_confirmed_swings, get_most_recent_confirmed_hh_ll, get_structure_signals
)
import pandas_ta as ta

# --- Setup Gemini ---
api_key = get_secret("gemini")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# --- Config ---
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

stockholm = pytz.timezone("Europe/Stockholm")

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
log_handler = TimedRotatingFileHandler("/app/logs/ema_alert.log", when="midnight", interval=1, backupCount=7)
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


# Gemini log setup with daily rotation
gemini_log = logging.getLogger("gemini")
gemini_log.setLevel(logging.INFO)

gemini_handler = TimedRotatingFileHandler(
    "/app/logs/gemini_analysis.log", when="midnight", interval=1, backupCount=7
)
gemini_handler.suffix = "%Y-%m-%d"
gemini_handler.setFormatter(formatter)

gemini_log.addHandler(gemini_handler)
gemini_log.propagate = False

EPIC_MAP = {
    "CS.D.EURUSD.MINI.IP": "EURUSD.1.MINI",
    "CS.D.GBPUSD.MINI.IP": "GBPUSD.1.MINI",
    "CS.D.USDJPY.MINI.IP": "USDJPY.100.MINI",
    "CS.D.AUDUSD.MINI.IP": "AUDUSD.1.MINI"
}

log.info(f"🚀 BOOTING... {datetime.utcnow().isoformat()} UTC")
ALERT_LOG_PATH = "/app/tradesetups/alert_log.json"

ORDER_API_URL = "http://fastapi-dev:8000/orders/place-order"  # Update if hosted elsewhere
API_SUBSCRIPTION_KEY = "436abe054a074894a0517e5172f0e5b6"

def load_alert_log():
    if not os.path.exists(ALERT_LOG_PATH):
        return {}
    with open(ALERT_LOG_PATH, "r") as f:
        return json.load(f)

def save_alert_log(log):
    with open(ALERT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)

def extract_json_from_text(text):
    match = re.search(r'{.*?}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

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

# --- Detect Alerts ---

def detect_ema_alerts(df: pd.DataFrame, epic: str) -> pd.DataFrame:
    

    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_50"] = ta.ema(df["close"], length=50)

    macd = ta.macd(df["close"])
    df["macd_line"] = macd["MACD_12_26_9"]
    df["signal_line"] = macd["MACDs_12_26_9"]
    df["macd_histogram"] = macd["MACDh_12_26_9"]

    df["rsi"] = ta.rsi(df["close"], length=14)

    df["prev_close"] = df["close"].shift(1)
    df["prev_ema_12"] = df["ema_12"].shift(1)

    df["bull_cross"] = (df["prev_close"] < df["prev_ema_12"]) & (df["close"] > df["ema_12"]) & (df["ema_12"] > df["ema_50"])
    df["bear_cross"] = (df["prev_close"] > df["prev_ema_12"]) & (df["close"] < df["ema_12"]) & (df["ema_50"] > df["ema_12"])

    macd_thresh = 0.003 if "USDJPY" in epic else 0.00003

    df["bull_alert"] = df["bull_cross"] & (
        (df["macd_histogram"] > macd_thresh) #| (df["rsi"] > 50)
    )
    df["bear_alert"] = df["bear_cross"] & (
        (df["macd_histogram"] < -macd_thresh) #| (df["rsi"] < 50)
    )

    return df



def detect_trend_cross_alerts(df, eps=1e-8, min_slope=0.00001):
    # Calculate shifted values if not already present
    df["prev_ema_12"] = df["ema_12"].shift(1)
    df["prev_ema_50"] = df["ema_50"].shift(1)

    if "macd_slope" not in df.columns:
        df["macd_slope"] = df["macd_line"].diff()

    # === Recent crossover detection (within 3 bars) ===
    cross_up_now     = (df["ema_12"] > df["ema_50"] + eps) & (df["ema_12"].shift(1) < df["ema_50"].shift(1) - eps)
    cross_up_recent  = cross_up_now | cross_up_now.shift(1) | cross_up_now.shift(2)

    cross_down_now   = (df["ema_12"] < df["ema_50"] - eps) & (df["ema_12"].shift(1) > df["ema_50"].shift(1) + eps)
    cross_down_recent = cross_down_now | cross_down_now.shift(1) | cross_down_now.shift(2)

    # === Bull trend cross ===
    df["bull_trend_cross"] = (
        cross_up_recent &
        (df["close"] > df["ema_12"] + eps) &
        (df["macd_line"] > df["signal_line"] + eps) &
        (df["macd_slope"] > min_slope)
    )

    # === Bear trend cross ===
    df["bear_trend_cross"] = (
        cross_down_recent &
        (df["close"] < df["ema_12"] - eps) &
        (df["macd_line"] < df["signal_line"] - eps) &
        (df["macd_slope"] < -min_slope)
    )

    return df


# --- Plot and Save ---
def plot_alert_chart(df, epic, timeframe, image_path, zones=None, position_note=None):
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

    # Bullish Alerts
    bull_df = df[df['bull_alert']]
    fig.add_trace(go.Scatter(
        x=bull_df['start_time'],
        y=bull_df['high'] + offset,
        mode="markers",
        marker=dict(color="green", size=10, symbol="triangle-up"),
        name="Bull"
    ))

    # Bearish Alerts
    bear_df = df[df['bear_alert']]
    fig.add_trace(go.Scatter(
        x=bear_df['start_time'],
        y=bear_df['low'] - offset,
        mode="markers",
        marker=dict(color="red", size=10, symbol="triangle-down"),
        name="Bear"
    ))

    # Structure Markers (HH, LL, HL, LH)
    if "structure" in df.columns:
        label_colors = {"HH": "purple", "LL": "brown", "HL": "blue", "LH": "orange"}
        for label in ["HH", "LL", "HL", "LH"]:
            label_df = df[df["structure"] == label]
            fig.add_trace(go.Scatter(
                x=label_df["start_time"],
                y=label_df["high"] + offset if label in ["HH", "LH"] else label_df["low"] - offset,
                mode="markers+text",
                marker=dict(color=label_colors[label], size=8, symbol="circle"),
                text=[label] * len(label_df),
                textposition="top center" if label in ["HH", "LH"] else "bottom center",
                name=label
            ))

    # Optional position annotation
    if position_note and zones:
        fig.add_annotation(
            x=df["start_time"].max(),
            y=(zones["equilibrium_top"] + zones["equilibrium_bottom"]) / 2,
            text=position_note,
            showarrow=False,
            font=dict(size=14, color="gray"),
            xanchor="right",
            yanchor="middle"
        )

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


def get_last_alert_type(df):
    # Filter rows with alerts
    alerts = df[df["bull_alert"] | df["bear_alert"]]
    if alerts.empty:
        return None

    # Get last alert row
    last = alerts.tail(1).iloc[0]
    alert_type = "bull" if last["bull_alert"] else "bear"

    return {
        "timestamp": last["start_time"],
        "type": alert_type,
        "price": last["close"]
    }


def extract_json_from_text(text):
    match = re.search(r'{.*?}', text, re.DOTALL)
    return match.group(0) if match else None

def analyze_chart_with_gemini(image_path, epic, alert_type):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"❌ Failed to open image: {e}")
        return False

    print("✅ Inside Gemini function")

    # Prompt logic
    if alert_type == "bull":
        prompt = """Please answer the following question with true or false only.
Respond as a JSON object like:
{"bullish_within_or_below_zone": true}

Only take the last triangle from the right into consideration. Ignore any other triangle. If there are no triangles in the picture then look for the last candle that closes above the blue line.

Question:
Is the last bullish candle (green triangle) above the blue line, is the candle green, and is the orange line below the blue line?
"""
    elif alert_type == "bear":
        prompt = """Please answer the following question with true or false only.
Respond as a JSON object like:
{"bearish_within_or_above_zone": true}

Only take the last triangle from the right into consideration. Ignore any other triangle. If there are no triangles in the picture then look for the last candle that closes below the blue line.

Question:
Is the last bearish candle (red triangle) below the blue line, is the candle red, and is the orange line above the blue line?
"""
    else:
        raise ValueError("alert_type must be 'bull' or 'bear'")

    try:
        response = gemini_model.generate_content([prompt, image])
    except Exception as e:
        print(f"❌ Failed to generate content: {e}")
        return False

    print("\n📈 Gemini Analysis (Raw):\n")
    if response and hasattr(response, "text"):
        print("📈 Gemini Analysis (Raw):\n%s", response.text)
        try:
            json_text = extract_json_from_text(response.text.strip())
            result = json.loads(json_text)
            if alert_type == "bull":
                valid = result.get("bullish_within_or_below_zone", False)
            else:
                valid = result.get("bearish_within_or_above_zone", False)
        except Exception as e:
            print("❌ Failed to parse Gemini JSON:", e)
            valid = False
    else:
        print("❌ Empty or invalid response from Gemini")
        valid = False

    print(f"\n✅ Validated {alert_type.title()} Alert:", valid)
    return valid

def get_epics(engine):
    query = text("SELECT DISTINCT epic FROM ig_candles ORDER BY epic")
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]

def send_order(external_epic, direction):
    # Reverse lookup: find the internal epic name for this external epic
    internal_epic = EPIC_MAP.get(external_epic)
    if not internal_epic:
        print(f"[WARN] No reverse mapping found for: {external_epic}")
        return

    body = {
        "epic": internal_epic,
        "direction": direction
    }

    headers = {
        "x-apim-gateway": "verified",
        "Content-Type": "application/json"  # ✅ updated
    }

    params = {
        "subscription-key": API_SUBSCRIPTION_KEY
    }

    try:
        response = requests.post(ORDER_API_URL, json=body, headers=headers, params=params)  # ✅ changed `data` to `json`
        if response.status_code == 200:
            print(f"[ORDER SENT] {internal_epic} -> {direction}")
        else:
            print(f"[ERROR] Failed to send order: {response.status_code} | {response.text}")
    except Exception as e:
        print(f"[EXCEPTION] Error sending order: {e}")

def log_alert_to_db(engine, epic, start_time, direction, price, alert_type):
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


def has_alert_been_sent(engine, epic, start_time, alert_type):
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
        return result.fetchone() is not None

def alert_exists_in_db(engine, epic, start_time, alert_type):
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

def get_stop_loss_from_swings(alert_time, alert_type, pivots):
    """
    Find the most recent LL (for bull) or LH (for bear) before the alert time
    """
    # Go in reverse order to find the latest pivot before alert_time
    for pivot in reversed(pivots):
        if pivot["time"] >= alert_time:
            continue
        if alert_type == "bull" and pivot["label"] == "LL":
            return pivot["price"]
        elif alert_type == "bear" and pivot["label"] == "LH":
            return pivot["price"]
    return None  # fallback if no swing found

def price_to_ig_points(price_difference: float, epic: str) -> int:
    """
    Convert a price difference (e.g. 0.0007) into IG 'points', depending on the instrument.

    Args:
        price_difference (float): Difference in price (e.g. 0.0007)
        epic (str): IG epic code (used to infer instrument type)

    Returns:
        int: IG points (used in stopDistance)
    """
    if "JPY" in epic:
        point_value = 0.01
    elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        point_value = 0.0001
    else:
        point_value = 1.0  # fallback

    points = price_difference / point_value
    return int(round(points))

def is_ranging(df_slice: pd.DataFrame, local_top: float, local_bottom: float, atr_ratio: float = 0.25) -> bool:
    """
    Determines if the market is ranging based on range size vs ATR.
    :param df_slice: The recent window of data.
    :param local_top: The local high over the window.
    :param local_bottom: The local low over the window.
    :param atr_ratio: Threshold ratio for range vs ATR to flag a range.
    :return: True if ranging, False otherwise.
    """
    range_size = abs(local_top - local_bottom)
    recent_atr = df_slice["atr"].iloc[-1]

    print(f"📏 Range size: {range_size:.5f}, ATR: {recent_atr:.5f}")

    # If the range is less than X times ATR → considered a range
    if range_size < (recent_atr * (1 / atr_ratio)):
        return True
    return False

# indicator settings
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


timeframe = 5
DISPLAY_CANDLES = 192
lookback_hours = 500
LOCAL_TZ = pytz.timezone("Europe/Stockholm")



def main():

   
    # load previous alerts
    alert_log = load_alert_log()
    
    # Get all epics
    #epics = get_epics(engine)
    epics = ["CS.D.EURUSD.MINI.IP"]  # Add more epics as needed

    for epic in epics:
        print(f"🔍 Scanning {epic}...")

        try:
            image_path = f"/app/tradesetups/trade_snapshot_{epic.replace('.', '_')}.png"
            df = fetch_candle_data(engine, epic, 5, 500)
            df = detect_ema_alerts(df,epic)
            #df = detect_trend_cross_alerts(df)
            df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
            
    
            chart_df = df.tail(200).copy()
            last_alert = get_last_alert_type(df)

            if not last_alert:
                log.info(f"⚠️ No alerts found for {epic}")
                continue
            # Step 1: Detect all pivots
            raw_pivots = detect_pivots(df, lookback=5) + detect_pivots(df, lookback=50)
            classified = classify_pivots(raw_pivots)

            direction = "BUY" if last_alert["type"] == "bull" else "SELL"
            # --- Evaluate alerts
            confirmed_alerts = []
            

            for idx, row in df.iterrows():
                condition = None
                alert_type = None
                alert_class = None

                # === Classic signal alerts ===
                if row.get('bull_alert', False):
                    alert_type = "bull"
                    alert_class = "Signal"
                    condition = "bull_alert"
                elif row.get('bear_alert', False):
                    alert_type = "bear"
                    alert_class = "Signal"
                    condition = "bear_alert"

                # === Trend crossover alerts ===
                elif row.get('bull_trend_cross', False):
                    alert_type = "bull"
                    alert_class = "Trend"
                    condition = "bull_trend_cross"
                elif row.get('bear_trend_cross', False):
                    alert_type = "bear"
                    alert_class = "Trend"
                    condition = "bear_trend_cross"
                else:
                    continue  # no alert

                # === Range filter at alert time ===
                

                # Create a copy of the slice to avoid SettingWithCopyWarning
                df_slice = df.iloc[max(0, idx - 20):idx + 1].copy()  # trailing 20 bars

                # Compute local range
                local_top = df_slice["high"].max()
                local_bottom = df_slice["low"].min()

               # Compute ATR separately (as a new Series)
                atr_series = ta.atr(df_slice["high"], df_slice["low"], df_slice["close"], length=14)

                # Merge it back explicitly with assign()
                df_slice = df_slice.assign(atr=atr_series.values)

                # Check for ranging condition
                if is_ranging(df_slice, local_top, local_bottom):
                    print(f"🔇 Ranging at {row['start_time']} — alert suppressed.")
                    continue

                alert_price = float(row['close'])
                alert_time = row['start_time']

                print(f"[{alert_time}] {'🟢' if alert_type == 'bull' else '🔴'} {alert_type.title()} {alert_class} Alert at {alert_price} ({condition})")



                # Optionally compute stop loss
                stop_loss_price = get_stop_loss_from_swings(alert_time, alert_type,  classified)
                ig_stop_points = price_to_ig_points(abs(alert_price - stop_loss_price), epic) if stop_loss_price else None

                confirmed_alerts.append({
                    "timestamp": alert_time,
                    "price": alert_price,
                    "type": alert_type,
                    "stop_loss": ig_stop_points
                })
                emoji = "🟢" if alert_type == "bull" else "🔴"
               

            # Only plot or validate if at least one alert passed the filter
            is_valid = False
            if confirmed_alerts:
                # Plot chart (optional)
                chart_df["structure"] = None  # Initialize the column

                # Suppose `confirmed_swings` is a list of dicts with keys: label (HH/LL/etc), time, price
                #img_path = plot_alert_chart(chart_df, epic, timeframe, image_path)


                # Gemini AI validation (optional)
                # is_valid = analyze_chart_with_gemini(img_path, epic, last_alert["type"])

                if isinstance(is_valid, bool):
                    print(f"✅ Gemini validation result: {is_valid}")
                else:
                    print("❌ Gemini validation result is missing or invalid")

                if is_valid:
                    # send_order(epic, direction)
                    log.info(f"📤 Order sent & logged: {epic} {direction}")
                else:
                    log.info(f"🚫 No order sent — invalid alert")
            else:
                log.info(f"⏭️ No confirmed alerts for {epic}")

        except Exception as e:
            log.exception(f"❌ Error processing {epic}: {e}")


def scan_and_trade():
    log.info(f"[{datetime.utcnow()}] Running scheduled scan...")
    main()


if __name__ == "__main__":

        print("⏳ Scheduler started... Running every 60 seconds.")
        scan_and_trade()  # Optional initial run

        """ schedule.every(1).minutes.do(scan_and_trade)
        while True:
            schedule.run_pending()
            time.sleep(1) """
