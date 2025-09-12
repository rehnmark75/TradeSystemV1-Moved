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
import requests
import schedule
import time
from datetime import datetime
from services.smc_structure import (
    detect_pivots, classify_pivots, get_recent_trailing_extremes,
    calculate_zones, convert_swings_to_plot_shapes, determine_bias,
    detect_structure_signals_luxalgo, TrailingExtremes, update_trailing_extremes
)

# --- Setup Gemini ---
api_key = get_secret("gemini")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# --- Config ---
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

# Create dated log filename
log_filename = datetime.utcnow().strftime("/app/logs/ema_alert_%Y-%m-%d.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also logs to console
    ]
)

log = logging.getLogger(__name__)

gemini_log = logging.getLogger("gemini")
gemini_log.setLevel(logging.INFO)

gemini_handler = logging.FileHandler("/app/logs/gemini_analysis_%Y-%m-%d.log")
gemini_handler.setLevel(logging.INFO)

gemini_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
gemini_handler.setFormatter(gemini_formatter)

gemini_log.addHandler(gemini_handler)
gemini_log.propagate = False  # Don't also print to console

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
def detect_ema_alerts(df):
    eps = 1e-8
    df = df.sort_values('start_time').reset_index(drop=True)

    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['prev_close'] = df['close'].shift(1)
    df['prev_ema_12'] = df['ema_12'].shift(1)

    df['bull_cross'] = (df['prev_close'] < df['prev_ema_12'] - eps) & (df['close'] > df['ema_12'] + eps)
    df['bull_condition'] = (df['close'] > df['ema_50'] + eps) & (df['ema_12'] > df['ema_50'] + eps)
    df['bull_alert'] = df['bull_cross'] & df['bull_condition']

    df['bear_cross'] = (df['prev_close'] > df['prev_ema_12'] + eps) & (df['close'] < df['ema_12'] - eps)
    df['bear_condition'] = df['ema_50'] > df['ema_12'] + eps
    df['bear_alert'] = df['bear_cross'] & df['bear_condition']

    return df

# --- Plot and Save ---
def plot_alert_chart(df, epic, timeframe, image_path,zones=None, position_note=None):

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

    if position_note:
        fig.add_annotation(
            x=df["start_time"].max(),
            y=(zones["equilibrium_top"] + zones["equilibrium_bottom"]) / 2,
            text=position_note,
            showarrow=False,
            font=dict(size=14, color="gray"),
            xanchor="right",
            yanchor="middle"
        )

    # Bearish Alerts
    bear_df = df[df['bear_alert']]
    fig.add_trace(go.Scatter(
        x=bear_df['start_time'],
        y=bear_df['low'] - offset,
        mode="markers",
        marker=dict(color="red", size=10, symbol="triangle-down"),
        name="Bear"
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
    image = Image.open(image_path)

    # Prompt logic
    if alert_type == "bull":
        prompt = f"""Please answer the following question with true or false only.
Respond as a JSON object like:
{{"bullish_within_or_below_zone": true}}

Only take the last triangle from the right into consideration. Ignore any other triangle. If there are no triangles in the picture then look for the last candle that closes above the blue line.

Question:
Is the last bullish candle that has a (green triangle) above the blue line and is the candle green and is the orange line below the blue line?
"""
    elif alert_type == "bear":
        prompt = f"""Please answer the following question with true or false only.
Respond as a JSON object like:
{{"bearish_within_or_above_zone": true}}

Only take the last triangle from the right into consideration. Ignore any other triangle. If there are no triangles in the picture then look for the last candle that closes below the blue line.

Question:
Is the last bearish  candle(red triangle) blow the blue line and is the candle red. The orange line should be above the blue line?
"""
    else:
        raise ValueError("alert_type must be 'bull' or 'bear'")

    # Call Gemini
    response = gemini_model.generate_content([prompt, image])

    print("\n📈 Gemini Analysis (Raw):\n")
    gemini_log.info("📈 Gemini Analysis (Raw):\n%s", response.text)

    json_text = extract_json_from_text(response.text)
    try:
        result = json.loads(json_text)
        if alert_type == "bull":
            valid = result.get("bullish_within_or_below_zone", False)
        else:
            valid = result.get("bearish_within_or_above_zone", False)
    except Exception as e:
        print("❌ Failed to parse Gemini JSON:", e)
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

    body = f"{internal_epic},{direction}"  # raw string format
    headers = {
        "x-apim-gateway": "verified",
        "Content-Type": "text/plain"
    }
    params = {
        "subscription-key": API_SUBSCRIPTION_KEY
    }

    try:
        response = requests.post(ORDER_API_URL, data=body, headers=headers, params=params)
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

timeframe = 15
lookback_hours = 200
DISPLAY_CANDLES = 200



def main():

   
    # load previous alerts
    alert_log = load_alert_log()
    
    # Get all epics
    epics = ["CS.D.GBPUSD.MINI.IP"]
    #epics = get_epics(engine)
    for epic in epics:
        print(f"🔍 Scanning {epic}...")

        try:
            image_path = "/app/tradesetups/verify-epic.png"
            df = fetch_candle_data(engine, epic, timeframe, lookback_hours)
            df = detect_ema_alerts(df)
            chart_df = df.tail(20)
            last_alert = get_last_alert_type(df)

            # --- Trim for display ---
            zone_df = df.tail(DISPLAY_CANDLES + 100).reset_index(drop=True)  # More bars for trailing zone calc
    
            trailing_extremes = TrailingExtremes()
            for _, row in zone_df.iterrows():
                update_trailing_extremes(trailing_extremes, row)

            zones = None
            if trailing_extremes.top and trailing_extremes.bottom and trailing_extremes.top != trailing_extremes.bottom:
                zones = calculate_zones(trailing_extremes.top, trailing_extremes.bottom)
            
            position_note = "Unknown"
            equilibrium_confirmed_bull = False
            equilibrium_confirmed_bear = False

            if zones and last_alert:
                alert_price = float(last_alert["price"])
                eq_top = zones["equilibrium_top"]
                eq_bot = zones["equilibrium_bottom"]
                alert_type = last_alert["type"] 

                if alert_price > eq_top:
                    position_note = "Above Equilibrium"
                elif alert_price < eq_bot:
                    position_note = "Below Equilibrium"
                else:
                    position_note = "Within Equilibrium"
                
                if alert_type == "bull":
                    equilibrium_confirmed_bull = alert_price < eq_bot or (eq_bot <= alert_price <= eq_top)
                elif alert_type == "bear":
                    equilibrium_confirmed_bear = alert_price > eq_top or (eq_bot <= alert_price <= eq_top)

            # if not last_alert:
            #     log.info(f"⚠️ No alerts found for {epic}")
            #     continue

            # # Check age
            # alert_time = pd.to_datetime(last_alert["timestamp"])
            # if datetime.utcnow() - alert_time > timedelta(minutes=30):
            #     log.info(f"⏩ Alert for {epic} is older than 30 minutes ({alert_time}), skipping.")
            #     continue

            # Check if already in DB
            """ if alert_exists_in_db(engine, epic, last_alert["timestamp"], last_alert["type"]):
                log.info(f"⏩ Alert for {epic} at {alert_time} ({last_alert['type']}) already exists in DB, skipping.")
                continue """
            
            direction = "BUY" if last_alert["type"] == "bull" else "SELL"

            # log alert to db so we dont run it again
            """ log_alert_to_db(  
                    engine,
                    epic,
                    last_alert["timestamp"],
                    direction,
                    float(last_alert["price"]),
                    alert_type=last_alert["type"]
                ) """
            # Proceed only if new
            
            #log.info(f"🕒 Last alert: {alert_time} for {epic}")
            #log.info(f"📢 Type: {last_alert['type']}")
            #log.info(f"💰 Price: {last_alert['price']:.5f}")
            
            # ✅ Tag alert as zone-confirmed based on its type
            last_alert["zone_confirmed"] = (
                equilibrium_confirmed_bull if last_alert["type"] == "bull" else equilibrium_confirmed_bear
            )

            # ✅ Only run Gemini if the alert passes the equilibrium check
            if last_alert["zone_confirmed"]:
                img_path = plot_alert_chart(chart_df, epic, timeframe, image_path, zones=zones, position_note=position_note)
                is_valid = analyze_chart_with_gemini(img_path, epic, last_alert["type"])
                print(f"✅ Gemini validation result: {is_valid}")
            else:
                print(f"⏭️ Skipping Gemini — {last_alert['type']} alert not in valid equilibrium zone.")

            #if is_valid:
            #    send_order(epic, direction)

                
            #    log.info(f"📤 Order sent & logged: {epic} {direction}")

        except Exception as e:
            log.exception(f"❌ Error processing {epic}: {e}")


def scan_and_trade():
    log.info(f"[{datetime.utcnow()}] Running scheduled scan...")
    main()


if __name__ == "__main__":

        print("⏳ Scheduler started... Running every 60 seconds.")
        scan_and_trade()  # Optional initial run
