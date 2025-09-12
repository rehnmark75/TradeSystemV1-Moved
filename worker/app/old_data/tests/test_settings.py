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
def fetch_candle_data(engine, epic, timeframe=15, lookback_hours=96):
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
def plot_alert_chart(df, epic, timeframe, image_path):
    top = df['high'].max()
    bottom = df['low'].min()
    premium = 0.7 * top + 0.3 * bottom
    discount = 0.3 * top + 0.7 * bottom
    equilibrium = 0.5 * (top + bottom)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df['start_time'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'],
                                 increasing_line_color="green", decreasing_line_color="red"))

    fig.add_trace(go.Scatter(x=df['start_time'], y=df['ema_12'], mode="lines", name="EMA12", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df['start_time'], y=df['ema_50'], mode="lines", name="EMA50", line=dict(color="orange")))

    fig.add_shape(type="line", x0=df['start_time'].iloc[0], x1=df['start_time'].iloc[-1],
                  y0=equilibrium, y1=equilibrium, line=dict(color="purple", width=1, dash="dash"))
    fig.add_hrect(y0=discount, y1=premium, fillcolor="lightgrey", opacity=0.2, line_width=0)

    bull_df = df[df['bull_alert']]
    fig.add_trace(go.Scatter(x=bull_df['start_time'], y=bull_df['close'],
                             mode="markers", marker=dict(color="green", size=10, symbol="triangle-up"),
                             name="Bull"))

    bear_df = df[df['bear_alert']]
    fig.add_trace(go.Scatter(x=bear_df['start_time'], y=bear_df['close'],
                             mode="markers", marker=dict(color="red", size=10, symbol="triangle-down"),
                             name="Bear"))

    fig.update_layout(title=f"{epic} - {timeframe}m EMA Alerts", xaxis_title="Time", yaxis_title="Price",
                      plot_bgcolor="white", width=1000, height=600)

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

Only take the last triangle from the right into consideration. Ignore any other triangle.

Question:
Is the last bullish (green triangle) alert within or below the grey equilibrium zone?
"""
    elif alert_type == "bear":
        prompt = f"""Please answer the following question with true or false only.
Respond as a JSON object like:
{{"bearish_within_or_above_zone": true}}

Only take the last triangle from the right into consideration. Ignore any other triangle.

Question:
Is the last bearish (red triangle) alert within or above the grey equilibrium zone?
"""
    else:
        raise ValueError("alert_type must be 'bull' or 'bear'")

    # Call Gemini
    response = gemini_model.generate_content([prompt, image])

    print("\n📈 Gemini Analysis (Raw):\n")
    print(response.text)

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


timeframe = 15
lookback_hours = 8

alert_log = load_alert_log()

def main():
    # Get all epics
    epics = get_epics(engine)
    for epic in epics:
        print(f"\n🔍 Scanning {epic}...")

        try:
            image_path = f"/app/tradesetups/{epic.replace('.', '_')}_snapshot.png"
            
            # Fetch & analyze candles
            df = fetch_candle_data(engine, epic, timeframe, lookback_hours)
            df = detect_ema_alerts(df)
            last_alert = get_last_alert_type(df)

            if last_alert:
                timestamp_str = str(last_alert["timestamp"])
                alert_key = f"{epic}:{timestamp_str}:{last_alert['type']}"

                # ✅ Check if we've already alerted
                if alert_log.get(epic) == alert_key:
                    print(f"⏩ Already alerted for {epic} at {timestamp_str}, skipping.")
                    continue
                
                print(f"Last alert: {last_alert['timestamp']}, for epic {epic}")
                print(f"Type: {last_alert['type']}")
                print(f"Price: {last_alert['price']:.5f}")


                # Plot and validate
                img_path = plot_alert_chart(df, epic, timeframe, image_path)
                is_valid = analyze_chart_with_gemini(img_path, epic, last_alert["type"])

                print("✅ Alert confirmed by Gemini:", is_valid)
            else:
                print("⚠️ No alerts found.")

            direction = None
            if last_alert["type"] == "bull":
                direction = "BUY"
            elif last_alert["type"] == "bear":
                direction = "SELL"
            
            if is_valid:
                #send_order(epic,direction)
                #alert_log[epic] = alert_key
                #save_alert_log(alert_log)
                print(f"📤 Order sent: {epic} {direction}")

        except Exception as e:
            print(f"❌ Error processing {epic}: {e}")

def scan_and_trade():
    print(f"[{datetime.utcnow()}] Running scheduled scan...")
    main()


if __name__ == "__main__":

        print("⏳ Scheduler started... Running every 60 seconds.")
        scan_and_trade()  # Optional initial run

        