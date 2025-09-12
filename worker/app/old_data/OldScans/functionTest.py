import os
import pandas as pd
import requests
from sqlalchemy import create_engine, text
import schedule
import time
import os
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import pytz

# Create dated log filename
log_filename = datetime.utcnow().strftime("/app/logs/scan_%Y-%m-%d.log")
ALERT_FILE = Path("/app/logs/last_alerts.json")

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

""" EPIC_MAP = {
    "CS.D.EURUSD.MINI.IP": "EURUSD.1.MINI",
    "CS.D.GBPUSD.MINI.IP": "GBPUSD.1.MINI",
    "CS.D.USDJPY.MINI.IP": "USDJPY.100.MINI",
    "CS.D.AUDUSD.MINI.IP": "AUDUSD.1.MINI"
} """

EPIC_MAP = {
    "CS.D.EURUSD.MINI.IP": "EURUSD.1.MINI"
}

print("BOOTING...")

ORDER_API_URL = "http://fastapi-dev:8000/orders/place-order"  # Update if hosted elsewhere
API_SUBSCRIPTION_KEY = "436abe054a074894a0517e5172f0e5b6"

def load_alerts():
    if ALERT_FILE.exists():
        return {k: pd.Timestamp(v) for k, v in json.loads(ALERT_FILE.read_text()).items()}
    return {}

def save_alerts(cache):
    data = {k: str(v) for k, v in cache.items()}
    ALERT_FILE.write_text(json.dumps(data, indent=2))

def get_database_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set.")
    return create_engine(db_url)

import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text

def fetch_candle_data(engine, epic, timeframe=15, lookback_hours=96):
    """
    Fetch raw candles from the database. If 15m is requested but only 5m exists,
    synthesize 15m candles from 5m data.
    """
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

    # Ensure timestamp is datetime
    df['start_time'] = pd.to_datetime(df['start_time'])

    if timeframe == 15 and source_tf == 5:
        # Resample to 15-minute candles
        df.set_index("start_time", inplace=True)

        df_15m = df.resample("15min", label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna().reset_index()

        return df_15m

    return df.reset_index(drop=True)


def get_epics(engine):
    query = text("SELECT DISTINCT epic FROM ig_candles ORDER BY epic")
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]

def debug_candle(df, timestamp_str):
    ts = pd.Timestamp(timestamp_str)
    row = df[df['start_time'] == ts]
    if row.empty:
        print(f"No candle found at {ts}")
        return

    row = row.iloc[0]
    print(f"\n--- Debug for {ts} ---")
    for col in ['close', 'ema_12', 'ema_50', 'prev_close', 'prev_ema_12', 'prev2_close', 'prev2_ema_12',
                'bull_cross', 'bull_condition', 'bull_alert',
                'bear_cross', 'bear_condition', 'bear_alert']:
        print(f"{col}: {row[col]}")

def calculate_emas(df, ema_short=12, ema_long=50):
    """
    Adds EMA columns to the dataframe.
    """
    df[f'ema_{ema_short}'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df[f'ema_{ema_long}'] = df['close'].ewm(span=ema_long, adjust=False).mean()
    return df

def detect_ema_alerts(df):
    eps = 1e-8
    df = df.sort_values('start_time').reset_index(drop=True)

    # Calculate EMAs
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Previous values for crossover logic
    df['prev_close'] = df['close'].shift(1)
    df['prev_ema_12'] = df['ema_12'].shift(1)

    # Candle direction
    df['is_bullish'] = df['close'] > df['open']
    df['is_bearish'] = df['close'] < df['open']

    # Bullish crossover logic
    df['bull_cross'] = (
        (df['prev_close'] < df['prev_ema_12'] - eps) &
        (df['close'] > df['ema_12'] + eps) &
        (df['is_bullish'])  # must be a bullish candle
    )
    df['bull_condition'] = (
        (df['close'] > df['ema_50'] + eps) &
        (df['ema_12'] > df['ema_50'] + eps)
    )
    df['bull_alert'] = df['bull_cross'] & df['bull_condition']

    # Bearish crossover logic
    df['bear_cross'] = (
        (df['prev_close'] > df['prev_ema_12'] + eps) &
        (df['close'] < df['ema_12'] - eps) &
        (df['is_bearish'])  # must be a bearish candle
    )
    df['bear_condition'] = (
        df['ema_50'] > df['ema_12'] + eps
    )
    df['bear_alert'] = df['bear_cross'] & df['bear_condition']

    return df





def main():
    try:
        #log.info("🛠️ main(): Getting DB engine...")
        #pd.set_option("display.max_columns", None)
        #pd.set_option("display.width", 200)
        engine = get_database_engine()

        #log.info("📥 main(): Fetching epics...")
        epics = get_epics(engine)
        #log.info(f"🔢 Found {len(epics)} epics.")

        #for epic in epics:
            #log.info(f"📊 Processing: {epic}")
        df = fetch_candle_data(engine, "CS.D.USDJPY.MINI.IP", lookback_hours=100)
        #print(f"🧾 DataFrame length: {len(df)}")
        #print(df.tail(10))
        
        
        #last_alerts = load_alerts()
        df = calculate_emas(df)
        #df = detect_crossovers(df, confirm_with_next_bar=False, debug=True)
        # Strict version (original logic, requires actual cross)
        df = detect_ema_alerts(df)
    
        df['start_time'] = df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Stockholm')
        alerts = df[df['bull_alert'] | df['bear_alert']].copy()
        alerts = alerts.sort_values('start_time').reset_index(drop=True)
        print(alerts[['start_time', 'bull_alert', 'bear_alert']])      
        #debug_candle(df, "2025-05-26 18:00:00+02:00")


    except Exception as e:
        print(f"🚨 Error in main(): {e}")


def scan_and_trade():
    print(f"[{datetime.utcnow()}] Running scheduled scan...")
    main()

# Schedule to run every minute
#schedule.every(1).minutes.do(scan_and_trade)

if __name__ == "__main__":
    print("⏳ Scheduler started... Running every 60 seconds.")
    main()
    #scan_and_trade()  # Optionally run once at startup

    #while True:
    #    schedule.run_pending()        time.sleep(1) """