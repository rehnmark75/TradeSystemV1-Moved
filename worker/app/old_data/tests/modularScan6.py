import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pandas_ta as ta
import pytz


from services.data_utils import *
from services.ema_signals import *
from services.backtesting import *
from services.live_scanner import *
from services.enhance_data import *


def extract_pair_from_epic(epic):
    """Extract currency pair from IG epic format"""
    # Example: 'CS.D.EURUSD.MINI.IP' -> 'EURUSD'
    parts = epic.split('.')
    if len(parts) >= 3:
        return parts[2]  # Usually the currency pair
    return 'EURUSD'  # Default fallback



# --- Config ---
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

EPIC_LIST = [
    'CS.D.EURUSD.MINI.IP'
]

epic = "CS.D.GBPUSD.MINI.IP"
pair = extract_pair_from_epic(epic)

#df_5m_enhanced, df_15m_enhanced, df_1h_enhanced = sr.enhance_candle_data_complete(engine, epic, 'EURUSD')
#df_15m_with_emas = sc.add_ema_indicators(df_15m_enhanced, periods=[9, 21, 200])

# Get the last 10 signals that would have triggered
# Run 5-minute historical analysis
signals = run_timezone_aware_backtesting(
    engine=engine,
    epic_list=EPIC_LIST,
    num_signals=20,
    user_timezone='Europe/Stockholm'  # Your timezone
)

print(f"Found {len(signals)} signals")