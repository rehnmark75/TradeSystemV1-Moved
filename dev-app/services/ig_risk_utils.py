import httpx
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
from .models import Candle
from typing import Literal,List, Dict, Optional
from config import API_BASE_URL
from datetime import datetime

async def get_ema_atr(epic: str, trading_headers: dict, resolution: str = "MINUTE_15", periods: int = 14) -> float:
    """
    Fetch historical price data from IG and calculate the EMA-based ATR.
    """
    limit = periods + 50  # Fetch enough data to warm up the EMA
    url = f"{API_BASE_URL}/prices/{epic}?resolution={resolution}&max={limit}&pageSize={limit}"

    headers = {
        **trading_headers,
        "Version": "3"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        prices = response.json()["prices"]

    df = pd.DataFrame([{
        "high": p["highPrice"]["bid"],
        "low": p["lowPrice"]["bid"],
        "close": p["closePrice"]["bid"]
    } for p in prices])

    df = df[::-1].reset_index(drop=True)  # Sort from oldest to newest

    if len(df) < periods + 1:
        raise ValueError(f"Not enough data to calculate EMA ATR({periods})")

    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df[["high", "prev_close"]].max(axis=1) - df[["low", "prev_close"]].min(axis=1)

    # EMA-based ATR
    atr = df["tr"].ewm(span=periods, adjust=False).mean().iloc[-1]
    return round(float(atr), 5)


async def calculate_dynamic_sl_tp(epic: str, trading_headers: dict, atr: float, rr_ratio: float = 2.0) -> dict:
    """
    Calculate valid stop-loss and take-profit distances based on ATR and IG market constraints.
    """
    url = f"{API_BASE_URL}/markets/{epic}"

    headers = {
        **trading_headers,
        "Version": "3"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        response_json = response.json()

    instrument = response_json["instrument"]
    dealing_rules = response_json["dealingRules"]
    #print("Dealing rules keys:", dealing_rules.keys())
    min_stop = float(dealing_rules["minNormalStopOrLimitDistance"]["value"])
    stop_step = float(dealing_rules["minStepDistance"]["value"])

    raw_stop = atr * 1.5
    valid_stop = max(min_stop, round(raw_stop / stop_step) * stop_step)

    raw_limit = valid_stop * rr_ratio
    valid_limit = round(raw_limit / stop_step) * stop_step

    return {
        "stopDistance": str(valid_stop),
        "limitDistance": str(valid_limit)
    }


def find_swing_based_stop_distance(
    db: Session,
    epic: str,
    direction: Literal["BUY", "SELL"],
    entry_price: float,
    timeframe: int = 5,
    lookback: int = 20,
    fallback_points: int = 10
) -> int:
    """
    Returns stop distance in points based on swing structure and price scaling.
    Falls back to `fallback_points` if no swing is found.
    """

    candles = (
        db.query(Candle)
        .filter(Candle.epic == epic, Candle.timeframe == timeframe)
        .order_by(Candle.start_time.desc())
        .limit(lookback)
        .all()
    )

    if len(candles) < 3:
        print("returning to fallback_points")
        return fallback_points

    candles = list(reversed(candles))

    for i in range(1, len(candles) - 1):
        prev = candles[i - 1]
        curr = candles[i]
        next_ = candles[i + 1]

        if direction == "BUY" and prev.low > curr.low < next_.low:
            distance = entry_price - curr.low
            break
        elif direction == "SELL" and prev.high < curr.high > next_.high:
            distance = curr.high - entry_price
            break
    else:
        return fallback_points  # No swing found

    # 🧠 Dynamic scaling based on price magnitude
    high_price = max(c.high for c in candles)
    scaling_factor = 100 if high_price > 100 else 10000

    return max(fallback_points, round(distance * scaling_factor))

def calculate_emas(df):
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df


def get_stop_distance_from_ema(engine, epic, direction, min_distance=0.0005, ema_lookback=100):
    """
    Fetch recent candles, calculate EMA12/EMA50, and return a stopDistance based on their spread.

    Args:
        engine (sqlalchemy.Engine): DB engine
        epic (str): Instrument epic
        direction (str): 'BUY' or 'SELL'
        min_distance (float): Minimum allowed stopDistance (e.g. 0.0005 for forex)
        ema_lookback (int): Number of candles to fetch for EMA calculation

    Returns:
        float: stopDistance (rounded to instrument-appropriate precision)
    """

    # Step 1: Fetch recent candles
    query = text("""
        SELECT start_time, close
        FROM candles
        WHERE epic = :epic AND timeframe = 15
        ORDER BY start_time DESC
        LIMIT :limit
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"epic": epic, "limit": ema_lookback})
    
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient data to calculate EMAs for {epic}")

    # Step 2: Ensure correct time order
    df = df.sort_values("start_time").reset_index(drop=True)

    # Step 3: Calculate EMAs using provided function
    df = calculate_emas(df)

    # Step 4: Get the latest row
    latest = df.iloc[-1]
    ema12 = latest['ema_12']
    ema50 = latest['ema_50']
    entry_price = latest['close']
    ema_distance = abs(ema12 - ema50)

    # Step 5: Determine rounding precision based on entry price
    if entry_price > 100:
        rounding = 2  # e.g. USDJPY
    elif entry_price > 10:
        rounding = 3
    else:
        rounding = 5  # e.g. EURUSD

    stop_distance = max(ema_distance, min_distance)
    print(f"📏 {epic} | EMA12: {ema12:.5f} | EMA50: {ema50:.5f} | ΔEMA: {ema_distance:.5f} → stopDistance: {stop_distance:.5f}")
    
    return round(stop_distance, rounding)


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

def calculate_trade_levels_from_signal(signals: List[Dict], swings: List[Dict], rr_ratio: float = 2.0) -> Optional[Dict]:
    """
    Calculate entry, stop loss, and take profit levels from the latest BOS/CHoCH signal.

    Parameters:
    - signals: List of BOS/CHoCH signals.
    - swings: Classified swing structure points (HH/LL etc).
    - rr_ratio: Risk-to-reward multiplier (e.g., 2.0 = TP is 2x distance from entry to SL)

    Returns:
    - A dict with trade direction, entry, stop_loss, take_profit, or None if no signal found.
    """

    if not signals or not swings:
        return None

    last_signal = signals[-1]
    direction = last_signal["direction"]
    entry_price = None
    stop_loss = None
    take_profit = None

    # Find the most recent opposite swing for SL reference
    if direction == "bullish":
        entry_price = last_signal["price"]
        recent_lows = [s for s in reversed(swings) if s["type"] == "low" and s["time"] <= last_signal["time"]]
        if not recent_lows:
            return None
        swing_low = recent_lows[0]["price"]
        stop_loss = swing_low
        risk = entry_price - stop_loss
        take_profit = entry_price + (risk * rr_ratio)

    elif direction == "bearish":
        entry_price = last_signal["price"]
        recent_highs = [s for s in reversed(swings) if s["type"] == "high" and s["time"] <= last_signal["time"]]
        if not recent_highs:
            return None
        swing_high = recent_highs[0]["price"]
        stop_loss = swing_high
        risk = stop_loss - entry_price
        take_profit = entry_price - (risk * rr_ratio)

    return {
        "direction": direction,
        "entry": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "signal_time": last_signal["confirmation_time"]
    }

class Pivot:
    def __init__(self, index: int, time: pd.Timestamp, price: float, type_: str):
        self.index = index
        self.time = time
        self.price = price
        self.type = type_  # 'high' or 'low'

# --- Data Types ---
def classify_pivots(pivots: List[Pivot]) -> List[Dict]:
    """ Assign HH/HL/LH/LL labels using last same-type swing reference """
    classified = []
    last_high = None
    last_low = None

    for p in pivots:
        label = None
        if p.type == "high":
            if last_high is None or p.price > last_high.price:
                label = "HH"
            else:
                label = "LH"
            last_high = p
        elif p.type == "low":
            if last_low is None or p.price > last_low.price:
                label = "HL"
            else:
                label = "LL"
            last_low = p

        classified.append({
            "index": p.index,
            "time": p.time,
            "price": p.price,
            "type": p.type,
            "label": label
        })

    return classified


class TrailingExtremes:
    def __init__(self):
        self.top = None
        self.bottom = None
        self.bar_time = None
        self.bar_index = None
        self.last_top_time = None
        self.last_bottom_time = None


# --- Core Functions ---
def detect_pivots(df: pd.DataFrame, lookback: int) -> List[Pivot]:
    """ Detect swing pivots using a leg-style approach """
    df = df.reset_index(drop=True)
    pivots = []
    for i in range(lookback, len(df) - lookback):
        high = df.loc[i, "high"]
        low = df.loc[i, "low"]

        is_swing_high = all(high > df.loc[i - j, "high"] and high > df.loc[i + j, "high"] for j in range(1, lookback + 1))
        is_swing_low = all(low < df.loc[i - j, "low"] and low < df.loc[i + j, "low"] for j in range(1, lookback + 1))

        if is_swing_high:
            pivots.append(Pivot(i, df.loc[i, "start_time"], high, "high"))
        elif is_swing_low:
            pivots.append(Pivot(i, df.loc[i, "start_time"], low, "low"))
    return pivots



def detect_structure_signals_luxalgo(swings: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """
    Detect BOS and CHoCH from classified swings using close confirmation logic (LuxAlgo style).
    Logs output to Streamlit sidebar for inspection.
    """
    signals = []
    trend_bias = None  # +1 bullish, -1 bearish
    debug_log = []

    debug_log.append("--- BOS/CHoCH Detection Debug ---")

    for i in range(1, len(swings)):
        prev = swings[i - 1]
        curr = swings[i]

        debug_log.append(f"Evaluating swing pair: prev={prev['label']}@{prev['price']} → curr={curr['label']}@{curr['price']}")

        if prev["type"] != curr["type"]:
            debug_log.append("  Skipped due to type mismatch")
            continue

        # Scan candles between prev and curr for confirmation
        segment = df[(df["start_time"] > prev["time"]) & (df["start_time"] <= curr["time"])]

        confirmation_row = None
        if curr["type"] == "high":
            confirmation_row = segment[segment["close"] > prev["price"]].head(1)
        elif curr["type"] == "low":
            confirmation_row = segment[segment["close"] < prev["price"]].head(1)

        if confirmation_row is None or confirmation_row.empty:
            debug_log.append("  No signal confirmed")
            continue

        confirm_time = confirmation_row.iloc[0]["start_time"]
        close = confirmation_row.iloc[0]["close"]

        label = None
        direction = None

        debug_log.append(f"  Confirming close={close} vs prev_price={prev['price']}")

        if curr["type"] == "high":
            if trend_bias == -1:
                label, direction = "CHoCH", "bullish"
            else:
                label, direction = "BOS", "bullish"
            trend_bias = 1
        elif curr["type"] == "low":
            if trend_bias == 1:
                label, direction = "CHoCH", "bearish"
            else:
                label, direction = "BOS", "bearish"
            trend_bias = -1

        debug_log.append(f"  → Signal confirmed: {label} ({direction}) at {confirm_time}")
        signals.append({
            "label": label,
            "direction": direction,
            "price": prev["price"],
            "time": confirm_time,
            "from_time": prev["time"],
            "to_time": curr["time"],
            "confirmation_time": confirm_time
        })

    debug_log.append(f"✅ Total signals detected: {len(signals)}")

    return signals

def get_sl_tp_from_labels(swings: List[Dict], risk_reward: float = 2.0) -> List[Dict]:
    """
    Generate SL/TP suggestions based on swing labels (HL/LH entries).
    SL is based on prior LL (for HL) or HH (for LH), with TP set using RR ratio.
    Prices are rounded based on instrument price level.
    """
    results = []

    for i, s in enumerate(swings):
        label = s["label"]
        time = s["time"]
        price = s["price"]
        type_ = s["type"]

        # Determine rounding rule
        if price > 100:
            rounding = 2  # e.g., USDJPY
        elif price > 10:
            rounding = 3
        else:
            rounding = 5  # e.g., EURUSD

        def round_price(p: float) -> float:
            return round(p, rounding)

        if label == "HL":  # Long setup
            prev_ll = next((x for x in reversed(swings[:i]) if x["label"] == "LL"), None)
            if prev_ll:
                sl = prev_ll["price"]
                tp = price + (price - sl) * risk_reward
                results.append({
                    "direction": "long",
                    "entry_label": label,
                    "entry_price": round_price(price),
                    "entry_time": time,
                    "stop_loss": round_price(sl),
                    "take_profit": round_price(tp),
                    "sl_label": prev_ll["label"],
                    "sl_time": prev_ll["time"]
                })

        elif label == "LH":  # Short setup
            prev_hh = next((x for x in reversed(swings[:i]) if x["label"] == "HH"), None)
            if prev_hh:
                sl = prev_hh["price"]
                tp = price - (sl - price) * risk_reward
                results.append({
                    "direction": "short",
                    "entry_label": label,
                    "entry_price": round_price(price),
                    "entry_time": time,
                    "stop_loss": round_price(sl),
                    "take_profit": round_price(tp),
                    "sl_label": prev_hh["label"],
                    "sl_time": prev_hh["time"]
                })

    return results
