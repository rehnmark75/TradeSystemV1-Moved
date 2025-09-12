#dev-app/trade-monitor.py

"""
Trade Monitoring Script

Overview:
This script monitors active trades and updates their stop/limit levels dynamically based on price movement and ATR volatility.

Key Features:
- Monitors trades marked as 'pending' or 'tracking' from the database.
- Uses a 1-hour ATR to determine initial stop distance (break-even trigger).
- Sends stop/limit adjustments to a FastAPI backend endpoint.
- Implements a dry-run mode to safely test logic without sending real orders.
- Includes seeding logic for inserting test candles and trades.

Logic Summary:
1. Every 5 seconds, the monitor loop fetches all trades marked as 'pending' or 'tracking'.
2. For each trade:
   - If it has moved ≥10 points from entry and has not yet triggered:
       • Calculates ATR (1H timeframe) as stop level.
       • Falls back to 8 points if ATR not available.
       • Sends stop to break-even using `sendlog()`.
   - If it has already triggered and has moved another ≥5 points:
       • Sends a trailing stop and limit adjustment (+5 points each).
   - Skips if already updated with the same offset.

Safety:
- Uses a DRY_RUN flag to suppress actual API calls.
- Thread-safe dictionary to avoid duplicate sends.
- Logs each action clearly.

To Test:
- Ensure DRY_RUN is set to True.
- Run the script and observe logs.
- Use `seed_test_trade_and_candles()` to populate fake trade + candle data.
"""

import time
import requests
import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from services.db import SessionLocal
from services.models import TradeLog, IGCandle
import threading
import logging
from logging.handlers import RotatingFileHandler
from decimal import Decimal, ROUND_DOWN

log_file = "/app/logs/trade_monitor.log"
logger = logging.getLogger("trade_monitor")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

from config import ADJUST_STOP_URL
ORDER_API_URL = ADJUST_STOP_URL
API_SUBSCRIPTION_KEY = "436abe054a074894a0517e5172f0e5b6"

_last_sent_offsets = {}
_offsets_lock = threading.Lock()

INITIAL_TRIGGER = 5
TRAILING_STEP = 5

# --- ATR calculation for higher timeframe (e.g. 1H candles) ---
def calculate_atr(db: Session, symbol: str, length: int = 14, timeframe_minutes: int = 60):
    candles = db.query(IGCandle).filter(
        IGCandle.epic == symbol,
        IGCandle.timeframe == timeframe_minutes
    ).order_by(IGCandle.start_time.desc()).limit(length + 1).all()

    if len(candles) < length + 1:
        return None

    candles = list(reversed(candles))
    trs = []
    for i in range(1, len(candles)):
        prev_close = candles[i - 1].close
        high = candles[i].high
        low = candles[i].low
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    return sum(trs) / len(trs)

def price_to_ig_points(price_difference: float, epic: str) -> int:
    if "JPY" in epic:
        point_value = 0.01
    elif any(x in epic for x in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]):
        point_value = 0.0001
    else:
        point_value = 1.0
    points = Decimal(price_difference) / Decimal(point_value)
    return int(points.quantize(Decimal('1.'), rounding=ROUND_DOWN))

def calculate_adjust_direction(direction: str, mode: str) -> str:
    if mode == "increase":
        return "increase" if direction == "BUY" else "decrease"
    elif mode == "decrease":
        return "decrease" if direction == "BUY" else "increase"
    return "increase" if direction == "BUY" else "decrease"

DRY_RUN = False  # Set to False to enable actual sending

def sendorder(epic: str, direction: str, stop_offset_points: int, limit_offset_points: int):
    current = (stop_offset_points, limit_offset_points)
    with _offsets_lock:
        last = _last_sent_offsets.get(epic)
        if last == current:
            logger.debug(f"[SKIPPED] No change for {epic}, already sent ∆stop={stop_offset_points}, ∆limit={limit_offset_points}")
            return

    adjust_stop = calculate_adjust_direction(direction, "increase")
    adjust_limit = calculate_adjust_direction(direction, "increase")

    body = {
        "epic": epic,
        "adjustDirectionStop": adjust_stop,
        "adjustDirectionLimit": adjust_limit,
        "stop_offset_points": stop_offset_points,
        "limit_offset_points": limit_offset_points
    }

    headers = {
        "X-APIM-Gateway": "verified",
        "X-API-KEY": API_SUBSCRIPTION_KEY,
        "Content-Type": "application/json"
    }

    try:
        if DRY_RUN:
            logger.info(f"[DRY RUN] Would send: {json.dumps(body)}")
            return

        response = requests.post(ORDER_API_URL, json=body, headers=headers)

        if response.status_code == 200:
            logger.info(f"[✅ SENT] {epic} ∆stop={stop_offset_points} ∆limit={limit_offset_points}")
            with _offsets_lock:
                _last_sent_offsets[epic] = current

        elif response.status_code == 404:
            logger.warning(f"[❌ CLOSED] No open position for {epic} — marking as closed in DB")
            with SessionLocal() as db:
                updated = db.query(TradeLog).filter(
                    TradeLog.symbol == epic,
                    TradeLog.status.in_(["pending", "tracking"])
                ).update({TradeLog.status: "closed"})
                db.commit()
                logger.info(f"[✔] Marked {updated} trade(s) as closed for {epic}")
        else:
            logger.warning(f"[ERROR] Send failed: {response.status_code} | {response.text}")

    except Exception as e:
        logger.error(f"[EXCEPTION] Failed to send stop/limit update for {epic}: {e}")

def get_latest_close(db: Session, symbol: str):
    return db.query(IGCandle).filter(IGCandle.epic == symbol).order_by(IGCandle.start_time.desc()).first()

def process_trade(trade: TradeLog, close: float, symbol: str, db: Session):
    entry = trade.entry_price
    direction = trade.direction.upper()
    reference = trade.last_trigger_price or entry
    price_move = close - entry if direction == "BUY" else entry - close
    pips_moved = price_to_ig_points(price_move, symbol)

    #logger.info(f"[Processing] Trade {trade.id} {symbol} Entry: {entry}, Close: {close}, Move: {pips_moved} pips")

    triggered = False

    if pips_moved >= INITIAL_TRIGGER and reference == entry:
        atr = calculate_atr(db, symbol, timeframe_minutes=60)
        min_stop_points = 8
        if atr:
            atr_points = price_to_ig_points(atr, symbol)
            logger.info(f"[ATR] 1H ATR for {symbol} = {atr:.5f} → {atr_points} pts")
            stop_offset_points = max(atr_points, min_stop_points)
        else:
            logger.warning(f"[ATR] 1H ATR unavailable for {symbol}. Using fallback = {min_stop_points} pts")
            stop_offset_points = min_stop_points

        sendorder(symbol, direction, stop_offset_points, 0)
        logger.info(f"[Break-even {direction}] Trade {trade.id} {symbol} moved {pips_moved} pips — stop to entry")
        triggered = True

    elif reference != entry and pips_moved >= INITIAL_TRIGGER:
        move_diff = close - reference if direction == "BUY" else reference - close
        move_points = price_to_ig_points(move_diff, symbol)
        if move_points >= TRAILING_STEP:
            sendorder(symbol, direction, TRAILING_STEP, TRAILING_STEP)
            logger.info(f"[Trailing {direction}] Trade {trade.id} {symbol} moved {pips_moved} pips — stop +{TRAILING_STEP}, limit +{TRAILING_STEP}")
            triggered = True
        else:
            logger.debug(f"[No Trigger] Trade {trade.id} {symbol} only moved {move_points} since last trigger")

    if triggered:
        trade.last_trigger_price = close
        trade.trigger_time = datetime.utcnow()
    trade.status = "tracking"

def seed_test_trade_and_candles():
    with SessionLocal() as db:
        from config import DEFAULT_EPICS
        symbol = DEFAULT_EPICS['EURUSD']
        now = datetime.utcnow()

        # Create dummy 1H candles
        for i in range(15):
            start_time = now - timedelta(hours=i + 1)
            candle = IGCandle(
                epic=symbol,
                start_time=start_time,
                open=1.0800 + i * 0.0001,
                high=1.0810 + i * 0.0001,
                low=1.0790 + i * 0.0001,
                close=1.0805 + i * 0.0001,
                volume=1000,
                timeframe=60
            )
            db.merge(candle)

        # Create dummy latest 5m candle
        db.merge(IGCandle(
            epic=symbol,
            start_time=now,
            open=1.0800,
            high=1.0820,
            low=1.0790,
            close=1.0815,
            volume=1200,
            timeframe=5
        ))

        # Insert a test trade
        db.add(TradeLog(
            symbol=symbol,
            entry_price=1.0800,
            direction="BUY",
            status="pending"
        ))

        db.commit()
        logger.info("✅ Seeded test trade and 1H/5m candles")

def monitor_pending_trades():
    logger.info("Starting trade monitor loop...")
    while True:
        with SessionLocal() as db:
            try:
                trades = db.query(TradeLog).filter(
                    TradeLog.status.in_(["pending", "tracking"]),
                    TradeLog.endpoint == "dev"
                ).all()
                for trade in trades:
                    symbol = trade.symbol
                    trade_id = trade.id

                    try:
                        latest = get_latest_close(db, symbol)
                        if not latest:
                            logger.debug(f"[SKIP] No candle for {symbol}")
                            continue
                        process_trade(trade, latest.close, symbol, db)
                        db.commit()
                    except Exception as trade_error:
                        logger.error(f"Error processing trade ID {trade_id} for {symbol}: {trade_error}")
                        trade.status = "closed"
                        trade.trigger_time = datetime.utcnow()
                        db.commit()
                        logger.warning(f"[Closed] Trade ID {trade_id} for {symbol} due to error: {trade_error}")
            except Exception as loop_error:
                logger.error(f"Error in monitor loop: {loop_error}")
        time.sleep(15)

def start_monitoring_thread():
    seed_test_trade_and_candles()
    thread = threading.Thread(target=monitor_pending_trades, daemon=True)
    thread.start()
