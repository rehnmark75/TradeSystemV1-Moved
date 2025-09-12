import asyncio
from asyncio import get_event_loop, run_coroutine_threadsafe
from datetime import timedelta, datetime
import logging
from lightstreamer.client import LightstreamerClient, Subscription, SubscriptionListener

from ..db import SessionLocal
from ..models import Candle
from .tradelistner import TradeListener
from .trade_utils import modify_stop_loss

task_registry = {"task": None}

stream_status = {
    "running": False,
    "epic": None,
    "tick_count": 0,
    "start_time": None,
    "last_status": "INIT",
    "disconnect_reason": None
}

active_stream = {
    "client": None,
    "subscription": None,
    "stop_flag": {"triggered": False}
}

#active_trades = {}
tracked_trades_by_epic = {}

logger = logging.getLogger("uvicorn")

# Optional: global candle backup store (not persisted across restarts)
candle_cache = {}

class PriceListener(SubscriptionListener):
    def __init__(self, epic, stop_flag, db_session, timeframe_minutes):
        self.epic = epic
        self.stop_flag = stop_flag
        self.db = db_session
        self.timeframe = timedelta(minutes=timeframe_minutes)

        self.key = f"{epic}_{timeframe_minutes}"
        self.current_candle = candle_cache.get(self.key)
        self.timeframe_minutes = timeframe_minutes
        self.tick_count = 0
        #stream_status["epic"] = epic
        #stream_status["tick_count"] = 0
        stream_status[self.key] = {"tick_count": 0, "epic": epic}
        
    def onItemUpdate(self, item_update):
        try:
            # === Candle logic ===
            self.tick_count += 1
            
            # FIXED: Use proper candle OHLC fields instead of calculated mid from BID/OFFER
            # Get actual candle data from IG streaming API
            try:
                # Primary candle fields
                mid_open = item_update.getValue("MID_OPEN")
                high = item_update.getValue("HIGH")
                low = item_update.getValue("LOW")
                last_traded = item_update.getValue("LAST_TRADED_PRICE")
                
                # Convert to float, handle None values
                current_price = float(last_traded) if last_traded else float(mid_open) if mid_open else None
                high_price = float(high) if high else current_price
                low_price = float(low) if low else current_price
                open_price = float(mid_open) if mid_open else current_price
                
                if current_price is None:
                    # Fallback to BID/OFFER if candle fields not available (should not happen)
                    bid = float(item_update.getValue("BID"))
                    offer = float(item_update.getValue("OFFER"))
                    current_price = (bid + offer) / 2
                    high_price = low_price = open_price = current_price
                    print(f"⚠️ Using fallback BID/OFFER for {self.epic} - candle fields not available")
                    
            except (ValueError, TypeError) as e:
                print(f"❌ Error parsing candle fields for {self.epic}: {e}")
                return
            
            now = datetime.utcnow()

            if not self.current_candle:
                self.start_new_candle(now, open_price, high_price, low_price, current_price)
            elif now >= self.current_candle["end_time"]:
                self.save_candle_to_db()
                self.start_new_candle(now, open_price, high_price, low_price, current_price)
            else:
                # Update candle with proper OHLC values
                self.current_candle["high"] = max(self.current_candle["high"], high_price)
                self.current_candle["low"] = min(self.current_candle["low"], low_price)
                self.current_candle["close"] = current_price
                self.current_candle["volume"] += 1

            # === TRADE TRACKING: Get BID/OFFER for trade management ===
            # Note: For trade tracking, we still need BID/OFFER (current market prices)
            # This is separate from candle data which uses historical OHLC
            try:
                bid = float(item_update.getValue("BID"))
                offer = float(item_update.getValue("OFFER"))
            except (ValueError, TypeError):
                # Skip trade tracking if BID/OFFER not available
                return

            epic_trades = tracked_trades_by_epic.get(self.epic.upper(), {})
            
            if not epic_trades:
                return

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop_policy().get_event_loop()

            for deal_id, trade in epic_trades.items():
                direction = trade["direction"]
                entry_price = trade["entry_price"]
                symbol_precision = 10000 if "USD" in self.epic and "." in self.epic else 100
                trailing_distance = trade.get("trailing_distance", 10) / symbol_precision
    
                stop_adjusted = trade.get("stop_adjusted", False)
                headers = trade.get("headers")
                
                if not headers:
                    print(f"⚠️ Skipping trade {deal_id} — missing headers")
                    continue
                
                if direction == "BUY":
                    
                    if bid - entry_price >= trailing_distance and not stop_adjusted:
                        new_stop = round(entry_price, 5)
                        print("direction is BUY, new stop")
                        asyncio.run_coroutine_threadsafe(
                            modify_stop_loss(deal_id, new_stop, headers),
                            loop
                        )
                        trade["stop_adjusted"] = True
                        print(f"📉 BUY trade {deal_id}: trailing stop triggered at {new_stop}")

                elif direction == "SELL":
                    
                    if entry_price - bid >= trailing_distance and not stop_adjusted:
                        new_stop = round(entry_price, 5)
                        print("direction is SELL, new stop")
                        asyncio.run_coroutine_threadsafe(
                            modify_stop_loss(deal_id, new_stop, headers),
                            loop
                        )
                        trade["stop_adjusted"] = True
                        print(f"📈 SELL trade {deal_id}: trailing stop triggered at {new_stop}")

        except Exception as e:
            print(f"🚨 Exception in {self.timeframe_minutes}m onItemUpdate: {e}")

        

    def start_new_candle(self, now, open_price, high_price, low_price, close_price):
        aligned_minute = (now.minute // self.timeframe_minutes) * self.timeframe_minutes
        start_time = now.replace(minute=aligned_minute, second=0, microsecond=0)
        end_time = start_time + self.timeframe

        self.current_candle = {
            "start_time": start_time,
            "end_time": end_time,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": 1
        }
        candle_cache[self.key] = self.current_candle

        print(f"🕒 Starting new {self.timeframe_minutes}m candle: {self.epic} from {start_time} to {end_time} | O:{open_price:.5f} H:{high_price:.5f} L:{low_price:.5f} C:{close_price:.5f}")

    def save_candle_to_db(self):
        if self.current_candle["volume"] < 2:
            print(f"⚠️ Skipping {self.timeframe_minutes}m candle for {self.epic} at {self.current_candle['start_time']} (only {self.current_candle['volume']} ticks)")
            return

        print(f"💾 Writing {self.timeframe_minutes}m candle to DB for {self.epic} at {self.current_candle['start_time']}")

        candle = Candle(
            start_time=self.current_candle["start_time"],
            epic=self.epic,
            timeframe=self.timeframe_minutes,
            open=self.current_candle["open"],
            high=self.current_candle["high"],
            low=self.current_candle["low"],
            close=self.current_candle["close"],
            volume=self.current_candle["volume"]
        )
        try:
            with SessionLocal() as session:
                session.add(candle)
                session.commit()
                open_price = candle.open
                close_price = candle.close
            print(f"✅ Saved {self.timeframe_minutes}m candle: {self.epic} | O:{open_price} C:{close_price}")
        except Exception as e:
            print(f"🚨 DB commit failed: {e}")

        self.current_candle = None
        candle_cache.pop(self.key, None)

class MultiTimeframeListener(SubscriptionListener):
    def __init__(self, epic, db_session, stop_flag):
        self.listeners = [
            PriceListener(epic, stop_flag, db_session, 1),
            PriceListener(epic, stop_flag, db_session, 5),
            PriceListener(epic, stop_flag, db_session, 15),
            PriceListener(epic, stop_flag, db_session, 60)  # 1-hour candles
        ]

    def onItemUpdate(self, item_update):
        for listener in self.listeners:
            listener.onItemUpdate(item_update)

class MyConnectionListener:
    def onStatusChange(self, status):
        stream_status["last_status"] = status
        print(f"🔌 Lightstreamer status: {status}")

    def onServerError(self, errorCode, errorMessage):
        print(f"❌ Lightstreamer error: {errorCode} - {errorMessage}")

async def stream_prices(epic, headers, entry_price, deal_id, timeframe_minutes=5):
    stop_stream_flag = {"triggered": False}
    db_session = SessionLocal()

    client = LightstreamerClient("https://demo-apd.marketdatasystems.com", "DEFAULT")
    user = headers['accountId']
    password = f"CST-{headers['CST']}|XST-{headers['X-SECURITY-TOKEN']}"

    client.connectionDetails.setUser(user)
    client.connectionDetails.setPassword(password)
    client.addListener(MyConnectionListener())

    # FIXED: Subscribe to proper candle OHLC fields instead of BID/OFFER
    subscription = Subscription(
        mode="MERGE",
        items=[f"MARKET:{epic}"],
        fields=["MID_OPEN", "HIGH", "LOW", "LAST_TRADED_PRICE", "UPDATE_TIME", "BID", "OFFER"]
    )
    subscription.addListener(MultiTimeframeListener(epic, db_session, stop_stream_flag))
    client.subscribe(subscription)

    active_stream.update({
        "client": client,
        "subscription": subscription,
        "stop_flag": stop_stream_flag
    })

    stream_status.update({
        "running": True,
        "epic": epic,
        "start_time": datetime.utcnow(),
        "tick_count": 0
    })

    try:
        print(f"🔄 Connecting to Lightstreamer for {epic}...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, client.connect)
        print(f"✅ Lightstreamer client.connect() returned for {epic}")

        while not stop_stream_flag["triggered"]:
            await asyncio.sleep(1)
            if stream_status["last_status"] in ["DISCONNECTED", "STALLED"]:
                logger.warning("Stream is stalled/disconnected")
                break
    finally:
        stream_status["running"] = False
        stream_status["epic"] = None
        stream_status["disconnect_reason"] = stream_status["last_status"]
        client.disconnect()
        db_session.close()

def track_trade_subscription(deal_id, epic, entry_price, direction, headers, trailing_distance=15):
    epic = epic.upper()

    if epic not in tracked_trades_by_epic:
        tracked_trades_by_epic[epic] = {}

    if deal_id in tracked_trades_by_epic[epic]:
        print(f"🔁 Already tracking trade {deal_id} for {epic}")
        return

    tracked_trades_by_epic[epic][deal_id] = {
        "entry_price": entry_price,
        "direction": direction,
        "headers": headers,
        "trailing_distance": trailing_distance,
        "stop_adjusted": False
    }

    print(f"🛰️ Now tracking trade {deal_id} for {epic}")



