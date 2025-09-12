import asyncio
import traceback
import logging
from lightstreamer.client import LightstreamerClient, Subscription, SubscriptionListener
from services.ig_orders import is_deal_closed

logger = logging.getLogger("uvicorn")

class PriceListener(SubscriptionListener):
    def __init__(self, epic, direction, sl_price, tp_price, stop_flag):
        self.epic = epic
        self.direction = direction
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.stop_flag = stop_flag
        self.tick_count = 0

    def onItemUpdate(self, item_update):
        self.tick_count += 1
        bid = float(item_update.getValue("BID"))
        offer = float(item_update.getValue("OFFER"))
        price = bid if self.direction == "BUY" else offer

        print(f"[📈 Tick #{self.tick_count}] {self.epic} - Bid: {bid}, Offer: {offer}")

        if self.direction == "BUY" and (price <= self.sl_price or price >= self.tp_price):
            self.stop_flag["triggered"] = True
        elif self.direction == "SELL" and (price >= self.sl_price or price <= self.tp_price):
            self.stop_flag["triggered"] = True

class MyConnectionListener:
    def onStatusChange(self, status):
        print(f"[🔌 Lightstreamer] Connection status: {status}")

    def onServerError(self, errorCode, errorMessage):
        print(f"[❌ Lightstreamer] Server error: {errorCode} - {errorMessage}")

async def stream_prices(
    epic: str,
    headers: dict,
    entry_price: float,
    direction: str,
    sl_price: float,
    tp_price: float,
    deal_id: str
):
    stop_stream_flag = {"triggered": False}

    client = LightstreamerClient("https://demo-apd.marketdatasystems.com", "DEFAULT")
    user = f"{headers['accountId']}"  # ✅ According to IG docs
    password = f"CST-{headers['CST']}|XST-{headers['X-SECURITY-TOKEN']}"

    client.connectionDetails.setUser(user)
    client.connectionDetails.setPassword(password)

    client.addListener(MyConnectionListener())

    subscription = Subscription(
        mode="MERGE",
        items=[f"MARKET:{epic}"],
        fields=["BID", "OFFER", "UPDATE_TIME"]
    )
    subscription.addListener(PriceListener(epic, direction, sl_price, tp_price, stop_stream_flag))

    client.subscribe(subscription)
    def blocking_start():
        client.connect()  # blocking call

    await asyncio.to_thread(blocking_start)
    
    client.connect()

    try:
        while not stop_stream_flag["triggered"]:
            await asyncio.sleep(1)

            # ✅ Auto-disconnect if the deal is closed
            if await is_deal_closed(deal_id, headers):
                print(f"✅ Deal {deal_id} is closed. Disconnecting stream...")
                break
    finally:
        client.disconnect()