import asyncio
import httpx
from lightstreamer.client import SubscriptionListener


# Make sure this matches your actual environment
API_BASE_URL = "https://demo-api.ig.com/gateway/deal"



class TradeListener(SubscriptionListener):
    def __init__(self, deal_id, entry_price, headers, direction, stop_callback, trailing_distance=15):
        print(f"🧩 TradeListener init: deal_id={deal_id}, entry={entry_price}, direction={direction}, trailing={trailing_distance}")
        #print(f"🧩 Headers: {headers}")

        try:
            self.deal_id = deal_id
            self.entry_price = float(entry_price)
            self.headers = headers
            self.direction = direction.upper()
            self.trailing_distance = trailing_distance
            self.max_price = self.entry_price
            self.stop_callback = stop_callback
            self.stop_adjusted = False
        except Exception as e:
            print(f"🚨 Error during TradeListener init: {e}")
            raise

    def onItemUpdate(self, update):
        print(f"🔄 onItemUpdate triggered for {self.deal_id}")
        try:
            status = update.getValue("DEAL_STATUS")
            opn_lvl = float(update.getValue("OPN_LVL")) if update.getValue("OPN_LVL") else self.entry_price
            pnl = update.getValue("PROFIT_AND_LOSS")

            print(f"[TRADE] {self.deal_id} | Status: {status} | PnL: {pnl} | Entry: {opn_lvl}")

            if status == "CLOSED":
                print(f"🔚 Trade {self.deal_id} closed. Unsubscribing.")
                self.stop_callback()
                return

            try:
                market_price = float(update.getValue("MARKET_PRICE"))
            except (TypeError, ValueError):
                return

            if self.direction == "BUY":
                if market_price > self.max_price:
                    self.max_price = market_price
                    if not self.stop_adjusted and self.max_price - self.entry_price >= self.trailing_distance / 10000:
                        new_stop = round(self.entry_price, 5)
                        asyncio.create_task(self.modify_stop_loss(new_stop))
            elif self.direction == "SELL":
                if market_price < self.max_price:
                    self.max_price = market_price
                    if not self.stop_adjusted and self.entry_price - self.max_price >= self.trailing_distance / 10000:
                        new_stop = round(self.entry_price, 5)
                        asyncio.create_task(self.modify_stop_loss(new_stop))
        except Exception as e:
            print(f"🚨 Error in onItemUpdate: {e}")

    async def modify_stop_loss(self, stop_level):
        print(f"🔧 Updating SL for {self.deal_id} to {stop_level}")

        payload = {
            "stopLevel": stop_level
        }

        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "Version": "2",
            "X-IG-API-KEY": self.headers.get("X-IG-API-KEY"),
            "CST": self.headers.get("CST"),
            "X-SECURITY-TOKEN": self.headers.get("X-SECURITY-TOKEN")
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{API_BASE_URL}/positions/otc/{self.deal_id}",
                    json=payload,
                    headers=headers
                )
                if response.status_code == 200:
                    self.stop_adjusted = True
                    print(f"✅ Stop loss updated for deal {self.deal_id}")
                else:
                    print(f"❌ Failed to update SL: {response.status_code} | {response.text}")
        except Exception as e:
            print(f"🚨 Exception during SL update: {e}")

