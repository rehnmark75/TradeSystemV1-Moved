# ig_stream_to_postgres.py
import asyncio
from trading_ig import IGService, IGStreamService
from trading_ig.lightstreamer import Subscription
import asyncpg
import os
from config import IG_API_KEY, IG_PWD, IG_USERNAME, IG_ACCOUNT_TYPE, DATABASE_URL
from services.keyvault import get_secret
from datetime import datetime

EPICS = [
    "CS.D.EURUSD.MINI.IP",
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP"
]

RESOLUTIONS = ["MINUTE_5", "MINUTE_15"]

FIELDS = ["BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE", "LTP_CLOSE"]

    
# IG credentials
username = IG_USERNAME
password = get_secret(IG_PWD)
api_key = get_secret(IG_API_KEY)
account_type = IG_ACCOUNT_TYPE

pg_dsn=DATABASE_URL

# Updated insert query for your schema
INSERT_QUERY = """
INSERT INTO ig_candles (
    start_time, epic, timeframe,
    open, high, low, close, volume
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8
)
ON CONFLICT DO NOTHING;
"""

def resolution_to_minutes(res: str) -> int:
    return int(res.split("_")[1]) if "_" in res else 1

async def insert_candle(pool, epic, resolution, values):
    try:
        ts = values.get("UTM")
        if not ts:
            return
        timestamp = datetime.utcfromtimestamp(float(ts) / 1000.0)
        open_ = float(values.get("BID_OPEN", "nan"))
        high = float(values.get("BID_HIGH", "nan"))
        low = float(values.get("BID_LOW", "nan"))
        close = float(values.get("BID_CLOSE", "nan"))
        volume = int(values.get("CONS_TICK_COUNT", "0"))  # or 0 if not available
        tf = resolution_to_minutes(resolution)

        async with pool.acquire() as conn:
            await conn.execute(
                INSERT_QUERY,
                timestamp, epic, tf,
                open_, high, low, close, volume
            )
        print(f"Inserted {epic} {tf}m @ {timestamp}")
    except Exception as e:
        print(f"Insert error: {e}")


async def main():
    # Create REST and streaming session
    ig_service = IGService(username, password, api_key, acc_type=account_type)
    ig_service.create_session()

    if not ig_service.lightstreamer_endpoint:
        raise RuntimeError("❌ IG session failed or Lightstreamer endpoint missing.")

    ig_stream = IGStreamService(ig_service)

    if ig_stream.ls_client is None:
        raise RuntimeError("❌ Lightstreamer client not initialized — check credentials or API key.")

    
    # Create DB connection pool
    pool = await asyncpg.create_pool(dsn=pg_dsn)

    # Subscriptions
    subs = []
    for epic in EPICS:
        for res in RESOLUTIONS:
            item = f"CHART:{epic}:{res}"
            sub = Subscription(
                mode="DISTINCT",
                items=[item],
                fields=FIELDS + ["UTM"]
            )
            sub.set_adapter("CHART")  # ✅ set adapter here

            async def handle_update(values, epic=epic, resolution=res):
                await insert_candle(pool, epic, resolution, values)

            def listener(update):
                asyncio.create_task(handle_update(update["values"]))

            sub.addlistener(listener)
            subs.append(sub)

    # Connect stream and attach subscriptions
    if ig_stream.ls_client is None:
        raise RuntimeError("❌ Lightstreamer client not initialized")

    ig_stream.ls_client.connect()

    for sub in subs:
        ig_stream.ls_client.subscribe(sub)

    print("✅ Streaming started.")
    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("🛑 Stopping...")
        ig_stream.disconnect()


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  # Needed for some event loop cases (e.g., in notebooks)
    asyncio.run(main())
