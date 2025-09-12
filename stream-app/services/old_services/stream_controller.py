import asyncio
import time
from datetime import datetime, timedelta
from .stream_manager import stream_prices, stream_status
from .ig_auth import ig_login
from ..keyvault import get_secret

KEYVAULT_NAME = "tradersdata"
IG_USERNAME = "rehnmarkhdemo"
IG_API_KEY = "demoapikey"
IG_PWD = "demopwd"

# Config
EPICS = [
    "CS.D.EURUSD.CEEM.IP",
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP"
]

STREAM_REFRESH_INTERVAL = 60  # seconds between stream health checks
AUTH_REFRESH_INTERVAL = 55 * 60  # refresh token every 55 minutes
RECONNECT_BACKOFF = 60  # wait at least 60 seconds between reconnects per epic
TIMEFRAME_MINUTES = 5

# Runtime state
stream_tasks = {}
last_reconnect_attempt = {}
headers = {}

async def start_stream(epic: str):
    global headers
    print(f"🚀 Launching stream for {epic}")
    coroutine = stream_prices(
        epic=epic,
        headers=headers,
        entry_price=0.0,
        deal_id=None,
        timeframe_minutes=TIMEFRAME_MINUTES
    )
    stream_tasks[epic] = asyncio.create_task(coroutine)
    last_reconnect_attempt[epic] = time.time()

async def stop_stream(epic: str):
    task = stream_tasks.get(epic)
    if task and not task.done():
        task.cancel()
        print(f"🛑 Stopped stream for {epic}")
    stream_tasks.pop(epic, None)

async def refresh_auth():
    global headers
    print("🔐 Refreshing IG session...")
    api_key = get_secret(IG_API_KEY)
    ig_pwd = get_secret(IG_PWD)

    auth = await ig_login(api_key, ig_pwd, IG_USERNAME)
    headers = {
        "CST": auth["CST"],
        "X-SECURITY-TOKEN": auth["X-SECURITY-TOKEN"],
        "accountId": auth["ACCOUNT_ID"]
    }
    print("✅ Auth refreshed.")

async def watchdog():
    while True:
        for epic in EPICS:
            status = stream_status.get("last_status")
            is_running = stream_status.get("running") and stream_status.get("epic") == epic
            now = time.time()

            if not is_running or status in ["DISCONNECTED", "STALLED"]:
                if now - last_reconnect_attempt.get(epic, 0) >= RECONNECT_BACKOFF:
                    print(f"⚠️ Stream {epic} is down ({status}). Reconnecting...")
                    await stop_stream(epic)
                    await asyncio.sleep(2)
                    await start_stream(epic)
        await asyncio.sleep(STREAM_REFRESH_INTERVAL)

async def auth_refresher():
    while True:
        await refresh_auth()
        await asyncio.sleep(AUTH_REFRESH_INTERVAL)

async def start_all_streams():
    await refresh_auth()
    for epic in EPICS:
        await start_stream(epic)
    # Run watchdogs in background
    asyncio.create_task(watchdog())
    asyncio.create_task(auth_refresher())

async def stop_all_streams():
    for epic in list(stream_tasks.keys()):
        await stop_stream(epic)


