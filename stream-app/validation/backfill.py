# backfill.py

import asyncio
import httpx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime, timedelta, time
from services.db import SessionLocal
from services.models import IGCandle
from igstream.ig_auth_prod import ig_login
from services.keyvault import get_secret

API_BASE_URL = "https://api.ig.com/gateway/deal"

EPICS = [
    "CS.D.AUDUSD.MINI.IP"
]

""" EPICS = [
    "CS.D.EURUSD.CEEM.IP",
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP"
] """

TIMEFRAME = "MINUTE_5"
RESOLUTION_TO_DELTA = {"MINUTE_5": timedelta(minutes=5)}

def parse_snapshot_time(ts_str):
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")

def generate_hourly_intervals(start_date: datetime, end_date: datetime, hour_from: int, hour_to: int):
    """Generate datetime intervals for each day between start_date and end_date,
       but only for specified hour ranges (e.g. 8 to 16)."""
    intervals = []
    current_day = start_date.date()

    while current_day <= end_date.date():
        day_start = datetime.combine(current_day, time(hour_from))
        day_end = datetime.combine(current_day, time(hour_to))

        # Only add if within the selected global range
        if day_end >= start_date and day_start <= end_date:
            interval_start = max(start_date, day_start)
            interval_end = min(end_date, day_end)
            intervals.append((interval_start, interval_end))

        current_day += timedelta(days=1)

    return intervals

async def backfill_candles(epic: str, from_dt: datetime, to_dt: datetime, headers):
    print("using backfill candle function")
    async with httpx.AsyncClient(base_url=API_BASE_URL, headers=headers) as client:
        print(f"\n📥 Starting backfill for {epic} [5m] from {from_dt} to {to_dt}")
        step = RESOLUTION_TO_DELTA[TIMEFRAME] * 1000  # ~1000 bars per request
        tf_val = 5

        cursor = from_dt
        while cursor < to_dt:
            chunk_end = min(cursor + step, to_dt)
            print(f"🔄 Fetching candles from {cursor} to {chunk_end}")

            url = f"/prices/{epic}"
            params = {
                "resolution": TIMEFRAME,
                "from": cursor.isoformat(),
                "to": chunk_end.isoformat(),
                "max": 1000
            }

            response = await client.get(url, params=params)

            print(f"🌐 Requesting: {response.url}")
            print(f"📡 Status Code: {response.status_code}")

            try:
                response.raise_for_status()
            except Exception as e:
                print(f"❌ Request failed: {e}")
                print(f"📩 Response content: {response.text}")
                raise  # Optional: re-raise or skip this chunk

            try:
                data = response.json()
            except Exception as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"🧾 Raw response: {response.text}")
                raise

            candles = data.get("prices", [])
            print(f"📊 Candles received: {len(candles)}")


            if not candles:
                print(f"⚠️ No candles returned for {epic} from {cursor} to {chunk_end}")
                cursor = chunk_end
                continue

            saved_count = 0
            with SessionLocal() as session:
                for c in candles:
                    ts = parse_snapshot_time(c["snapshotTime"])

                    exists = session.query(IGCandle).filter_by(
                        epic=epic, timeframe=tf_val, start_time=ts
                    ).first()
                    if exists:
                        continue

                    try:
                        open_ = c["openPrice"]["bid"]
                        high = c["highPrice"]["bid"]
                        low = c["lowPrice"]["bid"]
                        close = c["closePrice"]["bid"]
                    except (KeyError, TypeError):
                        print(f"⚠️ Skipping candle at {ts} due to missing price fields")
                        continue  # skip this candle

                    # Only insert if all price fields are present
                    if None in (open_, high, low, close):
                        print(f"⚠️ Skipping incomplete candle at {ts}")
                        continue

                    candle = IGCandle(
                        start_time=ts,
                        epic=epic,
                        timeframe=tf_val,
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=c.get("lastTradedVolume", 0),
                        ltv=c.get("lastTradedVolume", None),
                        cons_tick_count=None
                    )
                    session.add(candle)
                    saved_count += 1

                session.commit()

            print(f"✅ Saved {saved_count} new candles for {epic} [5m] from {cursor} to {chunk_end}")
            cursor = chunk_end

async def main():
    
    api_key = get_secret("prodapikey")
    ig_pwd = get_secret("prodpwd")
    ig_usr = "rehnmarkh"

    auth = await ig_login(api_key, ig_pwd, ig_usr, api_url=API_BASE_URL)
    headers = {
        "CST": auth["CST"],
        "X-SECURITY-TOKEN": auth["X-SECURITY-TOKEN"],
        "VERSION": "3",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-IG-API-KEY": api_key
    }

    # 🗓️ Define your backfill date and hourly range
    full_from_dt = datetime(2025, 6, 1, 22, 0)  # Sunday 22:00
    full_to_dt   = datetime(2025, 6, 2, 7, 0) 
    #hour_from = 22
    #hour_to = 7

    #intervals = generate_hourly_intervals(full_from_dt, full_to_dt, hour_from, hour_to)

    for epic in EPICS:
            print(f"📦 Dispatching backfill for {epic}")
            await backfill_candles(epic, full_from_dt, full_to_dt , headers)

if __name__ == "__main__":
    asyncio.run(main())
