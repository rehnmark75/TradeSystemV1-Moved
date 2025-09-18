#!/usr/bin/env python3
"""
Manual backfill for missing NZDUSD 5-minute data
Fills gap from August 23, 2025 to September 18, 2025
"""

import asyncio
import httpx
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime, timedelta
from services.db import SessionLocal
from services.models import IGCandle
from igstream.ig_auth_prod import ig_login
from services.keyvault import get_secret

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.ig.com/gateway/deal"

def parse_snapshot_time(ts_str):
    """Parse IG timestamp format"""
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")

async def backfill_nzdusd_5min():
    """Backfill NZDUSD 5-minute data for Aug 23 - Sep 18, 2025"""

    epic = "CS.D.NZDUSD.MINI.IP"
    timeframe = "MINUTE_5"
    tf_minutes = 5

    # Define the missing data period
    start_time = datetime(2025, 8, 23, 0, 0, 0)
    end_time = datetime(2025, 9, 18, 19, 25, 0)

    logger.info(f"🎯 NZDUSD 5-minute backfill")
    logger.info(f"📅 Period: {start_time} to {end_time}")
    logger.info(f"⏱️ Duration: {(end_time - start_time).days} days")

    # Check current data coverage
    with SessionLocal() as session:
        existing_count = session.query(IGCandle).filter(
            IGCandle.epic == epic,
            IGCandle.timeframe == tf_minutes,
            IGCandle.start_time >= start_time,
            IGCandle.start_time <= end_time
        ).count()

        expected_candles = int((end_time - start_time).total_seconds() / 60 / 5)
        logger.info(f"📊 Currently have {existing_count} candles in this period")
        logger.info(f"📊 Expected candles: {expected_candles}")
        logger.info(f"📊 Missing candles: {expected_candles - existing_count}")

    # Authenticate with IG API
    try:
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
        logger.info("✅ Authenticated with IG API")
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return

    # Process in 12-hour chunks to respect API limits
    chunk_hours = 12
    current_start = start_time
    total_saved = 0

    async with httpx.AsyncClient(base_url=API_BASE_URL, headers=headers, timeout=30.0) as client:
        while current_start < end_time:
            current_end = min(current_start + timedelta(hours=chunk_hours), end_time)

            logger.info(f"🔄 Fetching {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")

            try:
                url = f"/prices/{epic}"
                params = {
                    "resolution": timeframe,
                    "from": current_start.isoformat(),
                    "to": current_end.isoformat(),
                    "max": 1000
                }

                response = await client.get(url, params=params)
                logger.info(f"📡 Status Code: {response.status_code}")
                response.raise_for_status()

                data = response.json()
                candles = data.get("prices", [])
                logger.info(f"📊 Received {len(candles)} candles from API")

                if not candles:
                    logger.warning(f"⚠️ No candles returned for period")
                    current_start = current_end
                    continue

                saved_count = 0
                with SessionLocal() as session:
                    for candle_data in candles:
                        try:
                            # Parse timestamp
                            ts = parse_snapshot_time(candle_data["snapshotTime"])

                            # Check if already exists
                            exists = session.query(IGCandle).filter_by(
                                epic=epic, timeframe=tf_minutes, start_time=ts
                            ).first()

                            if exists:
                                continue

                            # Extract bid prices (using bid for consistency)
                            try:
                                open_price = candle_data["openPrice"]["bid"]
                                high_price = candle_data["highPrice"]["bid"]
                                low_price = candle_data["lowPrice"]["bid"]
                                close_price = candle_data["closePrice"]["bid"]
                            except (KeyError, TypeError):
                                logger.warning(f"⚠️ Skipping candle at {ts} due to missing price fields")
                                continue

                            # Validate price data
                            if None in (open_price, high_price, low_price, close_price):
                                logger.warning(f"⚠️ Skipping incomplete candle at {ts}")
                                continue

                            # Create new candle record
                            candle = IGCandle(
                                start_time=ts,
                                epic=epic,
                                timeframe=tf_minutes,
                                open=open_price,
                                high=high_price,
                                low=low_price,
                                close=close_price,
                                volume=candle_data.get("lastTradedVolume", 0),
                                ltv=candle_data.get("lastTradedVolume"),
                                cons_tick_count=None,
                                data_source="nzdusd_5m_backfill"
                            )
                            session.add(candle)
                            saved_count += 1

                        except Exception as e:
                            logger.warning(f"⚠️ Error processing candle: {e}")
                            continue

                    session.commit()
                    total_saved += saved_count
                    logger.info(f"✅ Saved {saved_count} new candles for this chunk")

            except Exception as e:
                if "exceeded-account-historical-data-allowance" in str(e):
                    logger.error(f"❌ API weekly limit exceeded!")
                    break
                else:
                    logger.error(f"❌ Error fetching data: {e}")

            # Rate limiting
            await asyncio.sleep(2)

            # Move to next chunk
            current_start = current_end

    logger.info(f"🎉 BACKFILL COMPLETE! Total saved: {total_saved} candles")

    # Final verification
    with SessionLocal() as session:
        final_count = session.query(IGCandle).filter(
            IGCandle.epic == epic,
            IGCandle.timeframe == tf_minutes,
            IGCandle.start_time >= start_time,
            IGCandle.start_time <= end_time
        ).count()

        backfilled_count = session.query(IGCandle).filter(
            IGCandle.epic == epic,
            IGCandle.timeframe == tf_minutes,
            IGCandle.data_source == "nzdusd_5m_backfill"
        ).count()

        expected = int((end_time - start_time).total_seconds() / 60 / 5)

        logger.info(f"\n📊 FINAL RESULTS:")
        logger.info(f"  Total 5m candles in period: {final_count}")
        logger.info(f"  Newly backfilled candles: {backfilled_count}")
        logger.info(f"  Expected candles: {expected}")
        logger.info(f"  Coverage: {final_count/expected*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(backfill_nzdusd_5min())