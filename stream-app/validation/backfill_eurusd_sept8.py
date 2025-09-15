#!/usr/bin/env python3
"""
Manual backfill for missing EURUSD data on September 8, 2025
The epic name change caused a gap in data collection
"""

import asyncio
import httpx
import logging
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

async def backfill_eurusd_sept8():
    """Backfill the missing EURUSD data for September 8, 2025"""
    
    epic = "CS.D.EURUSD.CEEM.IP"
    
    # Define the gap period - Sept 7 21:00 UTC to Sept 9 11:00 UTC
    # But we only want market hours (Sunday 21:00 to Monday end)
    start_time = datetime(2025, 9, 7, 21, 0, 0)  # Sunday 21:00 UTC (market open)
    end_time = datetime(2025, 9, 9, 11, 0, 0)    # Monday 11:00 UTC
    
    logger.info(f"Backfilling EURUSD data from {start_time} to {end_time}")
    
    # Authenticate with IG
    try:
        api_key = get_secret("prodapikey")
        ig_pwd = get_secret("prodpwd")
        ig_usr = "rehnmarkh"
        
        auth = await ig_login(api_key, ig_pwd, ig_usr, api_url="https://api.ig.com/gateway/deal")
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
    
    # Process in chunks to respect API limits
    timeframes = [
        ("MINUTE_5", 5),
        ("MINUTE_15", 15),
        ("HOUR", 60)
    ]
    
    for resolution, tf_minutes in timeframes:
        logger.info(f"\n📊 Processing {resolution} timeframe...")
        
        # Calculate chunk size based on timeframe
        if tf_minutes == 5:
            chunk_hours = 12  # ~144 candles per chunk
        elif tf_minutes == 15:
            chunk_hours = 24  # ~96 candles per chunk
        else:  # 60 min
            chunk_hours = 48  # ~48 candles per chunk
        
        current_start = start_time
        total_saved = 0
        
        while current_start < end_time:
            current_end = min(current_start + timedelta(hours=chunk_hours), end_time)
            
            # Skip weekend closure (Friday 21:00 to Sunday 21:00 UTC)
            if current_start.weekday() == 5:  # Saturday
                current_start = current_start + timedelta(days=1)
                continue
            if current_start.weekday() == 6 and current_start.hour < 21:  # Sunday before 21:00
                current_start = datetime(current_start.year, current_start.month, current_start.day, 21, 0, 0)
                if current_start >= end_time:
                    break
            
            logger.info(f"  Fetching {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    url = f"https://api.ig.com/gateway/deal/prices/{epic}"
                    params = {
                        "resolution": resolution,
                        "from": current_start.strftime("%Y-%m-%dT%H:%M:%S"),
                        "to": current_end.strftime("%Y-%m-%dT%H:%M:%S"),
                        "max": 1000
                    }
                    
                    response = await client.get(url, params=params, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    candles = data.get("prices", [])
                    
                    logger.info(f"    Received {len(candles)} candles")
                    
                    if candles:
                        saved_count = 0
                        with SessionLocal() as session:
                            for candle_data in candles:
                                try:
                                    # Parse timestamp
                                    ts_str = candle_data.get("snapshotTime", "")
                                    ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
                                    
                                    # Check if already exists
                                    exists = session.query(IGCandle).filter_by(
                                        epic=epic,
                                        timeframe=tf_minutes,
                                        start_time=ts
                                    ).first()
                                    
                                    if not exists:
                                        # Calculate mid prices from bid/ask
                                        bid_open = candle_data["openPrice"]["bid"]
                                        bid_high = candle_data["highPrice"]["bid"]
                                        bid_low = candle_data["lowPrice"]["bid"]
                                        bid_close = candle_data["closePrice"]["bid"]
                                        
                                        ask_open = candle_data["openPrice"]["ask"]
                                        ask_high = candle_data["highPrice"]["ask"]
                                        ask_low = candle_data["lowPrice"]["ask"]
                                        ask_close = candle_data["closePrice"]["ask"]
                                        
                                        # Calculate mid prices
                                        mid_open = (bid_open + ask_open) / 2
                                        mid_high = (bid_high + ask_high) / 2
                                        mid_low = (bid_low + ask_low) / 2
                                        mid_close = (bid_close + ask_close) / 2
                                        
                                        # Note: IG Markets changed data format in September 2025
                                        # EURUSD data now comes in correct format, no scaling needed
                                        
                                        candle = IGCandle(
                                            start_time=ts,
                                            epic=epic,
                                            timeframe=tf_minutes,
                                            open=mid_open,
                                            high=mid_high,
                                            low=mid_low,
                                            close=mid_close,
                                            volume=candle_data.get("lastTradedVolume", 0),
                                            ltv=candle_data.get("lastTradedVolume"),
                                            cons_tick_count=None,
                                            data_source="manual_backfill_sept8"
                                        )
                                        session.add(candle)
                                        saved_count += 1
                                
                                except Exception as e:
                                    logger.warning(f"      Error processing candle: {e}")
                                    continue
                            
                            session.commit()
                            total_saved += saved_count
                            logger.info(f"    ✅ Saved {saved_count} new candles")
                    
            except Exception as e:
                logger.error(f"    ❌ Error fetching data: {e}")
            
            # Rate limiting
            await asyncio.sleep(2)
            
            # Move to next chunk
            current_start = current_end
        
        logger.info(f"  📊 Total saved for {resolution}: {total_saved} candles")
    
    # Verify the backfill
    with SessionLocal() as session:
        count_5m = session.query(IGCandle).filter_by(
            epic=epic,
            timeframe=5,
            data_source="manual_backfill_sept8"
        ).count()
        
        count_15m = session.query(IGCandle).filter_by(
            epic=epic,
            timeframe=15,
            data_source="manual_backfill_sept8"
        ).count()
        
        count_60m = session.query(IGCandle).filter_by(
            epic=epic,
            timeframe=60,
            data_source="manual_backfill_sept8"
        ).count()
        
        logger.info("\n✅ BACKFILL COMPLETE!")
        logger.info(f"  5m candles: {count_5m}")
        logger.info(f"  15m candles: {count_15m}")
        logger.info(f"  60m candles: {count_60m}")

if __name__ == "__main__":
    asyncio.run(backfill_eurusd_sept8())