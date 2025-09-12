#!/usr/bin/env python3
"""
Targeted backfill for EURUSD data on September 8, 2025 18:35-20:55 UTC
This fills a specific gap in the data
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

async def backfill_specific_range():
    """Backfill EURUSD data for Sept 8 18:35-20:55 UTC"""
    
    epic = "CS.D.EURUSD.CEEM.IP"
    
    # Define the specific time range
    start_time = datetime(2025, 9, 8, 18, 35, 0)
    end_time = datetime(2025, 9, 8, 20, 55, 0)
    
    logger.info(f"🎯 Targeted backfill for EURUSD")
    logger.info(f"📅 Period: {start_time} to {end_time}")
    logger.info(f"⏱️ Duration: {(end_time - start_time).total_seconds() / 60:.0f} minutes")
    
    # Check what's currently in the database for this period
    with SessionLocal() as session:
        existing_count = session.query(IGCandle).filter(
            IGCandle.epic == epic,
            IGCandle.timeframe == 5,
            IGCandle.start_time >= start_time,
            IGCandle.start_time <= end_time
        ).count()
        
        logger.info(f"📊 Currently have {existing_count} candles in this period")
        expected_candles = int((end_time - start_time).total_seconds() / 60 / 5)
        logger.info(f"📊 Expected candles: {expected_candles}")
        logger.info(f"📊 Missing candles: {expected_candles - existing_count}")
    
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
    
    # Process each timeframe
    timeframes = [
        ("MINUTE_5", 5),
        ("MINUTE_15", 15),
        ("HOUR", 60)
    ]
    
    for resolution, tf_minutes in timeframes:
        logger.info(f"\n📊 Processing {resolution} timeframe...")
        
        # For this specific range, we'll process in smaller chunks to work around the 20-candle limit
        if tf_minutes == 5:
            # Process in 30-minute chunks (6 candles each)
            chunk_minutes = 30
        elif tf_minutes == 15:
            # Process in 1-hour chunks (4 candles each)
            chunk_minutes = 60
        else:  # 60 min
            # Process the whole range (only 2-3 candles)
            chunk_minutes = 180
        
        current_start = start_time
        total_saved = 0
        
        while current_start < end_time:
            current_end = min(current_start + timedelta(minutes=chunk_minutes), end_time)
            
            logger.info(f"  Fetching {current_start.strftime('%H:%M')} to {current_end.strftime('%H:%M')}")
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    url = f"https://api.ig.com/gateway/deal/prices/{epic}"
                    params = {
                        "resolution": resolution,
                        "from": current_start.strftime("%Y-%m-%dT%H:%M:%S"),
                        "to": current_end.strftime("%Y-%m-%dT%H:%M:%S"),
                        "max": 100  # Small request to avoid hitting limits
                    }
                    
                    response = await client.get(url, params=params, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    candles = data.get("prices", [])
                    
                    logger.info(f"    Received {len(candles)} candles from API")
                    
                    if candles:
                        saved_count = 0
                        updated_count = 0
                        
                        with SessionLocal() as session:
                            for candle_data in candles:
                                try:
                                    # Parse timestamp
                                    ts_str = candle_data.get("snapshotTime", "")
                                    ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
                                    
                                    # Check if already exists
                                    existing = session.query(IGCandle).filter_by(
                                        epic=epic,
                                        timeframe=tf_minutes,
                                        start_time=ts
                                    ).first()
                                    
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
                                    
                                    # Apply EURUSD scaling correction (divide by 10000)
                                    mid_open = mid_open / 10000
                                    mid_high = mid_high / 10000
                                    mid_low = mid_low / 10000
                                    mid_close = mid_close / 10000
                                    
                                    if existing:
                                        # Update existing record if data_source is not from API
                                        if existing.data_source != 'api_backfill_fixed' and existing.data_source != 'manual_backfill_sept8':
                                            existing.open = mid_open
                                            existing.high = mid_high
                                            existing.low = mid_low
                                            existing.close = mid_close
                                            existing.volume = candle_data.get("lastTradedVolume", 0)
                                            existing.ltv = candle_data.get("lastTradedVolume")
                                            existing.data_source = "targeted_backfill"
                                            existing.updated_at = datetime.utcnow()
                                            updated_count += 1
                                            logger.debug(f"      Updated {ts}")
                                    else:
                                        # Create new record
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
                                            data_source="targeted_backfill"
                                        )
                                        session.add(candle)
                                        saved_count += 1
                                        logger.debug(f"      Added {ts}")
                                
                                except Exception as e:
                                    logger.warning(f"      Error processing candle: {e}")
                                    continue
                            
                            session.commit()
                            total_saved += saved_count
                            
                            if saved_count > 0 or updated_count > 0:
                                logger.info(f"    ✅ Saved {saved_count} new, updated {updated_count} existing candles")
                    else:
                        logger.info(f"    ℹ️ No data returned for this period")
                    
            except Exception as e:
                if "exceeded-account-historical-data-allowance" in str(e):
                    logger.error(f"    ❌ API weekly limit exceeded!")
                    return
                else:
                    logger.error(f"    ❌ Error fetching data: {e}")
            
            # Rate limiting
            await asyncio.sleep(2)
            
            # Move to next chunk
            current_start = current_end
        
        logger.info(f"  📊 Total saved for {resolution}: {total_saved} candles")
    
    # Final verification
    logger.info("\n📊 FINAL VERIFICATION:")
    with SessionLocal() as session:
        # Check 5-minute data completeness
        final_count = session.query(IGCandle).filter(
            IGCandle.epic == epic,
            IGCandle.timeframe == 5,
            IGCandle.start_time >= start_time,
            IGCandle.start_time <= end_time
        ).count()
        
        expected = int((end_time - start_time).total_seconds() / 60 / 5)
        
        logger.info(f"  5m timeframe: {final_count}/{expected} candles")
        logger.info(f"  Completeness: {final_count/expected*100:.1f}%")
        
        if final_count < expected:
            # Show what's still missing
            all_times = set()
            current = start_time
            while current <= end_time:
                all_times.add(current)
                current += timedelta(minutes=5)
            
            existing_times = set()
            for row in session.query(IGCandle.start_time).filter(
                IGCandle.epic == epic,
                IGCandle.timeframe == 5,
                IGCandle.start_time >= start_time,
                IGCandle.start_time <= end_time
            ).all():
                existing_times.add(row[0])
            
            missing_times = sorted(all_times - existing_times)
            if missing_times:
                logger.info(f"\n  ⚠️ Still missing {len(missing_times)} candles:")
                for i, missing_time in enumerate(missing_times[:10]):  # Show first 10
                    logger.info(f"    - {missing_time}")
                if len(missing_times) > 10:
                    logger.info(f"    ... and {len(missing_times) - 10} more")
        else:
            logger.info("  ✅ Period is complete!")

if __name__ == "__main__":
    asyncio.run(backfill_specific_range())