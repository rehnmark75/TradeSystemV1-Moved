#!/usr/bin/env python3
"""
Selective backfill corrector to replace corrupted streaming data with correct API data
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from services.db import SessionLocal
from services.models import IGCandle
from igstream.ig_auth_prod import ig_login
from services.keyvault import get_secret

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "https://api.ig.com/gateway/deal"
RATE_LIMIT_DELAY = 2  # Seconds between API requests
EPICS_TO_FIX = [
    "CS.D.EURUSD.CEEM.IP",
    "CS.D.GBPUSD.MINI.IP", 
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP"
]
TIMEFRAMES_TO_FIX = [5, 15, 60]  # Focus on most important timeframes

class SelectiveBackfillCorrector:
    """Corrects corrupted historical data with accurate API data"""
    
    def __init__(self):
        self.headers = {}
        self.stats = {
            "candles_analyzed": 0,
            "candles_replaced": 0,
            "api_requests": 0,
            "errors": 0
        }
    
    async def authenticate(self):
        """Authenticate with IG Markets API"""
        try:
            api_key = get_secret("demoapikey")
            password = get_secret("demopwd")
            username = "rehnmarkhdemo"
            
            auth_result = await ig_login(api_key, password, username)
            self.headers = {
                "CST": auth_result["CST"],
                "X-SECURITY-TOKEN": auth_result["X-SECURITY-TOKEN"],
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            logger.info("✅ Authenticated with IG Markets API")
            return True
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False
    
    def _timeframe_to_resolution(self, timeframe: int) -> str:
        """Convert timeframe minutes to IG API resolution"""
        mapping = {
            5: "MINUTE_5",
            15: "MINUTE_15", 
            60: "HOUR"
        }
        return mapping.get(timeframe, "MINUTE_5")
    
    def parse_snapshot_time(self, snapshot_time: str) -> datetime:
        """Parse IG API snapshot time format"""
        # Format: "2025/09/02 17:55:00"
        return datetime.strptime(snapshot_time, "%Y/%m/%d %H:%M:%S")
    
    async def get_corrupted_periods(self, epic: str, timeframe: int, days_back: int = 7) -> List[Dict]:
        """Identify periods with corrupted data (from chart_streamer before fix)"""
        periods = []
        
        with SessionLocal() as session:
            # Get corrupted data from the last N days (before streaming fix)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            fix_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)  # Today 00:00
            
            # Find candles from chart_streamer source that are likely corrupted
            corrupted_candles = session.query(IGCandle).filter(
                IGCandle.epic == epic,
                IGCandle.timeframe == timeframe,
                IGCandle.start_time >= cutoff_date,
                IGCandle.start_time < fix_date,  # Only data before today's fix
                IGCandle.data_source == 'chart_streamer'
            ).order_by(IGCandle.start_time).all()
            
            if corrupted_candles:
                # Group into continuous periods for efficient API calls
                current_period = {"start": corrupted_candles[0].start_time, "candles": []}
                
                for candle in corrupted_candles:
                    # If there's a gap > 2 hours, start a new period
                    if current_period["candles"] and (candle.start_time - current_period["candles"][-1].start_time).total_seconds() > 7200:
                        current_period["end"] = current_period["candles"][-1].start_time
                        periods.append(current_period)
                        current_period = {"start": candle.start_time, "candles": []}
                    
                    current_period["candles"].append(candle)
                
                # Add final period
                if current_period["candles"]:
                    current_period["end"] = current_period["candles"][-1].start_time
                    periods.append(current_period)
        
        logger.info(f"Found {len(periods)} corrupted periods for {epic} {timeframe}m ({len(corrupted_candles)} candles total)")
        return periods
    
    async def fix_period(self, epic: str, timeframe: int, period: Dict) -> bool:
        """Fix a specific period of corrupted data"""
        resolution = self._timeframe_to_resolution(timeframe)
        
        # Add buffer to ensure we get all data
        from_dt = period["start"] - timedelta(minutes=timeframe)
        to_dt = period["end"] + timedelta(minutes=timeframe)
        
        logger.info(f"Fixing {epic} {timeframe}m from {period['start']} to {period['end']} ({len(period['candles'])} candles)")
        
        try:
            async with httpx.AsyncClient(base_url=API_BASE_URL, headers=self.headers) as client:
                url = f"/prices/{epic}"
                
                # Format dates for IG API
                from_str = from_dt.strftime("%Y-%m-%dT%H:%M:%S")
                to_str = to_dt.strftime("%Y-%m-%dT%H:%M:%S")
                
                params = {
                    "resolution": resolution,
                    "from": from_str,
                    "to": to_str,
                    "max": 1000
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                self.stats["api_requests"] += 1
                
                data = response.json()
                api_candles = data.get("prices", [])
                
                if not api_candles:
                    logger.warning(f"No API data returned for {epic} {timeframe}m period")
                    return False
                
                # Replace corrupted candles with correct API data
                replaced_count = 0
                with SessionLocal() as session:
                    for api_candle in api_candles:
                        timestamp = self.parse_snapshot_time(api_candle["snapshotTime"])
                        
                        # Check if this timestamp exists in our corrupted data
                        existing = session.query(IGCandle).filter(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time == timestamp
                        ).first()
                        
                        if existing and existing.data_source == 'chart_streamer':
                            # Replace with correct API data
                            try:
                                existing.open = api_candle["openPrice"]["mid"]
                                existing.high = api_candle["highPrice"]["mid"]
                                existing.low = api_candle["lowPrice"]["mid"]
                                existing.close = api_candle["closePrice"]["mid"]
                                existing.volume = api_candle.get("lastTradedVolume", existing.volume)
                                existing.ltv = api_candle.get("lastTradedVolume", existing.ltv)
                                existing.data_source = "api_backfill_corrected"
                                existing.updated_at = datetime.now()
                                
                                replaced_count += 1
                                self.stats["candles_replaced"] += 1
                                
                            except (KeyError, TypeError) as e:
                                logger.warning(f"Skipping incomplete API candle at {timestamp}: {e}")
                                self.stats["errors"] += 1
                                continue
                    
                    session.commit()
                
                logger.info(f"✅ Fixed {replaced_count} candles for {epic} {timeframe}m")
                await asyncio.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                return replaced_count > 0
                
        except Exception as e:
            logger.error(f"❌ Error fixing period for {epic} {timeframe}m: {e}")
            self.stats["errors"] += 1
            return False
    
    async def fix_epic_timeframe(self, epic: str, timeframe: int, days_back: int = 7):
        """Fix all corrupted data for a specific epic and timeframe"""
        logger.info(f"🔧 Starting correction for {epic} {timeframe}m (last {days_back} days)")
        
        periods = await self.get_corrupted_periods(epic, timeframe, days_back)
        self.stats["candles_analyzed"] += sum(len(p["candles"]) for p in periods)
        
        if not periods:
            logger.info(f"✅ No corrupted data found for {epic} {timeframe}m")
            return
        
        success_count = 0
        for period in periods:
            if await self.fix_period(epic, timeframe, period):
                success_count += 1
        
        logger.info(f"✅ Fixed {success_count}/{len(periods)} periods for {epic} {timeframe}m")
    
    async def run_correction(self, days_back: int = 7):
        """Run the selective correction process"""
        logger.info(f"🚀 Starting selective backfill correction (last {days_back} days)")
        
        if not await self.authenticate():
            return
        
        start_time = datetime.now()
        
        for epic in EPICS_TO_FIX:
            for timeframe in TIMEFRAMES_TO_FIX:
                await self.fix_epic_timeframe(epic, timeframe, days_back)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("SELECTIVE BACKFILL CORRECTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Candles analyzed: {self.stats['candles_analyzed']}")
        logger.info(f"Candles replaced: {self.stats['candles_replaced']}")
        logger.info(f"API requests: {self.stats['api_requests']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        if self.stats["candles_replaced"] > 0:
            success_rate = (self.stats["candles_replaced"] / self.stats["candles_analyzed"]) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
            logger.info("✅ Historical data corruption has been corrected!")
        else:
            logger.info("ℹ️ No corrupted data found to correct")

async def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    corrector = SelectiveBackfillCorrector()
    
    # Start with 2 days for safety
    print("🔧 Correcting last 2 days of corrupted historical data...")
    print("⚠️ This will replace chart_streamer data with accurate API data")
    
    await corrector.run_correction(days_back=2)

if __name__ == "__main__":
    asyncio.run(main())