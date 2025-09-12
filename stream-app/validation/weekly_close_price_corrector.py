#!/usr/bin/env python3
"""
Weekly Close Price Corrector - Comprehensive 1-week historical data correction
Fixes corrupted streaming data with accurate IG API prices for the past week
"""

import asyncio
import httpx
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from services.db import SessionLocal
from services.models import IGCandle
from igstream.ig_auth_prod import ig_login
from services.keyvault import get_secret
from sqlalchemy import and_

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weekly_correction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyClosePriceCorrector:
    """Comprehensive close price correction for 1-week historical data"""
    
    def __init__(self, dry_run: bool = True):
        self.headers = {}
        self.api_base = "https://api.ig.com/gateway/deal"
        self.dry_run = dry_run
        self.correction_stats = {
            'total_checked': 0,
            'corrections_needed': 0,
            'corrections_applied': 0,
            'api_calls_made': 0,
            'errors': 0
        }
        
        # All forex pairs from config (prioritized by importance)
        self.forex_pairs = [
            # Major pairs (highest priority)
            'CS.D.EURUSD.CEEM.IP',
            'CS.D.GBPUSD.MINI.IP', 
            'CS.D.USDJPY.MINI.IP',
            'CS.D.AUDUSD.MINI.IP',
            # Medium priority
            'CS.D.USDCHF.MINI.IP',
            'CS.D.USDCAD.MINI.IP',
            'CS.D.NZDUSD.MINI.IP',
            # JPY crosses (lower priority due to data limits)
            'CS.D.EURJPY.MINI.IP',
            #'CS.D.GBPJPY.MINI.IP',
            'CS.D.AUDJPY.MINI.IP'
            #'CS.D.CADJPY.MINI.IP',
            #'CS.D.CHFJPY.MINI.IP',
            #'CS.D.NZDJPY.MINI.IP'
        ]
        
        # Target timeframes (prioritized: 5m most important for trading)
        self.timeframes = [5, 15, 60]
        
        # Rate limiting for IG API limits
        self.last_api_call = 0
        self.api_call_delay = 2.0  # 30 requests/minute = 2 seconds between calls
        self.batch_size = 1000  # Larger batches for better coverage (IG API supports up to ~2000)
        
        # API limits tracking
        self.weekly_data_points_used = 0
        self.max_weekly_data_points = 9000  # Conservative limit (10k with buffer)
        
    async def authenticate(self) -> bool:
        """Authenticate with IG API"""
        try:
            api_key = get_secret("prodapikey")
            password = get_secret("prodpwd") 
            username = "rehnmarkh"
            
            auth_result = await ig_login(api_key, password, username)
            self.headers = {
                "CST": auth_result["CST"],
                "X-SECURITY-TOKEN": auth_result["X-SECURITY-TOKEN"],
                "VERSION": "3",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-IG-API-KEY": api_key
            }
            logger.info("✅ Authenticated with IG API")
            return True
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False
    
    async def rate_limit_delay(self):
        """Implement rate limiting between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.api_call_delay:
            await asyncio.sleep(self.api_call_delay - time_since_last_call)
        self.last_api_call = time.time()
    
    def get_resolution_string(self, timeframe_minutes: int) -> str:
        """Convert timeframe minutes to IG API resolution string"""
        resolution_map = {
            5: "MINUTE_5",
            15: "MINUTE_15", 
            60: "HOUR"
        }
        return resolution_map.get(timeframe_minutes, "MINUTE_5")
    
    def extract_close_price(self, candle_data: Dict) -> Optional[float]:
        """Extract the most accurate close price from IG API response"""
        try:
            if "closePrice" in candle_data:
                close_data = candle_data["closePrice"]
                
                # If it has bid/ask, calculate mid
                if isinstance(close_data, dict) and "bid" in close_data and "ask" in close_data:
                    return (close_data["bid"] + close_data["ask"]) / 2
                
                # If it's just a number
                elif isinstance(close_data, (int, float)):
                    return float(close_data)
            
            # Fallback - try lastTradedPrice
            if "lastTradedPrice" in candle_data:
                return float(candle_data["lastTradedPrice"])
            
            # Last resort - calculate from bid/ask if available
            if "bid" in candle_data and "ask" in candle_data:
                return (candle_data["bid"] + candle_data["ask"]) / 2
            
            return None
            
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Could not extract close price: {e}")
            return None
    
    def extract_ohlc_prices(self, candle_data: Dict) -> Optional[Dict[str, float]]:
        """Extract all OHLC prices from IG API response"""
        try:
            prices = {}
            
            for field in ['openPrice', 'highPrice', 'lowPrice', 'closePrice']:
                if field in candle_data:
                    price_data = candle_data[field]
                    if isinstance(price_data, dict) and "bid" in price_data and "ask" in price_data:
                        prices[field.replace('Price', '').lower()] = (price_data["bid"] + price_data["ask"]) / 2
                    elif isinstance(price_data, (int, float)):
                        prices[field.replace('Price', '').lower()] = float(price_data)
            
            return prices if len(prices) == 4 else None
            
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Could not extract OHLC prices: {e}")
            return None
    
    async def get_corrupted_entries(self, epic: str, timeframe: int, start_date: datetime, end_date: datetime) -> List[IGCandle]:
        """Get corrupted database entries for correction"""
        with SessionLocal() as session:
            corrupted_entries = session.query(IGCandle).filter(
                and_(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.start_time >= start_date,
                    IGCandle.start_time <= end_date,
                    IGCandle.data_source == 'chart_streamer'  # Only corrupted streaming data
                )
            ).order_by(IGCandle.start_time).all()
            
            # Convert to list to avoid session issues
            return [(entry.start_time, entry.close, entry) for entry in corrupted_entries]
    
    async def fetch_api_data(self, epic: str, timeframe: int, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Fetch accurate price data from IG API with smart batching"""
        await self.rate_limit_delay()
        
        # Estimate data points for this request
        hours_requested = (end_time - start_time).total_seconds() / 3600
        if timeframe == 5:
            estimated_points = int(hours_requested * 12)  # 12 candles per hour
        elif timeframe == 15:
            estimated_points = int(hours_requested * 4)   # 4 candles per hour  
        else:  # 60m
            estimated_points = int(hours_requested * 1)   # 1 candle per hour
        
        # Check if we would exceed the weekly limit
        if self.weekly_data_points_used + estimated_points > self.max_weekly_data_points:
            remaining_points = self.max_weekly_data_points - self.weekly_data_points_used
            if remaining_points < 100:  # Not enough points left for meaningful work
                logger.warning(f"⚠️ Would exceed weekly data limit ({self.weekly_data_points_used + estimated_points}/10k). Skipping {epic} {timeframe}m")
                return []
            else:
                # Reduce batch size to fit remaining quota
                old_batch = self.batch_size
                self.batch_size = min(self.batch_size, remaining_points)
                logger.info(f"🔄 Reducing batch size from {old_batch} to {self.batch_size} to stay within limits")
        
        try:
            async with httpx.AsyncClient(base_url=self.api_base, headers=self.headers, timeout=30.0) as client:
                resolution = self.get_resolution_string(timeframe)
                
                url = f"/prices/{epic}"
                params = {
                    "resolution": resolution,
                    "from": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "to": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "max": self.batch_size  # Request up to 1000 candles per API call
                }
                
                logger.debug(f"API Request: {url} with params: {params}")
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                api_candles = data.get("prices", [])
                
                # Track data points used
                actual_points = len(api_candles)
                self.weekly_data_points_used += actual_points
                
                self.correction_stats['api_calls_made'] += 1
                logger.info(f"📊 Used {actual_points} data points (total: {self.weekly_data_points_used}/10k)")
                
                return api_candles
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                try:
                    error_data = e.response.json()
                    error_code = error_data.get('errorCode', '')
                    
                    if 'exceeded-account-historical-data-allowance' in error_code:
                        logger.error("🚫 WEEKLY DATA LIMIT EXCEEDED!")
                        logger.error("📊 IG API weekly historical data allowance has been exhausted")
                        logger.error("⏰ Weekly limit resets every Monday at midnight GMT")
                        logger.error("💡 Solutions:")
                        logger.error("   1. Wait until Monday for limit reset")
                        logger.error("   2. Use --conservative mode to reduce API calls")
                        logger.error("   3. Focus on most important pairs only")
                        
                        # Set a flag to stop further API calls
                        self.weekly_data_points_used = self.max_weekly_data_points
                        return "LIMIT_EXCEEDED"
                    else:
                        logger.error(f"❌ 403 Forbidden - {error_code}: {error_data}")
                except:
                    logger.error(f"❌ 403 Forbidden: {e.response.text}")
            else:
                logger.error(f"❌ HTTP {e.response.status_code}: {e.response.text}")
            
            self.correction_stats['errors'] += 1
            return []
            
        except Exception as e:
            logger.error(f"❌ API fetch failed for {epic} {timeframe}m: {e}")
            self.correction_stats['errors'] += 1
            return []
    
    async def correct_period_data(self, epic: str, timeframe: int, start_date: datetime, end_date: datetime) -> int:
        """Correct corrupted data for a specific period using chunked processing"""
        period_days = (end_date - start_date).days
        logger.info(f"🔧 Processing {epic} {timeframe}m from {start_date.date()} to {end_date.date()} ({period_days} days)")
        
        # For long periods, process in daily chunks to handle API limitations
        if period_days > 2:
            return await self.correct_period_chunked(epic, timeframe, start_date, end_date)
        
        # Get corrupted entries from database
        logger.info(f"🗃️ Querying database for corrupted entries...")
        corrupted_entries = await self.get_corrupted_entries(epic, timeframe, start_date, end_date)
        
        if not corrupted_entries:
            logger.info(f"ℹ️ No corrupted data found for {epic} {timeframe}m in this period")
            return 0
        
        logger.info(f"📊 Found {len(corrupted_entries)} corrupted entries to check")
        
        # Fetch API data for the entire period
        logger.info(f"📡 Fetching API data from IG...")
        api_candles = await self.fetch_api_data(epic, timeframe, start_date, end_date)
        
        # Check for limit exceeded
        if api_candles == "LIMIT_EXCEEDED":
            logger.error(f"🚫 Stopping processing due to API data limit exceeded")
            return -1  # Special return code for limit exceeded
        
        logger.info(f"📥 Received {len(api_candles) if api_candles else 0} candles from API")
        
        if not api_candles:
            logger.warning(f"⚠️ No API data available for {epic} {timeframe}m")
            return 0
        
        # Create timestamp lookup for API data
        api_lookup = {}
        for api_candle in api_candles:
            try:
                timestamp_str = api_candle.get("snapshotTime", "")
                timestamp = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
                api_lookup[timestamp] = api_candle
            except ValueError:
                continue
        
        corrections_made = 0
        
        with SessionLocal() as session:
            for db_timestamp, db_close, db_entry in corrupted_entries:
                self.correction_stats['total_checked'] += 1
                
                if db_timestamp in api_lookup:
                    api_candle = api_lookup[db_timestamp]
                    
                    # Get correct prices from API
                    correct_prices = self.extract_ohlc_prices(api_candle)
                    correct_close = self.extract_close_price(api_candle)
                    
                    if correct_close is not None:
                        # Calculate pip difference
                        pip_multiplier = 10000 if 'JPY' not in epic else 100
                        difference_pips = abs(db_close - correct_close) * pip_multiplier
                        
                        # Correct any difference (all corrupted data should be fixed)
                        if difference_pips > 0.1:
                            self.correction_stats['corrections_needed'] += 1
                            
                            logger.info(f"🎯 {db_timestamp}: {db_close:.5f} -> {correct_close:.5f} ({difference_pips:.1f} pip correction)")
                            
                            if not self.dry_run:
                                # Apply correction
                                db_entry_actual = session.query(IGCandle).filter(
                                    and_(
                                        IGCandle.epic == epic,
                                        IGCandle.timeframe == timeframe,
                                        IGCandle.start_time == db_timestamp
                                    )
                                ).first()
                                
                                if db_entry_actual:
                                    # Update close price (most critical)
                                    db_entry_actual.close = correct_close
                                    
                                    # Update other OHLC if available
                                    if correct_prices:
                                        db_entry_actual.open = correct_prices.get('open', db_entry_actual.open)
                                        db_entry_actual.high = correct_prices.get('high', db_entry_actual.high)
                                        db_entry_actual.low = correct_prices.get('low', db_entry_actual.low)
                                    
                                    # Mark as corrected with api_backfill_fixed
                                    db_entry_actual.data_source = "api_backfill_fixed"
                                    db_entry_actual.updated_at = datetime.now()
                                    
                                    corrections_made += 1
                                    self.correction_stats['corrections_applied'] += 1
            
            if not self.dry_run:
                session.commit()
                logger.info(f"✅ Committed {corrections_made} corrections for {epic} {timeframe}m")
            else:
                logger.info(f"🔍 DRY RUN: Would correct {corrections_made} entries for {epic} {timeframe}m")
        
        return corrections_made
    
    async def correct_period_chunked(self, epic: str, timeframe: int, start_date: datetime, end_date: datetime) -> int:
        """Process a long period in daily chunks to handle API limitations"""
        logger.info(f"📅 Using chunked processing for better API coverage")
        
        total_corrections = 0
        current_date = start_date
        chunk_hours = 2  # Process 2-hour chunks to work within IG API's ~20 candle limit
        
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(hours=chunk_hours), end_date)
            
            logger.info(f"  📊 Processing chunk: {current_date.strftime('%m-%d %H:%M')} to {chunk_end.strftime('%m-%d %H:%M')}")
            
            # Get corrupted entries for this chunk
            corrupted_entries = await self.get_corrupted_entries(epic, timeframe, current_date, chunk_end)
            
            if corrupted_entries:
                logger.info(f"  🗃️ Found {len(corrupted_entries)} entries in this chunk")
                
                # Fetch API data for this smaller period
                api_candles = await self.fetch_api_data(epic, timeframe, current_date, chunk_end)
                
                # Check for limit exceeded
                if api_candles == "LIMIT_EXCEEDED":
                    logger.error(f"🚫 Stopping chunked processing due to API data limit exceeded")
                    return -1  # Special return code for limit exceeded
                
                if api_candles:
                    logger.info(f"  📥 Received {len(api_candles)} candles from API")
                    
                    # Check if we're getting incomplete coverage
                    expected_candles = len(corrupted_entries)
                    coverage_ratio = len(api_candles) / expected_candles if expected_candles > 0 else 1
                    
                    if coverage_ratio < 0.8 and expected_candles > 25:
                        logger.warning(f"  ⚠️ Low coverage: {len(api_candles)}/{expected_candles} candles ({coverage_ratio:.1%})")
                        logger.info(f"  💡 Consider using smaller chunks if many entries remain unprocessed")
                    
                    # Process corrections for this chunk
                    chunk_corrections = await self.process_chunk_corrections(
                        epic, timeframe, corrupted_entries, api_candles
                    )
                    
                    total_corrections += chunk_corrections
                    logger.info(f"  ✅ Applied {chunk_corrections} corrections in this chunk")
                else:
                    logger.warning(f"  ⚠️ No API data for chunk")
            else:
                logger.info(f"  ℹ️ No corrupted entries in this chunk")
            
            # Move to next chunk
            current_date = chunk_end
            
            # Skip weekend market closure (Friday 21:00 UTC to Sunday 21:00 UTC)
            if current_date.weekday() == 4 and current_date.hour >= 21:  # Friday after 21:00
                # Skip to Sunday 21:00
                days_to_sunday = (6 - current_date.weekday()) % 7
                if days_to_sunday == 0:  # Already Friday, go to next Sunday
                    days_to_sunday = 2
                current_date = current_date.replace(hour=21, minute=0, second=0, microsecond=0) + timedelta(days=days_to_sunday)
                logger.info(f"  ⏭️ Skipping weekend closure, resuming at {current_date}")
            
            # Brief pause between chunks
            await asyncio.sleep(self.api_call_delay)
        
        logger.info(f"📊 Chunked processing complete: {total_corrections} total corrections")
        return total_corrections
    
    async def process_chunk_corrections(self, epic: str, timeframe: int, corrupted_entries: List, api_candles: List[Dict]) -> int:
        """Process corrections for a single chunk of data"""
        # Create timestamp lookup for API data
        api_lookup = {}
        for api_candle in api_candles:
            try:
                timestamp_str = api_candle.get("snapshotTime", "")
                timestamp = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
                api_lookup[timestamp] = api_candle
            except ValueError:
                continue
        
        corrections_made = 0
        
        # Use a fresh session for updates
        with SessionLocal() as session:
            for db_timestamp, db_close, _ in corrupted_entries:  # We don't need the db_entry from the old session
                self.correction_stats['total_checked'] += 1
                
                if db_timestamp in api_lookup:
                    api_candle = api_lookup[db_timestamp]
                    
                    # Get correct prices from API
                    correct_prices = self.extract_ohlc_prices(api_candle)
                    correct_close = self.extract_close_price(api_candle)
                    
                    if correct_close is not None:
                        # Calculate pip difference
                        pip_multiplier = 10000 if 'JPY' not in epic else 100
                        difference_pips = abs(db_close - correct_close) * pip_multiplier
                        
                        # Correct any difference (all corrupted data should be fixed)
                        if difference_pips > 0.1:
                            self.correction_stats['corrections_needed'] += 1
                            
                            logger.info(f"    🎯 {db_timestamp.strftime('%m-%d %H:%M')}: {db_close:.5f} -> {correct_close:.5f} ({difference_pips:.1f} pip correction)")
                            
                            if not self.dry_run:
                                # Apply correction using update query for better reliability
                                update_data = {
                                    'close': correct_close,
                                    'data_source': 'api_backfill_fixed',
                                    'updated_at': datetime.now()
                                }
                                
                                # Add other OHLC if available
                                if correct_prices:
                                    if correct_prices.get('open'):
                                        update_data['open'] = correct_prices['open']
                                    if correct_prices.get('high'):
                                        update_data['high'] = correct_prices['high']
                                    if correct_prices.get('low'):
                                        update_data['low'] = correct_prices['low']
                                
                                rows_updated = session.query(IGCandle).filter(
                                    and_(
                                        IGCandle.epic == epic,
                                        IGCandle.timeframe == timeframe,
                                        IGCandle.start_time == db_timestamp
                                    )
                                ).update(update_data)
                                
                                if rows_updated > 0:
                                    corrections_made += 1
                                    self.correction_stats['corrections_applied'] += 1
                                    logger.debug(f"      ✅ Updated database entry for {db_timestamp}")
                                else:
                                    logger.warning(f"      ❌ Failed to update {db_timestamp} - entry not found")
            
            if not self.dry_run and corrections_made > 0:
                try:
                    session.commit()
                    logger.info(f"  ✅ Committed {corrections_made} corrections for chunk")
                    
                    # Verify the commit worked by checking one updated entry
                    if corrections_made > 0:
                        sample_check = session.query(IGCandle).filter(
                            IGCandle.data_source == 'api_backfill_fixed'
                        ).first()
                        if sample_check:
                            logger.debug(f"  ✅ Verified: Found updated entry with api_backfill_fixed")
                        else:
                            logger.warning(f"  ⚠️ Warning: No entries found with api_backfill_fixed after commit")
                    
                except Exception as e:
                    logger.error(f"  ❌ Commit failed: {e}")
                    session.rollback()
        
        return corrections_made
    
    async def correct_weekly_data(self, target_date: datetime = None, weeks_back: int = 1) -> Dict:
        """Correct multiple weeks of data ending at target_date (default: today 20:00 UTC)"""
        if target_date is None:
            # Default to today 20:00 UTC
            today = datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)
            target_date = today
        
        # Calculate weeks back from target_date
        start_date = target_date - timedelta(weeks=weeks_back)
        
        logger.info(f"🚀 Starting {weeks_back}-week correction from {start_date} to {target_date}")
        logger.info(f"📋 Mode: {'DRY RUN' if self.dry_run else 'LIVE CORRECTION'}")
        logger.info(f"📊 Processing {len(self.forex_pairs)} pairs × {len(self.timeframes)} timeframes")
        
        start_time = time.time()
        total_corrections = 0
        
        for epic_idx, epic in enumerate(self.forex_pairs, 1):
            logger.info(f"\n🔄 Processing {epic} ({epic_idx}/{len(self.forex_pairs)})")
            
            for tf_idx, timeframe in enumerate(self.timeframes, 1):
                logger.info(f"  ⏱️ Timeframe {timeframe}m ({tf_idx}/{len(self.timeframes)})")
                try:
                    corrections = await self.correct_period_data(epic, timeframe, start_date, target_date)
                    total_corrections += corrections
                    
                    # Summary for this timeframe
                    if corrections > 0:
                        logger.info(f"    ✅ {corrections} corrections {'applied' if not self.dry_run else 'identified'}")
                    else:
                        logger.info(f"    ℹ️ No corrections needed")
                    
                    # Brief pause between timeframes
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {epic} {timeframe}m: {e}")
                    self.correction_stats['errors'] += 1
                    continue
            
            # Pause between pairs to manage API rate limits
            await asyncio.sleep(1.0)
        
        execution_time = time.time() - start_time
        
        # Generate final report
        report = {
            'execution_time': execution_time,
            'total_corrections': total_corrections,
            'stats': self.correction_stats,
            'period': f"{start_date.date()} to {target_date.date()}",
            'mode': 'DRY_RUN' if self.dry_run else 'LIVE'
        }
        
        self.print_final_report(report)
        return report
    
    def print_final_report(self, report: Dict):
        """Print comprehensive correction report"""
        logger.info("\n" + "="*80)
        logger.info("📊 WEEKLY CLOSE PRICE CORRECTION REPORT")
        logger.info("="*80)
        logger.info(f"⏰ Period: {report['period']}")
        logger.info(f"🔧 Mode: {report['mode']}")
        logger.info(f"⏱️ Execution Time: {report['execution_time']:.2f} seconds")
        logger.info(f"📈 Total Entries Checked: {report['stats']['total_checked']}")
        logger.info(f"🎯 Corrections Needed: {report['stats']['corrections_needed']}")
        logger.info(f"✅ Corrections Applied: {report['stats']['corrections_applied']}")
        logger.info(f"🌐 API Calls Made: {report['stats']['api_calls_made']}")
        logger.info(f"❌ Errors Encountered: {report['stats']['errors']}")
        
        if report['mode'] == 'DRY_RUN':
            logger.info("\n💡 This was a DRY RUN - no data was modified")
            logger.info("   Run with dry_run=False to apply corrections")
        else:
            logger.info(f"\n🎉 Successfully corrected {report['stats']['corrections_applied']} entries")
            logger.info("   All corrected entries marked with data_source='api_backfill_fixed'")
        
        logger.info("="*80)

async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Correct multiple weeks of corrupted close price data')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Run in dry-run mode (default: True)')
    parser.add_argument('--live', action='store_true', help='Run in live mode (applies corrections)')
    parser.add_argument('--epic', type=str, help='Process single epic only (for testing)')
    parser.add_argument('--timeframe', type=int, choices=[5, 15, 60], help='Process single timeframe only (for testing)')
    parser.add_argument('--weeks', type=int, default=1, help='Number of weeks to go back (default: 1)')
    parser.add_argument('--priority-only', action='store_true', help='Process only major pairs (EURUSD, GBPUSD, USDJPY, AUDUSD)')
    parser.add_argument('--timeframe-only', type=int, choices=[5, 15, 60], help='Process only specified timeframe to save API limits')
    
    args = parser.parse_args()
    
    # Determine run mode
    dry_run = not args.live if args.live else args.dry_run
    
    corrector = WeeklyClosePriceCorrector(dry_run=dry_run)
    
    # Override for single epic/timeframe testing
    if args.epic:
        corrector.forex_pairs = [args.epic]
        logger.info(f"🧪 Test mode: Processing only {args.epic}")
    elif args.priority_only:
        corrector.forex_pairs = corrector.forex_pairs[:4]  # First 4 major pairs
        logger.info(f"🎯 Priority mode: Processing only major pairs {corrector.forex_pairs}")
    
    if args.timeframe:
        corrector.timeframes = [args.timeframe]
        logger.info(f"🧪 Test mode: Processing only {args.timeframe}m timeframe")
    elif args.timeframe_only:
        corrector.timeframes = [args.timeframe_only]
        logger.info(f"⏱️ Single timeframe mode: Processing only {args.timeframe_only}m to conserve API limits")
    
    if not await corrector.authenticate():
        logger.error("❌ Authentication failed - cannot proceed")
        return
    
    # Run the correction
    report = await corrector.correct_weekly_data(weeks_back=args.weeks)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"correction_report_{timestamp}.log"
    with open(report_file, 'w') as f:
        f.write(f"Weekly Close Price Correction Report\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Period: {report['period']}\n")
        f.write(f"Mode: {report['mode']}\n\n")
        f.write(f"Results:\n")
        for key, value in report['stats'].items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())