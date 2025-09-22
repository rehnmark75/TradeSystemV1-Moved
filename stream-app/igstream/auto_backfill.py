"""
Automated Backfill Service for IG Candle Data
Automatically detects and fills gaps in candle data
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from services.db import SessionLocal
from services.models import IGCandle, FailedGap
from igstream.gap_detector import GapDetector
from igstream.ig_auth_prod import ig_login
from services.keyvault import get_secret
from services.operation_tracker import track_gap_detection, track_gap_fill, track_auth_refresh
from config import IG_API_BASE_URL

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = IG_API_BASE_URL
MAX_CANDLES_PER_REQUEST = 18  # Realistic IG API limit (was 1000, but IG only returns ~20 max)
BACKFILL_RATE_LIMIT_DELAY = 2  # Seconds between API requests
MAX_BACKFILL_ATTEMPTS = 3
BACKFILL_BATCH_SIZE = 20  # Number of gaps to process in one run
MAX_CHUNK_SIZE = 18  # Maximum candles per chunk for large gap handling (match API limit)
EPIC_CHANGE_MAX_GAP_HOURS = 72  # Allow larger gaps for epic changes (was 48)

class AutoBackfillService:
    """Automated backfill service that detects and fills data gaps"""
    
    def market_is_open(self) -> bool:
        """Check if forex market is open"""
        now = datetime.now(timezone.utc)
        # IG closes Friday 20:30 UTC and reopens Sunday 22:00 UTC (extended for data stability)
        if now.weekday() == 5:  # Saturday
            return False
        if now.weekday() == 6 and now.hour < 22:  # Sunday before 22:00 UTC (extended for market reopening buffer)
            return False
        if now.weekday() == 4 and (now.hour >= 21 or (now.hour == 20 and now.minute >= 30)):  # Friday after 20:30 UTC
            return False
        return True
    
    def __init__(self, epics: List[str]):
        """
        Initialize the auto-backfill service
        
        Args:
            epics: List of trading pairs to monitor and backfill
        """
        self.epics = epics
        self.gap_detector = GapDetector(max_gap_hours=EPIC_CHANGE_MAX_GAP_HOURS)  # Increased to handle weekend gaps and epic changes
        self.headers = {}
        self.is_running = False
        self.last_backfill_time = {}
        self.backfill_stats = {
            "gaps_detected": 0,
            "gaps_filled": 0,
            "candles_recovered": 0,
            "failures": 0
        }
        
    async def initialize_auth(self):
        """Initialize or refresh IG authentication"""
        try:
            api_key = get_secret("prodapikey")
            ig_pwd = get_secret("prodpwd")
            ig_usr = "rehnmarkh"
            
            auth = await ig_login(api_key, ig_pwd, ig_usr, api_url=API_BASE_URL)
            self.headers = {
                "CST": auth["CST"],
                "X-SECURITY-TOKEN": auth["X-SECURITY-TOKEN"],
                "VERSION": "3",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-IG-API-KEY": api_key
            }
            logger.info("Auto-backfill service authenticated successfully")
            
            # Track successful authentication
            track_auth_refresh(success=True)
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate for auto-backfill: {e}")
            
            # Track failed authentication
            track_auth_refresh(success=False, error_message=str(e))
            return False
    
    def parse_snapshot_time(self, ts_str):
        """Parse IG timestamp format"""
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            return datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")

    def record_failed_gap(self, gap: Dict, failure_reason: str):
        """Record a gap that couldn't be backfilled to avoid repeated attempts"""
        try:
            with SessionLocal() as session:
                now = datetime.now(timezone.utc)

                # Check if this gap already exists
                existing = session.query(FailedGap).filter(
                    FailedGap.epic == gap["epic"],
                    FailedGap.timeframe == gap["timeframe"],
                    FailedGap.gap_start == gap["gap_start"]
                ).first()

                if existing:
                    # Update existing record
                    existing.last_attempted_at = now
                    existing.attempt_count += 1
                    logger.debug(f"Updated failed gap record for {gap['epic']} (attempt {existing.attempt_count})")
                else:
                    # Create new record
                    failed_gap = FailedGap(
                        epic=gap["epic"],
                        timeframe=gap["timeframe"],
                        gap_start=gap["gap_start"],
                        gap_end=gap["gap_end"],
                        failure_reason=failure_reason,
                        first_failed_at=now,
                        last_attempted_at=now,
                        attempt_count=1,
                        missing_candles=gap.get("missing_candles"),
                        gap_duration_minutes=gap.get("gap_duration_minutes")
                    )
                    session.add(failed_gap)
                    logger.info(f"Recorded failed gap for {gap['epic']} {gap['timeframe']}m: {failure_reason}")

                session.commit()
        except Exception as e:
            logger.error(f"Error recording failed gap: {e}")

    def is_gap_known_failed(self, gap: Dict) -> bool:
        """Check if a gap is already recorded as failed and should be skipped"""
        try:
            with SessionLocal() as session:
                existing = session.query(FailedGap).filter(
                    FailedGap.epic == gap["epic"],
                    FailedGap.timeframe == gap["timeframe"],
                    FailedGap.gap_start == gap["gap_start"]
                ).first()

                if existing:
                    # Only skip if it failed recently and with certain reasons
                    days_since_last_attempt = (datetime.now(timezone.utc) - existing.last_attempted_at).days

                    # Skip permanently failed gaps (no data available)
                    if existing.failure_reason == 'no_data_available':
                        return True

                    # Skip recently failed gaps (within 7 days) with high attempt count
                    if existing.attempt_count >= 3 and days_since_last_attempt < 7:
                        return True

                return False
        except Exception as e:
            logger.error(f"Error checking failed gap: {e}")
            return False

    def _timeframe_to_resolution(self, timeframe: int) -> str:
        """Convert timeframe in minutes to IG API resolution string"""
        # Use correct resolution format as per IG API docs
        # NOTE: Chart streaming uses "1MINUTE" but REST API uses "MINUTE"
        mapping = {
            1: "MINUTE",      # REST API format for 1-minute backfill
            5: "MINUTE_5",
            15: "MINUTE_15",
            60: "HOUR"
        }
        return mapping.get(timeframe, "MINUTE_5")
    
    def _split_large_gap(self, gap: Dict) -> List[Dict]:
        """
        Split large gaps into smaller chunks that can be handled by IG API
        Improved for better handling of epic changes and large historical gaps
        
        Args:
            gap: Gap dictionary from GapDetector
            
        Returns:
            List of smaller gap chunks, each within MAX_CHUNK_SIZE limit
        """
        # For epic changes, use slightly smaller chunks for better reliability
        chunk_size = MAX_CHUNK_SIZE - 2 if gap["missing_candles"] > 100 else MAX_CHUNK_SIZE
        
        if gap["missing_candles"] <= chunk_size:
            return [gap]  # Gap is small enough, return as-is
        
        chunks = []
        epic = gap["epic"]
        timeframe = gap["timeframe"]
        gap_start = gap["gap_start"]
        gap_end = gap["gap_end"]
        
        # Calculate time delta for each chunk using dynamic chunk size
        chunk_minutes = chunk_size * timeframe
        chunk_delta = timedelta(minutes=chunk_minutes)
        
        current_start = gap_start
        chunk_number = 1
        
        while current_start < gap_end:
            current_end = min(current_start + chunk_delta, gap_end)
            
            # Calculate missing candles for this chunk
            chunk_duration_minutes = (current_end - current_start).total_seconds() / 60
            chunk_missing_candles = int(chunk_duration_minutes / timeframe)
            
            # Skip empty chunks
            if chunk_missing_candles <= 0:
                break
                
            chunk = {
                "epic": epic,
                "timeframe": timeframe,
                "gap_start": current_start,
                "gap_end": current_end,
                "missing_candles": chunk_missing_candles,
                "priority": gap.get("priority", 1),
                "chunk_number": chunk_number,
                "total_chunks": None,  # Will be set after loop
                "original_gap_size": gap["missing_candles"]
            }
            
            chunks.append(chunk)
            current_start = current_end
            chunk_number += 1
        
        # Set total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks
        
        logger.info(f"Split large gap ({gap['missing_candles']} candles) into {total_chunks} chunks for {epic} {timeframe}m")
        
        return chunks
    
    async def backfill_gap(self, gap: Dict) -> str:
        """
        Backfill a specific gap
        
        Args:
            gap: Gap dictionary from GapDetector
            
        Returns:
            'success' if backfilled, 'skipped' if market closed, 'failed' otherwise
        """
        epic = gap["epic"]
        timeframe = gap["timeframe"]
        resolution = self._timeframe_to_resolution(timeframe)
        
        # Skip very recent gaps (less than 10 minutes old) as they might still be processing
        gap_age_minutes = (datetime.now(timezone.utc) - gap["gap_end"]).total_seconds() / 60
        if gap_age_minutes < 10:
            logger.info(f"Skipping recent gap for {epic} {timeframe}m (only {gap_age_minutes:.1f} minutes old)")
            return 'skipped'
        
        # Skip single-candle gaps less than 30 minutes old (likely streaming delays)
        if gap["missing_candles"] == 1 and gap_age_minutes < 30:
            logger.info(f"Skipping single-candle gap for {epic} (only {gap_age_minutes:.1f} minutes old)")
            return 'skipped'
            
        # Skip gaps from market closure periods
        gap_start = gap["gap_start"]
        gap_end = gap["gap_end"]
        
        # Helper function to check if a datetime is during market closure
        def is_market_closed_time(dt):
            # Saturday - always closed
            if dt.weekday() == 5:  # Saturday
                return True
            # Friday after 20:30 UTC - market closed (extended buffer)
            if dt.weekday() == 4 and (dt.hour >= 21 or (dt.hour == 20 and dt.minute >= 30)):  # Friday >= 20:30 UTC
                return True
            # Sunday before 22:00 UTC - market closed (extended for market reopening buffer)
            if dt.weekday() == 6 and dt.hour < 22:  # Sunday < 22:00 UTC
                return True
            return False
        
        # Skip gaps that occur entirely during market closure
        if is_market_closed_time(gap_start) and is_market_closed_time(gap_end):
            logger.info(f"✅ SKIPPING gap during market closure for {epic}: {gap_start} to {gap_end}")
            return 'skipped'
            
        # Skip gaps from problematic market opening periods (Sunday 21:00-21:30 UTC)
        if gap_start.weekday() == 6 and 21 <= gap_start.hour <= 21:  # Sunday 21:00-21:59 UTC
            logger.debug(f"Skipping gap during market opening period for {epic}: {gap_start}")
            return 'skipped'
        
        # Log chunk information if this is part of a larger gap
        chunk_info = ""
        if "chunk_number" in gap and "total_chunks" in gap:
            chunk_info = f" [chunk {gap['chunk_number']}/{gap['total_chunks']} of {gap['original_gap_size']} total candles]"
        
        logger.info(f"Backfilling gap for {epic} {timeframe}m: "
                   f"{gap['gap_start']} to {gap['gap_end']} "
                   f"({gap['missing_candles']} candles){chunk_info}")
        
        try:
            async with httpx.AsyncClient(base_url=API_BASE_URL, headers=self.headers) as client:
                # Adjust request time range slightly to ensure we get all data
                from_dt = gap["gap_start"] - timedelta(minutes=timeframe)
                to_dt = gap["gap_end"] + timedelta(minutes=timeframe)
                
                url = f"/prices/{epic}"
                # Format dates as IG API expects (no timezone info)
                # Ensure we work with naive datetime objects
                from_naive = from_dt.replace(tzinfo=None) if from_dt.tzinfo else from_dt
                to_naive = to_dt.replace(tzinfo=None) if to_dt.tzinfo else to_dt
                
                from_str = from_naive.strftime("%Y-%m-%dT%H:%M:%S")
                to_str = to_naive.strftime("%Y-%m-%dT%H:%M:%S")
                
                params = {
                    "resolution": resolution,
                    "from": from_str,
                    "to": to_str,
                    "max": min(MAX_CANDLES_PER_REQUEST, gap["missing_candles"] + 2)
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                candles = data.get("prices", [])
                
                if not candles:
                    # This can happen for very recent data, during low liquidity, or market closure
                    gap_start_str = gap["gap_start"].strftime("%A %H:%M UTC")
                    if gap["missing_candles"] == 1:
                        logger.debug(f"No candles returned for single-candle gap in {epic} at {gap_start_str} - likely false positive or market closure")
                    else:
                        # Check if gap occurred during likely market closure
                        dt = gap["gap_start"]
                        is_closure_time = (dt.weekday() == 5 or  # Saturday
                                         (dt.weekday() == 4 and (dt.hour >= 21 or (dt.hour == 20 and dt.minute >= 30))) or  # Friday >= 20:30 UTC
                                         (dt.weekday() == 6 and dt.hour < 22))  # Sunday < 22:00 UTC
                        if is_closure_time:
                            logger.debug(f"No candles returned for gap in {epic} during market closure ({gap_start_str}, {gap['missing_candles']} candles)")
                            self.record_failed_gap(gap, 'market_closed')
                        else:
                            logger.warning(f"No candles returned for gap in {epic} ({gap['missing_candles']} candles at {gap_start_str})")
                            self.record_failed_gap(gap, 'no_data_available')
                    return 'failed'
                
                # Store the candles
                saved_count = 0
                with SessionLocal() as session:
                    for candle_data in candles:
                        ts = self.parse_snapshot_time(candle_data["snapshotTime"])
                        
                        # Ensure timezone consistency for comparison
                        # Convert ts to naive datetime if it has timezone info
                        ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
                        gap_start = gap["gap_start"].replace(tzinfo=None) if gap["gap_start"].tzinfo else gap["gap_start"]
                        gap_end = gap["gap_end"].replace(tzinfo=None) if gap["gap_end"].tzinfo else gap["gap_end"]
                        
                        # Only save candles that are actually in the gap
                        if gap_start <= ts_naive <= gap_end:
                            # Check if candle already exists
                            exists = session.query(IGCandle).filter_by(
                                epic=epic,
                                timeframe=timeframe,
                                start_time=ts_naive
                            ).first()
                            
                            if not exists:
                                try:
                                    # CORRECTED: IG API only provides bid/ask, we need to calculate proper mid prices
                                    # This is the CORRECT approach for backfill (different from streaming data)
                                    bid_open = candle_data["openPrice"]["bid"]
                                    bid_high = candle_data["highPrice"]["bid"]
                                    bid_low = candle_data["lowPrice"]["bid"]
                                    bid_close = candle_data["closePrice"]["bid"]
                                    
                                    ask_open = candle_data["openPrice"]["ask"]
                                    ask_high = candle_data["highPrice"]["ask"]
                                    ask_low = candle_data["lowPrice"]["ask"]
                                    ask_close = candle_data["closePrice"]["ask"]
                                    
                                    # Calculate accurate mid prices (this is correct for REST API data)
                                    mid_open = (bid_open + ask_open) / 2
                                    mid_high = (bid_high + ask_high) / 2
                                    mid_low = (bid_low + ask_low) / 2
                                    mid_close = (bid_close + ask_close) / 2
                                    
                                    candle = IGCandle(
                                        start_time=ts_naive,
                                        epic=epic,
                                        timeframe=timeframe,
                                        open=mid_open,
                                        high=mid_high,
                                        low=mid_low,
                                        close=mid_close,
                                        volume=candle_data.get("lastTradedVolume", 0),
                                        ltv=candle_data.get("lastTradedVolume"),
                                        cons_tick_count=None,
                                        data_source="api_backfill_fixed"  # Mark as corrected backfill data
                                    )
                                    session.add(candle)
                                    saved_count += 1
                                    
                                except (KeyError, TypeError) as e:
                                    logger.warning(f"Skipping incomplete candle at {ts}: {e}")
                                    continue
                    
                    session.commit()
                
                logger.info(f"Successfully backfilled {saved_count} candles for {epic} {timeframe}m")
                self.backfill_stats["candles_recovered"] += saved_count
                
                return 'success' if saved_count > 0 else 'failed'
                
        except Exception as e:
            # Check if it's a 400 error for market closed periods - suppress noise
            if "400 Bad Request" in str(e) and not self.market_is_open():
                logger.debug(f"Backfill failed for {epic} (market closed): {e}")
                self.record_failed_gap(gap, 'market_closed')
            else:
                logger.error(f"Error backfilling gap for {epic}: {e}")
                self.record_failed_gap(gap, 'api_error')
            self.backfill_stats["failures"] += 1
            return 'failed'
    
    async def process_gaps(self, max_gaps: int = BACKFILL_BATCH_SIZE):
        """
        Process detected gaps with rate limiting and prioritization
        
        Args:
            max_gaps: Maximum number of gaps to process in one run
        """
        try:
            # Detect all gaps (removed 15m and 60m - both should be synthesized from base data)
            # Added 1m for parallel collection and future migration to 1m base
            all_gaps = self.gap_detector.detect_all_gaps(self.epics, timeframes=[1, 5])
            
            # Flatten and prioritize gaps
            gaps_list = []
            for gaps in all_gaps.values():
                gaps_list.extend(gaps)
            
            prioritized_gaps = self.gap_detector.prioritize_gaps(gaps_list)
            
            # Track gap detection operation
            track_gap_detection(epics_checked=len(self.epics), gaps_found=len(prioritized_gaps))
            
            if not prioritized_gaps:
                logger.debug("No gaps detected for backfilling")
                return
            
            # Split large gaps into manageable chunks
            chunked_gaps = []
            original_large_gaps = 0
            for gap in prioritized_gaps:
                gap_chunks = self._split_large_gap(gap)
                chunked_gaps.extend(gap_chunks)
                if len(gap_chunks) > 1:
                    original_large_gaps += 1
            
            # Update statistics
            self.backfill_stats["gaps_detected"] = len(prioritized_gaps)
            
            logger.info(f"Found {len(prioritized_gaps)} gaps to backfill")
            if original_large_gaps > 0:
                logger.info(f"Split {original_large_gaps} large gaps into {len(chunked_gaps)} total chunks")
            
            # Log details of gaps for debugging
            for gap in prioritized_gaps[:5]:  # Show first 5 original gaps
                size_info = f"({gap['missing_candles']} candles)"
                if gap['missing_candles'] > MAX_CHUNK_SIZE:
                    num_chunks = (gap['missing_candles'] + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE  # Ceiling division
                    size_info += f" → will be split into ~{num_chunks} chunks"
                logger.info(f"  Gap: {gap['epic']} {gap['timeframe']}m at {gap['gap_start']} {size_info}")
            
            # Process top priority gaps (now including chunks)
            gaps_to_process = chunked_gaps[:max_gaps * 3]  # Allow more chunks since each is smaller
            successful_fills = 0
            
            for gap in gaps_to_process:
                # Skip gaps that are known to have failed
                if self.is_gap_known_failed(gap):
                    logger.debug(f"Skipping known failed gap: {gap['epic']} {gap['timeframe']}m at {gap['gap_start']}")
                    continue

                # Rate limiting
                await asyncio.sleep(BACKFILL_RATE_LIMIT_DELAY)

                # Attempt backfill with retries (fewer retries for single-candle gaps)
                max_attempts = 1 if gap["missing_candles"] == 1 else MAX_BACKFILL_ATTEMPTS
                success = False
                
                for attempt in range(max_attempts):
                    result = await self.backfill_gap(gap)
                    
                    if result == 'success':
                        success = True
                        successful_fills += 1
                        self.backfill_stats["gaps_filled"] += 1
                        
                        # Track successful gap fill
                        track_gap_fill(
                            epic=gap["epic"],
                            gap_start=gap["gap_start"].strftime("%H:%M:%S"),
                            gap_end=gap["gap_end"].strftime("%H:%M:%S"),
                            candles_filled=gap["missing_candles"],
                            success=True
                        )
                        break
                    elif result == 'skipped':
                        # Gap was skipped (market closed, too recent, etc.) - don't retry
                        success = True  # Mark as success to avoid error tracking
                        break
                    else:  # 'failed'
                        if attempt < max_attempts - 1:
                            logger.warning(f"Backfill attempt {attempt + 1} failed, retrying...")
                            await asyncio.sleep(BACKFILL_RATE_LIMIT_DELAY * 2)
                
                if not success:
                    # Track failed gap fill
                    track_gap_fill(
                        epic=gap["epic"],
                        gap_start=gap["gap_start"].strftime("%H:%M:%S"),
                        gap_end=gap["gap_end"].strftime("%H:%M:%S"),
                        candles_filled=0,
                        success=False
                    )
                    
                    # Only log failure as error during market hours
                    if self.market_is_open():
                        logger.error(f"Failed to backfill gap after {MAX_BACKFILL_ATTEMPTS} attempts")
                    else:
                        logger.debug(f"Failed to backfill gap after {MAX_BACKFILL_ATTEMPTS} attempts (market closed)")
            
            logger.info(f"Backfill run complete: {successful_fills}/{len(gaps_to_process)} gaps filled")
            
        except Exception as e:
            logger.error(f"Error in gap processing: {e}")
    
    async def run_continuous(self, check_interval_minutes: int = 5):
        """
        Run continuous gap detection and backfilling
        
        Args:
            check_interval_minutes: How often to check for gaps (default: 5 minutes)
        """
        self.is_running = True
        logger.info(f"Starting continuous auto-backfill service (checking every {check_interval_minutes} minutes)")
        
        # Initialize authentication
        if not await self.initialize_auth():
            logger.error("Failed to initialize auth for auto-backfill service")
            return
        
        while self.is_running:
            try:
                # Check if market is open (skip backfill during market closure)
                if not self.market_is_open():
                    logger.debug("Market closed, skipping backfill check")
                else:
                    logger.info("Running automated gap detection and backfill...")
                    await self.process_gaps()
                    
                    # Log statistics periodically
                    if self.backfill_stats["gaps_detected"] > 0:
                        fill_rate = (self.backfill_stats["gaps_filled"] / 
                                   self.backfill_stats["gaps_detected"]) * 100
                        logger.info(f"Backfill stats: {self.backfill_stats['gaps_filled']}/{self.backfill_stats['gaps_detected']} "
                                  f"gaps filled ({fill_rate:.1f}%), "
                                  f"{self.backfill_stats['candles_recovered']} candles recovered")
                
                # Wait for next check
                await asyncio.sleep(check_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info("Auto-backfill service cancelled")
                break
            except Exception as e:
                logger.error(f"Error in continuous backfill loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
        
        self.is_running = False
        logger.info("Auto-backfill service stopped")
    
    def stop(self):
        """Stop the continuous backfill service"""
        self.is_running = False
        logger.info("Stopping auto-backfill service...")
    
    def get_statistics(self) -> Dict:
        """Get current backfill statistics"""
        now = datetime.now(timezone.utc)

        # Calculate next run time (approximate, since this depends on continuous loop)
        next_run = now + timedelta(minutes=5)  # Default check interval is 5 minutes

        # Prepare statistics in the format expected by Streamlit
        statistics = {
            "total_gaps_detected": self.backfill_stats["gaps_detected"],
            "total_gaps_filled": self.backfill_stats["gaps_filled"],
            "total_candles_recovered": self.backfill_stats["candles_recovered"],
            "total_failures": self.backfill_stats["failures"],
            "total_epics_monitored": len(self.epics)
        }

        return {
            "is_running": self.is_running,
            "stats": self.backfill_stats.copy(),  # Keep original format for compatibility
            "statistics": statistics,  # Add expected format for Streamlit
            "last_check": now.isoformat(),
            "last_run": now.isoformat(),  # Add expected timing fields
            "next_run": next_run.isoformat(),
            "run_interval": 5,  # Default check interval in minutes
            "monitored_epics": len(self.epics)
        }

# Convenience function for one-time gap detection and report
async def detect_and_report_gaps(epics: List[str]):
    """
    Detect and report gaps without backfilling
    
    Args:
        epics: List of trading pairs to check
        
    Returns:
        Formatted gap report string
    """
    detector = GapDetector()
    all_gaps = detector.detect_all_gaps(epics)
    report = detector.format_gap_report(all_gaps)
    
    # Also get statistics
    stats = detector.get_gap_statistics(epics)
    
    return report, stats