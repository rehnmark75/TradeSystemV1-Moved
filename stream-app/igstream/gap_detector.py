"""
Gap Detection Module for IG Candle Data
Detects missing candles in the database and identifies time ranges that need backfilling
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from sqlalchemy import select, func
from services.db import SessionLocal
from services.models import IGCandle, FailedGap
from config import DEFAULT_TEST_EPIC

logger = logging.getLogger(__name__)

class GapDetector:
    """Detects gaps in candle data for automated backfilling"""
    
    def __init__(self, max_gap_hours: int = 24):
        """
        Initialize gap detector

        Args:
            max_gap_hours: Maximum gap size to consider for backfilling (default 24 hours)
                          Larger gaps are likely intentional (weekend, holidays)
                          For epic changes, use 72+ hours
        """
        self.max_gap_hours = max_gap_hours
        # Known epic changes that may cause large gaps
        self.epic_change_patterns = {
            "CS.D.EURUSD.CEEM.IP": {
                "previous_epics": ["CS.D.EURUSD.MINI.IP"],
                "change_date": "2025-09-08",
                "max_gap_hours": 72  # Allow larger gaps for this epic
            }
        }
        # Track market opening gap attempts to reduce spam
        # Format: {epic_timeframe_gapstart: attempt_count}
        self.market_opening_attempts = {}

    def _is_market_opening_gap(self, gap_start: datetime, gap_end: datetime) -> bool:
        """Check if gap occurs during problematic market opening period"""
        # Sunday 21:00-21:30 UTC is when markets reopen and often has data issues
        return (gap_start.weekday() == 6 and  # Sunday
                21 <= gap_start.hour <= 21 and  # 21:00-21:59 UTC
                (gap_end - gap_start).total_seconds() <= 1800)  # <= 30 minutes

    def _should_skip_market_opening_gap(self, epic: str, timeframe: int, gap_start: datetime) -> bool:
        """Check if market opening gap should be skipped due to repeated failures"""
        gap_key = f"{epic}_{timeframe}m_{gap_start.strftime('%Y-%m-%d_%H:%M')}"
        attempt_count = self.market_opening_attempts.get(gap_key, 0)

        if attempt_count >= 5:
            return True

        # Increment attempt count
        self.market_opening_attempts[gap_key] = attempt_count + 1
        return False

    def _is_gap_known_failed(self, epic: str, timeframe: int, gap_start: datetime) -> bool:
        """Check if a gap is already recorded as failed and should be excluded from reports"""
        try:
            with SessionLocal() as session:
                existing = session.query(FailedGap).filter(
                    FailedGap.epic == epic,
                    FailedGap.timeframe == timeframe,
                    FailedGap.gap_start == gap_start
                ).first()

                if existing:
                    # Exclude permanently failed gaps (no data available)
                    if existing.failure_reason == 'no_data_available':
                        return True

                    # Exclude recently failed gaps (within 7 days) with high attempt count
                    # Ensure timezone consistency for datetime comparison
                    now_utc = datetime.now(timezone.utc)
                    last_attempted = existing.last_attempted_at
                    if last_attempted.tzinfo is None:
                        last_attempted = last_attempted.replace(tzinfo=timezone.utc)
                    days_since_last_attempt = (now_utc - last_attempted).days
                    if existing.attempt_count >= 3 and days_since_last_attempt < 7:
                        return True

                return False
        except Exception as e:
            logger.error(f"Error checking failed gap: {e}")
            return False

    def detect_gaps(self, epic: str, timeframe: int, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Detect gaps in candle data for a specific epic and timeframe
        
        Args:
            epic: The trading pair (e.g., "CS.D.EURUSD.CEEM.IP")
            timeframe: Timeframe in minutes (1, 5, 15, 30)
            start_time: Start of detection range (default: 48 hours ago)
            end_time: End of detection range (default: now)
            
        Returns:
            List of gap dictionaries with start/end times and missing candle count
        """
        gaps = []
        
        # Default time range: last 48 hours
        if not end_time:
            end_time = datetime.now(timezone.utc)
        if not start_time:
            start_time = end_time - timedelta(hours=48)
            
        try:
            with SessionLocal() as session:
                # Get all candles in the time range, ordered by time
                candles = session.query(IGCandle).filter(
                    IGCandle.epic == epic,
                    IGCandle.timeframe == timeframe,
                    IGCandle.start_time >= start_time,
                    IGCandle.start_time <= end_time
                ).order_by(IGCandle.start_time).all()
                
                if len(candles) < 2:
                    # Not enough data to detect gaps
                    logger.debug(f"Not enough candles for gap detection: {epic} {timeframe}m")
                    return gaps
                
                # Check for gaps between consecutive candles
                expected_delta = timedelta(minutes=timeframe)
                
                # Use epic-specific max gap hours if available
                epic_max_gap_hours = self.max_gap_hours
                if epic in self.epic_change_patterns:
                    epic_max_gap_hours = self.epic_change_patterns[epic]["max_gap_hours"]
                    logger.debug(f"Using epic-specific max gap hours for {epic}: {epic_max_gap_hours}h")
                
                max_gap_delta = timedelta(hours=epic_max_gap_hours)
                
                for i in range(1, len(candles)):
                    prev_candle = candles[i-1]
                    curr_candle = candles[i]
                    
                    # Ensure timestamps are timezone-aware for comparison
                    prev_time = prev_candle.start_time
                    curr_time = curr_candle.start_time
                    
                    # If timestamps are naive, assume UTC
                    if prev_time.tzinfo is None:
                        prev_time = prev_time.replace(tzinfo=timezone.utc)
                    if curr_time.tzinfo is None:
                        curr_time = curr_time.replace(tzinfo=timezone.utc)
                    
                    actual_delta = curr_time - prev_time
                    
                    # Check if there's a gap
                    if actual_delta > expected_delta:
                        # Calculate missing candles
                        missing_candles = int(actual_delta.total_seconds() / 60 / timeframe) - 1
                        
                        # Skip very large gaps (weekends, holidays) unless it's a known epic change
                        is_epic_change = (epic in self.epic_change_patterns and 
                                        actual_delta.total_seconds() / 3600 <= epic_max_gap_hours)
                        
                        if actual_delta > max_gap_delta and not is_epic_change:
                            gap_hours = actual_delta.total_seconds() / 3600
                            logger.debug(f"Skipping large gap ({gap_hours:.1f}h) for {epic} - likely market closure")
                            continue
                        
                        # Log epic change detection
                        if is_epic_change and actual_delta > timedelta(hours=self.max_gap_hours):
                            gap_hours = actual_delta.total_seconds() / 3600
                            logger.info(f"Detected potential epic change gap for {epic}: {gap_hours:.1f}h gap")
                        
                        # Skip small gaps during low volatility periods (single candle)
                        if missing_candles < 1:
                            continue

                        # Skip gaps that occur entirely during market closure to reduce log noise
                        gap_start_time = prev_time + expected_delta
                        gap_end_time = curr_time - expected_delta

                        def is_market_closed_time(dt):
                            """Check if a datetime is during market closure"""
                            if dt.weekday() == 5:  # Saturday
                                return True
                            if dt.weekday() == 4 and (dt.hour >= 21 or (dt.hour == 20 and dt.minute >= 30)):  # Friday >= 20:30 UTC
                                return True
                            if dt.weekday() == 6 and dt.hour < 22:  # Sunday < 22:00 UTC (extended for market reopening buffer)
                                return True
                            return False

                        # Skip gaps that occur entirely during market closure (reduces log spam)
                        if is_market_closed_time(gap_start_time) and is_market_closed_time(gap_end_time):
                            logger.debug(f"Skipping gap during market closure for {epic} {timeframe}m: "
                                       f"{missing_candles} candles from {gap_start_time} to {gap_end_time}")
                            continue

                        # Check for market opening gaps and skip if attempted too many times
                        if self._is_market_opening_gap(gap_start_time, gap_end_time):
                            if self._should_skip_market_opening_gap(epic, timeframe, gap_start_time):
                                attempt_count = self.market_opening_attempts.get(
                                    f"{epic}_{timeframe}m_{gap_start_time.strftime('%Y-%m-%d_%H:%M')}", 0
                                )
                                logger.debug(f"Skipping market opening gap for {epic} {timeframe}m after {attempt_count} attempts: "
                                           f"{missing_candles} candles from {gap_start_time} to {gap_end_time}")
                                continue
                            else:
                                # Log with attempt count for market opening gaps
                                attempt_count = self.market_opening_attempts.get(
                                    f"{epic}_{timeframe}m_{gap_start_time.strftime('%Y-%m-%d_%H:%M')}", 0
                                )
                                logger.info(f"Market opening gap detected for {epic} {timeframe}m (attempt {attempt_count}/5): "
                                          f"{missing_candles} candles from {gap_start_time} to {gap_end_time}")

                        gap_start = prev_time + expected_delta

                        # Skip gaps that are known to have failed
                        if self._is_gap_known_failed(epic, timeframe, gap_start):
                            logger.debug(f"Skipping known failed gap: {epic} {timeframe}m at {gap_start}")
                            continue

                        gap = {
                            "epic": epic,
                            "timeframe": timeframe,
                            "gap_start": gap_start,
                            "gap_end": curr_time - expected_delta,
                            "missing_candles": missing_candles,
                            "gap_duration_minutes": int(actual_delta.total_seconds() / 60) - timeframe,
                            "is_epic_change": is_epic_change,  # Flag for epic change gaps
                            "priority": 1 if is_epic_change else 2  # Higher priority for epic changes
                        }

                        gaps.append(gap)

                        # Check if gap is during market closure to reduce log noise
                        gap_start_dt = gap['gap_start']
                        is_market_closure = (gap_start_dt.weekday() == 5 or  # Saturday
                                           (gap_start_dt.weekday() == 4 and (gap_start_dt.hour >= 21 or (gap_start_dt.hour == 20 and gap_start_dt.minute >= 30))) or  # Friday >= 20:30 UTC
                                           gap_start_dt.weekday() == 6)  # Sunday - market closed all day

                        if is_market_closure:
                            # Log weekend gaps at debug level to reduce noise
                            logger.debug(f"Weekend gap detected in {epic} {timeframe}m: "
                                       f"{missing_candles} candles missing from "
                                       f"{gap['gap_start']} to {gap['gap_end']}")
                        else:
                            # Log normal trading hour gaps as warnings
                            logger.warning(f"Gap detected in {epic} {timeframe}m: "
                                         f"{missing_candles} candles missing from "
                                         f"{gap['gap_start']} to {gap['gap_end']}")
                
                # Check for gap at the beginning (missing recent data)
                if candles:
                    latest_candle = candles[-1]
                    latest_time = latest_candle.start_time
                    
                    # Ensure timestamp is timezone-aware
                    if latest_time.tzinfo is None:
                        latest_time = latest_time.replace(tzinfo=timezone.utc)
                    
                    expected_latest = end_time - expected_delta
                    
                    if latest_time < expected_latest:
                        actual_delta = end_time - latest_time
                        missing_candles = int(actual_delta.total_seconds() / 60 / timeframe) - 1
                        
                        # Only report recent gaps if they're significant (more than 2 candles or older than 15 minutes)
                        gap_age_minutes = (end_time - latest_time).total_seconds() / 60
                        if missing_candles > 0 and actual_delta <= max_gap_delta:
                            if missing_candles > 2 or gap_age_minutes > 15:
                                recent_gap_start = latest_time + expected_delta
                                recent_gap_end = end_time

                                # Check if this recent gap is during market closure
                                def is_market_closed_time(dt):
                                    """Check if a datetime is during market closure"""
                                    if dt.weekday() == 5:  # Saturday
                                        return True
                                    if dt.weekday() == 4 and (dt.hour >= 21 or (dt.hour == 20 and dt.minute >= 30)):  # Friday >= 20:30 UTC
                                        return True
                                    if dt.weekday() == 6 and dt.hour < 22:  # Sunday < 22:00 UTC (extended for market reopening buffer)
                                        return True
                                    return False

                                # Skip recent gaps during market closure to reduce log noise
                                if is_market_closed_time(recent_gap_start) and is_market_closed_time(recent_gap_end):
                                    logger.debug(f"Skipping recent gap during market closure for {epic} {timeframe}m: "
                                               f"{missing_candles} candles from {recent_gap_start} to now")
                                # Check for recent market opening gaps
                                elif self._is_market_opening_gap(recent_gap_start, recent_gap_end):
                                    if self._should_skip_market_opening_gap(epic, timeframe, recent_gap_start):
                                        attempt_count = self.market_opening_attempts.get(
                                            f"{epic}_{timeframe}m_{recent_gap_start.strftime('%Y-%m-%d_%H:%M')}", 0
                                        )
                                        logger.debug(f"Skipping recent market opening gap for {epic} {timeframe}m after {attempt_count} attempts: "
                                                   f"{missing_candles} candles from {recent_gap_start} to now")
                                    else:
                                        # Log with attempt count for recent market opening gaps
                                        attempt_count = self.market_opening_attempts.get(
                                            f"{epic}_{timeframe}m_{recent_gap_start.strftime('%Y-%m-%d_%H:%M')}", 0
                                        )
                                        logger.info(f"Recent market opening gap detected for {epic} {timeframe}m (attempt {attempt_count}/5): "
                                                  f"{missing_candles} candles from {recent_gap_start} to now")

                                        # Check if this recent gap is known to have failed
                                        recent_gap_start_actual = latest_time + expected_delta
                                        if not self._is_gap_known_failed(epic, timeframe, recent_gap_start_actual):
                                            gap = {
                                                "epic": epic,
                                                "timeframe": timeframe,
                                                "gap_start": recent_gap_start_actual,
                                                "gap_end": end_time,
                                                "missing_candles": missing_candles,
                                                "gap_duration_minutes": int(actual_delta.total_seconds() / 60) - timeframe,
                                                "is_recent": True  # Flag for recent data gaps
                                            }
                                            gaps.append(gap)
                                        else:
                                            logger.debug(f"Skipping known failed recent gap: {epic} {timeframe}m at {recent_gap_start_actual}")
                                else:
                                    # Check if this recent gap is known to have failed
                                    recent_gap_start_actual = latest_time + expected_delta
                                    if not self._is_gap_known_failed(epic, timeframe, recent_gap_start_actual):
                                        gap = {
                                            "epic": epic,
                                            "timeframe": timeframe,
                                            "gap_start": recent_gap_start_actual,
                                            "gap_end": end_time,
                                            "missing_candles": missing_candles,
                                            "gap_duration_minutes": int(actual_delta.total_seconds() / 60) - timeframe,
                                            "is_recent": True  # Flag for recent data gaps
                                        }
                                        gaps.append(gap)
                                        logger.warning(f"Recent gap detected in {epic} {timeframe}m: "
                                                     f"{missing_candles} candles missing from "
                                                     f"{gap['gap_start']} to now")
                                    else:
                                        logger.debug(f"Skipping known failed recent gap: {epic} {timeframe}m at {recent_gap_start_actual}")
                            else:
                                logger.debug(f"Ignoring minor recent gap in {epic} {timeframe}m: "
                                           f"{missing_candles} candles, {gap_age_minutes:.1f} minutes old")
                
        except Exception as e:
            logger.error(f"Error detecting gaps for {epic} {timeframe}m: {e}")
            
        return gaps
    
    def detect_all_gaps(self, epics: List[str], timeframes: List[int] = [5]) -> Dict[str, List[Dict]]:
        """
        Detect gaps for multiple epics and timeframes
        
        Args:
            epics: List of trading pairs to check
            timeframes: List of timeframes in minutes (default: [5])
            
        Returns:
            Dictionary mapping "epic_timeframe" to list of gaps
        """
        all_gaps = {}
        
        for epic in epics:
            for timeframe in timeframes:
                key = f"{epic}_{timeframe}m"
                gaps = self.detect_gaps(epic, timeframe)
                
                if gaps:
                    all_gaps[key] = gaps
                    logger.info(f"Found {len(gaps)} gaps for {key}")
                else:
                    logger.debug(f"No gaps found for {key}")
                    
        return all_gaps
    
    def get_gap_statistics(self, epics: List[str]) -> Dict:
        """
        Get statistics about gaps across all epics
        
        Args:
            epics: List of trading pairs to analyze
            
        Returns:
            Dictionary with gap statistics
        """
        stats = {
            "total_gaps": 0,
            "total_missing_candles": 0,
            "gaps_by_epic": {},
            "gaps_by_timeframe": {5: 0},
            "recent_gaps": 0,
            "largest_gap_minutes": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            all_gaps = self.detect_all_gaps(epics, [1])  # v2.17.0: Only check 1m gaps

            for key, gaps in all_gaps.items():
                epic = key.rsplit("_", 1)[0]
                timeframe = int(key.rsplit("_", 1)[1].replace("m", ""))

                if epic not in stats["gaps_by_epic"]:
                    stats["gaps_by_epic"][epic] = {"1m": 0, "missing_candles": 0}
                
                stats["gaps_by_epic"][epic][f"{timeframe}m"] = len(gaps)
                stats["gaps_by_timeframe"][timeframe] += len(gaps)
                stats["total_gaps"] += len(gaps)
                
                for gap in gaps:
                    stats["total_missing_candles"] += gap["missing_candles"]
                    stats["gaps_by_epic"][epic]["missing_candles"] += gap["missing_candles"]
                    
                    if gap.get("is_recent", False):
                        stats["recent_gaps"] += 1
                        
                    if gap["gap_duration_minutes"] > stats["largest_gap_minutes"]:
                        stats["largest_gap_minutes"] = gap["gap_duration_minutes"]
                        
        except Exception as e:
            logger.error(f"Error calculating gap statistics: {e}")
            stats["error"] = str(e)
            
        return stats
    
    def prioritize_gaps(self, gaps: List[Dict]) -> List[Dict]:
        """
        Prioritize gaps for backfilling based on recency and size
        
        Args:
            gaps: List of gap dictionaries
            
        Returns:
            Sorted list of gaps (highest priority first)
        """
        # Sort by:
        # 1. Recent gaps first (is_recent flag)
        # 2. Smaller gaps first (easier to backfill)
        # 3. Earlier gaps first (chronological order)
        
        return sorted(gaps, key=lambda g: (
            not g.get("is_recent", False),  # Recent gaps first
            g["missing_candles"],            # Smaller gaps first
            g["gap_start"]                   # Earlier gaps first
        ))
    
    def format_gap_report(self, all_gaps: Dict[str, List[Dict]]) -> str:
        """
        Format a human-readable gap report
        
        Args:
            all_gaps: Dictionary of gaps by epic_timeframe
            
        Returns:
            Formatted string report
        """
        if not all_gaps:
            return "No gaps detected in candle data."
        
        report = ["=" * 60, "GAP DETECTION REPORT", "=" * 60]
        report.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        report.append("")
        
        total_gaps = sum(len(gaps) for gaps in all_gaps.values())
        total_missing = sum(
            gap["missing_candles"] 
            for gaps in all_gaps.values() 
            for gap in gaps
        )
        
        report.append(f"Total Gaps Found: {total_gaps}")
        report.append(f"Total Missing Candles: {total_missing}")
        report.append("")
        
        for key, gaps in sorted(all_gaps.items()):
            epic, timeframe = key.rsplit("_", 1)
            report.append(f"\n{epic} ({timeframe}):")
            report.append("-" * 40)
            
            prioritized_gaps = self.prioritize_gaps(gaps)
            
            for i, gap in enumerate(prioritized_gaps[:5], 1):  # Show top 5 gaps
                recent_flag = " [RECENT]" if gap.get("is_recent", False) else ""
                report.append(
                    f"  {i}. {gap['gap_start'].strftime('%Y-%m-%d %H:%M')} to "
                    f"{gap['gap_end'].strftime('%H:%M')} "
                    f"({gap['missing_candles']} candles){recent_flag}"
                )
                
            if len(gaps) > 5:
                report.append(f"  ... and {len(gaps) - 5} more gaps")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)