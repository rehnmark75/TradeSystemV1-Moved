"""
Gap Detection Module for IG Candle Data
Detects missing candles in the database and identifies time ranges that need backfilling
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from sqlalchemy import select, func
from services.db import SessionLocal
from services.models import IGCandle
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
        """
        self.max_gap_hours = max_gap_hours
        
    def detect_gaps(self, epic: str, timeframe: int, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Detect gaps in candle data for a specific epic and timeframe
        
        Args:
            epic: The trading pair (e.g., "CS.D.EURUSD.CEEM.IP")
            timeframe: Timeframe in minutes (5, 15, 60)
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
                max_gap_delta = timedelta(hours=self.max_gap_hours)
                
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
                        
                        # Skip very large gaps (weekends, holidays)
                        if actual_delta > max_gap_delta:
                            logger.debug(f"Skipping large gap ({actual_delta}) for {epic} - likely market closure")
                            continue
                        
                        # Skip small gaps during low volatility periods (single candle)
                        if missing_candles < 1:
                            continue
                            
                        gap = {
                            "epic": epic,
                            "timeframe": timeframe,
                            "gap_start": prev_time + expected_delta,
                            "gap_end": curr_time - expected_delta,
                            "missing_candles": missing_candles,
                            "gap_duration_minutes": int(actual_delta.total_seconds() / 60) - timeframe
                        }
                        
                        gaps.append(gap)
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
                                gap = {
                                    "epic": epic,
                                    "timeframe": timeframe,
                                    "gap_start": latest_time + expected_delta,
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
                                logger.debug(f"Ignoring minor recent gap in {epic} {timeframe}m: "
                                           f"{missing_candles} candles, {gap_age_minutes:.1f} minutes old")
                
        except Exception as e:
            logger.error(f"Error detecting gaps for {epic} {timeframe}m: {e}")
            
        return gaps
    
    def detect_all_gaps(self, epics: List[str], timeframes: List[int] = [5, 60]) -> Dict[str, List[Dict]]:
        """
        Detect gaps for multiple epics and timeframes
        
        Args:
            epics: List of trading pairs to check
            timeframes: List of timeframes in minutes (default: [5, 15])
            
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
            "gaps_by_timeframe": {5: 0, 60: 0},
            "recent_gaps": 0,
            "largest_gap_minutes": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            all_gaps = self.detect_all_gaps(epics, [5, 60])
            
            for key, gaps in all_gaps.items():
                epic = key.rsplit("_", 1)[0]
                timeframe = int(key.rsplit("_", 1)[1].replace("m", ""))
                
                if epic not in stats["gaps_by_epic"]:
                    stats["gaps_by_epic"][epic] = {"5m": 0, "60m": 0, "missing_candles": 0}
                
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