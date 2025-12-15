"""
Silver Bullet Time Windows Helper

Handles time window validation for the ICT Silver Bullet strategy.
The strategy only trades during specific one-hour windows.

Time Windows (New York Time):
- London Open: 03:00 - 04:00 AM
- NY AM Session: 10:00 - 11:00 AM (BEST)
- NY PM Session: 02:00 - 03:00 PM
"""

import logging
from datetime import datetime, time
from typing import Dict, Optional, Tuple
from enum import Enum

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import pandas as pd


class SilverBulletSession(Enum):
    """Silver Bullet trading sessions"""
    LONDON_OPEN = "LONDON_OPEN"
    NY_AM = "NY_AM"
    NY_PM = "NY_PM"
    NONE = "NONE"


class SilverBulletTimeWindows:
    """
    Handles time window validation for ICT Silver Bullet strategy.

    The Silver Bullet strategy only trades during specific one-hour windows
    when institutional activity creates predictable liquidity patterns.
    """

    # Time windows in New York time (Eastern)
    # Format: (start_hour, start_minute, end_hour, end_minute)
    NY_TIME_WINDOWS = {
        SilverBulletSession.LONDON_OPEN: (3, 0, 4, 0),   # 03:00-04:00 NY
        SilverBulletSession.NY_AM: (10, 0, 11, 0),       # 10:00-11:00 NY
        SilverBulletSession.NY_PM: (14, 0, 15, 0),       # 14:00-15:00 NY
    }

    # Session quality scores (used for confidence calculation)
    SESSION_QUALITY = {
        SilverBulletSession.NY_AM: 1.00,       # Best - London/NY overlap
        SilverBulletSession.NY_PM: 0.90,       # Good - US afternoon activity
        SilverBulletSession.LONDON_OPEN: 0.85, # Good - European open
        SilverBulletSession.NONE: 0.0,         # Not in any window
    }

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.ny_tz = ZoneInfo("America/New_York")
        self.utc_tz = ZoneInfo("UTC")

    def is_in_silver_bullet_window(
        self,
        timestamp: datetime,
        enabled_sessions: list = None
    ) -> Tuple[bool, SilverBulletSession, str]:
        """
        Check if the given timestamp falls within a Silver Bullet window.

        Args:
            timestamp: The timestamp to check (can be timezone-aware or naive UTC)
            enabled_sessions: List of enabled session names (default: all)

        Returns:
            Tuple of (is_valid, session, reason)
        """
        try:
            # Convert to NY time
            ny_time = self._convert_to_ny_time(timestamp)
            current_time = ny_time.time()

            # Default to all sessions if not specified
            if enabled_sessions is None:
                enabled_sessions = ['LONDON_OPEN', 'NY_AM', 'NY_PM']

            # Check each window
            for session, (start_h, start_m, end_h, end_m) in self.NY_TIME_WINDOWS.items():
                if session == SilverBulletSession.NONE:
                    continue

                # Skip if session not enabled
                if session.value not in enabled_sessions:
                    continue

                window_start = time(start_h, start_m)
                window_end = time(end_h, end_m)

                if window_start <= current_time < window_end:
                    return (
                        True,
                        session,
                        f"In {session.value} window ({start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d} NY)"
                    )

            # Not in any window
            return (
                False,
                SilverBulletSession.NONE,
                f"Outside Silver Bullet windows (NY time: {current_time.strftime('%H:%M')})"
            )

        except Exception as e:
            self.logger.error(f"Error checking Silver Bullet window: {e}")
            return (False, SilverBulletSession.NONE, f"Error: {e}")

    def get_current_session(self, timestamp: datetime) -> SilverBulletSession:
        """
        Get the current active Silver Bullet session.

        Args:
            timestamp: The timestamp to check

        Returns:
            The active session or NONE
        """
        is_valid, session, _ = self.is_in_silver_bullet_window(timestamp)
        return session if is_valid else SilverBulletSession.NONE

    def get_session_quality(self, session: SilverBulletSession) -> float:
        """
        Get the quality score for a session (used in confidence calculation).

        Args:
            session: The session to get quality for

        Returns:
            Quality score between 0.0 and 1.0
        """
        return self.SESSION_QUALITY.get(session, 0.0)

    def get_next_window(
        self,
        timestamp: datetime,
        enabled_sessions: list = None
    ) -> Tuple[SilverBulletSession, datetime, int]:
        """
        Get the next Silver Bullet window and time until it opens.

        Args:
            timestamp: Current timestamp
            enabled_sessions: List of enabled session names

        Returns:
            Tuple of (next_session, window_start_time, minutes_until)
        """
        try:
            ny_time = self._convert_to_ny_time(timestamp)
            current_time = ny_time.time()

            if enabled_sessions is None:
                enabled_sessions = ['LONDON_OPEN', 'NY_AM', 'NY_PM']

            # Sort windows by start time
            windows = []
            for session, (start_h, start_m, end_h, end_m) in self.NY_TIME_WINDOWS.items():
                if session == SilverBulletSession.NONE:
                    continue
                if session.value not in enabled_sessions:
                    continue
                windows.append((session, time(start_h, start_m), time(end_h, end_m)))

            windows.sort(key=lambda x: x[1])

            # Find next window
            for session, window_start, window_end in windows:
                if current_time < window_start:
                    # This window is coming up today
                    window_dt = ny_time.replace(
                        hour=window_start.hour,
                        minute=window_start.minute,
                        second=0,
                        microsecond=0
                    )
                    minutes_until = int((window_dt - ny_time).total_seconds() / 60)
                    return (session, window_dt, minutes_until)

            # All windows have passed today, get first window tomorrow
            first_session, first_start, _ = windows[0]
            tomorrow = ny_time.replace(
                hour=first_start.hour,
                minute=first_start.minute,
                second=0,
                microsecond=0
            ) + pd.Timedelta(days=1)
            minutes_until = int((tomorrow - ny_time).total_seconds() / 60)

            return (first_session, tomorrow, minutes_until)

        except Exception as e:
            self.logger.error(f"Error getting next window: {e}")
            return (SilverBulletSession.NONE, timestamp, 0)

    def get_window_progress(
        self,
        timestamp: datetime
    ) -> Tuple[SilverBulletSession, float, int]:
        """
        Get the progress through the current window.

        Args:
            timestamp: Current timestamp

        Returns:
            Tuple of (session, progress_percentage, minutes_remaining)
        """
        try:
            is_valid, session, _ = self.is_in_silver_bullet_window(timestamp)

            if not is_valid:
                return (SilverBulletSession.NONE, 0.0, 0)

            ny_time = self._convert_to_ny_time(timestamp)
            current_time = ny_time.time()

            start_h, start_m, end_h, end_m = self.NY_TIME_WINDOWS[session]

            # Calculate progress
            window_start_minutes = start_h * 60 + start_m
            window_end_minutes = end_h * 60 + end_m
            current_minutes = current_time.hour * 60 + current_time.minute

            window_duration = window_end_minutes - window_start_minutes
            elapsed = current_minutes - window_start_minutes

            progress = elapsed / window_duration if window_duration > 0 else 0.0
            minutes_remaining = window_end_minutes - current_minutes

            return (session, progress, minutes_remaining)

        except Exception as e:
            self.logger.error(f"Error getting window progress: {e}")
            return (SilverBulletSession.NONE, 0.0, 0)

    def _convert_to_ny_time(self, timestamp: datetime) -> datetime:
        """
        Convert a timestamp to New York time.

        Args:
            timestamp: The timestamp to convert (can be naive UTC or timezone-aware)

        Returns:
            Timestamp in New York timezone
        """
        try:
            # Handle pandas Timestamp
            if isinstance(timestamp, pd.Timestamp):
                if timestamp.tz is None:
                    # Assume UTC if no timezone
                    timestamp = timestamp.tz_localize(self.utc_tz)
                return timestamp.tz_convert(self.ny_tz).to_pydatetime()

            # Handle regular datetime
            if timestamp.tzinfo is None:
                # Assume UTC if no timezone
                timestamp = timestamp.replace(tzinfo=self.utc_tz)

            return timestamp.astimezone(self.ny_tz)

        except Exception as e:
            self.logger.error(f"Error converting to NY time: {e}")
            # Return as-is if conversion fails
            return timestamp

    def get_session_info(self, session: SilverBulletSession) -> Dict:
        """
        Get detailed information about a session.

        Args:
            session: The session to get info for

        Returns:
            Dictionary with session details
        """
        if session == SilverBulletSession.NONE:
            return {
                'name': 'NONE',
                'description': 'Not in any Silver Bullet window',
                'quality': 0.0,
                'window': None,
                'best_pairs': []
            }

        session_details = {
            SilverBulletSession.LONDON_OPEN: {
                'name': 'London Open',
                'description': 'European market open - best for EUR/GBP pairs',
                'quality': self.SESSION_QUALITY[session],
                'window': '03:00-04:00 NY',
                'best_pairs': ['EURUSD', 'GBPUSD', 'EURGBP', 'EURJPY']
            },
            SilverBulletSession.NY_AM: {
                'name': 'NY AM Session',
                'description': 'London/NY overlap - highest volume, best session',
                'quality': self.SESSION_QUALITY[session],
                'window': '10:00-11:00 NY',
                'best_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD']
            },
            SilverBulletSession.NY_PM: {
                'name': 'NY PM Session',
                'description': 'US afternoon - good for USD pairs',
                'quality': self.SESSION_QUALITY[session],
                'window': '14:00-15:00 NY',
                'best_pairs': ['USDJPY', 'USDCAD', 'USDCHF', 'EURUSD']
            }
        }

        return session_details.get(session, {
            'name': session.value,
            'description': 'Unknown session',
            'quality': 0.0,
            'window': None,
            'best_pairs': []
        })

    def is_pair_optimal_for_session(
        self,
        pair: str,
        session: SilverBulletSession
    ) -> Tuple[bool, float]:
        """
        Check if a pair is optimal for the current session.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            session: The current session

        Returns:
            Tuple of (is_optimal, quality_multiplier)
        """
        session_info = self.get_session_info(session)
        best_pairs = session_info.get('best_pairs', [])

        if pair in best_pairs:
            return (True, 1.0)

        # Check if pair contains currencies suited to session
        if session == SilverBulletSession.LONDON_OPEN:
            if 'EUR' in pair or 'GBP' in pair:
                return (True, 0.9)
        elif session == SilverBulletSession.NY_AM:
            # NY AM is good for all majors
            return (True, 0.95)
        elif session == SilverBulletSession.NY_PM:
            if 'USD' in pair:
                return (True, 0.9)

        # Not optimal but still tradeable
        return (False, 0.8)
