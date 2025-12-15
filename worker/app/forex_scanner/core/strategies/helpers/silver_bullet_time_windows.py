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

    The Silver Bullet strategy only trades during specific time windows
    when institutional activity creates predictable liquidity patterns.

    Time windows are loaded from config_silver_bullet.py for flexibility.
    """

    def __init__(self, logger: logging.Logger = None, config=None):
        self.logger = logger or logging.getLogger(__name__)
        self.ny_tz = ZoneInfo("America/New_York")
        self.utc_tz = ZoneInfo("UTC")

        # Load config if not provided
        if config is None:
            try:
                from forex_scanner.configdata.strategies import config_silver_bullet as config
            except ImportError:
                config = None

        # Time windows in New York time (Eastern) - from config or defaults
        # Config stores UTC hours, we need NY hours
        # UTC to NY: During EST (Nov-Mar) subtract 5, during EDT (Mar-Nov) subtract 4
        # We'll convert at check time for accuracy

        # EXPANDED WINDOWS from config (stored as UTC hours)
        # London Open: 06:00-10:00 UTC = 01:00-05:00 NY (EST) or 02:00-06:00 NY (EDT)
        # NY AM: 13:00-17:00 UTC = 08:00-12:00 NY (EST) or 09:00-13:00 NY (EDT)
        # NY PM: 17:00-21:00 UTC = 12:00-16:00 NY (EST) or 13:00-17:00 NY (EDT)

        if config:
            # Get UTC hours from config and use them directly
            # We'll convert UTC to NY time at runtime for DST accuracy
            london_start = getattr(config, 'LONDON_OPEN_WINDOW_START', 6)  # UTC
            london_end = getattr(config, 'LONDON_OPEN_WINDOW_END', 10)  # UTC
            ny_am_start = getattr(config, 'NY_AM_WINDOW_START', 13)  # UTC
            ny_am_end = getattr(config, 'NY_AM_WINDOW_END', 17)  # UTC
            ny_pm_start = getattr(config, 'NY_PM_WINDOW_START', 17)  # UTC
            ny_pm_end = getattr(config, 'NY_PM_WINDOW_END', 21)  # UTC

            # Store as UTC windows - we'll check against UTC time directly
            self.UTC_TIME_WINDOWS = {
                SilverBulletSession.LONDON_OPEN: (london_start, 0, london_end, 0),
                SilverBulletSession.NY_AM: (ny_am_start, 0, ny_am_end, 0),
                SilverBulletSession.NY_PM: (ny_pm_start, 0, ny_pm_end, 0),
            }

            # Also store NY approximations for logging (assume EST/UTC-5 for display)
            self.NY_TIME_WINDOWS = {
                SilverBulletSession.LONDON_OPEN: (london_start - 5, 0, london_end - 5, 0),
                SilverBulletSession.NY_AM: (ny_am_start - 5, 0, ny_am_end - 5, 0),
                SilverBulletSession.NY_PM: (ny_pm_start - 5, 0, ny_pm_end - 5, 0),
            }

            # Get session quality from config
            config_quality = getattr(config, 'SESSION_QUALITY', {})
            self.SESSION_QUALITY = {
                SilverBulletSession.NY_AM: config_quality.get('NY_AM', 1.00),
                SilverBulletSession.NY_PM: config_quality.get('NY_PM', 0.90),
                SilverBulletSession.LONDON_OPEN: config_quality.get('LONDON_OPEN', 0.85),
                SilverBulletSession.NONE: 0.0,
            }

            self.logger.info(f"📅 Silver Bullet time windows loaded from config:")
            self.logger.info(f"   London Open: {london_start:02d}:00-{london_end:02d}:00 UTC")
            self.logger.info(f"   NY AM: {ny_am_start:02d}:00-{ny_am_end:02d}:00 UTC")
            self.logger.info(f"   NY PM: {ny_pm_start:02d}:00-{ny_pm_end:02d}:00 UTC")
        else:
            # Default hardcoded windows (original ICT spec, UTC hours)
            self.UTC_TIME_WINDOWS = {
                SilverBulletSession.LONDON_OPEN: (8, 0, 9, 0),   # 08:00-09:00 UTC
                SilverBulletSession.NY_AM: (15, 0, 16, 0),       # 15:00-16:00 UTC
                SilverBulletSession.NY_PM: (19, 0, 20, 0),       # 19:00-20:00 UTC
            }

            self.NY_TIME_WINDOWS = {
                SilverBulletSession.LONDON_OPEN: (3, 0, 4, 0),
                SilverBulletSession.NY_AM: (10, 0, 11, 0),
                SilverBulletSession.NY_PM: (14, 0, 15, 0),
            }

            self.SESSION_QUALITY = {
                SilverBulletSession.NY_AM: 1.00,
                SilverBulletSession.NY_PM: 0.90,
                SilverBulletSession.LONDON_OPEN: 0.85,
                SilverBulletSession.NONE: 0.0,
            }

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
            # Convert to UTC for comparison (config stores UTC hours)
            utc_time = self._convert_to_utc(timestamp)
            current_time = utc_time.time()
            current_hour = current_time.hour

            # Default to all sessions if not specified
            if enabled_sessions is None:
                enabled_sessions = ['LONDON_OPEN', 'NY_AM', 'NY_PM']

            # Check each window using UTC times from config
            for session, (start_h, start_m, end_h, end_m) in self.UTC_TIME_WINDOWS.items():
                if session == SilverBulletSession.NONE:
                    continue

                # Skip if session not enabled
                if session.value not in enabled_sessions:
                    continue

                window_start = time(start_h, start_m)
                window_end = time(end_h, end_m)

                if window_start <= current_time < window_end:
                    # Get NY equivalent for display
                    ny_start, ny_start_m, ny_end, ny_end_m = self.NY_TIME_WINDOWS.get(
                        session, (start_h - 5, 0, end_h - 5, 0)
                    )
                    return (
                        True,
                        session,
                        f"In {session.value} window ({ny_start:02d}:{ny_start_m:02d}-{ny_end:02d}:{ny_end_m:02d} NY / {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d} UTC)"
                    )

            # Not in any window - also convert to NY for display
            ny_time = self._convert_to_ny_time(timestamp)
            return (
                False,
                SilverBulletSession.NONE,
                f"Outside Silver Bullet windows (NY time: {ny_time.strftime('%H:%M')}, UTC: {current_time.strftime('%H:%M')})"
            )

        except Exception as e:
            self.logger.error(f"Error checking Silver Bullet window: {e}")
            return (False, SilverBulletSession.NONE, f"Error: {e}")

    def _convert_to_utc(self, timestamp: datetime) -> datetime:
        """
        Convert a timestamp to UTC.

        Args:
            timestamp: The timestamp to convert (can be naive UTC or timezone-aware)

        Returns:
            Timestamp in UTC timezone
        """
        try:
            # Handle pandas Timestamp
            if isinstance(timestamp, pd.Timestamp):
                if timestamp.tz is None:
                    # Assume UTC if no timezone
                    return timestamp.to_pydatetime()
                return timestamp.tz_convert(self.utc_tz).to_pydatetime()

            # Handle regular datetime
            if timestamp.tzinfo is None:
                # Assume UTC if no timezone
                return timestamp

            return timestamp.astimezone(self.utc_tz).replace(tzinfo=None)

        except Exception as e:
            self.logger.error(f"Error converting to UTC: {e}")
            return timestamp

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

            utc_time = self._convert_to_utc(timestamp)
            current_time = utc_time.time()

            start_h, start_m, end_h, end_m = self.UTC_TIME_WINDOWS[session]

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
