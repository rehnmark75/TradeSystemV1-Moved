"""
Master Pattern Session Analyzer
===============================
Utility functions for analyzing trading sessions and detecting session-specific
patterns for the ICT Power of 3 (AMD) strategy.

Sessions (UTC):
- Asian: 00:00-08:00 (Accumulation)
- London Open: 08:00-10:00 (Manipulation)
- Distribution: 08:00-12:00 (Entry window)
"""

import logging
from datetime import datetime, time, timedelta, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MasterPatternSessionAnalyzer:
    """
    Analyzes trading sessions for the Master Pattern (ICT Power of 3) strategy.

    Responsibilities:
    - Session detection (Asian, London, NY)
    - Daily open calculation
    - Session high/low identification
    - Time-based filtering
    """

    # Session definitions (UTC)
    SESSIONS = {
        'asian': {
            'start': time(0, 0),
            'end': time(8, 0),
            'description': 'Asian Session (Accumulation)',
        },
        'london_open': {
            'start': time(8, 0),
            'end': time(10, 0),
            'description': 'London Open (Manipulation)',
        },
        'london': {
            'start': time(8, 0),
            'end': time(16, 0),
            'description': 'London Session',
        },
        'ny_open': {
            'start': time(13, 0),
            'end': time(15, 0),
            'description': 'New York Open',
        },
        'new_york': {
            'start': time(13, 0),
            'end': time(21, 0),
            'description': 'New York Session',
        },
        'distribution': {
            'start': time(8, 0),
            'end': time(12, 0),
            'description': 'Distribution Window (Entry)',
        },
    }

    def __init__(self):
        """Initialize session analyzer."""
        self.logger = logging.getLogger(f"{__name__}.MasterPatternSessionAnalyzer")

    def get_current_session(self, timestamp: datetime) -> Optional[str]:
        """
        Get the current trading session for a timestamp.

        Args:
            timestamp: UTC datetime

        Returns:
            Session name or None if outside main sessions
        """
        current_time = timestamp.time()

        # Check in order of specificity
        if self._is_in_time_range(current_time, 'asian'):
            return 'asian'
        elif self._is_in_time_range(current_time, 'london_open'):
            return 'london_open'
        elif self._is_in_time_range(current_time, 'london'):
            return 'london'
        elif self._is_in_time_range(current_time, 'ny_open'):
            return 'ny_open'
        elif self._is_in_time_range(current_time, 'new_york'):
            return 'new_york'
        else:
            return None

    def _is_in_time_range(self, current_time: time, session_name: str) -> bool:
        """Check if time is within a session range."""
        session = self.SESSIONS.get(session_name)
        if not session:
            return False

        start = session['start']
        end = session['end']

        # Handle overnight sessions
        if start <= end:
            return start <= current_time < end
        else:
            return current_time >= start or current_time < end

    def is_asian_session(self, timestamp: datetime) -> bool:
        """Check if timestamp is in Asian session (accumulation phase)."""
        return self._is_in_time_range(timestamp.time(), 'asian')

    def is_london_open(self, timestamp: datetime) -> bool:
        """Check if timestamp is in London open window (manipulation phase)."""
        return self._is_in_time_range(timestamp.time(), 'london_open')

    def is_distribution_window(self, timestamp: datetime) -> bool:
        """Check if timestamp is in distribution/entry window."""
        return self._is_in_time_range(timestamp.time(), 'distribution')

    def is_entry_allowed(self, timestamp: datetime, cutoff: time = time(10, 0)) -> bool:
        """Check if entry is still allowed (before cutoff)."""
        return timestamp.time() < cutoff

    def get_daily_open(self, df: pd.DataFrame, target_date: Optional[date] = None) -> Optional[float]:
        """
        Get the daily open price (00:00 UTC candle open).

        Args:
            df: DataFrame with OHLCV data (must have datetime index or 'timestamp' column)
            target_date: Date to get open for (defaults to most recent)

        Returns:
            Daily open price or None if not found
        """
        try:
            # Ensure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif 'start_time' in df.columns:
                    df = df.set_index('start_time')
                else:
                    self.logger.warning("No datetime index or timestamp column found")
                    return None

            # Convert to UTC if needed
            if df.index.tz is not None:
                df = df.tz_convert('UTC')

            # Determine target date
            if target_date is None:
                target_date = df.index[-1].date()

            # Find the first candle of the day (closest to 00:00 UTC)
            day_start = datetime.combine(target_date, time(0, 0))
            day_end = datetime.combine(target_date, time(0, 30))  # Buffer for 5m/15m candles

            mask = (df.index >= pd.Timestamp(day_start)) & (df.index < pd.Timestamp(day_end))
            day_open_candles = df[mask]

            if len(day_open_candles) > 0:
                return float(day_open_candles.iloc[0]['open'])

            # Fallback: get first candle of the day
            mask = df.index.date == target_date
            day_data = df[mask]
            if len(day_data) > 0:
                return float(day_data.iloc[0]['open'])

            return None

        except Exception as e:
            self.logger.error(f"Error getting daily open: {e}")
            return None

    def get_session_data(
        self,
        df: pd.DataFrame,
        session_name: str,
        target_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Extract data for a specific session.

        Args:
            df: DataFrame with OHLCV data
            session_name: Name of session ('asian', 'london_open', etc.)
            target_date: Date to filter (defaults to most recent)

        Returns:
            DataFrame filtered to session time range
        """
        try:
            session = self.SESSIONS.get(session_name)
            if not session:
                self.logger.warning(f"Unknown session: {session_name}")
                return pd.DataFrame()

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif 'start_time' in df.columns:
                    df = df.set_index('start_time')

            # Determine target date
            if target_date is None:
                target_date = df.index[-1].date()

            # Build time range
            start_dt = datetime.combine(target_date, session['start'])
            end_dt = datetime.combine(target_date, session['end'])

            # Handle overnight sessions
            if session['start'] > session['end']:
                # Session spans midnight
                prev_date = target_date - timedelta(days=1)
                start_dt = datetime.combine(prev_date, session['start'])

            # Filter data
            mask = (df.index >= pd.Timestamp(start_dt)) & (df.index < pd.Timestamp(end_dt))
            return df[mask].copy()

        except Exception as e:
            self.logger.error(f"Error getting session data for {session_name}: {e}")
            return pd.DataFrame()

    def get_asian_session_range(
        self,
        df: pd.DataFrame,
        target_date: Optional[date] = None
    ) -> Optional[Dict]:
        """
        Get the Asian session range (accumulation zone).

        Args:
            df: DataFrame with OHLCV data
            target_date: Date to analyze

        Returns:
            Dict with 'high', 'low', 'range_pips', 'candle_count', 'daily_open'
        """
        try:
            asian_data = self.get_session_data(df, 'asian', target_date)

            if len(asian_data) == 0:
                return None

            range_high = float(asian_data['high'].max())
            range_low = float(asian_data['low'].min())
            daily_open = self.get_daily_open(df, target_date)

            return {
                'high': range_high,
                'low': range_low,
                'range': range_high - range_low,
                'midpoint': (range_high + range_low) / 2,
                'candle_count': len(asian_data),
                'daily_open': daily_open,
                'start_time': asian_data.index[0] if len(asian_data) > 0 else None,
                'end_time': asian_data.index[-1] if len(asian_data) > 0 else None,
            }

        except Exception as e:
            self.logger.error(f"Error calculating Asian session range: {e}")
            return None

    def get_session_high_low(
        self,
        df: pd.DataFrame,
        session_name: str,
        target_date: Optional[date] = None
    ) -> Optional[Dict]:
        """
        Get session high and low (potential liquidity targets).

        Args:
            df: DataFrame with OHLCV data
            session_name: Session to analyze
            target_date: Date to filter

        Returns:
            Dict with 'high', 'low', 'high_time', 'low_time'
        """
        try:
            session_data = self.get_session_data(df, session_name, target_date)

            if len(session_data) == 0:
                return None

            high_idx = session_data['high'].idxmax()
            low_idx = session_data['low'].idxmin()

            return {
                'high': float(session_data['high'].max()),
                'low': float(session_data['low'].min()),
                'high_time': high_idx,
                'low_time': low_idx,
            }

        except Exception as e:
            self.logger.error(f"Error getting session high/low: {e}")
            return None

    def get_previous_day_high_low(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Get previous day's high and low (liquidity targets).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with 'high', 'low' from previous day
        """
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif 'start_time' in df.columns:
                    df = df.set_index('start_time')

            # Get current date and previous date
            current_date = df.index[-1].date()
            prev_date = current_date - timedelta(days=1)

            # Filter to previous day
            mask = df.index.date == prev_date
            prev_day_data = df[mask]

            if len(prev_day_data) == 0:
                return None

            return {
                'high': float(prev_day_data['high'].max()),
                'low': float(prev_day_data['low'].min()),
                'date': prev_date,
            }

        except Exception as e:
            self.logger.error(f"Error getting previous day high/low: {e}")
            return None

    def calculate_atr_for_session(
        self,
        df: pd.DataFrame,
        session_name: str,
        period: int = 14,
        target_date: Optional[date] = None
    ) -> Optional[float]:
        """
        Calculate ATR for a specific session.

        Args:
            df: DataFrame with OHLCV data
            session_name: Session to calculate ATR for
            period: ATR period
            target_date: Date to filter

        Returns:
            ATR value for the session
        """
        try:
            session_data = self.get_session_data(df, session_name, target_date)

            if len(session_data) < period:
                return None

            # Calculate True Range
            high = session_data['high']
            low = session_data['low']
            close = session_data['close'].shift(1)

            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR
            atr = tr.rolling(window=period).mean().iloc[-1]

            return float(atr) if not pd.isna(atr) else None

        except Exception as e:
            self.logger.error(f"Error calculating session ATR: {e}")
            return None

    def get_time_until_session(self, current_time: datetime, session_name: str) -> Optional[timedelta]:
        """
        Calculate time until a session starts.

        Args:
            current_time: Current UTC datetime
            session_name: Target session

        Returns:
            Timedelta until session starts, or None if in session
        """
        try:
            session = self.SESSIONS.get(session_name)
            if not session:
                return None

            session_start = session['start']
            current = current_time.time()

            # If already in session
            if self._is_in_time_range(current, session_name):
                return timedelta(0)

            # Calculate time until session
            session_start_dt = datetime.combine(current_time.date(), session_start)

            if current > session_start:
                # Session is tomorrow
                session_start_dt += timedelta(days=1)

            return session_start_dt - current_time

        except Exception as e:
            self.logger.error(f"Error calculating time until session: {e}")
            return None

    def validate_session_sequence(
        self,
        accumulation_time: datetime,
        manipulation_time: datetime,
        entry_time: datetime
    ) -> Tuple[bool, str]:
        """
        Validate that AMD phases occurred in correct session sequence.

        Args:
            accumulation_time: When accumulation was detected
            manipulation_time: When manipulation occurred
            entry_time: When entry would be taken

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check accumulation was in Asian session
        if not self.is_asian_session(accumulation_time):
            return False, "Accumulation not in Asian session"

        # Check manipulation was in London open
        if not self.is_london_open(manipulation_time):
            return False, "Manipulation not in London open window"

        # Check entry is in distribution window
        if not self.is_distribution_window(entry_time):
            return False, "Entry not in distribution window"

        # Check sequence order
        if manipulation_time <= accumulation_time:
            return False, "Manipulation must occur after accumulation"

        if entry_time <= manipulation_time:
            return False, "Entry must occur after manipulation"

        # Check entry is before cutoff
        if not self.is_entry_allowed(entry_time):
            return False, "Entry time past cutoff"

        return True, "Valid AMD sequence"


# Module-level instance for convenience
session_analyzer = MasterPatternSessionAnalyzer()


# Convenience functions
def is_asian_session(timestamp: datetime) -> bool:
    """Check if timestamp is in Asian session."""
    return session_analyzer.is_asian_session(timestamp)


def is_london_open(timestamp: datetime) -> bool:
    """Check if timestamp is in London open window."""
    return session_analyzer.is_london_open(timestamp)


def is_distribution_window(timestamp: datetime) -> bool:
    """Check if timestamp is in distribution window."""
    return session_analyzer.is_distribution_window(timestamp)


def get_daily_open(df: pd.DataFrame, target_date: Optional[date] = None) -> Optional[float]:
    """Get daily open price."""
    return session_analyzer.get_daily_open(df, target_date)


def get_asian_range(df: pd.DataFrame, target_date: Optional[date] = None) -> Optional[Dict]:
    """Get Asian session range."""
    return session_analyzer.get_asian_session_range(df, target_date)
