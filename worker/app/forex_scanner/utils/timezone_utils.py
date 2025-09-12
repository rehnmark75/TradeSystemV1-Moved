# utils/timezone_utils.py
"""
Timezone utilities for Forex Scanner
Handles conversion between UTC (database) and local time (user)
"""

import pytz
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import logging


class TimezoneManager:
    """Manages timezone conversions for the forex scanner"""
    
    def __init__(self, user_timezone: str = 'Europe/Stockholm'):
        self.user_timezone = user_timezone
        self.utc_tz = pytz.UTC
        self.local_tz = pytz.timezone(user_timezone)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🌍 Timezone manager initialized: {user_timezone}")
    
    def utc_to_local(self, utc_datetime: datetime) -> datetime:
        """Convert UTC datetime to local timezone"""
        if utc_datetime.tzinfo is None:
            # Assume it's UTC if no timezone info
            utc_datetime = self.utc_tz.localize(utc_datetime)
        elif utc_datetime.tzinfo != self.utc_tz:
            # Convert to UTC first
            utc_datetime = utc_datetime.astimezone(self.utc_tz)
        
        return utc_datetime.astimezone(self.local_tz)
    
    def local_to_utc(self, local_datetime: datetime) -> datetime:
        """Convert local datetime to UTC"""
        if local_datetime.tzinfo is None:
            # Assume it's local time if no timezone info
            local_datetime = self.local_tz.localize(local_datetime)
        elif local_datetime.tzinfo != self.local_tz:
            # Convert to local first
            local_datetime = local_datetime.astimezone(self.local_tz)
        
        return local_datetime.astimezone(self.utc_tz)
    
    def get_current_local_time(self) -> datetime:
        """Get current time in local timezone"""
        return datetime.now(self.local_tz)
    
    def get_current_utc_time(self) -> datetime:
        """Get current time in UTC"""
        return datetime.now(self.utc_tz)
    
    def is_market_hours(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if current time (or specified time) is within market hours
        
        Args:
            check_time: Time to check (default: current local time)
            
        Returns:
            True if within market hours
        """
        # Import here to avoid circular imports
        import config
        
        # If market hours respect is disabled, always return True (24/5)
        if not getattr(config, 'RESPECT_MARKET_HOURS', True):
            if getattr(config, 'WEEKEND_SCANNING', False):
                return True  # 24/7 scanning
            else:
                # 24/5 scanning (exclude weekends)
                if check_time is None:
                    check_time = self.get_current_local_time()
                elif check_time.tzinfo is None:
                    check_time = self.local_tz.localize(check_time)
                elif check_time.tzinfo != self.local_tz:
                    check_time = check_time.astimezone(self.local_tz)
                
                weekday = check_time.weekday()  # 0=Monday, 6=Sunday
                
                # Forex 24/5: Monday 00:00 to Friday 23:59 + Sunday evening
                if weekday == 6:  # Sunday
                    # Sunday evening from 21:00 (markets open)
                    return check_time.hour >= 21
                elif weekday == 5:  # Saturday  
                    # Saturday morning until 01:00 (markets close)
                    return check_time.hour <= 1
                else:  # Monday-Friday
                    return True
        
        # Original market hours logic (when RESPECT_MARKET_HOURS = True)
        if check_time is None:
            check_time = self.get_current_local_time()
        elif check_time.tzinfo is None:
            check_time = self.local_tz.localize(check_time)
        elif check_time.tzinfo != self.local_tz:
            check_time = check_time.astimezone(self.local_tz)
        
        hour = check_time.hour
        weekday = check_time.weekday()  # 0=Monday, 6=Sunday
        
        # Get trading hours from config
        trading_hours = getattr(config, 'TRADING_HOURS', {
            'start_hour': 0,
            'end_hour': 23,
            'enabled_days': [0, 1, 2, 3, 4, 6],
            'enable_24_5': True
        })
        
        # Check if today is an enabled day
        if weekday not in trading_hours.get('enabled_days', [0, 1, 2, 3, 4]):
            return False
        
        # Check if 24/5 mode is enabled
        if trading_hours.get('enable_24_5', True):
            # 24/5 forex hours
            if weekday == 6:  # Sunday evening
                return hour >= 21
            elif weekday == 5:  # Saturday morning
                return hour <= 1
            else:  # Monday-Friday
                return True
        
        # Custom hours mode
        start_hour = trading_hours.get('start_hour', 0)
        end_hour = trading_hours.get('end_hour', 23)
        
        return start_hour <= hour <= end_hour
    
    def get_market_session(self, check_time: Optional[datetime] = None) -> str:
        """
        Determine current market session
        
        Args:
            check_time: Time to check (default: current UTC time)
            
        Returns:
            Market session name
        """
        if check_time is None:
            check_time = self.get_current_utc_time()
        elif check_time.tzinfo is None:
            check_time = self.utc_tz.localize(check_time)
        elif check_time.tzinfo != self.utc_tz:
            check_time = check_time.astimezone(self.utc_tz)
        
        hour = check_time.hour
        
        # UTC-based market sessions
        if 0 <= hour < 8:
            return 'Asian'
        elif 8 <= hour < 16:
            return 'European'  
        elif 16 <= hour < 24:
            return 'American'
        else:
            return 'Unknown'
    
    def add_timezone_columns_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add timezone-aware columns to DataFrame
        
        Args:
            df: DataFrame with 'start_time' column (assumed UTC)
            
        Returns:
            DataFrame with additional timezone columns
        """
        if df is None or 'start_time' not in df.columns:
            return df
        
        df_tz = df.copy()
        
        # Convert start_time to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_tz['start_time']):
            df_tz['start_time'] = pd.to_datetime(df_tz['start_time'])
        
        # Add UTC timezone info if not present
        if df_tz['start_time'].dt.tz is None:
            df_tz['start_time'] = df_tz['start_time'].dt.tz_localize('UTC')
        
        # Add local time column
        df_tz['start_time_local'] = df_tz['start_time'].dt.tz_convert(self.user_timezone)
        
        # Add market session info
        df_tz['market_session'] = df_tz['start_time'].apply(
            lambda x: self.get_market_session(x.to_pydatetime())
        )
        
        # Add market hours flag
        df_tz['is_market_hours'] = df_tz['start_time_local'].apply(
            lambda x: self.is_market_hours(x.to_pydatetime())
        )
        
        return df_tz
    
    def get_lookback_time_utc(self, hours_back: int) -> datetime:
        """
        Get UTC time for lookback queries
        
        Args:
            hours_back: Hours to look back from current time
            
        Returns:
            UTC datetime for database queries
        """
        current_utc = self.get_current_utc_time()
        return current_utc - timedelta(hours=hours_back)
    
    def format_time_for_display(self, utc_time: datetime) -> str:
        """
        Format UTC time for user display (in local timezone)
        
        Args:
            utc_time: UTC datetime
            
        Returns:
            Formatted string in local time
        """
        local_time = self.utc_to_local(utc_time)
        return local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    def parse_user_time_to_utc(self, time_str: str) -> datetime:
        """
        Parse user-provided time string to UTC
        
        Args:
            time_str: Time string (assumed to be in user's timezone)
            
        Returns:
            UTC datetime
        """
        # Parse the time string (assuming local timezone)
        try:
            local_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            local_dt = self.local_tz.localize(local_dt)
            return self.local_to_utc(local_dt)
        except ValueError:
            try:
                # Try with different format
                local_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
                local_dt = self.local_tz.localize(local_dt)
                return self.local_to_utc(local_dt)
            except ValueError:
                raise ValueError(f"Cannot parse time string: {time_str}")


# Global timezone manager instance
timezone_manager = TimezoneManager('Europe/Stockholm')


# Convenience functions
def utc_to_local(utc_datetime: datetime) -> datetime:
    """Convert UTC to local time"""
    return timezone_manager.utc_to_local(utc_datetime)


def local_to_utc(local_datetime: datetime) -> datetime:
    """Convert local to UTC time"""
    return timezone_manager.local_to_utc(local_datetime)


def get_current_local_time() -> datetime:
    """Get current local time"""
    return timezone_manager.get_current_local_time()


def get_current_utc_time() -> datetime:
    """Get current UTC time"""
    return timezone_manager.get_current_utc_time()


def is_market_hours() -> bool:
    """Check if currently in market hours"""
    return timezone_manager.is_market_hours()


def get_market_session() -> str:
    """Get current market session"""
    return timezone_manager.get_market_session()


def add_timezone_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add timezone columns to DataFrame"""
    return timezone_manager.add_timezone_columns_to_df(df)


def get_utc_cutoff_time(local_cutoff: datetime, user_timezone: str = 'Europe/Stockholm') -> datetime:
    """Convert local cutoff time to UTC for database queries"""
    tz_manager = TimezoneManager(user_timezone)
    return tz_manager.local_to_utc(local_cutoff)