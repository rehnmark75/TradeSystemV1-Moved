#!/usr/bin/env python3
"""
Quick test of timezone conversion
"""

import sys
import pandas as pd
import pytz
from datetime import datetime, timedelta

def convert_to_stockholm_time(utc_timestamp):
    """Test timezone conversion function"""
    try:
        # Define timezones
        utc_tz = pytz.UTC
        stockholm_tz = pytz.timezone('Europe/Stockholm')
        
        # Normalize the timestamp first
        if isinstance(utc_timestamp, str):
            dt = pd.to_datetime(utc_timestamp)
        elif isinstance(utc_timestamp, pd.Timestamp):
            dt = utc_timestamp.to_pydatetime()
        else:
            dt = utc_timestamp
        
        # Ensure it's timezone-aware UTC
        if dt.tzinfo is None:
            dt = utc_tz.localize(dt)
        elif dt.tzinfo != utc_tz:
            dt = dt.astimezone(utc_tz)
        
        # Convert to Stockholm time
        stockholm_dt = dt.astimezone(stockholm_tz)
        
        # Return as timezone-naive datetime (but in Stockholm time)
        return stockholm_dt.replace(tzinfo=None)
        
    except Exception as e:
        print(f"⚠️ Could not convert timestamp to Stockholm time: {e}")
        # Fallback: add 2 hours (approximate for Stockholm)
        if isinstance(utc_timestamp, str):
            dt = pd.to_datetime(utc_timestamp)
        elif isinstance(utc_timestamp, pd.Timestamp):
            dt = utc_timestamp.to_pydatetime()
        else:
            dt = utc_timestamp
        
        return dt + timedelta(hours=2)

# Test the conversion
test_utc_str = "2025-08-27 07:15:00"
print(f"Original UTC: {test_utc_str}")

result = convert_to_stockholm_time(test_utc_str)
print(f"Stockholm time: {result}")
print(f"Expected: 2025-08-27 09:15:00 (UTC+2)")