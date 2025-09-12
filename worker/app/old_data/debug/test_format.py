#!/usr/bin/env python3
"""
Test timestamp formatting after conversion
"""
import pandas as pd
import pytz
from datetime import datetime

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
        print(f"Error: {e}")
        return None

# Test the conversion and formatting
test_utc_str = "2025-08-27 07:15:00"
print(f"Original UTC string: {test_utc_str}")

stockholm_dt = convert_to_stockholm_time(test_utc_str)
print(f"Stockholm datetime: {stockholm_dt}")

# Format as string (like we do in the code)
formatted = stockholm_dt.strftime('%Y-%m-%d %H:%M:%S')
print(f"Formatted string: {formatted}")

# Test what happens in a mock signal
signal = {
    'timestamp': test_utc_str,
    'signal_type': 'SELL',
    'epic': 'CS.D.EURUSD.MINI.IP'
}

# Apply the same logic as in the code
stockholm_timestamp = convert_to_stockholm_time(signal['timestamp'])
signal['timestamp'] = stockholm_timestamp.strftime('%Y-%m-%d %H:%M:%S')

print(f"Final signal timestamp: {signal['timestamp']}")
print(f"Signal type: {signal['signal_type']}")