# data _utils.py


from datetime import datetime, timedelta
import pandas as pd
import pytz
from sqlalchemy import text

from services.enhance_data import *

def fetch_candle_data(engine, epic, timeframe=15, lookback_hours=500, user_timezone='Europe/Stockholm'):
    """
    Fetch candle data with automatic timezone handling
    
    Args:
        engine: Database engine
        epic: Epic code
        timeframe: Timeframe in minutes (5, 15, 60, etc.)
        lookback_hours: How many hours back to fetch
        user_timezone: Your timezone (default: Europe/Stockholm)
    
    Returns:
        DataFrame with timezone-aware timestamps
    """
    
    # Set up timezones
    utc_tz = pytz.UTC
    user_tz = pytz.timezone(user_timezone)
    
    # Calculate 'since' time in UTC (database storage format)
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    
    # Determine source timeframe
    source_tf = 5 if timeframe == 15 else timeframe
    
    query = text("""
        SELECT start_time, open, high, low, close, ltv
        FROM ig_candles
        WHERE epic = :epic
        AND timeframe = :timeframe
        AND start_time >= :since
        ORDER BY start_time ASC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {
            "epic": epic,
            "timeframe": source_tf,
            "since": since
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if df.empty:
        raise ValueError(f"No data returned for epic={epic}, timeframe={source_tf}")
    
    # Convert timestamps with timezone awareness
    df['start_time'] = pd.to_datetime(df['start_time'])
    
    # Add timezone information
    df['start_time_utc'] = df['start_time'].dt.tz_localize('UTC')  # Mark as UTC
    df['start_time_local'] = df['start_time_utc'].dt.tz_convert(user_timezone)  # Convert to your timezone
    df['start_time'] = df['start_time_utc'].dt.tz_localize(None)  # Keep as naive UTC for compatibility
    
    # Add timezone metadata for reference
    df.attrs['timezone_info'] = {
        'database_timezone': 'UTC',
        'user_timezone': user_timezone,
        'conversion_applied': True
    }
    
    # Handle 15-minute resampling if needed
    if timeframe == 15 and source_tf == 5:
        df.set_index("start_time", inplace=True)
        
        # Resample to 15-minute bars
        df_resampled = df.resample("15min", label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'ltv': 'sum'  # Sum volume over the resampled period
        }).dropna().reset_index()
        
        # Reapply timezone conversions after resampling
        df_resampled['start_time_utc'] = df_resampled['start_time'].dt.tz_localize('UTC')
        df_resampled['start_time_local'] = df_resampled['start_time_utc'].dt.tz_convert(user_timezone)
        df_resampled['start_time'] = df_resampled['start_time_utc'].dt.tz_localize(None)
        
        # Preserve timezone metadata
        df_resampled.attrs = df.attrs
        
        return df_resampled.reset_index(drop=True)
    
    return df.reset_index(drop=True)

def get_timezone_info(df):
    """
    Get timezone information from enhanced dataframe
    """
    if hasattr(df, 'attrs') and 'timezone_info' in df.attrs:
        info = df.attrs['timezone_info']
        print(f"📊 Timezone Info:")
        print(f"Database: {info['database_timezone']}")
        print(f"Your timezone: {info['user_timezone']}")
        print(f"Conversion applied: {info['conversion_applied']}")
        
        if len(df) > 0:
            latest = df.iloc[-1]
            print(f"Latest candle:")
            print(f"  UTC: {latest['start_time']}")
            if 'start_time_local' in df.columns:
                print(f"  Local: {latest['start_time_local'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        print("No timezone information available")

def display_sample_with_timezones(df, num_rows=3):
    """
    Display sample data with both UTC and local timestamps
    """
    if df.empty:
        print("No data to display")
        return
    
    print(f"\n📊 Sample Data (Last {num_rows} rows):")
    print("=" * 80)
    
    sample = df.tail(num_rows)
    for i, (idx, row) in enumerate(sample.iterrows()):
        print(f"\nRow {i+1}:")
        print(f"UTC Time: {row['start_time']}")
        if 'start_time_local' in df.columns:
            print(f"Local Time: {row['start_time_local'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"OHLC: O:{row['open']:.5f} H:{row['high']:.5f} L:{row['low']:.5f} C:{row['close']:.5f}")
        if 'ltv' in row and pd.notna(row['ltv']):
            print(f"Volume: {row['ltv']}")
    
    print("-" * 80)

# Enhanced function for timezone-aware cutoff times
def get_utc_cutoff_time(local_time, local_timezone='Europe/Stockholm'):
    """
    Convert local cutoff time to UTC for database queries
    
    Args:
        local_time: datetime in local timezone
        local_timezone: Your timezone string
    
    Returns:
        datetime in UTC (naive, for database comparison)
    """
    local_tz = pytz.timezone(local_timezone)
    
    # Ensure the time is timezone-aware
    if local_time.tzinfo is None:
        local_aware = local_tz.localize(local_time)
    else:
        local_aware = local_time
    
    # Convert to UTC and make naive for database comparison
    utc_time = local_aware.astimezone(pytz.UTC).replace(tzinfo=None)
    
    return utc_time