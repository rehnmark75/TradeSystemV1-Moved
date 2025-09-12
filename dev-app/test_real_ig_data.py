#!/usr/bin/env python3
"""
Synthesize 15-minute candles from 5-minute candle data
"""

import sys
sys.path.append('/app')

from services.db import SessionLocal
from services.models import IGCandle
from datetime import datetime, timedelta
from typing import Optional

def synthesize_15min_candle(db, symbol: str, target_time: datetime) -> Optional[dict]:
    """
    Synthesize a 15-minute candle from three 5-minute candles
    
    Args:
        db: Database session
        symbol: Trading symbol
        target_time: The 15-minute boundary time (e.g., 13:15, 13:30, 13:45, 14:00)
    
    Returns:
        Dictionary with OHLC data or None if insufficient data
    """
    
    # Get the three 5-minute candles that make up this 15-minute period
    # For 13:15 target, we need candles: 13:00, 13:05, 13:10
    start_time = target_time - timedelta(minutes=15)
    
    candles_5min = (db.query(IGCandle)
                   .filter(IGCandle.epic == symbol,
                           IGCandle.timeframe == 5,
                           IGCandle.start_time > start_time,
                           IGCandle.start_time <= target_time)
                   .order_by(IGCandle.start_time)
                   .all())
    
    if len(candles_5min) < 3:
        return None  # Not enough data to synthesize
    
    # Take only the last 3 candles (in case there are more)
    candles_5min = candles_5min[-3:]
    
    # Synthesize OHLC
    synthesized = {
        'start_time': candles_5min[0].start_time,
        'end_time': target_time,
        'open': candles_5min[0].open,  # Open of first candle
        'high': max(candle.high for candle in candles_5min),  # Highest high
        'low': min(candle.low for candle in candles_5min),    # Lowest low
        'close': candles_5min[-1].close,  # Close of last candle
        'volume': sum(candle.volume for candle in candles_5min),  # Sum of volumes
        'timeframe': 15
    }
    
    return synthesized

def get_latest_15min_candle(db, symbol: str) -> Optional[dict]:
    """
    Get the latest complete 15-minute candle by synthesizing from 5-minute data
    """
    
    # Find the latest 15-minute boundary
    now = datetime.utcnow()
    
    # Round down to nearest 15-minute boundary
    minutes = now.minute
    if minutes >= 45:
        latest_15min_boundary = now.replace(minute=45, second=0, microsecond=0)
    elif minutes >= 30:
        latest_15min_boundary = now.replace(minute=30, second=0, microsecond=0)
    elif minutes >= 15:
        latest_15min_boundary = now.replace(minute=15, second=0, microsecond=0)
    else:
        latest_15min_boundary = now.replace(minute=0, second=0, microsecond=0)
    
    # Try current boundary first, then previous if not enough data
    for i in range(3):  # Try current and 2 previous boundaries
        boundary = latest_15min_boundary - timedelta(minutes=15 * i)
        candle = synthesize_15min_candle(db, symbol, boundary)
        if candle:
            return candle
    
    return None

def test_15min_synthesis():
    """Test the 15-minute candle synthesis"""
    
    with SessionLocal() as db:
        print("=== 15-MINUTE CANDLE SYNTHESIS TEST ===")
        
        from config import DEFAULT_TEST_EPIC
        symbol = DEFAULT_TEST_EPIC
        
        # Get latest synthesized 15-min candle
        latest_15min = get_latest_15min_candle(db, symbol)
        
        if latest_15min:
            print("✅ Successfully synthesized 15-minute candle!")
            print(f"Time: {latest_15min['start_time']} to {latest_15min['end_time']}")
            print(f"OHLC: O={latest_15min['open']:.5f}, H={latest_15min['high']:.5f}, L={latest_15min['low']:.5f}, C={latest_15min['close']:.5f}")
            print(f"Volume: {latest_15min['volume']}")
            
            # Compare with manual check
            current_price = 144.972
            print(f"\nReal current price: {current_price}")
            print(f"15min candle close: {latest_15min['close']:.5f}")
            print(f"Difference: {abs(current_price - latest_15min['close']):.5f}")
            
            # Show age
            now = datetime.utcnow()
            age_minutes = (now - latest_15min['end_time']).total_seconds() / 60
            print(f"Candle age: {age_minutes:.1f} minutes")
            
            if age_minutes < 15:
                print("✅ Within current 15-minute period")
            else:
                print("⚠️  From previous 15-minute period")
                
        else:
            print("❌ Could not synthesize 15-minute candle")
            
            # Show available 5-minute data
            recent_5min = (db.query(IGCandle)
                          .filter(IGCandle.epic == symbol, IGCandle.timeframe == 5)
                          .order_by(IGCandle.start_time.desc())
                          .limit(5)
                          .all())
            
            print(f"\nAvailable 5-minute candles:")
            for candle in recent_5min:
                print(f"  {candle.start_time}: {candle.close:.5f}")

if __name__ == "__main__":
    test_15min_synthesis()