#!/usr/bin/env python3
"""
Investigate missing MACD signal for EURJPY at 05:15 UTC+2 (2025-08-28)
"""

import sys
sys.path.append('/app/forex_scanner')

from datetime import datetime, timedelta
import pandas as pd
import logging
from core.database import DatabaseManager
from core.strategies.macd_strategy import MACDStrategy
from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_missing_signal():
    """Investigate the missing EURJPY signal at 05:15 UTC+2"""
    
    print("🔍 INVESTIGATING MISSING EURJPY MACD SIGNAL")
    print("=" * 60)
    print(f"Time: 05:15 UTC+2 (03:15 UTC) on 2025-08-28")
    print(f"Pair: CS.D.EURJPY.MINI.IP") 
    print(f"Expected: MACD crossover with histogram -0.009")
    print(f"Price: ~5 pips below EMA200")
    print(f"RSI/ADX: Both DISABLED")
    
    # Initialize components
    db_manager = DatabaseManager(config.DATABASE_URL)
    
    # Target time: 05:15 UTC+2 = 03:15 UTC
    target_time_utc = datetime(2025, 8, 28, 3, 15, 0)
    target_time_local = datetime(2025, 8, 28, 5, 15, 0)  # UTC+2
    
    print(f"\nTarget times:")
    print(f"  UTC: {target_time_utc}")
    print(f"  Local (UTC+2): {target_time_local}")
    
    # Get data around that time (±30 minutes)
    start_time = target_time_utc - timedelta(minutes=30)
    end_time = target_time_utc + timedelta(minutes=30)
    
    query = """
    SELECT * FROM candles 
    WHERE epic = :epic 
    AND timeframe = :timeframe
    AND start_time >= :start_time 
    AND start_time <= :end_time
    ORDER BY start_time
    """
    
    epic = "CS.D.EURJPY.MINI.IP"
    timeframe = 15  # Database stores timeframe as integer (15 for 15m)
    
    try:
        df = db_manager.execute_query(query, {
            'epic': epic,
            'timeframe': timeframe, 
            'start_time': start_time,
            'end_time': end_time
        })
        
        if df is None or len(df) == 0:
            print(f"\n❌ No {timeframe} data found for {epic} around target time")
            return
            
        print(f"\n✅ Found {len(df)} candles around target time")
        
        # Debug: Show column names
        print(f"Available columns: {list(df.columns)}")
        
        # Convert timestamp
        df['start_time'] = pd.to_datetime(df['start_time'])
        df = df.set_index('start_time')
        
        # Display raw data around target time
        print(f"\n📊 Raw Candle Data:")
        for idx, row in df.iterrows():
            local_time = idx + timedelta(hours=2)  # Convert to UTC+2
            print(f"  {local_time.strftime('%H:%M')} UTC+2: O={row['open_price']:.3f} H={row['high_price']:.3f} L={row['low_price']:.3f} C={row['close_price']:.3f}")
        
        # Initialize MACD strategy for signal detection
        strategy = MACDStrategy()
        
        # Get extended data for proper indicator calculation (need more history)
        extended_start = target_time_utc - timedelta(hours=6)  # 6 hours of history
        
        extended_df = db_manager.execute_query(query, {
            'epic': epic,
            'timeframe': timeframe,
            'start_time': extended_start,
            'end_time': end_time
        })
        
        if extended_df is None or len(extended_df) < 50:
            print(f"\n❌ Insufficient extended data for indicator calculation")
            return
            
        extended_df['start_time'] = pd.to_datetime(extended_df['start_time'])
        extended_df = extended_df.set_index('start_time')
        
        print(f"\n✅ Extended dataset: {len(extended_df)} candles for indicator calculation")
        
        # Calculate indicators
        enhanced_df = strategy.data_helper.ensure_macd_indicators(extended_df)
        
        # Find the specific target candle
        target_candles = enhanced_df[
            (enhanced_df.index >= target_time_utc - timedelta(minutes=15)) & 
            (enhanced_df.index <= target_time_utc + timedelta(minutes=15))
        ]
        
        print(f"\n🎯 ANALYZING TARGET PERIOD:")
        print("-" * 40)
        
        for idx, row in target_candles.iterrows():
            local_time = idx + timedelta(hours=2)
            macd_hist = row.get('macd_histogram', 0)
            macd_hist_prev = row.get('macd_histogram_prev', 0) 
            ema200 = row.get('ema_200', 0)
            close_price = row['close_price']
            
            # Calculate distance from EMA200 in pips
            if ema200 > 0:
                distance_pips = (close_price - ema200) / 0.01  # EURJPY pip size is 0.01
            else:
                distance_pips = 0
                
            crossover_type = "NONE"
            if macd_hist_prev <= 0 and macd_hist > 0:
                crossover_type = "BULL"
            elif macd_hist_prev >= 0 and macd_hist < 0:
                crossover_type = "BEAR"
                
            print(f"\n⏰ {local_time.strftime('%H:%M')} UTC+2 ({idx.strftime('%H:%M')} UTC):")
            print(f"   Close: {close_price:.3f}")
            print(f"   EMA200: {ema200:.3f}")
            print(f"   Distance: {distance_pips:.1f} pips {'below' if distance_pips < 0 else 'above'} EMA200")
            print(f"   MACD Histogram: {macd_hist:.6f} (prev: {macd_hist_prev:.6f})")
            print(f"   Crossover: {crossover_type}")
            
            # Check if this is around our target time
            if abs((local_time - target_time_local).total_seconds()) <= 900:  # Within 15 minutes
                print(f"   🎯 TARGET CANDLE IDENTIFIED!")
                
                # Run signal detection on this specific candle
                print(f"\n🔍 RUNNING SIGNAL DETECTION:")
                print("-" * 30)
                
                # Get the slice up to this point for signal detection
                signal_df = enhanced_df.loc[:idx].copy()
                
                if len(signal_df) >= 2:
                    latest = signal_df.iloc[-1]
                    previous = signal_df.iloc[-2]
                    
                    print(f"Signal detection inputs:")
                    print(f"  Latest histogram: {latest.get('macd_histogram', 0):.6f}")
                    print(f"  Previous histogram: {previous.get('macd_histogram', 0):.6f}")
                    print(f"  EMA200: {latest.get('ema_200', 0):.3f}")
                    print(f"  Close: {latest['close_price']:.3f}")
                    
                    # Test signal detection
                    try:
                        signal = strategy.detect_signal(signal_df, epic, 1.5, timeframe)
                        
                        if signal:
                            print(f"  ✅ SIGNAL DETECTED: {signal.get('signal_type', 'Unknown')}")
                            print(f"     Confidence: {signal.get('confidence_score', 0):.1%}")
                            print(f"     Trigger: {signal.get('trigger_reason', 'Unknown')}")
                        else:
                            print(f"  ❌ NO SIGNAL DETECTED")
                            
                            # Detailed analysis of why signal was rejected
                            print(f"\n🔍 DETAILED REJECTION ANALYSIS:")
                            
                            # Check each filter step-by-step
                            signal_detector = MACDSignalDetector()
                            
                            # Test enhanced detection with full logging
                            enhanced_signal = signal_detector.detect_enhanced_macd_signal(
                                latest=latest,
                                previous=previous, 
                                epic=epic,
                                timeframe=timeframe,
                                df_enhanced=signal_df,
                                forex_optimizer=strategy.forex_optimizer
                            )
                            
                            if not enhanced_signal:
                                print(f"     Signal rejected at enhanced detection level")
                                
                                # Manual threshold check
                                current_hist = latest.get('macd_histogram', 0)
                                prev_hist = previous.get('macd_histogram', 0)
                                
                                # Get expected threshold
                                if strategy.forex_optimizer:
                                    threshold = strategy.forex_optimizer.get_macd_threshold_for_epic(epic)
                                else:
                                    threshold = 0.008  # Default for JPY pairs
                                    
                                crossover_strength = abs(current_hist - prev_hist)
                                
                                print(f"     Crossover strength: {crossover_strength:.6f}")
                                print(f"     Required threshold: {threshold:.6f}")
                                print(f"     Threshold met: {'✅ YES' if crossover_strength >= threshold else '❌ NO'}")
                                
                                if crossover_strength < threshold:
                                    print(f"     🚫 REJECTION REASON: Crossover too weak ({crossover_strength:.6f} < {threshold:.6f})")
                                
                    except Exception as e:
                        print(f"  ❌ ERROR during signal detection: {e}")
                        import traceback
                        traceback.print_exc()
                        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_missing_signal()