#!/usr/bin/env python3
"""
Diagnose Signal Alignment Issues
Compare backtest signals with actual IG chart data to identify misalignment causes
"""

import sys
import os
import asyncio
import psycopg2
from datetime import datetime, timedelta, timezone
import logging

# Add paths
sys.path.insert(0, '/app')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalAlignmentDiagnostic:
    def __init__(self):
        self.db_config = {
            "host": "postgres",
            "database": "forex", 
            "user": "postgres",
            "password": "postgres"
        }
    
    def get_recent_signals(self, epic: str = "CS.D.EURUSD.MINI.IP", hours_back: int = 48):
        """Get recent signals from trade_log for analysis"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Get recent signals with strategy information
            query = """
                SELECT 
                    trade_time, 
                    epic,
                    signal_type,
                    strategy,
                    entry_price,
                    confidence,
                    timeframe,
                    status
                FROM trade_log 
                WHERE epic = %s 
                AND trade_time >= NOW() - INTERVAL '%s hours'
                ORDER BY trade_time DESC 
                LIMIT 20
            """
            
            cur.execute(query, (epic, hours_back))
            rows = cur.fetchall()
            
            logger.info(f"✅ Found {len(rows)} recent signals for {epic}")
            
            signals = []
            for row in rows:
                signals.append({
                    'trade_time': row[0],
                    'epic': row[1], 
                    'signal_type': row[2],
                    'strategy': row[3],
                    'entry_price': row[4],
                    'confidence': row[5],
                    'timeframe': row[6],
                    'status': row[7]
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error fetching signals: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_candles_around_signal(self, signal_time: datetime, epic: str, timeframe: int = 5, window_minutes: int = 60):
        """Get candles around signal time for analysis"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Get candles in time window around signal
            start_time = signal_time - timedelta(minutes=window_minutes)
            end_time = signal_time + timedelta(minutes=window_minutes)
            
            query = """
                SELECT 
                    start_time,
                    open, high, low, close,
                    quality_score
                FROM ig_candles 
                WHERE epic = %s 
                AND timeframe = %s
                AND start_time >= %s
                AND start_time <= %s
                ORDER BY start_time
            """
            
            cur.execute(query, (epic, timeframe, start_time, end_time))
            rows = cur.fetchall()
            
            candles = []
            for row in rows:
                candles.append({
                    'start_time': row[0],
                    'open': row[1],
                    'high': row[2], 
                    'low': row[3],
                    'close': row[4],
                    'quality_score': row[5]
                })
            
            return candles
            
        except Exception as e:
            logger.error(f"❌ Error fetching candles: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def analyze_timezone_alignment(self):
        """Analyze timezone handling in database vs expected chart times"""
        logger.info("\\n" + "="*80)
        logger.info("🕐 TIMEZONE ALIGNMENT ANALYSIS")
        logger.info("="*80)
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Check database timezone settings
            cur.execute("SELECT now(), timezone('UTC', now()), timezone('Europe/Stockholm', now());")
            db_time, utc_time, stockholm_time = cur.fetchone()
            
            logger.info(f"Database local time: {db_time}")
            logger.info(f"Database UTC time: {utc_time}")
            logger.info(f"Stockholm time: {stockholm_time}")
            
            # Check recent candle timestamps vs current time
            cur.execute("""
                SELECT start_time, 
                       NOW() - start_time as time_diff,
                       timezone('Europe/Stockholm', start_time) as stockholm_time
                FROM ig_candles 
                WHERE epic = 'CS.D.EURUSD.MINI.IP' AND timeframe = 5
                ORDER BY start_time DESC LIMIT 5
            """)
            
            logger.info("\\n📊 RECENT CANDLE TIMESTAMPS:")
            logger.info("    UTC Time            | Stockholm Time      | Age")
            logger.info("    " + "-"*60)
            
            for row in cur.fetchall():
                utc_time, age, stockholm = row
                logger.info(f"    {utc_time} | {stockholm} | {age}")
                
            # IG typically uses UK time (UTC+0/UTC+1 depending on DST)
            # Most charts show UK time, so we need to check alignment
            logger.info("\\n🇬🇧 IG CHART TIME CONSIDERATIONS:")
            logger.info("   - IG charts typically display UK time (UTC+0 winter, UTC+1 summer)")
            logger.info("   - Current UK time would be UTC+1 (BST)")
            logger.info("   - Our database uses UTC")
            logger.info("   - Signals should account for this when comparing to charts")
            
        except Exception as e:
            logger.error(f"❌ Timezone analysis error: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def synthesize_15m_for_comparison(self, start_time: datetime, epic: str):
        """Synthesize 15m candles like the strategy does for comparison"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Get 5m candles for synthesis
            end_time = start_time + timedelta(hours=2)
            query = """
                SELECT start_time, open, high, low, close, quality_score
                FROM ig_candles
                WHERE epic = %s AND timeframe = 5
                AND start_time >= %s AND start_time <= %s
                ORDER BY start_time
            """
            
            cur.execute(query, (epic, start_time - timedelta(hours=1), end_time))
            rows = cur.fetchall()
            
            if len(rows) < 3:
                logger.warning(f"Insufficient 5m candles for 15m synthesis around {start_time}")
                return None
            
            # Simple 15m synthesis (3 consecutive 5m candles)
            synthesized = []
            for i in range(0, len(rows) - 2, 3):
                group = rows[i:i+3]
                if len(group) == 3:
                    # OHLC synthesis
                    open_price = group[0][1]  # First open
                    close_price = group[2][4]  # Last close
                    high_price = max(row[2] for row in group)  # Max high
                    low_price = min(row[3] for row in group)  # Min low
                    avg_quality = sum(row[5] for row in group) / 3
                    
                    synthesized.append({
                        'start_time': group[0][0],  # Use first 5m timestamp
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'quality': avg_quality
                    })
            
            return synthesized
            
        except Exception as e:
            logger.error(f"❌ Error synthesizing 15m candles: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def diagnose_signal_alignment(self, signal):
        """Detailed diagnosis of a specific signal's alignment"""
        signal_time = signal['trade_time']
        epic = signal['epic']
        
        logger.info(f"\\n🔍 ANALYZING SIGNAL: {signal['strategy']} {signal['signal_type']} at {signal_time}")
        
        # Get 5m candles around signal time
        candles_5m = self.get_candles_around_signal(signal_time, epic, timeframe=5, window_minutes=60)
        
        # Synthesize 15m candles around signal time  
        synthesized_15m = self.synthesize_15m_for_comparison(signal_time - timedelta(minutes=30), epic)
        
        if candles_5m:
            logger.info(f"\\n📊 5M CANDLES AROUND SIGNAL ({len(candles_5m)} candles):")
            logger.info("    Time                | OHLC                                    | Quality")
            logger.info("    " + "-"*80)
            
            # Find closest candle to signal
            closest_candle = None
            min_time_diff = timedelta(hours=1)
            
            for candle in candles_5m[-10:]:  # Last 10 candles
                time_diff = abs(candle['start_time'] - signal_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_candle = candle
                
                marker = "🎯" if time_diff < timedelta(minutes=10) else "  "
                logger.info(f"  {marker} {candle['start_time']} | O:{candle['open']:.5f} H:{candle['high']:.5f} L:{candle['low']:.5f} C:{candle['close']:.5f} | {candle['quality_score']:.3f}")
        
        if synthesized_15m:
            logger.info(f"\\n📈 SYNTHESIZED 15M CANDLES ({len(synthesized_15m)} candles):")
            logger.info("    Time                | OHLC                                    | Quality")
            logger.info("    " + "-"*80)
            
            for candle in synthesized_15m[-5:]:  # Last 5 candles
                time_diff = abs(candle['start_time'] - signal_time)
                marker = "🎯" if time_diff < timedelta(minutes=20) else "  "
                logger.info(f"  {marker} {candle['start_time']} | O:{candle['open']:.5f} H:{candle['high']:.5f} L:{candle['low']:.5f} C:{candle['close']:.5f} | {candle['quality']:.3f}")
        
        # Check for potential issues
        logger.info(f"\\n🔍 POTENTIAL ALIGNMENT ISSUES:")
        
        # Issue 1: Timezone misalignment
        if signal_time.tzinfo is None:
            logger.info("   ⚠️ Signal timestamp is timezone-naive (assumes UTC)")
        else:
            logger.info(f"   ✅ Signal timezone: {signal_time.tzinfo}")
        
        # Issue 2: Timeframe mismatch
        if signal.get('timeframe') == '5m' and closest_candle:
            time_diff = abs(closest_candle['start_time'] - signal_time)
            if time_diff > timedelta(minutes=5):
                logger.info(f"   ⚠️ Closest 5m candle is {time_diff} away from signal time")
        
        # Issue 3: Data quality
        if closest_candle and closest_candle['quality_score'] < 0.8:
            logger.info(f"   ⚠️ Low quality data around signal time: {closest_candle['quality_score']:.3f}")
        
        # Issue 4: IG chart time difference
        uk_time = signal_time + timedelta(hours=1)  # BST offset
        logger.info(f"   📍 Signal time in UK/IG time: {uk_time}")
        logger.info("   💡 When comparing to IG charts, use UK time (UTC+1 during BST)")
        
        return {
            'signal': signal,
            'closest_candle': closest_candle,
            'candles_5m': len(candles_5m),
            'synthesized_15m': len(synthesized_15m) if synthesized_15m else 0,
            'uk_time': uk_time
        }

def main():
    """Main diagnostic function"""
    diagnostic = SignalAlignmentDiagnostic()
    
    logger.info("🔍 SIGNAL ALIGNMENT DIAGNOSTIC STARTING")
    logger.info("="*80)
    
    # Analyze timezone setup
    diagnostic.analyze_timezone_alignment()
    
    # Get recent signals
    signals = diagnostic.get_recent_signals(hours_back=72)
    
    if not signals:
        logger.warning("⚠️ No recent signals found to analyze")
        return
    
    logger.info(f"\\n📊 ANALYZING {len(signals)} RECENT SIGNALS")
    logger.info("="*80)
    
    # Diagnose first few signals in detail
    for i, signal in enumerate(signals[:3]):
        try:
            diagnostic.diagnose_signal_alignment(signal)
        except Exception as e:
            logger.error(f"❌ Error analyzing signal {i+1}: {e}")
    
    # Summary recommendations
    logger.info("\\n" + "="*80)
    logger.info("💡 RECOMMENDATIONS FOR SIGNAL ALIGNMENT")
    logger.info("="*80)
    
    logger.info("1. 🕐 TIMEZONE HANDLING:")
    logger.info("   - Database stores UTC timestamps")
    logger.info("   - IG charts typically show UK time (UTC+1 during BST)")
    logger.info("   - Add 1 hour to signal times when comparing to IG charts")
    
    logger.info("\\n2. 📊 TIMEFRAME ALIGNMENT:")
    logger.info("   - Ensure backtest uses same timeframe as live signals (15m)")
    logger.info("   - Verify 15m synthesis produces consistent results")
    
    logger.info("\\n3. 🔍 DATA QUALITY:")
    logger.info("   - Check quality_score of candles around signal times")
    logger.info("   - Investigate any gaps or low-quality periods")
    
    logger.info("\\n4. 📈 CHART COMPARISON:")
    logger.info("   - When checking IG charts, convert UTC signal time to UK time")
    logger.info("   - Look at 15m timeframe charts (not 5m) for MACD signals")
    logger.info("   - Account for bid/ask spread differences")
    
    logger.info("\\n✅ Diagnostic complete!")

if __name__ == "__main__":
    main()