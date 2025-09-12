#!/usr/bin/env python3
"""
Candle Size Debug Script
Debug ATR calculation and candle size detection issues

Usage:
    python candle_debug.py investigate --pair EURUSD --timestamp "2025-08-12 13:40"
    python candle_debug.py atr-analysis --pair EURUSD
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'forex_scanner' in current_dir else current_dir
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

try:
    from core.detection.large_candle_filter import LargeCandleFilter
    from core.data_fetcher import DataFetcher
    from core.database import DatabaseManager
    import config
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    MODULES_AVAILABLE = False


class CandleDebugger:
    """Debug candle size calculations and ATR"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        if not MODULES_AVAILABLE:
            return
        
        try:
            self.db_manager = DatabaseManager(config.DATABASE_URL)
            self.data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
            self.filter = LargeCandleFilter()
            print("✅ Candle debugger initialized")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    def investigate_specific_candle(self, args):
        """Investigate a specific candle that was flagged as large"""
        
        epic_map = {
            "EURUSD": "CS.D.EURUSD.MINI.IP",
            "GBPUSD": "CS.D.GBPUSD.MINI.IP", 
            "USDJPY": "CS.D.USDJPY.MINI.IP"
        }
        
        epic = epic_map.get(args.pair, f"CS.D.{args.pair}.MINI.IP")
        target_time = datetime.strptime(args.timestamp, "%Y-%m-%d %H:%M")
        
        print(f"\n🔍 CANDLE INVESTIGATION")
        print("=" * 50)
        print(f"Target: {args.timestamp} UTC ({args.pair})")
        print(f"Epic: {epic}")
        print("=" * 50)
        
        try:
            # Get extended data for proper ATR calculation
            df = self.data_fetcher.get_enhanced_data(epic, args.pair, '5m', 200)
            if df is None or len(df) < 50:
                print("❌ Insufficient data")
                return 1
            
            # Find the target candle
            df['start_time_dt'] = pd.to_datetime(df['start_time']).dt.tz_localize(None)
            target_candles = df[df['start_time_dt'] == target_time]
            
            if target_candles.empty:
                print(f"❌ Candle not found at {target_time}")
                # Show nearby candles
                print(f"\n📋 Available candles around that time:")
                nearby_start = target_time - timedelta(minutes=30)
                nearby_end = target_time + timedelta(minutes=30)
                nearby = df[(df['start_time_dt'] >= nearby_start) & (df['start_time_dt'] <= nearby_end)]
                for _, candle in nearby.iterrows():
                    print(f"   {candle['start_time_dt']}: {candle['close']:.5f}")
                return 1
            
            target_candle = target_candles.iloc[0]
            target_index = target_candles.index[0]
            
            print(f"✅ Found target candle at index {target_index}")
            
            # Calculate ATR multiple ways to see what's wrong
            print(f"\n📊 ATR CALCULATION ANALYSIS")
            
            # Method 1: Filter's ATR calculation
            filter_atr = self.filter._calculate_atr(df.iloc[:target_index+1])
            print(f"   Filter ATR (14-period): {filter_atr:.7f}")
            
            # Method 2: Manual ATR calculation 
            manual_atr = self._calculate_atr_manual(df.iloc[:target_index+1])
            print(f"   Manual ATR (14-period): {manual_atr:.7f}")
            
            # Method 3: DataFrame ATR column if available
            if 'atr' in df.columns:
                df_atr = df.iloc[target_index]['atr']
                print(f"   DataFrame ATR: {df_atr:.7f}")
            else:
                print(f"   DataFrame ATR: Not available")
            
            # Method 4: Rolling ATR with different periods
            for period in [7, 14, 20]:
                rolling_atr = self._calculate_atr_manual(df.iloc[:target_index+1], period)
                print(f"   ATR ({period}-period): {rolling_atr:.7f}")
            
            # Analyze the target candle
            candle_range = target_candle['high'] - target_candle['low']
            candle_body = abs(target_candle['close'] - target_candle['open'])
            candle_upper_wick = target_candle['high'] - max(target_candle['open'], target_candle['close'])
            candle_lower_wick = min(target_candle['open'], target_candle['close']) - target_candle['low']
            
            print(f"\n🕯️  TARGET CANDLE DETAILS")
            print(f"   Timestamp: {target_candle['start_time']}")
            print(f"   Open:  {target_candle['open']:.5f}")
            print(f"   High:  {target_candle['high']:.5f}")
            print(f"   Low:   {target_candle['low']:.5f}")
            print(f"   Close: {target_candle['close']:.5f}")
            print(f"   Direction: {'BULL' if target_candle['close'] > target_candle['open'] else 'BEAR'}")
            
            pip_multiplier = 10000 if 'JPY' not in epic.upper() else 100
            
            print(f"\n📏 SIZE MEASUREMENTS")
            print(f"   Total range: {candle_range:.5f} ({candle_range * pip_multiplier:.1f} pips)")
            print(f"   Body size:   {candle_body:.5f} ({candle_body * pip_multiplier:.1f} pips)")
            print(f"   Upper wick:  {candle_upper_wick:.5f} ({candle_upper_wick * pip_multiplier:.1f} pips)")
            print(f"   Lower wick:  {candle_lower_wick:.5f} ({candle_lower_wick * pip_multiplier:.1f} pips)")
            print(f"   Body %: {(candle_body / candle_range * 100) if candle_range > 0 else 0:.1f}%")
            
            # Calculate ATR ratios
            print(f"\n🔢 ATR RATIO ANALYSIS")
            if filter_atr > 0:
                filter_ratio = candle_range / filter_atr
                print(f"   Range / Filter ATR: {filter_ratio:.1f}x")
            
            if manual_atr > 0:
                manual_ratio = candle_range / manual_atr
                print(f"   Range / Manual ATR: {manual_ratio:.1f}x")
            
            # Compare with recent candles for context
            print(f"\n📊 RECENT CANDLES COMPARISON")
            recent_candles = df.iloc[target_index-5:target_index+6]  # 5 before, target, 5 after
            
            print(f"   {'Index':<6} {'Time':<6} {'Range (pips)':<12} {'Body (pips)':<12} {'ATR Ratio':<10}")
            print(f"   {'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
            
            for idx, candle in recent_candles.iterrows():
                candle_rng = candle['high'] - candle['low']
                candle_bdy = abs(candle['close'] - candle['open'])
                atr_ratio = candle_rng / filter_atr if filter_atr > 0 else 0
                
                marker = " ← TARGET" if idx == target_index else ""
                time_str = candle['start_time_dt'].strftime('%H:%M') if 'start_time_dt' in candle else "??:??"
                
                print(f"   {idx:<6} {time_str:<6} {candle_rng * pip_multiplier:>10.1f}p {candle_bdy * pip_multiplier:>10.1f}p {atr_ratio:>8.1f}x{marker}")
            
            # Analyze ATR context
            print(f"\n🧮 ATR CONTEXT ANALYSIS")
            
            # Get the 14 candles used for ATR calculation
            atr_start = max(0, target_index - 13)
            atr_candles = df.iloc[atr_start:target_index+1]
            
            ranges = []
            for _, candle in atr_candles.iterrows():
                tr1 = candle['high'] - candle['low']
                tr2 = abs(candle['high'] - candle.get('prev_close', candle['close']))
                tr3 = abs(candle['low'] - candle.get('prev_close', candle['close']))
                true_range = max(tr1, tr2, tr3)
                ranges.append(true_range)
            
            avg_range = np.mean(ranges)
            range_std = np.std(ranges)
            
            print(f"   ATR period candles: {len(ranges)}")
            print(f"   Average range: {avg_range:.7f} ({avg_range * pip_multiplier:.1f} pips)")
            print(f"   Range std dev: {range_std:.7f} ({range_std * pip_multiplier:.1f} pips)")
            print(f"   Target range: {candle_range:.7f} ({candle_range * pip_multiplier:.1f} pips)")
            print(f"   Std deviations: {(candle_range - avg_range) / range_std if range_std > 0 else 0:.1f} σ")
            
            # Conclusion
            print(f"\n🎯 ANALYSIS CONCLUSION")
            if filter_atr < 0.0001:  # Very small ATR
                print(f"   ⚠️  ATR appears too small: {filter_atr:.7f}")
                print(f"   This makes normal candles appear 'large'")
                print(f"   Recommendation: Check ATR calculation or increase threshold")
            elif candle_range * pip_multiplier < 20:  # Small candle in pips
                print(f"   ✅ Candle is genuinely small: {candle_range * pip_multiplier:.1f} pips")
                print(f"   Filter may be over-sensitive for current market conditions")
            else:
                print(f"   🤔 Mixed signals - need more investigation")
            
            return 0
            
        except Exception as e:
            print(f"❌ Investigation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def analyze_atr_calculation(self, args):
        """Analyze ATR calculation across different periods"""
        
        epic_map = {
            "EURUSD": "CS.D.EURUSD.MINI.IP",
            "GBPUSD": "CS.D.GBPUSD.MINI.IP", 
            "USDJPY": "CS.D.USDJPY.MINI.IP"
        }
        
        epic = epic_map.get(args.pair, f"CS.D.{args.pair}.MINI.IP")
        
        print(f"\n🧮 ATR CALCULATION ANALYSIS")
        print("=" * 40)
        print(f"Pair: {args.pair}")
        print(f"Epic: {epic}")
        print("=" * 40)
        
        try:
            df = self.data_fetcher.get_enhanced_data(epic, args.pair, '5m', 100)
            if df is None or len(df) < 50:
                print("❌ Insufficient data")
                return 1
            
            # Calculate ATR with different periods
            print(f"\n📊 ATR COMPARISON (Different Periods)")
            print(f"   {'Period':<8} {'ATR Value':<12} {'ATR (pips)':<12} {'Threshold (pips)':<15}")
            print(f"   {'-'*8} {'-'*12} {'-'*12} {'-'*15}")
            
            pip_multiplier = 10000 if 'JPY' not in epic.upper() else 100
            
            for period in [7, 14, 20, 30]:
                atr_value = self._calculate_atr_manual(df, period)
                atr_pips = atr_value * pip_multiplier
                threshold_pips = atr_value * 2.5 * pip_multiplier
                
                print(f"   {period:<8} {atr_value:<12.7f} {atr_pips:<12.1f} {threshold_pips:<15.1f}")
            
            # Analyze recent volatility
            latest_20 = df.tail(20)
            ranges_pips = [(candle['high'] - candle['low']) * pip_multiplier for _, candle in latest_20.iterrows()]
            
            print(f"\n📈 RECENT VOLATILITY ANALYSIS (Last 20 Candles)")
            print(f"   Average range: {np.mean(ranges_pips):.1f} pips")
            print(f"   Median range:  {np.median(ranges_pips):.1f} pips")
            print(f"   Min range:     {np.min(ranges_pips):.1f} pips")
            print(f"   Max range:     {np.max(ranges_pips):.1f} pips")
            print(f"   Std deviation: {np.std(ranges_pips):.1f} pips")
            
            # Show individual recent candles
            print(f"\n🕯️  INDIVIDUAL CANDLE ANALYSIS")
            print(f"   {'Time':<6} {'Range':<8} {'Body':<8} {'ATR 14x':<8} {'Visual Size':<12}")
            print(f"   {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
            
            atr_14 = self._calculate_atr_manual(df, 14)
            
            for _, candle in latest_20.tail(10).iterrows():
                candle_range = candle['high'] - candle['low']
                candle_body = abs(candle['close'] - candle['open'])
                atr_ratio = candle_range / atr_14 if atr_14 > 0 else 0
                
                range_pips = candle_range * pip_multiplier
                body_pips = candle_body * pip_multiplier
                
                # Classify visual size
                if range_pips < 8:
                    visual_size = "Very Small"
                elif range_pips < 15:
                    visual_size = "Small"
                elif range_pips < 25:
                    visual_size = "Normal"
                elif range_pips < 40:
                    visual_size = "Large"
                else:
                    visual_size = "Very Large"
                
                time_str = pd.to_datetime(candle['start_time']).strftime('%H:%M')
                print(f"   {time_str:<6} {range_pips:>6.1f}p {body_pips:>6.1f}p {atr_ratio:>6.1f}x {visual_size:<12}")
            
            # Check if ATR is too small
            print(f"\n🎯 ATR SENSITIVITY ANALYSIS")
            print(f"   Current ATR (14): {atr_14:.7f} ({atr_14 * pip_multiplier:.1f} pips)")
            print(f"   Large threshold: {atr_14 * 2.5 * pip_multiplier:.1f} pips")
            
            # Compare to typical forex volatility
            typical_atr_pips = {
                'EURUSD': (8, 15),   # Typical range
                'GBPUSD': (12, 25),
                'USDJPY': (10, 20),
                'AUDUSD': (8, 18),
                'USDCAD': (8, 16)
            }
            
            if args.pair in typical_atr_pips:
                min_typical, max_typical = typical_atr_pips[args.pair]
                current_atr_pips = atr_14 * pip_multiplier
                
                print(f"\n📊 MARKET CONTEXT")
                print(f"   Typical {args.pair} ATR: {min_typical}-{max_typical} pips")
                print(f"   Current ATR: {current_atr_pips:.1f} pips")
                
                if current_atr_pips < min_typical:
                    print(f"   ⚠️  ATR is unusually LOW - market very quiet")
                    print(f"   Filter may be over-sensitive in low volatility")
                elif current_atr_pips > max_typical:
                    print(f"   ⚠️  ATR is unusually HIGH - market very volatile") 
                    print(f"   Filter sensitivity appears appropriate")
                else:
                    print(f"   ✅ ATR in normal range for {args.pair}")
            
            # Recommendation
            current_threshold_pips = atr_14 * 2.5 * pip_multiplier
            
            print(f"\n💡 RECOMMENDATIONS")
            if current_threshold_pips < 15:
                print(f"   🔧 Consider increasing LARGE_CANDLE_ATR_MULTIPLIER to 3.0-4.0")
                print(f"   🔧 Or add minimum pip threshold (e.g., min 15 pips regardless of ATR)")
                print(f"   Current threshold too low: {current_threshold_pips:.1f} pips")
            else:
                print(f"   ✅ Threshold appears reasonable: {current_threshold_pips:.1f} pips")
            
            return 0
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _calculate_atr_manual(self, df: pd.DataFrame, period: int = 14) -> float:
        """Manual ATR calculation for comparison"""
        try:
            if len(df) < period + 1:
                return 0.0
            
            # Calculate True Range
            high = df['high']
            low = df['low'] 
            close = df['close']
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 0.0
            
        except Exception as e:
            print(f"❌ Manual ATR calculation failed: {e}")
            return 0.0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Candle Size Debug Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Investigate command
    investigate_parser = subparsers.add_parser('investigate', help='Investigate specific candle')
    investigate_parser.add_argument('--pair', required=True, help='Currency pair')
    investigate_parser.add_argument('--timestamp', required=True, help='Timestamp (YYYY-MM-DD HH:MM)')
    
    # ATR analysis command
    atr_parser = subparsers.add_parser('atr-analysis', help='Analyze ATR calculation')
    atr_parser.add_argument('--pair', required=True, help='Currency pair')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    debugger = CandleDebugger()
    
    if args.command == 'investigate':
        return debugger.investigate_specific_candle(args)
    elif args.command == 'atr-analysis':
        return debugger.analyze_atr_calculation(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())