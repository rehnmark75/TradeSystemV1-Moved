#!/usr/bin/env python3
"""
Large Candle Filter Management Script
A standalone tool for testing, configuring, and monitoring the large candle filter

Usage:
    python filter_manager.py analyze --pair EURUSD
    python filter_manager.py test --signal BULL
    python filter_manager.py configure --preset strict
    python filter_manager.py stats
    python filter_manager.py status
    python filter_manager.py monitor --live
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'forex_scanner' in current_dir else current_dir
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

# Import project modules
try:
    from core.detection.large_candle_filter import LargeCandleFilter
    from core.data_fetcher import DataFetcher
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector
    import config
    
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("Make sure you're running this from the forex_scanner directory")
    MODULES_AVAILABLE = False


class FilterManager:
    """Standalone Large Candle Filter Management Tool"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        if not MODULES_AVAILABLE:
            self.logger.error("❌ Required modules not available")
            return
        
        # Initialize components
        try:
            self.db_manager = DatabaseManager(config.DATABASE_URL)
            self.data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
            self.filter = LargeCandleFilter()
            self.signal_detector = None  # Lazy load if needed
            
            self.logger.info("✅ Filter Manager initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            self.db_manager = None
            self.data_fetcher = None
            self.filter = None
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def analyze_movement(self, args):
        """Analyze recent price movement and large candles"""
        if not self._check_initialization():
            return 1
        
        # Use MINI epics by default (correct format for this system)
        epic_map = {
            "EURUSD": "CS.D.EURUSD.CEEM.IP",
            "GBPUSD": "CS.D.GBPUSD.MINI.IP", 
            "USDJPY": "CS.D.USDJPY.MINI.IP",
            "AUDUSD": "CS.D.AUDUSD.MINI.IP",
            "USDCAD": "CS.D.USDCAD.MINI.IP",
            "EURJPY": "CS.D.EURJPY.MINI.IP",
            "AUDJPY": "CS.D.AUDJPY.MINI.IP",
            "NZDUSD": "CS.D.NZDUSD.MINI.IP",
            "USDCHF": "CS.D.USDCHF.MINI.IP"
        }
        
        # Auto-convert pair to epic if needed
        if args.epic == 'CS.D.EURUSD.CFD.IP':  # Default value, convert to MINI
            epic = epic_map.get(args.pair, args.epic)
        else:
            epic = args.epic
            
        pair = args.pair
        timeframe = args.timeframe
        periods = args.periods
        
        print(f"\n🔍 LARGE CANDLE ANALYSIS")
        print("=" * 50)
        print(f"Pair: {pair}")
        print(f"Epic: {epic}")
        print(f"Timeframe: {timeframe}")
        print(f"Periods: {periods}")
        print("=" * 50)
        
        try:
            # Get data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe, periods + 20)
            if df is None or len(df) < 10:
                print("❌ Insufficient data available")
                return 1
            
            # Calculate ATR for context
            atr = self.filter._calculate_atr(df)
            latest_candles = df.tail(periods)
            
            print(f"\n📊 DATA OVERVIEW")
            print(f"   Total periods analyzed: {len(latest_candles)}")
            print(f"   ATR (14-period): {atr:.5f}")
            print(f"   Large candle threshold: {atr * self.filter.large_candle_multiplier:.5f}")
            print(f"   Filter sensitivity: {self.filter.large_candle_multiplier}x ATR")
            
            # Analyze each candle for large movements
            large_candles = []
            total_movement = 0
            extreme_candles = []
            
            for i, (idx, candle) in enumerate(latest_candles.iterrows()):
                candle_range = candle['high'] - candle['low']
                candle_body = abs(candle['close'] - candle['open'])
                atr_ratio = candle_range / atr if atr > 0 else 0
                
                total_movement += candle_range
                
                if atr_ratio > self.filter.large_candle_multiplier:
                    periods_ago = len(latest_candles) - i - 1
                    
                    # Get timestamp with better formatting
                    timestamp = candle.get('start_time', idx)
                    if hasattr(timestamp, 'strftime'):
                        formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M')
                    else:
                        formatted_timestamp = str(timestamp)
                    
                    candle_data = {
                        'periods_ago': periods_ago,
                        'timestamp': timestamp,
                        'formatted_timestamp': formatted_timestamp,
                        'range': candle_range,
                        'body': candle_body,
                        'atr_ratio': atr_ratio,
                        'direction': 'BULL' if candle['close'] > candle['open'] else 'BEAR',
                        'close': candle['close'],
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'volume': candle.get('ltv', candle.get('volume', 0)),
                        'body_percentage': (candle_body / candle_range * 100) if candle_range > 0 else 0
                    }
                    large_candles.append(candle_data)
                    
                    # Track extreme candles (>4x ATR)
                    if atr_ratio > 4.0:
                        extreme_candles.append(candle_data)
            
            # Report findings
            print(f"\n🔥 LARGE CANDLE DETECTION")
            print(f"   Large candles found: {len(large_candles)} ({len(large_candles)/len(latest_candles)*100:.1f}% of period)")
            print(f"   Extreme candles (>4x ATR): {len(extreme_candles)}")
            
            if large_candles:
                print(f"\n   📋 Recent Large Candles (showing last 15):")
                print(f"   {'Periods':<8} {'Timestamp':<17} {'ATR':<8} {'Dir':<4} {'Price':<9} {'Range':<8} {'Body%':<6}")
                print(f"   {'-'*8} {'-'*17} {'-'*8} {'-'*4} {'-'*9} {'-'*8} {'-'*6}")
                
                for lc in large_candles[-15:]:  # Show last 15
                    body_pct = f"{lc['body_percentage']:.1f}%"
                    range_pips = lc['range'] * (10000 if 'JPY' not in epic.upper() else 100)
                    print(f"   {lc['periods_ago']:>7}  {lc['formatted_timestamp']:<17} {lc['atr_ratio']:>6.1f}x {lc['direction']:<4} {lc['close']:<9.5f} {range_pips:>6.1f}p {body_pct:<6}")
            
            # Show extreme candles separately if there are many
            if len(extreme_candles) > 5:
                print(f"\n   ⚡ EXTREME CANDLES (>4x ATR):")
                print(f"   {'Periods':<8} {'Timestamp':<17} {'ATR':<8} {'Dir':<4} {'Price':<9} {'OHLC Details'}")
                print(f"   {'-'*8} {'-'*17} {'-'*8} {'-'*4} {'-'*9} {'-'*20}")
                
                for lc in extreme_candles[-10:]:  # Show last 10 extreme
                    ohlc = f"O:{lc['open']:.5f} H:{lc['high']:.5f} L:{lc['low']:.5f}"
                    print(f"   {lc['periods_ago']:>7}  {lc['formatted_timestamp']:<17} {lc['atr_ratio']:>6.1f}x {lc['direction']:<4} {lc['close']:<9.5f} {ohlc}")
            
            # Test filter on current conditions
            print(f"\n🚦 CURRENT FILTER STATUS")
            signal_types = ['BULL', 'BEAR']
            for signal_type in signal_types:
                should_block, reason = self.filter.should_block_entry(df, epic, signal_type, timeframe)
                status = "🚫 BLOCKED" if should_block else "✅ ALLOWED"
                print(f"   {signal_type:4s} signals: {status}")
                if reason:
                    print(f"        Reason: {reason}")
            
            # Movement summary
            pip_multiplier = 10000 if 'JPY' not in epic.upper() else 100
            total_pips = total_movement * pip_multiplier
            avg_candle_pips = total_pips / len(latest_candles)
            
            print(f"\n📈 MOVEMENT SUMMARY")
            print(f"   Total price movement: {total_pips:.1f} pips")
            print(f"   Average per candle: {avg_candle_pips:.1f} pips")
            print(f"   Largest single move: {max([lc['atr_ratio'] for lc in large_candles] + [0]):.1f}x ATR")
            
            # Enhanced risk assessment with timestamps
            recent_large = [lc for lc in large_candles if lc['periods_ago'] <= 3]
            if recent_large:
                print(f"\n⚠️  RISK ASSESSMENT")
                print(f"   {len(recent_large)} large candle(s) in last 3 periods:")
                for lc in recent_large:
                    print(f"     • {lc['formatted_timestamp']}: {lc['atr_ratio']:.1f}x ATR {lc['direction']} candle")
                print(f"   Market may be in exhaustion phase")
                print(f"   Recommend: Wait for consolidation before entries")
            else:
                print(f"\n✅ RISK ASSESSMENT")
                print(f"   No recent large candles detected")
                if large_candles:
                    last_large = large_candles[-1]
                    print(f"   Last large candle: {last_large['formatted_timestamp']} ({last_large['periods_ago']} periods ago)")
                print(f"   Market conditions appear stable for entries")
            
            return 0
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return 1
    
    def test_filter(self, args):
        """Test filter on current market conditions"""
        if not self._check_initialization():
            return 1
        
        # Use MINI epics by default (correct format for this system)
        epic_map = {
            "EURUSD": "CS.D.EURUSD.CEEM.IP",
            "GBPUSD": "CS.D.GBPUSD.MINI.IP", 
            "USDJPY": "CS.D.USDJPY.MINI.IP",
            "AUDUSD": "CS.D.AUDUSD.MINI.IP",
            "USDCAD": "CS.D.USDCAD.MINI.IP",
            "EURJPY": "CS.D.EURJPY.MINI.IP",
            "AUDJPY": "CS.D.AUDJPY.MINI.IP",
            "NZDUSD": "CS.D.NZDUSD.MINI.IP",
            "USDCHF": "CS.D.USDCHF.MINI.IP"
        }
        
        # Auto-convert pair to epic if needed
        if args.epic == 'CS.D.EURUSD.CFD.IP':  # Default value, convert to MINI
            epic = epic_map.get(args.pair, args.epic)
        else:
            epic = args.epic
            
        pair = args.pair
        timeframe = args.timeframe
        signal_type = args.signal
        
        print(f"\n🧪 FILTER TEST")
        print("=" * 30)
        print(f"Testing: {signal_type} signal")
        print(f"Pair: {pair}")
        print(f"Timeframe: {timeframe}")
        print("=" * 30)
        
        try:
            # Get data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe, 100)
            if df is None or len(df) < 10:
                print("❌ Insufficient data")
                return 1
            
            # Test the filter
            should_block, reason = self.filter.should_block_entry(df, epic, signal_type, timeframe)
            
            # Show result prominently
            if should_block:
                print(f"\n🚫 SIGNAL WOULD BE BLOCKED")
                print(f"   Reason: {reason}")
                print(f"   Recommendation: Wait for better conditions")
            else:
                print(f"\n✅ SIGNAL WOULD BE ALLOWED")
                print(f"   No blocking conditions detected")
                print(f"   Market conditions suitable for entry")
            
            # Show current market state
            latest = df.iloc[-1]
            atr = self.filter._calculate_atr(df)
            latest_range = latest['high'] - latest['low']
            atr_ratio = latest_range / atr if atr > 0 else 0
            
            print(f"\n📊 CURRENT MARKET STATE")
            print(f"   Current price: {latest['close']:.5f}")
            print(f"   Latest candle: {latest_range:.5f} ({atr_ratio:.1f}x ATR)")
            print(f"   ATR (14): {atr:.5f}")
            print(f"   Candle type: {'BULL' if latest['close'] > latest['open'] else 'BEAR'}")
            
            # Show filter configuration
            print(f"\n⚙️  FILTER CONFIGURATION")
            print(f"   ATR Multiplier: {self.filter.large_candle_multiplier}")
            print(f"   Consecutive Threshold: {self.filter.consecutive_large_threshold}")
            print(f"   Movement Threshold: {self.filter.excessive_movement_threshold} pips")
            print(f"   Cooldown Periods: {self.filter.filter_cooldown_periods}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return 1
    
    def configure_filter(self, args):
        """Configure filter settings"""
        if not self._check_initialization():
            return 1
        
        preset = args.preset
        
        presets = {
            'strict': {
                'large_candle_multiplier': 2.0,
                'consecutive_large_threshold': 1,
                'excessive_movement_threshold': 10,
                'filter_cooldown_periods': 5,
                'description': 'Very conservative - blocks most large movements'
            },
            'balanced': {
                'large_candle_multiplier': 2.5,
                'consecutive_large_threshold': 2,
                'excessive_movement_threshold': 15,
                'filter_cooldown_periods': 3,
                'description': 'Moderate filtering - good balance'
            },
            'permissive': {
                'large_candle_multiplier': 3.0,
                'consecutive_large_threshold': 3,
                'excessive_movement_threshold': 20,
                'filter_cooldown_periods': 2,
                'description': 'Liberal filtering - allows more aggressive entries'
            }
        }
        
        print(f"\n⚙️  FILTER CONFIGURATION")
        print("=" * 40)
        
        if preset not in presets:
            print(f"❌ Unknown preset: {preset}")
            print(f"Available presets: {list(presets.keys())}")
            return 1
        
        config_data = presets[preset]
        
        print(f"Applying '{preset}' preset...")
        print(f"Description: {config_data['description']}")
        print()
        
        # Apply configuration
        for key, value in config_data.items():
            if key != 'description' and hasattr(self.filter, key):
                old_value = getattr(self.filter, key)
                setattr(self.filter, key, value)
                print(f"   {key}: {old_value} → {value}")
        
        print(f"\n✅ Filter configured with '{preset}' preset")
        print(f"\n📝 To make permanent, add to config.py:")
        print(f"LARGE_CANDLE_FILTER_PRESET = '{preset}'")
        
        return 0
    
    def show_statistics(self, args):
        """Show filter statistics"""
        if not self._check_initialization():
            return 1
        
        print(f"\n📊 FILTER STATISTICS")
        print("=" * 30)
        
        stats = self.filter.get_filter_statistics()
        
        if stats['total_signals_checked'] == 0:
            print("📭 No signals processed yet")
            print("   Run some tests or live scanning to generate statistics")
            return 0
        
        total_blocked = (stats['filtered_large_candle'] + 
                        stats['filtered_excessive_movement'] + 
                        stats['filtered_parabolic_move'])
        
        print(f"📈 OVERALL STATISTICS")
        print(f"   Total signals checked: {stats['total_signals_checked']}")
        print(f"   Total signals blocked: {total_blocked}")
        print(f"   Overall filter rate: {total_blocked/stats['total_signals_checked']*100:.1f}%")
        
        print(f"\n🔍 DETAILED BREAKDOWN")
        print(f"   Large candle blocks: {stats['filtered_large_candle']}")
        print(f"   Excessive movement blocks: {stats['filtered_excessive_movement']}")
        print(f"   Parabolic movement blocks: {stats['filtered_parabolic_move']}")
        
        if 'filter_rate' in stats:
            rates = stats['filter_rate']
            print(f"\n📊 FILTER RATES")
            print(f"   Large candle rate: {rates['large_candle_rate']:.1f}%")
            print(f"   Excessive movement rate: {rates['excessive_movement_rate']:.1f}%")
            print(f"   Parabolic rate: {rates['parabolic_movement_rate']:.1f}%")
        
        return 0
    
    def show_status(self, args):
        """Show filter status and configuration"""
        if not self._check_initialization():
            return 1
        
        print(f"\n🔍 FILTER STATUS")
        print("=" * 30)
        
        print(f"📋 SYSTEM STATUS")
        print(f"   Filter available: ✅")
        print(f"   Database connected: {'✅' if self.db_manager else '❌'}")
        print(f"   Data fetcher ready: {'✅' if self.data_fetcher else '❌'}")
        
        print(f"\n⚙️  CURRENT CONFIGURATION")
        print(f"   ATR Multiplier: {self.filter.large_candle_multiplier}")
        print(f"   Consecutive Threshold: {self.filter.consecutive_large_threshold}")
        print(f"   Movement Threshold: {self.filter.excessive_movement_threshold} pips")
        print(f"   Lookback Periods: {self.filter.movement_lookback_periods}")
        print(f"   Cooldown Periods: {self.filter.filter_cooldown_periods}")
        
        # Determine current preset
        current_config = {
            'large_candle_multiplier': self.filter.large_candle_multiplier,
            'consecutive_large_threshold': self.filter.consecutive_large_threshold,
            'excessive_movement_threshold': self.filter.excessive_movement_threshold,
            'filter_cooldown_periods': self.filter.filter_cooldown_periods
        }
        
        presets = {
            'strict': {'large_candle_multiplier': 2.0, 'consecutive_large_threshold': 1, 'excessive_movement_threshold': 10, 'filter_cooldown_periods': 5},
            'balanced': {'large_candle_multiplier': 2.5, 'consecutive_large_threshold': 2, 'excessive_movement_threshold': 15, 'filter_cooldown_periods': 3},
            'permissive': {'large_candle_multiplier': 3.0, 'consecutive_large_threshold': 3, 'excessive_movement_threshold': 20, 'filter_cooldown_periods': 2}
        }
        
        matched_preset = None
        for preset_name, preset_config in presets.items():
            if all(current_config.get(k) == v for k, v in preset_config.items()):
                matched_preset = preset_name
                break
        
        print(f"\n📋 PRESET STATUS")
        if matched_preset:
            print(f"   Current preset: {matched_preset}")
        else:
            print(f"   Current preset: custom")
        
        # Show statistics if available
        stats = self.filter.get_filter_statistics()
        if stats['total_signals_checked'] > 0:
            total_blocked = (stats['filtered_large_candle'] + 
                           stats['filtered_excessive_movement'] + 
                           stats['filtered_parabolic_move'])
            print(f"\n📊 PERFORMANCE SUMMARY")
            print(f"   Signals processed: {stats['total_signals_checked']}")
            print(f"   Filter rate: {total_blocked/stats['total_signals_checked']*100:.1f}%")
        
        return 0
    
    def monitor_live(self, args):
        """Live monitoring mode"""
        if not self._check_initialization():
            return 1
        
        print(f"\n📡 LIVE MONITORING MODE")
        print("=" * 40)
        print("Monitoring large candle conditions...")
        print("Press Ctrl+C to stop")
        print("=" * 40)
        
        try:
            import time
            pairs = args.pairs.split(',') if args.pairs else ['EURUSD', 'GBPUSD', 'USDJPY']
            
            # Use correct MINI epic mapping
            epic_map = {
                'EURUSD': 'CS.D.EURUSD.CEEM.IP',
                'GBPUSD': 'CS.D.GBPUSD.MINI.IP', 
                'USDJPY': 'CS.D.USDJPY.MINI.IP',
                'AUDUSD': 'CS.D.AUDUSD.MINI.IP',
                'USDCAD': 'CS.D.USDCAD.MINI.IP',
                'EURJPY': 'CS.D.EURJPY.MINI.IP',
                'AUDJPY': 'CS.D.AUDJPY.MINI.IP',
                'NZDUSD': 'CS.D.NZDUSD.MINI.IP',
                'USDCHF': 'CS.D.USDCHF.MINI.IP'
            }
            
            while True:
                print(f"\n🕐 {datetime.now().strftime('%H:%M:%S')} - Checking conditions...")
                
                for pair in pairs:
                    epic = epic_map.get(pair, f'CS.D.{pair}.CFD.IP')
                    
                    try:
                        df = self.data_fetcher.get_enhanced_data(epic, pair, '5m', 20)
                        if df is None or len(df) < 10:
                            continue
                        
                        # Check both signal types
                        for signal_type in ['BULL', 'BEAR']:
                            should_block, reason = self.filter.should_block_entry(df, epic, signal_type, '5m')
                            
                            if should_block:
                                print(f"   🚫 {pair} {signal_type}: BLOCKED - {reason}")
                            else:
                                print(f"   ✅ {pair} {signal_type}: OK")
                    
                    except Exception as e:
                        print(f"   ❌ {pair}: Error - {str(e)[:50]}...")
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print(f"\n\n📡 Monitoring stopped")
            return 0
        except Exception as e:
            print(f"❌ Monitoring failed: {e}")
            return 1
    
    def reset_statistics(self, args):
        """Reset filter statistics"""
        if not self._check_initialization():
            return 1
        
        print(f"\n🔄 RESETTING STATISTICS")
        print("=" * 30)
        
        old_stats = self.filter.get_filter_statistics()
        self.filter.reset_statistics()
        
        print(f"✅ Statistics reset successfully")
        print(f"   Previous total: {old_stats['total_signals_checked']} signals")
        print(f"   Reset at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
    
    def _check_initialization(self):
        """Check if components are properly initialized"""
        if not MODULES_AVAILABLE:
            print("❌ Required modules not available")
            return False
        
        if not self.filter:
            print("❌ Filter not initialized")
            return False
        
        if not self.data_fetcher:
            print("❌ Data fetcher not available")
            return False
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Large Candle Filter Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_manager.py analyze --pair EURUSD --periods 50
  python filter_manager.py test --signal BULL --pair GBPUSD
  python filter_manager.py configure --preset strict
  python filter_manager.py monitor --live --pairs EURUSD,GBPUSD
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze recent price movement')
    analyze_parser.add_argument('--epic', default='CS.D.EURUSD.CEEM.IP', help='Trading epic (will auto-convert from pair)')
    analyze_parser.add_argument('--pair', default='EURUSD', help='Currency pair')
    analyze_parser.add_argument('--timeframe', default='5m', help='Timeframe')
    analyze_parser.add_argument('--periods', type=int, default=50, help='Periods to analyze')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test filter conditions')
    test_parser.add_argument('--epic', default='CS.D.EURUSD.CEEM.IP', help='Trading epic (will auto-convert from pair)')
    test_parser.add_argument('--pair', default='EURUSD', help='Currency pair')
    test_parser.add_argument('--timeframe', default='5m', help='Timeframe')
    test_parser.add_argument('--signal', choices=['BULL', 'BEAR'], default='BULL', help='Signal type')
    
    # Configure command
    config_parser = subparsers.add_parser('configure', help='Configure filter settings')
    config_parser.add_argument('--preset', choices=['strict', 'balanced', 'permissive'], 
                              default='balanced', help='Configuration preset')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show filter statistics')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show filter status')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Live monitoring mode')
    monitor_parser.add_argument('--live', action='store_true', help='Enable live monitoring')
    monitor_parser.add_argument('--pairs', default='EURUSD,GBPUSD,USDJPY', help='Comma-separated pairs')
    monitor_parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset filter statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize manager
    manager = FilterManager()
    
    # Route to appropriate handler
    if args.command == 'analyze':
        return manager.analyze_movement(args)
    elif args.command == 'test':
        return manager.test_filter(args)
    elif args.command == 'configure':
        return manager.configure_filter(args)
    elif args.command == 'stats':
        return manager.show_statistics(args)
    elif args.command == 'status':
        return manager.show_status(args)
    elif args.command == 'monitor':
        return manager.monitor_live(args)
    elif args.command == 'reset':
        return manager.reset_statistics(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())