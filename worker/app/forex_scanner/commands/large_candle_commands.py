# commands/large_candle_commands.py
"""
Large Candle Filter Testing and Configuration Commands
"""

import click
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
import json

try:
    from core.signal_detector import SignalDetector
    from core.detection.large_candle_filter import LargeCandleFilter
    from core.data_fetcher import DataFetcher
    import config
except ImportError:
    try:
        from forex_scanner.core.database import DatabaseManager
        from forex_scanner.core.signal_detector import SignalDetector
        from forex_scanner.core.data_fetcher import DataFetcher
        from forex_scanner.core.scanner import IntelligentForexScanner as ForexScanner
        from forex_scanner import config
    except ImportError as e:
        import sys
        print(f"Warning: Import fallback failed for {sys.modules[__name__]}: {e}")
        pass

class LargeCandleFilterCommands:
    """Commands for testing and configuring large candle filter"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.filter = LargeCandleFilter()
        self.data_fetcher = DataFetcher()
    
    @click.command()
    @click.option('--epic', '-e', default='CS.D.EURUSD.CFD.IP', help='Epic to analyze')
    @click.option('--pair', '-p', default='EURUSD', help='Currency pair')
    @click.option('--timeframe', '-t', default='5m', help='Timeframe')
    @click.option('--periods', '-n', default=50, help='Number of periods to analyze')
    def analyze_recent_movement(self, epic: str, pair: str, timeframe: str, periods: int):
        """Analyze recent price movement and large candles"""
        
        click.echo(f"\n🔍 Analyzing recent movement for {pair} ({timeframe})")
        click.echo("=" * 60)
        
        try:
            # Get data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe, periods + 20)
            if df is None or len(df) < 10:
                click.echo("❌ Insufficient data")
                return
            
            # Calculate ATR for context
            atr = self.filter._calculate_atr(df)
            latest_candles = df.tail(periods)
            
            click.echo(f"📊 Data Analysis:")
            click.echo(f"   Total periods: {len(latest_candles)}")
            click.echo(f"   ATR (14): {atr:.5f}")
            click.echo(f"   Large candle threshold: {atr * self.filter.large_candle_multiplier:.5f}")
            
            # Analyze each candle
            large_candles = []
            total_movement = 0
            
            for i, (idx, candle) in enumerate(latest_candles.iterrows()):
                candle_range = candle['high'] - candle['low']
                candle_body = abs(candle['close'] - candle['open'])
                atr_ratio = candle_range / atr if atr > 0 else 0
                
                total_movement += candle_range
                
                if atr_ratio > self.filter.large_candle_multiplier:
                    large_candles.append({
                        'index': i,
                        'timestamp': candle.get('start_time', idx),
                        'range': candle_range,
                        'body': candle_body,
                        'atr_ratio': atr_ratio,
                        'direction': 'BULL' if candle['close'] > candle['open'] else 'BEAR'
                    })
            
            # Report large candles
            click.echo(f"\n🔥 Large Candles Found: {len(large_candles)}")
            if large_candles:
                for lc in large_candles[-5:]:  # Show last 5
                    click.echo(f"   {lc['timestamp']}: {lc['atr_ratio']:.1f}x ATR, {lc['direction']}")
            
            # Test filter on current state
            current_signals = ['BULL', 'BEAR']
            for signal_type in current_signals:
                should_block, reason = self.filter.should_block_entry(df, epic, signal_type, timeframe)
                status = "🚫 BLOCKED" if should_block else "✅ ALLOWED"
                click.echo(f"   {signal_type} signal: {status}")
                if reason:
                    click.echo(f"      Reason: {reason}")
            
            # Movement analysis
            pip_multiplier = 10000 if 'JPY' not in epic.upper() else 100
            total_pips = total_movement * pip_multiplier
            avg_candle_pips = total_pips / len(latest_candles)
            
            click.echo(f"\n📈 Movement Summary:")
            click.echo(f"   Total movement: {total_pips:.1f} pips")
            click.echo(f"   Average per candle: {avg_candle_pips:.1f} pips")
            click.echo(f"   Large candle rate: {len(large_candles)/len(latest_candles)*100:.1f}%")
            
        except Exception as e:
            click.echo(f"❌ Analysis failed: {e}")
    
    @click.command()
    @click.option('--epic', '-e', default='CS.D.EURUSD.CFD.IP', help='Epic to test')
    @click.option('--pair', '-p', default='EURUSD', help='Currency pair')
    @click.option('--timeframe', '-t', default='5m', help='Timeframe')
    @click.option('--signal-type', '-s', type=click.Choice(['BULL', 'BEAR']), default='BULL', help='Signal type to test')
    def test_filter(self, epic: str, pair: str, timeframe: str, signal_type: str):
        """Test large candle filter on current market conditions"""
        
        click.echo(f"\n🧪 Testing Large Candle Filter")
        click.echo("=" * 40)
        
        try:
            # Get data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe, 100)
            if df is None or len(df) < 10:
                click.echo("❌ Insufficient data")
                return
            
            click.echo(f"📊 Testing {signal_type} signal for {pair}")
            click.echo(f"   Epic: {epic}")
            click.echo(f"   Timeframe: {timeframe}")
            click.echo(f"   Data points: {len(df)}")
            
            # Test the filter
            should_block, reason = self.filter.should_block_entry(df, epic, signal_type, timeframe)
            
            # Display result
            if should_block:
                click.echo(f"\n🚫 SIGNAL WOULD BE BLOCKED")
                click.echo(f"   Reason: {reason}")
            else:
                click.echo(f"\n✅ SIGNAL WOULD BE ALLOWED")
                click.echo(f"   No blocking conditions detected")
            
            # Show filter configuration
            click.echo(f"\n⚙️ Current Filter Configuration:")
            click.echo(f"   ATR Multiplier: {self.filter.large_candle_multiplier}")
            click.echo(f"   Consecutive Threshold: {self.filter.consecutive_large_threshold}")
            click.echo(f"   Movement Threshold: {self.filter.excessive_movement_threshold} pips")
            click.echo(f"   Lookback Periods: {self.filter.movement_lookback_periods}")
            click.echo(f"   Cooldown Periods: {self.filter.filter_cooldown_periods}")
            
            # Show recent market state
            latest = df.iloc[-1]
            atr = self.filter._calculate_atr(df)
            latest_range = latest['high'] - latest['low']
            atr_ratio = latest_range / atr if atr > 0 else 0
            
            click.echo(f"\n📊 Current Market State:")
            click.echo(f"   Current price: {latest['close']:.5f}")
            click.echo(f"   Latest candle range: {latest_range:.5f} ({atr_ratio:.1f}x ATR)")
            click.echo(f"   ATR(14): {atr:.5f}")
            
            # Get filter statistics
            stats = self.filter.get_filter_statistics()
            if stats['total_signals_checked'] > 0:
                click.echo(f"\n📈 Filter Statistics:")
                click.echo(f"   Total checked: {stats['total_signals_checked']}")
                click.echo(f"   Large candle blocks: {stats['filtered_large_candle']}")
                click.echo(f"   Movement blocks: {stats['filtered_excessive_movement']}")
                click.echo(f"   Parabolic blocks: {stats['filtered_parabolic_move']}")
            
        except Exception as e:
            click.echo(f"❌ Test failed: {e}")
    
    @click.command()
    @click.option('--preset', '-p', type=click.Choice(['strict', 'balanced', 'permissive']), 
                  default='balanced', help='Filter preset to apply')
    def configure_filter(self, preset: str):
        """Configure large candle filter settings"""
        
        click.echo(f"\n⚙️ Configuring Large Candle Filter")
        click.echo("=" * 40)
        
        presets = {
            'strict': {
                'LARGE_CANDLE_ATR_MULTIPLIER': 2.0,
                'CONSECUTIVE_LARGE_CANDLES_THRESHOLD': 1,
                'EXCESSIVE_MOVEMENT_THRESHOLD_PIPS': 10,
                'LARGE_CANDLE_FILTER_COOLDOWN': 5
            },
            'balanced': {
                'LARGE_CANDLE_ATR_MULTIPLIER': 2.5,
                'CONSECUTIVE_LARGE_CANDLES_THRESHOLD': 2,
                'EXCESSIVE_MOVEMENT_THRESHOLD_PIPS': 15,
                'LARGE_CANDLE_FILTER_COOLDOWN': 3
            },
            'permissive': {
                'LARGE_CANDLE_ATR_MULTIPLIER': 3.0,
                'CONSECUTIVE_LARGE_CANDLES_THRESHOLD': 3,
                'EXCESSIVE_MOVEMENT_THRESHOLD_PIPS': 20,
                'LARGE_CANDLE_FILTER_COOLDOWN': 2
            }
        }
        
        if preset not in presets:
            click.echo(f"❌ Unknown preset: {preset}")
            return
        
        preset_config = presets[preset]
        
        click.echo(f"📋 Applying '{preset}' preset:")
        for key, value in preset_config.items():
            click.echo(f"   {key}: {value}")
            # Apply to current filter instance
            if hasattr(self.filter, key.lower()):
                setattr(self.filter, key.lower(), value)
        
        click.echo(f"\n✅ Filter configured with '{preset}' preset")
        click.echo(f"\n📝 To make permanent, add this to config.py:")
        click.echo(f"LARGE_CANDLE_FILTER_PRESET = '{preset}'")
    
    @click.command()
    def show_statistics(self):
        """Show large candle filter statistics"""
        
        click.echo(f"\n📊 Large Candle Filter Statistics")
        click.echo("=" * 40)
        
        stats = self.filter.get_filter_statistics()
        
        if stats['total_signals_checked'] == 0:
            click.echo("📭 No signals have been processed yet")
            return
        
        click.echo(f"📈 Overall Statistics:")
        click.echo(f"   Total signals checked: {stats['total_signals_checked']}")
        click.echo(f"   Large candle blocks: {stats['filtered_large_candle']}")
        click.echo(f"   Excessive movement blocks: {stats['filtered_excessive_movement']}")
        click.echo(f"   Parabolic movement blocks: {stats['filtered_parabolic_move']}")
        
        total_blocked = (stats['filtered_large_candle'] + 
                        stats['filtered_excessive_movement'] + 
                        stats['filtered_parabolic_move'])
        
        click.echo(f"   Total blocked: {total_blocked}")
        
        if 'filter_rate' in stats:
            rates = stats['filter_rate']
            click.echo(f"\n📊 Filter Rates:")
            click.echo(f"   Large candle rate: {rates['large_candle_rate']:.1f}%")
            click.echo(f"   Excessive movement rate: {rates['excessive_movement_rate']:.1f}%")
            click.echo(f"   Parabolic movement rate: {rates['parabolic_movement_rate']:.1f}%")
            click.echo(f"   Total filter rate: {rates['total_filter_rate']:.1f}%")
    
    @click.command()
    def reset_statistics(self):
        """Reset large candle filter statistics"""
        
        self.filter.reset_statistics()
        click.echo("✅ Large candle filter statistics reset")


# Add to main CLI in main.py or commands/__init__.py
def add_large_candle_commands(cli_group):
    """Add large candle filter commands to CLI"""
    commands = LargeCandleFilterCommands()
    
    cli_group.add_command(commands.analyze_recent_movement, name='analyze-movement')
    cli_group.add_command(commands.test_filter, name='test-large-candle-filter')  
    cli_group.add_command(commands.configure_filter, name='configure-large-candle-filter')
    cli_group.add_command(commands.show_statistics, name='large-candle-stats')
    cli_group.add_command(commands.reset_statistics, name='reset-large-candle-stats')


# Usage examples:
# python main.py analyze-movement --epic CS.D.EURUSD.CFD.IP --pair EURUSD
# python main.py test-large-candle-filter --signal-type BULL --pair EURUSD
# python main.py configure-large-candle-filter --preset strict
# python main.py large-candle-stats
# python main.py reset-large-candle-stats