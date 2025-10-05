#!/usr/bin/env python3
"""
Signal Validation Tool - Recreate and validate specific backtest signal calculations
Usage: python signal_validation.py --timestamp "2025-08-22 09:15:00" --epic USDJPY --strategy macd
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))

# If we're in backtests/ subdirectory, go up one level to project root
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    # If we're in project root
    project_root = script_dir

sys.path.insert(0, project_root)

from core.data_fetcher import DataFetcher
from core.signal_detector import SignalDetector
from core.strategies.ema_strategy import EMAStrategy
from core.strategies.macd_strategy import MACDStrategy
from core.strategies.combined_strategy import CombinedStrategy
from analysis.technical import TechnicalAnalyzer
from analysis.volume import VolumeAnalyzer
try:
    import config
except ImportError:
    from forex_scanner import config


class SignalValidator:
    """Tool to validate specific backtest signal calculations by recreating them"""
    
    def __init__(self):
        self.logger = logging.getLogger('signal_validator')
        self.setup_logging()
        
        # Initialize components
        try:
            self.data_fetcher = DataFetcher()
        except TypeError:
            # DataFetcher might require different parameters, try without them
            from core.database import DatabaseManager
            self.db_manager = DatabaseManager(config.DATABASE_URL)
            self.data_fetcher = DataFetcher(self.db_manager)
        
        self.signal_detector = SignalDetector()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Volume analyzer might not exist, make it optional
        try:
            self.volume_analyzer = VolumeAnalyzer()
        except (ImportError, NameError):
            self.volume_analyzer = None
            self.logger.warning("VolumeAnalyzer not available, skipping volume analysis")
        
        # Initialize strategies with BACKTEST MODE
        try:
            self.ema_strategy = EMAStrategy(backtest_mode=True)
        except (ImportError, TypeError):
            self.ema_strategy = None
            self.logger.warning("EMAStrategy not available")
            
        try:
            self.macd_strategy = MACDStrategy(backtest_mode=True)
        except (ImportError, TypeError):
            self.macd_strategy = None
            self.logger.warning("MACDStrategy not available")
            
        try:
            self.combined_strategy = CombinedStrategy()
        except (ImportError, TypeError):
            self.combined_strategy = None
            self.logger.warning("CombinedStrategy not available")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def validate_signal(
        self,
        timestamp: str,
        epic: str,
        strategy: str = None,
        timeframe: str = '15m',
        show_raw_data: bool = False,
        export_format: str = None
    ) -> Dict[str, Any]:
        """
        Validate a specific signal by recreating the calculation
        
        Args:
            timestamp: Target timestamp (YYYY-MM-DD HH:MM:SS format)
            epic: Epic to analyze (e.g., USDJPY or CS.D.USDJPY.MINI.IP)
            strategy: Strategy to test (ema, macd, combined, or all)
            timeframe: Timeframe to analyze (default: 15m)
            show_raw_data: Show complete candle and indicator data
            export_format: Export format (json, csv, none)
            
        Returns:
            Dictionary with validation results
        """
        try:
            self.logger.info(f"🔍 Validating signal calculation:")
            self.logger.info(f"   📅 Timestamp: {timestamp}")
            self.logger.info(f"   📊 Epic: {epic}")
            self.logger.info(f"   🎯 Strategy: {strategy or 'all'}")
            self.logger.info(f"   ⏱️ Timeframe: {timeframe}")
            
            # Parse timestamp
            try:
                target_dt = pd.to_datetime(timestamp)
                self.logger.info(f"   🎯 Target datetime: {target_dt}")
            except Exception as e:
                self.logger.error(f"❌ Invalid timestamp format: {e}")
                return {'error': f'Invalid timestamp format: {timestamp}'}
            
            # Convert epic format if needed
            if not epic.startswith('CS.D.'):
                # Convert short form like 'USDJPY' to full epic
                epic_mapping = {
                    'EURUSD': 'CS.D.EURUSD.CEEM.IP',
                    'GBPUSD': 'CS.D.GBPUSD.MINI.IP', 
                    'USDJPY': 'CS.D.USDJPY.MINI.IP',
                    'AUDUSD': 'CS.D.AUDUSD.MINI.IP',
                    'USDCAD': 'CS.D.USDCAD.MINI.IP',
                    'USDCHF': 'CS.D.USDCHF.MINI.IP',
                    'NZDUSD': 'CS.D.NZDUSD.MINI.IP',
                    'EURJPY': 'CS.D.EURJPY.MINI.IP',
                    'GBPJPY': 'CS.D.GBPJPY.MINI.IP',
                    'AUDJPY': 'CS.D.AUDJPY.MINI.IP'
                }
                full_epic = epic_mapping.get(epic.upper())
                if full_epic:
                    self.logger.info(f"   🔄 Converted {epic} to {full_epic}")
                    epic = full_epic
                else:
                    self.logger.warning(f"⚠️ Unknown epic format: {epic}, proceeding anyway")
            
            # Extract pair from epic
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            self.logger.info(f"   💱 Pair: {pair}")
            
            # Get historical data around the target timestamp
            # We need extra data before the target to calculate indicators properly
            lookback_hours = 48  # 2 days of data to ensure indicators are calculated
            
            self.logger.info(f"   📥 Fetching historical data (lookback: {lookback_hours}h)")
            df = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=lookback_hours
            )
            
            if df is None or df.empty:
                self.logger.error(f"❌ No data available for {epic}")
                return {'error': f'No data available for {epic}'}
            
            self.logger.info(f"   📊 Data retrieved: {len(df)} candles")
            self.logger.info(f"   📅 Data range: {df.iloc[0]['datetime_utc']} to {df.iloc[-1]['datetime_utc']}")
            
            # Find the exact candle for our target timestamp
            target_candle_idx = self._find_target_candle(df, target_dt, timeframe)
            
            if target_candle_idx is None:
                self.logger.error(f"❌ No candle found for timestamp {timestamp}")
                return {'error': f'No candle found for timestamp {timestamp}'}
            
            target_candle = df.iloc[target_candle_idx]
            self.logger.info(f"   🎯 Found target candle at index {target_candle_idx}")
            self.logger.info(f"   📅 Candle time: {target_candle['datetime_utc']}")
            self.logger.info(f"   💰 OHLC: O={target_candle['open']:.5f}, H={target_candle['high']:.5f}, L={target_candle['low']:.5f}, C={target_candle['close']:.5f}")
            
            # Calculate all technical indicators up to this point
            df_with_indicators = self._calculate_all_indicators(df, target_candle_idx)
            
            # Get the indicator values at our target timestamp
            indicator_values = self._extract_indicator_values(df_with_indicators, target_candle_idx)
            
            # Test strategies and get detailed calculation info
            strategy_results = self._test_strategies(df_with_indicators, target_candle_idx, epic, pair, timeframe, strategy)
            
            # Compile validation results
            validation_result = {
                'timestamp': timestamp,
                'target_datetime': str(target_dt),
                'epic': epic,
                'pair': pair,
                'timeframe': timeframe,
                'candle_index': target_candle_idx,
                'candle_data': {
                    'datetime_utc': str(target_candle['datetime_utc']),
                    'open': float(target_candle['open']),
                    'high': float(target_candle['high']),
                    'low': float(target_candle['low']),
                    'close': float(target_candle['close']),
                    'volume': float(target_candle.get('volume', 0))
                },
                'technical_indicators': indicator_values,
                'strategy_results': strategy_results
            }
            
            if show_raw_data:
                validation_result['raw_data'] = {
                    'full_dataframe_shape': df.shape,
                    'columns': list(df.columns),
                    'target_row': {col: target_candle[col] for col in target_candle.index}
                }
            
            # Display results
            self._display_validation_results(validation_result)
            
            # Export if requested
            if export_format:
                self._export_results(validation_result, export_format, timestamp, epic)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Signal validation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _find_target_candle(self, df: pd.DataFrame, target_dt: pd.Timestamp, timeframe: str) -> Optional[int]:
        """Find the candle index that corresponds to our target timestamp"""
        try:
            # Convert target to timezone-aware if needed
            if target_dt.tz is None:
                target_dt = target_dt.tz_localize('UTC')
            elif target_dt.tz != pd.Timestamp.now().tz:
                target_dt = target_dt.tz_convert('UTC')
            
            # Parse timeframe to get minutes
            timeframe_minutes = self._parse_timeframe_minutes(timeframe)
            tolerance_minutes = timeframe_minutes // 2  # Half the timeframe as tolerance
            
            self.logger.info(f"   🔍 Searching for candle within ±{tolerance_minutes} minutes of target")
            
            # Find closest candle by datetime
            best_idx = None
            min_diff = float('inf')
            
            for idx, row in df.iterrows():
                candle_time = pd.to_datetime(row['datetime_utc'])
                if candle_time.tz is None:
                    candle_time = candle_time.tz_localize('UTC')
                
                time_diff = abs((candle_time - target_dt).total_seconds() / 60)  # Difference in minutes
                
                if time_diff <= tolerance_minutes and time_diff < min_diff:
                    min_diff = time_diff
                    best_idx = idx
            
            if best_idx is not None:
                # Convert to positional index
                positional_idx = df.index.get_loc(best_idx)
                self.logger.info(f"   ✅ Found candle at positional index {positional_idx}, time difference: {min_diff:.1f} minutes")
                return positional_idx
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding target candle: {e}")
            return None
    
    def _parse_timeframe_minutes(self, timeframe: str) -> int:
        """Parse timeframe string to minutes"""
        timeframe = timeframe.lower()
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24 * 60
        else:
            return 15  # Default to 15 minutes
    
    def _calculate_all_indicators(self, df: pd.DataFrame, target_idx: int) -> pd.DataFrame:
        """Calculate all technical indicators up to the target index"""
        try:
            self.logger.info(f"   🔢 Calculating technical indicators...")
            
            # Make a copy to avoid modifying original
            df_calc = df.copy()
            
            # Add all EMAs using the correct method
            ema_periods = [9, 21, 200]  # Standard periods
            df_calc = self.technical_analyzer.add_ema_indicators(df_calc, ema_periods)
            
            # Add MACD manually since we need to check the method name
            if 'macd_line' not in df_calc.columns:
                ema_12 = df_calc['close'].ewm(span=12).mean()
                ema_26 = df_calc['close'].ewm(span=26).mean()
                df_calc['macd_line'] = ema_12 - ema_26
                df_calc['macd_signal'] = df_calc['macd_line'].ewm(span=9).mean()
                df_calc['macd_histogram'] = df_calc['macd_line'] - df_calc['macd_signal']
            
            # Add basic volume analysis manually
            if 'volume' in df_calc.columns and 'volume_sma' not in df_calc.columns:
                df_calc['volume_sma'] = df_calc['volume'].rolling(20).mean()
                df_calc['volume_ratio'] = df_calc['volume'] / df_calc['volume_sma']
            
            self.logger.info(f"   ✅ Indicators calculated, total columns: {len(df_calc.columns)}")
            
            return df_calc
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df
    
    def _extract_indicator_values(self, df: pd.DataFrame, target_idx: int) -> Dict[str, Any]:
        """Extract all indicator values at the target timestamp"""
        try:
            target_row = df.iloc[target_idx]
            
            indicators = {}
            
            # EMA values
            ema_fields = ['ema_9', 'ema_21', 'ema_200', 'ema_short', 'ema_long', 'ema_trend']
            for field in ema_fields:
                if field in target_row and pd.notna(target_row[field]):
                    indicators[field] = float(target_row[field])
            
            # MACD values
            macd_fields = ['macd_line', 'macd_signal', 'macd_histogram']
            for field in macd_fields:
                if field in target_row and pd.notna(target_row[field]):
                    indicators[field] = float(target_row[field])
            
            # Volume values
            volume_fields = ['volume', 'volume_sma', 'volume_ratio']
            for field in volume_fields:
                if field in target_row and pd.notna(target_row[field]):
                    indicators[field] = float(target_row[field])
            
            # Support/Resistance
            sr_fields = ['support_level', 'resistance_level', 'support_strength', 'resistance_strength']
            for field in sr_fields:
                if field in target_row and pd.notna(target_row[field]):
                    indicators[field] = float(target_row[field])
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting indicators: {e}")
            return {}
    
    def _test_strategies(self, df: pd.DataFrame, target_idx: int, epic: str, pair: str, timeframe: str, strategy_filter: str = None) -> Dict[str, Any]:
        """Test all strategies at the target timestamp and show detailed calculations"""
        try:
            self.logger.info(f"   🎯 Testing strategies at target timestamp...")
            
            results = {}
            
            # Prepare data slice up to target point (strategies shouldn't see future data)
            df_for_strategy = df.iloc[:target_idx + 1].copy()
            
            spread_pips = getattr(config, 'SPREAD_PIPS', 1.5)
            
            # Test EMA Strategy
            if (not strategy_filter or 'ema' in strategy_filter.lower()) and self.ema_strategy:
                try:
                    ema_signal = self.ema_strategy.detect_signal(df_for_strategy, epic, spread_pips, timeframe)
                    results['ema_strategy'] = {
                        'signal_detected': ema_signal is not None,
                        'signal_data': ema_signal if ema_signal else None,
                        'calculation_details': self._get_ema_calculation_details(df_for_strategy, target_idx)
                    }
                except Exception as e:
                    results['ema_strategy'] = {'error': str(e)}
            
            # Test MACD Strategy  
            if (not strategy_filter or 'macd' in strategy_filter.lower()) and self.macd_strategy:
                try:
                    macd_signal = self.macd_strategy.detect_signal(df_for_strategy, epic, spread_pips, timeframe)
                    results['macd_strategy'] = {
                        'signal_detected': macd_signal is not None,
                        'signal_data': macd_signal if macd_signal else None,
                        'calculation_details': self._get_macd_calculation_details(df_for_strategy, target_idx)
                    }
                except Exception as e:
                    results['macd_strategy'] = {'error': str(e)}
            
            # Test Combined Strategy
            if (not strategy_filter or 'combined' in strategy_filter.lower()) and self.combined_strategy:
                try:
                    combined_signal = self.combined_strategy.detect_signal(df_for_strategy, epic, spread_pips, timeframe)
                    results['combined_strategy'] = {
                        'signal_detected': combined_signal is not None,
                        'signal_data': combined_signal if combined_signal else None,
                        'calculation_details': self._get_combined_calculation_details(df_for_strategy, target_idx)
                    }
                except Exception as e:
                    results['combined_strategy'] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing strategies: {e}")
            return {'error': str(e)}
    
    def _get_ema_calculation_details(self, df: pd.DataFrame, target_idx: int) -> Dict[str, Any]:
        """Get detailed EMA strategy calculation breakdown"""
        try:
            target_row = df.iloc[target_idx]
            
            details = {
                'price': float(target_row['close']),
                'ema_9': float(target_row.get('ema_9', 0)),
                'ema_21': float(target_row.get('ema_21', 0)),
                'ema_200': float(target_row.get('ema_200', 0)),
            }
            
            # EMA relationships
            details['price_vs_ema_9'] = details['price'] - details['ema_9']
            details['price_vs_ema_21'] = details['price'] - details['ema_21']
            details['price_vs_ema_200'] = details['price'] - details['ema_200']
            details['ema_9_vs_ema_21'] = details['ema_9'] - details['ema_21']
            details['ema_21_vs_ema_200'] = details['ema_21'] - details['ema_200']
            
            # Trend analysis
            details['bullish_alignment'] = (details['price'] > details['ema_9'] > details['ema_21'] > details['ema_200'])
            details['bearish_alignment'] = (details['price'] < details['ema_9'] < details['ema_21'] < details['ema_200'])
            
            # Check for crossovers (need previous values)
            if target_idx > 0:
                prev_row = df.iloc[target_idx - 1]
                prev_ema_9 = float(prev_row.get('ema_9', 0))
                prev_ema_21 = float(prev_row.get('ema_21', 0))
                
                details['ema_9_crossed_above_21'] = (prev_ema_9 <= prev_ema_21) and (details['ema_9'] > details['ema_21'])
                details['ema_9_crossed_below_21'] = (prev_ema_9 >= prev_ema_21) and (details['ema_9'] < details['ema_21'])
            
            return details
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_macd_calculation_details(self, df: pd.DataFrame, target_idx: int) -> Dict[str, Any]:
        """Get detailed MACD strategy calculation breakdown"""
        try:
            target_row = df.iloc[target_idx]
            
            details = {
                'price': float(target_row['close']),
                'macd_line': float(target_row.get('macd_line', 0)),
                'macd_signal': float(target_row.get('macd_signal', 0)),
                'macd_histogram': float(target_row.get('macd_histogram', 0)),
                'ema_200': float(target_row.get('ema_200', 0)),
            }
            
            # MACD relationships
            details['macd_line_vs_signal'] = details['macd_line'] - details['macd_signal']
            details['price_vs_ema_200'] = details['price'] - details['ema_200']
            details['histogram_positive'] = details['macd_histogram'] > 0
            details['histogram_negative'] = details['macd_histogram'] < 0
            details['price_above_ema_200'] = details['price'] > details['ema_200']
            details['price_below_ema_200'] = details['price'] < details['ema_200']
            
            # Check for histogram changes (need previous values)
            if target_idx > 0:
                prev_row = df.iloc[target_idx - 1]
                prev_histogram = float(prev_row.get('macd_histogram', 0))
                
                details['histogram_turned_positive'] = (prev_histogram <= 0) and (details['macd_histogram'] > 0)
                details['histogram_turned_negative'] = (prev_histogram >= 0) and (details['macd_histogram'] < 0)
                details['previous_histogram'] = prev_histogram
            
            return details
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_combined_calculation_details(self, df: pd.DataFrame, target_idx: int) -> Dict[str, Any]:
        """Get detailed combined strategy calculation breakdown"""
        try:
            # This would show how individual strategy results are combined
            details = {
                'note': 'Combined strategy aggregates EMA and MACD results',
                'combination_mode': getattr(config, 'COMBINED_STRATEGY_MODE', 'consensus'),
                'ema_weight': getattr(config, 'STRATEGY_WEIGHT_EMA', 0.6),
                'macd_weight': getattr(config, 'STRATEGY_WEIGHT_MACD', 0.4),
                'min_confidence': getattr(config, 'MIN_COMBINED_CONFIDENCE', 0.75)
            }
            
            return details
            
        except Exception as e:
            return {'error': str(e)}
    
    def _display_validation_results(self, result: Dict[str, Any]):
        """Display the validation results in detail"""
        self.logger.info(f"\n🔍 SIGNAL VALIDATION RESULTS:")
        self.logger.info("=" * 70)
        self.logger.info(f"   🎯 Target: {result['timestamp']} ({result['epic']})")
        self.logger.info(f"   📅 Actual candle time: {result['candle_data']['datetime_utc']}")
        self.logger.info(f"   📊 Candle index: {result['candle_index']}")
        
        # Show candle data
        candle = result['candle_data']
        self.logger.info(f"\n💰 CANDLE DATA:")
        self.logger.info("-" * 30)
        self.logger.info(f"   Open:   {candle['open']:.5f}")
        self.logger.info(f"   High:   {candle['high']:.5f}")
        self.logger.info(f"   Low:    {candle['low']:.5f}")
        self.logger.info(f"   Close:  {candle['close']:.5f}")
        self.logger.info(f"   Volume: {candle['volume']:.0f}")
        
        # Show technical indicators
        indicators = result['technical_indicators']
        if indicators:
            self.logger.info(f"\n📊 TECHNICAL INDICATORS:")
            self.logger.info("-" * 40)
            
            # Group by type
            ema_indicators = {k: v for k, v in indicators.items() if 'ema' in k.lower()}
            macd_indicators = {k: v for k, v in indicators.items() if 'macd' in k.lower()}
            other_indicators = {k: v for k, v in indicators.items() if 'ema' not in k.lower() and 'macd' not in k.lower()}
            
            if ema_indicators:
                self.logger.info("   📈 EMA Values:")
                for name, value in ema_indicators.items():
                    self.logger.info(f"      {name:12}: {value:.6f}")
            
            if macd_indicators:
                self.logger.info("   📊 MACD Values:")
                for name, value in macd_indicators.items():
                    self.logger.info(f"      {name:12}: {value:.6f}")
            
            if other_indicators:
                self.logger.info("   🔧 Other Indicators:")
                for name, value in other_indicators.items():
                    self.logger.info(f"      {name:12}: {value:.6f}")
        
        # Show strategy results
        strategy_results = result['strategy_results']
        for strategy_name, strategy_result in strategy_results.items():
            self.logger.info(f"\n🎯 {strategy_name.upper().replace('_', ' ')} RESULTS:")
            self.logger.info("-" * 50)
            
            if 'error' in strategy_result:
                self.logger.info(f"   ❌ Error: {strategy_result['error']}")
                continue
            
            signal_detected = strategy_result['signal_detected']
            self.logger.info(f"   📊 Signal Detected: {'✅ YES' if signal_detected else '❌ NO'}")
            
            if signal_detected and strategy_result['signal_data']:
                signal = strategy_result['signal_data']
                self.logger.info(f"   📈 Signal Type: {signal.get('signal_type', 'Unknown')}")
                self.logger.info(f"   💯 Confidence: {signal.get('confidence_score', 0):.1%}")
                self.logger.info(f"   💰 Price: {signal.get('price', 0):.5f}")
                self.logger.info(f"   🎯 Strategy: {signal.get('strategy', 'Unknown')}")
            
            # Show calculation details
            if 'calculation_details' in strategy_result:
                details = strategy_result['calculation_details']
                self.logger.info(f"   🔢 Calculation Details:")
                for key, value in details.items():
                    if isinstance(value, bool):
                        self.logger.info(f"      {key:20}: {'✅' if value else '❌'}")
                    elif isinstance(value, (int, float)):
                        self.logger.info(f"      {key:20}: {value:.6f}")
                    else:
                        self.logger.info(f"      {key:20}: {value}")
    
    def _export_results(self, result: Dict[str, Any], export_format: str, timestamp: str, epic: str):
        """Export validation results to file"""
        try:
            # Create filename
            clean_timestamp = timestamp.replace(':', '-').replace(' ', '_')
            filename = f"signal_validation_{epic}_{clean_timestamp}.{export_format}"
            
            if export_format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                self.logger.info(f"📁 Results exported to: {filename}")
                
            else:
                self.logger.warning(f"⚠️ Unsupported export format: {export_format}")
                
        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Signal Validation Tool - Recreate and validate specific signal calculations')
    
    # Required arguments
    parser.add_argument('--timestamp', required=True, help='Target timestamp (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--epic', required=True, help='Epic to analyze (e.g., USDJPY or CS.D.USDJPY.MINI.IP)')
    
    # Optional arguments
    parser.add_argument('--strategy', help='Strategy to test (ema, macd, combined, or leave empty for all)')
    parser.add_argument('--timeframe', default='15m', help='Timeframe to analyze (default: 15m)')
    parser.add_argument('--raw-data', action='store_true', help='Show complete raw data')
    parser.add_argument('--export', choices=['json'], help='Export results to file')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create validation tool
    validator = SignalValidator()
    
    try:
        # Validate the signal
        result = validator.validate_signal(
            timestamp=args.timestamp,
            epic=args.epic,
            strategy=args.strategy,
            timeframe=args.timeframe,
            show_raw_data=args.raw_data,
            export_format=args.export
        )
        
        if 'error' not in result:
            print(f"\n✅ Signal validation completed successfully!")
            print(f"💡 Check the detailed calculation breakdown above to validate the math")
            
            # Check if any signals were detected
            strategies_with_signals = []
            for strategy_name, strategy_result in result.get('strategy_results', {}).items():
                if strategy_result.get('signal_detected'):
                    strategies_with_signals.append(strategy_name)
            
            if strategies_with_signals:
                print(f"🎯 Signals detected by: {', '.join(strategies_with_signals)}")
            else:
                print(f"❌ No signals detected by any strategy at this timestamp")
                
            sys.exit(0)
        else:
            print(f"\n❌ Validation failed: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()