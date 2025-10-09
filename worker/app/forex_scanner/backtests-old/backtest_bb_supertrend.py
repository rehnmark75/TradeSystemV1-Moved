#!/usr/bin/env python3
"""
BB+Supertrend Strategy Backtest - Standalone Module
Run: python backtest_bb_supertrend.py --epic CS.D.EURUSD.CEEM.IP --days 7 --timeframe 15m
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))

# If we're in backtests/ subdirectory, go up one level to project root
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    # If we're in project root
    project_root = script_dir

sys.path.insert(0, project_root)

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
# 🔧 CRITICAL FIX: Import the modular version correctly
try:
    from core.strategies.bb_supertrend_strategy import create_bb_supertrend_strategy, BollingerSupertrendStrategy
    MODULAR_VERSION_AVAILABLE = True
    print("✅ Using MODULAR BB+Supertrend strategy")
except ImportError as e:
    print(f"❌ Failed to import modular BB strategy: {e}")
    from core.strategies.bb_supertrend_strategy import BollingerSupertrendStrategy
    MODULAR_VERSION_AVAILABLE = False
    print("⚠️ Falling back to legacy BB+Supertrend strategy")

from core.backtest.performance_analyzer import PerformanceAnalyzer
from core.backtest.signal_analyzer import SignalAnalyzer
try:
    import config
except ImportError:
    from forex_scanner import config


class BBSupertrendBacktest:
    """Dedicated BB+Supertrend Strategy Backtesting"""
    
    def __init__(self):
        self.logger = logging.getLogger('bb_supertrend_backtest')
        self.setup_logging()
        
        # Initialize components
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        self.strategy = None
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def initialize_bb_strategy(self, bb_config: str = None):
        """Initialize BB+Supertrend strategy with specific configuration"""
        
        # Get available BB configurations
        bb_configs = getattr(config, 'BB_SUPERTREND_CONFIGS', {
            'conservative': {
                'bb_period': 20,
                'bb_std_dev': 2.5,
                'supertrend_period': 14,
                'supertrend_multiplier': 3.5,
                'base_confidence': 0.70
            },
            'default': {
                'bb_period': 14,
                'bb_std_dev': 1.8,
                'supertrend_period': 8,
                'supertrend_multiplier': 2.5,
                'base_confidence': 0.60
            },
            'aggressive': {
                'bb_period': 14,
                'bb_std_dev': 1.8,
                'supertrend_period': 7,
                'supertrend_multiplier': 2.5,
                'base_confidence': 0.50
            }
        })
        
        # Use factory function for modular strategy
        config_name = bb_config or 'default'
        
        # 🔧 CRITICAL FIX: Ensure we use the modular version
        if MODULAR_VERSION_AVAILABLE:
            self.logger.info(f"✅ Using MODULAR BB+Supertrend strategy")
            self.strategy = create_bb_supertrend_strategy(
                bb_config_name=config_name, 
                data_fetcher=self.data_fetcher
            )
            
            # 🔍 DEBUG: Verify modular version is loaded
            if hasattr(self.strategy, 'forex_optimizer'):
                self.logger.info(f"🔥 Forex optimizer detected: {type(self.strategy.forex_optimizer)}")
            else:
                self.logger.warning(f"⚠️ No forex optimizer found in strategy")
                
            if hasattr(self.strategy, 'multi_timeframe_analyzer'):
                self.logger.info(f"🔄 MTF analyzer detected: {type(self.strategy.multi_timeframe_analyzer)}")
            else:
                self.logger.warning(f"⚠️ No MTF analyzer found in strategy")
        else:
            self.logger.warning(f"⚠️ Using LEGACY BB+Supertrend strategy (no modular features)")
            self.strategy = BollingerSupertrendStrategy(config_name)
        
        # Validate modular integration
        if hasattr(self.strategy, 'validate_modular_integration'):
            if not self.strategy.validate_modular_integration():
                self.logger.warning("⚠️ Modular integration validation failed")
        else:
            self.logger.warning("⚠️ Strategy doesn't have modular integration validation")
        
        self.logger.info("✅ BB+Supertrend Strategy initialized for backtest")
        self.logger.info(f"🔧 Using BB config: {config_name}")
        
        # 🔍 CRITICAL DEBUG: Verify which version we're using
        self.logger.info(f"📍 Strategy class: {type(self.strategy)}")
        self.logger.info(f"📍 Strategy module: {self.strategy.__class__.__module__}")
        
        # Check for modular components
        modular_components = {
            'forex_optimizer': hasattr(self.strategy, 'forex_optimizer'),
            'validator': hasattr(self.strategy, 'validator'), 
            'cache': hasattr(self.strategy, 'cache'),
            'data_helper': hasattr(self.strategy, 'data_helper'),
            'signal_detector': hasattr(self.strategy, 'signal_detector'),
            'multi_timeframe_analyzer': hasattr(self.strategy, 'multi_timeframe_analyzer')
        }
        
        self.logger.info(f"📍 Modular components: {modular_components}")
        
        # Check calculate_confidence method source
        if hasattr(self.strategy, 'calculate_confidence'):
            try:
                import inspect
                method_source = inspect.getsource(self.strategy.calculate_confidence)
                if 'FOREX OPTIMIZER ENHANCEMENTS' in method_source:
                    self.logger.info("✅ Using ENHANCED calculate_confidence with forex optimizer")
                elif 'Enhanced Signal Validator' in method_source:
                    self.logger.warning("⚠️ Using LEGACY calculate_confidence (Enhanced Validator only)")
                else:
                    self.logger.warning("⚠️ Using UNKNOWN calculate_confidence version")
            except Exception as e:
                self.logger.debug(f"Could not inspect calculate_confidence: {e}")
        
        # Get BB configuration details
        if config_name in bb_configs:
            periods = bb_configs[config_name]
            self.logger.info(f"   📊 BB Period: {periods.get('bb_period')}, Std Dev: {periods.get('bb_std_dev')}")
            self.logger.info(f"   📈 SuperTrend: Period {periods.get('supertrend_period')}, Multiplier {periods.get('supertrend_multiplier')}")
            self.logger.info(f"   🎯 Base Confidence: {periods.get('base_confidence', 0.6):.1%}")
        
        return config_name
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        bb_config: str = None,
        min_confidence: float = None,
        show_cache_stats: bool = False
    ) -> bool:
        """Run BB+Supertrend strategy backtest"""
        
        # Setup epic list
        epic_list = [epic] if epic else config.EPIC_LIST
        
        self.logger.info("🔥 BB+SUPERTREND STRATEGY BACKTEST")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Epic(s): {epic_list}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🎯 Show signals: {show_signals}")
        self.logger.info(f"📈 Show cache stats: {show_cache_stats}")
        
        # Override minimum confidence if specified
        if min_confidence:
            original_min_conf = getattr(config, 'MIN_CONFIDENCE', 0.7)
            config.MIN_CONFIDENCE = min_confidence
            self.logger.info(f"🎚️ Min confidence: {min_confidence:.1%} (was {original_min_conf:.1%})")
        
        try:
            # Initialize strategy
            config_name = self.initialize_bb_strategy(bb_config)
            
            all_signals = []
            epic_results = {}
            
            for current_epic in epic_list:
                self.logger.info(f"\n📈 Processing {current_epic}")
                
                # Get enhanced data - Extract pair from epic
                pair = self._extract_pair_from_epic(current_epic)
                
                df = self.data_fetcher.get_enhanced_data(
                    epic=current_epic,
                    pair=pair,
                    timeframe=timeframe,
                    lookback_hours=days * 24
                )
                
                if df.empty:
                    self.logger.warning(f"❌ No data available for {current_epic}")
                    epic_results[current_epic] = {'signals': 0, 'error': 'No data'}
                    continue
                
                self.logger.info(f"   📊 Data points: {len(df)}")
                self.logger.info(f"   📅 Date range: {df.iloc[0].get('datetime_utc', df.iloc[0].get('start_time', 'Unknown'))} to {df.iloc[-1].get('datetime_utc', df.iloc[-1].get('start_time', 'Unknown'))}")
                
                # Run BB+Supertrend backtest
                signals = self._run_bb_backtest(df, current_epic, timeframe)
                
                all_signals.extend(signals)
                epic_results[current_epic] = {'signals': len(signals)}
                
                self.logger.info(f"   🎯 BB+Supertrend signals found: {len(signals)}")
            
            # Display epic-by-epic results
            self._display_epic_results(epic_results)
            
            # Show cache performance if requested
            if show_cache_stats and hasattr(self.strategy, 'get_cache_stats'):
                self._display_cache_stats()
            
            # Overall analysis
            if all_signals:
                self.logger.info(f"\n✅ TOTAL BB+SUPERTREND SIGNALS: {len(all_signals)}")
                
                # Show individual signals if requested
                if show_signals:
                    self._display_signals(all_signals)
                
                # Performance analysis
                self._analyze_performance(all_signals)
                
                return True
            else:
                self.logger.warning("❌ No BB+Supertrend signals found in backtest period")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ BB+Supertrend backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_bb_backtest(self, df: pd.DataFrame, epic: str, timeframe: str) -> List[Dict]:
        """Run BB+Supertrend backtest on historical data"""
        signals = []
        
        # Process each candle for potential signals
        # Start from enough candles to ensure BB and SuperTrend are calculated
        min_start = max(50, self.strategy.min_bars if hasattr(self.strategy, 'min_bars') else 50)
        
        for i in range(min_start, len(df)):
            try:
                # Get data up to current point
                current_data = df.iloc[:i+1].copy()
                
                # Detect signal using BB+Supertrend strategy
                signal = self.strategy.detect_signal(
                    current_data, epic, config.SPREAD_PIPS, timeframe
                )
                
                if signal:
                    # Get the actual historical timestamp from the candle data
                    candle_row = df.iloc[i]
                    
                    # Extract timestamp - try multiple possible field names
                    historical_timestamp = None
                    timestamp_fields = ['datetime_utc', 'start_time', 'timestamp', 'datetime']
                    
                    for field in timestamp_fields:
                        if field in candle_row.index and not pd.isna(candle_row[field]):
                            historical_timestamp = candle_row[field]
                            break
                    
                    # If we found a timestamp, use it; otherwise fall back to index
                    if historical_timestamp is not None:
                        # Convert to datetime if it's not already
                        if isinstance(historical_timestamp, str):
                            try:
                                from dateutil import parser
                                historical_timestamp = parser.parse(historical_timestamp)
                            except:
                                # If parsing fails, try pandas
                                historical_timestamp = pd.to_datetime(historical_timestamp)
                        elif isinstance(historical_timestamp, pd.Timestamp):
                            historical_timestamp = historical_timestamp.to_pydatetime()
                    else:
                        # Fallback to DataFrame index if available
                        if hasattr(candle_row, 'name') and candle_row.name is not None:
                            historical_timestamp = candle_row.name
                            if isinstance(historical_timestamp, pd.Timestamp):
                                historical_timestamp = historical_timestamp.to_pydatetime()
                        else:
                            # Ultimate fallback - use a reasonable timestamp
                            from datetime import datetime, timedelta
                            historical_timestamp = datetime.now() - timedelta(days=7) + timedelta(minutes=i*15)
                    
                    # Debug: Print signal fields to understand confidence storage
                    self.logger.debug(f"🔍 Raw BB signal from strategy: {signal}")
                    
                    # Add backtest metadata with ACTUAL historical timestamp
                    signal['backtest_timestamp'] = historical_timestamp
                    signal['backtest_index'] = i
                    signal['candle_data'] = {
                        'open': float(df.iloc[i]['open']),
                        'high': float(df.iloc[i]['high']),
                        'low': float(df.iloc[i]['low']),
                        'close': float(df.iloc[i]['close'])
                    }
                    
                    # Add BB and SuperTrend levels for analysis
                    signal['bb_analysis'] = {
                        'bb_upper': df.iloc[i].get('bb_upper', 0),
                        'bb_middle': df.iloc[i].get('bb_middle', 0),
                        'bb_lower': df.iloc[i].get('bb_lower', 0),
                        'bb_width': df.iloc[i].get('bb_upper', 0) - df.iloc[i].get('bb_lower', 0),
                        'supertrend': df.iloc[i].get('supertrend', 0),
                        'supertrend_direction': df.iloc[i].get('supertrend_direction', 0)
                    }
                    
                    # CRITICAL: Override the signal timestamp with the actual historical timestamp
                    signal['timestamp'] = historical_timestamp
                    signal['alert_timestamp'] = historical_timestamp
                    signal['market_timestamp'] = historical_timestamp
                    
                    # Also update any datetime fields in the original signal to use historical time
                    if 'processing_timestamp' in signal:
                        signal['processing_timestamp'] = historical_timestamp.isoformat() if hasattr(historical_timestamp, 'isoformat') else str(historical_timestamp)
                    
                    # Standardize confidence field names
                    confidence_value = signal.get('confidence', signal.get('confidence_score', 0))
                    if confidence_value is not None:
                        # Ensure both field names are present for compatibility
                        signal['confidence'] = confidence_value
                        signal['confidence_score'] = confidence_value
                        
                        # Debug: Show what confidence value we're setting
                        self.logger.debug(f"🔍 Setting BB confidence: {confidence_value:.3f} ({confidence_value:.1%})")
                    
                    # Add performance metrics by looking ahead
                    enhanced_signal = self._add_performance_metrics(signal, df, i)
                    
                    signals.append(enhanced_signal)
                    
                    # Use the historical timestamp in debug output
                    formatted_timestamp = historical_timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(historical_timestamp, 'strftime') else str(historical_timestamp)
                    
                    self.logger.debug(f"📊 BB+SuperTrend signal at {formatted_timestamp}: "
                                    f"{signal.get('signal_type')} (conf: {confidence_value:.1%}) "
                                    f"BB: {signal['bb_analysis']['bb_width']:.6f} ST: {signal['bb_analysis']['supertrend_direction']}")
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue
        
        return signals
    
    def _add_performance_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Add performance metrics to signal by looking ahead"""
        try:
            enhanced_signal = signal.copy()
            
            # Get signal details
            entry_price = signal.get('price', signal.get('entry_price', df.iloc[signal_idx]['close']))
            signal_type = signal.get('signal_type', 'UNKNOWN')
            stop_loss = signal.get('stop_loss', signal.get('sl', 0))
            take_profit = signal.get('take_profit', signal.get('tp', 0))
            
            # Look ahead for performance (up to 24 hours / 96 bars for 15m timeframe)
            max_lookback = min(96, len(df) - signal_idx - 1)
            
            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]
                
                if signal_type == 'BULL':
                    # For buy signals, look for profit in higher prices
                    max_profit = (future_data['high'].max() - entry_price) * 10000  # Convert to pips
                    max_loss = (entry_price - future_data['low'].min()) * 10000
                elif signal_type == 'BEAR':
                    # For sell signals, look for profit in lower prices
                    max_profit = (entry_price - future_data['low'].min()) * 10000  # Convert to pips
                    max_loss = (future_data['high'].max() - entry_price) * 10000
                else:
                    max_profit = 0
                    max_loss = 0
                
                # Check if stop loss or take profit would have been hit
                stop_hit = False
                tp_hit = False
                bars_to_stop = None
                bars_to_tp = None
                
                if stop_loss and take_profit:
                    for j, (idx, row) in enumerate(future_data.iterrows()):
                        if signal_type == 'BULL':
                            if row['low'] <= stop_loss and not stop_hit:
                                stop_hit = True
                                bars_to_stop = j + 1
                            if row['high'] >= take_profit and not tp_hit:
                                tp_hit = True
                                bars_to_tp = j + 1
                        else:  # BEAR
                            if row['high'] >= stop_loss and not stop_hit:
                                stop_hit = True
                                bars_to_stop = j + 1
                            if row['low'] <= take_profit and not tp_hit:
                                tp_hit = True
                                bars_to_tp = j + 1
                        
                        # Stop if both hit or if one hits first
                        if stop_hit or tp_hit:
                            break
                
                enhanced_signal.update({
                    'max_profit_pips': round(max_profit, 1),
                    'max_loss_pips': round(max_loss, 1),
                    'profit_loss_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
                    'lookback_bars': max_lookback,
                    'stop_hit': stop_hit,
                    'tp_hit': tp_hit,
                    'bars_to_stop': bars_to_stop,
                    'bars_to_tp': bars_to_tp,
                    'trade_result': 'TP' if tp_hit and (not stop_hit or (bars_to_tp and bars_to_stop and bars_to_tp < bars_to_stop)) 
                                   else 'SL' if stop_hit 
                                   else 'OPEN'
                })
                
                # Calculate actual pips if stop/tp hit
                if enhanced_signal['trade_result'] == 'TP':
                    actual_pips = abs(take_profit - entry_price) * 10000
                    enhanced_signal['actual_result_pips'] = actual_pips
                elif enhanced_signal['trade_result'] == 'SL':
                    actual_pips = -abs(entry_price - stop_loss) * 10000
                    enhanced_signal['actual_result_pips'] = actual_pips
                else:
                    enhanced_signal['actual_result_pips'] = 0
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error adding performance metrics: {e}")
            return signal
    
    def _display_epic_results(self, epic_results: Dict):
        """Display results by epic"""
        self.logger.info("\n📊 RESULTS BY EPIC:")
        self.logger.info("-" * 30)
        
        for epic, result in epic_results.items():
            if 'error' in result:
                self.logger.info(f"   {epic}: ❌ {result['error']}")
            else:
                self.logger.info(f"   {epic}: {result['signals']} signals")
    
    def _display_cache_stats(self):
        """Display cache performance statistics"""
        try:
            cache_stats = self.strategy.get_cache_stats()
            
            self.logger.info("\n🚀 CACHE PERFORMANCE STATISTICS:")
            self.logger.info("-" * 40)
            
            if 'main_cache' in cache_stats:
                main_cache = cache_stats['main_cache']
                self.logger.info(f"   📊 Cache Hit Ratio: {main_cache.get('hit_ratio', 0):.1%}")
                self.logger.info(f"   🎯 Cache Hits: {main_cache.get('cache_hits', 0)}")
                self.logger.info(f"   ❌ Cache Misses: {main_cache.get('cache_misses', 0)}")
                self.logger.info(f"   💾 Cached Entries: {main_cache.get('cached_entries', {}).get('total', 0)}")
                self.logger.info(f"   ⏱️ Cache Performance: {main_cache.get('cache_performance', 'unknown')}")
            
            if 'forex_optimizer_cache' in cache_stats:
                forex_cache = cache_stats['forex_optimizer_cache']
                self.logger.info(f"   🔥 Forex Optimizer Cache: {forex_cache.get('cached_entries', 0)} entries")
            
            self.logger.info(f"   🏗️ Total Modules: {cache_stats.get('total_modules', 0)}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not display cache stats: {e}")
    
    def _display_signals(self, signals: List[Dict]):
        """Display individual signals with proper historical timestamps"""
        self.logger.info("\n🎯 INDIVIDUAL BB+SUPERTREND SIGNALS:")
        self.logger.info("-" * 60)
        
        # Use SignalAnalyzer if available
        try:
            self.signal_analyzer.display_signal_list(signals, max_signals=20)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not use SignalAnalyzer display: {e}")
            # Fallback to manual display with BB-specific info
            for i, signal in enumerate(signals[:20], 1):
                # Get the historical timestamp
                timestamp = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))
                
                # Format timestamp properly
                if hasattr(timestamp, 'strftime'):
                    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(timestamp, str):
                    # Try to parse and reformat if it's a string
                    try:
                        from dateutil import parser
                        parsed_timestamp = parser.parse(timestamp)
                        formatted_timestamp = parsed_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_timestamp = timestamp
                else:
                    formatted_timestamp = str(timestamp)
                
                epic = signal.get('epic', 'Unknown')
                signal_type = signal.get('signal_type', 'Unknown')
                confidence = signal.get('confidence', signal.get('confidence_score', 0))
                price = signal.get('price', signal.get('entry_price', 0))
                
                # BB-specific info
                bb_analysis = signal.get('bb_analysis', {})
                bb_width = bb_analysis.get('bb_width', 0)
                st_direction = bb_analysis.get('supertrend_direction', 0)
                
                # Performance info
                trade_result = signal.get('trade_result', 'OPEN')
                actual_pips = signal.get('actual_result_pips', 0)
                
                # Multi-timeframe info if available
                mtf_info = ""
                if 'multi_timeframe_analysis' in signal:
                    mtf_analysis = signal['multi_timeframe_analysis']
                    confluence_score = mtf_analysis.get('confluence_score', 0)
                    agreement_level = mtf_analysis.get('agreement_level', 'unknown')
                    mtf_info = f" | MTF: {confluence_score:.1%} ({agreement_level})"
                
                self.logger.info(f"{i:2d}. {formatted_timestamp} | {epic} | {signal_type} | "
                               f"Conf: {confidence:.1%} | Price: {price:.5f} | "
                               f"BB: {bb_width:.6f} | ST: {st_direction:+d} | "
                               f"Result: {trade_result} ({actual_pips:+.1f} pips){mtf_info}")
                
                # Add MTF details for first few signals if available
                if i <= 5 and 'multi_timeframe_analysis' in signal:
                    mtf_analysis = signal['multi_timeframe_analysis']
                    higher_tf_bias = mtf_analysis.get('higher_timeframe_bias', 'neutral')
                    mean_reversion_setup = mtf_analysis.get('mean_reversion_setup', False)
                    self.logger.info(f"     🔄 MTF Details: HTF bias={higher_tf_bias}, Mean reversion setup={mean_reversion_setup}")
    
    def _analyze_performance(self, signals: List[Dict]):
        """Analyze BB+Supertrend strategy performance"""
        try:
            # Debug: Print first signal to see field structure
            if signals:
                first_signal = signals[0]
                self.logger.debug(f"📊 Sample BB signal keys: {list(first_signal.keys())}")
                self.logger.debug(f"📊 Confidence field values: confidence={first_signal.get('confidence')}, confidence_score={first_signal.get('confidence_score')}")
            
            # Create custom performance analysis
            metrics = self._create_custom_bb_performance_analysis(signals)
            
            self.logger.info("\n📈 BB+SUPERTREND STRATEGY PERFORMANCE:")
            self.logger.info("=" * 50)
            self.logger.info(f"   📊 Total Signals: {metrics.get('total_signals', len(signals))}")
            self.logger.info(f"   🎯 Average Confidence: {metrics.get('avg_confidence', 0):.1%}")
            self.logger.info(f"   📈 Bull Signals: {metrics.get('bull_signals', 0)}")
            self.logger.info(f"   📉 Bear Signals: {metrics.get('bear_signals', 0)}")
            
            # BB-specific metrics
            if 'avg_bb_width' in metrics:
                self.logger.info(f"   📏 Average BB Width: {metrics['avg_bb_width']:.6f}")
            if 'supertrend_alignment' in metrics:
                self.logger.info(f"   📈 SuperTrend Alignment: {metrics['supertrend_alignment']:.1%}")
            
            # Multi-timeframe metrics
            if 'mtf_signals_count' in metrics:
                self.logger.info(f"   🔄 MTF Signals: {metrics['mtf_signals_count']}/{metrics['total_signals']} ({metrics['mtf_coverage']:.1%} coverage)")
                self.logger.info(f"   🔄 Average MTF Confluence: {metrics['avg_mtf_confluence']:.1%}")
                self.logger.info(f"   🔄 High Agreement Rate: {metrics['mtf_high_agreement_rate']:.1%}")
                self.logger.info(f"   🔄 Favorable HTF Bias Rate: {metrics['mtf_favorable_bias_rate']:.1%}")
                self.logger.info(f"   🔄 Mean Reversion Setup Rate: {metrics['mtf_mean_reversion_setup_rate']:.1%}")
            
            # Performance metrics
            if 'actual_win_rate' in metrics:
                self.logger.info(f"   🏆 Actual Win Rate: {metrics['actual_win_rate']:.1%}")
                
                # MTF performance correlation
                if 'mtf_win_rate' in metrics:
                    self.logger.info(f"   🔄 MTF Win Rate: {metrics['mtf_win_rate']:.1%}")
                    if 'winner_avg_confluence' in metrics and 'loser_avg_confluence' in metrics:
                        self.logger.info(f"   🔄 Winner Avg Confluence: {metrics['winner_avg_confluence']:.1%}")
                        self.logger.info(f"   🔄 Loser Avg Confluence: {metrics['loser_avg_confluence']:.1%}")
                        
                        # Show correlation insight
                        if metrics['winner_avg_confluence'] > metrics['loser_avg_confluence']:
                            diff = metrics['winner_avg_confluence'] - metrics['loser_avg_confluence']
                            self.logger.info(f"   ✅ Winners have {diff:.1%} higher confluence (MTF working!)")
                        else:
                            self.logger.info(f"   ⚠️ No clear MTF confluence advantage detected")
                self.logger.info(f"   🏆 Actual Win Rate: {metrics['actual_win_rate']:.1%}")
            if 'avg_winner_pips' in metrics:
                self.logger.info(f"   💰 Average Winner: {metrics['avg_winner_pips']:.1f} pips")
            if 'avg_loser_pips' in metrics:
                self.logger.info(f"   📉 Average Loser: {metrics['avg_loser_pips']:.1f} pips")
            if 'profit_factor' in metrics:
                self.logger.info(f"   📊 Profit Factor: {metrics['profit_factor']:.2f}")
            if 'total_pips' in metrics:
                self.logger.info(f"   💰 Total Pips: {metrics['total_pips']:+.1f}")
            
            # Risk metrics
            if 'max_drawdown_pips' in metrics:
                self.logger.info(f"   📉 Max Drawdown: {metrics['max_drawdown_pips']:.1f} pips")
            if 'largest_winner' in metrics:
                self.logger.info(f"   🚀 Largest Winner: {metrics['largest_winner']:.1f} pips")
            if 'largest_loser' in metrics:
                self.logger.info(f"   💀 Largest Loser: {metrics['largest_loser']:.1f} pips")
            
            # Confidence analysis
            if 'confidence_range' in metrics:
                conf_range = metrics['confidence_range']
                self.logger.info(f"   📈 Confidence Range: {conf_range['min']:.1%} - {conf_range['max']:.1%}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ BB performance analysis failed: {e}")
            
            # Enhanced fallback analysis
            total = len(signals)
            bull_count = sum(1 for s in signals if s.get('signal_type') == 'BULL')
            bear_count = total - bull_count
            
            # Calculate average confidence
            confidences = []
            for s in signals:
                conf = s.get('confidence', s.get('confidence_score', 0))
                if conf is not None:
                    if conf > 1:
                        conf = conf / 100.0
                    confidences.append(conf)
            
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            # Basic trade results
            tp_trades = [s for s in signals if s.get('trade_result') == 'TP']
            sl_trades = [s for s in signals if s.get('trade_result') == 'SL']
            open_trades = [s for s in signals if s.get('trade_result') == 'OPEN']
            
            win_rate = len(tp_trades) / (len(tp_trades) + len(sl_trades)) if (tp_trades or sl_trades) else 0
            
            self.logger.info("\n📈 BB+SUPERTREND PERFORMANCE (Basic):")
            self.logger.info("=" * 50)
            self.logger.info(f"   📊 Total Signals: {total}")
            self.logger.info(f"   🎯 Average Confidence: {avg_conf:.1%}")
            self.logger.info(f"   📈 Bull Signals: {bull_count}")
            self.logger.info(f"   📉 Bear Signals: {bear_count}")
            self.logger.info(f"   🏆 Win Rate: {win_rate:.1%}")
            self.logger.info(f"   ✅ Winners: {len(tp_trades)}")
            self.logger.info(f"   ❌ Losers: {len(sl_trades)}")
            self.logger.info(f"   ⏳ Open: {len(open_trades)}")

    def _create_custom_bb_performance_analysis(self, signals: List[Dict]) -> Dict:
        """Create custom performance analysis for BB+Supertrend strategy"""
        if not signals:
            return {}
        
        total_signals = len(signals)
        bull_signals = [s for s in signals if s.get('signal_type') == 'BULL']
        bear_signals = [s for s in signals if s.get('signal_type') == 'BEAR']
        
        # Calculate confidence properly
        confidences = []
        for s in signals:
            conf = s.get('confidence', s.get('confidence_score', 0))
            if conf is not None:
                confidences.append(conf)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # BB-specific analysis
        bb_widths = []
        supertrend_correct = 0
        
        # Multi-timeframe analysis
        mtf_signals = [s for s in signals if 'multi_timeframe_analysis' in s]
        mtf_confluence_scores = []
        mtf_high_agreement = 0
        mtf_favorable_bias = 0
        mtf_mean_reversion_setups = 0
        
        for s in signals:
            bb_analysis = s.get('bb_analysis', {})
            if 'bb_width' in bb_analysis:
                bb_widths.append(bb_analysis['bb_width'])
            
            # Check SuperTrend alignment
            st_direction = bb_analysis.get('supertrend_direction', 0)
            signal_type = s.get('signal_type', '')
            if (signal_type == 'BULL' and st_direction == 1) or (signal_type == 'BEAR' and st_direction == -1):
                supertrend_correct += 1
            
            # Multi-timeframe analysis
            if 'multi_timeframe_analysis' in s:
                mtf_analysis = s['multi_timeframe_analysis']
                
                confluence_score = mtf_analysis.get('confluence_score', 0)
                mtf_confluence_scores.append(confluence_score)
                
                if mtf_analysis.get('agreement_level') == 'high':
                    mtf_high_agreement += 1
                
                if mtf_analysis.get('higher_timeframe_bias') == 'favorable':
                    mtf_favorable_bias += 1
                
                if mtf_analysis.get('mean_reversion_setup', False):
                    mtf_mean_reversion_setups += 1
        
        # Trade results analysis
        tp_trades = [s for s in signals if s.get('trade_result') == 'TP']
        sl_trades = [s for s in signals if s.get('trade_result') == 'SL']
        closed_trades = tp_trades + sl_trades
        
        metrics = {
            'total_signals': total_signals,
            'bull_signals': len(bull_signals),
            'bear_signals': len(bear_signals),
            'avg_confidence': avg_confidence,
            'avg_bb_width': sum(bb_widths) / len(bb_widths) if bb_widths else 0,
            'supertrend_alignment': supertrend_correct / total_signals if total_signals > 0 else 0
        }
        
        # Add multi-timeframe metrics
        if mtf_signals:
            metrics.update({
                'mtf_signals_count': len(mtf_signals),
                'mtf_coverage': len(mtf_signals) / total_signals,
                'avg_mtf_confluence': sum(mtf_confluence_scores) / len(mtf_confluence_scores) if mtf_confluence_scores else 0,
                'mtf_high_agreement_rate': mtf_high_agreement / len(mtf_signals) if mtf_signals else 0,
                'mtf_favorable_bias_rate': mtf_favorable_bias / len(mtf_signals) if mtf_signals else 0,
                'mtf_mean_reversion_setup_rate': mtf_mean_reversion_setups / len(mtf_signals) if mtf_signals else 0
            })
        
        if confidences:
            metrics['confidence_range'] = {
                'min': min(confidences),
                'max': max(confidences)
            }
        
        # Performance calculations
        if closed_trades:
            actual_win_rate = len(tp_trades) / len(closed_trades)
            
            # Calculate pips
            winner_pips = [s.get('actual_result_pips', 0) for s in tp_trades]
            loser_pips = [abs(s.get('actual_result_pips', 0)) for s in sl_trades]  # Make positive for display
            all_trade_pips = [s.get('actual_result_pips', 0) for s in closed_trades]
            
            avg_winner = sum(winner_pips) / len(winner_pips) if winner_pips else 0
            avg_loser = sum(loser_pips) / len(loser_pips) if loser_pips else 0
            total_pips = sum(all_trade_pips)
            
            # Profit factor
            total_profit = sum(winner_pips)
            total_loss = sum(loser_pips)
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Drawdown calculation
            running_total = 0
            peak = 0
            max_drawdown = 0
            
            for pips in all_trade_pips:
                running_total += pips
                if running_total > peak:
                    peak = running_total
                drawdown = peak - running_total
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            metrics.update({
                'actual_win_rate': actual_win_rate,
                'avg_winner_pips': avg_winner,
                'avg_loser_pips': avg_loser,
                'profit_factor': profit_factor,
                'total_pips': total_pips,
                'max_drawdown_pips': max_drawdown,
                'largest_winner': max(winner_pips) if winner_pips else 0,
                'largest_loser': max(loser_pips) if loser_pips else 0,
                'total_trades_closed': len(closed_trades),
                'winners': len(tp_trades),
                'losers': len(sl_trades)
            })
            
            # Analyze MTF performance correlation
            if mtf_signals:
                mtf_tp_trades = [s for s in tp_trades if 'multi_timeframe_analysis' in s]
                mtf_sl_trades = [s for s in sl_trades if 'multi_timeframe_analysis' in s]
                mtf_closed_trades = mtf_tp_trades + mtf_sl_trades
                
                if mtf_closed_trades:
                    mtf_win_rate = len(mtf_tp_trades) / len(mtf_closed_trades)
                    
                    # Average confluence score for winners vs losers
                    winner_confluence = [s['multi_timeframe_analysis'].get('confluence_score', 0) for s in mtf_tp_trades]
                    loser_confluence = [s['multi_timeframe_analysis'].get('confluence_score', 0) for s in mtf_sl_trades]
                    
                    metrics.update({
                        'mtf_win_rate': mtf_win_rate,
                        'winner_avg_confluence': sum(winner_confluence) / len(winner_confluence) if winner_confluence else 0,
                        'loser_avg_confluence': sum(loser_confluence) / len(loser_confluence) if loser_confluence else 0
                    })
        
        return metrics
    
    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic code"""
        try:
            # Convert "CS.D.EURUSD.CEEM.IP" to "EURUSD"
            if '.D.' in epic and '.MINI.IP' in epic:
                parts = epic.split('.D.')
                if len(parts) > 1:
                    pair_part = parts[1].split('.MINI.IP')[0]
                    return pair_part
            
            # Fallback - try to extract from config.PAIR_INFO
            pair_info = getattr(config, 'PAIR_INFO', {})
            if epic in pair_info:
                return pair_info[epic].get('pair', 'EURUSD')
            
            # Default fallback
            self.logger.warning(f"⚠️ Could not extract pair from {epic}, using EURUSD")
            return 'EURUSD'
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting pair from {epic}: {e}, using EURUSD")
            return 'EURUSD'


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='BB+Supertrend Strategy Backtest')
    
    # Required arguments
    parser.add_argument('--epic', help='Epic to backtest (e.g., CS.D.EURUSD.CEEM.IP)')
    parser.add_argument('--days', type=int, default=7, help='Days to backtest (default: 7)')
    parser.add_argument('--timeframe', default='15m', help='Timeframe (default: 15m)')
    
    # Optional arguments
    parser.add_argument('--show-signals', action='store_true', help='Show individual signals')
    parser.add_argument('--bb-config', help='BB configuration (default, conservative, aggressive)')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')
    parser.add_argument('--show-cache-stats', action='store_true', help='Show cache performance statistics')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run backtest
    backtest = BBSupertrendBacktest()
    
    success = backtest.run_backtest(
        epic=args.epic,
        days=args.days,
        timeframe=args.timeframe,
        show_signals=args.show_signals,
        bb_config=args.bb_config,
        min_confidence=args.min_confidence,
        show_cache_stats=args.show_cache_stats
    )
    
    if success:
        print(f"\n✅ BB+Supertrend backtest completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ BB+Supertrend backtest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()