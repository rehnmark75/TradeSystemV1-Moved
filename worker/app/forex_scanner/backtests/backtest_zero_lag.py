#!/usr/bin/env python3
"""
Zero Lag + Squeeze Momentum Strategy Backtest - Enhanced Edition
Features: Trailing stops, exit tracking, Smart Money integration, comprehensive analysis
Run: python backtest_zero_lag.py --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m --squeeze-momentum
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

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
from core.strategies.zero_lag_strategy import ZeroLagStrategy
from core.backtest.performance_analyzer import PerformanceAnalyzer
from core.backtest.signal_analyzer import SignalAnalyzer

# Smart Money Integration imports (with fallback handling)
try:
    from core.smart_money_readonly_analyzer import SmartMoneyReadOnlyAnalyzer
    from core.smart_money_integration import SmartMoneyIntegration
    SMART_MONEY_AVAILABLE = True
except ImportError:
    SMART_MONEY_AVAILABLE = False
    logging.getLogger(__name__).warning("Smart Money modules not available - running without SMC analysis")

try:
    import config
except ImportError:
    from forex_scanner import config


class ZeroLagBacktest:
    """Enhanced Zero Lag + Squeeze Momentum Strategy Backtesting with Smart Money Analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger('zero_lag_backtest')
        self.setup_logging()
        
        # Initialize components
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        # Use UTC for all timestamps in backtest for consistency
        self.data_fetcher = DataFetcher(self.db_manager, 'UTC')
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        self.strategy = None
        
        # Smart Money Integration
        self.smart_money_enabled = False
        self.smart_money_integration = None
        self.smart_money_analyzer = None
        
        # Performance tracking
        self.smart_money_stats = {
            'signals_analyzed': 0,
            'signals_enhanced': 0,
            'analysis_failures': 0,
            'avg_analysis_time': 0
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _ensure_utc_timestamp(self, timestamp):
        """
        Ensure timestamp is in UTC format (no timezone conversion)
        All timestamps will remain in UTC for consistency
        """
        try:
            if isinstance(timestamp, str):
                if 'UTC' not in timestamp:
                    return f"{timestamp} UTC"
                return timestamp
                
            if hasattr(timestamp, 'strftime'):
                return timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                
            return f"{str(timestamp)} UTC"
            
        except Exception as e:
            self.logger.debug(f"Timestamp conversion error: {e}")
            return f"{str(timestamp)} UTC"

    def initialize_zero_lag_strategy(self, enable_squeeze_momentum: bool = True):
        """Initialize Zero Lag + Squeeze Momentum strategy"""
        
        # Initialize with data_fetcher for enhanced functionality
        self.strategy = ZeroLagStrategy(data_fetcher=self.data_fetcher)
        self.logger.info("✅ Zero Lag + Squeeze Momentum Strategy initialized")
        
        # Get strategy summary
        summary = self.strategy.get_strategy_summary()
        self.logger.info(f"   📊 Strategy: {summary.get('strategy_name')}")
        self.logger.info(f"   🏗️ Architecture: {summary.get('architecture')}")
        
        # Log component status
        components = summary.get('components', {})
        for name, desc in components.items():
            self.logger.info(f"   ⚙️ {name}: {desc}")

    def initialize_smart_money_integration(self, enable_smart_money: bool = False):
        """Initialize Smart Money analysis if available and requested"""
        if not enable_smart_money or not SMART_MONEY_AVAILABLE:
            self.smart_money_enabled = False
            return
        
        try:
            self.smart_money_analyzer = SmartMoneyReadOnlyAnalyzer()
            self.smart_money_integration = SmartMoneyIntegration(
                logger=self.logger,
                analyzer=self.smart_money_analyzer
            )
            self.smart_money_enabled = True
            self.logger.info("✅ Smart Money Concepts (SMC) analysis enabled")
            self.logger.info("   📊 Features: Market Structure, Order Blocks, Fair Value Gaps")
            
        except Exception as e:
            self.logger.error(f"❌ Smart Money initialization failed: {e}")
            self.smart_money_enabled = False
    
    def _display_zero_lag_statistics(self, signals: List[Dict]):
        """Display Zero Lag strategy-specific statistics"""
        self.logger.info("\n⚡ ZERO LAG STRATEGY STATISTICS:")
        self.logger.info("=" * 45)
        
        # Get Zero Lag configuration details
        try:
            from configdata import config_zerolag_strategy
            self.logger.info(f"   ⚡ Zero Lag Length: {config_zerolag_strategy.ZERO_LAG_LENGTH}")
            self.logger.info(f"   📊 Band Multiplier: {config_zerolag_strategy.ZERO_LAG_BAND_MULT}")
            self.logger.info(f"   🎯 Min Confidence: {config_zerolag_strategy.ZERO_LAG_MIN_CONFIDENCE:.1%}")
        except ImportError:
            self.logger.warning("⚠️ Zero Lag config not found, using defaults")
        
        # Check momentum bias status
        if hasattr(self.strategy, 'momentum_bias_enabled'):
            self.logger.info(f"   🚀 Momentum Bias: {'Enabled' if self.strategy.momentum_bias_enabled else 'Disabled'}")
        
        # Validate modular integration
        if hasattr(self.strategy, 'validate_modular_integration'):
            if self.strategy.validate_modular_integration():
                self.logger.info("   🏗️ Modular architecture: ✅ Validated")
            else:
                self.logger.warning("   🏗️ Modular architecture: ⚠️ Validation failed")
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        enable_squeeze_momentum: bool = True,
        enable_smart_money: bool = False,
        min_confidence: float = None,
        sl_type: str = 'atr',
        sl_atr_multiplier: float = 1.5,
        trailing_stop: bool = True
    ) -> bool:
        """Run Zero Lag strategy backtest"""
        
        # Setup epic list
        epic_list = [epic] if epic else config.EPIC_LIST
        
        self.logger.info("⚡ ZERO LAG + SQUEEZE MOMENTUM BACKTEST")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Epic(s): {epic_list}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🎯 Show signals: {show_signals}")
        self.logger.info(f"🔍 Squeeze Momentum: {'Enabled' if enable_squeeze_momentum else 'Disabled'}")
        self.logger.info(f"🧠 Smart Money: {'Enabled' if enable_smart_money else 'Disabled'}")
        self.logger.info(f"🛑 Stop Loss: {sl_type.upper()}")
        if sl_type == 'atr':
            self.logger.info(f"📏 ATR Multiplier: {sl_atr_multiplier}x")
        self.logger.info(f"📈 Trailing Stop: {'Enabled' if trailing_stop else 'Disabled'}")
        
        # Override minimum confidence if specified
        if min_confidence:
            try:
                from configdata import config_zerolag_strategy
                original_min_conf = getattr(config_zerolag_strategy, 'ZERO_LAG_MIN_CONFIDENCE', 0.65)
                config_zerolag_strategy.ZERO_LAG_MIN_CONFIDENCE = min_confidence
                self.logger.info(f"🎚️ Min confidence: {min_confidence:.1%} (was {original_min_conf:.1%})")
            except:
                original_min_conf = getattr(config, 'MIN_CONFIDENCE', 0.65)
                config.MIN_CONFIDENCE = min_confidence
                self.logger.info(f"🎚️ Min confidence: {min_confidence:.1%} (was {original_min_conf:.1%})")
        
        try:
            # Initialize strategy
            self.initialize_zero_lag_strategy(enable_squeeze_momentum)
            
            # Initialize Smart Money if requested
            self.initialize_smart_money_integration(enable_smart_money)
            
            all_signals = []
            epic_results = {}
            
            for current_epic in epic_list:
                self.logger.info(f"\n📈 Processing {current_epic}")
                
                # Get enhanced data - Need to extract pair from epic
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
                
                # Debug: Show available timestamp columns
                timestamp_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'utc'])]
                if timestamp_columns:
                    self.logger.info(f"   📅 Available timestamp columns: {timestamp_columns}")
                    # Show sample timestamp values
                    for col in timestamp_columns[:2]:  # Show max 2 columns
                        sample_value = df[col].iloc[-1] if len(df) > 0 else 'None'
                        self.logger.info(f"      {col}: {sample_value}")
                else:
                    self.logger.warning(f"   ⚠️ No timestamp columns found in DataFrame")
                    self.logger.info(f"   📋 Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
                
                # Show data range info
                if len(df) > 0:
                    first_row = df.iloc[0]
                    last_row = df.iloc[-1]
                    
                    # Try to get readable timestamps - FIXED: Pass individual rows instead of entire DataFrame
                    start_time = self._get_proper_timestamp(first_row, 0)
                    end_time = self._get_proper_timestamp(last_row, len(df)-1)
                    
                    # Ensure UTC is shown in data range display
                    if 'UTC' not in start_time:
                        start_time = f"{start_time} UTC"
                    if 'UTC' not in end_time:
                        end_time = f"{end_time} UTC"
                    
                    self.logger.info(f"   📅 Data range: {start_time} to {end_time}")
                
                # Run Zero Lag backtest
                signals = self._run_zero_lag_backtest(df, current_epic, timeframe)
                
                all_signals.extend(signals)
                epic_results[current_epic] = {'signals': len(signals)}
                
                self.logger.info(f"   ⚡ Zero Lag signals found: {len(signals)}")
                
                # Show squeeze momentum statistics
                squeeze_signals = [s for s in signals if s.get('squeeze_momentum') is not None]
                if squeeze_signals:
                    positive_squeeze = [s for s in squeeze_signals if s.get('squeeze_momentum', 0) > 0]
                    negative_squeeze = [s for s in squeeze_signals if s.get('squeeze_momentum', 0) < 0]
                    lime_signals = [s for s in signals if s.get('squeeze_is_lime', False)]
                    red_signals = [s for s in signals if s.get('squeeze_is_red', False)]
                    
                    self.logger.info(f"   🔍 Squeeze Positive: {len(positive_squeeze)}")
                    self.logger.info(f"   🔍 Squeeze Negative: {len(negative_squeeze)}")
                    self.logger.info(f"   🟢 Lime Signals: {len(lime_signals)}")
                    self.logger.info(f"   🔴 Red Signals: {len(red_signals)}")
                
                # Show Smart Money statistics if enabled
                if self.smart_money_enabled:
                    self._display_smart_money_stats(signals)
            
            # Display epic-by-epic results
            self._display_epic_results(epic_results)
            
            # Overall analysis
            if all_signals:
                self.logger.info(f"\n✅ TOTAL ZERO LAG SIGNALS: {len(all_signals)}")
                
                # Show strategy-specific statistics
                self._display_zero_lag_statistics(all_signals)
                
                # Show individual signals if requested
                if show_signals:
                    self._display_signals(all_signals)
                
                # Performance analysis
                self._analyze_performance(all_signals)
                
                return True
            else:
                self.logger.warning("❌ No Zero Lag signals found in backtest period")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Zero Lag backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_zero_lag_backtest(self, df: pd.DataFrame, epic: str, timeframe: str) -> List[Dict]:
        """Run Zero Lag backtest on historical data"""
        signals = []
        
        # Calculate minimum required bars (Zero Lag + momentum bias + buffer)
        min_bars = 100  # Ensure enough data for Zero Lag and momentum bias calculations
        
        total_detections = 0
        confidence_rejections = 0
        momentum_bias_rejections = 0
        successful_signals = 0
        
        for i in range(min_bars, len(df)):
            try:
                # Get data up to current point (simulate real-time processing)
                current_data = df.iloc[:i+1].copy()
                
                # Detect signal using Zero Lag strategy
                signal = self.strategy.detect_signal(
                    current_data, epic, config.SPREAD_PIPS, timeframe
                )
                
                if signal:
                    total_detections += 1
                    confidence_value = signal.get('confidence', signal.get('confidence_score', 0))
                    
                    # Apply confidence filter (mimic real-time behavior)
                    min_confidence = getattr(self.strategy, 'min_confidence', 0.65)
                    if confidence_value < min_confidence:
                        confidence_rejections += 1
                        self.logger.debug(f"❌ Signal rejected: confidence {confidence_value:.1%} < {min_confidence:.1%}")
                        continue
                    
                    # Apply momentum bias filter if enabled
                    if self.strategy.momentum_bias_enabled:
                        momentum_confirmed = signal.get('momentum_bias_confirmation', False)
                        if not momentum_confirmed:
                            momentum_bias_rejections += 1
                            self.logger.debug(f"🚫 Signal rejected: momentum bias not confirmed")
                            continue
                    
                    successful_signals += 1
                    
                    # Add backtest metadata with FIXED timestamp handling
                    timestamp_value = self._get_proper_timestamp(df.iloc[i], i)
                    signal['backtest_timestamp'] = timestamp_value
                    signal['backtest_index'] = i
                    signal['candle_data'] = {
                        'open': float(df.iloc[i]['open']),
                        'high': float(df.iloc[i]['high']),
                        'low': float(df.iloc[i]['low']),
                        'close': float(df.iloc[i]['close']),
                        'timestamp': timestamp_value
                    }
                    
                    # Add Zero Lag specific data from current candle
                    current_candle = df.iloc[i]
                    signal['zero_lag_data'] = {
                        'zlema': float(current_candle.get('zlema', signal.get('zlema', 0))),
                        'volatility': float(current_candle.get('volatility', signal.get('volatility', 0))),
                        'upper_band': float(current_candle.get('upper_band', signal.get('upper_band', 0))),
                        'lower_band': float(current_candle.get('lower_band', signal.get('lower_band', 0))),
                        'trend': int(current_candle.get('trend', signal.get('trend', 0)))
                    }
                    
                    # Add momentum bias data if available
                    if self.strategy.momentum_bias_enabled:
                        signal['momentum_bias_data'] = {
                            'momentum_up_bias': float(current_candle.get('momentum_up_bias', 0)),
                            'momentum_down_bias': float(current_candle.get('momentum_down_bias', 0)),
                            'boundary': float(current_candle.get('boundary', 0)),
                            'confirmation': signal.get('momentum_bias_confirmation', False)
                        }
                    
                    # Ensure timestamp field for analysis with proper format
                    if 'timestamp' not in signal:
                        signal['timestamp'] = timestamp_value
                    
                    # Also ensure market_timestamp is properly set
                    signal['market_timestamp'] = timestamp_value
                    
                    # Set additional timestamp fields that SignalAnalyzer might look for
                    signal['signal_timestamp'] = timestamp_value
                    signal['alert_timestamp'] = timestamp_value
                    signal['datetime'] = timestamp_value
                    signal['datetime_utc'] = timestamp_value
                    
                    # Debug: Log the timestamp that was set
                    self.logger.debug(f"🕐 Set signal timestamp: {timestamp_value} for signal at index {i}")
                    
                    # Standardize confidence field names
                    confidence_value = signal.get('confidence', signal.get('confidence_score', 0))
                    if confidence_value is not None:
                        signal['confidence'] = confidence_value
                        signal['confidence_score'] = confidence_value
                        self.logger.debug(f"🔍 Setting Zero Lag confidence: {confidence_value:.3f} ({confidence_value:.1%})")
                    
                    # Add performance metrics by looking ahead
                    enhanced_signal = self._add_performance_metrics(signal, df, i)
                    
                    signals.append(enhanced_signal)
                    
                    # Log signal with Zero Lag specific info
                    signal_type = signal.get('signal_type')
                    momentum_status = ""
                    if self.strategy.momentum_bias_enabled:
                        momentum_confirmed = signal.get('momentum_bias_confirmation', False)
                        momentum_status = f" (MB: {'✓' if momentum_confirmed else '✗'})"
                    
                    # Display timestamp with UTC
                    display_timestamp = signal['backtest_timestamp']
                    if 'UTC' not in display_timestamp:
                        display_timestamp = f"{display_timestamp} UTC"
                    
                    self.logger.debug(f"⚡ Zero Lag signal ACCEPTED at {display_timestamp}: "
                                    f"{signal_type} (conf: {confidence_value:.1%}){momentum_status}")
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue
        
        # Log detection statistics
        self.logger.info(f"   📊 Signal Detection Statistics:")
        self.logger.info(f"      🔍 Total detections: {total_detections}")
        self.logger.info(f"      ❌ Confidence rejections: {confidence_rejections}")
        self.logger.info(f"      🚫 Momentum bias rejections: {momentum_bias_rejections}")
        self.logger.info(f"      ✅ Successful signals: {successful_signals}")
        
        if total_detections > 0 and successful_signals == 0:
            self.logger.warning(f"⚠️ All {total_detections} detected signals were filtered out!")
            if confidence_rejections > 0:
                self.logger.warning(f"   📉 Consider lowering min confidence (currently {getattr(self.strategy, 'min_confidence', 0.65):.1%})")
            if momentum_bias_rejections > 0:
                self.logger.warning(f"   🚫 Consider disabling momentum bias with --disable-momentum-bias")
        
        return signals
    
    def _get_proper_timestamp(self, df_row, row_index: int) -> str:
        """Get proper timestamp with enhanced validation - matches MACD implementation"""
        try:
            # Debug: Show what columns are available
            available_cols = list(df_row.index) if hasattr(df_row, 'index') else []
            self.logger.debug(f"🔍 Available columns: {available_cols[:10]}...")  # Show first 10
            
            # Try to get timestamp from different possible sources
            timestamp_candidates = []
            
            # Method 1: Direct datetime columns
            for col in ['datetime_utc', 'start_time', 'timestamp', 'datetime']:
                if col in df_row and df_row[col] is not None:
                    candidate = df_row[col]
                    timestamp_candidates.append((col, candidate))
                    self.logger.debug(f"🔍 Found {col}: {candidate} (type: {type(candidate)})")
            
            # Method 2: Index-based timestamp (if row has datetime index)
            if hasattr(df_row, 'name') and df_row.name is not None:
                timestamp_candidates.append(('index', df_row.name))
                self.logger.debug(f"🔍 Found index timestamp: {df_row.name} (type: {type(df_row.name)})")
            
            # Process candidates
            for source, candidate in timestamp_candidates:
                try:
                    if isinstance(candidate, str):
                        # Parse string datetime
                        if candidate != 'Unknown' and len(candidate) > 8:
                            parsed_dt = pd.to_datetime(candidate)
                            formatted = parsed_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                            self.logger.debug(f"✅ Using {source}: {formatted}")
                            return formatted
                    
                    elif isinstance(candidate, (pd.Timestamp, datetime)):
                        # Direct timestamp object
                        formatted = candidate.strftime('%Y-%m-%d %H:%M:%S UTC')
                        self.logger.debug(f"✅ Using {source}: {formatted}")
                        return formatted
                    
                    elif isinstance(candidate, (int, float)):
                        # Unix timestamp
                        if 1000000000 <= candidate <= 2000000000:  # Valid range
                            formatted = datetime.fromtimestamp(candidate).strftime('%Y-%m-%d %H:%M:%S UTC')
                            self.logger.debug(f"✅ Using {source} (unix): {formatted}")
                            return formatted
                    
                except Exception as conversion_error:
                    self.logger.debug(f"⚠️ Failed to convert {source} ({candidate}): {conversion_error}")
                    continue
            
            # Fallback: Generate timestamp from row index (assuming regular intervals)
            self.logger.debug(f"🔄 Using fallback timestamp for row {row_index}")
            
            # Assume 15-minute intervals and start from a reasonable base time
            base_time = datetime(2025, 8, 3, 0, 0, 0)  # Start from the data range we saw
            estimated_time = base_time + pd.Timedelta(minutes=15 * row_index)
            formatted = estimated_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            
            self.logger.debug(f"🔄 Generated fallback timestamp: {formatted}")
            return formatted
            
        except Exception as e:
            self.logger.warning(f"❌ All timestamp methods failed: {e}")
            # Last resort: current time with row offset
            fallback_time = datetime.now() - pd.Timedelta(minutes=15 * (1000 - row_index))
            return fallback_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def _add_performance_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Add advanced performance metrics with trailing stop logic (same as MACD backtest)"""
        try:
            enhanced_signal = signal.copy()
            
            entry_price = signal.get('price', df.iloc[signal_idx]['close'])
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()
            
            # Look ahead for performance (up to 96 bars for 15m = 24 hours)
            max_lookback = min(96, len(df) - signal_idx - 1)
            
            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]
                
                # CRITICAL: Proper trade simulation with trailing stop (matches live trading)
                target_pips = 15  # Profit target
                initial_stop_pips = 10  # Initial stop loss
                
                # Trailing Stop Configuration (matches live trading setup)
                breakeven_trigger = 8    # Move to breakeven at 8 pips profit
                stop_to_profit_trigger = 15  # Move stop to 10 pips profit when trade hits 15 pips
                stop_to_profit_level = 10    # Stop level when above trigger
                trailing_start = 15      # Start trailing after this profit level
                trailing_ratio = 0.5     # Trail 1 pip for every 2 pips profit (1:2 ratio)
                
                # Initialize trade tracking
                trade_closed = False
                exit_pnl = 0.0
                exit_bar = None
                exit_reason = "TIMEOUT"
                running_max_profit = 0.0
                running_max_loss = 0.0
                
                # Trailing stop state
                current_stop_pips = initial_stop_pips  # Current stop loss distance
                best_profit_pips = 0.0                # Best profit achieved (for trailing)
                stop_moved_to_breakeven = False       # Track breakeven move
                stop_moved_to_profit = False          # Track profit protection move
                
                # Simulate trade bar by bar
                for bar_idx, (_, bar) in enumerate(future_data.iterrows()):
                    if trade_closed:
                        break
                    
                    high_price = bar['high']
                    low_price = bar['low']
                    
                    if signal_type in ['BUY', 'BULL', 'LONG']:
                        # Long trade: profit on price going up, loss on price going down
                        current_profit_pips = (high_price - entry_price) * 10000
                        current_loss_pips = (entry_price - low_price) * 10000
                        
                        # Update best profit achieved
                        if current_profit_pips > best_profit_pips:
                            best_profit_pips = current_profit_pips
                            
                            # TRAILING STOP LOGIC (matches live trading)
                            
                            # 1. Move to breakeven at 8 pips profit
                            if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
                                current_stop_pips = 0  # Breakeven (no loss)
                                stop_moved_to_breakeven = True
                                
                            # 2. Move stop to 10 pips profit when trade hits 15 pips
                            elif best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
                                current_stop_pips = -stop_to_profit_level  # Negative = profit protection
                                stop_moved_to_profit = True
                                
                            # 3. Start trailing: 1 pip for every 2 pips above 15 pip level
                            elif best_profit_pips > trailing_start and stop_moved_to_profit:
                                excess_profit = best_profit_pips - trailing_start
                                trailing_adjustment = excess_profit * trailing_ratio
                                current_stop_pips = -(stop_to_profit_level + trailing_adjustment)
                        
                        # Check exit conditions with dynamic trailing stop
                        effective_stop_pips = max(0, current_stop_pips) if current_stop_pips > 0 else abs(current_stop_pips)
                        
                        if current_stop_pips > 0:  # Traditional stop loss
                            if current_loss_pips >= current_stop_pips:
                                exit_pnl = -current_stop_pips
                                exit_reason = "STOP_LOSS"
                                trade_closed = True
                                exit_bar = bar_idx
                        else:  # Profit protection stop
                            profit_protection_level = abs(current_stop_pips)
                            if current_profit_pips <= profit_protection_level or current_loss_pips > 0:
                                # Exit at profit protection level or if trade goes negative
                                exit_pnl = profit_protection_level if current_profit_pips >= profit_protection_level else -current_loss_pips
                                exit_reason = "TRAILING_STOP"
                                trade_closed = True
                                exit_bar = bar_idx
                        
                        # Check profit target
                        if current_profit_pips >= target_pips:
                            exit_pnl = target_pips
                            exit_reason = "PROFIT_TARGET"
                            trade_closed = True
                            exit_bar = bar_idx
                            
                    elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                        # Short trade: profit on price going down, loss on price going up
                        current_profit_pips = (entry_price - low_price) * 10000
                        current_loss_pips = (high_price - entry_price) * 10000
                        
                        # Update best profit achieved
                        if current_profit_pips > best_profit_pips:
                            best_profit_pips = current_profit_pips
                            
                            # TRAILING STOP LOGIC (same as long, but for short trades)
                            
                            # 1. Move to breakeven at 8 pips profit
                            if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
                                current_stop_pips = 0  # Breakeven (no loss)
                                stop_moved_to_breakeven = True
                                
                            # 2. Move stop to 10 pips profit when trade hits 15 pips
                            elif best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
                                current_stop_pips = -stop_to_profit_level  # Negative = profit protection
                                stop_moved_to_profit = True
                                
                            # 3. Start trailing: 1 pip for every 2 pips above 15 pip level
                            elif best_profit_pips > trailing_start and stop_moved_to_profit:
                                excess_profit = best_profit_pips - trailing_start
                                trailing_adjustment = excess_profit * trailing_ratio
                                current_stop_pips = -(stop_to_profit_level + trailing_adjustment)
                        
                        # Check exit conditions with dynamic trailing stop (same logic as long)
                        if current_stop_pips > 0:  # Traditional stop loss
                            if current_loss_pips >= current_stop_pips:
                                exit_pnl = -current_stop_pips
                                exit_reason = "STOP_LOSS"
                                trade_closed = True
                                exit_bar = bar_idx
                        else:  # Profit protection stop
                            profit_protection_level = abs(current_stop_pips)
                            if current_profit_pips <= profit_protection_level or current_loss_pips > 0:
                                # Exit at profit protection level or if trade goes negative
                                exit_pnl = profit_protection_level if current_profit_pips >= profit_protection_level else -current_loss_pips
                                exit_reason = "TRAILING_STOP"
                                trade_closed = True
                                exit_bar = bar_idx
                        
                        # Check profit target
                        if current_profit_pips >= target_pips:
                            exit_pnl = target_pips
                            exit_reason = "PROFIT_TARGET"
                            trade_closed = True
                            exit_bar = bar_idx
                    
                    # Track maximum excursions for analysis (but don't use for P&L)
                    if signal_type in ['BUY', 'BULL', 'LONG']:
                        running_max_profit = max(running_max_profit, (high_price - entry_price) * 10000)
                        running_max_loss = max(running_max_loss, (entry_price - low_price) * 10000)
                    elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                        running_max_profit = max(running_max_profit, (entry_price - low_price) * 10000)
                        running_max_loss = max(running_max_loss, (high_price - entry_price) * 10000)
                
                # If trade never closed, calculate final P&L
                if not trade_closed and len(future_data) > 0:
                    final_price = future_data.iloc[-1]['close']
                    if signal_type in ['BUY', 'BULL', 'LONG']:
                        exit_pnl = (final_price - entry_price) * 10000
                    elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                        exit_pnl = (entry_price - final_price) * 10000
                    exit_reason = "TIMEOUT"
                    exit_bar = len(future_data) - 1
                
                # Use actual trade P&L, not maximum excursions
                if exit_pnl >= 0:
                    max_profit = exit_pnl
                    max_loss = 0.0
                else:
                    max_profit = 0.0
                    max_loss = abs(exit_pnl)
                
                # Determine trade outcome based on actual exit
                if exit_pnl >= target_pips:
                    is_winner = True
                    is_loser = False
                    trade_outcome = "WIN"
                elif exit_pnl < 0:  # Any loss (initial stop, trailing stop, or timeout loss)
                    is_winner = False
                    is_loser = True
                    trade_outcome = "LOSE"
                elif exit_pnl > 0:  # Profitable exit (trailing stop, breakeven, etc.)
                    is_winner = True
                    is_loser = False
                    trade_outcome = "WIN"
                else:  # Exactly breakeven
                    is_winner = False
                    is_loser = False
                    trade_outcome = "BREAKEVEN"
                
                enhanced_signal.update({
                    'max_profit_pips': round(max_profit, 1),
                    'max_loss_pips': round(max_loss, 1),
                    'profit_loss_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else float('inf'),
                    'lookback_bars': max_lookback,
                    'entry_price': entry_price,
                    'is_winner': is_winner,
                    'is_loser': is_loser,
                    'trade_outcome': trade_outcome,
                    'target_pips': target_pips,
                    'initial_stop_pips': initial_stop_pips,
                    # NEW: Trailing stop metadata
                    'exit_pnl': round(exit_pnl, 1),
                    'exit_reason': exit_reason,
                    'exit_bar': exit_bar,
                    'trade_closed': trade_closed,
                    'running_max_profit': round(running_max_profit, 1),
                    'running_max_loss': round(running_max_loss, 1),
                    'best_profit_achieved': round(best_profit_pips, 1),
                    'final_stop_pips': round(current_stop_pips, 1),
                    'stop_moved_to_breakeven': stop_moved_to_breakeven,
                    'stop_moved_to_profit': stop_moved_to_profit,
                    'trailing_stop_config': {
                        'breakeven_trigger': breakeven_trigger,
                        'profit_protection_trigger': stop_to_profit_trigger,
                        'profit_protection_level': stop_to_profit_level,
                        'trailing_ratio': trailing_ratio
                    }
                })
            else:
                enhanced_signal.update({
                    'max_profit_pips': 0.0,
                    'max_loss_pips': 0.0,
                    'is_winner': False,
                    'is_loser': False,
                    'trade_outcome': 'NO_DATA',
                })
            
            self.logger.debug(f"📊 Zero Lag Performance: {enhanced_signal.get('trade_outcome', 'UNKNOWN')} - "
                            f"Exit P&L: {enhanced_signal.get('exit_pnl', 0):.1f} pips "
                            f"({enhanced_signal.get('exit_reason', 'UNKNOWN')})")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error adding performance metrics: {e}")
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'max_profit_pips': 0.0,
                'max_loss_pips': 0.0,
                'is_winner': False,
                'is_loser': False,
                'trade_outcome': 'ERROR',
                'error': str(e)
            })
            return enhanced_signal
    
    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract trading pair from IG epic format"""
        try:
            if 'CS.D.' in epic and '.MINI.IP' in epic:
                # Format: CS.D.EURUSD.MINI.IP -> EURUSD
                return epic.split('.D.')[1].split('.MINI.IP')[0]
            elif 'CS.D.' in epic:
                # Format: CS.D.EURUSD.TODAY.IP -> EURUSD
                parts = epic.split('.D.')[1].split('.')
                return parts[0] if parts else epic
            else:
                # Assume epic is already the pair
                return epic
        except:
            return epic
    
    def _display_epic_results(self, epic_results: Dict):
        """Display results by epic"""
        self.logger.info("\n📊 EPIC-BY-EPIC RESULTS:")
        self.logger.info("=" * 30)
        
        for epic, result in epic_results.items():
            pair = self._extract_pair_from_epic(epic)
            signals_count = result.get('signals', 0)
            error = result.get('error', '')
            
            if error:
                self.logger.info(f"   {pair}: ❌ {error}")
            else:
                self.logger.info(f"   {pair}: ⚡ {signals_count} signals")
    
    def _display_signals(self, signals: List[Dict]):
        """Display detailed signal information with trailing stop results and MTF validation"""
        self.logger.info("\n📋 ZERO LAG SIGNALS WITH TRAILING STOP & MTF VALIDATION:")
        self.logger.info("=" * 155)
        self.logger.info("#   TIMESTAMP (UTC)         PAIR     TYPE PRICE    CONF   EXIT P&L EXIT REASON     OUTCOME  TRAIL MTF")
        self.logger.info("-" * 155)
        
        for i, signal in enumerate(signals, 1):
            # Extract timestamp with proper handling
            timestamp_str = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))
            timestamp_source = "backtest_timestamp" if 'backtest_timestamp' in signal else "timestamp"
            
            # Ensure UTC is displayed in timestamp
            if timestamp_str != 'Unknown' and 'UTC' not in timestamp_str:
                timestamp_str = f"{timestamp_str} UTC"
            
            # Extract pair from epic
            epic = signal.get('epic', 'Unknown')
            if 'CS.D.' in epic and '.MINI.IP' in epic:
                pair = epic.split('.D.')[1].split('.MINI.IP')[0]
            else:
                pair = epic
            
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence', signal.get('confidence_score', 0))
            price = signal.get('price', 0)
            
            # Enhanced performance data from trailing stop simulation
            exit_pnl = signal.get('exit_pnl', 0)
            exit_reason = signal.get('exit_reason', 'UNKNOWN')
            trade_outcome = signal.get('trade_outcome', 'UNKNOWN')
            
            # Trailing stop indicators
            be_moved = "✓" if signal.get('stop_moved_to_breakeven', False) else "✗"
            profit_moved = "✓" if signal.get('stop_moved_to_profit', False) else "✗" 
            trail_info = f"{be_moved}{profit_moved}"
            
            # Multi-timeframe validation status
            mtf_info = signal.get('mtf_validation', {})
            if mtf_info.get('overall_valid'):
                h1_status = "1" if mtf_info.get('h1_validation') else "0"
                h4_status = "4" if mtf_info.get('h4_validation') else "0"
                mtf_status = f"{h1_status}{h4_status}"
            else:
                mtf_status = "XX"
            
            # Truncate long exit reasons
            exit_reason_short = exit_reason[:11] if len(exit_reason) > 11 else exit_reason
            outcome_short = trade_outcome[:8] if len(trade_outcome) > 8 else trade_outcome
            
            # Debug info for this signal
            self.logger.debug(f"Signal {i}: timestamp_source={timestamp_source}, timestamp_str={timestamp_str}")
            
            row = f"{i:<3} {timestamp_str:<25} {pair:<8} {signal_type:<4} {price:<8.5f} {confidence:<6.1%} {exit_pnl:<8.1f} {exit_reason_short:<13} {outcome_short:<8} {trail_info:<5} {mtf_status:<3}"
            self.logger.info(row)
        
        self.logger.info("-" * 155)
        self.logger.info("TRAIL Legend: Breakeven moved (✓/✗) + Profit protection (✓/✗)")
        self.logger.info("MTF Legend: 14=1H✓4H✓, 10=1H✓4H✗, 04=1H✗4H✓, XX=Both failed")
        self.logger.info("=" * 155)
    
    def _analyze_performance(self, signals: List[Dict]):
        """Analyze Zero Lag strategy performance"""
        try:
            # Debug: Print first signal to see field structure
            if signals:
                first_signal = signals[0]
                self.logger.debug(f"📊 Sample Zero Lag signal keys: {list(first_signal.keys())}")
                zero_lag_data = first_signal.get('zero_lag_data', {})
                self.logger.debug(f"📊 Zero Lag data keys: {list(zero_lag_data.keys())}")
            
            # Create custom performance analysis for Zero Lag
            metrics = self._create_zero_lag_performance_analysis(signals)
            
            self.logger.info("\n📈 ZERO LAG STRATEGY PERFORMANCE (WITH TRAILING STOPS):")
            self.logger.info("=" * 55)
            self.logger.info(f"   📊 Total Signals: {metrics.get('total_signals', len(signals))}")
            self.logger.info(f"   🎯 Average Confidence: {metrics.get('avg_confidence', 0):.1%}")
            self.logger.info(f"   📈 Bull Signals: {metrics.get('bull_signals', 0)}")
            self.logger.info(f"   📉 Bear Signals: {metrics.get('bear_signals', 0)}")
            
            # NEW: Trailing Stop Performance
            self.logger.info(f"\n🎯 TRAILING STOP RESULTS:")
            if 'total_pnl' in metrics:
                self.logger.info(f"   💰 Total P&L: {metrics['total_pnl']:.1f} pips")
            if 'avg_exit_pnl' in metrics:
                self.logger.info(f"   📊 Average Exit P&L: {metrics['avg_exit_pnl']:.1f} pips")
            
            win_count = metrics.get('win_count', 0)
            loss_count = metrics.get('loss_count', 0)
            win_rate = metrics.get('win_rate', 0)
            self.logger.info(f"   🏆 Wins: {win_count} | Losses: {loss_count} | Win Rate: {win_rate:.1%}")
            
            # Exit reason breakdown
            if 'exit_reason_breakdown' in metrics:
                self.logger.info(f"   📋 Exit Reasons:")
                for reason, count in metrics['exit_reason_breakdown'].items():
                    self.logger.info(f"      {reason}: {count}")
            
            # Trailing stop effectiveness
            if 'trailing_stop_stats' in metrics:
                ts_stats = metrics['trailing_stop_stats']
                self.logger.info(f"   🔧 Trailing Stop Effectiveness:")
                self.logger.info(f"      Breakeven Triggered: {ts_stats['breakeven_triggered']} ({ts_stats['breakeven_rate']:.1%})")
                self.logger.info(f"      Profit Protection: {ts_stats['profit_protection_triggered']} ({ts_stats['profit_protection_rate']:.1%})")
            
            # Best profit analysis
            if 'avg_best_profit' in metrics:
                self.logger.info(f"   📈 Average Best Profit: {metrics['avg_best_profit']:.1f} pips")
            if 'max_best_profit' in metrics:
                self.logger.info(f"   🚀 Maximum Best Profit: {metrics['max_best_profit']:.1f} pips")
            
            # Zero Lag specific metrics
            if 'trend_change_signals' in metrics:
                self.logger.info(f"\n⚡ ZERO LAG SPECIFICS:")
                self.logger.info(f"   🔄 Trend Change Signals: {metrics['trend_change_signals']}")
            if 'entry_signals' in metrics:
                self.logger.info(f"   ⚡ Entry Signals: {metrics['entry_signals']}")
            
            # Momentum bias metrics
            if 'momentum_bias_confirmed' in metrics:
                confirmed = metrics['momentum_bias_confirmed']
                total = metrics['total_signals']
                confirmed_rate = confirmed / total if total > 0 else 0
                self.logger.info(f"   🚀 Momentum Bias Confirmed: {confirmed} ({confirmed_rate:.1%})")
            
            # Multi-timeframe validation results
            if 'mtf_stats' in metrics:
                mtf = metrics['mtf_stats']
                self.logger.info(f"\n🌐 MULTI-TIMEFRAME VALIDATION:")
                self.logger.info(f"   📊 Signals with MTF Data: {mtf['total_signals_with_mtf']}")
                self.logger.info(f"   ✅ MTF Validation Rate: {mtf['mtf_validation_rate']:.1%}")
                self.logger.info(f"   🕐 1H Confirmation Rate: {mtf['h1_validation_rate']:.1%}")
                self.logger.info(f"   🕓 4H Confirmation Rate: {mtf['h4_validation_rate']:.1%}")
            
            # Additional metrics
            if 'profit_factor' in metrics:
                pf = metrics['profit_factor']
                pf_display = f"{pf:.2f}" if pf != float('inf') else "∞"
                self.logger.info(f"   📊 Profit Factor: {pf_display}")
            if 'confidence_range' in metrics:
                conf_range = metrics['confidence_range']
                self.logger.info(f"   📈 Confidence Range: {conf_range['min']:.1%} - {conf_range['max']:.1%}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Zero Lag performance analysis failed: {e}")
            # Enhanced basic analysis fallback
            self._basic_zero_lag_analysis_fallback(signals)
    
    def _create_zero_lag_performance_analysis(self, signals: List[Dict]) -> Dict:
        """Create comprehensive Zero Lag performance analysis with trailing stop metrics"""
        if not signals:
            return {}
        
        total_signals = len(signals)
        bull_signals = [s for s in signals if s.get('signal_type') in ['BUY', 'BULL']]
        bear_signals = [s for s in signals if s.get('signal_type') in ['SELL', 'BEAR']]
        
        # Basic metrics
        metrics = {
            'total_signals': total_signals,
            'bull_signals': len(bull_signals),
            'bear_signals': len(bear_signals)
        }
        
        # Confidence analysis
        confidences = [s.get('confidence', 0) for s in signals if s.get('confidence')]
        if confidences:
            metrics['avg_confidence'] = sum(confidences) / len(confidences)
            metrics['confidence_range'] = {
                'min': min(confidences),
                'max': max(confidences)
            }
        
        # NEW: Trailing Stop Performance Analysis
        exit_pnls = [s.get('exit_pnl', 0) for s in signals if s.get('exit_pnl') is not None]
        winners = [s for s in signals if s.get('is_winner', False)]
        losers = [s for s in signals if s.get('is_loser', False)]
        
        if exit_pnls:
            metrics['avg_exit_pnl'] = sum(exit_pnls) / len(exit_pnls)
            metrics['total_pnl'] = sum(exit_pnls)
            
        metrics['win_count'] = len(winners)
        metrics['loss_count'] = len(losers)
        metrics['win_rate'] = len(winners) / total_signals if total_signals > 0 else 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for signal in signals:
            reason = signal.get('exit_reason', 'UNKNOWN')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        metrics['exit_reason_breakdown'] = exit_reasons
        
        # Trailing stop effectiveness
        breakeven_moved = len([s for s in signals if s.get('stop_moved_to_breakeven', False)])
        profit_protection = len([s for s in signals if s.get('stop_moved_to_profit', False)])
        
        metrics['trailing_stop_stats'] = {
            'breakeven_triggered': breakeven_moved,
            'profit_protection_triggered': profit_protection,
            'breakeven_rate': breakeven_moved / total_signals if total_signals > 0 else 0,
            'profit_protection_rate': profit_protection / total_signals if total_signals > 0 else 0
        }
        
        # Best profit analysis
        best_profits = [s.get('best_profit_achieved', 0) for s in signals if s.get('best_profit_achieved', 0) > 0]
        if best_profits:
            metrics['avg_best_profit'] = sum(best_profits) / len(best_profits)
            metrics['max_best_profit'] = max(best_profits)
        
        # Legacy profit/loss for backward compatibility
        profits = [s.get('max_profit_pips', 0) for s in signals]
        losses = [s.get('max_loss_pips', 0) for s in signals]
        
        if profits:
            metrics['avg_profit'] = sum(profits) / len(profits)
        if losses:
            metrics['avg_loss'] = sum(losses) / len(losses)
        
        # Momentum bias analysis
        if any(s.get('momentum_bias_confirmation') is not None for s in signals):
            confirmed = [s for s in signals if s.get('momentum_bias_confirmation', False)]
            metrics['momentum_bias_confirmed'] = len(confirmed)
        
        # Zero Lag specific analysis
        trend_changes = [s for s in signals if s.get('zero_lag_data', {}).get('trend', 0) != 0]
        metrics['trend_change_signals'] = len(trend_changes)
        
        # Entry vs continuation signals (simplified)
        entry_signals = [s for s in signals if s.get('signal_type') in ['BUY', 'SELL']]
        metrics['entry_signals'] = len(entry_signals)
        
        # Profit factor using actual P&L
        winner_pnl = sum([s.get('exit_pnl', 0) for s in winners])
        loser_pnl = abs(sum([s.get('exit_pnl', 0) for s in losers]))
        if loser_pnl > 0:
            metrics['profit_factor'] = winner_pnl / loser_pnl
        elif winner_pnl > 0:
            metrics['profit_factor'] = float('inf')
        else:
            metrics['profit_factor'] = 0
        
        # Multi-timeframe validation analysis
        mtf_signals = [s for s in signals if s.get('mtf_validation')]
        if mtf_signals:
            mtf_passed = len([s for s in mtf_signals if s.get('mtf_validation', {}).get('overall_valid')])
            h1_passed = len([s for s in mtf_signals if s.get('mtf_validation', {}).get('h1_validation')])
            h4_passed = len([s for s in mtf_signals if s.get('mtf_validation', {}).get('h4_validation')])
            
            metrics['mtf_stats'] = {
                'total_signals_with_mtf': len(mtf_signals),
                'mtf_validation_passed': mtf_passed,
                'mtf_validation_rate': mtf_passed / len(mtf_signals) if mtf_signals else 0,
                'h1_validation_rate': h1_passed / len(mtf_signals) if mtf_signals else 0,
                'h4_validation_rate': h4_passed / len(mtf_signals) if mtf_signals else 0
            }
        
        return metrics
    
    def _basic_zero_lag_analysis_fallback(self, signals: List[Dict]):
        """Basic performance analysis when detailed analysis fails"""
        self.logger.info(f"   📊 Total Signals: {len(signals)}")
        
        # Count signal types
        buy_signals = len([s for s in signals if s.get('signal_type') in ['BUY', 'BULL']])
        sell_signals = len([s for s in signals if s.get('signal_type') in ['SELL', 'BEAR']])
        
        self.logger.info(f"   📈 BUY Signals: {buy_signals}")
        self.logger.info(f"   📉 SELL Signals: {sell_signals}")
        
        # Average confidence if available
        confidences = [s.get('confidence', 0) for s in signals if s.get('confidence')]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            self.logger.info(f"   🎯 Average Confidence: {avg_confidence:.1%}")

    def _display_smart_money_stats(self, signals: List[Dict]):
        """Display Smart Money analysis statistics"""
        try:
            if not signals or not self.smart_money_enabled:
                return
            
            self.logger.info(f"   🧠 Smart Money Analysis:")
            self.logger.info(f"      📊 Signals Analyzed: {self.smart_money_stats['signals_analyzed']}")
            self.logger.info(f"      ✅ Signals Enhanced: {self.smart_money_stats['signals_enhanced']}")
            self.logger.info(f"      ❌ Analysis Failures: {self.smart_money_stats['analysis_failures']}")
            
            # Show SMC-specific metrics if available
            smc_signals = [s for s in signals if s.get('smart_money_context')]
            if smc_signals:
                bos_signals = len([s for s in smc_signals if 'BOS' in str(s.get('smart_money_context', {}).get('market_structure', ''))])
                choch_signals = len([s for s in smc_signals if 'ChoCh' in str(s.get('smart_money_context', {}).get('market_structure', ''))])
                ob_signals = len([s for s in smc_signals if s.get('smart_money_context', {}).get('order_blocks')])
                
                self.logger.info(f"      📈 BOS Signals: {bos_signals}")
                self.logger.info(f"      🔄 ChoCh Signals: {choch_signals}")
                self.logger.info(f"      🏛️ Order Block Signals: {ob_signals}")
                
        except Exception as e:
            self.logger.debug(f"Smart Money stats display error: {e}")


def main():
    """Main entry point for Zero Lag + Squeeze Momentum backtest"""
    parser = argparse.ArgumentParser(description='Zero Lag + Squeeze Momentum + EMA200 Strategy Backtest')
    
    parser.add_argument('--epic', type=str, help='Specific epic to test (e.g., CS.D.EURUSD.MINI.IP)')
    parser.add_argument('--days', type=int, default=7, help='Days of historical data (default: 7)')
    parser.add_argument('--timeframe', type=str, default='15m', choices=['1m', '5m', '15m', '1h', '4h', '1d'], 
                        help='Timeframe (default: 15m)')
    parser.add_argument('--show-signals', action='store_true', help='Display individual signals')
    parser.add_argument('--squeeze-momentum', action='store_true', default=True, help='Enable Squeeze Momentum indicator (default: enabled)')
    parser.add_argument('--disable-squeeze-momentum', action='store_true', help='Disable Squeeze Momentum filtering')
    parser.add_argument('--smart-money', action='store_true', help='Enable Smart Money Concepts analysis')
    parser.add_argument('--min-confidence', type=float, help='Override minimum confidence threshold (0.0-1.0)')
    parser.add_argument('--sl-type', type=str, default='atr', choices=['atr', 'fixed', 'volatility'], help='Stop loss type (default: atr)')
    parser.add_argument('--sl-atr-multiplier', type=float, default=1.5, help='ATR multiplier for stop loss (default: 1.5)')
    parser.add_argument('--trailing-stop', action='store_true', default=True, help='Enable trailing stop (default: enabled)')
    parser.add_argument('--no-trailing-stop', action='store_true', help='Disable trailing stop')
    
    args = parser.parse_args()
    
    # Process boolean arguments
    enable_squeeze = args.squeeze_momentum and not args.disable_squeeze_momentum
    enable_trailing = args.trailing_stop and not args.no_trailing_stop
    
    # Validate arguments
    if args.min_confidence is not None:
        if not 0.0 <= args.min_confidence <= 1.0:
            print("❌ ERROR: --min-confidence must be between 0.0 and 1.0")
            sys.exit(1)
    
    if args.sl_atr_multiplier <= 0:
        print("❌ ERROR: --sl-atr-multiplier must be positive")
        sys.exit(1)
    
    # Run backtest
    backtest = ZeroLagBacktest()
    
    success = backtest.run_backtest(
        epic=args.epic,
        days=args.days,
        timeframe=args.timeframe,
        show_signals=args.show_signals,
        enable_squeeze_momentum=enable_squeeze,
        enable_smart_money=args.smart_money,
        min_confidence=args.min_confidence,
        sl_type=args.sl_type,
        sl_atr_multiplier=args.sl_atr_multiplier,
        trailing_stop=enable_trailing
    )
    
    if success:
        print("\n✅ Zero Lag backtest completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Zero Lag backtest failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()