#!/usr/bin/env python3
"""
ENHANCED MACD Strategy Backtest with Smart Money Analysis - Complete Integration
Run: python backtest_macd.py --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m --smart-money

ENHANCEMENTS ADDED:
- Smart Money Concepts (SMC) integration
- Market Structure Analysis (BOS, ChoCh, Swing Points)
- Order Flow Analysis (Order Blocks, Fair Value Gaps)
- Institutional Supply/Demand Zone Detection
- Enhanced signal validation with smart money confluence
- Smart money performance metrics and analysis
- FIXED: Proper signal display and sorting like EMA backtest
- NEW: Single signal validation feature with detailed data inspection
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json

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
from core.strategies.macd_strategy import create_macd_strategy, MACDStrategy
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

from configdata import config as macd_config
try:
    import config
except ImportError:
    from forex_scanner import config


class EnhancedMACDBacktest:
    """Enhanced MACD Strategy Backtesting with Smart Money Analysis Integration"""
    
    def __init__(self):
        self.logger = logging.getLogger('enhanced_macd_backtest')
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
        
        Args:
            timestamp: Timestamp in various formats
            
        Returns:
            datetime: UTC datetime (timezone-naive)
        """
        try:
            from datetime import datetime
            
            # Normalize the timestamp first
            if isinstance(timestamp, str):
                dt = pd.to_datetime(timestamp)
            elif isinstance(timestamp, pd.Timestamp):
                dt = timestamp.to_pydatetime()
            elif isinstance(timestamp, (int, float)):
                # Unix timestamp
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = timestamp
            
            # If timezone-aware, convert to UTC and make naive
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                import pytz
                if dt.tzinfo != pytz.UTC:
                    dt = dt.astimezone(pytz.UTC)
                dt = dt.replace(tzinfo=None)
            
            # Return timezone-naive UTC datetime
            return dt
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not process timestamp: {e}")
            # Return original timestamp as-is
            if isinstance(timestamp, str):
                return pd.to_datetime(timestamp)
            elif isinstance(timestamp, pd.Timestamp):
                return timestamp.to_pydatetime()
            else:
                return timestamp
    
    def initialize_smart_money_analysis(self, enabled: bool = True):
        """Initialize Smart Money analysis components"""
        if not enabled or not SMART_MONEY_AVAILABLE:
            self.logger.info("🧠 Smart Money Analysis: ❌ DISABLED")
            self.smart_money_enabled = False
            return False
        
        try:
            # Initialize Smart Money components
            self.smart_money_integration = SmartMoneyIntegration(
                self.db_manager, 
                self.data_fetcher
            )
            self.smart_money_analyzer = SmartMoneyReadOnlyAnalyzer(self.data_fetcher)
            
            self.smart_money_enabled = True
            self.logger.info("🧠 Smart Money Analysis: ✅ ENABLED")
            self.logger.info("   📊 Market Structure Analysis: Available")
            self.logger.info("   🔄 Order Flow Analysis: Available") 
            self.logger.info("   🎯 Institutional Zone Detection: Available")
            self.logger.info("   📈 Smart Money Confluence: Available")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Smart Money initialization failed: {e}")
            self.smart_money_enabled = False
            return False
    
    def initialize_macd_strategy(self, optimization_config: Dict = None, epic: str = None, timeframe: str = '15m', use_optimal_parameters: bool = True):
        """ENHANCED: Initialize MACD strategy with database optimization support"""
        
        # Log optimization status
        if use_optimal_parameters and epic:
            self.logger.info(f"🎯 Initializing MACD strategy with DATABASE OPTIMIZATION for {epic}")
        else:
            self.logger.info(f"📊 Initializing MACD strategy with STATIC CONFIGURATION")
        
        # Use the new timeframe-aware strategy creation with enhanced optimization
        self.strategy = MACDStrategy(
            data_fetcher=self.data_fetcher,
            backtest_mode=True,
            epic=epic,
            timeframe=timeframe,
            use_optimized_parameters=use_optimal_parameters  # Enhanced to use parameter
        )
        self.logger.info(f"✅ Timeframe-aware MACD Strategy initialized for backtest ({timeframe})")
        
        # 🔍 CHECK MTF STATUS
        mtf_enabled = getattr(self.strategy, 'enable_mtf_analysis', False)
        mtf_analyzer = getattr(self.strategy, 'mtf_analyzer', None)
        has_mtf_method = hasattr(self.strategy, 'detect_signal_with_mtf')
        
        self.logger.info("🔍 [BACKTEST] MTF Status Check:")
        self.logger.info(f"   enable_mtf_analysis: {mtf_enabled}")
        self.logger.info(f"   mtf_analyzer exists: {mtf_analyzer is not None}")
        self.logger.info(f"   detect_signal_with_mtf exists: {has_mtf_method}")
        
        if mtf_enabled and mtf_analyzer and has_mtf_method:
            self.logger.info("✅ [BACKTEST] MTF Analysis READY - backtest will use MTF-enhanced detection")
        else:
            self.logger.warning("⚠️ [BACKTEST] MTF Analysis NOT READY - backtest will use standard detection")
            if not mtf_enabled:
                self.logger.warning("   - enable_mtf_analysis is False")
            if not mtf_analyzer:
                self.logger.warning("   - mtf_analyzer is missing")
            if not has_mtf_method:
                self.logger.warning("   - detect_signal_with_mtf method missing")
        
        # Apply optimization configuration if provided
        if optimization_config:
            self.logger.info(f"🔧 Applying optimization config: {optimization_config}")
            
            for pair, config_updates in optimization_config.items():
                try:
                    self.strategy.optimize_for_pair(pair, config_updates)
                    self.logger.info(f"   ✅ Optimized {pair} configuration")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Failed to optimize {pair}: {e}")
        
        # Log strategy status
        self.strategy.log_modular_status()
        
        # Log MACD configuration from actual strategy
        self.logger.info("   📊 MACD Parameters:")
        self.logger.info(f"     Fast EMA: {getattr(self.strategy, 'fast_ema', 12)}")
        self.logger.info(f"     Slow EMA: {getattr(self.strategy, 'slow_ema', 26)}") 
        self.logger.info(f"     Signal EMA: {getattr(self.strategy, 'signal_ema', 9)}")
        self.logger.info(f"     Timeframe: {getattr(self.strategy, 'timeframe', '15m')}")
        self.logger.info(f"     Epic: {getattr(self.strategy, 'epic', 'None')}")
        self.logger.info("     EMA200 filter: Enabled")
        self.logger.info("     MTF Analysis: ✅ Enabled" if mtf_enabled else "     MTF Analysis: ❌ Disabled")
        self.logger.info("     Smart Money Analysis: ✅ Enabled" if self.smart_money_enabled else "     Smart Money Analysis: ❌ Disabled")
        self.logger.info("     Architecture: Timeframe-aware with database optimization")
        
        return self.strategy
    
    def validate_single_signal(
        self,
        epic: str,
        timestamp: str,
        timeframe: str = None,
        show_raw_data: bool = False,
        show_calculations: bool = True,
        show_decision_tree: bool = True
    ) -> bool:
        """
        NEW FEATURE: Validate a single signal and show all data and calculations used
        
        Args:
            epic: Epic to analyze (e.g., CS.D.EURUSD.MINI.IP)
            timestamp: Timestamp of the signal to validate (e.g., "2025-08-04 15:30:00")
            timeframe: Timeframe to use for analysis
            show_raw_data: Show raw OHLC data around the signal
            show_calculations: Show detailed MACD and EMA calculations
            show_decision_tree: Show the decision-making process
        """
        
        # Use default timeframe if not specified (ALIGNMENT FIX)
        if timeframe is None:
            timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '15m')
            self.logger.debug(f"🔧 Using DEFAULT_TIMEFRAME from config: {timeframe}")
        
        self.logger.info("🔍 SINGLE SIGNAL VALIDATION")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Epic: {epic}")
        self.logger.info(f"⏰ Timestamp: {timestamp}")
        self.logger.info(f"📈 Timeframe: {timeframe}")
        
        try:
            # Initialize strategy with timeframe awareness
            self.initialize_macd_strategy(epic=epic, timeframe=timeframe)
            
            # Extract pair from epic
            pair = self._extract_pair_from_epic(epic)
            
            # Parse target timestamp with timezone handling
            try:
                target_time = pd.to_datetime(timestamp)
                # Ensure target_time is timezone-naive for comparison
                if target_time.tz is not None:
                    target_time = target_time.tz_localize(None)
            except Exception as e:
                self.logger.error(f"❌ Invalid timestamp format: {timestamp}")
                self.logger.error(f"   Use format: YYYY-MM-DD HH:MM:SS")
                self.logger.error(f"   Error: {e}")
                return False
            
            # Get extended data around the target timestamp (increased lookback)
            df = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=72  # Increased from 48 to ensure enough data for MACD
            )

            # 🔍 ADD THIS DEBUG:
            print(f"🔍 BACKTESTER: After get_enhanced_data: {df.shape if df is not None else 'None'}")
            if df is not None and len(df) > 1:
                print(f"Backtester First: {df.iloc[0]['start_time']} - Close: {df.iloc[0]['close']}")
                print(f"Backtester Last: {df.iloc[-1]['start_time']} - Close: {df.iloc[-1]['close']}")
            
            if df.empty:
                self.logger.error(f"❌ No data available for {epic}")
                return False
            
            # Find the closest data point to the target timestamp
            # Try different timestamp column names that might be available
            timestamp_cols = ['datetime_utc', 'start_time', 'timestamp', 'datetime']
            df_with_time = None
            used_col = None
            
            for col in timestamp_cols:
                if col in df.columns:
                    try:
                        df['datetime_parsed'] = pd.to_datetime(df[col], errors='coerce')
                        # Ensure timezone-naive for comparison
                        if df['datetime_parsed'].dt.tz is not None:
                            df['datetime_parsed'] = df['datetime_parsed'].dt.tz_localize(None)
                        
                        df_with_time = df.dropna(subset=['datetime_parsed'])
                        if not df_with_time.empty:
                            used_col = col
                            self.logger.debug(f"✅ Using timestamp column: {col}")
                            break
                    except Exception as e:
                        self.logger.debug(f"⚠️ Failed to parse column {col}: {e}")
                        continue
            
            # Try using DataFrame index if it contains timestamps
            if df_with_time is None or df_with_time.empty:
                try:
                    if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
                        df['datetime_parsed'] = df.index
                        # Ensure timezone-naive
                        if hasattr(df['datetime_parsed'].dtype, 'tz') and df['datetime_parsed'].dt.tz is not None:
                            df['datetime_parsed'] = df['datetime_parsed'].dt.tz_localize(None)
                        
                        df_with_time = df.copy()
                        used_col = 'index'
                        self.logger.debug(f"✅ Using DataFrame index as timestamp")
                    elif len(df) > 0:
                        # Try to convert index to datetime
                        df['datetime_parsed'] = pd.to_datetime(df.index, errors='coerce')
                        # Ensure timezone-naive
                        if df['datetime_parsed'].dt.tz is not None:
                            df['datetime_parsed'] = df['datetime_parsed'].dt.tz_localize(None)
                            
                        df_with_time = df.dropna(subset=['datetime_parsed'])
                        if not df_with_time.empty:
                            used_col = 'converted_index'
                            self.logger.debug(f"✅ Using converted DataFrame index")
                except Exception as e:
                    self.logger.debug(f"⚠️ Failed to use index: {e}")
            
            if df_with_time is None or df_with_time.empty:
                self.logger.error(f"❌ No valid timestamps found in data")
                self.logger.error(f"   Available columns: {list(df.columns)}")
                self.logger.error(f"   DataFrame index type: {type(df.index)}")
                return False
            
            self.logger.info(f"📅 Using timestamp column: {used_col}")
            
            # Find closest timestamp with proper timezone handling
            try:
                time_diffs = abs(df_with_time['datetime_parsed'] - target_time)
                closest_idx = time_diffs.idxmin()
                closest_time = df_with_time.loc[closest_idx, 'datetime_parsed']
                
                self.logger.info(f"🎯 Closest data point: {closest_time}")
                self.logger.info(f"   Time difference: {abs(closest_time - target_time)}")
            except Exception as e:
                self.logger.error(f"❌ Failed to find closest timestamp: {e}")
                self.logger.error(f"   Target time: {target_time} (type: {type(target_time)})")
                self.logger.error(f"   Data time example: {df_with_time['datetime_parsed'].iloc[0] if len(df_with_time) > 0 else 'None'}")
                return False
            
            # Get the data context around this point (ensure enough for MACD)
            data_idx = df_with_time.index.get_loc(closest_idx)
            min_bars = getattr(config, 'MIN_BARS_FOR_MACD', 50)
            
            # Ensure we have enough data before the signal point for MACD calculation
            context_start = max(0, data_idx - min_bars)  # Go back enough for MACD
            context_end = min(len(df), data_idx + 10)     # Small window after for context
            
            validation_data = df.iloc[context_start:context_end + 1].copy()
            signal_row_idx = data_idx - context_start  # Index within validation_data
            
            if signal_row_idx >= len(validation_data):
                self.logger.error(f"❌ Signal row index out of bounds")
                return False
            
            self.logger.info(f"📊 Analysis context: {len(validation_data)} data points")
            self.logger.info(f"   Signal at index: {signal_row_idx}")
            
            # Show raw data if requested
            if show_raw_data:
                self._show_raw_data_context(validation_data, signal_row_idx, closest_time)
            
            # Show detailed calculations
            if show_calculations:
                self._show_detailed_calculations(validation_data, signal_row_idx, epic, pair)
            
            # Attempt signal detection at this point with detailed debugging
            signal_data_slice = validation_data.iloc[:signal_row_idx + 1].copy()
            
            # Check if we have enough data for MACD calculation
            min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 200)
            if len(signal_data_slice) < min_bars:
                self.logger.warning(f"⚠️ Insufficient data for signal detection ({len(signal_data_slice)} < {min_bars})")
                self.logger.warning(f"   Proceeding with available data for validation only")
            
            # Detect signal using strategy with enhanced debugging
            self.logger.info("\n🎯 SIGNAL DETECTION ATTEMPT:")
            self.logger.info("-" * 40)
            
            # Get debug info from MACD strategy if available
            debug_signal = None
            debug_info = None
            
            # Try to get debug information from the strategy
            if hasattr(self.strategy, 'debug_signal_detection'):
                try:
                    debug_info = self.strategy.debug_signal_detection(
                        signal_data_slice, epic, config.SPREAD_PIPS, timeframe
                    )
                    self.logger.info("🔍 DETAILED MACD DEBUG INFORMATION:")
                    self.logger.info("-" * 50)
                    
                    # Show validation steps
                    if 'validation_steps' in debug_info:
                        self.logger.info("✅ Validation Steps Passed:")
                        for step in debug_info['validation_steps']:
                            self.logger.info(f"   {step}")
                    
                    # Show rejection reasons
                    if 'rejection_reasons' in debug_info:
                        self.logger.info("❌ Rejection Reasons:")
                        for reason in debug_info['rejection_reasons']:
                            self.logger.info(f"   {reason}")
                    
                    # Show specific validation details
                    if 'comprehensive_validation' in debug_info:
                        comp_val = debug_info['comprehensive_validation']
                        self.logger.info(f"\n🎯 Comprehensive Validation:")
                        self.logger.info(f"   Valid: {comp_val.get('is_valid', 'Unknown')}")
                        self.logger.info(f"   Confidence: {comp_val.get('confidence', 0):.1%}")
                        if 'validation_details' in comp_val:
                            for key, value in comp_val['validation_details'].items():
                                self.logger.info(f"   {key}: {value}")
                    
                    # Show performance calculations
                    if 'performance_calculations' in debug_info:
                        perf = debug_info['performance_calculations']
                        self.logger.info(f"\n📊 Performance Calculations:")
                        self.logger.info(f"   Efficiency Ratio: {perf.get('efficiency_ratio', 'Unknown')}")
                        self.logger.info(f"   Market Regime: {perf.get('market_regime', 'Unknown')}")
                    
                    # Show signal detection details
                    if 'signal_data' in debug_info:
                        sig_data = debug_info['signal_data']
                        self.logger.info(f"\n🎯 Signal Detection Details:")
                        self.logger.info(f"   Signal Type: {sig_data.get('signal_type', 'Unknown')}")
                        self.logger.info(f"   Trigger Reason: {sig_data.get('trigger_reason', 'Unknown')}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Debug info retrieval failed: {e}")
                    debug_info = None
            
            # CRITICAL FIX: Convert timestamp for historical session detection
            historical_timestamp = None
            try:
                historical_timestamp = target_time.to_pydatetime() if hasattr(target_time, 'to_pydatetime') else target_time
            except:
                historical_timestamp = None

            # Try MTF-enhanced detection first if available
            mtf_available = (hasattr(self.strategy, 'detect_signal_with_mtf') and 
                           getattr(self.strategy, 'enable_mtf_analysis', False))
            
            detected_signal = None
            detection_method = "none"
            
            if mtf_available:
                try:
                    detected_signal = self.strategy.detect_signal_with_mtf(
                        signal_data_slice, epic, config.SPREAD_PIPS, timeframe, historical_timestamp
                    )
                    detection_method = "mtf_enhanced"
                    self.logger.info("✅ Used MTF-enhanced detection")
                except Exception as e:
                    self.logger.warning(f"⚠️ MTF detection failed: {e}")
            
            # Fallback to standard detection
            if not detected_signal:
                try:
                    detected_signal = self.strategy.detect_signal(
                        signal_data_slice, epic, config.SPREAD_PIPS, timeframe, historical_timestamp
                    )
                    detection_method = "standard"
                    self.logger.info("✅ Used standard detection")
                except Exception as e:
                    self.logger.error(f"❌ Standard detection failed: {e}")
            
            # NEW: Check for momentum confirmation signals even if no immediate signal
            momentum_signal = None
            if not detected_signal and hasattr(self.strategy.signal_detector, 'check_momentum_confirmation_signals'):
                try:
                    # Build crossover history by processing all bars in sequence
                    # This simulates how the system would track crossovers in real-time
                    self.logger.debug("Building crossover history for momentum confirmation check...")
                    
                    # Process each bar in sequence to build crossover tracking
                    for i in range(len(signal_data_slice)):
                        current_bar = signal_data_slice.iloc[i]
                        
                        # For momentum confirmation, we need to track crossovers as they happen
                        if hasattr(self.strategy.signal_detector, 'crossover_tracker'):
                            # Process this bar to update crossover tracking
                            if i > 0:  # Need previous bar for crossover detection
                                prev_bar = signal_data_slice.iloc[i-1]
                                
                                # Check if this is a crossover moment
                                current_hist = current_bar.get('macd_histogram', 0.0)
                                prev_hist = prev_bar.get('macd_histogram', 0.0)
                                
                                # Detect crossover direction
                                if prev_hist <= 0 and current_hist > 0:
                                    # Bullish crossover
                                    crossover_signal = self.strategy.signal_detector.detect_enhanced_macd_signal(
                                        latest=current_bar,
                                        previous=prev_bar,
                                        epic=epic,
                                        timeframe=timeframe,
                                        df_enhanced=signal_data_slice.iloc[:i+1],
                                        forex_optimizer=self.strategy.forex_optimizer
                                    )
                                    # If no immediate signal, track as weak crossover
                                    if not crossover_signal:
                                        self.strategy.signal_detector.crossover_tracker.record_crossover(
                                            epic=epic,
                                            timeframe=timeframe,
                                            timestamp=current_bar.name,
                                            signal_type='BULL',
                                            histogram_value=current_hist,
                                            reason='Below threshold - tracking for momentum'
                                        )
                                elif prev_hist >= 0 and current_hist < 0:
                                    # Bearish crossover
                                    crossover_signal = self.strategy.signal_detector.detect_enhanced_macd_signal(
                                        latest=current_bar,
                                        previous=prev_bar,
                                        epic=epic,
                                        timeframe=timeframe,
                                        df_enhanced=signal_data_slice.iloc[:i+1],
                                        forex_optimizer=self.strategy.forex_optimizer
                                    )
                                    # If no immediate signal, track as weak crossover
                                    if not crossover_signal:
                                        self.strategy.signal_detector.crossover_tracker.record_crossover(
                                            epic=epic,
                                            timeframe=timeframe,
                                            timestamp=current_bar.name,
                                            signal_type='BEAR',
                                            histogram_value=current_hist,
                                            reason='Below threshold - tracking for momentum'
                                        )
                    
                    # Now check for momentum confirmation at the target timestamp
                    latest = signal_data_slice.iloc[-1]
                    
                    momentum_signal = self.strategy.signal_detector.check_momentum_confirmation_signals(
                        epic=epic,
                        timeframe=timeframe,
                        df_enhanced=signal_data_slice,
                        latest=latest,
                        forex_optimizer=self.strategy.forex_optimizer
                    )
                    
                    if momentum_signal:
                        detected_signal = momentum_signal
                        detection_method = "momentum_confirmation"
                        self.logger.info("🎯 Used momentum confirmation detection")
                        self.logger.info(f"   Initial crossover: {momentum_signal.get('momentum_confirmation', {}).get('initial_histogram', 'N/A')}")
                        self.logger.info(f"   Bars since crossover: {momentum_signal.get('momentum_confirmation', {}).get('bars_since_crossover', 'N/A')}")
                except Exception as e:
                    self.logger.debug(f"Momentum confirmation check failed: {e}")
            
            # Enhanced analysis of why no signal was detected
            if not detected_signal and debug_info:
                self.logger.info("\n🔍 FAILURE ANALYSIS:")
                self.logger.info("-" * 30)
                
                # Analyze the debug info to understand the failure
                if debug_info.get('signal_result') == 'SUCCESS':
                    self.logger.info("⚠️ Debug shows SUCCESS but no signal returned - possible bug")
                elif 'rejection_reasons' in debug_info and debug_info['rejection_reasons']:
                    self.logger.info("📊 Primary rejection reasons:")
                    for reason in debug_info['rejection_reasons'][:3]:  # Show top 3
                        self.logger.info(f"   • {reason}")
                
                # Check specific validation failures
                if 'comprehensive_validation' in debug_info:
                    comp_val = debug_info['comprehensive_validation']
                    is_valid = comp_val.get('is_valid', False)
                    confidence = comp_val.get('confidence', 0)
                    
                    if not is_valid:
                        self.logger.info(f"❌ Comprehensive validation failed")
                    elif confidence < 0.7:  # Assuming 70% threshold
                        self.logger.info(f"❌ Confidence too low: {confidence:.1%} < 70%")
                        
                        # Show confidence breakdown if available
                        if 'validation_details' in comp_val:
                            details = comp_val['validation_details']
                            self.logger.info(f"   Confidence breakdown:")
                            for key, value in details.items():
                                if 'confidence' in key.lower() or 'score' in key.lower():
                                    self.logger.info(f"     {key}: {value}")
                
                # Check for specific common issues
                if any('efficiency' in reason.lower() for reason in debug_info.get('rejection_reasons', [])):
                    self.logger.info("💡 Hint: Market efficiency may be too low for trading")
                
                if any('session' in reason.lower() for reason in debug_info.get('rejection_reasons', [])):
                    self.logger.info("💡 Hint: Trading session restrictions may be active")
                
                if any('deduplication' in reason.lower() for reason in debug_info.get('rejection_reasons', [])):
                    self.logger.info("💡 Hint: Recent signal deduplication may be blocking")
            
            elif not detected_signal:
                self.logger.info("\n🔍 BASIC FAILURE ANALYSIS:")
                self.logger.info("-" * 30)
                self.logger.info("❌ No debug information available")
                self.logger.info("💡 Possible reasons:")
                self.logger.info("   • Confidence calculation failed")
                self.logger.info("   • Session-based filtering active")
                self.logger.info("   • Signal deduplication blocking")
                self.logger.info("   • Market regime validation failed")
                self.logger.info("   • Efficiency ratio too low")
            
            # Show decision tree if requested
            if show_decision_tree:
                self._show_decision_tree(validation_data, signal_row_idx, epic, timeframe, detected_signal)
            
            # Display results with timestamp for correlation
            actual_timestamp = validation_data.iloc[signal_row_idx]['start_time'] if 'start_time' in validation_data.columns else validation_data.index[signal_row_idx]
            self.logger.info("\n📋 VALIDATION RESULTS:")
            self.logger.info("=" * 40)
            self.logger.info(f"🕐 Analysis Timestamp: {actual_timestamp}")
            self.logger.info(f"📊 Data Point Index: {signal_row_idx}/{len(validation_data)-1}")
            
            if detected_signal:
                self.logger.info("✅ SIGNAL DETECTED")
                self._display_detected_signal_details(detected_signal, detection_method)
                
                # Smart Money analysis if enabled
                if self.smart_money_enabled:
                    enhanced_signal = self._enhance_signal_with_smart_money(
                        detected_signal, signal_data_slice, epic, timeframe, pair
                    )
                    if enhanced_signal.get('smart_money_analysis'):
                        self._display_smart_money_analysis(enhanced_signal['smart_money_analysis'])
                
            else:
                self.logger.info("❌ NO SIGNAL DETECTED")
                self.logger.info("   The strategy did not generate a signal at this timestamp")
                self.logger.info("   This could be due to:")
                self.logger.info("   - Insufficient data for MACD calculation")
                self.logger.info("   - MACD conditions not met")
                self.logger.info("   - EMA200 filter conditions not met")
                self.logger.info("   - Confidence threshold not reached")
                self.logger.info("   - Recent signal deduplication")
            
            # Show strategy configuration
            self._show_strategy_configuration()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Signal validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_live_scanner_simulation(
            self,
            epic: str = None,
            days: int = 1,
            timeframe: str = None,
            show_signals: bool = True,
            enable_smart_money: bool = False
        ) -> bool:
            """
            Simulate the exact live scanner behavior using the REAL signal detection path
            """

            timeframe = timeframe or getattr(config, 'DEFAULT_TIMEFRAME', '15m')
            epic_list = [epic] if epic else config.EPIC_LIST

            self.logger.info("🔄 LIVE SCANNER SIMULATION MODE")
            self.logger.info("=" * 50)
            self.logger.info(f"📊 Epic(s): {epic_list}")
            self.logger.info(f"⏰ Timeframe: {timeframe}")
            self.logger.info(f"📅 Days: {days}")
            self.logger.info(f"🧠 Smart Money analysis: {enable_smart_money}")

            # Initialize Smart Money analysis if enabled
            if enable_smart_money:
                self.initialize_smart_money_analysis(enable_smart_money)

            try:
                # CRITICAL FIX: Use the EXACT same signal detection path as live scanner
                from core.signal_detector import SignalDetector
                signal_detector = SignalDetector(self.db_manager, 'UTC')

                self.logger.info("✅ Using real SignalDetector (matches live scanner exactly)")
                self.logger.info(f"   Smart Money: {'✅ Enabled' if self.smart_money_enabled else '❌ Disabled'}")

                all_signals = []

                for current_epic in epic_list:
                    self.logger.info(f"\n📈 Simulating live scanner for {current_epic}")

                    # Get pair info (same as live scanner)
                    pair_info = config.PAIR_INFO.get(current_epic, {'pair': 'EURUSD', 'pip_multiplier': 10000})
                    pair_name = pair_info['pair']

                    self.logger.info(f"   📊 Pair: {pair_name}")
                    self.logger.info(f"   📊 Timeframe: {timeframe}")
                    self.logger.info(f"   📊 BID adjustment: {config.USE_BID_ADJUSTMENT}")
                    self.logger.info(f"   📊 Spread pips: {config.SPREAD_PIPS}")

                    # CRITICAL FIX: Use EXACT live scanner detection method
                    if config.USE_BID_ADJUSTMENT:
                        signal = signal_detector.detect_signals_bid_adjusted(
                            current_epic, pair_name, config.SPREAD_PIPS, timeframe
                        )
                    else:
                        signal = signal_detector.detect_signals_mid_prices(
                            current_epic, pair_name, timeframe
                        )

                    if signal:
                        self.logger.info(f"   🎯 Live scanner simulation found signal!")
                        self.logger.info(f"      Type: {signal.get('signal_type')}")
                        self.logger.info(f"      Strategy: {signal.get('strategy')}")
                        self.logger.info(f"      Confidence: {signal.get('confidence', signal.get('confidence_score', 0)):.1%}")
                        self.logger.info(f"      Price: {signal.get('price', 0):.5f}")

                        signal['simulation_mode'] = 'live_scanner'
                        # Use UTC for consistency
                        signal['detected_at'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

                        all_signals.append(signal)
                    else:
                        self.logger.info(f"   ❌ Live scanner simulation found no signals")

                # Display results
                if all_signals:
                    self.logger.info(f"\n✅ LIVE SCANNER SIMULATION RESULTS:")
                    self.logger.info(f"   📊 Total signals: {len(all_signals)}")

                    if show_signals:
                        self._display_simulation_signals(all_signals)

                    return True
                else:
                    self.logger.warning("❌ Live scanner simulation found no signals")
                    return False

            except Exception as e:
                self.logger.error(f"❌ Live scanner simulation failed: {e}")
                import traceback
                traceback.print_exc()
                return False

    def _show_raw_data_context(self, df: pd.DataFrame, signal_idx: int, target_time: pd.Timestamp):
        """Show raw OHLC data around the signal point"""
        self.logger.info("\n📊 RAW DATA CONTEXT:")
        self.logger.info("-" * 80)
        self.logger.info("IDX  TIMESTAMP            OPEN     HIGH     LOW      CLOSE    VOLUME")
        self.logger.info("-" * 80)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            # Try to get timestamp from multiple sources
            timestamp_str = 'Unknown'
            for col in ['datetime_utc', 'start_time', 'timestamp', 'datetime']:
                if col in row and row[col] is not None:
                    try:
                        if isinstance(row[col], str):
                            timestamp_str = row[col][:19]  # Truncate to YYYY-MM-DD HH:MM:SS
                        elif hasattr(row[col], 'strftime'):
                            timestamp_str = row[col].strftime('%Y-%m-%d %H:%M:%S UTC')
                        break
                    except:
                        continue
            
            # If still no timestamp, try using row index
            if timestamp_str == 'Unknown' and hasattr(idx, 'strftime'):
                try:
                    timestamp_str = idx.strftime('%Y-%m-%d %H:%M:%S UTC')
                except:
                    pass
            
            open_val = row.get('open', 0)
            high_val = row.get('high', 0)
            low_val = row.get('low', 0)
            close_val = row.get('close', 0)
            volume_val = row.get('volume', row.get('ltv', 0))  # Try 'ltv' if 'volume' not available
            
            marker = " >>> " if i == signal_idx else "     "
            self.logger.info(f"{i:2d}{marker}{timestamp_str} {open_val:8.5f} {high_val:8.5f} "
                           f"{low_val:8.5f} {close_val:8.5f} {volume_val:8.0f}")
    
    def _show_detailed_calculations(self, df: pd.DataFrame, signal_idx: int, epic: str, pair: str):
        """Show detailed MACD and EMA calculations"""
        self.logger.info("\n🧮 DETAILED CALCULATIONS:")
        self.logger.info("-" * 50)
        
        try:
            # Get the signal row
            signal_row = df.iloc[signal_idx]
            prev_row = df.iloc[signal_idx - 1] if signal_idx > 0 else signal_row
            
            # MACD values
            macd_line = signal_row.get('macd_line', 0)
            macd_signal = signal_row.get('macd_signal', 0)
            macd_histogram = signal_row.get('macd_histogram', 0)
            macd_histogram_prev = prev_row.get('macd_histogram', 0)
            
            # EMA values
            ema_200 = signal_row.get('ema_200', 0)
            current_price = signal_row.get('close', 0)
            
            # Color determination
            macd_color = 'green' if macd_histogram > 0 else 'red'
            macd_color_prev = 'green' if macd_histogram_prev > 0 else 'red'
            
            self.logger.info(f"📈 MACD Calculations:")
            self.logger.info(f"   MACD Line: {macd_line:.6f}")
            self.logger.info(f"   MACD Signal: {macd_signal:.6f}")
            self.logger.info(f"   MACD Histogram: {macd_histogram:.6f} ({macd_color})")
            self.logger.info(f"   Previous Histogram: {macd_histogram_prev:.6f} ({macd_color_prev})")
            self.logger.info(f"   Histogram Change: {(macd_histogram - macd_histogram_prev):.6f}")
            self.logger.info(f"   Color Transition: {macd_color_prev} → {macd_color}")
            
            self.logger.info(f"\n📊 EMA Analysis:")
            self.logger.info(f"   Current Price: {current_price:.5f}")
            self.logger.info(f"   EMA200: {ema_200:.5f}")
            self.logger.info(f"   Price vs EMA200: {'ABOVE' if current_price > ema_200 else 'BELOW'}")
            self.logger.info(f"   Distance (pips): {abs(current_price - ema_200) * 10000:.1f}")
            
            # Thresholds
            try:
                if hasattr(self.strategy, 'forex_optimizer') and self.strategy.forex_optimizer:
                    # Try different method names for threshold retrieval
                    macd_threshold = None
                    min_confidence = None
                    
                    # Use correct method to get session-aware threshold
                    try:
                        # CRITICAL FIX: Get historical market session from signal timestamp
                        signal_timestamp = signal_row.name if hasattr(signal_row, 'name') and signal_row.name else signal_row.index[0] if hasattr(signal_row, 'index') else None
                        historical_session = None
                        
                        if signal_timestamp:
                            try:
                                # Convert to datetime if needed
                                if isinstance(signal_timestamp, str):
                                    from datetime import datetime
                                    signal_timestamp = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
                                elif hasattr(signal_timestamp, 'to_pydatetime'):
                                    signal_timestamp = signal_timestamp.to_pydatetime()
                                
                                historical_session = self.strategy.forex_optimizer.get_market_session_from_timestamp(signal_timestamp)
                                self.logger.debug(f"Historical session for {signal_timestamp}: {historical_session}")
                            except Exception as session_err:
                                self.logger.debug(f"Failed to get historical session: {session_err}")
                        
                        macd_threshold = self.strategy.forex_optimizer.get_macd_threshold_for_epic(epic, historical_session)
                        self.logger.debug(f"Retrieved MACD threshold: {macd_threshold} (session: {historical_session or 'current'})")
                    except Exception as e:
                        self.logger.debug(f"Failed to get MACD threshold: {e}")
                        # Use corrected fallback threshold based on pair type
                        if 'JPY' in epic.upper():
                            macd_threshold = 0.008  # Updated JPY threshold
                        else:
                            macd_threshold = 0.00008  # Updated non-JPY threshold
                        self.logger.debug(f"Using corrected fallback MACD threshold: {macd_threshold}")
                    
                    # Get strength thresholds for histogram magnitude comparison
                    try:
                        strength_thresholds = self.strategy.forex_optimizer.macd_strength_thresholds
                        min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.7)
                    except Exception as e:
                        self.logger.debug(f"Failed to get strength thresholds: {e}")
                        # Use corrected fallback strength thresholds
                        if 'JPY' in epic.upper():
                            strength_thresholds = {'moderate': 0.005, 'strong': 0.010, 'very_strong': 0.020}
                        else:
                            strength_thresholds = {'moderate': 0.00005, 'strong': 0.0001, 'very_strong': 0.0002}
                        min_confidence = 0.7
                    
                    # Calculate histogram magnitude and change for proper comparison
                    histogram_magnitude = abs(macd_histogram)
                    histogram_change = abs(macd_histogram - macd_histogram_prev)
                    
                    # Determine strength category based on histogram magnitude (like main strategy)
                    if histogram_magnitude >= strength_thresholds.get('very_strong', 0.0002):
                        strength_category = 'very_strong'
                    elif histogram_magnitude >= strength_thresholds.get('strong', 0.0001):
                        strength_category = 'strong'
                    elif histogram_magnitude >= strength_thresholds.get('moderate', 0.00005):
                        strength_category = 'moderate'
                    else:
                        strength_category = 'weak'
                    
                    self.logger.info(f"\n🎯 Thresholds & Analysis:")
                    self.logger.info(f"   MACD Threshold (change): {macd_threshold:.6f}")
                    self.logger.info(f"   Min Confidence: {min_confidence:.1%}")
                    self.logger.info(f"   Histogram Magnitude: {histogram_magnitude:.6f}")
                    self.logger.info(f"   Histogram Change: {histogram_change:.6f}")
                    self.logger.info(f"   Strength Category: {strength_category}")
                    self.logger.info(f"   Change vs Threshold: {'PASS' if histogram_change >= macd_threshold else 'FAIL'}")
                    self.logger.info(f"   Magnitude vs Moderate: {'PASS' if histogram_magnitude >= strength_thresholds.get('moderate', 0.00005) else 'FAIL'}")
                else:
                    self.logger.info(f"\n🎯 Thresholds:")
                    self.logger.info(f"   No forex optimizer available - using defaults")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not retrieve thresholds: {e}")
            
            # Volume analysis
            volume = signal_row.get('volume', signal_row.get('ltv', 0))  # Try 'ltv' if 'volume' not available
            volume_ratio = signal_row.get('volume_ratio', 1.0)
            
            self.logger.info(f"\n📊 Volume Analysis:")
            self.logger.info(f"   Current Volume: {volume:,.0f}")
            self.logger.info(f"   Volume Ratio: {volume_ratio:.2f}")
            self.logger.info(f"   Volume Assessment: {'HIGH' if volume_ratio > 1.2 else 'NORMAL' if volume_ratio > 0.8 else 'LOW'}")
            
            # Show available data columns for debugging
            self.logger.info(f"\n🔍 Available Data Columns:")
            available_cols = list(signal_row.index) if hasattr(signal_row, 'index') else []
            self.logger.info(f"   Total columns: {len(available_cols)}")
            
            # Show key columns
            key_cols = [col for col in available_cols if any(key in col.lower() for key in 
                       ['time', 'date', 'macd', 'ema', 'volume', 'open', 'high', 'low', 'close'])]
            self.logger.info(f"   Key columns: {key_cols[:15]}...")  # Show first 15 key columns
            
        except Exception as e:
            self.logger.error(f"❌ Error showing calculations: {e}")
    
    def _show_decision_tree(self, df: pd.DataFrame, signal_idx: int, epic: str, timeframe: str, detected_signal: Dict = None):
        """Show the decision-making process step by step"""
        self.logger.info("\n🌳 DECISION TREE:")
        self.logger.info("-" * 40)
        
        try:
            # Get the current and previous rows
            signal_row = df.iloc[signal_idx]
            prev_row = df.iloc[signal_idx - 1] if signal_idx > 0 else signal_row
            
            # Step 1: Data availability check
            self.logger.info("1️⃣ Data Availability Check:")
            required_fields = ['macd_histogram', 'ema_200', 'close']
            all_present = True
            
            for field in required_fields:
                present = field in signal_row and signal_row[field] is not None
                self.logger.info(f"   {field}: {'✅' if present else '❌'}")
                if not present:
                    all_present = False
            
            if not all_present:
                self.logger.info("   ❌ DECISION: Insufficient data - NO SIGNAL")
                return
            
            # Step 2: MACD crossover check
            self.logger.info("\n2️⃣ MACD Crossover Analysis:")
            macd_histogram = signal_row.get('macd_histogram', 0)
            macd_histogram_prev = prev_row.get('macd_histogram', 0)
            
            color_current = 'green' if macd_histogram > 0 else 'red'
            color_prev = 'green' if macd_histogram_prev > 0 else 'red'
            
            has_crossover = color_current != color_prev
            crossover_type = None
            
            if has_crossover:
                if color_current == 'green' and color_prev == 'red':
                    crossover_type = "bullish (red → green)"
                elif color_current == 'red' and color_prev == 'green':
                    crossover_type = "bearish (green → red)"
            
            self.logger.info(f"   Previous: {macd_histogram_prev:.6f} ({color_prev})")
            self.logger.info(f"   Current:  {macd_histogram:.6f} ({color_current})")
            self.logger.info(f"   Crossover: {'✅' if has_crossover else '❌'}")
            
            if has_crossover:
                self.logger.info(f"   Type: {crossover_type}")
            else:
                self.logger.info("   ❌ DECISION: No MACD crossover - NO SIGNAL")
                return
            
            # Step 3: Threshold check
            self.logger.info("\n3️⃣ Threshold Validation:")
            histogram_change = abs(macd_histogram - macd_histogram_prev)
            
            try:
                if hasattr(self.strategy, 'forex_optimizer') and self.strategy.forex_optimizer:
                    # Use correct method to get session-aware threshold
                    try:
                        macd_threshold = self.strategy.forex_optimizer.get_macd_threshold_for_epic(epic)
                    except Exception as e:
                        self.logger.debug(f"Failed to get MACD threshold: {e}")
                        # Use corrected fallback threshold
                        if 'JPY' in epic.upper():
                            macd_threshold = 0.008  # Updated JPY threshold
                        else:
                            macd_threshold = 0.00008  # Updated non-JPY threshold
                    
                    # Get strength thresholds for proper validation
                    try:
                        strength_thresholds = self.strategy.forex_optimizer.macd_strength_thresholds
                    except Exception as e:
                        # Use corrected fallback strength thresholds
                        if 'JPY' in epic.upper():
                            strength_thresholds = {'moderate': 0.005, 'strong': 0.010, 'very_strong': 0.020}
                        else:
                            strength_thresholds = {'moderate': 0.00005, 'strong': 0.0001, 'very_strong': 0.0002}
                    
                    # Calculate histogram magnitude for strength validation (like main strategy)
                    histogram_magnitude = abs(macd_histogram)
                    threshold_passed = histogram_change >= macd_threshold
                    magnitude_passed = histogram_magnitude >= strength_thresholds.get('moderate', 0.00005)
                    
                    self.logger.info(f"   Histogram Change: {histogram_change:.6f}")
                    self.logger.info(f"   Histogram Magnitude: {histogram_magnitude:.6f}")
                    self.logger.info(f"   Required Threshold (change): {macd_threshold:.6f}")
                    self.logger.info(f"   Required Magnitude (moderate): {strength_thresholds.get('moderate', 0.00005):.6f}")
                    self.logger.info(f"   Change Check: {'✅' if threshold_passed else '❌'}")
                    self.logger.info(f"   Magnitude Check: {'✅' if magnitude_passed else '❌'}")
                    self.logger.info(f"   Overall Signal Quality: {'✅ STRONG' if threshold_passed and magnitude_passed else '⚠️ WEAK' if threshold_passed or magnitude_passed else '❌ REJECTED'}")
                    
                    if not threshold_passed and not magnitude_passed:
                        self.logger.info("   ❌ DECISION: Neither threshold nor magnitude requirements met - NO SIGNAL")
                        return
                    elif not threshold_passed:
                        self.logger.info("   ⚠️ DECISION: Change threshold not met but magnitude sufficient - WEAK SIGNAL")
                    elif not magnitude_passed:
                        self.logger.info("   ⚠️ DECISION: Magnitude too low but change sufficient - WEAK SIGNAL")
                else:
                    self.logger.info("   ⚠️ No forex optimizer - using corrected fallback validation")
                    # Use corrected fallback thresholds
                    if 'JPY' in epic.upper():
                        macd_threshold = 0.008  # Updated JPY threshold
                        moderate_magnitude = 0.005  # JPY moderate strength
                    else:
                        macd_threshold = 0.00008  # Updated non-JPY threshold
                        moderate_magnitude = 0.00005  # Non-JPY moderate strength
                    
                    histogram_magnitude = abs(macd_histogram)
                    threshold_passed = histogram_change >= macd_threshold
                    magnitude_passed = histogram_magnitude >= moderate_magnitude
                    
                    self.logger.info(f"   Histogram Change: {histogram_change:.6f}")
                    self.logger.info(f"   Histogram Magnitude: {histogram_magnitude:.6f}")
                    self.logger.info(f"   Fallback Threshold (change): {macd_threshold:.6f}")
                    self.logger.info(f"   Fallback Magnitude: {moderate_magnitude:.6f}")
                    self.logger.info(f"   Change Check: {'✅' if threshold_passed else '❌'}")
                    self.logger.info(f"   Magnitude Check: {'✅' if magnitude_passed else '❌'}")
                    
                    if not threshold_passed and not magnitude_passed:
                        self.logger.info("   ❌ DECISION: Neither threshold nor magnitude requirements met - NO SIGNAL")
                        return
                    
            except Exception as e:
                self.logger.warning(f"   ⚠️ Threshold check failed: {e}")
                self.logger.info("   Continuing with basic validation...")
            
            # Step 4: EMA200 filter
            self.logger.info("\n4️⃣ EMA200 Trend Filter:")
            current_price = signal_row.get('close', 0)
            ema_200 = signal_row.get('ema_200', 0)
            
            price_above_ema = current_price > ema_200
            
            self.logger.info(f"   Current Price: {current_price:.5f}")
            self.logger.info(f"   EMA200: {ema_200:.5f}")
            self.logger.info(f"   Price Position: {'ABOVE' if price_above_ema else 'BELOW'} EMA200")
            
            # Check if trend aligns with signal
            trend_aligned = False
            if crossover_type and "bullish" in crossover_type and price_above_ema:
                trend_aligned = True
                self.logger.info("   ✅ Bullish crossover + price above EMA200 = ALIGNED")
            elif crossover_type and "bearish" in crossover_type and not price_above_ema:
                trend_aligned = True
                self.logger.info("   ✅ Bearish crossover + price below EMA200 = ALIGNED")
            else:
                self.logger.info("   ❌ Trend not aligned with crossover")
            
            if not trend_aligned:
                self.logger.info("   ❌ DECISION: EMA200 filter failed - NO SIGNAL")
                return
            
            # Step 5: Confidence calculation
            self.logger.info("\n5️⃣ Confidence Assessment:")
            if detected_signal:
                confidence = detected_signal.get('confidence', detected_signal.get('confidence_score', 0))
                self.logger.info(f"   Calculated Confidence: {confidence:.1%}")
                
                min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.7)
                confidence_passed = confidence >= min_confidence
                
                self.logger.info(f"   Required Minimum: {min_confidence:.1%}")
                self.logger.info(f"   Confidence Check: {'✅' if confidence_passed else '❌'}")
                
                if confidence_passed:
                    self.logger.info("   ✅ FINAL DECISION: ALL CHECKS PASSED - SIGNAL GENERATED")
                else:
                    self.logger.info("   ❌ FINAL DECISION: Confidence too low - NO SIGNAL")
            else:
                self.logger.info("   ⚠️ No detected signal to analyze confidence")
            
        except Exception as e:
            self.logger.error(f"❌ Error in decision tree analysis: {e}")
    
    def _display_detected_signal_details(self, signal: Dict, detection_method: str):
        """Display detailed information about the detected signal"""
        self.logger.info(f"📊 Signal Detection Method: {detection_method}")
        
        # Basic signal info
        signal_type = signal.get('signal_type', 'Unknown')
        confidence = signal.get('confidence', signal.get('confidence_score', 0))
        price = signal.get('price', 0)
        strategy = signal.get('strategy', 'Unknown')
        
        self.logger.info(f"   Signal Type: {signal_type}")
        self.logger.info(f"   Strategy: {strategy}")
        self.logger.info(f"   Confidence: {confidence:.1%}")
        self.logger.info(f"   Price: {price:.5f}")
        
        # MACD specific data
        if 'macd_values' in signal:
            macd_data = signal['macd_values']
            self.logger.info(f"   MACD Line: {macd_data.get('macd_line', 0):.6f}")
            self.logger.info(f"   MACD Signal: {macd_data.get('macd_signal', 0):.6f}")
            self.logger.info(f"   MACD Histogram: {macd_data.get('macd_histogram', 0):.6f}")
            self.logger.info(f"   EMA200: {macd_data.get('ema_200', 0):.5f}")
        
        # MTF analysis if available
        if 'mtf_analysis' in signal:
            mtf_data = signal['mtf_analysis']
            if mtf_data.get('enabled', False):
                aligned = mtf_data.get('aligned_timeframes', 0)
                total = mtf_data.get('total_timeframes', 0)
                alignment_score = mtf_data.get('momentum_score', 0)  # MACD uses momentum_score, not alignment_score
                
                self.logger.info(f"   MTF Analysis: {aligned}/{total} timeframes aligned")
                self.logger.info(f"   MTF Score: {alignment_score:.1%}")
                self.logger.info(f"   MTF Valid: {'✅' if mtf_data.get('mtf_valid', False) else '❌'}")
        
        # Strategy-specific metadata
        if 'strategy_metadata' in signal:
            metadata = signal['strategy_metadata']
            self.logger.info(f"   Strategy Version: {metadata.get('strategy_version', 'Unknown')}")
            self.logger.info(f"   Signal Basis: {metadata.get('signal_basis', 'Unknown')}")
    
    def _display_smart_money_analysis(self, smc_data: Dict):
        """Display Smart Money analysis if available"""
        self.logger.info("\n🧠 SMART MONEY ANALYSIS:")
        self.logger.info("-" * 40)
        
        if not smc_data.get('enabled', False):
            self.logger.info("❌ Smart Money analysis disabled")
            return
        
        structure_score = smc_data.get('structure_score', 0)
        order_flow_score = smc_data.get('order_flow_score', 0)
        confidence_boost = smc_data.get('confidence_boost', 0)
        
        self.logger.info(f"   Structure Score: {structure_score:.1%}")
        self.logger.info(f"   Order Flow Score: {order_flow_score:.1%}")
        self.logger.info(f"   Confidence Boost: {confidence_boost:.1%}")
        
        # Additional SMC details if available
        if 'market_structure' in smc_data:
            ms_data = smc_data['market_structure']
            self.logger.info(f"   Market Structure: {ms_data.get('trend', 'Unknown')}")
            self.logger.info(f"   BOS/ChoCh: {ms_data.get('structure_break', 'None')}")
        
        if 'order_flow' in smc_data:
            of_data = smc_data['order_flow']
            self.logger.info(f"   Order Blocks: {of_data.get('order_blocks_count', 0)}")
            self.logger.info(f"   Fair Value Gaps: {of_data.get('fvg_count', 0)}")
    
    def _show_strategy_configuration(self):
        """Show current strategy configuration"""
        self.logger.info("\n⚙️ STRATEGY CONFIGURATION:")
        self.logger.info("-" * 40)
        
        self.logger.info(f"   MACD Fast Period: 12")
        self.logger.info(f"   MACD Slow Period: 26")
        self.logger.info(f"   MACD Signal Period: 9")
        self.logger.info(f"   EMA200 Filter: Enabled")
        self.logger.info(f"   Min Confidence: {getattr(config, 'MIN_CONFIDENCE', 0.7):.1%}")
        self.logger.info(f"   Min Bars Required: {getattr(config, 'MIN_BARS_FOR_SIGNAL', 200)}")
        
        # Show integration status
        if hasattr(self.strategy, 'get_integration_status'):
            try:
                integration_status = self.strategy.get_integration_status()
                self.logger.info(f"   Forex Optimizer: {'✅' if integration_status.get('has_forex_optimizer') else '❌'}")
                self.logger.info(f"   Confidence Optimizer: {'✅' if integration_status.get('has_confidence_optimizer') else '❌'}")
            except:
                pass
        
        # MTF status
        mtf_enabled = getattr(self.strategy, 'enable_mtf_analysis', False)
        self.logger.info(f"   MTF Analysis: {'✅ Enabled' if mtf_enabled else '❌ Disabled'}")
        
        # Smart Money status
        self.logger.info(f"   Smart Money: {'✅ Enabled' if self.smart_money_enabled else '❌ Disabled'}")
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = None,
        show_signals: bool = False,
        min_confidence: float = None,
        optimization_config: Dict = None,
        enable_forex_integration: bool = True,
        enable_smart_money: bool = False,
        use_optimal_parameters: bool = True
    ) -> bool:
        """Run enhanced MACD strategy backtest with optional Smart Money analysis"""
        
        # Use default timeframe if not specified (ALIGNMENT FIX)
        if timeframe is None:
            timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '15m')
            self.logger.debug(f"🔧 Using DEFAULT_TIMEFRAME from config: {timeframe}")
        
        # Setup epic list
        epic_list = [epic] if epic else config.EPIC_LIST
        
        self.logger.info("🧪 ENHANCED MACD STRATEGY BACKTEST WITH DATABASE OPTIMIZATION")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Epic(s): {epic_list}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🎯 Show signals: {show_signals}")
        self.logger.info(f"🔗 Forex integration: {enable_forex_integration}")
        self.logger.info(f"🧠 Smart Money analysis: {enable_smart_money}")
        self.logger.info(f"🎯 Database optimization: {'✅ ENABLED' if use_optimal_parameters else '❌ DISABLED'}")
        
        # Initialize Smart Money analysis
        if enable_smart_money:
            smart_money_init_success = self.initialize_smart_money_analysis(True)
            if not smart_money_init_success:
                self.logger.warning("⚠️ Smart Money initialization failed - continuing without SMC analysis")
        
        # Override minimum confidence if specified
        original_min_conf = None
        if min_confidence:
            original_min_conf = getattr(config, 'MIN_CONFIDENCE', 0.7)
            config.MIN_CONFIDENCE = min_confidence
            self.logger.info(f"🎚️ Min confidence: {min_confidence:.1%} (was {original_min_conf:.1%})")
        
        try:
            # Initialize strategy with database optimization and timeframe awareness
            # Use first epic for strategy initialization (all use same parameters anyway)
            first_epic = epic_list[0] if epic_list else None
            self.initialize_macd_strategy(
                optimization_config=optimization_config, 
                epic=first_epic, 
                timeframe=timeframe,
                use_optimal_parameters=use_optimal_parameters
            )
            
            # Configure forex integration
            if enable_forex_integration:
                for current_epic in epic_list:
                    pair = self._extract_pair_from_epic(current_epic)
                    self.strategy.enable_forex_integration(pair)
                self.logger.info("✅ Enabled forex integration for all pairs")
            else:
                self.strategy.disable_forex_integration()
                self.logger.info("❌ Disabled forex integration")
            
            all_signals = []
            epic_results = {}
            
            for current_epic in epic_list:
                self.logger.info(f"\n📈 Processing {current_epic}")
                
                # Get enhanced data - Need to extract pair from epic
                pair = self._extract_pair_from_epic(current_epic)
                
                # Use the same lookback as the live scanner
                # For 15m: 168 hours (1 week = 672 bars)
                # This matches what _get_optimal_lookback_hours returns for live scanning
                optimal_lookback = {
                    '5m': 48,    # 2 days for 5m (576 bars)
                    '15m': 168,  # 1 week for 15m (672 bars)
                    '1h': 720,   # 1 month for 1h (720 bars)
                }.get(timeframe, 168)
                
                # Calculate total data needed
                # For 15m timeframe, we fetch 5m data and resample, so we need 3x more 5m bars
                if timeframe == '15m':
                    # We need 5m data for resampling to 15m
                    source_timeframe_minutes = 5
                    resampling_factor = 3  # 15m = 3 × 5m
                else:
                    source_timeframe_minutes = {'5m': 5, '15m': 15, '1h': 60}.get(timeframe, 15)
                    resampling_factor = 1
                
                source_bars_per_hour = 60 / source_timeframe_minutes
                
                # Calculate bars needed in the source timeframe (5m for 15m backtests)
                backtest_source_bars = int(days * 24 * source_bars_per_hour)
                optimal_lookback_source_bars = optimal_lookback * resampling_factor
                
                # Total source bars needed
                total_source_bars_needed = optimal_lookback_source_bars + backtest_source_bars
                
                self.logger.info(f"📊 Data calculation for {timeframe} backtest:")
                self.logger.info(f"   Source timeframe: {source_timeframe_minutes}m")
                self.logger.info(f"   Resampling factor: {resampling_factor}x")  
                self.logger.info(f"   Lookback needed: {optimal_lookback} {timeframe} bars = {optimal_lookback_source_bars} {source_timeframe_minutes}m bars")
                self.logger.info(f"   Backtest period: {days} days = {backtest_source_bars} {source_timeframe_minutes}m bars")
                self.logger.info(f"   Total source bars needed: {total_source_bars_needed}")
                
                # Always increase batch size for proper backtesting
                original_batch_size = self.data_fetcher.batch_size
                needed_batch_size = max(total_source_bars_needed + 500, 5000)  # Minimum 5000 for backtests
                self.data_fetcher.batch_size = min(needed_batch_size, 15000)  # Cap at 15k for safety
                
                self.logger.info(f"📊 Backtest batch size: {self.data_fetcher.batch_size} (was {original_batch_size})")
                
                # Calculate hours to request
                total_lookback_hours = int(total_source_bars_needed / source_bars_per_hour)
                
                df = self.data_fetcher.get_enhanced_data(
                    epic=current_epic,
                    pair=pair,
                    timeframe=timeframe,
                    lookback_hours=total_lookback_hours
                )
                
                # Restore original batch size
                self.data_fetcher.batch_size = original_batch_size
                
                if df is None or df.empty:
                    self.logger.warning(f"❌ No data available for {current_epic}")
                    epic_results[current_epic] = {'signals': 0, 'error': 'No data'}
                    continue
                
                self.logger.info(f"   📊 Data points: {len(df)}")
                
                # Enhanced date range logging with proper timestamp extraction
                try:
                    first_timestamp = self._get_proper_timestamp(df.iloc[0], 0)
                    last_timestamp = self._get_proper_timestamp(df.iloc[-1], len(df)-1)
                    self.logger.info(f"   📅 Date range: {first_timestamp} to {last_timestamp}")
                except Exception as e:
                    self.logger.info(f"   📅 Date range: Could not determine ({e})")
                
                # Run MACD backtest with Smart Money enhancement
                signals = self._run_enhanced_macd_backtest(df, current_epic, timeframe, pair)
                
                all_signals.extend(signals)
                epic_results[current_epic] = {'signals': len(signals)}
                
                self.logger.info(f"   🎯 MACD signals found: {len(signals)}")
                if self.smart_money_enabled:
                    enhanced_count = sum(1 for s in signals if s.get('smart_money_analysis'))
                    self.logger.info(f"   🧠 Smart Money enhanced: {enhanced_count}")
            
            # Display epic-by-epic results
            self._display_epic_results(epic_results)
            
            # Overall analysis
            if all_signals:
                self.logger.info(f"\n✅ TOTAL ENHANCED MACD SIGNALS: {len(all_signals)}")
                
                # Show individual signals if requested
                if show_signals:
                    self._display_enhanced_signals(all_signals)
                else:
                    # Show a brief summary even without --show-signals
                    self.logger.info("\n📋 SIGNAL SUMMARY:")
                    self.logger.info(f"   Latest signal: {all_signals[0].get('backtest_timestamp', 'Unknown') if all_signals else 'None'}")
                    self.logger.info(f"   Signal types: {set(s.get('signal_type', 'Unknown') for s in all_signals[:5])}")
                    self.logger.info("   Use --show-signals to see detailed list")
                
                # Enhanced performance analysis with Smart Money
                self._analyze_enhanced_performance(all_signals)
                
                # Smart Money specific analysis
                if self.smart_money_enabled:
                    self._analyze_smart_money_performance(all_signals)
                
                # Restore original config
                if original_min_conf is not None:
                    config.MIN_CONFIDENCE = original_min_conf
                
                return True
            else:
                self.logger.warning("❌ No MACD signals found in backtest period")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Enhanced MACD backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_enhanced_macd_backtest(self, df: pd.DataFrame, epic: str, timeframe: str, pair: str) -> List[Dict]:
        """Run MACD backtest with Smart Money enhancement and MTF analysis"""
        signals = []
        
        # Pre-calculate EMA200 on the full dataset if not already present
        if 'ema_200' not in df.columns or df['ema_200'].isna().all():
            self.logger.info("📊 Pre-calculating EMA200 on full dataset...")
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Determine the optimal lookback window size (same as live scanner)
        optimal_lookback_bars = {
            '5m': 576,   # 48 hours * 12 bars/hour
            '15m': 672,  # 168 hours * 4 bars/hour  
            '1h': 720,   # 720 hours * 1 bar/hour
        }.get(timeframe, 672)
        
        # MACD needs sufficient data for calculations
        min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 200)
        
        # Skip early bars where EMA200 is not yet valid
        # EMA200 needs at least 200 bars to be meaningful
        first_valid_ema200_idx = df['ema_200'].first_valid_index()
        if first_valid_ema200_idx is not None and first_valid_ema200_idx > min_bars:
            min_bars = max(min_bars, first_valid_ema200_idx + 1)
            self.logger.info(f"📊 Adjusted start to bar {min_bars} for valid EMA200 data")
        
        # Ensure we have enough history for the lookback window
        # But don't skip the entire backtest if we have slightly less data
        # The live scanner would still run with whatever data is available
        # FIXED: Don't limit to len(df) - 1, that leaves only 1 bar to check!
        if len(df) > optimal_lookback_bars:
            start_bar = max(min_bars, optimal_lookback_bars)
        else:
            # If we don't have enough data, start after minimum bars but process what we have
            start_bar = min_bars
            self.logger.warning(f"⚠️ Limited data: {len(df)} bars, optimal: {optimal_lookback_bars}. Starting from bar {start_bar}")
        
        # Check MTF availability
        mtf_available = (hasattr(self.strategy, 'detect_signal_with_mtf') and 
                        getattr(self.strategy, 'enable_mtf_analysis', False))
        
        self.logger.info(f"🧪 [BACKTEST] Starting Enhanced MACD backtest")
        self.logger.info(f"   Data points: {len(df)}")
        self.logger.info(f"   Optimal lookback: {optimal_lookback_bars} bars")
        self.logger.info(f"   Starting from bar: {start_bar}")
        self.logger.info(f"   MTF Available: {mtf_available}")
        self.logger.info(f"   Smart Money Available: {self.smart_money_enabled}")
        
        if start_bar >= len(df):
            self.logger.warning(f"⚠️ Not enough data for backtest. Need at least {start_bar} bars, have {len(df)}")
            return signals
        
        for i in range(start_bar, len(df)):
            try:
                # CRITICAL: Use fixed lookback window (same as live scanner)
                # Live scanner always has 'optimal_lookback_bars' of history available
                lookback_start = max(0, i - optimal_lookback_bars + 1)
                current_data = df.iloc[lookback_start:i+1].copy()
                
                # Preserve pre-calculated EMA200 values to avoid recalculation
                if 'ema_200' in df.columns:
                    current_data['ema_200'] = df['ema_200'].iloc[lookback_start:i+1]
                
                # CRITICAL FIX: Extract historical timestamp for session-aware thresholds
                current_bar = df.iloc[i]
                signal_timestamp = None
                try:
                    # Try to get timestamp from DataFrame index or columns
                    if hasattr(current_bar, 'name') and current_bar.name:
                        signal_timestamp = current_bar.name
                    elif 'datetime_utc' in current_bar.index:
                        signal_timestamp = current_bar['datetime_utc']
                    elif 'timestamp' in current_bar.index:
                        signal_timestamp = current_bar['timestamp']
                    
                    # Convert to datetime if needed
                    if isinstance(signal_timestamp, str):
                        from datetime import datetime
                        signal_timestamp = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
                    elif hasattr(signal_timestamp, 'to_pydatetime'):
                        signal_timestamp = signal_timestamp.to_pydatetime()
                        
                except Exception as ts_err:
                    self.logger.debug(f"Could not extract timestamp: {ts_err}")
                    signal_timestamp = None

                # ENHANCED: Use MTF-enhanced detection if available, otherwise standard detection
                if mtf_available:
                    self.logger.debug(f"🔄 [MTF] Using MTF-enhanced detection for bar {i}")
                    # FIX: Try to get actual timestamp from DataFrame
                    actual_timestamp = None
                    
                    # Check if DataFrame has timestamp columns
                    if 'start_time' in df.columns:
                        actual_timestamp = df.iloc[i]['start_time']
                    elif 'datetime_utc' in df.columns:
                        actual_timestamp = df.iloc[i]['datetime_utc']
                    elif hasattr(df.index[i], 'to_pydatetime'):
                        # Index is actually datetime
                        actual_timestamp = df.index[i]
                    else:
                        # Fallback - use integer index but try to convert
                        actual_timestamp = df.index[i] if i < len(df) else None
                    
                    # Keep original timestamp for detection (timezone conversion happens later)
                    signal = self.strategy.detect_signal_with_mtf(
                        current_data, epic, config.SPREAD_PIPS, timeframe, actual_timestamp
                    )
                else:
                    self.logger.debug(f"📊 [STANDARD] Using standard detection for bar {i}")
                    # FIX: Use actual DataFrame index timestamp instead of row position  
                    actual_timestamp = None
                    
                    # Check if DataFrame has timestamp columns
                    if 'start_time' in df.columns:
                        actual_timestamp = df.iloc[i]['start_time']
                    elif 'datetime_utc' in df.columns:
                        actual_timestamp = df.iloc[i]['datetime_utc']
                    elif hasattr(df.index[i], 'to_pydatetime'):
                        # Index is actually datetime
                        actual_timestamp = df.index[i]
                    else:
                        # Fallback - use integer index but try to convert
                        actual_timestamp = df.index[i] if i < len(df) else None
                    
                    # Keep original timestamp for detection (timezone conversion happens later)
                    signal = self.strategy.detect_signal(
                        current_data, epic, config.SPREAD_PIPS, timeframe, actual_timestamp
                    )
                
                # NEW: Also check for momentum confirmation signals even if no immediate signal
                momentum_signal = None
                if not signal and hasattr(self.strategy.signal_detector, 'check_momentum_confirmation_signals'):
                    try:
                        momentum_signal = self.strategy.signal_detector.check_momentum_confirmation_signals(
                            epic=epic,
                            timeframe=timeframe,
                            df_enhanced=current_data,
                            latest=df.iloc[i],
                            forex_optimizer=self.strategy.forex_optimizer
                        )
                        
                        if momentum_signal:
                            signal = momentum_signal
                            # Get the current bar timestamp for momentum signal (keep original for logging)
                            momentum_timestamp = self._get_proper_timestamp(df.iloc[i], i)
                            momentum_time_str = str(momentum_timestamp)
                            self.logger.info(f"🎯 MOMENTUM CONFIRMATION at {momentum_time_str}: {momentum_signal.get('signal_type', 'Unknown')}")
                    except Exception as e:
                        self.logger.debug(f"Momentum confirmation check failed in main loop: {e}")

                if signal:
                    # Add backtest metadata with original timestamp (timezone conversion happens later)
                    timestamp_value = self._get_proper_timestamp(df.iloc[i], i)
                    # Store original timestamp in UTC
                    signal['backtest_timestamp'] = timestamp_value
                    signal['backtest_index'] = i
                    # Ensure epic is preserved (might be lost during signal processing)
                    signal['epic'] = epic
                    signal['candle_data'] = {
                        'open': float(df.iloc[i]['open']),
                        'high': float(df.iloc[i]['high']),
                        'low': float(df.iloc[i]['low']),
                        'close': float(df.iloc[i]['close']),
                        'timestamp': timestamp_value
                    }
                    
                    # Set multiple timestamp fields for compatibility
                    signal['timestamp'] = timestamp_value
                    signal['market_timestamp'] = timestamp_value
                    
                    # Standardize confidence (same as EMA backtest)
                    confidence_value = signal.get('confidence', signal.get('confidence_score', 0))
                    if confidence_value is not None:
                        signal['confidence'] = confidence_value
                        signal['confidence_score'] = confidence_value
                    
                    # Log MTF information if available
                    mtf_info = ""
                    if 'mtf_analysis' in signal:
                        mtf_data = signal['mtf_analysis']
                        if mtf_data.get('enabled', False):
                            aligned = mtf_data.get('aligned_timeframes', 0)
                            total = mtf_data.get('total_timeframes', 0)
                            alignment_score = mtf_data.get('momentum_score', 0)  # MACD uses momentum_score
                            mtf_valid = mtf_data.get('mtf_valid', False)
                            mtf_info = f" | MTF: {aligned}/{total} aligned, score: {alignment_score:.1%}, valid: {mtf_valid}"
                    
                    # 🧠 SMART MONEY ENHANCEMENT
                    if self.smart_money_enabled:
                        signal = self._enhance_signal_with_smart_money(
                            signal, current_data, epic, timeframe, pair
                        )
                    
                    # Add performance metrics by looking ahead (same as EMA backtest)
                    enhanced_signal = self._add_performance_metrics(signal, df, i)
                    signals.append(enhanced_signal)
                    
                    # Only log basic signal info (reduced verbosity to allow summary table to show)
                    original_timestamp = signal['backtest_timestamp']
                    self.logger.debug(f"📊 MACD signal at {original_timestamp}: "
                                    f"{signal.get('signal_type')} (conf: {confidence_value:.1%}){mtf_info}")
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue
        
        # FIXED: Sort by timestamp (newest first) for consistency with EMA backtest
        sorted_signals = sorted(signals, key=lambda x: self._get_sortable_timestamp(x), reverse=True)
        
        # Ensure all timestamps are in UTC for consistency
        self.logger.debug("🕐 Ensuring all timestamps are in UTC...")
        for signal in sorted_signals:
            if 'timestamp' in signal:
                try:
                    utc_timestamp = signal['timestamp']
                    utc_dt = self._ensure_utc_timestamp(utc_timestamp)
                    # Convert to string format in UTC
                    signal['timestamp'] = utc_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                    signal['timezone'] = 'UTC'
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not process timestamp for signal: {e}")
            
            # Also process backtest_timestamp if it exists
            if 'backtest_timestamp' in signal:
                try:
                    original_timestamp = signal['backtest_timestamp']
                    utc_dt = self._ensure_utc_timestamp(original_timestamp)
                    # Convert to string format in UTC
                    signal['backtest_timestamp'] = utc_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                    
                    # Log the timestamp for traceability
                    self.logger.debug(f"🕐 UTC timestamp: {signal['backtest_timestamp']}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not process backtest_timestamp: {e}")
        
        self.logger.debug(f"📊 Generated {len(signals)} signals using modular strategy")
        self.logger.info("✅ All timestamps in UTC for consistency")
        
        return sorted_signals
    
    def _get_proper_timestamp(self, df_row, row_index: int) -> str:
        """Get proper timestamp with enhanced validation and debugging"""
        try:
            # Debug: Show what columns are available
            available_cols = list(df_row.index) if hasattr(df_row, 'index') else []
            self.logger.debug(f"🔍 Available columns: {available_cols[:10]}...")  # Show first 10
            
            # Try to get timestamp from different possible sources
            timestamp_candidates = []
            
            # Method 1: Direct datetime columns (expanded list)
            for col in ['datetime_utc', 'start_time', 'timestamp', 'datetime', 'time', 'date']:
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
                            # Try different datetime formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                                try:
                                    parsed_dt = datetime.strptime(candidate, fmt)
                                    return parsed_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                                except ValueError:
                                    continue
                            
                            # Try pandas parsing as fallback
                            try:
                                parsed_dt = pd.to_datetime(candidate)
                                return parsed_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                            except:
                                continue
                    
                    elif hasattr(candidate, 'strftime'):
                        # Already a datetime object
                        return candidate.strftime('%Y-%m-%d %H:%M:%S UTC')
                    
                    elif isinstance(candidate, (int, float)):
                        # Handle Unix timestamp
                        if candidate > 1000000000:  # Reasonable Unix timestamp
                            return datetime.fromtimestamp(candidate).strftime('%Y-%m-%d %H:%M:%S UTC')
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Failed to process {source}: {e}")
                    continue
            
            # Fallback: Generate estimated timestamp
            self.logger.debug(f"⚠️ No valid timestamp found, generating fallback")
            from datetime import datetime, timedelta
            base_time = datetime(2025, 8, 3, 0, 0, 0)
            estimated_time = base_time + timedelta(minutes=15 * row_index)
            return estimated_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            
        except Exception as e:
            self.logger.debug(f"⚠️ Timestamp extraction failed: {e}")
            fallback_time = datetime.now() - timedelta(minutes=15 * (1000 - row_index))
            return fallback_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def _get_sortable_timestamp(self, signal: Dict) -> pd.Timestamp:
        """Get timestamp for sorting (same as EMA backtest)"""
        try:
            timestamp_str = signal.get('backtest_timestamp', signal.get('timestamp', ''))
            if timestamp_str and timestamp_str != 'Unknown':
                return pd.to_datetime(timestamp_str)
            
            # Fallback using index
            index = signal.get('backtest_index', 0)
            base_time = pd.Timestamp('2025-08-04 00:00:00')
            return base_time + pd.Timedelta(minutes=15 * index)
            
        except Exception:
            return pd.Timestamp('1900-01-01')
    
    def _add_performance_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Add performance metrics by looking ahead (same as EMA backtest)"""
        try:
            enhanced_signal = signal.copy()
            
            entry_price = signal.get('price', df.iloc[signal_idx]['close'])
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()
            
            # Look ahead for performance (up to 96 bars for 15m = 24 hours)
            max_lookback = min(96, len(df) - signal_idx - 1)
            
            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]
                
                # CRITICAL FIX: Proper trade simulation with trailing stop (matches live trading)
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
    
    def _enhance_signal_with_smart_money(
        self, 
        signal: Dict, 
        df: pd.DataFrame, 
        epic: str, 
        timeframe: str,
        pair: str
    ) -> Dict:
        """Enhance signal with Smart Money analysis"""
        if not self.smart_money_enabled or not self.smart_money_analyzer:
            return signal
        
        try:
            start_time = datetime.now()
            
            # Use Smart Money analyzer to enhance the signal
            enhanced_signal = self.smart_money_analyzer.enhance_signal_with_smart_money(
                signal, df, epic, timeframe
            )
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Update stats
            self.smart_money_stats['signals_analyzed'] += 1
            
            if enhanced_signal and enhanced_signal.get('smart_money_analysis'):
                self.smart_money_stats['signals_enhanced'] += 1
                
                # Extract Smart Money scores for logging
                smc_data = enhanced_signal['smart_money_analysis']
                if smc_data.get('enabled', False):
                    structure_score = smc_data.get('structure_score', 0)
                    order_flow_score = smc_data.get('order_flow_score', 0)
                    confidence_boost = smc_data.get('confidence_boost', 0)
                    
                    self.logger.debug(f"🧠 Smart Money enhanced signal:")
                    self.logger.debug(f"   Structure Score: {structure_score:.1%}")
                    self.logger.debug(f"   Order Flow Score: {order_flow_score:.1%}")
                    self.logger.debug(f"   Confidence Boost: {confidence_boost:.1%}")
            
            # Track average analysis time
            total_time = self.smart_money_stats['avg_analysis_time'] * (self.smart_money_stats['signals_analyzed'] - 1)
            self.smart_money_stats['avg_analysis_time'] = (total_time + analysis_time) / self.smart_money_stats['signals_analyzed']
            
            return enhanced_signal if enhanced_signal else signal
            
        except Exception as e:
            self.smart_money_stats['analysis_failures'] += 1
            self.logger.warning(f"⚠️ Smart Money analysis failed: {e}")
            return signal
    
    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic code (same as EMA backtest)"""
        try:
            # Convert "CS.D.EURUSD.MINI.IP" to "EURUSD"
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
    
    def _display_epic_results(self, epic_results: Dict):
        """Display results by epic (same as EMA backtest)"""
        self.logger.info("\n📊 RESULTS BY EPIC:")
        self.logger.info("-" * 30)
        
        for epic, result in epic_results.items():
            if 'error' in result:
                self.logger.info(f"   {epic}: ❌ {result['error']}")
            else:
                self.logger.info(f"   {epic}: {result['signals']} signals")
    
    def _display_enhanced_signals(self, signals: List[Dict]):
        """Display individual signals with proper formatting and MTF information (FIXED like EMA backtest)"""
        self.logger.info("\n🎯 INDIVIDUAL ENHANCED MACD SIGNALS:")
        self.logger.info("=" * 140)
        self.logger.info("#   TIMESTAMP            PAIR     TYPE STRATEGY        PRICE    CONF   PROFIT   LOSS     R:R    MTF")
        self.logger.info("-" * 140)
        
        display_signals = signals[:20]  # Show max 20 signals
        
        for i, signal in enumerate(display_signals, 1):
            timestamp_str = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))
            
            epic = signal.get('epic', 'Unknown')
            if 'CS.D.' in epic and '.MINI.IP' in epic:
                pair = epic.split('.D.')[1].split('.MINI.IP')[0]
            elif epic == 'Unknown':
                pair = 'Unknown'  # Don't truncate "Unknown" to "nknown"
            else:
                pair = epic[-6:] if len(epic) >= 6 else epic
            
            signal_type = signal.get('signal_type', 'UNK')
            if signal_type in ['BUY', 'BULL', 'LONG']:
                type_display = 'BUY'
            elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                type_display = 'SELL'
            else:
                type_display = signal_type or 'UNK'
            
            confidence = signal.get('confidence', signal.get('confidence_score', 0))
            if confidence > 1:
                confidence = confidence / 100.0
            
            price = signal.get('price', 0)
            max_profit = signal.get('max_profit_pips', 0)
            max_loss = signal.get('max_loss_pips', 0)
            risk_reward = signal.get('profit_loss_ratio', max_profit / max_loss if max_loss > 0 else 0)
            
            # Add MTF information
            mtf_info = "N/A"
            if 'mtf_analysis' in signal:
                mtf_data = signal['mtf_analysis']
                if mtf_data.get('enabled', False):
                    aligned = mtf_data.get('aligned_timeframes', 0)
                    total = mtf_data.get('total_timeframes', 0)
                    score = mtf_data.get('momentum_score', 0)  # MACD uses momentum_score
                    valid = mtf_data.get('mtf_valid', False)
                    mtf_info = f"{aligned}/{total}({score:.0%}){'✅' if valid else '❌'}"
                else:
                    mtf_info = "DISABLED"
            
            row = f"{i:<3} {timestamp_str:<20} {pair:<8} {type_display:<4} {'macd_modular':<15} {price:<8.5f} {confidence:<6.1%} {max_profit:<8.1f} {max_loss:<8.1f} {risk_reward:<6.2f} {mtf_info:<10}"
            self.logger.info(row)
        
        self.logger.info("=" * 140)
        
        if len(signals) > 20:
            self.logger.info(f"📝 Showing latest 20 of {len(signals)} total signals (newest first)")
        else:
            self.logger.info(f"📝 Showing all {len(signals)} signals (newest first)")
        
        # Show MTF summary
        mtf_signals = [s for s in signals if 'mtf_analysis' in s and s['mtf_analysis'].get('enabled', False)]
        if mtf_signals:
            avg_alignment = sum(s['mtf_analysis'].get('momentum_score', 0) for s in mtf_signals) / len(mtf_signals)  # MACD uses momentum_score
            valid_mtf = sum(1 for s in mtf_signals if s['mtf_analysis'].get('mtf_valid', False))
            self.logger.info(f"\n📊 MTF Summary: {len(mtf_signals)}/{len(signals)} signals with MTF analysis")
            self.logger.info(f"   Average alignment: {avg_alignment:.1%}")
            self.logger.info(f"   MTF valid signals: {valid_mtf}/{len(mtf_signals)}")
        else:
            self.logger.info(f"\n📊 MTF Summary: No MTF analysis found in signals")

    def _display_simulation_signals(self, signals: List[Dict]):
        """Display live scanner simulation signals"""
        self.logger.info("\n🔄 LIVE SCANNER SIMULATION SIGNALS:")
        self.logger.info("=" * 80)
        self.logger.info("EPIC                    TYPE  CONF   PRICE     TIMESTAMP")
        self.logger.info("-" * 80)

        for signal in signals:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence', signal.get('confidence_score', 0))
            price = signal.get('price', 0)
            timestamp = signal.get('timestamp', 'Unknown')

            self.logger.info(f"{epic:<24} {signal_type:<4} {confidence:<6.1%} {price:<9.5f} {timestamp}")

        self.logger.info("=" * 80)

    def _display_signals(self, signals: List[Dict]):
        """Display individual signals using SignalAnalyzer (same as EMA backtest)"""
        self.logger.info("\n🎯 INDIVIDUAL MACD SIGNALS:")
        self.logger.info("-" * 50)
        
        # Use the correct method name from SignalAnalyzer
        try:
            self.signal_analyzer.display_signal_list(signals, max_signals=20)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not use SignalAnalyzer display: {e}")
            # Fallback to manual display
            for i, signal in enumerate(signals[:20], 1):
                timestamp = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))
                epic = signal.get('epic', 'Unknown')
                signal_type = signal.get('signal_type', 'Unknown')
                confidence = signal.get('confidence', signal.get('confidence_score', 0))
                price = signal.get('price', 0)
                
                # Add MACD context if available
                macd_info = ""
                if 'macd_values' in signal:
                    macd_vals = signal['macd_values']
                    histogram = macd_vals.get('macd_histogram', 0)
                    macd_info = f" | Hist: {histogram:.6f}"
                
                # Add MTF context if available
                mtf_info = ""
                if 'mtf_analysis' in signal:
                    mtf_data = signal['mtf_analysis']
                    if mtf_data.get('enabled', False):
                        aligned = mtf_data.get('aligned_timeframes', 0)
                        total = mtf_data.get('total_timeframes', 0)
                        mtf_info = f" | MTF: {aligned}/{total}"
                
                self.logger.info(f"{i:2d}. {timestamp} | {epic} | {signal_type} | Conf: {confidence:.1%} | Price: {price:.5f}{macd_info}{mtf_info}")
    
    def _analyze_enhanced_performance(self, signals: List[Dict]):
        """Analyze performance metrics with MTF information (same structure as EMA backtest)"""
        try:
            total_signals = len(signals)
            
            # Signal type categorization
            bull_signals = sum(1 for s in signals if s.get('signal_type', '').upper() in ['BUY', 'BULL', 'LONG'])
            bear_signals = sum(1 for s in signals if s.get('signal_type', '').upper() in ['SELL', 'BEAR', 'SHORT'])
            
            # Confidence analysis
            confidences = []
            for s in signals:
                conf = s.get('confidence', s.get('confidence_score', 0))
                if conf is not None:
                    if conf > 1:
                        conf = conf / 100.0
                    confidences.append(conf)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Performance metrics
            profit_signals = [s for s in signals if 'max_profit_pips' in s and 'max_loss_pips' in s]
            valid_performance_signals = [s for s in profit_signals if s.get('max_profit_pips', 0) > 0 or s.get('max_loss_pips', 0) > 0]
            
            self.logger.info("\n📈 ENHANCED MACD STRATEGY PERFORMANCE:")
            self.logger.info("=" * 50)
            self.logger.info(f"   📊 Total Signals: {total_signals}")
            self.logger.info(f"   🎯 Average Confidence: {avg_confidence:.1%}")
            self.logger.info(f"   📈 Bull Signals: {bull_signals}")
            self.logger.info(f"   📉 Bear Signals: {bear_signals}")
            
            # MTF Analysis Summary
            mtf_signals = [s for s in signals if 'mtf_analysis' in s and s['mtf_analysis'].get('enabled', False)]
            if mtf_signals:
                self.logger.info(f"\n🔄 MTF ANALYSIS SUMMARY:")
                self.logger.info(f"   📊 MTF Enhanced Signals: {len(mtf_signals)}/{total_signals} ({len(mtf_signals)/total_signals:.1%})")
                
                # MTF alignment statistics
                alignment_scores = [s['mtf_analysis'].get('momentum_score', 0) for s in mtf_signals]  # MACD uses momentum_score
                avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
                
                aligned_timeframes = [s['mtf_analysis'].get('aligned_timeframes', 0) for s in mtf_signals]
                total_timeframes = [s['mtf_analysis'].get('total_timeframes', 0) for s in mtf_signals]
                
                valid_mtf = sum(1 for s in mtf_signals if s['mtf_analysis'].get('mtf_valid', False))
                
                self.logger.info(f"   📈 Average Alignment Score: {avg_alignment:.1%}")
                self.logger.info(f"   ✅ MTF Valid Signals: {valid_mtf}/{len(mtf_signals)} ({valid_mtf/len(mtf_signals):.1%})")
                
                if aligned_timeframes and total_timeframes:
                    avg_aligned = sum(aligned_timeframes) / len(aligned_timeframes)
                    avg_total = sum(total_timeframes) / len(total_timeframes)
                    self.logger.info(f"   🔄 Average Timeframes Aligned: {avg_aligned:.1f}/{avg_total:.1f}")
                
                # MTF performance comparison
                mtf_valid_signals = [s for s in mtf_signals if s['mtf_analysis'].get('mtf_valid', False)]
                mtf_invalid_signals = [s for s in mtf_signals if not s['mtf_analysis'].get('mtf_valid', False)]
                
                if mtf_valid_signals and mtf_invalid_signals:
                    valid_profits = [s.get('max_profit_pips', 0) for s in mtf_valid_signals if 'max_profit_pips' in s]
                    invalid_profits = [s.get('max_profit_pips', 0) for s in mtf_invalid_signals if 'max_profit_pips' in s]
                    
                    if valid_profits and invalid_profits:
                        avg_valid_profit = sum(valid_profits) / len(valid_profits)
                        avg_invalid_profit = sum(invalid_profits) / len(invalid_profits)
                        self.logger.info(f"   💰 MTF Valid Avg Profit: {avg_valid_profit:.1f} pips")
                        self.logger.info(f"   💰 MTF Invalid Avg Profit: {avg_invalid_profit:.1f} pips")
            else:
                self.logger.info(f"\n🔄 MTF ANALYSIS: Not available or disabled")
            
            if valid_performance_signals:
                # UPDATED: Use actual trade outcomes from trailing stop simulation
                target_pips = 15
                
                # Classify based on actual exit P&L (more accurate with trailing stops)
                winners = [s for s in valid_performance_signals if s.get('exit_pnl', s.get('max_profit_pips', 0)) > 0]
                losers = [s for s in valid_performance_signals if s.get('exit_pnl', -s.get('max_loss_pips', 0)) < 0]
                breakeven = [s for s in valid_performance_signals if s.get('exit_pnl', 0) == 0]
                neutral = [s for s in valid_performance_signals if s not in winners and s not in losers and s not in breakeven]
                
                # CRITICAL FIX: Calculate averages from actual exit P&L (trailing stop aware)
                avg_profit = sum([s.get('exit_pnl', s.get('max_profit_pips', 0)) for s in winners]) / len(winners) if winners else 0
                avg_loss = sum([abs(s.get('exit_pnl', -s.get('max_loss_pips', 0))) for s in losers]) / len(losers) if losers else 0
                
                closed_trades = len(winners) + len(losers)
                win_rate = len(winners) / closed_trades if closed_trades > 0 else 0
                
                self.logger.info(f"\n💰 TRADE PERFORMANCE:")
                self.logger.info(f"   💰 Average Profit: {avg_profit:.1f} pips")
                self.logger.info(f"   📉 Average Loss: {avg_loss:.1f} pips")
                self.logger.info(f"   🏆 Win Rate: {win_rate:.1%}")
                self.logger.info(f"   📊 Trade Outcomes:")
                self.logger.info(f"      ✅ Winners: {len(winners)} (profitable exits)")
                self.logger.info(f"      ❌ Losers: {len(losers)} (loss exits)")
                self.logger.info(f"      ⚖️ Breakeven: {len(breakeven)} (no gain/loss)")
                self.logger.info(f"      ⚪ Neutral: {len(neutral)} (other)")
                
                # ENHANCED: Show top/worst trade execution details (limit verbose output)
                if winners:
                    self.logger.info(f"   🔍 TOP WINNERS (showing up to 5):")
                    top_winners = sorted(winners, key=lambda x: x.get('exit_pnl', x.get('max_profit_pips', 0)), reverse=True)[:5]
                    for i, signal in enumerate(top_winners, 1):
                        exit_pnl = signal.get('exit_pnl', signal.get('max_profit_pips', 0))
                        exit_reason = signal.get('exit_reason', 'UNKNOWN')
                        exit_bar = signal.get('exit_bar', 'N/A')
                        best_profit = signal.get('best_profit_achieved', 'N/A')
                        moved_to_be = signal.get('stop_moved_to_breakeven', False)
                        moved_to_profit = signal.get('stop_moved_to_profit', False)
                        
                        trailing_info = ""
                        if moved_to_profit:
                            trailing_info = " [TRAILED]"
                        elif moved_to_be:
                            trailing_info = " [BE]"
                            
                        self.logger.info(f"      {i}. {exit_pnl:.1f} pips ({exit_reason} at bar {exit_bar}){trailing_info}")
                        if best_profit != 'N/A' and best_profit > exit_pnl:
                            self.logger.info(f"         Best: {best_profit:.1f} pips (gave back {best_profit-exit_pnl:.1f})")
                        
                if losers:
                    self.logger.info(f"   🔍 WORST LOSERS (showing up to 5):")
                    worst_losers = sorted(losers, key=lambda x: x.get('exit_pnl', -x.get('max_loss_pips', 0)))[:5]
                    for i, signal in enumerate(worst_losers, 1):
                        exit_pnl = signal.get('exit_pnl', -signal.get('max_loss_pips', 0))
                        exit_reason = signal.get('exit_reason', 'UNKNOWN')
                        exit_bar = signal.get('exit_bar', 'N/A')
                        best_profit = signal.get('best_profit_achieved', 0)
                        moved_to_be = signal.get('stop_moved_to_breakeven', False)
                        
                        trailing_info = ""
                        if moved_to_be and best_profit >= 8:
                            trailing_info = " [WAS AT BE]"
                            
                        self.logger.info(f"      {i}. {exit_pnl:.1f} pips ({exit_reason} at bar {exit_bar}){trailing_info}")
                        if best_profit > 0:
                            self.logger.info(f"         Best: +{best_profit:.1f} pips before reversal")
                
                # Show exit reason statistics
                all_trades = winners + losers + neutral
                exit_reasons = {}
                for trade in all_trades:
                    reason = trade.get('exit_reason', 'UNKNOWN')
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
                self.logger.info(f"   🔍 EXIT REASONS: {dict(exit_reasons)}")
            else:
                self.logger.info(f"\n💰 TRADE PERFORMANCE:")
                self.logger.info(f"   💰 Average Profit: 0.0 pips (no valid data)")
                self.logger.info(f"   📉 Average Loss: 0.0 pips (no valid data)")
                self.logger.info(f"   🏆 Win Rate: 0.0% (no valid data)")
            
            # MACD-specific analysis
            self._analyze_macd_specific_performance(signals)
            
            # Integration performance analysis
            self._analyze_integration_performance(signals)
                
        except Exception as e:
            self.logger.error(f"⚠️ Performance analysis failed: {e}")
    
    def _analyze_macd_specific_performance(self, signals: List[Dict]):
        """Analyze MACD-specific performance characteristics"""
        self.logger.info("\n📊 MACD-SPECIFIC ANALYSIS:")
        self.logger.info("-" * 30)
        
        # Smart Money Analysis
        if self.smart_money_enabled:
            smart_money_signals = [s for s in signals if 'smart_money_analysis' in s and s['smart_money_analysis'].get('enabled', False)]
            standard_signals = len(signals) - len(smart_money_signals)
            
            self.logger.info(f"   🧠 Smart Money Enhanced: {len(smart_money_signals)}")
            self.logger.info(f"   📊 Standard MACD: {standard_signals}")
        
        # MACD histogram analysis (check for data availability)
        histogram_positive = 0
        histogram_negative = 0
        histogram_missing = 0
        
        for s in signals:
            # Try to find MACD data in different locations
            macd_histogram = None
            
            # Check nested macd_values structure
            if 'macd_values' in s and isinstance(s['macd_values'], dict):
                macd_histogram = s['macd_values'].get('macd_histogram')
            
            # Check direct signal fields
            if macd_histogram is None:
                macd_histogram = s.get('macd_histogram')
            
            if macd_histogram is not None:
                try:
                    if float(macd_histogram) > 0:
                        histogram_positive += 1
                    else:
                        histogram_negative += 1
                except (ValueError, TypeError):
                    histogram_missing += 1
            else:
                histogram_missing += 1
        
        self.logger.info(f"   📈 Histogram Positive: {histogram_positive}")
        self.logger.info(f"   📉 Histogram Negative: {histogram_negative}")
        if histogram_missing > 0:
            self.logger.info(f"   ❓ Histogram Missing: {histogram_missing}")
        
        # EMA200 filter analysis
        above_ema200 = 0
        below_ema200 = 0
        ema200_missing = 0
        
        for s in signals:
            price = s.get('price', 0)
            ema_200 = None
            
            # Check nested macd_values structure
            if 'macd_values' in s and isinstance(s['macd_values'], dict):
                ema_200 = s['macd_values'].get('ema_200')
            
            # Check direct signal fields
            if ema_200 is None:
                ema_200 = s.get('ema_200')
            
            # Also check common alternative field names
            if ema_200 is None:
                ema_200 = s.get('ema200')
            if ema_200 is None and 'candle_data' in s:
                ema_200 = s['candle_data'].get('ema_200')
            
            # Check if we have valid price and EMA200 data
            if price > 0 and ema_200 is not None:
                try:
                    ema_200_float = float(ema_200)
                    # Check for NaN or invalid values
                    if ema_200_float > 0 and not (ema_200_float != ema_200_float):  # Check for NaN
                        if price > ema_200_float:
                            above_ema200 += 1
                        else:
                            below_ema200 += 1
                    else:
                        ema200_missing += 1
                        self.logger.debug(f"⚠️ Invalid EMA200 value: {ema_200} (NaN or <= 0)")
                except (ValueError, TypeError) as e:
                    ema200_missing += 1
                    self.logger.debug(f"⚠️ Could not convert EMA200 to float: {ema_200} - {e}")
            else:
                ema200_missing += 1
                if price <= 0:
                    self.logger.debug(f"⚠️ Invalid price: {price}")
                if ema_200 is None:
                    # Debug: Show what fields are actually available
                    available_fields = list(s.keys())
                    self.logger.debug(f"⚠️ EMA200 not found in signal. Available fields: {available_fields}")
        
        self.logger.info(f"   📈 Signals above EMA200: {above_ema200}")
        self.logger.info(f"   📉 Signals below EMA200: {below_ema200}")
        if ema200_missing > 0:
            self.logger.info(f"   ❓ EMA200 data missing: {ema200_missing}")
        
        # MACD strength distribution
        strong_signals = 0
        moderate_signals = 0
        weak_signals = 0
        macd_missing = 0
        
        for s in signals:
            macd_histogram = None
            
            # Check nested macd_values structure
            if 'macd_values' in s and isinstance(s['macd_values'], dict):
                macd_histogram = s['macd_values'].get('macd_histogram')
            
            # Check direct signal fields
            if macd_histogram is None:
                macd_histogram = s.get('macd_histogram')
            
            if macd_histogram is not None:
                try:
                    abs_histogram = abs(float(macd_histogram))
                    if abs_histogram > 0.0001:
                        strong_signals += 1
                    elif abs_histogram >= 0.00005:
                        moderate_signals += 1
                    else:
                        weak_signals += 1
                except (ValueError, TypeError):
                    macd_missing += 1
            else:
                macd_missing += 1
        
        self.logger.info(f"   💪 Strong MACD signals: {strong_signals}")
        self.logger.info(f"   📊 Moderate MACD signals: {moderate_signals}")
        self.logger.info(f"   📝 Weak MACD signals: {weak_signals}")
        if macd_missing > 0:
            self.logger.info(f"   ❓ MACD data missing: {macd_missing}")
        
        # MACD crossover analysis
        bullish_crossovers = 0
        bearish_crossovers = 0
        invalid_crossovers = 0
        
        for s in signals:
            signal_type = s.get('signal_type', '').upper()
            macd_histogram = None
            
            # Check nested macd_values structure
            if 'macd_values' in s and isinstance(s['macd_values'], dict):
                macd_histogram = s['macd_values'].get('macd_histogram')
            
            # Check direct signal fields
            if macd_histogram is None:
                macd_histogram = s.get('macd_histogram')
            
            if macd_histogram is not None:
                try:
                    if signal_type in ['BULL', 'BUY', 'BULLISH'] and float(macd_histogram) > 0:
                        bullish_crossovers += 1
                    elif signal_type in ['BEAR', 'SELL', 'BEARISH'] and float(macd_histogram) < 0:
                        bearish_crossovers += 1
                    else:
                        invalid_crossovers += 1
                except (ValueError, TypeError):
                    invalid_crossovers += 1
            else:
                invalid_crossovers += 1
        
        self.logger.info(f"   🔄 Proper bullish crossovers: {bullish_crossovers}")
        self.logger.info(f"   🔄 Proper bearish crossovers: {bearish_crossovers}")
        if invalid_crossovers > 0:
            self.logger.info(f"   ⚠️ Invalid/missing crossovers: {invalid_crossovers}")
        
        # Data quality summary
        total_signals = len(signals)
        macd_data_quality = (total_signals - macd_missing) / total_signals if total_signals > 0 else 0
        ema_data_quality = (total_signals - ema200_missing) / total_signals if total_signals > 0 else 0
        
        self.logger.info(f"\n📊 MACD Data Quality:")
        self.logger.info(f"   MACD Histogram Available: {macd_data_quality:.1%}")
        self.logger.info(f"   EMA200 Data Available: {ema_data_quality:.1%}")
    
    def _analyze_smart_money_performance(self, signals: List[Dict]):
        """Analyze Smart Money specific performance metrics"""
        if not self.smart_money_enabled:
            return
        
        self.logger.info("\n🧠 SMART MONEY ANALYSIS PERFORMANCE:")
        self.logger.info("-" * 40)
        
        # Smart Money statistics
        self.logger.info(f"   📊 Signals Analyzed: {self.smart_money_stats['signals_analyzed']}")
        self.logger.info(f"   ✅ Signals Enhanced: {self.smart_money_stats['signals_enhanced']}")
        self.logger.info(f"   ❌ Analysis Failures: {self.smart_money_stats['analysis_failures']}")
        self.logger.info(f"   ⏱️ Average Analysis Time: {self.smart_money_stats['avg_analysis_time']:.3f}s")
        
        # Enhancement rate
        if self.smart_money_stats['signals_analyzed'] > 0:
            enhancement_rate = self.smart_money_stats['signals_enhanced'] / self.smart_money_stats['signals_analyzed']
            self.logger.info(f"   📈 Enhancement Rate: {enhancement_rate:.1%}")
    
    def _analyze_integration_performance(self, signals: List[Dict]):
        """Analyze performance of optimizer integration"""
        try:
            self.logger.info("\n🔗 INTEGRATION PERFORMANCE:")
            self.logger.info("-" * 30)
            
            # Check integration status
            integration_status = self.strategy.get_integration_status()
            
            self.logger.info(f"   🔧 Forex Optimizer: {'✅ Active' if integration_status.get('has_forex_optimizer') else '❌ Missing'}")
            self.logger.info(f"   🎯 Confidence Optimizer: {'✅ Active' if integration_status.get('has_confidence_optimizer') else '❌ Missing'}")
            
            # Smart Money status reporting
            smc_status = "✅ Active" if self.smart_money_enabled else "❌ Disabled"
            if self.smart_money_enabled:
                smc_enhanced_count = sum(1 for s in signals if 'smart_money_analysis' in s)
                smc_status += f" ({smc_enhanced_count} signals enhanced)"
            elif not SMART_MONEY_AVAILABLE:
                smc_status = "❌ Not Available (modules missing)"
            
            self.logger.info(f"   🧠 Smart Money Integration: {smc_status}")
            
            # Smart Money detailed status if enabled
            if self.smart_money_enabled:
                self.logger.info(f"   📊 SMC Analysis Success Rate: {(self.smart_money_stats['signals_enhanced'] / max(1, self.smart_money_stats['signals_analyzed'])) * 100:.1f}%")
                self.logger.info(f"   ⏱️ SMC Average Processing Time: {self.smart_money_stats['avg_analysis_time']:.3f}s")
            
            # Per-pair integration analysis
            pair_status = integration_status.get('pair_integration_status', {})
            integrated_pairs = sum(1 for status in pair_status.values() if status.get('integration_active', False))
            total_pairs = len(pair_status)
            
            self.logger.info(f"   🌍 Integrated pairs: {integrated_pairs}/{total_pairs}")
            
            # Confidence distribution analysis
            if signals:
                high_conf = sum(1 for s in signals if s.get('confidence', 0) > 0.7)
                med_conf = sum(1 for s in signals if 0.5 <= s.get('confidence', 0) <= 0.7)
                low_conf = sum(1 for s in signals if s.get('confidence', 0) < 0.5)
                
                self.logger.info(f"   📈 High confidence (>70%): {high_conf}")
                self.logger.info(f"   📊 Medium confidence (50-70%): {med_conf}")
                self.logger.info(f"   📉 Low confidence (<50%): {low_conf}")
            
            # Cache performance if available
            try:
                cache_stats = self.strategy.get_cache_stats()
                hit_ratio = cache_stats.get('hit_ratio', 0)
                self.logger.info(f"   🚀 Cache hit ratio: {hit_ratio:.1%}")
            except:
                self.logger.info(f"   🚀 Cache performance: Not available")
                
        except Exception as e:
            self.logger.debug(f"Integration performance analysis failed: {e}")


def main():
    """Main execution function with enhanced Smart Money options and signal validation"""
    parser = argparse.ArgumentParser(description='Enhanced MACD Strategy Backtest with Smart Money Analysis and Signal Validation')

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--simulate-live', action='store_true',
                           help='Simulate live scanner behavior using real signal detector')
    mode_group.add_argument('--backtest', action='store_true', default=True,
                           help='Run backtest using modular strategy (default)')

    # Required arguments
    parser.add_argument('--epic', help='Epic to backtest (e.g., CS.D.EURUSD.MINI.IP)')
    parser.add_argument('--days', type=int, default=7, help='Days to backtest (default: 7)')
    parser.add_argument('--timeframe', default=None, help=f'Timeframe (default: {getattr(config, "DEFAULT_TIMEFRAME", "15m")})')

    # Optional arguments
    parser.add_argument('--show-signals', action='store_true', help='Show individual signals')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    # Enhanced arguments
    parser.add_argument('--disable-forex-integration', action='store_true', help='Disable forex optimizer integration')
    parser.add_argument('--optimize-gbp', action='store_true', help='Apply GBP-specific optimizations')
    parser.add_argument('--optimize-jpy', action='store_true', help='Apply JPY-specific optimizations')
    
    # Smart Money arguments
    parser.add_argument('--smart-money', action='store_true', help='Enable Smart Money Concepts analysis')
    parser.add_argument('--smc-only', action='store_true', help='Only show Smart Money enhanced signals')
    parser.add_argument('--structure-analysis', action='store_true', help='Enable detailed market structure analysis')
    parser.add_argument('--order-flow-analysis', action='store_true', help='Enable detailed order flow analysis')
    parser.add_argument('--force-smart-money', action='store_true', help='Force enable Smart Money even if modules missing')
    parser.add_argument('--no-optimal-params', action='store_true', help='Disable database optimization (use static parameters)')
    
    # NEW: Signal validation arguments
    parser.add_argument('--validate-signal', help='Validate a specific signal by timestamp (format: "YYYY-MM-DD HH:MM:SS")')
    parser.add_argument('--show-raw-data', action='store_true', help='Show raw OHLC data around signal (use with --validate-signal)')
    parser.add_argument('--show-calculations', action='store_true', help='Show detailed MACD and EMA calculations (use with --validate-signal)')
    parser.add_argument('--show-decision-tree', action='store_true', help='Show decision-making process (use with --validate-signal)')
    
    args = parser.parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # NEW: Handle signal validation mode
    if args.validate_signal:
        if not args.epic:
            print("❌ ERROR: --epic is required when using --validate-signal")
            print("   Example: python backtest_macd.py --epic CS.D.EURUSD.MINI.IP --validate-signal \"2025-08-04 15:30:00\"")
            sys.exit(1)
        
        print("🔍 SIGNAL VALIDATION MODE")
        print("=" * 50)
        print(f"📊 Epic: {args.epic}")
        print(f"⏰ Timestamp: {args.validate_signal}")
        print(f"📈 Timeframe: {args.timeframe}")
        print(f"📊 Show raw data: {'✅' if args.show_raw_data else '❌'}")
        print(f"🧮 Show calculations: {'✅' if args.show_calculations else '❌'}")
        print(f"🌳 Show decision tree: {'✅' if args.show_decision_tree else '❌'}")
        
        # Initialize backtest for validation
        backtest = EnhancedMACDBacktest()
        
        # Smart Money setup for validation
        enable_smart_money = args.smart_money or args.smc_only or args.structure_analysis or args.order_flow_analysis or args.force_smart_money
        if enable_smart_money:
            smart_money_init_success = backtest.initialize_smart_money_analysis(True)
            if not smart_money_init_success and not args.force_smart_money:
                print("⚠️ WARNING: Smart Money modules not available for validation!")
                print("   Use --force-smart-money to continue anyway")
                sys.exit(1)
        
        # Run signal validation
        success = backtest.validate_single_signal(
            epic=args.epic,
            timestamp=args.validate_signal,
            timeframe=args.timeframe,
            show_raw_data=args.show_raw_data,
            show_calculations=args.show_calculations if args.show_calculations else True,  # Default to True
            show_decision_tree=args.show_decision_tree if args.show_decision_tree else True  # Default to True
        )
        
        if success:
            print("\n✅ Signal validation completed successfully!")
        else:
            print("\n❌ Signal validation failed!")
            sys.exit(1)
        
        sys.exit(0)
    
    # EXISTING: Handle normal backtest mode
    # Smart Money configuration and diagnostics
    enable_smart_money = args.smart_money or args.smc_only or args.structure_analysis or args.order_flow_analysis or args.force_smart_money
    
    print("🧠 SMART MONEY DIAGNOSTICS:")
    print(f"   Modules Available: {'✅' if SMART_MONEY_AVAILABLE else '❌'}")
    print(f"   Final enable_smart_money: {'✅' if enable_smart_money else '❌'}")
    
    if enable_smart_money and not SMART_MONEY_AVAILABLE and not args.force_smart_money:
        print("⚠️ WARNING: Smart Money modules not available!")
        print("   Use --force-smart-money to continue anyway")
        sys.exit(1)
    
    # Prepare optimization config
    optimization_config = {}
    
    if args.optimize_gbp:
        optimization_config['GBPUSD'] = {
            'base_confidence': 0.55,
            'volatility_adjustment': 0.65,
            'macd_thresholds': {'strong': 0.00015, 'moderate': 0.00008},
            'forex_integration': True
        }
        print("🔧 Applied GBP-specific optimizations")
    
    if args.optimize_jpy:
        optimization_config['USDJPY'] = {
            'base_confidence': 0.70,
            'volatility_adjustment': 0.95,
            'macd_thresholds': {'strong': 0.010, 'moderate': 0.005},
            'forex_integration': True
        }
        print("🔧 Applied JPY-specific optimizations")
    
    # Run enhanced backtest or live scanner simulation with Smart Money
    backtest = EnhancedMACDBacktest()

    if args.simulate_live:
        success = backtest.run_live_scanner_simulation(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=True,
            enable_smart_money=enable_smart_money
        )
    else:
        success = backtest.run_backtest(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=args.show_signals,
            min_confidence=args.min_confidence,
            optimization_config=optimization_config if optimization_config else None,
            enable_forex_integration=not args.disable_forex_integration,
            enable_smart_money=enable_smart_money,
            use_optimal_parameters=not args.no_optimal_params
        )
    
    if success:
        print("\n✅ Enhanced MACD backtest with Smart Money analysis completed successfully!")
        if enable_smart_money and SMART_MONEY_AVAILABLE:
            print("🧠 Smart Money Concepts analysis was included in results")
        elif enable_smart_money and not SMART_MONEY_AVAILABLE:
            print("⚠️ Smart Money analysis requested but modules not available")
        print("\n🔍 TIP: Use --validate-signal \"YYYY-MM-DD HH:MM:SS\" to validate specific signals")
        sys.exit(0)
    else:
        print("\n❌ Enhanced MACD backtest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()