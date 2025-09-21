#!/usr/bin/env python3
"""
Enhanced EMA Strategy Backtest with Smart Money Analysis - Complete Integration
Run: python backtest_ema.py --epic CS.D.EURUSD.CEEM.IP --days 7 --timeframe 15m --smart-money

ENHANCEMENTS ADDED:
- Smart Money Concepts (SMC) integration  
- Advanced trailing stop system (matches MACD and live trading)
- UTC timestamp consistency for all operations
- Market Structure Analysis (BOS, ChoCh, Swing Points)
- Order Flow Analysis (Order Blocks, Fair Value Gaps)
- Institutional Supply/Demand Zone Detection
- Enhanced signal validation with smart money confluence
- Smart money performance metrics and analysis
- Proper trailing stop simulation with breakeven and profit protection
- Exit reason tracking (PROFIT_TARGET, TRAILING_STOP, STOP_LOSS)
- Comprehensive trade outcome analysis
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
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

sys.path.insert(0, project_root)

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.strategies.ema_strategy import EMAStrategy
from performance_analyzer import PerformanceAnalyzer
from signal_analyzer import SignalAnalyzer

# Smart Money Integration imports (with fallback handling)
try:
    from core.smart_money_readonly_analyzer import SmartMoneyReadOnlyAnalyzer
    from core.smart_money_integration import SmartMoneyIntegration
    SMART_MONEY_AVAILABLE = True
except ImportError:
    SMART_MONEY_AVAILABLE = False
    logging.getLogger(__name__).warning("Smart Money modules not available - running without SMC analysis")

from configdata import config as strategy_config
try:
    import config
except ImportError:
    from forex_scanner import config


class EMABacktest:
    """Enhanced EMA Strategy Backtesting with Smart Money Analysis Integration"""
    
    def __init__(self):
        self.logger = logging.getLogger('ema_backtest')
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
            Formatted UTC timestamp string
        """
        try:
            # If it's already a string
            if isinstance(timestamp, str):
                # Ensure UTC suffix if not present
                if 'UTC' not in timestamp:
                    return f"{timestamp} UTC"
                return timestamp
                
            # If it's a pandas Timestamp or datetime, format it
            if hasattr(timestamp, 'strftime'):
                return timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                
            # Fallback
            return f"{str(timestamp)} UTC"
            
        except Exception as e:
            self.logger.debug(f"Timestamp conversion error: {e}")
            return f"{str(timestamp)} UTC"
    
    def initialize_ema_strategy(self, ema_config: str = None, epic: str = None, use_optimal_parameters: bool = True):
        """ENHANCED: Initialize EMA strategy with database optimization support"""
        
        # Check if we should use optimal parameters from database
        use_optimal = use_optimal_parameters and epic is not None
        
        if use_optimal:
            try:
                from optimization.optimal_parameter_service import OptimalParameterService
                param_service = OptimalParameterService()
                optimal_params = param_service.get_epic_parameters(epic)
                
                if optimal_params:
                    self.logger.info(f"🎯 Using DATABASE OPTIMAL parameters for {epic}:")
                    self.logger.info(f"   EMA Config: {optimal_params.ema_config}")
                    self.logger.info(f"   Confidence: {optimal_params.confidence_threshold:.1%}")
                    self.logger.info(f"   SL/TP: {optimal_params.stop_loss_pips:.0f}/{optimal_params.take_profit_pips:.0f} pips")
                    self.logger.info(f"   Performance Score: {optimal_params.performance_score:.3f}")
                    
                    # Initialize strategy with optimal parameters enabled
                    self.strategy = EMAStrategy(
                        data_fetcher=self.data_fetcher, 
                        backtest_mode=True,
                        use_optimal_parameters=True
                    )
                    
                    # Set the epic so strategy can load its optimal parameters
                    if hasattr(self.strategy, '_epic'):
                        self.strategy._epic = epic
                    
                    self.logger.info("✅ EMA Strategy initialized with DATABASE OPTIMIZATION")
                    return None
                    
            except Exception as e:
                self.logger.warning(f"❌ Failed to load optimal parameters for {epic}: {e}")
                self.logger.warning("   Falling back to static configuration...")
        
        # Fallback to static configuration
        original_config = None
        if ema_config and hasattr(strategy_config, 'ACTIVE_EMA_CONFIG'):
            original_config = strategy_config.ACTIVE_EMA_CONFIG
            self.logger.info(f"🔧 Using static EMA config: {ema_config}")
        
        # Initialize with static configuration
        self.strategy = EMAStrategy(data_fetcher=self.data_fetcher, backtest_mode=True)
        self.logger.info("✅ EMA Strategy initialized with STATIC CONFIGURATION")
        self.logger.info("🔥 BACKTEST MODE ENABLED: Time-based cooldowns disabled for historical analysis")
        
        # Get EMA configuration details for display
        ema_configs = getattr(strategy_config, 'EMA_STRATEGY_CONFIG', {})
        active_config = ema_config or getattr(strategy_config, 'ACTIVE_EMA_CONFIG', 'default')
        
        if active_config in ema_configs:
            periods = ema_configs[active_config]
            self.logger.info(f"   📊 EMA Periods: {periods.get('short')}/{periods.get('long')}/{periods.get('trend')}")
        
        return original_config
    
    def initialize_smart_money_integration(self, enable_smart_money: bool = False):
        """Initialize Smart Money analysis if available and requested"""
        if not enable_smart_money or not SMART_MONEY_AVAILABLE:
            self.smart_money_enabled = False
            self.smart_money_stats = {
                'signals_analyzed': 0,
                'signals_enhanced': 0,
                'analysis_failures': 0,
                'avg_analysis_time': 0.0
            }
            return
        
        try:
            self.smart_money_analyzer = SmartMoneyReadOnlyAnalyzer(
                data_fetcher=self.data_fetcher
            )
            
            self.smart_money_integration = SmartMoneyIntegration(
                database_manager=self.db_manager,
                data_fetcher=self.data_fetcher
            )
            
            self.smart_money_enabled = True
            self.smart_money_stats = {
                'signals_analyzed': 0,
                'signals_enhanced': 0,
                'analysis_failures': 0,
                'avg_analysis_time': 0.0
            }
            self.logger.info("🧠 Smart Money analysis ENABLED")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Smart Money initialization failed: {e}")
            self.smart_money_enabled = False
            self.smart_money_stats = {
                'signals_analyzed': 0,
                'signals_enhanced': 0,
                'analysis_failures': 0,
                'avg_analysis_time': 0.0
            }
    
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
                self.initialize_smart_money_integration(enable_smart_money)
            
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

    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        ema_config: str = None,
        min_confidence: float = None,
        enable_smart_money: bool = False,
        use_optimal_parameters: bool = True
    ) -> bool:
        """ENHANCED: Run EMA strategy backtest with database optimization support"""
        
        epic_list = [epic] if epic else config.EPIC_LIST
        
        self.logger.info("🧪 ENHANCED EMA STRATEGY BACKTEST")
        self.logger.info("=" * 40)
        self.logger.info(f"📊 Epic(s): {epic_list}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🎯 Show signals: {show_signals}")
        self.logger.info(f"🧠 Smart Money analysis: {enable_smart_money}")
        self.logger.info(f"🎯 Database optimization: {'✅ ENABLED' if use_optimal_parameters else '❌ DISABLED'}")
        
        # Initialize Smart Money analysis if enabled
        if enable_smart_money:
            self.initialize_smart_money_integration(enable_smart_money)
        
        if min_confidence:
            original_min_conf = getattr(config, 'MIN_CONFIDENCE', 0.7)
            config.MIN_CONFIDENCE = min_confidence
            self.logger.info(f"🎚️ Min confidence: {min_confidence:.1%} (was {original_min_conf:.1%})")
        
        try:
            all_signals = []
            epic_results = {}
            
            for current_epic in epic_list:
                self.logger.info(f"\n📈 Processing {current_epic}")
                
                # ENHANCED: Initialize strategy per epic for optimal parameters
                original_config = self.initialize_ema_strategy(
                    ema_config=ema_config, 
                    epic=current_epic,
                    use_optimal_parameters=use_optimal_parameters
                )
                
                # Extract pair from epic
                pair = self._extract_pair_from_epic(current_epic)
                
                # CRITICAL FIX: Use same data fetching as live scanner
                df = self.data_fetcher.get_enhanced_data(
                    epic=current_epic,
                    pair=pair,
                    timeframe=timeframe,
                    lookback_hours=days * 24,
                    ema_strategy=self.strategy  # Pass strategy for proper enhancement
                )
                
                if df is None:
                    self.logger.warning(f"❌ Failed to fetch data for {current_epic}")
                    epic_results[current_epic] = {'signals': 0, 'error': 'Data fetch failed'}
                    continue
                    
                if df.empty:
                    self.logger.warning(f"❌ No data available for {current_epic}")
                    epic_results[current_epic] = {'signals': 0, 'error': 'No data'}
                    continue
                
                self.logger.info(f"   📊 Data points: {len(df)}")
                
                # Show data range info
                if len(df) > 0:
                    first_row = df.iloc[0]
                    last_row = df.iloc[-1]
                    
                    start_time = self._get_proper_timestamp(first_row, 0)
                    end_time = self._get_proper_timestamp(last_row, len(df)-1)
                    
                    self.logger.info(f"   📅 Data range: {start_time} to {end_time}")
                
                # CRITICAL FIX: Run backtest using the CORRECT strategy method
                signals = self._run_ema_backtest_fixed(df, current_epic, timeframe)
                
                all_signals.extend(signals)
                epic_results[current_epic] = {'signals': len(signals)}
                
                self.logger.info(f"   🎯 EMA signals found: {len(signals)}")
            
            # Display results
            self._display_epic_results(epic_results)
            
            if all_signals:
                self.logger.info(f"\n✅ TOTAL EMA SIGNALS: {len(all_signals)}")
                
                if show_signals:
                    self._display_signals(all_signals)
                
                self._analyze_performance(all_signals)
                
                if original_config:
                    config.ACTIVE_EMA_CONFIG = original_config
                
                self.log_performance_summary()
                return True
            else:
                self.logger.warning("❌ No EMA signals found in backtest period")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ EMA backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_ema_backtest_fixed(self, df: pd.DataFrame, epic: str, timeframe: str) -> List[Dict]:
        """CRITICAL FIX: Run EMA backtest using the actual strategy detect_signal method"""
        signals = []
        
        # Use same minimum bars as live scanner
        min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)
        
        for i in range(min_bars, len(df)):
            try:
                # Get data up to current point (simulate real-time)
                current_data = df.iloc[:i+1].copy()
                
                # Get the current market timestamp for this iteration
                current_timestamp = self._get_proper_timestamp(df.iloc[i], i)
                
                # ENHANCED: Use MTF-enhanced detection if available, otherwise standard method
                # This prioritizes multi-timeframe analysis for higher quality signals
                if self.strategy.enable_mtf_analysis and self.strategy.mtf_analyzer:
                    self.logger.debug(f"🔄 Using MTF-enhanced detection for {epic}")
                    # Note: detect_signal_with_mtf doesn't accept evaluation_time parameter
                    signal = self.strategy.detect_signal_with_mtf(
                        current_data, epic, config.SPREAD_PIPS, timeframe
                    )
                else:
                    signal = self.strategy.detect_signal(
                        current_data, epic, config.SPREAD_PIPS, timeframe, 
                        evaluation_time=current_timestamp
                    )
                
                if signal:
                    # SMART MONEY ENHANCEMENT: Apply Smart Money analysis to detected signal
                    if self.smart_money_enabled and hasattr(self, 'smart_money_integration'):
                        self.logger.info(f"🧠 Applying Smart Money analysis to {signal.get('signal_type', 'Unknown')} signal...")
                        try:
                            enhanced_signal = self.smart_money_integration.enhance_signal_with_smart_money(
                                signal=signal,
                                epic=epic,
                                timeframe=timeframe
                            )
                            if enhanced_signal:
                                signal = enhanced_signal
                                self.smart_money_stats['signals_enhanced'] += 1
                                self.logger.info(f"✅ Smart Money enhancement applied successfully")
                            else:
                                self.logger.info(f"⚠️ Smart Money enhancement returned None, using original signal")
                            self.smart_money_stats['signals_analyzed'] += 1
                        except Exception as e:
                            self.logger.warning(f"❌ Smart Money enhancement failed: {e}")
                            self.smart_money_stats['analysis_failures'] += 1
                    
                    # Standardize confidence FIRST to avoid display bugs
                    confidence_value = signal.get('confidence', signal.get('confidence_score', 0))
                    if confidence_value is not None:
                        signal['confidence'] = confidence_value
                        signal['confidence_score'] = confidence_value
                    
                    # Log the signal detection with market timestamp
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"🎯 SIGNAL DETECTED at market time: {current_timestamp}")
                    self.logger.info(f"   Type: {signal.get('signal_type', 'Unknown')}")
                    self.logger.info(f"   Confidence: {signal.get('confidence', 0):.1%}")
                    self.logger.info(f"   Price: {signal.get('price', 0):.5f}")
                    
                    # Show Smart Money analysis if available
                    if signal.get('smart_money_analysis'):
                        sm_analysis = signal['smart_money_analysis']
                        self.logger.info(f"🧠 Smart Money Analysis:")
                        self.logger.info(f"   Market Structure: {sm_analysis.get('market_structure', {}).get('trend', 'N/A')}")
                        self.logger.info(f"   Order Flow: {sm_analysis.get('order_flow', {}).get('strength', 'N/A')}")
                        self.logger.info(f"   Confluence Score: {sm_analysis.get('confluence_score', 0):.1%}")
                    
                    self.logger.info(f"{'='*60}")
                    
                    # Add backtest metadata
                    timestamp_value = current_timestamp
                    signal['backtest_timestamp'] = timestamp_value
                    signal['backtest_index'] = i
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
                    
                    # Confidence already standardized earlier
                    
                    # Add performance metrics
                    enhanced_signal = self._add_performance_metrics(signal, df, i)
                    signals.append(enhanced_signal)
                    
                    self.logger.debug(f"📊 EMA signal at {signal['backtest_timestamp']}: "
                                    f"{signal.get('signal_type')} (conf: {confidence_value:.1%})")
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue
        
        # Sort by timestamp (newest first) for consistency with live scanner
        sorted_signals = sorted(signals, key=lambda x: self._get_sortable_timestamp(x), reverse=True)
        
        self.logger.debug(f"📊 Generated {len(signals)} signals using modular strategy")
        
        return sorted_signals
    
    def _get_proper_timestamp(self, df_row, row_index: int) -> str:
        """Get proper timestamp from data row (ensures UTC)"""
        try:
            # Try different timestamp sources
            for col in ['datetime_utc', 'start_time', 'timestamp', 'datetime']:
                if col in df_row and df_row[col] is not None:
                    candidate = df_row[col]
                    if isinstance(candidate, str) and candidate != 'Unknown':
                        # Ensure UTC suffix if not present
                        if 'UTC' not in candidate:
                            return f"{candidate} UTC"
                        return candidate
                    elif hasattr(candidate, 'strftime'):
                        return candidate.strftime('%Y-%m-%d %H:%M:%S UTC')
            
            # Try index if available
            if hasattr(df_row, 'name') and df_row.name is not None:
                if hasattr(df_row.name, 'strftime'):
                    return df_row.name.strftime('%Y-%m-%d %H:%M:%S UTC')
            
            # Fallback - use UTC time
            base_time = datetime(2025, 8, 3, 0, 0, 0)  # UTC base time
            estimated_time = base_time + timedelta(minutes=15 * row_index)
            return estimated_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            
        except Exception:
            # Use UTC for fallback
            fallback_time = datetime.utcnow() - timedelta(minutes=15 * (1000 - row_index))
            return fallback_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def _get_sortable_timestamp(self, signal: Dict) -> pd.Timestamp:
        """Get timestamp for sorting"""
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
        """Add performance metrics by looking ahead"""
        try:
            enhanced_signal = signal.copy()
            
            entry_price = signal.get('price', df.iloc[signal_idx]['close'])
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()
            
            # Look ahead for performance (up to 96 bars for 15m = 24 hours)
            max_lookback = min(96, len(df) - signal_idx - 1)
            
            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]
                
                if signal_type in ['BUY', 'BULL', 'LONG']:
                    highest_price = future_data['high'].max()
                    lowest_price = future_data['low'].min()
                    max_profit = max(0, (highest_price - entry_price) * 10000)
                    max_loss = max(0, (entry_price - lowest_price) * 10000)
                    
                elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                    highest_price = future_data['high'].max()
                    lowest_price = future_data['low'].min()
                    max_profit = max(0, (entry_price - lowest_price) * 10000)
                    max_loss = max(0, (highest_price - entry_price) * 10000)
                else:
                    max_profit = 0
                    max_loss = 0
                
                # CRITICAL FIX: Proper trade simulation with trailing stop (matches MACD and live trading)
                target_pips = 15  # Profit target
                initial_stop_pips = 10  # Initial stop loss
                
                # Trailing Stop Configuration (matches MACD and live trading setup)
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
                            
                            # TRAILING STOP LOGIC (matches MACD and live trading)
                            
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
                        
                        # Check exit conditions with dynamic trailing stop
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
                
                # Determine final trade outcome based on simulation
                if trade_closed:
                    if exit_reason == "PROFIT_TARGET":
                        trade_outcome = "WIN"
                        is_winner = True
                        is_loser = False
                        final_profit = exit_pnl
                        final_loss = 0
                    elif exit_reason in ["STOP_LOSS", "TRAILING_STOP"]:
                        if exit_pnl > 0:
                            trade_outcome = "WIN"
                            is_winner = True
                            is_loser = False
                            final_profit = exit_pnl
                            final_loss = 0
                        else:
                            trade_outcome = "LOSE"
                            is_winner = False
                            is_loser = True
                            final_profit = 0
                            final_loss = abs(exit_pnl)
                    else:
                        trade_outcome = "NEUTRAL"
                        is_winner = False
                        is_loser = False
                        final_profit = max(exit_pnl, 0)
                        final_loss = max(-exit_pnl, 0)
                else:
                    # CRITICAL FIX: Trade timeout - use realistic exit at current market price
                    # Instead of max excursions, exit at final available price
                    if len(future_data) > 0:
                        final_price = future_data.iloc[-1]['close']  # Exit at last available price
                        
                        if signal_type in ['BUY', 'BULL', 'LONG']:
                            final_exit_pnl = (final_price - entry_price) * 10000
                        else:  # SELL/SHORT
                            final_exit_pnl = (entry_price - final_price) * 10000
                        
                        # Determine realistic outcome based on final exit
                        if final_exit_pnl > 5.0:  # Profitable exit (>5 pips)
                            trade_outcome = "WIN_TIMEOUT"
                            is_winner = True
                            is_loser = False
                            final_profit = round(final_exit_pnl, 1)
                            final_loss = 0
                        elif final_exit_pnl < -3.0:  # Loss exit (>3 pips loss)
                            trade_outcome = "LOSE_TIMEOUT"
                            is_winner = False
                            is_loser = True
                            final_profit = 0
                            final_loss = round(abs(final_exit_pnl), 1)
                        else:  # Small profit/loss - breakeven
                            trade_outcome = "BREAKEVEN_TIMEOUT"
                            is_winner = False
                            is_loser = False
                            final_profit = max(final_exit_pnl, 0)
                            final_loss = max(-final_exit_pnl, 0)
                    else:
                        # No future data - neutral outcome
                        trade_outcome = "NO_DATA"
                        is_winner = False
                        is_loser = False
                        final_profit = 0
                        final_loss = 0
                
                enhanced_signal.update({
                    'max_profit_pips': round(final_profit, 1),
                    'max_loss_pips': round(final_loss, 1),
                    'profit_loss_ratio': round(final_profit / final_loss, 2) if final_loss > 0 else float('inf'),
                    'lookback_bars': max_lookback,
                    'entry_price': entry_price,
                    'is_winner': is_winner,
                    'is_loser': is_loser,
                    'trade_outcome': trade_outcome,
                    'exit_reason': exit_reason,
                    'exit_bar': exit_bar,
                    'exit_pnl': exit_pnl,
                    'target_pips': target_pips,
                    'initial_stop_pips': initial_stop_pips,
                    'trailing_stop_used': stop_moved_to_profit or stop_moved_to_breakeven,
                    'best_profit_achieved': best_profit_pips,
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
    
    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic"""
        try:
            if '.D.' in epic and '.MINI.IP' in epic:
                parts = epic.split('.D.')
                if len(parts) > 1:
                    pair_part = parts[1].split('.MINI.IP')[0]
                    return pair_part
            
            # Fallback to config
            pair_info = getattr(config, 'PAIR_INFO', {})
            if epic in pair_info:
                return pair_info[epic].get('pair', 'EURUSD')
            
            self.logger.warning(f"⚠️ Could not extract pair from {epic}, using EURUSD")
            return 'EURUSD'
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting pair from {epic}: {e}, using EURUSD")
            return 'EURUSD'
    
    def _display_epic_results(self, epic_results: Dict):
        """Display results by epic"""
        self.logger.info("\n📊 RESULTS BY EPIC:")
        self.logger.info("-" * 30)
        
        for epic, result in epic_results.items():
            if 'error' in result:
                self.logger.info(f"   {epic}: ❌ {result['error']}")
            else:
                self.logger.info(f"   {epic}: {result['signals']} signals")
    
    def _display_signals(self, signals: List[Dict]):
        """Display individual signals"""
        self.logger.info("\n🎯 INDIVIDUAL EMA SIGNALS:")
        self.logger.info("=" * 120)
        self.logger.info("#   TIMESTAMP            PAIR     TYPE STRATEGY        PRICE    CONF   PROFIT   LOSS     R:R   ")
        self.logger.info("-" * 120)
        
        display_signals = signals[:20]  # Show max 20 signals
        
        for i, signal in enumerate(display_signals, 1):
            timestamp_str = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))
            
            epic = signal.get('epic', 'Unknown')
            if 'CS.D.' in epic and '.MINI.IP' in epic:
                pair = epic.split('.D.')[1].split('.MINI.IP')[0]
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
            risk_reward = signal.get('profit_loss_ratio', max_profit / max_loss if max_loss > 0 else float('inf'))
            
            # Performance metrics now properly included in signals
            
            row = f"{i:<3} {timestamp_str:<20} {pair:<8} {type_display:<4} {'ema_modular':<15} {price:<8.5f} {confidence:<6.1%} {max_profit:<8.1f} {max_loss:<8.1f} {risk_reward:<6.2f}"
            self.logger.info(row)
        
        self.logger.info("=" * 120)
        
        if len(signals) > 20:
            self.logger.info(f"📝 Showing latest 20 of {len(signals)} total signals (newest first)")
        else:
            self.logger.info(f"📝 Showing all {len(signals)} signals (newest first)")
    
    def _analyze_performance(self, signals: List[Dict]):
        """Analyze performance metrics"""
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
            
            self.logger.info("\n📈 EMA STRATEGY PERFORMANCE:")
            self.logger.info("=" * 50)
            self.logger.info(f"   📊 Total Signals: {total_signals}")
            self.logger.info(f"   🎯 Average Confidence: {avg_confidence:.1%}")
            self.logger.info(f"   📈 Bull Signals: {bull_signals}")
            self.logger.info(f"   📉 Bear Signals: {bear_signals}")
            
            if valid_performance_signals:
                profits = [s['max_profit_pips'] for s in valid_performance_signals]
                losses = [s['max_loss_pips'] for s in valid_performance_signals]
                
                avg_profit = sum(profits) / len(profits)
                avg_loss = sum(losses) / len(losses)
                
                # Win/loss calculation using actual trade outcomes (with trailing stops)
                winners = [s for s in valid_performance_signals if s.get('trade_outcome') == 'WIN']
                losers = [s for s in valid_performance_signals if s.get('trade_outcome') == 'LOSE']
                neutral = [s for s in valid_performance_signals if s.get('trade_outcome') in ['NEUTRAL', 'TIMEOUT']]
                
                # Categorize by exit reason
                profit_target_exits = [s for s in winners if s.get('exit_reason') == 'PROFIT_TARGET']
                trailing_stop_exits = [s for s in winners + losers if s.get('exit_reason') == 'TRAILING_STOP']
                stop_loss_exits = [s for s in losers if s.get('exit_reason') == 'STOP_LOSS']
                
                closed_trades = len(winners) + len(losers)
                win_rate = len(winners) / closed_trades if closed_trades > 0 else 0
                
                self.logger.info(f"   💰 Average Profit: {avg_profit:.1f} pips")
                self.logger.info(f"   📉 Average Loss: {avg_loss:.1f} pips")
                self.logger.info(f"   🏆 Win Rate: {win_rate:.1%}")
                self.logger.info(f"   📊 Trade Outcomes:")
                self.logger.info(f"      ✅ Winners: {len(winners)} (profitable exits)")
                self.logger.info(f"      ❌ Losers: {len(losers)} (loss exits)")
                self.logger.info(f"      ⚪ Neutral/Timeout: {len(neutral)} (no clear outcome)")
                self.logger.info(f"   🎯 Exit Breakdown:")
                self.logger.info(f"      🏁 Profit Target: {len(profit_target_exits)} trades")
                self.logger.info(f"      📈 Trailing Stop: {len(trailing_stop_exits)} trades")
                self.logger.info(f"      🛑 Stop Loss: {len(stop_loss_exits)} trades")
                
                # Show trailing stop effectiveness
                trailing_wins = [s for s in trailing_stop_exits if s.get('trade_outcome') == 'WIN']
                if trailing_stop_exits:
                    trailing_effectiveness = len(trailing_wins) / len(trailing_stop_exits)
                    self.logger.info(f"   🔄 Trailing Stop Effectiveness: {trailing_effectiveness:.1%} ({len(trailing_wins)}/{len(trailing_stop_exits)})")
            else:
                self.logger.info(f"   💰 Average Profit: 0.0 pips (no valid data)")
                self.logger.info(f"   📉 Average Loss: 0.0 pips (no valid data)")
                self.logger.info(f"   🏆 Win Rate: 0.0% (no valid data)")
                
        except Exception as e:
            self.logger.error(f"⚠️ Performance analysis failed: {e}")
    
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
    
    def log_performance_summary(self):
        """Log Smart Money performance summary"""
        if hasattr(self, 'smart_money_enabled') and self.smart_money_enabled and hasattr(self, 'smart_money_stats') and self.smart_money_stats.get('signals_analyzed', 0) > 0:
            stats = self.smart_money_stats
            self.logger.info("\n🧠 SMART MONEY PERFORMANCE SUMMARY:")
            self.logger.info("=" * 50)
            self.logger.info(f"   📊 Signals analyzed: {stats.get('signals_analyzed', 0)}")
            self.logger.info(f"   ✅ Signals enhanced: {stats.get('signals_enhanced', 0)}")
            self.logger.info(f"   ❌ Analysis failures: {stats.get('analysis_failures', 0)}")
            analyzed = stats.get('signals_analyzed', 0)
            enhanced = stats.get('signals_enhanced', 0)
            if analyzed > 0:
                success_rate = enhanced / analyzed
                self.logger.info(f"   🎯 Success rate: {success_rate:.1%}")
            avg_time = stats.get('avg_analysis_time', 0.0)
            if avg_time > 0:
                self.logger.info(f"   ⏱️  Avg analysis time: {avg_time:.2f}s")
            self.logger.info("=" * 50)
        elif hasattr(self, 'smart_money_enabled') and self.smart_money_enabled:
            self.logger.info("\n🧠 Smart Money was enabled but no signals were analyzed")
    
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
            epic: Epic to analyze (e.g., CS.D.EURUSD.CEEM.IP)
            timestamp: Timestamp of the signal to validate (e.g., "2025-08-04 15:30:00")
            timeframe: Timeframe to use for analysis
            show_raw_data: Show raw OHLC data around the signal
            show_calculations: Show detailed EMA calculations
            show_decision_tree: Show the decision-making process
        """
        
        # Use default timeframe if not specified
        if timeframe is None:
            timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '15m')
            self.logger.debug(f"🔧 Using DEFAULT_TIMEFRAME from config: {timeframe}")
        
        self.logger.info("🔍 SINGLE SIGNAL VALIDATION")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Epic: {epic}")
        self.logger.info(f"⏰ Timestamp: {timestamp}")
        self.logger.info(f"📈 Timeframe: {timeframe}")
        
        try:
            # Initialize strategy first
            self.initialize_ema_strategy()
            
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
            
            # Get extended data around the target timestamp
            # CRITICAL FIX: Use same data window as backtest system for consistency
            # Default to 5 days (120 hours) to match typical backtest data window
            default_days = 5
            validation_lookback_hours = default_days * 24  # 120 hours = 5 days
            
            df = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=validation_lookback_hours  # FIXED: Now matches backtest data window
            )
            
            if df is None or df.empty:
                self.logger.error(f"❌ No data available for {epic}")
                return False
            
            # Find the closest data point to the target timestamp
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
            
            # Get the data context around this point
            data_idx = df_with_time.index.get_loc(closest_idx)
            min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)
            
            # Ensure we have enough data before the signal point for EMA calculations
            context_start = max(0, data_idx - min_bars)
            context_end = min(len(df), data_idx + 10)
            
            validation_data = df.iloc[context_start:context_end + 1].copy()
            signal_row_idx = data_idx - context_start
            
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
                
            # Show MACD values if available
            signal_row = validation_data.iloc[signal_row_idx]
            macd_histogram = signal_row.get('macd_histogram', None)
            if macd_histogram is not None:
                self.logger.info(f"\n🎯 MACD Filter Analysis:")
                self.logger.info(f"   Histogram: {macd_histogram:.6f}")
                self.logger.info(f"   Color: {'GREEN (Bullish)' if macd_histogram > 0 else 'RED (Bearish)'}")
                self.logger.info(f"   BUY Signal Valid: {'✅' if macd_histogram > 0 else '❌'}")
                self.logger.info(f"   SELL Signal Valid: {'✅' if macd_histogram < 0 else '❌'}")
            else:
                self.logger.info(f"\n🎯 MACD Filter Analysis: ❌ No MACD data available")
            
            # Attempt signal detection at this point
            signal_data_slice = validation_data.iloc[:signal_row_idx + 1].copy()
            
            # Check if we have enough data for EMA calculation
            if len(signal_data_slice) < min_bars:
                self.logger.warning(f"⚠️ Insufficient data for signal detection ({len(signal_data_slice)} < {min_bars})")
                self.logger.warning(f"   Proceeding with available data for validation only")
            
            # Detect signal using strategy with enhanced debugging
            self.logger.info("\n🎯 SIGNAL DETECTION ATTEMPT:")
            self.logger.info("-" * 40)
            
            # Convert timestamp for historical session detection
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
                    # Note: detect_signal_with_mtf doesn't accept evaluation_time parameter
                    detected_signal = self.strategy.detect_signal_with_mtf(
                        signal_data_slice, epic, config.SPREAD_PIPS, timeframe
                    )
                    detection_method = "mtf_enhanced"
                    self.logger.info("✅ Used MTF-enhanced detection")
                except Exception as e:
                    self.logger.warning(f"⚠️ MTF detection failed: {e}")
            
            # Fallback to standard detection
            if not detected_signal:
                try:
                    detected_signal = self.strategy.detect_signal(
                        signal_data_slice, epic, config.SPREAD_PIPS, timeframe, 
                        evaluation_time=historical_timestamp
                    )
                    detection_method = "standard"
                    self.logger.info("✅ Used standard detection")
                except Exception as e:
                    self.logger.error(f"❌ Standard detection failed: {e}")
            
            # SMART MONEY ENHANCEMENT: Apply Smart Money analysis to detected signal
            if detected_signal and hasattr(self, 'smart_money_enabled') and self.smart_money_enabled and hasattr(self, 'smart_money_integration'):
                self.logger.info(f"🧠 Applying Smart Money analysis to {detected_signal.get('signal_type', 'Unknown')} signal...")
                try:
                    enhanced_signal = self.smart_money_integration.enhance_signal_with_smart_money(
                        signal=detected_signal,
                        epic=epic,
                        timeframe=timeframe
                    )
                    if enhanced_signal:
                        detected_signal = enhanced_signal
                        if hasattr(self, 'smart_money_stats'):
                            self.smart_money_stats['signals_enhanced'] += 1
                        self.logger.info(f"✅ Smart Money enhancement applied successfully")
                        
                        # Display Smart Money analysis
                        if detected_signal.get('smart_money_analysis'):
                            sm_analysis = detected_signal['smart_money_analysis']
                            self.logger.info(f"🧠 Smart Money Analysis Results:")
                            self.logger.info(f"   Market Structure: {sm_analysis.get('market_structure', {}).get('trend', 'N/A')}")
                            self.logger.info(f"   Order Flow: {sm_analysis.get('order_flow', {}).get('strength', 'N/A')}")
                            self.logger.info(f"   Confluence Score: {sm_analysis.get('confluence_score', 0):.1%}")
                            if sm_analysis.get('supply_demand_zones'):
                                zones = sm_analysis['supply_demand_zones']
                                self.logger.info(f"   Supply/Demand Zones: {len(zones)} zones identified")
                    else:
                        self.logger.info(f"⚠️ Smart Money enhancement returned None, using original signal")
                    
                    if hasattr(self, 'smart_money_stats'):
                        self.smart_money_stats['signals_analyzed'] += 1
                except Exception as e:
                    self.logger.warning(f"❌ Smart Money enhancement failed: {e}")
                    if hasattr(self, 'smart_money_stats'):
                        self.smart_money_stats['analysis_failures'] += 1
            
            # Show decision tree if requested
            if show_decision_tree:
                self._show_decision_tree(validation_data, signal_row_idx, epic, timeframe, detected_signal)
            
            # Display results
            if detected_signal:
                self._display_detected_signal_details(detected_signal, detection_method)
            else:
                self.logger.info("\n❌ NO SIGNAL DETECTED")
                self.logger.info("   Possible reasons:")
                self.logger.info("   - EMA conditions not met")
                self.logger.info("   - Insufficient data for EMA calculation")
                self.logger.info("   - Price conditions not satisfied")
                self.logger.info("   - Confidence threshold not reached")
                self.logger.info("   - Recent signal deduplication")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Signal validation failed: {e}")
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
                            timestamp_str = row[col].strftime('%Y-%m-%d %H:%M:%S')
                        break
                    except:
                        continue
            
            # If still no timestamp, try using row index
            if timestamp_str == 'Unknown' and hasattr(idx, 'strftime'):
                try:
                    timestamp_str = idx.strftime('%Y-%m-%d %H:%M:%S')
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
        """Show detailed EMA calculations"""
        self.logger.info("\n🧮 DETAILED CALCULATIONS:")
        self.logger.info("-" * 50)
        
        try:
            # Get the signal row
            signal_row = df.iloc[signal_idx]
            prev_row = df.iloc[signal_idx - 1] if signal_idx > 0 else signal_row
            
            # EMA values
            ema_short = signal_row.get('ema_short', signal_row.get('ema_21', 0))
            ema_long = signal_row.get('ema_long', signal_row.get('ema_50', 0))
            ema_trend = signal_row.get('ema_trend', signal_row.get('ema_200', 0))
            current_price = signal_row.get('close', 0)
            
            # Previous values for trend analysis
            prev_ema_short = prev_row.get('ema_short', prev_row.get('ema_21', 0))
            prev_ema_long = prev_row.get('ema_long', prev_row.get('ema_50', 0))
            prev_price = prev_row.get('close', 0)
            
            self.logger.info(f"📈 EMA Calculations:")
            self.logger.info(f"   Short EMA (21): {ema_short:.5f}")
            self.logger.info(f"   Long EMA (50): {ema_long:.5f}")
            self.logger.info(f"   Trend EMA (200): {ema_trend:.5f}")
            self.logger.info(f"   Current Price: {current_price:.5f}")
            self.logger.info(f"   Previous Price: {prev_price:.5f}")
            
            self.logger.info(f"\n📊 EMA Analysis:")
            self.logger.info(f"   Price > Short EMA: {'✅' if current_price > ema_short else '❌'}")
            self.logger.info(f"   Short > Long EMA: {'✅' if ema_short > ema_long else '❌'}")
            self.logger.info(f"   Long > Trend EMA: {'✅' if ema_long > ema_trend else '❌'}")
            
            # Pip distances
            price_short_pips = abs(current_price - ema_short) * 10000
            short_long_pips = abs(ema_short - ema_long) * 10000
            long_trend_pips = abs(ema_long - ema_trend) * 10000
            
            self.logger.info(f"\n📏 Distances (pips):")
            self.logger.info(f"   Price to Short EMA: {price_short_pips:.1f}")
            self.logger.info(f"   Short to Long EMA: {short_long_pips:.1f}")
            self.logger.info(f"   Long to Trend EMA: {long_trend_pips:.1f}")
            
            # Pullback analysis
            if hasattr(self.strategy, 'pullback_tracker'):
                self.logger.info(f"\n🔄 Pullback Analysis:")
                # Try to get pullback state if available in the data
                pullback_state = signal_row.get('pullback_state', 'Unknown')
                if pullback_state != 'Unknown':
                    self.logger.info(f"   Pullback State: {pullback_state}")
                else:
                    # Calculate manually
                    if prev_price < prev_ema_short and current_price > ema_short:
                        self.logger.info(f"   Potential bullish pullback entry detected")
                    elif prev_price > prev_ema_short and current_price < ema_short:
                        self.logger.info(f"   Potential bearish pullback entry detected")
                    else:
                        self.logger.info(f"   No pullback entry pattern")
            
            # Thresholds
            try:
                min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.45)
                min_pip_separation = getattr(config, 'MIN_PIP_SEPARATION', 2.0)
                
                self.logger.info(f"\n⚙️ Strategy Thresholds:")
                self.logger.info(f"   Min Confidence: {min_confidence:.1%}")
                self.logger.info(f"   Min Pip Separation: {min_pip_separation} pips")
                self.logger.info(f"   Short-Long Separation OK: {'✅' if short_long_pips >= min_pip_separation else '❌'}")
                
            except Exception as threshold_error:
                self.logger.warning(f"⚠️ Could not retrieve thresholds: {threshold_error}")
                
        except Exception as e:
            self.logger.error(f"❌ Error showing calculations: {e}")
    
    def _show_decision_tree(self, df: pd.DataFrame, signal_idx: int, epic: str, timeframe: str, detected_signal: Dict = None):
        """Show the decision-making process step by step"""
        self.logger.info("\n🌳 DECISION TREE:")
        self.logger.info("-" * 40)
        
        try:
            signal_row = df.iloc[signal_idx]
            prev_row = df.iloc[signal_idx - 1] if signal_idx > 0 else signal_row
            
            # Get values
            current_price = signal_row.get('close', 0)
            ema_short = signal_row.get('ema_short', signal_row.get('ema_21', 0))
            ema_long = signal_row.get('ema_long', signal_row.get('ema_50', 0))
            ema_trend = signal_row.get('ema_trend', signal_row.get('ema_200', 0))
            prev_price = prev_row.get('close', 0)
            prev_ema_short = prev_row.get('ema_short', prev_row.get('ema_21', 0))
            
            # Decision steps
            self.logger.info("📝 Decision Steps:")
            
            # Step 1: Data availability
            data_ok = len(df) >= getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)
            self.logger.info(f"   1. Sufficient data ({len(df)} bars): {'✅' if data_ok else '❌'}")
            
            # Step 2: EMA calculations available
            emas_available = all([ema_short > 0, ema_long > 0, ema_trend > 0])
            self.logger.info(f"   2. EMA values calculated: {'✅' if emas_available else '❌'}")
            
            if emas_available:
                # Step 3: EMA cascade check
                cascade_bull = (current_price > ema_short and ema_short > ema_long and ema_long > ema_trend)
                cascade_bear = (current_price < ema_short and ema_short < ema_long and ema_long < ema_trend)
                cascade_ok = cascade_bull or cascade_bear
                cascade_type = 'BULL' if cascade_bull else ('BEAR' if cascade_bear else 'NONE')
                self.logger.info(f"   3. EMA cascade ({cascade_type}): {'✅' if cascade_ok else '❌'}")
                
                # Step 4: Pip separation check
                short_long_pips = abs(ema_short - ema_long) * 10000
                min_pip_separation = getattr(config, 'MIN_PIP_SEPARATION', 2.0)
                pip_separation_ok = short_long_pips >= min_pip_separation
                self.logger.info(f"   4. Pip separation ({short_long_pips:.1f} >= {min_pip_separation}): {'✅' if pip_separation_ok else '❌'}")
                
                # Step 5: Pullback detection
                pullback_bull = (prev_price < prev_ema_short and current_price > ema_short)
                pullback_bear = (prev_price > prev_ema_short and current_price < ema_short)
                pullback_detected = pullback_bull or pullback_bear
                pullback_type = 'BULL' if pullback_bull else ('BEAR' if pullback_bear else 'NONE')
                
                # Check pullback tolerance (2 pips max distance to EMA)
                distance_to_ema_pips = abs(current_price - ema_short) * 10000
                pullback_within_tolerance = distance_to_ema_pips <= 2.0
                
                self.logger.info(f"   5. Pullback entry ({pullback_type}): {'✅' if pullback_detected else '❌'}")
                self.logger.info(f"   6. Distance to EMA: {distance_to_ema_pips:.1f} pips (≤2.0): {'✅' if pullback_within_tolerance else '❌'}")
                
                # Step 7: MACD filter validation
                macd_histogram = signal_row.get('macd_histogram', 0)
                macd_valid = False
                if cascade_type == 'BULL':
                    macd_valid = macd_histogram > 0
                    self.logger.info(f"   7. MACD histogram for BUY: {macd_histogram:.6f} > 0: {'✅' if macd_valid else '❌'}")
                elif cascade_type == 'BEAR':
                    macd_valid = macd_histogram < 0
                    self.logger.info(f"   7. MACD histogram for SELL: {macd_histogram:.6f} < 0: {'✅' if macd_valid else '❌'}")
                else:
                    self.logger.info(f"   7. MACD validation: N/A (no signal direction)")
                
                # Step 8: Final signal logic
                signal_conditions_met = cascade_ok and pip_separation_ok and pullback_detected and pullback_within_tolerance and macd_valid
                expected_signal_type = cascade_type if signal_conditions_met else 'NONE'
                self.logger.info(f"   8. All conditions met: {'✅' if signal_conditions_met else '❌'}")
                
                # Compare with actual detection
                if detected_signal:
                    actual_signal_type = detected_signal.get('signal_type', 'UNKNOWN')
                    signal_matches = expected_signal_type in actual_signal_type or actual_signal_type in expected_signal_type
                    self.logger.info(f"   9. Expected vs Actual: {expected_signal_type} vs {actual_signal_type} {'✅' if signal_matches else '❌'}")
                else:
                    self.logger.info(f"   9. No signal detected (expected: {expected_signal_type})")
            
        except Exception as e:
            self.logger.error(f"❌ Error showing decision tree: {e}")
    
    def _display_detected_signal_details(self, signal: Dict, detection_method: str):
        """Display detailed information about the detected signal"""
        self.logger.info(f"\n✅ SIGNAL DETECTED")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Detection Method: {detection_method}")
        
        signal_type = signal.get('signal_type', 'Unknown')
        confidence = signal.get('confidence', signal.get('confidence_score', 0))
        price = signal.get('price', 0)
        epic = signal.get('epic', 'Unknown')
        
        if confidence > 1:
            confidence = confidence / 100.0
        
        self.logger.info(f"🎯 Signal Type: {signal_type}")
        self.logger.info(f"📈 Epic: {epic}")
        self.logger.info(f"💰 Price: {price:.5f}")
        self.logger.info(f"🎚️ Confidence: {confidence:.1%}")
        
        # Show strategy-specific details
        strategy = signal.get('strategy', 'Unknown')
        self.logger.info(f"📋 Strategy: {strategy}")
        
        # Show pullback details if available
        if 'pullback_entry' in signal:
            pullback_info = signal['pullback_entry']
            self.logger.info(f"🔄 Pullback Entry: {pullback_info}")
        
        # Show EMA details if available
        if any(key in signal for key in ['ema_short', 'ema_long', 'ema_trend']):
            self.logger.info(f"📊 EMA Values:")
            if 'ema_short' in signal:
                self.logger.info(f"   Short EMA: {signal['ema_short']:.5f}")
            if 'ema_long' in signal:
                self.logger.info(f"   Long EMA: {signal['ema_long']:.5f}")
            if 'ema_trend' in signal:
                self.logger.info(f"   Trend EMA: {signal['ema_trend']:.5f}")
        
        # Show additional metadata
        if 'timestamp' in signal:
            self.logger.info(f"⏰ Timestamp: {signal['timestamp']}")
        
        if 'stop_loss' in signal:
            self.logger.info(f"🛑 Stop Loss: {signal['stop_loss']:.5f}")
        
        if 'take_profit' in signal:
            self.logger.info(f"🎯 Take Profit: {signal['take_profit']:.5f}")


def main():
    """Main execution with enhanced implementation"""
    parser = argparse.ArgumentParser(description='Enhanced EMA Strategy Backtest with Smart Money Analysis')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--simulate-live', action='store_true',
                           help='Simulate live scanner behavior using real signal detector')
    mode_group.add_argument('--backtest', action='store_true', default=True,
                           help='Run backtest using modular strategy (default)')
    
    # Arguments
    parser.add_argument('--epic', help='Epic to test')
    parser.add_argument('--days', type=int, default=7, help='Days to test')
    parser.add_argument('--timeframe', help='Timeframe (default: from config)')
    parser.add_argument('--show-signals', action='store_true', help='Show individual signals')
    parser.add_argument('--ema-config', help='EMA configuration')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')
    parser.add_argument('--smart-money', action='store_true', help='Enable Smart Money analysis')
    parser.add_argument('--no-optimal-params', action='store_true', help='Disable database optimization (use static parameters)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    # NEW: Signal validation arguments
    parser.add_argument('--validate-signal', help='Validate a specific signal by timestamp (format: "YYYY-MM-DD HH:MM:SS")')
    parser.add_argument('--show-raw-data', action='store_true', help='Show raw OHLC data around signal (use with --validate-signal)')
    parser.add_argument('--show-calculations', action='store_true', help='Show detailed EMA calculations (use with --validate-signal)')
    parser.add_argument('--show-decision-tree', action='store_true', help='Show decision-making process (use with --validate-signal)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # NEW: Handle signal validation mode
    if args.validate_signal:
        if not args.epic:
            print("❌ ERROR: --epic is required when using --validate-signal")
            print("   Example: python backtest_ema.py --epic CS.D.EURUSD.CEEM.IP --validate-signal \"2025-08-28 19:30:00\"")
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
        backtest = EMABacktest()
        
        # Initialize Smart Money analysis if enabled
        if args.smart_money:
            backtest.initialize_smart_money_integration(enable_smart_money=True)
        
        # Run signal validation
        success = backtest.validate_single_signal(
            epic=args.epic,
            timestamp=args.validate_signal,
            timeframe=args.timeframe,
            show_raw_data=args.show_raw_data,
            show_calculations=args.show_calculations,
            show_decision_tree=args.show_decision_tree
        )
        
        sys.exit(0 if success else 1)
    
    backtest = EMABacktest()
    
    if args.simulate_live:
        success = backtest.run_live_scanner_simulation(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=True,
            enable_smart_money=args.smart_money
        )
    else:
        success = backtest.run_backtest(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=args.show_signals,
            ema_config=args.ema_config,
            min_confidence=args.min_confidence,
            enable_smart_money=args.smart_money,
            use_optimal_parameters=not args.no_optimal_params
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()