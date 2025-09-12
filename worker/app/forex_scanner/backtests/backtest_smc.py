#!/usr/bin/env python3
"""
Smart Money Concepts (SMC) Strategy Backtest
Run: python backtest_smc.py --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m

Features:
- SMC market structure analysis (BOS, ChoCH, swing points)
- Order block detection and validation
- Fair value gap identification and fills
- Supply/demand zone analysis
- Confluence-based signal generation
- Multi-timeframe confirmation
- Advanced trailing stop system
- Comprehensive performance analysis
- SMC-specific risk management
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
from core.strategies.smc_strategy_fast import SMCStrategyFast
from core.backtest.performance_analyzer import PerformanceAnalyzer
from core.backtest.signal_analyzer import SignalAnalyzer

from configdata import config as strategy_config
try:
    import config
except ImportError:
    from forex_scanner import config


class SMCBacktest:
    """SMC Strategy Backtesting with Advanced Analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger('smc_backtest')
        self.setup_logging()
        
        # Initialize components
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager, 'UTC')
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        self.strategy = None
        
        # SMC-specific tracking
        self.smc_stats = {
            'structure_breaks_detected': 0,
            'order_blocks_found': 0,
            'fair_value_gaps_found': 0,
            'confluence_signals': 0,
            'high_confluence_signals': 0,
            'avg_confluence_score': 0.0
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def initialize_smc_strategy(self, smc_config: str = None):
        """Initialize SMC strategy with configuration"""
        
        self.logger.info(f"🧠 Initializing SMC Strategy")
        if smc_config:
            self.logger.info(f"🔧 Using SMC config: {smc_config}")
        
        # Initialize with data_fetcher for multi-timeframe analysis + BACKTEST MODE
        self.strategy = SMCStrategyFast(
            smc_config_name=smc_config,
            data_fetcher=self.data_fetcher, 
            backtest_mode=True
        )
        
        self.logger.info("✅ SMC Strategy initialized")
        self.logger.info("🔥 BACKTEST MODE ENABLED: Time-based restrictions disabled")
        
        # Get SMC configuration details
        smc_config_summary = self.strategy.get_smc_analysis_summary()
        self.logger.info(f"   📊 Confluence required: {self.strategy.confluence_required}")
        self.logger.info(f"   📊 Min R:R ratio: {self.strategy.min_risk_reward}")
        self.logger.info(f"   📊 Config: {smc_config_summary.get('config_active', 'default')}")
        
        return self.strategy
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        smc_config: str = None,
        min_confidence: float = None
    ) -> bool:
        """Run SMC strategy backtest"""
        
        epic_list = [epic] if epic else config.EPIC_LIST
        
        self.logger.info("🧠 SMC STRATEGY BACKTEST")
        self.logger.info("=" * 40)
        self.logger.info(f"📊 Epic(s): {epic_list}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🎯 Show signals: {show_signals}")
        
        if min_confidence:
            original_min_conf = getattr(config, 'MIN_CONFIDENCE', 0.6)
            config.MIN_CONFIDENCE = min_confidence
            self.logger.info(f"🎚️ Min confidence: {min_confidence:.1%} (was {original_min_conf:.1%})")
        
        try:
            # Initialize strategy
            self.initialize_smc_strategy(smc_config)
            
            all_signals = []
            epic_results = {}
            
            for current_epic in epic_list:
                self.logger.info(f"\n📈 Processing {current_epic}")
                
                # Extract pair from epic
                pair = self._extract_pair_from_epic(current_epic)
                
                # Get enhanced data
                df = self.data_fetcher.get_enhanced_data(
                    epic=current_epic,
                    pair=pair,
                    timeframe=timeframe,
                    lookback_hours=days * 24
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
                
                # Run SMC backtest
                signals = self._run_smc_backtest(df, current_epic, timeframe)
                
                all_signals.extend(signals)
                epic_results[current_epic] = {'signals': len(signals)}
                
                self.logger.info(f"   🎯 SMC signals found: {len(signals)}")
            
            # Display results
            self._display_epic_results(epic_results)
            
            if all_signals:
                self.logger.info(f"\n✅ TOTAL SMC SIGNALS: {len(all_signals)}")
                
                if show_signals:
                    self._display_signals(all_signals)
                
                self._analyze_performance(all_signals)
                self._analyze_smc_specific_performance(all_signals)
                
                return True
            else:
                self.logger.warning("❌ No SMC signals found in backtest period")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ SMC backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_smc_backtest(self, df: pd.DataFrame, epic: str, timeframe: str) -> List[Dict]:
        """Run SMC backtest using the strategy"""
        signals = []
        
        # Use same minimum bars as other strategies
        min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)
        
        for i in range(min_bars, len(df)):
            try:
                # Get data up to current point (simulate real-time)
                current_data = df.iloc[:i+1].copy()
                
                # Get the current market timestamp
                current_timestamp = self._get_proper_timestamp(df.iloc[i], i)
                
                # Detect signal using SMC strategy
                signal = self.strategy.detect_signal(
                    current_data, epic, config.SPREAD_PIPS, timeframe, 
                    evaluation_time=current_timestamp
                )
                
                if signal:
                    # Log the signal detection
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"🧠 SMC SIGNAL DETECTED at market time: {current_timestamp}")
                    self.logger.info(f"   Type: {signal.get('signal_type', 'Unknown')}")
                    self.logger.info(f"   Confidence: {signal.get('confidence', 0):.1%}")
                    self.logger.info(f"   Price: {signal.get('price', 0):.5f}")
                    self.logger.info(f"   Break Type: {signal.get('break_type', 'Unknown')}")
                    self.logger.info(f"   Confluence Score: {signal.get('confluence_score', 0):.1f}")
                    self.logger.info(f"   Confluence Factors: {len(signal.get('confluence_factors', []))}")
                    
                    # Show confluence factors
                    factors = signal.get('confluence_factors', [])
                    if factors:
                        self.logger.info(f"   Factors: {', '.join(factors)}")
                    
                    # Show SMC analysis details
                    smc_analysis = signal.get('smc_analysis', {})
                    if smc_analysis:
                        if 'structure_break' in smc_analysis:
                            sb = smc_analysis['structure_break']
                            self.logger.info(f"   Structure: {sb.get('type', 'N/A')} (sig: {sb.get('significance', 0):.1%})")
                        
                        if 'order_blocks' in smc_analysis:
                            ob = smc_analysis['order_blocks']
                            self.logger.info(f"   Order Blocks: {ob.get('count', 0)} nearby")
                        
                        if 'fair_value_gaps' in smc_analysis:
                            fvg = smc_analysis['fair_value_gaps']
                            self.logger.info(f"   FVGs: {fvg.get('count', 0)} active")
                    
                    self.logger.info(f"{'='*60}")
                    
                    # Add backtest metadata
                    signal['backtest_timestamp'] = current_timestamp
                    signal['backtest_index'] = i
                    signal['candle_data'] = {
                        'open': float(df.iloc[i]['open']),
                        'high': float(df.iloc[i]['high']),
                        'low': float(df.iloc[i]['low']),
                        'close': float(df.iloc[i]['close']),
                        'timestamp': current_timestamp
                    }
                    
                    # Set timestamp fields for compatibility
                    signal['timestamp'] = current_timestamp
                    signal['market_timestamp'] = current_timestamp
                    
                    # Add performance metrics
                    enhanced_signal = self._add_performance_metrics(signal, df, i)
                    signals.append(enhanced_signal)
                    
                    # Update SMC stats
                    self._update_smc_stats(signal)
                    
                    self.logger.debug(f"📊 SMC signal at {signal['backtest_timestamp']}: "
                                    f"{signal.get('signal_type')} (conf: {signal.get('confidence', 0):.1%})")
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        sorted_signals = sorted(signals, key=lambda x: self._get_sortable_timestamp(x), reverse=True)
        
        self.logger.debug(f"📊 Generated {len(signals)} SMC signals")
        
        return sorted_signals
    
    def _update_smc_stats(self, signal: Dict):
        """Update SMC-specific statistics"""
        try:
            # Count confluence factors
            confluence_score = signal.get('confluence_score', 0.0)
            confluence_factors = signal.get('confluence_factors', [])
            
            self.smc_stats['confluence_signals'] += 1
            if confluence_score >= 3.0:
                self.smc_stats['high_confluence_signals'] += 1
            
            # Update average confluence score
            current_avg = self.smc_stats['avg_confluence_score']
            count = self.smc_stats['confluence_signals']
            self.smc_stats['avg_confluence_score'] = ((current_avg * (count - 1)) + confluence_score) / count
            
            # Count specific SMC components
            for factor in confluence_factors:
                if 'structure' in factor:
                    self.smc_stats['structure_breaks_detected'] += 1
                elif 'order_block' in factor:
                    self.smc_stats['order_blocks_found'] += 1
                elif 'fvg' in factor:
                    self.smc_stats['fair_value_gaps_found'] += 1
                    
        except Exception as e:
            self.logger.error(f"SMC stats update failed: {e}")
    
    def _add_performance_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Add performance metrics using SMC-based risk management"""
        try:
            enhanced_signal = signal.copy()
            
            entry_price = signal.get('price', df.iloc[signal_idx]['close'])
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()
            
            # Use SMC risk management levels if available
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            # Look ahead for performance (up to 96 bars for 15m = 24 hours)
            max_lookback = min(96, len(df) - signal_idx - 1)
            
            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]
                
                # Calculate max profit/loss potentials
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
                
                # SMC Risk Management Simulation
                if stop_loss and take_profit:
                    # Use SMC-defined levels
                    stop_distance_pips = abs(entry_price - stop_loss) * 10000
                    target_distance_pips = abs(take_profit - entry_price) * 10000
                else:
                    # Default SMC risk management
                    stop_distance_pips = 15  # Default stop
                    target_distance_pips = stop_distance_pips * signal.get('risk_reward_ratio', 1.5)
                
                # SMC Trailing Stop Configuration
                breakeven_trigger = 10    # Move to breakeven at 10 pips profit
                stop_to_profit_trigger = 20  # Move stop to profit at 20 pips
                stop_to_profit_level = 15    # Stop level when above trigger
                trailing_start = 25      # Start trailing after this profit level
                trailing_ratio = 0.6     # SMC trailing ratio
                
                # Initialize trade tracking
                trade_closed = False
                exit_pnl = 0.0
                exit_bar = None
                exit_reason = "TIMEOUT"
                
                # Trailing stop state
                current_stop_pips = stop_distance_pips
                best_profit_pips = 0.0
                stop_moved_to_breakeven = False
                stop_moved_to_profit = False
                
                # Simulate trade bar by bar
                for bar_idx, (_, bar) in enumerate(future_data.iterrows()):
                    if trade_closed:
                        break
                    
                    high_price = bar['high']
                    low_price = bar['low']
                    
                    if signal_type in ['BUY', 'BULL', 'LONG']:
                        current_profit_pips = (high_price - entry_price) * 10000
                        current_loss_pips = (entry_price - low_price) * 10000
                        
                        if current_profit_pips > best_profit_pips:
                            best_profit_pips = current_profit_pips
                            
                            # SMC Trailing Stop Logic
                            if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
                                current_stop_pips = 0
                                stop_moved_to_breakeven = True
                                
                            elif best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
                                current_stop_pips = -stop_to_profit_level
                                stop_moved_to_profit = True
                                
                            elif best_profit_pips > trailing_start and stop_moved_to_profit:
                                excess_profit = best_profit_pips - trailing_start
                                trailing_adjustment = excess_profit * trailing_ratio
                                current_stop_pips = -(stop_to_profit_level + trailing_adjustment)
                        
                        # Check exit conditions
                        if current_stop_pips > 0:
                            if current_loss_pips >= current_stop_pips:
                                exit_pnl = -current_stop_pips
                                exit_reason = "STOP_LOSS"
                                trade_closed = True
                                exit_bar = bar_idx
                        else:
                            profit_protection_level = abs(current_stop_pips)
                            if current_profit_pips <= profit_protection_level:
                                exit_pnl = profit_protection_level
                                exit_reason = "TRAILING_STOP"
                                trade_closed = True
                                exit_bar = bar_idx
                        
                        # Check profit target
                        if current_profit_pips >= target_distance_pips:
                            exit_pnl = target_distance_pips
                            exit_reason = "PROFIT_TARGET"
                            trade_closed = True
                            exit_bar = bar_idx
                            
                    elif signal_type in ['SELL', 'BEAR', 'SHORT']:
                        # Similar logic for short trades
                        current_profit_pips = (entry_price - low_price) * 10000
                        current_loss_pips = (high_price - entry_price) * 10000
                        
                        if current_profit_pips > best_profit_pips:
                            best_profit_pips = current_profit_pips
                            
                            if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
                                current_stop_pips = 0
                                stop_moved_to_breakeven = True
                                
                            elif best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
                                current_stop_pips = -stop_to_profit_level
                                stop_moved_to_profit = True
                                
                            elif best_profit_pips > trailing_start and stop_moved_to_profit:
                                excess_profit = best_profit_pips - trailing_start
                                trailing_adjustment = excess_profit * trailing_ratio
                                current_stop_pips = -(stop_to_profit_level + trailing_adjustment)
                        
                        if current_stop_pips > 0:
                            if current_loss_pips >= current_stop_pips:
                                exit_pnl = -current_stop_pips
                                exit_reason = "STOP_LOSS"
                                trade_closed = True
                                exit_bar = bar_idx
                        else:
                            profit_protection_level = abs(current_stop_pips)
                            if current_profit_pips <= profit_protection_level:
                                exit_pnl = profit_protection_level
                                exit_reason = "TRAILING_STOP"
                                trade_closed = True
                                exit_bar = bar_idx
                        
                        if current_profit_pips >= target_distance_pips:
                            exit_pnl = target_distance_pips
                            exit_reason = "PROFIT_TARGET"
                            trade_closed = True
                            exit_bar = bar_idx
                
                # Determine final trade outcome
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
                    # Trade timeout - realistic exit
                    if len(future_data) > 0:
                        final_price = future_data.iloc[-1]['close']
                        
                        if signal_type in ['BUY', 'BULL', 'LONG']:
                            final_exit_pnl = (final_price - entry_price) * 10000
                        else:
                            final_exit_pnl = (entry_price - final_price) * 10000
                        
                        if final_exit_pnl > 8.0:
                            trade_outcome = "WIN_TIMEOUT"
                            is_winner = True
                            is_loser = False
                            final_profit = round(final_exit_pnl, 1)
                            final_loss = 0
                        elif final_exit_pnl < -5.0:
                            trade_outcome = "LOSE_TIMEOUT"
                            is_winner = False
                            is_loser = True
                            final_profit = 0
                            final_loss = round(abs(final_exit_pnl), 1)
                        else:
                            trade_outcome = "BREAKEVEN_TIMEOUT"
                            is_winner = False
                            is_loser = False
                            final_profit = max(final_exit_pnl, 0)
                            final_loss = max(-final_exit_pnl, 0)
                    else:
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
                    'target_pips': target_distance_pips,
                    'stop_pips': stop_distance_pips,
                    'trailing_stop_used': stop_moved_to_profit or stop_moved_to_breakeven,
                    'best_profit_achieved': best_profit_pips,
                    'smc_risk_management': True
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
    
    def _get_proper_timestamp(self, df_row, row_index: int) -> str:
        """Get proper timestamp from data row (ensures UTC)"""
        try:
            # Try different timestamp sources
            for col in ['datetime_utc', 'start_time', 'timestamp', 'datetime']:
                if col in df_row and df_row[col] is not None:
                    candidate = df_row[col]
                    if isinstance(candidate, str) and candidate != 'Unknown':
                        if 'UTC' not in candidate:
                            return f"{candidate} UTC"
                        return candidate
                    elif hasattr(candidate, 'strftime'):
                        return candidate.strftime('%Y-%m-%d %H:%M:%S UTC')
            
            # Try index if available
            if hasattr(df_row, 'name') and df_row.name is not None:
                if hasattr(df_row.name, 'strftime'):
                    return df_row.name.strftime('%Y-%m-%d %H:%M:%S UTC')
            
            # Fallback
            base_time = datetime(2025, 8, 3, 0, 0, 0)
            estimated_time = base_time + timedelta(minutes=15 * row_index)
            return estimated_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            
        except Exception:
            fallback_time = datetime.utcnow() - timedelta(minutes=15 * (1000 - row_index))
            return fallback_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def _get_sortable_timestamp(self, signal: Dict) -> pd.Timestamp:
        """Get timestamp for sorting"""
        try:
            timestamp_str = signal.get('backtest_timestamp', signal.get('timestamp', ''))
            if timestamp_str and timestamp_str != 'Unknown':
                return pd.to_datetime(timestamp_str)
            
            index = signal.get('backtest_index', 0)
            base_time = pd.Timestamp('2025-08-04 00:00:00')
            return base_time + pd.Timedelta(minutes=15 * index)
            
        except Exception:
            return pd.Timestamp('1900-01-01')
    
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
        """Display individual SMC signals"""
        self.logger.info("\n🧠 INDIVIDUAL SMC SIGNALS:")
        self.logger.info("=" * 140)
        self.logger.info("#   TIMESTAMP            PAIR     TYPE BREAK_TYPE      PRICE    CONF  CONFL  PROFIT   LOSS     R:R  ")
        self.logger.info("-" * 140)
        
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
            
            break_type = signal.get('break_type', 'UNK')[:6]  # Truncate
            
            confidence = signal.get('confidence', signal.get('confidence_score', 0))
            if confidence > 1:
                confidence = confidence / 100.0
            
            confluence_score = signal.get('confluence_score', 0.0)
            
            price = signal.get('price', 0)
            max_profit = signal.get('max_profit_pips', 0)
            max_loss = signal.get('max_loss_pips', 0)
            risk_reward = signal.get('profit_loss_ratio', max_profit / max_loss if max_loss > 0 else float('inf'))
            
            row = f"{i:<3} {timestamp_str:<20} {pair:<8} {type_display:<4} {break_type:<12} {price:<8.5f} {confidence:<5.1%} {confluence_score:<6.1f} {max_profit:<8.1f} {max_loss:<8.1f} {risk_reward:<6.2f}"
            self.logger.info(row)
        
        self.logger.info("=" * 140)
        
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
            
            self.logger.info("\n📈 SMC STRATEGY PERFORMANCE:")
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
                
                # Win/loss calculation
                winners = [s for s in valid_performance_signals if s.get('trade_outcome') in ['WIN', 'WIN_TIMEOUT']]
                losers = [s for s in valid_performance_signals if s.get('trade_outcome') in ['LOSE', 'LOSE_TIMEOUT']]
                neutral = [s for s in valid_performance_signals if s.get('trade_outcome') in ['NEUTRAL', 'BREAKEVEN_TIMEOUT']]
                
                # Exit reason breakdown
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
                
                # SMC specific risk management
                smc_rm_trades = [s for s in valid_performance_signals if s.get('smc_risk_management', False)]
                if smc_rm_trades:
                    self.logger.info(f"   🧠 SMC Risk Management: {len(smc_rm_trades)} trades used SMC levels")
            else:
                self.logger.info(f"   💰 Average Profit: 0.0 pips (no valid data)")
                self.logger.info(f"   📉 Average Loss: 0.0 pips (no valid data)")
                self.logger.info(f"   🏆 Win Rate: 0.0% (no valid data)")
                
        except Exception as e:
            self.logger.error(f"⚠️ Performance analysis failed: {e}")
    
    def _analyze_smc_specific_performance(self, signals: List[Dict]):
        """Analyze SMC-specific performance metrics"""
        try:
            self.logger.info("\n🧠 SMC SPECIFIC ANALYSIS:")
            self.logger.info("=" * 50)
            
            # Confluence analysis
            high_confluence = [s for s in signals if s.get('confluence_score', 0) >= 3.0]
            medium_confluence = [s for s in signals if 2.0 <= s.get('confluence_score', 0) < 3.0]
            low_confluence = [s for s in signals if s.get('confluence_score', 0) < 2.0]
            
            self.logger.info(f"   📊 Confluence Distribution:")
            self.logger.info(f"      🔥 High (≥3.0): {len(high_confluence)} signals")
            self.logger.info(f"      🔶 Medium (2.0-3.0): {len(medium_confluence)} signals")
            self.logger.info(f"      🔸 Low (<2.0): {len(low_confluence)} signals")
            
            # Performance by confluence level
            if high_confluence:
                high_conf_winners = [s for s in high_confluence if s.get('is_winner', False)]
                high_conf_win_rate = len(high_conf_winners) / len(high_confluence)
                self.logger.info(f"      🎯 High Confluence Win Rate: {high_conf_win_rate:.1%}")
            
            # Break type analysis
            bos_signals = [s for s in signals if s.get('break_type', '') == 'BOS']
            choch_signals = [s for s in signals if s.get('break_type', '') == 'ChoCH']
            
            self.logger.info(f"   📊 Break Type Distribution:")
            self.logger.info(f"      📈 BOS (Continuation): {len(bos_signals)} signals")
            self.logger.info(f"      🔄 ChoCH (Reversal): {len(choch_signals)} signals")
            
            # Factor analysis
            all_factors = []
            for signal in signals:
                factors = signal.get('confluence_factors', [])
                all_factors.extend(factors)
            
            if all_factors:
                from collections import Counter
                factor_counts = Counter(all_factors)
                
                self.logger.info(f"   📊 Most Common Confluence Factors:")
                for factor, count in factor_counts.most_common(5):
                    self.logger.info(f"      • {factor}: {count} occurrences")
            
            # Overall SMC stats
            self.logger.info(f"\n🧠 SMC COMPONENT USAGE:")
            self.logger.info(f"   📊 Structure Breaks: {self.smc_stats['structure_breaks_detected']}")
            self.logger.info(f"   📊 Order Blocks: {self.smc_stats['order_blocks_found']}")
            self.logger.info(f"   📊 Fair Value Gaps: {self.smc_stats['fair_value_gaps_found']}")
            self.logger.info(f"   📊 Average Confluence: {self.smc_stats['avg_confluence_score']:.1f}")
            
        except Exception as e:
            self.logger.error(f"⚠️ SMC-specific analysis failed: {e}")
    
    def validate_single_signal(
        self,
        epic: str,
        timestamp: str,
        timeframe: str = None,
        show_raw_data: bool = False,
        show_smc_analysis: bool = True,
        show_confluence: bool = True
    ) -> bool:
        """Validate a single SMC signal and show detailed analysis"""
        
        if timeframe is None:
            timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '15m')
        
        self.logger.info("🧠 SMC SIGNAL VALIDATION")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Epic: {epic}")
        self.logger.info(f"⏰ Timestamp: {timestamp}")
        self.logger.info(f"📈 Timeframe: {timeframe}")
        
        try:
            # Initialize strategy
            self.initialize_smc_strategy()
            
            # Extract pair
            pair = self._extract_pair_from_epic(epic)
            
            # Parse timestamp
            try:
                target_time = pd.to_datetime(timestamp)
                if target_time.tz is not None:
                    target_time = target_time.tz_localize(None)
            except Exception as e:
                self.logger.error(f"❌ Invalid timestamp format: {timestamp}")
                return False
            
            # Get data
            df = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=5 * 24  # 5 days
            )
            
            if df is None or df.empty:
                self.logger.error(f"❌ No data available for {epic}")
                return False
            
            # Find closest timestamp
            # ... (similar timestamp finding logic as EMA backtest)
            
            # Show SMC analysis if requested
            if show_smc_analysis:
                self._show_smc_analysis_details(df, epic, timeframe)
            
            # Show confluence analysis if requested
            if show_confluence:
                self._show_confluence_analysis(df, len(df) - 1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SMC signal validation failed: {e}")
            return False
    
    def _show_smc_analysis_details(self, df: pd.DataFrame, epic: str, timeframe: str):
        """Show detailed SMC analysis"""
        self.logger.info("\n🧠 SMC ANALYSIS DETAILS:")
        self.logger.info("-" * 50)
        
        # Get strategy summary
        if hasattr(self.strategy, 'get_smc_analysis_summary'):
            summary = self.strategy.get_smc_analysis_summary()
            
            self.logger.info(f"📊 Strategy Type: {summary.get('strategy_type', 'Unknown')}")
            self.logger.info(f"📊 Confluence Required: {summary.get('confluence_required', 'N/A')}")
            self.logger.info(f"📊 Performance Optimized: {summary.get('performance_optimized', False)}")
            if 'market_structure' in summary:
                self.logger.info(f"📊 Market Structure: {summary.get('market_structure', {})}")
            if 'order_blocks' in summary:
                self.logger.info(f"📊 Order Blocks: {summary.get('order_blocks', {})}")
            if 'fair_value_gaps' in summary:
                self.logger.info(f"📊 Fair Value Gaps: {summary.get('fair_value_gaps', {})}")
    
    def _show_confluence_analysis(self, df: pd.DataFrame, current_index: int):
        """Show confluence analysis for current point"""
        self.logger.info("\n🎯 CONFLUENCE ANALYSIS:")
        self.logger.info("-" * 40)
        
        # This would show detailed confluence breakdown
        # Implementation would depend on strategy's confluence calculation
        self.logger.info("   Confluence factors analysis would be shown here")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='SMC Strategy Backtest')
    
    # Arguments
    parser.add_argument('--epic', help='Epic to test')
    parser.add_argument('--days', type=int, default=7, help='Days to test')
    parser.add_argument('--timeframe', help='Timeframe (default: from config)')
    parser.add_argument('--show-signals', action='store_true', help='Show individual signals')
    parser.add_argument('--smc-config', help='SMC configuration preset')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    # Signal validation arguments
    parser.add_argument('--validate-signal', help='Validate specific signal by timestamp')
    parser.add_argument('--show-raw-data', action='store_true', help='Show raw OHLC data')
    parser.add_argument('--show-smc-analysis', action='store_true', help='Show SMC analysis details')
    parser.add_argument('--show-confluence', action='store_true', help='Show confluence breakdown')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle signal validation
    if args.validate_signal:
        if not args.epic:
            print("❌ ERROR: --epic is required when using --validate-signal")
            sys.exit(1)
        
        print("🧠 SMC SIGNAL VALIDATION MODE")
        backtest = SMCBacktest()
        success = backtest.validate_single_signal(
            epic=args.epic,
            timestamp=args.validate_signal,
            timeframe=args.timeframe,
            show_raw_data=args.show_raw_data,
            show_smc_analysis=args.show_smc_analysis,
            show_confluence=args.show_confluence
        )
        sys.exit(0 if success else 1)
    
    # Regular backtest
    backtest = SMCBacktest()
    success = backtest.run_backtest(
        epic=args.epic,
        days=args.days,
        timeframe=args.timeframe,
        show_signals=args.show_signals,
        smc_config=args.smc_config,
        min_confidence=args.min_confidence
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()