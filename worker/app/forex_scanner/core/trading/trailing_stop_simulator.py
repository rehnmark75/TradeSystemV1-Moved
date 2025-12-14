# core/trading/trailing_stop_simulator.py
"""
Trailing Stop Simulator - Progressive 3-Stage System
Updated to match live trading system with:
- Break-Even trigger
- Stage 1: Initial profit lock
- Stage 2: Meaningful profit lock
- Stage 3: Percentage-based dynamic trailing
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class TrailingStopSimulator:
    """
    Simulates trade execution with Progressive 3-Stage trailing stop system

    Configuration loaded from pair-specific config in config.py:
    - break_even_trigger_points: Move to breakeven (e.g., 12 points)
    - stage1_trigger_points: Stage 1 activation (e.g., 16 points)
    - stage1_lock_points: Stage 1 profit lock (e.g., 4 points)
    - stage2_trigger_points: Stage 2 activation (e.g., 20 points)
    - stage2_lock_points: Stage 2 profit lock (e.g., 12 points)
    - stage3_trigger_points: Stage 3 activation (e.g., 23 points)
    - stage3_min_distance: Minimum trail distance (e.g., 4 points)
    """

    def __init__(self,
                 epic: str = None,
                 target_pips: float = 30.0,
                 initial_stop_pips: float = 20.0,
                 # Progressive 3-Stage parameters (with backward compatibility)
                 break_even_trigger: float = None,  # New parameter name
                 breakeven_trigger: float = None,   # Old parameter name (legacy)
                 stage1_trigger: float = 16.0,
                 stage1_lock: float = 4.0,
                 stage2_trigger: float = 20.0,
                 stage2_lock: float = 12.0,
                 stage3_trigger: float = 23.0,
                 stage3_min_distance: float = 4.0,
                 # Stage 2.5: MFE Protection parameters
                 mfe_protection_threshold_pct: float = 0.70,  # Trigger when profit reaches 70% of target
                 mfe_protection_decline_pct: float = 0.10,    # Trigger on 10% decline from peak
                 mfe_protection_lock_pct: float = 0.60,       # Lock 60% of MFE
                 # Legacy/system parameters (ignored for 3-stage system)
                 stop_to_profit_trigger: float = None,
                 stop_to_profit_level: float = None,
                 trailing_start: float = None,
                 trailing_ratio: float = None,
                 max_bars: int = 96,
                 time_exit_hours: float = None,  # 🔥 NEW: Time-based exit for scalping
                 use_fixed_sl_tp: bool = False,  # 🎯 NEW: Use FIXED SL/TP only (no trailing, for scalping)
                 use_atr: bool = False,
                 atr_multiplier: float = 2.0,
                 target_atr_multiplier: float = 3.0,
                 min_time_before_trailing_hours: float = 2.0,  # 🔧 NEW: Minimum time before trailing activates
                 breakeven_buffer_pips: float = 5.0,  # 🔧 NEW: Buffer above entry when moving to breakeven
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Progressive 3-Stage trailing stop simulator with MFE Protection

        Args:
            epic: Trading pair (e.g., 'CS.D.EURUSD.CEEM.IP') - loads config automatically
            target_pips: Profit target in pips
            initial_stop_pips: Initial stop loss in pips
            break_even_trigger: Move to breakeven at this profit level (new name)
            breakeven_trigger: Move to breakeven at this profit level (legacy name)
            stage1_trigger: Stage 1 activation profit level
            stage1_lock: Stage 1 profit to lock
            stage2_trigger: Stage 2 activation profit level
            stage2_lock: Stage 2 profit to lock
            stage3_trigger: Stage 3 activation profit level
            stage3_min_distance: Minimum trailing distance for Stage 3
            mfe_protection_threshold_pct: Stage 2.5 - Trigger when profit reaches this % of target (default: 0.70)
            mfe_protection_decline_pct: Stage 2.5 - Trigger on this % decline from peak (default: 0.10)
            mfe_protection_lock_pct: Stage 2.5 - Lock this % of MFE when triggered (default: 0.60)
            stop_to_profit_trigger: Legacy parameter (ignored)
            stop_to_profit_level: Legacy parameter (ignored)
            max_bars: Maximum bars to look ahead (default: 96 = 24 hours on 15m)
            use_atr: Use ATR-based dynamic stops
            atr_multiplier: ATR multiplier for stop loss
            target_atr_multiplier: ATR multiplier for profit target
            min_time_before_trailing_hours: Minimum hours before trailing stops activate (default: 2.0)
            breakeven_buffer_pips: Buffer in pips above entry when moving to breakeven (default: 5.0)
            logger: Optional logger instance
        """
        # Handle backward compatibility: prefer break_even_trigger, fallback to breakeven_trigger
        # 🔧 CRITICAL FIX: Changed default from 12.0 to 1.5R (1.5 * initial_stop_pips)
        # This prevents premature breakeven exits that were causing 46% breakeven rate
        be_trigger_default = initial_stop_pips * 1.5  # 1.5R instead of fixed 12 pips
        if break_even_trigger is not None:
            be_trigger = break_even_trigger
        elif breakeven_trigger is not None:
            be_trigger = breakeven_trigger
        else:
            be_trigger = be_trigger_default

        # Try to load pair-specific config if epic provided
        if epic:
            try:
                import sys
                sys.path.insert(0, '/app/forex_scanner')
                from config_trailing_stops import PAIR_TRAILING_CONFIGS

                pair_config = PAIR_TRAILING_CONFIGS.get(epic, {})
                if pair_config:
                    self.break_even_trigger = pair_config.get('break_even_trigger_points', be_trigger)
                    self.stage1_trigger = pair_config.get('stage1_trigger_points', stage1_trigger)
                    self.stage1_lock = pair_config.get('stage1_lock_points', stage1_lock)
                    self.stage2_trigger = pair_config.get('stage2_trigger_points', stage2_trigger)
                    self.stage2_lock = pair_config.get('stage2_lock_points', stage2_lock)
                    self.stage3_trigger = pair_config.get('stage3_trigger_points', stage3_trigger)
                    self.stage3_min_distance = pair_config.get('stage3_min_distance', stage3_min_distance)
                    # 🆕 Stage 2.5: MFE Protection config
                    self.mfe_protection_threshold_pct = pair_config.get('mfe_protection_threshold_pct', mfe_protection_threshold_pct)
                    self.mfe_protection_decline_pct = pair_config.get('mfe_protection_decline_pct', mfe_protection_decline_pct)
                    self.mfe_protection_lock_pct = pair_config.get('mfe_protection_lock_pct', mfe_protection_lock_pct)
                    if logger:
                        logger.info(f"📊 Loaded config for {epic}: BE={self.break_even_trigger}, "
                                  f"S1={self.stage1_trigger}→{self.stage1_lock}, "
                                  f"S2={self.stage2_trigger}→{self.stage2_lock}, "
                                  f"S3={self.stage3_trigger}, "
                                  f"MFE_Protection={self.mfe_protection_threshold_pct*100:.0f}%/{self.mfe_protection_decline_pct*100:.0f}%/{self.mfe_protection_lock_pct*100:.0f}%")
                else:
                    # Use defaults
                    self.break_even_trigger = be_trigger
                    self.stage1_trigger = stage1_trigger
                    self.stage1_lock = stage1_lock
                    self.stage2_trigger = stage2_trigger
                    self.stage2_lock = stage2_lock
                    self.stage3_trigger = stage3_trigger
                    self.stage3_min_distance = stage3_min_distance
            except Exception as e:
                if logger:
                    logger.warning(f"Could not load pair config for {epic}: {e}, using defaults")
                # Use defaults
                self.break_even_trigger = be_trigger
                self.stage1_trigger = stage1_trigger
                self.stage1_lock = stage1_lock
                self.stage2_trigger = stage2_trigger
                self.stage2_lock = stage2_lock
                self.stage3_trigger = stage3_trigger
                self.stage3_min_distance = stage3_min_distance
        else:
            # Use provided defaults
            self.break_even_trigger = be_trigger
            self.stage1_trigger = stage1_trigger
            self.stage1_lock = stage1_lock
            self.stage2_trigger = stage2_trigger
            self.stage2_lock = stage2_lock
            self.stage3_trigger = stage3_trigger
            self.stage3_min_distance = stage3_min_distance

        self.target_pips = target_pips
        self.initial_stop_pips = initial_stop_pips
        self.max_bars = max_bars
        self.time_exit_hours = time_exit_hours  # 🔥 NEW: Time-based exit (None = disabled)
        self.use_fixed_sl_tp = use_fixed_sl_tp  # 🎯 NEW: Fixed SL/TP mode (no trailing)
        self.use_atr = use_atr
        self.atr_multiplier = atr_multiplier
        self.target_atr_multiplier = target_atr_multiplier
        self.min_time_before_trailing_hours = min_time_before_trailing_hours  # 🔧 NEW: Minimum time before trailing
        self.breakeven_buffer_pips = breakeven_buffer_pips  # 🔧 NEW: Buffer when moving to breakeven
        # 🆕 Stage 2.5: MFE Protection parameters
        self.mfe_protection_threshold_pct = mfe_protection_threshold_pct  # Trigger when profit >= 70% of target
        self.mfe_protection_decline_pct = mfe_protection_decline_pct      # Trigger on 10% decline from peak
        self.mfe_protection_lock_pct = mfe_protection_lock_pct            # Lock 60% of MFE when triggered
        self.logger = logger or logging.getLogger(__name__)
        self.epic = epic

    def simulate_trade(self,
                      signal: Dict[str, Any],
                      df: pd.DataFrame,
                      signal_idx: int) -> Dict[str, Any]:
        """
        Simulate trade execution with Progressive 3-Stage trailing stop

        Args:
            signal: Signal dictionary with entry information
            df: DataFrame with OHLC price data
            signal_idx: Index in df where signal occurred

        Returns:
            Enhanced signal dictionary with trade outcome metrics
        """
        try:
            enhanced_signal = signal.copy()

            # Extract entry information
            entry_price = signal.get('entry_price') or signal.get('current_price') or signal.get('price') or df.iloc[signal_idx]['close']
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()
            signal_timestamp = signal.get('signal_timestamp') or signal.get('timestamp')
            signal_epic = signal.get('epic')  # Extract epic for pip calculation

            # Look ahead for performance (up to max_bars)
            max_lookback = min(self.max_bars, len(df) - signal_idx - 1)

            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]

                # Calculate ATR-based stops if enabled
                if self.use_atr:
                    atr_stops = self._calculate_atr_stops(df, signal_idx, signal)
                else:
                    atr_stops = None

                # Simulate the trade with Progressive 3-Stage system
                trade_result = self._simulate_trade_bars(
                    entry_price=entry_price,
                    signal_type=signal_type,
                    future_data=future_data,
                    atr_stops=atr_stops,
                    epic=signal_epic  # Pass epic for correct pip calculation
                )

                # Calculate holding time
                if trade_result['exit_bar'] is not None:
                    holding_time_minutes = (trade_result['exit_bar'] + 1) * 15

                    exit_timestamp = None
                    if signal_timestamp:
                        try:
                            from datetime import timedelta
                            if isinstance(signal_timestamp, str):
                                signal_timestamp = pd.to_datetime(signal_timestamp)
                            exit_timestamp = signal_timestamp + timedelta(minutes=holding_time_minutes)
                        except Exception as e:
                            self.logger.debug(f"Could not calculate exit timestamp: {e}")
                else:
                    holding_time_minutes = max_lookback * 15
                    exit_timestamp = None

                # Calculate exit price
                if trade_result['exit_bar'] is not None and trade_result['exit_bar'] < len(future_data):
                    exit_price = future_data.iloc[trade_result['exit_bar']]['close']
                else:
                    exit_price = future_data.iloc[-1]['close'] if len(future_data) > 0 else entry_price

                # Update signal with comprehensive metrics
                enhanced_signal.update({
                    # Trade outcome
                    'trade_result': trade_result['trade_outcome'],
                    'exit_reason': trade_result['exit_reason'],
                    'exit_price': exit_price,
                    'exit_timestamp': exit_timestamp,
                    'pips_gained': trade_result['final_profit'] if trade_result['is_winner'] else -trade_result['final_loss'],

                    # Display fields (used by logger)
                    'profit': round(trade_result['final_profit'], 1),
                    'loss': round(trade_result['final_loss'], 1),
                    'risk_reward': round(trade_result['final_profit'] / trade_result['final_loss'], 2) if trade_result['final_loss'] > 0 else (float('inf') if trade_result['final_profit'] > 0 else 0),

                    # Performance metrics
                    'holding_time_minutes': holding_time_minutes,
                    'max_favorable_excursion_pips': trade_result['best_profit_pips'],
                    'max_adverse_excursion_pips': trade_result.get('worst_loss_pips', 0),

                    # Trade classification
                    'is_winner': trade_result['is_winner'],
                    'is_loser': trade_result['is_loser'],

                    # Legacy fields for compatibility
                    'max_profit_pips': round(trade_result['final_profit'], 1),
                    'max_loss_pips': round(trade_result['final_loss'], 1),
                    'profit_loss_ratio': round(trade_result['final_profit'] / trade_result['final_loss'], 2) if trade_result['final_loss'] > 0 else float('inf'),
                    'entry_price': entry_price,
                    'exit_pnl': trade_result['exit_pnl'],
                    'exit_bar': trade_result['exit_bar'],

                    # Progressive 3-Stage metrics
                    'trailing_stop_used': trade_result.get('stage_reached', 0) > 0,
                    'stop_moved_to_breakeven': trade_result.get('breakeven_triggered', False),
                    'stage1_triggered': trade_result.get('stage1_triggered', False),
                    'stage2_triggered': trade_result.get('stage2_triggered', False),
                    'stage3_triggered': trade_result.get('stage3_triggered', False),
                    'stage_reached': trade_result.get('stage_reached', 0),
                    'best_profit_achieved': round(trade_result['best_profit_pips'], 1),

                    # Configuration used
                    'target_pips': self.target_pips,
                    'initial_stop_pips': self.initial_stop_pips,
                    'lookback_bars': max_lookback,
                })
            else:
                # No future data available
                enhanced_signal.update({
                    'trade_result': 'NO_DATA',
                    'exit_reason': 'NO_FUTURE_DATA',
                    'pips_gained': 0,
                    'profit': 0.0,
                    'loss': 0.0,
                    'risk_reward': 0.0,
                    'is_winner': False,
                    'is_loser': False,
                    'max_profit_pips': 0.0,
                    'max_loss_pips': 0.0,
                })

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"Error simulating trade: {e}")
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'trade_result': 'SIMULATION_ERROR',
                'exit_reason': f'Error: {str(e)}',
                'is_winner': False,
                'is_loser': False,
            })
            return enhanced_signal

    def _calculate_atr_stops(self, df: pd.DataFrame, signal_idx: int, signal: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ATR-based stop loss and profit targets"""
        try:
            # Try to get ATR from signal first
            atr_value = signal.get('atr')

            # If not in signal, try DataFrame
            if atr_value is None and 'atr' in df.columns:
                atr_value = df.iloc[signal_idx]['atr']

            # If still no ATR, calculate from recent bars
            if atr_value is None or pd.isna(atr_value):
                lookback_start = max(0, signal_idx - 14)
                recent_bars = df.iloc[lookback_start:signal_idx+1]

                if len(recent_bars) > 1:
                    tr = pd.DataFrame({
                        'hl': recent_bars['high'] - recent_bars['low'],
                        'hc': abs(recent_bars['high'] - recent_bars['close'].shift(1)),
                        'lc': abs(recent_bars['low'] - recent_bars['close'].shift(1))
                    })
                    true_range = tr.max(axis=1)
                    atr_value = true_range.mean()
                else:
                    atr_value = (df.iloc[signal_idx]['high'] - df.iloc[signal_idx]['low'])

            # Convert ATR to pips
            atr_pips = atr_value * 10000

            # Calculate dynamic stops based on ATR
            initial_stop_pips = atr_pips * self.atr_multiplier
            target_pips = atr_pips * self.target_atr_multiplier

            self.logger.debug(f"ATR-based stops: ATR={atr_pips:.1f} pips, "
                            f"stop={initial_stop_pips:.1f}, target={target_pips:.1f}")

            return {
                'initial_stop_pips': round(initial_stop_pips, 1),
                'target_pips': round(target_pips, 1),
                'atr_pips': round(atr_pips, 1)
            }

        except Exception as e:
            self.logger.warning(f"Error calculating ATR stops: {e}, using fixed stops")
            return None

    def _simulate_trade_bars(self,
                            entry_price: float,
                            signal_type: str,
                            future_data: pd.DataFrame,
                            atr_stops: Optional[Dict[str, float]] = None,
                            epic: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate trade execution bar by bar with Progressive 3-Stage trailing

        Args:
            entry_price: Entry price for the trade
            signal_type: BULL/BEAR/BUY/SELL signal type
            future_data: DataFrame with future price bars
            atr_stops: Optional ATR-based stop levels
            epic: Optional epic for pip calculation (JPY pairs need different multiplier)

        Returns:
            Dictionary with trade simulation results
        """
        # Use ATR-based stops if provided, otherwise use fixed stops
        if atr_stops:
            current_stop_pips = atr_stops['initial_stop_pips']
            target_pips = atr_stops['target_pips']
        else:
            current_stop_pips = self.initial_stop_pips
            target_pips = self.target_pips

        # Initialize trade tracking
        trade_closed = False
        exit_pnl = 0.0
        exit_bar = None
        exit_reason = "TIMEOUT"

        # Trailing stop state
        best_profit_pips = 0.0
        worst_loss_pips = 0.0
        breakeven_triggered = False
        stage1_triggered = False
        stage2_triggered = False
        stage3_triggered = False
        mfe_protection_triggered = False  # 🆕 Stage 2.5: MFE Protection
        stage_reached = 0

        # Normalize signal type
        is_long = signal_type in ['BUY', 'BULL', 'LONG']
        is_short = signal_type in ['SELL', 'BEAR', 'SHORT']

        # Determine pip multiplier based on epic (JPY pairs use 100, others use 10000)
        check_epic = epic or self.epic  # Use passed epic or instance epic
        if check_epic and 'JPY' in check_epic:
            pip_multiplier = 100  # JPY pairs: 1 pip = 0.01
        else:
            pip_multiplier = 10000  # Standard pairs: 1 pip = 0.0001

        # Simulate trade bar by bar
        self.logger.debug(f"Simulating {len(future_data)} bars, time_exit_hours={self.time_exit_hours}")

        for bar_idx, (_, bar) in enumerate(future_data.iterrows()):
            if trade_closed:
                break

            # 🔥 TIME-BASED EXIT: For scalping, close at breakeven after specified hours
            if self.time_exit_hours is not None:
                # Calculate time elapsed (assuming 5-minute bars for scalping)
                time_elapsed_hours = (bar_idx + 1) * 5 / 60  # bars * 5 minutes / 60 = hours

                if time_elapsed_hours >= self.time_exit_hours:
                    # Time exit: close at current price
                    close_price = bar['close']
                    if is_long:
                        time_exit_pnl = (close_price - entry_price) * pip_multiplier
                    else:
                        time_exit_pnl = (entry_price - close_price) * pip_multiplier

                    self.logger.debug(f"⏰ Time exit triggered at bar {bar_idx} ({time_elapsed_hours:.2f}h): P&L={time_exit_pnl:.1f} pips")

                    # If small P&L, force breakeven
                    if abs(time_exit_pnl) < 2.0:  # Less than 2 pips either way
                        trade_closed = True
                        exit_pnl = 0.0  # Force breakeven
                        exit_reason = "TIME_EXIT_BREAKEVEN"
                        exit_bar = bar_idx
                        self.logger.debug(f"⏰ Closing at breakeven (P&L < 2 pips)")
                        break
                    elif time_exit_pnl > 2.0:
                        # Small profit, take it
                        trade_closed = True
                        exit_pnl = time_exit_pnl
                        exit_reason = "TIME_EXIT_PROFIT"
                        exit_bar = bar_idx
                        self.logger.debug(f"⏰ Closing with profit: {time_exit_pnl:.1f} pips")
                        break
                    elif time_exit_pnl < -2.0:
                        # Small loss, accept it BUT cap at configured stop loss
                        trade_closed = True
                        # 🛑 CRITICAL FIX: Cap time exit losses at configured stop loss
                        capped_loss = min(abs(time_exit_pnl), current_stop_pips)
                        exit_pnl = -capped_loss
                        exit_reason = "TIME_EXIT_LOSS"
                        exit_bar = bar_idx
                        if abs(time_exit_pnl) > current_stop_pips:
                            self.logger.debug(f"⏰ Closing with CAPPED loss: Market={time_exit_pnl:.1f} pips, capped at SL={current_stop_pips:.1f} pips")
                        else:
                            self.logger.debug(f"⏰ Closing with loss: {time_exit_pnl:.1f} pips")
                        break

            high_price = bar['high']
            low_price = bar['low']

            if is_long:
                # Long trade: profit on price up, loss on price down
                current_profit_pips = (high_price - entry_price) * pip_multiplier
                current_loss_pips = (entry_price - low_price) * pip_multiplier

                # Track worst loss
                if current_loss_pips > worst_loss_pips:
                    worst_loss_pips = current_loss_pips

                # Update best profit (MFE tracking)
                if current_profit_pips > best_profit_pips:
                    best_profit_pips = current_profit_pips

                # 🎯 FIXED SL/TP MODE: Skip trailing for scalping (keep original SL/TP)
                if not self.use_fixed_sl_tp:
                    # Apply Progressive 3-Stage trailing stop logic with MFE Protection
                    # Called on every bar to check for MFE protection (profit decline from peak)
                    current_stop_pips, stage_info = self._update_progressive_trailing_stop(
                        best_profit_pips,
                        breakeven_triggered,
                        stage1_triggered,
                        stage2_triggered,
                        stage3_triggered,
                        bar_idx,  # Pass bar index for time-based checks
                        current_profit_pips  # Pass current profit for MFE protection
                    )

                    # Update stage flags
                    breakeven_triggered = stage_info['breakeven_triggered']
                    stage1_triggered = stage_info['stage1_triggered']
                    stage2_triggered = stage_info['stage2_triggered']
                    stage3_triggered = stage_info['stage3_triggered']
                    mfe_protection_triggered = stage_info.get('mfe_protection_triggered', False) or mfe_protection_triggered
                    stage_reached = max(stage_reached, stage_info['stage_reached'])

                # Check exit conditions
                trade_closed, exit_pnl, exit_reason = self._check_exit_conditions(
                    current_profit_pips,
                    current_loss_pips,
                    current_stop_pips,
                    target_pips
                )

                # 🆕 Update exit reason if MFE protection triggered
                if trade_closed and mfe_protection_triggered and exit_reason == "TRAILING_STOP":
                    exit_reason = "MFE_PROTECTION"

                if trade_closed:
                    exit_bar = bar_idx

            elif is_short:
                # Short trade: profit on price down, loss on price up
                current_profit_pips = (entry_price - low_price) * pip_multiplier
                current_loss_pips = (high_price - entry_price) * pip_multiplier

                # Track worst loss
                if current_loss_pips > worst_loss_pips:
                    worst_loss_pips = current_loss_pips

                # Update best profit (MFE tracking)
                if current_profit_pips > best_profit_pips:
                    best_profit_pips = current_profit_pips

                # 🎯 FIXED SL/TP MODE: Skip trailing for scalping (keep original SL/TP)
                if not self.use_fixed_sl_tp:
                    # Apply Progressive 3-Stage trailing stop logic with MFE Protection
                    # Called on every bar to check for MFE protection (profit decline from peak)
                    current_stop_pips, stage_info = self._update_progressive_trailing_stop(
                        best_profit_pips,
                        breakeven_triggered,
                        stage1_triggered,
                        stage2_triggered,
                        stage3_triggered,
                        bar_idx,  # Pass bar index for time-based checks
                        current_profit_pips  # Pass current profit for MFE protection
                    )

                    # Update stage flags
                    breakeven_triggered = stage_info['breakeven_triggered']
                    stage1_triggered = stage_info['stage1_triggered']
                    stage2_triggered = stage_info['stage2_triggered']
                    stage3_triggered = stage_info['stage3_triggered']
                    mfe_protection_triggered = stage_info.get('mfe_protection_triggered', False) or mfe_protection_triggered
                    stage_reached = max(stage_reached, stage_info['stage_reached'])

                # Check exit conditions
                trade_closed, exit_pnl, exit_reason = self._check_exit_conditions(
                    current_profit_pips,
                    current_loss_pips,
                    current_stop_pips,
                    target_pips
                )

                # 🆕 Update exit reason if MFE protection triggered
                if trade_closed and mfe_protection_triggered and exit_reason == "TRAILING_STOP":
                    exit_reason = "MFE_PROTECTION"

                if trade_closed:
                    exit_bar = bar_idx

        # Determine final trade outcome
        if trade_closed:
            if exit_reason == "PROFIT_TARGET":
                trade_outcome = "WIN"
                is_winner = True
                is_loser = False
                final_profit = exit_pnl
                final_loss = 0
            elif exit_reason in ["TIME_EXIT_PROFIT", "TIME_EXIT_BREAKEVEN", "TIME_EXIT_LOSS"]:
                # 🔥 TIME-BASED EXIT: Classify based on P&L
                if exit_reason == "TIME_EXIT_BREAKEVEN" or abs(exit_pnl) < 1.0:
                    trade_outcome = "BREAKEVEN_TIME_EXIT"
                    is_winner = False
                    is_loser = False
                    final_profit = max(exit_pnl, 0)
                    final_loss = max(-exit_pnl, 0)
                elif exit_pnl > 0:
                    trade_outcome = "WIN_TIME_EXIT"
                    is_winner = True
                    is_loser = False
                    final_profit = exit_pnl
                    final_loss = 0
                else:
                    trade_outcome = "LOSE_TIME_EXIT"
                    is_winner = False
                    is_loser = True
                    final_profit = 0
                    final_loss = abs(exit_pnl)
            elif exit_reason in ["STOP_LOSS", "TRAILING_STOP", "MFE_PROTECTION"]:
                # Handle all stop-based exits: regular SL, trailing stop, or MFE protection
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
                # Fallback for any unknown exit reason - classify based on P&L
                self.logger.warning(f"Unknown exit_reason '{exit_reason}', classifying by P&L")
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
            # Trade timeout
            if len(future_data) > 0:
                final_price = future_data.iloc[-1]['close']

                if is_long:
                    final_exit_pnl = (final_price - entry_price) * pip_multiplier
                else:
                    final_exit_pnl = (entry_price - final_price) * pip_multiplier

                # CRITICAL FIX: If loss exceeds stop loss, classify as STOP_LOSS not TIMEOUT
                # This prevents unlimited losses when trades time out beyond configured SL
                if final_exit_pnl < 0 and abs(final_exit_pnl) >= current_stop_pips:
                    trade_outcome = "LOSE"
                    exit_reason = "STOP_LOSS_TIMEOUT"  # Hit SL level during timeout period
                    is_loser = True
                    is_winner = False
                    final_loss = current_stop_pips  # Cap at configured stop loss
                    final_profit = 0
                    exit_pnl = -current_stop_pips
                    self.logger.debug(f"🛑 TIMEOUT with SL hit: {self.epic} Timeout P&L={final_exit_pnl:.1f} pips >= SL={current_stop_pips:.1f} pips, capping at SL")
                # Classify normal timeout outcome
                elif final_exit_pnl > 5.0:
                    trade_outcome = "WIN_TIMEOUT"
                    is_winner = True
                    is_loser = False
                    final_profit = round(final_exit_pnl, 1)
                    final_loss = 0
                elif final_exit_pnl < -3.0:
                    trade_outcome = "LOSE_TIMEOUT"
                    is_loser = True
                    is_winner = False
                    # CRITICAL FIX: Cap timeout losses at configured stop loss level
                    # Prevents 16.5 pip average losses when SL is configured at 6.0 pips
                    capped_loss = min(abs(final_exit_pnl), current_stop_pips)
                    final_loss = round(capped_loss, 1)
                    final_profit = 0
                    exit_pnl = -final_loss
                    if abs(final_exit_pnl) > current_stop_pips:
                        self.logger.debug(f"🛑 TIMEOUT loss capped: {self.epic} Market loss={abs(final_exit_pnl):.1f} pips, capped at SL={current_stop_pips:.1f} pips")
                else:
                    trade_outcome = "BREAKEVEN_TIMEOUT"
                    is_winner = False
                    is_loser = False
                    final_profit = max(final_exit_pnl, 0)
                    final_loss = max(-final_exit_pnl, 0)

                exit_pnl = final_exit_pnl
            else:
                trade_outcome = "NO_DATA"
                is_winner = False
                is_loser = False
                final_profit = 0
                final_loss = 0
                exit_pnl = 0

        return {
            'trade_outcome': trade_outcome,
            'exit_reason': exit_reason,
            'exit_bar': exit_bar,
            'exit_pnl': exit_pnl,
            'is_winner': is_winner,
            'is_loser': is_loser,
            'final_profit': final_profit,
            'final_loss': final_loss,
            'best_profit_pips': best_profit_pips,
            'worst_loss_pips': worst_loss_pips,
            'breakeven_triggered': breakeven_triggered,
            'stage1_triggered': stage1_triggered,
            'stage2_triggered': stage2_triggered,
            'stage3_triggered': stage3_triggered,
            'mfe_protection_triggered': mfe_protection_triggered,  # 🆕 Stage 2.5
            'stage_reached': stage_reached,
        }

    def _update_progressive_trailing_stop(self,
                                          best_profit_pips: float,
                                          breakeven_triggered: bool,
                                          stage1_triggered: bool,
                                          stage2_triggered: bool,
                                          stage3_triggered: bool,
                                          bar_idx: int = 0,
                                          current_profit_pips: float = None) -> Tuple[float, Dict[str, Any]]:
        """
        Update trailing stop using Progressive 3-Stage system with MFE Protection

        Stages:
        - Break-Even: Move stop to entry + buffer (small profit protection)
        - Stage 1: Lock initial profit (e.g., +4 pips)
        - Stage 2: Lock meaningful profit (e.g., +12 pips)
        - Stage 2.5: MFE Protection - Lock 60% of MFE when profit declines 10% from peak
        - Stage 3: Percentage-based dynamic trailing

        Args:
            best_profit_pips: Best profit achieved so far (MFE)
            breakeven_triggered: Whether break-even was triggered
            stage1_triggered: Whether stage 1 was triggered
            stage2_triggered: Whether stage 2 was triggered
            stage3_triggered: Whether stage 3 was triggered
            bar_idx: Current bar index (for time-based checks)
            current_profit_pips: Current profit level (for MFE protection calculation)

        Returns:
            Tuple of (current_stop_pips, stage_info_dict)
            - current_stop_pips: Negative = profit protection, Positive = risk
            - stage_info_dict: Stage trigger states
        """
        stage_info = {
            'breakeven_triggered': breakeven_triggered,
            'stage1_triggered': stage1_triggered,
            'stage2_triggered': stage2_triggered,
            'stage3_triggered': stage3_triggered,
            'mfe_protection_triggered': False,
            'stage_reached': 0
        }

        # 🔧 CRITICAL FIX: Check minimum time requirement before allowing trailing
        # Prevents premature exits within first 2 hours (8 bars on 15m timeframe)
        time_elapsed_hours = (bar_idx + 1) * 15 / 60  # bars * 15 minutes / 60 = hours
        if time_elapsed_hours < self.min_time_before_trailing_hours:
            # Too early - keep initial stop loss
            return self.initial_stop_pips, stage_info

        # 🆕 Stage 2.5: MFE Protection Rule
        # When profit reaches 70% of target AND then declines 10% from peak, lock 60% of MFE
        # This prevents giving back significant profits when momentum fades
        mfe_protection_threshold_pct = getattr(self, 'mfe_protection_threshold_pct', 0.70)
        mfe_protection_decline_pct = getattr(self, 'mfe_protection_decline_pct', 0.10)
        mfe_protection_lock_pct = getattr(self, 'mfe_protection_lock_pct', 0.60)

        mfe_protection_threshold = self.target_pips * mfe_protection_threshold_pct

        if current_profit_pips is not None and best_profit_pips >= mfe_protection_threshold:
            # Check if we're in decline (current profit < 90% of best)
            if best_profit_pips > 0:
                current_decline_pct = 1.0 - (current_profit_pips / best_profit_pips)
                if current_decline_pct >= mfe_protection_decline_pct:
                    # Trigger MFE Protection: Lock percentage of MFE
                    protected_profit = best_profit_pips * mfe_protection_lock_pct
                    stage_info['mfe_protection_triggered'] = True
                    stage_info['stage_reached'] = 2.5
                    self.logger.debug(
                        f"🛡️ MFE PROTECTION: Best={best_profit_pips:.1f}, Current={current_profit_pips:.1f}, "
                        f"Decline={current_decline_pct*100:.1f}% >= {mfe_protection_decline_pct*100:.0f}%, "
                        f"Locking {mfe_protection_lock_pct*100:.0f}% = {protected_profit:.1f} pips"
                    )
                    return -protected_profit, stage_info

        # Stage 3: Percentage-based dynamic trailing
        if best_profit_pips >= self.stage3_trigger:
            stage_info['stage3_triggered'] = True
            stage_info['stage_reached'] = 3

            # Calculate percentage-based trail distance
            if best_profit_pips >= 50:
                retracement_pct = 0.15  # 15% retracement for 50+ pips
            elif best_profit_pips >= 25:
                retracement_pct = 0.20  # 20% retracement for 25-49 pips
            else:
                retracement_pct = 0.25  # 25% retracement for 18-24 pips

            trail_distance = max(self.stage3_min_distance, best_profit_pips * retracement_pct)
            protected_profit = best_profit_pips - trail_distance

            return -protected_profit, stage_info

        # Stage 2: Lock meaningful profit
        elif best_profit_pips >= self.stage2_trigger:
            stage_info['stage2_triggered'] = True
            stage_info['stage_reached'] = 2
            return -self.stage2_lock, stage_info

        # Stage 1: Lock initial profit
        elif best_profit_pips >= self.stage1_trigger:
            stage_info['stage1_triggered'] = True
            stage_info['stage_reached'] = 1
            return -self.stage1_lock, stage_info

        # Break-Even: Move to entry + buffer for small profit protection
        # 🔧 CRITICAL FIX: Changed from 0.0 to -breakeven_buffer_pips
        # This protects a small profit (e.g., +5 pips) instead of exact breakeven
        elif best_profit_pips >= self.break_even_trigger:
            stage_info['breakeven_triggered'] = True
            return -self.breakeven_buffer_pips, stage_info

        # No stage triggered - use initial stop
        else:
            return self.initial_stop_pips, stage_info

    def _check_exit_conditions(self,
                               current_profit_pips: float,
                               current_loss_pips: float,
                               current_stop_pips: float,
                               target_pips: float) -> Tuple[bool, float, str]:
        """
        Check if trade should be exited

        Args:
            current_profit_pips: Current profit in pips
            current_loss_pips: Current loss in pips
            current_stop_pips: Current stop loss level (negative = profit protection)
            target_pips: Profit target

        Returns:
            Tuple of (trade_closed, exit_pnl, exit_reason)
        """
        # Check profit target first
        if current_profit_pips >= target_pips:
            self.logger.debug(f"✅ PROFIT_TARGET: {self.epic} Profit={current_profit_pips:.2f} pips >= Target={target_pips:.2f} pips")
            return True, target_pips, "PROFIT_TARGET"

        # Check stop loss
        if current_stop_pips > 0:
            # Traditional stop loss (risk)
            if current_loss_pips >= current_stop_pips:
                self.logger.debug(f"🛑 STOP_LOSS: {self.epic} Loss={current_loss_pips:.2f} pips >= SL={current_stop_pips:.2f} pips")
                return True, -current_stop_pips, "STOP_LOSS"
        else:
            # Profit protection stop (trailing/breakeven)
            profit_protection_level = abs(current_stop_pips)

            # Exit if profit drops below protection level OR if we go into loss
            if current_profit_pips <= profit_protection_level or current_loss_pips > 0:
                # Calculate actual exit PnL
                if current_profit_pips >= profit_protection_level:
                    exit_pnl = profit_protection_level
                else:
                    exit_pnl = -current_loss_pips if current_loss_pips > 0 else current_profit_pips

                self.logger.debug(f"🔄 TRAILING_STOP: {self.epic} Profit={current_profit_pips:.2f}, Loss={current_loss_pips:.2f}, Protection={profit_protection_level:.2f}, Exit_PnL={exit_pnl:.2f} pips")
                return True, exit_pnl, "TRAILING_STOP"

        # No exit condition met
        return False, 0.0, ""


def create_trailing_stop_simulator(epic: str = None,
                                   config: Optional[Dict[str, Any]] = None,
                                   logger: Optional[logging.Logger] = None) -> TrailingStopSimulator:
    """
    Factory function to create TrailingStopSimulator with Progressive 3-Stage system

    Args:
        epic: Trading pair (e.g., 'CS.D.EURUSD.CEEM.IP') - auto-loads config
        config: Optional configuration dictionary (overrides auto-loaded config)
        logger: Optional logger instance

    Returns:
        TrailingStopSimulator instance configured for Progressive 3-Stage system
    """
    if config:
        return TrailingStopSimulator(
            epic=epic,
            target_pips=config.get('target_pips', 30.0),
            initial_stop_pips=config.get('initial_stop_pips', 20.0),
            break_even_trigger=config.get('break_even_trigger', 12.0),
            stage1_trigger=config.get('stage1_trigger', 16.0),
            stage1_lock=config.get('stage1_lock', 4.0),
            stage2_trigger=config.get('stage2_trigger', 20.0),
            stage2_lock=config.get('stage2_lock', 12.0),
            stage3_trigger=config.get('stage3_trigger', 23.0),
            stage3_min_distance=config.get('stage3_min_distance', 4.0),
            max_bars=config.get('max_bars', 96),
            use_atr=config.get('use_atr', False),
            logger=logger
        )
    else:
        # Auto-load from epic if provided
        return TrailingStopSimulator(epic=epic, logger=logger)
