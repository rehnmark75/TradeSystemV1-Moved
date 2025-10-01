# core/trading/trailing_stop_simulator.py
"""
Trailing Stop Simulator - Trade outcome simulation with trailing stops

Simulates trade execution with:
- Initial stop loss
- Breakeven trigger
- Stop to profit trigger
- Trailing stop with configurable ratio
- Profit target

Returns realistic trade outcomes by processing tick-by-tick price movement.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class TrailingStopSimulator:
    """
    Simulates trade execution with trailing stop loss logic

    Supports both fixed and ATR-based dynamic stop loss:
    - Fixed mode: Uses fixed pip values
    - ATR mode: Stop loss = ATR * atr_multiplier (adapts to volatility)

    Configuration:
    - target_pips: Profit target (default: 15 pips, or ATR * target_atr_multiplier)
    - initial_stop_pips: Initial stop loss (default: 10 pips, or ATR * atr_multiplier)
    - breakeven_trigger: Move to breakeven at this profit (default: 8 pips or ATR * 0.8)
    - stop_to_profit_trigger: Move stop to profit at this level (default: 15 pips)
    - stop_to_profit_level: Stop level when triggered (default: 10 pips)
    - trailing_start: Start trailing after this profit (default: 15 pips)
    - trailing_ratio: Trail ratio (default: 0.5 = 1 pip per 2 pips profit)
    - use_atr: Enable ATR-based dynamic stops (default: False)
    - atr_multiplier: ATR multiplier for stop loss (default: 2.0)
    - target_atr_multiplier: ATR multiplier for profit target (default: 3.0)
    """

    def __init__(self,
                 target_pips: float = 15.0,
                 initial_stop_pips: float = 10.0,
                 breakeven_trigger: float = 8.0,
                 stop_to_profit_trigger: float = 15.0,
                 stop_to_profit_level: float = 10.0,
                 trailing_start: float = 15.0,
                 trailing_ratio: float = 0.5,
                 max_bars: int = 96,
                 use_atr: bool = False,
                 atr_multiplier: float = 2.0,
                 target_atr_multiplier: float = 3.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize trailing stop simulator

        Args:
            target_pips: Profit target in pips (fixed mode)
            initial_stop_pips: Initial stop loss in pips (fixed mode)
            breakeven_trigger: Move to breakeven at this profit level
            stop_to_profit_trigger: Move stop into profit at this level
            stop_to_profit_level: Stop level when moved to profit
            trailing_start: Start trailing after this profit level
            trailing_ratio: Trailing ratio (0.5 = trail 1 pip per 2 pips profit)
            max_bars: Maximum bars to look ahead (default: 96 = 24 hours on 15m)
            use_atr: Use ATR-based dynamic stops (default: False)
            atr_multiplier: ATR multiplier for stop loss (default: 2.0)
            target_atr_multiplier: ATR multiplier for profit target (default: 3.0)
            logger: Optional logger instance
        """
        self.target_pips = target_pips
        self.initial_stop_pips = initial_stop_pips
        self.breakeven_trigger = breakeven_trigger
        self.stop_to_profit_trigger = stop_to_profit_trigger
        self.stop_to_profit_level = stop_to_profit_level
        self.trailing_start = trailing_start
        self.trailing_ratio = trailing_ratio
        self.max_bars = max_bars
        self.use_atr = use_atr
        self.atr_multiplier = atr_multiplier
        self.target_atr_multiplier = target_atr_multiplier
        self.logger = logger or logging.getLogger(__name__)

    def simulate_trade(self,
                      signal: Dict[str, Any],
                      df: pd.DataFrame,
                      signal_idx: int) -> Dict[str, Any]:
        """
        Simulate trade execution with trailing stop

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

            # Look ahead for performance (up to max_bars)
            max_lookback = min(self.max_bars, len(df) - signal_idx - 1)

            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]

                # Calculate ATR-based stops if enabled
                if self.use_atr:
                    atr_stops = self._calculate_atr_stops(df, signal_idx, signal)
                else:
                    atr_stops = None

                # Simulate the trade
                trade_result = self._simulate_trade_bars(
                    entry_price=entry_price,
                    signal_type=signal_type,
                    future_data=future_data,
                    atr_stops=atr_stops
                )

                # Calculate holding time
                if trade_result['exit_bar'] is not None:
                    # Assuming 15m timeframe - adjust based on actual timeframe
                    holding_time_minutes = (trade_result['exit_bar'] + 1) * 15

                    # Calculate exit timestamp if we have signal timestamp
                    exit_timestamp = None
                    if signal_timestamp:
                        try:
                            import pandas as pd
                            from datetime import timedelta
                            if isinstance(signal_timestamp, str):
                                signal_timestamp = pd.to_datetime(signal_timestamp)
                            exit_timestamp = signal_timestamp + timedelta(minutes=holding_time_minutes)
                        except Exception as e:
                            self.logger.debug(f"Could not calculate exit timestamp: {e}")
                else:
                    holding_time_minutes = max_lookback * 15  # Full lookback period
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
                    'pips_gained': trade_result['final_profit'] - trade_result['final_loss'],

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

                    # Trailing stop metrics
                    'trailing_stop_used': trade_result['stop_moved_to_profit'] or trade_result['stop_moved_to_breakeven'],
                    'stop_moved_to_breakeven': trade_result['stop_moved_to_breakeven'],
                    'stop_moved_to_profit': trade_result['stop_moved_to_profit'],
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
                    'is_winner': False,
                    'is_loser': False,
                    'max_profit_pips': 0.0,
                    'max_loss_pips': 0.0,
                })

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"Error simulating trade: {e}")
            # Return signal with no simulation data
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'trade_result': 'SIMULATION_ERROR',
                'exit_reason': f'Error: {str(e)}',
                'is_winner': False,
                'is_loser': False,
            })
            return enhanced_signal

    def _calculate_atr_stops(self, df: pd.DataFrame, signal_idx: int, signal: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate ATR-based stop loss and profit targets

        Args:
            df: Full DataFrame with price data (includes signal bar)
            signal_idx: Index where signal occurred
            signal: Signal dictionary (may contain pre-calculated ATR)

        Returns:
            Dictionary with ATR-based stops: {initial_stop_pips, target_pips, breakeven_trigger}
        """
        try:
            # Try to get ATR from signal first (if strategy already calculated it)
            atr_value = signal.get('atr')

            # If not in signal, try to get from DataFrame
            if atr_value is None and 'atr' in df.columns:
                atr_value = df.iloc[signal_idx]['atr']

            # If still no ATR, calculate it from recent bars
            if atr_value is None or pd.isna(atr_value):
                # Calculate ATR from last 14 bars
                lookback_start = max(0, signal_idx - 14)
                recent_bars = df.iloc[lookback_start:signal_idx+1]

                if len(recent_bars) > 1:
                    # Calculate True Range for each bar
                    tr = pd.DataFrame({
                        'hl': recent_bars['high'] - recent_bars['low'],
                        'hc': abs(recent_bars['high'] - recent_bars['close'].shift(1)),
                        'lc': abs(recent_bars['low'] - recent_bars['close'].shift(1))
                    })
                    true_range = tr.max(axis=1)
                    atr_value = true_range.mean()
                else:
                    # Fallback: use recent volatility
                    atr_value = (df.iloc[signal_idx]['high'] - df.iloc[signal_idx]['low'])

            # Convert ATR to pips (ATR is in price units)
            atr_pips = atr_value * 10000

            # Calculate dynamic stops based on ATR
            initial_stop_pips = atr_pips * self.atr_multiplier
            target_pips = atr_pips * self.target_atr_multiplier

            # Phase 2: Structure-based stop adjustment (optional enhancement)
            try:
                # Import config to check if structure stops are enabled
                from configdata.strategies import config_momentum_strategy

                if getattr(config_momentum_strategy, 'MOMENTUM_USE_STRUCTURE_STOPS', False):
                    structure_stop_pips = self._calculate_structure_stop(df, signal_idx, signal, initial_stop_pips)
                    if structure_stop_pips is not None:
                        # Use the wider of ATR stop or structure stop (but cap at maximum)
                        max_stop = getattr(config_momentum_strategy, 'MOMENTUM_MAX_STOP_DISTANCE_PIPS', 25.0)
                        min_stop = getattr(config_momentum_strategy, 'MOMENTUM_MIN_STOP_DISTANCE_PIPS', 8.0)
                        initial_stop_pips = max(min_stop, min(max_stop, max(initial_stop_pips, structure_stop_pips)))
                        self.logger.debug(f"Structure-based stop: {structure_stop_pips:.1f} pips, "
                                        f"final stop: {initial_stop_pips:.1f} pips")
            except Exception as e:
                self.logger.debug(f"Structure stop calculation skipped: {e}")

            breakeven_trigger = initial_stop_pips * 0.8  # Move to BE at 80% of initial stop
            stop_to_profit_trigger = target_pips  # Move stop to profit at target
            stop_to_profit_level = initial_stop_pips  # Protect 1x ATR profit

            self.logger.debug(f"ATR-based stops: ATR={atr_pips:.1f} pips, "
                            f"stop={initial_stop_pips:.1f}, target={target_pips:.1f}")

            return {
                'initial_stop_pips': round(initial_stop_pips, 1),
                'target_pips': round(target_pips, 1),
                'breakeven_trigger': round(breakeven_trigger, 1),
                'stop_to_profit_trigger': round(stop_to_profit_trigger, 1),
                'stop_to_profit_level': round(stop_to_profit_level, 1),
                'atr_pips': round(atr_pips, 1)
            }

        except Exception as e:
            self.logger.warning(f"Error calculating ATR stops: {e}, using fixed stops")
            return None

    def _calculate_structure_stop(self, df: pd.DataFrame, signal_idx: int, signal: Dict[str, Any], atr_stop_pips: float) -> Optional[float]:
        """
        Phase 2: Calculate structure-based stop placement beyond recent swing points

        Args:
            df: Full DataFrame with price data
            signal_idx: Index where signal occurred
            signal: Signal dictionary
            atr_stop_pips: ATR-based stop distance for reference

        Returns:
            Structure-based stop distance in pips, or None if calculation fails
        """
        try:
            from configdata.strategies import config_momentum_strategy

            lookback_bars = getattr(config_momentum_strategy, 'MOMENTUM_STRUCTURE_LOOKBACK_BARS', 20)
            buffer_pips = getattr(config_momentum_strategy, 'MOMENTUM_STRUCTURE_BUFFER_PIPS', 2.0)

            # Get recent bars for structure analysis
            lookback_start = max(0, signal_idx - lookback_bars)
            recent_bars = df.iloc[lookback_start:signal_idx+1]

            if len(recent_bars) < 5:  # Need at least 5 bars
                return None

            # Get entry price and signal type
            entry_price = signal.get('entry_price') or signal.get('current_price') or df.iloc[signal_idx]['close']
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()

            # Find swing points based on signal direction
            if signal_type in ['BULL', 'BUY']:
                # For longs, place stop below recent swing low
                swing_low = recent_bars['low'].min()
                structure_stop_distance = (entry_price - swing_low) * 10000 + buffer_pips
            else:  # BEAR or SELL
                # For shorts, place stop above recent swing high
                swing_high = recent_bars['high'].max()
                structure_stop_distance = (swing_high - entry_price) * 10000 + buffer_pips

            self.logger.debug(f"Structure analysis: lookback={len(recent_bars)} bars, "
                            f"structure_stop={structure_stop_distance:.1f} pips, "
                            f"atr_stop={atr_stop_pips:.1f} pips")

            return structure_stop_distance

        except Exception as e:
            self.logger.debug(f"Structure stop calculation failed: {e}")
            return None

    def _simulate_trade_bars(self,
                            entry_price: float,
                            signal_type: str,
                            future_data: pd.DataFrame,
                            atr_stops: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Simulate trade execution bar by bar with trailing stop logic

        Args:
            entry_price: Entry price for the trade
            signal_type: BULL/BEAR/BUY/SELL signal type
            future_data: DataFrame with future price bars
            atr_stops: Optional ATR-based stop levels

        Returns:
            Dictionary with trade simulation results
        """
        # Use ATR-based stops if provided, otherwise use fixed stops
        if atr_stops:
            current_stop_pips = atr_stops['initial_stop_pips']
            target_pips = atr_stops['target_pips']
            breakeven_trigger = atr_stops['breakeven_trigger']
            stop_to_profit_trigger = atr_stops['stop_to_profit_trigger']
            stop_to_profit_level = atr_stops['stop_to_profit_level']
        else:
            current_stop_pips = self.initial_stop_pips
            target_pips = self.target_pips
            breakeven_trigger = self.breakeven_trigger
            stop_to_profit_trigger = self.stop_to_profit_trigger
            stop_to_profit_level = self.stop_to_profit_level

        # Initialize trade tracking
        trade_closed = False
        exit_pnl = 0.0
        exit_bar = None
        exit_reason = "TIMEOUT"

        # Trailing stop state
        best_profit_pips = 0.0
        worst_loss_pips = 0.0
        stop_moved_to_breakeven = False
        stop_moved_to_profit = False

        # Normalize signal type
        is_long = signal_type in ['BUY', 'BULL', 'LONG']
        is_short = signal_type in ['SELL', 'BEAR', 'SHORT']

        # Simulate trade bar by bar
        for bar_idx, (_, bar) in enumerate(future_data.iterrows()):
            if trade_closed:
                break

            high_price = bar['high']
            low_price = bar['low']

            if is_long:
                # Long trade: profit on price up, loss on price down
                current_profit_pips = (high_price - entry_price) * 10000
                current_loss_pips = (entry_price - low_price) * 10000

                # Track worst loss
                if current_loss_pips > worst_loss_pips:
                    worst_loss_pips = current_loss_pips

                # Update best profit and adjust trailing stop
                if current_profit_pips > best_profit_pips:
                    best_profit_pips = current_profit_pips

                    # Apply trailing stop logic
                    current_stop_pips = self._update_trailing_stop(
                        best_profit_pips,
                        stop_moved_to_breakeven,
                        stop_moved_to_profit,
                        breakeven_trigger,
                        stop_to_profit_trigger,
                        stop_to_profit_level
                    )

                    # Update flags
                    if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
                        stop_moved_to_breakeven = True
                    if best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
                        stop_moved_to_profit = True

                # Check exit conditions
                trade_closed, exit_pnl, exit_reason = self._check_exit_conditions(
                    current_profit_pips,
                    current_loss_pips,
                    current_stop_pips,
                    target_pips
                )

                if trade_closed:
                    exit_bar = bar_idx

            elif is_short:
                # Short trade: profit on price down, loss on price up
                current_profit_pips = (entry_price - low_price) * 10000
                current_loss_pips = (high_price - entry_price) * 10000

                # Track worst loss
                if current_loss_pips > worst_loss_pips:
                    worst_loss_pips = current_loss_pips

                # Update best profit and adjust trailing stop
                if current_profit_pips > best_profit_pips:
                    best_profit_pips = current_profit_pips

                    # Apply trailing stop logic
                    current_stop_pips = self._update_trailing_stop(
                        best_profit_pips,
                        stop_moved_to_breakeven,
                        stop_moved_to_profit,
                        breakeven_trigger,
                        stop_to_profit_trigger,
                        stop_to_profit_level
                    )

                    # Update flags
                    if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
                        stop_moved_to_breakeven = True
                    if best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
                        stop_moved_to_profit = True

                # Check exit conditions
                trade_closed, exit_pnl, exit_reason = self._check_exit_conditions(
                    current_profit_pips,
                    current_loss_pips,
                    current_stop_pips,
                    target_pips
                )

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
            # Trade timeout - use realistic exit at current market price
            if len(future_data) > 0:
                final_price = future_data.iloc[-1]['close']

                if is_long:
                    final_exit_pnl = (final_price - entry_price) * 10000
                else:
                    final_exit_pnl = (entry_price - final_price) * 10000

                # Classify timeout outcome
                if final_exit_pnl > 5.0:
                    trade_outcome = "WIN_TIMEOUT"
                    is_winner = True
                    is_loser = False
                    final_profit = round(final_exit_pnl, 1)
                    final_loss = 0
                elif final_exit_pnl < -3.0:
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
            'stop_moved_to_breakeven': stop_moved_to_breakeven,
            'stop_moved_to_profit': stop_moved_to_profit,
        }

    def _update_trailing_stop(self,
                             best_profit_pips: float,
                             stop_moved_to_breakeven: bool,
                             stop_moved_to_profit: bool,
                             breakeven_trigger: float,
                             stop_to_profit_trigger: float,
                             stop_to_profit_level: float) -> float:
        """
        Update trailing stop based on profit achieved

        Args:
            best_profit_pips: Best profit achieved
            stop_moved_to_breakeven: Whether stop moved to breakeven
            stop_moved_to_profit: Whether stop moved to profit
            breakeven_trigger: Profit level to move to breakeven
            stop_to_profit_trigger: Profit level to move stop to profit
            stop_to_profit_level: Profit level to protect when triggered

        Returns:
            Updated stop loss in pips (negative = profit protection)
        """
        # 1. Move to breakeven at trigger level
        if best_profit_pips >= breakeven_trigger and not stop_moved_to_breakeven:
            return 0.0  # Breakeven (no loss)

        # 2. Move stop to profit at trigger level
        elif best_profit_pips >= stop_to_profit_trigger and not stop_moved_to_profit:
            return -stop_to_profit_level  # Negative = profit protection

        # 3. Trail stop above profit trigger
        elif best_profit_pips > self.trailing_start and stop_moved_to_profit:
            excess_profit = best_profit_pips - self.trailing_start
            trailing_adjustment = excess_profit * self.trailing_ratio
            return -(stop_to_profit_level + trailing_adjustment)

        # No change - return initial stop (will be passed as current_stop_pips)
        return current_stop_pips if 'current_stop_pips' in locals() else self.initial_stop_pips

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
            current_stop_pips: Current stop loss level
            target_pips: Profit target

        Returns:
            Tuple of (trade_closed, exit_pnl, exit_reason)
        """
        # Check profit target first
        if current_profit_pips >= target_pips:
            return True, target_pips, "PROFIT_TARGET"

        # Check stop loss
        if current_stop_pips > 0:
            # Traditional stop loss (risk)
            if current_loss_pips >= current_stop_pips:
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

                return True, exit_pnl, "TRAILING_STOP"

        # No exit condition met
        return False, 0.0, ""


def create_trailing_stop_simulator(config: Optional[Dict[str, Any]] = None,
                                   logger: Optional[logging.Logger] = None) -> TrailingStopSimulator:
    """
    Factory function to create TrailingStopSimulator with optional configuration

    Args:
        config: Optional configuration dictionary
        logger: Optional logger instance

    Returns:
        TrailingStopSimulator instance
    """
    if config:
        return TrailingStopSimulator(
            target_pips=config.get('target_pips', 15.0),
            initial_stop_pips=config.get('initial_stop_pips', 10.0),
            breakeven_trigger=config.get('breakeven_trigger', 8.0),
            stop_to_profit_trigger=config.get('stop_to_profit_trigger', 15.0),
            stop_to_profit_level=config.get('stop_to_profit_level', 10.0),
            trailing_start=config.get('trailing_start', 15.0),
            trailing_ratio=config.get('trailing_ratio', 0.5),
            max_bars=config.get('max_bars', 96),
            logger=logger
        )
    else:
        return TrailingStopSimulator(logger=logger)