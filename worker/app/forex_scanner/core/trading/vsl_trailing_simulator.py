# core/trading/vsl_trailing_simulator.py
"""
VSL Trailing Simulator - Dynamic Virtual Stop Loss for Scalping

Simulates the dynamic VSL trailing system for backtesting:
- Initial VSL at -3 pips (majors) or -4 pips (JPY)
- Moves to breakeven (+0.5 pip) when profit reaches +3 pips
- Moves to stage1 (+2 pips) when profit reaches +4.5 pips
- Spread-aware threshold adjustment

This mirrors the live system in dev-app/services/virtual_stop_loss_service.py
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from config_virtual_stop_backtest import (
    DYNAMIC_VSL_ENABLED,
    SPREAD_AWARE_TRIGGERS_ENABLED,
    BASELINE_SPREAD_PIPS,
    MAX_SPREAD_PENALTY_PIPS,
    get_dynamic_vsl_config,
    get_vsl_pips,
    get_pip_multiplier,
    is_dynamic_vsl_enabled,
)


@dataclass
class VSLTradeState:
    """Tracks VSL state during trade simulation."""
    current_stage: str = "initial"      # initial, breakeven, stage1
    breakeven_triggered: bool = False
    stage1_triggered: bool = False
    peak_profit_pips: float = 0.0       # Maximum favorable excursion
    current_vsl_pips: float = 3.0       # Current VSL distance from entry (negative = profit lock)
    dynamic_vsl_price: Optional[float] = None


class VSLTrailingSimulator:
    """
    Simulates dynamic VSL trailing for scalp trades in backtesting.

    Stage progression (one-way, never goes back):
    1. Initial: VSL at -3 pips (or -4 for JPY) - the starting protection
    2. Breakeven: When +3 pips reached, VSL moves to entry +0.5 pip
    3. Stage 1: When +4.5 pips reached, VSL moves to entry +2 pips
    """

    def __init__(self, epic: str, logger: Optional[logging.Logger] = None):
        """
        Initialize VSL trailing simulator.

        Args:
            epic: Market epic (e.g., 'CS.D.EURUSD.CEEM.IP')
            logger: Optional logger instance
        """
        self.epic = epic
        self.logger = logger or logging.getLogger(__name__)
        self.config = get_dynamic_vsl_config(epic)
        self.pip_multiplier = get_pip_multiplier(epic)
        self.dynamic_enabled = is_dynamic_vsl_enabled()

        self.logger.debug(f"[VSL SIM] Initialized for {epic}: config={self.config}, dynamic={self.dynamic_enabled}")

    def simulate_trade(self,
                       signal: Dict[str, Any],
                       df: pd.DataFrame,
                       signal_idx: int,
                       timeframe_minutes: int = 5) -> Dict[str, Any]:
        """
        Simulate scalp trade with dynamic VSL trailing.

        Args:
            signal: Signal dictionary with entry information
            df: DataFrame with OHLC price data
            signal_idx: Index in df where signal occurred
            timeframe_minutes: Minutes per bar (default 5 for scalping)

        Returns:
            Enhanced signal dictionary with trade outcome metrics
        """
        try:
            enhanced_signal = signal.copy()

            # Extract entry information
            entry_price = signal.get('entry_price') or signal.get('current_price') or df.iloc[signal_idx]['close']
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()
            is_long = signal_type in ['BUY', 'BULL', 'LONG']

            # Get VSL config for this pair
            initial_vsl_pips = self.config['initial_vsl_pips']
            target_pips = self.config['target_pips']

            # Calculate max bars to look ahead (4 hours for scalping)
            max_bars = int(4 * 60 / timeframe_minutes)  # 4 hours
            actual_bars = min(max_bars, len(df) - signal_idx - 1)

            if actual_bars <= 0:
                enhanced_signal.update({
                    'trade_result': 'breakeven',
                    'trade_outcome': 'NO_DATA',
                    'exit_reason': 'NO_FUTURE_DATA',
                    'pips_gained': 0.0,
                })
                return enhanced_signal

            future_data = df.iloc[signal_idx + 1:signal_idx + 1 + actual_bars]

            # Initialize trade state
            state = VSLTradeState(
                current_vsl_pips=initial_vsl_pips,
            )

            # Simulate bar by bar
            result = self._simulate_bars(
                entry_price=entry_price,
                is_long=is_long,
                future_data=future_data,
                target_pips=target_pips,
                state=state,
            )

            # Calculate holding time
            holding_time_minutes = (result['exit_bar'] + 1) * timeframe_minutes if result['exit_bar'] is not None else actual_bars * timeframe_minutes

            # Map to database-compatible result
            trade_outcome = result['trade_outcome']
            if trade_outcome in ('PROFIT_TARGET', 'TP_HIT'):
                db_result = 'win'
            elif trade_outcome in ('VSL_STOP', 'STOP_LOSS'):
                db_result = 'loss'
            elif trade_outcome in ('BREAKEVEN', 'STAGE1_STOP'):
                db_result = 'breakeven' if result['exit_pnl'] <= 1.0 else 'win'
            else:
                db_result = 'breakeven'

            enhanced_signal.update({
                'trade_result': db_result,
                'trade_outcome': trade_outcome,
                'exit_reason': result['exit_reason'],
                'exit_price': result.get('exit_price', entry_price),
                'pips_gained': result['exit_pnl'],
                'profit': max(result['exit_pnl'], 0),
                'loss': max(-result['exit_pnl'], 0),
                'holding_time_minutes': holding_time_minutes,
                'max_favorable_excursion_pips': result['peak_profit_pips'],
                'is_winner': result['exit_pnl'] > 0,
                'is_loser': result['exit_pnl'] < 0,
                'entry_price': entry_price,
                'exit_bar': result['exit_bar'],

                # VSL-specific metrics
                'vsl_stage': result['final_stage'],
                'vsl_breakeven_triggered': result['breakeven_triggered'],
                'vsl_stage1_triggered': result['stage1_triggered'],
                'vsl_peak_profit_pips': result['peak_profit_pips'],
                'vsl_dynamic_enabled': self.dynamic_enabled,

                # Config used
                'initial_vsl_pips': initial_vsl_pips,
                'target_pips': target_pips,
            })

            return enhanced_signal

        except Exception as e:
            self.logger.error(f"[VSL SIM] Error simulating trade: {e}")
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'trade_result': 'breakeven',
                'trade_outcome': 'SIMULATION_ERROR',
                'exit_reason': f'Error: {str(e)}',
                'is_winner': False,
                'is_loser': False,
            })
            return enhanced_signal

    def _simulate_bars(self,
                       entry_price: float,
                       is_long: bool,
                       future_data: pd.DataFrame,
                       target_pips: float,
                       state: VSLTradeState) -> Dict[str, Any]:
        """
        Simulate trade bar by bar with dynamic VSL.

        Args:
            entry_price: Entry price
            is_long: True for BUY, False for SELL
            future_data: DataFrame with future bars
            target_pips: Take profit target in pips
            state: VSLTradeState tracking object

        Returns:
            Dictionary with simulation results
        """
        exit_bar = None
        exit_pnl = 0.0
        exit_reason = "TIMEOUT"
        exit_price = entry_price

        for bar_idx, (_, bar) in enumerate(future_data.iterrows()):
            high = bar['high']
            low = bar['low']
            close = bar['close']

            # Calculate current spread from bar data (estimate)
            bar_spread_pips = (high - low) * self.pip_multiplier * 0.1  # Estimate ~10% of range
            current_spread = max(bar_spread_pips, 0.5)  # Minimum 0.5 pip spread

            # Calculate current profit/loss
            if is_long:
                # Long: profit when price up, check low for SL, high for TP
                current_profit_pips = (high - entry_price) * self.pip_multiplier
                current_loss_check = (entry_price - low) * self.pip_multiplier
                tp_check_price = high
                sl_check_price = low
            else:
                # Short: profit when price down, check high for SL, low for TP
                current_profit_pips = (entry_price - low) * self.pip_multiplier
                current_loss_check = (high - entry_price) * self.pip_multiplier
                tp_check_price = low
                sl_check_price = high

            # Update peak profit (MFE)
            if current_profit_pips > state.peak_profit_pips:
                state.peak_profit_pips = current_profit_pips

            # Update dynamic VSL if enabled
            if self.dynamic_enabled:
                self._update_dynamic_vsl(state, current_profit_pips, current_spread)

            # Check TP first (price went far enough in our favor)
            if current_profit_pips >= target_pips:
                exit_bar = bar_idx
                exit_pnl = target_pips
                exit_reason = "PROFIT_TARGET"
                exit_price = tp_check_price
                self.logger.debug(f"[VSL SIM] TP hit at bar {bar_idx}: +{target_pips:.1f} pips")
                break

            # Check for dynamic VSL breach (profit protection mode)
            if state.current_vsl_pips < 0:
                # Negative VSL = profit protection mode (BE or Stage1 triggered)
                # The absolute value is the locked profit level
                locked_profit = abs(state.current_vsl_pips)

                # Check if current profit dropped below locked level
                mid_profit = (current_profit_pips + ((entry_price - close) * self.pip_multiplier if is_long else (close - entry_price) * self.pip_multiplier)) / 2

                # Use close price to estimate where we'd exit
                if is_long:
                    close_profit = (close - entry_price) * self.pip_multiplier
                else:
                    close_profit = (entry_price - close) * self.pip_multiplier

                # If close profit dropped to/below locked level, we exit
                if close_profit <= locked_profit:
                    if state.stage1_triggered:
                        exit_pnl = self.config['stage1_lock_pips']
                        exit_reason = "STAGE1_STOP"
                    elif state.breakeven_triggered:
                        exit_pnl = self.config['breakeven_lock_pips']
                        exit_reason = "BREAKEVEN"
                    else:
                        exit_pnl = locked_profit
                        exit_reason = "VSL_STOP"

                    exit_bar = bar_idx
                    if is_long:
                        exit_price = entry_price + (exit_pnl / self.pip_multiplier)
                    else:
                        exit_price = entry_price - (exit_pnl / self.pip_multiplier)

                    self.logger.debug(f"[VSL SIM] Dynamic VSL hit at bar {bar_idx}: "
                                     f"stage={state.current_stage}, close_profit={close_profit:.1f}, "
                                     f"locked={locked_profit:.1f}, exit={exit_pnl:.1f} pips")
                    break
            else:
                # Initial VSL mode - check for stop loss
                if current_loss_check >= state.current_vsl_pips:
                    exit_bar = bar_idx
                    exit_pnl = -state.current_vsl_pips
                    exit_reason = "VSL_STOP"
                    if is_long:
                        exit_price = entry_price - (state.current_vsl_pips / self.pip_multiplier)
                    else:
                        exit_price = entry_price + (state.current_vsl_pips / self.pip_multiplier)
                    self.logger.debug(f"[VSL SIM] Initial VSL hit at bar {bar_idx}: -{state.current_vsl_pips:.1f} pips")
                    break

        # Handle timeout - use last bar's close
        if exit_bar is None and len(future_data) > 0:
            last_close = future_data.iloc[-1]['close']
            if is_long:
                exit_pnl = (last_close - entry_price) * self.pip_multiplier
            else:
                exit_pnl = (entry_price - last_close) * self.pip_multiplier
            exit_price = last_close
            exit_reason = "TIMEOUT"
            exit_bar = len(future_data) - 1

        return {
            'exit_bar': exit_bar,
            'exit_pnl': round(exit_pnl, 2),
            'exit_reason': exit_reason,
            'exit_price': exit_price,
            'trade_outcome': self._determine_outcome(exit_reason, exit_pnl),
            'peak_profit_pips': round(state.peak_profit_pips, 2),
            'final_stage': state.current_stage,
            'breakeven_triggered': state.breakeven_triggered,
            'stage1_triggered': state.stage1_triggered,
        }

    def _update_dynamic_vsl(self, state: VSLTradeState, current_profit_pips: float, current_spread: float) -> None:
        """
        Update VSL level based on current profit - mirrors live system logic.

        Args:
            state: VSLTradeState to update
            current_profit_pips: Current profit in pips
            current_spread: Current spread in pips (for spread-aware adjustment)
        """
        # Get spread-adjusted breakeven trigger
        effective_be_trigger = self._get_effective_be_trigger(current_spread)

        # Stage 1 check (highest priority)
        if not state.stage1_triggered:
            if current_profit_pips >= self.config['stage1_trigger_pips']:
                state.stage1_triggered = True
                state.current_stage = "stage1"
                state.current_vsl_pips = -self.config['stage1_lock_pips']  # Negative = profit lock

                # Calculate dynamic VSL price (entry + lock distance)
                # Note: We set this but actual breach check uses the pips value
                self.logger.debug(f"[VSL SIM] → STAGE 1: Profit={current_profit_pips:.1f} pips, "
                                 f"Locking +{self.config['stage1_lock_pips']} pips")
                return

        # Breakeven check (only if stage1 not triggered)
        if not state.breakeven_triggered:
            if current_profit_pips >= effective_be_trigger:
                state.breakeven_triggered = True
                state.current_stage = "breakeven"
                state.current_vsl_pips = -self.config['breakeven_lock_pips']  # Negative = profit lock

                self.logger.debug(f"[VSL SIM] → BREAKEVEN: Profit={current_profit_pips:.1f} pips "
                                 f"(trigger={effective_be_trigger:.1f}), "
                                 f"Locking +{self.config['breakeven_lock_pips']} pips")

    def _get_effective_be_trigger(self, current_spread: float) -> float:
        """
        Get spread-adjusted breakeven trigger.

        Args:
            current_spread: Current spread in pips

        Returns:
            Effective breakeven trigger in pips
        """
        base_trigger = self.config['breakeven_trigger_pips']

        if not SPREAD_AWARE_TRIGGERS_ENABLED:
            return base_trigger

        if current_spread > BASELINE_SPREAD_PIPS:
            spread_penalty = min(
                current_spread - BASELINE_SPREAD_PIPS,
                MAX_SPREAD_PENALTY_PIPS
            )
            return base_trigger + spread_penalty

        return base_trigger

    def _determine_outcome(self, exit_reason: str, exit_pnl: float) -> str:
        """Determine trade outcome string."""
        if exit_reason == "PROFIT_TARGET":
            return "TP_HIT"
        elif exit_reason == "VSL_STOP":
            if exit_pnl < 0:
                return "VSL_STOP"
            else:
                return "VSL_PROFIT"  # VSL hit but in profit (shouldn't happen normally)
        elif exit_reason == "BREAKEVEN":
            return "BREAKEVEN_EXIT"
        elif exit_reason == "STAGE1_STOP":
            return "STAGE1_EXIT"
        elif exit_reason == "TIMEOUT":
            if exit_pnl > 1.0:
                return "TIMEOUT_WIN"
            elif exit_pnl < -1.0:
                return "TIMEOUT_LOSS"
            else:
                return "TIMEOUT_BE"
        return exit_reason  # Return the exit reason if no match


def create_vsl_trailing_simulator(epic: str,
                                   logger: Optional[logging.Logger] = None) -> VSLTrailingSimulator:
    """
    Factory function to create VSLTrailingSimulator.

    Args:
        epic: Trading pair (e.g., 'CS.D.EURUSD.CEEM.IP')
        logger: Optional logger instance

    Returns:
        VSLTrailingSimulator instance
    """
    return VSLTrailingSimulator(epic=epic, logger=logger)
