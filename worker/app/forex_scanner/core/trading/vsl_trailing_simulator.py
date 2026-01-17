# core/trading/vsl_trailing_simulator.py
"""
VSL Trailing Simulator - Dynamic Virtual Stop Loss for Scalping

Simulates the dynamic VSL trailing system for backtesting.
This mirrors the live system in dev-app/services/virtual_stop_loss_service.py

Configuration (synced with live settings Jan 17, 2026):
  Majors (EURUSD, GBPUSD, etc.):
    - Initial VSL: 5 pips
    - Breakeven trigger: +3 pips profit → lock +0.5 pip
    - Stage 1 trigger: +6 pips profit → lock +2.5 pips
    - Stage 2 trigger: +8 pips profit → lock +6 pips
    - Take Profit: 10 pips

  JPY pairs (USDJPY, EURJPY, etc.):
    - Initial VSL: 6 pips
    - Breakeven trigger: +2.5 pips profit → lock +0.5 pip
    - Stage 1 trigger: +6 pips profit → lock +2.5 pips
    - Stage 2 trigger: +8 pips profit → lock +5 pips
    - Take Profit: 10 pips

Simulation Resolution:
  - Uses 1-minute candles for trade tracking (when enabled via --scalp flag)
  - Signal generation stays on higher timeframes (15m/1H)
  - 1m resolution provides 5x more accurate SL/TP hit detection than 5m bars
  - Spread-aware threshold adjustment for realistic BE triggers

Spread Modeling (Source: ig.com/se Jan 2026):
  Uses actual IG Markets average spreads during active hours (01:00-22:00 CET):
  - EURUSD: 0.85 pips (tightest major)
  - GBPUSD: 1.40 pips
  - USDJPY: 0.94 pips
  - GBPJPY: 3.17 pips (widest tracked pair)
  - EURJPY: 1.97 pips
  Plus volatility adjustment based on bar range for realistic spread widening.

Slippage Modeling:
  Limit orders incur slippage when filled (entry at worse price than requested):
  - Base slippage: 0.2 pips (broker execution latency, queue position)
  - Volatility component: +0.01 pips per pip of bar range
  - Maximum cap: 0.8 pips (prevents unrealistic values during extreme volatility)
  Example: 5-pip bar → 0.25 pips slippage, 10-pip bar → 0.30 pips slippage
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

# =============================================================================
# IG MARKETS AVERAGE SPREADS (Source: ig.com/se Jan 2026)
# =============================================================================
# These are the average spreads during active trading hours (01:00-22:00 CET)
# Used for realistic backtest simulation instead of estimated values
#
# Format: {epic_suffix: average_spread_pips}
# epic_suffix is the pair identifier within the epic string
IG_AVERAGE_SPREADS = {
    # Major pairs - tighter spreads
    'EURUSD': 0.85,   # Min: 0.6, Avg: 0.85
    'GBPUSD': 1.40,   # Min: 0.9, Avg: 1.40
    'USDJPY': 0.94,   # Min: 0.7, Avg: 0.94
    'AUDUSD': 0.82,   # Min: 0.6, Avg: 0.82
    'USDCHF': 1.99,   # Min: 1.5, Avg: 1.99
    'USDCAD': 1.77,   # Min: 1.3, Avg: 1.77
    'NZDUSD': 2.05,   # Min: 1.8, Avg: 2.05

    # JPY crosses - wider spreads due to higher volatility
    'EURJPY': 1.97,   # Min: 1.5, Avg: 1.97
    'GBPJPY': 3.17,   # Min: 2.5, Avg: 3.17 (widest of tracked pairs)
    'AUDJPY': 1.79,   # Min: 1.3, Avg: 1.79
}

# Default spread for unknown pairs
DEFAULT_AVERAGE_SPREAD = 1.5

# =============================================================================
# SLIPPAGE CONFIGURATION
# =============================================================================
# Slippage occurs when limit orders fill at a slightly worse price than requested.
# This happens due to:
# - Fast market moves exceeding the limit price before execution
# - Broker execution latency
# - Order queue position
#
# Based on IG execution quality and typical forex conditions:
# - Normal conditions: 0.1-0.2 pips slippage
# - Volatile conditions: 0.3-0.5 pips slippage
# - News/high impact: 0.5-1.0+ pips slippage
#
# We use a conservative average of 0.2 pips base + volatility component

# Base slippage in pips (applied to all limit order fills)
BASE_SLIPPAGE_PIPS = 0.2

# Additional slippage per pip of bar range (volatility-based)
# A 5-pip bar adds 0.05 pips extra slippage, 10-pip bar adds 0.1 pips
VOLATILITY_SLIPPAGE_FACTOR = 0.01

# Maximum slippage cap (prevents unrealistic slippage during extreme volatility)
MAX_SLIPPAGE_PIPS = 0.8


def get_ig_average_spread(epic: str) -> float:
    """
    Get IG Markets average spread for an epic.

    Args:
        epic: Market epic (e.g., 'CS.D.EURUSD.CEEM.IP')

    Returns:
        Average spread in pips from IG data
    """
    # Extract pair from epic (e.g., 'CS.D.EURUSD.CEEM.IP' -> 'EURUSD')
    for pair, spread in IG_AVERAGE_SPREADS.items():
        if pair in epic:
            return spread
    return DEFAULT_AVERAGE_SPREAD


@dataclass
class VSLTradeState:
    """Tracks VSL state during trade simulation."""
    current_stage: str = "initial"      # initial, breakeven, stage1, stage2
    breakeven_triggered: bool = False
    stage1_triggered: bool = False
    stage2_triggered: bool = False      # NEW: Track Stage 2
    peak_profit_pips: float = 0.0       # Maximum favorable excursion
    current_vsl_pips: float = 3.0       # Current VSL distance from entry (negative = profit lock)
    dynamic_vsl_price: Optional[float] = None


class VSLTrailingSimulator:
    """
    Simulates dynamic VSL trailing for scalp trades in backtesting.

    Stage progression (one-way, never goes back):
    1. Initial: VSL at -5 pips (majors) or -6 pips (JPY) - starting protection
    2. Breakeven: When +3 pips reached, VSL moves to entry +0.5 pip
    3. Stage 1: When +6 pips reached, VSL locks +2.5 pips profit
    4. Stage 2: When +8 pips reached, VSL locks +6 pips profit (majors) or +5 pips (JPY)
    5. Take Profit: +10 pips

    Uses 1-minute candles for simulation when called from backtest_scanner in VSL mode,
    providing 5x better resolution than 5-minute bars for accurate SL/TP detection.
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

        # Get IG Markets average spread for this pair (from ig.com/se data)
        self.base_spread = get_ig_average_spread(epic)

        # Max bars to fetch for simulation
        # When using 1m candles: 240 bars = 4 hours (recommended for scalping)
        # When using 5m candles: 48 bars = 4 hours (legacy mode)
        # backtest_scanner uses 1m candles when VSL mode is enabled
        self.max_bars = 240  # 4 hours of 1m data for scalping

        self.logger.debug(f"[VSL SIM] Initialized for {epic}: spread={self.base_spread:.2f} pips, "
                         f"config={self.config}, dynamic={self.dynamic_enabled}")

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

            # =================================================================
            # LIMIT ORDER FILL SIMULATION
            # =================================================================
            # For limit orders with offset, we need to wait for price to reach
            # the limit entry level before starting the trade simulation.
            # This matches live trading where limit orders may not fill.
            # =================================================================
            order_type = signal.get('order_type', 'market')
            limit_offset_pips = signal.get('limit_offset_pips', 0) or signal.get('scalp_limit_offset_pips', 0)
            limit_expiry_minutes = signal.get('limit_expiry_minutes', 7)  # Default 7 min for scalp
            market_price = signal.get('market_price') or signal.get('current_price') or entry_price

            # Track slippage for reporting
            applied_slippage_pips = 0.0
            limit_entry_price = None

            if order_type == 'limit' and limit_offset_pips > 0:
                # Calculate limit entry price (offset from market price in direction of trade)
                if is_long:
                    # BUY limit: entry is ABOVE market price (momentum confirmation)
                    limit_entry_price = market_price + (limit_offset_pips / self.pip_multiplier)
                else:
                    # SELL limit: entry is BELOW market price (momentum confirmation)
                    limit_entry_price = market_price - (limit_offset_pips / self.pip_multiplier)

                # Calculate expiry in bars
                expiry_bars = max(1, int(limit_expiry_minutes / timeframe_minutes))

                # Find fill bar - when price reaches limit entry level
                fill_bar = None
                fill_bar_data = None
                for bar_idx, (_, bar) in enumerate(future_data.iterrows()):
                    if bar_idx >= expiry_bars:
                        break  # Limit order expired

                    if is_long:
                        # BUY limit: fills when price goes UP to limit entry
                        if bar['high'] >= limit_entry_price:
                            fill_bar = bar_idx
                            fill_bar_data = bar
                            break
                    else:
                        # SELL limit: fills when price goes DOWN to limit entry
                        if bar['low'] <= limit_entry_price:
                            fill_bar = bar_idx
                            fill_bar_data = bar
                            break

                # Apply slippage if order was filled
                if fill_bar is not None and fill_bar_data is not None:
                    # Calculate slippage based on bar volatility
                    # Wider bars = more likely to have slippage due to fast moves
                    bar_range_pips = (fill_bar_data['high'] - fill_bar_data['low']) * self.pip_multiplier
                    volatility_slippage = bar_range_pips * VOLATILITY_SLIPPAGE_FACTOR
                    applied_slippage_pips = min(BASE_SLIPPAGE_PIPS + volatility_slippage, MAX_SLIPPAGE_PIPS)

                    # Apply slippage - always makes entry price worse
                    # BUY: slippage means we enter HIGHER than limit price
                    # SELL: slippage means we enter LOWER than limit price
                    slippage_price_delta = applied_slippage_pips / self.pip_multiplier
                    if is_long:
                        entry_price = limit_entry_price + slippage_price_delta
                    else:
                        entry_price = limit_entry_price - slippage_price_delta

                    self.logger.debug(f"[VSL SIM] Limit fill with slippage: "
                                     f"limit={limit_entry_price:.5f}, "
                                     f"slippage={applied_slippage_pips:.2f} pips, "
                                     f"actual_entry={entry_price:.5f}")

                if fill_bar is None:
                    # Limit order not filled - return no trade result
                    self.logger.debug(f"[VSL SIM] Limit order not filled within {limit_expiry_minutes} min expiry "
                                     f"(offset={limit_offset_pips} pips, market={market_price:.5f}, limit={limit_entry_price:.5f})")
                    enhanced_signal.update({
                        'trade_result': 'breakeven',
                        'trade_outcome': 'LIMIT_NOT_FILLED',
                        'exit_reason': 'LIMIT_EXPIRED',
                        'pips_gained': 0.0,
                        'is_winner': False,
                        'is_loser': False,
                        'limit_offset_pips': limit_offset_pips,
                        'limit_entry_price': limit_entry_price,
                    })
                    return enhanced_signal

                # Limit filled - adjust future_data to start from fill bar
                self.logger.debug(f"[VSL SIM] Limit order filled at bar {fill_bar} "
                                 f"(entry={entry_price:.5f}, offset={limit_offset_pips} pips)")
                future_data = future_data.iloc[fill_bar + 1:]  # Start simulation from bar after fill

                if len(future_data) == 0:
                    enhanced_signal.update({
                        'trade_result': 'breakeven',
                        'trade_outcome': 'NO_DATA',
                        'exit_reason': 'NO_DATA_AFTER_FILL',
                        'pips_gained': 0.0,
                    })
                    return enhanced_signal

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
            if trade_outcome in ('PROFIT_TARGET', 'TP_HIT', 'TIMEOUT_WIN'):
                db_result = 'win'
            elif trade_outcome in ('VSL_STOP', 'STOP_LOSS', 'TIMEOUT_LOSS'):
                db_result = 'loss'
            elif trade_outcome in ('STAGE2_EXIT', 'STAGE2_STOP'):
                db_result = 'win'  # Stage 2 locks +4 pips (majors) or +5 pips (JPY)
            elif trade_outcome in ('BREAKEVEN', 'BREAKEVEN_EXIT', 'STAGE1_STOP', 'STAGE1_EXIT'):
                db_result = 'breakeven' if result['exit_pnl'] <= 1.0 else 'win'
            elif trade_outcome in ('TIMEOUT_BE', 'TIMEOUT'):
                db_result = 'breakeven'
            else:
                # Unknown outcome - use PnL to determine result
                if result['exit_pnl'] > 1.0:
                    db_result = 'win'
                elif result['exit_pnl'] < -1.0:
                    db_result = 'loss'
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
                'vsl_stage2_triggered': result['stage2_triggered'],
                'vsl_peak_profit_pips': result['peak_profit_pips'],
                'vsl_dynamic_enabled': self.dynamic_enabled,

                # Execution quality metrics
                'slippage_pips': applied_slippage_pips,
                'limit_entry_price': limit_entry_price,  # None for market orders
                'spread_pips': self.base_spread,  # IG average spread for this pair

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

            # Calculate current spread using IG Markets average spreads
            # Source: ig.com/se forex product information (Jan 2026)
            #
            # Base spread is the IG average spread for this pair during active hours:
            # - EURUSD: 0.85 pips, GBPUSD: 1.40 pips, USDJPY: 0.94 pips
            # - GBPJPY: 3.17 pips (widest), EURJPY: 1.97 pips, etc.
            #
            # We add a volatility component because spreads widen during:
            # - High volatility periods (news releases, market opens)
            # - Low liquidity (Asian session for non-JPY pairs)
            bar_range_pips = (high - low) * self.pip_multiplier

            # Volatility component: wider bars = wider spread
            # Scale: 10 pip bar adds ~0.3 pip to spread (reduced from 0.5)
            # This is more conservative since IG averages already include some volatility
            volatility_spread = bar_range_pips * 0.03

            # Combine base (from IG data) + volatility adjustment
            current_spread = self.base_spread + volatility_spread

            # Calculate current profit/loss with spread impact
            # In live trading:
            # - Long positions close at BID (mid - half_spread), making losses worse
            # - Short positions close at ASK (mid + half_spread), making losses worse
            half_spread = current_spread / 2

            if is_long:
                # Long: profit when price up, check low for SL, high for TP
                # Loss check includes spread penalty (we exit at BID which is lower)
                # Profit check also reduces by half spread (we take profit at BID)
                current_profit_pips = (high - entry_price) * self.pip_multiplier - half_spread
                current_loss_check = (entry_price - low) * self.pip_multiplier + half_spread
                tp_check_price = high
            else:
                # Short: profit when price down, check high for SL, low for TP
                # Loss check includes spread penalty (we exit at ASK which is higher)
                # Profit check also reduces by half spread (we cover short at ASK)
                current_profit_pips = (entry_price - low) * self.pip_multiplier - half_spread
                current_loss_check = (high - entry_price) * self.pip_multiplier + half_spread
                tp_check_price = low

            # Update peak profit (MFE) - for tracking only
            if current_profit_pips > state.peak_profit_pips:
                state.peak_profit_pips = current_profit_pips

            # ============================================================
            # CONSERVATIVE SIMULATION: Check stop loss FIRST
            # ============================================================
            # When both SL and profit targets could trigger in the same bar,
            # we assume the WORST case (SL hit first) because we don't know
            # the actual price path within the bar. This makes the backtest
            # more realistic and conservative.
            # ============================================================

            # Check for initial VSL stop FIRST (before any stage progression)
            # This is critical: if price touched our SL level, we're out regardless
            # of whether it also touched profit levels in the same bar
            if state.current_vsl_pips > 0:  # Still in initial VSL mode (not yet at BE)
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

            # Check for dynamic VSL breach (profit protection mode - BE/Stage1/Stage2 active)
            if state.current_vsl_pips < 0:
                # Negative VSL = profit protection mode (BE or Stage1/2 triggered)
                locked_profit = abs(state.current_vsl_pips)

                # Use close price to check if we dropped below locked level
                if is_long:
                    close_profit = (close - entry_price) * self.pip_multiplier
                else:
                    close_profit = (entry_price - close) * self.pip_multiplier

                # If close profit dropped to/below locked level, we exit
                if close_profit <= locked_profit:
                    if state.stage2_triggered:
                        exit_pnl = self.config['stage2_lock_pips']
                        exit_reason = "STAGE2_STOP"
                    elif state.stage1_triggered:
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

            # Only AFTER confirming we didn't hit SL, check for TP
            if current_profit_pips >= target_pips:
                exit_bar = bar_idx
                exit_pnl = target_pips
                exit_reason = "PROFIT_TARGET"
                exit_price = tp_check_price
                self.logger.debug(f"[VSL SIM] TP hit at bar {bar_idx}: +{target_pips:.1f} pips")
                break

            # Only AFTER confirming we survived this bar, update dynamic VSL stages
            # This ensures we don't "jump to BE" on a bar where we would have been stopped out
            if self.dynamic_enabled and state.current_vsl_pips > 0:
                # Use CLOSE price for stage progression (not HIGH) - more conservative
                # The close is a confirmed price level, high might have been a spike
                if is_long:
                    close_profit_for_stages = (close - entry_price) * self.pip_multiplier
                else:
                    close_profit_for_stages = (entry_price - close) * self.pip_multiplier

                self._update_dynamic_vsl(state, close_profit_for_stages, current_spread)

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
            'stage2_triggered': state.stage2_triggered,
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

        # Stage 2 check (highest priority)
        if not state.stage2_triggered and 'stage2_trigger_pips' in self.config:
            if current_profit_pips >= self.config['stage2_trigger_pips']:
                state.stage2_triggered = True
                state.current_stage = "stage2"
                state.current_vsl_pips = -self.config['stage2_lock_pips']  # Negative = profit lock

                self.logger.debug(f"[VSL SIM] → STAGE 2: Profit={current_profit_pips:.1f} pips, "
                                 f"Locking +{self.config['stage2_lock_pips']} pips")
                return

        # Stage 1 check (second highest priority)
        if not state.stage1_triggered:
            if current_profit_pips >= self.config['stage1_trigger_pips']:
                state.stage1_triggered = True
                state.current_stage = "stage1"
                state.current_vsl_pips = -self.config['stage1_lock_pips']  # Negative = profit lock

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
        elif exit_reason == "STAGE2_STOP":
            return "STAGE2_EXIT"
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
