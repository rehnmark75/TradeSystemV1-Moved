"""
Trade Simulator

Simulates trade outcomes using future price data.
Determines if stop loss or take profit is hit first.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class TradeResult:
    """Result of a simulated trade."""
    exit_price: float
    exit_timestamp: datetime
    exit_reason: str  # 'TP_HIT', 'SL_HIT', 'TIMEOUT', 'GAP_SL', 'GAP_TP'
    trade_result: str  # 'WIN', 'LOSS', 'BREAKEVEN'
    pnl_percent: float
    pnl_amount: float
    holding_days: int
    max_favorable_excursion: float  # Best unrealized P&L during trade
    max_adverse_excursion: float    # Worst unrealized P&L during trade


class TradeSimulator:
    """
    Simulates trade outcomes using future price data.

    Features:
    - Walks forward through future candles
    - Checks if SL or TP is hit (including gap scenarios)
    - Calculates P&L and holding period
    - Tracks maximum favorable/adverse excursion
    """

    # Default settings
    DEFAULT_MAX_HOLDING_DAYS = 20  # ~1 month of trading days
    BREAKEVEN_THRESHOLD = 0.001   # 0.1% threshold for breakeven

    def __init__(
        self,
        max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
        include_commission: bool = True,
        commission_pct: float = 0.001,  # 0.1% per trade
        include_slippage: bool = True,
        slippage_pct: float = 0.001     # 0.1% slippage
    ):
        self.max_holding_days = max_holding_days
        self.include_commission = include_commission
        self.commission_pct = commission_pct
        self.include_slippage = include_slippage
        self.slippage_pct = slippage_pct
        self.logger = logging.getLogger(__name__)

    def simulate_trade(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        signal_type: str,  # 'BUY' or 'SELL'
        entry_timestamp: datetime,
        future_data: pd.DataFrame
    ) -> Optional[TradeResult]:
        """
        Simulate a trade using future price data.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            signal_type: 'BUY' or 'SELL'
            entry_timestamp: When the trade was entered
            future_data: DataFrame with future OHLCV data

        Returns:
            TradeResult with outcome, or None if simulation fails
        """
        if future_data.empty:
            self.logger.warning("No future data for trade simulation")
            return None

        # Apply slippage to entry
        if self.include_slippage:
            if signal_type == 'BUY':
                entry_price = entry_price * (1 + self.slippage_pct)
            else:
                entry_price = entry_price * (1 - self.slippage_pct)

        # Limit future data to max holding period
        sim_data = future_data.head(self.max_holding_days)

        # Initialize tracking variables
        max_favorable = 0.0
        max_adverse = 0.0
        exit_price = None
        exit_timestamp = None
        exit_reason = None
        holding_days = 0

        # Walk through each future bar
        for idx, row in sim_data.iterrows():
            holding_days += 1

            bar_open = float(row['open'])
            bar_high = float(row['high'])
            bar_low = float(row['low'])
            bar_close = float(row['close'])
            bar_timestamp = row['timestamp']

            # Check for gap scenarios on open
            if signal_type == 'BUY':
                # Check if opened below stop loss (gap down)
                if bar_open <= stop_loss_price:
                    exit_price = bar_open
                    exit_timestamp = bar_timestamp
                    exit_reason = 'GAP_SL'
                    break

                # Check if opened above take profit (gap up)
                if bar_open >= take_profit_price:
                    exit_price = bar_open
                    exit_timestamp = bar_timestamp
                    exit_reason = 'GAP_TP'
                    break

                # Check intrabar stop loss hit (low breaches SL)
                if bar_low <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_timestamp = bar_timestamp
                    exit_reason = 'SL_HIT'
                    break

                # Check intrabar take profit hit (high breaches TP)
                if bar_high >= take_profit_price:
                    exit_price = take_profit_price
                    exit_timestamp = bar_timestamp
                    exit_reason = 'TP_HIT'
                    break

                # Track excursions based on close
                unrealized_pnl_pct = ((bar_close - entry_price) / entry_price) * 100
                max_favorable = max(max_favorable, unrealized_pnl_pct)
                max_adverse = min(max_adverse, unrealized_pnl_pct)

            else:  # SELL signal (short)
                # Check if opened above stop loss (gap up)
                if bar_open >= stop_loss_price:
                    exit_price = bar_open
                    exit_timestamp = bar_timestamp
                    exit_reason = 'GAP_SL'
                    break

                # Check if opened below take profit (gap down)
                if bar_open <= take_profit_price:
                    exit_price = bar_open
                    exit_timestamp = bar_timestamp
                    exit_reason = 'GAP_TP'
                    break

                # Check intrabar stop loss hit (high breaches SL)
                if bar_high >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_timestamp = bar_timestamp
                    exit_reason = 'SL_HIT'
                    break

                # Check intrabar take profit hit (low breaches TP)
                if bar_low <= take_profit_price:
                    exit_price = take_profit_price
                    exit_timestamp = bar_timestamp
                    exit_reason = 'TP_HIT'
                    break

                # Track excursions based on close
                unrealized_pnl_pct = ((entry_price - bar_close) / entry_price) * 100
                max_favorable = max(max_favorable, unrealized_pnl_pct)
                max_adverse = min(max_adverse, unrealized_pnl_pct)

        # If no exit triggered, timeout at last bar's close
        if exit_price is None:
            last_row = sim_data.iloc[-1]
            exit_price = float(last_row['close'])
            exit_timestamp = last_row['timestamp']
            exit_reason = 'TIMEOUT'

        # Apply slippage to exit
        if self.include_slippage:
            if signal_type == 'BUY':
                exit_price = exit_price * (1 - self.slippage_pct)
            else:
                exit_price = exit_price * (1 + self.slippage_pct)

        # Calculate P&L
        if signal_type == 'BUY':
            pnl_amount = exit_price - entry_price
            pnl_percent = (pnl_amount / entry_price) * 100
        else:
            pnl_amount = entry_price - exit_price
            pnl_percent = (pnl_amount / entry_price) * 100

        # Apply commission (round trip)
        if self.include_commission:
            commission_cost = entry_price * self.commission_pct * 2  # Entry + exit
            pnl_amount -= commission_cost
            pnl_percent -= self.commission_pct * 200  # Both legs

        # Determine trade result
        if pnl_percent > self.BREAKEVEN_THRESHOLD * 100:
            trade_result = 'WIN'
        elif pnl_percent < -self.BREAKEVEN_THRESHOLD * 100:
            trade_result = 'LOSS'
        else:
            trade_result = 'BREAKEVEN'

        return TradeResult(
            exit_price=round(exit_price, 4),
            exit_timestamp=exit_timestamp,
            exit_reason=exit_reason,
            trade_result=trade_result,
            pnl_percent=round(pnl_percent, 4),
            pnl_amount=round(pnl_amount, 4),
            holding_days=holding_days,
            max_favorable_excursion=round(max_favorable, 4),
            max_adverse_excursion=round(max_adverse, 4)
        )

    def simulate_from_signal(
        self,
        signal,  # PullbackSignal or similar
        future_data: pd.DataFrame
    ) -> Optional[TradeResult]:
        """
        Convenience method to simulate from a signal object.

        Args:
            signal: Signal object with entry_price, stop_loss_price, take_profit_price, etc.
            future_data: DataFrame with future OHLCV data

        Returns:
            TradeResult with outcome
        """
        return self.simulate_trade(
            entry_price=signal.entry_price,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
            signal_type=signal.signal_type,
            entry_timestamp=signal.signal_timestamp,
            future_data=future_data
        )
