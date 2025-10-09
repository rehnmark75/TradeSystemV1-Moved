"""
MACD Trade Manager

Handles complete trade lifecycle exactly as it would happen in live trading:
1. Signal generation and validation
2. Trade entry with realistic slippage
3. Stop loss and take profit management
4. Trade exit conditions
5. Performance tracking and reporting

This ensures backtest results accurately represent live trading performance.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TradeStatus(Enum):
    """Trade status enumeration"""
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    CLOSED_NEUTRAL = "closed_neutral"


class ExitReason(Enum):
    """Trade exit reason enumeration"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIMEOUT = "timeout"
    SIGNAL_REVERSAL = "signal_reversal"
    MANUAL_CLOSE = "manual_close"


@dataclass
class Trade:
    """Trade data structure"""
    id: int
    epic: str
    signal_type: str  # 'BULL' or 'BEAR'
    entry_time: datetime
    entry_price: float
    entry_confidence: float
    stop_loss: float
    take_profit: float
    position_size: float
    status: TradeStatus
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    profit_loss: Optional[float] = None
    profit_loss_pips: Optional[float] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    bars_held: int = 0


class MACDTradeManager:
    """
    Complete trade lifecycle management for MACD strategy

    Manages trades exactly as they would be handled in live trading,
    including realistic entry/exit mechanics and performance tracking.
    """

    def __init__(self, epic: str, initial_balance: float = 10000.0):
        self.epic = epic
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        # Trade tracking
        self.trades = []
        self.open_trades = []
        self.next_trade_id = 1

        # Performance metrics
        self.total_pips = 0.0
        self.winning_trades = 0
        self.losing_trades = 0

        # Risk management settings (realistic live trading values)
        self.max_position_size = 0.02  # 2% of balance per trade
        self.max_open_trades = 1  # Conservative approach
        self.slippage_pips = 0.5  # Realistic slippage

        self.logger = logging.getLogger(f"{__name__}.{epic}")

        # Pair-specific settings
        if 'JPY' in epic:
            self.pip_value = 0.01
            self.min_pip_movement = 0.01
        else:
            self.pip_value = 0.0001
            self.min_pip_movement = 0.0001

        self.logger.info(f"🏦 Trade Manager initialized for {epic}")
        self.logger.info(f"   Initial balance: ${initial_balance:,.2f}")
        self.logger.info(f"   Max position size: {self.max_position_size:.1%}")
        self.logger.info(f"   Pip value: {self.pip_value}")

    def can_open_trade(self) -> bool:
        """Check if a new trade can be opened"""
        if len(self.open_trades) >= self.max_open_trades:
            return False

        if self.current_balance <= self.initial_balance * 0.5:  # 50% drawdown limit
            return False

        return True

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Calculate risk per trade (distance to stop loss)
            risk_pips = abs(entry_price - stop_loss) / self.pip_value

            # Maximum risk per trade (2% of balance)
            max_risk_amount = self.current_balance * self.max_position_size

            # Calculate position size to not exceed risk limit
            position_size = max_risk_amount / (risk_pips * self.pip_value)

            # Apply minimum and maximum position sizes
            position_size = max(0.01, min(position_size, 10.0))  # Between 0.01 and 10 lots

            return round(position_size, 2)

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01  # Default minimum position

    def calculate_stop_loss_take_profit(self, signal: Dict, current_price: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            # Get configuration from signal or use defaults
            stop_loss_pips = signal.get('stop_loss_pips', 15.0)  # Default 15 pips
            take_profit_pips = signal.get('take_profit_pips', 30.0)  # Default 30 pips (2:1 R:R)

            if signal['signal_type'] == 'BULL':
                stop_loss = current_price - (stop_loss_pips * self.pip_value)
                take_profit = current_price + (take_profit_pips * self.pip_value)
            else:  # BEAR
                stop_loss = current_price + (stop_loss_pips * self.pip_value)
                take_profit = current_price - (take_profit_pips * self.pip_value)

            return round(stop_loss, 5), round(take_profit, 5)

        except Exception as e:
            self.logger.error(f"Error calculating SL/TP: {e}")
            return current_price, current_price

    def open_trade(self, signal: Dict, current_price: float, timestamp: datetime) -> Optional[Trade]:
        """Open a new trade based on signal"""
        try:
            if not self.can_open_trade():
                self.logger.warning(f"Cannot open trade: {len(self.open_trades)} trades already open")
                return None

            # Apply realistic slippage
            if signal['signal_type'] == 'BULL':
                entry_price = current_price + (self.slippage_pips * self.pip_value)
            else:  # BEAR
                entry_price = current_price - (self.slippage_pips * self.pip_value)

            # Calculate stop loss and take profit
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(signal, entry_price)

            # Calculate position size
            position_size = self.calculate_position_size(entry_price, stop_loss)

            # Create trade
            trade = Trade(
                id=self.next_trade_id,
                epic=self.epic,
                signal_type=signal['signal_type'],
                entry_time=timestamp,
                entry_price=entry_price,
                entry_confidence=signal.get('confidence', 0.5),
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                status=TradeStatus.OPEN
            )

            # Add to tracking
            self.trades.append(trade)
            self.open_trades.append(trade)
            self.next_trade_id += 1

            self.logger.info(f"📈 TRADE OPENED: {trade.signal_type} #{trade.id}")
            self.logger.info(f"   Entry: {trade.entry_price:.5f} at {timestamp}")
            self.logger.info(f"   SL: {trade.stop_loss:.5f}, TP: {trade.take_profit:.5f}")
            self.logger.info(f"   Size: {trade.position_size} lots")

            return trade

        except Exception as e:
            self.logger.error(f"Error opening trade: {e}")
            return None

    def update_trade_metrics(self, trade: Trade, current_price: float):
        """Update trade metrics (MFE/MAE)"""
        try:
            if trade.signal_type == 'BULL':
                # For long trades
                unrealized_pnl = current_price - trade.entry_price
                favorable_move = max(0, unrealized_pnl)
                adverse_move = min(0, unrealized_pnl)
            else:
                # For short trades
                unrealized_pnl = trade.entry_price - current_price
                favorable_move = max(0, unrealized_pnl)
                adverse_move = min(0, unrealized_pnl)

            # Update maximum favorable/adverse excursion
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable_move)
            trade.max_adverse_excursion = min(trade.max_adverse_excursion, adverse_move)

        except Exception as e:
            self.logger.error(f"Error updating trade metrics: {e}")

    def check_trade_exit(self, trade: Trade, current_price: float, timestamp: datetime) -> Optional[ExitReason]:
        """Check if trade should be exited"""
        try:
            # Update trade duration
            trade.bars_held += 1

            # Update MFE/MAE
            self.update_trade_metrics(trade, current_price)

            # Check stop loss
            if trade.signal_type == 'BULL' and current_price <= trade.stop_loss:
                return ExitReason.STOP_LOSS
            elif trade.signal_type == 'BEAR' and current_price >= trade.stop_loss:
                return ExitReason.STOP_LOSS

            # Check take profit
            if trade.signal_type == 'BULL' and current_price >= trade.take_profit:
                return ExitReason.TAKE_PROFIT
            elif trade.signal_type == 'BEAR' and current_price <= trade.take_profit:
                return ExitReason.TAKE_PROFIT

            # Check timeout (close after 48 hours / 192 bars)
            if trade.bars_held >= 192:
                return ExitReason.TIMEOUT

            return None

        except Exception as e:
            self.logger.error(f"Error checking trade exit: {e}")
            return None

    def close_trade(self, trade: Trade, exit_price: float, exit_reason: ExitReason, timestamp: datetime):
        """Close a trade and update metrics"""
        try:
            # Apply slippage on exit
            if trade.signal_type == 'BULL':
                final_exit_price = exit_price - (self.slippage_pips * self.pip_value)
            else:  # BEAR
                final_exit_price = exit_price + (self.slippage_pips * self.pip_value)

            # Calculate P&L
            if trade.signal_type == 'BULL':
                pnl_pips = (final_exit_price - trade.entry_price) / self.pip_value
            else:  # BEAR
                pnl_pips = (trade.entry_price - final_exit_price) / self.pip_value

            pnl_amount = pnl_pips * self.pip_value * trade.position_size

            # Update trade
            trade.exit_time = timestamp
            trade.exit_price = final_exit_price
            trade.exit_reason = exit_reason
            trade.profit_loss = pnl_amount
            trade.profit_loss_pips = pnl_pips

            # Update status
            if pnl_pips > 0:
                trade.status = TradeStatus.CLOSED_PROFIT
                self.winning_trades += 1
            elif pnl_pips < 0:
                trade.status = TradeStatus.CLOSED_LOSS
                self.losing_trades += 1
            else:
                trade.status = TradeStatus.CLOSED_NEUTRAL

            # Update balance and metrics
            self.current_balance += pnl_amount
            self.total_pips += pnl_pips

            # Remove from open trades
            self.open_trades.remove(trade)

            self.logger.info(f"📊 TRADE CLOSED: {trade.signal_type} #{trade.id}")
            self.logger.info(f"   Exit: {final_exit_price:.5f} at {timestamp}")
            self.logger.info(f"   Reason: {exit_reason.value}")
            self.logger.info(f"   P&L: {pnl_pips:+.1f} pips (${pnl_amount:+.2f})")
            self.logger.info(f"   Duration: {trade.bars_held} bars")

        except Exception as e:
            self.logger.error(f"Error closing trade: {e}")

    def process_bar(self, current_price: float, timestamp: datetime):
        """Process current market bar for all open trades"""
        try:
            trades_to_close = []

            for trade in self.open_trades:
                exit_reason = self.check_trade_exit(trade, current_price, timestamp)
                if exit_reason:
                    trades_to_close.append((trade, exit_reason))

            # Close trades that hit exit conditions
            for trade, exit_reason in trades_to_close:
                self.close_trade(trade, current_price, exit_reason, timestamp)

        except Exception as e:
            self.logger.error(f"Error processing bar: {e}")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            closed_trades = [t for t in self.trades if t.status != TradeStatus.OPEN]
            total_trades = len(closed_trades)

            if total_trades == 0:
                return {'error': 'No closed trades to analyze'}

            # Calculate metrics
            total_pnl = sum(t.profit_loss for t in closed_trades)
            total_pips = sum(t.profit_loss_pips for t in closed_trades)

            winning_trades = [t for t in closed_trades if t.profit_loss_pips > 0]
            losing_trades = [t for t in closed_trades if t.profit_loss_pips < 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

            avg_win = sum(t.profit_loss_pips for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.profit_loss_pips for t in losing_trades) / len(losing_trades) if losing_trades else 0

            profit_factor = abs(sum(t.profit_loss_pips for t in winning_trades) / sum(t.profit_loss_pips for t in losing_trades)) if losing_trades else float('inf')

            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_pips': total_pips,
                'total_pnl': total_pnl,
                'average_win_pips': avg_win,
                'average_loss_pips': avg_loss,
                'profit_factor': profit_factor,
                'balance_change': self.current_balance - self.initial_balance,
                'balance_change_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
                'max_favorable_excursion': max((t.max_favorable_excursion for t in closed_trades), default=0),
                'max_adverse_excursion': min((t.max_adverse_excursion for t in closed_trades), default=0)
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance: {e}")
            return {'error': str(e)}