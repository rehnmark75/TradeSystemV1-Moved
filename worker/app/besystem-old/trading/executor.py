# ================================
# 7. trading/executor.py
# ================================
import pandas as pd
from typing import Optional, Dict
from core.data_structures import Trade, Signal, Portfolio, TradeStatus
from trading.risk_manager import RiskManager
from analysis.signals import SignalType


# ================================
# 7. TRADE EXECUTION SIMULATOR
# ================================

class TradeExecutor:
    """Simulates trade execution for backtesting"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.trade_counter = 0
    
    def execute_signal(self, signal: Signal, portfolio: Portfolio) -> Optional[Trade]:
        """Execute a trading signal"""
        if not self.risk_manager.check_trade_allowed(signal, portfolio):
            return None
        
        # Calculate stop loss and take profit
        stop_loss_pips = self.risk_manager.calculate_stop_loss(signal)
        take_profit_pips = self.risk_manager.calculate_take_profit(signal, stop_loss_pips)
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(signal, portfolio, stop_loss_pips)
        
        if position_size <= 0:
            return None
        
        # Calculate entry price (including spread)
        entry_price = self._calculate_entry_price(signal)
        
        # Calculate stop loss and take profit prices
        if signal.signal_type == SignalType.BULL:
            stop_loss_price = entry_price - (stop_loss_pips / signal.pip_multiplier)
            take_profit_price = entry_price + (take_profit_pips / signal.pip_multiplier)
        else:
            stop_loss_price = entry_price + (stop_loss_pips / signal.pip_multiplier)
            take_profit_price = entry_price - (take_profit_pips / signal.pip_multiplier)
        
        # Create trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"T{self.trade_counter:06d}",
            signal=signal,
            entry_price=entry_price,
            entry_time=signal.timestamp,
            position_size=position_size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            status=TradeStatus.OPEN
        )
        
        # Add to portfolio
        portfolio.open_trades.append(trade)
        self._update_portfolio_margin(portfolio)
        
        return trade
    
    def _calculate_entry_price(self, signal: Signal) -> float:
        """Calculate actual entry price including spread"""
        spread = signal.spread_pips / signal.pip_multiplier
        
        if signal.signal_type == SignalType.BULL:
            return signal.price + (spread / 2)  # ASK price for buying
        else:
            return signal.price - (spread / 2)  # BID price for selling
    
    def _update_portfolio_margin(self, portfolio: Portfolio):
        """Update portfolio margin calculations"""
        margin_per_trade = 1000  # Simplified margin calculation
        portfolio.margin_used = len(portfolio.open_trades) * margin_per_trade
        portfolio.free_margin = portfolio.current_balance - portfolio.margin_used
    
    def update_open_trades(self, portfolio: Portfolio, current_data: Dict[str, pd.DataFrame]):
        """Update all open trades with current market data"""
        closed_trades = []
        
        for trade in portfolio.open_trades[:]:  # Create copy to iterate safely
            epic_data = current_data.get(trade.signal.epic)
            
            if epic_data is None or len(epic_data) == 0:
                continue
            
            # Get current price data
            current_bar = epic_data.iloc[-1]
            current_time = current_bar['start_time']
            
            # Update trade metrics
            self._update_trade_metrics(trade, current_bar)
            
            # Check for exit conditions
            if self._check_exit_conditions(trade, current_bar):
                self._close_trade(trade, current_bar, portfolio)
                closed_trades.append(trade)
        
        # Remove closed trades from open trades
        for trade in closed_trades:
            if trade in portfolio.open_trades:
                portfolio.open_trades.remove(trade)
        
        self._update_portfolio_metrics(portfolio)
    
    def _update_trade_metrics(self, trade: Trade, current_bar: pd.Series):
        """Update trade's running metrics"""
        current_price = current_bar['close']
        pip_multiplier = trade.signal.pip_multiplier
        
        if trade.signal.signal_type == SignalType.BULL:
            current_pnl_pips = (current_price - trade.entry_price) * pip_multiplier
        else:
            current_pnl_pips = (trade.entry_price - current_price) * pip_multiplier
        
        # Update max profit/loss
        if current_pnl_pips > trade.max_profit_pips:
            trade.max_profit_pips = current_pnl_pips
        
        if current_pnl_pips < 0 and abs(current_pnl_pips) > trade.max_loss_pips:
            trade.max_loss_pips = abs(current_pnl_pips)
        
        # Update duration
        trade.duration_minutes = int((current_bar['start_time'] - trade.entry_time).total_seconds() / 60)
    
    def _check_exit_conditions(self, trade: Trade, current_bar: pd.Series) -> bool:
        """Check if trade should be closed"""
        current_high = current_bar['high']
        current_low = current_bar['low']
        
        if trade.signal.signal_type == SignalType.BULL:
            # Check stop loss
            if current_low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.status = TradeStatus.CLOSED_LOSS
                return True
            
            # Check take profit
            if current_high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.status = TradeStatus.CLOSED_WIN
                return True
        
        else:  # BEAR
            # Check stop loss
            if current_high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.status = TradeStatus.CLOSED_LOSS
                return True
            
            # Check take profit
            if current_low <= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.status = TradeStatus.CLOSED_WIN
                return True
        
        # Check time-based exit (e.g., end of day)
        if trade.duration_minutes > 480:  # 8 hours max
            trade.exit_price = current_bar['close']
            trade.status = TradeStatus.CLOSED_BE
            return True
        
        return False
    
    def _close_trade(self, trade: Trade, current_bar: pd.Series, portfolio: Portfolio):
        """Close a trade and update portfolio"""
        if trade.exit_price is None:
            trade.exit_price = current_bar['close']
        
        trade.exit_time = current_bar['start_time']
        
        # Calculate final P&L
        pip_multiplier = trade.signal.pip_multiplier
        
        if trade.signal.signal_type == SignalType.BULL:
            trade.pnl_pips = (trade.exit_price - trade.entry_price) * pip_multiplier
        else:
            trade.pnl_pips = (trade.entry_price - trade.exit_price) * pip_multiplier
        
        # Calculate currency P&L (simplified)
        pip_value = self.risk_manager._calculate_pip_value(trade.signal.epic, trade.position_size)
        trade.pnl_currency = trade.pnl_pips * pip_value
        
        # Update portfolio
        portfolio.current_balance += trade.pnl_currency
        portfolio.total_pnl += trade.pnl_currency
        portfolio.daily_pnl += trade.pnl_currency
        portfolio.closed_trades.append(trade)
    
    def _update_portfolio_metrics(self, portfolio: Portfolio):
        """Update portfolio metrics"""
        # Calculate unrealized P&L for open trades
        unrealized_pnl = 0
        for trade in portfolio.open_trades:
            # Simplified unrealized P&L calculation
            pass  # Would need current market prices
        
        portfolio.equity = portfolio.current_balance + unrealized_pnl
        self._update_portfolio_margin(portfolio)