from core.data_structures import Signal, Portfolio
from core.config import EpicConfig

# ================================
# 6. RISK MANAGEMENT
# ================================

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, max_risk_per_trade: float = 0.02, max_daily_loss: float = 0.05,
                 max_open_trades: int = 5, position_sizing_method: str = 'fixed_percent'):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_daily_loss = max_daily_loss  # 5% daily loss limit
        self.max_open_trades = max_open_trades
        self.position_sizing_method = position_sizing_method
    
    def calculate_position_size(self, signal: Signal, portfolio: Portfolio, 
                              stop_loss_pips: float) -> float:
        """Calculate appropriate position size"""
        if self.position_sizing_method == 'fixed_percent':
            return self._calculate_fixed_percent_size(signal, portfolio, stop_loss_pips)
        elif self.position_sizing_method == 'kelly':
            return self._calculate_kelly_size(signal, portfolio, stop_loss_pips)
        else:
            return self._calculate_fixed_size(signal, portfolio)
    
    def _calculate_fixed_percent_size(self, signal: Signal, portfolio: Portfolio, 
                                    stop_loss_pips: float) -> float:
        """Calculate position size based on fixed percentage risk"""
        if stop_loss_pips <= 0:
            return 0
        
        risk_amount = portfolio.current_balance * self.max_risk_per_trade
        pip_value = self._calculate_pip_value(signal.epic, 1.0)  # For 1 unit
        max_loss = stop_loss_pips * pip_value
        
        if max_loss <= 0:
            return 0
        
        position_size = risk_amount / max_loss
        return round(position_size, 2)
    
    def _calculate_kelly_size(self, signal: Signal, portfolio: Portfolio, 
                            stop_loss_pips: float) -> float:
        """Calculate Kelly criterion-based position size"""
        # Simplified Kelly - would need historical win/loss data
        win_rate = min(0.6, signal.confidence_score)  # Conservative estimate
        avg_win = 15  # pips
        avg_loss = stop_loss_pips
        
        if avg_loss <= 0:
            return 0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        position_value = portfolio.current_balance * kelly_fraction
        pip_value = self._calculate_pip_value(signal.epic, 1.0)
        
        return round(position_value / (stop_loss_pips * pip_value), 2)
    
    def _calculate_fixed_size(self, signal: Signal, portfolio: Portfolio) -> float:
        """Calculate fixed position size"""
        return 10000  # Fixed 1 mini lot
    
    def _calculate_pip_value(self, epic: str, position_size: float) -> float:
        """Calculate pip value for given position size"""
        settings = EpicConfig.get_settings(epic)
        pip_multiplier = settings['pip_multiplier']
        
        # Simplified pip value calculation
        if 'JPY' in epic:
            return position_size * 0.01  # JPY pairs
        else:
            return position_size * 0.0001  # Other pairs
    
    def check_trade_allowed(self, signal: Signal, portfolio: Portfolio) -> bool:
        """Check if new trade is allowed based on risk rules"""
        # Check daily loss limit
        if portfolio.daily_pnl < -portfolio.initial_balance * self.max_daily_loss:
            return False
        
        # Check max open trades
        if len(portfolio.open_trades) >= self.max_open_trades:
            return False
        
        # Check minimum confidence
        if signal.confidence_score < 0.3:
            return False
        
        return True
    
    def calculate_stop_loss(self, signal: Signal, atr_value: float = None) -> float:
        """Calculate stop loss in pips"""
        base_stop = 15  # Base stop loss in pips
        
        # Adjust based on volatility
        volatility_multiplier = {
            'low': 0.8, 'medium': 1.0, 'high': 1.3, 'very_high': 1.6
        }.get(EpicConfig.get_settings(signal.epic)['volatility'], 1.0)
        
        stop_loss_pips = base_stop * volatility_multiplier
        
        # Use ATR if available
        if atr_value:
            atr_stop = atr_value * signal.pip_multiplier * 1.5
            stop_loss_pips = max(stop_loss_pips, atr_stop)
        
        return round(stop_loss_pips, 1)
    
    def calculate_take_profit(self, signal: Signal, stop_loss_pips: float) -> float:
        """Calculate take profit in pips"""
        # Base risk-reward ratio
        base_rr = 1.5
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + signal.confidence_score
        
        # Adjust based on market conditions
        distance_to_resistance = signal.distance_to_resistance_pips
        if distance_to_resistance and distance_to_resistance > 0:
            # Don't set TP beyond nearby resistance
            max_tp = distance_to_resistance * 0.8
            calculated_tp = stop_loss_pips * base_rr * confidence_multiplier
            return min(calculated_tp, max_tp)
        
        return stop_loss_pips * base_rr * confidence_multiplier