MAIN_INIT = '''
"""
Trading System - Complete Backtesting Framework
"""
from backtesting import BacktestEngine
from core import Signal, Trade, Portfolio, SignalType, TradeStatus
from examples import run_simple_backtest

__version__ = "1.0.0"
__all__ = [
    "BacktestEngine", 
    "Signal", 
    "Trade", 
    "Portfolio", 
    "SignalType", 
    "TradeStatus",
    "run_simple_backtest"
]
'''