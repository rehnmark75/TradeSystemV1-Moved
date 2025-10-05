# ============================================================================
# backtests/__init__.py
# ============================================================================

"""
Modular Backtesting System
Each strategy has its own dedicated backtest module
"""

from .backtest_base import BacktestBase
from .backtest_ema import EMABacktest
from .backtest_macd import EnhancedMACDBacktest as MACDBacktest
from .backtest_kama import KAMABacktest
from .backtest_bb_supertrend import BBSupertrendBacktest
from .backtest_zero_lag import ZeroLagBacktest
#from .backtest_combined import CombinedBacktest  # TEMP: disabled due to missing combined_strategy
#from .backtest_scalping import ScalpingBacktest
from .backtest_all import AllStrategiesBacktest

__all__ = [
    'BacktestBase',
    'EMABacktest',
    'MACDBacktest',
    'KAMABacktest',
    'BBSupertrendBacktest',
    'ZeroLagBacktest',
    #'CombinedBacktest',  # TEMP: disabled
    #'ScalpingBacktest',
    'AllStrategiesBacktest'
]