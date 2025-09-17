# core/strategies/__init__.py
"""
Trading Strategies Module
"""

from .base_strategy import BaseStrategy
from .ema_strategy import EMAStrategy, create_ema_strategy
from .macd_strategy import MACDStrategy
from .combined_strategy import CombinedStrategy
from .bb_supertrend_strategy import BollingerSupertrendStrategy
from .kama_strategy import KAMAStrategy
from .momentum_bias_strategy import MomentumBiasStrategy
from .zero_lag_strategy import ZeroLagStrategy
from .smc_strategy import SMCStrategy, create_smc_strategy
from .smc_strategy_fast import SMCStrategyFast, create_smc_strategy_fast
from .mean_reversion_strategy import MeanReversionStrategy, create_mean_reversion_strategy
__all__ = [
    'BaseStrategy',
    'EMAStrategy',
    'MACDStrategy',
    'CombinedStrategy',
    'BollingerSupertrendStrategy',
    'KAMAStrategy',
    'MomentumBiasStrategy',
    'ZeroLagStrategy',
    'SMCStrategy',
    'create_smc_strategy',
    'SMCStrategyFast',
    'create_smc_strategy_fast',
    'MeanReversionStrategy',
    'create_mean_reversion_strategy'
]