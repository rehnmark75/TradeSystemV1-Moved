# core/strategies/__init__.py
"""
Trading Strategies Module
"""

from .base_strategy import BaseStrategy
from .ema_strategy import EMAStrategy, create_ema_strategy
from .macd_strategy import MACDStrategy
# from .combined_strategy import CombinedStrategy  # Removed - strategy was disabled and unused
from .bb_supertrend_strategy import BollingerSupertrendStrategy
from .kama_strategy import KAMAStrategy
from .zero_lag_strategy import ZeroLagStrategy
from .momentum_strategy import MomentumStrategy
# from .smc_strategy import SMCStrategy, create_smc_strategy  # Removed - legacy strategy, SMCStrategyFast is in production
from .smc_strategy_fast import SMCStrategyFast, create_smc_strategy_fast
from .mean_reversion_strategy import MeanReversionStrategy, create_mean_reversion_strategy
from .ichimoku_strategy import IchimokuStrategy
from .ranging_market_strategy import RangingMarketStrategy
from .scalping_strategy import ScalpingStrategy
from .volume_profile_strategy import VolumeProfileStrategy
__all__ = [
    'BaseStrategy',
    'EMAStrategy',
    'MACDStrategy',
    # 'CombinedStrategy',  # Removed - strategy was disabled and unused
    'BollingerSupertrendStrategy',
    'KAMAStrategy',
    'ZeroLagStrategy',
    'MomentumStrategy',
    # 'SMCStrategy', 'create_smc_strategy',  # Removed - legacy strategy
    'SMCStrategyFast',
    'create_smc_strategy_fast',
    'MeanReversionStrategy',
    'create_mean_reversion_strategy',
    'IchimokuStrategy',
    'RangingMarketStrategy',
    'ScalpingStrategy',
    'VolumeProfileStrategy'
]