"""
Stock Scanner - Trading Strategies

Backtestable strategies:
- EMATrendPullbackStrategy: Pullback entries in EMA cascade (PF 2.02)
- MACDMomentumStrategy: MACD histogram zero-cross momentum (PF 1.71)
- ZLMACrossoverStrategy: Zero-Lag MA crossover signals (for backtesting)

Legacy (scanner-specific):
- ZeroLagMATrendStrategy: Original ZLMA scanner implementation
"""

from .zlma_trend import ZeroLagMATrendStrategy
from .ema_trend_pullback import EMATrendPullbackStrategy, PullbackSignal
from .macd_momentum import MACDMomentumStrategy, MACDMomentumSignal
from .zlma_crossover import ZLMACrossoverStrategy, ZLMASignal

__all__ = [
    # Backtestable strategies
    'EMATrendPullbackStrategy',
    'PullbackSignal',
    'MACDMomentumStrategy',
    'MACDMomentumSignal',
    'ZLMACrossoverStrategy',
    'ZLMASignal',
    # Legacy
    'ZeroLagMATrendStrategy',
]
