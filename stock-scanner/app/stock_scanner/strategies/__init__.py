"""
Stock Scanner - Trading Strategies

Backtestable Strategies (8 total):

Already Optimized:
- EMATrendPullbackStrategy: Pullback entries in EMA cascade (PF 2.02)
- MACDMomentumStrategy: MACD histogram zero-cross momentum (PF 1.71, WR 42%)
- ZLMACrossoverStrategy: Zero-Lag MA crossover signals (PF 1.55, WR 50%)

To Be Backtested:
- BreakoutConfirmationStrategy: 52W high breakouts with volume confirmation
- GapAndGoStrategy: Gap continuation plays
- ReversalStrategy: Unified climax + mean reversion + spring detection
- RSIDivergenceStrategy: Bullish RSI divergence reversals
- TrendReversalStrategy: Multi-day downtrend-to-uptrend transitions
"""

# Original / already optimized strategies
from .zlma_trend import ZeroLagMATrendStrategy
from .ema_trend_pullback import EMATrendPullbackStrategy, PullbackSignal
from .macd_momentum import MACDMomentumStrategy, MACDMomentumSignal
from .zlma_crossover import ZLMACrossoverStrategy, ZLMASignal

# New strategies to be backtested
from .breakout_confirmation import BreakoutConfirmationStrategy, BreakoutSignal
from .gap_and_go import GapAndGoStrategy, GapSignal
from .reversal_strategy import ReversalStrategy, ReversalSignal
from .rsi_divergence import RSIDivergenceStrategy, RSIDivergenceSignal
from .trend_reversal import TrendReversalStrategy, TrendReversalSignal

__all__ = [
    # Already optimized strategies
    'EMATrendPullbackStrategy',
    'PullbackSignal',
    'MACDMomentumStrategy',
    'MACDMomentumSignal',
    'ZLMACrossoverStrategy',
    'ZLMASignal',

    # To be backtested
    'BreakoutConfirmationStrategy',
    'BreakoutSignal',
    'GapAndGoStrategy',
    'GapSignal',
    'ReversalStrategy',
    'ReversalSignal',
    'RSIDivergenceStrategy',
    'RSIDivergenceSignal',
    'TrendReversalStrategy',
    'TrendReversalSignal',

    # Legacy (scanner-specific)
    'ZeroLagMATrendStrategy',
]
