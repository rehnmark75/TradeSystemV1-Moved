"""
Signal Scanner Strategies

8 consolidated scanners (down from 14):

Backtested & Optimized:
- ZLMATrendScanner: Zero-Lag MA crossover (PF 1.55, WR 50%)
- MACDMomentumScanner: MACD histogram momentum (PF 1.71, WR 42%)
- EMAPullbackScanner: EMA cascade pullback (PF 2.02)

To Be Backtested:
- BreakoutConfirmationScanner: 52W high breakouts with volume
- GapAndGoScanner: Gap continuation plays
- ReversalScanner: Capitulation + mean reversion + Wyckoff springs (merged)
- RSIDivergenceScanner: Price/RSI divergence reversals
- TrendReversalScanner: Multi-day downtrend-to-uptrend reversals
"""

# Backtested & optimized strategies
from .zlma_trend import ZLMATrendScanner
from .macd_momentum import MACDMomentumScanner
from .ema_pullback import EMAPullbackScanner

# To be backtested
from .breakout_confirmation import BreakoutConfirmationScanner
from .gap_and_go import GapAndGoScanner
from .reversal_scanner import ReversalScanner
from .rsi_divergence import RSIDivergenceScanner
from .trend_reversal import TrendReversalScanner

__all__ = [
    # Backtested & optimized
    'ZLMATrendScanner',
    'MACDMomentumScanner',
    'EMAPullbackScanner',
    # To be backtested
    'BreakoutConfirmationScanner',
    'GapAndGoScanner',
    'ReversalScanner',
    'RSIDivergenceScanner',
    'TrendReversalScanner',
]
