"""
Signal Scanner Strategies

7 consolidated scanners (TREND_REVERSAL removed - PF 1.09 too low):

Backtested & Optimized (PF > 1.5):
- ZLMATrendScanner: Zero-Lag MA crossover (PF 1.55, WR 50%)
- MACDMomentumScanner: MACD histogram momentum (PF 1.71, WR 42%)
- EMAPullbackScanner: EMA cascade pullback (PF 2.02)
- ReversalScanner: Capitulation + mean reversion + Wyckoff springs (PF 2.44)

Other Strategies (lower PF, included for diversity):
- BreakoutConfirmationScanner: 52W high breakouts with volume
- GapAndGoScanner: Gap continuation plays
- RSIDivergenceScanner: Price/RSI divergence reversals
"""

# Backtested & optimized strategies (PF > 1.5)
from .zlma_trend import ZLMATrendScanner
from .macd_momentum import MACDMomentumScanner
from .ema_pullback import EMAPullbackScanner
from .reversal_scanner import ReversalScanner

# Other strategies (lower PF)
from .breakout_confirmation import BreakoutConfirmationScanner
from .gap_and_go import GapAndGoScanner
from .rsi_divergence import RSIDivergenceScanner

__all__ = [
    # Backtested & optimized (PF > 1.5)
    'ZLMATrendScanner',
    'MACDMomentumScanner',
    'EMAPullbackScanner',
    'ReversalScanner',
    # Other strategies
    'BreakoutConfirmationScanner',
    'GapAndGoScanner',
    'RSIDivergenceScanner',
]
