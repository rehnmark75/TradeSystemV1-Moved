"""
Signal Scanner Strategies

Concrete implementations of trading scanners:
- TrendMomentumScanner: Pullback entries in established uptrends
- BreakoutConfirmationScanner: Volume-confirmed breakouts
- MeanReversionScanner: Oversold bounces with reversal patterns
- GapAndGoScanner: Gap continuation plays
- EarningsMomentumScanner: Post-earnings drift plays
- ShortSqueezeScanner: High short interest squeeze setups
- SectorRotationScanner: Sector leaders and laggards
- ZLMATrendScanner: Zero-Lag MA crossover signals

Backtested strategies (use strategy class directly):
- MACDMomentumScanner: MACD momentum (PF 1.71) - uses MACDMomentumStrategy
- EMAPullbackScanner: EMA pullback entries (PF 2.02) - uses EMATrendPullbackStrategy

AlphaSuite-adapted strategies:
- SellingClimaxScanner: Capitulation bottoming with volume spike and reversal
- RSIDivergenceScanner: Price/RSI divergence for reversal detection
- WyckoffSpringScanner: Accumulation spring with support test
- TrendReversalScanner: Downtrend-to-uptrend reversals with multi-day confirmation
"""

from .trend_momentum import TrendMomentumScanner
from .breakout_confirmation import BreakoutConfirmationScanner
from .mean_reversion import MeanReversionScanner
from .gap_and_go import GapAndGoScanner
from .earnings_momentum import EarningsMomentumScanner
from .short_squeeze import ShortSqueezeScanner
from .sector_rotation import SectorRotationScanner
from .zlma_trend import ZLMATrendScanner

# Backtested strategies
from .macd_momentum import MACDMomentumScanner
from .ema_pullback import EMAPullbackScanner

# AlphaSuite-adapted strategies
from .selling_climax import SellingClimaxScanner
from .rsi_divergence import RSIDivergenceScanner
from .wyckoff_spring import WyckoffSpringScanner
from .trend_reversal import TrendReversalScanner

__all__ = [
    'TrendMomentumScanner',
    'BreakoutConfirmationScanner',
    'MeanReversionScanner',
    'GapAndGoScanner',
    'EarningsMomentumScanner',
    'ShortSqueezeScanner',
    'SectorRotationScanner',
    'ZLMATrendScanner',
    # Backtested strategies
    'MACDMomentumScanner',
    'EMAPullbackScanner',
    # AlphaSuite-adapted strategies
    'SellingClimaxScanner',
    'RSIDivergenceScanner',
    'WyckoffSpringScanner',
    'TrendReversalScanner',
]
