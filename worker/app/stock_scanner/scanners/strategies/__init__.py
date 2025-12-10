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

Forex-adapted strategies:
- SMCEmaTrendScanner: SMC-style EMA trend following with swing structure
- EMACrossoverScanner: EMA cascade crossover with trend alignment
- MACDMomentumScanner: MACD momentum confluence with price structure
"""

from .trend_momentum import TrendMomentumScanner
from .breakout_confirmation import BreakoutConfirmationScanner
from .mean_reversion import MeanReversionScanner
from .gap_and_go import GapAndGoScanner
from .earnings_momentum import EarningsMomentumScanner
from .short_squeeze import ShortSqueezeScanner
from .sector_rotation import SectorRotationScanner

# Forex-adapted strategies
from .smc_ema_trend import SMCEmaTrendScanner
from .ema_crossover import EMACrossoverScanner
from .macd_momentum import MACDMomentumScanner

__all__ = [
    'TrendMomentumScanner',
    'BreakoutConfirmationScanner',
    'MeanReversionScanner',
    'GapAndGoScanner',
    'EarningsMomentumScanner',
    'ShortSqueezeScanner',
    'SectorRotationScanner',
    # Forex-adapted strategies
    'SMCEmaTrendScanner',
    'EMACrossoverScanner',
    'MACDMomentumScanner',
]
