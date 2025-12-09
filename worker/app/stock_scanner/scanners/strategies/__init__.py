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
"""

from .trend_momentum import TrendMomentumScanner
from .breakout_confirmation import BreakoutConfirmationScanner
from .mean_reversion import MeanReversionScanner
from .gap_and_go import GapAndGoScanner
from .earnings_momentum import EarningsMomentumScanner
from .short_squeeze import ShortSqueezeScanner
from .sector_rotation import SectorRotationScanner

__all__ = [
    'TrendMomentumScanner',
    'BreakoutConfirmationScanner',
    'MeanReversionScanner',
    'GapAndGoScanner',
    'EarningsMomentumScanner',
    'ShortSqueezeScanner',
    'SectorRotationScanner',
]
