"""
Signal Scanner Strategies

Concrete implementations of trading scanners:
- TrendMomentumScanner: Pullback entries in established uptrends
- BreakoutConfirmationScanner: Volume-confirmed breakouts
- MeanReversionScanner: Oversold bounces with reversal patterns
- GapAndGoScanner: Gap continuation plays
"""

from .trend_momentum import TrendMomentumScanner
from .breakout_confirmation import BreakoutConfirmationScanner
from .mean_reversion import MeanReversionScanner
from .gap_and_go import GapAndGoScanner

__all__ = [
    'TrendMomentumScanner',
    'BreakoutConfirmationScanner',
    'MeanReversionScanner',
    'GapAndGoScanner',
]
