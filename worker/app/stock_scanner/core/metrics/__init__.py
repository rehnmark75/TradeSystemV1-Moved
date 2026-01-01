"""
Stock Scanner - Metrics Calculation Module

Components:
- MetricsCalculator: Core technical metrics (EMA, RSI, ATR, etc.)
- RSCalculator: Relative Strength vs SPY calculations
- SectorAnalyzer: Sector rotation and RS analysis
"""

from .calculator import MetricsCalculator
from .rs_calculator import RSCalculator
from .sector_analyzer import SectorAnalyzer, run_sector_analysis

__all__ = [
    'MetricsCalculator',
    'RSCalculator',
    'SectorAnalyzer',
    'run_sector_analysis',
]
