"""
Stock Signal Scanners Module

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

Usage:
    from stock_scanner.scanners import ScannerManager, TradingViewExporter

    # Initialize
    manager = ScannerManager(db_manager)
    await manager.initialize()

    # Run all scanners
    signals = await manager.run_all_scanners()

    # Export to TradingView
    exporter = TradingViewExporter()
    csv = exporter.to_csv(signals)
"""

from .base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier
)
from .scoring import SignalScorer, ScoreComponents
from .exclusion_filters import (
    ExclusionFilterEngine, ExclusionReason, ExclusionResult,
    FilterConfig, get_conservative_filter, get_aggressive_filter,
    get_momentum_filter, get_mean_reversion_filter
)
from .scanner_manager import ScannerManager
from .export import TradingViewExporter

# Strategy scanners (8 total - consolidated from 14)
from .strategies import (
    # Backtested & optimized strategies
    ZLMATrendScanner,
    MACDMomentumScanner,
    EMAPullbackScanner,
    # To be backtested
    BreakoutConfirmationScanner,
    GapAndGoScanner,
    ReversalScanner,  # Merged: selling_climax + mean_reversion + wyckoff_spring
    RSIDivergenceScanner,
    TrendReversalScanner,
)

__all__ = [
    # Core
    'BaseScanner',
    'SignalSetup',
    'ScannerConfig',
    'SignalType',
    'QualityTier',

    # Scoring
    'SignalScorer',
    'ScoreComponents',

    # Filters
    'ExclusionFilterEngine',
    'ExclusionReason',
    'ExclusionResult',
    'FilterConfig',
    'get_conservative_filter',
    'get_aggressive_filter',
    'get_momentum_filter',
    'get_mean_reversion_filter',

    # Manager
    'ScannerManager',

    # Export
    'TradingViewExporter',

    # Strategy scanners (8 total)
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
