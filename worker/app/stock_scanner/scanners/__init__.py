"""
Stock Signal Scanners Module

Automated signal scanning system with multiple strategies:
- Trend Momentum: Pullback entries in established uptrends
- Breakout Confirmation: Volume-confirmed breakouts
- Mean Reversion: Oversold bounces with reversal patterns
- Gap & Go: Gap continuation plays

Each scanner generates scored signals with entry, stop-loss, and take-profit levels.

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

# Strategy scanners
from .strategies import (
    TrendMomentumScanner,
    BreakoutConfirmationScanner,
    MeanReversionScanner,
    GapAndGoScanner,
    ZLMATrendScanner,
    # Forex-adapted strategies
    SMCEmaTrendScanner,
    EMACrossoverScanner,
    MACDMomentumScanner,
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

    # Strategies
    'TrendMomentumScanner',
    'BreakoutConfirmationScanner',
    'MeanReversionScanner',
    'GapAndGoScanner',
    'ZLMATrendScanner',
    # Forex-adapted strategies
    'SMCEmaTrendScanner',
    'EMACrossoverScanner',
    'MACDMomentumScanner',
]
