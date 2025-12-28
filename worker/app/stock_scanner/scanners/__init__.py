"""
Stock Signal Scanners Module

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

# Strategy scanners (7 total - TREND_REVERSAL removed)
from .strategies import (
    # Backtested & optimized strategies (PF > 1.5)
    ZLMATrendScanner,
    MACDMomentumScanner,
    EMAPullbackScanner,
    ReversalScanner,  # PF 2.44 - Best performer
    # Other strategies (lower PF)
    BreakoutConfirmationScanner,
    GapAndGoScanner,
    RSIDivergenceScanner,
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

    # Strategy scanners (7 total)
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
