"""
Backtesting Module

NOTE: After January 2026 cleanup, BacktestEngine was archived.
The new backtest pipeline uses BacktestTradingOrchestrator directly.
Legacy BacktestEngine is in forex_scanner/archive/disabled_backtest/
"""

from .performance_analyzer import PerformanceAnalyzer
from .signal_analyzer import SignalAnalyzer

__all__ = [
    'PerformanceAnalyzer',
    'SignalAnalyzer'
]
