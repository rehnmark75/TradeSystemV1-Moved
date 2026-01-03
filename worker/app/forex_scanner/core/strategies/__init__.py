# core/strategies/__init__.py
"""
Trading Strategies Module

NOTE: After January 2026 cleanup, only SMC Simple strategy is active.
Legacy strategies have been archived to forex_scanner/archive/disabled_strategies/
"""

from .base_strategy import BaseStrategy
from .smc_simple_strategy import SMCSimpleStrategy, create_smc_simple_strategy

__all__ = [
    'BaseStrategy',
    'SMCSimpleStrategy',
    'create_smc_simple_strategy',
]
