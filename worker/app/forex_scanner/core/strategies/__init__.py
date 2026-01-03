# core/strategies/__init__.py
"""
Trading Strategies Module

NOTE: After January 2026 cleanup, only SMC Simple strategy is active.
Legacy strategies have been archived to forex_scanner/archive/disabled_strategies/

Adding a new strategy:
    1. Create strategy class implementing StrategyInterface
    2. Use @register_strategy('STRATEGY_NAME') decorator
    3. Enable in database (strategy_config.enabled_strategies table)
"""

from .base_strategy import BaseStrategy
from .smc_simple_strategy import SMCSimpleStrategy, create_smc_simple_strategy
from .strategy_registry import (
    StrategyRegistry,
    StrategyInterface,
    register_strategy,
)

__all__ = [
    # Base classes
    'BaseStrategy',
    'StrategyInterface',

    # Active strategies
    'SMCSimpleStrategy',
    'create_smc_simple_strategy',

    # Registry
    'StrategyRegistry',
    'register_strategy',
]
