"""
Backtest System Initialization
Sets up the modular backtest system with all available adapters
"""

import logging

# Import core services
from .backtest_service import (
    get_backtest_registry,
    get_backtest_runner,
    get_result_processor,
    BacktestConfig,
    BacktestResult,
    StrategyInfo
)

from .backtest_chart_service import get_chart_service
from .strategy_adapters.ema_adapter import EMAStrategyAdapter
from .strategy_adapters.macd_adapter import MACDStrategyAdapter

logger = logging.getLogger(__name__)


def initialize_backtest_system():
    """Initialize the backtest system with all available adapters"""
    logger.info("🚀 Initializing Modular Backtest System...")

    registry = get_backtest_registry()

    # Register specific adapters
    adapters = [
        ('ema', EMAStrategyAdapter()),
        ('macd', MACDStrategyAdapter()),
    ]

    registered_count = 0
    for strategy_name, adapter in adapters:
        if strategy_name not in registry.adapters:
            try:
                registry.register_adapter(strategy_name, adapter)
                registered_count += 1
                logger.info(f"✅ Registered {strategy_name} adapter")
            except Exception as e:
                logger.error(f"❌ Failed to register {strategy_name} adapter: {e}")

    # Discover additional strategies
    discovered_strategies = registry.discover_strategies()

    logger.info(f"🎯 Backtest system initialized:")
    logger.info(f"   - {registered_count} specific adapters registered")
    logger.info(f"   - {len(discovered_strategies)} total strategies discovered")
    logger.info(f"   - Available strategies: {list(discovered_strategies.keys())}")

    return registry


def get_system_status():
    """Get the current status of the backtest system"""
    registry = get_backtest_registry()

    return {
        'total_strategies': len(registry.strategies),
        'registered_adapters': len(registry.adapters),
        'available_strategies': list(registry.strategies.keys()),
        'adapter_strategies': list(registry.adapters.keys())
    }


# Auto-initialize if imported directly
if __name__ == "__main__":
    initialize_backtest_system()
else:
    # Initialize on import
    try:
        initialize_backtest_system()
    except Exception as e:
        logger.warning(f"Failed to auto-initialize backtest system: {e}")


__all__ = [
    'initialize_backtest_system',
    'get_system_status',
    'get_backtest_registry',
    'get_backtest_runner',
    'get_result_processor',
    'get_chart_service',
    'BacktestConfig',
    'BacktestResult',
    'StrategyInfo'
]