# core/strategies/strategy_registry.py
"""
Strategy Registry Pattern

Central registry for trading strategies. Strategies self-register on import,
making it easy to add new strategies without modifying SignalDetector.

Usage:
    # Get the singleton registry
    registry = StrategyRegistry.get_instance()

    # Get enabled strategies from database
    enabled = registry.get_enabled_strategies()

    # Get a strategy instance
    strategy = registry.get_strategy('SMC_SIMPLE', db_manager, data_fetcher)

Adding a new strategy:
    1. Create strategy class implementing StrategyInterface
    2. Add @register_strategy('STRATEGY_NAME') decorator
    3. Enable in database (strategy_config.enabled_strategies table)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any
from functools import wraps

from .signal_result import SignalResult

logger = logging.getLogger(__name__)


class StrategyInterface(ABC):
    """
    Interface that all strategies must implement.

    This ensures strategies have a consistent API for the SignalDetector.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Unique name for this strategy (e.g., 'SMC_SIMPLE')"""
        pass

    @abstractmethod
    def detect_signal(self, **kwargs) -> Optional[SignalResult]:
        """
        Detect trading signal.

        Returns:
            SignalResult dict if detected, None otherwise.
            See core/strategies/signal_result.py for the contract.
            Required fields: signal_type, strategy, epic, entry_price,
            risk_pips, reward_pips, confidence_score.
        """
        pass

    @abstractmethod
    def get_required_timeframes(self) -> List[str]:
        """
        Get list of timeframes this strategy requires.

        Returns:
            List of timeframe strings (e.g., ['4h', '1h', '15m'])
        """
        pass

    def reset_cooldowns(self) -> None:
        """Reset any internal cooldowns (optional, for backtesting)"""
        pass

    def flush_rejections(self) -> None:
        """Flush any pending rejections to database (optional)"""
        pass

    def get_lpf_extra_context(self, signal: Dict) -> Optional[Dict]:
        """Return strategy-specific data for the LPF (opt-in).

        Override to attach richer context — e.g. pair win-rate, setup
        strength, pattern confidence. Returned dict is stored in
        signal['_lpf_strategy_context'] before TradeValidator calls LPF.
        Default returns None (no extra context).
        """
        return None


class StrategyRegistry:
    """
    Central registry for trading strategies.

    Singleton pattern ensures one registry instance across the application.
    Strategies are loaded lazily on first access.
    """

    _instance: Optional['StrategyRegistry'] = None
    _strategies: Dict[str, Type] = {}
    _instances: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._db_manager = None
        self._data_fetcher = None
        self._config = None
        self.logger = logging.getLogger(__name__)
        self.logger.info("📋 Strategy Registry initialized")

    @classmethod
    def get_instance(cls) -> 'StrategyRegistry':
        """Get the singleton registry instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register(cls, name: str, strategy_class: Type) -> None:
        """
        Register a strategy class.

        Args:
            name: Strategy name (e.g., 'SMC_SIMPLE')
            strategy_class: Strategy class to register
        """
        cls._strategies[name.upper()] = strategy_class
        logger.info(f"📝 Registered strategy: {name}")

    def set_dependencies(self, db_manager=None, data_fetcher=None, config=None) -> None:
        """
        Set dependencies needed for strategy initialization.

        Args:
            db_manager: DatabaseManager instance
            data_fetcher: DataFetcher instance
            config: Config module or object
        """
        if db_manager:
            self._db_manager = db_manager
        if data_fetcher:
            self._data_fetcher = data_fetcher
        if config:
            self._config = config

    def get_strategy(self, name: str, **kwargs) -> Optional[Any]:
        """
        Get or create a strategy instance.

        Args:
            name: Strategy name
            **kwargs: Override dependencies for this instance

        Returns:
            Strategy instance or None if not found
        """
        name = name.upper()

        # Check if already instantiated
        if name in self._instances:
            return self._instances[name]

        # Check if registered
        if name not in self._strategies:
            self.logger.warning(f"⚠️ Strategy not registered: {name}")
            return None

        # Get dependencies
        db_manager = kwargs.get('db_manager', self._db_manager)
        data_fetcher = kwargs.get('data_fetcher', self._data_fetcher)
        config = kwargs.get('config', self._config)

        try:
            strategy_class = self._strategies[name]
            instance = strategy_class(
                config=config,
                db_manager=db_manager,
                logger=self.logger,
            )
            self._instances[name] = instance
            self.logger.info(f"✅ Strategy instantiated: {name}")
            return instance

        except Exception as e:
            self.logger.error(f"❌ Failed to instantiate strategy {name}: {e}")
            raise

    def get_enabled_strategies(self) -> List[str]:
        """
        Get list of enabled strategy names from database.

        Falls back to config file if database unavailable.

        Returns:
            List of enabled strategy names
        """
        # Try to get from database first
        if self._db_manager:
            try:
                from forex_scanner.services.scanner_config_service import get_scanner_config
                config_service = get_scanner_config()
                if hasattr(config_service, 'get_enabled_strategies'):
                    return config_service.get_enabled_strategies()
            except Exception as e:
                self.logger.debug(f"Could not get enabled strategies from DB: {e}")

        # Fallback: all registered strategies are considered enabled
        registered = list(self._strategies.keys())
        if registered:
            return registered
        self.logger.warning("⚠️ No strategies registered, returning empty list")
        return []

    def get_registered_strategies(self) -> List[str]:
        """Get list of all registered strategy names"""
        return list(self._strategies.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a strategy is registered"""
        return name.upper() in self._strategies

    def clear_instances(self) -> None:
        """Clear all strategy instances (useful for testing)"""
        self._instances.clear()
        self.logger.info("🗑️ Cleared all strategy instances")


def register_strategy(name: str):
    """
    Decorator to register a strategy class.

    Usage:
        @register_strategy('MY_STRATEGY')
        class MyStrategy(StrategyInterface):
            ...
    """
    def decorator(cls):
        StrategyRegistry.register(name, cls)
        return cls
    return decorator


# Auto-register SMC Simple on module import
def _auto_register_smc_simple():
    """Auto-register SMC Simple strategy if available"""
    try:
        from .smc_simple_strategy import SMCSimpleStrategy, create_smc_simple_strategy

        # Create a wrapper class that implements StrategyInterface
        class SMCSimpleStrategyWrapper:
            """Wrapper to make SMC Simple compatible with registry"""

            def __init__(self, config=None, db_manager=None, logger=None):
                self._strategy = create_smc_simple_strategy(
                    config=config,
                    db_manager=db_manager,
                    logger=logger
                )

            @property
            def strategy_name(self) -> str:
                return 'SMC_SIMPLE'

            def detect_signal(self, **kwargs) -> Optional[Dict]:
                return self._strategy.detect_signal(**kwargs)

            def get_required_timeframes(self) -> List[str]:
                if hasattr(self._strategy, 'get_required_timeframes'):
                    return self._strategy.get_required_timeframes()
                # Scalp mode uses 1h/5m/1m; non-scalp uses 4h/15m/5m
                if getattr(self._strategy, 'scalp_mode_enabled', False):
                    return ['1h', '5m', '1m']
                return ['4h', '15m', '5m']

            def reset_cooldowns(self) -> None:
                if hasattr(self._strategy, 'reset_cooldowns'):
                    self._strategy.reset_cooldowns()

            def flush_rejections(self) -> None:
                if hasattr(self._strategy, 'flush_rejections'):
                    self._strategy.flush_rejections()

        StrategyRegistry.register('SMC_SIMPLE', SMCSimpleStrategyWrapper)
        logger.debug("✅ Auto-registered SMC_SIMPLE strategy")

    except ImportError as e:
        logger.warning(f"⚠️ Could not auto-register SMC Simple: {e}")


# Run auto-registration when module is imported
_auto_register_smc_simple()


# All strategies self-register via @register_strategy decorator when their
# modules are imported. core/strategies/__init__.py imports every active
# strategy module, which fires the decorators — no explicit auto-register
# calls are needed here (and adding them causes circular imports since they
# run while the strategy module is still being loaded).
