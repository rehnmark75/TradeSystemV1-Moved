# core/backtest/strategy_registry.py
"""
Strategy Registry - Dynamic discovery and management of trading strategies

This module provides automatic discovery and registration of all available
trading strategies, providing a unified interface for the backtest system.

Features:
- Automatic strategy discovery from the strategies module
- Metadata extraction for each strategy
- Parameter validation and defaults
- Strategy capability detection (MTF, optimization, etc.)
- Consistent initialization interface
"""

import logging
import inspect
import importlib
from typing import Dict, List, Optional, Any, Type, Tuple
from dataclasses import dataclass
from enum import Enum
import os
import sys

try:
    from core.data_fetcher import DataFetcher
    import config
except ImportError:
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner import config


class StrategyCapability(Enum):
    """Strategy capabilities"""
    MTF_ANALYSIS = "mtf"
    OPTIMIZATION = "optimization"
    SMART_MONEY = "smart_money"
    PARAMETER_SWEEP = "parameter_sweep"
    SIGNAL_VALIDATION = "signal_validation"
    REAL_TIME = "real_time"
    BACKTEST = "backtest"


@dataclass
class StrategyMetadata:
    """Metadata for a registered strategy"""
    name: str
    display_name: str
    description: str
    strategy_class: Type
    capabilities: List[StrategyCapability]
    default_parameters: Dict[str, Any]
    parameter_ranges: Dict[str, Tuple[Any, Any]]
    timeframes_supported: List[str]
    requires_optimization: bool = False
    has_mtf_support: bool = False
    initialization_args: Dict[str, Any] = None


class StrategyRegistry:
    """
    Registry for managing and discovering trading strategies

    Automatically discovers strategies from the strategies module and provides
    a unified interface for initialization and metadata access.
    """

    def __init__(self):
        self.logger = logging.getLogger('strategy_registry')
        self.strategies: Dict[str, StrategyMetadata] = {}
        self.discovery_paths = [
            'core.strategies',
            'forex_scanner.core.strategies'
        ]

        # Auto-discover strategies on initialization
        self.discover_strategies()

    def discover_strategies(self):
        """Automatically discover and register strategies"""
        self.logger.info("🔍 Discovering available strategies...")

        # Manual registration of known strategies with their metadata
        self._register_known_strategies()

        # Attempt automatic discovery
        self._auto_discover_strategies()

        self.logger.info(f"📋 Registered {len(self.strategies)} strategies: {list(self.strategies.keys())}")

    def _register_known_strategies(self):
        """Register known strategies with their metadata"""

        # EMA Strategy
        try:
            from core.strategies.ema_strategy import EMAStrategy
            self.register_strategy(
                name='ema',
                display_name='EMA Strategy',
                description='Exponential Moving Average strategy with pullback entries',
                strategy_class=EMAStrategy,
                capabilities=[
                    StrategyCapability.MTF_ANALYSIS,
                    StrategyCapability.OPTIMIZATION,
                    StrategyCapability.SMART_MONEY,
                    StrategyCapability.BACKTEST,
                    StrategyCapability.REAL_TIME
                ],
                default_parameters={
                    'short_ema': 21,
                    'long_ema': 50,
                    'trend_ema': 200,
                    'confidence_threshold': 0.45,
                    'use_optimal_parameters': True
                },
                parameter_ranges={
                    'short_ema': (8, 34),
                    'long_ema': (34, 89),
                    'trend_ema': (144, 233),
                    'confidence_threshold': (0.3, 0.8)
                },
                timeframes_supported=['5m', '15m', '30m', '1h', '4h'],
                initialization_args={
                    'data_fetcher': 'required',
                    'backtest_mode': True,
                    'use_optimal_parameters': 'optional'
                }
            )
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import EMA strategy: {e}")

        # MACD Strategy
        try:
            from core.strategies.macd_strategy import MACDStrategy
            self.register_strategy(
                name='macd',
                display_name='MACD Strategy',
                description='MACD-based strategy with trend validation',
                strategy_class=MACDStrategy,
                capabilities=[
                    StrategyCapability.MTF_ANALYSIS,
                    StrategyCapability.OPTIMIZATION,
                    StrategyCapability.SMART_MONEY,
                    StrategyCapability.BACKTEST,
                    StrategyCapability.REAL_TIME
                ],
                default_parameters={
                    'fast_length': 12,
                    'slow_length': 26,
                    'signal_length': 9,
                    'confidence_threshold': 0.5,
                    'use_optimized_parameters': True
                },
                parameter_ranges={
                    'fast_length': (8, 21),
                    'slow_length': (21, 34),
                    'signal_length': (5, 13),
                    'confidence_threshold': (0.3, 0.8)
                },
                timeframes_supported=['5m', '15m', '30m', '1h', '4h'],
                initialization_args={
                    'data_fetcher': 'required',
                    'backtest_mode': True,
                    'epic': 'optional',
                    'timeframe': 'optional',
                    'use_optimized_parameters': 'optional'
                }
            )
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import MACD strategy: {e}")

        # KAMA Strategy
        try:
            from core.strategies.kama_strategy import KAMAStrategy
            self.register_strategy(
                name='kama',
                display_name='KAMA Strategy',
                description='Kaufman Adaptive Moving Average strategy',
                strategy_class=KAMAStrategy,
                capabilities=[
                    StrategyCapability.BACKTEST,
                    StrategyCapability.REAL_TIME
                ],
                default_parameters={
                    'kama_period': 21,
                    'fastest_sc': 2,
                    'slowest_sc': 30,
                    'confidence_threshold': 0.5
                },
                parameter_ranges={
                    'kama_period': (10, 34),
                    'fastest_sc': (2, 5),
                    'slowest_sc': (20, 50),
                    'confidence_threshold': (0.3, 0.8)
                },
                timeframes_supported=['15m', '30m', '1h', '4h'],
                initialization_args={}
            )
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import KAMA strategy: {e}")

        # Combined Strategy
        try:
            from core.strategies.combined_strategy import CombinedStrategy
            self.register_strategy(
                name='combined',
                display_name='Combined Strategy',
                description='Consensus-based combination of multiple strategies',
                strategy_class=CombinedStrategy,
                capabilities=[
                    StrategyCapability.BACKTEST,
                    StrategyCapability.REAL_TIME
                ],
                default_parameters={
                    'mode': 'consensus',
                    'min_consensus': 2,
                    'weight_ema': 1.0,
                    'weight_macd': 1.0
                },
                parameter_ranges={
                    'min_consensus': (2, 4),
                    'weight_ema': (0.5, 2.0),
                    'weight_macd': (0.5, 2.0)
                },
                timeframes_supported=['5m', '15m', '30m', '1h', '4h'],
                initialization_args={}
            )
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import Combined strategy: {e}")

        # BB+Supertrend Strategy
        try:
            from core.strategies.bb_supertrend_strategy import BollingerSupertrendStrategy
            self.register_strategy(
                name='bb_supertrend',
                display_name='Bollinger + Supertrend',
                description='Bollinger Bands with Supertrend strategy',
                strategy_class=BollingerSupertrendStrategy,
                capabilities=[
                    StrategyCapability.BACKTEST,
                    StrategyCapability.REAL_TIME
                ],
                default_parameters={
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'supertrend_period': 10,
                    'supertrend_multiplier': 3.0,
                    'config_name': 'default'
                },
                parameter_ranges={
                    'bb_period': (10, 30),
                    'bb_std': (1.5, 2.5),
                    'supertrend_period': (7, 14),
                    'supertrend_multiplier': (2.0, 4.0)
                },
                timeframes_supported=['15m', '30m', '1h', '4h'],
                initialization_args={
                    'config_name': 'optional'
                }
            )
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import BB+Supertrend strategy: {e}")

        # Zero Lag Strategy
        try:
            from core.strategies.zero_lag_strategy import ZeroLagStrategy
            self.register_strategy(
                name='zero_lag',
                display_name='Zero Lag Strategy',
                description='Zero lag moving average strategy',
                strategy_class=ZeroLagStrategy,
                capabilities=[
                    StrategyCapability.BACKTEST,
                    StrategyCapability.REAL_TIME
                ],
                default_parameters={
                    'period': 21,
                    'gain_limit': 50,
                    'confidence_threshold': 0.5
                },
                parameter_ranges={
                    'period': (10, 34),
                    'gain_limit': (20, 100),
                    'confidence_threshold': (0.3, 0.8)
                },
                timeframes_supported=['15m', '30m', '1h', '4h'],
                initialization_args={}
            )
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import Zero Lag strategy: {e}")

        # Scalping Strategy
        try:
            from core.strategies.scalping_strategy import ScalpingStrategy
            self.register_strategy(
                name='scalping',
                display_name='Scalping Strategy',
                description='High-frequency scalping strategy',
                strategy_class=ScalpingStrategy,
                capabilities=[
                    StrategyCapability.BACKTEST,
                    StrategyCapability.REAL_TIME
                ],
                default_parameters={
                    'fast_ema': 8,
                    'slow_ema': 21,
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                },
                parameter_ranges={
                    'fast_ema': (5, 13),
                    'slow_ema': (13, 34),
                    'rsi_period': (10, 21),
                    'rsi_overbought': (65, 80),
                    'rsi_oversold': (20, 35)
                },
                timeframes_supported=['1m', '5m', '15m'],
                initialization_args={}
            )
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import Scalping strategy: {e}")

        # SMC Strategy
        try:
            from core.strategies.smc_strategy_fast import SMCStrategyFast
            self.register_strategy(
                name='smc',
                display_name='Smart Money Concepts',
                description='Smart Money Concepts strategy',
                strategy_class=SMCStrategyFast,
                capabilities=[
                    StrategyCapability.SMART_MONEY,
                    StrategyCapability.BACKTEST,
                    StrategyCapability.REAL_TIME
                ],
                default_parameters={
                    'swing_length': 10,
                    'break_threshold': 0.1,
                    'confirmation_bars': 3
                },
                parameter_ranges={
                    'swing_length': (5, 20),
                    'break_threshold': (0.05, 0.2),
                    'confirmation_bars': (1, 5)
                },
                timeframes_supported=['15m', '30m', '1h', '4h'],
                initialization_args={}
            )
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import SMC strategy: {e}")

    def _auto_discover_strategies(self):
        """Attempt automatic discovery of strategies"""
        for path in self.discovery_paths:
            try:
                self._scan_module_for_strategies(path)
            except Exception as e:
                self.logger.debug(f"⚠️ Could not scan {path}: {e}")

    def _scan_module_for_strategies(self, module_path: str):
        """Scan a module for strategy classes"""
        try:
            module = importlib.import_module(module_path)
            module_dir = os.path.dirname(module.__file__)

            for filename in os.listdir(module_dir):
                if filename.endswith('_strategy.py') and not filename.startswith('__'):
                    strategy_name = filename[:-3]  # Remove .py

                    # Skip if already registered
                    simple_name = strategy_name.replace('_strategy', '')
                    if simple_name in self.strategies:
                        continue

                    try:
                        strategy_module = importlib.import_module(f"{module_path}.{strategy_name}")
                        self._extract_strategy_from_module(strategy_module, simple_name)
                    except Exception as e:
                        self.logger.debug(f"⚠️ Could not import {strategy_name}: {e}")

        except Exception as e:
            self.logger.debug(f"⚠️ Could not scan module {module_path}: {e}")

    def _extract_strategy_from_module(self, module, strategy_name: str):
        """Extract strategy class from module"""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (name.endswith('Strategy') and
                hasattr(obj, 'detect_signal') and
                obj.__module__ == module.__name__):

                # Auto-register with basic metadata
                self.register_strategy(
                    name=strategy_name,
                    display_name=name,
                    description=f"Auto-discovered {name}",
                    strategy_class=obj,
                    capabilities=[StrategyCapability.BACKTEST],
                    default_parameters={},
                    parameter_ranges={},
                    timeframes_supported=['15m'],
                    initialization_args={}
                )
                break

    def register_strategy(
        self,
        name: str,
        display_name: str,
        description: str,
        strategy_class: Type,
        capabilities: List[StrategyCapability],
        default_parameters: Dict[str, Any],
        parameter_ranges: Dict[str, Tuple[Any, Any]],
        timeframes_supported: List[str],
        initialization_args: Dict[str, Any] = None
    ):
        """Register a strategy with metadata"""

        # Analyze strategy class for capabilities
        has_mtf = hasattr(strategy_class, 'detect_signal_with_mtf')
        requires_opt = 'use_optimal_parameters' in (initialization_args or {})

        metadata = StrategyMetadata(
            name=name,
            display_name=display_name,
            description=description,
            strategy_class=strategy_class,
            capabilities=capabilities,
            default_parameters=default_parameters,
            parameter_ranges=parameter_ranges,
            timeframes_supported=timeframes_supported,
            requires_optimization=requires_opt,
            has_mtf_support=has_mtf,
            initialization_args=initialization_args or {}
        )

        self.strategies[name] = metadata
        self.logger.debug(f"✅ Registered strategy: {name} ({display_name})")

    def get_strategy(self, name: str) -> Optional[StrategyMetadata]:
        """Get strategy metadata by name"""
        return self.strategies.get(name)

    def get_all_strategies(self) -> Dict[str, StrategyMetadata]:
        """Get all registered strategies"""
        return self.strategies.copy()

    def get_strategies_with_capability(self, capability: StrategyCapability) -> List[str]:
        """Get strategies that have a specific capability"""
        return [
            name for name, metadata in self.strategies.items()
            if capability in metadata.capabilities
        ]

    def create_strategy_instance(
        self,
        name: str,
        data_fetcher: DataFetcher,
        epic: str = None,
        timeframe: str = '15m',
        use_optimal_parameters: bool = True,
        backtest_mode: bool = True,
        **kwargs
    ):
        """
        Create an instance of the specified strategy

        Args:
            name: Strategy name
            data_fetcher: Data fetcher instance
            epic: Epic for optimization (optional)
            timeframe: Timeframe for analysis
            use_optimal_parameters: Whether to use optimal parameters
            backtest_mode: Whether this is for backtesting
            **kwargs: Additional parameters

        Returns:
            Strategy instance or None if creation fails
        """
        metadata = self.get_strategy(name)
        if not metadata:
            self.logger.error(f"❌ Unknown strategy: {name}")
            return None

        try:
            # Build initialization arguments
            init_args = {}

            # Handle required arguments based on metadata
            if 'data_fetcher' in metadata.initialization_args:
                init_args['data_fetcher'] = data_fetcher

            if backtest_mode and 'backtest_mode' in metadata.initialization_args:
                init_args['backtest_mode'] = True

            if epic and 'epic' in metadata.initialization_args:
                init_args['epic'] = epic

            if timeframe and 'timeframe' in metadata.initialization_args:
                init_args['timeframe'] = timeframe

            # Handle optimization parameters
            if use_optimal_parameters and metadata.requires_optimization:
                if 'use_optimal_parameters' in metadata.initialization_args:
                    init_args['use_optimal_parameters'] = True
                elif 'use_optimized_parameters' in metadata.initialization_args:
                    init_args['use_optimized_parameters'] = True

            # Add any specific arguments for this strategy
            for arg_name, arg_requirement in metadata.initialization_args.items():
                if arg_name in kwargs:
                    init_args[arg_name] = kwargs[arg_name]
                elif arg_requirement == 'required' and arg_name not in init_args:
                    # Set reasonable defaults for required args not provided
                    if arg_name == 'config_name':
                        init_args[arg_name] = 'default'

            # Create the strategy instance
            strategy = metadata.strategy_class(**init_args)

            self.logger.debug(f"✅ Created {name} strategy instance")
            return strategy

        except Exception as e:
            self.logger.error(f"❌ Failed to create {name} strategy: {e}")
            return None

    def validate_parameters(self, name: str, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters for a strategy

        Args:
            name: Strategy name
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        metadata = self.get_strategy(name)
        if not metadata:
            return False, [f"Unknown strategy: {name}"]

        errors = []

        for param_name, param_value in parameters.items():
            if param_name in metadata.parameter_ranges:
                min_val, max_val = metadata.parameter_ranges[param_name]

                try:
                    # Convert to appropriate type for comparison
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        param_value = float(param_value)
                        if not (min_val <= param_value <= max_val):
                            errors.append(
                                f"{param_name}: {param_value} not in range [{min_val}, {max_val}]"
                            )
                except (ValueError, TypeError):
                    errors.append(f"{param_name}: Invalid value type")

        return len(errors) == 0, errors

    def get_strategy_help(self, name: str = None) -> str:
        """Get help information for strategies"""
        if name:
            metadata = self.get_strategy(name)
            if not metadata:
                return f"❌ Unknown strategy: {name}"

            help_text = f"""
📊 {metadata.display_name}
{'-' * (len(metadata.display_name) + 4)}

Description: {metadata.description}

Capabilities:
{chr(10).join(f"  • {cap.value}" for cap in metadata.capabilities)}

Default Parameters:
{chr(10).join(f"  • {k}: {v}" for k, v in metadata.default_parameters.items())}

Parameter Ranges:
{chr(10).join(f"  • {k}: {v[0]} - {v[1]}" for k, v in metadata.parameter_ranges.items())}

Supported Timeframes: {', '.join(metadata.timeframes_supported)}

MTF Support: {'✅' if metadata.has_mtf_support else '❌'}
Optimization: {'✅' if metadata.requires_optimization else '❌'}
"""
            return help_text.strip()
        else:
            # List all strategies
            help_text = "📋 Available Strategies:\n" + "=" * 25 + "\n\n"

            for name, metadata in self.strategies.items():
                help_text += f"• {name:<15} - {metadata.display_name}\n"
                help_text += f"  {metadata.description}\n"
                help_text += f"  Capabilities: {', '.join(cap.value for cap in metadata.capabilities)}\n\n"

            return help_text.strip()

    def get_compatible_strategies(self, timeframe: str) -> List[str]:
        """Get strategies compatible with a specific timeframe"""
        compatible = []
        for name, metadata in self.strategies.items():
            if timeframe in metadata.timeframes_supported:
                compatible.append(name)
        return compatible

    def export_registry(self) -> Dict:
        """Export registry as JSON-serializable dict"""
        export_data = {}
        for name, metadata in self.strategies.items():
            export_data[name] = {
                'display_name': metadata.display_name,
                'description': metadata.description,
                'capabilities': [cap.value for cap in metadata.capabilities],
                'default_parameters': metadata.default_parameters,
                'parameter_ranges': metadata.parameter_ranges,
                'timeframes_supported': metadata.timeframes_supported,
                'has_mtf_support': metadata.has_mtf_support,
                'requires_optimization': metadata.requires_optimization
            }
        return export_data


# Global registry instance
_registry = None

def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry instance"""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry