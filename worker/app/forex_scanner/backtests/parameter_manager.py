# ============================================================================
# backtests/parameter_manager.py - Unified Parameter Management System
# ============================================================================

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# Import optimization service
try:
    from optimization.optimal_parameter_service import OptimalParameterService
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.optimization.optimal_parameter_service import OptimalParameterService
        OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OPTIMIZATION_AVAILABLE = False

try:
    import config
except ImportError:
    from forex_scanner import config

try:
    from configdata import config as strategy_config
except ImportError:
    strategy_config = None


class ParameterSource(Enum):
    """Source of parameter values"""
    DATABASE_OPTIMAL = "database_optimal"
    STRATEGY_CONFIG = "strategy_config"
    USER_PROVIDED = "user_provided"
    FALLBACK_DEFAULT = "fallback_default"
    MARKET_ADAPTIVE = "market_adaptive"


@dataclass
class ParameterInfo:
    """Information about a parameter"""
    name: str
    value: Any
    source: ParameterSource
    description: str = ""
    validation_passed: bool = True
    validation_message: str = ""
    alternatives: List[Any] = field(default_factory=list)


@dataclass
class ParameterSet:
    """Complete set of parameters for a strategy"""
    strategy_name: str
    epic: Optional[str] = None
    parameters: Dict[str, ParameterInfo] = field(default_factory=dict)
    source_priority: List[ParameterSource] = field(default_factory=list)
    market_conditions: Optional[Dict[str, Any]] = None
    confidence_score: float = 1.0
    performance_score: Optional[float] = None

    def get_value(self, param_name: str, default: Any = None) -> Any:
        """Get parameter value"""
        if param_name in self.parameters:
            return self.parameters[param_name].value
        return default

    def get_all_values(self) -> Dict[str, Any]:
        """Get all parameter values as a dictionary"""
        return {name: info.value for name, info in self.parameters.items()}

    def get_source_info(self, param_name: str) -> Optional[ParameterSource]:
        """Get the source of a parameter"""
        if param_name in self.parameters:
            return self.parameters[param_name].source
        return None


class ParameterManager:
    """Unified parameter management system for all strategies"""

    def __init__(self, use_optimal_parameters: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_optimal_parameters = use_optimal_parameters and OPTIMIZATION_AVAILABLE
        self.optimal_service = OptimalParameterService() if self.use_optimal_parameters else None

        # Cache for optimal parameters to avoid repeated database calls
        self._optimal_cache = {}

        # Default parameter definitions for all strategies
        self._default_parameters = self._initialize_default_parameters()

        self.logger.info(f"ParameterManager initialized")
        self.logger.info(f"  Database optimization: {'✅ ENABLED' if self.use_optimal_parameters else '❌ DISABLED'}")

    def _initialize_default_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default parameter definitions for all strategies"""
        return {
            'ema': {
                'ema_config': {
                    'default': 'aggressive',
                    'type': 'select',
                    'options': ['conservative', 'aggressive', 'scalping'],
                    'description': 'EMA configuration preset'
                },
                'min_confidence': {
                    'default': 0.7,
                    'type': 'number',
                    'min': 0.1,
                    'max': 1.0,
                    'description': 'Minimum confidence threshold'
                },
                'enable_smart_money': {
                    'default': False,
                    'type': 'boolean',
                    'description': 'Enable Smart Money Concepts analysis'
                },
                'use_optimal_parameters': {
                    'default': True,
                    'type': 'boolean',
                    'description': 'Use database-optimized parameters'
                }
            },
            'macd': {
                'min_confidence': {
                    'default': 0.7,
                    'type': 'number',
                    'min': 0.1,
                    'max': 1.0,
                    'description': 'Minimum confidence threshold'
                },
                'enable_smart_money': {
                    'default': False,
                    'type': 'boolean',
                    'description': 'Enable Smart Money Concepts analysis'
                },
                'enable_forex_integration': {
                    'default': True,
                    'type': 'boolean',
                    'description': 'Enable forex-specific optimizations'
                },
                'use_optimal_parameters': {
                    'default': True,
                    'type': 'boolean',
                    'description': 'Use database-optimized parameters'
                }
            },
            'combined': {
                'combination_mode': {
                    'default': 'consensus',
                    'type': 'select',
                    'options': ['consensus', 'weighted'],
                    'description': 'Strategy combination method'
                },
                'min_confidence': {
                    'default': 0.7,
                    'type': 'number',
                    'min': 0.1,
                    'max': 1.0,
                    'description': 'Minimum confidence threshold'
                }
            },
            'bb_supertrend': {
                'bb_config': {
                    'default': 'default',
                    'type': 'select',
                    'options': ['default', 'tight', 'wide'],
                    'description': 'Bollinger Bands configuration'
                }
            },
            'kama': {
                'kama_period': {
                    'default': 14,
                    'type': 'number',
                    'min': 5,
                    'max': 50,
                    'description': 'KAMA calculation period'
                }
            },
            'smc': {
                'structure_detection': {
                    'default': True,
                    'type': 'boolean',
                    'description': 'Enable market structure detection'
                }
            },
            'ichimoku': {
                'tenkan_period': {
                    'default': 9,
                    'type': 'number',
                    'min': 5,
                    'max': 20,
                    'description': 'Tenkan-sen period'
                },
                'kijun_period': {
                    'default': 26,
                    'type': 'number',
                    'min': 10,
                    'max': 50,
                    'description': 'Kijun-sen period'
                },
                'senkou_b_period': {
                    'default': 52,
                    'type': 'number',
                    'min': 20,
                    'max': 100,
                    'description': 'Senkou Span B period'
                }
            },
            'scalping': {
                'scalping_mode': {
                    'default': 'aggressive',
                    'type': 'select',
                    'options': ['conservative', 'moderate', 'aggressive'],
                    'description': 'Scalping aggressiveness'
                },
                'min_pip_target': {
                    'default': 5,
                    'type': 'number',
                    'min': 2,
                    'max': 20,
                    'description': 'Minimum pip target'
                }
            },
            'zero_lag': {
                'zero_lag_period': {
                    'default': 21,
                    'type': 'number',
                    'min': 10,
                    'max': 50,
                    'description': 'Zero-lag EMA period'
                }
            },
            'mean_reversion': {
                'lookback_period': {
                    'default': 20,
                    'type': 'number',
                    'min': 10,
                    'max': 50,
                    'description': 'Mean reversion lookback period'
                },
                'deviation_threshold': {
                    'default': 2.0,
                    'type': 'number',
                    'min': 1.0,
                    'max': 4.0,
                    'description': 'Standard deviation threshold'
                }
            }
        }

    def get_parameters(
        self,
        strategy_name: str,
        epic: Optional[str] = None,
        user_parameters: Optional[Dict[str, Any]] = None,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> ParameterSet:
        """Get complete parameter set for a strategy with intelligent source selection"""

        self.logger.info(f"🔧 Getting parameters for {strategy_name} (epic: {epic})")

        # Initialize parameter set
        param_set = ParameterSet(
            strategy_name=strategy_name,
            epic=epic,
            market_conditions=market_conditions,
            source_priority=[
                ParameterSource.USER_PROVIDED,
                ParameterSource.DATABASE_OPTIMAL,
                ParameterSource.STRATEGY_CONFIG,
                ParameterSource.FALLBACK_DEFAULT
            ]
        )

        # Get available parameter definitions for this strategy
        strategy_defaults = self._default_parameters.get(strategy_name, {})

        # Try different parameter sources in priority order
        for param_name, param_def in strategy_defaults.items():
            param_info = self._get_parameter_from_sources(
                param_name=param_name,
                param_def=param_def,
                strategy_name=strategy_name,
                epic=epic,
                user_parameters=user_parameters,
                source_priority=param_set.source_priority
            )

            param_set.parameters[param_name] = param_info

        # Add any user parameters not in defaults
        if user_parameters:
            for param_name, param_value in user_parameters.items():
                if param_name not in param_set.parameters:
                    param_set.parameters[param_name] = ParameterInfo(
                        name=param_name,
                        value=param_value,
                        source=ParameterSource.USER_PROVIDED,
                        description="User-provided parameter"
                    )

        # Calculate confidence score based on sources used
        param_set.confidence_score = self._calculate_confidence_score(param_set)

        # Get performance score if using optimal parameters
        if self.use_optimal_parameters and epic:
            param_set.performance_score = self._get_performance_score(strategy_name, epic)

        self.logger.info(f"✅ Parameter set ready: {len(param_set.parameters)} parameters")
        self.logger.info(f"   Confidence: {param_set.confidence_score:.1%}")
        if param_set.performance_score:
            self.logger.info(f"   Performance score: {param_set.performance_score:.3f}")

        return param_set

    def _get_parameter_from_sources(
        self,
        param_name: str,
        param_def: Dict[str, Any],
        strategy_name: str,
        epic: Optional[str],
        user_parameters: Optional[Dict[str, Any]],
        source_priority: List[ParameterSource]
    ) -> ParameterInfo:
        """Get parameter value from available sources in priority order"""

        for source in source_priority:
            try:
                if source == ParameterSource.USER_PROVIDED and user_parameters:
                    if param_name in user_parameters:
                        value = user_parameters[param_name]
                        if self._validate_parameter(param_name, value, param_def):
                            return ParameterInfo(
                                name=param_name,
                                value=value,
                                source=source,
                                description=param_def.get('description', ''),
                                validation_passed=True
                            )

                elif source == ParameterSource.DATABASE_OPTIMAL and self.use_optimal_parameters and epic:
                    optimal_value = self._get_optimal_parameter(strategy_name, epic, param_name)
                    if optimal_value is not None:
                        return ParameterInfo(
                            name=param_name,
                            value=optimal_value,
                            source=source,
                            description=param_def.get('description', ''),
                            validation_passed=True
                        )

                elif source == ParameterSource.STRATEGY_CONFIG:
                    config_value = self._get_strategy_config_parameter(strategy_name, param_name)
                    if config_value is not None:
                        return ParameterInfo(
                            name=param_name,
                            value=config_value,
                            source=source,
                            description=param_def.get('description', ''),
                            validation_passed=True
                        )

                elif source == ParameterSource.FALLBACK_DEFAULT:
                    default_value = param_def.get('default')
                    if default_value is not None:
                        return ParameterInfo(
                            name=param_name,
                            value=default_value,
                            source=source,
                            description=param_def.get('description', ''),
                            validation_passed=True
                        )

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get {param_name} from {source}: {e}")
                continue

        # Fallback to None if all sources fail
        return ParameterInfo(
            name=param_name,
            value=None,
            source=ParameterSource.FALLBACK_DEFAULT,
            description=param_def.get('description', ''),
            validation_passed=False,
            validation_message="No valid source found"
        )

    def _get_optimal_parameter(self, strategy_name: str, epic: str, param_name: str) -> Optional[Any]:
        """Get optimal parameter value from database"""
        cache_key = f"{strategy_name}_{epic}"

        # Check cache first
        if cache_key not in self._optimal_cache:
            try:
                optimal_params = self.optimal_service.get_epic_parameters(epic)
                self._optimal_cache[cache_key] = optimal_params
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get optimal parameters for {epic}: {e}")
                return None

        optimal_params = self._optimal_cache.get(cache_key)
        if not optimal_params:
            return None

        # Map parameter names to optimal parameter attributes
        param_mapping = {
            'min_confidence': 'confidence_threshold',
            'ema_config': 'ema_config',
            'stop_loss_pips': 'stop_loss_pips',
            'take_profit_pips': 'take_profit_pips',
            'risk_reward_ratio': 'risk_reward_ratio'
        }

        if param_name in param_mapping:
            return getattr(optimal_params, param_mapping[param_name], None)

        return None

    def _get_strategy_config_parameter(self, strategy_name: str, param_name: str) -> Optional[Any]:
        """Get parameter from strategy configuration"""
        if not strategy_config:
            return None

        # Map strategy names to config attributes
        config_mapping = {
            'ema': {
                'ema_config': 'ACTIVE_EMA_CONFIG',
                'min_confidence': 'MIN_CONFIDENCE'
            },
            'macd': {
                'min_confidence': 'MIN_CONFIDENCE'
            }
        }

        strategy_map = config_mapping.get(strategy_name, {})
        config_attr = strategy_map.get(param_name)

        if config_attr and hasattr(strategy_config, config_attr):
            return getattr(strategy_config, config_attr)

        # Also check main config
        if hasattr(config, param_name.upper()):
            return getattr(config, param_name.upper())

        return None

    def _get_performance_score(self, strategy_name: str, epic: str) -> Optional[float]:
        """Get performance score for optimal parameters"""
        cache_key = f"{strategy_name}_{epic}"
        optimal_params = self._optimal_cache.get(cache_key)

        if optimal_params and hasattr(optimal_params, 'performance_score'):
            return optimal_params.performance_score

        return None

    def _validate_parameter(self, param_name: str, value: Any, param_def: Dict[str, Any]) -> bool:
        """Validate parameter value against definition"""
        param_type = param_def.get('type', 'text')

        try:
            if param_type == 'number':
                value = float(value)
                min_val = param_def.get('min')
                max_val = param_def.get('max')

                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False

            elif param_type == 'select':
                options = param_def.get('options', [])
                if options and value not in options:
                    return False

            elif param_type == 'boolean':
                if not isinstance(value, bool) and value not in [0, 1, '0', '1', 'true', 'false']:
                    return False

            return True

        except (ValueError, TypeError):
            return False

    def _calculate_confidence_score(self, param_set: ParameterSet) -> float:
        """Calculate confidence score based on parameter sources"""
        if not param_set.parameters:
            return 0.0

        source_weights = {
            ParameterSource.DATABASE_OPTIMAL: 1.0,
            ParameterSource.USER_PROVIDED: 0.9,
            ParameterSource.STRATEGY_CONFIG: 0.7,
            ParameterSource.FALLBACK_DEFAULT: 0.5,
            ParameterSource.MARKET_ADAPTIVE: 0.8
        }

        total_weight = 0.0
        total_params = 0

        for param_info in param_set.parameters.values():
            if param_info.validation_passed:
                weight = source_weights.get(param_info.source, 0.5)
                total_weight += weight
                total_params += 1

        return total_weight / total_params if total_params > 0 else 0.0

    def get_strategy_parameter_definitions(self, strategy_name: str) -> Dict[str, Any]:
        """Get parameter definitions for a strategy (for UI generation)"""
        return self._default_parameters.get(strategy_name, {})

    def get_all_strategies(self) -> List[str]:
        """Get list of all supported strategies"""
        return list(self._default_parameters.keys())

    def clear_cache(self):
        """Clear the optimal parameter cache"""
        self._optimal_cache.clear()
        self.logger.info("🗑️ Parameter cache cleared")

    def validate_parameter_set(self, param_set: ParameterSet) -> Dict[str, str]:
        """Validate entire parameter set and return validation errors"""
        errors = {}

        strategy_defaults = self._default_parameters.get(param_set.strategy_name, {})

        for param_name, param_info in param_set.parameters.items():
            if param_name in strategy_defaults:
                param_def = strategy_defaults[param_name]
                if not self._validate_parameter(param_name, param_info.value, param_def):
                    errors[param_name] = f"Invalid value for {param_name}: {param_info.value}"

        return errors

    def get_parameter_info_summary(self, param_set: ParameterSet) -> str:
        """Get a summary of parameter sources and values"""
        summary_lines = []
        summary_lines.append(f"Parameter Summary for {param_set.strategy_name}:")
        summary_lines.append(f"  Epic: {param_set.epic or 'ALL'}")
        summary_lines.append(f"  Confidence: {param_set.confidence_score:.1%}")

        if param_set.performance_score:
            summary_lines.append(f"  Performance Score: {param_set.performance_score:.3f}")

        summary_lines.append("  Parameters:")

        for param_name, param_info in param_set.parameters.items():
            source_emoji = {
                ParameterSource.DATABASE_OPTIMAL: "🎯",
                ParameterSource.USER_PROVIDED: "👤",
                ParameterSource.STRATEGY_CONFIG: "⚙️",
                ParameterSource.FALLBACK_DEFAULT: "📋",
                ParameterSource.MARKET_ADAPTIVE: "🌐"
            }

            emoji = source_emoji.get(param_info.source, "❓")
            status = "✅" if param_info.validation_passed else "❌"

            summary_lines.append(f"    {emoji} {param_name}: {param_info.value} {status}")

        return "\n".join(summary_lines)