# core/backtest/backtest_config.py
"""
Unified Backtest Configuration System

This module provides centralized configuration management for all backtest
operations, ensuring consistency and ease of use across the system.

Features:
- Centralized configuration management
- Template-based configuration presets
- Validation and error checking
- Environment-specific configurations
- Import/export capabilities
- Configuration inheritance and overrides
"""

import logging
import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import copy

try:
    from core.backtest.unified_backtest_engine import BacktestMode
    from core.backtest.parameter_manager import OptimizationMethod
    import config
except ImportError:
    from forex_scanner.core.backtest.unified_backtest_engine import BacktestMode
    from forex_scanner.core.backtest.parameter_manager import OptimizationMethod
    from forex_scanner import config


class ConfigValidationLevel(Enum):
    """Configuration validation levels"""
    STRICT = "strict"      # All parameters must be valid
    LENIENT = "lenient"    # Warnings for invalid parameters
    MINIMAL = "minimal"    # Only check critical parameters


@dataclass
class SmartMoneyConfig:
    """Smart Money analysis configuration"""
    enabled: bool = False
    market_structure_analysis: bool = True
    order_flow_analysis: bool = True
    supply_demand_zones: bool = True
    confluence_scoring: bool = True
    min_confluence_score: float = 0.6


@dataclass
class OptimizationConfig:
    """Parameter optimization configuration"""
    enabled: bool = False
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    scoring_metric: str = "win_rate"
    max_combinations: int = 100
    cross_validation_splits: int = 1
    parameter_ranges: Dict[str, str] = field(default_factory=dict)
    export_results: bool = True
    export_path: Optional[str] = None


@dataclass
class PerformanceConfig:
    """Performance analysis configuration"""
    include_trailing_stop: bool = True
    target_pips: float = 15.0
    initial_stop_pips: float = 10.0
    breakeven_trigger: float = 8.0
    profit_protection_trigger: float = 15.0
    profit_protection_level: float = 10.0
    trailing_ratio: float = 0.5
    lookback_bars: int = 96


@dataclass
class OutputConfig:
    """Output and reporting configuration"""
    show_signals: bool = False
    max_signals_display: int = 20
    show_performance_details: bool = True
    show_strategy_comparison: bool = False
    export_format: Optional[str] = None  # json, csv, html
    export_path: Optional[str] = None
    verbose: bool = False
    log_level: str = "INFO"


@dataclass
class ValidationConfig:
    """Signal validation configuration"""
    enabled: bool = False
    target_timestamp: Optional[str] = None
    show_raw_data: bool = False
    show_calculations: bool = True
    show_decision_tree: bool = True
    data_context_bars: int = 100


@dataclass
class UnifiedBacktestConfig:
    """
    Unified configuration for all backtest operations

    This class consolidates all configuration options into a single,
    consistent structure that can be used across the entire system.
    """

    # Core execution parameters
    mode: BacktestMode = BacktestMode.SINGLE_STRATEGY
    strategies: List[str] = field(default_factory=lambda: ['ema'])
    epics: List[str] = field(default_factory=lambda: ['CS.D.EURUSD.CEEM.IP'])
    timeframes: List[str] = field(default_factory=lambda: ['15m'])
    days: int = 7

    # Strategy parameters
    use_optimal_parameters: bool = True
    override_parameters: Dict[str, Any] = field(default_factory=dict)

    # Analysis configurations
    smart_money: SmartMoneyConfig = field(default_factory=SmartMoneyConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # System configuration
    validation_level: ConfigValidationLevel = ConfigValidationLevel.LENIENT
    config_name: Optional[str] = None
    created_at: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

        # Ensure lists are not empty
        if not self.strategies:
            self.strategies = ['ema']
        if not self.epics:
            self.epics = ['CS.D.EURUSD.CEEM.IP']
        if not self.timeframes:
            self.timeframes = ['15m']


class ConfigTemplates:
    """Predefined configuration templates for common use cases"""

    @staticmethod
    def quick_test() -> UnifiedBacktestConfig:
        """Quick test configuration for rapid testing"""
        return UnifiedBacktestConfig(
            mode=BacktestMode.SINGLE_STRATEGY,
            strategies=['ema'],
            epics=['CS.D.EURUSD.CEEM.IP'],
            timeframes=['15m'],
            days=1,
            use_optimal_parameters=True,
            output=OutputConfig(
                show_signals=True,
                max_signals_display=10,
                verbose=True
            ),
            config_name="quick_test",
            description="Quick test configuration for rapid strategy validation"
        )

    @staticmethod
    def comprehensive_analysis() -> UnifiedBacktestConfig:
        """Comprehensive analysis with all features enabled"""
        return UnifiedBacktestConfig(
            mode=BacktestMode.MULTI_STRATEGY,
            strategies=['ema', 'macd', 'kama', 'combined'],
            epics=['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.CEEM.IP', 'CS.D.USDJPY.CEEM.IP'],
            timeframes=['15m', '30m'],
            days=7,
            use_optimal_parameters=True,
            smart_money=SmartMoneyConfig(
                enabled=True,
                confluence_scoring=True,
                min_confluence_score=0.7
            ),
            output=OutputConfig(
                show_signals=True,
                show_performance_details=True,
                show_strategy_comparison=True,
                export_format='json',
                verbose=True
            ),
            config_name="comprehensive_analysis",
            description="Comprehensive multi-strategy analysis with Smart Money"
        )

    @staticmethod
    def parameter_optimization() -> UnifiedBacktestConfig:
        """Parameter optimization configuration"""
        return UnifiedBacktestConfig(
            mode=BacktestMode.PARAMETER_SWEEP,
            strategies=['ema'],
            epics=['CS.D.EURUSD.CEEM.IP'],
            timeframes=['15m'],
            days=14,
            use_optimal_parameters=False,  # Use sweep parameters instead
            optimization=OptimizationConfig(
                enabled=True,
                method=OptimizationMethod.GRID_SEARCH,
                scoring_metric="win_rate",
                max_combinations=50,
                parameter_ranges={
                    'confidence_threshold': '0.4-0.8:0.1',
                    'short_ema': '13-34:3',
                    'long_ema': '34-89:5'
                },
                export_results=True
            ),
            output=OutputConfig(
                show_performance_details=True,
                export_format='json',
                verbose=True
            ),
            config_name="parameter_optimization",
            description="Parameter optimization using grid search"
        )

    @staticmethod
    def signal_validation() -> UnifiedBacktestConfig:
        """Signal validation configuration"""
        return UnifiedBacktestConfig(
            mode=BacktestMode.VALIDATION,
            strategies=['ema'],
            epics=['CS.D.EURUSD.CEEM.IP'],
            timeframes=['15m'],
            days=5,
            validation=ValidationConfig(
                enabled=True,
                show_raw_data=True,
                show_calculations=True,
                show_decision_tree=True,
                data_context_bars=150
            ),
            output=OutputConfig(
                verbose=True,
                log_level="DEBUG"
            ),
            config_name="signal_validation",
            description="Detailed signal validation and inspection"
        )

    @staticmethod
    def performance_comparison() -> UnifiedBacktestConfig:
        """Performance comparison across strategies"""
        return UnifiedBacktestConfig(
            mode=BacktestMode.COMPARISON,
            strategies=['ema', 'macd', 'kama', 'combined', 'bb_supertrend'],
            epics=['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.CEEM.IP'],
            timeframes=['15m'],
            days=14,
            use_optimal_parameters=True,
            performance=PerformanceConfig(
                include_trailing_stop=True,
                target_pips=20.0,
                initial_stop_pips=12.0
            ),
            output=OutputConfig(
                show_strategy_comparison=True,
                show_performance_details=True,
                export_format='json',
                verbose=True
            ),
            config_name="performance_comparison",
            description="Compare performance across multiple strategies"
        )


class BacktestConfigManager:
    """
    Manager for backtest configurations

    Handles creation, validation, loading, saving, and management
    of backtest configurations.
    """

    def __init__(self, config_dir: str = None):
        self.logger = logging.getLogger('config_manager')

        # Default config directory
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'configs', 'backtest'
            )

        self.config_dir = os.path.abspath(config_dir)
        os.makedirs(self.config_dir, exist_ok=True)

        # Templates
        self.templates = ConfigTemplates()

        self.logger.debug(f"📋 Config manager initialized: {self.config_dir}")

    def create_config(
        self,
        template: str = None,
        **overrides
    ) -> UnifiedBacktestConfig:
        """
        Create a new configuration

        Args:
            template: Template name or None for default
            **overrides: Configuration overrides

        Returns:
            New UnifiedBacktestConfig instance
        """

        # Start with template or default
        if template:
            config = self.get_template(template)
        else:
            config = UnifiedBacktestConfig()

        # Apply overrides
        if overrides:
            config = self._apply_overrides(config, overrides)

        return config

    def get_template(self, template_name: str) -> UnifiedBacktestConfig:
        """Get a predefined template configuration"""

        template_methods = {
            'quick_test': self.templates.quick_test,
            'comprehensive': self.templates.comprehensive_analysis,
            'optimization': self.templates.parameter_optimization,
            'validation': self.templates.signal_validation,
            'comparison': self.templates.performance_comparison
        }

        if template_name not in template_methods:
            available = ', '.join(template_methods.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available: {available}")

        return template_methods[template_name]()

    def validate_config(
        self,
        config: UnifiedBacktestConfig,
        level: ConfigValidationLevel = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a configuration

        Args:
            config: Configuration to validate
            level: Validation level (uses config's level if None)

        Returns:
            Tuple of (is_valid, error_messages)
        """

        if level is None:
            level = config.validation_level

        errors = []
        warnings = []

        # Validate core parameters
        if not config.strategies:
            errors.append("At least one strategy must be specified")

        if not config.epics:
            errors.append("At least one epic must be specified")

        if not config.timeframes:
            errors.append("At least one timeframe must be specified")

        if config.days <= 0:
            errors.append("Days must be positive")

        # Validate mode-specific requirements
        if config.mode == BacktestMode.VALIDATION:
            if not config.validation.enabled:
                errors.append("Validation mode requires validation.enabled = True")

            if not config.validation.target_timestamp:
                errors.append("Validation mode requires validation.target_timestamp")

        elif config.mode == BacktestMode.PARAMETER_SWEEP:
            if not config.optimization.enabled:
                errors.append("Parameter sweep mode requires optimization.enabled = True")

            if not config.optimization.parameter_ranges:
                errors.append("Parameter sweep mode requires parameter_ranges")

        # Validate optimization config
        if config.optimization.enabled:
            if config.optimization.max_combinations <= 0:
                errors.append("max_combinations must be positive")

            if not config.optimization.scoring_metric:
                errors.append("scoring_metric must be specified for optimization")

        # Validate performance config
        if config.performance.target_pips <= 0:
            warnings.append("target_pips should be positive")

        if config.performance.initial_stop_pips <= 0:
            warnings.append("initial_stop_pips should be positive")

        # Validate timeframes
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        for tf in config.timeframes:
            if tf not in valid_timeframes:
                warnings.append(f"Timeframe '{tf}' may not be supported")

        # Handle validation level
        if level == ConfigValidationLevel.STRICT:
            errors.extend(warnings)
            warnings = []
        elif level == ConfigValidationLevel.MINIMAL:
            # Only check critical errors
            critical_errors = [e for e in errors if "must" in e.lower()]
            errors = critical_errors

        # Log warnings
        for warning in warnings:
            self.logger.warning(f"⚠️ Config warning: {warning}")

        return len(errors) == 0, errors

    def save_config(
        self,
        config: UnifiedBacktestConfig,
        filename: str = None,
        format: str = "json"
    ) -> str:
        """
        Save configuration to file

        Args:
            config: Configuration to save
            filename: Output filename (auto-generated if None)
            format: File format (json or yaml)

        Returns:
            Path to saved file
        """

        # Generate filename if not provided
        if filename is None:
            if config.config_name:
                filename = f"{config.config_name}.{format}"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_config_{timestamp}.{format}"

        # Ensure correct extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        filepath = os.path.join(self.config_dir, filename)

        # Convert to dictionary
        config_dict = asdict(config)

        # Convert enums to strings
        config_dict = self._serialize_enums(config_dict)

        try:
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            elif format == "yaml":
                with open(filepath, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"✅ Configuration saved: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"❌ Failed to save config: {e}")
            raise

    def load_config(self, filepath: str) -> UnifiedBacktestConfig:
        """
        Load configuration from file

        Args:
            filepath: Path to configuration file

        Returns:
            Loaded UnifiedBacktestConfig
        """

        if not os.path.isabs(filepath):
            filepath = os.path.join(self.config_dir, filepath)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.json'):
                    config_dict = json.load(f)
                elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported file format")

            # Deserialize enums and create config
            config_dict = self._deserialize_enums(config_dict)
            config = self._dict_to_config(config_dict)

            self.logger.info(f"✅ Configuration loaded: {filepath}")
            return config

        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {e}")
            raise

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all saved configurations"""

        configs = []

        for filename in os.listdir(self.config_dir):
            if filename.endswith(('.json', '.yaml', '.yml')):
                filepath = os.path.join(self.config_dir, filename)

                try:
                    # Load basic info without full parsing
                    with open(filepath, 'r') as f:
                        if filename.endswith('.json'):
                            data = json.load(f)
                        else:
                            data = yaml.safe_load(f)

                    configs.append({
                        'filename': filename,
                        'filepath': filepath,
                        'config_name': data.get('config_name', 'Unknown'),
                        'description': data.get('description', 'No description'),
                        'created_at': data.get('created_at', 'Unknown'),
                        'mode': data.get('mode', 'Unknown'),
                        'strategies': data.get('strategies', [])
                    })

                except Exception as e:
                    self.logger.warning(f"⚠️ Could not read config {filename}: {e}")

        return sorted(configs, key=lambda x: x.get('created_at', ''), reverse=True)

    def _apply_overrides(
        self,
        config: UnifiedBacktestConfig,
        overrides: Dict[str, Any]
    ) -> UnifiedBacktestConfig:
        """Apply override values to configuration"""

        # Create a copy to avoid modifying original
        config_dict = asdict(config)

        # Apply overrides with dot notation support
        for key, value in overrides.items():
            self._set_nested_value(config_dict, key, value)

        return self._dict_to_config(config_dict)

    def _set_nested_value(self, dict_obj: Dict, key: str, value: Any):
        """Set nested dictionary value using dot notation"""

        keys = key.split('.')
        current = dict_obj

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _serialize_enums(self, obj: Any) -> Any:
        """Convert enum objects to strings for serialization"""

        if isinstance(obj, dict):
            return {k: self._serialize_enums(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_enums(item) for item in obj]
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        else:
            return obj

    def _deserialize_enums(self, obj: Any) -> Any:
        """Convert enum strings back to enum objects"""

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k == 'mode' and isinstance(v, str):
                    result[k] = BacktestMode(v)
                elif k == 'method' and isinstance(v, str):
                    result[k] = OptimizationMethod(v)
                elif k == 'validation_level' and isinstance(v, str):
                    result[k] = ConfigValidationLevel(v)
                else:
                    result[k] = self._deserialize_enums(v)
            return result
        elif isinstance(obj, list):
            return [self._deserialize_enums(item) for item in obj]
        else:
            return obj

    def _dict_to_config(self, config_dict: Dict) -> UnifiedBacktestConfig:
        """Convert dictionary to UnifiedBacktestConfig"""

        # Create nested dataclass instances
        if 'smart_money' in config_dict:
            config_dict['smart_money'] = SmartMoneyConfig(**config_dict['smart_money'])

        if 'optimization' in config_dict:
            config_dict['optimization'] = OptimizationConfig(**config_dict['optimization'])

        if 'performance' in config_dict:
            config_dict['performance'] = PerformanceConfig(**config_dict['performance'])

        if 'output' in config_dict:
            config_dict['output'] = OutputConfig(**config_dict['output'])

        if 'validation' in config_dict:
            config_dict['validation'] = ValidationConfig(**config_dict['validation'])

        return UnifiedBacktestConfig(**config_dict)

    def get_available_templates(self) -> List[str]:
        """Get list of available template names"""
        return ['quick_test', 'comprehensive', 'optimization', 'validation', 'comparison']

    def clone_config(
        self,
        config: UnifiedBacktestConfig,
        new_name: str = None,
        **overrides
    ) -> UnifiedBacktestConfig:
        """Clone a configuration with optional modifications"""

        # Deep copy the original config
        config_dict = asdict(config)
        cloned_dict = copy.deepcopy(config_dict)

        # Update name and timestamp
        if new_name:
            cloned_dict['config_name'] = new_name

        cloned_dict['created_at'] = datetime.now().isoformat()

        # Apply overrides
        for key, value in overrides.items():
            self._set_nested_value(cloned_dict, key, value)

        return self._dict_to_config(cloned_dict)


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> BacktestConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = BacktestConfigManager()
    return _config_manager