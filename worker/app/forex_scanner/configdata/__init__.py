# configdata/__init__.py
"""
Config namespace that aggregates all configs
Smart Money Concepts (SMC) and Strategy configurations

NOTE: After January 2026 cleanup, only SMC Simple strategy is active.
Legacy strategy configs have been archived to forex_scanner/archive/
trading_config.py DELETED - settings now in database (scanner_global_config)
"""

# Import all config modules using the new modular structure
try:
    from forex_scanner.configdata.smc.smc_configdata import *
    from forex_scanner.configdata.strategies import *
    from forex_scanner.configdata.market_intelligence_config import *
except ImportError:
    # Fallback for different execution contexts (e.g., Docker container)
    try:
        from .smc.smc_configdata import *
        from .strategies import *
        from .market_intelligence_config import *
    except ImportError:
        # Last resort - try relative imports
        from smc.smc_configdata import *
        from strategies import *
        from market_intelligence_config import *


class Config:
    """
    Centralized configuration namespace with dot notation access

    Usage:
        from forex_scanner.configdata import config

        # Access SMC configs
        config.smc.SMART_MONEY_ENABLED
        config.SMART_MONEY_ENABLED  # Also available directly
    """

    def __init__(self):
        # Import all config modules
        # NOTE: trading_config.py DELETED - settings now in database
        try:
            from forex_scanner.configdata.smc import smc_configdata
            from forex_scanner.configdata import strategies
            from forex_scanner.configdata import market_intelligence_config
        except ImportError:
            try:
                from .smc import smc_configdata
                from . import strategies
                from . import market_intelligence_config
            except ImportError:
                import smc.smc_configdata as smc_configdata
                import strategies
                import market_intelligence_config

        # Store module references for dot notation access
        self.smc = smc_configdata
        self.strategies = strategies
        self.intelligence = market_intelligence_config

        # For backward compatibility, add all module attributes to self
        for module in [smc_configdata, strategies, market_intelligence_config]:
            for attr in dir(module):
                if not attr.startswith('_') and attr.isupper():
                    setattr(self, attr, getattr(module, attr))

    def get_config_summary(self) -> dict:
        """Get a summary of all loaded configurations"""
        summary = {
            'loaded_modules': ['smc', 'strategies', 'intelligence'],
            'smc_config': self.smc.get_smart_money_config_summary() if hasattr(self.smc, 'get_smart_money_config_summary') else {},
            'strategies_config': self.strategies.get_strategies_summary() if hasattr(self.strategies, 'get_strategies_summary') else {},
            'intelligence_config': self.intelligence.get_market_intelligence_config_summary() if hasattr(self.intelligence, 'get_market_intelligence_config_summary') else {},
            'total_configs': len([attr for attr in dir(self) if not attr.startswith('_') and attr.isupper()])
        }
        return summary

    def validate_all_configs(self) -> dict:
        """Validate all configuration modules"""
        validation_results = {
            'smc': self._validate_smc_config(),
            'strategies': True,  # Simplified - only SMC Simple active
            'intelligence': self._validate_intelligence_config()
        }
        validation_results['overall_valid'] = all(validation_results.values())
        return validation_results

    def _validate_smc_config(self) -> bool:
        """Validate SMC configuration"""
        try:
            required_smc_configs = [
                'SMART_MONEY_ENABLED',
                'PIP_SIZES',
                'get_pip_size_for_epic'
            ]

            for config_name in required_smc_configs:
                if not hasattr(self.smc, config_name):
                    print(f"❌ Missing SMC config: {config_name}")
                    return False

            print("✅ SMC configuration validation passed")
            return True

        except Exception as e:
            print(f"❌ SMC config validation failed: {e}")
            return False

    def _validate_intelligence_config(self) -> bool:
        """Validate market intelligence configuration"""
        try:
            if hasattr(self.intelligence, 'validate_market_intelligence_config'):
                validation_result = self.intelligence.validate_market_intelligence_config()
                if validation_result.get('valid', False):
                    print("✅ Market intelligence configuration validation passed")
                    return True
                else:
                    print("❌ Market intelligence configuration validation failed")
                    return False
            else:
                return True  # Don't fail if validation method is missing

        except Exception as e:
            print(f"❌ Intelligence config validation failed: {e}")
            return False

    def get_pip_size_for_epic(self, epic: str) -> float:
        """Convenience method to get pip size"""
        if hasattr(self.smc, 'get_pip_size_for_epic'):
            return self.smc.get_pip_size_for_epic(epic)
        return 0.0001

    def get_intelligence_preset(self) -> str:
        """Get current intelligence preset"""
        if hasattr(self.intelligence, 'INTELLIGENCE_PRESET'):
            return self.intelligence.INTELLIGENCE_PRESET
        return 'minimal'

    def is_intelligence_enabled(self) -> bool:
        """Check if market intelligence is enabled"""
        return getattr(self.intelligence, 'ENABLE_MARKET_INTELLIGENCE', False)


# Create singleton instance
config = Config()

# Export the singleton for easy importing
__all__ = ['config']

# Module metadata
__version__ = "2.0.0"
__description__ = "Modular configuration system - SMC Simple strategy only (legacy archived)"

# Validate configuration on import
try:
    validation_result = config.validate_all_configs()
    if not validation_result['overall_valid']:
        print("⚠️ Some configuration validations failed. Check settings.")
except Exception as e:
    print(f"⚠️ Configuration validation error: {e}")

# Startup info - reads from database when available
try:
    from forex_scanner.services.scanner_config_service import get_scanner_config
    _scanner_cfg = get_scanner_config()
    print(f"📊 ConfigData system loaded - SMC enabled: {_scanner_cfg.smart_money_readonly_enabled}")
    print(f"📈 Strategy: SMC Simple only (all legacy strategies removed)")
    print(f"🧠 Intelligence config loaded - Enabled: {_scanner_cfg.enable_market_intelligence_capture}, Preset: {_scanner_cfg.intelligence_preset}")
except Exception as _e:
    # Fallback for when database is not available (e.g., during migrations)
    print(f"📊 ConfigData system loaded - SMC enabled: {getattr(config, 'SMART_MONEY_ENABLED', False)}")
    print(f"📈 Strategy: SMC Simple only (all legacy strategies removed)")
    print(f"🧠 Intelligence config loaded (database unavailable: {_e})")
