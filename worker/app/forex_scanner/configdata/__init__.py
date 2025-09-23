# configdata/__init__.py
"""
Config namespace that aggregates all configs
Smart Money Concepts (SMC) and Strategy configurations
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

# Future config imports (examples for the pattern)
# from configdata.support_and_resistance.sr_config import *
# from configdata.trading.trading_config import *
# from configdata.api.api_config import *

class Config:
    """
    Centralized configuration namespace with dot notation access
    
    Usage:
        from forex_scanner.configdata import config
        
        # Access SMC configs
        config.smc.SMART_MONEY_ENABLED
        config.SMART_MONEY_ENABLED  # Also available directly
        
        # Future usage patterns
        # config.trading.RISK_PERCENTAGE
        # config.api.CLAUDE_API_KEY
    """
    
    def __init__(self):
        # Import all config modules
        try:
            from forex_scanner.configdata.smc import smc_configdata
            from forex_scanner.configdata import strategies
            from forex_scanner.configdata import market_intelligence_config
        except ImportError:
            # Fallback for different execution contexts
            try:
                from .smc import smc_configdata
                from . import strategies
                from . import market_intelligence_config
            except ImportError:
                # Last resort
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
        
        # Future module imports will follow this pattern:
        # from configdata.support_and_resistance import sr_config
        # from configdata.trading import trading_config
        # from configdata.api import api_config
        # 
        # self.sr = sr_config
        # self.trading = trading_config
        # self.api = api_config
        # 
        # # Add all attributes for backward compatibility
        # for module in [sr_config, trading_config, api_config]:
        #     for attr in dir(module):
        #         if not attr.startswith('_') and attr.isupper():
        #             setattr(self, attr, getattr(module, attr))
    
    def get_config_summary(self) -> dict:
        """
        Get a summary of all loaded configurations
        """
        summary = {
            'loaded_modules': ['smc', 'strategies', 'intelligence'],
            'smc_config': self.smc.get_smart_money_config_summary() if hasattr(self.smc, 'get_smart_money_config_summary') else {},
            'strategies_config': self.strategies.get_strategies_summary() if hasattr(self.strategies, 'get_strategies_summary') else {},
            'intelligence_config': self.intelligence.get_market_intelligence_config_summary() if hasattr(self.intelligence, 'get_market_intelligence_config_summary') else {},
            'total_configs': len([attr for attr in dir(self) if not attr.startswith('_') and attr.isupper()])
        }
        
        # Future modules will add their summaries here:
        # summary['trading_config'] = self.trading.get_summary() if hasattr(self.trading, 'get_summary') else {}
        # summary['api_config'] = self.api.get_summary() if hasattr(self.api, 'get_summary') else {}
        
        return summary
    
    def validate_all_configs(self) -> dict:
        """
        Validate all configuration modules
        """
        validation_results = {
            'smc': self._validate_smc_config(),
            'strategies': self._validate_strategies_config(),
            'intelligence': self._validate_intelligence_config()
        }

        # Future validations:
        # validation_results['trading'] = self._validate_trading_config()
        # validation_results['api'] = self._validate_api_config()

        validation_results['overall_valid'] = all(validation_results.values())
        return validation_results
    
    def _validate_smc_config(self) -> bool:
        """
        Validate SMC configuration completeness and correctness
        """
        try:
            required_smc_configs = [
                'SMART_MONEY_ENABLED',
                'STRUCTURE_SWING_LOOKBACK',
                'ORDER_FLOW_MIN_OB_SIZE_PIPS',
                'SMART_MONEY_STRUCTURE_WEIGHT',
                'SMART_MONEY_ORDER_FLOW_WEIGHT',
                'PIP_SIZES',
                'get_pip_size_for_epic'
            ]
            
            for config_name in required_smc_configs:
                if not hasattr(self.smc, config_name):
                    print(f"❌ Missing SMC config: {config_name}")
                    return False
            
            # Validate pip sizes dictionary
            pip_sizes = getattr(self.smc, 'PIP_SIZES', {})
            if not isinstance(pip_sizes, dict) or len(pip_sizes) == 0:
                print("❌ PIP_SIZES must be a non-empty dictionary")
                return False
            
            # Validate weight values are reasonable
            structure_weight = getattr(self.smc, 'SMART_MONEY_STRUCTURE_WEIGHT', 0)
            order_flow_weight = getattr(self.smc, 'SMART_MONEY_ORDER_FLOW_WEIGHT', 0)
            
            if not (0 <= structure_weight <= 1) or not (0 <= order_flow_weight <= 1):
                print("❌ SMC weights must be between 0 and 1")
                return False
            
            print("✅ SMC configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ SMC config validation failed: {e}")
            return False
    
    def _validate_strategies_config(self) -> bool:
        """
        Validate strategy configuration completeness and correctness
        """
        try:
            # Delegate to the strategies module validation
            if hasattr(self.strategies, 'validate_strategy_configs'):
                validation_result = self.strategies.validate_strategy_configs()
                return validation_result.get('overall_valid', False)
            else:
                print("⚠️ Strategy validation method not available")
                return True  # Don't fail if validation method is missing

        except Exception as e:
            print(f"❌ Strategy config validation failed: {e}")
            return False

    def _validate_intelligence_config(self) -> bool:
        """
        Validate market intelligence configuration completeness and correctness
        """
        try:
            # Delegate to the intelligence module validation
            if hasattr(self.intelligence, 'validate_market_intelligence_config'):
                validation_result = self.intelligence.validate_market_intelligence_config()
                if validation_result.get('valid', False):
                    print("✅ Market intelligence configuration validation passed")
                    return True
                else:
                    print("❌ Market intelligence configuration validation failed:")
                    for error in validation_result.get('errors', []):
                        print(f"   {error}")
                    return False
            else:
                print("⚠️ Intelligence validation method not available")
                return True  # Don't fail if validation method is missing

        except Exception as e:
            print(f"❌ Intelligence config validation failed: {e}")
            return False
    
    def reload_configs(self):
        """
        Reload all configuration modules (useful for development)
        """
        import importlib

        # Reload all config modules
        from forex_scanner.configdata.smc import smc_configdata
        from forex_scanner.configdata import strategies
        from forex_scanner.configdata import market_intelligence_config

        importlib.reload(smc_configdata)
        importlib.reload(strategies)
        importlib.reload(market_intelligence_config)

        self.smc = smc_configdata
        self.strategies = strategies
        self.intelligence = market_intelligence_config

        # Update direct attributes from all modules
        for module in [smc_configdata, strategies, market_intelligence_config]:
            for attr in dir(module):
                if not attr.startswith('_') and attr.isupper():
                    setattr(self, attr, getattr(module, attr))

        print("✅ Configuration modules reloaded")
        
        # Future reloads will follow this pattern:
        # importlib.reload(sr_config)
        # importlib.reload(trading_config)
        # importlib.reload(api_config)
    
    def get_pip_size_for_epic(self, epic: str) -> float:
        """
        Convenience method to get pip size (delegates to SMC config)
        """
        if hasattr(self.smc, 'get_pip_size_for_epic'):
            return self.smc.get_pip_size_for_epic(epic)
        else:
            # Fallback logic
            return 0.0001
    
    def get_macd_threshold_for_epic(self, epic: str) -> float:
        """
        Convenience method to get MACD threshold (delegates to strategy config)
        """
        if hasattr(self.strategies, 'get_macd_threshold_for_epic'):
            return self.strategies.get_macd_threshold_for_epic(epic)
        else:
            # Fallback logic
            return 0.00003  # Default non-JPY threshold
    
    def get_ema_config_for_epic(self, epic: str, market_condition: str = 'default') -> dict:
        """
        Convenience method to get EMA configuration (delegates to strategy config)
        """
        if hasattr(self.strategies, 'get_ema_config_for_epic'):
            return self.strategies.get_ema_config_for_epic(epic, market_condition)
        else:
            # Fallback logic
            return {'short': 21, 'long': 50, 'trend': 200}
    
    def get_ema_distance_for_epic(self, epic: str) -> float:
        """
        Convenience method to get EMA200 distance (delegates to strategy config)
        """
        if hasattr(self.strategies, 'get_ema_distance_for_epic'):
            return self.strategies.get_ema_distance_for_epic(epic)
        else:
            # Fallback logic
            return 5.0  # Default distance
    
    def get_ema_separation_for_epic(self, epic: str) -> float:
        """
        Convenience method to get EMA separation distance (delegates to strategy config)
        """
        if hasattr(self.strategies, 'get_ema_separation_for_epic'):
            return self.strategies.get_ema_separation_for_epic(epic)
        else:
            # Fallback logic
            return 3.0  # Default separation
    
    # ==================== ZERO-LAG STRATEGY CONVENIENCE METHODS ====================
    
    def get_zerolag_config_for_epic(self, epic: str, market_condition: str = 'default') -> dict:
        """
        Convenience method to get zero-lag configuration (delegates to strategy config)
        """
        if hasattr(self.strategies, 'get_zerolag_config_for_epic'):
            return self.strategies.get_zerolag_config_for_epic(epic, market_condition)
        else:
            # Fallback logic
            return {
                'zl_length': 50,
                'band_multiplier': 1.5,
                'min_confidence': 0.65,
                'bb_length': 20,
                'bb_mult': 2.0,
                'kc_length': 20,
                'kc_mult': 1.5
            }
    
    def get_zerolag_threshold_for_epic(self, epic: str) -> float:
        """
        Convenience method to get zero-lag threshold (delegates to strategy config)
        """
        if hasattr(self.strategies, 'get_zerolag_threshold_for_epic'):
            return self.strategies.get_zerolag_threshold_for_epic(epic)
        else:
            # Fallback logic
            if 'JPY' in epic.upper():
                return 0.003
            elif 'GBP' in epic.upper():
                return 0.00005
            else:
                return 0.00003
    
    def get_zerolag_squeeze_config_for_epic(self, epic: str) -> dict:
        """
        Convenience method to get zero-lag squeeze configuration (delegates to strategy config)
        """
        if hasattr(self.strategies, 'get_zerolag_squeeze_config_for_epic'):
            return self.strategies.get_zerolag_squeeze_config_for_epic(epic)
        else:
            # Fallback logic with epic-specific adjustments
            config = {
                'bb_length': 20,
                'bb_mult': 2.0,
                'kc_length': 20,
                'kc_mult': 1.5
            }
            
            epic_upper = epic.upper()
            if 'JPY' in epic_upper:
                config['bb_mult'] *= 1.3
                config['kc_mult'] *= 1.2
            
            return config

    # ==================== MARKET INTELLIGENCE CONVENIENCE METHODS ====================

    def get_intelligence_preset(self) -> str:
        """
        Convenience method to get current intelligence preset
        """
        if hasattr(self.intelligence, 'INTELLIGENCE_PRESET'):
            return self.intelligence.INTELLIGENCE_PRESET
        else:
            return 'minimal'  # Default fallback

    def set_intelligence_preset(self, preset_name: str):
        """
        Convenience method to set intelligence preset (delegates to intelligence config)
        """
        if hasattr(self.intelligence, 'set_intelligence_preset'):
            return self.intelligence.set_intelligence_preset(preset_name)
        else:
            print(f"⚠️ Intelligence preset switching not available")

    def get_intelligence_summary(self) -> dict:
        """
        Convenience method to get intelligence configuration summary
        """
        if hasattr(self.intelligence, 'get_market_intelligence_config_summary'):
            return self.intelligence.get_market_intelligence_config_summary()
        else:
            return {'error': 'Intelligence summary not available'}

    def is_intelligence_enabled(self) -> bool:
        """
        Convenience method to check if market intelligence is enabled
        """
        return getattr(self.intelligence, 'ENABLE_MARKET_INTELLIGENCE', False)

    def get_intelligence_threshold(self) -> float:
        """
        Convenience method to get current intelligence threshold
        """
        mode = getattr(self.intelligence, 'INTELLIGENCE_MODE', 'live_only')
        thresholds = getattr(self.intelligence, 'INTELLIGENCE_THRESHOLDS', {})
        return thresholds.get(mode, 0.5)

# Create singleton instance
config = Config()

# Export the singleton for easy importing
__all__ = ['config']

# Module metadata
__version__ = "1.0.0"
__description__ = "Modular configuration system with SMC, strategies, and market intelligence"

# Validate configuration on import (optional - can be disabled in production)
try:
    validation_result = config.validate_all_configs()
    if not validation_result['overall_valid']:
        print("⚠️ Some configuration validations failed. Check settings.")
except Exception as e:
    print(f"⚠️ Configuration validation error: {e}")

print(f"📊 ConfigData system loaded - SMC enabled: {getattr(config, 'SMART_MONEY_ENABLED', False)}")
print(f"📈 Strategy configs loaded - ZeroLag: {getattr(config, 'ZERO_LAG_STRATEGY', False)}, MACD: {getattr(config, 'MACD_EMA_STRATEGY', False)}, EMA: {getattr(config, 'SIMPLE_EMA_STRATEGY', False)}")
print(f"🧠 Intelligence config loaded - Enabled: {getattr(config, 'ENABLE_MARKET_INTELLIGENCE', False)}, Preset: {getattr(config, 'INTELLIGENCE_PRESET', 'unknown')}")