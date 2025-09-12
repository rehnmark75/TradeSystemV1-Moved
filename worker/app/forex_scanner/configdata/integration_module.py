"""
Integration Module for Forex Scanner Dynamic Configuration
Provides seamless integration between the existing forex scanner and dynamic configuration system
"""

import os
import json
import logging
import importlib.util
from typing import Dict, Any, Optional, List
from datetime import datetime

from dynamic_config_loader import (
    DatabaseConfigLoader, 
    LegacyConfigWrapper, 
    ConfigSetting, 
    ConfigValidationRule,
    initialize_config,
    get_config_loader
)


class ConfigMigrator:
    """Migrates static config.py to database-driven configuration"""
    
    def __init__(self, config_file_path: str = "config.py"):
        self.config_file_path = config_file_path
        self.logger = logging.getLogger(__name__)
    
    def extract_static_config(self) -> Dict[str, Any]:
        """Extract configuration values from static config.py file"""
        try:
            # Import the static config module
            spec = importlib.util.spec_from_file_location("config", self.config_file_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Extract configuration values
            config_dict = {}
            for attr_name in dir(config_module):
                if not attr_name.startswith('_'):  # Skip private attributes
                    value = getattr(config_module, attr_name)
                    if not callable(value):  # Skip functions/methods
                        config_dict[attr_name] = value
            
            self.logger.info(f"Extracted {len(config_dict)} settings from {self.config_file_path}")
            return config_dict
            
        except Exception as e:
            self.logger.error(f"Failed to extract static config: {e}")
            return {}
    
    def get_forex_scanner_config_schema(self) -> Dict[str, ConfigSetting]:
        """Define the complete forex scanner configuration schema"""
        return {
            # Trading Parameters
            "MIN_CONFIDENCE": ConfigSetting(
                name="MIN_CONFIDENCE",
                value=0.7,
                category="Trading Parameters",
                description="Minimum signal confidence threshold (0.0-1.0)",
                validation=ConfigValidationRule(min_value=0.0, max_value=1.0, data_type="float")
            ),
            "SPREAD_PIPS": ConfigSetting(
                name="SPREAD_PIPS",
                value=1.0,
                category="Trading Parameters", 
                description="Bid/ask spread in pips",
                validation=ConfigValidationRule(min_value=0.1, max_value=10.0, data_type="float")
            ),
            "DEFAULT_TIMEFRAME": ConfigSetting(
                name="DEFAULT_TIMEFRAME",
                value="5m",
                category="Trading Parameters",
                description="Primary analysis timeframe",
                validation=ConfigValidationRule(valid_options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], data_type="string")
            ),
            "SCAN_INTERVAL": ConfigSetting(
                name="SCAN_INTERVAL",
                value=60,
                category="Trading Parameters",
                description="Seconds between scans",
                validation=ConfigValidationRule(min_value=10, max_value=3600, data_type="int")
            ),
            
            # Strategy Settings
            "SIMPLE_EMA_STRATEGY": ConfigSetting(
                name="SIMPLE_EMA_STRATEGY",
                value=True,
                category="Strategy Settings",
                description="Enable simple EMA crossover strategy",
                validation=ConfigValidationRule(data_type="bool")
            ),
            "MACD_EMA_STRATEGY": ConfigSetting(
                name="MACD_EMA_STRATEGY", 
                value=True,
                category="Strategy Settings",
                description="Enable MACD EMA combined strategy",
                validation=ConfigValidationRule(data_type="bool")
            ),
            "COMBINED_STRATEGY_MODE": ConfigSetting(
                name="COMBINED_STRATEGY_MODE",
                value="consensus",
                category="Strategy Settings",
                description="How to combine multiple strategies",
                validation=ConfigValidationRule(valid_options=["consensus", "confirmation", "hierarchy"], data_type="string")
            ),
            "STRATEGY_WEIGHT_EMA": ConfigSetting(
                name="STRATEGY_WEIGHT_EMA",
                value=0.4,
                category="Strategy Settings",
                description="Weight for EMA strategy in combined mode",
                validation=ConfigValidationRule(min_value=0.0, max_value=1.0, data_type="float")
            ),
            "STRATEGY_WEIGHT_MACD": ConfigSetting(
                name="STRATEGY_WEIGHT_MACD",
                value=0.3,
                category="Strategy Settings", 
                description="Weight for MACD strategy in combined mode",
                validation=ConfigValidationRule(min_value=0.0, max_value=1.0, data_type="float")
            ),
            "STRATEGY_WEIGHT_VOLUME": ConfigSetting(
                name="STRATEGY_WEIGHT_VOLUME",
                value=0.2,
                category="Strategy Settings",
                description="Weight for volume analysis in combined mode", 
                validation=ConfigValidationRule(min_value=0.0, max_value=1.0, data_type="float")
            ),
            "STRATEGY_WEIGHT_BEHAVIOR": ConfigSetting(
                name="STRATEGY_WEIGHT_BEHAVIOR",
                value=0.1,
                category="Strategy Settings",
                description="Weight for behavior analysis in combined mode",
                validation=ConfigValidationRule(min_value=0.0, max_value=1.0, data_type="float")
            ),
            
            # Risk Management
            "DEFAULT_STOP_DISTANCE": ConfigSetting(
                name="DEFAULT_STOP_DISTANCE",
                value=20,
                category="Risk Management",
                description="Default stop loss distance in pips",
                validation=ConfigValidationRule(min_value=5, max_value=500, data_type="int")
            ),
            "DEFAULT_RISK_REWARD": ConfigSetting(
                name="DEFAULT_RISK_REWARD",
                value=2.0,
                category="Risk Management",
                description="Default risk/reward ratio",
                validation=ConfigValidationRule(min_value=0.5, max_value=10.0, data_type="float")
            ),
            "MAX_CONCURRENT_POSITIONS": ConfigSetting(
                name="MAX_CONCURRENT_POSITIONS",
                value=3,
                category="Risk Management",
                description="Maximum concurrent open positions",
                validation=ConfigValidationRule(min_value=1, max_value=20, data_type="int")
            ),
            "RISK_PER_TRADE_PERCENT": ConfigSetting(
                name="RISK_PER_TRADE_PERCENT",
                value=1.0,
                category="Risk Management",
                description="Risk percentage per trade",
                validation=ConfigValidationRule(min_value=0.1, max_value=5.0, data_type="float")
            ),
            
            # Market Intelligence
            "ENABLE_CLAUDE_ANALYSIS": ConfigSetting(
                name="ENABLE_CLAUDE_ANALYSIS",
                value=True,
                category="Market Intelligence",
                description="Enable Claude AI market analysis",
                validation=ConfigValidationRule(data_type="bool")
            ),
            "CLAUDE_API_KEY": ConfigSetting(
                name="CLAUDE_API_KEY",
                value="",
                category="Market Intelligence",
                description="Claude API key for AI analysis",
                validation=ConfigValidationRule(data_type="string"),
                is_sensitive=True
            ),
            "MARKET_ANALYSIS_DEPTH": ConfigSetting(
                name="MARKET_ANALYSIS_DEPTH",
                value="standard",
                category="Market Intelligence",
                description="Depth of market analysis",
                validation=ConfigValidationRule(valid_options=["basic", "standard", "deep"], data_type="string")
            ),
            
            # API Configuration  
            "DATABASE_URL": ConfigSetting(
                name="DATABASE_URL",
                value="sqlite:///forex_data.db",
                category="API Configuration",
                description="Database connection URL",
                validation=ConfigValidationRule(data_type="string"),
                is_sensitive=True,
                requires_restart=True
            ),
            "USE_BID_ADJUSTMENT": ConfigSetting(
                name="USE_BID_ADJUSTMENT",
                value=True,
                category="API Configuration",
                description="Adjust prices for bid/ask spread",
                validation=ConfigValidationRule(data_type="bool")
            ),
            "API_RATE_LIMIT": ConfigSetting(
                name="API_RATE_LIMIT",
                value=60,
                category="API Configuration",
                description="API requests per minute limit",
                validation=ConfigValidationRule(min_value=10, max_value=300, data_type="int")
            ),
            
            # Pair Management
            "EPIC_LIST": ConfigSetting(
                name="EPIC_LIST",
                value=[
                    "CS.D.EURUSD.MINI.IP",
                    "CS.D.GBPUSD.MINI.IP", 
                    "CS.D.USDJPY.MINI.IP",
                    "CS.D.USDCHF.MINI.IP",
                    "CS.D.AUDUSD.MINI.IP",
                    "CS.D.USDCAD.MINI.IP",
                    "CS.D.NZDUSD.MINI.IP"
                ],
                category="Pair Management",
                description="List of currency pairs to scan",
                validation=ConfigValidationRule(data_type="json")
            ),
            "MAJOR_PAIRS_ONLY": ConfigSetting(
                name="MAJOR_PAIRS_ONLY",
                value=True,
                category="Pair Management", 
                description="Scan only major currency pairs",
                validation=ConfigValidationRule(data_type="bool")
            ),
            "EXCLUDED_PAIRS": ConfigSetting(
                name="EXCLUDED_PAIRS",
                value=[],
                category="Pair Management",
                description="Currency pairs to exclude from scanning",
                validation=ConfigValidationRule(data_type="json")
            ),
            
            # System Settings
            "USER_TIMEZONE": ConfigSetting(
                name="USER_TIMEZONE",
                value="UTC",
                category="System Settings",
                description="User timezone for display",
                validation=ConfigValidationRule(data_type="string")
            ),
            "LOG_LEVEL": ConfigSetting(
                name="LOG_LEVEL", 
                value="INFO",
                category="System Settings",
                description="Logging level",
                validation=ConfigValidationRule(valid_options=["DEBUG", "INFO", "WARNING", "ERROR"], data_type="string")
            ),
            "ENABLE_PERFORMANCE_TRACKING": ConfigSetting(
                name="ENABLE_PERFORMANCE_TRACKING",
                value=True,
                category="System Settings",
                description="Track trading performance metrics",
                validation=ConfigValidationRule(data_type="bool")
            )
        }
    
    def migrate_static_config(self, database_url: str = None) -> bool:
        """Migrate static configuration to database"""
        try:
            # Extract existing configuration
            static_config = self.extract_static_config()
            
            # Get configuration schema
            schema = self.get_forex_scanner_config_schema()
            
            # Initialize database config loader
            if database_url:
                loader = DatabaseConfigLoader(database_url=database_url)
            else:
                loader = get_config_loader()
            
            # Populate database with schema and static values
            migrated_count = 0
            for setting_name, setting_schema in schema.items():
                # Use static value if available, otherwise use schema default
                value = static_config.get(setting_name, setting_schema.value)
                
                # Update the setting in database
                success = loader.set(
                    setting_name=setting_name,
                    value=value,
                    changed_by="migration",
                    reason="Migrated from static config.py"
                )
                
                if success:
                    migrated_count += 1
                    self.logger.info(f"Migrated {setting_name} = {value}")
                else:
                    self.logger.warning(f"Failed to migrate {setting_name}")
            
            self.logger.info(f"Migration complete: {migrated_count}/{len(schema)} settings migrated")
            return migrated_count > 0
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False


class ForexScannerIntegration:
    """Main integration class for forex scanner dynamic configuration"""
    
    def __init__(self, config_loader: DatabaseConfigLoader):
        self.config_loader = config_loader
        self.logger = logging.getLogger(__name__)
        
        # Create legacy wrapper for backward compatibility
        self.config_wrapper = LegacyConfigWrapper(config_loader)
    
    def get_config(self) -> LegacyConfigWrapper:
        """Get configuration wrapper that behaves like static config module"""
        return self.config_wrapper
    
    def update_config(self, setting_name: str, value: Any, changed_by: str = "system") -> bool:
        """Update a configuration setting"""
        return self.config_loader.set(setting_name, value, changed_by)
    
    def get_scanner_config(self) -> Dict[str, Any]:
        """Get configuration specifically formatted for scanner initialization"""
        return {
            'epic_list': self.config_loader.get('EPIC_LIST', []),
            'scan_interval': self.config_loader.get('SCAN_INTERVAL', 60),
            'claude_api_key': self.config_loader.get('CLAUDE_API_KEY', ''),
            'enable_claude_analysis': self.config_loader.get('ENABLE_CLAUDE_ANALYSIS', True),
            'use_bid_adjustment': self.config_loader.get('USE_BID_ADJUSTMENT', True),
            'spread_pips': self.config_loader.get('SPREAD_PIPS', 1.0),
            'min_confidence': self.config_loader.get('MIN_CONFIDENCE', 0.7),
            'user_timezone': self.config_loader.get('USER_TIMEZONE', 'UTC'),
            'database_url': self.config_loader.get('DATABASE_URL', 'sqlite:///forex_data.db')
        }
    
    def apply_configuration_preset(self, preset_name: str, applied_by: str = "system") -> bool:
        """Apply a predefined configuration preset"""
        return self.config_loader.apply_preset(preset_name, applied_by)
    
    def create_configuration_backup(self, backup_name: str, description: str = "") -> bool:
        """Create a backup of current configuration"""
        return self.config_loader.backup_current_config(
            backup_name=backup_name,
            description=description,
            created_by="integration"
        )
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration state"""
        summary = self.config_loader.get_config_summary()
        
        # Add forex-scanner specific metrics
        scanner_config = self.get_scanner_config()
        summary['forex_scanner'] = {
            'monitored_pairs': len(scanner_config['epic_list']),
            'claude_enabled': scanner_config['enable_claude_analysis'],
            'scan_frequency': f"{scanner_config['scan_interval']}s",
            'confidence_threshold': scanner_config['min_confidence']
        }
        
        return summary


def setup_forex_scanner_config(
    database_url: str = None,
    auto_migrate: bool = True,
    enable_hot_reload: bool = True,
    refresh_interval: int = 60
) -> ForexScannerIntegration:
    """
    Setup the forex scanner dynamic configuration system
    
    Args:
        database_url: Database connection URL
        auto_migrate: Automatically migrate static config.py if found
        enable_hot_reload: Enable automatic configuration refresh
        refresh_interval: How often to check for config changes (seconds)
        
    Returns:
        ForexScannerIntegration instance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize dynamic configuration loader
        config_loader = initialize_config(
            database_url=database_url,
            refresh_interval=refresh_interval,
            enable_hot_reload=enable_hot_reload,
            fallback_config_path="config.py"
        )
        
        # Perform automatic migration if requested and config.py exists
        if auto_migrate and os.path.exists("config.py"):
            logger.info("Performing automatic migration from config.py")
            migrator = ConfigMigrator("config.py")
            migration_success = migrator.migrate_static_config(database_url)
            
            if migration_success:
                logger.info("Configuration migration completed successfully")
                # Create backup of static config
                backup_name = f"static_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename("config.py", f"{backup_name}.py")
                logger.info(f"Static config.py backed up as {backup_name}.py")
            else:
                logger.warning("Configuration migration failed, using fallback")
        
        # Create integration instance
        integration = ForexScannerIntegration(config_loader)
        
        # Setup default presets if they don't exist
        _setup_default_presets(config_loader)
        
        logger.info("Forex scanner configuration system initialized successfully")
        return integration
        
    except Exception as e:
        logger.error(f"Failed to setup forex scanner configuration: {e}")
        raise


def _setup_default_presets(config_loader: DatabaseConfigLoader):
    """Setup default configuration presets"""
    logger = logging.getLogger(__name__)
    
    # Conservative preset
    conservative_settings = {
        'MIN_CONFIDENCE': 0.8,
        'DEFAULT_STOP_DISTANCE': 30,
        'DEFAULT_RISK_REWARD': 3.0,
        'MAX_CONCURRENT_POSITIONS': 2,
        'RISK_PER_TRADE_PERCENT': 0.5,
        'SCAN_INTERVAL': 120
    }
    
    # Aggressive preset
    aggressive_settings = {
        'MIN_CONFIDENCE': 0.6,
        'DEFAULT_STOP_DISTANCE': 15,
        'DEFAULT_RISK_REWARD': 1.5,
        'MAX_CONCURRENT_POSITIONS': 5,
        'RISK_PER_TRADE_PERCENT': 2.0,
        'SCAN_INTERVAL': 30
    }
    
    # Scalping preset
    scalping_settings = {
        'MIN_CONFIDENCE': 0.65,
        'DEFAULT_STOP_DISTANCE': 8,
        'DEFAULT_RISK_REWARD': 1.0,
        'MAX_CONCURRENT_POSITIONS': 3,
        'RISK_PER_TRADE_PERCENT': 1.5,
        'SCAN_INTERVAL': 15,
        'DEFAULT_TIMEFRAME': '1m'
    }
    
    presets = {
        'Conservative': conservative_settings,
        'Aggressive': aggressive_settings,
        'Scalping': scalping_settings
    }
    
    try:
        # This would require implementing preset storage in the database
        # For now, just log that presets would be created
        logger.info("Default configuration presets are ready to be created")
        # TODO: Implement preset creation in database
        
    except Exception as e:
        logger.warning(f"Failed to setup default presets: {e}")


# Convenience functions for backward compatibility
def get_forex_config():
    """Get forex scanner configuration (backward compatibility)"""
    integration = setup_forex_scanner_config()
    return integration.get_config()


def update_forex_config(setting_name: str, value: Any, changed_by: str = "system"):
    """Update forex scanner configuration (backward compatibility)"""
    integration = setup_forex_scanner_config()
    return integration.update_config(setting_name, value, changed_by)


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the integration module
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the forex scanner configuration system
    integration = setup_forex_scanner_config(
        database_url="postgresql://user:pass@localhost:5432/forex_config",
        auto_migrate=True,
        enable_hot_reload=True
    )
    
    # Get configuration (replaces 'import config')
    config = integration.get_config()
    
    # Use configuration exactly like before
    print(f"Min confidence: {config.MIN_CONFIDENCE}")
    print(f"Epic list: {config.EPIC_LIST}")
    print(f"Claude enabled: {config.ENABLE_CLAUDE_ANALYSIS}")
    
    # Update configuration dynamically
    success = integration.update_config(
        "MIN_CONFIDENCE", 
        0.75, 
        changed_by="admin"
    )
    print(f"Configuration update success: {success}")
    
    # Get scanner-specific configuration
    scanner_config = integration.get_scanner_config()
    print(f"Scanner config: {scanner_config}")
    
    # Create configuration backup
    backup_success = integration.create_configuration_backup(
        "before_optimization",
        "Backup before optimization changes"
    )
    print(f"Backup created: {backup_success}")
    
    # Get configuration summary
    summary = integration.get_configuration_summary()
    print(f"Configuration summary: {summary}")