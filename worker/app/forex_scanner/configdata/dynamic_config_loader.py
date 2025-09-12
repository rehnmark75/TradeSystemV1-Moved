"""
Dynamic Configuration Loader for Forex Scanner
Replaces static config.py with database-driven configuration management
"""

import os
import json
import time
import threading
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras
from contextlib import contextmanager


@dataclass
class ConfigValidationRule:
    """Validation rule for configuration settings"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_options: Optional[List] = None
    required: bool = False
    data_type: str = "string"


@dataclass 
class ConfigSetting:
    """Configuration setting with metadata"""
    name: str
    value: Any
    category: str
    description: str = ""
    validation: ConfigValidationRule = field(default_factory=ConfigValidationRule)
    is_sensitive: bool = False
    requires_restart: bool = False
    last_updated: Optional[datetime] = None


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class DatabaseConfigLoader:
    """
    Database-driven configuration loader with hot reloading capabilities
    Replaces the static config.py with dynamic configuration management
    """
    
    def __init__(self, 
                 database_url: str = None,
                 refresh_interval: int = 60,
                 enable_hot_reload: bool = True,
                 fallback_config_path: str = "config.py"):
        """
        Initialize the dynamic configuration loader
        
        Args:
            database_url: PostgreSQL connection string
            refresh_interval: How often to check for config changes (seconds)
            enable_hot_reload: Whether to automatically reload configuration
            fallback_config_path: Static config file to fall back to if DB unavailable
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.refresh_interval = refresh_interval
        self.enable_hot_reload = enable_hot_reload
        self.fallback_config_path = fallback_config_path
        
        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
        self._settings_metadata: Dict[str, ConfigSetting] = {}
        self._last_refresh = None
        self._refresh_thread = None
        self._stop_refresh = threading.Event()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Load initial configuration
        self.force_refresh()
        
        # Start hot reload thread if enabled
        if self.enable_hot_reload:
            self.start_hot_reload()
    
    def __del__(self):
        """Cleanup resources on destruction"""
        self.stop_hot_reload()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def start_hot_reload(self):
        """Start the hot reload background thread"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
            
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
        self.logger.info("Hot reload thread started")
    
    def stop_hot_reload(self):
        """Stop the hot reload background thread"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._stop_refresh.set()
            self._refresh_thread.join(timeout=5)
            self.logger.info("Hot reload thread stopped")
    
    def _refresh_loop(self):
        """Background thread for periodic configuration refresh"""
        while not self._stop_refresh.is_set():
            try:
                if self._has_configuration_changed():
                    self.force_refresh()
                    self.logger.info("Configuration automatically refreshed")
            except Exception as e:
                self.logger.error(f"Error in refresh loop: {e}")
            
            # Wait for the refresh interval or stop signal
            self._stop_refresh.wait(self.refresh_interval)
    
    def force_refresh(self) -> bool:
        """
        Force refresh configuration from database or fallback
        
        Returns:
            True if refresh successful, False otherwise
        """
        # Try database first
        if self.database_url and self._load_from_database():
            return True
        
        # Fall back to static config file
        self.logger.warning("Database unavailable, using fallback configuration")
        return self._load_from_fallback()
    
    def _load_from_database(self) -> bool:
        """Load configuration from database"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT name, value_text, value_number, value_boolean, data_type,
                               category, description, is_sensitive, requires_restart,
                               min_value, max_value, valid_options, updated_at
                        FROM config_settings 
                        WHERE is_active = true
                    """)
                    
                    settings = cur.fetchall()
                    
                new_cache = {}
                new_metadata = {}
                
                for setting in settings:
                    name = setting['name']
                    data_type = setting['data_type']
                    
                    try:
                        # Parse value based on data type
                        if data_type == 'boolean':
                            parsed_value = setting['value_boolean']
                        elif data_type in ['int', 'float']:
                            parsed_value = setting['value_number']
                        elif data_type == 'json':
                            parsed_value = json.loads(setting['value_text']) if setting['value_text'] else None
                        else:
                            parsed_value = setting['value_text']
                        
                        new_cache[name] = parsed_value
                        
                        # Parse valid options if available
                        valid_options = None
                        if setting['valid_options']:
                            try:
                                valid_options = json.loads(setting['valid_options'])
                            except json.JSONDecodeError:
                                valid_options = setting['valid_options'].split(',')
                        
                        # Create validation rule
                        validation_rule = ConfigValidationRule(
                            min_value=setting['min_value'],
                            max_value=setting['max_value'],
                            valid_options=valid_options,
                            data_type=data_type
                        )
                        
                        # Store metadata
                        new_metadata[name] = ConfigSetting(
                            name=name,
                            value=parsed_value,
                            category=setting['category'],
                            description=setting['description'],
                            validation=validation_rule,
                            is_sensitive=setting['is_sensitive'],
                            requires_restart=setting['requires_restart'],
                            last_updated=setting['updated_at']
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to parse setting {name}: {e}")
                        continue
                
                # Update cache atomically
                self._config_cache = new_cache
                self._settings_metadata = new_metadata
                self._last_refresh = datetime.now()
                
                self.logger.info(f"Loaded {len(new_cache)} configuration settings from database")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration from database: {e}")
            return False
    
    def _load_from_fallback(self) -> bool:
        """Load configuration from fallback static config file"""
        try:
            # Import the static config module
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", self.fallback_config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Extract configuration values
            fallback_cache = {}
            for attr_name in dir(config_module):
                if not attr_name.startswith('_'):  # Skip private attributes
                    value = getattr(config_module, attr_name)
                    if not callable(value):  # Skip functions/methods
                        fallback_cache[attr_name] = value
            
            self._config_cache = fallback_cache
            self._last_refresh = datetime.now()
            
            self.logger.warning(f"Loaded {len(fallback_cache)} settings from fallback config")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load fallback configuration: {e}")
            return False
    
    def _has_configuration_changed(self) -> bool:
        """Check if configuration has changed since last refresh"""
        if not self.database_url:
            return False
            
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT MAX(updated_at) FROM config_settings")
                    result = cur.fetchone()
                    
                    if result and result[0] and self._last_refresh:
                        latest_update = result[0]
                        return latest_update > self._last_refresh
                        
        except Exception as e:
            self.logger.error(f"Error checking for configuration changes: {e}")
            
        return False
    
    def get(self, setting_name: str, default: Any = None) -> Any:
        """
        Get configuration value by name
        
        Args:
            setting_name: Name of the configuration setting
            default: Default value if setting not found
            
        Returns:
            Configuration value or default
        """
        return self._config_cache.get(setting_name, default)
    
    def get_category(self, category: str) -> Dict[str, Any]:
        """
        Get all configuration settings for a specific category
        
        Args:
            category: Configuration category name
            
        Returns:
            Dictionary of settings in the category
        """
        result = {}
        for name, metadata in self._settings_metadata.items():
            if metadata.category == category:
                result[name] = self._config_cache.get(name)
        return result
    
    def set(self, setting_name: str, value: Any, changed_by: str = "system", reason: str = "") -> bool:
        """
        Update a configuration setting
        
        Args:
            setting_name: Name of the configuration setting
            value: New value for the setting
            changed_by: Who made the change
            reason: Reason for the change
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate the new value
            self._validate_setting_value(setting_name, value)
            
            # Update in database if available
            if self.database_url:
                with self.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        # Determine value column based on type
                        if isinstance(value, bool):
                            cur.execute("""
                                UPDATE config_settings 
                                SET value_boolean = %s, updated_at = NOW(), updated_by = %s, change_reason = %s
                                WHERE name = %s
                            """, (value, changed_by, reason, setting_name))
                        elif isinstance(value, (int, float)):
                            cur.execute("""
                                UPDATE config_settings 
                                SET value_number = %s, updated_at = NOW(), updated_by = %s, change_reason = %s
                                WHERE name = %s
                            """, (value, changed_by, reason, setting_name))
                        else:
                            # Handle strings and JSON
                            if isinstance(value, (dict, list)):
                                value_text = json.dumps(value)
                            else:
                                value_text = str(value)
                            
                            cur.execute("""
                                UPDATE config_settings 
                                SET value_text = %s, updated_at = NOW(), updated_by = %s, change_reason = %s
                                WHERE name = %s
                            """, (value_text, changed_by, reason, setting_name))
                    
                    conn.commit()
            
            # Update local cache
            self._config_cache[setting_name] = value
            
            # Update metadata if exists
            if setting_name in self._settings_metadata:
                self._settings_metadata[setting_name].value = value
                self._settings_metadata[setting_name].last_updated = datetime.now()
            
            self.logger.info(f"Updated setting {setting_name} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update setting {setting_name}: {e}")
            return False
    
    def _validate_setting_value(self, setting_name: str, value: Any):
        """Validate a configuration setting value"""
        if setting_name not in self._settings_metadata:
            return  # Skip validation for unknown settings
        
        metadata = self._settings_metadata[setting_name]
        validation = metadata.validation
        
        # Type validation
        if validation.data_type == 'int' and not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ConfigValidationError(f"{setting_name} must be an integer")
        
        elif validation.data_type == 'float' and not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ConfigValidationError(f"{setting_name} must be a number")
        
        elif validation.data_type == 'bool' and not isinstance(value, bool):
            if isinstance(value, str):
                value = value.lower() in ('true', '1', 'yes', 'on')
            else:
                raise ConfigValidationError(f"{setting_name} must be a boolean")
        
        # Range validation
        if validation.min_value is not None and isinstance(value, (int, float)):
            if value < validation.min_value:
                raise ConfigValidationError(f"{setting_name} must be >= {validation.min_value}")
        
        if validation.max_value is not None and isinstance(value, (int, float)):
            if value > validation.max_value:
                raise ConfigValidationError(f"{setting_name} must be <= {validation.max_value}")
        
        # Options validation
        if validation.valid_options and value not in validation.valid_options:
            raise ConfigValidationError(f"{setting_name} must be one of: {validation.valid_options}")
    
    def apply_preset(self, preset_name: str, applied_by: str = "system") -> bool:
        """
        Apply a configuration preset
        
        Args:
            preset_name: Name of the preset to apply
            applied_by: Who applied the preset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.database_url:
                self.logger.error("Database connection required for presets")
                return False
            
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get preset configuration
                    cur.execute("""
                        SELECT setting_name, setting_value, data_type
                        FROM config_presets 
                        WHERE preset_name = %s AND is_active = true
                    """, (preset_name,))
                    
                    preset_settings = cur.fetchall()
                    
                    if not preset_settings:
                        self.logger.error(f"Preset '{preset_name}' not found")
                        return False
                    
                    # Apply each setting in the preset
                    for setting in preset_settings:
                        setting_name = setting['setting_name']
                        setting_value = setting['setting_value']
                        data_type = setting['data_type']
                        
                        # Parse value based on type
                        if data_type == 'boolean':
                            parsed_value = setting_value.lower() in ('true', '1', 'yes')
                        elif data_type == 'int':
                            parsed_value = int(setting_value)
                        elif data_type == 'float':
                            parsed_value = float(setting_value)
                        elif data_type == 'json':
                            parsed_value = json.loads(setting_value)
                        else:
                            parsed_value = setting_value
                        
                        # Update the setting
                        self.set(setting_name, parsed_value, applied_by, f"Applied preset: {preset_name}")
                    
                    self.logger.info(f"Applied preset '{preset_name}' with {len(preset_settings)} settings")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to apply preset {preset_name}: {e}")
            return False
    
    def backup_current_config(self, backup_name: str, description: str = "", created_by: str = "system") -> bool:
        """
        Create a backup of current configuration
        
        Args:
            backup_name: Name for the backup
            description: Description of the backup
            created_by: Who created the backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.database_url:
                self.logger.error("Database connection required for backups")
                return False
            
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Create backup record
                    cur.execute("""
                        INSERT INTO config_backups (backup_name, description, created_by, created_at, config_data)
                        VALUES (%s, %s, %s, NOW(), %s)
                    """, (backup_name, description, created_by, json.dumps(self._config_cache)))
                    
                    conn.commit()
                    self.logger.info(f"Created configuration backup: {backup_name}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to create backup {backup_name}: {e}")
            return False
    
    def get_restart_required_settings(self) -> List[str]:
        """Get list of settings that require restart when changed"""
        return [
            name for name, metadata in self._settings_metadata.items()
            if metadata.requires_restart
        ]
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration state"""
        categories = {}
        for metadata in self._settings_metadata.values():
            if metadata.category not in categories:
                categories[metadata.category] = 0
            categories[metadata.category] += 1
        
        return {
            'total_settings': len(self._config_cache),
            'categories': categories,
            'restart_required_settings': len(self.get_restart_required_settings()),
            'last_refresh': self._last_refresh,
            'hot_reload_enabled': self.enable_hot_reload,
            'database_connected': self.database_url is not None
        }
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow direct attribute access to configuration values
        This maintains compatibility with existing config.py usage patterns
        """
        if name in self._config_cache:
            return self._config_cache[name]
        raise AttributeError(f"Configuration setting '{name}' not found")
    
    def __contains__(self, name: str) -> bool:
        """Check if configuration setting exists"""
        return name in self._config_cache
    
    def keys(self):
        """Get all configuration setting names"""
        return self._config_cache.keys()
    
    def values(self):
        """Get all configuration values"""
        return self._config_cache.values()
    
    def items(self):
        """Get all configuration items as (name, value) pairs"""
        return self._config_cache.items()


class LegacyConfigWrapper:
    """
    Wrapper to maintain compatibility with existing config.py usage
    Can be imported as 'config' to replace the static config module
    """
    
    def __init__(self, config_loader: DatabaseConfigLoader):
        self._loader = config_loader
    
    def __getattr__(self, name: str) -> Any:
        return self._loader.get(name)
    
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            # Allow setting configuration values
            self._loader.set(name, value, changed_by="code", reason="Direct assignment")
    
    def reload(self):
        """Reload configuration from database"""
        return self._loader.force_refresh()
    
    def get_summary(self):
        """Get configuration summary"""
        return self._loader.get_config_summary()


# Global configuration instance
_global_config_loader = None
_global_config_wrapper = None


def initialize_config(database_url: str = None, 
                     refresh_interval: int = 60,
                     enable_hot_reload: bool = True,
                     fallback_config_path: str = "config.py") -> DatabaseConfigLoader:
    """
    Initialize the global configuration loader
    
    Args:
        database_url: PostgreSQL connection string
        refresh_interval: How often to check for config changes (seconds)
        enable_hot_reload: Whether to automatically reload configuration
        fallback_config_path: Static config file to fall back to if DB unavailable
        
    Returns:
        Initialized configuration loader
    """
    global _global_config_loader, _global_config_wrapper
    
    _global_config_loader = DatabaseConfigLoader(
        database_url=database_url,
        refresh_interval=refresh_interval,
        enable_hot_reload=enable_hot_reload,
        fallback_config_path=fallback_config_path
    )
    
    _global_config_wrapper = LegacyConfigWrapper(_global_config_loader)
    
    return _global_config_loader


def get_config() -> LegacyConfigWrapper:
    """
    Get the global configuration wrapper
    Use this to replace 'import config' statements
    
    Returns:
        Configuration wrapper that behaves like the old config module
    """
    global _global_config_wrapper
    
    if _global_config_wrapper is None:
        # Auto-initialize with defaults
        initialize_config()
    
    return _global_config_wrapper


def get_config_loader() -> DatabaseConfigLoader:
    """
    Get the global configuration loader instance
    
    Returns:
        Configuration loader for advanced operations
    """
    global _global_config_loader
    
    if _global_config_loader is None:
        # Auto-initialize with defaults
        initialize_config()
    
    return _global_config_loader


# Convenience functions for common operations
def reload_config() -> bool:
    """Reload configuration from database"""
    return get_config_loader().force_refresh()


def set_config(setting_name: str, value: Any, changed_by: str = "system", reason: str = "") -> bool:
    """Update a configuration setting"""
    return get_config_loader().set(setting_name, value, changed_by, reason)


def get_config_value(setting_name: str, default: Any = None) -> Any:
    """Get a configuration value"""
    return get_config_loader().get(setting_name, default)


def apply_config_preset(preset_name: str, applied_by: str = "system") -> bool:
    """Apply a configuration preset"""
    return get_config_loader().apply_preset(preset_name, applied_by)


def backup_config(backup_name: str, description: str = "", created_by: str = "system") -> bool:
    """Backup current configuration"""
    return get_config_loader().backup_current_config(backup_name, description, created_by)


# Example usage and migration guide
if __name__ == "__main__":
    """
    Example usage and migration guide for replacing static config.py
    """
    
    # 1. Initialize the dynamic configuration system
    print("Initializing dynamic configuration...")
    config_loader = initialize_config(
        database_url="postgresql://user:pass@localhost:5432/forex_config",
        refresh_interval=60,
        enable_hot_reload=True
    )
    
    # 2. Get configuration wrapper (replaces 'import config')
    config = get_config()
    
    # 3. Use configuration exactly like before
    print(f"MIN_CONFIDENCE: {config.MIN_CONFIDENCE}")
    print(f"EPIC_LIST: {config.EPIC_LIST}")
    print(f"STRATEGY_WEIGHT_EMA: {config.STRATEGY_WEIGHT_EMA}")
    
    # 4. Update configuration dynamically
    config_loader.set("MIN_CONFIDENCE", 0.75, changed_by="admin", reason="Increased for better signals")
    
    # 5. Apply presets
    config_loader.apply_preset("Conservative", applied_by="admin")
    
    # 6. Check configuration status
    summary = config_loader.get_config_summary()
    print(f"Configuration summary: {summary}")
    
    # 7. Stop hot reload when done
    config_loader.stop_hot_reload()
    
    print("""
    Migration Guide:
    
    OLD CODE:
    --------
    import config
    confidence = config.MIN_CONFIDENCE
    epic_list = config.EPIC_LIST
    
    NEW CODE:
    --------
    from dynamic_config_loader import get_config
    config = get_config()
    confidence = config.MIN_CONFIDENCE
    epic_list = config.EPIC_LIST
    
    OR (for advanced usage):
    ----------------------
    from dynamic_config_loader import get_config_loader
    loader = get_config_loader()
    confidence = loader.get("MIN_CONFIDENCE", 0.7)
    loader.set("MIN_CONFIDENCE", 0.8, changed_by="user", reason="Optimization")
    """)