# validation/scanner_state_recreator.py
"""
Scanner State Recreator for Signal Validation

This module handles the recreation of historical scanner states, including:
- Configuration settings as they existed at validation time
- Strategy parameters and thresholds
- System settings and feature flags
- Market conditions and filters
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from copy import deepcopy

from forex_scanner.core.database import DatabaseManager
from forex_scanner.core.signal_detector import SignalDetector
from forex_scanner.core.data_fetcher import DataFetcher
from forex_scanner.core.processing.signal_processor import SignalProcessor
from forex_scanner.configdata import config as current_config
from forex_scanner import config as system_config

from .replay_config import ReplayConfig


class ScannerStateRecreator:
    """
    Recreates scanner state as it existed at a specific historical timestamp
    
    This class handles:
    - Configuration state management
    - Strategy parameter recreation
    - System settings restoration
    - Component initialization with historical settings
    """
    
    def __init__(self, db_manager: DatabaseManager, user_timezone: str = 'Europe/Stockholm'):
        """
        Initialize the scanner state recreator
        
        Args:
            db_manager: Database manager for state recreation
            user_timezone: User's timezone for timestamp handling
        """
        self.db_manager = db_manager
        self.user_timezone = user_timezone
        self.logger = logging.getLogger(__name__)
        
        # Store original configuration for restoration
        self._original_config = self._capture_current_config()
        
        # State tracking
        self._recreated_state = {}
        self._active_recreations = []
        
        self.logger.info(f"🔧 ScannerStateRecreator initialized")
        self.logger.info(f"   Timezone: {user_timezone}")
        self.logger.info(f"   Configuration time-travel: {'✅' if ReplayConfig.is_feature_enabled('enable_configuration_time_travel') else '❌'}")
    
    def recreate_scanner_state(
        self, 
        timestamp: datetime,
        epic_list: List[str] = None,
        strategy_filter: str = None
    ) -> Dict[str, Any]:
        """
        Recreate complete scanner state for the given timestamp
        
        Args:
            timestamp: Target timestamp for state recreation
            epic_list: List of epics to configure (optional)
            strategy_filter: Specific strategy to focus on (optional)
            
        Returns:
            Dictionary containing recreated scanner state
        """
        try:
            self.logger.info(f"🕒 Recreating scanner state for {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Ensure timestamp has timezone
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            # Build recreated state
            state = {
                'timestamp': timestamp,
                'configuration': self._recreate_configuration_state(timestamp, strategy_filter),
                'system_settings': self._recreate_system_settings(timestamp),
                'strategy_parameters': self._recreate_strategy_parameters(timestamp, strategy_filter),
                'market_filters': self._recreate_market_filters(timestamp),
                'epic_configuration': self._recreate_epic_configuration(epic_list or self._get_default_epic_list()),
                'feature_flags': self._recreate_feature_flags(timestamp),
                'metadata': {
                    'recreation_timestamp': datetime.now(timezone.utc),
                    'configuration_source': 'historical_recreation',
                    'strategy_filter': strategy_filter,
                    'epic_count': len(epic_list or self._get_default_epic_list())
                }
            }
            
            # Apply configuration changes if time-travel is enabled
            if ReplayConfig.is_feature_enabled('enable_configuration_time_travel'):
                self._apply_historical_configuration(state)
            
            # Store recreated state for cleanup
            self._recreated_state = state
            self._active_recreations.append(timestamp)
            
            self.logger.info(f"✅ Scanner state recreated")
            self.logger.info(f"   Strategies: {list(state['strategy_parameters'].keys())}")
            self.logger.info(f"   Epics: {len(state['epic_configuration'])}")
            self.logger.info(f"   Features: {sum(1 for v in state['feature_flags'].values() if v)}/{len(state['feature_flags'])}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Error recreating scanner state: {e}")
            return {}
    
    def create_signal_detector(self, state: Dict[str, Any]) -> Optional[SignalDetector]:
        """
        Create a SignalDetector configured with historical state
        
        Args:
            state: Recreated scanner state
            
        Returns:
            Configured SignalDetector instance
        """
        try:
            self.logger.info("🔧 Creating SignalDetector with historical configuration")
            
            # Create signal detector with recreated state
            signal_detector = SignalDetector(self.db_manager, self.user_timezone)
            
            # Configure strategies based on recreated state
            self._configure_strategies(signal_detector, state)
            
            return signal_detector
            
        except Exception as e:
            self.logger.error(f"❌ Error creating SignalDetector: {e}")
            return None
    
    def create_signal_processor(self, state: Dict[str, Any]) -> Optional[SignalProcessor]:
        """
        Create a SignalProcessor configured with historical state
        
        Args:
            state: Recreated scanner state
            
        Returns:
            Configured SignalProcessor instance
        """
        try:
            if not ReplayConfig.is_feature_enabled('enable_smart_money_validation'):
                return None
            
            self.logger.info("🧠 Creating SignalProcessor with historical configuration")
            
            # Create data fetcher for SignalProcessor
            data_fetcher = DataFetcher(self.db_manager, self.user_timezone)
            
            # Create SignalProcessor with historical settings
            processor = SignalProcessor(
                db_manager=self.db_manager,
                data_fetcher=data_fetcher,
                alert_history=None,  # Not needed for replay
                claude_analyzer=None,  # Will be configured separately
                notification_manager=None  # Not needed for replay
            )
            
            # Configure processor with historical state
            self._configure_signal_processor(processor, state)
            
            return processor
            
        except Exception as e:
            self.logger.error(f"❌ Error creating SignalProcessor: {e}")
            return None
    
    def restore_original_configuration(self) -> None:
        """Restore original configuration after validation"""
        try:
            if not self._original_config:
                return
            
            self.logger.info("🔄 Restoring original configuration")
            
            # Restore configuration values
            for key, value in self._original_config.items():
                if hasattr(system_config, key):
                    setattr(system_config, key, value)
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
            
            # Clear active recreations
            self._active_recreations.clear()
            self._recreated_state.clear()
            
            self.logger.info("✅ Original configuration restored")
            
        except Exception as e:
            self.logger.error(f"❌ Error restoring configuration: {e}")
    
    def _recreate_configuration_state(self, timestamp: datetime, strategy_filter: str = None) -> Dict[str, Any]:
        """Recreate configuration state for timestamp"""
        try:
            # Start with current configuration
            config_state = {
                'timestamp': timestamp,
                'min_confidence': getattr(system_config, 'MIN_CONFIDENCE', 0.7),
                'spread_pips': getattr(system_config, 'SPREAD_PIPS', 1.5),
                'use_bid_adjustment': getattr(system_config, 'USE_BID_ADJUSTMENT', False),
                'default_timeframe': getattr(system_config, 'DEFAULT_TIMEFRAME', '15m'),
                'min_bars_for_signal': getattr(system_config, 'MIN_BARS_FOR_SIGNAL', 200)
            }
            
            # Add strategy-specific configuration
            if strategy_filter:
                config_state['strategy_filter'] = strategy_filter
                config_state['strategy_config'] = self._get_strategy_configuration(strategy_filter, timestamp)
            
            # Historical configuration adjustments could go here
            # For now, use current configuration as baseline
            
            return config_state
            
        except Exception as e:
            self.logger.error(f"❌ Error recreating configuration state: {e}")
            return {}
    
    def _recreate_system_settings(self, timestamp: datetime) -> Dict[str, Any]:
        """Recreate system settings for timestamp"""
        try:
            return {
                'timestamp': timestamp,
                'scan_interval': getattr(system_config, 'SCAN_INTERVAL', 60),
                'database_url': getattr(system_config, 'DATABASE_URL', ''),
                'user_timezone': self.user_timezone,
                'enable_notifications': getattr(system_config, 'ENABLE_NOTIFICATIONS', True),
                'enable_claude_analysis': getattr(system_config, 'ENABLE_CLAUDE_ANALYSIS', True),
                'save_to_database': getattr(system_config, 'SAVE_TO_DATABASE', True)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error recreating system settings: {e}")
            return {}
    
    def _recreate_strategy_parameters(self, timestamp: datetime, strategy_filter: str = None) -> Dict[str, Dict]:
        """Recreate strategy parameters for timestamp"""
        try:
            strategies = {}
            
            # Define available strategies
            available_strategies = ['ema', 'macd', 'kama', 'zero_lag', 'momentum_bias']
            
            if strategy_filter:
                available_strategies = [s for s in available_strategies if strategy_filter.lower() in s.lower()]
            
            for strategy_name in available_strategies:
                if self._is_strategy_enabled(strategy_name, timestamp):
                    strategies[strategy_name] = self._get_strategy_configuration(strategy_name, timestamp)
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"❌ Error recreating strategy parameters: {e}")
            return {}
    
    def _recreate_market_filters(self, timestamp: datetime) -> Dict[str, Any]:
        """Recreate market filters for timestamp"""
        try:
            return {
                'timestamp': timestamp,
                'large_candle_filter_enabled': getattr(system_config, 'LARGE_CANDLE_FILTER_ENABLED', True),
                'volume_filter_enabled': getattr(system_config, 'VOLUME_FILTER_ENABLED', False),
                'volatility_filter_enabled': getattr(system_config, 'VOLATILITY_FILTER_ENABLED', False),
                'time_filter_enabled': getattr(system_config, 'TIME_FILTER_ENABLED', False),
                'news_filter_enabled': getattr(system_config, 'NEWS_FILTER_ENABLED', False)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error recreating market filters: {e}")
            return {}
    
    def _recreate_epic_configuration(self, epic_list: List[str]) -> Dict[str, Dict]:
        """Recreate epic-specific configuration"""
        try:
            epic_config = {}
            
            for epic in epic_list:
                # Get pair info
                pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
                
                epic_config[epic] = {
                    'pair': pair,
                    'pip_multiplier': 10000 if 'JPY' not in pair else 100,
                    'spread_pips': self._get_epic_spread(epic),
                    'enabled': True,
                    'strategy_overrides': self._get_epic_strategy_overrides(epic)
                }
            
            return epic_config
            
        except Exception as e:
            self.logger.error(f"❌ Error recreating epic configuration: {e}")
            return {}
    
    def _recreate_feature_flags(self, timestamp: datetime) -> Dict[str, bool]:
        """Recreate feature flags for timestamp"""
        try:
            # Base feature flags (these would be historically adjusted)
            return {
                'enable_deduplication': getattr(system_config, 'ENABLE_ALERT_DEDUPLICATION', True),
                'enable_smart_money': getattr(system_config, 'SMART_MONEY_READONLY_ENABLED', False),
                'enable_claude_analysis': getattr(system_config, 'ENABLE_CLAUDE_ANALYSIS', True),
                'enable_mtf_analysis': getattr(system_config, 'ENABLE_MTF_ANALYSIS', True),
                'enable_momentum_bias': getattr(system_config, 'MOMENTUM_BIAS_STRATEGY', False),
                'enable_large_candle_filter': getattr(system_config, 'LARGE_CANDLE_FILTER_ENABLED', True)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error recreating feature flags: {e}")
            return {}
    
    def _apply_historical_configuration(self, state: Dict[str, Any]) -> None:
        """Apply historical configuration to current system"""
        try:
            config_data = state.get('configuration', {})
            
            # Apply main configuration values
            for key, value in config_data.items():
                if key != 'timestamp' and hasattr(system_config, key.upper()):
                    setattr(system_config, key.upper(), value)
            
            # Apply feature flags
            feature_flags = state.get('feature_flags', {})
            for flag, enabled in feature_flags.items():
                flag_name = flag.upper()
                if hasattr(system_config, flag_name):
                    setattr(system_config, flag_name, enabled)
            
            self.logger.debug("🔧 Historical configuration applied")
            
        except Exception as e:
            self.logger.error(f"❌ Error applying historical configuration: {e}")
    
    def _configure_strategies(self, signal_detector: SignalDetector, state: Dict[str, Any]) -> None:
        """Configure strategies based on recreated state"""
        try:
            strategy_params = state.get('strategy_parameters', {})
            
            for strategy_name, config_data in strategy_params.items():
                strategy_obj = getattr(signal_detector, f'{strategy_name}_strategy', None)
                if strategy_obj:
                    # Apply strategy-specific configuration
                    self._apply_strategy_config(strategy_obj, config_data)
            
        except Exception as e:
            self.logger.error(f"❌ Error configuring strategies: {e}")
    
    def _configure_signal_processor(self, processor: SignalProcessor, state: Dict[str, Any]) -> None:
        """Configure SignalProcessor with historical state"""
        try:
            config_data = state.get('configuration', {})
            
            # Configure processor settings
            processor.min_confidence = config_data.get('min_confidence', 0.7)
            processor.enable_smart_money = state.get('feature_flags', {}).get('enable_smart_money', False)
            processor.enable_deduplication = state.get('feature_flags', {}).get('enable_deduplication', True)
            
            self.logger.debug("🔧 SignalProcessor configured with historical state")
            
        except Exception as e:
            self.logger.error(f"❌ Error configuring SignalProcessor: {e}")
    
    def _capture_current_config(self) -> Dict[str, Any]:
        """Capture current configuration for restoration"""
        try:
            config_snapshot = {}
            
            # Capture key configuration values
            config_attrs = [
                'MIN_CONFIDENCE', 'SPREAD_PIPS', 'USE_BID_ADJUSTMENT', 'DEFAULT_TIMEFRAME',
                'ENABLE_CLAUDE_ANALYSIS', 'ENABLE_NOTIFICATIONS', 'SAVE_TO_DATABASE',
                'SMART_MONEY_READONLY_ENABLED', 'ENABLE_ALERT_DEDUPLICATION'
            ]
            
            for attr in config_attrs:
                if hasattr(system_config, attr):
                    config_snapshot[attr] = getattr(system_config, attr)
            
            return config_snapshot
            
        except Exception as e:
            self.logger.error(f"❌ Error capturing current config: {e}")
            return {}
    
    def _is_strategy_enabled(self, strategy_name: str, timestamp: datetime) -> bool:
        """Check if strategy was enabled at timestamp"""
        # This would check historical strategy enablement
        # For now, check current configuration
        strategy_flags = {
            'ema': getattr(system_config, 'SIMPLE_EMA_STRATEGY', True),
            'macd': getattr(system_config, 'MACD_EMA_STRATEGY', False),
            'kama': getattr(system_config, 'KAMA_STRATEGY', False),
            'zero_lag': getattr(system_config, 'ZERO_LAG_STRATEGY', False),
            'momentum_bias': getattr(system_config, 'MOMENTUM_BIAS_STRATEGY', False)
        }
        
        return strategy_flags.get(strategy_name, False)
    
    def _get_strategy_configuration(self, strategy_name: str, timestamp: datetime) -> Dict[str, Any]:
        """Get strategy configuration for timestamp"""
        try:
            # This would fetch historical strategy configuration
            # For now, return current configuration
            strategy_configs = {
                'ema': {
                    'short_period': 21,
                    'long_period': 50,
                    'trend_period': 200,
                    'momentum_confirmation': True,
                    'mtf_analysis': True
                },
                'macd': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9,
                    'histogram_threshold': 0.0
                },
                'kama': {
                    'period': 14,
                    'min_efficiency': 0.1,
                    'trend_threshold': 0.05
                },
                'zero_lag': {
                    'period': 21,
                    'gain_limit': 50
                },
                'momentum_bias': {
                    'period': 14,
                    'threshold': 0.5
                }
            }
            
            return strategy_configs.get(strategy_name, {})
            
        except Exception as e:
            self.logger.error(f"❌ Error getting strategy configuration: {e}")
            return {}
    
    def _get_default_epic_list(self) -> List[str]:
        """Get default epic list"""
        return getattr(system_config, 'EPIC_LIST', ReplayConfig.DEFAULT_EPIC_LIST)
    
    def _get_epic_spread(self, epic: str) -> float:
        """Get spread for epic"""
        # This would return historical spread data
        # For now, return default
        return getattr(system_config, 'SPREAD_PIPS', 1.5)
    
    def _get_epic_strategy_overrides(self, epic: str) -> Dict[str, Any]:
        """Get epic-specific strategy overrides"""
        # This would return epic-specific strategy configurations
        return {}
    
    def _apply_strategy_config(self, strategy_obj, config_data: Dict[str, Any]) -> None:
        """Apply configuration to strategy object"""
        try:
            for key, value in config_data.items():
                if hasattr(strategy_obj, key):
                    setattr(strategy_obj, key, value)
        except Exception as e:
            self.logger.error(f"❌ Error applying strategy config: {e}")
    
    def get_active_recreations(self) -> List[datetime]:
        """Get list of active recreations"""
        return self._active_recreations.copy()
    
    def cleanup_recreations(self) -> None:
        """Clean up all recreated states"""
        self.restore_original_configuration()
        self.logger.info("🧹 Scanner state recreations cleaned up")