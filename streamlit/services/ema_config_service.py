"""
EMA Configuration Service for Streamlit
Provides dynamic EMA configuration based on epic and market conditions
"""

import os
import logging
import sys
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

# Add worker app to path for imports
worker_path = os.path.join(os.path.dirname(__file__), '../../worker/app')
if worker_path not in sys.path:
    sys.path.append(worker_path)

# Try to import pandas (optional for optimization service)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class EMAConfiguration:
    """EMA configuration for a specific epic"""
    epic: str
    short_period: int
    long_period: int
    trend_period: int
    source: str  # 'optimal', 'config', 'default'
    config_name: str
    description: str
    last_updated: Optional[datetime] = None
    performance_score: Optional[float] = None


class EMAConfigService:
    """Service to provide dynamic EMA configurations for chart visualization"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EMAConfigService")
        self._cache = {}  # Cache configurations for performance
        self._cache_duration = 300  # 5 minutes cache duration

        # Default fallback configuration
        self.default_config = EMAConfiguration(
            epic="default",
            short_period=21,
            long_period=50,
            trend_period=200,
            source="default",
            config_name="default",
            description="Default EMA configuration"
        )

    def get_ema_config(self, epic: str) -> EMAConfiguration:
        """
        Get EMA configuration for a specific epic

        Args:
            epic: Trading epic (e.g., 'CS.D.EURUSD.MINI.IP')

        Returns:
            EMAConfiguration object with the best available configuration
        """
        # Check cache first
        cache_key = f"ema_config_{epic}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['config']

        try:
            # Try to get optimal parameters from database first
            config = self._get_optimal_config(epic)
            if config:
                self._cache_config(cache_key, config)
                return config

            # Fallback to config-based parameters
            config = self._get_config_based(epic)
            if config:
                self._cache_config(cache_key, config)
                return config

        except Exception as e:
            self.logger.warning(f"Error getting EMA config for {epic}: {e}")

        # Final fallback to default
        self.logger.info(f"Using default EMA configuration for {epic}")
        return self.default_config

    def _get_optimal_config(self, epic: str) -> Optional[EMAConfiguration]:
        """Try to get optimal EMA configuration from database"""
        if not PANDAS_AVAILABLE:
            self.logger.debug("Pandas not available, skipping optimal config lookup")
            return None

        try:
            # Import the optimization service
            from forex_scanner.optimization.optimal_parameter_service import (
                get_epic_ema_config, get_epic_optimal_parameters
            )

            # Try to get EMA-specific optimal config
            optimal_ema = get_epic_ema_config(epic)
            if optimal_ema:
                return EMAConfiguration(
                    epic=epic,
                    short_period=optimal_ema.get('short', 21),
                    long_period=optimal_ema.get('long', 50),
                    trend_period=optimal_ema.get('trend', 200),
                    source="optimal",
                    config_name=optimal_ema.get('config_name', 'optimized'),
                    description=optimal_ema.get('description', 'Database-optimized EMA configuration'),
                    last_updated=datetime.now()
                )

            # Try to get general optimal parameters
            optimal_params = get_epic_optimal_parameters(epic)
            if optimal_params and hasattr(optimal_params, 'ema_config'):
                ema_periods = self._parse_ema_config(optimal_params.ema_config)
                if ema_periods:
                    return EMAConfiguration(
                        epic=epic,
                        short_period=ema_periods['short'],
                        long_period=ema_periods['long'],
                        trend_period=ema_periods['trend'],
                        source="optimal",
                        config_name=optimal_params.ema_config,
                        description=f"Optimized EMA configuration from {optimal_params.ema_config}",
                        last_updated=optimal_params.last_optimized,
                        performance_score=getattr(optimal_params, 'performance_score', None)
                    )

        except ImportError:
            self.logger.debug("Optimization service not available")
        except Exception as e:
            self.logger.debug(f"Could not get optimal config for {epic}: {e}")

        return None

    def _get_config_based(self, epic: str) -> Optional[EMAConfiguration]:
        """Get EMA configuration from config files"""
        try:
            # Import the config modules
            from forex_scanner.configdata.strategies.config_ema_strategy import (
                get_ema_config_for_epic, EMA_STRATEGY_CONFIG, ACTIVE_EMA_CONFIG
            )

            # Get epic-specific configuration
            config_dict = get_ema_config_for_epic(epic)
            config_name = ACTIVE_EMA_CONFIG

            if config_dict:
                return EMAConfiguration(
                    epic=epic,
                    short_period=config_dict.get('short', 21),
                    long_period=config_dict.get('long', 50),
                    trend_period=config_dict.get('trend', 200),
                    source="config",
                    config_name=config_name,
                    description=config_dict.get('description', f'Configuration-based EMA settings ({config_name})')
                )

        except ImportError:
            self.logger.debug("Config modules not available")
        except Exception as e:
            self.logger.debug(f"Could not get config-based settings for {epic}: {e}")

        return None

    def _parse_ema_config(self, config_name: str) -> Optional[Dict]:
        """Parse EMA config name to get periods"""
        try:
            from forex_scanner.configdata.strategies.config_ema_strategy import EMA_STRATEGY_CONFIG
            return EMA_STRATEGY_CONFIG.get(config_name)
        except:
            return None

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached config is still valid"""
        if cache_key not in self._cache:
            return False

        cache_time = self._cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self._cache_duration

    def _cache_config(self, cache_key: str, config: EMAConfiguration):
        """Cache configuration for performance"""
        self._cache[cache_key] = {
            'config': config,
            'timestamp': datetime.now()
        }

    def get_available_configurations(self) -> List[Dict]:
        """Get list of all available EMA configurations"""
        configurations = []

        try:
            from forex_scanner.configdata.strategies.config_ema_strategy import EMA_STRATEGY_CONFIG

            for name, config in EMA_STRATEGY_CONFIG.items():
                configurations.append({
                    'name': name,
                    'short': config.get('short'),
                    'long': config.get('long'),
                    'trend': config.get('trend'),
                    'description': config.get('description', ''),
                    'source': 'config'
                })

        except Exception as e:
            self.logger.debug(f"Could not load available configurations: {e}")

        return configurations

    def get_epic_performance_summary(self, epic: str) -> Optional[Dict]:
        """Get performance summary for epic's EMA configuration"""
        try:
            from forex_scanner.optimization.optimal_parameter_service import get_epic_optimal_parameters

            optimal_params = get_epic_optimal_parameters(epic)
            if optimal_params:
                return {
                    'performance_score': getattr(optimal_params, 'performance_score', None),
                    'last_optimized': getattr(optimal_params, 'last_optimized', None),
                    'confidence_threshold': getattr(optimal_params, 'confidence_threshold', None),
                    'risk_reward_ratio': getattr(optimal_params, 'risk_reward_ratio', None)
                }

        except Exception as e:
            self.logger.debug(f"Could not get performance summary for {epic}: {e}")

        return None

    def refresh_cache(self, epic: str = None):
        """Refresh cached configurations"""
        if epic:
            cache_key = f"ema_config_{epic}"
            if cache_key in self._cache:
                del self._cache[cache_key]
        else:
            self._cache.clear()

        self.logger.info(f"Cache refreshed for {epic if epic else 'all epics'}")


# Global service instance
_ema_config_service = None

def get_ema_config_service() -> EMAConfigService:
    """Get singleton instance of EMA config service"""
    global _ema_config_service
    if _ema_config_service is None:
        _ema_config_service = EMAConfigService()
    return _ema_config_service


def get_dynamic_ema_config(epic: str) -> EMAConfiguration:
    """Convenience function to get EMA configuration for an epic"""
    service = get_ema_config_service()
    return service.get_ema_config(epic)


def get_ema_periods_for_chart(epic: str) -> Tuple[int, int, int]:
    """Get EMA periods as tuple for chart usage"""
    config = get_dynamic_ema_config(epic)
    return config.short_period, config.long_period, config.trend_period


def get_ema_config_summary(epic: str) -> Dict:
    """Get comprehensive configuration summary for display"""
    service = get_ema_config_service()
    config = service.get_ema_config(epic)
    performance = service.get_epic_performance_summary(epic)

    summary = {
        'epic': config.epic,
        'periods': {
            'short': config.short_period,
            'long': config.long_period,
            'trend': config.trend_period
        },
        'source': config.source,
        'config_name': config.config_name,
        'description': config.description,
        'last_updated': config.last_updated
    }

    if performance:
        summary['performance'] = performance

    return summary