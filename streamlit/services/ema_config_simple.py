"""
Simple EMA Configuration Service for Streamlit
Self-contained service that doesn't depend on worker app files
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

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


class EMAConfigServiceSimple:
    """Simplified EMA configuration service for Streamlit"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EMAConfigServiceSimple")
        self._cache = {}  # Cache configurations for performance
        self._cache_duration = 300  # 5 minutes cache duration

        # Static configuration mapping (copied from worker config)
        self.ema_configs = {
            'default': {
                'short': 21, 'long': 50, 'trend': 200,
                'description': 'Less choppy configuration for cleaner signals - 21/50/200 EMAs'
            },
            'conservative': {
                'short': 20, 'long': 50, 'trend': 200,
                'description': 'Conservative approach for low volatility trending markets'
            },
            'aggressive': {
                'short': 5, 'long': 13, 'trend': 50,
                'description': 'Fast-reacting configuration for high volatility breakouts'
            },
            'scalping': {
                'short': 3, 'long': 8, 'trend': 21,
                'description': 'Ultra-fast configuration for scalping strategies'
            },
            'swing': {
                'short': 25, 'long': 55, 'trend': 200,
                'description': 'Slow and steady for swing trading'
            },
            'news_safe': {
                'short': 15, 'long': 30, 'trend': 200,
                'description': 'Safer configuration during news events'
            },
            'crypto': {
                'short': 7, 'long': 25, 'trend': 99,
                'description': 'Adapted for crypto-like high volatility markets'
            }
        }

        # Epic-specific configuration mapping
        self.epic_config_mapping = {
            'CS.D.EURUSD.MINI.IP': 'aggressive',
            'CS.D.GBPUSD.MINI.IP': 'aggressive',
            'CS.D.USDJPY.MINI.IP': 'default',
            'CS.D.AUDUSD.MINI.IP': 'conservative',
            'CS.D.NZDUSD.MINI.IP': 'conservative',
            'CS.D.USDCAD.MINI.IP': 'conservative',
            'CS.D.USDCHF.MINI.IP': 'default',
            'CS.D.AUDJPY.MINI.IP': 'aggressive',  # This should show 5/13/50
            'CS.D.EURJPY.MINI.IP': 'aggressive',
            'CS.D.GBPJPY.MINI.IP': 'scalping',
            'CS.D.EURGBP.MINI.IP': 'default',
            'CS.D.EURAUD.MINI.IP': 'swing',
            'CS.D.GBPAUD.MINI.IP': 'scalping'
        }

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
        """Get EMA configuration for a specific epic"""

        # Check cache first
        cache_key = f"ema_config_{epic}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['config']

        try:
            # Get configuration based on epic mapping
            config_name = self.epic_config_mapping.get(epic, 'default')
            config_dict = self.ema_configs.get(config_name, self.ema_configs['default'])

            config = EMAConfiguration(
                epic=epic,
                short_period=config_dict['short'],
                long_period=config_dict['long'],
                trend_period=config_dict['trend'],
                source="config",
                config_name=config_name,
                description=config_dict['description'],
                last_updated=datetime.now()
            )

            # Cache the result
            self._cache_config(cache_key, config)
            return config

        except Exception as e:
            self.logger.warning(f"Error getting EMA config for {epic}: {e}")
            return self.default_config

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

    def refresh_cache(self, epic: str = None):
        """Refresh cached configurations"""
        if epic:
            cache_key = f"ema_config_{epic}"
            if cache_key in self._cache:
                del self._cache[cache_key]
        else:
            self._cache.clear()

        self.logger.info(f"Cache refreshed for {epic if epic else 'all epics'}")

    def get_available_configurations(self) -> list:
        """Get list of all available configurations"""
        return [
            {
                'name': name,
                'short': config['short'],
                'long': config['long'],
                'trend': config['trend'],
                'description': config['description']
            }
            for name, config in self.ema_configs.items()
        ]


# Global service instance
_ema_config_service = None

def get_ema_config_service_simple() -> EMAConfigServiceSimple:
    """Get singleton instance of simple EMA config service"""
    global _ema_config_service
    if _ema_config_service is None:
        _ema_config_service = EMAConfigServiceSimple()
    return _ema_config_service


def get_dynamic_ema_config_simple(epic: str) -> EMAConfiguration:
    """Convenience function to get EMA configuration for an epic"""
    service = get_ema_config_service_simple()
    return service.get_ema_config(epic)


def get_ema_periods_for_chart_simple(epic: str) -> Tuple[int, int, int]:
    """Get EMA periods as tuple for chart usage"""
    config = get_dynamic_ema_config_simple(epic)
    return config.short_period, config.long_period, config.trend_period


def get_ema_config_summary_simple(epic: str) -> Dict:
    """Get comprehensive configuration summary for display"""
    service = get_ema_config_service_simple()
    config = service.get_ema_config(epic)

    return {
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