# core/intelligence/intelligence_config.py
"""
Configuration Wrapper for Market Intelligence Engine
Runtime configuration manager that uses the database-driven IntelligenceConfigService.

This provides backward compatibility while using database configuration as the source of truth.
The old configdata system is kept as a fallback for when database is unavailable.

Updated: 2026-01-03 - Now uses database-driven configuration via IntelligenceConfigService
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import database-driven config service (preferred)
try:
    from forex_scanner.services.intelligence_config_service import (
        get_intelligence_config_service,
        get_intelligence_config,
        IntelligenceConfig
    )
    HAS_DB_CONFIG = True
    logger.info("Database-driven intelligence config available")
except ImportError:
    HAS_DB_CONFIG = False
    logger.warning("Database intelligence config not available, will try configdata fallback")

# Fallback to configdata system
try:
    from forex_scanner.configdata import config as configdata_config
    HAS_CONFIGDATA = True
except ImportError:
    HAS_CONFIGDATA = False

# Final fallback - standalone mode
if not HAS_DB_CONFIG and not HAS_CONFIGDATA:
    logger.warning("No config system available, using standalone fallback mode")


class IntelligenceConfigManager:
    """
    Runtime configuration manager for Market Intelligence Engine

    Priority:
    1. Database-driven IntelligenceConfigService (preferred)
    2. ConfigData system (fallback)
    3. Standalone hardcoded presets (emergency fallback)
    """

    def __init__(self, preset: str = None):
        self.logger = logging.getLogger(__name__)
        self._db_config = None
        self.config_source = None
        self.config = {}
        self.preset = preset

        # Try database-driven config first (preferred)
        if HAS_DB_CONFIG:
            try:
                self._db_config = get_intelligence_config()
                self.preset = preset or self._db_config.preset
                self.logger.info(f"Using database-driven intelligence config with preset '{self.preset}'")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load database config: {e}, falling back")
                self._db_config = None

        # Fallback to configdata system
        if HAS_CONFIGDATA:
            self.config_source = configdata_config
            if preset is None:
                preset = self.config_source.get_intelligence_preset()
            self.preset = preset
            self.logger.info(f"Using configdata system with preset '{preset}'")
            return

        # Final fallback - standalone mode
        self.config_source = None
        self.preset = preset or 'collect_only'
        self.config = {}
        self.load_preset_fallback(self.preset)
        self.logger.warning(f"Using fallback mode with preset '{self.preset}'")

        # Try to import existing intelligence engine
        try:
            from .market_intelligence import MarketIntelligenceEngine
            self.intelligence_engine_class = MarketIntelligenceEngine
            self.has_intelligence_engine = True
        except ImportError:
            self.intelligence_engine_class = None
            self.has_intelligence_engine = False

    def _refresh_db_config(self):
        """Refresh database config if available"""
        if HAS_DB_CONFIG:
            try:
                self._db_config = get_intelligence_config()
            except Exception as e:
                self.logger.warning(f"Failed to refresh database config: {e}")

    def load_preset(self, preset_name: str):
        """Load a configuration preset"""
        if HAS_DB_CONFIG and self._db_config:
            # Use database service to update preset
            try:
                service = get_intelligence_config_service()
                if service.set_preset(preset_name):
                    self._refresh_db_config()
                    self.preset = preset_name
                    self.logger.info(f"Loaded preset '{preset_name}' via database")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to set preset via database: {e}")

        if HAS_CONFIGDATA and self.config_source:
            self.config_source.set_intelligence_preset(preset_name)
            self.preset = preset_name
            self.logger.info(f"Loaded preset '{preset_name}' via configdata system")
            return

        # Fallback to standalone mode
        self.load_preset_fallback(preset_name)

    def load_preset_fallback(self, preset_name: str):
        """Fallback preset loading when no config system is available"""
        presets = {
            'disabled': {
                'mode': 'disabled',
                'threshold': 0.0,
                'use_intelligence_engine': False,
                'components_enabled': {
                    'market_regime_filter': False,
                    'volatility_filter': False,
                    'volume_filter': False,
                    'time_filter': False,
                    'confidence_filter': True,
                },
                'description': 'No intelligence filtering - signals based on strategy only'
            },
            'minimal': {
                'mode': 'live_only',
                'threshold': 0.3,
                'use_intelligence_engine': False,
                'components_enabled': {
                    'market_regime_filter': False,
                    'volatility_filter': False,
                    'volume_filter': True,
                    'time_filter': False,
                    'confidence_filter': True,
                },
                'weights': {'volume': 0.5, 'confidence': 0.5},
                'description': 'Minimal filtering - volume + confidence only'
            },
            'balanced': {
                'mode': 'live_only',
                'threshold': 0.5,
                'use_intelligence_engine': True,
                'intelligence_weight': 0.4,
                'components_enabled': {
                    'market_regime_filter': True,
                    'volatility_filter': True,
                    'volume_filter': True,
                    'time_filter': False,
                    'confidence_filter': True,
                },
                'weights': {
                    'market_regime': 0.3,
                    'volatility': 0.2,
                    'volume': 0.2,
                    'confidence': 0.3
                },
                'description': 'Balanced filtering with MarketIntelligenceEngine'
            },
            'conservative': {
                'mode': 'enhanced',
                'threshold': 0.7,
                'use_intelligence_engine': True,
                'intelligence_weight': 0.6,
                'components_enabled': {
                    'market_regime_filter': True,
                    'volatility_filter': True,
                    'volume_filter': True,
                    'time_filter': True,
                    'confidence_filter': True,
                },
                'weights': {
                    'market_regime': 0.4,
                    'volatility': 0.2,
                    'volume': 0.2,
                    'confidence': 0.2
                },
                'description': 'Conservative filtering - fewer but higher quality signals'
            },
            'collect_only': {
                'mode': 'live_only',
                'threshold': 0.0,
                'use_intelligence_engine': True,  # Run engine for data collection
                'intelligence_weight': 0.0,  # But don't use for filtering
                'components_enabled': {
                    'market_regime_filter': False,
                    'volatility_filter': False,
                    'volume_filter': False,
                    'time_filter': False,
                    'confidence_filter': False,
                },
                'weights': {},
                'description': 'Full engine running for data collection, no signal filtering'
            },
            'testing': {
                'mode': 'backtest_consistent',
                'threshold': 0.4,
                'use_intelligence_engine': False,
                'components_enabled': {
                    'market_regime_filter': False,
                    'volatility_filter': True,
                    'volume_filter': True,
                    'time_filter': False,
                    'confidence_filter': True,
                },
                'weights': {
                    'volatility': 0.3,
                    'volume': 0.3,
                    'confidence': 0.4
                },
                'description': 'Testing mode - consistent with backtesting'
            }
        }

        if preset_name not in presets:
            self.logger.warning(f"Unknown preset '{preset_name}', using 'collect_only'")
            preset_name = 'collect_only'

        self.config = presets[preset_name].copy()
        self.preset = preset_name
        self.logger.info(f"Loaded '{preset_name}' preset: {self.config['description']}")

    def get_effective_threshold(self) -> float:
        """Get the current intelligence threshold"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.threshold

        if HAS_CONFIGDATA and self.config_source:
            return self.config_source.get_intelligence_threshold()

        return self.config.get('threshold', 0.0)

    def should_use_intelligence_engine(self) -> bool:
        """Check if we should use the MarketIntelligenceEngine"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.use_intelligence_engine

        if HAS_CONFIGDATA and self.config_source:
            preset_info = getattr(self.config_source.intelligence, 'INTELLIGENCE_PRESETS', {}).get(self.preset, {})
            return preset_info.get('use_intelligence_engine', True)

        return self.config.get('use_intelligence_engine', False)

    def get_intelligence_weight(self) -> float:
        """Get weight for intelligence engine in overall scoring"""
        if HAS_DB_CONFIG and self._db_config:
            # For collect_only mode, weight is 0 (no filtering)
            if self._db_config.preset == 'collect_only':
                return 0.0
            return 0.4  # Default weight

        if HAS_CONFIGDATA and self.config_source:
            preset_info = getattr(self.config_source.intelligence, 'INTELLIGENCE_PRESETS', {}).get(self.preset, {})
            return preset_info.get('intelligence_weight', 0.4)

        return self.config.get('intelligence_weight', 0.4)

    def is_component_enabled(self, component_name: str) -> bool:
        """Check if a specific intelligence component is enabled"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.is_component_enabled(component_name)

        if HAS_CONFIGDATA and self.config_source:
            components = getattr(self.config_source.intelligence, 'INTELLIGENCE_COMPONENTS_ENABLED', {})
            return components.get(component_name, True)

        return self.config.get('components_enabled', {}).get(component_name, True)

    def get_component_weight(self, component_name: str) -> float:
        """Get weight for a component"""
        clean_name = component_name.replace('_filter', '')

        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.get_component_weight(clean_name)

        if HAS_CONFIGDATA and self.config_source:
            weights = getattr(self.config_source.intelligence, 'INTELLIGENCE_WEIGHTS', {})
            return weights.get(clean_name, 0.2)

        return self.config.get('weights', {}).get(clean_name, 0.2)

    def get_regime_modifier(self, regime: str, strategy: str) -> float:
        """Get confidence modifier for regime-strategy combination"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.get_regime_modifier(regime, strategy)

        # Fallback - return 1.0 (no modification)
        return 1.0

    def calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate final intelligence score using configured weights"""
        # Check if intelligence is disabled
        if HAS_DB_CONFIG and self._db_config:
            if self._db_config.preset in ('disabled', 'collect_only'):
                return 1.0  # Always pass when disabled or collect_only

        mode = self.config.get('mode') if not (HAS_DB_CONFIG and self._db_config) else None
        if mode == 'disabled':
            return 1.0

        total_score = 0.0
        total_weight = 0.0

        for component, score in component_scores.items():
            component_filter = f"{component}_filter"
            if self.is_component_enabled(component_filter):
                weight = self.get_component_weight(component)
                total_score += score * weight
                total_weight += weight

        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 1.0  # No components enabled, pass all

        return min(1.0, max(0.0, final_score))

    def should_filter_signal(self, intelligence_score: float) -> bool:
        """Determine if signal should be filtered out"""
        if HAS_DB_CONFIG and self._db_config:
            if self._db_config.preset in ('disabled', 'collect_only'):
                return False  # Never filter when disabled or collect_only
            return intelligence_score < self._db_config.threshold

        mode = self.config.get('mode')
        if mode == 'disabled':
            return False

        threshold = self.get_effective_threshold()
        return intelligence_score < threshold

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current configuration status"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.get_summary()

        if HAS_CONFIGDATA and self.config_source:
            return self.config_source.get_intelligence_summary()

        # Fallback to local config
        enabled_components = [
            comp.replace('_filter', '')
            for comp, enabled in self.config.get('components_enabled', {}).items()
            if enabled
        ]

        return {
            'preset': self.preset,
            'mode': self.config.get('mode', 'unknown'),
            'threshold': self.get_effective_threshold(),
            'enabled_components': enabled_components,
            'component_count': len(enabled_components),
            'description': self.config.get('description', 'No description'),
            'is_disabled': self.config.get('mode') == 'disabled',
            'uses_intelligence_engine': self.should_use_intelligence_engine(),
            'intelligence_weight': self.get_intelligence_weight(),
            'source': 'database' if (HAS_DB_CONFIG and self._db_config) else 'fallback'
        }

    def log_signal_decision(self, epic: str, intelligence_score: float, passed: bool):
        """Log intelligence filtering decision"""
        threshold = self.get_effective_threshold()
        status = "PASSED" if passed else "FILTERED"
        engine_status = " (w/ Engine)" if self.should_use_intelligence_engine() else ""

        self.logger.info(
            f"{status} {epic}: Intelligence {intelligence_score:.1%} "
            f"{'≥' if passed else '<'} {threshold:.1%} "
            f"(preset: {self.preset}{engine_status})"
        )

    def get_market_regime_threshold(self) -> float:
        """Get threshold for market regime analysis"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.market_intelligence_min_confidence

        if HAS_CONFIGDATA and self.config_source:
            return getattr(self.config_source.intelligence, 'MARKET_INTELLIGENCE_MIN_CONFIDENCE', 0.5)

        return self.config.get('market_regime_threshold', 0.5)

    def is_smart_money_collection_enabled(self) -> bool:
        """Check if Smart Money data collection is enabled"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.enable_smart_money_collection

        return False

    def is_order_flow_collection_enabled(self) -> bool:
        """Check if Order Flow data collection is enabled"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.enable_order_flow_collection

        return False

    def get_cleanup_days(self) -> int:
        """Get number of days to retain intelligence data"""
        if HAS_DB_CONFIG and self._db_config:
            return self._db_config.intelligence_cleanup_days

        return 60  # Default


# Export the config system status
def get_config_system_status() -> Dict[str, Any]:
    """Get status of the intelligence configuration system"""
    return {
        'has_db_config': HAS_DB_CONFIG,
        'has_configdata': HAS_CONFIGDATA,
        'primary_source': 'database' if HAS_DB_CONFIG else ('configdata' if HAS_CONFIGDATA else 'fallback'),
    }


if __name__ == "__main__":
    # Test the configuration manager
    print("Testing Intelligence Configuration Manager")
    print("=" * 60)

    status = get_config_system_status()
    print(f"Config System Status: {status}")

    presets = ['disabled', 'minimal', 'balanced', 'conservative', 'collect_only', 'testing']

    for preset in presets:
        print(f"\nTesting preset: {preset}")
        config_mgr = IntelligenceConfigManager(preset)
        summary = config_mgr.get_status_summary()

        print(f"   Preset: {summary.get('preset')}")
        print(f"   Threshold: {summary.get('threshold', 0):.1%}")
        print(f"   Uses Engine: {summary.get('uses_intelligence_engine', False)}")
        print(f"   Components: {', '.join(summary.get('enabled_components', []))}")
        print(f"   Source: {summary.get('source', 'unknown')}")
