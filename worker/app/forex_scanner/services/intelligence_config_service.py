"""
Intelligence System Configuration Service

Provides database-driven configuration loading for the Market Intelligence system with:
- In-memory caching with configurable TTL
- Last-known-good fallback when DB unavailable
- Preset-based configuration with component enablement
- Regime-strategy confidence modifiers
- Hot reload support via configurable refresh intervals
"""

import os
import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from threading import RLock
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


@dataclass
class IntelligencePreset:
    """Definition of an intelligence preset"""
    name: str
    threshold: float
    use_intelligence_engine: bool
    description: str
    components_enabled: Dict[str, bool] = field(default_factory=dict)


@dataclass
class IntelligenceConfig:
    """Complete configuration object for Market Intelligence system"""

    # Core Settings
    preset: str = "collect_only"
    mode: str = "live_only"
    enabled: bool = True

    # Current preset details (loaded from presets table)
    threshold: float = 0.0
    use_intelligence_engine: bool = True

    # Component Weights
    weight_market_regime: float = 0.25
    weight_volatility: float = 0.25
    weight_volume: float = 0.25
    weight_time: float = 0.0
    weight_confidence: float = 0.25

    # Component Enablement
    enable_market_regime_filter: bool = False
    enable_volatility_filter: bool = False
    enable_volume_filter: bool = False
    enable_time_filter: bool = False
    enable_confidence_filter: bool = False
    enable_spread_filter: bool = False
    enable_recent_signals_filter: bool = True

    # Smart Money Settings
    enable_smart_money_collection: bool = True
    enable_order_flow_collection: bool = True
    smart_money_structure_validation: bool = False
    smart_money_order_flow_validation: bool = False

    # Enhanced Regime Detection (ADX-based)
    enhanced_regime_detection_enabled: bool = True
    adx_trending_threshold: int = 25
    adx_strong_trend_threshold: int = 40
    adx_weak_trend_threshold: int = 20
    ema_alignment_weight: float = 0.4
    adx_weight: float = 0.4
    momentum_weight: float = 0.2
    trending_score_threshold: float = 0.55
    ranging_score_threshold: float = 0.55
    breakout_score_threshold: float = 0.60
    collect_enhanced_regime_data: bool = True
    log_regime_comparison: bool = True
    separate_volatility_from_structure: bool = True
    volatility_as_modifier: bool = True

    # Probabilistic Confidence Modifiers
    enable_probabilistic_confidence_modifiers: bool = True
    min_confidence_modifier: float = 0.5

    # Market Bias Filter
    market_bias_filter_enabled: bool = False
    market_bias_min_consensus: float = 0.70
    market_intelligence_min_confidence: float = 0.45
    block_unsuitable_regimes: bool = False

    # Storage Settings
    enable_intelligence_storage: bool = True
    intelligence_cleanup_days: int = 60

    # Analysis Settings
    force_intelligence_analysis: bool = True
    intelligence_override_market_hours: bool = True
    use_historical_data_for_intelligence: bool = True
    intelligence_confidence_threshold: float = 0.3
    intelligence_volume_threshold: float = 0.2
    intelligence_volatility_min: float = 0.1
    intelligence_allow_low_volatility: bool = True
    intelligence_allow_ranging_markets: bool = True

    # Scanner Settings
    live_scanner_lookback_hours: int = 24
    enable_recent_historical_scan: bool = True
    use_backtest_data_logic_for_live: bool = True
    force_market_open: bool = False
    enable_weekend_scanning: bool = False
    max_data_age_minutes: int = 60
    minimum_candles_for_live_scan: int = 50

    # Debug Settings
    intelligence_debug_mode: bool = False
    intelligence_log_rejections: bool = True

    # Claude AI Integration
    claude_integrate_with_intelligence: bool = True
    claude_use_intelligence_in_prompts: bool = True
    claude_override_intelligence: bool = True
    intelligence_claude_weight: float = 0.3

    # Available presets (loaded from database)
    _presets: Dict[str, IntelligencePreset] = field(default_factory=dict)

    # Regime-Strategy modifiers (loaded from database)
    _regime_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Metadata
    loaded_at: datetime = field(default_factory=datetime.now)
    source: str = "database"  # 'database', 'cache', 'default'

    def get_preset_config(self, preset_name: str = None) -> Optional[IntelligencePreset]:
        """Get configuration for a specific preset"""
        name = preset_name or self.preset
        return self._presets.get(name)

    def get_regime_modifier(self, regime: str, strategy: str) -> float:
        """
        Get confidence modifier for regime-strategy combination.

        Returns 1.0 (no modification) if not found.
        """
        if regime not in self._regime_modifiers:
            return 1.0
        return self._regime_modifiers[regime].get(strategy, 1.0)

    def get_all_regime_modifiers(self, regime: str) -> Dict[str, float]:
        """Get all strategy modifiers for a given regime"""
        return self._regime_modifiers.get(regime, {})

    def get_component_weight(self, component: str) -> float:
        """Get weight for a specific intelligence component"""
        weight_map = {
            'market_regime': self.weight_market_regime,
            'volatility': self.weight_volatility,
            'volume': self.weight_volume,
            'time': self.weight_time,
            'confidence': self.weight_confidence,
        }
        return weight_map.get(component, 0.0)

    def is_component_enabled(self, component: str) -> bool:
        """Check if a specific intelligence component is enabled"""
        component_map = {
            'market_regime_filter': self.enable_market_regime_filter,
            'volatility_filter': self.enable_volatility_filter,
            'volume_filter': self.enable_volume_filter,
            'time_filter': self.enable_time_filter,
            'confidence_filter': self.enable_confidence_filter,
            'spread_filter': self.enable_spread_filter,
            'recent_signals_filter': self.enable_recent_signals_filter,
        }
        return component_map.get(component, False)

    def is_smart_money_enabled(self) -> bool:
        """Check if Smart Money data collection is enabled"""
        return self.enable_smart_money_collection or self.enable_order_flow_collection

    def get_enabled_components(self) -> List[str]:
        """Get list of enabled component names"""
        components = []
        if self.enable_market_regime_filter:
            components.append('market_regime')
        if self.enable_volatility_filter:
            components.append('volatility')
        if self.enable_volume_filter:
            components.append('volume')
        if self.enable_time_filter:
            components.append('time')
        if self.enable_confidence_filter:
            components.append('confidence')
        if self.enable_spread_filter:
            components.append('spread')
        if self.enable_recent_signals_filter:
            components.append('recent_signals')
        return components

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current intelligence configuration"""
        return {
            'preset': self.preset,
            'mode': self.mode,
            'enabled': self.enabled,
            'threshold': self.threshold,
            'use_engine': self.use_intelligence_engine,
            'enabled_components': self.get_enabled_components(),
            'smart_money_collection': self.enable_smart_money_collection,
            'order_flow_collection': self.enable_order_flow_collection,
            'enhanced_regime_detection': self.enhanced_regime_detection_enabled,
            'storage_enabled': self.enable_intelligence_storage,
            'cleanup_days': self.intelligence_cleanup_days,
            'debug_mode': self.intelligence_debug_mode,
            'source': self.source,
            'loaded_at': self.loaded_at.isoformat() if self.loaded_at else None,
        }


class IntelligenceConfigService:
    """
    Database-driven configuration service for Market Intelligence.

    Features:
    - Loads config from strategy_config database
    - Caches in memory with configurable TTL
    - Falls back to last-known-good config if DB unavailable
    - Thread-safe operations
    - Hot reload before each scan cycle
    """

    def __init__(
        self,
        database_url: str = None,
        cache_ttl_seconds: int = 300,
        enable_hot_reload: bool = True
    ):
        self.database_url = database_url or self._get_default_database_url()
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.enable_hot_reload = enable_hot_reload

        # Thread-safe cache
        self._lock = RLock()
        self._cached_config: Optional[IntelligenceConfig] = None
        self._cache_timestamp: Optional[datetime] = None
        self._last_known_good: Optional[IntelligenceConfig] = None

        # Load initial configuration
        self._load_initial_config()

    def _get_default_database_url(self) -> str:
        """Get database URL from environment"""
        return os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config'
        )

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        finally:
            if conn:
                conn.close()

    def _load_initial_config(self):
        """Load initial configuration on service startup"""
        try:
            self._load_from_database()
            logger.info("Intelligence config service initialized from database")
        except Exception as e:
            logger.warning(f"Failed to load initial intelligence config from database: {e}")
            # Create default config
            self._cached_config = IntelligenceConfig()
            self._cached_config.source = 'default'
            self._cache_timestamp = datetime.now()
            logger.info("Using default intelligence config")

    def get_config(self, force_refresh: bool = False) -> IntelligenceConfig:
        """
        Get current configuration, refreshing from DB if needed.

        Args:
            force_refresh: Force reload from database

        Returns:
            IntelligenceConfig object with all parameters
        """
        with self._lock:
            if force_refresh or self._should_refresh():
                try:
                    self._load_from_database()
                except Exception as e:
                    logger.error(f"Failed to load intelligence config from database: {e}")
                    # Fall back to last-known-good
                    if self._last_known_good is not None:
                        logger.warning("Using last-known-good intelligence configuration")
                        self._cached_config = copy.deepcopy(self._last_known_good)
                        self._cached_config.source = 'cache'
                        self._cache_timestamp = datetime.now()

            if self._cached_config is None:
                # Return default config if nothing else available
                self._cached_config = IntelligenceConfig()
                self._cached_config.source = 'default'
                self._cache_timestamp = datetime.now()

            return self._cached_config

    def _should_refresh(self) -> bool:
        """Check if cache has expired"""
        if not self.enable_hot_reload:
            return self._cached_config is None

        if self._cache_timestamp is None:
            return True

        return datetime.now() - self._cache_timestamp > self.cache_ttl

    def _load_from_database(self):
        """Load configuration from database"""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Load global config parameters
                cur.execute("""
                    SELECT parameter_name, parameter_value, value_type
                    FROM intelligence_global_config
                    WHERE is_active = TRUE
                    ORDER BY category, display_order
                """)
                param_rows = cur.fetchall()

                # Load presets
                cur.execute("""
                    SELECT id, preset_name, threshold, use_intelligence_engine, description
                    FROM intelligence_presets
                    WHERE is_active = TRUE
                    ORDER BY display_order
                """)
                preset_rows = cur.fetchall()

                # Load preset components
                cur.execute("""
                    SELECT p.preset_name, pc.component_name, pc.is_enabled
                    FROM intelligence_preset_components pc
                    JOIN intelligence_presets p ON p.id = pc.preset_id
                    WHERE p.is_active = TRUE
                """)
                component_rows = cur.fetchall()

                # Load regime-strategy modifiers
                cur.execute("""
                    SELECT regime, strategy, confidence_modifier
                    FROM intelligence_regime_modifiers
                    WHERE is_active = TRUE
                """)
                modifier_rows = cur.fetchall()

            # Build config object
            config = self._build_config(param_rows, preset_rows, component_rows, modifier_rows)
            config.source = 'database'
            config.loaded_at = datetime.now()

            # Update cache
            self._cached_config = config
            self._cache_timestamp = datetime.now()

            # Update last-known-good
            self._last_known_good = copy.deepcopy(config)

            logger.info(f"Loaded intelligence config (preset={config.preset}) from database")

    def _build_config(
        self,
        param_rows: List[Dict],
        preset_rows: List[Dict],
        component_rows: List[Dict],
        modifier_rows: List[Dict]
    ) -> IntelligenceConfig:
        """Build IntelligenceConfig from database rows"""
        config = IntelligenceConfig()

        # Build parameter lookup
        params = {}
        for row in param_rows:
            name = row['parameter_name']
            value = row['parameter_value']
            value_type = row['value_type']

            # Convert value based on type
            if value_type == 'bool':
                params[name] = value.lower() in ('true', '1', 'yes')
            elif value_type == 'int':
                params[name] = int(value)
            elif value_type == 'float':
                params[name] = float(value)
            else:
                params[name] = value

        # Map parameters to config attributes
        param_mapping = {
            'intelligence_preset': 'preset',
            'intelligence_mode': 'mode',
            'enable_market_intelligence': 'enabled',
            'weight_market_regime': 'weight_market_regime',
            'weight_volatility': 'weight_volatility',
            'weight_volume': 'weight_volume',
            'weight_time': 'weight_time',
            'weight_confidence': 'weight_confidence',
            'enable_market_regime_filter': 'enable_market_regime_filter',
            'enable_volatility_filter': 'enable_volatility_filter',
            'enable_volume_filter': 'enable_volume_filter',
            'enable_time_filter': 'enable_time_filter',
            'enable_confidence_filter': 'enable_confidence_filter',
            'enable_spread_filter': 'enable_spread_filter',
            'enable_recent_signals_filter': 'enable_recent_signals_filter',
            'enable_smart_money_collection': 'enable_smart_money_collection',
            'enable_order_flow_collection': 'enable_order_flow_collection',
            'smart_money_structure_validation': 'smart_money_structure_validation',
            'smart_money_order_flow_validation': 'smart_money_order_flow_validation',
            'enhanced_regime_detection_enabled': 'enhanced_regime_detection_enabled',
            'adx_trending_threshold': 'adx_trending_threshold',
            'adx_strong_trend_threshold': 'adx_strong_trend_threshold',
            'adx_weak_trend_threshold': 'adx_weak_trend_threshold',
            'ema_alignment_weight': 'ema_alignment_weight',
            'adx_weight': 'adx_weight',
            'momentum_weight': 'momentum_weight',
            'trending_score_threshold': 'trending_score_threshold',
            'ranging_score_threshold': 'ranging_score_threshold',
            'breakout_score_threshold': 'breakout_score_threshold',
            'collect_enhanced_regime_data': 'collect_enhanced_regime_data',
            'log_regime_comparison': 'log_regime_comparison',
            'separate_volatility_from_structure': 'separate_volatility_from_structure',
            'volatility_as_modifier': 'volatility_as_modifier',
            'enable_probabilistic_confidence_modifiers': 'enable_probabilistic_confidence_modifiers',
            'min_confidence_modifier': 'min_confidence_modifier',
            'market_bias_filter_enabled': 'market_bias_filter_enabled',
            'market_bias_min_consensus': 'market_bias_min_consensus',
            'market_intelligence_min_confidence': 'market_intelligence_min_confidence',
            'block_unsuitable_regimes': 'block_unsuitable_regimes',
            'enable_intelligence_storage': 'enable_intelligence_storage',
            'intelligence_cleanup_days': 'intelligence_cleanup_days',
            'force_intelligence_analysis': 'force_intelligence_analysis',
            'intelligence_override_market_hours': 'intelligence_override_market_hours',
            'use_historical_data_for_intelligence': 'use_historical_data_for_intelligence',
            'intelligence_confidence_threshold': 'intelligence_confidence_threshold',
            'intelligence_volume_threshold': 'intelligence_volume_threshold',
            'intelligence_volatility_min': 'intelligence_volatility_min',
            'intelligence_allow_low_volatility': 'intelligence_allow_low_volatility',
            'intelligence_allow_ranging_markets': 'intelligence_allow_ranging_markets',
            'live_scanner_lookback_hours': 'live_scanner_lookback_hours',
            'enable_recent_historical_scan': 'enable_recent_historical_scan',
            'use_backtest_data_logic_for_live': 'use_backtest_data_logic_for_live',
            'force_market_open': 'force_market_open',
            'enable_weekend_scanning': 'enable_weekend_scanning',
            'max_data_age_minutes': 'max_data_age_minutes',
            'minimum_candles_for_live_scan': 'minimum_candles_for_live_scan',
            'intelligence_debug_mode': 'intelligence_debug_mode',
            'intelligence_log_rejections': 'intelligence_log_rejections',
            'claude_integrate_with_intelligence': 'claude_integrate_with_intelligence',
            'claude_use_intelligence_in_prompts': 'claude_use_intelligence_in_prompts',
            'claude_override_intelligence': 'claude_override_intelligence',
            'intelligence_claude_weight': 'intelligence_claude_weight',
        }

        for db_name, attr_name in param_mapping.items():
            if db_name in params:
                setattr(config, attr_name, params[db_name])

        # Build presets dictionary
        preset_components = {}
        for row in component_rows:
            preset_name = row['preset_name']
            if preset_name not in preset_components:
                preset_components[preset_name] = {}
            preset_components[preset_name][row['component_name']] = row['is_enabled']

        config._presets = {}
        for row in preset_rows:
            preset = IntelligencePreset(
                name=row['preset_name'],
                threshold=float(row['threshold']),
                use_intelligence_engine=row['use_intelligence_engine'],
                description=row['description'] or '',
                components_enabled=preset_components.get(row['preset_name'], {})
            )
            config._presets[row['preset_name']] = preset

        # Set threshold and use_engine from active preset
        if config.preset in config._presets:
            active_preset = config._presets[config.preset]
            config.threshold = active_preset.threshold
            config.use_intelligence_engine = active_preset.use_intelligence_engine

        # Build regime-strategy modifiers dictionary
        config._regime_modifiers = {}
        for row in modifier_rows:
            regime = row['regime']
            if regime not in config._regime_modifiers:
                config._regime_modifiers[regime] = {}
            config._regime_modifiers[regime][row['strategy']] = float(row['confidence_modifier'])

        return config

    def get_preset(self) -> str:
        """Get current active preset name"""
        return self.get_config().preset

    def get_threshold(self) -> float:
        """Get current threshold from active preset"""
        return self.get_config().threshold

    def is_engine_enabled(self) -> bool:
        """Check if intelligence engine should run"""
        config = self.get_config()
        return config.enabled and config.use_intelligence_engine

    def get_regime_modifier(self, regime: str, strategy: str) -> float:
        """Get confidence modifier for regime-strategy combination"""
        return self.get_config().get_regime_modifier(regime, strategy)

    def is_smart_money_collection_enabled(self) -> bool:
        """Check if Smart Money data collection is enabled"""
        config = self.get_config()
        return config.enable_smart_money_collection or config.enable_order_flow_collection

    def get_cleanup_days(self) -> int:
        """Get number of days to retain intelligence data"""
        return self.get_config().intelligence_cleanup_days

    def invalidate_cache(self):
        """Force cache invalidation"""
        with self._lock:
            self._cache_timestamp = None
            logger.info("Intelligence config cache invalidated")

    def update_parameter(self, parameter_name: str, new_value: Any) -> bool:
        """
        Update a single parameter in the database.

        Args:
            parameter_name: Name of the parameter to update
            new_value: New value for the parameter

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Determine value type
                    if isinstance(new_value, bool):
                        value_str = 'true' if new_value else 'false'
                    else:
                        value_str = str(new_value)

                    cur.execute("""
                        UPDATE intelligence_global_config
                        SET parameter_value = %s, updated_at = NOW()
                        WHERE parameter_name = %s
                    """, (value_str, parameter_name))

                    # Log audit
                    cur.execute("""
                        INSERT INTO intelligence_config_audit
                        (table_name, change_type, changed_by, new_values)
                        VALUES ('intelligence_global_config', 'UPDATE', 'service',
                                %s::jsonb)
                    """, (json.dumps({parameter_name: value_str}),))

                    conn.commit()

            # Invalidate cache to pick up changes
            self.invalidate_cache()
            logger.info(f"Updated intelligence parameter {parameter_name}={new_value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update intelligence parameter {parameter_name}: {e}")
            return False

    def set_preset(self, preset_name: str) -> bool:
        """
        Change the active intelligence preset.

        Args:
            preset_name: Name of the preset to activate

        Returns:
            True if successful, False otherwise
        """
        return self.update_parameter('intelligence_preset', preset_name)


# Global singleton instance
_service_instance: Optional[IntelligenceConfigService] = None
_service_lock = RLock()


def get_intelligence_config_service(
    database_url: str = None,
    cache_ttl_seconds: int = 300,
    enable_hot_reload: bool = True
) -> IntelligenceConfigService:
    """Get singleton instance of intelligence config service"""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = IntelligenceConfigService(
                database_url=database_url,
                cache_ttl_seconds=cache_ttl_seconds,
                enable_hot_reload=enable_hot_reload
            )
        return _service_instance


def get_intelligence_config() -> IntelligenceConfig:
    """Convenience function to get current intelligence config"""
    return get_intelligence_config_service().get_config()
