"""
Scanner Global Configuration Service

Database-driven configuration service for the Forex Scanner.
Provides configuration loading with in-memory caching and fail-fast behavior.

Key Design Decisions:
1. Database is the ONLY source of truth - no fallback to config.py
2. Fail-fast on startup - scanner won't start without database
3. In-memory cache for runtime - survives temporary DB outages
4. Source logging - every config access logs [DB] or [CACHE:Xmin]

Usage:
    from forex_scanner.services.scanner_config_service import get_scanner_config

    config = get_scanner_config()
    print(config.scan_interval)  # 120
    print(config.source)         # 'database' or 'cache'
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
from decimal import Decimal

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """
    Complete configuration object for Scanner settings.

    All fields are populated from database - no defaults here.
    The database migration provides initial values.
    """

    # VERSION
    version: str = ""

    # SCANNER CORE SETTINGS
    scan_interval: int = 0
    min_confidence: float = 0.0
    default_timeframe: str = ""
    use_1m_base_synthesis: bool = False
    scan_align_to_boundaries: bool = False
    scan_boundary_offset_seconds: int = 0
    spread_pips: float = 1.5
    use_bid_adjustment: bool = False
    epic_list: List[str] = field(default_factory=list)
    pair_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # DUPLICATE DETECTION SETTINGS
    enable_duplicate_check: bool = False
    duplicate_sensitivity: str = ""
    signal_cooldown_minutes: int = 0
    alert_cooldown_minutes: int = 0
    strategy_cooldown_minutes: int = 0
    global_cooldown_seconds: int = 0
    max_alerts_per_hour: int = 0
    max_alerts_per_epic_hour: int = 0
    price_similarity_threshold: float = 0.0
    confidence_similarity_threshold: float = 0.0
    deduplication_preset: str = ""
    use_database_dedup_check: bool = False
    database_dedup_window_minutes: int = 0
    enable_signal_hash_check: bool = False
    deduplication_debug_mode: bool = False
    enable_price_similarity_check: bool = False
    enable_strategy_cooldowns: bool = False
    deduplication_lookback_hours: int = 0

    # RISK MANAGEMENT SETTINGS
    position_size_percent: float = 0.0
    stop_loss_pips: int = 0
    take_profit_pips: int = 0
    max_open_positions: int = 0
    max_daily_trades: int = 0
    risk_per_trade_percent: float = 0.0
    min_position_size: float = 0.0
    max_position_size: float = 0.0
    default_position_size: float = 1.0
    max_risk_per_trade: int = 0
    default_risk_reward: float = 0.0
    default_stop_distance: int = 0
    # Extended risk management settings (for RiskManager - NO FALLBACK)
    max_daily_loss_percent: float = 5.0
    max_trades_per_pair: int = 3
    min_account_balance: float = 1000.0
    daily_profit_target_percent: float = 3.0
    stop_on_daily_target: bool = False
    testing_max_stop_percent: float = 20.0
    testing_min_confidence: float = 0.0
    emergency_stop_enabled: bool = True
    disable_account_risk_validation: bool = False
    disable_position_sizing: bool = False
    account_balance: float = 10000.0

    # TRADING HOURS SETTINGS
    trading_start_hour: int = 0
    trading_end_hour: int = 0
    respect_market_hours: bool = False
    weekend_scanning: bool = False
    enable_trading_time_controls: bool = False
    trading_cutoff_time_utc: int = 0
    trade_cooldown_enabled: bool = False
    trade_cooldown_minutes: int = 0
    user_timezone: str = ""
    respect_trading_hours: bool = False

    # SAFETY FILTER SETTINGS
    enable_critical_safety_filters: bool = False
    enable_ema200_contradiction_filter: bool = False
    enable_ema_stack_contradiction_filter: bool = False
    require_indicator_consensus: bool = False
    min_confirming_indicators: int = 0
    enable_emergency_circuit_breaker: bool = False
    max_contradictions_allowed: int = 0
    active_safety_preset: str = ""
    enable_large_candle_filter: bool = False
    large_candle_atr_multiplier: float = 0.0
    consecutive_large_candles_threshold: int = 0
    movement_lookback_periods: int = 0
    large_candle_filter_cooldown: int = 0
    ema200_minimum_margin: float = 0.0
    safety_filter_log_level: str = ""
    excessive_movement_threshold_pips: int = 0

    # ADX FILTER SETTINGS
    adx_filter_enabled: bool = False
    adx_filter_mode: str = ""
    adx_period: int = 0
    adx_grace_period_bars: int = 0
    adx_thresholds: Dict[str, float] = field(default_factory=dict)
    adx_pair_multipliers: Dict[str, float] = field(default_factory=dict)

    # PRESETS (JSONB from database)
    deduplication_presets: Dict[str, Dict] = field(default_factory=dict)
    safety_filter_presets: Dict[str, Dict] = field(default_factory=dict)
    large_candle_filter_presets: Dict[str, Dict] = field(default_factory=dict)

    # SMC CONFLICT FILTER SETTINGS
    smart_money_readonly_enabled: bool = False
    smart_money_analysis_timeout: float = 0.0
    smart_money_min_data_points: int = 50
    smart_money_structure_weight: float = 0.4
    smart_money_order_flow_weight: float = 0.3
    smart_money_min_confidence_boost: float = 0.1
    smart_money_max_confidence_boost: float = 0.3
    smart_money_liquidity_sweep_enabled: bool = True
    smart_money_liquidity_sweep_weight: float = 0.2
    smart_money_liquidity_sweep_lookback_bars: int = 10
    smart_money_min_sweep_quality: float = 0.4
    smc_conflict_filter_enabled: bool = False
    smc_min_directional_consensus: float = 0.0
    smc_reject_order_flow_conflict: bool = False
    smc_reject_ranging_structure: bool = False
    smc_min_structure_score: float = 0.0

    # CLAUDE TRADE VALIDATION SETTINGS
    require_claude_approval: bool = False
    claude_fail_secure: bool = False
    claude_model: str = ""
    min_claude_quality_score: int = 0
    claude_include_chart: bool = False
    claude_chart_timeframes: List[str] = field(default_factory=list)
    claude_vision_enabled: bool = False
    claude_vision_strategies: List[str] = field(default_factory=list)
    claude_vision_save_directory: str = "claude_analysis_enhanced/vision_analysis"
    claude_validate_in_backtest: bool = False
    save_claude_rejections: bool = False
    claude_save_vision_artifacts: bool = False
    claude_analysis_mode: str = "minimal"
    claude_timeout: int = 30
    claude_strategic_focus: Optional[str] = None

    # CLAUDE INTEGRATION SETTINGS (for IntegrationManager)
    claude_analysis_enabled: bool = False
    use_advanced_claude_prompts: bool = True
    claude_analysis_level: str = "institutional"
    claude_auto_save: bool = True
    claude_save_directory: str = "claude_analysis"
    minio_enabled: bool = True  # MinIO storage for charts/artifacts

    # MULTI-TIMEFRAME ANALYSIS SETTINGS
    enable_multi_timeframe_analysis: bool = False
    min_confluence_score: float = 0.30
    mtf_enhanced_min_confidence: float = 0.60  # Lower threshold for MTF-validated signals

    # ENABLED STRATEGIES (replaces config.py flags)
    enabled_strategies: List[str] = field(default_factory=list)

    # DATA FETCHER OPTIMIZATION SETTINGS
    enable_data_cache: bool = False
    reduced_lookback_hours: bool = True
    lazy_indicator_loading: bool = True
    data_batch_size: int = 10000
    enable_support_resistance: bool = True
    enable_volume_analysis: bool = True
    enable_behavior_analysis: bool = False

    # INDICATOR SETTINGS (MACD, KAMA, BB, etc.)
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    kama_period: int = 10
    kama_fast: int = 2
    kama_slow: int = 30
    bb_period: int = 20
    bb_std_dev: float = 2.0
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    zero_lag_length: int = 21
    zero_lag_band_mult: float = 1.5
    two_pole_filter_length: int = 20
    two_pole_sma_length: int = 25
    two_pole_signal_delay: int = 4

    # DATA QUALITY SETTINGS (NO FALLBACK - database only)
    lookback_reduction_factor: float = 0.7
    enable_data_quality_filtering: bool = False
    block_trading_on_data_issues: bool = True
    min_quality_score_for_trading: float = 0.5

    # TRADING CONTROL FLAGS (NO FALLBACK - database only)
    auto_trading_enabled: bool = False
    enable_order_execution: bool = False

    # ADVANCED DEDUPLICATION SETTINGS
    enable_alert_deduplication: bool = True
    signal_hash_cache_expiry_minutes: int = 15
    max_signal_hash_cache_size: int = 1000
    enable_signal_hash_check: bool = False
    enable_time_based_hash_components: bool = False
    use_database_dedup_check: bool = True
    database_dedup_window_minutes: int = 15
    duplicate_window_hours: int = 24

    # DATABASE STORAGE SETTINGS (NO FALLBACK - database only)
    save_to_database: bool = True

    # S/R VALIDATION SETTINGS (NO FALLBACK - database only)
    enable_sr_validation: bool = True
    enable_enhanced_sr_validation: bool = True
    sr_analysis_timeframe: str = "15m"
    sr_lookback_hours: int = 72
    sr_left_bars: int = 15
    sr_right_bars: int = 15
    sr_volume_threshold: float = 20.0
    sr_level_tolerance_pips: float = 2.0
    sr_min_level_distance_pips: float = 20.0
    sr_recent_flip_bars: int = 50
    sr_min_flip_strength: float = 0.6
    sr_cache_duration_minutes: int = 10
    min_bars_for_sr_analysis: int = 100

    # SIGNAL FRESHNESS SETTINGS (NO FALLBACK - database only)
    enable_signal_freshness_check: bool = True
    max_signal_age_minutes: int = 30

    # NEWS FILTERING SETTINGS (NO FALLBACK - database only)
    enable_news_filtering: bool = True
    reduce_confidence_near_news: bool = True
    news_filter_fail_secure: bool = False
    economic_calendar_url: str = "http://economic-calendar:8091"
    news_high_impact_buffer_minutes: int = 30
    news_medium_impact_buffer_minutes: int = 15
    news_lookahead_hours: int = 4
    block_trades_before_high_impact_news: bool = True
    block_trades_before_medium_impact_news: bool = False
    critical_economic_events: List[str] = field(default_factory=lambda: [
        "Non-Farm Employment Change", "NFP", "FOMC", "Federal Funds Rate",
        "ECB Press Conference", "Interest Rate Decision", "CPI", "Core CPI",
        "GDP", "Employment", "Unemployment"
    ])
    news_cache_duration_minutes: int = 5
    news_service_timeout_seconds: int = 5

    # MARKET INTELLIGENCE SETTINGS (NO FALLBACK - database only)
    enable_market_intelligence_capture: bool = True
    enable_market_intelligence_filtering: bool = False
    market_intelligence_min_confidence: float = 0.7
    market_intelligence_block_unsuitable_regimes: bool = True
    market_bias_filter_enabled: bool = True
    market_bias_min_consensus: float = 0.70

    # INTELLIGENCE MANAGER SETTINGS (NO FALLBACK - database only)
    intelligence_preset: str = "minimal"
    intelligence_debug_mode: bool = False

    # EPIC VALIDATION SETTINGS (NO FALLBACK - database only)
    allowed_trading_epics: List[str] = field(default_factory=list)
    blocked_trading_epics: List[str] = field(default_factory=list)

    # TESTING & VALIDATION SETTINGS (NO FALLBACK - database only)
    strategy_testing_mode: bool = False
    validate_spread: bool = True
    max_spread_pips: float = 3.0
    min_signal_confirmations: int = 0
    scalping_min_confidence: float = 0.45

    # NOTIFICATION SETTINGS (NO FALLBACK - database only)
    notifications_enabled: bool = True

    # MARKET MONITOR SETTINGS (NO FALLBACK - database only)
    low_volatility_threshold: float = 0.5
    normal_volatility_threshold: float = 1.0
    high_volatility_threshold: float = 2.0
    extreme_volatility_threshold: float = 3.0
    tight_spread_threshold: float = 2.0
    normal_spread_threshold: float = 3.0
    wide_spread_threshold: float = 5.0
    market_sessions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    market_condition_cache_minutes: int = 5

    # ORDER EXECUTOR SETTINGS (NO FALLBACK - database only)
    order_max_retries: int = 3
    order_retry_base_delay: float = 2.0
    order_retry_max_delay: float = 60.0
    order_connect_timeout: float = 10.0
    order_read_timeout: float = 45.0
    order_total_timeout: float = 60.0
    order_circuit_breaker_threshold: int = 5
    order_circuit_breaker_recovery: float = 300.0
    dynamic_stops_enabled: bool = False

    # METADATA (set by service, not from DB)
    source: str = ""  # 'database' or 'cache'
    loaded_at: Optional[datetime] = None
    cache_age_minutes: float = 0.0
    config_id: int = 0

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """
        Check if a strategy is enabled in the database configuration.

        Supports multiple naming conventions:
        - 'SMC_SIMPLE' or 'smc_simple'
        - 'MACD_EMA_STRATEGY' or 'MACD_EMA' or 'macd_ema'
        - 'EMA_STRATEGY' or 'EMA' or 'ema'

        Args:
            strategy_name: The strategy name to check

        Returns:
            True if the strategy is in the enabled_strategies list
        """
        if not self.enabled_strategies:
            return False

        # Normalize the strategy name for comparison
        normalized_name = strategy_name.upper().replace('_STRATEGY', '').replace('_ENABLED', '')

        # Check against each enabled strategy (also normalized)
        for enabled in self.enabled_strategies:
            enabled_normalized = enabled.upper().replace('_STRATEGY', '').replace('_ENABLED', '')
            if normalized_name == enabled_normalized:
                return True

        return False

    def get_enabled_strategies(self) -> List[str]:
        """Get the list of enabled strategies."""
        return self.enabled_strategies or []


class ScannerConfigService:
    """
    Database-driven configuration service with in-memory caching.

    Behavior:
    - On startup: MUST load from database or fail
    - During runtime: Use cache, refresh from DB every TTL seconds
    - On DB failure mid-run: Continue using cached config with warning
    - On restart without DB: Fail immediately
    """

    def __init__(
        self,
        database_url: str = None,
        cache_ttl_seconds: int = 120,
        enable_hot_reload: bool = True
    ):
        self.database_url = database_url or self._get_default_database_url()
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.enable_hot_reload = enable_hot_reload

        self._lock = RLock()
        self._cached_config: Optional[ScannerConfig] = None
        self._cache_timestamp: Optional[datetime] = None

        # Fail-fast: Must load from database on startup
        self._load_initial_config()

    def _get_default_database_url(self) -> str:
        """Get database URL from environment or use default."""
        return os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config'
        )

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _load_initial_config(self):
        """
        Load configuration from database on startup.
        FAILS if database is unavailable - this is intentional.
        """
        try:
            self._load_from_database()
            logger.info(
                f"[CONFIG:DB] Scanner config service initialized - "
                f"v{self._cached_config.version}, {self._count_settings()} settings loaded"
            )
        except Exception as e:
            logger.critical(
                f"[CONFIG:FAIL] Cannot connect to strategy_config database: {e}"
            )
            logger.critical(
                "[CONFIG:FAIL] Scanner startup aborted - database required for configuration"
            )
            raise RuntimeError(
                f"Scanner config service failed to initialize: {e}. "
                f"Database is required - no fallback available."
            ) from e

    def _count_settings(self) -> int:
        """Count non-metadata settings for logging."""
        if not self._cached_config:
            return 0
        # Count fields that are not metadata
        metadata_fields = {'source', 'loaded_at', 'cache_age_minutes', 'config_id'}
        return len([
            f for f in self._cached_config.__dataclass_fields__
            if f not in metadata_fields
        ])

    def get_config(self, force_refresh: bool = False) -> ScannerConfig:
        """
        Get current configuration.

        Returns cached config if within TTL, otherwise refreshes from database.
        If database is unavailable, returns cached config with warning.

        Args:
            force_refresh: Force database refresh regardless of TTL

        Returns:
            ScannerConfig object with source metadata
        """
        with self._lock:
            if force_refresh or self._should_refresh():
                try:
                    self._load_from_database()
                except Exception as e:
                    if self._cached_config is not None:
                        # DB failed but we have cache - continue with warning
                        cache_age = self._get_cache_age_minutes()
                        logger.warning(
                            f"[CONFIG:CACHE:{cache_age:.0f}min] Database refresh failed: {e}"
                        )
                        # Update cache metadata
                        self._cached_config.source = 'cache'
                        self._cached_config.cache_age_minutes = cache_age
                    else:
                        # No cache and DB failed - this shouldn't happen after init
                        raise RuntimeError(
                            f"No cached config available and database failed: {e}"
                        )

            return self._cached_config

    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed based on TTL."""
        if not self.enable_hot_reload:
            return self._cached_config is None
        if self._cache_timestamp is None:
            return True
        return datetime.now() - self._cache_timestamp > self.cache_ttl

    def _get_cache_age_minutes(self) -> float:
        """Get age of cached config in minutes."""
        if self._cache_timestamp is None:
            return 0.0
        delta = datetime.now() - self._cache_timestamp
        return delta.total_seconds() / 60.0

    def _load_from_database(self):
        """Load configuration from database."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM scanner_global_config
                    WHERE is_active = TRUE
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()

                if row is None:
                    raise ValueError(
                        "No active configuration found in scanner_global_config table"
                    )

                config = self._build_config_from_row(dict(row))
                config.source = 'database'
                config.loaded_at = datetime.now()
                config.cache_age_minutes = 0.0
                config.config_id = row['id']

                # Check if config changed
                if self._cached_config and self._config_changed(config):
                    logger.info(
                        f"[CONFIG:DB] Configuration refreshed - changes detected"
                    )

                self._cached_config = config
                self._cache_timestamp = datetime.now()

    def _config_changed(self, new_config: ScannerConfig) -> bool:
        """Check if configuration has changed from cached version."""
        if not self._cached_config:
            return True

        # Compare key fields
        key_fields = [
            'scan_interval', 'min_confidence', 'deduplication_preset',
            'active_safety_preset', 'adx_filter_enabled', 'max_open_positions'
        ]

        for field_name in key_fields:
            old_val = getattr(self._cached_config, field_name, None)
            new_val = getattr(new_config, field_name, None)
            if old_val != new_val:
                return True

        return False

    def _build_config_from_row(self, row: Dict[str, Any]) -> ScannerConfig:
        """Build ScannerConfig from database row."""
        config = ScannerConfig()

        # Direct field mappings
        direct_fields = [
            'version', 'scan_interval', 'min_confidence', 'default_timeframe',
            'use_1m_base_synthesis', 'scan_align_to_boundaries', 'scan_boundary_offset_seconds',
            'spread_pips', 'use_bid_adjustment',
            'enable_duplicate_check', 'duplicate_sensitivity', 'signal_cooldown_minutes',
            'alert_cooldown_minutes', 'strategy_cooldown_minutes', 'global_cooldown_seconds',
            'max_alerts_per_hour', 'max_alerts_per_epic_hour', 'price_similarity_threshold',
            'confidence_similarity_threshold', 'deduplication_preset', 'use_database_dedup_check',
            'database_dedup_window_minutes', 'enable_signal_hash_check', 'deduplication_debug_mode',
            'enable_price_similarity_check', 'enable_strategy_cooldowns', 'deduplication_lookback_hours',
            'position_size_percent', 'stop_loss_pips', 'take_profit_pips', 'max_open_positions',
            'max_daily_trades', 'risk_per_trade_percent', 'min_position_size', 'max_position_size',
            'default_position_size', 'max_risk_per_trade', 'default_risk_reward', 'default_stop_distance',
            # Extended Risk Management Settings (for RiskManager)
            'max_daily_loss_percent', 'max_trades_per_pair', 'min_account_balance',
            'daily_profit_target_percent', 'stop_on_daily_target', 'testing_max_stop_percent',
            'testing_min_confidence', 'emergency_stop_enabled', 'disable_account_risk_validation',
            'disable_position_sizing', 'account_balance',
            'trading_start_hour', 'trading_end_hour', 'respect_market_hours', 'weekend_scanning',
            'enable_trading_time_controls', 'trading_cutoff_time_utc', 'trade_cooldown_enabled',
            'trade_cooldown_minutes', 'user_timezone', 'respect_trading_hours',
            'enable_critical_safety_filters', 'enable_ema200_contradiction_filter',
            'enable_ema_stack_contradiction_filter', 'require_indicator_consensus',
            'min_confirming_indicators', 'enable_emergency_circuit_breaker',
            'max_contradictions_allowed', 'active_safety_preset', 'enable_large_candle_filter',
            'large_candle_atr_multiplier', 'consecutive_large_candles_threshold',
            'movement_lookback_periods', 'large_candle_filter_cooldown', 'ema200_minimum_margin',
            'safety_filter_log_level', 'excessive_movement_threshold_pips',
            'adx_filter_enabled', 'adx_filter_mode', 'adx_period', 'adx_grace_period_bars',
            'smart_money_readonly_enabled', 'smart_money_analysis_timeout',
            'smart_money_min_data_points', 'smart_money_structure_weight',
            'smart_money_order_flow_weight', 'smart_money_min_confidence_boost',
            'smart_money_max_confidence_boost', 'smart_money_liquidity_sweep_enabled',
            'smart_money_liquidity_sweep_weight', 'smart_money_liquidity_sweep_lookback_bars',
            'smart_money_min_sweep_quality',
            'smc_conflict_filter_enabled', 'smc_min_directional_consensus',
            'smc_reject_order_flow_conflict', 'smc_reject_ranging_structure',
            'smc_min_structure_score',
            'require_claude_approval', 'claude_fail_secure', 'claude_model',
            'min_claude_quality_score', 'claude_include_chart', 'claude_vision_enabled',
            'claude_vision_save_directory', 'claude_validate_in_backtest', 'save_claude_rejections',
            'claude_save_vision_artifacts', 'claude_analysis_mode', 'claude_timeout',
            'claude_strategic_focus',
            # Claude Integration Settings (for IntegrationManager)
            'claude_analysis_enabled', 'use_advanced_claude_prompts', 'claude_analysis_level',
            'claude_auto_save', 'claude_save_directory', 'minio_enabled',
            'enable_multi_timeframe_analysis', 'min_confluence_score', 'mtf_enhanced_min_confidence',
            'enable_alert_deduplication', 'signal_hash_cache_expiry_minutes',
            'max_signal_hash_cache_size', 'enable_signal_hash_check',
            'enable_time_based_hash_components', 'use_database_dedup_check',
            'database_dedup_window_minutes', 'duplicate_window_hours', 'save_to_database',
            # Data Quality Settings
            'lookback_reduction_factor', 'enable_data_quality_filtering',
            'block_trading_on_data_issues', 'min_quality_score_for_trading',
            # Trading Control Flags
            'auto_trading_enabled', 'enable_order_execution',
            # S/R Validation Settings
            'enable_sr_validation', 'enable_enhanced_sr_validation', 'sr_analysis_timeframe',
            'sr_lookback_hours', 'sr_left_bars', 'sr_right_bars', 'sr_volume_threshold',
            'sr_level_tolerance_pips', 'sr_min_level_distance_pips', 'sr_recent_flip_bars',
            'sr_min_flip_strength', 'sr_cache_duration_minutes', 'min_bars_for_sr_analysis',
            # Signal Freshness Settings
            'enable_signal_freshness_check', 'max_signal_age_minutes',
            # News Filtering Settings
            'enable_news_filtering', 'reduce_confidence_near_news', 'news_filter_fail_secure',
            'economic_calendar_url', 'news_high_impact_buffer_minutes',
            'news_medium_impact_buffer_minutes', 'news_lookahead_hours',
            'block_trades_before_high_impact_news', 'block_trades_before_medium_impact_news',
            'news_cache_duration_minutes', 'news_service_timeout_seconds',
            # Market Intelligence Settings
            'enable_market_intelligence_capture', 'enable_market_intelligence_filtering',
            'market_intelligence_min_confidence', 'market_intelligence_block_unsuitable_regimes',
            'market_bias_filter_enabled', 'market_bias_min_consensus',
            # Intelligence Manager Settings
            'intelligence_preset', 'intelligence_debug_mode',
            # Epic Validation Settings
            'allowed_trading_epics', 'blocked_trading_epics',
            # Testing & Validation Settings
            'strategy_testing_mode', 'validate_spread', 'max_spread_pips',
            'min_signal_confirmations', 'scalping_min_confidence',
            # Notification Settings
            'notifications_enabled',
            # Market Monitor Settings
            'low_volatility_threshold', 'normal_volatility_threshold',
            'high_volatility_threshold', 'extreme_volatility_threshold',
            'tight_spread_threshold', 'normal_spread_threshold', 'wide_spread_threshold',
            'market_condition_cache_minutes',
            # Order Executor Settings
            'order_max_retries', 'order_retry_base_delay', 'order_retry_max_delay',
            'order_connect_timeout', 'order_read_timeout', 'order_total_timeout',
            'order_circuit_breaker_threshold', 'order_circuit_breaker_recovery',
            'dynamic_stops_enabled',
        ]

        # Fields that should be integers
        int_fields = {
            'scan_interval', 'scan_boundary_offset_seconds', 'signal_cooldown_minutes',
            'alert_cooldown_minutes', 'strategy_cooldown_minutes', 'global_cooldown_seconds',
            'max_alerts_per_hour', 'max_alerts_per_epic_hour', 'database_dedup_window_minutes',
            'deduplication_lookback_hours', 'stop_loss_pips', 'take_profit_pips',
            'max_open_positions', 'max_daily_trades', 'max_risk_per_trade', 'default_stop_distance',
            'trading_start_hour', 'trading_end_hour', 'trading_cutoff_time_utc',
            'trade_cooldown_minutes', 'min_confirming_indicators', 'max_contradictions_allowed',
            'consecutive_large_candles_threshold', 'movement_lookback_periods',
            'large_candle_filter_cooldown', 'excessive_movement_threshold_pips',
            'adx_period', 'adx_grace_period_bars', 'min_claude_quality_score',
            'signal_hash_cache_expiry_minutes', 'max_signal_hash_cache_size',
            # New integer fields
            'sr_lookback_hours', 'sr_left_bars', 'sr_right_bars', 'sr_recent_flip_bars',
            'sr_cache_duration_minutes', 'min_bars_for_sr_analysis', 'max_signal_age_minutes',
            'min_signal_confirmations', 'claude_timeout', 'duplicate_window_hours',
            # News Filtering integer fields
            'news_high_impact_buffer_minutes', 'news_medium_impact_buffer_minutes',
            'news_lookahead_hours', 'news_cache_duration_minutes', 'news_service_timeout_seconds',
            # Market Monitor integer fields
            'market_condition_cache_minutes',
            # Smart Money integer fields
            'smart_money_min_data_points', 'smart_money_liquidity_sweep_lookback_bars',
            # Order Executor integer fields
            'order_max_retries', 'order_circuit_breaker_threshold',
        }

        for field_name in direct_fields:
            if field_name in row and row[field_name] is not None:
                value = row[field_name]

                # Convert types as needed
                if field_name in int_fields:
                    value = int(value)
                elif isinstance(value, Decimal):
                    value = float(value)

                setattr(config, field_name, value)

        # Handle JSONB fields (dicts)
        jsonb_dict_fields = [
            'adx_thresholds', 'adx_pair_multipliers',
            'deduplication_presets', 'safety_filter_presets', 'large_candle_filter_presets',
            'pair_info',
            # Market Monitor JSONB fields
            'market_sessions',
        ]

        for field_name in jsonb_dict_fields:
            if field_name in row and row[field_name] is not None:
                value = row[field_name]
                if isinstance(value, str):
                    value = json.loads(value)
                setattr(config, field_name, value)

        # Handle JSONB fields (lists)
        jsonb_list_fields = [
            'claude_chart_timeframes', 'claude_vision_strategies', 'enabled_strategies',
            'epic_list', 'critical_economic_events'
        ]

        for field_name in jsonb_list_fields:
            if field_name in row and row[field_name] is not None:
                value = row[field_name]
                if isinstance(value, str):
                    value = json.loads(value)
                setattr(config, field_name, value)

        return config

    def invalidate_cache(self):
        """Force cache invalidation - next get_config() will reload from DB."""
        with self._lock:
            self._cache_timestamp = None
            logger.info("[CONFIG:DB] Cache invalidated - will reload on next access")

    def get_setting(self, name: str, default: Any = None) -> Any:
        """
        Get a single setting by name.

        Args:
            name: Setting name (case-insensitive, underscores allowed)
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        config = self.get_config()
        # Normalize name to lowercase with underscores
        normalized = name.lower().replace('-', '_')
        return getattr(config, normalized, default)


# =============================================================================
# SINGLETON INSTANCE MANAGEMENT
# =============================================================================

_service_instance: Optional[ScannerConfigService] = None
_service_lock = RLock()


def get_scanner_config_service(
    database_url: str = None,
    cache_ttl_seconds: int = 120,
    enable_hot_reload: bool = True
) -> ScannerConfigService:
    """
    Get singleton instance of scanner config service.

    Args:
        database_url: Optional database URL (uses env var if not provided)
        cache_ttl_seconds: Cache TTL in seconds (default 120)
        enable_hot_reload: Enable automatic cache refresh (default True)

    Returns:
        ScannerConfigService instance
    """
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = ScannerConfigService(
                database_url=database_url,
                cache_ttl_seconds=cache_ttl_seconds,
                enable_hot_reload=enable_hot_reload
            )
        return _service_instance


def get_scanner_config(force_refresh: bool = False) -> ScannerConfig:
    """
    Convenience function to get current scanner configuration.

    Args:
        force_refresh: Force database refresh regardless of TTL

    Returns:
        ScannerConfig object
    """
    return get_scanner_config_service().get_config(force_refresh=force_refresh)


def get_scanner_setting(name: str, default: Any = None) -> Any:
    """
    Convenience function to get a single scanner setting.

    Args:
        name: Setting name
        default: Default value if not found

    Returns:
        Setting value
    """
    return get_scanner_config_service().get_setting(name, default)


# =============================================================================
# HELPER FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================

def is_config_from_database() -> bool:
    """Check if current config is from database (vs cache)."""
    try:
        config = get_scanner_config()
        return config.source == 'database'
    except Exception:
        return False


def get_config_source_info() -> Dict[str, Any]:
    """Get detailed information about config source."""
    try:
        config = get_scanner_config()
        return {
            'source': config.source,
            'loaded_at': config.loaded_at.isoformat() if config.loaded_at else None,
            'cache_age_minutes': config.cache_age_minutes,
            'config_id': config.config_id,
            'version': config.version,
        }
    except Exception as e:
        return {
            'source': 'error',
            'error': str(e),
        }
