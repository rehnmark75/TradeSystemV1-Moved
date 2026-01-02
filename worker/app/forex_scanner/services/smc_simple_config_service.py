"""
SMC Simple Strategy Configuration Service

Provides database-driven configuration loading with:
- In-memory caching with configurable TTL
- Last-known-good fallback when DB unavailable
- Per-pair parameter resolution (global + overrides)
- Hot reload support via configurable refresh intervals
"""

import os
import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, time as dt_time
from threading import RLock
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


@dataclass
class SMCSimpleConfig:
    """Complete configuration object for SMC Simple strategy"""

    # STRATEGY METADATA
    version: str = "2.11.0"
    strategy_name: str = "SMC_SIMPLE"
    strategy_status: str = ""

    # TIER 1: 4H DIRECTIONAL BIAS
    htf_timeframe: str = "4h"
    ema_period: int = 50
    ema_buffer_pips: float = 2.5
    require_close_beyond_ema: bool = True
    min_distance_from_ema_pips: float = 3.0

    # TIER 2: 15M ENTRY TRIGGER
    trigger_timeframe: str = "15m"
    swing_lookback_bars: int = 20
    swing_strength_bars: int = 2
    require_body_close_break: bool = False
    wick_tolerance_pips: float = 3.0
    volume_confirmation_enabled: bool = True
    volume_sma_period: int = 20
    volume_spike_multiplier: float = 1.2

    # DYNAMIC SWING LOOKBACK
    use_dynamic_swing_lookback: bool = True
    swing_lookback_atr_low: int = 8
    swing_lookback_atr_high: int = 15
    swing_lookback_min: int = 15
    swing_lookback_max: int = 30

    # TIER 3: 5M EXECUTION
    entry_timeframe: str = "5m"
    pullback_enabled: bool = True
    fib_pullback_min: float = 0.236
    fib_pullback_max: float = 0.700
    fib_optimal_zone_min: float = 0.382
    fib_optimal_zone_max: float = 0.618
    max_pullback_wait_bars: int = 12
    pullback_confirmation_bars: int = 2

    # MOMENTUM MODE
    momentum_mode_enabled: bool = True
    momentum_min_depth: float = -0.50
    momentum_max_depth: float = 0.0
    momentum_confidence_penalty: float = 0.05

    # ATR-BASED SWING VALIDATION
    use_atr_swing_validation: bool = True
    atr_period: int = 14
    min_swing_atr_multiplier: float = 0.25
    fallback_min_swing_pips: float = 5.0

    # MOMENTUM QUALITY FILTER
    momentum_quality_enabled: bool = True
    min_breakout_atr_ratio: float = 0.5
    min_body_percentage: float = 0.20

    # LIMIT ORDER CONFIGURATION
    limit_order_enabled: bool = True
    limit_expiry_minutes: int = 45
    pullback_offset_atr_factor: float = 0.2
    pullback_offset_min_pips: float = 2.0
    pullback_offset_max_pips: float = 3.0
    momentum_offset_pips: float = 3.0
    min_risk_after_offset_pips: float = 5.0
    max_sl_atr_multiplier: float = 3.0
    max_sl_absolute_pips: float = 30.0
    max_risk_after_offset_pips: float = 55.0

    # RISK MANAGEMENT
    min_rr_ratio: float = 1.5
    optimal_rr_ratio: float = 2.5
    max_rr_ratio: float = 5.0
    sl_buffer_pips: int = 6
    sl_atr_multiplier: float = 1.0
    use_atr_stop: bool = True
    min_tp_pips: int = 8
    use_swing_target: bool = True
    tp_structure_lookback: int = 50
    risk_per_trade_pct: float = 1.0

    # FIXED SL/TP OVERRIDE (per-pair configurable)
    fixed_sl_tp_override_enabled: bool = True
    fixed_stop_loss_pips: float = 9.0
    fixed_take_profit_pips: float = 15.0

    # SESSION FILTER
    session_filter_enabled: bool = True
    london_session_start: dt_time = field(default_factory=lambda: dt_time(7, 0))
    london_session_end: dt_time = field(default_factory=lambda: dt_time(16, 0))
    ny_session_start: dt_time = field(default_factory=lambda: dt_time(12, 0))
    ny_session_end: dt_time = field(default_factory=lambda: dt_time(21, 0))
    allowed_sessions: List[str] = field(default_factory=lambda: ['london', 'new_york', 'overlap'])
    block_asian_session: bool = True

    # SIGNAL LIMITS
    max_concurrent_signals: int = 3
    signal_cooldown_hours: int = 3

    # ADAPTIVE COOLDOWN
    adaptive_cooldown_enabled: bool = True
    base_cooldown_hours: float = 2.0
    cooldown_after_win_multiplier: float = 0.5
    cooldown_after_loss_multiplier: float = 1.5
    consecutive_loss_penalty_hours: float = 1.0
    max_consecutive_losses_before_block: int = 3
    consecutive_loss_block_hours: float = 8.0
    win_rate_lookback_trades: int = 20
    high_win_rate_threshold: float = 0.60
    low_win_rate_threshold: float = 0.40
    critical_win_rate_threshold: float = 0.30
    high_win_rate_cooldown_reduction: float = 0.25
    low_win_rate_cooldown_increase: float = 0.50
    high_volatility_atr_multiplier: float = 1.5
    volatility_cooldown_adjustment: float = 0.30
    strong_trend_cooldown_reduction: float = 0.30
    session_change_reset_cooldown: bool = True
    min_cooldown_hours: float = 1.0
    max_cooldown_hours: float = 12.0

    # CONFIDENCE SCORING
    min_confidence_threshold: float = 0.48
    max_confidence_threshold: float = 0.75
    high_confidence_threshold: float = 0.75
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "ema_alignment": 0.20,
        "swing_break_quality": 0.20,
        "volume_strength": 0.20,
        "pullback_quality": 0.20,
        "rr_ratio": 0.20,
    })

    # VOLUME FILTER
    min_volume_ratio: float = 0.50
    volume_filter_enabled: bool = True
    allow_no_volume_data: bool = True

    # DYNAMIC CONFIDENCE THRESHOLDS
    volume_adjusted_confidence_enabled: bool = True
    high_volume_threshold: float = 0.70
    atr_adjusted_confidence_enabled: bool = True
    low_atr_threshold: float = 0.0004
    high_atr_threshold: float = 0.0008
    ema_distance_adjusted_confidence_enabled: bool = True
    near_ema_threshold_pips: float = 20.0
    far_ema_threshold_pips: float = 30.0

    # MACD ALIGNMENT FILTER
    macd_alignment_filter_enabled: bool = True
    macd_alignment_mode: str = 'momentum'
    macd_min_strength: float = 0.0

    # LOGGING & DEBUG
    enable_debug_logging: bool = True
    log_rejected_signals: bool = True
    log_swing_detection: bool = False
    log_ema_checks: bool = False

    # REJECTION TRACKING
    rejection_tracking_enabled: bool = True
    rejection_batch_size: int = 50
    rejection_log_to_console: bool = False
    rejection_retention_days: int = 90

    # BACKTEST SETTINGS
    backtest_spread_pips: float = 1.5
    backtest_slippage_pips: float = 0.5

    # ENABLED PAIRS
    enabled_pairs: List[str] = field(default_factory=lambda: [
        'CS.D.EURUSD.CEEM.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP',
        'CS.D.AUDUSD.MINI.IP',
        'CS.D.USDCAD.MINI.IP',
        'CS.D.NZDUSD.MINI.IP',
        'CS.D.EURJPY.MINI.IP',
        'CS.D.AUDJPY.MINI.IP',
    ])

    # PAIR PIP VALUES
    pair_pip_values: Dict[str, float] = field(default_factory=lambda: {
        'CS.D.EURUSD.CEEM.IP': 1.0,
        'CS.D.GBPUSD.MINI.IP': 0.0001,
        'CS.D.USDJPY.MINI.IP': 0.01,
        'CS.D.USDCHF.MINI.IP': 0.0001,
        'CS.D.AUDUSD.MINI.IP': 0.0001,
        'CS.D.USDCAD.MINI.IP': 0.0001,
        'CS.D.NZDUSD.MINI.IP': 0.0001,
        'CS.D.EURJPY.MINI.IP': 0.01,
        'CS.D.GBPJPY.MINI.IP': 0.01,
        'CS.D.AUDJPY.MINI.IP': 0.01,
    })

    # Per-pair overrides cache (populated from database)
    _pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    loaded_at: datetime = field(default_factory=datetime.now)
    source: str = "database"  # 'database', 'cache', 'default'
    config_id: int = 0

    def get_for_pair(self, epic: str, param_name: str, default: Any = None) -> Any:
        """Get parameter value with pair-specific override if exists"""
        # Check pair overrides first
        if epic in self._pair_overrides:
            override_data = self._pair_overrides[epic]

            # Check direct field override
            if param_name in override_data and override_data[param_name] is not None:
                return override_data[param_name]

            # Check parameter_overrides JSONB
            param_overrides = override_data.get('parameter_overrides', {})
            if param_name in param_overrides:
                return param_overrides[param_name]

        # Fall back to global value
        if hasattr(self, param_name):
            return getattr(self, param_name)
        return default

    def get_pip_value(self, epic: str) -> float:
        """Get pip value for a given epic"""
        return self.pair_pip_values.get(epic, 0.0001)

    def is_pair_enabled(self, epic: str) -> bool:
        """Check if pair is enabled"""
        return epic in self.enabled_pairs

    def get_optimal_pullback_zone(self) -> tuple:
        """Get optimal Fibonacci pullback zone"""
        return (self.fib_optimal_zone_min, self.fib_optimal_zone_max)

    def is_session_allowed(self, hour_utc: int) -> bool:
        """Check if current hour is in allowed trading session"""
        if not self.session_filter_enabled:
            return True

        if self.block_asian_session and (hour_utc >= 21 or hour_utc < 7):
            return False

        return 7 <= hour_utc <= 21

    def is_asian_session_allowed(self, epic: str) -> bool:
        """Check if Asian session trading is allowed for a specific pair"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('allow_asian_session') is not None:
                return override['allow_asian_session']
        return not self.block_asian_session

    def is_macd_filter_enabled(self, epic: str) -> bool:
        """Check if MACD filter is enabled for a specific pair"""
        if not self.macd_alignment_filter_enabled:
            return False
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('macd_filter_enabled') is not None:
                return override['macd_filter_enabled']
        return True

    def get_pair_sl_buffer(self, epic: str) -> int:
        """Get SL buffer for a specific pair"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('sl_buffer_pips') is not None:
                return override['sl_buffer_pips']
        return self.sl_buffer_pips

    def get_pair_min_confidence(self, epic: str) -> float:
        """Get minimum confidence threshold for a specific pair"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('min_confidence') is not None:
                return override['min_confidence']
        return self.min_confidence_threshold

    def get_pair_max_confidence(self, epic: str) -> float:
        """Get maximum confidence cap for a specific pair (confidence paradox filter)"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('max_confidence') is not None:
                return override['max_confidence']
        return self.max_confidence_threshold

    def get_pair_min_volume_ratio(self, epic: str) -> float:
        """Get minimum volume ratio for a specific pair"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('min_volume_ratio') is not None:
                return override['min_volume_ratio']
        return self.min_volume_ratio

    def get_pair_fixed_stop_loss(self, epic: str) -> float:
        """Get fixed stop loss in pips for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        Returns None if fixed_sl_tp_override_enabled is False.
        """
        if not self.fixed_sl_tp_override_enabled:
            return None
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('fixed_stop_loss_pips') is not None:
                return float(override['fixed_stop_loss_pips'])
        return self.fixed_stop_loss_pips

    def get_pair_fixed_take_profit(self, epic: str) -> float:
        """Get fixed take profit in pips for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        Returns None if fixed_sl_tp_override_enabled is False.
        """
        if not self.fixed_sl_tp_override_enabled:
            return None
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('fixed_take_profit_pips') is not None:
                return float(override['fixed_take_profit_pips'])
        return self.fixed_take_profit_pips

    def get_dynamic_confidence(
        self,
        epic: str,
        volume_ratio: Optional[float] = None,
        atr_value: Optional[float] = None,
        ema_distance_pips: Optional[float] = None
    ) -> float:
        """
        Get dynamic confidence threshold based on market conditions.
        Returns the effective confidence threshold after applying all adjustments.
        """
        base_confidence = self.get_pair_min_confidence(epic)
        override = self._pair_overrides.get(epic, {})

        # Volume adjustment
        if (self.volume_adjusted_confidence_enabled and
                volume_ratio is not None and
                volume_ratio >= self.high_volume_threshold):
            high_vol_conf = override.get('high_volume_confidence')
            if high_vol_conf is not None:
                base_confidence = min(base_confidence, high_vol_conf)

        # ATR adjustment (low ATR = calmer market = lower threshold)
        if self.atr_adjusted_confidence_enabled and atr_value is not None:
            if atr_value < self.low_atr_threshold:
                low_atr_conf = override.get('low_atr_confidence')
                if low_atr_conf is not None:
                    base_confidence = min(base_confidence, low_atr_conf)
            elif atr_value >= self.high_atr_threshold:
                high_atr_conf = override.get('high_atr_confidence')
                if high_atr_conf is not None:
                    base_confidence = max(base_confidence, high_atr_conf)

        # EMA distance adjustment
        if self.ema_distance_adjusted_confidence_enabled and ema_distance_pips is not None:
            if ema_distance_pips < self.near_ema_threshold_pips:
                near_ema_conf = override.get('near_ema_confidence')
                if near_ema_conf is not None:
                    base_confidence = min(base_confidence, near_ema_conf)
            elif ema_distance_pips >= self.far_ema_threshold_pips:
                far_ema_conf = override.get('far_ema_confidence')
                if far_ema_conf is not None:
                    base_confidence = max(base_confidence, far_ema_conf)

        return base_confidence

    def should_block_signal(self, epic: str, signal_data: dict) -> tuple:
        """
        Check if a signal should be blocked based on pair-specific conditions.

        Returns:
            Tuple of (should_block: bool, reason: str)
        """
        if epic not in self._pair_overrides:
            return False, ""

        override = self._pair_overrides[epic]
        blocking_config = override.get('blocking_conditions')
        if not blocking_config or not blocking_config.get('enabled', False):
            return False, ""

        conditions = blocking_config.get('conditions', {})
        blocking_logic = blocking_config.get('blocking_logic', 'any')
        block_reasons = []

        # Check EMA distance
        max_ema = conditions.get('max_ema_distance_pips')
        if max_ema is not None:
            ema_distance = signal_data.get('ema_distance_pips', 0)
            if ema_distance > max_ema:
                block_reasons.append(f"EMA distance {ema_distance:.1f} > {max_ema} pips")

        # Check volume confirmation
        if conditions.get('require_volume_confirmation', False):
            volume_confirmed = signal_data.get('volume_confirmed', False)
            if not volume_confirmed:
                block_reasons.append("No volume confirmation (required for this pair)")

        # Check momentum without volume
        if conditions.get('block_momentum_without_volume', False):
            pullback_depth = signal_data.get('pullback_depth', 0)
            volume_confirmed = signal_data.get('volume_confirmed', False)
            if pullback_depth < 0 and not volume_confirmed:
                block_reasons.append(f"Momentum entry without volume")

        # Check confidence override
        min_conf = conditions.get('min_confidence_override')
        if min_conf is not None:
            confidence = signal_data.get('confidence_score', 0)
            if confidence < min_conf:
                block_reasons.append(f"Confidence {confidence:.1%} < {min_conf:.1%}")

        # Determine if signal should be blocked
        if blocking_logic == 'any' and block_reasons:
            return True, f"Blocked: {'; '.join(block_reasons)}"
        elif blocking_logic == 'all' and len(block_reasons) == len([c for c in conditions.values() if c]):
            return True, f"Blocked (all conditions): {'; '.join(block_reasons)}"

        return False, ""

    def check_macd_alignment(
        self,
        signal_type: str,
        macd_line: float,
        macd_signal: float,
        macd_histogram: float = None
    ) -> tuple:
        """
        Check if MACD momentum aligns with trade direction.

        Returns:
            Tuple of (is_aligned: bool, reason: str)
        """
        if self.macd_alignment_mode == 'histogram':
            if macd_histogram is None:
                return True, "No histogram data"
            if signal_type == 'BULL':
                aligned = macd_histogram > 0
                direction = 'bullish' if macd_histogram > 0 else 'bearish'
            else:
                aligned = macd_histogram < 0
                direction = 'bearish' if macd_histogram < 0 else 'bullish'
            reason = f"MACD histogram {direction} ({macd_histogram:.6f})"
        else:
            # Momentum mode (default)
            if macd_line is None or macd_signal is None:
                return True, "No MACD data"

            macd_diff = macd_line - macd_signal
            min_strength = self.macd_min_strength

            if signal_type == 'BULL':
                aligned = macd_diff > min_strength
                direction = "bullish" if macd_diff > 0 else "bearish"
            else:
                aligned = macd_diff < -min_strength
                direction = "bearish" if macd_diff < 0 else "bullish"

            reason = f"MACD momentum {direction} (diff={macd_diff:.6f})"

        return aligned, reason


class SMCSimpleConfigService:
    """
    Database-driven configuration service with in-memory caching.

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
        cache_ttl_seconds: int = 120,
        enable_hot_reload: bool = True
    ):
        self.database_url = database_url or self._get_default_database_url()
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.enable_hot_reload = enable_hot_reload

        # Thread-safe cache
        self._lock = RLock()
        self._cached_config: Optional[SMCSimpleConfig] = None
        self._cache_timestamp: Optional[datetime] = None
        self._last_known_good: Optional[SMCSimpleConfig] = None

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
            logger.info("SMC Simple config service initialized from database")
        except Exception as e:
            logger.warning(f"Failed to load initial config from database: {e}")
            # Create default config
            self._cached_config = SMCSimpleConfig()
            self._cached_config.source = 'default'
            self._cache_timestamp = datetime.now()
            logger.info("Using default SMC Simple config")

    def get_config(self, force_refresh: bool = False) -> SMCSimpleConfig:
        """
        Get current configuration, refreshing from DB if needed.

        Args:
            force_refresh: Force reload from database

        Returns:
            SMCSimpleConfig object with all parameters
        """
        with self._lock:
            if force_refresh or self._should_refresh():
                try:
                    self._load_from_database()
                except Exception as e:
                    logger.error(f"Failed to load config from database: {e}")
                    # Fall back to last-known-good
                    if self._last_known_good is not None:
                        logger.warning("Using last-known-good configuration")
                        self._cached_config = copy.deepcopy(self._last_known_good)
                        self._cached_config.source = 'cache'
                        self._cache_timestamp = datetime.now()

            if self._cached_config is None:
                raise RuntimeError("No configuration available - database required")

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
                # Load global config
                cur.execute("""
                    SELECT * FROM smc_simple_global_config
                    WHERE is_active = TRUE
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                global_row = cur.fetchone()

                if global_row is None:
                    raise ValueError("No active configuration found in database")

                # Load pair overrides
                cur.execute("""
                    SELECT * FROM smc_simple_pair_overrides
                    WHERE config_id = %s AND is_enabled = TRUE
                """, (global_row['id'],))
                override_rows = cur.fetchall()

            # Build config object
            config = self._build_config_from_rows(global_row, override_rows)
            config.source = 'database'
            config.loaded_at = datetime.now()
            config.config_id = global_row['id']

            # Update cache
            self._cached_config = config
            self._cache_timestamp = datetime.now()

            # Update last-known-good
            self._last_known_good = copy.deepcopy(config)

            logger.info(f"Loaded SMC Simple config v{config.version} from database")

    def _build_config_from_rows(
        self,
        global_row: Dict,
        override_rows: List[Dict]
    ) -> SMCSimpleConfig:
        """Build SMCSimpleConfig from database rows"""
        config = SMCSimpleConfig()

        # Map database columns to config attributes (snake_case matches)
        direct_mappings = [
            'version', 'strategy_name', 'strategy_status',
            'htf_timeframe', 'ema_period', 'ema_buffer_pips',
            'require_close_beyond_ema', 'min_distance_from_ema_pips',
            'trigger_timeframe', 'swing_lookback_bars', 'swing_strength_bars',
            'require_body_close_break', 'wick_tolerance_pips',
            'volume_confirmation_enabled', 'volume_sma_period', 'volume_spike_multiplier',
            'use_dynamic_swing_lookback', 'swing_lookback_atr_low', 'swing_lookback_atr_high',
            'swing_lookback_min', 'swing_lookback_max',
            'entry_timeframe', 'pullback_enabled',
            'fib_pullback_min', 'fib_pullback_max',
            'fib_optimal_zone_min', 'fib_optimal_zone_max',
            'max_pullback_wait_bars', 'pullback_confirmation_bars',
            'momentum_mode_enabled', 'momentum_min_depth', 'momentum_max_depth',
            'momentum_confidence_penalty',
            'use_atr_swing_validation', 'atr_period', 'min_swing_atr_multiplier',
            'fallback_min_swing_pips',
            'momentum_quality_enabled', 'min_breakout_atr_ratio', 'min_body_percentage',
            'limit_order_enabled', 'limit_expiry_minutes',
            'pullback_offset_atr_factor', 'pullback_offset_min_pips', 'pullback_offset_max_pips',
            'momentum_offset_pips', 'min_risk_after_offset_pips',
            'max_sl_atr_multiplier', 'max_sl_absolute_pips', 'max_risk_after_offset_pips',
            'min_rr_ratio', 'optimal_rr_ratio', 'max_rr_ratio',
            'sl_buffer_pips', 'sl_atr_multiplier', 'use_atr_stop',
            'min_tp_pips', 'use_swing_target', 'tp_structure_lookback', 'risk_per_trade_pct',
            'fixed_sl_tp_override_enabled', 'fixed_stop_loss_pips', 'fixed_take_profit_pips',
            'session_filter_enabled', 'block_asian_session',
            'max_concurrent_signals', 'signal_cooldown_hours',
            'min_confidence_threshold', 'max_confidence_threshold', 'high_confidence_threshold',
            'min_volume_ratio', 'volume_filter_enabled', 'allow_no_volume_data',
            'volume_adjusted_confidence_enabled', 'high_volume_threshold',
            'atr_adjusted_confidence_enabled', 'low_atr_threshold', 'high_atr_threshold',
            'ema_distance_adjusted_confidence_enabled',
            'near_ema_threshold_pips', 'far_ema_threshold_pips',
            'macd_alignment_filter_enabled', 'macd_alignment_mode', 'macd_min_strength',
            'enable_debug_logging', 'log_rejected_signals', 'log_swing_detection', 'log_ema_checks',
            'rejection_tracking_enabled', 'rejection_batch_size',
            'rejection_log_to_console', 'rejection_retention_days',
            'backtest_spread_pips', 'backtest_slippage_pips',
        ]

        # Fields that must be integers (used for DataFrame slicing, loop counts, etc.)
        int_fields = {
            'ema_period', 'swing_lookback_bars', 'swing_strength_bars',
            'volume_sma_period', 'swing_lookback_atr_low', 'swing_lookback_atr_high',
            'swing_lookback_min', 'swing_lookback_max',
            'max_pullback_wait_bars', 'pullback_confirmation_bars',
            'atr_period', 'limit_expiry_minutes',
            'sl_buffer_pips', 'min_tp_pips', 'tp_structure_lookback',
            'max_concurrent_signals', 'signal_cooldown_hours',
            'max_consecutive_losses_before_block', 'win_rate_lookback_trades',
            'rejection_batch_size', 'rejection_retention_days',
        }

        for attr_name in direct_mappings:
            if attr_name in global_row and global_row[attr_name] is not None:
                value = global_row[attr_name]
                # Convert to appropriate type
                if attr_name in int_fields:
                    value = int(value)
                elif hasattr(value, 'as_integer_ratio'):  # Decimal to float
                    value = float(value)
                setattr(config, attr_name, value)

        # Handle time fields
        for time_field in ['london_session_start', 'london_session_end',
                          'ny_session_start', 'ny_session_end']:
            if time_field in global_row and global_row[time_field] is not None:
                value = global_row[time_field]
                if isinstance(value, str):
                    h, m = map(int, value.split(':')[:2])
                    value = dt_time(h, m)
                setattr(config, time_field, value)

        # Handle array fields
        if global_row.get('allowed_sessions'):
            config.allowed_sessions = list(global_row['allowed_sessions'])
        if global_row.get('enabled_pairs'):
            config.enabled_pairs = list(global_row['enabled_pairs'])

        # Handle JSONB fields
        if global_row.get('adaptive_cooldown_config'):
            cooldown = global_row['adaptive_cooldown_config']
            if isinstance(cooldown, str):
                cooldown = json.loads(cooldown)
            # Map to individual attributes
            config.adaptive_cooldown_enabled = cooldown.get('enabled', True)
            config.base_cooldown_hours = cooldown.get('base_cooldown_hours', 2.0)
            config.cooldown_after_win_multiplier = cooldown.get('cooldown_after_win_multiplier', 0.5)
            config.cooldown_after_loss_multiplier = cooldown.get('cooldown_after_loss_multiplier', 1.5)
            config.consecutive_loss_penalty_hours = cooldown.get('consecutive_loss_penalty_hours', 1.0)
            config.max_consecutive_losses_before_block = cooldown.get('max_consecutive_losses_before_block', 3)
            config.consecutive_loss_block_hours = cooldown.get('consecutive_loss_block_hours', 8.0)
            config.win_rate_lookback_trades = cooldown.get('win_rate_lookback_trades', 20)
            config.high_win_rate_threshold = cooldown.get('high_win_rate_threshold', 0.60)
            config.low_win_rate_threshold = cooldown.get('low_win_rate_threshold', 0.40)
            config.critical_win_rate_threshold = cooldown.get('critical_win_rate_threshold', 0.30)
            config.high_win_rate_cooldown_reduction = cooldown.get('high_win_rate_cooldown_reduction', 0.25)
            config.low_win_rate_cooldown_increase = cooldown.get('low_win_rate_cooldown_increase', 0.50)
            config.high_volatility_atr_multiplier = cooldown.get('high_volatility_atr_multiplier', 1.5)
            config.volatility_cooldown_adjustment = cooldown.get('volatility_cooldown_adjustment', 0.30)
            config.strong_trend_cooldown_reduction = cooldown.get('strong_trend_cooldown_reduction', 0.30)
            config.session_change_reset_cooldown = cooldown.get('session_change_reset_cooldown', True)
            config.min_cooldown_hours = cooldown.get('min_cooldown_hours', 1.0)
            config.max_cooldown_hours = cooldown.get('max_cooldown_hours', 12.0)

        if global_row.get('confidence_weights'):
            weights = global_row['confidence_weights']
            if isinstance(weights, str):
                weights = json.loads(weights)
            config.confidence_weights = weights

        if global_row.get('pair_pip_values'):
            pip_values = global_row['pair_pip_values']
            if isinstance(pip_values, str):
                pip_values = json.loads(pip_values)
            config.pair_pip_values = pip_values

        # Build pair overrides dict
        config._pair_overrides = {}
        for row in override_rows:
            epic = row['epic']
            config._pair_overrides[epic] = {
                'is_enabled': row.get('is_enabled', True),
                'description': row.get('description'),
                'parameter_overrides': row.get('parameter_overrides') or {},
                'allow_asian_session': row.get('allow_asian_session'),
                'sl_buffer_pips': row.get('sl_buffer_pips'),
                'min_confidence': row.get('min_confidence'),
                'max_confidence': row.get('max_confidence'),
                'min_volume_ratio': row.get('min_volume_ratio'),
                'high_volume_confidence': row.get('high_volume_confidence'),
                'low_atr_confidence': row.get('low_atr_confidence'),
                'high_atr_confidence': row.get('high_atr_confidence'),
                'near_ema_confidence': row.get('near_ema_confidence'),
                'far_ema_confidence': row.get('far_ema_confidence'),
                'macd_filter_enabled': row.get('macd_filter_enabled'),
                'blocking_conditions': row.get('blocking_conditions'),
                'fixed_stop_loss_pips': row.get('fixed_stop_loss_pips'),
                'fixed_take_profit_pips': row.get('fixed_take_profit_pips'),
            }

        return config

    def get_effective_config_for_pair(self, epic: str) -> Dict[str, Any]:
        """
        Get the effective configuration for a specific pair.
        Merges global config with pair-specific overrides.
        """
        config = self.get_config()

        # Start with global values
        effective = {}
        for attr_name in dir(config):
            if not attr_name.startswith('_') and not callable(getattr(config, attr_name)):
                value = getattr(config, attr_name)
                # Skip non-serializable types
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    effective[attr_name] = value
                elif isinstance(value, dt_time):
                    effective[attr_name] = value.strftime('%H:%M')

        # Apply pair overrides
        if epic in config._pair_overrides:
            overrides = config._pair_overrides[epic]
            if overrides.get('parameter_overrides'):
                effective.update(overrides['parameter_overrides'])

            # Apply specific override fields
            for field_name in ['sl_buffer_pips', 'min_confidence', 'max_confidence', 'allow_asian_session',
                              'min_volume_ratio', 'macd_filter_enabled']:
                if overrides.get(field_name) is not None:
                    effective[field_name] = overrides[field_name]

        return effective

    def invalidate_cache(self):
        """Force cache invalidation"""
        with self._lock:
            self._cache_timestamp = None
            logger.info("SMC Simple config cache invalidated")


# Global singleton instance
_service_instance: Optional[SMCSimpleConfigService] = None
_service_lock = RLock()


def get_smc_simple_config_service(
    database_url: str = None,
    cache_ttl_seconds: int = 120,
    enable_hot_reload: bool = True
) -> SMCSimpleConfigService:
    """Get singleton instance of config service"""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = SMCSimpleConfigService(
                database_url=database_url,
                cache_ttl_seconds=cache_ttl_seconds,
                enable_hot_reload=enable_hot_reload
            )
        return _service_instance


def get_smc_simple_config() -> SMCSimpleConfig:
    """Convenience function to get current config"""
    return get_smc_simple_config_service().get_config()
