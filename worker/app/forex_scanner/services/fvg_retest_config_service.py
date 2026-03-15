"""
FVG Retest Strategy Configuration Service

Database-driven configuration for the dual-mode FVG Retest strategy:
- Type A (Deep Value): FVG tap entries after BOS
- Type B (Institutional Initiation): Immediate entries on high-velocity BOS

Pattern: Follows smc_simple_config_service.py conventions.
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
class FVGRetestConfig:
    """Configuration object for FVG Retest strategy"""

    # STRATEGY METADATA
    strategy_name: str = "FVG_RETEST"
    version: str = "1.0.0"
    enabled: bool = True

    # TIMEFRAMES
    htf_timeframe: str = "1h"
    trigger_timeframe: str = "5m"

    # HTF FILTER
    htf_ema_period: int = 200

    # SHARED RISK
    fixed_stop_loss_pips: float = 10.0
    fixed_take_profit_pips: float = 12.0
    sl_buffer_pips: float = 3.0
    min_rr_ratio: float = 1.0

    # CONFIDENCE
    min_confidence: float = 0.65
    max_confidence: float = 0.90

    # SWING DETECTION (Pine: pivotLen=5 on higher TF; reduced to 3 for 5m chart)
    swing_lookback_bars: int = 40
    swing_strength_bars: int = 3
    atr_period: int = 14

    # TYPE A: FVG TAP (DEEP VALUE)
    fvg_min_size_pips: float = 3.0
    fvg_max_age_bars: int = 48  # was 20; aligned with 4h setup_expiry on 5m bars
    fvg_max_fill_pct: float = 0.80
    setup_expiry_hours: float = 4.0
    max_pending_per_pair: int = 3

    # TYPE B: INSTITUTIONAL INITIATION
    initiation_enabled: bool = True
    displacement_atr_multiplier: float = 1.2  # was 1.5; captures more institutional breaks
    follow_through_candles: int = 1  # was 2; single follow-through is sufficient on 5m
    volume_threshold_multiplier: float = 1.05  # was 1.1; lower bar for volume confirmation

    # VOLUME
    volume_sma_period: int = 20

    # COOLDOWN
    signal_cooldown_minutes: int = 60

    # ENABLED PAIRS
    enabled_pairs: List[str] = field(default_factory=list)

    # PAIR PIP VALUES (same as system)
    pair_pip_values: Dict[str, float] = field(default_factory=lambda: {
        'CS.D.EURUSD.CEEM.IP': 0.0001,
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
    source: str = "database"

    def get_pip_value(self, epic: str) -> float:
        return self.pair_pip_values.get(epic, 0.0001)

    def is_pair_enabled(self, epic: str) -> bool:
        if epic in self._pair_overrides:
            return self._pair_overrides[epic].get('is_enabled', True)
        return True

    def get_for_pair(self, epic: str, param_name: str, default: Any = None) -> Any:
        """Get parameter with pair-specific override fallback"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if param_name in override and override[param_name] is not None:
                return override[param_name]
            param_overrides = override.get('parameter_overrides', {})
            if param_name in param_overrides:
                return param_overrides[param_name]
        if hasattr(self, param_name):
            return getattr(self, param_name)
        return default

    def get_pair_fixed_stop_loss(self, epic: str) -> float:
        if epic in self._pair_overrides:
            val = self._pair_overrides[epic].get('fixed_stop_loss_pips')
            if val is not None:
                return float(val)
        return self.fixed_stop_loss_pips

    def get_pair_fixed_take_profit(self, epic: str) -> float:
        if epic in self._pair_overrides:
            val = self._pair_overrides[epic].get('fixed_take_profit_pips')
            if val is not None:
                return float(val)
        return self.fixed_take_profit_pips

    def get_pair_sl_buffer(self, epic: str) -> float:
        if epic in self._pair_overrides:
            val = self._pair_overrides[epic].get('sl_buffer_pips')
            if val is not None:
                return float(val)
        return self.sl_buffer_pips

    def get_pair_min_confidence(self, epic: str) -> float:
        if epic in self._pair_overrides:
            val = self._pair_overrides[epic].get('min_confidence')
            if val is not None:
                return float(val)
        return self.min_confidence

    def get_pair_cooldown_minutes(self, epic: str) -> int:
        if epic in self._pair_overrides:
            val = self._pair_overrides[epic].get('signal_cooldown_minutes')
            if val is not None:
                return int(val)
        return self.signal_cooldown_minutes


class FVGRetestConfigService:
    """
    Database-driven configuration service with in-memory caching.

    Features:
    - Loads config from strategy_config database
    - 120s TTL cache with RLock
    - Last-known-good fallback
    """

    def __init__(
        self,
        database_url: str = None,
        cache_ttl_seconds: int = 120,
    ):
        self.database_url = database_url or self._get_default_database_url()
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)

        self._lock = RLock()
        self._cached_config: Optional[FVGRetestConfig] = None
        self._cache_timestamp: Optional[datetime] = None
        self._last_known_good: Optional[FVGRetestConfig] = None

        self._load_initial_config()

    def _get_default_database_url(self) -> str:
        return os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config'
        )

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        finally:
            if conn:
                conn.close()

    def _load_initial_config(self):
        try:
            self._load_from_database()
            logger.info("FVG Retest config service initialized from database")
        except Exception as e:
            logger.warning(f"Failed to load FVG Retest config from database: {e}")
            self._cached_config = FVGRetestConfig()
            self._cached_config.source = 'default'
            self._cache_timestamp = datetime.now()
            logger.info("Using default FVG Retest config")

    def get_config(self, force_refresh: bool = False) -> FVGRetestConfig:
        with self._lock:
            if force_refresh or self._should_refresh():
                try:
                    self._load_from_database()
                except Exception as e:
                    logger.error(f"Failed to load FVG Retest config: {e}")
                    if self._last_known_good is not None:
                        logger.warning("Using last-known-good FVG Retest config")
                        self._cached_config = copy.deepcopy(self._last_known_good)
                        self._cached_config.source = 'cache'
                        self._cache_timestamp = datetime.now()

            if self._cached_config is None:
                raise RuntimeError("No FVG Retest configuration available")

            return self._cached_config

    def _should_refresh(self) -> bool:
        if self._cache_timestamp is None:
            return True
        return datetime.now() - self._cache_timestamp > self.cache_ttl

    def _load_from_database(self):
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Load global config (key-value rows)
                cur.execute("""
                    SELECT parameter_name, parameter_value, value_type
                    FROM fvg_retest_global_config
                    WHERE is_active = TRUE
                """)
                global_rows = cur.fetchall()

                # Load pair overrides
                cur.execute("""
                    SELECT * FROM fvg_retest_pair_overrides
                    WHERE is_enabled = TRUE
                """)
                override_rows = cur.fetchall()

        config = self._build_config(global_rows, override_rows)
        config.source = 'database'
        config.loaded_at = datetime.now()

        self._cached_config = config
        self._cache_timestamp = datetime.now()
        self._last_known_good = copy.deepcopy(config)

        logger.info(f"Loaded FVG Retest config v{config.version} from database")

    def _build_config(
        self,
        global_rows: list,
        override_rows: list,
    ) -> FVGRetestConfig:
        config = FVGRetestConfig()

        # Type conversion map
        type_converters = {
            'string': str,
            'int': int,
            'float': float,
            'bool': lambda v: str(v).lower() in ('true', '1', 'yes'),
            'json': json.loads,
        }

        # Map parameter names to config fields
        for row in global_rows:
            param_name = row['parameter_name']
            raw_value = row['parameter_value']
            value_type = row.get('value_type', 'string')

            if hasattr(config, param_name):
                try:
                    converter = type_converters.get(value_type, str)
                    setattr(config, param_name, converter(raw_value))
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to convert {param_name}={raw_value} as {value_type}: {e}")

        # Build pair overrides
        pair_overrides = {}
        for row in override_rows:
            epic = row['epic']
            override_data = dict(row)
            # Parse JSONB parameter_overrides
            if isinstance(override_data.get('parameter_overrides'), str):
                try:
                    override_data['parameter_overrides'] = json.loads(override_data['parameter_overrides'])
                except (json.JSONDecodeError, TypeError):
                    override_data['parameter_overrides'] = {}
            pair_overrides[epic] = override_data

        config._pair_overrides = pair_overrides

        # Build enabled pairs list from overrides
        config.enabled_pairs = [
            epic for epic, data in pair_overrides.items()
            if data.get('is_enabled', True)
        ]

        return config


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_service_instance: Optional[FVGRetestConfigService] = None
_service_lock = RLock()


def get_fvg_retest_config(force_refresh: bool = False) -> FVGRetestConfig:
    """Convenience function to get FVG Retest config (singleton service)"""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = FVGRetestConfigService()
    return _service_instance.get_config(force_refresh=force_refresh)
