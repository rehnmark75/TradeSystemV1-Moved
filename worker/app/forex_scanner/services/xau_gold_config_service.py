"""
XAU_GOLD Strategy Configuration Service

Database-driven config for the gold-optimized strategy.
Loads key/value rows from `xau_gold_global_config` and per-pair rows
from `xau_gold_pair_overrides`, with in-memory caching and
last-known-good fallback.
"""

from __future__ import annotations

import os
import json
import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class XAUGoldConfig:
    strategy_name: str = "XAU_GOLD"
    version: str = "1.0.0"
    enabled: bool = True
    config_set: str = "demo"

    # Timeframes
    htf_timeframe: str = "4h"
    trigger_timeframe: str = "1h"
    entry_timeframe: str = "15m"

    # Indicators
    ema_fast_period: int = 50
    ema_slow_period: int = 200
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    swing_lookback: int = 20

    # Confidence
    min_confidence: float = 0.58
    max_confidence: float = 0.75
    base_confidence: float = 0.50

    # Confluence weights
    w_htf_bias: float = 0.08
    w_bos_displacement: float = 0.08
    w_entry_pullback: float = 0.06
    w_dxy_confluence: float = 0.05
    w_rsi_neutral: float = 0.04

    # Risk
    sl_atr_multiplier: float = 1.5
    min_stop_loss_pips: float = 25.0
    max_stop_loss_pips: float = 80.0
    rr_ratio: float = 2.0
    min_rr_ratio: float = 1.33
    min_tp_pips: float = 15.0
    fixed_sl_tp_override_enabled: bool = False
    fixed_stop_loss_pips: float = 40.0
    fixed_take_profit_pips: float = 80.0

    # Regime
    adx_trending_threshold: float = 25.0
    adx_ranging_threshold: float = 20.0
    atr_expansion_pct: float = 85.0
    atr_pct_lookback_bars: int = 120
    block_ranging: bool = True
    block_expansion: bool = True

    # Session
    session_filter_enabled: bool = True
    london_start_hour: int = 7
    london_end_hour: int = 10
    ny_start_hour: int = 13
    ny_end_hour: int = 20
    rollover_start_hour: int = 21
    rollover_end_hour: int = 22
    asian_allowed: bool = False

    # Structure
    bos_displacement_atr_mult: float = 1.2
    fib_pullback_min: float = 0.382
    fib_pullback_max: float = 0.618
    bos_expiry_hours: float = 12.0
    bos_search_bars: int = 24
    entry_check_bars: int = 12
    require_ob_or_fvg: bool = True

    # Limits
    signal_cooldown_minutes: int = 180
    max_concurrent_signals: int = 1

    # Filters
    macd_filter_enabled: bool = True
    dxy_confluence_enabled: bool = True
    rsi_neutral_min: float = 40.0
    rsi_neutral_max: float = 60.0

    enabled_pairs: List[str] = field(default_factory=lambda: ["CS.D.CFEGOLD.CEE.IP"])

    # Pair overrides (keyed by epic)
    pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    source: str = "default"
    loaded_at: datetime = field(default_factory=datetime.now)

    # ---- per-pair helpers -------------------------------------------------

    def _pair(self, epic: str) -> Dict[str, Any]:
        return self.pair_overrides.get(epic, {})

    def is_pair_enabled(self, epic: str) -> bool:
        if self.enabled_pairs and epic not in self.enabled_pairs:
            return False
        row = self._pair(epic)
        if row and not row.get("is_enabled", True):
            return False
        return True

    def is_monitor_only(self, epic: str) -> bool:
        row = self._pair(epic)
        return bool(row.get("monitor_only", True))

    def get_pip_size(self, epic: str) -> float:
        row = self._pair(epic)
        return float(row.get("pip_size", 0.1))

    def _override(self, epic: str, key: str, default: Any) -> Any:
        row = self._pair(epic)
        if not row:
            return default
        if row.get(key) is not None:
            return row[key]
        bag = row.get("parameter_overrides") or {}
        if isinstance(bag, str):
            try:
                bag = json.loads(bag)
            except Exception:
                bag = {}
        if key in bag and bag[key] is not None:
            return bag[key]
        return default

    def get_pair_min_confidence(self, epic: str) -> float:
        return float(self._override(epic, "min_confidence", self.min_confidence))

    def get_pair_max_confidence(self, epic: str) -> float:
        return float(self._override(epic, "max_confidence", self.max_confidence))

    def get_pair_sl_atr_mult(self, epic: str) -> float:
        return float(self._override(epic, "sl_atr_multiplier", self.sl_atr_multiplier))

    def get_pair_rr_ratio(self, epic: str) -> float:
        return float(self._override(epic, "rr_ratio", self.rr_ratio))

    def get_pair_cooldown_minutes(self, epic: str) -> int:
        return int(self._override(epic, "signal_cooldown_minutes", self.signal_cooldown_minutes))

    def get_pair_fixed_stop_loss(self, epic: str) -> Optional[float]:
        if not self.fixed_sl_tp_override_enabled:
            return None
        val = self._override(epic, "fixed_stop_loss_pips", self.fixed_stop_loss_pips)
        return float(val) if val is not None else None

    def get_pair_fixed_take_profit(self, epic: str) -> Optional[float]:
        if not self.fixed_sl_tp_override_enabled:
            return None
        val = self._override(epic, "fixed_take_profit_pips", self.fixed_take_profit_pips)
        return float(val) if val is not None else None

    # ---- session helpers --------------------------------------------------

    def is_session_allowed(self, hour_utc: int) -> bool:
        if not self.session_filter_enabled:
            return True
        # Hard-block rollover
        if self.rollover_start_hour <= hour_utc < self.rollover_end_hour:
            return False
        in_london = self.london_start_hour <= hour_utc < self.london_end_hour
        in_ny = self.ny_start_hour <= hour_utc < self.ny_end_hour
        if in_london or in_ny:
            return True
        # Asian / off-hours
        return bool(self.asian_allowed)


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

_TRUE_STR = {"true", "1", "yes", "y", "t"}


def _coerce(value: str, value_type: str) -> Any:
    if value is None:
        return None
    t = (value_type or "string").lower()
    try:
        if t == "int":
            return int(float(value))
        if t == "float":
            return float(value)
        if t == "bool":
            return str(value).strip().lower() in _TRUE_STR
        if t == "json":
            if isinstance(value, (dict, list)):
                return value
            return json.loads(value)
    except Exception:
        logger.warning(f"XAU_GOLD config: failed to coerce {value!r} as {t}, keeping string")
    return value


# ---------------------------------------------------------------------------
# Service (singleton)
# ---------------------------------------------------------------------------

class XAUGoldConfigService:
    _instance: Optional["XAUGoldConfigService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_ttl_seconds: int = 300):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._lock = RLock()
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cached: Optional[XAUGoldConfig] = None
        self._cache_ts: Optional[datetime] = None
        self._last_known_good: Optional[XAUGoldConfig] = None
        self.config_set = os.getenv("TRADING_CONFIG_SET", "demo")
        self._trading_env = os.getenv("TRADING_ENVIRONMENT", "demo")
        self._db_url = os.getenv(
            "STRATEGY_CONFIG_DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/strategy_config",
        )

    @classmethod
    def get_instance(cls) -> "XAUGoldConfigService":
        return cls()

    def refresh(self) -> XAUGoldConfig:
        with self._lock:
            self._cached = None
            return self.get_config()

    def get_config(self) -> XAUGoldConfig:
        with self._lock:
            if self._cached is not None and self._cache_ts is not None:
                if datetime.now() - self._cache_ts < self._cache_ttl:
                    return self._cached
            try:
                cfg = self._load_from_db()
                self._cached = cfg
                self._cache_ts = datetime.now()
                self._last_known_good = copy.deepcopy(cfg)
                return cfg
            except Exception as e:
                logger.warning(f"XAU_GOLD config DB load failed (config_set={self.config_set}): {e}")
                if self._last_known_good is not None:
                    cfg = copy.deepcopy(self._last_known_good)
                    cfg.source = "cache"
                    self._cached = cfg
                    self._cache_ts = datetime.now()
                    return cfg
                if self._trading_env == "live":
                    raise RuntimeError(
                        f"Refusing to use default XAU_GOLD config in live mode "
                        f"(config_set={self.config_set}): {e}"
                    ) from e
                cfg = XAUGoldConfig()
                cfg.source = "default"
                cfg.config_set = self.config_set
                self._cached = cfg
                self._cache_ts = datetime.now()
                return cfg

    @contextmanager
    def _conn(self):
        conn = psycopg2.connect(self._db_url)
        try:
            yield conn
        finally:
            conn.close()

    def _load_from_db(self) -> XAUGoldConfig:
        cfg = XAUGoldConfig()
        cfg.config_set = self.config_set
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT parameter_name, parameter_value, value_type "
                    "FROM xau_gold_global_config "
                    "WHERE is_active = TRUE AND config_set = %s",
                    (self.config_set,),
                )
                for row in cur.fetchall():
                    name = row["parameter_name"]
                    if not hasattr(cfg, name) and name != "enabled_pairs":
                        continue
                    val = _coerce(row["parameter_value"], row["value_type"])
                    if name == "enabled_pairs" and isinstance(val, list):
                        cfg.enabled_pairs = [str(x) for x in val]
                    else:
                        setattr(cfg, name, val)

                cur.execute(
                    "SELECT * FROM xau_gold_pair_overrides WHERE config_set = %s",
                    (self.config_set,),
                )
                for row in cur.fetchall():
                    epic = row["epic"]
                    data = dict(row)
                    # Normalize jsonb
                    po = data.get("parameter_overrides")
                    if isinstance(po, str):
                        try:
                            data["parameter_overrides"] = json.loads(po)
                        except Exception:
                            data["parameter_overrides"] = {}
                    cfg.pair_overrides[epic] = data

        cfg.source = "database"
        cfg.loaded_at = datetime.now()
        logger.info(
            f"XAU_GOLD config loaded from DB "
            f"(config_set={self.config_set}, pairs={len(cfg.pair_overrides)}, enabled_pairs={cfg.enabled_pairs})"
        )
        return cfg


def get_xau_gold_config() -> XAUGoldConfig:
    return XAUGoldConfigService.get_instance().get_config()
