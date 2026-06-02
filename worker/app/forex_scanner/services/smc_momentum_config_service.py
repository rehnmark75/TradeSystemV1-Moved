"""
SMC_MOMENTUM Strategy Configuration Service

Database-driven config for the Liquidity Sweep + Rejection Wick strategy.
Loads key/value rows from `smc_momentum_global_config` and per-pair rows
from `smc_momentum_pair_overrides`, with in-memory caching and
last-known-good fallback.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

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
        logger.warning(f"SMC_MOMENTUM config: failed to coerce {value!r} as {t}")
    return value


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    return [value]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SMCMomentumConfig:
    strategy_name: str = "SMC_MOMENTUM"
    version: str = "1.0.0"
    is_active: bool = False
    config_set: str = "demo"

    # Timeframes
    htf_timeframe: str = "4h"
    entry_timeframe: str = "15m"
    atr_timeframe: str = "1h"

    # Sweep detection
    sweep_min_pips: float = 3.0
    sweep_max_pips: float = 15.0
    sweep_min_pips_jpy: float = 3.0
    sweep_max_pips_jpy: float = 30.0

    # Risk
    sl_buffer_pips: float = 5.0
    sl_buffer_pips_jpy: float = 8.0
    tp_atr_multiplier: float = 2.0
    atr_period: int = 14

    # Confidence
    min_confidence: float = 0.55
    max_confidence: float = 0.78

    # HTF alignment (Gate 1 confirmed load-bearing)
    htf_alignment_required: bool = True
    htf_ema_period: int = 50

    # Momentum quality filter
    momentum_filter_mode: str = "off"   # off | volume | atr_expansion
    volume_multiplier_threshold: float = 1.3
    atr_expansion_threshold: float = 1.3

    # Swing pivot settings
    swing_pivot_bars: int = 2
    swing_max_age_bars: int = 20

    # Rollover block
    rollover_block_enabled: bool = True
    rollover_start_hour: int = 21
    rollover_end_hour: int = 23

    # Cooldown
    cooldown_minutes: int = 240

    # Wick quality filter (fraction of bar range the wick must represent)
    wick_min_pct_of_range: float = 0.50

    # Kill zone filter (London 07-10 + NY 12-15 UTC; disabled by default for backward compat)
    kill_zone_only: bool = False
    kill_zone_london_start: int = 7
    kill_zone_london_end: int = 10
    kill_zone_ny_start: int = 12
    kill_zone_ny_end: int = 15

    # Asian session block (00:00-07:00 UTC; disabled by default)
    block_asian_session: bool = False
    asian_session_start: int = 0
    asian_session_end: int = 7

    # HTF trend strength gate (0 = disabled; >0 = min pips distance from EMA50)
    htf_min_distance_pips: float = 0.0

    # Pair overrides (keyed by epic)
    pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    source: str = "default"
    loaded_at: datetime = field(default_factory=datetime.now)

    # ---- per-pair helpers -------------------------------------------------

    def _pair(self, epic: str) -> Dict[str, Any]:
        return self.pair_overrides.get(epic, {})

    def _override(self, epic: str, key: str, default: Any) -> Any:
        row = self._pair(epic)
        if not row:
            return default
        val = row.get(key)
        if val is not None:
            return val
        bag = row.get("parameter_overrides") or {}
        if isinstance(bag, str):
            try:
                bag = json.loads(bag)
            except Exception:
                bag = {}
        if key in bag and bag[key] is not None:
            return bag[key]
        return default

    def is_pair_enabled(self, epic: str) -> bool:
        if not self.is_active:
            return False
        row = self._pair(epic)
        if not row:
            return False
        return bool(row.get("is_enabled", False))

    def is_monitor_only(self, epic: str) -> bool:
        # Default to False (trade enabled pairs). Only return True if explicitly set in DB.
        # Bug fix (May 29 2026): was defaulting to True, blocking all enabled pairs.
        val = self._override(epic, "monitor_only", False)
        if isinstance(val, str):
            return val.strip().lower() in _TRUE_STR
        return bool(val)

    def is_traded(self, epic: str) -> bool:
        row = self._pair(epic)
        return bool(row.get("is_traded", False))

    def get_pair_min_confidence(self, epic: str) -> float:
        return float(self._override(epic, "min_confidence", self.min_confidence))

    def get_pair_tp_atr_multiplier(self, epic: str) -> float:
        return float(self._override(epic, "tp_atr_multiplier", self.tp_atr_multiplier))

    def get_pair_sl_buffer_pips(self, epic: str, is_jpy: bool = False) -> float:
        default = self.sl_buffer_pips_jpy if is_jpy else self.sl_buffer_pips
        return float(self._override(epic, "sl_buffer_pips", default))

    def get_pair_sweep_min(self, epic: str, is_jpy: bool = False) -> float:
        default = self.sweep_min_pips_jpy if is_jpy else self.sweep_min_pips
        return float(self._override(epic, "sweep_min_pips", default))

    def get_pair_sweep_max(self, epic: str, is_jpy: bool = False) -> float:
        default = self.sweep_max_pips_jpy if is_jpy else self.sweep_max_pips
        return float(self._override(epic, "sweep_max_pips", default))

    def get_pair_htf_alignment_required(self, epic: str) -> bool:
        val = self._override(epic, "htf_alignment_required", self.htf_alignment_required)
        if isinstance(val, str):
            return val.strip().lower() in _TRUE_STR
        return bool(val)

    def get_pair_momentum_filter_mode(self, epic: str) -> str:
        return str(self._override(epic, "momentum_filter_mode", self.momentum_filter_mode))

    def get_pair_cooldown_minutes(self, epic: str) -> int:
        return int(self._override(epic, "cooldown_minutes", self.cooldown_minutes))

    def get_pair_htf_min_distance_pips(self, epic: str) -> float:
        return float(self._override(epic, "htf_min_distance_pips", self.htf_min_distance_pips))

    def get_pair_wick_min_pct(self, epic: str) -> float:
        return float(self._override(epic, "wick_min_pct_of_range", self.wick_min_pct_of_range))

    def get_pair_block_asian_session(self, epic: str) -> bool:
        val = self._override(epic, "block_asian_session", self.block_asian_session)
        if isinstance(val, str):
            return val.strip().lower() in _TRUE_STR
        return bool(val)

    def get_pair_allowed_directions(self, epic: str) -> List[str]:
        values = _coerce_list(self._override(epic, "allowed_directions", []))
        aliases = {
            "BUY": "BUY",
            "BULL": "BUY",
            "BULLISH": "BUY",
            "LONG": "BUY",
            "SELL": "SELL",
            "BEAR": "SELL",
            "BEARISH": "SELL",
            "SHORT": "SELL",
        }
        directions: List[str] = []
        for value in values:
            direction = aliases.get(str(value).strip().upper())
            if direction and direction not in directions:
                directions.append(direction)
        return directions

    def get_pair_allowed_hours_utc(self, epic: str) -> List[int]:
        return self._get_hour_list(epic, "allowed_hours_utc")

    def get_pair_blocked_hours_utc(self, epic: str) -> List[int]:
        return self._get_hour_list(epic, "blocked_hours_utc")

    def _get_hour_list(self, epic: str, key: str) -> List[int]:
        hours: List[int] = []
        for value in _coerce_list(self._override(epic, key, [])):
            try:
                hour = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= hour <= 23 and hour not in hours:
                hours.append(hour)
        return hours

    def is_rollover_hour(self, hour_utc: int) -> bool:
        if not self.rollover_block_enabled:
            return False
        return self.rollover_start_hour <= hour_utc < self.rollover_end_hour

    def is_blocked_session(self, hour_utc: int) -> bool:
        """True if the hour falls outside allowed trading windows (global config)."""
        if self.is_rollover_hour(hour_utc):
            return True
        if self.block_asian_session and self.asian_session_start <= hour_utc < self.asian_session_end:
            return True
        if self.kill_zone_only:
            in_london = self.kill_zone_london_start <= hour_utc < self.kill_zone_london_end
            in_ny = self.kill_zone_ny_start <= hour_utc < self.kill_zone_ny_end
            return not (in_london or in_ny)
        return False

    def is_pair_blocked_session(self, epic: str, hour_utc: int) -> bool:
        """True if the hour is blocked, respecting per-pair block_asian_session override."""
        if self.is_rollover_hour(hour_utc):
            return True
        if self.get_pair_block_asian_session(epic) and self.asian_session_start <= hour_utc < self.asian_session_end:
            return True
        if self.kill_zone_only:
            in_london = self.kill_zone_london_start <= hour_utc < self.kill_zone_london_end
            in_ny = self.kill_zone_ny_start <= hour_utc < self.kill_zone_ny_end
            return not (in_london or in_ny)
        return False


# ---------------------------------------------------------------------------
# Service (singleton + last-known-good fallback)
# ---------------------------------------------------------------------------

class SMCMomentumConfigService:
    _instance: Optional["SMCMomentumConfigService"] = None

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
        self._cached: Optional[SMCMomentumConfig] = None
        self._cache_ts: Optional[datetime] = None
        self._last_known_good: Optional[SMCMomentumConfig] = None
        self.config_set = os.getenv("TRADING_CONFIG_SET", "demo")
        self._trading_env = os.getenv("TRADING_ENVIRONMENT", "demo")
        self._db_url = os.getenv(
            "STRATEGY_CONFIG_DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/strategy_config",
        )

    @classmethod
    def get_instance(cls) -> "SMCMomentumConfigService":
        return cls()

    def refresh(self) -> SMCMomentumConfig:
        with self._lock:
            self._cached = None
            return self.get_config()

    def get_config(self) -> SMCMomentumConfig:
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
                logger.warning(f"SMC_MOMENTUM config DB load failed (config_set={self.config_set}): {e}")
                if self._last_known_good is not None:
                    cfg = copy.deepcopy(self._last_known_good)
                    cfg.source = "cache"
                    self._cached = cfg
                    self._cache_ts = datetime.now()
                    return cfg
                if self._trading_env == "live":
                    raise RuntimeError(
                        f"Refusing to use default SMC_MOMENTUM config in live mode: {e}"
                    ) from e
                cfg = SMCMomentumConfig()
                cfg.source = "default"
                cfg.config_set = self.config_set
                self._cached = cfg
                self._cache_ts = datetime.now()
                return cfg

    def _load_from_db(self) -> SMCMomentumConfig:
        cfg = SMCMomentumConfig()
        cfg.config_set = self.config_set
        conn = psycopg2.connect(self._db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT parameter_name, parameter_value, value_type "
                    "FROM smc_momentum_global_config "
                    "WHERE is_active = TRUE AND config_set = %s",
                    (self.config_set,),
                )
                for row in cur.fetchall():
                    name = row["parameter_name"]
                    if not hasattr(cfg, name):
                        continue
                    setattr(cfg, name, _coerce(row["parameter_value"], row["value_type"]))

                cur.execute(
                    "SELECT * FROM smc_momentum_pair_overrides WHERE config_set = %s",
                    (self.config_set,),
                )
                for row in cur.fetchall():
                    epic = row["epic"]
                    data = dict(row)
                    po = data.get("parameter_overrides")
                    if isinstance(po, str):
                        try:
                            data["parameter_overrides"] = json.loads(po)
                        except Exception:
                            data["parameter_overrides"] = {}
                    cfg.pair_overrides[epic] = data
        finally:
            conn.close()

        cfg.source = "database"
        cfg.loaded_at = datetime.now()
        logger.info(
            f"SMC_MOMENTUM config loaded (config_set={self.config_set}, "
            f"pairs={len(cfg.pair_overrides)}, is_active={cfg.is_active})"
        )
        return cfg


def get_smc_momentum_config() -> SMCMomentumConfig:
    return SMCMomentumConfigService.get_instance().get_config()


def apply_config_overrides(
    cfg: SMCMomentumConfig, overrides: Optional[Dict[str, Any]]
) -> SMCMomentumConfig:
    """Apply backtest --override key=value pairs to a loaded config in place."""
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if key == "_pair_overrides":
            if isinstance(value, dict):
                for epic, pair_params in value.items():
                    if not isinstance(pair_params, dict):
                        continue
                    row = cfg.pair_overrides.setdefault(
                        epic,
                        {
                            "epic": epic,
                            "is_enabled": True,
                            "is_traded": False,
                            "monitor_only": False,
                            "parameter_overrides": {},
                        },
                    )
                    bag = row.get("parameter_overrides") or {}
                    if isinstance(bag, str):
                        try:
                            bag = json.loads(bag)
                        except Exception:
                            bag = {}
                    for pair_key, pair_value in pair_params.items():
                        if pair_key in row:
                            row[pair_key] = pair_value
                        else:
                            bag[pair_key] = pair_value
                    row["parameter_overrides"] = bag
                    logger.info(f"SMC_MOMENTUM pair override applied: {epic}={pair_params}")
            continue
        if not hasattr(cfg, key):
            logger.debug(f"SMC_MOMENTUM override ignored (unknown key): {key}")
            continue
        current = getattr(cfg, key)
        try:
            if isinstance(current, bool):
                new_val = str(value).strip().lower() in _TRUE_STR
            elif isinstance(current, int):
                new_val = int(float(value))
            elif isinstance(current, float):
                new_val = float(value)
            else:
                new_val = value
            setattr(cfg, key, new_val)
            logger.info(f"SMC_MOMENTUM override applied: {key}={new_val}")
        except Exception as e:
            logger.warning(f"SMC_MOMENTUM override failed for {key}={value}: {e}")
    return cfg
