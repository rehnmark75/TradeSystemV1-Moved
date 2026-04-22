#!/usr/bin/env python3
"""
RANGE_FADE Config Service — DB-backed configuration with per-profile cache.

Source of truth: `strategy_config` database
    - eurusd_range_fade_global_config
    - eurusd_range_fade_pair_overrides

Falls back to code defaults if the tables are not present yet, so backtests and
tests keep working before the migration is applied.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras


logger = logging.getLogger(__name__)

EURUSD_EPIC = "CS.D.EURUSD.CEEM.IP"
EURJPY_EPIC = "CS.D.EURJPY.MINI.IP"
SUPPORTED_EPICS = [
    EURUSD_EPIC,
    EURJPY_EPIC,
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP",
    "CS.D.USDCHF.MINI.IP",
    "CS.D.USDCAD.MINI.IP",
    "CS.D.NZDUSD.MINI.IP",
    "CS.D.AUDJPY.MINI.IP",
]
_TRUE_STR = {"1", "true", "yes", "on", "y", "t"}


def _normalize_strategy_name(name: Any) -> Any:
    if name == "EURUSD_RANGE_FADE":
        return "RANGE_FADE"
    if name == "EURUSD_RANGE_FADE_5M":
        return "RANGE_FADE_5M"
    return name


@dataclass
class EURUSDRangeFadeConfig:
    strategy_name: str = "RANGE_FADE"
    profile_name: str = "15m"
    version: str = "0.3.0"

    enabled_pairs: List[str] = field(default_factory=lambda: list(SUPPORTED_EPICS))
    monitor_only: bool = True

    primary_timeframe: str = "15m"
    confirmation_timeframe: str = "1h"

    bb_period: int = 20
    bb_mult: float = 2.0
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    range_lookback_bars: int = 48
    range_proximity_pips: float = 4.0
    min_band_width_pips: float = 8.0
    max_band_width_pips: float = 45.0

    htf_ema_period: int = 50
    htf_slope_bars: int = 3
    allow_neutral_htf: bool = True

    max_current_range_pips: float = 16.0
    min_confidence: float = 0.52
    max_confidence: float = 0.84

    fixed_stop_loss_pips: float = 8.0
    fixed_take_profit_pips: float = 12.0
    signal_cooldown_minutes: int = 45

    london_start_hour_utc: int = 8
    new_york_end_hour_utc: int = 18

    pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _override(self, epic: str, key: str, default: Any) -> Any:
        row = self.pair_overrides.get(epic, {})
        if key in row and row[key] is not None:
            return row[key]
        bag = row.get("parameter_overrides") or {}
        if isinstance(bag, dict) and key in bag and bag[key] is not None:
            return bag[key]
        return default

    def is_pair_enabled(self, epic: str) -> bool:
        if self.enabled_pairs and epic not in self.enabled_pairs:
            return False
        row = self.pair_overrides.get(epic)
        if row is None:
            return epic in SUPPORTED_EPICS
        return bool(row.get("is_enabled", True))

    def is_pair_monitor_only(self, epic: str) -> bool:
        row = self.pair_overrides.get(epic, {})
        return bool(row.get("monitor_only", self.monitor_only))

    def get_pair_signal_cooldown_minutes(self, epic: str) -> int:
        return int(self._override(epic, "signal_cooldown_minutes", self.signal_cooldown_minutes))

    def is_session_allowed(self, hour_utc: int) -> bool:
        return self.london_start_hour_utc <= hour_utc <= self.new_york_end_hour_utc


def build_range_fade_config(profile: Optional[str] = None) -> EURUSDRangeFadeConfig:
    normalized = str(profile or "15m").strip().lower()
    if normalized in {"15m", "default", "base"}:
        return EURUSDRangeFadeConfig()
    if normalized in {"5m", "fast"}:
        return EURUSDRangeFadeConfig(
            strategy_name="RANGE_FADE_5M",
            profile_name="5m",
            primary_timeframe="5m",
            confirmation_timeframe="1h",
            rsi_oversold=32,
            rsi_overbought=68,
            range_lookback_bars=144,
            range_proximity_pips=3.0,
            min_band_width_pips=6.0,
            max_band_width_pips=28.0,
            allow_neutral_htf=False,
            max_current_range_pips=12.0,
            fixed_stop_loss_pips=8.0,
            fixed_take_profit_pips=12.0,
            signal_cooldown_minutes=30,
            london_start_hour_utc=6,
            new_york_end_hour_utc=18,
        )
    raise ValueError(f"Unsupported EURUSD range-fade profile: {profile}")


def _coerce_value(current: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(current, bool):
        return str(value).strip().lower() in _TRUE_STR
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value)
    return value


class EURUSDRangeFadeConfigService:
    _instance: Optional["EURUSDRangeFadeConfigService"] = None

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
        self._cached: Dict[str, EURUSDRangeFadeConfig] = {}
        self._cache_ts: Dict[str, datetime] = {}
        self._db_url = os.getenv(
            "STRATEGY_CONFIG_DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/strategy_config",
        )
        self._config_set = os.getenv("TRADING_CONFIG_SET", "demo")

    @classmethod
    def get_instance(cls) -> "EURUSDRangeFadeConfigService":
        return cls()

    def _cache_key(self, profile: str) -> str:
        return f"{self._config_set}:{profile}"

    def get_config(self, profile: Optional[str] = None) -> EURUSDRangeFadeConfig:
        normalized = str(profile or "15m").strip().lower()
        key = self._cache_key(normalized)
        now = datetime.now()
        with self._lock:
            if key in self._cached and key in self._cache_ts:
                if now - self._cache_ts[key] <= self._cache_ttl:
                    return copy.deepcopy(self._cached[key])
            cfg = self._load_from_database(normalized)
            self._cached[key] = cfg
            self._cache_ts[key] = now
            return copy.deepcopy(cfg)

    def refresh(self, profile: Optional[str] = None) -> EURUSDRangeFadeConfig:
        normalized = str(profile or "15m").strip().lower()
        key = self._cache_key(normalized)
        with self._lock:
            self._cached.pop(key, None)
            self._cache_ts.pop(key, None)
        return self.get_config(normalized)

    def _load_from_database(self, profile: str) -> EURUSDRangeFadeConfig:
        config = build_range_fade_config(profile)
        try:
            conn = psycopg2.connect(self._db_url)
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute(
                    """
                    SELECT * FROM eurusd_range_fade_global_config
                    WHERE is_active = TRUE AND profile_name = %s AND config_set = %s
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (config.profile_name, self._config_set),
                )
                row = cur.fetchone()
                if row:
                    for key in row.keys():
                        if not hasattr(config, key):
                            continue
                        current = getattr(config, key)
                        source_value = _normalize_strategy_name(row[key]) if key == "strategy_name" else row[key]
                        coerced = _coerce_value(current, source_value)
                        if coerced is not None:
                            setattr(config, key, coerced)

                cur.execute(
                    """
                    SELECT * FROM eurusd_range_fade_pair_overrides
                    WHERE profile_name = %s AND config_set = %s
                    """,
                    (config.profile_name, self._config_set),
                )
                pair_overrides: Dict[str, Dict[str, Any]] = {}
                for row in cur.fetchall():
                    pair_overrides[row["epic"]] = dict(row)
                config.pair_overrides = pair_overrides
                return config
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(
                "RANGE_FADE config DB load failed for profile=%s, using defaults: %s",
                profile,
                exc,
            )
            return config


def get_range_fade_config(profile: Optional[str] = None) -> EURUSDRangeFadeConfig:
    return EURUSDRangeFadeConfigService.get_instance().get_config(profile)


def build_eurusd_range_fade_config(profile: Optional[str] = None) -> EURUSDRangeFadeConfig:
    return build_range_fade_config(profile)


def get_eurusd_range_fade_config(profile: Optional[str] = None) -> EURUSDRangeFadeConfig:
    return get_range_fade_config(profile)
