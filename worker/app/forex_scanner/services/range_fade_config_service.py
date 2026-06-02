#!/usr/bin/env python3
"""
RANGE_FADE Config Service — DB-backed configuration with per-profile cache.

Source of truth: `strategy_config` database
    - range_fade_global_config
    - range_fade_pair_overrides

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
    return name


@dataclass
class RangeFadeConfig:
    strategy_name: str = "RANGE_FADE"
    profile_name: str = "5m"
    version: str = "0.3.0"

    enabled_pairs: List[str] = field(default_factory=lambda: list(SUPPORTED_EPICS))
    monitor_only: bool = True

    primary_timeframe: str = "5m"
    confirmation_timeframe: str = "1h"

    bb_period: int = 20
    bb_mult: float = 2.0
    rsi_period: int = 14
    rsi_oversold: int = 32
    rsi_overbought: int = 68

    range_lookback_bars: int = 144
    range_proximity_pips: float = 3.0
    min_band_width_pips: float = 6.0
    max_band_width_pips: float = 28.0

    htf_ema_period: int = 50
    htf_slope_bars: int = 3
    allow_neutral_htf: bool = False
    allowed_directions: str = ""  # comma-separated BUY/SELL list; blank allows both

    max_current_range_pips: float = 12.0
    adx_ceiling: float = 25.0
    buy_adx_ceiling: Optional[float] = None
    sell_adx_ceiling: Optional[float] = None
    htf_adx_ceiling: float = 999.0
    # NY/overlap-session-scoped 1h-ADX ceiling. Unlike htf_adx_ceiling (all sessions), this
    # fires ONLY inside the NY window (is_in_ny_session) where a strong 1h trend breaks ranges
    # and the fade has negative edge; outside NY the same condition is net-profitable so it is
    # deliberately NOT gated there. 999 = off (enable per config_set via DB, e.g. demo=35).
    ny_session_htf_adx_ceiling: float = 999.0
    er_floor: float = 0.0
    er_ceiling: float = 999.0
    buy_er_ceiling: Optional[float] = None
    sell_er_ceiling: Optional[float] = None
    er_period: int = 14
    min_macd_histogram_pips: float = 0.0
    min_confidence: float = 0.52
    max_confidence: float = 0.84
    # Confidence REJECT floor (distinct from min_confidence above, which is only a
    # SCALING anchor). Signals whose computed confidence < this are rejected at the
    # below_confidence_floor gate. The edge concentrates at conf>=0.60 (shipped in
    # fa97512); the 0.55-0.59 band is a net loser. DB-tunable per config_set and
    # per-pair (range_fade_pair_overrides.parameter_overrides). 0.0 = gate off.
    min_reject_confidence: float = 0.60

    fixed_stop_loss_pips: float = 8.0
    fixed_take_profit_pips: float = 12.0
    dynamic_sl_tp_enabled: bool = False
    dynamic_sl_band_width_sl_mult: float = 0.55
    dynamic_sl_band_width_tp_mult: float = 0.85
    dynamic_sl_min_pips: float = 5.0
    dynamic_sl_max_pips: float = 9.0
    dynamic_tp_min_pips: float = 8.0
    dynamic_tp_max_pips: float = 15.0
    signal_cooldown_minutes: int = 30

    london_start_hour_utc: int = 6
    new_york_end_hour_utc: int = 18
    ny_session_start_hour_utc: int = 15  # NY/overlap window for ny_session_htf_adx_ceiling
    ny_session_end_hour_utc: int = 20
    blocked_hours_utc: str = ""  # comma-separated hours to block, e.g. "7,8,15,16,18"
    buy_blocked_hours_utc: str = ""
    sell_blocked_hours_utc: str = ""
    buy_start_hour_utc: Optional[int] = None
    buy_end_hour_utc: Optional[int] = None
    sell_start_hour_utc: Optional[int] = None
    sell_end_hour_utc: Optional[int] = None
    buy_allowed_hours_utc: str = ""
    sell_allowed_hours_utc: str = ""
    buy_allowed_htf_biases: str = ""
    sell_allowed_htf_biases: str = ""
    post_loss_session_block_enabled: bool = False

    pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    backtest_overrides: Dict[str, Any] = field(default_factory=dict)

    def _override(self, epic: str, key: str, default: Any) -> Any:
        if key in self.backtest_overrides and self.backtest_overrides[key] is not None:
            return self.backtest_overrides[key]
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

    def get_pair_same_direction_cooldown_minutes(self, epic: str) -> int:
        """Cooldown applied to the same direction after a signal fires.
        Defaults to signal_cooldown_minutes so existing pairs are unaffected."""
        return int(self._override(epic, "same_direction_cooldown_minutes",
                                  self.get_pair_signal_cooldown_minutes(epic)))

    def get_pair_fixed_stop_loss_pips(self, epic: str) -> float:
        return float(self._override(epic, "fixed_stop_loss_pips", self.fixed_stop_loss_pips))

    def get_pair_fixed_take_profit_pips(self, epic: str) -> float:
        return float(self._override(epic, "fixed_take_profit_pips", self.fixed_take_profit_pips))

    def get_pair_rsi_oversold(self, epic: str, direction: str = "") -> int:
        if direction:
            direction_val = self._override(epic, f"{direction.lower()}_rsi_oversold", None)
            if direction_val is not None:
                return int(direction_val)
        return int(self._override(epic, "rsi_oversold", self.rsi_oversold))

    def get_pair_rsi_overbought(self, epic: str, direction: str = "") -> int:
        if direction:
            direction_val = self._override(epic, f"{direction.lower()}_rsi_overbought", None)
            if direction_val is not None:
                return int(direction_val)
        return int(self._override(epic, "rsi_overbought", self.rsi_overbought))

    def get_pair_bb_mult(self, epic: str) -> float:
        return float(self._override(epic, "bb_mult", self.bb_mult))

    def get_pair_range_lookback_bars(self, epic: str) -> int:
        return int(self._override(epic, "range_lookback_bars", self.range_lookback_bars))

    def get_pair_range_proximity_pips(self, epic: str) -> float:
        return float(self._override(epic, "range_proximity_pips", self.range_proximity_pips))

    def get_pair_min_band_width_pips(self, epic: str) -> float:
        return float(self._override(epic, "min_band_width_pips", self.min_band_width_pips))

    def get_pair_max_band_width_pips(self, epic: str) -> float:
        return float(self._override(epic, "max_band_width_pips", self.max_band_width_pips))

    def get_pair_max_current_range_pips(self, epic: str) -> float:
        return float(self._override(epic, "max_current_range_pips", self.max_current_range_pips))

    def get_pair_min_macd_histogram_pips(self, epic: str) -> float:
        return float(self._override(epic, "min_macd_histogram_pips", self.min_macd_histogram_pips))

    def get_pair_htf_adx_ceiling(self, epic: str) -> float:
        return float(self._override(epic, "htf_adx_ceiling", self.htf_adx_ceiling))

    def get_pair_ny_session_htf_adx_ceiling(self, epic: str) -> float:
        return float(self._override(epic, "ny_session_htf_adx_ceiling", self.ny_session_htf_adx_ceiling))

    def is_in_ny_session(self, hour_utc: int, epic: str = "") -> bool:
        start_hour = int(self._override(epic, "ny_session_start_hour_utc", self.ny_session_start_hour_utc))
        end_hour = int(self._override(epic, "ny_session_end_hour_utc", self.ny_session_end_hour_utc))
        return start_hour <= hour_utc <= end_hour

    def get_pair_adx_ceiling(self, epic: str, direction: str) -> float:
        direction_key = f"{direction.lower()}_adx_ceiling"
        direction_value = self._override(epic, direction_key, getattr(self, direction_key, None))
        if direction_value not in (None, ""):
            return float(direction_value)
        return float(self._override(epic, "adx_ceiling", self.adx_ceiling))

    def get_pair_er_floor(self, epic: str) -> float:
        return float(self._override(epic, "er_floor", self.er_floor))

    def get_pair_min_reject_confidence(self, epic: str) -> float:
        return float(self._override(epic, "min_reject_confidence", self.min_reject_confidence))

    def get_pair_er_ceiling(self, epic: str, direction: str) -> float:
        direction_key = f"{direction.lower()}_er_ceiling"
        direction_value = self._override(epic, direction_key, getattr(self, direction_key, None))
        if direction_value not in (None, ""):
            return float(direction_value)
        return float(self._override(epic, "er_ceiling", self.er_ceiling))

    def get_pair_er_period(self, epic: str) -> int:
        return int(self._override(epic, "er_period", self.er_period))

    def get_pair_blocked_hours(self, epic: str, direction: str = "") -> set:
        key = f"{direction.lower()}_blocked_hours_utc" if direction else "blocked_hours_utc"
        default = getattr(self, key, self.blocked_hours_utc)
        raw = self._override(epic, key, default)
        if not raw:
            return set()
        return {int(h.strip()) for h in str(raw).split(",") if h.strip().isdigit()}

    def is_direction_allowed(self, epic: str, direction: str) -> bool:
        raw = self._override(epic, "allowed_directions", self.allowed_directions)
        if not raw:
            return True
        allowed = {part.strip().upper() for part in str(raw).split(",") if part.strip()}
        return direction.upper() in allowed

    def is_session_allowed(self, hour_utc: int, epic: str = "") -> bool:
        start_hour = int(self._override(epic, "london_start_hour_utc", self.london_start_hour_utc))
        end_hour = int(self._override(epic, "new_york_end_hour_utc", self.new_york_end_hour_utc))
        if not (start_hour <= hour_utc <= end_hour):
            return False
        if epic and hour_utc in self.get_pair_blocked_hours(epic):
            return False
        return True

    def is_direction_session_allowed(self, hour_utc: int, epic: str, direction: str) -> bool:
        direction_lower = direction.lower()
        allowed_hours_raw = self._override(epic, f"{direction_lower}_allowed_hours_utc", "")
        if allowed_hours_raw:
            allowed_hours = {int(h.strip()) for h in str(allowed_hours_raw).split(",") if h.strip().isdigit()}
            return hour_utc in allowed_hours
        start_default = self._override(epic, "london_start_hour_utc", self.london_start_hour_utc)
        end_default = self._override(epic, "new_york_end_hour_utc", self.new_york_end_hour_utc)
        start_hour = int(self._override(epic, f"{direction_lower}_start_hour_utc", start_default))
        end_hour = int(self._override(epic, f"{direction_lower}_end_hour_utc", end_default))
        if not (start_hour <= hour_utc <= end_hour):
            return False
        blocked = self.get_pair_blocked_hours(epic) | self.get_pair_blocked_hours(epic, direction)
        return hour_utc not in blocked

    def is_post_loss_session_block_enabled(self, epic: str) -> bool:
        raw = self._override(epic, "post_loss_session_block_enabled", self.post_loss_session_block_enabled)
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in ("true", "1", "yes")

    def is_htf_bias_allowed(self, epic: str, direction: str, htf_bias: str) -> bool:
        default = "bullish,neutral" if direction.upper() == "BUY" else "bearish,neutral"
        raw = self._override(epic, f"{direction.lower()}_allowed_htf_biases", default)
        allowed = {part.strip().lower() for part in str(raw).split(",") if part.strip()}
        return htf_bias.lower() in allowed


def build_range_fade_config(profile: Optional[str] = None) -> RangeFadeConfig:
    normalized = str(profile or "5m").strip().lower()
    if normalized in {"5m", "fast", "default", "base"}:
        return RangeFadeConfig()
    raise ValueError(f"Unsupported range-fade profile: {profile} (only 5m is supported)")


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


class RangeFadeConfigService:
    _instance: Optional["RangeFadeConfigService"] = None

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
        self._cached: Dict[str, RangeFadeConfig] = {}
        self._cache_ts: Dict[str, datetime] = {}
        self._db_url = os.getenv(
            "STRATEGY_CONFIG_DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/strategy_config",
        )
        self._config_set = os.getenv("TRADING_CONFIG_SET", "demo")

    @classmethod
    def get_instance(cls) -> "RangeFadeConfigService":
        return cls()

    def _cache_key(self, profile: str) -> str:
        return f"{self._config_set}:{profile}"

    def get_config(self, profile: Optional[str] = None) -> RangeFadeConfig:
        normalized = str(profile or "5m").strip().lower()
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

    def refresh(self, profile: Optional[str] = None) -> RangeFadeConfig:
        normalized = str(profile or "5m").strip().lower()
        key = self._cache_key(normalized)
        with self._lock:
            self._cached.pop(key, None)
            self._cache_ts.pop(key, None)
        return self.get_config(normalized)

    def _load_from_database(self, profile: str) -> RangeFadeConfig:
        config = build_range_fade_config(profile)
        try:
            conn = psycopg2.connect(self._db_url)
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute(
                    """
                    SELECT * FROM range_fade_global_config
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
                    SELECT * FROM range_fade_pair_overrides
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


def get_range_fade_config(profile: Optional[str] = None) -> RangeFadeConfig:
    return RangeFadeConfigService.get_instance().get_config(profile)


def build_eurusd_range_fade_config(profile: Optional[str] = None) -> RangeFadeConfig:
    return build_range_fade_config(profile)


def get_eurusd_range_fade_config(profile: Optional[str] = None) -> RangeFadeConfig:
    return get_range_fade_config(profile)


# Back-compat aliases (deprecated — to be removed in a future release)
EURUSDRangeFadeConfig = RangeFadeConfig
EURUSDRangeFadeConfigService = RangeFadeConfigService
