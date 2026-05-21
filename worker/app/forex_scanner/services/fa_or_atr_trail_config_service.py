"""FA_OR_ATR_TRAIL configuration service.

Loads the demo/live enablement and pair monitor state from strategy_config.
The strategy logic remains code-owned; the database controls whether a pair
can emit live signals and whether those signals are tradeable or monitor-only.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.extras


logger = logging.getLogger(__name__)


@dataclass
class FAORATRTrailConfig:
    strategy_name: str = "FA_OR_ATR_TRAIL"
    version: str = "0.1.0"
    is_active: bool = False
    config_set: str = "demo"
    fa_or_session_start_hour: int = 8
    fa_or_session_end_hour: int = 20
    fa_or_adx_min: float = 18.0
    fa_or_min_slope_pips: float = 0.3
    fa_or_max_vwap_atr: float = 3.0
    fa_or_opening_range_bars: int = 6
    fa_or_value_area_std: float = 0.70
    fa_or_vp_lookback: int = 15
    fa_or_cooldown_bars: int = 5
    fa_or_sl_atr: float = 1.2
    fa_or_tp_atr: float = 2.0
    fa_or_trail_trigger_atr: float = 0.25
    fa_or_trail_distance_atr: float = 0.10
    fa_or_atr_period: int = 14
    fa_or_adx_period: int = 14
    fa_or_rsi_period: int = 14
    fa_or_htf_ema_period: int = 50
    fa_or_usdjpy_atr_floor_pips: float = 8.7
    pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source: str = "default"
    loaded_at: datetime = field(default_factory=datetime.now)

    def _pair(self, epic: str) -> Dict[str, Any]:
        return self.pair_overrides.get(epic, {})

    def get_for_pair(self, epic: str, key: str, default: Any = None) -> Any:
        row = self._pair(epic)
        if key in row and row[key] is not None:
            return row[key]
        bag = row.get("parameter_overrides") or {}
        if isinstance(bag, str):
            try:
                bag = json.loads(bag)
            except Exception:
                bag = {}
        return bag.get(key, default)

    def is_pair_enabled(self, epic: str) -> bool:
        if not self.is_active:
            return False
        row = self._pair(epic)
        return bool(row.get("is_enabled", False)) if row else False

    def is_monitor_only(self, epic: str) -> bool:
        return bool(self.get_for_pair(epic, "monitor_only", True))

    def is_traded(self, epic: str) -> bool:
        return bool(self.get_for_pair(epic, "is_traded", False))

    @classmethod
    def from_database(cls, database_url: Optional[str] = None, config_set: Optional[str] = None) -> "FAORATRTrailConfig":
        config = cls(config_set=config_set or os.getenv("TRADING_CONFIG_SET", "demo"))
        database_url = database_url or os.getenv(
            "STRATEGY_CONFIG_DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/strategy_config",
        )

        try:
            conn = psycopg2.connect(database_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute(
                """
                SELECT parameter_name, parameter_value, value_type
                FROM fa_or_atr_trail_global_config
                WHERE config_set = %s AND is_active = TRUE
                """,
                (config.config_set,),
            )
            for row in cur.fetchall():
                key = row["parameter_name"]
                if not hasattr(config, key):
                    continue
                value = row["parameter_value"]
                value_type = (row["value_type"] or "string").lower()
                if value_type == "bool":
                    value = str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
                elif value_type == "int":
                    value = int(float(value))
                elif value_type == "float":
                    value = float(value)
                setattr(config, key, value)

            cur.execute(
                """
                SELECT *
                FROM fa_or_atr_trail_pair_overrides
                WHERE config_set = %s
                """,
                (config.config_set,),
            )
            config.pair_overrides = {row["epic"]: dict(row) for row in cur.fetchall()}
            config.source = "database"
            config.loaded_at = datetime.now()

            cur.close()
            conn.close()
            logger.info(
                "[FA_OR_ATR_TRAIL] Loaded %s config: active=%s, %d pair overrides",
                config.config_set,
                config.is_active,
                len(config.pair_overrides),
            )
        except Exception as exc:
            logger.warning("[FA_OR_ATR_TRAIL] Using default-disabled config; DB load failed: %s", exc)

        return config


class FAORATRTrailConfigService:
    _instance: Optional["FAORATRTrailConfigService"] = None
    _lock = RLock()

    def __init__(self, cache_ttl_seconds: int = 120):
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._config: Optional[FAORATRTrailConfig] = None
        self._loaded_at: Optional[datetime] = None

    @classmethod
    def get_instance(cls) -> "FAORATRTrailConfigService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_config(self, force_refresh: bool = False) -> FAORATRTrailConfig:
        with self._lock:
            stale = self._loaded_at is None or datetime.now() - self._loaded_at > self.cache_ttl
            if force_refresh or self._config is None or stale:
                self._config = FAORATRTrailConfig.from_database()
                self._loaded_at = datetime.now()
            return self._config
