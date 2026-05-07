"""
ImpulseFade config service — DB-backed with 5-minute TTL cache.

Column-per-parameter pattern (same as MeanReversionConfigService).
Global config from impulse_fade_global_config; per-pair overrides from
impulse_fade_pair_overrides (direct column takes precedence over JSONB).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.extras


@dataclass
class ImpulseFadeConfig:
    """All parameters settable via DB with per-pair override support."""

    strategy_name: str = "IMPULSE_FADE"
    version: str = "1.0.0"

    # Session window (UTC hours, inclusive on both ends: 18 = 18:00-18:59)
    session_start_hour: int = 18
    session_end_hour: int = 22

    # Optional second session window — None means disabled
    session_start_hour_2: Optional[int] = None
    session_end_hour_2: Optional[int] = None

    # ATR body threshold
    atr_body_multiplier: float = 2.2
    atr_period: int = 14
    max_atr_pips: float = 15.0

    # Risk
    fixed_stop_loss_pips: float = 12.0
    fixed_take_profit_pips: float = 8.0
    time_stop_candles: int = 36

    # Confidence
    min_confidence: float = 0.58
    max_confidence: float = 0.85

    # Cooldown
    signal_cooldown_minutes: int = 60

    # Populated from DB
    _pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Per-pair getters — direct column → JSONB → global
    # ------------------------------------------------------------------

    def get_for_pair(self, epic: str, param_name: str, default: Any = None) -> Any:
        if epic in self._pair_overrides:
            row = self._pair_overrides[epic]
            if param_name in row and row[param_name] is not None:
                return row[param_name]
            jsonb = row.get("parameter_overrides") or {}
            if param_name in jsonb and jsonb[param_name] is not None:
                return jsonb[param_name]
        if hasattr(self, param_name):
            return getattr(self, param_name)
        return default

    def is_pair_enabled(self, epic: str) -> bool:
        if epic in self._pair_overrides:
            return bool(self._pair_overrides[epic].get("is_enabled", False))
        return False

    def is_pair_monitor_only(self, epic: str) -> bool:
        return bool(self.get_for_pair(epic, "monitor_only", True))

    def get_pair_atr_body_multiplier(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "atr_body_multiplier", self.atr_body_multiplier))

    def get_pair_max_atr_pips(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "max_atr_pips", self.max_atr_pips))

    def get_pair_fixed_stop_loss(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "fixed_stop_loss_pips", self.fixed_stop_loss_pips))

    def get_pair_fixed_take_profit(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "fixed_take_profit_pips", self.fixed_take_profit_pips))

    def get_pair_min_confidence(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "min_confidence", self.min_confidence))

    def get_pair_max_confidence(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "max_confidence", self.max_confidence))

    def get_pair_cooldown_minutes(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "signal_cooldown_minutes", self.signal_cooldown_minutes))

    def get_pair_session_start_hour(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "session_start_hour", self.session_start_hour))

    def get_pair_session_end_hour(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "session_end_hour", self.session_end_hour))

    def get_pair_session_start_hour_2(self, epic: str) -> Optional[int]:
        v = self.get_for_pair(epic, "session_start_hour_2", self.session_start_hour_2)
        return int(v) if v is not None else None

    def get_pair_session_end_hour_2(self, epic: str) -> Optional[int]:
        v = self.get_for_pair(epic, "session_end_hour_2", self.session_end_hour_2)
        return int(v) if v is not None else None

    # ------------------------------------------------------------------
    # DB loader
    # ------------------------------------------------------------------

    @classmethod
    def from_database(cls, database_url: Optional[str] = None) -> "ImpulseFadeConfig":
        config = cls()

        if database_url is None:
            database_url = os.getenv(
                "STRATEGY_CONFIG_DATABASE_URL",
                "postgresql://postgres:postgres@postgres:5432/strategy_config",
            )

        try:
            conn = psycopg2.connect(database_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute(
                """
                SELECT * FROM impulse_fade_global_config
                 WHERE is_active = TRUE
                 ORDER BY id DESC LIMIT 1
                """
            )
            row = cur.fetchone()
            if row:
                for key in row.keys():
                    if not hasattr(config, key):
                        continue
                    v = row[key]
                    if v is None:
                        continue
                    default_val = getattr(cls, key, None)
                    if isinstance(default_val, bool):
                        v = bool(v)
                    elif isinstance(default_val, int):
                        v = int(v)
                    elif isinstance(default_val, float):
                        v = float(v)
                    setattr(config, key, v)

            trading_env = os.getenv("TRADING_ENVIRONMENT", "demo")
            config_id = 2 if trading_env == "live" else 3
            cur.execute(
                "SELECT * FROM impulse_fade_pair_overrides WHERE config_id = %s",
                (config_id,),
            )
            config._pair_overrides = {r["epic"]: dict(r) for r in cur.fetchall()}

            cur.close()
            conn.close()

            logging.info(
                f"[IMPULSE_FADE] Loaded config v{config.version}: "
                f"session={config.session_start_hour}-{config.session_end_hour}UTC "
                f"atr_mult={config.atr_body_multiplier} max_atr={config.max_atr_pips} "
                f"SL/TP={config.fixed_stop_loss_pips}/{config.fixed_take_profit_pips} "
                f"{len(config._pair_overrides)} pair overrides"
            )
        except Exception as e:
            logging.warning(f"[IMPULSE_FADE] Using defaults; DB load failed: {e}")

        return config


class ImpulseFadeConfigService:
    _instance: Optional["ImpulseFadeConfigService"] = None

    def __init__(self) -> None:
        self._config: Optional[ImpulseFadeConfig] = None
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl_seconds = 300

    @classmethod
    def get_instance(cls) -> "ImpulseFadeConfigService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> ImpulseFadeConfig:
        now = datetime.now()
        if (self._config is None
                or self._last_refresh is None
                or (now - self._last_refresh).total_seconds() > self._cache_ttl_seconds):
            self._config = ImpulseFadeConfig.from_database()
            self._last_refresh = now
        return self._config

    def refresh(self) -> ImpulseFadeConfig:
        self._config = None
        return self.get_config()


def get_impulse_fade_config() -> ImpulseFadeConfig:
    return ImpulseFadeConfigService.get_instance().get_config()
