"""
DonchianTurtle config service — DB-backed with 5-minute TTL cache.

Column-per-parameter pattern (same as ImpulseFadeConfigService).
Global config from donchian_turtle_global_config; per-pair overrides from
donchian_turtle_pair_overrides (direct column takes precedence over JSONB).
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
class DonchianTurtleConfig:
    """All parameters settable via DB with per-pair override support."""

    strategy_name: str = "DONCHIAN_TURTLE"
    version: str = "1.0.0"

    # Channel parameters (S1 system)
    entry_bars: int = 20
    exit_bars: int = 10
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0

    # Direction gate
    long_only: bool = True

    # Risk fallbacks
    fixed_stop_loss_pips: float = 50.0
    fixed_take_profit_pips: float = 200.0

    # Confidence
    min_confidence: float = 0.50
    max_confidence: float = 0.95

    # Cooldown
    signal_cooldown_minutes: int = 240

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

    def get_pair_entry_bars(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "entry_bars", self.entry_bars))

    def get_pair_exit_bars(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "exit_bars", self.exit_bars))

    def get_pair_atr_stop_multiplier(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "atr_stop_multiplier", self.atr_stop_multiplier))

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

    # ------------------------------------------------------------------
    # DB loader
    # ------------------------------------------------------------------

    @classmethod
    def from_database(cls, database_url: Optional[str] = None) -> "DonchianTurtleConfig":
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
                SELECT * FROM donchian_turtle_global_config
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

            cur.execute("SELECT * FROM donchian_turtle_pair_overrides")
            config._pair_overrides = {r["epic"]: dict(r) for r in cur.fetchall()}

            cur.close()
            conn.close()

            logging.info(
                f"[DONCHIAN_TURTLE] Loaded config v{config.version}: "
                f"entry={config.entry_bars}bar exit={config.exit_bars}bar "
                f"atr_stop={config.atr_stop_multiplier}x long_only={config.long_only} "
                f"{len(config._pair_overrides)} pair overrides"
            )
        except Exception as e:
            logging.warning(f"[DONCHIAN_TURTLE] Using defaults; DB load failed: {e}")

        return config


class DonchianTurtleConfigService:
    _instance: Optional["DonchianTurtleConfigService"] = None

    def __init__(self) -> None:
        self._config: Optional[DonchianTurtleConfig] = None
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl_seconds = 300

    @classmethod
    def get_instance(cls) -> "DonchianTurtleConfigService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> DonchianTurtleConfig:
        now = datetime.now()
        if (self._config is None
                or self._last_refresh is None
                or (now - self._last_refresh).total_seconds() > self._cache_ttl_seconds):
            self._config = DonchianTurtleConfig.from_database()
            self._last_refresh = now
        return self._config

    def refresh(self) -> DonchianTurtleConfig:
        self._config = None
        return self.get_config()


def get_donchian_turtle_config() -> DonchianTurtleConfig:
    return DonchianTurtleConfigService.get_instance().get_config()
