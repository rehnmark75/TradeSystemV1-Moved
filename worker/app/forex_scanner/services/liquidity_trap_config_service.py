"""
LiquidityTrap config service — DB-backed with 5-minute TTL cache.

Column-per-parameter pattern (same as ImpulseFadeConfigService).
Global config from liquidity_trap_global_config; per-pair overrides from
liquidity_trap_pair_overrides (direct column takes precedence over JSONB).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras


@dataclass
class LiquidityTrapConfig:
    """All parameters settable via DB with per-pair override support."""

    strategy_name: str = "LIQUIDITY_TRAP"
    version: str = "1.0.0"

    # Detection params
    trigger_timeframe: str = "1h"
    daily_lookback_days: int = 1
    breakout_buffer_pips: float = 1.0
    wick_fraction_min: float = 0.35
    require_opposing_body: bool = True

    # ATR guard
    atr_period: int = 14
    min_atr_pips: float = 8.0
    max_atr_pips: float = 30.0

    # Risk
    fixed_stop_loss_pips: float = 14.0
    fixed_take_profit_pips: float = 20.0
    sl_buffer_pips: float = 3.0
    min_sl_pips: float = 8.0
    max_sl_pips: float = 22.0
    min_rr_ratio: float = 1.5
    max_tp_pips: float = 35.0

    # Session
    enabled_hours_default: Any = field(default_factory=lambda: list(range(7, 21)))  # 7-20 inclusive

    # Confidence
    min_confidence: float = 0.55
    max_confidence: float = 0.85

    # Cooldown
    signal_cooldown_minutes: int = 90

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

    def get_pair_min_atr(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "min_atr_pips", self.min_atr_pips))

    def get_pair_max_atr(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "max_atr_pips", self.max_atr_pips))

    def get_pair_enabled_hours(self, epic: str) -> List[int]:
        """Return the list of enabled UTC hours for this pair.

        Falls back to the global enabled_hours_default (a JSONB list).
        Handles both the case where psycopg2 returns a Python list directly
        and the case where it returns a JSON string.
        """
        raw = self.get_for_pair(epic, "enabled_hours", None)
        if raw is None:
            raw = self.enabled_hours_default
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = list(range(7, 21))
        if isinstance(raw, list):
            return [int(h) for h in raw]
        return list(range(7, 21))

    # ------------------------------------------------------------------
    # DB loader
    # ------------------------------------------------------------------

    @classmethod
    def from_database(cls, database_url: Optional[str] = None) -> "LiquidityTrapConfig":
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
                SELECT * FROM liquidity_trap_global_config
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
                "SELECT * FROM liquidity_trap_pair_overrides WHERE config_id = %s",
                (config_id,),
            )
            config._pair_overrides = {r["epic"]: dict(r) for r in cur.fetchall()}

            cur.close()
            conn.close()

            logging.info(
                f"[LIQUIDITY_TRAP] Loaded config v{config.version}: "
                f"trigger_tf={config.trigger_timeframe} "
                f"breakout_buf={config.breakout_buffer_pips}p wick_min={config.wick_fraction_min} "
                f"SL/TP={config.fixed_stop_loss_pips}/{config.fixed_take_profit_pips} "
                f"{len(config._pair_overrides)} pair overrides"
            )
        except Exception as e:
            logging.warning(f"[LIQUIDITY_TRAP] Using defaults; DB load failed: {e}")

        return config


class LiquidityTrapConfigService:
    _instance: Optional["LiquidityTrapConfigService"] = None

    def __init__(self) -> None:
        self._config: Optional[LiquidityTrapConfig] = None
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl_seconds = 300

    @classmethod
    def get_instance(cls) -> "LiquidityTrapConfigService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> LiquidityTrapConfig:
        now = datetime.now()
        if (self._config is None
                or self._last_refresh is None
                or (now - self._last_refresh).total_seconds() > self._cache_ttl_seconds):
            self._config = LiquidityTrapConfig.from_database()
            self._last_refresh = now
        return self._config

    def refresh(self) -> LiquidityTrapConfig:
        self._config = None
        return self.get_config()


def get_liquidity_trap_config() -> LiquidityTrapConfig:
    return LiquidityTrapConfigService.get_instance().get_config()
