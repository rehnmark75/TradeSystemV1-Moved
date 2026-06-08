"""
Squeeze Momentum config service.

DB-backed when squeeze_momentum_* tables exist, with monitor-only default
pair rows so the strategy remains backtestable before the migration is applied.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.extras


_DEFAULT_PAIRS: Dict[str, Dict[str, Any]] = {
    "CS.D.EURUSD.CEEM.IP": {"pair_name": "EURUSD", "is_enabled": True, "monitor_only": True},
    "CS.D.GBPUSD.MINI.IP": {"pair_name": "GBPUSD", "is_enabled": True, "monitor_only": True},
    "CS.D.AUDUSD.MINI.IP": {"pair_name": "AUDUSD", "is_enabled": True, "monitor_only": True},
    "CS.D.NZDUSD.MINI.IP": {"pair_name": "NZDUSD", "is_enabled": True, "monitor_only": True},
    "CS.D.USDCAD.MINI.IP": {"pair_name": "USDCAD", "is_enabled": True, "monitor_only": True},
    "CS.D.USDCHF.MINI.IP": {"pair_name": "USDCHF", "is_enabled": True, "monitor_only": True},
    "CS.D.USDJPY.MINI.IP": {"pair_name": "USDJPY", "is_enabled": True, "monitor_only": True},
    "CS.D.EURJPY.MINI.IP": {"pair_name": "EURJPY", "is_enabled": True, "monitor_only": True},
    "CS.D.AUDJPY.MINI.IP": {"pair_name": "AUDJPY", "is_enabled": True, "monitor_only": True},
}


@dataclass
class SqueezeMomentumConfig:
    strategy_name: str = "SQUEEZE_MOMENTUM"
    version: str = "1.0.0"

    entry_timeframe: str = "15m"
    htf_timeframe: str = "1h"
    bb_length: int = 20
    bb_mult: float = 2.0
    kc_length: int = 20
    kc_mult: float = 1.5
    use_true_range: bool = True
    htf_ema_period: int = 50
    adx_period: int = 14
    adx_min: float = 18.0
    require_adx_rising: bool = True
    squeeze_min_bars: int = 3
    squeeze_lookback_bars: int = 8
    require_release_bar: bool = True
    min_momentum_slope_atr: float = 0.03
    atr_period: int = 14
    stop_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.5
    min_confidence: float = 0.55
    max_confidence: float = 0.88
    signal_cooldown_minutes: int = 180
    block_asian_session: bool = True

    _pair_overrides: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {epic: dict(row) for epic, row in _DEFAULT_PAIRS.items()}
    )

    def get_for_pair(self, epic: str, param_name: str, default: Any = None) -> Any:
        row = self._pair_overrides.get(epic)
        if row:
            if param_name in row and row[param_name] is not None:
                return row[param_name]
            jsonb = row.get("parameter_overrides") or {}
            if param_name in jsonb and jsonb[param_name] is not None:
                return jsonb[param_name]
        if hasattr(self, param_name):
            return getattr(self, param_name)
        return default

    def is_pair_enabled(self, epic: str) -> bool:
        row = self._pair_overrides.get(epic)
        return bool(row.get("is_enabled", False)) if row else False

    def is_pair_monitor_only(self, epic: str) -> bool:
        return bool(self.get_for_pair(epic, "monitor_only", True))

    def get_pair_cooldown_minutes(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "signal_cooldown_minutes", self.signal_cooldown_minutes))

    @classmethod
    def from_database(cls, database_url: Optional[str] = None) -> "SqueezeMomentumConfig":
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
                SELECT * FROM squeeze_momentum_global_config
                 WHERE is_active = TRUE
                 ORDER BY id DESC LIMIT 1
                """
            )
            row = cur.fetchone()
            if row:
                for key in row.keys():
                    if not hasattr(config, key):
                        continue
                    value = row[key]
                    if value is None:
                        continue
                    current = getattr(config, key)
                    if isinstance(current, bool):
                        value = bool(value)
                    elif isinstance(current, int) and not isinstance(current, bool):
                        value = int(value)
                    elif isinstance(current, float):
                        value = float(value)
                    setattr(config, key, value)

            cur.execute("SELECT * FROM squeeze_momentum_pair_overrides")
            rows = cur.fetchall()
            if rows:
                config._pair_overrides = {r["epic"]: dict(r) for r in rows}

            cur.close()
            conn.close()
            logging.info(
                "[SQUEEZE_MOMENTUM] Loaded config v%s: %s/%s squeeze, BB=%s %.2f, KC=%s %.2f, %s pair rows",
                config.version,
                config.squeeze_min_bars,
                config.squeeze_lookback_bars,
                config.bb_length,
                config.bb_mult,
                config.kc_length,
                config.kc_mult,
                len(config._pair_overrides),
            )
        except Exception as exc:
            logging.warning("[SQUEEZE_MOMENTUM] Using defaults; DB load failed: %s", exc)

        return config


class SqueezeMomentumConfigService:
    _instance: Optional["SqueezeMomentumConfigService"] = None

    def __init__(self) -> None:
        self._config: Optional[SqueezeMomentumConfig] = None
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl_seconds = 300

    @classmethod
    def get_instance(cls) -> "SqueezeMomentumConfigService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> SqueezeMomentumConfig:
        now = datetime.now()
        if (
            self._config is None
            or self._last_refresh is None
            or (now - self._last_refresh).total_seconds() > self._cache_ttl_seconds
        ):
            self._config = SqueezeMomentumConfig.from_database()
            self._last_refresh = now
        return self._config

    def refresh(self) -> SqueezeMomentumConfig:
        self._config = None
        return self.get_config()


def get_squeeze_momentum_config() -> SqueezeMomentumConfig:
    return SqueezeMomentumConfigService.get_instance().get_config()
