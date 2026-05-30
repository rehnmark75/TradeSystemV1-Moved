"""
KAMA V2 Configuration Service

Loads per-pair overrides from `kama_v2_pair_overrides` in the `strategy_config`
database.  Falls back to KamaV2Config global defaults for any unset column.
5-minute TTL cache with last-known-good fallback on DB error.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

_STRATEGY_CONFIG_URL = os.environ.get(
    "STRATEGY_CONFIG_DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/strategy_config",
)
_CACHE_TTL = timedelta(seconds=300)


class KamaV2ConfigService:
    """Thread-safe, cached reader for kama_v2_pair_overrides."""

    def __init__(self, db_url: str = _STRATEGY_CONFIG_URL, cache_ttl_seconds: int = 300):
        self._db_url = db_url
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._lock = RLock()
        # {epic: {column: value, ...}}
        self._pair_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_pair_enabled(self, epic: str, config_set: str = "demo") -> bool:
        row = self._get_pair_row(epic, config_set)
        return bool(row.get("is_enabled", False)) if row else False

    def is_monitor_only(self, epic: str, config_set: str = "demo") -> bool:
        row = self._get_pair_row(epic, config_set)
        return bool(row.get("monitor_only", True)) if row else True

    def get_pair_sl_pips(self, epic: str, default: float = 10.0) -> float:
        row = self._get_pair_row(epic)
        v = row.get("fixed_stop_loss_pips") if row else None
        return float(v) if v is not None else default

    def get_pair_tp_pips(self, epic: str, default: float = 15.0) -> float:
        row = self._get_pair_row(epic)
        v = row.get("fixed_take_profit_pips") if row else None
        return float(v) if v is not None else default

    def get_pair_cross_er_min(self, epic: str, default: float = 0.35) -> float:
        row = self._get_pair_row(epic)
        v = row.get("cross_er_min") if row else None
        return float(v) if v is not None else default

    def get_pair_adx_min(self, epic: str, default: float = 0.0) -> float:
        row = self._get_pair_row(epic)
        v = row.get("adx_min") if row else None
        return float(v) if v is not None else default

    def get_pair_session_filter(self, epic: str, default: bool = False) -> bool:
        row = self._get_pair_row(epic)
        v = row.get("session_filter") if row else None
        return bool(v) if v is not None else default

    def get_pair_blocked_hours(self, epic: str, default: str = "21,22,23,0,1,2,3") -> str:
        row = self._get_pair_row(epic)
        v = row.get("blocked_hours_utc") if row else None
        return str(v) if v is not None else default

    def get_pair_cooldown_minutes(self, epic: str, default: float = 30.0) -> float:
        row = self._get_pair_row(epic)
        v = row.get("signal_cooldown_minutes") if row else None
        return float(v) if v is not None else default

    def invalidate(self) -> None:
        with self._lock:
            self._pair_cache.clear()
            self._cache_ts = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_pair_row(self, epic: str, config_set: str = "demo") -> Optional[Dict[str, Any]]:
        self._refresh_if_stale(config_set)
        return self._pair_cache.get(epic)

    def _refresh_if_stale(self, config_set: str = "demo") -> None:
        with self._lock:
            now = datetime.utcnow()
            if self._cache_ts and (now - self._cache_ts) < self._cache_ttl:
                return
            try:
                rows = self._load_from_db(config_set)
                self._pair_cache = {r["epic"]: dict(r) for r in rows}
                self._cache_ts = now
                logger.debug("[KAMA_V2_CFG] Loaded %d pair rows from DB", len(rows))
            except Exception as exc:
                logger.warning("[KAMA_V2_CFG] DB load failed (%s) — using last-known-good", exc)
                if self._cache_ts is None:
                    self._pair_cache = {}

    def _load_from_db(self, config_set: str) -> list:
        conn = psycopg2.connect(self._db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT epic, is_enabled, is_traded, monitor_only,
                           fixed_stop_loss_pips, fixed_take_profit_pips,
                           cross_er_min, adx_min, session_filter,
                           blocked_hours_utc, signal_cooldown_minutes,
                           parameter_overrides
                    FROM kama_v2_pair_overrides
                    WHERE config_set = %s
                    """,
                    (config_set,),
                )
                return cur.fetchall()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: Optional[KamaV2ConfigService] = None
_service_lock = RLock()


def get_kama_v2_config_service() -> KamaV2ConfigService:
    global _service
    with _service_lock:
        if _service is None:
            _service = KamaV2ConfigService()
        return _service
