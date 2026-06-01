"""INSIDE_DAY configuration service.

Reads per-pair enablement and monitor/trading posture from
`inside_day_pair_overrides`. Core strategy rules stay code-owned; the database
decides whether each validated pair can emit signals and whether those signals
are monitor-only.
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


class InsideDayConfigService:
    """Thread-safe, cached reader for inside_day_pair_overrides."""

    def __init__(self, db_url: str = _STRATEGY_CONFIG_URL, cache_ttl_seconds: int = 120):
        self._db_url = db_url
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._lock = RLock()
        self._pair_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Optional[datetime] = None

    def is_pair_enabled(self, epic: str, config_set: Optional[str] = None) -> bool:
        row = self._get_pair_row(epic, config_set)
        return bool(row.get("is_enabled", False)) if row else False

    def is_monitor_only(self, epic: str, config_set: Optional[str] = None) -> bool:
        row = self._get_pair_row(epic, config_set)
        return bool(row.get("monitor_only", True)) if row else True

    def is_traded(self, epic: str, config_set: Optional[str] = None) -> bool:
        row = self._get_pair_row(epic, config_set)
        return bool(row.get("is_traded", False)) if row else False

    def invalidate(self) -> None:
        with self._lock:
            self._pair_cache.clear()
            self._cache_ts = None

    def _get_pair_row(self, epic: str, config_set: Optional[str] = None) -> Optional[Dict[str, Any]]:
        config_set = config_set or os.getenv("TRADING_CONFIG_SET", "demo")
        self._refresh_if_stale(config_set)
        return self._pair_cache.get(epic)

    def _refresh_if_stale(self, config_set: str) -> None:
        with self._lock:
            now = datetime.utcnow()
            if self._cache_ts and (now - self._cache_ts) < self._cache_ttl:
                return
            try:
                rows = self._load_from_db(config_set)
                self._pair_cache = {row["epic"]: dict(row) for row in rows}
                self._cache_ts = now
                logger.debug("[INSIDE_DAY_CFG] Loaded %d pair rows from DB", len(rows))
            except Exception as exc:
                logger.warning("[INSIDE_DAY_CFG] DB load failed (%s) - using last-known-good", exc)
                if self._cache_ts is None:
                    self._pair_cache = {}

    def _load_from_db(self, config_set: str) -> list:
        conn = psycopg2.connect(self._db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT epic, pair_name, is_enabled, is_traded, monitor_only,
                           weekly_bias_q, inside_day_min_pips, inside_day_max_pips,
                           atr_period, atr_buffer_fraction, reward_risk,
                           base_confidence, parameter_overrides
                    FROM inside_day_pair_overrides
                    WHERE config_set = %s
                    """,
                    (config_set,),
                )
                return cur.fetchall()
        finally:
            conn.close()


_service: Optional[InsideDayConfigService] = None
_service_lock = RLock()


def get_inside_day_config_service() -> InsideDayConfigService:
    global _service
    with _service_lock:
        if _service is None:
            _service = InsideDayConfigService()
        return _service
