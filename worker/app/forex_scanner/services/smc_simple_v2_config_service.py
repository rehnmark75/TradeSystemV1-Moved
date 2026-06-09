"""SMC_SIMPLE_V2 configuration service.

Loads per-pair monitoring rows from `smc_simple_v2_pair_overrides`.
The strategy rules remain code-owned; DB rows decide whether each pair can
emit monitor/trade signals and provide per-pair parameter overrides.
"""
from __future__ import annotations

import json
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


class SMCSimpleV2ConfigService:
    """Thread-safe, cached reader for SMC_SIMPLE_V2 pair overrides."""

    def __init__(self, db_url: str = _STRATEGY_CONFIG_URL, cache_ttl_seconds: int = 300):
        self._db_url = db_url
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._lock = RLock()
        self._pair_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Optional[datetime] = None
        self._loaded_config_set: Optional[str] = None

    def get_pair_config(self, epic: str, config_set: Optional[str] = None) -> Optional[Dict[str, Any]]:
        config_set = config_set or os.getenv("TRADING_CONFIG_SET", "demo")
        if config_set != "demo":
            return None
        self._refresh_if_stale(config_set)
        row = self._pair_cache.get(epic)
        return dict(row) if row else None

    def is_pair_enabled(self, epic: str, config_set: Optional[str] = None) -> bool:
        config_set = config_set or os.getenv("TRADING_CONFIG_SET", "demo")
        if config_set != "demo":
            return False
        row = self.get_pair_config(epic, config_set)
        return bool(row.get("is_enabled", False)) if row else False

    def is_monitor_only(self, epic: str, config_set: Optional[str] = None) -> bool:
        config_set = config_set or os.getenv("TRADING_CONFIG_SET", "demo")
        if config_set != "demo":
            return True
        row = self.get_pair_config(epic, config_set)
        return bool(row.get("monitor_only", True)) if row else True

    def invalidate(self) -> None:
        with self._lock:
            self._pair_cache.clear()
            self._cache_ts = None
            self._loaded_config_set = None

    def _refresh_if_stale(self, config_set: str) -> None:
        with self._lock:
            now = datetime.utcnow()
            cache_valid = (
                self._cache_ts is not None
                and self._loaded_config_set == config_set
                and (now - self._cache_ts) < self._cache_ttl
            )
            if cache_valid:
                return
            try:
                rows = self._load_from_db(config_set)
                self._pair_cache = {row["epic"]: dict(row) for row in rows}
                self._cache_ts = now
                self._loaded_config_set = config_set
                logger.info("[SMC_SIMPLE_V2_CFG] Loaded %d pair rows (config_set=%s)", len(rows), config_set)
            except Exception as exc:
                logger.warning("[SMC_SIMPLE_V2_CFG] DB load failed (%s) - using last-known-good", exc)
                if self._cache_ts is None:
                    self._pair_cache = {}
                    self._loaded_config_set = config_set

    def _load_from_db(self, config_set: str) -> list:
        conn = psycopg2.connect(self._db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        epic,
                        pair_name,
                        is_enabled,
                        is_traded,
                        monitor_only,
                        structure_lookback_bars,
                        entry_lookback_bars,
                        retest_tolerance_pips,
                        sweep_tolerance_pips,
                        min_break_pips,
                        min_rejection_wick_ratio,
                        max_rejection_body_ratio,
                        min_confirm_body_ratio,
                        fixed_stop_loss_pips,
                        fixed_take_profit_pips,
                        directions,
                        entry_models,
                        min_signal_gap_minutes,
                        adx_min,
                        adx_max,
                        atr_percentile_min,
                        atr_percentile_max,
                        bb_width_percentile_min,
                        bb_width_percentile_max,
                        efficiency_ratio_min,
                        ema200_mode,
                        macd_mode,
                        allowed_hours_utc,
                        base_confidence,
                        parameter_overrides
                    FROM smc_simple_v2_pair_overrides
                    WHERE config_set = %s
                    """,
                    (config_set,),
                )
                rows = cur.fetchall()
                for row in rows:
                    bag = row.get("parameter_overrides")
                    if isinstance(bag, str):
                        try:
                            row["parameter_overrides"] = json.loads(bag)
                        except Exception:
                            row["parameter_overrides"] = {}
                return rows
        finally:
            conn.close()


_service: Optional[SMCSimpleV2ConfigService] = None
_service_lock = RLock()


def get_smc_simple_v2_config_service() -> SMCSimpleV2ConfigService:
    global _service
    with _service_lock:
        if _service is None:
            _service = SMCSimpleV2ConfigService()
        return _service
