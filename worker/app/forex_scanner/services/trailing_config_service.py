"""
Worker-side trailing config loader (DB-driven with file fallback).

Mirrors `dev-app/services/trailing_config_service.py` so backtests read the
same `strategy_config.trailing_pair_config` rows that drive live trading.
On DB miss/error, callers fall back to the legacy `PAIR_TRAILING_CONFIGS`
dict in `config_trailing_stops.py`.

Per-strategy lookup (Apr 2026):
    Lookup chain for a given (strategy, epic, is_scalp):
        1. (strategy, epic, is_scalp)
        2. (strategy, 'DEFAULT', is_scalp)
        3. ('DEFAULT', epic, is_scalp)
        4. ('DEFAULT', 'DEFAULT', is_scalp)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, Optional, Tuple

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)

CONFIG_FIELDS = (
    'early_breakeven_trigger_points',
    'early_breakeven_buffer_points',
    'stage1_trigger_points',
    'stage1_lock_points',
    'stage2_trigger_points',
    'stage2_lock_points',
    'stage3_trigger_points',
    'stage3_atr_multiplier',
    'stage3_min_distance',
    'min_trail_distance',
    'break_even_trigger_points',
    'early_failure_stop_enabled',
    'early_failure_check_bars',
    'early_failure_min_mfe_pips',
    'early_failure_stop_pips',
    'enable_partial_close',
    'partial_close_trigger_points',
    'partial_close_size',
)

DEFAULT_STRATEGY = 'DEFAULT'


class TrailingConfigService:
    def __init__(self, config_set: Optional[str] = None, cache_ttl_seconds: int = 120):
        # Worker BTs default to 'demo' since that mirrors the demo trading env.
        self.config_set = config_set or os.getenv('TRADING_ENVIRONMENT', 'demo')
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._db_url = os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config',
        )
        self._cache: Dict[Tuple[str, str, bool], Dict] = {}
        self._cache_loaded_at: Optional[datetime] = None
        self._lock = RLock()

    @contextmanager
    def _connection(self):
        conn = None
        try:
            conn = psycopg2.connect(self._db_url)
            yield conn
        finally:
            if conn is not None:
                conn.close()

    def _reload(self) -> None:
        if not PSYCOPG2_AVAILABLE:
            self._cache = {}
            self._cache_loaded_at = datetime.now()
            return
        try:
            with self._connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT strategy, epic, is_scalp, " + ', '.join(CONFIG_FIELDS)
                        + " FROM trailing_pair_config "
                        "WHERE config_set = %s AND is_active = TRUE",
                        (self.config_set,),
                    )
                    rows = cur.fetchall()
            new_cache: Dict[Tuple[str, str, bool], Dict] = {}
            for row in rows:
                cfg = {}
                for f in CONFIG_FIELDS:
                    v = row.get(f)
                    if v is not None and v.__class__.__name__ == 'Decimal':
                        v = float(v)
                    cfg[f] = v
                key = (row.get('strategy') or DEFAULT_STRATEGY, row['epic'], bool(row['is_scalp']))
                new_cache[key] = cfg
            self._cache = new_cache
            self._cache_loaded_at = datetime.now()
            logger.info(
                f"🔄 [Worker] TrailingConfigService loaded {len(new_cache)} rows "
                f"(config_set={self.config_set})"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ [Worker] TrailingConfigService DB load failed "
                f"(config_set={self.config_set}): {e} — falling back to file"
            )
            if self._cache_loaded_at is None:
                self._cache_loaded_at = datetime.now()

    def _ensure_cache(self) -> None:
        with self._lock:
            if self._cache_loaded_at is None:
                self._reload()
                return
            if datetime.now() - self._cache_loaded_at > self.cache_ttl:
                self._reload()

    def get_config(
        self,
        epic: str,
        is_scalp: bool = False,
        strategy: str = DEFAULT_STRATEGY,
    ) -> Dict:
        """Resolve trailing config for (strategy, epic, is_scalp).

        Falls back through:
            (strategy, epic) → (strategy, DEFAULT) → (DEFAULT, epic) → (DEFAULT, DEFAULT)
        Returns the first non-empty match with None values stripped so the
        caller's dict-merge falls back to file/defaults for missing fields.
        """
        self._ensure_cache()
        strategy = (strategy or DEFAULT_STRATEGY).upper()
        # Layer priority lowest-first so strategy/pair-specific values override.
        # DEFAULT/DEFAULT → DEFAULT/epic → strategy/DEFAULT → strategy/epic.
        layers = [
            (DEFAULT_STRATEGY, 'DEFAULT', is_scalp),
            (DEFAULT_STRATEGY, epic, is_scalp),
            (strategy, 'DEFAULT', is_scalp),
            (strategy, epic, is_scalp),
        ]
        merged: Dict = {}
        for key in layers:
            cfg = self._cache.get(key)
            if not cfg:
                continue
            for k, v in cfg.items():
                if v is not None:
                    merged[k] = v
        return merged

    def invalidate(self) -> None:
        with self._lock:
            self._cache_loaded_at = None


_singleton: Optional[TrailingConfigService] = None
_singleton_lock = RLock()


def get_trailing_config_service() -> TrailingConfigService:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = TrailingConfigService()
        return _singleton
