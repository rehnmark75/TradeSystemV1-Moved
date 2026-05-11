"""
Trailing Config Service — DB-backed, environment-scoped.

Replaces the hardcoded PAIR_TRAILING_CONFIGS / SCALP_TRAILING_CONFIGS dicts
previously in dev-app/config.py. Configs are now stored in
strategy_config.trailing_pair_config and scoped by TRADING_ENVIRONMENT
('demo' or 'live') so that each container sees only its own values.

Behaviour:
- On first call, loads all rows for the current config_set into memory.
- In-memory cache refreshes every 120s.
- If DB load fails:
    - live   → raise RuntimeError (fail-hard, same pattern as LPF / SMC config)
    - demo   → warn and continue with empty cache (returns {})
- Missing epic → falls back to the 'DEFAULT' row for the same config_set.
"""

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

# Columns that live in trailing_pair_config and are exposed to the caller
# (i.e. what used to be the dict value in PAIR_TRAILING_CONFIGS).
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


class TrailingConfigService:
    """Cached, environment-scoped loader for trailing_pair_config rows."""

    def __init__(self, config_set: Optional[str] = None, cache_ttl_seconds: int = 120):
        self.config_set = config_set or os.getenv('TRADING_ENVIRONMENT', 'demo')
        self._trading_env = os.getenv('TRADING_ENVIRONMENT', 'demo')
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
        """Load all active rows for this config_set into memory."""
        if not PSYCOPG2_AVAILABLE:
            msg = "psycopg2 not available — trailing config disabled"
            if self._trading_env == 'live':
                raise RuntimeError(f"TrailingConfigService: {msg}")
            logger.warning(f"⚠️ TrailingConfigService: {msg} (demo mode)")
            self._cache = {}
            self._cache_loaded_at = datetime.now()
            return

        try:
            with self._connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT strategy, epic, is_scalp, """
                        + ', '.join(CONFIG_FIELDS)
                        + """
                        FROM trailing_pair_config
                        WHERE config_set = %s AND is_active = TRUE
                        """,
                        (self.config_set,),
                    )
                    rows = cur.fetchall()

            new_cache: Dict[Tuple[str, str, bool], Dict] = {}
            for row in rows:
                strategy = (row.get('strategy') or 'DEFAULT')
                key = (strategy, row['epic'], bool(row['is_scalp']))
                # Normalise Decimal → float so downstream math (e.g. atr * multiplier)
                # works the same way it did with the old Python dict values.
                cfg = {}
                for f in CONFIG_FIELDS:
                    v = row.get(f)
                    if v is not None and hasattr(v, '__class__') and v.__class__.__name__ == 'Decimal':
                        v = float(v)
                    cfg[f] = v
                new_cache[key] = cfg

            self._cache = new_cache
            self._cache_loaded_at = datetime.now()
            logger.info(
                f"🔄 TrailingConfigService loaded {len(new_cache)} rows "
                f"(config_set={self.config_set})"
            )

        except Exception as e:
            logger.error(
                f"❌ TrailingConfigService: DB load failed "
                f"(config_set={self.config_set}): {e}"
            )
            if self._trading_env == 'live':
                raise RuntimeError(
                    f"TrailingConfigService refusing to serve in live mode with no DB data: {e}"
                ) from e
            # Demo: keep serving whatever stale cache we have (or empty)
            if self._cache_loaded_at is None:
                self._cache_loaded_at = datetime.now()  # avoid thrashing retries

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
        strategy: str = 'DEFAULT',
    ) -> Dict:
        """Return the trailing config dict for (strategy, epic, is_scalp).

        Lookup chain:
            (strategy, epic) → (strategy, DEFAULT) → (DEFAULT, epic) → (DEFAULT, DEFAULT)
        Returns {} if nothing matches.
        """
        self._ensure_cache()
        strategy = (strategy or 'DEFAULT').upper()
        # Layer priority lowest-first so strategy/pair values override DEFAULT.
        layers = [
            ('DEFAULT', 'DEFAULT', is_scalp),
            ('DEFAULT', epic, is_scalp),
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
        """Force a reload on the next call (used after UI saves)."""
        with self._lock:
            self._cache_loaded_at = None


# -------- Module-level singleton ----------------------------------------------

_singleton: Optional[TrailingConfigService] = None
_singleton_lock = RLock()


def get_trailing_config_service() -> TrailingConfigService:
    """Return the per-process TrailingConfigService singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = TrailingConfigService()
        return _singleton
