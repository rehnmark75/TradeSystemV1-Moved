"""
Trailing Ratio Service — DB-backed, environment-scoped.

Replaces the hardcoded DEFAULT_TRAILING_RATIOS / DEFAULT_SCALP_TRAILING_RATIOS
and PAIR_*_RATIO_OVERRIDES dicts previously in dev-app/config.py. Ratios are
now stored in strategy_config.trailing_ratio_config scoped by
TRADING_ENVIRONMENT ('demo' or 'live').

Resolution order for get_ratios(epic, is_scalp):
    1. Pair row (epic = <epic>, is_scalp = X) — NULL columns inherit from step 2
    2. DEFAULT row (epic = 'DEFAULT', is_scalp = X)
    3. Hardcoded fallback from config.py (demo-only graceful degrade)

In-memory cache refreshes every 120s. Live env fails hard on DB outage,
demo env warns and keeps serving stale/fallback values.
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

# All tunable ratio/floor columns exposed to callers. Must stay aligned with
# the DB schema AND with the keys read by compute_sltp_trailing_config().
RATIO_FIELDS = (
    'early_be_trigger_ratio',
    'stage1_trigger_ratio',
    'stage2_trigger_ratio',
    'stage3_trigger_ratio',
    'break_even_trigger_ratio',
    'partial_close_trigger_ratio',
    'stage1_lock_ratio',
    'stage2_lock_ratio',
    'early_be_buffer_points',
    'stage3_atr_multiplier',
    'stage3_min_distance_ratio',
    'min_trail_distance_ratio',
    'min_early_be_trigger',
    'min_stage1_trigger',
    'min_stage1_lock',
    'min_stage2_trigger',
    'min_stage2_lock',
    'min_stage3_trigger',
    'min_break_even_trigger',
    'min_trail_distance',
)


class TrailingRatioService:
    """Cached, env-scoped loader for trailing_ratio_config rows."""

    def __init__(self, config_set: Optional[str] = None, cache_ttl_seconds: int = 120):
        self.config_set = config_set or os.getenv('TRADING_ENVIRONMENT', 'demo')
        self._trading_env = os.getenv('TRADING_ENVIRONMENT', 'demo')
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._db_url = os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config',
        )
        # key = (epic, is_scalp) → dict of ratio fields (None values allowed)
        self._cache: Dict[Tuple[str, bool], Dict] = {}
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
            msg = "psycopg2 not available — trailing ratios falling back to file"
            if self._trading_env == 'live':
                raise RuntimeError(f"TrailingRatioService: {msg}")
            logger.warning(f"⚠️ TrailingRatioService: {msg} (demo mode)")
            self._cache = {}
            self._cache_loaded_at = datetime.now()
            return

        try:
            with self._connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT epic, is_scalp, "
                        + ', '.join(RATIO_FIELDS)
                        + " FROM trailing_ratio_config "
                        "WHERE config_set = %s AND is_active = TRUE",
                        (self.config_set,),
                    )
                    rows = cur.fetchall()

            new_cache: Dict[Tuple[str, bool], Dict] = {}
            for row in rows:
                key = (row['epic'], bool(row['is_scalp']))
                cfg = {}
                for f in RATIO_FIELDS:
                    v = row.get(f)
                    if v is not None and hasattr(v, '__class__') and v.__class__.__name__ == 'Decimal':
                        v = float(v)
                    cfg[f] = v
                new_cache[key] = cfg

            self._cache = new_cache
            self._cache_loaded_at = datetime.now()
            logger.info(
                f"🔄 TrailingRatioService loaded {len(new_cache)} rows "
                f"(config_set={self.config_set})"
            )

        except Exception as e:
            logger.error(
                f"❌ TrailingRatioService: DB load failed "
                f"(config_set={self.config_set}): {e}"
            )
            if self._trading_env == 'live':
                raise RuntimeError(
                    f"TrailingRatioService refusing to serve in live mode with no DB data: {e}"
                ) from e
            if self._cache_loaded_at is None:
                self._cache_loaded_at = datetime.now()

    def _ensure_cache(self) -> None:
        with self._lock:
            if self._cache_loaded_at is None:
                self._reload()
                return
            if datetime.now() - self._cache_loaded_at > self.cache_ttl:
                self._reload()

    def get_ratios(self, epic: Optional[str], is_scalp: bool) -> Dict:
        """Merge DEFAULT row with optional pair row; NULLs inherit from DEFAULT."""
        self._ensure_cache()
        default_cfg = self._cache.get(('DEFAULT', is_scalp), {})
        merged: Dict = {k: default_cfg.get(k) for k in RATIO_FIELDS}

        if epic:
            pair_cfg = self._cache.get((epic, is_scalp))
            if pair_cfg:
                for k, v in pair_cfg.items():
                    if v is not None:
                        merged[k] = v

        return merged

    def invalidate(self) -> None:
        with self._lock:
            self._cache_loaded_at = None


_singleton: Optional[TrailingRatioService] = None
_singleton_lock = RLock()


def get_trailing_ratio_service() -> TrailingRatioService:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = TrailingRatioService()
        return _singleton
