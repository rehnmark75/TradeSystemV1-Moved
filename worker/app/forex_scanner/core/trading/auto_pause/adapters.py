"""Per-strategy adapters for flipping the ``monitor_only`` flag.

Each active strategy stores its execute/monitor state in its own
``*_pair_overrides`` table in the ``strategy_config`` DB, and the storage is
NOT uniform:

  * ``monitor_only`` is a JSONB key (``parameter_overrides->>'monitor_only'``)
    for SMC_SIMPLE, and a direct boolean column for every other strategy.
  * The environment scope key is ``config_id`` (SMC_SIMPLE, IMPULSE_FADE),
    ``config_set`` (most), or absent (DONCHIAN_TURTLE — unique on epic alone).
  * RANGE_FADE additionally scopes by ``profile_name`` (only ``'5m'`` exists).

This module hides those differences behind a small registry so the rest of the
auto-pause layer can pause/resume any cell uniformly. Table names come only from
this trusted registry (never user input), so f-string interpolation of the
table name is safe; all values are passed as bound parameters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psycopg2

from .config import strategy_config_dsn

logger = logging.getLogger(__name__)

# environment ('demo'/'live') -> config_id value, for tables scoped by config_id
CONFIG_ID_BY_ENV: Dict[str, int] = {"demo": 3, "live": 2}


@dataclass(frozen=True)
class StrategyAdapter:
    strategy: str            # canonical name as stored in alert_history.strategy
    table: str               # the *_pair_overrides table
    monitor_only_kind: str   # 'jsonb' | 'column'
    scope_kind: str          # 'config_id' | 'config_set' | 'none'
    profile_name: Optional[str] = None  # RANGE_FADE -> '5m'; others None


# Only ACTIVE strategies are registered. Archived strategies
# (bb_supertrend, fvg_retest, range_structure, ranging_market, squeeze_momentum,
# volume_profile) are intentionally omitted. SMC_SIMPLE_V2 is not DB-wired yet.
REGISTRY: Dict[str, StrategyAdapter] = {
    "SMC_SIMPLE":      StrategyAdapter("SMC_SIMPLE",      "smc_simple_pair_overrides",      "jsonb",  "config_id"),
    "SMC_MOMENTUM":    StrategyAdapter("SMC_MOMENTUM",    "smc_momentum_pair_overrides",    "column", "config_set"),
    "RANGE_FADE":      StrategyAdapter("RANGE_FADE",      "range_fade_pair_overrides",      "column", "config_set", profile_name="5m"),
    "MEAN_REVERSION":  StrategyAdapter("MEAN_REVERSION",  "mean_reversion_pair_overrides",  "column", "config_set"),
    "IMPULSE_FADE":    StrategyAdapter("IMPULSE_FADE",    "impulse_fade_pair_overrides",    "column", "config_id"),
    "KAMA_V2":         StrategyAdapter("KAMA_V2",         "kama_v2_pair_overrides",         "column", "config_set"),
    "INSIDE_DAY":      StrategyAdapter("INSIDE_DAY",      "inside_day_pair_overrides",      "column", "config_set"),
    "XAU_GOLD":        StrategyAdapter("XAU_GOLD",        "xau_gold_pair_overrides",        "column", "config_set"),
    "DONCHIAN_TURTLE": StrategyAdapter("DONCHIAN_TURTLE", "donchian_turtle_pair_overrides", "column", "none"),
    "FA_OR_ATR_TRAIL": StrategyAdapter("FA_OR_ATR_TRAIL", "fa_or_atr_trail_pair_overrides", "column", "config_set"),
}


def get_adapter(strategy: str) -> Optional[StrategyAdapter]:
    return REGISTRY.get(str(strategy or "").upper())


# --------------------------------------------------------------------------- #
# Pure SQL builders (unit-testable, no DB access)
# --------------------------------------------------------------------------- #
def build_scope_clause(
    adapter: StrategyAdapter, epic: str, environment: str
) -> Tuple[str, List[Any]]:
    """Return (where_sql, params) selecting exactly the cell's row(s)."""
    clauses = ["epic = %s"]
    params: List[Any] = [epic]

    if adapter.scope_kind == "config_id":
        clauses.append("config_id = %s")
        params.append(CONFIG_ID_BY_ENV.get(environment, CONFIG_ID_BY_ENV["demo"]))
    elif adapter.scope_kind == "config_set":
        clauses.append("config_set = %s")
        params.append(environment)
    # scope_kind == 'none' -> epic only

    if adapter.profile_name is not None:
        clauses.append("profile_name = %s")
        params.append(adapter.profile_name)

    return " AND ".join(clauses), params


def monitor_only_set_sql(adapter: StrategyAdapter, value: bool) -> Tuple[str, List[Any]]:
    """Return (set_sql, params) to set monitor_only to ``value``.

    JSONB strategies store the flag as the text ``'true'`` and resume by
    *removing* the key (matching the existing project convention), so the cell
    falls back to its normal config rather than carrying a stale ``false``.
    """
    if adapter.monitor_only_kind == "jsonb":
        if value:
            return (
                "parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb) "
                "|| '{\"monitor_only\": \"true\"}'::jsonb",
                [],
            )
        return (
            "parameter_overrides = COALESCE(parameter_overrides, '{}'::jsonb) - 'monitor_only'",
            [],
        )
    # direct boolean column
    return ("monitor_only = %s", [value])


def monitor_only_select_expr(adapter: StrategyAdapter) -> str:
    if adapter.monitor_only_kind == "jsonb":
        return "(parameter_overrides->>'monitor_only')::boolean"
    return "monitor_only"


# --------------------------------------------------------------------------- #
# DB IO (thin; connection injectable for tests)
# --------------------------------------------------------------------------- #
def set_monitor_only(
    adapter: StrategyAdapter,
    epic: str,
    environment: str,
    value: bool,
    *,
    dsn: Optional[str] = None,
    conn: Any = None,
) -> int:
    """Set monitor_only for a cell. Returns number of rows affected."""
    set_sql, set_params = monitor_only_set_sql(adapter, value)
    scope_sql, scope_params = build_scope_clause(adapter, epic, environment)
    query = f"UPDATE {adapter.table} SET {set_sql} WHERE {scope_sql}"
    params = set_params + scope_params

    own = conn is None
    if own:
        conn = psycopg2.connect(dsn or strategy_config_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            affected = cur.rowcount
        if own:
            conn.commit()
        if affected == 0:
            logger.warning(
                "[AutoPause] set_monitor_only matched 0 rows for %s %s (%s) — "
                "scope mismatch?", adapter.strategy, epic, environment,
            )
        return affected
    finally:
        if own:
            conn.close()


def get_monitor_only(
    adapter: StrategyAdapter,
    epic: str,
    environment: str,
    *,
    dsn: Optional[str] = None,
    conn: Any = None,
) -> Optional[bool]:
    """Read current monitor_only for a cell. None if no matching row."""
    scope_sql, scope_params = build_scope_clause(adapter, epic, environment)
    query = (
        f"SELECT {monitor_only_select_expr(adapter)} AS monitor_only "
        f"FROM {adapter.table} WHERE {scope_sql} LIMIT 1"
    )
    own = conn is None
    if own:
        conn = psycopg2.connect(dsn or strategy_config_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute(query, scope_params)
            row = cur.fetchone()
        if row is None:
            return None
        return bool(row[0]) if row[0] is not None else False
    finally:
        if own:
            conn.close()


def _config_service_instance(strategy: str) -> Any:
    """Lazily resolve a strategy's config-service singleton (or None)."""
    s = strategy
    if s == "SMC_SIMPLE":
        from forex_scanner.services.smc_simple_config_service import get_smc_simple_config_service
        return get_smc_simple_config_service()
    if s == "RANGE_FADE":
        from forex_scanner.services.range_fade_config_service import RangeFadeConfigService
        return RangeFadeConfigService.get_instance()
    if s == "SMC_MOMENTUM":
        from forex_scanner.services.smc_momentum_config_service import SMCMomentumConfigService
        return SMCMomentumConfigService.get_instance()
    if s == "MEAN_REVERSION":
        from forex_scanner.services.mean_reversion_config_service import MeanReversionConfigService
        return MeanReversionConfigService.get_instance()
    if s == "IMPULSE_FADE":
        from forex_scanner.services.impulse_fade_config_service import ImpulseFadeConfigService
        return ImpulseFadeConfigService.get_instance()
    if s == "XAU_GOLD":
        from forex_scanner.services.xau_gold_config_service import XAUGoldConfigService
        return XAUGoldConfigService.get_instance()
    if s == "DONCHIAN_TURTLE":
        from forex_scanner.services.donchian_turtle_config_service import DonchianTurtleConfigService
        return DonchianTurtleConfigService.get_instance()
    if s == "FA_OR_ATR_TRAIL":
        from forex_scanner.services.fa_or_atr_trail_config_service import FAORATRTrailConfigService
        return FAORATRTrailConfigService.get_instance()
    if s == "KAMA_V2":
        from forex_scanner.services.kama_v2_config_service import get_kama_v2_config_service
        return get_kama_v2_config_service()
    if s == "INSIDE_DAY":
        from forex_scanner.services.inside_day_config_service import get_inside_day_config_service
        return get_inside_day_config_service()
    return None


def refresh_config_cache(strategy: str) -> bool:
    """Best-effort in-process refresh of a strategy's config cache.

    Called right after flipping ``monitor_only`` so the running scanner picks up
    the change immediately instead of waiting for the service TTL. On ANY error
    (or a service that exposes neither ``refresh`` nor ``invalidate_cache``) it
    falls back silently to the TTL (<= 5 min) — correctness never depends on
    this. Returns True if a refresh call was actually made.
    """
    s = str(strategy or "").upper()
    try:
        svc = _config_service_instance(s)
        if svc is None:
            logger.debug("[AutoPause] no config service mapped for %s; TTL fallback", s)
            return False
        for method in ("refresh", "invalidate_cache"):
            fn = getattr(svc, method, None)
            if callable(fn):
                fn()
                return True
        logger.debug("[AutoPause] %s service has no refresh/invalidate; TTL fallback", s)
        return False
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("[AutoPause] cache refresh for %s failed (%s); TTL fallback", s, exc)
        return False
