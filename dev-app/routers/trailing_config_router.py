"""
Trailing Config CRUD API.

Backs the trading-ui per-strategy trailing editor. Reads/writes
`strategy_config.trailing_pair_config` rows keyed by
`(strategy, config_set, epic, is_scalp)` and invalidates the in-memory
TrailingConfigService cache after each write so live trading picks up
the change without restart.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trailing-config", tags=["trailing-config"])

ALLOWED_STRATEGIES = {
    "DEFAULT",
    "SMC_SIMPLE",
    "XAU_GOLD",
    "RANGE_FADE",
    "MEAN_REVERSION",
    "RANGE_STRUCTURE",
}

ALLOWED_CONFIG_SETS = {"demo", "live"}

EDITABLE_FIELDS = (
    "early_breakeven_trigger_points",
    "early_breakeven_buffer_points",
    "stage1_trigger_points",
    "stage1_lock_points",
    "stage2_trigger_points",
    "stage2_lock_points",
    "stage3_trigger_points",
    "stage3_atr_multiplier",
    "stage3_min_distance",
    "min_trail_distance",
    "break_even_trigger_points",
    "enable_partial_close",
    "partial_close_trigger_points",
    "partial_close_size",
)


def _db_url() -> str:
    return os.getenv(
        "STRATEGY_CONFIG_DATABASE_URL",
        "postgresql://postgres:postgres@postgres:5432/strategy_config",
    )


@contextmanager
def _conn():
    c = psycopg2.connect(_db_url())
    try:
        yield c
    finally:
        c.close()


def _normalize_strategy(s: str) -> str:
    s = (s or "DEFAULT").upper()
    if s not in ALLOWED_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{s}'. Allowed: {sorted(ALLOWED_STRATEGIES)}",
        )
    return s


def _normalize_config_set(cs: str) -> str:
    cs = (cs or "demo").lower()
    if cs not in ALLOWED_CONFIG_SETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown config_set '{cs}'. Allowed: {sorted(ALLOWED_CONFIG_SETS)}",
        )
    return cs


def _invalidate_cache():
    try:
        from services.trailing_config_service import get_trailing_config_service
        get_trailing_config_service().invalidate()
    except Exception as e:
        logger.warning(f"Trailing cache invalidate skipped: {e}")


class TrailingConfigUpsert(BaseModel):
    strategy: str = Field(..., description="DEFAULT/SMC_SIMPLE/XAU_GOLD/...")
    config_set: str = Field(..., description="demo or live")
    epic: str = Field(..., description="Epic, or 'DEFAULT' for strategy-default row")
    is_scalp: bool = False
    is_active: bool = True
    change_reason: Optional[str] = None
    updated_by: Optional[str] = "trading-ui"

    early_breakeven_trigger_points: Optional[int] = None
    early_breakeven_buffer_points: Optional[int] = None
    stage1_trigger_points: Optional[int] = None
    stage1_lock_points: Optional[int] = None
    stage2_trigger_points: Optional[int] = None
    stage2_lock_points: Optional[int] = None
    stage3_trigger_points: Optional[int] = None
    stage3_atr_multiplier: Optional[float] = None
    stage3_min_distance: Optional[int] = None
    min_trail_distance: Optional[int] = None
    break_even_trigger_points: Optional[int] = None
    enable_partial_close: Optional[bool] = None
    partial_close_trigger_points: Optional[int] = None
    partial_close_size: Optional[float] = None


@router.get("/list")
def list_configs(
    config_set: str = Query("demo"),
    strategy: Optional[str] = Query(None),
):
    """List all trailing config rows, optionally filtered by strategy."""
    cs = _normalize_config_set(config_set)
    where = ["config_set = %s"]
    params: list = [cs]
    if strategy:
        where.append("strategy = %s")
        params.append(_normalize_strategy(strategy))
    sql = (
        "SELECT id, strategy, config_set, epic, is_scalp, is_active, "
        + ", ".join(EDITABLE_FIELDS)
        + ", updated_by, change_reason, created_at, updated_at "
        "FROM trailing_pair_config WHERE " + " AND ".join(where)
        + " ORDER BY strategy, epic, is_scalp"
    )
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return {"rows": rows, "count": len(rows)}


@router.get("")
def get_resolved(
    epic: str = Query(...),
    strategy: str = Query("DEFAULT"),
    config_set: str = Query("demo"),
    is_scalp: bool = Query(False),
):
    """Resolve a row using the same fallback chain as the loader, with provenance."""
    s = _normalize_strategy(strategy)
    cs = _normalize_config_set(config_set)
    candidates = [
        (s, epic, is_scalp),
        (s, "DEFAULT", is_scalp),
        ("DEFAULT", epic, is_scalp),
        ("DEFAULT", "DEFAULT", is_scalp),
    ]
    fields_sql = ", ".join(EDITABLE_FIELDS)
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        for st, ep, sc in candidates:
            cur.execute(
                f"SELECT id, strategy, epic, is_scalp, is_active, {fields_sql} "
                "FROM trailing_pair_config "
                "WHERE strategy=%s AND config_set=%s AND epic=%s AND is_scalp=%s "
                "AND is_active=TRUE",
                (st, cs, ep, sc),
            )
            row = cur.fetchone()
            if row:
                return {
                    "matched": {"strategy": st, "epic": ep, "is_scalp": sc},
                    "row": row,
                }
    return {"matched": None, "row": None}


@router.put("")
def upsert(payload: TrailingConfigUpsert):
    s = _normalize_strategy(payload.strategy)
    cs = _normalize_config_set(payload.config_set)
    cols = ["strategy", "config_set", "epic", "is_scalp", "is_active"]
    vals = [s, cs, payload.epic, payload.is_scalp, payload.is_active]
    for f in EDITABLE_FIELDS:
        v = getattr(payload, f)
        cols.append(f)
        vals.append(v)
    cols.extend(["updated_by", "change_reason"])
    vals.extend([payload.updated_by, payload.change_reason])

    placeholders = ", ".join(["%s"] * len(cols))
    update_set = ", ".join(
        f"{c}=EXCLUDED.{c}" for c in cols if c not in ("strategy", "config_set", "epic", "is_scalp")
    )
    sql = (
        f"INSERT INTO trailing_pair_config ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (strategy, config_set, epic, is_scalp) DO UPDATE SET {update_set} "
        f"RETURNING id"
    )
    with _conn() as c, c.cursor() as cur:
        cur.execute(sql, vals)
        fetched = cur.fetchone()
        if fetched is None:
            c.rollback()
            raise HTTPException(status_code=500, detail="upsert returned no id")
        row_id = fetched[0]
        c.commit()
    _invalidate_cache()
    return {"id": row_id, "ok": True}


@router.delete("/{row_id}")
def delete(row_id: int):
    with _conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM trailing_pair_config WHERE id = %s", (row_id,))
        c.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="row not found")
    _invalidate_cache()
    return {"deleted": row_id}


@router.get("/strategies")
def list_strategies():
    return {"strategies": sorted(ALLOWED_STRATEGIES), "config_sets": sorted(ALLOWED_CONFIG_SETS)}
