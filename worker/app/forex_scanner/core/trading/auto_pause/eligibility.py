"""Eligibility allowlist loader.

A cell receives auto-pause protection ONLY if it has an ``eligible=TRUE`` row in
``auto_pause_eligibility``. The baseline columns are a FROZEN promotion-time
record (never recomputed from recent performance) and exist to document why the
cell qualifies — see the migration for the full rationale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import psycopg2
import psycopg2.extras

from .config import strategy_config_dsn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EligibilityRecord:
    strategy: str
    epic: str
    config_set: str
    baseline_pf: Optional[float]
    baseline_n: Optional[int]
    monthly_trade_rate: Optional[float]


def load_eligible_cells(
    *, dsn: Optional[str] = None, conn: Any = None
) -> List[EligibilityRecord]:
    """Return all cells flagged eligible=TRUE."""
    query = """
        SELECT strategy, epic, config_set, baseline_pf, baseline_n, monthly_trade_rate
        FROM auto_pause_eligibility
        WHERE eligible = TRUE
        ORDER BY strategy, epic, config_set
    """
    own = conn is None
    if own:
        conn = psycopg2.connect(dsn or strategy_config_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[AutoPause] load_eligible_cells failed: %s", exc)
        return []
    finally:
        if own:
            conn.close()

    out: List[EligibilityRecord] = []
    for r in rows:
        out.append(
            EligibilityRecord(
                strategy=r["strategy"],
                epic=r["epic"],
                config_set=r["config_set"],
                baseline_pf=float(r["baseline_pf"]) if r["baseline_pf"] is not None else None,
                baseline_n=int(r["baseline_n"]) if r["baseline_n"] is not None else None,
                monthly_trade_rate=(
                    float(r["monthly_trade_rate"])
                    if r["monthly_trade_rate"] is not None
                    else None
                ),
            )
        )
    return out
