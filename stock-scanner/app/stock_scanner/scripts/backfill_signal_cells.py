"""Backfill character-cell columns on existing stock_scanner_signals rows.

Populates trend_state / vol_regime / liquidity_tier / cell_market_regime on
historical signals, computed AS-OF each row's signal_date (causal: only screening
metrics with calculation_date <= signal_date), using
stock_scanner.core.routing.cell_tagger.

Runs entirely inside the stock-scanner container:
    python -m stock_scanner.scripts.backfill_signal_cells
    python -m stock_scanner.scripts.backfill_signal_cells --only-missing
    python -m stock_scanner.scripts.backfill_signal_cells --dry-run --limit 200

By default it tags EVERY row (recompute). --only-missing skips rows that already
have a trend_state, so the job is cheap to re-run after new signals land.

This is a set-based, causal join done in SQL for speed (thousands of rows), then
verified against the pure Python classifier on a sample so the two paths agree.
Read/write to stock_scanner_signals cell columns ONLY -- nothing else is touched.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import psycopg2
import psycopg2.extras

from stock_scanner.core.routing.cell_tagger import (
    ADX_RANGE_MAX,
    ADX_TREND_MIN,
    ATR_HIGH_MIN,
    ATR_LOW_MAX,
    RVOL_HIGH_MIN,
    RVOL_THIN_MAX,
    classify_cell,
)

DB_URL = os.getenv("STOCKS_DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/stocks")


# Set-based causal backfill. For each target signal we join the most-recent
# screening row (LATERAL, calculation_date <= signal_date) and the most-recent
# market_context row, then bucket with the SAME thresholds as cell_tagger. The
# CASE bins are literal here for a single-pass UPDATE; they are kept identical to
# the constants imported above (asserted at runtime below).
_BACKFILL_SQL = """
WITH tgt AS (
    SELECT s.id, s.ticker, s.signal_date
    FROM stock_scanner_signals s
    WHERE s.signal_date IS NOT NULL
      {only_missing}
      {limit_filter}
),
metric AS (
    -- Coerce Postgres numeric 'NaN' to NULL here. Postgres sorts NaN as GREATER
    -- than every real number, so a raw "NaN >= 25" CASE bin would mislabel a
    -- missing metric as 'trend'/'high'. NULLIF(x,'NaN') makes SQL agree with the
    -- Python classifier (which treats NaN as missing -> None).
    SELECT t.id,
           NULLIF(m.adx,             'NaN'::numeric) AS adx,
           NULLIF(m.atr_percent,     'NaN'::numeric) AS atr_percent,
           NULLIF(m.relative_volume, 'NaN'::numeric) AS relative_volume
    FROM tgt t
    LEFT JOIN LATERAL (
        SELECT adx, atr_percent, relative_volume
        FROM stock_screening_metrics sm
        WHERE sm.ticker = t.ticker
          AND sm.calculation_date <= t.signal_date
        ORDER BY sm.calculation_date DESC
        LIMIT 1
    ) m ON TRUE
),
regime AS (
    SELECT t.id, mc.market_regime
    FROM tgt t
    LEFT JOIN LATERAL (
        SELECT market_regime
        FROM market_context mc2
        WHERE mc2.calculation_date <= t.signal_date
        ORDER BY mc2.calculation_date DESC
        LIMIT 1
    ) mc ON TRUE
)
UPDATE stock_scanner_signals s
SET
    trend_state = CASE
        WHEN metric.adx IS NULL THEN NULL
        WHEN metric.adx < {adx_range_max} THEN 'range'
        WHEN metric.adx >= {adx_trend_min} THEN 'trend'
        ELSE 'mid' END,
    vol_regime = CASE
        WHEN metric.atr_percent IS NULL THEN NULL
        WHEN metric.atr_percent < {atr_low_max} THEN 'low'
        WHEN metric.atr_percent >= {atr_high_min} THEN 'high'
        ELSE 'normal' END,
    liquidity_tier = CASE
        WHEN metric.relative_volume IS NULL THEN NULL
        WHEN metric.relative_volume < {rvol_thin_max} THEN 'thin'
        WHEN metric.relative_volume >= {rvol_high_min} THEN 'high'
        ELSE 'normal' END,
    cell_market_regime = NULLIF(TRIM(regime.market_regime), '')
FROM metric
JOIN regime ON regime.id = metric.id
WHERE s.id = metric.id
"""


def backfill(
    *,
    db_url: str = DB_URL,
    only_missing: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill cell columns. Returns a stats dict."""
    only_missing_clause = "AND s.trend_state IS NULL" if only_missing else ""
    limit_clause = f"ORDER BY s.id LIMIT {int(limit)}" if limit else ""

    sql = _BACKFILL_SQL.format(
        only_missing=only_missing_clause,
        limit_filter=limit_clause,
        adx_range_max=ADX_RANGE_MAX,
        adx_trend_min=ADX_TREND_MIN,
        atr_low_max=ATR_LOW_MAX,
        atr_high_min=ATR_HIGH_MIN,
        rvol_thin_max=RVOL_THIN_MAX,
        rvol_high_min=RVOL_HIGH_MIN,
    )

    conn = psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            updated = cur.rowcount

            # Post-backfill coverage snapshot (uses same connection, pre-commit).
            cur.execute(
                """
                SELECT
                    COUNT(*)                                          AS total,
                    COUNT(*) FILTER (WHERE trend_state IS NOT NULL)   AS has_trend,
                    COUNT(*) FILTER (WHERE vol_regime IS NOT NULL)    AS has_vol,
                    COUNT(*) FILTER (WHERE liquidity_tier IS NOT NULL) AS has_liq,
                    COUNT(*) FILTER (WHERE cell_market_regime IS NOT NULL) AS has_regime,
                    COUNT(*) FILTER (WHERE trend_state IS NOT NULL
                                       AND vol_regime IS NOT NULL)    AS has_2axis,
                    COUNT(*) FILTER (WHERE trend_state IS NOT NULL
                                       AND vol_regime IS NOT NULL
                                       AND liquidity_tier IS NOT NULL) AS has_3axis
                FROM stock_scanner_signals
                """
            )
            cov = cur.fetchone()

        if dry_run:
            conn.rollback()
        else:
            conn.commit()

        return {
            "updated": updated,
            "dry_run": dry_run,
            "coverage": dict(cov),
        }
    finally:
        conn.close()


def verify_sample(db_url: str = DB_URL, sample: int = 500) -> dict:
    """Cross-check the SQL backfill against the pure Python classifier.

    Recomputes the cell for a random sample of tagged rows via classify_cell and
    compares to the stored columns. Any mismatch indicates the SQL CASE bins have
    drifted from cell_tagger constants.
    """
    conn = psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)
    mismatches = 0
    checked = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.trend_state, s.vol_regime, s.liquidity_tier,
                       m.adx, m.atr_percent, m.relative_volume
                FROM stock_scanner_signals s
                JOIN LATERAL (
                    SELECT adx, atr_percent, relative_volume
                    FROM stock_screening_metrics sm
                    WHERE sm.ticker = s.ticker
                      AND sm.calculation_date <= s.signal_date
                    ORDER BY sm.calculation_date DESC
                    LIMIT 1
                ) m ON TRUE
                WHERE s.trend_state IS NOT NULL
                ORDER BY random()
                LIMIT %s
                """,
                (sample,),
            )
            for r in cur.fetchall():
                checked += 1
                cell = classify_cell(r["adx"], r["atr_percent"], r["relative_volume"])
                if (
                    cell["trend_state"] != r["trend_state"]
                    or cell["vol_regime"] != r["vol_regime"]
                    or cell["liquidity_tier"] != r["liquidity_tier"]
                ):
                    mismatches += 1
    finally:
        conn.close()
    return {"checked": checked, "mismatches": mismatches}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backfill signal character-cell columns")
    ap.add_argument("--db-url", default=DB_URL)
    ap.add_argument("--only-missing", action="store_true", help="only tag rows with NULL trend_state")
    ap.add_argument("--limit", type=int, default=None, help="cap rows processed (testing)")
    ap.add_argument("--dry-run", action="store_true", help="compute + rollback, no write")
    ap.add_argument("--verify", action="store_true", help="cross-check SQL vs Python classifier")
    args = ap.parse_args(argv)

    stats = backfill(
        db_url=args.db_url,
        only_missing=args.only_missing,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    cov = stats["coverage"]
    print("=" * 70)
    print("BACKFILL SIGNAL CELLS")
    print("=" * 70)
    print(f"rows updated     : {stats['updated']}  (dry_run={stats['dry_run']})")
    print(f"total signals    : {cov['total']}")
    print(f"has trend_state  : {cov['has_trend']}")
    print(f"has vol_regime   : {cov['has_vol']}")
    print(f"has liquidity    : {cov['has_liq']}")
    print(f"has cell_regime  : {cov['has_regime']}")
    print(f"taggable 2-axis  : {cov['has_2axis']}")
    print(f"taggable 3-axis  : {cov['has_3axis']}")

    if args.verify and not args.dry_run:
        v = verify_sample(args.db_url)
        print("-" * 70)
        print(f"verify sample    : checked={v['checked']} mismatches={v['mismatches']}")
        if v["mismatches"]:
            print("  !! SQL bins drifted from cell_tagger constants -- investigate")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
