"""Scanner cell-edge analyzer -- the LEARN half of the edge-map router.

For each (scanner_name x character-cell) over a rolling trailing window, from
CLEAN closed forward outcomes in stock_scanner_signals, compute:

    n, wins, pf, win_rate, avg_pnl_pct, window_start, window_end, calendar_days

and assign a verdict with guardrails, then upsert into scanner_cell_edge.

Character cell axes (tagged onto each signal by
stock_scanner.core.routing.cell_tagger; see migration 043):
    trend_state    : 'range' | 'mid'   | 'trend'
    vol_regime     : 'low'   | 'normal' | 'high'
    liquidity_tier : 'thin'  | 'normal' | 'high'

Two grids are computed every run so we can see whether the 3rd axis adds edge
dispersion. The router (a later stage) picks whichever grid the caller wants:
    * 2-axis  : trend x vol            (liquidity_tier = NULL, market_regime = NULL)
    * 3-axis  : trend x vol x liquidity(liquidity_tier set,  market_regime = NULL)

CLEAN-OUTCOME FILTER (critical):
    * realized_pnl_pct IS NOT NULL          (outcome resolved)
    * ABS(realized_pnl_pct) <= MAX_ABS_PNL_PCT (default 100) -- drop data blow-ups
    * status NOT IN ('invalid','data_error')  -- the contamination flag a separate
      agent is adding; guarded here whether or not those statuses exist yet.

VERDICT GUARDRAILS:
    insufficient : n < MIN_N (30) OR calendar_days < MIN_CALENDAR_DAYS (21)
                   -> kills single-regime few-week artifacts
    trade        : pf >= 1.3
    monitor      : 1.0 <= pf < 1.3
    block        : pf < 1.0

Run standalone (inside the stock-scanner container):
    python -m stock_scanner.analysis.scanner_cell_edge_analyzer
    python -m stock_scanner.analysis.scanner_cell_edge_analyzer --window-days 90 --dry-run

Import from the scheduler (call AFTER nightly outcome tracking has closed signals):
    from stock_scanner.analysis.scanner_cell_edge_analyzer import run
    summary = run(window_days=120)

Read-mostly: writes ONLY to scanner_cell_edge. Never touches signals, live, or demo.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
DB_URL = os.getenv("STOCKS_DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/stocks")

DEFAULT_WINDOW_DAYS = 120

# Clean-outcome filter
MAX_ABS_PNL_PCT = 100.0
CONTAMINATED_STATUSES = ("invalid", "data_error")

# Verdict guardrails
MIN_N = 30
MIN_CALENDAR_DAYS = 21
PF_TRADE = 1.3
PF_MONITOR = 1.0

# pf sentinel when there are wins but zero losing PnL (undefined ratio)
PF_CAP = 9999.0


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------
@dataclass
class CellEdge:
    scanner_name: str
    trend_state: str
    vol_regime: str
    liquidity_tier: Optional[str]
    market_regime: Optional[str]
    n: int
    wins: int
    pf: Optional[float]
    win_rate: Optional[float]
    avg_pnl_pct: Optional[float]
    window_start: Optional[date]
    window_end: Optional[date]
    calendar_days: Optional[int]
    verdict: str


@dataclass
class RunSummary:
    window_days: int
    window_start: date
    window_end: date
    clean_rows: int
    excluded_rows: int
    rows_2axis: int
    rows_3axis: int
    verdict_counts: dict = field(default_factory=dict)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def assign_verdict(n: int, calendar_days: Optional[int], pf: Optional[float]) -> str:
    if n < MIN_N or (calendar_days is None) or calendar_days < MIN_CALENDAR_DAYS:
        return "insufficient"
    if pf is None:
        # pf undefined only happens when wins exist but no losing PnL -> strong edge
        return "trade"
    if pf >= PF_TRADE:
        return "trade"
    if pf >= PF_MONITOR:
        return "monitor"
    return "block"


# ---------------------------------------------------------------------------
# Core query: aggregate one grid
# ---------------------------------------------------------------------------
# The clean-outcome filter is applied in the WHERE clause. status is guarded with
# a NOT IN so it is safe whether or not the 'invalid'/'data_error' statuses have
# been introduced yet by the outcome-flagging agent.
_GRID_SQL = """
    SELECT
        scanner_name,
        trend_state,
        vol_regime,
        {liq_select}                              AS liquidity_tier,
        COUNT(*)                                  AS n,
        COUNT(*) FILTER (WHERE realized_pnl_pct > 0)                    AS wins,
        SUM(realized_pnl_pct) FILTER (WHERE realized_pnl_pct > 0)       AS gross_gain,
        SUM(realized_pnl_pct) FILTER (WHERE realized_pnl_pct < 0)       AS gross_loss,
        AVG(realized_pnl_pct)                     AS avg_pnl_pct,
        MIN(signal_date)                          AS window_start,
        MAX(signal_date)                          AS window_end
    FROM stock_scanner_signals
    WHERE realized_pnl_pct IS NOT NULL
      AND ABS(realized_pnl_pct) <= %(max_abs_pnl)s
      AND (status IS NULL OR status NOT IN %(bad_status)s)
      AND signal_date >= %(win_start)s
      AND signal_date <= %(win_end)s
      AND trend_state IS NOT NULL
      AND vol_regime IS NOT NULL
      {liq_filter}
    GROUP BY scanner_name, trend_state, vol_regime{liq_group}
"""


def _compute_grid(cur, win_start: date, win_end: date, three_axis: bool) -> list[CellEdge]:
    if three_axis:
        # 3-axis grid: group by liquidity_tier as well (only tagged rows)
        sql = _GRID_SQL.format(
            liq_select="liquidity_tier",
            liq_filter="AND liquidity_tier IS NOT NULL",
            liq_group=", liquidity_tier",
        )
    else:
        # 2-axis rollup: liquidity_tier collapsed to NULL, not in GROUP BY
        sql = _GRID_SQL.format(
            liq_select="NULL::varchar",
            liq_filter="",
            liq_group="",
        )

    cur.execute(
        sql,
        {
            "max_abs_pnl": MAX_ABS_PNL_PCT,
            "bad_status": CONTAMINATED_STATUSES,
            "win_start": win_start,
            "win_end": win_end,
        },
    )
    rows = cur.fetchall()

    out: list[CellEdge] = []
    for r in rows:
        n = int(r["n"])
        wins = int(r["wins"] or 0)
        gross_gain = float(r["gross_gain"] or 0.0)
        gross_loss = float(r["gross_loss"] or 0.0)  # negative or 0
        avg_pnl = float(r["avg_pnl_pct"]) if r["avg_pnl_pct"] is not None else None
        ws = r["window_start"]
        we = r["window_end"]
        cal_days = (we - ws).days + 1 if (ws is not None and we is not None) else None
        win_rate = round(wins / n, 4) if n else None

        # profit factor = gross gains / abs(gross losses)
        if gross_loss < 0:
            pf: Optional[float] = round(min(gross_gain / abs(gross_loss), PF_CAP), 4)
        elif gross_gain > 0:
            pf = None  # wins but no losing PnL -> undefined (treated as strong)
        else:
            pf = 0.0  # no gains, no losses (all flat) -> weakest

        verdict = assign_verdict(n, cal_days, pf)

        out.append(
            CellEdge(
                scanner_name=r["scanner_name"],
                trend_state=r["trend_state"],
                vol_regime=r["vol_regime"],
                liquidity_tier=r["liquidity_tier"],
                market_regime=None,  # regime axis reserved; not populated in either grid yet
                n=n,
                wins=wins,
                pf=pf,
                win_rate=win_rate,
                avg_pnl_pct=round(avg_pnl, 4) if avg_pnl is not None else None,
                window_start=ws,
                window_end=we,
                calendar_days=cal_days,
                verdict=verdict,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------
_UPSERT_SQL = """
    INSERT INTO scanner_cell_edge
        (scanner_name, trend_state, vol_regime, liquidity_tier, market_regime,
         n, wins, pf, win_rate, avg_pnl_pct,
         window_start, window_end, calendar_days, verdict, computed_at)
    VALUES
        (%(scanner_name)s, %(trend_state)s, %(vol_regime)s, %(liquidity_tier)s, %(market_regime)s,
         %(n)s, %(wins)s, %(pf)s, %(win_rate)s, %(avg_pnl_pct)s,
         %(window_start)s, %(window_end)s, %(calendar_days)s, %(verdict)s, %(computed_at)s)
    ON CONFLICT (scanner_name, trend_state, vol_regime,
                 COALESCE(liquidity_tier, '__ALL__'),
                 COALESCE(market_regime,  '__ALL__'))
    DO UPDATE SET
        n             = EXCLUDED.n,
        wins          = EXCLUDED.wins,
        pf            = EXCLUDED.pf,
        win_rate      = EXCLUDED.win_rate,
        avg_pnl_pct   = EXCLUDED.avg_pnl_pct,
        window_start  = EXCLUDED.window_start,
        window_end    = EXCLUDED.window_end,
        calendar_days = EXCLUDED.calendar_days,
        verdict       = EXCLUDED.verdict,
        computed_at   = EXCLUDED.computed_at
"""


def _upsert(cur, edges: list[CellEdge], computed_at: datetime) -> None:
    for e in edges:
        cur.execute(
            _UPSERT_SQL,
            {
                "scanner_name": e.scanner_name,
                "trend_state": e.trend_state,
                "vol_regime": e.vol_regime,
                "liquidity_tier": e.liquidity_tier,
                "market_regime": e.market_regime,
                "n": e.n,
                "wins": e.wins,
                "pf": e.pf,
                "win_rate": e.win_rate,
                "avg_pnl_pct": e.avg_pnl_pct,
                "window_start": e.window_start,
                "window_end": e.window_end,
                "calendar_days": e.calendar_days,
                "verdict": e.verdict,
                "computed_at": computed_at,
            },
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    window_days: int = DEFAULT_WINDOW_DAYS,
    *,
    db_url: str = DB_URL,
    dry_run: bool = False,
    prune_stale: bool = True,
    conn=None,
) -> RunSummary:
    """Compute the edge-map and upsert into scanner_cell_edge.

    Args:
        window_days: trailing lookback in calendar days (default 120).
        db_url: connection string (default from STOCKS_DATABASE_URL).
        dry_run: compute + report but do not write.
        prune_stale: delete edge rows not refreshed by this run (cells that
            dropped out of the rolling window).
        conn: optional existing psycopg2 connection (e.g. from the scheduler).
            If provided it is used and NOT closed here.

    Returns:
        RunSummary.
    """
    win_end = date.today()
    win_start = date.fromordinal(win_end.toordinal() - int(window_days))
    computed_at = datetime.now(timezone.utc)

    own_conn = conn is None
    if own_conn:
        conn = psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Diagnostic: clean vs excluded row counts over the window
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE realized_pnl_pct IS NOT NULL
                          AND ABS(realized_pnl_pct) <= %(max_abs_pnl)s
                          AND (status IS NULL OR status NOT IN %(bad_status)s)
                    ) AS clean,
                    COUNT(*) FILTER (
                        WHERE realized_pnl_pct IS NOT NULL
                          AND (ABS(realized_pnl_pct) > %(max_abs_pnl)s
                               OR status IN %(bad_status)s)
                    ) AS excluded
                FROM stock_scanner_signals
                WHERE signal_date >= %(win_start)s AND signal_date <= %(win_end)s
                """,
                {
                    "max_abs_pnl": MAX_ABS_PNL_PCT,
                    "bad_status": CONTAMINATED_STATUSES,
                    "win_start": win_start,
                    "win_end": win_end,
                },
            )
            drow = cur.fetchone()
            clean_rows = int(drow["clean"] or 0)
            excluded_rows = int(drow["excluded"] or 0)

            edges_2 = _compute_grid(cur, win_start, win_end, three_axis=False)
            edges_3 = _compute_grid(cur, win_start, win_end, three_axis=True)
            all_edges = edges_2 + edges_3

            verdict_counts: dict = {}
            for e in all_edges:
                verdict_counts[e.verdict] = verdict_counts.get(e.verdict, 0) + 1

            if not dry_run:
                _upsert(cur, all_edges, computed_at)
                if prune_stale:
                    cur.execute(
                        "DELETE FROM scanner_cell_edge WHERE computed_at < %(ts)s",
                        {"ts": computed_at},
                    )
                conn.commit()

        return RunSummary(
            window_days=window_days,
            window_start=win_start,
            window_end=win_end,
            clean_rows=clean_rows,
            excluded_rows=excluded_rows,
            rows_2axis=len(edges_2),
            rows_3axis=len(edges_3),
            verdict_counts=verdict_counts,
            dry_run=dry_run,
        )
    finally:
        if own_conn:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_top(db_url: str, limit: int = 25) -> None:
    conn = psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT scanner_name, trend_state, vol_regime,
                       COALESCE(liquidity_tier, '-') AS liq,
                       n, wins, pf, win_rate, avg_pnl_pct, calendar_days, verdict
                FROM scanner_cell_edge
                ORDER BY n DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    hdr = f"{'scanner':<32} {'trend':<6} {'vol':<7} {'liq':<7} {'n':>5} {'w':>4} {'pf':>8} {'wr':>6} {'avg%':>7} {'cd':>4} {'verdict':<12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pf = "inf" if r["pf"] is None else f"{float(r['pf']):.2f}"
        wr = "" if r["win_rate"] is None else f"{float(r['win_rate']):.2f}"
        avg = "" if r["avg_pnl_pct"] is None else f"{float(r['avg_pnl_pct']):.2f}"
        print(
            f"{r['scanner_name']:<32} {r['trend_state']:<6} {r['vol_regime']:<7} "
            f"{r['liq']:<7} {r['n']:>5} {r['wins']:>4} {pf:>8} {wr:>6} {avg:>7} "
            f"{(r['calendar_days'] or 0):>4} {r['verdict']:<12}"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scanner cell-edge analyzer")
    ap.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    ap.add_argument("--db-url", default=DB_URL)
    ap.add_argument("--dry-run", action="store_true", help="compute but do not write")
    ap.add_argument("--no-prune", action="store_true", help="keep stale (out-of-window) edge rows")
    ap.add_argument("--top", type=int, default=25, help="print top-N cells by n after the run")
    args = ap.parse_args(argv)

    summary = run(
        window_days=args.window_days,
        db_url=args.db_url,
        dry_run=args.dry_run,
        prune_stale=not args.no_prune,
    )

    print("=" * 78)
    print("SCANNER CELL-EDGE ANALYZER")
    print("=" * 78)
    print(f"window          : {summary.window_start} -> {summary.window_end} ({summary.window_days}d)")
    print(f"clean outcomes  : {summary.clean_rows}")
    print(f"excluded (dirty): {summary.excluded_rows}")
    print(f"cells (2-axis)  : {summary.rows_2axis}")
    print(f"cells (3-axis)  : {summary.rows_3axis}")
    print(f"verdicts        : {summary.verdict_counts}")
    print(f"dry_run         : {summary.dry_run}")
    print()

    if not summary.dry_run:
        print(f"TOP {args.top} CELLS BY n")
        print("=" * 78)
        _print_top(args.db_url, limit=args.top)

    return 0


if __name__ == "__main__":
    sys.exit(main())
