import { NextResponse } from "next/server";
import { pool } from "../../../lib/db";

export const dynamic = "force-dynamic";

const NO_STORE_HEADERS = {
  "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
  Pragma: "no-cache",
  Expires: "0",
};

// Monitor-only scanners — mirrored from /api/signals/top. These still emit/log
// signals but are excluded from the tradable day-trade pool, so a 'block'
// verdict on one of them would NOT be a real would-drop (they are already not
// traded). The would-drop impact section subtracts them.
const MONITOR_ONLY_SCANNERS = ["high_retest", "zlma_trend", "regime_adaptive_composite"];

const num = (v: unknown): number | null => {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const rawWindow = Number(searchParams.get("windowDays") || 120);
  const windowDays = Number.isFinite(rawWindow)
    ? Math.min(730, Math.max(1, Math.floor(rawWindow)))
    : 120;

  const client = await pool.connect();
  try {
    // ---------------------------------------------------------------------
    // (a) EDGE MAP — 2-axis grid only (liquidity_tier IS NULL AND market_regime
    //     IS NULL). pf/win_rate/avg_pnl_pct cast ::float8 (pg numeric -> JS
    //     string gotcha, else .toFixed() on the client crashes).
    // ---------------------------------------------------------------------
    const edgeMapRes = await client.query(
      `
      SELECT
        scanner_name,
        trend_state,
        vol_regime,
        n,
        pf::float8          AS pf,
        win_rate::float8    AS win_rate,
        avg_pnl_pct::float8 AS avg_pnl_pct,
        calendar_days,
        verdict,
        computed_at
      FROM scanner_cell_edge
      WHERE liquidity_tier IS NULL
        AND market_regime IS NULL
      ORDER BY scanner_name, trend_state, vol_regime
      `
    );
    const edgeMap = edgeMapRes.rows.map((r) => ({
      scanner_name: String(r.scanner_name),
      trend_state: r.trend_state == null ? null : String(r.trend_state),
      vol_regime: r.vol_regime == null ? null : String(r.vol_regime),
      n: num(r.n),
      pf: num(r.pf),
      win_rate: num(r.win_rate),
      avg_pnl_pct: num(r.avg_pnl_pct),
      calendar_days: num(r.calendar_days),
      verdict: r.verdict == null ? null : String(r.verdict),
      computed_at: r.computed_at == null ? null : new Date(r.computed_at as string).toISOString(),
    }));

    // ---------------------------------------------------------------------
    // (b) VERDICT VALIDATION — the headline. For closed/realized signals in the
    //     window with a cell (trend_state + vol_regime), joined to their cell's
    //     verdict on the 2-axis grid, grouped by verdict. Contaminated rows
    //     excluded (status NOT IN ('data_error','invalid') AND |pnl| <= 100).
    //     PF computed in SQL with divide-by-zero guard (NULLIF -> null when no
    //     losing pnl); all numerics ::float8.
    // ---------------------------------------------------------------------
    const verdictRes = await client.query(
      `
      WITH closed AS (
        SELECT
          s.scanner_name,
          s.trend_state,
          s.vol_regime,
          s.realized_pnl_pct AS pnl
        FROM stock_scanner_signals s
        WHERE s.signal_timestamp >= NOW() - ($1 * INTERVAL '1 day')
          AND s.trend_state IS NOT NULL
          AND s.vol_regime IS NOT NULL
          AND s.realized_pnl_pct IS NOT NULL
          AND s.status NOT IN ('data_error', 'invalid')
          AND ABS(s.realized_pnl_pct) <= 100
      ),
      joined AS (
        SELECT c.pnl, e.verdict
        FROM closed c
        JOIN scanner_cell_edge e
          ON e.scanner_name = c.scanner_name
         AND e.trend_state  = c.trend_state
         AND e.vol_regime   = c.vol_regime
         AND e.liquidity_tier IS NULL
         AND e.market_regime  IS NULL
      )
      SELECT
        verdict,
        count(*)::int                                        AS n,
        count(*) FILTER (WHERE pnl > 0)::int                 AS wins,
        (count(*) FILTER (WHERE pnl > 0))::float8 / count(*) AS win_rate,
        (sum(pnl) FILTER (WHERE pnl > 0)
          / NULLIF(ABS(sum(pnl) FILTER (WHERE pnl < 0)), 0))::float8 AS pf,
        avg(pnl)::float8                                     AS avg_pnl_pct,
        (sum(pnl) FILTER (WHERE pnl > 0) > 0
          AND COALESCE(sum(pnl) FILTER (WHERE pnl < 0), 0) = 0) AS pf_is_inf
      FROM joined
      GROUP BY verdict
      ORDER BY verdict
      `,
      [windowDays]
    );
    const verdictValidation = verdictRes.rows.map((r) => ({
      verdict: r.verdict == null ? "unmapped" : String(r.verdict),
      n: num(r.n) ?? 0,
      wins: num(r.wins) ?? 0,
      win_rate: num(r.win_rate),
      // pf null when there were no losing trades but there were winners -> flag
      // as "inf" so the UI can distinguish "no data" from "no losses".
      pf: r.pf_is_inf === true ? "inf" : num(r.pf),
      avg_pnl_pct: num(r.avg_pnl_pct),
    }));

    // ---------------------------------------------------------------------
    // (c) WOULD-DROP IMPACT — recent tradable signals (scanner NOT monitor-only)
    //     whose cell verdict is 'block'. These are currently tradable but the
    //     router would drop them if enforced. Returns the individual rows plus a
    //     count-by-scanner summary.
    // ---------------------------------------------------------------------
    const monitorList = MONITOR_ONLY_SCANNERS.map((s) => `'${s}'`).join(", ");
    const wouldDropRes = await client.query(
      `
      SELECT
        s.ticker,
        s.scanner_name,
        s.trend_state,
        s.vol_regime,
        (s.scanner_name || '|' || s.trend_state || '|' || s.vol_regime) AS cell_key,
        e.pf::float8         AS cell_pf,
        e.n                  AS cell_n,
        s.signal_timestamp,
        s.status,
        -- only surface realized PnL for genuinely closed rows; a 'triggered'/
        -- 'active' row may carry a placeholder 0 that would misread as a flat exit.
        (CASE WHEN s.status = 'closed' THEN s.realized_pnl_pct ELSE NULL END)::float8 AS realized_pnl_pct
      FROM stock_scanner_signals s
      JOIN scanner_cell_edge e
        ON e.scanner_name = s.scanner_name
       AND e.trend_state  = s.trend_state
       AND e.vol_regime   = s.vol_regime
       AND e.liquidity_tier IS NULL
       AND e.market_regime  IS NULL
      WHERE s.signal_timestamp >= NOW() - ($1 * INTERVAL '1 day')
        AND s.trend_state IS NOT NULL
        AND s.vol_regime IS NOT NULL
        AND e.verdict = 'block'
        ${MONITOR_ONLY_SCANNERS.length ? `AND s.scanner_name NOT IN (${monitorList})` : ""}
      ORDER BY s.signal_timestamp DESC
      LIMIT 1000
      `,
      [windowDays]
    );
    const wouldDropSignals = wouldDropRes.rows.map((r) => ({
      ticker: String(r.ticker),
      scanner_name: String(r.scanner_name),
      cell_key: String(r.cell_key),
      cell_pf: num(r.cell_pf),
      cell_n: num(r.cell_n),
      signal_timestamp:
        r.signal_timestamp == null ? null : new Date(r.signal_timestamp as string).toISOString(),
      status: r.status == null ? null : String(r.status),
      realized_pnl_pct: num(r.realized_pnl_pct),
    }));
    // count-by-scanner summary (closed count + realized pnl where available)
    const byScannerMap = new Map<
      string,
      { scanner_name: string; count: number; closed: number; sum_realized_pnl_pct: number }
    >();
    for (const row of wouldDropSignals) {
      const cur =
        byScannerMap.get(row.scanner_name) ??
        { scanner_name: row.scanner_name, count: 0, closed: 0, sum_realized_pnl_pct: 0 };
      cur.count += 1;
      if (row.realized_pnl_pct != null) {
        cur.closed += 1;
        cur.sum_realized_pnl_pct += row.realized_pnl_pct;
      }
      byScannerMap.set(row.scanner_name, cur);
    }
    const wouldDropByScanner = [...byScannerMap.values()]
      .map((v) => ({
        ...v,
        avg_realized_pnl_pct: v.closed > 0 ? v.sum_realized_pnl_pct / v.closed : null,
      }))
      .sort((a, b) => b.count - a.count);

    return NextResponse.json(
      {
        meta: {
          window_days: windowDays,
          monitor_only_scanners: MONITOR_ONLY_SCANNERS,
          shadow_mode: true,
          generated_at: new Date().toISOString(),
        },
        edgeMap,
        verdictValidation,
        wouldDropImpact: {
          total: wouldDropSignals.length,
          byScanner: wouldDropByScanner,
          signals: wouldDropSignals,
        },
      },
      { headers: NO_STORE_HEADERS }
    );
  } catch (error) {
    console.error("cell-router monitor query failed", error);
    return NextResponse.json(
      { error: "Failed to load cell-router monitor data" },
      { status: 500, headers: NO_STORE_HEADERS }
    );
  } finally {
    client.release();
  }
}
