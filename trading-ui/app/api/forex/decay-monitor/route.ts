import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const NO_STORE_HEADERS = {
  "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
  Pragma: "no-cache",
  Expires: "0",
};

const num = (v: unknown): number | null => {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

// Mirror of AutoPauseParams defaults (worker auto_pause/config.py). Display
// only — the worker's env overrides are authoritative for enforcement.
const PARAMS = {
  shadow_window: 50,
  shadow_min_outcomes: 30,
  shadow_trip_pf: 0.8,
  shadow_trip_wr_drop: 0.12,
  shadow_max_consecutive_losses: 8,
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const env = searchParams.get("env") || "demo";

  const scClient = await strategyConfigPool.connect();
  try {
    // -----------------------------------------------------------------
    // (a) Enrolled cells + pause state (strategy_config DB)
    // -----------------------------------------------------------------
    const cellsRes = await scClient.query(
      `
      SELECT
        e.strategy, e.epic, e.config_set, e.eligible, e.trip_source,
        e.baseline_pf::float8         AS baseline_pf,
        e.baseline_shadow_pf::float8  AS baseline_shadow_pf,
        e.baseline_shadow_wr::float8  AS baseline_shadow_wr,
        e.baseline_shadow_n           AS baseline_shadow_n,
        e.baseline_source, e.auto_resume, e.notes,
        s.state                       AS pause_state,
        s.paused_at, s.pause_reason,
        s.resume_proposed_at, s.resume_proposal_count
      FROM auto_pause_eligibility e
      LEFT JOIN auto_pause_state s
        ON s.strategy = e.strategy AND s.epic = e.epic
       AND s.config_set = e.config_set
      WHERE e.eligible = TRUE AND e.config_set = $1
      ORDER BY e.strategy, e.epic
      `,
      [env]
    );

    // -----------------------------------------------------------------
    // (b) Recent events feed (strategy_config DB)
    // -----------------------------------------------------------------
    const eventsRes = await scClient.query(
      `
      SELECT id, event_type, strategy, epic, config_set, reason, metrics,
             created_at, notified_at
      FROM auto_pause_events
      WHERE config_set = $1
      ORDER BY created_at DESC
      LIMIT 50
      `,
      [env]
    );

    // -----------------------------------------------------------------
    // (c) Rolling shadow metrics per enrolled cell (forex DB) — one query:
    //     last-N resolved ref-grid outcomes per cell (window function) plus a
    //     30d daily sparkline. PF NULLIF-guarded, everything ::float8.
    // -----------------------------------------------------------------
    const strategies = cellsRes.rows.map((r) => String(r.strategy));
    const epics = cellsRes.rows.map((r) => String(r.epic));

    let rollingRows: Record<string, unknown>[] = [];
    let sparkRows: Record<string, unknown>[] = [];
    if (cellsRes.rows.length > 0) {
      const rollingRes = await forexPool.query(
        `
        WITH cells AS (
          SELECT * FROM unnest($1::text[], $2::text[]) AS c(strategy, epic)
        ),
        ranked AS (
          SELECT o.strategy, o.epic, o.ref_pnl_pips,
                 row_number() OVER (
                   PARTITION BY o.strategy, o.epic
                   ORDER BY o.signal_timestamp DESC
                 ) AS rn
          FROM monitor_only_outcomes o
          JOIN cells c ON c.strategy = o.strategy AND c.epic = o.epic
          WHERE coalesce(o.environment, $3) = $3
            AND o.status = 'RESOLVED'
            AND o.ref_pnl_pips IS NOT NULL
        )
        SELECT
          strategy, epic,
          count(*)::int AS n,
          (count(*) FILTER (WHERE ref_pnl_pips > 0))::float8
            / NULLIF(count(*), 0)                              AS win_rate,
          (sum(ref_pnl_pips) FILTER (WHERE ref_pnl_pips > 0)
            / NULLIF(ABS(sum(ref_pnl_pips) FILTER (WHERE ref_pnl_pips < 0)), 0)
          )::float8                                            AS pf,
          avg(ref_pnl_pips)::float8                            AS expectancy
        FROM ranked
        WHERE rn <= $4
        GROUP BY strategy, epic
        `,
        [strategies, epics, env, PARAMS.shadow_window]
      );
      rollingRows = rollingRes.rows;

      const sparkRes = await forexPool.query(
        `
        WITH cells AS (
          SELECT * FROM unnest($1::text[], $2::text[]) AS c(strategy, epic)
        )
        SELECT o.strategy, o.epic,
               date_trunc('day', o.signal_timestamp)::date AS day,
               count(*)::int AS n,
               (count(*) FILTER (WHERE o.ref_pnl_pips > 0))::int AS wins,
               sum(o.ref_pnl_pips)::float8 AS net_pips
        FROM monitor_only_outcomes o
        JOIN cells c ON c.strategy = o.strategy AND c.epic = o.epic
        WHERE coalesce(o.environment, $3) = $3
          AND o.status = 'RESOLVED'
          AND o.ref_pnl_pips IS NOT NULL
          AND o.signal_timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY o.strategy, o.epic, day
        ORDER BY day ASC
        `,
        [strategies, epics, env]
      );
      sparkRows = sparkRes.rows;
    }

    const rollingByCell = new Map<string, Record<string, unknown>>();
    for (const r of rollingRows) rollingByCell.set(`${r.strategy}__${r.epic}`, r);
    const sparksByCell = new Map<string, { day: string; n: number; wins: number; net_pips: number | null }[]>();
    for (const r of sparkRows) {
      const key = `${r.strategy}__${r.epic}`;
      const list = sparksByCell.get(key) ?? [];
      list.push({
        day: String(r.day),
        n: num(r.n) ?? 0,
        wins: num(r.wins) ?? 0,
        net_pips: num(r.net_pips),
      });
      sparksByCell.set(key, list);
    }

    const cells = cellsRes.rows.map((r) => {
      const key = `${r.strategy}__${r.epic}`;
      const roll = rollingByCell.get(key);
      const pf = roll ? num(roll.pf) : null;
      const wr = roll ? num(roll.win_rate) : null;
      const n = roll ? num(roll.n) ?? 0 : 0;
      const baseWr = num(r.baseline_shadow_wr);
      const tripWr = baseWr == null ? null : baseWr - PARAMS.shadow_trip_wr_drop;
      // Distance-to-trip on the WR leg (PF leg is absolute). Positive = headroom.
      const wrHeadroom = wr != null && tripWr != null ? wr - tripWr : null;
      const pfHeadroom = pf != null ? pf - PARAMS.shadow_trip_pf : null;
      const wouldTrip =
        n >= PARAMS.shadow_min_outcomes &&
        pf != null && pf < PARAMS.shadow_trip_pf &&
        wr != null && tripWr != null && wr < tripWr;
      return {
        strategy: String(r.strategy),
        epic: String(r.epic),
        config_set: String(r.config_set),
        trip_source: String(r.trip_source),
        auto_resume: Boolean(r.auto_resume),
        baseline_pf: num(r.baseline_pf),
        baseline_shadow_pf: num(r.baseline_shadow_pf),
        baseline_shadow_wr: baseWr,
        baseline_shadow_n: num(r.baseline_shadow_n),
        baseline_source: r.baseline_source == null ? null : String(r.baseline_source),
        notes: r.notes == null ? null : String(r.notes),
        pause_state: r.pause_state == null ? "active" : String(r.pause_state),
        paused_at: r.paused_at == null ? null : new Date(r.paused_at as string).toISOString(),
        pause_reason: r.pause_reason == null ? null : String(r.pause_reason),
        resume_proposed_at:
          r.resume_proposed_at == null ? null : new Date(r.resume_proposed_at as string).toISOString(),
        resume_proposal_count: num(r.resume_proposal_count) ?? 0,
        rolling: {
          n,
          pf,
          win_rate: wr,
          expectancy: roll ? num(roll.expectancy) : null,
          pf_headroom: pfHeadroom,
          wr_headroom: wrHeadroom,
          would_trip: wouldTrip,
        },
        spark: sparksByCell.get(key) ?? [],
      };
    });

    const events = eventsRes.rows.map((r) => ({
      id: num(r.id),
      event_type: String(r.event_type),
      strategy: String(r.strategy),
      epic: String(r.epic),
      config_set: String(r.config_set),
      reason: r.reason == null ? null : String(r.reason),
      metrics: r.metrics ?? null,
      created_at: new Date(r.created_at as string).toISOString(),
      notified: r.notified_at != null,
    }));

    return NextResponse.json(
      {
        meta: {
          env,
          params: PARAMS,
          generated_at: new Date().toISOString(),
        },
        cells,
        events,
      },
      { headers: NO_STORE_HEADERS }
    );
  } catch (error) {
    console.error("decay-monitor query failed", error);
    return NextResponse.json(
      { error: "Failed to load decay-monitor data" },
      { status: 500, headers: NO_STORE_HEADERS }
    );
  } finally {
    scClient.release();
  }
}
