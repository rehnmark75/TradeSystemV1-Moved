import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

function parseDays(v: string | null): number {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : 7;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const strategy = searchParams.get("strategy") ?? "ALL";
  const epic = searchParams.get("epic") ?? "ALL";

  const conditions: string[] = [
    `scan_timestamp >= NOW() - INTERVAL '${days} days'`,
  ];
  const params: unknown[] = [];

  if (strategy !== "ALL") {
    params.push(strategy);
    conditions.push(`strategy = $${params.length}`);
  }
  if (epic !== "ALL") {
    params.push(epic);
    conditions.push(`epic = $${params.length}`);
  }

  const where = conditions.join(" AND ");

  try {
    const result = await strategyConfigPool.query(
      `
      WITH raw_base AS (
        SELECT strategy, epic, pair, stage, direction, hour_utc, session
        FROM v_strategy_rejections_unified
        WHERE ${where}
      ),
      base AS (
        SELECT *
        FROM raw_base
        WHERE NOT (strategy = 'RANGE_FADE' AND stage = 'NO_TRIGGER')
      ),
      no_setup AS (
        SELECT COUNT(*) AS excluded_no_setup
        FROM raw_base
        WHERE strategy = 'RANGE_FADE' AND stage = 'NO_TRIGGER'
      ),
      totals AS (
        SELECT COUNT(*) AS total, COUNT(DISTINCT epic) AS unique_pairs
        FROM base
      ),
      by_strategy AS (
        SELECT strategy, COUNT(*) AS cnt
        FROM base GROUP BY strategy
      ),
      by_stage AS (
        SELECT stage, COUNT(*) AS cnt
        FROM base GROUP BY stage ORDER BY cnt DESC
      ),
      by_hour AS (
        SELECT hour_utc, COUNT(*) AS cnt
        FROM base WHERE hour_utc IS NOT NULL
        GROUP BY hour_utc ORDER BY hour_utc
      ),
      by_session AS (
        SELECT session, COUNT(*) AS cnt
        FROM base WHERE session IS NOT NULL
        GROUP BY session ORDER BY cnt DESC
      ),
      by_direction AS (
        SELECT direction, COUNT(*) AS cnt
        FROM base WHERE direction IS NOT NULL
        GROUP BY direction
      ),
      top_pair AS (
        SELECT pair, COUNT(*) AS cnt
        FROM base WHERE pair IS NOT NULL
        GROUP BY pair ORDER BY cnt DESC LIMIT 1
      )
      SELECT
        t.total,
        t.unique_pairs,
        tp.pair AS most_rejected_pair,
        (SELECT json_object_agg(strategy, cnt) FROM by_strategy) AS by_strategy,
        (SELECT json_object_agg(stage, cnt) FROM by_stage) AS by_stage,
        (SELECT json_object_agg(hour_utc, cnt) FROM by_hour) AS by_hour,
        (SELECT json_object_agg(session, cnt) FROM by_session) AS by_session,
        (SELECT json_object_agg(direction, cnt) FROM by_direction) AS by_direction,
        ns.excluded_no_setup
      FROM totals t
      CROSS JOIN no_setup ns
      LEFT JOIN top_pair tp ON true
      `,
      params
    );

    const row = result.rows[0];
    return NextResponse.json({
      total: Number(row?.total ?? 0),
      unique_pairs: Number(row?.unique_pairs ?? 0),
      most_rejected_pair: row?.most_rejected_pair ?? "N/A",
      by_strategy: row?.by_strategy ?? {},
      by_stage: row?.by_stage ?? {},
      by_hour: row?.by_hour ?? {},
      by_session: row?.by_session ?? {},
      by_direction: row?.by_direction ?? {},
      excluded_no_setup: Number(row?.excluded_no_setup ?? 0),
    });
  } catch (error) {
    console.error("Failed to load strategy rejection stats", error);
    return NextResponse.json(
      { error: "Failed to load stats" },
      { status: 500 }
    );
  }
}
