import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 1;
const DEFAULT_LIMIT = 25;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_DAYS;
  if (parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

function parseLimit(value: string | null) {
  if (!value) return DEFAULT_LIMIT;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_LIMIT;
  if (parsed <= 0) return DEFAULT_LIMIT;
  return parsed;
}

function parsePage(value: string | null) {
  if (!value) return 1;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 1;
  if (parsed <= 0) return 1;
  return parsed;
}

type AlertFilters = {
  days: number;
  status: string | null;
  strategy: string | null;
  pair: string | null;
};

function buildFilters(filters: AlertFilters) {
  const clauses = ["alert_timestamp >= NOW() - ($1::int || ' days')::interval"];
  const params: Array<string | number> = [filters.days];
  let idx = 2;

  if (filters.status === "Approved") {
    clauses.push("(claude_approved = TRUE OR claude_decision = 'APPROVE')");
  } else if (filters.status === "Rejected") {
    clauses.push(
      "(claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED')"
    );
  }

  if (filters.strategy && filters.strategy !== "All") {
    clauses.push(`strategy = $${idx}`);
    params.push(filters.strategy);
    idx += 1;
  }

  if (filters.pair && filters.pair !== "All") {
    clauses.push(`(pair = $${idx} OR epic ILIKE $${idx + 1})`);
    params.push(filters.pair, `%${filters.pair}%`);
    idx += 2;
  }

  return {
    whereSql: `WHERE ${clauses.join(" AND ")}`,
    params
  };
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const limit = parseLimit(searchParams.get("limit"));
  const page = parsePage(searchParams.get("page"));
  const status = searchParams.get("status");
  const strategy = searchParams.get("strategy");
  const pair = searchParams.get("pair");
  const offset = (page - 1) * limit;

  const { whereSql, params } = buildFilters({ days, status, strategy, pair });

  try {
    const [filtersResult, pairsResult, statsResult, countResult, alertsResult] = await Promise.all([
      forexPool.query(
        `SELECT DISTINCT strategy FROM alert_history WHERE strategy IS NOT NULL ORDER BY strategy`
      ),
      forexPool.query(
        `SELECT DISTINCT pair FROM alert_history WHERE pair IS NOT NULL ORDER BY pair`
      ),
      forexPool.query(
        `
        SELECT
          COUNT(*) as total_alerts,
          COUNT(CASE WHEN claude_approved = TRUE OR claude_decision = 'APPROVE' THEN 1 END) as approved,
          COUNT(CASE WHEN claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED' THEN 1 END) as rejected,
          ROUND(AVG(claude_score)::numeric, 2) as avg_score
        FROM alert_history
        ${whereSql}
        `,
        params
      ),
      forexPool.query(
        `
        SELECT COUNT(*) as total
        FROM alert_history
        ${whereSql}
        `,
        params
      ),
      forexPool.query(
        `
        SELECT
          id,
          alert_timestamp,
          epic,
          pair,
          signal_type,
          strategy,
          price,
          market_session,
          claude_score,
          claude_decision,
          claude_approved,
          claude_reason,
          claude_mode,
          claude_raw_response,
          vision_chart_url,
          status,
          alert_level,
          htf_candle_direction,
          htf_candle_direction_prev
        FROM alert_history
        ${whereSql}
        ORDER BY alert_timestamp DESC
        LIMIT $${params.length + 1} OFFSET $${params.length + 2}
        `,
        [...params, limit, offset]
      )
    ]);

    const strategies = ["All", ...(filtersResult.rows ?? []).map((row) => row.strategy)];
    const pairs = ["All", ...(pairsResult.rows ?? []).map((row) => row.pair)];

    const statsRow = statsResult.rows?.[0] ?? {};
    const totalAlerts = Number(countResult.rows?.[0]?.total ?? 0);
    const approved = Number(statsRow.approved ?? 0);
    const rejected = Number(statsRow.rejected ?? 0);
    const avgScore = Number(statsRow.avg_score ?? 0);

    const alerts = (alertsResult.rows ?? []).map((row) => ({
      ...row,
      price: row.price == null ? null : Number(row.price),
      claude_score: row.claude_score == null ? null : Number(row.claude_score),
      claude_approved: row.claude_approved == null ? null : Boolean(row.claude_approved)
    }));

    return NextResponse.json({
      filters: { strategies, pairs },
      stats: {
        total_alerts: totalAlerts,
        approved,
        rejected,
        avg_score: avgScore,
        approval_rate: totalAlerts ? (approved / totalAlerts) * 100 : 0
      },
      alerts,
      page,
      total_pages: totalAlerts ? Math.ceil(totalAlerts / limit) : 1,
      total_alerts: totalAlerts
    });
  } catch (error) {
    console.error("Failed to load alert history", error);
    return NextResponse.json(
      { error: "Failed to load alert history" },
      { status: 500 }
    );
  }
}
