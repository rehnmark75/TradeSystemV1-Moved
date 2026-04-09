import { NextResponse } from "next/server";
import { forexPool } from "../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 7;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));

  try {
    const result = await forexPool.query(`
      WITH base AS (
        SELECT step, pair, signal_type, confidence_score, rr_ratio,
               lpf_penalty, lpf_triggered_rules, created_at
        FROM validator_rejections
        WHERE created_at >= NOW() - INTERVAL '${days} days'
      ),
      totals AS (
        SELECT COUNT(*) AS total, COUNT(DISTINCT pair) AS unique_pairs
        FROM base
      ),
      by_step AS (
        SELECT step, COUNT(*) AS cnt
        FROM base GROUP BY step
      ),
      by_pair AS (
        SELECT pair, COUNT(*) AS cnt
        FROM base WHERE pair IS NOT NULL
        GROUP BY pair ORDER BY cnt DESC LIMIT 10
      ),
      by_direction AS (
        SELECT signal_type, COUNT(*) AS cnt
        FROM base WHERE signal_type IS NOT NULL
        GROUP BY signal_type
      ),
      top_step AS (
        SELECT step FROM by_step ORDER BY cnt DESC LIMIT 1
      ),
      top_pair AS (
        SELECT pair FROM by_pair LIMIT 1
      ),
      lpf_stats AS (
        SELECT
          COUNT(*) AS total_lpf,
          AVG(lpf_penalty) AS avg_penalty,
          MAX(lpf_penalty) AS max_penalty
        FROM base WHERE step = 'LPF'
      )
      SELECT
        t.total,
        t.unique_pairs,
        ts.step AS top_step,
        tp.pair AS top_pair,
        ls.total_lpf,
        ls.avg_penalty,
        ls.max_penalty,
        (SELECT json_object_agg(step, cnt) FROM by_step) AS by_step,
        (SELECT json_agg(json_build_object('pair', pair, 'count', cnt)) FROM by_pair) AS by_pair,
        (SELECT json_object_agg(signal_type, cnt) FROM by_direction) AS by_direction
      FROM totals t
      CROSS JOIN lpf_stats ls
      LEFT JOIN top_step ts ON true
      LEFT JOIN top_pair tp ON true
    `);

    const row = result.rows[0];
    return NextResponse.json({
      total: Number(row?.total ?? 0),
      unique_pairs: Number(row?.unique_pairs ?? 0),
      top_step: row?.top_step ?? "N/A",
      top_pair: row?.top_pair ?? "N/A",
      total_lpf: Number(row?.total_lpf ?? 0),
      avg_lpf_penalty: row?.avg_penalty ? Number(row.avg_penalty).toFixed(2) : null,
      max_lpf_penalty: row?.max_penalty ? Number(row.max_penalty).toFixed(2) : null,
      by_step: row?.by_step ?? {},
      by_pair: row?.by_pair ?? [],
      by_direction: row?.by_direction ?? {}
    });
  } catch (error) {
    console.error("validator-rejections/stats error:", error);
    return NextResponse.json({ error: "Failed to load stats" }, { status: 500 });
  }
}
