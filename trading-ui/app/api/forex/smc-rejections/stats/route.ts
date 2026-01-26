import { NextResponse } from "next/server";
import { forexPool } from "../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 7;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_DAYS;
  if (parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));

  try {
    const result = await forexPool.query(
      `
      WITH base_data AS (
        SELECT
          epic,
          pair,
          rejection_stage,
          confidence_score
        FROM smc_simple_rejections
        WHERE scan_timestamp >= NOW() - INTERVAL '${days} days'
      ),
      stage_counts AS (
        SELECT
          rejection_stage,
          COUNT(*) as stage_count
        FROM base_data
        GROUP BY rejection_stage
      ),
      totals AS (
        SELECT
          COUNT(*) as total,
          COUNT(DISTINCT epic) as unique_pairs
        FROM base_data
      ),
      near_misses AS (
        SELECT COUNT(*) as near_miss_count
        FROM base_data
        WHERE rejection_stage = 'CONFIDENCE'
          AND confidence_score >= 0.45
      ),
      smc_conflicts AS (
        SELECT COUNT(*) as conflict_count
        FROM base_data
        WHERE rejection_stage = 'SMC_CONFLICT'
      ),
      top_pair AS (
        SELECT pair, COUNT(*) as pair_count
        FROM base_data
        WHERE pair IS NOT NULL
        GROUP BY pair
        ORDER BY pair_count DESC
        LIMIT 1
      )
      SELECT
        t.total,
        t.unique_pairs,
        nm.near_miss_count,
        sc.conflict_count,
        tp.pair as most_rejected_pair,
        (SELECT json_object_agg(rejection_stage, stage_count) FROM stage_counts) as by_stage
      FROM totals t
      CROSS JOIN near_misses nm
      CROSS JOIN smc_conflicts sc
      LEFT JOIN top_pair tp ON true
      `
    );

    const row = result.rows[0];
    return NextResponse.json({
      total: Number(row?.total ?? 0),
      unique_pairs: Number(row?.unique_pairs ?? 0),
      near_misses: Number(row?.near_miss_count ?? 0),
      smc_conflicts: Number(row?.conflict_count ?? 0),
      most_rejected_pair: row?.most_rejected_pair ?? "N/A",
      by_stage: row?.by_stage ?? {}
    });
  } catch (error) {
    console.error("Failed to load SMC rejection stats", error);
    return NextResponse.json(
      { error: "Failed to load SMC rejection stats" },
      { status: 500 }
    );
  }
}
