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
    // Explode the lpf_triggered_rules JSONB array into individual rule counts
    const ruleBreakdown = await forexPool.query(`
      SELECT
        rule_name,
        COUNT(*) AS times_triggered,
        COUNT(DISTINCT pair) AS pairs_affected,
        AVG(vr.lpf_penalty) AS avg_total_penalty
      FROM validator_rejections vr,
           jsonb_array_elements_text(vr.lpf_triggered_rules) AS rule_name
      WHERE vr.step = 'LPF'
        AND vr.created_at >= NOW() - INTERVAL '${days} days'
        AND vr.lpf_triggered_rules IS NOT NULL
      GROUP BY rule_name
      ORDER BY times_triggered DESC
    `);

    const byPair = await forexPool.query(`
      SELECT
        pair,
        COUNT(*) AS total_lpf_blocks,
        AVG(lpf_penalty) AS avg_penalty,
        MAX(lpf_penalty) AS max_penalty
      FROM validator_rejections
      WHERE step = 'LPF'
        AND created_at >= NOW() - INTERVAL '${days} days'
        AND pair IS NOT NULL
      GROUP BY pair
      ORDER BY total_lpf_blocks DESC
    `);

    const hourly = await forexPool.query(`
      SELECT
        EXTRACT(HOUR FROM created_at) AS hour,
        COUNT(*) AS count
      FROM validator_rejections
      WHERE step = 'LPF'
        AND created_at >= NOW() - INTERVAL '${days} days'
      GROUP BY hour
      ORDER BY hour
    `);

    return NextResponse.json({
      rule_breakdown: ruleBreakdown.rows,
      by_pair: byPair.rows,
      hourly: hourly.rows
    });
  } catch (error) {
    console.error("validator-rejections/lpf error:", error);
    return NextResponse.json({ error: "Failed to load LPF detail" }, { status: 500 });
  }
}
