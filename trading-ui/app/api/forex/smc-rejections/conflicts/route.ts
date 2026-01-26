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
    const statsResult = await forexPool.query(
      `
      SELECT
        COUNT(*) as total,
        COUNT(DISTINCT epic) as unique_pairs,
        COUNT(DISTINCT market_session) as sessions_affected
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${days} days'
        AND rejection_stage = 'SMC_CONFLICT'
      `
    );

    const pairResult = await forexPool.query(
      `
      SELECT pair, COUNT(*) as count
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${days} days'
        AND rejection_stage = 'SMC_CONFLICT'
      GROUP BY pair
      ORDER BY count DESC
      `
    );

    const sessionResult = await forexPool.query(
      `
      SELECT market_session, COUNT(*) as count
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${days} days'
        AND rejection_stage = 'SMC_CONFLICT'
        AND market_session IS NOT NULL
      GROUP BY market_session
      ORDER BY count DESC
      `
    );

    const reasonResult = await forexPool.query(
      `
      SELECT rejection_reason, COUNT(*) as count
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${days} days'
        AND rejection_stage = 'SMC_CONFLICT'
      GROUP BY rejection_reason
      ORDER BY count DESC
      LIMIT 10
      `
    );

    const detailsResult = await forexPool.query(
      `
      SELECT
        id,
        scan_timestamp,
        epic,
        pair,
        rejection_reason,
        attempted_direction,
        current_price,
        market_hour,
        market_session,
        potential_entry,
        potential_stop_loss,
        potential_take_profit,
        potential_risk_pips,
        potential_reward_pips,
        potential_rr_ratio,
        confidence_score,
        rejection_details
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${days} days'
        AND rejection_stage = 'SMC_CONFLICT'
      ORDER BY scan_timestamp DESC
      LIMIT 500
      `
    );

    const statsRow = statsResult.rows[0] ?? {};
    return NextResponse.json({
      stats: {
        total: Number(statsRow.total ?? 0),
        unique_pairs: Number(statsRow.unique_pairs ?? 0),
        sessions_affected: Number(statsRow.sessions_affected ?? 0)
      },
      by_pair: pairResult.rows ?? [],
      by_session: sessionResult.rows ?? [],
      top_reasons: reasonResult.rows ?? [],
      details: detailsResult.rows ?? []
    });
  } catch (error) {
    console.error("Failed to load SMC conflict data", error);
    return NextResponse.json(
      { error: "Failed to load SMC conflict data" },
      { status: 500 }
    );
  }
}
