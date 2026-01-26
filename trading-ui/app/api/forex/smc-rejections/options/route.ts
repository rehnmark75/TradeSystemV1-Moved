import { NextResponse } from "next/server";
import { forexPool } from "../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const result = await forexPool.query(
      `
      SELECT
        COALESCE(
          (SELECT json_agg(DISTINCT rejection_stage ORDER BY rejection_stage)
           FROM smc_simple_rejections),
          '[]'::json
        ) as stages,
        COALESCE(
          (SELECT json_agg(DISTINCT pair ORDER BY pair)
           FROM smc_simple_rejections
           WHERE pair IS NOT NULL),
          '[]'::json
        ) as pairs,
        COALESCE(
          (SELECT json_agg(DISTINCT market_session ORDER BY market_session)
           FROM smc_simple_rejections
           WHERE market_session IS NOT NULL),
          '[]'::json
        ) as sessions,
        EXISTS (
          SELECT FROM information_schema.tables
          WHERE table_name = 'smc_simple_rejections'
        ) as table_exists
      `
    );

    const row = result.rows[0];
    if (!row || !row.table_exists) {
      return NextResponse.json({
        stages: ["All"],
        pairs: ["All"],
        sessions: ["All"],
        table_exists: false
      });
    }

    return NextResponse.json({
      stages: ["All", ...(row.stages ?? [])],
      pairs: ["All", ...(row.pairs ?? [])],
      sessions: ["All", ...(row.sessions ?? [])],
      table_exists: true
    });
  } catch (error) {
    console.error("Failed to load SMC rejection options", error);
    return NextResponse.json(
      { error: "Failed to load SMC rejection options" },
      { status: 500 }
    );
  }
}
