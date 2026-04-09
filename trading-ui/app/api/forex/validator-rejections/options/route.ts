import { NextResponse } from "next/server";
import { forexPool } from "../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const [stepsRes, pairsRes] = await Promise.all([
      forexPool.query(
        `SELECT DISTINCT step FROM validator_rejections
         WHERE created_at >= NOW() - INTERVAL '90 days'
         ORDER BY step`
      ),
      forexPool.query(
        `SELECT DISTINCT pair FROM validator_rejections
         WHERE pair IS NOT NULL AND created_at >= NOW() - INTERVAL '90 days'
         ORDER BY pair`
      )
    ]);

    return NextResponse.json({
      steps: stepsRes.rows.map((r) => r.step),
      pairs: pairsRes.rows.map((r) => r.pair),
      table_exists: true
    });
  } catch (error) {
    console.error("validator-rejections/options error:", error);
    return NextResponse.json({ steps: [], pairs: [], table_exists: false });
  }
}
