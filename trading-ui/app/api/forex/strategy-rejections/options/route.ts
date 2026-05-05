import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const [strategiesRes, epicsRes, stagesRes, hoursRes] = await Promise.all([
      strategyConfigPool.query(
        `SELECT DISTINCT strategy FROM v_strategy_rejections_unified ORDER BY strategy`
      ),
      strategyConfigPool.query(
        `SELECT DISTINCT epic, pair FROM v_strategy_rejections_unified ORDER BY epic`
      ),
      strategyConfigPool.query(
        `SELECT DISTINCT stage FROM v_strategy_rejections_unified ORDER BY stage`
      ),
      strategyConfigPool.query(
        `SELECT DISTINCT hour_utc FROM v_strategy_rejections_unified WHERE hour_utc IS NOT NULL ORDER BY hour_utc`
      ),
    ]);

    return NextResponse.json({
      strategies: strategiesRes.rows.map((r) => r.strategy),
      epics: epicsRes.rows,
      stages: stagesRes.rows.map((r) => r.stage),
      hours: hoursRes.rows.map((r) => Number(r.hour_utc)),
    });
  } catch (error) {
    console.error("Failed to load strategy rejection options", error);
    return NextResponse.json(
      { error: "Failed to load options" },
      { status: 500 }
    );
  }
}
