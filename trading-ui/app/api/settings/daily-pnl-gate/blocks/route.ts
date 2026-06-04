import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const environment = searchParams.get("environment") ?? "demo";
  const days = Math.min(parseInt(searchParams.get("days") ?? "30", 10), 90);

  try {
    const result = await strategyConfigPool.query(
      `SELECT id, blocked_at, environment, limit_hit,
              daily_pnl_sek::float8 AS daily_pnl_sek,
              profit_limit_sek::float8 AS profit_limit_sek,
              loss_limit_sek::float8 AS loss_limit_sek,
              epic, direction, alert_id, trigger_source
       FROM daily_pnl_gate_blocks
       WHERE environment = $1
         AND blocked_at >= NOW() - ($2 || ' days')::interval
       ORDER BY blocked_at DESC
       LIMIT 200`,
      [environment, days]
    );

    const summary = await strategyConfigPool.query(
      `SELECT limit_hit, COUNT(*)::int AS count, DATE(blocked_at) AS day
       FROM daily_pnl_gate_blocks
       WHERE environment = $1
         AND blocked_at >= NOW() - ($2 || ' days')::interval
       GROUP BY limit_hit, DATE(blocked_at)
       ORDER BY day DESC`,
      [environment, days]
    );

    return NextResponse.json({ rows: result.rows, summary: summary.rows });
  } catch (error) {
    console.error("Failed to load daily PnL gate blocks", error);
    return NextResponse.json({ error: "Failed to load blocks" }, { status: 500 });
  }
}
