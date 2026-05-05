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

  const conditions: string[] = [
    `scan_timestamp >= NOW() - INTERVAL '${days} days'`,
  ];
  const params: unknown[] = [];

  if (strategy !== "ALL") {
    params.push(strategy);
    conditions.push(`strategy = $${params.length}`);
  }

  const where = conditions.join(" AND ");

  try {
    const result = await strategyConfigPool.query(
      `
      SELECT
        strategy,
        stage,
        COUNT(*)                                                     AS total,
        COUNT(DISTINCT epic)                                         AS pairs_affected,
        ROUND(
          COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY strategy),
          1
        )                                                            AS pct_of_strategy
      FROM v_strategy_rejections_unified
      WHERE ${where}
      GROUP BY strategy, stage
      ORDER BY strategy, total DESC
      `,
      params
    );

    return NextResponse.json({ rows: result.rows });
  } catch (error) {
    console.error("Failed to load top rejection stages", error);
    return NextResponse.json(
      { error: "Failed to load top stages" },
      { status: 500 }
    );
  }
}
