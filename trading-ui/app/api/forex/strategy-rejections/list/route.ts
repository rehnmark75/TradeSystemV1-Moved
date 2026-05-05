import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

function parseDays(v: string | null): number {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : 7;
}

function parseLimit(v: string | null): number {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 && n <= 500 ? n : 100;
}

function parseOffset(v: string | null): number {
  const n = Number(v);
  return Number.isFinite(n) && n >= 0 ? n : 0;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const strategy = searchParams.get("strategy") ?? "ALL";
  const epic = searchParams.get("epic") ?? "ALL";
  const stage = searchParams.get("stage") ?? "ALL";
  const hour = searchParams.get("hour") ?? "ALL";
  const direction = searchParams.get("direction") ?? "ALL";
  const limit = parseLimit(searchParams.get("limit"));
  const offset = parseOffset(searchParams.get("offset"));

  const conditions: string[] = [
    `scan_timestamp >= NOW() - INTERVAL '${days} days'`,
  ];
  const params: unknown[] = [];

  if (strategy !== "ALL") {
    params.push(strategy);
    conditions.push(`strategy = $${params.length}`);
  }
  if (epic !== "ALL") {
    params.push(epic);
    conditions.push(`epic = $${params.length}`);
  }
  if (stage !== "ALL") {
    params.push(stage);
    conditions.push(`stage = $${params.length}`);
  }
  if (hour !== "ALL") {
    params.push(Number(hour));
    conditions.push(`hour_utc = $${params.length}`);
  }
  if (direction !== "ALL") {
    params.push(direction);
    conditions.push(`direction = $${params.length}`);
  }

  const where = conditions.join(" AND ");

  try {
    const [rowsRes, countRes] = await Promise.all([
      strategyConfigPool.query(
        `
        SELECT strategy, epic, pair, scan_timestamp, stage, reason,
               direction, hour_utc, session, details
        FROM v_strategy_rejections_unified
        WHERE ${where}
        ORDER BY scan_timestamp DESC
        LIMIT ${limit} OFFSET ${offset}
        `,
        params
      ),
      strategyConfigPool.query(
        `SELECT COUNT(*) AS total FROM v_strategy_rejections_unified WHERE ${where}`,
        params
      ),
    ]);

    return NextResponse.json({
      rows: rowsRes.rows,
      total: Number(countRes.rows[0]?.total ?? 0),
      limit,
      offset,
    });
  } catch (error) {
    console.error("Failed to load strategy rejection list", error);
    return NextResponse.json(
      { error: "Failed to load list" },
      { status: 500 }
    );
  }
}
