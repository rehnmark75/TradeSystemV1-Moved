import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_LIMIT = 2000;
const MAX_LIMIT = 5000;

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const epic = searchParams.get("epic");
  const from = searchParams.get("from");
  const to = searchParams.get("to");
  const limitParam = searchParams.get("limit");

  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }

  const limit = Math.min(
    Math.max(parseInt(limitParam ?? `${DEFAULT_LIMIT}`, 10) || DEFAULT_LIMIT, 1),
    MAX_LIMIT
  );

  try {
    const params: (string | number)[] = [epic];
    let where = "WHERE epic = $1";

    if (from) {
      params.push(from);
      where += ` AND scan_timestamp >= $${params.length}`;
    }
    if (to) {
      params.push(to);
      where += ` AND scan_timestamp <= $${params.length}`;
    }

    const countResult = await forexPool.query(
      `SELECT COUNT(*)::int AS total FROM smc_simple_rejections ${where}`,
      params
    );
    const total: number = countResult.rows[0]?.total ?? 0;

    params.push(limit);
    const limitIdx = params.length;

    const result = await forexPool.query(
      `
      SELECT
        id,
        scan_timestamp,
        rejection_stage,
        rejection_reason,
        attempted_direction,
        current_price,
        confidence_score,
        potential_stop_loss,
        potential_take_profit
      FROM smc_simple_rejections
      ${where}
      ORDER BY scan_timestamp ASC
      LIMIT $${limitIdx}
      `,
      params
    );

    const rejections = result.rows.map((row) => ({
      id: row.id as number,
      scan_timestamp: row.scan_timestamp as string,
      rejection_stage: row.rejection_stage as string,
      rejection_reason: row.rejection_reason as string,
      attempted_direction: row.attempted_direction as string | null,
      current_price: row.current_price == null ? null : Number(row.current_price),
      confidence_score: row.confidence_score == null ? null : Number(row.confidence_score),
      potential_stop_loss:
        row.potential_stop_loss == null ? null : Number(row.potential_stop_loss),
      potential_take_profit:
        row.potential_take_profit == null ? null : Number(row.potential_take_profit),
    }));

    return NextResponse.json({
      rejections,
      total,
      truncated: total > rejections.length,
    });
  } catch (error) {
    console.error("Failed to load smc rejections", error);
    return NextResponse.json(
      { error: "Failed to load smc rejections" },
      { status: 500 }
    );
  }
}
