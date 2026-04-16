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
    let where = "WHERE epic = $1 AND claude_approved = false";

    if (from) {
      params.push(from);
      where += ` AND alert_timestamp >= $${params.length}`;
    }
    if (to) {
      params.push(to);
      where += ` AND alert_timestamp <= $${params.length}`;
    }

    const countResult = await forexPool.query(
      `SELECT COUNT(*)::int AS total FROM alert_history ${where}`,
      params
    );
    const total: number = countResult.rows[0]?.total ?? 0;

    params.push(limit);
    const result = await forexPool.query(
      `
      SELECT
        id,
        alert_timestamp AS scan_timestamp,
        signal_type AS attempted_direction,
        price,
        confidence_score,
        claude_reason,
        claude_score
      FROM alert_history
      ${where}
      ORDER BY alert_timestamp ASC
      LIMIT $${params.length}
      `,
      params
    );

    const rejections = result.rows.map((row) => ({
      id: row.id as number,
      scan_timestamp: row.scan_timestamp as string,
      rejection_stage: "CLAUDE_REJECTED",
      rejection_reason: (row.claude_reason as string | null) ?? "No reason provided",
      attempted_direction: row.attempted_direction as string | null,
      current_price: row.price == null ? null : Number(row.price),
      confidence_score: row.confidence_score == null ? null : Number(row.confidence_score),
      potential_stop_loss: null,
      potential_take_profit: null,
    }));

    return NextResponse.json({ rejections, total, truncated: total > rejections.length });
  } catch (error) {
    console.error("Failed to load Claude rejections", error);
    return NextResponse.json({ error: "Failed to load Claude rejections" }, { status: 500 });
  }
}
