import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

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
    let where = "WHERE epic = $1 AND decision = 'blocked'";

    if (from) {
      params.push(from);
      where += ` AND signal_timestamp >= $${params.length}`;
    }
    if (to) {
      params.push(to);
      where += ` AND signal_timestamp <= $${params.length}`;
    }

    const countResult = await strategyConfigPool.query(
      `SELECT COUNT(*)::int AS total FROM loss_prevention_decisions ${where}`,
      params
    );
    const total: number = countResult.rows[0]?.total ?? 0;

    params.push(limit);
    const result = await strategyConfigPool.query(
      `
      SELECT
        id,
        signal_timestamp AS scan_timestamp,
        signal_type AS attempted_direction,
        confidence,
        total_penalty,
        triggered_rules
      FROM loss_prevention_decisions
      ${where}
      ORDER BY signal_timestamp ASC
      LIMIT $${params.length}
      `,
      params
    );

    const rejections = result.rows.map((row) => {
      const rules: { rule_name: string; penalty: number; category: string }[] =
        Array.isArray(row.triggered_rules) ? row.triggered_rules : [];
      const ruleNames = rules.map((r) => r.rule_name).join(", ");
      return {
        id: row.id as number,
        scan_timestamp: row.scan_timestamp as string,
        rejection_stage: "LPF_BLOCKED",
        rejection_reason: `penalty ${Number(row.total_penalty).toFixed(2)} — rules: ${ruleNames || "unknown"}`,
        attempted_direction: row.attempted_direction as string | null,
        current_price: null,
        confidence_score: row.confidence == null ? null : Number(row.confidence),
        potential_stop_loss: null,
        potential_take_profit: null,
      };
    });

    return NextResponse.json({ rejections, total, truncated: total > rejections.length });
  } catch (error) {
    console.error("Failed to load LPF rejections", error);
    return NextResponse.json({ error: "Failed to load LPF rejections" }, { status: 500 });
  }
}
