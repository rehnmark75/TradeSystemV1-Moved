import { NextResponse } from "next/server";
import { forexPool } from "../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 7;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const step = searchParams.get("step") || "All";
  const pair = searchParams.get("pair") || "All";
  const signalType = searchParams.get("signal_type") || "All";

  const params: Array<string | number> = [days];
  let query = `
    SELECT
      id, created_at, epic, pair, signal_type, strategy,
      confidence_score, step, rejection_reason,
      entry_price, risk_pips, reward_pips, rr_ratio,
      market_regime, market_session,
      lpf_penalty, lpf_would_block, lpf_triggered_rules
    FROM validator_rejections
    WHERE created_at >= NOW() - ($1 * INTERVAL '1 day')
  `;

  if (step !== "All") {
    params.push(step);
    query += ` AND step = $${params.length}`;
  }

  if (pair !== "All") {
    params.push(pair);
    query += ` AND pair = $${params.length}`;
  }

  if (signalType !== "All") {
    params.push(signalType);
    query += ` AND signal_type = $${params.length}`;
  }

  query += " ORDER BY created_at DESC LIMIT 500";

  try {
    const result = await forexPool.query(query, params);
    return NextResponse.json({ rows: result.rows ?? [] });
  } catch (error) {
    console.error("validator-rejections/list error:", error);
    return NextResponse.json({ error: "Failed to load list" }, { status: 500 });
  }
}
