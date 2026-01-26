import { NextResponse } from "next/server";
import { forexPool } from "../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 7;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_DAYS;
  if (parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const stage = searchParams.get("stage") || "All";
  const pair = searchParams.get("pair") || "All";
  const session = searchParams.get("session") || "All";

  const params: Array<string | number> = [days];
  let query = `
    SELECT
      id,
      scan_timestamp,
      epic,
      pair,
      rejection_stage,
      rejection_reason,
      rejection_details,
      attempted_direction,
      current_price,
      market_hour,
      market_session,
      ema_4h_value,
      ema_distance_pips,
      price_position_vs_ema,
      atr_15m,
      atr_percentile,
      volume_ratio,
      swing_high_level,
      swing_low_level,
      pullback_depth,
      fib_zone,
      swing_range_pips,
      potential_entry,
      potential_stop_loss,
      potential_take_profit,
      potential_risk_pips,
      potential_reward_pips,
      potential_rr_ratio,
      confidence_score,
      strategy_version,
      sr_blocking_level,
      sr_blocking_type,
      sr_blocking_distance_pips,
      sr_path_blocked_pct,
      target_distance_pips,
      macd_line,
      macd_signal,
      macd_histogram,
      macd_aligned,
      macd_momentum
    FROM smc_simple_rejections
    WHERE scan_timestamp >= NOW() - ($1 * INTERVAL '1 day')
  `;

  if (stage !== "All") {
    params.push(stage);
    query += ` AND rejection_stage = $${params.length}`;
  }

  if (pair !== "All") {
    params.push(pair);
    params.push(`%${pair}%`);
    query += ` AND (pair = $${params.length - 1} OR epic LIKE $${params.length})`;
  }

  if (session !== "All") {
    params.push(session);
    query += ` AND market_session = $${params.length}`;
  }

  query += " ORDER BY scan_timestamp DESC LIMIT 1000";

  try {
    const result = await forexPool.query(query, params);
    return NextResponse.json({ rows: result.rows ?? [] });
  } catch (error) {
    console.error("Failed to load SMC rejections list", error);
    return NextResponse.json(
      { error: "Failed to load SMC rejections list" },
      { status: 500 }
    );
  }
}
