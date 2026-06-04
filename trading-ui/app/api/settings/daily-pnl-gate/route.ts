import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const environment = searchParams.get("environment") ?? "demo";

  try {
    const result = await strategyConfigPool.query(
      `SELECT environment, is_enabled,
              profit_limit_sek::float8 AS profit_limit_sek,
              loss_limit_sek::float8 AS loss_limit_sek,
              updated_at
       FROM daily_pnl_gate_config WHERE environment = $1`,
      [environment]
    );
    if (result.rows.length === 0) {
      return NextResponse.json({ error: "Config not found" }, { status: 404 });
    }
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to load daily PnL gate config", error);
    return NextResponse.json({ error: "Failed to load config" }, { status: 500 });
  }
}

export async function PUT(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { environment, is_enabled, profit_limit_sek, loss_limit_sek } = body as {
    environment?: string;
    is_enabled?: boolean;
    profit_limit_sek?: number;
    loss_limit_sek?: number;
  };

  if (!environment || !["demo", "live"].includes(environment)) {
    return NextResponse.json({ error: "environment must be 'demo' or 'live'" }, { status: 400 });
  }
  if (typeof is_enabled !== "boolean") {
    return NextResponse.json({ error: "is_enabled must be a boolean" }, { status: 400 });
  }
  if (typeof profit_limit_sek !== "number" || profit_limit_sek <= 0) {
    return NextResponse.json({ error: "profit_limit_sek must be a positive number" }, { status: 400 });
  }
  if (typeof loss_limit_sek !== "number" || loss_limit_sek >= 0) {
    return NextResponse.json({ error: "loss_limit_sek must be a negative number" }, { status: 400 });
  }

  try {
    const result = await strategyConfigPool.query(
      `UPDATE daily_pnl_gate_config
       SET is_enabled = $1, profit_limit_sek = $2, loss_limit_sek = $3, updated_at = NOW()
       WHERE environment = $4
       RETURNING environment, is_enabled,
                 profit_limit_sek::float8 AS profit_limit_sek,
                 loss_limit_sek::float8 AS loss_limit_sek,
                 updated_at`,
      [is_enabled, profit_limit_sek, loss_limit_sek, environment]
    );
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to update daily PnL gate config", error);
    return NextResponse.json({ error: "Failed to update config" }, { status: 500 });
  }
}
