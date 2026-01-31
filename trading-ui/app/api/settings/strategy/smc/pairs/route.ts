import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

async function loadActiveSmcConfigId(client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT id
      FROM smc_simple_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `
  );
  return result.rows[0]?.id ?? null;
}

export async function GET() {
  try {
    const configId = await loadActiveSmcConfigId();
    if (!configId) {
      return NextResponse.json(
        { error: "No active SMC config found" },
        { status: 404 }
      );
    }

    const result = await strategyConfigPool.query(
      `
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1
        ORDER BY epic ASC
      `,
      [configId]
    );

    return NextResponse.json({ config_id: configId, overrides: result.rows ?? [] });
  } catch (error) {
    console.error("Failed to load pair overrides", error);
    return NextResponse.json(
      { error: "Failed to load pair overrides" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { epic, overrides, updated_by, change_reason } = body as {
    epic?: string;
    overrides?: Record<string, unknown>;
    updated_by?: string;
    change_reason?: string;
  };

  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }
  if (!updated_by || !change_reason) {
    return NextResponse.json(
      { error: "updated_by and change_reason are required" },
      { status: 400 }
    );
  }

  const configId = await loadActiveSmcConfigId();
  if (!configId) {
    return NextResponse.json(
      { error: "No active SMC config found" },
      { status: 404 }
    );
  }

  try {
    const result = await strategyConfigPool.query(
      `
        INSERT INTO smc_simple_pair_overrides
          (config_id, epic, parameter_overrides, updated_by, change_reason)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (config_id, epic)
        DO UPDATE SET
          parameter_overrides = EXCLUDED.parameter_overrides,
          updated_by = EXCLUDED.updated_by,
          change_reason = EXCLUDED.change_reason
        RETURNING *
      `,
      [configId, epic, overrides ?? {}, updated_by, change_reason]
    );

    await strategyConfigPool.query(
      `
        INSERT INTO smc_simple_config_audit
          (config_id, pair_override_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, $2, 'PAIR_UPSERT', $3, $4, $5, $6)
      `,
      [
        configId,
        result.rows[0]?.id ?? null,
        updated_by,
        change_reason,
        null,
        JSON.stringify({ parameter_overrides: overrides ?? {} })
      ]
    );

    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to upsert pair override", error);
    return NextResponse.json(
      { error: "Failed to upsert pair override" },
      { status: 500 }
    );
  }
}
