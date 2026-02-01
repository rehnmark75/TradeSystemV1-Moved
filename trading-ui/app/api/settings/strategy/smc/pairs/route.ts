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

  const { epic, overrides, updates, updated_by, change_reason } = body as {
    epic?: string;
    overrides?: Record<string, unknown>;
    updates?: Record<string, unknown>;
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
    const columnsResult = await strategyConfigPool.query(
      `
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'smc_simple_pair_overrides'
      `
    );
    const allowed = new Set(
      columnsResult.rows
        .map((row: { column_name: string }) => row.column_name)
        .filter(
          (column: string) =>
            ![
              "id",
              "config_id",
              "created_at",
              "updated_at",
              "updated_by",
              "change_reason",
              "epic"
            ].includes(column)
        )
    );

    const updatePayload: Record<string, unknown> = { ...(updates ?? {}) };
    if (updatePayload.parameter_overrides === undefined && overrides) {
      updatePayload.parameter_overrides = overrides;
    }

    const updateKeys = Object.keys(updatePayload).filter((key) => allowed.has(key));
    const columns = ["config_id", "epic", ...updateKeys, "updated_by", "change_reason"];
    const values: unknown[] = [configId, epic];
    updateKeys.forEach((key) => values.push(updatePayload[key]));
    values.push(updated_by);
    values.push(change_reason);

    const placeholders = columns.map((_, index) => `$${index + 1}`);
    const setClause = updateKeys
      .map((key) => `${key} = EXCLUDED.${key}`)
      .concat(["updated_by = EXCLUDED.updated_by", "change_reason = EXCLUDED.change_reason"])
      .join(", ");

    const result = await strategyConfigPool.query(
      `
        INSERT INTO smc_simple_pair_overrides
          (${columns.join(", ")})
        VALUES (${placeholders.join(", ")})
        ON CONFLICT (config_id, epic)
        DO UPDATE SET
          ${setClause}
        RETURNING *
      `,
      values
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
        JSON.stringify(updatePayload)
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
