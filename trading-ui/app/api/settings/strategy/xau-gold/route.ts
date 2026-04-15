import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";
import { validateXauGoldUpdates } from "../../../../../lib/validation/settingsSchemas";

export const dynamic = "force-dynamic";

function normalizeTimestamp(value: unknown) {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

async function loadActiveConfig(configSet: string, client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT parameter_name, parameter_value, value_type, updated_at, updated_by, change_reason
      FROM xau_gold_global_config
      WHERE is_active = TRUE AND config_set = $1
      ORDER BY display_order, parameter_name
    `,
    [configSet]
  );
  if (!result.rows.length) return null;

  const config: Record<string, unknown> = {
    updated_at: result.rows.reduce(
      (latest: string, row: { updated_at: string }) =>
        !latest || new Date(row.updated_at) > new Date(latest) ? row.updated_at : latest,
      ""
    ),
    updated_by: result.rows.find((row: { updated_by?: string | null }) => row.updated_by)?.updated_by ?? null,
    change_reason:
      result.rows.find((row: { change_reason?: string | null }) => row.change_reason)?.change_reason ?? null,
    config_set: configSet,
    strategy_name: "XAU_GOLD",
  };

  for (const row of result.rows) {
    const t = String(row.value_type ?? "string").toLowerCase();
    let value: unknown = row.parameter_value;
    try {
      if (t === "int" || t === "integer") value = parseInt(String(row.parameter_value), 10);
      else if (t === "float") value = Number(row.parameter_value);
      else if (t === "bool") value = String(row.parameter_value).toLowerCase() === "true";
      else if (t === "json") value = JSON.parse(String(row.parameter_value));
    } catch {
      value = row.parameter_value;
    }
    config[row.parameter_name] = value;
  }

  return config;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? "demo";
    const config = await loadActiveConfig(configSet);
    if (!config) {
      return NextResponse.json(
        { error: `No active XAU_GOLD config found for config_set='${configSet}'` },
        { status: 404 }
      );
    }
    return NextResponse.json(config);
  } catch (error) {
    console.error("Failed to load XAU_GOLD config", error);
    return NextResponse.json({ error: "Failed to load XAU_GOLD config" }, { status: 500 });
  }
}

export async function PATCH(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_by, change_reason, updated_at, config_set } = body as {
    updates: Record<string, unknown>;
    updated_by?: string;
    change_reason?: string;
    updated_at?: string;
    config_set?: string;
  };

  const configSet = config_set ?? "demo";
  if (!updates || typeof updates !== "object") {
    return NextResponse.json({ error: "Missing updates payload" }, { status: 400 });
  }
  if (!updated_by || !change_reason || !updated_at) {
    return NextResponse.json(
      { error: "updated_by, change_reason, and updated_at are required" },
      { status: 400 }
    );
  }

  const validation = validateXauGoldUpdates(updates);
  if (!validation.success) {
    return NextResponse.json(
      { error: "Validation failed", details: validation.error.flatten() },
      { status: 400 }
    );
  }

  const current = await loadActiveConfig(configSet);
  if (!current) {
    return NextResponse.json(
      { error: `No active XAU_GOLD config found for config_set='${configSet}'` },
      { status: 404 }
    );
  }
  if (normalizeTimestamp(current.updated_at) !== updated_at) {
    return NextResponse.json(
      {
        error: "conflict",
        message: "Settings were updated by another user",
        current_config: current,
      },
      { status: 409 }
    );
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    for (const [key, value] of Object.entries(updates)) {
      const serialized =
        value !== null && typeof value === "object" ? JSON.stringify(value) : String(value);
      const result = await client.query(
        `
          UPDATE xau_gold_global_config
          SET parameter_value = $1,
              updated_at = NOW(),
              updated_by = $2,
              change_reason = $3
          WHERE config_set = $4 AND parameter_name = $5 AND is_active = TRUE
          RETURNING id
        `,
        [serialized, updated_by, change_reason, configSet, key]
      );
      if (!result.rowCount) {
        throw new Error(`Unknown XAU_GOLD parameter: ${key}`);
      }
      await client.query(
        `
          INSERT INTO xau_gold_config_audit
            (config_set, table_name, record_id, parameter_name, old_value, new_value, change_type, changed_by, change_reason)
          VALUES ($1, 'xau_gold_global_config', $2, $3, NULL, $4, 'UPDATE', $5, $6)
        `,
        [configSet, result.rows[0].id, key, serialized, updated_by, change_reason]
      );
    }
    await client.query("COMMIT");
    const updated = await loadActiveConfig(configSet, client);
    return NextResponse.json(updated);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update XAU_GOLD config", error);
    return NextResponse.json({ error: "Failed to update XAU_GOLD config" }, { status: 500 });
  } finally {
    client.release();
  }
}
