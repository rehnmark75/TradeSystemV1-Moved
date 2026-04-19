import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

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
      SELECT *
      FROM mean_reversion_global_config
      WHERE is_active = TRUE AND config_set = $1
      ORDER BY updated_at DESC
      LIMIT 1
    `,
    [configSet]
  );
  return result.rows[0] ?? null;
}

async function loadActiveConfigForUpdate(configSet: string, client: any) {
  const result = await client.query(
    `
      SELECT *
      FROM mean_reversion_global_config
      WHERE is_active = TRUE AND config_set = $1
      ORDER BY updated_at DESC
      LIMIT 1
      FOR UPDATE
    `,
    [configSet]
  );
  return result.rows[0] ?? null;
}

async function getGlobalColumns(client?: any): Promise<Set<string>> {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'mean_reversion_global_config'
    `
  );
  return new Set(
    result.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter(
        (col: string) =>
          !["id", "config_set", "created_at", "updated_at", "updated_by", "change_reason", "is_active"].includes(col),
      ),
  );
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? "demo";
    const config = await loadActiveConfig(configSet);
    if (!config) {
      return NextResponse.json(
        { error: `No active MEAN_REVERSION config found for config_set='${configSet}'` },
        { status: 404 },
      );
    }
    return NextResponse.json({ ...config, strategy_name: "MEAN_REVERSION" });
  } catch (error) {
    console.error("Failed to load MEAN_REVERSION config", error);
    return NextResponse.json({ error: "Failed to load MEAN_REVERSION config" }, { status: 500 });
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
      { status: 400 },
    );
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const current = await loadActiveConfigForUpdate(configSet, client);
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: `No active MEAN_REVERSION config found for config_set='${configSet}'` },
        { status: 404 },
      );
    }
    if (normalizeTimestamp(current.updated_at) !== updated_at) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        {
          error: "conflict",
          message: "Settings were updated by another user",
          current_config: current,
        },
        { status: 409 },
      );
    }

    const allowed = await getGlobalColumns(client);
    const keys = Object.keys(updates).filter((k) => allowed.has(k));
    if (!keys.length) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "No valid fields to update" }, { status: 400 });
    }

    const values: unknown[] = [];
    const setClauses = keys.map((key, index) => {
      values.push(updates[key]);
      return `${key} = $${index + 1}`;
    });
    values.push(updated_by, change_reason, current.id);

    const result = await client.query(
      `
        UPDATE mean_reversion_global_config
        SET ${setClauses.join(", ")},
            updated_by = $${keys.length + 1},
            change_reason = $${keys.length + 2},
            updated_at = NOW()
        WHERE id = $${keys.length + 3}
        RETURNING *
      `,
      values,
    );

    await client.query("COMMIT");
    return NextResponse.json({ ...result.rows[0], strategy_name: "MEAN_REVERSION" });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update MEAN_REVERSION config", error);
    return NextResponse.json({ error: "Failed to update MEAN_REVERSION config" }, { status: 500 });
  } finally {
    client.release();
  }
}
