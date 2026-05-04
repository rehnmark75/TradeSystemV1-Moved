import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

function normalizeTimestamp(value: unknown) {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

async function loadActiveConfig(client?: { query: Function }) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT *
      FROM impulse_fade_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `,
  );
  return result.rows[0] ?? null;
}

async function loadActiveConfigForUpdate(client: { query: Function }) {
  const result = await client.query(
    `
      SELECT *
      FROM impulse_fade_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
      FOR UPDATE
    `,
  );
  return result.rows[0] ?? null;
}

async function getGlobalColumns(client?: { query: Function }): Promise<Set<string>> {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'impulse_fade_global_config'
    `,
  );
  return new Set(
    result.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter(
        (col: string) =>
          !["id", "strategy_name", "version", "created_at", "updated_at", "is_active"].includes(col),
      ),
  );
}

export async function GET() {
  try {
    const config = await loadActiveConfig();
    if (!config) {
      return NextResponse.json(
        { error: "No active IMPULSE_FADE config found" },
        { status: 404 },
      );
    }
    return NextResponse.json({ ...config, strategy_name: "IMPULSE_FADE" });
  } catch (error) {
    console.error("Failed to load IMPULSE_FADE config", error);
    return NextResponse.json({ error: "Failed to load IMPULSE_FADE config" }, { status: 500 });
  }
}

export async function PATCH(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_at } = body as {
    updates?: Record<string, unknown>;
    updated_at?: string;
  };

  if (!updates || typeof updates !== "object") {
    return NextResponse.json({ error: "Missing updates payload" }, { status: 400 });
  }
  if (!updated_at) {
    return NextResponse.json({ error: "updated_at is required" }, { status: 400 });
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const current = await loadActiveConfigForUpdate(client);
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: "No active IMPULSE_FADE config found" },
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
    values.push(current.id);

    const result = await client.query(
      `
        UPDATE impulse_fade_global_config
        SET ${setClauses.join(", ")}, updated_at = NOW()
        WHERE id = $${keys.length + 1}
        RETURNING *
      `,
      values,
    );

    await client.query("COMMIT");
    return NextResponse.json({ ...result.rows[0], strategy_name: "IMPULSE_FADE" });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update IMPULSE_FADE config", error);
    return NextResponse.json({ error: "Failed to update IMPULSE_FADE config" }, { status: 500 });
  } finally {
    client.release();
  }
}
