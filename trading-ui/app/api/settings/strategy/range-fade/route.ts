import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const DEFAULT_PROFILE = "5m";
const DEFAULT_CONFIG_SET = "demo";

function normalizeTimestamp(value: unknown) {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

async function loadActiveConfig(profileName: string, configSet: string, client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT *
      FROM eurusd_range_fade_global_config
      WHERE is_active = TRUE AND profile_name = $1 AND config_set = $2
      ORDER BY updated_at DESC
      LIMIT 1
    `,
    [profileName, configSet],
  );
  return result.rows[0] ?? null;
}

async function loadActiveConfigForUpdate(profileName: string, configSet: string, client: any) {
  const result = await client.query(
    `
      SELECT *
      FROM eurusd_range_fade_global_config
      WHERE is_active = TRUE AND profile_name = $1 AND config_set = $2
      ORDER BY updated_at DESC
      LIMIT 1
      FOR UPDATE
    `,
    [profileName, configSet],
  );
  return result.rows[0] ?? null;
}

async function getGlobalColumns(client?: any): Promise<Set<string>> {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'eurusd_range_fade_global_config'
    `,
  );
  return new Set(
    result.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter(
        (col: string) =>
          !["id", "profile_name", "created_at", "updated_at", "is_active"].includes(col),
      ),
  );
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const profileParam = searchParams.get("profile_name");
    const profile = profileParam && /^(5m|15m)$/.test(profileParam) ? profileParam : DEFAULT_PROFILE;
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;
    const config = await loadActiveConfig(profile, configSet);
    if (!config) {
      return NextResponse.json(
        { error: `No active RANGE_FADE config found for profile_name='${profile}', config_set='${configSet}'` },
        { status: 404 },
      );
    }
    return NextResponse.json({ ...config, strategy_name: "RANGE_FADE" });
  } catch (error) {
    console.error("Failed to load RANGE_FADE config", error);
    return NextResponse.json({ error: "Failed to load RANGE_FADE config" }, { status: 500 });
  }
}

export async function PATCH(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_at, config_set, profile_name } = body as {
    updates: Record<string, unknown>;
    updated_at?: string;
    config_set?: string;
    profile_name?: string;
  };

  const profile = profile_name && /^(5m|15m)$/.test(profile_name) ? profile_name : DEFAULT_PROFILE;
  const configSet = config_set ?? DEFAULT_CONFIG_SET;
  if (!updates || typeof updates !== "object") {
    return NextResponse.json({ error: "Missing updates payload" }, { status: 400 });
  }
  if (!updated_at) {
    return NextResponse.json({ error: "updated_at is required" }, { status: 400 });
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const current = await loadActiveConfigForUpdate(profile, configSet, client);
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: `No active RANGE_FADE config found for profile_name='${profile}', config_set='${configSet}'` },
        { status: 404 },
      );
    }
    if (normalizeTimestamp(current.updated_at) !== updated_at) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: "conflict", message: "Settings were updated by another user", current_config: current },
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
        UPDATE eurusd_range_fade_global_config
        SET ${setClauses.join(", ")},
            updated_at = NOW()
        WHERE id = $${keys.length + 1}
        RETURNING *
      `,
      values,
    );

    await client.query("COMMIT");
    return NextResponse.json({ ...result.rows[0], strategy_name: "RANGE_FADE" });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update RANGE_FADE config", error);
    return NextResponse.json({ error: "Failed to update RANGE_FADE config" }, { status: 500 });
  } finally {
    client.release();
  }
}
