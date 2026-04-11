import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

/** GET /api/settings/strategy/smc/snapshots — list settings snapshots */
export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `SELECT
        id, snapshot_name, description, base_config_id, base_config_version,
        created_at, updated_at, created_by, last_tested_at,
        test_count, is_promoted, is_backtest_only, is_active, tags,
        jsonb_object_keys(parameter_overrides) AS _unused
      FROM smc_backtest_snapshots
      WHERE is_active = TRUE AND is_backtest_only = FALSE
      ORDER BY created_at DESC`
    );

    // jsonb_object_keys expands rows — deduplicate by using a separate count query
    const listResult = await strategyConfigPool.query(
      `SELECT
        id, snapshot_name, description, base_config_id, base_config_version,
        created_at, updated_at, created_by, last_tested_at,
        test_count, is_promoted, is_backtest_only, is_active, tags,
        (SELECT count(*) FROM jsonb_object_keys(parameter_overrides)) AS field_count
      FROM smc_backtest_snapshots
      WHERE is_active = TRUE AND is_backtest_only = FALSE
      ORDER BY created_at DESC`
    );

    return NextResponse.json({ snapshots: listResult.rows });
  } catch (error) {
    console.error("Failed to list snapshots", error);
    return NextResponse.json({ error: "Failed to list snapshots" }, { status: 500 });
  }
}

/** POST /api/settings/strategy/smc/snapshots — create a settings snapshot */
export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { name, description, tags, created_by } = body as {
    name?: string;
    description?: string;
    tags?: string[];
    created_by?: string;
  };

  if (!name?.trim()) {
    return NextResponse.json({ error: "Snapshot name is required" }, { status: 400 });
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    // Load current active config
    const configResult = await client.query(
      `SELECT * FROM smc_simple_global_config WHERE is_active = TRUE ORDER BY updated_at DESC LIMIT 1`
    );
    const config = configResult.rows[0];
    if (!config) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "No active config found" }, { status: 404 });
    }

    // Strip system fields from the snapshot
    const systemFields = new Set(["id", "created_at", "updated_at", "updated_by", "change_reason", "is_active", "version"]);
    const overrides: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(config)) {
      if (!systemFields.has(key)) {
        overrides[key] = value;
      }
    }

    const insertResult = await client.query(
      `INSERT INTO smc_backtest_snapshots
        (snapshot_name, description, base_config_id, base_config_version,
         parameter_overrides, created_by, is_backtest_only, is_active, tags)
       VALUES ($1, $2, $3, $4, $5, $6, FALSE, TRUE, $7)
       RETURNING id, snapshot_name, description, created_at, tags`,
      [
        name.trim(),
        description ?? null,
        config.id,
        config.version ?? null,
        JSON.stringify(overrides),
        created_by ?? "admin",
        tags ?? [],
      ]
    );

    await client.query("COMMIT");
    return NextResponse.json({ snapshot: insertResult.rows[0] }, { status: 201 });
  } catch (error: any) {
    await client.query("ROLLBACK");
    if (error.code === "23505") {
      return NextResponse.json({ error: "A snapshot with that name already exists" }, { status: 409 });
    }
    console.error("Failed to create snapshot", error);
    return NextResponse.json({ error: "Failed to create snapshot" }, { status: 500 });
  } finally {
    client.release();
  }
}
