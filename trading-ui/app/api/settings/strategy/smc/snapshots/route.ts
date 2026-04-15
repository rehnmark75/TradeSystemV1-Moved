import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const SNAPSHOT_SYSTEM_FIELDS = new Set([
  "id", "created_at", "updated_at", "updated_by", "change_reason", "is_active", "version"
]);

function buildSnapshotPayload(
  configSet: string,
  globalConfig: Record<string, unknown>,
  pairOverrides: Array<Record<string, unknown>>
) {
  const filteredGlobalConfig: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(globalConfig)) {
    if (!SNAPSHOT_SYSTEM_FIELDS.has(key)) {
      filteredGlobalConfig[key] = value;
    }
  }

  return {
    snapshot_format: "smc_strategy_v2",
    config_set: configSet,
    global_config: filteredGlobalConfig,
    pair_overrides: pairOverrides,
  };
}

/** GET /api/settings/strategy/smc/snapshots — list settings snapshots */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set");
    const listResult = await strategyConfigPool.query(
      `SELECT
        s.id, s.snapshot_name, s.description, s.base_config_id, s.base_config_version,
        created_at, updated_at, created_by, last_tested_at,
        test_count, is_promoted, is_backtest_only, is_active, tags,
        CASE
          WHEN jsonb_typeof(s.parameter_overrides->'global_config') = 'object'
            THEN (SELECT count(*) FROM jsonb_object_keys(s.parameter_overrides->'global_config'))
          ELSE (SELECT count(*) FROM jsonb_object_keys(s.parameter_overrides))
        END AS field_count
      FROM smc_backtest_snapshots s
      LEFT JOIN smc_simple_global_config c ON c.id = s.base_config_id
      WHERE s.is_active = TRUE
        AND s.is_backtest_only = FALSE
        AND (
          $1::text IS NULL
          OR COALESCE(s.parameter_overrides->>'config_set', c.config_set) = $1
        )
      ORDER BY s.created_at DESC`,
      [configSet]
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
    config_set?: string;
  };
  const configSet = typeof body.config_set === "string" ? body.config_set : "demo";

  if (!name?.trim()) {
    return NextResponse.json({ error: "Snapshot name is required" }, { status: 400 });
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    // Load current active config for the selected environment
    const configResult = await client.query(
      `SELECT * FROM smc_simple_global_config
       WHERE is_active = TRUE AND config_set = $1
       ORDER BY updated_at DESC
       LIMIT 1`,
      [configSet]
    );
    const config = configResult.rows[0];
    if (!config) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: `No active config found for config_set='${configSet}'` }, { status: 404 });
    }

    const pairOverridesResult = await client.query(
      `SELECT *
       FROM smc_simple_pair_overrides
       WHERE config_id = $1
       ORDER BY epic ASC`,
      [config.id]
    );
    const overrides = buildSnapshotPayload(configSet, config, pairOverridesResult.rows);

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
        overrides,
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
