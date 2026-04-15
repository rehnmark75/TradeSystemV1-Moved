import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

type SnapshotEnvelope = {
  snapshot_format?: string;
  config_set?: string;
  global_config?: Record<string, unknown>;
  pair_overrides?: Array<Record<string, unknown>>;
};

function isSnapshotEnvelope(value: unknown): value is SnapshotEnvelope {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeSnapshotPayload(
  parameterOverrides: Record<string, unknown>,
  fallbackConfigSet: string | null
) {
  if (isSnapshotEnvelope(parameterOverrides) && parameterOverrides.global_config) {
    return {
      configSet: typeof parameterOverrides.config_set === "string"
        ? parameterOverrides.config_set
        : (fallbackConfigSet ?? "demo"),
      globalConfig: parameterOverrides.global_config,
      pairOverrides: Array.isArray(parameterOverrides.pair_overrides)
        ? parameterOverrides.pair_overrides
        : [],
    };
  }

  return {
    configSet: fallbackConfigSet ?? "demo",
    globalConfig: parameterOverrides,
    pairOverrides: [] as Array<Record<string, unknown>>,
  };
}

/** POST /api/settings/strategy/smc/snapshots/[id]/restore
 *  Apply snapshot values to the active config and pair overrides for the same environment.
 */
export async function POST(
  request: Request,
  { params }: { params: { id: string } }
) {
  const id = Number(params.id);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid snapshot ID" }, { status: 400 });
  }

  const body = await request.json().catch(() => null);
  const restoredBy = body?.restored_by ?? "admin";

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    const snapResult = await client.query(
      `SELECT s.*, c.config_set AS base_config_set
       FROM smc_backtest_snapshots s
       LEFT JOIN smc_simple_global_config c ON c.id = s.base_config_id
       WHERE s.id = $1 AND s.is_active = TRUE`,
      [id]
    );
    const snapshot = snapResult.rows[0];
    if (!snapshot) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "Snapshot not found" }, { status: 404 });
    }

    const normalized = normalizeSnapshotPayload(
      snapshot.parameter_overrides as Record<string, unknown>,
      snapshot.base_config_set ?? null
    );

    const configResult = await client.query(
      `SELECT *
       FROM smc_simple_global_config
       WHERE is_active = TRUE AND config_set = $1
       ORDER BY updated_at DESC
       LIMIT 1`,
      [normalized.configSet]
    );
    const config = configResult.rows[0];
    if (!config) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: `No active config found for config_set='${normalized.configSet}'` },
        { status: 404 }
      );
    }

    const colResult = await client.query(
      `SELECT column_name FROM information_schema.columns
       WHERE table_name = 'smc_simple_global_config' AND table_schema = 'public'`
    );
    const allowedCols = new Set(colResult.rows.map((row: { column_name: string }) => row.column_name));
    const systemFields = new Set(["id", "created_at", "updated_at", "is_active"]);

    const updates: string[] = [];
    const values: unknown[] = [];
    const previousValues: Record<string, unknown> = {};
    const newValues: Record<string, unknown> = {};
    let paramIndex = 1;

    for (const [key, value] of Object.entries(normalized.globalConfig)) {
      if (!allowedCols.has(key) || systemFields.has(key)) continue;
      previousValues[key] = config[key];
      newValues[key] = value;
      updates.push(`${key} = $${paramIndex++}`);
      values.push(typeof value === "object" ? JSON.stringify(value) : value);
    }

    if (updates.length > 0) {
      const changeReason = `Restored from snapshot: ${snapshot.snapshot_name}`;
      updates.push(`updated_at = NOW()`);
      updates.push(`updated_by = $${paramIndex++}`);
      updates.push(`change_reason = $${paramIndex++}`);
      values.push(restoredBy);
      values.push(changeReason);
      values.push(config.id);

      await client.query(
        `UPDATE smc_simple_global_config SET ${updates.join(", ")} WHERE id = $${paramIndex}`,
        values
      );

      await client.query(
        `INSERT INTO smc_simple_config_audit
           (config_id, change_type, changed_by, change_reason, previous_values, new_values)
         VALUES ($1, 'SNAPSHOT_RESTORE', $2, $3, $4, $5)`,
        [config.id, restoredBy, changeReason, JSON.stringify(previousValues), JSON.stringify(newValues)]
      );
    }

    if (Array.isArray(normalized.pairOverrides)) {
      await client.query(
        `DELETE FROM smc_simple_pair_overrides WHERE config_id = $1`,
        [config.id]
      );

      for (const override of normalized.pairOverrides) {
        const epic = typeof override.epic === "string" ? override.epic : null;
        if (!epic) continue;

        const columns: string[] = ["config_id", "epic"];
        const overrideValues: unknown[] = [config.id, epic];

        for (const [key, value] of Object.entries(override)) {
          if (["id", "config_id", "epic", "created_at", "updated_at"].includes(key)) continue;
          columns.push(key);
          overrideValues.push(value ?? null);
        }

        const placeholders = columns.map((_, index) => `$${index + 1}`).join(", ");
        await client.query(
          `INSERT INTO smc_simple_pair_overrides (${columns.join(", ")})
           VALUES (${placeholders})`,
          overrideValues
        );
      }
    }

    await client.query("COMMIT");
    return NextResponse.json({
      restored: true,
      snapshot_name: snapshot.snapshot_name,
      config_set: normalized.configSet,
      global_fields_restored: updates.length > 0 ? updates.length - 3 : 0,
      pair_overrides_restored: normalized.pairOverrides.length,
    });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to restore snapshot", error);
    return NextResponse.json({ error: "Failed to restore snapshot" }, { status: 500 });
  } finally {
    client.release();
  }
}
