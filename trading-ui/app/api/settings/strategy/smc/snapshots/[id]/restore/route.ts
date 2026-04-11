import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

/** POST /api/settings/strategy/smc/snapshots/[id]/restore
 *  Apply snapshot values to the live global config
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
  const restored_by = body?.restored_by ?? "admin";

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    // Load snapshot
    const snapResult = await client.query(
      `SELECT * FROM smc_backtest_snapshots WHERE id = $1 AND is_active = TRUE`,
      [id]
    );
    const snapshot = snapResult.rows[0];
    if (!snapshot) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "Snapshot not found" }, { status: 404 });
    }

    // Load current config
    const configResult = await client.query(
      `SELECT * FROM smc_simple_global_config WHERE is_active = TRUE ORDER BY updated_at DESC LIMIT 1`
    );
    const config = configResult.rows[0];
    if (!config) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "No active config found" }, { status: 404 });
    }

    // Get allowed columns from the config table (avoid SQL injection)
    const colResult = await client.query(
      `SELECT column_name FROM information_schema.columns
       WHERE table_name = 'smc_simple_global_config' AND table_schema = 'public'`
    );
    const allowedCols = new Set(colResult.rows.map((r: any) => r.column_name));
    const systemFields = new Set(["id", "created_at", "updated_at", "is_active"]);

    const overrides = snapshot.parameter_overrides as Record<string, unknown>;
    const updates: string[] = [];
    const values: unknown[] = [];
    let paramIndex = 1;

    // Record previous values for audit
    const previousValues: Record<string, unknown> = {};
    const newValues: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(overrides)) {
      if (!allowedCols.has(key) || systemFields.has(key)) continue;
      previousValues[key] = config[key];
      newValues[key] = value;
      updates.push(`${key} = $${paramIndex++}`);
      values.push(typeof value === "object" ? JSON.stringify(value) : value);
    }

    if (updates.length === 0) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "No applicable fields to restore" }, { status: 400 });
    }

    const changeReason = `Restored from snapshot: ${snapshot.snapshot_name}`;
    updates.push(`updated_at = NOW()`);
    updates.push(`updated_by = $${paramIndex++}`);
    updates.push(`change_reason = $${paramIndex++}`);
    values.push(restored_by);
    values.push(changeReason);
    values.push(config.id);

    await client.query(
      `UPDATE smc_simple_global_config SET ${updates.join(", ")} WHERE id = $${paramIndex}`,
      values
    );

    // Audit trail
    await client.query(
      `INSERT INTO smc_simple_config_audit
         (config_id, change_type, changed_by, change_reason, previous_values, new_values)
       VALUES ($1, 'SNAPSHOT_RESTORE', $2, $3, $4, $5)`,
      [config.id, restored_by, changeReason, JSON.stringify(previousValues), JSON.stringify(newValues)]
    );

    await client.query("COMMIT");
    return NextResponse.json({
      restored: true,
      snapshot_name: snapshot.snapshot_name,
      fields_restored: updates.length - 3, // subtract the system updates
    });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to restore snapshot", error);
    return NextResponse.json({ error: "Failed to restore snapshot" }, { status: 500 });
  } finally {
    client.release();
  }
}
