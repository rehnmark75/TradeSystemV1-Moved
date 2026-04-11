import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

/** GET /api/settings/strategy/smc/snapshots/[id]/compare
 *  Compare snapshot values against current live config.
 *  Returns { snapshot_values, current_values, diff[] }
 */
export async function GET(
  _request: Request,
  { params }: { params: { id: string } }
) {
  const id = Number(params.id);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid snapshot ID" }, { status: 400 });
  }

  try {
    const [snapResult, configResult] = await Promise.all([
      strategyConfigPool.query(
        `SELECT * FROM smc_backtest_snapshots WHERE id = $1 AND is_active = TRUE`,
        [id]
      ),
      strategyConfigPool.query(
        `SELECT * FROM smc_simple_global_config WHERE is_active = TRUE ORDER BY updated_at DESC LIMIT 1`
      ),
    ]);

    const snapshot = snapResult.rows[0];
    const config = configResult.rows[0];

    if (!snapshot) {
      return NextResponse.json({ error: "Snapshot not found" }, { status: 404 });
    }
    if (!config) {
      return NextResponse.json({ error: "No active config found" }, { status: 404 });
    }

    const snapshotValues = snapshot.parameter_overrides as Record<string, unknown>;
    const systemFields = new Set(["id", "created_at", "updated_at", "updated_by", "change_reason", "is_active", "version"]);

    const currentValues: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(config)) {
      if (!systemFields.has(key)) {
        currentValues[key] = value;
      }
    }

    // Build diff
    const allKeys = new Set([...Object.keys(snapshotValues), ...Object.keys(currentValues)]);
    const diff: Array<{ field: string; snapshot_value: unknown; current_value: unknown; changed: boolean }> = [];

    for (const field of allKeys) {
      const sv = snapshotValues[field];
      const cv = currentValues[field];
      const svStr = JSON.stringify(sv);
      const cvStr = JSON.stringify(cv);
      diff.push({
        field,
        snapshot_value: sv ?? null,
        current_value: cv ?? null,
        changed: svStr !== cvStr,
      });
    }

    diff.sort((a, b) => {
      // Changed fields first
      if (a.changed !== b.changed) return a.changed ? -1 : 1;
      return a.field.localeCompare(b.field);
    });

    return NextResponse.json({
      snapshot_name: snapshot.snapshot_name,
      snapshot_created_at: snapshot.created_at,
      snapshot_values: snapshotValues,
      current_values: currentValues,
      diff,
      changed_count: diff.filter((d) => d.changed).length,
    });
  } catch (error) {
    console.error("Failed to compare snapshot", error);
    return NextResponse.json({ error: "Failed to compare snapshot" }, { status: 500 });
  }
}
