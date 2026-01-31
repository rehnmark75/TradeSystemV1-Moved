import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";
import { validateScannerUpdates } from "../../../../lib/validation/settingsSchemas";

export const dynamic = "force-dynamic";

function normalizeTimestamp(value: unknown) {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

async function loadActiveScannerConfig(client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT *
      FROM scanner_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `
  );
  return result.rows[0] ?? null;
}

export async function GET() {
  try {
    const config = await loadActiveScannerConfig();
    if (!config) {
      return NextResponse.json(
        { error: "No active scanner config found" },
        { status: 404 }
      );
    }
    return NextResponse.json(config);
  } catch (error) {
    console.error("Failed to load scanner config", error);
    return NextResponse.json(
      { error: "Failed to load scanner config" },
      { status: 500 }
    );
  }
}

export async function PATCH(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_by, change_reason, updated_at, category } = body as {
    updates: Record<string, unknown>;
    updated_by?: string;
    change_reason?: string;
    updated_at?: string;
    category?: string;
  };

  if (!updates || typeof updates !== "object") {
    return NextResponse.json({ error: "Missing updates payload" }, { status: 400 });
  }

  if (!updated_by || !change_reason) {
    return NextResponse.json(
      { error: "updated_by and change_reason are required" },
      { status: 400 }
    );
  }

  if (!updated_at) {
    return NextResponse.json(
      { error: "updated_at is required for optimistic locking" },
      { status: 400 }
    );
  }

  const validation = validateScannerUpdates(updates);
  if (!validation.success) {
    return NextResponse.json(
      { error: "Validation failed", details: validation.error.flatten() },
      { status: 400 }
    );
  }

  const keys = Object.keys(updates);
  if (keys.length === 0) {
    return NextResponse.json({ error: "No updates provided" }, { status: 400 });
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    const current = await loadActiveScannerConfig(client);
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: "No active scanner config found" },
        { status: 404 }
      );
    }

    if (normalizeTimestamp(current.updated_at) !== updated_at) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        {
          error: "conflict",
          message: `Settings were updated by '${current.updated_by}' recently`,
          current_version: current.version,
          current_updated_at: current.updated_at,
          updated_by: current.updated_by,
          current_config: current
        },
        { status: 409 }
      );
    }

    const updateFields: string[] = [];
    const values: unknown[] = [];

    keys.forEach((key, index) => {
      updateFields.push(`${key} = $${index + 1}`);
      values.push(updates[key]);
    });

    updateFields.push(`updated_by = $${keys.length + 1}`);
    updateFields.push(`change_reason = $${keys.length + 2}`);
    values.push(updated_by);
    values.push(change_reason);

    values.push(current.id);
    values.push(updated_at);

    const updateQuery = `
      UPDATE scanner_global_config
      SET ${updateFields.join(", ")}
      WHERE id = $${keys.length + 3} AND updated_at = $${keys.length + 4}
    `;

    const updateResult = await client.query(updateQuery, values);
    if (updateResult.rowCount === 0) {
      await client.query("ROLLBACK");
      const latest = await loadActiveScannerConfig(client);
      return NextResponse.json(
        {
          error: "conflict",
          message: "Settings were updated by another user",
          current_version: latest?.version ?? current.version,
          current_updated_at: latest?.updated_at ?? current.updated_at,
          updated_by: latest?.updated_by ?? current.updated_by,
          current_config: latest ?? current
        },
        { status: 409 }
      );
    }

    const previousValues: Record<string, unknown> = {};
    keys.forEach((key) => {
      previousValues[key] = current[key];
    });

    await client.query(
      `
        INSERT INTO scanner_config_audit
        (config_id, change_type, changed_by, change_reason, previous_values, new_values, category)
        VALUES ($1, 'UPDATE', $2, $3, $4, $5, $6)
      `,
      [
        current.id,
        updated_by,
        change_reason,
        JSON.stringify(previousValues),
        JSON.stringify(updates),
        category ?? null
      ]
    );

    const updated = await loadActiveScannerConfig(client);
    await client.query("COMMIT");
    return NextResponse.json(updated);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update scanner config", error);
    return NextResponse.json(
      { error: "Failed to update scanner config" },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
