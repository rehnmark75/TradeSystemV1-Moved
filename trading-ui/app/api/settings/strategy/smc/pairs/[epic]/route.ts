import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

function normalizeTimestamp(value: unknown) {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

async function loadActiveSmcConfigId(client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT id
      FROM smc_simple_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `
  );
  return result.rows[0]?.id ?? null;
}

async function getOverrideColumns(client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'smc_simple_pair_overrides'
    `
  );
  return result.rows.map((row: { column_name: string }) => row.column_name);
}

export async function GET(
  _request: Request,
  { params }: { params: { epic: string } }
) {
  try {
    const configId = await loadActiveSmcConfigId();
    if (!configId) {
      return NextResponse.json(
        { error: "No active SMC config found" },
        { status: 404 }
      );
    }

    const result = await strategyConfigPool.query(
      `
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1 AND epic = $2
        LIMIT 1
      `,
      [configId, params.epic]
    );

    if (!result.rows[0]) {
      return NextResponse.json({ error: "Pair override not found" }, { status: 404 });
    }

    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to load pair override", error);
    return NextResponse.json(
      { error: "Failed to load pair override" },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: Request,
  { params }: { params: { epic: string } }
) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_by, change_reason, updated_at } = body as {
    updates?: Record<string, unknown>;
    updated_by?: string;
    change_reason?: string;
    updated_at?: string;
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

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const configId = await loadActiveSmcConfigId(client);
    if (!configId) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: "No active SMC config found" },
        { status: 404 }
      );
    }

    const currentResult = await client.query(
      `
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1 AND epic = $2
        LIMIT 1
      `,
      [configId, params.epic]
    );
    const current = currentResult.rows[0];
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "Pair override not found" }, { status: 404 });
    }

    if (normalizeTimestamp(current.updated_at) !== updated_at) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        {
          error: "conflict",
          message: "Pair override was updated by another user",
          current_updated_at: current.updated_at,
          updated_by: current.updated_by,
          current_override: current
        },
        { status: 409 }
      );
    }

    const columns = await getOverrideColumns(client);
    const allowed = new Set(
      columns.filter(
        (column: string) =>
          ![
            "id",
            "config_id",
            "created_at",
            "updated_at",
            "updated_by",
            "change_reason",
            "epic"
          ].includes(column)
      )
    );

    const keys = Object.keys(updates).filter((key) => allowed.has(key));
    if (keys.length === 0) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: "No valid fields to update" },
        { status: 400 }
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
      UPDATE smc_simple_pair_overrides
      SET ${updateFields.join(", ")}
      WHERE id = $${keys.length + 3} AND updated_at = $${keys.length + 4}
      RETURNING *
    `;

    const updateResult = await client.query(updateQuery, values);
    if (updateResult.rowCount === 0) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        {
          error: "conflict",
          message: "Pair override was updated by another user",
          current_updated_at: current.updated_at,
          updated_by: current.updated_by,
          current_override: current
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
        INSERT INTO smc_simple_config_audit
          (config_id, pair_override_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, $2, 'PAIR_UPDATE', $3, $4, $5, $6)
      `,
      [
        configId,
        updateResult.rows[0]?.id ?? null,
        updated_by,
        change_reason,
        JSON.stringify(previousValues),
        JSON.stringify(updates)
      ]
    );

    await client.query("COMMIT");
    return NextResponse.json(updateResult.rows[0]);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update pair override", error);
    return NextResponse.json(
      { error: "Failed to update pair override" },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: { epic: string } }
) {
  const body = await request.json().catch(() => null);
  const updated_by = body?.updated_by;
  const change_reason = body?.change_reason;

  if (!updated_by || !change_reason) {
    return NextResponse.json(
      { error: "updated_by and change_reason are required" },
      { status: 400 }
    );
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const configId = await loadActiveSmcConfigId(client);
    if (!configId) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: "No active SMC config found" },
        { status: 404 }
      );
    }

    const currentResult = await client.query(
      `
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1 AND epic = $2
        LIMIT 1
      `,
      [configId, params.epic]
    );
    const current = currentResult.rows[0];
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "Pair override not found" }, { status: 404 });
    }

    await client.query(
      `
        DELETE FROM smc_simple_pair_overrides
        WHERE id = $1
      `,
      [current.id]
    );

    await client.query(
      `
        INSERT INTO smc_simple_config_audit
          (config_id, pair_override_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, $2, 'PAIR_DELETE', $3, $4, $5, $6)
      `,
      [
        configId,
        current.id,
        updated_by,
        change_reason,
        JSON.stringify(current),
        null
      ]
    );

    await client.query("COMMIT");
    return NextResponse.json({ success: true });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to delete pair override", error);
    return NextResponse.json(
      { error: "Failed to delete pair override" },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
