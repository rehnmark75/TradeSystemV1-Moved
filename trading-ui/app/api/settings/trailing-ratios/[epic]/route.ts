import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";
import { RATIO_FIELDS } from "../../../../../lib/settings/trailingRatioFields";

export const dynamic = "force-dynamic";

function normalizeTimestamp(value: unknown) {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

export async function GET(
  request: Request,
  { params }: { params: { epic: string } }
) {
  const { searchParams } = new URL(request.url);
  const configSet = searchParams.get("config_set") ?? "demo";
  const isScalp = searchParams.get("is_scalp") === "true";

  try {
    const result = await strategyConfigPool.query(
      `SELECT * FROM trailing_ratio_config
        WHERE config_set = $1 AND epic = $2 AND is_scalp = $3
        LIMIT 1`,
      [configSet, params.epic, isScalp]
    );
    if (!result.rows[0]) {
      return NextResponse.json({ error: "Row not found" }, { status: 404 });
    }
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to load trailing ratio row", error);
    return NextResponse.json(
      { error: "Failed to load trailing ratio row" },
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

  const {
    updates,
    updated_by,
    change_reason,
    updated_at,
    config_set,
    is_scalp,
  } = body as {
    updates?: Record<string, unknown>;
    updated_by?: string;
    change_reason?: string;
    updated_at?: string;
    config_set?: string;
    is_scalp?: boolean;
  };

  const configSet = config_set ?? "demo";
  const scalp = Boolean(is_scalp);

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

  const allowed = new Set<string>(RATIO_FIELDS);
  const keys = Object.keys(updates).filter((k) => allowed.has(k));
  if (keys.length === 0) {
    return NextResponse.json({ error: "No valid fields to update" }, { status: 400 });
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    const currentResult = await client.query(
      `SELECT * FROM trailing_ratio_config
        WHERE config_set = $1 AND epic = $2 AND is_scalp = $3
        LIMIT 1
        FOR UPDATE`,
      [configSet, params.epic, scalp]
    );
    const current = currentResult.rows[0];
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "Row not found" }, { status: 404 });
    }

    if (normalizeTimestamp(current.updated_at) !== updated_at) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        {
          error: "conflict",
          message: "Row was updated by another user",
          current_updated_at: current.updated_at,
          updated_by: current.updated_by,
          current_row: current,
        },
        { status: 409 }
      );
    }

    const setParts: string[] = [];
    const values: unknown[] = [];
    keys.forEach((k, idx) => {
      setParts.push(`${k} = $${idx + 1}`);
      values.push(updates[k]);
    });
    setParts.push(`updated_by = $${keys.length + 1}`);
    setParts.push(`change_reason = $${keys.length + 2}`);
    values.push(updated_by);
    values.push(change_reason);
    values.push(current.id);

    const updateResult = await client.query(
      `UPDATE trailing_ratio_config
         SET ${setParts.join(", ")}
       WHERE id = $${keys.length + 3}
       RETURNING *`,
      values
    );

    const previousValues: Record<string, unknown> = {};
    keys.forEach((k) => {
      previousValues[k] = current[k];
    });

    await client.query(
      `INSERT INTO trailing_ratio_audit
         (config_id, change_type, changed_by, change_reason, previous_values, new_values)
       VALUES ($1, 'UPDATE', $2, $3, $4, $5)`,
      [
        current.id,
        updated_by,
        change_reason,
        JSON.stringify(previousValues),
        JSON.stringify(updates),
      ]
    );

    await client.query("COMMIT");
    return NextResponse.json(updateResult.rows[0]);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update trailing ratio row", error);
    return NextResponse.json(
      { error: "Failed to update trailing ratio row" },
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
  const configSet = body?.config_set ?? "demo";
  const scalp = Boolean(body?.is_scalp);

  if (!updated_by || !change_reason) {
    return NextResponse.json(
      { error: "updated_by and change_reason are required" },
      { status: 400 }
    );
  }

  // Guardrail: DEFAULT rows are the resolution baseline — deleting them
  // would leave the service without a fallback. Block at the API layer.
  if (params.epic === "DEFAULT") {
    return NextResponse.json(
      { error: "Cannot delete DEFAULT ratio row" },
      { status: 400 }
    );
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const currentResult = await client.query(
      `SELECT * FROM trailing_ratio_config
        WHERE config_set = $1 AND epic = $2 AND is_scalp = $3
        LIMIT 1`,
      [configSet, params.epic, scalp]
    );
    const current = currentResult.rows[0];
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "Row not found" }, { status: 404 });
    }
    await client.query(`DELETE FROM trailing_ratio_config WHERE id = $1`, [current.id]);
    await client.query(
      `INSERT INTO trailing_ratio_audit
         (config_id, change_type, changed_by, change_reason, previous_values, new_values)
       VALUES ($1, 'DELETE', $2, $3, $4, NULL)`,
      [current.id, updated_by, change_reason, JSON.stringify(current)]
    );
    await client.query("COMMIT");
    return NextResponse.json({ success: true });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to delete trailing ratio row", error);
    return NextResponse.json(
      { error: "Failed to delete trailing ratio row" },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
