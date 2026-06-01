import { NextResponse } from "next/server";
import {
  DEFAULT_CONFIG_SET,
  OVERRIDE_TABLE,
  getOverrideColumns,
  normalizeTimestamp,
  strategyConfigPool,
} from "../../common";

export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: { epic: string } }) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;
    const result = await strategyConfigPool.query(
      `SELECT * FROM ${OVERRIDE_TABLE} WHERE config_set = $1 AND epic = $2 LIMIT 1`,
      [configSet, params.epic],
    );
    if (!result.rows[0]) {
      return NextResponse.json({ error: "Pair override not found" }, { status: 404 });
    }
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to load INSIDE_DAY pair override", error);
    return NextResponse.json({ error: "Failed to load INSIDE_DAY pair override" }, { status: 500 });
  }
}

export async function PUT(request: Request, { params }: { params: { epic: string } }) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_at, config_set } = body as {
    updates?: Record<string, unknown>;
    updated_at?: string;
    config_set?: string;
  };
  if (!updates || !updated_at) {
    return NextResponse.json({ error: "updates and updated_at are required" }, { status: 400 });
  }

  const configSet = config_set ?? DEFAULT_CONFIG_SET;
  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const currentResult = await client.query(
      `
        SELECT * FROM ${OVERRIDE_TABLE}
        WHERE config_set = $1 AND epic = $2
        LIMIT 1
        FOR UPDATE
      `,
      [configSet, params.epic],
    );
    const current = currentResult.rows[0];
    if (!current) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "Pair override not found" }, { status: 404 });
    }
    if (normalizeTimestamp(current.updated_at) !== updated_at) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: "conflict", current_override: current, message: "Pair override updated by another user" },
        { status: 409 },
      );
    }

    const allowed = new Set(
      (await getOverrideColumns(client)).filter(
        (col: string) => !["id", "config_set", "epic", "created_at", "updated_at"].includes(col),
      ),
    );
    const keys = Object.keys(updates).filter((key) => allowed.has(key));
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
        UPDATE ${OVERRIDE_TABLE}
        SET ${setClauses.join(", ")}, updated_at = NOW()
        WHERE id = $${keys.length + 1}
        RETURNING *
      `,
      values,
    );

    await client.query("COMMIT");
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update INSIDE_DAY pair override", error);
    return NextResponse.json({ error: "Failed to update INSIDE_DAY pair override" }, { status: 500 });
  } finally {
    client.release();
  }
}

export async function DELETE(request: Request, { params }: { params: { epic: string } }) {
  const body = await request.json().catch(() => null);
  const configSet = body?.config_set ?? DEFAULT_CONFIG_SET;

  try {
    await strategyConfigPool.query(
      `DELETE FROM ${OVERRIDE_TABLE} WHERE config_set = $1 AND epic = $2`,
      [configSet, params.epic],
    );
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete INSIDE_DAY pair override", error);
    return NextResponse.json({ error: "Failed to delete INSIDE_DAY pair override" }, { status: 500 });
  }
}
