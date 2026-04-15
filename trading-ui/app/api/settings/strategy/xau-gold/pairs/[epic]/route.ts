import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

function normalizeTimestamp(value: unknown) {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

async function getOverrideColumns(client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'xau_gold_pair_overrides'
    `
  );
  return result.rows.map((row: { column_name: string }) => row.column_name);
}

export async function GET(request: Request, { params }: { params: { epic: string } }) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? "demo";
    const result = await strategyConfigPool.query(
      `
        SELECT *
        FROM xau_gold_pair_overrides
        WHERE config_set = $1 AND epic = $2
        LIMIT 1
      `,
      [configSet, params.epic]
    );

    if (!result.rows[0]) {
      return NextResponse.json({ error: "Pair override not found" }, { status: 404 });
    }
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to load XAU_GOLD pair override", error);
    return NextResponse.json({ error: "Failed to load XAU_GOLD pair override" }, { status: 500 });
  }
}

export async function PUT(request: Request, { params }: { params: { epic: string } }) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_by, change_reason, updated_at, config_set } = body as {
    updates?: Record<string, unknown>;
    updated_by?: string;
    change_reason?: string;
    updated_at?: string;
    config_set?: string;
  };

  if (!updates || !updated_by || !change_reason || !updated_at) {
    return NextResponse.json(
      { error: "updates, updated_by, change_reason, and updated_at are required" },
      { status: 400 }
    );
  }

  const configSet = config_set ?? "demo";
  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const currentResult = await client.query(
      `
        SELECT *
        FROM xau_gold_pair_overrides
        WHERE config_set = $1 AND epic = $2
        LIMIT 1
        FOR UPDATE
      `,
      [configSet, params.epic]
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
        { status: 409 }
      );
    }

    const allowed = new Set(
      (await getOverrideColumns(client)).filter(
        (column: string) =>
          !["id", "config_set", "created_at", "updated_at", "updated_by", "change_reason", "epic"].includes(column)
      )
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
    values.push(updated_by, change_reason, current.id);

    const result = await client.query(
      `
        UPDATE xau_gold_pair_overrides
        SET ${setClauses.join(", ")},
            updated_by = $${keys.length + 1},
            change_reason = $${keys.length + 2},
            updated_at = NOW()
        WHERE id = $${keys.length + 3}
        RETURNING *
      `,
      values
    );

    await client.query("COMMIT");
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update XAU_GOLD pair override", error);
    return NextResponse.json({ error: "Failed to update XAU_GOLD pair override" }, { status: 500 });
  } finally {
    client.release();
  }
}

export async function DELETE(request: Request, { params }: { params: { epic: string } }) {
  const body = await request.json().catch(() => null);
  const configSet = body?.config_set ?? "demo";
  if (!body?.updated_by || !body?.change_reason) {
    return NextResponse.json(
      { error: "updated_by and change_reason are required" },
      { status: 400 }
    );
  }

  try {
    await strategyConfigPool.query(
      `
        DELETE FROM xau_gold_pair_overrides
        WHERE config_set = $1 AND epic = $2
      `,
      [configSet, params.epic]
    );
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete XAU_GOLD pair override", error);
    return NextResponse.json({ error: "Failed to delete XAU_GOLD pair override" }, { status: 500 });
  }
}
