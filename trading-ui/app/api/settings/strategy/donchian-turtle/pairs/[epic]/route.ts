import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

function normalizeTimestamp(value: unknown): string {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

async function getOverrideColumns(client?: { query: Function }): Promise<string[]> {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `SELECT column_name FROM information_schema.columns WHERE table_name = 'donchian_turtle_pair_overrides'`,
  );
  return result.rows.map((row: { column_name: string }) => row.column_name);
}

export async function GET(_request: Request, { params }: { params: { epic: string } }) {
  try {
    const result = await strategyConfigPool.query(
      `SELECT * FROM donchian_turtle_pair_overrides WHERE epic = $1 LIMIT 1`,
      [params.epic],
    );
    if (!result.rows[0]) {
      return NextResponse.json({ error: "Pair override not found" }, { status: 404 });
    }
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to load DONCHIAN_TURTLE pair override", error);
    return NextResponse.json({ error: "Failed to load DONCHIAN_TURTLE pair override" }, { status: 500 });
  }
}

export async function PUT(request: Request, { params }: { params: { epic: string } }) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_at } = body as { updates?: Record<string, unknown>; updated_at?: string };

  if (!updates || !updated_at) {
    return NextResponse.json({ error: "updates and updated_at are required" }, { status: 400 });
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    const currentResult = await client.query(
      `SELECT * FROM donchian_turtle_pair_overrides WHERE epic = $1 LIMIT 1 FOR UPDATE`,
      [params.epic],
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
        (col: string) => !["id", "epic", "created_at", "updated_at"].includes(col),
      ),
    );

    const keys = Object.keys(updates).filter((k) => allowed.has(k));
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
      `UPDATE donchian_turtle_pair_overrides SET ${setClauses.join(", ")}, updated_at = NOW() WHERE id = $${keys.length + 1} RETURNING *`,
      values,
    );

    await client.query("COMMIT");
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update DONCHIAN_TURTLE pair override", error);
    return NextResponse.json({ error: "Failed to update DONCHIAN_TURTLE pair override" }, { status: 500 });
  } finally {
    client.release();
  }
}

export async function DELETE(_request: Request, { params }: { params: { epic: string } }) {
  try {
    await strategyConfigPool.query(
      `DELETE FROM donchian_turtle_pair_overrides WHERE epic = $1`,
      [params.epic],
    );
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete DONCHIAN_TURTLE pair override", error);
    return NextResponse.json({ error: "Failed to delete DONCHIAN_TURTLE pair override" }, { status: 500 });
  }
}
