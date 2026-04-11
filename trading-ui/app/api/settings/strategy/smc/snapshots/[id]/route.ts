import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

/** GET /api/settings/strategy/smc/snapshots/[id] — load single snapshot */
export async function GET(
  _request: Request,
  { params }: { params: { id: string } }
) {
  const id = Number(params.id);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid snapshot ID" }, { status: 400 });
  }

  try {
    const result = await strategyConfigPool.query(
      `SELECT * FROM smc_backtest_snapshots WHERE id = $1 AND is_active = TRUE`,
      [id]
    );
    if (!result.rows[0]) {
      return NextResponse.json({ error: "Snapshot not found" }, { status: 404 });
    }
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to load snapshot", error);
    return NextResponse.json({ error: "Failed to load snapshot" }, { status: 500 });
  }
}

/** PATCH /api/settings/strategy/smc/snapshots/[id] — update name/description/tags */
export async function PATCH(
  request: Request,
  { params }: { params: { id: string } }
) {
  const id = Number(params.id);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid snapshot ID" }, { status: 400 });
  }

  const body = await request.json().catch(() => null);
  if (!body) {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { name, description, tags } = body as {
    name?: string;
    description?: string;
    tags?: string[];
  };

  try {
    const updates: string[] = [];
    const values: unknown[] = [];
    let paramIndex = 1;

    if (name !== undefined) {
      updates.push(`snapshot_name = $${paramIndex++}`);
      values.push(name.trim());
    }
    if (description !== undefined) {
      updates.push(`description = $${paramIndex++}`);
      values.push(description);
    }
    if (tags !== undefined) {
      updates.push(`tags = $${paramIndex++}`);
      values.push(tags);
    }

    if (updates.length === 0) {
      return NextResponse.json({ error: "No fields to update" }, { status: 400 });
    }

    values.push(id);
    const result = await strategyConfigPool.query(
      `UPDATE smc_backtest_snapshots SET ${updates.join(", ")} WHERE id = $${paramIndex} AND is_active = TRUE RETURNING *`,
      values
    );
    if (!result.rows[0]) {
      return NextResponse.json({ error: "Snapshot not found" }, { status: 404 });
    }
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to update snapshot", error);
    return NextResponse.json({ error: "Failed to update snapshot" }, { status: 500 });
  }
}

/** DELETE /api/settings/strategy/smc/snapshots/[id] — soft delete */
export async function DELETE(
  _request: Request,
  { params }: { params: { id: string } }
) {
  const id = Number(params.id);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid snapshot ID" }, { status: 400 });
  }

  try {
    const result = await strategyConfigPool.query(
      `UPDATE smc_backtest_snapshots SET is_active = FALSE WHERE id = $1 RETURNING id`,
      [id]
    );
    if (!result.rows[0]) {
      return NextResponse.json({ error: "Snapshot not found" }, { status: 404 });
    }
    return NextResponse.json({ deleted: true });
  } catch (error) {
    console.error("Failed to delete snapshot", error);
    return NextResponse.json({ error: "Failed to delete snapshot" }, { status: 500 });
  }
}
