import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? "demo";

    const result = await strategyConfigPool.query(
      `
        SELECT *
        FROM xau_gold_pair_overrides
        WHERE config_set = $1
        ORDER BY epic ASC
      `,
      [configSet]
    );

    return NextResponse.json({ overrides: result.rows ?? [] });
  } catch (error) {
    console.error("Failed to load XAU_GOLD pair overrides", error);
    return NextResponse.json(
      { error: "Failed to load XAU_GOLD pair overrides" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { epic, updates, updated_by, change_reason, config_set } = body as {
    epic?: string;
    updates?: Record<string, unknown>;
    updated_by?: string;
    change_reason?: string;
    config_set?: string;
  };

  if (!epic || !updates || !updated_by || !change_reason) {
    return NextResponse.json(
      { error: "epic, updates, updated_by, and change_reason are required" },
      { status: 400 }
    );
  }

  const configSet = config_set ?? "demo";
  const columnsResult = await strategyConfigPool.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'xau_gold_pair_overrides'
    `
  );
  const allowed = new Set(
    columnsResult.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter((column: string) =>
        !["id", "config_set", "created_at", "updated_at", "updated_by", "change_reason", "epic"].includes(column)
      )
  );

  const keys = Object.keys(updates).filter((key) => allowed.has(key));
  const columns = ["config_set", "epic", ...keys, "updated_by", "change_reason"];
  const values: unknown[] = [configSet, epic];
  keys.forEach((key) => values.push(updates[key]));
  values.push(updated_by, change_reason);
  const placeholders = columns.map((_, index) => `$${index + 1}`);

  const setClause = [
    ...keys.map((key) => `${key} = EXCLUDED.${key}`),
    "updated_by = EXCLUDED.updated_by",
    "change_reason = EXCLUDED.change_reason",
    "updated_at = NOW()",
  ].join(", ");

  try {
    const result = await strategyConfigPool.query(
      `
        INSERT INTO xau_gold_pair_overrides (${columns.join(", ")})
        VALUES (${placeholders.join(", ")})
        ON CONFLICT (config_set, epic)
        DO UPDATE SET
          ${setClause}
        RETURNING *
      `,
      values
    );

    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to upsert XAU_GOLD pair override", error);
    return NextResponse.json(
      { error: "Failed to upsert XAU_GOLD pair override" },
      { status: 500 }
    );
  }
}
