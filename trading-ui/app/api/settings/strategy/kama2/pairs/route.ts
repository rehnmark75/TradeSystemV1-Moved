import { NextResponse } from "next/server";
import { DEFAULT_CONFIG_SET, OVERRIDE_TABLE, strategyConfigPool } from "../common";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;
    const result = await strategyConfigPool.query(
      `SELECT * FROM ${OVERRIDE_TABLE} WHERE config_set = $1 ORDER BY epic ASC`,
      [configSet],
    );
    return NextResponse.json({ overrides: result.rows ?? [] });
  } catch (error) {
    console.error("Failed to load KAMA_V2 pair overrides", error);
    return NextResponse.json({ error: "Failed to load KAMA_V2 pair overrides" }, { status: 500 });
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { epic, updates, overrides, config_set } = body as {
    epic?: string;
    updates?: Record<string, unknown>;
    overrides?: Record<string, unknown>;
    config_set?: string;
  };
  if (!epic) return NextResponse.json({ error: "epic is required" }, { status: 400 });

  const configSet = config_set ?? DEFAULT_CONFIG_SET;
  const updatePayload: Record<string, unknown> = { ...(updates ?? {}) };
  if (updatePayload.parameter_overrides === undefined && overrides) {
    updatePayload.parameter_overrides = overrides;
  }

  const columnsResult = await strategyConfigPool.query(
    `SELECT column_name FROM information_schema.columns WHERE table_name = '${OVERRIDE_TABLE}'`,
  );
  const allowed = new Set(
    columnsResult.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter((col: string) => !["id", "config_set", "epic", "created_at", "updated_at"].includes(col)),
  );
  const keys = Object.keys(updatePayload).filter((key) => allowed.has(key));
  const columns = ["config_set", "epic", ...keys];
  const values: unknown[] = [configSet, epic];
  keys.forEach((key) => values.push(updatePayload[key]));

  const placeholders = columns.map((_, index) => `$${index + 1}`);
  const setClause = [...keys.map((key) => `${key} = EXCLUDED.${key}`), "updated_at = NOW()"].join(", ");

  try {
    const result = await strategyConfigPool.query(
      `
        INSERT INTO ${OVERRIDE_TABLE} (${columns.join(", ")})
        VALUES (${placeholders.join(", ")})
        ON CONFLICT (config_set, epic)
        DO UPDATE SET ${setClause}
        RETURNING *
      `,
      values,
    );
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to upsert KAMA_V2 pair override", error);
    return NextResponse.json({ error: "Failed to upsert KAMA_V2 pair override" }, { status: 500 });
  }
}
