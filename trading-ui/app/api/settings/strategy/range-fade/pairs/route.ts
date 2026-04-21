import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const DEFAULT_PROFILE = "5m";
const DEFAULT_CONFIG_SET = "demo";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const profileParam = searchParams.get("profile_name");
    const profile = profileParam && /^(5m|15m)$/.test(profileParam) ? profileParam : DEFAULT_PROFILE;
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;

    const result = await strategyConfigPool.query(
      `
        SELECT *
        FROM eurusd_range_fade_pair_overrides
        WHERE profile_name = $1 AND config_set = $2
        ORDER BY epic ASC
      `,
      [profile, configSet],
    );

    return NextResponse.json({ overrides: result.rows ?? [] });
  } catch (error) {
    console.error("Failed to load RANGE_FADE pair overrides", error);
    return NextResponse.json(
      { error: "Failed to load RANGE_FADE pair overrides" },
      { status: 500 },
    );
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { epic, updates, overrides, config_set, profile_name } = body as {
    epic?: string;
    updates?: Record<string, unknown>;
    overrides?: Record<string, unknown>;
    config_set?: string;
    profile_name?: string;
  };

  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }

  const profile = profile_name && /^(5m|15m)$/.test(profile_name) ? profile_name : DEFAULT_PROFILE;
  const configSet = config_set ?? DEFAULT_CONFIG_SET;

  const updatePayload: Record<string, unknown> = { ...(updates ?? {}) };
  if (updatePayload.parameter_overrides === undefined && overrides) {
    updatePayload.parameter_overrides = overrides;
  }

  const columnsResult = await strategyConfigPool.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'eurusd_range_fade_pair_overrides'
    `,
  );
  const allowed = new Set(
    columnsResult.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter(
        (column: string) =>
          !["id", "profile_name", "config_set", "epic", "created_at", "updated_at"].includes(column),
      ),
  );

  const keys = Object.keys(updatePayload).filter((k) => allowed.has(k));
  const columns = ["profile_name", "config_set", "epic", ...keys];
  const values: unknown[] = [profile, configSet, epic];
  keys.forEach((k) => values.push(updatePayload[k]));

  const placeholders = columns.map((_, i) => `$${i + 1}`);
  const setClause = [
    ...keys.map((k) => `${k} = EXCLUDED.${k}`),
    "updated_at = NOW()",
  ].join(", ");

  try {
    const result = await strategyConfigPool.query(
      `
        INSERT INTO eurusd_range_fade_pair_overrides (${columns.join(", ")})
        VALUES (${placeholders.join(", ")})
        ON CONFLICT (epic, profile_name, config_set)
        DO UPDATE SET
          ${setClause}
        RETURNING *
      `,
      values,
    );

    return NextResponse.json(result.rows[0]);
  } catch (error) {
    console.error("Failed to upsert RANGE_FADE pair override", error);
    return NextResponse.json(
      { error: "Failed to upsert RANGE_FADE pair override" },
      { status: 500 },
    );
  }
}
