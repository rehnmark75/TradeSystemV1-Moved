import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";
import { RATIO_FIELDS } from "../../../../lib/settings/trailingRatioFields";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const configSet = searchParams.get("config_set") ?? "demo";
  const isScalpParam = searchParams.get("is_scalp");
  const isScalp = isScalpParam === "true";

  try {
    const cols = [
      "id",
      "config_set",
      "epic",
      "is_scalp",
      "is_active",
      ...RATIO_FIELDS,
      "updated_by",
      "change_reason",
      "updated_at",
    ].join(", ");

    const whereClauses: string[] = ["config_set = $1", "is_active = TRUE"];
    const params: unknown[] = [configSet];
    if (isScalpParam !== null) {
      whereClauses.push(`is_scalp = $${params.length + 1}`);
      params.push(isScalp);
    }

    const result = await strategyConfigPool.query(
      `SELECT ${cols}
         FROM trailing_ratio_config
        WHERE ${whereClauses.join(" AND ")}
        ORDER BY epic = 'DEFAULT' DESC, is_scalp ASC, epic ASC`,
      params
    );
    return NextResponse.json({ rows: result.rows ?? [] });
  } catch (error) {
    console.error("Failed to load trailing ratios", error);
    return NextResponse.json(
      { error: "Failed to load trailing ratios" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { epic, is_scalp, updates, updated_by, change_reason, config_set } =
    body as {
      epic?: string;
      is_scalp?: boolean;
      updates?: Record<string, unknown>;
      updated_by?: string;
      change_reason?: string;
      config_set?: string;
    };

  const configSet = config_set ?? "demo";
  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }
  if (!updated_by || !change_reason) {
    return NextResponse.json(
      { error: "updated_by and change_reason are required" },
      { status: 400 }
    );
  }

  const allowed = new Set<string>(RATIO_FIELDS);
  const keys = Object.keys(updates ?? {}).filter((k) => allowed.has(k));
  if (keys.length === 0) {
    return NextResponse.json(
      { error: "No valid fields provided" },
      { status: 400 }
    );
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    const insertCols = [
      "config_set",
      "epic",
      "is_scalp",
      ...keys,
      "updated_by",
      "change_reason",
    ];
    const values: unknown[] = [configSet, epic, Boolean(is_scalp)];
    keys.forEach((k) => values.push((updates as Record<string, unknown>)[k]));
    values.push(updated_by, change_reason);

    const placeholders = insertCols.map((_, i) => `$${i + 1}`).join(", ");
    const setClause = keys
      .map((k) => `${k} = EXCLUDED.${k}`)
      .concat([
        "updated_by = EXCLUDED.updated_by",
        "change_reason = EXCLUDED.change_reason",
        "updated_at = NOW()",
      ])
      .join(", ");

    const result = await client.query(
      `INSERT INTO trailing_ratio_config (${insertCols.join(", ")})
       VALUES (${placeholders})
       ON CONFLICT (config_set, epic, is_scalp)
       DO UPDATE SET ${setClause}
       RETURNING *`,
      values
    );

    await client.query(
      `INSERT INTO trailing_ratio_audit
         (config_id, change_type, changed_by, change_reason, previous_values, new_values)
       VALUES ($1, 'UPSERT', $2, $3, $4, $5)`,
      [
        result.rows[0]?.id ?? null,
        updated_by,
        change_reason,
        null,
        JSON.stringify(updates),
      ]
    );

    await client.query("COMMIT");
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to upsert trailing ratio", error);
    return NextResponse.json(
      { error: "Failed to upsert trailing ratio" },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
