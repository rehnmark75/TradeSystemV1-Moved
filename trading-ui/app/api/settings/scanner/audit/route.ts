import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Number(searchParams.get("limit") ?? 50);
  const category = searchParams.get("category");
  const configSet = searchParams.get("config_set") ?? "demo";

  try {
    const whereClauses: string[] = [
      `config_id IN (SELECT id FROM scanner_global_config WHERE config_set = $1)`
    ];
    const params: unknown[] = [configSet];
    if (category) {
      whereClauses.push(`category = $${params.length + 1}`);
      params.push(category);
    }
    params.push(limit);

    const result = await strategyConfigPool.query(
      `
        SELECT
          id,
          config_id,
          change_type,
          changed_by,
          changed_at,
          change_reason,
          previous_values,
          new_values,
          category
        FROM scanner_config_audit
        WHERE ${whereClauses.join(" AND ")}
        ORDER BY changed_at DESC
        LIMIT $${params.length}
      `,
      params
    );

    return NextResponse.json(result.rows ?? []);
  } catch (error) {
    console.error("Failed to load scanner audit history", error);
    return NextResponse.json(
      { error: "Failed to load scanner audit history" },
      { status: 500 }
    );
  }
}
