import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Number(searchParams.get("limit") ?? 50);
  const category = searchParams.get("category");

  try {
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
        ${category ? "WHERE category = $1" : ""}
        ORDER BY changed_at DESC
        LIMIT ${category ? "$2" : "$1"}
      `,
      category ? [category, limit] : [limit]
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
