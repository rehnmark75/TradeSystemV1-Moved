import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Number(searchParams.get("limit") ?? 50);
  const configSet = searchParams.get("config_set") ?? "demo";

  try {
    // Audit rows persist after a config row is deactivated/deleted. We use a
    // subquery so we still surface history for retired rows in that env.
    // When the config_id FK is SET NULL (cascade), we fall back to matching
    // through new_values.config_set if needed — for now, just filter via the
    // live FK relationship.
    const result = await strategyConfigPool.query(
      `
        SELECT a.id,
               a.config_id,
               a.change_type,
               a.changed_by,
               a.changed_at,
               a.change_reason,
               a.previous_values,
               a.new_values
        FROM trailing_config_audit a
        WHERE a.config_id IN (
          SELECT id FROM trailing_pair_config WHERE config_set = $1
        )
        ORDER BY a.changed_at DESC
        LIMIT $2
      `,
      [configSet, limit]
    );

    return NextResponse.json(result.rows ?? []);
  } catch (error) {
    console.error("Failed to load trailing audit", error);
    return NextResponse.json(
      { error: "Failed to load trailing audit" },
      { status: 500 }
    );
  }
}
