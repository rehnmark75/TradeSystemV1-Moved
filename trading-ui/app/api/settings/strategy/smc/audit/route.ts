import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Number(searchParams.get("limit") ?? 50);
  const configSet = searchParams.get("config_set") ?? "demo";

  try {
    const result = await strategyConfigPool.query(
      `
        SELECT
          id,
          config_id,
          pair_override_id,
          change_type,
          changed_by,
          changed_at,
          change_reason,
          previous_values,
          new_values
        FROM smc_simple_config_audit
        WHERE config_id IN (
          SELECT id FROM smc_simple_global_config WHERE config_set = $1
        )
        ORDER BY changed_at DESC
        LIMIT $2
      `,
      [configSet, limit]
    );

    return NextResponse.json(result.rows ?? []);
  } catch (error) {
    console.error("Failed to load SMC audit history", error);
    return NextResponse.json(
      { error: "Failed to load SMC audit history" },
      { status: 500 }
    );
  }
}
