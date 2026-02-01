import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

async function getOverrideColumns(client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'smc_simple_pair_overrides'
    `
  );
  return result.rows.map((row: { column_name: string }) => row.column_name);
}

export async function GET() {
  try {
    const columns = await getOverrideColumns();
    const allowed = columns.filter(
      (column: string) =>
        ![
          "id",
          "config_id",
          "created_at",
          "updated_at",
          "updated_by",
          "change_reason",
          "epic"
        ].includes(column)
    );
    return NextResponse.json({ columns: allowed });
  } catch (error) {
    console.error("Failed to load pair override columns", error);
    return NextResponse.json(
      { error: "Failed to load pair override columns" },
      { status: 500 }
    );
  }
}
