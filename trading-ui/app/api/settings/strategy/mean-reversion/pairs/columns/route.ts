import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'mean_reversion_pair_overrides'
      `,
    );

    const columns = result.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter(
        (column: string) =>
          !["id", "config_set", "created_at", "updated_at", "updated_by", "change_reason", "epic"].includes(
            column,
          ),
      );

    return NextResponse.json({ columns });
  } catch (error) {
    console.error("Failed to load MEAN_REVERSION pair override columns", error);
    return NextResponse.json(
      { error: "Failed to load MEAN_REVERSION pair override columns" },
      { status: 500 },
    );
  }
}
