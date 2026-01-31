import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `
        SELECT
          column_name,
          column_default
        FROM information_schema.columns
        WHERE table_name = 'scanner_global_config'
        ORDER BY ordinal_position
      `
    );

    const defaults: Record<string, string | null> = {};
    result.rows.forEach((row) => {
      defaults[row.column_name] = row.column_default ?? null;
    });

    return NextResponse.json(defaults);
  } catch (error) {
    console.error("Failed to load scanner defaults", error);
    return NextResponse.json(
      { error: "Failed to load scanner defaults" },
      { status: 500 }
    );
  }
}
