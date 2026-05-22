import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `SELECT column_name FROM information_schema.columns WHERE table_name = 'donchian_turtle_pair_overrides'`,
    );

    const columns = result.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter((col: string) => !["id", "epic", "created_at", "updated_at"].includes(col));

    return NextResponse.json({ columns });
  } catch (error) {
    console.error("Failed to load DONCHIAN_TURTLE pair override columns", error);
    return NextResponse.json({ error: "Failed to load DONCHIAN_TURTLE pair override columns" }, { status: 500 });
  }
}
