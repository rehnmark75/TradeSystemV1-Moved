import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const result = await strategyConfigPool.query(
      `SELECT column_name FROM information_schema.columns WHERE table_name = 'fa_or_atr_trail_pair_overrides'`,
    );
    const columns = result.rows
      .map((row: { column_name: string }) => row.column_name)
      .filter((col: string) => !["id", "config_set", "epic", "created_at", "updated_at"].includes(col));

    return NextResponse.json({ columns });
  } catch (error) {
    console.error("Failed to load FA_OR_ATR_TRAIL pair override columns", error);
    return NextResponse.json({ error: "Failed to load FA_OR_ATR_TRAIL pair override columns" }, { status: 500 });
  }
}
