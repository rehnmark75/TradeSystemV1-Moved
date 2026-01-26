import { NextResponse } from "next/server";
import { forexPool } from "../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_LIMIT = 100;

function parseLimit(value: string | null) {
  if (!value) return DEFAULT_LIMIT;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_LIMIT;
  if (parsed <= 0) return DEFAULT_LIMIT;
  return parsed;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = parseLimit(searchParams.get("limit"));

  try {
    const result = await forexPool.query(
      `
      SELECT
        id,
        symbol,
        direction,
        timestamp,
        status,
        profit_loss,
        pnl_currency
      FROM trade_log
      WHERE status IN ('closed', 'tracking')
      ORDER BY timestamp DESC
      LIMIT $1
      `,
      [limit]
    );

    const trades = (result.rows ?? []).map((row) => ({
      ...row,
      profit_loss: row.profit_loss == null ? null : Number(row.profit_loss),
      pnl_display:
        row.profit_loss == null
          ? "Open"
          : `${row.profit_loss >= 0 ? "+" : ""}${Number(row.profit_loss).toFixed(2)} ${
              row.pnl_currency ?? ""
            }`.trim()
    }));

    return NextResponse.json({ trades });
  } catch (error) {
    console.error("Failed to load trade list", error);
    return NextResponse.json(
      { error: "Failed to load trade list" },
      { status: 500 }
    );
  }
}
